import torch
import time

SYSTEM_PROMPT = (
    "A conversation between a User and an Assistant. "
    "The User asks a question; the Assistant solves it by first reasoning, then providing the final answer. "
    "The Assistant encloses its final answer in $\\boxed{}$."
)

def exponential_running_average(embedding_batch, **kwargs):
    """
    Compute the exponential running average of the embeddings in the batch.
    embedding_batch: Tensor of shape (B, S, H)
    Returns: Tensor of shape (B, S, H)
    """
    alpha = kwargs.get('alpha', 0.97)
    B, S, H = embedding_batch.shape
    new_embedding_batch = torch.zeros_like(embedding_batch)
    ema_batch = torch.zeros(B, H, device=embedding_batch.device)
    alphas_power = torch.pow(alpha, torch.arange(1, S + 1, device=embedding_batch.device))
    
    for k in range(S):
        current_hidden_state = embedding_batch[:, k, :]
        if k == 0:
            ema_batch = current_hidden_state
        else:
            ema_batch = alpha * ema_batch + (1 - alpha) * current_hidden_state
        new_embedding_batch[:, k, :] = ema_batch / (1 - alphas_power[k])
    
    return new_embedding_batch

def embedding_selection_uniform(batch_shape, seq_lengths, start_indices, min_interval, **kwargs):
    """
    Uniformly select embeddings with guaranteed minimum interval between indices.
    
    Args:
        batch_shape: Tuple (B, S, H) representing batch dimensions
        seq_lengths: Tensor of shape (B,) containing actual sequence length for each item
        start_indices: List of starting indices for each item in the batch
        min_interval: Minimum distance required between consecutive selected indices
    
    Returns:
        List of tensors containing selected indices for each item in the batch
    """
    B, S, H = batch_shape
    device = seq_lengths.device
    
    final_indices_list = []
    
    for batch_idx in range(B):
        start_idx = start_indices[batch_idx]
        end_idx = seq_lengths[batch_idx].item() - 1  # -1 to get last valid index
        
        # Calculate how many steps we can fit
        available_length = end_idx - start_idx
        
        if available_length < min_interval:
            # If not enough space, just use start and end
            indices = torch.tensor([start_idx, end_idx], device=device)
        else:
            # Calculate number of intervals we can fit
            num_intervals = available_length // min_interval
            
            # Generate uniformly spaced indices
            indices_list = [start_idx]
            for i in range(1, num_intervals + 1):
                next_idx = start_idx + i * min_interval
                if next_idx < end_idx:
                    indices_list.append(next_idx)
            
            # Always add the end index
            if indices_list[-1] != end_idx:
                indices_list.append(end_idx)
            
            indices = torch.tensor(indices_list, device=device)
        
        final_indices_list.append(indices)     
    
    return final_indices_list

def steps_selection(
                    hidden_states,
                    start_token_index,
                    attention_mask,
                    prompt_mask=None,
                    min_interval=50, 
                    alpha=0.7, 
                    mean_window = 5, 
                    selection_method=embedding_selection_uniform, 
                    average_method=exponential_running_average,
                    ):
    """
    Get the embedding of the responses using the provided model
    And separate the response into steps based on embeddings.
    """
    # Convert the full contexts into embeddings in batches
    with torch.inference_mode():
        # Compute the desired embedding by exponential running average from the first token to the last token for the whole batch
        new_embedding_batch = average_method(hidden_states, alpha=alpha, mean_window=mean_window)
        start_indices = torch.tensor([0]*new_embedding_batch.shape[0]) if prompt_mask is None else (1 - prompt_mask).sum(dim=1)
        start_indices = start_indices.to(attention_mask.device)
        seq_lengths = attention_mask.sum(dim=1) + start_indices
        # Sample indices to represent states based on the selection method
        response_indices_list = selection_method(
            batch_shape=new_embedding_batch.shape,
            seq_lengths=seq_lengths,
            start_indices=start_indices,
            min_interval=min_interval,
        )

        # Find the last index that is less than start_token_index for each item in the batch
        for i in range(len(response_indices_list)):
            indices = response_indices_list[i]
            curr_start_token_index = start_token_index[i]
            filtered_indices = indices[indices > curr_start_token_index - min_interval]
            filtered_indices[0] = curr_start_token_index
            response_indices_list[i] = filtered_indices - start_indices[i]
        # Downsample the representation embeddings based on the mean_window because the embeddings are smoothed
        representation_embedding_batch = [embeddings[start_indices[i]:seq_lengths[i]:mean_window].cpu() for i, embeddings in enumerate(new_embedding_batch)]
        representation_indices_list = [response_indices // mean_window for response_indices in response_indices_list]

        response_indices_list = [response_indices.tolist() for response_indices in response_indices_list]
        representation_indices_list = [representation_indices.tolist() for representation_indices in representation_indices_list]
        
    # Return the selected embeddings, state indices for responses, and state indices for representation embeddings
    return representation_embedding_batch, response_indices_list, representation_indices_list

def steps_embedding(model, 
                    tokenizer, 
                    system_prompt, 
                    problems_responses, 
                    batch_size, 
                    layer=-1, 
                    min_interval=50, 
                    alpha=0.97, 
                    mean_window = 100, 
                    selection_method=embedding_selection_uniform, 
                    average_method=exponential_running_average,
                    response_pattern="<|im_start|>assistant",
                    enable_thinking=False
                    ):
    """
    Get the embedding of the responses using the provided model
    And separate the response into steps based on embeddings.
    """


    def locate_start_token_indices(batch_contexts, offsets_list, pattern):
        """
        Locate the start token indices of a specific pattern in each context of the batch.
        batch_contexts: List of strings, each representing a full context.
        offsets_list: List of lists of tuples, each containing (start, end) character indices for tokens.
        pattern: The string pattern to locate.
        Returns: List of token indices where the pattern starts in each context.
        """
        start_token_indices = []
        for j in range(len(batch_contexts)):
            pattern_start_char_idx = batch_contexts[j].index(pattern) + len(pattern)
            for token_idx, (start, end) in enumerate(offsets_list[j]):
                if end >= pattern_start_char_idx:
                    start_token_indices.append(token_idx)
                    break
        return start_token_indices


    # Combine the system_prompt and problem_responses into full contexts
    full_contexts_list = []
    for problem in problems_responses:
        for response in problems_responses[problem]:
            if system_prompt:
                full_context = [{"role": "system", "content": system_prompt},
                                {"role": "user", "content": problem},]
            else:
                full_context = [{"role": "user", "content": problem},]
            full_context_input = tokenizer.apply_chat_template(full_context, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking) + response
            full_contexts_list.append(full_context_input)
    
    # Convert the full contexts into embeddings in batches
    with torch.inference_mode():
        all_embeddings_list = []
        all_response_indices_list = []
        all_representation_indices_list = []
        # Record the start time
        start_time = time.time()
        for i in range(0, len(full_contexts_list), batch_size):
            # Prepare the batch inputs
            batch_contexts = full_contexts_list[i:i+batch_size]
            model_inputs = tokenizer(batch_contexts, return_tensors="pt", padding=True, return_offsets_mapping=True).to(model.device)
            offsets_mapping = model_inputs.pop("offset_mapping")
            # get the hidden states for the current batch
            hidden_states_batch = model(**model_inputs, output_hidden_states=True)["hidden_states"][layer]

            # Find response start token index for each item in the batch
            batch_start_token_indices = locate_start_token_indices(batch_contexts, offsets_mapping, response_pattern)

            representation_embeddings_list, response_indices_list, representation_indices_list = steps_selection(
                hidden_states = hidden_states_batch,
                start_token_index = batch_start_token_indices,
                attention_mask = model_inputs['attention_mask'],
                prompt_mask = None,
                min_interval=min_interval,
                alpha=alpha,
                mean_window = mean_window,
                selection_method=selection_method,
                average_method=average_method,
            )
            all_embeddings_list.extend(representation_embeddings_list)
            all_response_indices_list.extend(response_indices_list)
            all_representation_indices_list.extend(representation_indices_list)

            if (i+1) % 1000 == 0:
                elapsed_time = time.time() - start_time
                print(f"Processed {i} / {len(full_contexts_list)} contexts for embeddings. Elapsed time: {elapsed_time:.2f} seconds.")

    # Group embeddings by problems
    problem_embeddings = {}
    problem_response_indices = {}
    problem_representation_indices = {}
    i = 0
    for problem in problems_responses:
        num_responses = len(problems_responses[problem])
        problem_embeddings_list = all_embeddings_list[i:i+num_responses]
        problem_response_indices_list = all_response_indices_list[i:i+num_responses]
        problem_representation_indices_list = all_representation_indices_list[i:i+num_responses]
            
        problem_embeddings[problem] = problem_embeddings_list
        problem_response_indices[problem] = problem_response_indices_list
        problem_representation_indices[problem] = problem_representation_indices_list
        i += num_responses
    
    return problem_embeddings, problem_response_indices, problem_representation_indices

def to_embeddings_indices_sequence(problem_embeddings, problem_representation_indices, problems_rewards):
    """
    Convert the problem embeddings into a sequence of embeddings and a sequence of nodes.
    Each node contains the start_idx, end_idx, and reward.
    """
    problem_embeddings_sequence = {}
    problem_nodes_sequence = {}
    problem_representations_sequence = {}
    for problem in problem_embeddings:
        embeddings_list = problem_embeddings[problem]
        rewards_list = problems_rewards[problem]
        representation_indices_list = problem_representation_indices[problem]

        # Each problem has its own large embeddings sequence
        # NOTE: .contiguous() is critical for multi-node training. When tensors are gathered
        embeddings_sequence = torch.cat(embeddings_list, dim=0).contiguous()
        problem_embeddings_sequence[problem] = embeddings_sequence

        # We traverse the indices to create the nodes sequence
        nodes_sequence = []
        representations_sequence = []
        start_idx = 0
        for idx, representation_indices in enumerate(representation_indices_list):
            representations_sequence.append((start_idx, start_idx + embeddings_list[idx].shape[0]))
            for representation_index in representation_indices:
                end_idx = representation_index + 1 + start_idx
                nodes_sequence.append(((start_idx, end_idx), rewards_list[idx]))

            start_idx += embeddings_list[idx].shape[0]

        problem_nodes_sequence[problem] = nodes_sequence
        problem_representations_sequence[problem] = representations_sequence

    return problem_embeddings_sequence, problem_nodes_sequence, problem_representations_sequence

def group_existing_nodes(existing_nodes):
    groups = {}
    for idx, (node, _reward) in enumerate(existing_nodes):
        start, end = node
        if end <= start:
            continue
        groups.setdefault(start, []).append((end, idx))
    for start in groups:
        groups[start].sort(key=lambda x: x[0])
    return groups

def compute_distances_for_nodes_fast(online_embeddings, 
                                     offline_embeddings, 
                                     online_nodes, 
                                     existing_nodes, 
                                     online_representations,):
    device = online_embeddings.device
    online_start = online_nodes[0][0]
    online_ends = torch.tensor([n[1] for n in online_nodes], device=device, dtype=torch.long)
    online_lens = (online_ends - online_start).to(torch.float32)
    online_end_max = int(online_ends.max().item())

    num_online = len(online_nodes)
    num_existing = len(existing_nodes)
    distances_nodes = torch.full((num_online, num_existing), float('inf'), device=device, dtype=torch.float32)

    row_idx = (online_ends - online_start - 1).clamp_min(0)
    row_idx = row_idx.to(torch.long)

    groups = group_existing_nodes(existing_nodes)
    for start, group_list in groups.items():
        ends = torch.tensor([g[0] for g in group_list], device=device, dtype=torch.long)
        col_idx = (ends - start - 1).clamp_min(0)
        col_idx = col_idx.to(torch.long)

        existing_lens = (ends - start).to(torch.float32)
        valid_mask = online_lens[:, None] >= existing_lens[None, :]

        max_end = int(ends.max().item())

        online_block = online_embeddings[online_start:online_end_max].float()
        offline_block = offline_embeddings[start:max_end].float()
        if online_block.numel() == 0 or offline_block.numel() == 0:
            continue

        sub = torch.cdist(online_block, offline_block, p=2)
        if sub.numel() == 0:
            continue

        cummin_cols = torch.cummin(sub, dim=1).values
        ps_rows = torch.cumsum(cummin_cols, dim=0)
        cummin_rows = torch.cummin(sub, dim=0).values
        ps_cols = torch.cumsum(cummin_rows, dim=1)

        sum_over_rows = ps_rows[row_idx[:, None], col_idx[None, :]]
        sum_over_cols = ps_cols[row_idx[:, None], col_idx[None, :]]

        online_len = online_lens[:, None]
        existing_len = existing_lens[None, :]
        dist_group = torch.where(
            online_len >= existing_len,
            sum_over_rows / online_len,
            sum_over_cols / existing_len,
        )
        if online_representations and start == online_representations[0]:
            dist_group = torch.where(valid_mask, dist_group, torch.full_like(dist_group, float('inf')))
        elif online_representations == None:
            dist_group = torch.where(valid_mask, dist_group, torch.full_like(dist_group, float('inf')))
        
        group_indices = [g[1] for g in group_list]
        distances_nodes[:, group_indices] = dist_group

    return distances_nodes

def calculate_weighted_distance_for_nodes(
                                          offline_embeddings, 
                                          existing_nodes, 
                                          online_embeddings, 
                                          online_nodes, 
                                          online_representations,
                                          max_k,
                                          min_k,
                                          t=1,
                                          min_distance=5.0,
                                          ):
    """
    Memory-efficient version of calculate_weighted_distance_for_nodes.
    
    Computes the full distance matrix in chunks on GPU, offloading to CPU as we go.
    This preserves the original optimization of computing distances once (no redundant
    computation for overlapping online nodes like [(0,50), (0,100), ...]).
    
    Args:
        offline_embeddings: Tensor of shape (N, H) - embeddings from offline/existing data
        existing_nodes: List of ((start_idx, end_idx), reward) tuples for existing nodes
        online_embeddings: Tensor of shape (M, H) - embeddings from online/current data
        online_nodes: List of (start_idx, end_idx) tuples for online nodes
        t: Temperature parameter for distance weighting (higher = sharper weights)
        min_distance: Minimum distance threshold to avoid division issues
    
    Returns:
        List of estimated state values for each online node
        
    Performance:
        - Uses GPU for fast distance computation (the expensive part)
        - Stores result on CPU (cheaper memory)
    """
    if len(online_nodes) == 0:
        return []

    # Limit per-process CPU threading to avoid oversubscription on multi-node runs
    prev_threads = torch.get_num_threads()
    if prev_threads > 4:
        torch.set_num_threads(4)

    online_embeddings = online_embeddings.to("cuda").float()
    offline_embeddings = offline_embeddings.to("cuda").float()

    # Step 2: Use the pre-computed matrix
    # k-scheduling is a trick to improve the estimation performance
    max_length = len(online_nodes)
    difference_k = (max_k - min_k) // max_length
    used_k = [max_k - difference_k * i for i in range(len(online_nodes))]

    per_nodes_estimated_values = []
    
    # Pre-process rewards into a tensor on the same device as embeddings
    rewards_list = [n[1] for n in existing_nodes]
    if len(rewards_list) > 0 and isinstance(rewards_list[0], torch.Tensor):
        all_rewards = torch.stack(rewards_list)
    else:
        all_rewards = torch.tensor(rewards_list)
    all_rewards = all_rewards.to(online_embeddings.device)

    distances_nodes = compute_distances_for_nodes_fast(
        online_embeddings=online_embeddings,
        offline_embeddings=offline_embeddings,
        online_nodes=online_nodes,
        existing_nodes=existing_nodes,
        online_representations=online_representations,
    )

    for node_idx, (online_node, k) in enumerate(zip(online_nodes, used_k)):
        distances_tensor = distances_nodes[node_idx]
        # Handle NaNs in distances (e.g. from NaN embeddings)
        distances_tensor = torch.nan_to_num(distances_tensor, nan=float('inf'))
        distances_tensor = torch.clamp_min(distances_tensor, min_distance)
        
        # Use the top k closest nodes to estimate the state value
        curr_k = min(k, len(existing_nodes))
        top_vals, top_indices = torch.topk(distances_tensor, curr_k, largest=False, sorted=True)
        
        top_rewards = all_rewards[top_indices]
        # Handle NaNs in rewards
        top_rewards = torch.nan_to_num(top_rewards, nan=0.0)

        # debug: print the top_vals, top_indices, and corresponding rewards
        #print(f"Online node {node_idx} (indices {online_node}):")
        #print(f"  Top indices: {top_indices.cpu().numpy()}")
        #print(f"  Top distances: {top_vals.cpu().numpy()}")
        #print(f"  Top rewards: {top_rewards.cpu().numpy()}")
        
        weights = 1.0 / (top_vals ** t)
        total_weight = torch.sum(weights)
        
        if total_weight < 1e-8:
             estimated_value = torch.tensor(0.0, device=online_embeddings.device)
        else:
             estimated_value = torch.sum(top_rewards * weights) / total_weight
             
        per_nodes_estimated_values.append(estimated_value.item())

    # Clean up
    del offline_embeddings
    del online_embeddings
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Restore threading setting
    if torch.get_num_threads() != prev_threads:
        torch.set_num_threads(prev_threads)

    return per_nodes_estimated_values

def calculate_GAE_advantages(per_nodes_estimated_value, indices, gae_lambda, max_sequence_length=None):
    """
    Calculate the GAE advantages for a list of nodes based on their estimated state values.
    per_nodes_estimated_value: List of estimated state values for each node.
    indices: List of start index of each node in the sequence.
    gae_lambda: GAE lambda parameter.
    Returns: List of advantages for each position in the sequence.
    """
    advantages = []
    for i in range(1, len(indices)):
        advantages.append(per_nodes_estimated_value[i] - per_nodes_estimated_value[i-1])
    # Now process the undetermined advantages using GAE
    GAE_advantages = 0.0
    for i, adv in reversed(list(enumerate(advantages))):
        GAE_advantages = adv + gae_lambda * GAE_advantages
        advantages[i] = GAE_advantages
    # Assign advantages to each position in the sequence
    expanded_advantages = []
    for i in range(1, len(indices)):
        position = indices[i]
        last_position = indices[i-1]
        advantage = advantages[i-1]
        expanded_advantages_segment = [advantage] * (position - last_position)
        expanded_advantages.extend(expanded_advantages_segment)

    if max_sequence_length is not None and len(expanded_advantages) < max_sequence_length:
        expanded_advantages.extend([0.0] * (max_sequence_length - len(expanded_advantages)))

    return expanded_advantages