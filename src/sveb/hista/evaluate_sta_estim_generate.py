from transformers import AutoTokenizer
# import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
# parse args
import argparse
import json
from contextlib import redirect_stdout
from rl.utils.hista_utils import *
from rl.utils.prepare_dataset import *
from tqdm import tqdm
from sveb.common import EvaluationReporter, generate_rollouts, make_result, noise_maes, score_responses, split_responses

def main(
    model_name,
    dataset_path,
    output_path,
    save_path,
    grpo_num,
    mcs_num,
    max_length,
    num_of_problems,
    t,
    layer,
    max_k,
    min_k,
    min_interval,
    alpha,
    mean_window,
    min_distance,
    selection_method,
    average_method,
    use_default_system_prompt=False,
    tp = 1,
    enable_thinking=False,
):
    rollouts = generate_rollouts(
        model_name, dataset_path, num_of_problems, grpo_num, mcs_num,
        max_length, use_default_system_prompt, tp, enable_thinking,
        temperature=0.7, replace=False, deduplicate=True,
    )
    dataset, outputs, tokenizer = rollouts.dataset, rollouts.outputs, rollouts.tokenizer
    MCTS_outputs, MCTS_positions = rollouts.continuation_outputs, rollouts.positions
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto", attn_implementation="flash_attention_2")

    problems_offline_responses = {}
    problems_offline_rewards = {}
    # Encode all offline problems and responses
    for data, output in tqdm(zip(dataset, outputs), total=len(dataset)):
        problem = data["problem"]
        responses = [item.text for item in output.outputs]
        rewards = score_responses(data, responses)
        correct_responses, wrong_responses = split_responses(responses, rewards)

        rewards = [1.0] * len(correct_responses) + [0.0] * len(wrong_responses)
        data["correct_responses"] = correct_responses
        data["wrong_responses"] = wrong_responses
        offline_response = correct_responses + wrong_responses
        # Encode the problem and responses
        problems_offline_responses[problem] = offline_response
        problems_offline_rewards[problem] = rewards

    reporter = EvaluationReporter()
    result_list = []
    # Open the output file
    with open(output_path, 'w') as f:
        # Redirect print statements to the file
        with redirect_stdout(f):
            for output, MCTS_output, MCTS_position, data in tqdm(zip(outputs, MCTS_outputs, MCTS_positions, dataset), total=len(dataset)):
                problem = data["problem"]
                response = output.outputs[0].text

                this_problem_offline_responses = {problem: problems_offline_responses[problem]}
                this_problem_offline_rewards = {problem: problems_offline_rewards[problem]}
                if use_default_system_prompt:
                    system_prompt = None
                else:
                    if data.get("verifier", None) == "code":
                        system_prompt = SYSTEM_PROMPT_CODE
                    else:
                        system_prompt = SYSTEM_PROMPT
                problem_representation_embeddings, problem_response_indices, problem_representation_indices = steps_embedding(
                    model, 
                    tokenizer, 
                    None if use_default_system_prompt else system_prompt, 
                    this_problem_offline_responses, 
                    batch_size=1, 
                    layer=layer,
                    min_interval=min_interval,
                    alpha=alpha,
                    mean_window=mean_window,
                    selection_method=selection_method,
                    average_method=average_method,
                    response_pattern="<think>" if "R1" in model_name else "<|im_start|>assistant",
                    enable_thinking=enable_thinking,
                )

                problem_embeddings_sequence, problem_nodes_sequence, problem_representations_sequence = to_embeddings_indices_sequence(
                    problem_representation_embeddings, problem_representation_indices, this_problem_offline_rewards)
                # Get the embeddings from the offline encoded responses
                response_index = this_problem_offline_responses[problem].index(response)
                response_reward = this_problem_offline_rewards[problem][response_index]
                response_indices = problem_response_indices[problem][response_index]

                # Tokenize the sampled response to get the last token position
                if use_default_system_prompt:
                    prompt = [
                        {"role": "user", "content": problem}
                    ]
                else:
                    prompt = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": problem},
                    ]
                
                prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking) + response[:MCTS_position]
                input_ids = tokenizer(prompt, return_tensors='pt').input_ids[0]
                last_token_position = input_ids.shape[0] - 1  # Last token position
                
                # Find the corresponding node in the offline encoded response
                last_offline_node_index = None
                for i in range(len(response_indices)):
                    if response_indices[i] >= last_token_position:
                        last_offline_node_index = i - 1
                        break

                online_representations = problem_representations_sequence[problem][response_index]
                online_nodes = [(0, index + 1) for index in problem_representation_indices[problem][response_index]]

                existing_nodes = problem_nodes_sequence[problem]

                per_nodes_estimated_value = calculate_weighted_distance_for_nodes(
                    problem_embeddings_sequence[problem], 
                    existing_nodes, 
                    problem_embeddings_sequence[problem][online_representations[0]:online_representations[1]], 
                    online_nodes, 
                    online_representations,
                    max_k,
                    min_k,
                    t,
                    min_distance,
                )

                estimated_value = per_nodes_estimated_value[last_offline_node_index]

                MCTS_responses = [response[:MCTS_position] + o.text for o in MCTS_output.outputs]
                # Compute the rewards for MCTS responses
                MCTS_rewards = score_responses(data, MCTS_responses)
                correct_list = [1 if r > 0 else 0 for r in MCTS_rewards]
                unbiased_state_value = sum(correct_list) / len(correct_list)

                grpo_state_value = len(data["correct_responses"]) / (len(data["correct_responses"]) + len(data["wrong_responses"]))

                case_noise = noise_maes(correct_list, range(1, mcs_num + 1))
                case_data = dict(data, sampled_response=response[:MCTS_position],
                                 output_reward=response_reward,
                                 unbiased_state_value=unbiased_state_value)
                reporter.add(case_data, estimated_value, grpo_state_value,
                             full_response=response, position=MCTS_position, noise=case_noise)
                result = make_result(case_data, data["correct_responses"],
                                     data["wrong_responses"], response,
                                     response[:MCTS_position], response_reward,
                                     unbiased_state_value, case_noise)
                result_list.append(result)

            reporter.summary()

    if save_path is not None:
        with open(save_path, 'w') as f:
            json.dump(result_list, f, indent=4)    

if __name__ == "__main__":
    # parse the arguments: model_name, dataset_path, output_path
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-instruct")
    parser.add_argument("--dataset_path", type=str, default="data/training_cache/")
    parser.add_argument("--output_path", type=str, default="output/adv_estim_sampling.log")
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--grpo_num", type=int, default=20)
    parser.add_argument("--mcs_num", type=int, default=10)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--num_of_problems", type=int, default=1000)
    parser.add_argument("--t", type=float, default=1.0)
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--max_k", type=int, default=66)
    parser.add_argument("--min_k", type=int, default=6)
    parser.add_argument("--min_interval", type=int, default=50)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--mean_window", type=int, default=5)
    parser.add_argument("--min_distance", type=float, default=5.0)
    parser.add_argument("--selection_method", type=str, default="distance", help="Method for selecting embeddings: embedding_selection_based_on_distance or embedding_selection_uniform")
    parser.add_argument("--average_method", type=str, default="ema", help="Method for averaging embeddings: ema or mean")
    parser.add_argument("--use_default_system_prompt", action='store_true', help="Use default system prompt if set.")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--enable_thinking", action='store_true', help="Whether to enable thinking mode for qwen3.")

    args = parser.parse_args()

    selection_method = embedding_selection_uniform

    average_method = exponential_running_average

    main(
        model_name=args.model_name,
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        save_path=args.save_path,
        grpo_num=args.grpo_num,
        mcs_num=args.mcs_num,
        max_length=args.max_length,
        num_of_problems=args.num_of_problems,
        t = args.t,
        layer = - args.layer,
        max_k = args.max_k,
        min_k = args.min_k,
        min_interval = args.min_interval,
        alpha = args.alpha,
        mean_window = args.mean_window,
        min_distance = args.min_distance,
        selection_method = selection_method,
        average_method = average_method,
        use_default_system_prompt=args.use_default_system_prompt,
        tp=args.tp,
        enable_thinking=args.enable_thinking
    )