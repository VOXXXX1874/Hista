from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
# import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
# parse args
import argparse
import random
from contextlib import redirect_stdout
import gc
import torch
from rl.utils.rewards import eval_answer_reward
from rl.utils.hista_utils import *
from rl.utils.prepare_dataset import *
from tqdm import tqdm
import json

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
    # Create a sampling params object.
    sampling_params= SamplingParams(temperature=0.7,
                                        max_tokens=max_length,
                                        n = grpo_num,
                                        seed = random.randint(0, 10000)
                                        )
    sampling_params_MCTS = SamplingParams(temperature=0.7,
                                        max_tokens=max_length,
                                        n = mcs_num,
                                        seed = random.randint(0, 10000)
                                        )
    # Create LLM object
    llm = LLM(model=model_name,  # replace your own model
                dtype='bfloat16',
                tensor_parallel_size=tp,  # number of gpu
                gpu_memory_utilization=0.7,  # prevent OOM
                trust_remote_code=True,
                )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    with open(dataset_path, 'r') as f:
        dataset = json.load(f)

    # Deduplicate the dataset based on the 'problem' field
    unique_problems = set()
    deduplicated_dataset = []
    for data in dataset:
        problem = data["problem"]
        if problem not in unique_problems:
            unique_problems.add(problem)
            deduplicated_dataset.append(data)
    dataset = deduplicated_dataset

    # random sample with num_of_problems
    dataset = random.sample(dataset, min(num_of_problems, len(dataset)))
    prompts = []
    for data in dataset:
        if use_default_system_prompt:
            prompt = [
                {"role": "user", "content": data["problem"]}
            ]
        else:
            if data.get("verifier", None) == "code":
                system_prompt = SYSTEM_PROMPT_CODE
            else:
                system_prompt = SYSTEM_PROMPT
            prompt = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": data["problem"]}
            ]
        #print("DEBUG prompt: ", prompt)
        input_prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking)
        #print("DEBUG input_prompt: ", input_prompt)
        prompts.append(input_prompt)

    # Generate answers
    outputs = llm.generate(prompts, sampling_params=sampling_params,)

    # MCTS prompts
    MCTS_prompts = []
    MCTS_positions = []
    for data, output in zip(dataset, outputs):
        response = output.outputs[0].text
        # Random select a position of ' ' from the sample response
        space_positions = [i for i, char in enumerate(
            response) if char == ' ']
        if not space_positions:
            selected_position = len(response)
        else:
            selected_position = random.choice(space_positions)
        MCTS_positions.append(selected_position)
        if use_default_system_prompt:
            MCTS_prompt = [{"role": "user", "content": data["problem"]},]
        else:
            if data.get("verifier", None) == "code":
                system_prompt = SYSTEM_PROMPT_CODE
            else:
                system_prompt = SYSTEM_PROMPT
            MCTS_prompt = [{"role": "system", "content": system_prompt},
                        {"role": "user", "content": data["problem"]},]
            
        #print("DEBUG MCTS prompt: ", MCTS_prompt)
        MCTS_input_prompt = tokenizer.apply_chat_template(MCTS_prompt, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking) + response[:selected_position]
        #print("DEBUG MCTS input_prompt: ", MCTS_input_prompt)
        MCTS_prompts.append(MCTS_input_prompt)

    # Generate MCTS answers
    MCTS_outputs = llm.generate(MCTS_prompts, sampling_params=sampling_params_MCTS)

    # Remove the model from GPU to save memory
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto", attn_implementation="flash_attention_2")

    problems_offline_responses = {}
    problems_offline_rewards = {}
    # Encode all offline problems and responses
    for data, output in tqdm(zip(dataset, outputs), total=len(dataset)):
        problem = data["problem"]
        responses = [output.outputs[i].text for i in range(len(output.outputs))]
        correct_responses = []
        wrong_responses = []
        
        # Calculate rewards for all responses
        completions = responses
        solutions = [data["solution"] if data.get("verifier", None) == "code" or data.get("verifier", None) == "general" else '$' + data["solution"] + '$' for _ in range(len(responses))]
        verifiers = [data.get("verifier", None) for _ in range(len(responses))]
        rewards = eval_answer_reward(completions = completions, 
                                     solutions = solutions, 
                                     silence=True,
                                     verifiers=verifiers, 
                                     problems=[problem]*len(responses))
        for response, reward in zip(responses, rewards):
            if reward > 0:
                correct_responses.append(response)
            else:
                wrong_responses.append(response)

        rewards = [1.0] * len(correct_responses) + [0.0] * len(wrong_responses)
        data["correct_responses"] = correct_responses
        data["wrong_responses"] = wrong_responses
        offline_response = correct_responses + wrong_responses
        # Encode the problem and responses
        problems_offline_responses[problem] = offline_response
        problems_offline_rewards[problem] = rewards

    unbiased_noise_mae_list = [[] for _ in range(mcs_num)]
    estimation_unbiased_mae_list = []
    grpo_unbiased_mae_list = []
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
                completions = MCTS_responses
                solutions = [data["solution"] if data.get("verifier", None) == "code" or data.get("verifier", None) == "general" else '$' + data["solution"] + '$' for _ in range(len(MCTS_responses))]
                verifiers = [data.get("verifier", None) for _ in range(len(MCTS_responses))]
                MCTS_rewards = eval_answer_reward(completions = completions, 
                                                 solutions = solutions, 
                                                 silence=True,
                                                 verifiers=verifiers, 
                                                 problems=[problem]*len(MCTS_responses))
                correct_list = [1 if r > 0 else 0 for r in MCTS_rewards]
                unbiased_state_value = sum(correct_list) / len(correct_list)

                grpo_state_value = len(data["correct_responses"]) / (len(data["correct_responses"]) + len(data["wrong_responses"]))

                # print the prompt, sampled response, selected position, estimated value function, unbiased value function, mae between estimated and unbiased value function
                print("Problem:", data["problem"])
                print("--------------------------------------------------")
                print("Response until Selected Position:", response[:MCTS_position])
                print("--------------------------------------------------")
                print("Final Reward of Sampled Response:", response_reward)
                print("--------------------------------------------------")
                print("Estimated Value Function with k:", estimated_value)
                print("Unbiased Value Function:", unbiased_state_value)
                print("GRPO Value Function:", grpo_state_value)
                estimation_unbiased_mae_list.append(abs(estimated_value - unbiased_state_value))
                grpo_unbiased_mae_list.append(abs(grpo_state_value - unbiased_state_value))
                used_samples = 1
                while used_samples <= mcs_num:
                    print("Used Samples:", used_samples)
                    # Random sample used_samples responses for estimating the unbiased value function with less samples
                    sampled_list = random.sample(list(range(len(correct_list))), min(used_samples, len(correct_list)))
                    sampled_correct_list = [correct_list[i] for i in sampled_list]
                    sampled_unbiased_state_value = sum(sampled_correct_list) / len(sampled_correct_list)
                    print("Unbiased Value Function with {} samples: {}".format(used_samples, sampled_unbiased_state_value))
                    print("mae of unbiased value function with {} samples: {}".format(used_samples, abs(unbiased_state_value - sampled_unbiased_state_value)))
                    unbiased_noise_mae_list[used_samples-1].append(abs(unbiased_state_value - sampled_unbiased_state_value))
                    used_samples += 1
                    print("--------------------------------------------------")
                print("==================================================")

                result = {
                    "problem": problem,
                    "solution": data["solution"],
                    "correct_responses": data["correct_responses"],
                    "wrong_responses": data["wrong_responses"],
                    "response": response,
                    "sampled_response": response[:MCTS_position],
                    "output_reward": response_reward,
                    "unbiased_state_value": unbiased_state_value,
                    "unbiased_state_value_noise": [unbiased_noise_mae_list[used_samples-1][-1] for used_samples in range(1, mcs_num+1)],
                }
                result_list.append(result)

            print("Average mae between Estimated and Unbiased Value Function:", sum(estimation_unbiased_mae_list) / len(estimation_unbiased_mae_list))
            print("Average mae between GRPO Unbiased and Unbiased Value Function:", sum(grpo_unbiased_mae_list) / len(grpo_unbiased_mae_list))
            for i in range(mcs_num):
                if len(unbiased_noise_mae_list[i]) > 0:
                    print("Average Unbiased Value Function Noise mae with {} samples: {}".format(i+1, sum(unbiased_noise_mae_list[i]) / len(unbiased_noise_mae_list[i])))
                else:
                    print("Average Unbiased Value Function Noise mae with {} samples: {}".format(i+1, "Unavailable"))

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
    parser.add_argument("--alpha", type=float, default=0.97)
    parser.add_argument("--mean_window", type=int, default=100)
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

# Usage example:
# PYTHONPATH=src python -m cv_extraction.evaluate_adv_estim --model_name "Qwen/Qwen2.5-1.5B-instruct" --dataset_path "data/training_cache/" --output_path "output/adv_estim_sampling.log" --num 50 --max_length 16384 --num_of_problems 4000