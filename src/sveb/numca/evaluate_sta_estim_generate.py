from rl.utils.rewards import eval_answer_reward
from rl.utils.numca_dict import *
from rl.utils.prepare_dataset import *
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
# import torch
# parse args
import argparse
import random
import json
from datasets import load_dataset
from contextlib import redirect_stdout

def _build_prompt(data, use_default_system_prompt):
    if use_default_system_prompt:
        return [
            {"role": "user", "content": data["problem"]},
        ]

    if data.get("verifier", None) == "code":
        system_prompt = SYSTEM_PROMPT_CODE
    else:
        system_prompt = SYSTEM_PROMPT
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": data["problem"]},
    ]


def main(
    model_name,
    dataset_path,
    output_path,
    save_path,
    mcs_num,
    grpo_num,
    max_length,
    num_of_problems,
    use_default_system_prompt=False,
    tp=1,
    enable_thinking=False,
):
    # Create a sampling params object.
    sampling_params= SamplingParams(temperature=0.9,
                                        max_tokens=max_length,
                                        n = grpo_num,
                                        seed = random.randint(0, 10000)
                                        )
    sampling_params_MCTS = SamplingParams(temperature=0.9,
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
    
    # random sample with num_of_problems, which can be larger than the dataset size but repeatable
    dataset = random.choices(dataset, k=num_of_problems)
    prompts = []
    for data in dataset:
        prompt = _build_prompt(data, use_default_system_prompt)
        input_prompt = tokenizer.apply_chat_template(
            prompt,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        prompts.append(input_prompt)

    # Generate answers
    outputs = llm.generate(prompts, sampling_params=sampling_params)

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
        MCTS_prompt = _build_prompt(data, use_default_system_prompt)
        MCTS_input_prompt = tokenizer.apply_chat_template(
            MCTS_prompt,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        ) + response[:selected_position]
        MCTS_prompts.append(MCTS_input_prompt)

    # Generate MCTS answers
    MCTS_outputs = llm.generate(MCTS_prompts, sampling_params=sampling_params_MCTS)

    estimation_unbiased_mae_list = []
    init_unbiased_mae_list = []
    grpo_unbiased_mae_list = []
    unbiased_noise_mae_list = [[] for _ in range(mcs_num)]
    result_list = []
    # Open the output file
    with open(output_path, 'w') as f:
        # Redirect print statements to the file
        with redirect_stdout(f):
            for output, MCTS_output, MCTS_position, data in zip(outputs, MCTS_outputs, MCTS_positions, dataset):
                completions = [o.text for o in output.outputs]
                problem = data["problem"]
                solutions = [data["solution"] if data.get("verifier", None) == "code" or data.get("verifier", None) == "general" else '$' + data["solution"] + '$' for _ in range(len(completions))]
                verifiers = [data.get("verifier", None) for _ in range(len(completions))]
                rewards = eval_answer_reward(
                    completions = completions, 
                    solutions = solutions, 
                    silence=True,
                    verifiers=verifiers, 
                    problems=[problem]*len(completions)
                )
                grpo_init_state_value = sum(rewards) / len(rewards)
                correct_responses = []
                wrong_responses = []
                for completion, reward in zip(completions, rewards):
                    if reward > 0:
                        correct_responses.append(completion)
                    else:
                        wrong_responses.append(completion)

                # Build the online NumCA table from the sampled completions.
                problem_numca_dict = Numca_dict()
                for completion, reward in zip(completions, rewards):
                    expressions, _ = number_parse(completion)
                    problem_numca_dict.update(expressions, reward)

                response = output.outputs[0].text
                final_reward = rewards[0]
                # Get the state and position for the response being evaluated.
                states, positions = number_parse(response)
                # Calculate the estimated advantage for the current response
                estimated_advantage = problem_numca_dict.advantages(states, positions, len(response), 0, final_reward)

                # Calculate the estimated value function for the selected position by summing the advantages from the root to the selected position
                estimated_state_value = problem_numca_dict.root_node.state_value
                for i in range(0, MCTS_position):
                    # Only add when there is a change (retain original intent)
                    if i + 1 >= len(estimated_advantage):
                        estimated_state_value += estimated_advantage[-1]
                        break
                    if abs(estimated_advantage[i] - estimated_advantage[i + 1]) > 1e-6:
                        estimated_state_value += estimated_advantage[i]

                MCTS_responses = [response[:MCTS_position] + o.text for o in MCTS_output.outputs]
                mcs_solutions = [
                    data["solution"]
                    if data.get("verifier", None) == "code" or data.get("verifier", None) == "general"
                    else '$' + data["solution"] + '$'
                    for _ in range(len(MCTS_responses))
                ]
                mcs_verifiers = [data.get("verifier", None) for _ in range(len(MCTS_responses))]
                mcs_rewards = eval_answer_reward(
                    completions = MCTS_responses,
                    solutions = mcs_solutions,
                    silence=True,
                    verifiers = mcs_verifiers,
                    problems = [problem]*len(MCTS_responses)
                )
                unbiased_state_value = sum(mcs_rewards) / len(mcs_rewards)


                # print the prompt, sampled response, selected position, estimated value function, unbiased value function, mae between estimated and unbiased value function
                print("Problem:", data["problem"])
                print("--------------------------------------------------")
                print("Sampled Response:", response)
                print("--------------------------------------------------")
                print("Selected Position:", MCTS_position)
                print("--------------------------------------------------")
                print("Response until Selected Position:", response[:MCTS_position])
                print("--------------------------------------------------")
                print("Init state Value Function", problem_numca_dict.root_node.state_value)
                print("Estimated Value Function:", estimated_state_value)
                print("Unbiased Value Function:", unbiased_state_value)
                print("mae between estimated and unbiased:", abs(estimated_state_value - unbiased_state_value))
                estimation_unbiased_mae_list.append(abs(estimated_state_value - unbiased_state_value))
                print("mae between init and unbiased:", abs(problem_numca_dict.root_node.state_value - unbiased_state_value))
                init_unbiased_mae_list.append(abs(problem_numca_dict.root_node.state_value - unbiased_state_value))
                print("mae between grpo and unbiased:", abs(grpo_init_state_value - unbiased_state_value))
                grpo_unbiased_mae_list.append(abs(grpo_init_state_value - unbiased_state_value))
                print("--------------------------------------------------")


                used_samples = 1
                while used_samples <= mcs_num:
                    print("Used Samples:", used_samples)
                    # Random sample used_samples responses for estimating the unbiased value function with less samples
                    sampled_list = random.sample(list(range(len(mcs_rewards))), min(used_samples, len(mcs_rewards)))
                    sampled_unbiased_state_value = sum(mcs_rewards[i] for i in sampled_list) / len(sampled_list)
                    print("Unbiased Value Function with {} samples: {}".format(used_samples, sampled_unbiased_state_value))
                    print("mae of unbiased value function with {} samples: {}".format(used_samples, abs(unbiased_state_value - sampled_unbiased_state_value)))
                    unbiased_noise_mae_list[used_samples-1].append(abs(unbiased_state_value - sampled_unbiased_state_value))
                    used_samples += 1
                    print("--------------------------------------------------")
                print("==================================================")

                result = {
                    "problem": problem,
                    "solution": data["solution"],
                    "verifier": data.get("verifier", None),
                    "correct_responses": correct_responses,
                    "wrong_responses": wrong_responses,
                    "response": response,
                    "sampled_response": response[:MCTS_position],
                    "output_reward": final_reward,
                    "unbiased_state_value": unbiased_state_value,
                    "unbiased_state_value_noise": [
                        unbiased_noise_mae_list[used_samples-1][-1]
                        for used_samples in range(1, mcs_num+1)
                    ],
                }
                result_list.append(result)

            print("Average Estimation mae between Estimated and Unbiased Value Function:", sum(estimation_unbiased_mae_list) / len(estimation_unbiased_mae_list))
            print("Average Estimation mae between Init and Unbiased Value Function:", sum(init_unbiased_mae_list) / len(init_unbiased_mae_list))
            print("Average Estimation mae between GRPO and Unbiased Value Function:", sum(grpo_unbiased_mae_list) / len(grpo_unbiased_mae_list))
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
    parser.add_argument("--mcs_num", type=int, default=50)
    parser.add_argument("--grpo_num", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=16384)
    parser.add_argument("--num_of_problems", type=int, default=1000)
    parser.add_argument("--use_default_system_prompt", action='store_true', help="Use default system prompt if set.")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--enable_thinking", action='store_true', help="Whether to enable thinking mode for qwen3.")

    args = parser.parse_args()

    main(
        model_name=args.model_name,
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        save_path=args.save_path,
        mcs_num=args.mcs_num,
        grpo_num=args.grpo_num,
        max_length=args.max_length,
        num_of_problems=args.num_of_problems,
        use_default_system_prompt=args.use_default_system_prompt,
        tp=args.tp,
        enable_thinking=args.enable_thinking,
    )
