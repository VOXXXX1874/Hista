from rl.utils.numca_dict import *
# parse args
import argparse
import random
from tqdm import tqdm
from contextlib import redirect_stdout
import json

def main(
    dataset_path,
    output_path,
    num_of_problems,
    use_default_system_prompt=False,
    enable_thinking=False,
):
    # Existing samples already include prompt-dependent model outputs. These
    # options are accepted for CLI parity with the generation script.
    _ = (use_default_system_prompt, enable_thinking)

    # Load the dataset
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    # random sample with num_of_problems, which can be larger than the dataset size but repeatable
    dataset = random.sample(dataset, k=num_of_problems)

    problems_offline_responses = {}
    problems_offline_rewards = {}
    for data in dataset:
        problem = data["problem"]
        response = data["response"]
        reward = 1.0 if data["output_reward"] > 0.5 else 0.0

        if problem not in problems_offline_responses:
            correct_responses = list(data["correct_responses"])
            wrong_responses = list(data["wrong_responses"])
            if reward > 0.5:
                if response not in correct_responses:
                    correct_responses.append(response)
            else:
                if response not in wrong_responses:
                    wrong_responses.append(response)

            offline_responses = correct_responses + wrong_responses
            rewards = [1.0] * len(correct_responses) + [0.0] * len(wrong_responses)
            problems_offline_responses[problem] = offline_responses
            problems_offline_rewards[problem] = rewards
            continue

        if response not in problems_offline_responses[problem]:
            problems_offline_responses[problem].append(response)
            problems_offline_rewards[problem].append(reward)

    estimation_unbiased_mae_list = []
    grpo_unbiased_mae_list = []
    unbiased_noise_mae_list = []

    # Open the output file
    with open(output_path, 'w') as f:
        # Redirect print statements to the file
        with redirect_stdout(f):
            for data in tqdm(dataset, total=len(dataset)):
                problem = data["problem"]
                response = data["response"]
                final_reward = 1.0 if data["output_reward"] > 0.5 else 0.0

                problem_numca_dict = Numca_dict()
                for completion, reward in zip(
                    problems_offline_responses[problem],
                    problems_offline_rewards[problem],
                ):
                    expressions, _ = number_parse(completion)
                    problem_numca_dict.update(expressions, reward)

                # Get the state and position for the response being evaluated.
                states, positions = number_parse(response)
                # Calculate the estimated advantage for the current response
                estimated_advantage = problem_numca_dict.advantages(states, positions, len(response), 0, final_reward)

                # Calculate the estimated value function for the selected position by summing the advantages from the root to the selected position
                estimated_state_value = problem_numca_dict.root_node.state_value
                MCTS_position = len(data["sampled_response"])
                for i in range(0, MCTS_position):
                    # Only add when there is a change (retain original intent)
                    if i + 1 >= len(estimated_advantage):
                        estimated_state_value += estimated_advantage[-1]
                        break
                    if abs(estimated_advantage[i] - estimated_advantage[i + 1]) > 1e-6:
                        estimated_state_value += estimated_advantage[i]

                unbiased_state_value = data["unbiased_state_value"]
                unbiased_noise_mae_list.append(data['unbiased_state_value_noise'])
                grpo_state_value = sum(problems_offline_rewards[problem]) / len(problems_offline_rewards[problem])

                # print the prompt, sampled response, selected position, estimated value function, unbiased value function, mae between estimated and unbiased value function
                print("Problem:", data["problem"])
                print("--------------------------------------------------")
                print("Response until Selected Position:", response[:MCTS_position])
                print("--------------------------------------------------")
                print("Final Reward of Sampled Response:", data["output_reward"])
                print("--------------------------------------------------")
                print("Estimated Value Function:", estimated_state_value)
                print("Unbiased Value Function:", unbiased_state_value)
                print("GRPO Value Function:", grpo_state_value)
                print("mae between estimated and unbiased:", abs(estimated_state_value - unbiased_state_value))
                estimation_unbiased_mae_list.append(abs(estimated_state_value - unbiased_state_value))
                print("mae between grpo and unbiased:", abs(grpo_state_value - unbiased_state_value))
                grpo_unbiased_mae_list.append(abs(grpo_state_value - unbiased_state_value))
                print("==================================================")

            print("Average Estimation mae between Estimated and Unbiased Value Function:", sum(estimation_unbiased_mae_list) / len(estimation_unbiased_mae_list))
            print("Average Estimation mae between GRPO and Unbiased Value Function:", sum(grpo_unbiased_mae_list) / len(grpo_unbiased_mae_list))
            for i in range(len(unbiased_noise_mae_list[0])):
                sum_noise = 0
                count = 0
                for j in range(len(unbiased_noise_mae_list)):
                    if i < len(unbiased_noise_mae_list[j]):
                        sum_noise += unbiased_noise_mae_list[j][i]
                        count += 1
                if count > 0:
                    print("Average Unbiased Value Function Noise mae with {} samples: {}".format((i+1), sum_noise / count))
                else:
                    print("Average Unbiased Value Function Noise mae with {} samples: {}".format((i+1), "Unavailable"))

if __name__ == "__main__":
    # parse the arguments: model_name, dataset_path, output_path
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="data/adv_estim_generate/1dot5B_dapo_1737_40/train.json")
    parser.add_argument("--output_path", type=str, default="output/adv_estim_sampling.log")
    parser.add_argument("--num_of_problems", type=int, default=1000)
    parser.add_argument("--use_default_system_prompt", action='store_true', help="Use default system prompt if set.")
    parser.add_argument("--enable_thinking", action='store_true', help="Whether to enable thinking mode for qwen3.")

    args = parser.parse_args()

    main(
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        num_of_problems=args.num_of_problems,
        use_default_system_prompt=args.use_default_system_prompt,
        enable_thinking=args.enable_thinking,
    )
