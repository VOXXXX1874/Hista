from rl.utils.numca_dict import *
from rl.utils.prepare_dataset import *
from transformers import AutoTokenizer
# import torch
# parse args
import argparse
import json
from datasets import load_dataset
from contextlib import redirect_stdout
from sveb.common import EvaluationReporter, generate_rollouts, make_parent_dirs, make_result, noise_maes, score_responses, split_responses

def main(
    model_name,
    dataset_path,
    output_path,
    save_path,
    mcs_num,
    grpo_num,
    max_length,
    generation_temperature,
    num_of_problems,
    use_default_system_prompt=False,
    tp=1,
    enable_thinking=False,
):
    make_parent_dirs(output_path, save_path)
    rollouts = generate_rollouts(
        model_name, dataset_path, num_of_problems, grpo_num, mcs_num,
        max_length, use_default_system_prompt, tp, enable_thinking,
        temperature=generation_temperature,
    )
    dataset, outputs = rollouts.dataset, rollouts.outputs
    MCTS_outputs, MCTS_positions = rollouts.continuation_outputs, rollouts.positions

    reporter = EvaluationReporter()
    result_list = []
    # Open the output file
    with open(output_path, 'w') as f:
        # Redirect print statements to the file
        with redirect_stdout(f):
            for output, MCTS_output, MCTS_position, data in zip(outputs, MCTS_outputs, MCTS_positions, dataset):
                completions = [o.text for o in output.outputs]
                rewards = score_responses(data, completions)
                grpo_init_state_value = sum(rewards) / len(rewards)
                correct_responses, wrong_responses = split_responses(completions, rewards)

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
                mcs_rewards = score_responses(data, MCTS_responses)
                unbiased_state_value = sum(mcs_rewards) / len(mcs_rewards)


                case_noise = noise_maes(mcs_rewards, range(1, mcs_num + 1))
                case_data = dict(data, sampled_response=response[:MCTS_position],
                                 output_reward=final_reward,
                                 unbiased_state_value=unbiased_state_value)
                reporter.add(case_data, estimated_state_value, grpo_init_state_value,
                             full_response=response, position=MCTS_position,
                             noise=case_noise,
                             extras={"Initial State Value Function": problem_numca_dict.root_node.state_value})
                result = make_result(case_data, correct_responses, wrong_responses, response,
                                     response[:MCTS_position], final_reward,
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
    parser.add_argument("--mcs_num", type=int, default=20)
    parser.add_argument("--grpo_num", type=int, default=20)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--generation_temperature", type=float, default=0.7)
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
        generation_temperature=args.generation_temperature,
        num_of_problems=args.num_of_problems,
        use_default_system_prompt=args.use_default_system_prompt,
        tp=args.tp,
        enable_thinking=args.enable_thinking,
    )
