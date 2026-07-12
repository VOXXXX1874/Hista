from rl.utils.numca_dict import *
# parse args
import argparse
from tqdm import tqdm
from contextlib import redirect_stdout
from sveb.common import EvaluationReporter, build_offline_pools, load_dataset

def main(
    dataset_path,
    output_path,
    num_of_problems,
):
    # Load the dataset
    dataset = load_dataset(dataset_path, num_of_problems)
    problems_offline_responses, problems_offline_rewards = build_offline_pools(dataset)

    reporter = EvaluationReporter()

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

                grpo_state_value = sum(problems_offline_rewards[problem]) / len(problems_offline_rewards[problem])
                reporter.add(data, estimated_state_value, grpo_state_value,
                             noise=data["unbiased_state_value_noise"])

            reporter.summary()

if __name__ == "__main__":
    # parse the arguments: model_name, dataset_path, output_path
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="data/adv_estim_generate/1dot5B_dapo_1737_40/train.json")
    parser.add_argument("--output_path", type=str, default="output/adv_estim_sampling.log")
    parser.add_argument("--num_of_problems", type=int, default=1000)

    args = parser.parse_args()

    main(
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        num_of_problems=args.num_of_problems,
    )
