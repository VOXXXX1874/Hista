from transformers import AutoTokenizer
# import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
# parse args
import argparse
from contextlib import redirect_stdout
from rl.utils.hista_utils import *
from tqdm import tqdm
from rl.utils.prepare_dataset import SYSTEM_PROMPT, SYSTEM_PROMPT_CODE
from sveb.common import EvaluationReporter, build_offline_pools, load_dataset, render_prompt

def main(
        model_name, 
        dataset_path, 
        output_path, 
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
        enable_thinking=False,
    ):
    # Load the dataset
    dataset = load_dataset(dataset_path, num_of_problems)
    if dataset[0].get("verifier", None) == "code":
        system_prompt = SYSTEM_PROMPT_CODE
    else:
        system_prompt = SYSTEM_PROMPT
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto", attn_implementation="flash_attention_2")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.eval()

    problems_offline_responses, problems_offline_rewards = build_offline_pools(dataset)

    reporter = EvaluationReporter()
    # Open the output file
    with open(output_path, 'w') as f:
        # Redirect print statements to the file
        with redirect_stdout(f):
            for data in tqdm(dataset, total=len(dataset)):
                problem = data["problem"]
                sampled_response = data["sampled_response"]
                full_response = data["response"]

                this_problem_offline_responses = {problem: problems_offline_responses[problem]}
                this_problem_offline_rewards = {problem: problems_offline_rewards[problem]}

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
                response_index = this_problem_offline_responses[problem].index(full_response)
                response_indices = problem_response_indices[problem][response_index]

                # Tokenize the sampled response to get the last token position
                prompt = render_prompt(
                    tokenizer, data, use_default_system_prompt, enable_thinking
                ) + sampled_response
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
                grpo_state_value = sum(problems_offline_rewards[problem]) / len(problems_offline_rewards[problem])
                reporter.add(data, estimated_value, grpo_state_value,
                             noise=data["unbiased_state_value_noise"])

            reporter.summary()

if __name__ == "__main__":
    # parse the arguments: model_name, dataset_path, output_path
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-instruct")
    parser.add_argument("--dataset_path", type=str, default="data/training_cache/")
    parser.add_argument("--output_path", type=str, default="output/adv_estim_sampling.log")
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
    parser.add_argument("--enable_thinking", action='store_true', help="Whether to enable thinking mode for qwen3.")

    args = parser.parse_args()

    if args.selection_method == "uniform":
        selection_method = embedding_selection_uniform
    else:
        raise ValueError("Invalid selection_method. Only 'uniform' is supported.")
    
    if args.average_method == "ema":
        average_method = exponential_running_average
    else:
        raise ValueError("Invalid average_method. Only 'ema' is supported.")
    
    print("Distance method is ignored. Using euclidean distance.")

    main(
        model_name=args.model_name,
        dataset_path=args.dataset_path,
        output_path=args.output_path,
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
        enable_thinking=args.enable_thinking,
    )
