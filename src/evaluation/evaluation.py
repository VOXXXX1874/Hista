import argparse
import contextlib
import json
import traceback
from pathlib import Path

from vllm import LLM, SamplingParams

from rl.utils.prepare_dataset import prepare_dataset
from rl.utils.rewards import eval_answer_reward, format_reward

REQUIRED_EVALUATION_CONFIG_FIELDS = {
    "max_output_tokens",
    "use_default_system_prompt",
    "temperature",
    "num_generations",
    "reward_function",
}


def load_evaluation_config(dataset_name, evaluation_config_name):
    config_path = Path(dataset_name) / ".evaluation_config" / "evaluation_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"evaluation_config.json not found for dataset: {dataset_name}")

    with config_path.open("r", encoding="utf-8") as file:
        all_configs = json.load(file)

    if evaluation_config_name not in all_configs:
        raise KeyError(
            f"evaluation config '{evaluation_config_name}' not found in {config_path}"
        )

    evaluation_config = all_configs[evaluation_config_name]
    missing_fields = REQUIRED_EVALUATION_CONFIG_FIELDS - set(evaluation_config)
    if missing_fields:
        missing = ", ".join(sorted(missing_fields))
        raise KeyError(
            f"evaluation config '{evaluation_config_name}' in {config_path} is missing: {missing}"
        )

    return evaluation_config


def evaluate_dataset(llm, output_folder, dataset_name, enable_thinking, evaluation_config_name, evaluation_config):
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    benchmark_name = Path(dataset_name).name
    log_file = output_path / f"benchmark_sampling_{benchmark_name}.log"
    result_file = output_path / f"result_benchmark_{benchmark_name}.json"

    with log_file.open("w", encoding="utf-8") as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            try:
                _evaluate_dataset(
                    llm=llm,
                    result_file=result_file,
                    dataset_name=dataset_name,
                    enable_thinking=enable_thinking,
                    evaluation_config_name=evaluation_config_name,
                    evaluation_config=evaluation_config,
                )
            except Exception:
                traceback.print_exc()
                raise


def _evaluate_dataset(llm, result_file, dataset_name, enable_thinking, evaluation_config_name, evaluation_config):
    max_output_tokens = evaluation_config["max_output_tokens"]
    use_default_system_prompt = evaluation_config["use_default_system_prompt"]
    temperature = evaluation_config["temperature"]
    num_generations = evaluation_config["num_generations"]
    reward_function = evaluation_config["reward_function"]

    print("=" * 100)
    print(f"Evaluating dataset: {dataset_name}")
    print(f"Using evaluation config: {evaluation_config_name} ({evaluation_config})")

    # evaluation dataset
    dataset = prepare_dataset(
        dataset_name,
        "test",
        use_default_system_prompt=use_default_system_prompt,
    )

    solutions = []
    prompts = []
    processes = []
    problems = []
    verifiers = []
    for data in dataset:
        solutions.append(data['solution'])
        prompts.append(data['prompt'])
        problems.append(data['problem'])
        processes.append(data['process']) if 'process' in data else processes.append('')
        verifiers.append(data['verifier']) if 'verifier' in data else verifiers.append(None)
    
    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=temperature,
                                     max_tokens=max_output_tokens,
                                     n=num_generations,
                                     )

    # # vllm generation
    outputs = llm.chat(prompts,
                           sampling_params,
                           chat_template_kwargs={'enable_thinking': enable_thinking})
    acc_scores = []
    format_scores = []
    result_all = []
    total_acc = 0
    total_format = 0

    completions_for_rewards = []
    problems_for_rewards = []
    solutions_for_rewards = []
    processes_for_rewards = []
    verifiers_for_rewards = []
    for output, gold_solution, gold_process, problem, verifier in zip (outputs, solutions, processes, problems, verifiers):

        for output_completion in output.outputs:
            completion = output_completion.text
            completions_for_rewards.append(completion)
            problems_for_rewards.append(problem)
            solutions_for_rewards.append(gold_solution)
            processes_for_rewards.append(gold_process)
            verifiers_for_rewards.append(verifier)

    if reward_function == 'eval_answer_reward':
        accuracy_rewards = eval_answer_reward(completions = completions_for_rewards,
                                     problems = problems_for_rewards,
                                     solutions = solutions_for_rewards,
                                     verifiers = verifiers_for_rewards)
    else:
        raise ValueError('reward function not found')
    
    for idx in range(len(completions_for_rewards)):
        completion = completions_for_rewards[idx]
        problem = problems_for_rewards[idx]
        gold_solution = solutions_for_rewards[idx]
        gold_process = processes_for_rewards[idx]
        verifier = verifiers_for_rewards[idx]

        acc_score = accuracy_rewards[idx]
        acc_scores.append(acc_score)
        total_acc += acc_score
        format_score = format_reward([[{"content": completion}]], silence=True)[0]
        format_scores.append(format_score)
        total_format += format_score

        result_all.append({
            'problem': problem,
            'gold_solution': gold_solution,
            'gold_process': gold_process,
            'verifier': verifier,
            'completion': completion,
            'acc_score': acc_score,
            'format_score': format_score,
        })

    print('='*100)
    print('eval num: ', len(acc_scores))
    print('eval acc: ', total_acc / len(acc_scores))
    print('eval format: ',total_format / len(format_scores))

    with result_file.open('w', encoding='utf-8') as file:
        json.dump(result_all, file, ensure_ascii=False, indent=4)

    print(f"Saved results to: {result_file}")

    return


def vllm_generate(model_name, output_folder, dataset_names, num_gpus, enable_thinking, evaluation_config_name):
    dataset_configs = {
        dataset_name: load_evaluation_config(dataset_name, evaluation_config_name)
        for dataset_name in dataset_names
    }

    # Create LLM object once and reuse it for all datasets.
    llm = LLM(model=model_name,  # replace your own model
              dtype='bfloat16',
              tensor_parallel_size=num_gpus,  # number of gpu
              gpu_memory_utilization=0.6,  # prevent OOM
              trust_remote_code=True,
              distributed_executor_backend='mp',
              )

    for dataset_name in dataset_names:
        evaluate_dataset(
            llm=llm,
            output_folder=output_folder,
            dataset_name=dataset_name,
            enable_thinking=enable_thinking,
            evaluation_config_name=evaluation_config_name,
            evaluation_config=dataset_configs[dataset_name],
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--model_name',  type=str, required=True,
                        help='model name path')
    parser.add_argument('--output_folder', type=str, required=True,
                        help='output path')
    parser.add_argument('--datasets', type=str, nargs='+', required=True,
                        help='datasets (must be processed local dataset)')
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='number of GPUs to use')
    parser.add_argument('--enable_thinking', type=lambda x: (str(x).lower() == 'true'), default=False,
                        help='whether to enable thinking mode for qwen3')
    parser.add_argument('--evaluation_config', choices=['qwen2.5', 'qwen3', 'r1-distill'], required=True,
                        help='evaluation config')
    args = parser.parse_args()

    vllm_generate(args.model_name,
                  args.output_folder,
                  args.datasets,
                  args.num_gpus,
                  args.enable_thinking,
                  args.evaluation_config)
