from vllm import LLM, SamplingParams
import json
import argparse
import random
import os
from rl.utils.prepare_dataset import prepare_dataset
from rl.utils.rewards import eval_answer_reward
from vllm_verifier.control import control_verifier_vllm

random.seed(42)

def main(model_name, dataset_path, output_path, num, max_length, tp, use_default_prompt, total_num, proxy_url):

    os.environ["VERIFIER_BASE_URL"] = f"{proxy_url}/v1"

    try:
        control_verifier_vllm("sleep", proxy_url)
    except Exception as e:
        print(f"Failed to control vLLM verifier: {e}. It is ok if you are not using vLLM verifier, please ignore this message. However, if you are using vLLM verifier, please make sure the proxy is running and the URL is correct.")
        print("Continuing with the main process...")

    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0.7,
                                        max_tokens=max_length,
                                        n = num,
                                        seed = random.randint(0, 10000)
                                        )
    # Create LLM object
    llm = LLM(model=model_name,  # replace your own model
                dtype='bfloat16',
                tensor_parallel_size=tp,  # number of gpu
                gpu_memory_utilization=0.7,  # prevent OOM
                trust_remote_code=True,
                distributed_executor_backend='mp',
                )

    dataset = prepare_dataset(dataset_path, "train", use_default_system_prompt=use_default_prompt)

    prompts = []
    count = 0
    for data in dataset:
        prompts.append(data['prompt'])
        count += 1
        if count == total_num:
            break

    # vllm generation
    outputs = llm.chat( prompts, 
                            sampling_params=sampling_params,
                            )

    # Delete the llm object to free up GPU memory
    del llm

    # Wake up the vLLM verifier if it is used
    try:
        control_verifier_vllm("wake_up", proxy_url)
    except Exception as e:
        print(f"Failed to wake up vLLM verifier: {e}. It is ok if you are not using vLLM verifier, please ignore this message. However, if you are using vLLM verifier, please make sure the proxy is running and the URL is correct.")
        print("Continuing with the main process...")

    results = []

    for output, data in zip(outputs, dataset):
        correct_responses = []
        wrong_responses = []
        completions = []
        for i in range(len(output.outputs)):
            completions.append(output.outputs[i].text)

        rewards = eval_answer_reward(completions, [data['solution']] * len(completions), verifiers=[data['verifier']] * len(completions), problems = [data['problem']] * len(completions))
        
        for completion, reward in zip(completions, rewards):
            if reward > 0.5:
                correct_responses.append(completion)
            else:
                wrong_responses.append(completion)

        data['correct_responses'] = correct_responses
        data['wrong_responses'] = wrong_responses
        data.pop('prompt', None)
        data.pop('silence', None)
        data['solution'] = data['solution'][1:-1] if data.get('verifier', 'default') == 'default' else data['solution']

        results.append(data)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save the result
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)

    # Sleep the vLLM verifier if it is used
    try:
        control_verifier_vllm("sleep", proxy_url)
    except Exception as e:
        print(f"Failed to sleep vLLM verifier: {e}. It is ok if you are not using vLLM verifier, please ignore this message. However, if you are using vLLM verifier, please make sure the proxy is running and the URL is correct.")
        print("Ending the main process...")

if __name__ == "__main__":
    # parse the arguments: model_name, dataset_path, output_path
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-instruct")
    parser.add_argument("--dataset_path", type=str, default="src/cv_extraction/openr1-220K/partition2")
    parser.add_argument("--output_path", type=str, default="openr1-extend-p2.json")
    parser.add_argument("--num", type=int, default=20)
    parser.add_argument("--max_length", type=int, default=16384)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--use_default_prompt", action='store_true', help="Whether to use the default prompt or not.")
    parser.add_argument("--total_num", type=int, default=99999)
    parser.add_argument("--proxy_url", type=str, default="http://localhost:8000", help="Proxy endpoint for vLLM control, e.g. http://localhost:8000")

    args = parser.parse_args()

    main(
        model_name=args.model_name,
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        num=args.num,
        max_length=args.max_length,
        tp=args.tp,
        use_default_prompt=args.use_default_prompt,
        total_num=args.total_num,
        proxy_url=args.proxy_url,
    )
