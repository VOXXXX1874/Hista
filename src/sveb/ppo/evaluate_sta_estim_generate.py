from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import argparse
import random
from datasets import load_dataset
from contextlib import redirect_stdout
import gc
import json
import torch
from rl.utils.rewards import eval_answer_reward
from ppo_sft.trainer.ppo_sft_trainer import CriticModelWrapper
from rl.utils.prepare_dataset import SYSTEM_PROMPT, SYSTEM_PROMPT_CODE
from tqdm import tqdm


def _build_prompt(data, use_default_system_prompt):
    if use_default_system_prompt:
        return [{"role": "user", "content": data["problem"]}]

    if data.get("verifier", None) == "code":
        system_prompt = SYSTEM_PROMPT_CODE
    else:
        system_prompt = SYSTEM_PROMPT
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": data["problem"]},
    ]


def _solution_for_reward(data):
    if data.get("verifier", None) in ("code", "general"):
        return data["solution"]
    return "$" + data["solution"] + "$"


def _estimate_state_value(critic_wrapper, tokenizer, input_text):
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        critic_outputs = critic_wrapper(input_ids=input_ids)
        values = critic_outputs["values"].cpu().to(torch.float32)
    return values[0, -1].item()


def main(
    action_model_name,
    critic_model_path,
    value_head_path,
    dataset_path,
    split,
    output_path,
    save_path,
    grpo_num,
    mcs_num,
    max_length,
    num_of_problems,
    use_default_system_prompt=False,
    tp=1,
    enable_thinking=False,
):
    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=max_length,
        n=grpo_num,
        seed=random.randint(0, 10000),
    )
    sampling_params_MCTS = SamplingParams(
        temperature=0.7,
        max_tokens=max_length,
        n=mcs_num,
        seed=random.randint(0, 10000),
    )

    llm = LLM(
        model=action_model_name,
        dtype="bfloat16",
        tensor_parallel_size=tp,
        gpu_memory_utilization=0.7,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(action_model_name)

    dataset = load_dataset(dataset_path, split=split)
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

    outputs = llm.generate(prompts, sampling_params=sampling_params)

    MCTS_prompts = []
    MCTS_positions = []
    for data, output in zip(dataset, outputs):
        response = output.outputs[0].text
        space_positions = [i for i, char in enumerate(response) if char == " "]
        selected_position = random.choice(space_positions) if space_positions else len(response)
        MCTS_positions.append(selected_position)

        MCTS_prompt = _build_prompt(data, use_default_system_prompt)
        MCTS_input_prompt = tokenizer.apply_chat_template(
            MCTS_prompt,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        ) + response[:selected_position]
        MCTS_prompts.append(MCTS_input_prompt)

    MCTS_outputs = llm.generate(MCTS_prompts, sampling_params=sampling_params_MCTS)

    del llm
    gc.collect()
    torch.cuda.empty_cache()

    print(f"Loading critic model from {critic_model_path} and value head from {value_head_path}")
    critic_wrapper = CriticModelWrapper.load_critic_wrapper(
        action_model_path=action_model_name,
        critic_model_path=critic_model_path,
        value_head_path=value_head_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    critic_wrapper.eval()
    tokenizer = AutoTokenizer.from_pretrained(action_model_name)

    unbiased_noise_mae_list = [[] for _ in range(mcs_num)]
    estimate_unbiased_mae_list = []
    grpo_unbiased_mae_list = []
    result_list = []

    with open(output_path, "w") as f:
        with redirect_stdout(f):
            for output, MCTS_output, MCTS_position, data in tqdm(
                zip(outputs, MCTS_outputs, MCTS_positions, dataset),
                total=len(dataset),
            ):
                problem = data["problem"]
                response = output.outputs[0].text
                completions = [o.text for o in output.outputs]
                solutions = [_solution_for_reward(data) for _ in completions]
                verifiers = [data.get("verifier", None) for _ in completions]
                rewards = eval_answer_reward(
                    completions=completions,
                    solutions=solutions,
                    silence=True,
                    verifiers=verifiers,
                    problems=[problem] * len(completions),
                )

                correct_responses = []
                wrong_responses = []
                for completion, reward in zip(completions, rewards):
                    if reward > 0:
                        correct_responses.append(completion)
                    else:
                        wrong_responses.append(completion)
                grpo_state_value = sum(rewards) / len(rewards)
                response_reward = rewards[0]

                critic_prompt = _build_prompt(data, use_default_system_prompt)
                critic_input_text = tokenizer.apply_chat_template(
                    critic_prompt,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=enable_thinking,
                ) + response[:MCTS_position]
                estimated_value = _estimate_state_value(critic_wrapper, tokenizer, critic_input_text)

                MCTS_responses = [response[:MCTS_position] + o.text for o in MCTS_output.outputs]
                MCTS_solutions = [_solution_for_reward(data) for _ in MCTS_responses]
                MCTS_verifiers = [data.get("verifier", None) for _ in MCTS_responses]
                MCTS_rewards = eval_answer_reward(
                    completions=MCTS_responses,
                    solutions=MCTS_solutions,
                    silence=True,
                    verifiers=MCTS_verifiers,
                    problems=[problem] * len(MCTS_responses),
                )
                correct_list = [1 if reward > 0 else 0 for reward in MCTS_rewards]
                unbiased_state_value = sum(correct_list) / len(correct_list)

                print("Problem:", data["problem"])
                print("--------------------------------------------------")
                print("Sampled Response:", response)
                print("--------------------------------------------------")
                print("Selected Position:", MCTS_position)
                print("--------------------------------------------------")
                print("Response until Selected Position:", response[:MCTS_position])
                print("--------------------------------------------------")
                print("Final Reward of Sampled Response:", response_reward)
                print("--------------------------------------------------")
                print("Estimated Value Function with k:", estimated_value)
                print("Unbiased Value Function:", unbiased_state_value)
                print("GRPO Value Function:", grpo_state_value)
                print("mae between Estimated and Unbiased Value Function:", abs(estimated_value - unbiased_state_value))
                print("mae between GRPO Unbiased and Unbiased Value Function:", abs(grpo_state_value - unbiased_state_value))
                print("--------------------------------------------------")
                estimate_unbiased_mae_list.append(abs(estimated_value - unbiased_state_value))
                grpo_unbiased_mae_list.append(abs(grpo_state_value - unbiased_state_value))

                used_samples = 1
                while used_samples <= mcs_num:
                    print("Used Samples:", used_samples)
                    sampled_list = random.sample(
                        list(range(len(correct_list))),
                        min(used_samples, len(correct_list)),
                    )
                    sampled_correct_list = [correct_list[i] for i in sampled_list]
                    sampled_unbiased_state_value = sum(sampled_correct_list) / len(sampled_correct_list)
                    noise_mae = abs(unbiased_state_value - sampled_unbiased_state_value)
                    print("Unbiased Value Function with {} samples: {}".format(used_samples, sampled_unbiased_state_value))
                    print("mae of unbiased value function with {} samples: {}".format(used_samples, noise_mae))
                    unbiased_noise_mae_list[used_samples - 1].append(noise_mae)
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
                    "output_reward": response_reward,
                    "unbiased_state_value": unbiased_state_value,
                    "unbiased_state_value_noise": [
                        unbiased_noise_mae_list[used_samples - 1][-1]
                        for used_samples in range(1, mcs_num + 1)
                    ],
                }
                result_list.append(result)

            print("Average mae between Estimated and Unbiased Value Function:", sum(estimate_unbiased_mae_list) / len(estimate_unbiased_mae_list))
            print("Average mae between GRPO Unbiased and Unbiased Value Function:", sum(grpo_unbiased_mae_list) / len(grpo_unbiased_mae_list))
            for i in range(mcs_num):
                if len(unbiased_noise_mae_list[i]) > 0:
                    print("Average Unbiased Value Function Noise mae with {} samples: {}".format(i + 1, sum(unbiased_noise_mae_list[i]) / len(unbiased_noise_mae_list[i])))
                else:
                    print("Average Unbiased Value Function Noise mae with {} samples: {}".format(i + 1, "Unavailable"))

    if save_path is not None:
        with open(save_path, "w") as f:
            json.dump(result_list, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--action_model_name", type=str, default="Qwen/Qwen2.5-1.5B-instruct")
    parser.add_argument("--critic_model_path", type=str, required=True)
    parser.add_argument("--value_head_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, default="data/training_cache/")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--output_path", type=str, default="output/adv_estim_sampling.log")
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--grpo_num", type=int, default=20)
    parser.add_argument("--mcs_num", "--num", dest="mcs_num", type=int, default=10)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--num_of_problems", type=int, default=1000)
    parser.add_argument("--use_default_system_prompt", action="store_true", help="Use default system prompt if set.")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--enable_thinking", action="store_true", help="Whether to enable thinking mode for qwen3.")

    args = parser.parse_args()

    main(
        action_model_name=args.action_model_name,
        critic_model_path=args.critic_model_path,
        value_head_path=args.value_head_path,
        dataset_path=args.dataset_path,
        split=args.split,
        output_path=args.output_path,
        save_path=args.save_path,
        grpo_num=args.grpo_num,
        mcs_num=args.mcs_num,
        max_length=args.max_length,
        num_of_problems=args.num_of_problems,
        use_default_system_prompt=args.use_default_system_prompt,
        tp=args.tp,
        enable_thinking=args.enable_thinking,
    )
