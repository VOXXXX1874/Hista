from transformers import AutoTokenizer
import argparse
import random
from contextlib import redirect_stdout
import torch
from ppo_sft.trainer.ppo_sft_trainer import CriticModelWrapper
from rl.utils.prepare_dataset import SYSTEM_PROMPT, SYSTEM_PROMPT_CODE
from tqdm import tqdm
import json


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
    output_path,
    num_of_problems,
    use_default_system_prompt=False,
    enable_thinking=False,
):
    with open(dataset_path, "r") as f:
        dataset = json.load(f)
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

    unbiased_noise_mae_list = []
    estimate_unbiased_mae_list = []
    grpo_unbiased_mae_list = []

    with open(output_path, "w") as f:
        with redirect_stdout(f):
            for data in tqdm(dataset, total=len(dataset)):
                problem = data["problem"]
                sampled_response = data["sampled_response"]

                prompt = _build_prompt(data, use_default_system_prompt)
                input_text = tokenizer.apply_chat_template(
                    prompt,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=enable_thinking,
                ) + sampled_response
                estimated_value = _estimate_state_value(critic_wrapper, tokenizer, input_text)

                unbiased_state_value = data["unbiased_state_value"]
                unbiased_noise_mae_list.append(data["unbiased_state_value_noise"])
                grpo_state_value = sum(problems_offline_rewards[problem]) / len(problems_offline_rewards[problem])

                print("Problem:", data["problem"])
                print("--------------------------------------------------")
                print("Response until Selected Position:", sampled_response)
                print("--------------------------------------------------")
                print("Final Reward of Sampled Response:", data["output_reward"])
                print("--------------------------------------------------")
                print("Estimated Value Function with k:", estimated_value)
                print("Unbiased Value Function:", unbiased_state_value)
                print("GRPO Value Function:", grpo_state_value)
                print("mae between Estimated and Unbiased Value Function:", abs(estimated_value - unbiased_state_value))
                print("mae between GRPO Unbiased and Unbiased Value Function:", abs(grpo_state_value - unbiased_state_value))
                print("--------------------------------------------------")
                estimate_unbiased_mae_list.append(abs(estimated_value - unbiased_state_value))
                grpo_unbiased_mae_list.append(abs(grpo_state_value - unbiased_state_value))
                print("==================================================")

            print("Average mae between Estimated and Unbiased Value Function:", sum(estimate_unbiased_mae_list) / len(estimate_unbiased_mae_list))
            print("Average mae between GRPO Unbiased and Unbiased Value Function:", sum(grpo_unbiased_mae_list) / len(grpo_unbiased_mae_list))
            for i in range(len(unbiased_noise_mae_list[0])):
                sum_noise = 0
                count = 0
                for j in range(len(unbiased_noise_mae_list)):
                    if i < len(unbiased_noise_mae_list[j]):
                        sum_noise += unbiased_noise_mae_list[j][i]
                        count += 1
                if count > 0:
                    print("Average Unbiased Value Function Noise mae with {} samples: {}".format(i + 1, sum_noise / count))
                else:
                    print("Average Unbiased Value Function Noise mae with {} samples: {}".format(i + 1, "Unavailable"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--action_model_name", type=str, default="Qwen/Qwen2.5-1.5B-instruct")
    parser.add_argument("--critic_model_path", type=str, required=True)
    parser.add_argument("--value_head_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, default="data/training_cache/")
    parser.add_argument("--output_path", type=str, default="output/adv_estim_sampling.log")
    parser.add_argument("--num_of_problems", type=int, default=1000)
    parser.add_argument("--use_default_system_prompt", action="store_true", help="Use default system prompt if set.")
    parser.add_argument("--enable_thinking", action="store_true", help="Whether to enable thinking mode for qwen3.")

    args = parser.parse_args()

    main(
        action_model_name=args.action_model_name,
        critic_model_path=args.critic_model_path,
        value_head_path=args.value_head_path,
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        num_of_problems=args.num_of_problems,
        use_default_system_prompt=args.use_default_system_prompt,
        enable_thinking=args.enable_thinking,
    )
