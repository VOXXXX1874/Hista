from transformers import AutoTokenizer
import argparse
from contextlib import redirect_stdout
import torch
from ppo_sft.trainer.ppo_sft_trainer import CriticModelWrapper
from rl.utils.prepare_dataset import SYSTEM_PROMPT, SYSTEM_PROMPT_CODE
from tqdm import tqdm
from sveb.common import EvaluationReporter, build_offline_pools, load_dataset, render_prompt


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
    dataset = load_dataset(dataset_path, num_of_problems)
    problems_offline_responses, problems_offline_rewards = build_offline_pools(dataset)

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

    reporter = EvaluationReporter()

    with open(output_path, "w") as f:
        with redirect_stdout(f):
            for data in tqdm(dataset, total=len(dataset)):
                problem = data["problem"]
                sampled_response = data["sampled_response"]

                input_text = render_prompt(
                    tokenizer, data, use_default_system_prompt, enable_thinking
                ) + sampled_response
                estimated_value = _estimate_state_value(critic_wrapper, tokenizer, input_text)

                grpo_state_value = sum(problems_offline_rewards[problem]) / len(problems_offline_rewards[problem])
                reporter.add(data, estimated_value, grpo_state_value,
                             noise=data["unbiased_state_value_noise"])

            reporter.summary()


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
