from transformers import AutoTokenizer
import argparse
from contextlib import redirect_stdout
import json
import torch
from ppo_sft.trainer.ppo_sft_trainer import CriticModelWrapper
from tqdm import tqdm
from sveb.common import EvaluationReporter, generate_rollouts, make_result, noise_maes, render_prompt, score_responses, split_responses


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
    save_path,
    grpo_num,
    mcs_num,
    max_length,
    num_of_problems,
    use_default_system_prompt=False,
    tp=1,
    enable_thinking=False,
):
    rollouts = generate_rollouts(
        action_model_name, dataset_path, num_of_problems, grpo_num, mcs_num,
        max_length, use_default_system_prompt, tp, enable_thinking,
        temperature=0.7,
    )
    dataset, outputs = rollouts.dataset, rollouts.outputs
    MCTS_outputs, MCTS_positions = rollouts.continuation_outputs, rollouts.positions

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
    result_list = []

    with open(output_path, "w") as f:
        with redirect_stdout(f):
            for output, MCTS_output, MCTS_position, data in tqdm(
                zip(outputs, MCTS_outputs, MCTS_positions, dataset),
                total=len(dataset),
            ):
                response = output.outputs[0].text
                completions = [o.text for o in output.outputs]
                rewards = score_responses(data, completions)
                correct_responses, wrong_responses = split_responses(completions, rewards)
                grpo_state_value = sum(rewards) / len(rewards)
                response_reward = rewards[0]

                critic_input_text = render_prompt(
                    tokenizer, data, use_default_system_prompt, enable_thinking
                ) + response[:MCTS_position]
                estimated_value = _estimate_state_value(critic_wrapper, tokenizer, critic_input_text)

                MCTS_responses = [response[:MCTS_position] + o.text for o in MCTS_output.outputs]
                MCTS_rewards = score_responses(data, MCTS_responses)
                correct_list = [1 if reward > 0 else 0 for reward in MCTS_rewards]
                unbiased_state_value = sum(correct_list) / len(correct_list)

                case_noise = noise_maes(correct_list, range(1, mcs_num + 1))
                case_data = dict(data, sampled_response=response[:MCTS_position],
                                 output_reward=response_reward,
                                 unbiased_state_value=unbiased_state_value)
                reporter.add(case_data, estimated_value, grpo_state_value,
                             full_response=response, position=MCTS_position, noise=case_noise)
                result = make_result(case_data, correct_responses, wrong_responses, response,
                                     response[:MCTS_position], response_reward,
                                     unbiased_state_value, case_noise)
                result_list.append(result)

            reporter.summary()

    if save_path is not None:
        with open(save_path, "w") as f:
            json.dump(result_list, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--action_model_name", type=str, default="Qwen/Qwen2.5-1.5B-instruct")
    parser.add_argument("--critic_model_path", type=str, required=True)
    parser.add_argument("--value_head_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, default="data/training_cache/")
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
