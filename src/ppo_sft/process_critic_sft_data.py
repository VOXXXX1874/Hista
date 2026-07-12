import argparse
import json
import os
import random

from transformers import AutoTokenizer

from rl.utils.prepare_dataset import SYSTEM_PROMPT, SYSTEM_PROMPT_CODE


def create_new_qa(qa, response, reward, tokenizer):
    """Create one response-level critic-SFT example."""
    system_prompt = (
        SYSTEM_PROMPT_CODE if qa.get("verifier") == "code" else SYSTEM_PROMPT
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": qa["problem"]},
        {"role": "assistant", "content": response},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, continue_final_message=True
    )
    return {"text": text, "reward": reward}


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return data


def merge_rollout_datasets(datasets):
    """Merge rollout files by problem, preserving all unique responses."""
    merged_by_problem = {}
    response_sets = {}
    for dataset in datasets:
        for qa in dataset:
            problem = qa["problem"]
            if problem not in merged_by_problem:
                merged_by_problem[problem] = dict(qa)
                merged_by_problem[problem]["correct_responses"] = []
                merged_by_problem[problem]["wrong_responses"] = []
                response_sets[problem] = {"correct": set(), "wrong": set()}

            merged = merged_by_problem[problem]
            seen = response_sets[problem]
            for response in qa.get("correct_responses", []):
                if response not in seen["correct"]:
                    merged["correct_responses"].append(response)
                    seen["correct"].add(response)
            for response in qa.get("wrong_responses", []):
                if response not in seen["wrong"]:
                    merged["wrong_responses"].append(response)
                    seen["wrong"].add(response)
    return list(merged_by_problem.values())


def load_relevant_rollouts(paths, sveb_data, mode):
    """Load files sequentially and discard mode-irrelevant rows immediately."""
    sveb_problems = {qa["problem"] for qa in sveb_data}
    relevant_datasets = []
    for path in paths:
        dataset = load_json(path)
        if mode == "ppo-1":
            relevant = [qa for qa in dataset if qa["problem"] not in sveb_problems]
        else:
            relevant = [qa for qa in dataset if qa["problem"] in sveb_problems]
        relevant_datasets.append(relevant)
        del dataset
    return merge_rollout_datasets(relevant_datasets)


def select_qa_dataset(sveb_data, rollout_data, mode, num_samples, rng):
    """Select problem-level examples according to the PPO epoch setting.

    ppo-1 uses rollout problems outside SVEB.  ppo-n uses SVEB problems for
    which responses already exist in rollout_data; SVEB metadata (in
    particular verifier) is retained and only the responses come from the
    rollout file.
    """
    sveb_by_problem = {}
    for qa in sveb_data:
        problem = qa["problem"]
        # A repeated benchmark row must not produce duplicate training data.
        sveb_by_problem.setdefault(problem, qa)

    if mode == "ppo-1":
        candidates = [
            qa for qa in rollout_data if qa["problem"] not in sveb_by_problem
        ]
    else:
        candidates = []
        seen = set()
        for rollout_qa in rollout_data:
            problem = rollout_qa["problem"]
            if problem not in sveb_by_problem or problem in seen:
                continue
            if not (
                rollout_qa.get("correct_responses")
                or rollout_qa.get("wrong_responses")
            ):
                continue
            qa = dict(sveb_by_problem[problem])
            qa["correct_responses"] = rollout_qa.get("correct_responses", [])
            qa["wrong_responses"] = rollout_qa.get("wrong_responses", [])
            candidates.append(qa)
            seen.add(problem)

    rng.shuffle(candidates)
    if num_samples > len(candidates):
        print(
            f"Warning: requested {num_samples} problems, but only "
            f"{len(candidates)} are available for {mode}; using all of them."
        )
    return candidates[:num_samples]


def expand_responses(qa_dataset, tokenizer):
    result = []
    for qa in qa_dataset:
        for response in qa.get("correct_responses", []):
            result.append(create_new_qa(qa, response, 1.0, tokenizer))
        for response in qa.get("wrong_responses", []):
            result.append(create_new_qa(qa, response, 0.0, tokenizer))
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Process PPO Critic SFT Dataset")
    parser.add_argument(
        "--sveb_data",
        required=True,
        help="Path to the state value estimation benchmark JSON file.",
    )
    parser.add_argument(
        "--rollout_data",
        nargs="+",
        required=True,
        help="Paths to one or more JSON files containing correct/wrong rollouts.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory in which train.json and test.json are written.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=3000,
        help="Maximum number of problem-level samples to use before splitting.",
    )
    parser.add_argument(
        "--num_test_samples",
        type=int,
        default=2000,
        help="Maximum number of response-level samples written to test.json.",
    )
    parser.add_argument(
        "--mode",
        choices=["ppo-1", "ppo-n"],
        required=True,
        help="ppo-1 is the first epoch; ppo-n denotes subsequent epochs.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--tokenizer",
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Tokenizer used to render the chat template.",
    )
    args = parser.parse_args()
    if args.num_samples < 1:
        parser.error("--num_samples must be positive")
    if args.num_test_samples < 1:
        parser.error("--num_test_samples must be positive")
    return args


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    sveb_data = load_json(args.sveb_data)
    rollout_data = load_relevant_rollouts(
        args.rollout_data, sveb_data, args.mode
    )

    qa_dataset = select_qa_dataset(
        sveb_data, rollout_data, args.mode, args.num_samples, rng
    )
    split_index = len(qa_dataset) // 5 * 4
    train_qa = qa_dataset[:split_index]
    test_qa = qa_dataset[split_index:]

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    train_data = expand_responses(train_qa, tokenizer)
    test_data = expand_responses(test_qa, tokenizer)
    rng.shuffle(train_data)
    rng.shuffle(test_data)
    test_data = test_data[: args.num_test_samples]

    os.makedirs(args.output_dir, exist_ok=True)
    for filename, data in (("train.json", train_data), ("test.json", test_data)):
        output_path = os.path.join(args.output_dir, filename)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    print(
        f"Selected {len(qa_dataset)} problems and wrote "
        f"{len(train_data)} train / {len(test_data)} test response examples "
        f"to {args.output_dir}."
    )


if __name__ == "__main__":
    main()
