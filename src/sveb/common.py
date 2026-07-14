"""Method-independent building blocks for SVEB generate and reuse runs."""

from __future__ import annotations

import gc
import json
import os
import random
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

SEPARATOR = "--------------------------------------------------"
CASE_SEPARATOR = "=================================================="


def make_parent_dirs(*paths: str | None) -> None:
    """Create parent directories for output files, ignoring optional paths."""
    for path in paths:
        if path is None:
            continue
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)


def build_prompt(data: dict[str, Any], use_default_system_prompt: bool) -> list[dict[str, str]]:
    """Build the chat messages in exactly one place for every SVEB method."""
    if use_default_system_prompt:
        return [{"role": "user", "content": data["problem"]}]
    from rl.utils.prepare_dataset import SYSTEM_PROMPT, SYSTEM_PROMPT_CODE

    system_prompt = SYSTEM_PROMPT_CODE if data.get("verifier") == "code" else SYSTEM_PROMPT
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": data["problem"]},
    ]


def render_prompt(tokenizer, data: dict[str, Any], use_default_system_prompt: bool, enable_thinking: bool) -> str:
    return tokenizer.apply_chat_template(
        build_prompt(data, use_default_system_prompt),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def solution_for_reward(data: dict[str, Any]) -> str:
    if data.get("verifier") in ("code", "general"):
        return data["solution"]
    return f'${data["solution"]}$'


def score_responses(data: dict[str, Any], responses: Sequence[str]) -> list[float]:
    """Apply the benchmark verifier consistently to a group of responses."""
    from rl.utils.rewards import eval_answer_reward

    return eval_answer_reward(
        completions=list(responses),
        solutions=[solution_for_reward(data)] * len(responses),
        silence=True,
        verifiers=[data.get("verifier")] * len(responses),
        problems=[data["problem"]] * len(responses),
    )


def split_responses(responses: Sequence[str], rewards: Sequence[float]) -> tuple[list[str], list[str]]:
    correct = [response for response, reward in zip(responses, rewards) if reward > 0]
    wrong = [response for response, reward in zip(responses, rewards) if reward <= 0]
    return correct, wrong


def sample_dataset(dataset: Sequence[dict[str, Any]], count: int, *, replace: bool = False) -> list[dict[str, Any]]:
    if count < 1:
        raise ValueError("num_of_problems must be positive")
    if not dataset:
        raise ValueError("dataset is empty")
    if replace:
        return random.choices(dataset, k=count)
    return random.sample(list(dataset), k=min(count, len(dataset)))


def load_dataset(path: str, count: int, *, replace: bool = False, deduplicate: bool = False) -> list[dict[str, Any]]:
    with open(path, "r") as file:
        dataset = json.load(file)
    if deduplicate:
        seen = set()
        dataset = [item for item in dataset if not (item["problem"] in seen or seen.add(item["problem"]))]
    return sample_dataset(dataset, count, replace=replace)


@dataclass
class GeneratedRollouts:
    dataset: list[dict[str, Any]]
    outputs: list[Any]
    continuation_outputs: list[Any]
    positions: list[int]
    tokenizer: Any


def generate_rollouts(
    model_name: str,
    dataset_path: str,
    num_of_problems: int,
    grpo_num: int,
    mcs_num: int,
    max_length: int,
    use_default_system_prompt: bool,
    tp: int,
    enable_thinking: bool,
    *,
    temperature: float,
    replace: bool = False,
    deduplicate: bool = True,
) -> GeneratedRollouts:
    """Run the common initial-state and selected-position sampling stages."""
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    initial_params = SamplingParams(
        temperature=temperature, max_tokens=max_length, n=grpo_num, seed=42
    )
    continuation_params = SamplingParams(
        temperature=temperature, max_tokens=max_length, n=mcs_num, seed=42
    )
    llm = LLM(
        model=model_name,
        dtype="bfloat16",
        tensor_parallel_size=tp,
        gpu_memory_utilization=0.7,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = load_dataset(
        dataset_path, num_of_problems, replace=replace, deduplicate=deduplicate
    )
    prompts = [
        render_prompt(tokenizer, data, use_default_system_prompt, enable_thinking)
        for data in dataset
    ]
    outputs = llm.generate(prompts, sampling_params=initial_params)

    positions = []
    continuation_prompts = []
    for data, output in zip(dataset, outputs):
        response = output.outputs[0].text
        spaces = [index for index, char in enumerate(response) if char == " "]
        position = random.choice(spaces) if spaces else len(response)
        positions.append(position)
        continuation_prompts.append(
            render_prompt(tokenizer, data, use_default_system_prompt, enable_thinking)
            + response[:position]
        )
    continuation_outputs = llm.generate(continuation_prompts, sampling_params=continuation_params)
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return GeneratedRollouts(dataset, outputs, continuation_outputs, positions, tokenizer)


def build_offline_pools(dataset: Iterable[dict[str, Any]]) -> tuple[dict[str, list[str]], dict[str, list[float]]]:
    """Aggregate reusable responses without mutating loaded dataset records."""
    responses_by_problem: dict[str, list[str]] = {}
    rewards_by_problem: dict[str, list[float]] = {}
    for data in dataset:
        problem = data["problem"]
        if problem not in responses_by_problem:
            correct = list(data.get("correct_responses", []))
            wrong = list(data.get("wrong_responses", []))
            responses_by_problem[problem] = correct + wrong
            rewards_by_problem[problem] = [1.0] * len(correct) + [0.0] * len(wrong)
        response = data["response"]
        if response not in responses_by_problem[problem]:
            responses_by_problem[problem].append(response)
            rewards_by_problem[problem].append(1.0 if data["output_reward"] > 0.5 else 0.0)
    return responses_by_problem, rewards_by_problem


def noise_maes(rewards: Sequence[float], sample_counts: Iterable[int]) -> list[float]:
    target = sum(rewards) / len(rewards)
    result = []
    for count in sample_counts:
        indices = random.sample(range(len(rewards)), min(count, len(rewards)))
        estimate = sum(rewards[index] for index in indices) / len(indices)
        result.append(abs(target - estimate))
    return result


class EvaluationReporter:
    """Uniform per-case logs and aggregate MAE reporting for all methods."""

    def __init__(self) -> None:
        self.estimated_maes: list[float] = []
        self.grpo_maes: list[float] = []
        self.noise_rows: list[list[float]] = []

    def add(self, data: dict[str, Any], estimated: float, grpo: float, *, full_response: str | None = None,
            position: int | None = None, noise: Sequence[float] | None = None, extras: dict[str, Any] | None = None) -> None:
        unbiased = data["unbiased_state_value"]
        sampled = data.get("sampled_response", "")
        print("Problem:", data["problem"])
        print(SEPARATOR)
        if full_response is not None:
            print("Sampled Response:", full_response)
            print(SEPARATOR)
        if position is not None:
            print("Selected Position:", position)
            print(SEPARATOR)
        print("Response until Selected Position:", sampled)
        print(SEPARATOR)
        print("Final Reward of Sampled Response:", data["output_reward"])
        print(SEPARATOR)
        if extras:
            for label, value in extras.items():
                print(f"{label}:", value)
        print("Estimated Value Function:", estimated)
        print("Unbiased Value Function:", unbiased)
        print("GRPO Value Function:", grpo)
        estimated_mae = abs(estimated - unbiased)
        grpo_mae = abs(grpo - unbiased)
        print("MAE between Estimated and Unbiased Value Function:", estimated_mae)
        print("MAE between GRPO and Unbiased Value Function:", grpo_mae)
        self.estimated_maes.append(estimated_mae)
        self.grpo_maes.append(grpo_mae)
        if noise is not None:
            self.noise_rows.append(list(noise))
        print(CASE_SEPARATOR)

    def summary(self) -> None:
        print("Average MAE between Estimated and Unbiased Value Function:", sum(self.estimated_maes) / len(self.estimated_maes))
        print("Average MAE between GRPO and Unbiased Value Function:", sum(self.grpo_maes) / len(self.grpo_maes))
        width = max((len(row) for row in self.noise_rows), default=0)
        for index in range(width):
            values = [row[index] for row in self.noise_rows if index < len(row)]
            value: float | str = sum(values) / len(values) if values else "Unavailable"
            print(f"Average Unbiased Value Function Noise MAE with {index + 1} samples:", value)


def make_result(data: dict[str, Any], correct: list[str], wrong: list[str], response: str,
                sampled_response: str, reward: float, unbiased: float, noise: Sequence[float]) -> dict[str, Any]:
    return {
        "problem": data["problem"],
        "solution": data["solution"],
        "verifier": data.get("verifier"),
        "correct_responses": correct,
        "wrong_responses": wrong,
        "response": response,
        "sampled_response": sampled_response,
        "output_reward": reward,
        "unbiased_state_value": unbiased,
        "unbiased_state_value_noise": list(noise),
    }
