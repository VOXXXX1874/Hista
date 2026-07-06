"""Reward functions for GRPO training."""

import re
from math_verify import LatexExtractionConfig, parse, verify
from math_verify.errors import TimeoutException
from latex2sympy2_extended import NormalizationConfig
from math_verify.parser import *
from sympy import nan, zoo
from openai import AsyncOpenAI
import asyncio
import random
import json
import os
import tempfile
import shutil
import subprocess
import threading
import httpx

general_verifier_prompt = (
    "User: ### Question: {problem}\n\n"
    "### Ground Truth Answer: {ground_truth}\n\n"
    "### Student Answer: {student_answer}\n\n"
    "For the above question, please verify if the student's answer is equivalent to the ground truth answer.\n"
    "Do not solve the question by yourself; just check if the student's answer is equivalent to the ground truth answer.\n"
    "If the student's answer is correct, output \"Final Decision: Yes\". "
    "If the student's answer is incorrect, output \"Final Decision: No\". Assistant:"
)


def _get_env_int(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = int(value)
        return parsed if parsed > 0 else default
    except ValueError:
        return default

VERIFIER_MAX_CONCURRENCY = _get_env_int("VERIFIER_MAX_CONCURRENCY", 100)
LOCAL_TEST_MAX_CONCURRENCY = _get_env_int("LOCAL_TEST_MAX_CONCURRENCY", 8)
CODE_TEST_TIMEOUT = _get_env_int("CODE_TEST_TIMEOUT", 20)
VERIFIER_MAX_TOKENS = _get_env_int("VERIFIER_MAX_TOKENS", 1024)
VERIFIER_BASE_URL = os.getenv("VERIFIER_BASE_URL", "http://localhost:8000/v1")
VERIFIER_MODEL = "TIGER-Lab/general-verifier"
CODE_SANDBOX_RUNTIME = os.getenv("CODE_SANDBOX_RUNTIME", "singularity").strip().lower()
_DEFAULT_SINGULARITY_IMAGE = os.path.abspath(
    os.path.join(os.getcwd(), "singularity_images", "hista_programming.sif")
)
_DEFAULT_DOCKER_IMAGE = "quay.io/jupyter/scipy-notebook:latest"
CODE_SANDBOX_IMAGE = os.getenv(
    "CODE_SANDBOX_IMAGE",
    _DEFAULT_DOCKER_IMAGE if CODE_SANDBOX_RUNTIME == "docker" else _DEFAULT_SINGULARITY_IMAGE,
)
CODE_SANDBOX_TMP_ROOT = os.path.abspath(os.getenv("CODE_SANDBOX_TMP_ROOT", "./tmp"))

async def _run_sync_with_limit(func, *args, sem=None):
    if sem is None:
        return await asyncio.to_thread(func, *args)
    async with sem:
        return await asyncio.to_thread(func, *args)

def outcome_reward(answer, solution):
    try:
        gold_parsed = parse(
            solution,
            extraction_mode="first_match",
            raise_on_error=True,
        )
        answer_parsed = parse(
            answer,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        boxed="all",
                        units=True,
                    ),
                    # Ensures that boxed is tried first
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="first_match",
            raise_on_error=True,
        )
        if len(answer_parsed) != 0 and (answer_parsed[0] == nan or answer_parsed[0] == zoo):
            return gold_parsed, 'nan', 0.0

        reward = float(verify(answer_parsed, gold_parsed, raise_on_error=True))

        return gold_parsed, answer_parsed, reward
    except TimeoutException as e:
        # Timeout during mathematical operations
        print(f"Timeout during verification: {e}")
        return None, None, 0.0
    except Exception as e:
        # Other errors
        print(f"Error during verification: {e}")
        return None, None, 0.0

def _extract_python_code(answer):
    code_pattern = r"```(?:python|py)?\s*(.*?)```"
    matches = re.findall(code_pattern, answer, re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip()

    answer_pattern = r"<answer>(.*?)</answer>"
    match = re.search(answer_pattern, answer, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    return answer.strip()

def _write_code_to_workdir(workdir, code):
    code_path = os.path.join(workdir, "executed_code.py")
    with open(code_path, "w", encoding="utf-8") as f:
        f.write(code)
        if not code.endswith("\n"):
            f.write("\n")


def _run_code_in_singularity_container(workdir, input_data=None, timeout=None):
    cmd = [
        "singularity",
        "exec",
        "-C",
        "--net",
        "--network",
        "none",
        "--no-home",
        "-B",
        f"{workdir}:/workspace",
        "--pwd",
        "/workspace",
        CODE_SANDBOX_IMAGE,
        "python",
        "executed_code.py",
    ]
    run_kwargs = {
        "text": True,
        "capture_output": True,
        "timeout": timeout,
        "check": False,
    }
    if input_data is None:
        run_kwargs["stdin"] = subprocess.DEVNULL
    else:
        run_kwargs["input"] = input_data
    return subprocess.run(cmd, **run_kwargs)


def _run_code_in_docker(workdir, input_data=None, timeout=None):
    cmd = [
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        "--read-only",
        "--tmpfs",
        "/tmp",
        "--cap-drop=ALL",
        "--security-opt",
        "no-new-privileges",
        "--pids-limit",
        "256",
        "--memory",
        "1G",
        "--cpus",
        "2",
        "-v",
        f"{workdir}:/workspace",
        "-w",
        "/workspace",
        "-u",
        f"{os.getuid()}:{os.getgid()}",
        CODE_SANDBOX_IMAGE,
        "python",
        "executed_code.py",
    ]
    run_kwargs = {
        "text": True,
        "capture_output": True,
        "timeout": timeout,
        "check": False,
    }
    if input_data is None:
        run_kwargs["stdin"] = subprocess.DEVNULL
    else:
        cmd.insert(2, "-i")
        run_kwargs["input"] = input_data
    return subprocess.run(cmd, **run_kwargs)


def _run_code_in_sandbox(code, input_data=None, timeout=None):
    timeout = timeout or CODE_TEST_TIMEOUT
    os.makedirs(CODE_SANDBOX_TMP_ROOT, exist_ok=True)
    workdir = tempfile.mkdtemp(prefix="code_", dir=CODE_SANDBOX_TMP_ROOT)
    try:
        _write_code_to_workdir(workdir, code)
        if CODE_SANDBOX_RUNTIME == "singularity":
            return _run_code_in_singularity_container(workdir, input_data=input_data, timeout=timeout)
        if CODE_SANDBOX_RUNTIME == "docker":
            return _run_code_in_docker(workdir, input_data=input_data, timeout=timeout)
        raise ValueError(
            f"Unsupported CODE_SANDBOX_RUNTIME={CODE_SANDBOX_RUNTIME!r}; "
            "expected 'singularity' or 'docker'."
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

def run_one_test_case_singularity_sync(test_case, code):
    result = _run_code_in_sandbox(code, input_data=test_case["input"])
    if result is None or result.returncode != 0:
        return False
    return result.stdout.strip() == test_case["output"].strip()

def run_test_cases_assert_singularity_sync(code):
    result = _run_code_in_sandbox(code)
    return result is not None and result.returncode == 0

async def run_test_cases(test_cases, code, verifier, local_test_sem=None):
    if "humaneval" in verifier:
        full_code = test_cases.replace("FLAGTOBEREPLACED", code)
        return await _run_sync_with_limit(run_test_cases_assert_singularity_sync, full_code, sem=local_test_sem)
    else:
        # Convert test_cases from string to list of dicts if necessary
        if isinstance(test_cases, str):
            test_cases = json.loads(test_cases)
        # Random sample up to 5 test cases
        sampled_test_cases = random.sample(test_cases, min(5, len(test_cases)))
        for test_case in sampled_test_cases:
            result = await _run_sync_with_limit(run_one_test_case_singularity_sync, test_case, code, sem=local_test_sem)
            if not result:
                return False
        return True

def _run_coroutine_in_new_thread(coro):
    result = {}

    def runner():
        try:
            result["value"] = asyncio.run(coro)
        except Exception as e:
            result["error"] = e

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join()
    if "error" in result:
        raise result["error"]
    return result.get("value")

def _is_verifier_yes(output):
    normalized = output.strip().lower()
    if "final decision: yes" in normalized:
        return True
    if "final decision: no" in normalized:
        return False
    if re.search(r"\byes\b", normalized) and not re.search(r"\bno\b", normalized):
        return True
    return False

def outcome_rewards_general_code(answers, solutions, problems, verifiers):
    async def _general_verifier_batch(answers, solutions, problems, verifiers):
        http_client = httpx.AsyncClient(trust_env=False)
        verifier_base_url = os.getenv("VERIFIER_BASE_URL", VERIFIER_BASE_URL)
        verifier_model = os.getenv("VERIFIER_MODEL", VERIFIER_MODEL)
        client = AsyncOpenAI(
            api_key="empty",
            base_url=verifier_base_url,
            http_client=http_client,
        )
        sem = asyncio.Semaphore(VERIFIER_MAX_CONCURRENCY)
        local_test_sem = asyncio.Semaphore(LOCAL_TEST_MAX_CONCURRENCY)

        async def score_one(ans, sol, prob, verifier):
            async with sem:
                try:
                    if verifier == "code" or (verifier is not None and verifier.startswith("code_")):
                        model_answer_extracted = _extract_python_code(ans)
                        if not model_answer_extracted:
                            return 0.0
                        reward = await run_test_cases(sol, model_answer_extracted, verifier, local_test_sem=local_test_sem)
                        return 1.0 if reward else 0.0

                    if verifier == "general_all":
                        model_answer_extracted = ans
                    elif verifier == "general":
                        parse_result = parse(
                            ans,
                            extraction_mode="first_match",
                            extraction_config=[LatexExtractionConfig()],
                            fallback_mode="first_match",
                        )
                        if len(parse_result) >= 2:
                            model_answer_extracted = parse_result[1]
                        else:
                            return 0.0
                    else:
                        return 0.0

                    prompt = general_verifier_prompt.format(
                        problem=prob,
                        ground_truth=sol,
                        student_answer=model_answer_extracted,
                    )
                    response = await client.completions.create(
                        model=verifier_model,
                        prompt=prompt,
                        max_tokens=VERIFIER_MAX_TOKENS,
                        temperature=0.0,
                    )
                    output = response.choices[0].text.strip()
                    return 1.0 if _is_verifier_yes(output) else 0.0
                except Exception as e:
                    print(f"Error in processing problem: {prob}")
                    print(e)
                    print(f"The solution was: {sol}")
                    return 0.0

        try:
            return await asyncio.gather(*(score_one(a, s, p, v) for a, s, p, v in zip(answers, solutions, problems, verifiers)))
        finally:
            await client.close()
            await http_client.aclose()
    
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_general_verifier_batch(answers, solutions, problems, verifiers))
    else:
        # already inside an event loop (e.g., notebook); offload to a dedicated thread
        return _run_coroutine_in_new_thread(
            _general_verifier_batch(answers, solutions, problems, verifiers)
        )
        
# for training
def accuracy_reward(completions, solution, silence, **kwargs):
    """Reward function that checks if the completion is the same as the ground truth."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    verifier = kwargs.get('verifier', [None]*len(contents))
    problem = kwargs.get('problem', [None]*len(contents))
    # Group the completions and solutions according to three different verifiers
    verifier_contents = {
        'general_code': [],
        'default': []
    }
    verifier_solutions = {
        'general_code': [],
        'default': []
    }
    verifier_problem = {
        'general_code': [],
        'default': []
    }
    verifier_verifier = {
        'general_code': [],
        'default': []
    }
    original_indices = []
    for i, (content, sol) in enumerate(zip(contents, solution)):
        if verifier[i] and ("general" in verifier[i] or "code" in verifier[i]):
            verifier_contents['general_code'].append(content)
            verifier_solutions['general_code'].append(sol)
            verifier_problem['general_code'].append("code" if verifier[i] == "code" else problem[i])
            verifier_verifier['general_code'].append(verifier[i])
            original_indices.append(('general_code', len(verifier_contents['general_code']) - 1))
        else:
            verifier_contents['default'].append(content)
            verifier_solutions['default'].append(sol)
            verifier_problem['default'].append(problem[i])
            verifier_verifier['default'].append(verifier[i])
            original_indices.append(('default', len(verifier_contents['default']) - 1))

    # Process the default verifier first
    default_rewards = []
    for content, sol in zip(verifier_contents['default'], verifier_solutions['default']):
        gold_parsed, answer_parsed, reward = outcome_reward(content, sol)
        if not silence[0]:
            print('-'*100)
            try:
                print('\nanswer_parsed:', answer_parsed, '\ngold_parsed:', gold_parsed, '\nreward:', reward)
            except:
                print('\nanswer_parsed:', 'NaN', '\ngold_parsed:', gold_parsed, '\nreward:', reward)
        default_rewards.append(reward)

    # Process the general_code verifier
    if len(verifier_contents['general_code']) > 0:
        general_code_rewards = outcome_rewards_general_code(verifier_contents['general_code'], verifier_solutions['general_code'], verifier_problem['general_code'], verifier_verifier['general_code'])

    # Combine the rewards back to the original order
    general_code_idx = 0
    default_idx = 0
    for v, idx in original_indices:
        if v == 'general_code':
            rewards.append(general_code_rewards[general_code_idx])
            general_code_idx += 1
        else:
            rewards.append(default_rewards[default_idx])
            default_idx += 1

    if not silence[0]:
        print('\naccuracy rewards:', rewards)

    return rewards

def length_reward_threshold(max_length, overlong_punishment_threshold):

    def length_reward(completions, solution, silence, **kwargs):
        """Reward function that gives higher reward for shorter completions."""
        rewards = []
        completion_ids_list = kwargs.get('completion_ids', [None]*len(completions))
        cache_length = max_length - max_length * overlong_punishment_threshold
        for completion_ids in completion_ids_list:
            if completion_ids is None:
                rewards.append(0.0)
            elif len(completion_ids) > max_length - cache_length and len(completion_ids) <= max_length:
                reward = (max_length - cache_length - len(completion_ids)) / cache_length
                rewards.append(reward)
            elif len(completion_ids) > max_length:
                rewards.append(-1.0)
            else:
                rewards.append(0.0)
        if not silence[0]:
            print('\nlength rewards:', rewards)
        return rewards

    return length_reward

def format_reward(completions, silence=False, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content, re.DOTALL) for content in completion_contents]

    rewards = [1.0 if match else 0.0 for match in matches]
    if not silence:
        print('\nformat rewards:', rewards)
        print('-'*100)
    return rewards

# for benchmark.py
# The verifier, silence, and other parameters are passed as one element
def eval_answer_reward(completions, solutions, silence=False, **kwargs):
    """Reward function that checks if the completion is the same as the ground truth."""
    # Get the "verifier" abd "problems" from kwargs if provided
    verifiers = kwargs.get('verifiers', [None]*len(completions))
    problems = kwargs.get('problems', [None]*len(completions))
    # Convert to the input format of accuracy_reward
    formatted_completions = [[{"content": c}] for c in completions]
    rewards = accuracy_reward(completions=formatted_completions, 
                              solution = solutions, 
                              silence=[silence], 
                              verifier=verifiers, 
                              problem=problems)

    return rewards

# for benchmark.py
def eval_format_reward(completions, silence=False, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    # Convert to the input format of format_reward
    formatted_completions = [[{"content": c}] for c in completions]
    rewards = format_reward(completions=formatted_completions, silence=silence)
    return rewards