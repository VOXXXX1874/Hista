import asyncio
import argparse
import time
from urllib.parse import urlparse

import httpx
from openai import AsyncOpenAI

# Manage vLLM runtime status.
async def control_vllm(action: str, management_url: str = "http://localhost:8000", level: int = 1):
    """
    action: "sleep" or "wake_up"
    level: 1 (offload weights to CPU memory and clear KV cache)
           2 (fully release GPU memory without backup)
    """
    async with httpx.AsyncClient(trust_env=False) as ctrl_client:
        if action == "sleep":
            url = f"{management_url}/sleep"
            response = await ctrl_client.post(url, params={"level": level})
            print(f"[vLLM] Sleep request sent (level={level}), status_code={response.status_code}")
        elif action == "wake_up":
            url = f"{management_url}/wake_up"
            response = await ctrl_client.post(url)
            print(f"[vLLM] Wake-up request sent, status_code={response.status_code}")


def get_management_url(base_url: str) -> str:
    parsed = urlparse(base_url)
    return f"{parsed.scheme}://{parsed.netloc}"

def generate_prompts(num_requests: int):
    prompts = []
    for i in range(num_requests):
        question = f"Factor the following quadratic: $3 x^2 + {60 + i} x - 810$"
        ground_truth = "\\frac{3(2x-9)(x+6)(x+10)}{2}"
        student_answer = "\\frac{3}{2}(x+6)(2x-9)(x+10)"
        prompt = (
            f"User: ### Question: {question}\n\n### Ground Truth Answer: {ground_truth}\n\n### Student Answer: {student_answer}\n\n"
            "For the above question, please verify if the student's answer is equivalent to the ground truth answer.\n"
            "Assistant:"
        )
        prompts.append(prompt)
    return prompts

async def send_request(
    semaphore: asyncio.Semaphore,
    client: AsyncOpenAI,
    prompt: str,
    req_id: int,
):
    async with semaphore:
        start_time = time.time()
        try:
            response = await client.completions.create(
                model="TIGER-Lab/general-verifier",
                prompt=prompt,
                max_tokens=512,
                temperature=0.0,
            )
            return {"success": True, "latency": time.time() - start_time, "error": None}
        except Exception as e:
            return {"success": False, "latency": time.time() - start_time, "error": str(e)}


def parse_args():
    parser = argparse.ArgumentParser(description="vLLM verifier load test")
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000/v1",
        help="OpenAI-compatible vLLM base URL, e.g. http://localhost:8000/v1",
    )
    return parser.parse_args()


async def main():
    args = parse_args()
    management_url = get_management_url(args.base_url)

    # Initialize async OpenAI client in main.
    http_client = httpx.AsyncClient(trust_env=False)
    client = AsyncOpenAI(
        base_url=args.base_url,
        api_key="empty",
        http_client=http_client,
    )

    TOTAL_REQUESTS = 100
    CONCURRENCY = 16

    try:
        # Wake up the verifier before running the load test.
        await control_vllm("wake_up", management_url=management_url)

        prompts = generate_prompts(TOTAL_REQUESTS)
        semaphore = asyncio.Semaphore(CONCURRENCY)

        print("=== Starting load test ===")
        start_wall_time = time.time()
        tasks = [send_request(semaphore, client, prompts[i], i) for i in range(TOTAL_REQUESTS)]
        results = await asyncio.gather(*tasks)
        total_wall_time = time.time() - start_wall_time

        success_count = len([r for r in results if r["success"]])
        print(f"====== Load test finished: {success_count} / {TOTAL_REQUESTS} succeeded ======")
        print(f"Total wall time: {total_wall_time:.2f}s")

        # Put verifier to sleep to release GPU memory for training.
        await control_vllm("sleep", management_url=management_url, level=1)
    finally:
        await http_client.aclose()

if __name__ == "__main__":
    asyncio.run(main())