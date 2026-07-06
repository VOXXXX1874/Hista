#!/usr/bin/env python3
"""Wake up all vLLM backend instances.

This script wakes vLLM instances via the proxy's /wake_up endpoint.
"""

import argparse
import asyncio
import sys

import httpx


async def wake_proxy(proxy_url: str, timeout: float) -> bool:
    async with httpx.AsyncClient(timeout=timeout) as client:
        target = proxy_url.rstrip("/")
        try:
            resp = await client.post(f"{target}/wake_up", timeout=timeout)
            if resp.status_code == 200:
                print(f"{target}: wake-up request accepted")
                return True
            print(f"{target}: returned status {resp.status_code}")
            return False
        except Exception as exc:  # pragma: no cover - best effort script
            print(f"{target}: {exc}")
            return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Wake up vLLM instances via proxy")
    parser.add_argument(
        "--proxy-url",
        default="http://localhost:8000",
        help="Proxy endpoint, e.g. http://localhost:8000",
    )
    parser.add_argument("--timeout", type=float, default=20.0, help="Per-request timeout in seconds")
    args = parser.parse_args()

    target = args.proxy_url.rstrip("/")
    print(f"Using proxy endpoint: {target}")

    success = asyncio.run(wake_proxy(target, args.timeout))
    print(f"Done: {1 if success else 0}/1 target(s) responded successfully")
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
