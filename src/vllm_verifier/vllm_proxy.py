# vllm_proxy.py
import argparse
import asyncio
from contextlib import asynccontextmanager
import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn

# 1. Argument Parsing
parser = argparse.ArgumentParser(description="vLLM Load Balancing Proxy")
parser.add_argument("--num-gpus", type=int, default=8, help="Number of GPUs to use / instances to run")
parser.add_argument("--proxy-port", type=int, default=8000, help="Port for the proxy server to listen on")
args, _ = parser.parse_known_args()

# Dynamically generate backend URLs from the proxy port:
# proxy P -> vLLM backends P+1, P+2, ..., P+num_gpus
backend_start_port = args.proxy_port + 1
VLLM_BACKENDS = [
    f"http://localhost:{port}"
    for port in range(backend_start_port, backend_start_port + args.num_gpus)
]
CONTROL_PATHS = {"sleep", "wake_up", "is_sleeping"}
SLEEP_MAX_ATTEMPTS = 3

# Globally reused asynchronous HTTP client with optimized connection pool
http_client = httpx.AsyncClient(
    timeout=120.0, 
    limits=httpx.Limits(max_connections=1000, max_keepalive_connections=200)
)

# 2. Lifespan management: Automatically send sleep signal on startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage the startup and shutdown lifespan of the proxy server."""
    print(f"\n⏳ [Proxy Startup] Detected {len(VLLM_BACKENDS)} backends. Waiting for them to be ready and sending initial sleep signals...")
    
    async def wait_and_sleep(backend_url):
        url = f"{backend_url}/sleep"
        # Probing and retry mechanism (up to 15 attempts, 2s interval to cover vLLM cold start time)
        for attempt in range(15):
            try:
                # Try to send a sleep request with level=1 to offload weights to CPU
                res = await http_client.post(url, params={"level": 1}, timeout=5.0)
                if res.status_code == 200:
                    print(f"✅ [Proxy Startup] Backend {backend_url} successfully put to sleep. VRAM released.")
                    return True
            except (httpx.ConnectError, httpx.ConnectTimeout):
                # vLLM instance is not fully up yet (port closed), continue waiting
                pass
            except Exception as e:
                print(f"⚠️ [Proxy Startup] Backend {backend_url} responded with an anomaly: {str(e)}. Retrying...")
            await asyncio.sleep(2)
        
        print(f"❌ [Proxy Startup] Warning: Backend {backend_url} did not become ready within the timeout. Please check vllm_gpu_*.log.")
        return False

    # FIX: Wrap the gather inside an async function so create_task receives a proper coroutine
    async def run_initial_sleeps():
        await asyncio.gather(*[wait_and_sleep(b) for b in VLLM_BACKENDS])

    # Concurrently send sleep signals to all backends in the background without blocking proxy startup
    asyncio.create_task(run_initial_sleeps())
    
    yield  # Proxy server is running and serving requests
    
    # Clean up the connection pool on shutdown
    await http_client.aclose()

# 3. Initialize FastAPI with the lifespan handler
app = FastAPI(lifespan=lifespan)

# Round-robin counter and lock
_counter = 0
counter_lock = asyncio.Lock()

async def get_next_backend():
    global _counter
    async with counter_lock:
        backend = VLLM_BACKENDS[_counter % len(VLLM_BACKENDS)]
        _counter += 1
        return backend


def _parse_response_body(res: httpx.Response):
    if not res.content:
        return None
    try:
        return res.json()
    except ValueError:
        return res.text


async def _request_backend_control(request, backend_url, path, headers, params, body):
    url = f"{backend_url}/{path}"
    try:
        res = await http_client.request(request.method, url, headers=headers, params=params, content=body)
        return {
            "backend": backend_url,
            "ok": 200 <= res.status_code < 300,
            "status_code": res.status_code,
            "body": _parse_response_body(res),
        }
    except Exception as e:
        return {
            "backend": backend_url,
            "ok": False,
            "error": repr(e),
        }


async def _verify_sleep_state(backend_url):
    try:
        res = await http_client.get(f"{backend_url}/is_sleeping")
        body = _parse_response_body(res)
        is_sleeping = isinstance(body, dict) and body.get("is_sleeping") is True
        return {
            "status_code": res.status_code,
            "body": body,
            "is_sleeping": is_sleeping,
        }
    except Exception as e:
        return {
            "is_sleeping": False,
            "error": repr(e),
        }


async def _sleep_backend_with_retries(request, backend_url, headers, params, body):
    attempts = []
    for attempt in range(1, SLEEP_MAX_ATTEMPTS + 1):
        result = await _request_backend_control(request, backend_url, "sleep", headers, params, body)
        await asyncio.sleep(0.2)
        sleep_state = await _verify_sleep_state(backend_url)
        result["attempt"] = attempt
        result["sleep_state"] = sleep_state
        attempts.append(result)

        if result["ok"] and sleep_state["is_sleeping"]:
            return {
                "backend": backend_url,
                "ok": True,
                "status_code": result.get("status_code"),
                "body": result.get("body"),
                "sleep_state": sleep_state,
                "attempt": attempt,
                "attempts": attempts,
            }

        if attempt < SLEEP_MAX_ATTEMPTS:
            await asyncio.sleep(0.5)

    final_result = attempts[-1]
    return {
        "backend": backend_url,
        "ok": False,
        "status_code": final_result.get("status_code"),
        "body": final_result.get("body"),
        "sleep_state": final_result.get("sleep_state"),
        "attempt": SLEEP_MAX_ATTEMPTS,
        "attempts": attempts,
        "error": "Backend did not enter sleep mode after retries",
    }

@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def route_proxy(request: Request, path: str):
    body = await request.body()
    headers = dict(request.headers)
    headers.pop("host", None)
    params = request.query_params

    # Broadcast logic (manually triggered control at runtime, e.g., /sleep, /wake_up)
    if path in CONTROL_PATHS:
        print(f"📢 [Proxy] Broadcasting control command: /{path} to all instances...")
        if path == "sleep":
            results = await asyncio.gather(*[
                _sleep_backend_with_retries(request, b, headers, params, body)
                for b in VLLM_BACKENDS
            ])
        else:
            results = await asyncio.gather(*[
                _request_backend_control(request, b, path, headers, params, body)
                for b in VLLM_BACKENDS
            ])

        ok = all(result["ok"] for result in results)
        status_code = 200 if ok else 502
        status = "ok" if ok else "error"
        return JSONResponse(
            status_code=status_code,
            content={"status": status, "path": path, "details": results},
        )

    # Round-robin logic (standard inference requests)
    backend = await get_next_backend()
    url = f"{backend}/{path}"
    
    try:
        req = http_client.build_request(request.method, url, headers=headers, params=params, content=body)
        res = await http_client.send(req, stream=True)
        return StreamingResponse(res.aiter_raw(), status_code=res.status_code, headers=dict(res.headers))
    except Exception as e:
        return Response(content=f"Proxy Error: {str(e)}", status_code=502)

if __name__ == "__main__":
    # Start the proxy server
    uvicorn.run(app, host="0.0.0.0", port=args.proxy_port, log_level="info")
