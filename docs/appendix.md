# Appendix: Verifier and Sandbox Setup

Math answers are graded locally, but hybrid data introduces two external execution paths:

- OpenR1, ScienceQA, and GeneralQA answers are judged by `TIGER-Lab/general-verifier`, served through vLLM.
- Generated programs are executed inside an isolated Singularity or Docker container.

These services are used by data sampling, policy training, and final evaluation. Set them up before working with samples whose `verifier` contains `general` or `code`.

## GeneralQA verifier architecture

The verifier consists of one vLLM backend per GPU and one load-balancing proxy:

```text
reward client
    |
    | OpenAI-compatible request: http://localhost:8000/v1/completions
    | control request:           http://localhost:8000/{wake_up,sleep,is_sleeping}
    v
proxy :8000
    ├── round robin -> backend on GPU 0 :8001
    ├── round robin -> backend on GPU 1 :8002
    └── round robin -> backend on GPU 2 :8003
```

Inference requests are routed to one backend in round-robin order. Lifecycle requests are broadcast to every backend. The proxy returns HTTP 502 if any backend fails a broadcast control request.

Sleep mode allows verifier GPUs to be reused for policy rollout or training:

- Level 1 offloads weights to CPU memory and clears the KV cache. This is the default and can wake up again without reloading the model from disk.
- Level 2 fully releases GPU memory without keeping a CPU backup. The repository's normal workflow uses level 1.

## Start the verifier service

Run all commands from the repository root. First create the log directory and configure local traffic not to use an HTTP proxy:

```bash
mkdir -p tmp

export no_proxy="localhost,127.0.0.1"
export NO_PROXY="localhost,127.0.0.1"
export VLLM_SERVER_DEV_MODE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export TOKENIZERS_PARALLELISM=false
```

`VLLM_SERVER_DEV_MODE=1` is required for the vLLM sleep/wake control endpoints. The initialization script starts the backends but does not create `tmp`; without the `mkdir` above, shell log redirection fails before `vllm serve` starts.

Start one verifier backend on each of eight GPUs, reserving 20% of each GPU's memory:

```bash
bash src/vllm_verifier/init_vllm_server.sh 8 8000 0.2 0
```

The positional arguments are:

| Position | Meaning | Default |
| --- | --- | --- |
| 1 | Number of one-GPU vLLM backends | `8` |
| 2 | Proxy port | `8000` |
| 3 | vLLM GPU memory utilization per backend | `0.2` |
| 4 | First physical GPU index | `0` |

For example, the following uses physical GPUs 4 and 5, proxy port 9000, and backend ports 9001 and 9002:

```bash
bash src/vllm_verifier/init_vllm_server.sh 2 9000 0.25 4
```

The script currently validates against an eight-GPU node: `NUM_GPUS` must be between 1 and 8, and `START_GPU + NUM_GPUS` cannot exceed 8. Modify its validation if the node exposes a different topology.

Each backend runs:

```text
TIGER-Lab/general-verifier
tensor parallel size: 1
maximum model length: 2048
maximum concurrent sequences: 32
sleep mode: enabled
```

Backend stdout and stderr are written to `tmp/vllm_gpu_<physical GPU>.log`. Watch the selected logs until the servers are ready:

```bash
tail -f tmp/vllm_gpu_0.log
```

Then start the proxy with the same backend count and proxy port:

```bash
python src/vllm_verifier/vllm_proxy.py \
  --num-gpus 8 \
  --proxy-port 8000 \
  > tmp/vllm_proxy_8000.log 2>&1 &
```

On startup, the proxy probes each backend for up to approximately 30 seconds and tries to put it into level-1 sleep. This initialization runs in the background, so the proxy port may become available before all backends have finished loading. Check both proxy and backend logs before continuing.

## Test and control the verifier

Run the provided load test:

```bash
python src/vllm_verifier/ping_vllm.py \
  --base-url http://localhost:8000/v1
```

It wakes all backends, sends 100 completion requests with concurrency 16, reports the success count and wall time, and then requests level-1 sleep. A healthy result should report `100 / 100 succeeded`.

To wake a sleeping verifier without running the load test:

```bash
python src/vllm_verifier/wakeup_vllm.py \
  --proxy-url http://localhost:8000
```

Control URLs use the proxy root without `/v1`. Inference clients use the OpenAI-compatible URL with `/v1`.

You can inspect or change state directly:

```bash
curl http://localhost:8000/is_sleeping
curl -X POST http://localhost:8000/wake_up
curl -X POST "http://localhost:8000/sleep?level=1"
```

The proxy broadcasts these requests and returns a `details` entry for every backend. Do not rely only on HTTP 200 from an individual vLLM backend; use the proxy response to confirm that the entire group reached the requested state.

## Connect repository workflows to the verifier

The reward client recognizes these environment variables:

```bash
export VERIFIER_BASE_URL=http://localhost:8000/v1
export VERIFIER_MODEL=TIGER-Lab/general-verifier
export VERIFIER_MAX_CONCURRENCY=100
export VERIFIER_MAX_TOKENS=1024
```

- `VERIFIER_BASE_URL` must be the proxy's OpenAI-compatible `/v1` URL. Its default is `http://localhost:8000/v1`.
- `VERIFIER_MODEL` is the model name sent in completion requests.
- `VERIFIER_MAX_CONCURRENCY` limits concurrent GeneralQA and code reward tasks; GeneralQA requests within that pool call the proxy.
- `VERIFIER_MAX_TOKENS` limits the verifier response length.

Positive integers are required for the concurrency and token settings; invalid or non-positive values silently fall back to their defaults. Set environment variables before importing or launching the Python entry point.

Lifecycle handling differs by workflow:

| Workflow | Verifier lifecycle |
| --- | --- |
| Hybrid policy training | With `manage_verifier_vllm_sleep: true`, the trainer wakes before reward calculation and sleeps afterward |
| `extra_sampling.py` | Attempts to manage sleep/wake through `--proxy_url` (default `http://localhost:8000`) around generation and reward calculation |
| Final `evaluation.py` | Does not manage lifecycle; wake the verifier before evaluation and sleep it afterward |
| Math-only work | Does not need the external verifier |

For training, `verifier_vllm_base_url` in the recipe or CLI is normalized and also used to set the reward client's endpoint. Start with the verifier asleep so its GPU memory is available to the colocated policy vLLM. Only one job should manage a proxy's lifecycle; otherwise one job can sleep the verifier while another is using it.

`extra_sampling.py` treats lifecycle failures as non-fatal so that math-only and code-only sampling can proceed without the external verifier. For GeneralQA data, however, the proxy must be available: reward-client failures are caught per sample and result in reward zero.

For final evaluation, use:

```bash
python src/vllm_verifier/wakeup_vllm.py --proxy-url http://localhost:8000

export VERIFIER_BASE_URL=http://localhost:8000/v1
python src/evaluation/evaluation.py ...

curl -X POST "http://localhost:8000/sleep?level=1"
```

## Stop the verifier

The backend and proxy are independent background processes. Stop the proxy and only the backend processes belonging to this service. On a dedicated node, the broad cleanup used during development is:

```bash
pkill -f "src/vllm_verifier/vllm_proxy.py"
pkill -f "vllm serve TIGER-Lab/general-verifier"
```

Inspect matching processes with `pgrep -af` first on shared machines. Avoid broad commands such as `pkill -f vllm`, because they can terminate unrelated policy rollout or other users' vLLM jobs.

## Verifier troubleshooting

### A backend never becomes ready

Check `tmp/vllm_gpu_<GPU>.log` for model download, CUDA, port, or memory errors. Confirm that the chosen physical GPU indices exist, the backend ports are free, and `tmp` existed before launching. Increase or decrease `GPU_MEMORY_UTILIZATION` according to available memory.

### Wake or sleep returns HTTP 502

Inspect the proxy response's `details` list to find the failing backend. For sleep, the proxy retries each backend up to three times and verifies `/is_sleeping`; persistent 502 therefore usually means that the backend is unavailable or did not actually enter sleep mode.

### Requests bypass localhost or time out

Set both lowercase and uppercase `no_proxy` variables. The reward and ping clients explicitly disable inherited proxy settings, but shell tools and other clients may not.

### Training fails during verifier control

Check that `verifier_vllm_base_url` points to the proxy rather than an individual backend, and that the port matches the proxy process. Training intentionally raises an error when any wake/sleep request fails, because continuing would produce incomplete rewards or unsafe memory overlap.

## Programming sandbox behavior

Programming answers are extracted from the last fenced Python block, an `<answer>` block, or finally the entire response. The extracted code is written to a fresh temporary directory and executed without network access. The directory is deleted after every run, including failures and timeouts.

There are two test formats:

- For verifiers containing `humaneval`, generated code replaces `FLAGTOBEREPLACED` in the provided assertion program, and that complete program must exit successfully.
- Other code verifiers parse a list of input/output test cases, randomly sample at most five, run the generated program separately on each input, and compare stripped stdout exactly with the expected output.

The random test is for avoiding extremely long reward calculation time during RL training on program.

Sandbox settings are controlled by:

```bash
export CODE_SANDBOX_RUNTIME=singularity
export CODE_SANDBOX_IMAGE="$(pwd)/singularity_images/hista_programming.sif"
export CODE_SANDBOX_TMP_ROOT="$(pwd)/tmp"
export CODE_TEST_TIMEOUT=20
export LOCAL_TEST_MAX_CONCURRENCY=8
```

`CODE_SANDBOX_RUNTIME` supports only `singularity` and `docker`. `CODE_TEST_TIMEOUT` applies to each container execution, and `LOCAL_TEST_MAX_CONCURRENCY` bounds simultaneous local test executions. Invalid non-positive integer values fall back to their defaults. These settings are resolved when `src/rl/utils/rewards.py` is imported, so export them before starting training, sampling, or evaluation.

## Singularity setup

Singularity is the default and was used for the paper's HPC experiments. Build the image from the repository root:

```bash
mkdir -p singularity_images tmp

singularity pull \
  singularity_images/hista_programming.sif \
  docker://quay.io/jupyter/scipy-notebook:latest
```

Test the same isolation pattern used by the reward code:

```bash
echo "print('Hello')" > tmp/executed_code.py

singularity exec \
  -C \
  --net --network none \
  --no-home \
  -B "$(pwd)/tmp:/workspace" \
  --pwd /workspace \
  "$(pwd)/singularity_images/hista_programming.sif" \
  python executed_code.py
```

Expected output:

```text
Hello
```

The production path binds only a newly created per-execution directory, not the entire shared `tmp` directory. `-C`, `--no-home`, and network isolation prevent generated code from seeing the host home directory or network.

If the image is stored elsewhere, set `CODE_SANDBOX_IMAGE` to its absolute `.sif` path. The default is `$(pwd)/singularity_images/hista_programming.sif`, so launching from outside the repository root changes the resolved default.

## Docker setup

Pull the default image and select Docker before launching Python:

```bash
docker pull quay.io/jupyter/scipy-notebook:latest

export CODE_SANDBOX_RUNTIME=docker
export CODE_SANDBOX_IMAGE=quay.io/jupyter/scipy-notebook:latest
mkdir -p tmp
```

Test the relevant restrictions:

```bash
echo "print('Hello')" > tmp/executed_code.py

docker run --rm \
  --network none \
  --read-only \
  --tmpfs /tmp \
  --cap-drop=ALL \
  --security-opt no-new-privileges \
  --pids-limit 256 \
  --memory 1G \
  --cpus 2 \
  -v "$(pwd)/tmp:/workspace" \
  -w /workspace \
  -u "$(id -u):$(id -g)" \
  quay.io/jupyter/scipy-notebook:latest \
  python executed_code.py
```

The reward implementation uses the same restrictions and adds `-i` only when a test case provides stdin. The container root filesystem is read-only, `/tmp` is an in-memory writable filesystem, all Linux capabilities are dropped, and CPU, memory, and process counts are limited.

The user running evaluation must have permission to access the Docker daemon. If rootless Docker uses different volume ownership behavior, verify that the container can read `executed_code.py` under the host user's UID and GID.

## Sandbox troubleshooting and extensions

- A timeout, missing runtime executable, or container launch `OSError` is treated as a failed test and receives reward zero.
- A non-zero program exit code, stderr-producing crash, or stdout mismatch also fails the test. Stderr itself is captured but not compared when the program exits successfully.
- If imports fail, rebuild or replace the image with the required packages and set `CODE_SANDBOX_IMAGE`; avoid weakening network or host-directory isolation.
- `CODE_SANDBOX_TMP_ROOT` must exist on a filesystem where the process can create and delete directories. The code creates the root if necessary.
- Apptainer and other runtimes are not currently recognized. Supporting one requires adding a new execution path in `src/rl/utils/rewards.py` with equivalent filesystem, network, resource, and timeout isolation.

Before using programming data in a long run, test both a successful program and a deliberate timeout through the selected container runtime, then keep the same runtime and image across all compared experiments.
