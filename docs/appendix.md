# Appendix: Verifier and Sandbox Setup

This page describes auxiliary services required by GeneralQA and programming data.

## Verifier Setup

Since GeneralQA answers are short phrases that cannot be parsed or compared directly, we use a small language model as a verifier to judge model answers based on ground-truth answers.

We use vLLM to set up standalone verification instances and enable sleep mode to free GPU memory.

First set:

```bash
export no_proxy="localhost,127.0.0.1"
export NO_PROXY="localhost,127.0.0.1"
export VLLM_SERVER_DEV_MODE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export TOKENIZERS_PARALLELISM=false
```

Then use this script to set up multiple vLLM instances. The following assumes one node with 8 GPUs and port 8000 available:

```bash
bash src/vllm_verifier/init_vllm_server.sh
```

You can adjust the number of GPUs, port, and GPU memory utilization through the 1st, 2nd, and 3rd arguments:

```bash
bash src/vllm_verifier/init_vllm_server.sh 1 9000 0.2
```

Suppose the proxy port is `p` and the GPU number is `n`. The script creates one vLLM instance per GPU and listens on backend ports `p+1`, `p+2`, and so on.

To manage those instances, we create a proxy that routes requests. You will see output like:

```text
Initializing vLLM server on 8 GPUs
   - Proxy port: 8000
   - vLLM backend ports: 8001-8008
Please initialize vLLM proxy server when the vllm is ready use command: python src/vllm_verifier/vllm_proxy.py --num-gpus 8 --proxy-port 8000 &
```

Check vLLM instance availability in `tmp/vllm_gpu_${i}.log`. If all instances are available, create the proxy with the command in the last line:

```bash
python src/vllm_verifier/vllm_proxy.py --num-gpus 8 --proxy-port 8000 &
```

After the router is created, the vLLM instances will be put to sleep. Test them with:

```bash
python src/vllm_verifier/ping_vllm.py --base-url http://localhost:8000/v1
```

They will be put to sleep after testing.

If you are preparing to run training, make sure the instances are sleeping when training begins. If you specify a different port, remember to modify the corresponding `verifier_vllm_base_url`.

If you are preparing to run sampling in data processing or run evaluation, wake up the instances:

```bash
python src/vllm_verifier/wakeup_vllm.py --proxy-url http://localhost:8000
```

After finishing the task, kill all vLLM instances with:

```bash
pkill -9 -f vllm
pkill -9 -f VLLM
```

## Sandbox Setup

The correctness of programming problems is judged by whether the generated program can pass all test cases within a limited time. Therefore, generated programs must run in an isolated sandbox.

Our experiments were conducted on an HPC environment equipped with **Singularity**, and the code is well tested with it.

First create `singularity_images` to store images locally. We use an image with basic numerical computation libraries such as NumPy and SciPy:

```bash
singularity pull ./singularity_images/hista_programming.sif docker://quay.io/jupyter/scipy-notebook:latest
```

Then test whether you can run the container:

```bash
echo "print('Hello')" > tmp/execute_code.py

singularity exec \
  -C \
  --net --network none \
  --no-home \
  -B ./tmp:/workspace \
  --pwd /workspace \
  ./singularity_images/hista_programming.sif \
  python execute_code.py
```

If you use a different version and encounter errors, modify `src/rl/utils/rewards.py` accordingly.

Besides **Singularity**, we also test and support **Docker**. Set:

```bash
export CODE_SANDBOX_RUNTIME=docker
```

Then test whether you can run the container:

```bash
docker pull quay.io/jupyter/scipy-notebook:latest
echo "print('Hello')" > tmp/execute_code.py
docker run -it --rm \
  --network none \
  --read-only \
  --tmpfs /tmp \
  --cap-drop=ALL \
  --security-opt no-new-privileges \
  --pids-limit 256 \
  --memory 1G \
  --cpus 2 \
  -v "$(pwd)/tmp:/workspace:nosuid,nodev" \
  -w /workspace \
  -u "$(id -u):$(id -g)" \
  quay.io/jupyter/scipy-notebook:latest \
  python execute_code.py
```

If you use a different version and encounter errors, also modify `src/rl/utils/rewards.py` accordingly.

If your HPC or PC uses other software, such as `apptainer`, you need to implement the corresponding runtime yourself. It should be similar to the existing Singularity and Docker paths.
