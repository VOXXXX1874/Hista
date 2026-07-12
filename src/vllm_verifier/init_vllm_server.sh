#!/bin/bash

NUM_GPUS=${1:-8}
PROXY_PORT=${2:-8000}
GPU_MEMORY_UTILIZATION=${3:-0.2}
START_GPU=${4:-0}

if ! [[ "$NUM_GPUS" =~ ^[0-9]+$ ]]; then
    echo "❌ ERROR: The number of GPUs must be an integer. You provided: $NUM_GPUS"
    exit 1
fi

if ! [[ "$PROXY_PORT" =~ ^[0-9]+$ ]]; then
    echo "❌ ERROR: The proxy port must be an integer. You provided: $PROXY_PORT"
    exit 1
fi

if ! [[ "$START_GPU" =~ ^[0-9]+$ ]]; then
    echo "❌ ERROR: The start GPU must be a non-negative integer. You provided: $START_GPU"
    exit 1
fi

if [ "$NUM_GPUS" -lt 1 ] || [ "$NUM_GPUS" -gt 8 ]; then
    echo "❌ ERROR: The number of GPUs must be between 1 and 8. You provided: $NUM_GPUS"
    exit 1
fi

if [ $((START_GPU + NUM_GPUS)) -gt 8 ]; then
    echo "❌ ERROR: Starting from GPU $START_GPU with $NUM_GPUS GPUs exceeds the available GPU range 0-7"
    exit 1
fi

if [ "$PROXY_PORT" -lt 1 ] || [ "$PROXY_PORT" -gt 65535 ]; then
    echo "❌ ERROR: The proxy port must be between 1 and 65535. You provided: $PROXY_PORT"
    exit 1
fi

LAST_VLLM_PORT=$((PROXY_PORT + NUM_GPUS))
if [ "$LAST_VLLM_PORT" -gt 65535 ]; then
    echo "❌ ERROR: Proxy port $PROXY_PORT with $NUM_GPUS GPUs requires vLLM port $LAST_VLLM_PORT, exceeding 65535"
    exit 1
fi

echo "🚀 Initializing vLLM server on $NUM_GPUS GPUs"
echo "   - GPU range: $START_GPU-$((START_GPU + NUM_GPUS - 1))"
echo "   - Proxy port: $PROXY_PORT"
echo "   - vLLM backend ports: $((PROXY_PORT + 1))-$LAST_VLLM_PORT"

export no_proxy="localhost,127.0.0.1"
export NO_PROXY="localhost,127.0.0.1"
export VLLM_SERVER_DEV_MODE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False

for ((i=0; i<NUM_GPUS; i++)); do
  PORT=$((PROXY_PORT + 1 + i))
  GPU=$((START_GPU + i))
  echo "-> Starting vLLM verifier on GPU $GPU at port $PORT"
  
  CUDA_VISIBLE_DEVICES=$GPU vllm serve TIGER-Lab/general-verifier \
    --host 0.0.0.0 \
    --port $PORT \
    --enforce-eager \
    --trust-remote-code \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
    --enable-sleep-mode \
    --max-model-len 2048 \
    --max-num-seqs 32 > "tmp/vllm_gpu_${GPU}.log" 2>&1 &
done

# auto-detect the number of GPUs and start the proxy server
echo "Please initialize vLLM proxy server when the vllm is ready use command: python src/vllm_verifier/vllm_proxy.py --num-gpus $NUM_GPUS --proxy-port $PROXY_PORT &"
