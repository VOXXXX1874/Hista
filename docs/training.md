# Training

This repository provides scripts and recipes for training on math-only and hybrid datasets.

## Training on Math Dataset

After all math data and benchmarks are prepared, use scripts under `scripts/train` to start training, or create your own script based on them.

The structure under `scripts/train` is:

```text
| -- scripts/train
|   | -- 24G
|       | -- math
|           | -- train_qwen2.5-1.5B.sh
|           | -- train_qwen3-0.6B.sh
|           | -- ...
|       | -- hybrid
|           | -- train_qwen2.5-1.5B.sh
|           | -- train_qwen3-0.6B.sh
|           | -- ...
|   | -- 80G
|       | -- math
|           | -- train_qwen2.5-3B.sh
|           | -- train_qwen3-1.7B.sh
|           | -- ...
|       | -- hybrid
|           | -- train_qwen2.5-3B.sh
|           | -- train_qwen3-1.7B.sh
|           | -- ...
```

`24G` and `80G` indicate the suggested GPU memory level. `math` and `hybrid` correspond to training on the math dataset and hybrid dataset. Available commands for one model are collected and labeled in each file.

An example training command is:

```bash
ACCELERATE_LOG_LEVEL=info \
accelerate launch \
--config_file recipes/zero3.yaml \
--num_processes=4 \
--main_process_port 29502 \
src/rl/grpo.py \
--config recipes/24G/Qwen2.5-1.5B/math/GRPO_base_dapo.yaml \
> ./output/Qwen2.5-1.5B/GRPO_base_dapo_sampling.log 2>&1
```

Arguments:

1. `config_file`: DeepSpeed configuration. This example uses DeepSpeed ZeRO-3 as the distributed training strategy.
2. `num_processes`: Number of GPUs allocated to training. This example uses 4 GPUs.
3. `main_process_port`: Main process port for communication between processes. If multiple training jobs run on one node, they must use different ports.
4. `config`: Training configuration, including algorithm, model, and data. This example runs DAPO on Qwen2.5-1.5B.

You can modify the commands accordingly. Each command corresponds to one recipe in `recipes`, which follows the same file structure.

## Training on Hybrid Dataset

After downloading or processing the hybrid data, make sure it is placed in the expected structure:

```text
| -- data
|   | -- MATH500
|   ...
|   | -- 1.5B
|       | -- hybrid
|           | -- train.json
|           | -- test.json
|       | -- SVEB_NUMBER
|           | -- train.json
|       ...
|   | -- 3B
|   ...
```

Because the hybrid dataset contains general QA and programming data, you need to set up the vLLM verifier and sandbox. See [Verifier Setup](appendix.md#verifier-setup) and [Sandbox Setup](appendix.md#sandbox-setup).

Make sure the vLLM verifier is sleeping before starting training.

Training command placement and recipes are similar to [Training on Math Dataset](#training-on-math-dataset). An example command is:

```bash
ACCELERATE_LOG_LEVEL=info \
accelerate launch \
--config_file recipes/zero3.yaml \
--num_processes=8 \
--main_process_port 29502 \
src/rl/grpo.py \
--config recipes/24G/Qwen2.5-1.5B/hybrid/GRPO_dapo_hybrid.yaml \
> ./output/Qwen2.5-1.5B/GRPO_dapo_hybrid_sampling.log 2>&1
```

Compared with the math-only command, this changes `config` and increases the number of GPUs to 8.

To save GPU memory, we wake the verifier only during reward calculation. If there are multiple training tasks, verifier management becomes difficult, so we suggest using all GPUs for one task. It is possible to use one verifier for multiple training tasks, or different verifiers for different training tasks, but that requires reading and understanding the relevant code.
