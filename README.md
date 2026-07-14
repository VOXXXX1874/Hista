# Hista

Open source code for the paper **"Hista and Numca: Estimate State Value Effectively for Large Language Model Reinforcement Learning"**.

This repository contains code for:

- Reinforcement learning with GRPO, Hista, and Numca.
- Data preparation for math, science, general QA, and programming tasks.
- State Value Estimation Benchmark (SVEB).
- Training and evaluation recipes for math-only and hybrid datasets.

## Release status

The current release includes RL training implementations and reproduction recipes for the **1.5B** and **3B** model scales on both the **math-only** and **hybrid** datasets. These are the primary supported configurations for reproducing the paper's results and overall experimental trends.

Additional releases are planned:

- Math-only and hybrid RL training support for the **7B** and **14B** model scales will be released incrementally.
- We have also implemented and validated Hista and Numca on the **verl** training framework. The verl-based implementation and its reproduction configurations will be open-sourced in a future update.

Unless stated otherwise, the current repository uses the TRL-based implementation documented below; the planned 7B/14B additions and verl implementation are not included yet.

## Installation

### Conda & pip

The code requires CUDA >= 12.2 and < 13.0. First create the environment:

```bash
conda create -n Hista python=3.12
conda activate Hista
```

Then install the dependencies:

```bash
pip install -e .
```

Finally, install flash attention. If compilation fails, we recommend downloading a compiled wheel from <https://github.com/Dao-AILab/flash-attention/releases?page=3>. For example:

```bash
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl

pip install flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
```

### Singularity & Docker

Coming soon.

### Required folders

Please create the following folders in the root directory of this repository:

```text
| -- Hista
|   | -- data
|   | -- output
|       | -- Qwen2.5-1.5B
|       | -- Qwen2.5-3B
|       | -- Qwen2.5-7B
|       ...
|   | -- tmp
```

- `data`: Stores final data for training and evaluation.
- `output`: Stores training logs, benchmark results, and checkpoints. Create the corresponding model folder before training.
- `tmp`: Stores temporary logs, such as standalone vLLM server logs.

## Documentation

The full reproduction guide is split by workflow:

- [Data preparation](docs/data_preparation.md): Download original or processed data, align formats, sample rollouts, select data by difficulty, and construct SVEB or hybrid training data.
- [State Value Estimation Benchmark](docs/sveb.md): Run SVEB generation and reuse pipelines.
- [Training](docs/training.md): Train on math-only and hybrid datasets with the provided scripts and recipes.
- [Evaluation](docs/evaluation.md): Deliver predefined evaluation configs and run benchmark evaluation.
- [Appendix: verifier and sandbox setup](docs/appendix.md): Set up the vLLM verifier and code execution sandbox required by hybrid dataset training and evaluation.

## Quick paths

- Fastest math-only reproduction: follow [Data preparation](docs/data_preparation.md#fastest-path-math-only), then [Training](docs/training.md#choose-an-entry-point-and-recipe).
- SVEB reproduction: prepare SVEB data in [Data preparation](docs/data_preparation.md), then follow [SVEB](docs/sveb.md).
- Hybrid training reproduction: prepare benchmarks and hybrid data in [Data preparation](docs/data_preparation.md), set up services in [Appendix](docs/appendix.md), then follow [Hybrid training](docs/training.md#hybrid-training).
