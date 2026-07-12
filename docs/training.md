# Training

This repository trains GRPO, Hista, and Numca policies with the entry points in `src/rl` and YAML recipes under `recipes`. The shell scripts in `scripts/train` collect representative launch commands; a recipe contains the model, data, algorithm, rollout, evaluation, checkpoint, and logging settings for one run.

## Choose an entry point and recipe

The three policy-training entry points share the same data preparation, reward functions, vLLM rollout backend, and most training arguments:

| Entry point | Trainer | Purpose |
| --- | --- | --- |
| `src/rl/grpo.py` | `GRPOPlusTrainer` | GRPO and DAPO/CSIPO baselines |
| `src/rl/hista.py` | `HistaTrainer` | Hista training |
| `src/rl/numca.py` | `NumcaTrainer` | Numca training |

Recipes are grouped by suggested GPU memory (`24G` or `80G`), model, and data type (`math` or `hybrid`). The memory label is a starting point rather than a hard requirement: sequence length, batch size, model size, vLLM utilization, GPU count, and ZeRO settings all affect actual memory use.

For example, the following launches the math-only DAPO baseline on four GPUs:

```bash
ACCELERATE_LOG_LEVEL=info \
accelerate launch \
  --config_file recipes/zero3.yaml \
  --num_processes 4 \
  --main_process_port 29502 \
  src/rl/grpo.py \
  --config recipes/24G/Qwen2.5-1.5B/math/GRPO_base_dapo.yaml \
  > output/Qwen2.5-1.5B/GRPO_base_dapo_sampling.log 2>&1
```

- `recipes/zero3.yaml` configures Accelerate and DeepSpeed ZeRO-3.
- `--num_processes` is normally the number of training GPUs.
- Each job on the same node must use a different `--main_process_port`.
- Command-line arguments after the recipe override the corresponding YAML values. This is useful for changing an output directory or verifier URL without copying a recipe.

The trainer detects the latest checkpoint in `output_dir` and resumes from it unless `resume_from_checkpoint` is set explicitly. Use a new or empty `output_dir` when the run should start from the original model.

## Reference training scripts

The shell scripts under `scripts/train` are the recommended starting point for reproducing the paper's experiments. They group together the launch commands for different algorithms under comparable hardware, model, and dataset settings:

```text
scripts/train/
├── 24G/
│   ├── math/
│   │   ├── train_qwen2.5-1.5B.sh
│   │   └── ...
│   ├── hybrid/
│   │   ├── train_qwen2.5-3B.sh
│   │   └── ...
│   └── ppo_sft/
│       ├── train_qwen2.5-1.5B.sh
│       └── ...
└── 80G/
    ├── math/
    │   ├── train_qwen2.5-1.5B.sh
    │   ├── train_qwen2.5-3B.sh
    │   └── ...
    └── hybrid/
        ├── train_qwen2.5-1.5B.sh
        ├── train_qwen2.5-3B.sh
        └── ...
```

- `24G` and `80G` identify the GPU-memory class for which the commands and recipes were prepared.
- `math` trains on the math-only dataset, whereas `hybrid` uses the model-specific mixture of math, GeneralQA, science, and programming data.
- `ppo_sft` trains the critic models required by the PPO statistical-estimation workflow described in [SVEB](sveb.md).
- A model-specific shell script collects the GRPO, Hista, Numca, DAPO, and CSIPO variants available for that model. Section labels such as `## dapo` and `## dapo + hista` identify the experiment represented by each command.
- Every policy command selects a matching YAML file under `recipes/<memory>/<model>/<data type>/`. Most experimental choices should be changed in that recipe; launcher-level choices such as GPU count, communication port, verifier URL, and log path remain in the shell script.

For example, a script block of the following form maps the algorithm entry point to one concrete recipe:

```bash
## dapo + hista

ACCELERATE_LOG_LEVEL=info \
accelerate launch \
  --config_file recipes/zero3.yaml \
  --main_process_port 29501 \
  --num_processes 8 \
  src/rl/hista.py \
  --config recipes/24G/Qwen2.5-3B/hybrid/Hista_inst_dapo.yaml \
  --verifier_vllm_base_url http://localhost:8000/v1 \
  > output/Qwen2.5-3B/Hista_inst_hybrid_dapo_sampling.log 2>&1
```

These reference scripts are intended to reproduce the paper's results and, in particular, its overall comparisons and trends. Exact numbers may still vary with random seeds, GPU type and count, CUDA/vLLM kernels, dependency versions, verifier responses, and dataset or model revisions. For the closest comparison, keep the provided recipe unchanged, use the indicated memory class and GPU count, prepare the exact corresponding dataset, and evaluate all checkpoints with the same evaluation configuration.

Commands in one shell file are independent experiment launches rather than stages of a single pipeline. Run the desired labeled block instead of executing every block blindly: some reference files contain background commands, and concurrent runs must not share an output directory, distributed port, or verifier proxy lifecycle.

## Data and reward configuration

`dataset_name` points to a local Hugging Face dataset directory. The training split is prepared by `src/rl/utils/prepare_dataset.py`, which constructs the conversational prompt and preserves fields such as `solution`, `problem`, and `verifier` for reward calculation.

A typical math recipe uses:

```yaml
dataset_name: data/MATH
reward_funcs: [accuracy]
reward_weights: [1.0]
silence: true
```

Math samples normally use the local symbolic/default verifier. Hybrid data may additionally contain `general` samples, which call the external language-model verifier, and `code` or `code_*` samples, which execute tests in the configured sandbox. Prepare those services as described in [Verifier and Sandbox Setup](appendix.md).

## DAPO dynamic sampling

`dynamic_sampling_scale` is a repository-specific extension; it is not implemented by the original TRL GRPO trainer. It was added here to support DAPO-style dynamic sampling.

```yaml
loss_type: dapo
epsilon_high: 0.28
dynamic_sampling_scale: 2
num_generations: 32
```

For every generation batch, the customized trainer does the following:

1. It samples `dynamic_sampling_scale` times the normal number of prompts.
2. It generates `num_generations` completions for every prompt and calculates their rewards.
3. It computes the mean `accuracy_reward` of each prompt group.
4. It ranks prompt groups by `abs(mean_accuracy - 0.5)` and retains the normal number of groups closest to `0.5`.
5. It sends all completions belonging to the retained groups into the policy update.

Thus, a scale of `2` generates twice as many prompt groups but keeps the original optimization batch size. It favors groups with mixed correct and incorrect answers, including graded accuracies close to 0.5; it does not independently retain the highest-reward completions. A larger scale provides a larger candidate pool at the cost of approximately proportional rollout and verification work.

Current constraints are important:

- `dynamic_sampling_scale` must be at least `1`.
- `(generation_batch_size / num_generations) * dynamic_sampling_scale` must be an integer.
- The configured reward functions must include `accuracy`, because selection explicitly looks for `accuracy_reward`.
- Dynamic sampling is supported only with colocated vLLM; `vllm_mode: server` raises `NotImplementedError` when the scale is not `1`.

Set `dynamic_sampling_scale: 1` to disable oversampling and selection while retaining the same trainer.

## Batch configuration

The most relevant batch parameters are:

```yaml
per_device_train_batch_size: 2
gradient_accumulation_steps: 16
num_generations: 32
num_iterations: 4
dynamic_sampling_scale: 2
```

- `per_device_train_batch_size` is the number of completion samples processed by each training process in one micro-batch.
- `gradient_accumulation_steps` is the number of micro-steps accumulated before an optimizer step. Increasing it raises the effective optimization batch without raising model forward/backward memory as much, but retains rollout data longer and slows optimizer updates.
- `num_generations` is the number of completions generated as one group for each prompt. GRPO-style advantages are normalized within this group, so it must divide the resolved global `generation_batch_size`.
- `num_iterations` controls how many policy-update passes reuse a generated batch. Larger values extract more updates from each expensive rollout but make the samples more off-policy.
- `dynamic_sampling_scale` changes the candidate rollout batch, not the number of retained samples used for optimization.

Ignoring optional TRL overrides such as `generation_batch_size` and `steps_per_generation`, the usual effective optimization batch is approximately:

```text
per_device_train_batch_size × number_of_processes × gradient_accumulation_steps
```

The number of retained prompt groups is the resolved `generation_batch_size / num_generations`; dynamic sampling initially draws that many groups multiplied by `dynamic_sampling_scale`. Check the resolved arguments printed near the start of the log after changing GPU count or batch settings. In particular, keep `generation_batch_size` divisible by both `num_generations` and the number of processes.

For out-of-memory errors, first lower `per_device_train_batch_size`, `max_completion_length`, or `vllm_gpu_memory_utilization`; compensate with `gradient_accumulation_steps` if the effective batch should remain similar. Lowering `num_generations` also saves rollout memory and time, but changes the group statistics used by the algorithm.

## Policy vLLM configuration

Policy rollouts currently require vLLM; the customized trainer asserts `use_vllm: true`. Reference recipes colocate vLLM with training:

```yaml
use_vllm: true
vllm_mode: colocate
vllm_tensor_parallel_size: 1
vllm_gpu_memory_utilization: 0.55
vllm_enable_sleep_mode: true
temperature: 0.7
max_completion_length: 4096
```

- `vllm_mode: colocate` loads the rollout engine on the training workers and is required for dynamic sampling.
- `vllm_tensor_parallel_size` controls how many GPUs form one vLLM tensor-parallel group. The reference recipes use one GPU per engine.
- `vllm_gpu_memory_utilization` reserves a fraction of GPU memory for the vLLM executor. Raise it cautiously if KV-cache capacity is insufficient; lower it if model training and vLLM contend for memory.
- `vllm_enable_sleep_mode` lets the colocated engine release memory while the policy performs forward/backward work.
- `temperature` affects rollout diversity, while `max_completion_length` is a major determinant of rollout time and KV-cache memory.

These settings control policy generation and are separate from the external verifier vLLM described below.

## Evaluation and checkpoint saving

Training-time evaluation is configured as follows in the reference recipes:

```yaml
do_eval: true
eval_strategy: steps
eval_steps: 40
num_generations_eval: 1
quick_eval_dataset: data/MATH
quick_eval_dataset_size: 4000
per_device_eval_batch_size: 1000

save_strategy: best
metric_for_best_model: rewards/mean
greater_is_better: true
save_only_model: true
save_total_limit: 1
```

`quick_eval_dataset` supplies the evaluation split. If it is larger than `quick_eval_dataset_size`, the first requested number of samples are selected; if it is shorter, the dataset is repeated and truncated to exactly that size. Evaluation runs every `eval_steps` optimizer steps when `eval_strategy: steps` is used.

This customized trainer evaluates the entire quick-eval set in one distributed batch. The following equality is mandatory:

```text
quick_eval_dataset_size × num_generations_eval
    = per_device_eval_batch_size × number_of_processes
```

For example, 4,000 evaluation prompts, one completion per prompt, and four processes require `per_device_eval_batch_size: 1000`. A mismatch raises `ValueError` during trainer initialization. Keeping `num_generations_eval: 1` makes evaluation cheaper, but it measures one stochastic completion per prompt.

With `save_strategy: best`, a checkpoint is saved when `metric_for_best_model` improves according to `greater_is_better`. The reference recipes use `rewards/mean`, so a larger mean evaluation reward replaces the current best checkpoint. `save_total_limit: 1` retains only one checkpoint, and `save_only_model: true` omits optimizer, scheduler, and RNG state from those checkpoints. That saves disk space, but a model-only checkpoint cannot faithfully resume training. Remove `save_only_model: true` if resumability is required.

The entry point also saves the final model and trainer state to `output_dir` after training completes.

## Hybrid training and verifier vLLM

Hybrid training requires the processed model-specific dataset, for example:

```text
data/
└── 3B/
    └── hybrid/
        ├── train.json
        └── test.json
```

Start the verifier backends and proxy using [Verifier Setup](appendix.md#verifier-setup), then ensure the verifier is sleeping before policy training begins. A hybrid recipe should contain:

```yaml
manage_verifier_vllm_sleep: true
verifier_vllm_base_url: http://localhost:8000/v1
verifier_vllm_sleep_level: 1
verifier_vllm_control_timeout: 120
```

When management is enabled, only the main training process sends proxy control requests. All processes synchronize around reward calculation; the trainer wakes the verifier, computes rewards, puts it back to sleep in a `finally` block, and then continues policy optimization. This prevents the verifier and colocated policy vLLM from occupying their peak memory at the same time. A failed wake/sleep request aborts the run rather than silently continuing.

`verifier_vllm_base_url` serves two purposes: the trainer derives the OpenAI-compatible reward endpoint from it, and it sends wake/sleep control requests to the same proxy. Both `http://localhost:8000` and `http://localhost:8000/v1` are normalized for reward requests, but the URL must identify the proxy port rather than an individual backend port. The shell scripts may override this value on the command line:

```bash
src/rl/grpo.py \
  --config recipes/24G/Qwen2.5-3B/hybrid/GRPO_inst_dapo.yaml \
  --verifier_vllm_base_url http://localhost:9000/v1
```

Use a dedicated verifier proxy for a training job unless you have coordinated its lifecycle explicitly. Two jobs sharing one proxy can race: one trainer may put the verifier to sleep while the other is calculating rewards. The reference `scripts/train/80G/hybrid/train_qwen2.5-3B.sh` uses ports 8000 and 9000 for different commands, illustrating the dedicated-proxy setup.

Programming samples also require the sandbox independently of verifier vLLM. Verify both services before starting a hybrid run.

## Pre-flight checklist

Before launching a long run, check that:

1. `dataset_name` and `quick_eval_dataset` contain the expected `train` and `test` splits.
2. `output_dir` is new, or resuming from its latest checkpoint is intentional.
3. `num_generations` divides the resolved generation batch and the dynamic-sampling constraints hold.
4. The evaluation batch equality above matches the selected GPU count.
5. Policy vLLM fits alongside the training model at the configured memory utilization.
6. For hybrid data, the verifier proxy is reachable and sleeping, and the code sandbox works.
7. Concurrent jobs use different distributed ports, output directories, and verifier proxies.
