# Evaluation

Training-time quick evaluation is used to select the best checkpoint, while `src/evaluation/evaluation.py` performs the final benchmark evaluation reported across models and methods. The latter loads a model once with vLLM, evaluates one or more processed benchmark datasets in sequence, calculates correctness and format rewards, and writes per-completion results and a summary log for every benchmark.

## Evaluation layout

The relevant files are organized as follows:

```text
src/evaluation/
├── evaluation.py
└── predefined_config/
    ├── MATH-500/
    │   └── evaluation_config.json
    ├── GSM8K/
    │   └── evaluation_config.json
    ├── AIME2425/
    │   └── evaluation_config.json
    └── ...

scripts/evaluation/
└── deliver_config.sh

data/
├── MATH-500/
│   ├── test.json
│   └── .evaluation_config/
│       └── evaluation_config.json
└── ...
```

`predefined_config` stores benchmark-specific generation settings. `deliver_config.sh` copies each configuration into the corresponding processed dataset directory, and `evaluation.py` reads the delivered copy rather than reading `src/evaluation/predefined_config` directly.

## Prepare benchmark data and configs

First download and process the benchmarks as described in [Data Preparation](data_preparation.md). Each dataset passed to the evaluator must be a local Hugging Face dataset with a `test` split and at least `problem` and `solution` fields. A `verifier` field selects specialized grading for GeneralQA and programming benchmarks; `process` is optional and is only preserved in the result record.

Then deliver all predefined configurations:

```bash
bash scripts/evaluation/deliver_config.sh
```

For every directory under `src/evaluation/predefined_config`, the script:

1. Finds `evaluation_config.json`.
2. Requires a same-named directory under `data`.
3. Creates `data/<benchmark>/.evaluation_config/`.
4. Copies the JSON file to that directory.

For example:

```text
src/evaluation/predefined_config/AIME2425/evaluation_config.json
    -> data/AIME2425/.evaluation_config/evaluation_config.json
```

The script uses strict error handling and stops if any source config or matching target dataset directory is missing. Therefore, process all listed benchmarks first, or manually copy only the configs for the subset you intend to evaluate.

If a predefined config is later edited, run `deliver_config.sh` again so the copy under `data` does not become stale.

## Run math benchmark evaluation

The following evaluates a Qwen2.5 model on several math benchmarks:

```bash
python src/evaluation/evaluation.py \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --output_folder output/benchmark_hybrid_instruct \
  --datasets \
    data/MATH-500 \
    data/GSM8K \
    data/OlympiadBench \
    data/amc23 \
    data/AIME2425 \
    data/GaoKao \
    data/CollegeMath \
  --evaluation_config qwen2.5 \
  --num_gpus 1
```

To evaluate a trained policy, replace `model_name` with its checkpoint directory:

```bash
python src/evaluation/evaluation.py \
  --model_name output/Qwen2.5-1.5B/GRPO-1.5B-base-dapo \
  --output_folder output/evaluation/GRPO-1.5B-base-dapo \
  --datasets data/MATH-500 data/AIME2425 \
  --evaluation_config qwen2.5 \
  --num_gpus 1
```

Arguments:

- `model_name` accepts either a Hugging Face model ID or a local model/checkpoint directory.
- `output_folder` receives one log and one JSON result file per benchmark. Existing files with the same names are overwritten.
- `datasets` is a space-separated list of processed local dataset directories. They are evaluated sequentially with the same loaded model.
- `evaluation_config` selects one named object from each dataset's delivered config. The CLI accepts `qwen2.5`, `qwen3`, and `r1-distill`.
- `num_gpus` becomes vLLM's `tensor_parallel_size`. Increase it when the model cannot fit on one GPU; it is not data-parallel evaluation.
- `enable_thinking` is passed to the tokenizer's chat template and is mainly intended for Qwen3. It defaults to `false`.

The evaluator currently initializes vLLM with `bfloat16`, `gpu_memory_utilization=0.6`, `trust_remote_code=True`, and the multiprocessing distributed executor. These values are hard-coded in `evaluation.py`; edit the source if a model or platform requires different settings.

## Select an evaluation config

Every benchmark config may define up to three model-family presets:

- `qwen2.5`: standard Qwen2.5/Qwen2.5-Instruct evaluation.
- `qwen3`: longer outputs for Qwen3; normally combine it with `--enable_thinking true` when evaluating the thinking mode.
- `r1-distill`: uses the model's default user-only prompt style and is intended for R1-distilled models.

The selected preset contains five required fields:

```json
{
  "qwen2.5": {
    "max_output_tokens": 4096,
    "use_default_system_prompt": false,
    "temperature": 0.0,
    "num_generations": 1,
    "reward_function": "eval_answer_reward"
  }
}
```

| Field | Effect |
| --- | --- |
| `max_output_tokens` | Maximum number of newly generated tokens per completion |
| `use_default_system_prompt` | If `true`, uses a user-only prompt asking for step-by-step reasoning and a boxed answer; otherwise uses this repository's system prompt |
| `temperature` | vLLM sampling temperature |
| `num_generations` | Number of completions sampled for each problem |
| `reward_function` | Grading function; currently only `eval_answer_reward` is supported |

Generation settings intentionally differ by benchmark. Deterministic or large benchmarks commonly use one completion with temperature zero, while smaller competition benchmarks use multiple sampled completions. For example, the predefined Qwen2.5 configs currently use 16 generations for AIME2425 and AMC23, but one generation for MATH-500, GSM8K, OlympiadBench, and CollegeMath.

Not every benchmark defines every model-family preset. In particular, several science, GeneralQA, and programming configs contain `qwen2.5` and `qwen3` but no `r1-distill`. Passing a missing preset raises `KeyError` before vLLM is initialized. Inspect the relevant JSON files before evaluating a heterogeneous benchmark list.

## What happens during evaluation

For one command, the evaluator performs these stages:

1. It loads and validates the selected config for every requested dataset. This happens before model initialization, so missing configs fail early.
2. It creates one vLLM instance and reuses it across all datasets.
3. For each dataset, it loads the `test` split and constructs chat prompts with `prepare_dataset`.
4. It calls `llm.chat` once for that dataset, requesting `num_generations` outputs per prompt.
5. It flattens all prompt-completion pairs and grades them with `eval_answer_reward`.
6. It independently applies `format_reward` to each completion.
7. It writes detailed JSON results and prints aggregate means to the benchmark log.

The reward path depends on each sample's `verifier`:

- Missing/default verifier: parses and compares the boxed mathematical answer locally.
- `general` or related GeneralQA verifier: calls the external verifier vLLM.
- `code` or `code_*`: extracts code and executes the appropriate tests in the configured sandbox.

For GeneralQA evaluation, start the verifier proxy as described in [Verifier Setup](appendix.md#verifier-setup) and wake it before running the evaluator. Unlike policy training, `evaluation.py` does not automatically manage verifier sleep. Programming benchmarks also require the sandbox from [Sandbox Setup](appendix.md#sandbox-setup).

## Metrics and result files

Suppose `output_folder` is `output/evaluation/model-a` and the dataset is `data/AIME2425`. Evaluation creates:

```text
output/evaluation/model-a/
├── benchmark_sampling_AIME2425.log
└── result_benchmark_AIME2425.json
```

The log contains the resolved dataset config and three summary values:

```text
eval num:  480
eval acc:  0.625
eval format:  0.9916666666666667
```

- `eval num` is `number of test problems × num_generations`.
- `eval acc` is the mean `acc_score` across all generated completions.
- `eval format` is the fraction/mean score of completions satisfying the expected response format.

When `num_generations` is greater than one, `eval acc` remains the sample-level mean accuracy over all completions. It is not pass@k, majority-vote accuracy, or an average of per-problem best scores. Compute those metrics separately from the detailed JSON if needed.

Each JSON item represents one completion:

```json
{
  "problem": "...",
  "gold_solution": "...",
  "gold_process": "...",
  "verifier": "default",
  "completion": "...",
  "acc_score": 1.0,
  "format_score": 1.0
}
```

With multiple generations, the same problem appears in consecutive records. The output does not currently include an explicit problem ID or generation index, so retain the original dataset and config when performing additional aggregation.

## Resource and reproducibility notes

- vLLM generates all prompts for one benchmark in one call, and all completions are accumulated before reward calculation. Large benchmarks, long outputs, or a high `num_generations` can require substantial host and GPU memory.
- `num_gpus` must match the GPUs visible to the process and form a valid tensor-parallel configuration for the model.
- Benchmark output filenames use only the final dataset directory name. Two dataset paths with the same basename will overwrite each other's results in one output folder.
- The evaluator overwrites a benchmark's previous log and JSON rather than resuming or skipping completed work. Use a distinct output folder for each model/checkpoint and evaluation setting.
- For comparable paper results, keep the delivered benchmark configs fixed across all checkpoints, use the same verifier and sandbox setup, and record the model revision and dependency versions.

## Pre-flight checklist

Before starting a full evaluation run, verify that:

1. Every dataset has a processed `test` split and a delivered `.evaluation_config/evaluation_config.json`.
2. The selected config name exists in every requested benchmark.
3. `enable_thinking` and the config family match the model's chat template.
4. The model fits at the requested tensor-parallel size and vLLM memory utilization.
5. External verifier instances are awake for GeneralQA, and the code sandbox works for programming tasks.
6. `output_folder` is unique for this checkpoint and configuration.
