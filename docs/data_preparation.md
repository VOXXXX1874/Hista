# Data Preparation

You can download and process data selectively according to the experiment you want to reproduce.

## Workflow selection

- To reproduce **training base models on the math dataset**, which is the fastest path without extra setup, download the math-related datasets and benchmarks in [Original Data](#original-data), then run [Align Math Data and Other Benchmarks](#align-math-data-and-other-benchmarks).
- To reproduce **State Value Estimation Benchmark (SVEB)**, download the SVEB-related processed dataset in [Processed Data](#processed-data), then go to [State Value Estimation Benchmark](sveb.md).
- To reproduce **training instruct models on the hybrid dataset**, download the benchmarks in [Original Data](#original-data), process the benchmarks, download the processed hybrid training dataset in [Processed Data](#processed-data), then go to [Training on Hybrid Dataset](training.md#training-on-hybrid-dataset).
- To test or modify the data processing procedure, download all data in [Original Data](#original-data), then run the full processing flow below.

Create a `data` folder in the repository root. This folder stores final data used for training and evaluation.

## Directory layout

Data preparation code and intermediate artifacts live under `src/data_preparation`. The following tree shows the common layout; directories such as `raw`, `processed`, and `partition_*` are populated by the commands described below.

```text
src/data_preparation/
|-- MATH/                         # math training data
|   |-- download.py
|   |-- process.py
|   |-- raw/                      # source-format downloads
|   `-- processed/                # unified train/test JSON files
|-- DAPO-17K/                     # math training data
|-- openr1-220K/                  # math training data for the hybrid setting
|-- MixtureOfThought/             # science training data
|-- GeneralQA/                    # general-QA training data
|-- verifiable_python/            # programming training data
|   # The five directories above follow the same pattern:
|   # download/process scripts, raw data, partition_1..3, and
|   # model-specific rollout/selection directories such as 1.5B/sampled.
|-- benchmark/
|   |-- MATH-500/                 # one directory per evaluation benchmark
|   |-- GSM8K/
|   |-- ...
|   `-- MBPP+/
|       # Each benchmark directory contains download.py/process.py and
|       # normally raw/test.json and processed/test.json.
|-- extra_sampling.py             # generate and verify extra responses
|-- merge_samples.py              # merge rollout results from partitions
|-- select_samples.py             # select examples by empirical difficulty
`-- merge_for_training.py         # sample, merge, shuffle, and split final data
```

`src/data_preparation` is the working area: downloaded and intermediate files remain there so that processing can be inspected or rerun. The repository-level `data/` directory is the consumption area: alignment and construction scripts copy or write the final files used by training and evaluation there.

## Fastest path: math-only

Run:

```bash
bash scripts/download/download_math.sh
bash scripts/process/align_math.sh
```

The first command downloads the MATH and DAPO-17K training sets and eight math benchmarks into their respective `raw/` directories. The second command normalizes them, removes MATH-500 overlap from the MATH test set, and replaces the corresponding repository-level `data/*` directories with copies of the normalized data. See the detailed descriptions below.

Then continue with [Training on Math Dataset](training.md#training-on-math-dataset).

## Data Downloading

### Original Data

Run the following script in the repository root to download **math-related training and benchmark data**. Downloaded data will be placed under the corresponding folders in `src/data_preparation` and `src/data_preparation/benchmark`.

```bash
bash scripts/download/download_math.sh
```

This runs each dataset's `download.py` from inside its dataset directory. It fetches MATH and DAPO-17K plus MATH-500, GSM8K, OlympiadBench, MinervaMath, AMC 2023, AIME 2024/2025, Gaokao 2023 Math (English), and CollegeMath. The result is source-format JSON under:

```text
src/data_preparation/{MATH,DAPO-17K}/raw/
src/data_preparation/benchmark/{MATH-500,GSM8K,OlympiadBench,MinervaMath,amc23,AIME2425,Gaokao2023-Math-En,CollegeMath}/raw/
```

Most downloads come from Hugging Face; CollegeMath is downloaded directly from the MathScale repository. This command does not create training-ready files in `data/`; that happens in `align_math.sh`.

Run the following script to download **other benchmark data related to science, general QA, and programming**.

```bash
bash scripts/download/download_other_benchmark.sh
```

This downloads the raw test splits for SciEval, TheoremQA, MinervaMath, MMLU-Pro, GPQA-Diamond, HumanEval+, and MBPP+ into the matching `src/data_preparation/benchmark/<dataset>/raw/` directories. MinervaMath is intentionally included here as a science benchmark even though the math-only downloader also fetches it. No files are copied to `data/` at this stage.

If you want to go through the construction of SVEB data and the hybrid dataset, run this script to download the **raw data about math, science, general QA, and programming**.

```bash
bash scripts/download/download_training_data.sh
```

This downloads the raw sources used by the full hybrid/SVEB construction flow: DAPO-17K and OpenR1-220K for math, the science split of Mixture-of-Thoughts, WebInstruct-verified for general QA, and verifiable-coding-problems-python for programming. Outputs are written to each dataset's `raw/` directory, except Mixture-of-Thoughts, which uses `raw_science/`.

### Processed Data

To save time on downloading and processing data, we upload processed data to Hugging Face. You can use it to reproduce SVEB results or train on the hybrid dataset.

#### Extra Rollouts

Coming soon.

#### SVEB data

Coming soon.

#### Hybrid dataset

Coming soon.

## Data Processing

### Align Math Data and Other Benchmarks

Different math datasets and benchmarks have different formats. We unify them into one format to avoid redundant code in training.

After the download is finished, run:

```bash
bash scripts/process/align_math.sh
```

This command performs two operations:

1. It converts each downloaded dataset to the common records expected by the training/evaluation code (including `problem`, `solution`, `verifier`, and `id` where applicable), writing `processed/train.json` and/or `processed/test.json` inside each dataset directory. While processing MATH, it excludes problems present in the raw MATH-500 test set and filters solutions whose final answer cannot be parsed.
2. It deletes and replaces the corresponding destination directories under `data/`. The final names are `data/MATH`, `data/DAPO`, `data/MATH-500`, `data/GSM8K`, `data/OlympiadBench`, `data/MinervaMath`, `data/amc23`, `data/AIME2425`, `data/GaoKao`, and `data/CollegeMath`.

Because the destination directories are replaced, do not keep manual changes inside those `data/*` directories when rerunning the command.

The downloaded data will be placed in the `raw` folder, and the processed data will be placed in the `processed` folder. For example:

```text
| -- Hista
|   | -- src
|       | -- data_preparation
|           | -- MATH
|               | -- raw
|               | -- processed
|           | -- benchmark
|               | -- MATH-500
|                   | -- raw
|                   | -- processed
```

The script automatically copies processed data to `data`.

To process other benchmarks from different fields, run:

```bash
bash scripts/process/align_others_benchmark.sh
```

This normalizes the downloaded science, general-QA, and programming benchmarks into `processed/test.json`, then deletes and replaces `data/SciEval`, `data/TheoremQA`, `data/MMLU-PRO`, `data/gpqa-diamond`, `data/HumanEval+`, and `data/MBPP+` with those processed directories.

### Sample and Align Other Training Data

Datasets such as `OpenR1-220K` and `verifiable-python` contain many samples. We use part of them to construct training and SVEB data, and chunk them for flexible processing.

```bash
bash scripts/process/align_others_training.sh
```

This prepares model-sampling inputs rather than final training data. It splits the already aligned DAPO-17K data into three roughly equal partitions; samples and splits 24,000 OpenR1-220K examples into three 8,000-example partitions; and samples 36,000 science examples, 36,000 general-QA examples, and 12,000 programming examples into three equal partitions per dataset. The outputs are `src/data_preparation/<dataset>/partition_{1,2,3}/train.json`.

The sampling scripts use a fixed Python seed, so reruns are reproducible for the same input ordering.

### Sample Extra Rollouts and Select Data by Difficulty

If you want to process `OpenR1-220K`, science, and general QA datasets, you need to set up a verifier. See [Verifier Setup](appendix.md#verifier-setup).

If you want to process the programming dataset, you need a sandbox environment. See [Sandbox Setup](appendix.md#sandbox-setup).

Sampling extra rollouts is necessary because raw data contains many samples that are either too easy or too difficult for the model to solve. Runtime depends on GPU resources and reproduction target.

Use the following scripts as references and schedule the sampling according to your resources:

```bash
scripts/process/sample_1.5B.sh
scripts/process/sample_3B.sh
scripts/process/sample_7B.sh
```

Sampled data will be placed in the `partition_x` folder in the corresponding dataset under `src/data_preparation`.

More precisely, each command loads the indicated Qwen2.5 Instruct model with vLLM, generates 20 responses per problem, verifies each response, and separates them into `correct_responses` and `wrong_responses`. Results are written to paths such as `src/data_preparation/DAPO-17K/1.5B/partition_1/sample_20.json`. Adjust `CUDA_VISIBLE_DEVICES`, `--tp`, model paths, and verifier `--proxy_url` before running; the scripts are resource-specific command templates rather than a lightweight preprocessing step.

Then create datasets in different difficulty tiers for each model:

```bash
scripts/process/select_1.5B.sh
scripts/process/select_3B.sh
scripts/process/select_7B.sh
```

For each dataset and model size, the selection script first merges the three `partition_*` rollout files into `<model-size>/sampled/sample_20.json`. It then computes difficulty from the fraction of correct responses and writes tiers such as `0007`, `0717`, `1737`, and `3777` under `<model-size>/sampled/<tier>/train.json`. Math datasets also receive optional `sampled_numca` tiers that require at least four distinct numeric targets. The selected ranges avoid groups where all responses are correct or all responses are wrong.

### Construct SVEB Data and Hybrid Training Data

Commands for creating hybrid training data for different models are collected and labeled in `scripts/construction/hybrid_training.sh`. One example is:

```bash
python src/data_preparation/merge_for_training.py \
    --input_files   src/data_preparation/DAPO-17K/1.5B/sampled/0717/train.json \
                    src/data_preparation/DAPO-17K/1.5B/sampled/1737/train.json \
                    src/data_preparation/DAPO-17K/1.5B/sampled/3777/train.json \
                    src/data_preparation/openr1-220K/1.5B/sampled/0717/train.json \
                    src/data_preparation/openr1-220K/1.5B/sampled/1737/train.json \
                    src/data_preparation/openr1-220K/1.5B/sampled/3777/train.json \
                    src/data_preparation/MixtureOfThought/1.5B/sampled/0717/train.json \
                    src/data_preparation/MixtureOfThought/1.5B/sampled/1737/train.json \
                    src/data_preparation/MixtureOfThought/1.5B/sampled/3777/train.json \
                    src/data_preparation/GeneralQA/1.5B/sampled/0717/train.json \
                    src/data_preparation/GeneralQA/1.5B/sampled/1737/train.json \
                    src/data_preparation/GeneralQA/1.5B/sampled/3777/train.json \
                    src/data_preparation/verifiable_python/1.5B/sampled/0007/train.json \
                    src/data_preparation/verifiable_python/1.5B/sampled/0717/train.json \
                    src/data_preparation/verifiable_python/1.5B/sampled/1737/train.json \
                    src/data_preparation/verifiable_python/1.5B/sampled/3777/train.json \
    --num_samples   1000 1000 500 1000 1000 1000 1000 1000 1000 1000 1000 1000 903 654 398 336 \
    --output_folder   data/1.5B/hybrid/
```

`input_files` and `num_samples` correspond one to one. With a fixed random seed, the script samples the requested number of examples from each input, assigns new sequential IDs, merges and shuffles them, and writes a 90%/10% split to `train.json` and `test.json` in `output_folder`.

Make sure those input files are prepared before running the construction command.

Commands for creating SVEB data for different models and fields are collected in `scripts/construction`, such as `sveb_[model_size].sh`. These scripts share `merge_for_training.py`, but enable `--exclude_test`, so all merged data is stored in a single `train.json` under `data/<model-size>/sveb_<field>/`.

Commands for preparing special data for the Numca algorithm are collected in `scripts/construction/numca_training.sh`. They consume the `sampled_numca` math tiers and produce 90%/10% `train.json` and `test.json` splits under `data/<model-size>/numca/`.
