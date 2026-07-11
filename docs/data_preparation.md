# Data Preparation

You can download and process data selectively according to the experiment you want to reproduce.

## Workflow selection

- To reproduce **training base models on the math dataset**, which is the fastest path without extra setup, download the math-related datasets and benchmarks in [Original Data](#original-data), then run [Align Math Data and Other Benchmarks](#align-math-data-and-other-benchmarks).
- To reproduce **State Value Estimation Benchmark (SVEB)**, download the SVEB-related processed dataset in [Processed Data](#processed-data), then go to [State Value Estimation Benchmark](sveb.md).
- To reproduce **training instruct models on the hybrid dataset**, download the benchmarks in [Original Data](#original-data), process the benchmarks, download the processed hybrid training dataset in [Processed Data](#processed-data), then go to [Training on Hybrid Dataset](training.md#training-on-hybrid-dataset).
- To test or modify the data processing procedure, download all data in [Original Data](#original-data), then run the full processing flow below.

Create a `data` folder in the repository root. This folder stores final data used for training and evaluation.

## Fastest path: math-only

Run:

```bash
bash scripts/download/download_math.sh
bash scripts/process/align_math.sh
```

Then continue with [Training on Math Dataset](training.md#training-on-math-dataset).

## Data Downloading

### Original Data

Run the following script in the repository root to download **math-related training and benchmark data**. Downloaded data will be placed under the corresponding folders in `src/data_preparation` and `src/data_preparation/benchmark`.

```bash
bash scripts/download/download_math.sh
```

Run the following script to download **other benchmark data related to science, general QA, and programming**.

```bash
bash scripts/download/download_other_benchmark.sh
```

If you want to go through the construction of SVEB data and the hybrid dataset, run this script to download the **raw data about math, science, general QA, and programming**.

```bash
bash scripts/download/download_training_data.sh
```

### Processed Data

To save time on downloading and processing data, we upload processed data to Hugging Face. You can use it to reproduce SVEB results or train on the hybrid dataset.

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

### Sample and Align Other Training Data

Datasets such as `OpenR1-220K` and `verifiable-python` contain many samples. We use part of them to construct training and SVEB data, and chunk them for flexible processing.

```bash
bash scripts/process/align_others_training.sh
```

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

Then create datasets in different difficulty tiers for each model:

```bash
scripts/process/select_1.5B.sh
scripts/process/select_3B.sh
scripts/process/select_7B.sh
```

This avoids groups where all responses are correct or all responses are wrong.

### Construct SVEB Data and Hybrid Training Data

Commands for creating hybrid training data for different models are collected and labeled in `scripts/construction/hybrid_training.sh`. One example is:

```bash
python src/data_preparation/merge_training.py \
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

`input_files` and `num_samples` correspond one to one. The script samples the specified number of data points from each input file, collects them into one file, splits them into `train.json` and `test.json`, and stores them in `output_folder`.

Make sure those input files are prepared before running the construction command.

Commands for creating SVEB data for different models and fields are collected in `scripts/construction`, such as `sveb_[model_size].sh`. These scripts share `merge_training.py`, but enable `--exclude_test` so all data is stored in a single `train.json`.

Commands for preparing special data for the Numca algorithm are collected in `scripts/construction/numca_training.sh`.
