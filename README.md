# Hista

Open source code for paper "Hista and Numca: Estimate State Value Effectively for Large Language Model Reinforcement Learning"

## 1. Installation

### 1.1 Conda & pip

It requires cuda >= 12.2 and < 13.0. First create environment

```bash
conda create -n Hista python=3.12
conda activate Hista
```

Then, install the dependencies

```bash
pip install -e .
```

Finally, install the flash attention. In case you encounter any issue in compilation, we recommend you to download a compiled wheel from https://github.com/Dao-AILab/flash-attention/releases?page=3, for example

```bash
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl

pip install flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
```

### 1.2 Singularity & Docker

Coming soon.

### 1.3 Folders

Please create following folders in the root directory of this repo

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

1. `data`: Store the final data for training and evaluation.
2. `output`: Store the training log, benchmark result, and checkpoints of different model. Remember to create the model folder in before you start training.
3. `tmp`: Store the temporary logs, like the running of standalone vllm server.

## 2. Data Downloading

You can download the data selectively according to the experiment you want to reproduce.

1. If you only want to reproduce the result of "Training base model on math dataset", which is the fastest path without extra setup, you should download the math-related dataset and benchmark in **2.1 Original Data**, and then process the data **3. Data Processing**.
2. If you want to reproduce the result on State Value Estimation Benchmark (SVEB), you should download the SVEB related processed dataset in **2.2 Processed Data**, and then go to **4. State Value Estimation Benchmark**.
3. If you want to reproduce the result of "Training instruct model on hybrid dataset", you should download the benchmarks in **2.1 Original Data** and process the benchmarks in **3. Data Processing**. Then, you should download the processed hybird training dataset in **2.2 Processed Data** and then go to **6. Training on Hybrid Dataset**.
4. If you want to test or modify the data processing procedure, you should download all the data in **2.1 Original Data** and then go to **3. Data Processing**.

### 2.1 Original Data

Run following script in root directory of this repo to download **math related training and benchmark data**. You will find the downloaded data in the corresponding folder under `src/data_preparation` and `src/data_preparation/benchmark`

```bash
bash scripts/download/download_math.sh
```

Run following script in root directory of this repo to download **other benchmarks data related to science, general QA, and programming.**

```bash
bash scripts/download/download_other_benchmark.sh
```

If you want to go through the construction of SVEB data and hybrid dataset, then run this script in root directory of this repo to download the **raw data about math, science, general QA, and programming**.

```bash
bash scripts/download/download_training_data.sh
```

### 2.2 Processed Data

To save your time on downloading and processing data, we upload our processed data to huggingface. You can use it to reproduce our result on SVEB or training on hybrid dataset.

#### 2.2.1 SVEB data

Coming soon.

#### 2.2.2 Hybrid Dataset

Comming soon.

## 3. Data Processing

You can process the data selectively according to the experiment you want to reproduce.

1. If you only want to reproduce the result of "Training base model on math dataset", which is the fastest path without extra setup, you only need to finish **3.1 Align Math Data Format** and then go to **5. Training on Math Dataset**.
2. If you want to process the data from scratch for testing or modifying the pipeline, you need to go through the whole procedure from **3.1** to **3.5**.

Please create a `data` folder in the root directory of this repo, where we will store final data used to train or evaluate

### 3.1 Align Math Data and Other Benchmarks

Different math dataset and benchmark have different format. We need to unify them into one format to avoid writing redundant code in our training program. Please ensure the downloading is finished correctly. Then, you can run the script to align format of all math dataset and benchmark

```bash
bash scripts/process/align_math.sh
```

The downloaded data will be placed in the `raw` folder, and the processed data will be placed in the `processed` folder. For example

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

The script will automatically copy the processed data to `data`. To process other benchmark from different fields, please use 

```bash
bash scripts/process/align_others_benchmark.sh
```

### 3.2 Sample & Align Other Data

Because the dataset like `OpenR1-220K` and `verifiable-python` contains too much data samples, we only use part of it to construct the training and SVEB data, and chunkize it for flexible process.

```bash
bash scripts/process/align_others_training.sh
```

### 3.3 Sample extra rollouts and Select the Data with Suitable Difficulty

If you want to process `OpenR1-220K`, science, and general QA dataset, you need to setup a verifier to do verification. Please follow **7.1 Verifier Setup**. If you want to process the Programming dataset, you need a sandbox environment. Please follow **7.2 Sandbox Setup**. Sampling extra rollouts for the data is neccessary because raw data contain lots of samples that is either too easy or too difficult for model to solve. It will consume some time depending on your gpu resources and your reproduce target. Therefore, we provide the basic commands for you to select. Please refer to following scripts and schedule the sampling.

```bash
scripts/process/sample_1.5B.sh
scripts/process/sample_3B.sh
scripts/process/sample_7B.sh
```

The sampled data will be placed in the `partition_x` folder in the corresponding dataset under `src/data_preparation`. We can create dataset in different difficulty tier for each model using the sampled data, which can prevent the appearing of group with all responses correct or all responses wrong.

```bash
scripts/process/select_1.5B.sh
scripts/process/select_3B.sh
scripts/process/select_7B.sh
```

### 3.5 Construct the SVEB Data and Hybrid Training Data

The commands for creating hybrid training data of different models is collected and labeled in `scripts/construction/hybrid_training.sh`. An example is 

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

where `input_files` and `num_samples` correspond one to one. We will sample this number of data from each input files, collect them into one file, split into `train.json` and `test.json`, and finally store in `output_folder`. Remember to have those data prepared when you run this data construction command.


The commands for creating the SVEB data of different models and different fields is collected and labeled in `scripts/construction` like `sveb_[model_size].sh`. It shares the same `merge_training.py` but enable `--exclude_test` so that all the data will be store in a single `train.json`. The commands for preparing the special data of Numca algorithm is collected and labeled in `scripts/construction/numca_training.sh`. 

## 4. State Value Estimation Benchmark

After you have downloaded or processed the data, you can use the script under `scripts/sveb` to reproduce our evaluation result on SVEB. There are two folders under `scripts/sveb`, `generate` and `reuse`. The script in the first folder will go through the whole pipeline of SVEB, including 1. sampling from the initial state, 2. sampling from the given position, and 3. evaluating state value estimation method. The script in the second folder will reuse the data generated in the stage 1 and 2, and only run 3. evaluate state value estimation method, which is far more efficient. Now you must start at `generate`, but you can do it in one method and reuse the trajectory while evaluating another two methods.

### 4.1 Hista

For the generating of SVEB data, you can refer to the commands in 

```text
scripts/sveb/hista/generate/sveb_qwen2.5-1.5B-Instruct.sh
```

and the reuse is in 

```text
scripts/sveb/hista/reuse/sveb_qwen2.5-1.5B-Instruct.sh
```

Please modify them accordingly.

### 4.2 Numca

Coming soon.

### 4.3 PPO

Coming soon.

## 5. Training on Math Dataset

After all the math data and benchmark is prepared, you can use the script under `scripts/train` to start training, or create your training script based on it. The structure under `scripts/train` is like

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

where `24G` and `80G` indicate the suggest GPU memory level to run the commands inside. `math` and `hybrid` corresponds to training on math dataset and hybrid dataset. The available commands for one model is collected and labeled in each file. An example of training script is

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

The meaning of arguments are
1. `config_file`: The configuration of deepspeed. In this example, we use DeepSpeed Zero3 as distributed training strategy.
2. `num_processes`: Number of GPU allocated to this training. In this example, we use 4 GPUs.
3. `main_process_port`: The port of main process for communication between different process. If you run mutiple training on one node, then they must have different main process port.
4. `config`: The configuration of training, including the algorithm, model, and data. In this example, the configuation is running DAPO algorithm on Qwen2.5-1.5B model.

You can modify them accordingly. Each command corresponds to one recipes in `recipes`, which share the same file structure.

## 6. Training on Hybrid Dataset

After downloading or processing the hybrid data, ensure they are placed in desired structure

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

Because the hybrid dataset contain general QA and programming data, we need to setup vLLM verifier and sandbox. Please follow the **7.1 Verifier Setup** and **7.2 Sandbox Setup**. Ensure the vLLM verifier is in sleep before starting training. The placement of training command and recipes is similar with **5. Training on Math Dataset**. An example of command is

```bash
ACCELERATE_LOG_LEVEL=info \
accelerate launch \
--config_file recipes/zero3.yaml \
--num_processes=8 \
--main_process_port 29502 \
src/rl/grpo.py \
--config recipes/24G/Qwen2.5-1.5B/hybrid/GRPO_dapo_ultra.yaml \
> ./output/Qwen2.5-1.5B/GRPO_dapo_ultra_sampling.log 2>&1
```

We only change the `config` and increase the number of GPUs to 8 (to save gpu memory, we choose to only wake verifier during reward calculation. If there are mutiple training task, it is difficult to manage, so we suggest use all gpu for one task. However, it is possible to use one verifier to serve mutiple training task or use different verifier for different training task, which require you to read and understand the code.)

## 7. Evaluation

In the provided recipes, we only train a fixed step number, and we will periodically run evaluation during the training. Each evaluation score will be recorded and compared with previous evaluation score. The best checkpoint will be saved to specified position. To measure the real generality of different methods, we evaluate the saved best checkpoint on different benchmarks. We have a predefined evaluation config in `src/evaluation/predefined`, which contain two type, `qwen2.5`, `r1-distill`, and `qwen3`. You can use this script to deliver predefined config to different benchmarks (make sure you have processed all the benchmarks)

```bash
bash scripts/evaluation/deliver_config.sh
```

Then, you can do evaluate `Qwen/Qwen2.5-1.5B-Instruct` on the math benchmarks using following command:

```bash
python src/evaluation/evaluation.py \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --output_folder output/benchmark_ultra_instruct \
  --datasets data/MATH-500 data/GSM8K data/OlympiadBench data/amc23 data/AIME2425 data/GaoKao data/CollegeMath \
  --evaluation_config qwen2.5 \
  --num_gpus 1
```

You can substitute the `model_name` with your checkpoints saved during training. For larger model, you can enable tensor parallel of vLLM through increasing the `num_gpus` arguments. Notice that `evaluation_config` has three available options, `qwen2.5`, `qwen3`, and `r1-distill`. Please use it accordingly.

## 8. Appendix

### 8.1 Verifier Setup

Since the answer of the GeneralQA dataset is a short phrase that cannot be parsed or compared directly, we need to small language model as verfier to judge the model answer based on ground truth answer. We will use vLLM to setup standalone instance for verification, and we will enable sleep mode to free GPU memory. Therefore, you need to first set

```bash
export no_proxy="localhost,127.0.0.1"
export NO_PROXY="localhost,127.0.0.1"
export VLLM_SERVER_DEV_MODE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export TOKENIZERS_PARALLELISM=false
```

Then, use this script to setup multiple vLLM instance (assume you have one node with 8 GPU and port 8000 is available):

```bash
bash src/vllm_verifier/init_vllm_server.sh
```

You can adjust the number of gpus, port, and gpu memory utilization through the 1st, 2nd, and 3rd arguments, for example

```bash
bash src/vllm_verifier/init_vllm_server.sh 1 9000 0.2
```

Suppose the port index is $p$ and gpu number is $n$, we will create one vLLM instance per gpu and listen to the port $p+1$, $p+2$, ... To manage those instances, we create a proxy to route different request. You will see the output of `init_vllm_server.sh` like

```text
🚀 Initializing vLLM server on 8 GPUs
   - Proxy port: 8000
   - vLLM backend ports: 8001-8008
Please initialize vLLM proxy server when the vllm is ready use command: python src/vllm_verifier/vllm_proxy.py --num-gpus 8 --proxy-port 8000 &
```

Then check the availability of vllm instance in `tmp/vllm_gpu_${i}.log`. If all of them are available, create the proxy use the command in the last line.

```bash
python src/vllm_verifier/vllm_proxy.py --num-gpus 8 --proxy-port 8000 &
```

After the router is created, the vLLM instance will be put to sleep. You can test them with

```bash
python src/vllm_verifier/ping_vllm.py --base-url http://localhost:8000/v1
```

They will be put to sleep after testing. If you prepare to run training, then make sure they are sleeping when you begin to train, and remember to modify the corresponding `verifier_vllm_base_url` is you specify a different port. If you prepare to run sampling in data processing, then you need to wake up those instance 

```bash
python src/vllm_verifier/wakeup_vllm.py --proxy-url http://localhost:8000
```

You can use following command to kill all the vllm instances after finishing the task

```bash
pkill -9 -f vllm
pkill -9 -f VLLM
```

### 8.2 Sandbox Setup

Since the correctness of programming problems is judged by whether the generated program can pass all the test cases in a limited time, we need to execute the program in an isolated sandbox in case the model generate malicious code. Our experiments is conducted on a HPC that only equipped with **Singularity**, and our code is well tested on it. First create `singularity_images` to store the images locally. We use an image with basic numerical computation libraries like numpy and scipy as the environment in sandbox:

```bash
singularity pull ./singularity_images/hista_programming.sif docker://quay.io/jupyter/scipy-notebook:latest
```

Then, test whether you can run the container

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

If you are using different version and you meet error, please modify the code in `src/rl/utils/rewards.py` accordingly. Beside **Singularity**, we also test and support **Docker**. You need to set this environment variable

```bash
export CODE_SANDBOX_RUNTIME=docker
```

Then test whether you can run the container

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

If you are using different version and you meet error, please also modify the code in `src/rl/utils/rewards.py` accordingly. If your HPC or your PC use other software, for example `apptainer`, you need to do the implementation by yourself, but it should be similar.