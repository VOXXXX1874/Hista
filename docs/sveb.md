# State Value Estimation Benchmark

The State Value Estimation Benchmark (SVEB) measures how accurately a method estimates the probability that generation from an intermediate reasoning state will eventually receive a positive reward. Hista, Numca, and a learned PPO critic are evaluated on the same sampled states and against the same Monte Carlo target.

## How the benchmark works

For each selected problem, a `generate` run performs the following stages:

1. Sample `grpo_num` complete responses from the initial state and verify them. Their average reward is the GRPO initial-state baseline.
2. Select one intermediate position from the first sampled response.
3. Sample `mcs_num` continuations from that position and verify them. Their average reward is treated as the unbiased state-value target.
4. Estimate the value at the selected state with Hista, Numca, or a PPO critic, then report its absolute error against the Monte Carlo target.

The main result is `Average MAE between Estimated and Unbiased Value Function`. The log also reports the GRPO baseline MAE and the noise MAE obtained when the Monte Carlo target is computed with fewer continuation samples.

The generated JSON stores the expensive, method-independent parts of the benchmark: the problem, initial responses and rewards, selected response prefix, final reward, Monte Carlo target, and target-noise estimates. A `reuse` run reads this JSON and evaluates another method or configuration without generating the initial and continuation rollouts again.

```text
SVEB input data
    |
    `-- generate: sample initial responses and continuations
            |-- method evaluation log (`--output_path`)
            `-- reusable trajectories (`--save_path`)
                    |
                    `-- reuse: evaluate another method/configuration
                            `-- new method evaluation log
```

## Prerequisites and directory layout

Run all commands from the repository root. Prepare or download the five SVEB fields as described in [Data Preparation](data_preparation.md#construct-sveb-data-and-hybrid-training-data):

```text
data/<model-size>/
|-- sveb_number/train.json
|-- sveb_math/train.json
|-- sveb_science/train.json
|-- sveb_general/train.json
`-- sveb_program/train.json
```

Each record must contain at least `problem`, `solution`, and `verifier`. Create the output directories before starting an evaluation because the Python entry points open their output files directly:

```bash
mkdir -p output/sveb/Qwen2.5-1.5B-Instruct tmp
```

The reference scripts are organized by method and execution mode:

```text
scripts/sveb/
|-- hista/{generate,reuse}/
|-- numca/{generate,reuse}/
`-- ppo/
    |-- prepare_data/
    |-- generate/
    `-- reuse/
```

The supplied shell scripts contain one command per field and start every command in the background with `&`. Running a whole script can therefore launch five model-serving jobs at once. Copy or uncomment only the fields that fit your available GPUs, and adjust `CUDA_VISIBLE_DEVICES`, `--tp`, model paths, and output paths before launching them.

## Generate once, then reuse

You must run one `generate` command before the corresponding `reuse` command unless you downloaded a compatible generated SVEB JSON. A typical Hista generation command is:

```bash
python src/sveb/hista/evaluate_sta_estim_generate.py \
    --model_name Qwen/Qwen2.5-1.5B-Instruct \
    --dataset_path data/1.5B/sveb_math/train.json \
    --output_path output/sveb/Qwen2.5-1.5B-Instruct/hista_sveb_math.log \
    --save_path output/sveb/Qwen2.5-1.5B-Instruct/sveb_math_S32_MCS20_NumPro3000.json \
    --grpo_num 32 \
    --mcs_num 20 \
    --max_length 4096 \
    --num_of_problems 3000 \
    --tp 1
```

This samples up to 3,000 distinct problems, generates 32 initial responses and 20 continuations per problem, writes detailed metrics to `--output_path`, and saves reusable cases to `--save_path`. The run is GPU- and verifier-intensive: reducing `--num_of_problems`, `--grpo_num`, `--mcs_num`, or `--max_length` is useful for a smoke test, but changes the benchmark configuration and result variance.

The common arguments are:

| Argument | Meaning |
| --- | --- |
| `--dataset_path` | SVEB `train.json` for one field, or a generated JSON in reuse mode. |
| `--output_path` | Detailed per-case log and aggregate MAE summary. |
| `--save_path` | Reusable generated cases; available only in generate mode. |
| `--num_of_problems` | Maximum number of evaluated cases. |
| `--grpo_num` | Initial-state responses used for the GRPO baseline. |
| `--mcs_num` | Continuations used to approximate the unbiased target. |
| `--max_length` | Maximum new tokens for each generation. |
| `--tp` | vLLM tensor-parallel GPU count. |
| `--use_default_system_prompt` | Use the model's default prompt instead of the repository math/code prompts. |
| `--enable_thinking` | Enable the tokenizer's thinking mode, for example for Qwen3. |

Keep the model, prompt flags, sampled trajectories, and verifier behavior fixed when comparing methods. Otherwise the methods are no longer evaluated on the same states and targets.

## Hista

Reference scripts:

```text
scripts/sveb/hista/generate/sveb_qwen2.5-1.5B-Instruct.sh
scripts/sveb/hista/reuse/sveb_qwen2.5-1.5B-Instruct.sh
```

Hista uses hidden-state representations from the action model and the rewards of sampled initial responses to estimate the selected state's value. In addition to the common arguments, its scripts expose representation layer and neighbor-selection parameters such as `--layer`, `--max_k`, `--min_k`, `--min_interval`, `--alpha`, `--mean_window`, and `--min_distance`.

After trajectories have been generated by any method, Hista can evaluate them with:

```bash
python src/sveb/hista/evaluate_sta_estim_from_existing.py \
    --model_name Qwen/Qwen2.5-1.5B-Instruct \
    --dataset_path output/sveb/Qwen2.5-1.5B-Instruct/sveb_math_S32_MCS20_NumPro3000.json \
    --output_path output/sveb/Qwen2.5-1.5B-Instruct/hista_sveb_math_reuse.log \
    --num_of_problems 3000
```

Reuse avoids vLLM rollout generation, but Hista still loads the action model to compute hidden-state representations and therefore still requires model inference memory.

Only `uniform` selection and `ema` averaging are currently supported. The generate entry point always uses them, while the reuse entry point rejects other values even though its argument help mentions additional choices.

## Numca

Reference scripts:

```text
scripts/sveb/numca/generate/sveb_qwen2.5-1.5B-Instruct.sh
scripts/sveb/numca/reuse/sveb_qwen2.5-1.5B-Instruct.sh
```

Numca extracts and aggregates numeric states from sampled responses. Its generate command accepts the common rollout arguments, while reuse needs only the generated JSON, output path, and number of problems:

```bash
python src/sveb/numca/evaluate_sta_estim_from_existing.py \
    --dataset_path output/sveb/Qwen2.5-1.5B-Instruct/sveb_math_S32_MCS20_NumPro3000.json \
    --output_path output/sveb/Qwen2.5-1.5B-Instruct/numca_sveb_math_reuse.log \
    --num_of_problems 3000
```

This is the lightest reuse path because it does not reload the action model or generate new responses.

## PPO critic

PPO evaluation first requires a trained critic and value head. To isolate state-value estimation from full actor-critic training, this repository freezes the action model and trains the critic by supervised fine-tuning on previously sampled, reward-labeled responses.

### 1. Prepare critic-SFT data

Obtain the extra rollout files described in [Data Preparation](data_preparation.md#extra-rollouts), or generate them with the rollout pipeline, then run:

```bash
bash scripts/sveb/ppo/prepare_data/ppo_sft_qwen2.5-1.5B-Instruct.sh
```

For every field, the script prepares both `ppo-1` and `ppo-n` datasets:

- `ppo-1` selects rollout problems outside SVEB, representing critic training from the first PPO epoch.
- `ppo-n` selects rollout problems inside SVEB for which labeled responses are available, representing subsequent epochs.

Problems are split 80%/20%, then expanded into one `{text, reward}` example per response and written to `data/1.5B/{ppo-1,ppo-n}/<field>/{train,test}.json`. Selection and shuffling use seed 42 by default. If fewer eligible problems are available than requested, the processor prints a warning and uses all available problems.

### 2. Train the critic

Reference training commands for all five fields are collected in:

```text
scripts/train/24G/ppo_sft/train_qwen2.5-1.5B.sh
```

For example, train the number critic on two gpus (each gpu has at least 24G memory) with:

```bash
ACCELERATE_LOG_LEVEL=info \
accelerate launch \
    --config_file recipes/zero3.yaml \
    --num_processes=2 \
    src/ppo_sft/sft.py \
    --config recipes/24G/Qwen2.5-1.5B/ppo_sft/Critic_on_number.yaml \
    > output/Qwen2.5-1.5B/ppo_critic_on_number.log 2>&1
```

The recipe reads `data/1.5B/ppo-n/number` and writes the best checkpoint under `output/Qwen2.5-1.5B/Critic_on_number/`. Use the checkpoint directory that was actually produced; do not assume a fixed checkpoint number.

### 3. Evaluate or reuse trajectories

Pass the trained checkpoint as both the critic model and value-head path:

```bash
python src/sveb/ppo/evaluate_sta_estim_generate.py \
    --action_model_name Qwen/Qwen2.5-1.5B-Instruct \
    --critic_model_path output/Qwen2.5-1.5B/Critic_on_number/checkpoint-XXXX \
    --value_head_path output/Qwen2.5-1.5B/Critic_on_number/checkpoint-XXXX \
    --dataset_path data/1.5B/sveb_number/train.json \
    --output_path output/sveb/Qwen2.5-1.5B-Instruct/ppo_sveb_number.log \
    --save_path output/sveb/Qwen2.5-1.5B-Instruct/sveb_number_S32_MCS20_NumPro3000.json \
    --grpo_num 32 \
    --mcs_num 20 \
    --num_of_problems 3000
```

The complete five-field templates are in `scripts/sveb/ppo/generate/sveb_qwen2.5-1.5B-Instruct.sh`. To evaluate a critic on trajectories already generated by Hista, Numca, or PPO, use the matching commands in `scripts/sveb/ppo/reuse/sveb_qwen2.5-1.5B-Instruct.sh`; point `--dataset_path` at the saved generated JSON and keep the same action-model prompt settings.

## Reading the outputs

Each evaluation log contains the selected state, sampled-response reward, estimated value, Monte Carlo target, GRPO initial-state value, and per-case absolute errors. Aggregate lines appear at the end:

```text
Average MAE between Estimated and Unbiased Value Function: ...
Average MAE between GRPO and Unbiased Value Function: ...
Average Unbiased Value Function Noise MAE with 1 samples: ...
...
```

Lower MAE is better. The first line is the primary comparison for the evaluated method. The second is the initial-state GRPO baseline, and the noise lines provide context for the irreducible sampling variation in a target estimated from finitely many continuations.
