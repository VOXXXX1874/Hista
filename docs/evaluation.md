# Evaluation

In the provided recipes, training runs for a fixed number of steps and periodically performs evaluation. Each evaluation score is recorded and compared with previous scores. The best checkpoint is saved to the specified position.

To measure the real generality of different methods, evaluate the saved best checkpoint on different benchmarks.

Predefined evaluation configs are located in `src/evaluation/predefined_config`. Available config types include:

- `qwen2.5`
- `qwen3`
- `r1-distill`

Use the following script to deliver predefined configs to different benchmarks. Make sure all benchmarks are processed first.

```bash
bash scripts/evaluation/deliver_config.sh
```

Then evaluate `Qwen/Qwen2.5-1.5B-Instruct` on math benchmarks with:

```bash
python src/evaluation/evaluation.py \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --output_folder output/benchmark_hybrid_instruct \
  --datasets data/MATH-500 data/GSM8K data/OlympiadBench data/amc23 data/AIME2425 data/GaoKao data/CollegeMath \
  --evaluation_config qwen2.5 \
  --num_gpus 1
```

You can substitute `model_name` with checkpoints saved during training.

For larger models, enable vLLM tensor parallelism by increasing `num_gpus`.

Use `evaluation_config` according to the model family. Available options are `qwen2.5`, `qwen3`, and `r1-distill`.
