# State Value Estimation Benchmark

After downloading or processing the data, use scripts under `scripts/sveb` to reproduce evaluation results on SVEB.

There are two folders under `scripts/sveb`:

- `generate`: Runs the whole SVEB pipeline, including sampling from the initial state, sampling from the given position, and evaluating the state value estimation method.
- `reuse`: Reuses data generated in stages 1 and 2, and only runs state value estimation evaluation. This is much more efficient.

Currently, you must start with `generate`. After generating trajectories with one method, you can reuse the trajectories while evaluating other methods.

## Hista

For SVEB data generation, refer to:

```text
scripts/sveb/hista/generate/sveb_qwen2.5-1.5B-Instruct.sh
```

For reuse:

```text
scripts/sveb/hista/reuse/sveb_qwen2.5-1.5B-Instruct.sh
```

Modify the scripts according to your model, data paths, and compute environment.

## Numca

Coming soon.

## PPO

Coming soon.
