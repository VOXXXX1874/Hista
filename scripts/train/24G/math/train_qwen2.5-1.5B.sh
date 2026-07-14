## dapo

ACCELERATE_LOG_LEVEL=info \
accelerate launch \
--config_file recipes/zero3.yaml \
--main_process_port 29501 \
--num_processes=4 \
src/rl/grpo.py \
--config recipes/24G/Qwen2.5-1.5B/math/GRPO_base_dapo.yaml \
> ./output/Qwen2.5-1.5B/GRPO_math_base_dapo_sampling.log 2>&1 &

## dapo + hista

ACCELERATE_LOG_LEVEL=info \
accelerate launch \
--config_file recipes/zero3.yaml \
--main_process_port 29502 \
--num_processes=4 \
src/rl/hista.py \
--config recipes/24G/Qwen2.5-1.5B/math/Hista_base_dapo.yaml \
> ./output/Qwen2.5-1.5B/Hista_math_base_dapo_sampling.log 2>&1

## csipo

ACCELERATE_LOG_LEVEL=info \
accelerate launch \
--config_file recipes/zero3.yaml \
--main_process_port 29501 \
--num_processes=4 \
src/rl/grpo.py \
--config recipes/24G/Qwen2.5-1.5B/math/GRPO_base_csipo.yaml \
> ./output/Qwen2.5-1.5B/GRPO_math_base_csipo_sampling.log 2>&1 &

## csipo + hista

ACCELERATE_LOG_LEVEL=info \
accelerate launch \
--config_file recipes/zero3.yaml \
--main_process_port 29502 \
--num_processes=4 \
src/rl/hista.py \
--config recipes/24G/Qwen2.5-1.5B/math/Hista_base_csipo.yaml \
> ./output/Qwen2.5-1.5B/Hista_math_base_csipo_sampling.log 2>&1