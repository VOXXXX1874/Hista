## dapo

ACCELERATE_LOG_LEVEL=info \
accelerate launch \
--config_file recipes/zero3.yaml \
--main_process_port 29501 \
--num_processes=1 \
src/rl/grpo.py \
--config recipes/80G/Qwen2.5-1.5B/hybrid/GRPO_dapo_ultra.yaml \
--verifier_vllm_base_url http://localhost:8000/v1 \
> ./output/Qwen2.5-1.5B/GRPO_ultra_dapo_sampling.log 2>&1

## dapo + hista

ACCELERATE_LOG_LEVEL=info \
accelerate launch \
--config_file recipes/zero3.yaml \
--num_processes=1 \
--main_process_port 29500 \
src/rl/hista.py \
--config recipes/80G/Qwen2.5-1.5B/hybrid/Hista_dapo_ultra.yaml \
--verifier_vllm_base_url http://localhost:9000/v1 \
> ./output/Qwen2.5-1.5B/Hista_ultra_dapo_sampling.log 2>&1

## csipo

ACCELERATE_LOG_LEVEL=info \
accelerate launch \
--config_file recipes/zero3.yaml \
--main_process_port 29501 \
--num_processes=1 \
src/rl/grpo.py \
--config recipes/80G/Qwen2.5-1.5B/hybrid/GRPO_csipo_ultra.yaml \
--verifier_vllm_base_url http://localhost:8000/v1 \
> ./output/Qwen2.5-1.5B/GRPO_ultra_csipo_sampling.log 2>&1

## csipo + hista

ACCELERATE_LOG_LEVEL=info \
accelerate launch \
--config_file recipes/zero3.yaml \
--num_processes=1 \
--main_process_port 29500 \
src/rl/hista.py \
--config recipes/80G/Qwen2.5-1.5B/hybrid/Hista_csipo_ultra.yaml \
--verifier_vllm_base_url http://localhost:9000/v1 \
> ./output/Qwen2.5-1.5B/Hista_ultra_csipo_sampling.log 2>&1