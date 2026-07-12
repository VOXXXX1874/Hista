## dapo

ACCELERATE_LOG_LEVEL=info \
accelerate launch \
--config_file recipes/zero3.yaml \
--main_process_port 29501 \
--num_processes=1 \
src/rl/grpo.py \
--config recipes/80G/Qwen2.5-3B/hybrid/GRPO_inst_dapo.yaml \
--verifier_vllm_base_url http://localhost:8000/v1 \
> ./output/Qwen2.5-3B/GRPO_inst_hybrid_dapo_sampling.log 2>&1

## dapo + hista

ACCELERATE_LOG_LEVEL=info \
accelerate launch \
--config_file recipes/zero3.yaml \
--num_processes=1 \
--main_process_port 29500 \
src/rl/hista.py \
--config recipes/80G/Qwen2.5-3B/hybrid/Hista_inst_dapo.yaml \
--verifier_vllm_base_url http://localhost:9000/v1 \
> ./output/Qwen2.5-3B/Hista_inst_hybrid_dapo_sampling.log 2>&1

## csipo

ACCELERATE_LOG_LEVEL=info \
accelerate launch \
--config_file recipes/zero3.yaml \
--main_process_port 29501 \
--num_processes=1 \
src/rl/grpo.py \
--config recipes/80G/Qwen2.5-3B/hybrid/GRPO_inst_csipo.yaml \
--verifier_vllm_base_url http://localhost:9000/v1 \
> ./output/Qwen2.5-3B/GRPO_inst_hybrid_csipo_sampling.log 2>&1

## csipo + hista

ACCELERATE_LOG_LEVEL=info \
accelerate launch \
--config_file recipes/zero3.yaml \
--num_processes=1 \
--main_process_port 29500 \
src/rl/hista.py \
--config recipes/80G/Qwen2.5-3B/hybrid/Hista_inst_csipo.yaml \
--verifier_vllm_base_url http://localhost:8000/v1 \
> ./output/Qwen2.5-3B/Hista_inst_hybrid_csipo_sampling.log 2>&1