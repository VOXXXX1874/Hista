## dapo

ACCELERATE_LOG_LEVEL=info \
accelerate launch \
--config_file recipes/zero3.yaml \
--main_process_port 29501 \
--num_processes=4 \
src/rl/grpo.py \
--config recipes/24G/Qwen2.5-1.5B/hybrid/GRPO_inst_dapo.yaml \
--verifier_vllm_base_url http://localhost:8000/v1 \
> ./output/Qwen2.5-1.5B/GRPO_inst_hybrid_dapo_sampling.log 2>&1 &

## dapo + hista

ACCELERATE_LOG_LEVEL=info \
accelerate launch \
--config_file recipes/zero3.yaml \
--main_process_port 29502 \
--num_processes=4 \
src/rl/hista.py \
--config recipes/24G/Qwen2.5-1.5B/hybrid/Hista_inst_dapo.yaml \
--verifier_vllm_base_url http://localhost:9000/v1 \
> ./output/Qwen2.5-1.5B/Hista_inst_hybrid_dapo_sampling.log 2>&1

## csipo

ACCELERATE_LOG_LEVEL=info \
accelerate launch \
--config_file recipes/zero3.yaml \
--main_process_port 29501 \
--num_processes=4 \
src/rl/grpo.py \
--config recipes/24G/Qwen2.5-1.5B/hybrid/GRPO_inst_csipo.yaml \
--verifier_vllm_base_url http://localhost:8000/v1 \
> ./output/Qwen2.5-1.5B/GRPO_inst_hybrid_csipo_sampling.log 2>&1 &

## csipo + hista

ACCELERATE_LOG_LEVEL=info \
accelerate launch \
--config_file recipes/zero3.yaml \
--main_process_port 29502 \
--num_processes=4 \
src/rl/hista.py \
--config recipes/24G/Qwen2.5-1.5B/hybrid/Hista_inst_csipo.yaml \
--verifier_vllm_base_url http://localhost:9000/v1 \
> ./output/Qwen2.5-1.5B/Hista_inst_hybrid_csipo_sampling.log 2>&1