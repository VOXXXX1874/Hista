ACCELERATE_LOG_LEVEL=info \
accelerate launch \
--config_file recipes/zero3.yaml \
--num_processes=2 src/ppo_sft/sft.py \
--config recipes/24G/Qwen2.5-1.5B/ppo_sft/Critic_on_number.yaml \
> ./output/Qwen2.5-1.5B/ppo_critic_on_number.log 2>&1

ACCELERATE_LOG_LEVEL=info \
accelerate launch \
--config_file recipes/zero3.yaml \
--num_processes=2 src/ppo_sft/sft.py \
--config recipes/24G/Qwen2.5-1.5B/ppo_sft/Critic_on_math.yaml \
> ./output/Qwen2.5-1.5B/ppo_critic_on_math.log 2>&1

ACCELERATE_LOG_LEVEL=info \
accelerate launch \
--config_file recipes/zero3.yaml \
--num_processes=2 src/ppo_sft/sft.py \
--config recipes/24G/Qwen2.5-1.5B/ppo_sft/Critic_on_science.yaml \
> ./output/Qwen2.5-1.5B/ppo_critic_on_science.log 2>&1

ACCELERATE_LOG_LEVEL=info \
accelerate launch \
--config_file recipes/zero3.yaml \
--num_processes=2 src/ppo_sft/sft.py \
--config recipes/24G/Qwen2.5-1.5B/ppo_sft/Critic_on_general.yaml \
> ./output/Qwen2.5-1.5B/ppo_critic_on_general.log 2>&1

ACCELERATE_LOG_LEVEL=info \
accelerate launch \
--config_file recipes/zero3.yaml \
--num_processes=2 src/ppo_sft/sft.py \
--config recipes/24G/Qwen2.5-1.5B/ppo_sft/Critic_on_program.yaml \
> ./output/Qwen2.5-1.5B/ppo_critic_on_program.log 2>&1