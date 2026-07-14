# PPO-N data preparation for Qwen2.5-1.5B-Instruct
python src/ppo_sft/process_critic_sft_data.py \
    --sveb_data data/1.5B/sveb_number/train.json \
    --rollout_data \
        src/data_preparation/DAPO-17K/1.5B/sampled/sample_20.json \
    --output_dir data/1.5B/ppo-n/number \
    --num_samples 3000 \
    --mode ppo-n

python src/ppo_sft/process_critic_sft_data.py \
    --sveb_data data/1.5B/sveb_math/train.json \
    --rollout_data \
        src/data_preparation/openr1-220K/1.5B/sampled/sample_20.json \
    --output_dir data/1.5B/ppo-n/math \
    --num_samples 3000 \
    --mode ppo-n

python src/ppo_sft/process_critic_sft_data.py \
    --sveb_data data/1.5B/sveb_science/train.json \
    --rollout_data \
        src/data_preparation/MixtureOfThought/1.5B/sampled/sample_20.json \
    --output_dir data/1.5B/ppo-n/science \
    --num_samples 3000 \
    --mode ppo-n

python src/ppo_sft/process_critic_sft_data.py \
    --sveb_data data/1.5B/sveb_general/train.json \
    --rollout_data \
        src/data_preparation/GeneralQA/1.5B/sampled/sample_20.json \
    --output_dir data/1.5B/ppo-n/general \
    --num_samples 3000 \
    --mode ppo-n

python src/ppo_sft/process_critic_sft_data.py \
    --sveb_data data/1.5B/sveb_program/train.json \
    --rollout_data \
        src/data_preparation/verifiable_python/1.5B/sampled/sample_20.json \
    --output_dir data/1.5B/ppo-n/program \
    --num_samples 1000 \
    --mode ppo-n

# PPO-1 data preparation for Qwen2.5-1.5B-Instruct
python src/ppo_sft/process_critic_sft_data.py \
    --sveb_data data/1.5B/sveb_number/train.json \
    --rollout_data \
        src/data_preparation/DAPO-17K/1.5B/sampled/sample_20.json \
    --output_dir data/1.5B/ppo-1/number \
    --num_samples 3000 \
    --mode ppo-1

python src/ppo_sft/process_critic_sft_data.py \
    --sveb_data data/1.5B/sveb_math/train.json \
    --rollout_data \
        src/data_preparation/openr1-220K/1.5B/sampled/sample_20.json \
    --output_dir data/1.5B/ppo-1/math \
    --num_samples 3000 \
    --mode ppo-1

python src/ppo_sft/process_critic_sft_data.py \
    --sveb_data data/1.5B/sveb_science/train.json \
    --rollout_data \
        src/data_preparation/MixtureOfThought/1.5B/sampled/sample_20.json \
    --output_dir data/1.5B/ppo-1/science \
    --num_samples 3000 \
    --mode ppo-1

python src/ppo_sft/process_critic_sft_data.py \
    --sveb_data data/1.5B/sveb_general/train.json \
    --rollout_data \
        src/data_preparation/GeneralQA/1.5B/sampled/sample_20.json \
    --output_dir data/1.5B/ppo-1/general \
    --num_samples 3000 \
    --mode ppo-1

python src/ppo_sft/process_critic_sft_data.py \
    --sveb_data data/1.5B/sveb_program/train.json \
    --rollout_data \
        src/data_preparation/verifiable_python/1.5B/sampled/sample_20.json \
    --output_dir data/1.5B/ppo-1/program \
    --num_samples 1000 \
    --mode ppo-1