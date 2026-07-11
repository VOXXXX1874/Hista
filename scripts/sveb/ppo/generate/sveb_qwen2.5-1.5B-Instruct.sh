# Assume the trained critic and value head is located at output/Qwen2.5-1.5B-Critic-SFT-{domain}/checkpoint-100
python src/sveb/ppo/evaluate_sta_estim_generate.py \
 	--action_model_name Qwen/Qwen2.5-1.5B-Instruct \
	--critic_model_path output/Qwen2.5-1.5B-Critic-SFT-Number/checkpoint-100 \
	--value_head_path output/Qwen2.5-1.5B-Critic-SFT-Number/checkpoint-100 \
	--dataset_path data/1.5B/sveb_number/train.json \
	--output_path output/sveb/Qwen2.5-1.5B-Instruct/ppo_sveb_number_S32_MCS20.log \
	--save_path output/sveb/Qwen2.5-1.5B-Instruct/sveb_number_S32_MCS20_NumPro3000.json \
	--num_of_problems 3000

python src/sveb/ppo/evaluate_sta_estim_generate.py \
 	--action_model_name Qwen/Qwen2.5-1.5B-Instruct \
	--critic_model_path output/Qwen2.5-1.5B-Critic-SFT-Math/checkpoint-100 \
	--value_head_path output/Qwen2.5-1.5B-Critic-SFT-Math/checkpoint-100 \
	--dataset_path data/1.5B/sveb_math/train.json \
	--output_path output/sveb/Qwen2.5-1.5B-Instruct/ppo_sveb_math_S32_MCS20.log \
	--save_path output/sveb/Qwen2.5-1.5B-Instruct/sveb_math_S32_MCS20_NumPro3000.json \
	--num_of_problems 3000

python src/sveb/ppo/evaluate_sta_estim_generate.py \
 	--action_model_name Qwen/Qwen2.5-1.5B-Instruct \
	--critic_model_path output/Qwen2.5-1.5B-Critic-SFT-Science/checkpoint-100 \
	--value_head_path output/Qwen2.5-1.5B-Critic-SFT-Science/checkpoint-100 \
	--dataset_path data/1.5B/sveb_science/train.json \
	--output_path output/sveb/Qwen2.5-1.5B-Instruct/ppo_sveb_science_S32_MCS20.log \
	--save_path output/sveb/Qwen2.5-1.5B-Instruct/sveb_science_S32_MCS20_NumPro3000.json \
	--num_of_problems 3000

python src/sveb/ppo/evaluate_sta_estim_generate.py \
 	--action_model_name Qwen/Qwen2.5-1.5B-Instruct \
	--critic_model_path output/Qwen2.5-1.5B-Critic-SFT-General/checkpoint-100 \
	--value_head_path output/Qwen2.5-1.5B-Critic-SFT-General/checkpoint-100 \
	--dataset_path data/1.5B/sveb_general/train.json \
	--output_path output/sveb/Qwen2.5-1.5B-Instruct/ppo_sveb_general_S32_MCS20.log \
	--save_path output/sveb/Qwen2.5-1.5B-Instruct/sveb_general_S32_MCS20_NumPro3000.json \
	--num_of_problems 3000

python src/sveb/ppo/evaluate_sta_estim_generate.py \
 	--action_model_name Qwen/Qwen2.5-1.5B-Instruct \
	--critic_model_path output/Qwen2.5-1.5B-Critic-SFT-Program/checkpoint-100 \
	--value_head_path output/Qwen2.5-1.5B-Critic-SFT-Program/checkpoint-100 \
	--dataset_path data/1.5B/sveb_program/train.json \
	--output_path output/sveb/Qwen2.5-1.5B-Instruct/ppo_sveb_program_S32_MCS20.log \
	--save_path output/sveb/Qwen2.5-1.5B-Instruct/sveb_program_S32_MCS20_NumPro3000.json \
	--num_of_problems 3000