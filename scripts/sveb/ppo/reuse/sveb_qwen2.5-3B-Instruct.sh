# Assume the trained critic and value head is located at output/Qwen2.5-3B/Critic_on_{domain}/checkpoint-400
python src/sveb/ppo/evaluate_sta_estim_from_existing.py \
 	--action_model_name Qwen/Qwen2.5-3B-Instruct \
	--critic_model_path output/Qwen2.5-3B/Critic_on_number/checkpoint-400 \
	--value_head_path output/Qwen2.5-3B/Critic_on_number/checkpoint-400 \
	--dataset_path output/sveb/Qwen2.5-3B-Instruct/sveb_number_S32_MCS20_NumPro3000.json \
	--output_path output/sveb/Qwen2.5-3B-Instruct/ppo_sveb_number_S32_MCS20.log \
	--num_of_problems 3000

python src/sveb/ppo/evaluate_sta_estim_from_existing.py \
 	--action_model_name Qwen/Qwen2.5-3B-Instruct \
	--critic_model_path output/Qwen2.5-3B/Critic_on_math/checkpoint-400 \
	--value_head_path output/Qwen2.5-3B/Critic_on_math/checkpoint-400 \
	--dataset_path output/sveb/Qwen2.5-3B-Instruct/sveb_math_S32_MCS20_NumPro3000.json \
	--output_path output/sveb/Qwen2.5-3B-Instruct/ppo_sveb_math_S32_MCS20.log \
	--num_of_problems 3000

python src/sveb/ppo/evaluate_sta_estim_from_existing.py \
 	--action_model_name Qwen/Qwen2.5-3B-Instruct \
	--critic_model_path output/Qwen2.5-3B/Critic_on_science/checkpoint-400 \
	--value_head_path output/Qwen2.5-3B/Critic_on_science/checkpoint-400 \
	--dataset_path output/sveb/Qwen2.5-3B-Instruct/sveb_science_S32_MCS20_NumPro3000.json \
	--output_path output/sveb/Qwen2.5-3B-Instruct/ppo_sveb_science_S32_MCS20.log \
	--num_of_problems 3000

python src/sveb/ppo/evaluate_sta_estim_from_existing.py \
 	--action_model_name Qwen/Qwen2.5-3B-Instruct \
	--critic_model_path output/Qwen2.5-3B/Critic_on_general/checkpoint-400 \
	--value_head_path output/Qwen2.5-3B/Critic_on_general/checkpoint-400 \
	--dataset_path output/sveb/Qwen2.5-3B-Instruct/sveb_general_S32_MCS20_NumPro3000.json \
	--output_path output/sveb/Qwen2.5-3B-Instruct/ppo_sveb_general_S32_MCS20.log \
	--num_of_problems 3000

python src/sveb/ppo/evaluate_sta_estim_from_existing.py \
 	--action_model_name Qwen/Qwen2.5-3B-Instruct \
	--critic_model_path output/Qwen2.5-3B/Critic_on_program/checkpoint-400 \
	--value_head_path output/Qwen2.5-3B/Critic_on_program/checkpoint-400 \
	--dataset_path output/sveb/Qwen2.5-3B-Instruct/sveb_program_S32_MCS20_NumPro3000.json \
	--output_path output/sveb/Qwen2.5-3B-Instruct/ppo_sveb_program_S32_MCS20.log \
	--num_of_problems 3000