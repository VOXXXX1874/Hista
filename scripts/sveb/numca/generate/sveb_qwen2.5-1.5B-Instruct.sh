python src/sveb/numca/evaluate_sta_estim_generate.py \
	--model_name Qwen/Qwen2.5-1.5B-Instruct \
	--dataset_path data/1.5B/sveb_number/train.json \
	--output_path output/sveb/Qwen2.5-1.5B-Instruct/numca_sveb_number_S32_MCS20.log \
	--save_path output/sveb/Qwen2.5-1.5B-Instruct/sveb_number_S32_MCS20_NumPro3000.json \
	--grpo_num 32 \
	--mcs_num 20 \
	--max_length 4096 \
	--num_of_problems 3000 > tmp/sveb_number.log 2>&1 &

python src/sveb/numca/evaluate_sta_estim_generate.py \
	--model_name Qwen/Qwen2.5-1.5B-Instruct \
	--dataset_path data/1.5B/sveb_math/train.json \
	--output_path output/sveb/Qwen2.5-1.5B-Instruct/numca_sveb_math_S32_MCS20.log \
	--save_path output/sveb/Qwen2.5-1.5B-Instruct/sveb_math_S32_MCS20_NumPro3000.json \
	--grpo_num 32 \
	--mcs_num 20 \
	--max_length 4096 \
	--num_of_problems 3000 > tmp/sveb_math.log 2>&1 &

python src/sveb/numca/evaluate_sta_estim_generate.py \
	--model_name Qwen/Qwen2.5-1.5B-Instruct \
	--dataset_path data/1.5B/sveb_program/train.json \
	--output_path output/sveb/Qwen2.5-1.5B-Instruct/numca_sveb_program_S32_MCS20.log \
	--save_path output/sveb/Qwen2.5-1.5B-Instruct/sveb_program_S32_MCS20_NumPro3000.json \
	--grpo_num 32 \
	--mcs_num 20 \
	--max_length 4096 \
	--num_of_problems 3000 > tmp/sveb_program.log 2>&1 &

python src/sveb/numca/evaluate_sta_estim_generate.py \
	--model_name Qwen/Qwen2.5-1.5B-Instruct \
	--dataset_path data/1.5B/sveb_science/train.json \
	--output_path output/sveb/Qwen2.5-1.5B-Instruct/numca_sveb_science_S32_MCS20.log \
	--save_path output/sveb/Qwen2.5-1.5B-Instruct/sveb_science_S32_MCS20_NumPro3000.json \
	--grpo_num 32 \
	--mcs_num 20 \
	--max_length 4096 \
	--num_of_problems 3000 > tmp/sveb_science.log 2>&1 &

python src/sveb/numca/evaluate_sta_estim_generate.py \
	--model_name Qwen/Qwen2.5-1.5B-Instruct \
	--dataset_path data/1.5B/sveb_general/train.json \
	--output_path output/sveb/Qwen2.5-1.5B-Instruct/numca_sveb_general_S32_MCS20.log \
	--save_path output/sveb/Qwen2.5-1.5B-Instruct/sveb_general_S32_MCS20_NumPro3000.json \
	--grpo_num 32 \
	--mcs_num 20 \
	--max_length 4096 \
	--num_of_problems 3000 > tmp/sveb_general.log 2>&1 &