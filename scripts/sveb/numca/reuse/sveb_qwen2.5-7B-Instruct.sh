python src/sveb/numca/evaluate_sta_estim_from_existing.py \
	--dataset_path output/sveb/Qwen2.5-7B-Instruct/sveb_number_S32_MCS20_NumPro3000.json \
	--output_path output/sveb/Qwen2.5-7B-Instruct/numca_sveb_number_S32_MCS20.log \
	--num_of_problems 3000 > tmp/sveb_number.log 2>&1 &

python src/sveb/numca/evaluate_sta_estim_from_existing.py \
	--dataset_path output/sveb/Qwen2.5-7B-Instruct/sveb_math_S32_MCS20_NumPro3000.json \
	--output_path output/sveb/Qwen2.5-7B-Instruct/numca_sveb_math_S32_MCS20.log \
	--num_of_problems 3000 > tmp/sveb_math.log 2>&1 &

python src/sveb/numca/evaluate_sta_estim_from_existing.py \
	--dataset_path output/sveb/Qwen2.5-7B-Instruct/sveb_program_S32_MCS20_NumPro3000.json \
	--output_path output/sveb/Qwen2.5-7B-Instruct/numca_sveb_program_S32_MCS20.log \
	--num_of_problems 3000 > tmp/sveb_program.log 2>&1 &

python src/sveb/numca/evaluate_sta_estim_from_existing.py \
	--dataset_path output/sveb/Qwen2.5-7B-Instruct/sveb_science_S32_MCS20_NumPro3000.json \
	--output_path output/sveb/Qwen2.5-7B-Instruct/numca_sveb_science_S32_MCS20.log \
	--num_of_problems 3000 > tmp/sveb_science.log 2>&1 &

python src/sveb/numca/evaluate_sta_estim_from_existing.py \
	--dataset_path output/sveb/Qwen2.5-7B-Instruct/sveb_general_S32_MCS20_NumPro3000.json \
	--output_path output/sveb/Qwen2.5-7B-Instruct/numca_sveb_general_S32_MCS20.log \
	--num_of_problems 3000 > tmp/sveb_general.log 2>&1 &