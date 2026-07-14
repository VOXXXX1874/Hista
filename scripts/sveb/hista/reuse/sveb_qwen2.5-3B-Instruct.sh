python src/sveb/hista/evaluate_sta_estim_from_existing.py \
	--model_name Qwen/Qwen2.5-3B-Instruct \
	--dataset_path output/sveb/Qwen2.5-3B-Instruct/sveb_number_S32_MCS20_NumPro3000.json \
	--output_path output/sveb/Qwen2.5-3B-Instruct/hista_sveb_number_S32_MCS20_t1_l1_max66_min6_m50_a07_mw5_mind5_uniform_ema_euclidean.log \
	--num_of_problems 3000 \
	--t 1 \
	--layer 1 \
    --max_k 66 \
    --min_k 6 \
	--min_interval 50 \
	--alpha 0.7 \
	--mean_window 5 \
	--min_distance 5 \
	--selection_method uniform \
	--average_method ema > tmp/sveb_number.log 2>&1 &

python src/sveb/hista/evaluate_sta_estim_from_existing.py \
	--model_name Qwen/Qwen2.5-3B-Instruct \
	--dataset_path output/sveb/Qwen2.5-3B-Instruct/sveb_math_S32_MCS20_NumPro3000.json \
	--output_path output/sveb/Qwen2.5-3B-Instruct/hista_sveb_math_S32_MCS20_t1_l1_max66_min6_m50_a07_mw5_mind5_uniform_ema_euclidean.log \
	--num_of_problems 3000 \
	--t 1 \
	--layer 1 \
    --max_k 66 \
    --min_k 6 \
	--min_interval 50 \
	--alpha 0.7 \
	--mean_window 5 \
	--min_distance 5 \
	--selection_method uniform \
	--average_method ema > tmp/sveb_math.log 2>&1 &

python src/sveb/hista/evaluate_sta_estim_from_existing.py \
	--model_name Qwen/Qwen2.5-3B-Instruct \
	--dataset_path output/sveb/Qwen2.5-3B-Instruct/sveb_program_S32_MCS20_NumPro3000.json \
	--output_path output/sveb/Qwen2.5-3B-Instruct/hista_sveb_program_S32_MCS20_t1_l1_max66_min6_m50_a07_mw5_mind5_uniform_ema_euclidean.log \
	--num_of_problems 3000 \
	--t 1 \
	--layer 1 \
    --max_k 66 \
    --min_k 6 \
	--min_interval 50 \
	--alpha 0.7 \
	--mean_window 5 \
	--min_distance 5 \
	--selection_method uniform \
	--average_method ema > tmp/sveb_program.log 2>&1 &

python src/sveb/hista/evaluate_sta_estim_from_existing.py \
	--model_name Qwen/Qwen2.5-3B-Instruct \
	--dataset_path output/sveb/Qwen2.5-3B-Instruct/sveb_science_S32_MCS20_NumPro3000.json \
	--output_path output/sveb/Qwen2.5-3B-Instruct/hista_sveb_science_S32_MCS20_t1_l1_max66_min6_m50_a07_mw5_mind5_uniform_ema_euclidean.log \
	--num_of_problems 3000 \
	--t 1 \
	--layer 1 \
    --max_k 66 \
    --min_k 6 \
	--min_interval 50 \
	--alpha 0.7 \
	--mean_window 5 \
	--min_distance 5 \
	--selection_method uniform \
	--average_method ema > tmp/sveb_science.log 2>&1 &

python src/sveb/hista/evaluate_sta_estim_from_existing.py \
	--model_name Qwen/Qwen2.5-3B-Instruct \
	--dataset_path output/sveb/Qwen2.5-3B-Instruct/sveb_general_S32_MCS20_NumPro3000.json \
	--output_path output/sveb/Qwen2.5-3B-Instruct/hista_sveb_general_S32_MCS20_t1_l1_max66_min6_m50_a07_mw5_mind5_uniform_ema_euclidean.log \
	--num_of_problems 3000 \
	--t 1 \
	--layer 1 \
    --max_k 66 \
    --min_k 6 \
	--min_interval 50 \
	--alpha 0.7 \
	--mean_window 5 \
	--min_distance 5 \
	--selection_method uniform \
	--average_method ema > tmp/sveb_general.log 2>&1 &