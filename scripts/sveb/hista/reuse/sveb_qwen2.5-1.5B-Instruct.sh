### normal base pure

python src/sveb/hista/evaluate_sta_estim_from_existing.py \
	--model_name Qwen/Qwen2.5-1.5B-Instruct \
	--dataset_path output/sveb/Qwen2.5-1.5B-Instruct/number_S32_MCS20_t1_l1_max66_min6_m50_a07_mw5_mind5_uniform_ema_euclidean.json \
	--output_path output/sveb/Qwen2.5-1.5B-Instruct/reuse_number_S32_MCS20_t1_l1_max66_min6_m50_a07_mw5_mind5_uniform_ema_euclidean.log \
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
	--model_name Qwen/Qwen2.5-1.5B-Instruct \
	--dataset_path output/sveb/Qwen2.5-1.5B-Instruct/math_S32_MCS20_t1_l1_max66_min6_m50_a07_mw5_mind5_uniform_ema_euclidean.json \
	--output_path output/sveb/Qwen2.5-1.5B-Instruct/reuse_math_S32_MCS20_t1_l1_max66_min6_m50_a07_mw5_mind5_uniform_ema_euclidean.log \
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
	--model_name Qwen/Qwen2.5-1.5B-Instruct \
	--dataset_path output/sveb/Qwen2.5-1.5B-Instruct/program_S32_MCS20_t1_l1_max66_min6_m50_a07_mw5_mind5_uniform_ema_euclidean.json \
	--output_path output/sveb/Qwen2.5-1.5B-Instruct/reuse_program_S32_MCS20_t1_l1_max66_min6_m50_a07_mw5_mind5_uniform_ema_euclidean.log \
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
	--model_name Qwen/Qwen2.5-1.5B-Instruct \
	--dataset_path output/sveb/Qwen2.5-1.5B-Instruct/program_S32_MCS20_t1_l1_max66_min6_m50_a07_mw5_mind5_uniform_ema_euclidean.json \
	--output_path output/sveb/Qwen2.5-1.5B-Instruct/reuse_program_S32_MCS20_t1_l1_max66_min6_m50_a07_mw5_mind5_uniform_ema_euclidean.log \
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
	--model_name Qwen/Qwen2.5-1.5B-Instruct \
	--dataset_path output/sveb/Qwen2.5-1.5B-Instruct/science_S32_MCS20_t1_l1_max66_min6_m50_a07_mw5_mind5_uniform_ema_euclidean.json \
	--output_path output/sveb/Qwen2.5-1.5B-Instruct/reuse_science_S32_MCS20_t1_l1_max66_min6_m50_a07_mw5_mind5_uniform_ema_euclidean.log \
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
	--model_name Qwen/Qwen2.5-1.5B-Instruct \
	--dataset_path output/sveb/Qwen2.5-1.5B-Instruct/general_S32_MCS20_t1_l1_max66_min6_m50_a07_mw5_mind5_uniform_ema_euclidean.json \
	--output_path output/sveb/Qwen2.5-1.5B-Instruct/reuse_general_S32_MCS20_t1_l1_max66_min6_m50_a07_mw5_mind5_uniform_ema_euclidean.log \
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