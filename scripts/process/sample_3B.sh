# Please modify the CUDA_VISIBLE_DEVICES and other arguments accordingly
# Do sampling for DAPO dataest
CUDA_VISIBLE_DEVICES=0 python src/data_preparation/extra_sampling.py --model_name Qwen/Qwen2.5-3B-Instruct --dataset_path src/data_preparation/DAPO-17K/partition_1 --output_path src/data_preparation/DAPO-17K/3B/partition_1/sample_20.json --num 20 --max_length 4096 --tp 1 &
CUDA_VISIBLE_DEVICES=1 python src/data_preparation/extra_sampling.py --model_name Qwen/Qwen2.5-3B-Instruct --dataset_path src/data_preparation/DAPO-17K/partition_2 --output_path src/data_preparation/DAPO-17K/3B/partition_2/sample_20.json --num 20 --max_length 4096 --tp 1 &
CUDA_VISIBLE_DEVICES=2 python src/data_preparation/extra_sampling.py --model_name Qwen/Qwen2.5-3B-Instruct --dataset_path src/data_preparation/DAPO-17K/partition_3 --output_path src/data_preparation/DAPO-17K/3B/partition_3/sample_20.json --num 20 --max_length 4096 --tp 1 &

# Do sampling for OpenR1 dataset
CUDA_VISIBLE_DEVICES=0 python src/data_preparation/extra_sampling.py --model_name Qwen/Qwen2.5-3B-Instruct --dataset_path src/data_preparation/openr1-220K/partition_1 --output_path src/data_preparation/openr1-220K/3B/partition_1/sample_20.json --num 20 --max_length 4096 --tp 1 --proxy_url http://localhost:8000
CUDA_VISIBLE_DEVICES=0 python src/data_preparation/extra_sampling.py --model_name Qwen/Qwen2.5-3B-Instruct --dataset_path src/data_preparation/openr1-220K/partition_2 --output_path src/data_preparation/openr1-220K/3B/partition_2/sample_20.json --num 20 --max_length 4096 --tp 1 --proxy_url http://localhost:8000
CUDA_VISIBLE_DEVICES=0 python src/data_preparation/extra_sampling.py --model_name Qwen/Qwen2.5-3B-Instruct --dataset_path src/data_preparation/openr1-220K/partition_3 --output_path src/data_preparation/openr1-220K/3B/partition_3/sample_20.json --num 20 --max_length 4096 --tp 1 --proxy_url http://localhost:8000

# Do sampling for Science dataset
CUDA_VISIBLE_DEVICES=0 python src/data_preparation/extra_sampling.py --model_name Qwen/Qwen2.5-3B-Instruct --dataset_path src/data_preparation/MixtureOfThought/partition_1 --output_path src/data_preparation/MixtureOfThought/3B/partition_1/sample_20.json --num 20 --max_length 4096 --tp 1 --proxy_url http://localhost:8000
CUDA_VISIBLE_DEVICES=0 python src/data_preparation/extra_sampling.py --model_name Qwen/Qwen2.5-3B-Instruct --dataset_path src/data_preparation/MixtureOfThought/partition_2 --output_path src/data_preparation/MixtureOfThought/3B/partition_2/sample_20.json --num 20 --max_length 4096 --tp 1 --proxy_url http://localhost:8000
CUDA_VISIBLE_DEVICES=0 python src/data_preparation/extra_sampling.py --model_name Qwen/Qwen2.5-3B-Instruct --dataset_path src/data_preparation/MixtureOfThought/partition_3 --output_path src/data_preparation/MixtureOfThought/3B/partition_3/sample_20.json --num 20 --max_length 4096 --tp 1 --proxy_url http://localhost:8000

# Do sampling for General QA dataset
CUDA_VISIBLE_DEVICES=0 python src/data_preparation/extra_sampling.py --model_name Qwen/Qwen2.5-3B-Instruct --dataset_path src/data_preparation/GeneralQA/partition_1 --output_path src/data_preparation/GeneralQA/3B/partition_1/sample_20.json --num 20 --max_length 4096 --tp 1 --proxy_url http://localhost:8000
CUDA_VISIBLE_DEVICES=0 python src/data_preparation/extra_sampling.py --model_name Qwen/Qwen2.5-3B-Instruct --dataset_path src/data_preparation/GeneralQA/partition_2 --output_path src/data_preparation/GeneralQA/3B/partition_2/sample_20.json --num 20 --max_length 4096 --tp 1 --proxy_url http://localhost:8000
CUDA_VISIBLE_DEVICES=0 python src/data_preparation/extra_sampling.py --model_name Qwen/Qwen2.5-3B-Instruct --dataset_path src/data_preparation/GeneralQA/partition_3 --output_path src/data_preparation/GeneralQA/3B/partition_3/sample_20.json --num 20 --max_length 4096 --tp 1 --proxy_url http://localhost:8000

# Do sampling for Verifiable Python dataset
CUDA_VISIBLE_DEVICES=0 python src/data_preparation/extra_sampling.py --model_name Qwen/Qwen2.5-3B-Instruct --dataset_path src/data_preparation/verifiable_python/partition_1 --output_path src/data_preparation/verifiable_python/3B/partition_1/sample_20.json --num 20 --max_length 4096 --tp 1
CUDA_VISIBLE_DEVICES=0 python src/data_preparation/extra_sampling.py --model_name Qwen/Qwen2.5-3B-Instruct --dataset_path src/data_preparation/verifiable_python/partition_2 --output_path src/data_preparation/verifiable_python/3B/partition_2/sample_20.json --num 20 --max_length 4096 --tp 1
CUDA_VISIBLE_DEVICES=0 python src/data_preparation/extra_sampling.py --model_name Qwen/Qwen2.5-3B-Instruct --dataset_path src/data_preparation/verifiable_python/partition_3 --output_path src/data_preparation/verifiable_python/3B/partition_3/sample_20.json --num 20 --max_length 4096 --tp 1