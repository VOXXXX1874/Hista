# DAPO-17K
## Merge the three chunks 
python src/data_preparation/merge_samples.py \
    --input_files src/data_preparation/DAPO-17K/1.5B/partition_1/sample_20.json src/data_preparation/DAPO-17K/1.5B/partition_2/sample_20.json src/data_preparation/DAPO-17K/1.5B/partition_3/sample_20.json \
    --output_file src/data_preparation/DAPO-17K/1.5B/sampled/sample_20.json
## Select the data from different difficulty tiers
python src/data_preparation/select_samples.py --data_path src/data_preparation/DAPO-17K/1.5B/sampled/sample_20.json --output_path src/data_preparation/DAPO-17K/1.5B/sampled/0007/train.json --upper 0.07 --lower 0.00
python src/data_preparation/select_samples.py --data_path src/data_preparation/DAPO-17K/1.5B/sampled/sample_20.json --output_path src/data_preparation/DAPO-17K/1.5B/sampled/0717/train.json --upper 0.17 --lower 0.07
python src/data_preparation/select_samples.py --data_path src/data_preparation/DAPO-17K/1.5B/sampled/sample_20.json --output_path src/data_preparation/DAPO-17K/1.5B/sampled/1737/train.json --upper 0.37 --lower 0.17
python src/data_preparation/select_samples.py --data_path src/data_preparation/DAPO-17K/1.5B/sampled/sample_20.json --output_path src/data_preparation/DAPO-17K/1.5B/sampled/3777/train.json --upper 0.77 --lower 0.37
# (Optional) Select the data with enough unique number to run numca algorithm
python src/data_preparation/select_samples.py --data_path src/data_preparation/DAPO-17K/1.5B/sampled/sample_20.json --output_path src/data_preparation/DAPO-17K/1.5B/sampled_numca/0007/train.json --upper 0.07 --lower 0.00 --targets_num 4
python src/data_preparation/select_samples.py --data_path src/data_preparation/DAPO-17K/1.5B/sampled/sample_20.json --output_path src/data_preparation/DAPO-17K/1.5B/sampled_numca/0717/train.json --upper 0.17 --lower 0.07 --targets_num 4
python src/data_preparation/select_samples.py --data_path src/data_preparation/DAPO-17K/1.5B/sampled/sample_20.json --output_path src/data_preparation/DAPO-17K/1.5B/sampled_numca/1737/train.json --upper 0.37 --lower 0.17 --targets_num 4
python src/data_preparation/select_samples.py --data_path src/data_preparation/DAPO-17K/1.5B/sampled/sample_20.json --output_path src/data_preparation/DAPO-17K/1.5B/sampled_numca/3777/train.json --upper 0.77 --lower 0.37 --targets_num 4


# OpenR1-220k
## Merge the three chunks 
python src/data_preparation/merge_samples.py \
    --input_files src/data_preparation/openr1-220K/1.5B/partition_1/sample_20.json src/data_preparation/openr1-220K/1.5B/partition_2/sample_20.json src/data_preparation/openr1-220K/1.5B/partition_3/sample_20.json \
    --output_file src/data_preparation/openr1-220K/1.5B/sampled/sample_20.json
## Select the data from different difficulty tiers
python src/data_preparation/select_samples.py --data_path src/data_preparation/openr1-220K/1.5B/sampled/sample_20.json --output_path src/data_preparation/openr1-220K/1.5B/sampled/0007/train.json --upper 0.07 --lower 0.00
python src/data_preparation/select_samples.py --data_path src/data_preparation/openr1-220K/1.5B/sampled/sample_20.json --output_path src/data_preparation/openr1-220K/1.5B/sampled/0717/train.json --upper 0.17 --lower 0.07
python src/data_preparation/select_samples.py --data_path src/data_preparation/openr1-220K/1.5B/sampled/sample_20.json --output_path src/data_preparation/openr1-220K/1.5B/sampled/1737/train.json --upper 0.37 --lower 0.17
python src/data_preparation/select_samples.py --data_path src/data_preparation/openr1-220K/1.5B/sampled/sample_20.json --output_path src/data_preparation/openr1-220K/1.5B/sampled/3777/train.json --upper 0.77 --lower 0.37
# (Optional) Select the data with enough unique number to run numca algorithm
python src/data_preparation/select_samples.py --data_path src/data_preparation/openr1-220K/1.5B/sampled/sample_20.json --output_path src/data_preparation/openr1-220K/1.5B/sampled_numca/0007/train.json --upper 0.07 --lower 0.00 --targets_num 4
python src/data_preparation/select_samples.py --data_path src/data_preparation/openr1-220K/1.5B/sampled/sample_20.json --output_path src/data_preparation/openr1-220K/1.5B/sampled_numca/0717/train.json --upper 0.17 --lower 0.07 --targets_num 4
python src/data_preparation/select_samples.py --data_path src/data_preparation/openr1-220K/1.5B/sampled/sample_20.json --output_path src/data_preparation/openr1-220K/1.5B/sampled_numca/1737/train.json --upper 0.37 --lower 0.17 --targets_num 4
python src/data_preparation/select_samples.py --data_path src/data_preparation/openr1-220K/1.5B/sampled/sample_20.json --output_path src/data_preparation/openr1-220K/1.5B/sampled_numca/3777/train.json --upper 0.77 --lower 0.37 --targets_num 4


# Mixture of Thought
python src/data_preparation/merge_samples.py \
    --input_files src/data_preparation/MixtureOfThought/1.5B/partition_1/sample_20.json src/data_preparation/MixtureOfThought/1.5B/partition_2/sample_20.json src/data_preparation/MixtureOfThought/1.5B/partition_3/sample_20.json \
    --output_file src/data_preparation/MixtureOfThought/1.5B/sampled/sample_20.json
## Select the data from different difficulty tiers
python src/data_preparation/select_samples.py --data_path src/data_preparation/MixtureOfThought/1.5B/sampled/sample_20.json --output_path src/data_preparation/MixtureOfThought/1.5B/sampled/0007/train.json --upper 0.07 --lower 0.00
python src/data_preparation/select_samples.py --data_path src/data_preparation/MixtureOfThought/1.5B/sampled/sample_20.json --output_path src/data_preparation/MixtureOfThought/1.5B/sampled/0717/train.json --upper 0.17 --lower 0.07
python src/data_preparation/select_samples.py --data_path src/data_preparation/MixtureOfThought/1.5B/sampled/sample_20.json --output_path src/data_preparation/MixtureOfThought/1.5B/sampled/1737/train.json --upper 0.37 --lower 0.17
python src/data_preparation/select_samples.py --data_path src/data_preparation/MixtureOfThought/1.5B/sampled/sample_20.json --output_path src/data_preparation/MixtureOfThought/1.5B/sampled/3777/train.json --upper 0.77 --lower 0.37


# General QA
python src/data_preparation/merge_samples.py \
    --input_files src/data_preparation/GeneralQA/1.5B/partition_1/sample_20.json src/data_preparation/GeneralQA/1.5B/partition_2/sample_20.json src/data_preparation/GeneralQA/1.5B/partition_3/sample_20.json \
    --output_file src/data_preparation/GeneralQA/1.5B/sampled/sample_20.json
## Select the data from different difficulty tiers
python src/data_preparation/select_samples.py --data_path src/data_preparation/GeneralQA/1.5B/sampled/sample_20.json --output_path src/data_preparation/GeneralQA/1.5B/sampled/0007/train.json --upper 0.07 --lower 0.00
python src/data_preparation/select_samples.py --data_path src/data_preparation/GeneralQA/1.5B/sampled/sample_20.json --output_path src/data_preparation/GeneralQA/1.5B/sampled/0717/train.json --upper 0.17 --lower 0.07
python src/data_preparation/select_samples.py --data_path src/data_preparation/GeneralQA/1.5B/sampled/sample_20.json --output_path src/data_preparation/GeneralQA/1.5B/sampled/1737/train.json --upper 0.37 --lower 0.17
python src/data_preparation/select_samples.py --data_path src/data_preparation/GeneralQA/1.5B/sampled/sample_20.json --output_path src/data_preparation/GeneralQA/1.5B/sampled/3777/train.json --upper 0.77 --lower 0.37


# Verifiable Python
python src/data_preparation/merge_samples.py \
    --input_files src/data_preparation/verifiable_python/1.5B/partition_1/sample_20.json src/data_preparation/verifiable_python/1.5B/partition_2/sample_20.json src/data_preparation/verifiable_python/1.5B/partition_3/sample_20.json \
    --output_file src/data_preparation/verifiable_python/1.5B/sampled/sample_20.json
## Select the data from different difficulty tiers
python src/data_preparation/select_samples.py --data_path src/data_preparation/verifiable_python/1.5B/sampled/sample_20.json --output_path src/data_preparation/verifiable_python/1.5B/sampled/0007/train.json --upper 0.07 --lower 0.00
python src/data_preparation/select_samples.py --data_path src/data_preparation/verifiable_python/1.5B/sampled/sample_20.json --output_path src/data_preparation/verifiable_python/1.5B/sampled/0717/train.json --upper 0.17 --lower 0.07
python src/data_preparation/select_samples.py --data_path src/data_preparation/verifiable_python/1.5B/sampled/sample_20.json --output_path src/data_preparation/verifiable_python/1.5B/sampled/1737/train.json --upper 0.37 --lower 0.17
python src/data_preparation/select_samples.py --data_path src/data_preparation/verifiable_python/1.5B/sampled/sample_20.json --output_path src/data_preparation/verifiable_python/1.5B/sampled/3777/train.json --upper 0.77 --lower 0.37


