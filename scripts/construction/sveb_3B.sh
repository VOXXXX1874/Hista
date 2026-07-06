#SVEB-NUMBER
python src/data_preparation/merge_training.py \
    --input_files   src/data_preparation/DAPO-17K/3B/sampled_numca/0717/train.json \
                    src/data_preparation/DAPO-17K/3B/sampled_numca/1737/train.json \
                    src/data_preparation/DAPO-17K/3B/sampled_numca/3777/train.json \
                    src/data_preparation/openr1-220K/3B/sampled_numca/0717/train.json \
                    src/data_preparation/openr1-220K/3B/sampled_numca/1737/train.json \
                    src/data_preparation/openr1-220K/3B/sampled_numca/3777/train.json \
    --num_samples   500 500 500 500 500 500 \
    --output_path   data/3B/SVEB-NUMBER/ \
    --exclude_test

#SVEB-MATH
python src/data_preparation/merge_training.py \
    --input_files   src/data_preparation/DAPO-17K/3B/sampled/0717/train.json \
                    src/data_preparation/DAPO-17K/3B/sampled/1737/train.json \
                    src/data_preparation/DAPO-17K/3B/sampled/3777/train.json \
                    src/data_preparation/openr1-220K/3B/sampled/0717/train.json \
                    src/data_preparation/openr1-220K/3B/sampled/1737/train.json \
                    src/data_preparation/openr1-220K/3B/sampled/3777/train.json \
    --num_samples   500 500 500 500 500 500 \
    --output_path   data/3B/SVEB-MATH/ \
    --exclude_test

# SVEB-SCIENCE
python src/data_preparation/merge_training.py \
    --input_files   src/data_preparation/MixtureOfThought/3B/sampled/0717/train.json \
                    src/data_preparation/MixtureOfThought/3B/sampled/1737/train.json \
                    src/data_preparation/MixtureOfThought/3B/sampled/3777/train.json \
    --num_samples   1000 1000 1000 \
    --output_folder data/3B/SVEB-SCIENCE/ \
    --exclude_test

# SVEB-GENERAL
python src/data_preparation/merge_training.py \
    --input_files   src/data_preparation/GeneralQA/3B/sampled/0717/train.json \
                    src/data_preparation/GeneralQA/3B/sampled/1737/train.json \
                    src/data_preparation/GeneralQA/3B/sampled/3777/train.json \
    --num_samples   1000 1000 1000 \
    --output_folder data/3B/SVEB-GENERAL/ \
    --exclude_test

# SVEB-PROGRAMMING
python src/data_preparation/merge_training.py \
    --input_files   src/data_preparation/verifiable_python/3B/sampled/0007/train.json \
                    src/data_preparation/verifiable_python/3B/sampled/0717/train.json \
                    src/data_preparation/verifiable_python/3B/sampled/1737/train.json \
                    src/data_preparation/verifiable_python/3B/sampled/3777/train.json \
    --num_samples   750 750 750 750 \
    --output_folder data/3B/SVEB-PROGRAMMING/ \
    --exclude_test