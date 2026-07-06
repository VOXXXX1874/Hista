#SVEB-NUMBER
python src/data_preparation/merge_training.py \
    --input_files   src/data_preparation/DAPO-17K/1.5B/sampled_numca/0717/train.json \
                    src/data_preparation/DAPO-17K/1.5B/sampled_numca/1737/train.json \
                    src/data_preparation/DAPO-17K/1.5B/sampled_numca/3777/train.json \
                    src/data_preparation/openr1-220K/1.5B/sampled_numca/0717/train.json \
                    src/data_preparation/openr1-220K/1.5B/sampled_numca/1737/train.json \
                    src/data_preparation/openr1-220K/1.5B/sampled_numca/3777/train.json \
    --num_samples   500 500 500 500 500 500 \
    --output_path   data/1.5B/SVEB-NUMBER/ \
    --exclude_test

#SVEB-MATH
python src/data_preparation/merge_training.py \
    --input_files   src/data_preparation/DAPO-17K/1.5B/sampled/0717/train.json \
                    src/data_preparation/DAPO-17K/1.5B/sampled/1737/train.json \
                    src/data_preparation/DAPO-17K/1.5B/sampled/3777/train.json \
                    src/data_preparation/openr1-220K/1.5B/sampled/0717/train.json \
                    src/data_preparation/openr1-220K/1.5B/sampled/1737/train.json \
                    src/data_preparation/openr1-220K/1.5B/sampled/3777/train.json \
    --num_samples   500 500 500 500 500 500 \
    --output_path   data/1.5B/SVEB-MATH/ \
    --exclude_test

# SVEB-SCIENCE
python src/data_preparation/merge_training.py \
    --input_files   src/data_preparation/MixtureOfThought/1.5B/sampled/0717/train.json \
                    src/data_preparation/MixtureOfThought/1.5B/sampled/1737/train.json \
                    src/data_preparation/MixtureOfThought/1.5B/sampled/3777/train.json \
    --num_samples   1000 1000 1000 \
    --output_folder data/1.5B/SVEB-SCIENCE/ \
    --exclude_test

# SVEB-GENERAL
python src/data_preparation/merge_training.py \
    --input_files   src/data_preparation/GeneralQA/1.5B/sampled/0717/train.json \
                    src/data_preparation/GeneralQA/1.5B/sampled/1737/train.json \
                    src/data_preparation/GeneralQA/1.5B/sampled/3777/train.json \
    --num_samples   1000 1000 1000 \
    --output_folder data/1.5B/SVEB-GENERAL/ \
    --exclude_test

# SVEB-PROGRAMMING
python src/data_preparation/merge_training.py \
    --input_files   src/data_preparation/verifiable_python/1.5B/sampled/0007/train.json \
                    src/data_preparation/verifiable_python/1.5B/sampled/0717/train.json \
                    src/data_preparation/verifiable_python/1.5B/sampled/1737/train.json \
                    src/data_preparation/verifiable_python/1.5B/sampled/3777/train.json \
    --num_samples   903 654 398 336 \
    --output_folder data/1.5B/SVEB-PROGRAMMING/ \
    --exclude_test