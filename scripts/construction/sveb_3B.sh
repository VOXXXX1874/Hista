#sveb_number
python src/data_preparation/merge_for_training.py \
    --input_files   src/data_preparation/DAPO-17K/3B/sampled_numca/0717/train.json \
                    src/data_preparation/DAPO-17K/3B/sampled_numca/1737/train.json \
                    src/data_preparation/DAPO-17K/3B/sampled_numca/3777/train.json \
                    src/data_preparation/openr1-220K/3B/sampled_numca/0717/train.json \
                    src/data_preparation/openr1-220K/3B/sampled_numca/1737/train.json \
                    src/data_preparation/openr1-220K/3B/sampled_numca/3777/train.json \
    --num_samples   500 500 500 500 500 500 \
    --output_folder   data/3B/sveb_number/ \
    --exclude_test

#sveb_math
python src/data_preparation/merge_for_training.py \
    --input_files   src/data_preparation/DAPO-17K/3B/sampled/0717/train.json \
                    src/data_preparation/DAPO-17K/3B/sampled/1737/train.json \
                    src/data_preparation/DAPO-17K/3B/sampled/3777/train.json \
                    src/data_preparation/openr1-220K/3B/sampled/0717/train.json \
                    src/data_preparation/openr1-220K/3B/sampled/1737/train.json \
                    src/data_preparation/openr1-220K/3B/sampled/3777/train.json \
    --num_samples   500 500 500 500 500 500 \
    --output_folder   data/3B/sveb_math/ \
    --exclude_test

# sveb_science
python src/data_preparation/merge_for_training.py \
    --input_files   src/data_preparation/MixtureOfThought/3B/sampled/0717/train.json \
                    src/data_preparation/MixtureOfThought/3B/sampled/1737/train.json \
                    src/data_preparation/MixtureOfThought/3B/sampled/3777/train.json \
    --num_samples   1000 1000 1000 \
    --output_folder data/3B/sveb_science/ \
    --exclude_test

# sveb_general
python src/data_preparation/merge_for_training.py \
    --input_files   src/data_preparation/GeneralQA/3B/sampled/0717/train.json \
                    src/data_preparation/GeneralQA/3B/sampled/1737/train.json \
                    src/data_preparation/GeneralQA/3B/sampled/3777/train.json \
    --num_samples   1000 1000 1000 \
    --output_folder data/3B/sveb_general/ \
    --exclude_test

# sveb_program
python src/data_preparation/merge_for_training.py \
    --input_files   src/data_preparation/verifiable_python/3B/sampled/0007/train.json \
                    src/data_preparation/verifiable_python/3B/sampled/0717/train.json \
                    src/data_preparation/verifiable_python/3B/sampled/1737/train.json \
                    src/data_preparation/verifiable_python/3B/sampled/3777/train.json \
    --num_samples   750 750 800 700 \
    --output_folder data/3B/sveb_program/ \
    --exclude_test