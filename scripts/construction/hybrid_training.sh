# Hybrid training data preparation script
## 1.5B
python src/data_preparation/merge_for_training.py \
    --input_files   src/data_preparation/DAPO-17K/1.5B/sampled/0717/train.json \
                    src/data_preparation/DAPO-17K/1.5B/sampled/1737/train.json \
                    src/data_preparation/DAPO-17K/1.5B/sampled/3777/train.json \
                    src/data_preparation/openr1-220K/1.5B/sampled/0717/train.json \
                    src/data_preparation/openr1-220K/1.5B/sampled/1737/train.json \
                    src/data_preparation/openr1-220K/1.5B/sampled/3777/train.json \
                    src/data_preparation/MixtureOfThought/1.5B/sampled/0717/train.json \
                    src/data_preparation/MixtureOfThought/1.5B/sampled/1737/train.json \
                    src/data_preparation/MixtureOfThought/1.5B/sampled/3777/train.json \
                    src/data_preparation/GeneralQA/1.5B/sampled/0717/train.json \
                    src/data_preparation/GeneralQA/1.5B/sampled/1737/train.json \
                    src/data_preparation/GeneralQA/1.5B/sampled/3777/train.json \
                    src/data_preparation/verifiable_python/1.5B/sampled/0007/train.json \
                    src/data_preparation/verifiable_python/1.5B/sampled/0717/train.json \
                    src/data_preparation/verifiable_python/1.5B/sampled/1737/train.json \
                    src/data_preparation/verifiable_python/1.5B/sampled/3777/train.json \
    --num_samples   1000 1000 500 1000 1000 1000 1000 1000 1000 1000 1000 1000 903 654 398 336 \
    --output_folder   data/1.5B/hybrid/

## 3B
python src/data_preparation/merge_for_training.py \
    --input_files   src/data_preparation/DAPO-17K/3B/sampled/0717/train.json \
                    src/data_preparation/DAPO-17K/3B/sampled/1737/train.json \
                    src/data_preparation/DAPO-17K/3B/sampled/3777/train.json \
                    src/data_preparation/openr1-220K/3B/sampled/0717/train.json \
                    src/data_preparation/openr1-220K/3B/sampled/1737/train.json \
                    src/data_preparation/openr1-220K/3B/sampled/3777/train.json \
                    src/data_preparation/MixtureOfThought/3B/sampled/0717/train.json \
                    src/data_preparation/MixtureOfThought/3B/sampled/1737/train.json \
                    src/data_preparation/MixtureOfThought/3B/sampled/3777/train.json \
                    src/data_preparation/GeneralQA/3B/sampled/0717/train.json \
                    src/data_preparation/GeneralQA/3B/sampled/1737/train.json \
                    src/data_preparation/GeneralQA/3B/sampled/3777/train.json \
                    src/data_preparation/verifiable_python/3B/sampled/0007/train.json \
                    src/data_preparation/verifiable_python/3B/sampled/0717/train.json \
                    src/data_preparation/verifiable_python/3B/sampled/1737/train.json \
                    src/data_preparation/verifiable_python/3B/sampled/3777/train.json \
    --num_samples   1000 1000 500 1000 1000 500 1000 1000 1000 1000 1000 1000 600 600 600 600 \
    --output_folder   data/3B/hybrid/

## 7B
python src/data_preparation/merge_for_training.py \
    --input_files   src/data_preparation/DAPO-17K/7B/sampled/0717/train.json \
                    src/data_preparation/DAPO-17K/7B/sampled/1737/train.json \
                    src/data_preparation/DAPO-17K/7B/sampled/3777/train.json \
                    src/data_preparation/openr1-220K/7B/sampled/0717/train.json \
                    src/data_preparation/openr1-220K/7B/sampled/1737/train.json \
                    src/data_preparation/openr1-220K/7B/sampled/3777/train.json \
                    src/data_preparation/MixtureOfThought/7B/sampled/0717/train.json \
                    src/data_preparation/MixtureOfThought/7B/sampled/1737/train.json \
                    src/data_preparation/MixtureOfThought/7B/sampled/3777/train.json \
                    src/data_preparation/GeneralQA/7B/sampled/0717/train.json \
                    src/data_preparation/GeneralQA/7B/sampled/1737/train.json \
                    src/data_preparation/GeneralQA/7B/sampled/3777/train.json \
                    src/data_preparation/verifiable_python/7B/sampled/0007/train.json \
                    src/data_preparation/verifiable_python/7B/sampled/0717/train.json \
                    src/data_preparation/verifiable_python/7B/sampled/1737/train.json \
                    src/data_preparation/verifiable_python/7B/sampled/3777/train.json \
    --num_samples   1000 1000 500 1000 1000 500 1000 1000 1000 1000 1000 1000 600 600 600 600 \
    --output_folder   data/7B/hybrid/

## 14B
python src/data_preparation/merge_for_training.py \
    --input_files   src/data_preparation/DAPO-17K/7B/sampled/0007/train.json \
                    src/data_preparation/DAPO-17K/7B/sampled/0717/train.json \
                    src/data_preparation/DAPO-17K/7B/sampled/1737/train.json \
                    src/data_preparation/openr1-220K/7B/sampled/0007/train.json \
                    src/data_preparation/openr1-220K/7B/sampled/0717/train.json \
                    src/data_preparation/openr1-220K/7B/sampled/1737/train.json \
                    src/data_preparation/MixtureOfThought/7B/sampled/0007/train.json \
                    src/data_preparation/MixtureOfThought/7B/sampled/0717/train.json \
                    src/data_preparation/MixtureOfThought/7B/sampled/1737/train.json \
                    src/data_preparation/GeneralQA/7B/sampled/0007/train.json \
                    src/data_preparation/GeneralQA/7B/sampled/0717/train.json \
                    src/data_preparation/GeneralQA/7B/sampled/1737/train.json \
                    src/data_preparation/verifiable_python/7B/sampled/0007/train.json \
                    src/data_preparation/verifiable_python/7B/sampled/0717/train.json \
                    src/data_preparation/verifiable_python/7B/sampled/1737/train.json \
                    src/data_preparation/verifiable_python/7B/sampled/3777/train.json \
    --num_samples   1000 1000 500 1000 1000 500 1000 1000 1000 1000 1000 1000 1000 1000 400 400 \
    --output_folder   data/14B/hybrid/