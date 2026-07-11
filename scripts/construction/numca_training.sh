# Numca algorithm training data preparation script
## 1.5B
python src/data_preparation/merge_for_training.py \
    --input_files   src/data_preparation/DAPO-17K/1.5B/sampled_numca/0717/train.json \
                    src/data_preparation/DAPO-17K/1.5B/sampled_numca/1737/train.json \
                    src/data_preparation/DAPO-17K/1.5B/sampled_numca/3777/train.json \
                    src/data_preparation/openr1-220K/1.5B/sampled_numca/0717/train.json \
                    src/data_preparation/openr1-220K/1.5B/sampled_numca/1737/train.json \
                    src/data_preparation/openr1-220K/1.5B/sampled_numca/3777/train.json \
    --num_samples   1000 1000 1000 1000 1000 1000 \
    --output_folder data/1.5B/numca/

## 3B
python src/data_preparation/merge_for_training.py \
    --input_files   src/data_preparation/DAPO-17K/3B/sampled_numca/0717/train.json \
                    src/data_preparation/DAPO-17K/3B/sampled_numca/1737/train.json \
                    src/data_preparation/DAPO-17K/3B/sampled_numca/3777/train.json \
                    src/data_preparation/openr1-220K/3B/sampled_numca/0717/train.json \
                    src/data_preparation/openr1-220K/3B/sampled_numca/1737/train.json \
                    src/data_preparation/openr1-220K/3B/sampled_numca/3777/train.json \
    --num_samples   1000 1000 1000 1000 1000 1000 \
    --output_folder data/3B/numca/

## 7B
python src/data_preparation/merge_for_training.py \
    --input_files   src/data_preparation/DAPO-17K/7B/sampled_numca/0717/train.json \
                    src/data_preparation/DAPO-17K/7B/sampled_numca/1737/train.json \
                    src/data_preparation/DAPO-17K/7B/sampled_numca/3777/train.json \
                    src/data_preparation/openr1-220K/7B/sampled_numca/0717/train.json \
                    src/data_preparation/openr1-220K/7B/sampled_numca/1737/train.json \
                    src/data_preparation/openr1-220K/7B/sampled_numca/3777/train.json \
    --num_samples   1000 1000 1000 1000 1000 1000 \
    --output_folder data/7B/numca/

## 14B
python src/data_preparation/merge_for_training.py \
    --input_files   src/data_preparation/DAPO-17K/7B/sampled_numca/0007/train.json \
                    src/data_preparation/DAPO-17K/7B/sampled_numca/0717/train.json \
                    src/data_preparation/DAPO-17K/7B/sampled_numca/1737/train.json \
                    src/data_preparation/openr1-220K/7B/sampled_numca/0007/train.json \
                    src/data_preparation/openr1-220K/7B/sampled_numca/0717/train.json \
                    src/data_preparation/openr1-220K/7B/sampled_numca/1737/train.json \
    --num_samples   1000 1000 1000 1000 1000 1000 \
    --output_folder   data/14B/numca/