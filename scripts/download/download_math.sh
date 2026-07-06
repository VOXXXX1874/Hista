# Assume you are in the root directory of this repository
ROOT_DIR=$(pwd)

# Download the training set
cd $ROOT_DIR/src/data_preparation/MATH && python download.py
cd $ROOT_DIR/src/data_preparation/DAPO-17K && python download.py

# Download the benchmarks
cd $ROOT_DIR/src/data_preparation/benchmark/MATH-500 && python download.py
cd $ROOT_DIR/src/data_preparation/benchmark/GSM8K && python download.py
cd $ROOT_DIR/src/data_preparation/benchmark/OlympiadBench && python download.py
cd $ROOT_DIR/src/data_preparation/benchmark/MinervaMath && python download.py
cd $ROOT_DIR/src/data_preparation/benchmark/amc23 && python download.py
cd $ROOT_DIR/src/data_preparation/benchmark/AIME2425 && python download.py
cd $ROOT_DIR/src/data_preparation/benchmark/Gaokao2023-Math-En && python download.py
cd $ROOT_DIR/src/data_preparation/benchmark/CollegeMath && mkdir raw && wget "https://github.com/microsoft/unilm/blob/master/mathscale/MWPBench/data/full_test.json?raw=true" -O full_test.json && mv full_test.json raw/test.json

cd $ROOT_DIR