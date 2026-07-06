# Assume you are in the root directory of this repository
ROOT_DIR=$(pwd)

# Process the training set
cd $ROOT_DIR/src/data_preparation/MATH && python process.py --benchmark_data_dir $ROOT_DIR/src/data_preparation/benchmark/MATH-500/raw
cd $ROOT_DIR/src/data_preparation/DAPO-17K && python process.py

# Process the benchmarks
cd $ROOT_DIR/src/data_preparation/benchmark/MATH-500 && python process.py
cd $ROOT_DIR/src/data_preparation/benchmark/GSM8K && python process.py
cd $ROOT_DIR/src/data_preparation/benchmark/OlympiadBench && python process.py
cd $ROOT_DIR/src/data_preparation/benchmark/MinervaMath && python process.py
cd $ROOT_DIR/src/data_preparation/benchmark/amc23 && python process.py
cd $ROOT_DIR/src/data_preparation/benchmark/AIME2425 && python process.py
cd $ROOT_DIR/src/data_preparation/benchmark/Gaokao2023-Math-En && python process.py
cd $ROOT_DIR/src/data_preparation/benchmark/CollegeMath && python process.py

# Move the processed data to the data directory
cd $ROOT_DIR
rm -r -f data/MATH && cp -r src/data_preparation/MATH/processed data/MATH
rm -r -f data/DAPO && cp -r src/data_preparation/DAPO-17K/processed data/DAPO
rm -r -f data/MATH-500 && cp -r src/data_preparation/benchmark/MATH-500/processed data/MATH-500
rm -r -f data/GSM8K && cp -r src/data_preparation/benchmark/GSM8K/processed data/GSM8K
rm -r -f data/OlympiadBench && cp -r src/data_preparation/benchmark/OlympiadBench/processed data/OlympiadBench
rm -r -f data/MinervaMath && cp -r src/data_preparation/benchmark/MinervaMath/processed data/MinervaMath
rm -r -f data/amc23 && cp -r src/data_preparation/benchmark/amc23/processed data/amc23
rm -r -f data/AIME2425 && cp -r src/data_preparation/benchmark/AIME2425/processed data/AIME2425
rm -r -f data/GaoKao && cp -r src/data_preparation/benchmark/Gaokao2023-Math-En/processed data/GaoKao
rm -r -f data/CollegeMath && cp -r src/data_preparation/benchmark/CollegeMath/processed data/CollegeMath