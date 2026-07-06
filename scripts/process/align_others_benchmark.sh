# Assume you are in the root directory of this repository
ROOT_DIR=$(pwd)

# Process the benchmarks
cd $ROOT_DIR/src/data_preparation/benchmark/SciEval && python process.py
cd $ROOT_DIR/src/data_preparation/benchmark/TheoremQA && python process.py
cd $ROOT_DIR/src/data_preparation/benchmark/MMLU-PRO && python process.py
cd $ROOT_DIR/src/data_preparation/benchmark/gpqa-diamond && python process.py
cd $ROOT_DIR/src/data_preparation/benchmark/HumanEval+ && python process.py
cd $ROOT_DIR/src/data_preparation/benchmark/MBPP+ && python process.py


# Move the processed data to the data directory
cd $ROOT_DIR
rm -r -f data/SciEval && cp -r src/data_preparation/benchmark/SciEval/processed data/SciEval
rm -r -f data/TheoremQA && cp -r src/data_preparation/benchmark/TheoremQA/processed data/TheoremQA
rm -r -f data/MMLU-PRO && cp -r src/data_preparation/benchmark/MMLU-PRO/processed data/MMLU-PRO
rm -r -f data/gpqa-diamond && cp -r src/data_preparation/benchmark/gpqa-diamond/processed data/gpqa-diamond
rm -r -f data/HumanEval+ && cp -r src/data_preparation/benchmark/HumanEval+/processed data/HumanEval+
rm -r -f data/MBPP+ && cp -r src/data_preparation/benchmark/MBPP+/processed data/MBPP+