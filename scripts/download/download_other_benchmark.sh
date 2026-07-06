# Assume you are in the root directory of this repository
ROOT_DIR=$(pwd)

# Download the science related benchmarks
cd $ROOT_DIR/src/data_preparation/benchmark/SciEval && python download.py
cd $ROOT_DIR/src/data_preparation/benchmark/TheoremQA && python download.py
# MinervaMath is classified as science related benchmark in this settings
cd $ROOT_DIR/src/data_preparation/benchmark/MinervaMath && python download.py 

# Download the General QA benchmarks
cd $ROOT_DIR/src/data_preparation/benchmark/MMLU-PRO && python download.py
cd $ROOT_DIR/src/data_preparation/benchmark/gpqa-diamond && python download.py

# Download the Programming benchmarks
cd $ROOT_DIR/src/data_preparation/benchmark/HumanEval+ && python download.py
cd $ROOT_DIR/src/data_preparation/benchmark/MBPP+ && python download.py

cd $ROOT_DIR