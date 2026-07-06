# Assume you are in the root directory of this repository
ROOT_DIR=$(pwd)

# Process the training set
cd $ROOT_DIR/src/data_preparation/DAPO-17K && python split.py
cd $ROOT_DIR/src/data_preparation/openr1-220K && python process.py
cd $ROOT_DIR/src/data_preparation/MixtureOfThought && python process_science.py
cd $ROOT_DIR/src/data_preparation/GeneralQA && python process.py
cd $ROOT_DIR/src/data_preparation/verifiable_python && python process.py

cd $ROOT_DIR