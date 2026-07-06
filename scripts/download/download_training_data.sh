# Assume you are in the root directory of this repository
ROOT_DIR=$(pwd)

# Download the original data about math, which is from DAPO-17K and OpenR1-220K
cd $ROOT_DIR/src/data_preparation/DAPO-17K && python download.py
cd $ROOT_DIR/src/data_preparation/openr1-220K && python download.py

# Download the original data about science, which is the 'science' partition of Mixture-of-Thoughts dataset.
cd $ROOT_DIR/src/data_preparation/MixtureOfThought && python download.py

# Download the original data about general QA, where we follow GeneralReasoner and use WebInstruct-verified
cd $ROOT_DIR/src/data_preparation/GeneralQA && python download.py

# Download the orignal data about programming, where we use verifiable python dataset to align with our benchmark and avoid complicated engineering.
cd $ROOT_DIR/src/data_preparation/verifiable_python && python download.py

cd $ROOT_DIR