from datasets import load_dataset

# download the dataset
dataset = load_dataset("OpenDFM/SciEval")

# Save the training and testing dataset as json files
dataset["test"].to_json("raw/test.json")

