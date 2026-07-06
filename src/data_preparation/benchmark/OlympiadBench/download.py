from datasets import load_dataset

# download the dataset
dataset = load_dataset("math-ai/olympiadbench")

# Save the training and testing dataset as json files
dataset["test"].to_json("raw/test.json")

