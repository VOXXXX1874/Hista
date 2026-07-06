from datasets import load_dataset

# download the dataset
dataset = load_dataset("TIGER-Lab/WebInstruct-verified")

# Save the training and testing dataset as json files
dataset["train"].to_json("raw/train.json")
