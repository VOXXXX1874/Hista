from datasets import load_dataset

# download the dataset
dataset = load_dataset("open-r1/DAPO-Math-17k-Processed", "en")

# Save the training and testing dataset as json files
dataset["train"].to_json("raw/train.json")
