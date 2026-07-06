from datasets import load_dataset

# download the dataset
dataset = load_dataset("open-r1/Mixture-of-Thoughts", "science")

# Save the training and testing dataset as json files
dataset["train"].to_json("raw_science/train.json")
