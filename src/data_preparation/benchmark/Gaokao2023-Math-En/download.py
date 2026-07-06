from datasets import load_dataset

# download the dataset
dataset = load_dataset("MARIO-Math-Reasoning/Gaokao2023-Math-En")

# Save the training and testing dataset as json files
dataset["train"].to_json("raw/test.json")

