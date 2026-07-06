from datasets import load_dataset

# download the dataset
dataset_24 = load_dataset("math-ai/aime24")
dataset_25 = load_dataset("math-ai/aime25")

# Save the training and testing dataset as json files
dataset_24["test"].to_json("raw/test_24.json")
dataset_25["test"].to_json("raw/test_25.json")