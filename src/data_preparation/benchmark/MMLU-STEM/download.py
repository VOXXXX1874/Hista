from datasets import load_dataset

# download the dataset
dataset = load_dataset("TIGER-Lab/MMLU-STEM")

# Save the training and testing dataset as json files
dataset["test"].to_json("test.json")

