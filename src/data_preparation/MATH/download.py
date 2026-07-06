from datasets import load_dataset

# download the dataset
dataset = load_dataset("DigitalLearningGmbH/MATH-lighteval")

# Save the training and testing dataset as json files
dataset["train"].to_json("raw/train.json")
dataset["test"].to_json("raw/test.json")

