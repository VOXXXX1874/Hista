from datasets import load_dataset

# download the dataset
dataset = load_dataset("TIGER-Lab/TheoremQA")

# Drop the picture columns if exist
if "Picture" in dataset["test"].column_names:
    dataset["test"] = dataset["test"].remove_columns("Picture")

# Save the training and testing dataset as json files
dataset["test"].to_json("raw/test.json")

