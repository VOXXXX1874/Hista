import json
import random
import os

random.seed(42)

# Read the raw training dataset line by line, each line is a json object
with open("raw/train.json", "r") as f:
    train_data = [json.loads(line) for line in f]

processed_dataset = []
for i, item in enumerate(train_data):
    # Naive filter:  Filter out the math and science data to avoid overlapping
    if item["category"] == "Mathematics" or item["difficulty"] == "Junior High School":
        continue
    if item["category"] in ["Physics", "Chemistry", "Biology"] and random.random() < 0.7:
        continue
    # Only preserve the "prompt" and "answer" column
    # Rename the "prompt" to "problem" and "answer" to "solution"
    processed_item = {
        "problem": item["question"],
        "solution": item["answer"],
        "id": i,
        "verifier": "general"
    }
    processed_dataset.append(processed_item)

os.makedirs("partition_1", exist_ok=True)
os.makedirs("partition_2", exist_ok=True)
os.makedirs("partition_3", exist_ok=True)

# Random sample 36000 items from the processed dataset
processed_dataset = random.sample(processed_dataset, 36000)
processed_dataset_1 = processed_dataset[:12000]
processed_dataset_2 = processed_dataset[12000:24000]
processed_dataset_3 = processed_dataset[24000:]

# Save the processed dataset
with open("partition_1/train.json", "w") as f:
    json.dump(processed_dataset_1, f, indent=4)

with open("partition_2/train.json", "w") as f:
    json.dump(processed_dataset_2, f, indent=4)

with open("partition_3/train.json", "w") as f:
    json.dump(processed_dataset_3, f, indent=4)