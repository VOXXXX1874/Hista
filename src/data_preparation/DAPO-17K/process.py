import json
import os

# Read the raw training dataset
with open("raw/train.json", "r") as f:
    lines = f.readlines()
    train_data = [json.loads(line) for line in lines]

processed_dataset = []
for i, item in enumerate(train_data):
    # Only preserve the "prompt" and "solution" column
    # Rename the "prompt" to "problem"
    processed_item = {
        "problem": item["prompt"],
        "solution": item["solution"],
        "verifier": "default",
        "id": i
    }
    processed_dataset.append(processed_item)

# Create the output directory if it doesn't exist
os.makedirs("processed", exist_ok=True)

# Save the processed dataset
with open("processed/train.json", "w") as f:
    json.dump(processed_dataset, f, indent=4)