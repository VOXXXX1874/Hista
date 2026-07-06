import json
import os

# Read the raw testing dataset line by line, each line is a json object
with open("raw/test_24.json", "r") as f:
    test_data_24 = [json.loads(line) for line in f]

with open("raw/test_25.json", "r") as f:
    test_data_25 = [json.loads(line) for line in f]

test_data = test_data_24 + test_data_25

processed_dataset = []
for i, item in enumerate(test_data):

    processed_item = {
        "problem": item["problem"],
        "solution": item["answer"] if "answer" in item else item["solution"],
        "id": i,
        "verifier": "default"
    }
    processed_dataset.append(processed_item)

os.makedirs("processed", exist_ok=True)

# Save the processed dataset
with open("processed/test.json", "w") as f:
    json.dump(processed_dataset, f, indent=4)