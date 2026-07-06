import json
import os

# Read the raw testing dataset line by line, each line is a json object
with open("raw/test.json", "r") as f:
    test_data = [json.loads(line) for line in f]

processed_dataset = []
for i, item in enumerate(test_data):
    answer = item["answer"]
    # Parse the final answer after the last '####' if exists
    if "####" in answer:
        answer = answer.split("####")[-1].strip()
    else:
        continue

    processed_item = {
        "problem": item["question"],
        "solution": answer,
        "verifier": "default",
        "id": i
    }
    processed_dataset.append(processed_item)

os.makedirs("processed", exist_ok=True)

# Save the processed dataset
with open("processed/test.json", "w") as f:
    json.dump(processed_dataset, f, indent=4)