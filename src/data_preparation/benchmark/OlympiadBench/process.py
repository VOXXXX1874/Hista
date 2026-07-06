import json
import os

# Read the raw testing dataset line by line, each line is a json object
with open("raw/test.json", "r") as f:
    test_data = [json.loads(line) for line in f]

processed_dataset = []
for i, item in enumerate(test_data):
    # only keep the text-only question
    if item["modality"] != "Text-only":
        continue
    # Only keep the question with one final answer
    if len(item["final_answer"]) != 1:
        continue 
    processed_item = {
        "problem": item["question"],
        "solution": item["final_answer"][0],
        "id": i,
        "verifier": "default"
    }
    processed_dataset.append(processed_item)

os.makedirs("processed", exist_ok=True)

# Save the processed dataset
with open("processed/test.json", "w") as f:
    json.dump(processed_dataset, f, indent=4)