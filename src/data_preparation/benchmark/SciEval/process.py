import json
import re
import random
import os

random.seed(42)

# Read the raw testing dataset line by line, each line is a json object
with open("raw/test.json", "r") as f:
    test_data = [json.loads(line) for line in f.readlines()]

processed_dataset = []
for i, item in enumerate(test_data):
    answer = item["answer"]
    if answer:
        if len(answer) > 1:
            continue  # skip if there are multiple answers
    else:
        continue  # skip if there is no answer
    answer = answer[0]
    problem = item["question"]
    # Remove the "\n\nAnswer:" in the problem statement if it exists
    # This will encourage the model to reasoning instead of directly outputting the answer
    problem = re.sub(r"\n\nAnswer:.*", "", problem).strip()

    processed_item = {
        "problem": problem,
        "solution": answer,
        "verifier": "general",
        "id": i
    }
    processed_dataset.append(processed_item)

os.makedirs("processed", exist_ok=True)

# Save the processed dataset
with open("processed/test.json", "w") as f:
    json.dump(processed_dataset, f, indent=4)