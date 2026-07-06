import json
import os
import re

# Read the raw testing dataset line by line, each line is a json object
with open("raw/test.json", "r") as f:
    test_data = [json.loads(line) for line in f]

processed_dataset = []
for i, item in enumerate(test_data):
    if "college_math" in item["data_source"]:
        answer = item["answer"]
        # Heuristic filter that remove
        # 1. answers that too long to parse, which are always wrapped by \begin{} instead of $
        # 2. answers that contain extra words like "gallons" or "miles"
        if answer[0] == '$' and answer[-1] == '$':
            answer = answer[1:-1]
        else:
            # Check whether the answer is a number without $
            match = re.search(r"\d+", answer)
            if match and match.group(0) == answer.strip():
                answer = match.group(0)
            else:
                continue  # skip this item if the answer format is not recognized

        processed_item = {
            "problem": item["question"],
            "solution": answer,
            "id": i,
            "verifier": "default"
        }
        processed_dataset.append(processed_item)

os.makedirs("processed", exist_ok=True)

# Save the processed dataset
with open("processed/test.json", "w") as f:
    json.dump(processed_dataset, f, indent=4)