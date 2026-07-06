import json
import re
import random

random.seed(42)

# Read the raw testing dataset line by line, each line is a json object
with open("raw/test.json", "r") as f:
    test_data = [json.loads(line) for line in f]

processed_dataset = []
for item in test_data:
    # Combine the question and choices into a problem
    choices = item["choices"]
    choices_str = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])
    problem = f"Question: \n {item['question']} \n Candidates: \n{choices_str}"
    processed_item = {
        "problem": problem,
        "solution": chr(65 + item["answer"]),
    }
    processed_dataset.append(processed_item)


# Save the processed dataset
with open("processed/test.json", "w") as f:
    json.dump(processed_dataset, f, indent=4)