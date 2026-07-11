import json
import random
import os

random.seed(42)

# Read the raw training dataset line by line, each line is a json object
with open("raw/train.json", "r") as f:
    train_data = [json.loads(line) for line in f]

processed_dataset = []
for i, item in enumerate(train_data):
    # Naive filter: Only keep items with at least 6 test cases.
    if len(item["verification_info"]["test_cases"]) < 6:
        continue
    # Store the test cases in solution field
    solution = json.dumps(item["verification_info"]["test_cases"])
    processed_item = {
        "problem": item["problem_statement"] + "\n-----Packages-----\n" + f"You can use common packages like numpy and pandas. You can also use built-in functions and libraries.",
        "solution": solution,
        "verifier": "code",
        "id": i
    }
    processed_dataset.append(processed_item)

os.makedirs("partition_1", exist_ok=True)
os.makedirs("partition_2", exist_ok=True)
os.makedirs("partition_3", exist_ok=True)

# Random sample 12000 items from the processed dataset
processed_dataset = random.sample(processed_dataset, 12000)
processed_dataset_1 = processed_dataset[:4000]
processed_dataset_2 = processed_dataset[4000:8000]
processed_dataset_3 = processed_dataset[8000:12000]

# Save the processed dataset
with open("partition_1/train.json", "w") as f:
    json.dump(processed_dataset_1, f, indent=4)

with open("partition_2/train.json", "w") as f:
    json.dump(processed_dataset_2, f, indent=4)

with open("partition_3/train.json", "w") as f:
    json.dump(processed_dataset_3, f, indent=4)