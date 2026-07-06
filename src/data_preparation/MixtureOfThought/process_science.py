import json
import re
import random
import os

random.seed(42)

# Read the raw training dataset line by line, each line is a json object
with open("raw_science/train.json", "r") as f:
    train_data = [json.loads(line) for line in f]

processed_dataset = []
for i, item in enumerate(train_data):
    messages = item["messages"]
    problem = messages[0]["content"]
    solution = messages[1]["content"]
    # Parse the content inside "\boxed{}" as the solution
    match = re.search(r"\\boxed\{(.*?)\}", solution, re.DOTALL)
    if match:
        final_solution = match.group(1).strip()
        processed_dataset.append(
            {
                "problem": problem, 
                "solution": final_solution,
                "id": i,
                "verifier": "general"
            }
        )

# Random sample 36000 items from the processed dataset
processed_dataset = random.sample(processed_dataset, 36000)
processed_dataset_1 = processed_dataset[:12000]
processed_dataset_2 = processed_dataset[12000:24000]
processed_dataset_3 = processed_dataset[24000:]

os.makedirs("partition_1", exist_ok=True)
os.makedirs("partition_2", exist_ok=True)
os.makedirs("partition_3", exist_ok=True)


# Save the processed dataset
with open("partition_1/train.json", "w") as f:
    json.dump(processed_dataset_1, f, indent=4)

with open("partition_2/train.json", "w") as f:
    json.dump(processed_dataset_2, f, indent=4)

with open("partition_3/train.json", "w") as f:
    json.dump(processed_dataset_3, f, indent=4)