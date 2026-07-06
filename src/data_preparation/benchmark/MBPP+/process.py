import json
import random
import os

random.seed(42)

INSTRUCTIONS_PREFIX = '''You can use common libraries like `numpy` and `pandas`. Your code should pass following test cases:
```python
'''

INSTRUCTIONS_TAIL = '''
```
'''

# Read the raw testing dataset line by line, each line is a json object
with open("raw/test.json", "r") as f:
    test_data = [json.loads(line) for line in f]

processed_dataset = []
for i, item in enumerate(test_data):
    prompt = item["prompt"]
    # Add neccessary instructions to the prompt
    prompt = prompt + INSTRUCTIONS_PREFIX
    solution = "\n".join(item["test_list"])
    prompt = prompt + solution + INSTRUCTIONS_TAIL
    real_solution = "\n\nFLAGTOBEREPLACED\n\n" + item['test']

    processed_dataset.append({
        "problem": prompt,
        "solution": real_solution,
        "verifier": "code_humaneval",
        "id": i
    })
    
os.makedirs("processed", exist_ok=True)

# Save the processed dataset
with open("processed/test.json", "w") as f:
    json.dump(processed_dataset, f, indent=4)