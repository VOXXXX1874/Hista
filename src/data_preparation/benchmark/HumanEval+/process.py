import json
import random
import os
random.seed(42)

INSTRUCTIONS_PREFIX = '''
Implement the following function according to the given specification in Python. Please output the whole function and import statements instead of just the function body. You can use common libraries like `numpy` and `pandas`.
```python
'''

INSTRUCTIONS_TAIL = '''
```
'''

TEST = '''
check({function_name})
'''

# Read the raw testing dataset line by line, each line is a json object
with open("raw/test.json", "r") as f:
    test_data = [json.loads(line) for line in f]

processed_dataset = []
for i, item in enumerate(test_data):
    prompt = item["prompt"]
    # Add neccessary instructions to the prompt
    prompt = INSTRUCTIONS_PREFIX + prompt + INSTRUCTIONS_TAIL
    solution = item["test"]
    solution += "\n\nFLAGTOBEREPLACED\n\n"
    solution += TEST.format(function_name=item["entry_point"])
    processed_dataset.append({
        "problem": prompt,
        "solution": solution,
        "verifier": "code_humaneval",
        "id": i
    })

os.makedirs("processed", exist_ok=True)
    
# Save the processed dataset
with open("processed/test.json", "w") as f:
    json.dump(processed_dataset, f, indent=4)