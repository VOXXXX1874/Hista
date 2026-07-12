import json
from math_verify import parse, LatexExtractionConfig
from sympy import nan, zoo
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark_data_dir", type=str, default="raw")
    args = parser.parse_args()

    # Read the raw/train.json file
    with open('raw/train.json', 'r') as f:
        lines = f.readlines()
        training_data = [json.loads(line) for line in lines]

    # Read the raw/test.json file
    with open('raw/test.json', 'r') as f:
        lines = f.readlines()
        testing_data = [json.loads(line) for line in lines]

    # Read the benchmark data from the specified directory
    with open(f'{args.benchmark_data_dir}/test.json', 'r') as f:
        lines = f.readlines()
        benchmark_data = [json.loads(line) for line in lines]

    new_training_data = []
    new_testing_data = []

    # Remove the benchmark problems from the testing data
    benchmark_problems = set(item['problem'] for item in benchmark_data)
    testing_data = [item for item in testing_data if item['problem'] not in benchmark_problems]

    # Naive filtering: Remove items with unparsable or invalid solutions (nan or zoo). Extra the final answer instead of the whole solution.
    for i, item in enumerate(training_data):
        solution = item.get('solution', '')
        parse_result = parse(solution, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()], fallback_mode='first_match')
        if len(parse_result) >= 2 and parse_result[0] != nan and parse_result[0] != zoo:
            processed_item = {
                "problem": item["problem"],
                "solution": parse_result[1],
                "verifier": "default",
                "id": i
            }
            new_training_data.append(processed_item)

    for i, item in enumerate(testing_data):
        solution = item.get('solution', '')
        parse_result = parse(solution, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()], fallback_mode='first_match')

        if len(parse_result) >= 2 and parse_result[0] != nan and parse_result[0] != zoo:
            processed_item = {
                "problem": item["problem"],
                "solution": parse_result[1],
                "verifier": "default",
                "id": i
            }
            new_testing_data.append(processed_item)

    # Create the processed directory if it doesn't exist
    os.makedirs('processed', exist_ok=True)

    # Save the processed training dataset
    with open('processed/train.json', 'w') as f:
        json.dump(new_training_data, f, indent=4)   

    # Save the processed testing dataset
    with open('processed/test.json', 'w') as f:
        json.dump(new_testing_data, f, indent=4)