# Import libraries to read json files and manipulate data
import json
# regular expression library for pattern matching
import re
import os
import argparse

def extract_numbers_from_string(s):
    """
    Extracts all numbers from a given string and returns them as a list of integers.
    """
    try:
        # Regular expression to match numbers
        number_pattern = re.compile(r'\d+')
        # Find all numbers in the string
        numbers = number_pattern.findall(s)
        # Convert the numbers to integers
        return [int(num) for num in numbers]
    except Exception:
        return []

def main(data_path, output_path, upper, lower, keep_responses, targets_num):
    # load the extend20-p1 dataset
    with open(data_path, "r") as f:
        original_dataset = json.load(f)

    # Initialize an empty list to store the processed data
    suitable_dataset = []
    # Iterate through each item in the dataset
    for item in original_dataset:
        if (len(item['wrong_responses']) + len(item['correct_responses'])) <= 0 or len(item['correct_responses']) / (len(item['wrong_responses']) + len(item['correct_responses'])) <= lower or len(item['correct_responses']) / (len(item['wrong_responses']) + len(item['correct_responses'])) > upper:
            continue
        # Extract the question, solution, and answer from the item
        question = item['problem']
        solution = item['solution']
        if keep_responses:
            correct_response = item['correct_responses']
            wrong_response = item['wrong_responses']
        else:
            correct_response = item.pop('correct_responses', [])
            wrong_response = item.pop('wrong_responses', [])
        # Remove the content after "\box" in the solution
        correct_response = [resp.split('\\box')[0].strip() for resp in correct_response]
        # Find all numbers in the question
        numbers_question = set(extract_numbers_from_string(question))
        # Find all numbers in the solution using the regular expression
        numbers_solution = set(extract_numbers_from_string(solution))
        # Find all numbers in the correct response
        numbers_correct_response = set()
        for resp in correct_response:
            numbers_correct_response.update(extract_numbers_from_string(resp))

        # Find the numbers that are in the solution but not in the question and the answer
        target_cv = numbers_correct_response - numbers_question - numbers_solution
        if len(target_cv) < targets_num:
            continue
    
        suitable_dataset.append(item)

    print("Total processed problems:", len(suitable_dataset), ", Original problems:", len(original_dataset))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save the processed dataset to a new json file
    with open(output_path, "w") as f:
        json.dump(suitable_dataset, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="DAPO-17K/extend20/train.json")
    parser.add_argument("--output_path", type=str, default="DAPO-17K/extend20/train_processed.json")
    parser.add_argument("--upper", type=float, default=0.77)
    parser.add_argument("--lower", type=float, default=0.37)
    parser.add_argument("--keep_responses", type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--targets_num", type=int, default=-999)
    args = parser.parse_args()
    main(args.data_path, args.output_path, args.upper, args.lower, args.keep_responses, args.targets_num)
