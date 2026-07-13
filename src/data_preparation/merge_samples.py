import json
import argparse
import random
import os

random.seed(42)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_files", type=str, nargs="+", required=True, help="List of input JSONL files to merge")
    parser.add_argument("--output_file", type=str, required=True, help="Output JSONL file")
    args = parser.parse_args()

    merged_data = []
    id = 0
    for input_file in args.input_files:
        data = json.load(open(input_file, "r"))
        for item in data:
            item["id"] = id
            item["solution"] = item["solution"].replace("$", "")
            id += 1
            merged_data.append(item)

    # Shuffle the merged data
    random.shuffle(merged_data)

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    with open(args.output_file, "w") as f:
        json.dump(merged_data, f, indent=4)