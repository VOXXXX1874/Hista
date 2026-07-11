import json
import argparse
import random
import os

random.seed(42)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_files", type=str, nargs="+", required=True, help="List of input JSON files to merge")
    parser.add_argument("--num_samples", type=int, nargs="+", required=True, help="Number of samples to take from each input file")
    parser.add_argument("--output_folder", type=str, required=True, help="Output folder for JSON files")
    parser.add_argument("--exclude_test", action="store_true", help="Whether to exclude test data in the output")
    args = parser.parse_args()

    merged_data = []
    id = 0
    for input_file, num_sample in zip(args.input_files, args.num_samples):
        data = json.load(open(input_file, "r"))
        sampled_data = random.sample(data, num_sample)
        for item in sampled_data:
            item["id"] = id
            id += 1
            merged_data.append(item)

    # Shuffle the merged data
    random.shuffle(merged_data)

    os.makedirs(args.output_folder, exist_ok=True)

    if args.exclude_test:
        # If excluding test data, save all merged data to train.json
        with open(os.path.join(args.output_folder, "train.json"), "w") as f:
            json.dump(merged_data, f, indent=4)
    else:
        # Use 8/10 of the data for training, and 2/10 for validation
        split_index = int(0.9 * len(merged_data))
        train_data = merged_data[:split_index]
        val_data = merged_data[split_index:]

        with open(os.path.join(args.output_folder, "train.json"), "w") as f:
            json.dump(train_data, f, indent=4)

        with open(os.path.join(args.output_folder, "test.json"), "w") as f:
            json.dump(val_data, f, indent=4)