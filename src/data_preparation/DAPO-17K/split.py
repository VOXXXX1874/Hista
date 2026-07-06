import json
import os

# Read the raw training dataset
with open("processed/train.json", "r") as f:
    train_data = json.load(f)

partition_1 = train_data[:len(train_data)//3]
partition_2 = train_data[len(train_data)//3:2*len(train_data)//3]
partition_3 = train_data[2*len(train_data)//3:]

# Create the output directory if it doesn't exist
os.makedirs("partition_1", exist_ok=True)
os.makedirs("partition_2", exist_ok=True)
os.makedirs("partition_3", exist_ok=True)

# Save the partitions
with open("partition_1/train.json", "w") as f:
    json.dump(partition_1, f, indent=4)

with open("partition_2/train.json", "w") as f:
    json.dump(partition_2, f, indent=4)

with open("partition_3/train.json", "w") as f:
    json.dump(partition_3, f, indent=4)