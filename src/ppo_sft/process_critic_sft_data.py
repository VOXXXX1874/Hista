import json
from transformers import AutoTokenizer
import random
import argparse
from rl.utils.prepare_dataset import SYSTEM_PROMPT, SYSTEM_PROMPT_CODE

def create_new_qa(qa, response, reward, tokenizer, system_prompt):
    new_qa = {}
    message = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": qa["problem"]},
        {"role": "assistant", "content": response},
    ]
    # apply chat template to the new message
    text = tokenizer.apply_chat_template(
        message, tokenize=False, continue_final_message=True
    )
    new_qa["text"] = text
    new_qa["reward"] = reward
    return new_qa

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process PPO Critic SFT Dataset")
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to the input JSON file containing the QA dataset.",
    )
    parser.add_argument(
        "--output_train_file",
        type=str,
        default="train.json",
        help="Path to the output JSON file for the processed training dataset.",
    )
    parser.add_argument(
        "--output_test_file",
        type=str,
        default="test.json",
        help="Path to the output JSON file for the processed testing dataset.",
    )
    args = parser.parse_args()

    # Read the "result.json" file
    with open(args.input_file, "r") as f:
        qa_dataset = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

    train_qa_dataset = qa_dataset[:len(qa_dataset)//5 * 4]
    test_qa_dataset = qa_dataset[len(qa_dataset)//5 * 4:]

    # for each sample in train.json, process the segment
    ppo_sft_dataset_train = []
    for i, qa in enumerate(train_qa_dataset):
        # get the answer
        correct_responses = qa["correct_responses"]
        for response in correct_responses:
            new_qa = create_new_qa(qa, response, 1.0, tokenizer, SYSTEM_PROMPT)
            ppo_sft_dataset_train.append(new_qa)
        wrong_responses = qa["wrong_responses"]
        for response in wrong_responses:
            new_qa = create_new_qa(qa, response, 0.0, tokenizer, SYSTEM_PROMPT)
            ppo_sft_dataset_train.append(new_qa)

    ppo_sft_dataset_test = []
    for i, qa in enumerate(test_qa_dataset):
        # get the answer
        correct_responses = qa["correct_responses"]
        for response in correct_responses:
            new_qa = create_new_qa(qa, response, 1.0, tokenizer, SYSTEM_PROMPT)
            ppo_sft_dataset_test.append(new_qa)
        wrong_responses = qa["wrong_responses"]
        for response in wrong_responses:
            new_qa = create_new_qa(qa, response, 0.0, tokenizer, SYSTEM_PROMPT)
            ppo_sft_dataset_test.append(new_qa)


    # shuffle the segmented dataset
    random.shuffle(ppo_sft_dataset_train)
    random.shuffle(ppo_sft_dataset_test)

    # Save the segmented dataset to a new file
    with open(args.output_train_file, "w") as f:
        json.dump(ppo_sft_dataset_train, f, indent=4)

    with open(args.output_test_file, "w") as f:
        json.dump(ppo_sft_dataset_test, f, indent=4)