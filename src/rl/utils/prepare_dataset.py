from datasets import load_dataset

SYSTEM_PROMPT = (
    "A conversation between a User and an Assistant. "
    "The User asks a question; the Assistant solves it by first reasoning, then providing the final answer. "
    "The Assistant encloses its final answer in \\boxed{}. If it is multi-choice, only provide the letter corresponding to the final answer in \\boxed{}."
)

SYSTEM_PROMPT_CODE = (
    "A conversation between a User and an Assistant. "
    "The User asks a python programming question; the Assistant solves it by first reasoning, then providing the final solution. "
    "The Assistant encloses its final answer in triple backticks ```python\n...\n```."
)

def prepare_dataset(dataset_name, split, silence=True, use_default_system_prompt=False):
    def make_latex(example):
        if example.get("verifier", None) and ( "code" in example.get("verifier", None) or "general" in example.get("verifier", None)):
            pass
        else:
            example["solution"] = '$' + str(example["solution"]) + '$'
        return example

    # Format into conversation
    def make_conversation(example):
        if use_default_system_prompt:
            return {
                "prompt": [
                    {"role": "user", "content": example["problem"] + "\nPlease reason step by step and put your final answer within \\boxed{}."},
                ],
            }
        else:
            if example.get("verifier", None) == "code":
                return {
                    "prompt": [
                        {"role": "system", "content": SYSTEM_PROMPT_CODE},
                        {"role": "user", "content": example["problem"]},
                    ],
                }
            else:
                return {
                    "prompt": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": example["problem"]},
                    ],
                }

    def add_attributes(example, attribute_names, attribute_values):
        for name, value in zip(attribute_names, attribute_values):
            example[name] = value
        return example

    dataset = load_dataset(dataset_name, split=split)

    # The dataset is assumed to have 'problem' and 'solution' columns
    # Where the solution column only contains the final answer without any formatting (or at most $$ signs)
    dataset = dataset.map(make_latex) 
    dataset = dataset.map(make_conversation)
    dataset = dataset.map(lambda x: add_attributes(x, ["silence"], [silence]))

    # display one example from the dataset
    print(f'Example from dataset ({dataset_name}) ({split}): {dataset[0]}')

    return dataset
