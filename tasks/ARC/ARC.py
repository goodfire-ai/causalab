import json, random, os
import sys
import os

# Get the current file's directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory of the parent directory (grandparent)
grandparent_dir = os.path.dirname(os.path.dirname(current_dir))

# Add the grandparent directory to the path
sys.path.append(grandparent_dir)
from copy import deepcopy
from causal_model import CausalModel 
from task import Task
from model_units.LM_units import TokenPosition, get_last_token_index
import re
from datasets import load_dataset, Dataset

def load_hf_dataset(dataset_path, split, hf_token=None, size=None, name=None, parse_fn=None):
    """
    Load a HuggingFace dataset and reformat it to be compatible with the Task object.
    
    Parameters:
        dataset_path (str): The path or name of the HF dataset
        split (str): "train", "test", or "validation"
        hf_token (str): HuggingFace token
        name (str, optional): Sub-configuration name for the dataset (if any)
        parse_fn (callable, optional): 
            A function that takes a single row from a dataset and returns a string or dict
            to be placed in the "input" column. This is where we do dataset-specific parsing 
            (e.g. extracting digits for arithmetic). If None, we default to using `row["prompt"]`
            or `row["question"]`
    """
    base_dataset = load_dataset(dataset_path, name, split=split, token=hf_token)
    if size != None:
        if size > len(base_dataset):
            size = len(base_dataset)
        base_dataset = base_dataset.shuffle().select(range(size))
        
    # Retrieve all counterfactual names
    sample = base_dataset[0]
    counterfactual_names = [k for k in sample.keys() if k.endswith('_counterfactual')]
    data_dict = {
        counterfactual_name: {"input": [], "counterfactual_inputs": []}
        for counterfactual_name in counterfactual_names
    }
    for row in base_dataset:
        if len(row["choices"]["label"]) > 4:
            continue
        if parse_fn is not None:
            # parse_fn is something like parse_arithmetic_example(row) => returns a string or dict
            input_obj = parse_fn(row) 
        else:
            print("Not able to parse input.")
            input_obj = row.get("question", row.get("prompt", ""))

        for counterfactual_name in counterfactual_names:
            if counterfactual_name not in ["answerPosition_counterfactual", "randomLetter_counterfactual", "answerPosition_randomLetter_counterfactual"]:
                continue
            if counterfactual_name in row:
                cf_data = row[counterfactual_name] 
            else:
                cf_data = []
            
            data_dict[counterfactual_name]["input"].append(input_obj)
            counterfactual_obj = [parse_fn(cf_data)] # assume it is a list of dict
            data_dict[counterfactual_name]["counterfactual_inputs"].append(counterfactual_obj)

    datasets = {}
    for counterfactual_name in data_dict:
        if counterfactual_name not in ["answerPosition_counterfactual", "randomLetter_counterfactual", "answerPosition_randomLetter_counterfactual"]:
            continue
        name = counterfactual_name.replace("_counterfactual", "_" + split)
        datasets[name] = Dataset.from_dict(data_dict[counterfactual_name])

    return datasets

def get_task(hf=True, size=None, include_private: bool = False):
    # Load dataset from HuggingFace with customized parsing
    datasets = {}
    for split in ["train", "validation", "test"]:
        temp = load_hf_dataset(
            dataset_path="mech-interp-bench/arc_easy",
            split=split,
            parse_fn=parse_arc_easy_example,
            size=size
        )
        datasets.update(temp)

    if include_private:
        private = load_hf_dataset(
            dataset_path="mech-interp-bench/arc_easy_private_test",
            split="test",
            parse_fn=parse_arc_easy_example,
            size=size
        )
        datasets.update({k+"private":v for k,v in private.items()})
    
    # Build a prompt-to-answerKey lookup dictionary for efficient searching
    prompt_to_answerkey = {}
    for dataset_name, dataset in datasets.items():
        for row in dataset:
            if "full_prompt" in row["input"]:
                prompt_to_answerkey[row["input"]["full_prompt"]] = row["input"]["answerKey"]
            
            # Add counterfactual inputs to the lookup
            for cf_input in row["counterfactual_inputs"]:
                if "full_prompt" in cf_input:
                    prompt_to_answerkey[cf_input["full_prompt"]] = cf_input["answerKey"]
    
    # Define input_parser with access to the lookup table
    def input_parser(text):
        # Split into question and choices
        lines = text.strip().split('\n')
        question_line = lines[0]
        choices = lines[1:-1]  # Exclude the "Answer:" line
        
        # Build output dictionary
        output = {
            'full_prompt': text
        }
        
        # Parse choices
        for i, choice in enumerate(choices):
            symbol, choice_text = choice.split('. ', 1)
            output[f'symbol{i}'] = symbol
            output[f'choice{i}'] = choice_text
        
        # Lookup answerKey in O(1) time using the dictionary
        output['answerKey'] = prompt_to_answerkey.get(text, 0)  # Default to 0 if not found
        
        return output
    
    NUM_CHOICES = 4
    ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    variables = ["full_prompt", "answer_pointer", "answer", "answerKey"] + ["symbol" + str(x) for x in range(NUM_CHOICES)] 

    values = {}
    values.update({"symbol" + str(x): ALPHABET for x in range(NUM_CHOICES)})
    values.update({"answer_pointer": range(NUM_CHOICES), "answer": ALPHABET})
    values.update({"full_prompt": None})
    values.update({"answerKey": None})

    parents = {"answer":["answer_pointer"] + ["symbol" + str(x) for x in range(NUM_CHOICES)], 
            "answer_pointer": ["answerKey"],
            "answerKey": [],
            "full_prompt": []}

    parents.update({"symbol" + str(x): [] for x in range(NUM_CHOICES)})

    def get_symbol():
        return random.choice(list(ALPHABET))

    def get_answer_pointer(answerKey):
        return answerKey

    def get_answer(answer_pointer, *symbols):
        return " " + symbols[answer_pointer]

    mechanisms = {
        "full_prompt": lambda:"Question: Which factor will most likely cause a person to develop a fever?\nA. a leg muscle relaxing after exercise\nB. a bacterial population in the bloodstream\nC. several viral particles on the skin\nD. carbohydrates being digested in the stomach\nAnswer:",
        **{f"symbol{i}": get_symbol for i in range(NUM_CHOICES)},
        "answer_pointer": get_answer_pointer,
        "answer": get_answer,
        "answerKey":lambda: 0 
    }

    # Create and initialize the model
    model = CausalModel(variables, values, parents, mechanisms)

    def input_dumper(setting):
        return setting["full_prompt"]

    def output_dumper(setting):
        return setting["answer"]

    return Task(model, datasets, input_dumper, output_dumper, input_parser, id=f"ARC_easy")

def parse_arc_easy_example(row):
    """
    Customized parsing function for the ARC Easy dataset.
    """
    full_prompt = row.get("prompt")
    data_point = {
        "full_prompt": full_prompt
    }

    choice_labels = row["choices"]["label"]

    for i in range(len(choice_labels)):
        data_point[f"symbol{i}"] = str(choice_labels[i]) 
    
    data_point["answerKey"] = row["answerKey"]

    return data_point

def get_token_positions(pipeline, task):
    def get_correct_symbol_index(prompt, pipeline, task):
        """
        Find the index of the correct answer symbol token in the prompt.
        
        Args:
            prompt (str): The prompt text
            pipeline: The tokenizer pipeline
            task: The task object with causal model and input loader
            
        Returns:
            list[int]: List containing the index of the correct answer symbol token
        """
        # Run the model to get the answer position
        output = task.causal_model.run_forward(task.input_loader(prompt))
        pointer = output["answer_pointer"]
        correct_symbol = output[f"symbol{pointer}"]
        
        # Tokenize the entire prompt
        tokenized_prompt = pipeline.load(prompt)["input_ids"][0]
        
        # Get all the tokens as strings for analysis
        all_tokens = pipeline.tokenizer.convert_ids_to_tokens(tokenized_prompt)
        
        # Find potential token indices for each answer option (A, B, C, D)
        option_indices = []
        for i, token in enumerate(all_tokens):
            # Different tokenizers might represent single letters differently
            # Some might have them as standalone tokens, others might include punctuation
            if correct_symbol in token:
                # Check if this is actually an answer option and not a random occurrence
                # Look for patterns like "A.", "A:", "A ", etc.
                context = prompt[max(0, prompt.find(token) - 1):prompt.find(token) + len(token) + 1]
                if re.search(rf"{correct_symbol}[\.\:\s]", context):
                    option_indices.append(i)
        
        # If we have exactly one match, return it
        if len(option_indices) == 1:
            return [option_indices[0]]
        
        # If we have multiple matches, we need to disambiguate
        # One approach is to look for the option in the format of "A. answer"
        for i in option_indices:
            # Check a few tokens ahead to see if this seems like a option-answer pattern
            if i < len(all_tokens) - 2:  # Ensure we can look ahead
                # Look for patterns like "A. " followed by content
                context_tokens = "".join(all_tokens[i:i+3])
                if re.search(rf"{correct_symbol}[\.\:\s]", context_tokens):
                    return [i]
        
        # If still no definitive match, fallback to regex on original text
        # and then map to token index
        matches = list(re.finditer(rf"\b{re.escape(correct_symbol)}[\.\:\s]", prompt))
        if matches:
            # Take the first match position and find the closest token
            match_pos = matches[0].start()
            
            # Find which token contains or is closest to this position
            cumulative_length = 0
            for i, token in enumerate(all_tokens):
                token_text = pipeline.tokenizer.convert_tokens_to_string([token])
                token_length = len(token_text)
                
                if match_pos >= cumulative_length and match_pos < cumulative_length + token_length:
                    return [i]
                
                cumulative_length += token_length
        
        # If all else fails
        raise ValueError(f"Could not find correct symbol {correct_symbol} token in prompt: {prompt}")

    # Create TokenPosition object
    token_positions = [
        TokenPosition(lambda x: get_correct_symbol_index(x, pipeline, task), pipeline, id="correct_symbol"),
        # TokenPosition(lambda x: [get_correct_symbol_index(x, pipeline, task)[0]+1], pipeline, id="correct_symbol_period"),
        TokenPosition(lambda x: get_last_token_index(x, pipeline), pipeline, id="last_token")
    ]
    return token_positions

def is_unique(lst):
    return len(lst) == len(set(lst))
