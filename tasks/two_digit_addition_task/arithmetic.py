import sys, os
# Get the current file's directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory of the parent directory (grandparent)
grandparent_dir = os.path.dirname(os.path.dirname(current_dir))

# Add the grandparent directory to the path
sys.path.append(grandparent_dir)
from causal_model import CausalModel
from copy import deepcopy
from task import Task
from model_units.LM_units import TokenPosition, get_last_token_index
import re
import random

def get_task(hf=True, size=None, include_private: bool = False):
    variables = [
        "op1_tens", "op1_ones",
        "op2_tens", "op2_ones",
        "ones_carry", 
        "hundreds_out", "tens_out", "ones_out"
    ]

    # Allowed values for each variable.
    values = {
        "op1_tens": [i for i in range(10)],
        "op1_ones": [i for i in range(10)],
        "op2_tens": [i for i in range(10)],
        "op2_ones": [i for i in range(10)],
        "ones_carry": [0, 1],
        # For output intervention nodes: a digit (0–9) or None (meaning no intervention).
        "ones_out": list(range(10)) + [None],
        "tens_out": list(range(10)) + [None],
        "hundreds_out": [0, 1],
    }

    # Specify parent relationships for each node.
    parents = {
        # Base input nodes.
        "op1_tens": [],
        "op1_ones": [],
        "op2_tens": [],
        "op2_ones": [],
        # Carry nodes computed from the corresponding digit nodes.
        "ones_carry": ["op1_ones", "op2_ones"],
        # Output intervention nodes (no parents).
        "ones_out": ["op1_ones", "op2_ones"],
        "tens_out": ["op1_tens", "op2_tens", "ones_carry"],
        "hundreds_out": ["op1_tens", "op2_tens", "ones_carry"],
    }

    # Define the mechanisms (the functions computing each node’s value).
    mechanisms = {
        # Base input nodes: randomly choose a digit.
        "op1_tens": lambda: random.choice(values["op1_tens"]),
        "op1_ones": lambda: random.choice(values["op1_ones"]),
        "op2_tens": lambda: random.choice(values["op2_tens"]),
        "op2_ones": lambda: random.choice(values["op2_ones"]),
        # Compute ones carry: for addition, a carry occurs if the sum of ones digits > 9.
        "ones_carry": lambda op1_ones, op2_ones: 1 if int(op1_ones) + int(op2_ones) > 9 else 0,
        # Output intervention nodes default to no intervention.
        "ones_out": lambda op1_ones, op2_ones: (op1_ones + op2_ones) % 10,
        "tens_out": lambda op1_tens, op2_tens, ones_carry: (op1_tens + op2_tens + ones_carry) % 10,
        "hundreds_out": lambda op1_tens, op2_tens, ones_carry: 1 if op1_tens + op2_tens + ones_carry > 9 else 0,
    }

    model = CausalModel(variables, values, parents, mechanisms)

    def input_dumper(input):
        return f"The sum of {input['op1_tens']}{input['op1_ones']} and {input['op2_tens']}{input['op2_ones']} is "

    def output_dumper(setting):
        return f"{setting['hundreds_out']}{setting['tens_out']}{setting['ones_out']}"

    if hf:
        datasets = {}
        for split in ["train", "test"]:
            temp = model.load_hf_dataset(
                dataset_path="mech-interp-bench/arithmetic_addition",
                split=split,
                size=size,
                parse_fn=parse_arithmetic_example,
                ignore_names=["ones_op1", "ones_op2", "tens_op1", "tens_op2", "tens_carry"]
            )
            datasets.update(temp)

        if include_private:
            private = model.load_hf_dataset(
                dataset_path="mech-interp-bench/arithmetic_addition_private_test",
                split="test",
                size=size,
                parse_fn=parse_arithmetic_example,
                ignore_names=["ones_op1", "ones_op2", "tens_op1", "tens_op2", "tens_carry"]
            )
            datasets.update({k+"private":v for k,v in private.items()})

        return Task(model, datasets, input_dumper, output_dumper, id="arithmetic")

    def random_counterfactual_pair():
        return {"input": model.sample_input(),"counterfactual_inputs": [model.sample_input()]}

    def one_carry_counterfactual_pair():
        input = model.sample_input(filter=lambda x: x["op1_ones"] + x["op2_ones"] != 9)
        counterfactual_input = deepcopy(input)
        if input["op1_ones"] + input["op2_ones"] > 9:
            remainder = (input["op1_ones"] + input["op2_ones"]) % 10
            counterfactual_input["op1_ones"] = remainder - random.choice(list(range(remainder+1)))
            counterfactual_input["op2_ones"] = remainder - counterfactual_input["op1_ones"]
        else:
            goal = (input["op1_ones"] + input["op2_ones"]) + 10
            counterfactual_input["op1_ones"] = goal - random.choice(list(range(goal-9, min(goal, 10))))
            counterfactual_input["op2_ones"] = goal - counterfactual_input["op1_ones"]
        return {"input": input,"counterfactual_inputs": [counterfactual_input]}

    datasets = {"random_counterfactual": model.generate_counterfactual_dataset(size, random_counterfactual_pair),
                "one_carry_counterfactual": model.generate_counterfactual_dataset(size, one_carry_counterfactual_pair),}

    return datasets



def get_token_positions(pipeline, task): 
    def get_op1_last_token_index(prompt, pipeline):
        matches = list(re.finditer(r"\b\d+\b", prompt))
        if len(matches) < 2:
            raise ValueError(f"Prompt must contain at least two numbers: {prompt}")
        
        op1_match = matches[-2]  # Second to last match
        op1 = prompt[op1_match.start():op1_match.end()]
        
        # Get the substring up to the op1 match end
        substring = prompt[:op1_match.end()]
        tokenized_substring = list(pipeline.load(substring)["input_ids"][0])
        
        # The last token of op1 will be at the end of the substring
        return [len(tokenized_substring) - 1]

    def get_op2_last_token_index(prompt, pipeline):
        matches = list(re.finditer(r"\b\d+\b", prompt))
        if len(matches) < 2:
            raise ValueError(f"Prompt must contain at least two numbers: {prompt}")
        
        op2_match = matches[-1]  # Last match
        op2 = prompt[op2_match.start():op2_match.end()]
        
        # Get the substring up to the op2 match end
        substring = prompt[:op2_match.end()]
        tokenized_substring = list(pipeline.load(substring)["input_ids"][0])
        
        # The last token of op2 will be at the end of the substring
        return [len(tokenized_substring) - 1]

    token_positions =[ 
        # TokenPosition(lambda x: get_op1_last_token_index(x, pipeline), pipeline,  id="op1_last"),
        TokenPosition(lambda x: get_op2_last_token_index(x, pipeline), pipeline,  id="op2_last"),
        TokenPosition(lambda x: get_last_token_index(x, pipeline), pipeline,  id="last")
    ]

    return token_positions

def parse_arithmetic_example(row):
    """
    Customized parsing function for the arithmetic task.
    """
    prompt_str = row.get("prompt", "")
    matches = re.findall(r"\d+", prompt_str)
    op1_tens, op1_ones, op2_tens, op2_ones = 0, 0, 0, 0

    if len(matches) >= 2:
        op1_str, op2_str = matches[-2], matches[-1]
        
        def parse_operand(num_str):
            if len(num_str) == 1:
                return 0, int(num_str)
            else:
                return int(num_str[-2]), int(num_str[-1])
        
        op1_tens, op1_ones = parse_operand(op1_str)
        op2_tens, op2_ones = parse_operand(op2_str)
    
    return {
        "op1_tens": op1_tens,
        "op1_ones": op1_ones,
        "op2_tens": op2_tens,
        "op2_ones": op2_ones
    }