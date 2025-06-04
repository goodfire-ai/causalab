import json, random, os
from copy import deepcopy
import sys
import os

# Get the current file's directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory of the parent directory (grandparent)
grandparent_dir = os.path.dirname(os.path.dirname(current_dir))

# Add the grandparent directory to the path
sys.path.append(grandparent_dir)
from causal_model import CausalModel 
from task import Task
from model_units.LM_units import TokenPosition, get_last_token_index
import re


def get_task(hf=True, size=None, include_private: bool = False):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'object_color_pairs.json')
    #Load grandparent directory
    with open(path, 'r') as f:
            data = json.load(f)

    OBJECTS = [item['object'] for item in data]
    COLORS = [item['color'] for item in data]
    COLOR_OBJECTS = [(item["color"], item["object"]) for item in data]

    NUM_CHOICES = 4
    ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    variables = ["question"] + ["symbol" + str(x) for x in range(NUM_CHOICES)] + ["choice" + str(x) for x in range(NUM_CHOICES)] + [ "answer_pointer", "answer"]

    values = {"choice" + str(x): COLORS for x in range(NUM_CHOICES)}
    values.update({"symbol" + str(x): ALPHABET for x in range(NUM_CHOICES)})
    values.update({"answer_pointer": range(NUM_CHOICES), "answer": ALPHABET})
    values.update({"question": COLOR_OBJECTS })

    parents = {"answer":["answer_pointer"] + ["symbol" + str(x) for x in range(NUM_CHOICES)], 
            "answer_pointer": ["question"] + ["choice" + str(x) for x in range(NUM_CHOICES)],
                "question": []}
    parents.update({"choice" + str(x): [] for x in range(NUM_CHOICES)})
    parents.update({"symbol" + str(x): [] for x in range(NUM_CHOICES)})




    def get_question():
        return random.choice(COLOR_OBJECTS)

    def get_symbol():
        return random.choice(list(ALPHABET))

    def get_choice():
        return random.choice(COLORS)

    def get_answer_pointer(question, *choices):
        for i, choice in enumerate(choices):
            if choice == question[0]:
                return i
        return random.randrange(NUM_CHOICES)

    def get_answer(answer_pointer, *symbols):
        return " " + symbols[answer_pointer]

    mechanisms = {
        "question": get_question,
        
        **{f"symbol{i}": get_symbol for i in range(NUM_CHOICES)},
        
        **{f"choice{i}": get_choice for i in range(NUM_CHOICES)},
        
        "answer_pointer": get_answer_pointer,
        "answer": get_answer
    }

    # Create and initialize the model
    model = CausalModel(variables, values, parents, mechanisms)

    def input_dumper(input):
        output = f"Question: The {input['question'][1]} is {input['question'][0]}. What color is the {input['question'][1]}?"
        for i in range(NUM_CHOICES):
            output += f"\n{input[f'symbol{i}']}. {input[f'choice{i}']}"
        output += f"\nAnswer:"
        return output

    def output_dumper(setting):
        return setting["answer"]


    if hf:
        # Load dataset from HuggingFace with customized parsing
        datasets = {}
        for split in ["train", "validation", "test"]:
            temp = model.load_hf_dataset(
                dataset_path="mech-interp-bench/copycolors_mcqa",
                split=split,
                name=f"{NUM_CHOICES}_answer_choices",
                parse_fn=parse_mcqa_example,
                size=size,
                ignore_names=["noun", "color", "symbol"]
            )
            datasets.update(temp)

        if include_private:
            private = model.load_hf_dataset(
                dataset_path="mech-interp-bench/copycolors_mcqa_private_test",
                split="test",
                name=f"{NUM_CHOICES}_answer_choices",
                parse_fn=parse_mcqa_example,
                size=size,
                ignore_names=["noun", "color", "symbol"]
            )
            datasets.update({k+"private":v for k,v in private.items()})

        return Task(model, datasets, input_dumper, output_dumper,input_parser, id=f"{NUM_CHOICES}_answer_MCQA")

    def filter(x): 
        return x["question"][0] in [x[f"choice{i}"] for i in range(NUM_CHOICES)] and is_unique([x["choice" + str(i)] for i in range(NUM_CHOICES)] + [x["symbol" + str(i)] for i in range(NUM_CHOICES)])

    def random_counterfactual_pair():
        return {"input": model.sample_input(filter=filter), "counterfactual_inputs": [model.sample_input(filter=filter)]}

    def randomLetter_randomPosition_counterfactual_pair():
        input = model.sample_input(filter=filter)
        counterfactual_input = model.sample_input(filter=filter)
        counterfactual_input["question"] = input["question"]
        return {
            "input": input,
            "counterfactual_inputs": [counterfactual_input]
        }

    def newPosition_counterfactual_pair():
        input = model.sample_input(filter=filter)
        counterfactual_input = deepcopy(input)
        pointer = model.run_forward(input)["answer_pointer"]
        # Choose a random new position different from current one
        available_positions = [i for i in range(NUM_CHOICES) if i != pointer]
        new_position = random.choice(available_positions)
        # Swap the choices to move correct answer to new position
        correct_choice = counterfactual_input[f"choice{pointer}"]
        counterfactual_input[f"choice{pointer}"] = counterfactual_input[f"choice{new_position}"]
        counterfactual_input[f"choice{new_position}"] = correct_choice

        return {
            "input": input,
            "counterfactual_inputs": [counterfactual_input]
        }

    def newLetter_counterfactual_pair():
        # Get valid input where question color appears in choices
        input = model.sample_input(filter=filter)
        # Create deep copy for counterfactual
        counterfactual_input = deepcopy(input)
        # Get the current symbols to avoid duplicates in new set
        used_symbols = [input[f"symbol{i}"] for i in range(NUM_CHOICES)]
        # Generate new set of unique symbols
        available_symbols = [s for s in ALPHABET if s not in used_symbols]
        new_symbols = random.sample(available_symbols, NUM_CHOICES)
        # Update all symbols in counterfactual
        for i in range(NUM_CHOICES):
            counterfactual_input[f"symbol{i}"] = new_symbols[i]
        
        return {
            "input": input,
            "counterfactual_inputs": [counterfactual_input]
        }

    datasets = {"random_counterfactual": model.generate_counterfactual_dataset(size, random_counterfactual_pair),
    "randomLetter_randomPosition_counterfactual": model.generate_counterfactual_dataset(size, randomLetter_randomPosition_counterfactual_pair),
    "newPosition_counterfactual": model.generate_counterfactual_dataset(size, newPosition_counterfactual_pair),
    "newLetter_counterfactual": model.generate_counterfactual_dataset(size, newLetter_counterfactual_pair)}
    return Task(model, datasets, input_dumper, output_dumper, input_parser,f"{NUM_CHOICES}_answer_MCQA")

def get_token_positions(pipeline, task):
    def get_correct_symbol_index(prompt, pipeline, task):
        """
        Find the index of the correct answer symbol in the prompt.
        
        Args:
            prompt (str): The prompt text
            pipeline: The tokenizer pipeline
            
        Returns:
            list[int]: List containing the index of the correct answer symbol token
        """
        # Run the model to get the answer position
        output = task.causal_model.run_forward(task.input_loader(prompt))
        pointer = output["answer_pointer"]
        correct_symbol = output[f"symbol{pointer}"]
        
        # Find all single uppercase letters in the prompt
        matches = list(re.finditer(r"\b[A-Z]\b", prompt))
        
        # Find the match corresponding to our correct symbol
        symbol_match = None
        for match in matches:
            if prompt[match.start():match.end()] == correct_symbol:
                symbol_match = match
                break
                
        if not symbol_match:
            raise ValueError(f"Could not find correct symbol {correct_symbol} in prompt: {prompt}")
        
        # Get the substring up to the symbol match end
        substring = prompt[:symbol_match.end()]
        tokenized_substring = list(pipeline.load(substring)["input_ids"][0])
        
        # The symbol token will be at the end of the substring
        return [len(tokenized_substring) - 1]

    # Create TokenPosition object
    token_positions = [
        TokenPosition(lambda x: get_correct_symbol_index(x, pipeline, task), pipeline, id="correct_symbol"),
        # TokenPosition(lambda x: [get_correct_symbol_index(x, pipeline, task)[0]+1], pipeline, id="correct_symbol_period"),
        TokenPosition(lambda x: get_last_token_index(x, pipeline), pipeline, id="last_token")
    ]
    return token_positions

def is_unique(lst):
    return len(lst) == len(set(lst))

def parse_mcqa_example(row):
    """
    Customized parsing function for the MCQA task.
    """
    q_str = row.get("question", row.get("prompt", ""))

    if " is " in q_str:
        noun, color = q_str.split(" is ", 1)
    elif " are " in q_str:
        noun, color = q_str.split(" are ", 1)
    else:
        noun = "object"
        color = "unknown"
    noun = noun.strip().lower()
    color = color.split(".", 1)[0].strip().lower()

    choice_labels = row["choices"]["label"]
    choice_texts  = row["choices"]["text"]

    data_point = {
        "question": (color, noun)
    }

    for i in range(len(choice_labels)):
        data_point[f"symbol{i}"] = str(choice_labels[i]) 
        data_point[f"choice{i}"] = str(choice_texts[i])

    return data_point

def input_parser(text):
    # Split into question and choices
    lines = text.strip().split('\n')
    question_line = lines[0]
    choices = lines[1:-1]  # Exclude the "Answer:" line
    
    # Parse question
    # Format: "Question: The {object} is {adjective}. What color is the {object}?"
    question_text = question_line.replace("Question: ", "")
    object_start = question_text.index("The ") + 4
    object_end = question_text.index(" is")
    object_name = question_text[object_start:object_end]
    
    adjective_start = question_text.index("is ") + 3
    adjective_end = question_text.index(". What")
    adjective = question_text[adjective_start:adjective_end]
    
    # Build output dictionary
    output = {
        'question': (adjective, object_name)
    }
    
    # Parse choices
    for i, choice in enumerate(choices):
        symbol, text = choice.split('. ', 1)
        output[f'symbol{i}'] = symbol
        output[f'choice{i}'] = text
    
    return output
