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

def get_data(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def get_task(hf=True, size=None, include_private: bool = False):
    variables = ["output_position", "output_token", "template", "name_A", "name_B", "name_C", "place", "object", "logit_diff"]

    name_path = os.path.join("tasks", os.path.join("IOI_task", 'names.json'))
    objects_path = os.path.join("tasks", os.path.join("IOI_task", 'objects.json'))
    places_path = os.path.join("tasks", os.path.join("IOI_task", 'places.json'))
    templates_path = os.path.join("tasks", os.path.join("IOI_task", 'templates.json'))

    NAMES = get_data(name_path)
    OBJECTS = get_data(objects_path)
    PLACES = get_data(places_path)
    TEMPLATES = get_data(templates_path)
    values = {
        "output_position": [0, 1],
        "output_token": NAMES,
        "name_A": NAMES,
        "name_B": NAMES,
        "name_C": NAMES,
        "place": PLACES,
        "object": OBJECTS,
        "template": TEMPLATES,
        "logit_diff": None
    }
    
    def get_template():
        return random.choice(TEMPLATES)

    def get_name_A():
        return random.choice(NAMES)
    
    def get_name_B():
        return random.choice(NAMES)
    
    def get_name_C():
        return random.choice(NAMES)
    
    def get_place():
        return random.choice(PLACES)
    
    def get_object():
        return random.choice(OBJECTS)
    
    def get_output_position(name_A, name_B, name_C):
        if name_C == name_A:
            return 1
        elif name_C == name_B:
            return 0
        else:
            return "Error"
        # assert False, f"{name_C} should match {name_B} or {name_A}"
    
    def get_output_token(name_A, name_B, name_C):
        if name_C == name_A:
            return name_B
        elif name_C == name_B:
            return name_A
        else:
            return "Error"
        # assert False, f"{name_C} should match {name_B} or {name_A}"

    def get_logit_diff(name_A, name_B, name_C, output_token, output_position):
        token_signal = None 
        if (name_C == name_A and output_token == name_B) or (name_C == name_B and output_token == name_A):
            token_signal = 1
        elif (name_C == name_A and output_token == name_A) or (name_C == name_B and output_token == name_B):
            token_signal = -1

        position_signal = None 
        if (name_C == name_A and output_position == 1) or (name_C == name_B and output_position == 0):
            position_signal = 1
        elif (name_C == name_A and output_position == 0) or (name_C == name_B and output_position == 1):
            position_signal = -1

        # print(token_signal , position_signal)
        # print(0.295 + 0.63 * token_signal + 2.235 * position_signal)
        return 0.295 + 0.63 * token_signal + 2.235 * position_signal


    parents = {
        "template": [],
        "name_A": [],
        "name_B": [],
        "name_C": [],
        "place": [],
        "object": [],
        "output_token": ["name_A", "name_B", "name_C"],
        "output_position": ["name_A", "name_B", "name_C"],
        "logit_diff": ["name_A", "name_B", "name_C", "output_token", "output_position"]
    }

    mechanisms = {
        "template": get_template,
        "name_A": get_name_A,
        "name_B": get_name_B,
        "name_C": get_name_C,
        "place": get_place,
        "object": get_object,
        "output_token": get_output_token,
        "output_position": get_output_position,
        "logit_diff": get_logit_diff
    }

    model = CausalModel(variables, values, parents, mechanisms)

    def input_dumper(input):
        template = input['template']
        template = template.replace("{name_A}", input["name_A"])
        template = template.replace("{name_B}", input["name_B"])
        template = template.replace("{name_C}", input["name_C"])
        if input["object"] is not None:
            template = template.replace("{object}", input["object"])
        if input["place"] is not None:
            template = template.replace("{place}", input["place"])
        
        return template
    
    def output_dumper(setting): # connects with the checker
        return setting["output_token"]

    def input_parser(input):
        # Helper to convert template into regex and track variable order
        def extract_vars(prompt):
            prompt = ' '.join(prompt.split())  # Normalize whitespace

            def template_to_regex(template):
                pattern = re.escape(template)
                var_counts = {}
                
                # Match all {var} placeholders in order
                all_vars = re.findall(r"\{(name_A|name_B|name_C|place|object)\}", template)

                for var in all_vars:
                    var_counts[var] = var_counts.get(var, 0) + 1

                    if var_counts[var] == 1:
                        group = f"(?P<{var}>[^,\.]+)"
                    else:
                        # Avoid redefining the same named group
                        group = r"[^,\.]+"

                    escaped_var = re.escape(f"{{{var}}}")
                    pattern = pattern.replace(escaped_var, group, 1)  # only replace the first occurrence

                return re.compile(f"^{pattern}$")

            for template in TEMPLATES:
                regex = template_to_regex(template)
                match = regex.match(prompt)
                if match:
                    return match.groupdict(), template

            print(f"Prompt '{prompt}' does not match any template.")
        output = {}
        if "metadata" in input:
            output["name_A"] = input["metadata"]["subject"]
            output["name_B"] = input["metadata"]["indirect_object"]
            output["name_C"] = input["metadata"]["subject"]
            output["object"] = input["metadata"]["object"] if "object" in input["metadata"] else None
            output["place"] = input["metadata"]["place"] if "place" in input["metadata"] else None
            output["template"] = input["template"]
        else:
            variables = {}
            try:
                variables, template = extract_vars(input['prompt'])
                output["name_A"] = variables["name_A"]
                output["name_B"] = variables["name_B"]
                output["name_C"] = variables["name_C"]
                output["object"] = variables["object"] if "object" in variables else None
                output["place"] = variables["place"] if "place" in variables else None
                output["template"] = template
            except Exception as e:
                print(f"Error parsing prompt: {input['prompt']} {output}")
                print(e)
                assert False
            


        return output
    
    if hf:
        print('HUGGINGFACE!')
        # Load dataset from HuggingFace with customized parsing
        datasets = {}
        for split in ["train",  "test"]:
            temp = model.load_hf_dataset(
                dataset_path="mech-interp-bench/ioi",
                split=split,
                parse_fn=input_parser,
                size=size,
                ignore_names=["random", "abc"]
            )
            datasets.update(temp)

        if include_private:
            private = model.load_hf_dataset(
                dataset_path="mech-interp-bench/ioi_private_test",
                split="test",
                parse_fn=input_parser,
                size=size,
                ignore_names=["random", "abc"]
            )
            datasets.update({k+"private":v for k,v in private.items()})

    return Task(model, datasets, input_dumper, output_dumper,input_parser, id=f"ioi_task")


def get_token_positions(pipeline, task):
    def get_intervention_locations(x, pipeline):
        """Helper method for getting intervention locations for specific attention heads + token positions"""
        tokens = list(range(pipeline.load(x)['input_ids'].shape[-1]))
        return tokens
    return TokenPosition(lambda x: get_intervention_locations(x, pipeline), pipeline, id="all")