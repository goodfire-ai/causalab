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
from datasets import load_dataset
from causal_model import CausalModel 
from task import Task
from model_units.LM_units import TokenPosition, get_last_token_index
import re


# TODO: Create a class to store all the RAVEL data!
# Load RAVEL metadata.
_RAVEL_DATA_DIR = os.path.dirname(os.path.abspath(__file__))
_CITY_ENTITY = {}
_OFFSET_DICT = {}
_PROMPT_TO_TEMPLATE = {}


def generate_prompt_to_template(task):
  prompt_to_template = {}
  for counterfactual_name, items in task.counterfactual_datasets.items():
      for ex in items["input"]:
          p, t = ex["prompt"], ex["template"]
          prompt_to_template[p] = t

      for ex in items["counterfactual_inputs"]:
          p, t = ex[0]["prompt"], ex[0]["template"]
          prompt_to_template[p] = t
  print('#Prompts:', len(prompt_to_template))
  json.dump(prompt_to_template,
            open(os.path.join(_RAVEL_DATA_DIR, "ravel_prompt_to_template.json"), "w"),
            ensure_ascii=False, indent=2)
  return prompt_to_template


def get_task(hf=True, size=None, include_private: bool = False):
    if not _CITY_ENTITY:
        _CITY_ENTITY.update(json.load(open(os.path.join(_RAVEL_DATA_DIR, 'ravel_city_entity.json'))))

    # Define causal model
    attributes = ["Continent", "Country", "Language"]
    variables = ["prompt", "entity", "queried_attribute", "answer"] + attributes
   
    parents = {
        "prompt": [],
        "entity": ["prompt"],
        "queried_attribute": ["prompt"],
        "answer": ["entity", "queried_attribute", "Continent", "Country", "Language"],
        "Continent": ["entity"],
        "Country": ["entity"], 
        "Language": ["entity"],
    }
    # Assign values after loading the dataset
    values = {
        "prompt": set(), 
        "entity": set(), 
        "queried_attribute": set(),
        "answer": set(), 
        "Continent": set(),
        "Country": set(),
        "Language": set(),
    }

    def get_prompt():
        pass
    def get_entity(prompt):
        return random.choice(list(_CITY_ENTITY))
    def get_queried_attribute(prompt):
        return random.choice(attributes)
    def get_answer(entity, q_attr, *attr_values):
        if q_attr == "wikipedia":
            return None 
        idx = attributes.index(q_attr)
        return attr_values[idx]
    def get_continent(entity):
        return _CITY_ENTITY[entity]["Continent"]
    def get_country(entity):
        return _CITY_ENTITY[entity]["Country"]
    def get_language(entity):
        return _CITY_ENTITY[entity]["Language"]

    mechanisms = {
        "prompt": get_prompt,
        "entity": get_entity,
        "queried_attribute": get_queried_attribute,
        "answer": get_answer,
        "Continent": get_continent,
        "Country": get_country,
        "Language": get_language,
    }
    model = CausalModel(variables, values, parents, mechanisms)


    def input_dumper(input_data):
        return input_data["prompt"]

    def output_dumper(setting):
        return setting["answer"]
        
    if hf:
        datasets = {}
        for split in ["train", "val", "test"]:
            temp = model.load_hf_dataset(
                dataset_path="yiksiu/mib_ravel",
                split=split,
                parse_fn=parse_ravel_example,
                size=size,
                shuffle=True
            )
            datasets.update(temp)

        if include_private:
            private = model.load_hf_dataset(
                dataset_path="yiksiu/mib_ravel_private_test",
                split="test",
                parse_fn=parse_ravel_example,
                size=size,
                shuffle=True
            )
            datasets.update({k+"private":v for k,v in private.items()})

        return Task(model, datasets, input_dumper, output_dumper, id="RAVEL_task")

    else:
        from datasets import load_dataset
        ravel_data = load_dataset("yiksiu/mib_ravel", split="train")

        def prompt_template_counterfactual():
            example = random.choice(ravel_data) 
            return {
                "input": {"prompt": example["prompt"],
                        "label": example["label"]},
                "counterfactual_inputs": [example["prompt_template_counterfactual"]] # Ensure it is a list of dictionary
        }
        def attribute_counterfactual():
            example = random.choice(ravel_data) 
            return {
                "input": {"prompt": example["prompt"],
                        "label": example["label"]},
                "counterfactual_inputs": [example["attribute_counterfactual"]]
            }
        def wikipedia_counterfactual():
            example = random.choice(ravel_data) 
            return {
                "input": {"prompt": example["prompt"],
                        "label": example["label"]}, 
                "counterfactual_inputs": [example["wikipedia_counterfactual"]]
            } 
        datasets = {
            "prompt_template_counterfactual": model.generate_counterfactual_dataset(size, prompt_template_counterfactual),
            "attribute_counterfactual": model.generate_counterfactual_dataset(size, attribute_counterfactual),
            "wikipedia_counterfactual": model.generate_counterfactual_dataset(size, wikipedia_counterfactual)
        }
        return Task(model, datasets, input_dumper, output_dumper, id="RAVEL_task")
    


def get_token_positions(pipeline, task, model_name):
    def get_entity_last_token_position(prompt, pipeline):
        # Look up token position in the cached mapping.
        if prompt not in _PROMPT_TO_TEMPLATE:
            # Try reload.
            _PROMPT_TO_TEMPLATE.update(json.load(open(os.path.join(_RAVEL_DATA_DIR, 'ravel_prompt_to_template.json'))))
            if prompt not in _PROMPT_TO_TEMPLATE:
                raise ValueError(f"Prompt not found in prompt_to_template:\n{prompt}")
        template_str = _PROMPT_TO_TEMPLATE[prompt]
        if template_str not in _OFFSET_DICT:
            # Try reload.
            _OFFSET_DICT.update(
                json.load(open(os.path.join(_RAVEL_DATA_DIR, f'{model_name}_city_prompt_to_entity_position.json'))))
            if template_str not in _OFFSET_DICT:
                raise ValueError(f"Template not in offset dict:\n{template_str}")

        offset = _OFFSET_DICT[template_str]
        encoding = pipeline.load(prompt)
        token_ids = encoding["input_ids"][0]
        length = len(token_ids)

        if offset < 0:
            index = length + offset
        else:
            index = offset

        if index < 0 or index >= length:
            raise ValueError(f"Offset {offset} out of range for token length {length}\nPrompt:\n{prompt}")

        return [index]
    
    token_positions = [
        TokenPosition(lambda x: get_last_token_index(x, pipeline), pipeline, id="last_token"),
        TokenPosition(lambda x: get_entity_last_token_position(x, pipeline), pipeline, id="entity_last_token")
    ]
    return token_positions


def parse_ravel_example(row):
    """
    Convert a single dataset row into a dict for the RAVEL causal model.
    """
    return {
        "prompt": row["prompt"],
        "entity": row["entity"],
        "queried_attribute": row["attribute"],
    }
