import random
import re
from causal.causal_model import CausalModel, CounterfactualDataset
from neural.LM_units import TokenPosition, get_last_token_index
from tasks.task import Task

# Constants
OBJECT_COLORS = [
    ("banana", "yellow"), ("grass", "green"), ("strawberry", "red"),
    ("coconut", "brown"), ("eggplant", "purple"), ("blueberry", "blue"),
    ("carrot", "orange"), ("coal", "black"), ("snow", "white"),
    ("ivory", "white"), ("cauliflower", "white"), ("bubblegum", "pink"),
    ("lemon", "yellow"), ("lime", "green"), ("ruby", "red"),
    ("chocolate", "brown"), ("emerald", "green"), ("sapphire", "blue"),
    ("pumpkin", "orange")
]
OBJECTS, COLORS = zip(*OBJECT_COLORS)
COLORS = list(set(COLORS))  # Ensure unique colors

NUM_CHOICES = 2
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
TEMPLATES = [
    "The <object> is <color>. What color is the <object>?" +
    "".join([f"\n<symbol{str(i)}>. <choice{str(i)}>" for i in range(NUM_CHOICES)]) +
    "\nAnswer:"
]


# Causal Model Functions
def fill_template(*args):
    """Fill in the template with object, color, symbols, and choices."""
    template, object_color = args[0], args[1]
    symbols = args[2:2 + NUM_CHOICES]
    choices = args[2 + NUM_CHOICES:2 + 2 * NUM_CHOICES]

    object_name, color = object_color
    filled_template = template.replace("<object>", object_name).replace("<color>", color)
    for i, symbol in enumerate(symbols):
        filled_template = filled_template.replace(f"<symbol{i}>", symbol)
    for i, choice in enumerate(choices):
        filled_template = filled_template.replace(f"<choice{i}>", choice)
    return filled_template


def get_answer_position(object_color, *choices):
    """Determine which choice position contains the correct answer."""
    for i, choice in enumerate(choices):
        if choice == object_color[1]:
            return i


def get_answer(answer_position, *symbols):
    """Get the symbol corresponding to the correct answer position."""
    if answer_position is None:
        return None
    return symbols[answer_position]


variables = (
    ["template", "object_color", "raw_input"] +
    ["symbol" + str(x) for x in range(NUM_CHOICES)] +
    ["choice" + str(x) for x in range(NUM_CHOICES)] +
    ["answer_position", "answer", "raw_output"]
)

values = {"choice" + str(x): COLORS for x in range(NUM_CHOICES)}
values.update({"symbol" + str(x): ALPHABET for x in range(NUM_CHOICES)})
values.update({"answer_position": range(NUM_CHOICES), "answer": ALPHABET})
values.update({"template": TEMPLATES})
values.update({"object_color": OBJECT_COLORS})
values.update({"raw_input": None, "raw_output": None})

parents = {
    "template": [],
    "object_color": [],
    "raw_input": (
        ["template", "object_color"] +
        ["symbol" + str(x) for x in range(NUM_CHOICES)] +
        ["choice" + str(x) for x in range(NUM_CHOICES)]
    ),
    "answer_position": (
        ["object_color"] +
        ["choice" + str(x) for x in range(NUM_CHOICES)]
    ),
    "answer": (
        ["answer_position"] +
        ["symbol" + str(x) for x in range(NUM_CHOICES)]
    ),
    "raw_output": ["answer"],
}
parents.update({"choice" + str(x): [] for x in range(NUM_CHOICES)})
parents.update({"symbol" + str(x): [] for x in range(NUM_CHOICES)})

mechanisms = {
    "template": lambda: random.choice(TEMPLATES),
    "object_color": lambda: random.choice(OBJECT_COLORS),
    **{f"symbol{i}": lambda: random.choice(ALPHABET) for i in range(NUM_CHOICES)},
    **{f"choice{i}": lambda: random.choice(COLORS) for i in range(NUM_CHOICES)},
    "raw_input": fill_template,
    "answer_position": get_answer_position,
    "answer": get_answer,
    "raw_output": lambda x: " " + x if x is not None else None,
}

positional_causal_model = CausalModel(
    variables,
    values,
    parents,
    mechanisms,
    id=f"{NUM_CHOICES}_answer_MCQA"
)


# Dataset Sampler Functions
def sample_answerable_question():
    """Sample a question where the correct answer appears in the choices."""
    input_sample = positional_causal_model.sample_input()
    # Sample unique choices and symbols
    choices = random.sample(COLORS, NUM_CHOICES)
    symbols = random.sample(ALPHABET, NUM_CHOICES)
    for idx in range(NUM_CHOICES):
        input_sample["choice" + str(idx)] = choices[idx]
        input_sample["symbol" + str(idx)] = symbols[idx]
    # Ensure the correct color is in the choices
    if input_sample["object_color"][1] not in [input_sample["choice" + str(x)] for x in range(NUM_CHOICES)]:
        index = random.randint(0, NUM_CHOICES - 1)
        input_sample["choice" + str(index)] = input_sample["object_color"][1]
    input_sample["raw_input"] = positional_causal_model.run_forward(input_sample)["raw_input"]
    return input_sample


def same_symbol_different_position():
    """
    Generate a counterfactual where the answer position changes but symbols stay the same.
    This swaps the choices and symbols at two positions.
    """
    input_sample = sample_answerable_question()
    counterfactual = input_sample.copy()
    del counterfactual["raw_input"]

    pos = positional_causal_model.run_forward(input_sample)["answer_position"]
    new_pos = random.choice([i for i in range(NUM_CHOICES) if i != pos])
    counterfactual["choice" + str(pos)] = input_sample["choice" + str(new_pos)]
    counterfactual["choice" + str(new_pos)] = input_sample["choice" + str(pos)]
    counterfactual["symbol" + str(pos)] = input_sample["symbol" + str(new_pos)]
    counterfactual["symbol" + str(new_pos)] = input_sample["symbol" + str(pos)]

    input_sample["raw_input"] = positional_causal_model.run_forward(input_sample)["raw_input"]
    counterfactual["raw_input"] = positional_causal_model.run_forward(counterfactual)["raw_input"]
    return {"input": input_sample, "counterfactual_inputs": [counterfactual]}


def different_symbol():
    """
    Generate a counterfactual where both the answer position and all symbols change.
    """
    input_sample = sample_answerable_question()
    counterfactual = input_sample.copy()
    del counterfactual["raw_input"]


    # Different symbols
    current_symbols = [input_sample["symbol" + str(i)] for i in range(NUM_CHOICES)]
    complement = [x for x in ALPHABET if x not in current_symbols]
    new_symbols = random.sample(complement, NUM_CHOICES)
    for i in range(NUM_CHOICES):
        counterfactual["symbol" + str(i)] = new_symbols[i]

    input_sample["raw_input"] = positional_causal_model.run_forward(input_sample)["raw_input"]
    counterfactual["raw_input"] = positional_causal_model.run_forward(counterfactual)["raw_input"]
    return {"input": input_sample, "counterfactual_inputs": [counterfactual]}


def random_counterfactual():
    """
    Generate a completely random counterfactual by sampling two independent inputs.
    """
    input_sample = sample_answerable_question()
    counterfactual = sample_answerable_question()
    input_sample["raw_input"] = positional_causal_model.run_forward(input_sample)["raw_input"]
    counterfactual["raw_input"] = positional_causal_model.run_forward(counterfactual)["raw_input"]
    return {"input": input_sample, "counterfactual_inputs": [counterfactual]}


# Token Position Functions
def get_symbol_index(input_sample, pipeline, index):
    """
    Find the index of the correct answer symbol in the prompt.

    Args:
        input_sample (Dict): The input dictionary to a causal model
        pipeline: The tokenizer pipeline
        causal_model: The causal model

    Returns:
        list[int]: List containing the index of the correct answer symbol token
    """
    target_symbol = input_sample[f"symbol{index}"]
    prompt = input_sample["raw_input"]

    # Find all single uppercase letters in the prompt
    matches = list(re.finditer(r"\b[A-Z]\b", prompt))

    # Find the match corresponding to our target symbol
    symbol_match = None
    for match in matches:
        if prompt[match.start():match.end()] == target_symbol:
            symbol_match = match
            break

    if not symbol_match:
        raise ValueError(f"Could not find symbol {target_symbol} in prompt: {prompt}")

    # Step 1: Load FULL prompt WITH padding (as normal)
    tokenized_prompt_padded = list(pipeline.load(prompt)["input_ids"][0])
    pad_token_id = pipeline.tokenizer.pad_token_id

    # Step 2: Find where content starts (first non-padding token)
    content_start_idx = 0
    for i, token in enumerate(tokenized_prompt_padded):
        if token != pad_token_id:
            content_start_idx = i
            break

    # Step 3: Extract just the content portion (no padding)
    content_tokens = [t for t in tokenized_prompt_padded if t != pad_token_id]

    # Step 4: Tokenize substring WITHOUT padding to get what we're looking for
    substring = prompt[:symbol_match.end()]
    tokenized_substring = list(pipeline.load(substring, no_padding=True)["input_ids"][0])

    # Step 5: Find where substring ends in the content portion
    m = len(tokenized_substring)
    if m == 0:
        raise ValueError(f"Substring tokenized to empty sequence: {substring}")

    end_idx_in_content = next(
        (i + m for i in range(len(content_tokens) - m + 1)
         if content_tokens[i:i+m] == tokenized_substring),
        -1
    )

    if end_idx_in_content == -1:
        raise ValueError(f"Could not find tokenized substring in prompt")

    # Step 6: Convert to padded coordinate system
    token_index_in_padded = content_start_idx + end_idx_in_content - 1

    return [token_index_in_padded]


def create_symbol_token_position(pipeline, index):
    """Create a TokenPosition for the correct answer symbol."""
    return TokenPosition(
        lambda x: get_symbol_index(x, pipeline, index),
        pipeline,
        id=f"symbol{index}"
    )


def create_symbol_period_token_position(pipeline, index):
    """Create a TokenPosition for the period after the correct answer symbol."""
    return TokenPosition(
        lambda x: [get_symbol_index(x, pipeline, index)[0] + 1],
        pipeline,
        id=f"symbol{index}_period"
    )

def get_correct_symbol_index(input_sample, pipeline):
    """
    Find the index of the correct answer symbol in the prompt.

    Args:
        input_sample (Dict): The input dictionary to a causal model
        pipeline: The tokenizer pipeline
        causal_model: The causal model

    Returns:
        list[int]: List containing the index of the correct answer symbol token
    """
    # Run the model to get the answer position
    output = positional_causal_model.run_forward(input_sample)
    pos = output["answer_position"]
    correct_symbol = output[f"symbol{pos}"]
    prompt = input_sample["raw_input"]

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

    # Step 1: Load FULL prompt WITH padding (as normal)
    tokenized_prompt_padded = list(pipeline.load(prompt)["input_ids"][0])
    pad_token_id = pipeline.tokenizer.pad_token_id

    # Step 2: Find where content starts (first non-padding token)
    content_start_idx = 0
    for i, token in enumerate(tokenized_prompt_padded):
        if token != pad_token_id:
            content_start_idx = i
            break

    # Step 3: Extract just the content portion (no padding)
    content_tokens = [t for t in tokenized_prompt_padded if t != pad_token_id]

    # Step 4: Tokenize substring WITHOUT padding to get what we're looking for
    substring = prompt[:symbol_match.end()]
    tokenized_substring = list(pipeline.load(substring, no_padding=True)["input_ids"][0])

    # Step 5: Find where substring ends in the content portion
    m = len(tokenized_substring)
    if m == 0:
        raise ValueError(f"Substring tokenized to empty sequence: {substring}")

    end_idx_in_content = next(
        (i + m for i in range(len(content_tokens) - m + 1)
         if content_tokens[i:i+m] == tokenized_substring),
        -1
    )

    if end_idx_in_content == -1:
        raise ValueError(f"Could not find tokenized substring in prompt")

    # Step 6: Convert to padded coordinate system
    token_index_in_padded = content_start_idx + end_idx_in_content - 1

    return [token_index_in_padded]


def create_correct_symbol_token_position(pipeline):
    """Create a TokenPosition for the correct answer symbol."""
    return TokenPosition(
        lambda x: get_correct_symbol_index(x, pipeline),
        pipeline,
        id="correct_symbol"
    )


def create_correct_symbol_period_token_position(pipeline):
    """Create a TokenPosition for the period after the correct answer symbol."""
    return TokenPosition(
        lambda x: [get_correct_symbol_index(x, pipeline)[0] + 1],
        pipeline,
        id="correct_symbol_period"
    )


def create_last_token_position(pipeline):
    """Create a TokenPosition for the last token in the input."""
    return TokenPosition(
        lambda x: get_last_token_index(x, pipeline),
        pipeline,
        id="last_token"
    )


# Task Definition
MCQA_task = Task(
    name="MCQA",
    causal_models={
        "positional": positional_causal_model,
    },
    dataset_generators={
        "different_symbol": different_symbol,
        "same_symbol_different_position": same_symbol_different_position,
        "random_counterfactual": random_counterfactual,
    },
    token_positions={
        "symbol0": lambda x: create_symbol_token_position(x, 0),
        "symbol0_period": lambda x:create_symbol_period_token_position(x, 0),
        "symbol1": lambda x: create_symbol_token_position(x, 1),
        "symbol1_period": lambda x:create_symbol_period_token_position(x, 1),
        "correct_symbol": create_correct_symbol_token_position,
        "correct_symbol_period": create_correct_symbol_period_token_position,
        "last_token": create_last_token_position,
    }
)