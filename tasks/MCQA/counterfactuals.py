"""Counterfactual dataset generators for the MCQA task.

This module provides functions to generate counterfactual pairs for testing
different causal hypotheses about the MCQA task.
"""

import random
from .causal_models import positional_causal_model, COLORS, ALPHABET, NUM_CHOICES


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
    positional_causal_model.new_raw_input(input_sample)
    return input_sample


def same_symbol_different_position():
    """
    Generate a counterfactual where the answer position changes but symbols stay the same.
    This swaps the choices and symbols at two positions.
    """
    input_sample = sample_answerable_question()
    counterfactual = input_sample.copy()

    pos = positional_causal_model.run_forward(input_sample)["answer_position"]
    new_pos = random.choice([i for i in range(NUM_CHOICES) if i != pos])
    counterfactual["choice" + str(pos)] = input_sample["choice" + str(new_pos)]
    counterfactual["choice" + str(new_pos)] = input_sample["choice" + str(pos)]
    counterfactual["symbol" + str(pos)] = input_sample["symbol" + str(new_pos)]
    counterfactual["symbol" + str(new_pos)] = input_sample["symbol" + str(pos)]

    positional_causal_model.new_raw_input(input_sample)
    positional_causal_model.new_raw_input(counterfactual)
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

    positional_causal_model.new_raw_input(input_sample)
    positional_causal_model.new_raw_input(counterfactual)
    return {"input": input_sample, "counterfactual_inputs": [counterfactual]}


def random_counterfactual():
    """
    Generate a completely random counterfactual by sampling two independent inputs.
    """
    input_sample = sample_answerable_question()
    counterfactual = sample_answerable_question()
    return {"input": input_sample, "counterfactual_inputs": [counterfactual]}
