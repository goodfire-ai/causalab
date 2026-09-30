"""
Counterfactual generator functions for the task.
"""

import random

from causalab.causal.counterfactuals import CounterfactualExample

from .causal_models import CAUSAL_MODEL
from .config import PATTERNS
from .templates import (  # pyright: ignore[reportPrivateUsage]
    TEMPLATES,
    _sample_pattern_values,
)


def sample_balanced_input(model=CAUSAL_MODEL, rng=random):
    """Sample a balanced input across the four patterns."""
    pattern = rng.choice(PATTERNS)
    v1, v2, v3, v4 = _sample_pattern_values(pattern, rng)
    template = rng.choice(TEMPLATES)
    return model.new_trace(
        {
            "template": template,
            "var_1": v1,
            "var_2": v2,
            "var_3": v3,
            "var_4": v4,
            "icl_seed": rng.randrange(2**32),
        }
    )


def random_counterfactual():
    """Generate a random counterfactual by sampling two independent balanced inputs."""
    input_sample = sample_balanced_input()
    counterfactual = sample_balanced_input()

    return CounterfactualExample(
        input=input_sample, counterfactual_inputs=[counterfactual]
    )


COUNTERFACTUAL_GENERATORS = {
    "random_counterfactual": random_counterfactual,
}


def generate_dataset(model, n: int, seed: int = 42) -> list[CounterfactualExample]:
    """Generate n counterfactual examples using balanced sampling."""
    model = CAUSAL_MODEL if model is None else model
    rng = random.Random(seed)
    return [
        {
            "input": sample_balanced_input(model, rng),
            "counterfactual_inputs": [sample_balanced_input(model, rng)],
        }
        for _ in range(n)
    ]
