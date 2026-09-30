"""Counterfactual dataset generators for the natural_domains_arithmetic task."""

from causalab.causal.counterfactuals import CounterfactualExample


def generate_dataset(model, n: int, seed: int = 42) -> list[CounterfactualExample]:
    """Generate n counterfactual examples using the given model."""
    import random

    state = random.getstate()
    random.seed(seed)
    examples = []
    for _ in range(n):
        input_sample = model.sample_input()
        counterfactual = model.sample_input()
        examples.append(
            {"input": input_sample, "counterfactual_inputs": [counterfactual]}
        )
    random.setstate(state)
    return examples
