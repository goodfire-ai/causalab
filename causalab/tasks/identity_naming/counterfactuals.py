"""Counterfactual dataset generators for the identity_naming task.

Cycles through phrasing templates to provide prompt variation for centroid
computation. Each example gets a different template, but phrasing is not
a causal variable — it's metadata on the dataset row.
"""

from causalab.causal.counterfactuals import CounterfactualExample


def _sample_trace(model, template, rng):
    """Sample a random entity and build a trace with the given template."""
    entity = rng.choice(model.values["entity"])
    trace = model.new_trace({"entity": entity})
    return trace.intervene("raw_input", template.format(entity=entity))


def generate_dataset(model, n: int, seed: int = 42) -> list[CounterfactualExample]:
    """Generate n counterfactual examples, cycling through phrasing templates.

    Each example uses a different template for its raw_input, giving multiple
    activations per entity/result pair. The counterfactual swaps the entity.
    """
    import random

    from .config import IdentityNamingConfig

    config: IdentityNamingConfig = model._in_config  # type: ignore[attr-defined]
    templates = config.templates
    n_templates = len(templates)

    rng = random.Random(seed)
    examples = []
    for i in range(n):
        template = templates[i % n_templates]
        input_sample = _sample_trace(model, template, rng)
        counterfactual = _sample_trace(model, template, rng)
        examples.append(
            {"input": input_sample, "counterfactual_inputs": [counterfactual]}
        )
    return examples
