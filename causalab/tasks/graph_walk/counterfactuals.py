"""Counterfactual dataset generation for the graph_walk task."""

from __future__ import annotations

import random
from typing import Any

from causalab.causal.model import CausalModel


def generate_graph_walk_dataset(
    causal_model: CausalModel,
    n_examples: int,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Generate counterfactual examples from a graph walk causal model.

    Cycles through node_coordinates for even coverage.
    """
    rng = random.Random(seed)
    examples: list[dict[str, Any]] = []
    coords = causal_model.values["node_coordinates"]
    for i in range(n_examples):
        coord = coords[i % len(coords)]
        input_trace = causal_model.new_trace(
            {"node_coordinates": coord, "walk_seed": rng.randrange(2**32)}
        )
        cf_coord = rng.choice([c for c in coords if c != coord])
        cf_trace = causal_model.new_trace(
            {"node_coordinates": cf_coord, "walk_seed": rng.randrange(2**32)}
        )
        examples.append(
            {
                "input": input_trace,
                "counterfactual_inputs": [cf_trace],
            }
        )
    return examples


def make_walk_steering_examples(walk_texts: list[str]) -> list[dict[str, Any]]:
    """Convert walk texts into CounterfactualExample-compatible dicts."""
    return [
        {"input": {"raw_input": text}, "counterfactuals": []} for text in walk_texts
    ]


# Standard alias for load_task() convention
generate_dataset = generate_graph_walk_dataset
