"""
Counterfactual dataset generation for entity binding tasks.

Provides functions to generate counterfactual examples by swapping entity groups
while keeping the query structure the same.
"""

import random

from causalab.causal.counterfactuals import CounterfactualExample

from .causal_models import (
    create_positional_entity_causal_model,
    sample_valid_entity_binding_input,
)
from .config import EntityBindingTaskConfig, create_sample_love_config


def swap_query_group(
    config: EntityBindingTaskConfig, change_answer: bool = False
) -> CounterfactualExample:
    """
    Generate a counterfactual by swapping the queried entity group with another group.

    Tests whether the model retrieves information based on which entity group is queried
    (by entity identity), rather than relying on positional information.

    Example with 2 groups:
        Input:
            Entities: g0=(Pete, jam), g1=(Ann, pie)
            Query: group 1, entity 0 -> asking about Ann
            Prompt: "... Pete loves jam, Ann loves pie. What does Ann love?"
            Answer: "pie"

        Counterfactual (swapped g0 and g1):
            Entities: g0=(Ann, pie), g1=(Pete, jam)
            Query: group 0, entity 0 -> now asking about Ann (who moved to g0)
            Prompt: "... Ann loves pie, Pete loves jam. What does Ann love?"
            Answer: "pie"

    The counterfactual swaps entity groups but keeps the SAME QUERY ENTITY.
    The positional_answer changes (now points to the new group position of Ann).

    Args:
        config: The task configuration
        change_answer: If True, replace the answer entity in the counterfactual with a new entity
            from the same pool (different from all entities in the sample).

    Returns:
        CounterfactualExample:
    """
    model = create_positional_entity_causal_model(config)
    input_sample = sample_valid_entity_binding_input(config, model=model)

    query_group = input_sample["query_group"]
    active_groups = input_sample["active_groups"]

    other_groups = [g for g in range(active_groups) if g != query_group]
    if not other_groups:
        import warnings

        warnings.warn(
            f"swap_query_group called with only one active group ({active_groups}). "
            "Falling back to random counterfactual sampling."
        )
        counterfactual = sample_valid_entity_binding_input(config, model=model)
        return CounterfactualExample(
            input=input_sample, counterfactual_inputs=[counterfactual]
        )

    swap_group = random.choice(other_groups)

    # Build counterfactual from input variables
    cf_dict = {var: input_sample[var] for var in model.inputs}

    # Swap entities between query_group and swap_group
    entities_per_group = input_sample["entities_per_group"]
    for e in range(entities_per_group):
        key_query = f"entities[{query_group},{e}]"
        key_swap = f"entities[{swap_group},{e}]"
        cf_dict[key_query], cf_dict[key_swap] = cf_dict[key_swap], cf_dict[key_query]

    # Update query_group: the original query entities are now at swap_group
    cf_dict["query_group"] = swap_group

    if change_answer:
        answer_index = cf_dict["answer_index"]
        answer_key = f"entities[{swap_group},{answer_index}]"

        used_entities = set()
        for g in range(cf_dict["active_groups"]):
            for e in range(cf_dict["entities_per_group"]):
                entity = cf_dict.get(f"entities[{g},{e}]")
                if entity:
                    used_entities.add(entity)

        available = [
            ent for ent in config.entity_pools[answer_index] if ent not in used_entities
        ]

        if available:
            new_answer = random.choice(available)
            cf_dict[answer_key] = new_answer

    counterfactual = model.new_trace(cf_dict)
    return {"input": input_sample, "counterfactual_inputs": [counterfactual]}


def random_counterfactual(config: EntityBindingTaskConfig) -> CounterfactualExample:
    """
    Generate a completely random counterfactual by sampling two independent inputs.

    Baseline condition — the counterfactual is unrelated to the input.

    Args:
        config:
    """
    model = create_positional_entity_causal_model(config)
    input_sample = sample_valid_entity_binding_input(config, model=model)
    counterfactual = sample_valid_entity_binding_input(config, model=model)

    return CounterfactualExample(
        input=input_sample, counterfactual_inputs=[counterfactual]
    )


# ============================================================================
# Zero-arg wrappers using default config
# ============================================================================

_default_config = create_sample_love_config()


def _swap_query_group() -> CounterfactualExample:
    return swap_query_group(_default_config)


def _swap_query_group_change_answer() -> CounterfactualExample:
    return swap_query_group(_default_config, change_answer=True)


def _random_counterfactual() -> CounterfactualExample:
    return random_counterfactual(_default_config)


COUNTERFACTUAL_GENERATORS: dict[str, callable] = {
    "swap_query_group": _swap_query_group,
    "swap_query_group_change_answer": _swap_query_group_change_answer,
    "random_counterfactual": _random_counterfactual,
}


def generate_dataset(causal_model, n: int, seed: int) -> list[CounterfactualExample]:
    """Generate n counterfactual pairs by resampling query_group.

    Each pair has the same entity groups but a different query group,
    causing the query entity and answer to change. This produces clean
    counterfactuals where both positional_answer and raw_output differ.

    Args:
        causal_model (CausalModel): The causal model (used for sampling and trace creation)
        n: Number of examples to generate
        seed: Random seed for reproducibility

    Returns:
        List of CounterfactualExample dicts
    """
    import random as _rng

    _rng.seed(seed)
    examples = []

    for _ in range(n):
        base = sample_valid_entity_binding_input(_default_config, model=causal_model)
        query_group = base["query_group"]
        active_groups = base["active_groups"]
        other_groups = [g for g in range(active_groups) if g != query_group]

        if not other_groups:
            cf = sample_valid_entity_binding_input(_default_config, model=causal_model)
        else:
            cf_group = _rng.choice(other_groups)
            cf_inputs = {var: base[var] for var in causal_model.inputs}
            cf_inputs["query_group"] = cf_group
            cf = causal_model.new_trace(cf_inputs)

        examples.append(CounterfactualExample(input=base, counterfactual_inputs=[cf]))

    return examples
