"""
Causal model implementation for entity binding tasks.

The positional entity model searches for the query entity across all groups,
then retrieves from the matched group position. This tests how neural networks
perform entity-based retrieval with an explicit position-finding intermediate.
"""

import random
from typing import Any

from causalab.causal import Dom, FamilyDom, V, family, mechanism
from causalab.causal.model import CausalModel, CausalTrace
from causalab.causal.scoring import ScoringSpec, build_output_tokens

from .config import EntityBindingTaskConfig, create_sample_love_config


def sample_valid_entity_binding_input(
    config: EntityBindingTaskConfig,
    model: CausalModel,
    ensure_positional_uniqueness: bool = True,
) -> CausalTrace:
    """
    Sample a valid input for entity binding causal models.

    Ensures:
    - Active groups have all entities filled
    - Query group is within active groups
    - Query indices and answer index are valid for the group size
    - A question template exists for the query pattern
    - (Optional) Entities at the same position across groups are distinct

    Args:
        config: Task configuration
        model: CausalModel used to create the trace
        ensure_positional_uniqueness: If True, entities at the same position are distinct
            across groups. Required for the positional model to avoid ambiguity.

    Returns:
        CausalTrace with input values and computed variables
    """
    max_attempts = 100

    for _ in range(max_attempts):
        active_groups = config.max_groups  # Always use max groups for simplicity
        query_group = random.randint(0, active_groups - 1)

        if config.fixed_query_indices is not None:
            query_indices = config.fixed_query_indices
        else:
            query_indices = tuple(
                [random.randint(0, config.max_entities_per_group - 1)]
            )

        if config.fixed_answer_index is not None:
            answer_index = config.fixed_answer_index
        else:
            answer_index = random.randint(0, config.max_entities_per_group - 1)

        if answer_index in query_indices:
            continue

        if (query_indices, answer_index) not in config.question_templates:
            continue

        input_sample: dict[str, Any] = {
            "query_group": query_group,
            "query_indices": query_indices,
            "answer_index": answer_index,
            "active_groups": active_groups,
            "entities_per_group": config.max_entities_per_group,
        }

        used_entities_per_group = [set() for _ in range(active_groups)]
        used_entities_per_position = [
            set() for _ in range(config.max_entities_per_group)
        ]

        all_valid = True
        for g in range(active_groups):
            for e in range(config.max_entities_per_group):
                key = f"entities[{g},{e}]"

                if e in config.entity_pools:
                    available = config.entity_pools[e][:]
                    available = [
                        ent
                        for ent in available
                        if ent not in used_entities_per_group[g]
                    ]

                    if ensure_positional_uniqueness:
                        available = [
                            ent
                            for ent in available
                            if ent not in used_entities_per_position[e]
                        ]

                    if not available:
                        all_valid = False
                        break

                    entity = random.choice(available)
                    input_sample[key] = entity
                    used_entities_per_group[g].add(entity)
                    used_entities_per_position[e].add(entity)
                else:
                    input_sample[key] = None

            if not all_valid:
                break

        if all_valid:
            input_sample["statement_template"] = config.statement_template
            # queries[{e}] are computed variables — do not pass them as inputs
            return model.new_trace(input_sample)

    raise ValueError(
        f"Failed to sample valid entity binding input after {max_attempts} attempts. "
        f"Entity pools may be too small for the constraints."
    )


def _matching_positions(role, query_indices, query, active_groups, entities, positions):
    if role not in query_indices or query is None:
        return ()
    return tuple(
        positions[g]
        for g in range(active_groups)
        if entities[g] == query and positions[g] is not None
    )


def _intersection(queries, indices):
    candidates = [set(queries[i]) for i in indices if queries[i]]
    if not candidates:
        return None
    intersection = set.intersection(*candidates)
    return next(iter(intersection)) if len(intersection) == 1 else None


def _render(
    config,
    entities,
    queries,
    query_indices,
    answer_index,
    active_groups,
    entities_per_group,
):
    template = config.build_mega_template(active_groups, query_indices, answer_index)
    values = {}
    for g in range(active_groups):
        for r in range(entities_per_group):
            value = entities[g, r]
            values[f"g{g}_e{r}"] = value if value is not None else f"MISSING_{g}_{r}"
    values["query_entity"] = queries[query_indices[0]]
    for r in range(entities_per_group):
        values[config.entity_roles.get(r, f"entity{r}")] = queries[r]
    return config.fill_template(template, values)


def create_positional_entity_causal_model(
    config: EntityBindingTaskConfig,
) -> CausalModel:
    """Retain the positional retrieval graph; family indices name each node."""
    groups, roles = config.max_groups, config.max_entities_per_group
    keys = [(g, r) for g in range(groups) for r in range(roles)]
    entity_domains = FamilyDom(
        {(g, r): Dom(config.entity_pools.get(r, []) + [None]) for g, r in keys}
    )
    query_domains = FamilyDom(
        {r: Dom(config.entity_pools.get(r, []) + [None]) for r in range(roles)}
    )
    patterns = list(
        dict.fromkeys(
            [(r,) for r in range(roles)] + [key[0] for key in config.question_templates]
        )
    )
    if (
        config.fixed_query_indices is not None
        and config.fixed_query_indices not in patterns
    ):
        patterns.append(config.fixed_query_indices)
    positions = Dom(list(range(groups)) + [None])

    @mechanism
    def equations(
        entities: entity_domains,
        query_group: Dom(range(groups)),
        query_indices: Dom(patterns),
        answer_index: Dom(range(roles)),
        active_groups: Dom(range(groups + 1)),
        entities_per_group: Dom(range(roles + 1)),
        statement_template: Dom([config.statement_template]),
    ):
        queries = family(size=roles, domain=query_domains)
        for r in range(roles):
            queries[r] = (
                entities[query_group, r] if query_group < active_groups else None
            )
        positional_entities = family(keys=keys, domain=positions)
        for g, r in keys:
            positional_entities[g, r] = g if entities[g, r] is not None else None
        question_template = V(
            config.question_templates.get(
                (query_indices, answer_index), "What is the answer?"
            ),
            domain=Dom(str),
        )
        positional_queries = family(
            size=roles, domain=Dom.sequence(Dom(range(groups)), max_length=groups)
        )
        for r in range(roles):
            positional_queries[r] = _matching_positions(
                r,
                query_indices,
                queries[r],
                active_groups,
                tuple([entities[g, r] for g in range(groups)]),
                tuple([positional_entities[g, r] for g in range(groups)]),
            )
        positional_answer = V(
            _intersection(positional_queries, query_indices), domain=positions
        )
        raw_input = V(
            _render(
                config,
                entities,
                queries,
                query_indices,
                answer_index,
                active_groups,
                entities_per_group,
            ),
            domain=Dom(str),
            lazy=True,
        )
        if (
            positional_answer is not None
            and positional_answer < active_groups
            and answer_index < entities_per_group
        ):
            answer = entities[positional_answer, answer_index]
        else:
            answer = None
        raw_output = V(answer if answer is not None else "UNKNOWN", domain=Dom(str))
        return positional_answer

    def valid_observation(trace):
        # Inactive values remain legal interventions. Observational prompts use
        # the configured full grid and one of its supported question patterns.
        return (
            trace["active_groups"] == groups
            and trace["entities_per_group"] == roles
            and (trace["query_indices"], trace["answer_index"])
            in config.question_templates
            and all(
                trace[f"entities[{g},{r}]"] is not None
                for g, r in keys
                if r in config.entity_pools
            )
        )

    model_id = (
        f"entity_binding_positional_entity_"
        f"{config.max_groups}g_{config.max_entities_per_group}e"
    )

    # The answer is a bound entity name (any of the pooled entities). Declare its
    # surface forms once: the deduped union of all entity pools, each as its
    # ``[" entity", "entity"]`` forms. The probability path reads these
    # (dedup of the 12 answer tokens falls out of the distinct form-groups), and
    # the grader uses ``string_mode="prefix"`` — the entity may be followed by
    # continuation tokens under the multi-token (``max_new_tokens=4``) contract,
    # which is exactly what the former checker.py's ``startswith`` accepted.
    #
    # Declared on ``raw_output`` — the variable that *holds* the answer entity —
    # not on ``positional_answer``, the interchange target, whose values are
    # group indices (``0``, ``1``, …). The former declaration keyed entity
    # names under ``positional_answer``: the string checker still graded (its
    # literal fallback never looked the value up) while the serializer, keying
    # each row by the variable's actual value, refused every row as
    # undeclared. One declaration, on the variable it describes, and the two
    # paths cannot disagree (``tests/tasks/test_scoring_differential.py``).
    all_entities: list[str] = []
    for pool in config.entity_pools.values():
        all_entities.extend(pool)
    all_entities = list(dict.fromkeys(all_entities))
    return CausalModel(
        equations,
        id=model_id,
        input_filter=valid_observation,
        scoring=ScoringSpec(
            forms={"raw_output": build_output_tokens(all_entities)},
            string_mode="prefix",
        ),
    )


# Module-level default causal model using the love config
_default_config = create_sample_love_config()
causal_model = create_positional_entity_causal_model(_default_config)

# Required exports for the causalab runner
CAUSAL_MODEL = causal_model
TARGET_VARIABLE = "positional_answer"
