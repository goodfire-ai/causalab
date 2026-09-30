"""Causal models for the natural_domains_arithmetic factory task.

Unified implementation for weekdays, months, and hours domains.
All share the DAG: (entity, number) → result → raw_output.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable

from causalab.causal import Dom, V, mechanism
from causalab.causal.compiler import ConfigurationCopier
from causalab.causal.model import CausalModel
from causalab.causal.scoring import ScoringSpec, build_output_tokens
from causalab.tasks.random_words import get_random_words

from .config import NaturalDomainConfig

# ---------------------------------------------------------------------------
# Factory: create causal model from config
# ---------------------------------------------------------------------------


def create_causal_model(config: NaturalDomainConfig) -> CausalModel:
    """Create a causal model for a natural-domain arithmetic task.

    Args:
        config: NaturalDomainConfig specifying domain, entities, etc.

    Returns:
        CausalModel with variables: entity, number, result, raw_input, raw_output.
    """
    config = ConfigurationCopier()(config)
    entities = config.entities
    numbers = config.numbers
    number_to_int = config.number_to_int
    result_entities = (
        config.result_entities if config.result_entities is not None else entities
    )
    template = config.template
    output_prefix = config.output_prefix

    entity_to_index = {e: i for i, e in enumerate(entities)}
    templates = template if isinstance(template, list) else [template]
    multi_template = isinstance(template, list)

    def compute_base(entity, number):
        if config.compute_result is not None:
            return config.compute_result(entity, number, config)
        idx = (entity_to_index[entity] + number_to_int[number]) % config.modulus
        return result_entities[idx]

    # When number_groups is configured with >1 bin, result becomes a tuple
    # (entity_result, group_index) so centroid computation gets 2D structure.
    has_groups = bool(config.number_groups) and len(config.number_groups or []) > 1
    number_to_group: dict[str, int] = {}
    if has_groups:
        # Narrow the Optional via assert — has_groups already implies non-None.
        assert config.number_groups is not None
        bins = config.number_groups
        for n in numbers:
            n_int = number_to_int[n]
            for i, (lo, hi) in enumerate(bins):
                if lo <= n_int <= hi:
                    number_to_group[n] = i
                    break
        n_groups = len(bins)
        # Result values: all (entity_result, group) combos
        result_values = [(re, g) for re in result_entities for g in range(n_groups)]
    else:
        result_values = list(result_entities)

    def compute_result(entity, number):
        value = compute_base(entity, number)
        return (value, number_to_group[number]) if has_groups else value

    if multi_template:

        @mechanism
        def equations(
            entity: Dom(entities), number: Dom(numbers), template: Dom(templates)
        ):
            result = V(compute_result(entity, number), domain=Dom(result_values))
            raw_input = V(  # noqa: F841
                template.format(entity=entity, number=number), domain=Dom(str)
            )
            raw_output = V(  # noqa: F841
                output_prefix + (result[0] if has_groups else result), domain=Dom(str)
            )
            return result
    else:

        @mechanism
        def equations(entity: Dom(entities), number: Dom(numbers)):
            result = V(compute_result(entity, number), domain=Dom(result_values))
            raw_input = V(  # noqa: F841
                templates[0].format(entity=entity, number=number), domain=Dom(str)
            )
            raw_output = V(  # noqa: F841
                output_prefix + (result[0] if has_groups else result), domain=Dom(str)
            )
            return result

    # Build embeddings
    embeddings: dict[str, Callable[[Any], list[float]]] = {}
    if config.entity_embedding is not None:
        embeddings["entity"] = config.entity_embedding
        if has_groups:
            embeddings["result"] = lambda v, _emb=config.entity_embedding: _emb(
                v[0]
            ) + [float(v[1])]
        else:
            embeddings["result"] = config.entity_embedding
    else:
        embeddings["entity"] = lambda v, _m=entity_to_index: [float(_m[v])]
        re_to_idx = {e: i for i, e in enumerate(result_entities)}
        if has_groups:
            embeddings["result"] = lambda v, _m=re_to_idx: [
                float(_m[v[0]]),
                float(v[1]),
            ]
        else:
            embeddings["result"] = lambda v, _m=re_to_idx: [float(_m[v])]

    # Always provide number embedding
    embeddings["number"] = lambda v, _m=number_to_int: [float(_m[v])]

    # Compute periods for cyclic variables
    periods: dict[str, float] = {}
    if config.cyclic and config.modulus is not None:
        periods["entity"] = config.modulus
        has_groups = config.number_groups and len(config.number_groups) > 1
        if has_groups:
            periods["result_0"] = config.modulus
        else:
            periods["result"] = config.modulus
        if config.number_is_cyclic:
            periods["number"] = config.modulus

    # Declare the answer's surface forms per result value. The answer is
    # the result entity, emitted as ``output_prefix + entity``; the case-sensitive
    # ``[" entity", "entity"]`` forms cover both BPE spacings (the grader's
    # lowercase tolerance lives in the probability path, not the declaration).
    # 1D: keyed by the entity. 2D (number_groups): keyed by the (entity, group)
    # tuple — all groups of one entity share its forms, so form-groups collapse
    # the N_entities × N_groups tuples back to N_entities score tokens. That
    # shared-form-group dedup replaces the former output_token_values override.
    if has_groups:
        forms = {
            "result": {
                (re, g): build_output_tokens([re])[re]
                for re in result_entities
                for g in range(n_groups)
            }
        }
    else:
        forms = {"result": build_output_tokens(result_values)}
    scoring = ScoringSpec(forms=forms)

    # For non-cyclic domains with a custom compute_result, some (entity, number)
    # pairs may produce results outside the configured result_entities (e.g.
    # alphabet "letter+N" overflowing past Z). Filter those out at the input
    # level so dataset enumeration respects the boundary.
    input_filter: Callable[[Any], bool] | None = None
    if (
        not config.cyclic
        and config.compute_result is not None
        and config.result_entities is not None
    ):
        valid_results = set(result_values)

        def _input_filter(trace, _compute=compute_result, _valid=valid_results):
            return _compute(trace["entity"], trace["number"]) in _valid

        input_filter = _input_filter

    model = CausalModel(
        equations,
        id=f"natural_domains_arithmetic_{config.domain_type}",
        embeddings=embeddings,
        periods=periods,
        scoring=scoring,
        input_filter=input_filter,
    )
    model._nda_config = config  # type: ignore[attr-defined]
    return model


# ---------------------------------------------------------------------------
# Random baseline factory
# ---------------------------------------------------------------------------


def create_random_causal_model(config: NaturalDomainConfig) -> CausalModel:
    """Create a random-word baseline model for the domain.

    Replaces entities with random words and uses cyclic modular arithmetic.
    """
    config = ConfigurationCopier()(config)
    random_entities = get_random_words(len(config.entities))
    baseline = replace(
        config,
        entities=random_entities,
        result_entities=random_entities,
        modulus=len(random_entities),
        cyclic=True,
        compute_result=None,
        number_groups=None,
        entity_embedding=None,
    )
    model = create_causal_model(baseline)
    model.id = f"natural_domains_arithmetic_{config.domain_type}_random"
    model.periods = {}
    model._nda_config = config
    return model


# ---------------------------------------------------------------------------
# Standard exports for load_task()
# ---------------------------------------------------------------------------

CREATE_CAUSAL_MODEL = create_causal_model
CREATE_RANDOM_CAUSAL_MODEL = create_random_causal_model
TARGET_VARIABLE = "result"

# Static stubs — dynamic getters below override these in the loader
CYCLIC_VARIABLES: set[str] = set()
EMBEDDINGS: dict[str, Callable] = {}


def GET_VARIABLE_VALUES(model: CausalModel) -> dict[str, list]:
    """Derive variable values from the model."""
    return {
        "entity": model.values["entity"],
        "number": model.values["number"],
        "result": model.values["result"],
    }


def GET_CYCLIC_VARIABLES(model: CausalModel) -> set[str]:
    """Derive cyclic variables from the stored config."""
    config: NaturalDomainConfig = model._nda_config  # type: ignore[attr-defined]
    cyclic: set[str] = set()
    if config.cyclic:
        cyclic.add("entity")
        cyclic.add("result")
    if config.number_is_cyclic:
        cyclic.add("number")
    return cyclic


def GET_EMBEDDINGS(model: CausalModel) -> dict[str, Callable]:
    """Return the embeddings dict stored on the model."""
    return model.embeddings


def GET_PERIODIC_INFO(model: CausalModel) -> dict[str, int] | None:
    """Derive period info from the stored config."""
    config: NaturalDomainConfig = model._nda_config  # type: ignore[attr-defined]
    if not config.cyclic:
        return None
    info: dict[str, int] = {}
    modulus = config.modulus
    assert modulus is not None
    info["entity"] = modulus
    # When result is a tuple, extract_parameters_from_dataset expands to
    # result_0 (entity index, cyclic) and result_1 (group index, linear).
    has_groups = config.number_groups and len(config.number_groups) > 1
    if has_groups:
        info["result_0"] = modulus
        # result_1 is linear (group index) — not in periodic_info
    else:
        info["result"] = modulus
    if config.number_is_cyclic:
        info["number"] = modulus
    return info


def GET_TEMPLATE(model: CausalModel) -> str | list[str]:
    """Return the prompt template(s) from the stored config."""
    config: NaturalDomainConfig = model._nda_config  # type: ignore[attr-defined]
    return config.template
