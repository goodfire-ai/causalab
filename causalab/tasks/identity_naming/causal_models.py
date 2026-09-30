"""Causal models for the identity_naming factory task.

DAG: entity -> result -> raw_output, entity -> raw_input.
Phrasing templates are NOT causal variables — they provide prompt variation
for centroid computation but don't affect the result.
"""

from __future__ import annotations

from typing import Any, Callable

from causalab.causal import Dom, V, mechanism
from causalab.causal.compiler import ConfigurationCopier
from causalab.causal.model import CausalModel
from causalab.causal.scoring import ScoringSpec

from .config import IdentityNamingConfig

# ---------------------------------------------------------------------------
# Factory: create causal model from config
# ---------------------------------------------------------------------------


def create_causal_model(config: IdentityNamingConfig) -> CausalModel:
    """Create a causal model for an identity-naming task.

    The causal model uses the FIRST template as the default. Multiple
    templates are used in generate_dataset for prompt variation, but
    phrasing is not a causal variable.
    """
    config = ConfigurationCopier()(config)
    entities = config.entities
    entity_to_result = config.entity_to_result
    result_values = sorted(set(entity_to_result.values()), key=lambda v: (len(v), v))
    template = config.templates[0]
    output_prefix = config.output_prefix

    @mechanism
    def equations(entity: Dom(entities)):
        result = V(entity_to_result[entity], domain=Dom(result_values))
        raw_input = V(template.format(entity=entity), domain=Dom(str))  # noqa: F841
        raw_output = V(output_prefix + result, domain=Dom(str))  # noqa: F841
        return result

    # Build embeddings
    embeddings: dict[str, Callable[[Any], list[float]]] = {}
    if config.entity_embedding is not None:
        embeddings["entity"] = config.entity_embedding
    else:
        entity_to_idx = {e: i for i, e in enumerate(entities)}
        embeddings["entity"] = lambda v, _m=entity_to_idx: [float(_m[v])]

    if config.result_embedding is not None:
        embeddings["result"] = config.result_embedding
    else:
        result_to_idx = {r: i for i, r in enumerate(result_values)}
        embeddings["result"] = lambda v, _m=result_to_idx: [float(_m[v])]

    # The answer is the canonical name, emitted as ``output_prefix + name``
    # (a single token). Declare that surface form per result value in
    # the task's ``ScoringSpec``: the probability path reads it, and the
    # (exact) grader matches the generated name — replacing the former
    # GET_RESULT_TOKEN_PATTERN/checker.py.
    scoring = ScoringSpec(
        forms={"result": {v: [output_prefix + str(v)] for v in result_values}}
    )

    model = CausalModel(
        equations,
        id=f"identity_naming_{config.domain_type}",
        embeddings=embeddings,
        scoring=scoring,
    )
    model._in_config = config  # type: ignore[attr-defined]
    return model


# ---------------------------------------------------------------------------
# Standard exports for load_task()
# ---------------------------------------------------------------------------

CREATE_CAUSAL_MODEL = create_causal_model
TARGET_VARIABLE = "result"
CYCLIC_VARIABLES: set[str] = set()
EMBEDDINGS: dict[str, Callable] = {}


def GET_VARIABLE_VALUES(model: CausalModel) -> dict[str, list]:
    return {
        "entity": model.values["entity"],
        "result": model.values["result"],
    }


def GET_CYCLIC_VARIABLES(model: CausalModel) -> set[str]:
    return set()


def GET_EMBEDDINGS(model: CausalModel) -> dict[str, Callable]:
    return model.embeddings


def GET_PERIODIC_INFO(model: CausalModel) -> dict[str, int] | None:
    return None


def GET_TEMPLATE(model: CausalModel) -> str:
    config: IdentityNamingConfig = model._in_config  # type: ignore[attr-defined]
    return config.templates[0]
