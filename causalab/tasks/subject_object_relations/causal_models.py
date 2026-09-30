"""Causal model for the ``subject_object_relations`` factory task.

DAG (mirrors ``identity_naming`` with a subject→object lookup):

    subject ──> object ──> raw_output
       └──────> raw_input

``object`` is a deterministic lookup ``subject_to_object[subject]`` — the LRE
relation's law. Phrasing templates are NOT causal variables; the causal model
renders ``raw_input`` from ``templates[0]`` and ``counterfactuals.py`` provides
template variation by overriding ``raw_input`` per example.

Objects can be multi-token ("Washington D.C."); the answer is graded first-token
/ prefix-aware via the task's ``ScoringSpec`` (``string_mode="prefix"``). The
curation sweep (see README) records which relations are single-token-decodable
and first-token-distinct enough to clear the accuracy gate on a given model.
"""

from __future__ import annotations

from typing import Any, Callable

from causalab.causal import Dom, V, mechanism
from causalab.causal.compiler import ConfigurationCopier
from causalab.causal.model import CausalModel
from causalab.causal.scoring import ScoringSpec, build_output_tokens

from .config import SubjectObjectRelationsConfig

_SUBJECT_PLACEHOLDER = "{subject}"


def _coerce_config(config: Any) -> SubjectObjectRelationsConfig:
    """Accept a config object, a relation-name str, or a mapping with ``relation``.

    ``resolve_task`` (runner) feeds the generic factory path the raw Hydra
    task-config *dict* (``{"relation": ..., "name": ..., ...}``); a direct
    ``load_task(..., task_cfg=...)`` may pass a ``SubjectObjectRelationsConfig``
    or a bare relation name. All three normalise here.
    """
    if isinstance(config, SubjectObjectRelationsConfig):
        return config
    if isinstance(config, str):
        return SubjectObjectRelationsConfig(relation=config)
    try:
        relation = config["relation"]  # dict or OmegaConf DictConfig
    except (TypeError, KeyError) as exc:
        raise ValueError(
            "task_cfg for subject_object_relations must be a "
            "SubjectObjectRelationsConfig, a relation-name string, or a mapping "
            f"carrying a 'relation' key; got {type(config).__name__}."
        ) from exc
    return SubjectObjectRelationsConfig(relation=str(relation))


def create_causal_model(config: Any) -> CausalModel:
    """Create the causal model for one relation."""
    config = ConfigurationCopier()(_coerce_config(config))

    subjects = config.subjects
    subject_to_object = config.subject_to_object
    objects = config.objects  # distinct answer space, order-stable
    template = config.templates[0]
    output_prefix = config.output_prefix

    @mechanism
    def equations(subject: Dom(subjects)):
        object = V(subject_to_object[subject], domain=Dom(objects))
        raw_input = V(template.replace(_SUBJECT_PLACEHOLDER, subject), domain=Dom(str))  # noqa: F841
        raw_output = V(output_prefix + object, domain=Dom(str))  # noqa: F841
        return object

    subject_to_idx = {s: i for i, s in enumerate(subjects)}
    object_to_idx = {o: i for i, o in enumerate(objects)}
    embeddings: dict[str, Callable[[Any], list[float]]] = {
        "subject": lambda v, _m=subject_to_idx: [float(_m[v])],
        "object": lambda v, _m=object_to_idx: [float(_m[v])],
    }

    # The answer is the object string, emitted as ``output_prefix + object``.
    # Objects may be multi-token, so grade prefix-aware (first-token match for
    # a single-token generation; startswith for a longer generation).
    model = CausalModel(
        equations,
        id=f"subject_object_relations_{config.relation}",
        embeddings=embeddings,
        scoring=ScoringSpec(
            forms={"object": build_output_tokens(objects)}, string_mode="prefix"
        ),
    )
    model._in_config = config  # type: ignore[attr-defined]
    return model


# ---------------------------------------------------------------------------
# Standard exports for load_task()
# ---------------------------------------------------------------------------

CREATE_CAUSAL_MODEL = create_causal_model
TARGET_VARIABLE = "object"
CYCLIC_VARIABLES: set[str] = set()
EMBEDDINGS: dict[str, Callable] = {}


def GET_VARIABLE_VALUES(model: CausalModel) -> dict[str, list]:
    return {
        "subject": model.values["subject"],
        "object": model.values["object"],
    }


def GET_CYCLIC_VARIABLES(model: CausalModel) -> set[str]:
    return set()


def GET_EMBEDDINGS(model: CausalModel) -> dict[str, Callable]:
    return model.embeddings


def GET_PERIODIC_INFO(model: CausalModel) -> dict[str, int] | None:
    return None


def GET_TEMPLATE(model: CausalModel) -> str:
    config: SubjectObjectRelationsConfig = model._in_config  # type: ignore[attr-defined]
    return config.templates[0]
