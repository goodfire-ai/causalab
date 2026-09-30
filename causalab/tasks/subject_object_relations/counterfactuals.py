"""Counterfactual generators for the ``subject_object_relations`` task.

Primary generator (``generate_dataset`` — the loader convention): each pair is a
base subject and a counterfactual subject whose **object differs**, so the
answer flips. This is the LRE interchange semantics — patching the object
identity should change the prediction. Templates are cycled by index for prompt
variation (phrasing is not a causal variable — it overrides ``raw_input`` only).

Also provides an independent-resample generator (base and counterfactual drawn
independently; the counterfactual may share the base's object) as a noise-floor
reference, and zero-arg wrappers over the default relation for systems that
introspect ``COUNTERFACTUAL_GENERATORS`` as ``() -> CounterfactualExample``.

``generate_dataset`` / ``generate_resample_dataset`` are seed-reproducible via a
local ``random.Random(seed)`` (never touching the global RNG). The zero-arg
introspection wrappers each build a fresh unseeded ``random.Random()`` — a single
independent sample apiece — so they never touch global RNG state either.
"""

from __future__ import annotations

import random

from causalab.causal.counterfactuals import CounterfactualExample
from causalab.causal.model import CausalTrace

from .causal_models import create_causal_model
from .config import SubjectObjectRelationsConfig

_SUBJECT_PLACEHOLDER = "{subject}"

# Default relation for the zero-arg introspection wrappers (built lazily).
_DEFAULT_RELATION = "word_first_letter"


def _make_trace(model, subject: str, template: str) -> CausalTrace:
    """Build a base trace for ``subject`` and render ``raw_input`` from ``template``.

    ``raw_input`` is overridden (via ``intervene``) so a cycled template varies
    the phrasing without making ``template`` a causal variable — the causal
    model's own ``raw_input`` mechanism uses ``templates[0]``.
    """
    trace = model.new_trace({"subject": subject})
    return trace.intervene("raw_input", template.replace(_SUBJECT_PLACEHOLDER, subject))


def _flip_object_pair(
    model, template: str, rng: random.Random
) -> CounterfactualExample:
    """One base example + a counterfactual subject whose object differs."""
    config: SubjectObjectRelationsConfig = model._in_config  # type: ignore[attr-defined]
    subjects = config.subjects
    subject_to_object = config.subject_to_object

    base_subject = rng.choice(subjects)
    base_object = subject_to_object[base_subject]
    cf_candidates = [s for s in subjects if subject_to_object[s] != base_object]
    # Degenerate single-object relations can't flip; fall back to any subject so
    # the generator still yields a well-formed pair (curation flags such relations).
    cf_subject = rng.choice(cf_candidates) if cf_candidates else rng.choice(subjects)

    return {
        "input": _make_trace(model, base_subject, template),
        "counterfactual_inputs": [_make_trace(model, cf_subject, template)],
    }


def _resample_pair(model, template: str, rng: random.Random) -> CounterfactualExample:
    """Base + an independently resampled counterfactual (object may coincide)."""
    config: SubjectObjectRelationsConfig = model._in_config  # type: ignore[attr-defined]
    subjects = config.subjects
    return {
        "input": _make_trace(model, rng.choice(subjects), template),
        "counterfactual_inputs": [_make_trace(model, rng.choice(subjects), template)],
    }


def generate_dataset(model, n: int, seed: int = 42) -> list[CounterfactualExample]:
    """Generate ``n`` object-flip counterfactual pairs, cycling templates by index."""
    config: SubjectObjectRelationsConfig = model._in_config  # type: ignore[attr-defined]
    templates = config.templates
    n_templates = len(templates)

    rng = random.Random(seed)
    return [_flip_object_pair(model, templates[i % n_templates], rng) for i in range(n)]


def generate_resample_dataset(
    model, n: int, seed: int = 42
) -> list[CounterfactualExample]:
    """Generate ``n`` independent-resample pairs (noise-floor reference)."""
    config: SubjectObjectRelationsConfig = model._in_config  # type: ignore[attr-defined]
    templates = config.templates
    n_templates = len(templates)

    rng = random.Random(seed)
    return [_resample_pair(model, templates[i % n_templates], rng) for i in range(n)]


# ---------------------------------------------------------------------------
# Zero-arg wrappers over the default relation (introspection convention)
# ---------------------------------------------------------------------------


def _get_default_model():
    """Lazily build (and cache) the default-relation causal model.

    Kept lazy so importing this module does not eagerly read a relation JSON —
    only the zero-arg introspection wrappers below touch it.
    """
    model = getattr(_get_default_model, "_model", None)
    if model is None:
        model = create_causal_model(
            SubjectObjectRelationsConfig(relation=_DEFAULT_RELATION)
        )
        _get_default_model._model = model  # type: ignore[attr-defined]
    return model


def flip_object() -> CounterfactualExample:
    """Zero-arg: one object-flip pair over the default relation."""
    model = _get_default_model()
    rng = random.Random()  # fresh, unseeded — a single sample per call
    templates = model._in_config.templates  # type: ignore[attr-defined]
    return _flip_object_pair(model, rng.choice(templates), rng)


def random_counterfactual() -> CounterfactualExample:
    """Zero-arg: one independent-resample pair over the default relation."""
    model = _get_default_model()
    rng = random.Random()  # fresh, unseeded — a single sample per call
    templates = model._in_config.templates  # type: ignore[attr-defined]
    return _resample_pair(model, rng.choice(templates), rng)


COUNTERFACTUAL_GENERATORS = {
    "flip_object": flip_object,
    "random_counterfactual": random_counterfactual,
}
