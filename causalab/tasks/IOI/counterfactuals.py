"""Counterfactual generators for the IOI task.

Provides ``generate_dataset(model, n, seed)`` per the loader convention plus
a few zero-arg generators that the baseline analysis introspects for the
counterfactual sanity check.

Each generator returns a [`CounterfactualExample`][]-shaped dict with
``"input"`` (a well-formed IOI trace) and ``"counterfactual_inputs"``
(list of one trace).
"""

from __future__ import annotations

import random

from causalab.causal.counterfactuals import CounterfactualExample
from causalab.causal.model import CausalTrace

from .causal_models import NAMES, positional_causal_model


def _sample_well_formed_input(model=positional_causal_model) -> CausalTrace:
    """Sample a trace where ``name_A != name_B`` and ``name_C ∈ {A, B}``.

    Uses ``model.sample_input`` (which honours the model's input filter) so
    the returned trace is guaranteed well-formed.
    """
    return model.sample_input()


def _swap_name_C(
    input_sample: CausalTrace, model=positional_causal_model
) -> CausalTrace:
    """Counterfactual: keep ``name_A`` / ``name_B`` fixed, flip ``name_C`` to
    the other introduced name. This swaps the IO answer.
    """
    if input_sample["name_C"] == input_sample["name_A"]:
        new_name_C = input_sample["name_B"]
    else:
        new_name_C = input_sample["name_A"]

    cf_inputs = {var: input_sample[var] for var in model.inputs}
    cf_inputs["name_C"] = new_name_C
    return model.new_trace(cf_inputs)


def flip_name_C() -> CounterfactualExample:
    """Generator: flip ``name_C`` → flips the IO answer.

    This is the canonical IOI counterfactual: it isolates the subject-of-action
    signal (``name_C``) from the entity identities (``name_A`` / ``name_B``).
    """
    input_sample = _sample_well_formed_input()
    counterfactual = _swap_name_C(input_sample)
    return {"input": input_sample, "counterfactual_inputs": [counterfactual]}


def random_counterfactual() -> CounterfactualExample:
    """Generator: independently re-sample a full input. Doesn't deconfound
    anything in particular but provides a noise-floor reference for the
    sanity check.
    """
    input_sample = _sample_well_formed_input()
    counterfactual = _sample_well_formed_input()
    return {"input": input_sample, "counterfactual_inputs": [counterfactual]}


def _abc_corruption(
    input_sample: CausalTrace,
    rng: random.Random,
    model=positional_causal_model,
) -> CausalTrace:
    """ABC counterfactual: replace all three name slots with three *distinct*
    random names (Wang et al. 2022, §3.1).

    The base IOI prompt repeats one name (``name_C`` equals ``name_A`` or
    ``name_B``); the ABC source instead fills the A/B/C slots with three different
    random names, destroying the repeated-name structure so the source carries no
    IOI-specific signal while keeping the template, place, and object intact. The
    resulting trace is intentionally *not* a well-formed IOI example
    (``name_C ∉ {name_A, name_B}``); that is fine, because the source is only used
    to read the sender's activation, never graded.
    """
    a, b, c = rng.sample(NAMES, 3)
    cf_inputs = {var: input_sample[var] for var in model.inputs}
    cf_inputs["name_A"], cf_inputs["name_B"], cf_inputs["name_C"] = a, b, c
    return model.new_trace(cf_inputs)


def generate_abc_dataset(model, n: int, seed: int = 42) -> list[CounterfactualExample]:
    """Build ``n`` examples pairing a well-formed IOI base with an ABC corruption.

    This is the path-patching corruption from the IOI paper's §3.1: each base
    input is a normal IOI trace, and its counterfactual source replaces all three
    names with three distinct randoms. ``model`` is accepted for loader-convention
    compatibility but unused (sampling goes through the module-level causal model
    so the input filter is honoured for the base).
    """
    rng = random.Random(seed)
    rng_state = random.getstate()
    random.seed(seed)
    try:
        out: list[CounterfactualExample] = []
        for _ in range(n):
            base = _sample_well_formed_input()
            out.append(
                {"input": base, "counterfactual_inputs": [_abc_corruption(base, rng)]}
            )
        return out
    finally:
        random.setstate(rng_state)


def generate_dataset(model, n: int, seed: int = 42) -> list[CounterfactualExample]:
    """Build ``n`` counterfactual examples by flipping ``name_C``.

    ``model`` is accepted for loader-convention compatibility but unused;
    sampling goes through the module-level causal model so the input filter
    is honoured.
    """
    rng_state = random.getstate()
    random.seed(seed)
    try:
        return [flip_name_C() for _ in range(n)]
    finally:
        random.setstate(rng_state)
