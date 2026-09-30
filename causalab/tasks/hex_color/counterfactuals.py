"""Counterfactual dataset generators for the ``hex_color`` task.

The stimulus is the hex code; the answer is its perceptual colour. The natural
counterfactuals therefore swap the stimulus:

* ``different_color`` — the counterfactual hex has a **different** colour label,
  so intervening on ``color`` changes the answer. This is the deconfounding
  generator ``generate_dataset`` uses.
* ``same_color_different_hex`` — a different hex with the **same** colour label
  (the answer is unchanged; isolates "which hex" from "which colour").
* ``random`` — two independent stimuli.

All generators sample from the module-level ``CAUSAL_MODEL`` (the ``model``
argument to ``generate_dataset`` is accepted for loader-convention parity but
unused, mirroring MCQA).

``generate_dataset`` is seed-reproducible via a local ``random.Random(seed)``
(never touching the global RNG). The zero-arg wrappers in
``COUNTERFACTUAL_GENERATORS`` each build a fresh unseeded ``random.Random()`` —
a single sample apiece for introspection.
"""

from __future__ import annotations

import random

from causalab.causal.counterfactuals import CounterfactualExample

from .causal_models import CAUSAL_MODEL, COLORS, HEX_TO_LABEL, HEXES, HEXES_BY_COLOR


def _sample_hex(rng: random.Random) -> str:
    return rng.choice(HEXES)


def _different_color(rng: random.Random) -> CounterfactualExample:
    """Base + a counterfactual whose colour label differs from the base's.

    Intervening on ``color`` (or resampling the stimulus) changes the answer.
    """
    base_hex = _sample_hex(rng)
    base_color = HEX_TO_LABEL[base_hex]
    other_colors = [c for c in COLORS if c != base_color]
    cf_color = rng.choice(other_colors)
    cf_hex = rng.choice(HEXES_BY_COLOR[cf_color])
    return {
        "input": CAUSAL_MODEL.new_trace({"hex": base_hex}),
        "counterfactual_inputs": [CAUSAL_MODEL.new_trace({"hex": cf_hex})],
    }


def _same_color_different_hex(rng: random.Random) -> CounterfactualExample:
    """Base + a counterfactual with the same colour but a different hex stimulus."""
    base_hex = _sample_hex(rng)
    base_color = HEX_TO_LABEL[base_hex]
    candidates = [h for h in HEXES_BY_COLOR[base_color] if h != base_hex]
    cf_hex = rng.choice(candidates)
    return {
        "input": CAUSAL_MODEL.new_trace({"hex": base_hex}),
        "counterfactual_inputs": [CAUSAL_MODEL.new_trace({"hex": cf_hex})],
    }


def _random_counterfactual(rng: random.Random) -> CounterfactualExample:
    """Two independently sampled stimuli."""
    return {
        "input": CAUSAL_MODEL.new_trace({"hex": _sample_hex(rng)}),
        "counterfactual_inputs": [CAUSAL_MODEL.new_trace({"hex": _sample_hex(rng)})],
    }


# Zero-arg introspection wrappers — each builds a fresh unseeded RNG so a call
# yields a single independent sample without touching global RNG state.
def different_color() -> CounterfactualExample:
    return _different_color(random.Random())


def same_color_different_hex() -> CounterfactualExample:
    return _same_color_different_hex(random.Random())


def random_counterfactual() -> CounterfactualExample:
    return _random_counterfactual(random.Random())


COUNTERFACTUAL_GENERATORS = {
    "different_color": different_color,
    "same_color_different_hex": same_color_different_hex,
    "random": random_counterfactual,
}


def generate_dataset(model, n: int, seed: int = 42) -> list[CounterfactualExample]:
    """Generate ``n`` counterfactual examples via the ``different_color`` swap.

    ``different_color`` cleanly deconfounds ``color`` (every counterfactual
    flips the colour label). ``model`` is accepted for loader-convention
    compatibility but unused — generation reads the module-level ``CAUSAL_MODEL``.
    Seed-reproducible via a local ``random.Random(seed)`` (global RNG untouched).
    """
    rng = random.Random(seed)
    return [_different_color(rng) for _ in range(n)]
