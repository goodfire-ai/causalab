"""Causal model for the ``hex_color`` perceptual colour-classification task.

The task casts colour naming as a tiny causal DAG over a single stimulus::

    hex → color → raw_output
    hex → raw_input

``hex`` is the input: a hue-jittered ``#RRGGBB`` code drawn from the bundled 600
stimuli (100 per colour × 6). ``color`` is the perceptual label the stimulus
maps to (one of the six colour words), looked up from the bundled data.
``raw_input`` is the filled prompt (the six choices inlined MCQA-style);
``raw_output`` is ``" " + color`` — the word the model is expected to emit.

Design notes:

* **Singleton task.** The stimulus set is fixed (bundled ``sources/hex_color.json``),
  so the model is a module-level ``CAUSAL_MODEL`` constant — no factory config.
* **Periodic hue embedding.** ``color`` carries a 1-D embedding (its hue-centre
  in degrees) with a 360° period, mirroring ``natural_domains_arithmetic``'s
  cyclic ``result``. Manifold/geometry analyses read these; the baseline does
  not require them.
* **Scoring is the colour word.** Unlike MCQA (which scores an option *letter*),
  the answer here *is* the colour word, so the default path already scores "the
  value". The task's ``ScoringSpec`` (``forms`` on ``color``) drives both the
  probability path and the string grader (``ScoringSpec.grader``): all six
  colours are single-token, so plain exact match suffices and the task declares
  no bespoke ``full_string_checker``.
* **``indigo`` dropped (7 → 6).** The source dataset had seven classes; ``indigo``
  (hue 258°, wedged between blue 235° and purple 285°) is excluded because the
  golden fixture (Qwen3-4B-Instruct) labels indigo swatches "purple"
  ~0.999-confident. With every indigo swatch wrong, 7-colour balanced
  accuracy is at most 6/7 ≈ 0.86 (< the 0.9 accuracy floor of the task's
  golden check).
  Dropping it both makes the task viable on the fixture *and* removes the only
  multi-token colour (``indigo`` → ``["ind", "igo"]``), which is why no bespoke
  checker is needed.

Data provenance: the stimuli come from Llama-3.1-8B DAS work
(``<hex-color-das-source>/data.json``),
but only the model-agnostic stimulus content is bundled — no tokenizer/position
fields, and the 100 ``indigo`` rows are excluded at build time. Colours/
hue-centres/template come from that dataset's ``config.json``.
"""

from __future__ import annotations

import json

from causalab.causal import Dom, V, mechanism
from causalab.causal.model import CausalModel
from causalab.causal.scoring import ScoringSpec, build_output_tokens

from .config import (
    COLORS,
    DATA_PATH,
    HUE_CENTERS_DEG,
    HUE_PERIOD,
    PROMPT_TEMPLATE,
)

# ---------------------------------------------------------------------------
# Bundled stimulus data
# ---------------------------------------------------------------------------


def _load_stimuli() -> list[dict]:
    """Load the bundled 600-stimulus set (never reads ``external artifact storage`` at runtime)."""
    with open(DATA_PATH, "r") as f:
        return json.load(f)


STIMULI: list[dict] = _load_stimuli()

# Ordered hex list (the ``hex`` input variable's value domain) and the
# hex → colour-label lookup that drives the ``color`` mechanism.
HEXES: list[str] = [row["hex"] for row in STIMULI]
HEX_TO_LABEL: dict[str, str] = {row["hex"]: row["label"] for row in STIMULI}

# Convenience index used by the counterfactual generators: colour → its hexes.
HEXES_BY_COLOR: dict[str, list[str]] = {c: [] for c in COLORS}
for _row in STIMULI:
    HEXES_BY_COLOR[_row["label"]].append(_row["hex"])


# ---------------------------------------------------------------------------
# Mechanisms
# ---------------------------------------------------------------------------


@mechanism
def equations(hex: Dom(HEXES)):
    color = V(HEX_TO_LABEL[hex], domain=Dom(COLORS))
    raw_input = V(PROMPT_TEMPLATE.replace("{hex}", hex), domain=Dom(str))  # noqa: F841
    raw_output = V(" " + color, domain=Dom(str))  # noqa: F841
    return color


def _build_causal_model() -> CausalModel:
    return CausalModel(
        equations,
        id="hex_color",
        # The answer is the colour word (``raw_output = " " + color``). Declaring
        # the mechanical ``[" red", "red"]`` forms in the task's ``ScoringSpec``
        # drives both the probability path (score-token resolution / per-class
        # distributions) and the exact-match grader. All six colours are
        # single-token, so no bespoke ``full_string_checker`` is needed.
        scoring=ScoringSpec(forms={"color": build_output_tokens(COLORS)}),
        # Periodic hue embedding: each colour sits at its hue-centre on a circle
        # that wraps at 360°.
        embeddings={"color": lambda c: [HUE_CENTERS_DEG[c]]},
        periods={"color": HUE_PERIOD},
    )


CAUSAL_MODEL = _build_causal_model()


# ---------------------------------------------------------------------------
# Standard exports for load_task()
# ---------------------------------------------------------------------------

TARGET_VARIABLE = "color"
TEMPLATE = PROMPT_TEMPLATE
