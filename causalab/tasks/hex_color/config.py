"""Task-specific configuration and constants for the ``hex_color`` task.

``hex_color`` is a perceptual colour-classification task: the model is shown a
hue-jittered hex code and must name the colour it best matches, choosing from
six fixed colour words inlined MCQA-style in the prompt. The six colours sit on
the hue circle, so the answer variable (``color``) carries a *periodic*
embedding (its hue-centre in degrees) with a 360° period — mirroring the cyclic
machinery in ``natural_domains_arithmetic``.

**``indigo`` dropped (7 → 6 colours).** The source dataset defined seven colour
classes; ``indigo`` (hue 258°, between blue 235° and purple 285°) is dropped
because the golden fixture (Qwen3-4B-Instruct) cannot perceptually separate it
— it labels indigo swatches "purple" ~0.999-confident. With every indigo
swatch wrong, 7-colour balanced accuracy is at most 6/7 ≈ 0.86, below the 0.9
accuracy floor of the task's golden check (see ``README.md`` /
``causal_models.py``).

Constants here are the model-agnostic task definition. The 600 stimuli (100 per
colour × 6) live in the bundled ``sources/hex_color.json`` (see ``causal_models.py``);
they were generated for Llama-3.1-8B DAS work, but only the model-agnostic
stimulus content (hex + RGB/HSV + colour label) is consumed — no tokenizer- or
position-specific fields, and the 100 ``indigo`` rows are excluded at build time.
"""

from __future__ import annotations

from pathlib import Path

TASK_NAME = "hex_color"

# The six colour classes, in hue order (red → purple around the circle). This
# is also the fixed order they are inlined as choices in the prompt. ``indigo``
# is intentionally absent (see module docstring).
COLORS = [
    "red",
    "orange",
    "yellow",
    "green",
    "blue",
    "purple",
]

# Hue-circle centre of each colour class, in degrees (0–360). Source:
# ``das/config.json`` ``hue_centers_deg`` (``indigo``'s 258° dropped with the
# class). Used as the periodic embedding of the ``color`` variable; paired with
# ``HUE_PERIOD`` below.
HUE_CENTERS_DEG: dict[str, float] = {
    "red": 0.0,
    "orange": 30.0,
    "yellow": 58.0,
    "green": 120.0,
    "blue": 235.0,
    "purple": 285.0,
}

# Hue is an angle: the colour circle wraps at 360° (purple ≈ 285° is closer to
# red ≈ 0° than a linear reading suggests). Declared as the ``color`` period so
# manifold/geometry analyses treat the class axis as cyclic.
HUE_PERIOD = 360.0

# Prompt template. Adapted from ``das/config.json`` ``prompt_template`` with the
# ``indigo`` choice removed. The six choices are inlined verbatim (fixed order,
# every example), so unlike MCQA the option set never permutes — the only thing
# that varies is the ``{hex}`` code.
PROMPT_TEMPLATE = (
    "Question: Which color name best describes the hex code {hex}? "
    "Choose one of: red, orange, yellow, green, blue, purple.\n"
    "Answer:"
)

# Bundled stimulus file (model-agnostic; committed with the package like IOI's
# sources/). Generated tables live in data/; the inputs a generator reads live
# in sources/. Never read from ``external artifact storage`` at runtime.
DATA_PATH = Path(__file__).resolve().parent / "sources" / "hex_color.json"

# Token-length requirements.
MAX_TASK_TOKENS = 64  # generous headroom for the prompt + a hex code
MAX_NEW_TOKENS = 1  # single-token classification (all 6 colours are single-token)
