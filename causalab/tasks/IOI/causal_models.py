"""Causal model for the IOI (Indirect Object Identification) task.

Canonical IOI: a sentence introduces two distinct named entities at positions
A and B, then names one of them as the subject (``name_C``) of a giving
action. The model must predict the *other* name — the indirect object (IO).
For example::

    After Alice and Bob went to the park, Alice gave a ball to ____   →   Bob

Input variables:
    template:  prompt template (single canonical form for the coverage runner).
    name_A:    first introduced name.
    name_B:    second introduced name (distinct from ``name_A``).
    name_C:    subject of the giving action — equal to either ``name_A`` or
               ``name_B``.
    place:     filler for ``{place}``.
    object:    filler for ``{object}``.

Computed variables:
    IO:          the indirect object — whichever of ``name_A`` / ``name_B``
                 differs from ``name_C``. The target variable.
    raw_input:   filled template string.
    raw_output:  ``" " + IO`` (the model's expected next-token output).

This is a coverage-oriented task: it exists so the runner pipeline can
anchor a smoke + golden tier on ``causalab.tasks.IOI.*`` (see docs/TESTS.md
"Coverage-oriented runners"). It is **not** a scientifically meaningful
run.
"""

from __future__ import annotations

import json
from pathlib import Path

from causalab.causal import Dom, V, mechanism
from causalab.causal.model import CausalModel, CausalTrace
from causalab.causal.scoring import ScoringSpec, build_output_tokens

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def _load(path: Path) -> list[str]:
    with open(path, "r") as f:
        return json.load(f)


IOI_SOURCES = Path(__file__).resolve().parent / "sources"
NAMES: list[str] = _load(IOI_SOURCES / "names.json")
OBJECTS: list[str] = _load(IOI_SOURCES / "objects.json")
PLACES: list[str] = _load(IOI_SOURCES / "places.json")
ALL_TEMPLATES: list[str] = _load(IOI_SOURCES / "templates.json")

# Coverage runner uses a single canonical template — keeps token-position
# logic simple and avoids the multi-template factory plumbing. The other
# templates remain available via ``ALL_TEMPLATES`` for ad-hoc work.
CANONICAL_TEMPLATE: str = (
    "After {name_A} and {name_B} went to the {place}, {name_C} gave a {object} to"
)
TEMPLATES: list[str] = [CANONICAL_TEMPLATE]


# ---------------------------------------------------------------------------
# Mechanisms
# ---------------------------------------------------------------------------


@mechanism
def equations(
    template: Dom(TEMPLATES),
    name_A: Dom(NAMES),
    name_B: Dom(NAMES),
    name_C: Dom(NAMES),
    place: Dom(PLACES),
    object: Dom(OBJECTS),
):
    IO = V(name_A if name_C == name_B else name_B, domain=Dom(NAMES))
    raw_input = V(
        template.format(
            name_A=name_A, name_B=name_B, name_C=name_C, place=place, object=object
        ),
        domain=Dom(str),
    )
    raw_output = V(" " + IO, domain=Dom(str))
    return IO


def _input_filter(t: CausalTrace) -> bool:
    """Reject inputs where the IOI question is ill-formed.

    Requires ``name_A != name_B`` (distinct entities) and
    ``name_C in {name_A, name_B}`` (subject is one of the introduced names).
    """
    name_A = t["name_A"]
    name_B = t["name_B"]
    name_C = t["name_C"]
    return name_A != name_B and name_C in (name_A, name_B)


# ---------------------------------------------------------------------------
# Causal model
# ---------------------------------------------------------------------------


def _build_causal_model() -> CausalModel:
    return CausalModel(
        equations,
        id="ioi",
        # The answer is the indirect-object name (``raw_output = " " + IO``);
        # the model emits a single name token, so exact match on the declared
        # ``[" name", "name"]`` forms is the right grader — the task's
        # ``ScoringSpec``, in place of the former checker.py.
        scoring=ScoringSpec(forms={"IO": build_output_tokens(NAMES)}),
        input_filter=_input_filter,
    )


positional_causal_model = _build_causal_model()


# ---------------------------------------------------------------------------
# Standard exports for load_task()
# ---------------------------------------------------------------------------

CAUSAL_MODEL = positional_causal_model
TARGET_VARIABLE = "IO"
TEMPLATE = CANONICAL_TEMPLATE
