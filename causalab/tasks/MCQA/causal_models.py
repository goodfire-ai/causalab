"""Causal models for the MCQA task.

This module defines the causal model structure for multiple choice question answering,
including variables, values, parent relationships, and mechanisms.
"""

from causalab.causal import Dom, FamilyDom, V, mechanism
from causalab.causal.model import CausalModel
from causalab.causal.scoring import ScoringSpec, build_output_tokens

# Constants
OBJECTS = [
    "ball",
    "car",
    "house",
    "shirt",
    "flower",
    "pen",
    "cup",
    "hat",
    "bag",
    "shoe",
]
COLORS = [
    "red",
    "blue",
    "green",
    "yellow",
    "purple",
    "orange",
    "pink",
    "brown",
    "black",
    "white",
]

NUM_CHOICES = 2
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
TEMPLATES = [
    "The {object} is {color}. What color is the {object}?"
    + "".join(
        [f"\n{{symbols[{str(i)}]}}. {{choices[{str(i)}]}}" for i in range(NUM_CHOICES)]
    )
    + "\nAnswer:"
]


def _fill_template(template, object, color, choices, symbols):
    filled = template.replace("{object}", object).replace("{color}", color)
    for i in range(NUM_CHOICES):
        filled = filled.replace(f"{{symbols[{i}]}}", symbols[i])
        filled = filled.replace(f"{{choices[{i}]}}", choices[i])
    return filled


@mechanism
def equations(
    template: Dom(TEMPLATES),
    object: Dom(OBJECTS),
    color: Dom(COLORS),
    choices: FamilyDom(Dom(COLORS), size=NUM_CHOICES),
    symbols: FamilyDom(Dom(ALPHABET), size=NUM_CHOICES),
):
    answer_position = V(choices.index(color), domain=Dom(range(NUM_CHOICES)))
    answer = V(symbols[answer_position], domain=Dom(ALPHABET))
    raw_input = V(  # noqa: F841
        _fill_template(template, object, color, choices, symbols), domain=Dom(str)
    )
    raw_output = V(" " + answer, domain=Dom(str))  # noqa: F841
    return answer


# The task's definition of correct, once. Two variables declare forms:
# ``answer`` (the option *letter* — what ``raw_output`` is, ``" " + answer``)
# and ``answer_position`` (the module ``TARGET_VARIABLE``, the variable an
# interchange targets). ``answer_variable="answer"`` says which one the
# generated string is graded against: the former derived checker was keyed on
# ``answer_position``, whose forms are the digits ``" 0"`` / ``" 1"`` the model
# never emits, and only a literal-match fallback on ``raw_output`` made it
# grade letters at all — while the serialized answer forms for the same
# example were the digits. That was the disagreement the scoring differential
# (``tests/tasks/test_scoring_differential.py``) exists to refuse. The retired
# ``score_by: value`` convention (a colour word accepted in place of the
# letter) belonged to a runner that no longer exists; a colour word is now an
# undeclared value and the grader refuses it rather than crediting it.
positional_causal_model = CausalModel(
    equations,
    id=f"{NUM_CHOICES}_answer_MCQA",
    scoring=ScoringSpec(
        forms={
            "answer": build_output_tokens(list(ALPHABET)),
            "answer_position": build_output_tokens(list(range(NUM_CHOICES))),
        },
        answer_variable="answer",
    ),
)


# ---------------------------------------------------------------------------
# Standard exports for load_task()
# ---------------------------------------------------------------------------

CAUSAL_MODEL = positional_causal_model
TARGET_VARIABLE = "answer_position"
TEMPLATE = TEMPLATES[0]
