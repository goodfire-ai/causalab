"""
Causal model definitions for the task.

DAG: (var_1, var_2) → left_equality
     (var_3, var_4) → right_equality
     (left_equality, right_equality) → result_equality → raw_output
"""

from causalab.causal import Dom, Exo, V, mechanism
from causalab.causal.model import CausalModel
from causalab.causal.scoring import ScoringSpec

from .config import LETTERS, TASK_NAME
from .templates import TEMPLATES, fill_template


@mechanism
def equations(
    template: Dom(TEMPLATES),
    var_1: Dom(LETTERS),
    var_2: Dom(LETTERS),
    var_3: Dom(LETTERS),
    var_4: Dom(LETTERS),
    icl_seed: Exo(Dom(range(2**32))),
):
    left_equality = V(var_1 == var_2)
    right_equality = V(var_3 == var_4)
    result_equality = V(left_equality == right_equality)
    raw_input = V(  # noqa: F841
        fill_template(template, var_1, var_2, var_3, var_4, seed=icl_seed),
        domain=Dom(str),
    )
    raw_output = V("1" if result_equality else "0", domain=Dom(str))  # noqa: F841
    return result_equality


# All three equality variables have boolean values, but the model emits the
# digit "1" (True) or "0" (False). Declare those surface forms once, per value
# in the task's one ``ScoringSpec``: the probability path reads
# them, and the grader uses ``string_mode="prefix"`` — ``raw_output`` is the
# bare digit possibly followed by text, so a generation that starts with it is
# correct, exactly as the former checker.py's ``startswith`` was. The graded
# string is ``result_equality``'s digit (``answer_variable``); the other two
# variables declare the same forms for the probability path.
_EQUALITY_VARS = ("left_equality", "right_equality", "result_equality")
_EQUALITY_FORMS: dict[object, list[str]] = {True: [" 1", "1"], False: [" 0", "0"]}

CAUSAL_MODEL = CausalModel(
    equations,
    id=TASK_NAME,
    scoring=ScoringSpec(
        forms={v: dict(_EQUALITY_FORMS) for v in _EQUALITY_VARS},
        answer_variable="result_equality",
        string_mode="prefix",
    ),
)


# ---------------------------------------------------------------------------
# Exports for load_task()
# ---------------------------------------------------------------------------

TARGET_VARIABLE = "result_equality"
