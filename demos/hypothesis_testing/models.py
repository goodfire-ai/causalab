"""Addition modulo three with two candidate intermediate variables."""

from causalab.causal import Dom, V, mechanism
from causalab.causal.model import CausalModel
from causalab.causal.scoring import ScoringSpec


@mechanism
def equations(a: Dom([0, 1, 2]), b: Dom([0, 1, 2]), context: Dom(int)):
    a_value = V(a, domain=Dom([0, 1, 2]))
    b_value = V(b, domain=Dom([0, 1, 2]))
    raw_input = V(f"Case {context}: ({a} + {b}) mod 3 =", domain=Dom(str))  # noqa: F841
    answer = V((a_value + b_value) % 3, domain=Dom([0, 1, 2]))
    raw_output = V(str(answer), domain=Dom(str))  # noqa: F841
    return answer


MODELS = {
    "addition": CausalModel(
        equations,
        id="addition",
        scoring=ScoringSpec(
            forms={"answer": {i: [str(i)] for i in range(3)}}, answer_variable="answer"
        ),
    )
}
DEFAULT_MODEL = "addition"
HYPOTHESES = {"swap_a": ("addition", ["a_value"]), "swap_b": ("addition", ["b_value"])}
TARGETS = ["swap_a"]
