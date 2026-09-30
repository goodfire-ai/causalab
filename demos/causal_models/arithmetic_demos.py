"""Observationally identical modular-addition models with different interventions."""

from causalab.causal import Dom, V, mechanism
from causalab.causal.model import CausalModel


def make_addition(order="left", modulus=10):
    """``(A + B + C) mod modulus``, summed left to right or right to left.

    ``left`` computes ``S = (A + B) % modulus`` and then ``Y = (S + C) %
    modulus``; ``right`` computes ``S = (B + C) % modulus`` and then ``Y = (S +
    A) % modulus``. Both compute the same function of the three digits and
    differ in their intermediate ``S``.
    """
    if order not in ("left", "right"):
        raise ValueError("order must be left or right")
    digits = range(modulus)

    @mechanism
    def equations(A: Dom(digits), B: Dom(digits), C: Dom(digits)):
        if order == "left":
            S = V((A + B) % modulus)
            Y = V((S + C) % modulus)
        else:
            S = V((B + C) % modulus)
            Y = V((S + A) % modulus)
        raw_input = V(  # noqa: F841
            [A, B, C], domain=Dom.sequence(Dom(digits), length=3, container=list)
        )
        raw_output = V(Y)  # noqa: F841
        return Y

    return CausalModel(equations, id=f"add_{order}")
