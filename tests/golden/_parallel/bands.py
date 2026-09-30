"""The band rule of the parallel golden (``docs/model_parallelism.md`` §10.6),
a pure function of what a capture measures.

A sharded run is the world-1 computation up to the reduction order its
collectives add, so a float output class differs from world 1 by a few units
in the last place of its own dtype at the magnitude it lives at, carried
through the layers above. The band therefore rests on two measured facts
about a class — the largest absolute difference against world 1
(``max_abs_diff``) and the largest magnitude of its world-1 outputs
(``scale``) — and on the outputs' dtype::

    band = max(FACTOR × max_abs_diff, ULPS × ulp(dtype, scale), FLOOR)

- **``FACTOR`` = 3.** The recorded maximum is one draw of the reduction-order
  noise — one document, one kernel selection (cuBLAS chooses by shape) — and a
  recapture after a driver or kernel change needs headroom; three leaves a
  fivefold regression outside whenever the measured maximum is the band's
  binding term (``5 > 3``).
- **``ULPS`` = 2.** Two units in the last place of the output's dtype at its
  scale — one rounding on each side of a re-associated sum — is the smallest
  difference two differently-reduced bf16 values can show, so a class that
  measured exactly zero at logit scale is held to a couple of ulps there
  (``0.5`` at a scale of 40 in bf16) rather than to the absolute floor. For
  an fp32 output (a fitted bundle) the term is ``2^-22`` of its scale and the
  measured maximum decides.
- **``FLOOR`` = 1e-3.** The absolute floor, for a class that measured zero at
  zero scale.

The rule is monotone in both measured inputs, never below the floor, always
holds the measured maximum (``FACTOR ≥ 1``), and an injected fivefold
regression of a class whose measured maximum is binding falls outside —
``tests/golden/test_parallel_record.py`` holds those as hypothesis properties
and the twenty-fold rule as their mutation.

**Routing** is not a float class: near-ties can flip expert assignments
when reduction order changes. Compare the fraction of differing slots,
with a band of twice the measured fraction, bounded between 0.01 and 1.
The factor of two allows variation across captures.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Mapping

__all__ = [
    "FACTOR",
    "FLOOR",
    "Measurement",
    "PRECISION",
    "ROUTING",
    "ROUTING_FACTOR",
    "ROUTING_FLOOR",
    "ULPS",
    "UnknownDtype",
    "band_for",
    "dtype_name",
    "routing_band",
    "ulp",
]

FACTOR = 3
ULPS = 2
FLOOR = 1e-3

ROUTING = "routing"
ROUTING_FACTOR = 2
ROUTING_FLOOR = 0.01

#: Significand precision in bits (the implicit leading one included) per
#: dtype spelling the documents use: the spacing of the representable values
#: in ``[2^e, 2^(e+1))`` is ``2^(e - (p - 1))``.
PRECISION: Mapping[str, int] = {"bf16": 8, "fp16": 11, "fp32": 24, "fp64": 53}


class UnknownDtype(ValueError):
    """A dtype spelling the rule has no precision for."""

    def __init__(self, dtype: str) -> None:
        self.dtype = dtype
        super().__init__(
            f"the band rule knows no dtype {dtype!r}; it knows {sorted(PRECISION)}"
        )


def dtype_name(dtype: Any) -> str:
    """The document spelling of a torch floating dtype (``torch.bfloat16`` →
    ``"bf16"``), refused by name for any other."""
    spelled = {
        "torch.bfloat16": "bf16",
        "torch.float16": "fp16",
        "torch.float32": "fp32",
        "torch.float64": "fp64",
    }.get(str(dtype))
    if spelled is None:
        raise UnknownDtype(str(dtype))
    return spelled


def ulp(dtype: str, scale: float) -> float:
    """One unit in the last place of ``dtype`` at magnitude ``scale``: the
    spacing of its representable values around ``scale``; ``0.0`` at a scale
    of zero (or a non-finite one), so the floor alone holds there.

    Raises:
        UnknownDtype: ``dtype`` is not in `PRECISION`.
    """
    if dtype not in PRECISION:
        raise UnknownDtype(dtype)
    if not math.isfinite(scale) or scale <= 0.0:
        return 0.0
    _, exponent = math.frexp(scale)  # scale = m × 2^exponent, m in [0.5, 1)
    return math.ldexp(1.0, exponent - 1 - (PRECISION[dtype] - 1))


def band_for(max_abs_diff: float, scale: float, dtype: str) -> float:
    """The band a float class is held to (module docstring)."""
    return max(FACTOR * max_abs_diff, ULPS * ulp(dtype, scale), FLOOR)


def routing_band(fraction: float) -> float:
    """The band the routing fraction is held to (module docstring)."""
    return min(1.0, max(ROUTING_FACTOR * fraction, ROUTING_FLOOR))


@dataclasses.dataclass(frozen=True)
class Measurement:
    """What a capture measured for one float class or one of its files: the
    largest absolute difference against the world-1 run, the largest
    magnitude among the world-1 values, and their dtype."""

    max_abs_diff: float
    scale: float
    dtype: str

    @property
    def resolution(self) -> float:
        return ulp(self.dtype, self.scale)

    @property
    def band(self) -> float:
        return band_for(self.max_abs_diff, self.scale, self.dtype)

    def join(self, other: Measurement) -> Measurement:
        """The class measurement over two members: the larger difference,
        and the scale and dtype of the member with the coarser resolution
        (the larger ``ulp``; the larger scale on a tie), so the joined band
        is exactly ``max`` of the members' resolution terms."""
        coarser = max((self, other), key=lambda m: (m.resolution, m.scale, m.dtype))
        return Measurement(
            max(self.max_abs_diff, other.max_abs_diff), coarser.scale, coarser.dtype
        )

    def entry(self) -> dict[str, Any]:
        """The record's entry for a class: the three measured fields and the
        band they yield."""
        return {
            "max_abs_diff": self.max_abs_diff,
            "scale": self.scale,
            "dtype": self.dtype,
            "band": self.band,
        }
