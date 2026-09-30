"""The band a CPU fit-parity smoke holds a launched fit to, against its
world-1 twin.

Every ``*_run.py`` fit smoke pins the largest difference it measured per
output class (``MEASURED`` in the module) and refuses a drift past twice
that — a re-measurement, never a silent pass. The measurements were taken
on macOS/arm64; Linux x86 BLAS can round the same fp32 fits differently
at the last bit. The rule therefore has a platform floor,
`FIT_BAND_FLOOR`, about two fp32
ulps at unit scale — below which a drift is the platform's, not the
code's. The absolute ``BAND`` each module also asserts is unchanged."""

from __future__ import annotations

__all__ = ["FIT_BAND_FLOOR", "fit_band"]

#: The smallest band a pinned measurement widens to: fp32 reduction noise
#: across BLAS implementations at unit scale.
FIT_BAND_FLOOR = 2e-7


def fit_band(measured: float) -> float:
    """Twice the pinned measurement, never below `FIT_BAND_FLOOR`."""
    return max(measured * 2, FIT_BAND_FLOOR)
