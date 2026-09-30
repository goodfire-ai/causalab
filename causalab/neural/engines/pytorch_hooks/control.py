"""Adjust training hyperparameters from measured gate size.

``train.control`` uses a PID controller to make ``hard_mask_size`` follow a
setpoint. With ``e_t = signal_t - setpoint_t``, the update is::

    u_t = kp*(e_t-e_prev) + ki*e_t + kd*((e_t-e_prev)-(e_prev-e_prev2))

Linear control clips ``w + u_t`` to bounds. Log control clips
``log(w) + u_t`` to log bounds, giving relative weight changes. The count
error bounds the integral term; ``d_clip`` limits the derivative.
At the first update, the previous signal equals the current signal and
the previous setpoint is the ramp's start.

This is the incremental (velocity) form of a discrete PID law; see
Åström & Murray, *Feedback Systems* (2008), ch. 10. The error is
expressed as kept units. The controller uses plain Python and receives
floats from the train loop.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Mapping

__all__ = ["CONTROL_DEFAULTS_ENGINE", "PidController", "ramp_setpoint"]

#: The engine-side view of the controller defaults the canonical form
#: materializes (``schema.CONTROL_DEFAULTS``); read here so a spec built in
#: code without the canonicalizer still runs with the documented values.
CONTROL_DEFAULTS_ENGINE: dict[str, Any] = {
    "kd": 0.0,
    "space": "log",
    "bounds": (1e-8, 1e8),
    "d_clip": 5.0,
}


def ramp_setpoint(
    start: float, end: float, frac: float, step: int, total_steps: int
) -> float:
    """The setpoint after ``step`` updates of ``total_steps``: linear from
    ``start`` to ``end`` over the first ``frac`` of the run, then held — the
    same arithmetic ``train.anneal`` uses for a hyperparameter."""
    ramp_steps = max(1, int(frac * total_steps))
    progress = min(1.0, step / ramp_steps)
    return start + (end - start) * progress


@dataclasses.dataclass
class PidController:
    """One controlled value, moved by a PID on the signal-minus-setpoint
    error (module docstring). ``value`` is the controlled hyperparameter's
    current value; [`step`][] takes one observation and returns the new
    value."""

    kp: float
    ki: float
    kd: float
    value: float
    space: str
    bounds: tuple[float, float]
    d_clip: float
    #: the setpoint before the first update — the ramp's start
    setpoint_before: float
    _previous_signal: float | None = dataclasses.field(default=None, repr=False)
    _previous_setpoint: float | None = dataclasses.field(default=None, repr=False)
    _previous_rate_error: float = dataclasses.field(default=0.0, repr=False)

    def __post_init__(self) -> None:
        if self.space not in ("log", "linear"):
            raise ValueError(f"unknown control space {self.space!r}")
        low, high = self.bounds
        if not low < high:
            raise ValueError(f"control bounds must be increasing, got {self.bounds}")
        if self.space == "log" and low <= 0.0:
            raise ValueError(
                f"log-space control needs positive bounds, got {self.bounds}"
            )
        if self.space == "log" and self.value <= 0.0:
            raise ValueError(
                f"log-space control needs a positive initial value, got {self.value}"
            )
        self.value = self._clip(self.value)

    def _clip(self, value: float) -> float:
        low, high = self.bounds
        return min(high, max(low, value))

    def step(self, signal: float, setpoint: float) -> float:
        """Observe ``signal`` against ``setpoint`` and move the value."""
        previous_signal = (
            signal if self._previous_signal is None else self._previous_signal
        )
        previous_setpoint = (
            self.setpoint_before
            if self._previous_setpoint is None
            else self._previous_setpoint
        )
        error = signal - setpoint
        rate_error = (signal - previous_signal) - (setpoint - previous_setpoint)
        derivative = rate_error - self._previous_rate_error
        derivative = max(-self.d_clip, min(self.d_clip, derivative))
        u = self.kp * rate_error + self.ki * error + self.kd * derivative
        if self.space == "log":
            # clip in log space *before* exponentiating: a large gain must
            # saturate at the bound, not overflow
            low, high = (math.log(b) for b in self.bounds)
            self.value = math.exp(min(high, max(low, math.log(self.value) + u)))
        else:
            self.value = self._clip(self.value + u)
        self._previous_signal = signal
        self._previous_setpoint = setpoint
        self._previous_rate_error = rate_error
        return self.value


def build_controller(spec: Mapping[str, Any], *, initial: float) -> PidController:
    """A controller from its parsed (or canonical) ``train.control`` entry,
    defaults filled from [`CONTROL_DEFAULTS_ENGINE`][]."""
    gains = spec["gains"]
    bounds = spec.get("bounds", CONTROL_DEFAULTS_ENGINE["bounds"])
    start, _end, _frac = spec["setpoint"]["ramp"]
    return PidController(
        kp=float(gains["kp"]),
        ki=float(gains["ki"]),
        kd=float(gains.get("kd", CONTROL_DEFAULTS_ENGINE["kd"])),
        value=float(initial),
        space=str(spec.get("space", CONTROL_DEFAULTS_ENGINE["space"])),
        bounds=(float(bounds[0]), float(bounds[1])),
        d_clip=float(spec.get("d_clip", CONTROL_DEFAULTS_ENGINE["d_clip"])),
        setpoint_before=float(start),
    )
