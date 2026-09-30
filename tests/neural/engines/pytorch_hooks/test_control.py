"""``train.control`` (spec §2.11): the closed-loop schedule, torch-free.

The load-bearing test is the oracle: a textbook PID, written independently
below in the rate/count parametrization, is driven on zeroed-head counts,
while ours is driven on the kept counts the fit reports — and the two log
weights agree to floating-point precision at every update. Both are the
incremental (velocity) form of a discrete PID law in log space (Åström &
Murray, *Feedback Systems*, 2008, ch. 10). Everything the module docstring
claims about signs and the first update is that one test.
"""

from __future__ import annotations

import math

import pytest

from causalab.neural.engines.pytorch_hooks.control import (
    PidController,
    build_controller,
    ramp_setpoint,
)

pytestmark = pytest.mark.unit


class _TextbookPid:
    """The oracle: an incremental PID in the rate/count parametrization, with a
    log-space update, a clipped derivative and a bounded log multiplier. Before
    any count is observed it takes the actual rate as ``0.0``."""

    def __init__(self, kp, ki, kd, init_mult, mult_min=1e-8, mult_max=1e8, d_clip=5.0):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.log_mult = math.log(max(init_mult, mult_min))
        self.log_mult_min, self.log_mult_max = math.log(mult_min), math.log(mult_max)
        self.d_clip = d_clip
        self._prev_rate_error = 0.0
        self._prev_n_zero = None

    def step(self, actual_rate, target_rate, count_error):
        rate_error = target_rate - actual_rate
        derivative = rate_error - self._prev_rate_error
        derivative = max(-self.d_clip, min(self.d_clip, derivative))
        self._prev_rate_error = rate_error
        u = self.kp * rate_error + self.ki * count_error + self.kd * derivative
        self.log_mult = max(
            self.log_mult_min, min(self.log_mult_max, self.log_mult + u)
        )
        return math.exp(self.log_mult)


def _ours(**overrides) -> PidController:
    base = dict(
        kp=0.1,
        ki=0.001,
        kd=0.0,
        value=0.025,
        space="log",
        bounds=(1e-8, 1e8),
        d_clip=5.0,
        setpoint_before=16.0,
    )
    return PidController(**{**base, **overrides})


def test_the_setpoint_ramp_is_the_anneal_arithmetic():
    assert ramp_setpoint(16, 0, 0.5, 0, 100) == 16.0
    assert ramp_setpoint(16, 0, 0.5, 25, 100) == 8.0
    assert ramp_setpoint(16, 0, 0.5, 50, 100) == 0.0
    assert ramp_setpoint(16, 0, 0.5, 100, 100) == 0.0  # held after the ramp
    assert ramp_setpoint(16, 0, 1.0, 40, 100) == pytest.approx(9.6)


def test_a_signal_above_a_falling_setpoint_raises_the_weight_monotonically():
    control = _ours()
    values = [
        control.step(16.0, ramp_setpoint(16, 0, 0.5, s, 100)) for s in range(1, 21)
    ]
    assert all(b > a for a, b in zip(values, values[1:]))
    assert values[0] > 0.025


def test_a_signal_on_the_setpoint_leaves_the_weight_alone():
    """After the first update. On the first, no rate has been observed yet, so
    the rate error is the ramp's own slope, as if the actual rate were
    ``0.0``, and the weight moves once by ``kp · slope``; from then on a signal
    that tracks the setpoint moves it no further."""
    control = _ours()
    after_first = control.step(15.68, ramp_setpoint(16, 0, 0.5, 1, 100))
    assert after_first == pytest.approx(0.025 * math.exp(0.1 * 0.32))
    for step in range(2, 51):
        setpoint = ramp_setpoint(16, 0, 0.5, step, 100)
        control.step(setpoint, setpoint)
    assert control.value == pytest.approx(after_first)


def test_linear_space_adds_where_log_space_multiplies():
    log_space = _ours(kp=1.0, ki=0.0)
    linear = _ours(kp=1.0, ki=0.0, space="linear", bounds=(0.0, 1e8))
    # one update with a rate error of exactly 1: log ← log + 1, w ← w + 1
    assert log_space.step(16.0, 15.0) == pytest.approx(0.025 * math.e)
    assert linear.step(16.0, 15.0) == pytest.approx(1.025)


def test_bounds_and_the_derivative_clip_bind():
    bounded = _ours(kp=100.0, bounds=(1e-3, 1.0))
    assert bounded.step(16.0, 0.0) == 1.0
    clipped = _ours(kp=0.0, ki=0.0, kd=1.0, d_clip=0.5)
    # first update: rate error 16 − 0 = 16, derivative 16 clipped to 0.5
    assert clipped.step(16.0, 0.0) == pytest.approx(0.025 * math.exp(0.5))
    unclipped = _ours(kp=0.0, ki=0.0, kd=1.0, d_clip=100.0)
    assert unclipped.step(16.0, 0.0) == pytest.approx(0.025 * math.exp(16.0))


def test_the_controller_refuses_an_impossible_setup():
    with pytest.raises(ValueError, match="positive"):
        _ours(bounds=(0.0, 1.0))
    with pytest.raises(ValueError, match="increasing"):
        _ours(bounds=(1.0, 1.0))
    with pytest.raises(ValueError, match="positive initial"):
        _ours(value=0.0)
    with pytest.raises(ValueError, match="space"):
        _ours(space="sqrt")


@pytest.mark.parametrize("kd", [0.0, 0.3])
def test_our_law_is_the_textbook_pid_on_the_kept_count(kd):
    """The oracle. The textbook PID tracks the *zeroed* count against a linear
    target; we track the *kept* count against the mirrored ramp. Same
    gains, same log-space update, same clipped derivative — same weights."""
    units, total = 16, 40
    kp, ki, init = 0.1, 0.001, 0.025
    oracle = _TextbookPid(kp, ki, kd, init_mult=init)
    ours = _ours(kp=kp, ki=ki, kd=kd, value=init, setpoint_before=float(units))
    # a fit that prunes in bursts, stalls, then overshoots the ramp
    n_zero = [0, 0, 1, 3, 3, 3, 4, 8, 8, 9, 12, 12, 12, 13, 16, 16, 16, 16, 15, 16]
    n_zero += [16] * (total - len(n_zero))
    target_rate = units / total  # zeroed units per update
    for step, zero in enumerate(n_zero, start=1):
        actual_rate = (
            0.0 if oracle._prev_n_zero is None else float(zero - oracle._prev_n_zero)
        )
        oracle._prev_n_zero = zero
        target_n_zero = units * min(1.0, step / total)
        oracle_value = oracle.step(actual_rate, target_rate, target_n_zero - zero)
        our_value = ours.step(
            float(units - zero), ramp_setpoint(units, 0, 1.0, step, total)
        )
        assert our_value == pytest.approx(oracle_value, rel=1e-12), f"step {step}"


def test_build_controller_reads_the_documented_defaults():
    spec = {
        "kind": "pid",
        "signal": {"hard_mask_size": "gate"},
        "setpoint": {"ramp": [16, 0, 0.5]},
        "gains": {"kp": 0.1, "ki": 0.001},
    }
    control = build_controller(spec, initial=0.025)
    assert (control.kd, control.space, control.bounds, control.d_clip) == (
        0.0,
        "log",
        (1e-8, 1e8),
        5.0,
    )
    assert control.setpoint_before == 16.0 and control.value == 0.025
