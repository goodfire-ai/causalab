"""Numerical-tier pin for the IOI task's symbolic outputs.

Re-walks ``generate_dataset(model, n, seed)`` at the seed sequence
fixed in `tests._helpers.task_pins.TASK_SEEDS` and asserts every
sample matches the sidecar ``pinned_samples.json`` byte-for-byte.

This pins the LM-free symbolic layer (``CausalModel`` +
counterfactual generator). It is **not** a "golden" — goldens are the
runner-scope, full-pipeline pins under ``tests/end_to_end/goldens/``
that exercise a small model which reliably solves the task. See
docs/TESTS.md "Task numerical pins" vs "Runner-golden standard" for the
distinction.

Refresh the sidecar via::

    uv run python scripts/update_task_pins.py --task=IOI
    uv run python scripts/update_task_pins.py --task=IOI \\
        --i-have-reviewed-the-diff
"""

from __future__ import annotations

import pytest

from causalab.tasks.IOI import causal_models, counterfactuals  # noqa: F401
from tests._helpers.task_pins import load_pinned_samples, walk_task_samples

pytestmark = pytest.mark.numerical_unit


def test_pinned_samples() -> None:
    expected = load_pinned_samples(__file__)
    actual = walk_task_samples("IOI")

    expected_samples = expected.get("samples", [])
    actual_samples = actual.get("samples", [])

    assert len(actual_samples) == len(expected_samples), (
        f"sample count drift: expected {len(expected_samples)}, "
        f"got {len(actual_samples)}. Regenerate via "
        f"`uv run python scripts/update_task_pins.py --task=IOI "
        f"--i-have-reviewed-the-diff` after reviewing the diff."
    )
    for i, (exp, got) in enumerate(zip(expected_samples, actual_samples)):
        assert got == exp, (
            f"sample[{i}] (seed={exp.get('seed')}) drift. Regenerate via "
            f"`uv run python scripts/update_task_pins.py --task=IOI "
            f"--i-have-reviewed-the-diff` after reviewing the diff."
        )
