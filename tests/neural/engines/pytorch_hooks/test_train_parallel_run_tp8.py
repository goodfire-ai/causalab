"""The ``tp=8`` MoE fit of ``test_train_parallel_run.py``, in the golden tier.

The fit is that module's gloo parity on CPU, but its eight ranks and the
world-1 solo together peak at about 8.5 GB resident (measured serially on
2026-09-29), more than a 7 GB CPU machine holds. It is in the golden tier,
so it runs on a machine with more host memory. The CUDA twin,
``tests/golden/test_parallel_worlds.py``'s ``tp=8`` fit, needs eight
devices and skips below that, so on fewer devices this is the one ``tp=8``
fit.
The replicated K/V input-gradient sum itself is also checked in the CPU
tier by ``test_kv_replication.py``; this test adds the end-to-end CLI fit
through the optimizer and the parity band of world 1.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from causalab.cli import main

from tests.neural.engines.pytorch_hooks import test_train_parallel_run as train

pytestmark = [
    pytest.mark.golden,
    pytest.mark.usefixtures("checked_gradients_gloo"),
]

offline = train.offline
never_load = train.never_load
moe_solo = train.moe_solo


def test_tp8_fit_on_the_moe_lands_within_the_band(
    moe_solo: tuple[Path, Path], tmp_path: Path, never_load: None
) -> None:
    """``tp=8`` above the fixture's four KV heads: the K/V projections are
    replicated (docs/model_parallelism.md §6.6) and their **input**
    gradient summed over the tensor group (``kv_replication.py``,
    ``partial_gradient.summed_over``).
    Without this sum, averaging eight partials yields only one eighth of
    the K/V path's gradient."""
    document, solo = moe_solo
    out = tmp_path / "tp8"
    assert main(train._argv(document, out, "--parallel", "tp=8")) == 0  # pyright: ignore[reportPrivateUsage]
    train._assert_parity(solo, out, train._block(tensor=8), "moe tp=8")  # pyright: ignore[reportPrivateUsage]
