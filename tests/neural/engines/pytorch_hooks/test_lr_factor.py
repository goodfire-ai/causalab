"""The training loop's learning-rate factor against HF's scheduler: pinned
values on a fixed schedule (``numerical_unit``), moved out of
``test_train.py`` so that module carries one tier.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.numerical_unit


def test_lr_factor_is_hfs_linear_warmup_then_decay() -> None:
    """``get_linear_schedule_with_warmup`` at 10 % warm-up over 100 updates: 0 at
    the first update, 1 at the end of the warm-up, 0 after the last update;
    linear on both sides. No warm-up: a straight decay from 1."""
    from causalab.neural.engines.pytorch_hooks.train import lr_factor

    assert lr_factor(0, 100, 0.1) == 0.0
    assert lr_factor(5, 100, 0.1) == pytest.approx(0.5)
    assert lr_factor(10, 100, 0.1) == pytest.approx(1.0)
    assert lr_factor(55, 100, 0.1) == pytest.approx(0.5)
    assert lr_factor(100, 100, 0.1) == 0.0
    assert lr_factor(0, 100, 0.0) == pytest.approx(1.0)
    assert lr_factor(50, 100, 0.0) == pytest.approx(0.5)
    # HF's own scheduler agrees at every update
    from transformers import get_linear_schedule_with_warmup

    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([p], lr=1.0)
    hf = get_linear_schedule_with_warmup(
        opt, num_warmup_steps=10, num_training_steps=100
    )
    for step in range(100):
        assert opt.param_groups[0]["lr"] == pytest.approx(lr_factor(step, 100, 0.1))
        opt.step()
        hf.step()
