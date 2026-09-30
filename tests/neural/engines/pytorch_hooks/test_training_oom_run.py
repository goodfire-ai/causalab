"""Abort a real sharded model's training forward after a rank-local OOM."""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
import torch

from causalab.neural.engines.pytorch_hooks.budget import DistributedOutOfMemory
from causalab.neural.engines.pytorch_hooks.loading import load_model
from causalab.neural.engines.pytorch_hooks.sharding import Sharding
from causalab.neural.engines.pytorch_hooks.train import run_cohort_training
from causalab.neural.shared.parallel.collective import Collective, TorchCollective
from causalab.neural.shared.parallel.fragments import Fragments
from causalab.protocol.parallel import ParallelGeometry
from tests._helpers.gloo_world import GlooWorld, RankCrashed
from tests.neural.engines.pytorch_hooks._drive import executor_for
from tests.neural.engines.pytorch_hooks.conftest import TINY_LLAMA
from tests.neural.engines.pytorch_hooks.test_fit_cohort import _request, _train_doc
from tests.neural.engines.pytorch_hooks.test_train import (
    ANSWERS,
    BASES,
    COUNTERFACTUALS,
)

pytestmark = pytest.mark.smoke

_MARK_VARIABLE = "CAUSALAB_TEST_TRAINING_OOM_MARK"


def _fail_before_projection(module: Any, inputs: Any) -> None:
    raise torch.OutOfMemoryError("injected before the first sharded projection")


def _train_with_one_rank_failing(rank: int, collective: Collective) -> None:
    assert isinstance(collective, TorchCollective)
    torch.set_num_threads(1)
    bundle = load_model(
        TINY_LLAMA,
        device=str(collective.device),
        sharding=Sharding.from_mesh(collective.mesh),
    )
    raw = _train_doc(epochs=1, eval_every=None)
    executor = executor_for(
        raw,
        bundle,
        base_texts=BASES,
        counterfactual_texts=COUNTERFACTUALS,
        extra_columns={"label": ANSWERS},
    )
    executor.fragments = Fragments(collective)
    if rank == 1:
        bundle.blocks[0].self_attn.q_proj.register_forward_pre_hook(
            _fail_before_projection
        )
    try:
        run_cohort_training([executor.doc], [executor], _request(), fit_rows=2)
    except DistributedOutOfMemory as error:
        Path(os.environ[_MARK_VARIABLE]).write_text(str(error))
        raise


def test_sharded_forward_oom_aborts_without_a_retry_collective(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mark = tmp_path / "oom.txt"
    monkeypatch.setenv(_MARK_VARIABLE, str(mark))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    with pytest.raises(RankCrashed) as failure:
        GlooWorld(ParallelGeometry(tensor=2)).run(_train_with_one_rank_failing)
    assert mark.exists(), str(failure.value)
    assert "collective sequence cannot be retried safely" in mark.read_text()
