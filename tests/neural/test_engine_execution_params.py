"""The reference engine's execution parameters (spec §8, execution scale):
``batch_rows`` and ``fit_rows`` as constructor defaults, overridden per
request by ``RunContext.execution``, and recorded — from the same
resolution — by [`execution_record`][causalab.protocol.receipt.execution_record].

Constructor, resolution and receipt only: running a fit needs the train loop,
which is the engine tests' business.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from causalab.neural.engines.pytorch_hooks.engine import PytorchHooksEngine
from causalab.protocol.engine import RunContext
from causalab.protocol.parallel import ParallelGeometry
from causalab.protocol.receipt import execution_record

pytestmark = pytest.mark.unit


def _request(tmp_path: Path, **execution: Any) -> RunContext:
    """A request with nothing to run — the engine's resolution reads only
    ``execution``."""
    return RunContext(
        env=None,  # pyright: ignore[reportArgumentType]
        output_dir=tmp_path,
        execution=execution,
    )


# --------------------------------------------------------------------------- #
# constructor
# --------------------------------------------------------------------------- #


def test_fit_rows_is_stored_and_defaults_to_unbounded():
    assert PytorchHooksEngine().fit_rows is None
    assert PytorchHooksEngine(fit_rows=8).fit_rows == 8
    engine = PytorchHooksEngine(batch_rows=4, fit_rows=8)
    assert (engine.batch_rows, engine.fit_rows) == (4, 8)


@pytest.mark.parametrize("bad", (0, -1))
def test_fit_rows_must_be_a_positive_row_count(bad: int):
    with pytest.raises(ValueError, match="fit_rows"):
        PytorchHooksEngine(fit_rows=bad)


def test_execution_request_defaults_to_no_overrides(tmp_path):
    request = RunContext(
        env=None,  # pyright: ignore[reportArgumentType]
        output_dir=tmp_path,
    )
    assert dict(request.execution) == {}


# --------------------------------------------------------------------------- #
# per-request resolution
# --------------------------------------------------------------------------- #


def test_effective_bounds_are_the_engines_without_an_override(tmp_path):
    engine = PytorchHooksEngine(batch_rows=4, fit_rows=8)
    request = _request(tmp_path)
    assert engine.effective_batch_rows(request) == 4
    assert engine.effective_fit_rows(request) == 8


def test_request_execution_overrides_the_engines_bounds(tmp_path):
    engine = PytorchHooksEngine(batch_rows=4, fit_rows=8)
    request = _request(tmp_path, fit_rows=2, batch_rows=16)
    assert engine.effective_fit_rows(request) == 2
    assert engine.effective_batch_rows(request) == 16
    # one key at a time: the other keeps the engine's
    assert (
        PytorchHooksEngine(fit_rows=8).effective_fit_rows(
            _request(tmp_path, fit_rows=4)
        )
        == 4
    )
    assert (
        PytorchHooksEngine(batch_rows=4).effective_batch_rows(
            _request(tmp_path, fit_rows=4)
        )
        == 4
    )


def test_a_present_null_means_unbounded_for_this_request(tmp_path):
    engine = PytorchHooksEngine(batch_rows=4, fit_rows=8)
    request = _request(tmp_path, fit_rows=None, batch_rows=None)
    assert engine.effective_fit_rows(request) is None
    assert engine.effective_batch_rows(request) is None


@pytest.mark.parametrize("bad", (0, -3, "8", 2.0, True))
def test_a_bad_override_is_refused(tmp_path, bad: Any):
    engine = PytorchHooksEngine()
    with pytest.raises(ValueError, match="fit_rows"):
        engine.effective_fit_rows(_request(tmp_path, fit_rows=bad))
    with pytest.raises(ValueError, match="batch_rows"):
        engine.effective_batch_rows(_request(tmp_path, batch_rows=bad))


# --------------------------------------------------------------------------- #
# the receipt
# --------------------------------------------------------------------------- #


def test_execution_record_reports_the_engines_bounds_without_a_request():
    engine = PytorchHooksEngine(batch_rows=4, fit_rows=8)
    assert execution_record(engine) == {
        "batch_rows": 4,
        "device": "cpu",
        "fit_rows": 8,
        "model_source": "loaded",
        "parallel": {
            "data": 1,
            "data_mode": "points",
            "pipeline": 1,
            "context": 1,
            "tensor": 1,
            "expert": 1,
            "world": 1,
            "launcher": "solo",
        },
    }
    assert execution_record(PytorchHooksEngine()) == {
        "batch_rows": None,
        "device": "cpu",
        "fit_rows": None,
        "model_source": "loaded",
        "parallel": {
            "data": 1,
            "data_mode": "points",
            "pipeline": 1,
            "context": 1,
            "tensor": 1,
            "expert": 1,
            "world": 1,
            "launcher": "solo",
        },
    }


def test_execution_record_shows_the_requests_override(tmp_path):
    engine = PytorchHooksEngine(batch_rows=4, fit_rows=8)
    assert execution_record(engine, _request(tmp_path, fit_rows=2)) == {
        "batch_rows": 4,
        "device": "cpu",
        "fit_rows": 2,
        "model_source": "loaded",
        "parallel": {
            "data": 1,
            "data_mode": "points",
            "pipeline": 1,
            "context": 1,
            "tensor": 1,
            "expert": 1,
            "world": 1,
            "launcher": "solo",
        },
    }
    assert execution_record(engine, _request(tmp_path, batch_rows=None)) == {
        "batch_rows": None,
        "device": "cpu",
        "fit_rows": 8,
        "model_source": "loaded",
        "parallel": {
            "data": 1,
            "data_mode": "points",
            "pipeline": 1,
            "context": 1,
            "tensor": 1,
            "expert": 1,
            "world": 1,
            "launcher": "solo",
        },
    }


def test_execution_record_names_the_placement_the_engine_was_built_with():
    """``execution.device`` is the engine's ``device`` at world 1, a comma
    list included. Above world 1 each rank of a CUDA world runs on
    ``cuda:LOCAL_RANK``, so the record names the world's word ``cuda``, and
    a world receipt compares equal to a solo run's on ``cuda``."""
    assert execution_record(PytorchHooksEngine(device="cuda:1"))["device"] == "cuda:1"
    assert execution_record(PytorchHooksEngine(device="cpu,mps"))["device"] == "cpu,mps"

    class CudaRank:
        name = "rank"
        device = "cuda:3"
        parallel = ParallelGeometry(tensor=2)

    class GlooRank(CudaRank):
        device = "cpu"

    assert execution_record(CudaRank())["device"] == "cuda"
    assert execution_record(GlooRank())["device"] == "cpu"  # a CPU world keeps its word


def test_execution_record_tolerates_an_engine_without_the_knob(tmp_path):
    """The nnsight engine declares no ``fit_rows``; the receipt says ``null``
    under the same key so two receipts compare on one field."""

    class Bare:
        name = "bare"

    assert execution_record(Bare()) == {
        "batch_rows": None,
        "device": None,
        "fit_rows": None,
        "model_source": "loaded",
        "parallel": {
            "data": 1,
            "data_mode": "points",
            "pipeline": 1,
            "context": 1,
            "tensor": 1,
            "expert": 1,
            "world": 1,
            "launcher": "solo",
        },
    }
