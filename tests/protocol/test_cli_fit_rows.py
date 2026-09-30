"""``--fit-rows N`` (spec §8, execution scale; §9): the rows-per-grad-forward
bound of a fit, plumbed exactly like ``--batch-rows`` — to the reference
engine's constructor and the run receipt, never to the canonical document,
the points or the digest.
"""

from __future__ import annotations

import json
import sys
import types

import pytest

from causalab.cli import main

from tests.protocol.test_cli import (
    _argv,  # pyright: ignore[reportPrivateUsage]
    _CapturingEngine,  # pyright: ignore[reportPrivateUsage]
    _run_argv,  # pyright: ignore[reportPrivateUsage]
)

pytestmark = pytest.mark.unit


class _FitCapturingEngine(_CapturingEngine):
    """The capturing stub with the ``fit_rows`` constructor argument the
    reference engine takes; ``last`` is this class's own slot."""

    last: "_FitCapturingEngine | None" = None  # pyright: ignore[reportIncompatibleVariableOverride]

    def __init__(
        self,
        *,
        device: str = "cpu",
        batch_rows: int | None = None,
        fit_rows: int | None = None,
    ) -> None:
        super().__init__(device=device, batch_rows=batch_rows)
        self.fit_rows = fit_rows


def _install(monkeypatch, engine_cls):
    """Swap the lazily-imported reference engine module for a stub."""
    stub = types.ModuleType("causalab.neural.engines.pytorch_hooks")
    setattr(stub, "PytorchHooksEngine", engine_cls)  # noqa: B010 — a stub module
    monkeypatch.setitem(sys.modules, "causalab.neural.engines.pytorch_hooks", stub)
    engine_cls.last = None
    return engine_cls


@pytest.fixture
def fit_engine(monkeypatch):
    return _install(monkeypatch, _FitCapturingEngine)


@pytest.fixture
def base_engine(monkeypatch):
    """``test_cli.py``'s own stub, whose constructor takes no ``fit_rows``."""
    return _install(monkeypatch, _CapturingEngine)


def test_fit_rows_goes_to_the_engine_and_the_receipt_not_the_document(
    fit_engine, artifacts_root, tmp_path, capsys
):
    """The bound reaches the engine's constructor and the receipt's
    ``execution`` block, and leaves the canonical document and the points —
    so the digest is the unbounded run's."""
    code = main(
        _run_argv(
            "02_interchange_im.json",
            artifacts_root,
            tmp_path,
            "--fit-rows",
            "8",
            "--record",
        )
    )
    assert code == 0
    assert fit_engine.last.fit_rows == 8
    assert fit_engine.last.batch_rows is None
    record = json.loads((tmp_path / "protocol.json").read_text())
    assert record["execution"] == {
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
    assert "fit_rows" not in json.dumps(record["canonical"])
    assert "fit_rows" not in json.dumps(record["points"])
    capsys.readouterr()
    assert main(_argv("digest", "02_interchange_im.json", artifacts_root)) == 0
    assert capsys.readouterr().out.strip() == record["document_digest"]


def test_fit_rows_and_batch_rows_are_independent_knobs(
    fit_engine, artifacts_root, tmp_path
):
    code = main(
        _run_argv(
            "02_interchange_im.json",
            artifacts_root,
            tmp_path,
            "--batch-rows",
            "4",
            "--fit-rows",
            "16",
            "--record",
        )
    )
    assert code == 0
    assert fit_engine.last.batch_rows == 4
    assert fit_engine.last.fit_rows == 16
    record = json.loads((tmp_path / "protocol.json").read_text())
    assert record["execution"] == {
        "batch_rows": 4,
        "device": "cpu",
        "fit_rows": 16,
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


def test_fit_rows_defaults_to_one_grad_forward_per_cohort(
    fit_engine, artifacts_root, tmp_path
):
    """No flag: the engine is built unbounded and the receipt says ``null``
    under the same key, so a reader of two receipts compares one field."""
    assert (
        main(_run_argv("02_interchange_im.json", artifacts_root, tmp_path, "--record"))
        == 0
    )
    assert fit_engine.last.fit_rows is None
    record = json.loads((tmp_path / "protocol.json").read_text())
    assert record["execution"] == {
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


def test_no_fit_rows_flag_builds_the_engine_without_the_kwarg(
    base_engine, artifacts_root, tmp_path
):
    """The stub in ``test_cli.py`` takes no ``fit_rows``; an unflagged run
    must keep constructing it, so the kwarg is passed only when set."""
    assert (
        main(_run_argv("02_interchange_im.json", artifacts_root, tmp_path, "--record"))
        == 0
    )
    assert base_engine.last is not None
    record = json.loads((tmp_path / "protocol.json").read_text())
    assert record["execution"]["fit_rows"] is None


@pytest.mark.parametrize("bad", ("0", "-2", "many"))
def test_fit_rows_refuses_a_non_positive_count(
    fit_engine, artifacts_root, tmp_path, bad: str
):
    with pytest.raises(SystemExit) as exit_info:
        main(
            _run_argv(
                "02_interchange_im.json", artifacts_root, tmp_path, "--fit-rows", bad
            )
        )
    assert exit_info.value.code == 2
    assert fit_engine.last is None


def test_fit_rows_refuses_an_explicit_nnsight_pin(
    fit_engine, artifacts_root, tmp_path, capsys
):
    """Fail closed, like ``--batch-rows``: the nnsight engine has no grad
    path, so nothing would honour the bound."""
    with pytest.raises(SystemExit) as exit_info:
        main(
            _run_argv(
                "01_harvest_im.json",
                artifacts_root,
                tmp_path,
                "--engine",
                "nnsight",
                "--fit-rows",
                "3",
            )
        )
    assert exit_info.value.code == 2
    err = capsys.readouterr().err
    assert "--engine nnsight" in err and "--fit-rows" in err
    assert fit_engine.last is None


@pytest.mark.parametrize("engine", ("pytorch_hooks", "auto"))
def test_fit_rows_runs_under_the_reference_engine_and_auto(
    fit_engine, artifacts_root, tmp_path, engine: str
):
    code = main(
        _run_argv(
            "02_interchange_im.json",
            artifacts_root,
            tmp_path,
            "--engine",
            engine,
            "--fit-rows",
            "3",
            "--record",
        )
    )
    assert code == 0
    assert fit_engine.last.fit_rows == 3
    record = json.loads((tmp_path / "protocol.json").read_text())
    assert record["execution"]["fit_rows"] == 3
