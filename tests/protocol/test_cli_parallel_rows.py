"""The ``rows`` mode of the data axis through the CLI and ``run_protocol``
(``docs/model_parallelism.md`` §8.3, §9), with the reference engine stubbed:
the flag reaches the engine as ``data_mode="rows"``, every replica runs
**every** point (nothing is sharded — the twin ``dp=2`` over points
refuses the one-point DAS document outright), the receipt records
``"data_mode": "rows"``, and the two document rules are refused before any
engine executes: a document with no ``train``, and ``train.batch.pairs``
below the replica count. The dry run reports the same two refusals beside
the model facts, and prints the mode in its ``parallel`` line.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from causalab.cli import main
from causalab.protocol.pipeline import compile_protocol
from causalab.protocol.reports import dry_run
from causalab.protocol.parallel import ParallelGeometry, format_geometry

from tests.protocol._env import CORPUS_DIR, FIXTURES
from tests.protocol.test_cli_parallel import (  # pyright: ignore[reportPrivateUsage]
    HOOKS_MODULE,
    NNSIGHT_MODULE,
    SOLO,
    _Hooks,
    _install,
    _JoinedRankZero,
    _Nnsight,
    _record,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def engines(monkeypatch: pytest.MonkeyPatch) -> None:
    """``test_cli_parallel``'s stubs, installed the same way."""
    _install(monkeypatch, HOOKS_MODULE, "PytorchHooksEngine", _Hooks)
    _install(monkeypatch, NNSIGHT_MODULE, "NnsightEngine", _Nnsight)
    _Hooks.last = None
    _Nnsight.last = None


@pytest.fixture
def as_joined_rank_zero(monkeypatch: pytest.MonkeyPatch):
    """Rank 0 of a joined world with the group entry stubbed
    (``test_cli_parallel``'s fixture): a world above 1 is a launched world."""
    from causalab.neural.shared.parallel import launcher

    def preset(world: int) -> None:
        for name, value in (
            ("WORLD_SIZE", str(world)),
            ("RANK", "0"),
            ("LOCAL_RANK", "0"),
            ("MASTER_ADDR", "127.0.0.1"),
            ("MASTER_PORT", "29500"),
        ):
            monkeypatch.setenv(name, value)
        monkeypatch.delenv(launcher.LAUNCHER_VARIABLE, raising=False)
        monkeypatch.setattr(
            launcher,
            "enter",
            lambda launch, geometry, device: _JoinedRankZero(geometry.data),
        )
        monkeypatch.setattr(launcher, "leave", lambda publisher, status=0: None)

    return preset


DAS = "04_das_im.json"
INTERCHANGE = "02_interchange_im.json"
ROWS2 = ParallelGeometry(data=2, data_mode="rows")


def _run_argv(name: str, artifacts_root: Path, out: Path, *extra: str) -> list[str]:
    """``run`` through the ``auto`` engine with the receipt asked for
    (``--record``: the receipt is the engine's to write, off by default)."""
    return [
        "run",
        str(CORPUS_DIR / name),
        "--data-root",
        str(FIXTURES / "data"),
        "--artifacts-root",
        str(artifacts_root),
        "--engine",
        "auto",
        "--record",
        "--out",
        str(out),
        *extra,
    ]


def _dry_run_argv(name: str, artifacts_root: Path, *extra: str) -> list[str]:
    return [
        "dry-run",
        str(CORPUS_DIR / name),
        "--data-root",
        str(FIXTURES / "data"),
        "--artifacts-root",
        str(artifacts_root),
        "--engine",
        "auto",
        *extra,
    ]


def _compiled(env, name: str):
    return compile_protocol(CORPUS_DIR / name, env=env)


# --------------------------------------------------------------------------- #
# run: the mode reaches the engine, every replica runs every point
# --------------------------------------------------------------------------- #


def test_rows_reaches_the_engine_and_every_point_reaches_every_replica(
    engines, artifacts_root, tmp_path, as_joined_rank_zero
) -> None:
    """The one-point DAS document at ``dp=2:rows`` runs — the twin, ``dp=2``
    over points, is refused by name since two replicas cannot shard one
    point — and the replica runs the whole campaign: nothing is sharded."""
    as_joined_rank_zero(2)
    assert (
        main(_run_argv(DAS, artifacts_root, tmp_path, "--parallel", "dp=2:rows")) == 0
    )
    assert _Hooks.last is not None and _Hooks.last.parallel == ROWS2
    run = _Hooks.last.run
    assert run is not None and run.points is None
    assert _Hooks.last.steps is not None and len(_Hooks.last.steps) == 1
    assert _Hooks.last.shard == range(1)  # the shard is the selection
    record = _record(tmp_path)
    assert record["execution"]["parallel"] == {
        **SOLO,
        "data": 2,
        "data_mode": "rows",
        "world": 2,
        "launcher": "joined",
    }
    assert "rows" not in json.dumps(record["canonical"])
    assert "data_mode" not in json.dumps(record["points"])


def test_points_over_a_one_point_document_is_refused_where_rows_runs(
    engines, artifacts_root, tmp_path, as_joined_rank_zero, capsys
) -> None:
    """The refusal is the engine's shard arithmetic (``publish.point_shard``):
    the engine is entered and executes nothing — no step of its own, no
    receipt."""
    as_joined_rank_zero(2)
    assert main(_run_argv(DAS, artifacts_root, tmp_path, "--parallel", "dp=2")) == 1
    assert "--parallel.data" in capsys.readouterr().err
    assert _Hooks.last is not None and _Hooks.last.shard is None
    assert not (tmp_path / "protocol.json").exists()


def test_the_receipt_says_points_at_world_one(
    engines, artifacts_root, tmp_path
) -> None:
    assert main(_run_argv(INTERCHANGE, artifacts_root, tmp_path)) == 0
    assert _record(tmp_path)["execution"]["parallel"]["data_mode"] == "points"


# --------------------------------------------------------------------------- #
# run: the two document rules, before any engine executes
# --------------------------------------------------------------------------- #


def test_rows_on_a_document_without_train_is_refused_before_execution(
    engines, artifacts_root, tmp_path, as_joined_rank_zero, capsys
) -> None:
    as_joined_rank_zero(2)
    code = main(
        _run_argv(INTERCHANGE, artifacts_root, tmp_path, "--parallel", "dp=2:rows")
    )
    assert code == 1
    err = capsys.readouterr().err
    assert "[P4]" in err and "--parallel.data" in err and "declares no train" in err
    assert _Hooks.last is not None and _Hooks.last.run is None
    assert not (tmp_path / "protocol.json").exists()


def test_rows_with_pairs_below_the_replicas_is_refused_beside_its_twin(
    engines, artifacts_root, tmp_path, as_joined_rank_zero, capsys
) -> None:
    as_joined_rank_zero(2)
    accepted = tmp_path / "accepted"
    assert (
        main(
            _run_argv(
                DAS,
                artifacts_root,
                accepted,
                "--parallel",
                "dp=2:rows",
                "--set",
                "train.batch.pairs=2",
            )
        )
        == 0
    )
    refused = tmp_path / "refused"
    code = main(
        _run_argv(
            DAS,
            artifacts_root,
            refused,
            "--parallel",
            "dp=2:rows",
            "--set",
            "train.batch.pairs=1",
        )
    )
    assert code == 1
    err = capsys.readouterr().err
    assert "train.batch.pairs=1" in err and "at least dp" in err
    assert not (refused / "protocol.json").exists()


# --------------------------------------------------------------------------- #
# dry-run: the same two rules beside the model facts
# --------------------------------------------------------------------------- #


def test_the_dry_run_reports_the_rows_rules(env) -> None:
    accepted = dry_run(_compiled(env, DAS), env, parallel=ROWS2)
    assert accepted.parallel is not None
    assert accepted.parallel.refusals == () and accepted.ok

    no_train = dry_run(_compiled(env, INTERCHANGE), env, parallel=ROWS2)
    assert no_train.parallel is not None
    assert len(no_train.parallel.refusals) == 1
    assert "declares no train" in no_train.parallel.refusals[0]
    assert not no_train.ok
    assert [r.code for r in no_train.refusals] == ["P4"]

    # the model facts come first, the document rule after them
    both = dry_run(
        _compiled(env, INTERCHANGE),
        env,
        parallel=ParallelGeometry(data=2, expert=2, data_mode="rows"),
    )
    assert both.parallel is not None
    assert [text.split(":")[0] for text in both.parallel.refusals] == [
        "--parallel.expert",
        "--parallel.data",
    ]

    # points never reads the document
    points = dry_run(
        _compiled(env, INTERCHANGE), env, parallel=ParallelGeometry(data=2)
    )
    assert points.parallel is not None and points.parallel.refusals == ()


def test_the_dry_run_cli_prints_the_mode(artifacts_root, capsys) -> None:
    assert main(_dry_run_argv(DAS, artifacts_root, "--parallel", "dp=2:rows")) == 0
    out = capsys.readouterr().out
    assert f"parallel  {format_geometry(ROWS2)} (world 2): accepted" in out
    assert "dp=2:rows,pp=1,cp=1,tp=1,ep=1" in out

    assert (
        main(_dry_run_argv(INTERCHANGE, artifacts_root, "--parallel", "dp=2:rows")) == 1
    )
    captured = capsys.readouterr()
    assert "refused (1)" in captured.out
    assert "declares no train" in captured.out + captured.err
