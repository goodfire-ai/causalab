"""Data parallelism over points through the real CLI (``docs/model_parallelism.md``
§3, §8.3, §9): ``--parallel dp=2 --device cpu`` on the tiny Llama.

The fan-out fixture's scan retargeted to tiny Llama — four points on two axes
(``positions.tap`` × ``sites.target.layers``), two metric tables, one saved
read — runs once at world 1 and once at ``dp=2``. The parent of the spawn
never loads a model (its ``load_model`` is monkeypatched to raise; the
children are fresh processes), the two ``gloo`` children each run their
half of the points, and replica 0's rank joins the shards by point digest.
The assertion is **byte identity**: every table and every safetensors file
equal to the world-1 run's bytes, the receipt equal minus its
``execution.parallel`` block, which says ``"launcher": "spawned"``,
``"data": 2``; the event stream equal event for event minus its timestamps
and the ``forwards`` a shard-wise interning legitimately changes. The
``joined`` variant presets the environment the way ``torchrun`` does and
starts the two ranks itself; ``--points`` composes (the replicas shard the
selected range); and the parent reports a failed child's rank and status.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import pytest
import torch
import torch.multiprocessing as mp

from causalab.cli import main
from causalab.neural.engines.pytorch_hooks import engine as hooks_engine
from causalab.neural.shared.parallel import launcher
from causalab.neural.shared.parallel.launcher import spawn
from causalab.protocol.parallel import ParallelGeometry
from causalab.neural.shared.parallel.spawn import reserve_port

# `parallel_world`: every test here runs a document or a fit across a spawned
# multi-rank process world — minutes on the CI runner. The PR gate deselects the
# marker; the nightly CPU job runs it (docs/TESTS.md).
pytestmark = [pytest.mark.smoke, pytest.mark.parallel_world]

REPO = Path(__file__).resolve().parents[4]
SCAN = REPO / "tests" / "workflow" / "fixtures" / "fan_out" / "protocols" / "scan.json"
DATA = REPO / "tests" / "protocol" / "fixtures" / "data"
TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"
TINY_REVISION = "9fb191250dd56d0ba7ec9785a025ed29c03d5998"

TABLES = ("iia.json", "logit_diff.json")
TENSORS = ("v_cf.safetensors",)
RECEIPT = "protocol.json"
EVENTS = "events.jsonl"


def _document(tmp: Path) -> Path:
    """The scan on tiny Llama's two layers, plus the counterfactual read
    saved as a tensor file so the join is held to safetensors bytes too."""
    doc = json.loads(SCAN.read_text())
    doc["model"] = {"key": TINY, "revision": TINY_REVISION, "dtype": "fp32"}
    doc["method"]["sites"]["target"]["layers"] = {"sweep": [0, 1]}
    # the counterfactual is read on the corpus's un-intervened model (§2.9)
    doc["method"]["save"].append(
        {
            "read": "v_cf",
            "model": "original_counterfactual",
            "file_path": "v_cf.safetensors",
        }
    )
    target = tmp / "scan.json"
    target.write_text(json.dumps(doc, indent=2))
    return target


def _argv(document: Path, out: Path, *extra: str, device: str = "cpu") -> list[str]:
    """The CLI ``run`` of ``document`` into ``out`` on ``device`` (``cpu``: the
    gloo tier; the CUDA twin, ``tests/golden/test_parallel_worlds.py``, passes
    ``cuda``)."""
    return [
        "run",
        str(document),
        "--engine",
        "pytorch_hooks",
        "--record",
        "--data-root",
        str(DATA),
        "--artifacts-root",
        str(document.parent),
        "--out",
        str(out),
        "--device",
        device,
        *extra,
    ]


def _receipt(out: Path) -> dict[str, Any]:
    return json.loads((out / RECEIPT).read_text())


def _events(out: Path) -> list[tuple[str, dict[str, Any]]]:
    lines = [json.loads(line) for line in (out / EVENTS).read_text().splitlines()]
    return [
        (
            line["event"],
            {k: v for k, v in line["payload"].items() if k != "forwards"},
        )
        for line in lines
    ]


def _assert_byte_identical(solo: Path, parallel: Path, block: dict[str, Any]) -> None:
    """Every artifact of ``parallel`` is ``solo``'s to the byte, the receipt
    minus ``execution.parallel`` (which is ``block``), the stream minus
    timestamps and forward counts."""
    assert sorted(p.name for p in solo.iterdir()) == sorted(
        p.name for p in parallel.iterdir()
    )
    for name in (*TABLES, *TENSORS):
        assert (parallel / name).read_bytes() == (solo / name).read_bytes(), name
    a, b = _receipt(solo), _receipt(parallel)
    assert b["execution"]["parallel"] == block
    assert a["execution"]["parallel"]["launcher"] == "solo"
    del a["execution"]["parallel"], b["execution"]["parallel"]
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    assert "fires" in b and "scoring" in b
    assert _events(solo) == _events(parallel)


@pytest.fixture
def never_load(monkeypatch: pytest.MonkeyPatch) -> None:
    """The parent of a spawn never loads a model: in *this* process the
    loader raises; the children are fresh interpreters and load normally."""

    def never(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the spawn parent entered load_model")

    monkeypatch.setattr(hooks_engine, "load_model", never)


@pytest.fixture(scope="module")
def solo_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """The world-1 run every parallel run is compared against."""
    tmp = tmp_path_factory.mktemp("dp-solo")
    document = _document(tmp)
    out = tmp / "solo"
    assert main(_argv(document, out)) == 0
    return document, out


def test_dp2_spawned_is_byte_identical_to_world_one(
    solo_run: tuple[Path, Path], tmp_path: Path, never_load: None
) -> None:
    document, solo = solo_run
    out = tmp_path / "dp2"
    assert main(_argv(document, out, "--parallel", "dp=2")) == 0
    _assert_byte_identical(
        solo,
        out,
        {
            "data": 2,
            "data_mode": "points",
            "pipeline": 1,
            "context": 1,
            "tensor": 1,
            "expert": 1,
            "world": 2,
            "launcher": "spawned",
        },
    )
    assert len(_receipt(out)["points"]) == 4


def test_dp2_composes_with_points(
    solo_run: tuple[Path, Path], tmp_path: Path, never_load: None
) -> None:
    """``--points 1:4`` at ``dp=2``: the replicas shard the selected three
    points (2 + 1) and the joined run is the world-1 ``--points 1:4`` run."""
    document, _ = solo_run
    solo, parallel = tmp_path / "solo", tmp_path / "dp2"
    monkeypatched = hooks_engine.load_model
    # the world-1 twin loads in this process: restore the loader for it
    import causalab.neural.engines.pytorch_hooks.loading as loading

    hooks_engine.load_model = loading.load_model
    try:
        assert main(_argv(document, solo, "--points", "1:4")) == 0
    finally:
        hooks_engine.load_model = monkeypatched
    assert main(_argv(document, parallel, "--points", "1:4", "--parallel", "dp=2")) == 0
    _assert_byte_identical(
        solo,
        parallel,
        {
            "data": 2,
            "data_mode": "points",
            "pipeline": 1,
            "context": 1,
            "tensor": 1,
            "expert": 1,
            "world": 2,
            "launcher": "spawned",
        },
    )
    assert [p["index"] for p in _receipt(parallel)["points"]] == [1, 2, 3]


def test_more_replicas_than_points_is_refused_by_the_children(
    solo_run: tuple[Path, Path], tmp_path: Path, never_load: None, capsys
) -> None:
    document, _ = solo_run
    code = main(_argv(document, tmp_path / "dp8", "--parallel", "dp=8"))
    assert code == 1
    err = capsys.readouterr().err
    assert "rank" in err
    assert not (tmp_path / "dp8" / RECEIPT).exists()


# --------------------------------------------------------------------------- #
# joined: the environment torchrun sets
# --------------------------------------------------------------------------- #


def _joined_entry(rank: int, argv: Sequence[str], world: int, port: int) -> None:
    """One ``torchrun``-style process: the group variables preset, no spawn
    parent's mark, then the CLI as the user typed it."""
    os.environ.update(
        {
            "WORLD_SIZE": str(world),
            "RANK": str(rank),
            "LOCAL_RANK": str(rank),
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(port),
        }
    )
    os.environ.pop(launcher.LAUNCHER_VARIABLE, None)
    code = main(list(argv))
    if code:
        sys.exit(code)


def test_dp2_joined_is_byte_identical_to_world_one(
    solo_run: tuple[Path, Path], tmp_path: Path
) -> None:
    document, solo = solo_run
    out = tmp_path / "joined"
    with reserve_port() as hold:
        mp.spawn(
            _joined_entry,
            args=(_argv(document, out, "--parallel", "dp=2"), 2, hold.port),
            nprocs=2,
            join=True,
        )
    _assert_byte_identical(
        solo,
        out,
        {
            "data": 2,
            "data_mode": "points",
            "pipeline": 1,
            "context": 1,
            "tensor": 1,
            "expert": 1,
            "world": 2,
            "launcher": "joined",
        },
    )


# --------------------------------------------------------------------------- #
# the parent's status
# --------------------------------------------------------------------------- #


def _exit_with_rank(argv: Sequence[str]) -> int:
    """A child entry that fails on rank 1 with status 3 and succeeds elsewhere."""
    return 3 if os.environ["RANK"] == "1" else 0


def _raise_on_rank_zero(argv: Sequence[str]) -> int:
    if os.environ["RANK"] == "0":
        raise RuntimeError("rank zero fell over")
    return 0


def _record_environment(argv: Sequence[str]) -> int:
    """Each child writes the launch variables it was handed."""
    target = Path(argv[0]) / f"rank{os.environ['RANK']}.json"
    target.write_text(
        json.dumps(
            {
                name: os.environ.get(name)
                for name in (
                    "RANK",
                    "LOCAL_RANK",
                    "WORLD_SIZE",
                    "MASTER_ADDR",
                    "MASTER_PORT",
                    launcher.LAUNCHER_VARIABLE,
                    "OMP_NUM_THREADS",
                )
            }
            # what torch's intra-op pool was actually sized to — the
            # variable is read as torch is imported, so this is the test
            | {"torch_threads": torch.get_num_threads()}
        )
    )
    return 0


def test_the_parent_returns_a_failed_childs_status_naming_the_rank(capsys) -> None:
    code = spawn(ParallelGeometry(data=2), ["--nothing"], entry=_exit_with_rank)
    assert code == 3
    err = capsys.readouterr().err
    assert "rank 1" in err and "3" in err


def test_the_parent_reports_a_child_that_raised(capsys) -> None:
    code = spawn(ParallelGeometry(data=2), ["--nothing"], entry=_raise_on_rank_zero)
    assert code != 0
    err = capsys.readouterr().err
    assert "rank 0" in err and "rank zero fell over" in err


def test_every_child_is_handed_its_launch_variables(tmp_path: Path) -> None:
    assert (
        spawn(ParallelGeometry(data=2), [str(tmp_path)], entry=_record_environment) == 0
    )
    seen = {
        rank: json.loads((tmp_path / f"rank{rank}.json").read_text()) for rank in (0, 1)
    }
    for rank, env in seen.items():
        assert env["RANK"] == env["LOCAL_RANK"] == str(rank)
        assert env["WORLD_SIZE"] == "2"
        assert env["MASTER_ADDR"] == "127.0.0.1"
        assert env["MASTER_PORT"].isdigit()
        assert env[launcher.LAUNCHER_VARIABLE] == "spawned"
    assert seen[0]["MASTER_PORT"] == seen[1]["MASTER_PORT"]
    # the parent's own environment is untouched
    assert (
        "RANK" not in os.environ or os.environ.get(launcher.LAUNCHER_VARIABLE) is None
    )


def _fresh_torch_threads(environ: dict[str, str]) -> int:
    """What torch's intra-op pool comes to in a fresh interpreter under
    ``environ``: the platform's own reading of ``OMP_NUM_THREADS`` — torch
    sizes the pool from it as it is imported, but not verbatim: the Linux
    x86 wheel takes MKL's count, which caps the variable at what MKL will
    use for the cores it sees (a 4-vCPU GitHub runner answers 1 to a 3)."""
    out = subprocess.run(
        [sys.executable, "-c", "import torch; print(torch.get_num_threads())"],
        env=environ,
        check=True,
        capture_output=True,
        text=True,
    )
    return int(out.stdout.strip())


@pytest.mark.parametrize("parent", [None, "3"], ids=["bare", "set"])
def test_a_spawned_child_runs_with_one_intra_op_thread_unless_told_otherwise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, parent: str | None
) -> None:
    """§3: a spawned rank sizes its OpenMP pool to one thread, as a
    ``torchrun`` rank does, unless the parent's environment names a count.
    This avoids oversubscribing the CPU. Two things are asserted: the
    variable the child was handed — the spawn's own
    contract — and the pool's size, which is torch's reading of it as the
    child imports torch: exactly one when the spawn set it, and what a
    fresh interpreter under the same variable reports when the parent
    named a count (`_fresh_torch_threads`; on a runner where torch
    caps the count, the child caps it the same way). The parent's own
    environment is as it was."""
    from causalab.neural.shared.parallel.spawn import THREADS_VARIABLE

    if parent is None:
        monkeypatch.delenv(THREADS_VARIABLE, raising=False)
    else:
        monkeypatch.setenv(THREADS_VARIABLE, parent)
    assert (
        spawn(ParallelGeometry(data=2), [str(tmp_path)], entry=_record_environment) == 0
    )
    expected = (
        1
        if parent is None
        else _fresh_torch_threads({**os.environ, THREADS_VARIABLE: parent})
    )
    for rank in (0, 1):
        env = json.loads((tmp_path / f"rank{rank}.json").read_text())
        assert env[THREADS_VARIABLE] == ("1" if parent is None else parent)
        assert env["torch_threads"] == expected
    assert os.environ.get(THREADS_VARIABLE) == parent
