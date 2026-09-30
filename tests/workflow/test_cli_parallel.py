"""``causalab run <workflow> --parallel <axes>`` (``docs/model_parallelism.md``
§3, §11; workflow spec §8, §9): a workflow document at a world above 1 goes
through the SPMD launcher exactly as an intervention specification does —
the parent spawns, a rank joins — and every rank then runs the workflow's
steps in lockstep with the same engine geometry, the joiner alone writing
the ROOT and the step records.

``unit``, beside ``tests/protocol/test_cli_parallel.py`` whose stand-ins
this file mirrors: the dispatch (a world above 1 spawns before any engine
is built; ``dp > 1`` is refused by name before any spawn — the runner is one
process with one engine list); world 1 unchanged to the byte with and
without the flag; joined rank 0 publishing with ``execution.parallel`` in
every step record and the same digests as world 1; a rank that does not
publish following the joiner's agreed decisions — running only the engine's
half of each protocol step, writing nothing under the ROOT, and stopping
with the joiner's refusal rendered to the character.

The stub engine writes one row per point into each save table (the
``test_fan_out.py`` stub, narrowed), so a step verifies and publishes
without a model.
"""

from __future__ import annotations

import json
import shutil
import sys
import types
from pathlib import Path
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from causalab.cli import main
from causalab.io.env import FileArtifacts, FileDatasets, ResolutionEnv
from causalab.io.step_record import SIDECAR
from causalab.io.tables import write_table
from causalab.neural.shared.parallel import launcher
from causalab.neural.shared.sweep import signed_steps
from causalab.protocol.compiled import CompiledProtocol
from causalab.protocol.engine import Engine, RunContext, RunResult, StepRecord
from causalab.protocol.lockstep import SOLO as SOLO_LOCKSTEP
from causalab.protocol.lockstep import Decided, Outcome, refusal_of
from causalab.protocol.lowering import point_count
from causalab.protocol.parallel import ONE, ParallelGeometry
from causalab.protocol.rules.errors import ProtocolError
from causalab.protocol.schema import COMPONENTS
from causalab.workflow.cli import check_geometry
from causalab.workflow.document import load_workflow
from causalab.workflow.runner import run_workflow

from tests.protocol._env import FIXTURES

pytestmark = pytest.mark.unit

HOOKS_MODULE = "causalab.neural.engines.pytorch_hooks"
NNSIGHT_MODULE = "causalab.neural.engines.nnsight_tracing"
FAN_OUT = Path(__file__).parent / "fixtures" / "fan_out"
STEPS = ("scan", "best")

#: The receipt's ``execution.parallel`` block for ``tp=2`` on a joined rank.
TP2_JOINED = {
    "data": 1,
    "data_mode": "points",
    "pipeline": 1,
    "context": 1,
    "tensor": 2,
    "expert": 1,
    "world": 2,
    "launcher": "joined",
}


class _Rows(Engine):
    """Stands in for the reference engine: records its constructor's
    geometry and every handoff (the compiled document and its run context),
    signs the steps the run selects, and writes one row per point into each
    save table when its publisher publishes — a rank that does not publish
    returns the signed steps and no files, as the reference engine does."""

    built: list["_Rows"] = []

    name = "pytorch_hooks"
    capabilities = frozenset(
        {"grad", "paired_forward", "full_logits", "pytorch_fn_local"}
    )
    components = frozenset(COMPONENTS)
    writable_components = frozenset(COMPONENTS)
    is_local = True

    def __init__(
        self,
        *,
        device: str = "cpu",
        batch_rows: int | None = None,
        cuda_graphs: bool = False,
        fit_rows: int | None = None,
        parallel: ParallelGeometry = ONE,
    ) -> None:
        self.device = device
        self.batch_rows = batch_rows
        self.fit_rows = fit_rows
        self.parallel = parallel
        self.runs: list[tuple[CompiledProtocol, RunContext]] = []
        #: the steps each run signed, in run order — what the step record's
        #: ``point_digests`` copy
        self.signed: list[tuple[StepRecord, ...]] = []
        type(self).built.append(self)

    def execute(self, compiled: CompiledProtocol, run: RunContext) -> RunResult:
        self.runs.append((compiled, run))
        steps = signed_steps(
            compiled, run.env, indices=run.indices(point_count(compiled.axes))
        )
        records = tuple(step.record for step in steps)
        self.signed.append(records)
        if not run.publisher.publish:
            return RunResult(files={}, steps=records)
        files: dict[str, Path] = {}
        for entry in steps[0].raw["method"]["save"]:
            if "read" not in entry:
                continue  # a derived record (`location_ledger`) is no table
            rows = [
                {
                    "example_id": "0",
                    # the table's label is the entry's file stem (§2.12)
                    "metric": Path(str(entry["file_path"])).stem,
                    "value": int(digest[:8], 16) / 16**8,
                    # a structured coordinate (`positions.tap`) as the JSON text
                    # a table column holds, as the real engine writes it
                    **{
                        axis: json.dumps(v, sort_keys=True)
                        if isinstance(v, (dict, list))
                        else v
                        for axis, v in coords.items()
                    },
                    "produced_by": digest,
                }
                for digest, coords in ((step.digest, step.coords) for step in steps)
            ]
            target = run.output_dir / str(entry["file_path"])
            write_table(target, rows)
            files[str(entry["file_path"])] = target
        return RunResult(files=files, steps=records)


class _Nnsight(Engine):
    name = "nnsight"
    capabilities = frozenset({"paired_forward", "full_logits", "pytorch_fn_local"})
    components = frozenset(COMPONENTS)
    writable_components = frozenset(COMPONENTS)
    is_local = True

    def __init__(self, *, device: str = "cpu") -> None:
        self.device = device

    def execute(self, compiled: CompiledProtocol, run: RunContext) -> RunResult:
        from tests._helpers.stub_engine import stub_execute

        return stub_execute(self, compiled, run)


@pytest.fixture
def engines(monkeypatch: pytest.MonkeyPatch) -> type[_Rows]:
    for module, attr, cls in (
        (HOOKS_MODULE, "PytorchHooksEngine", _Rows),
        (NNSIGHT_MODULE, "NnsightEngine", _Nnsight),
    ):
        stub = types.ModuleType(module)
        setattr(stub, attr, cls)
        monkeypatch.setitem(sys.modules, module, stub)
    _Rows.built = []
    return _Rows


@pytest.fixture
def no_group(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("WORLD_SIZE", "RANK", "LOCAL_RANK", launcher.LAUNCHER_VARIABLE):
        monkeypatch.delenv(name, raising=False)


class _Rank:
    """A launched rank's publisher with the process-group join stubbed at
    the system boundary: ``publish`` says whether it is the joiner."""

    launcher = "joined"
    replica = 0
    replicas = 1

    def __init__(self, publish: bool) -> None:
        self.publish = publish

    def gather(self, payload: Any) -> Any:
        return (payload,) if self.publish else None


class _Fed:
    """A follower's lockstep fed the joiner's outcomes in run order."""

    def __init__(self, *outcomes: Outcome) -> None:
        self.outcomes = list(outcomes)
        self.calls = 0

    def agree(self, outcome: Outcome | None) -> Outcome:
        assert outcome is None, "a follower passes None"
        self.calls += 1
        return self.outcomes.pop(0)


@pytest.fixture
def joined(monkeypatch: pytest.MonkeyPatch):
    """Preset the environment ``torchrun`` would for ``rank`` of ``world``
    and stub the launcher's group entry and lockstep."""

    def preset(world: int, rank: int, publisher: Any, lockstep: Any) -> None:
        for name, value in (
            ("WORLD_SIZE", str(world)),
            ("RANK", str(rank)),
            ("LOCAL_RANK", str(rank)),
            ("MASTER_ADDR", "127.0.0.1"),
            ("MASTER_PORT", "29500"),
        ):
            monkeypatch.setenv(name, value)
        monkeypatch.delenv(launcher.LAUNCHER_VARIABLE, raising=False)
        monkeypatch.setattr(
            launcher, "enter", lambda launch, geometry, device: publisher
        )
        monkeypatch.setattr(launcher, "lockstep", lambda publisher_: lockstep)
        monkeypatch.setattr(launcher, "leave", lambda publisher_, status=0: None)

    return preset


# --------------------------------------------------------------------------- #
# the fixture: scan -> best, unfanned, in a private copy
# --------------------------------------------------------------------------- #


def _workflow(tmp: Path) -> Path:
    root = tmp / "wf"
    shutil.copytree(FAN_OUT, root)
    raw = json.loads((root / "scan_wf.json").read_text())
    raw["steps"]["scan"].pop("fan_out")
    raw["output_dir"] = "run"
    document = root / "scan_wf.json"
    document.write_text(json.dumps(raw, indent=2) + "\n")
    return document


def _argv(document: Path, out: Path, *extra: str) -> list[str]:
    return [
        "run",
        str(document),
        "--data-root",
        str(FIXTURES / "data"),
        "--artifacts-root",
        str(document.parent),
        "--engine",
        "auto",
        "--out",
        str(out),
        *extra,
    ]


def _record(out: Path, step: str) -> dict[str, Any]:
    return json.loads((out / "run" / step / SIDECAR).read_text())


def _manifest(out: Path) -> dict[str, Any]:
    return json.loads((out / "run" / "workflow.json").read_text())


# --------------------------------------------------------------------------- #
# the dispatch: spawn before any engine; dp > 1 refused before any spawn
# --------------------------------------------------------------------------- #


def test_a_workflow_at_a_world_above_one_with_no_group_spawns_and_builds_no_engine(
    engines, no_group, tmp_path, monkeypatch
) -> None:
    """§3: the parent of a spawn re-launches ``main`` with the same argv —
    through the launcher's ``spawn`` seam — and exits with the children's
    status: no engine, no ROOT."""
    spawned: dict[str, Any] = {}

    def fake_spawn(geometry, argv, *, entry=None, device=None):
        spawned["geometry"] = geometry
        spawned["argv"] = list(argv)
        return 7

    monkeypatch.setattr(launcher, "spawn", fake_spawn)
    document = _workflow(tmp_path)
    argv = _argv(document, tmp_path / "out", "--parallel", "tp=2")
    assert main(argv) == 7
    assert spawned == {"geometry": ParallelGeometry(tensor=2), "argv": argv}
    assert engines.built == [], "the parent builds no engine"
    assert not (tmp_path / "out").exists()


def test_a_data_axis_above_one_is_refused_by_name_before_any_spawn(
    engines, no_group, tmp_path, monkeypatch, capsys
) -> None:
    """§11: the workflow runner is one process with one engine list, so the
    data axis is not served under a workflow — refused as ``P4`` naming
    ``--parallel.data`` and the way to shard a step, before any child starts
    and on every rank alike."""

    def never(*args, **kwargs):
        raise AssertionError("dp > 1 must be refused before the spawn")

    monkeypatch.setattr(launcher, "spawn", never)
    document = _workflow(tmp_path)
    code = main(_argv(document, tmp_path / "out", "--parallel", "dp=2,tp=2"))
    assert code == 1
    err = capsys.readouterr().err
    assert err.startswith("refused: [P4]")
    assert "--parallel.data" in err and "fan_out" in err and "workflow" in err
    assert engines.built == [] and not (tmp_path / "out").exists()


def test_check_geometry_refuses_only_the_data_axis() -> None:
    for geometry in (ONE, ParallelGeometry(pipeline=2, tensor=2, expert=2)):
        check_geometry(geometry)
    for text in ("dp=2", "dp=2:rows", "dp=3,tp=2"):
        from causalab.protocol.parallel import parse_geometry

        with pytest.raises(ProtocolError) as info:
            check_geometry(parse_geometry(text))
        assert info.value.code == "P4" and info.value.path == "--parallel.data"


def test_the_pure_verbs_and_dry_run_never_reach_the_launcher(
    engines, no_group, tmp_path, monkeypatch, capsys
) -> None:
    def never(*args, **kwargs):
        raise AssertionError("no launch for a verb that runs nothing")

    monkeypatch.setattr(launcher, "detect", never)
    monkeypatch.setattr(launcher, "spawn", never)
    document = _workflow(tmp_path)
    assert (
        main(
            [
                "dry-run",
                str(document),
                "--data-root",
                str(FIXTURES / "data"),
                "--engine",
                "auto",
                "--parallel",
                "tp=2",
            ]
        )
        == 1
    )
    assert "workflow-level dry run" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# world 1 is unchanged
# --------------------------------------------------------------------------- #


def test_world_one_is_byte_identical_with_and_without_the_flag(
    engines, no_group, tmp_path
) -> None:
    """A geometry of one is no geometry: every step record and the manifest
    are the same bytes, and no digest moves."""
    bare, flagged = _workflow(tmp_path / "a"), _workflow(tmp_path / "b")
    assert main(_argv(bare, tmp_path / "a" / "out")) == 0
    assert main(_argv(flagged, tmp_path / "b" / "out", "--parallel", "tp=1")) == 0
    for step in STEPS:
        a = (tmp_path / "a" / "out" / "run" / step / SIDECAR).read_bytes()
        b = (tmp_path / "b" / "out" / "run" / step / SIDECAR).read_bytes()
        assert a == b, step
    assert _manifest(tmp_path / "a" / "out") == _manifest(tmp_path / "b" / "out")
    assert (
        _record(tmp_path / "b" / "out", "scan")["execution"]["parallel"]["launcher"]
        == "solo"
    )
    assert len(engines.built) == 2 and all(e.parallel == ONE for e in engines.built)


# --------------------------------------------------------------------------- #
# joined rank 0 publishes; a rank that does not publish follows
# --------------------------------------------------------------------------- #


def test_joined_rank_zero_runs_every_step_and_publishes_with_the_geometry(
    engines, joined, tmp_path, capsys
) -> None:
    """The joiner of a ``tp=2`` world: the same ROOT a world-1 run writes —
    identities and digests untouched — with ``execution.parallel`` in each
    protocol step's record saying ``joined`` at world 2, and the engine's
    run context carrying the rank's publisher."""
    solo = _workflow(tmp_path / "solo")
    assert main(_argv(solo, tmp_path / "solo" / "out")) == 0
    document = _workflow(tmp_path / "joined")
    joined(2, 0, _Rank(publish=True), SOLO_LOCKSTEP)
    out = tmp_path / "joined" / "out"
    assert main(_argv(document, out, "--parallel", "tp=2")) == 0
    a, b = _record(tmp_path / "solo" / "out", "scan"), _record(out, "scan")
    assert b["execution"]["parallel"] == TP2_JOINED
    for key in ("identity", "document_digest", "point_digests", "files", "engine"):
        assert a[key] == b[key], key
    assert {k for k in a if a[k] != b[k]} == {"execution"}
    assert _record(out, "best")["status"] == "completed"
    assert _manifest(out)["steps"].keys() == {"scan", "best"}
    hooks = engines.built[-1]
    assert hooks.parallel == ParallelGeometry(tensor=2)
    ((_, run),) = hooks.runs
    assert run.publisher.launcher == "joined" and run.publisher.publish
    out_text = capsys.readouterr().out
    assert "completed scan" in out_text


def test_a_rank_that_does_not_publish_follows_and_writes_nothing(
    engines, joined, tmp_path, capsys
) -> None:
    """Rank 1 of ``tp=2``: it runs the engine's half of the protocol step —
    the same handoff, its publisher saying it does not publish — and takes
    every other decision from the joiner: no ROOT, no attempt directory, no
    script run, nothing printed."""
    solo = _workflow(tmp_path / "solo")
    assert main(_argv(solo, tmp_path / "solo" / "out")) == 0
    scan, best = (_record(tmp_path / "solo" / "out", s) for s in STEPS)
    fed = _Fed(
        Decided("turn", "scan", {"reused": None}),
        Decided("attempt", "scan", scan),
        Decided("turn", "best", {"reused": None}),
        Decided("attempt", "best", best),
        Decided("manifest", None, _manifest(tmp_path / "solo" / "out")),
    )
    document = _workflow(tmp_path / "follower")
    joined(2, 1, _Rank(publish=False), fed)
    out = tmp_path / "follower" / "out"
    capsys.readouterr()  # the world-1 run's lines
    assert main(_argv(document, out, "--parallel", "tp=2")) == 0
    assert fed.outcomes == [] and fed.calls == 5
    assert not out.exists(), "a rank that does not publish writes nothing"
    hooks = engines.built[-1]
    ((_, run),) = hooks.runs
    assert not run.publisher.publish
    (signed,) = hooks.signed
    assert [step.digest for step in signed] == scan["point_digests"]
    assert run.output_dir == out / "run" / "scan"
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""


def test_the_followers_lockstep_refusal_is_a_refusal_on_this_rank_too(
    engines, joined, tmp_path, capsys
) -> None:
    """Refusals stay refusals on every rank: the joiner's ``P4`` at the
    scan's attempt reaches rank 1 as the same ``refused: [P4] …`` line and
    exit 1, after rank 1 ran its half of the step."""
    err = ProtocolError(
        "P4", "the fit straddles two stages", path="--parallel.pipeline"
    )
    fed = _Fed(
        Decided("turn", "scan", {"reused": None}),
        refusal_of("attempt", "scan", err),
        refusal_of("manifest", None, err),
    )
    document = _workflow(tmp_path)
    joined(2, 1, _Rank(publish=False), fed)
    out = tmp_path / "out"
    assert main(_argv(document, out, "--parallel", "tp=2")) == 1
    assert fed.outcomes == []
    assert capsys.readouterr().err == f"refused: {err}\n"
    assert not out.exists()
    assert len(engines.built[-1].runs) == 1


def test_a_follower_takes_the_joiners_reuse_and_runs_no_engine(
    engines, joined, tmp_path
) -> None:
    """``--resume`` is decided by the joiner, who reads the tree: a follower
    told the scan is reused runs no engine for it."""
    solo = _workflow(tmp_path / "solo")
    assert main(_argv(solo, tmp_path / "solo" / "out")) == 0
    scan, best = (_record(tmp_path / "solo" / "out", s) for s in STEPS)
    fed = _Fed(
        Decided("turn", "scan", {"reused": {**scan, "status": "reused"}}),
        Decided("turn", "best", {"reused": {**best, "status": "reused"}}),
        Decided("manifest", None, _manifest(tmp_path / "solo" / "out")),
    )
    document = _workflow(tmp_path / "follower")
    joined(2, 1, _Rank(publish=False), fed)
    out = tmp_path / "follower" / "out"
    assert main(_argv(document, out, "--parallel", "tp=2", "--resume")) == 0
    assert fed.outcomes == []
    assert engines.built[-1].runs == []
    assert not out.exists()


# --------------------------------------------------------------------------- #
# the lockstep is per step, never per point
# --------------------------------------------------------------------------- #


class _Counting:
    """The joiner's lockstep at world 1 — the identity — tallying every
    agreement by moment and step."""

    def __init__(self) -> None:
        self.moments: list[tuple[str, str | None]] = []

    def agree(self, outcome: Outcome | None) -> Outcome:
        assert outcome is not None, "the joiner passes its outcome"
        self.moments.append((outcome.moment, outcome.step))
        return outcome


def _env(root: Path) -> ResolutionEnv:
    return ResolutionEnv(
        datasets=FileDatasets(root=root, fallback_roots=(FIXTURES / "data",)),
        artifacts=FileArtifacts(root=root),
    )


@settings(
    max_examples=30,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(layers=st.integers(min_value=1, max_value=12))
def test_the_lockstep_agrees_per_step_never_per_point(
    tmp_path_factory: pytest.TempPathFactory, layers: int
) -> None:
    """Agreement counts depend on steps, not expanded points: one for each
    step's turn and record, then one for the manifest. A per-point
    agreement would make the count grow with ``layers``."""
    tmp = tmp_path_factory.mktemp("per-step")
    document = _workflow(tmp)
    scan = document.parent / "protocols" / "scan.json"
    raw = json.loads(scan.read_text())
    raw["method"]["sites"]["target"]["layers"] = {"sweep": list(range(layers))}
    scan.write_text(json.dumps(raw))
    env = _env(document.parent)
    engine, lockstep = _Rows(), _Counting()
    run_workflow(
        load_workflow(document, env),
        env,
        tmp / "out",
        engine,
        publisher=_Rank(publish=True),
        lockstep=lockstep,
    )
    (signed,) = engine.signed
    assert len(signed) == 2 * layers, "two positions per layer"
    assert lockstep.moments == [
        ("turn", "scan"),
        ("attempt", "scan"),
        ("turn", "best"),
        ("attempt", "best"),
        ("manifest", None),
    ]


def test_a_protocol_step_meets_check_parallel_before_the_engine_runs(
    tmp_path: Path,
) -> None:
    """``pipeline.check_parallel`` runs at the protocol step's door as at
    ``run_protocol``'s, on every rank before any weights, so the ranks refuse
    alike and none waits on a collective. The scan declares no ``train``, so
    ``dp=2:rows`` has no minibatch to split (``docs/model_parallelism.md``
    §8.3). The CLI refuses ``dp > 1`` for a workflow before this door
    (``check_geometry``); the runner is checked on its own here."""
    document = _workflow(tmp_path)
    env = _env(document.parent)
    engine = _Rows(parallel=ParallelGeometry(data=2, data_mode="rows"))
    with pytest.raises(ProtocolError) as err:
        run_workflow(load_workflow(document, env), env, tmp_path / "out", engine)
    assert err.value.code == "P4" and "declares no train" in str(err.value)
    assert engine.runs == [], "the engine ran before the refusal"
