"""Declared fan-out and joins (workflow spec §2.9, §5 rule 19;
T15–T18 and the censuses) — on CPU, with a stub engine.

* **The censuses.** `MAX_RULE` is 20 and §5 numbers it; the two closed
  vocabularies — what a fan-out is declared `over`, what a join may `require`
  — are §2.9's tables member for member; every scope executes (`SCOPES ==
  EXECUTABLE_SCOPES`); the child separator is outside rule 3's alphabet.
* **The closure guard.** ``causalab/workflow/fan_out.py`` is a member of no
  hashed script's closure and of ``SHARED``, reaches no engine module, and
  loading a fanned-out workflow imports no torch.
* **T15 — the join.** A two-shard fan-out and an axis fan-out of a
  swept protocol step publish tables row for row equal to the unsharded
  run's, under one normalized receipt (the parent's ``_step.json``: the full
  ``axes`` and ``point_digests``, a ``fan_out`` and a ``join`` block, no
  ``engine`` and no ``execution``); the shipped ``select`` reads the joined
  table as it reads the unsharded one; the stream has no new payload key. A
  row of another child's point injected after that child published is a
  **duplicate** refusal; a row the join cannot place (no point named,
  or an index outside its child's points) is refused naming the file and the
  row, never appended out of order; a digest one child declares twice is a
  duplicate naming the child; a digest no child declared is a foreign point;
  a side table keyed by a digest-valued ``point`` joins in point order. A
  fanned-out **behavioral** step's joined ``continuations.json`` and
  ``outcomes.json`` are byte-identical to the unsharded run's — a
  ``continuations.json`` row's int ``point`` is re-based beside the
  ``point_digest`` that places it — and a row whose ``point`` index and
  ``point_digest`` disagree is refused; a metric row's columns are open and it
  is placed by its coordinate columns first, so an int column named ``point``
  on a metric row is a coordinate the join never touches.
* **T16 — missing vs selected.** A row or a digest deleted from a child
  before the join is a **missing** refusal with a distinct message — the join
  ``failed``, its dependents ``blocked``, the attempt retained; a
  ``require: selected`` join under a ``per_target`` conditional publishes the
  children the verdicts left and names the skipped child by
  ``evidence_identity``; a ``global`` conditional over a fanned-out step skips
  the parent and every child. Only a coordinate-placed metric row and a
  ``point_digest`` row promise a row per point: a side table placed by a
  digest-valued ``point``
  that one child published sparsely and another not at all joins without a
  missing refusal. The sparse save files are **declared**
  (``_SPARSE_FILES``, censused against ``outputs.py``'s three side tables and
  the writers' row shapes), never inferred from the rows found, so a dense
  table one child published empty beside a child that omitted it is the
  missing-file refusal; a fanned-out step whose
  document claims one of those names (by the path's final component) for its
  own save file is refused at load under rule 19, the same document loads
  unfanned and under any other name fans out and joins, the refusal's name
  set is ``_SPARSE_FILES`` itself (each name is spelled once in
  ``fan_out.py``), and §2.9's three names are censused against the code.
* **T17 — T13's third run + T14.** No shipped or demo entry gains a key; both
  pins hold from the pin file; identities are unchanged for every existing
  kind; a child is identified by its parent's entry and its shard, so a
  changed width re-runs every child under ``--resume``.
* **T18 — declared, not run-time.** Every refusal of the grammar beside a
  loading, running twin.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, cast

import pytest

from causalab.cli import main as cli_main
from causalab.io.events import EVENTS_FILE, read_events, terminal
from causalab.io.step_record import SIDECAR
from causalab.neural.shared.sweep import signed_steps
from causalab.protocol.identity import import_closure
from causalab.protocol.compiled import CompiledProtocol
from causalab.protocol.engine import (
    CONTINUATIONS_FILE,
    Engine,
    RunContext,
    RunResult,
    StepRecord,
)
from causalab.protocol.estimand import IDENTITY_COLUMNS
from causalab.protocol.lowering import point_count
from causalab.io.env import FileArtifacts, FileDatasets, ResolutionEnv
from causalab.protocol.schema import COMPONENTS
from causalab.io.tables import write_table
from causalab.tasks import TASKS_ROOT
from causalab.workflow import fan_out
from causalab.workflow import manifest as mf
from causalab.workflow import runner
from causalab.workflow.behavioral import DECISION_FILE, OUTCOMES_FILE
from causalab.workflow.conditional import EXECUTABLE_SCOPES, SCOPES
from causalab.workflow.document import (
    MAX_RULE,
    STEP_TYPES,
    BehavioralStep,
    ConditionalStep,
    LoadedWorkflow,
    ProtocolStep,
    WorkflowError,
    load_workflow,
)
from causalab.workflow.fan_out import (
    CHILD_SEPARATOR,
    FAN_OUT_RULE,
    JOIN_POLICIES,
    OVER_KEYS,
)
from causalab.workflow.runner import run_workflow
from tests.workflow.test_closure_census import (
    CLOSURES,
    DEMO_WORKFLOWS,
    REDUCE,
    SHARED,
    SHIPPED,
    WORKFLOWS,
    _closure,  # pyright: ignore[reportPrivateUsage]
    _demo_env,  # pyright: ignore[reportPrivateUsage]
)
from tests.workflow.test_conditional import (
    CHAIN,
    _identity_kinds,  # pyright: ignore[reportPrivateUsage]
    _load as _chain_load,  # pyright: ignore[reportPrivateUsage]
    _members,  # pyright: ignore[reportPrivateUsage]
    _run as _chain_run,  # pyright: ignore[reportPrivateUsage]
    _sentinels,  # pyright: ignore[reportPrivateUsage]
    _snapshot,  # pyright: ignore[reportPrivateUsage]
    _statuses,  # pyright: ignore[reportPrivateUsage]
    _tree as _chain_tree,  # pyright: ignore[reportPrivateUsage]
)
from tests.workflow.test_controls import _section  # pyright: ignore[reportPrivateUsage]

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).parent / "fixtures" / "fan_out"
BEHAVIORAL_FIXTURES = Path(__file__).parent / "fixtures" / "behavioral"
CONDITIONAL_FIXTURES = Path(__file__).parent / "fixtures" / "conditional"
PROTOCOL_DATA = REPO / "tests" / "protocol" / "fixtures" / "data"
MODULE = "causalab/workflow/fan_out.py"
SECTION_29 = "### 2.9 `fan_out` — a declared fan-out and its join"
#: every key a fan-out puts on a parent's entry, or a child's record carries,
#: and no existing entry may (§7)
FAN_OUT_KEYS = ("fan_out", "shard", "join")
LAYERS = "sites.target.layers"
#: the scan fixture's axes, in expansion order
SCAN_AXES = ("positions.tap", LAYERS)
PROBE_LAYERS = "sites.probe.layers"
SAID = "positions.said_answer"
CHECKER = {
    "task": "natural_domains_arithmetic",
    "task_cfg": {"domain_type": "weekdays"},
}


# --------------------------------------------------------------------------- #
# the fixture tree and the stub engine
# --------------------------------------------------------------------------- #


def _tree(tmp: Path) -> Path:
    """A private copy of the fan-out fixtures, plus the behavioral fixtures'
    table (a real per-child decision wants it) and the conditional fixtures'
    two scripts (a global conditional over a fanned-out step wants them)."""
    root = tmp / "fan_out"
    shutil.copytree(FIXTURES, root)
    shutil.copytree(BEHAVIORAL_FIXTURES / "qa", root / "qa")
    shutil.copytree(CONDITIONAL_FIXTURES / "scripts", root / "scripts")
    (root / "measurement.json").write_text(json.dumps({"score": 0.9}))
    return root


def _env(root: Path) -> ResolutionEnv:
    return ResolutionEnv(
        datasets=FileDatasets(root=root, fallback_roots=(PROTOCOL_DATA, TASKS_ROOT)),
        artifacts=FileArtifacts(root=root),
    )


def _raw(root: Path) -> dict[str, Any]:
    return json.loads((root / "scan_wf.json").read_text())


def _load(root: Path, raw: dict[str, Any] | None = None) -> LoadedWorkflow:
    return load_workflow(
        raw if raw is not None else root / "scan_wf.json", _env(root), workflow_dir=root
    )


def _run(loaded: LoadedWorkflow, root: Path, out: Path, **kw: Any) -> Any:
    return run_workflow(loaded, _env(root), out, _Rows(), **kw)


def _refused(
    root: Path, raw: dict[str, Any], *, rule: int = FAN_OUT_RULE, path: str
) -> WorkflowError:
    with pytest.raises(WorkflowError) as err:
        _load(root, raw)
    assert err.value.rule == rule, str(err.value)
    assert path in str(err.value), str(err.value)
    return err.value


def _fanned(
    raw: dict[str, Any], step: str, over: dict[str, Any], require: str = "all"
) -> dict[str, Any]:
    raw["steps"][step]["fan_out"] = {"over": over, "join": {"require": require}}
    return raw


def _unfanned(raw: dict[str, Any], step: str = "scan") -> dict[str, Any]:
    raw["steps"][step].pop("fan_out", None)
    return raw


def _value(digest: str, example: int) -> float:
    return int(digest[:8], 16) / 16**8 + example / 100


def _plain(value: Any) -> Any:
    return (
        json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
    )


def _axes(loaded: LoadedWorkflow, step: str) -> list[str]:
    return [axis.id for axis in loaded.inner[step].expansion.axes]


def _parent_index(loaded: LoadedWorkflow, step: str, row: Mapping[str, Any]) -> int:
    """The parent's point index a metric row's coordinate columns name — the
    join's placement rule, restated over the expansion for the tests."""
    axes = _axes(loaded, step)
    key = tuple(_plain(row[axis]) for axis in axes)
    for index, point in enumerate(loaded.inner[step].expansion.points):
        if tuple(_plain(point.coords[axis]) for axis in axes) == key:
            return index
    raise AssertionError(f"no point of {step!r} at {key}")


class _Rows(Engine):
    """A stub engine serving the fixture documents: one metric row per
    requested point and example into each of the document's save tables —
    the coordinates as columns, no digest on the row — and, for a
    decoding request, a `continuations.json` whose text says the answer at
    layer 0 of the swept `probe` site and says nothing at layer 1, so the
    children of a `sites.probe.layers` fan-out decide pass and fail."""

    def __init__(self) -> None:
        self.name = "pytorch_hooks"
        self.capabilities = frozenset(
            {"grad", "paired_forward", "full_logits", "pytorch_fn_local", "generate"}
        )
        self.components = frozenset(COMPONENTS)
        self.writable_components = frozenset(COMPONENTS)
        self.is_local = True
        self.runs: list[tuple[CompiledProtocol, RunContext]] = []

    def execute(self, compiled: CompiledProtocol, run: RunContext) -> RunResult:
        self.runs.append((compiled, run))
        files: dict[str, Path] = {}
        # the engine's sweep: the steps `run` selects, enumerated and signed
        steps = signed_steps(
            compiled, run.env, indices=run.indices(point_count(compiled.axes))
        )
        points = [step.raw for step in steps]
        digests = [step.digest for step in steps]
        coords_by_point = [step.coords for step in steps]
        saves = points[0]["method"]["save"]
        n_examples = 2
        rows_by_file: dict[str, list[dict[str, Any]]] = {
            str(entry["file_path"]): [] for entry in saves
        }
        for point, (raw, digest, coords) in enumerate(
            zip(points, digests, coords_by_point)
        ):
            for entry in saves:
                for example in range(n_examples):
                    rows_by_file[str(entry["file_path"])].append(
                        {
                            "example_id": str(example),
                            "metric": str(entry["file_path"])
                            .rsplit("/", 1)[-1]
                            .rsplit(".", 1)[0],
                            "value": _value(digest, example),
                            **{axis: _plain(v) for axis, v in coords.items()},
                        }
                    )
        for rel, rows in rows_by_file.items():
            target = run.output_dir / rel
            write_table(target, rows)
            files[rel] = target
        if run.decoding is not None:
            ref = points[0]["data"]["base"]["dataset"]
            table = run.env.datasets.rows(ref)
            lines: list[dict[str, Any]] = []
            for point, (digest, coords) in enumerate(zip(digests, coords_by_point)):
                says = coords.get(PROBE_LAYERS, 0) == 0
                for example, row in enumerate(table):
                    lines.append(
                        {
                            "point": point,
                            "point_digest": digest,
                            "model": "original",
                            "input": "base",
                            "example_id": str(example),
                            "steps": 4,
                            "width": 2,
                            "truncated": False,
                            "token_ids": [1, 2],
                            "text": f" {row['result']}" if says else " nowhere",
                            "offsets": [],
                        }
                    )
            target = run.output_dir / "continuations.json"
            write_table(target, lines)
            files["continuations.json"] = target
        return RunResult(files=files, steps=tuple(step.record for step in steps))


def _behavioral_raw(
    root: Path,
    *,
    scope: str = "per_target",
    over: dict[str, Any] | None = None,
    min_correct_rate: float = 1.0,
    require: str = "selected",
) -> dict[str, Any]:
    """The per-child chain: a fanned-out behavioral producer, a per-child
    conditional over it, and two fanned-out gated protocol steps."""
    over = over if over is not None else {"axis": PROBE_LAYERS}
    return {
        "version": "1",
        "output_dir": "per_child",
        "steps": {
            "qualify": {
                "type": "behavioral",
                "document": "protocols/probe.json",
                "decoding": {"mode": "deterministic"},
                "checker": dict(CHECKER),
                "split": "development",
                "thresholds": {
                    "min_examples": 1,
                    "min_valid_rate": 0.0,
                    "min_correct_rate": min_correct_rate,
                },
                "decision": {"on_pass": "advance", "on_fail": "narrow"},
                "fan_out": {"over": over, "join": {"require": "all"}},
            },
            "gate": {
                "type": "conditional",
                "predicate": {
                    "decision": {"step": "qualify"},
                    "field": "outcome",
                    "eq": "pass",
                },
                "on_true": ["apply"],
                "on_false": ["narrow"],
                "scope": scope,
            },
            "apply": {
                "type": "intervention_protocol",
                "document": "protocols/apply.json",
                "fan_out": {"over": over, "join": {"require": require}},
            },
            "narrow": {
                "type": "intervention_protocol",
                "document": "protocols/apply.json",
                "set": {"positions.tap": {"sweep": [{"index": -1}, {"index": -3}]}},
                "fan_out": {"over": over, "join": {"require": require}},
            },
        },
    }


def _gated_raw(root: Path, score: float) -> dict[str, Any]:
    """A global conditional (the conditional chain's decision step over a script's
    measurement) gating the fanned-out scan on its true side."""
    (root / "measurement.json").write_text(json.dumps({"score": score}))
    raw = _raw(root)
    raw["steps"]["measure"] = {
        "type": "script",
        "script": {"path": "scripts/measure.py"},
        "inputs": {"measurement": {"path": str(root / "measurement.json")}},
        "outputs": {
            "values": {"file": "values.json", "keys": {"score": 0.5, "k": 8}},
            "table": {"file": "scores.json", "columns": {"score": "float64"}},
        },
    }
    raw["steps"]["gate_k"] = {
        "type": "decision",
        "values": {"step": "measure", "file": "values.json"},
        "rule": {"score": {"ge": 0.5}},
        "decision": {"on_pass": "advance", "on_fail": "narrow"},
    }
    raw["steps"]["gate"] = {
        "type": "conditional",
        "predicate": {"decision": {"step": "gate_k"}, "field": "outcome", "eq": "pass"},
        "on_true": ["scan"],
        "on_false": ["probe"],
        "scope": "global",
    }
    raw["steps"]["probe"] = {
        "type": "script",
        "script": {"path": "scripts/writer.py"},
        "inputs": {},
        "outputs": {"out": {"file": "probe.json", "keys": {"ran": True}}},
    }
    return raw


def _record(run_root: Path, step: str) -> dict[str, Any]:
    return json.loads((run_root / step / SIDECAR).read_text())


def _payload_keys(run_root: Path, event: str) -> set[str]:
    return {
        key
        for line in read_events(run_root / EVENTS_FILE)
        if line["event"] == event
        for key in line["payload"]
    }


# --------------------------------------------------------------------------- #
# the censuses
# --------------------------------------------------------------------------- #


def test_rule_19_is_the_fan_out_rule_and_the_last() -> None:
    # §2.10: rule 20, the nested workflow's, follows; the ceiling is
    # pinned exactly by the last rule's own suite (test_nested.py), so a later
    # rule raises it without editing this one; the sixth kind is its; a fan-out
    # is still a field on the two document kinds
    assert FAN_OUT_RULE == 19 and MAX_RULE >= 19
    assert STEP_TYPES == (
        "intervention_protocol",
        "script",
        "behavioral",
        "decision",
        "conditional",
        "workflow",
    )
    section = _section("## 5. Validation")
    items = {
        int(number): text
        for number, text in re.findall(
            r"^(\d+)\. (.+?)(?=^\d+\. |\Z)", section, re.M | re.S
        )
    }
    assert max(items) == MAX_RULE
    for word in (
        "`fan_out`",
        "over",
        "shards",
        "join",
        "`selected`",
        "`per_target`",
        "`per_variable`",
        "**missing**",
        "**duplicate**",
        "names the field",
        "§2.9",
    ):
        assert word in items[19], word


def test_the_over_and_require_tables_are_the_code() -> None:
    assert _members(SECTION_29, "over") == list(OVER_KEYS) == ["axis", "shards"]
    assert _members(SECTION_29, "require") == list(JOIN_POLICIES) == ["all", "selected"]


def test_every_scope_executes() -> None:
    assert EXECUTABLE_SCOPES == SCOPES
    assert set(fan_out.SCOPE_ROOTS) == set(SCOPES) - {"global"}
    assert fan_out.SCOPE_ROOTS == {"per_target": "sites", "per_variable": "positions"}


def test_the_child_separator_is_outside_the_authored_alphabet(tmp_path: Path) -> None:
    """Rule 3 refuses an authored `@`, so a child can never collide with an
    authored step and needs no collision check (§1.1)."""
    assert CHILD_SEPARATOR == "@" == mf.CHILD_SEPARATOR
    assert fan_out.child_name("fit", 3) == "fit@3"
    assert fan_out.parent_of("fit@3") == "fit" and fan_out.parent_of("fit") is None
    root = _tree(tmp_path)
    raw = _raw(root)
    raw["steps"]["scan@0"] = raw["steps"].pop("best")
    with pytest.raises(WorkflowError) as err:
        _load(root, raw)
    assert err.value.rule == 3, str(err.value)


# --------------------------------------------------------------------------- #
# the closure guard, and torch-free
# --------------------------------------------------------------------------- #


def test_fan_out_is_in_no_hashed_closure() -> None:
    assert MODULE not in SHARED
    for module in (*CLOSURES, REDUCE):
        assert MODULE not in _closure(module), module


#: The torch-free enumerator — the one module under ``neural/`` the
#: workflow layer's engine-free modules reach.
ENUMERATOR = "causalab/neural/shared/sweep.py"


def test_fan_out_reaches_no_engine_module() -> None:
    """Engine-free: the one member under ``neural/`` is the torch-free
    enumerator the workflow layer reads the steps through
    (``neural/shared/sweep.py``, pinned torch-free in a subprocess by
    ``tests/neural/shared/test_sweep.py``) — never an engine, never
    numerics."""
    members = import_closure(REPO / MODULE, root=REPO)
    assert members, "the closure walk found nothing"
    assert [m for m in members if m.startswith("causalab/neural/")] == [ENUMERATOR]


_PROBE = """
import json, sys
from pathlib import Path
import causalab.workflow.fan_out
from causalab.io.env import FileArtifacts, FileDatasets, ResolutionEnv
from causalab.tasks import TASKS_ROOT
from causalab.workflow.document import load_workflow
root = Path(sys.argv[1]); data = Path(sys.argv[2])
env = ResolutionEnv(datasets=FileDatasets(root=data, fallback_roots=(TASKS_ROOT,)), artifacts=FileArtifacts(root=root))
loaded = load_workflow(root / "scan_wf.json", env)
print(json.dumps({"digest": loaded.digest, "order": list(loaded.order), "torch": "torch" in sys.modules}))
"""


def test_loading_a_fanned_out_workflow_imports_no_torch() -> None:
    completed = subprocess.run(
        [sys.executable, "-c", _PROBE, str(FIXTURES), str(PROTOCOL_DATA)],
        capture_output=True,
        text=True,
        cwd=str(REPO),
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    assert result["torch"] is False
    assert result["order"] == ["scan@0", "scan@1", "scan", "best"]
    loaded = load_workflow(
        FIXTURES / "scan_wf.json",
        ResolutionEnv(
            datasets=FileDatasets(root=PROTOCOL_DATA, fallback_roots=(TASKS_ROOT,)),
            artifacts=FileArtifacts(root=FIXTURES),
        ),
    )
    assert result["digest"] == loaded.digest


def test_the_sparse_save_files_and_the_row_shapes_are_the_writers(
    tmp_path: Path,
) -> None:
    """The sparse-file census. ``fan_out.py`` cannot
    import ``outputs.py`` (a module-level torch import; the closure guard
    above is load-bearing), so it restates the sparse save files — and this
    test, which may import freely, holds the restatement to the writers.

    (a) ``_SPARSE_FILES`` is exactly the three side tables ``write_outputs``
    writes only when there is something to write; neither dense behavioral
    file is in it. (b) The row shapes the join places by, against the real
    writers on CPU: a metric row (``MetricTable``) carries no digest key at
    all — neither ``point_digest`` nor a fixed ``point`` column — its open
    columns are the coordinates alone (what places it), so a coordinate named
    ``point`` lands as a plain ``point`` column; a
    ``continuations.json`` row (the engine's ``_write_continuations``) carries
    an int ``point`` beside its ``point_digest``; a ``train_eval.json`` record
    (``TrainEvalScore.as_record``) carries a ``str`` ``point`` and neither
    dense key, and the ``fit_diagnostics.json`` / ``routing_mismatch.json``
    rows — literals inside ``execute_document``'s run loop, unreachable
    without a model — are pinned by source to the same shape; an
    ``outcomes.json`` row (``behavioral._outcome_rows``, run here over the CPU
    stub's continuations) carries an int ``point`` and neither dense key. A
    fourth engine table, or a writer moving a key, fails here."""
    import torch

    from causalab.neural.engines.pytorch_hooks import engine as hooks_engine
    from causalab.io import results_io
    from causalab.neural.shared import results as outputs
    from causalab.neural.shared.encoding import Continuation
    from causalab.neural.shared.execution import TrainEvalScore

    sparse = fan_out._SPARSE_FILES  # pyright: ignore[reportPrivateUsage]
    dense = fan_out._DENSE_KEYS  # pyright: ignore[reportPrivateUsage]
    # (a) the declaration is the writers' three side tables, and nothing else
    assert set(sparse) == {
        results_io.TRAIN_EVAL_FILE,
        results_io.FIT_DIAGNOSTICS_FILE,
        results_io.ROUTING_MISMATCH_FILE,
    }
    assert len(sparse) == 3
    assert {CONTINUATIONS_FILE, OUTCOMES_FILE}.isdisjoint(sparse)
    assert set(dense) == {"point_digest"}
    digest = "ab" * 32
    identity = {column: "x" for column in IDENTITY_COLUMNS}
    root = _tree(tmp_path)
    # (b) a metric row: the coordinates as columns, no digest key, no fixed
    # `point`
    table = outputs.MetricTable()
    table.add("iia", [0.5], {"sites.target.layers": 3}, identity=identity)
    (metric,) = table.rows
    assert metric["sites.target.layers"] == 3
    assert {"point_digest", "point"}.isdisjoint(metric)
    fixed = outputs.MetricTable()
    fixed.add("iia", [0.5], {}, identity=identity)
    assert set(fixed.rows[0]) == {
        "example_id",
        "metric",
        "value",
        *IDENTITY_COLUMNS,
        outputs.ELIGIBLE_COLUMN,
    }
    # ... and its open columns are the coordinates: a coordinate named `point`
    # is a plain `point` column, which the join must not touch
    coordinate = outputs.MetricTable()
    coordinate.add("iia", [0.5], {"point": 3}, identity=identity)
    assert coordinate.rows[0]["point"] == 3
    # the writer's coordinate serialization is the join's (`fan_out._plain`)
    for value in (3, 0.5, "x", True, [1, 2], {"b": 1, "a": 2}):
        assert fan_out._plain(value) == outputs._plain(value)  # pyright: ignore[reportPrivateUsage]

    # a continuations row: int `point` (the index into the request) beside
    # `point_digest`, written by the real engine writer over one decoded batch
    class _Decoded:
        role_rows = {"base": [{"example_id": "row-0", "split": "development"}]}

        def input_token_ids(self, input_role: str) -> list[list[int]]:
            assert input_role == "base"
            return [[3]]

        def eos_token_ids(self) -> tuple[int, ...]:
            return (0,)

        def continuations(self) -> dict[tuple[str, str], Continuation]:
            return {
                ("original", "base"): Continuation(
                    token_ids=torch.tensor([[1, 2, 0]]),
                    widths=(2,),
                    texts=(" x",),
                    offsets=(((0, 1), (1, 2)),),
                )
            }

    run = RunContext(output_dir=tmp_path / "engine", env=_env(root))
    path = hooks_engine._write_continuations(  # pyright: ignore[reportPrivateUsage]
        run, cast("list[Any]", [_Decoded()]), steps=(StepRecord(0, {}, digest),)
    )
    (continuation,) = json.loads(path.read_text())
    assert continuation["point"] == 0 and continuation["point_digest"] == digest
    # the side tables: a `str` `point`, neither dense key
    evaluation = TrainEvalScore(
        split="development", metrics={"acc": 1.0}, passes=1
    ).as_record(point=digest, coords={"sites.target.layers": 3})
    assert evaluation["point"] == digest
    assert "point_digest" not in evaluation
    source = (REPO / "causalab" / "neural" / "shared" / "execution.py").read_text()
    for writer in ("train_evals", "fit_diagnostics", "routing_mismatch"):
        assert source.count(f"{writer}.append(") == 1, writer
    assert re.search(
        r"train_evals\.append\(\s*outcome\.eval_score\.as_record\(point=point_digest,",
        source,
    )
    literals = dict(
        re.findall(
            r"(fit_diagnostics|routing_mismatch)\.append\(\s*\{(.*?)\}\s*\)",
            source,
            flags=re.DOTALL,
        )
    )
    assert sorted(literals) == ["fit_diagnostics", "routing_mismatch"]
    for body in literals.values():
        assert '"point": point_digest,' in body
        assert '"point_digest"' not in body
    # an outcomes row: int `point`, neither dense key — the real behavioral
    # writer over the stub's continuations
    plain = _run(_load(root, _lone_behavioral_raw(root)), root, tmp_path / "plain")
    outcomes = json.loads((plain.run_root / "qualify" / OUTCOMES_FILE).read_text())
    assert outcomes
    for row in outcomes:
        assert isinstance(row["point"], int)
        assert "point_digest" not in row
    # (c) the load-time refusal's name set IS `_SPARSE_FILES`:
    # `check_fan_out` reads the tuple and
    # restates no name — each is spelled exactly once in `fan_out.py`, in
    # the tuple, so a fourth side table added there is refused at load too
    module_source = (REPO / MODULE).read_text()
    check = inspect.getsource(fan_out.check_fan_out)
    assert "_SPARSE_FILES" in check
    for name in sparse:
        assert module_source.count(f'"{name}"') == 1, name
        assert name not in check, name


def test_section_2_9_names_exactly_the_sparse_side_tables_the_code_declares() -> None:
    """The spec names every sparse side table and permits missing child files."""
    sparse = fan_out._SPARSE_FILES  # pyright: ignore[reportPrivateUsage]
    paragraph = next(
        p for p in _section(SECTION_29).split("\n\n") if "`train_eval.json`" in p
    )
    named = re.findall(r"[\w./-]+\.json", paragraph)
    assert sorted(named) == sorted(sparse), paragraph
    assert "sparse" in paragraph and "absent from individual children" in paragraph


# --------------------------------------------------------------------------- #
# the expansion is derived, never authored
# --------------------------------------------------------------------------- #


def test_the_fixture_expands_at_load_and_the_children_are_never_canonical(
    tmp_path: Path,
) -> None:
    root = _tree(tmp_path)
    loaded = _load(root)
    assert loaded.order == ("scan@0", "scan@1", "scan", "best")
    assert loaded.levels == (("scan@0", "scan@1"), ("scan",), ("best",))
    assert loaded.dependencies == {
        "scan@0": (),
        "scan@1": (),
        "scan": ("scan@0", "scan@1"),
        "best": ("scan",),
    }
    assert loaded.children == {"scan": ("scan@0", "scan@1")}
    assert sorted(loaded.canonical["steps"]) == ["best", "scan"]
    entry = loaded.canonical["steps"]["scan"]
    assert entry["fan_out"] == {"over": {"shards": 2}, "join": {"require": "all"}}
    assert "shard" not in entry and "join" not in entry
    for index, child in enumerate(loaded.children["scan"]):
        step = loaded.document.steps[child]
        assert isinstance(step, ProtocolStep) and step.fan_out is None
        assert step.shard == {
            "index": index,
            "of": 2,
            "over": {"shards": 2},
            "range": [4 * index, 4 * index + 4],
            "points": list(range(4 * index, 4 * index + 4)),
        }
        assert loaded.inner[child] is loaded.inner["scan"]
        assert loaded.inner_digests[child] == loaded.inner_digests["scan"]
    # the parent's entry alone decides the digest: unfanned, no key at all
    plain = _load(root, _unfanned(_raw(root)))
    assert "fan_out" not in plain.canonical["steps"]["scan"]
    assert plain.children == {} and plain.order == ("scan", "best")
    assert plain.digest != loaded.digest


@pytest.mark.parametrize(
    ("over", "expected"),
    [
        ({"shards": 3}, [[0, 1, 2], [3, 4, 5], [6, 7]]),
        ({"shards": 8}, [[i] for i in range(8)]),
        ({"axis": LAYERS}, [[0, 4], [1, 5], [2, 6], [3, 7]]),
        ({"axis": "positions.tap"}, [[0, 1, 2, 3], [4, 5, 6, 7]]),
    ],
    ids=["shards_3", "shards_8", "axis_layers", "axis_tap"],
)
def test_the_expansion_rule(
    tmp_path: Path, over: dict[str, Any], expected: list[list[int]]
) -> None:
    """Shards are contiguous ranges, sizes differing by at most one, the first
    longer; an axis gives one child per value in compiled coordinate order,
    holding the indices whose coordinate equals the value."""
    root = _tree(tmp_path)
    loaded = _load(root, _fanned(_raw(root), "scan", over))
    children = loaded.children["scan"]
    assert len(children) == len(expected)
    points = loaded.inner["scan"].expansion.points
    for index, (child, indices) in enumerate(zip(children, expected)):
        shard = loaded.document.steps[child].shard
        assert shard is not None and shard["points"] == indices
        assert shard["index"] == index and shard["of"] == len(expected)
        if "axis" in over:
            assert {
                points[i].coords[over["axis"]] == shard["value"] for i in indices
            } == {True}
        else:
            assert shard["range"] == [indices[0], indices[-1] + 1]


# --------------------------------------------------------------------------- #
# T15 — the join
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "over",
    [{"shards": 2}, {"axis": LAYERS}],
    ids=["two_shards", "axis_layers"],
)
def test_t15_a_fan_out_joins_row_for_row_to_the_unsharded_run(
    tmp_path: Path, over: dict[str, Any]
) -> None:
    root = _tree(tmp_path)
    plain = _run(_load(root, _unfanned(_raw(root))), root, tmp_path / "plain")
    fanned_loaded = _load(root, _fanned(_raw(root), "scan", over))
    fanned = _run(fanned_loaded, root, tmp_path / "fanned")
    children = fanned_loaded.children["scan"]
    for rel in ("iia.json", "logit_diff.json"):
        assert (fanned.run_root / "scan" / rel).read_bytes() == (
            plain.run_root / "scan" / rel
        ).read_bytes(), rel
    # the shipped select over the joined table reads what it reads unsharded
    assert (fanned.run_root / "best" / "values.json").read_text() == (
        plain.run_root / "best" / "values.json"
    ).read_text()
    # one normalized receipt: the parent's record
    joined, single = _record(fanned.run_root, "scan"), _record(plain.run_root, "scan")
    assert joined["axes"] == single["axes"] == ["positions.tap", LAYERS]
    assert joined["point_digests"] == single["point_digests"]
    assert joined["coords"] == single["coords"]
    assert joined["points"] == single["points"] == 8
    assert joined["document_digest"] == single["document_digest"]
    assert "method" not in joined and "method" not in single  # no method digest
    assert joined["digests"] == single["digests"]  # the same bytes, file for file
    assert "engine" not in joined and "execution" not in joined
    assert "engine" in single and "execution" in single
    assert joined["identity"] == fanned_loaded.step_digests["scan"]
    assert joined["fan_out"] == {
        "over": over,
        "width": len(children),
        "children": list(children),
    }
    assert joined["join"]["require"] == "all"
    assert set(joined["join"]["consumed"]) == set(children)
    for child in children:
        record = _record(fanned.run_root, child)
        assert joined["join"]["consumed"][child] == {
            "identity": record["identity"],
            "points": record["point_digests"],
            "digests": record["digests"],
        }
        assert record["identity"] == fanned_loaded.step_digests[child]
        assert record["shard"] == fanned_loaded.document.steps[child].shard
        assert record["points"] == len(record["shard"]["points"])
        assert len(record["coords"]) == len(record["point_digests"])
        assert record["coords"] == [
            dict(fanned_loaded.inner["scan"].expansion.points[i].coords)
            for i in record["shard"]["points"]
        ]
        assert "engine" in record and "execution" in record
    assert joined["join"]["n_points"] == 8
    assert joined["join"]["n_missing"] == 0 and joined["join"]["n_duplicate"] == 0
    assert "skipped" not in joined["join"]  # `all`: nothing to name
    assert joined["disposition"] == "accepted"
    # the stream: every child then the parent, no new payload key
    lines = read_events(fanned.run_root / EVENTS_FILE)
    started = [
        line["payload"]["step"] for line in lines if line["event"] == "phase_started"
    ]
    assert started == [*children, "scan", "best"]
    assert _payload_keys(fanned.run_root, "phase_started") == {"step", "type"}
    # `forwards` is a protocol child's (§4.3, §8) — the join's line carries none
    assert _payload_keys(fanned.run_root, "phase_completed") == {
        "step",
        "status",
        "forwards",
    }
    assert {
        key
        for line in lines
        if line["event"] == "phase_completed" and line["payload"]["step"] == "scan"
        for key in line["payload"]
    } == {"step", "status"}
    assert terminal(fanned.run_root / EVENTS_FILE)
    # the manifest: children + parent entries, all completed
    assert _statuses(fanned) == {
        **{c: "completed" for c in children},
        "scan": "completed",
        "best": "completed",
    }
    assert not (fanned.run_root / mf.ATTEMPTS_DIR).exists()


def _tamper_at(step: str, edit: Callable[[Path], None]) -> Callable[[Any], None]:
    """An event sink that edits a child's published unit the instant its
    `phase_completed` line is written — after the child published, before the
    join runs (the injection point of the duplicate and missing mutations)."""

    def sink(line: Any) -> None:
        if line["event"] == "phase_completed" and line["payload"]["step"] == step:
            edit(Path(line["_run_root"]) if "_run_root" in line else RUN_ROOT[0])

    return sink


RUN_ROOT: list[Path] = [Path(".")]


def _run_tampered(
    root: Path,
    out: Path,
    step: str,
    edit: Callable[[Path], None],
    raw: dict[str, Any] | None = None,
) -> tuple[WorkflowError, Path]:
    loaded = _load(root, raw)
    run_root = out / loaded.document.output_dir
    RUN_ROOT[0] = run_root
    with pytest.raises(WorkflowError) as err:
        _run(loaded, root, out, sink=_tamper_at(step, edit))
    return err.value, run_root


def test_t15_mutation_a_duplicate_point_is_refused_one_point_one_child(
    tmp_path: Path,
) -> None:
    """The duplicate mutation: a row of another child's point injected into a
    child's published table after that child published — the join refuses it
    as a duplicate by digest; without the check the joined table carries N+1
    rows and nothing else notices."""
    root = _tree(tmp_path)

    def inject(run_root: Path) -> None:
        first = json.loads((run_root / "scan@0" / "iia.json").read_text())
        table = run_root / "scan@1" / "iia.json"
        rows = json.loads(table.read_text())
        rows.append(first[0])  # scan@0's first point, now published twice
        table.write_text(json.dumps(rows))

    err, run_root = _run_tampered(root, tmp_path / "runs", "scan@1", inject)
    assert err.rule == FAN_OUT_RULE
    assert "one point, one child" in str(err) and "'scan@0' and 'scan@1'" in str(err)
    manifest = json.loads((run_root / mf.MANIFEST).read_text())["steps"]
    assert manifest["scan"]["status"] == "failed"
    assert manifest["best"]["status"] == "blocked"
    assert (run_root / mf.ATTEMPTS_DIR / "scan" / "0001" / mf.ATTEMPT_RECORD).is_file()
    assert not (run_root / "scan").exists()


def test_a_row_the_join_cannot_place_is_refused_naming_the_file_and_the_row(
    tmp_path: Path,
) -> None:
    """The unplaceable row, the fail-closed half: a row naming no point (no
    coordinate columns, no `point_digest`, no `point`) and a row whose `point`
    index is outside its child's points are each a rule-19 refusal naming the
    file, the row and the child — an earlier join appended both at the end of
    the joined table and the per-file missing check never saw them."""
    root = _tree(tmp_path)

    def rogue(run_root: Path) -> None:
        table = run_root / "scan@1" / "iia.json"
        rows = json.loads(table.read_text())
        rows.append({"example_id": "0", "metric": "iia", "value": 0.5})
        table.write_text(json.dumps(rows))

    err, run_root = _run_tampered(root, tmp_path / "rogue", "scan@1", rogue)
    assert err.rule == FAN_OUT_RULE
    assert "row 8 of 'iia.json' in 'scan@1' names no point" in str(err)
    assert "never appended out of order" in str(err)
    manifest = json.loads((run_root / mf.MANIFEST).read_text())["steps"]
    assert manifest["scan"]["status"] == "failed"
    assert manifest["best"]["status"] == "blocked"

    def out_of_range(run_root: Path) -> None:
        table = run_root / "scan@1" / "iia.json"
        rows = json.loads(table.read_text())
        rows.append({"example_id": "0", "metric": "iia", "value": 0.5, "point": 7})
        table.write_text(json.dumps(rows))

    err2, _ = _run_tampered(
        _tree(tmp_path / "again"), tmp_path / "range", "scan@1", out_of_range
    )
    assert err2.rule == FAN_OUT_RULE
    assert (
        "row 8 of 'iia.json' in 'scan@1' names point 7, which is not an index of "
        "the child's 4 point(s)"
    ) in str(err2)


def test_a_digest_valued_point_column_places_a_row_in_point_order(
    tmp_path: Path,
) -> None:
    """The unplaceable row, the ordering half: the engine's side tables
    (`train_eval.json`, `fit_diagnostics.json`, `routing_mismatch.json`) name
    their point in a `point` column holding the digest string, and carry no
    coordinate columns; the join places such a row by that digest. Over an
    axis the children interleave the parent's points, so a join that ignored
    the column came out grouped by child. The rows here are the stub's metric
    rows re-shaped into that form: the coordinate columns stripped and the
    digest the child's record declares at the row's point put in `point`."""
    root = _tree(tmp_path)
    plain = _run(_load(root, _unfanned(_raw(root))), root, tmp_path / "plain")
    loaded = _load(root, _fanned(_raw(root), "scan", {"axis": LAYERS}))
    fanned = _run(loaded, root, tmp_path / "fanned")
    axes = _axes(loaded, "scan")
    by_child: list[str] = []
    for child in loaded.children["scan"]:
        record = _record(fanned.run_root, child)
        table = fanned.run_root / child / "iia.json"
        rows = json.loads(table.read_text())
        for row in rows:
            local = record["shard"]["points"].index(_parent_index(loaded, "scan", row))
            row["point"] = record["point_digests"][local]
            for axis in axes:
                del row[axis]
            by_child.append(row["point"])
        table.write_text(json.dumps(rows))
    step = loaded.document.steps["scan"]
    assert isinstance(step, ProtocolStep)
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    fan_out.run_join_step(
        "scan", step, loaded, fanned.run_root, attempt, {"tree_digest": "x"}
    )
    joined = [row["point"] for row in json.loads((attempt / "iia.json").read_text())]
    expected = [
        loaded.inner["scan"].point_digests[_parent_index(loaded, "scan", row)]
        for row in json.loads((plain.run_root / "scan" / "iia.json").read_text())
    ]
    assert joined == expected
    assert by_child != expected  # child order is not point order over an axis


def _lone_behavioral_raw(
    root: Path, over: dict[str, Any] | None = None
) -> dict[str, Any]:
    """The per-child chain's behavioral producer alone: fanned out `over`
    with `require: all`, or unfanned when `over` is None."""
    raw = _behavioral_raw(root)
    qualify = dict(raw["steps"]["qualify"])
    if over is None:
        qualify.pop("fan_out")
    else:
        qualify["fan_out"] = {"over": over, "join": {"require": "all"}}
    return {**raw, "steps": {"qualify": qualify}}


@pytest.mark.parametrize(
    "over",
    [{"shards": 2}, {"axis": PROBE_LAYERS}],
    ids=["two_shards", "axis_probe_layers"],
)
def test_t15_a_fanned_out_behavioral_step_joins_continuations_byte_for_byte(
    tmp_path: Path, over: dict[str, Any]
) -> None:
    """A `continuations.json` row carries BOTH keys — `point` the index
    into the child's request and `point_digest` the point itself — so the
    digest places it and the join must still re-base the index; before the
    fix a two-child join published `point: 0, 0, 1, 1, 0, 0, 1, 1` where the
    unsharded run publishes `0..3`, beside an `outcomes.json` (int `point`
    only) that WAS re-based. The valid twin of
    `test_a_row_whose_point_index_and_point_digest_disagree_is_refused`."""
    root = _tree(tmp_path)
    plain = _run(_load(root, _lone_behavioral_raw(root)), root, tmp_path / "plain")
    loaded = _load(root, _lone_behavioral_raw(root, over))
    fanned = _run(loaded, root, tmp_path / "fanned")
    for rel in (CONTINUATIONS_FILE, OUTCOMES_FILE):
        assert (fanned.run_root / "qualify" / rel).read_bytes() == (
            plain.run_root / "qualify" / rel
        ).read_bytes(), rel
    joined = json.loads((fanned.run_root / "qualify" / CONTINUATIONS_FILE).read_text())
    assert [row["point"] for row in joined] == sorted(row["point"] for row in joined)
    assert {row["point"] for row in joined} == {0, 1, 2, 3}
    # the children's own copies keep their local indices — the join re-based
    local = {
        row["point"]
        for child in loaded.children["qualify"]
        for row in json.loads(
            (fanned.run_root / child / CONTINUATIONS_FILE).read_text()
        )
    }
    assert local == {0, 1}
    joined_record, single = (
        _record(fanned.run_root, "qualify"),
        _record(plain.run_root, "qualify"),
    )
    assert joined_record["outcomes"] == single["outcomes"]
    assert joined_record["point_digests"] == single["point_digests"]
    assert _statuses(fanned)["qualify"] == "completed"


def test_a_row_whose_point_index_and_point_digest_disagree_is_refused(
    tmp_path: Path,
) -> None:
    """The two-key row, the fail-closed half: a `continuations.json` row whose `point`
    index names one of the child's points and whose `point_digest` names
    another is a rule-19 refusal naming the file, the row, the child, both
    indices and the digest — never re-based silently to either; an index
    outside the child's points beside a digest is the same refusal class.
    Twin: `test_t15_a_fanned_out_behavioral_step_joins_continuations_byte_for_byte`."""
    root = _tree(tmp_path)
    loaded = _load(root, _lone_behavioral_raw(root, {"axis": PROBE_LAYERS}))
    fanned = _run(loaded, root, tmp_path / "fanned")
    child = loaded.children["qualify"][1]
    step = loaded.document.steps["qualify"]
    assert isinstance(step, BehavioralStep)
    table = fanned.run_root / child / CONTINUATIONS_FILE
    rows = json.loads(table.read_text())
    assert rows[0]["point"] == 0
    rows[0]["point"] = 1  # the child's other point; the digest still says 0
    table.write_text(json.dumps(rows))
    own = _record(fanned.run_root, child)["shard"]["points"]
    with pytest.raises(WorkflowError) as err:
        fan_out.run_join_step(
            "qualify",
            step,
            loaded,
            fanned.run_root,
            tmp_path / "a1",
            {"tree_digest": "x"},
        )
    assert err.value.rule == FAN_OUT_RULE
    assert (
        f"row 0 of 'continuations.json' in '{child}' names point 1 (the child's "
        f"point at that index is the parent's point {own[1]}) beside point_digest "
        f"{rows[0]['point_digest']} (the parent's point {own[0]})"
    ) in str(err.value)
    assert "the row's two keys disagree about which point it describes" in str(
        err.value
    )
    rows[0]["point"] = 5
    table.write_text(json.dumps(rows))
    with pytest.raises(WorkflowError) as err2:
        fan_out.run_join_step(
            "qualify",
            step,
            loaded,
            fanned.run_root,
            tmp_path / "a2",
            {"tree_digest": "x"},
        )
    assert err2.value.rule == FAN_OUT_RULE
    assert (
        f"row 0 of 'continuations.json' in '{child}' names point 5 beside "
        f"point_digest {rows[0]['point_digest']}, and 5 is not an index of the "
        "child's 2 point(s)"
    ) in str(err2.value)
    assert "disagree" not in str(err2.value)


@pytest.mark.parametrize("value", ["local_index", 99], ids=["in_range", "out_of_range"])
def test_a_coordinate_placed_row_keeps_an_int_column_named_point(
    tmp_path: Path, value: Any
) -> None:
    """The re-base is gated on the
    `point_digest` pairing, not on "an int column named `point`". A metric
    row's columns are open — `outputs.py` splats the point's coordinates in
    as columns and nothing reserves a name — so a metric row carrying an int
    `point` is a row with a coordinate named `point`: the join places it by
    its coordinate columns (that rule dispatches first) and publishes the
    column value for value the
    child's, whether the value falls inside the child's shard (an earlier join
    silently rewrote it to the parent's index) or outside it (an earlier join
    refused it as "the row's two keys"). Twin:
    `test_t15_a_fanned_out_behavioral_step_joins_continuations_byte_for_byte`,
    where the int `point` sits beside a `point_digest` and IS re-based."""
    root = _tree(tmp_path)
    loaded = _load(root, _fanned(_raw(root), "scan", {"shards": 2}))
    fanned = _run(loaded, root, tmp_path / "fanned")
    child = loaded.children["scan"][1]
    record = _record(fanned.run_root, child)
    own = record["shard"]["points"]
    table = fanned.run_root / child / "iia.json"
    rows = json.loads(table.read_text())
    for row in rows:
        local = own.index(_parent_index(loaded, "scan", row))
        row["point"] = local if value == "local_index" else value
        assert row["point"] != own[local]  # were it re-based, it would change
    table.write_text(json.dumps(rows))
    step = loaded.document.steps["scan"]
    assert isinstance(step, ProtocolStep)
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    joined_record = fan_out.run_join_step(
        "scan", step, loaded, fanned.run_root, attempt, {"tree_digest": "x"}
    )
    assert joined_record["status"] == "completed"
    joined = json.loads((attempt / "iia.json").read_text())
    ours = [row for row in joined if _parent_index(loaded, "scan", row) in set(own)]
    assert ours == rows  # the child's rows, the `point` column included
    theirs = [
        row for row in joined if _parent_index(loaded, "scan", row) not in set(own)
    ]
    assert theirs and all("point" not in row for row in theirs)
    assert (attempt / "logit_diff.json").read_bytes() == (
        fanned.run_root / "scan" / "logit_diff.json"
    ).read_bytes()


def test_a_sparse_side_table_one_child_published_joins_without_a_missing_refusal(
    tmp_path: Path,
) -> None:
    """The engine's side tables (`routing_mismatch.json` here) are
    written per occurrence and only when there is something to write
    (`outputs.py:355-359`), so a child publishing one with a row for ONE of
    its two points and a child publishing none of that file is not an
    incomplete run: the join places the rows by their digest-valued `point`,
    in the parent's order, and holds neither the file-presence check nor the
    per-file completeness check against it. The twin at the end: a dense table
    (a coordinate-placed metric table) a child did not publish IS the
    missing-file refusal."""
    root = _tree(tmp_path)
    loaded = _load(root, _fanned(_raw(root), "scan", {"axis": LAYERS}))
    fanned = _run(loaded, root, tmp_path / "fanned")
    a, b, c, d = loaded.children["scan"]  # four layers, two points each
    full = list(loaded.inner["scan"].point_digests)
    rows_by_child = {
        a: [{"point": _record(fanned.run_root, a)["point_digests"][1], "n": 1}],
        c: [{"point": _record(fanned.run_root, c)["point_digests"][0], "n": 2}],
    }
    for child, rows in rows_by_child.items():
        record = _record(fanned.run_root, child)
        write_table(fanned.run_root / child / "routing_mismatch.json", rows)
        record["files"] = sorted([*record["files"], "routing_mismatch.json"])
        (fanned.run_root / child / SIDECAR).write_text(json.dumps(record))
    for child in (b, d):
        assert "routing_mismatch.json" not in _record(fanned.run_root, child)["files"]
    step = loaded.document.steps["scan"]
    assert isinstance(step, ProtocolStep)
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    record = fan_out.run_join_step(
        "scan", step, loaded, fanned.run_root, attempt, {"tree_digest": "x"}
    )
    assert record["status"] == "completed"
    assert "routing_mismatch.json" in record["files"]
    joined = json.loads((attempt / "routing_mismatch.json").read_text())
    expected = sorted(
        (*rows_by_child[a], *rows_by_child[c]), key=lambda r: full.index(r["point"])
    )
    assert joined == expected and len(joined) == 2  # parent order, two rows
    # the metric tables are untouched by the side table
    assert (attempt / "iia.json").read_bytes() == (
        fanned.run_root / "scan" / "iia.json"
    ).read_bytes()
    # a metric table (coordinate-placed) is dense: absent from one child, missing
    record_b = _record(fanned.run_root, b)
    record_b["files"] = [rel for rel in record_b["files"] if rel != "iia.json"]
    (fanned.run_root / b / SIDECAR).write_text(json.dumps(record_b))
    with pytest.raises(WorkflowError) as err:
        fan_out.run_join_step(
            "scan",
            step,
            loaded,
            fanned.run_root,
            tmp_path / "twin",
            {"tree_digest": "x"},
        )
    assert err.value.rule == FAN_OUT_RULE
    assert f"'{b}' published no 'iia.json'" in str(err.value)


def test_an_empty_dense_table_beside_a_child_that_omitted_it_is_missing(
    tmp_path: Path,
) -> None:
    """Density is the writer's, not
    the data's. An earlier join read it off the rows found, so a dense table one
    child published EMPTY and another omitted from `files` tallied zero dense
    rows, passed the file-presence gate, short-circuited the per-file check on
    the same emptiness and joined `completed` with an empty table. Declared
    (`_SPARSE_FILES`), the gate refuses the absent child whatever the present
    children published, before any table is read. Twin:
    `test_a_sparse_side_table_one_child_published_joins_without_a_missing_refusal`
    — a declared side table one child omitted joins from the children that
    have it."""
    root = _tree(tmp_path)
    loaded = _load(root, _fanned(_raw(root), "scan", {"shards": 2}))
    fanned = _run(loaded, root, tmp_path / "fanned")
    a, b = loaded.children["scan"]
    write_table(fanned.run_root / a / "iia.json", [])
    record_b = _record(fanned.run_root, b)
    record_b["files"] = [rel for rel in record_b["files"] if rel != "iia.json"]
    (fanned.run_root / b / SIDECAR).write_text(json.dumps(record_b))
    step = loaded.document.steps["scan"]
    assert isinstance(step, ProtocolStep)
    with pytest.raises(WorkflowError) as err:
        fan_out.run_join_step(
            "scan",
            step,
            loaded,
            fanned.run_root,
            tmp_path / "attempt",
            {"tree_digest": "x"},
        )
    assert err.value.rule == FAN_OUT_RULE
    assert f"'{b}' published no 'iia.json'" in str(err.value)
    assert "one point, one child" not in str(err.value)
    assert not (tmp_path / "attempt" / "iia.json").exists()


def test_a_point_declared_twice_by_one_child_is_a_duplicate_refusal(
    tmp_path: Path,
) -> None:
    """A repeated digest: one child's `point_digests` repeating a digest has
    one owner in the cross-child check and fills two slots of the parent's
    list, so an earlier join counted it neither duplicate nor missing and the
    receipt carried the repeat."""
    root = _tree(tmp_path)

    def repeat(run_root: Path) -> None:
        sidecar = run_root / "scan@1" / SIDECAR
        record = json.loads(sidecar.read_text())
        record["point_digests"][1] = record["point_digests"][0]
        sidecar.write_text(json.dumps(record))

    err, run_root = _run_tampered(root, tmp_path / "runs", "scan@1", repeat)
    assert err.rule == FAN_OUT_RULE
    assert "is declared twice in the point_digests of 'scan@1'" in str(err)
    assert "one point, one child" in str(err)
    assert "'scan@0' and 'scan@1'" not in str(err)
    manifest = json.loads((run_root / mf.MANIFEST).read_text())["steps"]
    assert manifest["scan"]["status"] == "failed"


def test_a_foreign_point_is_refused_naming_what_the_children_declared(
    tmp_path: Path,
) -> None:
    """A foreign point, over the two row shapes that name a point by
    something the join looks up: a metric row whose coordinate columns name
    a point no child published is refused naming those coordinates, and a
    side-table row naming a digest no child declared is refused saying
    exactly that (the join's point list is the children's own `point_digests`
    and `coords`; a deferred document has no load-time list) — never "not a
    point of the parent (N compiled points)"."""
    root = _tree(tmp_path)

    def relocate(run_root: Path) -> None:
        table = run_root / "scan@1" / "iia.json"
        rows = json.loads(table.read_text())
        rows[0][LAYERS] = 99  # no layer of the sweep
        table.write_text(json.dumps(rows))

    err, _ = _run_tampered(root, tmp_path / "runs", "scan@1", relocate)
    assert err.rule == FAN_OUT_RULE
    assert "row 0 of 'iia.json' in 'scan@1' names coordinates" in str(err)
    assert f"'{LAYERS}': 99" in str(err)
    assert "which no child of 'scan' published" in str(err)
    assert "compiled points" not in str(err) and "expansion" not in str(err)
    assert "one point, one child" not in str(err)

    foreign = "f" * 64

    def rename(run_root: Path) -> None:
        # a side-table row: no coordinate columns, a digest-valued `point`
        table = run_root / "scan@1" / "iia.json"
        rows = json.loads(table.read_text())
        rows.append({"metric": "iia", "value": 0.5, "point": foreign})
        table.write_text(json.dumps(rows))

    err2, _ = _run_tampered(
        _tree(tmp_path / "again"), tmp_path / "digest", "scan@1", rename
    )
    assert err2.rule == FAN_OUT_RULE
    assert (
        f"point {foreign}, named by a row 'scan@1' published, is no point any "
        "child of 'scan' declared as its own"
    ) in str(err2)
    assert "compiled points" not in str(err2)


PICK = """
import json
from pathlib import Path


def main(inputs, outputs):
    Path(outputs["values"]).write_text(json.dumps({"layers": inputs["layers"]}))
"""


def _deferred_sweep_raw(root: Path, emitted: list[int]) -> dict[str, Any]:
    """The scan fixture with its layer sweep fed by a script step: `pick`
    emits `layers` as ``emitted`` and declares the representative ``[0, 1]``,
    so the load-time compile sees layers 0 and 1 and the run-time compile
    sees ``emitted`` — the same number of points at other coordinates."""
    (root / "scripts" / "pick.py").write_text(PICK)
    raw = _raw(root)
    scan = dict(raw["steps"]["scan"])
    scan["set"] = {LAYERS: {"sweep": {"artifact": "pick", "key": "layers"}}}
    raw["steps"] = {
        "pick": {
            "type": "script",
            "script": {"path": "scripts/pick.py"},
            "inputs": {"layers": list(emitted)},
            "outputs": {"values": {"file": "values.json", "keys": {"layers": [0, 1]}}},
        },
        "scan": scan,
        "best": raw["steps"]["best"],
    }
    return raw


def test_a_deferred_sweep_joins_on_the_run_time_coordinates(tmp_path: Path) -> None:
    """A fanned-out document whose sweep values come from a step's
    `values.json`, with a representative that differs from the emitted value:
    the children's rows carry the run-time coordinates (layers 2 and 3), the
    load-time expansion knows only the representative (layers 0 and 1), and
    the join places every row — through the coordinates the children
    published, never the load-time compile — to the same tables an unfanned
    run writes."""
    root = _tree(tmp_path)
    emitted = [2, 3]
    fanned_loaded = _load(
        root, _fanned(_deferred_sweep_raw(root, emitted), "scan", {"shards": 2})
    )
    load_time = {p.coords[LAYERS] for p in fanned_loaded.inner["scan"].expansion.points}
    assert load_time == {0, 1}  # the representative, not the emitted value
    fanned = _run(fanned_loaded, root, tmp_path / "fanned")
    plain = _run(
        _load(root, _unfanned(_deferred_sweep_raw(root, emitted))),
        root,
        tmp_path / "plain",
    )
    assert _statuses(fanned) == {
        "pick": "completed",
        "scan@0": "completed",
        "scan@1": "completed",
        "scan": "completed",
        "best": "completed",
    }
    for rel in ("iia.json", "logit_diff.json"):
        assert (fanned.run_root / "scan" / rel).read_bytes() == (
            plain.run_root / "scan" / rel
        ).read_bytes(), rel
    rows = json.loads((fanned.run_root / "scan" / "iia.json").read_text())
    assert {row[LAYERS] for row in rows} == set(emitted)
    assert (
        len(rows) == 4 * 2
    )  # four run-time points (2 taps x 2 layers), two examples each
    joined, single = _record(fanned.run_root, "scan"), _record(plain.run_root, "scan")
    assert joined["point_digests"] == single["point_digests"]
    assert joined["coords"] == single["coords"]
    assert {coords[LAYERS] for coords in joined["coords"]} == set(emitted)
    # the children published the run-time coordinates, aligned with their digests
    for child in fanned_loaded.children["scan"]:
        record = _record(fanned.run_root, child)
        assert len(record["coords"]) == len(record["point_digests"]) == 2
        assert {coords[LAYERS] for coords in record["coords"]} <= set(emitted)
    assert (
        json.loads((fanned.run_root / "best" / "values.json").read_text())["best_layer"]
        in emitted
    )


def test_a_child_record_without_coordinates_is_a_missing_point(
    tmp_path: Path,
) -> None:
    """The join places a metric row through the coordinates the children
    published; a child's record without them (a record written by another
    runner, or tampered) is a missing point, refused as a record without
    `point_digests` is."""
    root = _tree(tmp_path)

    def strip(run_root: Path) -> None:
        sidecar = run_root / "scan@1" / SIDECAR
        record = json.loads(sidecar.read_text())
        del record["coords"]
        sidecar.write_text(json.dumps(record))

    err, _ = _run_tampered(root, tmp_path / "runs", "scan@1", strip)
    assert err.rule == FAN_OUT_RULE
    assert "was published by no child" in str(err)
    assert "'scan@1' published the coordinates of 0 of its 4 points" in str(err)


def test_a_row_at_a_skipped_childs_coordinates_is_foreign_under_selected(
    tmp_path: Path,
) -> None:
    """Under `require: selected` the skipped child's points are not missing,
    but nothing places a row there: a row of the unskipped child rewritten to
    the skipped child's coordinates names a point no child of the join
    published, and is refused under rule 19 saying so."""
    root = _tree(tmp_path)
    raw = _behavioral_raw(root)  # apply@0 runs (layer 0), apply@1 is skipped

    def relocate(run_root: Path) -> None:
        table = run_root / "apply@0" / "iia.json"
        rows = json.loads(table.read_text())
        rows[0][PROBE_LAYERS] = 1  # apply@1's coordinate
        table.write_text(json.dumps(rows))

    err, run_root = _run_tampered(root, tmp_path / "runs", "apply@0", relocate, raw)
    assert err.rule == FAN_OUT_RULE
    assert "row 0 of 'iia.json' in 'apply@0' names coordinates" in str(err)
    assert f"'{PROBE_LAYERS}': 1" in str(err)
    assert "which no child of 'apply' published" in str(err)
    manifest = json.loads((run_root / mf.MANIFEST).read_text())["steps"]
    assert manifest["apply"]["status"] == "failed"
    assert manifest["apply@1"]["status"] == "skipped"


# --------------------------------------------------------------------------- #
# T16 — missing vs selected
# --------------------------------------------------------------------------- #


def test_t16_a_missing_row_is_a_distinct_refusal(tmp_path: Path) -> None:
    root = _tree(tmp_path)

    def delete_row(run_root: Path) -> None:
        # every row at the first row's coordinates: one point's rows
        table = run_root / "scan@1" / "iia.json"
        rows = json.loads(table.read_text())
        point = {axis: rows[0][axis] for axis in SCAN_AXES}
        table.write_text(
            json.dumps([r for r in rows if {a: r[a] for a in SCAN_AXES} != point])
        )

    err, run_root = _run_tampered(root, tmp_path / "rows", "scan@1", delete_row)
    assert err.rule == FAN_OUT_RULE
    assert "was published by no child in 'iia.json'" in str(err)
    assert "'scan@1' published 3 of its 4 points" in str(err)
    assert "one point, one child" not in str(err)
    manifest = json.loads((run_root / mf.MANIFEST).read_text())["steps"]
    assert (
        manifest["scan"]["status"] == "failed"
        and manifest["best"]["status"] == "blocked"
    )
    assert (run_root / mf.ATTEMPTS_DIR / "scan" / "0001" / mf.ATTEMPT_RECORD).is_file()

    def delete_digest(run_root: Path) -> None:
        sidecar = run_root / "scan@1" / SIDECAR
        record = json.loads(sidecar.read_text())
        record["point_digests"] = record["point_digests"][1:]
        sidecar.write_text(json.dumps(record))

    err2, _ = _run_tampered(
        _tree(tmp_path / "again"), tmp_path / "digests", "scan@1", delete_digest
    )
    assert err2.rule == FAN_OUT_RULE
    assert "was published by no child" in str(err2)
    assert "'scan@1' published 3 of its 4 points" in str(err2)
    assert str(err2) != str(err)


def test_t16_a_missing_row_is_a_distinct_refusal_for_a_point_digest_table(
    tmp_path: Path,
) -> None:
    """T16 over the second dense shape: a `continuations.json` row names its
    point by `point_digest`, whose writer promises one row per point, so a
    point with no row in it is the missing-row refusal exactly as for a
    coordinate-placed metric table (`test_t16_a_missing_row_is_a_distinct_refusal`)
    — the sparse-table split exempts only digest-valued `point` side tables."""
    root = _tree(tmp_path)
    loaded = _load(root, _lone_behavioral_raw(root, {"axis": PROBE_LAYERS}))
    fanned = _run(loaded, root, tmp_path / "fanned")
    child = loaded.children["qualify"][1]
    step = loaded.document.steps["qualify"]
    assert isinstance(step, BehavioralStep)
    table = fanned.run_root / child / CONTINUATIONS_FILE
    rows = json.loads(table.read_text())
    digest = rows[0]["point_digest"]
    table.write_text(json.dumps([r for r in rows if r["point_digest"] != digest]))
    with pytest.raises(WorkflowError) as err:
        fan_out.run_join_step(
            "qualify",
            step,
            loaded,
            fanned.run_root,
            tmp_path / "a",
            {"tree_digest": "x"},
        )
    assert err.value.rule == FAN_OUT_RULE
    assert f"point {digest}" in str(err.value)
    assert "was published by no child in 'continuations.json'" in str(err.value)
    assert f"'{child}' published 1 of its 2 points" in str(err.value)


def test_t16_an_absent_child_is_missing_not_skipped(tmp_path: Path) -> None:
    """The pure join over a tree missing one child: the refusal names the
    child as failed or absent."""
    root = _tree(tmp_path)
    loaded = _load(root)
    out = tmp_path / "runs"
    result = _run(loaded, root, out)
    shutil.rmtree(result.run_root / "scan@1")
    step = loaded.document.steps["scan"]
    assert isinstance(step, ProtocolStep)
    with pytest.raises(WorkflowError) as err:
        fan_out.run_join_step(
            "scan",
            step,
            loaded,
            result.run_root,
            tmp_path / "attempt",
            {"tree_digest": "x"},
        )
    assert err.value.rule == FAN_OUT_RULE
    assert "'scan@1' is failed or absent" in str(err.value)


def test_t16_a_selected_join_publishes_the_children_the_verdicts_left(
    tmp_path: Path,
) -> None:
    """The per_target twin: qualify@0 passes (layer 0 says the answer),
    qualify@1 fails; gate@0 keeps apply@0, gate@1 keeps narrow@1; the two
    selective joins publish one child each and name the skipped one by
    `evidence_identity`."""
    root = _tree(tmp_path)
    loaded = _load(root, _behavioral_raw(root))
    assert loaded.children == {
        "qualify": ("qualify@0", "qualify@1"),
        "apply": ("apply@0", "apply@1"),
        "narrow": ("narrow@0", "narrow@1"),
        "gate": ("gate@0", "gate@1"),
    }
    gate0 = loaded.document.steps["gate@0"]
    assert isinstance(gate0, ConditionalStep)
    assert gate0.predicate["decision"]["step"] == "qualify@0"
    assert gate0.on_true == ("apply@0",) and gate0.on_false == ("narrow@0",)
    assert set(loaded.dependencies["apply@0"]) >= {"gate@0", "gate"}
    assert set(loaded.dependencies["gate@0"]) == {"qualify@0"}
    result = _run(loaded, root, tmp_path / "runs")
    run_root = result.run_root
    assert _statuses(result) == {
        "qualify@0": "completed",
        "qualify@1": "completed",
        "qualify": "completed",
        "gate@0": "completed",
        "gate@1": "completed",
        "gate": "completed",
        "apply@0": "completed",
        "apply@1": "skipped",
        "apply": "completed",
        "narrow@0": "skipped",
        "narrow@1": "completed",
        "narrow": "completed",
    }
    assert not (run_root / "apply@1").exists() and not (run_root / "narrow@0").exists()
    decision1 = json.loads((run_root / "qualify@1" / DECISION_FILE).read_text())
    assert decision1["outcome"] == "fail"
    apply = _record(run_root, "apply")
    assert apply["join"]["require"] == "selected"
    assert list(apply["join"]["consumed"]) == ["apply@0"]
    assert apply["join"]["skipped"] == [
        {
            "child": "apply@1",
            "skipped_by": result.manifest["steps"]["apply@1"]["skipped_by"],
        }
    ]
    assert (
        apply["join"]["skipped"][0]["skipped_by"]["evidence_identity"]
        == (decision1["evidence_identity"])
    )
    assert apply["join"]["skipped"][0]["skipped_by"]["conditional"] == "gate@1"
    rows = json.loads((run_root / "apply" / "iia.json").read_text())
    assert {row[PROBE_LAYERS] for row in rows} == {0}
    assert len(rows) == 2 * 2  # two points at layer 0, two examples each
    assert apply["points"] == 4 and len(apply["point_digests"]) == 4
    assert apply["join"]["n_points"] == 2
    # the behavioral join: one decision over the summed counts
    qualify = _record(run_root, "qualify")
    decision = json.loads((run_root / "qualify" / DECISION_FILE).read_text())
    counts = qualify["outcomes"]
    assert counts["n"] == 16 and counts["correct"] == 8
    assert decision["measured_inputs"]["n"] == 16
    assert decision["outcome"] == "fail" and decision["step"] == "qualify"
    outcomes_sha = hashlib.sha256(
        (run_root / "qualify" / OUTCOMES_FILE).read_bytes()
    ).hexdigest()
    assert (
        decision["evidence_identity"]
        == f"{outcomes_sha}:{loaded.step_digests['qualify']}"
    )
    assert qualify["decision"] == DECISION_FILE and "engine" not in qualify
    assert qualify["retain"]["n"] == 16 and qualify["split"] == "development"
    joined = json.loads((run_root / "qualify" / OUTCOMES_FILE).read_text())
    assert [row["point"] for row in joined] == sorted(row["point"] for row in joined)
    assert {row["point"] for row in joined} == {0, 1, 2, 3}
    # the parent conditional: one verdict per child
    gate = _record(run_root, "gate")
    assert gate["verdicts"] == {"gate@0": True, "gate@1": False}
    assert gate["evidence"]["evidence_identity"] == decision["evidence_identity"]
    assert sorted(gate["skipped"]) == ["apply@1", "narrow@0"]
    assert terminal(run_root / EVENTS_FILE)


def test_t16_a_global_conditional_skips_the_parent_and_its_children(
    tmp_path: Path,
) -> None:
    """The global conditional's second half: `scan` on the false side — its
    children are skipped with it (no `scan@i/` directory), `best` transitively."""
    root = _tree(tmp_path)
    loaded = _load(root, _gated_raw(root, score=0.1))
    result = _run(loaded, root, tmp_path / "runs")
    assert _statuses(result) == {
        "measure": "completed",
        "gate_k": "completed",
        "gate": "completed",
        "scan@0": "skipped",
        "scan@1": "skipped",
        "scan": "skipped",
        "best": "skipped",
        "probe": "completed",
    }
    for name in ("scan@0", "scan@1", "scan", "best"):
        assert not (result.run_root / name).exists(), name
    assert result.manifest["steps"]["scan@0"]["skipped_by"]["conditional"] == "gate"
    assert result.manifest["steps"]["scan@0"]["skipped_by"]["transitive_from"] == []
    assert sorted(_record(result.run_root, "gate")["skipped"]) == [
        "scan",
        "scan@0",
        "scan@1",
    ]
    # the twin: a passing measurement runs every child and the join
    passing = _run(_load(root, _gated_raw(root, score=0.9)), root, tmp_path / "pass")
    assert (
        _statuses(passing)["scan"] == "completed"
        and _statuses(passing)["probe"] == "skipped"
    )


def test_t16_a_selected_join_with_every_child_skipped_is_skipped(
    tmp_path: Path,
) -> None:
    """Nothing left to publish: when every child of a selective join is
    skipped the join is skipped with them, never a receipt over nothing."""
    root = _tree(tmp_path)
    # min_correct_rate 0.0: every child passes, so `narrow` loses every child
    loaded = _load(root, _behavioral_raw(root, min_correct_rate=0.0))
    result = _run(loaded, root, tmp_path / "runs")
    statuses = _statuses(result)
    assert (
        statuses["apply@0"] == statuses["apply@1"] == statuses["apply"] == "completed"
    )
    assert (
        statuses["narrow@0"] == statuses["narrow@1"] == statuses["narrow"] == "skipped"
    )
    assert sorted(
        result.manifest["steps"]["narrow"]["skipped_by"]["transitive_from"]
    ) == [
        "narrow@0",
        "narrow@1",
    ]
    assert not (result.run_root / "narrow").exists()


# --------------------------------------------------------------------------- #
# T17 — the legitimate population, identities, --resume
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "path", SHIPPED + DEMO_WORKFLOWS, ids=[p.stem for p in SHIPPED + DEMO_WORKFLOWS]
)
def test_t17_every_existing_workflow_loads_with_no_fan_out_key(
    path: Path, env: Any
) -> None:
    """A §2.9 key on any entry would move every workflow digest in the repo
    (the mutation: emit `fan_out` unconditionally in `_canonicalize`)."""
    loaded = load_workflow(path, env if path.parent == WORKFLOWS else _demo_env(path))
    assert loaded.children == {}
    for name, entry in loaded.canonical["steps"].items():
        assert not set(entry) & set(FAN_OUT_KEYS), (name, sorted(entry))
        assert CHILD_SEPARATOR not in name


def test_t17_the_conditional_chain_is_unchanged(env: Any, tmp_path: Path) -> None:
    """T14 again: identities for every existing kind, the chain's order and
    records with no §2.9 key, and a clean run then `--resume` byte-identical."""
    kinds: set[str] = set()
    for name in ("mean_ablation.json", "weekdays.json", "pca_basis.json"):
        kinds |= _identity_kinds(load_workflow(WORKFLOWS / name, env))
    root = _chain_tree(tmp_path)
    loaded = _chain_load(root)
    kinds |= _identity_kinds(loaded)
    assert kinds == {"intervention_protocol", "script", "decision", "conditional"}
    assert loaded.order == CHAIN and loaded.children == {}
    out = tmp_path / "runs"
    first = _chain_run(loaded, root, out)
    for name in ("measure", "gate_k", "gate", "fit", "report"):
        record = _record(first.run_root, name)
        assert not set(record) & set(FAN_OUT_KEYS), name
    before = _snapshot(first.run_root)
    again = _chain_run(loaded, root, out, resume=True)
    assert _snapshot(again.run_root) == before
    assert (
        _statuses(again)["fit"] == "reused" and _statuses(again)["probe"] == "skipped"
    )


def test_t17_identities_of_the_join_and_its_children(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    loaded = _load(root)
    for name, step in loaded.document.steps.items():
        got = runner._step_identity(loaded, name, step)  # pyright: ignore[reportPrivateUsage]
        if name in ("scan", "scan@0", "scan@1", "best"):
            assert got == loaded.step_digests[name], name
        if (
            isinstance(step, ProtocolStep)
            and step.fan_out is None
            and step.shard is None
        ):
            assert got == loaded.inner_digests[name]  # an unfanned protocol step
    # an unfanned protocol step keeps its inner digest and has no entry digest
    plain = _load(root, _unfanned(_raw(root)))
    step = plain.document.steps["scan"]
    assert runner._step_identity(plain, "scan", step) == plain.inner_digests["scan"]  # pyright: ignore[reportPrivateUsage]
    assert "scan" not in plain.step_digests
    # the children's identities differ from each other and from the parent's
    digests = {loaded.step_digests[n] for n in ("scan", "scan@0", "scan@1")}
    assert len(digests) == 3
    # a width change changes every child's identity and the parent's
    three = _load(root, _fanned(_raw(root), "scan", {"shards": 3}))
    assert three.step_digests["scan"] != loaded.step_digests["scan"]
    assert three.step_digests["scan@0"] != loaded.step_digests["scan@0"]
    assert three.step_digests["scan@1"] != loaded.step_digests["scan@1"]
    assert three.digest != loaded.digest


def test_t17_a_clean_run_then_resume_reuses_children_and_join(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    loaded = _load(root)
    out = tmp_path / "runs"
    first = _run(loaded, root, out)
    before = _snapshot(first.run_root)
    again = _run(loaded, root, out, resume=True)
    assert _statuses(again) == {
        "scan@0": "reused",
        "scan@1": "reused",
        "scan": "reused",
        "best": "reused",
    }
    assert _snapshot(again.run_root) == before


def test_t17_a_width_change_re_runs_every_child_under_resume(tmp_path: Path) -> None:
    """The identity mutation's second half: were a child identified by the inner digest,
    `--resume` after `shards: 2 → 3` would reuse `scan@0` with the wrong
    points; it is identified by its parent's entry and its shard instead."""
    root = _tree(tmp_path)
    out = tmp_path / "runs"
    first = _run(_load(root), root, out)
    old_child = _record(first.run_root, "scan@0")
    assert old_child["points"] == 4
    three = _load(root, _fanned(_raw(root), "scan", {"shards": 3}))
    again = _run(three, root, out, resume=True)
    statuses = _statuses(again)
    assert statuses["scan@0"] == statuses["scan@1"] == statuses["scan@2"] == "completed"
    assert statuses["scan"] == "completed"
    new_child = _record(again.run_root, "scan@0")
    assert new_child["points"] == 3 and new_child["identity"] != old_child["identity"]
    assert new_child["shard"]["points"] == [0, 1, 2]
    # the prior units are retained, superseded, never overwritten
    assert (
        again.run_root
        / mf.ATTEMPTS_DIR
        / "scan@0"
        / f"0001{mf.SUPERSEDED_SUFFIX}"
        / SIDECAR
    ).is_file()
    # the joined table is the same bytes: the width is not in the data
    assert (again.run_root / "scan" / "iia.json").read_bytes() == (
        first.run_root / "scan" / "iia.json"
    ).read_bytes()


def test_the_join_runs_no_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The join is re-made under `--resume` from reused children with
    `route_engine` and the reference engine's `load_model` monkeypatched to
    raise: neither is entered, because the join consumes records and tables."""
    root = _tree(tmp_path)
    loaded = _load(root)
    out = tmp_path / "runs"
    first = _run(loaded, root, out)
    shutil.rmtree(first.run_root / "scan")
    _sentinels(monkeypatch)
    again = run_workflow(loaded, _env(root), out, None, resume=True)
    assert _statuses(again) == {
        "scan@0": "reused",
        "scan@1": "reused",
        "scan": "completed",
        "best": "reused",
    }


def test_the_evidence_clause_re_runs_a_join_whose_child_moved(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    loaded = _load(root)
    out = tmp_path / "runs"
    first = _run(loaded, root, out)
    step = loaded.document.steps["scan"]
    record = _record(first.run_root, "scan")
    assert fan_out.evidence_holds(step, first.run_root / "scan", record)
    # a child re-published under another identity: the join does not hold
    sidecar = first.run_root / "scan@1" / SIDECAR
    child = json.loads(sidecar.read_text())
    child["identity"] = "0" * 64
    sidecar.write_text(json.dumps(child))
    assert not fan_out.evidence_holds(step, first.run_root / "scan", record)
    again = _run(loaded, root, out, resume=True)
    assert _statuses(again)["scan@1"] == "completed"  # not reusable either
    # …and the child re-ran to the very record the join consumed, so the
    # join holds again and is reused: the clause binds to the children's
    # records, not to the fact that a child ran
    assert _statuses(again)["scan"] == "reused"
    # the join's own memory of a child moved: the children stay reused, the
    # join alone is re-made
    join_sidecar = again.run_root / "scan" / SIDECAR
    joined = json.loads(join_sidecar.read_text())
    joined["join"]["consumed"]["scan@1"]["identity"] = "1" * 64
    join_sidecar.write_text(json.dumps(joined))
    assert not fan_out.evidence_holds(step, again.run_root / "scan", joined)
    third = _run(loaded, root, out, resume=True)
    assert _statuses(third) == {
        "scan@0": "reused",
        "scan@1": "reused",
        "scan": "completed",
        "best": "reused",
    }
    # every other kind holds trivially
    assert fan_out.evidence_holds(
        loaded.document.steps["best"], first.run_root / "best", {}
    )
    assert fan_out.evidence_holds(
        loaded.document.steps["scan@0"], first.run_root / "scan@0", {}
    )


# --------------------------------------------------------------------------- #
# T18 — declared, not run-time: the refusals and their twins
# --------------------------------------------------------------------------- #


def _scan(changes: dict[str, Any]) -> Callable[[dict[str, Any]], dict[str, Any]]:
    def mutate(raw: dict[str, Any]) -> dict[str, Any]:
        for key, value in changes.items():
            if value is None:
                raw["steps"]["scan"].pop(key, None)
            else:
                raw["steps"]["scan"][key] = value
        return raw

    return mutate


def _with_other(
    raw: dict[str, Any], control: dict[str, Any] | None = None
) -> dict[str, Any]:
    raw["steps"]["other"] = {
        "type": "intervention_protocol",
        "document": "protocols/scan.json",
    }
    if control is not None:
        raw["steps"]["other"]["control"] = control
    return raw


def _with_bundle(root: Path, raw: dict[str, Any]) -> dict[str, Any]:
    doc = json.loads((root / "protocols" / "scan.json").read_text())
    doc["method"]["save"].append(
        {
            "read": "v_cf",
            "model": "original_counterfactual",
            "file_path": "v_cf.safetensors",
        }
    )
    (root / "protocols" / "scan_bundle.json").write_text(json.dumps(doc))
    raw["steps"]["scan"]["document"] = "protocols/scan_bundle.json"
    return raw


def _with_ledger(root: Path, raw: dict[str, Any]) -> dict[str, Any]:
    """scan.json plus a `location_ledger` save entry (IM spec §2.12) — the
    one non-value save kind; its rows name their point by digest string, and
    the ledger is a certificate, so a fan-out over it is
    refused at load."""
    doc = json.loads((root / "protocols" / "scan.json").read_text())
    doc["method"]["save"].append(
        {"kind": "location_ledger", "file_path": "ledger.json"}
    )
    (root / "protocols" / "scan_ledger.json").write_text(json.dumps(doc))
    raw["steps"]["scan"]["document"] = "protocols/scan_ledger.json"
    return raw


def _with_save_named(root: Path, raw: dict[str, Any], file_path: str) -> dict[str, Any]:
    """scan.json with its `logit_diff` metric table saved under `file_path`
    instead of `logit_diff.json` — a `SaveEntry.file_path` is a free string
    (IM spec §2.12), so a document may claim any name, one of the engine's
    sparse side tables' included."""
    doc = json.loads((root / "protocols" / "scan.json").read_text())
    (entry,) = [e for e in doc["method"]["save"] if e["file_path"] == "logit_diff.json"]
    entry["file_path"] = file_path
    (root / "protocols" / "scan_named.json").write_text(json.dumps(doc))
    raw["steps"]["scan"]["document"] = "protocols/scan_named.json"
    return raw


#: the refusals built against the fixture tree (a sibling document is written)
_TREE_CASES: dict[str, Callable[[Path, dict[str, Any]], dict[str, Any]]] = {
    "fan_out_on_a_bundle_saving_document": _with_bundle,
    "fan_out_on_a_ledger_saving_document": _with_ledger,
}


FAN_OUT_REFUSALS: dict[str, tuple[Callable[..., dict[str, Any]], int, str, str]] = {
    "shards_auto": (
        _scan({"fan_out": {"over": {"shards": "auto"}, "join": {"require": "all"}}}),
        19,
        "steps.scan.fan_out.over.shards",
        "never read at run time",
    ),
    "over_rows": (
        _scan({"fan_out": {"over": {"rows": 2}, "join": {"require": "all"}}}),
        19,
        "steps.scan.fan_out.over",
        "one of ['axis', 'shards'], got 'rows'",
    ),
    "shards_a_float": (
        _scan({"fan_out": {"over": {"shards": 2.5}, "join": {"require": "all"}}}),
        19,
        "steps.scan.fan_out.over.shards",
        "literal integer of at least 2",
    ),
    "shards_a_bool": (
        _scan({"fan_out": {"over": {"shards": True}, "join": {"require": "all"}}}),
        19,
        "steps.scan.fan_out.over.shards",
        "literal integer of at least 2",
    ),
    "shards_one": (
        _scan({"fan_out": {"over": {"shards": 1}, "join": {"require": "all"}}}),
        19,
        "steps.scan.fan_out.over.shards",
        "at least 2",
    ),
    "shards_more_than_points": (
        _scan({"fan_out": {"over": {"shards": 9}, "join": {"require": "all"}}}),
        19,
        "steps.scan.fan_out.over.shards",
        "compiles 8 point(s)",
    ),
    "axis_not_of_the_document": (
        _scan(
            {
                "fan_out": {
                    "over": {"axis": "sites.nope.layers"},
                    "join": {"require": "all"},
                }
            }
        ),
        19,
        "steps.scan.fan_out.over.axis",
        "(has ['positions.tap', 'sites.target.layers'])",
    ),
    "axis_empty": (
        _scan({"fan_out": {"over": {"axis": ""}, "join": {"require": "all"}}}),
        19,
        "steps.scan.fan_out.over.axis",
        "non-empty string",
    ),
    "over_two_keys": (
        _scan(
            {
                "fan_out": {
                    "over": {"axis": LAYERS, "shards": 2},
                    "join": {"require": "all"},
                }
            }
        ),
        19,
        "steps.scan.fan_out.over",
        "exactly one of",
    ),
    "over_missing": (
        _scan({"fan_out": {"join": {"require": "all"}}}),
        19,
        "steps.scan.fan_out.over",
        "declares 'over'",
    ),
    "join_missing": (
        _scan({"fan_out": {"over": {"shards": 2}}}),
        19,
        "steps.scan.fan_out.join",
        "declares 'join'",
    ),
    "require_unknown": (
        _scan({"fan_out": {"over": {"shards": 2}, "join": {"require": "any"}}}),
        19,
        "steps.scan.fan_out.join.require",
        "not one of ['all', 'selected']",
    ),
    "require_missing": (
        _scan({"fan_out": {"over": {"shards": 2}, "join": {}}}),
        19,
        "steps.scan.fan_out.join.require",
        "declares 'require'",
    ),
    "gpus_key": (
        _scan(
            {"fan_out": {"over": {"shards": 2}, "join": {"require": "all"}, "gpus": 4}}
        ),
        19,
        "steps.scan.fan_out",
        "unknown key 'gpus'",
    ),
    "fan_out_not_an_object": (
        _scan({"fan_out": 2}),
        19,
        "steps.scan.fan_out",
        "is an object",
    ),
    "selected_with_no_per_child_conditional": (
        _scan({"fan_out": {"over": {"shards": 2}, "join": {"require": "selected"}}}),
        19,
        "steps.scan.fan_out.join.require",
        "no per_target or per_variable conditional gates",
    ),
    "a_child_as_an_input": (
        lambda raw: (
            raw["steps"]["best"]["inputs"].__setitem__(
                "table", {"step": "scan@0", "file": "iia.json"}
            )
            or raw
        ),
        19,
        "steps.best.inputs.table",
        "'scan@0' is a child of 'scan'; a reference names the join",
    ),
    "a_child_in_after": (
        lambda raw: raw["steps"]["best"].__setitem__("after", ["scan@1"]) or raw,
        19,
        "steps.best.after",
        "is a child of 'scan'",
    ),
    "fan_out_on_a_control": (
        lambda raw: _scan({"control": {"of": "other", "kind": "self_swap"}})(
            _with_other(raw)
        ),
        19,
        "steps.scan.fan_out",
        "declares 'control'",
    ),
    "fan_out_on_a_control_of_target": (
        lambda raw: _with_other(raw, control={"of": "scan", "kind": "self_swap"}),
        19,
        "steps.scan.fan_out",
        "declares a control of",
    ),
    "fan_out_on_a_bundle_saving_document": (
        None,  # built against the tree (`_TREE_CASES`)
        19,
        "steps.scan.fan_out",
        "saves the bundle(s) ['v_cf.safetensors']",
    ),
    "fan_out_on_a_ledger_saving_document": (
        None,  # built against the tree (`_TREE_CASES`)
        19,
        "steps.scan.fan_out",
        "saves the non-value entry(ies) ['location_ledger (ledger.json)']",
    ),
}


@pytest.mark.parametrize("case", sorted(FAN_OUT_REFUSALS))
def test_a_malformed_fan_out_is_refused_naming_the_field(
    tmp_path: Path, case: str
) -> None:
    mutate, rule, path, message = FAN_OUT_REFUSALS[case]
    root = _tree(tmp_path)
    raw = _TREE_CASES[case](root, _raw(root)) if mutate is None else mutate(_raw(root))
    err = _refused(root, raw, rule=rule, path=path)
    assert message in str(err), str(err)


def test_a_ledger_saving_document_loads_unfanned() -> None:
    """The ledger refusal's valid twin: the same
    ledger-saving document loads as an ordinary, unfanned step — the refusal
    is on the fan-out, not on the ledger; the fanned document with ordinary
    per-point tables is T15's fixture."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = _tree(Path(tmp))
        loaded = _load(root, _unfanned(_with_ledger(root, _raw(root))))
        entry = loaded.inner["scan"].compiled.document.save[-1]
        assert entry.kind == "location_ledger"
        assert str(entry.file_path) == "ledger.json"
        assert "scan" not in loaded.children


@pytest.mark.parametrize(
    "file_path",
    [
        *fan_out._SPARSE_FILES,  # pyright: ignore[reportPrivateUsage]
        pytest.param(
            f"tables/{fan_out._SPARSE_FILES[0]}",  # pyright: ignore[reportPrivateUsage]
            id="nested",
        ),
    ],
)
def test_a_fanned_out_document_claiming_a_sparse_side_table_name_is_refused(
    tmp_path: Path, file_path: str
) -> None:
    """The join's density decision keys
    on the three side tables' NAMES, and a document's `save[].file_path` is a
    free string, so a document claiming one for a dense metric table would
    have that table exempted from the missing-file check (and overwritten by
    the engine's side table, `write_outputs`' order). Refused at load under
    rule 19, beside the bundle and ledger refusals, naming the step and the
    file; the name is compared by the path's final component, so a directory
    prefix (`nested`) does not slip it past. Parametrized over
    ``_SPARSE_FILES`` itself — the refusal's name set is the tuple. Twins:
    `test_a_document_saving_any_other_name_fans_out_and_the_claimant_loads_unfanned`."""
    root = _tree(tmp_path)
    raw = _with_save_named(root, _raw(root), file_path)  # the two-shard fixture
    err = _refused(root, raw, path="steps.scan.fan_out")
    message = str(err)
    assert err.rule == 19 and "'fan_out' on 'scan'" in message, message
    assert f"saves ['{file_path}']" in message, message
    assert f"'{Path(file_path).name}'" in message, message
    assert "sparse side table" in message and "cannot claim that name" in message


def test_a_document_saving_any_other_name_fans_out_and_the_claimant_loads_unfanned(
    tmp_path: Path,
) -> None:
    """The valid twins: the same document with the
    table saved under any other name loads fanned and joins, the renamed
    table row for row equal to the child-published one; and the claimant
    document loads as an ordinary, unfanned step — the refusal is on the
    fan-out (the name is the join's), and a non-fanned step keeps the
    pre-existing writer-side hazard, which is the engine's, not this layer's."""
    root = _tree(tmp_path)
    loaded = _load(root, _with_save_named(root, _raw(root), "scores.json"))
    assert loaded.children["scan"] == ("scan@0", "scan@1")
    assert [str(e.file_path) for e in loaded.inner["scan"].compiled.document.save] == [
        "iia.json",
        "scores.json",
    ]
    fanned = _run(loaded, root, tmp_path / "fanned")
    record = _record(fanned.run_root, "scan")
    assert record["status"] == "completed" and "scores.json" in record["files"]
    joined = json.loads((fanned.run_root / "scan" / "scores.json").read_text())
    assert len(joined) == len(loaded.inner["scan"].point_digests) * 2  # two examples
    axes = _axes(loaded, "scan")
    assert {tuple(row[a] for a in axes) for row in joined} == {
        tuple(_plain(point.coords[a]) for a in axes)
        for point in loaded.inner["scan"].expansion.points
    }
    other = _tree(tmp_path / "unfanned")
    unfanned = _load(
        other, _unfanned(_with_save_named(other, _raw(other), "train_eval.json"))
    )
    assert "scan" not in unfanned.children
    assert (
        str(unfanned.inner["scan"].compiled.document.save[-1].file_path)
        == "train_eval.json"
    )


def _per_child(**changes: Any) -> Callable[[Path], dict[str, Any]]:
    def build(root: Path) -> dict[str, Any]:
        return _behavioral_raw(root, **changes)

    return build


def _unfanned_producer(root: Path) -> dict[str, Any]:
    raw = _behavioral_raw(root)
    del raw["steps"]["qualify"]["fan_out"]
    return raw


def _unfanned_gated(root: Path) -> dict[str, Any]:
    raw = _behavioral_raw(root)
    del raw["steps"]["apply"]["fan_out"]
    return raw


def _gated_over_differs(root: Path) -> dict[str, Any]:
    raw = _behavioral_raw(root)
    raw["steps"]["apply"]["fan_out"]["over"] = {"axis": "positions.tap"}
    return raw


def _decision_producer(root: Path) -> dict[str, Any]:
    raw = _gated_raw(root, 0.9)
    raw["steps"]["gate"]["scope"] = "per_target"
    return raw


SCOPE_REFUSALS: dict[str, tuple[Callable[[Path], dict[str, Any]], str, str]] = {
    "producer_not_fanned_out": (
        _unfanned_producer,
        "steps.gate.scope",
        "not a fanned-out behavioral step",
    ),
    "producer_a_decision_step": (
        _decision_producer,
        "steps.gate.scope",
        "not a fanned-out behavioral step",
    ),
    "per_target_over_shards": (
        _per_child(over={"shards": 2}),
        "steps.gate.scope",
        "neither a target nor a variable",
    ),
    "per_variable_over_shards": (
        _per_child(scope="per_variable", over={"shards": 2}),
        "steps.gate.scope",
        "neither a target nor a variable",
    ),
    "per_target_over_a_positions_axis": (
        _per_child(over={"axis": SAID}),
        "steps.gate.scope",
        "per_target is declared over an axis under 'sites.'",
    ),
    "per_variable_over_a_sites_axis": (
        _per_child(scope="per_variable", over={"axis": PROBE_LAYERS}),
        "steps.gate.scope",
        "per_variable over an axis under 'positions.'",
    ),
    "gated_step_not_fanned_out": (
        _unfanned_gated,
        "steps.gate.on_true",
        "declares no fan_out",
    ),
    "gated_step_over_differs": (
        _gated_over_differs,
        "steps.gate.on_true",
        "the two are equal, so verdict i gates child i",
    ),
}


@pytest.mark.parametrize("case", sorted(SCOPE_REFUSALS))
def test_a_per_child_scope_is_held_to_its_producers_fan_out(
    tmp_path: Path, case: str
) -> None:
    build, path, message = SCOPE_REFUSALS[case]
    root = _tree(tmp_path)
    err = _refused(root, build(root), rule=FAN_OUT_RULE, path=path)
    assert message in str(err), str(err)


def _behavioral_over_said(root: Path) -> dict[str, Any]:
    """The per_variable twin: qualify fanned out over the answer position's
    two decode budgets, gating two behavioral steps over the same axis. Each
    child holds one point per probe layer, so every child passes at
    `min_correct_rate` 0.5 and the true side runs whole."""
    raw = _behavioral_raw(
        root, scope="per_variable", over={"axis": SAID}, min_correct_rate=0.5
    )
    for name in ("apply", "narrow"):
        raw["steps"][name] = {
            **raw["steps"]["qualify"],
            "thresholds": {
                "min_examples": 1,
                "min_valid_rate": 0.0,
                "min_correct_rate": 0.0,
            },
            "fan_out": {"over": {"axis": SAID}, "join": {"require": "selected"}},
        }
    return raw


TWINS: dict[str, tuple[Callable[[Path], dict[str, Any]], dict[str, str] | None]] = {
    "the_fixture_two_shards": (lambda root: _raw(root), None),
    "three_shards": (lambda root: _fanned(_raw(root), "scan", {"shards": 3}), None),
    "one_shard_per_point": (
        lambda root: _fanned(_raw(root), "scan", {"shards": 8}),
        None,
    ),
    "axis_layers": (lambda root: _fanned(_raw(root), "scan", {"axis": LAYERS}), None),
    "axis_tap": (
        lambda root: _fanned(_raw(root), "scan", {"axis": "positions.tap"}),
        None,
    ),
    "require_all_spelled_beside_a_control_free_other": (
        lambda root: _with_other(_raw(root)),
        None,
    ),
    "a_global_conditional_over_a_fanned_out_step": (
        lambda root: _gated_raw(root, 0.9),
        {
            "measure": "completed",
            "gate_k": "completed",
            "gate": "completed",
            "scan@0": "completed",
            "scan@1": "completed",
            "scan": "completed",
            "best": "completed",
            "probe": "skipped",
        },
    ),
    "per_target_selected": (
        lambda root: _behavioral_raw(root),
        {
            "qualify@0": "completed",
            "qualify@1": "completed",
            "qualify": "completed",
            "gate@0": "completed",
            "gate@1": "completed",
            "gate": "completed",
            "apply@0": "completed",
            "apply@1": "skipped",
            "apply": "completed",
            "narrow@0": "skipped",
            "narrow@1": "completed",
            "narrow": "completed",
        },
    ),
    "per_target_all_on_the_gated_step": (
        # `all` under a per-child verdict: one skipped child skips the join
        lambda root: _behavioral_raw(root, require="all"),
        {
            "qualify@0": "completed",
            "qualify@1": "completed",
            "qualify": "completed",
            "gate@0": "completed",
            "gate@1": "completed",
            "gate": "completed",
            "apply@0": "completed",
            "apply@1": "skipped",
            "apply": "skipped",
            "narrow@0": "skipped",
            "narrow@1": "completed",
            "narrow": "skipped",
        },
    ),
    "per_variable_selected_over_behavioral_gated_steps": (
        _behavioral_over_said,
        {
            "qualify@0": "completed",
            "qualify@1": "completed",
            "qualify": "completed",
            "gate@0": "completed",
            "gate@1": "completed",
            "gate": "completed",
            "apply@0": "completed",
            "apply@1": "completed",
            "apply": "completed",
            "narrow@0": "skipped",
            "narrow@1": "skipped",
            "narrow": "skipped",
        },
    ),
}


@pytest.mark.parametrize("twin", sorted(TWINS))
def test_the_valid_twin_loads_and_runs_to_completed(tmp_path: Path, twin: str) -> None:
    """Every refusal has a twin that passes — each
    `over` form, each `require`, each scope over its axis root, a receipt on
    a fanned-out behavioral join, a global conditional over a fanned-out
    step — and each runs on the CPU stub to its expected statuses."""
    build, expected = TWINS[twin]
    root = _tree(tmp_path)
    loaded = _load(root, build(root))
    result = _run(loaded, root, tmp_path / "runs")
    statuses = _statuses(result)
    if expected is None:
        assert set(statuses.values()) == {"completed"}, statuses
        assert set(statuses) == set(loaded.order)
    else:
        assert statuses == expected
    assert terminal(result.run_root / EVENTS_FILE)
    for name, entry in result.manifest["steps"].items():
        step = loaded.document.steps[name]
        if (
            entry["status"] == "completed"
            and name in loaded.children
            and isinstance(step, (ProtocolStep, BehavioralStep))
        ):
            record = _record(result.run_root, name)
            assert "fan_out" in record and "join" in record, name


def test_every_over_form_every_policy_and_every_scope_has_a_twin() -> None:
    joined = " ".join(TWINS)
    for word in (
        "shards",
        "axis_layers",
        "axis_tap",
        "per_target",
        "per_variable",
        "all",
        "selected",
    ):
        assert word in joined, word


# --------------------------------------------------------------------------- #
# the pure grammar, the receipt's key table, and explain
# --------------------------------------------------------------------------- #


def test_parse_fan_out_is_the_grammar() -> None:
    assert fan_out.parse_fan_out(
        {"over": {"shards": 2}, "join": {"require": "all"}}, "steps.x.fan_out"
    ) == {"over": {"shards": 2}, "join": {"require": "all"}}
    assert fan_out.parse_fan_out(
        {"over": {"axis": LAYERS}, "join": {"require": "selected"}}, "steps.x.fan_out"
    ) == {"over": {"axis": LAYERS}, "join": {"require": "selected"}}
    with pytest.raises(WorkflowError) as err:
        fan_out.parse_fan_out({"over": {"shards": 2}}, "steps.x.fan_out")
    assert err.value.rule == 19 and "no default is materialized" in str(err.value)


def test_the_receipt_keys_are_the_spec_table(tmp_path: Path) -> None:
    """§2.9's receipt table names every key of the join's record and no
    other; `engine` and `execution` are absent by design."""
    root = _tree(tmp_path)
    result = _run(_load(root), root, tmp_path / "runs")
    record = _record(result.run_root, "scan")
    documented = _members(SECTION_29, "key")
    # `decision` sits on a behavioral join alone; every other key on any join
    assert set(documented) - {"decision"} <= set(record), sorted(
        set(documented) - set(record)
    )
    assert "decision" in documented
    assert {"engine", "execution"}.isdisjoint(record)
    assert set(record["fan_out"]) == {"over", "width", "children"}
    assert set(record["join"]) == {
        "require",
        "consumed",
        "n_points",
        "n_missing",
        "n_duplicate",
    }


def test_fan_out_enters_the_entry_only_when_authored(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    fanned = _load(root)
    plain = _load(root, _unfanned(_raw(root)))
    assert "fan_out" in fanned.canonical["steps"]["scan"]
    assert "fan_out" not in plain.canonical["steps"]["scan"]
    assert set(plain.canonical["steps"]["scan"]) == {
        "type",
        "document",
        "document_digest",
    }
    assert fanned.canonical["steps"]["best"] == plain.canonical["steps"]["best"]


def test_explain_prints_the_fan_out(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _tree(tmp_path)
    code = cli_main(
        [
            "explain",
            "--engine",
            "auto",
            str(root / "scan_wf.json"),
            "--data-root",
            str(PROTOCOL_DATA),
            "--artifacts-root",
            str(root),
        ]
    )
    out = capsys.readouterr().out
    assert code == 0, out
    assert "schedule  3 levels" in out
    assert "level 0: scan@0, scan@1" in out
    assert "fan-out over 2 shards → 2 children, join all" in out
    assert "scan@0: intervention_protocol protocols/scan.json — 8 point(s)" in out
    assert "shard 0 of 2: 4 point(s)" in out
    assert "shard 1 of 2: 4 point(s)" in out
    assert (
        fan_out.describe(
            _load(root, _fanned(_raw(root), "scan", {"axis": LAYERS})).document.steps[
                "scan"
            ],
            ("a", "b", "c", "d"),
        )
        == f"fan-out over {LAYERS} → 4 children, join all"
    )
