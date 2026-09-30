"""Nested reusable workflows (``docs/workflow_protocol.md`` §2.10, §5 rule 20;
T19–T22 and the censuses) — on CPU, with the conditional fixtures' two
scripts and the fan-out fixtures' stub engine.

* **The censuses.** Six step kinds; `MAX_RULE` is 20 and §5 numbers it; the
  §2.10 field table is `WorkflowStep`'s fields; `causalab/workflow/nested.py`
  is in no hashed closure, reaches no engine module, and loading a nested
  workflow imports no torch; `/` is outside rule 3's alphabet.
* **T19 — digested by reference.** A nested workflow's digest enters the outer
  digest by reference: editing the inner document's canonical content moves
  the outer digest and `workflow_digest` equals the inner's own; a
  non-canonical byte moves nothing; the inner steps are never entries of the
  outer; every shipped and demo workflow loads with no §2.10 key and both
  pins hold from the pin file (T13's third run).
* **T20 — self-inclusion and depth.** A document including itself, directly
  or through a chain, is refused naming the chain; three levels load, run and
  `explain` as three indents; an intervention specification is not a
  workflow document.
* **T21 — one namespace, no shadowing.** An outer and an inner step of one
  local name both publish, each under its own root; an inner reference
  resolves to the inner's own step; a reference to the `workflow` step is
  refused; an inner document's artifact reference derives its edge to the
  inner step by longest prefix.
* **T22 — receipts, skips, resume across the boundary.** A failed and a
  missing receipt on an inner decision are two distinct `W18` refusals before
  any engine; an inner receipt is checked under its sub-root; a receipt on the
  `workflow` step gates every inner step; an outer conditional skips the whole
  nested workflow, each inner step its own `skipped` entry on one stream with
  one terminal line; `--resume` reuses every inner step and re-runs exactly
  what an inner edit moved; the existing chains are unchanged; an inner
  fan-out joins under its sub-root.
* **Refusals, each beside its valid twin**, every one
  naming the field.
"""

from __future__ import annotations

import dataclasses
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import pytest

from causalab.cli import main as cli_main
from causalab.io.events import EVENTS_FILE, read_events, terminal
from causalab.io.step_record import SIDECAR
from causalab.protocol.identity import import_closure
from causalab.io.env import FileArtifacts, FileDatasets, ResolutionEnv
from causalab.tasks import TASKS_ROOT
from causalab.workflow import conditional as cond
from causalab.workflow import manifest as mf
from causalab.workflow import nested
from causalab.workflow import runner
from causalab.workflow.behavioral import DECISION_FILE
from causalab.workflow.conditional import CONDITIONAL_RULE
from causalab.workflow.document import (
    CONTROL_RULE,
    MAX_RULE,
    REQUIRED_CONTROL_KINDS,
    STEP_TYPES,
    ConditionalStep,
    LoadedWorkflow,
    ProtocolStep,
    WorkflowError,
    WorkflowStep,
    load_workflow,
    producer_of,
)
from causalab.workflow.fan_out import FAN_OUT_RULE
from causalab.workflow.nested import NESTED_RULE, SEPARATOR
from causalab.workflow.runner import run_workflow
from tests.protocol.test_vocabulary_census import CODE
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
    _run as _chain_run,  # pyright: ignore[reportPrivateUsage]
    _sentinels,  # pyright: ignore[reportPrivateUsage]
    _snapshot,  # pyright: ignore[reportPrivateUsage]
    _statuses,  # pyright: ignore[reportPrivateUsage]
    _Stub,  # pyright: ignore[reportPrivateUsage]
    _tables,  # pyright: ignore[reportPrivateUsage]
    _tree as _chain_tree,  # pyright: ignore[reportPrivateUsage]
)
from tests.workflow.test_controls import (
    EXTERNAL,
    PROTOCOLS,
    _protocol,  # pyright: ignore[reportPrivateUsage]
    _section,  # pyright: ignore[reportPrivateUsage]
    _workflow,  # pyright: ignore[reportPrivateUsage]
)
from tests.workflow.test_fan_out import (
    FIXTURES as FAN_OUT_FIXTURES,
)
from tests.workflow.test_fan_out import (
    PROTOCOL_DATA,
    _behavioral_raw,  # pyright: ignore[reportPrivateUsage]
    _load as _fan_load,  # pyright: ignore[reportPrivateUsage]
    _Rows,  # pyright: ignore[reportPrivateUsage]
    _tree as _fan_tree,  # pyright: ignore[reportPrivateUsage]
)

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "docs" / "workflow_protocol.md"
FIXTURES = Path(__file__).parent / "fixtures" / "nested"
CONDITIONAL_FIXTURES = Path(__file__).parent / "fixtures" / "conditional"
BEHAVIORAL_FIXTURES = Path(__file__).parent / "fixtures" / "behavioral"
MODULE = "causalab/workflow/nested.py"
SECTION_210 = "### 2.10 `workflow` — a nested reusable workflow"
#: the keys a `workflow` entry carries and no other entry may (§7), and the
#: manifest's record-only map
NESTED_KEYS = ("workflow_digest", "nested")
#: the fixture's work steps, in the derived order
OUTER = ("measure", "tail/measure", "tail/best", "report", "tail/gate_k")
#: an intervention specification the stub engine serves (the fan-out fixture's)
SCAN = "protocols/scan.json"


# --------------------------------------------------------------------------- #
# the fixture tree
# --------------------------------------------------------------------------- #


def _tree(tmp: Path, *, score: float = 0.9, outer_score: float = 0.3) -> Path:
    """A private copy of the nested fixtures with the conditional fixtures'
    two scripts beside each document that names them, the fan-out fixture's
    intervention specifications (for a `set` on an inner document step and an
    inner fan-out) and the behavioral fixture's (for a receipt-gated protocol
    step), and two measurement files — the inner's and the outer's — as
    absolute paths, so two runs of one document differ in nothing but what
    was measured."""
    root = tmp / "nested"
    shutil.copytree(FIXTURES, root)
    shutil.copytree(CONDITIONAL_FIXTURES / "scripts", root / "scripts")
    shutil.copytree(CONDITIONAL_FIXTURES / "scripts", root / "deep" / "scripts")
    shutil.copytree(FAN_OUT_FIXTURES / "protocols", root / "protocols")
    shutil.copytree(
        BEHAVIORAL_FIXTURES / "protocols", root / "protocols", dirs_exist_ok=True
    )
    shutil.copytree(BEHAVIORAL_FIXTURES / "qa", root / "qa")
    _measure(root, score)
    _outer_measure(root, outer_score)
    for name, file in (
        ("tail.json", "measurement.json"),
        ("outer.json", "outer_measurement.json"),
    ):
        raw = json.loads((root / name).read_text())
        raw["steps"]["measure"]["inputs"]["measurement"] = {"path": str(root / file)}
        (root / name).write_text(json.dumps(raw, indent=2))
    return root


def _measure(root: Path, score: float) -> None:
    (root / "measurement.json").write_text(json.dumps({"score": score}))


def _outer_measure(root: Path, score: float) -> None:
    (root / "outer_measurement.json").write_text(json.dumps({"score": score}))


def _env(root: Path) -> ResolutionEnv:
    return ResolutionEnv(
        datasets=FileDatasets(root=root, fallback_roots=(PROTOCOL_DATA, TASKS_ROOT)),
        artifacts=FileArtifacts(root=root),
    )


def _raw(root: Path, name: str = "outer.json") -> dict[str, Any]:
    return json.loads((root / name).read_text())


def _write(root: Path, name: str, raw: dict[str, Any]) -> Path:
    (root / name).write_text(json.dumps(raw, indent=2))
    return root / name


def _load(
    root: Path, raw: dict[str, Any] | None = None, *, name: str = "outer.json"
) -> LoadedWorkflow:
    return load_workflow(
        raw if raw is not None else root / name, _env(root), workflow_dir=root
    )


def _run(
    loaded: LoadedWorkflow,
    root: Path,
    out: Path,
    engine: Any = None,
    **kw: Any,
) -> Any:
    return run_workflow(loaded, _env(root), out, engine, **kw)


def _refused(
    root: Path, raw: dict[str, Any], *, rule: int = NESTED_RULE, path: str
) -> WorkflowError:
    with pytest.raises(WorkflowError) as err:
        _load(root, raw)
    assert err.value.rule == rule, str(err.value)
    assert path in str(err.value), str(err.value)
    return err.value


def _tail(root: Path, **steps: Any) -> dict[str, Any]:
    """Edit the inner document on disk: add or replace the named steps."""
    raw = _raw(root, "tail.json")
    for name, step in steps.items():
        if step is None:
            raw["steps"].pop(name, None)
        else:
            raw["steps"][name] = step
    _write(root, "tail.json", raw)
    return raw


def _record(run_root: Path, name: str) -> dict[str, Any]:
    return json.loads((run_root / name / SIDECAR).read_text())


def _values(run_root: Path, name: str, file: str) -> dict[str, Any]:
    return json.loads((run_root / name / file).read_text())


def _writer(file: str, **extra: Any) -> dict[str, Any]:
    return {
        "type": "script",
        "script": {"path": "scripts/writer.py"},
        "inputs": {},
        "outputs": {"out": {"file": file, "keys": {"ran": True}}},
        **extra,
    }


def _outer_decision(raw: dict[str, Any]) -> dict[str, Any]:
    """A decision over the outer `measure` (score 0.3 by default: `fail`)."""
    raw["steps"]["gate_k0"] = {
        "type": "decision",
        "values": {"step": "measure", "file": "values.json"},
        "rule": {"score": {"ge": 0.5}},
        "decision": {"on_pass": "advance", "on_fail": "narrow"},
    }
    return raw


def _gated(
    raw: dict[str, Any], on_true: list[str], on_false: list[str]
) -> dict[str, Any]:
    raw = _outer_decision(raw)
    raw["steps"]["gate"] = {
        "type": "conditional",
        "predicate": {
            "decision": {"step": "gate_k0"},
            "field": "outcome",
            "eq": "pass",
        },
        "on_true": on_true,
        "on_false": on_false,
        "scope": "global",
    }
    return raw


def _scan_inner(root: Path, fan_out: dict[str, Any] | None = None) -> Path:
    """An inner workflow the stub engine serves: the fan-out fixture's scan
    (eight points on two axes) and the shipped `select` over its table."""
    scan: dict[str, Any] = {"type": "intervention_protocol", "document": SCAN}
    if fan_out is not None:
        scan["fan_out"] = fan_out
    return _write(
        root,
        "inner_scan.json",
        {
            "version": "1",
            "output_dir": "scan_alone",
            "steps": {
                "scan": scan,
                "best": {
                    "type": "script",
                    "script": {"module": "causalab.workflow.scripts.select"},
                    "inputs": {
                        "table": {"step": "scan", "file": "iia.json"},
                        "value": "value",
                        "emit": {"best_layer": "sites.target.layers"},
                    },
                    "outputs": {
                        "values": {"file": "values.json", "keys": {"best_layer": 0}}
                    },
                },
            },
        },
    )


def _scan_outer(root: Path, fan_out: dict[str, Any] | None = None) -> dict[str, Any]:
    _scan_inner(root, fan_out)
    return {
        "version": "1",
        "output_dir": "nested_scan",
        "steps": {
            "tail": {"type": "workflow", "document": "inner_scan.json"},
            "report": _writer(
                "report.json",
                inputs={
                    "k": {
                        "step": "tail/best",
                        "file": "values.json",
                        "key": "best_layer",
                    }
                },
            ),
        },
    }


# --------------------------------------------------------------------------- #
# the censuses
# --------------------------------------------------------------------------- #


def test_six_step_kinds_and_rule_20_is_the_nested_rule_and_the_last() -> None:
    assert STEP_TYPES == (
        "intervention_protocol",
        "script",
        "behavioral",
        "decision",
        "conditional",
        "workflow",
    )
    # the rules are numbered in landing order: 15 qualify once, 16 site
    # equivalence, 17 behavioral, 18 conditional, 19 fan-out, 20 this one —
    # each pinned exactly by its own suite,
    # the ceiling only bounded here so a later rule raises it without an edit
    assert NESTED_RULE == 20 and MAX_RULE >= 20 and FAN_OUT_RULE == 19
    section = _section("## 5. Validation")
    items = {
        int(number): text
        for number, text in re.findall(
            r"^(\d+)\. (.+?)(?=^\d+\. |\Z)", section, re.M | re.S
        )
    }
    assert max(items) == MAX_RULE
    for number, word in (
        (15, "runs before its target"),
        (16, "site-equivalent"),
        (17, "behavioral"),
        (18, "conditional"),
        (19, "fan-out"),
    ):
        assert word in items[number], (number, word)
    for word in (
        "`workflow`",
        "§2.10",
        "`set`",
        "`<step>/<inner>`",
        "includes itself",
        "`control`",
        "`waive`",
        "`fan_out`",
        "`workflow_digest`",
        "names the field",
    ):
        assert word in items[20], word
    assert (
        "`intervention_protocol · script · behavioral · decision · conditional · workflow`"
        in _section("### 2.1 `steps` — common fields")
    )


def test_the_field_table_is_the_workflow_step() -> None:
    """§2.10's one field table (a plain-word first header cell) lists exactly
    the authored fields of `WorkflowStep` — `type` and `description` are
    every step's (§2.1)."""
    tables = _tables(SECTION_210, "field")
    assert len(tables) == 1
    documented = [CODE.findall(row[0])[0] for row in tables[0]]
    authored = [
        f.name
        for f in dataclasses.fields(WorkflowStep)
        if f.name not in ("type", "description")
    ]
    assert documented == authored == ["document", "set", "requires_receipt", "after"]


def test_nested_is_in_no_hashed_closure() -> None:
    assert MODULE not in SHARED
    for module in (*CLOSURES, REDUCE):
        assert MODULE not in _closure(module), module


def test_nested_reaches_no_engine_module() -> None:
    """Engine-free: the one member under ``neural/`` is the torch-free
    enumerator the workflow layer reads the steps through
    (``neural/shared/sweep.py``), never an engine."""
    members = import_closure(REPO / MODULE, root=REPO)
    assert members, "the closure walk found nothing"
    assert [m for m in members if m.startswith("causalab/neural/")] == [
        "causalab/neural/shared/sweep.py"
    ]


_PROBE = """
import json, sys
from pathlib import Path
import causalab.workflow.nested
from causalab.io.env import FileArtifacts, FileDatasets, ResolutionEnv
from causalab.workflow.document import load_workflow
root = Path(sys.argv[1])
env = ResolutionEnv(datasets=FileDatasets(root=root), artifacts=FileArtifacts(root=root))
loaded = load_workflow(root / "outer.json", env)
print(json.dumps({"digest": loaded.digest, "torch": "torch" in sys.modules,
                  "order": list(loaded.order)}))
"""


def test_loading_a_nested_workflow_imports_no_torch(tmp_path: Path) -> None:
    """A subprocess, as in test_load_is_torch_free.py: conftest has already
    imported torch here."""
    root = _tree(tmp_path)
    completed = subprocess.run(
        [sys.executable, "-c", _PROBE, str(root)],
        capture_output=True,
        text=True,
        cwd=str(REPO),
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    assert result["torch"] is False
    assert result["digest"] == _load(root).digest
    assert tuple(result["order"]) == OUTER


def test_the_separator_is_outside_the_authored_alphabet(tmp_path: Path) -> None:
    """Rule 3 refuses an authored `/`, so a flattened name can never collide
    with an authored step and needs no collision check (§1, §2.10)."""
    assert SEPARATOR == "/"
    assert nested.qualified("tail", "best") == "tail/best"
    assert nested.qualified("", "best") == "best"
    assert nested.local_name("tail", "tail/fit@0") == "fit@0"
    assert nested.local_name("tail", "measure") is None
    assert producer_of("tail/best/values.json", ("tail", "tail/best")) == "tail/best"
    assert producer_of("tailor/x.json", ("tail", "tail/best")) is None
    root = _tree(tmp_path)
    raw = _raw(root)
    raw["steps"]["tail/measure"] = raw["steps"].pop("report")
    with pytest.raises(WorkflowError) as err:
        _load(root, raw)
    assert err.value.rule == 3, str(err.value)


# --------------------------------------------------------------------------- #
# T19 — digested by reference
# --------------------------------------------------------------------------- #


def test_t19_the_inner_digest_enters_the_outer_by_reference(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    before = _load(root)
    inner = _load(root, name="tail.json")
    entry = before.canonical["steps"]["tail"]
    assert entry["workflow_digest"] == inner.digest == before.nested["tail"].digest
    assert before.inner_digest_kind["tail"] == "workflow"
    # a canonical byte of the inner moves the outer, and only through the one key
    raw = _raw(root, "tail.json")
    raw["steps"]["gate_k"]["rule"]["score"] = {"ge": 0.6}
    _write(root, "tail.json", raw)
    moved = _load(root)
    assert moved.digest != before.digest
    assert (
        moved.canonical["steps"]["tail"]["workflow_digest"]
        == _load(root, name="tail.json").digest
    )
    assert (
        moved.canonical["steps"]["tail"]["workflow_digest"] != entry["workflow_digest"]
    )
    for name in ("measure", "report"):
        assert moved.canonical["steps"][name] == before.canonical["steps"][name]
    assert moved.step_digests["tail/gate_k"] != before.step_digests["tail/gate_k"]
    assert moved.step_digests["tail/measure"] == before.step_digests["tail/measure"]
    # a non-canonical byte moves nothing: the inner's output_dir, whitespace
    raw["steps"]["gate_k"]["rule"]["score"] = {"ge": 0.5}
    raw["output_dir"] = "elsewhere"
    (root / "tail.json").write_text(json.dumps(raw))  # no indent either
    assert _load(root).digest == before.digest
    assert _load(root).canonical == before.canonical
    # a workflow's description is canonical (§7), so the inner's moves the outer
    raw["description"] = "reworded"
    _write(root, "tail.json", raw)
    assert _load(root).digest != before.digest


def test_t19_the_inner_steps_are_not_canonical_entries(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    loaded = _load(root)
    assert set(loaded.canonical["steps"]) == {"measure", "tail", "report"}
    entry = loaded.canonical["steps"]["tail"]
    assert set(entry) == {
        "type",
        "document",
        "workflow_digest",
    }  # set/after only when authored
    assert entry["type"] == "workflow" and entry["document"] == "tail.json"
    assert len(entry["workflow_digest"]) == 64
    for name, other in loaded.canonical["steps"].items():
        if name != "tail":
            assert not set(other) & set(NESTED_KEYS), name
    # the flattened table, order and identities
    assert loaded.order == OUTER
    assert set(loaded.document.steps) == {*OUTER, "tail"}
    assert isinstance(loaded.document.steps["tail"], WorkflowStep)
    assert loaded.dependencies["report"] == ("tail/best",)
    assert loaded.dependencies["tail/best"] == ("tail/measure",)
    assert loaded.dependencies["tail"] == ()  # a container's own edges: none authored
    assert {"tail", "tail/measure", "tail/gate_k", "tail/best"} <= set(
        loaded.step_digests
    )
    assert (
        loaded.step_digests["tail/best"] == loaded.nested["tail"].step_digests["best"]
    )
    assert loaded.levels == (
        ("measure", "tail/measure"),
        ("tail/best", "tail/gate_k"),
        ("report",),
    )
    # `after` and `set` enter the entry only when authored
    raw = _raw(root)
    raw["steps"]["tail"]["after"] = ["measure"]
    with_after = _load(root, raw)
    assert with_after.canonical["steps"]["tail"]["after"] == ["measure"]
    assert with_after.dependencies["tail"] == ("measure",)
    assert (
        "measure" in with_after.dependencies["tail/measure"]
    )  # inherited by the inner roots
    assert with_after.digest != loaded.digest


def test_t19_two_workflow_steps_over_one_document(tmp_path: Path) -> None:
    """Two nestings of one document with different `set` are
    two loads and two digests; with identical `set` they share a digest but
    never a directory."""
    root = _tree(tmp_path)
    raw = _scan_outer(root)
    narrow = {"scan": {"sites.target.layers": {"sweep": [0, 1]}}}
    raw["steps"]["tail2"] = {
        "type": "workflow",
        "document": "inner_scan.json",
        "set": narrow,
    }
    raw["steps"]["tail3"] = {
        "type": "workflow",
        "document": "inner_scan.json",
        "set": narrow,
    }
    loaded = _load(root, raw)
    entries = loaded.canonical["steps"]
    assert entries["tail"]["workflow_digest"] != entries["tail2"]["workflow_digest"]
    assert entries["tail2"]["workflow_digest"] == entries["tail3"]["workflow_digest"]
    assert entries["tail2"]["set"] == narrow and "set" not in entries["tail"]
    assert (
        loaded.step_digests["tail2"] == loaded.step_digests["tail3"]
    )  # the names differ
    result = _run(loaded, root, tmp_path / "runs", _Rows())
    assert set(_statuses(result).values()) == {"completed"}
    assert _record(result.run_root, "tail/scan")["points"] == 8
    assert _record(result.run_root, "tail2/scan")["points"] == 4
    assert _record(result.run_root, "tail3/scan")["points"] == 4
    for name in ("tail", "tail2", "tail3"):
        assert (result.run_root / name / "scan" / "iia.json").is_file()
        assert (result.run_root / name / "best" / "values.json").is_file()


@pytest.mark.parametrize(
    "path", SHIPPED + DEMO_WORKFLOWS, ids=[p.stem for p in SHIPPED + DEMO_WORKFLOWS]
)
def test_t19_every_existing_workflow_loads_with_no_nested_key(
    path: Path, env: Any
) -> None:
    """T13's third run: a §2.10 key on any entry would move every workflow
    digest in the repo (the mutation: emit `workflow_digest` on every arm)."""
    loaded = load_workflow(path, env if path.parent == WORKFLOWS else _demo_env(path))
    assert loaded.nested == {}
    for name, entry in loaded.canonical["steps"].items():
        assert entry["type"] in ("intervention_protocol", "script"), name
        assert not set(entry) & set(NESTED_KEYS), (name, sorted(entry))
        assert SEPARATOR not in name
    assert "workflow" not in loaded.inner_digest_kind.values()


# --------------------------------------------------------------------------- #
# T20 — self-inclusion and depth
# --------------------------------------------------------------------------- #


def _loading(root: Path, name: str) -> WorkflowError:
    """Load `name`, expecting a refusal — and turning a load that never
    returns (the mutation: the chain check dropped) into a visible red, not a hang."""
    sys.setrecursionlimit(600)
    try:
        with pytest.raises(WorkflowError) as err:
            _load(root, name=name)
    except RecursionError:
        pytest.fail(
            "the recursive load never returned — a document including itself was not refused"
        )
    finally:
        sys.setrecursionlimit(1000)
    return err.value


def test_t20_a_document_including_itself_through_a_chain_is_refused(
    tmp_path: Path,
) -> None:
    root = _tree(tmp_path)
    _write(
        root,
        "loop_a.json",
        {
            "version": "1",
            "output_dir": "a",
            "steps": {"x": {"type": "workflow", "document": "loop_b.json"}},
        },
    )
    _write(
        root,
        "loop_b.json",
        {
            "version": "1",
            "output_dir": "b",
            "steps": {"y": {"type": "workflow", "document": "loop_a.json"}},
        },
    )
    err = _loading(root, "loop_a.json")
    assert err.rule == NESTED_RULE
    message = str(err)
    assert "includes itself" in message
    assert "loop_a.json -> loop_b.json -> loop_a.json" in message
    assert "steps.y.document" in message  # the field, at the level that closed the loop


def test_t20_a_document_including_itself_directly_is_refused(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    _write(
        root,
        "self.json",
        {
            "version": "1",
            "output_dir": "s",
            "steps": {"me": {"type": "workflow", "document": "self.json"}},
        },
    )
    err = _loading(root, "self.json")
    assert err.rule == NESTED_RULE
    assert "self.json -> self.json" in str(err) and "steps.me.document" in str(err)


def test_t20_three_levels_load_run_and_explain(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _tree(tmp_path)
    loaded = load_workflow(root / "deep" / "a.json", _env(root))
    assert loaded.order == ("x/y/z", "done")
    assert loaded.levels == (("x/y/z",), ("done",))
    assert loaded.dependencies["done"] == ("x/y/z",)
    assert set(loaded.nested) == {"x"} and set(loaded.nested["x"].nested) == {"y"}
    owner, local, rel = nested.locate(loaded, "x/y/z")
    assert owner is loaded.nested["x"].nested["y"] and (local, rel) == ("z", "x/y")
    assert [c[1:] for c in nested.containers(loaded, "x/y/z")] == [
        ("x", ""),
        ("y", "x"),
    ]
    assert nested.sub_roots(loaded) == ["x/y", "x"]
    assert (
        loaded.canonical["steps"]["x"]["workflow_digest"] == loaded.nested["x"].digest
    )
    assert (
        loaded.nested["x"].canonical["steps"]["y"]["workflow_digest"]
        == loaded.nested["x"].nested["y"].digest
    )
    result = _run(loaded, root, tmp_path / "runs")
    assert _statuses(result) == {"x/y/z": "completed", "done": "completed"}
    assert (result.run_root / "x" / "y" / "z" / "z.json").is_file()
    assert (result.run_root / "done" / "done.json").is_file()
    assert result.manifest["nested"] == {
        "x": {
            "document": "b.json",
            "workflow_digest": loaded.nested["x"].digest,
            "steps": ["x/y/z"],
        },
        "x/y": {
            "document": "c.json",
            "workflow_digest": loaded.nested["x"].nested["y"].digest,
            "steps": ["x/y/z"],
        },
    }
    assert not (result.run_root / mf.ATTEMPTS_DIR).exists()
    assert not (result.run_root / "x" / mf.ATTEMPTS_DIR).exists()
    assert not (result.run_root / "x" / "y" / mf.ATTEMPTS_DIR).exists()
    assert (
        cli_main(
            [
                "explain",
                "--engine",
                "auto",
                str(root / "deep" / "a.json"),
                "--data-root",
                str(root),
                "--artifacts-root",
                str(root),
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "  level 0: x/y/z" in out
    # no whole-workflow digest is printed (§7): the nesting identity the outer
    # entry carries is the inner steps' identities folded, and those are shown
    assert "\n  x: workflow b.json — 1 step(s)\n" in out
    assert "\n    y: workflow c.json — 1 step(s)\n" in out
    assert "workflow digest" not in out
    assert "\n      z: script scripts/writer.py -> z.json" in out
    assert "\n  done: script scripts/writer.py -> done.json" in out


def test_t20_an_intervention_specification_is_not_a_workflow_document(
    tmp_path: Path,
) -> None:
    root = _tree(tmp_path)
    raw = _raw(root)
    raw["steps"]["tail"]["document"] = SCAN
    err = _refused(root, raw, path="steps.tail.document")
    assert "no 'steps' section" in str(err) and "intervention_protocol" in str(err)


# --------------------------------------------------------------------------- #
# T21 — one namespace, no shadowing
# --------------------------------------------------------------------------- #


def test_t21_an_outer_and_an_inner_step_share_a_local_name(tmp_path: Path) -> None:
    """`measure` outside, `tail/measure` inside: both run, each publishes to
    its own directory, and the inner reference to `measure` reads the inner's
    own values (the mutation: the sub-root removed would read the outer's 0.3)."""
    root = _tree(tmp_path)
    loaded = _load(root)
    result = _run(loaded, root, tmp_path / "runs")
    run_root = result.run_root
    assert _statuses(result) == {name: "completed" for name in OUTER}
    assert _values(run_root, "measure", "values.json")["score"] == 0.3
    assert _values(run_root, "tail/measure", "values.json")["score"] == 0.9
    assert (
        _values(run_root, "tail/best", "best.json")["k"] == 0.9
    )  # the inner's own measure
    assert _values(run_root, "report", "report.json")["k"] == 0.9  # through tail/best
    assert (
        _record(run_root, "tail/measure")["identity"]
        == loaded.step_digests["tail/measure"]
    )
    assert _record(run_root, "tail/best")["inputs"] == {
        "k": "measure/values.json"
    }  # as authored
    assert terminal(run_root / EVENTS_FILE)
    steps_on_stream = [
        r["payload"].get("step") for r in read_events(run_root / EVENTS_FILE)
    ]
    assert set(steps_on_stream) - {None} == set(OUTER)  # never a bare inner-local name
    assert (
        sum(
            r["event"] == "campaign_terminal"
            for r in read_events(run_root / EVENTS_FILE)
        )
        == 1
    )


def test_t21_a_reference_to_the_workflow_step_is_refused(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    raw = _raw(root)
    raw["steps"]["report"]["inputs"]["k"] = {
        "step": "tail",
        "file": "best.json",
        "key": "k",
    }
    err = _refused(root, raw, path="steps.report.inputs.k")
    assert "publishes no file" in str(err) and "tail/<step>" in str(err)


def test_t21_an_inner_documents_artifact_reference_derives_the_edge_to_the_inner_step(
    tmp_path: Path,
) -> None:
    """An outer intervention specification reads the inner values object by
    the IM grammar (`artifact: "tail/measure"`, the step whose `values.json`
    it is): the edge is to `tail/measure` (the longest step name), never to
    `tail`, and the representative is the inner script's declared key (the
    mutation: the head alone finds no `(tail, k)` representative)."""
    root = _tree(tmp_path)
    raw = _raw(root)
    raw["steps"]["probe_doc"] = {
        "type": "intervention_protocol",
        "document": SCAN,
        "set": {"sites.target.layers": {"artifact": "tail/measure", "key": "k"}},
    }
    loaded = _load(root, raw)
    assert loaded.dependencies["probe_doc"] == ("tail/measure",)
    assert loaded.inner_digest_kind["probe_doc"] == "authored"
    result = _run(loaded, root, tmp_path / "runs", _Rows())
    assert _statuses(result)["probe_doc"] == "completed"
    # `k` is 8: one layer, two positions — the unoverridden scan has eight points
    assert _record(result.run_root, "probe_doc")["points"] == 2
    assert _record(result.run_root, "probe_doc")["axes"] == ["positions.tap"]


def test_t21_an_unknown_inner_step_is_refused_under_rule_4(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    raw = _raw(root)
    raw["steps"]["report"]["inputs"]["k"] = {
        "step": "tail/nope",
        "file": "x.json",
        "key": "k",
    }
    err = _refused(root, raw, rule=4, path="steps.report.inputs.k")
    assert "unknown step 'tail/nope'" in str(err)


def test_t21_a_child_of_an_inner_fan_out_is_addressed_through_its_join(
    tmp_path: Path,
) -> None:
    root = _tree(tmp_path)
    raw = _scan_outer(root, {"over": {"shards": 2}, "join": {"require": "all"}})
    raw["steps"]["report"]["inputs"]["k"] = {"step": "tail/scan@0", "file": "iia.json"}
    err = _refused(root, raw, rule=FAN_OUT_RULE, path="steps.report.inputs.k")
    assert "'tail/scan@0' is a child of 'tail/scan'" in str(err)


# --------------------------------------------------------------------------- #
# T22 — receipts, skips and resume across the boundary
# --------------------------------------------------------------------------- #


def _receipt_raw(root: Path, outcome: str = "pass") -> dict[str, Any]:
    """The outer with a real protocol step (gpt2 on paper, torch-free at load)
    requiring the inner decision's receipt."""
    raw = _raw(root)
    raw["steps"]["apply"] = {
        "type": "intervention_protocol",
        "document": "protocols/qa_probe.json",
        "set": {"data.base.dataset": "qa/data#development"},
        "requires_receipt": {"step": "tail/gate_k", "outcome": outcome},
    }
    return raw


def test_t22_a_failed_inner_receipt_prevents_allocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _tree(tmp_path, score=0.1)  # tail/gate_k: fail
    loaded = _load(root, _receipt_raw(root))
    assert loaded.dependencies["apply"] == ("tail/gate_k",)
    _sentinels(monkeypatch)
    with pytest.raises(WorkflowError) as err:
        _run(loaded, root, tmp_path / "runs", _Stub())
    assert err.value.rule == CONDITIONAL_RULE
    message = str(err.value)
    assert "steps.apply.requires_receipt" in message and "'tail/gate_k'" in message
    assert "carries outcome 'fail'" in message and "not allocated" in message
    run_root = tmp_path / "runs" / "nested"
    manifest = json.loads((run_root / mf.MANIFEST).read_text())["steps"]
    assert manifest["tail/gate_k"]["status"] == "completed"
    assert manifest["apply"]["status"] == "failed"
    assert "W18" in manifest["apply"]["error"]["message"]
    assert not (run_root / mf.ATTEMPTS_DIR / "apply").exists()
    assert not (run_root / "apply").exists()
    assert (run_root / "tail" / "gate_k" / DECISION_FILE).is_file()


def test_t22_a_skipped_inner_producer_skips_the_receipt_bearing_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An inner conditional skipped `tail/gate_k`: its record names the
    inner-local `gate_k` and the runner folds it under the sub-root, so the
    outer `apply` — after `tail/gate_k` by its receipt edge — is skipped with
    it (§2.8's transitive rule), never allocated (m22b would leave
    `tail/gate_k` running and publishing its receipt, and `apply` reaching the
    engine)."""
    root = _tree(tmp_path, score=0.9)
    _tail(
        root,
        pre={
            "type": "decision",
            "values": {"step": "measure", "file": "values.json"},
            "rule": {"k": {"in": [4]}},  # k is 8: fail
            "decision": {"on_pass": "advance", "on_fail": "narrow"},
        },
        pregate={
            "type": "conditional",
            "predicate": {
                "decision": {"step": "pre"},
                "field": "outcome",
                "eq": "pass",
            },
            "on_true": ["gate_k"],
            "on_false": ["best"],
            "scope": "global",
        },
    )
    loaded = _load(root, _receipt_raw(root))
    assert "tail/pregate" in loaded.dependencies["tail/gate_k"]
    _sentinels(monkeypatch)
    result = _run(loaded, root, tmp_path / "runs", _Stub())
    run_root = result.run_root
    statuses = _statuses(result)
    assert (
        statuses["tail/pre"] == "completed" and statuses["tail/pregate"] == "completed"
    )
    assert statuses["tail/gate_k"] == "skipped" and statuses["tail/best"] == "completed"
    assert statuses["apply"] == "skipped"
    by = result.manifest["steps"]["tail/gate_k"]["skipped_by"]
    assert by["conditional"] == "tail/pregate" and by["decision_step"] == "tail/pre"
    assert by["transitive_from"] == []
    assert result.manifest["steps"]["apply"]["skipped_by"]["transitive_from"] == [
        "tail/gate_k"
    ]
    assert not (run_root / "tail" / "gate_k").exists()
    assert not (run_root / "apply").exists()
    assert _record(run_root, "tail/pregate")["skipped"] == ["gate_k"]  # as written
    assert terminal(run_root / EVENTS_FILE)


def test_t22_a_missing_inner_receipt_is_a_distinct_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The inner producer published, then its receipt vanished before the
    outer step's turn (injected at `tail/gate_k`'s `published` seam): a
    different `W18`, naming absence under the sub-root and saying it is not a
    skip; still before any engine or model."""
    root = _tree(tmp_path, score=0.9)
    loaded = _load(root, _receipt_raw(root))
    run_root = tmp_path / "runs" / "nested"

    def vanish(name: str, at: str | None) -> None:
        if name == "published" and at == "tail/gate_k":
            (run_root / "tail" / "gate_k" / DECISION_FILE).unlink()

    monkeypatch.setattr(runner, "_boundary", vanish)
    _sentinels(monkeypatch)
    with pytest.raises(WorkflowError) as err:
        _run(loaded, root, tmp_path / "runs", _Stub())
    assert err.value.rule == CONDITIONAL_RULE
    message = str(err.value)
    assert "steps.apply.requires_receipt" in message and "'tail/gate_k'" in message
    assert "does not exist" in message and "not a skip" in message
    assert str(run_root / "tail" / "gate_k" / DECISION_FILE) in message
    assert "carries outcome" not in message  # distinct from the failed message
    manifest = json.loads((run_root / mf.MANIFEST).read_text())["steps"]
    assert manifest["tail/gate_k"]["status"] == "completed"
    assert manifest["apply"]["status"] == "failed"
    assert not (run_root / mf.ATTEMPTS_DIR / "apply").exists()


def test_t22_the_pass_twin_reaches_the_engine(tmp_path: Path) -> None:
    root = _tree(tmp_path, score=0.9)
    loaded = _load(root, _receipt_raw(root))
    with pytest.raises(AssertionError, match="executed"):
        _run(loaded, root, tmp_path / "runs", _Stub())
    assert (tmp_path / "runs" / "nested" / mf.ATTEMPTS_DIR / "apply").is_dir()


def test_t22_an_inner_receipt_is_checked_under_the_sub_root(tmp_path: Path) -> None:
    """An inner step's `requires_receipt` names an inner producer: the check
    reads `<run_root>/tail/gate_k/decision.json` (m22a: given the run root
    it reads `<run_root>/gate_k/…`, absent, and says *missing* where the
    receipt *failed*)."""
    root = _tree(tmp_path, score=0.1)
    raw = _tail(root)
    raw["steps"]["best"]["requires_receipt"] = {"step": "gate_k", "outcome": "pass"}
    _write(root, "tail.json", raw)
    loaded = _load(root)
    assert "tail/gate_k" in loaded.dependencies["tail/best"]
    with pytest.raises(WorkflowError) as err:
        _run(loaded, root, tmp_path / "runs")
    assert err.value.rule == CONDITIONAL_RULE
    assert "carries outcome 'fail'" in str(
        err.value
    ) and "steps.tail/best.requires_receipt" in str(err.value)
    run_root = tmp_path / "runs" / "nested"
    manifest = json.loads((run_root / mf.MANIFEST).read_text())["steps"]
    assert manifest["tail/gate_k"]["status"] == "completed"
    assert manifest["tail/best"]["status"] == "failed"
    assert manifest["report"]["status"] == "blocked"
    assert not (run_root / "tail" / mf.ATTEMPTS_DIR / "best").exists()
    # the pure check: the same step, under the sub-root and under the run root
    step = loaded.nested["tail"].document.steps["best"]
    with pytest.raises(WorkflowError, match="carries outcome 'fail'"):
        cond.check_receipt("best", step, run_root / "tail")
    with pytest.raises(WorkflowError, match="does not exist"):
        cond.check_receipt("best", step, run_root)


def test_t22_a_receipt_on_the_workflow_step_gates_every_inner_step(
    tmp_path: Path,
) -> None:
    root = _tree(tmp_path, outer_score=0.1)  # gate_k0: fail
    raw = _outer_decision(_raw(root))
    raw["steps"]["tail"]["requires_receipt"] = {"step": "gate_k0", "outcome": "pass"}
    loaded = _load(root, raw)
    assert loaded.canonical["steps"]["tail"]["requires_receipt"] == {
        "step": "gate_k0",
        "outcome": "pass",
    }
    assert loaded.dependencies["tail"] == ("gate_k0",)
    assert (
        "gate_k0" in loaded.dependencies["tail/measure"]
    )  # inherited by the inner root
    with pytest.raises(WorkflowError) as err:
        _run(loaded, root, tmp_path / "runs")
    assert err.value.rule == CONDITIONAL_RULE
    assert "steps.tail.requires_receipt" in str(
        err.value
    ) and "carries outcome 'fail'" in str(err.value)
    run_root = tmp_path / "runs" / "nested"
    manifest = json.loads((run_root / mf.MANIFEST).read_text())["steps"]
    assert manifest["tail/measure"]["status"] == "failed"
    assert manifest["tail/gate_k"]["status"] == "blocked"
    assert manifest["tail/best"]["status"] == "blocked"
    assert manifest["report"]["status"] == "blocked"
    assert not (run_root / "tail" / "measure").exists()
    # the pass twin runs everything
    passing = _tree(tmp_path / "pass", outer_score=0.9)
    twin = _outer_decision(_raw(passing))
    twin["steps"]["tail"]["requires_receipt"] = {"step": "gate_k0", "outcome": "pass"}
    again = _run(_load(passing, twin), passing, tmp_path / "pass" / "runs")
    assert set(_statuses(again).values()) == {"completed"}


def test_t22_an_outer_conditional_skips_the_whole_nested_workflow(
    tmp_path: Path,
) -> None:
    root = _tree(tmp_path, outer_score=0.3)  # gate_k0: fail -> on_true skipped
    raw = _gated(_raw(root), on_true=["tail"], on_false=["probe"])
    raw["steps"]["probe"] = _writer("probe.json")
    loaded = _load(root, raw)
    assert "gate" in loaded.dependencies["tail/measure"]  # the gate edge, inherited
    assert "gate" not in loaded.dependencies["tail/best"]  # by the roots alone
    result = _run(loaded, root, tmp_path / "runs")
    run_root = result.run_root
    statuses = _statuses(result)
    assert statuses == {
        "measure": "completed",
        "gate_k0": "completed",
        "gate": "completed",
        "probe": "completed",
        "tail/measure": "skipped",
        "tail/gate_k": "skipped",
        "tail/best": "skipped",
        "report": "skipped",
    }
    decision = json.loads((run_root / "gate_k0" / DECISION_FILE).read_text())
    for name in ("tail/measure", "tail/gate_k", "tail/best"):
        by = result.manifest["steps"][name]["skipped_by"]
        assert by["conditional"] == "gate" and by["transitive_from"] == []
        assert by["evidence_identity"] == decision["evidence_identity"]
        assert not (run_root / name).exists()
    assert result.manifest["steps"]["report"]["skipped_by"]["transitive_from"] == [
        "tail/best"
    ]
    assert not (run_root / "tail").exists()  # the container dir, removed when empty
    assert _record(run_root, "gate")["skipped"] == [
        "tail"
    ]  # the record, as the conditional wrote it
    records = read_events(run_root / EVENTS_FILE)
    skipped_lines = [
        r["payload"]["step"]
        for r in records
        if r["event"] == "phase_completed" and r["payload"]["status"] == "skipped"
    ]
    inner_and_report = {"tail/measure", "tail/gate_k", "tail/best", "report"}
    assert skipped_lines == [n for n in loaded.order if n in inner_and_report]
    assert sum(r["event"] == "campaign_terminal" for r in records) == 1
    assert {r["payload"].get("step") for r in records} - {None} == set(loaded.order)
    # the pass twin runs the nested workflow and skips `probe`
    passing = _tree(tmp_path / "pass", outer_score=0.9)
    twin = _gated(_raw(passing), on_true=["tail"], on_false=["probe"])
    twin["steps"]["probe"] = _writer("probe.json")
    again = _run(_load(passing, twin), passing, tmp_path / "pass" / "runs")
    assert _statuses(again)["probe"] == "skipped"
    assert {
        _statuses(again)[n]
        for n in ("tail/measure", "tail/gate_k", "tail/best", "report")
    } == {"completed"}


def test_t22_a_clean_run_then_resume_reuses_every_inner_step(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    loaded = _load(root)
    out = tmp_path / "runs"
    first = _run(loaded, root, out)
    run_root = first.run_root
    assert not (run_root / mf.ATTEMPTS_DIR).exists()
    assert not (run_root / "tail" / mf.ATTEMPTS_DIR).exists()
    before = _snapshot(run_root)
    again = _run(loaded, root, out, resume=True)
    assert _statuses(again) == {name: "reused" for name in OUTER}
    assert _snapshot(run_root) == before
    assert "workflow_digest" not in again.manifest  # no run-level identity (§7)
    assert again.manifest["nested"] == first.manifest["nested"]
    # an inner edit moves the outer digest and re-runs exactly what it moved
    raw = _raw(root, "tail.json")
    raw["steps"]["gate_k"]["rule"]["score"] = {"ge": 0.6}
    _write(root, "tail.json", raw)
    edited = _load(root)
    assert edited.digest != loaded.digest
    third = _run(edited, root, out, resume=True)
    assert _statuses(third) == {
        "measure": "reused",
        "tail/measure": "reused",
        "tail/gate_k": "completed",  # its identity moved (m22c: never reused at all)
        "tail/best": "reused",
        "report": "reused",
    }
    assert (
        third.manifest["nested"]["tail"]["workflow_digest"]
        == edited.nested["tail"].digest
    )
    assert (
        run_root
        / "tail"
        / mf.ATTEMPTS_DIR
        / "gate_k"
        / f"0001{mf.SUPERSEDED_SUFFIX}"
        / DECISION_FILE
    ).is_file()
    assert (
        _record(run_root, "tail/gate_k")["identity"]
        == edited.step_digests["tail/gate_k"]
    )
    # the retained unit is read under the sub-root (`tail/.attempts/gate_k/`)
    # on the publish path and on the reuse path alike (§8): a second unchanged
    # `--resume` reuses `tail/gate_k` and its entry lists the unit the third
    # run's publish listed
    retained = third.manifest["steps"]["tail/gate_k"]["superseded"]
    assert retained
    fourth = _run(edited, root, out, resume=True)
    assert _statuses(fourth)["tail/gate_k"] == "reused"
    assert fourth.manifest["steps"]["tail/gate_k"]["superseded"] == retained


def test_t22_the_existing_chains_are_unchanged(env: Any, tmp_path: Path) -> None:
    """T14 again: identities for every existing kind, no §2.10 key on the
    conditional chain or the fan-out fixture, a clean run then `--resume`
    byte-identical, `nested` empty."""
    kinds: set[str] = set()
    for name in ("mean_ablation.json", "weekdays.json", "pca_basis.json"):
        kinds |= _identity_kinds(load_workflow(WORKFLOWS / name, env))
    chain_root = _chain_tree(tmp_path)
    chain = _chain_load(chain_root)
    kinds |= _identity_kinds(chain)
    assert kinds == {"intervention_protocol", "script", "decision", "conditional"}
    assert chain.order == CHAIN and chain.nested == {}
    fanned = _fan_load(_fan_tree(tmp_path / "fan"))
    assert fanned.nested == {} and set(fanned.children) == {"scan"}
    for loaded in (chain, fanned):
        for name, entry in loaded.canonical["steps"].items():
            assert not set(entry) & set(NESTED_KEYS), name
    out = tmp_path / "runs"
    first = _chain_run(chain, chain_root, out)
    assert "nested" not in first.manifest
    before = _snapshot(first.run_root)
    again = _chain_run(chain, chain_root, out, resume=True)
    assert _snapshot(again.run_root) == before
    assert (
        _statuses(again)["fit"] == "reused" and _statuses(again)["probe"] == "skipped"
    )


def test_t22_an_inner_fan_out_joins_under_its_sub_root(tmp_path: Path) -> None:
    """`fan_out` inside the inner document is §2.9's, under the
    sub-root — children `tail/scan@i`, the join `tail/scan` — and the outer
    reads the joined table by `tail/best`."""
    root = _tree(tmp_path)
    loaded = _load(
        root, _scan_outer(root, {"over": {"shards": 2}, "join": {"require": "all"}})
    )
    assert loaded.children == {"tail/scan": ("tail/scan@0", "tail/scan@1")}
    assert nested.locate(loaded, "tail/scan@0") == (
        loaded.nested["tail"],
        "scan@0",
        "tail",
    )
    assert loaded.order == (
        "tail/scan@0",
        "tail/scan@1",
        "tail/scan",
        "tail/best",
        "report",
    )
    result = _run(loaded, root, tmp_path / "runs", _Rows())
    assert set(_statuses(result).values()) == {"completed"}
    join = _record(result.run_root, "tail/scan")
    assert join["fan_out"]["children"] == [
        "scan@0",
        "scan@1",
    ]  # the record, in its own table
    assert join["join"]["require"] == "all" and join["points"] == 8
    assert (result.run_root / "tail" / "scan@0" / "iia.json").is_file()
    assert (
        _values(result.run_root, "report", "report.json")["k"]
        == _values(result.run_root, "tail/best", "values.json")["best_layer"]
    )
    assert result.manifest["nested"]["tail"]["steps"] == [
        "tail/scan@0",
        "tail/scan@1",
        "tail/scan",
        "tail/best",
    ]


def test_t22_an_inner_conditionals_skipped_by_carries_flattened_names(
    tmp_path: Path,
) -> None:
    """A conditional *inside* the nested document (`tail/pregate`, reading
    `tail/pre`) skips `tail/gate_k`: every name in the skipped step's
    `skipped_by` block — the conditional and its `decision_step` — is a key
    of the manifest's step table, on the stream too (§2.10), on a fresh run
    and when the conditional is reused under `--resume`; the record itself
    keeps the inner-local spelling it wrote. The twin is
    `test_t22_an_outer_conditional_skips_the_whole_nested_workflow` (an outer
    conditional, whose sub-root is the run root)."""
    root = _tree(tmp_path, score=0.9)
    _tail(
        root,
        pre={
            "type": "decision",
            "values": {"step": "measure", "file": "values.json"},
            "rule": {"k": {"in": [4]}},  # k is 8: fail
            "decision": {"on_pass": "advance", "on_fail": "narrow"},
        },
        pregate={
            "type": "conditional",
            "predicate": {
                "decision": {"step": "pre"},
                "field": "outcome",
                "eq": "pass",
            },
            "on_true": ["gate_k"],
            "on_false": ["best"],
            "scope": "global",
        },
    )
    loaded = _load(root)
    out = tmp_path / "runs"
    for resume in (False, True):
        result = _run(loaded, root, out, resume=resume)
        steps = result.manifest["steps"]
        assert steps["tail/pregate"]["status"] == ("reused" if resume else "completed")
        assert steps["tail/gate_k"]["status"] == "skipped"
        skipped = {n: e["skipped_by"] for n, e in steps.items() if "skipped_by" in e}
        assert set(skipped) == {"tail/gate_k"}
        for name, by in skipped.items():
            assert by["conditional"] in steps, (name, by)
            assert by["decision_step"] in steps, (name, by)
        assert skipped["tail/gate_k"]["conditional"] == "tail/pregate"
        assert skipped["tail/gate_k"]["decision_step"] == "tail/pre"
        events = [
            r["payload"]
            for r in read_events(result.run_root / EVENTS_FILE)
            if r["event"] == "phase_completed" and r["payload"]["status"] == "skipped"
        ]
        # one stream, continued by the resume: one skipped line per run
        assert [e["step"] for e in events] == ["tail/gate_k"] * (2 if resume else 1)
        assert events[-1]["skipped_by"]["conditional"] == "tail/pregate"
        assert events[-1]["skipped_by"]["decision_step"] == "tail/pre"
    record = _record(out / "nested", "tail/pregate")
    assert record["skipped"] == ["gate_k"] and record["evidence"]["step"] == "pre"


def _deep_receipt_tree(root: Path, *, on: str) -> dict[str, Any]:
    """Two levels: the outer nests `mid.json` as `a`, which nests `tail.json`
    as `b`. With the measurement at 0.1 every `gate_k` fails. `on` is where
    the `requires_receipt` sits: the container `b` (naming `mid.json`'s own
    `gate_k`) or `tail.json`'s `best` (naming its own `gate_k`)."""
    _measure(root, 0.1)
    tail = _raw(root, "tail.json")
    if on == "inner_step":
        tail["steps"]["best"]["requires_receipt"] = {
            "step": "gate_k",
            "outcome": "pass",
        }
        _write(root, "tail.json", tail)
    mid: dict[str, Any] = {
        "version": "1",
        "output_dir": "mid_alone",
        "steps": {
            "measure": tail["steps"]["measure"],
            "gate_k": tail["steps"]["gate_k"],
            "b": {"type": "workflow", "document": "tail.json"},
        },
    }
    if on == "container":
        mid["steps"]["b"]["requires_receipt"] = {"step": "gate_k", "outcome": "pass"}
    _write(root, "mid.json", mid)
    return {
        "version": "1",
        "output_dir": "deep_receipt",
        "steps": {"a": {"type": "workflow", "document": "mid.json"}},
    }


def test_t22_a_deep_containers_receipt_refusal_names_its_flattened_name(
    tmp_path: Path,
) -> None:
    """A receipt on the container `a/b` (declared in `mid.json` as `b`) fails:
    the refusal names `steps.a/b.requires_receipt` — the flattened name the
    stream and the manifest carry, never the bare `b` of its owner's table.
    Depth 1 (`test_t22_a_receipt_on_the_workflow_step_gates_every_inner_step`)
    cannot tell the two apart; the pass twin is
    `test_t22_the_pass_twin_reaches_the_engine`."""
    root = _tree(tmp_path)
    loaded = _load(root, _deep_receipt_tree(root, on="container"))
    assert "a/gate_k" in loaded.dependencies["a/b/measure"]
    with pytest.raises(WorkflowError) as err:
        _run(loaded, root, tmp_path / "runs")
    assert err.value.rule == CONDITIONAL_RULE
    message = str(err.value)
    assert "steps.a/b.requires_receipt" in message, message
    assert "'a/b' requires 'pass'" in message and "carries outcome 'fail'" in message
    assert "steps.b.requires_receipt" not in message
    # the producer too is spelled as the manifest spells it — one spelling in
    # one message: `mid.json` says `gate_k`, the run says `a/gate_k`
    assert "receipt 'a/gate_k' carries outcome 'fail'" in message, message
    assert "'gate_k'" not in message
    manifest = json.loads(
        (tmp_path / "runs" / "deep_receipt" / mf.MANIFEST).read_text()
    )
    assert manifest["steps"]["a/b/measure"]["status"] == "failed"
    assert manifest["steps"]["a/gate_k"]["status"] == "completed"
    # the receipt read is the file the old sub-root arithmetic reached:
    # `<run>/a/gate_k/decision.json`, whose evidence the message quotes
    receipt = tmp_path / "runs" / "deep_receipt" / "a" / "gate_k" / DECISION_FILE
    decision = json.loads(receipt.read_text())
    assert decision["outcome"] == "fail"
    assert f"evidence {decision['evidence_identity']}" in message
    # the pure check over the rebased step against the run root: a missing
    # receipt names that same path
    with pytest.raises(WorkflowError) as missing:
        cond.check_receipt("a/b", loaded.document.steps["a/b"], tmp_path / "empty")
    assert str(tmp_path / "empty" / "a" / "gate_k" / DECISION_FILE) in str(
        missing.value
    )


def test_t22_a_deep_inner_steps_receipt_refusal_names_its_flattened_name(
    tmp_path: Path,
) -> None:
    """An inner step two levels down (`a/b/best`, declared in `tail.json` as
    `best`) requires a receipt that fails: the refusal names
    `steps.a/b/best.requires_receipt`, never `steps.best.…`."""
    root = _tree(tmp_path)
    loaded = _load(root, _deep_receipt_tree(root, on="inner_step"))
    assert "a/b/gate_k" in loaded.dependencies["a/b/best"]
    with pytest.raises(WorkflowError) as err:
        _run(loaded, root, tmp_path / "runs")
    assert err.value.rule == CONDITIONAL_RULE
    message = str(err.value)
    assert "steps.a/b/best.requires_receipt" in message, message
    assert "'a/b/best' requires 'pass'" in message
    assert "steps.best.requires_receipt" not in message
    # the producer by its flattened name too, never `tail.json`'s `gate_k`
    assert "receipt 'a/b/gate_k' carries outcome 'fail'" in message, message
    assert "'gate_k'" not in message
    manifest = json.loads(
        (tmp_path / "runs" / "deep_receipt" / mf.MANIFEST).read_text()
    )
    assert manifest["steps"]["a/b/best"]["status"] == "failed"
    assert manifest["steps"]["a/b/gate_k"]["status"] == "completed"
    receipt = tmp_path / "runs" / "deep_receipt" / "a" / "b" / "gate_k" / DECISION_FILE
    decision = json.loads(receipt.read_text())
    assert decision["outcome"] == "fail"
    assert f"evidence {decision['evidence_identity']}" in message
    with pytest.raises(WorkflowError) as missing:
        cond.check_receipt(
            "a/b/best", loaded.document.steps["a/b/best"], tmp_path / "empty"
        )
    assert str(tmp_path / "empty" / "a" / "b" / "gate_k" / DECISION_FILE) in str(
        missing.value
    )


def test_emit_as_renames_only_the_attempted_step() -> None:
    """The stream carries flattened names (§2.10, §4.3): a payload about the
    step being attempted (its local name) is renamed; one naming another step
    — a control's subject in an instrument-failure warning — and one naming
    no step pass through unchanged."""
    seen: list[tuple[str, dict[str, Any]]] = []
    emit = runner._emit_as(  # pyright: ignore[reportPrivateUsage]
        lambda event, payload: seen.append((event, dict(payload))), "fit", "tail/fit"
    )
    emit("phase_started", {"step": "fit", "type": "script"})
    emit(
        "warning",
        {"step": "apply", "reason": "instrument_failure", "certified_by": "fit"},
    )
    emit("campaign_terminal", {"status": "completed"})
    assert seen == [
        ("phase_started", {"step": "tail/fit", "type": "script"}),
        (
            "warning",
            {"step": "apply", "reason": "instrument_failure", "certified_by": "fit"},
        ),
        ("campaign_terminal", {"status": "completed"}),
    ]


@pytest.mark.parametrize("steps", [[], "abc"], ids=["a_list", "a_string"])
def test_a_non_mapping_inner_steps_table_is_refused_under_rule_1(
    tmp_path: Path, steps: Any
) -> None:
    """An inner document whose `steps` is not an object, mounted with a
    non-empty `set`: the shape is the inner parse's to refuse (rule 1), and
    the outer wraps that refusal as its own (§2.10) — never a `KeyError` out
    of laying the `set` over a table that is not there. The twin is
    `test_the_valid_twin_loads_and_runs_to_completed[set_on_inner_document_step]`
    (a well-formed inner with the same kind of `set`)."""
    root = _tree(tmp_path)
    _write(root, "inner.json", {"version": "1", "output_dir": "x", "steps": steps})
    raw = _raw(root)
    raw["steps"]["tail"] = {
        "type": "workflow",
        "document": "inner.json",
        "set": {"a": {"model.key": "gpt2"}},
    }
    with pytest.raises(WorkflowError) as err:
        _load(root, raw)
    assert err.value.rule == NESTED_RULE, str(err.value)
    message = str(err.value)
    assert "workflow 'inner.json' does not load: [W1]" in message, message
    assert "'steps' is a non-empty object" in message
    assert "steps.tail" in message


def _mounted_per_child(root: Path, *, scope: str = "per_target") -> dict[str, Any]:
    """The fan-out fixture's per-child chain with its conditional taken out,
    nested as `tail`; the outer declares the per-child conditional over the
    mounted producer `tail/qualify`, gating `tail/apply` / `tail/narrow`."""
    inner = _behavioral_raw(root, require="all")
    del inner["steps"]["gate"]
    _write(root, "per_child.json", inner)
    return {
        "version": "1",
        "output_dir": "nested_per_child",
        "steps": {
            "tail": {"type": "workflow", "document": "per_child.json"},
            "gate": {
                "type": "conditional",
                "predicate": {
                    "decision": {"step": "tail/qualify"},
                    "field": "outcome",
                    "eq": "pass",
                },
                "on_true": ["tail/apply"],
                "on_false": ["tail/narrow"],
                "scope": scope,
            },
        },
    }


def test_a_per_child_conditional_over_a_mounted_fan_out_is_refused_under_rule_20(
    tmp_path: Path,
) -> None:
    """Rule 19 would accept it (`rebase` keeps `fan_out`, so the mounted
    producer and gated steps satisfy `check_scope`) and the outer's expansion
    runs over its own steps alone, so the conditional would load and never
    expand. Refused under rule 20 naming the conditional, the mounted step,
    the container and the fix. The twins: a per-child conditional over an
    *own* fan-out expands and runs
    (`test_fan_out.py::test_t16_a_selected_join_publishes_the_children_the_verdicts_left`)
    and one *inside* the nested document expands under its sub-root
    (`test_a_per_child_conditional_inside_the_inner_document_expands_under_its_sub_root`)."""
    root = _tree(tmp_path)
    err = _refused(root, _mounted_per_child(root), path="steps.gate.scope")
    message = str(err)
    assert "scope 'per_target' on 'gate' names 'tail/qualify'" in message, message
    assert "a step of the nested workflow 'tail'" in message
    assert "declare the per-child conditional inside 'tail''s own document" in message
    assert "'per_child.json'" in message
    # the scope is the difference: the same document with a global scope loads
    global_scope = _mounted_per_child(root, scope="global")
    assert _load(root, global_scope).children["tail/qualify"] == (
        "tail/qualify@0",
        "tail/qualify@1",
    )


def test_a_per_child_conditional_inside_the_inner_document_expands_under_its_sub_root(
    tmp_path: Path,
) -> None:
    """The refusal's twin: the whole per-child chain — producer, conditional,
    gated steps — inside the nested document is §2.9's, expanded by the inner
    load and mounted with its children (`tail/gate@i` reads `tail/qualify@i`)
    and runs to the verdicts the fan-out twin pins, under the sub-root."""
    root = _tree(tmp_path)
    _write(root, "per_child.json", _behavioral_raw(root))
    raw = {
        "version": "1",
        "output_dir": "nested_per_child",
        "steps": {"tail": {"type": "workflow", "document": "per_child.json"}},
    }
    loaded = _load(root, raw)
    assert loaded.children["tail/gate"] == ("tail/gate@0", "tail/gate@1")
    gate0 = loaded.document.steps["tail/gate@0"]
    assert isinstance(gate0, ConditionalStep)
    assert gate0.predicate["decision"]["step"] == "tail/qualify@0"
    assert gate0.on_true == ("tail/apply@0",) and gate0.on_false == ("tail/narrow@0",)
    result = _run(loaded, root, tmp_path / "runs", _Rows())
    statuses = _statuses(result)
    assert statuses == {
        "tail/qualify@0": "completed",
        "tail/qualify@1": "completed",
        "tail/qualify": "completed",
        "tail/gate@0": "completed",
        "tail/gate@1": "completed",
        "tail/gate": "completed",
        "tail/apply@0": "completed",
        "tail/apply@1": "skipped",
        "tail/apply": "completed",
        "tail/narrow@0": "skipped",
        "tail/narrow@1": "completed",
        "tail/narrow": "completed",
    }
    by = result.manifest["steps"]["tail/apply@1"]["skipped_by"]
    assert by["conditional"] == "tail/gate@1"  # its `decision_step`
    assert _record(result.run_root, "tail/gate")["verdicts"] == {
        "gate@0": True,
        "gate@1": False,
    }  # the record, in its own table


def _deep_mounted_per_child(root: Path) -> dict[str, Any]:
    """`_mounted_per_child` one level deeper: the outer mounts `mid.json` as
    `a`, `mid.json` mounts the per-child chain (`per_child.json`) as `b`, and
    the outer declares the per-child conditional over `a/b/qualify`."""
    inner = _behavioral_raw(root, require="all")
    del inner["steps"]["gate"]
    _write(root, "per_child.json", inner)
    _write(
        root,
        "mid.json",
        {
            "version": "1",
            "output_dir": "mid_per_child",
            "steps": {"b": {"type": "workflow", "document": "per_child.json"}},
        },
    )
    return {
        "version": "1",
        "output_dir": "nested_per_child_deep",
        "steps": {
            "a": {"type": "workflow", "document": "mid.json"},
            "gate": {
                "type": "conditional",
                "predicate": {
                    "decision": {"step": "a/b/qualify"},
                    "field": "outcome",
                    "eq": "pass",
                },
                "on_true": ["a/b/apply"],
                "on_false": ["a/b/narrow"],
                "scope": "per_target",
            },
        },
    }


def test_the_per_child_refusal_names_the_innermost_container(tmp_path: Path) -> None:
    """At depth 2 the advice names the container that declares the fan-out —
    `a/b` and `per_child.json` — never the outermost `a` and `mid.json`, which
    declare nothing the conditional could sit beside. Depth 1
    (`test_a_per_child_conditional_over_a_mounted_fan_out_is_refused_under_rule_20`)
    cannot tell the two apart."""
    root = _tree(tmp_path)
    err = _refused(root, _deep_mounted_per_child(root), path="steps.gate.scope")
    message = str(err)
    assert "scope 'per_target' on 'gate' names 'a/b/qualify'" in message, message
    assert "a step of the nested workflow 'a/b'" in message
    assert (
        "declare the per-child conditional inside 'a/b''s own document, "
        "'per_child.json'" in message
    )
    assert "'a''s own document" not in message and "'mid.json'" not in message


# --------------------------------------------------------------------------- #
# rule 14 across the boundary: a control featurizer drawn inside a nested workflow
# --------------------------------------------------------------------------- #


def _draw_tail(
    root: Path, *, seed: int | None = 7, inner_fits: bool = False
) -> dict[str, Any]:
    """`test_controls.py::test_t8_the_random_mask_chain_declares_its_control`
    with its drawing half moved into a reusable tail: `draw_tail.json`
    draws (`causalab.analysis.random_mask`, the seed on the script step's
    inputs) from a gate bundle a step of its own publishes; the outer fits,
    nests the tail and applies the drawn gate as its fit's `matched_random`
    control, loading it from `tail/draw/gate.safetensors` — `<step>` there is
    `tail/draw`, a script step of the flattened table, never its container
    `tail`. The publishing step is `source`, the conditional fixtures' writer
    declaring `gate.safetensors` (a load never runs it), because the outer
    engages rule 14 and rule 20 refuses a nested fit under an engaged outer;
    with ``inner_fits`` the tail is the t8 shape itself — `fit` (`dbm.json`)
    and `draw` — the nesting that refusal is about."""
    shutil.copytree(
        CONDITIONAL_FIXTURES / "scripts", root / "scripts", dirs_exist_ok=True
    )
    publisher = "fit" if inner_fits else "source"
    draw: dict[str, Any] = {
        "type": "script",
        "script": {"module": "causalab.analysis.random_mask"},
        "inputs": {"gate": {"step": publisher, "file": "gate.safetensors"}},
        "outputs": {"gate": "gate.safetensors"},
    }
    if seed is not None:
        draw["inputs"]["seed"] = seed
    source: dict[str, Any] = (
        _protocol(PROTOCOLS / "dbm.json")
        if inner_fits
        else {
            "type": "script",
            "script": {"path": "scripts/writer.py"},
            "inputs": {},
            "outputs": {"gate": "gate.safetensors"},
        }
    )
    _write(
        root,
        "draw_tail.json",
        {
            "version": "1",
            "output_dir": "draw_tail",
            "steps": {publisher: source, "draw": draw},
        },
    )
    return _workflow(
        {
            "fit": _protocol(PROTOCOLS / "dbm.json", waive={"self_swap": EXTERNAL}),
            "tail": {"type": "workflow", "document": "draw_tail.json"},
            "apply_control": _protocol(
                PROTOCOLS / "dbm_apply.json",
                set={"featurizers.gate.file_path": "tail/draw/gate.safetensors"},
                control={
                    "of": "fit",
                    "kind": "matched_random",
                    "seeds": [7],
                    "min_draws": 1,
                },
            ),
        }
    )


def _controls_refusal(raw: dict[str, Any], root: Path, env: Any) -> str:
    with pytest.raises(WorkflowError) as err:
        load_workflow(raw, env, workflow_dir=root)
    assert err.value.rule == CONTROL_RULE, str(err.value)
    assert "steps.apply_control.control.seeds" in str(err.value), str(err.value)
    return str(err.value)


def test_a_control_featurizer_drawn_inside_a_nested_workflow_records_the_inner_seed(
    tmp_path: Path, env: Any
) -> None:
    """The t8 chain with `draw` moved into a tail loads: the control's seed
    provenance is the inner draw's, found by `producer_of` over the flattened
    table (`tail/draw`), not by the head of the path (`tail`, a `workflow`
    step, which has no seed and refused the shape with the wrong reason). The
    two t8 mutations name the inner step by its flattened name. The twins:
    t8 itself at one root, and
    `test_a_control_featurizer_loaded_from_a_nested_step_that_draws_no_seed_is_still_refused`."""
    raw = _draw_tail(tmp_path)
    loaded = load_workflow(raw, env, workflow_dir=tmp_path)
    assert loaded.order.index("tail/draw") < loaded.order.index("apply_control")
    assert "tail/draw" in loaded.dependencies["apply_control"]
    assert loaded.canonical["steps"]["apply_control"]["control"]["seeds"] == [7]
    assert producer_of("tail/draw/gate.safetensors", loaded.document.steps) == (
        "tail/draw"
    )
    raw["steps"]["apply_control"]["control"]["seeds"] = [8]
    message = _controls_refusal(raw, tmp_path, env)
    assert "drawn by step 'tail/draw' at seed 7 — record every draw" in message, message
    message = _controls_refusal(_draw_tail(tmp_path, seed=None), tmp_path, env)
    assert "drawn by step 'tail/draw', whose inputs carry no integer 'seed'" in message


def test_a_control_featurizer_loaded_from_a_nested_step_that_draws_no_seed_is_still_refused(
    tmp_path: Path, env: Any
) -> None:
    """The fail-closed twin: a `file_path` under a nested step that draws at
    no seed (`tail/source`, the script step that publishes the gate the draw
    reads, no `seed` among its inputs) is still refused with the terminal
    `W14` naming the inner step — a control that names no seed source records
    nothing. (The other terminal arm, a non-script inner producer, has no
    nested twin under an engaged outer: only a fit publishes a gate bundle,
    and rule 20 refuses a nested fit there.)"""
    raw = _draw_tail(tmp_path)
    raw["steps"]["apply_control"]["set"] = {
        "featurizers.gate.file_path": "tail/source/gate.safetensors"
    }
    message = _controls_refusal(raw, tmp_path, env)
    assert (
        "drawn by step 'tail/source', whose inputs carry no integer 'seed'" in message
    )


def test_an_engaged_outer_refuses_a_nested_document_that_contains_a_fit(
    tmp_path: Path, env: Any
) -> None:
    """Rule 14's coverage requirement is per workflow document, and rule 20
    fails it closed across the boundary (§2.2, §2.10). `_draw_tail`'s outer
    engages the layer — its `fit` waives `self_swap`, its `apply_control`
    declares `matched_random` of `fit` — and `tail/fit`, the same `dbm.json`
    and so a fit, can declare nothing and waive nothing (rule 20 forbids it)
    while the outer's declarations cannot reach it: the nesting is refused
    under `W20` at the `workflow` step's `document`, naming the inner fit by
    its flattened name, instead of loading a fit no control holds. The
    loading twins are
    `test_a_nested_fit_loads_under_an_outer_that_engages_nothing` and
    `test_an_engaged_outer_nests_a_document_with_no_fit`; the one-root twin is
    `test_the_same_two_fits_as_own_steps_of_one_document_are_held`."""
    raw = _draw_tail(tmp_path, inner_fits=True)
    own = {n: s for n, s in raw["steps"].items() if s["type"] != "workflow"}
    assert any("control" in s or "waive" in s for s in own.values())  # engaged
    with pytest.raises(WorkflowError) as err:
        load_workflow(raw, env, workflow_dir=tmp_path)
    assert err.value.rule == NESTED_RULE, str(err.value)
    message = str(err.value)
    assert "steps.tail.document" in message, message
    assert "contains the fit 'tail/fit'" in message, message
    assert "not held by the outer's controls" in message, message
    assert "declare the control in the outer's own steps" in message, message


def test_a_nested_fit_loads_under_an_outer_that_engages_nothing(
    tmp_path: Path, env: Any
) -> None:
    """The same nesting with the outer's `waive` and `control` removed engages
    nothing, so rule 14 requires nothing of any fit — the outer's or the
    nested one's — and the document loads with `tail/fit` a fit that declares
    nothing, as the shipped fit campaigns that predate the layer do."""
    raw = _draw_tail(tmp_path, inner_fits=True)
    del raw["steps"]["fit"]["waive"]
    del raw["steps"]["apply_control"]["control"]
    loaded = load_workflow(raw, env, workflow_dir=tmp_path)
    own = {
        name: step
        for name, step in loaded.document.steps.items()
        if SEPARATOR not in name and isinstance(step, ProtocolStep)
    }
    assert not any(s.control is not None or s.waive is not None for s in own.values())
    inner_fit = loaded.document.steps["tail/fit"]
    assert isinstance(inner_fit, ProtocolStep)
    assert loaded.inner["tail/fit"].compiled.document.train is not None  # a fit
    assert inner_fit.control is None and inner_fit.waive is None


def test_an_engaged_outer_nests_a_document_with_no_fit(
    tmp_path: Path, env: Any
) -> None:
    """`_draw_tail`'s default shape — the engaged outer over a tail whose
    publisher is a script step, so the tail holds no fit — loads: the refusal
    is about an unheld fit, not about nesting under an engaged outer."""
    loaded = load_workflow(_draw_tail(tmp_path), env, workflow_dir=tmp_path)
    nested_steps = {
        name: step
        for name, step in loaded.document.steps.items()
        if name.startswith("tail" + SEPARATOR)
    }
    assert set(nested_steps) == {"tail/source", "tail/draw"}
    assert not any(
        isinstance(step, ProtocolStep)
        and loaded.inner[name].compiled.document.train is not None
        for name, step in nested_steps.items()
    )


def test_the_same_two_fits_as_own_steps_of_one_document_are_held(env: Any) -> None:
    """The fail-closed twin at one root: `_draw_tail`'s two fits as OWN steps
    of one document — `fit` (waives `self_swap`; `apply_control` declares its
    `matched_random`) and `fit2` (the same `dbm.json`, nothing declared,
    nothing waived), with t8's `draw` at the root — is refused under rule 14
    naming `fit2` and the first required kind it lacks; without `fit2` it is
    t8's shape and loads."""
    steps = {
        "fit": _protocol(PROTOCOLS / "dbm.json", waive={"self_swap": EXTERNAL}),
        "fit2": _protocol(PROTOCOLS / "dbm.json"),
        "draw": {
            "type": "script",
            "script": {"module": "causalab.analysis.random_mask"},
            "inputs": {"gate": {"step": "fit", "file": "gate.safetensors"}, "seed": 7},
            "outputs": {"gate": "gate.safetensors"},
        },
        "apply_control": _protocol(
            PROTOCOLS / "dbm_apply.json",
            set={"featurizers.gate.file_path": "draw/gate.safetensors"},
            control={
                "of": "fit",
                "kind": "matched_random",
                "seeds": [7],
                "min_draws": 1,
            },
        ),
    }
    with pytest.raises(WorkflowError) as err:
        load_workflow(_workflow(steps), env)
    assert err.value.rule == CONTROL_RULE, str(err.value)
    message = str(err.value)
    assert "steps.fit2" in message, message
    assert REQUIRED_CONTROL_KINDS[0] == "self_swap"
    assert (
        "step 'fit2' declares a fit; control 'self_swap' is neither declared by a "
        "step nor waived" in message
    ), message
    del steps["fit2"]
    load_workflow(_workflow(steps), env)  # t8's shape, every fit covered


#: The lines under `causalab/workflow/` that take the head of a `/`-separated
#: name and are not `producer_of`'s fallback — `(module, the line, why it is
#: not a producer lookup)`. A step name is a path now (§2.10); every site that
#: asks which step a run-tree path belongs to asks `producer_of`.
HEAD_SPLIT_ALLOWLIST: tuple[tuple[str, str, str], ...] = (
    (
        "causalab/workflow/nested.py",
        "head, _, tail = rest.partition(SEPARATOR)",
        "`_walk` descends one container at a time, checking `head in "
        "owner.nested`: it names containers, never a producer",
    ),
    (
        "causalab/workflow/cli.py",
        "head = name.split(nested.SEPARATOR, 1)[0]",
        "`explain` groups the listing under the top-level `workflow` step, "
        "which is a flattened name's head by construction",
    ),
    (
        "causalab/workflow/runner.py",
        'return ref.split("/", 1)[0] in self.step_names',
        "a yes/no over the head: a run-tree path's head is a key of the table "
        "at that root (a step or its container), and step names shadow the "
        "external root (§3)",
    ),
)
HEAD_SPLIT = re.compile(
    r"""\.(?:split\(\s*(?:"/"|(?:nested\.)?SEPARATOR)\s*,\s*1\s*\)"""
    r"""|partition\(\s*(?:"/"|(?:nested\.)?SEPARATOR)\s*\))"""
)


def test_no_workflow_module_splits_a_run_tree_path_at_its_first_slash() -> None:
    """The census the seed-provenance site would have failed: no module under
    `causalab/workflow/` takes a run-tree path's producer from its head
    (`.split("/", 1)[0]`, `.partition("/")[0]`, the `SEPARATOR` spellings)
    except as `producer_of`'s fallback on the same line, or on an allowlisted
    line with its reason — and every allowlisted line is still there."""
    hits: dict[tuple[str, str], int] = {}
    for module in sorted((REPO / "causalab" / "workflow").rglob("*.py")):
        rel = module.relative_to(REPO).as_posix()
        for number, line in enumerate(module.read_text().splitlines(), 1):
            if HEAD_SPLIT.search(line):
                hits[(rel, line.strip())] = number
    fallbacks = {key for key in hits if "producer_of(" in key[1]}
    allowed = {(module, line) for module, line, _ in HEAD_SPLIT_ALLOWLIST}
    unexplained = sorted(
        f"{module}:{hits[(module, line)]}: {line}"
        for module, line in hits
        if (module, line) not in fallbacks | allowed
    )
    assert not unexplained, unexplained
    assert allowed <= set(hits), sorted(allowed - set(hits))  # no stale entry
    assert fallbacks, "the census pattern matches nothing"


# --------------------------------------------------------------------------- #
# refusals, each naming the field — and their valid twins
# --------------------------------------------------------------------------- #


def _tail_step(raw: dict[str, Any], **changes: Any) -> dict[str, Any]:
    raw["steps"]["tail"].update(changes)
    return raw


def _with_control(root: Path, raw: dict[str, Any], of: str) -> dict[str, Any]:
    raw["steps"]["ctl"] = {
        "type": "intervention_protocol",
        "document": SCAN,
        "control": {"of": of, "kind": "self_swap"},
    }
    return raw


def _inner_waives(root: Path, raw: dict[str, Any]) -> dict[str, Any]:
    raw = _scan_outer(root)
    inner = _raw(root, "inner_scan.json")
    inner["steps"]["scan"]["waive"] = {"matched_random": "no_fit"}
    _write(root, "inner_scan.json", inner)
    return raw


#: name → (build(root, raw) -> raw, rule, the path the refusal names, a phrase)
NESTED_REFUSALS: dict[
    str, tuple[Callable[[Path, dict[str, Any]], dict[str, Any]], int, str, str]
] = {
    "set_names_an_unknown_inner_step": (
        lambda root, raw: _tail_step(raw, set={"nope": {"model.key": "gpt2"}}),
        NESTED_RULE,
        "steps.tail.set.nope",
        "not a step of 'tail.json'",
    ),
    "set_names_a_script_step": (
        lambda root, raw: _tail_step(raw, set={"measure": {"model.key": "gpt2"}}),
        NESTED_RULE,
        "steps.tail.set.measure",
        "a 'script' step",
    ),
    "set_names_a_nested_workflow_step": (
        lambda root, raw: (
            _write(
                root,
                "mid.json",
                {
                    "version": "1",
                    "output_dir": "m",
                    "steps": {"inner": {"type": "workflow", "document": "tail.json"}},
                },
            )
            and _tail_step(raw, document="mid.json", set={"inner": {"measure": {}}})
        ),
        NESTED_RULE,
        "steps.tail.set.inner",
        "'set' reaches one level",
    ),
    "set_value_is_not_a_map": (
        lambda root, raw: _tail_step(raw, set={"measure": 3}),
        NESTED_RULE,
        "steps.tail.set.measure",
        "own 'set'",
    ),
    "set_is_not_a_map": (
        lambda root, raw: _tail_step(raw, set=["measure"]),
        NESTED_RULE,
        "steps.tail.set",
        "maps inner step names",
    ),
    "document_is_an_intervention_specification": (
        lambda root, raw: _tail_step(raw, document=SCAN),
        NESTED_RULE,
        "steps.tail.document",
        "no 'steps' section",
    ),
    "document_is_missing": (
        lambda root, raw: _tail_step(raw, document="absent.json"),
        4,
        "steps.tail",
        "not found",
    ),
    "inner_document_does_not_load": (
        lambda root, raw: (
            _tail(
                root,
                gate_k={
                    "type": "decision",
                    "values": {"step": "measure", "file": "values.json"},
                    "rule": {"nope": {"ge": 1}},
                    "decision": {"on_pass": "advance", "on_fail": "narrow"},
                },
            )
            and raw
        ),
        NESTED_RULE,
        "steps.tail",
        "does not load: [W4]",
    ),
    "reference_names_the_workflow_step": (
        lambda root, raw: (
            raw["steps"]["report"]["inputs"].__setitem__(
                "k", {"step": "tail", "file": "best.json", "key": "k"}
            )
            or raw
        ),
        NESTED_RULE,
        "steps.report.inputs.k",
        "publishes no file",
    ),
    "receipt_names_the_workflow_step": (
        lambda root, raw: (
            raw["steps"]["report"].__setitem__(
                "requires_receipt", {"step": "tail", "outcome": "pass"}
            )
            or raw
        ),
        NESTED_RULE,
        "steps.report.requires_receipt.step",
        "writes no receipt",
    ),
    "predicate_names_the_workflow_step": (
        lambda root, raw: (
            raw["steps"].__setitem__(
                "gate",
                {
                    "type": "conditional",
                    "predicate": {
                        "decision": {"step": "tail"},
                        "field": "outcome",
                        "eq": "pass",
                    },
                    "on_true": ["report"],
                    "on_false": ["measure"],
                    "scope": "global",
                },
            )
            or raw
        ),
        NESTED_RULE,
        "steps.gate.predicate.decision.step",
        "writes no receipt",
    ),
    "values_names_the_workflow_step": (
        lambda root, raw: (
            raw["steps"].__setitem__(
                "d",
                {
                    "type": "decision",
                    "values": {"step": "tail", "file": "values.json"},
                    "rule": {"score": {"ge": 0.5}},
                    "decision": {"on_pass": "advance", "on_fail": "narrow"},
                },
            )
            or raw
        ),
        NESTED_RULE,
        "steps.d.values.step",
        "publishes no file",
    ),
    "control_of_the_workflow_step": (
        lambda root, raw: _with_control(root, raw, "tail"),
        NESTED_RULE,
        "steps.ctl.control.of",
        "a workflow step",
    ),
    "control_of_an_inner_step": (
        lambda root, raw: _with_control(root, _scan_outer(root), "tail/scan"),
        NESTED_RULE,
        "steps.ctl.control.of",
        "a nested workflow's step",
    ),
    "inner_declares_waive": (
        _inner_waives,
        NESTED_RULE,
        "steps.tail.document",
        "declares 'waive'",
    ),
    "fan_out_on_the_workflow_step": (
        lambda root, raw: _tail_step(
            raw, fan_out={"over": {"shards": 2}, "join": {"require": "all"}}
        ),
        1,
        "steps.tail",
        "unknown key 'fan_out'",
    ),
    "after_names_an_unknown_inner_step": (
        lambda root, raw: (
            raw["steps"]["report"].__setitem__("after", ["tail/nope"]) or raw
        ),
        4,
        "steps.report",
        "unknown step 'tail/nope'",
    ),
    # rule 18 sees through the boundary: `report` reads `tail/best`, so the
    # two sides are not dependency-disjoint — the container is its members
    "conditional_sides_cross_the_nesting_boundary": (
        lambda root, raw: _gated(raw, on_true=["tail"], on_false=["report"]),
        CONDITIONAL_RULE,
        "steps.gate.on_false",
        "'on_false' names 'report', which depends on "
        "['tail/best', 'tail/measure'] in 'on_true'",
    ),
}


@pytest.mark.parametrize("case", sorted(NESTED_REFUSALS))
def test_a_malformed_nesting_is_refused_naming_the_field(
    tmp_path: Path, case: str
) -> None:
    build, rule, path, phrase = NESTED_REFUSALS[case]
    root = _tree(tmp_path)
    err = _refused(root, build(root, _raw(root)), rule=rule, path=path)
    assert phrase in str(err), str(err)


def _twin_two_levels(
    root: Path,
) -> tuple[dict[str, Any], Any, dict[str, str] | None]:
    return _raw(root / "deep", "a.json"), None, None


def _twin_set_on_inner_document_step(
    root: Path,
) -> tuple[dict[str, Any], Any, dict[str, str] | None]:
    raw = _scan_outer(root)
    raw["steps"]["tail"]["set"] = {"scan": {"sites.target.layers": {"sweep": [0, 1]}}}
    return raw, _Rows(), None


def _twin_receipt_on_inner_decision(
    root: Path,
) -> tuple[dict[str, Any], Any, dict[str, str] | None]:
    raw = _raw(root)
    raw["steps"]["report"]["requires_receipt"] = {
        "step": "tail/gate_k",
        "outcome": "pass",
    }
    return raw, None, None


def _twin_conditional_gating_the_workflow_step(
    root: Path,
) -> tuple[dict[str, Any], Any, dict[str, str] | None]:
    _outer_measure(root, 0.9)  # gate_k0: pass -> on_false skipped
    raw = _gated(_raw(root), on_true=["tail"], on_false=["probe"])
    raw["steps"]["probe"] = _writer("probe.json")
    expected = {n: "completed" for n in (*OUTER, "gate_k0", "gate")}
    return raw, None, {**expected, "probe": "skipped"}


def _twin_conditional_sides_disjoint_across_the_boundary(
    root: Path,
) -> tuple[dict[str, Any], Any, dict[str, str] | None]:
    """The refusal's twin: `on_true: [tail]` against a `report` that reads the
    *outer* `measure` — no step of one side depends on the other, so the
    conditional loads and runs (gate_k0: fail -> `tail` skipped, `report` runs)."""
    raw = _gated(_raw(root), on_true=["tail"], on_false=["report"])
    raw["steps"]["report"]["inputs"]["k"] = {
        "step": "measure",
        "file": "values.json",
        "key": "score",
    }
    skipped = {n: "skipped" for n in ("tail/measure", "tail/gate_k", "tail/best")}
    completed = {n: "completed" for n in ("measure", "gate_k0", "gate", "report")}
    return raw, None, {**skipped, **completed}


def _twin_after_the_workflow_step(
    root: Path,
) -> tuple[dict[str, Any], Any, dict[str, str] | None]:
    raw = _raw(root)
    raw["steps"]["report"] = _writer("report.json", after=["tail"])
    return raw, None, None


def _twin_reference_to_an_inner_step(
    root: Path,
) -> tuple[dict[str, Any], Any, dict[str, str] | None]:
    return _raw(root), None, None


def _twin_inner_fan_out(
    root: Path,
) -> tuple[dict[str, Any], Any, dict[str, str] | None]:
    return (
        _scan_outer(
            root, {"over": {"axis": "sites.target.layers"}, "join": {"require": "all"}}
        ),
        _Rows(),
        None,
    )


def _twin_receipt_on_the_workflow_step(
    root: Path,
) -> tuple[dict[str, Any], Any, dict[str, str] | None]:
    _outer_measure(root, 0.9)
    raw = _outer_decision(_raw(root))
    raw["steps"]["tail"]["requires_receipt"] = {"step": "gate_k0", "outcome": "pass"}
    return raw, None, None


def _twin_same_document_twice(
    root: Path,
) -> tuple[dict[str, Any], Any, dict[str, str] | None]:
    raw = _raw(root)
    raw["steps"]["tail2"] = {"type": "workflow", "document": "tail.json"}
    return raw, None, None


def _twin_control_beside_a_nested_workflow(
    root: Path,
) -> tuple[dict[str, Any], Any, dict[str, str] | None]:
    """The outer engages the controls layer on its own steps; the nested
    workflow's steps are not held to it (rule 14 runs over the outer's own)."""
    raw = _scan_outer(root)
    raw["steps"]["probe"] = {
        "type": "intervention_protocol",
        "document": SCAN,
        "waive": {"matched_random": "no_fit"},
    }
    return raw, _Rows(), None


TWINS: dict[
    str, Callable[[Path], tuple[dict[str, Any], Any, dict[str, str] | None]]
] = {
    "two_levels": _twin_two_levels,
    "set_on_inner_document_step": _twin_set_on_inner_document_step,
    "receipt_on_inner_decision": _twin_receipt_on_inner_decision,
    "conditional_gating_the_workflow_step": _twin_conditional_gating_the_workflow_step,
    "conditional_sides_disjoint_across_the_boundary": (
        _twin_conditional_sides_disjoint_across_the_boundary
    ),
    "after_the_workflow_step": _twin_after_the_workflow_step,
    "reference_to_an_inner_step": _twin_reference_to_an_inner_step,
    "inner_fan_out": _twin_inner_fan_out,
    "receipt_on_the_workflow_step": _twin_receipt_on_the_workflow_step,
    "same_document_twice": _twin_same_document_twice,
    "control_beside_a_nested_workflow": _twin_control_beside_a_nested_workflow,
}


@pytest.mark.parametrize("twin", sorted(TWINS))
def test_the_valid_twin_loads_and_runs_to_completed(tmp_path: Path, twin: str) -> None:
    """Every refusal has a twin that passes — a two-level
    nest, a `set` on an inner document step, a receipt on an inner decision
    and on the `workflow` step, a conditional gating the whole nested
    workflow, a reference to and an `after` on inner steps, an inner fan-out,
    one document nested twice, a control beside a nested workflow — each
    loading and running to `completed` on CPU."""
    root = _tree(tmp_path)
    raw, engine, expected = TWINS[twin](root)
    workflow_dir = root / "deep" if twin == "two_levels" else root
    loaded = load_workflow(raw, _env(root), workflow_dir=workflow_dir)
    assert loaded.nested, "a twin nests"
    result = _run(loaded, root, tmp_path / "runs", engine)
    statuses = _statuses(result)
    if expected is None:
        assert set(statuses.values()) == {"completed"}, statuses
    else:
        assert statuses == expected
    assert set(statuses) == set(loaded.order)
    assert terminal(result.run_root / EVENTS_FILE)
    assert set(result.manifest["nested"]) == {
        n for n, s in loaded.document.steps.items() if isinstance(s, WorkflowStep)
    }
    for name in loaded.nested:
        assert name not in statuses  # a container has no status


def test_every_refusal_has_a_twin() -> None:
    joined = " ".join(TWINS)
    for word in (
        "set",
        "receipt",
        "conditional",
        "after",
        "reference",
        "fan_out",
        "control",
        "two_levels",
    ):
        assert word in joined, word
