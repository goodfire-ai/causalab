"""One compiler, four doors (spec §9.1).

Two kinds of document are the most likely to validate under one resolution
context and execute under another: a document read from another directory with
**overrides**, and an artifact whose value is only known at run time —
**deferred** under workflow validation. An earlier version had four callers and four
compositions; this file compiles exactly that document through every door and
compares everything that identifies the compile:

* the canonical document and every point's canonical form;
* the point list — coordinates, in order;
* the document digest and every point digest.

(a) ``causalab run`` with ``--set``, (b) ``run_protocol`` handed the compiled
document, (c) the workflow runner's protocol step with the step's ``set``, and
(d) workflow validation of the same step with the artifact deferred to a
declared representative — all four byte-identical, and (d) equal to (c) exactly
when the representative equals what the producing step emits.

Around the acceptance test: the stage list is the spec's table (census — the
table's ``validate`` and ``route`` rows are ``pipeline.validate``'s and ``STAGES`` is the build alone), sugar resolves
before validation and hashing in that one order, several points' distinct
violations are reported together while a lone violation keeps its text byte
for byte, the compiler stays torch-free, and no module outside ``pipeline.py``
composes the pipeline (the grep-proof, as a test).
"""

from __future__ import annotations

import ast
import functools
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import pytest

from causalab.cli import main
from causalab.protocol import pipeline
from causalab.protocol import pipeline as compiler
from causalab.protocol.compiled import CompiledProtocol
from causalab.io.sources import DIAGNOSTIC_KINDS
from causalab.protocol.pipeline import (
    STAGES,
    check_engine,
    compile_protocol,
    read_document,
)
from causalab.protocol.engine import Engine, RunContext, RunResult, requires_campaign
from causalab.protocol.rules.errors import ValidationError, ValidationErrors
from causalab.io.env import FileArtifacts, FileDatasets, ResolutionEnv
from causalab.tasks import TASKS_ROOT
from causalab.protocol.pipeline import run_protocol
from causalab.protocol.schema import COMPONENTS, parse_document
from causalab.protocol.rules.document import validate_document
from causalab.workflow.document import load_workflow
from causalab.workflow.runner import run_workflow

from tests._helpers.stub_engine import stub_execute
from tests.protocol._docs import UNWRITTEN, base_doc, in_order
from tests.protocol._env import FIXTURES, steps_of
from tests.protocol._spec import INTERNALS, first_column, spec_section, spec_tables
from tests._helpers.paths import PROTOCOLS_DIR, WORKFLOWS_DIR


pytestmark = pytest.mark.unit


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "docs" / "intervention_protocol.md"
LOCATE = PROTOCOLS_DIR / "weekdays_locate_scan.json"

#: The layer the producing step emits — and, in the divergence fixture, the
#: representative it declares. `weekdays_locate_scan` sweeps this axis; binding
#: it to a step's value leaves the `positions.tap` sweep, so the compiled
#: campaign has two points and the point list is not trivially one entry.
LAYER = 18
#: The override every door applies: one of the two fields an override may
#: *create* (`CREATABLE_PATHS`), so it lands on a document that never authored
#: it and moves the digest in every path alike.
OVERRIDE = {"model.dtype": "bf16"}


# --------------------------------------------------------------------------- #
# the divergence fixture
# --------------------------------------------------------------------------- #


def _write_document(root: Path, *, axes: bool = False) -> Path:
    """`weekdays_locate_scan` in a *sibling directory* of the workflow, with
    the swept layer bound to a value another step emits
    (`{"artifact": "pick", …}`). With ``axes``, the `positions.tap` sweep is
    respelled as a `rows` axis of the §3.2 group (one field per row, the same
    two specs) referenced at the entry — the same two points, through the
    `axes` stage."""
    document: dict[str, Any] = json.loads(LOCATE.read_text())
    document["method"]["sites"]["target"]["layers"] = {
        "artifact": "pick",
        "key": "layer",
    }
    if axes:
        specs = document["method"]["positions"]["tap"]["sweep"]
        document["method"]["positions"]["tap"] = {"axis": "tap.pos"}
        document = {
            "header": document["header"],
            "model": document["model"],
            "data": document["data"],
            "axes": {"tap": {"rows": [{"pos": spec} for spec in specs]}},
            "method": document["method"],
        }
    docs = root / "docs"
    docs.mkdir(parents=True)
    path = docs / "apply.json"
    path.write_text(json.dumps(document))
    return path


def _write_workflow(root: Path, *, representative: int, emitted: int) -> Path:
    """A `pick` script step that emits `layer` (declaring `representative`,
    writing `emitted`) and a protocol step over the document with the override
    as its `set`."""
    scripts = root / "scripts"
    scripts.mkdir(parents=True)
    (scripts / "pick.py").write_text(
        "import json\n\n\n"
        "def main(inputs, outputs):\n"
        f"    outputs['values'].write_text(json.dumps({{'layer': {emitted}}}))\n"
    )
    workflow = {
        "version": "1",
        "output_dir": "run",
        "steps": {
            "pick": {
                "type": "script",
                "script": {"path": "scripts/pick.py"},
                "inputs": {},
                "outputs": {
                    "values": {"file": "values.json", "keys": {"layer": representative}}
                },
            },
            "locate": {
                "type": "intervention_protocol",
                "document": "docs/apply.json",
                "set": dict(OVERRIDE),
            },
        },
    }
    path = root / "wf.json"
    path.write_text(json.dumps(workflow))
    return path


class _Recorder(Engine):
    """An engine that records every handoff — the compiled document and its
    run context — and executes nothing: the seam through which paths (a)–(c)
    hand a compile to an engine."""

    last: "_Recorder | None" = None

    name = "recorder"
    capabilities = frozenset(
        {"grad", "paired_forward", "full_logits", "pytorch_fn_local", "generate"}
    )
    components = frozenset(COMPONENTS)
    writable_components = frozenset(COMPONENTS)
    is_local = True

    def __init__(self, *, device: str = "cpu", batch_rows: int | None = None) -> None:
        # `batch_rows` is the reference engine's constructor bound (`--batch-rows`),
        # which `engine_router.load_engine` passes to whatever stands in
        # for the engine
        self.device = device
        self.batch_rows = batch_rows
        self.runs: list[tuple[CompiledProtocol, RunContext]] = []
        type(self).last = self

    def execute(self, compiled: CompiledProtocol, run: RunContext) -> RunResult:
        self.runs.append((compiled, run))
        return stub_execute(self, compiled, run)


@pytest.fixture
def recorder_engine(monkeypatch: pytest.MonkeyPatch) -> type[_Recorder]:
    """The CLI's lazily-imported reference engine, replaced by the recorder
    (the `sys.modules` stub `docs/TESTS.md` names)."""
    import types

    stub = types.ModuleType("causalab.neural.engines.pytorch_hooks")
    stub.PytorchHooksEngine = _Recorder  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "causalab.neural.engines.pytorch_hooks", stub)
    _Recorder.last = None
    return _Recorder


def _divergence(tmp_path: Path, *, axes: bool) -> dict[str, Any]:
    artifacts = tmp_path / "artifacts"
    shutil.copytree(FIXTURES / "artifacts", artifacts)
    (artifacts / "pick").mkdir()
    (artifacts / "pick" / "values.json").write_text(json.dumps({"layer": LAYER}))
    env = ResolutionEnv(
        datasets=FileDatasets(root=FIXTURES / "data", fallback_roots=(TASKS_ROOT,)),
        artifacts=FileArtifacts(root=artifacts),
    )
    return {
        "env": env,
        "artifacts": artifacts,
        "document": _write_document(tmp_path, axes=axes),
        "workflow": _write_workflow(tmp_path, representative=LAYER, emitted=LAYER),
    }


@pytest.fixture
def divergence(tmp_path: Path) -> dict[str, Any]:
    """The deferred-artifact case, standalone-resolvable too: the artifact the
    document reads exists under the artifacts root with the emitted value, so
    the CLI and `run_protocol` resolve it for real while the workflow defers
    it at load and reads the run tree at run."""
    return _divergence(tmp_path, axes=False)


@pytest.fixture
def divergence_axes(tmp_path: Path) -> dict[str, Any]:
    """The same case with the position sweep spelled as a §3.2 `rows` axis."""
    return _divergence(tmp_path, axes=True)


@pytest.fixture
def workflow_env(divergence: dict[str, Any]) -> ResolutionEnv:
    """The workflow's environment has no `pick/values.json` outside the run
    tree: what the protocol step reads at run is what the script step wrote."""
    artifacts = divergence["artifacts"].parent / "artifacts-workflow"
    shutil.copytree(FIXTURES / "artifacts", artifacts)
    return ResolutionEnv(
        datasets=FileDatasets(root=FIXTURES / "data", fallback_roots=(TASKS_ROOT,)),
        artifacts=FileArtifacts(root=artifacts),
    )


def _identity_of_handoff(
    compiled: CompiledProtocol, run: RunContext, record: Mapping[str, Any]
) -> dict[str, Any]:
    """Everything that identifies a compile, as it reached an engine and a
    run receipt: the canonical document, the canonical points, the
    coordinates, the digests — the points read off the compiled object at
    the indices the run context selects, as an engine reads them."""
    steps = steps_of(compiled, run.env)  # enumerated and signed as the engine does
    indices = run.indices(len(steps.points))
    return {
        "canonical": record["canonical"],
        "canonical_points": [steps.canonical[i] for i in indices],
        "coords": [dict(steps.points[i].coords) for i in indices],
        "document_digest": record["document_digest"],
        "point_digests": [steps.digests[i] for i in indices],
    }


def _identity_of_compiled(
    compiled: CompiledProtocol, env: ResolutionEnv
) -> dict[str, Any]:
    steps = steps_of(compiled, env)
    return {
        "canonical": compiled.canonical,
        "canonical_points": list(steps.canonical),
        "coords": [dict(c) for c in steps.coords],
        "document_digest": compiled.digests.document,
        "point_digests": list(steps.digests),
    }


def _cli_path(divergence: dict[str, Any], out: Path) -> dict[str, Any]:
    """(a) — the CLI's `run` verb with `--set`, through the recorder."""
    code = main(
        [
            "run",
            str(divergence["document"]),
            "--data-root",
            str(FIXTURES / "data"),
            "--artifacts-root",
            str(divergence["artifacts"]),
            "--set",
            "model.dtype=bf16",
            "--engine",
            "pytorch_hooks",
            "--out",
            str(out),
            "--record",
        ]
    )
    assert code == 0
    assert _Recorder.last is not None and len(_Recorder.last.runs) == 1
    record = json.loads((out / "protocol.json").read_text())
    return _identity_of_handoff(*_Recorder.last.runs[0], record)


def _run_protocol_path(divergence: dict[str, Any], out: Path) -> dict[str, Any]:
    """(b) — `run_protocol` handed the compiled document (the way a caller
    with overrides, or a campaign layer, reaches a run)."""
    env: ResolutionEnv = divergence["env"]
    compiled = compile_protocol(
        divergence["document"],
        env=env,
        base_dir=divergence["document"].parent,
        overrides=OVERRIDE,
        engine=None,
    )
    engine = _Recorder()
    run_protocol(compiled, env, engine, out, record=True)
    record = json.loads((out / "protocol.json").read_text())
    return _identity_of_handoff(*engine.runs[0], record)


def _workflow_paths(
    workflow: Path, env: ResolutionEnv, out: Path
) -> tuple[dict[str, Any], dict[str, Any], Any]:
    """(d) then (c): workflow validation of the step with the artifact
    deferred, then the runner's protocol step against the run tree."""
    loaded = load_workflow(workflow, env)
    inner = loaded.inner["locate"]
    validated = _identity_of_compiled(inner.compiled, env)
    engine = _Recorder()
    result = run_workflow(loaded, env, out, engine)
    step = result.manifest["steps"]["locate"]
    compiled, run = engine.runs[0]
    indices = run.indices(len(steps_of(compiled, env).points))
    executed = {
        "canonical": None,  # a step record carries digests, not the document
        "canonical_points": [steps_of(compiled, env).canonical[i] for i in indices],
        "coords": [dict(steps_of(compiled, env).points[i].coords) for i in indices],
        "document_digest": step["document_digest"],
        "point_digests": [steps_of(compiled, env).digests[i] for i in indices],
    }
    assert step["point_digests"] == executed["point_digests"]
    assert compiled.campaign_digest == step["document_digest"]
    return validated, executed, loaded


# --------------------------------------------------------------------------- #
# the acceptance test — one document through every door
# --------------------------------------------------------------------------- #


def test_the_four_doors_compile_the_divergence_case_byte_identically(
    divergence: dict[str, Any],
    workflow_env: ResolutionEnv,
    recorder_engine: type[_Recorder],
    tmp_path: Path,
) -> None:
    """A document in a sibling directory, an override and a deferred artifact:
    canonical form, point list and digests are the same through the CLI,
    `run_protocol`, the workflow runner and workflow validation — and
    validation's deferred compile equals the run's once the artifact resolves
    to the declared representative.

    Fails without the compiler because `compile_protocol` does not exist; and
    the property it asserts was not a property before — the base composed this
    document in three places and could not hand `run_protocol` an override.
    """
    cli = _cli_path(divergence, tmp_path / "out-cli")
    api = _run_protocol_path(divergence, tmp_path / "out-api")
    validated, executed, loaded = _workflow_paths(
        divergence["workflow"], workflow_env, tmp_path / "out-wf"
    )

    assert cli == api
    assert validated["canonical"] == cli["canonical"]
    for key in ("canonical_points", "coords", "document_digest", "point_digests"):
        assert cli[key] == validated[key] == executed[key], key
    # the fixture is the case it claims to be
    assert loaded.inner_digest_kind["locate"] == "authored"  # deferred at load
    assert cli["canonical"]["model"]["dtype"] == "bf16"  # the override landed
    assert len(cli["point_digests"]) == 2  # the positions sweep survived
    assert [set(c) for c in cli["coords"]] == [{"positions.tap"}] * 2  # layer bound
    assert cli["canonical"]["method"]["sites"]["target"]["layers"] == [LAYER]


def test_the_four_doors_compile_an_axes_document_byte_identically(
    divergence_axes: dict[str, Any],
    recorder_engine: type[_Recorder],
    tmp_path: Path,
) -> None:
    """The divergence case with its position sweep spelled as a §3.2 `rows`
    axis: the `axes` stage runs in every door alike, so canonical form (with
    the block), point list (one `axes.tap` coordinate per point) and digests
    agree across the CLI, `run_protocol`, the workflow runner and workflow
    validation — and the points are the two the sweep spelling gives."""
    artifacts = divergence_axes["artifacts"].parent / "artifacts-workflow"
    shutil.copytree(FIXTURES / "artifacts", artifacts)
    workflow_env = ResolutionEnv(
        datasets=FileDatasets(root=FIXTURES / "data", fallback_roots=(TASKS_ROOT,)),
        artifacts=FileArtifacts(root=artifacts),
    )
    cli = _cli_path(divergence_axes, tmp_path / "out-cli")
    api = _run_protocol_path(divergence_axes, tmp_path / "out-api")
    validated, executed, loaded = _workflow_paths(
        divergence_axes["workflow"], workflow_env, tmp_path / "out-wf"
    )
    assert cli == api
    assert validated["canonical"] == cli["canonical"]
    for key in ("canonical_points", "coords", "document_digest", "point_digests"):
        assert cli[key] == validated[key] == executed[key], key
    assert loaded.inner_digest_kind["locate"] == "authored"  # deferred at load
    # the record round-trips through sorted-key JSON, so compare as a set
    assert set(cli["canonical"]) == {"header", "model", "data", "axes", "method"}
    assert [tuple(spec) for spec in cli["canonical"]["axes"]["tap"]["rows"]] == [
        ("pos",)
    ] * 2
    assert cli["coords"] == [{"axes.tap": 0}, {"axes.tap": 1}]
    assert cli["canonical"]["method"]["sites"]["target"]["layers"] == [LAYER]
    # the points are the sweep spelling's points, to the digest
    plain = _run_protocol_path(
        _divergence(tmp_path / "plain", axes=False), tmp_path / "out-plain"
    )
    assert plain["point_digests"] == cli["point_digests"]
    assert plain["canonical_points"] == cli["canonical_points"]
    assert plain["document_digest"] != cli["document_digest"]  # the campaign says how


def test_the_deferred_compile_differs_exactly_when_the_representative_does(
    divergence: dict[str, Any], workflow_env: ResolutionEnv, tmp_path: Path
) -> None:
    """The other half of the acceptance test: validation's compile is
    honest about being deferred. Declare a representative the step does not
    emit and (d) no longer equals (c) — the run's identity is the real
    value's — while (c) still equals the standalone doors."""
    shutil.copytree(divergence["document"].parent, tmp_path / "other" / "docs")
    workflow = _write_workflow(
        tmp_path / "other", representative=LAYER - 1, emitted=LAYER
    )
    validated, executed, loaded = _workflow_paths(
        workflow, workflow_env, tmp_path / "out"
    )
    assert loaded.inner_digest_kind["locate"] == "authored"
    assert validated["document_digest"] != executed["document_digest"]
    assert validated["point_digests"] != executed["point_digests"]
    standalone = _run_protocol_path(divergence, tmp_path / "out-api")
    for key in ("canonical_points", "coords", "document_digest", "point_digests"):
        assert standalone[key] == executed[key], key


def test_run_protocol_compiles_a_path_through_the_same_function(
    divergence: dict[str, Any], tmp_path: Path
) -> None:
    """Handed a path, `run_protocol` compiles it with no overrides against the
    file's own directory — so a relative method reference resolves — and what
    reaches the engine is what `compile_protocol` returns for those inputs."""
    env: ResolutionEnv = divergence["env"]
    engine = _Recorder()
    run_protocol(divergence["document"], env, engine, tmp_path / "out")
    want = compile_protocol(
        divergence["document"],
        env=env,
        base_dir=divergence["document"].parent,
        overrides=None,
        engine=None,
    )
    compiled, run = engine.runs[0]
    assert run.points is None  # every point, no shard
    assert compiled.campaign_digest == want.digests.document
    assert list(steps_of(compiled, env).digests) == list(steps_of(want, env).digests)
    assert list(steps_of(compiled, env).canonical) == list(
        steps_of(want, env).canonical
    )


def test_a_read_prefix_compiles_as_the_source_does(
    divergence: dict[str, Any],
) -> None:
    """`read_document` is the compiler's first two stages on their own: handing
    its result back in compiles to the same digests as the source with the same
    overrides — one read, run in two halves."""
    env: ResolutionEnv = divergence["env"]
    document: Path = divergence["document"]
    prefix = read_document(document, document.parent, OVERRIDE)
    assert prefix.raw["model"]["dtype"] == "bf16"
    via_prefix = compile_protocol(
        prefix, env=env, base_dir=document.parent, overrides=None, engine=None
    )
    via_source = compile_protocol(
        document, env=env, base_dir=document.parent, overrides=OVERRIDE, engine=None
    )
    assert via_prefix.digests == via_source.digests
    assert via_prefix.canonical == via_source.canonical
    assert steps_of(via_prefix, env).canonical == steps_of(via_source, env).canonical


# --------------------------------------------------------------------------- #
# the order is data — and sugar resolves before validation and hashing
# --------------------------------------------------------------------------- #


#: The two rows of the spec's stage table that are not build stages: the
#: checklist over the points and the engine's shortfall are
#: ``pipeline.validate``'s, run on the built object. The table keeps the rows;
#: the census below holds ``STAGES`` to the table with exactly these two set
#: aside.
VALIDATE_ROWS = ("validate", "route")


def test_the_stage_list_is_the_spec_table() -> None:
    """The order is data: `STAGES` and the §9.1 table agree, in order — and
    every stage has an implementation. A new compiler stage is one entry and
    one row, inserted where the order says. The table's `validate` and `route`
    rows are `pipeline.validate`'s, so they are set aside here rather than expected in `STAGES`."""
    tables = spec_tables(spec_section("## 9.1 Compilation stages", INTERNALS))
    stage_table = next(t for t in tables if t[0][0].strip("` ") == "stage")
    documented = first_column(stage_table)
    assert set(VALIDATE_ROWS) <= set(documented)
    assert tuple(s for s in documented if s not in VALIDATE_ROWS) == STAGES
    assert not set(VALIDATE_ROWS) & set(STAGES)
    assert tuple(pipeline._STAGE) == STAGES  # pyright: ignore[reportPrivateUsage]
    assert STAGES.index("read") < STAGES.index("override")
    assert STAGES.index("override") < STAGES.index("resolve")
    assert "expand" not in STAGES  # the sweep is the engine's
    assert STAGES.index("gate") < STAGES.index("canonicalize")
    assert STAGES.index("canonicalize") < STAGES.index("digest")


def test_the_diagnostic_kinds_are_the_spec_table() -> None:
    tables = spec_tables(spec_section("## 9.1 Compilation stages", INTERNALS))
    kinds = next(t for t in tables if t[0][0].strip("` ") == "kind")
    assert set(first_column(kinds)) == set(DIAGNOSTIC_KINDS)
    assert len(DIAGNOSTIC_KINDS) == len(set(DIAGNOSTIC_KINDS))
    with pytest.raises(AssertionError):
        compiler.Diagnostic("not_a_kind", "x")  # type: ignore[arg-type]


#: Every output the §9.1 table names, and where a `CompiledProtocol` carries
#: it: its own field, or a read-only property over the fields that
#: replaced it (`canonical` → `explicit`, `digests` → `campaign_digest`,
#: `axes` → the axes and the lowered tree they index into — the sweep is the
#: engine's, so no points ride on the object — `capabilities`
#: derived from the representatives on first read).
OUTPUT_CARRIERS: dict[str, tuple[str, ...]] = {
    "canonical": ("explicit",),
    "axes": ("axes",),  # beside `tree` and `named_axes`, the tree they index into
    "data": ("data",),
    "artifacts": ("artifacts",),
    "capabilities": ("tree", "axes", "named_axes"),
    "digests": ("campaign_digest",),
    "diagnostics": ("diagnostics",),
    "positions": ("positions",),  # derived, not identity
}


def test_the_seven_outputs_are_the_spec_table() -> None:
    """The §9.1 table and the object are held together by a fixed map: every
    documented output is a field of `CompiledProtocol` or a property over
    the fields `OUTPUT_CARRIERS` names, in the table's order, and the field
    tuple is exactly the ten pinned below —
    so a new field or a new output is a deliberate edit in both places."""
    import dataclasses

    tables = spec_tables(spec_section("## 9.1 Compilation stages", INTERNALS))
    outputs = next(t for t in tables if t[0][0].strip("` ") == "output")
    documented = first_column(outputs)
    assert documented == tuple(OUTPUT_CARRIERS)
    fields = tuple(field.name for field in dataclasses.fields(CompiledProtocol))
    assert set(fields) == {
        "document",
        "explicit",
        "tree",
        "axes",
        "named_axes",
        "campaign_digest",
        "data",
        "artifacts",
        "diagnostics",
        "positions",
    }
    for name, carriers in OUTPUT_CARRIERS.items():
        assert set(carriers) <= set(fields), name
        if carriers != (name,):
            assert isinstance(
                getattr(CompiledProtocol, name), (property, functools.cached_property)
            ), name
    assert documented[-1] == "positions" == fields[-1]  # the derived one is last


def test_sugar_resolves_before_validation_and_hashing(env: ResolutionEnv) -> None:
    """The case the four-doors refactor exists for, named once: sweeps expand
    *before* validation (a violation in one point of two is found), overrides
    land *before* validation (an override that repairs a violation makes the
    document valid), and the digest is the overridden, expanded document's."""
    swept = base_doc()
    swept["method"]["sites"]["tgt"]["layers"] = {"sweep": [3, 99]}  # gpt2 has 12 layers
    with pytest.raises(ValidationError) as err:
        compile_protocol(swept, env=env)
    assert err.value.rule == 4
    assert "99" in str(err.value)

    broken = base_doc()
    broken["method"]["sites"]["tgt"]["layers"] = 99
    with pytest.raises(ValidationError):
        compile_protocol(broken, env=env)
    repaired = compile_protocol(broken, env=env, overrides={"sites.tgt.layers": 3})
    assert (
        repaired.digests.document
        == compile_protocol(base_doc(), env=env).digests.document
    )


# --------------------------------------------------------------------------- #
# ValidationErrors across points — and the lone refusal, byte for byte
# --------------------------------------------------------------------------- #


def _direct_text(raw: dict[str, Any]) -> str:
    """The refusal `validate_document` itself produces for one concrete point
    — the text the loader always reported."""
    with pytest.raises(ValidationError) as err:
        validate_document(parse_document(in_order(raw)))
    return str(err.value)


def _unswept_text(raw: dict[str, Any], env: ResolutionEnv) -> str:
    """The refusal one concrete point gets on its own through the whole load —
    for a refusal canonicalization raises rather than the checklist."""
    with pytest.raises(ValidationError) as err:
        compile_protocol(raw, env=env)
    assert not isinstance(err.value, ValidationErrors)
    return str(err.value)


def test_distinct_violations_across_points_are_reported_together(
    env: ResolutionEnv,
) -> None:
    """Two points breaking the checklist differently: the loader reported the
    first and stopped; the compiler reports both, as a `ValidationErrors`
    whose first entry is the loader's refusal, byte for byte."""
    doc = base_doc()
    doc["method"]["reads"]["v_cf"]["site"] = {"sweep": ["nope1", "nope2"]}
    with pytest.raises(ValidationErrors) as err:
        compile_protocol(doc, env=env)
    assert len(err.value.errors) == 2
    first = base_doc()
    first["method"]["reads"]["v_cf"]["site"] = "nope1"
    assert str(err.value.errors[0]) == _direct_text(first)
    assert "'nope2'" in str(err.value.errors[1])
    assert err.value.rule == 4  # a caller asserting the rule still can


def test_canonicalization_refusals_are_collected_across_points_too(
    env: ResolutionEnv,
) -> None:
    """A layer outside the model is refused by canonicalization, per point,
    not by the checklist — the compile collects those the same way. The base
    reported layer 99 and stopped; the first entry is that text, byte for byte."""
    doc = base_doc()
    doc["method"]["sites"]["tgt"]["layers"] = {"sweep": [99, 100]}  # gpt2 has 12 layers
    with pytest.raises(ValidationErrors) as err:
        compile_protocol(doc, env=env)
    assert len(err.value.errors) == 2
    point = base_doc()
    point["method"]["sites"]["tgt"]["layers"] = 99
    assert str(err.value.errors[0]) == _unswept_text(point, env)
    assert "layer 100" in str(err.value.errors[1])


def test_a_lone_violation_is_raised_as_itself(env: ResolutionEnv) -> None:
    """One point of two refuses: the refusal is the single `ValidationError`,
    not a one-entry collection, with the text unchanged."""
    doc = base_doc()
    doc["method"]["reads"]["v_cf"]["site"] = {"sweep": ["nope1", "tgt"]}
    with pytest.raises(ValidationError) as err:
        compile_protocol(doc, env=env)
    assert not isinstance(err.value, ValidationErrors)
    point = base_doc()
    point["method"]["reads"]["v_cf"]["site"] = "nope1"
    assert str(err.value) == _direct_text(point)


def test_identical_violations_across_points_collapse_to_one(
    env: ResolutionEnv,
) -> None:
    """Every point breaks the same rule the same way (the violation is on an
    unswept field): still one refusal, with the loader's exact text."""
    doc = base_doc()
    doc["method"]["reads"]["v_cf"]["site"] = "nope"
    doc["method"]["reads"]["logits"]["pos"] = {"sweep": [-1, -2]}
    with pytest.raises(ValidationError) as err:
        compile_protocol(doc, env=env)
    assert not isinstance(err.value, ValidationErrors)
    point = base_doc()
    point["method"]["reads"]["v_cf"]["site"] = "nope"
    assert str(err.value) == _direct_text(point)


# --------------------------------------------------------------------------- #
# the other outputs, and the valid-work twins of every report
# --------------------------------------------------------------------------- #


def test_capabilities_are_the_registry_derived_requirement(env: ResolutionEnv) -> None:
    compiled = compile_protocol(
        base_doc(), env=env, base_dir=None, overrides=None, engine=None
    )
    assert compiled.capabilities == requires_campaign(
        list(steps_of(compiled, env).documents)
    )
    assert compiled.capabilities  # a paired interchange needs something


def test_a_capability_shortfall_is_refused_when_capabilities_are_given(
    env: ResolutionEnv,
) -> None:
    """`engine_capabilities` given: a shortfall against what the document
    requires is *refused* (rule 13, the routing text; an earlier
    version reported it as a `capability_shortfall` diagnostic and returned). No
    engine given decides nothing and reports nothing, and an engine that
    covers the document compiles to the same digests — the valid-work twin.
    The diagnostic kind survives for the dry run that reports per candidate
    engine (`check_engine`)."""
    raw = base_doc()
    unknown = compile_protocol(raw, env=env, base_dir=None, overrides=None, engine=None)
    assert unknown.diagnostics == ()
    with pytest.raises(ValidationError) as err:
        compile_protocol(
            raw, env=env, base_dir=None, overrides=None, engine=frozenset()
        )
    assert err.value.rule == 13 and "paired_forward" in str(err.value)
    covered = compile_protocol(
        raw, env=env, base_dir=None, overrides=None, engine=unknown.capabilities
    )
    assert covered.diagnostics == ()
    assert covered.digests == unknown.digests  # the engine never moves a digest
    check_engine(covered, unknown.capabilities)
    with pytest.raises(ValidationError) as again:
        check_engine(covered, frozenset())
    assert again.value.rule == 13


def test_rule_13_is_decided_from_the_engine_capabilities(env: ResolutionEnv) -> None:
    """The base's `engine_is_local` is the compiler's `pytorch_fn_local`
    capability: a local engine may run a `pytorch_fn` write, a non-local one
    refuses it under rule 13, and no engine given decides nothing."""
    raw = base_doc()
    # a `pytorch_fn` names a `code` declaration, never a bare qualname (§2.8.1)
    raw["method"]["code"] = {
        "relu": {"locator": "tests.protocol._code_under_test.scale"}
    }
    raw["method"]["writes"]["patch"]["do"] = {"pytorch_fn": {"code": "relu"}}
    del raw["method"]["reads"]["v_cf"]
    del raw["method"]["intervened_models"][UNWRITTEN]  # nothing left to read on it
    del raw["data"]["counterfactual"]
    raw = in_order(raw)
    unknown = compile_protocol(raw, env=env, base_dir=None, overrides=None, engine=None)
    # an engine covering exactly what the document requires — routing on the
    # component entries is rule 13's shortfall too, so a set that offered
    # `pytorch_fn_local` alone would be refused for the components it lacks
    local = frozenset({"pytorch_fn_local"})
    compile_protocol(
        raw, env=env, base_dir=None, overrides=None, engine=unknown.capabilities | local
    )
    with pytest.raises(ValidationError) as err:
        compile_protocol(
            raw,
            env=env,
            base_dir=None,
            overrides=None,
            engine=unknown.capabilities - local,
        )
    assert err.value.rule == 13 and "pytorch_fn" in str(err.value)


def test_data_identities_and_schemas(env: ResolutionEnv) -> None:
    compiled = compile_protocol(
        base_doc(), env=env, base_dir=None, overrides=None, engine=None
    )
    assert set(compiled.data) == {"weekdays/data#train"}
    identity = compiled.data["weekdays/data#train"]
    assert identity.digest == compiled.canonical["data"]["base"]["digest"]
    assert identity.columns == tuple(env.datasets.columns("weekdays/data#train"))
    assert "input" in identity.columns


def test_artifacts_say_what_was_read_and_what_was_deferred(
    divergence: dict[str, Any], workflow_env: ResolutionEnv
) -> None:
    """Standalone, the value reference resolved for real; under workflow
    validation the same reference is marked deferred. The `file_path` half is
    covered by the shipped `weekdays_8b` workflow, whose `apply` step loads a
    bundle from the `fit` step's run tree."""
    env: ResolutionEnv = divergence["env"]
    document: Path = divergence["document"]
    standalone = compile_protocol(
        document, env=env, base_dir=document.parent, overrides=OVERRIDE, engine=None
    )
    assert [(a.path, a.reference, a.key, a.deferred) for a in standalone.artifacts] == [
        ("sites.target.layers", "pick", "layer", False)
    ]
    assert standalone.diagnostics == ()
    deferred = load_workflow(divergence["workflow"], workflow_env).inner["locate"]
    assert [a.deferred for a in deferred.compiled.artifacts] == [True]

    from tests.protocol._env import build_env, write_rot_fixture

    root = divergence["artifacts"].parent / "artifacts-shipped"
    shutil.copytree(FIXTURES / "artifacts", root)
    write_rot_fixture(root)
    shipped = load_workflow(WORKFLOWS_DIR / "weekdays.json", build_env(root))
    apply = shipped.inner["apply"].compiled
    assert apply is not None
    files = [a for a in apply.artifacts if a.key is None]
    assert files and all(a.deferred and a.identity is None for a in files)
    assert [d.kind for d in apply.diagnostics] == ["deferred_check"] * len(files)
    locate = shipped.inner["locate"].compiled
    assert locate is not None and locate.diagnostics == ()  # no deferral, no report


# --------------------------------------------------------------------------- #
# torch-free, and the grep-proof
# --------------------------------------------------------------------------- #


def test_the_compiler_imports_no_torch() -> None:
    """In a subprocess, since `tests/conftest.py` has torch loaded already."""
    probe = (
        "import sys, json\n"
        "import causalab.protocol.pipeline\n"
        "print(json.dumps(sorted(m for m in ('torch', 'numpy', 'safetensors') "
        "if m in sys.modules)))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, cwd=str(REPO)
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout.strip().splitlines()[-1]) == []


#: The calls that *are* the pipeline. A module calling two or more of them is
#: composing the sequence for itself — which is exactly what the pre-compiler
#: loader did and what the four callers must never do again.
PIPELINE_CALLS = frozenset(
    {
        "check_protocol_version",
        "resolve_artifact_fields",
        "validate_document",
        "canonicalize",
    }
)
#: The modules a pipeline call is attributed through when spelled `m.f(...)`.
PIPELINE_MODULES = frozenset(
    {
        "schema",
        "resolve",
        "sweep",
        "validate",
        "canonical",
        "_canonical",
        # the canonical form's home
        "explicit",
        "_explicit",
    }
)
#: Where the pipeline is composed — ``pipeline.py``; ``compile.py``
#: is a facade over its two verbs and calls none of these itself.
COMPILER = REPO / "causalab" / "protocol" / "pipeline.py"


def _pipeline_calls(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id in PIPELINE_CALLS:
            found.add(func.id)
        elif (
            isinstance(func, ast.Attribute)
            and func.attr in PIPELINE_CALLS
            and isinstance(func.value, ast.Name)
            and func.value.id in PIPELINE_MODULES
        ):
            found.add(func.attr)
    return found


def test_no_module_outside_the_compiler_composes_the_pipeline() -> None:
    """The grep-proof, as a test: outside `pipeline.py`, no module under
    `causalab/` calls more than one of the pipeline's functions. `loader.py`
    called all five before this; it calls none now (`schema.parse_document`
    calls the version check, which is one), and neither does the
    `compile.py` facade."""
    offenders = {
        str(path.relative_to(REPO)): sorted(calls)
        for path in sorted((REPO / "causalab").rglob("*.py"))
        if path != COMPILER and len(calls := _pipeline_calls(path)) > 1
    }
    assert not offenders, f"these modules compose the pipeline: {offenders}"
    assert _pipeline_calls(COMPILER) == PIPELINE_CALLS
