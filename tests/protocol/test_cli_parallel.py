"""``--parallel`` (``docs/model_parallelism.md`` §2, §8.5, §9): the flag on
``causalab run`` and ``causalab dry-run``, plumbed like ``--device`` — to the
reference engine's constructor, never into the document — the receipt's
``execution.parallel`` block, and the dry run's ``parallel`` fact.

Geometry is execution, never identity (spec §8): a document run at ``tp=4``
and the same document run on one device carry the same canonical form, the
same ``document_digest`` and the same point digests, and differ in their
receipts under ``execution.parallel`` and nowhere else. Beside
``test_cli.py``'s ``--device`` / ``--batch-rows`` tests, which this file
mirrors, and ``test_dry_run.py``'s offline probe, which it reuses.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from causalab.cli import main
from causalab.io.env import ResolutionEnv
from causalab.neural.shared.engine_router import load_engine, route
from causalab.neural.shared.receipt import (
    emit_run_events,
    run_events,
    write_run_record,
)
from causalab.neural.shared.sweep import signed_steps
from causalab.protocol.compiled import CompiledProtocol
from causalab.protocol.engine import Engine, RunContext, RunResult, StepRecord
from causalab.protocol.lowering import point_count
from causalab.protocol.parallel import ONE, ParallelGeometry, format_geometry
from causalab.protocol.pipeline import compile_protocol
from causalab.protocol.publish import LAUNCHERS, is_joiner, point_shard
from causalab.protocol.receipt import execution_record
from causalab.protocol.reports import dry_run
from causalab.protocol.rules.errors import ProtocolError
from causalab.protocol.schema import COMPONENTS
from causalab.workflow.runner import QUALIFICATION_IDENTITY_FIELDS

from tests.protocol._env import CORPUS_DIR, FIXTURES
from tests.protocol.test_dry_run import (
    _argv as _dry_run_argv,  # pyright: ignore[reportPrivateUsage]
)
from tests.protocol.test_dry_run import _offline  # pyright: ignore[reportPrivateUsage]

pytestmark = pytest.mark.unit

INTERCHANGE = "02_interchange_im.json"
HOOKS_MODULE = "causalab.neural.engines.pytorch_hooks"
NNSIGHT_MODULE = "causalab.neural.engines.nnsight_tracing"

#: What the receipt says at world 1: every axis one, this process alone.
SOLO = {
    "data": 1,
    "data_mode": "points",
    "pipeline": 1,
    "context": 1,
    "tensor": 1,
    "expert": 1,
    "world": 1,
    "launcher": "solo",
}


def _stub_execute(
    engine: "_Hooks | _Nnsight", compiled: CompiledProtocol, run: RunContext
) -> RunResult:
    """What a stand-in owes the doors (``tests/_helpers/stub_engine.py``),
    plus the reference engine's two data-parallel moves
    (``neural/shared/execution.py``): the campaign's steps enumerated and
    signed on every rank, this replica's contiguous shard of them
    ([`point_shard`][causalab.protocol.publish.point_shard] — which refuses more
    replicas than points by name), and the receipt and stream written by the
    joiner alone, when asked. Executes nothing."""
    steps = signed_steps(
        compiled, run.env, indices=run.indices(point_count(compiled.axes))
    )
    geometry = getattr(engine, "parallel", ONE)
    publisher = run.publisher
    if publisher.replicas > 1 and geometry.data_mode != "rows":
        shard = point_shard(range(len(steps)), publisher.replica, publisher.replicas)
    else:
        shard = range(len(steps))
    engine.steps = tuple(step.record for step in steps)
    engine.shard = shard
    if not (run.record and is_joiner(publisher)):
        return RunResult(files={}, steps=engine.steps)
    write_run_record(compiled, run, engine, engine.steps)
    log = run_events(compiled, run, engine.steps)
    result = RunResult(files={}, steps=engine.steps)
    emit_run_events(log, result)
    return result


class _Hooks(Engine):
    """Stands in for the reference engine: records its constructor's
    ``parallel`` and the handoff (the compiled document and its run
    context), and executes nothing."""

    last: "_Hooks | None" = None

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
        parallel: ParallelGeometry = ONE,
    ) -> None:
        self.device = device
        self.batch_rows = batch_rows
        self.parallel = parallel
        self.compiled: CompiledProtocol | None = None
        self.run: RunContext | None = None
        #: the steps this engine signed, and the shard of them it ran —
        #: ``None`` until ``execute`` decided them
        self.steps: tuple[StepRecord, ...] | None = None
        self.shard: range | None = None
        type(self).last = self

    def execute(self, compiled, run):
        self.compiled = compiled
        self.run = run
        return _stub_execute(self, compiled, run)


class _HooksWithoutParallel(_Hooks):
    """A stand-in that predates the flag: a run without ``--parallel`` must
    build it, so the kwarg is passed only when a geometry was asked for."""

    def __init__(self, *, device: str = "cpu", batch_rows: int | None = None) -> None:
        super().__init__(device=device, batch_rows=batch_rows)


class _Nnsight(Engine):
    last: "_Nnsight | None" = None

    name = "nnsight"
    capabilities = frozenset({"paired_forward", "full_logits", "pytorch_fn_local"})
    components = frozenset(COMPONENTS)
    writable_components = frozenset(COMPONENTS)
    is_local = True

    def __init__(self, *, device: str = "cpu") -> None:
        self.device = device
        self.steps: tuple[StepRecord, ...] | None = None
        self.shard: range | None = None
        type(self).last = self

    def execute(self, compiled, run):
        return _stub_execute(self, compiled, run)


def _install(
    monkeypatch: pytest.MonkeyPatch, module: str, attr: str, cls: type
) -> None:
    stub = types.ModuleType(module)
    setattr(stub, attr, cls)
    monkeypatch.setitem(sys.modules, module, stub)


@pytest.fixture
def engines(monkeypatch: pytest.MonkeyPatch) -> tuple[type[_Hooks], type[_Nnsight]]:
    _install(monkeypatch, HOOKS_MODULE, "PytorchHooksEngine", _Hooks)
    _install(monkeypatch, NNSIGHT_MODULE, "NnsightEngine", _Nnsight)
    _Hooks.last = None
    _Nnsight.last = None
    return _Hooks, _Nnsight


class _JoinedRankZero:
    """The publisher of joined rank 0 with the process-group join stubbed
    at the system boundary: this process publishes and joins, its gather is
    the identity, and the receipt's word is ``joined``."""

    launcher = "joined"
    replica = 0
    publish = True

    def __init__(self, replicas: int) -> None:
        self.replicas = replicas

    def gather(self, payload):
        return (payload,)


@pytest.fixture
def as_joined_rank_zero(monkeypatch: pytest.MonkeyPatch):
    """Preset the environment ``torchrun`` would for rank 0 of ``world`` and
    stub the launcher's group entry — a world above 1 is a launched world
    (§3), and a lone test process is rank 0 of one only by saying so."""
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


def _run_argv(artifacts_root: Path, out: Path, *extra: str) -> list[str]:
    """``run`` of the interchange document with the receipt asked for
    (``--record``: the receipt is the engine's to write, and off by default)
    through the ``auto`` engine, which is the reference engine."""
    return [
        "run",
        str(CORPUS_DIR / INTERCHANGE),
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


def _record(out: Path) -> dict[str, Any]:
    return json.loads((out / "protocol.json").read_text())


# --------------------------------------------------------------------------- #
# the flag reaches the engine, never the document
# --------------------------------------------------------------------------- #


def test_parallel_goes_to_the_reference_engine(
    engines, artifacts_root, tmp_path, as_joined_rank_zero
) -> None:
    """§2: ``PytorchHooksEngine(parallel=ParallelGeometry(...))`` — the CLI's
    grammar, parsed once, handed to the constructor beside ``device``. A
    world above 1 is a launched world (§3), so this process runs as joined
    rank 0 of it, and ``--device cuda`` is ``cuda:LOCAL_RANK`` on a rank."""
    as_joined_rank_zero(4)
    code = main(
        _run_argv(
            artifacts_root, tmp_path, "--parallel", "tp=2,ep=4", "--device", "cuda"
        )
    )
    assert code == 0
    hooks = engines[0].last
    assert hooks is not None
    assert hooks.parallel == ParallelGeometry(tensor=2, expert=4)
    assert hooks.device == "cuda:0"
    assert hooks.compiled is not None and hooks.run is not None
    assert "parallel" not in json.dumps(hooks.compiled.canonical)
    assert hooks.run.publisher.launcher == "joined"


def test_without_the_flag_the_engine_is_built_as_before(
    monkeypatch, artifacts_root, tmp_path
) -> None:
    """No ``--parallel``: the constructor is not handed the kwarg, so a
    stand-in that predates it still builds — the ``fit_rows`` discipline."""
    _install(monkeypatch, HOOKS_MODULE, "PytorchHooksEngine", _HooksWithoutParallel)
    _install(monkeypatch, NNSIGHT_MODULE, "NnsightEngine", _Nnsight)
    assert main(_run_argv(artifacts_root, tmp_path)) == 0
    assert _HooksWithoutParallel.last is not None
    assert _HooksWithoutParallel.last.parallel == ONE


def test_a_malformed_geometry_is_refused_as_p4_naming_the_axis(
    engines, artifacts_root, tmp_path, capsys
) -> None:
    code = main(_run_argv(artifacts_root, tmp_path, "--parallel", "tp=four"))
    assert code == 1
    err = capsys.readouterr().err
    assert err.startswith("refused: [P4]")
    assert "--parallel.tensor" in err
    assert engines[0].last is None, "no engine is built for a refused geometry"


def test_a_world_above_one_with_a_pinned_nnsight_engine_is_refused(
    engines, artifacts_root, tmp_path, capsys
) -> None:
    """Fail closed, like ``--batch-rows``: the nnsight engine is single-device
    (§8.5), and a pinned nnsight run would drop the geometry while the
    receipt said otherwise."""
    with pytest.raises(SystemExit) as exit_info:
        main(
            _run_argv(
                artifacts_root, tmp_path, "--engine", "nnsight", "--parallel", "tp=2"
            )
        )
    assert exit_info.value.code == 2
    err = capsys.readouterr().err
    assert "--engine nnsight" in err and "--parallel" in err
    # world 1 is no geometry at all: the pinned nnsight run proceeds
    assert (
        main(
            _run_argv(
                artifacts_root, tmp_path, "--engine", "nnsight", "--parallel", "tp=1"
            )
        )
        == 0
    )
    assert engines[1].last is not None


def test_under_auto_a_world_above_one_builds_the_reference_engine_alone(
    engines,
) -> None:
    """§8.5: ``auto`` resolves to the reference engine, built with the
    geometry; the nnsight engine is single-device, so asking for it by name
    above a world of 1 is refused as ``P4`` naming the flag and the
    geometry, while at world 1 it builds as before."""
    geometry = ParallelGeometry(tensor=2)
    built = route("auto", device="cpu", parallel=geometry)
    assert built.name == "pytorch_hooks"
    assert built.parallel == geometry
    with pytest.raises(ProtocolError) as info:
        load_engine("nnsight", device="cpu", parallel=geometry)
    assert info.value.code == "P4" and info.value.path == "--parallel"
    assert "nnsight" in str(info.value)
    assert format_geometry(geometry) in str(info.value)
    assert load_engine("nnsight", device="cpu", parallel=ONE).name == "nnsight"
    assert load_engine("nnsight", device="cpu").name == "nnsight"
    assert route("auto", device="cpu").name == "pytorch_hooks"


# --------------------------------------------------------------------------- #
# the receipt: execution.parallel, the one recorder
# --------------------------------------------------------------------------- #


def test_the_receipt_records_the_geometry_under_execution(
    engines, artifacts_root, tmp_path, as_joined_rank_zero
) -> None:
    """§9: ``execution.parallel`` beside ``batch_rows`` — the five axes, the
    world and the launcher, the real word of the process's launch — and
    nothing about it in a digest. Joined rank 0 of a world of 8; no data
    axis, since the interchange document has one point and ``dp=2`` over it
    is refused by name (more replicas than points)."""
    as_joined_rank_zero(8)
    assert (
        main(_run_argv(artifacts_root, tmp_path, "--parallel", "pp=2,tp=2,ep=4")) == 0
    )
    record = _record(tmp_path)
    assert record["execution"]["parallel"] == {
        "data": 1,
        "data_mode": "points",
        "pipeline": 2,
        "context": 1,
        "tensor": 2,
        "expert": 4,
        "world": 8,
        "launcher": "joined",
    }
    assert "parallel" not in json.dumps(record["canonical"])
    assert "parallel" not in json.dumps(record["points"])


def test_the_receipt_says_solo_at_world_one(engines, artifacts_root, tmp_path) -> None:
    assert main(_run_argv(artifacts_root, tmp_path)) == 0
    assert _record(tmp_path)["execution"]["parallel"] == SOLO
    assert LAUNCHERS == ("solo", "spawned", "joined")


def test_the_receipt_reads_the_geometry_off_the_engine(engines) -> None:
    """The recorder reads the engine (an engine declaring nothing is solo),
    so the receipt cannot claim a geometry the engine does not hold."""
    assert execution_record(_Nnsight())["parallel"] == SOLO
    assert execution_record(_Hooks(parallel=ParallelGeometry(pipeline=3)))[
        "parallel"
    ] == {
        **SOLO,
        "pipeline": 3,
        "world": 3,
    }


def test_geometry_is_execution_never_identity(
    engines, artifacts_root, tmp_path, as_joined_rank_zero
) -> None:
    """Spec §8: the same document with and without ``--parallel`` has one
    canonical form, one document digest and one point list; the two receipts
    differ under ``execution.parallel`` and nowhere else."""
    solo, sharded = tmp_path / "solo", tmp_path / "sharded"
    assert main(_run_argv(artifacts_root, solo)) == 0
    as_joined_rank_zero(8)
    assert main(_run_argv(artifacts_root, sharded, "--parallel", "tp=4,ep=8")) == 0
    a, b = _record(solo), _record(sharded)
    assert a["canonical"] == b["canonical"]
    assert a["document_digest"] == b["document_digest"]
    assert a["points"] == b["points"]
    assert json.dumps(a["canonical"], sort_keys=True) == json.dumps(
        b["canonical"], sort_keys=True
    )
    assert {k: v for k, v in a["execution"].items() if k != "parallel"} == {
        k: v for k, v in b["execution"].items() if k != "parallel"
    }
    assert a["execution"]["parallel"] != b["execution"]["parallel"]
    assert {k for k in a if a[k] != b[k]} == {"execution"}


def test_qualification_identity_excludes_the_whole_execution_block(engines) -> None:
    """§9: ``QUALIFICATION_IDENTITY_FIELDS`` stays closed — a control
    qualified at one geometry qualifies its target at any other."""
    execution = execution_record(_Hooks(parallel=ParallelGeometry(tensor=2)))
    assert "parallel" in execution
    assert set(QUALIFICATION_IDENTITY_FIELDS).isdisjoint(execution)
    assert set(QUALIFICATION_IDENTITY_FIELDS).isdisjoint(execution["parallel"])


# --------------------------------------------------------------------------- #
# the workflow CLI shares the engine router
# --------------------------------------------------------------------------- #


def test_the_workflow_cli_hands_the_geometry_to_the_same_engine(
    engines, monkeypatch, tmp_path, as_joined_rank_zero
) -> None:
    """§2: the workflow runner receives the one engine ``--engine`` names,
    built by the same router with the same geometry — on every rank of the
    launched world (§3, §11; ``tests/workflow/test_cli_parallel.py`` is the
    launch itself), here joined rank 0 of four."""
    import causalab.workflow
    import causalab.workflow.document

    as_joined_rank_zero(4)
    captured: dict[str, Any] = {}

    def fake_load_workflow(path, env, *, overrides):
        return types.SimpleNamespace(document=types.SimpleNamespace(steps={}))

    def fake_run_workflow(loaded, env, out, engine, **kwargs):
        captured["engine"] = engine
        return types.SimpleNamespace(manifest={"steps": {}}, run_root=out)

    monkeypatch.setattr(causalab.workflow.document, "load_workflow", fake_load_workflow)
    monkeypatch.setattr(causalab.workflow, "run_workflow", fake_run_workflow)
    document = tmp_path / "workflow.json"
    document.write_text(json.dumps({"version": "1", "output_dir": "run", "steps": {}}))
    code = main(
        [
            "run",
            str(document),
            "--engine",
            "auto",
            "--out",
            str(tmp_path / "out"),
            "--parallel",
            "pp=2,tp=2",
        ]
    )
    assert code == 0
    hooks = captured["engine"]
    assert isinstance(hooks, _Hooks)
    assert hooks.parallel == ParallelGeometry(pipeline=2, tensor=2)


# --------------------------------------------------------------------------- #
# dry-run: the geometry checked before any weights
# --------------------------------------------------------------------------- #


def _compiled(env: ResolutionEnv):
    return compile_protocol(CORPUS_DIR / INTERCHANGE, env=env)


def test_the_dry_run_reports_the_geometry_and_its_refusals(env) -> None:
    """The ``parallel`` fact: the geometry and ``check``'s refusals against
    the entry the compile resolved (the dense Llama-3.1-8B here), the
    refusals also in ``report.refusals`` so ``ok`` is honest."""
    accepted = dry_run(_compiled(env), env, parallel=ParallelGeometry(tensor=2))
    assert accepted.parallel is not None
    assert accepted.parallel.geometry == ParallelGeometry(tensor=2)
    assert accepted.parallel.refusals == ()
    assert accepted.ok

    refused = dry_run(
        _compiled(env), env, parallel=ParallelGeometry(tensor=2, expert=2)
    )
    assert refused.parallel is not None
    assert (
        refused.parallel.refusals
        and "--parallel.expert" in refused.parallel.refusals[0]
    )
    assert not refused.ok
    assert [r.code for r in refused.refusals] == ["P4"]
    assert refused.refusals[0].message == refused.parallel.refusals[0]


def test_the_dry_run_refuses_context_on_a_hybrid_tower_unless_waived(
    env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§8.4: the same dense document resolved as a hybrid tower (a
    ``linear_attention`` layer declared) is refused at ``cp=2`` by name, the
    waiver lifts the refusal, and a malformed waiver is itself refused."""
    import dataclasses as dc

    from causalab.protocol.parallel import CONTEXT_EXPERIMENTAL_VARIABLE
    from causalab.protocol.registry import get_model_info

    def hybrid(key: str):
        info = get_model_info(key)
        streams = ("full_attention",) * (info.num_layers - 1) + ("linear_attention",)
        return dc.replace(info, layer_types=streams)

    hybrid_env = dc.replace(env, model_info=hybrid)
    monkeypatch.delenv(CONTEXT_EXPERIMENTAL_VARIABLE, raising=False)
    refused = dry_run(
        _compiled(hybrid_env), hybrid_env, parallel=ParallelGeometry(context=2)
    )
    assert refused.parallel is not None and not refused.ok
    (text,) = refused.parallel.refusals
    assert text.startswith("--parallel.context:") and "linear_attention" in text
    assert CONTEXT_EXPERIMENTAL_VARIABLE in text

    dense = dry_run(_compiled(env), env, parallel=ParallelGeometry(context=2))
    assert dense.parallel is not None and dense.parallel.refusals == ()

    monkeypatch.setenv(CONTEXT_EXPERIMENTAL_VARIABLE, "1")
    waived = dry_run(
        _compiled(hybrid_env), hybrid_env, parallel=ParallelGeometry(context=2)
    )
    assert waived.parallel is not None and waived.parallel.refusals == ()

    monkeypatch.setenv(CONTEXT_EXPERIMENTAL_VARIABLE, "maybe")
    with pytest.raises(ProtocolError) as err:
        dry_run(_compiled(hybrid_env), hybrid_env, parallel=ParallelGeometry(context=2))
    assert CONTEXT_EXPERIMENTAL_VARIABLE in str(err.value)


def test_the_dry_run_answers_for_an_entry_that_declares_no_plan(env) -> None:
    """``ModelInfo.parallel_plan`` defaults to ``None`` (an out-of-tree
    ``register_model`` entry that declares none): the dry run reports —
    nothing unapplied, the memory estimate undecided by name — rather than
    dereferencing the missing plan under ``tp > 1``."""
    import dataclasses as dc

    from causalab.protocol.registry import get_model_info

    def planless(key: str):
        return dc.replace(get_model_info(key), parallel_plan=None)

    planless_env = dc.replace(env, model_info=planless)
    report = dry_run(
        _compiled(planless_env), planless_env, parallel=ParallelGeometry(tensor=2)
    )
    assert report.parallel is not None
    assert report.parallel.unapplied == ()
    assert report.parallel.memory is None
    # undecided by name (the cache is checked before the plan, so the reason
    # here is the uncached checkpoint's), never a dereference of the plan
    assert report.parallel.memory_undecided is not None
    assert "memory" in report.undecided_topics


def test_the_dry_run_names_the_vocabulary_row_it_did_not_apply(env) -> None:
    """§6.1/§11: a config's ``embedding_rowwise`` row is dropped at
    derivation, the embedding whole on every rank; under ``tp > 1`` the
    dry-run's parallel fact says so by name, and says nothing at ``tp=1``
    (nothing would have been sharded)."""
    import dataclasses as dc

    from causalab.protocol.registry import ParallelPlan, get_model_info

    def with_vocab_row(key: str):
        info = get_model_info(key)
        plan = ParallelPlan(
            info.parallel_plan.rows, {"embed_tokens": "embedding_rowwise"}
        )
        return dc.replace(info, parallel_plan=plan)

    vocab_env = dc.replace(env, model_info=with_vocab_row)
    sharded = dry_run(
        _compiled(vocab_env), vocab_env, parallel=ParallelGeometry(tensor=2)
    )
    assert sharded.parallel is not None and sharded.parallel.refusals == ()
    assert sharded.parallel.unapplied == ("embed_tokens (embedding_rowwise)",)

    replicated = dry_run(
        _compiled(vocab_env), vocab_env, parallel=ParallelGeometry(expert=1)
    )
    assert replicated.parallel is not None and replicated.parallel.unapplied == ()
    plain = dry_run(_compiled(env), env, parallel=ParallelGeometry(tensor=2))
    assert plain.parallel is not None and plain.parallel.unapplied == ()


def test_without_a_geometry_the_dry_run_has_no_parallel_fact(env) -> None:
    assert dry_run(_compiled(env), env).parallel is None


def test_the_dry_run_cli_prints_the_fact_offline(artifacts_root) -> None:
    """Through the real CLI, in a subprocess: exit 0 and the ``parallel``
    line for an accepted geometry, exit 1 naming the axis for a refused one
    — and torch never imported either way. The assertions read the geometry
    line's prefix only, so they hold whether or not this machine caches the
    checkpoint; extending them to the ``: accepted`` suffix or the memory
    lines needs ``HF_HUB_CACHE`` pointed at an empty directory, as
    ``test_dry_run_memory.py`` does."""
    accepted = _offline(
        _dry_run_argv(INTERCHANGE, artifacts_root, "--parallel", "tp=2")
    )
    assert accepted["code"] == 0, accepted["err"]
    assert "parallel  dp=1,pp=1,cp=1,tp=2,ep=1 (world 2)" in accepted["out"]
    assert not accepted["torch"]

    refused = _offline(_dry_run_argv(INTERCHANGE, artifacts_root, "--parallel", "ep=2"))
    assert refused["code"] == 1
    assert "--parallel.expert" in refused["err"] and "[P4]" not in refused["out"]
    assert "parallel  dp=1,pp=1,cp=1,tp=1,ep=2 (world 2)" in refused["out"]
    assert not refused["torch"]

    malformed = _offline(
        _dry_run_argv(INTERCHANGE, artifacts_root, "--parallel", "tp=0")
    )
    assert malformed["code"] == 1
    assert (
        "refused: [P4]" in malformed["err"] and "--parallel.tensor" in malformed["err"]
    )
    assert not malformed["torch"]


def test_the_dry_run_cli_prints_no_parallel_line_without_the_flag(
    artifacts_root, capsys
) -> None:
    assert main(_dry_run_argv(INTERCHANGE, artifacts_root)) == 0
    assert "parallel " not in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# the launch (§3): a world above 1 spawns or joins before any engine is built
# --------------------------------------------------------------------------- #


def test_a_world_above_one_with_no_group_spawns_and_builds_no_engine(
    engines, artifacts_root, tmp_path, monkeypatch
) -> None:
    """The parent of a spawn re-launches ``main`` with the same argv —
    recorded here through the launcher's ``spawn`` seam — and exits with the
    children's status without building an engine or loading anything."""
    from causalab.neural.shared.parallel import launcher

    for name in ("WORLD_SIZE", "RANK", "LOCAL_RANK", launcher.LAUNCHER_VARIABLE):
        monkeypatch.delenv(name, raising=False)
    spawned: dict[str, Any] = {}

    def fake_spawn(geometry, argv, *, entry=None, device=None):
        spawned["geometry"] = geometry
        spawned["argv"] = list(argv)
        return 7

    monkeypatch.setattr(launcher, "spawn", fake_spawn)
    argv = _run_argv(artifacts_root, tmp_path, "--parallel", "dp=2")
    assert main(argv) == 7
    assert spawned == {"geometry": ParallelGeometry(data=2), "argv": argv}
    assert engines[0].last is None, "the parent builds no engine"
    assert not (tmp_path / "protocol.json").exists()


def test_a_group_disagreeing_with_the_geometry_is_refused(
    engines, artifacts_root, tmp_path, monkeypatch, capsys
) -> None:
    for name, value in (("WORLD_SIZE", "4"), ("RANK", "0"), ("LOCAL_RANK", "0")):
        monkeypatch.setenv(name, value)
    code = main(_run_argv(artifacts_root, tmp_path, "--parallel", "dp=2"))
    assert code == 1
    err = capsys.readouterr().err
    assert err.startswith("refused: [P4]") and "WORLD_SIZE=4" in err
    assert engines[0].last is None


def test_a_joined_rank_that_does_not_join_the_campaign_writes_nothing(
    engines, artifacts_root, tmp_path, monkeypatch
) -> None:
    """Replica 1's publisher contributes its shard and writes no receipt
    even when one is asked for; the engine still receives the whole
    selection (§8.3: ``points`` is the campaign on every rank), signs every
    step of it, and derives its own contiguous shard."""
    from causalab.neural.shared.parallel import launcher

    class _Contributor(_JoinedRankZero):
        replica = 1

        def gather(self, payload):
            return None

    for name, value in (("WORLD_SIZE", "2"), ("RANK", "1"), ("LOCAL_RANK", "1")):
        monkeypatch.setenv(name, value)
    monkeypatch.delenv(launcher.LAUNCHER_VARIABLE, raising=False)
    monkeypatch.setattr(
        launcher, "enter", lambda launch, geometry, device: _Contributor(geometry.data)
    )
    monkeypatch.setattr(launcher, "leave", lambda publisher, status=0: None)
    # the fixture's 8-point scan has points for two replicas; the one-point
    # interchange document does not
    scan = (
        Path(__file__).resolve().parents[1]
        / "workflow"
        / "fixtures"
        / "fan_out"
        / "protocols"
        / "scan.json"
    )
    argv = [
        "run",
        str(scan),
        "--data-root",
        str(FIXTURES / "data"),
        "--artifacts-root",
        str(artifacts_root),
        "--engine",
        "auto",
        "--record",
        "--out",
        str(tmp_path),
        "--parallel",
        "dp=2",
    ]
    assert main(argv) == 0
    hooks = engines[0].last
    assert hooks is not None and hooks.run is not None
    assert hooks.run.publisher.replica == 1
    assert hooks.run.points is None, "the whole selection reaches every rank"
    assert hooks.steps is not None and len(hooks.steps) == 8
    assert hooks.shard == range(4, 8)
    assert not (tmp_path / "protocol.json").exists()
    assert not (tmp_path / "events.jsonl").exists()


def test_parse_refusals_are_protocol_errors(engines, artifacts_root, tmp_path) -> None:
    """The grammar's refusal is the same class every other refusal is, so
    the CLI's one ``except ProtocolError`` prints it."""
    from causalab.protocol.parallel import parse_geometry

    with pytest.raises(ProtocolError):
        parse_geometry("tp=2,tp=2")
