"""The engine contract and the handoff.

Every engine is entered the same way — ``execute(compiled, run)`` — with the
**same** [`CompiledProtocol`][causalab.protocol.compiled.CompiledProtocol] every door
reads and a [`RunContext`][causalab.protocol.engine.RunContext] that pre-decides
nothing the engine decides dynamically: no digest, no canonical form, no
per-point tuple rides on it. [`handoff`][causalab.protocol.pipeline.handoff]
validates against the routed engine before it executes, and a document run's
receipt is on disk before ``execute`` is entered — the invariant a crashed run
depends on, kept from ``test_compile_protocol`` / ``test_run_receipt``.
"""

from __future__ import annotations

import dataclasses
import inspect
import json
from pathlib import Path
from typing import Any

import pytest

import causalab.protocol
import causalab.protocol.engine as engine_module
from causalab.io.events import EVENTS_FILE, read_events
from causalab.neural.shared.receipt import emit_run_events
from causalab.protocol import RUN_RECORD_NAME, RunContext, pipeline, run_protocol
from causalab.protocol.pipeline import compile_protocol
from causalab.protocol.compiled import CompiledProtocol
from causalab.protocol.engine import Engine, RunResult, StepRecord, check_steps_signed
from causalab.protocol.parallel import ONE
from causalab.protocol.receipt import parallel_record
from causalab.protocol.rules.errors import ProtocolError, ValidationError
from causalab.protocol.pipeline import handoff
from causalab.io.env import ResolutionEnv
from causalab.protocol.schema import COMPONENTS

from tests._helpers.stub_engine import stub_record
from tests.protocol._docs import base_doc, in_order
from tests.protocol._env import steps_of


pytestmark = pytest.mark.unit

#: The §8 verbs a stub needs to serve ``base_doc``.
VERBS = frozenset({"grad", "paired_forward", "full_logits", "pytorch_fn_local"})


class _Stub(Engine):
    """Records every handoff in order with the other events a test injects,
    and executes nothing."""

    def __init__(
        self,
        events: list[str] | None = None,
        *,
        components: frozenset[str] = frozenset(COMPONENTS),
        on_execute: Any = None,
    ) -> None:
        self.name = "stub"
        self.capabilities = VERBS
        self.components = components
        self.writable_components = components
        self.is_local = True
        self.events = [] if events is None else events
        self.runs: list[tuple[CompiledProtocol, RunContext]] = []
        self.on_execute = on_execute

    def execute(self, compiled: CompiledProtocol, run: RunContext) -> RunResult:
        self.events.append("execute")
        self.runs.append((compiled, run))
        # the engine's half of the contract: the steps enumerated and signed,
        # the receipt and the stream written when asked, before any forward
        steps, log = stub_record(self, compiled, run)
        if self.on_execute is not None:
            self.on_execute(compiled, run)
        result = RunResult(files={}, steps=steps)
        if log is not None:
            emit_run_events(log, result)
        return result


def _compile(env: ResolutionEnv) -> CompiledProtocol:
    return compile_protocol(
        in_order(base_doc()), env=env, base_dir=None, overrides=None, engine=None
    )


# --------------------------------------------------------------------------- #
# the contract's two halves
# --------------------------------------------------------------------------- #


def test_the_run_context_carries_no_identity(tmp_path: Path) -> None:
    """The fields, exactly: where outputs land, the environment, the shard,
    and what the caller authored about execution — nothing the compiled
    object already is."""
    fields = {f.name: f for f in dataclasses.fields(RunContext)}
    assert set(fields) == {
        "output_dir",
        "env",
        "points",
        "execution",
        "decoding",
        "sink",
        "record",
        "publisher",
    }
    identity = {
        "points_raw",
        "canonical",
        "digests",
        "coords",
        "document_digest",
        "point_digests",
        "campaign_digest",
    }
    assert not set(fields) & identity
    assert RunContext.__dataclass_params__.frozen  # pyright: ignore[reportAttributeAccessIssue]
    assert not fields["sink"].compare, "a listener is not part of the run"
    run = RunContext(output_dir=tmp_path, env=None)  # pyright: ignore[reportArgumentType]
    assert run.points is None and run.decoding is None and run.sink is None
    assert dict(run.execution) == {}
    with pytest.raises(dataclasses.FrozenInstanceError):
        run.points = (0,)  # pyright: ignore[reportAttributeAccessIssue]
    # a listener changes no equality: two runs of one shard are the same run
    assert run == RunContext(output_dir=tmp_path, env=None, sink=print)  # pyright: ignore[reportArgumentType]


def test_the_shard_is_indices_and_none_is_every_point(tmp_path: Path) -> None:
    every = RunContext(output_dir=tmp_path, env=None)  # pyright: ignore[reportArgumentType]
    assert every.indices(3) == (0, 1, 2)
    shard = RunContext(output_dir=tmp_path, env=None, points=(2, 0))  # pyright: ignore[reportArgumentType]
    assert shard.indices(3) == (2, 0), "a shard's order is the run order"


def test_execute_takes_the_compiled_document_and_the_run_context() -> None:
    """One signature for every engine — the ABC's, and both shipped engines'
    (imported lazily: the reference engine pulls torch)."""
    assert list(inspect.signature(Engine.execute).parameters) == [
        "self",
        "compiled",
        "run",
    ]
    from causalab.neural.engines.nnsight_tracing.engine import NnsightEngine
    from causalab.neural.engines.pytorch_hooks.engine import PytorchHooksEngine

    for engine in (PytorchHooksEngine, NnsightEngine):
        assert list(inspect.signature(engine.execute).parameters) == [
            "self",
            "compiled",
            "run",
        ], engine.__name__
    # deleted, not shimmed: neither the module nor the package spells it
    assert not hasattr(engine_module, "ExecutionRequest")
    assert "ExecutionRequest" not in engine_module.__all__
    assert not hasattr(causalab.protocol, "ExecutionRequest")
    assert "ExecutionRequest" not in causalab.protocol.__all__


# --------------------------------------------------------------------------- #
# the handoff
# --------------------------------------------------------------------------- #


def test_handoff_validates_before_it_executes(
    env: ResolutionEnv, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``validate`` against the routed engine, with the data rules, then
    ``execute`` — in that order, with the same object."""
    events: list[str] = []
    seen: list[dict[str, Any]] = []
    original = pipeline.validate

    def spy(compiled: Any, engine: Any = None, **kwargs: Any) -> Any:
        events.append("validate")
        seen.append({"engine": engine, **kwargs})
        return original(compiled, engine, **kwargs)

    compiled = _compile(env)  # its own `validate` is not the handoff's
    monkeypatch.setattr(pipeline, "validate", spy)
    engine = _Stub(events)
    run = RunContext(output_dir=tmp_path, env=env)
    result = handoff(compiled, engine, run)
    assert events == ["validate", "execute"]
    assert seen == [{"engine": engine, "env": env, "data": True}]
    ((received, received_run),) = engine.runs
    assert received_run is run
    # the engine receives the object `resolve_positions` returned — the
    # same identity with the derived positions attached, nothing else changed
    assert received.positions is not None
    assert dataclasses.replace(received, positions=None) == compiled
    assert result.files == {} and result.steps == steps_of(compiled, env).records


def test_handoff_refuses_a_shortfall_before_execute_is_entered(
    env: ResolutionEnv, tmp_path: Path
) -> None:
    """An engine that cannot serve the document is refused under rule 13 —
    the routing text, generated from the missing capabilities — and its
    ``execute`` is never called."""
    events: list[str] = []
    engine = _Stub(events, components=frozenset())  # serves no component
    compiled = _compile(env)
    with pytest.raises(ValidationError) as err:
        handoff(compiled, engine, RunContext(output_dir=tmp_path, env=env))
    assert err.value.rule == 13
    assert "component:" in str(err.value)
    assert events == [] and engine.runs == []
    assert not tmp_path.exists() or not any(tmp_path.iterdir())


def test_an_engine_that_signs_no_steps_is_refused_at_the_door(
    env: ResolutionEnv, tmp_path: Path
) -> None:
    """The other half of the contract, guarded: the doors copy
    the step digests off ``result.steps``, so an engine that returns a
    ``RunResult`` without them — legal before, when the runner derived the
    digests from the compile — is refused by name instead of writing an empty
    record. ``handoff`` and both workflow doors call the same check."""

    class _Unsigned(_Stub):
        def execute(self, compiled: CompiledProtocol, run: RunContext) -> RunResult:
            self.events.append("execute")
            return RunResult(files={})  # steps == ()

    compiled = _compile(env)
    engine = _Unsigned()
    engine.name = "unsigned"
    run = RunContext(output_dir=tmp_path, env=env)
    with pytest.raises(ProtocolError, match=r"engine 'unsigned' signed 0 step"):
        handoff(compiled, engine, run)
    assert engine.events == ["execute"], "refused after execute, not before"
    # the check itself: the indices must be the shard's, in run order
    steps = steps_of(compiled, env).records
    ok = RunResult(files={}, steps=steps)
    check_steps_signed(ok, range(len(steps)), engine)
    with pytest.raises(ProtocolError, match="one StepRecord per selected index"):
        check_steps_signed(ok, range(len(steps) + 1), engine)
    # run order, not a set: the same two indices swapped are refused
    swapped = (StepRecord(1, {}, "b" * 64), StepRecord(0, {}, "a" * 64))
    with pytest.raises(ProtocolError, match=r"signed 2 step\(s\) \[1, 0\]"):
        check_steps_signed(RunResult(files={}, steps=swapped), range(2), engine)


# --------------------------------------------------------------------------- #
# the receipt is on disk before the first forward
# --------------------------------------------------------------------------- #


def test_the_engine_writes_the_receipt_after_signing_and_before_its_forward(
    env: ResolutionEnv, tmp_path: Path
) -> None:
    """The receipt is the engine's. At the ``run_protocol``
    door asked to record, the context says ``record`` and the engine — here the stub, playing
    the shared driver's opening moves — writes ``protocol.json`` and opens the
    stream once it has enumerated and signed the steps and before its first
    forward (``on_execute`` fires at that point); the receipt's points are
    exactly the steps the result returns, signed with the same hasher the
    compiler's pins were made with."""
    out = tmp_path / "out"
    observed: dict[str, Any] = {}

    def on_execute(compiled: CompiledProtocol, run: RunContext) -> None:
        receipt = out / RUN_RECORD_NAME
        observed["receipt"] = (
            json.loads(receipt.read_text()) if receipt.is_file() else None
        )
        observed["events"] = [r["event"] for r in read_events(out / EVENTS_FILE)]
        observed["run"] = run

    engine = _Stub(on_execute=on_execute)
    compiled = _compile(env)
    result = run_protocol(compiled, env, engine, out, record=True)

    receipt = observed["receipt"]
    assert receipt is not None, "no receipt before the engine's first forward"
    assert receipt["document_digest"] == compiled.campaign_digest
    steps = steps_of(compiled, env)
    assert [p["digest"] for p in receipt["points"]] == list(steps.digests)
    assert [p["coords"] for p in receipt["points"]] == [dict(c) for c in steps.coords]
    assert receipt["execution"] == {
        "batch_rows": None,
        "device": None,
        "fit_rows": None,
        "model_source": "loaded",
        "parallel": parallel_record(ONE),
    }
    assert observed["events"] == ["phase_started"]
    # the result carries what the engine signed: the receipt's points, exactly
    assert result.steps == steps.records
    assert [(s.index, s.digest) for s in result.steps] == [
        (p["index"], p["digest"]) for p in receipt["points"]
    ]
    # the context the engine saw: this door's, recording, and nothing of the
    # identity
    run = observed["run"]
    assert run == RunContext(output_dir=out, env=env, record=True) and run.record
    assert dataclasses.replace(engine.runs[0][0], positions=None) == compiled
    # the stream is finished once the engine has returned
    assert [r["event"] for r in read_events(out / EVENTS_FILE)][-1] == (
        "campaign_terminal"
    )


def test_a_context_that_does_not_record_writes_nothing_beside_the_outputs(
    env: ResolutionEnv, tmp_path: Path
) -> None:
    """The workflow doors: ``record=False`` — the step's ``_step.json`` is the
    receipt and the runner's stream is the stream — so the engine writes no
    ``protocol.json`` and no ``events.jsonl``, and still returns the signed
    steps."""
    out = tmp_path / "step"
    engine = _Stub()
    compiled = _compile(env)
    result = handoff(
        compiled, engine, RunContext(output_dir=out, env=env, record=False)
    )
    assert result.steps == steps_of(compiled, env).records
    assert not (out / RUN_RECORD_NAME).exists()
    assert not (out / EVENTS_FILE).exists()


def test_the_run_door_records_nothing_unless_asked(
    env: ResolutionEnv, tmp_path: Path
) -> None:
    """``run_protocol`` defaults to ``record=False``: the run writes its saved
    tables and nothing beside them — no ``protocol.json``, no
    ``events.jsonl`` — and still returns the signed steps. The receipt and the
    stream are opt-in (``record=True``; the CLI's ``--record``)."""
    out = tmp_path / "quiet"
    engine = _Stub()
    compiled = _compile(env)
    result = run_protocol(compiled, env, engine, out)
    ((_, run),) = engine.runs
    assert not run.record
    assert result.steps == steps_of(compiled, env).records
    assert not (out / RUN_RECORD_NAME).exists()
    assert not (out / EVENTS_FILE).exists()


def test_a_sink_without_record_is_refused(env: ResolutionEnv, tmp_path: Path) -> None:
    """A sink is handed the lines of ``events.jsonl``, which only a recording
    run writes; a sink with ``record=False`` would deliver nothing, so the
    door refuses the combination before any engine is entered."""
    engine = _Stub()
    with pytest.raises(ValueError, match="record=True"):
        run_protocol(_compile(env), env, engine, tmp_path / "out", sink=print)
    assert engine.runs == []
    assert not (tmp_path / "out").exists()


def test_a_bare_context_records_nothing(env: ResolutionEnv, tmp_path: Path) -> None:
    """``record`` defaults to ``False``: a caller driving ``execute`` directly
    — a test helper, a script — gets the signed steps and nothing beside its
    outputs; the ``run_protocol`` door asks for the receipt
    only when its caller does (``record=True``)."""
    run = RunContext(output_dir=tmp_path / "bare", env=env)
    assert not run.record
    result = _Stub().execute(_compile(env), run)
    assert result.steps and not (tmp_path / "bare").exists()


def test_a_document_the_data_rules_refuse_never_enters_the_engine(
    env: ResolutionEnv, tmp_path: Path
) -> None:
    """The handoff's data pass — here rule 4, a metric naming a column the
    base table lacks — refuses before ``execute`` is entered, with
    ``causalab validate``'s text byte for byte. The receipt and
    the stream are the engine's, written once it has enumerated and signed
    the steps, so a document refused before the engine is entered leaves
    **nothing** behind — no receipt, no stream, no output directory."""
    out = tmp_path / "refused"
    raw = base_doc()
    raw["method"]["save"][0]["aggregation"]["a"] = "no_such_column"
    compiled = compile_protocol(
        in_order(raw), env=env, base_dir=None, overrides=None, engine=None
    )
    engine = _Stub()
    with pytest.raises(ValidationError) as err:
        run_protocol(compiled, env, engine, out)
    assert err.value.rule == 4
    with pytest.raises(ValidationError) as same:
        pipeline.validate(compiled, engine, env=env, data=True)
    assert str(err.value) == str(same.value)
    assert engine.runs == []
    assert not out.exists()


def test_a_shard_reaches_the_engine_as_indices(
    env: ResolutionEnv, tmp_path: Path
) -> None:
    """``--points START:STOP`` is the run context's ``points``; the receipt
    lists exactly those points and the campaign digest is untouched."""
    out = tmp_path / "shard"
    engine = _Stub()
    compiled = _compile(env)
    n = len(steps_of(compiled, env).points)
    run_protocol(compiled, env, engine, out, points=f"0:{n}", record=True)
    ((seen, run),) = engine.runs
    assert dataclasses.replace(seen, positions=None) == compiled  # positions attached
    assert run.points == tuple(range(n))
    receipt = json.loads((out / RUN_RECORD_NAME).read_text())
    assert [p["index"] for p in receipt["points"]] == list(range(n))
    assert receipt["document_digest"] == compiled.campaign_digest
    (first,) = [r for r in read_events(out / EVENTS_FILE) if r["seq"] == 0]
    assert first["points"] == [0, n]
