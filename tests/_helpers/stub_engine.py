"""What a stub engine owes the doors since the sweep moved engine-side.

A door hands an engine the compiled document and a run context and reads
back [`steps`][causalab.protocol.engine.RunResult.steps] — the steps the engine
enumerated and signed — and, at the ``run_protocol`` door with
``record=True``, the receipt and the event stream the engine wrote before
its first forward (``RunContext.record``). A test's stand-in that "executes nothing" still owes
that half of the contract, or the runner's step record carries no digests and
the receipt is never on disk. These two helpers are the shared driver's
opening moves ([`causalab.neural.shared.execution.execute_request`][]), so
every stub spells them once.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from causalab.io.events import EventLog
from causalab.neural.shared.receipt import emit_run_events, run_events, write_run_record
from causalab.neural.shared.step_rules import check_steps
from causalab.neural.shared.sweep import enumerate_steps, sign_steps
from causalab.protocol.compiled import CompiledProtocol
from causalab.protocol.engine import Engine, RunContext, RunResult, StepRecord
from causalab.protocol.schema import parse_document

__all__ = ["stub_execute", "stub_record"]


def stub_record(
    engine: Engine, compiled: CompiledProtocol, run: RunContext
) -> tuple[tuple[StepRecord, ...], EventLog | None]:
    """Enumerate the steps ``run`` selects, hold them to the per-step
    checklist, sign them, and — when ``run.record`` — write the receipt and
    open the stream, exactly as the shared driver does before its first
    forward. Returns the signed steps and the open log (``None`` when
    nothing is recorded)."""
    expansion = enumerate_steps(compiled)
    indices = run.indices(len(expansion.points))
    check_steps(
        [parse_document(expansion.points[index].raw) for index in indices],
        run.env,
        coords=[expansion.points[index].coords for index in indices],
    )
    steps = tuple(
        step.record for step in sign_steps(expansion, run.env, indices=indices)
    )
    if not run.record:
        return steps, None
    write_run_record(compiled, run, engine, steps)
    return steps, run_events(compiled, run, steps)


def stub_execute(
    engine: Engine,
    compiled: CompiledProtocol,
    run: RunContext,
    *,
    files: Mapping[str, Path] | None = None,
    summaries: tuple[Mapping[str, Any], ...] = (),
) -> RunResult:
    """The result of an engine that executed nothing: the steps it would have
    run, signed, the ``files`` it claims to have written, and — when asked —
    the receipt and a finished stream."""
    steps, log = stub_record(engine, compiled, run)
    result = RunResult(files=dict(files or {}), summaries=summaries, steps=steps)
    if log is not None:
        emit_run_events(log, result)
    return result
