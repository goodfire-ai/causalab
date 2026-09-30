"""Write run receipts and events when RunContext.record is enabled.

Both files are opt-in: ``run_protocol(..., record=True)`` or the CLI's
``--record``. Without them a run writes its saved tables only, and the
``fires`` counts stay in the returned result's summaries.

Before planning or any forward, the driver writes ``protocol.json`` with
the canonical document, signed steps, provenance, and execution fields.
After the points run it adds what they observed: the ``fires`` counts, the
measured bounds, the ragged geometry and the ``models`` list with each
model's resolved commit (``causalab.protocol.receipt``).
It opens ``events.jsonl`` with ``phase_started``. Completed runs append
progress and metric events followed by commit, completion, and terminal
events. An exception can leave the stream without a terminal event.

Workflow execution uses its runner's event stream and each step's
``_step.json``. Receipt schema helpers remain in ``protocol.receipt``.
This module is torch-free at import.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

from causalab.io.events import EVENTS_FILE, EventLog
from causalab.protocol.receipt import RUN_RECORD_NAME, execution_record

if TYPE_CHECKING:
    from causalab.protocol.compiled import CompiledProtocol
    from causalab.protocol.engine import RunContext, RunResult, StepRecord

__all__ = ["emit_run_events", "run_events", "write_run_record"]


def write_run_record(
    compiled: CompiledProtocol,
    run: RunContext,
    engine: Any,
    steps: Sequence[StepRecord],
) -> Path:
    """``<run.output_dir>/protocol.json`` — the receipt of what ran.

    The saved tables say what the numbers are; this says what produced them:
    the canonical document (every default materialized, dtype and quantization
    included), its digest, the per-point provenance digests of the ``steps``
    the engine signed for this run (the points ``run`` selects, in run
    order), and — under ``execution`` — the batch geometry the chosen
    ``engine`` runs under ([`execution_record`][],
    resolved against the run's own ``execution`` block). It is what someone
    reproducing the run reads first, and it is written **before** execution
    so a crashed run still says what it was, which is why the geometry is
    taken from the engine's bound rather than from anything the run observed.
    The ``execution.parallel`` block names the launcher the run's publisher
    carries (``solo`` in one process).
    """
    record: dict[str, Any] = {
        "document_digest": compiled.digests.document,
        "canonical": compiled.canonical,
        "execution": execution_record(engine, run, launcher=run.publisher.launcher),
        "points": [
            {
                "index": step.index,
                "digest": step.digest,
                "coords": dict(step.coords),
            }
            for step in steps
        ],
    }
    run.output_dir.mkdir(parents=True, exist_ok=True)
    target = run.output_dir / RUN_RECORD_NAME
    target.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return target


def run_events(
    compiled: CompiledProtocol, run: RunContext, steps: Sequence[StepRecord]
) -> EventLog:
    """Open the document run's event stream, ``<run.output_dir>/events.jsonl``
    ([`causalab.io.events`][]; workflow spec §4.3), and append
    ``phase_started``. Every line carries the run's identity — the campaign
    digest and the ``[start, stop)`` shard it covers — and ``run.sink`` is
    the adapter handed each line after its local write; its failure becomes a
    ``warning`` line and changes nothing else. Opened over an existing stream
    it continues it. Returns the [`EventLog`][]."""
    selected = tuple(step.index for step in steps)
    # the identity's shard is ``[start, stop)``: the document run's ``points``
    # is ``parse_points``' contiguous, non-empty range (or every point), so
    # its ends are the bounds; a door opening a stream for an arbitrary
    # index tuple must record the indices instead
    assert selected and selected == tuple(range(selected[0], selected[-1] + 1)), (
        "run_events: the document run's shard is a contiguous range"
    )
    log = EventLog(
        run.output_dir / EVENTS_FILE,
        identity={
            "document_digest": compiled.digests.document,
            "points": [selected[0], selected[-1] + 1],
        },
        sink=run.sink,
    )
    log.emit("phase_started", {"phase": "execute", "n_points": len(selected)})
    return log


def emit_run_events(log: EventLog, result: RunResult) -> None:
    """The lines a finished execution appends, from what the engine returned
    (§4.3). The engine runs the campaign whole, so per-point ``progress`` is
    reported when it has: one line per executed step in run order, each
    naming its ``point_digest`` (``result.steps``), then a ``metric`` line
    per summarized metric — the same ``_summary_stat`` value ``explain``
    prints, never a fact the run receipt does not carry — and
    ``result_committed`` naming the files written."""
    total = len(result.steps)
    for index, step in enumerate(result.steps):
        log.emit(
            "progress",
            {
                "point_digest": step.digest,
                "index": index,
                "completed": index + 1,
                "total": total,
            },
        )
    for summary in result.summaries:
        for name, value in sorted(summary.get("metrics", {}).items()):
            log.emit(
                "metric",
                {"point_digest": summary.get("point"), "name": name, "value": value},
            )
    log.emit("result_committed", {"files": sorted(result.files)})
    log.emit("phase_completed", {"phase": "execute", "forwards": result.forwards})
    log.emit("campaign_terminal", {"outcome": "completed"})
