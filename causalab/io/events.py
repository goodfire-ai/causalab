"""Read and append the local ``events.jsonl`` stream.

Each event has a monotonically increasing sequence number and one of the
kinds in ``EVENTS``. Execution records its starting offset so status derivation
reads that run's events. The workflow manifest derives its status values from
this stream.

Event delivery through a sink is separate from scientific execution. Delivery
failures produce warning events. Experiment identity and reuse checks use the
recorded documents and artifacts."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

__all__ = [
    "EVENTS",
    "EVENTS_FILE",
    "SCHEMA_VERSION",
    "EventLog",
    "EventSink",
    "read_events",
    "terminal",
]

#: The line format's version. A reader that meets another number refuses the
#: line rather than guessing at its shape.
SCHEMA_VERSION = 1

#: The closed vocabulary of ``event``, in the order the event table of the
#: workflow spec (``docs/workflow_protocol.md`` §4.3) lists it.
#: ``campaign_terminal`` keeps its name, with a campaign read as one document
#: run. A run that did not finish leaves a stream without one — that absence
#: is what "did not finish" reads as.
EVENTS: tuple[str, ...] = (
    "phase_started",
    "progress",
    "metric",
    "warning",
    "result_committed",
    "phase_completed",
    "campaign_terminal",
)

#: The stream's filename — beside ``protocol.json`` for a protocol run and
#: beside ``workflow.json`` for a workflow run. A workflow step name matches
#: ``[A-Za-z0-9_-]+`` (workflow spec §5 rule 3), so the dot keeps it from ever
#: colliding with a step directory.
EVENTS_FILE = "events.jsonl"

#: The adapter seam: called with each line's record after the local write.
#: Whatever it raises is caught and becomes a ``warning`` line.
EventSink = Callable[[Mapping[str, Any]], None]

#: The keys every line has, in the order they are written. Identity keys sit
#: between ``ts`` and ``payload`` and may not shadow any of these.
_FIXED_KEYS = ("schema_version", "event", "seq", "ts", "payload")


def _now() -> str:
    """UTC, ISO 8601, microseconds — one stamp per line, never compared."""
    return datetime.now(timezone.utc).isoformat()


def _parse_line(line: str, path: Path, number: int) -> dict[str, Any]:
    """One line of the stream as a record, or a ``ValueError`` naming the
    line — a torn tail or a foreign line is reported, never skipped."""
    try:
        record = json.loads(line)
    except json.JSONDecodeError as err:
        raise ValueError(
            f"{path}:{number} is not a complete JSON line ({err.msg}); the "
            "stream is cut mid-line"
        ) from None
    if not isinstance(record, dict):
        raise ValueError(f"{path}:{number} is not a JSON object")
    if record.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"{path}:{number} has schema_version {record.get('schema_version')!r}; "
            f"this reader understands {SCHEMA_VERSION}"
        )
    if record.get("event") not in EVENTS:
        raise ValueError(
            f"{path}:{number} names event {record.get('event')!r}, not one of "
            f"{list(EVENTS)}"
        )
    return record


def read_events(path: Path) -> list[dict[str, Any]]:
    """Every line of the stream at ``path``, in order; ``[]`` if there is no
    file. A line that does not parse, carries another schema version, names
    an event outside [`EVENTS`][] or breaks the ``seq`` sequence (``0, 1,
    2, …`` — a deleted or duplicated line) raises ``ValueError`` naming the
    line."""
    path = Path(path)
    if not path.is_file():
        return []
    return list(_iter_events(path))


def _iter_events(path: Path) -> Iterator[dict[str, Any]]:
    # `seq` starts at 0 and each line follows the last by exactly one: a
    # deleted or duplicated line is a stream that was rewritten, and reads
    # as such rather than as a valid shorter (or longer) one
    prev = -1
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.endswith("\n"):
                raise ValueError(
                    f"{path}:{number} has no line terminator; the stream is cut "
                    "mid-line"
                )
            record = _parse_line(line, path, number)
            seq = record.get("seq")
            if seq != prev + 1:
                raise ValueError(
                    f"{path}:{number} has seq {seq!r}, expected {prev + 1}; the "
                    "stream is append-only and gap-free"
                )
            prev = seq
            yield record


def terminal(path: Path) -> bool:
    """Whether the stream at ``path`` ends in ``campaign_terminal`` — the one
    way a reader tells a finished run from one that stopped. False means the
    run did not reach its end state: it crashed before its record was written
    or it was interrupted (``KeyboardInterrupt``, ``SystemExit`` — a workflow
    run still writes its manifest then, and appends no terminal line). True
    with ``outcome: failed`` means a step failed and the runner still
    finished: the manifest was written. No file, or an empty one, is not
    finished."""
    records = read_events(path)
    return bool(records) and records[-1]["event"] == "campaign_terminal"


class EventLog:
    """One run's stream: an append-only writer over ``path``.

    ``identity`` is the run's — ``{"document_digest", "points"}`` for a
    protocol run, ``{}`` for a workflow run, whose identities are per step —
    and is copied into every line. ``sink`` is the optional adapter ([`EventSink`][]).

    Opening a log over an existing stream **continues** it: ``seq`` picks up
    after the last line, so a ``--resume`` run appends to the first run's
    stream rather than starting a second. A stream whose tail is torn, or
    whose ``seq`` has a gap, refuses to open (the reader's ``ValueError``) —
    appending to a half-written or rewritten stream would bury the damage.
    (A well-formed foreign line with no ``seq`` cannot come from this writer;
    the reader refuses it as out of sequence rather than inventing a number.)

    [`opened_at`][] is the ``seq`` this opening's first line gets, so
    ``[r for r in read_events(path) if r["seq"] >= log.opened_at]`` is exactly
    what this opening wrote — the lines one run appended, which is what the
    manifest's statuses are derived from (`causalab.workflow.derived`).
    """

    def __init__(
        self,
        path: Path,
        *,
        identity: Mapping[str, Any],
        sink: EventSink | None = None,
    ) -> None:
        shadowed = sorted(set(identity) & set(_FIXED_KEYS))
        if shadowed:
            raise ValueError(f"identity keys shadow the line's own: {shadowed}")
        self.path = Path(path)
        self.identity: dict[str, Any] = dict(identity)
        self.sink = sink
        existing = read_events(self.path)
        self._seq = existing[-1]["seq"] if existing else -1
        #: the ``seq`` of the first line this opening writes
        self.opened_at: int = self._seq + 1

    def emit(
        self, event: str, payload: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        """Append one line and return its record — **the line as written**.

        The line is on disk — flushed — before the sink sees it. What the sink
        receives, and what is returned, is ``json.loads`` of the written line:
        a payload value the writer had to stringify (a ``pathlib.Path``, say)
        reaches the sink as the ``str`` the file holds, so a remote adapter
        re-serializes byte-identically. The sink and the caller see the same
        object. A sink that raises produces one further ``warning`` line,
        written locally and not handed to the sink; the exception goes no
        further.
        """
        if event not in EVENTS:
            raise ValueError(
                f"unknown event {event!r}; the vocabulary is {list(EVENTS)}"
            )
        record = self._write(event, payload)
        if self.sink is not None:
            try:
                self.sink(record)
            except Exception as exc:  # the seam's contract: nothing propagates
                self._write(
                    "warning",
                    {
                        "reason": "sink_failed",
                        "event": event,
                        "seq": record["seq"],
                        "error": repr(exc),
                    },
                )
        return record

    def _write(self, event: str, payload: Mapping[str, Any] | None) -> dict[str, Any]:
        self._seq += 1
        record: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "event": event,
            "seq": self._seq,
            "ts": _now(),
            **self.identity,
            "payload": dict(payload) if payload is not None else {},
        }
        line = json.dumps(record, sort_keys=False, default=str)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
        # exactly what the file holds — not the pre-`default=str` dict
        return json.loads(line)
