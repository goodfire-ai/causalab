"""Derive workflow statuses from a run's event stream.

Completed phases and failure events determine each attempted step's status.
The schedule propagates skipped and blocked states to dependent steps. The
manifest writer compares this result with the runner's state before publishing."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from causalab.workflow.manifest import classify_unreached

__all__ = ["derive_statuses"]

#: What ``phase_completed`` may say about a workflow step (§4.3's table): the
#: three words for a step that reached its end this run — published, reused,
#: or skipped by a decision (§2.8; a skip is a terminal word, never
#: ``failed``). Anything else on the stream is refused rather than copied into
#: the manifest.
_TERMINAL_WORDS = frozenset({"completed", "reused", "skipped"})


def derive_statuses(
    records: Iterable[Mapping[str, Any]],
    *,
    order: Sequence[str],
    dependencies: Mapping[str, tuple[str, ...]],
    selective: frozenset[str] = frozenset(),
) -> dict[str, str]:
    """``{step: status}`` for every step of ``order``, from one run's lines.
    ``selective`` names the joins declaring ``require: selected`` (spec §2.9),
    which a skipped child does not skip — handed through to
    [`classify_unreached`][].

    ``records`` are the lines *this run* appended (``read_events`` sliced from
    [`causalab.io.events.EventLog.opened_at`][]); a ``--resume`` run's own
    lines say ``reused``, and the first run's ``completed`` lines before them
    are not its history. The last terminal line per step wins, so a stream
    that carries more than one run still derives the latest word.

    Pure: reads nothing but ``records``. Refuses (``ValueError``) a line whose
    ``payload`` is not an object, a terminal line (``phase_completed``, or a
    ``warning`` with ``reason: attempt_failed``) that names a step outside
    ``order``, or a ``phase_completed`` whose ``status`` is not one of the two
    terminal words — a stream this function cannot read is not one the
    manifest may be derived from.
    """
    order = tuple(order)
    known = set(order)
    reached: dict[str, dict[str, Any]] = {}
    for record in records:
        payload = record.get("payload") or {}
        if not isinstance(payload, Mapping):
            raise ValueError(
                f"events.jsonl seq {record.get('seq')!r}: payload is "
                f"{type(payload).__name__}, not an object"
            )
        # only a step's two terminal lines are read for status;
        # `phase_started`, `progress`, `metric` and any other `warning` say
        # nothing here
        event = record.get("event")
        publish = event == "phase_completed"
        failed = event == "warning" and payload.get("reason") == "attempt_failed"
        if not (publish or failed):
            continue
        step = payload.get("step")
        if step not in known:
            raise ValueError(
                f"events.jsonl seq {record.get('seq')!r}: "
                f"{'phase_completed' if publish else 'attempt_failed'} names "
                f"step {step!r}, not one of {list(order)}"
            )
        if failed:
            reached[step] = {"status": "failed"}
            continue
        status = payload.get("status")
        if status not in _TERMINAL_WORDS:
            raise ValueError(
                f"events.jsonl seq {record.get('seq')!r}: phase_completed for "
                f"{step!r} says {status!r}, not one of {sorted(_TERMINAL_WORDS)}"
            )
        reached[step] = {"status": status}
    # `classify_unreached` skips every step already in `reached`, so the two
    # maps are disjoint and one merge is the whole schedule
    entries = {
        **reached,
        **classify_unreached(order, dependencies, reached, selective=selective),
    }
    return {name: entries[name]["status"] for name in order}
