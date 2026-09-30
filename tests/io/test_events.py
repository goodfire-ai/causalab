"""The local append-only event stream (workflow spec §4.3; `causalab/io/events.py`).

Four things are pinned here, none needing a model. **The stream**: `seq` is strictly
increasing and gap-free, a second `EventLog` over the same file continues the
sequence rather than restarting it, `terminal()` reads the absence of
`campaign_terminal` as "did not finish", and a stream cut mid-line — or one
with a middle line deleted or duplicated, a `seq` gap — is *reported*, never
skipped — a torn tail or a rewritten stream that read as a clean one would be
a lie about where the run stopped. **The census**: the seven names in
`EVENTS` equal the spec's §4.3 table, in order; the mutation (an eighth name
with no row) is shown to go red rather than asserted in prose. **Fail-closed
with its twin**: an unknown event is refused naming the vocabulary,
and every one of the seven logs. **The sink seam**: a sink that raises
produces one `warning` line per delivery and nothing propagates — and the
mutation is run in-test: with the swallow monkeypatched away the exception
*does* propagate, so the test proves the swallow is what bites.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from causalab.io import events as ev
from causalab.io.events import (
    EVENTS,
    EVENTS_FILE,
    SCHEMA_VERSION,
    EventLog,
    read_events,
    terminal,
)

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "docs" / "workflow_protocol.md"

IDENTITY = {"document_digest": "f" * 64, "points": [0, 4]}


def _log(tmp_path: Path, **kwargs: Any) -> EventLog:
    return EventLog(
        tmp_path / EVENTS_FILE, identity=IDENTITY, sink=kwargs.pop("sink", None)
    )


# --------------------------------------------------------------------------- #
# the line
# --------------------------------------------------------------------------- #


def test_a_line_carries_the_schema_the_identity_and_the_payload(tmp_path: Path) -> None:
    log = _log(tmp_path)
    record = log.emit("phase_started", {"step": "fit", "type": "script"})
    (line,) = (tmp_path / EVENTS_FILE).read_text().splitlines()
    on_disk = json.loads(line)
    assert on_disk == record
    assert list(on_disk) == [
        "schema_version",
        "event",
        "seq",
        "ts",
        "document_digest",
        "points",
        "payload",
    ]
    assert on_disk["schema_version"] == SCHEMA_VERSION == 1
    assert on_disk["seq"] == 0
    assert on_disk["document_digest"] == IDENTITY["document_digest"]
    assert on_disk["points"] == IDENTITY["points"]
    assert on_disk["payload"] == {"step": "fit", "type": "script"}
    assert re.match(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}\+00:00$", on_disk["ts"]
    )


def test_an_identity_key_may_not_shadow_the_lines_own(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="shadow"):
        EventLog(tmp_path / EVENTS_FILE, identity={"seq": 1})


# --------------------------------------------------------------------------- #
# seq, append-only, terminal, torn tail
# --------------------------------------------------------------------------- #


def test_seq_is_gap_free_over_many_emits(tmp_path: Path) -> None:
    log = _log(tmp_path)
    for i in range(50):
        log.emit(EVENTS[i % len(EVENTS)], {"i": i})
    seqs = [r["seq"] for r in read_events(tmp_path / EVENTS_FILE)]
    assert seqs == list(range(50))


def test_a_second_opening_continues_the_sequence(tmp_path: Path) -> None:
    """Append-only across openings: the `--resume` run's lines follow the
    first run's, in one file, with one sequence."""
    first = _log(tmp_path)
    first.emit("phase_started", {"step": "a"})
    first.emit("phase_completed", {"step": "a"})
    before = (tmp_path / EVENTS_FILE).read_bytes()

    second = _log(tmp_path)
    second.emit("phase_started", {"step": "a"})
    after = (tmp_path / EVENTS_FILE).read_bytes()

    assert after.startswith(before), "an earlier line was rewritten"
    assert [r["seq"] for r in read_events(tmp_path / EVENTS_FILE)] == [0, 1, 2]


def test_terminal_is_the_last_line_being_campaign_terminal(tmp_path: Path) -> None:
    path = tmp_path / EVENTS_FILE
    assert terminal(path) is False, "no file is not finished"
    log = _log(tmp_path)
    log.emit("phase_started", {"step": "a"})
    assert terminal(path) is False
    log.emit("campaign_terminal", {"outcome": "completed"})
    assert terminal(path) is True
    # a line after the terminal one un-finishes the stream: last line rules
    log.emit("warning", {"reason": "late"})
    assert terminal(path) is False


def test_a_stream_cut_mid_line_is_reported_not_swallowed(tmp_path: Path) -> None:
    path = tmp_path / EVENTS_FILE
    log = _log(tmp_path)
    log.emit("phase_started", {"step": "a"})
    log.emit("campaign_terminal", {"outcome": "completed"})
    whole = path.read_bytes()
    path.write_bytes(whole[:-7])  # the run died mid-write of its last line

    with pytest.raises(ValueError, match="cut mid-line"):
        read_events(path)
    with pytest.raises(ValueError, match="cut mid-line"):
        terminal(path)
    # a writer does not bury the tear under a new line either
    with pytest.raises(ValueError, match="cut mid-line"):
        _log(tmp_path)

    # the *complete* prefix is still a valid stream
    path.write_bytes(whole.rsplit(b"\n", 2)[0] + b"\n")
    assert [r["event"] for r in read_events(path)] == ["phase_started"]
    assert terminal(path) is False


def test_a_deleted_or_duplicated_middle_line_is_refused_as_a_seq_gap(
    tmp_path: Path,
) -> None:
    """The stream is authoritative for status (§4.3), so a stream with a line
    removed — or one repeated — may not read as a valid one: `seq` starts at
    0 and each line follows the last by exactly one, and the reader,
    `terminal()` and a writer opening over it all refuse. The intact stream
    still reads."""
    path = tmp_path / EVENTS_FILE
    log = _log(tmp_path)
    for i in range(4):
        log.emit(EVENTS[i], {"i": i})
    log.emit("campaign_terminal", {"outcome": "completed"})
    lines = path.read_text().splitlines(keepends=True)
    intact = read_events(path)
    assert [r["seq"] for r in intact] == [0, 1, 2, 3, 4]
    assert terminal(path) is True

    # a deleted middle line: seq 3 follows seq 1
    path.write_text("".join(lines[:2] + lines[3:]))
    with pytest.raises(ValueError, match=r":3 has seq 3, expected 2"):
        read_events(path)
    with pytest.raises(ValueError, match=r"has seq 3, expected 2"):
        terminal(path)
    # a writer does not continue a stream with a hole in it
    with pytest.raises(ValueError, match=r"has seq 3, expected 2"):
        _log(tmp_path)

    # a duplicated line: seq 2 twice
    path.write_text("".join(lines[:3] + [lines[2]] + lines[3:]))
    with pytest.raises(ValueError, match=r":4 has seq 2, expected 3"):
        read_events(path)

    # the first line must be seq 0: a stream with its head cut off is not one
    path.write_text("".join(lines[1:]))
    with pytest.raises(ValueError, match=r":1 has seq 1, expected 0"):
        read_events(path)

    # the intact stream still reads, and a writer continues it
    path.write_text("".join(lines))
    assert read_events(path) == intact
    assert _log(tmp_path).opened_at == 5


def test_a_foreign_line_is_refused_by_the_reader(tmp_path: Path) -> None:
    path = tmp_path / EVENTS_FILE
    path.write_text(
        json.dumps({"schema_version": 2, "event": "progress", "seq": 0}) + "\n"
    )
    with pytest.raises(ValueError, match="schema_version 2"):
        read_events(path)
    path.write_text(
        json.dumps({"schema_version": 1, "event": "heartbeat", "seq": 0}) + "\n"
    )
    with pytest.raises(ValueError, match="heartbeat"):
        read_events(path)


# --------------------------------------------------------------------------- #
# fail-closed, with its twin
# --------------------------------------------------------------------------- #


def test_an_unknown_event_is_refused_naming_the_vocabulary(tmp_path: Path) -> None:
    log = _log(tmp_path)
    with pytest.raises(ValueError, match="phase_started.*campaign_terminal"):
        log.emit("heartbeat")
    assert not (tmp_path / EVENTS_FILE).exists(), "a refused event left a line"


def test_every_event_in_the_vocabulary_logs(tmp_path: Path) -> None:
    """The refusal's twin: the seven names are each accepted and written."""
    log = _log(tmp_path)
    for name in EVENTS:
        log.emit(name)
    assert [r["event"] for r in read_events(tmp_path / EVENTS_FILE)] == list(EVENTS)


# --------------------------------------------------------------------------- #
# the sink seam
# --------------------------------------------------------------------------- #


class Delivered(Exception):
    """What the raising sink raises."""


def _raising_sink(record: Any) -> None:
    raise Delivered(record["event"])


def test_a_sink_failure_becomes_one_warning_and_nothing_propagates(
    tmp_path: Path,
) -> None:
    seen: list[str] = []

    def sink(record: Any) -> None:
        seen.append(record["event"])
        raise Delivered(record["event"])

    log = _log(tmp_path, sink=sink)
    log.emit("phase_started", {"step": "a"})
    log.emit("result_committed", {"step": "a", "files": []})
    log.emit("campaign_terminal", {"outcome": "completed"})

    records = read_events(tmp_path / EVENTS_FILE)
    assert [r["event"] for r in records] == [
        "phase_started",
        "warning",
        "result_committed",
        "warning",
        "campaign_terminal",
        "warning",
    ]
    assert [r["seq"] for r in records] == list(range(6))
    warnings_ = [r for r in records if r["event"] == "warning"]
    assert [w["payload"]["event"] for w in warnings_] == [
        "phase_started",
        "result_committed",
        "campaign_terminal",
    ]
    assert [w["payload"]["seq"] for w in warnings_] == [0, 2, 4]
    assert all(w["payload"]["reason"] == "sink_failed" for w in warnings_)
    assert all(w["payload"]["error"].startswith("Delivered(") for w in warnings_)
    # the warning about a failed delivery is not itself delivered
    assert seen == ["phase_started", "result_committed", "campaign_terminal"]


def test_a_sink_that_works_sees_every_line_after_it_is_on_disk(tmp_path: Path) -> None:
    path = tmp_path / EVENTS_FILE
    seen: list[tuple[int, int]] = []

    def sink(record: Any) -> None:
        # by the time the sink runs, the line is on disk
        seen.append((record["seq"], len(read_events(path))))

    log = _log(tmp_path, sink=sink)
    log.emit("phase_started")
    log.emit("phase_completed")
    assert seen == [(0, 1), (1, 2)]
    assert [r["event"] for r in read_events(path)] == [
        "phase_started",
        "phase_completed",
    ]


def test_the_sink_and_the_caller_receive_the_line_as_written(tmp_path: Path) -> None:
    """A payload value the writer had to stringify — a `pathlib.Path` — reaches
    the sink as the `str` the file holds, and `emit` returns the same object
    the sink saw: a remote adapter re-serializes byte-identically."""
    seen: list[Any] = []
    log = _log(tmp_path, sink=seen.append)
    returned = log.emit("result_committed", {"step": "a", "where": tmp_path / "a"})
    (line,) = (tmp_path / EVENTS_FILE).read_text().splitlines()
    on_disk = json.loads(line)
    assert seen == [on_disk]
    assert returned is seen[0]
    assert seen[0]["payload"]["where"] == str(tmp_path / "a")
    assert isinstance(seen[0]["payload"]["where"], str)
    assert json.dumps(seen[0], sort_keys=False) == line


def test_the_mutation_without_the_swallow_the_sink_exception_propagates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The sink seam's mutation, run rather than described: replace `emit` with one that
    calls the sink and does not catch, and the failure escapes — so the
    swallow in `EventLog.emit` is the thing the previous test is proving."""

    def unswallowed(self: EventLog, event: str, payload: Any = None) -> dict[str, Any]:
        record = self._write(event, payload)  # pyright: ignore[reportPrivateUsage]
        assert self.sink is not None
        self.sink(record)
        return record

    monkeypatch.setattr(EventLog, "emit", unswallowed)
    log = _log(tmp_path, sink=_raising_sink)
    with pytest.raises(Delivered):
        log.emit("phase_started")


# --------------------------------------------------------------------------- #
# the census against the spec table
# --------------------------------------------------------------------------- #


def _spec_event_table() -> list[str]:
    """The §4.3 table's left column — the rows under the header `event`, up to
    the first row that is not a backticked member (the protocol census's
    parser, copied)."""
    text = SPEC.read_text()
    # anchored on the title, not the number: a renumbered §4.3 must fail by
    # name here, not by `IndexError`
    heading = re.search(r"^###? [\d.]* ?Event stream\s*$", text, flags=re.M)
    assert heading is not None, "no §4.3 Event stream section in the workflow spec"
    section = re.split(r"^##+ ", text[heading.end() :], maxsplit=1, flags=re.M)[0]
    rows = [
        [c.strip() for c in line.strip().strip("|").split("|")]
        for line in section.splitlines()
        if line.strip().startswith("|")
    ]
    headers = [i for i, row in enumerate(rows) if row[0] == "event"]
    assert headers, "no §4.3 Event stream section / no header row `event`"
    header = headers[0]
    names: list[str] = []
    for row in rows[header + 2 :]:
        if not row[0].startswith("`"):
            break
        names.append(row[0].strip("`"))
    return names


def test_the_spec_table_lists_exactly_the_seven_events() -> None:
    documented = _spec_event_table()
    assert documented, "no event table under §4.3"
    assert documented == list(EVENTS)
    assert len(EVENTS) == 7
    assert len(set(EVENTS)) == 7
    # `campaign_terminal` keeps its name: campaign = document
    assert EVENTS[-1] == "campaign_terminal"


def test_the_mutation_an_eighth_event_without_a_row_is_red(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ev, "EVENTS", (*EVENTS, "heartbeat"))
    assert _spec_event_table() != list(ev.EVENTS)


def test_every_logged_event_is_in_the_vocabulary(tmp_path: Path) -> None:
    """The census's second half over a real stream, sink failures included."""
    log = _log(tmp_path, sink=_raising_sink)
    for name in EVENTS:
        log.emit(name)
    assert {r["event"] for r in read_events(tmp_path / EVENTS_FILE)} <= set(EVENTS)


def test_the_module_is_stdlib_only() -> None:
    """`protocol/run.py` reaches this module from a torch-free verb, and no
    hashed script's closure may grow by it: nothing outside the stdlib."""
    import ast

    source = (REPO / "causalab" / "io" / "events.py").read_text()
    imported = {
        (
            node.names[0].name if isinstance(node, ast.Import) else node.module or ""
        ).split(".")[0]
        for node in ast.walk(ast.parse(source))
        if isinstance(node, (ast.Import, ast.ImportFrom))
    }
    assert imported <= {"json", "datetime", "pathlib", "typing", "__future__"}, imported
