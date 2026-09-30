"""The workflow runner's event stream (workflow spec §4.3, §8).

A three-step script chain — the same shape `test_attempt_publish.py` uses —
run on CPU with no engine. What is pinned: the per-step order `phase_started`
→ `result_committed` → `phase_completed`, with `result_committed` at the
publish moment and `campaign_terminal` last; the stream beside `workflow.json`
and in **no** step directory (a mutable sidecar beside immutable outputs);
**T9** — `workflow.json` and every published file byte-identical with and
without a sink that raises on every line, the stream holding one `warning`
per failed delivery; `--resume` reusing every step with the stream present
and appending to it (the stream is not an input to reuse); a failed run's
stream ending `warning` → `campaign_terminal outcome=failed`, because the
manifest was written (**T10**; a run that dies before its `finally` leaves no
terminal line, and an *interrupted* run — `KeyboardInterrupt`, `SystemExit` —
writes its manifest and no terminal line either: interrupted is not failed);
a pre-existing stream the runner cannot read refuses the run before it starts,
as a `ProtocolError` with nothing written; a manifest the `finally` withheld
on a clean run is refused at the return, not asserted; and
**T12** — both pinned workflow digests unmoved by the two modules that are in
no script's import closure.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from causalab.io.events import EVENTS, EVENTS_FILE, read_events, terminal
from causalab.protocol.rules.errors import ProtocolError
from causalab.workflow import manifest as mf
from causalab.workflow import runner
from causalab.workflow.document import load_workflow
from causalab.workflow.runner import run_workflow

from tests.workflow.test_attempt_publish import (
    DECLARED,
    FIRST,
    RAISING,
    SECOND,
    THIRD,
    _document,  # pyright: ignore[reportPrivateUsage]
)
from tests._helpers.paths import WORKFLOWS_DIR

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[2]
WORKFLOWS = WORKFLOWS_DIR


class Delivered(Exception):
    pass


def _raising_sink(record: Any) -> None:
    raise Delivered(record["event"])


@pytest.fixture()
def chain_dir(tmp_path: Path) -> Path:
    """The three-step chain's scripts, as `test_attempt_publish.py` lays them out."""
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "first.py").write_text(FIRST)
    (scripts / "second.py").write_text(SECOND)
    (scripts / "third.py").write_text(THIRD)
    (scripts / "raising.py").write_text(RAISING)
    return tmp_path


def _load(chain_dir: Path, env: Any, **kwargs: Any) -> Any:
    return load_workflow(_document(**kwargs), env, workflow_dir=chain_dir)


def _files(root: Path) -> dict[str, bytes]:
    """Every file under ``root`` except the stream itself, which carries
    timestamps and is the one file a run is allowed to differ in."""
    return {
        str(p.relative_to(root)): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.name != EVENTS_FILE
    }


def test_the_stream_narrates_each_step_in_order_and_ends_terminal(
    chain_dir: Path, env: Any, tmp_path: Path
) -> None:
    loaded = _load(chain_dir, env)
    result = run_workflow(loaded, env, tmp_path / "runs", None)
    run_root = result.run_root
    records = read_events(run_root / EVENTS_FILE)

    assert [r["seq"] for r in records] == list(range(len(records)))
    # a workflow run has no run-level identity (§4.3, §7): its steps carry theirs
    assert all("workflow_digest" not in r for r in records)
    assert {r["event"] for r in records} <= set(EVENTS)

    expected: list[tuple[str, str | None]] = []
    for name in loaded.order:
        expected += [
            ("phase_started", name),
            ("result_committed", name),
            ("phase_completed", name),
        ]
    expected.append(("campaign_terminal", None))
    assert [(r["event"], r["payload"].get("step")) for r in records] == expected

    committed = {
        r["payload"]["step"]: r["payload"]["files"]
        for r in records
        if r["event"] == "result_committed"
    }
    assert {step: set(files) for step, files in committed.items()} == DECLARED
    assert records[-1]["payload"] == {
        "outcome": "completed",
        "steps": {name: "completed" for name in loaded.order},
    }
    assert terminal(run_root / EVENTS_FILE)


def test_the_stream_sits_beside_the_manifest_and_in_no_step_directory(
    chain_dir: Path, env: Any, tmp_path: Path
) -> None:
    loaded = _load(chain_dir, env)
    run_root = run_workflow(loaded, env, tmp_path / "runs", None).run_root
    assert (run_root / EVENTS_FILE).is_file()
    assert (run_root / mf.MANIFEST).is_file()
    for name in loaded.order:
        assert not (run_root / name / EVENTS_FILE).exists()
        assert set(p.name for p in (run_root / name).iterdir()) == DECLARED[name] | {
            "_step.json"
        }
    assert not (run_root / mf.ATTEMPTS_DIR).exists()


def test_t9_a_raising_sink_changes_no_output_and_leaves_one_warning_per_line(
    chain_dir: Path, env: Any, tmp_path: Path
) -> None:
    loaded = _load(chain_dir, env)
    plain = run_workflow(loaded, env, tmp_path / "plain", None).run_root
    sunk = run_workflow(
        loaded, env, tmp_path / "sunk", None, sink=_raising_sink
    ).run_root

    assert (plain / mf.MANIFEST).read_bytes() == (sunk / mf.MANIFEST).read_bytes()
    assert _files(plain) == _files(sunk)

    records = read_events(sunk / EVENTS_FILE)
    delivered = [r for r in records if r["event"] != "warning"]
    warnings_ = [r for r in records if r["event"] == "warning"]
    assert len(warnings_) == len(delivered) == 3 * len(loaded.order) + 1
    # each warning follows the line it is about, and names it
    for delivered_line, warning in zip(delivered, warnings_):
        assert warning["seq"] == delivered_line["seq"] + 1
        assert warning["payload"] == {
            "reason": "sink_failed",
            "event": delivered_line["event"],
            "seq": delivered_line["seq"],
            "error": f"Delivered('{delivered_line['event']}')",
        }
    # the no-sink stream is the delivered half, event for event
    assert [r["event"] for r in read_events(plain / EVENTS_FILE)] == [
        r["event"] for r in delivered
    ]
    # ...and the sink's failure is the last thing on the stream, so the
    # terminal question is answered by the delivered line before it
    assert records[-2]["event"] == "campaign_terminal"


def test_resume_ignores_the_stream_and_appends_to_it(
    chain_dir: Path, env: Any, tmp_path: Path
) -> None:
    loaded = _load(chain_dir, env)
    out = tmp_path / "runs"
    first = run_workflow(loaded, env, out, None)
    first_stream = (first.run_root / EVENTS_FILE).read_bytes()
    first_files = _files(first.run_root)

    second = run_workflow(loaded, env, out, None, resume=True)
    assert {n: e["status"] for n, e in second.manifest["steps"].items()} == {
        name: "reused" for name in loaded.order
    }
    # every published unit byte-identical; the manifest differs by design
    # (`completed` → `reused`) and is compared by status above
    step_files = {k: v for k, v in _files(second.run_root).items() if k != mf.MANIFEST}
    assert step_files == {k: v for k, v in first_files.items() if k != mf.MANIFEST}

    stream = (second.run_root / EVENTS_FILE).read_bytes()
    assert stream.startswith(first_stream), "the first run's lines were rewritten"
    records = read_events(second.run_root / EVENTS_FILE)
    assert [r["seq"] for r in records] == list(range(len(records)))
    second_half = records[len(first_stream.splitlines()) :]
    assert [
        (r["event"], r["payload"].get("step") or r["payload"].get("status"))
        for r in second_half
    ] == [
        item
        for name in loaded.order
        for item in (("phase_started", name), ("phase_completed", name))
    ] + [("campaign_terminal", None)]
    assert all(
        r["payload"]["status"] == "reused"
        for r in second_half
        if r["event"] == "phase_completed"
    )
    assert records[-1]["payload"]["steps"] == {name: "reused" for name in loaded.order}


def test_a_failed_run_warns_and_ends_terminal_failed_because_the_manifest_was_written(
    chain_dir: Path, env: Any, tmp_path: Path
) -> None:
    """T10's second half. The manifest is still written (§8) and the stream
    still says the run ended — as `failed` — because the manifest was
    written: a `campaign_terminal outcome=failed` line **is** present. A run
    that dies before its `finally` leaves neither (the next test)."""
    loaded = _load(chain_dir, env, scripts={"second": "raising.py"})
    with pytest.raises(RuntimeError, match="the step died"):
        run_workflow(loaded, env, tmp_path / "runs", None)
    run_root = tmp_path / "runs" / "chain"
    records = read_events(run_root / EVENTS_FILE)
    events = [(r["event"], r["payload"].get("step")) for r in records]
    assert events == [
        ("phase_started", "first"),
        ("result_committed", "first"),
        ("phase_completed", "first"),
        ("phase_started", "second"),
        ("warning", "second"),
        ("campaign_terminal", None),
    ]
    warning = records[4]["payload"]
    assert warning["reason"] == "attempt_failed"
    assert warning["error"] == {"type": "RuntimeError", "message": "the step died"}
    assert records[-1]["payload"] == {
        "outcome": "failed",
        "steps": {"first": "completed", "second": "failed", "third": "blocked"},
    }
    manifest = json.loads((run_root / mf.MANIFEST).read_text())
    assert records[-1]["payload"]["steps"] == {
        n: e["status"] for n, e in manifest["steps"].items()
    }


def test_a_run_that_dies_before_the_manifest_leaves_no_terminal_line(
    chain_dir: Path, env: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The absence that "did not finish" reads as: the manifest write itself
    fails on a clean run, so nothing may claim the run ended."""

    def dying(*args: Any, **kwargs: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(runner, "write_manifest", dying)
    loaded = _load(chain_dir, env)
    with pytest.raises(OSError, match="disk full"):
        run_workflow(loaded, env, tmp_path / "runs", None)
    stream = tmp_path / "runs" / "chain" / EVENTS_FILE
    assert not terminal(stream)
    assert read_events(stream)[-1]["event"] == "phase_completed"


@pytest.mark.parametrize(
    "interrupt",
    [KeyboardInterrupt(), SystemExit(3)],
    ids=["KeyboardInterrupt", "SystemExit"],
)
def test_an_interrupted_run_writes_its_manifest_and_no_terminal_line(
    chain_dir: Path,
    env: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interrupt: BaseException,
) -> None:
    """Interrupted is not failed. A `BaseException` that is not an `Exception`
    raised inside a step still gets the `finally`'s manifest — the reached
    steps classified, the chain behind the interrupted one `blocked`, an
    independent unreached step `pending` — but **no** `campaign_terminal`
    line: the run did not reach its end, so `terminal()` reads False exactly
    as it does for a run that died before its `finally`, and the interrupt
    propagates as itself (`SystemExit` keeping its code). A step *failure*
    (the previous-but-one test) ends `outcome=failed`; this does not."""
    real = runner._run_script_step  # pyright: ignore[reportPrivateUsage]

    def interrupted(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "second":
            raise interrupt
        return real(name, *args, **kwargs)

    monkeypatch.setattr(runner, "_run_script_step", interrupted)
    loaded = _load(chain_dir, env, aside=True)
    assert loaded.order == ("first", "second", "third", "aside")
    with pytest.raises(type(interrupt)) as info:
        run_workflow(loaded, env, tmp_path / "runs", None)
    assert info.value is interrupt
    if isinstance(interrupt, SystemExit):
        assert interrupt.code == 3
    run_root = tmp_path / "runs" / "chain"
    manifest = json.loads((run_root / mf.MANIFEST).read_text())
    assert {n: e["status"] for n, e in manifest["steps"].items()} == {
        "first": "completed",
        "second": "failed",
        "third": "blocked",
        "aside": "pending",
    }
    assert manifest["steps"]["second"]["error"]["type"] == type(interrupt).__name__
    records = read_events(run_root / EVENTS_FILE)
    assert [(r["event"], r["payload"].get("step")) for r in records] == [
        ("phase_started", "first"),
        ("result_committed", "first"),
        ("phase_completed", "first"),
        ("phase_started", "second"),
        ("warning", "second"),
    ]
    assert records[-1]["payload"]["reason"] == "attempt_failed"
    assert not terminal(run_root / EVENTS_FILE), "an interrupted run is not finished"
    assert not any(r["event"] == "campaign_terminal" for r in records)


@pytest.mark.parametrize("damage", ["torn_tail", "seq_gap"])
def test_a_pre_existing_stream_the_runner_cannot_read_refuses_the_run_before_it_starts(
    chain_dir: Path, env: Any, tmp_path: Path, damage: str
) -> None:
    """Opening the log reads the existing `events.jsonl` back to
    continue its `seq`; one it cannot read — a torn tail, or a hole in the
    sequence — is a `ProtocolError` chaining the reader's error, raised before
    any step: no `workflow.json`, no attempt directory, no step directory, and
    the damaged stream byte-for-byte as it was. Its twin is
    `test_resume_ignores_the_stream_and_appends_to_it`: an intact stream is
    continued."""
    run_root = tmp_path / "runs" / "chain"
    run_root.mkdir(parents=True)
    stream = run_root / EVENTS_FILE
    line = {"schema_version": 1, "event": "phase_started", "payload": {}}
    if damage == "torn_tail":
        text = json.dumps({**line, "seq": 0}) + "\n" + '{"schema_version": 1, "ev'
    else:
        text = (
            json.dumps({**line, "seq": 0})
            + "\n"
            + json.dumps({**line, "seq": 2})
            + "\n"
        )
    stream.write_text(text)

    loaded = _load(chain_dir, env)
    with pytest.raises(ProtocolError, match="cannot be read") as info:
        run_workflow(loaded, env, tmp_path / "runs", None)
    assert isinstance(info.value.__cause__, ValueError)
    assert str(stream) in str(info.value.__cause__)
    assert "move the sidecar aside" in str(info.value)
    assert stream.read_text() == text, "the damaged stream was touched"
    assert not (run_root / mf.MANIFEST).exists()
    assert not (run_root / f"{mf.MANIFEST}.tmp").exists()
    assert not (run_root / mf.ATTEMPTS_DIR).exists()
    assert sorted(p.name for p in run_root.iterdir()) == [EVENTS_FILE]


def test_a_manifest_the_finally_withheld_on_a_clean_run_is_refused_not_asserted(
    chain_dir: Path, env: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The returned scientific record is guarded by a `ProtocolError`, not by
    an `assert` that `python -O` drops. Unreachable by construction — the
    derivation returns `None` only while a failure is propagating out of the
    `finally` — so the derivation is stubbed to withhold on a clean run."""
    monkeypatch.setattr(runner, "_derived_statuses", lambda *a, **k: None)
    loaded = _load(chain_dir, env)
    with pytest.raises(ProtocolError, match="no manifest to return"):
        run_workflow(loaded, env, tmp_path / "runs", None)
    run_root = tmp_path / "runs" / "chain"
    assert not (run_root / mf.MANIFEST).exists()
    assert not terminal(run_root / EVENTS_FILE), "a terminal line without a manifest"


# --------------------------------------------------------------------------- #
# T12 — the legitimate population
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", ["mean_ablation.json", "weekdays.json"])
def test_the_event_layer_is_in_no_shipped_scripts_closure(env: Any, name: str) -> None:
    """`events.py` and `derived.py` are in no shipped script's repository
    closure (spec §4.2) — a layering fact, not an identity one: no shipped
    entry carries a closure key at all."""
    from causalab.protocol.identity import import_closure

    loaded = load_workflow(WORKFLOWS / name, env)
    for entry in loaded.canonical["steps"].values():
        assert "closure" not in entry
        if entry["type"] == "script":
            module = REPO / (entry["script"]["module"].replace(".", "/") + ".py")
            closure = import_closure(module, root=REPO)
            assert "causalab/io/events.py" not in closure
            assert "causalab/workflow/derived.py" not in closure
