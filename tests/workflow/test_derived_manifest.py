"""`workflow.json`'s status words are derived from the event stream (workflow
spec §4.3, §8).

The three-step script chain from `test_attempt_publish.py`, on CPU with no
engine. **T13**: for a clean run, a `--resume` run and a run with one raising
step, `derive_statuses` over the lines the run appended equals the status map
`workflow.json` carries **and** the `campaign_terminal` line's `steps` — the
manifest cannot disagree with execution history because it is derived from
it. The *mutations* are run in-test: an in-memory status the log never
recorded is refused before the manifest is written, naming the step and both
words (and chaining the step failure in flight); a runner that reads no log
is refused the same way; and with the runner's derivation stubbed to mirror
the mutated memory, the manifest is written wrong and regenerating from the
log is what catches it. The derivation on an empty log, and on a log holding
only `phase_started`, is `pending` for every step (§8 has no word for
"started"). **When the stream cannot decide** no manifest is written and the
failure in flight is left as it was: a stream torn while a step failure
propagates is a `ProtocolWarning` beside it (the failure keeps its type); torn
on a clean run it is a `ProtocolError` chaining the read error; and an
interrupt (`KeyboardInterrupt`, `SystemExit`) landing between a memory
assignment and its emit propagates as itself — never downgraded to a
`ProtocolError` — with `--resume` reusing the unit it left published. **T14**
: appending tracker lines to the stream
after scientific completion moves no checksum of any other file — while
`terminal()` flips to `False` and `workflow.json` still says what it said,
which is the two lifecycles — and a second `--resume` run onto that tree
appends to the stream, reuses every step and leaves every step directory's
checksums unchanged. `_step.json` is **not** derived (its `status: completed`
is the attempt's verified record, written before the publish the stream
narrates); that is pinned here too, with the reason.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from causalab.io.events import EVENTS_FILE, EventLog, read_events, terminal
from causalab.io.step_record import SIDECAR
from causalab.protocol.rules.errors import ProtocolError, ProtocolWarning
from causalab.workflow import manifest as mf
from causalab.workflow import runner
from causalab.workflow.derived import derive_statuses
from causalab.workflow.document import load_workflow
from causalab.workflow.runner import run_workflow

from tests.workflow.test_attempt_publish import (
    FIRST,
    RAISING,
    SECOND,
    THIRD,
    InjectedFailure,
    _document,  # pyright: ignore[reportPrivateUsage]
)

pytestmark = pytest.mark.unit


@pytest.fixture()
def chain_dir(tmp_path: Path) -> Path:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "first.py").write_text(FIRST)
    (scripts / "second.py").write_text(SECOND)
    (scripts / "third.py").write_text(THIRD)
    (scripts / "raising.py").write_text(RAISING)
    return tmp_path


def _load(chain_dir: Path, env: Any, **kwargs: Any) -> Any:
    return load_workflow(_document(**kwargs), env, workflow_dir=chain_dir)


def _statuses(manifest: dict[str, Any]) -> dict[str, str]:
    return {name: entry["status"] for name, entry in manifest["steps"].items()}


def _manifest(run_root: Path) -> dict[str, Any]:
    return json.loads((run_root / mf.MANIFEST).read_text())


def _derived(loaded: Any, records: list[dict[str, Any]]) -> dict[str, str]:
    return derive_statuses(
        records, order=loaded.order, dependencies=loaded.dependencies
    )


def _own(records: list[dict[str, Any]], opened_at: int) -> list[dict[str, Any]]:
    """The lines one opening of the log appended (`EventLog.opened_at`)."""
    return [r for r in records if r["seq"] >= opened_at]


def _checksums(root: Path) -> dict[str, str]:
    """sha256 of every file under ``root`` except the stream — walked sorted,
    keyed by relative path — the run tree's scientific bytes."""
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.name != EVENTS_FILE
    }


# --------------------------------------------------------------------------- #
# T13 — the manifest's statuses are the stream's
# --------------------------------------------------------------------------- #


def test_t13_a_clean_run_derives_completed_for_every_step(
    chain_dir: Path, env: Any, tmp_path: Path
) -> None:
    loaded = _load(chain_dir, env)
    result = run_workflow(loaded, env, tmp_path / "runs", None)
    records = read_events(result.run_root / EVENTS_FILE)
    derived = _derived(loaded, records)

    assert derived == {name: "completed" for name in loaded.order}
    assert derived == _statuses(_manifest(result.run_root))
    assert derived == _statuses(dict(result.manifest))
    # `campaign_terminal.steps` is a view of the same derivation
    assert records[-1]["event"] == "campaign_terminal"
    assert records[-1]["payload"]["steps"] == derived


def test_t13_a_resume_run_derives_reused_from_its_own_lines(
    chain_dir: Path, env: Any, tmp_path: Path
) -> None:
    """The second run's statuses come from the lines *it* appended; the first
    run's `completed` lines before them are the first run's history."""
    loaded = _load(chain_dir, env)
    out = tmp_path / "runs"
    first = run_workflow(loaded, env, out, None)
    first_records = read_events(first.run_root / EVENTS_FILE)

    second = run_workflow(loaded, env, out, None, resume=True)
    records = read_events(second.run_root / EVENTS_FILE)
    own = _own(records, opened_at=len(first_records))
    assert own and own[0]["seq"] == first_records[-1]["seq"] + 1
    derived = _derived(loaded, own)

    assert derived == {name: "reused" for name in loaded.order}
    assert derived == _statuses(_manifest(second.run_root))
    assert derived == _statuses(dict(second.manifest))
    assert records[-1]["payload"]["steps"] == derived
    # the first run's lines still derive the first run's word — history is
    # not rewritten — and the whole stream derives the latest word
    assert _derived(loaded, first_records) == {
        name: "completed" for name in loaded.order
    }
    assert _derived(loaded, records) == derived


def test_t13_a_failed_run_derives_failed_blocked_and_pending(
    chain_dir: Path, env: Any, tmp_path: Path
) -> None:
    """`second` raises: `first` completed, `second` failed, `third` blocked
    (downstream of the failure), `aside` pending (independent, unreached)."""
    loaded = _load(chain_dir, env, aside=True, scripts={"second": "raising.py"})
    assert loaded.order == ("first", "second", "third", "aside")
    with pytest.raises(RuntimeError, match="the step died"):
        run_workflow(loaded, env, tmp_path / "runs", None)
    run_root = tmp_path / "runs" / "chain"
    records = read_events(run_root / EVENTS_FILE)
    derived = _derived(loaded, records)

    assert derived == {
        "first": "completed",
        "second": "failed",
        "third": "blocked",
        "aside": "pending",
    }
    manifest = _manifest(run_root)
    assert derived == _statuses(manifest)
    assert manifest["steps"]["third"]["blocked_by"] == ["second"]
    assert manifest["steps"]["second"]["error"]["message"] == "the step died"
    assert records[-1]["payload"] == {"outcome": "failed", "steps": derived}


# --------------------------------------------------------------------------- #
# T13's mutations — run, not described
# --------------------------------------------------------------------------- #


def _memory_says_completed(
    order: tuple[str, ...], dependencies: Any, steps: Any, **_: Any
) -> dict[str, dict[str, Any]]:
    """The mutation: an in-memory word the log never recorded — every
    unreached step marked `completed`, as if it had been published."""
    return {name: {"status": "completed"} for name in order if name not in steps}


def test_a_skipped_step_derives_skipped_and_its_unreached_dependent_follows() -> None:
    """§2.8, §8: `phase_completed {status: skipped}` is a terminal
    word — the step derives `skipped`, never `failed` or `blocked` — and a
    dependent the run never reached derives `skipped` too, by the manifest's
    own rule, before the `blocked` rule: nothing upstream failed, a decision
    took the step out of the run."""
    order = ("measure", "gate", "fit", "probe", "report")
    deps = {
        "measure": (),
        "gate": ("measure",),
        "fit": ("gate",),
        "probe": ("gate",),
        "report": ("fit",),
    }
    by = {"conditional": "gate", "evidence_identity": "abc:def", "transitive_from": []}
    records = [
        {"seq": 0, "event": "phase_started", "payload": {"step": "measure"}},
        {
            "seq": 1,
            "event": "phase_completed",
            "payload": {"step": "measure", "status": "completed"},
        },
        {
            "seq": 2,
            "event": "phase_completed",
            "payload": {"step": "gate", "status": "completed"},
        },
        {"seq": 3, "event": "phase_started", "payload": {"step": "fit"}},
        {
            "seq": 4,
            "event": "phase_completed",
            "payload": {"step": "fit", "status": "skipped", "skipped_by": by},
        },
    ]
    assert derive_statuses(records, order=order, dependencies=deps) == {
        "measure": "completed",
        "gate": "completed",
        "fit": "skipped",
        "probe": "pending",
        "report": "skipped",
    }
    # the manifest's own rule, on the same facts: the dependent is `skipped`
    # with the step it followed, and a failed upstream would make it `blocked`
    entries = mf.classify_unreached(
        order,
        deps,
        {
            "measure": {"status": "completed"},
            "gate": {"status": "completed"},
            "fit": {"status": "skipped", "skipped_by": by},
        },
    )
    # one shape for a reached and an unreached skip:
    # the six `SKIPPED_BY_FIELDS`, the decision fields inherited from `fit`
    assert entries["report"] == {
        "status": "skipped",
        "skipped_by": {
            "conditional": "gate",
            "decision_step": None,
            "decision_type": None,
            "outcome": None,
            "evidence_identity": "abc:def",
            "transitive_from": ["fit"],
        },
    }
    assert set(entries["report"]["skipped_by"]) == set(mf.SKIPPED_BY_FIELDS)
    assert entries["probe"] == {"status": "pending"}
    blocked = mf.classify_unreached(
        order, deps, {"measure": {"status": "completed"}, "gate": {"status": "failed"}}
    )
    assert (
        blocked["fit"]["status"] == "blocked"
        and blocked["report"]["status"] == "blocked"
    )


def test_a_nested_workflows_steps_derive_under_their_flattened_names() -> None:
    """§2.10: a nested workflow's steps are in `order` as
    `<step>/<inner>` and the stream narrates them under those names, so the
    derivation reads them like any step — and refuses a line naming the
    `workflow` step itself, which is in no order and has no status."""
    order = ("measure", "tail/measure", "tail/gate_k", "tail/best", "report")
    deps = {
        "measure": (),
        "tail/measure": (),
        "tail/gate_k": ("tail/measure",),
        "tail/best": ("tail/measure",),
        "report": ("tail/best",),
    }
    records = [
        {"seq": i, "event": "phase_completed", "payload": {"step": s, "status": w}}
        for i, (s, w) in enumerate(
            [
                ("measure", "completed"),
                ("tail/measure", "completed"),
                ("tail/gate_k", "reused"),
                ("tail/best", "skipped"),
            ]
        )
    ]
    assert derive_statuses(records, order=order, dependencies=deps) == {
        "measure": "completed",
        "tail/measure": "completed",
        "tail/gate_k": "reused",
        "tail/best": "skipped",
        "report": "skipped",
    }
    container = {
        "seq": 4,
        "event": "phase_completed",
        "payload": {"step": "tail", "status": "completed"},
    }
    with pytest.raises(ValueError, match="names step 'tail'"):
        derive_statuses([*records, container], order=order, dependencies=deps)


def test_a_selected_join_derives_completed_beside_a_skipped_child() -> None:
    """§2.9: a join declaring `require: selected` is not skipped by
    its own skipped children — an unreached one derives `pending` (nothing
    upstream failed, the join has children left), while an `all` join, or a
    selective join whose *other* upstream is skipped, derives `skipped` by
    the manifest's own rule (`propagates`)."""
    order = ("gate@0", "gate@1", "apply@0", "apply@1", "apply", "report")
    deps = {
        "gate@0": (),
        "gate@1": (),
        "apply@0": ("gate@0",),
        "apply@1": ("gate@1",),
        "apply": ("apply@0", "apply@1"),
        "report": ("apply",),
    }
    reached = {
        "gate@0": {"status": "completed"},
        "gate@1": {"status": "completed"},
        "apply@0": {"status": "completed"},
        "apply@1": {"status": "skipped"},
    }
    selected = mf.classify_unreached(
        order, deps, reached, selective=frozenset({"apply"})
    )
    assert selected["apply"] == {"status": "pending"}
    assert selected["report"] == {"status": "pending"}
    everything = mf.classify_unreached(order, deps, reached)
    # the six `SKIPPED_BY_FIELDS`, all `None` here:
    # the skipped child carries no `skipped_by` block to inherit from
    bare = {field: None for field in mf.SKIPPED_BY_FIELDS}
    assert everything["apply"] == {
        "status": "skipped",
        "skipped_by": {**bare, "transitive_from": ["apply@1"]},
    }
    assert everything["report"]["status"] == "skipped"
    # a skip from a non-child upstream skips a selective join like any step
    deps_after = {**deps, "apply": ("apply@0", "apply@1", "measure")}
    reached_after = {**reached, "measure": {"status": "skipped"}}
    other = mf.classify_unreached(
        ("measure", *order), deps_after, reached_after, selective=frozenset({"apply"})
    )
    assert other["apply"]["skipped_by"] == {**bare, "transitive_from": ["measure"]}
    assert mf.propagates("apply@1", "apply", frozenset({"apply"})) is False
    assert mf.propagates("apply@1", "apply", frozenset()) is True
    assert mf.propagates("apply@1", "report", frozenset({"apply"})) is True
    # through the stream: the same facts derive the same words
    records = [
        {"seq": i, "event": "phase_completed", "payload": {"step": s, "status": w}}
        for i, (s, w) in enumerate(
            [
                ("gate@0", "completed"),
                ("gate@1", "completed"),
                ("apply@0", "completed"),
                ("apply@1", "skipped"),
            ]
        )
    ]
    assert (
        derive_statuses(
            records, order=order, dependencies=deps, selective=frozenset({"apply"})
        )["apply"]
        == "pending"
    )
    assert (
        derive_statuses(records, order=order, dependencies=deps)["apply"] == "skipped"
    )


def test_t13_mutation_an_in_memory_status_the_log_never_recorded_is_refused(
    chain_dir: Path, env: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`second` raises and the runner's memory calls `third` `completed`; the
    log recorded no publish for `third`, so it derives `blocked`. The manifest
    is not written; the refusal names the step and both words and chains the
    step failure that was in flight."""
    monkeypatch.setattr(runner, "classify_unreached", _memory_says_completed)
    loaded = _load(chain_dir, env, scripts={"second": "raising.py"})
    with pytest.raises(ProtocolError, match=r"'third'.*'completed'.*'blocked'") as info:
        run_workflow(loaded, env, tmp_path / "runs", None)
    assert isinstance(info.value.__cause__, RuntimeError)
    assert str(info.value.__cause__) == "the step died"
    run_root = tmp_path / "runs" / "chain"
    assert not (run_root / mf.MANIFEST).exists(), "written disagreeing with the log"
    assert not (run_root / f"{mf.MANIFEST}.tmp").exists()
    stream = run_root / EVENTS_FILE
    assert not terminal(stream), "a terminal line without a manifest"
    assert read_events(stream)[-1]["event"] == "warning"


def test_t13_mutation_a_runner_that_reads_no_log_is_refused(
    chain_dir: Path, env: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The derivation really reads the stream: a runner handed an empty
    history derives `pending` for a step its memory published, and refuses."""
    monkeypatch.setattr(runner, "read_events", lambda path: [])
    loaded = _load(chain_dir, env)
    with pytest.raises(ProtocolError, match=r"'first'.*'completed'.*'pending'") as info:
        run_workflow(loaded, env, tmp_path / "runs", None)
    assert info.value.__cause__ is None  # no step failure was in flight
    run_root = tmp_path / "runs" / "chain"
    assert not (run_root / mf.MANIFEST).exists()
    # the steps themselves were published — the refusal is about the record
    for name in loaded.order:
        assert (run_root / name / SIDECAR).is_file()


def test_t13_mutation_with_the_derivation_stubbed_the_regenerated_status_differs(
    chain_dir: Path, env: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The base's behaviour, reproduced: the runner's memory is mutated *and*
    its derivation is stubbed to mirror the mutation, so the check cannot
    fire and the manifest is written saying `third` completed. Regenerating
    the statuses from the log — T13 — is what shows the manifest wrong."""
    real = runner.derive_statuses

    def mirroring(
        records: Any, *, order: Any, dependencies: Any, **kw: Any
    ) -> dict[str, str]:
        honest = real(records, order=order, dependencies=dependencies, **kw)
        return {
            name: "completed" if word in ("blocked", "pending") else word
            for name, word in honest.items()
        }

    monkeypatch.setattr(runner, "classify_unreached", _memory_says_completed)
    monkeypatch.setattr(runner, "derive_statuses", mirroring)
    loaded = _load(chain_dir, env, scripts={"second": "raising.py"})
    with pytest.raises(RuntimeError, match="the step died"):
        run_workflow(loaded, env, tmp_path / "runs", None)
    run_root = tmp_path / "runs" / "chain"
    written = _statuses(_manifest(run_root))
    assert written["third"] == "completed"  # the lie the base could write
    regenerated = _derived(loaded, read_events(run_root / EVENTS_FILE))
    assert regenerated["third"] == "blocked"
    assert regenerated != written


# --------------------------------------------------------------------------- #
# the derivation alone
# --------------------------------------------------------------------------- #

ORDER = ("first", "second", "third", "aside")
DEPENDENCIES = {
    "first": (),
    "second": ("first",),
    "third": ("first", "second"),
    "aside": (),
}


def _line(event: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {"schema_version": 1, "event": event, "seq": 0, "payload": payload}


def test_the_derivation_on_an_empty_log_is_pending_for_every_step() -> None:
    assert derive_statuses([], order=ORDER, dependencies=DEPENDENCIES) == {
        name: "pending" for name in ORDER
    }


def test_a_started_unfinished_step_is_pending() -> None:
    """§8 has no word for "started": a step whose only line is
    `phase_started` did not reach its end, and nothing upstream failed."""
    records = [_line("phase_started", {"step": "first", "type": "script"})]
    assert derive_statuses(records, order=ORDER, dependencies=DEPENDENCIES) == {
        name: "pending" for name in ORDER
    }


def test_a_failure_blocks_downstream_and_leaves_the_rest_pending() -> None:
    records = [
        _line("phase_started", {"step": "first", "type": "script"}),
        _line("warning", {"step": "first", "reason": "attempt_failed", "error": {}}),
    ]
    assert derive_statuses(records, order=ORDER, dependencies=DEPENDENCIES) == {
        "first": "failed",
        "second": "blocked",
        "third": "blocked",
        "aside": "pending",
    }


def test_a_sink_failure_warning_is_not_a_step_failure() -> None:
    """Only `reason: attempt_failed` is a step's terminal word; the seam's
    own `sink_failed` warnings say nothing about any step."""
    records = [
        _line("phase_started", {"step": "first", "type": "script"}),
        _line("warning", {"reason": "sink_failed", "event": "phase_started", "seq": 0}),
        _line("result_committed", {"step": "first", "files": []}),
        _line("phase_completed", {"step": "first", "status": "completed"}),
        _line(
            "warning", {"reason": "sink_failed", "event": "phase_completed", "seq": 3}
        ),
    ]
    assert derive_statuses(records, order=ORDER, dependencies=DEPENDENCIES) == {
        "first": "completed",
        "second": "pending",
        "third": "pending",
        "aside": "pending",
    }


def test_a_line_the_derivation_cannot_read_is_refused() -> None:
    """Fail-closed: a `phase_completed` with a word outside the two terminal
    ones, or naming a step the workflow has not, is refused rather than
    copied into the manifest. Its twin is every run above."""
    with pytest.raises(ValueError, match="'first' says 'done'"):
        derive_statuses(
            [_line("phase_completed", {"step": "first", "status": "done"})],
            order=ORDER,
            dependencies=DEPENDENCIES,
        )
    with pytest.raises(ValueError, match="names step 'fourth'"):
        derive_statuses(
            [_line("phase_completed", {"step": "fourth", "status": "completed"})],
            order=ORDER,
            dependencies=DEPENDENCIES,
        )
    with pytest.raises(ValueError, match="attempt_failed names step 'fourth'"):
        derive_statuses(
            [_line("warning", {"step": "fourth", "reason": "attempt_failed"})],
            order=ORDER,
            dependencies=DEPENDENCIES,
        )


def test_a_line_whose_payload_is_not_an_object_is_refused() -> None:
    """A well-formed line carrying a list for `payload` is refused like the
    other unreadable lines — a `ValueError` naming the line, not an
    `AttributeError` from inside the derivation."""
    record: dict[str, Any] = {
        "schema_version": 1,
        "event": "phase_completed",
        "seq": 7,
        "payload": ["first", "completed"],
    }
    with pytest.raises(ValueError, match="seq 7: payload is list, not an object"):
        derive_statuses([record], order=ORDER, dependencies=DEPENDENCIES)


# --------------------------------------------------------------------------- #
# when the stream cannot decide: no manifest, the failure in flight unchanged
# --------------------------------------------------------------------------- #


def _tear(stream: Path) -> None:
    """Append a line without its terminator — the tail a process dying inside
    `handle.write` leaves — so `read_events` refuses the stream."""
    with stream.open("a", encoding="utf-8") as handle:
        handle.write('{"schema_version": 1, "event": "progress", "seq": 99')


def test_a_torn_stream_on_a_clean_run_is_refused_and_no_manifest_is_written(
    chain_dir: Path, env: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The manifest is derived from the stream: a stream the runner cannot read
    back is a `ProtocolError` chaining the read error, and no manifest is
    written — never one from memory in the stream's place."""
    loaded = _load(chain_dir, env)
    run_root = tmp_path / "runs" / "chain"

    def tear_after_the_last_publish(name: str, step: str | None) -> None:
        if name == "published" and step == "third":
            _tear(run_root / EVENTS_FILE)

    monkeypatch.setattr(runner, "_boundary", tear_after_the_last_publish)
    with pytest.raises(ProtocolError, match="events.jsonl could not be read") as info:
        run_workflow(loaded, env, tmp_path / "runs", None)
    assert isinstance(info.value.__cause__, ValueError)
    assert "no line terminator" in str(info.value.__cause__)
    assert not (run_root / mf.MANIFEST).exists()
    assert not (run_root / f"{mf.MANIFEST}.tmp").exists()
    # the steps themselves were published — the refusal is about the record
    for name in loaded.order:
        assert (run_root / name / SIDECAR).is_file()


def test_a_torn_stream_does_not_mask_the_step_failure_in_flight(
    chain_dir: Path, env: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The read of the sidecar is held to the rule its writes follow: with a
    step failure propagating, an unreadable stream is a `ProtocolWarning`
    beside it, the manifest is withheld, and the failure keeps its type."""
    loaded = _load(chain_dir, env)
    run_root = tmp_path / "runs" / "chain"

    def tear_and_die(name: str, step: str | None) -> None:
        if name == "published" and step == "first":
            _tear(run_root / EVENTS_FILE)
            raise InjectedFailure("published in first")

    monkeypatch.setattr(runner, "_boundary", tear_and_die)
    with pytest.warns(ProtocolWarning, match="events.jsonl could not be read"):
        with pytest.raises(InjectedFailure, match="published in first"):
            run_workflow(loaded, env, tmp_path / "runs", None)
    assert not (run_root / mf.MANIFEST).exists()
    assert not (run_root / f"{mf.MANIFEST}.tmp").exists()


@pytest.mark.parametrize(
    "interrupt",
    [KeyboardInterrupt(), SystemExit(3)],
    ids=["KeyboardInterrupt", "SystemExit"],
)
def test_an_interrupt_between_memory_and_emit_stays_an_interrupt(
    chain_dir: Path,
    env: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interrupt: BaseException,
) -> None:
    """The window: `first` is published and `completed` in memory when the
    interrupt lands, before `result_committed` is emitted — the stream says
    `pending`. That disagreement is not an emitter bug and is not turned into
    a `ProtocolError`: a `ProtocolWarning` names both words, no manifest is
    written, and the interrupt propagates as itself (`SystemExit` keeping its
    code). `--resume` finds the published unit and reuses it byte-identically."""
    loaded = _load(chain_dir, env)
    out = tmp_path / "runs"
    run_root = out / "chain"
    real_emit = EventLog.emit

    def emit(self: EventLog, event: str, payload: Any = None) -> dict[str, Any]:
        if event == "result_committed" and payload["step"] == "first":
            raise interrupt
        return real_emit(self, event, payload)

    monkeypatch.setattr(EventLog, "emit", emit)
    with pytest.warns(
        ProtocolWarning,
        match=r"'first' is 'completed' in memory and 'pending' on the stream",
    ):
        with pytest.raises(type(interrupt)) as info:
            run_workflow(loaded, env, out, None)
    assert info.value is interrupt
    if isinstance(interrupt, SystemExit):
        assert interrupt.code == 3
    monkeypatch.undo()
    assert not (run_root / mf.MANIFEST).exists()
    assert not (run_root / f"{mf.MANIFEST}.tmp").exists()
    assert (run_root / "first" / SIDECAR).is_file()
    assert [r["event"] for r in read_events(run_root / EVENTS_FILE)] == [
        "phase_started"
    ]
    unit_before = _checksums(run_root / "first")

    second = run_workflow(loaded, env, out, None, resume=True)
    assert _statuses(dict(second.manifest)) == {
        "first": "reused",
        "second": "completed",
        "third": "completed",
    }
    assert _checksums(run_root / "first") == unit_before
    assert _statuses(_manifest(run_root)) == _statuses(dict(second.manifest))
    assert terminal(run_root / EVENTS_FILE)


# --------------------------------------------------------------------------- #
# T14 — two lifecycles
# --------------------------------------------------------------------------- #


def test_t14_tracker_lines_after_completion_move_no_scientific_checksum(
    chain_dir: Path, env: Any, tmp_path: Path
) -> None:
    """The requirement: append tracker logs after scientific completion; the
    scientific manifest and its checksums remain unchanged while the telemetry
    sidecar records the mutation. The asymmetry — `terminal()` flips to False
    while `workflow.json` says what it said — *is* the two lifecycles."""
    loaded = _load(chain_dir, env)
    result = run_workflow(loaded, env, tmp_path / "runs", None)
    run_root = result.run_root
    stream = run_root / EVENTS_FILE
    before = _checksums(run_root)
    manifest_before = (run_root / mf.MANIFEST).read_bytes()
    assert terminal(stream)

    # the tracker log after scientific completion: a fresh opening of the
    # finished stream, two more lines
    tracker = EventLog(stream, identity={})
    tracker.emit("warning", {"reason": "debug_link", "target": "runs/chain/first"})
    tracker.emit("warning", {"reason": "debug_link", "target": "runs/chain/third"})

    assert _checksums(run_root) == before
    assert (run_root / mf.MANIFEST).read_bytes() == manifest_before
    assert not terminal(stream), "the appended lines moved the terminal marker"
    records = read_events(stream)
    assert [r["event"] for r in records[-3:]] == [
        "campaign_terminal",
        "warning",
        "warning",
    ]
    assert [r["seq"] for r in records] == list(range(len(records)))
    # ...and the appended lines change no derived status either
    assert _derived(loaded, records) == _statuses(_manifest(run_root))


def test_t14_a_resume_onto_the_appended_tree_reuses_every_step_and_appends(
    chain_dir: Path, env: Any, tmp_path: Path
) -> None:
    """The manifest may be rewritten by the second run — its statuses change
    from `completed` to `reused` by design (§8: `reused` is the word for a
    unit `--resume` found), so its bytes are not compared. What is: every
    step directory's checksums, the manifest's `steps` map with the status
    word removed (every other field is the earlier run's record, carried
    over), and that the stream was appended to, never rewritten."""
    loaded = _load(chain_dir, env)
    out = tmp_path / "runs"
    first = run_workflow(loaded, env, out, None)
    run_root = first.run_root
    stream = run_root / EVENTS_FILE
    tracker = EventLog(stream, identity={})
    tracker.emit("warning", {"reason": "debug_link", "target": "runs/chain/first"})
    tracker.emit("warning", {"reason": "debug_link", "target": "runs/chain/third"})
    stream_before = stream.read_bytes()
    steps_before = {name: _checksums(run_root / name) for name in loaded.order}
    manifest_before = _manifest(run_root)

    second = run_workflow(loaded, env, out, None, resume=True)
    assert second.run_root == run_root
    manifest_after = _manifest(run_root)
    assert _statuses(manifest_after) == {name: "reused" for name in loaded.order}
    assert {name: _checksums(run_root / name) for name in loaded.order} == steps_before

    def without_status(manifest: dict[str, Any]) -> dict[str, Any]:
        return {
            name: {k: v for k, v in entry.items() if k != "status"}
            for name, entry in manifest["steps"].items()
        }

    assert without_status(manifest_after) == without_status(manifest_before)
    assert stream.read_bytes().startswith(stream_before), "the stream was rewritten"
    assert terminal(stream)
    records = read_events(stream)
    own = _own(records, opened_at=len(stream_before.splitlines()))
    assert _derived(loaded, own) == _statuses(manifest_after)
    assert records[-1]["payload"]["steps"] == _statuses(manifest_after)


def test_step_json_is_not_derived_it_is_the_attempts_verified_record(
    chain_dir: Path, env: Any, tmp_path: Path
) -> None:
    """Decision: `_step.json` keeps its own lifecycle. It is written into the
    attempt directory before the publish — when the stream holds only
    `phase_started` for its step — and its `status: completed` is the fact
    that the attempt verified; `result_committed` narrates its publication.
    After `--resume` the manifest says `reused` while every `_step.json`
    still says `completed`: the record is the earlier run's, byte for byte."""
    loaded = _load(chain_dir, env)
    out = tmp_path / "runs"
    first = run_workflow(loaded, env, out, None)
    sidecars = {
        name: (first.run_root / name / SIDECAR).read_bytes() for name in loaded.order
    }
    for name in loaded.order:
        assert json.loads(sidecars[name])["status"] == "completed"
    second = run_workflow(loaded, env, out, None, resume=True)
    assert _statuses(dict(second.manifest)) == {name: "reused" for name in loaded.order}
    for name in loaded.order:
        assert (second.run_root / name / SIDECAR).read_bytes() == sidecars[name]
