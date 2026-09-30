"""A document run's event stream (workflow spec §4.3; IM spec §9).

**T9**, on tiny-random through the reference engine: the same document run
twice with ``record=True``, once with no sink and once with a sink that raises
on every line —
`protocol.json`, every output file and the reported manifest are
**byte-identical** (`test_run_protocol_api.py`'s comparison, minus the stream
itself), and the stream of the sunk run holds one `warning` per failed
delivery and still ends in `campaign_terminal`. A remote adapter's failure
cannot change scientific execution — measured, not stated.

**T10's second half** needs no model: an engine that raises mid-run leaves a
stream with `phase_started` and no terminal line. A protocol run has no
`--resume` (`run.py`'s module docstring), so there is nothing here for the
stream to be ignored by; the workflow half is `tests/workflow/test_events_workflow.py`.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from causalab.cli import register_model_key
from causalab.neural.shared.engine_router import load_engine
from causalab.io.events import EVENTS, EVENTS_FILE, read_events, terminal
from causalab.protocol import RUN_RECORD_NAME, run_protocol
from causalab.protocol.engine import Engine, RunResult
from causalab.protocol.pipeline import compile_protocol
from causalab.io.env import FileArtifacts, FileDatasets, ResolutionEnv
from causalab.protocol.schema import COMPONENTS

from tests._helpers.stub_engine import stub_record
from tests.neural.engines.pytorch_hooks.conftest import TINY_LLAMA
from tests.protocol._env import CORPUS_DIR, FIXTURES


pytestmark = pytest.mark.smoke

#: The document `test_run_protocol_api.py` runs, for the same reason: two
#: metrics, a counterfactual role, a save manifest with more than one entry.
DOCUMENT = CORPUS_DIR / "02_interchange_im.json"
OVERRIDES = {"model.key": TINY_LLAMA, "sites.target.layers": 1}


class Delivered(Exception):
    pass


def _raising_sink(record: Any) -> None:
    raise Delivered(record["event"])


def _files(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != EVENTS_FILE
    }


@pytest.fixture(scope="module")
def artifacts_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("artifacts")
    shutil.copytree(FIXTURES / "artifacts", root, dirs_exist_ok=True)
    # what `causalab run` does before compiling: the tiny model's static
    # config into the registry, so the retargeted document canonicalizes
    register_model_key({"model": {"key": TINY_LLAMA}})
    return root


@pytest.fixture(scope="module")
def both_runs(tmp_path_factory: pytest.TempPathFactory, artifacts_root: Path):
    """One document, run with no sink and with a sink that raises on every line."""
    env = ResolutionEnv(
        datasets=FileDatasets(root=FIXTURES / "data"),
        artifacts=FileArtifacts(root=artifacts_root),
    )
    loaded = compile_protocol(DOCUMENT, env=env, overrides=OVERRIDES)
    base = tmp_path_factory.mktemp("events")
    plain, sunk = base / "plain", base / "sunk"
    plain_result = run_protocol(
        loaded, env, load_engine("pytorch_hooks", device="cpu"), plain, record=True
    )
    sunk_result = run_protocol(
        loaded,
        env,
        load_engine("pytorch_hooks", device="cpu"),
        sunk,
        record=True,
        sink=_raising_sink,
    )
    return loaded, plain, sunk, plain_result, sunk_result


def test_t9_the_receipt_and_every_output_are_byte_identical(both_runs) -> None:
    _, plain, sunk, plain_result, sunk_result = both_runs
    assert (plain / RUN_RECORD_NAME).read_bytes() == (
        sunk / RUN_RECORD_NAME
    ).read_bytes()
    plain_files, sunk_files = _files(plain), _files(sunk)
    assert set(plain_files) == set(sunk_files)
    assert not [name for name in plain_files if plain_files[name] != sunk_files[name]]
    assert sorted(plain_result.files) == sorted(sunk_result.files)
    assert plain_result.summaries == sunk_result.summaries
    # the stream is not in the receipt, in any output, or in the result
    assert EVENTS_FILE not in (plain / RUN_RECORD_NAME).read_text()
    assert EVENTS_FILE not in plain_result.files


def test_the_stream_sits_beside_the_receipt_with_the_documents_identity(
    both_runs,
) -> None:
    loaded, plain, _, result, _ = both_runs
    records = read_events(plain / EVENTS_FILE)
    assert records, "no stream beside protocol.json"
    receipt = json.loads((plain / RUN_RECORD_NAME).read_text())
    n_points = len(receipt["points"])
    for record in records:
        assert record["document_digest"] == receipt["document_digest"]
        assert record["points"] == [0, n_points]
    assert [r["seq"] for r in records] == list(range(len(records)))
    assert {r["event"] for r in records} <= set(EVENTS)

    events = [r["event"] for r in records]
    assert events[0] == "phase_started"
    assert records[0]["payload"] == {"phase": "execute", "n_points": n_points}
    progress = [r for r in records if r["event"] == "progress"]
    assert [p["payload"]["point_digest"] for p in progress] == [
        point["digest"] for point in receipt["points"]
    ]
    assert [p["payload"]["completed"] for p in progress] == list(range(1, n_points + 1))
    metrics = [r for r in records if r["event"] == "metric"]
    assert {m["payload"]["name"] for m in metrics} == {
        name for summary in result.summaries for name in summary["metrics"]
    }
    committed = [r for r in records if r["event"] == "result_committed"]
    assert len(committed) == 1
    assert committed[0]["payload"] == {"files": sorted(result.files)}
    assert events[-2:] == ["phase_completed", "campaign_terminal"]
    assert records[-2]["payload"] == {"phase": "execute", "forwards": result.forwards}
    assert terminal(plain / EVENTS_FILE)


def test_t9_the_sunk_stream_holds_one_warning_per_failed_delivery(both_runs) -> None:
    _, plain, sunk, _, _ = both_runs
    plain_records = read_events(plain / EVENTS_FILE)
    records = read_events(sunk / EVENTS_FILE)
    delivered = [r for r in records if r["event"] != "warning"]
    warnings_ = [r for r in records if r["event"] == "warning"]
    assert len(warnings_) == len(delivered) == len(plain_records)
    assert [r["event"] for r in delivered] == [r["event"] for r in plain_records]
    for line, warning in zip(delivered, warnings_):
        assert warning["seq"] == line["seq"] + 1
        assert warning["payload"]["reason"] == "sink_failed"
        assert warning["payload"]["event"] == line["event"]
        assert warning["payload"]["seq"] == line["seq"]
    assert records[-2]["event"] == "campaign_terminal"


# --------------------------------------------------------------------------- #
# T10 — a run that raises leaves no terminal line
# --------------------------------------------------------------------------- #


class _Dying(Engine):
    """An engine whose execution raises after it has written the receipt and
    opened the stream — the engine's first move — and before any
    forward."""

    name = "dying"
    capabilities = frozenset(
        {"grad", "paired_forward", "full_logits", "pytorch_fn_local", "generate"}
    )
    components = frozenset(COMPONENTS)
    writable_components = frozenset(COMPONENTS)
    is_local = True

    def execute(self, compiled: Any, run: Any) -> RunResult:
        stub_record(self, compiled, run)
        raise RuntimeError("the forward died")


def test_a_run_that_raises_leaves_a_stream_without_a_terminal_line(
    tmp_path: Path, artifacts_root: Path
) -> None:
    env = ResolutionEnv(
        datasets=FileDatasets(root=FIXTURES / "data"),
        artifacts=FileArtifacts(root=artifacts_root),
    )
    loaded = compile_protocol(DOCUMENT, env=env, overrides=OVERRIDES)
    out = tmp_path / "dead"
    with pytest.raises(RuntimeError, match="the forward died"):
        run_protocol(loaded, env, _Dying(), out, record=True)
    assert (out / RUN_RECORD_NAME).is_file(), "the receipt is written before execution"
    records = read_events(out / EVENTS_FILE)
    assert [r["event"] for r in records] == ["phase_started"]
    assert not terminal(out / EVENTS_FILE)


def test_verbose_progress_lines_follow_the_run(
    tmp_path: Path,
    artifacts_root: Path,
    both_runs,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The lines ``causalab run --verbose`` prints (IM spec §9), on the
    shared execution loop's logger: the selection, each point's model load
    and readiness, its run and completion, and the output write, in run
    order. They are a sidecar of the run, so the receipt and the event
    stream are what a silent run writes (T9 above is the byte comparison)."""
    import logging

    from causalab.cli import VERBOSE_LOGGER

    env = ResolutionEnv(
        datasets=FileDatasets(root=FIXTURES / "data"),
        artifacts=FileArtifacts(root=artifacts_root),
    )
    loaded = compile_protocol(DOCUMENT, env=env, overrides=OVERRIDES)
    with caplog.at_level(logging.INFO, logger=VERBOSE_LOGGER):
        result = run_protocol(
            loaded,
            env,
            load_engine("pytorch_hooks", device="cpu"),
            tmp_path,
            record=True,
        )
    lines = [r.getMessage() for r in caplog.records if r.name == VERBOSE_LOGGER]
    n = len(result.steps)
    digest = result.steps[0].digest[:12]
    assert lines[0] == f"{n} of {n} points selected; engine pytorch_hooks"
    assert lines[1] == f"point 1/{n} {digest}: loading model {TINY_LLAMA}@main"
    assert lines[2].startswith(f"point 1/{n} {digest}: ready in ")
    assert lines[3] == f"point 1/{n} {digest}: running"
    assert lines[4].startswith(f"point 1/{n} {digest}: done in ")
    assert lines[-1] == f"wrote {len(result.files)} file(s) to {tmp_path}"
    assert not any("fitting" in line for line in lines), "no train section here"
    assert terminal(tmp_path / EVENTS_FILE)
    # the sidecar changed nothing on disk: every file but the stream is the
    # silent run's, byte for byte
    _, plain, *_ = both_runs
    assert _files(tmp_path) == _files(plain)
