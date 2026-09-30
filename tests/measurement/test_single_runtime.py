"""Timing-only execution retains outputs without numerical-only work."""

from contextlib import contextmanager
import json
from types import SimpleNamespace

import pytest
from hypothesis import given, strategies as st

from causalab.measurement.analysis.receipts import load_measurement
from causalab.measurement.collection import Operation, collect, file_hash, write_record
from causalab.measurement.runtime.worker import Worker
from causalab.measurement.study.scheduler import run_schedule, schedule

pytestmark = pytest.mark.unit


def plan():
    return {
        "mode": "single",
        "observation_policy": "not_requested",
        "observations": {},
        "arms": {"source": {}},
        "cases": {"workflow": {"kind": "workflow"}},
        "warmups": 1,
        "seeds": [0, 1],
        "repeats": 2,
        "order_seed": 3,
        "profile": {"cases": []},
    }


def forbidden(*args, **kwargs):
    pytest.fail("numerical-only operation was invoked")


def test_collector_skips_numerics_and_keeps_trace(tmp_path):
    calls = []

    @contextmanager
    def prepare(seed, directory):
        def run():
            calls.append(directory.name)
            (directory / "saved.txt").write_text(str(seed))

        yield Operation(run, forbidden, numerics_context=forbidden)

    receipt = collect(
        prepare,
        tmp_path / "result",
        case="workflow",
        input_identity="input",
        scope="workflow",
        reset_policy="fresh",
        seeds=[3],
        repeats=2,
        warmups=1,
        mode="single",
        observation_policy="not_requested",
        profile=True,
    )
    record, samples = load_measurement(receipt, require_observations=False)
    assert len(calls) == 4
    assert not any("numerics" in name for name in calls)
    assert not list(receipt.parent.rglob("*.safetensors"))
    assert record["trace"]["status"] == "completed"
    assert record["trace"]["observation_status"] == "not_requested"
    assert all(s["observation_status"] == "not_requested" for s in samples.values())
    with pytest.raises(ValueError, match="requires observations"):
        load_measurement(receipt)


def make_worker(tmp_path):
    worker = object.__new__(Worker)
    worker.plan = plan()
    worker.config = {"device": "cpu", "input_identity": "fixed"}
    worker.identity = {"source": "fixed"}
    worker.fitting = {"fit"}
    worker.loaded = SimpleNamespace(document=SimpleNamespace(output_dir="outputs"))

    @contextmanager
    def prepare(case, seed, directory):
        def run():
            output = directory / "outputs"
            output.mkdir()
            (output / "saved.txt").write_text(str(seed))

        yield Operation(run, forbidden)

    worker.prepare = prepare
    return worker


def test_worker_binds_saved_outputs_without_observations(tmp_path):
    worker = make_worker(tmp_path)
    receipt = worker.sample(
        {"case": "workflow", "seed": 0, "directory": str(tmp_path / "run")}
    )
    record, samples = load_measurement(receipt, require_observations=False)
    sample = samples[0, 0]
    assert len(sample["output_files"]) == 1
    assert sample["observer_check"] == {"status": "not_requested"}
    saved = receipt.parent / next(iter(sample["output_files"]))
    assert saved.read_text() == "0"
    saved.write_text("tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_measurement(receipt, require_observations=False)


@given(st.integers(1, 8), st.integers(1, 5), st.integers(0, 100))
def test_single_schedule_cardinality(seeds, repeats, order_seed):
    authored = {
        **plan(),
        "seeds": list(range(seeds)),
        "repeats": repeats,
        "order_seed": order_seed,
    }
    cells = schedule(authored)
    assert len(cells) == seeds * repeats
    assert all(cell["arms"] == ["source"] for cell in cells)
    assert {(c["seed"], c["repeat"]) for c in cells} == {
        (s, r) for s in range(seeds) for r in range(repeats)
    }


def test_single_scheduler_resume_and_output_integrity(tmp_path):
    calls = []

    @contextmanager
    def open_session(arm):
        assert arm == "source"
        worker = make_worker(tmp_path)
        sample = worker.sample

        def collect_sample(case, seed, repeat, directory):
            calls.append((seed, repeat))
            return sample({"case": case, "seed": seed, "directory": str(directory)})

        worker.sample = collect_sample
        yield worker

    output = tmp_path / "study"
    grouped = run_schedule(plan(), {}, output, open_session)
    assert len(calls) == 4
    receipt = grouped["workflow"]["source"]
    _, samples = load_measurement(receipt, require_observations=False)
    assert len(samples) == 4
    assert all(
        (output / name).is_file()
        for s in samples.values()
        for name in s["output_files"]
    )
    run_schedule(plan(), {}, output, open_session, resume=True)
    assert len(calls) == 4
    saved = output / next(iter(samples[0, 0]["output_files"]))
    saved.write_text("tampered")
    with pytest.raises(ValueError, match="block changed"):
        run_schedule(plan(), {}, output, open_session, resume=True)


def test_scheduler_rejects_receipt_downgrade(tmp_path):
    @contextmanager
    def open_session(arm):
        worker = make_worker(tmp_path)
        sample = worker.sample
        worker.sample = lambda case, seed, repeat, directory: sample(
            {"case": case, "seed": seed, "directory": str(directory)}
        )
        yield worker

    authored = {
        **plan(),
        "observation_policy": "required",
        "observations": {"value": {}},
    }
    with pytest.raises(ValueError, match="observation policy"):
        run_schedule(authored, {}, tmp_path / "study", open_session)
    assert not list((tmp_path / "study" / "blocks").iterdir())


def test_single_profiles_follow_clean_samples_and_resume_without_recollection(tmp_path):
    events = []
    interrupted = [True]

    @contextmanager
    def open_session(arm):
        worker = make_worker(tmp_path)
        sample = worker.sample

        def collect_sample(case, seed, repeat, directory):
            events.append("clean")
            return sample({"case": case, "seed": seed, "directory": str(directory)})

        def capture(case, seed, repeat, directory):
            events.append("capture")
            directory.mkdir()
            if interrupted[0]:
                (directory / "partial.trace").write_text("incomplete")
                raise KeyboardInterrupt("capture interrupted")
            artifact = directory / "trace.json"
            artifact.write_text('{"traceEvents": []}')
            saved = directory / "saved.txt"
            saved.write_text("profile output")
            receipt = directory / "measurement.json"
            write_record(
                receipt,
                {
                    "mode": "single",
                    "observation_policy": "not_requested",
                    "capture_plan": {"backends": ["torch"], "modes": ["warm"]},
                    "captures": [
                        {
                            "id": "torch:warm",
                            "pair_id": "torch",
                            "backend": "torch",
                            "mode": "warm",
                            "status": "completed",
                            "seed": seed,
                            "repeat": repeat,
                            "observation_status": "not_requested",
                            "output_files": {saved.name: file_hash(saved)},
                            "artifacts": [
                                {"file": artifact.name, "sha256": file_hash(artifact)}
                            ],
                        }
                    ],
                },
            )
            return receipt

        worker.sample, worker.capture = collect_sample, capture
        yield worker

    authored = {**plan(), "profile": {"cases": ["workflow"], "backends": {"torch": {}}}}
    output = tmp_path / "study"
    with pytest.raises(KeyboardInterrupt, match="capture interrupted"):
        run_schedule(authored, {}, output, open_session)
    assert events == ["clean"] * 4 + ["capture"]
    assert len(list((output / "blocks").iterdir())) == 4
    interrupted[0] = False
    result = run_schedule(authored, {}, output, open_session, resume=True)
    assert events == ["clean"] * 4 + ["capture"] * 2
    record, _ = load_measurement(
        result["workflow"]["source"], require_observations=False
    )
    assert record["captures"][0]["status"] == "completed"
    saved = output / next(iter(record["captures"][0]["output_files"]))
    assert saved.read_text() == "profile output"
    run_schedule(authored, {}, output, open_session, resume=True)
    assert len(events) == 6
    saved.write_text("changed")
    with pytest.raises(ValueError, match="profiler capture changed"):
        run_schedule(authored, {}, output, open_session, resume=True)


@pytest.mark.parametrize("missing", [False, True])
def test_single_requested_observations_remain_required(tmp_path, missing):
    import torch

    @contextmanager
    def prepare(seed, directory):
        yield Operation(
            lambda: torch.tensor([float(seed)]),
            lambda result: {} if missing else {"value": result},
        )

    if missing:
        with pytest.raises(ValueError, match="observations need"):
            collect(
                prepare,
                tmp_path / "result",
                case="case",
                input_identity="fixed",
                scope="op",
                reset_policy="fresh",
                seeds=[0],
                repeats=1,
                mode="single",
                observation_policy="required",
            )
        assert (
            json.loads((tmp_path / "result/measurement.json").read_text())["status"]
            == "failed"
        )
    else:
        receipt = collect(
            prepare,
            tmp_path / "result",
            case="case",
            input_identity="fixed",
            scope="op",
            reset_policy="fresh",
            seeds=[0],
            repeats=1,
            mode="single",
            observation_policy="required",
        )
        record, samples = load_measurement(receipt)
        assert record["mode"] == "single"
        assert "value" in samples[0, 0]["tensors"]
