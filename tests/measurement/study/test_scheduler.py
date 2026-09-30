"""The process boundary is replaced; real collection and artifact checks run."""

from contextlib import contextmanager
import json

import pytest
import torch

from causalab.measurement.analysis.compare import compare
from causalab.measurement.collection import Operation, collect
from causalab.measurement.study.scheduler import run_schedule, schedule

pytestmark = pytest.mark.unit


def plan():
    return {
        "arms": {"before": {}, "after": {}},
        "seeds": [0, 1],
        "repeats": 2,
        "warmups": 0,
        "order_seed": 3,
        "cases": {"apply": {}},
        "profile": {"cases": []},
    }


def test_schedule_balances_order_and_pairs_repeats():
    cells = schedule(plan())
    assert cells == schedule(plan())
    assert {tuple(c["arms"]) for c in cells} == {
        ("before", "after"),
        ("after", "before"),
    }
    assert len({(c["case"], c["seed"], c["repeat"]) for c in cells}) == 4
    assert sum(c["arms"][0] == "before" for c in cells) == 2


class Boundary:
    def __init__(self):
        self.calls = []
        self.live = 0
        self.fail = None
        self.version = 1

    @contextmanager
    def open(self, arm):
        assert self.live == 0, "full-size arm models must not coexist"
        self.live += 1
        parent = self

        class Worker:
            identity = {"arm": arm, "version": parent.version}

            def sample(self, case, seed, repeat, directory):
                parent.calls.append((arm, seed, repeat))
                if parent.fail == len(parent.calls):
                    raise RuntimeError("process failure")

                @contextmanager
                def prepare(seed, work):
                    tensor = torch.tensor([float(seed)])
                    yield Operation(lambda: tensor + 1, lambda result: {"site": result})

                return collect(
                    prepare,
                    directory,
                    case=case,
                    input_identity="fixed",
                    scope="op",
                    reset_policy="fresh",
                    seeds=[seed],
                    repeats=1,
                    warmups=0,
                )

        try:
            yield Worker()
        finally:
            self.live -= 1


def test_resume_reuses_only_verified_paired_blocks(tmp_path):
    boundary = Boundary()
    boundary.fail = 4  # First pair complete; second pair loses its second arm.
    output = tmp_path / "study"
    with pytest.raises(RuntimeError, match="process failure"):
        run_schedule(plan(), {"input": 1}, output, boundary.open)
    partial = json.loads((output / "study.json").read_text())
    assert len(partial["blocks"]) == 1
    assert partial["status"] == "failed"
    boundary.fail = None
    result = run_schedule(plan(), {"input": 1}, output, boundary.open, resume=True)
    assert len(boundary.calls) == 10  # Reruns BOTH sides of the incomplete pair.
    report = compare(
        result["apply"]["before"], result["apply"]["after"], bootstrap_draws=100
    )
    assert len(report["timing"]["per_seed"]) == 2
    assert (
        report["observations"]["site"]["across_seed_means"]["mean_output_drift"]["rms"]
        == 0
    )
    run_schedule(plan(), {"input": 1}, output, boundary.open, resume=True)
    assert len(boundary.calls) == 10
    block = next((output / "blocks").glob("*/block.json"))
    details = json.loads(block.read_text())
    artifact = block.parent / next(
        name for name in details["files"] if name.endswith(".safetensors")
    )
    artifact.write_bytes(b"changed")
    with pytest.raises(ValueError, match="block changed"):
        run_schedule(plan(), {"input": 1}, output, boundary.open, resume=True)


def test_identity_changes_refuse_resume_before_reuse(tmp_path):
    boundary = Boundary()
    output = tmp_path / "study"
    run_schedule(plan(), {"input": 1}, output, boundary.open)
    with pytest.raises(ValueError, match="authored study"):
        run_schedule(plan(), {"input": 2}, output, boundary.open, resume=True)
    boundary.version = 2
    with pytest.raises(ValueError, match="observed before"):
        run_schedule(plan(), {"input": 1}, output, boundary.open, resume=True)
    assert len(boundary.calls) == 8


def test_preflight_failure_retains_identity_and_can_resume(tmp_path):
    boundary = Boundary()
    output = tmp_path / "study"

    @contextmanager
    def fail_after(arm):
        if arm == "after":
            raise RuntimeError("preflight failure")
        with boundary.open(arm) as session:
            yield session

    with pytest.raises(RuntimeError, match="preflight failure"):
        run_schedule(plan(), {"input": 1}, output, fail_after)
    partial = json.loads((output / "study.json").read_text())
    assert partial["status"] == "failed"
    assert set(partial["arms"]) == {"before"}
    assert not boundary.calls
    boundary.version = 2
    with pytest.raises(ValueError, match="observed before"):
        run_schedule(plan(), {"input": 1}, output, boundary.open, resume=True)
    boundary.version = 1
    run_schedule(plan(), {"input": 1}, output, boundary.open, resume=True)
    assert len(boundary.calls) == 8


def test_failed_common_evaluation_repeats_both_native_arms(tmp_path):
    boundary = Boundary()
    output = tmp_path / "study"

    def failed_evaluation(*args):
        raise RuntimeError("evaluation failure")

    with pytest.raises(RuntimeError, match="evaluation failure"):
        run_schedule(plan(), {}, output, boundary.open, finish_block=failed_evaluation)
    assert not list((output / "blocks").iterdir())
    assert len(boundary.calls) == 2
    run_schedule(plan(), {}, output, boundary.open, resume=True)
    assert len(boundary.calls) == 10


def test_different_attested_inputs_refuse_collection_before_either_arm_runs(tmp_path):
    boundary = Boundary()

    @contextmanager
    def differing_inputs(arm):
        with boundary.open(arm) as session:
            session.identity = {**session.identity, "comparison_identity": arm}
            yield session

    with pytest.raises(ValueError, match="scientific inputs"):
        run_schedule(plan(), {}, tmp_path / "study", differing_inputs)
    assert not boundary.calls


def test_managed_captures_follow_all_clean_pairs_and_resume_independently(tmp_path):
    from safetensors.torch import save_file
    from causalab.measurement.collection import file_hash, write_record

    boundary = Boundary()
    events = []
    interrupt = [True]

    @contextmanager
    def open_session(arm):
        with boundary.open(arm) as session:
            sample = session.sample

            def clean(*args, **kwargs):
                assert "profile" not in kwargs
                events.append("clean")
                receipt = sample(*args, **kwargs)
                record = json.loads(receipt.read_text())
                for row in record["samples"]:
                    row["unobserved_observations"] = row["observations"]
                    row["resident_unobserved_observations"] = row["observations"]
                write_record(receipt, record)
                return receipt

            def capture(case, seed, repeat, directory):
                events.append("capture")
                if arm == "after" and interrupt[0]:
                    raise KeyboardInterrupt("capture interrupted")
                directory.mkdir()
                observations = directory / "observations.safetensors"
                save_file({"site": torch.tensor([float(seed + 1)])}, str(observations))
                captures = []
                for mode in ("cold", "warm"):
                    native = directory / f"{mode}.nsys-rep"
                    native.write_bytes(b"opaque native data")
                    captures.append(
                        {
                            "id": f"nsys:{mode}",
                            "pair_id": "nsys",
                            "backend": "nsys",
                            "mode": mode,
                            "seed": seed,
                            "repeat": repeat,
                            "status": "completed",
                            "artifact_format": "nsys-rep",
                            "artifacts": [
                                {"file": native.name, "sha256": file_hash(native)}
                            ],
                            "observations": {
                                "file": observations.name,
                                "sha256": file_hash(observations),
                            },
                        }
                    )
                receipt = directory / "measurement.json"
                write_record(
                    receipt,
                    {
                        "captures": captures,
                        "capture_plan": {
                            "backends": ["nsys"],
                            "modes": ["cold", "warm"],
                        },
                    },
                )
                return receipt

            session.sample, session.capture = clean, capture
            yield session

    study = plan()
    study["profile"] = {"cases": ["apply"], "backends": {"nsys": {}}}
    study["cases"]["apply"]["cold_process"] = True
    output = tmp_path / "study"
    with pytest.raises(KeyboardInterrupt, match="capture interrupted"):
        run_schedule(study, {"input": 1}, output, open_session)
    assert events == ["clean"] * 8 + ["capture"] * 2
    assert len(boundary.calls) == 8
    interrupt[0] = False
    result = run_schedule(study, {"input": 1}, output, open_session, resume=True)
    assert len(boundary.calls) == 8  # No timing repeat after profiling interruption.
    assert events[-1] == "capture" and len(events) == 11
    report = compare(
        result["apply"]["before"], result["apply"]["after"], bootstrap_draws=100
    )
    assert report["capture_pairs"]["before"][0]["status"] == "completed"
    assert all(
        c["observation_check"]["exactly_equal"] for c in report["captures"]["after"]
    )
    run_schedule(study, {"input": 1}, output, open_session, resume=True)
    assert len(events) == 11
    native = output / "profiles/apply/before/cold.nsys-rep"
    native.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="completed profiler capture changed"):
        run_schedule(study, {"input": 1}, output, open_session, resume=True)
    assert len(boundary.calls) == 8
