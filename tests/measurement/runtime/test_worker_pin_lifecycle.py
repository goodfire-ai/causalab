"""Every worker load holds its selected arm's frozen source/input census."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from causalab.measurement.census import CensusError, collect_pins
from causalab.measurement.runtime.worker import Worker
from tests.measurement.deployment.test_remote_pins import _census, _study

pytestmark = pytest.mark.unit
REPO = Path(__file__).resolve().parents[3]


def _worker(tmp_path, *, arm="before"):
    workflow, env = _study(tmp_path)
    worker = Worker.__new__(Worker)
    worker.path = workflow
    worker.raw = json.loads(workflow.read_text())
    worker.raw.pop("measurement")
    worker.raw["steps"]["report"] = {
        "type": "script",
        "script": {"module": "causalab.measurement.analysis.compare"},
        "inputs": {},
        "outputs": {"out": {"file": "out.json", "columns": {"n": "int64"}}},
    }
    worker.raw.pop("pins")
    worker.raw["pins"] = _census(worker.raw, env, workflow_dir=workflow.parent)
    worker.env = env
    worker.fitting = set()
    worker.pin_contract = None
    worker.config = {"arm": arm, "package_root": str(REPO)}
    return worker


def test_initial_load_freezes_full_census_without_changing_authored_document(tmp_path):
    worker = _worker(tmp_path)
    original = worker.path.read_bytes()
    census = collect_pins(worker.load(7), worker.env.datasets)
    assert worker.pin_contract.pins == census
    assert worker.raw["pins"] == census
    assert worker.path.read_bytes() == original
    assert collect_pins(worker.load(11), worker.env.datasets) == census


def test_candidate_resolves_own_source_pins_but_baseline_holds_authored_hash(tmp_path):
    worker = _worker(tmp_path, arm="after")
    authored = worker.raw["pins"]
    assert authored["scripts"]
    for name in authored["scripts"]:
        authored["scripts"][name] = "0" * 64
    original = deepcopy(worker.raw)
    census = collect_pins(worker.load(7), worker.env.datasets)
    assert worker.raw == original
    assert worker.pin_contract.source["scripts"] == census["scripts"]
    assert set(census["scripts"].values()) != {"0" * 64}
    worker.pin_contract = None
    worker.config["arm"] = "before"
    worker.raw["pins"] = authored
    with pytest.raises(CensusError, match="pins\\."):
        worker.load(7)


def test_expected_census_survives_new_process_setup(tmp_path):
    worker = _worker(tmp_path)
    worker.load(7)
    expected = deepcopy(worker.pin_contract.pins)
    key = next(iter(expected["scripts"]))
    expected["scripts"][key] = "0" * 64
    worker.pin_contract = None
    worker.config["resolved_pins"] = expected
    with pytest.raises(CensusError, match="pins\\."):
        worker.load(7)


def test_ancestor_subset_checks_used_pins_without_requiring_omitted_steps(tmp_path):
    worker = _worker(tmp_path)
    worker.load(7)
    raw = deepcopy(worker.raw)
    raw["steps"].pop("replica")
    loaded = worker.load_subset(raw, worker.env)
    assert set(loaded.document.steps) == {"locate", "report"}
    census = collect_pins(loaded, worker.env.datasets)
    assert census["scripts"] == worker.pin_contract.source["scripts"]
    # Mutating the file after the full load must not slip through a subset load.
    protocol = worker.path.parent / raw["steps"]["locate"]["document"]
    protocol.write_text(protocol.read_text() + "\n")
    with pytest.raises(CensusError, match="pins\\."):
        worker.load_subset(raw, worker.env)


@pytest.mark.parametrize("change", ["workflow", "protocol"])
def test_worker_refuses_benchmark_replacement_before_loading_models(
    tmp_path, monkeypatch, change
):
    from causalab.measurement.runtime import worker as runtime
    from causalab.measurement.runtime.benchmark import BenchmarkIdentityError
    from causalab.protocol.pipeline import read_document
    from causalab.measurement.study.scheduler import digest

    fixture = _worker(tmp_path)
    fixture.path.write_text(json.dumps(fixture.raw))
    authored = {
        name: dict(
            read_document(
                fixture.path.parent / step["document"],
                fixture.path.parent,
                step.get("set", {}),
            ).raw
        )
        for name, step in fixture.raw["steps"].items()
        if step["type"] == "intervention_protocol"
    }
    expected = digest(
        {"workflow": fixture.raw, "protocols": authored, "observations": {}}
    )
    if change == "workflow":
        changed = deepcopy(fixture.raw)
        changed["steps"]["report"]["inputs"]["added"] = "new-value"
        fixture.path.write_text(json.dumps(changed))
    else:
        path = fixture.path.parent / fixture.raw["steps"]["locate"]["document"]
        content = json.loads(path.read_text())
        content["description"] = "Changed benchmark document"
        path.write_text(json.dumps(content))
    monkeypatch.setattr(
        runtime,
        "execution_identity",
        lambda device: {"implementation": {"location": str(REPO / "causalab")}},
    )
    monkeypatch.setattr(
        runtime, "attest_installation", lambda config, package: config["source_commit"]
    )
    with pytest.raises(BenchmarkIdentityError, match="benchmark definition changed"):
        Worker(
            {
                "device": "cpu",
                "package_root": str(REPO),
                "source_commit": "a" * 40,
                "arm": "before",
                "workflow": str(fixture.path),
                "data_root": str(tmp_path),
                "artifacts_root": str(tmp_path),
                "plan": {"observations": {}},
                "input_identity": expected,
            }
        )
