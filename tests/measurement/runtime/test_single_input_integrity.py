"""Omitting numerical observations never disables single-source input attestation."""

from copy import deepcopy
import json
import shutil

import pytest

from causalab.measurement.plan import execution_plan
from causalab.measurement.census import CensusError, collect_pins, strip_pins
from causalab.io.env import FileDatasets, ResolutionEnv, split_dataset_ref
from causalab.tables import SPLIT_COLUMN
from causalab.workflow.document import load_workflow, parse_workflow
from tests.measurement.runtime.test_worker_pin_lifecycle import _worker

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("policy", ["required", "not_requested"])
@pytest.mark.parametrize("input_kind", ["dataset", "declared_file"])
def test_single_worker_prepare_and_resume_reject_changed_input_bytes(
    tmp_path, policy, input_kind
):
    worker = _worker(tmp_path, arm="source")
    authored = json.loads(worker.path.read_text())["measurement"]
    authored["mode"] = "single"
    authored["source"] = authored.pop("arms")["before"]
    authored["cases"] = {"workflow": {"kind": "workflow", "cold_process": False}}
    if policy == "not_requested":
        authored.pop("observations")
    definition = {key: value for key, value in worker.raw.items() if key != "pins"}
    measurement = parse_workflow({**definition, "measurement": authored}).measurement
    worker.plan = execution_plan(measurement)
    worker.config["plan"] = worker.plan
    assert worker.plan["source_pin_anchor"] == "source"
    assert worker.plan["observation_policy"] == policy

    # Shadow the shipped table with identical private bytes before freezing it.
    dataset_ref = next(iter(worker.raw["pins"]["datasets"]))
    dataset_base, dataset_split = split_dataset_ref(dataset_ref)
    dataset = tmp_path / "data" / f"{dataset_base}.json"
    dataset.parent.mkdir(parents=True)
    shutil.copyfile(worker.env.datasets._file(dataset_base), dataset)
    worker.env = ResolutionEnv(
        datasets=FileDatasets(root=tmp_path / "data"),
        artifacts=worker.env.artifacts,
        model_info=worker.env.model_info,
    )
    declared = tmp_path / "declared-input.json"
    declared.write_text('{"value": 1}')
    worker.raw["steps"]["report"]["inputs"] = {"declared": {"path": declared.name}}
    loaded = load_workflow(
        strip_pins(worker.raw)[0], worker.env, workflow_dir=worker.path.parent
    )
    worker.raw["pins"] = collect_pins(loaded, worker.env.datasets)
    worker.load(7)
    frozen = deepcopy(worker.pin_contract.pins)
    assert frozen["datasets"][dataset_ref]
    assert frozen["files"][declared.name]

    if input_kind == "dataset":
        rows = json.loads(dataset.read_text())
        selected = next(
            row
            for row in rows
            if dataset_split is None or row[SPLIT_COLUMN] == dataset_split
        )
        selected["input"] += " changed input"
        dataset.write_text(json.dumps(rows))
        expected_error = r"pins\.datasets"
    else:
        declared.write_text('{"value": 2}')
        expected_error = r"pins\.files"

    # The real prepare reloads and checks pins before yielding executable work.
    with pytest.raises(CensusError, match=expected_error):
        with worker.prepare("workflow", 7, tmp_path / "attempt"):
            pytest.fail("changed input reached workflow execution")
    assert worker.pin_contract.pins == frozen
    assert not (tmp_path / "attempt").exists()

    # A resumed process has no resident contract; its authored/frozen pins still
    # refuse the same changed dataset or declared input before any model work.
    worker.pin_contract = None
    worker.config["resolved_pins"] = frozen
    with pytest.raises(CensusError, match=expected_error):
        worker.load(7)
    assert worker.pin_contract is None
