"""Cold receipts keep resident reference artifacts outside the timed output set."""

from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from urllib.parse import unquote

from hypothesis import given, settings, strategies as st
import pytest
import torch

from causalab.measurement.analysis.receipts import (
    ReceiptValidationError,
    load_measurement,
)
from causalab.measurement.analysis.single import render_single, summarize_single
from causalab.measurement.collection import file_hash, write_record
from causalab.measurement.study import controller, scheduler
from tests.measurement.analysis.test_compare import receipt as numerical_receipt
from tests.measurement.analysis.test_single_receipts import timing_receipt


def cold_receipt(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    numerical: bool,
    payloads: list[bytes] | None = None,
) -> Path:
    path = numerical_receipt(root, [[[1.0]]]) if numerical else timing_receipt(root)
    record = json.loads(path.read_text())
    record.update(
        mode="single",
        observation_policy="required" if numerical else "not_requested",
        case="workflow",
        context={"cache_policy": {"kind": "resident"}},
        warmups=[],
        trace={"status": "not_requested"},
    )
    payloads = [b"resident"] if payloads is None else payloads
    resident = root / "resident/outputs"
    resident.mkdir(parents=True)
    for index, payload in enumerate(payloads):
        (resident / f"artifact-{index}.json").write_bytes(payload)
    record["samples"][0].update(
        peak_memory=None,
        workflow_outputs="resident/outputs",
        output_files={
            file.relative_to(root).as_posix(): file_hash(file)
            for file in resident.iterdir()
        },
    )
    write_record(path, record)
    workflow = root / "workflow.json"
    workflow.write_text(json.dumps({"output_dir": "outputs"}))
    session = object.__new__(controller.ProcessSession)
    session.identity = {"models": {}}
    session.python, session.logs = sys.executable, root
    session.config = {
        "plan": {
            "observation_policy": record["observation_policy"],
            "observations": {},
        },
        "workflow": str(workflow),
        "controller_root": str(root),
        "package_root": str(root),
    }

    def execute(command: list[str], **kwargs: object) -> SimpleNamespace:
        config = json.loads(Path(command[-1]).read_text())
        output = Path(config["cold"]["directory"]) / "outputs"
        output.mkdir(parents=True)
        for index, payload in enumerate(payloads):
            (output / f"artifact-{index}.json").write_bytes(b"cold:" + payload)
        return SimpleNamespace(
            stdout=json.dumps(
                {
                    "identity": session.identity,
                    "status": "completed",
                    "cache_policy": {"kind": "cold"},
                }
            )
        )

    monkeypatch.setattr(controller.subprocess, "run", execute)
    if numerical:
        from causalab.measurement.runtime import observations

        monkeypatch.setattr(
            observations, "observations", lambda *args: {"value": torch.tensor([1.0])}
        )
    session._cold("workflow", 0, root, path)
    return path


@pytest.mark.unit
@pytest.mark.parametrize("numerical", [False, True])
def test_cold_outputs_remain_scoped_after_aggregation_and_fetch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, numerical: bool
) -> None:
    original = tmp_path / "original"
    cell = original / "blocks/0"
    cell.mkdir(parents=True)
    path = cold_receipt(cell / "source", monkeypatch, numerical=numerical)
    sample = json.loads(path.read_text())["samples"][0]
    assert set(sample["output_files"]) == {"cold/outputs/artifact-0.json"}
    assert set(sample["resident_output_files"]) == {"resident/outputs/artifact-0.json"}
    plan = {
        "mode": "single",
        "observation_policy": "required" if numerical else "not_requested",
        "cases": {"workflow": {}},
        "arms": {"source": {}},
        "seeds": [0],
        "repeats": 1,
        "warmups": 0,
    }
    grouped = scheduler._aggregate(
        original,
        plan,
        [
            {
                "cell": {"case": "workflow", "seed": 0, "repeat": 0, "index": 0},
                "directory": "blocks/0",
            }
        ],
    )
    aggregate = grouped["workflow"]["source"].relative_to(original)
    fetched = tmp_path / "fetched"
    shutil.copytree(original, fetched)
    shutil.rmtree(original)
    path = fetched / aggregate
    _, samples = load_measurement(path, require_observations=numerical)
    sample = samples[0, 0]
    for field in ("output_files", "resident_output_files"):
        assert len(sample[field]) == 1
        assert all(name.startswith("blocks/0/source/") for name in sample[field])
    report_root = fetched / "reports"
    report_root.mkdir()
    report = summarize_single(path, report_root)
    assert {row["file"] for row in report["outputs"]} == set(sample["output_files"])
    assert {row["file"] for row in report["resident_outputs"]} == set(
        sample["resident_output_files"]
    )
    assert all(row["scope"] == report["scope"] for row in report["outputs"])
    assert all(
        "excluded from cold timing" in row["scope"]
        for row in report["resident_outputs"]
    )
    page = render_single(report)
    primary, resident = page.split("<h3>Resident reference outputs</h3>")
    assert "resident/outputs/artifact-0.json" not in primary
    assert "excluded from cold timing" in resident
    assert "resident/outputs/artifact-0.json" in resident
    for link in re.findall(r'href="([^"]+)"', page):
        assert (report_root / unquote(link)).is_file()


@pytest.mark.unit
@pytest.mark.parametrize("numerical", [False, True])
@pytest.mark.parametrize("field", ["output_files", "resident_output_files"])
def test_both_scoped_output_families_reject_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, numerical: bool, field: str
) -> None:
    path = cold_receipt(tmp_path / "sample", monkeypatch, numerical=numerical)
    _, samples = load_measurement(path, require_observations=numerical)
    name = next(iter(samples[0, 0][field]))
    (path.parent / name).write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_measurement(path, require_observations=numerical)
    with pytest.raises(ValueError, match="hash mismatch"):
        summarize_single(path, tmp_path)


@pytest.mark.unit
@pytest.mark.parametrize("numerical", [False, True])
@pytest.mark.parametrize("invalid", [None, [], {"artifact": "invalid"}])
def test_resident_output_manifests_are_validated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, numerical: bool, invalid: object
) -> None:
    path = cold_receipt(tmp_path / "sample", monkeypatch, numerical=numerical)
    record = json.loads(path.read_text())
    record["samples"][0]["resident_output_files"] = invalid
    write_record(path, record)
    with pytest.raises(ReceiptValidationError, match="output_files"):
        load_measurement(path, require_observations=numerical)


@pytest.mark.property
@settings(max_examples=20, deadline=None)
@given(
    payloads=st.lists(st.binary(max_size=50), min_size=1, max_size=5),
    numerical=st.booleans(),
)
def test_output_partition_preserves_every_artifact_hash(
    payloads: list[bytes], numerical: bool
) -> None:
    with TemporaryDirectory() as directory, pytest.MonkeyPatch.context() as patch:
        path = cold_receipt(
            Path(directory) / "sample", patch, numerical=numerical, payloads=payloads
        )
        _, samples = load_measurement(path, require_observations=numerical)
        sample = samples[0, 0]
        cold, resident = sample["output_files"], sample["resident_output_files"]
        assert set(cold).isdisjoint(resident)
        assert len(cold) == len(resident) == len(payloads)
        for index, payload in enumerate(payloads):
            for prefix, manifest, contents in (
                ("cold", cold, b"cold:" + payload),
                ("resident", resident, payload),
            ):
                name = f"{prefix}/outputs/artifact-{index}.json"
                assert (path.parent / name).read_bytes() == contents
                assert manifest[name] == file_hash(path.parent / name)
