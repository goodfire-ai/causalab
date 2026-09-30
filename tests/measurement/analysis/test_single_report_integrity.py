"""Malformed single-source evidence cannot downgrade the authored contract."""

import json
import re
from urllib.parse import unquote

import pytest
import torch

from causalab.measurement.analysis.receipts import load_measurement
from causalab.measurement.analysis.reports import write_reports
from causalab.measurement.collection import file_hash, write_record
from tests.measurement.analysis.test_compare import receipt
from tests.measurement.analysis.test_single_receipts import timing_receipt
from tests.measurement.analysis.test_single_reports import single_fixture

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "contract",
    [
        {"observations": {"score": {"kind": "tensor"}}},
        {
            "observation_policy": "required",
            "observations": {"score": {"kind": "tensor"}},
        },
    ],
)
def test_requested_observations_cannot_be_downgraded_by_receipt(tmp_path, contract):
    path = timing_receipt(tmp_path / "source")
    with pytest.raises(ValueError, match="observation policy"):
        write_reports(
            {"workflow": {"source": path}},
            {"mode": "single", **contract},
            tmp_path / "reports",
        )


@pytest.mark.parametrize(
    "value",
    [torch.tensor([float("nan")]), torch.tensor([float("inf")]), torch.tensor([])],
)
def test_single_numerical_receipts_refuse_invalid_tensor_values(tmp_path, value):
    from causalab.measurement.analysis.receipts import ReceiptValidationError

    path = receipt(tmp_path / "source", [[{"score": value}]])
    with pytest.raises(ReceiptValidationError, match="nonfinite|empty") as raised:
        load_measurement(path, require_observations=False)
    assert raised.value.path == path
    assert raised.value.reason


@pytest.mark.parametrize(
    "value",
    [
        None,
        [],
        ["output.json"],
        {"output.json": None},
        {"output.json": 3},
        {"output.json": "bad"},
    ],
)
@pytest.mark.parametrize("location", ["sample", "capture"])
def test_output_manifests_require_a_mapping_of_files_to_hashes(
    tmp_path, value, location
):
    from causalab.measurement.analysis.receipts import ReceiptValidationError

    path = single_fixture(tmp_path / "source")
    record = json.loads(path.read_text())
    record["samples" if location == "sample" else "captures"][0]["output_files"] = value
    write_record(path, record)
    with pytest.raises(ReceiptValidationError, match="output_files"):
        load_measurement(path, require_observations=False)


@pytest.mark.parametrize(
    "mutation",
    ["seed", "repeat", "status", "empty_artifact", "missing_file", "observation_specs"],
)
def test_legacy_trace_identity_and_evidence_are_validated(tmp_path, mutation):
    from causalab.measurement.analysis.receipts import ReceiptValidationError

    path = timing_receipt(tmp_path / "source")
    trace = path.parent / "trace.json"
    trace.write_text("{}")
    record = json.loads(path.read_text())
    record["trace"] = {
        "status": "completed",
        "seed": 0,
        "repeat": 0,
        "observation_status": "not_requested",
        "file": trace.name,
        "sha256": file_hash(trace),
    }
    if mutation in ("seed", "repeat"):
        record["trace"][mutation] = 999
    elif mutation == "status":
        record["trace"]["status"] = "made_up"
    elif mutation == "empty_artifact":
        trace.write_text("")
        record["trace"]["sha256"] = file_hash(trace)
    elif mutation == "missing_file":
        del record["trace"]["file"]
    else:
        record["trace"]["observation_specs"] = {"score": {"kind": "tensor"}}
    write_record(path, record)
    with pytest.raises(ReceiptValidationError, match="trace"):
        load_measurement(path, require_observations=False)


def test_reserved_case_filenames_and_artifact_links_do_not_collide(tmp_path):
    collections = {}
    for case in ("index", "study", "cases", "case-index", "resident"):
        path = single_fixture(tmp_path / case)
        record = json.loads(path.read_text())
        record["case"] = case
        write_record(path, record)
        collections[case] = {"source": path}
    output = tmp_path / "reports"
    report = write_reports(collections, {"mode": "single"}, output)
    assert report["cases"]["resident"]["files"]["report"] == "resident.html"
    for case, value in report["cases"].items():
        assert (
            json.loads((output / value["files"]["summary"]).read_text())["case"] == case
        )
        page = output / value["files"]["report"]
        assert "<h2>Clean timings</h2>" in page.read_text()
    for page in output.rglob("*.html"):
        for link in re.findall(r'href="([^"]+)"', page.read_text()):
            assert (page.parent / unquote(link)).is_file()


def test_cold_report_retains_resident_memory_with_a_separate_scope(tmp_path):
    path = single_fixture(tmp_path / "source")
    record = json.loads(path.read_text())
    record["scope"] = "cold-process workflow worker wall"
    sample = record["samples"][0]
    sample["resident_peak_memory"] = sample["peak_memory"]
    sample["peak_memory"] = None
    write_record(path, record)
    report = write_reports(
        {"workflow": {"source": path}}, {"mode": "single"}, tmp_path / "reports"
    )["cases"]["workflow"]["measurement"]
    assert report["memory"]["status"] == "unavailable"
    resident = report["memory"]["resident_reference"]
    assert resident["status"] == "available"
    assert resident["scope"] != record["scope"]
    assert "resident" in resident["scope"]
    assert resident["units"] == "bytes"
    assert resident["samples"][0]["peak_memory"]["allocated_bytes"] == 1024
    assert "1024" in (tmp_path / "reports/workflow.html").read_text()
