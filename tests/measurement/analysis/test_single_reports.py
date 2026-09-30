"""Single-source reports preserve absolute measurements and portable evidence links."""

import json
import re
import shutil
from urllib.parse import unquote

import pytest

from causalab.measurement.analysis.receipts import load_measurement
from causalab.measurement.analysis.reports import write_reports
from causalab.measurement.collection import file_hash, write_record
from tests.measurement.analysis.test_single_receipts import timing_receipt
from tests.measurement.analysis.test_compare import receipt

pytestmark = pytest.mark.unit


def single_fixture(root):
    path = timing_receipt(root)
    record = json.loads(path.read_text())
    record["scope"] = "resident workflow wall"
    record["context"] = {
        "cache_policy": {"external_caches": "not flushed"},
        "worker": {
            "source_commit": "abc123",
            "execution": {"cuda_graphs": True},
            "execution_probe": {
                "cases": {"workflow": {"python_dispatch": ["CUDAGraph.replay"]}}
            },
        },
    }
    record["samples"][0]["peak_memory"] = {
        "allocated_bytes": 1024,
        "reserved_bytes": 2048,
    }
    for filename, content in [
        ("output.json", "{}"),
        ("trace.json", '{"traceEvents":[]}'),
        ("capture.log", "done"),
    ]:
        (root / filename).write_text(content)
    record["samples"][0]["output_files"] = {
        "output.json": file_hash(root / "output.json")
    }
    record["capture_plan"] = {"backends": ["torch"], "modes": ["warm"]}
    record["captures"] = [
        {
            "id": "torch:warm",
            "backend": "torch",
            "mode": "warm",
            "seed": 0,
            "repeat": 0,
            "status": "completed",
            "observation_status": "not_requested",
            "artifacts": [
                {"file": "trace.json", "sha256": file_hash(root / "trace.json")}
            ],
            "logs": [
                {"file": "capture.log", "sha256": file_hash(root / "capture.log")}
            ],
            "output_files": {"output.json": file_hash(root / "output.json")},
            "coverage": {"scope": "workflow"},
            "options": {"record_shapes": True},
        }
    ]
    write_record(path, record)
    return path


def test_timing_only_report_and_fetched_regeneration(tmp_path):
    root = tmp_path / "original"
    root.mkdir()
    path = single_fixture(root / "source")
    plan = {"mode": "single", "source": {"revision": "abc123"}}
    study = write_reports({"workflow": {"source": path}}, plan, root / "reports")
    case = study["cases"]["workflow"]["measurement"]
    assert case["timing"]["per_seed"][0]["seconds"]["mean"] == 1.0
    assert case["timing"]["per_seed"][0]["seconds"]["standard_deviation"] is None
    assert case["memory"]["samples"][0]["peak_memory"]["allocated_bytes"] == 1024
    assert case["execution"]["graph_replay_observed"] is True
    assert case["observation_check"]["status"] == "not_requested"
    assert case["captures"][0]["observation_check"]["status"] == "not_requested"
    assert case["collection_status"] == "completed"
    assert case["capture_status"] == "completed"
    assert (
        "measurement/analysis/single.py" in study["analysis"]["dependency_files_sha256"]
    )
    assert (
        "measurement/analysis/receipts.py"
        in study["analysis"]["dependency_files_sha256"]
    )
    fetched = tmp_path / "fetched"
    shutil.copytree(root, fetched)
    shutil.rmtree(root)
    write_reports(
        {"workflow": {"source": fetched / "source/measurement.json"}},
        plan,
        fetched / "reports",
    )
    for page in (fetched / "reports").glob("*.html"):
        text = page.read_text()
        for forbidden in ("before_after", "speedup", "equivalence", "acceptance"):
            assert forbidden not in text.lower()
        for link in re.findall(r'href="([^"]+)"', text):
            assert not link.startswith("file:")
            assert (page.parent / unquote(link)).is_file()
    assert "output.json" in (fetched / "reports/workflow.html").read_text()
    assert "capture.log" in (fetched / "reports/workflow.html").read_text()


def test_failed_capture_preserves_clean_timing_and_cold_memory_unavailable(tmp_path):
    path = single_fixture(tmp_path / "source")
    record = json.loads(path.read_text())
    record["scope"] = "cold-process workflow worker wall"
    record["samples"][0]["resident_peak_memory"] = record["samples"][0].pop(
        "peak_memory"
    )
    record["samples"][0]["peak_memory"] = None
    record["captures"][0].update(status="failed", error="tool exited", artifacts=[])
    write_record(path, record)
    study = write_reports(
        {"workflow": {"source": path}}, {"mode": "single"}, tmp_path / "reports"
    )
    case = study["cases"]["workflow"]["measurement"]
    assert case["timing"]["per_seed"][0]["seconds"]["count"] == 1
    assert case["capture_status"] == "incomplete"
    assert case["memory"]["status"] == "unavailable"
    assert case["memory"]["samples"][0]["peak_memory"] is None
    assert "tool exited" in (tmp_path / "reports/workflow.html").read_text()


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_policy",
        "missing_status",
        "observations",
        "output_hash",
        "artifact_hash",
    ],
)
def test_timing_capture_integrity(tmp_path, mutation):
    path = single_fixture(tmp_path / "source")
    record = json.loads(path.read_text())
    capture = record["captures"][0]
    if mutation == "missing_policy":
        del record["observation_policy"]
    elif mutation == "missing_status":
        del capture["observation_status"]
    elif mutation == "observations":
        capture["observations"] = capture["artifacts"][0]
    elif mutation == "output_hash":
        (path.parent / "output.json").write_text("changed")
    else:
        (path.parent / "trace.json").write_text("changed")
    write_record(path, record)
    with pytest.raises(ValueError):
        load_measurement(path, require_observations=False)


def test_optional_observations_check_capture_against_clean_outputs(tmp_path):
    path = receipt(tmp_path / "source", [[[2.0]]])
    record = json.loads(path.read_text())
    record.update(mode="single", observation_policy="required")
    sample = record["samples"][0]
    sample["unobserved_observations"] = sample["observations"]
    (path.parent / "trace.json").write_text("{}")
    record["captures"] = [
        {
            "id": "torch:warm",
            "backend": "torch",
            "mode": "warm",
            "seed": 0,
            "repeat": 0,
            "status": "completed",
            "observations": sample["observations"],
            "artifacts": [
                {"file": "trace.json", "sha256": file_hash(path.parent / "trace.json")}
            ],
        }
    ]
    write_record(path, record)
    study = write_reports(
        {"test": {"source": path}}, {"mode": "single"}, tmp_path / "reports"
    )
    capture = study["cases"]["test"]["measurement"]["captures"][0]
    assert capture["observation_check"]["exactly_equal"] is True
    assert capture["comparison_reference"] == "clean timing-pass outputs"


def test_single_report_refuses_multiple_targets(tmp_path):
    path = timing_receipt(tmp_path / "source")
    with pytest.raises(ValueError, match="exactly one source"):
        write_reports(
            {"workflow": {"source": path, "extra": path}},
            {"mode": "single"},
            tmp_path / "reports",
        )


def test_capture_outputs_are_verified_independently_of_clean_outputs(tmp_path):
    path = single_fixture(tmp_path / "source")
    record = json.loads(path.read_text())
    record["samples"][0].pop("output_files")
    write_record(path, record)
    (path.parent / "output.json").write_text("changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_measurement(path, require_observations=False)


def test_seed_statistics_and_samples_remain_separate(tmp_path):
    path = timing_receipt(tmp_path / "source")
    record = json.loads(path.read_text())
    record["plan"] = {"seeds": [2, 7], "repeats": 2}
    record["samples"] = [
        {
            "seed": seed,
            "repeat": repeat,
            "seconds": seconds,
            "observation_status": "not_requested",
        }
        for seed, values in [(2, [1.0, 3.0]), (7, [10.0, 14.0])]
        for repeat, seconds in enumerate(values)
    ]
    write_record(path, record)
    study = write_reports(
        {"workflow": {"source": path}}, {"mode": "single"}, tmp_path / "reports"
    )
    rows = study["cases"]["workflow"]["measurement"]["timing"]["per_seed"]
    assert [
        (row["seed"], row["seconds"]["mean"], row["seconds"]["sample_variance"])
        for row in rows
    ] == [(2, 2.0, 2.0), (7, 12.0, 8.0)]
    assert rows[1]["samples"] == [
        {"repeat": 0, "seconds": 10.0},
        {"repeat": 1, "seconds": 14.0},
    ]


def test_single_report_provenance_is_separate_from_resume_validators():
    from causalab.measurement.paths import source_identity
    from causalab.measurement.analysis.reports import analysis_identity

    execution = source_identity()
    analysis = analysis_identity()["dependency_files_sha256"]
    assert "causalab/measurement/analysis/single.py" not in execution
    assert "causalab/measurement/analysis/receipts.py" in execution
    assert "causalab/measurement/analysis/profiles.py" in execution
    assert "measurement/analysis/single.py" in analysis
    assert "measurement/analysis/receipts.py" in analysis
    assert "measurement/analysis/statistics.py" in analysis


def test_analysis_dependency_identity_hashes_every_new_helper(tmp_path, monkeypatch):
    from causalab.measurement.analysis import (
        compare,
        profiles,
        receipts,
        reports,
        single,
        stability,
        statistics,
        summary,
        training,
    )

    modules = (
        compare,
        profiles,
        receipts,
        reports,
        single,
        stability,
        statistics,
        summary,
        training,
    )
    package = tmp_path / "causalab/measurement/analysis"
    package.mkdir(parents=True)
    for module in modules:
        path = package / (module.__name__.rsplit(".", 1)[-1] + ".py")
        path.write_text("original")
        monkeypatch.setattr(module, "__file__", str(path))
    original = reports.analysis_identity()
    assert set(original["dependency_files_sha256"]) == {
        "measurement/analysis/single.py",
        "measurement/analysis/receipts.py",
        "measurement/analysis/statistics.py",
    }
    for name in ("single", "receipts", "statistics"):
        path = package / f"{name}.py"
        path.write_text("changed")
        updated = reports.analysis_identity()
        key = f"measurement/analysis/{name}.py"
        assert (
            updated["dependency_files_sha256"][key]
            != original["dependency_files_sha256"][key]
        )
        assert updated["files_sha256"] == original["files_sha256"]
        path.write_text("original")
