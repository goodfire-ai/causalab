"""Remote workflow comparisons preserve each document's authored contract."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile

from hypothesis import given, settings, strategies as st
import pytest

from causalab.measurement.census import CensusError
from causalab.measurement.collection import file_hash, write_record
from causalab.measurement.deployment import remote
from causalab.measurement.deployment.source_pins import SourcePinError
from tests.measurement.deployment.test_remote_pins import _census, _study
from tests.measurement.deployment.test_source_pins import _pinned, _revisions

pytestmark = pytest.mark.unit


def _workflow_study(root: Path) -> Path:
    baseline, _ = _study(root)
    candidate_dir = root / "candidate"
    candidate_dir.mkdir()
    candidate, _ = _study(candidate_dir, revision="candidate-model")
    candidate_raw = json.loads(candidate.read_text())
    candidate_raw.pop("measurement")
    write_record(candidate, candidate_raw)
    raw = json.loads(baseline.read_text())
    raw["measurement"]["comparison"] = "workflow"
    raw["measurement"]["arms"]["after"]["workflow"] = "candidate/workflow.json"
    write_record(baseline, raw)
    return baseline


@pytest.mark.parametrize("eager", ["absent", "baseline", "candidate"])
def test_freeze_preserves_distinct_documents_and_portable_pins(
    tmp_path: Path, eager: str
) -> None:
    document = _workflow_study(tmp_path)
    original = document.read_bytes()
    candidate_original = (tmp_path / "candidate/workflow.json").read_bytes()
    raw = json.loads(original)
    if eager != "absent":
        raw["measurement"]["arms"]["eager"] = {"revision": "HEAD"}
        if eager == "candidate":
            raw["measurement"]["arms"]["eager"]["workflow"] = "candidate/workflow.json"
        write_record(document, raw)
        original = document.read_bytes()
    frozen = remote._freeze_study(document, tmp_path / "frozen")
    write_record(tmp_path / "frozen/workflow.json", frozen)
    moved = tmp_path / "relocated"
    shutil.move(tmp_path / "frozen", moved)
    # The existing fixture supplies a real resolver without importing arm code.
    environment = tmp_path / "environment"
    environment.mkdir()
    _, env = _study(environment)
    assert _census(moved / "workflow.json", env) == frozen["pins"]
    for arm in ("after", "eager"):
        if arm not in frozen["measurement"]["arms"]:
            continue
        reference = frozen["measurement"]["arms"][arm].get("workflow")
        if reference is None:
            assert arm == "eager" and eager == "baseline"
            continue
        path = moved / reference
        candidate = json.loads(path.read_text())
        assert "measurement" not in candidate
        assert _census(path, env) == candidate["pins"]
        assert candidate["pins"]["documents"] == {
            step["document"]: file_hash(path.parent / step["document"])
            for step in candidate["steps"].values()
        }
        specification = path.parent / candidate["steps"]["locate"]["document"]
        assert json.loads(specification.read_text())["model"]["revision"] == (
            "candidate-model"
        )
        assert candidate["pins"]["documents"] != frozen["pins"]["documents"]
    assert document.read_bytes() == original
    assert (tmp_path / "candidate/workflow.json").read_bytes() == candidate_original


def test_freeze_refuses_stale_candidate_document_pins(tmp_path: Path) -> None:
    document = _workflow_study(tmp_path)
    with (tmp_path / "candidate/methods/locate.json").open("a") as stream:
        stream.write("\n")
    with pytest.raises(CensusError, match="pins.documents"):
        remote._freeze_study(document, tmp_path / "frozen")


@settings(max_examples=6, deadline=None)
@given(revision=st.text(alphabet="abcdef0123456789", min_size=1, max_size=12))
def test_freezing_keeps_candidate_overrides_isolated(revision: str) -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        document = _workflow_study(root)
        candidate = root / "candidate/workflow.json"
        raw = json.loads(candidate.read_text())
        for step in raw["steps"].values():
            step["set"]["model.revision"] = revision
        write_record(candidate, raw)
        destination = root / "frozen"
        frozen = remote._freeze_study(document, destination)
        candidate_path = (
            destination / frozen["measurement"]["arms"]["after"]["workflow"]
        )
        selected = json.loads(candidate_path.read_text())
        for name, step in selected["steps"].items():
            candidate_spec = candidate_path.parent / step["document"]
            baseline_spec = destination / frozen["steps"][name]["document"]
            assert json.loads(candidate_spec.read_text())["model"]["revision"] == (
                revision
            )
            assert json.loads(baseline_spec.read_text())["model"]["revision"] == "main"


def test_workflow_named_step_does_not_overwrite_packaged_manifest(
    tmp_path: Path,
) -> None:
    document = _workflow_study(tmp_path)
    for path in (document, tmp_path / "candidate/workflow.json"):
        raw = json.loads(path.read_text())
        raw["steps"]["workflow"] = raw["steps"].pop("locate")
        if "measurement" in raw:
            raw["measurement"]["cases"]["locate"]["step"] = "workflow"
            raw["measurement"]["observations"]["score"]["step"] = "workflow"
        write_record(path, raw)
    destination = tmp_path / "frozen"
    frozen = remote._freeze_study(document, destination)
    write_record(destination / "workflow.json", frozen)
    candidate_path = destination / frozen["measurement"]["arms"]["after"]["workflow"]
    for path in (destination / "workflow.json", candidate_path):
        workflow = json.loads(path.read_text())
        specification = path.parent / workflow["steps"]["workflow"]["document"]
        assert specification != path
        assert "model" in json.loads(specification.read_text())
        assert "steps" in workflow


@pytest.mark.parametrize("comparison", ["code", "workflow"])
@pytest.mark.parametrize("step_name", ["bindings", "workflow"])
def test_deployment_metadata_preserves_reserved_step_documents(
    tmp_path: Path, comparison: str, step_name: str
) -> None:
    document = (
        _workflow_study(tmp_path) if comparison == "workflow" else _study(tmp_path)[0]
    )
    authored = [document]
    if comparison == "workflow":
        authored.append(tmp_path / "candidate/workflow.json")
    for path in authored:
        raw = json.loads(path.read_text())
        raw["steps"][step_name] = raw["steps"].pop("locate")
        if "measurement" in raw:
            raw["measurement"]["cases"]["locate"]["step"] = step_name
            raw["measurement"]["observations"]["score"]["step"] = step_name
        write_record(path, raw)

    destination = tmp_path / "frozen"
    frozen = remote._freeze_study(document, destination)
    write_record(destination / "workflow.json", frozen)
    write_record(destination / "bindings.json", {"device": "cpu", "arms": {}})
    environment = tmp_path / "environment"
    environment.mkdir()
    _, env = _study(environment)
    packaged = [destination / "workflow.json"]
    if comparison == "workflow":
        packaged.append(
            destination / frozen["measurement"]["arms"]["after"]["workflow"]
        )
    for path in packaged:
        raw = json.loads(path.read_text())
        assert raw["pins"]["documents"] == {
            step["document"]: file_hash(path.parent / step["document"])
            for step in raw["steps"].values()
        }
        assert _census(path, env) == raw["pins"]


def test_freeze_refuses_measurement_block_in_selected_workflow(tmp_path: Path) -> None:
    document = _workflow_study(tmp_path)
    candidate = tmp_path / "candidate/workflow.json"
    raw = json.loads(candidate.read_text())
    pins = raw.pop("pins")
    raw["measurement"] = json.loads(document.read_text())["measurement"]
    raw["pins"] = pins
    write_record(candidate, raw)
    with pytest.raises(ValueError, match="must not contain a measurement block"):
        remote._freeze_study(document, tmp_path / "frozen")


@pytest.mark.parametrize("invalid", ["symlink", "selector"])
def test_freeze_checks_selected_workflows_before_packaging(
    tmp_path: Path, invalid: str
) -> None:
    document = _workflow_study(tmp_path)
    candidate = tmp_path / "candidate/workflow.json"
    if invalid == "symlink":
        outside = tmp_path.parent / f"{tmp_path.name}-outside.json"
        outside.write_bytes(candidate.read_bytes())
        candidate.unlink()
        candidate.symlink_to(outside)
        message = "workflow path must stay within the study directory"
    else:
        raw = json.loads(candidate.read_text())
        raw["steps"].pop("locate")
        write_record(candidate, raw)
        message = "unknown workflow step 'locate'"
    with pytest.raises(ValueError, match=message):
        remote._freeze_study(document, tmp_path / "frozen")
    assert not (tmp_path / "frozen").exists()


@pytest.mark.parametrize("failure", ["commit", "candidate_source", "none"])
def test_launch_preflights_each_workflow_before_ssh(
    tmp_path: Path, monkeypatch, failure: str
) -> None:
    document = _workflow_study(tmp_path)
    repository = tmp_path / "repo"
    before, after = _revisions(repository)
    raw = json.loads(document.read_text())
    raw["measurement"]["arms"]["before"]["revision"] = before
    raw["measurement"]["arms"]["after"]["revision"] = (
        after if failure == "commit" else before
    )
    raw["pins"]["code"] = {}
    write_record(document, raw)
    candidate = tmp_path / "candidate/workflow.json"
    raw = json.loads(candidate.read_text())
    raw["pins"]["code"] = _pinned("code")["pins"]["code"]
    if failure == "candidate_source":
        raw["pins"]["code"]["causalab.example"] = "0" * 64
    write_record(candidate, raw)
    bindings = tmp_path / "bindings.json"
    write_record(
        bindings,
        {
            "arms": {
                arm: {"repository": str(repository), "python": "/remote/python"}
                for arm in ("before", "after")
            },
            "device": "cpu",
            "data_root": "/data",
            "artifacts_root": "/artifacts",
        },
    )
    calls = []

    class ReachedSSH(RuntimeError):
        pass

    def stop_at_ssh(*args, **kwargs):
        calls.append(args)
        raise ReachedSSH("preflight complete")

    monkeypatch.setattr(remote.SSH, "call", stop_at_ssh)
    args = remote.parser().parse_args(
        [
            "launch",
            str(document),
            "--bindings",
            str(bindings),
            "--host",
            "compute",
            "--receipt",
            str(tmp_path / "job.json"),
        ]
    )
    if failure == "commit":
        with pytest.raises(ValueError, match="same.*commit"):
            remote.launch(args)
    elif failure == "candidate_source":
        with pytest.raises(SourcePinError, match="after.*pins.code.causalab.example"):
            remote.launch(args)
    else:
        with pytest.raises(ReachedSSH, match="preflight complete"):
            remote.launch(args)
    assert len(calls) == (1 if failure == "none" else 0)
    assert not args.receipt.exists()
