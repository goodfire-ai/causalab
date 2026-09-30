"""Remote I/O is the mocked boundary; extraction and hashes remain real."""

import io
import json
import shutil
import tarfile
from pathlib import Path

import pytest

from causalab.measurement.collection import file_hash, write_record
from causalab.measurement.study.controller import load_installation
from causalab.measurement.deployment.remote import fetch
from causalab.measurement.study.scheduler import manifest

pytestmark = pytest.mark.unit


def test_launch_transports_sources_for_compute_host_builds(tmp_path, monkeypatch):
    from causalab.measurement.deployment import remote

    repo = Path(__file__).resolve().parents[3]
    document = tmp_path / "study.json"
    shutil.copyfile(
        repo / "examples/measurements/qwen/clean.json", tmp_path / "clean.json"
    )
    document.write_text(
        json.dumps(
            {
                "version": "1",
                "output_dir": "research",
                "steps": {
                    "clean": {"type": "intervention_protocol", "document": "clean.json"}
                },
                "measurement": {
                    "version": 1,
                    "arms": {arm: {"revision": "HEAD"} for arm in ("before", "after")},
                    "cases": {"clean": {"kind": "operation", "step": "clean"}},
                    "seeds": [7],
                    "repeats": 1,
                    "observations": {
                        "logits": {
                            "step": "clean",
                            "file": "logits.safetensors",
                            "kind": "tensor",
                        }
                    },
                },
            }
        )
    )
    bindings = tmp_path / "bindings.json"
    bindings.write_text(
        json.dumps(
            {
                "arms": {
                    arm: {"repository": str(repo), "python": "/remote/venv/bin/python"}
                    for arm in ("before", "after")
                },
                "device": "cuda:0",
                "data_root": "/remote/data",
                "artifacts_root": "/remote/artifacts",
            }
        )
    )
    received = {}

    class Transport:
        def __init__(self, host):
            assert host == "compute"

        def call(self, command, **kwargs):
            if "stdin" in kwargs:
                with tarfile.open(
                    fileobj=io.BytesIO(kwargs["stdin"].read())
                ) as payload:
                    names = payload.getnames()
                    assert not any(
                        "installations/" in name or name.endswith((".whl", ".so"))
                        for name in names
                    )
                    received["bindings"] = json.load(
                        payload.extractfile("study/bindings.json")
                    )
                    for arm in ("before", "after"):
                        assert f"sources/{arm}/source.tar" in names
                        received[arm] = json.load(
                            payload.extractfile(f"sources/{arm}/source.json")
                        )
                return b""
            if "data" in kwargs:
                received["job"] = json.loads(kwargs["data"])
                return b""
            return json.dumps("/remote/home").encode()

    monkeypatch.setattr(remote, "SSH", Transport)
    monkeypatch.setattr(remote, "query", lambda *args: {"status": "submitted"})
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
    result = remote.launch(args)
    assert result["status"] == "submitted"
    for arm in ("before", "after"):
        binding = received["bindings"]["arms"][arm]
        assert set(binding) == {"source", "python"}
        assert binding["source"].endswith(f"/sources/{arm}")
        assert (
            received["job"]["sources"][arm]["source_archive_sha256"]
            == received[arm]["source_archive_sha256"]
        )


def test_prepared_installation_relocates_but_never_ignores_changed_bytes(tmp_path):
    root = tmp_path / "uploaded"
    (root / "installed/causalab").mkdir(parents=True)
    (root / "installed/causalab/code.py").write_text("selected_source = 1\n")
    commit = "a" * 40
    write_record(
        root / "installation.json",
        {
            "source_commit": commit,
            "package_root": "/old/host/installed",
            "installed_files": manifest(root / "installed"),
        },
    )
    record = load_installation(root, commit)
    assert record["package_root"] == str(root / "installed")
    with pytest.raises(ValueError, match="pinned"):
        load_installation(root, "main")
    with pytest.raises(ValueError, match="source commit"):
        load_installation(root, "b" * 40)
    (root / "installed/causalab/code.py").write_text("selected_source = 2\n")
    with pytest.raises(ValueError, match="installation changed"):
        load_installation(root, commit)


@pytest.mark.parametrize("corrupt", [False, True])
@pytest.mark.parametrize(
    "artifact_name", ["trace.json", "capture.nsys-rep", "capture.ncu-rep"]
)
def test_fetch_streams_and_verifies_before_publication(
    tmp_path, monkeypatch, corrupt, artifact_name
):
    trace = tmp_path / artifact_name
    trace.write_text('{"traceEvents":[]}')
    archive = tmp_path / "export.tar"
    record = {
        "schema_version": 1,
        "state": {"status": "completed"},
        "files": {
            f"results/{artifact_name}": "0" * 64 if corrupt else file_hash(trace)
        },
    }
    with tarfile.open(archive, "w") as stream:
        stream.add(trace, arcname=f"results/{artifact_name}")
        payload = json.dumps(record).encode()
        info = tarfile.TarInfo("export.json")
        info.size = len(payload)
        stream.addfile(info, io.BytesIO(payload))

    class SSHBoundary:
        def __init__(self, host):
            assert host == "compute"

        def call(self, command, *, output):
            assert "export" in command
            with archive.open("rb") as stream:
                shutil.copyfileobj(stream, output)
            return b""

    monkeypatch.setattr("causalab.measurement.deployment.remote.SSH", SSHBoundary)
    receipt = {
        "kind": "measurement",
        "host": "compute",
        "control_python": "python3",
        "remote_job": "/job",
    }
    output = tmp_path / "fetched"
    if corrupt:
        with pytest.raises(ValueError, match="manifest mismatch"):
            fetch(receipt, output)
        assert not output.exists()
    else:
        assert fetch(receipt, output) == {"status": "completed"}
        assert (output / f"results/{artifact_name}").read_bytes() == trace.read_bytes()
        with pytest.raises(ValueError, match="fresh output"):
            fetch(receipt, output)
