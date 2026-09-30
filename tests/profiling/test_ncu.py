"""Contract tests for managed Nsight Compute capture without a CUDA install."""

from dataclasses import replace
import json
from pathlib import Path
import subprocess

import pytest

from causalab.profiling import CaptureRequest, ProfilerUnavailable
from causalab.profiling import ncu

pytestmark = pytest.mark.unit


@pytest.fixture
def request_spec(tmp_path, monkeypatch):
    monkeypatch.setattr(ncu.shutil, "which", lambda name: "/opt/nsight/ncu")
    return CaptureRequest(
        command=("/venv/bin/python", "-m", "worker", "input with spaces.json"),
        output_dir=tmp_path / "capture dir",
        mode="cold",
        options={},
        device="cuda:0",
    )


def test_defaults_are_bounded_and_round_trip_without_subprocess(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("normalizing options must not inspect the compute environment")

    monkeypatch.setattr(ncu.subprocess, "run", forbidden)
    monkeypatch.setattr(ncu.shutil, "which", forbidden)
    normalized = ncu.normalize_options({})
    assert normalized["launch_count"] == 10
    assert normalized["launch_skip"] == 0
    assert normalized["set"] == "basic"
    assert normalized["replay_mode"] == "kernel"
    assert normalized["cache_control"] == "all"
    assert normalized["clock_control"] == "none"
    assert ncu.normalize_options(json.loads(json.dumps(normalized))) == normalized


@pytest.mark.parametrize(
    "options",
    [
        {"arbitrary_args": ["--launch-count", "0"]},
        {"set": "--full"},
        {"set": None},
        {"sections": "SpeedOfLight"},
        {"sections": ["SpeedOfLight", "SpeedOfLight"]},
        {"sections": [False]},
        {"metrics": ["metric,other_metric"]},
        {"metrics": ["$(command)"]},
        {"metrics": ["--set"]},
        {"sections": ["--kill"]},
        {"launch_skip": -1},
        {"launch_skip": True},
        {"launch_count": 0},
        {"launch_count": None},
        {"launch_count": 1.5},
        {"launch_count": "10"},
        {"launch_count": False},
        {"kernel_name": ""},
        {"kernel_name": "regex:"},
        {"kernel_name": "name\x00bad"},
        {"kernel_name": "name\nbad"},
        {"kernel_name": []},
        {"kernel_name_base": "unknown"},
        {"cache_control": "unknown"},
        {"clock_control": "base"},
        {"replay_mode": "application"},
        {"replay_mode": "range"},
        {"replay_mode": "app-range"},
    ],
)
def test_rejects_unsupported_or_ambiguous_options(options):
    with pytest.raises(ValueError, match="ncu"):
        ncu.normalize_options(options)


@pytest.mark.parametrize("mode,profile_from_start", [("cold", "on"), ("warm", "off")])
def test_launch_scopes_capture_and_preserves_complete_child(
    request_spec, mode, profile_from_start
):
    spec = replace(request_spec, mode=mode)
    argv = ncu.command(spec)
    assert argv[0] == "/opt/nsight/ncu"
    assert argv[argv.index("--profile-from-start") + 1] == profile_from_start
    assert argv[argv.index("--target-processes") + 1] == "application-only"
    assert argv[argv.index("--config-file") + 1] == "off"
    assert argv[argv.index("--rename-kernels") + 1] == "off"
    assert argv[argv.index("--kill") + 1] == "off"
    assert argv[argv.index("--launch-count") + 1] == "10"
    assert argv[argv.index("--clock-control") + 1] == "none"
    assert argv[-len(spec.command) :] == list(spec.command)
    assert "--disable-profiler-start-stop" not in argv
    assert "--force-overwrite" not in argv
    assert ncu.BACKEND.capture_control == "cuda_profiler_api"


def test_explicit_metrics_sections_and_kernel_filter_are_argv_values(request_spec):
    # Metacharacters remain a single argv value and cannot become shell commands
    # or separate NCU options, even if the filter begins with dashes.
    kernel_filter = "--launch-count=0;$(touch /tmp/not-executed)"
    spec = replace(
        request_spec,
        options={
            "sections": ["SpeedOfLight", "MemoryWorkloadAnalysis"],
            "metrics": ["sm__cycles_elapsed.avg", "dram__bytes_read.sum"],
            "kernel_name": kernel_filter,
            "launch_skip": 2,
            "launch_count": 3,
            "cache_control": "none",
        },
    )
    argv = ncu.command(spec)
    assert "--set" not in argv
    assert argv.count("--section") == 2
    assert argv[argv.index("--metrics") + 1] == (
        "sm__cycles_elapsed.avg,dram__bytes_read.sum"
    )
    assert f"--kernel-name={kernel_filter}" in argv
    assert argv[argv.index("--launch-skip") + 1] == "2"
    assert argv[argv.index("--launch-count") + 1] == "3"
    assert argv[argv.index("--cache-control") + 1] == "none"
    normalized = ncu.normalize_options(spec.options)
    assert ncu.normalize_options(normalized) == normalized


def test_artifact_identity_does_not_accept_stale_glob(request_spec):
    request_spec.output_dir.mkdir()
    stale = request_spec.output_dir / "old.ncu-rep"
    stale.write_bytes(b"old report")
    expected = request_spec.output_dir / "capture.ncu-rep"
    assert ncu.artifacts(request_spec) == [expected]
    argv = ncu.command(request_spec)
    assert Path(argv[argv.index("--export") + 1]) == expected
    assert not expected.exists()  # Creation and nonempty checks belong to controller.


@pytest.mark.parametrize("directory", ["literal%name", "macro%p", "escaped%%name"])
def test_file_macros_cannot_change_artifact_destination(request_spec, directory):
    spec = replace(request_spec, output_dir=request_spec.output_dir / directory)
    with pytest.raises(ValueError, match="file macros"):
        ncu.command(spec)
    with pytest.raises(ValueError, match="file macros"):
        ncu.artifacts(spec)


def test_probe_records_host_tool_and_limits(monkeypatch):
    monkeypatch.setattr(ncu.shutil, "which", lambda name: "/host/tools/ncu")

    def run(argv, **kwargs):
        assert argv == ["/host/tools/ncu", "--version"]
        assert kwargs == {
            "capture_output": True,
            "text": True,
            "timeout": 15,
            "check": True,
        }
        return subprocess.CompletedProcess(
            argv, 0, "NVIDIA Nsight Compute 2026.3\n", ""
        )

    monkeypatch.setattr(ncu.subprocess, "run", run)
    metadata = ncu.probe()
    assert metadata["executable"] == "/host/tools/ncu"
    assert metadata["version"] == "NVIDIA Nsight Compute 2026.3"
    assert metadata["capabilities"]["performance_timing"] is False
    assert metadata["capabilities"]["replay_modes"] == ["kernel"]
    assert (
        metadata["environment"]
        == ncu.BACKEND.environment
        == {"NV_COMPUTE_PROFILER_DISABLE_STOCK_FILE_DEPLOYMENT": "1"}
    )
    assert "stock" in metadata["capabilities"]["section_definitions"]


def test_missing_tool_fails_explicitly(request_spec, monkeypatch):
    monkeypatch.setattr(ncu.shutil, "which", lambda name: None)
    with pytest.raises(ProfilerUnavailable, match="compute host PATH"):
        ncu.probe()
    with pytest.raises(ProfilerUnavailable, match="compute host PATH"):
        ncu.command(request_spec)


@pytest.mark.parametrize(
    "error",
    [
        OSError("not executable"),
        subprocess.TimeoutExpired("ncu", 15),
        subprocess.CalledProcessError(1, "ncu"),
    ],
)
def test_broken_tool_probe_fails_explicitly(monkeypatch, error):
    monkeypatch.setattr(ncu.shutil, "which", lambda name: "/host/tools/ncu")

    def run(*args, **kwargs):
        raise error

    monkeypatch.setattr(ncu.subprocess, "run", run)
    with pytest.raises(ProfilerUnavailable, match="version probe failed"):
        ncu.probe()


def test_empty_version_is_not_a_successful_probe(monkeypatch):
    monkeypatch.setattr(ncu.shutil, "which", lambda name: "/host/tools/ncu")
    monkeypatch.setattr(
        ncu.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, "", "")
    )
    with pytest.raises(ProfilerUnavailable, match="no version information"):
        ncu.probe()


def test_invalid_capture_requests(request_spec):
    with pytest.raises(ValueError, match="mode"):
        ncu.command(replace(request_spec, mode="other"))
    with pytest.raises(ProfilerUnavailable, match="CUDA"):
        ncu.command(replace(request_spec, device="cpu"))
    with pytest.raises(ValueError, match="child command"):
        ncu.command(replace(request_spec, command=()))
    with pytest.raises(ValueError, match="child executable"):
        ncu.command(replace(request_spec, command=("--set=full",)))
