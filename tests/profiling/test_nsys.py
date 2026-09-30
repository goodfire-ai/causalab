"""Nsight Systems launch boundaries without requiring a GPU or installed tool."""

from dataclasses import replace
import json
from pathlib import Path
import subprocess

import pytest

from causalab.profiling import CaptureRequest, ProfilerUnavailable, get_backend
from causalab.profiling import nsys

pytestmark = pytest.mark.unit


@pytest.fixture
def request_spec(tmp_path, monkeypatch):
    monkeypatch.setattr(nsys.shutil, "which", lambda name: "/tools with spaces/nsys")
    return CaptureRequest(
        command=("/python with spaces", "worker.py", "--value", "$(not-a-shell)"),
        output_dir=tmp_path / "capture space",
        mode="cold",
        options=nsys.normalize_options({}),
        device="cuda:0",
    )


def test_cold_command_starts_at_launch_and_keeps_child_argv_intact(request_spec):
    argv = nsys.command(request_spec)
    assert argv[:2] == ["/tools with spaces/nsys", "profile"]
    assert argv[-4:] == list(request_spec.command)
    assert "--capture-range=none" in argv
    assert not any(arg.startswith("--capture-range-end") for arg in argv)
    assert "--trace=cuda,nvtx,osrt" in argv
    assert "--sample=none" in argv
    assert "--cpuctxsw=none" in argv
    assert "--force-overwrite=false" in argv
    assert "--cuda-graph-trace=node" in argv
    assert "--cuda-memory-usage=false" in argv
    assert "--export=none" in argv and "--stats=false" in argv
    assert f"--output={request_spec.output_dir}/capture" in argv
    assert nsys.artifacts(request_spec) == [
        request_spec.output_dir / "capture.nsys-rep"
    ]


def test_warm_command_uses_api_range_without_killing_child(request_spec):
    argv = nsys.command(replace(request_spec, mode="warm"))
    assert "--capture-range=cudaProfilerApi" in argv
    assert "--capture-range-end=stop" in argv
    assert "--kill=none" in argv
    assert not any("stop-shutdown" in arg for arg in argv)
    assert argv[-4:] == list(request_spec.command)


def test_option_normalization_roundtrip_and_argv(request_spec):
    raw = {"cuda_graph_trace": "graph", "cuda_memory_usage": True}
    normalized = nsys.normalize_options(raw)
    assert nsys.normalize_options(json.loads(json.dumps(normalized))) == raw
    assert normalized is not raw
    argv = nsys.command(replace(request_spec, options=normalized))
    assert "--cuda-graph-trace=graph" in argv
    assert "--cuda-memory-usage=true" in argv


@pytest.mark.parametrize(
    "options",
    [
        {"extra_args": ["--sample=system-wide"]},
        {"output": "/elsewhere"},
        {"cuda_graph_trace": "node --kill=sigkill"},
        {"cuda_graph_trace": []},
        {"cuda_graph_trace": None},
        {"cuda_memory_usage": "false"},
        {"cuda_memory_usage": 1},
    ],
)
def test_invalid_options_fail_before_any_probe(options, monkeypatch):
    monkeypatch.setattr(nsys.shutil, "which", lambda _: pytest.fail("tool probe"))
    with pytest.raises(ValueError):
        nsys.normalize_options(options)


@pytest.mark.parametrize("mode", ["hot", "", "cold --kill=sigkill"])
def test_invalid_mode_rejected(request_spec, mode):
    with pytest.raises(ValueError, match="capture mode"):
        nsys.command(replace(request_spec, mode=mode))


@pytest.mark.parametrize("child", [(), ("--bad",)])
def test_invalid_child_rejected(request_spec, child):
    with pytest.raises(ValueError, match="child executable"):
        nsys.command(replace(request_spec, command=child))


def test_output_macros_are_not_expanded(request_spec):
    with pytest.raises(ValueError, match="macros"):
        nsys.command(replace(request_spec, output_dir=Path("/tmp/%p/capture")))


def test_artifacts_are_exact_not_a_glob(request_spec):
    request_spec.output_dir.mkdir()
    (request_spec.output_dir / "stale.nsys-rep").write_bytes(b"previous capture")
    expected = nsys.artifacts(request_spec)
    assert expected == [request_spec.output_dir / "capture.nsys-rep"]
    assert not expected[0].exists()


def test_missing_tool_is_explicit(monkeypatch, request_spec):
    monkeypatch.setattr(nsys.shutil, "which", lambda _: None)
    for operation in (nsys.probe, lambda: nsys.command(request_spec)):
        with pytest.raises(ProfilerUnavailable, match="compute host PATH"):
            operation()


def test_probe_records_version_and_limits_execution(monkeypatch):
    calls = []
    monkeypatch.setattr(nsys.shutil, "which", lambda _: "/tools/nsys")

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(
            argv, 0, "NVIDIA Nsight Systems 2025.3\n", ""
        )

    monkeypatch.setattr(nsys.subprocess, "run", run)
    result = nsys.probe()
    assert result["executable"] == "/tools/nsys"
    assert result["version"] == "NVIDIA Nsight Systems 2025.3"
    assert result["capabilities"]["warm_cuda_profiler_api"] is True
    assert calls == [
        (
            ["/tools/nsys", "--version"],
            {"capture_output": True, "text": True, "timeout": 10, "check": False},
        )
    ]


@pytest.mark.parametrize(
    "failure",
    [
        subprocess.TimeoutExpired(["nsys", "--version"], 10),
        OSError("executable disappeared"),
        subprocess.CompletedProcess(["nsys"], 1, "", "driver failure"),
        subprocess.CompletedProcess(["nsys"], 0, "", ""),
    ],
)
def test_probe_failure_is_explicit(monkeypatch, failure):
    monkeypatch.setattr(nsys.shutil, "which", lambda _: "/tools/nsys")

    def run(*args, **kwargs):
        if isinstance(failure, Exception):
            raise failure
        return failure

    monkeypatch.setattr(nsys.subprocess, "run", run)
    with pytest.raises(ProfilerUnavailable, match="nsys --version failed"):
        nsys.probe()


def test_backend_contract():
    backend = get_backend("nsys")
    assert backend is nsys.BACKEND
    assert backend.artifact_format == "nsys-rep"
    assert backend.capture_control == "cuda_profiler_api"
    assert "process launch" in backend.startup_coverage
    assert "overhead" in backend.perturbation
