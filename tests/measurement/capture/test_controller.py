"""Capture lifecycle, scientific reset, subprocess boundaries and failure isolation."""

from contextlib import contextmanager, nullcontext
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from causalab.measurement.collection import Operation, file_hash, write_record
from causalab.profiling import ProfilerUnavailable
from causalab.profiling.torch import BACKEND
from causalab.measurement.capture import controller
from causalab.measurement.capture import worker as child

pytestmark = pytest.mark.smoke


def _config(tmp_path, *, cold=True):
    return {
        "controller_root": str(tmp_path),
        "package_root": str(tmp_path),
        "device": "cpu",
        "plan": {
            "warmups": 0,
            "profile": {"backends": {"torch": {}}, "timeout_seconds": 10},
            "cases": {"case": {"cold_process": cold}},
        },
    }


def _identity():
    return {
        "models": {"m": {"files": {"weights": "digest"}}},
        "execution_probe": {"observed": True},
    }


@pytest.mark.parametrize("cold,modes", [(True, ["cold", "warm"]), (False, ["warm"])])
def test_capture_pair_uses_file_ipc_and_preserves_clean_sample(
    tmp_path, monkeypatch, cold, modes
):
    config = _config(tmp_path, cold=cold)
    receipt = tmp_path / "measurement.json"
    sample = {"seconds": 0.1, "observations": {"file": "clean.safetensors"}}
    write_record(
        receipt,
        {
            "status": "completed",
            "samples": [sample],
            "trace": {"status": "not_requested"},
        },
    )
    seen = []

    def execute(command, **kwargs):
        captured = json.loads(Path(command[-1]).read_text())
        spec = captured["capture"]
        seen.append(spec["mode"])
        assert captured["verified_model_files"] == {"m": {"weights": "digest"}}
        assert captured["execution_probe"] == _identity()["execution_probe"]
        target = Path(spec["directory"])
        (target / "trace.json").write_text('{"traceEvents": []}')
        observed = target / "observations.safetensors"
        observed.write_bytes(b"captured-values")
        write_record(
            Path(spec["result"]),
            {
                "status": "completed",
                "identity": _identity(),
                "observations": {"sha256": file_hash(observed)},
                "observation_specs": {},
                "instrumentation": {},
                "coverage": {"warmups": 1 if spec["mode"] == "warm" else 0},
            },
        )
        kwargs["log"].write("profiler chatter is not JSON IPC\n")

    monkeypatch.setattr(controller, "execute", execute)
    controller.run_captures(
        sys.executable, config, _identity(), "case", 7, tmp_path, receipt
    )
    result = json.loads(receipt.read_text())
    assert seen == modes
    assert result["samples"] == [sample]
    assert result["trace"] == {"status": "not_requested"}
    assert result["capture_plan"] == {"backends": ["torch"], "modes": modes}
    for record, mode in zip(result["captures"], modes, strict=True):
        assert record["status"] == "completed"
        assert record["id"] == f"torch:{mode}"
        assert record["pair_id"] == "torch"
        assert record["seed"] == 7
        assert record["coverage"]["warmups"] == (mode == "warm")
        for artifact in [*record["artifacts"], record["observations"], *record["logs"]]:
            assert file_hash(tmp_path / artifact["file"]) == artifact["sha256"]


@pytest.mark.parametrize(
    "failure", ["unavailable", "exit", "missing", "identity", "empty"]
)
def test_profiler_failure_keeps_clean_measurements_and_attempts_both_modes(
    tmp_path, monkeypatch, failure
):
    receipt = tmp_path / "measurement.json"
    write_record(receipt, {"status": "completed", "samples": [{"seconds": 1.0}]})
    attempts = []

    def probe():
        attempts.append("probe")
        if failure == "unavailable":
            raise ProfilerUnavailable("no tool")
        return {"version": "test"}

    def execute(command, **kwargs):
        spec = json.loads(Path(command[-1]).read_text())["capture"]
        target = Path(spec["directory"])
        if failure == "exit":
            raise subprocess.CalledProcessError(1, command)
        if failure == "missing":
            return
        write_record(
            Path(spec["result"]),
            {
                "status": "completed",
                "identity": {} if failure == "identity" else _identity(),
            },
        )
        (target / "trace.json").write_text("")

    monkeypatch.setattr(
        controller, "get_backend", lambda _: replace(BACKEND, probe=probe)
    )
    monkeypatch.setattr(controller, "execute", execute)
    controller.run_captures(
        sys.executable, _config(tmp_path), _identity(), "case", 7, tmp_path, receipt
    )
    result = json.loads(receipt.read_text())
    assert result["status"] == "completed"
    assert result["samples"] == [{"seconds": 1.0}]
    assert len(attempts) == 2
    assert {row["status"] for row in result["captures"]} == (
        {"unavailable"} if failure == "unavailable" else {"failed"}
    )


@pytest.mark.parametrize("mode", ["cold", "warm"])
@pytest.mark.parametrize("external", [False, True])
def test_worker_capture_boundaries_and_reset(tmp_path, monkeypatch, mode, external):
    import torch
    from causalab.measurement.runtime import worker as measurement_worker
    from causalab.measurement.runtime import observations as measurement_observations
    from causalab.measurement.capture import ranges as measurement_profiling

    events = []
    active = False
    workers = []

    class Profiler:
        def __enter__(self):
            nonlocal active
            assert not active
            active = True
            events.append("start")
            return self

        def __exit__(self, *args):
            nonlocal active
            active = False
            events.append("stop")

        def export_chrome_trace(self, path):
            assert not active
            Path(path).write_text('{"traceEvents": []}')

    class Worker:
        prepared = False

        def __init__(self, config):
            workers.append(self)
            events.append("worker")
            assert active == (mode == "cold" and not external)
            self.identity = _identity()
            self.engine, self.bundles = None, {}
            self.plan = {"cases": {"case": {}}, "observations": {}}
            self.loaded = SimpleNamespace(
                document=SimpleNamespace(output_dir="workflow")
            )

        @contextmanager
        def prepare(self, case, seed, directory):
            events.append("reset")
            assert seed == 7
            self.prepared = True

            def run():
                events.append("run")
                return torch.tensor([float(seed)])

            def observe(result):
                assert not active
                assert self.prepared, "observations need the prepared operation"
                events.append("observe")
                return {"value": result}

            try:
                yield Operation(run, observe)
            finally:
                self.prepared = False

    @contextmanager
    def phases(*args, **kwargs):
        yield {"calls": {"forward": 1}}

    monkeypatch.setattr(measurement_worker, "Worker", Worker)

    def specs(*args, **kwargs):
        assert workers[-1].prepared
        assert not active
        return {}

    monkeypatch.setattr(measurement_observations, "observation_specs", specs)
    monkeypatch.setattr(measurement_profiling, "phase_ranges", phases)
    monkeypatch.setattr(torch.profiler, "profile", lambda **kwargs: Profiler())
    if external:
        monkeypatch.setattr(
            child,
            "get_backend",
            lambda _: replace(BACKEND, capture_control="cuda_profiler_api"),
        )
        monkeypatch.setattr(child, "_cuda_capture", lambda *args: Profiler())
        monkeypatch.setattr(
            torch.profiler,
            "profile",
            lambda **kwargs: pytest.fail("nested native profiler"),
        )
    config = _config(tmp_path)
    config["capture"] = {
        "backend": "torch",
        "mode": mode,
        "case": "case",
        "seed": 7,
        "directory": str(tmp_path),
        "result": str(tmp_path / "result.json"),
        "options": {},
    }
    result = child.capture(config)
    assert result["status"] == "completed"
    assert result["coverage"]["warmups"] == (mode == "warm")
    expected = (
        ["worker", "reset", "run", "reset", "start", "run", "stop", "observe"]
        if mode == "warm"
        else (
            ["worker", "reset", "run", "observe"]
            if external
            else ["start", "worker", "reset", "run", "stop", "observe"]
        )
    )
    assert events == expected
    assert (tmp_path / "trace.json").exists() != external


@pytest.mark.parametrize("leader_exits", [False, True])
def test_cleanup_kills_term_ignoring_descendant(tmp_path, leader_exits):
    pidfile = tmp_path / "child.pid"
    script = tmp_path / "wrapper.py"
    script.write_text(
        "import subprocess,sys,time\nfrom pathlib import Path\n"
        "child='import os,signal,sys,time; from pathlib import Path; "
        "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
        "Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(60)'\n"
        "subprocess.Popen([sys.executable,'-c',child,sys.argv[1]])\n"
        "while not Path(sys.argv[1]).exists(): time.sleep(.01)\n"
        + ("" if leader_exits else "time.sleep(60)\n")
    )
    with (tmp_path / "log").open("w") as log:
        with (
            nullcontext() if leader_exits else pytest.raises(subprocess.TimeoutExpired)
        ):
            controller.execute(
                [sys.executable, str(script), str(pidfile)],
                cwd=tmp_path,
                log=log,
                timeout=1,
            )
    child_pid = int(pidfile.read_text())
    for _ in range(20):
        state = subprocess.run(
            ["ps", "-o", "stat=", "-p", str(child_pid)], capture_output=True, text=True
        ).stdout.strip()
        if not state or state.startswith("Z"):
            break
        time.sleep(0.05)
    assert not state or state.startswith("Z"), (
        f"capture descendant survived: {child_pid} {state}"
    )


@pytest.mark.parametrize("exit_code", [0, 7, None])
def test_group_signals_only_target_an_unreaped_owned_child(
    tmp_path, monkeypatch, exit_code
):
    original_popen, original_killpg = subprocess.Popen, controller.os.killpg
    owned = {}
    signals = []

    def popen(*args, **kwargs):
        process = original_popen(*args, **kwargs)
        owned[process.pid] = process
        return process

    def killpg(pid, sig):
        assert owned[pid].returncode is None, "PID ownership ended before group signal"
        signals.append(sig)
        original_killpg(pid, sig)

    monkeypatch.setattr(controller.subprocess, "Popen", popen)
    monkeypatch.setattr(controller.os, "killpg", killpg)
    code = (
        "import time; time.sleep(60)"
        if exit_code is None
        else f"raise SystemExit({exit_code})"
    )
    expected = (
        pytest.raises(subprocess.TimeoutExpired)
        if exit_code is None
        else pytest.raises(subprocess.CalledProcessError)
        if exit_code
        else nullcontext()
    )
    with (tmp_path / "log").open("w") as log, expected:
        controller.execute(
            [sys.executable, "-c", code], cwd=tmp_path, log=log, timeout=0.5
        )
    assert signals == [controller.signal.SIGTERM, controller.signal.SIGKILL]
    assert all(process.returncode is not None for process in owned.values())


def test_cleanup_refuses_to_signal_an_already_reaped_process(monkeypatch):
    monkeypatch.setattr(
        controller.os, "killpg", lambda *_: pytest.fail("signalled released PID")
    )
    controller._stop_group(SimpleNamespace(returncode=0))


def test_cuda_profiler_api_stops_even_on_operation_error():
    calls = []
    runtime = SimpleNamespace(
        cudaProfilerStart=lambda: calls.append("start"),
        cudaProfilerStop=lambda: calls.append("stop"),
    )
    cuda = SimpleNamespace(
        synchronize=lambda device: calls.append("sync"),
        cudart=lambda: runtime,
        check_error=lambda result: None,
    )
    with pytest.raises(ValueError):
        with child._cuda_capture(SimpleNamespace(cuda=cuda), "cuda:0"):
            raise ValueError("operation failed")
    assert calls == ["sync", "start", "sync", "stop"]


def test_cuda_profiler_stop_runs_when_sync_fails():
    calls = []

    def synchronize(device):
        calls.append("sync")
        if calls.count("sync") == 2:
            raise RuntimeError("async CUDA error")

    runtime = SimpleNamespace(
        cudaProfilerStart=lambda: calls.append("start"),
        cudaProfilerStop=lambda: calls.append("stop"),
    )
    cuda = SimpleNamespace(
        synchronize=synchronize, cudart=lambda: runtime, check_error=lambda result: None
    )
    with pytest.raises(RuntimeError, match="async CUDA error"):
        with child._cuda_capture(SimpleNamespace(cuda=cuda), "cuda:0"):
            pass
    assert calls == ["sync", "start", "sync", "stop"]


def test_session_capture_releases_resident_model_before_launch(tmp_path, monkeypatch):
    from causalab.measurement.study.controller import ProcessSession

    events = []
    session = object.__new__(ProcessSession)
    session.python, session.config, session.identity = sys.executable, {}, {}
    monkeypatch.setattr(session, "close", lambda: events.append("close"))

    def capture(*args):
        events.append("capture")
        assert json.loads(args[-1].read_text()) == {"captures": []}

    monkeypatch.setattr(controller, "run_captures", capture)
    result = session.capture("case", 7, 0, tmp_path / "profile")
    assert result == tmp_path / "profile" / "measurement.json"
    assert events == ["close", "capture"]


@pytest.mark.parametrize(
    "backends,native",
    [({"torch": {}}, True), ({"nsys": {}}, False), ({"ncu": {}}, False)],
)
def test_preflight_only_uses_selected_native_torch_backend(
    tmp_path, monkeypatch, backends, native
):
    from causalab.measurement.runtime import probe as measurement_probe
    from causalab.measurement.runtime.worker import Worker

    flags = []
    worker = object.__new__(Worker)
    worker.config = {
        "device": "cpu",
        "arm": "before",
        "probe_directory": str(tmp_path / "probe"),
    }
    worker.plan = {
        "cases": {"case": {}},
        "seeds": [7],
        "profile": {"cases": ["case"], "backends": backends},
    }
    worker.identity = {"comparison_identity": "scientific-inputs"}
    worker.bundles = {}

    def probe(*args, **kwargs):
        flags.append(kwargs["profile"])
        return {"logical_token_rows": []}

    monkeypatch.setattr(measurement_probe, "probe_case", probe)
    monkeypatch.setattr(measurement_probe, "reuse_evidence", lambda record: record)
    worker.attest_execution()
    assert flags == [native]


def test_sigterm_cleans_capture_process_group(tmp_path):
    import os
    import signal

    pidfile = tmp_path / "pids.json"
    wrapper = tmp_path / "wrapper.py"
    wrapper.write_text(
        "import os,subprocess,sys,time,json\nfrom pathlib import Path\n"
        "child=subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        "Path(sys.argv[1]).write_text(json.dumps([os.getpid(),child.pid]))\n"
        "time.sleep(60)\n"
    )
    supervisor = tmp_path / "controller.py"
    repository = Path(__file__).resolve().parents[3]
    supervisor.write_text(
        "import sys\nfrom pathlib import Path\n"
        f"sys.path.insert(0, {str(repository)!r})\n"
        "from causalab.measurement.capture.controller import execute\n"
        "with Path(sys.argv[3]).open('w') as log:\n"
        " execute([sys.executable,sys.argv[1],sys.argv[2]],cwd=Path(sys.argv[1]).parent,log=log,timeout=20)\n"
    )
    with (tmp_path / "controller.log").open("w") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                str(supervisor),
                str(wrapper),
                str(pidfile),
                str(tmp_path / "capture.log"),
            ],
            stdout=log,
            stderr=log,
        )
        group = None
        try:
            deadline = time.monotonic() + 10
            while not pidfile.exists() and time.monotonic() < deadline:
                time.sleep(0.02)
            group, child_pid = json.loads(pidfile.read_text())
            process.send_signal(signal.SIGTERM)
            assert process.wait(timeout=10) != 0
            state = subprocess.run(
                ["ps", "-o", "stat=", "-p", str(child_pid)],
                capture_output=True,
                text=True,
            ).stdout.strip()
            assert not state or state.startswith("Z")
        finally:
            if group is not None:
                try:
                    os.killpg(group, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            if process.poll() is None:
                process.kill()
            process.wait()
    assert (
        "profiling controller received SIGTERM"
        in (tmp_path / "controller.log").read_text()
    )
