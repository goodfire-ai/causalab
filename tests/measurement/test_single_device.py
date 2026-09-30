"""Unsupported collection layouts fail before artifacts or child processes."""

import json
from types import SimpleNamespace

from hypothesis import given, settings, strategies as st
import pytest

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def single_process(monkeypatch):
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv("LOCAL_WORLD_SIZE", raising=False)


@settings(max_examples=30)
@given(index=st.integers(min_value=0, max_value=65535))
def test_one_selected_gpu_is_valid_regardless_of_visible_devices(index):
    from causalab.measurement.device import require_single_device

    require_single_device(f"cuda:{index}", environment={"CUDA_VISIBLE_DEVICES": "0,1"})


@pytest.mark.parametrize("device", ["cpu", "cuda", "cuda:0", "cuda:2"])
def test_single_device_and_single_process_launch_are_supported(device):
    from causalab.measurement.device import require_single_device

    require_single_device(
        device, environment={"WORLD_SIZE": "1", "LOCAL_WORLD_SIZE": "1"}
    )


@pytest.mark.parametrize(
    "device",
    [
        "cuda:0,cuda:1",
        "cuda:0, cuda:1",
        "cuda:",
        "cudafoo",
        "cuda:-1",
        "cpu,cpu",
        "",
        None,
        ["cuda:0", "cuda:1"],
    ],
)
def test_device_lists_and_invalid_spellings_are_refused(device):
    from causalab.measurement.device import (
        MeasurementDeviceError,
        require_single_device,
    )

    with pytest.raises(MeasurementDeviceError, match="single GPU"):
        require_single_device(device, environment={})


@settings(max_examples=30)
@given(
    world=st.integers(min_value=2, max_value=1024),
    key=st.sampled_from(["WORLD_SIZE", "LOCAL_WORLD_SIZE"]),
)
def test_distributed_launch_is_refused_even_with_one_local_device(world, key):
    from causalab.measurement.device import (
        MeasurementDeviceError,
        require_single_device,
    )

    with pytest.raises(MeasurementDeviceError, match=key):
        require_single_device("cuda:0", environment={key: str(world)})


@pytest.mark.parametrize("world", ["0", "-1", "", "two"])
def test_malformed_world_size_cannot_bypass_guard(world):
    from causalab.measurement.device import (
        MeasurementDeviceError,
        require_single_device,
    )

    with pytest.raises(MeasurementDeviceError, match="WORLD_SIZE"):
        require_single_device("cpu", environment={"WORLD_SIZE": world})


@pytest.mark.parametrize(
    "entry",
    [
        "study",
        "session",
        "captures",
        "capture",
        "capture_serve",
        "worker",
        "probe",
        "collect",
    ],
)
@pytest.mark.parametrize("distributed", [False, True])
def test_entrypoints_refuse_before_writing_or_starting_work(
    tmp_path, monkeypatch, entry, distributed
):
    from causalab.measurement.collection import collect
    from causalab.measurement.capture import controller as captures
    from causalab.measurement.capture import worker as capture_worker
    from causalab.measurement.runtime.probe import probe_case
    from causalab.measurement.runtime.worker import Worker
    from causalab.measurement.study import controller
    from causalab.measurement.device import MeasurementDeviceError

    device = "cuda:0" if distributed else "cuda:0,cuda:1"
    if distributed:
        monkeypatch.setenv("WORLD_SIZE", "2")
    config = {"device": device}
    output = tmp_path / "output"
    receipt = tmp_path / "measurement.json"
    receipt.write_text('{"samples": [1]}')
    bindings = tmp_path / "bindings.json"
    bindings.write_text(json.dumps(config))

    def unexpected(*args, **kwargs):
        pytest.fail("collection work started before the single-device guard")

    monkeypatch.setattr(controller.subprocess, "Popen", unexpected)
    monkeypatch.setattr(captures, "execute", unexpected)
    calls = {
        "collect": lambda: collect(
            unexpected,
            output,
            case="case",
            input_identity="input",
            scope="scope",
            reset_policy="reset",
            device=device,
        ),
        "study": lambda: controller.run(tmp_path / "study.json", bindings, output),
        "session": lambda: controller.ProcessSession("python", config, output),
        "captures": lambda: captures.run_captures(
            "python", config, {}, "case", 0, output, receipt
        ),
        "capture": lambda: capture_worker.capture(config),
        "capture_serve": lambda: capture_worker.serve(config),
        "worker": lambda: Worker(config),
        "probe": lambda: probe_case(
            unexpected, {}, output, case="case", seed=0, device=device
        ),
    }
    with pytest.raises(MeasurementDeviceError, match="single GPU"):
        calls[entry]()
    assert not output.exists()
    assert receipt.read_text() == '{"samples": [1]}'
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "bindings.json",
        "measurement.json",
    ]


@pytest.mark.parametrize("axis", ["data", "pipeline", "tensor", "expert", "context"])
def test_probe_refuses_distributed_bundle_before_preparing(tmp_path, axis):
    from causalab.measurement.device import MeasurementDeviceError
    from causalab.measurement.runtime.probe import probe_case
    from causalab.protocol.parallel import ParallelGeometry

    bundle = SimpleNamespace(geometry=ParallelGeometry(**{axis: 2}))
    with pytest.raises(MeasurementDeviceError, match="parallel"):
        probe_case(
            None,
            {"model": bundle},
            tmp_path / "probe",
            case="case",
            seed=0,
            device="cpu",
        )
    assert not (tmp_path / "probe").exists()


def test_probe_refuses_model_spread_across_devices_before_preparing(tmp_path):
    from causalab.measurement.device import MeasurementDeviceError
    from causalab.measurement.runtime.probe import probe_case

    bundle = SimpleNamespace(devices=SimpleNamespace(spelling="cuda:0,cuda:1"))
    with pytest.raises(MeasurementDeviceError, match="single GPU"):
        probe_case(
            None,
            {"model": bundle},
            tmp_path / "probe",
            case="case",
            seed=0,
            device="cuda:0",
        )
    assert not (tmp_path / "probe").exists()


@pytest.mark.parametrize("world", [1, 2, 8])
def test_runtime_guard_checks_initialized_process_group_without_environment(
    monkeypatch, world
):
    import torch.distributed as distributed
    from causalab.measurement.device import (
        MeasurementDeviceError,
        require_single_device_runtime,
    )

    monkeypatch.setattr(distributed, "is_available", lambda: True)
    monkeypatch.setattr(distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(distributed, "get_world_size", lambda: world)
    if world == 1:
        require_single_device_runtime("cpu")
    else:
        with pytest.raises(MeasurementDeviceError, match="parallel world size"):
            require_single_device_runtime("cpu")


def test_guard_never_enumerates_or_initializes_cuda(monkeypatch):
    import torch
    from causalab.measurement.device import require_single_device_runtime

    def unexpected(*args, **kwargs):
        pytest.fail("guard must use requested placement, not CUDA availability")

    for name in ("is_available", "device_count", "current_device", "init"):
        monkeypatch.setattr(torch.cuda, name, unexpected)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    require_single_device_runtime("cuda:0")
