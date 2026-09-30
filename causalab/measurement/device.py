"""Device and process limits for measurement artifact collection."""

from __future__ import annotations

from collections.abc import Mapping
import os
import re


class MeasurementDeviceError(ValueError):
    """The requested layout cannot produce supported measurement artifacts."""


_LIMIT = (
    "measurement and profiling artifact collection supports one CPU or a single GPU "
    "in one process; use device 'cpu', 'cuda', or 'cuda:N' without a distributed launch"
)


def require_single_device(
    device: object,
    *,
    world: int = 1,
    environment: Mapping[str, str] | None = None,
) -> None:
    """Check the requested layout without initializing or enumerating CUDA."""
    if not isinstance(device, str) or not re.fullmatch(r"cpu|cuda(?::[0-9]+)?", device):
        raise MeasurementDeviceError(f"{_LIMIT}; got device {device!r}")
    if world != 1:
        raise MeasurementDeviceError(f"{_LIMIT}; got parallel world size {world}")
    environment = os.environ if environment is None else environment
    for name in ("WORLD_SIZE", "LOCAL_WORLD_SIZE"):
        value = environment.get(name, "1")
        try:
            size = int(value)
        except ValueError as exc:
            raise MeasurementDeviceError(f"{_LIMIT}; invalid {name}={value!r}") from exc
        if size != 1:
            raise MeasurementDeviceError(f"{_LIMIT}; got {name}={value!r}")


def require_single_device_runtime(device: object) -> None:
    """Also reject an initialized multi-rank group in an executing worker."""
    require_single_device(device)
    import torch.distributed as distributed

    if distributed.is_available() and distributed.is_initialized():
        require_single_device(device, world=distributed.get_world_size())
