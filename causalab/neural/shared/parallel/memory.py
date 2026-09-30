"""The memory pre-flight's torch half (``docs/model_parallelism.md`` §2
"check", §3, §11): read this rank's device and refuse the geometry by name
before any weight is read.

A geometry that exhausts device memory may fail inside a collective after
loading. The loader knows the bytes each rank will read before it issues a
read (``shard_read.read_plan``, §5.3),
the geometry says what is replicated and what is sharded, and the CUDA
runtime reports driver-free memory and reusable allocator cache. Resident
weights must fit in their sum; estimated workflow headroom is advisory.
The decision is made
here, on each rank after ``launcher.enter`` and before its load, from
those facts, and refused as ``P4`` on ``--parallel`` naming the rank, the
device, the bytes, and the geometries that would fit
([`memory_check`][]).

**Where the facts are known.** The spawn parent has no CUDA context and
must not open one on every card the children are about to use, so it does
not check (``check_spawn_devices`` covers the device count there); a
``dry-run`` has no device and prints the estimate instead. Off CUDA —
``--device cpu``, the gloo tiers — the check is the identity: there is no
card to fill. A CUDA device whose ``mem_get_info`` fails is left to the
load as well: the rule never refuses a run it cannot measure.
"""

from __future__ import annotations

import dataclasses
import logging

from causalab.protocol.rules.errors import ProtocolError
from causalab.protocol.parallel import ParallelGeometry, format_geometry
from causalab.protocol.parallel_memory import (
    RULE,
    EstimateRule,
    PlacementTable,
    format_bytes,
    memory_check,
    memory_estimate,
)
from causalab.protocol.registry import ModelInfo

__all__ = ["DeviceMemory", "device_memory", "preflight"]

logger = logging.getLogger(__name__)

_FLAG = "--parallel"


@dataclasses.dataclass(frozen=True)
class DeviceMemory:
    """Driver-free, total, and allocator-cached unused bytes on one device."""

    free: int
    total: int
    cached: int = 0

    @property
    def available(self) -> int:
        """Bytes the load can use, bounded by capacity if readings race."""
        return min(self.total, self.free + max(0, self.cached))


def device_memory(device: str) -> DeviceMemory | None:
    """The free, total, and reusable cached bytes of a CUDA ``device`` word (``cuda``,
    ``cuda:1``); ``None`` for any other word, or when CUDA is unavailable or
    the query fails — the check is then the identity."""
    if not device.strip().startswith("cuda"):
        return None
    import torch  # noqa: PLC0415 — the pure verbs stay torch-free

    if not torch.cuda.is_available():
        return None
    try:
        target = torch.device(device)
        free, total = torch.cuda.mem_get_info(target)
        cached = max(
            0, torch.cuda.memory_reserved(target) - torch.cuda.memory_allocated(target)
        )
    except (RuntimeError, ValueError, AssertionError) as err:
        logger.warning(
            "memory preflight: mem_get_info failed on %s (%s); the load decides",
            device,
            err,
        )
        return None
    return DeviceMemory(free=int(free), total=int(total), cached=int(cached))


def preflight(
    *,
    geometry: ParallelGeometry,
    rank: int,
    device: str,
    dtype: str,
    table: PlacementTable,
    info: ModelInfo,
    memory: DeviceMemory | None = None,
    rule: EstimateRule = RULE,
) -> DeviceMemory | None:
    """Reject weights that exceed available memory; warn on estimated headroom.

    ``memory`` is the device's
    reading when the caller has one (a test's); else [`device_memory`][]
    is asked, and ``None`` (off CUDA, no query) makes the check the
    identity. Returns the reading it decided on, for the caller's log.

    Raises:
        ProtocolError: ``P4`` on ``--parallel`` — the refusal
            [`memory_check`][] writes.
    """
    if memory is None:
        memory = device_memory(device)
    if memory is None:
        return None
    refusal = memory_check(
        geometry=geometry,
        rank=rank,
        device=device,
        dtype=dtype,
        table=table,
        info=info,
        free=memory.available,
        total=memory.total,
        rule=rule,
    )
    if refusal is not None:
        # the line opens with the flag the way §2's check lines do; the
        # error names the flag as its path, so the prefix is not said twice
        raise ProtocolError("P4", refusal.removeprefix(f"{_FLAG}: "), path=_FLAG)
    estimate = memory_estimate(
        geometry=geometry, rank=rank, dtype=dtype, table=table, info=info, rule=rule
    )
    fields = {
        "parallel": format_geometry(geometry),
        "rank": rank,
        "device": device,
        "resident_bytes": estimate.resident,
        "estimated_bytes": estimate.footprint,
        "available_bytes": memory.available,
        "free_bytes": memory.free,
        "cached_bytes": memory.cached,
        "total_bytes": memory.total,
    }
    if estimate.footprint > memory.available:
        logger.warning(
            "memory preflight: rank %d on %s has %s available for %s resident "
            "weights and a %s estimated peak; continuing with limited headroom",
            rank,
            device,
            format_bytes(memory.available),
            format_bytes(estimate.resident),
            format_bytes(estimate.footprint),
            extra=fields,
        )
    else:
        logger.info(
            "memory preflight: resident weights and estimated headroom fit",
            extra=fields,
        )
    return memory
