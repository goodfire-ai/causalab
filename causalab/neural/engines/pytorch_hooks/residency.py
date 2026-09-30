"""Residency: what a loaded model holds on its device against what the
loader read (``docs/model_parallelism.md`` §5.3, "one copy").

Shard-on-read must reduce resident storage as well as bytes read. This
module checks for oversized backing allocations, unowned tensors, and
excess allocator reservations after loading.

[`Residency`][] is the census the loader takes right after a sharded
load: per parameter the local tensor's element count, item size and the
bytes of the storage backing it — a parameter that is a ``1 / world`` view
of a whole allocation shows a storage larger than its elements — the
parameters sharing one storage (tied weights, legitimately), and, on CUDA,
the allocator's ``allocated`` and ``reserved`` bytes on the rank's device
(``None`` elsewhere). Given a [`Census`][] taken **before** the load
(``since``) it also walks the live tensors of the process (``gc``) and sums
the storages on that device that were not alive at the census and no
parameter or buffer owns — what the load left behind: a copy held anywhere,
a cache of ``to_local()`` materialisations included. The difference is the
point: a rank process holds tensors of its own before the load — a model
loaded earlier, a tokenizer's tables — and a test process the fixtures and
caches of every test before, none of them the load's; off an exclusive
device the whole-process count would name them. A storage is the same one
when it is alive at the same address with the same size; a freed address
re-used by a larger or smaller allocation is new. [`residency_problems`][]
states the rule over the written record: every parameter read is resident
with exactly the elements the plan requested and no larger backing storage;
the device holds no more than the parameters ([`ALLOCATED_SLACK`][]) and
reserves no more than it holds ([`RESERVED_SLACK`][]); nothing unowned
above [`UNOWNED_SLACK`][]. The parallel golden holds every rank's report
to it on the real checkpoint, the gloo tier on the fixtures.
"""

from __future__ import annotations

import dataclasses
import gc
from typing import Any, Mapping

import torch
from torch.distributed.tensor import DTensor

from causalab.protocol.parallel_memory import DISK_WORDS

__all__ = [
    "ALLOCATED_SLACK",
    "RESERVED_SLACK",
    "UNOWNED_SLACK",
    "Census",
    "Residency",
    "conversion_note",
    "residency_problems",
]

#: Bytes the device may hold beyond the parameters right after a load: the
#: CUDA context's own tensors and the loader's last transient.
ALLOCATED_SLACK = 512 << 20
#: Bytes the allocator may reserve beyond what it holds: the tail of the
#: weights that did not fit transformers' warm-up block (its estimate divides
#: every planned parameter by the world, the replicated ones too) lands in
#: fresh segments, plus fragmentation.
RESERVED_SLACK = 2 << 30
#: Bytes of live tensors on the device that no parameter or buffer owns, when
#: censused: rotary tables and index tensors a forward left behind are
#: kilobytes; one held expert shard is hundreds of megabytes.
UNOWNED_SLACK = 64 << 20

#: The document's word of a torch dtype's name (``str(dtype)`` less the
#: ``torch.`` prefix) — the resident half of "read as fp32, held as bf16".
_TORCH_WORDS: Mapping[str, str] = {
    "float16": "fp16",
    "bfloat16": "bf16",
    "float32": "fp32",
    "float64": "fp64",
}


def _local(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def _storage_key(tensor: torch.Tensor) -> tuple[str, int]:
    storage = tensor.untyped_storage()
    return (str(tensor.device), storage.data_ptr())


#: A storage's identity in a census: its device and address.
StorageKey = tuple[str, int]


@dataclasses.dataclass(frozen=True)
class Census:
    """The live tensor storages on a device at one moment (module
    docstring): by [`StorageKey`][], the bytes each holds. Taken before a
    load, it is the baseline the load's residency is measured against."""

    device: str
    storages: Mapping[StorageKey, int]

    def __init__(self, device: torch.device | str, storages: Mapping[StorageKey, int]):
        object.__setattr__(self, "device", str(device))
        object.__setattr__(self, "storages", dict(storages))

    @classmethod
    def take(cls, device: torch.device) -> "Census":
        """Walk the process's live tensors (``gc``) on ``device`` now."""
        return cls(device, _live_storages(device))

    def holds(self, key: StorageKey, nbytes: int) -> bool:
        """The storage was alive at the census: same address, same size."""
        return self.storages.get(key) == nbytes


@dataclasses.dataclass(frozen=True)
class Residency:
    """What ``model`` holds on ``device`` after a load (module docstring),
    by parameter and buffer name (``named_parameters`` then
    ``named_buffers``, a sharded parameter by its local shard)."""

    device: str
    elements_resident: Mapping[str, int]
    itemsize_resident: Mapping[str, int]
    bytes_resident: Mapping[str, int]
    shared: tuple[tuple[str, ...], ...]
    device_bytes_allocated: int | None
    device_bytes_reserved: int | None
    bytes_unowned: int | None
    #: Per parameter the dtype it is held in (``bfloat16``) — against the
    #: report's ``dtype_on_disk`` this names a converting load (§5.3).
    dtype_resident: Mapping[str, str] = dataclasses.field(default_factory=dict)

    @classmethod
    def of(
        cls,
        model: torch.nn.Module,
        device: torch.device,
        *,
        since: Census | None = None,
    ) -> "Residency":
        """Measure ``model`` on ``device``. ``since``, a [`Census`][] of
        the device from before the load, adds the walk over the process's
        live tensors: [`bytes_unowned`][] is what appeared since it that
        no parameter owns; without one it is ``None``.

        Raises:
            ValueError: ``since`` is a census of another device type.
        """
        if since is not None and torch.device(since.device).type != device.type:
            raise ValueError(
                f"a census of {since.device} is no baseline for a load on {device}"
            )
        elements: dict[str, int] = {}
        itemsize: dict[str, int] = {}
        nbytes: dict[str, int] = {}
        dtypes: dict[str, str] = {}
        owners: dict[tuple[str, int], list[str]] = {}
        # tied parameters are two names on one storage: keep both
        named = list(model.named_parameters(remove_duplicate=False)) + list(
            model.named_buffers(remove_duplicate=False)
        )
        for name, tensor in named:
            local = _local(tensor)
            if local.device.type == "meta" or local.layout != torch.strided:
                continue
            elements[name] = local.numel()
            itemsize[name] = local.element_size()
            nbytes[name] = local.untyped_storage().nbytes()
            dtypes[name] = str(local.dtype).removeprefix("torch.")
            owners.setdefault(_storage_key(local), []).append(name)
        shared = tuple(tuple(names) for names in owners.values() if len(names) > 1)
        allocated = reserved = None
        if device.type == "cuda":
            allocated = torch.cuda.memory_allocated(device)
            reserved = torch.cuda.memory_reserved(device)
        unowned = (
            _unowned_bytes(device, set(owners), since=since)
            if since is not None
            else None
        )
        return cls(
            device=str(device),
            elements_resident=elements,
            itemsize_resident=itemsize,
            bytes_resident=nbytes,
            shared=shared,
            device_bytes_allocated=allocated,
            device_bytes_reserved=reserved,
            bytes_unowned=unowned,
            dtype_resident=dtypes,
        )

    @property
    def bytes_total(self) -> int:
        """Bytes of the distinct storages the parameters and buffers hold —
        a shared storage counted once."""
        first_of = {name: group[0] for group in self.shared for name in group}
        counted: set[str] = set()
        total = 0
        for name, size in self.bytes_resident.items():
            owner = first_of.get(name, name)
            if owner in counted:
                continue
            counted.add(owner)
            total += size
        return total

    def record(self) -> dict[str, Any]:
        """The JSON block the load report carries ([`residency_problems`][]
        reads it back)."""
        return {
            "device": self.device,
            "elements_resident": dict(self.elements_resident),
            "itemsize_resident": dict(self.itemsize_resident),
            "bytes_resident": dict(self.bytes_resident),
            "bytes_total": self.bytes_total,
            "shared": [list(group) for group in self.shared],
            "device_bytes_allocated": self.device_bytes_allocated,
            "device_bytes_reserved": self.device_bytes_reserved,
            "bytes_unowned": self.bytes_unowned,
            "dtype_resident": dict(self.dtype_resident),
        }


def _same_device(tensor: torch.Tensor, device: torch.device) -> bool:
    if tensor.device.type != device.type:
        return False
    if device.index is None or tensor.device.index is None:
        return True
    return tensor.device.index == device.index


def _live_storages(device: torch.device) -> dict[StorageKey, int]:
    """The strided storages of the process's live tensors on ``device``,
    each once with its bytes (the walk a [`Census`][] takes)."""
    gc.collect()
    seen: dict[StorageKey, int] = {}
    for obj in gc.get_objects():
        # by type, not isinstance: the latter reads ``__class__``, which a
        # deprecated module object (torch.distributed.reduce_op) answers
        # with a warning
        if not issubclass(type(obj), torch.Tensor):
            continue
        local = _local(obj)
        if not _same_device(local, device) or local.layout != torch.strided:
            continue
        seen.setdefault(_storage_key(local), local.untyped_storage().nbytes())
    return seen


def _unowned_bytes(
    device: torch.device, owned: set[StorageKey], *, since: Census | None
) -> int:
    """Bytes of the live tensor storages on ``device`` outside ``owned``
    and not alive at ``since`` (every live storage when ``None``), each
    storage once."""
    return sum(
        nbytes
        for key, nbytes in _live_storages(device).items()
        if key not in owned and not (since is not None and since.holds(key, nbytes))
    )


def conversion_note(record: Mapping[str, Any]) -> str:
    """What a rank's load report says it converted (§5.3 "the load's peak
    under a dtype conversion"): ``"; 464 parameters read as fp32, held as
    bf16 — a conversion's staging belongs on the host, never in the
    device's pool"`` when the report's ``dtype_on_disk`` and the residency's
    ``dtype_resident`` disagree on any parameter, ``""`` else (a report
    without either block converts nothing it can name). The device clauses
    of [`residency_problems`][] carry it, so a refusal of a converting
    load names the conversion."""
    on_disk: Mapping[str, str] = record.get("dtype_on_disk") or {}
    resident: Mapping[str, str] = record.get("dtype_resident") or {}
    counts: dict[tuple[str, str], int] = {}
    for name, stored in on_disk.items():
        held = resident.get(name)
        words = [DISK_WORDS.get(word) for word in stored.split("/")]
        # a tensor stored outside the float dtypes (an integer buffer) is
        # never cast — transformers keeps its own dtype — so it is no
        # conversion whatever it is held as
        if held is None or any(word is None for word in words):
            continue
        read_as = "/".join(word for word in words if word is not None)
        held_as = _TORCH_WORDS.get(held, held)
        if read_as != held_as:
            counts[(read_as, held_as)] = counts.get((read_as, held_as), 0) + 1
    if not counts:
        return ""
    named = ", ".join(
        f"{count} parameter{'s' if count != 1 else ''} read as {read_as}, held as {held_as}"
        for (read_as, held_as), count in sorted(counts.items())
    )
    return (
        f"; {named} — a conversion's staging belongs on the host, never in the "
        "device's pool"
    )


def residency_problems(record: Mapping[str, Any]) -> list[str]:
    """How a rank's load report falls short of one copy (module docstring):
    the report's ``elements_requested`` (the plan) against its residency
    block, and the device counters against the parameters — a device clause
    of a converting load naming the conversion ([`conversion_note`][])."""
    problems: list[str] = []
    note = conversion_note(record)
    requested: Mapping[str, int] = record["elements_requested"]
    elements: Mapping[str, int] = record["elements_resident"]
    itemsize: Mapping[str, int] = record["itemsize_resident"]
    nbytes: Mapping[str, int] = record["bytes_resident"]
    # a storage several parameters share is one copy, judged once as a group
    # below — never member by member, where every view of it would read as a
    # leak. Two shapes are one copy: names of one tensor (a tied head, every
    # member owning the whole storage) and disjoint views of one allocation
    # (a fused projection split into parameters, the members' bytes summing
    # to the storage); anything else is a storage larger than what it backs
    groups: list[tuple[str, ...]] = [tuple(g) for g in record.get("shared", ())]
    grouped = {name for group in groups for name in group}
    for name, want in requested.items():
        have = elements.get(name)
        if have is None:
            problems.append(f"{name}: read but not resident")
            continue
        if have != want:
            problems.append(f"{name}: {have} elements resident, {want} read")
        own = have * itemsize[name]
        if name not in grouped and nbytes[name] != own:
            problems.append(
                f"{name}: backed by a storage of {nbytes[name]} bytes, not its own "
                f"{own} — a view of a larger allocation"
            )
    for group in groups:
        members = [name for name in group if name in elements]
        if not members:
            continue
        storage = nbytes[members[0]]
        own = [elements[name] * itemsize[name] for name in members]
        tied = len(set(own)) == 1 and own[0] == storage
        if not tied and sum(own) != storage:
            problems.append(
                f"{' / '.join(group)}: one storage of {storage} bytes backs "
                f"{sum(own)} bytes of parameters — a view of a larger allocation"
            )
    total = record["bytes_total"]
    allocated = record.get("device_bytes_allocated")
    reserved = record.get("device_bytes_reserved")
    if allocated is not None and allocated > total + ALLOCATED_SLACK:
        problems.append(
            f"the device holds {allocated} bytes against {total} of parameters "
            f"(slack {ALLOCATED_SLACK}): a copy lives outside the model{note}"
        )
    if allocated is not None and reserved is not None:
        if reserved > allocated + RESERVED_SLACK:
            problems.append(
                f"the allocator reserves {reserved} bytes against {allocated} held "
                f"(slack {RESERVED_SLACK}): a segment the weights did not reuse{note}"
            )
    unowned = record.get("bytes_unowned")
    if unowned is not None and unowned > UNOWNED_SLACK:
        problems.append(
            f"{unowned} bytes of live tensors on {record['device']} that no "
            f"parameter owns (slack {UNOWNED_SLACK})"
        )
    return problems
