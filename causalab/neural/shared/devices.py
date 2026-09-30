"""The device map: where a loaded model's embedding, blocks and head live.

[`DeviceMap`][] records the device of the embedding, every decoder block,
and the final norm + head (``docs/model_parallelism.md`` §5.1) — the devices
of *one process*, so that
the layers of a model too large for one accelerator can be spread over
several without any process parallelism. It is distinct from a tensor's
``Placement`` (§4): a placement says how one tensor is sharded across ranks;
the map says which device each layer runs on.

The user's word is one device — ``cpu``, ``cuda``, ``cuda:1``, ``mps`` — or a
comma list — ``cuda:0,cuda:1``. A list splits the blocks into contiguous even
ranges in device order, the remainder going to the last device; the embedding
sits on the first device, the final norm and head on the last. Refused by
name: an empty list or entry, a repeated device, more devices than blocks,
and a list mixing CUDA with any other device type — the DeltaNet kernel path
is bound once per model (``shared/kernels.py``), for CUDA or not, so a tower
straddling the two has no one path.

Spellings are normalised to values: a bare accelerator (``cuda``, ``mps``)
carries the ordinal its tensors report (``mps:0`` for ``mps``; the current
device for ``cuda``), so a map parsed from the user's string and a map
derived from a placed model's parameters compare
equal. The string the user gave is kept as [`DeviceMap.requested`][], for
the record and the loader's cache key, outside equality.

This module lives in the shared layer because the shared services read it
(``services.check_caller_bundle``, the kernel path) and the nnsight bundle
exposes its one device through the same type; the shared layer imports no
engine.
"""

from __future__ import annotations

import dataclasses
from typing import Sequence

import torch
from torch.distributed.tensor import DTensor

from causalab.protocol.rules.errors import ProtocolError
from causalab.protocol.registry import TreeAddress

__all__ = ["DeviceMap", "module_device", "normalize_device"]

#: Device types whose tensors report an ordinal. A bare spelling of one of
#: these is given the ordinal its tensors would carry, so ``mps`` and the
#: ``mps:0`` a parameter reports are one value. ``cpu`` and ``meta`` tensors
#: report no ordinal and are left as spelled.
_ORDINAL_TYPES = frozenset({"cuda", "mps", "xpu", "hpu", "npu", "mtia"})


def normalize_device(text: str) -> torch.device:
    """One device as a value: ``torch.device(text)`` with the ordinal a
    tensor on it would report — the current CUDA device for a bare ``cuda``
    (ordinal 0 when no CUDA runtime is present, which is where torch would
    also place it), ordinal 0 for any other bare accelerator. Refuses a
    string torch does not know as a device, by name."""
    spelled = text.strip()
    if not spelled:
        raise ProtocolError("P4", "device: an empty entry names no device")
    try:
        device = torch.device(spelled)
    except (RuntimeError, TypeError, ValueError) as err:
        raise ProtocolError(
            "P4", f"device {spelled!r} is not a torch device: {err}"
        ) from err
    # a bare spelling has no ordinal (torch types ``index`` as int; it is
    # ``None`` at runtime for ``torch.device("cuda")``)
    index = getattr(device, "index", None)
    if index is not None or device.type not in _ORDINAL_TYPES:
        return device
    if device.type == "cuda" and torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device(device.type, 0)


def _device_of(tensor: torch.Tensor) -> torch.device:
    """A tensor's device — a sharded parameter's is where its local shard
    is (a DTensor reports the mesh's device type; its shard is the fact)."""
    if isinstance(tensor, DTensor):
        return tensor.to_local().device
    return tensor.device


def module_device(
    module: torch.nn.Module, *, what: str, empty: torch.device | None = None
) -> torch.device:
    """The one device a module's parameters and buffers are on. A module
    whose tensors straddle devices, one with a tensor on ``meta`` (offloaded,
    or never materialised) and one with no tensors at all are refused by
    name: none of them has a device to run on — except that ``empty``, when
    given, is the device a module with no tensors is recorded on (a pipeline
    stage's identity layers run on the stage's device)."""
    found = {_device_of(t) for t in module.parameters()} | {
        _device_of(t) for t in module.buffers()
    }
    if not found:
        if empty is not None:
            return empty
        raise ProtocolError(
            "P4", f"{what} has no parameter or buffer, so it is on no device"
        )
    if len(found) > 1:
        raise ProtocolError(
            "P4",
            f"{what} straddles devices {sorted(map(str, found))}; every block, "
            "the embedding and the head must each sit on one device",
        )
    (device,) = found
    if device.type == "meta":
        raise ProtocolError(
            "P4",
            f"{what} is on the meta device (offloaded, or never materialised); "
            "a placed model holds every parameter on a real device",
        )
    return device


@dataclasses.dataclass(frozen=True)
class DeviceMap:
    """The device of the embedding, of each decoder block and of the final
    norm + head (module docstring). ``requested`` is the spelling the user
    gave — or, for a map derived from a model, its canonical
    [`spelling`][] — kept for the record and the loader's cache key, and
    deliberately outside equality: ``cuda`` and ``cuda:0`` are one placement."""

    embedding: torch.device
    blocks: tuple[torch.device, ...]
    head: torch.device
    requested: str = dataclasses.field(compare=False, hash=False)

    def __post_init__(self) -> None:
        if not self.blocks:
            raise ProtocolError("P4", "device map: a tower with no blocks")
        types = {d.type for d in (self.embedding, *self.blocks, self.head)}
        if "cuda" in types and len(types) > 1:
            others = sorted(types - {"cuda"})
            raise ProtocolError(
                "P4",
                f"device map mixes CUDA with {others}: the DeltaNet kernel path is "
                "bound once per model, for CUDA or not (shared/kernels.py), so a "
                "tower straddling the two has no one path. Place every layer on "
                "CUDA devices, or on none.",
            )

    # ------------------------------------------------------------------ #
    # construction
    # ------------------------------------------------------------------ #

    @classmethod
    def parse(cls, text: str, num_layers: int) -> "DeviceMap":
        """The user's ``--device`` word over a tower of ``num_layers`` blocks
        (module docstring): one device everywhere, or a comma list split
        into contiguous even block ranges in order, the remainder to the
        last device, embedding first and head last."""
        if num_layers < 1:
            raise ProtocolError(
                "P4", f"device map: a tower of {num_layers} blocks has nothing to place"
            )
        if not text.strip():
            raise ProtocolError("P4", "device: an empty string names no device")
        devices: list[torch.device] = []
        for entry in text.split(","):
            device = normalize_device(entry)
            if device in devices:
                raise ProtocolError(
                    "P4",
                    f"device {text!r} repeats {device}: each device of a list "
                    "takes one contiguous range of blocks, so a repeat has no "
                    "meaning",
                )
            devices.append(device)
        if len(devices) > num_layers:
            raise ProtocolError(
                "P4",
                f"device {text!r} names {len(devices)} devices for a tower of "
                f"{num_layers} block(s); every device must hold at least one block",
            )
        base, remainder = divmod(num_layers, len(devices))
        blocks: list[torch.device] = []
        for i, device in enumerate(devices):
            count = base + (remainder if i == len(devices) - 1 else 0)
            blocks.extend([device] * count)
        return cls(
            embedding=devices[0],
            blocks=tuple(blocks),
            head=devices[-1],
            requested=text,
        )

    @classmethod
    def of_modules(
        cls,
        embedding: torch.nn.Module,
        blocks: Sequence[torch.nn.Module],
        head: torch.nn.Module,
        *,
        empty: torch.device | None = None,
    ) -> "DeviceMap":
        """The map a placed model actually realizes, read off its parameters
        — what [`ModelBundle.from_model`][causalab.neural.engines.pytorch_hooks.loading.ModelBundle.from_model] derives for a caller-owned
        model, and what the loader records for what it placed. A block whose
        parameters straddle devices is refused by index
        ([`module_device`][]); a sharded parameter counts where its local
        shard is. ``empty`` is the device a module with no tensors is
        recorded on — a pipeline stage's identity layers — refused by name
        when not given. ``requested`` is the canonical spelling."""
        placed = cls(
            embedding=module_device(embedding, what="the embedding", empty=empty),
            blocks=tuple(
                module_device(block, what=f"block {i}", empty=empty)
                for i, block in enumerate(blocks)
            ),
            head=module_device(head, what="the head", empty=empty),
            requested="",
        )
        return dataclasses.replace(placed, requested=placed.spelling)

    # ------------------------------------------------------------------ #
    # reading the map
    # ------------------------------------------------------------------ #

    def device_of(self, layer: int) -> torch.device:
        """The device block ``layer`` runs on. Total over ``range(len(blocks))``;
        anything else is a caller error, never a wrapped index."""
        if not 0 <= layer < len(self.blocks):
            raise IndexError(
                f"layer {layer} is outside the tower of {len(self.blocks)} block(s)"
            )
        return self.blocks[layer]

    @property
    def single(self) -> torch.device | None:
        """The one device when every entry agrees, else ``None`` — the
        question CUDA graphs and the kernel binding ask."""
        devices = self.devices
        if len(devices) == 1:
            (device,) = devices
            return device
        return None

    @property
    def devices(self) -> frozenset[torch.device]:
        return frozenset((self.embedding, *self.blocks, self.head))

    @property
    def is_cuda(self) -> bool:
        """Whether the model runs on CUDA — one answer for the whole tower,
        because mixing was refused at construction."""
        return self.head.type == "cuda"

    @property
    def spelling(self) -> str:
        """The canonical comma list: the distinct devices in placement order
        (embedding, blocks, head), each with its ordinal. ``parse`` of it
        gives back an equal map."""
        return ",".join(str(d) for d in self._ordered())

    def _ordered(self) -> list[torch.device]:
        ordered: list[torch.device] = []
        for device in (self.embedding, *self.blocks, self.head):
            if device not in ordered:
                ordered.append(device)
        return ordered

    # ------------------------------------------------------------------ #
    # the map over a module tree
    # ------------------------------------------------------------------ #

    def module_map(self, tree: TreeAddress) -> dict[str, torch.device]:
        """The transformers ``device_map``: one entry per placed module
        prefix — the embedding, each block, the final norm, the head — and
        ``""`` for everything else (rotary tables, ties), which rides with
        the embedding.

        The values are ``torch.device`` objects, deliberately: transformers
        reads them wherever it places a parameter, while accelerate's
        ``dispatch_model`` (which transformers runs over any map with more
        than one device) recognises *offload* by the strings ``"cpu"`` and
        ``"disk"`` alone — under a spelled ``"cpu"`` it moves those weights to
        ``meta`` and executes the block on the accelerator, which is not
        placement. ``torch.device("cpu")`` is placement, so its hooks are
        plain moves, which the loader strips (``weights.py``)."""
        placed = {"": self.embedding, tree.embedding: self.embedding}
        for i, device in enumerate(self.blocks):
            placed[f"{tree.blocks}.{i}"] = device
        placed[tree.final_norm] = self.head
        placed[tree.lm_head] = self.head
        return placed

    def device_for(self, name: str, tree: TreeAddress) -> torch.device:
        """The device parameter ``name`` (a model state-dict key) is placed
        on: the longest entry of [`module_map`][] that is a dotted prefix
        of it — ``model.layers.1`` never swallows ``model.layers.10``."""
        placed = self.module_map(tree)
        best: str | None = None
        for prefix in placed:
            if prefix == "" or name == prefix or name.startswith(prefix + "."):
                if best is None or len(prefix) > len(best):
                    best = prefix
        assert best is not None  # "" matches everything
        return placed[best]
