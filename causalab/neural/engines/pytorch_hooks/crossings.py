"""The residual stream's crossings between layers placed on different devices.

A [`DeviceMap`][] over several devices
puts block ``k`` on one device and block ``k + 1`` on another; the decoder
loop hands the residual from one to the next as is, so the second block
would see an input on the wrong device. [`place_crossings`][] installs one
forward pre-hook per block, and one on the final norm, that moves every
tensor among the block's arguments — the residual, the causal mask (or the
per-stream mask table of a hybrid tower), the rotary tables, the position
ids — onto that module's device. A tensor already there is left untouched,
so on a single-device model nothing moves; a ``Cache`` object is not a
tensor and is left alone (each layer's entries land where that layer's
keys and values are computed).

The hooks are registered at load, before any executor hook, so a write
pre-hook on a block sees its residual already on the block's device, and a
capture stays where its hook produced it. transformers would otherwise hand
a multi-device map to accelerate's ``dispatch_model``, whose hooks treat a
``cpu`` entry as *offload* — the weights moved to ``meta`` and the block
executed on the accelerator — which is not placement; the loader strips
those and installs these (``weights.py``).
"""

from __future__ import annotations

from typing import Any, Callable

import torch
from torch.utils.hooks import RemovableHandle

from causalab.neural.shared.devices import DeviceMap
from causalab.protocol.rules.errors import ProtocolError
from causalab.protocol.registry import TreeAddress, walk

__all__ = ["place_crossings"]


def place_crossings(
    model: torch.nn.Module, devices: DeviceMap, tree: TreeAddress
) -> list[RemovableHandle]:
    """Install the crossings ``devices`` needs on ``model`` (module
    docstring): one pre-hook per block moving its inputs onto
    ``devices.device_of(layer)``, one on the final norm moving onto
    ``devices.head``. Returns the handles, for a caller that wants them off."""
    blocks = walk(model, tree.blocks)
    norm = walk(model, tree.final_norm)
    if blocks is None or norm is None:
        raise ProtocolError(
            "P4",
            f"the family's tree addresses blocks at {tree.blocks!r} and the final "
            f"norm at {tree.final_norm!r}, but this model ({type(model).__name__}) "
            "lacks one of them — nothing to place",
        )
    if len(blocks) != len(devices.blocks):
        raise ProtocolError(
            "P4",
            f"the device map places {len(devices.blocks)} block(s) but the model "
            f"has {len(blocks)}",
        )
    handles = [
        block.register_forward_pre_hook(
            _crossing(devices.device_of(layer)), with_kwargs=True
        )
        for layer, block in enumerate(blocks)
    ]
    handles.append(
        norm.register_forward_pre_hook(_crossing(devices.head), with_kwargs=True)
    )
    return handles


def _crossing(
    device: torch.device,
) -> Callable[
    [Any, tuple[Any, ...], dict[str, Any]], tuple[tuple[Any, ...], dict[str, Any]]
]:
    def hook(
        _module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        return _moved(args, device), _moved(kwargs, device)

    return hook


def _moved(value: Any, device: torch.device) -> Any:
    """``value`` with every tensor in it on ``device`` — through tuples
    (named ones kept), lists and dicts; anything else as is."""
    if isinstance(value, torch.Tensor):
        return value if value.device == device else value.to(device)
    if isinstance(value, tuple):
        moved = [_moved(v, device) for v in value]
        fields = getattr(value, "_fields", None)
        return type(value)(*moved) if fields is not None else tuple(moved)
    if isinstance(value, list):
        return [_moved(v, device) for v in value]
    if isinstance(value, dict):
        return {k: _moved(v, device) for k, v in value.items()}
    return value
