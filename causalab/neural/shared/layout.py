"""Convert declared native shapes to the executor's tensor layout.

The contract layout is ``(batch, position, feature)``. ``to_contract``
unpacks declared axes, selects a fused projection part, orders the axes,
and flattens inner dimensions. Rank and width checks reject tensors that
contradict the site's FeatureShape. Shapes without a contract form retain
their native layout.

``from_contract`` returns the written tensor to the model. Conversions may
copy, so callers must use that return value. A fused projection writes its
selected part back into the supplied native tensor. Tuple taps choose their
tensor with ``tuple_index``, which defaults to zero.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from causalab.protocol.registry.shapes import INNER_KINDS, Axis, FeatureShape

__all__ = [
    "LayoutError",
    "from_contract",
    "rebuild_payload",
    "tap_tensor",
    "to_contract",
]


class LayoutError(ValueError):
    """A tensor does not have the shape its tap's descriptor claims.

    Deliberately *not* a [`ProtocolError`][causalab.protocol.rules.errors.ProtocolError], and
    so it carries no ``P``/``V`` code: the shape is a field the site table sets
    on [`ResolvedSite`][causalab.neural.shared.sites.ResolvedSite], never
    something a document author writes, so this can only fire on a mismatch
    between our table and a model's real module — an internal invariant, and
    the protocol codes exist to name rules a *document* broke.

    What would change that: making the shape authorable (a per-family override
    in a document), at which point a bad entry becomes an author error and this
    should either derive from ``ProtocolError`` or be wrapped at the executor
    boundary with a code.
    """


def _unpack(tensor: torch.Tensor, shape: FeatureShape, batch_size: int) -> torch.Tensor:
    """Step 1: one tensor dimension per declared axis, in declared order."""
    groups = shape.native_groups
    if tensor.dim() != len(groups):
        raise LayoutError(
            f"shape {shape.describe()} expects a {len(groups)}-D tensor, got "
            f"shape {tuple(tensor.shape)}"
        )
    sizes: list[int] = []
    for dim, group in enumerate(groups):
        extent = tensor.shape[dim]
        if len(group) == 1:
            axis = group[0]
            if axis.width is not None and extent != axis.width:
                raise LayoutError(
                    f"shape {shape.describe()} declares {axis.label} of width "
                    f"{axis.width}, but dim {dim} of {tuple(tensor.shape)} is "
                    f"{extent}"
                )
            sizes.append(extent)
        elif group[0].kind == "batch":
            if batch_size <= 0 or extent % batch_size:
                raise LayoutError(
                    f"shape {shape.describe()} cannot split {extent} rows into "
                    f"batch_size {batch_size}: the tap is flattened over "
                    "(batch, position), so the row count must be a multiple of "
                    "the batch size"
                )
            sizes.extend([batch_size, extent // batch_size])
        else:
            widths = [a.width or 1 for a in group]
            if math.prod(widths) != extent:
                raise LayoutError(
                    f"shape {shape.describe()} packs "
                    f"{'·'.join(str(w) for w in widths)} = {math.prod(widths)} "
                    f"into dim {dim}, but {tuple(tensor.shape)} has {extent} there"
                )
            sizes.extend(widths)
    return tensor.reshape(*sizes)


def _contract_order(axes: list[Axis]) -> list[int]:
    """Step 3's permutation: batch, position, then the inner axes in order."""
    order = [i for i, a in enumerate(axes) if a.kind == "batch"]
    order += [i for i, a in enumerate(axes) if a.kind == "position"]
    order += [i for i, a in enumerate(axes) if a.kind in INNER_KINDS]
    return order


def to_contract(
    tensor: torch.Tensor, shape: FeatureShape, *, batch_size: int
) -> torch.Tensor:
    """Reshape a tap's native tensor into ``(batch, position, feature)``.

    Args:
        tensor: the tensor as the module produced or consumed it.
        shape: the tap's declared native shape.
        batch_size: rows in the encoded batch. Needed to un-flatten a
            ``flat_batch`` shape, where the position count is only recoverable
            as ``tensor.shape[0] // batch_size``.

    Returns:
        A ``(batch, position, feature)`` tensor — a view where the conversion
        permits one (see the module docstring on aliasing). A shape with no
        contract form is returned untouched.

    Raises:
        LayoutError: the tensor's rank or a static width contradicts the
            declaration.
    """
    if not shape.has_contract_form:
        return tensor
    if shape.state_axes:
        # a state matrix crosses the executor in its native layout — there is
        # no feature vector to flatten to — but its rank and static widths are
        # still checked, so a wrong tap raises rather than being reinterpreted
        return _unpack(tensor, shape, batch_size)
    unpacked = _unpack(tensor, shape, batch_size)
    axes = list(shape.axes)
    if shape.fused_index is not None:
        which = next(i for i, a in enumerate(axes) if a.kind == "fused")
        unpacked = unpacked.select(which, shape.fused_index)
        axes.pop(which)
    order = _contract_order(axes)
    moved = unpacked.permute(*order)
    n_inner = len(order) - 2
    if n_inner == 0:
        # no feature axis at all (`input_ids`): give the contract a width-1 one.
        # unsqueeze, not reshape, so the view survives.
        return moved.unsqueeze(-1)
    if n_inner == 1:
        return moved
    return moved.reshape(moved.shape[0], moved.shape[1], -1)


def from_contract(
    tensor: torch.Tensor,
    shape: FeatureShape,
    *,
    batch_size: int,
    native: torch.Tensor | None = None,
) -> torch.Tensor:
    """Invert [`to_contract`][], returning the module's native shape.

    ``from_contract(to_contract(x, s), s)`` round-trips to ``x``'s shape for
    every shape. Used on the write path, where the model must receive back the
    shape it was going to produce.

    Args:
        native: the tensor [`to_contract`][] was given. Required only for a
            **fused** shape, where the component named one split of a shared
            projection and the other splits have to survive the write: the
            contract tensor is scattered back into ``native``, which is then
            returned. Ignored otherwise.
    """
    if not shape.has_contract_form:
        return tensor
    if shape.state_axes:
        return tensor
    axes = [a for a in shape.axes if a.kind != "fused"]
    order = _contract_order(axes)
    n_inner = len(order) - 2
    if n_inner == 0:
        if tensor.dim() != 3 or tensor.shape[-1] != 1:
            raise LayoutError(
                f"shape {shape.describe()} has no feature axis, so it expects a "
                "3-D contract tensor of width 1 to drop back, got shape "
                f"{tuple(tensor.shape)}"
            )
        moved = tensor.squeeze(-1)
    else:
        inner = [a.width or 1 for a in shape.feature_axes]
        if tensor.dim() != 3 or tensor.shape[-1] != math.prod(inner):
            raise LayoutError(
                f"shape {shape.describe()} expects a 3-D contract tensor of "
                f"width {math.prod(inner)}, got shape {tuple(tensor.shape)}"
            )
        moved = (
            tensor
            if n_inner == 1
            else tensor.reshape(tensor.shape[0], tensor.shape[1], *inner)
        )
    inverse = [0] * len(order)
    for contract_dim, axis_dim in enumerate(order):
        inverse[axis_dim] = contract_dim
    unpacked = moved.permute(*inverse)
    if shape.fused_index is not None:
        if native is None:
            raise LayoutError(
                f"shape {shape.describe()} names split {shape.fused_index} of a "
                "fused projection, so writing it back needs the native tensor "
                "the other splits live in — pass native=..."
            )
        whole = _unpack(native, shape, batch_size)
        which = next(i for i, a in enumerate(shape.axes) if a.kind == "fused")
        whole.select(which, shape.fused_index).copy_(unpacked)
        return native
    return _pack(unpacked, shape, batch_size)


def _pack(tensor: torch.Tensor, shape: FeatureShape, batch_size: int) -> torch.Tensor:
    """Invert `_unpack`: re-merge the axes each native dimension holds."""
    sizes: list[int] = []
    dim = 0
    for group in shape.native_groups:
        sizes.append(math.prod(tensor.shape[dim + k] for k in range(len(group))))
        dim += len(group)
    return tensor.reshape(*sizes)


def tap_tensor(payload: Any, tuple_index: int | None) -> torch.Tensor:
    """The tensor a tap means, out of a module's output or input payload.

    ``tuple_index=None`` keeps the historical rule — element 0 of a tuple,
    otherwise the payload itself. An explicit index addresses a specific element
    and fails loudly if the payload is not a tuple, because a tap that asked for
    element 2 and silently got the whole tensor would read the wrong thing.
    """
    if tuple_index is None:
        return payload[0] if isinstance(payload, tuple) else payload
    if not isinstance(payload, tuple):
        raise LayoutError(
            f"tap declares tuple_index={tuple_index} but the module payload is "
            f"{type(payload).__name__}, not a tuple"
        )
    if not -len(payload) <= tuple_index < len(payload):
        raise LayoutError(
            f"tap declares tuple_index={tuple_index} but the module returned a "
            f"{len(payload)}-tuple"
        )
    return payload[tuple_index]


def rebuild_payload(payload: Any, tuple_index: int | None, value: torch.Tensor) -> Any:
    """Put ``value`` back where [`tap_tensor`][] took it from.

    Preserves the rest of a tuple payload so a write does not drop the cache or
    the attention weights a module also returned.
    """
    if not isinstance(payload, tuple):
        return value
    index = 0 if tuple_index is None else tuple_index
    items = list(payload)
    items[index] = value
    return tuple(items)
