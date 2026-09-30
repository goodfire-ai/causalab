"""Apply the protocol's mechanisms in feature space.

At an address, the absolute write applies first and additive deltas sum.
The executor handles dimension selection and the pre-write error term.
A Gaussian write draws ``(batch, positions, width)`` once per application
from a locally seeded Torch generator outside the model.
"""

from __future__ import annotations

import importlib
from typing import Any, Callable, Mapping, Sequence

import torch

from causalab.protocol.rules.errors import ProtocolError
from causalab.protocol.schema import ADDITIVE_MECHANISMS, CodeSpec, Do

__all__ = ["apply_absolute", "apply_delta", "is_additive", "row_role_bounds"]

#: Resolve an operand to a tensor/scalar: the executor passes a lookup over
#: read values, params, and dotted featurizer slots.
OperandLookup = Callable[[Any], torch.Tensor | float]


def is_additive(do: Do) -> bool:
    return str(do.mechanism) in ADDITIVE_MECHANISMS


def _operand(lookup: OperandLookup, value: Any) -> torch.Tensor | float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return lookup(value)


def apply_absolute(
    do: Do,
    f: torch.Tensor,
    lookup: OperandLookup,
    *,
    code: Mapping[str, CodeSpec] | None = None,
) -> torch.Tensor:
    """The absolute-class write ``f ← …`` for one mechanism; ``f`` is the
    pre-write feature slice (already dims-selected).

    ``code`` is the document's ``code`` table, which only ``pytorch_fn``
    reads (§2.8.1)."""
    mech = str(do.mechanism)
    payload = do.payload
    if mech == "swap":
        return _coerce(_operand(lookup, payload), f)
    if mech == "lerp":
        alpha = _scalar(_operand(lookup, payload["alpha"]))
        op = _coerce(_operand(lookup, payload["op"]), f)
        return (1.0 - alpha) * f + alpha * op
    if mech == "affine":
        a = _operand(lookup, payload["A"])
        b = _operand(lookup, payload["b"])
        if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
            raise ProtocolError("P2", "affine A/b must resolve to tensors")
        return f @ a.T.to(f.dtype) + b.to(f.dtype)
    if mech == "renormalize":
        raise AssertionError(
            "renormalize is applied by the executor after additive deltas "
            "(apply_renormalize) — under strict absolute-first order it would "
            "always be the identity; surfaced as a spec question"
        )
    if mech == "clamp":
        lo = _scalar(_operand(lookup, payload["lo"]))
        hi = _scalar(_operand(lookup, payload["hi"]))
        return f.clamp(min=lo, max=hi)
    if mech == "pytorch_fn":
        return _apply_pytorch_fn(payload, f, code)
    raise ProtocolError("P4", f"{mech!r} is not an absolute mechanism")


def _apply_pytorch_fn(
    payload: Any, f: torch.Tensor, code: Mapping[str, CodeSpec] | None
) -> torch.Tensor:
    """Call a declared local function (§2.8.1).

    ``args`` are the declaration's typed JSON keywords, and ``row_roles``
    becomes the ``role -> (start, stop)`` map that replaces inferring roles
    from physical batch positions — passed only when the declaration carries
    roles, so a plain one-tensor function is still called ``fn(f)``. The
    loader has already checked the signature agrees (§5 rule 24).

    ``data_inputs`` and ``env_inputs`` are *not* passed: they are identity and
    an allowlist, not a delivery channel (§2.8.1). The function opens its own
    declared path, and the load refuses one it did not declare.
    """
    name = str(payload["code"])
    spec = (code or {}).get(name)
    if spec is None:
        raise ProtocolError(
            "P2",
            f"pytorch_fn names code declaration {name!r}, which this document "
            "does not carry — the executor was handed a document the loader "
            "did not validate (§2.8.1)",
        )
    locator = str(spec.locator)
    module_name, _, attr = locator.rpartition(".")
    fn = getattr(importlib.import_module(module_name), attr)
    kwargs: dict[str, Any] = dict(spec.args)
    if spec.row_roles:
        kwargs["row_roles"] = row_role_bounds(spec)
    return fn(f, **kwargs)


def row_role_bounds(spec: CodeSpec) -> dict[str, tuple[int, int]]:
    """``role -> (start, stop)`` half-open row bounds, in declared order."""
    bounds: dict[str, tuple[int, int]] = {}
    start = 0
    for role in spec.row_roles:
        bounds[role.role] = (start, start + role.rows)
        start += role.rows
    return bounds


def apply_delta(
    do: Do,
    f_pre: torch.Tensor,
    lookup: OperandLookup,
    *,
    batch: int,
    n_pos: int,
    rows: slice | Sequence[int] | None = None,
) -> torch.Tensor:
    """The additive-class delta for one mechanism (summed by the caller).

    ``batch`` is the row count of the **whole** batch the write addresses and
    ``rows`` the slice of it ``f_pre`` holds, when a forward covers only a
    window of the rows (§8, execution scale) — or the row indices, when a
    ragged write lands one width bucket of the window at a time (§5 rule 19,
    ``exact_length_buckets``). A ``gaussian`` draw is made over all ``batch``
    rows and sliced or indexed, so the noise a row receives is the same in
    one forward as in several, and the same whichever rows share its bucket —
    the RNG contract the parity goldens pin.
    """
    mech = str(do.mechanism)
    payload = do.payload
    if mech == "add_scaled":
        alpha = _scalar(_operand(lookup, payload["alpha"]))
        op = _coerce(_operand(lookup, payload["op"]), f_pre)
        return alpha * op
    if mech == "gaussian":
        seed = int(payload["seed"])
        scale = float(payload["scale"])
        generator = torch.Generator().manual_seed(seed)
        draw = torch.randn(
            (batch, n_pos, f_pre.shape[-1]), generator=generator, dtype=torch.float32
        )
        if rows is not None:
            draw = draw[rows] if isinstance(rows, slice) else draw[list(rows)]
        return scale * draw.to(dtype=f_pre.dtype, device=f_pre.device).reshape(
            f_pre.shape
        )
    raise ProtocolError("P4", f"{mech!r} is not an additive mechanism")


def apply_renormalize(f: torch.Tensor, f_pre: torch.Tensor) -> torch.Tensor:
    """``f ← f·‖f₀‖/‖f‖`` with ``f₀`` the pre-write feature value. Runs after
    the additive deltas (the only order under which it is not the identity);
    it still counts as the address's one absolute write for rule 8."""
    target = f_pre.norm(dim=-1, keepdim=True)
    return f * (target / f.norm(dim=-1, keepdim=True).clamp_min(1e-12))


def _scalar(value: torch.Tensor | float) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.item())
    return float(value)


def _coerce(value: torch.Tensor | float, like: torch.Tensor) -> torch.Tensor:
    """Move an operand onto the written slice's device/dtype and check the
    one-way broadcast (right-aligned; the classic width mismatch — a counterfactual
    read contributing a different number of positions than the write
    addresses — must fail legibly here, not as a scatter assert)."""
    if not isinstance(value, torch.Tensor):
        return torch.full_like(like, float(value))
    value = value.to(device=like.device, dtype=like.dtype)
    ok = value.dim() <= like.dim() and all(
        v == 1 or v == t for v, t in zip(reversed(value.shape), reversed(like.shape))
    )
    if not ok:
        raise ProtocolError(
            "P2",
            f"operand of shape {tuple(value.shape)} does not broadcast to the "
            f"{tuple(like.shape)} slice it writes — position widths must "
            "pair up per example, or the operand must be a broadcastable vector",
        )
    return value


def operand_names(payload: Any) -> tuple[str, ...]:
    """Names referenced by a mechanism payload (mirrors the validator)."""
    if isinstance(payload, str):
        return (payload,)
    if isinstance(payload, Mapping):
        return tuple(v for v in payload.values() if isinstance(v, str))
    return ()
