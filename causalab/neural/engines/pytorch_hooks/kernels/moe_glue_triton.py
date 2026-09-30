"""Triton implementations of the expert reference operations.

Kernels follow ``moe_glue_reference`` in floating-point order.
``enable_fp_fusion=False`` preserves separate rounding where ATen requires
it. Gate math uses libdevice exponentiation and round-to-nearest division.

``available`` reports Triton support. Grids depend on shape and launches
avoid host reads, allowing CUDA capture after compilation.
"""

from __future__ import annotations

from typing import Any

import torch

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
except Exception:  # noqa: BLE001 — no Triton, whatever the reason: no kernels
    triton = None
    tl = None
    libdevice = None

__all__ = [
    "available",
    "counting_sort",
    "epilogue_backward",
    "epilogue_forward",
    "gather_rows",
    "gather_rows_backward",
    "silu_mul_backward",
    "silu_mul_forward",
]

#: Every launch: no fused multiply-add where ATen's kernels round twice.
_LAUNCH: dict[str, Any] = {"enable_fp_fusion": False}

#: The 32 lanes × 4-vector row-sum block of ATen's vectorized reduce.
ROW_SUM_LANES = 32
ROW_SUM_VEC = 4

_SORT_BLOCK = 1024
_TOKEN_BLOCK = 512
_GATE_BLOCK = 512
#: One gather program per row, the whole row in one block up to this width.
_GATHER_MAX_BLOCK = 4096


def available() -> bool:
    return triton is not None


def _pow2_at_least(n: int) -> int:
    return 1 << max(n - 1, 0).bit_length()


if triton is not None:
    #: Largest int64: the sentinel for a padded slot in the per-token rank. A
    #: jitted body may read a module global only as a ``tl.constexpr``.
    _NEVER = tl.constexpr((1 << 63) - 1)

    @triton.jit
    def _counting_sort_kernel(
        ids_ptr, perm_ptr, inv_ptr, offsets_ptr, num_pairs, BLOCK: tl.constexpr
    ):
        """Program ``b`` places expert ``b``'s pairs: first how many pairs
        route below ``b`` (its base), then a stable pass writing each of its
        pairs at base + rank; the inclusive count is ``offsets[b]``."""
        expert = tl.program_id(0)
        offs = tl.arange(0, BLOCK)
        below = tl.zeros((BLOCK,), dtype=tl.int32)
        for start in range(0, num_pairs, BLOCK):
            idx = start + offs
            ids = tl.load(ids_ptr + idx, mask=idx < num_pairs, other=-1)
            below += ((ids >= 0) & (ids < expert)).to(tl.int32)
        cursor = tl.sum(below)
        for start in range(0, num_pairs, BLOCK):
            idx = start + offs
            ids = tl.load(ids_ptr + idx, mask=idx < num_pairs, other=-1)
            hit = ids == expert
            counts = hit.to(tl.int32)
            position = cursor + tl.cumsum(counts, axis=0) - counts
            tl.store(perm_ptr + position, idx.to(tl.int64), mask=hit)
            tl.store(inv_ptr + idx, position.to(tl.int64), mask=hit)
            cursor += tl.sum(counts)
        tl.store(offsets_ptr + expert, cursor)

    @triton.jit
    def _gather_rows_kernel(
        hidden_ptr, perm_ptr, out_ptr, width, top_k, BLOCK: tl.constexpr
    ):
        row = tl.program_id(0).to(tl.int64)
        source = tl.load(perm_ptr + row) // top_k
        cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = cols < width
        values = tl.load(hidden_ptr + source * width + cols, mask=mask)
        tl.store(out_ptr + row * width + cols, values, mask=mask)

    @triton.jit
    def _gather_rows_backward_kernel(
        grad_ptr,
        inv_ptr,
        out_ptr,
        width,
        TOP_K: tl.constexpr,
        K_PAD: tl.constexpr,
        PER_STEP: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """Token ``t``'s ``TOP_K`` sorted-array rows, folded in ascending
        array position — the order the index backward's stable sort puts
        them in — rounding after each addition or once (``PER_STEP``)."""
        token = tl.program_id(0).to(tl.int64)
        slots = tl.arange(0, K_PAD)
        positions = tl.load(
            inv_ptr + token * TOP_K + slots, mask=slots < TOP_K, other=_NEVER
        )
        rank = tl.sum((positions[None, :] < positions[:, None]).to(tl.int32), axis=1)
        cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = cols < width
        out_dtype = out_ptr.dtype.element_ty
        acc = tl.zeros((BLOCK,), dtype=tl.float32)
        for r in tl.static_range(TOP_K):
            row = tl.sum(tl.where(rank == r, positions, 0))
            grad = tl.load(grad_ptr + row * width + cols, mask=mask, other=0.0).to(
                tl.float32
            )
            acc = acc + grad
            if PER_STEP:
                acc = acc.to(out_dtype).to(tl.float32)
        tl.store(out_ptr + token * width + cols, acc.to(out_dtype), mask=mask)

    @triton.jit
    def _epilogue_forward_kernel(
        proj_ptr,
        weights_ptr,
        inv_ptr,
        out_ptr,
        width,
        TOP_K: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """Token ``t``: ``X(proj[row_s] · w_s)`` for its ``TOP_K`` slots,
        slot ``s`` into fp32 accumulator ``s % 4``, the four combined left
        to right, rounded to ``X`` then to the output dtype."""
        token = tl.program_id(0).to(tl.int64)
        cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = cols < width
        x_dtype = proj_ptr.dtype.element_ty
        acc0 = tl.zeros((BLOCK,), dtype=tl.float32)
        acc1 = tl.zeros((BLOCK,), dtype=tl.float32)
        acc2 = tl.zeros((BLOCK,), dtype=tl.float32)
        acc3 = tl.zeros((BLOCK,), dtype=tl.float32)
        for s in tl.static_range(TOP_K):
            row = tl.load(inv_ptr + token * TOP_K + s)
            weight = tl.load(weights_ptr + token * TOP_K + s).to(tl.float32)
            proj = tl.load(proj_ptr + row * width + cols, mask=mask, other=0.0).to(
                tl.float32
            )
            product = (proj * weight).to(x_dtype).to(tl.float32)
            if s % 4 == 0:
                acc0 = acc0 + product
            elif s % 4 == 1:
                acc1 = acc1 + product
            elif s % 4 == 2:
                acc2 = acc2 + product
            else:
                acc3 = acc3 + product
        total = ((acc0 + acc1) + acc2) + acc3
        tl.store(
            out_ptr + token * width + cols,
            total.to(x_dtype).to(out_ptr.dtype.element_ty),
            mask=mask,
        )

    @triton.jit
    def _epilogue_backward_kernel(
        grad_ptr,
        proj_ptr,
        weights_ptr,
        perm_ptr,
        d_proj_ptr,
        d_weights_ptr,
        width,
        top_k,
        LANES: tl.constexpr,
        VEC: tl.constexpr,
    ):
        """Sorted row ``i`` of pair ``perm[i] = t·k + s``: ``d proj[i] =
        X(g[t] · w)`` and ``d w[t·k+s] = rowsum(X(g[t] · proj[i]))`` in the
        vectorized reduce's order: ``LANES`` lanes × ``VEC``-vectors per
        step, four accumulators by vector position combined left to right,
        then the adjacent-pair shuffle tree over the lanes."""
        tl.static_assert(LANES == 32)
        tl.static_assert(VEC == 4)
        WIDTH: tl.constexpr = LANES * VEC
        row = tl.program_id(0).to(tl.int64)
        pair = tl.load(perm_ptr + row)
        token = pair // top_k
        x_dtype = proj_ptr.dtype.element_ty
        weight = tl.load(weights_ptr + pair).to(tl.float32)
        cols = tl.arange(0, WIDTH)
        acc = tl.zeros((WIDTH,), dtype=tl.float32)
        for start in range(0, width, WIDTH):
            c = start + cols
            mask = c < width
            g = tl.load(grad_ptr + token * width + c, mask=mask, other=0.0).to(
                tl.float32
            )
            g = g.to(x_dtype).to(tl.float32)
            proj = tl.load(proj_ptr + row * width + c, mask=mask, other=0.0).to(
                tl.float32
            )
            tl.store(d_proj_ptr + row * width + c, (g * weight).to(x_dtype), mask=mask)
            product = (g * proj).to(x_dtype).to(tl.float32)
            acc = tl.where(mask, acc + product, acc)
        # acc[4·lane + i]: the lane's accumulator for vector position i
        even, odd = tl.split(tl.reshape(acc, (LANES, 2, 2)))
        a0, a2 = tl.split(even)
        a1, a3 = tl.split(odd)
        lane = ((a0 + a1) + a2) + a3
        v = tl.sum(tl.reshape(lane, (16, 2)), axis=1)
        v = tl.sum(tl.reshape(v, (8, 2)), axis=1)
        v = tl.sum(tl.reshape(v, (4, 2)), axis=1)
        v = tl.sum(tl.reshape(v, (2, 2)), axis=1)
        total = tl.sum(v)
        tl.store(d_weights_ptr + pair, total.to(d_weights_ptr.dtype.element_ty))

    @triton.jit
    def _silu_mul_forward_kernel(gate_up_ptr, out_ptr, inner, BLOCK: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)
        cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = cols < inner
        x_dtype = gate_up_ptr.dtype.element_ty
        gate = tl.load(gate_up_ptr + row * (2 * inner) + cols, mask=mask, other=0.0).to(
            tl.float32
        )
        up = tl.load(
            gate_up_ptr + row * (2 * inner) + inner + cols, mask=mask, other=0.0
        ).to(tl.float32)
        activated = (
            libdevice.div_rn(gate, 1.0 + libdevice.exp(-gate))
            .to(x_dtype)
            .to(tl.float32)
        )
        tl.store(out_ptr + row * inner + cols, (activated * up).to(x_dtype), mask=mask)

    @triton.jit
    def _silu_mul_backward_kernel(
        grad_ptr, gate_up_ptr, out_ptr, inner, FMA: tl.constexpr, BLOCK: tl.constexpr
    ):
        """``d up = X(dh · silu)``, ``d silu = X(dh · up)``, then
        ``silu_backward``: ``dy · s · (1 + x · (1 − s))`` — with the inner
        multiply-add fused when ``FMA``, as nvcc contracts it."""
        row = tl.program_id(0).to(tl.int64)
        cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = cols < inner
        x_dtype = gate_up_ptr.dtype.element_ty
        gate = tl.load(gate_up_ptr + row * (2 * inner) + cols, mask=mask, other=0.0).to(
            tl.float32
        )
        up = tl.load(
            gate_up_ptr + row * (2 * inner) + inner + cols, mask=mask, other=0.0
        ).to(tl.float32)
        dh = tl.load(grad_ptr + row * inner + cols, mask=mask, other=0.0).to(tl.float32)
        denominator = 1.0 + libdevice.exp(-gate)
        activated = libdevice.div_rn(gate, denominator).to(x_dtype).to(tl.float32)
        s = libdevice.div_rn(1.0, denominator)
        d_up = (dh * activated).to(x_dtype)
        d_activated = (dh * up).to(x_dtype).to(tl.float32)
        if FMA:
            inner_term = libdevice.fma(gate, 1.0 - s, 1.0)
        else:
            inner_term = 1.0 + gate * (1.0 - s)
        d_gate = ((d_activated * s) * inner_term).to(x_dtype)
        tl.store(out_ptr + row * (2 * inner) + cols, d_gate, mask=mask)
        tl.store(out_ptr + row * (2 * inner) + inner + cols, d_up, mask=mask)


def _require() -> None:
    if triton is None:
        raise RuntimeError(
            "the fused MoE glue kernels need Triton, which is not importable"
        )


def counting_sort(
    expert_ids: torch.Tensor, num_experts: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """[`.moe_glue_reference.stable_counting_sort`][causalab.neural.engines.pytorch_hooks.kernels.moe_glue_reference.stable_counting_sort] on the device: one
    launch of ``num_experts`` programs."""
    _require()
    ids = expert_ids.contiguous()
    num_pairs = ids.numel()
    perm = torch.empty(num_pairs, dtype=torch.int64, device=ids.device)
    inv_perm = torch.empty_like(perm)
    offsets = torch.empty(num_experts, dtype=torch.int32, device=ids.device)
    _counting_sort_kernel[(num_experts,)](
        ids,
        perm,
        inv_perm,
        offsets,
        num_pairs,
        BLOCK=_SORT_BLOCK,
        num_warps=4,
        **_LAUNCH,
    )
    return perm, inv_perm, offsets


def gather_rows(hidden: torch.Tensor, perm: torch.Tensor, top_k: int) -> torch.Tensor:
    _require()
    hidden = hidden.contiguous()
    num_pairs, width = perm.numel(), hidden.shape[-1]
    out = torch.empty(num_pairs, width, dtype=hidden.dtype, device=hidden.device)
    block = min(_pow2_at_least(width), _GATHER_MAX_BLOCK)
    grid = (num_pairs, triton.cdiv(width, block))
    _gather_rows_kernel[grid](
        hidden, perm, out, width, top_k, BLOCK=block, num_warps=8, **_LAUNCH
    )
    return out


def gather_rows_backward(
    grad_sorted: torch.Tensor, inv_perm: torch.Tensor, top_k: int, rounding: str
) -> torch.Tensor:
    _require()
    grad_sorted = grad_sorted.contiguous()
    width = grad_sorted.shape[-1]
    num_tokens = inv_perm.numel() // top_k
    out = torch.empty(
        num_tokens, width, dtype=grad_sorted.dtype, device=grad_sorted.device
    )
    grid = (num_tokens, triton.cdiv(width, _TOKEN_BLOCK))
    _gather_rows_backward_kernel[grid](
        grad_sorted,
        inv_perm,
        out,
        width,
        TOP_K=top_k,
        K_PAD=_pow2_at_least(top_k),
        PER_STEP=rounding == "per_step",
        BLOCK=_TOKEN_BLOCK,
        num_warps=4,
        **_LAUNCH,
    )
    return out


def epilogue_forward(
    proj_out: torch.Tensor,
    weights: torch.Tensor,
    inv_perm: torch.Tensor,
    top_k: int,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    _require()
    proj_out = proj_out.contiguous()
    weights = weights.contiguous()
    width = proj_out.shape[-1]
    num_tokens = inv_perm.numel() // top_k
    out = torch.empty(num_tokens, width, dtype=out_dtype, device=proj_out.device)
    grid = (num_tokens, triton.cdiv(width, _TOKEN_BLOCK))
    _epilogue_forward_kernel[grid](
        proj_out,
        weights,
        inv_perm,
        out,
        width,
        TOP_K=top_k,
        BLOCK=_TOKEN_BLOCK,
        num_warps=4,
        **_LAUNCH,
    )
    return out


def epilogue_backward(
    grad_final: torch.Tensor,
    proj_out: torch.Tensor,
    weights: torch.Tensor,
    perm: torch.Tensor,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    _require()
    grad_final = grad_final.contiguous()
    proj_out = proj_out.contiguous()
    weights = weights.contiguous()
    num_pairs, width = proj_out.shape
    d_proj = torch.empty_like(proj_out)
    d_weights = torch.empty_like(weights)
    _epilogue_backward_kernel[(num_pairs,)](
        grad_final,
        proj_out,
        weights,
        perm,
        d_proj,
        d_weights,
        width,
        top_k,
        LANES=ROW_SUM_LANES,
        VEC=ROW_SUM_VEC,
        num_warps=1,
        **_LAUNCH,
    )
    return d_proj, d_weights


def silu_mul_forward(gate_up: torch.Tensor) -> torch.Tensor:
    _require()
    gate_up = gate_up.contiguous()
    rows, doubled = gate_up.shape
    inner = doubled // 2
    out = torch.empty(rows, inner, dtype=gate_up.dtype, device=gate_up.device)
    grid = (rows, triton.cdiv(inner, _GATE_BLOCK))
    _silu_mul_forward_kernel[grid](
        gate_up, out, inner, BLOCK=_GATE_BLOCK, num_warps=4, **_LAUNCH
    )
    return out


def silu_mul_backward(
    grad: torch.Tensor, gate_up: torch.Tensor, fma: bool
) -> torch.Tensor:
    _require()
    grad = grad.contiguous()
    gate_up = gate_up.contiguous()
    rows, doubled = gate_up.shape
    inner = doubled // 2
    out = torch.empty_like(gate_up)
    grid = (rows, triton.cdiv(inner, _GATE_BLOCK))
    _silu_mul_backward_kernel[grid](
        grad, gate_up, out, inner, FMA=fma, BLOCK=_GATE_BLOCK, num_warps=4, **_LAUNCH
    )
    return out
