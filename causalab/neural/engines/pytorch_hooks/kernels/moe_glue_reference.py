"""The ATen semantics of the grouped-experts glue, written out in the order
the CUDA kernels compute them — the contract the Triton kernels of
[`.moe_glue_triton`][causalab.neural.engines.pytorch_hooks.kernels.moe_glue_triton] are held to, bit for bit.

Transformers' ``grouped_mm_experts_forward`` (and the engine's copy in
``experts_path.py``) surrounds its two grouped GEMMs with glue over
``S = tokens · top_k`` rows: a sort of the (token, slot) pairs by expert, a
row gather of the hidden states into that order, the routing-weight
multiply, the un-sort, a sum over the ``top_k`` slots, and — between the
GEMMs — ``silu(gate) * up``. Each is an ATen kernel with a fixed floating
point order; reproducing the outputs *and every gradient* to the bit means
reproducing those orders. This module states them as plain torch over
``float32`` (IEEE on every device, so the references run on the CPU), each
with the ATen source it was read from (torch 2.9.0):

* **Sort** (``cuda/Sort.cpp`` ``sort_cuda_kernel``, ``cuda/Sort.cu``
  ``sortKeyValueInplace``): ``torch.sort(expert_ids)`` with ``stable=False``
  is nonetheless stable on CUDA for every length above 32 — a warp merge
  sort to 128, a block radix sort to 4096, cub's segmented radix sort
  beyond — and unstable (bitonic) at 32 and below. [`stable_counting_sort`][]
  is that stable order: within an expert, rows keep ascending original index.

* **Gather** ``hidden_states[perm // top_k]``: a plain row gather forward;
  the backward is ``index_put_(accumulate=True)`` on a zero tensor, which on
  CUDA sorts the indices (stable radix sort, ``cuda/Indexing.cu``
  ``index_put_with_sort_kernel``) and folds each token's duplicates **in
  sorted-array order**, i.e. in the order the token's slots appear in
  ``perm``. The rounding depends on the row width: ``indexing_backward_kernel``
  (``H > 32``) re-reads the ``bf16`` output between duplicates, so the sum is
  rounded to the output dtype **after every addition**;
  ``indexing_backward_kernel_small_stride`` (``H ≤ 32``) accumulates the
  duplicates in fp32 and rounds **once**. [`index_backward_rounding`][]
  picks by width; [`gather_rows_backward`][] folds accordingly. (The CPU
  kernel, ``cpu/IndexKernel.cpp``, is the per-step fold in array order — the
  same as CUDA's wide kernel, which is what the CPU tests pin.)

* **Weight, un-sort, slot sum** ``(proj_out * w[:, None])[inv_perm].view(T,
  k, H).sum(1).to(dtype)``: the multiply is one fp32 product rounded to the
  promoted dtype ``X``; the un-sort moves rows; the sum over the ``k`` slots
  is a reduction along a non-fastest dimension (``cuda/Reduce.cuh``
  ``thread_reduce_impl``) — one thread per output, **four interleaved fp32
  accumulators** (``vt0 = 4``): slot ``s`` lands in accumulator ``s % 4``,
  and the four are combined left to right, ``((a0 + a1) + a2) + a3``, then
  rounded to ``X``. [`slot_sum_cuda_order`][].

* **Row sum** (the backward of the multiply for the routing weight:
  ``(grad * proj_out).sum(-1)``): a reduction along the fastest dimension
  (``Reduce.cuh`` ``setReduceConfig`` / ``input_vectorized_thread_reduce_impl``
  / ``block_x_reduce``): for widths above 128 each of 32 lanes loads
  4-element vectors at a 32-vector stride into **four accumulators indexed
  by position in the vector**, combines them left to right, then a warp
  shuffle tree pairs adjacent lanes ``(0,1), (2,3), …`` five times; for
  widths of 128 and below the lanes are unvectorized with ``vt0 = 4``
  accumulators by iteration. [`row_sum_cuda_order`][] emulates the launch
  configuration; [`row_sum_config`][] names it and refuses the
  configurations it does not model (a vector tail, a warp split at widths
  from 8192), so the fused path falls back there.

* **Gate** ``silu(gate) * up`` (``cuda/ActivationSiluKernel.cu``): silu is
  ``x / (1 + exp(-x))`` in fp32 rounded to ``X``; the multiply is one fp32
  product rounded to ``X``. Backward: ``d_up = X(dh · silu)``, ``d_silu =
  X(dh · up)`` (the mul backward), then ``silu_backward`` in fp32,
  ``dy · s · (1 + x · (1 − s))`` with ``s = 1 / (1 + exp(-x))``, rounded
  once. [`silu_mul_forward`][], [`silu_mul_backward`][]. nvcc contracts
  the backward's ``1 + x · (1 − s)`` into one fused multiply-add (📐
  on an H100: the Triton kernel matches ATen with ``libdevice.fma`` and not
  without, 333 of 3840 fp32 elements apart); plain torch has no fma, so this
  module's backward is the uncontracted form — equal to ATen's in bf16,
  within an ulp in fp16 / fp32. On the CPU ``exp`` itself is vendor-specific
  (Sleef on x86, another libm on arm64), so even the activation can sit an
  ulp from ATen's vectorized ``silu`` there. The Triton kernel, not this
  formula, carries the bit claim — on the H100, by the golden suite.
"""

from __future__ import annotations

import dataclasses
from typing import Literal

import torch

__all__ = [
    "IndexBackwardRounding",
    "RowSumConfig",
    "UnsupportedReduction",
    "epilogue_backward",
    "epilogue_forward",
    "gather_rows",
    "gather_rows_backward",
    "index_backward_rounding",
    "row_sum_config",
    "row_sum_cuda_order",
    "silu_mul_backward",
    "silu_mul_forward",
    "slot_sum_cuda_order",
    "stable_counting_sort",
]

#: How the index backward rounds a token's duplicates: after every addition
#: (``indexing_backward_kernel``, widths above 32) or once at the end
#: (``indexing_backward_kernel_small_stride``, widths of 32 and below).
IndexBackwardRounding = Literal["per_step", "once"]

#: ``cuda/Indexing.cu``: ``sliceSize <= warp_size`` takes the small-stride kernel.
SMALL_STRIDE_WIDTH = 32

#: ``cuda/Sort.cu``: ``!stable && sort_size <= 32`` takes the unstable bitonic sort.
UNSTABLE_SORT_LENGTH = 32

#: ``Reduce.cuh``: ``mnt_wrapper<T>::MAX_NUM_THREADS`` for every real dtype.
_MAX_REDUCE_THREADS = 512
_WARP = 32
#: ``gpu_reduce_kernel`` defaults: four accumulators, four-wide input vectors.
_VT0 = 4
_INPUT_VEC = 4


class UnsupportedReduction(ValueError):
    """A reduction whose CUDA launch configuration this module does not
    emulate — the fused path is not taken for it."""

    def __init__(self, reason: str, *, rows: int, width: int) -> None:
        self.reason = reason
        self.rows = rows
        self.width = width
        super().__init__(f"row sum over ({rows}, {width}): {reason}")


def _last_pow2(n: int) -> int:
    """ATen's ``last_pow2``: the largest power of two not above ``n``."""
    return 1 << (max(n, 1).bit_length() - 1)


def index_backward_rounding(width: int) -> IndexBackwardRounding:
    """Which fold the CUDA index backward applies to a token's duplicates
    at this row width (module docstring)."""
    return "once" if width <= SMALL_STRIDE_WIDTH else "per_step"


def stable_counting_sort(
    expert_ids: torch.Tensor, num_experts: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(perm, inv_perm, offsets)`` of the stable sort by expert: ``perm``
    lists the (token, slot) pair indices expert by expert, ascending within
    an expert; ``inv_perm[perm] = arange``; ``offsets`` is the int32
    inclusive cumulative count per expert, as ``cumsum(histc)`` gives it."""
    if expert_ids.dim() != 1:
        raise ValueError(f"expert ids must be one-dimensional, got {expert_ids.shape}")
    perm = torch.sort(expert_ids, stable=True).indices
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(perm.numel(), device=perm.device, dtype=perm.dtype)
    counts = torch.bincount(expert_ids, minlength=num_experts)[:num_experts]
    offsets = torch.cumsum(counts, dim=0, dtype=torch.int32)
    return perm, inv_perm, offsets


def gather_rows(hidden: torch.Tensor, perm: torch.Tensor, top_k: int) -> torch.Tensor:
    """``hidden[perm // top_k]``: row ``i`` of the sorted array is the hidden
    state of the token pair ``perm[i]`` belongs to."""
    return hidden[perm // top_k]


def gather_rows_backward(
    grad_sorted: torch.Tensor,
    inv_perm: torch.Tensor,
    top_k: int,
    rounding: IndexBackwardRounding,
) -> torch.Tensor:
    """The gradient of [`gather_rows`][] for ``hidden``: each token's
    ``top_k`` sorted-array rows folded in ascending array position, rounded
    per ``rounding`` (module docstring)."""
    num_tokens = inv_perm.numel() // top_k
    positions = inv_perm.view(num_tokens, top_k).sort(dim=1).values
    rows = grad_sorted[positions].float()  # (tokens, top_k, width)
    acc = torch.zeros_like(rows[:, 0])
    for slot in range(top_k):
        acc = acc + rows[:, slot]
        if rounding == "per_step":
            acc = acc.to(grad_sorted.dtype).float()
    return acc.to(grad_sorted.dtype)


def slot_sum_cuda_order(x: torch.Tensor) -> torch.Tensor:
    """``x.sum(dim=1)`` for ``x`` of shape ``(tokens, slots, width)`` in the
    order the CUDA reduction takes it (module docstring): slot ``s`` into
    fp32 accumulator ``s % 4``, the four combined left to right, rounded to
    ``x.dtype`` once."""
    xf = x.float()
    accs = [torch.zeros_like(xf[:, 0]) for _ in range(_VT0)]
    for slot in range(x.shape[1]):
        accs[slot % _VT0] = accs[slot % _VT0] + xf[:, slot]
    total = accs[0]
    for acc in accs[1:]:
        total = total + acc
    return total.to(x.dtype)


@dataclasses.dataclass(frozen=True)
class RowSumConfig:
    """The part of ATen's reduce launch that fixes the summation order of a
    row sum: ``lanes`` threads share a row, each loading ``vec`` elements
    per step (``vec == 1``: unvectorized, four accumulators by step)."""

    lanes: int
    vec: int


def row_sum_config(rows: int, width: int) -> RowSumConfig:
    """``setReduceConfig`` for ``x.sum(-1)`` over a contiguous ``(rows,
    width)`` tensor whose rows start 16-byte aligned; refuses what
    [`row_sum_cuda_order`][] does not emulate."""
    vectorized = width > 128
    if vectorized and width % _INPUT_VEC:
        raise UnsupportedReduction("a vector tail", rows=rows, width=width)
    vec = _INPUT_VEC if vectorized else 1
    dim0 = width // vec
    dim0_pow2 = _last_pow2(dim0) if dim0 < _MAX_REDUCE_THREADS else _MAX_REDUCE_THREADS
    dim1_pow2 = _last_pow2(rows) if rows < _MAX_REDUCE_THREADS else _MAX_REDUCE_THREADS
    lanes = min(dim0_pow2, _WARP)
    height = min(dim1_pow2, _MAX_REDUCE_THREADS // lanes)
    lanes = min(dim0_pow2, _MAX_REDUCE_THREADS // height)
    values_per_thread = -(-width // lanes)
    if values_per_thread >= min(height * 16, 256):
        raise UnsupportedReduction("a split across warps", rows=rows, width=width)
    return RowSumConfig(lanes=lanes, vec=vec)


def row_sum_cuda_order(x: torch.Tensor) -> torch.Tensor:
    """``x.sum(dim=-1)`` for a contiguous ``(rows, width)`` tensor in the
    order the CUDA reduction takes it (module docstring), rounded to
    ``x.dtype`` once."""
    rows, width = x.shape
    config = row_sum_config(rows, width)
    xf = x.float()
    lanes, vec = config.lanes, config.vec
    if vec > 1:
        vectors = xf.view(rows, width // vec, vec)
        acc = torch.zeros(rows, lanes, vec, dtype=torch.float32, device=x.device)
        for start in range(0, width // vec, lanes):
            take = min(lanes, width // vec - start)
            acc[:, :take] = acc[:, :take] + vectors[:, start : start + take]
        lane = acc[:, :, 0]
        for i in range(1, vec):
            lane = lane + acc[:, :, i]
    else:
        acc = torch.zeros(rows, lanes, _VT0, dtype=torch.float32, device=x.device)
        for step, start in enumerate(range(0, width, lanes)):
            take = min(lanes, width - start)
            slot = step % _VT0
            acc[:, :take, slot] = acc[:, :take, slot] + xf[:, start : start + take]
        lane = acc[:, :, 0]
        for i in range(1, _VT0):
            lane = lane + acc[:, :, i]
    # block_x_reduce: shared-memory halves down to a warp, then the shuffle tree
    while lane.shape[1] > _WARP:
        half = lane.shape[1] // 2
        lane = lane[:, :half] + lane[:, half:]
    while lane.shape[1] > 1:
        lane = lane[:, 0::2] + lane[:, 1::2]
    return lane[:, 0].to(x.dtype)


def epilogue_forward(
    proj_out: torch.Tensor,
    weights: torch.Tensor,
    inv_perm: torch.Tensor,
    top_k: int,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """``(proj_out * weights_sorted[:, None])[inv_perm].view(T, k, H).sum(1)
    .to(out_dtype)`` with ``weights`` in token order (so no gather of the
    weights): the product rounded to the promoted dtype, the slot sum in
    CUDA order."""
    promoted = torch.promote_types(proj_out.dtype, weights.dtype)
    num_tokens = inv_perm.numel() // top_k
    rows = proj_out[inv_perm.view(num_tokens, top_k)].float()
    w = weights.view(num_tokens, top_k, 1).float()
    products = (rows * w).to(promoted)
    return slot_sum_cuda_order(products).to(out_dtype)


def epilogue_backward(
    grad_final: torch.Tensor,
    proj_out: torch.Tensor,
    weights: torch.Tensor,
    perm: torch.Tensor,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(d proj_out, d weights)`` of [`epilogue_forward`][]: row ``i`` of
    the sorted array belongs to pair ``perm[i] = t · k + s``; ``d proj_out[i]
    = X(g[t] · w[t, s])`` rounded to ``proj_out``'s dtype, ``d weights[t, s]
    = rowsum(X(g[t] · proj_out[i]))`` in CUDA order, rounded to ``weights``'
    dtype — where ``g`` is the incoming gradient cast to ``X``."""
    promoted = torch.promote_types(proj_out.dtype, weights.dtype)
    g = grad_final.to(promoted).float()[perm // top_k]  # (S, width)
    w = weights[perm].float().unsqueeze(-1)
    d_proj = (g * w).to(promoted).to(proj_out.dtype)
    products = (g * proj_out.float()).to(promoted)
    d_weights_sorted = row_sum_cuda_order(products).to(weights.dtype)
    d_weights = torch.empty_like(weights)
    d_weights[perm] = d_weights_sorted
    return d_proj, d_weights


def silu_mul_forward(gate_up: torch.Tensor) -> torch.Tensor:
    """``silu(gate) * up`` over the ``[gate | up]`` halves of the last axis,
    with silu's and the product's single roundings."""
    gate, up = gate_up.chunk(2, dim=-1)
    g = gate.float()
    activated = (g / (1 + torch.exp(-g))).to(gate_up.dtype)
    return (activated.float() * up.float()).to(gate_up.dtype)


def silu_mul_backward(grad: torch.Tensor, gate_up: torch.Tensor) -> torch.Tensor:
    """``d gate_up`` of [`silu_mul_forward`][]: the mul backward's two
    rounded products, then ``silu_backward`` in fp32 rounded once."""
    gate, up = gate_up.chunk(2, dim=-1)
    g = gate.float()
    s = 1 / (1 + torch.exp(-g))
    # silu(gate) as the forward saved it: x / (1 + exp(-x)), not x * s
    activated = (g / (1 + torch.exp(-g))).to(gate_up.dtype)
    dh = grad.float()
    d_up = (dh * activated.float()).to(gate_up.dtype)
    d_activated = (dh * up.float()).to(gate_up.dtype).float()
    d_gate = (d_activated * s * (1 + g * (1 - s))).to(gate_up.dtype)
    return torch.cat([d_gate, d_up], dim=-1)
