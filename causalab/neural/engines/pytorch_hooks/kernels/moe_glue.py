"""Select fused expert kernels and run their autograd functions.

``plan_moe_glue`` checks device, dtype, shape, and options before tensor
operations. CPU calls, unavailable Triton, or disabled options use eager
operations. Each kernel has further conditions:

* Sort requires more than 32 pairs and at most MAX_COUNTING_SORT_PAIRS.
  This preserves the CUDA sort's stable order at supported lengths.
* Gather uses the ATen forward and fuses backward with width-dependent
  rounding.
* Epilogue requires matching dtypes and a row reduction with 32 vectorized
  lanes and a single warp.
* Gate requires the library's default SiLU activation.

``moe_glue_reference`` specifies each numerical order. Autograd saves the
tensors, dimensions, and dtypes required by backward.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import torch

from causalab.neural.engines.pytorch_hooks.kernels import (
    moe_glue_reference as reference,
)
from causalab.neural.engines.pytorch_hooks.kernels import moe_glue_triton as kernels
from causalab.neural.shared.kernel_options import MoeGlueOptions

__all__ = [
    "GATE_FMA",
    "GluePlan",
    "MAX_COUNTING_SORT_PAIRS",
    "counting_sort",
    "fused_epilogue",
    "fused_gate",
    "fused_gather",
    "plan_moe_glue",
]

#: Whether the gate backward fuses ``1 + x · (1 − s)`` into one multiply-add,
#: as nvcc's default contraction does for ATen's ``silu_backward``. The CUDA
#: parity suite checks both forms; this is the one it found to match.
GATE_FMA = True

#: The widest ``top_k`` the gather backward ranks in registers.
_MAX_TOP_K = 128

#: The most pairs the counting sort is faster than cub's radix sort for
#: (module docstring): the training batches (S = 9 984 / 4 368) are well
#: below, the eval batch (S = 93 600) above and keeps ``torch.sort``.
MAX_COUNTING_SORT_PAIRS = 32768


@dataclasses.dataclass(frozen=True)
class GluePlan:
    """Which fused kernels one grouped-experts call runs."""

    sort: bool = False
    gather: bool = False
    epilogue: bool = False
    gate: bool = False

    @property
    def any(self) -> bool:
        return self.sort or self.gather or self.epilogue or self.gate


def _epilogue_admits(num_pairs: int, width: int) -> bool:
    try:
        config = reference.row_sum_config(num_pairs, width)
    except reference.UnsupportedReduction:
        return False
    return config.lanes == kernels.ROW_SUM_LANES and config.vec == kernels.ROW_SUM_VEC


def plan_moe_glue(
    *,
    hidden_states: torch.Tensor,
    top_k_weights: torch.Tensor,
    expert_dtype: torch.dtype,
    num_pairs: int,
    top_k: int,
    default_silu_gate: bool,
    options: MoeGlueOptions,
) -> GluePlan:
    """The plan for one call (module docstring); no tensor op is run."""
    if (
        hidden_states.device.type != "cuda"
        or not options.kernels
        or not kernels.available()
    ):
        return GluePlan()
    width = hidden_states.shape[-1]
    one_dtype = hidden_states.dtype == top_k_weights.dtype == expert_dtype
    return GluePlan(
        sort=options.enabled("sort")
        and reference.UNSTABLE_SORT_LENGTH < num_pairs <= MAX_COUNTING_SORT_PAIRS,
        gather=options.enabled("gather") and top_k <= _MAX_TOP_K,
        epilogue=options.enabled("epilogue")
        and one_dtype
        and _epilogue_admits(num_pairs, width),
        gate=options.enabled("gate") and default_silu_gate,
    )


def counting_sort(
    expert_ids: torch.Tensor, num_experts: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(perm, inv_perm, offsets)`` — what ``torch.sort`` + ``histc`` +
    ``cumsum`` + the inverse scatter produce, in one launch."""
    return kernels.counting_sort(expert_ids, num_experts)


class _FusedGather(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        hidden: torch.Tensor,
        perm: torch.Tensor,
        inv_perm: torch.Tensor,
        top_k: int,
    ) -> torch.Tensor:
        ctx.save_for_backward(inv_perm)
        ctx.top_k = top_k
        ctx.rounding = reference.index_backward_rounding(hidden.shape[-1])
        # the forward stays ATen's row gather (📐 on an H100, S = 9984 × 2048
        # bf16: `index` 30.8 µs, the Triton copy 39.6 µs); the win is the
        # backward, which replaces the sorted scatter-add
        return hidden[perm // top_k]

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> tuple[torch.Tensor, None, None, None]:
        (inv_perm,) = ctx.saved_tensors
        return (
            kernels.gather_rows_backward(grad, inv_perm, ctx.top_k, ctx.rounding),
            None,
            None,
            None,
        )


class _FusedEpilogue(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        proj_out: torch.Tensor,
        weights: torch.Tensor,
        inv_perm: torch.Tensor,
        perm: torch.Tensor,
        top_k: int,
        out_dtype: torch.dtype,
    ) -> torch.Tensor:
        ctx.save_for_backward(proj_out, weights, perm)
        ctx.top_k = top_k
        return kernels.epilogue_forward(proj_out, weights, inv_perm, top_k, out_dtype)

    @staticmethod
    def backward(
        ctx: Any, grad: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, None, None, None, None]:
        proj_out, weights, perm = ctx.saved_tensors
        d_proj, d_weights = kernels.epilogue_backward(
            grad, proj_out, weights, perm, ctx.top_k
        )
        return d_proj, d_weights, None, None, None, None


class _FusedGate(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, gate_up: torch.Tensor, fma: bool) -> torch.Tensor:
        ctx.save_for_backward(gate_up)
        ctx.fma = fma
        return kernels.silu_mul_forward(gate_up)

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> tuple[torch.Tensor, None]:
        (gate_up,) = ctx.saved_tensors
        return kernels.silu_mul_backward(grad, gate_up, ctx.fma), None


def fused_gather(
    hidden: torch.Tensor, perm: torch.Tensor, inv_perm: torch.Tensor, top_k: int
) -> torch.Tensor:
    """``hidden[perm // top_k]`` with the index backward's exact fold."""
    return _FusedGather.apply(hidden, perm, inv_perm, top_k)


def fused_epilogue(
    proj_out: torch.Tensor,
    weights: torch.Tensor,
    inv_perm: torch.Tensor,
    perm: torch.Tensor,
    top_k: int,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Weight, un-sort, slot-sum and cast in one launch each way."""
    return _FusedEpilogue.apply(proj_out, weights, inv_perm, perm, top_k, out_dtype)


def fused_gate(gate_up: torch.Tensor, fma: bool = GATE_FMA) -> torch.Tensor:
    """``silu(gate) * up`` over the ``[gate | up]`` halves."""
    return _FusedGate.apply(gate_up, fma)
