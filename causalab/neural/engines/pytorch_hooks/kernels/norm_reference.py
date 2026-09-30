"""Order-explicit torch references for the fused norm and rotary kernels
(``norm_triton.py``): the ATen op sequence of each module forward the kernels
replace, and of the autograd graph behind it, one rounding per line.

Sources (transformers 5.16 ``modeling_qwen3_5_moe.py``; torch 2.9 CUDA):

* ``Qwen3_5MoeRMSNorm.forward`` — ``x.float()``; ``x.pow(2)`` (``x · x``,
  ``PowKernel.cu``); ``.mean(-1, keepdim=True)`` (``Reduce.cuh`` ``MeanOps``:
  the row sum in [`.moe_glue_reference.row_sum_cuda_order`][causalab.neural.engines.pytorch_hooks.kernels.moe_glue_reference.row_sum_cuda_order]'s launch
  order, then ``· factor`` with ``factor = float(rows) / float(rows · width)``
  — [`mean_factor`][]); ``+ eps``; ``torch.rsqrt``; ``x · rstd``;
  ``· (1.0 + weight.float())``; ``.type_as(x)``.
* its backward, node by node — ``ToCopyBackward0`` (``grad.float()``);
  ``MulBackward0`` (``grad · (1 + w)``; the weight is frozen, so no column
  sum); ``MulBackward0`` of ``x · rstd`` (``grad_u · rstd`` for ``x``, and for
  ``rstd`` autograd's ``sum_to``: ``(grad_u · x).sum(-1, keepdim=True)``, the
  same reduce launch as the mean); ``RsqrtBackward0`` (``(-0.5 · g) · ((r · r)
  · r)``); ``AddBackward0``; ``MeanBackward1`` (``g / N``: a division by a CPU
  scalar multiplies by the fp32 reciprocal, [`reciprocal`][]);
  ``PowBackward0`` (``g · (2 · x)``: ``pow(1)`` is a copy); the two ``x``
  cotangents added; ``ToCopyBackward0`` (``.to(x.dtype)``).
* ``Qwen3_5MoeRMSNormGated.forward`` — the same norm to ``x · rstd``, then
  ``weight · h.to(dtype)`` (one fp32 product rounded to ``dtype``), ``· silu(
  gate.float())`` in fp32 (``x / (1 + exp(-x))``, ``ActivationSiluKernel.cu``),
  ``.to(dtype)``. Backward: ``grad_v = (g · s).to(dtype)``, ``grad_s = g · v``;
  ``silu_backward`` ``(g · σ) · (1 + x · (1 − σ))`` with ``σ = 1 / (1 +
  exp(-x))`` — nvcc contracts the inner term into one fused multiply-add;
  plain torch cannot, so this module's form is the uncontracted one (equal
  in bf16, an ulp apart in fp32; the kernel uses ``libdevice.fma``); ``.to(
  gate.dtype)``; ``grad_u = (grad_v · w).to(dtype)``; the norm's backward from
  ``grad_u``.
* ``apply_rotary_pos_emb`` — ``q_rot · cos + rotate_half(q_rot) · sin`` over
  the first ``cos.shape[-1]`` head dims, the rest passed through; every
  product and the sum rounded to the dtype. Backward through the slices,
  ``cat`` and ``neg``: each half of ``q_rot`` receives one product from each
  branch and adding the slice backward's zeros is exact, so ``dq₁ = X(g₁·c₁)
  + X(g₂·s₂)``, ``dq₂ = X(g₂·c₂) − X(g₁·s₁)``, the pass-through dims the
  gradient itself.

On the CPU these references agree with the modules to rounding only: CPU
``mean`` / ``sum`` reduce in another order, and ``rsqrt`` / ``exp`` are the
vendor libm's. On CUDA they are the modules to the bit — the property
``tests/golden/test_fused_norm_kernels.py`` pins at the workflow's shapes,
along with each kernel being its reference.
"""

from __future__ import annotations

import dataclasses
import struct

import torch

from causalab.neural.engines.pytorch_hooks.kernels.moe_glue_reference import (
    UnsupportedReduction,
    row_sum_config,
    row_sum_cuda_order,
)

__all__ = [
    "RowGeometry",
    "UnsupportedReduction",
    "gated_rms_norm_backward",
    "gated_rms_norm_forward",
    "mean_factor",
    "reciprocal",
    "rms_norm_backward",
    "rms_norm_forward",
    "rotary_backward",
    "rotary_forward",
    "row_geometry",
    "row_sum_config",
    "rows_view",
]


def _fp32(value: float) -> float:
    """``value`` rounded to the nearest fp32 (the C++ ``float`` it stands
    for), as a Python float."""
    return struct.unpack("f", struct.pack("f", value))[0]


def mean_factor(rows: int, width: int) -> float:
    """``MeanOps``' ``factor`` for ``x.mean(-1)`` over ``(rows, width)``:
    ``float(num_output_elements) / numel`` evaluated in fp32 (the counts
    converted to fp32 first, as C++ does; the double quotient rounded to
    fp32 is the fp32 quotient, double rounding being harmless for a 53-bit
    intermediate)."""
    return _fp32(_fp32(rows) / _fp32(rows * width))


def reciprocal(width: int) -> float:
    """The fp32 ``1 / width`` a division by the CPU scalar ``width``
    multiplies by (``div_true_kernel_cuda``)."""
    return _fp32(1.0 / _fp32(width))


@dataclasses.dataclass(frozen=True)
class RowGeometry:
    """``x`` as ``(rows, width)`` rows of one stride — the ints a plan keeps
    so that no view is taken where autograd would record it."""

    rows: int
    row_stride: int


def row_geometry(x: torch.Tensor) -> RowGeometry | None:
    """The ``(rows, row_stride)`` under which ``x`` is a ``(rows, width)``
    tensor with a unit inner stride — what the kernels index — or ``None``
    when its leading dimensions do not merge into one (the fused path is
    not taken; the module's own forward would copy it into a contiguous
    fp32 tensor). Reads shape and strides only."""
    if x.ndim == 0 or x.shape[-1] == 0 or x.stride(-1) != 1:
        return None
    width = x.shape[-1]
    if x.ndim == 1:
        return RowGeometry(rows=1, row_stride=width)
    rows = x.numel() // width
    leading = [(s, st) for s, st in zip(x.shape[:-1], x.stride()[:-1]) if s != 1]
    if not leading:
        return RowGeometry(rows=rows, row_stride=width)
    # every leading dimension steps by the row stride times the product of
    # the dimensions inside it — one row stride for the whole tensor
    row_stride = leading[-1][1]
    if row_stride < width:
        return None
    expected = row_stride
    for size, stride in reversed(leading):
        if stride != expected:
            return None
        expected *= size
    return RowGeometry(rows=rows, row_stride=row_stride)


def rows_view(x: torch.Tensor) -> torch.Tensor | None:
    """``x`` viewed as [`row_geometry`][] says, sharing its storage — or
    ``None`` where the geometry is. A differentiable view under grad mode:
    the kernels' callers take it inside their ``Function``."""
    geometry = row_geometry(x)
    if geometry is None:
        return None
    return x.as_strided((geometry.rows, x.shape[-1]), (geometry.row_stride, 1))


def _mean_cuda_order(x: torch.Tensor) -> torch.Tensor:
    """``x.mean(-1, keepdim=True)`` for a contiguous fp32 ``(rows, width)``
    tensor in the CUDA launch's order."""
    rows, width = x.shape
    return (row_sum_cuda_order(x) * mean_factor(rows, width)).unsqueeze(1)


def rms_norm_forward(
    x: torch.Tensor, weight: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """``Qwen3_5MoeRMSNorm.forward`` over ``x (rows, width)``: ``(y in
    x.dtype, rstd (rows, 1) fp32)``."""
    xf = x.float().contiguous()
    rstd = torch.rsqrt(_mean_cuda_order(xf * xf) + eps)
    y = (xf * rstd) * (1.0 + weight.float())
    return y.to(x.dtype), rstd


def rms_norm_backward(
    grad: torch.Tensor, x: torch.Tensor, weight: torch.Tensor, rstd: torch.Tensor
) -> torch.Tensor:
    """The autograd graph of [`rms_norm_forward`][], node by node (module
    docstring), from ``grad`` in ``y``'s dtype to ``dx`` in ``x``'s."""
    width = x.shape[-1]
    xf = x.float().contiguous()
    grad_u = grad.float() * (1.0 + weight.float())
    return _norm_backward_from_grad_u(grad_u, xf, rstd, width).to(x.dtype)


def _norm_backward_from_grad_u(
    grad_u: torch.Tensor, xf: torch.Tensor, rstd: torch.Tensor, width: int
) -> torch.Tensor:
    grad_x1 = grad_u * rstd
    grad_rstd = row_sum_cuda_order((grad_u * xf).contiguous()).unsqueeze(1)
    grad_var = (-0.5 * grad_rstd) * ((rstd * rstd) * rstd)
    gv = grad_var * reciprocal(width)
    grad_x2 = gv * (2.0 * xf)
    return grad_x1 + grad_x2


def _silu_parts(gate: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(silu(gate), sigmoid(gate))`` as ATen spells them in fp32."""
    gf = gate.float()
    denominator = 1.0 + torch.exp(-gf)
    return gf / denominator, 1.0 / denominator


def gated_rms_norm_forward(
    x: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """``Qwen3_5MoeRMSNormGated.forward`` over ``x, gate (rows, width)``:
    ``(o in x.dtype, rstd (rows, 1) fp32)``. ``weight`` shares ``x``'s dtype."""
    xf = x.float().contiguous()
    rstd = torch.rsqrt(_mean_cuda_order(xf * xf) + eps)
    v = weight * (xf * rstd).to(x.dtype)
    s, _ = _silu_parts(gate)
    return (v * s).to(x.dtype), rstd


def gated_rms_norm_backward(
    grad: torch.Tensor,
    x: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    rstd: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The autograd graph of [`gated_rms_norm_forward`][]: ``(dx in x's
    dtype, dgate in gate's dtype)``."""
    width = x.shape[-1]
    xf = x.float().contiguous()
    g = grad.float()
    v = weight * (xf * rstd).to(x.dtype)
    s, sigma = _silu_parts(gate)
    grad_v = (g * s).to(x.dtype)
    grad_s = g * v.float()
    gf = gate.float()
    dgate = ((grad_s * sigma) * (1.0 + gf * (1.0 - sigma))).to(gate.dtype)
    grad_u = (grad_v * weight).float()
    return _norm_backward_from_grad_u(grad_u, xf, rstd, width).to(x.dtype), dgate


def rotary_forward(
    q: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """``apply_rotary_pos_emb`` for one tensor: ``cos`` / ``sin`` already
    unsqueezed to broadcast over ``q`` and ``cos.shape[-1]`` the rotary
    width."""
    rot = cos.shape[-1]
    half = rot // 2
    q1, q2 = q[..., :half], q[..., half:rot]
    c1, c2 = cos[..., :half], cos[..., half:]
    s1, s2 = sin[..., :half], sin[..., half:]
    e1 = q1 * c1 + (-q2) * s1
    e2 = q2 * c2 + q1 * s2
    return torch.cat([e1, e2, q[..., rot:]], dim=-1)


def rotary_backward(
    grad: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """The gradient of [`rotary_forward`][] with respect to ``q`` (module
    docstring)."""
    rot = cos.shape[-1]
    half = rot // 2
    g1, g2 = grad[..., :half], grad[..., half:rot]
    c1, c2 = cos[..., :half], cos[..., half:]
    s1, s2 = sin[..., :half], sin[..., half:]
    dq1 = g1 * c1 + g2 * s2
    dq2 = g2 * c2 + (-(g1 * s1))
    return torch.cat([dq1, dq2, grad[..., rot:]], dim=-1)
