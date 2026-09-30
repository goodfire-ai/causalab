"""Triton norm and rotary kernels with the reference operation order.

``norm_reference`` records ATen operations and rounding. Launches disable
implicit multiply-add fusion and use libdevice for rsqrt, exponentiation,
division, and the explicitly fused SiLU backward term.

Each row uses ATen's reduction layout: lanes load vectors, accumulate in
four slots, combine them left to right, then reduce through shared memory
and adjacent-lane shuffles. ``row_sum_config`` checks supported shapes.
A second cached read supplies the elementwise output. Grids depend on shape
and support CUDA capture after compilation. ``available`` reports Triton
support.
"""

from __future__ import annotations

import dataclasses
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
    "RowLaunch",
    "available",
    "gated_rms_norm_backward",
    "gated_rms_norm_forward",
    "pow2_at_least",
    "rms_norm_backward",
    "rms_norm_forward",
    "rotary_backward",
    "rotary_forward",
]

#: Every launch: no fused multiply-add where ATen's kernels round twice.
_LAUNCH: dict[str, Any] = {"enable_fp_fusion": False}

#: The widest pass-through block of the rotary kernel (head dims past the
#: rotary width, copied).
MAX_PASS_BLOCK = 4096


def available() -> bool:
    return triton is not None


def _require() -> None:
    if triton is None:
        raise RuntimeError("the fused norm kernels need triton")


def pow2_at_least(n: int) -> int:
    """The smallest power of two that is at least ``n`` (``1`` for ``n ≤ 1``)."""
    return 1 << max(n - 1, 0).bit_length()


@dataclasses.dataclass(frozen=True)
class RowLaunch:
    """One norm launch over ``(rows, width)``: the reduce order's ``lanes``
    and ``vec`` ([`.moe_glue_reference.row_sum_config`][causalab.neural.engines.pytorch_hooks.kernels.moe_glue_reference.row_sum_config]), the number of
    ``lanes · vec`` steps across the row, ``MeanOps``' factor and the
    backward's fp32 reciprocal of the width; the elementwise tail's block is
    the power of two at or above the width."""

    rows: int
    width: int
    lanes: int
    vec: int
    steps: int
    mean_factor: float
    reciprocal: float

    @property
    def block(self) -> int:
        return pow2_at_least(self.width)

    @property
    def num_warps(self) -> int:
        return max(1, min(8, self.block // 512))


if triton is not None:

    @triton.jit
    def _combine_vec4(acc, LANES: tl.constexpr):
        """``acc[4·lane + i]`` → per lane ``((a0 + a1) + a2) + a3``."""
        even, odd = tl.split(tl.reshape(acc, (LANES, 2, 2)))
        a0, a2 = tl.split(even)
        a1, a3 = tl.split(odd)
        return ((a0 + a1) + a2) + a3

    @triton.jit
    def _lane_tree(lane, LANES: tl.constexpr):
        """``block_x_reduce``: halves down to a warp, then adjacent pairs."""
        if LANES > 256:
            lane = tl.sum(tl.reshape(lane, (2, 256)), axis=0)
        if LANES > 128:
            lane = tl.sum(tl.reshape(lane, (2, 128)), axis=0)
        if LANES > 64:
            lane = tl.sum(tl.reshape(lane, (2, 64)), axis=0)
        if LANES > 32:
            lane = tl.sum(tl.reshape(lane, (2, 32)), axis=0)
        v = tl.sum(tl.reshape(lane, (16, 2)), axis=1)
        v = tl.sum(tl.reshape(v, (8, 2)), axis=1)
        v = tl.sum(tl.reshape(v, (4, 2)), axis=1)
        v = tl.sum(tl.reshape(v, (2, 2)), axis=1)
        return tl.sum(v)

    @triton.jit
    def _silu(gate):
        return libdevice.div_rn(gate, 1.0 + libdevice.exp(-gate))

    @triton.jit
    def _gated_grad_u(g, x, gate, w, rstd, X_DTYPE: tl.constexpr):
        """The gated backward's cotangent of ``x · rstd``, from the output
        gradient ``g`` (fp32): ``(X(g · s) · w).to(X)``."""
        u = (x * rstd).to(X_DTYPE).to(tl.float32)
        grad_v = (g * _silu(gate)).to(X_DTYPE).to(tl.float32)
        return (grad_v * w).to(X_DTYPE).to(tl.float32), u

    @triton.jit
    def _rms_norm_fwd_kernel(
        x_ptr,
        w_ptr,
        y_ptr,
        rstd_ptr,
        width,
        x_stride,
        eps,
        factor,
        LANES: tl.constexpr,
        VEC: tl.constexpr,
        STEPS: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        x_row = x_ptr + row * x_stride
        CH: tl.constexpr = LANES * VEC
        cols = tl.arange(0, CH)
        if VEC == 4:
            acc = tl.zeros((CH,), dtype=tl.float32)
            for step in tl.static_range(STEPS):
                c = step * CH + cols
                m = c < width
                x = tl.load(x_row + c, mask=m, other=0.0).to(tl.float32)
                acc = tl.where(m, acc + x * x, acc)
            lane = _combine_vec4(acc, LANES)
        else:
            acc0 = tl.zeros((CH,), dtype=tl.float32)
            acc1 = tl.zeros((CH,), dtype=tl.float32)
            acc2 = tl.zeros((CH,), dtype=tl.float32)
            acc3 = tl.zeros((CH,), dtype=tl.float32)
            for step in tl.static_range(STEPS):
                c = step * CH + cols
                m = c < width
                x = tl.load(x_row + c, mask=m, other=0.0).to(tl.float32)
                p = x * x
                if step % 4 == 0:
                    acc0 = tl.where(m, acc0 + p, acc0)
                elif step % 4 == 1:
                    acc1 = tl.where(m, acc1 + p, acc1)
                elif step % 4 == 2:
                    acc2 = tl.where(m, acc2 + p, acc2)
                else:
                    acc3 = tl.where(m, acc3 + p, acc3)
            lane = ((acc0 + acc1) + acc2) + acc3
        total = _lane_tree(lane, LANES)
        rstd = libdevice.rsqrt(total * factor + eps)
        cols = tl.arange(0, BLOCK)
        m = cols < width
        x = tl.load(x_row + cols, mask=m, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=m, other=0.0).to(tl.float32)
        y = (x * rstd) * (1.0 + w)
        tl.store(y_ptr + row * width + cols, y.to(y_ptr.dtype.element_ty), mask=m)
        tl.store(rstd_ptr + row, rstd)

    @triton.jit
    def _rms_norm_bwd_kernel(
        g_ptr,
        x_ptr,
        w_ptr,
        rstd_ptr,
        dx_ptr,
        width,
        g_stride,
        x_stride,
        recip,
        LANES: tl.constexpr,
        VEC: tl.constexpr,
        STEPS: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        g_row = g_ptr + row * g_stride
        x_row = x_ptr + row * x_stride
        rstd = tl.load(rstd_ptr + row)
        CH: tl.constexpr = LANES * VEC
        cols = tl.arange(0, CH)
        if VEC == 4:
            acc = tl.zeros((CH,), dtype=tl.float32)
            for step in tl.static_range(STEPS):
                c = step * CH + cols
                m = c < width
                g = tl.load(g_row + c, mask=m, other=0.0).to(tl.float32)
                w = tl.load(w_ptr + c, mask=m, other=0.0).to(tl.float32)
                x = tl.load(x_row + c, mask=m, other=0.0).to(tl.float32)
                t = (g * (1.0 + w)) * x
                acc = tl.where(m, acc + t, acc)
            lane = _combine_vec4(acc, LANES)
        else:
            acc0 = tl.zeros((CH,), dtype=tl.float32)
            acc1 = tl.zeros((CH,), dtype=tl.float32)
            acc2 = tl.zeros((CH,), dtype=tl.float32)
            acc3 = tl.zeros((CH,), dtype=tl.float32)
            for step in tl.static_range(STEPS):
                c = step * CH + cols
                m = c < width
                g = tl.load(g_row + c, mask=m, other=0.0).to(tl.float32)
                w = tl.load(w_ptr + c, mask=m, other=0.0).to(tl.float32)
                x = tl.load(x_row + c, mask=m, other=0.0).to(tl.float32)
                t = (g * (1.0 + w)) * x
                if step % 4 == 0:
                    acc0 = tl.where(m, acc0 + t, acc0)
                elif step % 4 == 1:
                    acc1 = tl.where(m, acc1 + t, acc1)
                elif step % 4 == 2:
                    acc2 = tl.where(m, acc2 + t, acc2)
                else:
                    acc3 = tl.where(m, acc3 + t, acc3)
            lane = ((acc0 + acc1) + acc2) + acc3
        grad_rstd = _lane_tree(lane, LANES)
        grad_var = (-0.5 * grad_rstd) * ((rstd * rstd) * rstd)
        gv = grad_var * recip
        cols = tl.arange(0, BLOCK)
        m = cols < width
        g = tl.load(g_row + cols, mask=m, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=m, other=0.0).to(tl.float32)
        x = tl.load(x_row + cols, mask=m, other=0.0).to(tl.float32)
        grad_u = g * (1.0 + w)
        dx = grad_u * rstd + gv * (2.0 * x)
        tl.store(dx_ptr + row * width + cols, dx.to(dx_ptr.dtype.element_ty), mask=m)

    @triton.jit
    def _gated_rms_norm_fwd_kernel(
        x_ptr,
        gate_ptr,
        w_ptr,
        o_ptr,
        rstd_ptr,
        width,
        x_stride,
        gate_stride,
        eps,
        factor,
        LANES: tl.constexpr,
        VEC: tl.constexpr,
        STEPS: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        x_row = x_ptr + row * x_stride
        X_DTYPE: tl.constexpr = x_ptr.dtype.element_ty
        CH: tl.constexpr = LANES * VEC
        cols = tl.arange(0, CH)
        if VEC == 4:
            acc = tl.zeros((CH,), dtype=tl.float32)
            for step in tl.static_range(STEPS):
                c = step * CH + cols
                m = c < width
                x = tl.load(x_row + c, mask=m, other=0.0).to(tl.float32)
                acc = tl.where(m, acc + x * x, acc)
            lane = _combine_vec4(acc, LANES)
        else:
            acc0 = tl.zeros((CH,), dtype=tl.float32)
            acc1 = tl.zeros((CH,), dtype=tl.float32)
            acc2 = tl.zeros((CH,), dtype=tl.float32)
            acc3 = tl.zeros((CH,), dtype=tl.float32)
            for step in tl.static_range(STEPS):
                c = step * CH + cols
                m = c < width
                x = tl.load(x_row + c, mask=m, other=0.0).to(tl.float32)
                p = x * x
                if step % 4 == 0:
                    acc0 = tl.where(m, acc0 + p, acc0)
                elif step % 4 == 1:
                    acc1 = tl.where(m, acc1 + p, acc1)
                elif step % 4 == 2:
                    acc2 = tl.where(m, acc2 + p, acc2)
                else:
                    acc3 = tl.where(m, acc3 + p, acc3)
            lane = ((acc0 + acc1) + acc2) + acc3
        total = _lane_tree(lane, LANES)
        rstd = libdevice.rsqrt(total * factor + eps)
        cols = tl.arange(0, BLOCK)
        m = cols < width
        x = tl.load(x_row + cols, mask=m, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=m, other=0.0).to(tl.float32)
        gate = tl.load(gate_ptr + row * gate_stride + cols, mask=m, other=0.0).to(
            tl.float32
        )
        u = (x * rstd).to(X_DTYPE).to(tl.float32)
        v = (w * u).to(X_DTYPE).to(tl.float32)
        o = v * _silu(gate)
        tl.store(o_ptr + row * width + cols, o.to(o_ptr.dtype.element_ty), mask=m)
        tl.store(rstd_ptr + row, rstd)

    @triton.jit
    def _gated_rms_norm_bwd_kernel(
        g_ptr,
        x_ptr,
        gate_ptr,
        w_ptr,
        rstd_ptr,
        dx_ptr,
        dgate_ptr,
        width,
        g_stride,
        x_stride,
        gate_stride,
        recip,
        LANES: tl.constexpr,
        VEC: tl.constexpr,
        STEPS: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        g_row = g_ptr + row * g_stride
        x_row = x_ptr + row * x_stride
        gate_row = gate_ptr + row * gate_stride
        rstd = tl.load(rstd_ptr + row)
        X_DTYPE: tl.constexpr = x_ptr.dtype.element_ty
        CH: tl.constexpr = LANES * VEC
        cols = tl.arange(0, CH)
        if VEC == 4:
            acc = tl.zeros((CH,), dtype=tl.float32)
            for step in tl.static_range(STEPS):
                c = step * CH + cols
                m = c < width
                g = tl.load(g_row + c, mask=m, other=0.0).to(tl.float32)
                x = tl.load(x_row + c, mask=m, other=0.0).to(tl.float32)
                gate = tl.load(gate_row + c, mask=m, other=0.0).to(tl.float32)
                w = tl.load(w_ptr + c, mask=m, other=0.0).to(tl.float32)
                grad_u, _ = _gated_grad_u(g, x, gate, w, rstd, X_DTYPE)
                acc = tl.where(m, acc + grad_u * x, acc)
            lane = _combine_vec4(acc, LANES)
        else:
            acc0 = tl.zeros((CH,), dtype=tl.float32)
            acc1 = tl.zeros((CH,), dtype=tl.float32)
            acc2 = tl.zeros((CH,), dtype=tl.float32)
            acc3 = tl.zeros((CH,), dtype=tl.float32)
            for step in tl.static_range(STEPS):
                c = step * CH + cols
                m = c < width
                g = tl.load(g_row + c, mask=m, other=0.0).to(tl.float32)
                x = tl.load(x_row + c, mask=m, other=0.0).to(tl.float32)
                gate = tl.load(gate_row + c, mask=m, other=0.0).to(tl.float32)
                w = tl.load(w_ptr + c, mask=m, other=0.0).to(tl.float32)
                grad_u, _ = _gated_grad_u(g, x, gate, w, rstd, X_DTYPE)
                t = grad_u * x
                if step % 4 == 0:
                    acc0 = tl.where(m, acc0 + t, acc0)
                elif step % 4 == 1:
                    acc1 = tl.where(m, acc1 + t, acc1)
                elif step % 4 == 2:
                    acc2 = tl.where(m, acc2 + t, acc2)
                else:
                    acc3 = tl.where(m, acc3 + t, acc3)
            lane = ((acc0 + acc1) + acc2) + acc3
        grad_rstd = _lane_tree(lane, LANES)
        grad_var = (-0.5 * grad_rstd) * ((rstd * rstd) * rstd)
        gv = grad_var * recip
        cols = tl.arange(0, BLOCK)
        m = cols < width
        g = tl.load(g_row + cols, mask=m, other=0.0).to(tl.float32)
        x = tl.load(x_row + cols, mask=m, other=0.0).to(tl.float32)
        gate = tl.load(gate_row + cols, mask=m, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=m, other=0.0).to(tl.float32)
        grad_u, u = _gated_grad_u(g, x, gate, w, rstd, X_DTYPE)
        v = (w * u).to(X_DTYPE).to(tl.float32)
        denominator = 1.0 + libdevice.exp(-gate)
        sigma = libdevice.div_rn(1.0, denominator)
        grad_s = g * v
        dgate = (grad_s * sigma) * libdevice.fma(gate, 1.0 - sigma, 1.0)
        tl.store(
            dgate_ptr + row * width + cols,
            dgate.to(dgate_ptr.dtype.element_ty),
            mask=m,
        )
        dx = grad_u * rstd + gv * (2.0 * x)
        tl.store(dx_ptr + row * width + cols, dx.to(dx_ptr.dtype.element_ty), mask=m)

    @triton.jit
    def _rotary_kernel(
        x_ptr,
        cos_ptr,
        sin_ptr,
        out_ptr,
        heads,
        positions,
        head_dim,
        sx_b,
        sx_h,
        sx_t,
        sc_b,
        sc_h,
        sc_t,
        ss_b,
        ss_h,
        ss_t,
        HALF: tl.constexpr,
        PASS_BLOCK: tl.constexpr,
        HAS_PASS: tl.constexpr,
        BACKWARD: tl.constexpr,
    ):
        """One ``(batch, head, position)`` row: the rotary half-pairs and
        the pass-through dims. Forward ``e₁ = X(x₁c₁) + X((−x₂)s₁)``, ``e₂ =
        X(x₂c₂) + X(x₁s₂)``; backward ``dq₁ = X(g₁c₁) + X(g₂s₂)``, ``dq₂ =
        X(g₂c₂) − X(g₁s₁)``; each sum rounded to the dtype once more."""
        pid = tl.program_id(0).to(tl.int64)
        t = pid % positions
        bh = pid // positions
        h = bh % heads
        b = bh // heads
        x_row = x_ptr + b * sx_b + h * sx_h + t * sx_t
        c_row = cos_ptr + b * sc_b + h * sc_h + t * sc_t
        s_row = sin_ptr + b * ss_b + h * ss_h + t * ss_t
        out_row = out_ptr + pid * head_dim
        DT: tl.constexpr = out_ptr.dtype.element_ty
        j = tl.arange(0, HALF)
        x1 = tl.load(x_row + j).to(tl.float32)
        x2 = tl.load(x_row + HALF + j).to(tl.float32)
        c1 = tl.load(c_row + j).to(tl.float32)
        c2 = tl.load(c_row + HALF + j).to(tl.float32)
        s1 = tl.load(s_row + j).to(tl.float32)
        s2 = tl.load(s_row + HALF + j).to(tl.float32)
        if BACKWARD:
            o1 = (x1 * c1).to(DT).to(tl.float32) + (x2 * s2).to(DT).to(tl.float32)
            o2 = (x2 * c2).to(DT).to(tl.float32) - (x1 * s1).to(DT).to(tl.float32)
        else:
            o1 = (x1 * c1).to(DT).to(tl.float32) + ((-x2) * s1).to(DT).to(tl.float32)
            o2 = (x2 * c2).to(DT).to(tl.float32) + (x1 * s2).to(DT).to(tl.float32)
        tl.store(out_row + j, o1.to(DT))
        tl.store(out_row + HALF + j, o2.to(DT))
        if HAS_PASS:
            p = tl.arange(0, PASS_BLOCK)
            m = p < head_dim - 2 * HALF
            v = tl.load(x_row + 2 * HALF + p, mask=m, other=0.0)
            tl.store(out_row + 2 * HALF + p, v, mask=m)


def rms_norm_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    launch: RowLaunch,
    *,
    shape: torch.Size,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``x (rows, width)`` with a unit inner stride → ``(y, rstd (rows,)
    fp32)``; ``y`` in ``x``'s dtype, allocated contiguous as ``shape`` — the
    caller's ``rows · width`` elements, row ``r`` at offset ``r · width`` —
    so no view stands between the caller's tensor and the result."""
    _require()
    y = torch.empty(shape, dtype=x.dtype, device=x.device)
    rstd = torch.empty(launch.rows, dtype=torch.float32, device=x.device)
    _rms_norm_fwd_kernel[(launch.rows,)](
        x,
        weight,
        y,
        rstd,
        launch.width,
        x.stride(0),
        eps,
        launch.mean_factor,
        LANES=launch.lanes,
        VEC=launch.vec,
        STEPS=launch.steps,
        BLOCK=launch.block,
        num_warps=launch.num_warps,
        **_LAUNCH,
    )
    return y, rstd


def rms_norm_backward(
    grad: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    rstd: torch.Tensor,
    launch: RowLaunch,
    *,
    shape: torch.Size,
) -> torch.Tensor:
    """``dx`` in ``x``'s dtype, allocated as ``shape`` ([`rms_norm_forward`][])."""
    _require()
    dx = torch.empty(shape, dtype=x.dtype, device=x.device)
    _rms_norm_bwd_kernel[(launch.rows,)](
        grad,
        x,
        weight,
        rstd,
        dx,
        launch.width,
        grad.stride(0),
        x.stride(0),
        launch.reciprocal,
        LANES=launch.lanes,
        VEC=launch.vec,
        STEPS=launch.steps,
        BLOCK=launch.block,
        num_warps=launch.num_warps,
        **_LAUNCH,
    )
    return dx


def gated_rms_norm_forward(
    x: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    launch: RowLaunch,
    *,
    shape: torch.Size,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(o, rstd)``; ``o`` in ``x``'s dtype, allocated as ``shape``
    ([`rms_norm_forward`][])."""
    _require()
    o = torch.empty(shape, dtype=x.dtype, device=x.device)
    rstd = torch.empty(launch.rows, dtype=torch.float32, device=x.device)
    _gated_rms_norm_fwd_kernel[(launch.rows,)](
        x,
        gate,
        weight,
        o,
        rstd,
        launch.width,
        x.stride(0),
        gate.stride(0),
        eps,
        launch.mean_factor,
        LANES=launch.lanes,
        VEC=launch.vec,
        STEPS=launch.steps,
        BLOCK=launch.block,
        num_warps=launch.num_warps,
        **_LAUNCH,
    )
    return o, rstd


def gated_rms_norm_backward(
    grad: torch.Tensor,
    x: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    rstd: torch.Tensor,
    launch: RowLaunch,
    *,
    shape: torch.Size,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(dx, dgate)`` in ``x``'s and ``gate``'s dtypes, both allocated as
    ``shape`` ([`rms_norm_forward`][])."""
    _require()
    dx = torch.empty(shape, dtype=x.dtype, device=x.device)
    dgate = torch.empty(shape, dtype=gate.dtype, device=x.device)
    _gated_rms_norm_bwd_kernel[(launch.rows,)](
        grad,
        x,
        gate,
        weight,
        rstd,
        dx,
        dgate,
        launch.width,
        grad.stride(0),
        x.stride(0),
        gate.stride(0),
        launch.reciprocal,
        LANES=launch.lanes,
        VEC=launch.vec,
        STEPS=launch.steps,
        BLOCK=launch.block,
        num_warps=launch.num_warps,
        **_LAUNCH,
    )
    return dx, dgate


def _rotary(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, backward: bool
) -> torch.Tensor:
    """``x (B, H, T, D)`` with a unit inner stride; ``cos`` / ``sin`` expanded
    to ``(B, H, T, rot)``. The output is contiguous."""
    _require()
    batch, heads, positions, head_dim = x.shape
    rot = cos.shape[-1]
    half = rot // 2
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    passthrough = head_dim - rot
    _rotary_kernel[(batch * heads * positions,)](
        x,
        cos,
        sin,
        out,
        heads,
        positions,
        head_dim,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        cos.stride(0),
        cos.stride(1),
        cos.stride(2),
        sin.stride(0),
        sin.stride(1),
        sin.stride(2),
        HALF=half,
        PASS_BLOCK=pow2_at_least(max(passthrough, 1)),
        HAS_PASS=passthrough > 0,
        BACKWARD=backward,
        num_warps=1,
        **_LAUNCH,
    )
    return out


def rotary_forward(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    return _rotary(x, cos, sin, backward=False)


def rotary_backward(
    grad: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    return _rotary(grad, cos, sin, backward=True)
