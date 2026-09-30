"""The fused norms and rotary embedding (``pytorch_hooks/kernels/fused_norms.py``)
on a machine without CUDA: the order-explicit references are the modules to
rounding (to the bit where the ops are elementwise), the plans admit what
they say and nothing off CUDA, the binding installs and restores the pinned
classes and function of the tiny Qwen3.5-MoE and changes no number here,
and the pinned sources still match the installed transformers (the drift
canary). The kernels themselves run only under
``tests/golden/test_fused_norm_kernels.py``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as modeling

from causalab.neural.engines.pytorch_hooks.kernels import fused_norms
from causalab.neural.engines.pytorch_hooks.kernels import norm_reference as ref
from causalab.neural.engines.pytorch_hooks.kernels import norm_triton as kernels
from causalab.neural.engines.pytorch_hooks.kernels.fused_norms import (
    PINNED,
    canonical_source,
    fused_norm_path,
    plan_norm,
    plan_rotary,
    targets_of,
)
from causalab.neural.engines.pytorch_hooks.loading import ModelBundle
from causalab.neural.shared.kernel_options import ENV_FUSED_NORMS, FusedNormOptions

pytestmark = pytest.mark.numerical_unit

ALL = FusedNormOptions()
DTYPES = (torch.float32, torch.bfloat16)
#: (rows, width): the reduce's vectorized launch (widths above 128), the
#: unvectorized one (128 and below), a lane count above a warp (few rows),
#: and a partial last step.
SHAPES = ((16, 2048), (40, 256), (64, 128), (5, 2048), (37, 96))


def _norm(width: int, dtype: torch.dtype) -> Any:
    norm = modeling.Qwen3_5MoeRMSNorm(width, eps=1e-6).to(dtype)
    norm.weight.data.normal_()
    norm.requires_grad_(False)
    return norm


def _gated(width: int, dtype: torch.dtype) -> Any:
    norm = modeling.Qwen3_5MoeRMSNormGated(width, eps=1e-6).to(dtype)
    norm.weight.data.normal_()
    norm.requires_grad_(False)
    return norm


def _tolerance(dtype: torch.dtype, scale: torch.Tensor) -> float:
    """One rounding of the result in ``dtype`` plus the CPU reduce's fp32
    drift, relative to the tensor's largest magnitude."""
    unit = 2.0**-8 if dtype == torch.bfloat16 else 2.0**-20
    return float(scale.detach().float().abs().max()) * unit * 2 + 1e-6


class TestReferences:
    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("rows,width", SHAPES)
    def test_rms_norm_is_the_module_to_rounding(
        self, rows: int, width: int, dtype: torch.dtype
    ) -> None:
        torch.manual_seed(rows * width)
        norm = _norm(width, dtype)
        x = torch.randn(rows, width).to(dtype).requires_grad_(True)
        y = norm(x)
        grad = torch.randn_like(y)
        (y * grad).sum().backward()
        assert x.grad is not None
        ours, rstd = ref.rms_norm_forward(x.detach(), norm.weight, norm.eps)
        assert ours.dtype == y.dtype and rstd.shape == (rows, 1)
        assert torch.allclose(ours.float(), y.float(), atol=_tolerance(dtype, y))
        dx = ref.rms_norm_backward(grad, x.detach(), norm.weight, rstd)
        assert dx.dtype == x.dtype
        assert torch.allclose(
            dx.float(), x.grad.float(), atol=_tolerance(dtype, x.grad)
        )

    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("rows,width", SHAPES)
    def test_gated_rms_norm_is_the_module_to_rounding(
        self, rows: int, width: int, dtype: torch.dtype
    ) -> None:
        torch.manual_seed(rows * width + 1)
        norm = _gated(width, dtype)
        x = torch.randn(rows, width).to(dtype).requires_grad_(True)
        gate = torch.randn(rows, width).to(dtype).requires_grad_(True)
        o = norm(x, gate)
        grad = torch.randn_like(o)
        (o * grad).sum().backward()
        assert x.grad is not None and gate.grad is not None
        ours, rstd = ref.gated_rms_norm_forward(
            x.detach(), gate.detach(), norm.weight, norm.variance_epsilon
        )
        assert torch.allclose(ours.float(), o.float(), atol=_tolerance(dtype, o))
        dx, dgate = ref.gated_rms_norm_backward(
            grad, x.detach(), gate.detach(), norm.weight, rstd
        )
        assert torch.allclose(
            dx.float(), x.grad.float(), atol=_tolerance(dtype, x.grad)
        )
        assert torch.allclose(
            dgate.float(), gate.grad.float(), atol=_tolerance(dtype, gate.grad)
        )

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_rotary_is_the_library_function_to_the_bit(
        self, dtype: torch.dtype
    ) -> None:
        """Elementwise ops only: the same roundings on every device."""
        torch.manual_seed(3)
        batch, heads, kv_heads, positions, head_dim, rot = 2, 4, 2, 5, 32, 16
        q = torch.randn(batch, positions, heads, head_dim).to(dtype).transpose(1, 2)
        k = torch.randn(batch, positions, kv_heads, head_dim).to(dtype).transpose(1, 2)
        q.requires_grad_(True)
        k.requires_grad_(True)
        cos = torch.randn(batch, positions, rot).to(dtype)
        sin = torch.randn(batch, positions, rot).to(dtype)
        q_embed, k_embed = modeling.apply_rotary_pos_emb(q, k, cos, sin)
        grad_q, grad_k = torch.randn_like(q_embed), torch.randn_like(k_embed)
        (q_embed * grad_q).sum().backward()
        (k_embed * grad_k).sum().backward()
        assert q.grad is not None and k.grad is not None
        cos_u, sin_u = cos.unsqueeze(1), sin.unsqueeze(1)
        assert torch.equal(ref.rotary_forward(q.detach(), cos_u, sin_u), q_embed)
        assert torch.equal(ref.rotary_forward(k.detach(), cos_u, sin_u), k_embed)
        assert torch.equal(ref.rotary_backward(grad_q, cos_u, sin_u), q.grad)
        assert torch.equal(ref.rotary_backward(grad_k, cos_u, sin_u), k.grad)

    def test_mean_factor_and_reciprocal_are_fp32(self) -> None:
        assert ref.mean_factor(1248, 2048) == 2.0**-11
        assert ref.mean_factor(93600, 2048) == 2.0**-11
        assert ref.reciprocal(2048) == 2.0**-11
        # a width that is not a power of two rounds in fp32
        factor = ref.mean_factor(37, 96)
        assert factor == float(torch.tensor(37.0 / (37 * 96), dtype=torch.float32))
        assert ref.reciprocal(96) == float(torch.tensor(1 / 96, dtype=torch.float32))


class TestRowsView:
    def test_contiguous_and_merged_leading_dims(self) -> None:
        x = torch.randn(2, 3, 4, 16)
        rows = ref.rows_view(x)
        assert rows is not None and rows.shape == (24, 16) and rows.stride() == (16, 1)
        assert torch.equal(rows, x.reshape(24, 16))

    def test_a_chunk_keeps_its_row_stride_without_a_copy(self) -> None:
        """The attention's ``q`` is a chunk of ``q_proj``'s output: rows of
        256 in a 512-wide buffer."""
        full = torch.randn(2, 3, 4, 32)
        half = torch.chunk(full, 2, dim=-1)[0]
        assert ref.row_geometry(half) == ref.RowGeometry(rows=24, row_stride=32)
        rows = ref.rows_view(half)
        assert rows is not None and rows.stride() == (32, 1)
        assert rows.data_ptr() == half.data_ptr()
        assert torch.equal(rows, half.reshape(24, 16))

    def test_a_transposed_tensor_does_not_merge(self) -> None:
        x = torch.randn(2, 3, 4, 16).transpose(1, 2)
        assert ref.rows_view(x) is None
        assert ref.rows_view(torch.randn(4, 16).t()) is None

    def test_size_one_dims_are_free(self) -> None:
        x = torch.randn(1, 6, 1, 16)
        rows = ref.rows_view(x)
        assert rows is not None and rows.shape == (6, 16)


def _cuda_shaped(shape: tuple[int, ...], dtype: torch.dtype) -> Any:
    """What a plan reads off a contiguous frozen tensor, as if on CUDA — a
    stand-in for the checks that need no tensor op (the CUDA gate itself is
    exercised through ``plan_norm`` with a real CPU tensor below)."""
    strides = tuple(
        int(torch.tensor(shape[i + 1 :]).prod()) if i + 1 < len(shape) else 1
        for i in range(len(shape))
    )
    numel = 1
    for size in shape:
        numel *= size
    return SimpleNamespace(
        device=torch.device("cuda"),
        dtype=dtype,
        shape=shape,
        ndim=len(shape),
        stride=lambda i=None: strides if i is None else strides[i],
        numel=lambda: numel,
        requires_grad=False,
        is_contiguous=lambda: True,
    )


class TestPlans:
    def test_nothing_off_cuda_whatever_the_options_say(self) -> None:
        norm = _norm(2048, torch.bfloat16)
        x = torch.randn(16, 2048, dtype=torch.bfloat16)
        assert plan_norm(x, norm.weight, options=ALL) is None
        gated = _gated(128, torch.bfloat16)
        gate = torch.randn(16, 128, dtype=torch.bfloat16)
        assert (
            plan_norm(gate, gated.weight, gate, options=ALL, activation="silu") is None
        )
        q = torch.randn(2, 4, 5, 32, dtype=torch.bfloat16)
        cos = torch.randn(2, 1, 5, 16, dtype=torch.bfloat16)
        assert not plan_rotary(q, cos, cos, ALL)

    def test_a_disabled_kernel_never_plans(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(kernels, "available", lambda: True)
        off = FusedNormOptions(kernels=frozenset())
        norm = _norm(2048, torch.bfloat16)
        x = torch.randn(16, 2048, dtype=torch.bfloat16)
        assert plan_norm(x, norm.weight, options=off) is None

    @pytest.mark.parametrize("rows,width", SHAPES + ((1, 2048), (3, 64)))
    def test_row_launch_follows_the_reduce_config(self, rows: int, width: int) -> None:
        launch = fused_norms.row_launch(rows, width)
        assert launch is not None
        config = ref.row_sum_config(rows, width)
        assert (launch.lanes, launch.vec) == (config.lanes, config.vec)
        assert launch.steps * launch.lanes * launch.vec >= width
        assert (launch.steps - 1) * launch.lanes * launch.vec < width
        assert launch.block >= width and launch.block & (launch.block - 1) == 0
        assert launch.mean_factor == ref.mean_factor(rows, width)

    def test_widths_the_reduce_order_is_not_modelled_for_are_refused(self) -> None:
        assert fused_norms.row_launch(16, 16) is None  # below 32 lanes
        assert fused_norms.row_launch(16, 8192) is None  # a warp split
        assert fused_norms.row_launch(16, 130) is None  # a vector tail
        assert fused_norms.row_launch(16, 4096) is not None

    def test_a_trainable_weight_is_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The weight's column sum is the one ATen order not mirrored."""
        monkeypatch.setattr(kernels, "available", lambda: True)
        norm = _norm(256, torch.float32)
        norm.weight.requires_grad_(True)
        x = cast(torch.Tensor, _cuda_shaped((16, 256), torch.float32))
        # the weight check runs before any device-specific view
        assert plan_norm(x, norm.weight, options=ALL) is None

    def test_a_tensor_subclass_is_refused_before_any_read(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Under ``tp > 1`` the registry's ``replicated_with_grad_allreduce``
        rows make ``q_norm`` / ``k_norm`` weights replicated DTensors, and the
        class-level rebind reads ``self.weight`` straight off the module:
        every shape and device check passes for such a wrapper and the
        kernel would want a data pointer it does not have. A wrapper
        subclass of either operand is refused by type; a parameter is not a
        wrapper."""

        class _Wrapper(torch.Tensor):
            pass

        monkeypatch.setattr(kernels, "available", lambda: True)
        x = cast(torch.Tensor, _cuda_shaped((16, 256), torch.float32))
        on_cuda = cast(torch.Tensor, _cuda_shaped((256,), torch.float32))
        assert plan_norm(x, on_cuda, options=ALL) is not None  # the control
        wrapped_weight = torch.ones(256).as_subclass(_Wrapper)
        assert isinstance(wrapped_weight, torch.Tensor)
        assert plan_norm(x, wrapped_weight, options=ALL) is None
        wrapped_x = torch.ones(16, 256).as_subclass(_Wrapper)
        assert plan_norm(wrapped_x, on_cuda, options=ALL) is None
        assert fused_norms._wrapped(torch.nn.Parameter(torch.ones(4))) is False  # pyright: ignore[reportPrivateUsage]
        assert fused_norms._wrapped(on_cuda) is False  # pyright: ignore[reportPrivateUsage]

    def test_the_weight_must_share_the_device(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The kernels read the weight through a raw pointer, so a weight on
        another device is refused before it could be a silent wrong read."""
        monkeypatch.setattr(kernels, "available", lambda: True)
        x = cast(torch.Tensor, _cuda_shaped((16, 256), torch.float32))
        on_cuda = cast(torch.Tensor, _cuda_shaped((256,), torch.float32))
        plan = plan_norm(x, on_cuda, options=ALL)
        assert plan is not None and (plan.rows, plan.row_stride) == (16, 256)
        on_cpu = _norm(256, torch.float32).weight
        assert plan_norm(x, on_cpu, options=ALL) is None

    def test_a_gated_norm_admits_only_silu(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The activation is part of the call: the kernel spells ``silu``."""
        monkeypatch.setattr(kernels, "available", lambda: True)
        x = cast(torch.Tensor, _cuda_shaped((16, 128), torch.bfloat16))
        weight = cast(torch.Tensor, _cuda_shaped((128,), torch.bfloat16))
        plan = plan_norm(x, weight, x, options=ALL, activation="silu")
        assert plan is not None and plan.gate_stride == 128
        assert plan_norm(x, weight, x, options=ALL, activation="gelu") is None
        assert plan_norm(x, weight, x, options=ALL) is None

    def test_rotary_admission(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(kernels, "available", lambda: True)
        cuda = torch.device("cuda")

        def tensor_like(**fields: Any) -> torch.Tensor:
            return cast(torch.Tensor, SimpleNamespace(**fields))

        q = tensor_like(
            device=cuda,
            dtype=torch.bfloat16,
            ndim=4,
            shape=(2, 4, 5, 256),
            stride=lambda i: 1,
        )
        cos_fields: dict[str, Any] = dict(
            device=cuda,
            dtype=torch.bfloat16,
            ndim=4,
            shape=(2, 1, 5, 64),
            stride=lambda i: 1,
            requires_grad=False,
        )
        cos = tensor_like(**cos_fields)
        assert plan_rotary(q, cos, cos, ALL)
        assert not plan_rotary(
            q, cos, cos, FusedNormOptions(kernels=frozenset({"norm"}))
        )
        odd = tensor_like(**{**cos_fields, "shape": (2, 1, 5, 48)})
        assert not plan_rotary(q, odd, odd, ALL)
        promoted = tensor_like(**{**cos_fields, "dtype": torch.float32})
        assert not plan_rotary(q, promoted, promoted, ALL)
        trainable = tensor_like(**{**cos_fields, "requires_grad": True})
        assert not plan_rotary(q, trainable, trainable, ALL)
        other_batch = tensor_like(**{**cos_fields, "shape": (3, 1, 5, 64)})
        assert not plan_rotary(q, other_batch, other_batch, ALL)


class TestBinding:
    def test_the_installed_transformers_is_the_pinned_one(self) -> None:
        """The drift canary: a bump that changes a mirrored line fails here
        (and unbinds the kernel), so the copy is re-examined."""
        assert (
            canonical_source(modeling.Qwen3_5MoeRMSNorm.forward)
            == PINNED["norm_forward"]
        )
        assert (
            canonical_source(getattr(modeling.Qwen3_5MoeRMSNorm, "_norm"))
            == PINNED["norm_norm"]
        )
        assert (
            canonical_source(modeling.Qwen3_5MoeRMSNormGated.forward)
            == PINNED["gated_forward"]
        )
        assert canonical_source(modeling.rotate_half) == PINNED["rotate_half"]
        assert canonical_source(modeling.apply_rotary_pos_emb) == PINNED["apply_rotary"]

    def test_targets_of_the_tiny_moe(self, qwen35moe_bundle: ModelBundle) -> None:
        targets = targets_of(qwen35moe_bundle.model)
        assert targets.norms == (modeling.Qwen3_5MoeRMSNorm,)
        assert targets.gated == (modeling.Qwen3_5MoeRMSNormGated,)
        assert targets.rotary == (modeling,)
        # scanned once per model object
        assert targets_of(qwen35moe_bundle.model) is targets

    def test_installs_restores_and_changes_nothing_off_cuda(
        self, qwen35moe_bundle: ModelBundle
    ) -> None:
        model = qwen35moe_bundle.model
        ids = qwen35moe_bundle.tokenizer(["the quick brown fox"], return_tensors="pt")
        original_norm = modeling.Qwen3_5MoeRMSNorm.__dict__["forward"]
        original_gated = modeling.Qwen3_5MoeRMSNormGated.__dict__["forward"]
        original_rotary = modeling.apply_rotary_pos_emb
        with torch.no_grad():
            base = model(**ids).logits
            with fused_norm_path(model, ALL):
                norm_fwd = modeling.Qwen3_5MoeRMSNorm.forward
                gated_fwd = modeling.Qwen3_5MoeRMSNormGated.forward
                assert getattr(norm_fwd, "__wrapped__") is original_norm
                assert getattr(gated_fwd, "__wrapped__") is original_gated
                rotary = modeling.apply_rotary_pos_emb
                assert getattr(rotary, "__wrapped__") is original_rotary
                bound = model(**ids).logits
        assert modeling.Qwen3_5MoeRMSNorm.__dict__["forward"] is original_norm
        assert modeling.Qwen3_5MoeRMSNormGated.__dict__["forward"] is original_gated
        assert modeling.apply_rotary_pos_emb is original_rotary
        assert torch.equal(base, bound)

    def test_a_nested_entry_binds_once(self, qwen35moe_bundle: ModelBundle) -> None:
        """An enclosing entry's dispatchers are left alone: one plan per call,
        and the unwind lands on the library's functions."""
        model = qwen35moe_bundle.model
        original = modeling.Qwen3_5MoeRMSNorm.__dict__["forward"]
        original_rotary = modeling.apply_rotary_pos_emb
        with fused_norm_path(model, ALL):
            outer = modeling.Qwen3_5MoeRMSNorm.__dict__["forward"]
            outer_rotary = modeling.apply_rotary_pos_emb
            with fused_norm_path(model, ALL):
                assert modeling.Qwen3_5MoeRMSNorm.__dict__["forward"] is outer
                assert modeling.apply_rotary_pos_emb is outer_rotary
            assert modeling.Qwen3_5MoeRMSNorm.__dict__["forward"] is outer
            assert modeling.apply_rotary_pos_emb is outer_rotary
        assert modeling.Qwen3_5MoeRMSNorm.__dict__["forward"] is original
        assert modeling.apply_rotary_pos_emb is original_rotary

    def test_a_scan_under_a_live_binding_sees_the_pinned_sources(
        self, qwen35moe_bundle: ModelBundle
    ) -> None:
        """``targets_of`` reads the bound names; under an enclosing entry
        those are dispatchers, and a model first scanned then must find the
        same targets rather than cache an empty set for good."""
        model = qwen35moe_bundle.model
        expected = targets_of(model)
        assert expected.any
        memo = fused_norms._TARGETS  # pyright: ignore[reportPrivateUsage]
        with fused_norm_path(model, ALL):
            memo.pop(model, None)
            assert targets_of(model) == expected

    def test_an_empty_option_set_installs_nothing(
        self, qwen35moe_bundle: ModelBundle, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        original = modeling.Qwen3_5MoeRMSNorm.__dict__["forward"]
        monkeypatch.setenv(ENV_FUSED_NORMS, "off")
        with fused_norm_path(qwen35moe_bundle.model):
            assert modeling.Qwen3_5MoeRMSNorm.__dict__["forward"] is original

    def test_a_model_without_the_pinned_classes_is_left_alone(
        self, llama_bundle: ModelBundle
    ) -> None:
        targets = targets_of(llama_bundle.model)
        assert not targets.any
