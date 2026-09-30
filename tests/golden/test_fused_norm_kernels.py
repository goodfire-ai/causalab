"""The fused norm and rotary kernels on the device, held to the bit
(``pytorch_hooks/kernels/fused_norms.py``): the accelerator half of
``tests/neural/engines/pytorch_hooks/kernels/test_fused_norms.py``.

Three layers, each ``torch.equal`` against transformers' own modules and
function, forward and every gradient, on the workflow's shapes (the A3B
cohort: ``[1248, 2048]`` residual rows, ``[1248·16, 256]`` / ``[1248·2,
256]`` q / k heads, ``[1248·32, 128]`` DeltaNet output heads, ``[96, 16, 13,
256]`` attention with a 64-wide rotary) and small odd ones, in bf16 and
fp32:

1. **the ATen orders hold on this device** — the order-explicit references
   of ``norm_reference.py`` are the modules;
2. **each kernel is its reference** — through the autograd functions;
3. **the bound path is the model** — the tiny Qwen3.5-MoE's logits and the
   gradient into an input embedding with and without ``fused_norm_path``,
   and once more captured in a CUDA graph and replayed.

A failure prints the count of differing elements and the largest gap.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as modeling

from causalab.neural.engines.pytorch_hooks.kernels import fused_norms
from causalab.neural.engines.pytorch_hooks.kernels import norm_reference as ref
from causalab.neural.engines.pytorch_hooks.kernels import norm_triton as kernels
from causalab.neural.engines.pytorch_hooks.kernels.fused_norms import (
    fused_norm_path,
    plan_norm,
    plan_rotary,
)
from causalab.neural.shared.kernel_options import FusedNormOptions

pytestmark = [
    pytest.mark.golden,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
    pytest.mark.skipif(not kernels.available(), reason="requires Triton"),
]

DEVICE = "cuda"
DTYPES = (torch.bfloat16, torch.float32)
ALL = FusedNormOptions()

#: (rows, width) for the residual and head norms.
NORM_SHAPES = (
    (1248, 2048),
    (19968, 256),
    (2496, 256),
    (13, 2048),
    (37, 96),
    (1, 2048),  # a decode row: 512 lanes, the whole row in one step
    (2, 4096),
)
#: (rows, width) for the gated norm.
GATED_SHAPES = ((39936, 128), (13, 128), (40, 64), (1, 128), (3, 32))
#: (batch, heads, kv heads, positions, head dim, rotary dim).
ROTARY_SHAPES = (
    (96, 16, 2, 13, 256, 64),
    (1, 16, 2, 13, 256, 64),
    (2, 4, 2, 5, 32, 32),
)


def _assert_equal(name: str, ours: torch.Tensor, theirs: torch.Tensor) -> None:
    assert ours.shape == theirs.shape, f"{name}: shape {ours.shape} vs {theirs.shape}"
    assert ours.dtype == theirs.dtype, f"{name}: dtype {ours.dtype} vs {theirs.dtype}"
    if torch.equal(ours, theirs):
        return
    diff = (ours.float() - theirs.float()).abs()
    count = int((diff > 0).sum())
    raise AssertionError(
        f"{name}: {count} of {diff.numel()} elements differ, max |Δ| {diff.max().item():.3e}"
        f" (max |ref| {theirs.float().abs().max().item():.3e})"
    )


def _assert_one_node(out: torch.Tensor, *inputs: torch.Tensor) -> None:
    """The fused op is the only autograd node between each input and
    ``out``: the input's own producer (``AccumulateGrad`` for a leaf) is
    what the node's edge points at — no recorded view in between."""
    assert out.grad_fn is not None
    for i, tensor in enumerate(inputs):
        edge = out.grad_fn.next_functions[i][0]
        assert edge is not None
        expected = (
            "torch::autograd::AccumulateGrad"
            if tensor.grad_fn is None
            else tensor.grad_fn.name()
        )
        assert edge.name() == expected, f"input {i}: {edge.name()} vs {expected}"


def _norm(width: int, dtype: torch.dtype) -> Any:
    norm = modeling.Qwen3_5MoeRMSNorm(width, eps=1e-6).to(dtype).to(DEVICE)
    norm.weight.data.normal_()
    norm.requires_grad_(False)
    return norm


def _gated(width: int, dtype: torch.dtype) -> Any:
    norm = modeling.Qwen3_5MoeRMSNormGated(width, eps=1e-6).to(dtype).to(DEVICE)
    norm.weight.data.normal_()
    norm.requires_grad_(False)
    return norm


def _module_norm(
    norm: Any, x: torch.Tensor, grad: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    leaf = x.detach().clone().requires_grad_(True)
    y = norm(leaf)
    y.backward(grad)
    assert leaf.grad is not None
    return y.detach(), leaf.grad


def _module_gated(
    norm: Any, x: torch.Tensor, gate: torch.Tensor, grad: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    leaf_x = x.detach().clone().requires_grad_(True)
    leaf_g = gate.detach().clone().requires_grad_(True)
    o = norm(leaf_x, leaf_g)
    o.backward(grad)
    assert leaf_x.grad is not None and leaf_g.grad is not None
    return o.detach(), leaf_x.grad, leaf_g.grad


class TestNormOrders:
    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("rows,width", NORM_SHAPES)
    def test_reference_and_kernel_are_the_module(
        self, rows: int, width: int, dtype: torch.dtype
    ) -> None:
        torch.manual_seed(rows)
        norm = _norm(width, dtype)
        x = torch.randn(rows, width, device=DEVICE).to(dtype)
        grad = torch.randn(rows, width, device=DEVICE).to(dtype)
        y, dx = _module_norm(norm, x, grad)
        ours, rstd = ref.rms_norm_forward(x, norm.weight, norm.eps)
        _assert_equal("reference y", ours, y)
        _assert_equal(
            "reference dx", ref.rms_norm_backward(grad, x, norm.weight, rstd), dx
        )
        plan = plan_norm(x, norm.weight, options=ALL)
        assert plan is not None
        leaf = x.clone().requires_grad_(True)
        plan = plan_norm(leaf, norm.weight, options=ALL)
        assert plan is not None
        out = fused_norms.fused_rms_norm(leaf, norm.weight, norm.eps, plan)
        _assert_one_node(out, leaf)
        out.backward(grad)
        assert leaf.grad is not None
        _assert_equal("kernel y", out.detach(), y)
        _assert_equal("kernel dx", leaf.grad, dx)

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_the_attention_q_chunk_keeps_its_stride(self, dtype: torch.dtype) -> None:
        """``q`` is a chunk of ``q_proj``'s output: rows of 256 in a 512-wide
        buffer, viewed ``[B, T, H, 256]``."""
        torch.manual_seed(7)
        norm = _norm(256, dtype)
        full = torch.randn(96, 13, 16, 512, device=DEVICE).to(dtype)
        q = torch.chunk(full, 2, dim=-1)[0]
        grad = torch.randn(96, 13, 16, 256, device=DEVICE).to(dtype)
        y, dq = _module_norm(norm, q, grad)
        leaf = q.detach().clone()
        chunk = torch.chunk(full.clone().requires_grad_(True), 2, dim=-1)[0]
        plan = plan_norm(chunk, norm.weight, options=ALL)
        assert plan is not None and plan.row_stride == 512
        out = fused_norms.fused_rms_norm(chunk, norm.weight, norm.eps, plan)
        # the chunk's own ``SplitBackward0`` feeds the kernel: no recorded
        # view in between (its backward would zero-fill the 512-wide base)
        _assert_one_node(out, chunk)
        (dchunk,) = torch.autograd.grad(out, chunk, grad)
        _assert_equal("kernel y", out.detach(), y)
        _assert_equal("kernel dq", dchunk, dq)
        del leaf

    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("rows,width", GATED_SHAPES)
    def test_gated_reference_and_kernel_are_the_module(
        self, rows: int, width: int, dtype: torch.dtype
    ) -> None:
        torch.manual_seed(rows + 1)
        norm = _gated(width, dtype)
        x = torch.randn(rows, width, device=DEVICE).to(dtype)
        gate = torch.randn(rows, width, device=DEVICE).to(dtype)
        grad = torch.randn(rows, width, device=DEVICE).to(dtype)
        o, dx, dgate = _module_gated(norm, x, gate, grad)
        ours, rstd = ref.gated_rms_norm_forward(
            x, gate, norm.weight, norm.variance_epsilon
        )
        _assert_equal("reference o", ours, o)
        ref_dx, ref_dgate = ref.gated_rms_norm_backward(
            grad, x, gate, norm.weight, rstd
        )
        _assert_equal("reference dx", ref_dx, dx)
        leaf_x = x.clone().requires_grad_(True)
        leaf_g = gate.clone().requires_grad_(True)
        plan = plan_norm(
            leaf_x,
            norm.weight,
            leaf_g,
            options=ALL,
            activation=norm.activation,
        )
        assert plan is not None
        out = fused_norms.fused_gated_rms_norm(
            leaf_x, leaf_g, norm.weight, norm.variance_epsilon, plan
        )
        _assert_one_node(out, leaf_x, leaf_g)
        out.backward(grad)
        assert leaf_x.grad is not None and leaf_g.grad is not None
        _assert_equal("kernel o", out.detach(), o)
        _assert_equal("kernel dx", leaf_x.grad, dx)
        _assert_equal("kernel dgate", leaf_g.grad, dgate)
        # Plain torch cannot express the fma contraction of 1 + x*(1-sigma),
        # so reference dgate can differ from ATen by one ulp. The kernel's
        # libdevice.fma is exact; use an absolute bound because the
        # contracted term cancels to near zero in places.
        ulp = 2.0**-7 if dtype == torch.bfloat16 else 2.0**-21
        scale = float(dgate.float().abs().max())
        assert torch.allclose(
            ref_dgate.float(), dgate.float(), rtol=ulp, atol=ulp * scale
        )


class TestRotary:
    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize(
        "batch,heads,kv_heads,positions,head_dim,rot", ROTARY_SHAPES
    )
    def test_kernel_is_the_library_function(
        self,
        batch: int,
        heads: int,
        kv_heads: int,
        positions: int,
        head_dim: int,
        rot: int,
        dtype: torch.dtype,
    ) -> None:
        torch.manual_seed(batch + heads)
        q = torch.randn(batch, positions, heads, head_dim, device=DEVICE).to(dtype)
        k = torch.randn(batch, positions, kv_heads, head_dim, device=DEVICE).to(dtype)
        q, k = q.transpose(1, 2), k.transpose(1, 2)
        cos = torch.randn(batch, positions, rot, device=DEVICE).to(dtype)
        sin = torch.randn(batch, positions, rot, device=DEVICE).to(dtype)
        grad_q = torch.randn(q.shape, device=DEVICE).to(dtype)
        grad_k = torch.randn(k.shape, device=DEVICE).to(dtype)

        def run(fn: Any) -> tuple[torch.Tensor, ...]:
            lq = q.detach().clone().requires_grad_(True)
            lk = k.detach().clone().requires_grad_(True)
            qe, ke = fn(lq, lk, cos, sin)
            dq, dk = torch.autograd.grad((qe, ke), (lq, lk), (grad_q, grad_k))
            return qe.detach(), ke.detach(), dq, dk

        theirs = run(modeling.apply_rotary_pos_emb)
        assert plan_rotary(q, cos.unsqueeze(1), sin.unsqueeze(1), ALL)
        ours = run(fused_norms.rotary_dispatcher(ALL, modeling.apply_rotary_pos_emb))
        for name, a, b in zip(("q_embed", "k_embed", "dq", "dk"), ours, theirs):
            _assert_equal(name, a, b)


class TestBoundModel:
    """The tiny Qwen3.5-MoE on the device: logits and the gradient into the
    input embeddings with the fused path bound are the library's, eager and
    replayed from a CUDA graph."""

    @pytest.fixture(scope="class")
    def bundle(self) -> Any:
        from causalab.neural.engines.pytorch_hooks.loading import load_model

        return load_model("tiny-random/qwen3.5-moe", device=DEVICE)

    @pytest.fixture(scope="class")
    def inputs(self, bundle: Any) -> tuple[torch.Tensor, torch.Tensor]:
        ids = bundle.tokenizer(
            ["the quick brown fox jumps over", "a slow green turtle"],
            return_tensors="pt",
            padding=True,
        ).to(DEVICE)
        with torch.no_grad():
            embed = bundle.model.get_input_embeddings()(ids.input_ids)
        return embed, ids.attention_mask

    @staticmethod
    def _run(
        model: Any, embed: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        leaf = embed.detach().clone().requires_grad_(True)
        out = model(inputs_embeds=leaf, attention_mask=mask).logits
        (grad,) = torch.autograd.grad(out.float().square().sum(), leaf)
        return out.detach(), grad

    def test_bound_path_is_the_model(
        self, bundle: Any, inputs: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        embed, mask = inputs
        base_logits, base_grad = self._run(bundle.model, embed, mask)
        targets = fused_norms.targets_of(bundle.model)
        assert targets.norms and targets.gated and targets.rotary
        with fused_norm_path(bundle.model, ALL):
            logits, grad = self._run(bundle.model, embed, mask)
        _assert_equal("logits", logits, base_logits)
        _assert_equal("d embeddings", grad, base_grad)

    def test_fused_ops_are_capturable(self) -> None:
        """The three fused ops, forward and backward, captured in one CUDA
        graph on static inputs and replayed: the kernels launch under
        capture (no host sync, no dynamic shape) and replay the eager
        numbers. (The tiny model's own eager MoE routing is not capturable,
        so the whole model is checked through the engine's graph mode.)"""
        torch.manual_seed(11)
        dtype = torch.bfloat16
        norm = _norm(2048, dtype)
        gated = _gated(128, dtype)
        x = torch.randn(1248, 2048, device=DEVICE).to(dtype)
        gx = torch.randn(1248, 2048, device=DEVICE).to(dtype)
        h = torch.randn(39936, 128, device=DEVICE).to(dtype)
        gate = torch.randn(39936, 128, device=DEVICE).to(dtype)
        gh = torch.randn(39936, 128, device=DEVICE).to(dtype)
        q = torch.randn(96, 13, 16, 256, device=DEVICE).to(dtype).transpose(1, 2)
        k = torch.randn(96, 13, 2, 256, device=DEVICE).to(dtype).transpose(1, 2)
        cos = torch.randn(96, 13, 64, device=DEVICE).to(dtype)
        sin = torch.randn(96, 13, 64, device=DEVICE).to(dtype)
        gq = torch.randn(q.shape, device=DEVICE).to(dtype)
        gk = torch.randn(k.shape, device=DEVICE).to(dtype)
        rotary = fused_norms.rotary_dispatcher(ALL, modeling.apply_rotary_pos_emb)

        def step() -> tuple[torch.Tensor, ...]:
            lx = x.clone().requires_grad_(True)
            plan = plan_norm(lx, norm.weight, options=ALL)
            assert plan is not None
            y = fused_norms.fused_rms_norm(lx, norm.weight, norm.eps, plan)
            lh = h.clone().requires_grad_(True)
            lg = gate.clone().requires_grad_(True)
            gplan = plan_norm(lh, gated.weight, lg, options=ALL, activation="silu")
            assert gplan is not None
            o = fused_norms.fused_gated_rms_norm(
                lh, lg, gated.weight, gated.variance_epsilon, gplan
            )
            lq, lk = q.clone().requires_grad_(True), k.clone().requires_grad_(True)
            qe, ke = rotary(lq, lk, cos, sin)
            grads = torch.autograd.grad(
                (y, o, qe, ke), (lx, lh, lg, lq, lk), (gx, gh, gq, gk)
            )
            return (y.detach(), o.detach(), qe.detach(), ke.detach(), *grads)

        eager = step()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):  # the warm-up compiles the kernels
                step()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = step()
        for t in captured:
            t.zero_()
        graph.replay()
        torch.cuda.synchronize()
        names = ("y", "o", "q_embed", "k_embed", "dx", "dh", "dgate", "dq", "dk")
        for name, ours, theirs in zip(names, captured, eager):
            _assert_equal(f"replayed {name}", ours, theirs)
