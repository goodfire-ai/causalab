"""The engine's grouped-experts forward (``pytorch_hooks/experts_path.py``).

Transformers' ``grouped_mm_experts_forward`` carries two sentinel masks for
expert parallelism whose backward runs — a full ``(S, hidden)`` fill each,
per MoE layer — in a single-process model where the masks are all-``False``.
The engine dispatches to its own copy of the function without them, and with
the un-sort's backward as the gather it is. What this pins:

* on the tiny MoE's own experts module, the copy and the library function
  agree **to the bit** — output, input gradient, both expert-weight gradients
  — under a downstream that weights every element differently, in fp32 and
  in bf16 (the production dtype; the fixture has no expert biases, so the
  mirrored bias lines run only through the drift canary);
* a model that can route to a sentinel is handed to the library function
  unchanged, and the predicate fails *safe*: the expert-parallel flag, a
  module whose id space is not its weights' expert axis (what transformers'
  sharding leaves behind), or a module on which neither can be read;
* the un-sort's custom backward equals autograd's on a permutation and is not
  a valid gather backward on an index with a repeat (the reason it is used
  only for the un-sort);
* the dispatch entry is installed for the duration of an engine forward —
  every MoE layer's experts call goes through the copy — and restored after,
  with the experts-interface taps still counting the two grouped linears;
* the **drift canaries**: a digest of the library function's source against
  the set the copy was checked against, so *any* edit to it — a line added
  as much as one moved — fails here and sends a reader back to the copy; the
  mirrored lines named alongside it, so the failure says what moved; and the
  sentinel's origin (the config flag, the sharding rewrite of
  ``num_experts``, the router's fill), so the predicate's premises are
  re-checked by the same bump.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import transformers
import transformers.integrations.moe as moe

from causalab.neural.engines.pytorch_hooks import experts_path
from causalab.neural.engines.pytorch_hooks.experts_path import (
    _PermuteRows,
    lean_experts_path,
    lean_grouped_mm_forward,
    may_route_to_sentinels,
)
from causalab.neural.engines.pytorch_hooks.loading import ModelBundle
from causalab.protocol.schema import PROTOCOL_VERSION

from ._drive import base_data_section, executor_for
from tests.protocol._docs import saved
from .test_sites_round3_moe_interior import MOE_LAYER, TEXT

pytestmark = pytest.mark.smoke

#: sha256 of ``grouped_mm_experts_forward``'s source (trailing whitespace
#: stripped per line) in every transformers release the copy was checked
#: against. Keyed by digest, not version: a bump that leaves the function
#: byte-identical stays green; one that changes it fails the canary below —
#: re-check ``lean_grouped_mm_forward`` line by line against the new source,
#: then add the digest.
ACCEPTED_LIBRARY_DIGESTS = {
    "feb6f2016c7996abe84285cfc63c55dee7b8162f7c25ecc11d3ddca6e91567eb",  # 5.16.1
}


def _experts(bundle: ModelBundle) -> Any:
    return bundle.model.model.layers[MOE_LAYER].mlp.experts


def _routing(
    experts: Any, tokens: int, seed: int, dtype: torch.dtype = torch.float32
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A hidden batch and a routing table the way the block's router shapes
    them: top-k expert ids per token and softmax weights over the chosen."""
    generator = torch.Generator().manual_seed(seed)
    hidden = torch.randn(tokens, experts.hidden_dim, generator=generator)
    logits = torch.randn(tokens, experts.num_experts, generator=generator)
    top_k = experts.config.num_experts_per_tok
    scores, index = torch.topk(logits, top_k, dim=-1)
    return hidden.to(dtype), index, torch.softmax(scores, dim=-1).to(dtype)


def _downstream(out: torch.Tensor) -> torch.Tensor:
    """Every element weighted differently in fp32, so a mis-permutation cannot
    cancel. On the bf16 leg the cast's backward rounds the upstream gradient
    to bf16 (integers exact only to 256), so neighbouring weights can coincide
    there; parity on that leg rests on the element-wise ``torch.equal`` over
    the full output and gradient tensors, not on weight distinctness."""
    weights = torch.arange(1, out.numel() + 1, dtype=torch.float32).reshape(out.shape)
    return (out.float() * weights).sum()


def _run(
    fn: Any,
    experts: Any,
    hidden: torch.Tensor,
    index: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Output and the gradients of the input and both expert weights."""
    source = hidden.clone().requires_grad_(True)
    params = (experts.gate_up_proj, experts.down_proj)
    out = fn(experts, source, index, weights)
    grads = torch.autograd.grad(_downstream(out), (source, *params))
    return (out.detach(), *(g.detach() for g in grads))


@pytest.fixture()
def trainable_experts(qwen35moe_bundle: ModelBundle) -> Any:
    experts = _experts(qwen35moe_bundle)
    params = (experts.gate_up_proj, experts.down_proj)
    flags = [p.requires_grad for p in params]
    for p in params:
        p.requires_grad_(True)
    try:
        yield experts
    finally:
        for p, flag in zip(params, flags):
            p.requires_grad_(flag)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_copy_matches_the_library_to_the_bit(
    trainable_experts: Any, seed: int, dtype: torch.dtype
) -> None:
    # the fixture's experts carry no bias: the bias lines are mirrored but
    # only the drift canary exercises them
    assert not trainable_experts.has_bias
    experts = (
        trainable_experts
        if dtype is torch.float32
        else copy.deepcopy(trainable_experts).to(dtype)
    )
    hidden, index, weights = _routing(experts, tokens=7, seed=seed, dtype=dtype)
    ours = _run(lean_grouped_mm_forward, experts, hidden, index, weights)
    theirs = _run(moe.grouped_mm_experts_forward, experts, hidden, index, weights)
    names = ("output", "d_hidden", "d_gate_up_proj", "d_down_proj")
    for name, a, b in zip(names, ours, theirs):
        assert a.dtype == dtype, name
        assert torch.equal(a, b), name
        assert torch.isfinite(a).all(), name


class TestMayRouteToSentinels:
    def test_the_fixture_cannot(self, qwen35moe_bundle: ModelBundle) -> None:
        assert not may_route_to_sentinels(_experts(qwen35moe_bundle))

    def test_the_expert_parallel_flag_can(
        self, qwen35moe_bundle: ModelBundle, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        experts = _experts(qwen35moe_bundle)
        monkeypatch.setattr(
            experts.config,
            "distributed_config",
            SimpleNamespace(enable_expert_parallel=True),
            raising=False,
        )
        assert may_route_to_sentinels(experts)

    def test_a_sharded_id_space_can(
        self, qwen35moe_bundle: ModelBundle, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """What transformers' ``MoEParamShard`` leaves behind under EP:
        ``num_experts`` the per-rank count, the weight's expert axis the
        global one — with or without the flag on the config."""
        experts = _experts(qwen35moe_bundle)
        assert experts.num_experts == experts.gate_up_proj.shape[0]
        monkeypatch.setattr(experts, "num_experts", experts.num_experts // 2)
        assert may_route_to_sentinels(experts)

    def test_the_repository_expert_axis_can(
        self, qwen35moe_bundle: ModelBundle, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Shard-on-read leaves local weights, a matching local expert count
        and the expert-parallel mark needed for sentinel routing."""
        experts = _experts(qwen35moe_bundle)
        assert experts.num_experts == experts.gate_up_proj.shape[0]
        assert not may_route_to_sentinels(experts)
        monkeypatch.setattr(
            experts, experts_path.EXPERT_PARALLEL_MARK, True, raising=False
        )
        assert may_route_to_sentinels(experts)

    def test_an_unreadable_id_space_can(self) -> None:
        """No premise readable, no shortcut taken."""
        assert may_route_to_sentinels(SimpleNamespace())
        assert may_route_to_sentinels(
            SimpleNamespace(has_gate=True, gate_up_proj=torch.zeros(4, 2, 2))
        )
        assert may_route_to_sentinels(SimpleNamespace(has_gate=False, num_experts=4))
        assert not may_route_to_sentinels(
            SimpleNamespace(has_gate=False, up_proj=torch.zeros(4, 2, 2), num_experts=4)
        )


def test_an_expert_parallel_model_is_handed_to_the_library(
    qwen35moe_bundle: ModelBundle, monkeypatch: pytest.MonkeyPatch
) -> None:
    experts = _experts(qwen35moe_bundle)
    monkeypatch.setattr(
        experts.config,
        "distributed_config",
        SimpleNamespace(enable_expert_parallel=True),
        raising=False,
    )
    assert may_route_to_sentinels(experts)
    calls: list[int] = []
    real = moe.grouped_mm_experts_forward

    def spy(module: Any, *args: Any) -> torch.Tensor:
        calls.append(1)
        return real(module, *args)

    monkeypatch.setattr(moe, "grouped_mm_experts_forward", spy)
    hidden, index, weights = _routing(experts, tokens=5, seed=3)
    with torch.no_grad():
        out = lean_grouped_mm_forward(experts, hidden, index, weights)
        assert torch.equal(out, real(experts, hidden, index, weights))
    assert calls == [1]


class TestPermuteRows:
    def test_the_gradient_is_the_plain_index_s_on_a_permutation(self) -> None:
        torch.manual_seed(0)
        rows = torch.randn(9, 4, requires_grad=True)
        order = torch.randperm(9)
        inverse = torch.empty_like(order)
        inverse[order] = torch.arange(9)
        out = _PermuteRows.apply(rows, order, inverse)
        _downstream(out).backward()
        reference = rows.detach().clone().requires_grad_(True)
        _downstream(reference[order]).backward()
        assert torch.equal(out.detach(), rows.detach()[order])
        assert rows.grad is not None and reference.grad is not None
        assert torch.equal(rows.grad, reference.grad)

    def test_mutation_a_repeat_in_the_index_is_not_a_permutation(self) -> None:
        """Why the shortcut is confined to the un-sort: with a repeated row
        the gather-by-inverse drops a contribution the sorted accumulate
        keeps."""
        torch.manual_seed(1)
        rows = torch.randn(4, 3, requires_grad=True)
        order = torch.tensor([0, 2, 2, 1])
        inverse = torch.tensor([0, 3, 1, 3])  # one consistent inverse of it
        out = _PermuteRows.apply(rows, order, inverse)
        _downstream(out).backward()
        reference = rows.detach().clone().requires_grad_(True)
        _downstream(reference[order]).backward()
        assert rows.grad is not None and reference.grad is not None
        assert not torch.equal(rows.grad, reference.grad)


def _doc(component: str = "block_output") -> dict:
    return {
        "header": {"protocol_version": PROTOCOL_VERSION},
        "model": {"key": "test", "revision": "main"},
        "data": base_data_section(with_counterfactual=False),
        "method": {
            "sites": {"tap": {"component": component, "layers": [MOE_LAYER]}},
            "intervened_models": {"original": {"input": "base", "reads": ["r"]}},
            "reads": {"r": {"site": "tap", "pos": -1}},
            "save": [saved("r", "original", "a.safetensors")],
        },
    }


def test_the_entry_is_installed_for_the_context_and_restored() -> None:
    before = moe.ALL_EXPERTS_FUNCTIONS["grouped_mm"]
    with lean_experts_path():
        assert moe.ALL_EXPERTS_FUNCTIONS["grouped_mm"] is lean_grouped_mm_forward
    assert moe.ALL_EXPERTS_FUNCTIONS["grouped_mm"] is before


def test_every_moe_layer_of_an_engine_forward_runs_the_copy(
    qwen35moe_bundle: ModelBundle, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[Any] = []
    real = lean_grouped_mm_forward

    def spy(module: Any, *args: Any) -> torch.Tensor:
        calls.append(module)
        return real(module, *args)

    monkeypatch.setattr(experts_path, "lean_grouped_mm_forward", spy)
    before = moe.ALL_EXPERTS_FUNCTIONS["grouped_mm"]
    value = executor_for(_doc(), qwen35moe_bundle, base_texts=[TEXT]).dense_value("r")
    assert torch.isfinite(value).all()
    moe_layers = [
        block.mlp.experts
        for block in qwen35moe_bundle.blocks
        if hasattr(block.mlp, "experts")
    ]
    assert moe_layers and calls == moe_layers
    assert moe.ALL_EXPERTS_FUNCTIONS["grouped_mm"] is before


def test_the_experts_interface_taps_wrap_the_copy(
    qwen35moe_bundle: ModelBundle, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The taps capture the dispatch entry at their entry and count its two
    grouped linears: an interior read through them still resolves, and it
    is the copy they wrapped."""
    calls: list[int] = []
    real = lean_grouped_mm_forward

    def spy(module: Any, *args: Any) -> torch.Tensor:
        calls.append(1)
        return real(module, *args)

    monkeypatch.setattr(experts_path, "lean_grouped_mm_forward", spy)
    value = executor_for(
        _doc("expert_activation"), qwen35moe_bundle, base_texts=[TEXT]
    ).dense_value("r")
    assert value.dim() == 3 and torch.isfinite(value).all()
    assert calls


class TestDriftCanary:
    def test_the_library_function_is_the_one_this_module_mirrors(self) -> None:
        source = inspect.getsource(moe.grouped_mm_experts_forward)
        normalized = "\n".join(line.rstrip() for line in source.splitlines())
        digest = hashlib.sha256(normalized.encode()).hexdigest()
        version = transformers.__version__
        assert digest in ACCEPTED_LIBRARY_DIGESTS, (
            f"transformers {version}: grouped_mm_experts_forward (sha256 {digest}) "
            "is not a function experts_path.lean_grouped_mm_forward was checked "
            "against. Re-check the copy against it, then add the digest."
        )

    def test_the_mirrored_lines_are_where_the_copy_says(self) -> None:
        """Alongside the digest: what moved, when it does."""
        source = inspect.getsource(moe.grouped_mm_experts_forward)
        assert source.count("masked_fill_(sentinel_mask, 0.0)") == 2
        assert "sentinel_mask = (expert_ids_g >= self.num_experts)" in source
        assert source.count("_grouped_linear(") == 2
        assert source.count("_bias[expert_ids_g] if self.has_bias else None") == 3
        assert "hidden_states[perm // num_top_k]" in source
        assert "weighted_out[inv_perm]" in source
        assert ".view(num_tokens, num_top_k, hidden_dim).sum(dim=1)" in source
        assert "torch.histc(" in source

    def test_the_sentinel_still_comes_from_where_the_predicate_looks(self) -> None:
        """``may_route_to_sentinels`` reads two premises off transformers:
        the ``enable_expert_parallel`` flag on the distributed config, and
        the sharding step that rewrites ``module.num_experts`` to the
        per-rank count (the router's sentinel is that count)."""
        from transformers.distributed import tensor_parallel
        from transformers.distributed.configuration_utils import DistributedConfig

        assert "enable_expert_parallel" in inspect.getsource(DistributedConfig)
        shard = inspect.getsource(tensor_parallel.MoEParamShard.shard_param)
        assert (
            "module.num_experts = global_num_experts // expert_parallel_size" in shard
        )
        router = inspect.getsource(
            tensor_parallel.EpRouterParallel.transform_output_post_forward
        )
        assert "num_local_experts = num_experts // ep_size" in router
        assert "masked_fill(router_indices == -1, num_local_experts)" in router


def test_interleaved_enterers_restore_the_entry_once_the_last_leaves() -> None:
    """Two simulated ranks are two threads in one process, each entering the
    manager around its own forward; when they leave in the order they came
    (not the nested order) the entry must still end as it was found, and be
    the copy for as long as either is inside."""
    import threading

    before = moe.ALL_EXPERTS_FUNCTIONS["grouped_mm"]
    a_in, b_in, a_out, b_out = (threading.Event() for _ in range(4))
    seen: dict[str, Any] = {}

    def rank_a() -> None:
        with lean_experts_path():
            a_in.set()
            b_in.wait()
        seen["after_a_left"] = moe.ALL_EXPERTS_FUNCTIONS["grouped_mm"]
        a_out.set()

    def rank_b() -> None:
        a_in.wait()
        with lean_experts_path():
            b_in.set()
            a_out.wait()
            seen["while_b_inside"] = moe.ALL_EXPERTS_FUNCTIONS["grouped_mm"]
        b_out.set()

    threads = [threading.Thread(target=rank_a), threading.Thread(target=rank_b)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    assert b_out.is_set(), "the ranks did not both leave"
    assert seen["after_a_left"] is lean_grouped_mm_forward
    assert seen["while_b_inside"] is lean_grouped_mm_forward
    assert moe.ALL_EXPERTS_FUNCTIONS["grouped_mm"] is before


def test_a_nested_enterer_in_one_thread_restores_on_the_outermost_exit() -> None:
    before = moe.ALL_EXPERTS_FUNCTIONS["grouped_mm"]
    with lean_experts_path():
        with lean_experts_path():
            assert moe.ALL_EXPERTS_FUNCTIONS["grouped_mm"] is lean_grouped_mm_forward
        assert moe.ALL_EXPERTS_FUNCTIONS["grouped_mm"] is lean_grouped_mm_forward
    assert moe.ALL_EXPERTS_FUNCTIONS["grouped_mm"] is before


class TestActFnIsHooked:
    """The fused gate kernel skips the ``act_fn`` call it reproduces; the
    experts-interface taps read and edit the ``activation`` slot through a
    forward hook on that very call, so the plan must yield the gate to the
    module's own implementation while a hook is attached."""

    def test_a_forward_hook_on_act_fn_is_seen_and_its_removal_too(
        self, qwen35moe_bundle: ModelBundle
    ) -> None:
        experts = _experts(qwen35moe_bundle)
        assert experts_path.has_default_silu_gate(experts)
        assert not experts_path.act_fn_is_hooked(experts)
        handle = experts.act_fn.register_forward_hook(lambda m, i, o: o)
        try:
            assert experts_path.act_fn_is_hooked(experts)
        finally:
            handle.remove()
        assert not experts_path.act_fn_is_hooked(experts)

    def test_a_module_without_act_fn_is_not_hooked(self) -> None:
        assert not experts_path.act_fn_is_hooked(torch.nn.Linear(2, 2))
