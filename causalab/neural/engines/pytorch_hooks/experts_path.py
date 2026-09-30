"""Run the grouped expert path with equivalent local optimizations.

``lean_grouped_mm_forward`` removes sentinel masks when expert parallelism
is disabled. Every expert ID is then local. Permutation backward gathers
with the inverse permutation. Tests compare outputs and all gradients with
the Transformers function using exact equality. Expert-parallel models
use the library path.

``lean_experts_path`` overrides grouped dispatch for one engine forward,
then restores it. The executor enters it before expert taps so they wrap
the selected function and count its two grouped projections.

Eligible CUDA calls use fused sort, gather-backward, routing, reduction,
and gate kernels from ``kernels.moe_glue``. Each preserves the ATen order
specified in ``moe_glue_reference``. ``CAUSALAB_MOE_GLUE`` selects kernels;
unsupported shapes run the corresponding eager operations. A source
canary checks the Transformers lines this implementation mirrors.
"""

from __future__ import annotations

import contextlib
from typing import Any, Iterator

import torch

from causalab.neural.engines.pytorch_hooks.experts_registry import EntryInstall
from causalab.neural.engines.pytorch_hooks.kernels import moe_glue
from causalab.neural.shared.kernel_options import MoeGlueOptions

__all__ = [
    "EXPERT_PARALLEL_MARK",
    "act_fn_is_hooked",
    "has_default_silu_gate",
    "lean_experts_path",
    "lean_grouped_mm_forward",
    "may_route_to_sentinels",
]


def has_default_silu_gate(module: Any) -> bool:
    """Whether the experts module's gate is the library's default
    ``silu(gate) * up`` — ``_apply_gate`` is ``_default_apply_gate`` and
    ``act_fn`` is silu (``torch.nn.SiLU``, or transformers' own
    ``SiLUActivation`` that ``ACT2FN["silu"]`` builds, whose forward is
    ``F.silu``) — the one the fused gate kernel reproduces; a custom gate or
    another activation keeps the module's. Forward hooks on ``act_fn`` also
    keep the module call so intervention taps can read and replace its output.
    """
    import transformers.activations as activations
    import transformers.integrations.moe as moe

    if not getattr(module, "has_gate", False):
        return False
    apply_gate = getattr(module, "_apply_gate", None)
    default = getattr(apply_gate, "__func__", apply_gate) is moe._default_apply_gate
    silu = (torch.nn.SiLU, activations.SiLUActivation)
    activation = getattr(module, "act_fn", None)
    return (
        default
        and isinstance(activation, silu)
        and not activation._forward_hooks
        and not activation._forward_pre_hooks
    )


def act_fn_is_hooked(module: Any) -> bool:
    """Whether something watches the module's ``act_fn`` call — a forward
    hook on it, the experts-interface taps' way of reading and editing the
    ``activation`` slot (``experts_interface.py``). The fused gate kernel
    computes ``silu(gate) * up`` without calling ``act_fn`` (the same
    numbers, bit for bit), so under a hook it must yield to the module's own
    gate so the tap observes the activation call."""
    act_fn = getattr(module, "act_fn", None)
    hooks = getattr(act_fn, "_forward_hooks", None)
    return bool(hooks)


#: Set on an experts module by ``sharding.apply_sharding`` when it installs
#: ``MoeExpertsParallel`` over the repository's own ``expert`` axis
#: (``docs/model_parallelism.md`` §5, §6.3): the router then writes the
#: sentinel id into slots owned by other ranks, and the shard-on-read loader
#: has left the module's weight *local* — ``num_experts == weight.shape[0]``
#: — so the predicates below also need this explicit marker to detect
#: sentinel-bearing routing tables.
EXPERT_PARALLEL_MARK = "_causalab_expert_parallel"


def may_route_to_sentinels(module: Any) -> bool:
    """Whether the experts module's routing table can hold the EP sentinel
    id — ``num_local_experts``, the id the router writes into a slot owned
    by another rank, one past the module's local expert range. Decided
    positively, so a premise that cannot be read takes the library path
    (whose masks make a sentinel harmless) rather than this module's (which
    would leave the sentinel rows uninitialised):

    * the module carries [`EXPERT_PARALLEL_MARK`][] — the repository's own
      expert axis, whose loader keeps the weight local (above); or
    * the model was loaded with ``distributed_config.enable_expert_parallel``
      (read from the config the experts module carries — ``self.config``, set
      by transformers' ``use_experts_implementation`` decorator); or
    * the module's id space is not its weights' whole expert axis:
      transformers' ``MoEParamShard`` rewrites ``module.num_experts`` to the
      per-rank count while the parameter keeps its global expert dim, so
      ``num_experts != weight.shape[0]`` is a sharded module however it was
      asked for — and a module on which neither can be read counts as one.
    """
    if getattr(module, EXPERT_PARALLEL_MARK, False):
        return True
    config = getattr(module, "config", None)
    distributed = getattr(config, "distributed_config", None)
    if getattr(distributed, "enable_expert_parallel", False):
        return True
    weight_name = "gate_up_proj" if getattr(module, "has_gate", True) else "up_proj"
    weight = getattr(module, weight_name, None)
    num_experts = getattr(module, "num_experts", None)
    if weight is None or num_experts is None:
        return True
    return int(weight.shape[0]) != int(num_experts)


class _PermuteRows(torch.autograd.Function):
    """``rows[order]`` for a permutation ``order`` whose inverse is known:
    the backward is the gather by the inverse, not a sorted scatter-add."""

    @staticmethod
    def forward(
        ctx: Any, rows: torch.Tensor, order: torch.Tensor, inverse: torch.Tensor
    ) -> torch.Tensor:
        ctx.save_for_backward(inverse)
        return rows[order]

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        (inverse,) = ctx.saved_tensors
        return grad[inverse], None, None


def lean_grouped_mm_forward(
    self: torch.nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    """``grouped_mm_experts_forward`` (module docstring) without the sentinel
    masks, for a module that cannot route to a sentinel; the library function
    itself for one that can."""
    import transformers.integrations.moe as moe

    if may_route_to_sentinels(self):
        return moe.grouped_mm_experts_forward(
            self, hidden_states, top_k_index, top_k_weights
        )

    device = hidden_states.device
    num_top_k = top_k_index.size(-1)
    num_tokens = hidden_states.size(0)
    hidden_dim = hidden_states.size(-1)

    sample_weights = top_k_weights.reshape(-1)
    expert_ids = top_k_index.reshape(-1)
    # which glue runs fused (kernels/moe_glue.py): decided before any tensor
    # op, empty off CUDA or without Triton
    plan = moe_glue.plan_moe_glue(
        hidden_states=hidden_states,
        top_k_weights=top_k_weights,
        expert_dtype=self.down_proj.dtype,
        num_pairs=expert_ids.numel(),
        top_k=num_top_k,
        # a hooked act_fn is read through its call, which the fused gate
        # would skip (act_fn_is_hooked)
        default_silu_gate=has_default_silu_gate(self) and not act_fn_is_hooked(self),
        options=MoeGlueOptions.from_env(),
    )

    # S = tokens · top_k (token, slot) pairs, sorted by expert; inv_perm is
    # the un-sort; offsets the per-expert inclusive counts grouped_mm takes
    if plan.sort:
        perm, inv_perm, offsets = moe_glue.counting_sort(expert_ids, self.num_experts)
        expert_ids_g = expert_ids[perm] if self.has_bias else None
    else:
        expert_ids_g, perm = torch.sort(expert_ids)
        inv_perm = torch.empty_like(perm)
        inv_perm[perm] = torch.arange(perm.size(0), device=device)
        # histc rather than bincount, as the library does (CUDA-graph safe);
        # CPU/MPS histc wants a float input
        histc_input = (
            expert_ids_g.float()
            if device.type in ("cpu", "mps")
            else expert_ids_g.int()
        )
        tokens_per_expert = torch.histc(
            histc_input, bins=self.num_experts, min=0, max=self.num_experts - 1
        )
        offsets = torch.cumsum(tokens_per_expert, dim=0, dtype=torch.int32)
    # no sentinel: every id is below num_experts, so offsets[-1] == S and the
    # kernel writes every row — nothing to clamp, nothing to mask

    if plan.gather:
        selected_hidden_states_g = moe_glue.fused_gather(
            hidden_states, perm, inv_perm, num_top_k
        )
    else:
        selected_hidden_states_g = hidden_states[perm // num_top_k]

    if self.has_gate:
        selected_weights = self.gate_up_proj
        selected_biases = (
            self.gate_up_proj_bias[expert_ids_g] if self.has_bias else None
        )
    else:
        selected_weights = self.up_proj
        selected_biases = self.up_proj_bias[expert_ids_g] if self.has_bias else None

    # the fused [gate | up] (or plain up) projection, per expert
    proj_out = moe._grouped_linear(
        selected_hidden_states_g,
        selected_weights,
        offsets,
        bias=selected_biases,
        is_transposed=self.is_transposed,
    )
    if plan.gate:
        proj_out = moe_glue.fused_gate(proj_out)
    elif self.has_gate:
        proj_out = self._apply_gate(proj_out)
    else:
        proj_out = self.act_fn(proj_out)

    # the down-projection, per expert
    selected_biases = self.down_proj_bias[expert_ids_g] if self.has_bias else None
    proj_out = moe._grouped_linear(
        proj_out,
        self.down_proj,
        offsets,
        bias=selected_biases,
        is_transposed=self.is_transposed,
    )

    if plan.epilogue:
        # weight, un-sort, slot sum and cast in one launch each way; the
        # routing weights are read in token order, so no gather of them
        return moe_glue.fused_epilogue(
            proj_out, sample_weights, inv_perm, perm, num_top_k, hidden_states.dtype
        )

    sample_weights_g = sample_weights[perm]
    weighted_out = proj_out * sample_weights_g.unsqueeze(-1)

    # un-sort: a permutation, whose backward is the gather by `perm`
    weighted_out = _PermuteRows.apply(weighted_out, inv_perm, perm)

    # the library's deterministic reshape+sum over the slots (fp32 accumulate)
    final_hidden_states = weighted_out.view(num_tokens, num_top_k, hidden_dim).sum(
        dim=1
    )
    return final_hidden_states.to(hidden_states.dtype)


#: The one installation of the copy shared by every active enterer
#: (``experts_registry.py``): the ranks of a simulated world are threads of
#: one process, each entering the manager around its own forward and leaving
#: in its own order, so the entry is installed by the first active call and
#: restored by the last — never by a "previous" one thread captured while
#: another was already inside.
_LEAN = EntryInstall("grouped_mm", lambda: lean_grouped_mm_forward)


@contextlib.contextmanager
def lean_experts_path() -> Iterator[None]:
    """While active, ``"grouped_mm"`` dispatches to
    [`lean_grouped_mm_forward`][]; the entry that was there is put back
    when the last active caller leaves (restore-not-delete, as
    `.experts_interface_taps`). Re-entrant across threads and nesting:
    concurrent enterers share one installation."""
    with _LEAN.installed():
        yield
