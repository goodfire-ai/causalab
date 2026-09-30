"""The styles over plain tensors and any ``Collective`` (the package
docstring's second tier).

[`FragmentStyles`][] installs the partition of each style on ordinary
parameters — this rank's chunk of every sharded weight, cut by
[`partition_of`][] from a model held whole — and wraps each module's
forward with the autograd pairs of ``parallel/autograd.py`` and
``partial_gradient.summed_over`` over the collective it was built with. The
arithmetic is transformers' own, spelled without DTensor:

* a **colwise** projection runs on its rows and hands back this rank's
  columns of the output; its input's gradient is a partial sum, summed on
  backward (DTensor's ``from_local`` of a replicated input does the same);
* a **rowwise** projection runs on its columns of the input and all-reduce-
  sums the partial output, the bias added once after the sum, the backward
  the identity (DTensor keeps a replicated gradient rather than re-partition
  it);
* a **gathered** output is all-gathered along the feature axis, the backward
  this rank's slice;
* the **experts** module's hidden input has its gradient summed, the routing
  weights' too under tensor parallelism, and its output is all-reduce-summed;
* the **router**'s scores are zeroed and its ids remapped on the slots of
  other ranks' experts (``fragments.remap_routing``, entry for entry
  transformers' ``EpRouterParallel``), the input gradient summed under
  expert parallelism (§7);
* a **replicated** norm's parameter gradients are summed on backward.

Every collective is a method of the collective, so under the tests'
``SimulatedWorld`` a forward and a backward of a sharded model run on one
thread with fixed-order reductions, and under ``gloo`` the same code runs
on real ranks. The conformance suite holds this tier and the DTensor tier
to one set of programs.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, ClassVar, Mapping

import torch

from causalab.neural.engines.pytorch_hooks.experts_path import EXPERT_PARALLEL_MARK
from causalab.neural.engines.pytorch_hooks.kv_replication import KvReplicated
from causalab.neural.engines.pytorch_hooks.partial_gradient import summed_over
from causalab.neural.engines.pytorch_hooks.styles import (
    Group,
    Style,
    StyleError,
    partition_of,
)
from causalab.neural.shared.parallel.autograd import gather_for_edit, sum_for_edit
from causalab.neural.shared.parallel.collective import Collective
from causalab.neural.shared.parallel.fragments import expert_slot_mask, remap_routing
from causalab.protocol.registry import KV_REPLICATED, PlanRow

__all__ = ["FragmentStyles"]


class _Fragment:
    """The shared half: the partition installed on plain parameters, the
    group checked against the collective, no forward wrap."""

    name: ClassVar[str]

    def __init__(self, collective: Collective) -> None:
        self.collective = collective

    def _check(self, group: Group) -> None:
        rank, size = self.collective.rank(group.axis), self.collective.size(group.axis)
        if (rank, size) != (group.rank, group.size):
            raise StyleError(
                f"{self.name}: the sharding places this rank at {group.rank} of "
                f"{group.size} on the {group.axis} axis, but the collective is rank "
                f"{rank} of {size} there"
            )

    def validate(
        self, module: torch.nn.Module, parameter: str, group: Group, *, path: str
    ) -> None:
        self._check(group)

    def shard(self, module: torch.nn.Module, parameter: str, group: Group) -> None:
        self._check(group)
        whole = module._parameters.get(parameter)  # pyright: ignore[reportPrivateUsage]
        if whole is None:
            return
        partition = partition_of(self.name, whole.ndim, parameter)
        if partition.whole:
            return
        local = partition.local(whole.detach(), group.rank, group.size)
        module._parameters[parameter] = torch.nn.Parameter(  # pyright: ignore[reportPrivateUsage]
            local, requires_grad=whole.requires_grad
        )

    def install(
        self, module: torch.nn.Module, group: Group, *, expert_parallel: bool
    ) -> None:
        self._check(group)


def _wrap(
    module: torch.nn.Module,
    transform: Callable[[Callable[..., Any], tuple[Any, ...], dict[str, Any]], Any],
) -> None:
    """Replace ``module.forward`` by ``transform(original, args, kwargs)``."""
    original = module.forward

    def forward(*args: Any, **kwargs: Any) -> Any:
        return transform(original, args, kwargs)

    module.forward = forward


def _signature(module: torch.nn.Module) -> inspect.Signature:
    """The forward's signature, for a wrap that reads or replaces an argument
    by its *name* however the caller spelled the call — positionally or by
    keyword. The DTensor tier has no such dependence (its placements travel
    with the tensors), so this tier must not either: a wrap reading
    ``args[i]`` would silently skip a keyword-spelled argument. A forward
    taking ``*args`` or ``**kwargs`` cannot be bound and is refused by name.

    Raises:
        StyleError: a variadic forward.
    """
    signature = inspect.signature(module.forward)
    for parameter in signature.parameters.values():
        if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            raise StyleError(
                f"{type(module).__name__}.forward takes {parameter}: the fragment "
                "tier binds a forward's arguments by name and cannot bind a "
                "variadic one"
            )
    return signature


class _Colwise(_Fragment):
    """Rows of the weight; the input gradient summed; the output this
    rank's columns."""

    name = "colwise"

    def install(
        self, module: torch.nn.Module, group: Group, *, expert_parallel: bool
    ) -> None:
        self._check(group)
        axis, collective = group.axis, self.collective

        def transform(original: Any, args: tuple[Any, ...], kwargs: dict) -> Any:
            return original(summed_over(args[0], axis, collective), *args[1:], **kwargs)

        _wrap(module, transform)


class _PackedColwise(_Colwise):
    """As ``colwise``, the rows two interleaved halves (gate, up)."""

    name = "packed_colwise"


class _GatherOutput(_Fragment):
    """Rows of the weight; the input gradient summed; the output
    all-gathered along the feature axis."""

    name = "colwise_gather_output"

    def validate(
        self, module: torch.nn.Module, parameter: str, group: Group, *, path: str
    ) -> None:
        self._check(group)
        meta = module._parameters.get(parameter)  # pyright: ignore[reportPrivateUsage]
        if meta is None or meta.ndim == 0:
            return
        width = meta.shape[
            partition_of(self.name, meta.ndim, parameter).axis(meta.ndim)
        ]
        if width % group.size:
            raise ValueError(
                f"The output size of `{path.rsplit('.', 1)[0]}` ({width}) must be "
                f"divisible by the tensor parallel size ({group.size}) when "
                "gathering a colwise output."
            )

    def install(
        self, module: torch.nn.Module, group: Group, *, expert_parallel: bool
    ) -> None:
        self._check(group)
        axis, collective = group.axis, self.collective

        def transform(original: Any, args: tuple[Any, ...], kwargs: dict) -> Any:
            out = original(summed_over(args[0], axis, collective), *args[1:], **kwargs)
            return gather_for_edit(out, -1, axis, collective)

        _wrap(module, transform)


class _Rowwise(_Fragment):
    """Columns of the weight, the bias whole; the partial output
    all-reduce-summed, the bias added once after."""

    name = "rowwise"

    def install(
        self, module: torch.nn.Module, group: Group, *, expert_parallel: bool
    ) -> None:
        self._check(group)
        axis, collective = group.axis, self.collective
        parameters: Mapping[str, Any] = module._parameters  # pyright: ignore[reportPrivateUsage]

        # The bias leaves the module for the one wrapped call and returns in
        # the ``finally``, so ``original`` computes the bias-free partial
        # product and the bias is added once after the all-reduce. That is
        # process-global state on the module for the window of the call,
        # and it rests on two properties of this style's callers, neither
        # asserted by the wrap: ``original`` — ``Linear``'s forward — issues
        # no collective, so a simulated rank cannot hand off inside the
        # window and no peer runs while the bias is out; and no rank shares
        # a module object with another (``shard`` gave each its own chunk),
        # so no other thread reads the module meanwhile. The bias is read
        # per call, so a parameter swapped between two calls is restored as
        # the tensor that was current for the call.
        def transform(original: Any, args: tuple[Any, ...], kwargs: dict) -> Any:
            bias = parameters.get("bias")
            if bias is not None:
                module._parameters["bias"] = None  # pyright: ignore[reportPrivateUsage]
            try:
                partial = original(*args, **kwargs)
            finally:
                if bias is not None:
                    module._parameters["bias"] = bias  # pyright: ignore[reportPrivateUsage]
            total = sum_for_edit(partial, axis, collective)
            return total if bias is None else total + bias

        _wrap(module, transform)


class _Replicated(_Fragment):
    """The whole weight on every rank; its gradients summed on backward."""

    name = "replicated_with_grad_allreduce"

    def install(
        self, module: torch.nn.Module, group: Group, *, expert_parallel: bool
    ) -> None:
        self._check(group)
        axis, collective = group.axis, self.collective

        def sum_gradients(mod: Any, grad_input: Any, grad_output: Any) -> None:
            for parameter in mod.parameters(recurse=False):
                if parameter.grad is not None:
                    parameter.grad = collective.all_reduce_sum(parameter.grad, axis)

        module.register_full_backward_hook(sum_gradients)


class _GroupedGemm(_Fragment):
    """The experts dimension chunked; the module's ``num_experts`` becomes
    the local count, so its forward and the router's sentinel agree."""

    name = "grouped_gemm"

    def validate(
        self, module: torch.nn.Module, parameter: str, group: Group, *, path: str
    ) -> None:
        self._check(group)
        meta = module._parameters.get(parameter)  # pyright: ignore[reportPrivateUsage]
        if meta is None or not hasattr(module, "num_experts"):
            return
        if meta.shape[0] % group.size:
            raise ValueError(
                f"Cannot evenly shard {meta.shape[0]} experts across {group.size} "
                "expert-parallel ranks."
            )

    def shard(self, module: torch.nn.Module, parameter: str, group: Group) -> None:
        whole = module._parameters.get(parameter)  # pyright: ignore[reportPrivateUsage]
        super().shard(module, parameter, group)
        if whole is not None and hasattr(module, "num_experts"):
            module.num_experts = whole.shape[0] // group.size


class _MoeExperts(_Fragment):
    """The experts module: the hidden input's gradient summed (the routing
    weights' too under tensor parallelism), the output all-reduce-summed."""

    name = "moe_tp_experts"

    def install(
        self, module: torch.nn.Module, group: Group, *, expert_parallel: bool
    ) -> None:
        self._check(group)
        axis, collective = group.axis, self.collective
        if expert_parallel:
            # the router now writes sentinel ids for other ranks' experts,
            # and the weight is local: say so, since the lean experts path
            # cannot read it off the shapes
            setattr(module, EXPERT_PARALLEL_MARK, True)
        # the experts forward is ``(hidden_states, top_k_index, top_k_weights)``:
        # the first and the third are read by name (``_signature``)
        signature = _signature(module)
        names = list(signature.parameters)
        hidden_name = names[0]
        weights_name = names[2] if len(names) >= 3 else None

        def transform(original: Any, args: tuple[Any, ...], kwargs: dict) -> Any:
            bound = signature.bind(*args, **kwargs)
            arguments = bound.arguments
            arguments[hidden_name] = summed_over(
                arguments[hidden_name], axis, collective
            )
            if (
                weights_name is not None
                and weights_name in arguments
                and not expert_parallel
            ):
                arguments[weights_name] = summed_over(
                    arguments[weights_name], axis, collective
                )
            out = original(*bound.args, **bound.kwargs)
            return None if out is None else sum_for_edit(out, axis, collective)

        _wrap(module, transform)


class _EpRouter(_Fragment):
    """The router runs replicated; its scores are zeroed and its ids
    remapped on the slots of other ranks' experts, and under expert
    parallelism its input gradient is summed over the group (§7)."""

    name = "ep_router"

    def install(
        self, module: torch.nn.Module, group: Group, *, expert_parallel: bool
    ) -> None:
        self._check(group)
        axis, collective = group.axis, self.collective
        rank, size = group.rank, group.size
        signature = _signature(module)
        hidden_name = next(iter(signature.parameters))

        def transform(original: Any, args: tuple[Any, ...], kwargs: dict) -> Any:
            bound = signature.bind(*args, **kwargs)
            if expert_parallel:
                bound.arguments[hidden_name] = summed_over(
                    bound.arguments[hidden_name], axis, collective
                )
            logits, scores, indices, *extra = original(*bound.args, **bound.kwargs)
            num_experts = _num_experts(module)
            if num_experts % size:
                raise ValueError(
                    f"num_experts must be divisible by ep_size: {num_experts} % "
                    f"{size} != 0"
                )
            owned = expert_slot_mask(indices, num_experts, rank, size)
            scores = scores.masked_fill(~owned, 0.0)
            indices = remap_routing(indices, num_experts, rank, size)
            return logits, scores, indices, *extra

        _wrap(module, transform)


def _num_experts(module: torch.nn.Module) -> int:
    count = getattr(module, "num_experts", None)
    if count is None:
        count = getattr(getattr(module, "config", None), "num_experts", None)
    if count is None:
        raise AttributeError(
            f"Router module {type(module).__name__} is missing `num_experts` and "
            "`config.num_experts`"
        )
    return int(count)


_STYLES: dict[str, type[_Fragment]] = {
    cls.name: cls
    for cls in (
        _Colwise,
        _PackedColwise,
        _GatherOutput,
        _Rowwise,
        _Replicated,
        _GroupedGemm,
        _MoeExperts,
        _EpRouter,
    )
}


class FragmentStyles:
    """The [`Styles`][] over plain tensors and ``collective`` (module
    docstring)."""

    def __init__(self, collective: Collective) -> None:
        self.collective = collective

    def style(self, row: PlanRow) -> Style:
        if row.style == KV_REPLICATED:
            return KvReplicated(row.repeat, self.collective)
        cls = _STYLES.get(row.style)
        if cls is None:
            raise StyleError(
                f"parallel style {row.style!r} has no fragment implementation; "
                f"the styles are {sorted([*_STYLES, KV_REPLICATED])}"
            )
        return cls(self.collective)
