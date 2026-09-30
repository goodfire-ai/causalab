"""The ``kv_replicated`` style (``docs/model_parallelism.md`` §6.6): tensor
parallelism above a GQA model's key-value head count.

transformers' ``colwise`` shards ``k_proj`` / ``v_proj`` over the KV heads,
so ``tp`` is bounded by ``num_kv_heads``. Above it the registry rewrites the
two rows to this style (``ParallelPlan.for_geometry``), and [`KvReplicated`][]
applies them as a [`Style`][causalab.neural.engines.pytorch_hooks.styles.Style] of the repository's own — the
same object in both tiers, since its arithmetic runs over the
[`Collective`][] and never
touches DTensor:

* ``shard`` leaves the weight (and bias) a **plain parameter**: whole on
  every rank of the tensor group, read whole by shard-on-read (a
  parameter whose partition is whole is read whole, ``shard_read.read_plan``),
  and whole in ``LoadReport.bytes_requested``;
* ``install`` wraps the module's forward to **narrow its output** to the
  one KV head this rank's query heads read: the width ``H_kv · d`` is
  ``H_kv = tp / repeat`` chunks and rank ``r`` keeps chunk ``r // repeat``
  (``protocol/kv_replication.py``, the map the geometry check admitted). The
  bias is inside the narrowed output, so a biased projection (Qwen2) is right
  by construction. On backward two sums over the group (§7): the parameter
  gradient — each rank's covers its own query heads' path — as
  ``ReplicatedWithGradAllReduce`` sums the norms', and the **input**
  gradient (``partial_gradient.summed_over``): the narrowed output's
  backward hands the residual this rank's query heads' path through its one
  held KV head, a partial the colwise query gets summed by DTensor and the
  router by the ``ep_router`` style. These partial gradients must be
  summed, not averaged;
* [`KvReplicated.repeat_locally`][] then divides the mixer's
  ``num_key_value_groups`` by ``repeat``: the library's own ``repeat_kv``
  (eager), ``enable_gqa`` (sdpa) and the flash kernels all read that
  attribute, so with one KV head held and ``H / tp`` query heads per rank
  the local repeat is ``H / tp`` and the attention function runs **unchanged**
  — no interception, nothing transcribed. The mixer infers its head count
  from the narrowed width (``.view(*, -1, head_dim)``), so the KV cache holds
  the one head too.

The tap side is the placement table's: the module's output is
``Sharded(axis, "tensor", repeat)`` — a repeated shard whose ``whole`` is the
model's ``H_kv`` heads on every rank (``placements.py``, ``fragments.py``).
"""

from __future__ import annotations

from typing import Any

import torch

from causalab.neural.engines.pytorch_hooks.partial_gradient import summed_over
from causalab.neural.engines.pytorch_hooks.styles import Group
from causalab.neural.shared.parallel.collective import Collective
from causalab.protocol.rules.errors import ProtocolError

__all__ = ["KvReplicated", "held_chunk"]


def held_chunk(rank: int, repeat: int) -> int:
    """The KV head (chunk of the projection's output) rank ``rank`` of the
    tensor group holds under a replication of ``repeat``: ``rank // repeat``
    — the same map ``KvHeads.kv_heads`` spells and ``fragment`` of the
    repeated shard takes. One seam, so a test can mutate it."""
    return rank // repeat


class KvReplicated:
    """The style object for a ``kv_replicated`` row of ``repeat`` (module
    docstring), its collectives through ``collective``. Built per row by a
    [`Styles`][causalab.neural.engines.pytorch_hooks.styles.Styles]: the repeat is the geometry's, so no singleton
    fits."""

    def __init__(self, repeat: int, collective: Collective) -> None:
        if isinstance(repeat, bool) or not isinstance(repeat, int) or repeat < 1:
            raise ValueError(
                f"KvReplicated: repeat must be a positive int, got {repeat!r}"
            )
        self.repeat = repeat
        self.collective = collective

    def validate(
        self, module: torch.nn.Module, parameter: str, group: Group, *, path: str
    ) -> None:
        """The output width is whole KV heads over the group: ``tp`` is
        ``repeat`` ranks per head, and the width is ``tp / repeat`` chunks.

        Raises:
            ValueError: the group is not whole repeats, or the width is not
                whole heads.
        """
        size = group.size
        if size % self.repeat:
            raise ValueError(
                f"a tensor group of {size} is not whole repeats of {self.repeat}"
            )
        meta = module._parameters.get(parameter)  # pyright: ignore[reportPrivateUsage]
        if meta is None or meta.ndim == 0:
            return
        width = meta.shape[0]  # ``Linear.weight`` is (out, in); its bias (out,)
        if width % (size // self.repeat):
            raise ValueError(
                f"the output width {width} of {path!r} is not "
                f"{size // self.repeat} whole KV heads (tp={size}, one head per "
                f"{self.repeat} ranks)"
            )

    def shard(self, module: torch.nn.Module, parameter: str, group: Group) -> None:
        """A plain parameter: whole on every rank, read whole."""
        return None

    def install(
        self, module: torch.nn.Module, group: Group, *, expert_parallel: bool = False
    ) -> None:
        """Narrow the output to this rank's KV head; sum the input and the
        parameter gradients over the group on backward (module docstring)."""
        heads = group.size // self.repeat
        chunk = held_chunk(group.rank, self.repeat)
        axis = group.axis
        collective = self.collective
        original_forward = module.forward

        def kv_forward(input: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
            out = original_forward(
                summed_over(input, axis, collective), *args, **kwargs
            )
            width = out.shape[-1]
            if width % heads:
                raise ProtocolError(
                    "P4",
                    f"--parallel.tensor: the replicated K/V projection emits {width} "
                    f"features, not {heads} whole KV heads",
                )
            per_head = width // heads
            return out.narrow(-1, chunk * per_head, per_head)

        module.forward = kv_forward

        def _all_reduce_grads(mod: Any, grad_input: Any, grad_output: Any) -> None:
            for parameter in mod.parameters(recurse=False):
                if parameter.grad is not None:
                    parameter.grad = collective.all_reduce_sum(parameter.grad, axis)

        module.register_full_backward_hook(_all_reduce_grads)

    def repeat_locally(self, mixer: Any, path: str) -> None:
        """Divide the mixer's ``num_key_value_groups`` by ``repeat``: the
        query heads per *held* KV head — ``H / tp`` with one head held.
        Idempotent per mixer (``apply_plan`` calls it once per mixer, not
        once per K/V row).

        Raises:
            ProtocolError: ``P4`` — the mixer has no ``num_key_value_groups``
                (a family whose attention repeats KV heads another way), or
                the groups are not whole repeats (the straddle the geometry
                check refuses; restated at the seam that would compute a
                wrong number).
        """
        groups = (
            getattr(mixer, "num_key_value_groups", None) if mixer is not None else None
        )
        if not isinstance(groups, int) or isinstance(groups, bool):
            raise ProtocolError(
                "P4",
                f"--parallel.tensor: the mixer at {path!r} "
                f"({type(mixer).__name__}) carries no integer num_key_value_groups; "
                "KV-head replication relies on the library repeating the held KV "
                "head by that attribute (docs/model_parallelism.md §6.6)",
            )
        if groups % self.repeat:
            raise ProtocolError(
                "P4",
                f"--parallel.tensor: the mixer at {path!r} repeats each KV head "
                f"{groups} times, which is not whole repeats of {self.repeat}: a "
                "rank's query heads would straddle two KV heads",
            )
        mixer.num_key_value_groups = groups // self.repeat
