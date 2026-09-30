"""Tap tensors inside grouped expert dispatch.

The grouped path sorts token-slot pairs by expert, runs the fused gate/up
projection, applies the activation, and runs the down projection before
restoring token order. Expert weights are stored as 3-D parameters, so
these tensors are reached within the dispatch function.

The context temporarily overrides ``ALL_EXPERTS_FUNCTIONS['grouped_mm']``
and restores its previous value. It handles tapped modules and passes other
calls through. Other expert implementations are rejected because their
operation order and interior addresses differ.

The wrapper repeats the grouped path's sort on the same input within the
call. Reconstruction tests pin the token/slot mapping, including ties.
Exactly two ``_grouped_linear`` calls and ``has_gate=True`` are required to
identify the fused gate/up and down projections.

The ``"grouped_mm"`` entry is one function installed by the first manager
and restored by the last (``experts_registry.EntryInstall``), looking each
module up across every entered manager's table (``_TABLES``); a module two
tables name is refused by name. The ``_grouped_linear`` global a tapped call
patches goes through a per-thread ``SymbolDispatch``. This manager's window
nests inside ``experts_path.lean_experts_path``'s; leaving out of order is
refused.
"""

from __future__ import annotations

import contextlib
import dataclasses
import threading
from typing import Any, Callable, Iterator, Mapping

import torch

from causalab.neural.engines.pytorch_hooks.experts_path import may_route_to_sentinels
from causalab.neural.engines.pytorch_hooks.experts_registry import EntryInstall
from causalab.neural.shared.symbol_dispatch import (
    AttributeSymbol,
    LazySymbol,
    SymbolDispatch,
)
from causalab.protocol.registry import EXPERTS_FUNCTION_SLOTS
from causalab.protocol.rules.errors import ProtocolError

__all__ = [
    "EXPERTS_SLOTS",
    "ExpertsInstallError",
    "ExpertsTap",
    "experts_implementation_of",
    "experts_interface_taps",
]


class ExpertsInstallError(RuntimeError):
    """A manager misusing the ``grouped_mm`` entry (module docstring, "One
    entry, many managers"): a module two entered tables name, a window
    leaving under an install made over it, the dispatch called with no
    manager entered. A programming error of a manager, never a document's,
    so it is not a protocol refusal."""


#: The points inside the grouped experts function a component may name, in the
#: order the function reaches them. ``"gate_up"`` is the fused ``[gate | up]``
#: projection (its two halves are separate components, via the descriptor's
#: fused axis); ``"activation"`` is the shared ``act_fn``'s output (the
#: activated gate half, before the ``· up`` multiply — the same tensor
#: ``mlp_activation`` names on the llama family). ``"neuron_output"`` is
#: ``act(gate) * up`` at the down-projection input. ``"down"`` captures the
#: down-projection output before the routing weight is applied. Derived from
#: the registry's component → slot map, in its order, so a slot the registry
#: gains is a slot here (the dispatch's *call* order is ``grouped_linear``'s,
#: not this tuple's); ``placements.py`` derives its copy the same way.
EXPERTS_SLOTS: tuple[str, ...] = tuple(dict.fromkeys(EXPERTS_FUNCTION_SLOTS.values()))
assert EXPERTS_SLOTS == ("gate_up", "activation", "neuron_output", "down")


@dataclasses.dataclass(frozen=True)
class ExpertsTap:
    """One read and/or edit at a named point inside the experts function.

    Values cross this interface **token-major**: the wrapper un-sorts the
    expert-sorted rows with the inverse permutation it computed, hands
    ``(tokens, top_k · width)`` to the tap along with the routing table
    ``top_k_index (tokens, top_k)``, and re-sorts whatever an edit returns. So
    a tap never sees the expert-sorted order, and the permutation never leaves
    this module.

    ``read`` observes the value as the model computed it (before any edit from
    the same tap); ``edit`` is handed a **clone** and returns the tensor to use
    in its place.
    """

    slot: str
    read: Callable[[torch.Tensor, torch.Tensor], None] | None = None
    edit: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None

    def __post_init__(self) -> None:
        if self.slot not in EXPERTS_SLOTS:
            raise ValueError(
                f"unknown experts-interface slot {self.slot!r}; "
                f"expected one of {EXPERTS_SLOTS}"
            )


def experts_implementation_of(model: Any) -> str:
    """The experts implementation a loaded model dispatches on.

    Read from the config the modeling code itself reads
    (``config._experts_implementation``, set at load time; the text config on a
    multimodal wrapper). This is the fact every experts-interface tap is pinned
    against: the interior tensors this module addresses are the *grouped*
    function's locals, and a different implementation computes different
    intermediates in a different order even where the block's output agrees.
    """
    config = getattr(model.config, "text_config", None) or model.config
    return str(getattr(config, "_experts_implementation", "<undeclared>"))


def _apply(
    taps: tuple[ExpertsTap, ...], slot: str, value: torch.Tensor, idx: torch.Tensor
) -> torch.Tensor:
    """Run every tap declared for ``slot``, in order — the ordering contract of
    `.attention_interface._apply`, verbatim: within one tap the read runs
    before the edit, across taps registration order decides, and the executor
    registers edits before reads so a same-forward read sees the written value.
    """
    for tap in taps:
        if tap.slot != slot:
            continue
        if tap.read is not None:
            tap.read(value, idx)
        if tap.edit is not None:
            value = tap.edit(value.clone(), idx)
    return value


def _has(taps: tuple[ExpertsTap, ...], slot: str) -> bool:
    return any(tap.slot == slot for tap in taps)


@contextlib.contextmanager
def experts_interface_taps(
    taps: Mapping[int, tuple[ExpertsTap, ...]],
) -> Iterator[None]:
    """Install reads and edits inside the grouped experts function.

    Args:
        taps: ``id(experts module) -> taps``. An experts module absent from the
            mapping is untouched and pays only a dict lookup — the scoping that
            keeps a tap at one layer from changing any other layer's
            arithmetic.

    The ``"grouped_mm"`` registry entry is **restored** on exit (the key exists
    in the library's global mapping, so containment is restore-not-delete —
    setting back the callable that dispatch resolved before entry preserves
    even a pre-existing local override) once the last manager entered on any
    thread has left (module docstring, "How the call is intercepted"). The
    mapping is read live, so entries added while the manager is entered are
    seen by the next call.
    """
    if not taps:
        yield
        return
    with _TABLES_LOCK:
        _TABLES.append(taps)
    _INSTALL.enter()
    try:
        yield
    finally:
        with _TABLES_LOCK:
            del _TABLES[next(i for i, table in enumerate(_TABLES) if table is taps)]
        _INSTALL.leave()


#: Guards the tables' append and identity removal.
_TABLES_LOCK = threading.Lock()
#: The tap tables of every entered manager, looked up across.
_TABLES: list[Mapping[int, tuple[ExpertsTap, ...]]] = []


def _entries(module: Any) -> tuple[ExpertsTap, ...]:
    """The one entered table's taps for ``module``, ``()`` for a module no
    table names; a module two tables name is refused (module docstring)."""
    key = id(module)
    with _TABLES_LOCK:
        tables = list(_TABLES)
    found = [entries for table in tables if (entries := table.get(key))]
    if len(found) > 1:
        raise ExpertsInstallError(
            f"the experts module {type(module).__name__} ({key}) is tapped by "
            f"{len(found)} entered managers: a table names a module another "
            "entered manager's table also names, so its call cannot be routed "
            "(experts_interface.py, 'One entry, many managers')"
        )
    return found[0] if found else ()


def _dispatch_grouped_mm(
    module: Any,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    """The one ``grouped_mm`` entry while any manager is entered: the
    library's function for a module no table names, the tapped forward
    otherwise."""
    real_impl = _INSTALL.previous
    if real_impl is None:
        raise ExpertsInstallError(
            "the grouped_mm dispatch was called with no manager entered and "
            "nothing beneath it: a stale reference to the dispatch, or an "
            "install window that was not nested in another's "
            "(experts_interface.py, 'One entry, many managers')"
        )
    entries = _entries(module)
    if not entries:
        return real_impl(module, hidden_states, top_k_index, top_k_weights)
    return _tapped_forward(
        module, hidden_states, top_k_index, top_k_weights, entries, real_impl
    )


#: The one installation of the dispatch over the registry entry, shared by
#: every entered manager on every thread (``experts_registry.py``): the first
#: captures the entry it found, the last restores it.
_INSTALL = EntryInstall("grouped_mm", lambda: _dispatch_grouped_mm)


def _tapped_forward(
    module: Any,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    entries: tuple[ExpertsTap, ...],
    real_impl: Callable[..., torch.Tensor],
) -> torch.Tensor:
    """One tapped experts call: recompute the sort, patch the two grouped
    linears and hook the shared ``act_fn`` for the duration of the one real
    call, and convert every crossing tensor between the function's
    expert-sorted rows and the taps' token-major form."""
    if getattr(module, "has_gate", None) is not True:
        # a has_gate=False family also calls _grouped_linear twice, but call 1
        # is then the plain up-projection — labeling it [gate | up] would be
        # the silent-wrong-tensor failure the descriptors exist to prevent
        raise ProtocolError(
            "P4",
            f"an experts-interface tap on {type(module).__name__}: this experts "
            "module does not declare a gated projection (has_gate is "
            f"{getattr(module, 'has_gate', None)!r}), so the first grouped "
            "linear is not [gate | up] and the slot labels here would lie. "
            "Extend experts_interface.py for this family.",
        )

    tokens = hidden_states.shape[0]
    expert_ids = top_k_index.reshape(-1)
    # ⚠️ the same unstable sort the grouped function performs, recomputed on the
    # same input inside its dynamic extent; the reconstruction-identity test is
    # what pins the tie order (module docstring)
    sorted_ids, perm = torch.sort(expert_ids)
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(perm.numel(), device=perm.device)
    # Under expert parallelism (docs/model_parallelism.md §6.3) the table
    # carries the sentinel ``num_experts`` on the slots this rank does not
    # own; the grouped function sorts them to the tail, skips them, and
    # leaves their rows of every intermediate **uninitialised** — its forward
    # output and its backward ``d_input`` alike (the library's own path is
    # covered by one pre-mask and one post-mask around the whole function).
    # A tap sits between the grouped matmuls, so it masks on both sides:
    # the view a tap sees holds zeros on the sentinel rows — an ``ExpertLocal``
    # slot nobody here owns, so the all-reduce that makes it whole is exact —
    # and the rows handed back are masked again, whose backward zeroes the
    # garbage ``d_input`` the next matmul's backward leaves there before it
    # can reach an edit's parameters (an additive edit would sum every row
    # into its gradient; a stale NaN there is a NaN fit). Whether a sentinel
    # can appear is a plan-time, rank-uniform fact of the module
    # (``may_route_to_sentinels``: the expert axis's mark, the library's EP
    # flag, or a per-rank id space), decided once here — never read off the
    # table — so the world-1 path takes the plain gathers and pays for no
    # mask; the masked path is bit-identical where the mask is all-False.
    sentinel = (
        (sorted_ids >= module.num_experts).unsqueeze(-1)
        if may_route_to_sentinels(module)
        else None
    )

    def token_major(rows: torch.Tensor) -> torch.Tensor:
        """(S, width) expert-sorted → (tokens, top_k · width), sentinel rows zero."""
        masked = rows if sentinel is None else rows.masked_fill(sentinel, 0)
        return masked[inv_perm].reshape(tokens, -1)

    def expert_sorted(value: torch.Tensor, width: int) -> torch.Tensor:
        """The inverse: (tokens, top_k · width) → (S, width) expert-sorted,
        the sentinel rows zero — in the value and, through the mask's
        backward, in the gradient flowing back into the tap."""
        by_expert = value.reshape(-1, width)[perm]
        return by_expert if sentinel is None else by_expert.masked_fill(sentinel, 0)

    def run(slot: str, rows: torch.Tensor) -> torch.Tensor:
        if not _has(entries, slot):
            return rows
        value = _apply(entries, slot, token_major(rows), top_k_index)
        return expert_sorted(value, rows.shape[-1])

    gl_calls = 0

    def grouped_linear(*args: Any, **kwargs: Any) -> torch.Tensor:
        nonlocal gl_calls
        gl_calls += 1
        if gl_calls == 2 and _has(entries, "neuron_output"):
            # The down-projection consumes the complete act(gate) * up value.
            if args:
                args = (run("neuron_output", args[0]), *args[1:])
            else:
                kwargs["input"] = run("neuron_output", kwargs["input"])
        out = _GROUPED_LINEAR.real(*args, **kwargs)
        if gl_calls == 1:
            return run("gate_up", out)
        if gl_calls == 2:
            return run("down", out)
        return out  # counted; refused below rather than mislabeled here

    act_calls = 0

    def act_hook(_m: Any, _i: Any, out: torch.Tensor) -> torch.Tensor:
        nonlocal act_calls
        act_calls += 1
        return run("activation", out)

    handle = module.act_fn.register_forward_hook(act_hook)
    try:
        with _GROUPED_LINEAR.tapped(grouped_linear):
            result = real_impl(module, hidden_states, top_k_index, top_k_weights)
    finally:
        handle.remove()
    _check_call_counts(module, gl_calls, act_calls)
    return result


def _moe() -> Any:
    """The library module both symbols live in, imported at first use like
    every transformers import of this module."""
    import transformers.integrations.moe as moe

    return moe


#: The stand-in for ``moe._grouped_linear`` while any tapped call is active,
#: the tapped forward per thread; ``real`` is the library's own function,
#: what a tapped call computes with.
_GROUPED_LINEAR = SymbolDispatch(
    LazySymbol(lambda: AttributeSymbol(_moe(), "_grouped_linear"))
)


def _check_call_counts(module: Any, gl_calls: int, act_calls: int) -> None:
    """Refuse a family whose grouped forward does not have the measured shape.

    📐 On the grouped path ``_grouped_linear`` fires exactly twice per block
    (the fused up-projection, then the down-projection) and the shared
    ``act_fn`` exactly once. Any other count means a different factorization —
    the eager loop fires ``act_fn`` once per *hit expert* — and the slot labels
    above would be attached to the wrong tensors.
    """
    if gl_calls == 2 and act_calls == 1:
        return
    raise ProtocolError(
        "P4",
        f"the grouped experts forward of {type(module).__name__} called "
        f"_grouped_linear {gl_calls} times and act_fn {act_calls} times, not "
        "(2, 1). This backend labels call 1 '[gate | up]', call 2 'down' and "
        "the activation 'act_fn(gate)', and with any other shape it cannot say "
        "which tensor it read — extend experts_interface.py for this family "
        "rather than tapping whichever call came first.",
    )
