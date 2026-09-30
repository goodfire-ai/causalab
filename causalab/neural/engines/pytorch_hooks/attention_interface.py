"""Read and write tensors inside eager attention.

Query and key are attention-call arguments after RoPE; key precedes
``repeat_kv``. Scores are the softmax input, probabilities its output,
and ``z`` the attention call's return. Editing a returned probability
tensor at a module hook would occur after the value multiply consumed it.

The executor selects eager attention for these taps. This context registers
an eager wrapper for one forward and restores the previous registry state
on exit. ``TorchFunctionMode`` intercepts ``F.softmax`` inside the real
attention call. It checks that exactly one matching softmax runs. Other
softmax entry points and ambiguous families are rejected.

Score edits feed the model's softmax. Probability edits feed its value
multiply. The model therefore performs its own attention computation.
The registry permits the supported mechanisms on scores and restricts
probability writes to swaps, which preserve row normalization.

The ``"eager"`` registry entry is process-global while installed. Managers
share one dispatcher: each adds its tap table to ``_TABLES``; the first
installs the entry and the last removes it. Two managers tapping one mixer
at once are refused by name.

Under context parallelism (model_parallelism.md §8.4) the wrapper is
installed for every forward. The ``query`` and ``key`` taps run on the
rank's own chunk; keys and values are then gathered over the context group
along the position axis, the query stays local, and the mask becomes the
whole frame's causal mask at this rank's query rows. The frame is the one
the executor bound around the forward (``context.current()``).
"""

from __future__ import annotations

import contextlib
import dataclasses
import importlib
from typing import Any, Callable, Iterator, Mapping

import torch
from torch.overrides import TorchFunctionMode

from causalab.neural.shared.parallel.context import SequenceFrame, current
from causalab.protocol.rules.errors import ProtocolError

__all__ = [
    "INTERFACE_SLOTS",
    "InterfaceTap",
    "attention_interface_taps",
    "module_eager_attention",
]

#: The points inside the attention function a component may name, in the order
#: the function reaches them. ``"probs"`` is the pattern's *write* slot — reading
#: it is an ordinary module tap, because the mixer returns it.
INTERFACE_SLOTS: tuple[str, ...] = ("query", "key", "scores", "probs", "z")

#: The slots served by intercepting the softmax rather than by touching an
#: argument or a return value.
_SOFTMAX_SLOTS: frozenset[str] = frozenset({"scores", "probs"})


@dataclasses.dataclass(frozen=True)
class InterfaceTap:
    """One read and/or edit at a named point inside the attention function.

    ``read`` is handed the tensor as the function sees it. ``edit`` is handed a
    **clone** and returns the tensor to use in its place, so an edit that
    mutates in place and an edit that returns a new tensor are both correct and
    neither can reach the model's own storage by accident.
    """

    slot: str
    read: Callable[[torch.Tensor], None] | None = None
    edit: Callable[[torch.Tensor], torch.Tensor] | None = None

    def __post_init__(self) -> None:
        if self.slot not in INTERFACE_SLOTS:
            raise ValueError(
                f"unknown attention-interface slot {self.slot!r}; "
                f"expected one of {INTERFACE_SLOTS}"
            )


class _SoftmaxTap(TorchFunctionMode):
    """Intercept the one ``F.softmax`` inside a tapped attention function.

    Strict on purpose: ``torch.nn.functional.softmax`` and nothing else, and the
    call is counted so that a family which softmaxes twice is refused rather than
    silently tapped at the first one.
    """

    def __init__(
        self,
        on_input: Callable[[torch.Tensor], torch.Tensor] | None = None,
        on_output: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        self.on_input = on_input
        self.on_output = on_output
        self.calls = 0

    def __torch_function__(
        self,
        func: Any,
        types: Any,
        args: tuple[Any, ...] = (),
        kwargs: Mapping[str, Any] | None = None,
    ) -> Any:
        kwargs = dict(kwargs or {})
        if func is not torch.nn.functional.softmax:
            return func(*args, **kwargs)
        self.calls += 1
        if self.on_input is not None and args:
            args = (self.on_input(args[0]), *args[1:])
        out = func(*args, **kwargs)
        return out if self.on_output is None else self.on_output(out)


def _apply(taps: "tuple[InterfaceTap, ...]", slot: str, value: torch.Tensor):
    """Run every tap declared for ``slot``, in order.

    Two orderings, and only one of them is this function's to choose:

    * **within one tap**, the read runs before the edit — a tap that does both
      observes the value the model computed, not its own replacement;
    * **across taps**, registration order decides, and the executor registers
      edits before reads. So a document that reads and writes the same slot in
      one forward sees the *written* value.

    That second one is not an accident of this file: it is what the module-hook
    path already does, because ``_installed`` is entered before ``_capturing``
    and hooks fire in registration order. 📐 Measured equal on both paths —
    ``attention_premix`` (a module boundary) and ``attention_query`` /
    ``attention_z`` (interface slots) all read back exactly the written value,
    difference 0.0. The two tap mechanisms have to agree here or the same
    document would mean different things depending on which components it named.
    """
    for tap in taps:
        if tap.slot != slot:
            continue
        if tap.read is not None:
            tap.read(value)
        if tap.edit is not None:
            value = tap.edit(value.clone())
    return value


def _has(taps: "tuple[InterfaceTap, ...]", slot: str) -> bool:
    return any(tap.slot == slot for tap in taps)


#: The tap tables of every installed manager, most recent last; a mixer is
#: looked up across them (module docstring, "one entry, many managers").
_TABLES: list[Mapping[int, "tuple[InterfaceTap, ...]"]] = []
#: The registry state the first manager found, restored by the last.
_ENTRY: dict[str, Any] = {"count": 0, "had_key": False, "previous": None}


def _entries(module: Any) -> "tuple[InterfaceTap, ...]":
    key = id(module)
    for table in reversed(_TABLES):
        entries = table.get(key)
        if entries:
            return entries
    return ()


def _chunked() -> SequenceFrame | None:
    """The forward's frame when its positions are split over a context group
    above one (§8.4), else ``None``."""
    frame = current()
    return frame if frame is not None and frame.size > 1 else None


def _gather_kv(
    frame: SequenceFrame,
    key: torch.Tensor,
    value: torch.Tensor,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The whole frame's keys and values from every rank's chunk, and the
    causal mask of the whole frame at this rank's query rows. The gathered
    keys feed **this rank's** queries, so the gather is the faithful one
    (``SequenceFrame.gather_faithful``, §7): in backward the gradient of
    chunk ``j``'s keys sums every rank's queries' contribution."""
    return (
        frame.gather_faithful(key, 2),
        frame.gather_faithful(value, 2),
        frame.attention_mask(dtype),
    )


def _dispatch(
    module: Any,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Resolved from the MODULE's own modeling file, per call: while the
    # registry entry is installed it intercepts every attention forward, so
    # borrowing one family's function would silently replace another's math
    # (gemma-2's eager soft-caps the logits, say).
    real = module_eager_attention(module)
    entries = _entries(module)
    frame = _chunked()
    if not entries and frame is None:
        return real(
            module,
            query,
            key,
            value,
            attention_mask,
            scaling=scaling,
            dropout=dropout,
            **kwargs,
        )

    query = _apply(entries, "query", query)
    key = _apply(entries, "key", key)
    if frame is not None:
        key, value, attention_mask = _gather_kv(frame, key, value, query.dtype)

    wants_softmax = any(_has(entries, slot) for slot in _SOFTMAX_SLOTS)
    if not wants_softmax:
        out, weights = real(
            module,
            query,
            key,
            value,
            attention_mask,
            scaling=scaling,
            dropout=dropout,
            **kwargs,
        )
        return _apply(entries, "z", out), weights

    def on_scores(scores: torch.Tensor) -> torch.Tensor:
        return _apply(entries, "scores", scores)

    def on_probs(probs: torch.Tensor) -> torch.Tensor:
        # Returning the edited pattern is the whole write: the model's own
        # eager function receives it and does its own value multiply, so
        # nothing here has to know what that multiply is.
        return _apply(entries, "probs", probs)

    mode = _SoftmaxTap(
        on_scores if _has(entries, "scores") else None,
        on_probs if _has(entries, "probs") else None,
    )
    with mode:
        out, weights = real(
            module,
            query,
            key,
            value,
            attention_mask,
            scaling=scaling,
            dropout=dropout,
            **kwargs,
        )
    _check_one_softmax(module, mode.calls)
    return _apply(entries, "z", out), weights


def _install(registry: Any) -> None:
    if _ENTRY["count"] == 0:
        _ENTRY["had_key"] = "eager" in registry
        _ENTRY["previous"] = registry["eager"] if _ENTRY["had_key"] else None
        registry["eager"] = _dispatch
    _ENTRY["count"] += 1


def _uninstall(registry: Any) -> None:
    _ENTRY["count"] -= 1
    if _ENTRY["count"]:
        return
    if _ENTRY["had_key"]:
        registry["eager"] = _ENTRY["previous"]
    else:
        _unregister(registry, "eager")
    _ENTRY["had_key"], _ENTRY["previous"] = False, None


@contextlib.contextmanager
def attention_interface_taps(
    taps: Mapping[int, "tuple[InterfaceTap, ...]"],
) -> Iterator[None]:
    """Install reads and edits inside the eager attention function — and,
    under context parallelism, the all-gather of keys and values (module
    docstring), for every mixer of the forward.

    Args:
        taps: ``id(mixer module) -> taps``. A mixer absent from the mapping is
            untouched and pays only a dict lookup, which is what keeps a tap at
            one layer from changing any other layer's arithmetic. The mapping
            is read live, so entries added while the manager is entered are
            seen by the next call.
    The ``"eager"`` registry key is removed on exit (or restored, if something
    else had registered one) once the last manager leaves, which puts
    ``get_interface`` back on the module default.

    Raises:
        ValueError: a mixer already tapped by an installed manager.
    """
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    if not taps and _chunked() is None:
        yield
        return
    shared = {key for table in _TABLES for key in table} & set(taps)
    if shared:
        raise ValueError(
            f"attention_interface_taps: {len(shared)} mixer(s) are already tapped "
            "by an installed manager; two managers on one mixer would let the "
            "inner's edits replace the outer's"
        )
    _TABLES.append(taps)
    _install(ALL_ATTENTION_FUNCTIONS)
    try:
        yield
    finally:
        _TABLES.remove(taps)
        _uninstall(ALL_ATTENTION_FUNCTIONS)


def module_eager_attention(module: Any) -> Callable[..., Any]:
    """The mixer's own ``eager_attention_forward``.

    Resolved from the modeling file the module's class was defined in, per call.
    While the registry entry is installed it intercepts *every* attention
    forward, so borrowing one family's function would silently replace another
    family's math — gemma-2's eager soft-caps its logits, for instance. A family
    whose modeling file exports no such function is refused by name rather than
    served somebody else's.

    ⚠️ Deliberately asks for **only** this symbol. An earlier version also needed
    ``repeat_kv`` here, to redo the value multiply after a pattern edit; 📐 GPT-2
    exports the first and not the second (no GQA, nothing to repeat), so asking
    for both made a plain read of the attention interior on gpt2 fail with a
    message about pattern writes. A later version removed the second requirement
    entirely along with the recompute that needed it.
    """
    modeling = importlib.import_module(type(module).__module__)
    found = getattr(modeling, "eager_attention_forward", None)
    if found is None:
        raise ProtocolError(
            "P4",
            f"an attention-interface tap on {type(module).__name__}: its "
            f"modeling module {type(module).__module__!r} exports no "
            "'eager_attention_forward'. Extend attention_interface.py for this "
            "family — borrowing another family's version would silently change "
            "what the model computes.",
        )
    return found


def _check_one_softmax(module: Any, calls: int) -> None:
    """Refuse a family whose eager attention does not softmax exactly once.

    📐 All three CI fixtures call it exactly once. A family that calls it twice —
    soft-capping, a second sliding-window pass — would have *a* softmax tapped,
    and which one would depend on source order. That is the silent-wrong-tensor
    failure the whole descriptor effort exists to prevent, so it is refused by
    name; zero calls means the function did not softmax at all, which means the
    tap read nothing.
    """
    if calls == 1:
        return
    raise ProtocolError(
        "P4",
        f"the eager attention of {type(module).__name__} called "
        f"torch.nn.functional.softmax {calls} times, not once. This backend taps "
        "the softmax to reach the attention scores, and with "
        f"{'no call' if calls == 0 else 'more than one'} it cannot say which "
        "tensor it read. Extend attention_interface.py for this family rather "
        "than tapping whichever call came first.",
    )


def _unregister(registry: Any, name: str) -> None:
    """Remove a key from ``ALL_ATTENTION_FUNCTIONS``.

    ``AttentionInterface`` is dict-like but its deletion surface has moved
    between versions, so try the documented spelling first and fall back to the
    backing mapping. Leaving the key installed would silently keep the wrapper
    in force for the rest of the process, which is the one outcome worth being
    thorough about.
    """
    try:
        del registry[name]
        return
    except (KeyError, TypeError, AttributeError):
        pass
    backing = getattr(registry, "_local_mapping", None)
    if isinstance(backing, dict):
        backing.pop(name, None)
