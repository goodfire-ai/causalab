"""Tap the Gated DeltaNet convolution and delta-rule boundaries.

The mixer calls module-global convolution and delta-rule functions.
Taps observe the convolution return, post-convolution and pre-normalization
q/k/v, gates ``beta`` and ``g``, and ``core_attn_out`` before its norm and gate.
Prefill uses the chunked kernel; cached decode uses the recurrent kernel
and convolution update. All four globals are wrapped together.

Wrappers come from the tapped mixer's modeling module and call the functions
captured at entry, preserving installed dispatch. Mixer hooks mark the
active dynamic extent, so other mixers pass through. Exit restores globals.
A kernelized forward outside the modeling file or a missing required global
raises an error naming the unsupported path.

Each global is patched through one ``SymbolDispatch`` per (modeling module,
name) for the whole process; a manager's wrappers are one per-thread layer
on it, above the torch-path guard's and the short-sequence dispatcher's
layers. The first layer to enter installs the dispatch, the last to leave
restores it. A simulated world runs its ranks as threads, so two managers
can be entered at once.

Under context parallelism (model_parallelism.md §6.4, §8.4) the chunked
kernel receives the previous rank's final state (``context.handoff_state``)
and the causal conv1d the previous chunk's last inputs
(``context.chunked_conv``). The wrappers are installed for every DeltaNet
modeling module whether or not a document taps the boundary, so the
DeltaNet layers run sequentially across the context group.
"""

from __future__ import annotations

import contextlib
import dataclasses
import importlib
from typing import Any, Callable, Iterator, Mapping

import torch

from causalab.neural.shared.kernels import KERNEL_GLOBALS, kernel_modules
from causalab.neural.shared.parallel.context import (
    SequenceFrame,
    chunked_conv,
    current,
    handoff_state,
)
from causalab.neural.shared.symbol_dispatch import SymbolDispatch, dispatch_for
from causalab.protocol.rules.errors import ProtocolError

__all__ = [
    "DELTA_SLOTS",
    "DeltaTap",
    "delta_kernel_taps",
]

#: The points at the kernel boundary a component may name, in the order the
#: forward reaches them. ``"conv"`` is the causal-conv function's return
#: (channels-first); the five argument slots and ``"kernel_output"`` belong to
#: whichever delta-rule kernel the forward dispatches (chunked at prefill,
#: recurrent at cached decode steps — tapped identically).
DELTA_SLOTS: tuple[str, ...] = (
    "conv",
    "query",
    "key",
    "value",
    "beta",
    "decay",
    "kv_mem",
    "state_update",
    "state",
    "kernel_output",
)

#: The per-step interior. At prefill these exist only inside the
#: recurrent formulation, which the chunked kernel never materializes — so a
#: read **steps the library's own recurrent kernel** in the chunked call's
#: shadow (nothing transcribed: every number is the library's), and a state
#: write substitutes that stepwise loop for the chunked call (path-forcing,
#: measured 5.4e-7 on the logits, pinned per layer). At decode the model runs
#: the recurrent kernel natively and all three are plain per-step captures.
_STATE_SLOTS: frozenset[str] = frozenset({"kv_mem", "state_update", "state"})

#: The four module globals swapped together, per modeling module — the same
#: four ``shared/kernels.py`` binds to the torch path for a model off CUDA.
_GLOBALS: tuple[str, ...] = KERNEL_GLOBALS


@dataclasses.dataclass(frozen=True)
class DeltaTap:
    """One read and/or edit at a named point at the kernel boundary.

    Same contract as [`.attention_interface.InterfaceTap`][causalab.neural.engines.pytorch_hooks.attention_interface.InterfaceTap]: ``read`` is
    handed the tensor as the function sees it, ``edit`` is handed a **clone**
    and returns the replacement.

    ``edit_state`` is the state write's own surface — ``(step, S_t) -> S_t`` —
    because a state edit must **feed forward**: step ``t``'s replacement is
    what step ``t+1`` decays and writes into, so the whole-tensor ``edit``
    contract cannot express it. Only the ``"state"`` slot may carry one, and
    carrying one is what switches the chunked call to the stepwise
    substitution.
    """

    slot: str
    read: Callable[[torch.Tensor], None] | None = None
    edit: Callable[[torch.Tensor], torch.Tensor] | None = None
    edit_state: Callable[[int, torch.Tensor], torch.Tensor] | None = None

    def __post_init__(self) -> None:
        if self.slot not in DELTA_SLOTS:
            raise ValueError(
                f"unknown delta-kernel slot {self.slot!r}; expected one of "
                f"{DELTA_SLOTS}"
            )
        if self.edit_state is not None and self.slot != "state":
            raise ValueError(
                f"edit_state is the state write's surface; slot {self.slot!r} "
                "cannot carry one"
            )


def _apply(taps: tuple[DeltaTap, ...], slot: str, value: torch.Tensor) -> torch.Tensor:
    """The ordering contract of `.attention_interface._apply`, verbatim."""
    for tap in taps:
        if tap.slot != slot:
            continue
        if tap.read is not None:
            tap.read(value)
        if tap.edit is not None:
            value = tap.edit(value.clone())
    return value


def _has(taps: tuple[DeltaTap, ...], slot: str) -> bool:
    return any(tap.slot == slot for tap in taps)


def _modeling_module(mixer: Any) -> Any:
    """The modeling file a tapped mixer's kernels live in — with the two
    refusals the module docstring names."""
    cls = type(mixer)
    forward_home = getattr(cls.forward, "__module__", None)
    if forward_home != cls.__module__:
        raise ProtocolError(
            "P4",
            f"a delta-kernel tap on {cls.__name__}: its forward comes from "
            f"{forward_home!r}, not its own modeling module {cls.__module__!r} "
            "— a kernelize()d (hub-kernel) mixer computes inside a fused kernel "
            "no module-global patch can reach. Load the model without "
            "kernelize(), or extend delta_interface.py for this kernel.",
        )
    modeling = importlib.import_module(cls.__module__)
    missing = [name for name in _GLOBALS if not hasattr(modeling, name)]
    if missing:
        raise ProtocolError(
            "P4",
            f"a delta-kernel tap on {cls.__name__}: its modeling module "
            f"{cls.__module__!r} exports no {', '.join(missing)}. Extend "
            "delta_interface.py for this family — borrowing another family's "
            "kernels would silently change what the model computes.",
        )
    return modeling


def _chunked() -> SequenceFrame | None:
    """The forward's frame when its positions are split over a context group
    above one (§8.4), else ``None``."""
    frame = current()
    return frame if frame is not None and frame.size > 1 else None


@contextlib.contextmanager
def delta_kernel_taps(
    taps: Mapping[Any, tuple[DeltaTap, ...]], *, model: Any = None
) -> Iterator[None]:
    """Install reads and edits at the DeltaNet kernel boundary — and, under
    context parallelism, the state and conv-history handoffs (module
    docstring), in every DeltaNet modeling module of ``model``.

    Args:
        taps: ``mixer module -> taps`` (keyed by the module object itself —
            unlike the attention registry, the wrappers here never receive the
            module as an argument, so the mixers are also where the dynamic
            extent is tracked). A mixer absent from the mapping is untouched.
        model: the model the forward runs — under ``cp > 1`` its kernel
            modeling modules are patched whether or not a mixer is tapped
            (``kernels.kernel_modules``); ignored otherwise.

    All four globals are restored on exit, in every patched modeling module,
    once the last layer in flight on any thread has left (module docstring).
    """
    chunked = _chunked() is not None
    if not taps and not (chunked and model is not None):
        yield
        return

    #: the mixer whose forward is currently executing, if it is tapped
    active: dict[str, Any] = {"mixer": None, "seq_len": None}

    per_module: dict[int, Any] = {}
    modelings = [_modeling_module(mixer) for mixer in taps]
    if chunked and model is not None:
        modelings.extend(kernel_modules(model))
    for modeling in modelings:
        per_module.setdefault(id(modeling), modeling)

    def conv_wrapper(dispatch: SymbolDispatch, *, prefill: bool) -> Callable[..., Any]:
        def wrapped(hidden_states: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
            real = dispatch.real
            frame = _chunked() if prefill else None
            if frame is not None:
                # the previous chunk's tail is this chunk's history (§8.4);
                # `args[0]` is the conv weight, whose width says how much
                out = chunked_conv(frame, real, hidden_states, *args, **kwargs)
            else:
                out = real(hidden_states, *args, **kwargs)
            current = _by_identity(taps, active["mixer"])
            if not current or not _has(current, "conv"):
                return out
            # ⚠️ Under a cache, `update_conv_state` may hand the conv a tensor
            # longer than the mixer's own sequence (prepended state), and the
            # forward keeps only the last seq_len columns. The tap addresses
            # exactly what the forward keeps, so the same slice is applied
            # here — a no-op when the lengths already agree.
            seq_len = active["seq_len"]
            if seq_len is not None and out.shape[-1] != seq_len:
                kept = _apply(current, "conv", out[..., -seq_len:])
                out = out.clone()
                out[..., -seq_len:] = kept
                return out
            return _apply(current, "conv", out)

        return wrapped

    def kernel_wrapper(
        dispatch: SymbolDispatch, recurrent: SymbolDispatch, modeling: Any
    ) -> Callable[..., Any]:
        def wrapped(
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            g: torch.Tensor | None = None,
            beta: torch.Tensor | None = None,
            **kwargs: Any,
        ) -> Any:
            real = dispatch.real
            current = _by_identity(taps, active["mixer"])
            frame = _chunked()
            if not current and frame is None:
                return real(query, key, value, g=g, beta=beta, **kwargs)
            # the argument taps run on the rank's own chunk, before the
            # handoff — their gathers come before the point-to-point, in
            # the same order on every rank of the group
            query = _apply(current, "query", query)
            key = _apply(current, "key", key)
            value = _apply(current, "value", value)
            if beta is not None:
                beta = _apply(current, "beta", beta)
            if g is not None:
                g = _apply(current, "decay", g)

            # the per-step reads are collected inside the kernel call and
            # applied after it — after the handoff, under cp > 1: a read
            # gathers over the context group, and a gather between this
            # rank's kernel and its send would wait on the rank waiting to
            # receive that very state (the §6.5 rule, "no collective inside
            # the sequential step")
            faces: list[tuple[str, torch.Tensor]] = []
            offset = 0 if frame is None else frame.chunk.start

            def run(
                initial_state: torch.Tensor | None,
            ) -> tuple[torch.Tensor, torch.Tensor | None]:
                call = dict(kwargs)
                if frame is not None:
                    if call.get("initial_state") is not None:
                        raise ProtocolError(
                            "P4",
                            f"--parallel.context: the DeltaNet kernel was handed a "
                            f"cached state under cp={frame.size}; a cached forward "
                            "(a decode) is not served under context parallelism "
                            "(docs/model_parallelism.md §8.4)",
                        )
                    call["initial_state"] = initial_state
                    call["output_final_state"] = True
                if any(tap.slot in _STATE_SLOTS for tap in current):
                    out, state, deferred = _with_state_taps(
                        current,
                        real,
                        recurrent.real,
                        modeling,
                        query,
                        key,
                        value,
                        g,
                        beta,
                        call,
                        offset=offset,
                    )
                    faces.extend(deferred)
                    return out, state
                return real(query, key, value, g=g, beta=beta, **call)

            if frame is None:
                out, state = run(None)
            else:
                # rank r runs after rank r − 1: its final state is this
                # chunk's initial state (§6.4); the kernel's state is
                # (batch, heads, d_k, d_v) in float32. ``value`` links the
                # received state to this rank's graph, so a fit's gradient
                # flows back to the chunk below (§7)
                out, state = handoff_state(
                    frame,
                    run,
                    shape=(key.shape[0], key.shape[2], key.shape[-1], value.shape[-1]),
                    dtype=torch.float32,
                    device=value.device,
                    link=value,
                )
                if not kwargs.get("output_final_state", False):
                    state = None
            for slot, face in faces:
                _apply(current, slot, face)
            return _apply(current, "kernel_output", out), state

        return wrapped

    with contextlib.ExitStack() as stack:
        for modeling in per_module.values():
            conv = dispatch_for(modeling, "causal_conv1d_fn")
            conv_update = dispatch_for(modeling, "causal_conv1d_update")
            chunk = dispatch_for(modeling, "torch_chunk_gated_delta_rule")
            recurrent = dispatch_for(modeling, "torch_recurrent_gated_delta_rule")
            stack.enter_context(conv.tapped(conv_wrapper(conv, prefill=True)))
            stack.enter_context(
                conv_update.tapped(conv_wrapper(conv_update, prefill=False))
            )
            stack.enter_context(
                chunk.tapped(kernel_wrapper(chunk, recurrent, modeling))
            )
            stack.enter_context(
                recurrent.tapped(kernel_wrapper(recurrent, recurrent, modeling))
            )
        for mixer in taps:
            pre = mixer.register_forward_pre_hook(_enter(active, mixer))
            post = mixer.register_forward_hook(_leave(active))
            stack.callback(pre.remove)
            stack.callback(post.remove)
        yield


def _l2norm_of(modeling: Any) -> Callable[..., torch.Tensor]:
    """The modeling file's own ``l2norm`` — needed to form k̂ for the derived
    per-step faces, resolved per family like everything else here."""
    found = getattr(modeling, "l2norm", None)
    if found is None:
        raise ProtocolError(
            "P4",
            f"a per-step state tap needs the modeling module "
            f"{modeling.__name__!r} to export 'l2norm' (the normalization its "
            "own kernel applies to k), and it does not — extend "
            "delta_interface.py for this family.",
        )
    return found


def _state_faces(
    k_t: torch.Tensor,
    v_t: torch.Tensor,
    g_t: torch.Tensor,
    beta_t: torch.Tensor,
    s_prev: torch.Tensor,
    l2norm: Callable[..., torch.Tensor],
    use_l2: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The two derived per-step faces, from adjacent states:
    ``kv_mem_t = (S_{t-1}·exp(g_t) · k̂_t).sum(-2)`` and
    ``delta_t = (v_t − kv_mem_t)·β_t`` — the recurrent kernel's own lines
    (``modeling:369-374``), computed in float32 exactly as it computes them,
    and pinned by the reconstruction identity
    ``S_t == S_{t-1}·exp(g_t) + k̂_t ⊗ delta_t`` against the kernel's returned
    states.

    Args are one step's slices: ``k_t/v_t (b, h, d)``, ``g_t/beta_t (b, h)``,
    ``s_prev (b, h, d_k, d_v)`` in float32.
    """
    k_hat = l2norm(k_t, dim=-1, eps=1e-6) if use_l2 else k_t
    decayed = s_prev * g_t.to(torch.float32).exp()[..., None, None]
    kv_mem = (decayed * k_hat.to(torch.float32).unsqueeze(-1)).sum(dim=-2)
    delta = (v_t.to(torch.float32) - kv_mem) * beta_t.to(torch.float32).unsqueeze(-1)
    return kv_mem, delta


def _stepwise(
    real_recurrent: Callable[..., Any],
    l2norm: Callable[..., torch.Tensor],
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    use_l2: bool,
    edit_state: Callable[[int, torch.Tensor], torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Drive the library's own recurrent kernel one timestep at a time.

    No transcription: every state is the kernel's own return, threaded back in
    as the next step's ``initial_state``. ``edit_state`` (a state write) is
    applied to each step's state *before* it threads forward, which is the
    whole reason a write substitutes this loop for the chunked call.

    Returns ``(out, final_state, states, kv_mems, deltas)`` with the per-step
    tensors stacked on a steps axis: ``states (b, s, h, d_k, d_v)``,
    ``kv_mems/deltas (b, s, h, d_v)``.
    """
    batch, seq_len, heads, d_k = key.shape
    d_v = value.shape[-1]
    state = initial_state
    state_fp = (
        torch.zeros(batch, heads, d_k, d_v, dtype=torch.float32, device=value.device)
        if state is None
        else state.to(torch.float32)
    )
    outs: list[torch.Tensor] = []
    states: list[torch.Tensor] = []
    kv_mems: list[torch.Tensor] = []
    deltas: list[torch.Tensor] = []
    for t in range(seq_len):
        kv_mem, delta = _state_faces(
            key[:, t], value[:, t], g[:, t], beta[:, t], state_fp, l2norm, use_l2
        )
        out_t, new_state = real_recurrent(
            query[:, t : t + 1],
            key[:, t : t + 1],
            value[:, t : t + 1],
            g=g[:, t : t + 1],
            beta=beta[:, t : t + 1],
            initial_state=state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=use_l2,
        )
        if edit_state is not None:
            new_state = edit_state(t, new_state)
        outs.append(out_t)
        states.append(new_state)
        kv_mems.append(kv_mem)
        deltas.append(delta)
        state = new_state
        state_fp = new_state.to(torch.float32)
    return (
        torch.cat(outs, dim=1),
        state,
        torch.stack(states, dim=1),
        torch.stack(kv_mems, dim=1),
        torch.stack(deltas, dim=1),
    )


Faces = list[tuple[str, torch.Tensor]]


def _with_state_taps(
    current: tuple[DeltaTap, ...],
    real: Callable[..., Any],
    real_recurrent: Callable[..., Any],
    modeling: Any,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor | None,
    beta: torch.Tensor | None,
    kwargs: dict[str, Any],
    *,
    offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor | None, Faces]:
    """Serve the per-step slots around one kernel call: the kernel's output
    and state, and the per-step faces — ``(slot, tensor)`` for ``kv_mem``,
    ``state_update`` and ``state`` — for the caller to hand the reads
    **after** the call (module docstring, "Context parallelism"). ``real``
    is the kernel the forward dispatched, ``real_recurrent`` the modeling
    module's own recurrent kernel the per-step loop drives (the same object
    at a decode step). A state write's ``edit_state`` runs inside, per step,
    at the step's position in the padded frame: ``offset`` is this rank's
    first position under context parallelism (the writer's positions are
    the frame's), zero at ``cp=1``.

    Three cases, in the order they are checked:

    * a **single cached decode step** (the recurrent kernel, one token): the
      per-step interior is the stock path — the state is the call's own
      return, the faces derive from its ``initial_state`` and arguments,
      nothing extra runs;
    * a **state write** at prefill: the stepwise loop *substitutes* for the
      chunked call, so edits feed forward. That is path-forcing and carries
      the measured cost (📐 logits 5.4e-7, pinned as a per-layer test bound) —
      paid only when a write targets the state, only at this layer;
    * **reads** at prefill: the chunked kernel still runs untouched (the
      forward's numbers are bit-identical) and the loop runs in its shadow,
      costing O(seq) extra kernel calls at this layer only.
    """
    assert g is not None and beta is not None  # the modeling call always passes both
    l2norm = _l2norm_of(modeling)
    use_l2 = bool(kwargs.get("use_qk_l2norm_in_kernel", False))
    initial_state = kwargs.get("initial_state")
    wants_final = bool(kwargs.get("output_final_state", False))
    edits = [tap.edit_state for tap in current if tap.edit_state is not None]

    def edit_state(step: int, state: torch.Tensor) -> torch.Tensor:
        for edit in edits:
            state = edit(step + offset, state)
        return state

    if real is real_recurrent and key.shape[1] == 1 and not edits:
        # a native decode step: state = the call's own return, faces derived
        out, new_state = real(query, key, value, g=g, beta=beta, **kwargs)
        state_fp = (
            torch.zeros(
                key.shape[0],
                key.shape[2],
                key.shape[-1],
                value.shape[-1],
                dtype=torch.float32,
                device=value.device,
            )
            if initial_state is None
            else initial_state.to(torch.float32)
        )
        kv_mem, delta = _state_faces(
            key[:, 0], value[:, 0], g[:, 0], beta[:, 0], state_fp, l2norm, use_l2
        )
        faces: Faces = [
            ("kv_mem", kv_mem.unsqueeze(1)),
            ("state_update", delta.unsqueeze(1)),
        ]
        if new_state is not None:
            faces.append(("state", new_state.unsqueeze(1)))
        return out, new_state, faces

    if edits:
        # substitution: the loop IS the forward for this layer
        out, final_state, states, kv_mems, deltas = _stepwise(
            real_recurrent,
            l2norm,
            query,
            key,
            value,
            g,
            beta,
            initial_state,
            use_l2,
            edit_state=edit_state,
        )
        faces = [("kv_mem", kv_mems), ("state_update", deltas), ("state", states)]
        return out, (final_state if wants_final else None), faces

    # reads only: the base forward is untouched — the chunked kernel still
    # runs and the logits are bit-identical; the loop runs in its shadow
    out, state = real(query, key, value, g=g, beta=beta, **kwargs)
    _, _, states, kv_mems, deltas = _stepwise(
        real_recurrent, l2norm, query, key, value, g, beta, initial_state, use_l2
    )
    faces = [("kv_mem", kv_mems), ("state_update", deltas), ("state", states)]
    return out, state, faces


def _by_identity(
    taps: Mapping[Any, tuple[DeltaTap, ...]], mixer: Any
) -> tuple[DeltaTap, ...]:
    """The taps for one mixer, by object identity (nn.Module hashes by
    identity, so a plain lookup is exactly this — kept as a function so the
    intent survives a mapping type that hashes differently)."""
    if mixer is None:
        return ()
    return taps.get(mixer, ())


def _enter(active: dict[str, Any], mixer: Any) -> Callable[..., None]:
    def hook(_m: Any, args: tuple[Any, ...]) -> None:
        active["mixer"] = mixer
        hidden = args[0] if args else None
        active["seq_len"] = (
            int(hidden.shape[1]) if isinstance(hidden, torch.Tensor) else None
        )

    return hook


def _leave(active: dict[str, Any]) -> Callable[..., None]:
    def hook(_m: Any, _args: Any, _out: Any) -> None:
        active["mixer"] = None
        active["seq_len"] = None

    return hook
