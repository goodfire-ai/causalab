"""Route eligible short sequences to the single-chunk delta kernel.

The context layers a dispatcher over ``torch_chunk_gated_delta_rule`` in
each modeling module — one per-thread layer on the symbol's
``SymbolDispatch``, never a ``setattr`` — and removes it on exit. Selection
uses shapes and flags without reading tensor values or synchronizing the
device, so warm-up and capture agree. Long sequences, initial states,
variable-length batches, and non-CUDA models run the binding beneath the
layer unchanged. The reference engine enters it after the torch-path guard
and before the DeltaNet taps.

The hooks engine enters this context after the device-kernel guard and
before DeltaNet taps, which then observe the selected kernel's arguments
and results.
"""

from __future__ import annotations

import contextlib
from typing import Any, Callable, Iterator

import torch

from causalab.neural.shared.gdn_short.options import (
    CHUNK_KERNEL_GLOBAL,
    ShortSeqKernelOptions,
)
from causalab.neural.shared.gdn_short.triton_kernel import (
    MAX_SEQ_LEN,
    single_chunk_gated_delta_rule,
)
from causalab.neural.shared.kernels import kernel_modules
from causalab.neural.shared.symbol_dispatch import dispatch_for

__all__ = [
    "ShortSeqKernelOptions",
    "selects_single_chunk",
    "short_seq_dispatcher",
    "short_seq_kernel_path",
]


def _power_of_two_in_range(size: int) -> bool:
    return 16 <= size <= 256 and not size & (size - 1)


def selects_single_chunk(
    *,
    seq_len: int,
    key_heads: int,
    value_heads: int,
    key_dim: int,
    value_dim: int,
    device_type: str,
    threshold: int,
    initial_state: bool,
    varlen: bool,
) -> bool:
    """Whether a chunk-kernel call with these facts runs the single-chunk
    kernel: on CUDA, ``1 <= T <= threshold`` (``threshold <=``
    [`MAX_SEQ_LEN`][]), from a zero state, equal-length sequences, key
    heads dividing value heads, power-of-two head dimensions in ``[16,
    256]``."""
    return (
        device_type == "cuda"
        and 1 <= seq_len <= min(threshold, MAX_SEQ_LEN)
        and not initial_state
        and not varlen
        and value_heads % key_heads == 0
        and _power_of_two_in_range(key_dim)
        and _power_of_two_in_range(value_dim)
    )


def short_seq_dispatcher(
    bound: Callable[..., Any],
    options: ShortSeqKernelOptions,
    short: Callable[..., Any] | None = None,
) -> Callable[..., Any]:
    """The dispatcher over ``bound`` (the binding beneath it — the layer
    below on the symbol's dispatch, read at call time): the mixer's call
    shape — ``(q, k, v, g=, beta=, **kwargs)`` — with the routing decision
    from [`selects_single_chunk`][]. ``short`` is the single-chunk kernel,
    this module's [`single_chunk_gated_delta_rule`][] unless a test hands
    in a stand-in."""
    if short is None:
        short = single_chunk_gated_delta_rule

    def dispatch(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        g: torch.Tensor | None = None,
        beta: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> Any:
        if (
            g is not None
            and beta is not None
            and selects_single_chunk(
                seq_len=int(query.shape[1]),
                key_heads=int(key.shape[2]),
                value_heads=int(value.shape[2]),
                key_dim=int(key.shape[-1]),
                value_dim=int(value.shape[-1]),
                device_type=query.device.type,
                threshold=options.threshold,
                initial_state=kwargs.get("initial_state") is not None,
                varlen=kwargs.get("cu_seqlens") is not None,
            )
        ):
            return short(query, key, value, g, beta, **kwargs)
        return bound(query, key, value, g=g, beta=beta, **kwargs)

    dispatch.__wrapped__ = bound  # type: ignore[attr-defined]
    return dispatch


@contextlib.contextmanager
def short_seq_kernel_path(
    model: torch.nn.Module, options: ShortSeqKernelOptions | None = None
) -> Iterator[None]:
    """While active, every chunk-kernel call of ``model``'s DeltaNet mixers
    on this thread goes through [`short_seq_dispatcher`][] (module
    docstring). ``None`` reads the options from the environment; a disabled
    option installs nothing. The layer is left on exit either way."""
    if options is None:
        options = ShortSeqKernelOptions.from_env()
    if not options.enabled:
        yield
        return
    with contextlib.ExitStack() as layers:
        for modeling in kernel_modules(model):
            dispatch = dispatch_for(modeling, CHUNK_KERNEL_GLOBAL)
            layers.enter_context(
                dispatch.tapped(short_seq_dispatcher(dispatch.below, options))
            )
        yield
