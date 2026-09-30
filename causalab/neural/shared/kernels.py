"""Bind non-CUDA models to Transformers' PyTorch kernels.

Transformers resolves optional FLA and causal-convolution functions at
import time. ``torch_kernel_path`` temporarily binds CPU forwards to its
PyTorch implementations and restores the globals on exit. The wrapper
shape stays intact so nnsight can follow ``implementation_0`` addresses.
CUDA forwards retain the installed kernel path.

Both engines enter this guard before DeltaNet taps, which wrap the selected
functions. ``bind_kernel_path`` also sets the family globals at load time
for direct model calls outside an engine. Such calls use the device path
from the latest load of that family; engine forwards select their own path
through the guard.

Every binding goes through the symbol's one ``SymbolDispatch``
(``dispatch_for``), never ``setattr`` on the module: a simulated world's
ranks are threads of one process mid-forward at once. The guard holds the
torch path while any forward off CUDA is in flight and restores the
environment's binding when the last leaves; the loader's binding rebinds
that bottom. The short-sequence dispatcher and the taps are layers above it.
"""

from __future__ import annotations

import contextlib
import importlib
import inspect
import weakref
from typing import Any, Callable, Iterator

import torch
from torch.distributed.tensor import DTensor

from causalab.neural.shared.symbol_dispatch import dispatch_for

__all__ = [
    "KERNEL_GLOBALS",
    "bind_kernel_path",
    "kernel_modules",
    "modeling_modules",
    "torch_kernel_path",
    "torch_implementation",
]

#: The DeltaNet mixer's kernel-boundary globals, per modeling module — the
#: same four ``delta_interface`` swaps for its taps — each with the hub-kernel
#: name transformers decorates it under (the decorator's first argument).
_HUB_NAMES: dict[str, str] = {
    "causal_conv1d_fn": "causal_conv1d_fn",
    "causal_conv1d_update": "causal_conv1d_update",
    "torch_chunk_gated_delta_rule": "chunk_gated_delta_rule",
    "torch_recurrent_gated_delta_rule": "fused_recurrent_gated_delta_rule",
}
KERNEL_GLOBALS: tuple[str, ...] = tuple(_HUB_NAMES)

#: A package name nothing provides: handed to transformers' decorator so its
#: import fails and the wrapper it builds dispatches to the torch function.
_NO_PACKAGE = "causalab_torch_kernel_path"

#: torch function -> the hub-shaped wrapper over it, built once so the module
#: global is the same object on every forward
_TORCH_PATHS: dict[int, Callable[..., Any]] = {}

#: (modeling module name, global) -> the binding the environment installed —
#: what [`bind_kernel_path`][] puts back for a CUDA load after a CPU load
_INSTALLED: dict[tuple[str, str], Callable[..., Any]] = {}


def torch_implementation(fn: Callable[..., Any]) -> Callable[..., Any]:
    """The innermost function under ``fn``'s ``functools.wraps`` chain — for a
    transformers kernel global, its torch implementation; for an undecorated
    function, ``fn`` itself."""
    seen: set[int] = set()
    while id(fn) not in seen:
        seen.add(id(fn))
        inner = getattr(fn, "__wrapped__", None)
        if inner is None:
            break
        fn = inner
    return fn


#: model -> the modeling modules its forward reaches, scanned once: the
#: per-forward bindings (this guard, ``gdn_short``'s, the fused norms') run on
#: every forward, and a model's module tree does not move. Keyed weakly on
#: the module itself, never on ``id()``: an address is unique only among live
#: objects, and a collected model's entry must not answer for the next model
#: allocated where it was
_KERNEL_MODULES: "weakref.WeakKeyDictionary[torch.nn.Module, tuple[Any, ...]]" = (
    weakref.WeakKeyDictionary()
)


def modeling_modules(model: torch.nn.Module) -> tuple[Any, ...]:
    """Every module some submodule of ``model`` is defined in, in the order
    the module tree reaches them — the files whose module-global functions
    the model's forward calls by name, and so what a per-forward binding
    rebinds in. Scanned once per model object."""
    cached = _KERNEL_MODULES.get(model)
    if cached is not None:
        return cached
    out: list[Any] = []
    seen: set[str] = set()
    for module in model.modules():
        name = type(module).__module__
        if name in seen:
            continue
        seen.add(name)
        try:
            modeling = importlib.import_module(name)
        except ImportError:  # a class defined in a namespace importlib cannot reach
            continue
        out.append(modeling)
    found = tuple(out)
    _KERNEL_MODULES[model] = found
    return found


def kernel_modules(model: torch.nn.Module) -> list[Any]:
    """The modeling modules of ``model`` that export all of
    [`KERNEL_GLOBALS`][] — the files whose kernel dispatch this model's
    forward reaches."""
    return [
        modeling
        for modeling in modeling_modules(model)
        if all(hasattr(modeling, attr) for attr in KERNEL_GLOBALS)
    ]


def _dispatches_to(wrapper: Callable[..., Any], torch_fn: Callable[..., Any]) -> bool:
    """Whether ``wrapper`` — a transformers kernel global — already calls
    ``torch_fn``: its closure's ``implementation`` is the torch function. A
    function whose closure the inspection cannot read is treated as not."""
    try:
        nonlocals = inspect.getclosurevars(wrapper).nonlocals
    except (TypeError, ValueError):
        return False
    return nonlocals.get("implementation") is torch_fn


def _torch_path(name: str, torch_fn: Callable[..., Any]) -> Callable[..., Any]:
    """Transformers' own wrapper over ``torch_fn`` with no package to prefer:
    the same object the module global is on a machine without the extras."""
    cached = _TORCH_PATHS.get(id(torch_fn))
    if cached is None:
        from transformers.integrations.hub_kernels import (
            use_kernel_func_from_hub_with_fallback,
        )

        cached = use_kernel_func_from_hub_with_fallback(_HUB_NAMES[name], _NO_PACKAGE)(
            torch_fn
        )
        _TORCH_PATHS[id(torch_fn)] = cached
    return cached


def _on_cuda(model: torch.nn.Module) -> bool:
    """Whether the model's weights are on CUDA, read off its first parameter.

    One answer for the whole model, because a bundle's
    [`DeviceMap`][causalab.neural.shared.devices.DeviceMap] refuses a tower mixing
    CUDA with any other device type — at parse, and when derived from a
    placed model — so every parameter agrees with the first. The reference
    executor passes the map's own answer (``on_cuda=``) and never reaches
    this inspection; a bare model handed in without a map is read here.
    """
    for parameter in model.parameters():
        # a sharded parameter is a DTensor: its local shard is where it is
        local = parameter.to_local() if isinstance(parameter, DTensor) else parameter
        return local.device.type == "cuda"
    return False


def bind_kernel_path(model: torch.nn.Module, *, on_cuda: bool | None = None) -> None:
    """Bind the DeltaNet kernel globals of every modeling module ``model``
    reaches for the device its weights are on, and leave them bound (module
    docstring): the torch path off CUDA, the installed kernels on CUDA. A
    model with no such module, or a machine without the extras, is left as it
    is.

    ``on_cuda`` overrides the inspection of the weights — for a loader that
    knows the device it will place the model on before the weights are there
    (nnsight dispatches on first trace, so at load they are still on
    ``meta``)."""
    if on_cuda is None:
        on_cuda = _on_cuda(model)
    for modeling in kernel_modules(model):
        for name in KERNEL_GLOBALS:
            dispatch = dispatch_for(modeling, name)
            current = dispatch.bound
            torch_fn = torch_implementation(current)
            key = (modeling.__name__, name)
            already_torch = torch_fn is current or _dispatches_to(current, torch_fn)
            if not already_torch:
                # whatever dispatches elsewhere is the environment's binding
                _INSTALLED[key] = current
            if on_cuda:
                installed = _INSTALLED.get(key)
                if installed is not None and current is not installed:
                    dispatch.rebind(installed)
            elif not already_torch:
                dispatch.rebind(_torch_path(name, torch_fn))


@contextlib.contextmanager
def torch_kernel_path(
    model: torch.nn.Module, *, on_cuda: bool | None = None
) -> Iterator[None]:
    """While active, a model whose weights are not on CUDA runs transformers'
    torch implementations of the DeltaNet kernel globals (module docstring);
    a CUDA model is untouched. A hold on each symbol's dispatch: the last
    forward in flight to leave restores the environment's binding.

    ``on_cuda`` is the bundle's device map's answer (``DeviceMap.is_cuda``);
    left ``None``, the weights are inspected (`_on_cuda`)."""
    if on_cuda if on_cuda is not None else _on_cuda(model):
        yield
        return
    with contextlib.ExitStack() as holds:
        for modeling in kernel_modules(model):
            for name in KERNEL_GLOBALS:
                dispatch = dispatch_for(modeling, name)
                current = dispatch.bound
                torch_fn = torch_implementation(current)
                if torch_fn is current or _dispatches_to(current, torch_fn):
                    continue
                holds.enter_context(dispatch.holding(_torch_path(name, torch_fn)))
        yield
