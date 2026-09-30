"""A model off CUDA runs transformers' torch kernels, whatever is installed.

Transformers resolves the DeltaNet mixer's four kernel globals once, at
import time, to the optional package's CUDA kernel when that package is
importable — with no device check. With the ``flash-linear-attention`` extra
installed, every CPU forward of ``tiny-random/qwen3.5-moe`` (this whole test
tier) died inside ``causal_conv1d_fn`` with ``Expected x.is_cuda()``. The
engine now binds those globals to their torch implementations for the
duration of a forward when the model's weights are not on CUDA
(``shared/kernels.py``), and restores them after.

The extra cannot be installed on the CPU test machines, so the "installed
kernel" is simulated exactly as transformers binds it: a module global whose
body refuses CPU tensors and whose ``__wrapped__`` is the torch function.
"""

from __future__ import annotations

import importlib
from typing import Any, Callable, Iterator

import pytest
import torch

from causalab.neural.engines.pytorch_hooks.loading import ModelBundle
from causalab.neural.shared import kernels
from causalab.protocol.schema import PROTOCOL_VERSION
from causalab.neural.shared.kernels import (
    KERNEL_GLOBALS,
    bind_kernel_path,
    torch_implementation,
    torch_kernel_path,
)

from ._drive import base_data_section, executor_for
from tests.protocol._docs import saved
from .test_sites_round4_deltanet import DELTANET_LAYER, TEXT

pytestmark = pytest.mark.smoke


def _modeling(bundle: ModelBundle) -> Any:
    return importlib.import_module(type(bundle.mixer_at(DELTANET_LAYER)).__module__)


def _cuda_only(torch_fn: Callable[..., Any]) -> Callable[..., Any]:
    """What the extra binds: a function that refuses CPU tensors, wrapping
    the torch implementation the way ``functools.wraps`` does."""

    def kernel(x: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        if not x.is_cuda:
            raise RuntimeError("Expected x.is_cuda() to be true, but got false.")
        return torch_fn(x, *args, **kwargs)  # pragma: no cover — CPU tests

    kernel.__wrapped__ = torch_fn  # type: ignore[attr-defined]
    return kernel


@pytest.fixture()
def installed_extra(
    qwen35moe_bundle: ModelBundle, monkeypatch: pytest.MonkeyPatch
) -> Iterator[dict[str, Any]]:
    """The modeling module as it looks with the CUDA kernels installed."""
    modeling = _modeling(qwen35moe_bundle)
    bound: dict[str, Any] = {}
    for name in KERNEL_GLOBALS:
        original = getattr(modeling, name)
        bound[name] = _cuda_only(torch_implementation(original))
        monkeypatch.setattr(modeling, name, bound[name])
    yield bound


def _doc(component: str) -> dict:
    return {
        "header": {"protocol_version": PROTOCOL_VERSION},
        "model": {"key": "test", "revision": "main"},
        "data": base_data_section(with_counterfactual=False),
        "method": {
            "sites": {"tap": {"component": component, "layers": [DELTANET_LAYER]}},
            "intervened_models": {"original": {"input": "base", "reads": ["r"]}},
            "reads": {"r": {"site": "tap", "pos": -1}},
            "save": [saved("r", "original", "a.safetensors")],
        },
    }


def test_the_simulated_kernel_really_breaks_a_bare_forward(
    qwen35moe_bundle: ModelBundle, installed_extra: dict[str, Any]
) -> None:
    """The premise: with the CUDA binding in place, the model's own forward
    on CPU raises — so a passing engine forward below is the guard's doing."""
    ids = qwen35moe_bundle.tokenizer(TEXT, return_tensors="pt")["input_ids"]
    with pytest.raises(RuntimeError, match="is_cuda"):
        qwen35moe_bundle.model(input_ids=ids)


def test_an_ordinary_forward_runs_on_the_torch_path(
    qwen35moe_bundle: ModelBundle, installed_extra: dict[str, Any]
) -> None:
    executor = executor_for(_doc("block_output"), qwen35moe_bundle, base_texts=[TEXT])
    value = executor.dense_value("r")
    assert value.shape[0] == 1 and torch.isfinite(value).all()
    # the globals are the "installed" ones again once the forward returned
    modeling = _modeling(qwen35moe_bundle)
    for name in KERNEL_GLOBALS:
        assert getattr(modeling, name) is installed_extra[name], name


def test_a_kernel_boundary_tap_wraps_the_torch_path(
    qwen35moe_bundle: ModelBundle, installed_extra: dict[str, Any]
) -> None:
    """The delta taps capture the globals at their entry; the torch path is
    bound before them, so the tap wraps the implementation that runs."""
    executor = executor_for(_doc("delta_query"), qwen35moe_bundle, base_texts=[TEXT])
    value = executor.dense_value("r")
    assert value.dim() == 3 and torch.isfinite(value).all()


def test_the_numbers_are_the_torch_paths(
    qwen35moe_bundle: ModelBundle, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rebinding changes nothing a machine without the extras computes: the
    same read with and without the simulated kernel is bit-identical."""
    plain = executor_for(
        _doc("block_output"), qwen35moe_bundle, base_texts=[TEXT]
    ).dense_value("r")
    modeling = _modeling(qwen35moe_bundle)
    for name in KERNEL_GLOBALS:
        monkeypatch.setattr(
            modeling, name, _cuda_only(torch_implementation(getattr(modeling, name)))
        )
    guarded = executor_for(
        _doc("block_output"), qwen35moe_bundle, base_texts=[TEXT]
    ).dense_value("r")
    assert torch.equal(plain, guarded)


def test_a_cpu_load_binds_the_family_to_the_torch_path_for_bare_forwards(
    qwen35moe_bundle: ModelBundle, installed_extra: dict[str, Any]
) -> None:
    """What the loader does for a CPU model: after it, the model's own
    forward — outside any engine — runs, on transformers' hub-shaped wrapper
    over the torch function, and the binding stays."""
    modeling = _modeling(qwen35moe_bundle)
    bind_kernel_path(qwen35moe_bundle.model)
    ids = qwen35moe_bundle.tokenizer(TEXT, return_tensors="pt")["input_ids"]
    with torch.no_grad():
        logits = qwen35moe_bundle.model(input_ids=ids).logits
    assert torch.isfinite(logits).all()
    for name in KERNEL_GLOBALS:
        bound = getattr(modeling, name)
        assert bound is not installed_extra[name], name
        assert torch_implementation(bound) is torch_implementation(
            installed_extra[name]
        ), name
    # idempotent: the same wrapper objects on a second call
    before = {name: getattr(modeling, name) for name in KERNEL_GLOBALS}
    bind_kernel_path(qwen35moe_bundle.model)
    for name in KERNEL_GLOBALS:
        assert getattr(modeling, name) is before[name], name


def test_a_cuda_load_puts_the_installed_kernels_back(
    qwen35moe_bundle: ModelBundle,
    installed_extra: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The most recent load's device decides: a CPU load binds the torch
    path, a CUDA load of the same family restores what the environment
    installed. CUDA is simulated, since this tier has none."""
    modeling = _modeling(qwen35moe_bundle)
    bind_kernel_path(qwen35moe_bundle.model)
    assert all(
        getattr(modeling, name) is not installed_extra[name] for name in KERNEL_GLOBALS
    )
    monkeypatch.setattr(kernels, "_on_cuda", lambda _model: True)
    bind_kernel_path(qwen35moe_bundle.model)
    for name in KERNEL_GLOBALS:
        assert getattr(modeling, name) is installed_extra[name], name


def test_torch_implementation_unwraps_to_the_innermost_function() -> None:
    def inner(x: int) -> int:
        return x

    def outer(x: int) -> int:  # pragma: no cover
        return inner(x)

    outer.__wrapped__ = inner  # type: ignore[attr-defined]

    def outermost(x: int) -> int:  # pragma: no cover
        return outer(x)

    outermost.__wrapped__ = outer  # type: ignore[attr-defined]
    assert torch_implementation(outermost) is inner
    assert torch_implementation(inner) is inner


def test_a_machine_without_the_extras_rebinds_nothing(
    qwen35moe_bundle: ModelBundle,
) -> None:
    """The module globals already dispatch to the torch functions, so the
    guard leaves the very same objects in place."""
    modeling = _modeling(qwen35moe_bundle)
    before = {name: getattr(modeling, name) for name in KERNEL_GLOBALS}
    with torch_kernel_path(qwen35moe_bundle.model):
        for name in KERNEL_GLOBALS:
            assert getattr(modeling, name) is before[name], name


def test_the_module_scan_memo_dies_with_its_model() -> None:
    """The memo of a model's kernel modules is keyed weakly on the module:
    once the model is collected its entry is gone, so a new model allocated
    at the same address is scanned afresh rather than served a dead one's
    (possibly empty) list."""
    import gc

    class _Probe(torch.nn.Module):
        """A module class no other test builds, so the memo can be searched
        for it without meeting another test's model."""

    model = _Probe()
    with torch_kernel_path(model):
        pass
    memo = kernels._KERNEL_MODULES  # pyright: ignore[reportPrivateUsage]
    assert model in memo
    del model
    gc.collect()
    assert not any(isinstance(key, _Probe) for key in list(memo))


def test_a_model_without_kernel_globals_is_left_alone() -> None:
    """A family with no DeltaNet mixer (or any plain module) has nothing to
    rebind: the context is a no-op, and its own module's names are untouched."""
    model = torch.nn.Linear(2, 2)
    with torch_kernel_path(model):
        assert torch.nn.Linear is type(model)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_a_cuda_model_keeps_the_installed_kernel(
    qwen35moe_bundle: ModelBundle, installed_extra: dict[str, Any]
) -> None:  # pragma: no cover — CPU tier
    model = qwen35moe_bundle.model.to("cuda")
    try:
        modeling = _modeling(qwen35moe_bundle)
        with torch_kernel_path(model):
            for name in KERNEL_GLOBALS:
                assert getattr(modeling, name) is installed_extra[name], name
    finally:
        qwen35moe_bundle.model.to("cpu")
