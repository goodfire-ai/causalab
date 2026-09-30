"""The nnsight engine, too, runs a CPU model on transformers' torch kernels.

Same failure and same guard as the reference engine's
``test_kernel_path.py``: with the ``flash-linear-attention`` extra installed,
transformers binds the DeltaNet kernel globals to CUDA kernels at import time,
and a CPU trace of ``tiny-random/qwen3.5-moe`` dies inside
``causal_conv1d_fn``. The trace is wrapped in ``shared.kernels.torch_kernel_path``.
The installed kernel is simulated the way transformers binds it: a module
global refusing CPU tensors whose ``__wrapped__`` is the torch function.
"""

from __future__ import annotations

import importlib
from typing import Any, Callable, Iterator

import pytest
import torch

from causalab.neural.engines.nnsight_tracing.executor import TracePointExecutor
from causalab.neural.engines.nnsight_tracing.loading import NnsightBundle
from causalab.neural.shared.kernels import (
    KERNEL_GLOBALS,
    bind_kernel_path,
    torch_implementation,
)
from causalab.protocol.schema import PROTOCOL_VERSION, parse_document
from causalab.protocol.rules.document import validate_document

from tests.protocol._docs import in_order, saved


pytestmark = pytest.mark.smoke

TEXT = "the quick brown fox jumps"


def _modeling(bundle: NnsightBundle) -> Any:
    """The modeling module of the fixture's DeltaNet mixer, found the way the
    guard finds it: a submodule class whose module exports the globals."""
    torch_module = getattr(bundle.model, "_module")
    for module in torch_module.modules():
        modeling = importlib.import_module(type(module).__module__)
        if all(hasattr(modeling, name) for name in KERNEL_GLOBALS):
            return modeling
    raise AssertionError("the fixture has no DeltaNet mixer")


def _cuda_only(torch_fn: Callable[..., Any]) -> Callable[..., Any]:
    def kernel(x: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        if not x.is_cuda:
            raise RuntimeError("Expected x.is_cuda() to be true, but got false.")
        return torch_fn(x, *args, **kwargs)  # pragma: no cover — CPU tests

    kernel.__wrapped__ = torch_fn  # type: ignore[attr-defined]
    return kernel


@pytest.fixture()
def installed_extra(
    trace_qwen: NnsightBundle, monkeypatch: pytest.MonkeyPatch
) -> Iterator[dict[str, Any]]:
    modeling = _modeling(trace_qwen)
    bound: dict[str, Any] = {}
    for name in KERNEL_GLOBALS:
        bound[name] = _cuda_only(torch_implementation(getattr(modeling, name)))
        monkeypatch.setattr(modeling, name, bound[name])
    yield bound


def _executor(bundle: NnsightBundle, component: str) -> TracePointExecutor:
    raw = {
        "header": {"protocol_version": PROTOCOL_VERSION},
        "model": {"key": "test", "revision": "main"},
        "data": {"base": {"dataset": "inline", "field": "input"}},
        "method": {
            "intervened_models": {"original": {"input": "base", "reads": ["r"]}},
            "sites": {"tap": {"component": component, "layers": [0]}},
            "reads": {"r": {"site": "tap", "pos": -1}},
            "save": [saved("r", "original", "a.safetensors")],
        },
    }
    doc = parse_document(in_order(raw))
    validate_document(doc, engine_is_local=True)
    return TracePointExecutor(
        doc,
        bundle,
        role_rows={"base": [{"input": TEXT}]},
        role_fields={"base": "input"},
        load_tensors=lambda path: (_ for _ in ()).throw(KeyError(path)),
    )


def test_a_trace_runs_on_the_torch_path_and_restores_the_globals(
    trace_qwen: NnsightBundle, installed_extra: dict[str, Any]
) -> None:
    value = _executor(trace_qwen, "block_output").dense_value("r")
    assert value.shape[0] == 1 and torch.isfinite(value).all()
    modeling = _modeling(trace_qwen)
    for name in KERNEL_GLOBALS:
        assert getattr(modeling, name) is installed_extra[name], name


def test_an_interior_tap_still_peels_the_hub_wrapper(
    trace_qwen: NnsightBundle, installed_extra: dict[str, Any]
) -> None:
    """The nnsight address table peels the hub wrapper's ``implementation_0``
    before descending into the kernel body, so the torch path is bound as
    transformers' own wrapper over the torch function, not the bare function."""
    value = _executor(trace_qwen, "deltanet_query").dense_value("r")
    assert value.dim() == 3 and torch.isfinite(value).all()


def test_a_cpu_load_binds_the_family_for_bare_traces(
    trace_qwen: NnsightBundle, installed_extra: dict[str, Any]
) -> None:
    """What the nnsight loader does for a CPU device: a trace of the model
    itself — no executor in between (the address canary's shape) — runs on the
    torch path afterwards."""
    bind_kernel_path(getattr(trace_qwen.model, "_module"), on_cuda=False)
    with trace_qwen.model.trace(TEXT):
        pass
    modeling = _modeling(trace_qwen)
    for name in KERNEL_GLOBALS:
        assert getattr(modeling, name) is not installed_extra[name], name


def test_the_numbers_are_the_torch_paths(
    trace_qwen: NnsightBundle, monkeypatch: pytest.MonkeyPatch
) -> None:
    plain = _executor(trace_qwen, "block_output").dense_value("r")
    modeling = _modeling(trace_qwen)
    for name in KERNEL_GLOBALS:
        monkeypatch.setattr(
            modeling, name, _cuda_only(torch_implementation(getattr(modeling, name)))
        )
    guarded = _executor(trace_qwen, "block_output").dense_value("r")
    assert torch.equal(plain, guarded)
