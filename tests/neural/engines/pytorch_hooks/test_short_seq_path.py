"""The single-chunk binding in the engine (``shared/gdn_short/binding.py``).

On this tier the model is on the CPU, so the real decision never routes a
call (``selects_single_chunk`` wants CUDA) and the forward's numbers are
untouched. The plumbing is exercised anyway: with the decision told the
tensors are on CUDA and the kernel stood in by the float32 closed form, the
executor's forward reaches the stand-in at the kernel boundary with the
mixer's arguments, the DeltaNet block's output agrees with the untouched
forward within float32 tolerance (same function, another operation order),
and the module global is restored afterwards.
"""

from __future__ import annotations

import importlib
from typing import Any

import pytest
import torch

from causalab.neural.engines.pytorch_hooks.loading import ModelBundle
from causalab.neural.shared.gdn_short import binding
from causalab.neural.shared.gdn_short.options import (
    CHUNK_KERNEL_GLOBAL,
    ShortSeqKernelOptions,
)
from causalab.neural.shared.gdn_short.reference import (
    single_chunk_gated_delta_rule_torch,
)
from causalab.protocol.schema import PROTOCOL_VERSION

from ._drive import base_data_section, executor_for
from tests.protocol._docs import saved
from .test_sites_round4_deltanet import DELTANET_LAYER, TEXT

pytestmark = pytest.mark.smoke


def _modeling(bundle: ModelBundle) -> Any:
    return importlib.import_module(type(bundle.mixer_at(DELTANET_LAYER)).__module__)


def _doc() -> dict[str, Any]:
    return {
        "header": {"protocol_version": PROTOCOL_VERSION},
        "model": {"key": "test", "revision": "main"},
        "data": base_data_section(with_counterfactual=False),
        "method": {
            "sites": {"tap": {"component": "block_output", "layers": [DELTANET_LAYER]}},
            "intervened_models": {"original": {"input": "base", "reads": ["r"]}},
            "reads": {"r": {"site": "tap", "pos": -1}},
            "save": [saved("r", "original", "a.safetensors")],
        },
    }


def _read(bundle: ModelBundle) -> torch.Tensor:
    return executor_for(_doc(), bundle, base_texts=[TEXT]).dense_value("r")


def test_off_cuda_the_forward_is_untouched_and_the_global_restored(
    qwen35moe_bundle: ModelBundle, monkeypatch: pytest.MonkeyPatch
) -> None:
    modeling = _modeling(qwen35moe_bundle)
    before = getattr(modeling, CHUNK_KERNEL_GLOBAL)
    baseline = _read(qwen35moe_bundle)
    monkeypatch.setattr(
        ShortSeqKernelOptions, "from_env", classmethod(lambda cls, env=None: cls(16))
    )
    monkeypatch.setattr(
        binding,
        "single_chunk_gated_delta_rule",
        lambda *a, **k: pytest.fail("a CPU call must never reach the kernel"),
    )
    value = _read(qwen35moe_bundle)
    assert torch.equal(value, baseline)
    assert getattr(modeling, CHUNK_KERNEL_GLOBAL) is before


def test_the_executor_reaches_the_short_kernel_at_the_boundary(
    qwen35moe_bundle: ModelBundle, monkeypatch: pytest.MonkeyPatch
) -> None:
    modeling = _modeling(qwen35moe_bundle)
    before = getattr(modeling, CHUNK_KERNEL_GLOBAL)
    baseline = _read(qwen35moe_bundle)
    calls: list[dict[str, Any]] = []

    def stand_in(q: Any, k: Any, v: Any, g: Any, beta: Any, **kwargs: Any) -> Any:
        calls.append(dict(shape=tuple(q.shape), **kwargs))
        o, _ = single_chunk_gated_delta_rule_torch(
            q, k, v, g, beta, use_qk_l2norm=kwargs["use_qk_l2norm_in_kernel"]
        )
        return o.to(q.dtype), None

    real = binding.selects_single_chunk
    monkeypatch.setattr(
        binding,
        "selects_single_chunk",
        lambda **facts: real(**{**facts, "device_type": "cuda"}),
    )
    monkeypatch.setattr(binding, "single_chunk_gated_delta_rule", stand_in)
    monkeypatch.setattr(
        ShortSeqKernelOptions, "from_env", classmethod(lambda cls, env=None: cls(16))
    )
    value = _read(qwen35moe_bundle)
    assert calls, "no chunk-kernel call reached the stand-in"
    tokens = qwen35moe_bundle.tokenizer(TEXT, return_tensors="pt")["input_ids"].shape[1]
    for call in calls:
        assert call["shape"][1] == tokens <= 16
        assert call["use_qk_l2norm_in_kernel"] is True
        assert call["initial_state"] is None and call["output_final_state"] is False
    torch.testing.assert_close(value, baseline, atol=1e-4, rtol=1e-4)
    assert getattr(modeling, CHUNK_KERNEL_GLOBAL) is before
