"""Cross-engine parity on Apple's MPS backend.

The CPU parity suite (``tests/neural/engines/nnsight_tracing/test_parity_*``)
compares the two engines on one device — and its conftest pins
``torch.backends.mps.is_available`` to ``False``, because nnsight's dispatch
places a model on ``mps:0`` on a Mac regardless of ``device_map``, which
would compare MPS numerics against CPU ones. This module is
the one place the MPS path itself is compared: both engines asked for
``mps``, one ``block_output`` read on the tiny hybrid fixture, agreeing at the
CPU suite's tolerance.

Golden tier, out of CI's reach: it skips unless an MPS device is available,
so it runs only on a local Mac. It lives here rather than beside the CPU
suite so the CPU conftest's guard does not disable the very device it tests.
"""

from __future__ import annotations

import pytest
import torch

from tests._helpers import engines
from tests._helpers.a3b_sweep import assert_same, read_doc, stream_layers

pytestmark = [
    pytest.mark.golden,
    pytest.mark.skipif(
        not torch.backends.mps.is_available(), reason="needs an MPS device (a Mac)"
    ),
]

TINY_QWEN35_MOE = "tiny-random/qwen3.5-moe"
BASE_TEXTS = ["the quick brown fox jumps", "a small red hen sits still"]


@pytest.fixture(scope="module")
def hooks_qwen_mps():  # pragma: no cover — MPS only
    from causalab.neural.engines.pytorch_hooks.loading import load_model

    return load_model(TINY_QWEN35_MOE, device="mps")


@pytest.fixture(scope="module")
def trace_qwen_mps():  # pragma: no cover — MPS only
    from causalab.neural.engines.nnsight_tracing.loading import load_model

    # eager pinned, as the CPU suite pins it: like against like
    return load_model(TINY_QWEN35_MOE, device="mps", attn_implementation="eager")


def test_a_block_output_read_agrees_across_engines_on_mps(
    hooks_qwen_mps, trace_qwen_mps
):  # pragma: no cover — MPS only
    delta_layer, _ = stream_layers(hooks_qwen_mps)
    hooks, trace = engines.both_executors(
        read_doc("block_output", delta_layer),
        hooks_qwen_mps,
        trace_qwen_mps,
        base_texts=BASE_TEXTS,
    )
    hooked, traced = hooks.read_value("r"), trace.read_value("r")
    # the forwards ran on the device asked for — both towers' weights are there
    # (nnsight places its model at first dispatch, so this is read after the trace)
    for bundle in (hooks_qwen_mps, trace_qwen_mps):
        module = getattr(bundle.model, "_module", bundle.model)
        assert next(module.parameters()).device.type == "mps", bundle.key
    assert_same(hooked.cpu(), traced.cpu(), f"block_output @ L{delta_layer} on mps")
