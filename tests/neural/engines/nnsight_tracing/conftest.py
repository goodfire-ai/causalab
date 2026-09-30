"""Fixtures for the nnsight engine's parity suite.

Both engines load the same checkpoints in fp32 with eager attention, so
parity is like-against-like: any disagreement is an executor bug, not a
kernel or dtype story. The whole directory skips when the ``nnsight`` extra
is not installed.

The MPS guard: nnsight's dispatch places models itself and, on a Mac, lands
them on ``mps:0`` regardless of ``device_map`` (measured) —
which would compare MPS numerics against the reference engine's CPU ones.
CI has no MPS; this guard makes local runs match it.
"""

from __future__ import annotations

import pytest
import torch

nnsight = pytest.importorskip("nnsight")

# must run before the first dispatch — see the module docstring
torch.backends.mps.is_available = lambda: False  # type: ignore[method-assign]

from causalab.neural.engines.nnsight_tracing.loading import (  # noqa: E402
    NnsightBundle,
)
from causalab.neural.engines.nnsight_tracing.loading import (  # noqa: E402
    load_model as load_trace_model,
)
from causalab.neural.engines.pytorch_hooks.loading import (  # noqa: E402
    ModelBundle,
)
from causalab.neural.engines.pytorch_hooks.loading import (  # noqa: E402
    load_model as load_hooks_model,
)

TINY_LLAMA = "hf-internal-testing/tiny-random-LlamaForCausalLM"
TINY_QWEN35_MOE = "tiny-random/qwen3.5-moe"


@pytest.fixture(scope="session")
def hooks_llama() -> ModelBundle:
    return load_hooks_model(TINY_LLAMA)


@pytest.fixture(scope="session")
def trace_llama() -> NnsightBundle:
    # eager pinned explicitly: the reference engine loads eager, and parity
    # must compare like against like (§7.3)
    return load_trace_model(TINY_LLAMA, attn_implementation="eager")


@pytest.fixture(scope="session")
def hooks_qwen() -> ModelBundle:
    return load_hooks_model(TINY_QWEN35_MOE)


@pytest.fixture(scope="session")
def trace_qwen() -> NnsightBundle:
    return load_trace_model(TINY_QWEN35_MOE, attn_implementation="eager")


@pytest.fixture(scope="session")
def trace_qwen_default_impl() -> NnsightBundle:
    """The unpinned path: the checkpoint's own default (sdpa) — what the
    engine's loader gives a real document, and what the on-demand switch is
    tested against."""
    return load_trace_model(TINY_QWEN35_MOE)
