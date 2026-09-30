"""The eviction the golden conftest applies at module boundaries (CPU guard).

Runs in the CPU tier against *fresh* loader caches installed with
``monkeypatch`` — the helper resolves the loaders at call time, so it evicts
whatever the modules currently expose — because the CPU tiers' session
fixtures rely on the real cache's identity (a hooked ``bundle`` must be the
object the engine's own ``load_model`` call returns) and clearing it
mid-session would make an unrelated test flaky. The GPU half — that the
boundary eviction lets ``test_readout_a3b.py`` fit after the paper goldens —
is what the golden tier, on a CUDA device, measures.
"""

from __future__ import annotations

import pytest

from causalab.neural.engines.nnsight_tracing import loading as nnsight_loading
from causalab.neural.engines.pytorch_hooks import loading as hooks_loading
from tests._helpers.resident_models import evict_resident_models

pytestmark = pytest.mark.unit

TINY_LLAMA = "hf-internal-testing/tiny-random-LlamaForCausalLM"


@pytest.fixture
def fresh_caches(monkeypatch: pytest.MonkeyPatch):
    """Empty loader caches for this test alone, the session's bundles untouched."""
    loaders = []
    for module in (hooks_loading, nnsight_loading):
        fresh = module.load_model.renewed()
        monkeypatch.setattr(module, "load_model", fresh)
        loaders.append(fresh)
    return tuple(loaders)


def test_eviction_empties_both_loader_caches_and_the_next_load_is_fresh(fresh_caches):
    hooks, nnsight = fresh_caches
    first = hooks(TINY_LLAMA)
    assert hooks(TINY_LLAMA) is first, (
        "the loader is cached; the test would prove nothing otherwise"
    )
    assert hooks.cache_info().currsize == 1

    evict_resident_models()

    assert hooks.cache_info().currsize == 0
    assert nnsight.cache_info().currsize == 0
    assert hooks(TINY_LLAMA) is not first


def test_eviction_is_idempotent_on_empty_caches(fresh_caches):
    hooks, nnsight = fresh_caches
    evict_resident_models()
    evict_resident_models()
    assert hooks.cache_info().currsize == 0
    assert nnsight.cache_info().currsize == 0
