"""Golden tier — the family inventory's acceptance on the real Qwen3.6-35B-A3B.

    the inventory must enumerate 40 attention and 40 MLP components exactly
    once, label 10 full-attention and 30 DeltaNet layers, and reject invalid
    stream and site combinations

``registry.inventory`` on the loaded tower (the streams read off the modules,
the family detected structurally) against the offline inventory of the
registry entry: the two must agree layer for layer, and both must be the
counts above. The tiny tier is
``tests/neural/engines/pytorch_hooks/test_family_plugin.py``. Needs an
accelerator and the real weights (``-m golden``).
"""

from __future__ import annotations

import pytest
import torch

from causalab.neural.engines.pytorch_hooks.loading import load_model
from causalab.neural.shared.sites import resolve_site
from causalab.protocol.rules.errors import ProtocolError
from causalab.protocol.registry import (
    COMPONENT_STREAMS,
    DOCS_TABLE_MODEL,
    get_model_info,
    inventory,
)
from causalab.protocol.schema import SiteSpec

pytestmark = pytest.mark.golden

FULL_LAYERS = tuple(range(3, 40, 4))


@pytest.fixture(scope="module")
def bundle():
    if not torch.cuda.is_available():
        pytest.skip("the A3B inventory tier needs an accelerator")
    return load_model(DOCS_TABLE_MODEL, dtype="bf16", device="cuda")


def test_the_loaded_tower_is_forty_forty_ten_thirty(bundle):
    inv = inventory(bundle)
    assert bundle.adapter.family == "llama_tree"
    assert len(inv.layers) == 40
    assert inv.count("full_attention") == 10 and inv.count("linear_attention") == 30
    assert inv.where("attention_premix") == FULL_LAYERS
    assert len(inv.where("attention_output")) == 40  # 40 attention (mixer) components
    assert len(inv.where("mlp_output")) == 40  # 40 MLP components
    assert inv.where("delta_premix") == tuple(
        i for i in range(40) if i not in FULL_LAYERS
    )
    assert inv.where("mlp_activation") == ()


def test_the_loaded_inventory_equals_the_offline_one(bundle):
    """The entry's `layer_types` and the modules agree, so `validate` (offline)
    and the run (loaded) enumerate the same inventory."""
    loaded, offline = inventory(bundle), inventory(get_model_info(DOCS_TABLE_MODEL))
    assert [li.stream for li in loaded.layers] == [li.stream for li in offline.layers]
    for a, b in zip(loaded.layers, offline.layers):
        assert a.components == b.components, a.layer
    assert loaded.layerless == offline.layerless


@pytest.mark.parametrize(
    ("component", "layer"),
    [
        ("attention_premix", 0),
        ("attention_probs", 4),
        ("delta_premix", 3),
        ("deltanet_state", 39),
    ],
)
def test_an_invalid_stream_site_combination_is_refused(bundle, component, layer):
    inv = inventory(bundle)
    assert component not in inv.layers[layer].components
    assert COMPONENT_STREAMS[component] != inv.layers[layer].stream
    with pytest.raises(ProtocolError, match="mixer") as excinfo:
        resolve_site(bundle, SiteSpec(component=component, layers=(layer,)))
    assert excinfo.value.reason == "component_unavailable"
