"""The family plugin contract — T1 and the tiny tier of T3.

**T1.** A fourth family (``tests/_helpers/synthetic_family.py``: a tiny decoder
whose tree shares no child name with the llama or GPT-2 trees) is registered
from a module outside ``causalab/neural/`` and serves reads and writes on the
residual and MLP sites through the unchanged reference engine. *Mutation:*
deleting one tap from its adapter turns the component into a **named registry
refusal** — the family, the component and what the family does serve — never
the bare ``AttributeError: … has no attribute 'o_proj'`` the pre-fix resolver
raised when a lookup ran before anyone asked whether the family had the
module.

**T3, tiny tier.** ``registry.inventory`` on ``tiny-random/qwen3.5-moe`` (4
layers: 3 Gated DeltaNet + 1 full attention) is the 4-layer analogue of
``Qwen/Qwen3.6-35B-A3B``'s counts, and it agrees with what the resolver serves at every
(layer, component) — the inventory is the one producer, the resolver its
consumer. *Mutation:* an inventory that listed ``attention_premix`` at a
DeltaNet layer fails the agreement check with the ``_FULL_ATTENTION_ONLY``
refusal. The A3B tier (40 / 40 / 10 / 30) is ``tests/golden/test_a3b_inventory.py``.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import pytest
import torch
from transformers import AutoTokenizer

import causalab.neural
from causalab.neural.engines.pytorch_hooks.loading import ModelBundle
from causalab.neural.shared import sites
from causalab.neural.shared.devices import DeviceMap
from causalab.neural.shared.sites import adapter_of, resolve_site
from causalab.protocol.rules.errors import ProtocolError
from causalab.protocol.registry import (
    CAPABILITIES,
    COMPONENT_STREAMS,
    FAMILIES,
    FamilyAdapter,
    Inventory,
    LayerInventory,
    Tap,
    family_for,
    identities_for,
    inventory,
    register_family,
)
from causalab.protocol.schema import (
    COMPONENTS,
    LAYERLESS_COMPONENTS,
    PROTOCOL_VERSION,
    SiteSpec,
)

from tests._helpers import synthetic_family as synth
from tests.neural.engines.pytorch_hooks._drive import base_data_section, executor_for
from tests.neural.engines.pytorch_hooks.conftest import TINY_LLAMA
from tests.protocol._docs import saved

pytestmark = pytest.mark.smoke

TEXT = "the quick brown fox jumps"
CF_TEXT = "a slow green turtle sleeps deeply"

#: The block-level sites the synthetic family declares (the residual stream and
#: the MLP), i.e. everything but the model boundary.
BLOCK_SITES = tuple(
    c for c in synth.SYNTHETIC_TREE.taps if c not in LAYERLESS_COMPONENTS
)


@pytest.fixture(scope="module")
def synthetic_bundle() -> ModelBundle:
    return ModelBundle(
        key=synth.KEY,
        revision="main",
        model=synth.build_model(),
        tokenizer=_tokenizer(),
        info=synth.INFO,
        devices=DeviceMap.parse("cpu", synth.INFO.num_layers),
        dtype="fp32",
    )


def _tokenizer() -> Any:
    tokenizer = AutoTokenizer.from_pretrained(TINY_LLAMA)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _read_doc(component: str, layer: int | None = 0) -> dict[str, Any]:
    site: dict[str, Any] = {"component": component}
    if layer is not None:
        site["layers"] = layer
    return {
        "header": {"protocol_version": PROTOCOL_VERSION},
        "model": {"key": synth.KEY, "revision": "main"},
        "data": base_data_section(with_counterfactual=False),
        "method": {
            "intervened_models": {"original": {"input": "base", "reads": ["r"]}},
            "sites": {"tap": site},
            "reads": {"r": {"site": "tap", "pos": "all"}},
            "save": [saved("r", "original", "a.safetensors")],
        },
    }


def _swap_doc(component: str, layer: int = 0) -> dict[str, Any]:
    return {
        "header": {"protocol_version": PROTOCOL_VERSION},
        "model": {"key": synth.KEY, "revision": "main"},
        "data": base_data_section(with_counterfactual=True),
        "method": {
            "intervened_models": {
                "original_counterfactual": {
                    "input": "counterfactual",
                    "reads": ["v_cf"],
                },
                "original_base": {"input": "base", "reads": ["clean"]},
                "patched": {"input": "base", "reads": ["after"], "writes": ["patch"]},
            },
            "sites": {
                "tap": {"component": component, "layers": [layer]},
                "lm_head": {"component": "lm_head"},
            },
            "reads": {
                "v_cf": {"site": "tap", "pos": -1},
                "clean": {"site": "lm_head", "pos": -1},
                "after": {"site": "lm_head", "pos": -1},
            },
            "writes": {"patch": {"site": "tap", "pos": -1, "do": {"swap": "v_cf"}}},
            "save": [
                saved("after", "patched", "p.safetensors"),
                saved("clean", "original_base", "c.safetensors"),
            ],
        },
    }


# --------------------------------------------------------------------------- #
# T1 — registered from outside causalab/neural/, detected, served
# --------------------------------------------------------------------------- #


def test_the_fourth_family_lives_outside_causalab_neural():
    neural = Path(causalab.neural.__file__).parent
    module = Path(synth.__file__).resolve()
    assert neural not in module.parents, module
    assert synth.FAMILY in FAMILIES
    assert FAMILIES[synth.FAMILY] is synth.SYNTHETIC_TREE


def test_the_family_is_detected_structurally(synthetic_bundle):
    """The bundle's adapter is the synthetic one — by the module tree, with no
    ``model_type`` anywhere (the entry's ``family`` is ``None``)."""
    assert synthetic_bundle.info.family is None
    assert synthetic_bundle.adapter.family == synth.FAMILY
    assert adapter_of(synthetic_bundle) is synth.SYNTHETIC_TREE
    assert not synthetic_bundle.is_gpt2_family
    assert len(synthetic_bundle.blocks) == synth.LAYERS
    assert synthetic_bundle.streams == ("full_attention",) * synth.LAYERS
    assert type(synthetic_bundle.mixer_at(0)).__name__ == "SynthMixer"


@pytest.mark.parametrize("component", BLOCK_SITES)
def test_every_declared_block_site_resolves_and_reads(synthetic_bundle, component):
    site = resolve_site(synthetic_bundle, SiteSpec(component=component, layers=(1,)))
    tap = synth.SYNTHETIC_TREE.taps[component]
    assert site.kind == tap.kind and site.layer == 1 and site.component == component
    value = executor_for(_read_doc(component, 1), synthetic_bundle, base_texts=[TEXT])
    read = value.read_value("r")
    width = synth.INNER if component == "mlp_activation" else synth.HIDDEN
    assert read.shape[0] == 1 and read.shape[-1] == width, (component, read.shape)


@pytest.mark.parametrize("component", sorted(LAYERLESS_COMPONENTS))
def test_every_model_boundary_site_resolves(synthetic_bundle, component):
    site = resolve_site(synthetic_bundle, SiteSpec(component=component))
    expected = {
        "input_ids": "Embedding",
        "embeddings": "Embedding",
        "ln_final": "LayerNorm",
        "lm_head": "Linear",
    }[component]
    assert type(site.module).__name__ == expected
    assert site.kind == ("in" if component == "input_ids" else "out")


@pytest.mark.parametrize(
    "component",
    [c for c in BLOCK_SITES if CAPABILITIES[c].writes is not None],
)
def test_a_swap_write_lands_on_every_writable_block_site(synthetic_bundle, component):
    """Writes reach the forward: a counterfactual swap moves the logits and a
    self-swap moves them by exactly nothing — on a family the engine had never
    seen before this module registered it."""

    def moved(doc: dict[str, Any]) -> float:
        executor = executor_for(
            doc, synthetic_bundle, base_texts=[TEXT], counterfactual_texts=[CF_TEXT]
        )
        after, clean = executor.read_value("after"), executor.read_value("clean")
        return float((after - clean).abs().max())

    assert moved(_swap_doc(component)) > 1e-6, component
    self_swap = _swap_doc(component)  # v_cf taken on the un-intervened base
    del self_swap["method"]["intervened_models"]["original_counterfactual"]
    self_swap["method"]["intervened_models"]["original_base"]["reads"].append("v_cf")
    assert moved(self_swap) == 0.0, component


def test_the_declared_identities_hold_on_the_new_family(synthetic_bundle):
    """The two residual identities the adapter declares, evaluated from the
    rows: which component, from which inputs, to which tolerance."""
    doc = _read_doc("block_input")
    others = ("attention_output", "block_mid", "mlp_output", "block_output")
    for name in others:
        doc["method"]["sites"][f"{name}_site"] = {"component": name, "layers": [0]}
        doc["method"]["reads"][f"read_{name}"] = {"site": f"{name}_site", "pos": "all"}
        doc["method"]["intervened_models"]["original"]["reads"].append(f"read_{name}")
        doc["method"]["save"].append(
            saved(f"read_{name}", "original", f"{name}.safetensors")
        )
    executor = executor_for(doc, synthetic_bundle, base_texts=[TEXT])
    values = {"block_input": executor.read_value("r")}
    values.update({n: executor.read_value(f"read_{n}") for n in others})
    declared = identities_for(synth.FAMILY)
    assert {i.name for i in declared} == {"residual_mid", "residual_out"}
    for identity in declared:
        left, right = identity.inputs
        atol, rtol = identity.tolerance_for(synthetic_bundle.dtype)
        torch.testing.assert_close(
            values[left] + values[right],
            values[identity.component],
            atol=atol,
            rtol=rtol,
        )


# --------------------------------------------------------------------------- #
# T1's mutation — a deleted row is a named registry refusal
# --------------------------------------------------------------------------- #


@pytest.fixture()
def restore_synthetic():
    yield
    register_family(synth.SYNTHETIC_TREE)


def _variant(**changes: Any) -> FamilyAdapter:
    return dataclasses.replace(synth.SYNTHETIC_TREE, **changes)


def test_a_deleted_tap_is_a_named_registry_refusal(synthetic_bundle, restore_synthetic):
    """Delete ``mlp_output`` from the family's taps: the component is refused by
    the registry — naming the family, the component and what it does serve,
    with the ``component_unavailable`` reason — and no module is ever looked
    up, so no ``AttributeError`` can escape."""
    taps = {c: t for c, t in synth.SYNTHETIC_TREE.taps.items() if c != "mlp_output"}
    register_family(_variant(taps=taps))
    bundle = dataclasses.replace(synthetic_bundle)  # a fresh detection
    assert bundle.adapter.tap_for("mlp_output") is None
    with pytest.raises(ProtocolError) as excinfo:
        resolve_site(bundle, SiteSpec(component="mlp_output", layers=(0,)))
    err = excinfo.value
    assert err.reason == "component_unavailable"
    assert f"family {synth.FAMILY!r} declares no tap" in str(err)
    assert "'mlp_output'" in str(err) and "'mlp_input'" in str(err)  # what it serves
    # the sibling still serves — the refusal is per component, not per family
    assert (
        resolve_site(bundle, SiteSpec(component="mlp_input", layers=(0,))).kind == "in"
    )


def test_an_attention_component_the_family_never_declared_is_refused_by_name(
    synthetic_bundle,
):
    """The pre-fix failure ``sites.py`` records — ``AttributeError: … has no
    attribute 'o_proj'`` out of the tap table — cannot happen: the family says
    which components it serves before any module is touched."""
    for component in ("attention_premix", "attention_result", "attention_query"):
        with pytest.raises(ProtocolError) as excinfo:
            resolve_site(synthetic_bundle, SiteSpec(component=component, layers=(0,)))
        assert excinfo.value.reason == "component_unavailable"
        assert "declares no tap" in str(excinfo.value), component


def test_a_tap_at_a_child_the_tree_lacks_is_refused_by_name(
    synthetic_bundle, restore_synthetic
):
    """The other way a family and a model disagree: the adapter names a child
    the block does not have. Refused naming the family's claim and the block's
    real children — the table and the model disagree — not an AttributeError."""
    taps = dict(synth.SYNTHETIC_TREE.taps)
    taps["attention_input_norm"] = Tap("block", "norm_missing")
    register_family(_variant(taps=taps))
    bundle = dataclasses.replace(synthetic_bundle)
    with pytest.raises(ProtocolError) as excinfo:
        resolve_site(bundle, SiteSpec(component="attention_input_norm", layers=(0,)))
    message = str(excinfo.value)
    assert excinfo.value.reason == "component_unavailable"
    assert "'block.norm_missing'" in message and "'norm_a'" in message


def test_a_tree_no_family_detects_is_refused_at_the_registry():
    with pytest.raises(ProtocolError, match="no registered model family detects"):
        family_for(torch.nn.Linear(2, 2))


def test_two_families_detecting_one_tree_is_refused_not_ordered(
    synthetic_bundle, restore_synthetic
):
    register_family(dataclasses.replace(synth.SYNTHETIC_TREE, family="synthetic_twin"))
    try:
        with pytest.raises(ProtocolError, match="all detect this module tree"):
            family_for(synthetic_bundle.model)
    finally:
        # the twin is a test artefact; the registry has no unregister on purpose
        from causalab.protocol import registry

        del registry._FAMILIES["synthetic_twin"]


# --------------------------------------------------------------------------- #
# T3, tiny tier — the inventory is what the resolver serves
# --------------------------------------------------------------------------- #

QWEN_STREAMS = ("linear_attention",) * 3 + ("full_attention",)


def _agrees_with_the_resolver(bundle: ModelBundle, inv: Inventory) -> list[str]:
    """Every listed (layer, component) resolves; every stream-bound component
    the inventory leaves off a layer is refused there with the architectural
    reason. The disagreements, as text."""
    failures: list[str] = []
    for layer in inv.layers:
        for component in layer.components:
            try:
                resolve_site(
                    bundle, SiteSpec(component=component, layers=(layer.layer,))
                )
            except ProtocolError as err:
                failures.append(f"listed but refused: {component}@{layer.layer}: {err}")
        for component, stream in COMPONENT_STREAMS.items():
            if component in layer.components or stream == layer.stream:
                continue
            try:
                resolve_site(
                    bundle, SiteSpec(component=component, layers=(layer.layer,))
                )
            except ProtocolError as err:
                if "mixer" not in str(err):
                    failures.append(
                        f"wrong refusal for {component}@{layer.layer}: {err}"
                    )
            else:
                failures.append(f"absent but served: {component}@{layer.layer}")
    return failures


def test_the_tiny_hybrid_inventory_is_hydras_counts_in_miniature(qwen35moe_bundle):
    inv = sites.inventory(qwen35moe_bundle)
    assert tuple(li.stream for li in inv.layers) == QWEN_STREAMS
    assert inv.count("linear_attention") == 3 and inv.count("full_attention") == 1
    # "40 attention + 40 MLP": one mixer output and one MLP output per layer
    assert inv.where("attention_output") == (0, 1, 2, 3)
    assert inv.where("mlp_output") == (0, 1, 2, 3)
    # each full-attention component exactly once — at the one full layer
    full_only = [c for c, s in COMPONENT_STREAMS.items() if s == "full_attention"]
    for component in full_only:
        assert inv.where(component) == (3,), component
    # each DeltaNet component at exactly the three DeltaNet layers
    linear_only = [c for c, s in COMPONENT_STREAMS.items() if s == "linear_attention"]
    for component in linear_only:
        assert inv.where(component) == (0, 1, 2), component
    # the MoE surface at every layer; the dense activation nowhere
    assert inv.where("expert_activation") == (0, 1, 2, 3)
    assert inv.where("mlp_activation") == ()
    assert inv.layerless == ("input_ids", "embeddings", "ln_final", "lm_head")
    # the mechanisms are the rows'
    layer3 = inv.layers[3]
    assert layer3.reads["attention_probs"] == {"pytorch_hooks", "nnsight"}
    assert layer3.writes["attention_probs"] == {"swap"}
    assert layer3.writes["attention_result"] is None


def test_the_inventory_agrees_with_the_resolver_at_every_cell(qwen35moe_bundle):
    failures = _agrees_with_the_resolver(
        qwen35moe_bundle, sites.inventory(qwen35moe_bundle)
    )
    assert not failures, "\n".join(failures)


def test_offline_and_served_inventories_agree_where_the_entry_knows(
    qwen35moe_bundle, bundle, synthetic_bundle
):
    """Compare static entries with the sites on each loaded model.

    The tiny qwen3.5-moe config includes `intermediate_size` for its MoE
    tower. Its static inventory includes `mlp_activation` and
    `mlp_neuron_output`. The loaded tree refuses these two dense sites.
    Refusal snapshot entry 29 records the original activation case.
    The A3B entry declares no dense width. Its inventories agree everywhere
    (`tests/golden/test_a3b_inventory.py`)."""
    for b in (bundle, synthetic_bundle):
        assert inventory(b) == sites.inventory(b)
    offline, served = inventory(qwen35moe_bundle), sites.inventory(qwen35moe_bundle)
    assert offline.layerless == served.layerless
    for a, s_ in zip(offline.layers, served.layers):
        assert set(a.components) - set(s_.components) == {
            "mlp_activation",
            "mlp_neuron_output",
        }, a.layer
        assert set(s_.components) <= set(a.components)


def test_an_inventory_listing_premix_at_a_deltanet_layer_fails_the_check(
    qwen35moe_bundle,
):
    """T3's mutation: ``attention_premix`` injected at layer 0 is caught by the
    agreement check with the ``_FULL_ATTENTION_ONLY`` refusal."""
    inv = sites.inventory(qwen35moe_bundle)
    layer0 = inv.layers[0]
    mutated = dataclasses.replace(
        inv,
        layers=(
            dataclasses.replace(
                layer0, components=(*layer0.components, "attention_premix")
            ),
            *inv.layers[1:],
        ),
    )
    failures = _agrees_with_the_resolver(qwen35moe_bundle, mutated)
    assert len(failures) == 1
    assert "listed but refused: attention_premix@0" in failures[0]
    assert "needs a full-attention mixer" in failures[0]


def test_the_inventory_of_the_dense_fixtures_agrees_too(bundle):
    inv = sites.inventory(bundle)
    assert all(li.stream == "full_attention" for li in inv.layers)
    assert len(inv.layers) == bundle.info.num_layers
    assert not _agrees_with_the_resolver(bundle, inv)
    # no MoE, no DeltaNet on a dense tower
    assert inv.where("router_scores") == () and inv.where("delta_qkv") == ()


def test_the_inventory_of_the_new_family_is_its_declared_taps(synthetic_bundle):
    inv = sites.inventory(synthetic_bundle)
    assert len(inv.layers) == synth.LAYERS
    for li in inv.layers:
        assert set(li.components) == set(BLOCK_SITES)
    assert set(inv.layerless) == set(LAYERLESS_COMPONENTS)
    assert not _agrees_with_the_resolver(synthetic_bundle, inv)
    assert isinstance(inv.layers[0], LayerInventory)
    assert set(COMPONENTS) > set(BLOCK_SITES)  # the vocabulary is the global one
