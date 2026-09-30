"""The Qwen3.5-MoE architecture must actually load.

Everything in the hookpoint-vocabulary work targets ``qwen3_5_moe`` (the text
tower of Qwen3.6-35B-A3B), which only exists from transformers 5.16. These tests
are the gate on that bump: they assert the architecture is importable, that the
engine's own loader reaches the text tower rather than the composite
vision-language model, and that the layer stack really is hybrid — because the
per-layer split is the assumption the hookpoint vocabulary builds on.

They deliberately assert *structure*, not activations: numerical behaviour is
pinned by the parity goldens, which this bump leaves untouched.
"""

from __future__ import annotations

import importlib
import re

import pytest

from .conftest import TINY_QWEN35_MOE

#: structural assertions on a tiny CPU model — no numerics, no GPU
pytestmark = pytest.mark.smoke

#: the first transformers release carrying ``models/qwen3_5_moe`` — the floor
#: ``pyproject.toml`` declares (``transformers>=5.16``) and the lock resolves
TRANSFORMERS_FLOOR = (5, 16)

_RELEASE = re.compile(r"^(\d+)\.(\d+)")


def release_meets_floor(
    version: str, floor: tuple[int, int] = TRANSFORMERS_FLOOR
) -> bool:
    """``True`` when ``version``'s leading ``MAJOR.MINOR`` is at or above ``floor``.

    Only the two leading numeric segments count, so a pre-release of a later
    major (``"6.0.0.dev0"``) passes and ``"5.15.9"`` is refused. A string that
    does not begin ``int.int`` is refused too: an unparseable version can never
    pass the gate by accident.
    """
    match = _RELEASE.match(version)
    if match is None:
        return False
    return (int(match.group(1)), int(match.group(2))) >= floor


def test_transformers_ships_the_architecture():
    """The installed transformers is at least 5.16 and carries ``qwen3_5_moe``.

    This is the gate on the floor, and it *fails* — it does not skip — when the
    lock slips back: the version is asserted against ``TRANSFORMERS_FLOOR`` and
    the architecture module is imported outright rather than through
    ``pytest.importorskip``, which turned a pinned-back lock into a green skip.
    On the locked 5.16.1 the same test is the positive witness: it passes.
    """
    import transformers

    assert release_meets_floor(transformers.__version__), (
        f"transformers {transformers.__version__} is below the "
        f"{'.'.join(map(str, TRANSFORMERS_FLOOR))} floor pyproject.toml declares; "
        "qwen3_5_moe does not exist there and the lock must not slip back"
    )
    # a missing module raises ModuleNotFoundError here, which is a failure
    modeling = importlib.import_module(
        "transformers.models.qwen3_5_moe.modeling_qwen3_5_moe"
    )
    assert hasattr(modeling, "Qwen3_5MoeForCausalLM")


@pytest.mark.parametrize(
    ("version", "ok"),
    [
        ("5.16.1", True),  # the locked release
        ("5.16", True),
        ("5.17.0", True),
        ("6.0.0.dev0", True),  # a later major's pre-release still clears the floor
        ("5.15.9", False),  # the last minor without qwen3_5_moe
        ("4.57.1", False),  # the parity goldens' capture context — below the floor
        ("dev", False),  # unparseable never passes
    ],
)
def test_the_floor_parse_refuses_what_is_below_5_16(version: str, ok: bool):
    """The negative half of the gate, without installing an old transformers."""
    assert release_meets_floor(version) is ok


def test_the_config_offers_a_text_tower_beside_the_vlm():
    """The reason ``AutoModelForCausalLM`` is the right entry point.

    ``qwen3_5_moe`` is registered with *both* auto classes. That is what lets the
    engine load a plain causal LM and ignore the vision tower entirely — and it
    is why nnsight's ``LanguageModel``, which refuses anything registered for
    image-text-to-text, cannot load this checkpoint without help.
    """
    from transformers.models.auto.modeling_auto import (
        MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
        MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES,
    )

    assert "qwen3_5_moe" in MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES
    assert "qwen3_5_moe" in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES


def test_loader_reaches_the_text_tower_not_the_vlm(qwen35moe_bundle):
    """``load_model`` must give us the causal LM, with no vision tower attached."""
    model = qwen35moe_bundle.model
    assert type(model).__name__ == "Qwen3_5MoeForCausalLM"
    assert not hasattr(model, "visual"), "the vision tower must not be loaded"
    assert hasattr(model, "lm_head")
    assert hasattr(model.model, "layers")


def test_the_layer_stack_is_hybrid(qwen35moe_bundle):
    """The assumption the whole vocabulary rests on: block type is per-layer.

    A DeltaNet layer carries ``linear_attn`` and no ``self_attn``; a full-attention
    layer carries the reverse. Any site resolver that reads ``block.self_attn``
    unconditionally is wrong on this model, and this test is what says so.
    """
    model = qwen35moe_bundle.model
    layer_types = getattr(model.config, "layer_types", None)
    assert layer_types is not None, "config must expose layer_types"
    assert set(layer_types) == {"linear_attention", "full_attention"}, layer_types

    layers = model.model.layers
    assert len(layers) == len(layer_types)
    for idx, (block, block_type) in enumerate(zip(layers, layer_types)):
        if block_type == "linear_attention":
            assert hasattr(block, "linear_attn"), idx
            assert not hasattr(block, "self_attn"), idx
        else:
            assert hasattr(block, "self_attn"), idx
            assert not hasattr(block, "linear_attn"), idx


def test_every_layer_has_a_sparse_moe_block(qwen35moe_bundle):
    """Unlike the token mixer, the channel mixer is uniform across the stack."""
    for idx, block in enumerate(qwen35moe_bundle.model.model.layers):
        mlp = block.mlp
        assert type(mlp).__name__ == "Qwen3_5MoeSparseMoeBlock", idx
        # the four sub-taps the MoE components resolve against
        for attr in ("gate", "experts", "shared_expert", "shared_expert_gate"):
            assert hasattr(mlp, attr), (idx, attr)


def test_model_info_unwraps_the_composite_config(qwen35moe_bundle):
    """The composite config nests text fields; ``ModelInfo`` must see through it."""
    info = qwen35moe_bundle.info
    assert info.hidden_size > 0
    assert info.num_experts is not None and info.num_experts > 1, (
        "router_logits width comes from num_experts; a composite config must not "
        "hide it"
    )


def test_fixture_key_is_the_documented_one():
    """Guards against the fixture silently drifting to another checkpoint."""
    assert TINY_QWEN35_MOE == "tiny-random/qwen3.5-moe"


def test_the_registry_layer_pattern_is_the_module_probe(qwen35moe_bundle):
    """The two halves of the stream check must agree on every layer.

    The canonicalizer refuses a stream-bound component against
    ``ModelInfo.layer_types`` (read from the config), the site resolver against
    the mixer child the block actually carries. This is the one place both
    answers exist for one model, so it is where a disagreement would show.
    """
    assert qwen35moe_bundle.info.layer_types == qwen35moe_bundle.streams
