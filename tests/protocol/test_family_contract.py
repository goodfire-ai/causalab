"""The family plugin contract, torch-free.

What a family declares (``registry.FamilyAdapter``), the built-in two, the
typed backend pairs and the alias rule they are held to, the ``grouped_mm``
predicate decided at load, and the offline inventory of the
``Qwen/Qwen3.6-35B-A3B`` entry — its 40 / 40 / 10 / 30 counts, from the
registry alone.
"""

from __future__ import annotations

import dataclasses
import re
import types
from typing import Any

import pytest

from causalab.protocol.schema.explicit import canonicalize
from causalab.protocol.rules.document import validate_document
from causalab.protocol.rules.errors import ProtocolError, ValidationError
from causalab.protocol.registry import (
    BACKEND_PAIRS,
    CAPABILITIES,
    COMPONENT_STREAMS,
    DOCS_TABLE_MODEL,
    FAMILIES,
    GPT2_TREE,
    GPTJ_TREE,
    HOOK_KINDS,
    INTERIOR_ROWS,
    LLAMA_TREE,
    PREDICATES,
    RELATIONS,
    TAP_SCOPES,
    BackendPair,
    FamilyAdapter,
    Identity,
    ModelInfo,
    Tap,
    TreeAddress,
    _check_aliases,
    alias_would_rebind,
    backend_pair,
    component_shape,
    family,
    family_for,
    get_model_info,
    inventory,
    mixer_children,
    predicate_holds,
    register_family,
    register_model,
    site_group_map,
    unavailable_at_load,
)
from causalab.protocol.schema import (
    COMPONENTS,
    DEPRECATED_COMPONENTS,
    DEPRECATED_IN,
    LAYERLESS_COMPONENTS,
    STREAMS,
    parse_document,
)

from tests.protocol._docs import base_doc, in_order
from tests.protocol._env import FIXTURES, build_env


pytestmark = pytest.mark.unit

ENV = build_env(FIXTURES / "artifacts")
A3B = get_model_info(DOCS_TABLE_MODEL)


def _validate_sites(raw: dict[str, Any]) -> None:
    """Rule 4's address half — a site's layer, stream, component and head
    against the model's static metadata — is the checklist's since the rules
    package: ``validate_document`` decides it
    with the environment's ``model_info``, and ``canonicalize`` no longer
    refuses an illegal site (the compiler's ``validate`` stage runs first)."""
    validate_document(parse_document(in_order(raw)), model_info=ENV.model_info)


MOE_ROWS = {c for c, r in CAPABILITIES.items() if "moe" in r.requires}
DELTA_ROWS = {c for c, s in COMPONENT_STREAMS.items() if s == "linear_attention"}


# --------------------------------------------------------------------------- #
# the contract: what a family declares
# --------------------------------------------------------------------------- #


def _ns(**children: Any) -> types.SimpleNamespace:
    return types.SimpleNamespace(**children)


#: A GPT-2 block and a GPT-J block, by their children alone: the two trees
#: share the root (``transformer.h``) and differ inside the block.
GPT2_BLOCK = _ns(
    ln_1=object(),
    ln_2=object(),
    attn=_ns(c_attn=object(), c_proj=object()),
    mlp=_ns(c_fc=object(), c_proj=object()),
)
GPTJ_BLOCK = _ns(
    ln_1=object(),
    attn=_ns(q_proj=object(), k_proj=object(), v_proj=object(), out_proj=object()),
    mlp=_ns(fc_in=object(), fc_out=object()),
)


def _transformer_root(*blocks: Any) -> types.SimpleNamespace:
    return _ns(transformer=_ns(h=list(blocks)))


def test_the_three_built_in_trees_are_registered_and_detect_by_structure():
    assert {"llama_tree", "gpt2_tree", "gptj_tree"} <= set(FAMILIES)
    assert family("llama_tree") is LLAMA_TREE and family("gpt2_tree") is GPT2_TREE
    assert family("gptj_tree") is GPTJ_TREE
    llama_like = types.SimpleNamespace(
        model=types.SimpleNamespace(layers=[], embed_tokens=object(), norm=object())
    )
    gpt2_like = _transformer_root(GPT2_BLOCK)
    assert family_for(llama_like) is LLAMA_TREE
    assert family_for(gpt2_like) is GPT2_TREE
    assert family_for(_transformer_root(GPTJ_BLOCK)) is GPTJ_TREE
    both = types.SimpleNamespace(
        model=llama_like.model, transformer=gpt2_like.transformer
    )
    with pytest.raises(ProtocolError, match="all detect this module tree"):
        family_for(both)
    with pytest.raises(ProtocolError, match="no registered model family detects"):
        family_for(types.SimpleNamespace())
    with pytest.raises(ProtocolError, match="no model family named 'mamba_tree'"):
        family("mamba_tree")


def test_the_llama_tree_serves_the_whole_vocabulary_and_gpt2_the_dense_part():
    """Per-family availability over the one global vocabulary: the tree
    the hybrid lives in declares every component; the GPT-2 tree declares no
    MoE and no DeltaNet tap — those are refused first by the architecture
    (stream, ``moe``) and, were a tree to pass those, by the family."""
    assert set(LLAMA_TREE.taps) == set(COMPONENTS)
    assert not (set(GPT2_TREE.taps) & MOE_ROWS)
    assert not (set(GPT2_TREE.taps) & DELTA_ROWS)
    assert set(GPT2_TREE.taps) == set(COMPONENTS) - MOE_ROWS - DELTA_ROWS
    for adapter in (LLAMA_TREE, GPT2_TREE):
        for component in INTERIOR_ROWS:
            assert adapter.taps[component].from_row  # the rows stay the address
        assert all(tap.kind in HOOK_KINDS for tap in adapter.taps.values())
        assert all(tap.scope in TAP_SCOPES for tap in adapter.taps.values())


def test_the_two_trees_differ_where_the_module_trees_differ():
    differ = {c for c in GPT2_TREE.taps if GPT2_TREE.taps[c] != LLAMA_TREE.taps[c]}
    assert differ == {
        "attention_input_norm",
        "block_mid",
        "mlp_input_norm",
        "attention_premix",
        "attention_result",
        "mlp_activation",
        "mlp_neuron_output",
    }
    assert GPT2_TREE.taps["mlp_activation"] == Tap("mlp", "c_proj", "in")
    assert LLAMA_TREE.taps["mlp_activation"] == Tap("mlp", "act_fn")
    assert (
        GPT2_TREE.tree.blocks == "transformer.h"
        and LLAMA_TREE.tree.blocks == "model.layers"
    )


def test_the_transformer_root_is_told_apart_by_its_blocks():
    """🐞 GPT-2's predicate was ``transformer.h`` alone, which every tree on
    that root satisfies. The block's children decide now: GPT-2's
    ``ln_2`` + ``attn.c_proj``, GPT-J's split projections + ``fc_out`` with no
    ``ln_2``. A root no predicate recognizes is refused, never guessed."""
    assert not GPT2_TREE.detect(_transformer_root(GPTJ_BLOCK))
    assert not GPTJ_TREE.detect(_transformer_root(GPT2_BLOCK))
    # a stand-in first block (a pipeline stage's identity) does not hide the tree
    assert family_for(_transformer_root(object(), GPTJ_BLOCK)) is GPTJ_TREE
    assert family_for(_transformer_root(object(), GPT2_BLOCK)) is GPT2_TREE
    # GPT-J's attention with a second norm is a sequential block: not GPT-J
    sequential = _ns(**{**vars(GPTJ_BLOCK), "ln_2": object()})
    # GPT-Neo's shape: ln_2, but the projections under attn.attention
    neo = _ns(
        ln_1=object(),
        ln_2=object(),
        attn=_ns(attention=_ns(out_proj=object())),
        mlp=_ns(c_fc=object(), c_proj=object()),
    )
    for block in (sequential, neo):
        with pytest.raises(ProtocolError, match="no registered model family detects"):
            family_for(_transformer_root(block))
    with pytest.raises(ProtocolError, match="no registered model family detects"):
        family_for(_transformer_root())


#: GPT-J's taps, spelled from ``modeling_gptj.py`` (``GPTJBlock``,
#: ``GPTJAttention``, ``GPTJMLP``) rather than read off the adapter.
GPTJ_EXPECTED_TAPS = {
    "attention_input_norm": Tap("block", "ln_1"),
    "attention_premix": Tap("mixer", "out_proj", "in"),
    "mlp_activation": Tap("mlp", "fc_out", "in"),
    "mlp_neuron_output": Tap("mlp", "fc_out", "in"),
    "attention_output": Tap("mixer"),
    "mlp_input": Tap("mlp", kind="in"),
    "mlp_output": Tap("mlp"),
    "block_input": Tap("block", kind="in"),
    "block_output": Tap("block"),
}


def test_the_gptj_tree_serves_the_parallel_block():
    """The parallel block has one norm and one residual add: no ``block_mid``,
    no ``mlp_input_norm``. Its attention never calls the transformers
    attention interface, so the function slots and the pattern are not
    declared either. Everything else of the dense vocabulary is."""
    taps = GPTJ_TREE.taps
    for component, tap in GPTJ_EXPECTED_TAPS.items():
        assert taps[component] == tap, component
    not_declared = {
        "block_mid",
        "mlp_input_norm",
        "attention_probs",
        "attention_query",
        "attention_key",
        "attention_scores",
        "attention_z",
    }
    assert not (set(taps) & not_declared)
    assert not (set(taps) & (MOE_ROWS | DELTA_ROWS))
    assert set(taps) == set(GPT2_TREE.taps) - not_declared
    assert taps["attention_result"].derivation == "attention_result"
    assert taps["attention_result"].shape_of == "attention_premix"
    assert all(taps[c].from_row for c in INTERIOR_ROWS)
    assert GPTJ_TREE.tree == GPT2_TREE.tree
    assert GPTJ_TREE.mixers == {"attn": "full_attention"}


def test_the_gptj_identity_is_the_parallel_residual_in_the_blocks_order():
    (parallel,) = GPTJ_TREE.identities
    assert parallel.name == "residual_parallel" and parallel.additive
    assert parallel.component == "block_output"
    # GPTJBlock.forward: attn_outputs + feed_forward_hidden_states + residual
    assert parallel.inputs == ("attention_output", "mlp_output", "block_input")
    assert parallel.tolerance_for("fp32") == (0.0, 0.0)
    with pytest.raises(ValueError, match="no tolerance for dtype 'bf16'"):
        parallel.tolerance_for("bf16")


def test_the_gpt_j_6b_entry_has_sixteen_heads_of_256():
    """The registry row (the ``float16`` revision's config.json) and what the
    rows derive from it offline: the head-major premix, the head-grouped
    gate's map, the MLP width."""
    info = get_model_info("EleutherAI/gpt-j-6b")
    assert (info.hidden_size, info.num_layers, info.vocab_size) == (4096, 28, 50400)
    assert (info.num_heads, info.num_kv_heads, info.head_dim) == (16, 16, 256)
    assert info.intermediate_size == 4 * 4096
    assert info.native_dtype == "fp16" and info.family == "gptj"
    assert info.parallel_plan is not None and info.parallel_plan.empty
    premix = component_shape(info, "attention_premix")
    assert premix.head_space == 16 and premix.width == 4096
    assert site_group_map(info, "head", "attention_premix") == (16, 256)
    assert component_shape(info, "mlp_activation").width == 16384


def test_mixer_children_are_the_union_and_the_stream_table_reads_them():
    from causalab.neural.shared import model_tree

    children = mixer_children()
    assert children["self_attn"] == "full_attention"
    assert children["attn"] == "full_attention"
    assert children["linear_attn"] == "linear_attention"
    assert set(children.values()) <= set(STREAMS)
    assert set(model_tree.FULL_ATTENTION_CHILDREN) >= {"self_attn", "attn"}
    # GPT-2 and GPT-J share the child name; the table lists it once
    assert len(set(model_tree.FULL_ATTENTION_CHILDREN)) == len(
        model_tree.FULL_ATTENTION_CHILDREN
    )
    assert model_tree.LINEAR_ATTENTION_CHILDREN == ("linear_attn",)


def test_a_tap_is_validated_at_construction():
    with pytest.raises(ValueError, match="tap scope"):
        Tap("layer")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="tap kind"):
        Tap("block", kind="hook")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="dotted child path"):
        Tap("block", "ln-1")
    with pytest.raises(ValueError, match="names its slot"):
        Tap("mixer", kind="delta")
    with pytest.raises(ValueError, match="from_row tap is the mixer's"):
        Tap("block", "x", from_row=True)
    with pytest.raises(ValueError, match="'in' tap"):
        Tap("block", "norm", writeback="block_output")
    with pytest.raises(ValueError, match="not a component"):
        Tap("block", "norm", "in", writeback="")
    with pytest.raises(ValueError, match="dotted path"):
        TreeAddress(blocks="", embedding="e", final_norm="n")


def _adapter(**changes) -> FamilyAdapter:
    base = dict(
        family="probe_tree",
        detect=lambda model: False,
        tree=TreeAddress(blocks="a.b", embedding="a.e", final_norm="a.n"),
        mixers={"mix": "full_attention"},
        taps={"block_output": Tap("block")},
    )
    base.update(changes)
    return FamilyAdapter(**base)


def test_an_adapter_is_validated_at_construction():
    assert _adapter().serves("block_output") and not _adapter().serves("lm_head")
    with pytest.raises(ValueError, match="capability row first"):
        _adapter(taps={"my_new_tensor": Tap("block")})
    with pytest.raises(ValueError, match="mixer"):
        _adapter(mixers={"mix": "mamba"})
    with pytest.raises(ValueError, match="declares no mixer child"):
        _adapter(mixers={})
    with pytest.raises(ValueError, match="probes for"):
        _adapter(probes={"has_router": lambda *_: None})
    with pytest.raises(ValueError, match="not an interior row"):
        _adapter(taps={"block_output": Tap("mixer", from_row=True)})
    with pytest.raises(ValueError, match="not an identifier"):
        _adapter(family="probe tree")
    with pytest.raises(ValueError, match="not a component"):
        Identity("i", "block_output", ("nothing",), "f", {"fp32": (0.0, 0.0)})
    with pytest.raises(ValueError, match="declares no tolerance"):
        Identity("i", "block_output", ("block_input",), "f", {})
    ident = Identity("i", "block_output", ("block_input",), "f", {"fp32": (0.0, 0.0)})
    with pytest.raises(ValueError, match="no tolerance for dtype 'bf16'"):
        ident.tolerance_for("bf16")


def test_a_mixer_child_means_one_stream_across_families():
    from causalab.protocol import registry

    clash = _adapter(family="clash_tree", mixers={"self_attn": "linear_attention"})
    with pytest.raises(ValueError, match="a mixer child name means one stream"):
        register_family(clash)
    assert "clash_tree" not in FAMILIES
    fine = _adapter(family="fine_tree", mixers={"mix": "full_attention"})
    register_family(fine)
    try:
        assert FAMILIES["fine_tree"] is fine
        assert mixer_children()["mix"] == "full_attention"
    finally:
        del registry._FAMILIES["fine_tree"]


def test_a_family_with_no_identity_declares_none():
    assert GPT2_TREE.identity_for("routed_output") is None
    assert {i.name for i in GPT2_TREE.identities} == {"residual_mid", "residual_out"}
    assert {i.name for i in LLAMA_TREE.identities} == {
        "residual_mid",
        "residual_out",
        "routed_sum",
        "delta_state_recurrence",
    }
    from causalab.protocol.registry import identity

    with pytest.raises(ProtocolError, match="declares no reconstruction identity"):
        identity("gpt2_tree", "routed_output")
    assert identity("llama_tree", "delta_state").tolerance_for("fp32") == (0.0, 0.0)


# --------------------------------------------------------------------------- #
# eight aliases redirect; three pairs stay two typed names
# --------------------------------------------------------------------------- #


_RESIDUAL_OUT = Identity(
    "residual_out",
    "block_output",
    ("block_mid", "mlp_output"),
    "block_output == block_mid + mlp_output",
    {"fp32": (0.0, 0.0)},
    additive=True,
)

#: a counterexample: identical scope/child/target shape to
#: ``block_mid``, but the target is a *function* of the tap, not a sum over it
_O_PROJ = Identity(
    "o_proj_fn",
    "attention_output",
    ("attention_premix",),
    "attention_output == attention_premix @ W_o",
    {"fp32": (0.0, 0.0)},
)


def test_additive_is_checked_against_the_formula_not_trusted():
    """The flag says "the plain sum of its inputs", and that is what lets a
    write at one addend be carried as an *unscaled* delta — so a formula that
    is not that sum cannot claim it."""
    # claimed but not a sum
    with pytest.raises(ValueError, match="additive=True"):
        dataclasses.replace(_O_PROJ, additive=True)
    with pytest.raises(ValueError, match="additive=True"):
        Identity(
            "scaled",
            "block_output",
            ("block_mid", "mlp_output"),
            "block_output == 0.5 * block_mid + mlp_output",
            {"fp32": (0.0, 0.0)},
            additive=True,
        )
    # ...and a sum that does not claim it, which is how the requirement went
    # missing on the synthetic family: the flag has teeth, so the formula may
    # not disagree with it in either direction
    with pytest.raises(ValueError, match="so it is additive"):
        dataclasses.replace(_RESIDUAL_OUT, additive=False)
    # the near miss with the flag *omitted* — the silent direction, and what
    # the whitespace-normalized comparison exists for: `b+c` is the same claim
    # as `b + c`, so it cannot read as a non-sum and quietly drop the
    # write-back requirement
    with pytest.raises(ValueError, match="so it is additive"):
        Identity(
            "residual_out",
            "block_output",
            ("block_mid", "mlp_output"),
            "block_output == block_mid+mlp_output",
            {"fp32": (0.0, 0.0)},
        )
    # the repo's two non-sums are unaffected
    assert {i.name for i in LLAMA_TREE.identities if not i.additive} == {
        "routed_sum",
        "delta_state_recurrence",
    }
    assert {i.name for i in LLAMA_TREE.identities if i.additive} == {
        "residual_mid",
        "residual_out",
    }


def test_a_declared_writeback_names_a_component_the_family_taps():
    """Unconditional, because ``sites._writeback`` reads the target's module,
    depth and payload element off that component's own tap for *every*
    declared write-back — identity or not."""
    # no identities at all, and it is still refused at construction
    with pytest.raises(ValueError, match="does not tap"):
        _adapter(
            taps={"block_mid": Tap("block", "norm_b", "in", writeback="block_output")}
        )
    # and a write-back to its own component is degenerate: the value the
    # enclosing forward saved before this tap cannot be this tap
    with pytest.raises(ValueError, match="write-back to itself"):
        _adapter(
            taps={"block_mid": Tap("block", "norm_b", "in", writeback="block_mid")}
        )


def test_a_writeback_target_is_a_plain_module_output_boundary():
    """``sites._writeback`` resolves the target with ``_tap_module`` alone, so
    the only target it reproduces is one ``resolve_site`` would serve from that
    same call — the generic module-boundary branch. ``kind == "out"`` is
    necessary but not sufficient: three earlier branches of that dispatch take
    ``out`` taps and yield a different module or a different payload element
    than the component names, and each is reachable by declaration."""
    cases = {
        # an input tap resolves to the enclosing module, and would order the
        # landing before the write it is a delta of
        "its tap is 'in'": {
            "block_input": Tap("block", kind="in"),
            "block_mid": Tap("block", "norm_b", "in", writeback="block_input"),
        },
        # element 1 of the mixer's output is the attention pattern — the one
        # element attention_interface.py exists to keep writes away from
        "function slot 'probs'": {
            "attention_probs": Tap("mixer", tuple_index=1, slot="probs"),
            "block_mid": Tap("block", "norm_b", "in", writeback="attention_probs"),
        },
        # quieter and worse: resolves to a real module (the mixer) while the
        # component names an interior line
        "is from_row": {
            "attention_gate": Tap("mixer", from_row=True),
            "block_mid": Tap("block", "norm_b", "in", writeback="attention_gate"),
        },
        # served by the experts dispatch, not as a module boundary
        "routed (MoE) component": {
            "expert_output": Tap("mlp"),
            "block_mid": Tap("block", "norm_b", "in", writeback="expert_output"),
        },
    }
    for expected, taps in cases.items():
        with pytest.raises(ValueError, match=re.escape(expected)):
            _adapter(taps=taps)


def test_every_declared_writeback_lands_downstream_of_its_site():
    """The assumption the synthetic landing is built on: the delta is issued
    at the target's rank, so the target has to come *after* the site in the
    forward. It cannot be checked in `registry` (``plan`` imports it, not the
    reverse), so it is pinned here."""
    from causalab.protocol.positions.alignment import COMPONENT_RANK

    declared = [
        (adapter.family, component, tap.writeback)
        for adapter in FAMILIES.values()
        for component, tap in adapter.taps.items()
        if tap.writeback is not None
    ]
    assert declared, "no family declares a write-back — this test would be vacuous"
    for family_name, component, target in declared:
        assert COMPONENT_RANK[target] > COMPONENT_RANK[component], (
            f"{family_name}: {component} writes back to {target}, which is not "
            "downstream of it"
        )


def test_an_addend_of_an_additive_identity_declares_its_writeback():
    """A known bug class (a pre-MLP write that missed the residual stream),
    refused on the declaration rather than remembered.

    A write at a component tapped as the input of a *child* module cannot
    reach the value the enclosing forward already saved for the addition, so a
    family declaring an additive identity over it and no ``writeback`` ships a
    site whose write silently breaks the identity it declares — while a clean
    forward still satisfies it, which is what makes the omission easy to
    miss."""
    taps = {
        "block_output": Tap("block"),
        "mlp_output": Tap("mlp"),
        "block_mid": Tap("block", "norm_b", "in"),
    }
    with pytest.raises(ValueError, match="must also reach 'block_output'"):
        _adapter(taps=taps, identities=(_RESIDUAL_OUT,))
    wrong = dict(taps, block_mid=Tap("block", "norm_b", "in", writeback="mlp_output"))
    with pytest.raises(ValueError, match="makes 'block_output' the value"):
        _adapter(taps=wrong, identities=(_RESIDUAL_OUT,))
    good = dict(taps, block_mid=Tap("block", "norm_b", "in", writeback="block_output"))
    assert (
        _adapter(taps=good, identities=(_RESIDUAL_OUT,)).tap_for("block_mid").writeback
        == "block_output"
    )


def test_only_an_additive_identity_asks_for_a_writeback():
    """The discriminator is the declared ``additive``, not the tap's shape.

    ``attention_premix`` is an ``in`` tap on a child whose module's output is
    ``attention_output`` — structurally identical to ``block_mid``. But a
    write at ``o_proj``'s input propagates through ``o_proj`` on its own, so
    requiring a write-back there would add the delta to a value that already
    reflects it. Nothing in the shape separates the two; the flag does."""
    functional = {
        "attention_output": Tap("mixer"),
        "attention_premix": Tap("mixer", "o_proj", "in"),
    }
    adapter = _adapter(taps=functional, identities=(_O_PROJ,))
    assert adapter.tap_for("attention_premix").writeback is None
    # an `in` tap on the scope module itself is exempt too: nothing saved yet
    residual_mid = Identity(
        "residual_mid",
        "block_mid",
        ("block_input", "attention_output"),
        "block_mid == block_input + attention_output",
        {"fp32": (0.0, 0.0)},
        additive=True,
    )
    exempt = _adapter(
        taps={
            "block_input": Tap("block", kind="in"),
            "block_mid": Tap("block", "norm_b", "in", writeback="block_output"),
            "block_output": Tap("block"),
            "mlp_output": Tap("mlp"),
            "attention_output": Tap("mixer"),
        },
        identities=(residual_mid, _RESIDUAL_OUT),
    )
    assert exempt.tap_for("block_input").writeback is None


def test_an_addend_of_two_additive_identities_is_refused_as_ambiguous():
    """One tap declares one target, so a write owing a delta to two is
    refused rather than resolved by declaration order."""
    second = Identity(
        "also_additive",
        "ln_final",
        ("block_mid", "mlp_output"),
        "ln_final == block_mid + mlp_output",
        {"fp32": (0.0, 0.0)},
        additive=True,
    )
    with pytest.raises(ValueError, match="an addend of"):
        _adapter(
            taps={
                "block_output": Tap("block"),
                "ln_final": Tap("final_norm"),
                "mlp_output": Tap("mlp"),
                "block_mid": Tap("block", "norm_b", "in", writeback="block_output"),
            },
            identities=(_RESIDUAL_OUT, second),
        )


def test_the_eleven_pairs_split_eight_to_three():
    assert len(BACKEND_PAIRS) == 11
    aliased = [p for p in BACKEND_PAIRS if p.aliased]
    typed = [p for p in BACKEND_PAIRS if not p.aliased]
    assert len(aliased) == 8 and len(typed) == 3
    assert {p.relation for p in typed} == {"gva_tile", "chunk_boundary"}
    assert set(RELATIONS) == {"identical", "gva_tile", "chunk_boundary"}
    for pair in aliased:
        assert DEPRECATED_COMPONENTS[pair.nnsight] == pair.hooks
        assert pair.nnsight not in COMPONENTS and pair.hooks in COMPONENTS
        assert CAPABILITIES[pair.hooks].reads == {"pytorch_hooks", "nnsight"}
        assert CAPABILITIES[pair.hooks].aliases == (pair.nnsight,)
    for pair in typed:
        assert CAPABILITIES[pair.hooks].reads == {"pytorch_hooks"}
        assert CAPABILITIES[pair.nnsight].reads == {"nnsight"}
        assert pair.nnsight not in DEPRECATED_COMPONENTS


def test_the_canonical_spelling_is_the_engine_neutral_one():
    """`delta_*` names the delta rule the tensor belongs to; the retired
    spellings echo the modeling file's variable names (`core_attn_out`,
    `mixed_qkv`) — module paths, which the public vocabulary does not carry."""
    assert all(p.hooks.startswith("delta_") for p in BACKEND_PAIRS)
    assert all(p.nnsight.startswith("deltanet_") for p in BACKEND_PAIRS)
    assert DEPRECATED_COMPONENTS["deltanet_core_out"] == "delta_kernel_output"
    assert DEPRECATED_COMPONENTS["deltanet_gated_out"] == "delta_premix"
    assert DEPRECATED_COMPONENTS["deltanet_qkv_conv"] == "delta_conv"


def test_every_alias_has_a_deprecation_version():
    assert set(DEPRECATED_IN) == set(DEPRECATED_COMPONENTS)
    assert set(DEPRECATED_IN.values()) == {"1"}  # the one protocol version
    for component, row in CAPABILITIES.items():
        if row.aliases:
            assert row.deprecated_in == DEPRECATED_IN[row.aliases[0]], component
        else:
            assert row.deprecated_in is None, component


def test_a_retired_spelling_parses_to_its_replacement_and_digests_identically():
    for alias, canonical in DEPRECATED_COMPONENTS.items():
        if alias == "attention_value":
            continue
        old, new = base_doc(), base_doc()
        old["model"]["key"] = new["model"]["key"] = DOCS_TABLE_MODEL
        old["method"]["sites"]["tgt"] = {"component": alias, "layers": [0]}
        new["method"]["sites"]["tgt"] = {"component": canonical, "layers": [0]}
        assert parse_document(in_order(old)).sites["tgt"].component == canonical
        assert canonicalize(in_order(old), ENV) == canonicalize(in_order(new), ENV)


def test_the_three_typed_pairs_are_not_folded():
    for pair in BACKEND_PAIRS:
        if pair.aliased:
            continue
        raw = base_doc()
        raw["model"]["key"] = DOCS_TABLE_MODEL
        raw["method"]["sites"]["tgt"] = {"component": pair.nnsight, "layers": [0]}
        assert parse_document(in_order(raw)).sites["tgt"].component == pair.nnsight


@pytest.mark.parametrize(
    "pair", [p for p in BACKEND_PAIRS if not p.aliased], ids=lambda p: p.nnsight
)
def test_aliasing_a_typed_pair_is_refused_as_a_rebind(pair: BackendPair):
    """T2's mutation, at the vocabulary: point an alias at a `gva_tile` (or
    `chunk_boundary`) pair and the alias census refuses with the relation —
    a shape / timing refusal, never a silent tile."""
    reason = alias_would_rebind(pair.nnsight, pair.hooks)
    assert reason is not None and pair.relation in reason and "rebind" in reason
    mutated = {**DEPRECATED_COMPONENTS, pair.nnsight: pair.hooks}
    with pytest.raises(AssertionError, match="would rebind, not redirect"):
        _check_aliases(mutated)
    # and the two really do differ in shape (gva) or timing (chunk) on the A3B
    left = component_shape(A3B, pair.hooks)
    right = component_shape(A3B, pair.nnsight)
    assert left.describe() != right.describe() or left.width != right.width


def test_the_alias_table_passes_its_own_census():
    _check_aliases()  # the import-time check, re-run
    for alias, canonical in DEPRECATED_COMPONENTS.items():
        assert alias_would_rebind(alias, canonical) is None, alias


def test_the_shape_rule_refuses_two_vocabulary_names_of_different_shape():
    reason = alias_would_rebind("attention_key", "attention_query")
    assert reason is not None and "shapes differ" in reason
    assert alias_would_rebind("delta_kv_mem", "delta_state_update") is None


def test_backend_pair_is_looked_up_by_either_spelling():
    assert backend_pair("delta_query").relation == "gva_tile"
    assert backend_pair("deltanet_query") is backend_pair("delta_query")
    assert backend_pair("delta_state").chunk == 64
    assert backend_pair("deltanet_core_out").hooks == "delta_kernel_output"
    with pytest.raises(AssertionError, match="not a spelling of any backend pair"):
        backend_pair("block_output")
    with pytest.raises(ValueError, match="chunk_boundary pairs declare the chunk"):
        BackendPair("delta_state", "deltanet_state", "chunk_boundary", "w")
    with pytest.raises(ValueError, match="not in"):
        BackendPair("a", "b", "same", "w")  # type: ignore[arg-type]


def test_the_typed_pairs_are_read_by_the_sweep_helper_not_owned():
    sweep = pytest.importorskip("tests._helpers.a3b_sweep")  # imports torch
    assert sweep.DELTA_FAMILY_PAIRS == tuple(
        (p.hooks, p.nnsight, p.relation) for p in BACKEND_PAIRS if not p.aliased
    )
    assert not hasattr(sweep, "DELTA_CHUNK")
    assert set(sweep.SHARED_LINEAR_ONLY) == {
        p.hooks for p in BACKEND_PAIRS if p.aliased
    }


# --------------------------------------------------------------------------- #
# grouped_mm — a declared requirement resolved during validation
# --------------------------------------------------------------------------- #


def _moe_entry(knob: str | None) -> ModelInfo:
    return dataclasses.replace(
        A3B, key=f"contract/moe-{knob}", experts_implementation=knob
    )


def test_grouped_mm_is_decided_by_the_entry_that_carries_the_knob():
    assert "grouped_mm" in PREDICATES
    assert predicate_holds(_moe_entry(None), "grouped_mm") is None
    assert predicate_holds(_moe_entry("grouped_mm"), "grouped_mm") is True
    assert predicate_holds(_moe_entry("eager"), "grouped_mm") is False
    assert A3B.experts_implementation is None  # the built-in entry: the run decides


def test_the_routed_interior_is_refused_at_load_on_another_dispatch():
    eager = _moe_entry("eager")
    reason = unavailable_at_load(eager, "expert_activation")
    assert reason is not None
    assert "experts_implementation='eager'" in reason and "grouped_mm" in reason
    # the twin: the default dispatch and an undeclared knob both load
    assert unavailable_at_load(_moe_entry("grouped_mm"), "expert_activation") is None
    assert unavailable_at_load(_moe_entry(None), "expert_activation") is None
    # a module-boundary MoE component is not dispatch-pinned
    assert unavailable_at_load(eager, "routed_output") is None


def test_a_document_on_the_routed_interior_is_refused_at_load_by_the_knob():
    eager, grouped = _moe_entry("eager"), _moe_entry("grouped_mm")
    register_model(eager)
    register_model(grouped)
    raw = base_doc()
    raw["method"]["sites"]["tgt"] = {"component": "expert_activation", "layers": [0]}
    raw["model"]["key"] = eager.key
    with pytest.raises(ValidationError) as excinfo:
        _validate_sites(raw)
    assert excinfo.value.rule == 4 and excinfo.value.reason == "component_unavailable"
    assert "experts_implementation='eager'" in str(excinfo.value)
    raw["model"]["key"] = grouped.key
    _validate_sites(raw)  # valid work still passes
    canonicalize(in_order(raw), ENV)


def test_the_adapter_reads_the_knob_off_a_loaded_config_only():
    from causalab.protocol.registry import model_info_from_hf_config

    config = types.SimpleNamespace(
        num_attention_heads=2,
        hidden_size=8,
        num_hidden_layers=1,
        vocab_size=10,
        intermediate_size=16,
        model_type="probe",
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=8,
    )
    assert model_info_from_hf_config("probe", config).experts_implementation is None
    config._experts_implementation = "grouped_mm"
    assert (
        model_info_from_hf_config("probe", config).experts_implementation
        == "grouped_mm"
    )


# --------------------------------------------------------------------------- #
# the inventory, offline, on the A3B entry — 40 / 40 / 10 / 30
# --------------------------------------------------------------------------- #


def test_the_a3b_inventory_is_forty_forty_ten_thirty():
    inv = inventory(A3B)
    assert inv.model == DOCS_TABLE_MODEL and len(inv.layers) == 40
    assert inv.count("full_attention") == 10 and inv.count("linear_attention") == 30
    full = tuple(range(3, 40, 4))
    assert inv.where("attention_premix") == full
    assert inv.where("attention_probs") == full
    assert inv.where("delta_premix") == tuple(i for i in range(40) if i not in full)
    # 40 attention + 40 MLP: a mixer output and an MLP output at every layer
    assert len(inv.where("attention_output")) == 40
    assert len(inv.where("mlp_output")) == 40 and len(inv.where("routed_output")) == 40
    assert inv.where("mlp_activation") == ()  # no dense MLP on this tower
    assert inv.layerless == ("input_ids", "embeddings", "ln_final", "lm_head")


def test_the_inventory_never_lists_a_component_off_its_stream():
    """Invalid stream × site combinations are absent by construction — the
    inventory is a query over the rows' `stream` cell, the same cell the
    checklist and the resolver refuse from."""
    inv = inventory(A3B)
    for li in inv.layers:
        for component in li.components:
            assert COMPONENT_STREAMS.get(component, li.stream) == li.stream
        for component, stream in COMPONENT_STREAMS.items():
            if stream != li.stream:
                assert component not in li.components
                raw = base_doc()
                raw["model"]["key"] = DOCS_TABLE_MODEL
                raw["method"]["sites"]["tgt"] = {
                    "component": component,
                    "layers": [li.layer],
                }
                with pytest.raises(ValidationError, match="exists only on a"):
                    _validate_sites(raw)
                break  # one refusal per layer is the point; the rest is the census


def test_the_inventory_carries_the_rows_mechanisms():
    inv = inventory(A3B)
    layer3 = inv.layers[3]
    assert layer3.reads["delta_premix"] if "delta_premix" in layer3.reads else True
    assert layer3.reads["attention_probs"] == {"pytorch_hooks", "nnsight"}
    assert layer3.writes["attention_probs"] == {"swap"}
    assert layer3.writes["router_logits"] is None
    layer0 = inv.layers[0]
    assert layer0.reads["delta_state"] == {"pytorch_hooks"}
    assert layer0.reads["deltanet_state"] == {"nnsight"}
    assert layer0.reads["delta_beta"] == {"pytorch_hooks", "nnsight"}


def test_an_entry_without_a_layer_pattern_has_no_offline_inventory():
    with pytest.raises(ValidationError, match="declares no layer pattern"):
        inventory(get_model_info("gpt2"))
    loaded = types.SimpleNamespace(
        info=get_model_info("gpt2"), streams=("full_attention",) * 12, adapter=None
    )
    inv = inventory(loaded)
    assert len(inv.layers) == 12 and inv.where("router_scores") == ()
    assert inv.where("mlp_activation") == tuple(range(12))


def test_the_inventory_respects_a_family_that_serves_less():
    loaded = types.SimpleNamespace(
        info=get_model_info("gpt2"), streams=("full_attention",) * 12, adapter=GPT2_TREE
    )
    inv = inventory(loaded)
    assert set(inv.layers[0].components) <= set(GPT2_TREE.taps)
    assert "attention_premix" in inv.layers[0].components
    assert set(inv.layerless) == set(LAYERLESS_COMPONENTS)
