"""MoE components: the router, the routed output, and the shared expert.

Nine components of the hookpoint vocabulary, every one a plain module output
or input — the router is a module returning a 3-tuple and the experts are a
fused module, so nothing here needs the ragged value shape that the
per-expert interior does (that is ``expert_output``).

Two kinds of assertion, as in ``test_sites_round1_block.py``. Shapes are 📐
measurements against a real ``qwen3_5_moe`` checkpoint, so a mismatch is a
finding. Identities are stronger: they pin that the three router taps are
*mutually consistent* — that
``router_scores`` really is the renormalized top-k of ``softmax(router_logits)``
gathered at ``expert_idx`` — which no shape assertion could show, and which is
exactly what a wrong ``tuple_index`` would break.
"""

from __future__ import annotations

import copy

import pytest
import torch

from causalab.neural.shared.sites import (
    resolve_site,
)
from causalab.protocol.rules.errors import ProtocolError, ValidationError
from causalab.protocol.positions.alignment import COMPONENT_RANK
from causalab.protocol.registry import CAPABILITIES, component_width
from causalab.protocol.schema import COMPONENTS, PROTOCOL_VERSION, SiteSpec

from tests.protocol._docs import saved

from ._drive import base_data_section, executor_for

pytestmark = pytest.mark.smoke

#: The nine addressable MoE components.
MOE_COMPONENTS = (
    "router_logits",
    "router_scores",
    "expert_idx",
    "routed_output",
    "shared_expert_gate_proj",
    "shared_expert_up_proj",
    "shared_expert_activation",
    "shared_expert_output",
    "shared_expert_gate",
)

#: One valid ``do`` payload per non-``swap`` mechanism the refusal must cover:
#: an absolute-class write, an additive one, and a clamp — enough that the rule
#: is clearly about the *component*, not about one spelling.
_MECHANISM_PAYLOADS: dict[str, dict] = {
    "add_scaled": {"add_scaled": {"op": "v_cf", "alpha": 5.0}},
    "lerp": {"lerp": {"op": "v_cf", "alpha": 0.5}},
    "clamp": {"clamp": {"lo": 0.0, "hi": 1.0}},
}

TEXT = "the quick brown fox jumps"
#: 📐 measured on tiny-random/qwen3.5-moe: hidden 8, 128 experts, top-10,
#: shared-expert inner width 32. One read position, so (1, 1, width).
EXPECTED_WIDTH = {
    "router_logits": 128,
    "router_scores": 10,
    "expert_idx": 10,
    "routed_output": 8,
    "shared_expert_gate_proj": 32,
    "shared_expert_up_proj": 32,
    "shared_expert_activation": 32,
    "shared_expert_output": 8,
    "shared_expert_gate": 1,
}


def _doc(component: str, *, layer: int = 0, featurizer: bool = False) -> dict:
    read: dict = {"site": "tap", "pos": {"index": 1}}
    doc: dict = {
        "header": {"protocol_version": PROTOCOL_VERSION},
        "model": {"key": "test", "revision": "main"},
        "data": base_data_section(with_counterfactual=False),
        "method": {
            "intervened_models": {"original": {"input": "base", "reads": ["r"]}},
            "sites": {"tap": {"component": component, "layers": [layer]}},
            "reads": {"r": read},
            "save": [saved("r", "original", "a.safetensors")],
        },
    }
    if featurizer:
        doc["method"]["featurizers"] = {"f": {"kind": "subspace", "k": 1}}
        read["featurizer"] = "f"
    return doc


def _read(bundle, component: str, *, layer: int = 0) -> torch.Tensor:
    executor = executor_for(_doc(component, layer=layer), bundle, base_texts=[TEXT])
    return executor.read_value("r")


@pytest.fixture(scope="module")
def moe_reads(qwen35moe_bundle) -> dict[str, torch.Tensor]:
    """Every MoE component, read through a real document at one position."""
    return {c: _read(qwen35moe_bundle, c) for c in MOE_COMPONENTS}


# --------------------------------------------------------------------------- #
# vocabulary and metadata
# --------------------------------------------------------------------------- #


def test_all_nine_joined_the_closed_vocabulary():
    for component in MOE_COMPONENTS:
        assert component in COMPONENTS, component


def test_the_moe_interior_ranks_inside_the_block():
    """These were 71 and 72 — *after* ``mlp_output`` (70) — which contradicted
    the comment they carried. Unreachable only because ``router_logits`` refused
    to resolve; it resolves now, so the order has to be right."""
    order = [
        "mlp_input",
        "router_logits",
        "router_scores",
        "expert_idx",
        "expert_output",
        "routed_output",
        "shared_expert_gate_proj",
        "shared_expert_up_proj",
        "shared_expert_activation",
        "shared_expert_output",
        "shared_expert_gate",
        "mlp_output",
        "block_output",
    ]
    ranks = [COMPONENT_RANK[c] for c in order]
    assert ranks == sorted(ranks), dict(zip(order, ranks))


def test_the_registry_reads_the_moe_numbers_off_the_text_config(qwen35moe_bundle):
    """📐 Both live on ``config.text_config``, not the top-level (heterogeneous)
    config — the same hazard as ``num_experts``."""
    info = qwen35moe_bundle.info
    assert info.num_experts == 128
    assert info.num_experts_per_tok == 10
    assert info.shared_expert_intermediate_size == 32


# --------------------------------------------------------------------------- #
# widths
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "component",
    [c for c in MOE_COMPONENTS if c != "expert_idx"],
)
def test_each_width_matches_the_tensor_it_describes(
    qwen35moe_bundle, moe_reads, component: str
):
    """The width table and the measured tensor must agree — a width that is
    merely *plausible* is the failure mode here (e.g. reading the dense
    ``intermediate_size`` for the shared expert)."""
    assert (
        component_width(qwen35moe_bundle.info, component) == EXPECTED_WIDTH[component]
    )
    assert moe_reads[component].shape[-1] == EXPECTED_WIDTH[component]


def test_expert_idx_refuses_a_width_because_it_is_a_routing_table(qwen35moe_bundle):
    """The same rule ``input_ids`` needs: integer ids, no feature space."""
    with pytest.raises(ValidationError) as excinfo:
        component_width(qwen35moe_bundle.info, "expert_idx")
    assert excinfo.value.rule == 4
    assert "routing table" in str(excinfo.value)


def test_the_fixture_cannot_distinguish_the_three_inner_widths(qwen35moe_bundle):
    """⚠️ A caveat on the test above, recorded so it is not mistaken for proof.

    On this fixture ``intermediate_size``, ``moe_intermediate_size`` and
    ``shared_expert_intermediate_size`` are all 32, so reading the wrong one
    would pass. The registry reads the most specific spelling first and never
    falls back to the dense ``intermediate_size``; that ordering is the real
    guarantee, and this test says out loud that the fixture does not check it.
    """
    info = qwen35moe_bundle.info
    assert info.shared_expert_intermediate_size == info.intermediate_size == 32


# --------------------------------------------------------------------------- #
# the taps: shape and tuple index
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("component", MOE_COMPONENTS)
def test_every_moe_tap_is_flat(qwen35moe_bundle, component: str):
    """📐 ``Qwen3_5MoeSparseMoeBlock`` reshapes to ``(-1, hidden)`` before the
    router, so the whole interior is flattened over (batch, position)."""
    site = resolve_site(qwen35moe_bundle, SiteSpec(component=component, layers=(0,)))
    assert site.shape.flat_batch


def test_the_router_taps_declare_the_three_tuple_elements(qwen35moe_bundle):
    """``Qwen3_5MoeTopKRouter`` returns (logits, scores, indices). The historical
    "element 0 of a tuple" rule would have handed back the logits for all three,
    which is why ``tuple_index`` exists."""
    indices = {}
    for component in ("router_logits", "router_scores", "expert_idx"):
        site = resolve_site(
            qwen35moe_bundle, SiteSpec(component=component, layers=(0,))
        )
        assert site.module is qwen35moe_bundle.blocks[0].mlp.gate
        indices[component] = site.tuple_index
    assert indices == {"router_logits": 0, "router_scores": 1, "expert_idx": 2}


def test_the_three_router_taps_are_three_different_tensors(moe_reads):
    """Anti-vacuity for the tuple indices: a wrong index would alias."""
    logits, scores = moe_reads["router_logits"], moe_reads["router_scores"]
    assert logits.shape != scores.shape
    assert moe_reads["expert_idx"].dtype != scores.dtype


def test_an_expert_sub_axis_refuses(qwen35moe_bundle):
    """None of these tensors is indexed by expert: the router's axes are
    all-experts or top-k, and the shared expert is not a routed one. ``expert``
    parses and nothing read it — refusing beats silently ignoring it, which is
    the mistake ``stream`` made before it was read per layer."""
    with pytest.raises(ProtocolError) as excinfo:
        resolve_site(
            qwen35moe_bundle,
            SiteSpec(component="router_scores", layers=(0,), expert=3),
        )
    assert "no per-expert axis" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# the identities — the taps are mutually consistent, not merely shaped
# --------------------------------------------------------------------------- #


def test_router_probs_recomputed_from_logits_sums_to_one(moe_reads):
    """End to end. ``router_probs`` is derived, not a component: this
    is how a user gets it, and it must be a real distribution."""
    probs = torch.softmax(moe_reads["router_logits"].float(), dim=-1)
    torch.testing.assert_close(probs.sum(-1), torch.ones_like(probs.sum(-1)))


def test_router_scores_sum_to_one_because_the_top_k_is_renormalized(moe_reads):
    scores = moe_reads["router_scores"].float()
    torch.testing.assert_close(
        scores.sum(-1), torch.ones_like(scores.sum(-1)), atol=1e-5, rtol=1e-5
    )


def test_expert_idx_are_valid_integer_expert_ids(qwen35moe_bundle, moe_reads):
    idx = moe_reads["expert_idx"]
    assert not idx.dtype.is_floating_point
    assert int(idx.min()) >= 0
    assert int(idx.max()) < qwen35moe_bundle.info.num_experts
    # top-k picks distinct experts
    flat = idx.reshape(-1, idx.shape[-1])
    for row in flat:
        assert len(set(row.tolist())) == row.numel()


def test_the_three_router_taps_agree_with_each_other(moe_reads):
    """The strongest statement available here, and the one that pins the tuple
    indices semantically rather than positionally:

        router_scores == renormalize(softmax(router_logits) gathered at expert_idx)

    If ``router_scores`` and ``expert_idx`` were swapped, or either took the
    wrong tuple element, this fails. 📐 Measured exact (max abs diff 0.0).
    """
    probs = torch.softmax(moe_reads["router_logits"].float(), dim=-1).squeeze(1)
    idx = moe_reads["expert_idx"].squeeze(1)
    gathered = torch.gather(probs, -1, idx)
    renormalized = gathered / gathered.sum(-1, keepdim=True)
    torch.testing.assert_close(
        moe_reads["router_scores"].float().squeeze(1),
        renormalized,
        atol=1e-6,
        rtol=1e-6,
    )


def test_the_shared_expert_interior_composes(qwen35moe_bundle, moe_reads):
    """SwiGLU, from the taps alone::

    shared_expert_activation == silu(gate_proj(x)) * up_proj(x)
    shared_expert_output     == down_proj(shared_expert_activation)
    """
    shared = qwen35moe_bundle.blocks[0].mlp.shared_expert
    gate = moe_reads["shared_expert_gate_proj"]
    up = moe_reads["shared_expert_up_proj"]
    torch.testing.assert_close(
        moe_reads["shared_expert_activation"], shared.act_fn(gate) * up
    )
    torch.testing.assert_close(
        moe_reads["shared_expert_output"],
        shared.down_proj(moe_reads["shared_expert_activation"]),
    )


def test_the_mlp_output_is_the_two_branches_combined(qwen35moe_bundle, moe_reads):
    """``mlp_out`` = routed + sigmoid(shared_expert_gate) * shared_expert_output.

    ``shared_gated`` is the derived box — this is the identity that
    makes it derivable, and it pins ``routed_output`` and both shared-expert
    taps against a component that already existed.
    """
    mlp_out = _read(qwen35moe_bundle, "mlp_output")
    combined = (
        moe_reads["routed_output"]
        + torch.sigmoid(moe_reads["shared_expert_gate"])
        * moe_reads["shared_expert_output"]
    )
    torch.testing.assert_close(mlp_out, combined, atol=1e-5, rtol=1e-5)


# --------------------------------------------------------------------------- #
# featurizers
# --------------------------------------------------------------------------- #


def test_a_featurizer_on_expert_idx_refuses_by_rule_number(qwen35moe_bundle):
    """The end-to-end check's second half."""
    with pytest.raises(ValidationError) as excinfo:
        executor_for(
            _doc("expert_idx", featurizer=True), qwen35moe_bundle, base_texts=[TEXT]
        ).read_value("r")
    assert excinfo.value.rule == 4


def test_router_scores_columns_are_a_ranking_not_a_basis(qwen35moe_bundle):
    """⚠️ An open design question, recorded as evidence.

    ``router_scores`` HAS a width (``num_experts_per_tok``), so a featurizer —
    a ``subspace`` fit across positions — is accepted today. This measures why
    that acceptance is worth a second look before anyone reads such a fit:
    column *j* is "the j-th ranked expert for this token", and which expert
    that is changes token by token.

    📐 Measured on the fixture, one 5-token prompt: column 0 names experts
    [104, 22, 95, 38, 8] at positions 0-4 — five different experts in one
    coordinate. A subspace fitted across positions therefore mixes unrelated
    experts' scores into the same direction, and the resulting basis means
    nothing, while looking exactly like every other subspace fit in a report.

    This test does not decide the question — it pins the fact the decision
    rests on, so whichever way it goes, it goes with evidence. If the answer is
    "refuse", the change is a ``component_width`` special case and this test
    flips to asserting the refusal.
    """
    doc = copy.deepcopy(_doc("expert_idx"))
    doc["method"]["reads"]["r"]["pos"] = {"all": True}
    ids = executor_for(doc, qwen35moe_bundle, base_texts=[TEXT]).read_value("r")
    ids = ids.reshape(-1, ids.shape[-1]).long()
    assert ids.shape[0] > 1, "need several positions for the property to be visible"

    distinct_per_column = [len(set(ids[:, j].tolist())) for j in range(ids.shape[-1])]
    assert max(distinct_per_column) > 1, (
        "column j named the same expert at every position — on this input the "
        "ranking happens to be stable, and the hazard below is not visible here"
    )

    # and the acceptance this documents: the fit is allowed, today
    executor_for(
        _doc("router_scores", featurizer=True), qwen35moe_bundle, base_texts=[TEXT]
    ).run_all()


def test_a_featurizer_on_a_flat_moe_tap_is_fine(qwen35moe_bundle):
    """Anti-vacuity, and the property that matters for `flat_td`: the featurizer
    sees the contract shape, so a k=1 subspace over a 128-wide router works."""
    executor = executor_for(
        _doc("router_logits", featurizer=True), qwen35moe_bundle, base_texts=[TEXT]
    )
    assert executor.read_value("r").shape == (1, 1, 1)


def test_a_non_moe_family_refuses_the_moe_components(llama_bundle):
    """tiny-llama has a dense MLP, so these have nothing to tap and must say so
    rather than AttributeError from inside a hook."""
    with pytest.raises(ProtocolError) as excinfo:
        resolve_site(llama_bundle, SiteSpec(component="router_logits", layers=(0,)))
    assert "sparse-MoE block" in str(excinfo.value)
    assert excinfo.value.reason == "component_unavailable"


# --------------------------------------------------------------------------- #
# writes: a write that cannot reach anything must refuse, not no-op
# --------------------------------------------------------------------------- #


def _swap_doc(component: str, *, layer: int | None = 0, pos: int = 1) -> dict:
    site: dict = {"component": component}
    if layer is not None and component not in ("input_ids",):
        site["layers"] = layer
    return {
        "header": {"protocol_version": PROTOCOL_VERSION},
        "model": {"key": "test", "revision": "main"},
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
            "sites": {"tgt": site, "lm_head": {"component": "lm_head"}},
            "reads": {
                "v_cf": {"site": "tgt", "pos": {"index": pos}},
                "clean": {"site": "lm_head", "pos": {"index": -1}},
                "after": {"site": "lm_head", "pos": {"index": -1}},
            },
            "writes": {
                "patch": {"site": "tgt", "pos": {"index": pos}, "do": {"swap": "v_cf"}}
            },
            "save": [
                saved("after", "patched", "p.safetensors"),
                saved("clean", "original_base", "c.safetensors"),
            ],
        },
    }


#: Every MoE component a write is *allowed* to reach — derived from the two
#: tables rather than listed, so a component added to one and forgotten in the
#: other cannot quietly skip the causal check below. ``shared_expert_gate_proj``
#: and ``shared_expert_up_proj`` were exactly that gap when this was a literal
#: list: writable, and the only two siblings with no write-moves-logits pin.
WRITABLE_MOE_COMPONENTS = tuple(
    c for c in MOE_COMPONENTS if CAPABILITIES[c].writes is not None
)


def test_every_writable_moe_component_is_causally_checked():
    """The guard on the guard: the parametrization below must cover the whole
    writable surface, so this pins the arithmetic rather than trusting it."""
    assert len(WRITABLE_MOE_COMPONENTS) == len(MOE_COMPONENTS) - len(
        [c for c in MOE_COMPONENTS if CAPABILITIES[c].writes is None]
    )
    assert set(WRITABLE_MOE_COMPONENTS) | {"router_logits"} == set(MOE_COMPONENTS)


@pytest.mark.parametrize("component", WRITABLE_MOE_COMPONENTS)
def test_a_write_through_a_flat_tap_actually_changes_the_logits(
    qwen35moe_bundle, component: str
):
    """The view property, checked causally rather than by aliasing:
    ``flat_td`` conversion must return a view, or the write lands
    in a discarded copy and the run silently reports the clean numbers.

    Swapping the counterfactual value into base at (L0, p1) must move the
    logits. 📐 It does, for every writable MoE tap — including the two
    projection taps, which feed ``silu(gate) * up -> down_proj`` and so cannot
    move the shared expert's output without moving the logits.
    """
    executor = executor_for(
        _swap_doc(component),
        qwen35moe_bundle,
        base_texts=[TEXT],
        counterfactual_texts=["a slow green turtle sleeps deeply"],
    )
    clean = executor.read_value("clean")
    patched = executor.read_value("after")
    assert float((patched - clean).abs().max()) > 0.0, (
        f"a write to {component} left the logits unchanged — the tap is not "
        "reaching native storage"
    )


@pytest.mark.parametrize("component", ["router_logits", "input_ids"])
def test_a_write_that_cannot_reach_anything_refuses(qwen35moe_bundle, component: str):
    """📐 The finding this rule exists for.

    ``Qwen3_5MoeSparseMoeBlock.forward`` destructures the router as
    ``_, routing_weights, selected_experts = self.gate(...)`` — the logits go to
    ``_`` and are never read again. So a write to ``router_logits`` moves the
    logits by exactly 0.0 while every other MoE write moves them (the test
    above). A silent no-op is the worst available outcome: the run succeeds and
    the conclusion is wrong.

    ``input_ids`` is refused for the *opposite* reason, and the registry row's
    ``why`` spells the difference out: 📐 a write there does land
    (the tap is the embedding's pre-hook input, and mutating it changes the ids
    the model looks up). It is refused because token ids are not an activation
    — editing them is a change to the dataset, and belongs in the row's text.
    """
    with pytest.raises(ProtocolError) as excinfo:
        executor_for(
            _swap_doc(component),
            qwen35moe_bundle,
            base_texts=[TEXT],
            counterfactual_texts=["a slow green turtle sleeps deeply"],
        ).read_value("after")
    message = str(excinfo.value)
    assert "no write" in message and "may change" in message
    # and it must say what to do instead
    assert CAPABILITIES[component].why.split(";")[0][:20] in message


# --------------------------------------------------------------------------- #
# expert_idx: a routing table is labels, not features
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("mechanism", ["add_scaled", "lerp", "clamp"])
def test_arithmetic_on_the_routing_table_refuses(qwen35moe_bundle, mechanism: str):
    """📐 The finding: before this rule, ``add_scaled`` over ``expert_idx`` ran
    to completion with no refusal anywhere.

    The routing table is int64 expert *ids*. Scaling or offsetting them is
    arithmetic on labels: the result names whichever experts it happens to land
    on where it stays in range, and fails at the gather where it does not —
    on CUDA a device-side assert, raised far from the write that caused it.
    Neither outcome is a result, so the write is refused by name, at the plan,
    before the model runs.
    """
    doc = copy.deepcopy(_swap_doc("expert_idx"))
    doc["method"]["writes"]["patch"]["do"] = _MECHANISM_PAYLOADS[mechanism]
    # `clamp` names no operand, which would leave `v_cf` dead and trip V11
    # before the rule under test fires — save it so the refusal is what we see
    doc["method"]["save"].append(
        saved("v_cf", "original_counterfactual", "vcf.safetensors")
    )
    with pytest.raises(ProtocolError) as excinfo:
        executor_for(
            doc,
            qwen35moe_bundle,
            base_texts=[TEXT],
            counterfactual_texts=["a slow green turtle sleeps deeply"],
        ).read_value("after")
    message = str(excinfo.value)
    assert mechanism in message and "expert_idx" in message
    assert "swap" in message


def test_a_swap_of_the_routing_table_is_still_allowed(qwen35moe_bundle):
    """The rule narrows the mechanism, not the component: replacing the whole
    table with one read from elsewhere is the supported way to reroute, and it
    still moves the logits (the parametrized causal check above covers it —
    asserted here too so the refusal cannot be widened by accident)."""
    executor = executor_for(
        _swap_doc("expert_idx"),
        qwen35moe_bundle,
        base_texts=[TEXT],
        counterfactual_texts=["a slow green turtle sleeps deeply"],
    )
    moved = float(
        (executor.read_value("after") - executor.read_value("clean")).abs().max()
    )
    assert moved > 0.0
