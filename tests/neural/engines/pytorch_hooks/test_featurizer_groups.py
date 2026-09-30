"""The grouped gates (spec §2.5 ``group``): by head, and by expert neuron.

A DBM gate normally holds one ``theta`` per coordinate. Declared with
``group: head`` on a head-major component it holds one per **head** and
expands it over that head's coordinates, so the mask it fits is a set of
heads: every coordinate of head ``h`` receives ``σ(θ_h/T)`` in training and
``θ_h > 0`` in eval, the L1 term is a mean over heads, and a bundle carries
the ``(heads, head_dim)`` map it was fitted over.

Declared with ``group: expert_neuron`` on ``expert_activation`` it holds the
whole ``(num_experts, d_expert)`` table, and a token's routed slots look their
rows up through ``expert_idx`` — so the same neuron of the same expert has one
parameter whichever slot it fills, and an expert a token did not activate
cannot touch it. The swap through such a gate joins slots by expert: a base
slot whose expert is active on the counterfactual side takes that expert's
counterfactual activation, and one whose expert is inactive there keeps its
base value, with the count recorded per write, layer and example. The tests
below pin every clause against a hand computation on the tiny MoE fixture.

Two components are head-major and served by one code path: ``attention_premix``
(the o-projection's input, query-head space, full-attention layers) and
``delta_premix`` (the out-projection's input, value-head space, Gated DeltaNet
layers). The tiny MoE fixture has both — layers 0-2 DeltaNet, layer 3 full
attention — which is what the group-map and per-head o-projection tests below
run on; the ``site_group_map`` derivation itself is protocol code, so its
refusals (rule 23, group legality) are pinned offline in
``tests/protocol/test_grouped_gate.py``.

The load-bearing test is the hand-computed one: zeroing one head's hard mask
and swapping must remove exactly that head's contribution,
``W_o[:, h·d:(h+1)·d] @ premix[h·d:(h+1)·d]``, and nothing else.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from causalab.neural.engines.pytorch_hooks.loading import ModelBundle, load_model
from causalab.neural.shared.featurizers import Gate, build_stack
from causalab.neural.shared.featurizers.build import (
    _stage_width,  # pyright: ignore[reportPrivateUsage]
)
from causalab.io.tensor_files import TensorBundle
from causalab.neural.shared.sites import resolve_site
from causalab.neural.shared.model_tree import stream_at
from causalab.protocol.rules.errors import ProtocolError, ValidationError
from causalab.protocol.registry import (
    component_shape,
    gate_group_map,
    gate_param_shape,
    get_model_info,
    head_group_map,
    site_group_map,
)
from causalab.protocol.schema import FeaturizerSpec, SiteSpec

from tests.neural.engines.pytorch_hooks._drive import base_data_section, executor_for
from tests.neural.engines.pytorch_hooks.conftest import TINY_QWEN35_MOE
from tests.protocol._docs import UNWRITTEN, saved

TEXT = "the quick brown fox jumps"
CF_TEXT = "a slow green turtle sleeps"

#: The un-intervened network read on base, beside `UNWRITTEN` on the
#: counterfactual (§2.9's names for a network read un-intervened on both).
ORIGINAL_BASE = "original_base"
INPUT_OF = {ORIGINAL_BASE: "base", UNWRITTEN: "counterfactual", "masked": "base"}


def _bound_doc(
    reads: dict[str, tuple[str, dict[str, Any]]], **method: Any
) -> dict[str, Any]:
    """A document over ``reads`` — ``name: (model, address)`` — with every
    model listing the reads taken on it (``masked`` landing ``writes``) and
    every read saved as a tensor under its own name."""
    models: dict[str, dict[str, Any]] = {}
    for name, (model, _) in reads.items():
        models.setdefault(model, {"input": INPUT_OF[model], "reads": []})
        models[model]["reads"].append(name)
    models["masked"]["writes"] = list(method["writes"])
    return {
        "header": {"protocol_version": "4"},
        "model": {"key": "test", "revision": "main"},
        "data": base_data_section(with_counterfactual=True),
        "method": {
            "intervened_models": models,
            **method,
            "reads": {name: address for name, (_, address) in reads.items()},
            "save": [
                saved(name, model, f"{name}.safetensors")
                for name, (model, _) in reads.items()
            ],
        },
    }


#: (component, layer) for the fixture's two head-major families.
FULL_ATTENTION = ("attention_premix", 3)
DELTANET = ("delta_premix", 0)
FAMILIES = [
    pytest.param(*FULL_ATTENTION, id="full_attention"),
    pytest.param(*DELTANET, id="deltanet"),
]


@pytest.fixture(scope="module")
def moe() -> ModelBundle:
    return load_model(TINY_QWEN35_MOE)


# --------------------------------------------------------------------------- #
# the gate alone
# --------------------------------------------------------------------------- #


@pytest.mark.unit
class TestGroupedGateMath:
    def test_one_parameter_per_head(self) -> None:
        gate = Gate(4 * 8, group="head", groups=(4, 8))
        assert gate.theta.shape == (4,)
        assert gate.width == 32
        assert _stage_width(gate) == 32  # the cache check sizes by the site width

    def test_the_mask_is_constant_within_every_head_soft_and_hard(self) -> None:
        gate = Gate(3 * 5, group="head", groups=(3, 5))
        with torch.no_grad():
            gate.theta.copy_(torch.tensor([2.0, -1.0, 0.0]))
        x = torch.ones(15)
        gate.train()
        soft, soft_err = gate.featurize(x)
        gate.eval()
        hard, hard_err = gate.featurize(x)
        for head, value in enumerate(gate.theta.tolist()):
            block = slice(head * 5, (head + 1) * 5)
            assert torch.equal(
                soft[block], torch.full((5,), float(torch.sigmoid(torch.tensor(value))))
            )
            assert torch.equal(hard[block], torch.full((5,), float(value > 0)))
        # θ == 0 is not selected, exactly as the per-coordinate gate has it
        assert torch.equal(hard, torch.tensor([1.0] * 5 + [0.0] * 10))
        assert torch.equal(soft + soft_err, x) and torch.equal(hard + hard_err, x)

    def test_the_ungrouped_gate_is_unchanged(self) -> None:
        gate = Gate(6)
        assert gate.groups is None and gate.group is None
        assert gate.theta.shape == (6,) and _stage_width(gate) == 6
        with torch.no_grad():
            gate.theta.copy_(torch.tensor([1.0, -1.0, 0.5, -0.5, 0.0, 2.0]))
        gate.eval()
        assert torch.equal(
            gate.featurize(torch.ones(6))[0], torch.tensor([1.0, 0, 1, 0, 0, 1])
        )

    def test_a_group_map_must_tile_the_width(self) -> None:
        with pytest.raises(ValueError, match="does not tile"):
            Gate(10, group="head", groups=(3, 4))
        with pytest.raises(ValueError, match="come together"):
            Gate(12, groups=(3, 4))

    def test_from_theta_keeps_the_map_and_needs_one_entry_per_group(self) -> None:
        gate = Gate.from_theta(torch.tensor([1.0, -1.0]), group="head", groups=(2, 3))
        assert gate.width == 6 and gate.groups == (2, 3) and gate.group == "head"
        assert not gate.theta.requires_grad
        with pytest.raises(ValueError, match="needs 2 parameters"):
            Gate.from_theta(torch.ones(6), group="head", groups=(2, 3))


# --------------------------------------------------------------------------- #
# the map, derived from the fixture's two head-major components
# --------------------------------------------------------------------------- #


def _site(bundle: ModelBundle, component: str, layer: int, head: int | None = None):
    return resolve_site(
        bundle, SiteSpec(component=component, layers=(layer,), head=head)
    )


def _map(bundle: ModelBundle, component: str, layer: int, head: int | None = None):
    """The registry's offline reading of the map at a declared site — the one
    the canonicalizer and the loader use — for the tests below to hold against
    the resolved site's slices and the loaded modules' widths."""
    del layer  # the map is per component family; the layer picks the fixture site
    return site_group_map(bundle.info, "head", component, head=head)


@pytest.mark.unit
class TestGroupMaps:
    def test_the_fixture_has_both_families_where_this_file_says(self, moe) -> None:
        """Pinned so the two parametrizations below test what they claim: the
        registry's per-layer streams and the loaded modules agree."""
        blocks = moe.model.model.layers
        assert stream_at(blocks, FULL_ATTENTION[1], key=moe.key) == "full_attention"
        assert stream_at(blocks, DELTANET[1], key=moe.key) == "linear_attention"
        assert moe.info.layer_types == (
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        )

    def test_full_attention_groups_are_query_heads(self, moe) -> None:
        info = moe.info
        assert _map(moe, *FULL_ATTENTION) == (info.num_heads, info.head_dim)
        # and the o-projection really is that wide, so the map tiles the tensor
        assert moe.mixer_at(FULL_ATTENTION[1]).o_proj.in_features == (
            info.num_heads * info.head_dim
        )

    def test_deltanet_groups_are_value_heads(self, moe) -> None:
        info = moe.info
        assert _map(moe, *DELTANET) == (
            info.linear_num_value_heads,
            info.linear_value_head_dim,
        )
        assert moe.mixer_at(DELTANET[1]).out_proj.in_features == (
            info.linear_num_value_heads * info.linear_value_head_dim
        )

    def test_the_groups_are_the_head_slices(self, moe) -> None:
        """A group *is* the head a ``head`` field would select: same slices,
        in the same order, on both families."""
        for component, layer in (FULL_ATTENTION, DELTANET):
            groups, group_width = _map(moe, component, layer)
            for head in range(groups):
                site = _site(moe, component, layer, head)
                assert site.feature_slice == slice(
                    head * group_width, (head + 1) * group_width
                )

    def test_a_site_naming_one_head_is_refused(self, moe) -> None:
        """H groups over one head is one group — a coordinate-wise gate under
        a name that claims otherwise (§5.23), so the registry refuses rather
        than resolving a single-group map."""
        component, layer = FULL_ATTENTION
        with pytest.raises(ValidationError, match="already selects head 2") as err:
            _map(moe, component, layer, head=2)
        assert err.value.rule == 23

    def test_a_headless_component_has_no_map(self, moe) -> None:
        with pytest.raises(Exception, match="no head axis"):
            head_group_map(
                component_shape(moe.info, "block_output"), 8, component="block_output"
            )

    # --------------------------------------------------------------------------- #
    # the gate inside a document: build, swap, hand-computed o-projection
    # --------------------------------------------------------------------------- #


def _head_dbm_doc(component: str, layer: int) -> dict[str, Any]:
    """A DBM-shaped document on one head-major site: the counterfactual premix
    read through the gate, swapped into the base run through the same gate,
    plus the plain reads the hand computation needs."""
    return _bound_doc(
        {
            "v_cf": (UNWRITTEN, {"site": "tgt", "pos": -1, "featurizer": "gate"}),
            "pre_base": (ORIGINAL_BASE, {"site": "tgt", "pos": -1}),
            "pre_cf": (UNWRITTEN, {"site": "tgt", "pos": -1}),
            "out_base": (ORIGINAL_BASE, {"site": "out", "pos": -1}),
            "out_masked": ("masked", {"site": "out", "pos": -1}),
        },
        sites={
            "tgt": {"component": component, "layers": [layer]},
            "out": {"component": "attention_output", "layers": [layer]},
        },
        featurizers={"gate": {"kind": "gate", "group": "head"}},
        writes={
            "mask": {
                "site": "tgt",
                "pos": -1,
                "featurizer": "gate",
                "do": {"swap": "v_cf"},
            }
        },
    )


def _out_proj(bundle: ModelBundle, layer: int) -> torch.nn.Linear:
    mixer = bundle.mixer_at(layer)
    return mixer.o_proj if hasattr(mixer, "o_proj") else mixer.out_proj


def _projected(weight: torch.Tensor, width: int, sources: list[torch.Tensor]) -> Any:
    """``Σ_h W[:, slice_h] @ source_h[slice_h]`` — the per-head o-projection,
    head ``h`` taken from the tensor ``sources[h]`` names."""
    total = torch.zeros(weight.shape[0], dtype=weight.dtype)
    for head, source in enumerate(sources):
        block = slice(head * width, (head + 1) * width)
        total = total + weight[:, block] @ source[block]
    return total


@pytest.mark.property
class TestHeadSwap:
    @pytest.mark.parametrize("component, layer", FAMILIES)
    def test_the_document_builds_one_parameter_per_head(
        self, moe, component: str, layer: int
    ) -> None:
        executor = executor_for(
            _head_dbm_doc(component, layer),
            moe,
            base_texts=[TEXT],
            counterfactual_texts=[CF_TEXT],
        )
        gate = executor.stage("gate")
        assert isinstance(gate, Gate)
        assert gate.groups == _map(moe, component, layer)
        assert gate.theta.numel() == gate.groups[0]
        assert not gate.training  # eval semantics: the hard split

    @pytest.mark.parametrize("component, layer", FAMILIES)
    def test_zeroing_one_head_removes_exactly_that_heads_contribution(
        self, moe, component: str, layer: int
    ) -> None:
        """Hard mask ``+1`` on every head but ``h``: the masked run's mixer
        output must equal the hand-computed per-head o-projection with head
        ``h`` taken from base and every other head from the counterfactual —
        and differ from the full swap by ``W_o[:, h] @ (base − cf)[h]`` alone."""
        executor = executor_for(
            _head_dbm_doc(component, layer),
            moe,
            base_texts=[TEXT],
            counterfactual_texts=[CF_TEXT],
        )
        gate = executor.stage("gate")
        groups = gate.groups
        assert groups is not None
        heads, width = groups
        kept = 2
        proj = _out_proj(moe, layer)
        weight = proj.weight.detach()
        assert proj.bias is None, "fixture assumption: no o-projection bias"

        pre_base = executor.read_value("pre_base")[0, 0].detach()
        pre_cf = executor.read_value("pre_cf")[0, 0].detach()
        # the plain mixer output is the whole premix projected: the identity the
        # hand computation rests on, checked before it is used
        out_base = executor.read_value("out_base")[0, 0].detach()
        torch.testing.assert_close(out_base, weight @ pre_base, rtol=1e-5, atol=1e-5)

        with torch.no_grad():
            gate.theta.fill_(1.0)
            gate.theta[kept] = -1.0
        executor.reset_reads()
        out_masked = executor.read_value("out_masked")[0, 0].detach()
        sources = [pre_base if head == kept else pre_cf for head in range(heads)]
        expected = _projected(weight, width, sources)
        torch.testing.assert_close(out_masked, expected, rtol=1e-5, atol=1e-5)

        with torch.no_grad():
            gate.theta.fill_(1.0)
        executor.reset_reads()
        out_full = executor.read_value("out_masked")[0, 0].detach()
        torch.testing.assert_close(out_full, weight @ pre_cf, rtol=1e-5, atol=1e-5)
        block = slice(kept * width, (kept + 1) * width)
        one_head = weight[:, block] @ (pre_base - pre_cf)[block]
        torch.testing.assert_close(
            out_masked - out_full, one_head, rtol=1e-5, atol=1e-5
        )
        # the removed head's contribution is not nothing — the check bit
        assert one_head.abs().max() > 0


# --------------------------------------------------------------------------- #
# reload: the bundle must be the fit's, on the fit's heads
# --------------------------------------------------------------------------- #


def _reload(
    moe: ModelBundle,
    theta: torch.Tensor,
    record: dict[str, Any],
    *,
    group: str | None = "head",
    component: str = FULL_ATTENTION[0],
    layer: int = FULL_ATTENTION[1],
) -> Gate:
    """Build a ``file_path`` gate against one fixture site from an in-memory
    bundle whose ``theta`` entry carries ``record`` as its stamped identity."""
    site = _site(moe, component, layer)
    spec = FeaturizerSpec(kind="gate", group=group, file_path="fit/gate.safetensors")
    stack = build_stack(
        "gate",
        {"gate": spec},
        width=site.shape.width or 0,
        load_tensors=lambda _path: TensorBundle(
            tensors={"theta": theta}, entry_coords={"theta": record}
        ),
        stage_cache={},
        site_shape=site.shape,
        site_component=site.component,
    )
    (stage,) = stack.stages
    assert isinstance(stage, Gate)
    return stage


@pytest.mark.unit
class TestReload:
    def test_the_fits_own_bundle_reloads_as_the_same_hard_mask(self, moe) -> None:
        heads, width = _map(moe, *FULL_ATTENTION)
        theta = torch.linspace(-1, 1, heads)
        record = {"group": "head", "group_map": "[8, 32]"}
        gate = _reload(moe, theta, record)
        assert gate.groups == (heads, width) and not gate.theta.requires_grad
        assert torch.equal(
            gate.featurize(torch.ones(heads * width))[0].view(heads, width)[:, 0],
            (theta > 0).float(),
        )

    def test_another_group_map_refuses(self, moe) -> None:
        with pytest.raises(ProtocolError, match="grouped over 4 heads of 64"):
            _reload(moe, torch.ones(8), {"group": "head", "group_map": "[4, 64]"})

    def test_another_head_count_refuses(self, moe) -> None:
        """A 16-head fit at an 8-head site. With a stamped map the map check
        speaks first (above); a bundle carrying only the group kind is refused
        by the count — and counted in heads, not coordinates."""
        with pytest.raises(ProtocolError, match="16 heads but the site here is 8"):
            _reload(moe, torch.ones(16), {"group": "head"})

    def test_a_per_coordinate_document_refuses_a_grouped_bundle(self, moe) -> None:
        with pytest.raises(ProtocolError, match="fitted with group='head'"):
            _reload(
                moe,
                torch.ones(8),
                {"group": "head", "group_map": "[8, 32]"},
                group=None,
            )

    def test_a_grouped_document_refuses_a_per_coordinate_bundle(self, moe) -> None:
        with pytest.raises(ProtocolError, match="256 heads but the site here is 8"):
            _reload(moe, torch.ones(256), {})

    def test_a_grouped_gate_after_a_rotation_refuses(self, moe) -> None:
        """Rule 23 refused this at load; the executor holds the same line when
        handed the chain directly: after a fitted basis "head" names nothing."""
        site = _site(moe, *FULL_ATTENTION)
        specs = {
            "rot": FeaturizerSpec(kind="subspace", k=64, parametrization="cayley"),
            "gate": FeaturizerSpec(kind="gate", group="head"),
        }
        with pytest.raises(ProtocolError, match="must be the first stage of its chain"):
            build_stack(
                ("rot", "gate"),
                specs,
                width=site.shape.width or 0,
                load_tensors=lambda _p: None,
                stage_cache={},
                site_shape=site.shape,
                site_component=site.component,
            )

    def test_a_grouped_gate_after_a_standardize_refuses(self, moe) -> None:
        """The rule is positional: a grouped gate is the first stage of its
        chain or it is refused, a per-coordinate standardize ahead of it
        included — the executor holds rule 23's line."""
        site = _site(moe, *FULL_ATTENTION)
        width = site.shape.width or 0
        specs = {
            "z": FeaturizerSpec(kind="standardize", file_path="fit/z.safetensors"),
            "gate": FeaturizerSpec(kind="gate", group="head"),
        }
        with pytest.raises(ProtocolError, match="must be the first stage of its chain"):
            build_stack(
                ("z", "gate"),
                specs,
                width=width,
                load_tensors=lambda _path: TensorBundle(
                    tensors={"mu": torch.zeros(width), "sigma": torch.ones(width)},
                    entry_coords={"mu": {}, "sigma": {}},
                ),
                stage_cache={},
                site_shape=site.shape,
                site_component=site.component,
            )

    def test_a_grouped_gate_needs_a_site_shape(self) -> None:
        with pytest.raises(ProtocolError, match="declares no shape"):
            build_stack(
                "gate",
                {"gate": FeaturizerSpec(kind="gate", group="head")},
                width=256,
                load_tensors=lambda _p: None,
                stage_cache={},
            )


#: The Qwen3.6 target's two head-major families are the same width and laid
#: out differently — 16 query heads of 256 and 32 value heads of 128, 4096
#: coordinates either way — which the tiny MoE fixture cannot show (its two
#: families share one map). Registry-only: no weights are loaded.
QWEN36 = "Qwen/Qwen3.6-35B-A3B"


@pytest.mark.unit
class TestOneFeaturizerOneGroupMap:
    """A gate named at two sites is one stage, built once and reused from the
    stage cache; the second site has to lay its units out exactly as the first
    did, or the shared ``theta`` would mean one thing per site."""

    def _build(self, stage_cache: dict, component: str) -> Gate:
        info = get_model_info(QWEN36)
        shape = component_shape(info, component)
        assert shape.width == 4096
        stack = build_stack(
            "gate",
            {"gate": FeaturizerSpec(kind="gate", group="head")},
            width=shape.width,
            load_tensors=lambda _p: None,
            stage_cache=stage_cache,
            site_shape=shape,
            site_component=component,
            model_info=info,
        )
        (stage,) = stack.stages
        assert isinstance(stage, Gate)
        return stage

    def test_the_registry_says_the_two_families_differ_only_in_layout(self) -> None:
        info = get_model_info(QWEN36)
        assert site_group_map(info, "head", "attention_premix") == (16, 256)
        assert site_group_map(info, "head", "delta_premix") == (32, 128)

    def test_a_second_site_with_the_same_map_reuses_the_stage(self) -> None:
        cache: dict = {}
        first = self._build(cache, "attention_premix")
        assert first.groups == (16, 256) and first.theta.shape == (16,)
        assert self._build(cache, "attention_premix") is first

    def test_a_second_site_laid_out_differently_refuses(self) -> None:
        """Same width, so the width check passes; only the map check can tell
        16 heads of 256 from 32 heads of 128."""
        cache: dict = {}
        self._build(cache, "attention_premix")
        with pytest.raises(ProtocolError, match="one featurizer, one group map"):
            self._build(cache, "delta_premix")


# --------------------------------------------------------------------------- #
# the expert-keyed gate (spec §2.5 ``group: expert_neuron``)
# --------------------------------------------------------------------------- #

#: 📐 fixture numbers: 128 experts, top-10, d_expert 32 — the token-major
#: contract width of ``expert_activation`` is 10 · 32 = 320.
EXPERTS, TOP_K, D_EXPERT = 128, 10, 32
MOE_LAYER = 0


@pytest.mark.unit
class TestExpertNeuronGateMath:
    def test_one_parameter_per_expert_neuron_whatever_top_k(self) -> None:
        gate = Gate(TOP_K * D_EXPERT, group="expert_neuron", groups=(EXPERTS, D_EXPERT))
        assert gate.theta.shape == (EXPERTS, D_EXPERT)
        assert gate.theta.numel() == EXPERTS * D_EXPERT
        assert gate.width == TOP_K * D_EXPERT
        assert _stage_width(gate) == TOP_K * D_EXPERT
        assert gate.needs_routing and not Gate(6).needs_routing

    def test_slots_use_the_parameters_of_the_expert_filling_them(self) -> None:
        """Two tokens routed to different experts use different rows of the
        table; change θ for expert ``e`` and only the slots holding ``e``
        change — on whichever slot ``e`` fills for each token."""
        gate = Gate(2 * 3, group="expert_neuron", groups=(4, 3))
        gate.eval()
        routing = torch.tensor([[[0, 1]], [[2, 0]]])  # token 0: e0,e1; token 1: e2,e0
        x = torch.ones(2, 1, 6)
        with torch.no_grad():
            gate.theta.fill_(-1.0)
        before, _ = gate.featurize(x, routing=routing)
        assert torch.equal(before, torch.zeros(2, 1, 6))
        with torch.no_grad():
            gate.theta[0] = torch.tensor([1.0, -1.0, 1.0])  # expert 0: neurons 0 and 2
        after, err = gate.featurize(x, routing=routing)
        expected = torch.zeros(2, 1, 6)
        expected[0, 0, 0:3] = torch.tensor(
            [1.0, 0.0, 1.0]
        )  # token 0 holds e0 in slot 0
        expected[1, 0, 3:6] = torch.tensor(
            [1.0, 0.0, 1.0]
        )  # token 1 holds e0 in slot 1
        assert torch.equal(after, expected)
        assert torch.equal(after + err, x)

    def test_an_inactive_experts_parameters_leave_a_token_alone(self) -> None:
        gate = Gate(2 * 3, group="expert_neuron", groups=(4, 3))
        gate.train()
        routing = torch.tensor([[[0, 1]]])
        x = torch.randn(1, 1, 6)
        with torch.no_grad():
            gate.theta.normal_()
        before, _ = gate.featurize(x, routing=routing)
        with torch.no_grad():
            gate.theta[2] = 5.0
            gate.theta[3] = -5.0
        after, _ = gate.featurize(x, routing=routing)
        assert torch.equal(after, before)

    def test_soft_and_hard_masks_follow_the_routing(self) -> None:
        gate = Gate(2 * 2, group="expert_neuron", groups=(3, 2))
        with torch.no_grad():
            gate.theta.copy_(torch.tensor([[2.0, -1.0], [0.0, 3.0], [-2.0, 1.0]]))
        routing = torch.tensor([[2, 1]])
        x = torch.ones(1, 4)
        gate.train()
        soft, _ = gate.featurize(x, routing=routing)
        assert torch.equal(soft, torch.sigmoid(gate.theta[[2, 1]]).reshape(1, 4))
        gate.eval()
        hard, _ = gate.featurize(x, routing=routing)
        assert torch.equal(hard, torch.tensor([[0.0, 1.0, 0.0, 1.0]]))

    def test_the_routing_table_is_required(self) -> None:
        gate = Gate(2 * 3, group="expert_neuron", groups=(4, 3))
        with pytest.raises(ProtocolError, match="needs the routing table"):
            gate.featurize(torch.ones(6))
        with pytest.raises(ProtocolError, match="2 routed slots"):
            gate.featurize(torch.ones(1, 6), routing=torch.tensor([[0, 1, 2]]))

    def test_the_map_must_tile_slots(self) -> None:
        with pytest.raises(ValueError, match="not a whole number"):
            Gate(7, group="expert_neuron", groups=(4, 3))

    def test_from_theta_keeps_the_table_and_needs_the_site_width(self) -> None:
        theta = torch.arange(12.0).reshape(4, 3) - 5
        gate = Gate.from_theta(theta, group="expert_neuron", groups=(4, 3), width=6)
        assert gate.width == 6 and gate.groups == (4, 3)
        assert torch.equal(gate.theta, theta) and not gate.theta.requires_grad
        # a flat table from a bundle comes back two-dimensional
        flat = Gate.from_theta(
            theta.reshape(-1), group="expert_neuron", groups=(4, 3), width=6
        )
        assert flat.theta.shape == (4, 3)
        with pytest.raises(ValueError, match="needs the site width"):
            Gate.from_theta(theta, group="expert_neuron", groups=(4, 3))
        with pytest.raises(ValueError, match="needs 12 parameters"):
            Gate.from_theta(
                torch.ones(6), group="expert_neuron", groups=(4, 3), width=6
            )


@pytest.mark.unit
class TestExpertNeuronGroupMap:
    def test_the_map_is_the_fixtures_expert_table(self, moe) -> None:
        site = _site(moe, "expert_activation", MOE_LAYER)
        assert gate_group_map(
            "expert_neuron",
            site.shape,
            site.shape.width or 0,
            component="expert_activation",
            info=moe.info,
        ) == (EXPERTS, D_EXPERT)
        assert site.shape.width == TOP_K * D_EXPERT

    @pytest.mark.parametrize(
        "component", ["shared_expert_activation", "expert_output", "block_output"]
    )
    def test_any_other_component_is_refused_by_name(self, moe, component) -> None:
        shape = component_shape(moe.info, component)
        with pytest.raises(ValidationError, match=f"'{component}'"):
            gate_group_map(
                "expert_neuron",
                shape,
                shape.width or 0,
                component=component,
                info=moe.info,
            )

    def test_a_gate_not_at_the_components_own_width_is_refused(self, moe) -> None:
        shape = component_shape(moe.info, "expert_activation")
        with pytest.raises(ValidationError, match="no stage before it") as err:
            gate_group_map(
                "expert_neuron", shape, 8, component="expert_activation", info=moe.info
            )
        assert err.value.rule == 23

    def test_the_param_shape_is_the_table(self) -> None:
        assert gate_param_shape(None, None, 320) == (320,)
        assert gate_param_shape("head", (8, 32), 256) == (8,)
        assert gate_param_shape("expert_neuron", (EXPERTS, D_EXPERT), 320) == (
            EXPERTS,
            D_EXPERT,
        )
        # an indexed map (§2.5 `boundary`): one β whatever the width
        assert gate_param_shape(None, None, 320, parametrization="boundary") == (1,)


# --------------------------------------------------------------------------- #
# the gate inside a document: routing join, swap under mismatch, mismatch count
# --------------------------------------------------------------------------- #


def _expert_dbm_doc(pos: Any = -1) -> dict[str, Any]:
    """A DBM-shaped document on one MoE layer: the counterfactual routed
    interior read through an expert-keyed gate and swapped into the base run
    through the same gate, a plain gate doing the same on the shared expert,
    plus the raw reads the hand computation needs (both inputs' activations
    and routing tables)."""
    return _bound_doc(
        {
            "routed_cf": (
                UNWRITTEN,
                {"site": "routed", "pos": pos, "featurizer": "routed_gate"},
            ),
            "shared_cf": (
                UNWRITTEN,
                {"site": "shared", "pos": pos, "featurizer": "shared_gate"},
            ),
            "pre_base": (ORIGINAL_BASE, {"site": "routed", "pos": pos}),
            "pre_cf": (UNWRITTEN, {"site": "routed", "pos": pos}),
            "idx_base": (ORIGINAL_BASE, {"site": "idx", "pos": pos}),
            "idx_cf": (UNWRITTEN, {"site": "idx", "pos": pos}),
            "post": ("masked", {"site": "routed", "pos": pos}),
            "shared_base": (ORIGINAL_BASE, {"site": "shared", "pos": pos}),
            "shared_raw_cf": (UNWRITTEN, {"site": "shared", "pos": pos}),
            "shared_post": ("masked", {"site": "shared", "pos": pos}),
        },
        sites={
            "routed": {"component": "expert_activation", "layers": [MOE_LAYER]},
            "shared": {
                "component": "shared_expert_activation",
                "layers": [MOE_LAYER],
            },
            "idx": {"component": "expert_idx", "layers": [MOE_LAYER]},
        },
        featurizers={
            "routed_gate": {"kind": "gate", "group": "expert_neuron"},
            "shared_gate": {"kind": "gate"},
        },
        writes={
            "mask_routed": {
                "site": "routed",
                "pos": pos,
                "featurizer": "routed_gate",
                "do": {"swap": "routed_cf"},
            },
            "mask_shared": {
                "site": "shared",
                "pos": pos,
                "featurizer": "shared_gate",
                "do": {"swap": "shared_cf"},
            },
        },
    )


def _aligned_by_hand(
    pre_base: torch.Tensor,
    pre_cf: torch.Tensor,
    idx_base: torch.Tensor,
    idx_cf: torch.Tensor,
    *,
    open_experts: set[int] | None = None,
) -> tuple[torch.Tensor, int]:
    """The plan's swap rule, slot by slot: a base slot holding expert ``e``
    takes the counterfactual activation of ``e`` at the same token when ``e``
    is active there (and, when ``open_experts`` is given, only if ``e``'s gate
    is open), else keeps its base value. Also the number of base slots with no
    counterfactual source."""
    rows, n_pos = idx_base.shape[0], idx_base.shape[1]
    base = pre_base.reshape(rows, n_pos, TOP_K, D_EXPERT)
    cf = pre_cf.reshape(rows, n_pos, TOP_K, D_EXPERT)
    expected = base.clone()
    missing = 0
    for i in range(rows):
        for p in range(n_pos):
            for k in range(TOP_K):
                e = int(idx_base[i, p, k])
                hits = (idx_cf[i, p] == e).nonzero()
                if not len(hits):
                    missing += 1
                    continue
                if open_experts is None or e in open_experts:
                    expected[i, p, k] = cf[i, p, int(hits[0])]
    return expected.reshape(pre_base.shape), missing


@pytest.mark.property
class TestExpertSwap:
    @pytest.fixture()
    def executor(self, moe):
        return executor_for(
            _expert_dbm_doc(), moe, base_texts=[TEXT], counterfactual_texts=[CF_TEXT]
        )

    def test_the_document_builds_the_table_and_a_separate_shared_gate(
        self, executor
    ) -> None:
        routed, shared = executor.stage("routed_gate"), executor.stage("shared_gate")
        assert isinstance(routed, Gate) and isinstance(shared, Gate)
        assert routed.groups == (EXPERTS, D_EXPERT)
        assert routed.theta.shape == (EXPERTS, D_EXPERT)
        assert shared.groups is None and shared.theta.shape == (D_EXPERT,)
        assert routed.theta is not shared.theta
        assert not routed.training and not shared.training

    def test_the_fixture_routes_the_pair_differently_at_the_last_token(
        self, executor
    ) -> None:
        """Pinned so the swap tests below decide something: some but not all of
        the base experts are active on the counterfactual side."""
        idx_base = executor.read_value("idx_base")[0, 0]
        idx_cf = executor.read_value("idx_cf")[0, 0]
        shared = set(idx_base.tolist()) & set(idx_cf.tolist())
        assert 0 < len(shared) < TOP_K

    def test_a_hard_open_gate_swaps_matched_slots_and_keeps_the_rest(
        self, executor
    ) -> None:
        """θ = +1 everywhere: every base slot whose expert is active on the
        counterfactual side receives that expert's counterfactual activation,
        wherever it sits in the counterfactual ranking; a slot whose expert is
        inactive there keeps its base value exactly."""
        routed = executor.stage("routed_gate")
        with torch.no_grad():
            routed.theta.fill_(1.0)
        pre_base, pre_cf = (
            executor.read_value("pre_base"),
            executor.read_value("pre_cf"),
        )
        idx_base, idx_cf = (
            executor.read_value("idx_base"),
            executor.read_value("idx_cf"),
        )
        post = executor.read_value("post")
        expected, missing = _aligned_by_hand(pre_base, pre_cf, idx_base, idx_cf)
        torch.testing.assert_close(post, expected, atol=0.0, rtol=0.0)
        assert 0 < missing < TOP_K
        # the slot-for-slot swap the plain mechanism does is *not* what landed
        assert not torch.equal(post, pre_cf)

    def test_closing_one_experts_gate_changes_only_its_slots(self, executor) -> None:
        """Hard-open everywhere but expert ``e``: only the slots holding ``e``
        differ from the fully open run, and they hold base."""
        routed = executor.stage("routed_gate")
        idx_base, idx_cf = (
            executor.read_value("idx_base"),
            executor.read_value("idx_cf"),
        )
        matched = [
            int(e) for e in idx_base[0, 0].tolist() if e in set(idx_cf[0, 0].tolist())
        ]
        closed = matched[0]
        with torch.no_grad():
            routed.theta.fill_(1.0)
        executor.reset_reads()
        open_all = executor.read_value("post")
        with torch.no_grad():
            routed.theta[closed] = -1.0
        executor.reset_reads()
        one_closed = executor.read_value("post")
        pre_base, pre_cf = (
            executor.read_value("pre_base"),
            executor.read_value("pre_cf"),
        )
        expected, _ = _aligned_by_hand(
            pre_base,
            pre_cf,
            idx_base,
            idx_cf,
            open_experts=set(range(EXPERTS)) - {closed},
        )
        torch.testing.assert_close(one_closed, expected, atol=0.0, rtol=0.0)
        slot = idx_base[0, 0].tolist().index(closed)
        differs = (one_closed != open_all).reshape(TOP_K, D_EXPERT).any(-1)
        assert differs.tolist() == [k == slot for k in range(TOP_K)]

    def test_an_inactive_experts_parameters_do_not_reach_the_output(
        self, executor
    ) -> None:
        routed = executor.stage("routed_gate")
        idx_base = executor.read_value("idx_base")[0, 0].tolist()
        inactive = next(e for e in range(EXPERTS) if e not in idx_base)
        with torch.no_grad():
            routed.theta.fill_(1.0)
        executor.reset_reads()
        before = executor.read_value("post")
        with torch.no_grad():
            routed.theta[inactive] = -7.0
        executor.reset_reads()
        after = executor.read_value("post")
        assert torch.equal(after, before)

    def test_the_shared_gate_is_a_plain_swap_on_its_own_keys(self, executor) -> None:
        routed, shared = executor.stage("routed_gate"), executor.stage("shared_gate")
        with torch.no_grad():
            shared.theta.fill_(1.0)
            shared.theta[3] = -1.0
            routed.theta.fill_(-1.0)
        executor.reset_reads()
        post = executor.read_value("shared_post")
        base, cf = (
            executor.read_value("shared_base"),
            executor.read_value("shared_raw_cf"),
        )
        expected = cf.clone()
        expected[..., 3] = base[..., 3]
        torch.testing.assert_close(post, expected, atol=0.0, rtol=0.0)
        # and the routed table's state leaves the shared expert alone
        with torch.no_grad():
            routed.theta.fill_(1.0)
        executor.reset_reads()
        torch.testing.assert_close(
            executor.read_value("shared_post"), expected, atol=0.0, rtol=0.0
        )

    def test_the_mismatch_count_is_recorded_per_write_layer_and_example(
        self, executor
    ) -> None:
        executor.read_value("post")
        idx_base, idx_cf = (
            executor.read_value("idx_base"),
            executor.read_value("idx_cf"),
        )
        _, missing = _aligned_by_hand(
            executor.read_value("pre_base"),
            executor.read_value("pre_cf"),
            idx_base,
            idx_cf,
        )
        assert executor.routing_mismatch == {
            ("mask_routed", MOE_LAYER, 0): (missing, TOP_K)
        }

    def test_all_positions_count_every_slot(self, moe) -> None:
        executor = executor_for(
            _expert_dbm_doc("all"),
            moe,
            base_texts=[TEXT],
            counterfactual_texts=[CF_TEXT],
        )
        routed = executor.stage("routed_gate")
        with torch.no_grad():
            routed.theta.fill_(1.0)
        post = executor.read_value("post")
        idx_base, idx_cf = (
            executor.read_value("idx_base"),
            executor.read_value("idx_cf"),
        )
        expected, missing = _aligned_by_hand(
            executor.read_value("pre_base"),
            executor.read_value("pre_cf"),
            idx_base,
            idx_cf,
        )
        torch.testing.assert_close(post, expected, atol=0.0, rtol=0.0)
        n_pos = idx_base.shape[1]
        assert executor.routing_mismatch == {
            ("mask_routed", MOE_LAYER, 0): (missing, n_pos * TOP_K)
        }

    def test_an_operand_without_routing_is_refused(self, moe) -> None:
        """A swap through the expert-keyed gate needs a source whose slots can
        be joined by expert; a read from anywhere else has no expert ids."""
        doc = _expert_dbm_doc()
        doc["method"]["sites"]["other"] = {
            "component": "block_input",
            "layers": [MOE_LAYER],
        }
        doc["method"]["reads"]["routed_cf"] = {"site": "other", "pos": -1}
        executor = executor_for(
            doc, moe, base_texts=[TEXT], counterfactual_texts=[CF_TEXT]
        )
        with pytest.raises(ProtocolError, match="carries no routing table"):
            executor.read_value("post")

    def test_dims_through_an_expert_keyed_gate_is_refused(self, moe) -> None:
        doc = _expert_dbm_doc()
        doc["method"]["writes"]["mask_routed"]["dims"] = [0, 1]
        doc["method"]["reads"]["routed_cf"]["dims"] = [0, 1]
        executor = executor_for(
            doc, moe, base_texts=[TEXT], counterfactual_texts=[CF_TEXT]
        )
        with pytest.raises(ProtocolError, match="'dims' through an expert-keyed gate"):
            executor.read_value("post")

    def test_a_row_window_records_role_rows(self, moe) -> None:
        """Microbatched, the routing slices by row like every other capture,
        and the records name the role's row, not the window's."""
        whole = executor_for(
            _expert_dbm_doc(),
            moe,
            base_texts=[TEXT, CF_TEXT, TEXT],
            counterfactual_texts=[CF_TEXT, TEXT, CF_TEXT],
        )
        windowed = executor_for(
            _expert_dbm_doc(),
            moe,
            base_texts=[TEXT, CF_TEXT, TEXT],
            counterfactual_texts=[CF_TEXT, TEXT, CF_TEXT],
            batch_rows=2,
        )
        for executor in (whole, windowed):
            with torch.no_grad():
                executor.stage("routed_gate").theta.fill_(1.0)
        torch.testing.assert_close(
            windowed.read_value("post"), whole.read_value("post"), rtol=1e-6, atol=1e-6
        )
        assert windowed.routing_mismatch == whole.routing_mismatch
        assert sorted(key[2] for key in whole.routing_mismatch) == [0, 1, 2]


@pytest.mark.unit
class TestExpertNeuronReload:
    def test_the_fits_own_table_reloads_as_the_same_hard_mask(self, moe) -> None:
        site = _site(moe, "expert_activation", MOE_LAYER)
        theta = torch.linspace(-1, 1, EXPERTS * D_EXPERT).reshape(EXPERTS, D_EXPERT)
        spec = FeaturizerSpec(
            kind="gate", group="expert_neuron", file_path="fit/gate.safetensors"
        )
        stack = build_stack(
            "gate",
            {"gate": spec},
            width=site.shape.width or 0,
            load_tensors=lambda _path: TensorBundle(
                tensors={"theta": theta},
                entry_coords={
                    "theta": {"group": "expert_neuron", "group_map": "[128, 32]"}
                },
            ),
            stage_cache={},
            site_shape=site.shape,
            site_component=site.component,
            model_info=moe.info,
        )
        (gate,) = stack.stages
        assert isinstance(gate, Gate)
        assert gate.groups == (EXPERTS, D_EXPERT) and not gate.theta.requires_grad
        assert torch.equal(gate.theta, theta)

    def test_another_expert_table_refuses(self, moe) -> None:
        site = _site(moe, "expert_activation", MOE_LAYER)
        spec = FeaturizerSpec(
            kind="gate", group="expert_neuron", file_path="fit/gate.safetensors"
        )
        with pytest.raises(
            ProtocolError, match="grouped over 64 experts of 32 neurons"
        ):
            build_stack(
                "gate",
                {"gate": spec},
                width=site.shape.width or 0,
                load_tensors=lambda _path: TensorBundle(
                    tensors={"theta": torch.zeros(64, 32)},
                    entry_coords={
                        "theta": {"group": "expert_neuron", "group_map": "[64, 32]"}
                    },
                ),
                stage_cache={},
                site_shape=site.shape,
                site_component=site.component,
                model_info=moe.info,
            )

    def test_a_head_grouped_bundle_refuses_an_expert_document(self, moe) -> None:
        site = _site(moe, "expert_activation", MOE_LAYER)
        spec = FeaturizerSpec(
            kind="gate", group="expert_neuron", file_path="fit/gate.safetensors"
        )
        with pytest.raises(ProtocolError, match="fitted with group='head'"):
            build_stack(
                "gate",
                {"gate": spec},
                width=site.shape.width or 0,
                load_tensors=lambda _path: TensorBundle(
                    tensors={"theta": torch.zeros(EXPERTS * D_EXPERT)},
                    entry_coords={"theta": {"group": "head"}},
                ),
                stage_cache={},
                site_shape=site.shape,
                site_component=site.component,
                model_info=moe.info,
            )

    def test_an_expert_gate_needs_the_model(self, moe) -> None:
        site = _site(moe, "expert_activation", MOE_LAYER)
        with pytest.raises(ProtocolError, match="no expert table"):
            build_stack(
                "gate",
                {"gate": FeaturizerSpec(kind="gate", group="expert_neuron")},
                width=site.shape.width or 0,
                load_tensors=lambda _p: None,
                stage_cache={},
                site_shape=site.shape,
                site_component=site.component,
            )
