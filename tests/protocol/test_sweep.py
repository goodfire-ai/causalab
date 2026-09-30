"""Sweep discovery and deterministic expansion (spec §3)."""

from __future__ import annotations

import pytest

from causalab.protocol.rules.errors import ValidationError
from causalab.neural.shared.sweep import expand
from causalab.protocol.lowering import coordinate_label, find_axes

from tests.protocol._docs import LOGIT_DIFF, base_doc, in_order, term

pytestmark = pytest.mark.unit


def swept_doc():
    raw = base_doc()
    raw["method"]["positions"] = {
        "tap": {"sweep": [{"index": -1}, {"variable": "subject"}]}
    }
    raw["method"]["sites"]["tgt"]["layers"] = {"sweep": {"range": [0, 4]}}
    raw["method"]["reads"]["v_cf"]["pos"] = "tap"
    raw["method"]["writes"]["patch"]["pos"] = "tap"
    return in_order(raw)


def test_axes_in_document_order():
    axes = find_axes(swept_doc())
    assert [a.id for a in axes] == ["positions.tap", "sites.tgt.layers"]
    assert axes[1].values == (0, 1, 2, 3)


def test_cross_product_last_axis_fastest():
    expansion = expand(swept_doc())
    assert len(expansion.points) == 8
    coords = [p.coords for p in expansion.points]
    assert coords[0] == {"positions.tap": {"index": -1}, "sites.tgt.layers": 0}
    assert coords[1] == {"positions.tap": {"index": -1}, "sites.tgt.layers": 1}
    assert coords[4] == {
        "positions.tap": {"variable": "subject"},
        "sites.tgt.layers": 0,
    }


def test_substitution_produces_concrete_points():
    expansion = expand(swept_doc())
    point = expansion.points[5]
    assert point.raw["method"]["sites"]["tgt"]["layers"] == 1
    assert point.raw["method"]["positions"]["tap"] == {"variable": "subject"}
    # entities off the axes are untouched
    assert (
        point.raw["method"]["reads"]["logits"]
        == swept_doc()["method"]["reads"]["logits"]
    )


def test_unswept_document_expands_to_itself():
    raw = base_doc()
    expansion = expand(raw)
    assert not expansion.is_swept
    assert len(expansion.points) == 1
    assert expansion.points[0].raw == raw


def test_range_step():
    raw = base_doc()
    raw["method"]["sites"]["tgt"]["layers"] = {"sweep": {"range": [0, 10, 3]}}
    axes = find_axes(raw)
    assert axes[0].values == (0, 3, 6, 9)


def test_wrapper_inside_list_rejected():
    raw = base_doc()
    raw["method"]["reads"]["v_cf"]["dims"] = [0, {"sweep": [1, 2]}]
    with pytest.raises(ValidationError) as err:
        find_axes(raw)
    assert err.value.rule == 14


def test_nested_wrapper_rejected():
    raw = base_doc()
    raw["method"]["sites"]["tgt"]["layers"] = {"sweep": [1, {"sweep": [2, 3]}]}
    with pytest.raises(ValidationError) as err:
        find_axes(raw)
    assert err.value.rule == 14


def test_empty_axis_rejected():
    raw = base_doc()
    raw["method"]["sites"]["tgt"]["layers"] = {"sweep": []}
    with pytest.raises(ValidationError) as err:
        find_axes(raw)
    assert err.value.rule == 14


def test_coordinate_labels():
    assert coordinate_label({"featurizers.rot.k": 8}, entry="rot") == "[k=8]"
    assert coordinate_label({"sites.target.layers": 5}) == "[target.layers=5]"
    assert coordinate_label({"train.seed": 0}) == "[seed=0]"


def test_a_named_objective_terms_weight_is_one_axis():
    """The shared penalty over many gates is one weight, so sweeping it is one
    axis — the coordinate a sparsity curve is drawn along (§2.11, §3)."""
    raw = base_doc()
    raw["method"]["train"] = {
        "objective": {
            "fit": term("logits", "patched", dict(LOGIT_DIFF), weight=1.0),
            "sparsity": {"weight": {"sweep": [0.001, 0.01, 0.1]}, "l1": ["g0", "g1"]},
        },
        "params": ["g0", "g1"],
        "optimizer": {"name": "adamw", "lr": 1e-2},
        "steps": {"epochs": 1},
        "batch": {"pairs": 2},
    }
    axes = find_axes(in_order(raw))
    assert [a.id for a in axes] == ["train.objective.sparsity.weight"]
    assert axes[0].values == (0.001, 0.01, 0.1)
    expansion = expand(in_order(raw))
    assert len(expansion.points) == 3
    point = expansion.points[1]
    assert point.coords == {"train.objective.sparsity.weight": 0.01}
    assert point.raw["method"]["train"]["objective"]["sparsity"] == {
        "weight": 0.01,
        "l1": ["g0", "g1"],
    }
    assert coordinate_label(point.coords) == "[objective.sparsity.weight=0.01]"
