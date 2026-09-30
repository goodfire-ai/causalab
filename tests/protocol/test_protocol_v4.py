"""The reads-first surface of a `Document` (protocol v4, §2.7–§2.10).

A read is an address; the model that lists it decides which forward the
address is gathered from, so the runtime's unit of value is the read
**bound to a model** (`ReadRef`), and an aggregation is a reduction
over one bound read that lives where it is consumed
(`BoundAggregation`). These tests pin the accessors the runtime reads
through — first synthesized from a protocol-3 document, later authored.
"""

from __future__ import annotations

from typing import Any

import pytest

from causalab.protocol.schema import (
    AggregationSpec,
    BoundAggregation,
    ReadRef,
    do_operand_slots,
    operand_params,
    operand_reads,
    parse_document,
    read_is_vocabulary,
)

from tests.protocol._docs import (
    LOGIT_DIFF,
    aggregation,
    base_doc,
    in_order,
    saved,
    term,
)


pytestmark = pytest.mark.unit


def _doc(raw: dict[str, Any] | None = None):
    return parse_document(in_order(raw or base_doc()))


class TestReadsOnModels:
    def test_each_model_lists_the_reads_taken_on_it(self) -> None:
        doc = _doc()
        assert doc.intervened_models["patched"].reads == ("logits",)
        assert doc.intervened_models["patched"].write_names == ("patch",)
        assert not doc.intervened_models["patched"].is_unwritten

    def test_read_refs_bind_every_read_to_its_model(self) -> None:
        doc = _doc()
        assert doc.read_refs() == (
            ReadRef("v_cf", "original_counterfactual"),
            ReadRef("logits", "patched"),
        )
        assert doc.models_of("v_cf") == ("original_counterfactual",)
        assert doc.models_of("no_such_read") == ()
        assert doc.bound("logits") == ReadRef("logits", "patched")

    def test_group_of_a_bound_read_is_its_model_on_its_input(self) -> None:
        doc = _doc()
        assert doc.group_of(ReadRef("v_cf", "original_counterfactual")) == (
            "original_counterfactual",
            "counterfactual",
        )
        assert doc.group_of(ReadRef("logits", "patched")) == ("patched", "base")
        assert doc.input_of("patched") == "base"
        with pytest.raises(KeyError):
            doc.group_of(ReadRef("logits", None))

    def test_a_write_list_under_a_sweep_is_not_known_to_be_unwritten(self) -> None:
        raw = base_doc()
        raw["method"]["intervened_models"]["patched"]["writes"] = {
            "sweep": [["patch"], []]
        }
        im = _doc(raw).intervened_models["patched"]
        assert im.write_names is None
        assert not im.is_unwritten

    def test_str_of_a_ref_spells_model_slash_read(self) -> None:
        assert str(ReadRef("logits", "patched")) == "patched/logits"
        assert str(ReadRef("logits", None)) == "logits"


class TestOperands:
    def test_operand_reads_are_bound_and_params_are_not(self) -> None:
        raw = base_doc()
        raw["method"]["params"] = {"direction": {"shape": [8], "init": "zeros"}}
        raw["method"]["writes"]["patch"]["do"] = {
            "add_scaled": {"op": "direction", "alpha": 0.5}
        }
        raw["method"]["writes"]["swap"] = {
            "site": "tgt",
            "pos": -1,
            "do": {"swap": "v_cf"},
        }
        raw["method"]["intervened_models"]["patched"]["writes"] = ["patch", "swap"]
        doc = _doc(raw)
        assert operand_reads(doc, doc.writes["swap"].do) == (
            ReadRef("v_cf", "original_counterfactual"),
        )
        assert operand_reads(doc, doc.writes["patch"].do) == ()
        assert operand_params(doc, doc.writes["patch"].do) == ("direction",)
        # the parser binds a bare read operand to its one model (§2.7)
        assert do_operand_slots(doc.writes["swap"].do) == {
            "": ReadRef("v_cf", "original_counterfactual")
        }
        assert do_operand_slots(doc.writes["patch"].do) == {
            "op": "direction",
            "alpha": 0.5,
        }


class TestAggregations:
    def test_a_saved_metric_is_an_aggregation_owned_by_its_save_entry(self) -> None:
        doc = _doc()
        (agg,) = doc.aggregations()
        assert isinstance(agg, BoundAggregation)
        assert agg.owner == "save[0]"
        assert agg.label == "ld"
        assert agg.read == ReadRef("logits", "patched")
        assert agg.target is None
        assert isinstance(agg.spec, AggregationSpec)
        assert agg.spec.kind == agg.kind == "logit_diff"
        assert agg.spec.fields == {"a": "cf_answer", "b": "base_answer"}
        assert doc.saved_aggregations() == (agg,)
        assert (
            doc.aggregation_at("save[0]") is agg or doc.aggregation_at("save[0]") == agg
        )
        assert doc.aggregation_at("save[7]") is None

    def test_a_kl_target_is_a_bound_read(self) -> None:
        raw = base_doc()
        raw["method"]["reads"]["logits_orig"] = {"site": "lm_head", "pos": -1}
        raw["method"]["intervened_models"]["original_base"] = {
            "input": "base",
            "reads": ["logits_orig"],
        }
        raw["method"]["save"].append(
            saved(
                "logits",
                "patched",
                "kl.json",
                aggregation(
                    "kl", target={"read": "logits_orig", "model": "original_base"}
                ),
            )
        )
        doc = _doc(raw)
        drift = doc.aggregation_at("save[1]")
        assert drift is not None
        assert drift.read == ReadRef("logits", "patched")
        assert drift.target == ReadRef("logits_orig", "original_base")
        assert drift.spec.fields["target"] == ReadRef("logits_orig", "original_base")

    def test_objective_and_eval_terms_own_their_aggregations(self) -> None:
        raw = base_doc()
        raw["method"]["featurizers"] = {
            "rot": {"kind": "subspace", "k": 8, "parametrization": "cayley"}
        }
        raw["method"]["reads"]["v_cf"]["featurizer"] = "rot"
        raw["method"]["writes"]["patch"]["featurizer"] = "rot"
        ce = aggregation("cross_entropy", target="label")
        raw["method"]["train"] = {
            "objective": {"ce": term("logits", "patched", dict(ce), weight=1.0)},
            "params": ["rot"],
            "optimizer": {"name": "adamw", "lr": 1e-3},
            "steps": {"epochs": 1},
            "batch": {"pairs": 2},
            "eval": {
                "every": {"epochs": 1},
                "split": "eval",
                "aggregations": {
                    "ld": term("logits", "patched", dict(LOGIT_DIFF)),
                    "ce": term("logits", "patched", dict(ce)),
                },
            },
            "early_stop": {"on": "ld", "patience": 1, "mode": "max"},
            "seed": 0,
        }
        raw["method"]["save"].append(saved("logits", "patched", "ce.json", dict(ce)))
        doc = _doc(raw)
        owners = [agg.owner for agg in doc.aggregations()]
        assert owners == [
            "save[0]",
            "save[1]",
            "train.objective.ce",
            "train.eval.aggregations.ld",
            "train.eval.aggregations.ce",
        ]
        (ce,) = doc.objective_aggregations()
        assert ce.owner == "train.objective.ce" and ce.label == "ce"
        # a cross-entropy `target` is a dataset column, not a read
        assert ce.target is None and ce.spec.fields["target"] == "label"
        assert [agg.label for agg in doc.eval_aggregations()] == ["ld", "ce"]
        assert doc.early_stop_label() == "ld"

    def test_read_is_vocabulary_asks_the_read_alone(self) -> None:
        raw = base_doc()
        raw["method"]["reads"]["logits"]["dims"] = [0, 1, 2]
        doc = _doc(raw)
        assert read_is_vocabulary(_doc(), "logits")
        assert not read_is_vocabulary(_doc(), "v_cf")  # tgt is not lm_head
        assert not read_is_vocabulary(doc, "logits")  # dims re-index the axis
        assert not read_is_vocabulary(doc, "missing")


class TestUnwritten:
    def test_the_document_says_which_models_land_no_write(self) -> None:
        doc = _doc()
        assert doc.is_unwritten("original_counterfactual")  # lands no write
        assert not doc.is_unwritten("patched")
        raw = base_doc()
        raw["method"]["intervened_models"]["patched"]["writes"] = {
            "sweep": [["patch"], []]
        }
        assert not _doc(raw).is_unwritten("patched")  # unknown, not unwritten


class TestSaveAxes:
    """A value inside a ``save`` entry has a name identity — the entry's index
    — so a sweep or axis wrapper there is an axis (``save[i].aggregation.k``),
    where a wrapper inside any other list still has none."""

    def test_a_sweep_inside_a_save_entry_is_an_axis(self) -> None:
        from causalab.protocol.lowering import find_axes, substitute

        raw = in_order(base_doc())
        raw["method"]["save"][0]["aggregation"] = {
            "kind": "top_k",
            "k": {"sweep": [2, 4]},
        }
        (axis,) = find_axes(raw)
        assert axis.id == "save[0].aggregation.k"
        assert axis.path == ("method", "save[0]", "aggregation", "k")
        assert axis.values == (2, 4)
        point = substitute(raw, {axis.path: 4}, ())
        assert point["method"]["save"][0]["aggregation"] == {"kind": "top_k", "k": 4}

    def test_a_sweep_inside_any_other_list_still_has_no_name(self) -> None:
        from causalab.protocol.lowering import find_axes
        from causalab.protocol.rules.errors import ValidationError

        raw = in_order(base_doc())
        raw["method"]["reads"]["logits"]["dims"] = [{"sweep": [0, 1]}]
        with pytest.raises(ValidationError, match="no name identity"):
            find_axes(raw)

    def test_a_save_axis_labels_as_the_field_it_sweeps(self) -> None:
        from causalab.protocol.lowering import short_coords

        assert short_coords({"save[0].aggregation.k": 4, "sites.tgt.layers": 3}) == {
            "k": 4,
            "tgt.layers": 3,
        }
        assert short_coords({"save[2].reduce": "mean"}) == {"reduce": "mean"}

    def test_a_set_override_addresses_a_save_entry_by_index(self) -> None:
        from causalab.io.sources import apply_overrides

        raw = in_order(base_doc())
        raw["method"]["save"][0]["aggregation"] = {"kind": "top_k", "k": 2}
        out = apply_overrides(raw, {"save[0].aggregation.k": 4})
        assert out["method"]["save"][0]["aggregation"]["k"] == 4
