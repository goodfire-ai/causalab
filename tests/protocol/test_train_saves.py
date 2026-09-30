"""A save entry that names a training metric (§2.12).

``{"train": "<name>", "file_path": …}`` names a named objective term or an
eval aggregation label and means the save entry ``{"read", "model",
"aggregation", "file_path"}`` copied from that term, so a fit's saved
tables and its loss and eval terms share one definition. Inside a save
every read reference is ``{"read", "model"}``: the bare-name sugar (§2.7)
stays with write operands and with ``kl`` / ``js`` targets outside a save.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from causalab.protocol.rules.errors import ParseError, ValidationError
from causalab.protocol.schema import ReadRef, inline_train_saves, parse_document
from causalab.protocol.schema.explicit import canonical_bytes, canonicalize

from tests.protocol._docs import (
    LOGIT_DIFF,
    UNWRITTEN,
    aggregation,
    base_doc,
    in_order,
    saved,
    term,
)


pytestmark = pytest.mark.unit

CE = aggregation("cross_entropy", target="label")


def fit_doc(*save: dict[str, Any]) -> dict[str, Any]:
    """`base_doc` as a DAS fit: a named ``ce`` objective term beside an
    ``l2`` regularizer, and an ``ld`` eval label — the ``das.json`` shape.
    ``save`` replaces the manifest; the trained rotation is always saved."""
    raw = base_doc()
    method = raw["method"]
    method["featurizers"] = {
        "rot": {"kind": "subspace", "k": 8, "parametrization": "cayley"}
    }
    method["reads"]["v_cf"]["featurizer"] = "rot"
    method["writes"]["patch"]["featurizer"] = "rot"
    method["train"] = {
        "objective": {
            "ce": term("logits", "patched", dict(CE), weight=1.0),
            "shrink": {"weight": 0.01, "l2": "rot"},
        },
        "params": ["rot"],
        "optimizer": {"name": "adamw", "lr": 1e-3},
        "steps": {"epochs": 1},
        "batch": {"pairs": 2},
        "eval": {
            "every": {"epochs": 1},
            "split": "weekdays/data#test",
            "aggregations": {"ld": term("logits", "patched", dict(LOGIT_DIFF))},
        },
        "early_stop": {"on": "ld", "patience": 1, "mode": "max"},
        "seed": 0,
    }
    method["save"] = [
        *save,
        {"value": "rot", "site": "tgt", "file_path": "rot.safetensors"},
    ]
    return in_order(raw)


def by_reference() -> dict[str, Any]:
    return fit_doc(
        {"train": "ld", "file_path": "ld.json"},
        {"train": "ce", "file_path": "ce.json"},
    )


def inline() -> dict[str, Any]:
    return fit_doc(
        saved("logits", "patched", "ld.json", dict(LOGIT_DIFF)),
        saved("logits", "patched", "ce.json", dict(CE)),
    )


def _kl_doc(target: Any) -> dict[str, Any]:
    """`fit_doc` with a second logits read on the un-intervened base model,
    so a ``kl`` term has a target to compare against — listed by one model,
    so the bare name binds wherever the sugar is legal."""
    raw = fit_doc()
    method = raw["method"]
    method["reads"]["logits_orig"] = {"site": "lm_head", "pos": -1}
    method["intervened_models"]["original_base"] = {
        "input": "base",
        "reads": ["logits_orig"],
    }
    method["train"]["objective"]["drift"] = term(
        "logits", "patched", aggregation("kl", target=target), weight=0.1
    )
    return in_order(raw)


class TestAReferenceIsItsInlineTwin:
    def test_a_train_save_parses_to_the_entry_it_names(self) -> None:
        assert parse_document(by_reference()).save == parse_document(inline()).save

    def test_an_eval_label_and_an_objective_term_each_resolve(self) -> None:
        ld, ce, _rot = parse_document(by_reference()).save
        assert ld.read == ReadRef("logits", "patched")
        assert ld.aggregation is not None and ld.aggregation.kind == "logit_diff"
        assert ld.label == "ld"
        assert ce.aggregation is not None and ce.aggregation.kind == "cross_entropy"

    def test_the_label_is_the_file_stem_not_the_train_name(self) -> None:
        doc = parse_document(fit_doc({"train": "ld", "file_path": "iia.json"}))
        assert doc.save[0].label == "iia"

    def test_a_reference_and_its_inline_twin_are_one_canonical_form(self, env) -> None:
        referenced = canonicalize(by_reference(), env)
        assert canonical_bytes(referenced) == canonical_bytes(
            canonicalize(inline(), env)
        )
        assert referenced["method"]["save"][1] == {
            "read": "logits",
            "model": "patched",
            "aggregation": {"kind": "cross_entropy", "target": "label"},
            "file_path": "ce.json",
        }

    def test_inline_train_saves_spells_each_reference_out(self) -> None:
        method = by_reference()["method"]
        assert inline_train_saves(method) == inline()["method"]["save"]
        assert method["save"][0] == {"train": "ld", "file_path": "ld.json"}  # a copy


class TestRefusals:
    def _refused(self, raw: dict[str, Any], match: str) -> ValidationError:
        with pytest.raises(ValidationError, match=match) as err:
            parse_document(raw)
        return err.value

    def test_a_regularizer_reduces_no_read(self) -> None:
        err = self._refused(
            fit_doc({"train": "shrink", "file_path": "shrink.json"}),
            "'shrink' is a regularizer",
        )
        assert err.rule == 4
        assert err.path == "save[0].train"

    def test_an_unknown_name_names_the_candidates(self) -> None:
        err = self._refused(
            fit_doc({"train": "c", "file_path": "ce.json"}),
            r"names no objective term or eval label.*\['ce', 'ld'\]",
        )
        assert err.rule == 4

    def test_a_close_name_is_suggested(self) -> None:
        self._refused(
            fit_doc({"train": "lld", "file_path": "ld.json"}), "did you mean 'ld'"
        )

    def test_a_name_in_both_namespaces_is_ambiguous(self) -> None:
        raw = fit_doc({"train": "ce", "file_path": "ce.json"})
        raw["method"]["train"]["eval"]["aggregations"]["ce"] = term(
            "logits", "patched", dict(CE)
        )
        err = self._refused(raw, "both an objective term and an eval label")
        assert err.rule == 4

    def test_a_positional_objective_has_no_names(self) -> None:
        raw = fit_doc({"train": "ce", "file_path": "ce.json"})
        raw["method"]["train"]["objective"] = [
            [1.0, term("logits", "patched", dict(CE))]
        ]
        self._refused(raw, "positional objective terms have no name")

    def test_a_document_without_train_has_nothing_to_name(self) -> None:
        raw = fit_doc({"train": "ce", "file_path": "ce.json"})
        del raw["method"]["train"]
        raw["method"]["save"].pop()  # the rotation is no longer trained
        self._refused(raw, "no 'train' section")

    @pytest.mark.parametrize(
        "extra",
        [{"reduce": "mean"}, {"aggregation": dict(CE)}, {"read": "logits"}],
        ids=["reduce", "aggregation", "read"],
    )
    def test_a_train_save_carries_nothing_of_its_own(
        self, extra: dict[str, Any]
    ) -> None:
        (key,) = extra
        raw = fit_doc({"train": "ce", "file_path": "ce.json", **extra})
        with pytest.raises(ParseError, match=f"no {key!r} of its own"):
            parse_document(raw)

    def test_a_train_save_needs_a_file_path(self) -> None:
        with pytest.raises(ParseError, match="needs 'file_path'"):
            parse_document(fit_doc({"train": "ce"}))

    def test_a_swept_term_is_refused(self) -> None:
        # the reference moves with the term, point by point, where the
        # inline copy the canonical form writes would be a second axis
        raw = fit_doc({"train": "ld", "file_path": "ld.json"})
        eval_term = raw["method"]["train"]["eval"]["aggregations"]["ld"]
        eval_term["aggregation"]["a"] = {"sweep": ["cf_answer", "label"]}
        err = self._refused(raw, "carries a sweep")
        assert err.rule == 14

    def test_a_swept_weight_is_not_the_term_the_save_copies(self) -> None:
        raw = fit_doc({"train": "ce", "file_path": "ce.json"})
        raw["method"]["train"]["objective"]["ce"]["weight"] = {"sweep": [0.5, 1.0]}
        assert parse_document(raw).save == parse_document(inline()).save[1:]


class TestReadReferencesInsideSaves:
    def test_a_bare_kl_target_in_a_save_is_refused(self) -> None:
        raw = _kl_doc({"read": "logits_orig", "model": "original_base"})
        raw["method"]["save"].insert(
            0,
            saved(
                "logits", "patched", "kl.json", aggregation("kl", target="logits_orig")
            ),
        )
        with pytest.raises(ParseError, match=r'\{"read", "model"\}') as err:
            parse_document(raw)
        assert err.value.path == "save[0].aggregation.target"

    def test_the_qualified_kl_target_in_a_save_binds(self) -> None:
        qualified = {"read": "logits_orig", "model": "original_base"}
        raw = _kl_doc(qualified)
        raw["method"]["save"].insert(
            0,
            saved("logits", "patched", "kl.json", aggregation("kl", target=qualified)),
        )
        (kl,) = [a for a in parse_document(raw).aggregations() if a.label == "kl"]
        assert kl.target == ReadRef("logits_orig", "original_base")

    def test_the_bare_target_outside_a_save_still_binds(self) -> None:
        doc = parse_document(_kl_doc("logits_orig"))
        (drift,) = [a for a in doc.objective_aggregations() if a.label == "drift"]
        assert drift.target == ReadRef("logits_orig", "original_base")

    def test_a_train_save_copies_no_bare_target(self) -> None:
        raw = _kl_doc("logits_orig")
        raw["method"]["save"].insert(0, {"train": "drift", "file_path": "kl.json"})
        with pytest.raises(ParseError, match="spell the term's target") as err:
            parse_document(raw)
        assert err.value.path == "train.objective.drift.aggregation.target"

    def test_a_train_save_copies_a_qualified_target(self) -> None:
        qualified = {"read": "logits_orig", "model": "original_base"}
        raw = _kl_doc(qualified)
        twin = copy.deepcopy(raw)
        raw["method"]["save"].insert(0, {"train": "drift", "file_path": "kl.json"})
        twin["method"]["save"].insert(
            0,
            saved("logits", "patched", "kl.json", aggregation("kl", target=qualified)),
        )
        assert parse_document(raw).save == parse_document(twin).save

    def test_a_write_operand_keeps_the_bare_name(self) -> None:
        # §2.7: the sugar is scoped to write operands and targets outside saves
        (write,) = parse_document(by_reference()).writes.values()
        assert write.do.payload == ReadRef("v_cf", UNWRITTEN)
