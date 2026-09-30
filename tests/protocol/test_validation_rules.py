"""One failing document per load-error checklist rule (spec §5).

Each test mutates a minimal valid document into exactly one violation and
asserts the loader refuses with that rule's code — the checklist is the
contract, the rule number is the assertion.

Rule 2 is the exception, and reads that way: it recommends a section order
rather than requiring one, so its tests assert a warning and a document that
still loads.
"""

from __future__ import annotations

import copy
import json
import re
import warnings
from pathlib import Path
from typing import Any

import pytest

from causalab.protocol.rules.errors import (
    RULES,
    RULES_BY_NUMBER,
    ParseError,
    ProtocolWarning,
    Rule,
    ValidationError,
    ValidationErrors,
    lookup_rule,
    rule_registry,
)
from causalab.protocol.pipeline import compile_protocol
from causalab.protocol.schema import GATE_MAPS, parse_document
from causalab.neural.shared.sweep import expand
from causalab.protocol.rules.document import validate_document

from tests.protocol._docs import (
    LOGIT_DIFF,
    UNWRITTEN,
    base_doc,
    in_order,
    saved,
    term,
)


pytestmark = pytest.mark.unit


def parse_and_validate(raw: dict[str, Any], **kwargs: Any) -> None:
    validate_document(parse_document(in_order(raw)), **kwargs)


def _ld(**extra: Any) -> dict[str, Any]:
    """``base_doc``'s logit difference as an objective (or eval) term: the
    ``ld`` the ancestor's ``[w, "ld"]`` named, bound where it is consumed
    (§2.11)."""
    return term("logits", "patched", dict(LOGIT_DIFF), **extra)


def _without_the_counterfactual_read(doc: dict[str, Any]) -> dict[str, Any]:
    """Drop ``v_cf`` and, with it, the un-intervened model that existed only
    to take it: a model nobody reads is refused at parse (§2.9), so the two
    go together whenever a test replaces the swap's operand."""
    del doc["method"]["reads"]["v_cf"]
    del doc["method"]["intervened_models"][UNWRITTEN]
    return doc


def expect_rule(rule: int, raw: dict[str, Any], **kwargs: Any) -> ValidationError:
    with pytest.raises(ValidationError) as err:
        parse_and_validate(raw, **kwargs)
    assert err.value.rule == rule, f"expected V{rule}, got {err.value}"
    return err.value


def test_base_document_is_valid():
    parse_and_validate(base_doc())


# rule 1 — strict keys, closed enums, no authored derived fields ------------- #


def test_rule_1_unknown_key_with_suggestion():
    doc = base_doc()
    doc["method"]["sites"]["tgt"]["layer"] = 3  # the protocol_version 2 spelling
    with pytest.raises(ParseError) as err:
        parse_document(doc)
    assert err.value.code == "P3"
    assert "'layers'" in str(err.value)  # the suggestion
    assert "causalab migrate" in str(err.value)  # and the verb that carries it
    doc = base_doc()
    doc["method"]["sites"]["tgt"]["layerz"] = 3
    with pytest.raises(ParseError) as err:
        parse_document(doc)
    assert err.value.code == "P3"
    assert "did you mean 'layers'" in str(err.value)


def test_rule_1_closed_enum_with_suggestion():
    doc = base_doc()
    doc["method"]["sites"]["tgt"]["component"] = "block_out"
    with pytest.raises(ParseError) as err:
        parse_document(doc)
    assert err.value.code == "P4"
    assert "block_output" in str(err.value)


def test_rule_1_derived_field_not_authorable():
    doc = base_doc()
    doc["data"]["base"]["digest"] = "abc"  # stamped at load, never authored
    with pytest.raises(ParseError):
        parse_document(doc)


# rule 2 — section order: a recommendation, and what it recommends ----------- #


def test_rule_2_section_order_warns_and_parses():
    """Order is a reading convention, so the unconventional document loads.

    It used to be refused. Nothing downstream ever consulted the order — the
    canonical form emits §1's order whatever the file's is — so the rejection
    only ever caught documents that were already the same experiment.
    """
    doc = base_doc()
    reordered = {"header": doc["header"], "data": doc["data"], "model": doc["model"]}
    reordered.update({k: v for k, v in doc.items() if k not in reordered})
    with pytest.warns(ProtocolWarning, match="recommended"):
        parsed = parse_document(reordered)
    assert parsed == parse_document(doc)


def test_rule_2_save_not_last_warns_and_parses():
    doc = base_doc()
    save = doc["method"].pop("save")
    doc["method"] = {"save": save, **doc["method"]}
    with pytest.warns(ProtocolWarning, match="recommended"):
        parse_document(doc)


def test_rule_2_the_warning_names_the_recommended_order():
    """A warning that does not say what to do instead is a warning nobody
    acts on."""
    doc = base_doc()
    reordered = {"save": doc["method"]["save"]}
    reordered.update({k: v for k, v in doc["method"].items() if k != "save"})
    doc["method"] = reordered
    with pytest.warns(ProtocolWarning) as caught:
        parse_document(doc)
    message = str(caught[0].message)
    assert "recommended ['intervened_models'" in message
    assert message.rstrip().endswith("(§5 rule 2)")


def test_rule_2_a_conventional_document_is_silent():
    with warnings.catch_warnings():
        warnings.simplefilter("error", ProtocolWarning)
        parse_document(base_doc())


def test_a_missing_save_is_still_refused():
    """The retired half of rule 2 was `save` last; its *presence* was never
    rule 2's to enforce, and still is not."""
    doc = base_doc()
    del doc["method"]["save"]
    with pytest.raises(ParseError, match="save"):
        parse_document(doc)


# rule 3 — one namespace, reserved names -------------------------------------- #


def test_rule_3_duplicate_name_across_sections():
    doc = base_doc()
    doc["method"]["reads"]["tgt"] = {"site": "tgt", "pos": -1}
    doc["method"]["intervened_models"]["original_base"] = {
        "input": "base",
        "reads": ["tgt"],
    }
    doc["method"]["save"].append(saved("tgt", "original_base", "t.safetensors"))
    expect_rule(3, doc)


def test_rule_3_reserved_name():
    doc = base_doc()
    doc["method"]["positions"] = {"base": {"index": -1}}
    doc["method"]["reads"]["v_cf"]["pos"] = "base"
    expect_rule(3, doc)


def test_rule_3_all_is_reserved():
    """A positions entry named ``all`` would shadow the bare-string sugar,
    so the name is reserved outright (§5.3)."""
    doc = base_doc()
    doc["method"]["positions"] = {"all": {"index": -1}}
    expect_rule(3, doc)


# rule 4 — every reference resolves ------------------------------------------- #


def test_rule_4_unknown_site():
    doc = base_doc()
    doc["method"]["writes"]["patch"]["site"] = "nowhere"
    expect_rule(4, doc)


def test_rule_4_metric_on_non_lm_head_read():
    doc = base_doc()
    doc["method"]["save"][0].update(read="v_cf", model=UNWRITTEN)
    expect_rule(4, doc)


@pytest.mark.parametrize(
    "spec",
    [
        {
            "kind": "class_probs",
            "groups": {"days": ["Monday"]},
        },
        {
            "kind": "token_logits",
            "tokens": ["Monday", "Friday"],
        },
        {
            "kind": "token_logit",
            "token": "cf_answer",
        },
        {
            "kind": "cross_entropy",
            "target": "cf_answer",
        },
        {"kind": "match", "expected": "cf_answer"},
    ],
)
def test_rule_4_still_binds_the_token_space_kinds_to_lm_head(spec):
    """``top_k`` was loosened to any read; its siblings were not. Each of
    these resolves an authored string to a token id, which only an
    ``lm_head`` read can be indexed by."""
    doc = base_doc()
    doc["method"]["save"].append(saved("v_cf", UNWRITTEN, "m.json", spec))
    expect_rule(4, doc)


def test_rule_4_top_k_binds_to_a_read_at_any_component():
    """The point of the change: a top-k over a wide read is the reduction that
    keeps the wide tensor off disk, so it must be expressible."""
    doc = base_doc()
    doc["method"]["save"].append(
        saved(
            "v_cf", UNWRITTEN, "tk.json", {"kind": "top_k", "k": 4, "by": "abs_value"}
        )
    )
    parse_and_validate(doc)


def test_rule_4_top_k_by_prob_off_lm_head_is_refused():
    """A softmax across a residual stream normalizes over an axis that is not
    an event space — the resulting numbers are probabilities of nothing."""
    doc = base_doc()
    doc["method"]["save"].append(
        saved("v_cf", UNWRITTEN, "tk.json", {"kind": "top_k", "k": 4, "by": "prob"})
    )
    err = expect_rule(4, doc)
    assert "prob" in str(err)


def test_rule_4_top_k_by_prob_on_lm_head_is_legal():
    doc = base_doc()
    doc["method"]["save"].append(
        saved("logits", "patched", "tk.json", {"kind": "top_k", "k": 4, "by": "prob"})
    )
    parse_and_validate(doc)


def _lm_head_read_through(**extra: object) -> dict[str, Any]:
    """A doc with a second lm_head read that does NOT hand the projection on
    unchanged — the site says vocabulary, the read's value is not."""
    doc = base_doc()
    doc["method"]["reads"]["flogits"] = {"site": "lm_head", "pos": -1, **extra}
    doc["method"]["intervened_models"]["patched"]["reads"].append("flogits")
    return doc


def test_rule_4_top_k_by_prob_over_a_featurized_lm_head_read_is_refused():
    """The site alone cannot answer "is this the vocabulary?": a featurizer
    re-expresses the projection in its own latents, so `prob` there would
    softmax an axis that is not an event space — even though the site says
    lm_head, which is exactly the case a component check waves through."""
    doc = _lm_head_read_through(featurizer="f")
    doc["method"]["featurizers"] = {
        "f": {"kind": "subspace", "k": 4, "parametrization": "cayley"}
    }
    doc["method"]["save"].append(
        saved("flogits", "patched", "tk.json", {"kind": "top_k", "k": 2, "by": "prob"})
    )
    err = expect_rule(4, doc)
    assert "featurizer" in str(err)


def test_rule_4_top_k_by_prob_over_a_dims_sliced_lm_head_read_is_refused():
    """`dims` re-indexes a slice, so entry j is no longer token j — the same
    hole as the featurizer, through the other read transform."""
    doc = _lm_head_read_through(dims=[0, 1, 2])
    doc["method"]["save"].append(
        saved("flogits", "patched", "tk.json", {"kind": "top_k", "k": 2, "by": "prob"})
    )
    err = expect_rule(4, doc)
    assert "dims" in str(err)


def test_rule_4_top_k_by_value_over_a_featurized_lm_head_read_is_legal():
    """Anti-vacuity: the refusal is about `prob`'s softmax, not about
    featurized reads — ranking latents by signed value is the use case
    any-read top_k exists for."""
    doc = _lm_head_read_through(featurizer="f")
    doc["method"]["featurizers"] = {
        "f": {"kind": "subspace", "k": 4, "parametrization": "cayley"}
    }
    doc["method"]["save"].append(
        saved("flogits", "patched", "tk.json", {"kind": "top_k", "k": 2, "by": "value"})
    )
    parse_and_validate(doc)


def test_rule_4_a_token_space_kind_over_a_featurized_lm_head_read_is_refused():
    """`match` resolves an authored string to a token id and indexes the read
    with it — over a featurizer's latents that indexes the wrong axis, so the
    lm_head site is not enough for the token-space kinds either."""
    doc = _lm_head_read_through(featurizer="f")
    doc["method"]["featurizers"] = {
        "f": {"kind": "subspace", "k": 4, "parametrization": "cayley"}
    }
    doc["method"]["save"].append(
        saved(
            "flogits",
            "patched",
            "m.json",
            {"kind": "match", "expected": "cf_answer"},
        )
    )
    err = expect_rule(4, doc)
    assert "featurizer" in str(err)


def _token_logits_over(read: str) -> dict[str, Any]:
    """A save entry tabulating two answer logits of ``read`` on ``patched``."""
    return saved(
        read,
        "patched",
        "a.json",
        {
            "kind": "token_logits",
            "tokens": ["Monday", "Friday"],
        },
    )


def test_rule_4_token_logits_over_a_featurized_lm_head_read_is_refused():
    """``token_logits`` looks like ``top_k`` in what it emits, but it starts
    from authored strings and indexes the read by their ids — which is the
    step a featurizer's latents make meaningless, so it binds like the other
    string-resolving kinds and not like ``top_k``."""
    doc = _lm_head_read_through(featurizer="f")
    doc["method"]["featurizers"] = {
        "f": {"kind": "subspace", "k": 4, "parametrization": "cayley"}
    }
    doc["method"]["save"].append(_token_logits_over("flogits"))
    err = expect_rule(4, doc)
    assert "featurizer" in str(err)


def test_rule_4_token_logits_over_a_dims_sliced_lm_head_read_is_refused():
    doc = _lm_head_read_through(dims=[0, 1, 2])
    doc["method"]["save"].append(_token_logits_over("flogits"))
    err = expect_rule(4, doc)
    assert "dims" in str(err)


def test_rule_4_token_logits_over_a_plain_lm_head_read_is_legal():
    """Anti-vacuity for the two refusals above."""
    doc = base_doc()
    doc["method"]["save"].append(_token_logits_over("logits"))
    parse_and_validate(doc)


# rule 5 — read bindings ------------------------------------------------------ #


def test_rule_5_save_entry_names_a_model_that_does_not_list_the_read():
    """A save entry's ``{"read", "model"}`` is a binding, and the model it
    names has to take the read (§2.9)."""
    doc = base_doc()
    doc["method"]["save"][0]["model"] = "original_counterfactual"
    err = expect_rule(5, doc)
    assert "'logits'" in str(err) and "'original_counterfactual'" in str(err)


def test_rule_5_read_model_undeclared():
    doc = base_doc()
    doc["method"]["save"][0]["model"] = "ghost"
    err = expect_rule(5, doc)
    assert "'ghost'" in str(err)


# rule 6 — operands are reads, params, or literal scalars --------------------- #


def test_rule_6_operand_names_a_site():
    doc = base_doc()
    doc["method"]["writes"]["patch"]["do"] = {"swap": "tgt"}
    err = expect_rule(6, doc)
    assert "names a sites entry" in str(err)


# rule 7 — membership + acyclicity -------------------------------------------- #


def test_rule_7_write_in_no_im():
    doc = base_doc()
    doc["method"]["writes"]["orphan"] = {
        "site": "tgt",
        "pos": -1,
        "do": {"swap": "v_cf"},
    }
    expect_rule(7, doc)


def test_rule_7_model_graph_cycle():
    doc = base_doc()
    del doc["method"]["reads"]["v_cf"]
    doc["method"]["reads"]["r_a"] = {"site": "tgt", "pos": -1}
    doc["method"]["reads"]["r_b"] = {"site": "tgt", "pos": -1}
    doc["method"]["writes"] = {
        "e_a": {"site": "tgt", "pos": -1, "do": {"swap": "r_b"}},
        "e_b": {"site": "tgt", "pos": -1, "do": {"swap": "r_a"}},
    }
    doc["method"]["intervened_models"] = {
        "im_a": {"input": "base", "reads": ["r_a", "logits"], "writes": ["e_a"]},
        "im_b": {"input": "base", "reads": ["r_b"], "writes": ["e_b"]},
    }
    doc["method"]["save"][0]["model"] = "im_a"
    expect_rule(7, doc)


# rule 8 — one absolute write per address --------------------------------------- #


def test_rule_8_two_absolute_writes_same_address():
    doc = base_doc()
    doc["method"]["writes"]["patch2"] = {
        "site": "tgt",
        "pos": -1,
        "do": {"swap": "v_cf"},
    }
    doc["method"]["intervened_models"]["patched"]["writes"].append("patch2")
    expect_rule(8, doc)


def test_rule_8_all_positions_overlaps_everything():
    """An all-positions write covers every token, so it is never provably
    disjoint from another write at the same site — including one pinned to a
    single index."""
    doc = base_doc()
    doc["method"]["writes"]["patch_all"] = {
        "site": "tgt",
        "pos": {"all": True},
        "do": {"swap": "v_cf"},
    }
    doc["method"]["intervened_models"]["patched"]["writes"].append("patch_all")
    expect_rule(8, doc)


def test_rule_8_all_positions_overlaps_itself():
    doc = base_doc()
    doc["method"]["writes"]["patch"]["pos"] = "all"
    doc["method"]["writes"]["patch_all"] = {
        "site": "tgt",
        "pos": "all",
        "do": {"swap": "v_cf"},
    }
    doc["method"]["intervened_models"]["patched"]["writes"].append("patch_all")
    expect_rule(8, doc)


def test_rule_8_all_positions_absolute_plus_additive_composes():
    """The mechanism-class order (§2.8) still applies at an all address —
    overlap only forbids a *second absolute* write."""
    doc = base_doc()
    doc["method"]["writes"]["patch"]["pos"] = "all"
    doc["method"]["writes"]["nudge"] = {
        "site": "tgt",
        "pos": "all",
        "do": {"add_scaled": {"op": "v_cf", "alpha": 0.5}},
    }
    doc["method"]["intervened_models"]["patched"]["writes"].append("nudge")
    parse_and_validate(doc)


def test_rule_9_all_positions_dims_must_be_disjoint():
    """Rule 9 composes with the all spelling exactly as it does elsewhere."""
    doc = base_doc()
    doc["method"]["writes"]["patch"]["dims"] = [0, 1]
    doc["method"]["writes"]["patch_all"] = {
        "site": "tgt",
        "pos": "all",
        "dims": [1, 2],
        "do": {"swap": "v_cf"},
    }
    doc["method"]["intervened_models"]["patched"]["writes"].append("patch_all")
    expect_rule(9, doc)


def test_rule_8_additive_write_composes():
    doc = base_doc()
    doc["method"]["writes"]["nudge"] = {
        "site": "tgt",
        "pos": -1,
        "do": {"add_scaled": {"op": "v_cf", "alpha": 0.5}},
    }
    doc["method"]["intervened_models"]["patched"]["writes"].append("nudge")
    parse_and_validate(doc)  # absolute + additive at one address is legal (§2.8)


# rule 9 — dims disjointness ---------------------------------------------------- #


def test_rule_9_intersecting_dims_absolutes():
    doc = base_doc()
    doc["method"]["writes"]["patch"]["dims"] = [0, 1]
    doc["method"]["writes"]["patch2"] = {
        "site": "tgt",
        "pos": -1,
        "dims": [1, 2],
        "do": {"swap": "v_cf"},
    }
    doc["method"]["intervened_models"]["patched"]["writes"].append("patch2")
    expect_rule(9, doc)


def test_rule_9_disjoint_dims_absolutes_are_legal():
    doc = base_doc()
    doc["method"]["writes"]["patch"]["dims"] = [0, 1]
    doc["method"]["writes"]["patch2"] = {
        "site": "tgt",
        "pos": -1,
        "dims": [2, 3],
        "do": {"swap": "v_cf"},
    }
    doc["method"]["intervened_models"]["patched"]["writes"].append("patch2")
    parse_and_validate(doc)


# rule 10 — the save manifest --------------------------------------------------- #


def test_rule_10_a_raw_read_alone_is_a_complete_manifest():
    """An aggregation lives on the entry that consumes it (§2.10), so a
    manifest that saves the bound read plain and aggregates nothing has
    nothing left unsaved."""
    doc = base_doc()
    doc["method"]["save"] = [
        {"read": "logits", "model": "patched", "file_path": "l.safetensors"}
    ]
    parse_and_validate(doc)


def test_rule_10_untrained_featurizer_not_saveable():
    doc = base_doc()
    doc["method"]["featurizers"] = {
        "rot": {"kind": "subspace", "k": 4, "parametrization": "cayley"}
    }
    doc["method"]["reads"]["v_cf"]["featurizer"] = "rot"
    doc["method"]["writes"]["patch"]["featurizer"] = "rot"
    doc["method"]["save"].append(
        {"value": "rot", "site": "tgt", "file_path": "rot.safetensors"}
    )
    expect_rule(10, doc)


def test_rule_10_reduce_on_a_metric_refused():
    """§2.12: a metric is already a reduction over its read."""
    doc = base_doc()
    doc["method"]["save"][0]["reduce"] = "mean"
    expect_rule(10, doc)


def test_rule_10_reduce_on_a_featurizer_bundle_refused():
    doc = base_doc()
    doc["method"]["featurizers"] = {
        "rot": {"kind": "subspace", "k": 4, "parametrization": "cayley"}
    }
    doc["method"]["reads"]["v_cf"]["featurizer"] = "rot"
    doc["method"]["writes"]["patch"]["featurizer"] = "rot"
    doc["method"]["train"] = {
        "objective": [[1.0, _ld()]],
        "params": ["rot"],
        "optimizer": {"name": "adam", "lr": 0.001},
        "steps": {"epochs": 1},
        "batch": {"pairs": 2},
    }
    doc["method"]["save"].append(
        {
            "value": "rot",
            "site": "tgt",
            "file_path": "rot.safetensors",
            "reduce": "mean",
        }
    )
    # a featurizer bundle's entry has no `reduce` field at all: refused as an
    # unknown key at parse (§2.12: `reduce` applies to read entries)
    with pytest.raises(ParseError) as err:
        parse_document(in_order(doc))
    assert err.value.code == "P3" and "reduce" in str(err.value)


def test_a_reduced_read_is_a_valid_save():
    doc = base_doc()
    doc["method"]["save"].append(
        saved("v_cf", UNWRITTEN, "mean.safetensors", reduce="mean")
    )
    parse_and_validate(doc)


def test_an_entry_selector_needs_a_file_path():
    """'entry' selects *inside* a loaded bundle — there is nothing to
    select in a featurizer that is fitted rather than loaded."""
    doc = base_doc()
    doc["method"]["featurizers"] = {
        "rot": {
            "kind": "subspace",
            "k": 4,
            "parametrization": "cayley",
            "entry": {"k": 4},
        }
    }
    doc["method"]["reads"]["v_cf"]["featurizer"] = "rot"
    with pytest.raises(ParseError):
        parse_and_validate(doc)


def test_a_featurizer_may_not_rename_its_slots():
    """A featurizer bundle's slots come from its kind; only a params
    constant may name the tensor it wants."""
    doc = base_doc()
    doc["method"]["featurizers"] = {
        "rot": {
            "kind": "subspace",
            "k": 4,
            "parametrization": "cayley",
            "file_path": "rot.safetensors",
            "entry": {"slot": "acts"},
        }
    }
    doc["method"]["reads"]["v_cf"]["featurizer"] = "rot"
    with pytest.raises(ParseError):
        parse_and_validate(doc)


# rule 11 — sinks ---------------------------------------------------------------- #


def test_rule_11_dead_read():
    doc = base_doc()
    doc["method"]["reads"]["extra"] = {"site": "tgt", "pos": -1}
    doc["method"]["intervened_models"]["original_base"] = {
        "input": "base",
        "reads": ["extra"],
    }
    expect_rule(11, doc)


def test_rule_11_dead_site():
    doc = base_doc()
    doc["method"]["sites"]["spare"] = {"component": "block_output", "layers": [1]}
    expect_rule(11, doc)


# rule 12 — featurizer legality: trainability, and composition -------------------- #


def test_rule_12_loaded_featurizer_trained():
    doc = base_doc()
    doc["method"]["featurizers"] = {
        "rot": {
            "kind": "subspace",
            "k": 4,
            "parametrization": "cayley",
            "file_path": "rot.safetensors",
        }
    }
    doc["method"]["reads"]["v_cf"]["featurizer"] = "rot"
    doc["method"]["writes"]["patch"]["featurizer"] = "rot"
    ce = {"kind": "cross_entropy", "target": "label"}
    doc["method"]["train"] = {
        "objective": [[1.0, term("logits", "patched", ce)]],
        "params": ["rot"],
        "optimizer": {"name": "adamw", "lr": 1e-3},
        "steps": {"epochs": 1},
        "batch": {"pairs": 2},
    }
    doc["method"]["save"].append(saved("logits", "patched", "ce.json", dict(ce)))
    doc["method"]["save"].append(
        {"value": "rot", "site": "tgt", "file_path": "rot.safetensors"}
    )
    expect_rule(12, doc)


#: Two declarable kinds, so a composition test has two distinct stages.
_KINDS: dict[str, dict[str, Any]] = {
    "rot": {"kind": "subspace", "k": 4, "parametrization": "cayley"},
    "gate": {"kind": "gate"},
}


def _with_chain(chain: Any, *, on: str = "reads") -> dict[str, Any]:
    """A document whose ``v_cf`` read (or ``patch`` write) composes ``chain``.

    Only the featurizers the chain names are declared: an unused declaration
    is rule 11, and a test for rule 12 that trips rule 11 instead is a test of
    nothing.
    """
    doc = base_doc()
    named = [chain] if isinstance(chain, str) else list(chain)
    doc["method"]["featurizers"] = {
        name: _KINDS[name] for name in dict.fromkeys(named) if name in _KINDS
    }
    if on == "reads":
        doc["method"]["reads"]["v_cf"]["featurizer"] = chain
    else:
        doc["method"]["writes"]["patch"]["featurizer"] = chain
    return doc


def test_rule_12_an_empty_composition_is_refused():
    """`[]` re-expresses nothing, and the canonical form would not record it.

    ``identity`` is a declarable kind, so a document that means "no
    featurizer" already has two honest spellings — omit the key, or name one.
    A third that silently means the same thing is the sort of implicit default
    this layer refuses everywhere else.
    """
    err = expect_rule(12, _with_chain([]))
    assert err.path == "reads.v_cf.featurizer"
    assert "identity" in str(err)


def test_rule_12_a_repeated_stage_is_refused():
    """A stage's width comes from its position in the chain, so a name in two
    positions has one derived width and two input widths — and both stages
    would share one parameter slot."""
    err = expect_rule(12, _with_chain(["rot", "rot"]))
    assert err.path == "reads.v_cf.featurizer"
    assert "'rot'" in str(err) and "position" in str(err)


def test_rule_12_a_repeat_is_refused_on_a_write_too():
    """Writes carry the same reference and the same width derivation."""
    err = expect_rule(12, _with_chain(["gate", "gate"], on="writes"))
    assert err.path == "writes.patch.featurizer"


def test_rule_12_a_repeat_is_found_among_distinct_stages():
    """Not just a two-element chain: the check is over the whole composition."""
    err = expect_rule(12, _with_chain(["rot", "gate", "rot"]))
    assert "repeats ['rot']" in str(err), (
        "the message should name the whole chain and, separately, only the "
        f"stage that repeats — got {err}"
    )


def test_rule_12_a_legal_composition_still_loads():
    """The converse — distinct stages left-to-right are what §2.5 is *for*."""
    parse_and_validate(_with_chain(["rot", "gate"]))


def test_rule_12_a_single_name_is_still_a_legal_chain():
    parse_and_validate(_with_chain("rot"))


def test_rule_4_still_catches_an_undeclared_stage():
    """The stage-resolution half is unchanged, and keeps its own rule: a
    missing declaration is a reference that does not resolve, not an illegal
    composition."""
    err = expect_rule(4, _with_chain(["rot", "nope"]))
    assert "'nope'" in str(err)


# rule 13 — pytorch_fn is local-only ---------------------------------------------- #


def test_rule_13_pytorch_fn_on_non_local_engine():
    doc = base_doc()
    doc["method"]["code"] = {
        "relu": {"locator": "tests.protocol._code_under_test.scale"}
    }
    doc["method"]["writes"]["patch"]["do"] = {"pytorch_fn": {"code": "relu"}}
    # no longer an operand; would trip the sink rule
    _without_the_counterfactual_read(doc)
    expect_rule(13, doc, engine_is_local=False)
    parse_and_validate(doc, engine_is_local=True)  # a local engine may run it


# rule 24 — a code declaration agrees with the source it names -------------------- #
# rule 25 — declared row roles match the resolved data ---------------------------- #
# Both live in tests/protocol/test_code_identity.py, where the fixture modules
# they need are written and mutated; these two lines are the census's pointer.


def test_rule_24_locator_naming_no_python_source():
    doc = base_doc()
    doc["method"]["code"] = {"corrupt": {"locator": "no_such_package_anywhere.corrupt"}}
    doc["method"]["writes"]["patch"]["do"] = {"pytorch_fn": {"code": "corrupt"}}
    _without_the_counterfactual_read(doc)
    expect_rule(24, doc)


def test_rule_25_row_roles_need_the_resolved_data():
    """Rule 25 is a ``validate --data`` rule (like rule 20): the bare load
    cannot see the tables, so it must *not* refuse here."""
    doc = base_doc()
    doc["method"]["code"] = {
        "corrupt": {
            "locator": "tests.protocol._code_under_test.corrupt",
            "row_roles": [
                {"role": "clean", "rows": 1},
                {"role": "corrupted", "rows": 10},
            ],
        }
    }
    doc["method"]["writes"]["patch"]["do"] = {"pytorch_fn": {"code": "corrupt"}}
    _without_the_counterfactual_read(doc)
    parse_and_validate(doc)


# rule 14 — sweep wrappers + point cap --------------------------------------------- #


def test_rule_14_malformed_sweep():
    doc = base_doc()
    doc["method"]["sites"]["tgt"]["layers"] = {"sweep": {"start": 0}}
    with pytest.raises(ValidationError) as err:
        expand(doc)
    assert err.value.rule == 14


def test_rule_14_point_cap():
    doc = base_doc()
    doc["method"]["sites"]["tgt"]["layers"] = {"sweep": {"range": [0, 100]}}
    doc["method"]["reads"]["v_cf"]["dims"] = {"sweep": {"range": [0, 100]}}
    with pytest.raises(ValidationError) as err:
        expand(doc, point_cap=4096)
    assert err.value.rule == 14


# rule 15 — artifact-valued fields resolve ------------------------------------------ #


def test_rule_15_missing_artifact(env):
    doc = base_doc()
    doc["method"]["sites"]["tgt"]["layers"] = {
        "artifact": "nowhere/locate",
        "key": "best_layer",
    }
    with pytest.raises(ValidationError) as err:
        compile_protocol(in_order(doc), env=env)
    assert err.value.rule == 15


def test_rule_15_artifact_identity_mismatch(env):
    doc = base_doc()
    doc["model"] = {"key": "Qwen/Qwen3-8B", "revision": "main"}
    doc["method"]["sites"]["tgt"] = {"component": "block_output", "layers": [18]}
    doc["method"]["featurizers"] = {
        "rot": {
            "kind": "subspace",
            "k": 16,  # the fixture bundle was fitted with k=8
            "parametrization": "cayley",
            "file_path": "artifacts/weekdays/qwen3_8b/subspace/rot_k8.safetensors",
        }
    }
    doc["method"]["reads"]["v_cf"]["featurizer"] = "rot"
    doc["method"]["writes"]["patch"]["featurizer"] = "rot"
    with pytest.raises(ValidationError) as err:
        compile_protocol(in_order(doc), env=env)
    assert err.value.rule == 15
    assert "ArtifactIdentity" in str(err.value)


def test_rule_15_artifact_identity_match_passes(env):
    doc = copy.deepcopy(base_doc())
    # the fixture bundle is stamped bf16, as corpus 09 declares
    doc["model"] = {
        "key": "Qwen/Qwen3-8B",
        "revision": "main",
        "dtype": "bf16",
    }
    doc["method"]["sites"]["tgt"] = {"component": "block_output", "layers": [18]}
    doc["method"]["featurizers"] = {
        "rot": {
            "kind": "subspace",
            "k": 8,
            "parametrization": "cayley",
            "file_path": "artifacts/weekdays/qwen3_8b/subspace/rot_k8.safetensors",
        }
    }
    doc["method"]["reads"]["v_cf"]["featurizer"] = "rot"
    doc["method"]["writes"]["patch"]["featurizer"] = "rot"
    compile_protocol(in_order(doc), env=env)


def _write_gate_bundle(root, rel: str, *, theta_len: int, **identity) -> None:
    """A hand-written fitted-gate bundle: a stamped header and a zero
    ``theta`` of ``theta_len`` fp32 entries, no tensor library involved."""
    import json
    import struct

    from causalab.io.env import build_artifact_identity

    target = root / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    stamp = build_artifact_identity(
        model_key="Qwen/Qwen3.6-35B-A3B",
        model_revision="main",
        model_dtype="bf16",
        dtype="fp32",
        trained_on="weekdays/train",
        engine="pytorch_hooks",
        commit="fixture",
        **identity,
    )
    header = {
        "__metadata__": stamp,
        "theta": {
            "dtype": "F32",
            "shape": [theta_len],
            "data_offsets": [0, 4 * theta_len],
        },
    }
    header_bytes = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    with target.open("wb") as fh:
        fh.write(struct.pack("<Q", len(header_bytes)))
        fh.write(header_bytes)
        fh.write(bytes(4 * theta_len))


def _head_gate_apply_doc(component: str, layer: int, *, group: str | None) -> dict:
    doc = base_doc()
    doc["model"] = {"key": "Qwen/Qwen3.6-35B-A3B", "revision": "main", "dtype": "bf16"}
    doc["method"]["sites"]["tgt"] = {"component": component, "layers": [layer]}
    gate: dict = {"kind": "gate", "file_path": "fit/gate.safetensors"}
    if group is not None:
        gate["group"] = group
    doc["method"]["featurizers"] = {"gate": gate}
    doc["method"]["reads"]["v_cf"]["featurizer"] = "gate"
    doc["method"]["writes"]["patch"]["featurizer"] = "gate"
    return in_order(doc)


@pytest.fixture()
def head_gate_env(tmp_path):
    """An environment holding a head-grouped gate fitted at ``attention_premix``
    layer 19 of Qwen3.6 — 16 query heads of 256 — under ``fit/gate.safetensors``."""
    from tests.protocol._env import build_env

    _write_gate_bundle(
        tmp_path,
        "fit/gate.safetensors",
        theta_len=16,
        site={"component": "attention_premix", "layers": [19]},
        group="head",
        group_map=[16, 256],
    )
    return tmp_path, build_env(tmp_path)


def test_rule_15_a_head_grouped_gate_reloads_at_its_own_address(head_gate_env):
    _, env = head_gate_env
    compile_protocol(
        _head_gate_apply_doc("attention_premix", 19, group="head"), env=env
    )


def test_rule_15_a_head_grouped_gate_refuses_another_component(head_gate_env):
    """The fixture's two head-major components have different head layouts;
    the site is what names the address, and it is the first thing that
    disagrees."""
    _, env = head_gate_env
    with pytest.raises(ValidationError) as err:
        compile_protocol(
            _head_gate_apply_doc("delta_premix", 18, group="head"), env=env
        )
    assert err.value.rule == 15
    assert "ArtifactIdentity mismatch on 'site'" in str(err.value)


def test_rule_15_a_head_grouped_gate_refuses_another_group_map(tmp_path):
    """Same site, same group kind, a different head layout stamped — only a
    hand-made or foreign bundle can get here, and it is refused by the map, not
    by a width mismatch somewhere in the build."""
    from tests.protocol._env import build_env

    _write_gate_bundle(
        tmp_path,
        "fit/gate.safetensors",
        theta_len=16,
        site={"component": "attention_premix", "layers": [19]},
        group="head",
        group_map=[8, 512],
    )
    with pytest.raises(ValidationError) as err:
        compile_protocol(
            _head_gate_apply_doc("attention_premix", 19, group="head"),
            env=build_env(tmp_path),
        )
    assert err.value.rule == 15
    assert "ArtifactIdentity mismatch on 'group_map'" in str(err.value)
    assert "[16, 256]" in str(err.value) and "[8, 512]" in str(err.value)


def test_rule_15_a_head_grouped_gate_refuses_a_per_coordinate_document(head_gate_env):
    """A document that declares no group would read 16 head parameters as 16
    coordinates of a 4096-wide site; the refusal says which it was fitted as."""
    _, env = head_gate_env
    with pytest.raises(ValidationError) as err:
        compile_protocol(
            _head_gate_apply_doc("attention_premix", 19, group=None), env=env
        )
    assert err.value.rule == 15
    assert "fitted with group 'head'" in str(err.value)


def test_rule_15_a_grouped_document_refuses_a_per_coordinate_bundle(tmp_path):
    from tests.protocol._env import build_env

    _write_gate_bundle(
        tmp_path,
        "fit/gate.safetensors",
        theta_len=4096,
        site={"component": "attention_premix", "layers": [19]},
    )
    with pytest.raises(ValidationError) as err:
        compile_protocol(
            _head_gate_apply_doc("attention_premix", 19, group="head"),
            env=build_env(tmp_path),
        )
    assert err.value.rule == 15
    assert "missing 'group'" in str(err.value)


# §2.3 column positions / §2.10 match modes ---------------------------------- #


def test_position_needs_exactly_one_anchor_form():
    doc = base_doc()
    doc["method"]["positions"] = {"p": {"variable": "entity", "column": "entity"}}
    with pytest.raises(ParseError):
        parse_and_validate(doc)


def test_position_column_and_scope_are_exclusive():
    doc = base_doc()
    doc["method"]["positions"] = {"p": {"column": "entity", "scope": {"variable": "x"}}}
    with pytest.raises(ParseError):
        parse_and_validate(doc)


def test_anchor_ref_takes_one_of_variable_or_column():
    doc = base_doc()
    doc["method"]["positions"] = {"p": {"index": 1, "relative_to": {"nope": "x"}}}
    with pytest.raises(ParseError):
        parse_and_validate(doc)


def test_unknown_match_mode_is_a_closed_enum_error():
    doc = base_doc()
    doc["method"]["save"].append(
        saved(
            "logits",
            "patched",
            "m.json",
            {
                "kind": "match",
                "expected": "label",
                # the task-side spelling; the aggregation's is first_token
                "mode": "prefix",
            },
        )
    )
    with pytest.raises(ParseError) as err:
        parse_and_validate(doc)
    assert "first_token" in str(err.value)  # the suggestion names the real mode


def test_rule_8_column_positions_are_conservatively_overlapping():
    """Two writes at one address whose positions come from *different* columns
    could hit the same token — a column holds data, not a template slot — so
    the absolute-write rule refuses rather than assuming disjointness."""
    doc = base_doc()
    doc["method"]["positions"] = {"a": {"column": "entity"}, "b": {"column": "number"}}
    doc["method"]["reads"]["v2"] = {"site": "tgt", "pos": "a"}
    doc["method"]["intervened_models"][UNWRITTEN]["reads"] = ["v_cf", "v2"]
    doc["method"]["writes"] = {
        "patch": {"site": "tgt", "pos": "a", "do": {"swap": "v_cf"}},
        "patch2": {"site": "tgt", "pos": "b", "do": {"swap": "v2"}},
    }
    doc["method"]["intervened_models"]["patched"]["writes"] = [
        "patch",
        "patch2",
    ]
    expect_rule(8, doc)


# rule 16 — generation is read-only and prefill-only ------------------------- #


def _reads_the_continuation(doc: dict[str, Any]) -> dict[str, Any]:
    """Point the saved read at the continuation, which is legal."""
    doc["method"]["positions"] = {
        "tail": {"generated": {"max_new_tokens": 8}, "index": -1}
    }
    doc["method"]["reads"]["logits"]["pos"] = "tail"
    return doc


def test_a_read_may_address_the_continuation():
    """The positive control for rule 16: reads are exactly what the frame is
    for, so nothing about a generate read is a load error."""
    parse_and_validate(_reads_the_continuation(base_doc()))


def test_rule_16_write_at_a_generated_position():
    doc = _reads_the_continuation(base_doc())
    doc["method"]["writes"]["patch"]["pos"] = "tail"
    err = expect_rule(16, doc)
    assert "addressed in the prompt frame" in str(err)


def test_rule_16_write_at_an_inline_generated_position():
    """Inline specs are refused on the same footing as named ones — the rule
    reads the resolved spec, not the spelling."""
    doc = _reads_the_continuation(base_doc())
    doc["method"]["writes"]["patch"]["pos"] = {
        "generated": {"max_new_tokens": 4},
        "index": -1,
    }
    expect_rule(16, doc)


def _writes_during_generation(doc: dict[str, Any]) -> dict[str, Any]:
    """The positive shape for §2.9's flag: the patched model is decoded, its
    one write sits at ``pos: -1`` with a literal operand, and the read the
    swap used to consume is gone (rule 11 would otherwise refuse the
    now-unconsumed read)."""
    doc = _reads_the_continuation(doc)
    doc["method"]["writes"]["patch"]["do"] = {"swap": 0.0}
    _without_the_counterfactual_read(doc)
    doc["method"]["intervened_models"]["patched"]["writes_during_generation"] = True
    return doc


def test_writes_during_generation_is_legal_at_the_decode_step_forms():
    """The positive control for the flag: a decoded model whose writes sit at
    ``{"index": -1}`` (and, below, at ``all``) with literal operands."""
    parse_and_validate(_writes_during_generation(base_doc()))
    doc = _writes_during_generation(base_doc())
    doc["method"]["writes"]["patch"]["pos"] = "all"
    parse_and_validate(doc)


def test_writes_during_generation_false_is_the_absent_field():
    doc = base_doc()
    doc["method"]["intervened_models"]["patched"]["writes_during_generation"] = False
    parse_and_validate(doc)


def test_writes_during_generation_is_a_bool():
    doc = base_doc()
    doc["method"]["intervened_models"]["patched"]["writes_during_generation"] = "yes"
    with pytest.raises(ParseError):
        parse_document(in_order(doc))


def test_rule_16_writes_during_generation_needs_a_decoded_model():
    """A field that governs nothing may not be declared: no read decodes the
    model, so there is no step for the writes to fire in."""
    doc = base_doc()
    doc["method"]["writes"]["patch"]["do"] = {"swap": 0.0}
    _without_the_counterfactual_read(doc)
    doc["method"]["intervened_models"]["patched"]["writes_during_generation"] = True
    err = expect_rule(16, doc)
    assert "no read decodes it" in str(err)


def test_rule_16_writes_during_generation_refuses_a_prompt_position():
    """``{"index": 2}`` names a prompt token; a decode step has none."""
    doc = _writes_during_generation(base_doc())
    doc["method"]["writes"]["patch"]["pos"] = {"index": 2}
    err = expect_rule(16, doc)
    assert "one token per row" in str(err)


def test_rule_16_writes_during_generation_refuses_a_read_operand():
    doc = _reads_the_continuation(base_doc())
    doc["method"]["intervened_models"]["patched"]["writes_during_generation"] = True
    err = expect_rule(16, doc)
    assert "as an operand" in str(err)


def test_rule_16_writes_during_generation_refuses_gaussian():
    doc = _writes_during_generation(base_doc())
    doc["method"]["writes"]["patch"]["do"] = {
        "gaussian": {"seed": 0, "scale": 0.1, "axis": "tp_duplicated"}
    }
    err = expect_rule(16, doc)
    assert "gaussian" in str(err)


def test_rule_16_train_with_a_generated_position():
    """A greedy decode is an argmax chain: there is no gradient path from a
    continuation read back to a featurizer's parameters."""
    doc = _reads_the_continuation(base_doc())
    doc["method"]["featurizers"] = {"rot": {"kind": "subspace", "k": 2}}
    doc["method"]["reads"]["v_cf"]["featurizer"] = "rot"
    doc["method"]["train"] = {
        "objective": [[1.0, _ld()]],
        "params": ["rot"],
        "optimizer": {"name": "adamw", "lr": 0.001},
        "steps": {"epochs": 1},
        "batch": {"pairs": 2},
    }
    doc["method"]["save"].append(
        {"value": "rot", "site": "tgt", "file_path": "rot.safetensors"}
    )
    err = expect_rule(16, doc)
    assert "gradient path" in str(err)


# the checklist census — code and spec name the same rules ------------------- #

SPEC = Path(__file__).resolve().parents[2] / "docs" / "intervention_protocol.md"
TESTS = Path(__file__).resolve().parents[1]
#: A numbered rule with its slug and title.
SPEC_ITEM = re.compile(r"^(\d+)\. \*\*([a-z][a-z0-9_]*)\*\*\s*[:;]\s*(.*)$", re.M)
#: the fixed spelling of an entry in ``RULES`` — so a slug appears in one
#: shape here, in the spec and in the code (a stricter check than "no spaces")
SLUG = re.compile(r"^[a-z][a-z0-9_]*$")


def _spec_items() -> list[tuple[int, str, str]]:
    """§5's ``(number, slug, opening text)`` items; refuses an empty read."""
    section = SPEC.read_text().split("## 5. Validation")[1].split("\n## ")[0]
    # an item wraps at 80 columns; its continuation lines are indented and are
    # not sub-bullets, so join them back onto the head line before matching
    # (the lookahead sits before the indent: after it, backtracking on the
    # indent would step it past the `- ` it is meant to see)
    section = re.sub(r"\n(?![ \t]*- )[ \t]+", " ", section)
    items = [(int(n), slug, text) for n, slug, text in SPEC_ITEM.findall(section)]
    assert items, "no §5 items parsed — the item spelling changed, not the rules"
    return items


def test_the_checklist_and_the_spec_agree_on_every_rule():
    """`RULES` is what every ValidationError is checked against, so it has to
    match §5 item for item — not merely in count, which is the guard this one
    replaces: a count agrees when a rule was renamed in one place, when two
    rules swapped numbers, and when a duplicate slipped in beside a deletion.
    Identity of ``(number, slug)`` pairs catches all three."""
    items = _spec_items()
    spec_pairs = {(n, slug) for n, slug, _ in items}
    code_pairs = {(rule.number, rule.slug) for rule in RULES.values()}
    assert spec_pairs == code_pairs, {
        "spec only": spec_pairs - code_pairs,
        "code only": code_pairs - spec_pairs,
    }
    # each title is how the spec opens that item — the same words in both
    for n, _slug, text in items:
        assert text.startswith(RULES_BY_NUMBER[n].title), (n, text)
    # a census against an empty table passes for the wrong reason
    assert len(RULES) >= 22
    # ids are unique in both spellings, in code and in the spec
    slugs = [rule.slug for rule in RULES.values()]
    numbers = [rule.number for rule in RULES.values()]
    assert len(set(slugs)) == len(slugs) == len({slug for _, slug, _ in items})
    assert len(set(numbers)) == len(numbers) == len({n for n, _, _ in items})
    assert all(SLUG.match(slug) for slug in slugs), slugs
    # the dict is keyed by the slug it holds, in spec order
    assert list(RULES) == slugs == [slug for _, slug, _ in items]


def _rules_named_by_tests() -> tuple[set[int], set[str]]:
    """Every rule number and slug some test under ``tests/`` asserts.

    Four spellings count: ``expect_rule(<n>|"<slug>"``, ``.rule == <n>``,
    ``.rule_id == "<slug>"`` and a ``def test_rule_<n>_`` name. The workflow
    checklist (``W<n>``, ``test_workflow.py``) shares the last spelling and is
    a different family, so that file is left out.
    """
    numbers: set[int] = set()
    slugs: set[str] = set()
    for path in TESTS.rglob("test_*.py"):
        if path.name == "test_workflow.py":
            continue
        text = path.read_text()
        numbers.update(int(n) for n in re.findall(r"expect_rule\((\d+)\b", text))
        slugs.update(re.findall(r"expect_rule\(\"([a-z0-9_]+)\"", text))
        numbers.update(int(n) for n in re.findall(r"\.rule == (\d+)\b", text))
        slugs.update(re.findall(r"\.rule_id == \"([a-z0-9_]+)\"", text))
        numbers.update(
            int(n) for n in re.findall(r"^def test_rule_(\d+)_", text, flags=re.M)
        )
    return numbers, slugs


def test_every_rule_has_a_test_that_names_it():
    """The checklist's contract is one failing document per rule, asserted by
    its id. A rule no test names is a rule the suite cannot tell from its
    neighbours — the drift the old count-only guard let through."""
    numbers, slugs = _rules_named_by_tests()
    untested = [
        rule.slug
        for rule in RULES.values()
        if rule.number not in numbers and rule.slug not in slugs
    ]
    assert not untested, f"no test asserts these rules: {untested}"


def test_rule_ids_are_names_not_positions():
    """Two PRs each appending "the next" rule collide on its number; the
    registry refuses that at import instead of letting both render as one
    code. And a rule's number is frozen: dropping a neighbour moves nothing."""
    a, b, c = Rule("a", 1, "A"), Rule("b", 2, "B"), Rule("c", 3, "C")
    with pytest.raises(
        ValueError, match="number 1 is claimed by both 'a' and 'also_1'"
    ):
        rule_registry([a, Rule("also_1", 1, "A again")])
    with pytest.raises(ValueError, match="slug 'a' is declared twice"):
        rule_registry([a, Rule("a", 9, "A again")])
    full = rule_registry([a, b, c])
    assert list(full) == ["a", "b", "c"] and full["c"].number == 3
    without_b = rule_registry([a, c])
    assert without_b["c"].number == 3 and without_b["c"].code == "V3"
    assert without_b["a"] == full["a"] and without_b["c"] == full["c"]
    # today's frozen facts: the number and the code of the newest rule
    assert RULES["split_declaration"].number == 22
    assert RULES["split_declaration"].code == "V22"


def test_a_rule_is_the_same_error_by_number_and_by_slug():
    by_number = ValidationError(22, "m", path="p")
    by_slug = ValidationError("split_declaration", "m", path="p")
    for err in (by_number, by_slug):
        assert (err.rule, err.rule_id, err.code) == (22, "split_declaration", "V22")
        assert err.path == "p" and err.message == "m"
    assert str(by_number) == str(by_slug) == "[V22] at p m"


@pytest.mark.parametrize(
    "unknown",
    # the next unused number follows the registry, so a rule-adding PR does
    # not have to touch this list; ``bool`` is an ``int`` but names no rule
    [0, max(RULES_BY_NUMBER) + 1, 10_000, True, "no_such_rule", "V4", ""],
)
def test_an_unknown_rule_is_refused_at_construction(unknown: int | str):
    with pytest.raises(AssertionError, match="unknown checklist rule"):
        ValidationError(unknown, "m")
    with pytest.raises(AssertionError, match="unknown checklist rule"):
        lookup_rule(unknown)


@pytest.mark.parametrize("number", sorted(RULES_BY_NUMBER))
def test_the_rendered_code_is_unchanged(number: int):
    """The user-visible form of every refusal — ``[V<n>] at <path> <message>``
    — is a frozen contract: it is quoted in docs, in demo transcripts and in
    the ``description`` of shipped documents whose digests are pinned."""
    assert str(ValidationError(number, "m", path="p")) == f"[V{number}] at p m"
    assert str(ValidationError(number, "m")) == f"[V{number}] m"
    err = ValidationError(number, "m")
    assert err.rule == number and err.code == f"V{number}"
    assert err.rule_id == RULES_BY_NUMBER[number].slug


def test_the_codes_match_the_committed_snapshot():
    """``rule_codes.json`` is the slug → ``V<n>`` table as it stood when the
    ids became names. A new rule adds a line; an existing line never changes,
    because the number it records was frozen when that rule landed."""
    snapshot = json.loads((Path(__file__).parent / "rule_codes.json").read_text())
    assert snapshot == {rule.slug: rule.code for rule in RULES.values()}
    assert snapshot == {slug: ValidationError(slug, "m").code for slug in RULES}


def test_rule_4_decode_over_a_prompt_frame_read():
    """``decode`` reduces tokens a decode produced; in the prompt frame there
    are none — only tokens that were given."""
    doc = base_doc()
    doc["method"]["save"] = [
        saved("logits", "patched", "said.json", {"kind": "decode"})
    ]
    err = expect_rule(4, doc)
    assert "generated" in str(err)


def test_decode_over_a_continuation_read_is_legal():
    doc = _reads_the_continuation(base_doc())
    doc["method"]["save"] = [
        saved("logits", "patched", "said.json", {"kind": "decode"})
    ]
    parse_and_validate(doc)


# --------------------------------------------------------------------------- #
# independent violations are reported together
# --------------------------------------------------------------------------- #


def _three_unrelated_violations() -> dict[str, Any]:
    """One document breaking rules 10, 11 and 17 — three different subjects.

    Rule 10 (the save manifest), rule 11 (the sink rule) and rule 17 (the
    model's realization) share no inputs, so a reader fixing one learns
    nothing about the other two. That is what makes reporting them one at a
    time three round trips rather than one.
    """
    doc = base_doc()
    # rule 11 — a declared position nothing addresses
    doc["method"]["positions"] = {"orphan": {"index": 3}}
    # rule 17 — `double_quant` is 4-bit vocabulary, not int8's
    doc["model"] = {
        **doc["model"],
        "quantization": {"scheme": "int8", "double_quant": True},
    }
    # rule 10 — two entries write one file
    doc["method"]["save"].append(
        {"read": "logits", "model": "patched", "file_path": "ld.json"}
    )
    return in_order(doc)


def test_three_unrelated_violations_are_reported_together():
    """The acceptance: three problems, three refusals, three paths.

    The docstring of `validate_document` used to promise "the first
    violation", and a document with three problems cost three edit-and-rerun
    cycles to learn what one run already knew.
    """
    with pytest.raises(ValidationErrors) as caught:
        parse_and_validate(_three_unrelated_violations())

    err = caught.value
    assert len(err.errors) == 3, [str(e) for e in err.errors]
    assert {e.rule for e in err.errors} == {10, 11, 17}
    assert all(e.path for e in err.errors), "every violation names its own path"
    rendered = str(err)
    for one in err.errors:
        assert str(one) in rendered, "the listing carries each violation verbatim"


def test_the_aggregate_is_still_a_validation_error():
    """So a caller that only wants "this document was refused" is unaffected,
    and a test asserting a rule number still can: `rule`, `code` and `path`
    are the first violation's, in checklist order."""
    with pytest.raises(ValidationError) as caught:
        parse_and_validate(_three_unrelated_violations())

    err = caught.value
    assert isinstance(err, ValidationErrors)
    assert err.rule == 10 and err.code == "V10"
    assert err.rule == err.errors[0].rule and err.path == err.errors[0].path


def test_one_violation_is_raised_as_itself():
    """The single-error path is untouched — same type, same message.

    Every other test in this file depends on that, which is the point: making
    three errors legible must not change what one error looks like.
    """
    doc = base_doc()
    doc["method"]["positions"] = {"orphan": {"index": 3}}
    err = expect_rule(11, in_order(doc))
    assert not isinstance(err, ValidationErrors)


def test_a_gating_rule_still_stops_the_pass():
    """Rules 3 and 4 keep first-error behaviour, because the checks after them
    are not evaluable rather than merely uninteresting: they index
    `doc.writes[...]` and friends directly, so an unresolved reference would
    raise KeyError instead of refusing.

    This document breaks rule 4 *and* rule 17; only rule 4 is reported.
    """
    doc = base_doc()
    doc["method"]["reads"]["v_cf"]["site"] = "not_a_site"
    doc["model"] = {
        **doc["model"],
        "quantization": {"scheme": "int8", "double_quant": True},
    }
    err = expect_rule(4, in_order(doc))
    assert not isinstance(err, ValidationErrors)


def test_collecting_never_hides_a_crash():
    """Only ValidationError is collected. Anything else propagates, because a
    collector that swallowed a KeyError would turn a bug in the validator into
    a silently partial validation."""
    import causalab.protocol.rules.document as validate_module

    doc = parse_document(base_doc())
    boom = KeyError("a bug in a check, not a bad document")

    def explode(_doc: Any) -> None:
        raise boom

    original = validate_module._check_sinks
    validate_module._check_sinks = explode
    try:
        with pytest.raises(KeyError):
            validate_module.validate_document(doc)
    finally:
        validate_module._check_sinks = original


# rule 21 — operand reachability -------------------------------------------- #


def _two_site_doc(
    read_site: dict[str, Any], write_site: dict[str, Any]
) -> dict[str, Any]:
    """A document whose write at ``write_site`` is fed from ``read_site``.

    Deliberately additive: rule 20 is about where the operand came from, not
    about the mechanism, and an `add_scaled` delta is the idiom that makes the
    geometry load-bearing (a routed per-edge contribution, §2.8).
    """
    doc = base_doc()
    doc["method"]["sites"]["src"] = read_site
    doc["method"]["sites"]["dst"] = write_site
    doc["method"]["reads"]["v_src"] = {"site": "src", "pos": -1}
    # base_doc's own target read and site are replaced wholesale here, and an
    # unreferenced read or site is rule 11 — which would mask rule 20.
    doc["method"]["reads"].pop("v_cf")
    doc["method"]["intervened_models"][UNWRITTEN]["reads"] = ["v_src"]
    doc["method"]["sites"].pop("tgt")
    doc["method"]["writes"] = {
        "patch": {
            "site": "dst",
            "pos": -1,
            "do": {"add_scaled": {"op": "v_src", "alpha": 1.0}},
        }
    }
    return doc


def test_rule_21_operand_read_deeper_than_the_write_is_refused():
    doc = _two_site_doc(
        {"component": "block_output", "layers": [9]},
        {"component": "block_output", "layers": [3]},
    )
    err = expect_rule(21, doc)
    assert "'v_src'" in str(err) and "layer 9" in str(err) and "layer 3" in str(err)


def test_rule_21_operand_read_upstream_of_the_write_is_legal():
    parse_and_validate(
        _two_site_doc(
            {"component": "block_output", "layers": [1]},
            {"component": "block_output", "layers": [3]},
        )
    )


def test_rule_21_equal_depth_is_legal_because_that_is_harvest_inject():
    """Corpus 03's shape: a receiver's value read in one model and injected at
    the very same address in another. Equal depth must never be refused."""
    parse_and_validate(
        _two_site_doc(
            {"component": "block_output", "layers": [3]},
            {"component": "block_output", "layers": [3]},
        )
    )


def test_rule_21_is_block_order_aware_within_one_layer():
    """A head's residual contribution feeding the same layer's MLP is a real
    sequential edge; the reverse direction is not."""
    parse_and_validate(
        _two_site_doc(
            {"component": "attention_result", "layers": [3], "head": 1},
            {"component": "mlp_input", "layers": [3]},
        )
    )
    err = expect_rule(
        21,
        _two_site_doc(
            {"component": "mlp_output", "layers": [3]},
            {"component": "attention_output", "layers": [3]},
        ),
    )
    assert "mlp_output" in str(err) and "attention_output" in str(err)


def test_rule_21_lm_head_sorts_after_every_block():
    """The two layer-less trunk components are deeper than any block, so an
    `lm_head`-derived operand can land nowhere but the trunk's own tail."""
    doc = base_doc()
    doc["method"]["reads"].pop("v_cf")
    doc["method"]["reads"]["v_logits"] = {
        "site": "lm_head",
        "pos": -1,
        "dims": list(range(768)),  # gpt2 hidden, so the widths agree
    }
    doc["method"]["intervened_models"][UNWRITTEN]["reads"] = ["v_logits"]
    doc["method"]["writes"] = {
        "patch": {
            "site": "tgt",
            "pos": -1,
            "do": {"add_scaled": {"op": "v_logits", "alpha": 1.0}},
        }
    }
    err = expect_rule(21, doc)
    assert "lm_head" in str(err)


def test_rule_21_ignores_params_and_literal_operands():
    """A param has no address, so it has no geometry to be wrong about."""
    doc = base_doc()
    doc["method"]["params"] = {"bias": {"file_path": "p.safetensors"}}
    doc["method"]["writes"] = {
        "patch": {
            "site": "tgt",
            "pos": -1,
            "do": {"add_scaled": {"op": "bias", "alpha": 1.0}},
        }
    }
    _without_the_counterfactual_read(doc)
    parse_and_validate(doc)


def test_rule_21_constrains_the_alpha_slot_too():
    """`alpha` may name a read (§2.8 operands), and a coefficient taken from
    downstream is as unattributable as a downstream value."""
    doc = base_doc()
    doc["method"]["sites"]["deep"] = {"component": "block_output", "layers": [9]}
    doc["method"]["reads"]["k"] = {"site": "deep", "pos": -1, "dims": [0]}
    doc["method"]["intervened_models"][UNWRITTEN]["reads"] = ["v_cf", "k"]
    doc["method"]["writes"] = {
        "patch": {
            "site": "tgt",
            "pos": -1,
            "do": {"add_scaled": {"op": "v_cf", "alpha": "k"}},
        }
    }
    err = expect_rule(21, doc)
    assert "'k'" in str(err)


def test_the_whole_corpus_is_upstream_or_equal():
    """The rule is a *new refusal*, so its blast radius is the claim worth
    pinning: no shipped document routes an operand backwards.

    Read straight off the JSON rather than through the loader — that keeps the
    claim about the authored corpus and not about one point of an expansion,
    and it stays true for the two documents whose layer is a sweep or an
    artifact reference (skipped here, concrete at every point, and covered by
    ``test_corpus.py`` loading all of them under the live rule).
    """
    import json

    from causalab.protocol.positions.alignment import COMPONENT_RANK, UNRANKED

    from tests.protocol._env import CORPUS_DIR

    def depth(site: dict[str, Any]) -> tuple[int, int] | None:
        component = site["component"]
        if component not in COMPONENT_RANK:
            return None
        rank = COMPONENT_RANK.get(component, UNRANKED)
        if component in ("ln_final", "lm_head"):
            return (1_000_000, rank)
        layers = site.get("layers", [0])  # a band: its shallowest member
        if not isinstance(layers, list) or not layers:
            return None  # a sweep or an artifact wrapper
        layer = layers[0]
        return None if not isinstance(layer, int) else (layer, rank)

    files = sorted(CORPUS_DIR.glob("*_im.json"))
    assert files, "corpus did not resolve"
    compared = 0
    for path in files:
        doc = json.loads(path.read_text())["method"]
        sites, reads = doc.get("sites", {}), doc.get("reads", {})
        for wname, write in doc.get("writes", {}).items():
            payload = next(iter(write["do"].values()))
            slots = (
                [payload]
                if isinstance(payload, str)
                else [payload.get(k) for k in ("op", "alpha")]
                if isinstance(payload, dict)
                else []
            )
            target = depth(sites[write["site"]])
            for slot in slots:
                if not isinstance(slot, str) or slot not in reads:
                    continue
                source = depth(sites[reads[slot]["site"]])
                if target is None or source is None:
                    continue
                compared += 1
                assert source <= target, (
                    f"{path.name}: write {wname!r} reads {slot!r} from deeper "
                    f"({source} > {target}) — rule 21 would refuse it"
                )
    # 18, not 20: corpus 07's layer is a sweep and 08's is an artifact
    # reference, so neither is comparable off the authored JSON.
    assert compared >= 18, f"only {compared} operand edges compared"


# --------------------------------------------------------------------------- #
# rule 4 on regularizer lists (§2.11)
# --------------------------------------------------------------------------- #


def _two_gate_fit(objective, params=("g0", "g1")) -> dict[str, Any]:
    doc = base_doc()
    doc["method"]["sites"]["tgt2"] = {"component": "block_output", "layers": [2]}
    doc["method"]["featurizers"] = {"g0": {"kind": "gate"}, "g1": {"kind": "gate"}}
    doc["method"]["reads"]["v_cf"]["featurizer"] = "g0"
    doc["method"]["reads"]["v2"] = {"site": "tgt2", "pos": -1, "featurizer": "g1"}
    doc["method"]["intervened_models"][UNWRITTEN]["reads"] = ["v_cf", "v2"]
    doc["method"]["writes"]["patch"]["featurizer"] = "g0"
    doc["method"]["writes"]["patch2"] = {
        "site": "tgt2",
        "pos": -1,
        "featurizer": "g1",
        "do": {"swap": "v2"},
    }
    doc["method"]["intervened_models"]["patched"]["writes"] = ["patch", "patch2"]
    doc["method"]["train"] = {
        "objective": objective,
        "params": list(params),
        "optimizer": {"name": "adamw", "lr": 1e-2},
        "steps": {"epochs": 1},
        "batch": {"pairs": 2},
    }
    doc["method"]["save"] += [
        {"value": "g0", "site": "tgt", "file_path": "g0.safetensors"},
        {"value": "g1", "site": "tgt2", "file_path": "g1.safetensors"},
    ]
    return doc


def _one_gate_at_two_sites_fit() -> dict[str, Any]:
    """``_two_gate_fit`` with one declared gate named from both sites: one
    parameter set at two addresses (§2.5, one name at several sites)."""
    doc = _two_gate_fit([[1.0, _ld()]], params=("g0",))
    doc["method"]["featurizers"] = {"g0": {"kind": "gate"}}
    doc["method"]["reads"]["v2"]["featurizer"] = "g0"
    doc["method"]["writes"]["patch2"]["featurizer"] = "g0"
    doc["method"]["save"] = [
        entry for entry in doc["method"]["save"] if entry.get("value") != "g1"
    ]
    return doc


def test_one_gate_named_at_two_sites_is_one_parameter_set_to_every_rule():
    """§2.5: a mask tied across two layers is one gate named from both sites'
    reads and writes — `train.params` lists it once, `save` writes one bundle
    (stamped with the site its entry names), and rules 4, 10, 11 and 12 take
    the spelling as they take a gate at one site."""
    parse_and_validate(_one_gate_at_two_sites_fit())
    # the same gate saved at the other site it is used at is as good a bundle
    other = _one_gate_at_two_sites_fit()
    other["method"]["save"][-1]["site"] = "tgt2"
    parse_and_validate(other)


def test_a_regularizer_list_over_two_trained_gates_validates():
    parse_and_validate(_two_gate_fit([[1.0, _ld()], [0.01, {"l1": ["g0", "g1"]}]]))
    parse_and_validate(
        _two_gate_fit(
            {
                "fit": _ld(weight=1.0),
                "sparsity": {"weight": 0.01, "l1": ["g0", "g1"]},
            }
        )
    )


def test_rule_4_regularizer_list_unknown_name():
    err = expect_rule(4, _two_gate_fit([[1.0, _ld()], [0.01, {"l1": ["g0", "ghost"]}]]))
    assert "ghost" in str(err) and err.path == "train.objective[1]"


def test_rule_4_regularizer_list_names_featurizers_not_slots():
    err = expect_rule(
        4, _two_gate_fit([[1.0, _ld()], [0.01, {"l1": ["g0", "g1.theta"]}]])
    )
    assert "g1.theta" in str(err)
    # the single-name form still takes a dotted slot
    parse_and_validate(_two_gate_fit([[1.0, _ld()], [0.01, {"l1": "g1.theta"}]]))


def test_rule_4_l0_names_a_gate_not_a_rotation():
    """``l0`` is a gate's expected kept fraction (§2.11); a subspace has no
    mask to count, so an ``l0`` over it is refused rather than silently read
    as ``|p|``."""
    doc = base_doc()
    doc["method"]["featurizers"] = {
        "rot": {"kind": "subspace", "k": 4, "parametrization": "cayley"}
    }
    doc["method"]["reads"]["v_cf"]["featurizer"] = "rot"
    doc["method"]["writes"]["patch"]["featurizer"] = "rot"
    doc["method"]["train"] = {
        "objective": [[1.0, _ld()], [0.01, {"l0": "rot"}]],
        "params": ["rot"],
        "optimizer": {"name": "adamw", "lr": 1e-3},
        "steps": {"epochs": 1},
        "batch": {"pairs": 2},
    }
    doc["method"]["save"] += [
        {"value": "rot", "site": "tgt", "file_path": "rot.safetensors"}
    ]
    err = expect_rule(4, doc)
    assert "l0" in str(err) and "mask" in str(err)


def test_rule_4_costs_keys_are_the_terms_own_targets():
    """§2.11 ``costs``: a multiplier keyed by a name the term does not
    penalize refers to nothing, in either spelling; the twins — every key a
    target, a dotted single target costed by its dotted name — validate."""
    err = expect_rule(
        4,
        _two_gate_fit(
            [[1.0, _ld()], [0.01, {"l1": ["g0"], "costs": {"g1": 0.5}}]],
            params=("g0", "g1"),
        ),
    )
    assert "g1" in str(err) and err.path == "train.objective[1].costs"
    err = expect_rule(
        4,
        _two_gate_fit(
            {
                "fit": _ld(weight=1.0),
                "sparsity": {"weight": 0.01, "l1": ["g0", "g1"], "costs": {"g2": 1.0}},
            }
        ),
    )
    assert "g2" in str(err) and err.path == "train.objective.sparsity.costs"
    parse_and_validate(
        _two_gate_fit(
            [
                [1.0, _ld()],
                [0.01, {"l1": ["g0", "g1"], "costs": {"g0": 1.0, "g1": 0.25}}],
            ]
        )
    )
    parse_and_validate(
        _two_gate_fit(
            [[1.0, _ld()], [0.01, {"l1": ["g0", "g1"], "costs": "parameter_count"}]]
        )
    )
    parse_and_validate(
        _two_gate_fit(
            [[1.0, _ld()], [0.01, {"l1": "g1.theta", "costs": {"g1.theta": 2.0}}]]
        )
    )


def test_rule_4_a_constraint_holds_a_gates_density_and_has_no_weight_to_schedule():
    """§2.11 ``constraint``: every target is a gate (a rotation's |p| has no
    density), and the term has no weight for an ``anneal`` or ``control`` to
    address; the twins — an `l1` constraint on two sigmoid gates, an `l0`
    constraint on a hard-concrete gate — validate."""
    constraint = {"target": 0.1, "dual": {"lr": 0.05}}
    parse_and_validate(
        _two_gate_fit(
            {
                "fit": _ld(weight=1.0),
                "density": {"l1": ["g0", "g1"], "constraint": constraint},
            }
        )
    )
    doc = base_doc()
    doc["method"]["featurizers"] = {
        "rot": {"kind": "subspace", "k": 4, "parametrization": "cayley"}
    }
    doc["method"]["reads"]["v_cf"]["featurizer"] = "rot"
    doc["method"]["writes"]["patch"]["featurizer"] = "rot"
    doc["method"]["train"] = {
        "objective": {
            "fit": _ld(weight=1.0),
            "density": {"l1": "rot", "constraint": constraint},
        },
        "params": ["rot"],
        "optimizer": {"name": "adamw", "lr": 1e-3},
        "steps": {"epochs": 1},
        "batch": {"pairs": 2},
    }
    doc["method"]["save"] += [
        {"value": "rot", "site": "tgt", "file_path": "rot.safetensors"}
    ]
    err = expect_rule(4, doc)
    assert "not a gate" in str(err) and err.path == "train.objective.density.constraint"
    scheduled = _two_gate_fit(
        {
            "fit": _ld(weight=1.0),
            "density": {"l1": ["g0", "g1"], "constraint": constraint},
        }
    )
    scheduled["method"]["train"]["anneal"] = {
        "train.objective.density.weight": [1.0, 0.0, 0.5]
    }
    err = expect_rule(4, scheduled)
    assert "dual pair" in str(err)
    hard = _two_gate_fit(
        {
            "fit": _ld(weight=1.0),
            "density": {"l0": "g0", "constraint": constraint},
        }
    )
    hard["method"]["featurizers"]["g0"]["parametrization"] = "hard_concrete"
    parse_and_validate(hard)


def _position_gate_doc(*, read_pos=None, write_pos=None) -> dict:
    doc = base_doc()
    doc["method"]["featurizers"] = {"pg": {"kind": "gate", "axis": "position"}}
    doc["method"]["reads"]["v_cf"]["pos"] = read_pos or {"span": [0, 3]}
    doc["method"]["writes"]["patch"]["pos"] = write_pos or {"span": [0, 3]}
    doc["method"]["writes"]["patch"]["featurizer"] = "pg"
    return doc


def test_a_position_gate_on_a_fixed_span_validates():
    """§2.5 ``axis``: the legitimate case — one fixed window, read and
    write. Fails without the change on rule 1 (unknown key `axis`). An
    ``atomic`` window is the same window (``atomic`` decides rule 8's write
    cardinality, not the feature space), so it validates too."""
    parse_and_validate(_position_gate_doc())
    atomic = {"span": [0, 3], "atomic": True}
    parse_and_validate(_position_gate_doc(read_pos=atomic, write_pos=atomic))


def test_rule_4_a_position_gate_addresses_a_fixed_span_of_one_length():
    err = expect_rule(4, _position_gate_doc(write_pos=-1))
    assert "fixed span" in str(err) and err.path == "writes.patch.pos"
    assert "an `index` or a span of one position is one scalar" in str(err)
    # the same one-token address under the other spelling is refused alike
    err = expect_rule(4, _position_gate_doc(write_pos={"span": [0, 1]}))
    assert "span of one position" in str(err)
    err = expect_rule(4, _position_gate_doc(write_pos={"all": True}))
    assert "fixed span" in str(err)
    # a scoped span is sliced out of the anchor's run (2 positions on a
    # 2-token subject under [0, 3]); a generated one is clipped to the row's
    # decode — neither has one length on every row
    err = expect_a_rule(
        4,
        _position_gate_doc(
            write_pos={"span": [0, 3], "scope": {"variable": "subject"}}
        ),
    )
    assert "fixed span" in str(err) and "scope" in str(err)
    generated = _position_gate_doc()
    generated["method"]["reads"]["v_cf"]["featurizer"] = "pg"
    generated["method"]["reads"]["v_cf"]["pos"] = {
        "span": [0, 2],
        "generated": {"max_new_tokens": 4},
    }
    err = expect_a_rule(4, generated)
    assert "fixed span" in str(err) and "generated" in str(err)
    doc = _position_gate_doc()
    doc["method"]["reads"]["v_cf"]["featurizer"] = "pg"
    doc["method"]["reads"]["v_cf"]["pos"] = {"span": [0, 4]}
    err = expect_rule(4, doc)
    assert "one gate, one window" in str(err) and err.path == "featurizers.pg.axis"


def expect_a_rule(rule: int, raw: dict[str, Any]) -> ValidationError:
    """Like ``expect_rule`` when the document may also trip an unrelated rule
    (the checklist reports independent violations together): the violation
    under ``rule`` is returned, whichever came first."""
    with pytest.raises(ValidationError) as err:
        parse_and_validate(raw)
    matching = [e for e in getattr(err.value, "errors", [err.value]) if e.rule == rule]
    assert matching, f"expected V{rule} among {err.value}"
    return matching[0]


def test_rule_4_regularizer_target_not_in_train_params():
    """A penalty on a featurizer the fit never moves is a dead declaration,
    in either form, and used to surface as a KeyError inside the loop."""
    err = expect_rule(
        4, _two_gate_fit([[1.0, _ld()], [0.01, {"l1": ["g0", "g1"]}]], params=("g0",))
    )
    assert "train.params" in str(err) and "g1" in str(err)
    err = expect_rule(
        4,
        _two_gate_fit(
            {
                "fit": _ld(weight=1.0),
                "sparsity": {"weight": 0.01, "l2": "g1"},
            },
            params=("g0",),
        ),
    )
    assert err.path == "train.objective.sparsity"


def test_rule_4_named_objective_read_undeclared():
    doc = _two_gate_fit({"fit": _ld(weight=1.0)})
    doc["method"]["train"]["objective"]["fit"]["read"] = "ghost"
    err = expect_rule(4, doc)
    assert err.path == "train.objective.fit.read"


# rule 27 — a segment anchor names a declared segment; a span is well-formed --- #


def test_rule_27_undeclared_segment_anchor():
    """The pure half of segments and spans (`test_segments.py`, `test_spans.py`) holds every
    variant; this is the checklist's one-failing-document-per-rule entry."""
    doc = base_doc()
    doc["method"]["segments"] = {"frame": "chat"}
    doc["method"]["positions"] = {
        "p": {"index": -1, "scope": {"segment": "assistant_prefx"}}
    }
    doc["method"]["reads"]["v_cf"]["pos"] = "p"
    err = expect_rule(27, doc)
    assert err.rule_id == "segment_declared"
    assert err.path == "positions.p"
    assert "assistant_prefix" in str(err)  # the suggestion
    doc["method"]["positions"]["p"]["scope"]["segment"] = "assistant_prefix"
    parse_and_validate(doc)  # the twin


def test_rule_27_frame_outside_the_vocabulary():
    doc = base_doc()
    doc["method"]["segments"] = {"frame": "plain"}
    err = expect_rule(27, doc)
    assert err.path == "segments.frame"


def _gate_fit(featurizer: dict, objective: list) -> dict:
    doc = base_doc()
    doc["method"]["featurizers"] = {"gate": featurizer}
    doc["method"]["reads"]["v_cf"]["featurizer"] = "gate"
    doc["method"]["writes"]["patch"]["featurizer"] = "gate"
    doc["method"]["train"] = {
        "objective": objective,
        "params": ["gate"],
        "optimizer": {"name": "adamw", "lr": 1e-3},
        "steps": {"epochs": 1},
        "batch": {"pairs": 2},
    }
    doc["method"]["save"] += [
        {"value": "gate", "site": "tgt", "file_path": "gate.safetensors"}
    ]
    return doc


def test_rule_4_l0_pairs_with_a_sampled_mask_and_l1_with_a_deterministic_one():
    """§2.11: the mask penalty is keyed on the gate's map. ``l0`` on a
    ``sigmoid`` (or ``clamp``) gate is ``l1`` under a second name — the relaxed
    mask is already the kept probability — and ``l1`` on a ``hard_concrete``
    gate penalizes a mask its training forward never uses; both are refused,
    naming the spelling that was meant."""
    err = expect_rule(
        4, _gate_fit({"kind": "gate"}, [[1.0, _ld()], [0.01, {"l0": "gate"}]])
    )
    assert "'l1'" in str(err) and "deterministic" in str(err)
    err = expect_rule(
        4,
        _gate_fit(
            {"kind": "gate", "parametrization": "clamp"},
            [[1.0, _ld()], [0.01, {"l0": "gate"}]],
        ),
    )
    assert "'l1'" in str(err)
    err = expect_rule(
        4,
        _gate_fit(
            {"kind": "gate", "parametrization": "hard_concrete"},
            [[1.0, _ld()], [0.01, {"l1": "gate"}]],
        ),
    )
    assert "'l0'" in str(err) and "never uses" in str(err)
    # the two legal pairs, and l2 (a plain |θ|² weight penalty) under either
    for featurizer, kind in (
        ({"kind": "gate"}, "l1"),
        ({"kind": "gate", "parametrization": "hard_concrete"}, "l0"),
        ({"kind": "gate", "parametrization": "hard_concrete"}, "l2"),
    ):
        validate_document(
            parse_document(
                in_order(_gate_fit(featurizer, [[1.0, _ld()], [0.01, {kind: "gate"}]]))
            ),
            engine_is_local=True,
        )


def test_rule_4_a_swept_parametrization_is_refused_when_any_arm_mismatches():
    """A swept map with a penalty one arm cannot take is refused up front,
    naming the arm — the rule the parser applies to the hard-concrete
    constants. Compile would refuse the whole run on that point anyway (every
    expanded point is validated before weights load); saying it here is the
    better message. A sweep whose every arm pairs is legal."""
    err = expect_rule(
        4,
        _gate_fit(
            {
                "kind": "gate",
                "parametrization": {"sweep": ["sigmoid", "hard_concrete"]},
            },
            [[1.0, _ld()], [0.01, {"l0": "gate"}]],
        ),
    )
    assert "'sigmoid'" in str(err) and "one arm of the sweep" in str(err)
    err = expect_rule(
        4,
        _gate_fit(
            {
                "kind": "gate",
                "parametrization": {"sweep": ["sigmoid", "hard_concrete"]},
            },
            [[1.0, _ld()], [0.01, {"l1": "gate"}]],
        ),
    )
    assert "hard_concrete" in str(err) and "one arm of the sweep" in str(err)
    validate_document(
        parse_document(
            in_order(
                _gate_fit(
                    {
                        "kind": "gate",
                        "parametrization": {"sweep": ["sigmoid", "clamp"]},
                    },
                    [[1.0, _ld()], [0.01, {"l1": "gate"}]],
                )
            )
        ),
        engine_is_local=True,
    )


def test_rule_4_a_temperature_anneal_stays_positive():
    """The schedule IS the temperature for an annealed fit (an authored one
    beside it is refused), so it gets the authored field's check: the mask
    divides by T (or β), and a non-positive end is a division by zero or an
    inverted sample, not a sharper mask."""
    for parametrization in (None, "hard_concrete"):
        featurizer = {"kind": "gate"}
        kind = "l1"
        if parametrization:
            featurizer["parametrization"] = parametrization
            kind = "l0"
        for schedule in ([1.0, 0.0, 0.5], [1.0, -1.0, 0.5], [0.0, 0.5, 0.5]):
            doc = _gate_fit(dict(featurizer), [[1.0, _ld()], [0.01, {kind: "gate"}]])
            doc["method"]["train"]["anneal"] = {"gate.theta.temperature": schedule}
            err = expect_rule(4, doc)
            assert "positive" in str(err)
        doc = _gate_fit(dict(featurizer), [[1.0, _ld()], [0.01, {kind: "gate"}]])
        doc["method"]["train"]["anneal"] = {"gate.theta.temperature": [1.0, 0.01, 0.5]}
        validate_document(parse_document(in_order(doc)), engine_is_local=True)


def test_rule_4_an_authored_temperature_beside_its_anneal_is_refused():
    """The anneal's start replaces the authored β before the first forward
    (``_set_anneal`` runs at step 0), so two fields would name one number and
    one silently win."""
    doc = _gate_fit(
        {"kind": "gate", "parametrization": "hard_concrete", "temperature": 0.5},
        [[1.0, _ld()], [0.01, {"l0": "gate"}]],
    )
    doc["method"]["train"]["anneal"] = {"gate.theta.temperature": [0.5, 0.2, 0.5]}
    err = expect_rule(4, doc)
    assert "authors temperature" in str(err)
    del doc["method"]["featurizers"]["gate"]["temperature"]
    validate_document(parse_document(in_order(doc)), engine_is_local=True)


def test_rule_4_a_dead_rule_needs_a_fit_that_trains_the_gate():
    """§2.5 ``dead`` acts in the post-step projection (``freeze_after``) or
    the training backward (``leak``); on a gate no step moves it would be
    digested and never act. Refused on a gate outside ``train.params`` and on
    a document with no ``train`` at all; a trained gate takes either rule."""
    for dead in ({"freeze_after": 3}, {"leak": 0.05}):
        doc = _gate_fit(
            {"kind": "gate", "dead": dead}, [[1.0, _ld()], [0.01, {"l1": "gate"}]]
        )
        validate_document(parse_document(in_order(doc)), engine_is_local=True)
        # the gate exists and is read through, but a second featurizer is trained
        doc = _gate_fit({"kind": "gate", "dead": dead}, [[1.0, _ld()]])
        doc["method"]["featurizers"]["rot"] = {"kind": "subspace", "k": 2}
        doc["method"]["train"]["params"] = ["rot"]
        doc["method"]["save"][-1] = {
            "value": "rot",
            "site": "tgt",
            "file_path": "rot.safetensors",
        }
        err = expect_rule(4, doc)
        assert "not in train.params" in str(err) and f"dead.{next(iter(dead))}" in str(
            err
        )
        # no fit at all
        doc = _gate_fit({"kind": "gate", "dead": dead}, [[1.0, _ld()]])
        del doc["method"]["train"]
        doc["method"]["save"] = doc["method"]["save"][:-1]
        err = expect_rule(4, doc)
        assert "no train section" in str(err)


def test_a_rank_entry_needs_a_gate_and_a_json_path():
    """§2.12 ``rank`` orders a gate's units by theta: a document with no gate has
    nothing to rank, and the table is JSON like every other non-value table."""
    raw = base_doc()
    raw["method"]["save"].append({"kind": "rank", "file_path": "rank.json"})
    err = expect_rule(10, raw)
    assert "declares no gate" in str(err)
    raw = base_doc()
    raw["method"]["featurizers"] = {"g": {"kind": "gate"}}
    raw["method"]["reads"]["v_cf"]["featurizer"] = "g"
    raw["method"]["writes"]["patch"]["featurizer"] = "g"
    raw["method"]["save"].append({"kind": "rank", "file_path": "rank.safetensors"})
    err = expect_rule(10, raw)
    assert "'.json'" in str(err)
    raw["method"]["save"][-1] = {"kind": "rank", "file_path": "rank.json"}
    parse_and_validate(raw)


def test_rule_4_a_budget_gate_takes_no_penalty_no_anneal_and_is_no_signal():
    """§2.5 ``budget``: the mask sums to the step's budget by construction, so a
    penalty has nothing to move, there is no temperature, and the kept count
    is the document's cut — nothing a controller moves can move it."""
    budget = {
        "kind": "gate",
        "parametrization": "budget",
        "k_schedule": {"kind": "fixed", "k": 2},
    }
    for penalty in ("l1", "l0"):
        err = expect_rule(
            4, _gate_fit(budget, [[1.0, _ld()], [0.01, {penalty: "gate"}]])
        )
        assert "budget" in str(err) and "penalty" in str(err)
    doc = _gate_fit(budget, [[1.0, _ld()]])
    doc["method"]["train"]["anneal"] = {"gate.theta.temperature": [1.0, 0.1, 0.5]}
    err = expect_rule(4, doc)
    assert GATE_MAPS["budget"].no_temperature_because in str(err)
    doc = _gate_fit(budget, {"fit": _ld(weight=1.0)})
    doc["method"]["train"]["control"] = {
        "train.objective.fit.weight": {
            "kind": "pid",
            "signal": {"hard_mask_size": "gate"},
            "setpoint": {"ramp": [2, 0, 0.5]},
            "gains": {"kp": 0.1, "ki": 0.01},
        }
    }
    err = expect_rule(4, doc)
    assert "budget gate" in str(err) and "k_schedule.eval" in str(err)
    parse_and_validate(_gate_fit(budget, [[1.0, _ld()]]))


def test_rule_4_a_budget_pool_does_not_mix_split_and_unsplit_members():
    """§2.5 the mapping form: `{"forward": "hard", "backward": "budget"}` and
    plain `"budget"` parse to the same `parametrization`, so the pool rule
    must compare `forward` too — one member thresholding its share of the
    pooled mask at ½ breaks "one pool, one ranking on one scale"."""
    budget = {"parametrization": "budget", "k_schedule": {"kind": "fixed", "k": 2}}
    doc = _two_gate_fit([[1.0, _ld()]])
    doc["method"]["featurizers"]["g0"] = {"kind": "gate", **budget, "pool": "p"}
    doc["method"]["featurizers"]["g1"] = {
        "kind": "gate",
        **budget,
        "parametrization": {"forward": "hard", "backward": "budget"},
        "pool": "p",
    }
    err = expect_a_rule(4, doc)
    assert "disagree on 'forward'" in str(err), str(err)
    assert err.path == "featurizers.g0.parametrization.forward"  # an authored key
    assert "axes" not in str(err)  # the parser refuses a swept mapping form
    # the twin: both members split, or neither
    doc["method"]["featurizers"]["g0"]["parametrization"] = {
        "forward": "hard",
        "backward": "budget",
    }
    parse_and_validate(doc)  # a pool that agrees on the split is fine


def test_rule_4_a_position_gate_named_at_two_sites_is_held_to_one_window():
    """§2.5 ``axis`` + "one name, several sites": θ is one tensor, so one
    position gate at `[0, 3)` on one layer and `[0, 4)` on another is one gate
    over two windows — refused at load under rule 4, not at the build; the
    same name at the same window on both layers validates (a mask over
    positions tied across the two)."""
    doc = _position_gate_doc()
    method = doc["method"]
    # deeper than the read's layer 3, so the operand routes to it (rule 21)
    method["sites"]["tgt2"] = {"component": "block_output", "layers": [4]}
    method["writes"]["patch2"] = {
        "site": "tgt2",
        "pos": {"span": [0, 4]},
        "featurizer": "pg",
        "do": {"swap": "v_cf"},
    }
    method["intervened_models"]["patched"]["writes"] = ["patch", "patch2"]
    err = expect_a_rule(4, doc)
    assert "one gate, one window" in str(err) and err.path == "featurizers.pg.axis"
    method["writes"]["patch2"]["pos"] = {"span": [0, 3]}
    parse_and_validate(doc)


# §2.5 `parametrization: boundary` — rule 4's chain half, rule 10's rank save - #


def _boundless_fit(
    *, chain=("rot", "bnd"), objective=None, extra: dict | None = None
) -> dict:
    """A Boundless DAS fit on the base document: a rotation with a boundary
    gate behind it, both trained, `l1` on the gate unless told otherwise."""
    doc = base_doc()
    featurizers: dict = {
        "bnd": {"kind": "gate", "parametrization": "boundary", **(extra or {})}
    }
    if "rot" in chain:
        featurizers["rot"] = {"kind": "subspace", "k": 4, "parametrization": "cayley"}
    if "g" in chain:
        featurizers["g"] = {"kind": "gate"}
    if "pca" in chain:
        featurizers["pca"] = {
            "kind": "pca",
            "k": 4,
            "file_path": "pca/basis.safetensors",
        }
    doc["method"]["featurizers"] = featurizers
    doc["method"]["reads"]["v_cf"]["featurizer"] = list(chain)
    doc["method"]["writes"]["patch"]["featurizer"] = list(chain)
    trained = [n for n in chain if n in ("rot", "bnd", "g")]
    doc["method"]["train"] = {
        "objective": objective or [[1.0, _ld()], [0.05, {"l1": "bnd"}]],
        "params": trained,
        "optimizer": {"name": "adamw", "lr": 1e-3},
        "steps": {"epochs": 1},
        "batch": {"pairs": 2},
    }
    doc["method"]["save"] += [
        {"value": n, "site": "tgt", "file_path": f"{n}.safetensors"} for n in trained
    ]
    return doc


def test_a_boundary_gate_behind_an_ordered_basis_validates_with_l1_anneal_and_control():
    """The legitimate spellings: behind a `subspace` or a `pca`, `l1` on the
    gate, its temperature annealed, and its kept count (`⌈β⌉`) as a control
    signal — all fall out of the `GateMap` row (rule 4 reads it)."""
    doc = _boundless_fit()
    doc["method"]["train"]["anneal"] = {"bnd.theta.temperature": [1.0, 0.01, 0.5]}
    parse_and_validate(doc)
    parse_and_validate(_boundless_fit(chain=("pca", "bnd")))
    controlled = _boundless_fit(objective={"fit": _ld(weight=1.0)})
    controlled["method"]["train"]["control"] = {
        "train.objective.fit.weight": {
            "kind": "pid",
            "signal": {"hard_mask_size": "bnd"},
            "setpoint": {"ramp": [2, 0, 0.5]},
            "gains": {"kp": 0.1, "ki": 0.01},
        }
    }
    parse_and_validate(controlled)


def test_rule_4_a_boundary_gate_directly_follows_a_subspace_or_pca():
    err = expect_rule(4, _boundless_fit(chain=("bnd",)))
    assert "first in its chain" in str(err) and "ordered" in str(err)
    assert err.path == "reads.v_cf.featurizer"
    err = expect_rule(4, _boundless_fit(chain=("g", "bnd")))
    assert "behind 'g' (gate)" in str(err)
    # one stage between the basis and the boundary is one too many (v1)
    err = expect_rule(4, _boundless_fit(chain=("rot", "g", "bnd")))
    assert "behind 'g' (gate)" in str(err)


def test_rule_4_l0_is_refused_on_a_boundary_gate():
    err = expect_rule(
        4, _boundless_fit(objective=[[1.0, _ld()], [0.05, {"l0": "bnd"}]])
    )
    assert "'l1'" in str(err) and "deterministic" in str(err)


def test_rule_10_a_rank_save_is_refused_beside_a_boundary_gate():
    doc = _boundless_fit()
    doc["method"]["save"].append({"kind": "rank", "file_path": "rank.json"})
    err = expect_rule(10, doc)
    assert "no units to order" in str(err) and "'bnd'" in str(err)
