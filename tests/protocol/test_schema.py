"""Strict-parse behavior: sugar, aliases, wrapper shapes, raw loading."""

from __future__ import annotations

import copy
import json

import pytest

from causalab.protocol.schema.explicit import canonical_bytes, canonicalize, digest
from causalab.protocol.rules.errors import ParseError, ValidationError
from causalab.protocol.schema import PositionSpec, Sweep, load_raw, parse_document
from causalab.protocol.lowering import find_axes

from causalab.tables import inline_ref
from tests.protocol._docs import (
    LOGIT_DIFF,
    UNWRITTEN,
    base_doc,
    base_only_doc,
    by_label,
    in_order,
    inline_doc,
    saved,
    term,
)

pytestmark = pytest.mark.unit


def _fit(**extra):
    """An objective or eval term over `base_doc`'s one aggregation —
    the logit difference on ``patched``'s logits (§2.11)."""
    return term("logits", "patched", dict(LOGIT_DIFF), **extra)


def _ld(raw: dict) -> dict:
    """The aggregation `base_doc`'s one save entry tabulates (§2.12)."""
    return raw["method"]["save"][0]["aggregation"]


def test_load_raw_rejects_duplicate_keys():
    with pytest.raises(ParseError) as err:
        load_raw(
            '{"header": {"protocol_version": "4"}, "header": {"protocol_version": "4"}}'
        )
    assert err.value.code == "P2"


def test_load_raw_rejects_non_finite():
    with pytest.raises(ParseError):
        load_raw('{"version": NaN}')


def test_load_raw_rejects_non_object_top_level():
    with pytest.raises(ParseError) as err:
        load_raw("[1, 2]")
    assert err.value.code == "P1"


def test_int_pos_sugar_expands():
    doc = parse_document(base_doc())
    pos = doc.reads["v_cf"].pos
    assert isinstance(pos, PositionSpec) and pos.index == -1


def test_the_retired_neural_model_alias_is_an_unknown_group():
    """v1 accepted `neural_model` for `model`; v2 has one spelling, and the
    refusal suggests it (`causalab migrate` rewrites the alias)."""
    raw = base_doc()
    raw["neural_model"] = raw.pop("model")
    with pytest.raises(ParseError) as err:
        parse_document(raw)
    assert err.value.code == "P3" and "model" in str(err.value)


def test_missing_required_section():
    raw = base_doc()
    del raw["method"]["sites"]
    with pytest.raises(ParseError) as err:
        parse_document(raw)
    assert err.value.code == "P2"


def test_unsupported_version():
    raw = base_doc()
    raw["header"]["protocol_version"] = "99"
    with pytest.raises(ParseError):
        parse_document(raw)


def test_position_spec_needs_exactly_one_anchor():
    raw = base_doc()
    raw["method"]["positions"] = {"p": {"index": -1, "variable": "x"}}
    raw["method"]["reads"]["v_cf"]["pos"] = "p"
    with pytest.raises(ParseError):
        parse_document(in_order(raw))


def test_all_pos_sugar_expands():
    """The bare string ``"all"`` is the all-positions sugar, not a lookup in
    the positions table (§2.3)."""
    raw = base_doc()
    raw["method"]["reads"]["v_cf"]["pos"] = "all"
    pos = parse_document(raw).reads["v_cf"].pos
    assert isinstance(pos, PositionSpec) and pos.all is True


def test_all_anchor_parses_inline():
    raw = base_doc()
    raw["method"]["reads"]["v_cf"]["pos"] = {"all": True}
    pos = parse_document(raw).reads["v_cf"].pos
    assert isinstance(pos, PositionSpec) and pos.all is True and pos.index is None


def test_all_is_exclusive_with_the_other_anchors():
    raw = base_doc()
    raw["method"]["reads"]["v_cf"]["pos"] = {"all": True, "index": -1}
    with pytest.raises(ParseError):
        parse_document(raw)


@pytest.mark.parametrize("modifier", ["scope", "relative_to"])
def test_all_takes_no_modifier(modifier):
    """``scope``/``relative_to`` narrow an index or span; there is nothing
    left to narrow inside "every token"."""
    raw = base_doc()
    raw["method"]["reads"]["v_cf"]["pos"] = {
        "all": True,
        modifier: {"variable": "subject"},
    }
    with pytest.raises(ParseError):
        parse_document(raw)


def test_all_is_a_flag_not_a_selection():
    raw = base_doc()
    raw["method"]["reads"]["v_cf"]["pos"] = {"all": [0, 1]}
    with pytest.raises(ParseError) as err:
        parse_document(raw)
    assert "all" in str(err.value)


def test_layerless_component_rejects_layer():
    raw = base_doc()
    raw["method"]["sites"]["lm_head"]["layers"] = 0
    with pytest.raises(ParseError):
        parse_document(raw)


def test_do_has_exactly_one_mechanism():
    raw = base_doc()
    raw["method"]["writes"]["patch"]["do"] = {"swap": "v_cf", "renormalize": True}
    with pytest.raises(ParseError):
        parse_document(raw)


def test_sweep_wrapper_parses_to_axis_values():
    raw = base_doc()
    raw["method"]["sites"]["tgt"]["layers"] = {"sweep": [1, 2, 3]}
    doc = parse_document(raw)
    layers = doc.sites["tgt"].layers
    # each swept value is a layer index and denotes the one-layer band
    assert isinstance(layers, Sweep) and layers.values == ((1,), (2,), (3,))


def test_entry_level_sweep_on_positions():
    raw = base_doc()
    raw["method"]["positions"] = {
        "tap": {"sweep": [{"index": -1}, {"variable": "subject"}]}
    }
    raw["method"]["reads"]["v_cf"]["pos"] = "tap"
    raw["method"]["writes"]["patch"]["pos"] = "tap"
    doc = parse_document(in_order(raw))
    tap = doc.positions["tap"]
    assert isinstance(tap, Sweep) and len(tap.values) == 2


# --------------------------------------------------------------------------- #
# §2.10 token_form — optional, `id` only; an answer string is tokenized as written
# --------------------------------------------------------------------------- #


def test_metric_token_form_is_absent_by_default():
    """An answer string is tokenized as written, so a document has nothing to
    say about its form: the key is optional and an unauthored key is ``None``
    on the spec (and absent from the canonical form — the shipped-digest pins
    hold that half)."""
    raw = base_doc()
    assert "token_form" not in _ld(raw)
    assert by_label(parse_document(in_order(raw)))["ld"].token_form is None


@pytest.mark.parametrize(
    ("retired", "rewrite"),
    [
        ("auto", "so the rewrite needs the tokenizer"),
        ("bare", "as s.lstrip(' ')"),
        ("space_prefixed", "as ' ' + s.lstrip(' ')"),
    ],
)
def test_metric_token_form_refuses_the_retired_values_by_name(
    retired: str, rewrite: str
):
    """The three values that used to rewrite an answer's leading space are
    refused with the replacement stated: put the space in the string. The
    logit difference reads its answers from dataset columns, so the refusal
    names them and states the table rewrite that keeps the scored tokens.
    Dropping the key alone would change the scored token of a value written
    without the space."""
    raw = base_doc()
    _ld(raw)["token_form"] = retired
    with pytest.raises(ParseError) as err:
        parse_document(in_order(raw))
    assert err.value.code == "P4"
    assert err.value.path == "save[0].aggregation.token_form"
    message = str(err.value)
    assert f"token_form {retired!r} was retired" in message
    assert "tokenized as written" in message
    assert "a: 'cf_answer'; b: 'base_answer'" in message
    assert "each member of a list of forms included" in message
    assert rewrite in message
    assert "shipped tables" not in message


def test_a_retired_token_form_on_literal_answers_states_the_rewrite():
    """A literal answer list is in the document, so the refusal states the
    rewrite of each string and names no column."""
    raw = base_doc()
    raw["method"]["save"][0]["aggregation"] = {
        "kind": "class_probs",
        "groups": {"city": ["Seattle"]},
        "token_form": "space_prefixed",
    }
    with pytest.raises(ParseError) as err:
        parse_document(in_order(raw))
    assert err.value.code == "P4"
    message = str(err.value)
    assert "' Seattle'" in message and "'Seattle'" in message
    assert "as ' ' + s.lstrip(' ')" in message
    assert "dataset columns" not in message


def test_metric_token_form_is_not_required_where_it_does_not_apply():
    """``kl`` compares two reads and ``top_k`` ranks a vector: neither resolves
    an authored string to a token id, so neither takes the field at all."""
    raw = base_doc()
    raw["method"]["reads"]["cf_logits"] = {"site": "lm_head", "pos": -1}
    raw["method"]["intervened_models"][UNWRITTEN]["reads"].append("cf_logits")
    raw["method"]["save"] += [
        saved(
            "logits",
            "patched",
            "kl.json",
            {"kind": "kl", "target": {"read": "cf_logits", "model": UNWRITTEN}},
        ),
        saved("logits", "patched", "tk.json", {"kind": "top_k", "k": 3, "by": "prob"}),
    ]
    parsed = parse_document(in_order(raw))
    assert by_label(parsed)["kl"].token_form is None


def test_metric_token_form_parses_id():
    """The one value: the column holds integer vocabulary ids."""
    raw = base_doc()
    _ld(raw)["token_form"] = "id"
    assert by_label(parse_document(in_order(raw)))["ld"].token_form == "id"


def test_metric_token_form_rejects_an_unknown_form():
    raw = base_doc()
    _ld(raw)["token_form"] = "spaced"
    with pytest.raises(ParseError) as err:
        parse_document(in_order(raw))
    assert err.value.code == "P4"


def test_metric_token_form_is_refused_on_kinds_that_resolve_no_string():
    """``kl`` compares two reads and ``top_k`` reports indices it found —
    neither turns an authored string into a token id, so the knob is
    meaningless. Still true now that ``top_k`` runs over any read: it decodes
    an index only when the read taps ``lm_head``, and never resolves one."""
    raw = base_doc()
    raw["method"]["save"][0]["aggregation"] = {
        "kind": "top_k",
        "k": 3,
        "by": "prob",
        "token_form": "id",
    }
    with pytest.raises(ParseError) as err:
        parse_document(in_order(raw))
    assert err.value.code == "P3"


def test_top_k_needs_a_ranking_rule():
    """``by`` is mandatory: only the author knows whether the read's axis has
    meaningful negative entries, and guessing changes the answer."""
    raw = base_doc()
    raw["method"]["save"].append(
        saved("logits", "patched", "tk.json", {"kind": "top_k", "k": 3})
    )
    with pytest.raises(ParseError) as err:
        parse_document(in_order(raw))
    assert err.value.code == "P2"
    assert "by" in str(err.value)


@pytest.mark.parametrize("by", ["value", "abs_value", "prob"])
def test_top_k_ranking_vocabulary(by):
    raw = base_doc()
    raw["method"]["save"].append(
        saved("logits", "patched", "tk.json", {"kind": "top_k", "k": 3, "by": by})
    )
    doc = parse_document(in_order(raw))
    assert by_label(doc)["tk"].fields["by"] == by


def test_top_k_ranking_is_a_closed_enum():
    raw = base_doc()
    raw["method"]["save"].append(
        saved(
            "logits", "patched", "tk.json", {"kind": "top_k", "k": 3, "by": "softmax"}
        )
    )
    with pytest.raises(ParseError) as err:
        parse_document(in_order(raw))
    assert err.value.code == "P4"


def test_top_k_ranking_is_not_sweepable():
    """A sweep over ``by`` would fork a campaign on how a plot is read rather
    than on a research variable — the reasoning that keeps ``token_form`` off
    §3's wrappers."""
    raw = base_doc()
    raw["method"]["save"].append(
        saved(
            "logits",
            "patched",
            "tk.json",
            {"kind": "top_k", "k": 3, "by": {"sweep": ["value", "abs_value"]}},
        )
    )
    with pytest.raises(ValidationError) as err:
        parse_document(in_order(raw))
    assert err.value.rule == 14


def test_metric_token_form_is_not_sweepable():
    """A sweep over token_form would fork a campaign on a tokenizer detail
    rather than a research variable; §3 wrappers stay off this field."""
    raw = base_doc()
    _ld(raw)["token_form"] = {"sweep": ["id"]}
    with pytest.raises(ValidationError) as err:
        parse_document(in_order(raw))
    assert err.value.rule == 14


# the continuation frame (§2.3) ---------------------------------------------- #


def _generated(anchor: dict, budget: int = 8) -> dict:
    return {"generated": {"max_new_tokens": budget}, **anchor}


@pytest.mark.parametrize(
    "anchor,attr,expected",
    [
        ({"all": True}, "all", True),
        ({"index": -1}, "index", -1),
        ({"span": [0, 3]}, "span", (0, 3)),
        ({"variable": "expected"}, "variable", "expected"),
    ],
)
def test_generated_frame_takes_every_anchor(anchor, attr, expected):
    """``generated`` is a frame selector, not an anchor: the anchor
    vocabulary is unchanged inside the continuation."""
    raw = base_doc()
    raw["method"]["reads"]["v_cf"]["pos"] = _generated(anchor)
    pos = parse_document(raw).reads["v_cf"].pos
    assert isinstance(pos, PositionSpec)
    assert getattr(pos, attr) == expected
    assert pos.generated == {"max_new_tokens": 8}


def test_generated_needs_an_anchor():
    raw = base_doc()
    raw["method"]["reads"]["v_cf"]["pos"] = {"generated": {"max_new_tokens": 8}}
    with pytest.raises(ParseError) as err:
        parse_document(raw)
    assert "exactly one" in str(err.value)


@pytest.mark.parametrize(
    "extra",
    [
        {"column": "entity"},
        {"index": 0, "scope": {"variable": "subject"}},
        {"index": 1, "relative_to": {"variable": "subject"}},
    ],
)
def test_generated_refuses_prompt_frame_notions(extra):
    """A ``column`` holds a substring of the *input* text and
    ``scope``/``relative_to`` anchor on a prompt variable's token run —
    neither exists in a frame the prompt does not contain."""
    raw = base_doc()
    raw["method"]["reads"]["v_cf"]["pos"] = {
        "generated": {"max_new_tokens": 8},
        **extra,
    }
    with pytest.raises(ParseError) as err:
        parse_document(raw)
    assert "generated" in str(err.value)


@pytest.mark.parametrize(
    "budget,fragment",
    [
        ({"max_new_tokens": 0}, "at least"),
        ({"max_new_tokens": -3}, "at least"),
        ({}, "max_new_tokens"),
        ({"max_new_tokens": 4, "temperature": 0.7}, "temperature"),
        (8, "expected an object"),
    ],
)
def test_generated_budget_shapes(budget, fragment):
    """The budget is a mapping so stopping conditions can join it later —
    a bare int, a missing budget, and sampling knobs all refuse."""
    raw = base_doc()
    raw["method"]["reads"]["v_cf"]["pos"] = {"generated": budget, "index": -1}
    with pytest.raises(ParseError) as err:
        parse_document(raw)
    assert fragment in str(err.value)


def test_generated_budget_is_sweepable():
    raw = base_doc()
    raw["method"]["positions"] = {
        "tail": {"generated": {"max_new_tokens": {"sweep": [4, 16]}}, "index": -1}
    }
    raw["method"]["reads"]["v_cf"]["pos"] = "tail"
    pos = parse_document(in_order(raw)).positions["tail"]
    assert isinstance(pos, PositionSpec)
    assert isinstance(pos.generated["max_new_tokens"], Sweep)


def test_prompt_frame_positions_carry_no_generated():
    """Every pre-existing position stays prompt-frame: the field is absent,
    not defaulted, so existing canonical forms are untouched."""
    pos = parse_document(base_doc()).reads["v_cf"].pos
    assert isinstance(pos, PositionSpec) and pos.generated is None


def test_decode_metric_takes_no_value_fields():
    raw = base_doc()
    raw["method"]["save"][0]["aggregation"] = {"kind": "decode"}
    metric = by_label(parse_document(in_order(raw)))["ld"]
    assert str(metric.kind) == "decode"
    assert dict(metric.fields) == {}


def test_decode_metric_rejects_a_stray_field():
    """The kind reduces the tokens a decode produced; there is nothing to
    parametrize, so an extra key is a typo, not an option."""
    raw = base_doc()
    raw["method"]["save"][0]["aggregation"] = {"kind": "decode", "k": 1}
    with pytest.raises(ParseError):
        parse_document(in_order(raw))


# --------------------------------------------------------------------- #
# `init` on a subspace (§2.5)
# --------------------------------------------------------------------- #


def _subspace_doc(**featurizer):
    """``base_doc`` with a trained-shape ``subspace`` on its patch."""
    raw = base_doc()
    raw["method"]["featurizers"] = {
        "rot": {"kind": "subspace", "k": 2, "parametrization": "cayley", **featurizer}
    }
    raw["method"]["reads"]["v_cf"]["featurizer"] = "rot"
    raw["method"]["writes"]["patch"]["featurizer"] = "rot"
    return raw


def test_subspace_init_parses_the_object_form():
    raw = _subspace_doc(init={"file_path": "pca/basis.safetensors"})
    doc = parse_document(in_order(raw))
    assert doc.featurizers["rot"].init == {"file_path": "pca/basis.safetensors"}


def test_subspace_init_keeps_an_authored_entry():
    raw = _subspace_doc(init={"file_path": "pca/basis.safetensors", "entry": {"k": 32}})
    doc = parse_document(in_order(raw))
    assert doc.featurizers["rot"].init == {
        "file_path": "pca/basis.safetensors",
        "entry": {"k": 32},
    }


def test_subspace_init_refuses_the_string_form():
    """The field used to parse as an unused scalar string. Nothing ever wrote
    one, and a bare path is not a selection — the object form is the only
    spelling."""
    with pytest.raises(ParseError) as err:
        parse_document(in_order(_subspace_doc(init="pca/basis.safetensors")))
    assert err.value.code == "P2"


def test_subspace_init_needs_a_file_path():
    with pytest.raises(ParseError, match="needs a file_path"):
        parse_document(in_order(_subspace_doc(init={"entry": {"k": 32}})))


def test_subspace_init_refuses_an_unknown_key():
    """Closed like every other object: the basis's rank is its own, read from
    its stamp, never authored here."""
    raw = _subspace_doc(init={"file_path": "b.safetensors", "k": 4})
    with pytest.raises(ParseError) as err:
        parse_document(in_order(raw))
    assert err.value.code == "P3"


def test_subspace_init_and_file_path_are_exclusive():
    """A loaded featurizer has its weights in the file: it draws nothing and
    trains nothing, so there is no start to set — the same rule `seed` already
    follows."""
    raw = _subspace_doc(
        init={"file_path": "pca/basis.safetensors"}, file_path="rot.safetensors"
    )
    with pytest.raises(ParseError, match="loaded featurizer uses the weights"):
        parse_document(in_order(raw))


def test_subspace_init_and_seed_are_legal_together():
    """The seed still has work with an init: it completes the basis and
    orders the batches."""
    raw = _subspace_doc(init={"file_path": "pca/basis.safetensors"}, seed=3)
    doc = parse_document(in_order(raw))
    assert doc.featurizers["rot"].seed == 3
    assert doc.featurizers["rot"].init == {"file_path": "pca/basis.safetensors"}


def test_init_is_a_field_of_the_kinds_that_have_a_start():
    """A `subspace` starts from a basis and a `gate` from a fill or a saved
    theta (§2.5); a `pca`, an `sae` or an `identity` is loaded or fixed, never
    fitted, so it has no start to set and the key is refused as unknown."""
    for kind, extra in (("pca", {"k": 4}), ("sae", {}), ("identity", {})):
        raw = base_doc()
        raw["method"]["featurizers"] = {
            "f": {"kind": kind, "init": {"file_path": "b.safetensors"}, **extra}
        }
        with pytest.raises(ParseError) as err:
            parse_document(in_order(raw))
        assert err.value.code == "P3", kind


# §2.10 token_logits — literal token strings, one list for the run ----------- #


def _token_logits(**overrides: object) -> dict:
    spec: dict = {
        "kind": "token_logits",
        "tokens": ["Monday", "Friday"],
    }
    spec.update(overrides)
    return spec


def _answers(raw: dict, spec: dict) -> None:
    """Save ``spec`` over ``patched``'s logits under the label ``answers``."""
    raw["method"]["save"].append(saved("logits", "patched", "answers.json", spec))


def test_token_logits_parses_its_token_list():
    raw = base_doc()
    _answers(raw, _token_logits())
    metric = by_label(parse_document(in_order(raw)))["answers"]
    assert metric.fields["tokens"] == ("Monday", "Friday")
    assert metric.token_form is None


def test_token_logits_refuses_id():
    """A literal token list is strings by construction; there is no column
    of ids for ``id`` to name."""
    raw = base_doc()
    _answers(raw, _token_logits(token_form="id"))
    with pytest.raises(ParseError) as err:
        parse_document(in_order(raw))
    assert err.value.code == "P2"
    assert "token_form='id'" in str(err.value)


def test_token_logits_refuses_an_empty_list():
    raw = base_doc()
    _answers(raw, _token_logits(tokens=[]))
    with pytest.raises(ParseError) as err:
        parse_document(in_order(raw))
    assert err.value.code == "P2"


def test_token_logits_refuses_a_non_string_entry():
    raw = base_doc()
    _answers(raw, _token_logits(tokens=["Monday", 7]))
    with pytest.raises(ParseError) as err:
        parse_document(in_order(raw))
    assert err.value.code == "P2"


def test_token_logits_lists_both_spellings_as_two_answers():
    """``["X", " X"]`` is two answers: each string is tokenized as written,
    and on a byte-level BPE they are two rows. Whether a *tokenizer* folds
    them onto one id is checked where the metric resolves them."""
    raw = base_doc()
    _answers(raw, _token_logits(tokens=["Monday", " Monday"]))
    metric = by_label(parse_document(in_order(raw)))["answers"]
    assert metric.fields["tokens"] == ("Monday", " Monday")


def test_token_logits_refuses_an_answer_listed_twice():
    """A string repeated letter for letter would carry one logit under two
    names. Caught torch-free, at parse."""
    raw = base_doc()
    _answers(raw, _token_logits(tokens=["Monday", "Monday"]))
    with pytest.raises(ParseError) as err:
        parse_document(in_order(raw))
    assert err.value.code == "P2"
    assert "twice" in str(err.value)


def test_token_logits_refuses_an_empty_entry():
    """An empty string names no token. A lone space is a token (the one a
    model emits before a digit on gpt2), so it is allowed."""
    raw = base_doc()
    _answers(raw, _token_logits(tokens=["Monday", ""]))
    with pytest.raises(ParseError) as err:
        parse_document(in_order(raw))
    assert err.value.code == "P2"
    assert "empty string" in str(err.value)
    raw["method"]["save"][-1]["aggregation"] = _token_logits(tokens=["Monday", " "])
    metric = by_label(parse_document(in_order(raw)))["answers"]
    assert metric.fields["tokens"] == ("Monday", " ")


def test_token_logits_tokens_are_not_sweepable():
    """The answer space is fixed per campaign, like ``token_form`` and
    ``top_k.by``: a sweep over it would fork the campaign on what gets saved
    rather than on a research variable."""
    raw = base_doc()
    _answers(raw, _token_logits(tokens={"sweep": [["Monday"], ["Friday"]]}))
    with pytest.raises(ValidationError) as err:
        parse_document(in_order(raw))
    assert err.value.rule == 14
    # the aggregation lives on the entry that saves it (§2.10)
    assert err.value.path == "save[1].aggregation.tokens"


def test_token_logits_rejects_a_column_style_field_with_a_suggestion():
    """``token`` (singular) is ``token_logit``'s column field; on this kind it
    is a typo for ``tokens``, and the closed-key check says so."""
    raw = base_doc()
    spec = _token_logits()
    spec["token"] = spec.pop("tokens")
    _answers(raw, spec)
    with pytest.raises(ParseError) as err:
        parse_document(in_order(raw))
    assert err.value.code == "P3"
    assert "tokens" in str(err.value)


# §2.5 gate `group` — a closed enum, legal on the gate alone ------------------ #


def _gated(featurizer: dict) -> dict:
    raw = base_doc()
    raw["method"]["featurizers"] = {"gate": featurizer}
    raw["method"]["reads"]["v_cf"]["featurizer"] = "gate"
    raw["method"]["writes"]["patch"]["featurizer"] = "gate"
    return in_order(raw)


def test_gate_group_parses_and_defaults_to_none():
    doc = parse_document(_gated({"kind": "gate", "group": "head"}))
    assert doc.featurizers["gate"].group == "head"
    assert parse_document(_gated({"kind": "gate"})).featurizers["gate"].group is None


def test_gate_group_is_a_closed_enum_with_suggestions():
    with pytest.raises(ParseError) as err:
        parse_document(_gated({"kind": "gate", "group": "heads"}))
    assert err.value.code == "P4"
    assert "'head'" in str(err.value)


def test_group_is_not_a_field_of_any_other_kind():
    """Only a gate has a parameter per unit to share; on a rotation the key
    is unknown, and refused as one."""
    with pytest.raises(ParseError) as err:
        parse_document(
            _gated(
                {
                    "kind": "subspace",
                    "k": 4,
                    "parametrization": "cayley",
                    "group": "head",
                }
            )
        )
    assert err.value.code == "P3"


# --------------------------------------------------------------------------- #
# train.objective: regularizers over several featurizers, and the named form
# --------------------------------------------------------------------------- #


def _two_gate_train(objective):
    raw = base_doc()
    raw["method"]["sites"]["tgt2"] = {"component": "block_output", "layers": [2]}
    raw["method"]["featurizers"] = {"g0": {"kind": "gate"}, "g1": {"kind": "gate"}}
    raw["method"]["reads"]["v_cf"]["featurizer"] = "g0"
    raw["method"]["reads"]["v2"] = {"site": "tgt2", "pos": -1, "featurizer": "g1"}
    raw["method"]["intervened_models"][UNWRITTEN]["reads"].append("v2")
    raw["method"]["writes"]["patch"]["featurizer"] = "g0"
    raw["method"]["writes"]["patch2"] = {
        "site": "tgt2",
        "pos": -1,
        "featurizer": "g1",
        "do": {"swap": "v2"},
    }
    raw["method"]["intervened_models"]["patched"]["writes"] = ["patch", "patch2"]
    raw["method"]["train"] = {
        "objective": objective,
        "params": ["g0", "g1"],
        "optimizer": {"name": "adamw", "lr": 1e-2},
        "steps": {"epochs": 1},
        "batch": {"pairs": 2},
    }
    raw["method"]["save"] += [
        {"value": "g0", "site": "tgt", "file_path": "g0.safetensors"},
        {"value": "g1", "site": "tgt2", "file_path": "g1.safetensors"},
    ]
    return in_order(raw)


def test_a_regularizer_may_name_a_list_of_featurizers():
    doc = parse_document(_two_gate_train([[1.0, _fit()], [0.01, {"l1": ["g1", "g0"]}]]))
    assert doc.train is not None
    (_, sparsity) = doc.train.objective
    assert sparsity.weight == 0.01 and sparsity.aggregation is None
    assert sparsity.regularizer == (
        "l1",
        ("g1", "g0"),
    )  # authored order; canonical sorts
    assert sparsity.name is None and sparsity.path(1) == "train.objective[1]"


def test_a_single_name_regularizer_is_a_one_name_list():
    doc = parse_document(_two_gate_train([[1.0, _fit()], [0.5, {"l2": "g0"}]]))
    assert doc.train is not None
    assert doc.train.objective[1].regularizer == ("l2", ("g0",))


def test_a_regularizer_may_declare_its_reduction_in_both_spellings():
    """§2.11 ``reduce``: ``mean`` (the unspelled default) or ``sum`` over the
    concatenated per-unit quantities; ``None`` when unauthored so the canonical
    form materializes nothing."""
    positional = parse_document(
        _two_gate_train([[1.0, _fit()], [0.01, {"l1": ["g0", "g1"], "reduce": "sum"}]])
    )
    assert positional.train is not None
    assert positional.train.objective[1].reduce == "sum"
    assert positional.train.objective[1].regularizer == ("l1", ("g0", "g1"))
    named = parse_document(
        _two_gate_train(
            {
                "fit": _fit(weight=1.0),
                "sparsity": {"weight": 0.01, "l1": ["g0", "g1"], "reduce": "sum"},
            }
        )
    )
    assert named.train is not None
    assert named.train.objective[1].reduce == "sum"
    plain = parse_document(_two_gate_train([[1.0, _fit()], [0.01, {"l1": "g0"}]]))
    assert plain.train is not None and plain.train.objective[1].reduce is None


def test_an_unknown_reduction_and_a_reduction_on_a_metric_are_refused():
    with pytest.raises(ParseError) as err:
        parse_document(
            _two_gate_train([[1.0, _fit()], [0.01, {"l1": "g0", "reduce": "max"}]])
        )
    assert err.value.code == "P4" and err.value.path == "train.objective[1][1].reduce"
    with pytest.raises(ParseError) as err:
        parse_document(
            _two_gate_train(
                {
                    "fit": _fit(weight=1.0, reduce="sum"),
                    "sparsity": {"weight": 0.01, "l1": "g0"},
                }
            )
        )
    assert err.value.code == "P2" and err.value.path == "train.objective.fit.reduce"


def test_a_regularizer_may_declare_per_target_costs_in_both_spellings():
    """§2.11 ``costs``: a table ``{target: c}`` or the word
    ``parameter_count``; ``None`` when unauthored so the canonical form
    materializes nothing. Whether the keys are the term's targets is rule 4."""
    positional = parse_document(
        _two_gate_train(
            [
                [1.0, _fit()],
                [0.01, {"l1": ["g0", "g1"], "reduce": "sum", "costs": {"g1": 0.25}}],
            ]
        )
    )
    assert positional.train is not None
    assert positional.train.objective[1].costs == {"g1": 0.25}
    assert positional.train.objective[1].reduce == "sum"
    named = parse_document(
        _two_gate_train(
            {
                "fit": _fit(weight=1.0),
                "sparsity": {
                    "weight": 0.01,
                    "l1": ["g0", "g1"],
                    "costs": "parameter_count",
                },
            }
        )
    )
    assert named.train is not None
    assert named.train.objective[1].costs == "parameter_count"
    assert named.train.objective[1].reduce is None
    plain = parse_document(_two_gate_train([[1.0, _fit()], [0.01, {"l1": "g0"}]]))
    assert plain.train is not None and plain.train.objective[1].costs is None
    # an integer cost is a float, so `1` and `1.0` are one declaration
    whole = parse_document(
        _two_gate_train([[1.0, _fit()], [0.01, {"l1": "g0", "costs": {"g0": 2}}]])
    )
    assert whole.train is not None and whole.train.objective[1].costs == {"g0": 2.0}


@pytest.mark.parametrize(
    "costs, code, leaf",
    [
        ({}, "P2", ""),
        ({"g0": 0}, "P2", ".g0"),
        ({"g0": -1.0}, "P2", ".g0"),
        ({"g0": True}, "P2", ".g0"),
        ({"g0": "cheap"}, "P2", ".g0"),
        ({"g0": float("inf")}, "P2", ".g0"),
        (
            {"g0": 10**400},
            "P2",
            ".g0",
        ),  # an int too large for a double: P2, not OverflowError
        ({"g0": {"sweep": [0.25, 0.5]}}, "P2", ".g0"),  # a cost is not swept
        ({"sweep": [0.25, 0.5]}, "P2", ""),  # nor is the table
        ({1: 0.5}, "P2", ""),
        ("per_unit", "P4", ""),
        ([1.0], "P2", ""),
    ],
)
def test_a_malformed_costs_field_is_refused_naming_the_path(costs, code, leaf):
    with pytest.raises(ParseError) as err:
        parse_document(
            _two_gate_train([[1.0, _fit()], [0.01, {"l1": "g0", "costs": costs}]])
        )
    assert err.value.code == code
    assert err.value.path == f"train.objective[1][1].costs{leaf}"


def _constraint(**over):
    spec = {"target": 0.1, "dual": {"lr": 0.05}}
    spec.update(over)
    return spec


def test_a_named_mask_term_may_carry_a_lagrangian_constraint_instead_of_a_weight():
    """§2.11 ``constraint``: a target density with a dual pair; the term has
    no weight (``None``), ``dual.init`` is ``None`` when unauthored so the
    canonical form materializes nothing, and ``init`` reads as (0, 0)."""
    doc = parse_document(
        _two_gate_train(
            {
                "fit": _fit(weight=1.0),
                "density": {"l1": ["g0", "g1"], "constraint": _constraint()},
            }
        )
    )
    assert doc.train is not None
    density = doc.train.objective[1]
    assert density.weight is None and density.name == "density"
    assert density.regularizer == ("l1", ("g0", "g1"))
    assert density.constraint is not None
    assert density.constraint.target == 0.1 and density.constraint.dual_lr == 0.05
    assert density.constraint.dual_init is None and density.constraint.init == (
        0.0,
        0.0,
    )
    with_init = parse_document(
        _two_gate_train(
            {
                "fit": _fit(weight=1.0),
                "density": {
                    "l1": "g0",
                    "reduce": "mean",
                    "constraint": _constraint(dual={"lr": 0.05, "init": [1, 0.5]}),
                },
            }
        )
    )
    assert with_init.train is not None
    assert with_init.train.objective[1].constraint.dual_init == (1.0, 0.5)
    # λ₁ ranges over ℝ (the fit drives it negative itself), so a warm start
    # below zero is legal; only λ₂ is held non-negative
    warm = parse_document(
        _two_gate_train(
            {
                "fit": _fit(weight=1.0),
                "density": {
                    "l1": "g0",
                    "constraint": _constraint(dual={"lr": 0.05, "init": [-2, 0.5]}),
                },
            }
        )
    )
    assert warm.train is not None
    assert warm.train.objective[1].constraint.dual_init == (-2.0, 0.5)
    # a costs *table* composes: the target is held on the cost-weighted density
    costed = parse_document(
        _two_gate_train(
            {
                "fit": _fit(weight=1.0),
                "density": {
                    "l1": ["g0", "g1"],
                    "costs": {"g1": 0.5},
                    "constraint": _constraint(),
                },
            }
        )
    )
    assert costed.train is not None
    assert costed.train.objective[1].costs == {"g1": 0.5}
    assert costed.train.objective[1].constraint is not None
    # every other term keeps its weight and no constraint
    assert (
        doc.train.objective[0].constraint is None
        and doc.train.objective[0].weight == 1.0
    )


@pytest.mark.parametrize(
    "term, leaf",
    [
        ({"weight": 0.01, "l1": "g0", "constraint": _constraint()}, ".weight"),
        ({"l1": "g0"}, ""),  # no weight and no constraint
        (_fit(weight=1.0, constraint=_constraint()), ".constraint"),
        ({"l2": "g0", "constraint": _constraint()}, ".constraint"),
        ({"l1": "g0", "reduce": "sum", "constraint": _constraint()}, ".reduce"),
        # `parameter_count` divides the density by N: no longer a fraction
        (
            {"l1": "g0", "costs": "parameter_count", "constraint": _constraint()},
            ".costs",
        ),
        ({"l1": "g0", "constraint": _constraint(target=0.0)}, ".constraint.target"),
        ({"l1": "g0", "constraint": _constraint(target=1.0)}, ".constraint.target"),
        (
            {"l1": "g0", "constraint": _constraint(dual={"lr": 0})},
            ".constraint.dual.lr",
        ),
        (
            {"l1": "g0", "constraint": _constraint(dual={"lr": -1.0})},
            ".constraint.dual.lr",
        ),
        (
            {"l1": "g0", "constraint": _constraint(dual={"init": [0, 0]})},
            ".constraint.dual",
        ),
        (
            {"l1": "g0", "constraint": _constraint(dual={"lr": 0.1, "init": [0.0]})},
            ".constraint.dual.init",
        ),
        (
            # λ₂ is the quadratic penalty's coefficient; negative, the term is
            # concave and the gate is driven away from the target
            {"l1": "g0", "constraint": _constraint(dual={"lr": 0.1, "init": [0, -5]})},
            ".constraint.dual.init",
        ),
        ({"l1": "g0", "constraint": {"target": 0.1}}, ".constraint"),
        ({"l1": "g0", "constraint": _constraint(rho=1.0)}, ".constraint"),
    ],
)
def test_a_malformed_constraint_term_is_refused_naming_the_path(term, leaf):
    with pytest.raises(ParseError) as err:
        parse_document(_two_gate_train({"fit": _fit(weight=1.0), "density": term}))
    # an unknown key inside the block is the parser's strict-keys code (P3)
    assert err.value.code in ("P2", "P3"), err.value
    assert err.value.path == f"train.objective.density{leaf}", err.value.path


@pytest.mark.parametrize(
    "term, leaf",
    [
        ({"l1": "g0", "constraint": {"sweep": [_constraint()]}}, ".constraint"),
        (
            {"l1": "g0", "constraint": _constraint(target={"sweep": [0.05, 0.1]})},
            ".constraint.target",
        ),
        # §3.2's spelling, un-lowered: a document with no `axes` group reaches
        # the gate with the wrapper intact (in a real compile too)
        (
            {"l1": "g0", "constraint": _constraint(target={"axis": "t"})},
            ".constraint.target",
        ),
        (
            {"l1": "g0", "constraint": _constraint(dual={"sweep": [{"lr": 0.1}]})},
            ".constraint.dual",
        ),
        (
            {"l1": "g0", "constraint": _constraint(dual={"lr": {"sweep": [0.1, 0.5]}})},
            ".constraint.dual.lr",
        ),
        (
            {
                "l1": "g0",
                "constraint": _constraint(
                    dual={"lr": 0.1, "init": {"sweep": [[0, 0], [1, 1]]}}
                ),
            },
            ".constraint.dual.init",
        ),
    ],
)
def test_a_swept_constraint_field_is_refused_saying_it_is_not_swept(term, leaf):
    """One constraint, one target: a sweep wrapper anywhere in the block is
    refused *naming sweeping* — `_scalar_number` would call the wrapper a dict
    and `_check_keys` an unknown key, at the same path, which is why the code
    and the path alone would not pin this."""
    with pytest.raises(ParseError) as err:
        parse_document(_two_gate_train({"fit": _fit(weight=1.0), "density": term}))
    assert err.value.code == "P2", err.value
    assert err.value.path == f"train.objective.density{leaf}", err.value.path
    assert "not swept (nor an `axis`)" in str(err.value), err.value


def test_a_constraint_target_bound_to_a_declared_axis_is_refused_naming_axis():
    """The path a real compile takes with the likely authoring mistake — a
    density → score curve off a shared `axes` column: `_axes` lowers
    `{"axis": "t"}` to the sweep it stands for *before* the gate, so the
    guard matches on `sweep` and only the words "(nor an `axis`)" tell the
    author their axis reference was the refused thing. Both the lowering and
    the wording are pinned; trimming the parenthetical fails here."""
    from causalab.protocol.lowering import lower_axes, parse_axes

    raw = _two_gate_train(
        {
            "fit": _fit(weight=1.0),
            "density": {"l1": "g0", "constraint": _constraint(target={"axis": "t"})},
        }
    )
    raw["axes"] = {"t": {"values": [0.05, 0.1]}}
    lowered = lower_axes(raw, parse_axes(raw, lambda _key: None))  # type: ignore[arg-type,return-value]
    target = lowered["method"]["train"]["objective"]["density"]["constraint"]["target"]
    assert target == {"sweep": [0.05, 0.1]} and "axes" not in lowered
    with pytest.raises(ParseError) as err:
        parse_document(lowered)
    assert err.value.path == "train.objective.density.constraint.target"
    assert "not swept (nor an `axis`)" in str(err.value), err.value


def test_a_positional_constraint_is_refused_toward_the_named_form():
    with pytest.raises(ParseError) as err:
        parse_document(
            _two_gate_train(
                [[1.0, _fit()], [0.01, {"l1": "g0", "constraint": _constraint()}]]
            )
        )
    assert (
        err.value.code == "P2" and err.value.path == "train.objective[1][1].constraint"
    )
    assert "named form" in str(err.value)


def test_costs_on_a_metric_term_is_refused():
    with pytest.raises(ParseError) as err:
        parse_document(
            _two_gate_train(
                {
                    "fit": _fit(weight=1.0, costs={"g0": 1.0}),
                    "sparsity": {"weight": 0.01, "l1": "g0"},
                }
            )
        )
    assert err.value.code == "P2" and err.value.path == "train.objective.fit.costs"


@pytest.mark.parametrize("names", [[], ["g0", "g0"], ["g0", 1]])
def test_an_empty_repeated_or_non_string_regularizer_list_is_refused(names):
    with pytest.raises(ParseError) as err:
        parse_document(_two_gate_train([[1.0, _fit()], [0.01, {"l1": names}]]))
    assert err.value.code == "P2"
    assert err.value.path == "train.objective[1][1].l1"


def _two_gate_train_with_optimizer(optimizer):
    raw = _two_gate_train([[1.0, _fit()], [0.01, {"l1": ["g0", "g1"]}]])
    raw["method"]["train"]["optimizer"] = optimizer
    return raw


def test_lr_and_weight_decay_may_be_keyed_by_the_trained_params():
    """§2.11: a mapping over the entries of ``train.params`` gives each its own
    step size — the case is a rotation at 1e-3 beside a gate at 0.1 in one fit,
    which one scalar cannot express."""
    doc = parse_document(
        _two_gate_train_with_optimizer(
            {
                "name": "adamw",
                "lr": {"g0": 1e-3, "g1": 0.1},
                "weight_decay": {"g0": 0.0, "g1": 0.01},
            }
        )
    )
    assert doc.train is not None
    assert doc.train.optimizer["lr"] == {"g0": 1e-3, "g1": 0.1}
    assert doc.train.optimizer["weight_decay"] == {"g0": 0.0, "g1": 0.01}


def test_a_per_params_lr_naming_an_untrained_featurizer_is_refused():
    with pytest.raises(ParseError) as err:
        parse_document(
            _two_gate_train_with_optimizer(
                {"name": "adamw", "lr": {"g0": 1e-3, "g1": 0.1, "g2": 0.5}}
            )
        )
    assert err.value.code == "P2"
    assert err.value.path == "train.optimizer.lr"
    assert "g2" in str(err.value)


def test_a_per_params_lr_leaving_a_trained_param_without_a_value_is_refused():
    """No hidden default: an entry of ``params`` the mapping omits would be
    stepped at *some* rate the document never wrote."""
    with pytest.raises(ParseError) as err:
        parse_document(
            _two_gate_train_with_optimizer({"name": "adamw", "lr": {"g0": 1e-3}})
        )
    assert err.value.code == "P2"
    assert err.value.path == "train.optimizer.lr"
    assert "g1" in str(err.value)


def test_a_per_params_lr_value_must_be_a_number():
    with pytest.raises(ParseError) as err:
        parse_document(
            _two_gate_train_with_optimizer(
                {"name": "adamw", "lr": {"g0": 1e-3, "g1": "fast"}}
            )
        )
    assert err.value.code == "P2"
    assert err.value.path == "train.optimizer.lr.g1"


def test_a_regularizer_kind_is_l1_or_l2_with_a_suggestion():
    with pytest.raises(ParseError) as err:
        parse_document(_two_gate_train([[1.0, _fit()], [0.01, {"L1": ["g0", "g1"]}]]))
    assert err.value.code == "P2" and "l1" in str(err.value)


def test_the_named_objective_form_parses_to_the_same_terms():
    named = {
        "fit": _fit(weight=1.0),
        "sparsity": {"weight": 0.01, "l1": ["g0", "g1"]},
    }
    doc = parse_document(_two_gate_train(named))
    assert doc.train is not None
    fit, sparsity = doc.train.objective
    assert (fit.weight, fit.name) == (1.0, "fit")
    assert fit.aggregation is not None and fit.aggregation.kind == "logit_diff"
    assert sparsity.regularizer == ("l1", ("g0", "g1"))
    assert sparsity.path(1) == "train.objective.sparsity"


def test_a_named_term_needs_a_weight_and_exactly_one_kind():
    with pytest.raises(ParseError) as err:
        parse_document(_two_gate_train({"fit": _fit()}))
    assert err.value.path == "train.objective.fit" and "weight" in str(err.value)
    with pytest.raises(ParseError) as err:
        parse_document(_two_gate_train({"both": _fit(weight=1.0, l1="g0")}))
    assert "exactly one" in str(err.value)
    # protocol 3's `metric` key is refused by name, toward `causalab migrate`
    with pytest.raises(ParseError) as err:
        parse_document(_two_gate_train({"fit": {"weight": 1.0, "metric": "ld"}}))
    assert err.value.code == "P3" and err.value.path == "train.objective.fit.metric"


def test_a_named_terms_weight_may_be_swept_where_a_positional_one_may_not():
    named = {
        "fit": _fit(weight=1.0),
        "sparsity": {"weight": {"sweep": [0.01, 0.1]}, "l1": ["g0", "g1"]},
    }
    doc = parse_document(_two_gate_train(named))
    assert doc.train is not None
    assert isinstance(doc.train.objective[1].weight, Sweep)
    positional = _two_gate_train(
        [[1.0, _fit()], [{"sweep": [0.01, 0.1]}, {"l1": ["g0", "g1"]}]]
    )
    parse_document(positional)  # the shape gate passes; the axis has no name
    with pytest.raises(ValidationError) as err:
        find_axes(positional)
    assert err.value.rule == 14


@pytest.mark.parametrize("objective", [[], {}, {"sweep": [[[1.0, _fit()]]]}])
def test_an_empty_or_wholly_swept_objective_is_refused(objective):
    with pytest.raises(ParseError) as err:
        parse_document(_two_gate_train(objective))
    assert err.value.path == "train.objective"


# --------------------------------------------------------------------------- #
# §2.2 `shuffle: {seed}` — the shuffled_source control's one data verb
# --------------------------------------------------------------------------- #


def _shuffled(shuffle, *, role="counterfactual", listed=False):
    """``base_doc`` with ``shuffle`` on ``role`` (``counterfactual[1]`` when
    ``listed``)."""
    doc = base_doc()
    if listed:
        cf = doc["data"]["counterfactual"]
        doc["data"]["counterfactual"] = [dict(cf), {**cf, "shuffle": shuffle}]
        doc["method"]["intervened_models"][UNWRITTEN]["input"] = "counterfactual[1]"
    else:
        doc["data"][role] = {**doc["data"][role], "shuffle": shuffle}
    return in_order(doc)


def _mapped_gate(parametrization, **extra):
    doc = base_doc()
    doc["method"]["featurizers"] = {
        "g": {"kind": "gate", "parametrization": parametrization, **extra}
    }
    doc["method"]["reads"]["v_cf"]["featurizer"] = "g"
    doc["method"]["writes"]["patch"]["featurizer"] = "g"
    return in_order(doc)


def test_the_mapping_form_of_parametrization_splits_forward_and_backward():
    """§2.5: ``{"forward": "hard", "backward": <map>}`` parses to the map on
    ``parametrization`` and ``"hard"`` on ``forward``; the string form leaves
    ``forward`` ``None``."""
    doc = parse_document(_mapped_gate({"forward": "hard", "backward": "sigmoid"}))
    assert doc.featurizers["g"].parametrization == "sigmoid"
    assert doc.featurizers["g"].forward == "hard"
    doc = parse_document(
        _mapped_gate(
            {"forward": "hard", "backward": "budget"},
            k_schedule={"kind": "fixed", "k": 2},
        )
    )
    assert doc.featurizers["g"].parametrization == "budget"
    plain = parse_document(_mapped_gate("clamp"))
    assert (
        plain.featurizers["g"].parametrization == "clamp"
        and plain.featurizers["g"].forward is None
    )


@pytest.mark.parametrize(
    "parametrization, code, leaf",
    [
        ({"forward": "hard"}, "P2", ""),
        ({"backward": "sigmoid"}, "P2", ""),
        ({"forward": "soft", "backward": "sigmoid"}, "P2", ".forward"),
        ({"forward": "sampled", "backward": "hard_concrete"}, "P2", ".forward"),
        ({"forward": "ste", "backward": "sigmoid"}, "P4", ".forward"),
        ({"forward": "hard", "backward": "cayley"}, "P4", ".backward"),
        ({"forward": "hard", "backward": "sigmoid", "temperature": 1.0}, "P3", ""),
    ],
)
def test_a_malformed_mapping_form_is_refused_naming_the_path(
    parametrization, code, leaf
):
    with pytest.raises(ParseError) as err:
        parse_document(_mapped_gate(parametrization))
    assert err.value.code == code, err.value
    assert err.value.path == f"featurizers.g.parametrization{leaf}"


def test_the_mapping_form_is_for_gates_only():
    raw = base_doc()
    raw["method"]["featurizers"] = {
        "rot": {
            "kind": "subspace",
            "k": 4,
            "parametrization": {"forward": "hard", "backward": "cayley"},
        }
    }
    raw["method"]["reads"]["v_cf"]["featurizer"] = "rot"
    raw["method"]["writes"]["patch"]["featurizer"] = "rot"
    with pytest.raises(ParseError) as err:
        parse_document(in_order(raw))
    assert err.value.code == "P2" and "rotation map" in str(err.value)


def test_the_mapping_form_of_parametrization_is_not_swept_and_says_so():
    """A sweep arm that is the mapping form is refused naming the rule, not
    with `_enum`'s unknown-value message against a stringified dict."""
    doc = base_doc()
    doc["method"]["featurizers"] = {
        "gate": {
            "kind": "gate",
            "parametrization": {
                "sweep": ["sigmoid", {"forward": "hard", "backward": "sigmoid"}]
            },
        }
    }
    doc["method"]["writes"]["patch"]["featurizer"] = "gate"
    with pytest.raises(ParseError) as err:
        parse_document(in_order(doc))
    assert err.value.code == "P2" and "not swept" in str(err.value)
    assert err.value.path.startswith("featurizers.gate.parametrization")
    # the dual: a swept `backward` inside the mapping form, same rule
    doc["method"]["featurizers"]["gate"]["parametrization"] = {
        "forward": "hard",
        "backward": {"sweep": ["sigmoid", "clamp"]},
    }
    with pytest.raises(ParseError) as err:
        parse_document(in_order(doc))
    assert err.value.code == "P2" and "not swept" in str(err.value)
    assert err.value.path == "featurizers.gate.parametrization.backward"


def _positional(gate: dict, *, pos=None):
    """``base_doc`` with ``gate`` at the target site on a ``span`` window
    (read and write alike), or on ``pos`` when given."""
    doc = base_doc()
    window = {"span": [0, 3]} if pos is None else pos
    doc["method"]["featurizers"] = {"pg": gate}
    doc["method"]["reads"]["v_cf"]["pos"] = window
    doc["method"]["writes"]["patch"]["pos"] = window
    doc["method"]["writes"]["patch"]["featurizer"] = "pg"
    return in_order(doc)


def test_axis_position_parses_and_the_default_has_no_spelling():
    """§2.5 ``axis``: ``"position"`` on a gate, ``None`` otherwise; as with
    ``group``, ``feature`` is not a value."""
    doc = parse_document(_positional({"kind": "gate", "axis": "position"}))
    assert doc.featurizers["pg"].axis == "position"
    plain = parse_document(_positional({"kind": "gate"}))
    assert plain.featurizers["pg"].axis is None
    with pytest.raises(ParseError) as err:
        parse_document(_positional({"kind": "gate", "axis": "feature"}))
    assert err.value.code == "P4" and err.value.path == "featurizers.pg.axis"


def test_axis_takes_no_group_and_no_pool():
    with pytest.raises(ParseError) as err:
        parse_document(
            _positional({"kind": "gate", "axis": "position", "group": "site"})
        )
    assert err.value.code == "P2" and err.value.path == "featurizers.pg.group"
    with pytest.raises(ParseError) as err:
        parse_document(
            _positional(
                {
                    "kind": "gate",
                    "axis": "position",
                    "parametrization": "budget",
                    "k_schedule": {"kind": "fixed", "k": 1},
                    "pool": "p",
                }
            )
        )
    assert err.value.code == "P2" and err.value.path == "featurizers.pg.pool"


def test_span_length_is_the_fixed_windows_size_or_none():
    from causalab.protocol.schema import PositionSpec, span_length

    assert span_length(PositionSpec(span=(0, 3))) == 3
    assert span_length(PositionSpec(span=(3, 3))) is None
    assert span_length(PositionSpec(span=(3, 4))) is None  # one position: not a window
    # row-dependent windows: sliced out of an anchor's run, clipped to the
    # row's decode — `b − a` is not the realized length on every row
    scoped = PositionSpec(span=(0, 3), scope=("variable", "subject"))
    assert span_length(scoped) is None
    assert span_length(PositionSpec(span=(-3, -1), scope=("variable", "s"))) is None
    relative = PositionSpec(span=(0, 3), relative_to=("variable", "subject"))
    assert span_length(relative) is None
    assert (
        span_length(PositionSpec(span=(0, 3), generated={"max_new_tokens": 4})) is None
    )
    assert span_length(PositionSpec(index=-1)) is None
    assert span_length(PositionSpec(all=True)) is None
    assert span_length("named") is None
    # `atomic` composes: it decides rule 8's write cardinality and §2.3's
    # alignment run, not the window — the one `SpanSpec` field that is not a
    # selector, and the twin below answers the same
    from causalab.protocol.schema.explicit import _window_length_raw
    from causalab.protocol.positions.spans import SpanSpec

    assert span_length(SpanSpec(span=(0, 3), atomic=True)) == 3
    assert _window_length_raw({"span": [0, 3], "atomic": True}, {}) == 3
    assert _window_length_raw({"span": [0, 3]}, {}) == 3
    assert _window_length_raw({"span": [0, 3], "scope": {"variable": "s"}}, {}) is None


def _drawn(draw, *, role="counterfactual", field="counterfactual_inputs"):
    """``base_doc`` with ``draw`` on ``role`` and its field set to ``field``."""
    doc = base_doc()
    doc["data"][role] = {**doc["data"][role], "field": field, "draw": draw}
    return in_order(doc)


def test_draw_parses_on_a_counterfactual_role_with_a_bare_list_field():
    """§2.2 ``draw``: the authored mapping, ``None`` unauthored; ``eval`` is
    the fixed member (0 unless authored) and the bare field is the column."""
    doc = parse_document(_drawn({"kind": "uniform"}))
    cf = doc.data["counterfactual"]
    assert cf.draw == {"kind": "uniform"} and cf.eval_member == 0
    assert cf.draw_column == "counterfactual_inputs"
    assert doc.data["base"].draw is None and doc.data["base"].draw_column is None
    doc = parse_document(_drawn({"kind": "uniform", "eval": 2}))
    assert doc.data["counterfactual"].eval_member == 2
    assert parse_document(in_order(base_doc())).data["counterfactual"].draw is None


def test_draw_is_refused_on_base_and_beside_an_indexed_field():
    with pytest.raises(ParseError) as err:
        parse_document(_drawn({"kind": "uniform"}, role="base", field="input"))
    assert err.value.code == "P2" and err.value.path == "data.base.draw"
    with pytest.raises(ParseError) as err:
        parse_document(_drawn({"kind": "uniform"}, field="counterfactual_inputs[0]"))
    assert err.value.code == "P2" and err.value.path == "data.counterfactual.field"
    assert "bare" in str(err.value)


@pytest.mark.parametrize(
    "draw, code, leaf",
    [
        ({}, "P2", ""),
        ({"kind": "gaussian"}, "P4", ".kind"),
        ({"kind": "uniform", "eval": -1}, "P2", ".eval"),
        ({"kind": "uniform", "eval": True}, "P2", ".eval"),
        ({"kind": "uniform", "eval": "0"}, "P2", ".eval"),
        ({"kind": "uniform", "seed": 3}, "P3", ""),
        (7, "P2", ""),
    ],
)
def test_a_malformed_draw_is_refused_naming_the_path(draw, code, leaf):
    with pytest.raises(ParseError) as err:
        parse_document(_drawn(draw))
    assert err.value.code == code, err.value
    assert err.value.path == f"data.counterfactual.draw{leaf}"


def test_draw_kind_is_not_sweepable():
    with pytest.raises(ValidationError) as err:
        parse_document(_drawn({"kind": {"sweep": ["uniform"]}}))
    assert err.value.rule == 14 and err.value.path == "data.counterfactual.draw.kind"


def test_shuffle_parses_on_a_counterfactual_role():
    """``DataRole.shuffle`` is the authored mapping; an unauthored role has
    ``None``. Fails without the change: ``[P3] unknown key 'shuffle'``."""
    doc = parse_document(_shuffled({"seed": 3}))
    assert doc.data["counterfactual"].shuffle == {"seed": 3}
    assert doc.data["base"].shuffle is None
    assert parse_document(in_order(base_doc())).data["counterfactual"].shuffle is None


def test_shuffle_parses_on_one_entry_of_a_list_valued_role():
    doc = parse_document(_shuffled({"seed": 3}, listed=True))
    first, second = doc.data["counterfactual"]
    assert first.shuffle is None and second.shuffle == {"seed": 3}


def test_shuffle_is_refused_on_base():
    """The base role is the population — never permuted. The mutation that
    accepts ``shuffle`` on ``base`` fails here."""
    with pytest.raises(ParseError) as err:
        parse_document(_shuffled({"seed": 3}, role="base"))
    assert err.value.code == "P2"
    assert err.value.path == "data.base.shuffle"
    assert "population" in str(err.value)


def test_a_role_less_data_block_is_the_base_role():
    """§2.2: role names earn their place only when a counterfactual is
    present, so a block naming none is ``base``. Fails without the change:
    ``[P2] data needs a 'base' role``."""
    short, explicit = base_only_doc(), base_only_doc()
    short["data"] = explicit["data"]["base"]
    assert parse_document(short).data == parse_document(explicit).data


def test_a_counterfactual_without_a_base_is_refused():
    """The author started naming roles and stopped: refused at ``data``, not
    wrapped into a base role that then fails on an unknown key."""
    doc = base_only_doc()
    doc["data"] = {
        **doc["data"]["base"],
        "counterfactual": {"dataset": "weekdays/data#train", "field": "input"},
    }
    with pytest.raises(ParseError) as err:
        parse_document(doc)
    assert err.value.code == "P2" and err.value.path == "data"
    assert "'base'" in str(err.value)


def test_a_role_less_data_block_keeps_strict_keys():
    """A typo in the short form is judged against the role's keys, at the
    role's path, so the message says what the block was read as."""
    doc = base_only_doc()
    doc["data"] = {"dataset": "weekdays/data#train", "input_string": ["x"]}
    with pytest.raises(ParseError) as err:
        parse_document(doc)
    assert err.value.code == "P3" and err.value.path == "data.base"
    assert "input_string" in str(err.value)


def test_an_inline_role_is_a_resolved_role():
    """§2.2: ``inputs`` arrives at every consumer as a dataset ref plus the
    fixed column, so nothing downstream tells the spellings apart. Fails
    without the change: ``[P3] unknown key 'inputs'``."""
    role = parse_document(inline_doc("a", "b")).data["base"]
    assert role.dataset == inline_ref(["a", "b"])
    assert (
        role.dataset.startswith("inline:") and len(role.dataset) == len("inline:") + 64
    )
    assert role.field == "input" and role.shuffle is None and role.draw is None


def test_an_inline_ref_is_content_addressed():
    assert inline_ref(["a", "b"]) == inline_ref(["a", "b"])
    assert inline_ref(["a", "b"]) != inline_ref(["b", "a"])


def test_an_inline_counterfactual_pairs_and_shuffles():
    doc = inline_doc()
    doc["data"] = {
        "base": doc["data"],
        "counterfactual": {
            "inputs": ["The Eiffel Tower is located in"],
            "shuffle": {"seed": 1},
        },
    }
    parsed = parse_document(doc).data
    assert parsed["counterfactual"].shuffle == {"seed": 1}
    assert parsed["counterfactual"].dataset != parsed["base"].dataset


@pytest.mark.parametrize(
    "role, code, where, fragment",
    [
        (
            {"inputs": ["x"], "dataset": "weekdays/data#train", "field": "input"},
            "P2",
            "data.base",
            "not both",
        ),
        ({"field": "input"}, "P2", "data.base", "not both and not neither"),
        ({"inputs": ["x"], "field": "input"}, "P2", "data.base.field", "drop 'field'"),
        (
            {"inputs": ["x"], "draw": {"kind": "uniform"}},
            "P2",
            "data.base.draw",
            "one prompt per row",
        ),
        ({"inputs": []}, "P2", "data.base.inputs", "at least one"),
        ({"inputs": ["x", ""]}, "P2", "data.base.inputs[1]", "empty prompt"),
        ({"inputs": "x"}, "P2", "data.base.inputs", "list of strings"),
        ({"inputs": [1]}, "P2", "data.base.inputs", "list of strings"),
        (
            {"dataset": inline_ref(["x"]), "field": "input"},
            "P2",
            "data.base.dataset",
            "derived ref",
        ),
    ],
    ids=[
        "both",
        "neither",
        "field",
        "draw",
        "empty-list",
        "empty-prompt",
        "not-a-list",
        "not-strings",
        "authored-ref",
    ],
)
def test_inline_role_refusals(role, code, where, fragment):
    doc = inline_doc()
    doc["data"] = role
    with pytest.raises(ParseError) as err:
        parse_document(doc)
    assert err.value.code == code and err.value.path == where
    assert fragment in str(err.value)


def test_inputs_are_not_sweepable():
    doc = inline_doc()
    doc["data"] = {"inputs": {"sweep": [["x"], ["y"]]}}
    with pytest.raises(ValidationError) as err:
        parse_document(doc)
    assert err.value.rule == 14 and err.value.path == "data.base.inputs"


@pytest.mark.parametrize(
    "seed, fragment",
    [(True, "bool"), ("3", "str"), (2.5, "float"), (None, "NoneType")],
    ids=["bool", "str", "float", "null"],
)
def test_shuffle_seed_is_an_integer_and_not_a_bool(seed, fragment):
    with pytest.raises(ParseError) as err:
        parse_document(_shuffled({"seed": seed}))
    assert err.value.code == "P2"
    assert err.value.path == "data.counterfactual.shuffle.seed"
    assert fragment in str(err.value)


def test_shuffle_takes_exactly_the_key_seed():
    with pytest.raises(ParseError) as err:
        parse_document(_shuffled({"seed": 1, "extra": 2}))
    assert err.value.code == "P3"
    assert err.value.path == "data.counterfactual.shuffle"
    assert "'extra'" in str(err.value)
    with pytest.raises(ParseError) as missing:
        parse_document(_shuffled({}))
    assert missing.value.code == "P2"
    assert missing.value.path == "data.counterfactual.shuffle"
    assert "'seed'" in str(missing.value)


def test_shuffle_is_an_object():
    with pytest.raises(ParseError) as err:
        parse_document(_shuffled(7))
    assert err.value.code == "P2"
    assert err.value.path == "data.counterfactual.shuffle"


def test_shuffle_seed_is_not_sweepable():
    """One document is one pairing: a swept seed would make a shuffled-source
    control differ from its target by an axis, not by one field."""
    with pytest.raises(ValidationError) as err:
        parse_document(_shuffled({"seed": {"sweep": [0, 1]}}))
    assert err.value.rule == 14
    assert err.value.path == "data.counterfactual.shuffle.seed"


# --------------------------------------------------------------------------- #
# writes.<w>.ragged — the ragged-window policy (§2.8, §5 rule 19)
# --------------------------------------------------------------------------- #


def _ragged_doc(ragged: object) -> dict:
    doc = base_doc()
    doc["method"]["writes"]["patch"]["ragged"] = ragged
    return in_order(doc)


def test_ragged_is_absent_by_default():
    """No default is materialized: absent is ``None`` on the spec, and the
    executor reads it as ``refuse`` — the behaviour every document had."""
    assert parse_document(in_order(base_doc())).writes["patch"].ragged is None


#: spelled out so this module collects on a tree without the field and the
#: tests below fail at the parse (P3) — the fails-without witness
RAGGED_POLICIES = ("refuse", "exact_length_buckets", "padded_masked")


def test_the_policy_vocabulary_is_the_schema_s():
    from causalab.protocol.schema import RAGGED_FIELD, RAGGED_POLICIES as policies

    assert policies == RAGGED_POLICIES and RAGGED_FIELD == "ragged"


@pytest.mark.parametrize("policy", RAGGED_POLICIES)
def test_ragged_parses_each_policy(policy: str):
    assert (
        parse_document(_ragged_doc({"policy": policy})).writes["patch"].ragged == policy
    )


def test_ragged_policy_is_a_closed_enum_with_a_suggestion():
    with pytest.raises(ParseError) as err:
        parse_document(_ragged_doc({"policy": "exact_length_bucket"}))
    assert err.value.code == "P4"
    assert err.value.path == "writes.patch.ragged.policy"
    assert "exact_length_buckets" in str(err.value)  # the suggestion


def test_ragged_needs_a_policy():
    with pytest.raises(ParseError) as err:
        parse_document(_ragged_doc({}))
    assert err.value.code == "P2"
    assert err.value.path == "writes.patch.ragged"
    assert "'policy'" in str(err.value)


def test_ragged_takes_exactly_the_key_policy():
    with pytest.raises(ParseError) as err:
        parse_document(_ragged_doc({"policy": "refuse", "buckets": 4}))
    assert err.value.code == "P3"
    assert err.value.path == "writes.patch.ragged"
    assert "'buckets'" in str(err.value)


def test_ragged_is_an_object():
    with pytest.raises(ParseError) as err:
        parse_document(_ragged_doc("refuse"))
    assert err.value.code == "P2"
    assert err.value.path == "writes.patch.ragged"


def test_ragged_is_not_sweepable():
    """How a ragged window lands is an execution strategy, fixed per campaign
    like ``minimum_count`` — never a research axis."""
    with pytest.raises(ValidationError) as err:
        parse_document(
            _ragged_doc({"sweep": [{"policy": "refuse"}, {"policy": "padded_masked"}]})
        )
    assert err.value.rule == 14
    assert err.value.path == "writes.patch.ragged"


def test_a_write_still_refuses_an_unknown_key_beside_ragged():
    doc = base_doc()
    doc["method"]["writes"]["patch"]["raggedness"] = {"policy": "refuse"}
    with pytest.raises(ParseError) as err:
        parse_document(in_order(doc))
    assert err.value.code == "P3"
    assert "'raggedness'" in str(err.value)
    assert "'ragged'" in str(err.value)  # the suggestion names the field


def test_ragged_enters_the_canonical_form_only_when_authored(env):
    """Canonical only when authored (§7): a document with no ``ragged`` field
    canonicalizes byte-identically to before the field existed, so no pinned
    digest moves; an authored field rides through verbatim and moves the
    digest, as any authored write field does."""
    plain = canonicalize(in_order(base_doc()), env)
    assert "ragged" not in json.dumps(plain)
    assert canonical_bytes(plain) == canonical_bytes(
        canonicalize(in_order(base_doc()), env)
    )
    declared = canonicalize(_ragged_doc({"policy": "padded_masked"}), env)
    assert declared["method"]["writes"]["patch"]["ragged"] == {
        "policy": "padded_masked"
    }
    assert digest(declared) != digest(plain)
    # `refuse` spelled out is a different document from `refuse` implied: the
    # field has no materialized default, so the spelling is the author's
    spelled = canonicalize(_ragged_doc({"policy": "refuse"}), env)
    assert spelled["method"]["writes"]["patch"]["ragged"] == {"policy": "refuse"}
    assert digest(spelled) != digest(plain)


# §2.10 `js` and `restrict` — a target that is a read, an answer set by shape
# --------------------------------------------------------------------------- #


def _js_doc(**metric_extra):
    raw = base_doc()
    raw["method"]["reads"]["cf_logits"] = {"site": "lm_head", "pos": -1}
    raw["method"]["intervened_models"][UNWRITTEN]["reads"].append("cf_logits")
    raw["method"]["save"].append(
        saved(
            "logits",
            "patched",
            "js.json",
            {
                "kind": "js",
                "target": {"read": "cf_logits", "model": UNWRITTEN},
                **metric_extra,
            },
        )
    )
    return in_order(raw)


def test_js_parses_unrestricted_without_token_form():
    """Unrestricted, `js` is `kl`'s symmetric twin: two reads, no string
    resolved, so `token_form` is not accepted."""
    parsed = parse_document(_js_doc())
    (target,) = by_label(parsed)["js"].fields.values()
    assert target.read == "cf_logits"  # bound to the model that lists it
    assert by_label(parsed)["js"].token_form is None


def test_js_restrict_accepts_a_column_name_of_ids():
    parsed = parse_document(_js_doc(restrict="valid_answers", token_form="id"))
    assert by_label(parsed)["js"].fields["restrict"] == "valid_answers"
    assert by_label(parsed)["js"].token_form == "id"


def test_js_restrict_accepts_a_literal_answer_list():
    parsed = parse_document(_js_doc(restrict=["Yes", "No"]))
    assert tuple(by_label(parsed)["js"].fields["restrict"]) == ("Yes", "No")


def test_js_with_restrict_parses_without_token_form():
    """`restrict` makes the kind resolve strings, and strings are tokenized as
    written — nothing left for a document to declare."""
    parsed = parse_document(_js_doc(restrict=["Yes", "No"]))
    assert by_label(parsed)["js"].token_form is None


def test_js_without_restrict_refuses_token_form():
    with pytest.raises(ParseError) as err:
        parse_document(_js_doc(token_form="id"))
    assert err.value.code == "P3"


def test_js_restrict_is_not_sweepable():
    """An answer space is not a research variable — the `token_form` and
    `tokens` reasoning, applied to both spellings."""
    with pytest.raises(ParseError) as err:
        parse_document(_js_doc(restrict={"sweep": [["Yes"], ["No"]]}))
    assert err.value.code == "P2"


def test_js_restrict_literal_refuses_a_duplicated_answer():
    with pytest.raises(ParseError):
        parse_document(_js_doc(restrict=["Yes", "Yes"]))


def test_a_restrict_column_is_a_metric_column_and_a_literal_is_not():
    """The one predicate `validate --data`, the eligibility count and the
    run share: a string `restrict` is a column the table must carry; a literal
    list and the target read are not."""
    from causalab.protocol.schema import metric_column_fields

    by_column = parse_document(_js_doc(restrict="valid"))
    by_literal = parse_document(_js_doc(restrict=["Yes"]))
    plain = parse_document(_js_doc())
    assert metric_column_fields(by_label(by_column)["js"]) == {"restrict": "valid"}
    assert metric_column_fields(by_label(by_literal)["js"]) == {}
    assert metric_column_fields(by_label(plain)["js"]) == {}
    # and the existing kinds are unchanged by the refactor
    assert metric_column_fields(by_label(plain)["ld"]) == {
        "a": "cf_answer",
        "b": "base_answer",
    }


def test_an_unrestricted_js_binds_to_any_read_and_a_restricted_one_to_lm_head():
    """Rule 4 decided per document: unrestricted, `js` compares two whole
    distributions and binds where `kl` does; restricted, it resolves answer
    strings to token ids and needs a plain `lm_head` read like every
    token-space kind."""
    from causalab.protocol.rules.document import validate_document

    raw = base_doc()
    # the un-intervened network on base, read at the target site
    raw["method"]["reads"]["v_base"] = {"site": "tgt", "pos": -1}
    raw["method"]["intervened_models"]["original_base"] = {
        "input": "base",
        "reads": ["v_base"],
    }
    raw["method"]["save"].append(
        saved(
            "v_base",
            "original_base",
            "js.json",
            {"kind": "js", "target": {"read": "v_cf", "model": UNWRITTEN}},
        )
    )
    validate_document(parse_document(in_order(raw)), engine_is_local=True)

    raw["method"]["save"][1]["aggregation"].update(restrict=["Yes", "No"])
    with pytest.raises(ValidationError) as err:
        validate_document(parse_document(in_order(raw)), engine_is_local=True)
    assert err.value.rule == 4
    assert "vocabulary" in str(err.value)


def test_js_target_must_be_a_read_on_the_same_component():
    from causalab.protocol.rules.document import validate_document

    raw = _js_doc()
    # block_output, not lm_head: the js entry is the second save entry
    raw["method"]["save"][1]["aggregation"]["target"] = {
        "read": "v_cf",
        "model": UNWRITTEN,
    }
    with pytest.raises(ValidationError) as err:
        validate_document(parse_document(raw), engine_is_local=True)
    assert err.value.rule == 4
    assert "js compares two reads" in str(err.value)


# --------------------------------------------------------------------------- #
# §2.5 `init` on a gate — a fill, or a saved theta
# --------------------------------------------------------------------------- #


def _gate_init_doc(init, **gate_extra):
    raw = base_doc()
    raw["method"]["featurizers"] = {"g": {"kind": "gate", "init": init, **gate_extra}}
    raw["method"]["reads"]["v_cf"]["featurizer"] = "g"
    raw["method"]["writes"]["patch"]["featurizer"] = "g"
    return in_order(raw)


def test_gate_init_fill_parses_to_a_mask_value():
    doc = parse_document(_gate_init_doc({"fill": 0.99}))
    assert doc.featurizers["g"].init == {"fill": 0.99}
    assert parse_document(_gate_init_doc({"fill": 1})).featurizers["g"].init == {
        "fill": 1.0
    }


def test_gate_init_accepts_the_saved_theta_form():
    doc = parse_document(
        _gate_init_doc({"file_path": "fit/gate.safetensors", "entry": {"step": 40}})
    )
    assert doc.featurizers["g"].init == {
        "file_path": "fit/gate.safetensors",
        "entry": {"step": 40},
    }


def test_gate_init_names_one_start():
    with pytest.raises(ParseError) as err:
        parse_document(_gate_init_doc({"fill": 0.5, "file_path": "g.safetensors"}))
    assert err.value.code == "P2" and "not both" in str(err.value)
    with pytest.raises(ParseError, match="not both"):
        parse_document(_gate_init_doc({"fill": 0.5, "entry": {"k": 1}}))
    with pytest.raises(ParseError, match="or, on a gate, a fill"):
        parse_document(_gate_init_doc({"entry": {"k": 1}}))


@pytest.mark.parametrize("fill", [-0.1, 1.5, "0.9"])
def test_gate_init_fill_is_a_number_in_the_unit_interval(fill):
    with pytest.raises(ParseError) as err:
        parse_document(_gate_init_doc({"fill": fill}))
    assert err.value.code == "P2"


def test_gate_init_fill_is_not_a_subspace_spelling():
    """The subspace `init` is closed as before: a basis has no fill."""
    raw = _subspace_doc(init={"fill": 0.5})
    with pytest.raises(ParseError) as err:
        parse_document(in_order(raw))
    assert err.value.code == "P3"


def test_gate_init_fill_is_sweepable():
    """Whether the start decides the mask is a research question, so the
    fill is an axis like any named field (§3)."""
    raw = _gate_init_doc({"fill": {"sweep": [0.5, 0.99]}})
    parsed = parse_document(raw)
    assert isinstance(parsed.featurizers["g"].init["fill"], Sweep)
    assert any(
        axis.path == ("method", "featurizers", "g", "init", "fill")
        for axis in find_axes(raw)
    )


# --------------------------------------------------------------------------- #
# §2.11 `train.control` — a closed-loop schedule on a hyperparameter
# --------------------------------------------------------------------------- #

CONTROL_TARGET = "train.objective.sparsity.weight"


def _control_doc(control=None, *, objective=None, **gate_extra):
    raw = base_doc()
    raw["method"]["featurizers"] = {"g": {"kind": "gate", **gate_extra}}
    raw["method"]["reads"]["v_cf"]["featurizer"] = "g"
    raw["method"]["writes"]["patch"]["featurizer"] = "g"
    raw["method"]["train"] = {
        "objective": objective
        if objective is not None
        else {
            "fit": _fit(weight=1.0),
            "sparsity": {"weight": 0.025, "l1": "g"},
        },
        "params": ["g"],
        "optimizer": {"name": "adam", "lr": 0.1},
        "steps": {"epochs": 2},
        "batch": {"pairs": 2},
        "seed": 0,
    }
    if control is not None:
        raw["method"]["train"]["control"] = control
    raw["method"]["save"].append(
        {"value": "g", "site": "tgt", "file_path": "g.safetensors"}
    )
    return in_order(raw)


def _pid(**overrides):
    return {
        "kind": "pid",
        "signal": {"hard_mask_size": "g"},
        "setpoint": {"ramp": [12, 0, 0.5]},
        "gains": {"kp": 0.1, "ki": 0.001},
        **overrides,
    }


def test_control_parses_and_keeps_only_what_was_authored():
    doc = parse_document(_control_doc({CONTROL_TARGET: _pid()}))
    entry = doc.train.control[CONTROL_TARGET]
    assert entry["kind"] == "pid"
    assert entry["signal"] == {"hard_mask_size": "g"}
    assert entry["setpoint"] == {"ramp": [12.0, 0.0, 0.5]}
    assert entry["gains"] == {"kp": 0.1, "ki": 0.001}
    assert "space" not in entry  # defaults are the canonical form's to fill


@pytest.mark.parametrize(
    "bad, code, needle",
    [
        ({"kind": "bang"}, "P4", "pid"),
        ({"signal": {"hard_mask_size": "g", "other": "g"}}, "P2", "exactly one"),
        ({"signal": {"kept_heads": "g"}}, "P4", "hard_mask_size"),
        ({"setpoint": {"ramp": [12, 0, 0.0]}}, "P2", "frac"),
        ({"setpoint": {"constant": 3}}, "P3", "constant"),
        ({"gains": {"kp": 0.1}}, "P2", "ki"),
        ({"bounds": [1.0, 0.5]}, "P2", "increasing"),
        ({"d_clip": 0}, "P2", "positive"),
        ({"space": "sqrt"}, "P4", "log"),
    ],
)
def test_control_refuses_a_malformed_entry(bad, code, needle):
    with pytest.raises(ParseError) as err:
        parse_document(_control_doc({CONTROL_TARGET: _pid(**bad)}))
    assert err.value.code == code
    assert needle in str(err.value)


def test_control_gains_are_sweepable_and_the_rest_is_not():
    raw = _control_doc(
        {CONTROL_TARGET: _pid(gains={"kp": {"sweep": [0.1, 0.3]}, "ki": 0.001})}
    )
    assert any(
        axis.path == ("method", "train", "control", CONTROL_TARGET, "gains", "kp")
        for axis in find_axes(raw)
    )
    with pytest.raises(ValidationError, match="sweep wrapper is not allowed"):
        parse_document(_control_doc({CONTROL_TARGET: _pid(kind={"sweep": ["pid"]})}))


def test_an_anneal_parses_both_spellings_and_refuses_a_ratio_through_zero():
    """§2.11 `anneal`: the list is a linear schedule; the mapping names its
    endpoints and a `shape`; a geometric shape has no meaning across zero."""
    from causalab.protocol.schema import AnnealSchedule

    raw = _control_doc()
    raw["method"]["train"]["anneal"] = {
        "g.theta.temperature": [1.0, 0.01, 0.5],
        "train.objective.sparsity.weight": {
            "from": 0.01,
            "to": 30,
            "frac": 0.5,
            "shape": "geometric",
        },
    }
    train = parse_document(raw).train
    assert train.anneal["g.theta.temperature"] == AnnealSchedule(1.0, 0.01, 0.5)
    weight = train.anneal["train.objective.sparsity.weight"]
    assert weight == AnnealSchedule(0.01, 30.0, 0.5, "geometric")
    assert weight.value_at(0, 100) == 0.01
    assert weight.value_at(25, 100) == pytest.approx((0.01 * 30) ** 0.5)
    assert weight.value_at(50, 100) == pytest.approx(30.0)
    assert weight.value_at(99, 100) == pytest.approx(30.0)  # holds after the ramp

    for bad, code, needle in (
        ({"from": -1.0, "to": 1.0, "frac": 0.5, "shape": "geometric"}, "P2", "sign"),
        ({"from": 0.0, "to": 1.0, "frac": 0.5, "shape": "geometric"}, "P2", "zero"),
        ({"from": 1.0, "to": 0.1, "frac": 0.5, "shape": "sqrt"}, "P4", "geometric"),
        ({"from": 1.0, "to": 0.1}, "P2", "frac"),
        ({"from": 1.0, "to": 0.1, "frac": 0.5, "steps": 3}, "P3", "steps"),
        ([1.0, 0.1], "P2", "[start, end, frac]"),
        ("fast", "P2", "anneal schedule"),
    ):
        raw = _control_doc()
        raw["method"]["train"]["anneal"] = {"g.theta.temperature": bad}
        with pytest.raises(ParseError) as err:
            parse_document(raw)
        assert err.value.code == code, (bad, str(err.value))
        assert needle in str(err.value), (bad, str(err.value))


def test_an_annealed_weight_names_a_numeric_named_term():
    """Rule 4 for a weight `anneal` (§2.11), the open-loop twin of the same
    `control` rule: a positional term has no name, a swept weight is not one
    start, and one path may not be both annealed and controlled."""
    from causalab.protocol.rules.document import validate_document

    def check(raw):
        validate_document(parse_document(raw), engine_is_local=True)

    raw = _control_doc()
    raw["method"]["train"]["anneal"] = {CONTROL_TARGET: [0.025, 30.0, 0.5]}
    check(raw)

    raw = _control_doc(objective=[[1.0, _fit()], [0.025, {"l1": "g"}]])
    raw["method"]["train"]["anneal"] = {CONTROL_TARGET: [0.025, 30.0, 0.5]}
    with pytest.raises(ValidationError) as err:
        check(raw)
    assert err.value.rule == 4 and "named objective term" in str(err.value)

    raw = _control_doc()
    raw["method"]["train"]["anneal"] = {"train.objective.fit.read": [0.0, 1.0, 0.5]}
    with pytest.raises(ValidationError) as err:
        check(raw)
    assert err.value.rule == 4

    raw = _control_doc(
        objective={
            "fit": _fit(weight=1.0),
            "sparsity": {"weight": {"sweep": [0.01, 0.1]}, "l1": "g"},
        }
    )
    raw["method"]["train"]["anneal"] = {CONTROL_TARGET: [0.025, 30.0, 0.5]}
    with pytest.raises(ValidationError) as err:
        check(raw)
    assert err.value.rule == 4 and "schedule's start" in str(err.value)

    raw = _control_doc({CONTROL_TARGET: _pid()})
    raw["method"]["train"]["anneal"] = {CONTROL_TARGET: [0.025, 30.0, 0.5]}
    with pytest.raises(ValidationError) as err:
        check(raw)
    assert err.value.rule == 4 and "both annealed and controlled" in str(err.value)


def _phased_doc(phases, **train_extra):
    raw = _control_doc()
    raw["method"]["featurizers"]["rot"] = {"kind": "subspace", "k": 4}
    raw["method"]["reads"]["v_cf"]["featurizer"] = ["rot", "g"]
    raw["method"]["writes"]["patch"]["featurizer"] = ["rot", "g"]
    raw["method"]["train"]["params"] = ["rot", "g"]
    raw["method"]["train"]["phases"] = phases
    raw["method"]["train"].update(train_extra)
    raw["method"]["save"].append(
        {"value": "rot", "site": "tgt", "file_path": "rot.safetensors"}
    )
    return in_order(raw)


def test_phases_parse_and_partition_the_run():
    """§2.11 `phases`: consecutive windows in one unit, the last frac 1.0,
    params a non-empty subset of train.params; a phase's optimizer and anneal
    are parsed as the top-level ones, keyed by the phase's own params."""
    from causalab.protocol.schema import AnnealSchedule, PhaseSpec

    train = parse_document(
        _phased_doc(
            [
                {"until": {"frac": 0.1}, "params": ["g"]},
                {
                    "until": {"frac": 0.8},
                    "params": ["rot", "g"],
                    "anneal": {"g.theta.temperature": [1.0, 0.05, 1.0]},
                },
                {
                    "until": {"frac": 1.0},
                    "params": ["rot"],
                    "freeze_masks": ["g"],
                    "optimizer": {"lr": {"rot": 3e-4}},
                },
            ]
        )
    ).train
    assert train.phases == (
        PhaseSpec(until={"frac": 0.1}, params=("g",)),
        PhaseSpec(
            until={"frac": 0.8},
            params=("rot", "g"),
            anneal={"g.theta.temperature": AnnealSchedule(1.0, 0.05, 1.0)},
        ),
        PhaseSpec(
            until={"frac": 1.0},
            params=("rot",),
            optimizer={"lr": {"rot": 3e-4}},
            freeze_masks=("g",),
        ),
    )
    updates = parse_document(
        _phased_doc(
            [
                {"until": {"updates": 3}, "params": ["g"]},
                {"until": {"updates": 8}, "params": ["rot"]},
            ]
        )
    ).train
    assert [p.until for p in updates.phases] == [{"updates": 3}, {"updates": 8}]

    for bad, code, needle in (
        ([], "P2", "non-empty"),
        ([{"until": {"frac": 0.5}, "params": ["g"]}], "P2", "ends at 1.0"),
        (
            [
                {"until": {"frac": 0.5}, "params": ["g"]},
                {"until": {"frac": 0.5}, "params": ["rot"]},
            ],
            "P2",
            "consecutive",
        ),
        (
            [
                {"until": {"frac": 0.5}, "params": ["g"]},
                {"until": {"updates": 4}, "params": ["rot"]},
            ],
            "P2",
            "one unit",
        ),
        ([{"until": {"frac": 1.0}, "params": ["other"]}], "P2", "never widens"),
        ([{"until": {"frac": 1.0}, "params": []}], "P2", "at least one"),
        ([{"until": {"frac": 1.0}, "params": ["g", "g"]}], "P2", "once"),
        ([{"until": {"frac": 1.5}, "params": ["g"]}], "P2", "(0, 1]"),
        (
            [{"until": {"frac": 1.0, "updates": 2}, "params": ["g"]}],
            "P2",
            "exactly one",
        ),
        ([{"params": ["g"]}], "P2", "until"),
        (
            [
                {
                    "until": {"frac": 1.0},
                    "params": ["g"],
                    "optimizer": {"lr": {"rot": 0.1}},
                }
            ],
            "P2",
            "does not train",
        ),
        (
            [{"until": {"frac": 1.0}, "params": ["g"], "optimizer": {"eps": 0.1}}],
            "P3",
            "eps",
        ),
        ([{"until": {"frac": 1.0}, "params": ["g"], "steps": 3}], "P3", "steps"),
    ):
        with pytest.raises(ParseError) as err:
            parse_document(_phased_doc(bad))
        assert err.value.code == code, (bad, str(err.value))
        assert needle in str(err.value), (bad, str(err.value))


def test_phase_names_resolve_against_the_fit_they_narrow():
    """Rule 4 for `phases` (§2.11): a phase's anneal names a featurizer the
    phase trains (or a named term's weight) and no path a top-level schedule
    already moves; `freeze_masks` names gates the phase does not train."""
    from causalab.protocol.rules.document import validate_document

    def check(raw):
        validate_document(parse_document(raw), engine_is_local=True)

    good = [
        {"until": {"frac": 0.5}, "params": ["g"]},
        {
            "until": {"frac": 1.0},
            "params": ["rot"],
            "freeze_masks": ["g"],
            "anneal": {"train.objective.sparsity.weight": [0.025, 30.0, 1.0]},
        },
    ]
    check(_phased_doc(good))

    bad = copy.deepcopy(good)
    bad[1]["anneal"] = {"g.theta.temperature": [1.0, 0.1, 1.0]}
    with pytest.raises(ValidationError) as err:
        check(_phased_doc(bad))
    assert err.value.rule == 4 and "does not train in this phase" in str(err.value)

    bad = copy.deepcopy(good)
    bad[1]["freeze_masks"] = ["rot"]
    with pytest.raises(ValidationError) as err:
        check(_phased_doc(bad))
    assert err.value.rule == 4 and "not a gate" in str(err.value)

    bad = copy.deepcopy(good)
    bad[0]["freeze_masks"] = ["g"]  # phase 0 trains g
    with pytest.raises(ValidationError) as err:
        check(_phased_doc(bad))
    assert err.value.rule == 4 and "pin it or train it" in str(err.value)

    with pytest.raises(ValidationError) as err:
        check(
            _phased_doc(
                good, anneal={"train.objective.sparsity.weight": [0.025, 1.0, 1.0]}
            )
        )
    assert err.value.rule == 4 and "one value, one schedule" in str(err.value)

    with pytest.raises(ValidationError) as err:
        check(_phased_doc(good, control={CONTROL_TARGET: _pid()}))
    assert err.value.rule == 4 and "one value, one schedule" in str(err.value)

    bad = copy.deepcopy(good)
    bad[1]["anneal"] = {"train.objective.nope.weight": [0.025, 30.0, 1.0]}
    with pytest.raises(ValidationError) as err:
        check(_phased_doc(bad))
    assert err.value.rule == 4 and "named objective term" in str(err.value)


def test_control_targets_resolve_and_signals_are_trained_gates():
    """Rule 4 for `control` (§2.11): the target is a named term's weight or a
    trained featurizer's hyperparameter, the signal a trained gate, and no
    path is both annealed and controlled."""
    from causalab.protocol.rules.document import validate_document

    def check(raw):
        validate_document(parse_document(raw), engine_is_local=True)

    check(_control_doc({CONTROL_TARGET: _pid()}))  # the shape the plan runs
    check(_control_doc({"g.theta.temperature": _pid()}))  # an anneal-style path

    with pytest.raises(ValidationError) as err:
        check(
            _control_doc(
                {CONTROL_TARGET: _pid()},
                objective=[[1.0, _fit()], [0.025, {"l1": "g"}]],
            )
        )
    assert err.value.rule == 4 and "named objective term" in str(err.value)

    with pytest.raises(ValidationError) as err:
        check(_control_doc({"train.objective.fit.read": _pid()}))
    assert err.value.rule == 4

    raw = _control_doc({CONTROL_TARGET: _pid(signal={"hard_mask_size": "rot"})})
    raw["method"]["featurizers"]["rot"] = {"kind": "subspace", "k": 4}
    raw["method"]["train"]["params"].append("rot")
    raw["method"]["save"].append(
        {"value": "rot", "site": "tgt", "file_path": "rot.safetensors"}
    )
    with pytest.raises(ValidationError) as err:
        check(raw)
    assert err.value.rule == 4 and "not a trained gate" in str(err.value)

    raw = _control_doc({"g.theta.temperature": _pid()})
    raw["method"]["train"]["anneal"] = {"g.theta.temperature": [1.0, 0.01, 0.5]}
    with pytest.raises(ValidationError) as err:
        check(raw)
    assert err.value.rule == 4 and "both annealed and controlled" in str(err.value)


# --------------------------------------------------------------------------- #
# §2.12 `trajectory` — the fit's checkpoints as one bundle
# --------------------------------------------------------------------------- #


def _trajectory_doc(entry):
    raw = base_doc()
    raw["method"]["featurizers"] = {"g": {"kind": "gate"}}
    raw["method"]["reads"]["v_cf"]["featurizer"] = "g"
    raw["method"]["writes"]["patch"]["featurizer"] = "g"
    raw["method"]["train"] = {
        "objective": [[1.0, _fit()]],
        "params": ["g"],
        "optimizer": {"name": "adam", "lr": 0.1},
        "steps": {"epochs": 2},
        "batch": {"pairs": 2},
    }
    raw["method"]["save"].append(
        {"value": "g", "site": "tgt", "file_path": "g.safetensors"}
    )
    raw["method"]["save"].append(entry)
    return in_order(raw)


@pytest.mark.parametrize("every", [{"count": 20}, {"updates": 5}, {"epochs": 1}])
def test_a_trajectory_entry_parses_its_spacing(every):
    doc = parse_document(
        _trajectory_doc(
            {"kind": "trajectory", "every": every, "file_path": "t.safetensors"}
        )
    )
    (entry,) = [e for e in doc.save if e.kind == "trajectory"]
    assert entry.every == every and entry.value == "trajectory"


@pytest.mark.parametrize(
    "entry, code",
    [
        ({"kind": "trajectory", "file_path": "t.safetensors"}, "P2"),
        (
            {
                "kind": "trajectory",
                "every": {"count": 2, "epochs": 1},
                "file_path": "t.safetensors",
            },
            "P2",
        ),
        (
            {"kind": "trajectory", "every": {"count": 0}, "file_path": "t.safetensors"},
            "P2",
        ),
        (
            {"kind": "trajectory", "every": {"steps": 3}, "file_path": "t.safetensors"},
            "P3",
        ),
        (
            {"kind": "location_ledger", "every": {"count": 3}, "file_path": "l.json"},
            "P3",
        ),
    ],
)
def test_a_trajectory_entry_refuses_a_malformed_spacing(entry, code):
    with pytest.raises(ParseError) as err:
        parse_document(_trajectory_doc(entry))
    assert err.value.code == code


# --------------------------------------------------------------------------- #
# train.control.<target>.signal over a LIST of gates (§2.11)
# --------------------------------------------------------------------------- #


def _two_gate_control_doc(signal):
    """`_control_doc` with a second trained head gate `h` on a second site, so
    a signal may name both — the many-layer DBM shape (§2.11)."""
    raw = _control_doc({CONTROL_TARGET: _pid(signal={"hard_mask_size": signal})})
    raw["method"]["sites"]["tgt2"] = {"component": "block_output", "layers": [1]}
    raw["method"]["featurizers"]["h"] = {"kind": "gate"}
    raw["method"]["reads"]["v_cf2"] = {"site": "tgt2", "pos": -1, "featurizer": "h"}
    raw["method"]["intervened_models"][UNWRITTEN]["reads"].append("v_cf2")
    raw["method"]["writes"]["patch2"] = {
        "site": "tgt2",
        "pos": -1,
        "featurizer": "h",
        "do": {"swap": "v_cf2"},
    }
    for im in raw["method"]["intervened_models"].values():
        if im.get("writes"):  # the written model gains the second gate's write
            im["writes"] = list(im["writes"]) + ["patch2"]
    raw["method"]["train"]["params"].append("h")
    raw["method"]["train"]["objective"]["sparsity"]["l1"] = ["g", "h"]
    raw["method"]["save"].append(
        {"value": "h", "site": "tgt2", "file_path": "h.safetensors"}
    )
    return in_order(raw)


def test_a_control_signal_may_sum_several_gates():
    doc = parse_document(_two_gate_control_doc(["g", "h"]))
    assert doc.train.control[CONTROL_TARGET]["signal"] == {"hard_mask_size": ["g", "h"]}


@pytest.mark.parametrize(
    "signal, needle",
    [([], "non-empty"), (["g", "g"], "once"), ([3], "string"), ({"g": 1}, "non-empty")],
)
def test_a_list_signal_is_a_non_empty_list_of_distinct_gate_names(signal, needle):
    with pytest.raises(ParseError) as err:
        parse_document(_two_gate_control_doc(signal))
    assert err.value.code == "P2" and needle in str(err.value)


def test_every_gate_of_a_list_signal_must_be_trained():
    from causalab.protocol.rules.document import validate_document

    validate_document(
        parse_document(_two_gate_control_doc(["g", "h"])), engine_is_local=True
    )
    raw = _two_gate_control_doc(["g", "h"])
    raw["method"]["train"]["params"].remove("h")
    raw["method"]["train"]["objective"]["sparsity"]["l1"] = "g"
    raw["method"]["save"] = [e for e in raw["method"]["save"] if e.get("value") != "h"]
    with pytest.raises(ValidationError) as err:
        validate_document(parse_document(raw), engine_is_local=True)
    assert err.value.rule == 4 and "'h'" in str(err.value)


# gate `dead` — the dead-unit rule (§2.5) ---------------------------------------- #


def test_gate_dead_parses_one_rule_and_defaults_to_none(env):
    """`dead` is optional and one of two rules; absent, the spec carries
    ``None`` and the canonical form no key — no existing digest moves."""
    assert parse_document(_gate_init_doc({"fill": 0.5})).featurizers["g"].dead is None
    frozen = parse_document(_gate_init_doc({"fill": 0.5}, dead={"freeze_after": 20}))
    assert frozen.featurizers["g"].dead == {"freeze_after": 20}
    leaky = parse_document(_gate_init_doc({"fill": 0.5}, dead={"leak": 0.01}))
    assert leaky.featurizers["g"].dead == {"leak": 0.01}
    plain = canonicalize(_gate_init_doc({"fill": 0.5}), env)
    assert "dead" not in plain["method"]["featurizers"]["g"]
    canon = canonicalize(_gate_init_doc({"fill": 0.5}, dead={"freeze_after": 20}), env)
    assert canon["method"]["featurizers"]["g"]["dead"] == {"freeze_after": 20}


def test_gate_dead_names_exactly_one_rule():
    """A frozen unit takes no gradient and a leaking one exists to keep taking
    it: the two rules contradict each other on one gate."""
    with pytest.raises(ParseError, match="exactly one rule"):
        parse_document(
            _gate_init_doc({"fill": 0.5}, dead={"freeze_after": 2, "leak": 0.1})
        )
    with pytest.raises(ParseError, match="names a rule"):
        parse_document(_gate_init_doc({"fill": 0.5}, dead={}))
    with pytest.raises(ParseError) as err:
        parse_document(_gate_init_doc({"fill": 0.5}, dead={"freeze": 2}))
    assert err.value.code == "P3" and "freeze_after" in str(err.value)


@pytest.mark.parametrize("value", [0, -1, 1.5, True, "2"])
def test_gate_dead_freeze_after_is_a_positive_integer(value):
    with pytest.raises(ParseError, match="positive integer"):
        parse_document(_gate_init_doc({"fill": 0.5}, dead={"freeze_after": value}))


@pytest.mark.parametrize("value", [0, 1, 1.0, -0.1, True, "0.1"])
def test_gate_dead_leak_is_strictly_inside_the_unit_interval(value):
    with pytest.raises(ParseError, match="strictly inside"):
        parse_document(_gate_init_doc({"fill": 0.5}, dead={"leak": value}))


def test_gate_dead_is_a_training_rule_so_a_loaded_gate_refuses_it():
    raw = base_doc()
    raw["method"]["featurizers"] = {
        "g": {"kind": "gate", "file_path": "fit/g.safetensors", "dead": {"leak": 0.1}}
    }
    with pytest.raises(ParseError, match="loaded gate uses a fixed mask"):
        parse_document(in_order(raw))


def test_gate_dead_is_not_a_field_of_any_other_kind():
    raw = base_doc()
    raw["method"]["featurizers"] = {
        "rot": {"kind": "subspace", "k": 2, "dead": {"leak": 0.1}}
    }
    with pytest.raises(ParseError) as err:
        parse_document(in_order(raw))
    assert err.value.code == "P3"


# --------------------------------------------------------------------------- #
# §2.5 top_k — a loaded gate read out at a count; §2.12 rank
# --------------------------------------------------------------------------- #


def _loaded_gate_doc(gate: dict, save_extra: list | None = None):
    raw = base_doc()
    raw["method"]["featurizers"] = {"g": {"kind": "gate", **gate}}
    raw["method"]["reads"]["v_cf"]["featurizer"] = "g"
    raw["method"]["writes"]["patch"]["featurizer"] = "g"
    raw["method"]["save"].extend(save_extra or [])
    return in_order(raw)


def test_gate_top_k_parses_and_is_sweepable():
    doc = parse_document(_loaded_gate_doc({"file_path": "g.safetensors", "top_k": 3}))
    assert doc.featurizers["g"].top_k == 3
    assert (
        parse_document(_loaded_gate_doc({"file_path": "g.safetensors"}))
        .featurizers["g"]
        .top_k
        is None
    )
    swept = parse_document(
        _loaded_gate_doc({"file_path": "g.safetensors", "top_k": {"sweep": [0, 2, 4]}})
    )
    assert isinstance(swept.featurizers["g"].top_k, Sweep)
    assert swept.featurizers["g"].top_k.values == (0, 2, 4)


def test_gate_top_k_needs_a_file_path():
    """A fit is read through its map's own split; a cut of a training mask
    would make the loss and the eval disagree about which units are on."""
    with pytest.raises(ParseError) as err:
        parse_document(_loaded_gate_doc({"top_k": 3}))
    assert err.value.code == "P2" and "file_path" in str(err.value)


@pytest.mark.parametrize("top_k", [-1, 2.5, True, "3"])
def test_gate_top_k_is_a_non_negative_integer(top_k):
    with pytest.raises(ParseError) as err:
        parse_document(_loaded_gate_doc({"file_path": "g.safetensors", "top_k": top_k}))
    assert err.value.code == "P2"


def test_gate_top_k_is_a_gate_field():
    raw = base_doc()
    raw["method"]["featurizers"] = {
        "r": {"kind": "subspace", "k": 2, "file_path": "r.safetensors", "top_k": 1}
    }
    with pytest.raises(ParseError) as err:
        parse_document(in_order(raw))
    assert err.value.code == "P3"


def test_gate_top_k_enters_the_canonical_form_only_when_authored():
    """Absent, nothing changes: a document that names no ``top_k`` is byte for
    byte the document it was before the field existed."""
    from pathlib import Path

    from tests.protocol._env import build_env

    env = build_env(Path(__file__).parent / "fixtures" / "artifacts")
    plain = _loaded_gate_doc({"file_path": "weekdays/qwen25_7b/locate.json"})
    cut = _loaded_gate_doc({"file_path": "weekdays/qwen25_7b/locate.json", "top_k": 2})
    plain_form = canonicalize(plain, env)
    cut_form = canonicalize(cut, env)
    assert "top_k" not in plain_form["method"]["featurizers"]["g"]
    assert cut_form["method"]["featurizers"]["g"]["top_k"] == 2


def test_a_rank_entry_parses_as_a_non_value_kind():
    doc = parse_document(
        _loaded_gate_doc(
            {"file_path": "g.safetensors"},
            [{"kind": "rank", "file_path": "rank.json"}],
        )
    )
    (entry,) = [e for e in doc.save if e.kind == "rank"]
    assert entry.value == "rank" and entry.file_path == "rank.json"
    assert entry.every is None and entry.read is None and entry.site is None


def test_a_rank_entry_takes_no_spacing():
    with pytest.raises(ParseError) as err:
        parse_document(
            _loaded_gate_doc(
                {"file_path": "g.safetensors"},
                [{"kind": "rank", "every": {"count": 3}, "file_path": "rank.json"}],
            )
        )
    assert err.value.code == "P3"


# --------------------------------------------------------------------------- #
# §2.5 budget — k_schedule and stop_grad_shift
# --------------------------------------------------------------------------- #


def _budget_fit_doc(gate: dict):
    raw = base_doc()
    raw["method"]["featurizers"] = {"g": {"kind": "gate", **gate}}
    raw["method"]["reads"]["v_cf"]["featurizer"] = "g"
    raw["method"]["writes"]["patch"]["featurizer"] = "g"
    raw["method"]["train"] = {
        "objective": [[1.0, _fit()]],
        "params": ["g"],
        "optimizer": {"name": "adam", "lr": 0.1},
        "steps": {"epochs": 1},
        "batch": {"pairs": 2},
    }
    raw["method"]["save"].append(
        {"value": "g", "site": "tgt", "file_path": "g.safetensors"}
    )
    return in_order(raw)


def test_budget_gate_parses_its_schedule():
    doc = parse_document(
        _budget_fit_doc(
            {"parametrization": "budget", "k_schedule": {"kind": "fixed", "k": 3}}
        )
    )
    assert doc.featurizers["g"].k_schedule == {"kind": "fixed", "k": 3}
    assert doc.featurizers["g"].stop_grad_shift is None
    doc = parse_document(
        _budget_fit_doc(
            {
                "parametrization": "budget",
                "k_schedule": {"kind": "log_uniform", "low": 1, "high": 7, "eval": 2},
                "stop_grad_shift": True,
            }
        )
    )
    assert doc.featurizers["g"].k_schedule == {
        "kind": "log_uniform",
        "low": 1,
        "high": 7,
        "eval": 2,
    }
    assert doc.featurizers["g"].stop_grad_shift is True
    swept = parse_document(
        _budget_fit_doc(
            {
                "parametrization": "budget",
                "k_schedule": {"kind": "fixed", "k": {"sweep": [2, 4, 8]}},
            }
        )
    )
    assert isinstance(swept.featurizers["g"].k_schedule["k"], Sweep)


@pytest.mark.parametrize(
    "gate, needle",
    [
        ({"parametrization": "budget"}, "k_schedule"),
        ({"k_schedule": {"kind": "fixed", "k": 3}}, "require budget"),
        (
            {"parametrization": "clamp", "stop_grad_shift": True},
            "require budget",
        ),
        (
            {"parametrization": "budget", "k_schedule": {"kind": "fixed"}},
            "names 'k'",
        ),
        (
            {
                "parametrization": "budget",
                "k_schedule": {"kind": "uniform", "low": 1, "high": 4},
            },
            "'eval'",
        ),
        (
            {
                "parametrization": "budget",
                "k_schedule": {"kind": "uniform", "low": 5, "high": 4, "eval": 1},
            },
            "ordered",
        ),
        (
            {
                "parametrization": "budget",
                "k_schedule": {"kind": "log_uniform", "low": 0, "high": 4, "eval": 1},
            },
            "at least 1",
        ),
        (
            {
                "parametrization": "budget",
                "k_schedule": {"kind": "fixed", "k": 2},
                "stop_grad_shift": 1,
            },
            "true or false",
        ),
        (
            {
                "parametrization": "budget",
                "k_schedule": {"kind": "fixed", "k": 2},
                "temperature": 0.5,
            },
            "requires hard_concrete or boundary",
        ),
    ],
)
def test_budget_gate_refusals_at_parse(gate, needle):
    with pytest.raises(ParseError) as err:
        parse_document(_budget_fit_doc(gate))
    assert err.value.code in ("P2", "P3") and needle in str(err.value)


def test_a_loaded_budget_gate_takes_top_k_not_a_schedule():
    with pytest.raises(ParseError) as err:
        parse_document(
            _loaded_gate_doc(
                {
                    "parametrization": "budget",
                    "file_path": "g.safetensors",
                    "k_schedule": {"kind": "fixed", "k": 2},
                }
            )
        )
    assert "top_k" in str(err.value)
    doc = parse_document(
        _loaded_gate_doc(
            {"parametrization": "budget", "file_path": "g.safetensors", "top_k": 2}
        )
    )
    assert doc.featurizers["g"].top_k == 2 and doc.featurizers["g"].k_schedule is None


# --------------------------------------------------------------------------- #
# §2.11 optimizer.schedule — linear_warmup_decay
# --------------------------------------------------------------------------- #


def _fit_with_optimizer(optimizer: dict):
    raw = base_doc()
    raw["method"]["featurizers"] = {"g": {"kind": "gate"}}
    raw["method"]["reads"]["v_cf"]["featurizer"] = "g"
    raw["method"]["writes"]["patch"]["featurizer"] = "g"
    raw["method"]["train"] = {
        "objective": [[1.0, _fit()]],
        "params": ["g"],
        "optimizer": optimizer,
        "steps": {"epochs": 1},
        "batch": {"pairs": 2},
    }
    raw["method"]["save"].append(
        {"value": "g", "site": "tgt", "file_path": "g.safetensors"}
    )
    return in_order(raw)


def test_optimizer_schedule_is_a_closed_vocabulary_with_warmup_under_linear_only():
    from causalab.protocol.schema import OPTIMIZER_SCHEDULES

    assert OPTIMIZER_SCHEDULES == ("constant", "linear_warmup_decay")
    doc = parse_document(
        _fit_with_optimizer(
            {
                "name": "adam",
                "lr": 0.3,
                "schedule": "linear_warmup_decay",
                "warmup_frac": 0.1,
            }
        )
    )
    assert doc.train.optimizer["schedule"] == "linear_warmup_decay"
    assert doc.train.optimizer["warmup_frac"] == 0.1
    with pytest.raises(ParseError) as err:
        parse_document(
            _fit_with_optimizer({"name": "adam", "lr": 0.3, "schedule": "cosine"})
        )
    assert err.value.code == "P4"
    with pytest.raises(ParseError, match="belongs to schedule"):
        parse_document(
            _fit_with_optimizer({"name": "adam", "lr": 0.3, "warmup_frac": 0.1})
        )
    with pytest.raises(ParseError, match="in \\[0, 1\\)"):
        parse_document(
            _fit_with_optimizer(
                {
                    "name": "adam",
                    "lr": 0.3,
                    "schedule": "linear_warmup_decay",
                    "warmup_frac": 1.0,
                }
            )
        )


def test_a_drawn_roles_resolved_field_is_its_eval_member():
    """§2.2: the field a forward tokenizes is one property, asked by every
    mirror (`resolve_roles`, the loader's variables check, the data
    identity): `column[eval]` for a drawn role, the authored field otherwise.
    """
    doc = base_doc()
    doc["data"]["counterfactual"] = {
        **doc["data"]["counterfactual"],
        "field": "counterfactual_inputs",
        "draw": {"kind": "uniform", "eval": 1},
    }
    parsed = parse_document(in_order(doc))
    assert parsed.data["counterfactual"].resolved_field == "counterfactual_inputs[1]"
    assert parsed.data["base"].resolved_field == "input"
    plain = parse_document(in_order(base_doc()))
    assert plain.data["counterfactual"].resolved_field == "counterfactual_inputs[0]"


# §2.5 `parametrization: boundary` — the per-unit fields are refused at parse -- #


def _boundary_doc(extra: dict | None = None, chain=("rot", "bnd")) -> dict:
    doc = base_doc()
    doc["method"]["featurizers"] = {
        "rot": {"kind": "subspace", "k": 4, "parametrization": "cayley"},
        "bnd": {"kind": "gate", "parametrization": "boundary", **(extra or {})},
    }
    doc["method"]["reads"]["v_cf"]["featurizer"] = list(chain)
    doc["method"]["writes"]["patch"]["featurizer"] = list(chain)
    return in_order(doc)


def test_a_boundary_gate_parses_with_a_temperature_and_a_fill():
    doc = parse_document(_boundary_doc({"temperature": 0.5, "init": {"fill": 0.25}}))
    spec = doc.featurizers["bnd"]
    assert spec.parametrization == "boundary" and spec.temperature == 0.5
    assert spec.init == {"fill": 0.25}


@pytest.mark.parametrize(
    ("field", "value", "needle"),
    [
        ("group", "site", "requires one theta entry per unit"),
        ("axis", "position", "ordered feature basis"),
        ("dead", {"leak": 0.1}, "requires theta entries per unit"),
        ("stretch", [-0.1, 1.1], "requires hard_concrete"),
    ],
)
def test_a_boundary_gate_refuses_the_per_unit_fields_in_the_tables_words(
    field, value, needle
):
    with pytest.raises(ParseError) as err:
        parse_document(_boundary_doc({field: value}))
    assert err.value.path == f"featurizers.bnd.{field}", err.value.path
    assert needle in str(err.value) and "'boundary'" in str(err.value)


def test_a_loaded_boundary_gate_refuses_top_k_and_a_pooled_readout():
    with pytest.raises(ParseError) as err:
        parse_document(_boundary_doc({"file_path": "fit/bnd.safetensors", "top_k": 3}))
    assert err.value.path == "featurizers.bnd.top_k"
    assert "requires a ranking of units" in str(err.value)
    with pytest.raises(ParseError) as err:
        parse_document(_boundary_doc({"file_path": "fit/bnd.safetensors", "pool": "p"}))
    assert err.value.path == "featurizers.bnd.pool"
    assert "requires per-unit maps" in str(err.value)


def test_a_boundary_gate_refuses_init_from_scores():
    scores = {"file_path": "scores.json", "unit": "unit", "value": "score", "keep": 2}
    with pytest.raises(ParseError) as err:
        parse_document(_boundary_doc({"init": {"from_scores": scores}}))
    assert err.value.path == "featurizers.bnd.init.from_scores"
    assert "no unit to place" in str(err.value)
