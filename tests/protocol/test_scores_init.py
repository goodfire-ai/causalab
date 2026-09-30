"""A gate's ``init.from_scores`` at the protocol layer (spec §2.5, rule 32).

The parser's half: the spelling, its defaults and its refusals. The canonical
form's half: the table's bytes enter the digest as
``init.from_scores.content_digest``, the two defaulted columns are
materialized so an authored default and an omitted one digest identically,
and ``keep`` above the unit count is refused as the document canonicalizes —
where the width and the group map are derived, so no model and no table read
is needed for that half (the coverage half is the build's, tested beside the
builder in ``tests/neural/shared/test_scores_init.py``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from causalab.protocol.schema.explicit import canonicalize
from causalab.protocol.rules.errors import RULES, ParseError, ValidationError
from causalab.protocol.pipeline import compile_protocol
from causalab.protocol.schema import (
    SCORES_INIT_DEFAULTS,
    SCORES_INIT_KEYS,
    Sweep,
    parse_document,
)

from tests.protocol._docs import base_doc, in_order
from tests.protocol._env import build_env
from tests.protocol.test_grouped_gate import GPT2_HEADS, gate_doc


pytestmark = pytest.mark.unit

SCORES = "scan/head_scores.json"
POSITION_SCORES = "scan/position_scores.json"


def _init_doc(init: dict[str, Any], **gate_extra: Any) -> dict[str, Any]:
    raw = base_doc()
    raw["method"]["featurizers"] = {"g": {"kind": "gate", "init": init, **gate_extra}}
    raw["method"]["reads"]["v_cf"]["featurizer"] = "g"
    raw["method"]["writes"]["patch"]["featurizer"] = "g"
    return in_order(raw)


def _scores(**extra: Any) -> dict[str, Any]:
    return {"file_path": SCORES, **extra}


# -- the parser ------------------------------------------------------------- #


def test_from_scores_parses_with_its_two_defaults_materialized() -> None:
    doc = parse_document(_init_doc({"from_scores": _scores(keep=3)}))
    assert doc.featurizers["g"].init == {
        "from_scores": {
            "file_path": SCORES,
            "unit": SCORES_INIT_DEFAULTS["unit"],
            "value": SCORES_INIT_DEFAULTS["value"],
            "keep": 3,
        }
    }


def test_from_scores_accepts_the_scale_form_a_unit_list_and_a_where_filter() -> None:
    doc = parse_document(
        _init_doc(
            {
                "from_scores": _scores(
                    unit=["expert", "neuron"],
                    value="mean",
                    where={"layer": 15},
                    scale=0.5,
                )
            }
        )
    )
    assert doc.featurizers["g"].init == {
        "from_scores": {
            "file_path": SCORES,
            "unit": ["expert", "neuron"],
            "value": "mean",
            "where": {"layer": 15},
            "scale": 0.5,
        }
    }


def test_keep_and_scale_are_sweepable() -> None:
    doc = parse_document(_init_doc({"from_scores": _scores(keep={"sweep": [1, 2, 4]})}))
    assert doc.featurizers["g"].init["from_scores"]["keep"] == Sweep(values=(1, 2, 4))
    doc = parse_document(
        _init_doc({"from_scores": _scores(scale={"sweep": [0.5, 2.0]})})
    )
    assert doc.featurizers["g"].init["from_scores"]["scale"] == Sweep(values=(0.5, 2.0))


def test_from_scores_is_one_of_the_three_starts() -> None:
    with pytest.raises(ParseError, match="not both"):
        parse_document(_init_doc({"from_scores": _scores(keep=1), "fill": 0.5}))
    with pytest.raises(ParseError, match="not both"):
        parse_document(
            _init_doc({"from_scores": _scores(keep=1), "file_path": "g.safetensors"})
        )
    with pytest.raises(ParseError, match="or, on a gate, a fill or from_scores"):
        parse_document(_init_doc({"entry": {"k": 1}}))


def test_from_scores_reads_the_table_exactly_one_way() -> None:
    with pytest.raises(ParseError, match="exactly one of the two"):
        parse_document(_init_doc({"from_scores": _scores(keep=2, scale=1.0)}))
    with pytest.raises(ParseError, match="exactly one of the two"):
        parse_document(_init_doc({"from_scores": _scores()}))


@pytest.mark.parametrize(
    "bad", [{"keep": 0}, {"keep": -1}, {"keep": 1.5}, {"scale": 0}]
)
def test_keep_is_a_positive_integer_and_scale_a_positive_number(bad) -> None:
    with pytest.raises(ParseError):
        parse_document(_init_doc({"from_scores": _scores(**bad)}))


def test_from_scores_needs_the_table_and_refuses_unknown_keys() -> None:
    with pytest.raises(ParseError, match="needs a file_path"):
        parse_document(_init_doc({"from_scores": {"keep": 1}}))
    with pytest.raises(ParseError):
        parse_document(_init_doc({"from_scores": _scores(keep=1, top_k=1)}))
    with pytest.raises(ParseError, match="scalar each row must equal"):
        parse_document(
            _init_doc({"from_scores": _scores(keep=1, where={"layer": [1]})})
        )
    with pytest.raises(ParseError, match="non-empty list"):
        parse_document(_init_doc({"from_scores": _scores(keep=1, unit=[])}))


def test_from_scores_is_a_gate_start_only() -> None:
    raw = base_doc()
    raw["method"]["featurizers"] = {
        "rot": {"kind": "subspace", "k": 2, "init": {"from_scores": _scores(keep=1)}}
    }
    raw["method"]["reads"]["v_cf"]["featurizer"] = "rot"
    raw["method"]["writes"]["patch"]["featurizer"] = "rot"
    with pytest.raises(ParseError):
        parse_document(in_order(raw))


def test_the_key_vocabulary_is_closed_and_the_defaults_are_in_it() -> None:
    assert set(SCORES_INIT_DEFAULTS) <= set(SCORES_INIT_KEYS)
    assert {"file_path", "keep", "scale", "where"} <= set(SCORES_INIT_KEYS)


# -- the canonical form, and rule 32's canonicalize-time half --------------- #


@pytest.fixture
def scored_env(tmp_path: Path):
    """An artifact root holding a 12-row head score table (gpt2's heads)."""
    table = tmp_path / SCORES
    table.parent.mkdir(parents=True)
    rows = [{"head": h, "value": float(GPT2_HEADS - h)} for h in range(GPT2_HEADS)]
    table.write_text(json.dumps(rows))
    return build_env(tmp_path)


@pytest.fixture
def position_scored_env(tmp_path: Path):
    """An artifact root holding a three-row *position* score table (§2.5
    ``axis``): a per-unit table whose unit index is a token position, the
    shape an attribution scan over a window writes. (A position gate's own
    ``rank.json`` is the same table under ``unit`` / ``theta``.)"""
    table = tmp_path / POSITION_SCORES
    table.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"position": t, "value": float(3 - t)} for t in range(3)]
    table.write_text(json.dumps(rows))
    return build_env(tmp_path)


def _position_gate(keep: int) -> dict[str, Any]:
    """A position gate over the window ``[0, 3)`` seeded from the position
    table (§2.5 ``axis``): its θ has three units, one per position."""
    doc = _init_doc(
        {"from_scores": _scores(file_path=POSITION_SCORES, unit="position", keep=keep)},
        axis="position",
    )
    for entry in (doc["method"]["reads"]["v_cf"], doc["method"]["writes"]["patch"]):
        entry["pos"] = {"span": [0, 3]}
    return in_order(doc)


def _head_gate(init: dict[str, Any]) -> dict[str, Any]:
    doc = gate_doc(group="head", component="attention_premix")
    doc["method"]["featurizers"]["g"]["init"] = init
    return in_order(doc)


def test_the_tables_bytes_enter_the_canonical_form(scored_env, tmp_path: Path) -> None:
    doc = _head_gate({"from_scores": _scores(unit="head", keep=3)})
    canon = canonicalize(doc, scored_env)["method"]["featurizers"]["g"]
    scores = canon["init"]["from_scores"]
    assert len(scores["content_digest"]) == 64
    assert scores["unit"] == "head" and scores["value"] == "value"
    # a different table is a different start: the digest moves with the bytes
    (tmp_path / SCORES).write_text(json.dumps([{"head": 0, "value": 1.0}]))
    moved = canonicalize(doc, scored_env)["method"]["featurizers"]["g"]
    assert moved["init"]["from_scores"]["content_digest"] != scores["content_digest"]


def test_an_authored_default_and_an_omitted_one_digest_identically(scored_env) -> None:
    omitted = canonicalize(
        _head_gate({"from_scores": _scores(unit="head", keep=3)}), scored_env
    )
    authored = canonicalize(
        _head_gate({"from_scores": _scores(unit="head", value="value", keep=3)}),
        scored_env,
    )
    assert omitted == authored


def test_rule_32_keep_above_the_unit_count_is_refused_at_load(scored_env) -> None:
    assert RULES["scores_init"].number == 32 and RULES["scores_init"].code == "V32"
    with pytest.raises(ValidationError) as err:
        compile_protocol(
            _head_gate({"from_scores": _scores(unit="head", keep=GPT2_HEADS + 1)}),
            env=scored_env,
        )
    assert err.value.rule == 32
    assert err.value.path == "featurizers.g.init.from_scores.keep"
    assert f"exceeds the gate's {GPT2_HEADS} units" in str(err.value)


def test_rule_32_counts_positions_on_a_position_gate(position_scored_env) -> None:
    """§2.5 ``axis``: the bound is over θ's units, which on a position gate are
    the window's positions — `keep: 4` on a three-position gate was checked
    against the feature width (768) before this branch and passed."""
    with pytest.raises(ValidationError) as err:
        compile_protocol(_position_gate(keep=4), env=position_scored_env)
    assert (
        err.value.rule == 32 and err.value.path == "featurizers.g.init.from_scores.keep"
    )
    assert "exceeds the gate's 3 units" in str(err.value)


def test_rule_32_the_whole_position_count_is_a_legal_keep(position_scored_env) -> None:
    assert compile_protocol(_position_gate(keep=3), env=position_scored_env) is not None


def test_rule_32_the_whole_unit_count_is_a_legal_keep(scored_env) -> None:
    loaded = compile_protocol(
        _head_gate({"from_scores": _scores(unit="head", keep=GPT2_HEADS)}),
        env=scored_env,
    )
    assert loaded is not None


def test_a_missing_table_is_the_artifact_refusal(scored_env) -> None:
    with pytest.raises(ValidationError) as err:
        compile_protocol(
            _head_gate({"from_scores": {"file_path": "nowhere.json", "keep": 1}}),
            env=scored_env,
        )
    assert err.value.rule == 15
