"""The decode's derived shape: depth, materialization, capability (§4, §6, §8).

The point of these tests is that the *cost* of a generate document is a
value the planner emits, not an engine heuristic: a document that wants one
token's distribution must not oblige the same work as one that wants every
token's. So they assert the plan, never memory.
"""

from __future__ import annotations

from typing import Any

import pytest

from causalab.protocol.engine import requires
from causalab.neural.shared.plan import plan_point
from causalab.protocol.schema import parse_document

from tests.protocol._docs import aggregation, base_doc, in_order, saved


pytestmark = pytest.mark.unit


def probe_doc(anchor: dict[str, Any], budget: int = 8) -> dict[str, Any]:
    """A minimal document whose saved metric reduces a continuation read."""
    raw = base_doc()
    raw["method"]["positions"] = {
        "cont": {"generated": {"max_new_tokens": budget}, **anchor}
    }
    raw["method"]["reads"]["logits"]["pos"] = "cont"
    return in_order(raw)


def group_for(raw: dict[str, Any], model: str = "patched"):
    plan = plan_point(parse_document(raw))
    return next(g for g in plan.groups if g.model == model)


def test_decode_depth_is_the_budget():
    group = group_for(probe_doc({"index": -1}, budget=12))
    assert group.decode_depth == 12


def test_prompt_frame_documents_do_not_decode():
    for group in plan_point(parse_document(base_doc())).groups:
        assert group.decode_depth == 0
        assert group.materialize == ()


def test_depth_is_the_max_over_the_groups_reads():
    """Two reads of one model at different budgets share one decode: the run
    goes as deep as the deepest, each read windows its own."""
    raw = probe_doc({"index": -1}, budget=4)
    raw["method"]["positions"]["long"] = {
        "generated": {"max_new_tokens": 16},
        "all": True,
    }
    raw["method"]["reads"]["tail"] = {"site": "lm_head", "pos": "long"}
    raw["method"]["intervened_models"]["patched"]["reads"].append("tail")
    raw["method"]["save"].append(saved("tail", "patched", "tail.safetensors"))
    group = group_for(in_order(raw))
    assert group.decode_depth == 16
    assert {m.read for m in group.materialize} == {"logits", "tail"}


def test_a_metric_input_needs_a_distribution():
    """`logits` feeds a logit_diff, so its addressed position has to exist as
    a full vocabulary vector somewhere."""
    group = group_for(probe_doc({"index": -1}))
    (item,) = group.materialize
    assert (item.read, item.site) == ("logits", "lm_head")
    assert item.needs_distribution is True


def test_saving_a_read_obliges_building_it():
    """A continuation harvest at a non-head site: saved, so the engine owes
    the tensor whatever the site is. The *false* branch of
    ``needs_distribution`` is unreachable in v1 — every metric kind reduces
    logits — and becomes reachable when metric kinds declare a domain and an
    ids-only kind (a text probe) stops counting as a consumer."""
    raw = base_doc()
    raw["method"]["positions"] = {
        "cont": {"generated": {"max_new_tokens": 8}, "all": True}
    }
    raw["method"]["reads"] = {
        "v_cf": raw["method"]["reads"]["v_cf"],
        "acts": {"site": "tgt", "pos": "cont"},
    }
    raw["method"]["intervened_models"]["patched"]["reads"] = ["acts"]
    raw["method"]["save"] = [saved("acts", "patched", "acts.safetensors")]
    group = group_for(in_order(raw))
    (item,) = group.materialize
    assert (item.read, item.site) == ("acts", "tgt")
    assert item.needs_distribution is True


def test_a_generating_group_does_not_elide():
    """Elision ends the forward at the deepest tap; a decode needs the head
    on every step, so there is nothing left to skip."""
    group = group_for(probe_doc({"index": -1}))
    assert group.stop_after is None


def test_generate_is_a_required_capability():
    doc = parse_document(probe_doc({"index": -1}))
    assert "generate" in requires(doc)
    assert "generate" not in requires(parse_document(base_doc()))


def test_decode_depth_is_not_in_the_group_key():
    """A decode changes what a group *produces*, not what its prefill
    computes — so two points differing only in depth still share a prefill,
    the same reason taps are not in the key."""
    short = group_for(probe_doc({"index": -1}, budget=4))
    long = group_for(probe_doc({"index": -1}, budget=32))
    assert short.decode_depth != long.decode_depth
    assert short.key == long.key


def _decode_metric(raw: dict[str, Any]) -> dict[str, Any]:
    """Replace the document's aggregations with a single ids-domain one."""
    raw["method"]["save"] = [
        saved("logits", "patched", "said.json", aggregation("decode"))
    ]
    return in_order(raw)


def test_an_ids_only_metric_obliges_no_distribution():
    """A text probe reads the tokens the decode produced. Nothing downstream
    wants the vocabulary, so the plan must not ask for it — this is the
    whole reason metric kinds carry a domain."""
    group = group_for(_decode_metric(probe_doc({"all": True})))
    (item,) = group.materialize
    assert item.needs_distribution is False


def test_a_distribution_metric_still_obliges_one():
    group = group_for(probe_doc({"all": True}))
    (item,) = group.materialize
    assert item.needs_distribution is True


def test_saving_the_read_obliges_a_distribution_even_with_an_ids_metric():
    """The save manifest is the other consumer: an ids-domain metric does not
    excuse writing the read itself to disk."""
    raw = _decode_metric(probe_doc({"all": True}))
    raw["method"]["save"].append(saved("logits", "patched", "logits.safetensors"))
    group = group_for(in_order(raw))
    (item,) = group.materialize
    assert item.needs_distribution is True
