"""Shared document builders for the protocol tests."""

from __future__ import annotations

from typing import Any

from causalab.protocol.schema import GROUP_ORDER, METHOD_SECTIONS, PROTOCOL_VERSION

#: The un-intervened model `base_doc` reads its counterfactual on —
#: the name `causalab migrate` gives it (§2.9: `original_<role>` when the
#: network is read un-intervened on a role other than base alone), so the
#: literal canonicalizes byte for byte as its protocol-3 ancestor migrated.
UNWRITTEN = "original_counterfactual"

#: `base_doc`'s one aggregation, the margin `ld.json` tabulates.
LOGIT_DIFF: dict[str, Any] = {
    "kind": "logit_diff",
    "a": "cf_answer",
    "b": "base_answer",
}


def base_doc() -> dict[str, Any]:
    """A minimal valid interchange document on gpt2 (layer 3 < 12): the
    counterfactual's residual at layer 3, read on the un-intervened model,
    swapped into base in ``patched``, whose logits the one save entry
    reduces to a logit difference (§1)."""
    return {
        "header": {"protocol_version": PROTOCOL_VERSION},
        "model": {"key": "gpt2", "revision": "main"},
        "data": {
            "base": {"dataset": "weekdays/data#train", "field": "input"},
            "counterfactual": {
                "dataset": "weekdays/data#train",
                "field": "counterfactual_inputs[0]",
            },
        },
        "method": {
            "intervened_models": {
                UNWRITTEN: {"input": "counterfactual", "reads": ["v_cf"]},
                "patched": {"input": "base", "reads": ["logits"], "writes": ["patch"]},
            },
            "sites": {
                "tgt": {"component": "block_output", "layers": [3]},
                "lm_head": {"component": "lm_head"},
            },
            "reads": {
                "v_cf": {"site": "tgt", "pos": -1},
                "logits": {"site": "lm_head", "pos": -1},
            },
            "writes": {"patch": {"site": "tgt", "pos": -1, "do": {"swap": "v_cf"}}},
            "save": [saved("logits", "patched", "ld.json", dict(LOGIT_DIFF))],
        },
    }


def base_only_doc() -> dict[str, Any]:
    """`base_doc` with no counterfactual: the one read of it, the model
    that took it and its operand go, and the write swaps in a literal zero
    (§2.8)."""
    doc = base_doc()
    del doc["data"]["counterfactual"]
    del doc["method"]["reads"]["v_cf"]
    del doc["method"]["intervened_models"][UNWRITTEN]
    doc["method"]["writes"]["patch"]["do"] = {"swap": 0}
    return doc


def inline_doc(*inputs: str) -> dict[str, Any]:
    """`base_only_doc` over an inline table (§2.2), shaped like the intro
    ablation demo: the prompts are the data, and the aggregation names its
    answer literally, so the document needs no dataset column at all."""
    doc = base_only_doc()
    doc["data"] = {"inputs": list(inputs or ("The Space Needle is located in",))}
    doc["method"]["save"] = [
        saved(
            "logits",
            "patched",
            "ld.json",  # the ancestor kept its file; the table's label stays `ld`
            aggregation("class_probs", groups={"Seattle": [" Seattle"]}),
        )
    ]
    return doc


def aggregation(kind: str, **fields: Any) -> dict[str, Any]:
    """One ``aggregation`` block (§2.10): the kind plus its value fields."""
    return {"kind": kind, **fields}


def saved(
    read: str,
    model: str,
    file_path: str,
    aggregation: dict[str, Any] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """One read save entry (§2.12): a tensor of the read on ``model``, or —
    with ``aggregation`` — a table of the reduction over it."""
    entry: dict[str, Any] = {"read": read, "model": model}
    if aggregation is not None:
        entry["aggregation"] = aggregation
    entry.update(extra)
    entry["file_path"] = file_path
    return entry


def term(
    read: str, model: str, aggregation: dict[str, Any], **extra: Any
) -> dict[str, Any]:
    """An objective or eval term's fields (§2.11): the bound read and the
    reduction over it, plus ``weight`` or whatever else the term carries."""
    return {"read": read, "model": model, "aggregation": aggregation, **extra}


def in_order(raw: dict[str, Any]) -> dict[str, Any]:
    """Rebuild a mutated document with its groups, and the method's sections,
    in the §1 order — test mutations append sections at the dict end, and while
    that no longer refuses the document (§5 rule 2 warns now), it does raise a
    warning the test isn't about.

    A method section spread at the top level (``{**base_doc(), "segments":
    …}``) is hoisted into ``method`` first: the spelling predates the groups,
    and what such a test is about is the section, never the shape (§1 —
    `tests.protocol.test_protocol_v2` pins the shape rule itself)."""
    method: dict[str, Any] = dict(raw.get("method") or {})
    for key in METHOD_SECTIONS:
        if key in raw:
            method[key] = raw[key]
    out = {key: raw[key] for key in GROUP_ORDER if key in raw and key != "method"}
    if method or "method" in raw:
        out["method"] = {key: method[key] for key in METHOD_SECTIONS if key in method}
    return {key: out[key] for key in GROUP_ORDER if key in out}


def by_label(doc: Any) -> dict[str, Any]:
    """A parsed document's aggregations by label (§2.10): a save entry's file
    stem, an objective term's name or an eval key — the first owner wins when
    two entries share a label."""
    out: dict[str, Any] = {}
    for agg in doc.aggregations():
        out.setdefault(agg.label, agg.spec)
    return out
