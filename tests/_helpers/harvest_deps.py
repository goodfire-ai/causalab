"""Skip predicates for pieces the document generators' tests lean on that
this tree does not carry yet.

``scripts/expand_layers.py``, ``scripts/add_routing_reads.py`` and
``scripts/joint_dbm.py`` landed ahead of three things their tests assume.
Each dependency is a *predicate* here that probes the live registry or
schema, so a gated test switches itself on the moment the piece lands —
never a bare skip that a hand has to lift, never a copy of the piece under
test:

* the ``Qwen/Qwen3.6-35B-A3B`` registry row (`has_a3b_entry`) and the
  per-layer ``ModelInfo.layer_types`` stream pattern
  (`registry_declares_layer_types`) the heads family reads;
* the grouped gate featurizer, ``"group": "head"`` / ``"expert_neuron"``
  (`gate_accepts_group`);
* the named ``train.objective`` form with one list-valued regularizer over
  several gates (`train_accepts_named_objective`).

The markers below are the predicates as ``skipif`` marks with the missing
piece in their reason, so a skip report says what is missing.
"""

from __future__ import annotations

import dataclasses
import warnings
from typing import Any

import pytest

from causalab.protocol.rules.errors import ProtocolError
from causalab.protocol.registry import ModelInfo, get_model_info
from causalab.protocol.schema import parse_document

from tests.protocol._docs import LOGIT_DIFF, base_doc, in_order, term


A3B = "Qwen/Qwen3.6-35B-A3B"


def has_a3b_entry() -> bool:
    """Whether the A3B is a built-in registry row."""
    try:
        get_model_info(A3B)
    except ProtocolError:
        return False
    return True


def registry_declares_layer_types() -> bool:
    """Whether ``ModelInfo`` carries the per-layer stream pattern — the field ``model_info_from_hf_config`` fills from an HF config's
    ``layer_types`` and the heads family of ``joint_dbm.py`` reads."""
    return any(field.name == "layer_types" for field in dataclasses.fields(ModelInfo))


def _parses(doc: dict[str, Any]) -> bool:
    """Whether the schema accepts ``doc`` — the §5.1 strict parse alone, no
    resolution, no cross-reference checks, section-order warnings silenced."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            parse_document(in_order(doc))
        except ProtocolError:
            return False
    return True


def gate_accepts_group() -> bool:
    """Whether a gate featurizer may declare ``group``."""
    doc = base_doc()
    doc["featurizers"] = {"g": {"kind": "gate", "group": "head"}}
    return _parses(doc)


def train_accepts_named_objective() -> bool:
    """Whether ``train.objective`` may be the named form with one regularizer
    over a list of gates."""
    doc = base_doc()
    doc["featurizers"] = {"g": {"kind": "gate"}}
    doc["train"] = {
        "objective": {
            "fit": {"weight": 1.0, **term("logits", "patched", LOGIT_DIFF)},
            "sparsity": {"weight": 0.1, "l1": ["g"]},
        },
        "params": ["g"],
        "optimizer": {"name": "adamw", "lr": 1e-3},
        "steps": {"epochs": 1},
        "batch": {"pairs": 2},
    }
    return _parses(doc)


needs_a3b_entry = pytest.mark.skipif(
    not has_a3b_entry(),
    reason=f"needs the {A3B} registry row",
)
needs_layer_types = pytest.mark.skipif(
    not registry_declares_layer_types(),
    reason="needs ModelInfo.layer_types",
)
needs_grouped_gate = pytest.mark.skipif(
    not gate_accepts_group(),
    reason="needs the grouped gate featurizer",
)
needs_named_objective = pytest.mark.skipif(
    not train_accepts_named_objective(),
    reason="needs the named train.objective sparsity term",
)
