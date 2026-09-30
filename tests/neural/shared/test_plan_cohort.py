"""Which points of a campaign may fit **together** (spec §4, "Cohorts").

A cohort's optimizer step is one forward over the concatenation of every
member's minibatch, each member's writes landing on its own rows. That is
one forward only when every member runs the same network realization on the
same rows in the same frame — so the cohort key is exactly the network, the
data identity per input role, and the ``segments`` frame; everything a member
trains, sweeps or schedules (featurizer specs, seed, objective, optimizer,
epochs, eval cadence, the write's layer) is the member's own. A point without
a ``train`` section fits nothing and is never in a cohort.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from causalab.neural.shared.plan import cohort_key, fit_cohorts
from causalab.protocol.schema import Document, parse_document

from tests.protocol._docs import aggregation, base_doc, in_order, term


pytestmark = pytest.mark.unit

#: The fit's objective: a cross-entropy over the patched logits.
CE = term(
    "logits",
    "patched",
    aggregation("cross_entropy", target="label"),
)

IDENTITY = {
    "base": "0" * 64 + "#input",
    "counterfactual": "0" * 64 + "#counterfactual_inputs[0]",
}


def _train_doc(**changes: Any) -> dict[str, Any]:
    raw = base_doc()
    method = raw["method"]
    method["featurizers"] = {"rot": {"kind": "subspace", "k": 4}}
    method["reads"]["v_cf"]["featurizer"] = "rot"
    method["writes"]["patch"]["featurizer"] = "rot"
    method["train"] = {
        "objective": [[1.0, CE]],
        "params": ["rot"],
        "optimizer": {"name": "adamw", "lr": 1e-3},
        "steps": {"epochs": 2},
        "batch": {"pairs": 2},
        "seed": 0,
    }
    method["save"].append({"value": "rot", "site": "tgt", "file_path": "rot.st"})
    for dotted, value in changes.items():
        *path, last = dotted.split(".")
        # a method section is addressed by its own name (§1 layout)
        node: Any = raw if path[0] in ("model", "data") else method
        for part in path:
            node = node[part]
        node[last] = value
    return raw


def _doc(raw: dict[str, Any]) -> Document:
    return parse_document(in_order(raw))


def test_a_point_without_train_has_no_cohort_key() -> None:
    assert cohort_key(_doc(base_doc()), IDENTITY) is None


def test_the_key_ignores_what_the_member_owns() -> None:
    """Rank, seed, objective, optimizer, schedule, eval cadence and the
    write's layer are per member: the forward they share does not depend on
    any of them."""
    reference = cohort_key(_doc(_train_doc()), IDENTITY)
    assert reference is not None
    variants = [
        _train_doc(**{"featurizers.rot.k": 8}),
        _train_doc(**{"train.seed": 3}),
        _train_doc(**{"train.optimizer": {"name": "sgd", "lr": 0.1}}),
        _train_doc(**{"train.steps": {"epochs": 5}}),
        _train_doc(**{"train.batch": {"pairs": 1}}),
        _train_doc(**{"sites.tgt.layers": 7}),
    ]
    dbm = _train_doc()
    dbm["method"]["featurizers"] = {"rot": {"kind": "gate"}}
    dbm["method"]["train"]["objective"] = [[1.0, CE], [0.01, {"l1": "rot"}]]
    dbm["method"]["train"]["anneal"] = {"rot.theta.temperature": [1.0, 0.01, 0.5]}
    variants.append(dbm)
    for raw in variants:
        assert cohort_key(_doc(raw), IDENTITY) == reference


def test_the_key_is_the_network_the_rows_and_the_frame() -> None:
    reference = cohort_key(_doc(_train_doc()), IDENTITY)
    other_model = _train_doc(**{"model.dtype": "bf16"})
    assert cohort_key(_doc(other_model), IDENTITY) != reference
    other_rows = dict(IDENTITY, base="1" * 64 + "#input")
    assert cohort_key(_doc(_train_doc()), other_rows) != reference
    framed = _train_doc()
    framed["method"]["segments"] = {"frame": "chat"}
    assert cohort_key(_doc(framed), IDENTITY) != reference


def test_an_attention_interior_site_separates_the_cohort() -> None:
    """A member writing at an attention-function interior runs its forwards
    under eager whatever the document's backend; a cohort forward runs under
    one implementation, so such a member never shares one with a member
    whose sites are module boundaries."""
    reference = cohort_key(_doc(_train_doc()), IDENTITY)
    interior = _train_doc(**{"sites.tgt.component": "attention_query"})
    assert cohort_key(_doc(interior), IDENTITY) != reference
    docs = [_doc(_train_doc()), _doc(interior), _doc(_train_doc(**{"train.seed": 1}))]
    assert fit_cohorts(docs, [IDENTITY] * 3) == ((0, 2), (1,))


def test_cohorts_partition_the_points_in_first_appearance_order() -> None:
    """Two train points on one realization, an inference point between them,
    and a train point on another realization: the two matching fits form one
    cohort, the others stand alone, and every index appears exactly once."""
    docs = [
        _doc(_train_doc()),
        _doc(base_doc()),
        _doc(_train_doc(**{"train.seed": 1})),
        _doc(_train_doc(**{"model.dtype": "bf16"})),
    ]
    cohorts = fit_cohorts(docs, [IDENTITY] * len(docs))
    assert cohorts == ((0, 2), (1,), (3,))
    assert sorted(i for cohort in cohorts for i in cohort) == list(range(len(docs)))


def test_cohorts_need_one_identity_per_point() -> None:
    docs = [_doc(_train_doc()), _doc(_train_doc())]
    with pytest.raises(ValueError, match="lockstep"):
        fit_cohorts(docs, [IDENTITY])


def test_the_key_does_not_move_when_a_document_is_copied() -> None:
    raw = _train_doc()
    assert cohort_key(_doc(raw), IDENTITY) == cohort_key(
        _doc(copy.deepcopy(raw)), IDENTITY
    )
