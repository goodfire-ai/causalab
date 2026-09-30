"""The manifold_fig4 documents: the spellings the paper's code sums, and the
paths the manifold_path step writes."""

from __future__ import annotations

import json

import pytest

from tests.demos.papers._scripts import PAPERS, load_script

pytestmark = pytest.mark.unit


def _document(name: str) -> dict:
    return json.loads((PAPERS / "protocols" / name).read_text())


def test_the_documents_save_the_lowercase_spellings() -> None:
    """The baseline and steering documents save a third ``class_probs``
    group with the lowercase spellings that are one token, which the paper's
    code sums into ``p(x)`` (``tokenize_variable_values``)."""
    lower = {"Monday": [" monday"], "Friday": [" friday"], "Sunday": [" sunday"]}
    for name in ("manifold_fig4_baseline.json", "manifold_fig4_steer.json"):
        groups = [s["aggregation"]["groups"] for s in _document(name)["method"]["save"]]
        assert lower in groups, name


def test_the_steering_document_writes_the_steps_paths() -> None:
    """The steering document's pair axis is the manifold_path step's pair
    list, each row names the two bundles the workflow declares for its pair,
    and each strategy writes one swap: the chord point in the full residual
    stream, or the spline point in the ``pca`` featurizer."""
    names = load_script("manifold_fig4", "manifold_path").path_names()
    document = _document("manifold_fig4_steer.json")
    workflow = json.loads((PAPERS / "workflows" / "manifold_fig4.json").read_text())
    outputs = workflow["steps"]["manifold_path"]["outputs"]
    rows = document["axes"]["pair"]["rows"]
    assert [row["pair"] for row in rows] == names
    for row in rows:
        for column in ("waypoints", "chord"):
            slot = f"{column}_{row['pair']}"
            assert row[column] == f"manifold_path/{outputs[slot]}"
    method = document["method"]
    assert method["intervened_models"]["linear"]["writes"] == ["lin_chord"]
    assert method["writes"]["lin_chord"] == {
        "site": "target",
        "pos": "slot",
        "do": {"swap": "chord"},
    }
    assert method["writes"]["man_waypoint"]["featurizer"] == "pca"
