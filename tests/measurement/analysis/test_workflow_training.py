"""Configuration changes retain paired fit work without losing document evidence."""

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from hypothesis import given, strategies as st
import pytest

from causalab.measurement.analysis.compare import compare
from causalab.measurement.analysis.training import compare_training
from causalab.measurement.collection import write_record
from tests.measurement.analysis.test_compare import receipt
from tests.measurement.analysis.test_training import sample

pytestmark = pytest.mark.unit


def _samples() -> tuple[dict, dict]:
    before, after = sample(), sample()
    fit = after["numerics_context"]["fits"][0]
    fit["identity"]["protocol"] = "changed-training-budget"
    fit["optimizer_steps"] = 2
    fit["batches"].append({**fit["batches"][0], "update": 1})
    return {(7, 0): before}, {(7, 0): after}


def test_workflow_pair_keeps_both_document_identities_and_update_counts() -> None:
    before, after = _samples()
    result = compare_training(before, after, [(7, 0)], comparison="workflow")
    row = result["per_sample"][0]
    assert row["status"] == "compared"
    fit = row["fits"][0]
    assert fit["before"]["identity"]["protocol"] == "fixed"
    assert fit["after"]["identity"]["protocol"] == "changed-training-budget"
    assert fit["before"]["optimizer_steps"] == 1
    assert fit["after"]["optimizer_steps"] == 2
    assert fit["optimizer_steps_match"] is False
    assert fit["logical_schedule_matches"] is False
    assert {row["fit"]["protocol"] for row in result["within_seed"]} == {
        "fixed",
        "changed-training-budget",
    }


@given(st.sampled_from(["step", "coords", "train_params", "extra_identity"]))
def test_only_document_digest_is_excluded_from_workflow_pairing(field: str) -> None:
    before, after = _samples()
    identity = after[7, 0]["numerics_context"]["fits"][0]["identity"]
    identity[field] = {
        "step": "another_fit",
        "coords": {"sites.target.layers": 2},
        "train_params": ["different_parameters"],
        "extra_identity": "future-identity-dimension",
    }[field]
    result = compare_training(before, after, [(7, 0)], comparison="workflow")
    assert result["per_sample"][0]["status"] == "unaligned_fit_identities"
    assert result["per_sample"][0]["fits"] == []


def test_duplicate_stable_fit_coordinates_are_ambiguous() -> None:
    before, after = _samples()
    fits = after[7, 0]["numerics_context"]["fits"]
    duplicate = deepcopy(fits[0])
    duplicate["identity"]["protocol"] = "another-document"
    fits.append(duplicate)
    with pytest.raises(ValueError, match="ambiguous"):
        compare_training(before, after, [(7, 0)], comparison="workflow")


@pytest.mark.parametrize("field", ["step", "protocol", "coords", "train_params"])
def test_missing_fit_identity_fields_are_refused(field: str) -> None:
    before, after = _samples()
    del after[7, 0]["numerics_context"]["fits"][0]["identity"][field]
    with pytest.raises(ValueError, match="identity"):
        compare_training(before, after, [(7, 0)], comparison="workflow")


@pytest.mark.parametrize(
    "field,value",
    [("step", None), ("coords", None), ("train_params", None), ("protocol", "")],
)
def test_malformed_fit_identity_fields_are_refused(field: str, value: Any) -> None:
    before, after = _samples()
    after[7, 0]["numerics_context"]["fits"][0]["identity"][field] = value
    with pytest.raises(ValueError, match="identity"):
        compare_training(before, after, [(7, 0)], comparison="workflow")


def test_code_comparison_keeps_strict_document_identity() -> None:
    before, after = _samples()
    implicit = compare_training(before, after, [(7, 0)])
    assert implicit == compare_training(before, after, [(7, 0)], comparison="code")
    assert implicit["per_sample"][0]["status"] == "unaligned_fit_identities"


def _receipts(
    tmp_path: Path, modes: tuple[str | None, str | None]
) -> tuple[Path, Path]:
    before, after = _samples()
    paths = []
    for arm, samples, mode in zip(("before", "after"), (before, after), modes):
        path = receipt(tmp_path / arm, [[[1.0]]])
        record = json.loads(path.read_text())
        record["samples"][0].update(samples[7, 0])
        if mode is not None:
            record["context"] = {"worker": {"comparison_kind": mode}}
        write_record(path, record)
        paths.append(path)
    return paths[0], paths[1]


def test_comparison_uses_attested_workflow_mode_for_training(tmp_path: Path) -> None:
    paths = _receipts(tmp_path, ("workflow", "workflow"))
    result = compare(*paths, bootstrap_draws=100)
    diagnostic = result["comparison_summary"]["diagnostic_work"][0]
    assert diagnostic["before_updates"] == 1
    assert diagnostic["after_updates"] == 2


@pytest.mark.parametrize(
    "modes",
    [
        ("workflow", "code"),
        ("code", "workflow"),
        ("workflow", None),
        (None, "workflow"),
        ("invalid", "invalid"),
    ],
)
def test_comparison_refuses_mixed_or_unknown_modes(
    tmp_path: Path, modes: tuple[str | None, str | None]
) -> None:
    paths = _receipts(tmp_path, modes)
    with pytest.raises(ValueError, match="comparison"):
        compare(*paths, bootstrap_draws=100)
