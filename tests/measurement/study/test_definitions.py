"""One shared measurement plan selects independently authored workflow arms."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from hypothesis import given, strategies as st
import pytest

from causalab.measurement.spec import MeasurementSpecError, parse_measurement

pytestmark = pytest.mark.unit


def _study() -> dict[str, Any]:
    return {
        "version": "1",
        "output_dir": "outputs",
        "steps": {"read": {"type": "intervention_protocol", "document": "read.json"}},
        "measurement": {
            "version": 1,
            "comparison": "workflow",
            "arms": {
                "before": {"revision": "HEAD"},
                "after": {"revision": "HEAD", "workflow": "candidate.json"},
                "eager": {"revision": "HEAD"},
            },
            "seeds": [0],
            "repeats": 1,
            "cases": {"read": {"kind": "operation", "step": "read"}},
            "observations": {
                "logits": {
                    "step": "read",
                    "file": "logits.safetensors",
                    "kind": "tensor",
                }
            },
        },
    }


def _files(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    raw = _study()
    document = tmp_path / "study.json"
    document.write_text(json.dumps(raw))
    candidate = {key: value for key, value in raw.items() if key != "measurement"}
    (tmp_path / "candidate.json").write_text(json.dumps(candidate))
    (tmp_path / "read.json").write_text(
        json.dumps(
            {
                "header": {"protocol_version": "4"},
                "model": {"key": "tiny", "revision": "main", "dtype": "fp32"},
                "data": {"base": {"dataset": "prompts", "field": "input"}},
                "method": {},
            }
        )
    )
    return document, parse_measurement(raw["measurement"], raw["steps"])


@given(st.sampled_from(["code", None]))
def test_default_code_normalization_preserves_existing_plan(mode: str | None) -> None:
    raw = _study()
    plan = raw["measurement"]
    del plan["arms"]["after"]["workflow"]
    plan.pop("comparison")
    expected = parse_measurement(plan, raw["steps"])
    if mode is not None:
        plan["comparison"] = mode
    assert parse_measurement(plan, raw["steps"]) == expected
    assert "comparison" not in expected


@given(
    st.lists(
        st.from_regex(r"[a-z][a-z0-9_-]{0,8}", fullmatch=True), min_size=1, max_size=4
    )
)
def test_portable_workflow_paths_round_trip(parts: list[str]) -> None:
    raw = _study()
    reference = "/".join(parts) + ".json"
    raw["measurement"]["arms"]["after"]["workflow"] = reference
    plan = parse_measurement(raw["measurement"], raw["steps"])
    assert plan["arms"]["after"]["workflow"] == reference
    assert parse_measurement(plan, raw["steps"]) == plan


@pytest.mark.parametrize(
    "reference",
    [
        "",
        "/a.json",
        "../a.json",
        "a/../b.json",
        "a\\b.json",
        "C:/a.json",
        "a\x00.json",
        "a\n.json",
        "a//b.json",
        "./a.json",
        "a/",
        "a?.json",
    ],
)
def test_unsafe_workflow_paths_are_refused(reference: str) -> None:
    raw = _study()
    raw["measurement"]["arms"]["after"]["workflow"] = reference
    with pytest.raises(MeasurementSpecError, match="workflow"):
        parse_measurement(raw["measurement"], raw["steps"])


@pytest.mark.parametrize(
    "change",
    ["missing_after", "before_workflow", "code_workflow", "unknown_comparison"],
)
def test_comparison_contract_refuses_ambiguous_authorship(change: str) -> None:
    raw = _study()
    plan = raw["measurement"]
    if change == "missing_after":
        del plan["arms"]["after"]["workflow"]
    elif change == "before_workflow":
        plan["arms"]["before"]["workflow"] = "baseline.json"
    elif change == "code_workflow":
        plan["comparison"] = "code"
    else:
        plan["comparison"] = "both"
    with pytest.raises(MeasurementSpecError, match="comparison|workflow"):
        parse_measurement(plan, raw["steps"])


def test_identical_definitions_have_identical_identity_and_eager_defaults(
    tmp_path: Path,
) -> None:
    from causalab.measurement.study.definitions import load_definitions

    document, plan = _files(tmp_path)
    definitions = load_definitions(document, plan)
    assert definitions["before"].path == document
    assert definitions["after"].path == tmp_path / "candidate.json"
    assert definitions["eager"] == definitions["before"]
    assert (
        definitions["before"].benchmark_identity
        == definitions["after"].benchmark_identity
    )
    assert definitions["after"].fitting == frozenset()


@given(st.sampled_from(["fp16", "bf16"]))
def test_changed_intervention_override_changes_arm_identity(
    tmp_path_factory: Any, dtype: str
) -> None:
    from causalab.measurement.study.definitions import load_definitions

    tmp_path = tmp_path_factory.mktemp("definition")
    document, plan = _files(tmp_path)
    candidate_path = tmp_path / "candidate.json"
    candidate = json.loads(candidate_path.read_text())
    candidate["steps"]["read"]["set"] = {"model.dtype": dtype}
    candidate_path.write_text(json.dumps(candidate))
    definitions = load_definitions(document, plan)
    assert definitions["before"].protocols["read"]["model"]["dtype"] == "fp32"
    assert definitions["after"].protocols["read"]["model"]["dtype"] == dtype
    assert (
        definitions["before"].benchmark_identity
        != definitions["after"].benchmark_identity
    )


@pytest.mark.parametrize(
    "invalid",
    [
        "missing",
        "nested",
        "step",
        "observation",
        "evaluation",
        "fitting_evaluation",
        "symlink",
    ],
)
def test_definition_errors_identify_the_arm_and_document(
    tmp_path: Path, invalid: str
) -> None:
    from causalab.measurement.study.definitions import DefinitionError, load_definitions

    document, plan = _files(tmp_path)
    path = tmp_path / "candidate.json"
    candidate = json.loads(path.read_text())
    if invalid == "missing":
        path.unlink()
    elif invalid == "symlink":
        outside = tmp_path.parent / f"{tmp_path.name}-outside.json"
        outside.write_text(json.dumps(candidate))
        path.unlink()
        path.symlink_to(outside)
    else:
        if invalid == "nested":
            candidate["measurement"] = copy.deepcopy(plan)
        elif invalid in {"step", "observation", "evaluation"}:
            candidate["steps"]["other"] = candidate["steps"].pop("read")
            if invalid == "observation":
                plan["cases"] = {"all": {"kind": "workflow"}}
                plan["profile"]["cases"] = []
            elif invalid == "evaluation":
                # Keep measured steps valid while selecting a missing evaluation step.
                candidate["steps"]["read"] = candidate["steps"].pop("other")
                baseline = json.loads(document.read_text())
                baseline["steps"]["evaluate"] = dict(baseline["steps"]["read"])
                document.write_text(json.dumps(baseline))
                plan["observations"]["eval"] = {
                    "step": "evaluate",
                    "file": "logits.safetensors",
                    "kind": "tensor",
                }
                plan["evaluation"] = {"arm": "before", "cases": {"read": ["evaluate"]}}
        elif invalid == "fitting_evaluation":
            fitting = json.loads((tmp_path / "read.json").read_text())
            fitting["method"]["train"] = {"steps": {"updates": 1}}
            (tmp_path / "fit.json").write_text(json.dumps(fitting))
            candidate["steps"]["read"]["document"] = "fit.json"
            plan["evaluation"] = {"arm": "before", "cases": {"read": ["read"]}}
        path.write_text(json.dumps(candidate))
    with pytest.raises(DefinitionError) as raised:
        load_definitions(document, plan)
    assert raised.value.arm == "after"
    assert raised.value.path == path
    assert raised.value.reason
    if invalid == "symlink":
        assert "within the study directory" in raised.value.reason
    elif invalid == "fitting_evaluation":
        assert "non-fitting" in raised.value.reason
    elif invalid == "nested":
        assert "measurement block" in raised.value.reason


def test_selected_workflow_resolves_its_own_intervention_directory(
    tmp_path: Path,
) -> None:
    from causalab.measurement.study.definitions import load_definitions

    document, plan = _files(tmp_path)
    variants = tmp_path / "variants"
    variants.mkdir()
    (tmp_path / "candidate.json").rename(variants / "candidate.json")
    specification = json.loads((tmp_path / "read.json").read_text())
    specification["model"]["dtype"] = "bf16"
    (variants / "read.json").write_text(json.dumps(specification))
    plan["arms"]["after"]["workflow"] = "variants/candidate.json"
    plan["arms"]["eager"]["workflow"] = "variants/candidate.json"
    definitions = load_definitions(document, plan)
    assert definitions["eager"] == definitions["after"]
    assert definitions["after"].protocols["read"]["model"]["dtype"] == "bf16"
