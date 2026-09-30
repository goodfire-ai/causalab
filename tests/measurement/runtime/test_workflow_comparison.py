"""Workflow contrasts share inputs while independently attesting each definition."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

from hypothesis import given, strategies as st
import pytest

import causalab
from causalab.measurement.census import CensusError
from causalab.measurement.runtime.comparison import (
    ComparisonInputError,
    checkpoint_aliases,
    logical_input_rows,
    workflow_inputs,
)
from causalab.measurement.runtime.pins import PinContract
from causalab.protocol.identity import resolve_locator

pytestmark = pytest.mark.unit


def evidence(dtype="fp32", *, commit="a" * 40):
    return {
        "source_commit": commit,
        "data": {"prompts": {"digest": "data", "columns": ["input"]}},
        "artifacts": {"fixed.safetensors": "artifact"},
        "models": {
            dtype: {
                "realization": {"key": "model", "revision": "pinned", "dtype": dtype},
                "files": {"model.safetensors": "weights"},
                "tokenizer": "tokenizer",
            }
        },
    }


def bindings():
    return {"inference": [{"base": {"dataset": "prompts", "field": "input"}}]}


def inputs(identity, *, pins=None, data=None):
    return workflow_inputs(
        identity,
        {"documents": {"inference.json": "one"}, "datasets": {"prompts": "data"}}
        if pins is None
        else pins,
        bindings() if data is None else data,
    )


def test_precision_and_quantization_change_realization_without_changing_checkpoint():
    before, after = evidence(), evidence("bf16")
    after["models"]["bf16"]["realization"]["quantization"] = {"kind": "int8"}
    assert inputs(before) == inputs(after)
    left = [{"model": "fp32", "token_ids": [1, 2]}]
    right = [{"model": "bf16", "token_ids": [1, 2]}]
    assert logical_input_rows(
        left, checkpoint_aliases(before["models"])
    ) == logical_input_rows(right, checkpoint_aliases(after["models"]))


@pytest.mark.parametrize(
    "field", ["source_commit", "data", "artifacts", "weights", "tokenizer", "revision"]
)
def test_fixed_execution_or_scientific_inputs_change_comparison_identity(field):
    before = evidence()
    after = deepcopy(before)
    if field in {"source_commit", "data", "artifacts"}:
        after[field] = "changed"
    elif field == "weights":
        after["models"]["fp32"]["files"]["model.safetensors"] = "changed"
    elif field == "tokenizer":
        after["models"]["fp32"]["tokenizer"] = "changed"
    else:
        after["models"]["fp32"]["realization"][field] = "changed"
    assert inputs(before) != inputs(after)


def test_documents_can_differ_but_external_sources_cannot():
    before = {"documents": {"before.json": "one"}, "scripts": {"local.py": "fixed"}}
    after = {"documents": {"after.json": "two"}, "scripts": {"local.py": "fixed"}}
    assert inputs(evidence(), pins=before) == inputs(evidence(), pins=after)
    after["scripts"]["local.py"] = "changed"
    assert inputs(evidence(), pins=before) != inputs(evidence(), pins=after)


def test_role_swaps_and_changed_point_order_are_not_equivalent_inputs():
    first = {
        "inference": [
            {"base": {"dataset": "a"}, "source": {"dataset": "b"}},
            {"base": {"dataset": "c"}},
        ]
    }
    swapped = deepcopy(first)
    swapped["inference"][0] = {"base": {"dataset": "b"}, "source": {"dataset": "a"}}
    assert inputs(evidence(), data=first) != inputs(evidence(), data=swapped)
    assert inputs(evidence(), data=first) != inputs(
        evidence(), data={"inference": first["inference"][::-1]}
    )


@given(st.lists(st.lists(st.integers(0, 100), max_size=8), min_size=1, max_size=8))
def test_token_rows_ignore_repeated_dispatch_but_hold_tokens(rows):
    aliases = checkpoint_aliases(evidence()["models"])
    raw = [{"model": "fp32", "token_ids": row} for row in rows]
    expected = logical_input_rows(raw, aliases)
    assert logical_input_rows(list(reversed(raw)) + raw, aliases) == expected
    assert (
        logical_input_rows(raw + [{"model": "fp32", "token_ids": [101]}], aliases)
        != expected
    )


def test_unattested_probe_model_is_a_structured_error():
    with pytest.raises(ComparisonInputError) as caught:
        logical_input_rows([{"model": "unattested", "token_ids": [1]}], {})
    assert caught.value.field == "logical_token_rows.model"


@pytest.mark.parametrize("arm", ["before", "after", "eager"])
def test_each_workflow_holds_its_own_authored_source_pins(arm):
    module = "causalab.measurement.collection"
    source = hashlib.sha256(resolve_locator(module).path.read_bytes()).hexdigest()
    actual = {"code": {module: source}}
    package = Path(causalab.__file__).resolve().parent.parent
    contract = PinContract.resolve(
        actual, actual, arm=arm, package_root=package, comparison="workflow"
    )
    contract.check(actual)
    with pytest.raises(CensusError, match="pins.code"):
        PinContract.resolve(
            {"code": {module: "0" * 64}},
            actual,
            arm=arm,
            package_root=package,
            comparison="workflow",
        )


def test_worker_checks_its_arm_definition_instead_of_shared_study_identity(
    tmp_path, monkeypatch
):
    from causalab.measurement.runtime import worker as runtime
    from causalab.measurement.runtime.benchmark import (
        benchmark_identity,
        BenchmarkIdentityError,
    )
    from causalab.protocol.pipeline import read_document
    from tests.measurement.runtime.test_worker_pin_lifecycle import _worker, REPO

    fixture = _worker(tmp_path)
    fixture.path.write_text(json.dumps(fixture.raw))
    authored = {
        name: dict(
            read_document(
                fixture.path.parent / step["document"],
                fixture.path.parent,
                step.get("set", {}),
            ).raw
        )
        for name, step in fixture.raw["steps"].items()
        if step["type"] == "intervention_protocol"
    }
    expected = benchmark_identity(fixture.raw, authored, {})
    monkeypatch.setattr(
        runtime,
        "execution_identity",
        lambda device: {"implementation": {"location": str(REPO / "causalab")}},
    )
    monkeypatch.setattr(
        runtime, "attest_installation", lambda config, package: config["source_commit"]
    )

    class ReachedLoading(RuntimeError):
        pass

    def loaded(self, seed):
        raise ReachedLoading

    monkeypatch.setattr(runtime.Worker, "load", loaded)
    config = {
        "device": "cpu",
        "package_root": str(REPO),
        "source_commit": "a" * 40,
        "arm": "after",
        "workflow": str(fixture.path),
        "data_root": str(tmp_path),
        "artifacts_root": str(tmp_path),
        "plan": {"comparison": "workflow", "observations": {}, "seeds": [0]},
        "input_identity": "shared-study",
        "benchmark_identity": expected,
    }
    from causalab.measurement.study.scheduler import digest

    contract = {
        "kind": "workflow",
        "source_commit": "a" * 40,
        "definitions": {"after": expected},
    }
    config["comparison_contract"] = contract
    config["input_identity"] = digest(contract)
    with pytest.raises(ReachedLoading):
        runtime.Worker(config)
    changed = deepcopy(fixture.raw)
    changed["steps"]["report"]["inputs"]["added"] = "changed"
    fixture.path.write_text(json.dumps(changed))
    with pytest.raises(BenchmarkIdentityError):
        runtime.Worker(config)

    fixture.path.write_text(json.dumps(fixture.raw))
    config["plan"]["comparison"] = "code"
    config["input_identity"] = "different benchmark"
    with pytest.raises(BenchmarkIdentityError):
        runtime.Worker(config)
    config["input_identity"] = expected
    config["benchmark_identity"] = "ignored in code mode"
    with pytest.raises(ReachedLoading):
        runtime.Worker(config)


@pytest.mark.parametrize("arm", ["after", "eager"])
def test_worker_workflow_mode_does_not_rebase_stale_source_pins(tmp_path, arm):
    from tests.measurement.runtime.test_worker_pin_lifecycle import _worker

    worker = _worker(tmp_path, arm=arm)
    worker.plan = {"comparison": "workflow"}
    for module in worker.raw["pins"]["scripts"]:
        worker.raw["pins"]["scripts"][module] = "0" * 64
    with pytest.raises(CensusError, match="pins.scripts"):
        worker.load(0)


@pytest.mark.parametrize(
    "change", [None, "kind", "source_commit", "definition", "identity"]
)
def test_shared_contract_is_bound_to_the_workers_own_definition_and_commit(change):
    from causalab.measurement.runtime.comparison import check_workflow_contract
    from causalab.measurement.study.scheduler import digest

    contract = {
        "kind": "workflow",
        "source_commit": "a" * 40,
        "definitions": {"before": "first", "after": "second"},
    }
    identity = digest(contract)
    if change in {"kind", "source_commit"}:
        contract[change] = "changed"
        identity = digest(contract)
    elif change == "definition":
        contract["definitions"]["after"] = "changed"
        identity = digest(contract)
    elif change == "identity":
        identity = "changed"
    arguments = {
        "arm": "after",
        "source_commit": "a" * 40,
        "benchmark_identity": "second",
        "input_identity": identity,
    }
    if change is None:
        check_workflow_contract(contract, **arguments)
    else:
        with pytest.raises(ComparisonInputError):
            check_workflow_contract(contract, **arguments)
