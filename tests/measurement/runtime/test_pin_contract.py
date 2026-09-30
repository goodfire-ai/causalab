"""Per-arm source pins preserve the shared benchmark and freeze later loads."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from hypothesis import given, strategies as st
import pytest

import causalab
from causalab.measurement.census import CensusError, collect_pins
from causalab.measurement.runtime.pins import PinContract, SourceOwnershipError
from causalab.protocol.identity import resolve_locator

pytestmark = pytest.mark.unit
PACKAGE_ROOT = Path(causalab.__file__).resolve().parent.parent
MODULE = "causalab.measurement.collection"


def census():
    return {
        "documents": {"protocol.json": "a" * 64},
        "datasets": {"data": "b" * 64},
        "code": {
            MODULE: hashlib.sha256(
                resolve_locator(MODULE).path.read_bytes()
            ).hexdigest()
        },
    }


def test_baseline_holds_authored_source_but_candidate_resolves_own_source():
    actual = census()
    authored = {**actual, "code": {MODULE: "0" * 64}}
    with pytest.raises(CensusError, match="pins.code"):
        PinContract.resolve(authored, actual, arm="before", package_root=PACKAGE_ROOT)
    contract = PinContract.resolve(
        authored, actual, arm="after", package_root=PACKAGE_ROOT
    )
    assert contract.source == {"code": actual["code"]}
    assert contract.shared == {key: actual[key] for key in ("documents", "datasets")}
    assert contract.pins == actual
    contract.check(actual)


@pytest.mark.parametrize("arm", ["before", "after", "eager"])
def test_shared_document_mismatch_is_never_rebased(arm):
    actual = census()
    authored = {**actual, "documents": {"protocol.json": "0" * 64}}
    with pytest.raises(CensusError, match="pins.documents"):
        PinContract.resolve(authored, actual, arm=arm, package_root=PACKAGE_ROOT)


@pytest.mark.parametrize(
    "key",
    [
        "external.helper",
        "scripts/local.py",
        "external.helper#sibling.py",
        "causalab.example#sibling.py",
    ],
)
def test_external_and_closure_sources_are_shared_even_without_authored_pins(key):
    actual = {"scripts": {key: "a" * 64}}
    contract = PinContract.resolve(None, actual, arm="after", package_root=PACKAGE_ROOT)
    assert contract.source == {}
    assert contract.shared == actual
    assert contract.pins == actual
    with pytest.raises(CensusError, match="pins.scripts"):
        PinContract.resolve(
            {"scripts": {key: "b" * 64}}, actual, arm="after", package_root=PACKAGE_ROOT
        )


def test_census_hash_must_match_real_selected_source():
    actual = {"code": {MODULE: "0" * 64}}
    with pytest.raises(SourceOwnershipError) as error:
        PinContract.resolve(None, actual, arm="after", package_root=PACKAGE_ROOT)
    assert error.value.arm == "after"
    assert error.value.key == MODULE
    assert "bytes" in str(error.value)


def test_import_resolution_outside_selected_installation_is_refused(tmp_path):
    with pytest.raises(SourceOwnershipError, match="selected"):
        PinContract.resolve(None, census(), arm="after", package_root=tmp_path)


def test_missing_selected_module_is_structured():
    with pytest.raises(SourceOwnershipError):
        PinContract.resolve(
            None,
            {"code": {"causalab.missing_pin_contract_module": "a" * 64}},
            arm="after",
            package_root=PACKAGE_ROOT,
        )


def test_frozen_contract_is_independent_of_mutated_input_and_returned_maps():
    actual = census()
    contract = PinContract.resolve(None, actual, arm="after", package_root=PACKAGE_ROOT)
    original = census()
    actual["code"][MODULE] = "0" * 64
    contract.pins["code"][MODULE] = "1" * 64
    contract.source["code"][MODULE] = "2" * 64
    contract.shared["documents"]["protocol.json"] = "3" * 64
    contract.check(original)
    with pytest.raises(CensusError):
        contract.check(actual)


def test_subset_allows_only_explicit_generated_files_and_holds_source():
    contract = PinContract.resolve(
        None, census(), arm="after", package_root=PACKAGE_ROOT
    )
    actual = {"code": census()["code"], "files": {"fit/weights.safetensors": "f" * 64}}
    contract.check_subset(actual, produced_files={"fit/weights.safetensors"})
    with pytest.raises(CensusError, match="pins.files"):
        contract.check_subset(actual)
    with pytest.raises(CensusError, match="pins.code"):
        contract.check_subset(
            {**actual, "code": {MODULE: "0" * 64}},
            produced_files={"fit/weights.safetensors"},
        )


def test_produced_file_exception_cannot_override_existing_external_pin():
    actual = {"files": {"fit/input": "a" * 64}}
    contract = PinContract.resolve(None, actual, arm="after", package_root=PACKAGE_ROOT)
    with pytest.raises(CensusError, match="pins.files"):
        contract.check_subset(
            {"files": {"fit/input": "b" * 64}}, produced_files={"fit/input"}
        )


@given(
    st.dictionaries(
        st.text(alphabet="abcdef", min_size=1, max_size=10),
        st.binary().map(lambda contents: hashlib.sha256(contents).hexdigest()),
        min_size=1,
    )
)
def test_every_frozen_shared_pin_is_held_in_full_and_subset_loads(entries):
    original = {"files": entries}
    contract = PinContract.resolve(
        None, original, arm="after", package_root=PACKAGE_ROOT
    )
    for key, value in entries.items():
        contract.check_subset({"files": {key: value}})
        changed = ("0" if value[0] != "0" else "1") + value[1:]
        with pytest.raises(CensusError):
            contract.check_subset({"files": {key: changed}})
        with pytest.raises(CensusError):
            contract.check({"files": {**entries, key: changed}})


@pytest.fixture
def deferred_replay(tmp_path):
    import numpy as np
    from safetensors.numpy import save_file

    from causalab.io.env import ResolutionEnv
    from causalab.workflow.document import load_workflow
    from causalab.workflow.runner import OverlayArtifacts
    from tests.measurement.deployment.test_remote_pins import _study

    workflow, env = _study(tmp_path)
    protocol = tmp_path / "methods/locate.json"
    document = json.loads(protocol.read_text())
    method = document["method"]
    method["params"] = {
        "external": {"file_path": "external.safetensors"},
        "generated": {"file_path": "fit/weights.safetensors"},
    }
    method["writes"]["patch"]["do"] = {
        "add_scaled": {"op": "generated", "alpha": "external"}
    }
    method["reads"].pop("v_cf")
    # v_cf was the only read of the counterfactual run; without it that run
    # has nothing to record.
    method["intervened_models"].pop("original_counterfactual")
    document["method"] = {
        key: method[key]
        for key in (
            "intervened_models",
            "positions",
            "sites",
            "params",
            "reads",
            "writes",
            "save",
        )
    }
    protocol.write_text(json.dumps(document))
    raw = {
        "version": "1",
        "output_dir": "run",
        "steps": {
            "fit": {
                "type": "script",
                "script": {"module": "causalab.workflow.scripts.select"},
                "inputs": {},
                "outputs": {"weights": "weights.safetensors"},
            },
            "replay": {
                "type": "intervention_protocol",
                "document": "methods/locate.json",
            },
        },
    }
    save_file({"value": np.array([1.0])}, str(tmp_path / "external.safetensors"))
    fit_root = tmp_path / "run"
    (fit_root / "fit").mkdir(parents=True)
    save_file({"value": np.array([2.0])}, str(fit_root / "fit/weights.safetensors"))
    full = load_workflow(raw, env, workflow_dir=workflow.parent)
    overlay = ResolutionEnv(
        datasets=env.datasets,
        artifacts=OverlayArtifacts(fit_root, env.artifacts, frozenset(raw["steps"])),
        model_info=env.model_info,
    )
    subset = load_workflow(
        {**raw, "steps": {"replay": raw["steps"]["replay"]}},
        overlay,
        workflow_dir=workflow.parent,
    )
    return (
        collect_pins(full, env.datasets),
        collect_pins(subset, overlay.datasets),
        env,
        tmp_path / "external.safetensors",
    )


@pytest.mark.parametrize("authored_kind", ["placeholder", "actual", "absent"])
@pytest.mark.parametrize("arm", ["before", "after", "eager"])
def test_deferred_external_file_is_attested_and_replay_subset_holds_actual_bytes(
    deferred_replay, authored_kind, arm
):
    full, subset, env, external = deferred_replay
    assert full["files"] == {"external.safetensors": "0" * 64}
    actual = env.artifacts.file_digest("external.safetensors")
    assert subset["files"]["external.safetensors"] == actual
    assert "fit/weights.safetensors" in subset["files"]
    authored = json.loads(json.dumps(full))
    if authored_kind == "actual":
        authored["files"]["external.safetensors"] = actual
    elif authored_kind == "absent":
        authored = None
    contract = PinContract.resolve(
        authored,
        full,
        arm=arm,
        package_root=PACKAGE_ROOT,
        file_digests={"external.safetensors": actual},
    )
    assert contract.shared["files"] == {"external.safetensors": actual}
    contract.check(full, file_digests={"external.safetensors": actual})
    contract.check_subset(subset, produced_files={"fit/weights.safetensors"})
    external.write_bytes(external.read_bytes() + b"changed")
    changed = env.artifacts.file_digest("external.safetensors")
    with pytest.raises(CensusError, match="pins.files"):
        contract.check(full, file_digests={"external.safetensors": changed})
    with pytest.raises(CensusError, match="pins.files"):
        contract.check_subset(
            {**subset, "files": {**subset["files"], "external.safetensors": changed}},
            produced_files={"fit/weights.safetensors"},
        )


@pytest.mark.parametrize("arm", ["before", "after"])
def test_authored_concrete_external_hash_is_checked_even_when_loader_defers(
    deferred_replay, arm
):
    full, _, env, _ = deferred_replay
    authored = {**full, "files": {"external.safetensors": "f" * 64}}
    with pytest.raises(CensusError, match="pins.files"):
        PinContract.resolve(
            authored,
            full,
            arm=arm,
            package_root=PACKAGE_ROOT,
            file_digests={
                "external.safetensors": env.artifacts.file_digest(
                    "external.safetensors"
                )
            },
        )


def test_deferred_file_requires_real_digest_before_contract_can_freeze():
    from causalab.measurement.runtime.pins import DeferredFilePinError

    with pytest.raises(DeferredFilePinError):
        PinContract.resolve(
            None,
            {"files": {"external": "0" * 64}},
            arm="after",
            package_root=PACKAGE_ROOT,
        )
