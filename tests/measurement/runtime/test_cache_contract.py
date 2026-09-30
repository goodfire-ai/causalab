"""Cache provenance describes selected-arm memos without warming or clearing them."""

from contextlib import contextmanager
import json
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from hypothesis import example, given, strategies as st
import pytest

from causalab.measurement.runtime import worker

pytestmark = pytest.mark.unit

MEMOS = {
    "causalab.neural.shared.encoding": ("_TOKENIZED",),
    "causalab.protocol.answers": ("_ENCODED_IDS",),
    "causalab.neural.shared.metrics": ("_ENCODED_IDS",),
    "causalab.protocol.resolve": ("_checked_table_text",),
    "causalab.neural.shared.gather": ("_dense_index", "_flat_index"),
    "causalab.neural.shared.kernels": ("_KERNEL_MODULES", "_INSTALLED", "_TORCH_PATHS"),
    "causalab.io.env": ("_TABLE_DIGESTS",),
}


@given(
    st.fixed_dictionaries(
        {
            module: st.one_of(st.none(), st.sets(st.sampled_from(attributes)))
            for module, attributes in MEMOS.items()
        }
    )
)
@example({module: set(attributes) for module, attributes in MEMOS.items()})
@example({module: {attributes[0]} for module, attributes in MEMOS.items()})
def test_capability_discovery_preserves_selected_arm_memos(states):
    from causalab.measurement.runtime.cache import cache_provenance

    with patch.dict(sys.modules):
        memos = []
        expected = {}
        for module_name, attributes in MEMOS.items():
            present = states[module_name]
            sys.modules.pop(module_name, None)
            module = None if present is None else ModuleType(module_name)
            if module is not None:
                sys.modules[module_name] = module
            for attribute in attributes:
                key = f"{module_name}.{attribute}"
                if module is None:
                    expected[key] = "module_not_loaded"
                elif attribute in present:
                    memo = {"already_warm": object()}
                    setattr(module, attribute, memo)
                    memos.append((module, attribute, memo))
                    expected[key] = "available"
                else:
                    expected[key] = "attribute_missing"
        evidence = cache_provenance("resident")
        assert evidence["host_memos"] == expected
        assert evidence["lifetime"] == "resident_worker"
        for module, attribute, memo in memos:
            assert vars(module)[attribute] is memo
            assert list(memo) == ["already_warm"]


def test_cache_discovery_does_not_import_historical_arm_modules(monkeypatch):
    from causalab.measurement.runtime.cache import cache_provenance

    for module in MEMOS:
        monkeypatch.delitem(sys.modules, module, raising=False)
    evidence = cache_provenance("cold_process")
    assert (
        evidence["lifetime"]
        == "fresh_process; may warm during preparation and execution"
    )
    assert all(
        value == "module_not_loaded" for value in evidence["host_memos"].values()
    )
    assert all(module not in sys.modules for module in MEMOS)


def test_resident_receipt_declares_retained_memos_outside_worker_identity(
    tmp_path, monkeypatch
):
    instance = worker.Worker.__new__(worker.Worker)
    instance.plan = {
        "cases": {"inference": {"kind": "workflow"}},
        "warmups": 1,
        "observations": {},
    }
    instance.config = {"input_identity": "frozen-inputs", "device": "cpu"}
    instance.fitting = set()
    instance.identity = {"comparison_identity": "shared-logical-inputs"}
    instance.loaded = SimpleNamespace(document=SimpleNamespace(output_dir="research"))

    def fake_collect(prepare, directory, **kwargs):
        assert "retained host memos" in kwargs["reset_policy"]
        receipt = directory / "measurement.json"
        receipt.write_text(
            json.dumps(
                {
                    "samples": [],
                    "context": kwargs["context"],
                    "reset_policy": kwargs["reset_policy"],
                }
            )
        )
        return receipt

    monkeypatch.setattr(worker, "collect", fake_collect)
    receipt = instance.sample(
        {"case": "inference", "seed": 7, "directory": str(tmp_path)}
    )
    record = json.loads(receipt.read_text())
    assert record["context"]["cache_policy"]["lifetime"] == "resident_worker"
    assert record["context"]["cache_policy"]["observed_at"] == "after_execution"
    assert (
        record["context"]["worker"]
        == instance.identity
        == {"comparison_identity": "shared-logical-inputs"}
    )


def test_cold_worker_reports_cache_lifetime_separately_from_identity(
    tmp_path, monkeypatch, capsys
):
    from causalab.measurement import Operation

    class FakeWorker:
        identity = {"comparison_identity": "shared-logical-inputs"}

        def __init__(self, config):
            pass

        @contextmanager
        def prepare(self, case, seed, directory):
            yield Operation(lambda: None, lambda result: {})

    monkeypatch.setattr(worker, "Worker", FakeWorker)
    worker.serve(
        {
            "device": "cpu",
            "cold": {
                "case": "inference",
                "seed": 7,
                "directory": str(tmp_path / "cold"),
            },
        }
    )
    response = json.loads(capsys.readouterr().out)
    assert (
        response["cache_policy"]["lifetime"]
        == "fresh_process; may warm during preparation and execution"
    )
    assert response["identity"] == FakeWorker.identity


def test_cold_receipt_replaces_resident_cache_policy_with_child_evidence(
    tmp_path, monkeypatch
):
    import torch

    from causalab.measurement.study import controller
    from causalab.measurement.runtime import observations

    instance = controller.ProcessSession.__new__(controller.ProcessSession)
    instance.identity = {"models": {}}
    workflow = tmp_path / "workflow.json"
    workflow.write_text(json.dumps({"output_dir": "research"}))
    instance.config = {
        "controller_root": str(tmp_path),
        "package_root": str(tmp_path),
        "workflow": str(workflow),
        "plan": {"observations": {}},
    }
    instance.python = sys.executable
    instance.logs = tmp_path / "logs"
    instance.logs.mkdir()
    resident = {"lifetime": "resident_worker"}
    cold = {
        "lifetime": "fresh_process; may warm during preparation and execution",
        "host_memos": {"selected-arm-only": "available"},
    }
    response = {
        "status": "completed",
        "identity": instance.identity,
        "cache_policy": cold,
    }
    monkeypatch.setattr(
        controller.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=json.dumps(response)),
    )
    monkeypatch.setattr(
        observations, "observations", lambda *args: {"value": torch.tensor([1.0])}
    )
    monkeypatch.setattr(observations, "observation_specs", lambda *args: {})
    receipt = tmp_path / "measurement.json"
    receipt.write_text(
        json.dumps(
            {
                "context": {"cache_policy": resident},
                "reset_policy": "resident policy",
                "samples": [{"seconds": 1, "peak_memory": {}, "observations": {}}],
            }
        )
    )
    instance._cold("inference", 7, tmp_path, receipt)
    record = json.loads(receipt.read_text())
    assert record["context"]["cache_policy"] == cold
    assert record["context"]["resident_cache_policy"] == resident
    assert "fresh process" in record["reset_policy"]
    assert "retained host memos" not in record["reset_policy"]


def test_different_arm_memo_capabilities_are_comparable(tmp_path):
    from causalab.measurement.analysis.compare import compare
    from causalab.measurement.runtime.cache import RESIDENT_RESET_POLICY
    from tests.measurement.analysis.test_compare import receipt

    paths = [receipt(tmp_path / arm, [[[1.0]]]) for arm in ("before", "after")]
    for path, availability in zip(paths, ("available", "absent"), strict=True):
        record = json.loads(path.read_text())
        record["reset_policy"] = RESIDENT_RESET_POLICY
        record["context"] = {
            "cache_policy": {"host_memos": {"encoding._TOKENIZED": availability}}
        }
        path.write_text(json.dumps(record))
    assert compare(*paths, bootstrap_draws=100)["case"] == "test"


@pytest.mark.parametrize("mode", ["cold", "warm"])
def test_capture_receipt_describes_its_own_cache_lifetime(tmp_path, monkeypatch, mode):
    from contextlib import nullcontext
    import torch

    from causalab.measurement import Operation
    from causalab.measurement.capture import worker as capture_worker, ranges
    from causalab.measurement.runtime import observations

    class FakeWorker:
        def __init__(self, config):
            self.identity = {}
            self.engine, self.bundles = None, {}
            self.plan = {"cases": {"inference": {}}, "observations": {}}
            self.loaded = SimpleNamespace(
                document=SimpleNamespace(output_dir="research")
            )

        @contextmanager
        def prepare(self, *args):
            yield Operation(lambda: None, lambda result: {"value": torch.tensor([1.0])})

    monkeypatch.setattr(worker, "Worker", FakeWorker)
    monkeypatch.setattr(
        capture_worker,
        "get_backend",
        lambda name: SimpleNamespace(capture_control="cuda_profiler_api"),
    )
    monkeypatch.setattr(capture_worker, "_cuda_capture", lambda *args: nullcontext())
    monkeypatch.setattr(ranges, "phase_ranges", lambda *args, **kwargs: nullcontext({}))
    monkeypatch.setattr(observations, "observation_specs", lambda *args, **kwargs: {})
    result = capture_worker.capture(
        {
            "device": "cpu",
            "plan": {"warmups": 1},
            "capture": {
                "directory": str(tmp_path),
                "backend": "nsys",
                "mode": mode,
                "case": "inference",
                "seed": 7,
            },
        }
    )
    coverage = result["coverage"]
    if mode == "cold":
        assert "fresh process" in coverage["reset_policy"]
        assert coverage["cache_policy"]["lifetime"].startswith("fresh_process")
    else:
        assert "retained host memos" in coverage["reset_policy"]
        assert coverage["cache_policy"]["lifetime"] == "resident_worker"
