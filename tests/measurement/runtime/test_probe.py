"""Real tensor operations certify input capture and observer removal."""

from contextlib import contextmanager
import hashlib
import sys
from types import SimpleNamespace

import pytest
import torch

from causalab.measurement.collection import Operation
from causalab.measurement.runtime.probe import (
    observe_execution,
    probe_case,
    reuse_evidence,
)

pytestmark = pytest.mark.unit


class Model(torch.nn.Module):
    def forward(self, input_ids, attention_mask):
        return input_ids.float().square() * attention_mask


def test_capture_preserves_values_and_removes_padding_from_logical_rows():
    model = Model()
    ids = torch.tensor([[0, 2, 3], [4, 5, 6]])
    mask = torch.tensor([[0, 1, 1], [1, 1, 1]])
    expected = model(input_ids=ids, attention_mask=mask)
    previous = sys.getprofile()
    with observe_execution({"model": SimpleNamespace(model=model)}) as record:
        result = model(input_ids=ids, attention_mask=mask)
    assert torch.equal(result, expected)
    assert not model._forward_pre_hooks
    assert sys.getprofile() is previous
    assert record["logical_token_rows"] == [
        {"model": "model", "token_ids": [2, 3]},
        {"model": "model", "token_ids": [4, 5, 6]},
    ]
    captured = record["model_inputs"][0]["tensors"]["input_ids"]
    assert captured["shape"] == [2, 3]
    assert captured["sha256"] == hashlib.sha256(ids.numpy().tobytes()).hexdigest()


def test_hybrid_prompt_masks_preserve_padding_identity_and_capture_all_tensors():
    class HybridModel(torch.nn.Module):
        def forward(self, input_ids, attention_mask):
            mask = attention_mask
            if isinstance(mask, dict):
                mask = mask["linear_attention"]
            return input_ids.float().square() * mask

    model = HybridModel()
    bundle = {"model": SimpleNamespace(model=model)}
    ids = torch.tensor([[0, 2, 3], [4, 5, 6]])
    padding = torch.tensor([[0, 1, 1], [1, 1, 1]])
    causal = torch.ones(2, 1, 3, 3).tril()
    with observe_execution(bundle) as before:
        expected = model(input_ids=ids, attention_mask=padding)
    masks = {"full_attention": causal, "linear_attention": padding}
    with observe_execution(bundle) as after:
        actual = model(input_ids=ids, attention_mask=masks)
    assert torch.equal(expected, actual)
    assert before["logical_token_rows"] == after["logical_token_rows"]
    assert after["logical_token_rows"][0]["token_ids"] == [2, 3]
    captured = after["model_inputs"][0]["tensors"]
    for kind, tensor in masks.items():
        assert (
            captured[f"attention_mask.{kind}"]["sha256"]
            == hashlib.sha256(tensor.numpy().tobytes()).hexdigest()
        )
    assert not model._forward_pre_hooks


def test_transformed_mask_without_padding_evidence_does_not_claim_unpadded_rows():
    class MaskedModel(torch.nn.Module):
        def forward(self, input_ids, attention_mask):
            return input_ids.float()

    model = MaskedModel()
    with observe_execution({"model": SimpleNamespace(model=model)}) as observed:
        model(
            input_ids=torch.tensor([[0, 2, 3]]),
            attention_mask={"full_attention": torch.ones(1, 1, 3, 3)},
        )
    assert "attention_mask.full_attention" in observed["model_inputs"][0]["tensors"]
    assert observed["logical_token_rows"] == []


def test_failed_execution_restores_hooks_and_python_profiler():
    model = Model()
    previous = sys.getprofile()
    with (
        pytest.raises(TypeError),
        observe_execution({"model": SimpleNamespace(model=model)}),
    ):
        model(input_ids=torch.tensor([[1, 2]]), attention_mask=None)
    assert sys.getprofile() is previous
    assert not model._forward_pre_hooks


def test_graph_capture_does_not_allocate_observer_snapshots(monkeypatch):
    model = Model()
    ids = torch.tensor([[2, 3]])
    mask = torch.ones_like(ids)
    expected = model(input_ids=ids, attention_mask=mask)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)

    def refuse_clone(*args, **kwargs):
        raise AssertionError("observer allocated inside CUDA graph capture")

    with observe_execution({"model": SimpleNamespace(model=model)}) as observed:
        with monkeypatch.context() as capture:
            capture.setattr(torch.Tensor, "clone", refuse_clone)
            actual = model(input_ids=ids, attention_mask=mask)
        assert torch.equal(actual, expected)
        monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
        model(input_ids=ids, attention_mask=mask)

    assert observed["graph_capture_callbacks_skipped"] == 1
    assert len(observed["model_inputs"]) == 1
    assert observed["logical_token_rows"] == [{"model": "model", "token_ids": [2, 3]}]
    assert not model._forward_pre_hooks


def test_logical_token_identity_ignores_physical_batching_and_repeated_forwards():
    model = Model()
    bundle = {"model": SimpleNamespace(model=model)}
    with observe_execution(bundle) as together:
        model(
            input_ids=torch.tensor([[0, 2, 3], [4, 5, 6]]),
            attention_mask=torch.tensor([[0, 1, 1], [1, 1, 1]]),
        )
    with observe_execution(bundle) as separate:
        for row in ([2, 3], [4, 5, 6], [2, 3]):
            tokens = torch.tensor([row])
            model(input_ids=tokens, attention_mask=torch.ones_like(tokens))

    def evidence(observed):
        return reuse_evidence(
            {
                "case": "same",
                "seed": 0,
                "conditions": "same",
                "native_profile": {"status": "not_requested"},
                "observed": observed,
                "operators": {},
                "device_kernels": {},
            }
        )

    left, right = evidence(together), evidence(separate)
    assert left["model_inputs"] != right["model_inputs"]
    assert left["logical_token_rows"] == right["logical_token_rows"]


def test_probe_runs_real_operators_and_refuses_to_reuse_changed_tokens(tmp_path):
    model = Model()
    ids = torch.tensor([[1, 2]])
    calls = []

    @contextmanager
    def prepare(case, seed, directory):
        calls.append((case, seed))
        yield Operation(
            lambda: model(input_ids=ids, attention_mask=torch.ones_like(ids)),
            lambda result: {"values": result},
        )

    def run(name):
        return probe_case(
            prepare,
            {"model": SimpleNamespace(model=model)},
            tmp_path / name,
            case="operation",
            seed=7,
            device="cpu",
        )

    first, same = run("first"), run("same")
    assert reuse_evidence(first) == reuse_evidence(same)
    assert first["operators"]["aten::pow"] > 0
    assert not first["device_kernels"]
    ids[0, 1] = 3
    assert reuse_evidence(run("changed")) != reuse_evidence(first)
    assert calls == [("operation", 7)] * 6  # warmup and freshly prepared probe


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_disabled_probe_keeps_input_evidence_without_native_profiler(
    tmp_path, monkeypatch, device
):
    model = Model()
    ids = torch.tensor([[1, 2]])
    calls = []

    def refuse_profile(*args, **kwargs):
        raise AssertionError("native profiling was disabled")

    monkeypatch.setattr(torch.profiler, "profile", refuse_profile)
    # Exercise CUDA's evidence policy without requiring an accelerator.
    monkeypatch.setattr(torch.cuda, "synchronize", lambda target: None)

    @contextmanager
    def prepare(case, seed, directory):
        calls.append(directory.name)
        yield Operation(
            lambda: model(input_ids=ids, attention_mask=torch.ones_like(ids)),
            lambda result: {"values": result},
        )

    record = probe_case(
        prepare,
        {"model": SimpleNamespace(model=model)},
        tmp_path / "probe",
        case="operation",
        seed=7,
        device=device,
        profile=False,
    )
    evidence = reuse_evidence(record)
    assert calls == ["warmup", "observed"]
    assert evidence["logical_token_rows"] == [{"model": "model", "token_ids": [1, 2]}]
    assert evidence["native_profile"] == {"status": "not_requested"}
    assert evidence["operators"] == evidence["device_kernels"] == []
    assert not model._forward_pre_hooks

    with pytest.raises(ValueError, match="observed no token inputs"):
        probe_case(
            prepare,
            {},
            tmp_path / "missing",
            case="operation",
            seed=7,
            device=device,
            profile=False,
        )
