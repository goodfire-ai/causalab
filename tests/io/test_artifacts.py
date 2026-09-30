"""Round-trip tests for the safetensors + meta JSON helpers in causalab.io.artifacts."""

from __future__ import annotations

import json
import logging
import os

import pytest
import torch

from causalab.io import artifacts
from causalab.io.artifacts import (
    ARTIFACT_FORMAT_VERSION,
    load_module,
    load_tensors_with_meta,
    save_module,
    save_tensors_with_meta,
)


pytestmark = pytest.mark.unit


class TestSaveTensorsWithMeta:
    def test_round_trip_preserves_tensors_and_meta(self, tmp_path: object) -> None:
        tensors = {
            "alpha": torch.randn(3, 4),
            "beta": torch.arange(8, dtype=torch.float32).reshape(2, 4),
        }
        meta = {"label": "demo", "epoch": 7, "ratios": [0.1, 0.2]}

        st_path, meta_path = save_tensors_with_meta(
            tensors, meta, str(tmp_path), "demo"
        )

        assert os.path.exists(st_path) and st_path.endswith(".safetensors")
        assert os.path.exists(meta_path) and meta_path.endswith(".meta.json")

        loaded_tensors, loaded_meta = load_tensors_with_meta(str(tmp_path), "demo")

        assert set(loaded_tensors.keys()) == set(tensors.keys())
        for k, v in tensors.items():
            assert torch.equal(loaded_tensors[k], v)

        # Reserved keys are present and the user keys round-trip.
        assert loaded_meta["_format_version"] == ARTIFACT_FORMAT_VERSION
        assert loaded_meta["_schema"] == "tensors_with_meta"
        for k, v in meta.items():
            assert loaded_meta[k] == v

    def test_empty_tensors_dict_round_trips(self, tmp_path: object) -> None:
        # safetensors will still write a valid (empty-payload) file.
        st_path, _ = save_tensors_with_meta(
            {}, {"note": "no tensors"}, str(tmp_path), "empty"
        )
        assert os.path.exists(st_path)
        tensors, meta = load_tensors_with_meta(str(tmp_path), "empty")
        assert tensors == {}
        assert meta["note"] == "no tensors"


class TestSaveModule:
    def test_round_trip_preserves_state_dict_and_extras(self, tmp_path: object) -> None:
        torch.manual_seed(0)
        module = torch.nn.Linear(4, 3)

        extra_meta = {"config": {"in": 4, "out": 3, "bias": True}}
        extra_tensors = {"preprocess_mean": torch.randn(4)}

        st_path, meta_path = save_module(
            module,
            str(tmp_path),
            "linear",
            extra_meta=extra_meta,
            extra_tensors=extra_tensors,
        )
        assert os.path.exists(st_path) and os.path.exists(meta_path)

        def factory(meta: dict[str, object]) -> torch.nn.Module:
            cfg = meta["config"]
            assert isinstance(cfg, dict)
            return torch.nn.Linear(cfg["in"], cfg["out"], bias=cfg["bias"])

        loaded, meta, loaded_extras = load_module(factory, str(tmp_path), "linear")

        assert isinstance(loaded, torch.nn.Linear)
        # Forward pass parity on the same input.
        x = torch.randn(2, 4)
        assert torch.allclose(loaded(x), module(x))

        assert meta["_class"].endswith(".Linear")
        assert meta["_schema"] == "module"
        assert "preprocess_mean" in loaded_extras
        assert torch.equal(
            loaded_extras["preprocess_mean"], extra_tensors["preprocess_mean"]
        )

    def test_extra_meta_cannot_overwrite_reserved_keys(self, tmp_path: object) -> None:
        module = torch.nn.Linear(2, 2)
        with pytest.raises(ValueError, match="reserved key"):
            save_module(
                module,
                str(tmp_path),
                "bad",
                extra_meta={"_schema": "hijack"},
            )


class TestVersionRejection:
    def test_load_rejects_future_format_version(self, tmp_path: object) -> None:
        # Write a pair manually with a bumped version.
        save_tensors_with_meta(
            {"x": torch.zeros(2)}, {"k": "v"}, str(tmp_path), "future"
        )
        meta_path = os.path.join(str(tmp_path), "future.meta.json")
        with open(meta_path) as f:
            meta = json.load(f)
        meta["_format_version"] = ARTIFACT_FORMAT_VERSION + 1
        with open(meta_path, "w") as f:
            json.dump(meta, f)

        with pytest.raises(ValueError, match="format version"):
            load_tensors_with_meta(str(tmp_path), "future")

    def test_load_rejects_missing_version(self, tmp_path: object) -> None:
        save_tensors_with_meta(
            {"x": torch.zeros(2)}, {"k": "v"}, str(tmp_path), "noversion"
        )
        meta_path = os.path.join(str(tmp_path), "noversion.meta.json")
        with open(meta_path) as f:
            meta = json.load(f)
        del meta["_format_version"]
        with open(meta_path, "w") as f:
            json.dump(meta, f)

        with pytest.raises(ValueError, match="missing"):
            load_tensors_with_meta(str(tmp_path), "noversion")


def test_format_version_constant_exposed() -> None:
    # Sanity check that the constant lives where the policy doc says it does.
    assert isinstance(artifacts.ARTIFACT_FORMAT_VERSION, int)
    assert artifacts.ARTIFACT_FORMAT_VERSION >= 1


def test_save_tensors_with_meta_rejects_reserved_meta_keys(tmp_path: object) -> None:
    with pytest.raises(ValueError, match="reserved key"):
        save_tensors_with_meta(
            {"x": torch.zeros(2)},
            {"_format_version": 999},
            str(tmp_path),
            "bad",
        )
    with pytest.raises(ValueError, match="reserved key"):
        save_tensors_with_meta(
            {"x": torch.zeros(2)},
            {"_schema": "hijack"},
            str(tmp_path),
            "bad2",
        )


def test_load_module_rejects_factory_class_mismatch(tmp_path: object) -> None:
    # Save a Linear, then try to load via a factory that returns Conv1d.
    module = torch.nn.Linear(4, 3)
    save_module(module, str(tmp_path), "linear")

    def wrong_factory(meta: dict[str, object]) -> torch.nn.Module:
        # Returns a different class than what was saved.
        return torch.nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1)

    with pytest.raises(ValueError, match=r"Factory produced .* declares"):
        load_module(wrong_factory, str(tmp_path), "linear")


def test_check_format_version_warns_on_older_version(
    tmp_path: object,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Write a normal pair, then rewrite meta with an older version.
    save_tensors_with_meta({"x": torch.zeros(2)}, {"k": "v"}, str(tmp_path), "old")
    meta_path = os.path.join(str(tmp_path), "old.meta.json")
    with open(meta_path) as f:
        meta = json.load(f)
    older = ARTIFACT_FORMAT_VERSION - 1
    meta["_format_version"] = older
    with open(meta_path, "w") as f:
        json.dump(meta, f)

    with caplog.at_level(logging.WARNING, logger="causalab.io.artifacts"):
        tensors, loaded_meta = load_tensors_with_meta(str(tmp_path), "old")

    assert torch.equal(tensors["x"], torch.zeros(2))
    assert loaded_meta["_format_version"] == older
    assert any("older artifact format" in rec.getMessage() for rec in caplog.records), (
        f"Expected an 'older artifact format' warning; got: {[r.getMessage() for r in caplog.records]}"
    )


def test_check_format_version_rejects_newer_version(tmp_path: object) -> None:
    save_tensors_with_meta({"x": torch.zeros(2)}, {"k": "v"}, str(tmp_path), "newer")
    meta_path = os.path.join(str(tmp_path), "newer.meta.json")
    with open(meta_path) as f:
        meta = json.load(f)
    meta["_format_version"] = ARTIFACT_FORMAT_VERSION + 1
    with open(meta_path, "w") as f:
        json.dump(meta, f)

    with pytest.raises(ValueError, match="format version"):
        load_tensors_with_meta(str(tmp_path), "newer")


class TestSaveInterventionResults:
    """The stored-artifact schema at the io boundary: producers
    hand ``save_intervention_results`` the legacy one-synthetic-batch
    ``raw_results`` view (``GenerationResult.to_raw_results()``); the on-disk
    schema is independent of the run's internal batch split (the flat result
    erased batch boundaries before io ever sees them, the
    ``batch_size < n_examples`` case) and unchanged vs legacy runs with
    ``batch_size >= n_examples``. Legacy multi-batch runs stored one inner
    ``"string"`` list per batch (and single-example batches a bare str);
    ``to_raw_results()`` always emits the one-batch nesting instead."""

    def _results_by_key(self, n: int = 3, width: int = 2):
        from causalab.neural.pipeline import GenerationResult

        result = GenerationResult(
            sequences=torch.arange(n * width, dtype=torch.long).reshape(n, width),
            strings=[f"out_{i}" for i in range(n)],
        )
        return {
            ("layer0", "pos0"): {
                "avg_score": 0.5,
                "scores_by_variable": {},
                "raw_results": result.to_raw_results(),
            }
        }
