"""Check exported DBM data against evaluated masks and recorded provenance."""

from __future__ import annotations

import base64
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from causalab.analysis import export_dbm as exporter
from causalab.neural.shared.engine_router import route
from causalab.neural.shared.sweep import signed_steps
from causalab.protocol import run_protocol
from causalab.protocol.schema.explicit import canonical_model
from causalab.protocol.pipeline import compile_protocol

from tests.neural.engines.pytorch_hooks.test_joint_dbm_run import (
    GATES,
    HEADS,
    PENALTIES,
    TEST,
    fit,
    generate,
)
from tests.protocol._env import FIXTURES, build_env

__all__ = ["fit"]


def read(path: Path):
    return json.loads(path.read_text())


def write(path: Path, value) -> None:
    path.write_text(json.dumps(value) + "\n")


def rows_at(rows: list[dict], coords: dict) -> list[dict]:
    """The metric rows a receipt point owns: those carrying its coordinates."""
    key = exporter.coordinate_key(coords)
    return [
        row
        for row in rows
        if exporter.coordinate_key({axis: row.get(axis) for axis in coords}) == key
    ]


def decode_mask(record: dict, count: int) -> np.ndarray:
    if record["encoding"] == "bitset":
        raw = np.frombuffer(base64.b64decode(record["data"]), dtype=np.uint8)
        return np.unpackbits(raw, bitorder="little")[: record["unit_count"]].astype(
            bool
        )
    result = np.zeros(count, dtype=bool)
    for start, stop in record["ranges"]:
        result[start:stop] = True
    return result


@pytest.mark.unit
class TestMaskEncoding:
    @pytest.mark.parametrize(
        "mask,encoding",
        [
            (np.zeros(0, dtype=bool), "ranges"),
            (np.zeros(257, dtype=bool), "ranges"),
            (np.ones(4099, dtype=bool), "ranges"),
            (np.arange(4099) % 2 == 0, "bitset"),
            ((np.arange(4099) > 100) & (np.arange(4099) < 104), "ranges"),
        ],
    )
    def test_membership_survives_both_encodings(self, mask, encoding):
        record = exporter.encoded_mask(mask)
        assert record["encoding"] == encoding
        np.testing.assert_array_equal(decode_mask(record, len(mask)), mask)

    def test_bitset_keeps_the_last_unit_beyond_a_full_byte(self):
        mask = np.arange(1027) % 2 == 0
        record = exporter.encoded_mask(mask)
        assert record["unit_count"] == 1027
        decoded = decode_mask(record, len(mask))
        assert len(decoded) == 1027 and decoded[-1]
        np.testing.assert_array_equal(decoded, mask)


@pytest.mark.unit
class TestMetricSummary:
    @pytest.mark.parametrize(
        "metric,values", [("iia", [0.0, 1.0]), ("logit_diff", [-2.0, 4.0])]
    )
    def test_only_eligible_examples_contribute(self, metric, values):
        rows = [
            {"example_id": str(i), "metric": metric, "value": value, "eligible": True}
            for i, value in enumerate(values)
        ]
        rows.append(
            {
                "example_id": "missing",
                "metric": metric,
                "value": None,
                "eligible": False,
            }
        )
        summary = exporter.metric_summary(rows, metric)
        assert summary["value"] == sum(values) / len(values)
        assert summary["n"] == 2 and summary["total"] == 3
        assert summary["standard_error"] == pytest.approx(
            abs(values[1] - values[0]) / 2
        )

    def test_no_eligible_examples_has_no_numeric_score(self):
        rows = [{"example_id": "0", "metric": "iia", "value": None, "eligible": False}]
        summary = exporter.metric_summary(rows, "iia")
        assert summary["value"] is None and summary["n"] == 0
        assert summary["reason"]

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), None, 0.5])
    def test_invalid_eligible_iia_values_are_refused(self, value):
        with pytest.raises(ValueError):
            exporter.metric_summary(
                [{"example_id": "0", "metric": "iia", "value": value}], "iia"
            )

    def test_duplicate_examples_and_wrong_metrics_are_refused(self):
        row = {"example_id": "0", "metric": "iia", "value": 1}
        with pytest.raises(ValueError, match="repeats"):
            exporter.metric_summary([row, row], "iia")
        with pytest.raises(ValueError, match="another metric"):
            exporter.metric_summary([row], "logit_diff")


@pytest.fixture(scope="module")
def evaluated(fit):
    root, _, _ = fit
    generate(
        root / "report_apply.json",
        "--apply-all",
        "--fit-dir",
        "fit",
        "--confirmation",
        TEST,
        "--artifacts-root",
        str(root),
    )
    env = build_env(root)
    loaded = compile_protocol(root / "report_apply.json", env=env)
    run_protocol(loaded, env, route("auto", device="cpu"), root / "report_apply")
    write(
        root / "manifest.json",
        {
            "experiments": [
                {
                    "id": "output-heads",
                    "title": "Output heads",
                    "evaluations": [
                        {
                            "document": "report_apply.json",
                            "run_dir": "report_apply",
                            "artifacts_root": ".",
                            "data_root": str(FIXTURES / "data"),
                        }
                    ],
                }
            ],
        },
    )
    return root


@pytest.fixture
def copied(evaluated, tmp_path):
    root = tmp_path / "evaluation"
    shutil.copytree(evaluated, root)
    return root


@pytest.mark.smoke
class TestEvaluatedExport:
    def test_fit_identity_survives_a_moved_run_tree(self, evaluated, copied):
        original = exporter.export(evaluated / "manifest.json")
        moved = exporter.export(copied / "manifest.json")
        assert [p["fit_id"] for p in original["experiments"][0]["points"]] == [
            p["fit_id"] for p in moved["experiments"][0]["points"]
        ]

        assert [p["artifact_id"] for p in original["experiments"][0]["points"]] == [
            p["artifact_id"] for p in moved["experiments"][0]["points"]
        ]

    def test_fresh_cli_requires_explicit_model_registration(self, evaluated, tmp_path):
        script = Path(__file__).resolve().parents[2] / "scripts/export_dbm.py"
        command = [
            sys.executable,
            str(script),
            "--manifest",
            str(evaluated / "manifest.json"),
            "--out",
            str(tmp_path / "export.json"),
        ]
        refused = subprocess.run(command, capture_output=True, text=True)
        assert refused.returncode == 1 and "registry" in refused.stderr
        accepted = subprocess.run(
            [*command, "--register-from-hf"], capture_output=True, text=True
        )
        assert accepted.returncode == 0, accepted.stderr
        assert len(read(tmp_path / "export.json")["experiments"][0]["points"]) == len(
            PENALTIES
        )

    def test_each_point_has_the_exact_evaluated_mask_and_both_metrics(self, evaluated):
        report = exporter.export(evaluated / "manifest.json")
        assert report["synthetic"] is False
        (experiment,) = report["experiments"]
        assert len(experiment["points"]) == len(PENALTIES)
        assert experiment["omitted_points"] == []
        assert {g["family"] for g in experiment["gates"]} == {
            "normal_attention",
            "gated_delta_net",
        }
        ranks = read(evaluated / "report_apply/rank.json")
        tables = {
            name: read(evaluated / f"report_apply/{name}.json")
            for name in ("iia", "logit_diff")
        }
        for point in experiment["points"]:
            count = 0
            for gate in GATES:
                expected = [
                    r["hard"]
                    for r in sorted(ranks, key=lambda r: r["unit"])
                    if r["point"] == point["id"] and r["featurizer"] == gate
                ]
                actual = decode_mask(point["masks"][gate], HEADS)
                np.testing.assert_array_equal(actual, expected)
                count += int(actual.sum())
                fit = point["provenance"]["fits"][gate]
                assert fit["file_path"].endswith(f"fit/{gate}.safetensors")
                assert fit["entry"]
            assert point["selected_count"] == count
            assert point["eligible_count"] == HEADS * len(GATES)
            assert point["sparsity"] == 1 - count / (HEADS * len(GATES))
            for name, rows in tables.items():
                matching = [
                    r for r in rows_at(rows, point["coords"]) if r.get("eligible", True)
                ]
                assert point["metrics"][name]["value"] == pytest.approx(
                    sum(r["value"] for r in matching) / len(matching)
                )
                assert point["metrics"][name]["n"] == len(matching)

    def test_unmeasured_cells_stay_outside_the_slider(self, copied):
        # the run directory holds only the saved tables, so the points are
        # the steps the compiler signs, as the exporter enumerates them
        env = build_env(copied)
        loaded = compile_protocol(copied / "report_apply.json", env=env)
        unmeasured = signed_steps(loaded, env)[-1]
        omitted_digest = unmeasured.digest
        for name in ("iia", "logit_diff"):
            path = copied / f"report_apply/{name}.json"
            rows = read(path)
            dropped = rows_at(rows, dict(unmeasured.coords))
            assert dropped
            write(path, [row for row in rows if row not in dropped])
        (experiment,) = exporter.export(copied / "manifest.json")["experiments"]
        assert len(experiment["points"]) == len(PENALTIES) - 1
        assert omitted_digest not in {point["id"] for point in experiment["points"]}
        assert {point["id"] for point in experiment["omitted_points"]} == {
            omitted_digest
        }

    def test_a_row_off_every_signed_step_is_refused(self, copied):
        # the run directory holds no receipt, so the pairing of a document
        # with its run directory is the manifest author's claim: a document
        # or a frozen fit edited after the run is not detected here. What is
        # checked is that every metric row sits on a step the document signs.
        path = copied / "report_apply/iia.json"
        rows = read(path)
        rows[0]["axes.fit_cell"] = 999
        write(path, rows)
        with pytest.raises(ValueError, match="Unrecorded point"):
            exporter.export(copied / "manifest.json")

    def test_partial_metric_examples_are_refused(self, copied):
        path = copied / "report_apply/logit_diff.json"
        rows = read(path)
        write(path, rows[1:])
        with pytest.raises(ValueError):
            exporter.export(copied / "manifest.json")

    def test_fitting_results_cannot_enter_the_report(self, copied):
        manifest = read(copied / "manifest.json")
        manifest["experiments"][0]["evaluations"][0]["document"] = "fit.json"
        write(copied / "manifest.json", manifest)
        with pytest.raises(ValueError, match="apply"):
            exporter.export(copied / "manifest.json")


@pytest.mark.unit
class TestFrozenMask:
    def test_threshold_is_strictly_above_zero(self, tmp_path):
        from safetensors.numpy import save_file

        theta = np.array(
            [-1.0, 0.0, np.nextafter(np.float32(0), np.float32(1)), 1.0],
            dtype=np.float32,
        )
        path = tmp_path / "gate.safetensors"
        save_file({"theta": theta}, path)
        mask, provenance = exporter.frozen_mask({"file_path": path.name}, {}, tmp_path)
        np.testing.assert_array_equal(mask, [False, False, True, True])
        assert provenance == {"file_path": str(path.resolve()), "entry": "theta"}


@pytest.mark.smoke
class TestPopulationAndComparison:
    def test_metric_coordinates_use_full_native_axis_ids(self, copied):
        document = read(copied / "report_apply.json")
        entries = [row["entry"] for row in document.pop("axes")["fit_cell"]["rows"]]
        for gate in document["method"]["featurizers"].values():
            gate["entry"] = {"sweep": entries}
        write(copied / "report_apply.json", document)
        env = build_env(copied)
        loaded = compile_protocol(copied / "report_apply.json", env=env)
        run_protocol(loaded, env, route("auto", device="cpu"), copied / "report_apply")
        points = exporter.export(copied / "manifest.json")["experiments"][0]["points"]
        assert len(points) == len(PENALTIES) ** 2
        assert all(axis.startswith("featurizers.") for axis in points[0]["coords"])
        path = copied / "report_apply/iia.json"
        rows = read(path)
        axis = next(iter(points[0]["coords"]))
        rows[0][axis] = "changed"
        write(path, rows)
        with pytest.raises(ValueError, match="Unrecorded point"):
            exporter.export(copied / "manifest.json")

    def test_logit_difference_uses_answer_changing_pairs(self, copied):
        from causalab.io.env import FileArtifacts, FileDatasets, ResolutionEnv

        data_root = copied / "data"
        shutil.copytree(FIXTURES / "data", data_root)
        data_path = data_root / "weekdays/data.json"
        pairs = read(data_path)
        unchanged = next(row for row in pairs if row["split"] == "test")
        unchanged["base_answer"] = unchanged["label"]
        write(data_path, pairs)
        env = ResolutionEnv(
            datasets=FileDatasets(root=data_root), artifacts=FileArtifacts(root=copied)
        )
        loaded = compile_protocol(copied / "report_apply.json", env=env)
        run_protocol(
            loaded, env, route("auto", device="cpu"), copied / "filtered_apply"
        )
        manifest = read(copied / "manifest.json")
        evaluation = manifest["experiments"][0]["evaluations"][0]
        evaluation.update({"data_root": "data", "run_dir": "filtered_apply"})
        write(copied / "manifest.json", manifest)
        points = exporter.export(copied / "manifest.json")["experiments"][0]["points"]
        rows = read(copied / "filtered_apply/logit_diff.json")
        for point in points:
            assert point["metrics"]["iia"]["n"] == 2
            assert point["metrics"]["logit_diff"]["n"] == 1
            expected = next(
                row["value"]
                for row in rows_at(rows, point["coords"])
                if row["example_id"] == "1"
            )
            assert point["metrics"]["logit_diff"]["value"] == expected
            assert (
                point["provenance"]["logit_difference_population"]
                == "answer-changing pairs"
            )

    @pytest.mark.parametrize(
        "position_kind", ["inline_readout", "named_readout", "named_gate"]
    )
    def test_one_curve_cannot_mix_positions(self, copied, position_kind):
        document = read(copied / "report_apply.json")
        if position_kind.startswith("named"):
            method = document["method"]
            if position_kind == "named_readout":
                method["reads"]["logits"]["pos"] = "position"
            else:
                for write_spec in method["writes"].values():
                    write_spec["pos"] = "position"
                    method["reads"][write_spec["do"]["swap"]]["pos"] = "position"
            document["method"] = {"positions": {"position": {"index": -1}}, **method}
            write(copied / "report_apply.json", document)
            env = build_env(copied)
            loaded = compile_protocol(copied / "report_apply.json", env=env)
            run_protocol(
                loaded, env, route("auto", device="cpu"), copied / "report_apply"
            )
            document["method"]["positions"]["position"] = {"index": 0}
        else:
            document["method"]["reads"]["logits"]["pos"] = 0
        write(copied / "other_apply.json", document)
        env = build_env(copied)
        loaded = compile_protocol(copied / "other_apply.json", env=env)
        run_protocol(loaded, env, route("auto", device="cpu"), copied / "other_apply")
        manifest = read(copied / "manifest.json")
        evaluations = manifest["experiments"][0]["evaluations"]
        evaluations.append(
            {**evaluations[0], "document": "other_apply.json", "run_dir": "other_apply"}
        )
        write(copied / "manifest.json", manifest)
        with pytest.raises(
            ValueError, match="readout differ|different component universes"
        ):
            exporter.export(copied / "manifest.json")


@pytest.mark.unit
class TestManifestContract:
    def test_model_contract_retains_the_evaluation_dtype(self):
        model = {"key": "gpt2", "revision": "main", "dtype": "fp32"}
        assert exporter.model_manifest(model)["dtype"] == "fp32"
        assert exporter.model_manifest({**model, "dtype": "bf16"})["dtype"] == "bf16"

    @pytest.mark.parametrize(
        "field,values",
        [
            ("dtype", ["fp32", "bf16"]),
            ("quantization", [None, {"scheme": "int8"}]),
        ],
    )
    def test_one_curve_cannot_mix_model_realizations(
        self, tmp_path, monkeypatch, field, values
    ):
        def evaluated(item, base, **kwargs):
            model = exporter.model_manifest(
                canonical_model(
                    {
                        "key": "gpt2",
                        "revision": "main",
                        "dtype": "fp32",
                        field: item[field],
                    }
                )
            )
            gates = [{"id": "gate", "component": "attention_head", "position": -1}]
            points = [
                {
                    "id": json.dumps(item[field]),
                    "provenance": {"comparison": {"dataset": "same"}},
                }
            ]
            return model, gates, points, []

        monkeypatch.setattr(exporter, "evaluation", evaluated)
        manifest = tmp_path / "manifest.json"
        write(
            manifest,
            {
                "experiments": [
                    {
                        "id": "heads",
                        "title": "Heads",
                        "evaluations": [{field: value} for value in values],
                    }
                ]
            },
        )
        with pytest.raises(ValueError, match="different model"):
            exporter.export(manifest)


@pytest.mark.unit
class TestMeasuredMaskContract:
    @pytest.fixture
    def method(self):
        aggregation = {
            "iia": {
                "kind": "match",
                "expected": "label",
            },
            "logit_diff": {
                "kind": "logit_diff",
                "a": "label",
                "b": "base_answer",
            },
        }
        return {
            "intervened_models": {
                "masked": {
                    "input": "base",
                    "reads": ["iia_read", "difference_read"],
                    "writes": ["first", "second"],
                },
                "original_counterfactual": {"input": "counterfactual", "reads": []},
            },
            "sites": {
                "answer": {"component": "lm_head"},
                "answer_alias": {"component": "lm_head"},
            },
            "reads": {
                "iia_read": {"site": "answer", "pos": {"index": -1}},
                "difference_read": {"site": "answer_alias", "pos": {"index": -1}},
            },
            "writes": {
                "first": {"featurizer": "gate_a"},
                "second": {"featurizer": "gate_b"},
            },
            "save": [
                {
                    "read": "iia_read",
                    "model": "masked",
                    "aggregation": aggregation["iia"],
                    "file_path": "iia.json",
                },
                {
                    "read": "difference_read",
                    "model": "masked",
                    "aggregation": aggregation["logit_diff"],
                    "file_path": "logit_diff.json",
                },
            ],
        }

    def test_equivalent_read_and_site_names_resolve_to_the_same_measurement(
        self, method
    ):
        exporter.validate_measurement(method, [{"id": "gate_a"}, {"id": "gate_b"}])

    def test_named_metric_positions_resolve_before_comparison(self, method):
        method["positions"] = {"last": {"index": -1}, "first": {"index": 0}}
        method["reads"]["difference_read"]["pos"] = "last"
        exporter.validate_measurement(method, [{"id": "gate_a"}, {"id": "gate_b"}])
        method["reads"]["difference_read"]["pos"] = "first"
        with pytest.raises(ValueError, match="same resolved read"):
            exporter.validate_measurement(method, [{"id": "gate_a"}, {"id": "gate_b"}])

    @pytest.mark.parametrize(
        "mutate",
        [
            lambda m: m["reads"]["difference_read"].__setitem__("pos", {"index": 0}),
            # the difference read measured on another model (§2.9)
            lambda m: (
                m["intervened_models"]["masked"]["reads"].remove("difference_read"),
                m["intervened_models"]["original_counterfactual"]["reads"].append(
                    "difference_read"
                ),
                m["save"][1].__setitem__("model", "original_counterfactual"),
            ),
        ],
        ids=["pos", "model"],
    )
    def test_different_metric_reads_are_refused(self, method, mutate):
        mutate(method)
        with pytest.raises(ValueError, match="same resolved read"):
            exporter.validate_measurement(method, [{"id": "gate_a"}, {"id": "gate_b"}])

    def test_original_model_scores_cannot_describe_a_saved_mask(self, method):
        # both reads measured on the un-intervened model: no mask is described
        method["intervened_models"]["masked"]["reads"] = []
        method["intervened_models"]["original_counterfactual"]["reads"] = [
            "iia_read",
            "difference_read",
        ]
        for entry in method["save"]:
            entry["model"] = "original_counterfactual"
        with pytest.raises(ValueError, match="intervened model"):
            exporter.validate_measurement(method, [{"id": "gate_a"}, {"id": "gate_b"}])

    def test_an_exported_gate_must_act_in_the_measured_model(self, method):
        method["intervened_models"]["masked"]["writes"] = ["first"]
        with pytest.raises(ValueError, match="Every exported gate"):
            exporter.validate_measurement(method, [{"id": "gate_a"}, {"id": "gate_b"}])

    def test_an_empty_gate_universe_is_refused(self, method):
        with pytest.raises(ValueError, match="at least one gate"):
            exporter.validate_measurement(method, [])


@pytest.mark.unit
class TestPositions:
    @pytest.mark.parametrize(
        "value,expected",
        [(0, 0), ({"index": -1}, -1), ({"all": True}, "all"), ("all", "all")],
    )
    def test_scalar_positions_have_portable_keys(self, value, expected):
        assert exporter.position(value, {}) == expected
        assert exporter.position("patch", {"positions": {"patch": value}}) == expected

    @pytest.mark.parametrize(
        "value", [{"indices": [0, 1]}, {"variable": "answer"}, True, "missing"]
    )
    def test_unsupported_positions_are_refused(self, value):
        with pytest.raises(ValueError, match="scalar token index"):
            exporter.position(value, {})

    def test_import_needs_only_stdlib(self):
        result = subprocess.run(
            [sys.executable, "-S", "-c", "import causalab.analysis.export_dbm"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr


@pytest.mark.unit
class TestDbmAggregations:
    """The two report tables are located by kind through the document's
    aggregations, not by the names ``iia`` / ``logit_diff``."""

    def _doc(self, *, iia_label: str = "iia", same_read: bool = True):
        from causalab.protocol.schema import parse_document
        from tests.protocol._docs import aggregation, base_doc, in_order, saved

        raw = base_doc()
        method = raw["method"]
        match = aggregation("match", expected="cf_answer")
        read, model = "logits", "patched"
        if not same_read:
            # the same address, read on the un-intervened network on base
            read, model = "logits_orig", "original_base"
            method["reads"][read] = {"site": "lm_head", "pos": -1}
            method["intervened_models"][model] = {"input": "base", "reads": [read]}
        method["save"].append(saved(read, model, f"{iia_label}.json", match))
        return parse_document(in_order(raw))

    def test_the_tables_are_found_by_kind_whatever_their_labels(self):
        found = exporter.dbm_aggregations(self._doc(iia_label="accuracy"))
        assert found["iia"].label == "accuracy"
        assert found["logit_diff"].label == "ld"
        assert (
            found["iia"].owner == "save[1]" and found["logit_diff"].owner == "save[0]"
        )
        assert exporter._save_index(found["iia"].owner) == 1

    def test_two_reads_are_refused(self):
        with pytest.raises(ValueError, match="same read on the same model"):
            exporter.dbm_aggregations(self._doc(same_read=False))

    def test_a_missing_kind_is_refused_naming_what_was_found(self):
        from causalab.protocol.schema import parse_document
        from tests.protocol._docs import base_doc, in_order

        with pytest.raises(ValueError, match="found 0 match and 1 logit_diff"):
            exporter.dbm_aggregations(parse_document(in_order(base_doc())))
