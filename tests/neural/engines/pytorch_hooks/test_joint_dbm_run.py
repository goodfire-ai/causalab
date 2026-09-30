"""A generated joint-DBM document runs (``scripts/joint_dbm.py``).

The heads family on the tiny MoE fixture over two layers — layer 2 (Gated
DeltaNet, value heads at ``delta_premix``) and layer 3 (full attention, query
heads at ``attention_premix``) — with two penalties and one seed, a tiny step
budget, through ``run_protocol``: one bundle per gate keyed by the shared
weight and the seed, and the metric tables carrying exactly those two
coordinate columns. Then the generated apply document, validated by the script
against the fit's real bundles, replays one cell on the fixture's test table.

The generator is run as a subprocess, the way a user runs it, with
``--register-from-hf`` because the fixture is not a registry entry; the test
process registers the same key before loading.

The whole module waits on three pieces this tree may not carry yet
(``tests._helpers.harvest_deps``): the registry's per-layer ``layer_types``
the heads family reads, the grouped gate and the named ``train.objective``.
Each is a predicate, so the module runs on its own
once they land.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from safetensors.torch import load_file

from causalab.cli import register_model_key
from causalab.neural.shared.engine_router import route
from causalab.protocol import run_protocol
from causalab.protocol.pipeline import compile_protocol
from causalab.io.sources import load_text
from causalab.protocol.lowering import point_count
from causalab.io.env import FileDatasets, read_safetensors_metadata
from causalab.io.tables import read_table

from tests._helpers.harvest_deps import (
    needs_grouped_gate,
    needs_layer_types,
    needs_named_objective,
)
from tests.neural.engines.pytorch_hooks.conftest import TINY_QWEN35_MOE
from tests.protocol._env import FIXTURES, build_env

pytestmark = [
    pytest.mark.smoke,
    needs_layer_types,
    needs_grouped_gate,
    needs_named_objective,
]

REPO = Path(__file__).resolve().parents[4]
SCRIPT = REPO / "scripts/joint_dbm.py"

PENALTIES = [0.01, 0.1]
PENALTY_AXIS = "train.objective.sparsity.weight"
SEED_AXIS = "train.seed"
#: the fixture's two head-major families: layer 2 is DeltaNet, layer 3 full attention
GATES = {"gate_L2_heads": "delta_premix", "gate_L3_heads": "attention_premix"}
HEADS = 8
#: The fixture tables the fit trains on and the apply scores, by split.
TRAIN = "weekdays/data#train"
TEST = "weekdays/data#test"


def fixture_rows(ref: str) -> int:
    """The rows the fixture data root serves for ``ref``, through the loader's
    own resolver, so a table length is never hard-coded here."""
    return len(FileDatasets(root=FIXTURES / "data").rows(ref))


COMMON = [
    "--family",
    "heads",
    "--model",
    TINY_QWEN35_MOE,
    "--dtype",
    "fp32",
    "--register-from-hf",
    "--variable",
    "weekday",
    "--dataset",
    TRAIN,
    "--validation",
    TEST,
    "--layers",
    "2:4",
    "--penalties",
    *map(str, PENALTIES),
    "--seeds",
    "0",
    "--epochs",
    "1",
    "--save-rank",
    "--batch-pairs",
    "2",
    "--data-root",
    str(FIXTURES / "data"),
]


def generate(out: Path, *args: str) -> str:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), *COMMON, *args, "--out", str(out)],
        capture_output=True,
        text=True,
        cwd=REPO,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


@pytest.fixture(scope="module")
def fit(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("joint_dbm")
    line = generate(root / "fit.json")
    assert "(2 gates, 2 points, digest " in line
    env = build_env(root)
    document = load_text(root / "fit.json")
    register_model_key(document)
    loaded = compile_protocol(root / "fit.json", env=env)
    result = run_protocol(loaded, env, route("auto", device="cpu"), root / "fit")
    return root, loaded, result


class TestGeneratedFit:
    def test_the_document_sweeps_the_shared_weight_and_the_seed(self, fit) -> None:
        _, loaded, _ = fit
        assert [axis.id for axis in loaded.axes] == [PENALTY_AXIS, SEED_AXIS]
        assert point_count(loaded.axes) == 2
        sites = loaded.canonical["method"]["sites"]
        assert {g: sites[g.removeprefix("gate_")]["component"] for g in GATES} == GATES

    @pytest.mark.parametrize("gate", sorted(GATES))
    def test_one_bundle_per_gate_with_an_entry_per_cell(self, fit, gate: str) -> None:
        _, _, result = fit
        bundle = Path(result.files[f"{gate}.safetensors"])
        tensors = load_file(str(bundle))
        assert set(tensors) == {
            f"theta[objective.sparsity.weight={w},seed=0]" for w in PENALTIES
        }
        assert all(t.shape == (HEADS,) for t in tensors.values())
        header = read_safetensors_metadata(bundle)
        assert header is not None and header["group"] == "head"
        entries = json.loads(header["entries"])
        assert {tuple(sorted(e["coords"].items())) for e in entries.values()} == {
            (("objective.sparsity.weight", w), ("seed", 0)) for w in PENALTIES
        }

    @pytest.mark.parametrize("table", ["iia.json", "ce.json", "logit_diff.json"])
    def test_the_tables_carry_the_two_coordinate_columns(self, fit, table: str) -> None:
        _, _, result = fit
        rows = read_table(Path(result.files[table]))
        # the train table's rows, once per point
        assert len(rows) == fixture_rows(TRAIN) * len(PENALTIES)
        assert all(
            set(row)
            == {
                "example_id",
                "metric",
                "value",
                PENALTY_AXIS,
                SEED_AXIS,
                "unit",
                "estimand_version",
                "eligible",  # the eligibility record (§2.10)
            }
            for row in rows
        )
        assert sorted({row[PENALTY_AXIS] for row in rows}) == PENALTIES

    def test_the_diagnostics_report_both_gates_per_point(self, fit) -> None:
        _, _, result = fit
        records = json.loads(Path(result.files["fit_diagnostics.json"]).read_text())
        assert len(records) == len(PENALTIES)
        for record in records:
            assert set(record["featurizers"]) == set(GATES)
            assert all(g["groups"] == HEADS for g in record["featurizers"].values())


@pytest.fixture(scope="module")
def applied(fit):
    """The apply document for the (0.1, seed 0) cell, validated by the script
    against the fit's bundles, then run on the fixture's test table."""
    root, _, _ = fit
    line = generate(
        root / "apply.json",
        "--apply",
        "--fit-dir",
        "fit",
        "--penalty",
        "0.1",
        "--seed",
        "0",
        "--confirmation",
        TEST,
        "--artifacts-root",
        str(root),
    )
    assert "(2 gates, 1 point, digest " in line
    env = build_env(root)
    loaded = compile_protocol(root / "apply.json", env=env)
    result = run_protocol(loaded, env, route("auto", device="cpu"), root / "apply")
    return loaded, result


class TestGeneratedApply:
    def test_every_gate_is_loaded_from_the_fit_at_the_selected_cell(
        self, applied
    ) -> None:
        loaded, _ = applied
        assert loaded.document.train is None
        for gate in GATES:
            spec = loaded.tree["method"]["featurizers"][gate]
            assert spec["file_path"] == f"fit/{gate}.safetensors"
            assert spec["entry"] == {"objective.sparsity.weight": 0.1, "seed": 0}

    def test_it_scores_the_confirmation_table_without_coordinates(
        self, applied
    ) -> None:
        _, result = applied
        assert set(result.files) == {
            "iia.json",
            "ce.json",
            "logit_diff.json",
            "rank.json",
        }
        rows = read_table(Path(result.files["iia.json"]))
        assert len(rows) == fixture_rows(TEST)  # the confirmation table, one point
        assert all(
            set(row)
            == {
                "example_id",
                "metric",
                "value",
                "unit",
                "estimand_version",
                "eligible",
            }
            for row in rows
        )
        assert all(row["value"] in (0.0, 1.0) for row in rows)


class TestGeneratedApplyAll:
    def test_every_saved_mask_is_replayed_at_its_own_cell(self, fit):
        root, _, fit_result = fit
        generate(
            root / "apply_all.json",
            "--apply-all",
            "--fit-dir",
            "fit",
            "--confirmation",
            TEST,
            "--artifacts-root",
            str(root),
        )
        env = build_env(root)
        loaded = compile_protocol(root / "apply_all.json", env=env)
        assert point_count(loaded.axes) == len(PENALTIES)
        result = run_protocol(
            loaded, env, route("auto", device="cpu"), root / "apply_all"
        )
        rows = read_table(Path(result.files["iia.json"]))
        assert len(rows) == fixture_rows(TEST) * len(PENALTIES)
        assert {row["axes.fit_cell"] for row in rows} == set(range(len(PENALTIES)))
        fit_ranks = json.loads(Path(fit_result.files["rank.json"]).read_text())
        apply_ranks = json.loads(Path(result.files["rank.json"]).read_text())
        assert len(fit_ranks) == len(apply_ranks) == len(PENALTIES) * len(GATES) * HEADS
        expected = {
            (
                PENALTIES.index(row["coords"][PENALTY_AXIS]),
                row["featurizer"],
                row["unit"],
            ): row["hard"]
            for row in fit_ranks
        }
        actual = {
            (row["coords"]["axes.fit_cell"], row["featurizer"], row["unit"]): row[
                "hard"
            ]
            for row in apply_ranks
        }
        assert actual == expected
        for metric in ("iia", "logit_diff"):
            metric_rows = read_table(Path(result.files[f"{metric}.json"]))
            assert len(metric_rows) == fixture_rows(TEST) * len(PENALTIES)


class TestGeneratedAllNeurons:
    def test_independent_tokens_fit_and_replay_complete_neuron_outputs(self, tmp_path):
        generate(
            tmp_path / "fit.json",
            "--family",
            "all-neurons",
            "--positions",
            "0",
            "1",
            "--epochs",
            "1",
            "--penalties",
            "0.1",
        )
        env = build_env(tmp_path)
        register_model_key(load_text(tmp_path / "fit.json"))
        loaded = compile_protocol(tmp_path / "fit.json", env=env)
        assert len(loaded.document.featurizers) == 12
        fit = run_protocol(loaded, env, route("auto", device="cpu"), tmp_path / "fit")
        generate(
            tmp_path / "apply.json",
            "--family",
            "all-neurons",
            "--positions",
            "0",
            "1",
            "--penalties",
            "0.1",
            "--apply-all",
            "--fit-dir",
            "fit",
            "--confirmation",
            TEST,
            "--artifacts-root",
            str(tmp_path),
        )
        applied = compile_protocol(tmp_path / "apply.json", env=env)
        result = run_protocol(
            applied, env, route("auto", device="cpu"), tmp_path / "apply"
        )
        fit_ranks = json.loads(Path(fit.files["rank.json"]).read_text())
        apply_ranks = json.loads(Path(result.files["rank.json"]).read_text())
        expected = {(r["featurizer"], r["unit"]): r["hard"] for r in fit_ranks}
        actual = {(r["featurizer"], r["unit"]): r["hard"] for r in apply_ranks}
        assert actual == expected
        assert {r["featurizer"].rsplit("_P", 1)[1] for r in apply_ranks} == {"0", "1"}
        for metric in ("iia", "logit_diff"):
            rows = read_table(Path(result.files[f"{metric}.json"]))
            assert len(rows) == fixture_rows(TEST)
        from causalab.analysis.export_dbm import export

        manifest = tmp_path / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "experiments": [
                        {
                            "id": "neurons",
                            "title": "Complete neurons",
                            "evaluations": [
                                {
                                    "document": "apply.json",
                                    "run_dir": "apply",
                                    "artifacts_root": ".",
                                    "data_root": str(FIXTURES / "data"),
                                }
                            ],
                        }
                    ]
                }
            )
        )
        exported = export(manifest)
        experiment = exported["experiments"][0]
        assert {gate["position"] for gate in experiment["gates"]} == {0, 1}
        assert len(experiment["gates"]) == 12
        assert experiment["position_mode"] == "independent"
        assert experiment["points"][0]["selected_count"] == sum(actual.values())
        (tmp_path / "output_dbm.json").write_text(json.dumps(exported))
