"""Run both workflow forms on a random model with non-weekday classes."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from causalab.cli import main
from tests.demos.papers._scripts import PAPERS

pytestmark = pytest.mark.smoke

CONFIG = {
    "model": {
        "key": "hf-internal-testing/tiny-random-LlamaForCausalLM",
        "revision": "9fb191250dd56d0ba7ec9785a025ed29c03d5998",
        "dtype": "fp32",
    },
    "layer": 0,
    "classes": list("ABCD"),
    "label_field": "answer",
    "pca_components": 3,
    "fit_dataset": "fit",
    "steering_dataset": "heldout",
    "waypoints": 3,
    "behavior_samples": 32,
    "pairs": ["A_B", "A_C", "B_D"],
    "answer_groups": [{label: [label] for label in "ABCD"}],
}


def test_split_and_combined_manifold_workflows(tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location(
        "_build_manifold_workflows", PAPERS / "build_manifold_workflows.py"
    )
    assert spec is not None and spec.loader is not None
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    data = tmp_path / "data"
    data.mkdir()
    rows = [
        {
            "input": f"Circle location {place}. Option {label}. Answer:",
            "answer": label,
            "split": "all",
        }
        for label, place in zip("ABCD", ("north", "east", "south", "west"))
    ]
    (data / "fit.json").write_text(json.dumps(rows), encoding="utf-8")
    (data / "heldout.json").write_text(
        json.dumps([{"input": "Which circle location? Answer:", "split": "all"}]),
        encoding="utf-8",
    )
    docs, runs = tmp_path / "documents", tmp_path / "runs"
    builder.build(CONFIG, data, docs, runs)

    def run(name: str, documents: Path = docs, output: Path = runs) -> None:
        assert (
            main(
                [
                    "run",
                    str(documents / f"{name}.json"),
                    "--data-root",
                    str(data),
                    "--out",
                    str(output),
                    "--device",
                    "cpu",
                    "--engine",
                    "pytorch_hooks",
                ]
            )
            == 0
        )

    run("manifold_fit")
    run("path_steering")
    run("manifold_comparison")
    split = runs / "path_steering" / "energy"
    combined = runs / "manifold_comparison" / "energy"
    for file in ("energy.json", "geometry.json", "summary.json", "trajectory.json"):
        assert (split / file).read_bytes() == (combined / file).read_bytes()
    energy = json.loads((split / "energy.json").read_text())
    assert len(energy) == 2 * len(CONFIG["pairs"])
    assert {row["n_prompts"] for row in energy} == {1}

    single_docs, single_runs = tmp_path / "single_docs", tmp_path / "single_runs"
    builder.build({**CONFIG, "pairs": ["A_B"]}, data, single_docs, runs)
    run("path_steering", single_docs, single_runs)
    summary = json.loads(
        (single_runs / "path_steering/energy/summary.json").read_text()
    )
    assert summary["n_pairs"] == 1
    assert summary["paired_t"] is summary["paired_p"] is None
