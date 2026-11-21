"""Generate runnable CausaLab workflows for a cyclic concept."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE / "workflows" / "scripts" / "manifold_fig4"


def _write(path: Path, document: dict[str, Any]) -> None:
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def _ref(step: str, file: str) -> dict[str, str]:
    return {"step": step, "file": file}


def _external(path: Path, directory: Path) -> dict[str, str]:
    return {"path": Path(os.path.relpath(path, directory)).as_posix()}


def _protocol(name: str, model: dict, dataset: str) -> dict:
    source = HERE / "protocols" / f"manifold_fig4_{name}.json"
    doc = json.loads(source.read_text(encoding="utf-8"))
    doc["header"]["description"] = f"Cyclic manifold workflow: {name}."
    doc["model"] = model
    doc["data"]["base"]["dataset"] = dataset
    return doc


def _saves(groups: list[dict], models: dict[str, str]) -> list[dict]:
    return [
        {
            "read": read,
            "model": model,
            "aggregation": {"kind": "class_probs", "groups": group},
            "file_path": f"{model}_forms{i}.json",
        }
        for model, read in models.items()
        for i, group in enumerate(groups)
    ]


def build(config: dict, data_root: Path, directory: Path, runs: Path) -> None:
    """Write fitting, reusable steering, and one-command comparison documents."""
    classes = config["classes"]
    if (
        len(classes) < 3
        or len(set(classes)) != len(classes)
        or any(
            not re.fullmatch(r"[A-Za-z0-9]+", c)
            or c
            in {
                "other",
                "day",
                "n",
                "theta",
                "method",
                "pair",
                "step",
                "t",
                "example",
                "bc_to_manifold",
            }
            for c in classes
        )
    ):
        raise ValueError(
            "classes must be at least three distinct, non-reserved alphanumeric labels"
        )
    groups = config["answer_groups"]
    if (
        not groups
        or set(groups[0]) != set(classes)
        or any(not group or set(group) - set(classes) for group in groups)
    ):
        raise ValueError(
            "the first answer group must cover every class; later ones may cover subsets"
        )
    forms = [form for group in groups for tokens in group.values() for form in tokens]
    if not all(group.values() and all(group.values()) for group in groups) or len(
        set(forms)
    ) != len(forms):
        raise ValueError("answer forms must be nonempty and occur in exactly one group")
    steps = config.get("waypoints", 50)
    k = config["pca_components"]
    if steps < 2 or k < 2:
        raise ValueError("need at least two waypoints and two PCA components")
    all_pairs = [f"{a}_{b}" for a, b in itertools.combinations(classes, 2)]
    pairs = config.get("pairs", all_pairs)
    if not pairs or len(set(pairs)) != len(pairs) or set(pairs) - set(all_pairs):
        raise ValueError("pairs must be distinct and in the order declared by classes")
    if config.get("behavior_samples", 1050) < 2:
        raise ValueError("need at least two behavior samples")

    fit_dataset = config["fit_dataset"]
    steering_dataset = config.get("steering_dataset", fit_dataset)
    fit_table = data_root / f"{fit_dataset}.json"
    steering_table = data_root / f"{steering_dataset}.json"
    label_field = config.get("label_field", "result")
    rows = json.loads(fit_table.read_text(encoding="utf-8"))
    if {str(row[label_field]) for row in rows} != set(classes) or len(rows) <= k:
        raise ValueError(
            "the fitting table must cover all classes and have more than k rows"
        )
    if not steering_table.is_file():
        raise ValueError(f"missing steering table {steering_table}")

    directory.mkdir(parents=True, exist_ok=True)
    scripts_dir = directory / "scripts"
    scripts_dir.mkdir(exist_ok=True)
    for name in (
        "splines",
        "manifold_path",
        "behavior_manifold",
        "energy",
        "import_fit",
    ):
        shutil.copyfile(SCRIPTS / f"{name}.py", scripts_dir / f"{name}.py")

    model = config["model"]
    baseline = _protocol("baseline", model, fit_dataset)
    harvest = _protocol("harvest", model, fit_dataset)
    steer = _protocol("steer", model, steering_dataset)
    baseline["method"]["save"] = _saves(groups, {"original": "logits"})
    for doc in (harvest, steer):
        doc["method"]["positions"]["slot"] = {"index": config.get("position", -1)}
        doc["method"]["sites"]["target"] = {
            "component": config.get("component", "block_output"),
            "layers": [config["layer"]],
        }

    projected = config.get("projected_chord", True)
    steer["method"]["featurizers"]["pca"].update(
        k=k, file_path="fit_inputs/basis.safetensors"
    )
    if projected:
        steer["method"]["writes"]["lin_chord"]["featurizer"] = "pca"
    steer["method"]["save"] = _saves(
        groups, {"linear": "logits_linear", "manifold": "logits_manifold"}
    )
    steer["axes"]["path"]["rows"] = [
        {"step": step, "sel": {"step": step}} for step in range(steps)
    ]
    steer["axes"]["pair"]["rows"] = [
        {
            "pair": pair,
            **{
                key: f"fit_inputs/{key}_{pair}.safetensors"
                for key in ("waypoints", "chord")
            },
        }
        for pair in pairs
    ]
    for name, doc in (("baseline", baseline), ("harvest", harvest), ("steer", steer)):
        _write(directory / f"{name}.json", doc)

    # Keep the table schemas in sync with the paper workflow. Only the
    # distribution columns and the authored pair outputs vary by experiment.
    source = HERE / "workflows" / "manifold_fig4.json"
    fit = json.loads(source.read_text(encoding="utf-8"))["steps"]
    energy = fit.pop("energy")
    fit.pop("steer")
    for step, name in (("baseline", "baseline"), ("harvest_l28", "harvest")):
        fit[step]["document"] = f"{name}.json"
    fit["pca48"]["inputs"]["k"] = k
    common = {
        "classes": classes,
        "label_field": label_field,
        "table": _external(fit_table, directory),
    }
    path = fit["manifold_path"]
    path["script"]["path"] = "scripts/manifold_path.py"
    path["inputs"].update(common, steps=steps, projected_chord=projected)
    fit_contract = {
        "model": model,
        "site": harvest["method"]["sites"]["target"],
        "position": harvest["method"]["positions"]["slot"],
        "classes": classes,
        "label_field": label_field,
        "fit_dataset": fit_dataset,
        "fit_table_sha256": hashlib.sha256(fit_table.read_bytes()).hexdigest(),
        "answer_groups": groups,
        "pca_components": k,
        "projected_chord": projected,
        "waypoints": steps,
        "behavior_samples": config.get("behavior_samples", 1050),
    }
    path["inputs"]["contract"] = fit_contract
    path_outputs = {
        f"{kind}_{pair}": f"{kind}_{pair}.safetensors"
        for pair in all_pairs
        for kind in ("waypoints", "chord")
    }
    path["outputs"] = {
        **{
            slot: path["outputs"][slot]
            for slot in ("centroids", "points", "spline", "path")
        },
        **path_outputs,
        "contract": "contract.json",
        "lengths": {
            "file": "lengths.json",
            "columns": {
                "pair": "string",
                "manifold_length": "float64",
                "linear_length": "float64",
            },
        },
    }
    spellings = [f"forms{i}" for i in range(len(groups))]
    behavior = fit["behavior"]
    behavior["script"]["path"] = "scripts/behavior_manifold.py"
    behavior["inputs"] = {
        **common,
        "samples": config.get("behavior_samples", 1050),
        "spellings": spellings,
        **{s: _ref("baseline", f"original_{s}.json") for s in spellings},
    }
    bins = [*classes, "other"]
    for output in behavior["outputs"].values():
        output["columns"] = {
            **{
                field: dtype
                for field, dtype in output["columns"].items()
                if dtype != "float64" or field == "theta"
            },
            **dict.fromkeys(bins, "float64"),
        }

    product_refs = {
        "basis": _ref("pca48", "basis.safetensors"),
        "behavior_centroids": _ref("behavior", "behavior_centroids.json"),
        "behavior_spline": _ref("behavior", "behavior_spline.json"),
        "lengths": _ref("manifold_path", "lengths.json"),
        **{
            f"{kind}_{pair}": _ref("manifold_path", path_outputs[f"{kind}_{pair}"])
            for pair in pairs
            for kind in ("waypoints", "chord")
        },
    }
    products = {slot: ref["file"] for slot, ref in product_refs.items()}
    validation = {"contract": _ref("manifold_path", "contract.json")}
    energy["script"]["path"] = "scripts/energy.py"
    energy["inputs"] = {
        "classes": classes,
        "spellings": spellings,
        "behavior_centroids": _ref("fit_inputs", products["behavior_centroids"]),
        "behavior_spline": _ref("fit_inputs", products["behavior_spline"]),
        "activation_lengths": _ref("fit_inputs", products["lengths"]),
        **{
            f"{method}_{s}": _ref("steer", f"{method}_{s}.json")
            for method in ("manifold", "linear")
            for s in spellings
        },
    }
    trajectory = energy["outputs"]["trajectory"]["columns"]
    energy["outputs"]["trajectory"]["columns"] = {
        **{
            field: trajectory[field]
            for field in ("method", "pair", "step", "t", "n", "bc_to_manifold")
        },
        **dict.fromkeys([*bins, *(f"band_{c}" for c in bins)], "float64"),
    }
    energy["outputs"]["energy"]["columns"].update(
        coherence="float64",
        worst_coherence="float64",
    )
    # A table with no declared columns permits null for undefined correlations.
    energy["outputs"]["geometry"] = "geometry.json"
    steer_steps = {
        "fit_inputs": {
            "type": "script",
            "script": {"path": "scripts/import_fit.py"},
            "inputs": {**product_refs, **validation, "expected": fit_contract},
            "outputs": products,
        },
        "steer": {"type": "intervention_protocol", "document": "steer.json"},
        "energy": energy,
    }
    _workflow(directory, "manifold_fit", fit)
    _workflow(directory, "manifold_comparison", {**fit, **steer_steps})
    fit_root = runs / "manifold_fit"
    steer_steps["fit_inputs"]["inputs"] = {
        **{
            slot: _external(fit_root / ref["step"] / ref["file"], directory)
            for slot, ref in (product_refs | validation).items()
        },
        "expected": fit_contract,
    }
    _workflow(directory, "path_steering", steer_steps)


def _workflow(directory: Path, name: str, steps: dict) -> None:
    _write(
        directory / f"{name}.json",
        {
            "version": "1",
            "description": "Cyclic manifold steering. Saved tables are available for post-hoc plotting.",
            "output_dir": name,
            "steps": steps,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--documents", required=True, type=Path)
    parser.add_argument(
        "--runs", required=True, type=Path, help="The workflow invocation's --out root."
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    build(
        config, args.data_root.resolve(), args.documents.resolve(), args.runs.resolve()
    )


if __name__ == "__main__":
    main()
