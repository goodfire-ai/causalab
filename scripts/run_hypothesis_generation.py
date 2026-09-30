#!/usr/bin/env python3
"""Run the CPU-only hypothesis distinguishability check.

The hypotheses directory must contain ``models.py`` and ``counterfactuals.py``.
See ``demos/hypothesis_testing/`` for their interfaces and exact-pair export.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from causalab.causal.model_comparison import distinguishability_report


def load_modules(hypotheses_dir: Path) -> tuple[ModuleType, ModuleType]:
    """Load the researcher-authored modules from ``hypotheses_dir``."""
    resolved = hypotheses_dir.resolve()
    missing = [
        name
        for name in ("models.py", "counterfactuals.py")
        if not (resolved / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"{resolved} is missing {', '.join(missing)}; start from the templates "
            "in demos/hypothesis_testing/ (models.py and counterfactuals.py)"
        )

    # counterfactuals.py may import models.py by its ordinary module name.
    # This command runs one hypotheses directory at a time, so canonical names
    # are clearer and safer than leaving modules from another directory cached.
    sys.modules.pop("models", None)
    sys.modules.pop("counterfactuals", None)
    sys.path.insert(0, str(resolved))
    try:
        models = importlib.import_module("models")
        counterfactuals = importlib.import_module("counterfactuals")
    finally:
        sys.path.remove(str(resolved))
    return models, counterfactuals


def build_report(
    hypotheses_dir: Path,
    *,
    n: int,
    random_n: int,
    seed: int,
    export_dir: Path | None = None,
) -> dict[str, Any]:
    models, counterfactuals = load_modules(hypotheses_dir)
    for name in ("MODELS", "DEFAULT_MODEL", "HYPOTHESES"):
        if not hasattr(models, name):
            raise AttributeError(f"models.py must define {name}")
    for name in ("make_datasets", "random_pairs"):
        if not hasattr(counterfactuals, name):
            raise AttributeError(f"counterfactuals.py must define {name}")

    model_registry = models.MODELS
    default_model_name = models.DEFAULT_MODEL
    if default_model_name not in model_registry:
        raise KeyError(f"DEFAULT_MODEL {default_model_name!r} is not present in MODELS")

    hypotheses = dict(models.HYPOTHESES)
    hypotheses.setdefault("null", (default_model_name, []))
    hypotheses.setdefault("all", (default_model_name, ["raw_output"]))
    for name, (model_name, _variables) in hypotheses.items():
        if model_name not in model_registry:
            raise KeyError(
                f"hypothesis {name!r} names model {model_name!r}, which is not present in MODELS"
            )

    targets = list(getattr(models, "TARGETS", []))
    if not targets:
        targets = [name for name in hypotheses if name not in ("null", "all")]
    unknown_targets = [name for name in targets if name not in hypotheses]
    if unknown_targets:
        raise KeyError(f"TARGETS contains unknown hypotheses: {unknown_targets}")

    generator_model = model_registry[default_model_name]
    datasets = counterfactuals.make_datasets(generator_model, n=n, seed=seed)
    random_pairs = counterfactuals.random_pairs(generator_model, random_n, seed=seed)
    core = distinguishability_report(
        model_registry, hypotheses, targets, datasets, random_pairs
    )
    if export_dir is not None:
        from causalab.analysis.hypothesis_artifacts import export_hypotheses

        export_hypotheses(
            model_registry,
            hypotheses,
            targets,
            datasets,
            counterfactuals.DATASET_ROLES,
            export_dir,
        )
    return {
        "n": n,
        "random_n": random_n,
        "seed": seed,
        "hypotheses": {
            name: {"model": model_name, "targets": list(variables)}
            for name, (model_name, variables) in hypotheses.items()
        },
        "targets": targets,
        "dataset_roles": getattr(counterfactuals, "DATASET_ROLES", {}),
        **core,
    }


def print_summary(report: dict[str, Any], threshold: float) -> None:
    for dataset_name, dataset in report["datasets"].items():
        role = report["dataset_roles"].get(dataset_name, {})
        role_text = (
            f" [{role.get('family', role.get('width', '?'))}/{role.get('split', '?')}]"
            if role
            else ""
        )
        print(f"\n=== {dataset_name}{role_text} (n={dataset['size']}) ===")
        for target, result in dataset["per_target"].items():
            vs_null = result["vs_null"]
            vs_all = result["vs_all"]

            def rate_text(value):
                return "unavailable" if value is None else f"{value:.2f}"

            print(
                f"  target {target!r}: vs_null={rate_text(vs_null)}  vs_all={rate_text(vs_all)}"
            )
            for alternative, rate in sorted(
                result["alternatives"].items(),
                key=lambda item: -1 if item[1] is None else item[1],
                reverse=True,
            ):
                if alternative in ("null", "all"):
                    continue
                note = (
                    "  (confounded on this dataset)"
                    if rate is not None and rate < threshold
                    else ""
                )
                print(f"    {rate_text(rate)}  vs {alternative}{note}")

    print(
        f"\n=== always-confounded groups (large random run, n={report['random_n']}) ==="
    )
    if report["always_confounded"]:
        for group in report["always_confounded"]:
            print(f"  {{ {', '.join(group)} }}")
        print("  These groups are empirical; a rare distinguishing pair may be absent.")
    else:
        print("  none")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare causal-model hypotheses on counterfactual datasets."
    )
    parser.add_argument("hypotheses_dir", type=Path)
    parser.add_argument(
        "--export-dir",
        type=Path,
        help="Empty directory for exact pair tables and predictions",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSON path (default: HYPOTHESES_DIR/distinguishability.json)",
    )
    parser.add_argument("--n", type=int, default=300, help="Pairs per designed dataset")
    parser.add_argument(
        "--random-n", type=int, default=1_000, help="Pairs in the broad random check"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--confound-threshold",
        type=float,
        default=0.2,
        help="Only controls which low rates are annotated in the console summary",
    )
    args = parser.parse_args()
    if args.n < 1 or args.random_n < 1:
        parser.error("--n and --random-n must be at least 1")
    return args


def main() -> None:
    args = parse_args()
    report = build_report(
        args.hypotheses_dir,
        n=args.n,
        random_n=args.random_n,
        seed=args.seed,
        export_dir=args.export_dir,
    )
    output = args.output or args.hypotheses_dir / "distinguishability.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print_summary(report, args.confound_threshold)
    print(f"\nWrote {output}")


if __name__ == "__main__":
    main()
