"""Build sourced Qwen/H100/B200 catalogs offline: python -m examples.sol.build_qwen36."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from causalab.sol.hardware import b200_sxm, h100_sxm
from causalab.sol.model import catalog
from causalab.sol.qwen36_v2 import CONTRACT
from causalab.sol.qwen36 import Qwen36A3B, qwen36_workloads


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/sol/qwen36"))
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--sequence", type=int, default=128)
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--source-batches", type=int, default=10)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--suffix-layers", type=int, default=13)
    parser.add_argument("--rank", type=int, default=8)
    args = parser.parse_args()
    model = Qwen36A3B(args.batch, args.sequence)
    works = qwen36_workloads(
        model,
        updates=args.updates,
        source_batches=args.source_batches,
        eval_batches=args.eval_batches,
        suffix_layers=args.suffix_layers,
        rank=args.rank,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    def write(name: str, value: dict) -> None:
        (args.output_dir / name).write_text(
            json.dumps(value, indent=2, allow_nan=False) + "\n"
        )

    metadata = {
        **model.metadata(),
        "catalog_contract": CONTRACT,
        "forward_operation_breakdown_scope": "whole-forward architecture diagnostic; see workload phases for contract-specific work",
    }
    write("model.json", metadata)
    lines = [
        "# Qwen3.6-35B-A3B on H100 SXM / B200 SXM",
        "",
        f"Execution contract: `{CONTRACT}`. See the methodology for output/cache requirements.",
        "",
        "Generated with `uv run python -m examples.sol.build_qwen36`. "
        "Published hardware peaks; analytical costs, not measured latency. "
        "See `docs/speed_of_light.md` in the source checkout for the methodology.",
        "",
        f"Global batch **{args.batch}**, padded sequence **{args.sequence}**, BF16 text weights, "
        f"FP32 featurizers and Delta core ({model.delta_algorithm}). All times below are **seconds**.",
        "",
        f"Training totals: {args.updates} updates, {args.source_batches} cold source batches, "
        f"{args.eval_batches} evaluation forward batches, last {args.suffix_layers} layers in backward, "
        f"rank {args.rank} at one residual-stream position. Loss costs are excluded; v2 includes evaluation featurizer work.",
        "",
        "Expert traffic assumes independent uniform top-8 routing. Transient memory is an incomplete estimate; "
        "memory feasibility is unknown unless persistent residency exceeds capacity. Only implemented data-parallel layouts are enumerated.",
        "",
    ]
    for key, hw in (("h100_sxm", h100_sxm()), ("b200_sxm", b200_sxm())):
        result = catalog(works, hw)
        result["model"] = metadata
        result["interpretation"] += (
            " Expert weight traffic is conditional on uniform independent routing."
        )
        write(
            f"{key}.input.json",
            {
                "schema_version": 1,
                "hardware": asdict(hw),
                "workloads": [asdict(w) for w in works],
            },
        )
        write(f"{key}.catalog.json", result)
        lines += [
            f"## {hw.name}",
            "",
            f"[Full phase/layout catalog]({key}.catalog.json) · [Portable input]({key}.input.json)",
            "",
            "| Operation | 1 GPU | 2 GPUs DP | 4 GPUs DP | 8 GPUs DP |",
            "|---|---:|---:|---:|---:|",
        ]
        for work in works:
            cells = []
            for dp, tp in ((1, 1), (2, 1), (4, 1), (8, 1)):
                rows = [
                    r
                    for r in result["references"]
                    if r["workload"] == work.name and r["dp"] == dp and r["tp"] == tp
                ]
                cells.append(
                    "n/a"
                    if not rows
                    else "persistent residency exceeds capacity"
                    if rows[0]["sol_seconds"] is None
                    else f"{rows[0]['sol_seconds']:.6f}"
                )
            lines.append("| " + work.name + " | " + " | ".join(cells) + " |")
        lines.append("")
    (args.output_dir / "README.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
