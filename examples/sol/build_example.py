"""Generate illustrative inputs without downloading a model or probing a GPU."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from causalab.sol.model import Hardware
from causalab.sol.recipes import DenseTransformer, dense_catalog_workloads

hardware = Hardware(
    name="hypothetical-node (NOT a vendor specification)",
    flops_per_second={"bf16_dense": 1e15, "fp32": 6e13},
    hbm_bytes_per_second=3e12,
    memory_bytes=80e9,
    link_bytes_per_second=200e9,
    collective_latency_seconds=1e-6,
    provenance="Illustrative round numbers only; replace with sourced hardware specifications or measurements.",
)
workloads = dense_catalog_workloads(
    DenseTransformer(
        layers=32, width=4096, mlp_width=11008, vocabulary=32000, sequence=128, batch=16
    ),
    updates=100,
    source_batches=10,
    eval_batches=20,
    suffix_layers=13,
    rank=8,
)
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, default=Path("results/sol/input.json"))
args = parser.parse_args()
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(
    json.dumps(
        {
            "schema_version": 1,
            "hardware": asdict(hardware),
            "workloads": [asdict(w) for w in workloads],
        },
        indent=2,
    )
    + "\n"
)
