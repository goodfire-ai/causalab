"""The parallel golden's soak document (``docs/model_parallelism.md`` §10.6
"the soak", §11): the inference document (`.inference`) with its
target swept over **layers × two positions** — `POINTS` points on the
A3B (the first 25 layers, the answer slot and the one before it), eight on
the four-layer tiny MoE — captured through the recorder at `GEOMETRIES`, so every
rank writes the per-point memory trace (`.recorder`,
``runs.TRACES``), which [`flat_after_warmup`][causalab.neural.shared.parallel.soak.flat_after_warmup]
holds flat after `WARMUP` points within `SLACK` bytes per
point.

Nothing here is compared to world 1: the parity tiers hold the numbers,
this document holds the memory. It runs at the geometries alone
(`run_geometries`), no oracle.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from causalab.neural.shared.parallel.soak import Sample, flat_after_warmup
from causalab.protocol.registry import get_model_info

from tests.golden._parallel import inference, runs
from tests.golden._parallel.runs import Document, Realization

__all__ = [
    "DOCUMENT",
    "GEOMETRIES",
    "POINTS",
    "POSITIONS",
    "SLACK",
    "WARMUP",
    "author",
    "describe",
    "layers_swept",
    "problems",
    "run_geometries",
]

#: The two positions the swap is swept over, and the point budget: a run
#: long enough for a per-point leak to show against the warm-up (the
#: budget bounds the layers swept, ``POINTS // len(POSITIONS)`` of them).
POSITIONS: tuple[dict[str, Any], ...] = ({"index": -1}, {"index": -2})
POINTS = 50
#: The geometries the soak runs at: the two model axes that shard the A3B.
GEOMETRIES: tuple[str, ...] = ("ep=2", "tp=2")
#: The rule's parameters: the first five points fill the interning store
#: and open the allocator's pool; after them allocated bytes may drift by
#: at most one MiB per point — 45 MiB over the run, against a model of tens
#: of GiB — and the reserved pool may not climb every point.
WARMUP = 5
SLACK = 1 << 20


def layers_swept(realization: Realization) -> int:
    """How many layers the target is swept over: the point budget's share,
    never more than the model has."""
    return min(get_model_info(realization.model).num_layers, POINTS // len(POSITIONS))


def author(tmp: Path, realization: Realization) -> Path:
    """The inference document with its target swept over the first
    `layers_swept` layers and its write position over
    `POSITIONS` (the named position ``tap``, as corpus 07 sweeps)."""
    target = inference.author(tmp, realization)
    doc = json.loads(target.read_text())
    method = doc["method"]
    method["sites"]["target"]["layers"] = {
        "sweep": {"range": [0, layers_swept(realization)]}
    }
    method["reads"]["v_cf"]["pos"] = "tap"
    method["writes"]["patch"]["pos"] = "tap"
    # `positions` leads the method (docs/intervention_protocol.md §1 order)
    positions = {**method.get("positions", {}), "tap": {"sweep": list(POSITIONS)}}
    doc["method"] = {
        "positions": positions,
        **{k: v for k, v in method.items() if k != "positions"},
    }
    target.write_text(json.dumps(doc, indent=2))
    return target


def describe(realization: Realization) -> dict[str, Any]:
    return {
        **inference.describe(realization),
        "soak": {
            "layers": layers_swept(realization),
            "positions": list(POSITIONS),
            "points": layers_swept(realization) * len(POSITIONS),
        },
    }


DOCUMENT = Document(
    name="soak",
    exact=(),
    banded=GEOMETRIES,
    author=author,
    describe=describe,
    measure=inference.measure_run,
    classes=inference.DOCUMENT.classes,
    recorded=True,
)


def run_geometries(
    root: Path, realization: Realization, geometries: tuple[str, ...] = GEOMETRIES
) -> dict[str, Path]:
    """The soak document at every geometry — no world-1 oracle — under
    ``root / soak``; a run whose receipt exists resumes."""
    base = root / DOCUMENT.name
    base.mkdir(parents=True, exist_ok=True)
    authored = DOCUMENT.author(base, realization)
    return {
        geometry: runs.run(
            authored,
            base / runs.out_name(geometry),
            "--parallel",
            geometry,
            realization=realization,
            argv=DOCUMENT.argv,
            recorded=True,
        )
        for geometry in geometries
    }


def problems(
    traces: list[list[Sample]], *, warmup: int = WARMUP, slack: int = SLACK
) -> list[str]:
    """Every rank's trace under the rule, each problem naming its rank."""
    return [
        f"rank {rank}: {problem}"
        for rank, samples in enumerate(traces)
        for problem in flat_after_warmup(samples, warmup=warmup, slack=slack)
    ]
