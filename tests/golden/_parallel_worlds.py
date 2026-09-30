"""The world-4 / world-8 tiers' shared half (``docs/model_parallelism.md`` §2,
§5.3, §10.6): the geometry's receipt block, the per-test device gate, the
sweep that gives ``dp`` its points, the smoke tier's drift rule for pinned
maxima, and the loader rule generalised from the two-rank record's
``1 / world`` to a geometry with several axes — used by the tiny-fixture
CUDA twin of the smokes (``test_parallel_worlds.py``), the A3B module
(``test_parallel_world4.py``) and their CPU guard
(``test_parallel_worlds_rules.py``), so every rule is written once and held
without a GPU.

**The loader rule under several axes.** The two-rank record
(``tests/golden/parallel_goldens.json``) holds every sharded parameter at
exactly ``1 / world`` of its bytes on disk — the rule while
``tensor == expert == model == world``. In general a parameter sharded on
the tensor axis is read at ``1 / tensor`` and one on the expert axis at
``1 / expert`` (§5.3: the shard plan follows the plan row's axis), the data
and context axes replicate, and a pipeline partitions the parameters
between its stages. So at ``tp=2,ep=4`` (world 4) attention weights are read
at ``1 / 2`` and experts at ``1 / 4``; at ``dp=2,tp=2`` (world 4) every
sharded parameter is ``1 / 2``; at ``ep=8`` every sharded parameter is
``1 / 8 = 1 / world``. `load_problems` is that rule: every sharded
parameter's denominator is a model-group axis above one, the ranks of one
pipeline stage name the same parameters and shard the same ones, two stages
have none in common, and an axis of one shards nothing. A K/V projection
replicated above the KV heads (§6.6, ``tp=8`` on the tiny MoE's four KV
heads) is read whole and is simply not among the sharded.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest
import torch

from causalab.neural.engines.pytorch_hooks.residency import residency_problems
from causalab.protocol.parallel import MeshLayout, ParallelGeometry, parse_geometry
from causalab.protocol.receipt import parallel_record

__all__ = [
    "block",
    "geometry_of",
    "load_problems",
    "needs",
    "pin_problems",
    "receipt",
    "receipt_problems",
    "sharded_denominators",
    "stage_names",
    "swept",
    "totals",
]

RECEIPT = "protocol.json"


def needs(world: int) -> pytest.MarkDecorator:
    """The gate of one case: skipped, naming the count, below ``world`` CUDA
    devices — a spawn puts one rank on each device of the node (§3)."""
    return pytest.mark.skipif(
        torch.cuda.device_count() < world, reason=f"needs {world} CUDA devices"
    )


def geometry_of(text: str) -> ParallelGeometry:
    """``--parallel`` text to geometry (the CLI's own grammar)."""
    return parse_geometry(text)


def block(text: str, launcher: str = "spawned") -> dict[str, Any]:
    """The receipt's ``execution.parallel`` block a run at ``--parallel text``
    writes under ``launcher`` (§9) — the runner's own builder, so the test
    cannot spell the block differently from the receipt."""
    return parallel_record(geometry_of(text), launcher)


def swept(document: Path, layers: Sequence[int]) -> Path:
    """``document`` with its ``target`` site swept over ``layers`` — one point
    per layer, so ``dp`` over points has points to shard and a pipeline a
    write on more than one stage — written beside it as ``<stem>_swept.json``."""
    doc = json.loads(document.read_text())
    doc["method"]["sites"]["target"]["layers"] = {"sweep": list(layers)}
    target = document.with_name(f"{document.stem}_swept.json")
    target.write_text(json.dumps(doc, indent=2))
    return target


# --------------------------------------------------------------------------- #
# the receipt
# --------------------------------------------------------------------------- #


def receipt(out: Path) -> dict[str, Any]:
    return json.loads((out / RECEIPT).read_text())


def receipt_problems(
    solo: Path, parallel: Path, text: str, launcher: str = "spawned"
) -> list[str]:
    """How the parallel run's receipt disagrees with the world-1 run's beyond
    ``execution.parallel``, which must be [`block`][causalab.neural.engines.pytorch_hooks.kernels.norm_triton.RowLaunch.block] of ``text``; the
    fit's measured bounds (``fit_rows_resolved`` / ``fit_rows_shrinks``)
    are inside ``execution`` and so held key by key with the rest."""
    a, b = receipt(solo), receipt(parallel)
    problems: list[str] = []
    expected = block(text, launcher)
    if b["execution"]["parallel"] != expected:
        problems.append(
            f"execution.parallel is {b['execution']['parallel']!r}, not {expected!r}"
        )
    if a["execution"]["parallel"]["launcher"] != "solo":
        problems.append("the oracle was not a solo run")
    del a["execution"]["parallel"], b["execution"]["parallel"]
    for key in ("fit_rows_resolved", "fit_rows_shrinks"):
        if a["execution"].get(key) != b["execution"].get(key):
            problems.append(
                f"execution.{key}: {b['execution'].get(key)!r} != "
                f"{a['execution'].get(key)!r} at world 1"
            )
    if json.dumps(a, sort_keys=True) != json.dumps(b, sort_keys=True):
        problems.append("the receipts differ beyond execution.parallel")
    return problems


# --------------------------------------------------------------------------- #
# pinned maxima: the smoke tier's drift rule
# --------------------------------------------------------------------------- #


def pin_problems(
    label: str,
    measured: Mapping[str, float],
    pinned: Mapping[str, Mapping[str, float]],
    *,
    band: float,
    floor: float,
) -> list[str]:
    """The smoke tier's rule for a measured run against its pinned maxima
    (``test_tensor_expert_parallel_run._assert_measured``): every class
    within ``band``; every class the pin names measured, no other; and none
    past twice its pinned maximum (or ``floor``) — a drift past the record
    is a re-measurement, not a silent pass. A label with no pin is a
    problem, never a vacuous pass."""
    problems = [
        f"{label} {kind}: {worst!r} > band {band!r}"
        for kind, worst in measured.items()
        if not worst <= band
    ]
    recorded = pinned.get(label)
    if recorded is None:
        return [*problems, f"{label}: no pinned maxima; measured {dict(measured)!r}"]
    if set(measured) != set(recorded):
        problems.append(
            f"{label}: classes {sorted(measured)} != pinned {sorted(recorded)}"
        )
    for kind, worst in measured.items():
        if kind in recorded and not worst <= max(recorded[kind] * 2, floor):
            problems.append(
                f"{label} {kind}: {worst!r} past twice the pinned {recorded[kind]!r}"
            )
    return problems


# --------------------------------------------------------------------------- #
# the loader under several axes
# --------------------------------------------------------------------------- #


def sharded_denominators(record: Mapping[str, Any]) -> dict[str, float]:
    """Per parameter read at less than its bytes on disk, the ratio
    ``on_disk / requested`` — the shard count it was read at (an integer for
    a contiguous ``1 / n`` shard)."""
    requested, on_disk = record["bytes_requested"], record["bytes_on_disk"]
    return {
        name: on_disk[name] / requested[name]
        for name in requested
        if name in on_disk and requested[name] != on_disk[name]
    }


def _stage(geometry: ParallelGeometry, rank: int) -> int:
    return MeshLayout(geometry).rank_in(rank, "pipeline")


def load_problems(text: str, reports: Sequence[Mapping[str, Any]]) -> list[str]:
    """How a geometry's per-rank load reports fall short of §5.3 under
    several axes (module docstring)."""
    geometry = geometry_of(text)
    world = geometry.world
    problems: list[str] = []
    if len(reports) != world:
        return [f"{len(reports)} reports for a world of {world}"]
    allowed = {geometry.tensor, geometry.expert} - {1}
    names: list[set[str]] = []
    sharded: list[set[str]] = []
    for rank, record in enumerate(reports):
        if record["rank"] != rank or record["world"] != world:
            problems.append(
                f"report {rank} says rank {record['rank']} of {record['world']}"
            )
        requested, on_disk = record["bytes_requested"], record["bytes_on_disk"]
        if set(requested) != set(on_disk) or not requested:
            problems.append(
                f"rank {rank}: the report's two tables name different parameters"
            )
        denominators = sharded_denominators(record)
        for name, ratio in sorted(denominators.items()):
            if ratio != int(ratio) or int(ratio) not in allowed:
                problems.append(
                    f"rank {rank}: {name} read at 1/{ratio:g} of its bytes; the "
                    f"axes above one are {sorted(allowed) or 'none'}"
                )
        # one copy (residency.py): resident elements equal the plan's, the
        # device holds and reserves no more than the parameters
        problems.extend(f"rank {rank}: {p}" for p in residency_problems(record))
        names.append(set(requested))
        sharded.append(set(denominators))
    if allowed:
        if not sharded[0]:
            problems.append("no parameter is sharded")
        if any(s == n for s, n in zip(sharded, names)):
            problems.append("every parameter is sharded; none is replicated")
    elif any(sharded):
        problems.append(
            f"{text} shards a parameter: {sorted(set().union(*sharded))[:3]}"
        )
    by_stage: dict[int, list[int]] = {}
    for rank in range(world):
        by_stage.setdefault(_stage(geometry, rank), []).append(rank)
    for stage, ranks in sorted(by_stage.items()):
        first = ranks[0]
        for rank in ranks[1:]:
            if names[rank] != names[first]:
                problems.append(
                    f"stage {stage}: ranks {first} and {rank} name different parameters"
                )
            if sharded[rank] != sharded[first]:
                problems.append(
                    f"stage {stage}: ranks {first} and {rank} shard different parameters"
                )
    stages = sorted(by_stage)
    for i, a in enumerate(stages):
        for b in stages[i + 1 :]:
            shared = names[by_stage[a][0]] & names[by_stage[b][0]]
            if shared:
                problems.append(f"stages {a} and {b} both read {sorted(shared)[:3]}")
    return problems


def stage_names(text: str, reports: Sequence[Mapping[str, Any]]) -> set[str]:
    """Every parameter the stages of a run together name — one rank per
    stage — the set a pipeline must cover the model with."""
    geometry = geometry_of(text)
    seen: dict[int, set[str]] = {}
    for rank, record in enumerate(reports):
        seen.setdefault(_stage(geometry, rank), set(record["bytes_requested"]))
    return set().union(*seen.values()) if seen else set()


def totals(reports: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    """Per rank, the bytes requested and the bytes on disk, summed — the
    record's ``load`` block shape."""
    return {
        f"rank{record['rank']}": {
            "bytes_requested": sum(record["bytes_requested"].values()),
            "bytes_on_disk": sum(record["bytes_on_disk"].values()),
        }
        for record in reports
    }
