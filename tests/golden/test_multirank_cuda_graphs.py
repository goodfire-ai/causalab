"""CUDA graphs across ranks against the eager path at the same geometry, on
real Qwen (``docs/cuda_graphs.md`` "Multi-rank execution").

One document: a DAS layer sweep (block outputs, Cayley ``k=8``) fit as a
captured two-member cohort on the months table, with every slot full — 39
training rows at ``pairs=13`` — and a fixed step count, so the captured
frame is the eager cohort's arithmetic exactly. It runs at ``tp=2`` on the
dense Qwen3-4B and at ``ep=2`` on the Qwen3.6 A3B (layers 12 and 18), and at
``dp=2`` and ``tp=2,dp=2`` on the Qwen3-4B (layers 12, 18, 24 and 30, two
per replica), bf16, eager and with ``--cuda-graphs``: every output file is
byte-identical between the two, the receipts agree, and every rank of the
graph run replayed its graphs with no fallback to the eager path
(``tests/golden/_graph_census.py``). Graphs against world 1 is not the
claim: sharding moves reductions either way (``docs/model_parallelism.md``
§11), and that band is the parallel golden's. Each case skips, naming the
count, below its world's CUDA devices.

Every run is a ``causalab run`` subprocess through the spawn parent, one
geometry's world resident at a time. Set ``CAUSALAB_MULTIRANK_GRAPHS_ROOT``
to run under (and resume from) a kept root.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

import pytest
import torch

from causalab.protocol.parallel import parse_geometry
from tests.golden import _graph_census as census
from tests.golden import _parallel as par
from tests.golden import _parallel_worlds as worlds

pytestmark = [
    pytest.mark.golden,
    pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two CUDA devices"),
]

ROOT_VARIABLE = "CAUSALAB_MULTIRANK_GRAPHS_ROOT"
REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "benchmarks" / "cuda_graphs" / "data"

#: The geometry and the model it runs on.
CASES: dict[str, str] = {
    "tp=2": "Qwen/Qwen3-4B-Instruct-2507",
    "ep=2": "Qwen/Qwen3.6-35B-A3B",
    "dp=2": "Qwen/Qwen3-4B-Instruct-2507",
    "tp=2,dp=2": "Qwen/Qwen3-4B-Instruct-2507",
}
#: The swept layers, `MEMBERS` per replica of the data axis over points. A
#: replica runs a contiguous shard of the campaign's points
#: (``causalab.protocol.publish.point_shards``), and its cohort is its shard:
#: at ``dp=2`` two layers would leave each replica one member, which no
#: cohort graph holds (``graph_cohort.cohort_graph_reason``). Four layers
#: give each replica two — 12 and 18, the model-parallel cases' cohort, and
#: 24 and 30. Rows are not split over points: every member still fits all 39
#: rows in three full minibatches of 13, in a 26-row cohort slot.
LAYERS = (12, 18, 24, 30)
MEMBERS = 2
#: 39 training rows in three full minibatches: no padded slot, whose
#: repeated rows change MoE group sizes and bf16 rounding
#: (``docs/cuda_graphs.md`` "Training and evaluation").
PAIRS = 13
#: Two epochs: six captured steps, and two evaluations — the first eager (the
#: layout's first use), the second replayed.
EPOCHS = 2
#: Above the cohort's 26 slot rows, so its members share one window.
FIT_ROWS = "64"


def point_replicas(geometry: str) -> int:
    """How many replicas split the campaign's points: the data axis over
    points, else 1 (``dp=N:rows`` runs every point on every replica)."""
    parsed = parse_geometry(geometry)
    return parsed.data if parsed.data_mode == "points" else 1


def sweep(geometry: str) -> list[int]:
    """The layers ``geometry``'s document sweeps: `MEMBERS` per point
    replica, so each replica's cohort has `MEMBERS` members."""
    return list(LAYERS[: MEMBERS * point_replicas(geometry)])


def document(model: str, layers: list[int]) -> dict[str, Any]:
    return {
        "header": {
            "protocol_version": "4",
            "description": "multi-rank CUDA graphs: a full-slot DAS layer sweep",
        },
        "model": {"key": model, "revision": "main", "dtype": "bf16"},
        "data": {
            "base": {"dataset": "months/data#train", "field": "input"},
            "counterfactual": {
                "dataset": "months/data#train",
                "field": "counterfactual_inputs[0]",
            },
        },
        "method": {
            "intervened_models": {
                "original_counterfactual": {
                    "input": "counterfactual",
                    "reads": ["v_cf"],
                },
                "patched": {"input": "base", "reads": ["logits"], "writes": ["patch"]},
            },
            "sites": {
                "target": {"component": "block_output", "layers": {"sweep": layers}},
                "lm_head": {"component": "lm_head"},
            },
            "featurizers": {
                "rot": {"kind": "subspace", "k": 8, "parametrization": "cayley"}
            },
            "reads": {
                "v_cf": {"site": "target", "pos": -1, "featurizer": "rot"},
                "logits": {"site": "lm_head", "pos": -1},
            },
            "writes": {
                "patch": {
                    "site": "target",
                    "pos": -1,
                    "featurizer": "rot",
                    "do": {"swap": "v_cf"},
                }
            },
            "train": {
                "objective": {
                    "ce": {
                        "weight": 1.0,
                        "read": "logits",
                        "model": "patched",
                        "aggregation": {"kind": "cross_entropy", "target": "label"},
                    }
                },
                "params": ["rot"],
                "optimizer": {"name": "adamw", "lr": 0.001, "weight_decay": 0.0},
                "steps": {"epochs": EPOCHS},
                "batch": {"pairs": PAIRS},
                "precision": {"feature": "fp32", "loss": "fp32"},
                "eval": {
                    "every": {"epochs": 1},
                    "split": "months/data#test",
                    "aggregations": {
                        "iia": {
                            "read": "logits",
                            "model": "patched",
                            "aggregation": {
                                "kind": "logit_diff",
                                "a": "cf_answer",
                                "b": "base_answer",
                            },
                        }
                    },
                },
                "seed": 0,
            },
            "save": [
                {"train": "iia", "file_path": "iia.json"},
                {"train": "ce", "file_path": "ce.json"},
                {"value": "rot", "site": "target", "file_path": "rot.safetensors"},
            ],
        },
    }


#: ``(geometry, "eager" | "graphs") → (the run's output, its wall seconds)``
Runs = Callable[[str, str], tuple[Path, float]]

CENSUS = "graph_census"


def _run(root: Path, geometry: str, mode: str) -> tuple[Path, float]:
    """``causalab run`` of the case's document at ``geometry`` through the
    census entry, eager or with ``--cuda-graphs``; a kept run whose receipt
    exists is not repeated (its wall is then ``nan``)."""
    base = root / par.out_name(geometry)
    base.mkdir(parents=True, exist_ok=True)
    doc = base / "document.json"
    doc.write_text(json.dumps(document(CASES[geometry], sweep(geometry)), indent=2))
    out = base / mode
    if (out / par.RECEIPT).exists():
        return out, float("nan")
    arguments = [
        "run",
        str(doc),
        "--engine",
        "pytorch_hooks",
        "--record",
        "--data-root",
        str(DATA),
        "--artifacts-root",
        str(base),
        "--out",
        str(out),
        "--device",
        "cuda",
        "--fit-rows",
        FIT_ROWS,
        "--parallel",
        geometry,
        *(["--cuda-graphs"] if mode == "graphs" else []),
    ]
    env = {
        **os.environ,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        census.CENSUS_VARIABLE: str(out / CENSUS),
    }
    started = time.monotonic()
    completed = subprocess.run(
        [sys.executable, "-m", census.__name__, *arguments],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        cwd=REPO,
    )
    wall = time.monotonic() - started
    if completed.returncode != 0:
        (out / par.RECEIPT).unlink(missing_ok=True)
        raise par.RunFailed(arguments, completed.returncode, completed.stderr)
    return out, wall


@pytest.fixture(scope="module")
def runs(tmp_path_factory: pytest.TempPathFactory) -> Runs:
    kept = os.environ.get(ROOT_VARIABLE)
    root = Path(kept) if kept else tmp_path_factory.mktemp("multirank_graphs")
    root.mkdir(parents=True, exist_ok=True)
    cache: dict[tuple[str, str], tuple[Path, float]] = {}

    def get(geometry: str, mode: str) -> tuple[Path, float]:
        if (geometry, mode) not in cache:
            cache[geometry, mode] = _run(root, geometry, mode)
        return cache[geometry, mode]

    return get


def _censuses(out: Path, geometry: str) -> list[dict[str, Any]]:
    return [
        json.loads(census.census_path(out / CENSUS, rank).read_text())
        for rank in range(par.world_of(geometry))
    ]


def _cases() -> list[Any]:
    """Every geometry, gated on its own world's CUDA devices."""
    return [
        pytest.param(geometry, marks=worlds.needs(par.world_of(geometry)), id=geometry)
        for geometry in sorted(CASES)
    ]


#: The graph path's words for turning eager (``cuda_graphs``,
#: ``graph_cohort``, ``train``).
FALLBACKS = ("CUDA graphs disabled", "eagerly", "uses eager execution")


@pytest.mark.parametrize("geometry", _cases())
def test_graphs_are_the_eager_run_of_the_same_geometry_to_the_byte(
    runs: Runs, geometry: str
) -> None:
    eager, eager_wall = runs(geometry, "eager")
    graphs, graphs_wall = runs(geometry, "graphs")
    # the whole run's wall, model load included; shown by `-rA`
    print(f"{geometry}: eager {eager_wall:.1f} s, graphs {graphs_wall:.1f} s")
    assert par.exact_differences(eager, graphs) == []
    assert (
        par.receipts_agree(eager, graphs, geometry, reference_geometry=geometry) == []
    )


@pytest.mark.parametrize("geometry", _cases())
def test_every_rank_replays_its_graphs_and_none_falls_back(
    runs: Runs, geometry: str
) -> None:
    graphs, _ = runs(geometry, "graphs")
    for rank in _censuses(graphs, geometry):
        assert rank["replays"] > 0, rank
        assert any("cohort graph captured" in m for m in rank["messages"]), rank
        fallbacks = [
            m for m in rank["messages"] if any(word in m for word in FALLBACKS)
        ]
        assert fallbacks == [], rank
    eager, _ = runs(geometry, "eager")
    assert all(rank["replays"] == 0 for rank in _censuses(eager, geometry))
