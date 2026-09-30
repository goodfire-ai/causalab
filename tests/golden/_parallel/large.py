"""The large model's documents (``docs/model_parallelism.md`` §10.6 "The
large model", §11): the parallel golden on ``meta-llama/Llama-3.1-70B`` —
131.42 GiB of bf16 weights, a checkpoint that fits no single card — on
eight H100s, and across two nodes of four.

There is **no world-1 run**: the resident weights exceed one card. The
documents use the recorded **oracle geometry** (``runs.Document.oracle``)
``pp=4`` (placement, no reduction moves, §8). Admission can also allow
``tp=2`` and ``pp=2`` with a headroom warning; those geometries are outside
this record's coverage. The rules:

- every other exact geometry (``pp=8``) is byte-identical to the oracle —
  which is the pairwise identity of the exact set;
- the tensor geometries (``tp=4``, ``tp=2,pp=2``, ``tp=8``) land within the
  format-2 band (`.bands`) measured against the oracle: the bf16 rule
  the A3B's ``tp=2`` is held to, justified for a dense family by the tiny
  Llama's ``tp`` at the fp32 ulp and the dense fp32 fit at the rule's floor
  (§10.6) — the reduction order is the only difference a tensor group adds;
- every rank's load report obeys the several-axes loader rule
  (``tests/golden/_parallel_worlds.load_problems``: ``1 / tp`` for a
  tensor-sharded parameter, whole otherwise, stages disjoint), holds one
  copy of what it read (``residency_problems``) and ``bytes_unowned`` is 0;
- every rank's measured peak (the recorder's, at exit) sits **within the
  pre-flight's estimated footprint** and **above its resident weights**
  (``runs.memory_problems``), and the estimate's slack over the peak is
  written into the record (``estimate`` beside ``memory``). The rule was
  measured on cards that fill (§11: within 5 % above 60 GiB); on this
  model no fitting geometry comes within a quarter of the card, so the
  golden pins the ratio and does not assert a 5 % agreement it cannot have.

Two documents (`DOCUMENTS`), both through the recording entry
(`.recorder`) so every rank writes its peak memory:

- ``large`` — the inference document (`.inference`) authored from the
  smoke's **dense** template (the tiny Llama's: ``stream``,
  ``sharded_read``, ``sharded_write``; no experts, no routing table), the
  target swept over layers 3 and 43 — on different stages of every pipeline
  (``pp=2`` stage 1, ``pp=4`` stage 2, ``pp=8`` stage 4); ``pp=4`` the
  oracle, ``pp=8`` exact, ``tp=4`` / ``tp=2,pp=2`` / ``tp=8`` banded;
- ``das_large`` — the corpus DAS fit at ``attention_query`` (`.fit`,
  ``--fit-rows 16``, ten epochs, no early stop) at ``pp=4`` (oracle) and
  ``pp=8`` exact, ``tp=4`` banded, the §7 gradient classes included.

**Across two nodes.** A case spelled ``<geometry>@2x4`` is the same
document run as a ``torchrun`` world of eight over two nodes of four
cards. The test process cannot launch it: the coordinator runs
`torchrun_argv` on each node (``python -m tests.golden._parallel.large
commands …`` prints the exact lines), writing into `two_node_out`
under the kept root, and the golden picks the run up from there — skipping
by name, with the commands, while it is absent — and holds it to the same
rules with the receipt's launcher ``joined``.

Every device gate is a function of the device count (`skip_reason`),
held on the CPU by ``tests/golden/test_parallel_large_rules.py``.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import os
import sys
from pathlib import Path
from typing import Any

from huggingface_hub import constants as hf_constants

from causalab.protocol.checkpoint_census import checkpoint_targets, tree_of
from causalab.protocol.parallel import parse_geometry
from causalab.protocol.parallel_memory import (
    DTYPE_ITEMSIZES,
    RULE,
    Placement,
    estimate_resident,
    placement_table,
)
from causalab.protocol.registry import get_model_info

from tests._helpers.header_census import LLAMA70B_CENSUS, load_census
from tests.golden import _parallel_worlds as worlds
from tests.golden._parallel import fit, inference, recorder, runs
from tests.golden._parallel.runs import LARGE, LARGE_MODEL, Document, Realization
from tests.neural.engines.pytorch_hooks.conftest import TINY_LLAMA

__all__ = [
    "CENSUS",
    "DAS",
    "DOCUMENTS",
    "INFERENCE",
    "NODE_CARDS",
    "TWO_NODE",
    "TWO_NODE_CASES",
    "describe",
    "devices_needed",
    "estimate",
    "geometry_of_case",
    "hub_cache",
    "is_two_node",
    "skip_reason",
    "table",
    "torchrun_argv",
    "torchrun_environment",
    "two_node_out",
    "two_node_skip",
]

MODEL = LARGE_MODEL
REALIZATION: Realization = LARGE
CENSUS = LLAMA70B_CENSUS

ORACLE = "pp=4"
EXACT: tuple[str, ...] = (ORACLE, "pp=8")
BANDED: tuple[str, ...] = ("tp=4", "tp=2,pp=2", "tp=8")


# --------------------------------------------------------------------------- #
# the pre-flight's estimate off the census
# --------------------------------------------------------------------------- #


@functools.lru_cache(maxsize=1)
def table() -> dict[str, Placement]:
    """The 70B's placement table off its header census — the same table
    the loader's pre-flight builds off the real headers (§2)."""
    info = get_model_info(MODEL)
    census = load_census(CENSUS)
    elements: dict[str, int] = {}
    for key, (_, shape) in census.items():
        count = 1
        for n in shape:
            count *= n
        elements[key] = count
    tree = tree_of(elements, info.num_layers)
    if tree is None or info.parallel_plan is None:
        raise RuntimeError(f"{MODEL}: no registered tree matches its census")
    targets = checkpoint_targets(elements, tree, info.num_layers)
    if targets is None:
        raise RuntimeError(f"{MODEL}: the census's text tower cannot be told apart")
    return placement_table(elements, targets, info.parallel_plan, tree, info)


def estimate(geometry: str) -> dict[str, dict[str, int]]:
    """Per rank of ``geometry``, the resident weights and the footprint the
    measured rule expects (``parallel_memory``), at the realization's dtype
    — what ``dry-run --parallel`` prints and the pre-flight reports."""
    parsed = parse_geometry(geometry)
    itemsize = DTYPE_ITEMSIZES[REALIZATION.dtype]
    resident = estimate_resident(table(), parsed, itemsize)
    footprint = RULE.footprint(table(), parsed, itemsize)
    return {
        f"rank{rank}": {"resident": resident[rank], "footprint": footprint[rank]}
        for rank in range(parsed.world)
    }


# --------------------------------------------------------------------------- #
# the documents
# --------------------------------------------------------------------------- #


def describe(realization: Realization) -> dict[str, Any]:
    return {
        "protocol": "02_interchange_im + the tp smoke tier's boundary reads and writes (dense)",
        "layer": realization.layer,
        "sweep": list(inference.sweep(realization)),
    }


def _author(tmp: Path, realization: Realization) -> Path:
    return inference.author(tmp, realization, template=TINY_LLAMA)


INFERENCE = Document(
    name="large",
    exact=EXACT,
    banded=BANDED,
    author=_author,
    describe=describe,
    measure=inference.measure_run,
    classes=inference.DENSE_CLASSES,
    recorded=True,
    realization=REALIZATION,
    oracle=ORACLE,
    load_rule=worlds.load_problems,
    estimate=estimate,
)

DAS = dataclasses.replace(
    fit.DAS,
    name="das_large",
    exact=EXACT,
    banded=("tp=4",),
    realization=REALIZATION,
    oracle=ORACLE,
    load_rule=worlds.load_problems,
    estimate=estimate,
)

#: The large model's documents by name, in capture order — kept apart from
#: the two-rank record's `DOCUMENTS` so the
#: 2-GPU replay and the default capture never launch a world of eight.
DOCUMENTS: dict[str, Document] = {d.name: d for d in (INFERENCE, DAS)}


# --------------------------------------------------------------------------- #
# device gates
# --------------------------------------------------------------------------- #


def devices_needed(document: Document) -> int:
    """The largest world among the document's geometries: what one node
    must show to run it whole."""
    return max(runs.world_of(g) for g in document.geometries)


def skip_reason(
    devices: int, geometry: str, document: Document | None = None
) -> str | None:
    """Why a case cannot run on a node with ``devices`` CUDA devices, by
    name — or ``None`` when it can. A case needs its own world; with a
    ``document`` given, the oracle's world too (every comparison rests on
    the oracle's run). A two-node case runs on no single node's count: it
    is picked up from the kept root (`two_node_skip`)."""
    if is_two_node(geometry):
        return None
    needed = runs.world_of(geometry)
    if document is not None and document.oracle is not None:
        needed = max(needed, runs.world_of(document.oracle))
    if devices < needed:
        return f"{geometry} needs {needed} CUDA devices; this node shows {devices}"
    return None


# --------------------------------------------------------------------------- #
# across two nodes
# --------------------------------------------------------------------------- #

#: The case suffix: the geometry as a ``torchrun`` world over two nodes of
#: `NODE_CARDS` cards each.
TWO_NODE = "@2x4"
NODE_CARDS = 4
TWO_NODE_CASES: tuple[str, ...] = (f"pp=8{TWO_NODE}", f"tp=8{TWO_NODE}")


def is_two_node(case: str) -> bool:
    return case.endswith(TWO_NODE)


def geometry_of_case(case: str) -> str:
    """The ``--parallel`` text of a case, its two-node suffix removed."""
    return case[: -len(TWO_NODE)] if is_two_node(case) else case


def two_node_out(root: Path, document: Document, case: str) -> Path:
    """Where the two-node run of ``case`` lives under the kept root: beside
    the spawned geometries' directories, suffixed ``_2x4``."""
    return root / document.name / f"{runs.out_name(geometry_of_case(case))}_2x4"


def hub_cache() -> str:
    """The Hub cache every rank of a two-node run reads the 70B from.

    ``HF_HUB_CACHE`` when the coordinator sets it, else huggingface_hub's own
    default (``huggingface_hub.constants.HF_HUB_CACHE``, which also honours
    ``HF_HOME``). Read at call time, so the printed commands name the cache
    of the shell that prints them.
    """
    return os.environ.get("HF_HUB_CACHE") or hf_constants.HF_HUB_CACHE


def torchrun_environment(out: Path, document: Document) -> dict[str, str]:
    """The environment every rank of a two-node run needs: offline, the
    Hub cache (`hub_cache`), the one interface, the load reports and the recorder
    under ``out`` (the same layout ``runs.run`` gives a spawned run), the
    §7 check for a fit."""
    env = {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_CACHE": hub_cache(),
        # eth0 is the conventional name of a node's first Ethernet interface.
        # When the nodes reach each other over an interface with another name,
        # set NCCL_SOCKET_IFNAME and GLOO_SOCKET_IFNAME to it in the printed
        # command.
        "NCCL_SOCKET_IFNAME": "eth0",
        "GLOO_SOCKET_IFNAME": "eth0",
        "OMP_NUM_THREADS": "1",
        "CUDA_VISIBLE_DEVICES": ",".join(str(i) for i in range(NODE_CARDS)),
        "CAUSALAB_LOAD_REPORT_DIR": str(out / runs.REPORTS),
    }
    if document.recorded:
        env[recorder.GRADIENTS_VARIABLE] = str(out / runs.GRADIENTS)
    if document.trains:
        env[runs.GRADIENT_AGREEMENT_VARIABLE] = runs.gradient_agreement()
    return env


def torchrun_argv(
    document: Document,
    authored: Path,
    out: Path,
    case: str,
    *,
    node_rank: int,
    endpoint: str,
    python: str = ".venv/bin/python",
) -> list[str]:
    """The ``torchrun`` command one node types for a two-node case: two
    nodes of `NODE_CARDS` ranks, the c10d rendezvous at ``endpoint``
    (``<node0 address>:<port>``), the recording entry where the document
    is recorded, ``causalab run`` of the authored document into ``out``
    with the case's geometry — the arguments ``runs.run`` gives a spawn."""
    if not is_two_node(case):
        raise ValueError(f"{case!r} is not a two-node case ({TWO_NODE})")
    if node_rank not in (0, 1):
        raise ValueError(f"node_rank must be 0 or 1, got {node_rank}")
    entry = recorder.__name__ if document.recorded else "causalab.cli"
    return [
        python,
        "-m",
        "torch.distributed.run",
        "--nnodes",
        "2",
        "--nproc-per-node",
        str(NODE_CARDS),
        "--node-rank",
        str(node_rank),
        "--rdzv-backend",
        "c10d",
        "--rdzv-endpoint",
        endpoint,
        "-m",
        entry,
        "run",
        str(authored),
        "--engine",
        "pytorch_hooks",
        "--record",
        "--data-root",
        str(runs.DATA),
        "--artifacts-root",
        str(authored.parent),
        "--out",
        str(out),
        "--device",
        REALIZATION.device,
        *document.argv,
        "--parallel",
        geometry_of_case(case),
    ]


def two_node_skip(root: Path | None, document: Document, case: str) -> str | None:
    """Why a two-node case cannot be judged yet, by name: no kept root, or
    no receipt under `two_node_out` — naming the command that
    produces one. ``None`` when the run is there."""
    if root is None:
        return (
            f"{case}: a two-node run is picked up from the kept root; set "
            f"CAUSALAB_PARALLEL_GOLDENS_ROOT and run "
            f"`python -m {__name__} commands --root ROOT --endpoint HOST:PORT`"
        )
    out = two_node_out(root, document, case)
    if not (out / runs.RECEIPT).exists():
        return (
            f"{case}: no receipt under {out}; on each node run the torchrun line "
            f"`python -m {__name__} commands --root {root} --endpoint HOST:PORT` prints"
        )
    return None


def _commands(root: Path, endpoint: str, python: str) -> list[str]:
    lines: list[str] = []
    for document in DOCUMENTS.values():
        base = root / document.name
        base.mkdir(parents=True, exist_ok=True)
        authored = document.author(base, runs.realization_of(document))
        for case in TWO_NODE_CASES:
            if geometry_of_case(case) not in document.geometries:
                continue
            out = two_node_out(root, document, case)
            env = " ".join(
                f"{k}={v}" for k, v in torchrun_environment(out, document).items()
            )
            for node_rank in (0, 1):
                argv = torchrun_argv(
                    document,
                    authored,
                    out,
                    case,
                    node_rank=node_rank,
                    endpoint=endpoint,
                    python=python,
                )
                lines.append(f"# {document.name} {case}, node {node_rank}")
                lines.append(f"{env} {' '.join(argv)}")
    return lines


def main(argv: list[str] | None = None) -> int:
    """``python -m tests.golden._parallel.large commands --root ROOT
    --endpoint HOST:PORT [--python .venv/bin/python]``: author the large
    documents under ``ROOT`` and print, per document, two-node case and
    node, the exact environment and ``torchrun`` line."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    sub = parser.add_subparsers(dest="verb", required=True)
    commands = sub.add_parser("commands")
    commands.add_argument("--root", type=Path, required=True)
    commands.add_argument("--endpoint", required=True, help="<node0 address>:<port>")
    commands.add_argument("--python", default=".venv/bin/python")
    args = parser.parse_args(argv)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    for line in _commands(args.root, args.endpoint, args.python):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
