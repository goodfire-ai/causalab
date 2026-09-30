"""The parallel golden's runs (``docs/model_parallelism.md`` §10.6): a
[`Document`][causalab.protocol.schema.types.Document] (how to author and measure one document of the record), a
`Realization` (the model, dtype and device it runs on), the
``causalab run`` subprocesses, and the exact comparisons and loader rules
every document shares.

Every run is a subprocess on ``realization.device`` — the world-1 oracle
with no ``--parallel``, a geometry through the production spawn parent —
so the calling process never loads a model and one geometry's world is
resident at a time (``tests/golden/conftest.py``'s invariant, by
construction). Each rank writes its
[`LoadReport`][causalab.neural.engines.pytorch_hooks.shard_read.LoadReport] under
``out / REPORTS``; a recorded document (a fit) runs through
`.recorder`, which adds every rank's per-step gradients and peak memory
under ``out / GRADIENTS``.
"""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from causalab.neural.engines.pytorch_hooks.residency import residency_problems
from causalab.neural.shared.parallel.agreements import AGREEMENT_VARIABLE
from causalab.protocol.parallel import parse_geometry
from causalab.protocol.receipt import parallel_record

from tests.golden._parallel import recorder
from tests.golden._parallel.measure import Measured

__all__ = [
    "A3B",
    "DATA",
    "DENSE",
    "DENSE_MODEL",
    "Document",
    "EVENTS",
    "GRADIENTS",
    "GRADIENT_AGREEMENT",
    "GRADIENT_AGREEMENT_VARIABLE",
    "GRADIENT_CLASS",
    "LARGE",
    "LARGE_MODEL",
    "LoadRule",
    "MODEL",
    "REPORTS",
    "RECEIPT",
    "Realization",
    "RunFailed",
    "TRACES",
    "WORLD",
    "estimate_slack",
    "exact_differences",
    "gradient_agreement",
    "load_problems",
    "load_reports",
    "memory_problems",
    "memory_replay_problems",
    "out_name",
    "parallel_block",
    "receipt",
    "realization_of",
    "receipts_agree",
    "run",
    "run_all",
    "sharded_parameters",
    "traces",
    "world_of",
]

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "tests" / "protocol" / "fixtures" / "data"
MODEL = "Qwen/Qwen3.6-35B-A3B"
WORLD = 2

RECEIPT = "protocol.json"
EVENTS = "events.jsonl"
#: The subdirectories of a run's output the harness adds: every rank's load
#: report (``loading.LOAD_REPORT_VARIABLE``) and, for a recorded document,
#: the recorder's gradients and memory.
REPORTS = "load_reports"
GRADIENTS = "gradients"
#: the recorder's per-point memory trace, one ``rank<r>.jsonl`` per rank,
#: written for every recorded document (``recorder.MEMORY_TRACE_VARIABLE``)
TRACES = "memory_trace"

#: The §7 guard's runtime check (``agreements.AGREEMENT_VARIABLE``,
#: ``CAUSALAB_GRADIENT_AGREEMENT``: the tolerance, relative to the gradient's
#: largest entry, at which the ranks' gradients are compared before the
#: mean) is **on** for every fit run of the record: the children inherit the
#: environment, so the value below is set for a recorded document unless the
#: capture's environment already spells one, and the value used is written
#: into the record. ``1e-6`` is the gloo smokes' band
#: (``GRADIENT_AGREEMENT_GLOO``, measured ``0.0`` there); a real backend that
#: refuses names the disagreement, and the pinned value moves above it with
#: its measurement in the record.
GRADIENT_AGREEMENT_VARIABLE = AGREEMENT_VARIABLE
GRADIENT_AGREEMENT = "1e-6"


def gradient_agreement() -> str:
    """The tolerance the fit runs are checked at: the environment's when it
    spells one, else `GRADIENT_AGREEMENT`."""
    return os.environ.get(GRADIENT_AGREEMENT_VARIABLE) or GRADIENT_AGREEMENT


@dataclasses.dataclass(frozen=True)
class Realization:
    """The model a document runs on, its dtype and device, and the layer the
    documents tap — the A3B in bf16 on CUDA for the record, a tiny fixture
    in fp32 on the CPU for the harness's own smoke."""

    model: str
    dtype: str
    device: str
    layer: int


#: The record's realization: the first full-attention layer of the A3B's
#: 3-linear-then-1-full schedule (the tiny MoE twin's too).
A3B = Realization(MODEL, "bf16", "cuda", 3)

#: The dense fp32 realization of the ``das_dense`` fit (§10.6): the largest
#: registered dense model in the node's cache that two 80 GB cards hold in
#: fp32 with room for a fit's captures — 4B parameters, 16 GB of weights —
#: at the same layer index. In fp32 the forward's reduction-order noise is
#: at the fp32 ulp, so the fit is reproducible across geometries to a few
#: ulps where the A3B's bf16 fits are not.
DENSE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
DENSE = Realization(DENSE_MODEL, "fp32", "cuda", 3)

#: The model that fits no single card (§10.6 "The large model", §11): the
#: dense ``meta-llama/Llama-3.1-70B``, 131.42 GiB of bf16 weights over 80
#: layers, an untied head — a complete checkpoint of a served family whose
#: weights exceed an 80 GB card. Its documents have no
#: world-1 run (refused by name, ``tests/protocol/test_parallel_memory_large.py``)
#: and stand on an **oracle geometry** instead (`Document.oracle`).
#: The layer is a full-attention layer near the front, as the A3B's; the
#: inference sweep's second layer sits forty above it (``inference.SWEEP_STRIDES``),
#: on another stage of every pipeline the golden runs.
LARGE_MODEL = "meta-llama/Llama-3.1-70B"
LARGE = Realization(LARGE_MODEL, "bf16", "cuda", 3)

#: The class name of a fit's pre-mean gradient against the reference run
#: (``fit.GRADIENT``); a recorded document with this class is a fit, held to
#: the §7 runtime check, and its block carries ``gradient_agreement``.
GRADIENT_CLASS = "gradient"

#: The per-rank loader rule a document is held to: ``(geometry text, the
#: ranks' load reports) → problems``.
LoadRule = Callable[[str, Sequence[Mapping[str, Any]]], list[str]]


@dataclasses.dataclass(frozen=True)
class Document:
    """One document of the record: how to author it for a realization, how
    to measure a parallel run of it against the reference run, which
    geometries are held exact and which banded, the classes its measure
    returns, and the extra ``causalab run`` arguments it needs. The
    reference run is world 1 — or, for a model no card holds, the run of
    the **oracle** geometry (`oracle`), one of the exact geometries:
    every other exact geometry is then byte-identical to it, which is the
    pairwise identity of the exact set, and the banded ones are measured
    against it."""

    name: str
    exact: tuple[str, ...]
    banded: tuple[str, ...]
    author: Callable[[Path, Realization], Path]
    describe: Callable[[Realization], dict[str, Any]]
    measure: Callable[[Path, Path, Realization], Measured]
    classes: frozenset[str]
    argv: tuple[str, ...] = ()
    #: whether the runs go through `.recorder` — every rank's peak
    #: device memory at exit, and a fit's gradients
    recorded: bool = False
    #: the document's own realization, when it is not the record's
    #: (`A3B`): the ``das_dense`` fit's dense model in fp32, the large
    #: model's documents. Written into its block of the record and compared
    #: on replay.
    realization: Realization | None = None
    #: the geometry whose run stands where world 1 would (class docstring);
    #: ``None`` for a document with a world-1 run. Must be one of ``exact``.
    oracle: str | None = None
    #: the loader rule the ranks' reports are held to: `load_problems`
    #: (the two-rank record's ``1 / world``) unless the document names the
    #: several-axes rule (``tests/golden/_parallel_worlds.load_problems``)
    load_rule: LoadRule | None = None
    #: the pre-flight's estimate per geometry, ``geometry text → {"rank0":
    #: {"resident": bytes, "footprint": bytes}, …}`` off the checkpoint's
    #: header census (``large.estimate``); when set, every recorded run's
    #: peaks are held to it (`memory_problems`) and it is written
    #: into the block as ``estimate``
    estimate: Callable[[str], dict[str, dict[str, int]]] | None = None

    def __post_init__(self) -> None:
        if self.oracle is not None and self.oracle not in self.exact:
            raise ValueError(
                f"document {self.name!r}: the oracle {self.oracle!r} must be one of "
                f"its exact geometries {list(self.exact)}"
            )

    @property
    def geometries(self) -> tuple[str, ...]:
        return self.exact + self.banded

    @property
    def compared(self) -> tuple[str, ...]:
        """The geometries measured against the reference: every geometry but
        the oracle, which is the reference."""
        return tuple(g for g in self.geometries if g != self.oracle)

    @property
    def trains(self) -> bool:
        """Whether the document is a fit (it measures a gradient class)."""
        return GRADIENT_CLASS in self.classes

    def loader_problems(
        self, geometry: str, reports: Sequence[Mapping[str, Any]]
    ) -> list[str]:
        rule = self.load_rule if self.load_rule is not None else load_problems
        return rule(geometry, reports)


def world_of(geometry: str) -> int:
    """The world a ``--parallel`` text spells (``dp=2:rows`` → 2)."""
    return parse_geometry(geometry).world


def realization_of(document: Document, default: Realization = A3B) -> Realization:
    """The realization ``document`` runs on: its own when it names one,
    else ``default`` — the record's."""
    return document.realization if document.realization is not None else default


class RunFailed(RuntimeError):
    """A ``causalab run`` subprocess exited non-zero; its stderr is the message."""

    def __init__(self, argv: Sequence[str], code: int, stderr: str) -> None:
        self.argv = tuple(argv)
        self.code = code
        super().__init__(f"causalab {' '.join(argv)} exited {code}:\n{stderr[-4000:]}")


def out_name(geometry: str) -> str:
    """``dp=2:rows`` → ``dp2-rows``: the run's directory under the root."""
    return geometry.replace("=", "").replace(":", "-").replace(",", "_")


def run(
    document: Path,
    out: Path,
    *extra: str,
    realization: Realization = A3B,
    argv: Sequence[str] = (),
    recorded: bool = False,
) -> Path:
    """``causalab run`` in a subprocess — the world-1 oracle with no
    ``extra``, a geometry with ``--parallel <axes>`` — its load reports
    under ``out / REPORTS`` and, when ``recorded``, its gradients and memory
    under ``out / GRADIENTS``. A run whose receipt exists is complete and is
    not repeated (a kept root resumes; a 70 GB load per geometry is not
    re-run to re-measure it). Returns ``out``.

    Raises:
        RunFailed: the run exited non-zero.
    """
    if (out / RECEIPT).exists():
        return out
    arguments = [
        "run",
        str(document),
        "--engine",
        "pytorch_hooks",
        "--record",
        "--data-root",
        str(DATA),
        "--artifacts-root",
        str(document.parent),
        "--out",
        str(out),
        "--device",
        realization.device,
        *argv,
        *extra,
    ]
    env = {
        **os.environ,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "CAUSALAB_LOAD_REPORT_DIR": str(out / REPORTS),
    }
    entry = "causalab.cli"
    if recorded:
        entry = recorder.__name__
        env[recorder.GRADIENTS_VARIABLE] = str(out / GRADIENTS)
        env[recorder.MEMORY_TRACE_VARIABLE] = str(out / TRACES)
        env[GRADIENT_AGREEMENT_VARIABLE] = gradient_agreement()
    completed = subprocess.run(
        [sys.executable, "-m", entry, *arguments],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        # a receipt a publishing rank wrote before another rank died must
        # not make a kept root resume the run as complete.
        (out / RECEIPT).unlink(missing_ok=True)
        raise RunFailed(arguments, completed.returncode, completed.stderr)
    return out


def run_all(
    root: Path, document: Document, realization: Realization | None = None
) -> dict[str, Path]:
    """The oracle and every geometry of ``document``, one after another:
    ``{"solo": out, "pp=2": out, …}`` under ``root / document.name``, on
    ``realization`` — by default the document's own, else the record's
    (`realization_of`), so a document that names its model (the dense
    fp32 ``das_dense``) is never authored on the A3B by a caller that
    forgot to say which."""
    if realization is None:
        realization = realization_of(document)
    base = root / document.name
    base.mkdir(parents=True, exist_ok=True)
    authored = document.author(base, realization)
    kwargs: dict[str, Any] = {
        "realization": realization,
        "argv": document.argv,
        "recorded": document.recorded,
    }
    outputs: dict[str, Path] = {}
    if document.oracle is None:
        outputs["solo"] = run(authored, base / "solo", **kwargs)
    for geometry in document.geometries:
        outputs[geometry] = run(
            authored, base / out_name(geometry), "--parallel", geometry, **kwargs
        )
    if document.oracle is not None:
        # the reference is the oracle geometry's run (Document docstring):
        # "solo" names the reference for every measure and comparison
        outputs["solo"] = outputs[document.oracle]
    return outputs


# --------------------------------------------------------------------------- #
# the exact comparisons
# --------------------------------------------------------------------------- #


def receipt(out: Path) -> dict[str, Any]:
    return json.loads((out / RECEIPT).read_text())


def parallel_block(geometry: str, launcher: str = "spawned") -> dict[str, Any]:
    """The receipt's ``execution.parallel`` for a geometry launched by
    ``launcher`` (the CLI's spawn; ``joined`` under ``torchrun``)."""
    return parallel_record(parse_geometry(geometry), launcher)


def receipts_agree(
    solo: Path,
    parallel: Path,
    geometry: str,
    *,
    reference_geometry: str | None = None,
    launcher: str = "spawned",
) -> list[str]:
    """How the parallel receipt disagrees with the reference's, beyond
    ``execution.parallel`` — which must be the geometry's block under
    ``launcher`` on the parallel side, and on the reference side the
    world-1 block (``launcher: solo``) or, for a document with an oracle,
    the block of ``reference_geometry`` (a spawned run)."""
    a, b = receipt(solo), receipt(parallel)
    problems: list[str] = []
    if b["execution"]["parallel"] != parallel_block(geometry, launcher):
        problems.append(
            f"execution.parallel is {b['execution']['parallel']!r}, not "
            f"{parallel_block(geometry, launcher)!r}"
        )
    if reference_geometry is None:
        if a["execution"]["parallel"]["launcher"] != "solo":
            problems.append("the oracle was not a solo run")
    elif a["execution"]["parallel"] != parallel_block(reference_geometry):
        problems.append(
            f"the reference's execution.parallel is {a['execution']['parallel']!r}, "
            f"not the oracle's {parallel_block(reference_geometry)!r}"
        )
    del a["execution"]["parallel"], b["execution"]["parallel"]
    if json.dumps(a, sort_keys=True) != json.dumps(b, sort_keys=True):
        problems.append("the receipts differ beyond execution.parallel")
    return problems


def output_files(out: Path) -> list[str]:
    """The run's own outputs: every table and tensor file but the receipt
    and the event stream (timestamps), the harness's subdirectories aside."""
    return sorted(
        p.name
        for p in out.iterdir()
        if p.is_file()
        and p.suffix in (".json", ".safetensors")
        and p.name not in (RECEIPT, EVENTS)
    )


def exact_differences(solo: Path, parallel: Path) -> list[str]:
    """Every output file of ``parallel`` that is not ``solo``'s to the byte
    (§8: ``dp`` over points and ``pp`` move no reduction)."""
    names = output_files(solo)
    theirs = output_files(parallel)
    if theirs != names:
        return [
            "the two runs wrote different files: "
            f"missing {sorted(set(names) - set(theirs))}, "
            f"extra {sorted(set(theirs) - set(names))}"
        ]
    return [
        name
        for name in names
        if (parallel / name).read_bytes() != (solo / name).read_bytes()
    ]


# --------------------------------------------------------------------------- #
# the loader's 1 / world
# --------------------------------------------------------------------------- #


def load_reports(out: Path, world: int = WORLD) -> list[dict[str, Any]]:
    """Every rank's report of a geometry's run, in rank order."""
    from causalab.neural.engines.pytorch_hooks.loading import load_report_path

    return [
        json.loads(load_report_path(out / REPORTS, rank).read_text())
        for rank in range(world)
    ]


def memory_problems(
    estimates: Mapping[str, Mapping[str, int]],
    peaks: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    """How a geometry's measured peaks fall short of the pre-flight's
    estimate (§11): per rank, the **reserved** peak (the allocator's pool,
    what the card sees) is within the estimated footprint — the estimate is
    a bound on the run — and the **allocated** peak is at least the resident
    weights — the rank held what it read. ``estimates`` is
    ``{"rank0": {"resident": bytes, "footprint": bytes}, …}`` (the census's
    arithmetic, ``large.estimate``), ``peaks`` the recorder's
    ``{"rank0": {"peak_bytes_allocated", "peak_bytes_reserved", …}}``. A
    rank in one table and not the other, or a rank that recorded no peak
    (no CUDA), is named."""
    problems: list[str] = []
    if set(estimates) != set(peaks):
        return [
            f"estimated ranks {sorted(estimates)} but measured {sorted(peaks)}",
        ]
    for name in sorted(estimates):
        estimate, peak = estimates[name], peaks[name]
        allocated, reserved = (
            peak.get("peak_bytes_allocated"),
            peak.get("peak_bytes_reserved"),
        )
        if allocated is None or reserved is None:
            problems.append(f"{name}: recorded no device peak ({peak.get('device')!r})")
            continue
        if reserved > estimate["footprint"]:
            problems.append(
                f"{name}: reserved {reserved} bytes at the peak, above the estimated "
                f"footprint {estimate['footprint']}: the pre-flight's bound does not hold"
            )
        if allocated < estimate["resident"]:
            problems.append(
                f"{name}: allocated {allocated} bytes at the peak, below the resident "
                f"weights {estimate['resident']}: the rank did not hold what it read"
            )
    return problems


def memory_replay_problems(
    recorded: Mapping[str, Mapping[str, Any]],
    fresh: Mapping[str, Mapping[str, Any]],
    *,
    tolerance: float = 0.05,
) -> list[str]:
    """How a replay's per-rank **allocated** peaks drift from the record's:
    beyond ``tolerance`` of the recorded peak on any rank (the same
    document, the same code path, the same weights: the allocated peak is
    reproducible to the allocator's rounding, and a drift past 5 % is a
    change in what the run holds)."""
    problems: list[str] = []
    if set(recorded) != set(fresh):
        return [f"recorded ranks {sorted(recorded)} but measured {sorted(fresh)}"]
    for name in sorted(recorded):
        was, now = (
            recorded[name].get("peak_bytes_allocated"),
            fresh[name].get("peak_bytes_allocated"),
        )
        if was is None or now is None:
            problems.append(
                f"{name}: a peak is missing (recorded {was!r}, fresh {now!r})"
            )
        elif abs(now - was) > tolerance * was:
            problems.append(
                f"{name}: allocated peak {now} against the recorded {was} "
                f"({(now - was) / was:+.1%}, tolerance {tolerance:.0%})"
            )
    return problems


def estimate_slack(
    estimates: Mapping[str, Mapping[str, int]], peaks: Mapping[str, Mapping[str, Any]]
) -> dict[str, float]:
    """Per rank, how far the estimated footprint sits above the reserved
    peak, as a fraction of the peak (``footprint / reserved − 1``) — the
    §11 "pre-flight against measured" number, pinned in the record for the
    reader; not a rule."""
    return {
        name: estimates[name]["footprint"] / peaks[name]["peak_bytes_reserved"] - 1.0
        for name in sorted(estimates)
        if peaks.get(name, {}).get("peak_bytes_reserved")
    }


def traces(out: Path, world: int = WORLD) -> list[list[Any]]:
    """Every rank's memory trace of a recorded run, in rank order
    ([`read_trace`][causalab.neural.shared.parallel.soak.read_trace])."""
    from causalab.neural.shared.parallel.soak import read_trace

    return [
        read_trace(recorder.trace_path(out / TRACES, rank)) for rank in range(world)
    ]


def sharded_parameters(record: Mapping[str, Any]) -> set[str]:
    requested, on_disk = record["bytes_requested"], record["bytes_on_disk"]
    return {name for name in requested if requested[name] != on_disk[name]}


def load_problems(geometry: str, reports: Sequence[Mapping[str, Any]]) -> list[str]:
    """How a geometry's load reports fall short of §5.3: under ``tp`` / ``ep``
    every sharded parameter is read at exactly ``1 / world`` of its bytes
    and every other one whole, the sharded set non-empty and the same on
    every rank; under ``pp`` every stage reads its own parameters whole and
    nothing else, the stages disjoint; under ``dp`` (over points or rows)
    a replica reads the whole model."""
    axis = geometry.split("=")[0]
    problems: list[str] = []
    world = len(reports)
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
        # one copy (residency.py): resident elements equal the plan's, the
        # device holds and reserves no more than the parameters
        problems.extend(f"rank {rank}: {p}" for p in residency_problems(record))
        for name in sharded_parameters(record):
            if on_disk[name] % world or requested[name] != on_disk[name] // world:
                problems.append(
                    f"rank {rank}: {name} requested {requested[name]} of "
                    f"{on_disk[name]} bytes, not 1/{world}"
                )
    sharded = [sharded_parameters(r) for r in reports]
    names = [set(r["bytes_requested"]) for r in reports]
    if axis in ("tp", "ep"):
        if not sharded[0]:
            problems.append("no parameter is sharded")
        if any(s != sharded[0] for s in sharded):
            problems.append("the ranks shard different parameters")
        if any(s == n for s, n in zip(sharded, names)):
            problems.append("every parameter is sharded; none is replicated")
        if any(n != names[0] for n in names):
            problems.append("the ranks name different parameters")
    else:
        if any(sharded):
            problems.append(
                f"{geometry} shards a parameter: {sorted(set().union(*sharded))[:3]}"
            )
        if axis == "pp":
            if set.intersection(*names):
                problems.append("two stages read the same parameter")
        elif any(n != names[0] for n in names):
            problems.append("the replicas name different parameters")
    return problems
