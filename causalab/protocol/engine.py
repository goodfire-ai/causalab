"""Define the execution engine interface and run context.

An engine implements ``execute(compiled, run)`` and declares its capabilities.
``RunContext`` supplies the output directory, resolution environment, selected
points, execution bounds, decoding settings, and event sink. Results use the
shared ``RunResult`` format.

The interface can be imported before an execution engine or tensor library loads."""

from __future__ import annotations

import abc
import dataclasses
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

from causalab.protocol.publish import SOLO, Publisher
from causalab.protocol.registry.engines import component_capability
from causalab.protocol.results import Denominator, Resolution
from causalab.protocol.rules.capability import (
    CAPABILITIES,
    requires,
    requires_campaign,
    train_capabilities,
)
from causalab.protocol.rules.errors import ProtocolError

if TYPE_CHECKING:
    from causalab.io.env import ResolutionEnv
    from causalab.protocol.compiled import CompiledProtocol

__all__ = [
    "Engine",
    "CAPABILITIES",
    "CONTINUATIONS_FILE",
    "RunContext",
    "RunResult",
    "StepRecord",
    "check_steps_signed",
    "component_capability",
    "requires",
    "requires_campaign",
    "train_capabilities",
]

#: The run-keyed engine output a decode writes when
#: [`RunContext.decoding`][] is set: one row per generated row of every
#: decoding group — ``point``, ``point_digest``, ``model``, ``input``,
#: ``example``, ``steps`` (the budget), ``width``, ``truncated``, the real
#: ``token_ids``, ``text`` and per-token char ``offsets``. Not a ``save`` kind:
#: the document's ``save`` section is unchanged, so no document digest moves.
CONTINUATIONS_FILE = "continuations.json"


@dataclasses.dataclass(frozen=True)
class RunContext:
    """What one run adds to a compiled document: where outputs land, the
    resolution environment, which points this shard covers and what the
    caller authored about execution. Everything an engine reads that is
    **not** identity — the identity is the
    [`CompiledProtocol`][] handed beside it,
    and nothing of it is copied here: no digest, no canonical form, no
    per-point tuple.

    ``points`` are the shard's step indices into the engine's enumeration —
    the canonical order [`causalab.neural.shared.sweep.enumerate_steps`][]
    walks (the cross product of ``compiled.axes``, last axis fastest) — in
    run order — ``run_protocol``'s ``--points START:STOP`` range or a
    fanned-out workflow child's selection (workflow spec §2.9) — and ``None``
    is every point. The campaign digest is untouched by a shard, so a shard's
    artifacts still stamp and dedup as members of the whole campaign.

    ``record`` says whether the engine writes the run receipt
    (``protocol.json``) and the event stream (``events.jsonl``) into
    ``output_dir`` before the first forward ([`causalab.neural.shared.receipt`][]). ``False`` is the default everywhere: a run writes its saved
    tables and nothing beside them. ``run_protocol`` sets it when its caller
    asks (``record=True``; the CLI's ``--record``); the workflow doors never
    do, since their receipt is the step's ``_step.json`` and their stream is
    the runner's, beside the manifest and never inside a step directory.

    ``execution`` is the run's own execution parameters — ``batch_rows``
    (rows per no-grad forward) and ``fit_rows`` (rows per grad forward of a
    fit), each a positive integer or ``None`` for unbounded — which override
    the engine's constructor defaults for this run and nothing else; an
    absent key leaves the engine's value in force. A workflow step's
    ``execution`` block arrives here (workflow spec §2.2). It is execution,
    never identity (§8): it enters no canonical form, no digest and no
    stamp, and the run receipt is its one recorder."""

    output_dir: Path
    env: ResolutionEnv
    points: tuple[int, ...] | None = None
    execution: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    #: How the run's ``generated`` frames are decoded — a workflow
    #: ``behavioral`` step's ``decoding`` block (workflow spec §2.7):
    #: ``{"mode": "deterministic"}`` or ``{"mode": "sampled", "seed", "temperature",
    #: "top_p"}``. ``None`` — every other door — is the greedy decode the
    #: document alone specifies, byte for byte what it produced before this
    #: field existed. Set, the engine also writes [`CONTINUATIONS_FILE`][]
    #: into its result. Execution, not identity: the block enters no
    #: canonical document, no document digest and no stamp; the step record
    #: is its one recorder (``execution.decoding``).
    decoding: Mapping[str, Any] | None = None
    #: The event-stream adapter ([`causalab.io.events.EventSink`][]): the
    #: callable handed each line of a document run's ``events.jsonl`` after
    #: its local write. Excluded from equality — two runs of one shard are
    #: the same run whoever is listening.
    sink: Callable[[Mapping[str, Any]], None] | None = dataclasses.field(
        default=None, compare=False
    )
    #: Whether the engine writes ``protocol.json`` and ``events.jsonl`` into
    #: ``output_dir``. ``False``, the default, writes nothing beside the
    #: outputs; ``run_protocol`` sets it only when its caller opts in
    #: (``record=True``; the CLI's ``--record``), and the workflow doors keep
    #: their own records.
    record: bool = False
    #: This process's place in the run (``docs/model_parallelism.md`` §3,
    #: §8.3; [`causalab.protocol.publish`][]): whether its outputs leave
    #: it, which data-parallel replica it is, and the gather the join runs
    #: over. The **one** seam every writer asks — the engine's output
    #: writer, the receipt's amenders, the continuations table — so exactly
    #: one process writes a campaign. [`SOLO`][],
    #: the default, is world 1: this process publishes and its gather is the
    #: identity, today's path to the byte. Under data parallelism over points
    #: the engine derives this replica's contiguous shard of ``points``
    #: ([`point_shard`][causalab.protocol.publish.point_shard]); ``points`` stays the
    #: whole selection on every rank.
    publisher: Publisher = SOLO

    def indices(self, n_points: int) -> tuple[int, ...]:
        """The point indices this run covers, in run order: ``points``, or
        every index below ``n_points`` when the run is not a shard."""
        return tuple(range(n_points)) if self.points is None else self.points


@dataclasses.dataclass(frozen=True)
class StepRecord:
    """One executed step's identity, as the engine signed it: its index into
    the canonical enumeration order, its coordinates (axis id → value, §3)
    and its provenance digest (§7) — the ``points[]`` entry of the run
    receipt, and what a workflow step's record and controls ledger copy."""

    index: int
    coords: Mapping[str, Any]
    digest: str


@dataclasses.dataclass(frozen=True)
class RunResult:
    """What an execution produced: saved files (save-manifest paths →
    absolute paths on disk) and per-point summaries for `explain`-style
    reporting."""

    files: Mapping[str, Path]
    summaries: tuple[Mapping[str, Any], ...] = ()
    #: The steps the engine executed, in run order, each with the digest it
    #: signed it with ([`causalab.neural.shared.sweep`][]) — the provenance
    #: units the receipt's ``points`` block lists and a workflow step record's
    #: ``point_digests`` / ``coords`` copy. Empty from an engine that
    #: executed nothing.
    steps: tuple[StepRecord, ...] = ()
    #: Forward groups the engine actually ran across the whole campaign.
    #: ``num_forwards`` (§4) is what the *plan* derives per point; this is what
    #: execution cost, so the two together say how much of §3's cross-point
    #: interning an engine claimed. An engine that shares nothing reports
    #: points × groups; one that interns fully reports the campaign's distinct
    #: group keys ([`causalab.neural.shared.plan.interned_groups`][]). The
    #: inner passes of a fit are not forward groups and are not counted.
    forwards: int = 0
    #: One [`Resolution`][] per result cell —
    #: every ``save`` entry of every executed point, in run order. An
    #: ``Unavailable`` cell is a legal cell with nothing to measure (a scoped
    #: slice that selected no rows); it is in the result with its reason code
    #: and it is in the denominator. Never an ``Invalid``: a defect stops
    #: validation before anything executes (spec §4.1).
    cells: tuple[Resolution, ...] = ()

    @property
    def denominator(self) -> Denominator:
        """``eligible`` of ``total`` cells, the excluded ones by reason — the
        numbers a summary reads instead of keeping its own books."""
        return Denominator.of(self.cells)


class Engine(abc.ABC):
    """One execution engine, described by data and entered through one
    method. Implementations own the §8 services (SiteResolver, position
    resolution, planning, mechanisms, featurizers, metrics, training, RNG,
    stamping) internally — the seam is the document, not the services."""

    #: Engine name, for refusal messages and ArtifactIdentity stamping.
    name: str = "abstract"
    #: The §8 capability set this engine supports.
    capabilities: frozenset[str] = frozenset()
    #: Components this engine's site resolver serves. The matching
    #: ``component:<name>`` capabilities are generated (never listed in
    #: ``capabilities`` by hand), so the closed vocabulary stays
    #: [`Component`][causalab.protocol.schema.types.Component].
    components: frozenset[str] = frozenset()
    #: The subset of ``components`` this engine can land a write on.
    writable_components: frozenset[str] = frozenset()
    #: Local engines may run ``pytorch_fn`` writes (§2.8).
    is_local: bool = False

    @property
    def effective_capabilities(self) -> frozenset[str]:
        """``capabilities`` plus the generated component entries — what
        the capability check actually compares against [`requires`][]."""
        return (
            self.capabilities
            | {component_capability(c) for c in self.components}
            | {component_capability(c, write=True) for c in self.writable_components}
        )

    @abc.abstractmethod
    def execute(self, compiled: CompiledProtocol, run: RunContext) -> RunResult:
        """Run the points ``run`` selects of ``compiled`` and write everything
        their save manifests name under ``run.output_dir``."""


def check_steps_signed(
    result: RunResult, indices: Sequence[int], engine: Engine
) -> None:
    """Hold an engine to the enumeration half of the contract:
    ``result.steps`` carries one [`StepRecord`][] per index of the
    shard, in run order — what the receipt's ``points`` block and a workflow
    step's record and controls ledger copy. An engine that returned a
    [`RunResult`][] without them would otherwise write an empty record
    silently; every door refuses it here instead, naming the engine."""
    signed = tuple(step.index for step in result.steps)
    if signed != tuple(indices):
        raise ProtocolError(
            "P2",
            f"engine {engine.name!r} signed {len(signed)} step(s) "
            f"{list(signed)} for a run of {len(indices)} point(s) "
            f"{list(indices)} — Engine.execute returns one StepRecord per "
            "selected index, in run order (RunResult.steps)",
        )
