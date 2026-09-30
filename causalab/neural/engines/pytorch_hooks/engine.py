"""Run compiled interventions through native PyTorch hooks.

Registry rows declare the engine's components and capabilities, including
training, generation, quantized weights, and writable attention probabilities.
This module supplies the loader, executor, and train loop to the shared
execution driver, which assembles metrics and outputs.
"""

from __future__ import annotations

import dataclasses
import functools
from pathlib import Path
from typing import Any, Mapping, Sequence

from causalab.neural.engines.pytorch_hooks.executor import Interning, PointExecutor
from causalab.neural.engines.pytorch_hooks.cuda_graphs import (
    GraphExecutor,
    GraphPool,
    make_executor,
)
from causalab.neural.engines.pytorch_hooks.graph_reuse import FitGraphCache
from causalab.neural.engines.pytorch_hooks.loading import (
    ModelBundle,
    Sharding,
    load_model,
)
from causalab.neural.shared.execution import execute_request
from causalab.neural.engines.pytorch_hooks.rows import RowSplit
from causalab.neural.shared.parallel.collective import SOLO, Collective, TorchCollective
from causalab.neural.shared.parallel.serving import (
    check_collective,
    process_mesh,
)
from causalab.protocol.positions.resolve import StepResolution
from causalab.io.tensor_files import load_table, load_tensors
from causalab.protocol.rules.capability import check_caller_bundle
from causalab.protocol.positions.roles import resolve_roles
from causalab.protocol.schema.explicit import canonical_model
from causalab.protocol.results import example_labels
from causalab.protocol.compiled import CompiledProtocol
from causalab.protocol.engine import (
    CONTINUATIONS_FILE,
    Engine,
    RunContext,
    RunResult,
    StepRecord,
)
from causalab.io.tables import write_table
from causalab.protocol.parallel import ONE, ParallelGeometry, format_geometry
from causalab.protocol.publish import SOLO as SOLO_PUBLISHER
from causalab.protocol.publish import Publisher
from causalab.protocol.registry import components_served_by, declared_capabilities
from causalab.protocol.rules.errors import ParseError
from causalab.protocol.schema import Document

__all__ = ["PytorchHooksEngine"]


class PytorchHooksEngine(Engine):
    name = "pytorch_hooks"
    # The engine-level verbs are a row of the capability registry
    # (`registry.ENGINE_VERBS`), plus the write verbs the rows charge for the
    # components this engine serves — not a literal here.
    capabilities = declared_capabilities("pytorch_hooks")
    # Which components this engine serves is a row in the capability registry
    # (`registry.CAPABILITIES`, the `reads` cell), not a literal here: the
    # module-boundary and attention-interface vocabulary, writes included, and
    # the routed-expert interior this engine reaches by wrapping the grouped
    # experts dispatch. What the rows leave to the nnsight engine:
    # `expert_permutation` (the serving kernel's own bookkeeping, a `.source`
    # line with no dispatch-slot face) and the Gated DeltaNet interior —
    # tensors inside a fused forward where no hook can reach. Read-only /
    # swap-only components and stream constraints are *protocol policy* (the
    # rows' `writes` and `stream` cells, applied by the shared executor and
    # `validate`), not capability gaps: declaring router_logits unwritable
    # here would turn "a write here reaches nothing, write router_scores
    # instead" into "try another engine", the wrong answer for every engine —
    # which is why `writable_components` is the same set.
    components = components_served_by("pytorch_hooks")
    writable_components = components
    is_local = True

    def __init__(
        self,
        *,
        device: str = "cpu",
        cuda_graphs: bool = False,
        batch_rows: int | None = None,
        fit_rows: int | None = None,
        bundle: ModelBundle | None = None,
        parallel: ParallelGeometry = ONE,
        collective: Collective = SOLO,
        sharding: Sharding | None = None,
    ) -> None:
        """``device`` places the model — one device (``cpu``, ``cuda``,
        ``cuda:1``, ``mps``) or a comma list (``cuda:0,cuda:1``) spreading
        the layers across the devices of this process, embedding first and
        head last ([`DeviceMap`][causalab.neural.shared.devices.DeviceMap]; a
        memory lever, not a speed-up); ``batch_rows`` bounds how many rows
        one no-grad forward covers; ``fit_rows`` bounds how many rows one
        **grad** forward of a fit covers; ``bundle`` is a caller-owned model
        to run instead of loading one; ``parallel`` is the geometry this
        engine runs under (``docs/model_parallelism.md`` §2) — all ones, the
        default, is today's path; a world above 1 runs over ``collective``,
        the one seam through which the engine talks to other ranks (§3):
        [`SOLO`][], the
        default, is world 1, and at ``world > 1`` [`execute`][] builds the
        ``torch.distributed`` collective over the process's one mesh — the
        launcher's, riding on the request's publisher — when none is handed
        in, refusing by name, before any weights load, where no group is
        initialised (every axis is served: ``pipeline`` stages in
        ``stages.py``, ``context`` chunks in ``parallel/context.py``).
        ``sharding`` is this
        rank's place in that geometry —
        its rank and the meshes of its groups (``sharding.py``) — handed to
        [`load_model`][] so the registry's plan is applied and each rank
        reads its own shard of the weights (``docs/model_parallelism.md``
        §5.2–5.3); its geometry must be ``parallel``, and when none is
        handed in it is derived from the same mesh as the collective
        (``Sharding.from_mesh``), so the hooks gather over exactly the
        groups the plan was applied over. A collective handed in without a
        mesh and without a sharding is refused by name: the loader cannot
        shard without the meshes. Execution, never identity: the run receipt
        (``execution.parallel``) is its one recorder.

        The document's ``model.attn_implementation`` selects the attention
        backend. Attention-interior forwards temporarily use eager and restore
        the selection on every exit path.

        Placement and geometry are execution (the engine's call, §8);
        precision is not — dtype and quantization come from each point's own
        ``model`` section. With ``batch_rows`` set, a forward group over more
        rows runs as several forwards over row windows whose captures are
        concatenated in row order (executor.py); the numbers equal the
        single-forward run up to dtype rounding, and nothing about it enters
        a digest or a stamp. It bounds every no-grad forward — document runs
        and ``train.eval`` passes — but not a training minibatch:
        ``train.batch.pairs`` is the document's own batching knob for grad
        forwards, so the execution bound applies to the no-grad passes and to
        ``train.eval``.

        ``fit_rows`` is the grad-forward counterpart: the points of a swept
        campaign that declare ``train`` are fitted together as a cohort, one
        forward per optimizer step over the concatenation of every member's
        minibatch, and ``fit_rows`` bounds how many rows that forward covers
        — the members are packed into forwards under the bound, a member's
        own minibatch (its ``train.batch.pairs`` rows) is never split, and
        ``None`` measures the bound on the cohort's first step from the
        device's free memory (``budget.py``; unbounded off CUDA), reporting
        it as ``execution.fit_rows_resolved``. Like ``batch_rows`` it
        is execution, never identity: nothing about it enters a canonical
        form, a digest or a stamp, and the run receipt (``execution.fit_rows``)
        is its one recorder. Both bounds are this engine's defaults; a
        request's own ``execution`` block overrides either for that request
        ([`effective_batch_rows`][], [`effective_fit_rows`][]).

        With ``bundle`` set (built by [`ModelBundle.from_model`][], spec §9),
        the executor runs that model and [`load_model`][] is never called:
        the engine never loads, moves, frees or changes the training mode of
        a caller-owned model,
        and every hook it installs is removed on every exit path. Before any
        forward the document's canonical ``model`` realization is checked
        against the bundle's ``key`` / ``revision`` / ``dtype`` /
        ``quantization`` and ``device`` against this argument
        ([`check_caller_bundle`][]); a
        disagreement refuses. The run receipt says which way the model came
        in ([`model_source`][]), and nothing else does.

        Raises:
            ValueError: ``batch_rows`` or ``fit_rows`` is not a positive row
                count. This is the one runtime check; the CLI's argparse type
                refuses the same values before they reach here, and the
                executor and the train loop trust what the engine hands them.
        """
        self.device = device
        self.cuda_graphs = cuda_graphs
        self.batch_rows = _row_bound("batch_rows", batch_rows)
        self.fit_rows = _row_bound("fit_rows", fit_rows)
        self.bundle = bundle
        self.parallel = parallel
        self.collective = collective
        if sharding is not None and sharding.geometry != parallel:
            raise ValueError(
                f"sharding is for geometry {format_geometry(sharding.geometry)} "
                f"but the engine runs under {format_geometry(parallel)}"
            )
        self.sharding = sharding
        #: the allocator pool every graph this engine captures is captured
        #: into (``cuda_graphs.GraphPool``): opened on the first capture and
        #: kept for the engine's lifetime, so the segments one fit's graphs
        #: grew serve every later fit and request instead of being returned
        #: to the device and re-allocated per fit; a fit's graphs themselves
        #: still belong to their request. Released when the engine is.
        self._graph_pool: GraphPool | None = None

    @property
    def model_source(self) -> str:
        """``"caller"`` when this engine runs a bundle handed to it, else
        ``"loaded"`` — execution provenance for the run receipt (§8), never
        part of a canonical form, a digest or a stamp."""
        return "caller" if self.bundle is not None else "loaded"

    def effective_batch_rows(self, run: RunContext) -> int | None:
        """The no-grad row bound this ``run`` runs under: its own
        ``execution.batch_rows`` when the key is present — ``None`` there
        meaning unbounded for this run — else this engine's.

        Raises:
            ValueError: the override is not a positive row count.
        """
        return _row_bound(
            "batch_rows", run.execution.get("batch_rows", self.batch_rows)
        )

    def effective_fit_rows(self, run: RunContext) -> int | None:
        """The grad-forward row bound this ``run``'s fits run under: its own
        ``execution.fit_rows`` when the key is present — ``None`` there
        meaning every cohort member in one forward — else this engine's.

        Raises:
            ValueError: the override is not a positive row count.
        """
        return _row_bound("fit_rows", run.execution.get("fit_rows", self.fit_rows))

    # ------------------------------------------------------------------ #

    def execute(self, compiled: CompiledProtocol, run: RunContext) -> RunResult:
        from causalab.neural.engines.pytorch_hooks.train import run_cohort_training

        collective, sharding = self.collective, self.sharding
        # docs/model_parallelism.md §3: at world > 1 the collective is the
        # one handed in, or the torch.distributed one over the process's one
        # mesh, and its group sizes are the geometry's (`_world`, below)
        # a run carrying no publisher at all is world 1 (`SOLO`): the
        # boundary tests hand `execute` a bare stand-in and never reach here
        publisher: Publisher = getattr(run, "publisher", SOLO_PUBLISHER)
        if publisher.replicas != self.parallel.data:
            # a data-parallel geometry needs the launched world its replicas
            # run in (§3): a run carrying no launch — a Python caller, or
            # the workflow runner, which hands one engine to one process
            # — would run every point on one device and call it dp
            raise ParseError(
                "P4",
                f"--parallel {format_geometry(self.parallel)} asks for "
                f"{self.parallel.data} data-parallel replicas, but this run "
                f"was launched into {publisher.replicas} "
                f"({publisher.launcher}) — run the document through "
                "`causalab run`, which spawns the replicas or joins a torchrun "
                "group; a workflow shards a step with fan_out.over.shards instead",
                path="--parallel.data",
            )
        if self.parallel.world > 1:
            collective, sharding = self._world(publisher, collective, sharding)
        # one executor per point, kept so a decoding run's continuations
        # can be published after the campaign ran (workflow spec §2.7)
        executors: list[PointExecutor] = []

        def factory(
            doc: Document,
            ctx: RunContext,
            coords: Mapping[str, Any],
            interning: Interning | None,
            resolution: StepResolution | None,
        ) -> PointExecutor:
            executor = self._executor(
                doc,
                ctx,
                coords=coords,
                interning=interning,
                resolution=resolution,
                collective=collective,
                sharding=sharding,
            )
            executors.append(executor)
            return executor

        graph_pool = self.graph_pool() if self.cuda_graphs else None
        graph_cache = FitGraphCache(pool=graph_pool) if self.cuda_graphs else None
        try:
            result = execute_request(
                compiled,
                run,
                engine_name=self.name,
                executor_factory=factory,
                train_runner=functools.partial(
                    run_cohort_training,
                    fit_rows=self.effective_fit_rows(run),
                    graph_cache=graph_cache,
                    graph_pool=graph_pool,
                ),
                # this engine's executor consults the shared ForwardCache, so it
                # claims §3's cross-point interning and can report what it paid
                intern_forwards=True,
                engine=self,
            )
        finally:
            if graph_cache is not None:
                graph_cache.close()
            for executor in executors:
                if isinstance(executor, GraphExecutor):
                    executor.close()
        if run.decoding is None:
            return result
        # a run that declared its decoding gets the continuation as an
        # output — run-keyed, not a `save` kind, so no document digest
        # moves (workflow spec §2.7)
        path = _write_continuations(run, executors, steps=result.steps)
        if path is None:
            return result
        return dataclasses.replace(
            result, files={**result.files, CONTINUATIONS_FILE: path}
        )

    def graph_pool(self) -> GraphPool:
        """The engine's [`GraphPool`][] (``_graph_pool``), opened on
        first use; one closed by an out-of-memory fallback — it hands out no
        more handles — is replaced, so the next run captures into a
        shared pool again rather than into private pools per graph."""
        if self._graph_pool is None or self._graph_pool.closed:
            self._graph_pool = GraphPool()
        return self._graph_pool

    # ------------------------------------------------------------------ #

    def _world(
        self,
        publisher: Publisher,
        collective: Collective,
        sharding: Sharding | None,
    ) -> tuple[Collective, Sharding | None]:
        """The collective and the sharding a world above 1 runs over —
        **one mesh per process** (``docs/model_parallelism.md`` §3): with no
        collective handed in, the process's mesh (the launcher's, off the
        publisher) yields the ``torch.distributed`` collective and, unless a
        caller-owned bundle or a sharding came in, this rank's sharding; a
        handed-in collective is held to the geometry's group sizes, and one
        that carries a mesh yields the sharding the same way.

        Raises:
            ParseError: ``P4`` naming ``--parallel`` — no process group, a
                collective of the wrong sizes, or a collective with no mesh
                and no sharding beside it when a model is to be loaded.
        """
        mesh = None
        if collective is SOLO:
            mesh = process_mesh(self.parallel, publisher)
            collective = TorchCollective(mesh)
        elif isinstance(collective, TorchCollective):
            mesh = collective.mesh
        check_collective(self.parallel, collective)
        if self.bundle is not None or sharding is not None:
            return collective, sharding
        if mesh is None:
            raise ParseError(
                "P4",
                f"--parallel {format_geometry(self.parallel)} loads a sharded "
                "model, and the collective handed in carries no mesh to shard "
                "over: hand the engine this rank's Sharding beside it "
                "(Sharding.from_mesh), or let execute build both from the "
                "launcher's process group",
                path="--parallel",
            )
        return collective, Sharding.from_mesh(mesh)

    def _executor(
        self,
        doc: Document,
        run: RunContext,
        *,
        grad_enabled: bool = False,
        coords: Mapping[str, Any] | None = None,
        interning: Interning | None = None,
        resolution: StepResolution | None = None,
        collective: Collective = SOLO,
        sharding: Sharding | None = None,
    ) -> PointExecutor:
        realization = canonical_model(doc.raw["model"])
        if self.bundle is not None:
            # a caller-owned model: checked against the document and the
            # geometry, never loaded, never inserted into load_model's cache
            check_caller_bundle(
                self.bundle, realization, device=self.device, geometry=self.parallel
            )
            bundle = self.bundle
        else:
            bundle = load_model(
                str(doc.model.key),
                str(doc.model.revision),
                dtype=str(realization["dtype"]),
                device=self.device,
                quantization=realization.get("quantization"),
                # the loader is memoized on its bound arguments, and a test
                # hooks the world-1 bundle by asking for it the way this call
                # does — so the keyword is added only when there is a sharding
                **({"sharding": sharding} if sharding is not None else {}),
                **(
                    {"attn_implementation": realization["attn_implementation"]}
                    if "attn_implementation" in realization
                    else {}
                ),
            )
        role_rows, role_fields = resolve_roles(doc, run.env)
        executor = make_executor(
            doc,
            bundle,
            cuda_graphs=self.cuda_graphs,
            decoding=run.decoding,
            collective=collective,
            role_rows=role_rows,
            role_fields=role_fields,
            load_tensors=functools.partial(load_tensors, run),
            load_table=functools.partial(load_table, run),
            grad_enabled=grad_enabled,
            coords=coords,
            interning=interning,
            batch_rows=self.effective_batch_rows(run),
            resolved=resolution.positions if resolution is not None else None,
        )
        # the run's decode spec rides on the executor (None = the argmax)
        executor.decoding = run.decoding
        if self.parallel.data_mode == "rows":
            # data parallelism over rows (docs/model_parallelism.md §8.3):
            # the fit splits every minibatch across the replicas through this
            executor.rows = RowSplit(collective, self.parallel)
        return executor


def _write_continuations(
    run: RunContext,
    executors: Sequence[PointExecutor],
    *,
    steps: Sequence[StepRecord],
) -> Path | None:
    """``continuations.json`` under the run's output directory: one row
    per generated row of every decoding group of every point — ``point``
    (the canonical index), ``point_digest``, ``model``, ``input``,
    ``example``, ``steps`` (the budget), ``width`` (tokens before the first
    EOS), ``truncated`` (``width == steps``), the real ``token_ids``, the
    decoded ``text`` and each token's ``[start, end)`` char span in it — the
    [`Continuation`][causalab.neural.shared.generated.Continuation] as a table.
    ``steps`` are the executed points' signed records, in the order the
    executors were built.

    Through the run's publisher like every other writer
    (``docs/model_parallelism.md`` §3): a rank that does not publish writes
    nothing; a publishing rank hands its rows to the joiner; the joiner
    writes every replica's rows in campaign point order. ``None`` when this
    process wrote nothing."""
    publisher = run.publisher
    if not publisher.publish:
        return None
    rows: list[dict[str, Any]] = []
    for index, executor in enumerate(executors):
        digest = steps[index].digest if index < len(steps) else None
        point = steps[index].index if index < len(steps) else index
        for (model, input_role), continuation in sorted(
            executor.continuations().items()
        ):
            steps = continuation.steps
            input_ids = executor.input_token_ids(input_role)
            eos_ids = list(executor.eos_token_ids())
            labels = example_labels(executor.role_rows[input_role])
            for row, width in enumerate(continuation.widths):
                text = continuation.texts[row] if row < len(continuation.texts) else ""
                offsets = (
                    continuation.offsets[row] if row < len(continuation.offsets) else ()
                )
                rows.append(
                    {
                        "point": point,
                        "point_digest": digest,
                        "model": model,
                        "input": input_role,
                        "example_id": labels[row],
                        "split": executor.role_rows[input_role][row].get("split"),
                        "input_ids": input_ids[row],
                        "steps": steps,
                        "width": int(width),
                        "truncated": int(width) >= steps,
                        "token_ids": continuation.real_ids(row),
                        "greedy_token_id": (
                            int(continuation.token_ids[row, 0])
                            if (run.decoding or {}).get("mode", "deterministic")
                            == "deterministic"
                            else None
                        ),
                        "emitted_ids": [
                            int(t)
                            for t in continuation.token_ids[
                                row, : int(width) + (int(width) < steps)
                            ]
                        ],
                        "terminal_eos_id": (
                            int(continuation.token_ids[row, width])
                            if int(width) < steps
                            else None
                        ),
                        "padding_ids": [
                            int(t)
                            for t in continuation.token_ids[
                                row, int(width) + (int(width) < steps) :
                            ]
                        ],
                        "stop_reason": "eos" if int(width) < steps else "length",
                        "eos_token_ids": eos_ids,
                        "decoding": dict(run.decoding or {"mode": "deterministic"}),
                        "text": text,
                        "offsets": [list(span) for span in offsets[: int(width)]],
                    }
                )
    gathered = publisher.gather(rows)
    if gathered is None:
        return None
    # every replica's rows, in campaign point order — the shards arrive in
    # replica order and each names its points, so a stable sort by campaign
    # index is the join
    joined = sorted(
        (row for shard in gathered for row in shard), key=lambda row: row["point"]
    )
    target = run.output_dir / CONTINUATIONS_FILE
    write_table(target, joined)
    return target


def _row_bound(name: str, value: Any) -> int | None:
    """``value`` as a row bound: a positive ``int`` (never a ``bool``) or
    ``None`` for unbounded; anything else is a `ValueError` naming
    ``name``, whether it came from the constructor or a run's
    ``execution`` block."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive row count, got {value!r}")
    return value
