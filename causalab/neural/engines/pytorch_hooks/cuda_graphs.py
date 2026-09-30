"""CUDA replay for supported Qwen3/Qwen3.6 last-token fits.

Captures belong to the execution request. ``GraphPool`` belongs to the
engine and retains allocator segments across fits and requests.
Unsupported documents use PointExecutor. Unexpected capture errors
propagate. See ``docs/cuda_graphs.md`` for support and memory rules.
"""

from __future__ import annotations

# This engine-owned lowering operates on the executor's internal batch buffers.
# pyright: reportPrivateUsage=false

import contextlib
import copy
import gc
import logging
import time
import weakref
from typing import Any, Callable, Iterator

import torch

from causalab.neural.engines.pytorch_hooks.budget import OOMPolicy, abort_distributed
from causalab.neural.engines.pytorch_hooks.executor import (
    PointExecutor,
    _resumed,
    prompt_masks,
)
from causalab.neural.shared.featurizers import Gate, Stage, featurizer_cache
from causalab.neural.shared.head import HEAD_INPUT
from causalab.neural.shared.parallel import heartbeat
from causalab.neural.shared.parallel.collective import SOLO, Collective
from causalab.neural.shared.parallel.deadline import REPLAY
from causalab.neural.shared.parallel.fragments import Fragments
from causalab.neural.shared.parallel.placement import Axis
from causalab.neural.shared.plan import is_unwritten, write_names
from causalab.protocol.positions.roles import input_roles
from causalab.neural.shared.sites import resolve_site
from torch.utils._pytree import tree_map
from causalab.protocol.schema import Document, PositionSpec, ReadRef, operand_reads

_log = logging.getLogger(__name__)


def stage_layout(stage: Stage) -> tuple[Any, ...]:
    """Storage and Python control state that must agree before fit reuse."""
    tensors = dict(stage.named_parameters()) | dict(stage.named_buffers())
    return (
        # Torch creates a distinct ParametrizedSubspace class per instance.
        # Compare its qualified name and the actual parametrization children.
        tuple(
            (name, type(module).__module__, type(module).__qualname__)
            for name, module in stage.named_modules()
        ),
        tuple(
            (
                name,
                value.shape,
                value.dtype,
                value.device,
                value.stride(),
                value.requires_grad,
            )
            for name, value in tensors.items()
        ),
    )


def copy_stage_state(target: Stage, source: Stage) -> None:
    """Copy complete fit state without rebinding captured storage.

    Cayley's seed-dependent base is a buffer, not a parameter. The seed
    attribute itself stays with the captured document's construction metadata.
    Gate temperature is staged by Replay; its temporary capture tensor stays
    owned by that replay as well.
    """
    if target is source:
        return
    if stage_layout(target) != stage_layout(source):
        raise ValueError("incompatible featurizer state for CUDA graph reuse")
    incoming = dict(source.named_parameters()) | dict(source.named_buffers())
    with torch.no_grad():
        for name, value in (
            dict(target.named_parameters()) | dict(target.named_buffers())
        ).items():
            value.copy_(incoming[name])
    target.train(source.training)
    if isinstance(target, Gate) and isinstance(source, Gate):
        target.temperature = source.temperature
        target.hard_eval = source.hard_eval


def copy_executor_stages(target: PointExecutor, source: PointExecutor) -> None:
    for name, stage in target.stage_cache.items():
        copy_stage_state(stage, source.stage(name))


def graph_device(bundle: Any) -> torch.device:
    """The one device a graph executor's bundle is on. A bundle spanning
    devices was refused by [`unsupported_reason`][] before a
    [`GraphExecutor`][] was built, so reaching one here is a broken
    promise, not a user error. (A CPU bundle can reach here: the eligibility
    tests build a [`GraphExecutor`][] by hand and run its eager paths.)"""
    single = bundle.devices.single
    if single is None:
        raise AssertionError(
            "CUDA graphs run on one device; the bundle is placed on "
            f"{bundle.devices.spelling}"
        )
    return single


#: Axes whose groups a captured graph cannot serve: pipeline stages agree on
#: the host after each forward, and context chunks exchange state inside it
#: (docs/cuda_graphs.md "Multi-rank execution").
UNCAPTURED_AXES: tuple[Axis, ...] = ("pipeline", "context")

#: Where a graph holder's out-of-memory failure happened, for the abort
#: under ``OOMPolicy.ABORT`` (`TrainingGraphs`, ``graph_cohort``).
IN_GRAPH = "inside a captured CUDA graph of a distributed model window"


def unsupported_reason(
    doc: Document, bundle: Any, collective: Collective = SOLO
) -> str | None:
    """Conservative eligibility; this is an execution option, not protocol data.
    ``collective`` is the world the executor runs over; a data axis above one
    reads its mode off the geometry the bundle was loaded under."""
    refused = [axis for axis in UNCAPTURED_AXES if collective.size(axis) > 1]
    if refused:
        # the run falls back to the eager executor, naming the axes
        return (
            "CUDA graphs run under tensor, expert and data parallelism; the "
            f"collective spans more than one rank on the {' and '.join(refused)} "
            + ("axes" if len(refused) > 1 else "axis")
        )
    geometry = getattr(bundle, "geometry", None)
    if (
        collective.size("data") > 1
        and getattr(geometry, "data_mode", "points") == "rows"
    ):
        # the eager step weighs each replica's loss by its rows' share of the
        # minibatch before the gradients are summed over the replicas
        # (rows.py); the captured objective takes no such share
        return (
            "CUDA graphs do not serve data parallelism over rows: the captured "
            "training step lacks each replica's share of the loss"
        )
    single = bundle.devices.single
    if single is None:
        # graphs are single-device (docs/model_parallelism.md §11): the run
        # falls back to the eager executor, as for every reason here
        return (
            "CUDA graphs are single-device; the bundle places its layers on "
            f"{bundle.devices.spelling}"
        )
    if single.type != "cuda":
        return "CUDA graphs require a CUDA device"
    if doc.train is not None and doc.train.phases:
        return "phased training requires eager execution"
    if doc.train is not None and (
        doc.train.control or any(entry.kind == "trajectory" for entry in doc.save)
    ):
        return "training controllers and trajectories require eager execution"
    if doc.train is not None and any(
        term.constraint is not None for term in doc.train.objective
    ):
        # §2.11: the dual pair steps on the eager loop; the fit graphs map
        # optimizer parameters onto worker stages by identity and a dual is
        # no stage's parameter — so the point runs eager, as `control` does
        return "Lagrangian constraint duals require eager execution"
    if doc.train is not None and any(
        spec.draw is not None for spec in input_roles(doc).values()
    ):
        # §2.2: a *fit* rebuilds a drawn role's minibatch executors every
        # epoch; the shapes are invariant but the rebuild is what capture
        # cannot follow. An apply document reads the fixed `eval` member at
        # one width and rebuilds nothing — capturable.
        return "a drawn role rebuilds its minibatch executors each epoch"
    if any(agg.spec.kind == "js" for agg in doc.aggregations()):
        return "JS objectives require eager execution"
    if any(agg.spec.kind == "soft_accuracy" for agg in doc.objective_aggregations()):
        return "soft accuracy objectives require eager execution"
    config = bundle.model.config
    if (
        config.model_type not in {"qwen3", "qwen3_5_moe_text"}
        or getattr(config, "use_sliding_window", False)
        or getattr(config, "_attn_implementation", None) != "eager"
        or bundle.quantization is not None
        or bundle.model.training
        or any(p.requires_grad for p in bundle.model.parameters())
    ):
        return "validated model path is frozen, unquantized Qwen3/Qwen3.6 text with eager attention"
    if (
        config.model_type == "qwen3_5_moe_text"
        and getattr(config, "_experts_implementation", None) != "grouped_mm"
    ):
        return "Qwen3.6 capture requires the validated grouped_mm expert backend"
    for entry in (*doc.reads.values(), *doc.writes.values()):
        pos = doc.positions[entry.pos] if isinstance(entry.pos, str) else entry.pos
        if pos != PositionSpec(index=-1) or entry.dims is not None:
            return "only whole-feature last-token prompt reads/writes are supported"
        site = doc.sites[str(entry.site)]
        if isinstance(site.layers, tuple) and len(site.layers) != 1:
            return "multi-layer bands require eager execution"
        if site.component not in {"block_output", "lm_head"} or any(
            value is not None for value in (site.head, site.expert, site.stream)
        ):
            return "only block_output and lm_head taps are supported"
        if isinstance(entry.featurizer, tuple):
            return "featurizer chains are not supported"
    if len(doc.writes) > 1:
        return "only one swap write is supported"
    for write in doc.writes.values():
        # the parser binds a read operand to `ReadRef` (§2.7)
        if write.do.mechanism != "swap" or not operand_reads(doc, write.do):
            return "only a swap from another read is supported"
    if len(doc.featurizers) > 1:
        return "only one featurizer is supported"
    for spec in doc.featurizers.values():
        if spec.kind == "gate":
            if spec.parametrization in {"hard_concrete", "budget"}:
                return "stochastic and budget gates require eager execution"
            if spec.dead is not None:
                return "dead-unit gate rules require eager execution"
        if spec.kind not in {"identity", "gate", "subspace"} or (
            spec.kind == "subspace" and spec.parametrization not in {None, "cayley"}
        ):
            return "only identity, gate and Cayley subspace featurizers are supported"
    if doc.train is not None and doc.train.anneal is not None:
        if any(
            target.split(".")[0] not in doc.featurizers
            or doc.featurizers[target.split(".")[0]].kind != "gate"
            or target.rsplit(".", 1)[-1] != "temperature"
            for target in doc.train.anneal
        ):
            return "only gate temperature annealing is supported"
    return None


def make_executor(
    doc: Document,
    bundle: Any,
    *,
    cuda_graphs: bool = False,
    decoding: Any = None,
    collective: Collective = SOLO,
    **kwargs: Any,
) -> PointExecutor:
    """The executor for one point: a [`GraphExecutor`][] when CUDA graphs
    are asked for and every eligibility rule holds, else the eager
    [`PointExecutor`][]. ``collective`` is the engine's
    (``docs/model_parallelism.md`` §3): the executor's ``whole`` / ``fragment``
    run over it, and a pipeline, context or rows-split world refuses graphs
    by name ([`unsupported_reason`][])."""
    reason = unsupported_reason(doc, bundle, collective) if cuda_graphs else None
    if cuda_graphs and decoding is not None:
        reason = "continuation decoding requires eager execution"
    bound = kwargs.get("batch_rows")
    if (
        cuda_graphs
        and bound is not None
        and any(len(rows) > bound for rows in kwargs.get("role_rows", {}).values())
    ):
        reason = "row-window inference uses the ordinary bounded executor"
    interning = kwargs.get("interning")
    if cuda_graphs and reason is None and interning is not None:
        # the union holds sites as the reads *capture* them (shared/head.py):
        # an lm_head read at the last token is served from ln_final, whose
        # tensor has block_output's contract and gathers the same way
        if any(
            site.component not in {"block_output", "lm_head", HEAD_INPUT}
            for sites in interning.cache.wanted.values()
            for site in sites
        ):
            reason = (
                "shared forward cache requests taps outside the supported graph path"
            )
    if cuda_graphs and reason is not None:
        logging.getLogger(__name__).info("CUDA graphs disabled: %s", reason)
    cls = GraphExecutor if cuda_graphs and reason is None else PointExecutor
    executor = cls(doc, bundle, **kwargs)
    if collective is not SOLO:
        executor.fragments = Fragments(collective)
    return executor


def _device_allocations(device: torch.device) -> int | None:
    """Count device allocations, including private pools; return None off CUDA."""
    if device.type != "cuda":
        return None
    return int(torch.cuda.memory_stats(device).get("num_device_alloc", 0))


def captured_pass(work: Callable[[], Any]) -> Callable[[], Any]:
    """Run ``work`` with a fresh, isolated [`featurizer_cache`][causalab.neural.shared.featurizers.sharing.featurizer_cache] scope.

    Each featurizer runs once within the pass. Reusing eager or warm-up cache
    entries during capture would record stale values as parameters change.
    """

    def run() -> Any:
        with featurizer_cache(isolated=True):
            return work()

    return run


class GraphPool:
    """Share one allocator pool and capture stream across an engine's graphs.

    Training buckets and evaluation graphs reuse intermediate storage. Captures
    share a stream because the allocator caches blocks per stream. Warm-up uses
    [`allocating`][] to cache the working set before capture; the allocator
    cannot reclaim memory under pressure during capture.

    A ``torch.cuda.MemPool`` retains the pool across fits and requests, even with
    no live graphs. Each ``Replay`` keeps its pool alive. The owner must close all
    graph holders before [`close`][] to avoid PyTorch's destructor assertion.
    Closing warns about and resets remaining graphs; later captures use private
    pools.

    ``use_on_oom=True`` lets eager allocations borrow free pool blocks on OOM.
    After a capture or replay OOM, call [`close_if_unused`][] after the failed
    frame unwinds, so its traceback no longer retains a graph. Release requires
    no live graphs in the engine, including fit-cache banks. Otherwise, eager
    fallback can borrow free blocks, but the pool retains its segments. Borrowed
    tensors and library workspaces do not prevent closure and remain allocated
    until freed.

    Replaying one graph may overwrite another's outputs. Every replay site must:

    * Replay graphs serially on the current stream.
    * Consume or clone every output before another graph replays: the optimizer
      consumes gradients, inference clones captures, and evaluation scores and
      releases reads.

    Inputs not produced by a graph (tokens, masks, labels, staged operands,
    frozen sources and prefix buffers) must be allocated outside the pool.
    These rules span fits and requests. Fit-owned holders close in ``finally``;
    retained fit-cache banks replay only within their active fit, and the engine
    serves one request at a time. See ``docs/cuda_graphs.md``.

    [`allocating`][] also routes unrelated device allocations from other
    threads into the pool during warm-up.
    """

    def __init__(self) -> None:
        self._pool: Any = None
        self._stream: Any = None
        self.device: torch.device | None = None
        #: Resolve with the handle so an unindexed ``cuda`` device cannot drift
        #: with the current device between capture and warm-up.
        self._index: int | None = None
        self.closed = False
        #: the graphs captured into the pool that are still alive
        self._graphs: weakref.WeakSet[Any] = weakref.WeakSet()

    def handle(self, device: torch.device) -> tuple[int, int] | None:
        """The pool id to capture into on ``device``, opened on first use;
        ``None`` once closed (the capture keeps a private pool)."""
        if self.closed:
            return None
        if self._pool is None:
            with torch.cuda.device(device):
                # free blocks serve eager allocations that would otherwise
                # fail: the OOM fallback runs eagerly beside the pool
                self._pool = torch.cuda.MemPool(use_on_oom=True)
                self._stream = torch.cuda.Stream(device=device)
                self._index = torch.cuda.current_device()
            self.device = device
        elif self.device != device:
            raise ValueError(f"graph pool is on {self.device}, capture on {device}")
        return self._pool.id

    def stream(self, device: torch.device) -> Any:
        """The one stream every capture into the pool runs on."""
        if self.handle(device) is None:
            raise ValueError("a closed graph pool has no capture stream")
        return self._stream

    @contextlib.contextmanager
    def allocating(self, device: torch.device) -> Iterator[None]:
        """Route all device allocations into the pool; no-op after closure.

        Warm-up uses the capture stream so capture can reuse its freed blocks.
        Routing covers every thread, including autograd; ``torch.cuda.use_mem_pool``
        routes only the calling thread and would miss backward allocations.

        Pool routing prevents the allocator from reclaiming its ordinary cache
        under pressure, so empty that cache first. Pool blocks remain cached.
        Leave cycle collection to ``torch.cuda.graph`` on capture entry; blocks
        still held by cycles cannot be reclaimed by the allocator anyway.
        """
        handle = self.handle(device)
        if handle is None:
            yield
            return
        allocator: Any = torch._C
        index = self._index
        assert index is not None  # resolved with the handle
        torch.cuda.empty_cache()
        # Begin/end control device-wide routing; release balances begin's pool
        # registration. MemPool retains its own registration and cached blocks
        # until close, when its destructor requires that registration to be last.
        allocator._cuda_beginAllocateToPool(index, handle)
        try:
            yield
        finally:
            allocator._cuda_endAllocateToPool(index, handle)
            allocator._cuda_releasePool(index, handle)

    def captured(self, graph: Any) -> None:
        """Note a graph captured into the pool (``Replay``)."""
        if not self.closed:
            self._graphs.add(graph)

    def close_if_unused(self) -> bool:
        """Release the pool when no live graph is captured into it — a graph
        holder's out-of-memory fallback, handing the working set back to the
        eager remainder of the fit. ``True`` when the pool was released."""
        if self._graphs:
            return False
        self.close()
        return True

    def close(self) -> None:
        """Release the pool. Every graph holder must be closed already; a
        graph still alive is a closing-order bug — warned about, and reset
        here rather than let outlive the pool."""
        self.closed = True
        if self._pool is None:
            return
        if self._graphs:
            assert self.device is not None
            _log.warning(
                "graph pool released with %d live graph(s): a holder outlived "
                "its pool; the graphs are reset and cannot replay",
                len(self._graphs),
            )
            torch.cuda.synchronize(self.device)
            for graph in list(self._graphs):
                graph.reset()
        self._graphs.clear()
        self._pool = None
        self._stream = None
        self._index = None
        self.device = None

    def __del__(self) -> None:
        # unreachable while a Replay holds the pool; the last line of defence
        # for the allocator's use-count assertion if a pool is dropped unclosed
        try:
            self.close()
        except BaseException:  # noqa: BLE001 — a finalizer must not raise
            pass


class _Recorded:
    """An event recorded after a replay, as the replay deadline's
    [`Completion`][causalab.neural.shared.parallel.deadline.Completion]. The
    heartbeat thread queries it under the replay's device, since that thread's
    current device is ``cuda:0``. ``Event.query`` never waits."""

    def __init__(self, event: torch.cuda.Event, device: torch.device) -> None:
        self.event = event
        self.device = device

    def query(self) -> bool:
        with torch.cuda.device(self.device):
            return self.event.query()


def capture_window() -> contextlib.AbstractContextManager[None]:
    """Pause the replay deadline for a capture about to begin
    ([`Outstanding.capturing`][causalab.neural.shared.parallel.deadline.Outstanding.capturing]).
    Open it after draining the device, so a stuck earlier replay is still
    refused. A no-op without a running heartbeat."""
    beat = heartbeat.running()
    if beat is None:
        return contextlib.nullcontext()
    return beat.outstanding.capturing()


def bound_replay(device: torch.device) -> None:
    """Register the replay just enqueued on ``device`` with the replay
    deadline (``docs/cuda_graphs.md`` "Hung replays"). A no-op without a
    running heartbeat: world 1, or a process that never joined a group."""
    beat = heartbeat.running()
    if beat is None:
        return
    stream = torch.cuda.current_stream(device)
    event = torch.cuda.Event()
    event.record(stream)
    on = torch.device("cuda", stream.device_index)
    beat.enqueued((on.index, stream.cuda_stream), REPLAY, _Recorded(event, on))


class Replay:
    """One captured callable; optimizer updates deliberately remain eager."""

    # One side-stream pass initializes kernels before capture. Repeating it
    # per executor adds substantial cost to short fits without changing work.
    warmup_steps = 1

    def __init__(
        self,
        work: Callable[[], Any],
        stages: dict[str, Stage],
        *,
        device: torch.device,
        parameters: list[torch.nn.Parameter] | None = None,
        pool: GraphPool | None = None,
        warmup: bool = True,
    ) -> None:
        """Capture ``work`` in the supplied pool, or a private pool if unavailable.

        PyTorch documents shared-pool replay in capture order. Evaluation
        graphs interleave with training under the [`GraphPool`][] rules:
        replay serially on one stream and consume outputs before the next
        replay. GPU coverage lives in ``tests/golden/test_graph_cohort.py``.

        ``warmup`` runs ``work`` eagerly on the capture stream to initialize
        kernels, libraries and autograd, and populate the pool's working set.
        Set it to ``False`` only after an equivalent eager pass on the same
        storage, as in ``graph_cohort.EvaluationGraphs``. Capture then reuses
        cached blocks or grows the pool as needed.
        """
        self.parameters = parameters or []
        self.device = device
        self.temperatures = [
            (stage, stage.theta.new_tensor(stage.temperature))
            for stage in stages.values()
            if isinstance(stage, Gate)
        ]
        run = captured_pass(work)
        # the fit's shared pool and its capture stream; None (or a closed
        # pool) keeps a private pool on a stream of its own. The replay holds
        # the pool so it cannot be collected while this graph is alive.
        self.pool = pool
        handle: Any = None if pool is None else pool.handle(device)
        # the pool while it is open; a closed one leaves the capture private
        shared = pool if handle is not None else None
        debug = _log.isEnabledFor(logging.DEBUG)
        allocations = _device_allocations(device) if debug else None
        started = time.perf_counter()
        with torch.cuda.device(device), self._temperatures():
            stream = (
                shared.stream(device)
                if shared is not None
                else torch.cuda.Stream(device=device)
            )
            stream.wait_stream(torch.cuda.current_stream(device))
            if warmup:
                # into the pool, so the capture below finds the working set
                # cached on its stream (GraphPool.allocating)
                routing = (
                    shared.allocating(device)
                    if shared is not None
                    else contextlib.nullcontext()
                )
                with torch.cuda.stream(stream), routing:
                    for _ in range(self.warmup_steps):
                        self._clear_grad()
                        run()
            torch.cuda.current_stream(device).wait_stream(stream)
            torch.cuda.synchronize(device)
            self._clear_grad()
            warmed = time.perf_counter()
            self.graph = torch.cuda.CUDAGraph()
            if shared is not None:
                shared.captured(self.graph)
            # after the drain above, so a stuck replay is still refused by name
            with (
                capture_window(),
                torch.cuda.graph(self.graph, stream=stream, pool=handle),
            ):
                self.output = run()
            self.gradients = [p.grad for p in self.parameters]
        if debug:
            after = _device_allocations(device)
            # the first span is the warm-up pass(es) plus the wait for the
            # device to drain what was queued before this capture
            _log.debug(
                "graph captured: warm-up and drain %.1f ms (%d pass(es)), capture "
                "%.1f ms, pool %s, device allocations %s",
                (warmed - started) * 1e3,
                self.warmup_steps if warmup else 0,
                (time.perf_counter() - warmed) * 1e3,
                "shared" if shared is not None else "private",
                None if allocations is None or after is None else after - allocations,
            )
        self.replays = 0

    def _clear_grad(self) -> None:
        for parameter in self.parameters:
            parameter.grad = None

    @contextlib.contextmanager
    def _temperatures(self) -> Iterator[None]:
        previous = [stage.capture_temperature for stage, _ in self.temperatures]
        try:
            for stage, value in self.temperatures:
                stage.capture_temperature = value
            yield
        finally:
            for (stage, _), value in zip(self.temperatures, previous):
                stage.capture_temperature = value

    def __call__(self) -> Any:
        for stage, value in self.temperatures:
            value.fill_(stage.temperature)
        # Another graph or eager step may have rebound .grad. The captured
        # backward writes these exact buffers and overwrites on each replay.
        for parameter, gradient in zip(self.parameters, self.gradients):
            parameter.grad = gradient
        self.graph.replay()
        bound_replay(self.device)
        self.replays += 1
        return self.output


class GraphExecutor(PointExecutor):
    """Same hook executor, with prepared masks/positions and inference replay."""

    cuda_graphs = True
    # One-shot forwards should never pay capture costs. A second execution
    # establishes reuse; repeated held-out evaluation then uses replay.
    inference_capture_after = 1
    #: whether this point's fit keeps the campaign store for its minibatch and
    #: eval executors (a captured cohort does, graph_cohort.py: shared sources
    #: and prefix resume come from it); a solo graph fit runs on its frozen
    #: cache instead
    keep_store = False
    #: the pool this executor's inference replays are captured into — the
    #: one the fit's graphs share (the engine's, or one the loop opened), set
    #: by train.py on the fit's eval executor when it is built
    #: (``_eval_executor``); None (one-shot inference, a point off a fit, the
    #: point executor itself) keeps a private pool per graph
    graph_pool: GraphPool | None = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._masks: dict[int, Any] = {}
        self._position_ids: dict[int, torch.Tensor] = {}
        #: prepared masks for batches this executor did not build — an eager
        #: cohort's concatenated frame, rebuilt per forward — keyed by shape
        #: and per-row lengths (left padding), a handful kept
        self._transient: dict[Any, tuple[Any, torch.Tensor]] = {}
        self._inference_graphs: dict[Any, tuple[Replay, dict[str, torch.Tensor]]] = {}
        self._inference_calls: dict[Any, int] = {}
        self._recording = False
        self._frozen: dict[str, dict[Any, Any]] = {"sources": {}, "prefixes": {}}
        self._frozen_versions: tuple[Any, ...] | None = None
        self._frozen_ready = False

    def close(self) -> None:
        """Release fit-owned evaluation captures before the next fit starts."""
        if self._inference_graphs:
            torch.cuda.synchronize(graph_device(self.bundle))
        self.reset_reads()
        self._inference_graphs.clear()
        self._inference_calls.clear()
        self._transient.clear()
        self._clear_frozen()

    def prepare(self) -> None:
        if self._masks:
            return
        # Resolve and bounds-check the ordinary positions outside capture.
        # An empty token sequence must still fail through the normal resolver.
        self.check_write_widths()
        for ref in self.doc.read_refs():
            model, role = self.doc.group_of(ref)
            batch = self._batch(role)
            super()._positions(self.doc.reads[ref.read].pos, batch, role)
            for name in write_names(self.doc, model) or ():
                super()._positions(self.doc.writes[name].pos, batch, role)
        for entry in (*self.doc.reads.values(), *self.doc.writes.values()):
            if isinstance(entry.featurizer, str):
                self.stage(entry.featurizer)
        for role in self.role_rows:
            batch = self._batch(role)
            self._masks[id(batch)], self._position_ids[id(batch)] = self.prepare_batch(
                batch
            )

    def prepare_batch(self, batch) -> tuple[Any, torch.Tensor]:
        """The prebuilt attention mask(s) and position ids ``_model_forward``
        hands the model for ``batch``, computed eagerly — outside capture, so
        a replay reads them from fixed storage. A cohort's captured frame
        (``graph_cohort.py``) is prepared through here and registered under
        its own id. The masks are [`prompt_masks`][], the eager executor's own."""
        with torch.no_grad():
            position_ids = batch.position_ids()
            masks = prompt_masks(
                self.bundle.model, batch.input_ids, batch.attention_mask, position_ids
            )
        return masks, position_ids

    def _model_forward(
        self, batch, depth, window=None, start: int = 0, *, backward: bool = False
    ):
        # `backward` is the stage forward's (stages.py): graphs run at world 1,
        # one stage, where the model call is its own backward's graph
        assert depth == 0, "CUDA graphs require prompt-only execution"
        assert window is None or (window.start, window.stop) == (
            0,
            batch.input_ids.shape[0],
        ), "CUDA graphs require a whole-batch window"
        key = id(batch)
        if key in self._masks:
            masks, position_ids = self._masks[key], self._position_ids[key]
        else:
            # a batch this executor did not prepare — an eager cohort's
            # concatenated frame (cohort.py builds one per forward), whose id
            # is transient but whose masks depend only on its shape and on
            # where each row's real tokens start under the position frame's
            # always-left-padding invariant (protocol/positions/encoding.py) (a row has at least one). The
            # frame carries those indices, so the signature costs no host
            # read: the launches below queue behind the device's backlog
            # instead of the CPU waiting for it to drain here. Repeated
            # frames reuse masks while resident in this FIFO cache.
            signature = (tuple(batch.input_ids.shape), batch.first_reals)
            cached = self._transient.get(signature)
            if cached is None:
                cached = self.prepare_batch(batch)
                if len(self._transient) >= 4:
                    del self._transient[next(iter(self._transient))]
                self._transient[signature] = cached
            masks, position_ids = cached
        return self.bundle.model(
            input_ids=batch.input_ids,
            attention_mask=masks,
            position_ids=position_ids,
            use_cache=False,
        )

    def _positions(self, pos, batch, input_role, *, cell=None):
        if not self._masks:
            return super()._positions(pos, batch, input_role, cell=cell)
        return [[batch.padded_len - 1] for _ in batch.texts]

    @staticmethod
    def _gather(tensor, per_row, what):
        # Own storage, matching the eager advanced-index gather.
        return tensor[:, -1:, :].clone()

    def _apply_writes_to_contract(
        self,
        entries,
        input_role,
        batch,
        tensor,
        *,
        per_row=None,
        rows=None,
        routing=None,
    ):
        for ename, write, site in entries:
            value = tensor[:, -1:, :].clone()
            tensor[:, -1:, :] = self._written_value(
                ename, write, site, value, rows=rows, routing=routing
            ).to(tensor.dtype)

    def _publish(self, key, label, capture, routing):
        if not self._recording:
            super()._publish(key, label, capture, routing)

    def _clear_frozen(self) -> None:
        self._frozen = {"sources": {}, "prefixes": {}}
        self._frozen_versions = None
        self._frozen_ready = False

    def _input_versions(self) -> tuple[Any, ...]:
        return tuple(
            (id(value), value._version)
            for role in sorted(self.role_rows)
            for value in (self._batch(role).input_ids, self._batch(role).attention_mask)
        )

    def _check_frozen_inputs(self) -> None:
        versions = self._input_versions()
        if self._frozen_versions is not None and versions != self._frozen_versions:
            # Captures may reference an old prefix buffer. Discard both before
            # preparing new values; ordinary inference graphs stage tokens only.
            self.close()
            self._masks.clear()
            self._position_ids.clear()
        self._frozen_versions = versions

    def prepare_frozen(self, objective: Callable[[], Any]) -> None:
        """Materialize immutable raw sources/prefixes outside graph capture.

        Trainable source featurizers still run on every loss evaluation. A
        minibatch owns these constants; workers own separate replay storage.
        """
        self._check_frozen_inputs()
        if self._frozen_ready:
            return
        self.reset_reads()
        try:
            with torch.no_grad():
                objective()
            self._frozen_ready = True
        except BaseException:
            self._clear_frozen()
            raise
        finally:
            self.reset_reads()

    def stage_frozen(self, source: "GraphExecutor", *, allocate: bool = False) -> None:
        with torch.no_grad():
            if allocate:
                self._frozen = tree_map(
                    lambda value: value.detach().clone()
                    if isinstance(value, torch.Tensor)
                    else value,
                    source._frozen,
                )
            else:
                tree_map(
                    lambda target, value: target.copy_(value)
                    if isinstance(target, torch.Tensor)
                    else target,
                    self._frozen,
                    source._frozen,
                )
        self._frozen_versions = self._input_versions()
        self._frozen_ready = True

    def _forward_group(self, model, input_role, **kwargs):
        # Frozen work belongs to fits and their held-out evaluation. One-shot
        # inference retains its ordinary input-staging/capture behavior.
        writes = tuple(self.doc.writes.values())
        if (
            self.interning is not None
            or self.doc.train is None
            or kwargs.get("depth", 0)
            or len(writes) != 1
            or self.doc.sites[str(writes[0].site)].component != "block_output"
        ):
            return self._run_forward_group(model, input_role, **kwargs)
        self._check_frozen_inputs()
        key = (model, input_role)
        if is_unwritten(self.doc, model):
            if key not in self._frozen["sources"]:
                result = self._run_forward_group(model, input_role, **kwargs)
                self._frozen["sources"][key] = tree_map(
                    lambda value: value.detach().clone()
                    if isinstance(value, torch.Tensor)
                    else value,
                    result,
                )
            return self._frozen["sources"][key]
        if key in self._frozen["prefixes"]:
            depth, prefix = self._frozen["prefixes"][key]
            with _resumed(self.bundle.blocks, depth, prefix):
                return self._run_forward_group(model, input_role, **kwargs)
        site = resolve_site(self.bundle, self.doc.sites[str(writes[0].site)])
        if site.layer == 0:
            return self._run_forward_group(model, input_role, **kwargs)

        def save_prefix(module, args):
            self._frozen["prefixes"][key] = (site.layer, args[0].detach().clone())

        handle = self.bundle.blocks[site.layer].register_forward_pre_hook(save_prefix)
        try:
            return self._run_forward_group(model, input_role, **kwargs)
        finally:
            handle.remove()

    def _run_forward_group(
        self,
        model,
        input_role,
        *,
        shared_key,
        capture_sites,
        depth,
        gen_capture_sites=None,
        prefix=None,
    ):
        self.prepare()
        assert depth == 0 and not gen_capture_sites, (
            "CUDA graphs require prompt-only execution"
        )
        # Campaign prefix plans own shared residuals. Keep their ordinary
        # execution path; fit-local frozen prefixes have no interning plan.
        if self.grad_enabled or prefix is not None:
            return super()._forward_group(
                model,
                input_role,
                shared_key=shared_key,
                capture_sites=capture_sites,
                depth=depth,
                gen_capture_sites=gen_capture_sites or {},
                prefix=prefix,
            )
        # Resolve operands before capture. Each replay stages fresh values in
        # fixed storage, including operands served from the forward cache.
        operands: dict[ReadRef, torch.Tensor] = {}
        names = write_names(self.doc, model)
        assert names is not None
        for name in names:
            for ref in operand_reads(self.doc, self.doc.writes[name].do):
                operands[ref] = self.dense_value(ref).to(graph_device(self.bundle))
        modes = tuple(
            (name, s.training, getattr(s, "hard_eval", None))
            for name, s in self.stage_cache.items()
        )
        key = (model, input_role, modes)
        calls = self._inference_calls.get(key, 0)
        self._inference_calls[key] = calls + 1
        if calls < self.inference_capture_after:
            return super()._forward_group(
                model,
                input_role,
                shared_key=shared_key,
                capture_sites=capture_sites,
                depth=depth,
                gen_capture_sites=gen_capture_sites or {},
                prefix=prefix,
            )
        if key not in self._inference_graphs:
            # Drop old modes rather than retaining train/eval variants forever.
            old_modes = [old for old in self._inference_graphs if old[:2] == key[:2]]
            if old_modes:
                torch.cuda.synchronize(graph_device(self.bundle))
                for old in old_modes:
                    del self._inference_graphs[old]
            static = {ref: value.detach().clone() for ref, value in operands.items()}
            saved_reads = self._read_values
            self._read_values = {**saved_reads, **static}
            self._recording = True
            try:

                def work():
                    with torch.no_grad():
                        capture, routing, _ = super(GraphExecutor, self)._forward_group(
                            model,
                            input_role,
                            shared_key=shared_key,
                            capture_sites=capture_sites,
                            depth=0,
                            gen_capture_sites={},
                        )
                    return capture, routing

                replay = Replay(
                    work,
                    self.stage_cache,
                    device=graph_device(self.bundle),
                    pool=self.graph_pool,
                )
            finally:
                self._recording = False
                self._read_values = saved_reads
            self._inference_graphs[key] = replay, static
        replay, static = self._inference_graphs[key]
        for ref, value in operands.items():
            static[ref].copy_(value)
        capture, routing = replay()
        # ForwardCache and consumers own their tensors. Later replays must
        # never overwrite an earlier point's saved activations — and on the
        # fit's shared pool (GraphPool) another graph's replay may reuse
        # these blocks, so the copy happens before anything else replays.
        capture = {key: value.detach().clone() for key, value in capture.items()}
        routing = {key: value.detach().clone() for key, value in routing.items()}
        self._publish(
            shared_key,
            f"{model}/{input_role}",
            {key: value for key, value in capture.items() if value.numel()},
            routing,
        )
        return capture, routing, None


class TrainingGraphs:
    """One captured step per shape/mask bucket of a fit, all on one pool.

    Matching batches stage tokens and labels into their bucket's executor.
    Masks are part of the key: no stale padding or position IDs on replay.
    Buckets retain their gradients and loss but share intermediate storage
    through [`GraphPool`][]. Held-out inference uses the same pool.
    [`close`][] releases the buckets; the pool owner releases the pool after
    all graph holders close. ``FitGraphCache`` may retain banks across fits.
    An allocation OOM during capture or replay releases the bank and runs the
    remainder of the fit eagerly. Under ``OOMPolicy.ABORT`` (a geometry whose
    windows carry collectives) the run ends instead
    ([`abort_distributed`][causalab.neural.engines.pytorch_hooks.budget.abort_distributed]),
    since this rank alone turning eager would desynchronize its peers.
    """

    def __init__(
        self,
        parameters: list[torch.nn.Parameter],
        *,
        pool: GraphPool | None = None,
        oom_policy: OOMPolicy = OOMPolicy.RETRY,
    ) -> None:
        self.parameters = parameters
        self.oom_policy = oom_policy
        # a pool the bank opened itself is the bank's to release; one handed
        # in belongs to the fit's owner, which closes every holder first
        self._owns_pool = pool is None
        self.pool = GraphPool() if pool is None else pool
        self.buckets: dict[Any, tuple[GraphExecutor, Any, Replay]] = {}
        self.keys: dict[int, Any] = {}
        self.disabled = False

    def backward(self, executor: PointExecutor, objective: Any) -> bool:
        try:
            return self._backward(executor, objective)
        except torch.OutOfMemoryError as error:
            if self.oom_policy is OOMPolicy.ABORT:
                abort_distributed(error, IN_GRAPH)
            torch.cuda.synchronize(graph_device(executor.bundle))
            # The failed worker/capture frame must unwind before releasing
            # cached pools. Do not swallow capture-invalidated CUDA errors.
            self._release()
            self.disabled = True
            executor.reset_reads()
        # Outside the except block: the traceback no longer retains the
        # partially constructed replay or its tensors — nor its graph, which
        # the pool counted as live until this point.
        gc.collect()
        # The eager remainder needs the working set the graphs held: give the
        # pool back unless the fit's inference replays still use it (then its
        # free blocks still serve eager allocations on OOM).
        self.pool.close_if_unused()
        torch.cuda.empty_cache()
        logging.getLogger(__name__).info(
            "CUDA training allocation ran out of memory; remaining fit uses eager execution"
        )
        return False

    def _backward(self, executor: PointExecutor, objective: Any) -> bool:
        if self.disabled or not isinstance(executor, GraphExecutor):
            return False
        versions = executor._input_versions()
        cached_key = self.keys.get(id(executor))
        if cached_key is None or cached_key[0] != versions:
            self.keys[id(executor)] = (
                versions,
                tuple(
                    (
                        role,
                        tuple(executor._batch(role).input_ids.shape),
                        tuple(
                            executor._batch(role)
                            .attention_mask.cpu()
                            .flatten()
                            .tolist()
                        ),
                    )
                    for role in sorted(executor.role_rows)
                ),
            )
        key = self.keys[id(executor)][1]
        if key not in self.buckets:
            device = graph_device(executor.bundle)
            # All buckets keep the first capture's storage, including after
            # a compatible fit supplies a new optimizer and parameters.
            captured = next(iter(self.buckets.values()), None)
            worker = GraphExecutor(
                captured[0].doc if captured else executor.doc,
                executor.bundle,
                role_rows=executor.role_rows,
                role_fields=executor.role_fields,
                load_tensors=executor.load_tensors,
                load_table=executor.load_table,
                stage_cache=captured[0].stage_cache
                if captured
                else copy.deepcopy(executor.stage_cache),
                grad_enabled=True,
                batches={
                    role: copy.deepcopy(executor.frame(role))
                    for role in executor.role_rows
                },
                coords=captured[0].coords if captured else executor.coords,
            )
            executor.prepare_frozen(objective)
            worker.stage_frozen(executor, allocate=True)
            copy_executor_stages(worker, executor)
            prepared_objective = objective.for_executor(worker)
            worker.prepare()

            def work():
                worker.reset_reads()
                loss = prepared_objective()
                loss.backward()
                return loss.detach()

            if captured:
                captured_parameters = captured[2].parameters
            else:
                parameter_names = {
                    id(value): (name, key)
                    for name, stage in executor.stage_cache.items()
                    for key, value in stage.named_parameters()
                }
                worker_parameters = {
                    (name, key): value
                    for name, stage in worker.stage_cache.items()
                    for key, value in stage.named_parameters()
                }
                captured_parameters = [
                    worker_parameters[parameter_names[id(parameter)]]
                    for parameter in self.parameters
                ]
            replay = Replay(
                work,
                worker.stage_cache,
                device=device,
                parameters=captured_parameters,
                pool=self.pool,
            )
            self.buckets[key] = worker, prepared_objective, replay
        worker, prepared_objective, replay = self.buckets[key]
        copy_executor_stages(worker, executor)
        for role in worker.role_rows:
            worker._batch(role).input_ids.copy_(executor._batch(role).input_ids)
        executor.prepare_frozen(objective)
        worker.stage_frozen(executor)
        prepared_objective.copy_labels(objective)
        replay()
        # The gradients live in the shared pool: another bucket captured later
        # may hold its outputs in this replay's intermediates, and vice versa.
        # The loop's optimizer step reads them before anything else replays.
        for current, gradient in zip(self.parameters, replay.gradients, strict=True):
            current.grad = gradient
        return True

    def close(self) -> None:
        self._release()
        if self._owns_pool:
            self.pool.close()

    def _release(self) -> None:
        """Release the buckets and their gradients, leaving the pool: the
        OOM fallback hands it back only once the failed frame has unwound."""
        for worker, _, _ in self.buckets.values():
            torch.cuda.synchronize(graph_device(worker.bundle))
            worker.close()
        self.buckets.clear()
        self.keys.clear()
        for parameter in self.parameters:
            parameter.grad = None
