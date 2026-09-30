"""Capture a cohort's differentiable step in one CUDA graph.

Each member owns a fixed slot of ``min(pairs, dataset_size)`` rows. Short
minibatches repeat their last row with zero loss weight. Padding can change
MoE group sizes and bf16 rounding. Full active slots preserve the eager
cohort's arithmetic. Stopped members retain slots with ignored gradients.

Capture includes the intervened forward, member objectives, summed loss,
and backward. Optimizer updates, projection, schedules, and scoring remain
in Python. Evaluation uses a separate capture. Replays copy current inputs
and prepared minibatch data into persistent storage.

The campaign store continues to share source forwards. Captures copy source
values and the prefix below the shallowest write into owned buffers.
Missing prefixes are computed on the same concatenated frame. Allocation
failure or an incompatible padded mask releases the graph and continues
with the eager cohort.
"""

from __future__ import annotations

# This engine-owned lowering operates on the executors' internal buffers, as
# cuda_graphs.py does.
# pyright: reportPrivateUsage=false

import copy
import dataclasses
import gc
import logging
from typing import Any, Callable, Mapping, Sequence

import torch
from torch.utils._pytree import tree_map

from causalab.neural.engines.pytorch_hooks.cohort import (
    _concat_frames,
    cohort_entries,
    groups_read_by,
    run_groups,
)
from causalab.neural.engines.pytorch_hooks.budget import OOMPolicy, abort_distributed
from causalab.neural.engines.pytorch_hooks.cuda_graphs import (
    IN_GRAPH,
    GraphExecutor,
    GraphPool,
    Replay,
    copy_executor_stages,
    graph_device,
)
from causalab.neural.engines.pytorch_hooks.executor import (
    PointExecutor,
    Resume,
    _resumable,
)
from causalab.neural.shared.encoding import EncodedBatch
from causalab.neural.shared.executor import PrefixKey, RowWindow, TapKey, tap_key
from causalab.neural.shared.featurizers import Gate, Stage
from causalab.neural.shared.plan import group_reads, is_unwritten, write_names
from causalab.protocol.schema import ReadRef, operand_reads

__all__ = [
    "CohortGraphs",
    "EvaluationGraphs",
    "Member",
    "cohort_graph_reason",
    "pad_rows",
    "padded_indices",
]

_log = logging.getLogger(__name__)


def padded_indices(indices: Sequence[int], pairs: int) -> list[int]:
    """The minibatch's rows, then its last row repeated up to ``pairs`` —
    the slot a member's minibatch occupies in the captured frame. A whole
    minibatch is its own slot; a longer one is a programming error."""
    rows = list(indices)
    if not rows:
        raise ValueError("a slot holds at least one row")
    if len(rows) > pairs:
        raise ValueError(f"{len(rows)} rows do not fit a slot of {pairs}")
    return rows + [rows[-1]] * (pairs - len(rows))


def slotted_frame(batch: EncodedBatch, real: int, total: int) -> EncodedBatch:
    """``batch``'s ``real`` rows laid into a slot of ``total`` — the last row
    repeated, as [`pad_rows`][] lays a tensor — with every per-row field of
    the frame, its first-real cache included, sliced in step. Building the
    slot as a row selection rather than by replacing the two tensors is what
    keeps the frame's cached indices the mask's: a replaced mask with the
    minibatch's shorter cache is refused by the frame. A minibatch that
    fills its slot *is* the slot — returned as is, its tensors aliased
    rather than copied, as [`pad_rows`][] does."""
    if batch.input_ids.shape[0] != real:
        raise ValueError(
            f"expected a {real}-row minibatch to slot into {total}, got "
            f"{int(batch.input_ids.shape[0])} rows"
        )
    if real == total:
        return batch
    return batch.select(padded_indices(range(real), total))


def pad_rows(tensor: torch.Tensor, real: int, total: int) -> torch.Tensor:
    """``tensor``'s first ``real`` rows, then row ``real - 1`` repeated to
    ``total`` rows along dim 0 — a minibatch's per-row values laid into its
    slot. A tensor already ``total`` rows long is returned as is."""
    if tensor.shape[0] == total:
        return tensor
    if tensor.shape[0] != real:
        raise ValueError(
            f"expected {real} rows to pad to {total}, got {tuple(tensor.shape)}"
        )
    index = torch.arange(total, device=tensor.device).clamp_(max=real - 1)
    return tensor.index_select(0, index)


def slot_weights(real: int, total: int, device: torch.device | str) -> torch.Tensor:
    """The loss weight of each slot row: one on the minibatch's rows, zero
    on the padding."""
    weight = torch.zeros(total, dtype=torch.float32, device=device)
    weight[:real] = 1.0
    return weight


def cohort_graph_reason(
    executors: Sequence[PointExecutor],
    fit_rows: int | None,
    pairs: Sequence[int],
) -> str | None:
    """Why this cohort cannot be captured as one graph, or ``None``.

    Every member must already be graph-eligible (a [`GraphExecutor`][],
    which ``make_executor`` only builds for the validated CUDA path), and an
    authored ``fit_rows`` must hold the whole layout: the captured frame is
    every member's slot at once, so a bound that would split the members into
    windows has nothing to capture. Off CUDA there are no graphs at all.
    """
    if len(executors) < 2:
        return "a cohort graph needs at least two members"
    if not all(isinstance(executor, GraphExecutor) for executor in executors):
        return "every member of a captured cohort must be graph-eligible"
    single = executors[0].bundle.devices.single
    if single is None:
        return (
            "CUDA graphs are single-device; the bundle places its layers on "
            f"{executors[0].bundle.devices.spelling}"
        )
    if single.type != "cuda":
        return "CUDA graphs require a CUDA device"
    slots = sum(pairs)
    if fit_rows is not None and fit_rows < slots:
        return (
            f"authored fit_rows={fit_rows} is below the cohort's {slots} slot "
            "rows; the members would run in windows"
        )
    return None


@dataclasses.dataclass(frozen=True)
class Member:
    """One fit's share of the captured cohort: its point executor (the frame
    and stage cache the workers copy), the stages it trains, its optimizer's
    parameters in optimizer order, the reads its objective needs, and the
    rows of its slot (``train.batch.pairs``)."""

    key: int
    executor: PointExecutor
    stages: Mapping[str, Stage]
    parameters: Sequence[torch.nn.Parameter]
    objective_reads: Sequence[str]
    pairs: int


@dataclasses.dataclass(frozen=True)
class WindowItem:
    """One member's step: which member, the rows of the minibatch it drew,
    the minibatch executor over those rows, and the objective built on it
    (whose labels are this minibatch's)."""

    key: int
    indices: Sequence[int]
    minibatch: PointExecutor
    objective: Any


_Copies = list[tuple[torch.Tensor, torch.Tensor]]
#: a left-padded mask's identity within a slot: its padded width and each
#: row's first real token (``EncodedBatch.first_reals``)
_MaskKey = tuple[int, tuple[int, ...]]
_Group = tuple[str, str]


@dataclasses.dataclass
class _Prepared:
    """One minibatch laid into its slot, computed once: the copies a step
    makes into captured storage (target, source), including per-row prepared
    masks and positions, and the store key of its resume prefix. The padded
    prefix is retained lazily once the store has computed it."""

    #: tokens: into the worker's own batch and into the frame's rows
    tokens: _Copies
    #: labels, padding weight, source activations
    values: _Copies
    masks: dict[str, tuple[_MaskKey, _Copies]]
    prefix_key: PrefixKey | None
    real: int
    prefix: torch.Tensor | None = None


@dataclasses.dataclass
class _Slot:
    member: Member
    index: int
    worker: GraphExecutor
    objective: Any
    weight: torch.Tensor
    #: this member's rows of the captured frame
    rows: slice
    #: the captured parameters' positions in the replay's gradient list
    gradient_slice: slice
    #: the write operands and the ``(model, input)`` source group each reads
    operands: dict[ReadRef, _Group]
    #: the worker's stage tensors and the member's they are staged from
    state: _Copies
    #: worker gate, member gate: the annealed temperature is a Python float
    gates: list[tuple[Gate, Gate]]
    #: by ``id(minibatch executor)``
    prepared: dict[int, _Prepared] = dataclasses.field(default_factory=dict)
    mask_cache: dict[tuple[str, _MaskKey], _Copies] = dataclasses.field(
        default_factory=dict
    )
    staged_masks: dict[str, _MaskKey] = dataclasses.field(default_factory=dict)


class _LayoutMismatch(Exception):
    """A staged value has a structure the captured buffers cannot take."""


class _StopForward(Exception):
    """Raised from a pre-hook to end a forward once the prefix is stored."""


def _single_role(members: Sequence[tuple[PointExecutor, Sequence[str]]]) -> str | None:
    """A capturable layout has one role with exactly one entry per member."""
    entries = cohort_entries(members)
    if len(entries) != 1:
        return None
    role, group = next(iter(entries.items()))
    if len(group) != len(members) or {id(entry.executor) for entry in group} != {
        id(ex) for ex, _ in members
    }:
        return None
    return role


def _trained_group(executor: PointExecutor, reads: Sequence[ReadRef]) -> _Group:
    """The ``(model, input)`` group the objective's reads are taken on — the
    one the cohort forward runs."""
    for model, role in groups_read_by(executor.doc, reads):
        if not is_unwritten(executor.doc, model):
            return model, role
    raise AssertionError("a fit's objective reads no intervened model")


def _operand_groups(executor: PointExecutor) -> dict[ReadRef, _Group]:
    """The reads a member's writes take their payload from, and the source
    group each is a tap of — what the worker's frozen buffers must hold."""
    doc = executor.doc
    out: dict[ReadRef, _Group] = {}
    for model in doc.intervened_models:
        for ename in write_names(doc, model) or ():
            for ref in operand_reads(doc, doc.writes[ename].do):
                if ref not in out:
                    out[ref] = doc.group_of(ref)
    return out


def _group_taps(executor: PointExecutor, group: _Group) -> dict[TapKey, Any]:
    """The tap keys of ``group``'s reads on ``executor``'s document — the
    sites the group's forward **captures** for them (``shared/head.py``),
    which is what the store and the frozen sources are keyed by."""
    model, role = group
    reads = [
        (ref.read, executor.doc.reads[ref.read])
        for ref in group_reads(executor.doc, model, role)
    ]
    taps = executor._read_taps(model, role, reads)
    return {
        tap_key(tap.capture): executor.doc.reads[rname] for rname, tap in taps.items()
    }


class CohortGraphs:
    """The captured pass of one cohort: built on the first step that brings
    every member, replayed on every later one (module docstring).

    ``make_objective(executor, stages, weight=)`` builds a member's objective
    on a worker — the loop's [`TrainingObjective`][causalab.neural.engines.pytorch_hooks.train.TrainingObjective] — with the slot's
    padding weight; it is injected so this module does not import the loop.
    ``pool`` is the fit's [`GraphPool`][] the step graph is captured into
    — the loop hands the same one to [`EvaluationGraphs`][] and owns its
    release; by default the bank opens one of its own. ``oom_policy`` is the
    fit's: under ``OOMPolicy.ABORT`` an out-of-memory capture or replay ends
    the run by name instead of turning this rank alone eager
    ([`TrainingGraphs`][causalab.neural.engines.pytorch_hooks.cuda_graphs.TrainingGraphs]).
    A `_LayoutMismatch` keeps the eager fallback under every policy: it
    follows from the staged rows and the store, which every rank of a model
    group holds alike, so the ranks turn eager on the same step.
    """

    def __init__(
        self,
        members: Sequence[Member],
        *,
        make_objective: Callable[..., Any],
        pool: GraphPool | None = None,
        oom_policy: OOMPolicy = OOMPolicy.RETRY,
    ) -> None:
        if len(members) < 2:
            raise ValueError("a cohort graph needs at least two members")
        self.members = list(members)
        self.make_objective = make_objective
        self.oom_policy = oom_policy
        self._owns_pool = pool is None
        self.pool = GraphPool() if pool is None else pool
        self.by_key = {member.key: i for i, member in enumerate(self.members)}
        self.slots: list[_Slot] = []
        #: the captured frame per input role: the slots' batches concatenated
        #: once, in fixed storage the graph reads; tokens are staged into it
        self.frames: dict[str, EncodedBatch] = {}
        #: the block the captured forward resumes at (0: a full forward) and,
        #: per role, the residual buffer entering it over the frame's rows
        self.resume_at = 0
        self.prefix: dict[str, torch.Tensor] = {}
        self.replay: Replay | None = None
        self.disabled = False
        self.replays = 0
        self.eager_steps = 0
        self.prefix_forwards = 0

    # ------------------------------------------------------------------ #

    def backward(self, window: Sequence[WindowItem]) -> bool:
        """Run one optimizer step's forward and backward for ``window``'s
        members through the captured graph, leaving each member's gradients
        on its parameters; ``False`` when the step must run eagerly instead
        (graphs released, or a layout the buffers cannot hold)."""
        if self.disabled:
            return False
        try:
            if self.replay is None:
                if len(window) != len(self.members):
                    # the layout is every member; a first step with fewer
                    # (a member already exhausted) has no full frame to capture
                    self.eager_steps += 1
                    return False
                self._capture(window)
            return self._step(window)
        except (torch.OutOfMemoryError, _LayoutMismatch) as error:
            if (
                isinstance(error, torch.OutOfMemoryError)
                and self.oom_policy is OOMPolicy.ABORT
            ):
                abort_distributed(error, IN_GRAPH)
            reason = str(error)
            # a bool, not the exception: binding `error` past this clause
            # would keep its traceback, and so the failed capture's graph,
            # alive through the pool release below
            out_of_memory = isinstance(error, torch.OutOfMemoryError)
            device = graph_device(self.members[0].executor.bundle)
            torch.cuda.synchronize(device)
            self._release()
            self.disabled = True
            for item in window:
                item.minibatch.reset_reads()
        # Outside the except block, so the traceback no longer retains the
        # partially built replay, its tensors or its graph (live to the pool
        # until then) before the cache is released.
        gc.collect()
        if out_of_memory:
            # the eager cohort needs the working set back: release the pool
            # unless the evaluation graph or inference replays still hold it.
            # A layout mismatch is not memory pressure; the pool stays for them.
            self.pool.close_if_unused()
        torch.cuda.empty_cache()
        _log.info(
            "CUDA cohort graph released; remaining passes run eagerly: %s", reason
        )
        self.eager_steps += 1
        return False

    def close(self) -> None:
        """Release the workers and the graph, and a pool the bank opened
        itself; a member's gradients pointed into the graph's pool, so they
        are dropped too."""
        self._release()
        if self._owns_pool:
            self.pool.close()

    def _release(self) -> None:
        if self.slots:
            torch.cuda.synchronize(graph_device(self.members[0].executor.bundle))
        for slot in self.slots:
            slot.worker.close()
        self.slots = []
        self.frames = {}
        self.prefix = {}
        self.replay = None
        for member in self.members:
            for parameter in member.parameters:
                parameter.grad = None

    # ------------------------------------------------------------------ #

    def _capture(self, window: Sequence[WindowItem]) -> None:
        items = sorted(window, key=lambda item: self.by_key[item.key])
        device = graph_device(self.members[0].executor.bundle)
        parameters: list[torch.nn.Parameter] = []
        stages: dict[str, Stage] = {}
        offset = 0
        for item in items:
            index = self.by_key[item.key]
            member = self.members[index]
            point = member.executor
            padded = padded_indices(item.indices, member.pairs)
            frames = {role: point.frame(role) for role in point.role_rows}
            worker = GraphExecutor(
                point.doc,
                point.bundle,
                role_rows={
                    role: [rows[i] for i in padded]
                    for role, rows in point.role_rows.items()
                },
                role_fields=point.role_fields,
                load_tensors=point.load_tensors,
                load_table=point.load_table,
                stage_cache=copy.deepcopy(point.stage_cache),
                grad_enabled=True,
                batches={role: frame.select(padded) for role, frame in frames.items()},
                coords=point.coords,
            )
            # the one layout-checked copy; every step after stages the
            # tensors pairwise (`_Slot.state`)
            copy_executor_stages(worker, item.minibatch)
            weight = slot_weights(len(item.indices), member.pairs, device)
            objective = self.make_objective(
                worker,
                {name: worker.stage(name) for name in member.stages},
                weight=weight,
            )
            worker.prepare()
            # the member's parameters, in its optimizer's order, as the
            # worker holds them — the tensors the captured backward writes
            names = {
                id(value): (name, key)
                for name, stage in point.stage_cache.items()
                for key, value in stage.named_parameters()
            }
            held = {
                (name, key): value
                for name, stage in worker.stage_cache.items()
                for key, value in stage.named_parameters()
            }
            captured = [held[names[id(p)]] for p in member.parameters]
            start = len(parameters)
            parameters.extend(captured)
            state: _Copies = []
            gates: list[tuple[Gate, Gate]] = []
            for name, stage in worker.stage_cache.items():
                stages[f"{index}/{name}"] = stage
                source = point.stage(name)
                held_state = dict(stage.named_parameters()) | dict(
                    stage.named_buffers()
                )
                source_state = dict(source.named_parameters()) | dict(
                    source.named_buffers()
                )
                state.extend((held_state[key], source_state[key]) for key in held_state)
                if isinstance(stage, Gate) and isinstance(source, Gate):
                    gates.append((stage, source))
            self.slots.append(
                _Slot(
                    member=member,
                    index=index,
                    worker=worker,
                    objective=objective,
                    weight=weight,
                    rows=slice(offset, offset + member.pairs),
                    gradient_slice=slice(start, len(parameters)),
                    operands=_operand_groups(worker),
                    state=state,
                    gates=gates,
                )
            )
            offset += member.pairs

        workers = [slot.worker for slot in self.slots]
        objectives = [slot.objective for slot in self.slots]
        reads = [tuple(slot.member.objective_reads) for slot in self.slots]
        # the cohort forward's frame per role, built once: the graph reads its
        # storage, and the lead worker holds its prepared masks under its id
        lead = workers[0]
        role = _single_role(list(zip(workers, reads)))
        if role is None:
            raise _LayoutMismatch(
                "a captured cohort needs one input role with one group per member"
            )
        frame = _concat_frames([worker._batch(role) for worker in workers])
        lead._masks[id(frame)], lead._position_ids[id(frame)] = lead.prepare_batch(
            frame
        )
        self.frames[role] = frame
        # every worker's source activations over its slot — from the store
        # when the members have one, else computed by the worker — and the
        # first minibatch's values laid in; then the prefix the forward
        # resumes from, computed for this composition where the store lacks it
        for slot, item in zip(self.slots, items, strict=True):
            self._seed_sources(slot, item.minibatch)
        self.resume_at = self._resume_depth(items)
        prepared = [self._prepare(slot, item) for slot, item in zip(self.slots, items)]
        with torch.no_grad():
            for slot, one in zip(self.slots, prepared, strict=True):
                # Workers and the concatenated frame already hold this step's
                # tokens; only subsequent replays need to stage one.tokens.
                for target, source in one.values:
                    target.copy_(source)
                self._stage_masks(slot, one)
        # The copies above intentionally restaged these frozen inputs. Accept
        # their new versions before warmup reads the worker's source buffers.
        for worker in workers:
            worker._frozen_versions = worker._input_versions()
        if self.resume_at:
            self._ensure_prefixes(list(zip(self.slots, items, prepared)))
            self._stage_prefixes(list(zip(self.slots, prepared)))

        def work() -> torch.Tensor:
            for worker in workers:
                worker.reset_reads()
            by_role = cohort_entries(list(zip(workers, reads)))
            for role, entries in by_role.items():
                resume = (
                    Resume(start=self.resume_at, cached=self.prefix[role], store={})
                    if role in self.prefix
                    else None
                )
                run_groups(entries, frame=self.frames[role], resume=resume)
            loss = torch.zeros((), device=device)
            for objective in objectives:
                loss = loss + objective()
            loss.backward()
            return loss.detach()

        self.replay = Replay(
            work,
            stages,
            device=device,
            parameters=parameters,
            pool=self.pool,
        )
        _log.info(
            "cohort graph captured: %d members, %d rows, resumes at block %d",
            len(self.slots),
            sum(slot.member.pairs for slot in self.slots),
            self.resume_at,
        )

    def _step(self, window: Sequence[WindowItem]) -> bool:
        assert self.replay is not None
        active: dict[int, WindowItem] = {self.by_key[item.key]: item for item in window}
        staged: list[tuple[_Slot, WindowItem, _Prepared]] = []
        for slot in self.slots:
            item = active.get(slot.index)
            if item is None:
                continue  # exhausted: its slot replays stale and unread
            prepared = slot.prepared.get(id(item.minibatch))
            if prepared is None:
                prepared = self._prepare(slot, item)
            self._stage(slot, prepared)
            staged.append((slot, item, prepared))
        if self.resume_at:
            # after the tokens: a missing prefix is computed over the frame
            # as it now stands, then every active slot's is laid in
            self._ensure_prefixes(staged)
            self._stage_prefixes([(slot, prepared) for slot, _, prepared in staged])
        self.replay()
        self.replays += 1
        # the gradients sit in the fit's shared pool, where the evaluation
        # graph's replay may reuse their blocks: the loop steps every member's
        # optimizer before it evaluates anyone
        gradients = self.replay.gradients
        for slot in self.slots:
            if slot.index not in active:
                continue
            for parameter, gradient in zip(
                slot.member.parameters, gradients[slot.gradient_slice], strict=True
            ):
                parameter.grad = gradient
        return True

    def _stage(self, slot: _Slot, prepared: _Prepared) -> None:
        """Copy prepared minibatch data and current parameters into fixed storage."""
        with torch.no_grad():
            for target, source in (*prepared.tokens, *prepared.values):
                target.copy_(source)
            self._stage_masks(slot, prepared)
            for target, source in slot.state:
                target.copy_(source)
        for held, source in slot.gates:
            held.temperature = source.temperature

    @staticmethod
    def _stage_masks(slot: _Slot, prepared: _Prepared) -> None:
        for role, (key, copies) in prepared.masks.items():
            if slot.staged_masks.get(role) == key:
                continue
            for target, source in copies:
                target.copy_(source)
            slot.staged_masks[role] = key

    # ------------------------------------------------------------------ #
    # the constants: source activations and the resume prefix

    def _sources(
        self, minibatch: PointExecutor, operands: Mapping[ReadRef, _Group]
    ) -> dict[_Group, tuple[dict[TapKey, torch.Tensor], dict[TapKey, torch.Tensor]]]:
        """The raw captures (and routing) of the operand groups over
        ``minibatch``'s rows: from the store when the member has one — the
        pass shared across the cohort — else the graph executor's own frozen
        cache, which computed them."""
        assert isinstance(minibatch, GraphExecutor)
        with torch.no_grad():
            for operand in operands:
                minibatch.read_value(operand)
        minibatch.reset_reads()
        out: dict[
            _Group, tuple[dict[TapKey, torch.Tensor], dict[TapKey, torch.Tensor]]
        ] = {}
        for group in set(operands.values()):
            taps = _group_taps(minibatch, group)
            if minibatch.interning is None:
                capture, routing, _ = minibatch._frozen["sources"][group]
                out[group] = (
                    {key: capture[key] for key in taps},
                    {key: routing[key] for key in taps if key in routing},
                )
                continue
            group_key = minibatch._group_key(*group)
            assert group_key is not None
            cache = minibatch.interning.cache
            key = minibatch._capture_key(group_key)
            captured = cache.captured.get(key)
            if captured is None or any(k not in captured for k in taps):
                raise _LayoutMismatch(f"the store holds no capture of {group}")
            routing = cache.routing.get(key, {})
            out[group] = (
                {k: captured[k] for k in taps},
                {k: routing[k] for k in taps if k in routing},
            )
        return out

    def _seed_sources(self, slot: _Slot, minibatch: PointExecutor) -> None:
        """Give the worker its frozen source entries — its own storage, the
        slot's rows — so its captured source reads are served from buffers
        the steps then stage into (``GraphExecutor._forward_group``)."""
        worker = slot.worker
        total = slot.member.pairs
        real = len(minibatch.rows_for_metrics())
        for group, (capture, routing) in self._sources(
            minibatch, slot.operands
        ).items():
            with torch.no_grad():
                worker._frozen["sources"][group] = (
                    {k: pad_rows(v, real, total).clone() for k, v in capture.items()},
                    {k: pad_rows(v, real, total).clone() for k, v in routing.items()},
                    None,
                )
        worker._frozen_versions = worker._input_versions()
        worker._frozen_ready = True

    def _resume_depth(self, items: Sequence[WindowItem]) -> int:
        """The block the captured forward resumes at: the shallowest
        ``resume_at`` across the members' plans — a write or tap below it on
        any member forbids skipping the block — when every member has a store
        to hold prefixes in and the model's block loop is one the swap was
        verified for; ``0`` runs the whole stack."""
        bundle = self.members[0].executor.bundle
        if not _resumable(bundle):
            return 0
        last = len(bundle.blocks) - 1
        depth = last
        for slot, item in zip(self.slots, items, strict=True):
            minibatch = item.minibatch
            if minibatch.interning is None:
                return 0
            group = _trained_group(minibatch, slot.member.objective_reads)
            plan = minibatch._prefix_plan(minibatch._group_key(*group))
            if plan is None:
                return 0
            depth = min(depth, plan.resume_at, last)
        return max(depth, 0)

    def _prefix_key_for(
        self, slot: _Slot, minibatch: PointExecutor, real: int
    ) -> PrefixKey:
        group = _trained_group(minibatch, slot.member.objective_reads)
        plan = minibatch._prefix_plan(minibatch._group_key(*group))
        assert plan is not None
        return minibatch._prefix_key(plan, RowWindow(0, real, real), self.resume_at)

    def _ensure_prefixes(
        self, staged: Sequence[tuple[_Slot, WindowItem, _Prepared]]
    ) -> None:
        """Store the resume prefix of every slice this step brings that the
        store lacks, the way the eager cohort would have on this step: one
        un-intervened pass over the frame as staged, each missing member's
        rows stored under its own key, stopped at the resume block."""
        missing = [
            (slot, item, prepared)
            for slot, item, prepared in staged
            if prepared.prefix_key is not None
            and prepared.prefix_key not in item.minibatch.interning.cache.prefixes  # type: ignore[union-attr]
        ]
        if not missing:
            return
        lead_slot, lead_item, _ = missing[0]
        minibatch = lead_item.minibatch
        assert minibatch.interning is not None
        role = _trained_group(minibatch, lead_slot.member.objective_reads)[1]
        frame = self.frames[role]
        total = int(frame.input_ids.shape[0])
        store = {
            prepared.prefix_key: slice(slot.rows.start, slot.rows.start + prepared.real)
            for slot, _, prepared in missing
            if prepared.prefix_key is not None
        }
        blocks = minibatch.bundle.blocks
        block = blocks[min(self.resume_at, len(blocks) - 1)]

        def stop(_module: Any, _args: Any) -> None:
            raise _StopForward()

        # the storing hooks are prepended by the forward; this one runs after
        handle = block.register_forward_pre_hook(stop)
        try:
            with torch.no_grad(), minibatch._attention_backend([]):
                minibatch._forward_window(
                    frame,
                    RowWindow(0, total, total),
                    write_hooks=[],
                    tapped=[],
                    depth=0,
                    resume=Resume(start=0, cached=None, store=store),
                )
        except _StopForward:
            pass
        finally:
            handle.remove()
        self.prefix_forwards += 1

    def _stage_prefixes(self, staged: Sequence[tuple[_Slot, _Prepared]]) -> None:
        (role,) = self.frames  # _capture validated a single shared input role
        for slot, prepared in staged:
            if prepared.prefix_key is None:
                continue
            item_cache = slot.member.executor.interning
            assert item_cache is not None
            if prepared.prefix is None:
                stored = item_cache.cache.prefixes[prepared.prefix_key]
                prepared.prefix = pad_rows(stored, prepared.real, slot.member.pairs)
            stored = prepared.prefix
            buffer = self.prefix.get(role)
            if buffer is None:
                total = int(self.frames[role].input_ids.shape[0])
                buffer = torch.empty(
                    (total, *stored.shape[1:]), dtype=stored.dtype, device=stored.device
                )
                self.prefix[role] = buffer
            with torch.no_grad():
                buffer[slot.rows].copy_(stored)

    def _prepare(self, slot: _Slot, item: WindowItem) -> _Prepared:
        """Everything a step stages for ``item``'s minibatch, computed once:
        its rows never change, so neither do its padded tokens, masks, labels
        or source activations, nor the store key of its resume prefix."""
        worker, minibatch = slot.worker, item.minibatch
        assert isinstance(minibatch, GraphExecutor)
        real, total = len(item.indices), slot.member.pairs
        tokens: _Copies = []
        values: _Copies = []
        masks: dict[str, tuple[_MaskKey, _Copies]] = {}
        with torch.no_grad():
            for role in worker.role_rows:
                slotted = slotted_frame(minibatch._batch(role), real, total)
                ids, mask = slotted.input_ids, slotted.attention_mask
                tokens.append((worker._batch(role).input_ids, ids))
                frame = self.frames.get(role)
                if frame is not None:
                    tokens.append((frame.input_ids[slot.rows], ids))
                # A left-padded mask is identified by its width and where
                # each row's real tokens start — the frame's cached indices,
                # so preparing a minibatch makes no host read and the staging
                # of a first epoch overlaps the previous step's device work
                # too. (A slot's width is fixed; naming it keeps the key a
                # mask's identity on its own.)
                key = (slotted.padded_len, slotted.first_reals)
                copies = slot.mask_cache.get((role, key))
                if copies is None:
                    copies = [(worker._batch(role).attention_mask, mask)]
                    if frame is not None:
                        copies.append((frame.attention_mask[slot.rows], mask))
                        prepared_masks, positions = worker.prepare_batch(slotted)
                        lead = self.slots[0].worker
                        copies.append(
                            (lead._position_ids[id(frame)][slot.rows], positions)
                        )
                        try:
                            tree_map(
                                lambda target, new: _mask_copies(
                                    copies, target, new, slot.rows
                                ),
                                lead._masks[id(frame)],
                                prepared_masks,
                            )
                        except ValueError as error:
                            raise _LayoutMismatch(
                                f"prepared mask structure: {error}"
                            ) from error
                    slot.mask_cache[(role, key)] = copies
                masks[role] = key, copies
            values.append((slot.weight, slot_weights(real, total, slot.weight.device)))
            for name, fields in slot.objective.labels.items():
                for field, value in fields.items():
                    values.append(
                        (
                            value,
                            pad_rows(item.objective.labels[name][field], real, total),
                        )
                    )
        for group, (capture, routing) in self._sources(
            minibatch, slot.operands
        ).items():
            held_capture, held_routing, _ = worker._frozen["sources"][group]
            for key, held in held_capture.items():
                source = pad_rows(capture[key], real, total)
                if source.shape != held.shape:
                    raise _LayoutMismatch("a source capture changed shape")
                values.append((held, source))
            for key, held in held_routing.items():
                values.append((held, pad_rows(routing[key], real, total)))
        prefix_key = (
            self._prefix_key_for(slot, minibatch, real) if self.resume_at else None
        )
        prepared = _Prepared(
            tokens=tokens, values=values, masks=masks, prefix_key=prefix_key, real=real
        )
        slot.prepared[id(minibatch)] = prepared
        return prepared


def _mask_copies(copies: _Copies, target: Any, source: Any, rows: slice) -> None:
    """Validate per-row prepared masks before staging any captured storage."""
    if target is None and source is None:
        return
    if not isinstance(target, torch.Tensor) or not isinstance(source, torch.Tensor):
        raise _LayoutMismatch("a prepared mask changed between a tensor and none")
    held = target[rows]
    if held.shape != source.shape:
        raise _LayoutMismatch(
            f"a prepared mask changed shape: {held.shape} / {source.shape}"
        )
    copies.append((held, source))


class EvaluationGraphs:
    """The eval passes of one cohort as one CUDA graph, replayed for whichever
    members are due.

    An eval pass is a fixed layout for the whole fit: the same split rows for
    every member, the members' own stages (the tensors the optimizer steps —
    an eval executor shares its point's stage cache), the trained group's
    forward in eval mode. It is captured **once** — on the second pass that
    brings every member, the first having warmed the store with the split's
    sources and prefixes — over the members' rows concatenated in cohort
    order, and replayed on every later pass. A member that early-stops keeps
    its slot, exactly as in [`CohortGraphs`][]: its rows still flow through
    the replay and its reads are left where they are — never copied out,
    never scored — so a membership change forces neither a recapture nor a
    fall back to the eager pass.

    **Exactness.** The captured forward is the eager cohort eval over the same
    concatenated frame — the same row count, the same kernels — so a replay's
    scores are the eager pass's while every member is due, and after a member
    stops they are the eager pass's *over the full frame*: the row count never
    moves for the members still scoring, where the eager pass shrinks its
    frame and moves bf16 rounding with it. A stopped member's rows are still
    computed on every replay — the trade the training graph makes too, whose
    stopped slots keep stepping — and it pays because the eval's wall was
    never its forward: a replay of the full frame is one launch and no host
    work, where an eager pass over fewer rows concatenates its frame on the
    host every time, and prepares its masks again whenever the shape has
    left ``_model_forward``'s four-entry transient cache (before this
    capture the eval ran ~215 ms of wall for 6 ms of GPU per pass).

    **One layout per cohort.** The layout is the first due set the fit
    evaluates, and the capture is that set's. A due set *disjoint* from it is
    another split's members: it runs as the eager cohort and leaves the
    layout be, before and after the capture — so a cohort whose members
    evaluate on two splits captures the first split's set and evaluates the
    second eagerly, per forward, as before. A due set that *overlaps* the
    layout without being it — a member stopped before any capture, or one
    on a longer ``every`` period joining — becomes the layout while nothing
    is captured yet (the next pass over it captures), and releases the
    capture once there is one, like any set the capture holds no slot for.

    **One frame, prepared once.** The frame is built and prepared once —
    masks and position ids, ``GraphExecutor.prepare_batch`` — when the layout
    is first seen, and registered under its id on the lead executor; the
    first, eager pass runs ``run_groups(frame=)`` against it
    ([`frames_for`][]) instead of a per-forward concatenation, whose masks
    ``_model_forward`` keys by shape and prepares again once its four-entry
    transient cache has evicted them (the key carries the rows' first real
    tokens, so that path makes no host read either); and the capture records
    that same storage — owned here for the fit, never a cache entry an
    eviction can drop, which is what a graph's fixed addresses need. The
    one host wait in the eval is the copy of what the scorer selects from
    the due members' read values — the answer columns, not the vocabulary
    (``train._score``) — which the scores and the early-stop decision need.

    **Memory.** Sources and prefixes stay the campaign store's; the frame is
    tokens, masks and position ids; and the graph is captured into the fit's
    [`GraphPool`][] when the caller hands one in (``pool``) — the training
    cohort's, so the eval forward's activations reuse the blocks the training
    capture freed instead of a second private pool (by default the bank opens
    a pool of its own and releases it with [`close`][]). That is safe
    because the two graphs replay in stream order and each one's outputs —
    the gradients, the read values — are consumed before the other replays
    (the two rules ``GraphPool`` states). The capture records **without a
    warm-up pass** (``Replay(warmup=False)``): the layout's first pass has
    just run the same forward eagerly, on the same frame and storage, so
    every kernel it launches is compiled, tuned and loaded, and there is no
    backward to start an autograd thread for — the warm-up would repeat a
    pass the fit has already completed, and its working set is the step
    graph's forward, already in the pool. A CUDA out-of-memory in capture
    or replay
    releases the graph and hands the remaining passes to the eager path; so
    does a set that overlaps the layout with a member the capture
    holds no slot for (a disjoint set — another split's — runs eagerly
    beside it), and so does a row bound that has fallen below the layout's
    group ([`holds`][]): the bound falls only because the device ran
    short and never rises again, and the layout's replay is the fit's
    widest eval frame, so the capture is given back rather than held
    against a later due set small enough to fit again as members stop — a
    trade of those replays for the pool blocks — and not recaptured at the
    smaller size, the churn the training graph declines too. Under
    ``oom_policy`` ``OOMPolicy.ABORT`` the out-of-memory ends the run by name
    instead, as the step graph's does ([`CohortGraphs`][]).
    """

    def __init__(
        self,
        *,
        pool: GraphPool | None = None,
        oom_policy: OOMPolicy = OOMPolicy.RETRY,
    ) -> None:
        self.oom_policy = oom_policy
        self._owns_pool = pool is None
        self.pool = GraphPool() if pool is None else pool
        #: the layout: every member the cohort evaluates together, with the
        #: reads its eval needs, in cohort order; ``None`` before the first pass
        self.layout: tuple[tuple[int, tuple[str, ...]], ...] | None = None
        self.members: list[GraphExecutor] = []
        self.frames: dict[str, EncodedBatch] = {}
        self.entries: dict[str, list[Any]] = {}
        self.bank: Replay | None = None
        self.groups: list[set[_Group]] = []
        self.disabled = False

    @staticmethod
    def _signature(
        members: Sequence[tuple[PointExecutor, Sequence[str]]],
    ) -> tuple[tuple[int, tuple[str, ...]], ...]:
        return tuple((id(ex), tuple(reads)) for ex, reads in members)

    @staticmethod
    def _capturable(members: Sequence[tuple[PointExecutor, Sequence[str]]]) -> bool:
        return (
            len(members) > 0
            and all(isinstance(ex, GraphExecutor) for ex, _ in members)
            and _single_role(members) is not None
        )

    def frames_for(
        self, members: Sequence[tuple[PointExecutor, Sequence[str]]]
    ) -> dict[str, EncodedBatch] | None:
        """The prepared frame an eager pass over ``members`` runs on — the
        layout's own, when ``members`` are the whole layout (its first pass,
        warming the store; [`forward`][] laid it out); ``None`` for any
        other set — a window of a larger group — which concatenates per
        forward as an eager cohort does."""
        if self.disabled or self.layout is None:
            return None
        return dict(self.frames) if self._signature(members) == self.layout else None

    def forward(self, members: Sequence[tuple[PointExecutor, Sequence[str]]]) -> bool:
        """Serve the due ``members``' trained-group reads from the replay;
        ``False`` when the pass runs eagerly instead — the layout's first
        pass (on [`frames_for`][]'s frame; the next full pass captures),
        a set disjoint from the layout (another split's members, the layout
        untouched), or a set the capture holds no slot for, which releases
        it."""
        if self.disabled:
            return False
        if not self._capturable(members):
            self.invalidate()
            return False
        signature = self._signature(members)
        if self.layout is not None and not set(signature) & set(self.layout):
            return False  # another split's set: eager, this layout kept
        try:
            if self.bank is None:
                if self.layout != signature:
                    # a layout not seen before — every member, or what is
                    # left of them when one stopped before any capture: this
                    # pass warms it eagerly on its frame, the next one captures
                    self._lay_out(members)
                    return False
                self._capture()
            assert self.layout is not None and self.bank is not None
            if not set(signature) <= set(self.layout):
                self.invalidate()
                return False
            reads = self.bank()
            due = {id(ex) for ex, _ in members}
            for ex, groups, values in zip(
                self.members, self.groups, reads, strict=True
            ):
                if id(ex) not in due:
                    continue  # a stopped member's slot replays stale and unread
                # graph-owned device storage, valid until the next replay on
                # the shared pool (the step graph's may reuse these blocks):
                # the scorer (`train._score`) consumes it now, gathering what
                # the metric selects and copying the columns, not the
                # vocabulary, and releases it before that replay
                ex._read_values.update(values)
                ex._groups_run.update(groups)
            return True
        except torch.OutOfMemoryError as error:
            if self.oom_policy is OOMPolicy.ABORT:
                abort_distributed(error, IN_GRAPH)
            self._release()
            self.disabled = True
        # Unwind the failed capture's frames (which keep its graph live to the
        # pool) before releasing allocator caches.
        gc.collect()
        # the eager evals need the working set back unless the step graph or
        # inference replays still hold the pool
        self.pool.close_if_unused()
        torch.cuda.empty_cache()
        _log.info(
            "CUDA cohort evaluation graph released after OOM; remaining evals run eagerly"
        )
        return False

    def _lay_out(self, members: Sequence[tuple[PointExecutor, Sequence[str]]]) -> None:
        """Adopt ``members`` as the layout: their positions resolved, their
        rows concatenated into one frame per role, prepared once and
        registered on the lead — the storage both the eager first pass and
        the capture read."""
        self._release_frames()
        self.layout = self._signature(members)
        # every member is kept: `self.members`, `self.entries` and the
        # replay's per-member reads index one and the same sequence
        # (`_capturable` held, so each one is a graph executor)
        self.members = []
        for ex, _ in members:
            assert isinstance(ex, GraphExecutor)
            self.members.append(ex)
        lead = self.members[0]
        # Store hits and batched forwards can leave non-leading executors
        # unprepared. Resolve their positions outside capture, host-side.
        for ex in self.members:
            ex.prepare()
        self.entries = dict(cohort_entries(members))
        for role, group in self.entries.items():
            frame = _concat_frames([entry.executor._batch(role) for entry in group])
            self.frames[role] = frame
            lead._masks[id(frame)], lead._position_ids[id(frame)] = lead.prepare_batch(
                frame
            )

    def _capture(self) -> None:
        lead = self.members[0]
        stages = {
            f"{index}/{name}": stage
            for index, ex in enumerate(self.members)
            for name, stage in ex.stage_cache.items()
        }
        entries, frames = self.entries, self.frames

        def work():
            with torch.no_grad():
                for ex in self.members:
                    ex.reset_reads()
                for role, group in entries.items():
                    # The first eager pass warmed the sources and stored the
                    # prefixes. Their device storage remains owned by the fit.
                    run_groups(group, frame=frames[role])
                return tuple(dict(ex._read_values) for ex in self.members)

        previous = [ex.device_reads for ex in self.members]
        try:
            for ex in self.members:
                ex.device_reads = True
            # no warm-up: the layout's first, eager pass ran this forward on
            # this frame (class docstring, "Memory")
            self.bank = Replay(
                work,
                stages,
                device=graph_device(lead.bundle),
                pool=self.pool,
                warmup=False,
            )
            self.groups = [set(ex._groups_run) for ex in self.members]
        finally:
            for ex, mode in zip(self.members, previous, strict=True):
                ex.device_reads = mode

    def holds(self, members: Sequence[tuple[PointExecutor, Sequence[str]]]) -> bool:
        """Whether the capture serves any of ``members`` — what the caller
        asks before giving it back when the row bound has fallen below
        their group (``train._evaluate``: the device ran short; the pool
        blocks are worth more than the replays a later, smaller due set
        could still take)."""
        if self.bank is None or self.layout is None:
            return False
        return bool(set(self._signature(members)) & set(self.layout))

    def invalidate(self) -> None:
        """Release an existing capture when evaluation takes another path —
        a due set the capture holds no slot for, a member the graph cannot
        hold, or a group the row bound no longer runs as one window. The
        graph, not the pool: a layout change is not the end of the fit."""
        if self.bank is not None:
            self._release()
            self.disabled = True

    def _release_frames(self) -> None:
        if self.members:
            lead = self.members[0]
            for frame in self.frames.values():
                lead._masks.pop(id(frame), None)
                lead._position_ids.pop(id(frame), None)
        self.frames.clear()
        self.entries.clear()

    def close(self) -> None:
        """Release the capture, and a pool the bank opened itself."""
        self._release()
        if self._owns_pool:
            self.pool.close()

    def _release(self) -> None:
        if self.members:
            device = graph_device(self.members[0].bundle)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            self._release_frames()
            for ex in self.members:
                ex.reset_reads()
        self.bank = None
        self.layout = None
        self.members.clear()
        self.groups.clear()
