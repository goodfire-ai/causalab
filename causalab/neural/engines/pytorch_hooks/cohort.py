"""Run compatible fits in one model forward.

A cohort concatenates its members' minibatches. Each member writes and
reads its own rows, while the forward captures the union of taps.
The summed loss differentiates into separate parameter sets. Batch shape
can affect floating-point rounding.

Prefix resume starts at the deepest block for which every member has
stored input. Captures are stored under each member's key with its rows.
This path handles changing trained groups. Constant groups use the campaign
store; decoding, dynamic operand dependencies, and per-step DeltaNet state
writes use the individual executor path. ``batchable`` checks eligibility.

Under a pipeline (model_parallelism.md §6.5, §8.3) members may write on
different stages. Each member's write fires on the stage holding its module
alone, so a member's tally is summed over the pipeline
(``agreements.summed_fires``) before its declaration is checked; the
members' parameters are synced from their owners by the training loop.
"""

# pyright: reportPrivateUsage=false
# — this module is the other half of executor.py's group forward: it composes
# PointExecutor's own steps (address resolution, hook building, the window
# forward, read finalization, prefix keys) across several executors, and those
# steps are private to the executor on purpose. Same package, one contract.

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Mapping, Sequence

import torch

from causalab.neural.engines.pytorch_hooks.executor import (
    PointExecutor,
    Resume,
    _refuse_interior,
    _resumable,
)
from causalab.neural.shared.encoding import EncodedBatch
from causalab.neural.shared.executor import (
    PrefixKey,
    PrefixPlan,
    RowWindow,
    TapKey,
    tap_key,
)
from causalab.neural.shared.fires import (
    FireTally,
    GroupFires,
    check_fires,
    group_label,
)
from causalab.neural.shared.head import ReadTap
from causalab.neural.shared.parallel.agreements import Agreements, summed_fires
from causalab.neural.shared.plan import group_reads, is_unwritten, write_names
from causalab.neural.shared.sites import ResolvedSite, resolve_site
from causalab.protocol.positions.encoding import generated_budget
from causalab.protocol.schema import ReadRef, ReadSpec, operand_reads

__all__ = ["Entry", "batchable", "cohort_entries", "groups_read_by", "run_groups"]


@dataclasses.dataclass(frozen=True)
class Entry:
    """One forward group of one executor: ``model`` on ``input_role``."""

    executor: PointExecutor
    model: str
    input_role: str


def batchable(entry: Entry) -> bool:
    """Whether ``entry``'s group may run as one sub-batch of a cohort forward
    (module docstring): an intervened model the store cannot serve, that does
    not decode, whose write operands are all reads off fit-constant models,
    and whose writes include no per-step state edit."""
    ex, model, role = entry.executor, entry.model, entry.input_role
    doc = ex.doc
    if is_unwritten(doc, model) or ex._may_intern(model):
        return False
    for ref in group_reads(doc, model, role):
        if generated_budget(doc, doc.reads[ref.read].pos) is not None:
            return False
    for ename in write_names(doc, model) or ():
        write = doc.writes[ename]
        for ref in operand_reads(doc, write.do):
            if ref.model not in ex.fit_constant_models:
                return False
        site = resolve_site(ex.bundle, doc.sites[str(write.site)])
        if site.kind == "delta" and site.interface_slot == "state":
            return False
    return True


@dataclasses.dataclass
class _Part:
    """One entry's share of the cohort forward."""

    entry: Entry
    batch: EncodedBatch
    rows: slice
    window: RowWindow
    taps: list[tuple[str, ReadSpec]]
    #: per read, the site it names, the site the forward taps for it and
    #: the projection between them (``shared/head.py``)
    read_taps: dict[str, ReadTap]
    hooks: list[tuple[ResolvedSite, Callable[..., Any]]]
    #: the written-to addresses the hooks were built for: what the owning
    #: stage recorded there is shared over the pipeline with the tally
    addresses: Mapping[Any, tuple[ResolvedSite, list[tuple[str, Any, ResolvedSite]]]]
    prefix: PrefixPlan | None
    #: this member's writers count their firings here (§4 "Fires")
    tally: FireTally


def run_groups(
    entries: Sequence[Entry],
    *,
    frame: EncodedBatch | None = None,
    resume: Resume | None = None,
) -> None:
    """Run every entry's group, all on one input role, as one forward over
    the concatenation of their rows; fill each executor's read values for
    the group as `PointExecutor._run_group` would have.

    One entry runs on its executor's own path. Several must share the model,
    the input role, the grad mode, the campaign store and the frame width —
    what [`fit_cohorts`][causalab.neural.shared.plan.fit_cohorts] promised of the points
    and the loop promised of the executors it hands in; a disagreement is a
    programming error and raises. Every entry must be [`batchable`][].

    ``frame`` is the members' concatenated batch when the caller holds a
    persistent one — a captured cohort (``graph_cohort.py``) stages tokens
    into fixed storage the graph reads; by default it is built here per call.
    ``resume`` likewise: the caller's resume decision (a captured cohort hands
    in a prefix buffer of its own), by default derived from the store here.
    """
    if not entries:
        return
    if len(entries) == 1:
        entries[0].executor._run_group(entries[0].model, entries[0].input_role)
        return
    lead = entries[0].executor
    role = entries[0].input_role
    store = lead.interning.cache if lead.interning is not None else None
    for entry in entries:
        ex = entry.executor
        if ex.bundle is not lead.bundle:
            raise AssertionError("a cohort forward runs one loaded model")
        if entry.input_role != role:
            raise AssertionError("a cohort forward runs one input role")
        if ex.grad_enabled != lead.grad_enabled:
            raise AssertionError("a cohort forward runs in one grad mode")
        if (ex.interning.cache if ex.interning is not None else None) is not store:
            raise AssertionError("a cohort forward runs against one store")
        if ex.interning is not None and ex.interning.counted:
            raise AssertionError(
                "a cohort forward runs a fit's inner passes, never a counted one"
            )
        if not batchable(entry):
            raise AssertionError(f"group {entry.model!r} on {role!r} is not batchable")

    parts: list[_Part] = []
    offset = 0
    for entry in entries:
        ex, model = entry.executor, entry.model
        ex.check_write_widths()
        taps = [
            (ref.read, ex.doc.reads[ref.read])
            for ref in group_reads(ex.doc, model, role)
        ]
        read_taps = ex._read_taps(model, role, taps)
        for rname, tap in read_taps.items():
            _refuse_interior(f"read {rname!r}", tap.site)
        # operands first, as the executor's own path does — the source
        # groups, served from the store or run once per slice
        names = write_names(ex.doc, model) or ()
        for ename in names:
            for ref in operand_reads(ex.doc, ex.doc.writes[ename].do):
                ex.read_value(ref)
        addresses = ex._resolve_write_addresses(names)
        for site, _ in addresses.values():
            _refuse_interior(f"write at {site.component!r}", site)
        batch = ex._batch(role)
        size = int(batch.input_ids.shape[0])
        window = RowWindow(0, size, size)
        tally = FireTally()
        parts.append(
            _Part(
                entry=entry,
                batch=batch,
                rows=slice(offset, offset + size),
                window=window,
                taps=taps,
                read_taps=read_taps,
                hooks=ex._build_write_hooks(addresses, role, batch, window, tally),
                addresses=addresses,
                prefix=ex._prefix_plan(ex._group_key(model, role)),
                tally=tally,
            )
        )
        offset += size
    if len({part.batch.padded_len for part in parts}) != 1:
        raise AssertionError("cohort members are encoded in different frames")

    if frame is None:
        frame = _concat_frames([part.batch for part in parts])
    elif int(frame.input_ids.shape[0]) != offset:
        raise AssertionError("the caller's frame does not hold the members' rows")
    total = offset
    # one writer per address, dispatching to each member on its own rows —
    # the tensor a hook sees is the whole batch's, the slice a member's
    # writer edits is a view into it
    address_sites: dict[TapKey, ResolvedSite] = {}
    writers: dict[TapKey, list[tuple[slice, Callable[..., Any]]]] = {}
    for part in parts:
        for site, fn in part.hooks:
            address_sites.setdefault(tap_key(site), site)
            writers.setdefault(tap_key(site), []).append((part.rows, fn))
    write_hooks = [
        (address_sites[key], _dispatching(fns)) for key, fns in writers.items()
    ]
    tapped: list[ResolvedSite] = []
    seen: set[TapKey] = set()
    for part in parts:
        for tap in part.read_taps.values():
            if tap_key(tap.capture) not in seen:
                seen.add(tap_key(tap.capture))
                tapped.append(tap.capture)

    # the attention backend for this forward (eager when any tap or write is
    # an attention interior; the executor's own `_forward_group` does the same
    # around its windows), entered before the resume is derived: the prefix
    # keys carry the backend they were computed under
    with lead._attention_backend([*tapped, *(site for site, _ in write_hooks)]):
        capture, idx_capture, _ = lead._forward_window(
            frame,
            RowWindow(0, total, total),
            write_hooks=write_hooks,
            tapped=tapped,
            depth=0,
            prefix=None,
            resume=resume
            if resume is not None
            else lambda: _cohort_resume(parts, lead),
        )

    # the count compared to each member's declaration is the whole
    # pipeline's (docs/model_parallelism.md §6.5): a member's write fires on
    # the stage holding its module alone, and one cohort's members may sit
    # on different stages, so each member's tally is summed over the stages
    # before the check — in member order, the same on every rank, since the
    # entries are the campaign's points in campaign order; the identity at
    # world 1. A state writer, whose steps the context ranks partition
    # (`whole_steps`), is never batchable.
    agreements = Agreements(lead.fragments.collective)
    # every agreement first, every check after: a check that refuses on one
    # rank must not leave another rank waiting on a later member's collective
    tallies: list[FireTally] = []
    for part in parts:
        tallies.append(summed_fires(part.tally, agreements))
        # the owner's routing-mismatch counts beside its tally (§6.5), the
        # member's own stage forward being the lead's
        ex = part.entry.executor
        ex.share_routing_mismatch(part.addresses, ex._stages)
    for part, tally in zip(parts, tallies):
        ex = part.entry.executor
        # every member's writers fired the count their kind declares in this
        # one forward, or the member's point is refused — the same check the
        # solo path makes per window (§4 "Fires"); the record is the member's
        label = group_label(part.entry.model, role)
        check_fires(label, tally)
        fires = GroupFires()
        fires.fold(tally)
        record = fires.record()
        if record:
            ex.fires[(part.entry.model, role)] = record
        for rname, read in part.taps:
            tap = part.read_taps[rname]
            key = tap_key(tap.capture)
            idx = idx_capture.get(key)
            ref = ReadRef(rname, part.entry.model)
            ex._read_values[ref] = ex._finalize_read(
                ref,
                read,
                tap.site,
                capture[key][part.rows],
                part.batch,
                role,
                project=tap.project,
                expert_idx=None if idx is None else idx[part.rows],
            )
        ex._groups_run.add((part.entry.model, role))


def _dispatching(
    writers: Sequence[tuple[slice, Callable[..., Any]]],
) -> Callable[..., None]:
    def apply(tensor: torch.Tensor, routing: torch.Tensor | None = None) -> None:
        for rows, fn in writers:
            fn(tensor[rows], None if routing is None else routing[rows])

    return apply


def _concat_frames(batches: Sequence[EncodedBatch]) -> EncodedBatch:
    """The members' frames as one batch, in member order — one padded width,
    so the tensors concatenate and every per-row field appends."""
    return EncodedBatch(
        texts=tuple(text for batch in batches for text in batch.texts),
        input_ids=torch.cat([batch.input_ids for batch in batches]),
        attention_mask=torch.cat([batch.attention_mask for batch in batches]),
        offset_mapping=tuple(row for batch in batches for row in batch.offset_mapping),
        prefix_lengths=tuple(
            length for batch in batches for length in batch.prefix_lengths
        ),
        segments=(
            tuple(row for batch in batches for row in batch.segments)
            if any(batch.segments for batch in batches)
            else ()
        ),
        first_reals=tuple(index for batch in batches for index in batch.first_reals),
    )


def _cohort_resume(parts: Sequence[_Part], lead: PointExecutor) -> Resume:
    """Where the cohort forward starts and what it stores (§4 "Resume"),
    over every member's rows.

    The start is the **deepest block every member holds a stored prefix
    for**, at or below each member's own ceiling — a member writing at layer
    20 beside one writing at layer 12 starts at 12 (its layer-20 prefix is
    no use to a forward that must run block 12 for its neighbour), and the
    residuals handed to block 12 are the members' own, concatenated. Each
    member then stores every wanted depth up to its own ``write_depth`` the
    store lacks, under its own key and over its own rows, exactly as its
    solo pass would (`PointExecutor._prefix_window`).
    """
    nothing = Resume(start=0, cached=None, store={})
    if lead.interning is None or not _resumable(lead.bundle):
        return nothing
    cache = lead.interning.cache
    last = len(lead.bundle.blocks) - 1
    candidates: list[set[int]] = []
    for part in parts:
        plan = part.prefix
        if plan is None:
            candidates.append(set())
            continue
        ceiling = min(plan.resume_at, last)
        held: set[int] = set()
        for depth in cache.wanted_prefix_depths.get(plan.base_key, ()):
            if not 0 < min(depth, last) <= ceiling:
                continue
            key = part.entry.executor._prefix_key(plan, part.window, depth)
            if key in cache.prefixes:
                held.add(min(depth, last))
        candidates.append(held)
    common = set.intersection(*candidates) if candidates else set()
    start = max(common) if common else 0
    cached: torch.Tensor | None = None
    if start:
        pieces: list[torch.Tensor] = []
        for part in parts:
            plan = part.prefix
            assert plan is not None
            # the plan depth that names block `start` — the wanted depth
            # itself, or a past-every-block depth clamped to the last block
            depth = next(
                d
                for d in sorted(cache.wanted_prefix_depths[plan.base_key])
                if min(d, last) == start
                and part.entry.executor._prefix_key(plan, part.window, d)
                in cache.prefixes
            )
            pieces.append(
                cache.prefixes[
                    part.entry.executor._prefix_key(plan, part.window, depth)
                ]
            )
        cached = torch.cat(pieces)
    store: dict[PrefixKey, slice | None] = {}
    for part in parts:
        plan = part.prefix
        if plan is None:
            continue
        for depth in sorted(cache.wanted_prefix_depths.get(plan.base_key, ())):
            if min(depth, last) < max(start, 1) or depth > plan.write_depth:
                continue
            if cache.prefix_owed.get((plan.base_key, depth), 0) <= 0:
                continue
            key = part.entry.executor._prefix_key(plan, part.window, depth)
            if key not in cache.prefixes and key not in store:
                store[key] = part.rows
    return Resume(start=start, cached=cached, store=store)


def groups_read_by(doc: Any, reads: Sequence[ReadRef | str]) -> list[tuple[str, str]]:
    """The ``(model, input)`` groups the bound reads are taken on, in first
    appearance order — what a loss or an eval pass needs run. A bare name is
    the read's one binding, as on the executor surface."""
    out: list[tuple[str, str]] = []
    for read in reads:
        ref = doc.bound(read) if isinstance(read, str) else read
        group = doc.group_of(ref)
        if group not in out:
            out.append(group)
    return out


def cohort_entries(
    executors: Sequence[tuple[PointExecutor, Sequence[ReadRef | str]]],
) -> Mapping[str, list[Entry]]:
    """The batchable entries of several executors, by input role: for each
    executor, the groups its named reads need that [`batchable`][] admits.
    Whatever is left out runs on the executor's own path when the read is
    asked for."""
    by_role: dict[str, list[Entry]] = {}
    for executor, reads in executors:
        for model, role in groups_read_by(executor.doc, reads):
            entry = Entry(executor=executor, model=model, input_role=role)
            if batchable(entry):
                by_role.setdefault(role, []).append(entry)
    return by_role
