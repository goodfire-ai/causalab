"""Build forward groups and identify reusable work.

Each point runs the original model on inputs where it is read, plus each
intervened model on its declared input. The plan derives ``num_forwards``.
A forward group's structural key contains its model, active writes,
operand dependency closure, and input binding as canonical JSON.
Matching keys share a group across points. Engines choose fusion, batching,
and staging when executing the plan.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Iterable, Mapping, Sequence

from causalab.protocol.schema.explicit import canonical_model_ref
from causalab.protocol.rules.errors import ProtocolError
from causalab.protocol.registry import FAMILIES
from causalab.protocol.schema import (
    METRIC_DOMAINS,
    Document,
    PositionSpec,
    ReadRef,
    operand_params,
    operand_reads,
)
from causalab.protocol.positions.alignment import (
    COMPONENT_RANK,
    PAST_BLOCKS,
    UNRANKED,
    site_depth,
    site_depths,
    static_alignment,
)
from causalab.protocol.lowering import lower_bands
from causalab.protocol.positions.encoding import generated_budget
from causalab.protocol.positions.roles import input_roles

__all__ = [
    "COMPONENT_RANK",
    "PAST_BLOCKS",
    "UNRANKED",
    "ForwardGroup",
    "GroupKey",
    "Materialization",
    "PointPlan",
    "Tap",
    "bound_read",
    "cohort_key",
    "fit_cohorts",
    "fit_constant_models",
    "generated_budget",
    "group_reads",
    "interned_groups",
    "is_unwritten",
    "plan_point",
    "closure_digest",
    "read_label",
    "saved_raw_reads",
    "write_names",
    "site_depth",
    "site_depths",
    "static_alignment",
]

#: A forward group's identity (§3): the canonical JSON — sorted keys, compact
#: separators — of everything that determines the group's activations (network
#: realization, model closure, input role, data digest + field, segments). A
#: structural key, not a hash: two groups intern iff their keys are equal, and
#: the key never leaves the process (no receipt, record or pin carries one), so
#: there is nothing to keep short.
GroupKey = str


#: Intra-block execution order of the component vocabulary — engine-free
#: data used only to find a group's deepest tap (elision, §4). ``ln_final``
#: and ``lm_head`` sort after every block.
#:
#: **Numbered in hundreds, with the attention band deliberately spread.** The
#: values are ordinal — only their order is ever read, never the numbers — but
#: changing one changes group elision, therefore every closure digest, and the
#: operand-reachability comparison (§5.21, via [`site_depth`][]); so the point
#: of the spacing is that inserting a component never renumbers an existing one.
#: The attention interior is where the vocabulary grew (the pre-RoPE
#: projections, the gate, the post-RoPE q/k, the scores, the mixer output and
#: the per-head result), and the reserved slots below say where each goes so
#: that each addition is an insertion rather than a re-pin.
#:
#: Every reserved slot is now claimed; the gaps that remain are for whatever
#: the MoE and DeltaNet interiors need.
@dataclasses.dataclass(frozen=True)
class Tap:
    """One value to materialize in a group's forward: a read's address."""

    read: str
    site: str
    depth: tuple[int, int]  # (layer, component rank) — the elision key


@dataclasses.dataclass(frozen=True)
class Materialization:
    """What one continuation read obliges an engine to build.

    ``needs_distribution`` is the expensive bit: a vocabulary-wide tensor
    per addressed position. It is false when the read is neither saved nor
    reduced by a metric that consumes distributions, in which case the
    engine must not build one (§8). *How* it avoids building one — a
    narrowed projection, per-step captures, a replay — is the engine's
    choice; this is the requirement, not the mechanism."""

    read: str
    site: str
    needs_distribution: bool


@dataclasses.dataclass(frozen=True)
class ForwardGroup:
    """One forward pass: a model (original or an IM) on one input role.

    ``key`` is the structural identity of everything that determines this
    group's activations ([`GroupKey`][]) — equal keys across points mean
    one shared forward. ``decode_depth`` is the greedy budget this group must
    decode for (0 = prefill only), and ``materialize`` states what its
    continuation reads oblige — both derived, never authored (§6).

    ``base_key`` is the key the same input role has under **no writes** —
    the identity of the un-intervened prefix of this forward (§4 "Resume").
    Every model on one input shares it, whatever it writes, and an unwritten
    model with no interior read has ``key == base_key``, so a pass of either
    kind can leave the residual entering a block behind for the others.
    ``write_depth`` is the shallowest block any in-force write lands in,
    [`PAST_BLOCKS`][] when none does (an unwritten model, or writes past
    the last block): blocks strictly below it — and the residual entering it
    — are exactly what the un-intervened forward computes on the same rows.
    ``unwritten`` says the group's model lands no write at all
    ([`is_unwritten`][]): it is what the prefix *is*, and never resumes."""

    model: str
    input: str
    taps: tuple[Tap, ...]
    key: GroupKey
    base_key: GroupKey
    write_depth: int
    unwritten: bool = False
    decode_depth: int = 0
    materialize: tuple[Materialization, ...] = ()

    @property
    def stop_after(self) -> tuple[int, int] | None:
        """The deepest tap's depth — an engine may end the forward there
        (§4 elision). ``None`` when the group has no taps, and also when it
        decodes: every decode step needs the head, so there is nothing to
        elide.

        Read this off the *interned* group, not a single point's, whenever
        several points share the forward: the shared pass has to reach every
        tap any of them asked for, so the depth it may stop at is the deepest
        of the union. [`interned_groups`][] builds exactly that group."""
        if self.decode_depth:
            return None
        return max((tap.depth for tap in self.taps), default=None)

    @property
    def resume_at(self) -> int:
        """The block an engine may *start* this forward at (§4 "Resume"),
        given the residual entering it from an un-intervened pass over the
        same rows — the mirror image of [`stop_after`][]. 0 means never.

        The shallowest block any write **or any tap** touches: below the
        first write the forward is the un-intervened one, but a tap below it
        still needs its block to run, so a read at layer 1 pins a write at
        layer 20 to block 1. An unwritten model never resumes — it is what
        the prefix *is* — and neither does a group that decodes (§4: nothing
        is elided there either). Past every block ([`PAST_BLOCKS`][]) is left for the
        engine to clamp, as it alone knows the model's depth.

        Like ``stop_after``, read this off the *interned* group: the one pass
        a shared key earns serves every sharer's taps, so the block it may
        start at is the shallowest of the union, not of the point that ran
        first. [`interned_groups`][] builds exactly that group."""
        if self.unwritten or self.decode_depth:
            return 0
        return min([self.write_depth, *(tap.depth[0] for tap in self.taps)])


@dataclasses.dataclass(frozen=True)
class PointPlan:
    """The derived execution shape of one concrete compiled intervention."""

    groups: tuple[ForwardGroup, ...]

    @property
    def num_forwards(self) -> int:
        return len(self.groups)


def plan_point(
    doc: Document, *, data_identity: Mapping[str, Any] | None = None
) -> PointPlan:
    """Derive the forward groups of one concrete document.

    ``data_identity`` (input role → the identity of the rows that role
    reads: the canonical form's content digest of the selected rows, §2.2,
    plus the field — never the ref's name) folds the input data into group
    keys so two points reading different data never intern together, and
    two points reading the same rows under two names do; omit it for a
    purely structural plan.

    A band site (§2.4 ``layers``) is planned as its per-layer members
    ([`lower_bands`][]) — the taps, depths and resume point are the
    hand-written N-site document's, which is what the executor runs."""
    doc = lower_bands(doc)
    groups: list[ForwardGroup] = []
    seen: set[tuple[str, str]] = set()
    # the un-intervened model, once per input it is read on (§4), in read
    # declaration order
    for ref in doc.read_refs():
        model, role = doc.group_of(ref)
        if model not in doc.intervened_models and (model, role) not in seen:
            seen.add((model, role))
            groups.append(_build_group(doc, model, role, data_identity))
    for im_name, im in doc.intervened_models.items():
        groups.append(_build_group(doc, im_name, str(im.input), data_identity))
    return PointPlan(groups=tuple(groups))


def interned_groups(plans: Iterable[PointPlan]) -> tuple[ForwardGroup, ...]:
    """The campaign's forward groups once §3's content-dedup is applied:
    groups sharing a ``key`` merge into **one**, whose taps are the union
    of theirs.

    ``sum(p.num_forwards for p in plans)`` is what a per-point loop pays;
    ``len(interned_groups(plans))`` is what the campaign actually owes. For a
    32-layer × 2-position interchange scan that is 65 rather than 128 — the
    64 patched forwards are genuinely distinct, but the counterfactual
    harvest depends on nothing swept, so its 64 instances become one forward
    carrying 32 taps (the position axis moves the gather, not the pass).
    Taps are absent from the key precisely so this falls out
    of value identity (reading layer 3 or layer 23 of the same un-intervened
    forward is the same forward), which is also why the merged group must
    carry the union: the one pass it earns has to serve every point.

    A merged group's ``decode_depth`` is the deepest any sharer needs, since
    a decode changes what the group produces rather than what its prefill
    computes. Its ``model``/``input`` come from the first sharer — equal
    keys mean equal model *closures*, so two intervened models that differ
    only in name merge, and either name describes the pass.

    Callers build each [`PointPlan`][] with its own ``data_identity``, so
    points reading different data never merge here.
    """
    merged: dict[GroupKey, ForwardGroup] = {}
    for plan in plans:
        for group in plan.groups:
            first = merged.get(group.key)
            if first is None:
                merged[group.key] = group
                continue
            taps = list(first.taps)
            taps.extend(tap for tap in group.taps if tap not in taps)
            seen = {item.read for item in first.materialize}
            materialize = list(first.materialize)
            materialize.extend(
                item for item in group.materialize if item.read not in seen
            )
            merged[group.key] = dataclasses.replace(
                first,
                taps=tuple(taps),
                decode_depth=max(first.decode_depth, group.decode_depth),
                materialize=tuple(materialize),
            )
    return tuple(merged.values())


def _build_group(
    doc: Document,
    model: str,
    input_role: str,
    data_identity: Mapping[str, Any] | None,
) -> ForwardGroup:
    taps = tuple(
        Tap(
            read=ref.read,
            site=str(doc.reads[ref.read].site),
            depth=site_depth(doc, str(doc.reads[ref.read].site)),
        )
        for ref in group_reads(doc, model, input_role)
    )
    identity = dict(data_identity or {})
    key = _group_key(doc, model, input_role, identity)
    # the same input under no writes: the identity of everything this
    # forward computes below its first write (§4 "Resume"). Equal to `key`
    # for an unwritten model with no interior read
    base_key = _group_key(doc, model, input_role, identity, prefix=True)
    # The key stays activation-identity: a decode changes what the group
    # *produces*, not what its prefill computes — the same reason taps are
    # not in it. Two points that differ only in decode depth share a prefill.
    depth = 0
    materialize: list[Materialization] = []
    for ref in group_reads(doc, model, input_role):
        read = doc.reads[ref.read]
        budget = generated_budget(doc, read.pos)
        if budget is None:
            continue
        depth = max(depth, budget)
        materialize.append(
            Materialization(
                read=ref.read,
                site=str(read.site),
                needs_distribution=_needs_distribution(doc, ref),
            )
        )
    return ForwardGroup(
        model=model,
        input=input_role,
        taps=taps,
        key=key,
        base_key=base_key,
        write_depth=_write_depth(doc, model),
        unwritten=is_unwritten(doc, model),
        decode_depth=depth,
        materialize=tuple(materialize),
    )


def _attention_requirement(
    doc: Document, model: str, input_role: str
) -> dict[str, str]:
    """Interior reads/writes change the implementation, not merely the taps.

    Include continuation reads too: prefill and decode must use one backend.
    The requirement follows operand closures so downstream edits cannot share
    results produced from different source implementations.
    """
    interiors = {
        component
        for adapter in FAMILIES.values()
        for component, tap in adapter.taps.items()
        if tap.slot is not None and tap.kind not in {"delta", "experts"}
    }
    sites = [doc.reads[ref.read].site for ref in group_reads(doc, model, input_role)]
    sites.extend(doc.writes[name].site for name in write_names(doc, model) or ())
    if any(
        isinstance(doc.sites[str(site)].component, str)
        and doc.sites[str(site)].component in interiors
        for site in sites
    ):
        return {"attention_requirement": "eager"}
    return {}


def _group_key(
    doc: Document,
    model: str,
    input_role: str,
    identity: Mapping[str, Any],
    *,
    prefix: bool = False,
) -> GroupKey:
    """One forward group's [`GroupKey`][]: the canonical JSON of its
    closure. The body below *is* the identity — every member is there because
    changing it changes the activations, and nothing else is.

    ``prefix`` asks for the key of the **un-intervened prefix** on the same
    input instead ([`ForwardGroup.base_key`][causalab.neural.shared.plan.ForwardGroup.base_key]): the closure with no
    writes and no attention requirement — what every model on the input
    computes below its first write."""
    body = {
        # the *realization*, not just the name: `canonical_model_ref` is the
        # canonical form's own function, so dtype and the quantization block
        # are in the group's identity exactly as they are in the document's
        "network": canonical_model_ref(doc.model),
        **({} if prefix else _attention_requirement(doc, model, input_role)),
        "model": _prefix_closure(input_role, identity)
        if prefix
        else _model_closure(doc, model, input_role, set(), identity),
        "input": input_role,
        "data": identity.get(input_role),
        # the frame the rows are encoded in (§2.2.1): a `segments.frame: chat`
        # row is rendered through the tokenizer's chat template before it is
        # tokenized, so the same rows under the same model are a different
        # token sequence — a different activation under what would otherwise
        # be the same key. Campaign-invariant today (the section takes no
        # sweep wrapper), so this changes nothing within a request; it makes
        # "equal key, equal tokens" a property of the key rather than of the
        # parser.
        "segments": doc.segments,
    }
    return json.dumps(body, sort_keys=True, separators=(",", ":"), default=_encode)


def bound_read(doc: Document, name: str) -> ReadRef:
    """Read ``name`` bound to the one model that lists it — the
    [`ReadRef`][causalab.protocol.schema.types.ReadRef] every runtime table is keyed
    by (§2.7): a read is one address, and the model that lists it decides
    which forward the address is gathered from. A read listed by several
    models has no one binding and is refused: name the model."""
    ref = doc.bound(name)
    if ref.model is None:
        raise ProtocolError(
            "P2",
            f"read {name!r} is taken on {len(doc.models_of(name))} models "
            f"({', '.join(doc.models_of(name)) or 'none'}) — a bare read name "
            "binds only when exactly one model lists it; name the model",
        )
    return ref


def read_label(ref: ReadRef) -> str:
    """The spelling one bound read has in a record: ``<model>/<read>``. The
    read alone is not a value — the same read on two models is two tensors."""
    return f"{ref.model}/{ref.read}"


def saved_raw_reads(doc: Document) -> frozenset[ReadRef]:
    """The bound reads a ``save`` entry writes **as tensors** (§2.12): the
    reads whose whole value some consumer asks for, as opposed to a
    reduction over it."""
    return doc.saved_raw_reads()


def is_unwritten(doc: Document, model: str) -> bool:
    """Whether ``model`` lands no write: the un-intervened model, or a
    declared model whose write list is empty. The predicate every "is this
    the original?" question reduces to; a model whose write list is still
    under a sweep wrapper is not known to be unwritten and answers False."""
    return doc.is_unwritten(model)


def write_names(doc: Document, model: str) -> tuple[str, ...] | None:
    """The writes in force in ``model``: ``()`` for the un-intervened model,
    ``None`` while a declared model's list is still under a sweep wrapper
    (a caller that needs a point document refuses; one that tolerates
    templates reads it as "no write I can see")."""
    im = doc.intervened_models.get(model)
    if im is None:
        return ()
    return tuple(im.writes) if isinstance(im.writes, tuple) else None


def _write_depth(doc: Document, model: str) -> int:
    """The shallowest block one of ``model``'s in-force writes lands in —
    [`PAST_BLOCKS`][] for an unwritten model and for writes that touch no
    block.

    The block is [`site_depth`][]'s layer coordinate: an in-block component
    at layer ``L`` is inside block ``L`` whichever side of it the hook rides,
    so ``block_output`` at ``L`` counts as ``L`` (block ``L`` has to run for
    its output hook to fire) and ``block_input`` at ``L`` as ``L`` too; the
    two layer-less trunk components count as 0 (``embeddings``,
    ``input_ids``: before every block) and past every block (``ln_final``,
    ``lm_head``). Conservative by construction — the residual entering the
    named block is pre-write on every one of them. A write the plan cannot
    see — a ``writes`` list, a site name or a site component still under a
    sweep wrapper — is 0."""
    names = write_names(doc, model)
    if names is None:
        # "no write I can see" must not read as "past every block": that is
        # the permissive direction — a resume past a write it cannot see, and
        # post-write residuals stored as un-intervened. `fit_constant_models`
        # refuses this shape outright; the plan is also asked about template
        # documents (`plan_point` with no data identity), so this answers
        # with the depth that resumes nothing and stores nothing.
        return 0
    if not names:
        return PAST_BLOCKS
    depths: list[int] = []
    for ename in names:
        site_name = str(doc.writes[ename].site)
        if site_name not in doc.sites or not isinstance(
            doc.sites[site_name].component, str
        ):
            # the neighbouring door: `site_depth` narrows a component it
            # cannot read to the trunk, i.e. `PAST_BLOCKS` — the same
            # permissive direction, so the same answer
            return 0
        depths.append(site_depth(doc, site_name)[0])
    return min([PAST_BLOCKS, *depths])


def group_reads(doc: Document, model: str, input_role: str) -> tuple[ReadRef, ...]:
    """The bound reads gathered from the ``(model, input_role)`` forward, in
    read declaration order — a group's taps, as every engine enumerates them."""
    return tuple(
        ref for ref in doc.read_refs() if doc.group_of(ref) == (model, input_role)
    )


def _needs_distribution(doc: Document, ref: ReadRef) -> bool:
    """Whether anything downstream of the bound read consumes a full
    distribution.

    Saving the read is the obvious case. So is any aggregation in the
    ``distribution`` domain (§2.10). An ``ids`` kind does **not** count: it
    consumes the tokens the decode produced, so a text-only probe obliges no
    vocabulary projection anywhere — which is the whole point of stating the
    requirement rather than always paying it."""
    if ref in saved_raw_reads(doc):
        return True
    for agg in doc.aggregations():
        domain = METRIC_DOMAINS.get(str(agg.spec.kind), "distribution")
        if agg.read == ref and domain == "distribution":
            return True
        if agg.target == ref:
            return True
    return False


def closure_digest(
    doc: Document,
    read: "ReadRef | str",
    *,
    data_identity: Mapping[str, Any] | None = None,
) -> str:
    """The content identity of one read's value: its address plus the full
    closure of the model it reads in. Equal digests across points mean one
    shared harvest (§3).

    "The model it reads in" includes **how it is realized**: an fp32 and a bf16
    harvest of the same address are different tensors, so they are different
    content and must not share. See `_build_group` for the bug this
    closed.
    """
    ref = bound_read(doc, read) if isinstance(read, str) else read
    body = {
        "network": canonical_model_ref(doc.model),
        "read": _read_closure(doc, ref, set(), dict(data_identity or {})),
    }
    return hashlib.sha256(
        json.dumps(
            body, sort_keys=True, separators=(",", ":"), default=_encode
        ).encode()
    ).hexdigest()


def _prefix_closure(input_role: str, identity: Mapping[str, Any]) -> dict[str, Any]:
    """The closure of the un-intervened forward on ``input_role``: the same
    shape `_model_closure` gives a written model, with no writes and no
    fit dependency — so an unwritten model's key is every sharer's
    ``base_key`` by construction."""
    return {
        "input": input_role,
        "data": identity.get(input_role),
        "writes": {},
        "train": None,
    }


def _model_closure(
    doc: Document,
    model: str,
    input_role: str,
    visiting: set[str],
    identity: Mapping[str, Any],
) -> Any:
    """Everything that determines a model's activations: for a written
    model, the in-force writes and, recursively, their operand reads'
    closures; for an unwritten one, `_prefix_closure`. The validated
    acyclicity (§5.7) bounds the recursion; ``visiting`` is a belt-and-braces
    guard."""
    if is_unwritten(doc, model):
        return _prefix_closure(input_role, identity)
    if model in visiting:
        raise AssertionError(
            f"cycle through {model!r} — validation should have refused this"
        )
    im = doc.intervened_models[model]
    train_dep: Any = None
    writes: dict[str, Any] = {}
    for ename in sorted(im.writes if isinstance(im.writes, tuple) else ()):
        write = doc.writes[ename]
        operands: dict[str, Any] = {}
        # a read's closure carries the model it is taken on, so a swap fed by
        # an intervened model differs from one fed by the un-intervened forward
        for ref in operand_reads(doc, write.do):
            operands[ref.read] = _read_closure(doc, ref, visiting | {model}, identity)
        for op in operand_params(doc, write.do):
            operands[op] = _entry(doc.params, op) if op in doc.params else op
        writes[ename] = {
            "site": _entry(doc.sites, str(write.site)),
            "pos": _pos_entry(doc, write.pos),
            "featurizer": _featurizer_entry(doc, write.featurizer),
            "dims": write.dims,
            "do": {str(write.do.mechanism): write.do.payload},
            "operands": operands,
        }
        if doc.train is not None and (
            _uses_trained_featurizer(doc, write.featurizer)
            or any(
                op.split(".", 1)[0] in _trained_roots(doc)
                for op in operand_params(doc, write.do)
            )
        ):
            # a trained featurizer's weights — or a trained free tensor the
            # write consumes — are a function of the whole fit: two points
            # differing only in train.seed must never intern
            train_dep = dataclasses.asdict(doc.train)
    return {
        "input": im.input,
        "data": identity.get(str(im.input)),
        "writes": writes,
        "train": train_dep,
    }


def fit_constant_models(doc: Document) -> frozenset[str]:
    """The models whose forward a ``train`` document **cannot change**.

    A fit re-runs its groups every optimizer step, but only the groups a
    trained parameter can reach actually differ between steps. ``original``
    never does — the network's weights are frozen at load (§2.11) — and an
    intervened model does not either when every in-force write is fed by
    nothing the fit moves: no trained featurizer on the write, no
    ``train.params`` root among its param operands, and every read operand
    taken raw (through no trained featurizer) off a model that is itself
    constant. That last clause is the recursion: a swap fed by a read *on*
    ``patched`` moves whenever the rotation does, even though the swap itself
    is unfeaturized.

    Without a ``train`` section nothing moves and every model is constant.

    What consumes this: an engine's train loop, to run a constant group once
    per row slice and serve its raw capture on every later step, epoch, eval
    pass and point (§3, §4). The shipped DAS and DBM methods reduce to the
    un-intervened model alone — their one write goes through the trained
    featurizer. Over-inclusion here would hand a stale activation to a
    gradient step, which is why the exclusions are stated per dependency
    rather than by "grad enabled".
    """
    models = tuple(
        dict.fromkeys(
            (*(str(ref.model) for ref in doc.read_refs()), *doc.intervened_models)
        )
    )
    if doc.train is None:
        return frozenset(models)
    trained_roots = _trained_roots(doc)

    def constant(model: str, visiting: set[str]) -> bool:
        if is_unwritten(doc, model):
            return True
        if model in visiting:
            raise AssertionError(
                f"cycle through {model!r} — validation should have refused this"
            )
        names = write_names(doc, model)
        if names is None:
            # "no writes I can see" must not read as "constant": that is the
            # unsafe direction, so a non-point document is refused loudly
            raise AssertionError(
                f"{model!r}.writes is unexpanded — expansion should have "
                "produced a point document before anything asks what a fit "
                "cannot change"
            )
        for ename in names:
            write = doc.writes[ename]
            if _uses_trained_featurizer(doc, write.featurizer):
                return False
            for op in operand_params(doc, write.do):
                if op.split(".", 1)[0] in trained_roots:
                    return False
            for ref in operand_reads(doc, write.do):
                read = doc.reads[ref.read]
                if _uses_trained_featurizer(doc, read.featurizer):
                    return False
                assert ref.model is not None
                if not constant(ref.model, visiting | {model}):
                    return False
        return True

    return frozenset(model for model in models if constant(model, set()))


def _trained_roots(doc: Document) -> frozenset[str]:
    """The featurizer and params names a fit moves: ``train.params`` with any
    ``.slot`` suffix dropped. Empty without a ``train`` section."""
    if doc.train is None:
        return frozenset()
    return frozenset(p.split(".", 1)[0] for p in doc.train.params)


def _uses_trained_featurizer(doc: Document, ref: Any) -> bool:
    if doc.train is None or ref is None:
        return False
    trained = _trained_roots(doc)
    chain = (ref,) if isinstance(ref, str) else tuple(ref)
    return any(name in trained for name in chain)


def _read_closure(
    doc: Document, ref: ReadRef, visiting: set[str], identity: Mapping[str, Any]
) -> Any:
    read = doc.reads[ref.read]
    model, input_role = doc.group_of(ref)
    return {
        **_attention_requirement(doc, model, input_role),
        "site": _entry(doc.sites, str(read.site)),
        "pos": _pos_entry(doc, read.pos),
        "featurizer": _featurizer_entry(doc, read.featurizer),
        "dims": read.dims,
        "input": input_role,
        "data": identity.get(input_role),
        "model": _model_closure(doc, model, input_role, visiting, identity),
        "train": dataclasses.asdict(doc.train)
        if doc.train is not None and _uses_trained_featurizer(doc, read.featurizer)
        else None,
    }


def _entry(table: Mapping[str, Any], name: str) -> Any:
    return dataclasses.asdict(table[name])


def _pos_entry(doc: Document, pos: Any) -> Any:
    if isinstance(pos, str):
        resolved = doc.positions[pos]
        return (
            dataclasses.asdict(resolved)
            if isinstance(resolved, PositionSpec)
            else str(resolved)
        )
    return dataclasses.asdict(pos) if isinstance(pos, PositionSpec) else pos


def _featurizer_entry(doc: Document, ref: Any) -> Any:
    if ref is None:
        return None
    chain = (ref,) if isinstance(ref, str) else tuple(ref)
    return [dataclasses.asdict(doc.featurizers[name]) for name in chain]


def cohort_key(doc: Document, data_identity: Mapping[str, str]) -> GroupKey | None:
    """The identity of the forward a fit's members can **share** (§4
    "Cohorts"), or ``None`` for a document that fits nothing.

    Points of a campaign whose keys agree may fit together: one forward per
    optimizer step over the concatenation of every member's minibatch, each
    member's writes landing on its own rows. That is one forward exactly when
    every member runs the same network realization over the same rows in the
    same frame — so the key is the realization (``canonical_model_ref``, dtype
    and quantization included), the data identity per input role (the content
    digest and field the rows are encoded from, as the group keys carry it),
    and the ``segments`` section that frames them. Nothing a member
    trains, sweeps or schedules is in it: featurizer specs, the seed, the
    objective, the optimizer, the step budget, the eval cadence and the
    write's address are each member's own, applied to its own rows — with one
    exception. Whether an intervened model's sites reach an **attention
    interior** (`_attention_requirement`) decides the attention
    implementation its forward runs under, and a cohort forward runs under one
    implementation for every member; so that requirement, per intervened
    model, is in the key, and a member whose sites need eager never shares a
    forward with one whose sites do not.

    Like a [`GroupKey`][], the canonical JSON of that body — structural,
    not hashed; it is only ever a dict key inside [`fit_cohorts`][].
    """
    if doc.train is None:
        return None
    body = {
        "network": canonical_model_ref(doc.model),
        "data": dict(data_identity),
        "segments": _method_section(doc, "segments"),
        "attention": {
            name: _attention_requirement(doc, name, str(im.input))
            for name, im in doc.intervened_models.items()
        },
    }
    return json.dumps(body, sort_keys=True, separators=(",", ":"), default=_encode)


def _method_section(doc: Document, name: str) -> Any:
    """One section of the document's ``method`` group as authored (§1), or
    ``None`` when absent."""
    method = doc.raw.get("method")
    return method.get(name) if isinstance(method, Mapping) else None


def fit_cohorts(
    docs: Sequence[Document], identities: Sequence[Mapping[str, str]]
) -> tuple[tuple[int, ...], ...]:
    """Partition a campaign's point indices into fit cohorts (§4 "Cohorts"):
    the train points sharing a [`cohort_key`][] form one cohort each, in
    first-appearance order; every other point stands alone. Every index
    appears exactly once, so a campaign loop can run the partition in order
    and cover the campaign."""
    if len(docs) != len(identities):
        raise ValueError(
            f"{len(docs)} documents but {len(identities)} data identities — the "
            "two are in lockstep per point"
        )
    cohorts: dict[GroupKey, list[int]] = {}
    order: list[tuple[int, ...] | GroupKey] = []
    for index, (doc, identity) in enumerate(zip(docs, identities)):
        key = cohort_key(doc, identity)
        if key is None:
            order.append((index,))
            continue
        if key not in cohorts:
            cohorts[key] = []
            order.append(key)
        cohorts[key].append(index)
    return tuple(
        tuple(cohorts[entry]) if isinstance(entry, str) else entry for entry in order
    )


def _encode(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    if isinstance(obj, tuple):
        return list(obj)
    raise TypeError(f"unencodable {type(obj).__name__} in a plan closure")


# --------------------------------------------------------------------------- #
# the campaign's plans (formerly execution.py)
# --------------------------------------------------------------------------- #


def campaign_plans(
    docs: Sequence[Document], canonical: Sequence[Mapping[str, Any]]
) -> tuple[PointPlan, ...]:
    """The per-point plans a campaign executes from.

    Public because the interning claim is checkable arithmetic:
    [`interned_groups`][] over these plans is how
    many forward groups a run *owes*, and
    [`forwards`][causalab.protocol.engine.RunResult.forwards] is what it paid. One
    derivation, so the number a caller verifies against is the number
    execution keyed on.

    ``canonical`` is the points' canonical forms, in lockstep with ``docs``
    ([`canonical`][causalab.neural.shared.sweep.SignedStep.canonical]): the data
    half of a group's identity is read from there, never recomputed.
    """
    if len(docs) != len(canonical):
        raise ProtocolError(
            "P2",
            f"{len(docs)} point documents but {len(canonical)} canonical forms "
            "— the two are in lockstep per point",
        )
    return tuple(
        plan_point(doc, data_identity=_data_identity(doc, form))
        for doc, form in zip(docs, canonical)
    )


def _data_identity(doc: Document, canonical: Mapping[str, Any]) -> dict[str, str]:
    """Input role → the identity of the rows that role will be encoded from.

    Folded into every forward-group key so two points reading *different*
    data on the same role never intern together. What determines a role's batch
    is the **content** of the rows the ref selects plus the one field the
    executor tokenizes out of them (``DataRole.resolved_field`` —
    ``<column>[eval]`` for a drawn role, §2.2 — the spelling
    [`resolve_roles`][causalab.protocol.positions.roles.resolve_roles] hands the engine),
    so the identity is ``"<content digest>#<field>"`` where the digest is the
    one the canonical form already stamped for that role (§2.2, §7: sha256 over
    the selected rows, not the file). The ref's *name* is deliberately absent:
    two tables under one name must never intern, and one table under two names
    must — which a name-keyed identity got backwards on both counts.

    The role names mirror ``resolve_roles`` (``counterfactual[0]`` for a
    tuple-valued role) so the keys line up with the plan's ``input``.
    """
    digests = _role_digests(doc, canonical)
    return {
        role_name: f"{digests[role_name]}#{role_spec.resolved_field}"
        for role_name, role_spec in input_roles(doc).items()
    }


def _role_digests(doc: Document, canonical: Mapping[str, Any]) -> dict[str, str]:
    """Input role → the content digest of the rows it reads, read off the
    point's canonical form (``data.<role>.digest``; a tuple-valued role is a
    list there, indexed in step with ``resolve_roles``).

    Read, not recomputed: the canonical form's digest is the one the point
    digest committed to, so there is one content digest per table in the
    system and the interning identity cannot drift from the provenance one.
    A role without a stamped digest is refused rather than named — nothing
    executable lacks one (rows resolve through the same ref the stamp did),
    so this only fires on a canonical form that is not this point's.
    """
    stamped = canonical.get("data")
    if not isinstance(stamped, Mapping):
        raise ProtocolError(
            "P2", "canonical form carries no 'data' section to read digests from"
        )
    digests: dict[str, str] = {}
    for role, value in doc.data.items():
        entries = value if isinstance(value, tuple) else (value,)
        forms_raw = stamped.get(role)
        # the shapes must agree, never broadcast: a tuple-valued role is a
        # list in the canonical form and a single role is one mapping, so a
        # form of the other shape is not this point's and is refused
        if isinstance(value, tuple) != isinstance(forms_raw, (list, tuple)):
            raise ProtocolError(
                "P2",
                f"data role {role!r} is {'tuple' if isinstance(value, tuple) else 'single'}"
                "-valued but its canonical form is not — the two are the same point's",
            )
        forms = list(forms_raw) if isinstance(forms_raw, (list, tuple)) else [forms_raw]
        if len(forms) != len(entries):
            raise ProtocolError(
                "P2",
                f"data role {role!r} has {len(entries)} entries but its "
                f"canonical form has {len(forms)}",
            )
        for j, form in enumerate(forms):
            role_name = role if not isinstance(value, tuple) else f"{role}[{j}]"
            digest = form.get("digest") if isinstance(form, Mapping) else None
            if not isinstance(digest, str) or not digest:
                raise ProtocolError(
                    "P2",
                    f"data role {role_name!r} has no content digest in its "
                    "canonical form — the interning identity is the digest of "
                    "the rows a role reads (§2.2), never the ref's name",
                )
            digests[role_name] = digest
    return digests
