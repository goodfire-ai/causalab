"""Cache captures and prefixes across a campaign.

Tap, capture, and prefix keys identify reusable work. ``ForwardCache``
owns the stored values; ``Interning`` gives one point access to them.
``InterningMixin`` connects that handle to ExecutorBase and engine forwards.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Iterable, Literal, Mapping, TYPE_CHECKING

import torch

from causalab.neural.shared.executor.ragged import RowWindow
from causalab.neural.shared.plan import GroupKey
from causalab.neural.shared.sites import ResolvedSite
from causalab.protocol.registry.shapes import FeatureShape
from causalab.protocol.schema import SiteSpec


#: What identifies a tap for capture-sink sharing — see [`tap_key`][].
TapKey = tuple[int, str, FeatureShape, int | None, str | None, int | None, tuple | None]


def tap_key(site: ResolvedSite, source: Any = None) -> TapKey:
    """Identity of a tap for capture-sink sharing.

    Two sites may share a module and side yet mean different tensors — a
    different tuple element, or the same tensor read through a different shape
    — so the shape and tuple index are part of the identity. Keying on the
    module alone would let one tap read another's tensor.

    ``expert`` is part of the identity for the write path's sake: two writes
    at the same interior slot naming *different* experts must land as two
    separately masked applications, and the address grouping keys on this.

    ``source`` is an engine-specific *interior* address (the nnsight engine's
    [`SourceAddress`][causalab.neural.engines.nnsight_tracing.addresses.SourceAddress]):
    two interior taps may share the module, side and even shape while meaning
    different ops inside its forward, so its identity joins the key. It
    defaults to ``None`` so a hook-engine tap's key is unchanged.
    """
    return (
        id(site.module),
        site.kind,
        site.shape,
        site.tuple_index,
        site.interface_slot,
        site.expert,
        None
        if source is None
        else (
            source.op_pattern,
            source.peel,
            source.field,
            source.arg,
            source.tuple_index,
        ),
    )


#: What one entry of the [`ForwardCache`][] store is keyed by — a group's
#: [`GroupKey`][] plus **which rows** of the role it
#: was run on:
#:
#: * ``key`` — the role's whole rows (a campaign point's own forward);
#: * ``(key, (i0, i1, …))`` — the role's rows at those indices (a fit's
#:   minibatch);
#: * ``(key, "<split ref>")`` — the role's field over ``train.eval``'s split.
#:
#: The group key is the same in all three: it is the identity of the
#: *forward*, and the rows it ran over are a second coordinate, not a
#: different forward. [`ForwardCache.wanted`][] stays keyed by the bare
#: group key because the tap union applies to every row selection of one
#: forward alike.
CaptureKey = GroupKey | tuple[GroupKey, tuple[int, ...] | str]

#: What one cached **prefix** is keyed by (§4 "Resume"): the un-intervened
#: identity of the forward ([`base_key`][causalab.neural.shared.plan.ForwardGroup.base_key]),
#: the rows coordinate of the [`CaptureKey`][] (``None`` for the whole role,
#: a fit's slice otherwise), the row window ``(start, stop)`` the forward ran
#: over, and the block whose incoming residual the entry holds. The window is
#: part of it because a window is padded to its own frame: rows ``[0, 2)`` of a
#: four-row role and the same two rows as a minibatch are different tensors.
#: Hooks may append the attention implementation to distinguish accelerated
#: prefixes from eager ones. The first four coordinates (including those used
#: to release all variants of a prefix) retain their meaning.
PrefixKey = (
    tuple[GroupKey, tuple[int, ...] | str | None, tuple[int, int], int]
    | tuple[GroupKey, tuple[int, ...] | str | None, tuple[int, int], int, str]
)


#: What one forward group's pass decided about reuse (§3, §4 "Resume"):
#: ``ran`` — the group's forward was executed; ``served`` — the group was
#: served an earlier pass's captures under the same capture key, no forward;
#: ``resumed`` — a window of an executed forward started from a cached
#: prefix instead of block 0.
ReuseKind = Literal["ran", "served", "resumed"]


@dataclasses.dataclass(frozen=True)
class Reuse:
    """One reuse decision of a counted (whole-role) pass, as the store
    records it in [`ForwardCache.decisions`][]: the ``kind`` and the key
    it was made on — the group key (the pass's label when the group has
    none) for ``ran`` / ``served``, ``<prefix identity>@<block>`` for
    ``resumed``. Every coordinate is torch-free and plan-derived, so the
    record is one program of the document and the point set: every rank of
    a world makes the same decisions in the same order as the world-1 run
    over the same points (``docs/model_parallelism.md`` §10.6, the reuse
    pin). A record, never a decision input: nothing reads it back."""

    kind: ReuseKind
    key: GroupKey | str


@dataclasses.dataclass(frozen=True)
class PrefixPlan:
    """Where one forward group's pass may start, and what it may leave behind
    (§4 "Resume") — the plan's arithmetic, per group key, in the form the
    engine reads at each window.

    ``resume_at`` is [`resume_at`][causalab.neural.shared.plan.ForwardGroup.resume_at] of
    the *interned* group (0 = never). ``write_depth`` is the group's: the
    residual entering block ``d`` is the un-intervened one iff
    ``d <= write_depth``, so a pass may store the prefix at ``d`` only up to
    there — an intervened pass at layer 1 must never hand a later point at
    layer 3 a post-write residual as its prefix. Both are in the plan's block
    coordinate ([`PAST_BLOCKS`][causalab.protocol.positions.alignment.PAST_BLOCKS] for "past every
    block"); the engine clamps them to the model it loaded."""

    base_key: GroupKey
    resume_at: int
    write_depth: int


@dataclasses.dataclass(frozen=True)
class ForwardCache:
    """The campaign-wide store that makes §3's interning real at run time.

    The planner already says which forward groups a swept document shares: a
    group's ``key`` is the structural identity of everything that determines
    its activations, and taps are deliberately **not** part of it, because
    reading layer 3 or layer 23 of the same un-intervened forward is the same
    forward. A per-point loop ignores that and re-runs the shared group once
    per point; this is where the plan's guarantee gets claimed.

    ``wanted`` is the union of tap sites the **whole campaign** asks of each
    group key, so the first point to reach a group captures every address a later
    point will want. ``captured`` holds those activations *raw* — before the
    positional gather and before any featurizer — which is why points tapping
    one address through different featurizers, dims or positions still share
    a single forward. ``routing`` carries the experts-interface sub-axis table
    beside them: an interface capture without the dispatch indices it joins on
    is not a replayable value, so the two travel together or not at all.

    The trade is compute for memory: a 32-layer harvest shared by 32 points
    holds 32 captures at once where the per-point loop held one. That is
    inherent in making one pass serve the union, and it is still a large net
    win — 32 separate passes cost ~16x one full pass even when each is elided
    at its own tap. A fit adds its own slices (§4 "Fits"): the constant groups
    of its minibatches and of its eval split are captured once each and served
    on every later step, epoch, eval pass and point, so the store also holds
    roughly one more copy of the training rows' tapped activations plus one of
    the eval rows', on the capture device — against ``2·B·E + 2·E`` forwards
    per point saved down to ``B + B·E + 1 + E`` for the first point and
    ``B·E + E`` for each later one. A slice pass captures the **campaign's**
    tap union for its key, not only the fit's own taps, so a campaign that
    also taps ``lm_head`` on the source group from an inference point makes
    the fit retain vocabulary-wide captures over the training rows too.

    A whole-role capture lives exactly as long as a pass is still **owed** it.
    ``owed`` is the plan's count of forward-group instances per key across
    every point; a pass that runs or is served a key settles one, and the
    capture is dropped when the count reaches zero. A key only one pass
    keys into is never stored at all — publishing it would pin a capture no
    later point can ask for, and for a swept *patched* group tapping
    ``lm_head`` that is a full-vocabulary tensor per point (150 rows of
    gemma-2-2b-it hold 13 GiB each; three of them exhaust an 80 GB device
    where the per-point loop never held more than one). Sliced keys are not
    settled: a fit's inner passes re-read them every step, so their count is
    not a plan quantity. Those live as long as the request.

    ``prefixes`` adds, per (prefix identity, rows, window, depth), one
    residual of ``rows × seq × d_model``, and one per intervened *input*
    rather than per point, since every model writing on the same rows at or
    above a depth shares that entry. Their lifetime is the plan's as well:
    ``prefix_owed`` counts, per (prefix identity, depth), the group instances
    across every point whose interned ``resume_at`` reaches that depth — the
    passes that could still start from it. A whole-role pass settling its
    group (`ExecutorBase._settle`) settles every depth up to its own
    ``resume_at``, and at zero every entry of that identity and depth is
    dropped, sliced keys included, and no pass stores a depth nobody is owed
    any more. What that bounds: at most ``|wanted depths|`` residuals per
    (identity, rows, window) are live at once — for an *ascending* scan the
    first pass stores every wanted depth and each dies only when the last
    group reaching it settles, so the peak is the whole set (a 10-depth scan
    at 2k context on the 40-block Qwen3.6-35B-A3B: ~6.5 GB); the refcount removes the tail
    after the last sharer, and a fit's per-minibatch and eval slices die with
    the point, since the point's own pass runs *after* the fit. A depth past
    every block (``PAST_BLOCKS``) and the last block's depth name the same
    residual and are stored twice — a deliberate duplicate that keeps the
    refcount in plan coordinates. Capping the live set to the deepest ``k``
    entries per identity is a possible follow-up, not done here.

    **Prefixes** (§4 "Resume"). ``prefix_plans`` says, per group key, the
    block its pass may start at and how deep its pass stays un-intervened;
    ``wanted_prefix_depths`` says, per prefix identity, every depth some group
    of the campaign will resume at, so the first pass over those rows — an
    ``original`` group's, or a shallower intervened model's — stores each of
    them on its way. Resuming is independent of the capture interning above:
    a fit's minibatch pass of the trained model is never *served* (its
    activations move every step) yet still *resumes*, because the prefix
    below its write is fit-constant even when the group is not.

    The store is keyed by [`CaptureKey`][] (a group key, plus the row
    slice it ran over) and [`TapKey`][], both engine-neutral, but only an
    engine whose ``_run_group`` consults it actually interns;
    [`forwards`][causalab.protocol.engine.RunResult.forwards] then reports
    ``len(executed)``. A fit's inner passes are tallied apart — the constant
    groups it ran in ``inner_executed``, the ones it was served in
    ``inner_served`` — because they are not forward groups and are not
    counted; [`execute_request`][causalab.neural.shared.execution.execute_request]
    reports the per-point difference as ``fit_forwards`` in each point's
    summary — the ``explain``-style record [`RunResult`][causalab.protocol.engine.RunResult] hands back to
    the caller, not a file in the run tree.
    """

    #: group key -> every site the campaign taps in that group's forward
    wanted: Mapping[GroupKey, tuple[SiteSpec, ...]] = dataclasses.field(
        default_factory=dict
    )
    #: capture key -> the raw activations one forward left, per tap
    captured: dict[CaptureKey, dict[TapKey, torch.Tensor]] = dataclasses.field(
        default_factory=dict
    )
    #: capture key -> the experts routing table beside those captures
    routing: dict[CaptureKey, dict[TapKey, torch.Tensor]] = dataclasses.field(
        default_factory=dict
    )
    #: group key -> whole-role passes (across every point's plan) that
    #: still owe this key a run or a serving; the publisher's own pass
    #: included. Decremented by `ExecutorBase._settle`; at zero the
    #: key's captures are dropped. A key absent here is not tracked: its
    #: captures are stored and kept for the request (the hand-built caches
    #: of the unit tests, or an engine that plans no campaign).
    owed: dict[GroupKey, int] = dataclasses.field(default_factory=dict)
    #: one entry per forward group actually run, in order — the number
    #: [`forwards`][causalab.protocol.engine.RunResult.forwards] reports. The
    #: group's key when the pass keyed into the store, its ``model/input``
    #: label otherwise (`ExecutorBase._publish`)
    executed: list[GroupKey | str] = dataclasses.field(default_factory=list)
    #: one entry per fit-constant group a fit's inner executor (a minibatch,
    #: the eval pass) actually ran and published under a sliced key — the
    #: passes the cache did not save it. Never part of ``forwards``: inner
    #: passes of a fit are not forward groups. The non-constant pass a fit
    #: differentiates through every step is not recorded here either — it
    #: is the fit's own cost, one per step, and no cache is consulted for it.
    inner_executed: list[GroupKey] = dataclasses.field(default_factory=list)
    #: one entry per fit-constant group a fit's inner executor asked for and
    #: was **served** from a sliced key instead of running — the passes the
    #: cache did save. ``inner_executed`` + ``inner_served`` is what a fit
    #: would have paid for its constant groups without the store.
    inner_served: list[GroupKey] = dataclasses.field(default_factory=list)
    #: group key -> where its pass may start and how deep it stays
    #: un-intervened (§4 "Resume"); a key absent here never resumes
    prefix_plans: Mapping[GroupKey, PrefixPlan] = dataclasses.field(
        default_factory=dict
    )
    #: prefix identity (``base_key``) -> every block some group of the
    #: campaign resumes at, i.e. the residuals a pass over those rows stores
    wanted_prefix_depths: Mapping[GroupKey, frozenset[int]] = dataclasses.field(
        default_factory=dict
    )
    #: prefix key -> the residual entering that block, detached, on the
    #: capture device
    prefixes: dict[PrefixKey, torch.Tensor] = dataclasses.field(default_factory=dict)
    #: (prefix identity, depth) -> whole-role group instances across every
    #: point's plan whose ``resume_at`` reaches ``depth``; the plan's count of
    #: passes that may still start from that prefix. Decremented by
    #: `ExecutorBase._settle`; at zero every ``prefixes`` entry of that
    #: identity and depth is dropped. Absent pairs are not tracked and live
    #: for the request, as ``owed`` treats an absent key
    prefix_owed: dict[tuple[GroupKey, int], int] = dataclasses.field(
        default_factory=dict
    )
    #: one entry per window a pass resumed, holding the block it started at —
    #: ``len`` is how many forwards resumed, ``sum`` how many blocks were
    #: skipped. Both inner and counted passes land here: a resume is a saving
    #: whichever kind of pass it happens in, and a forward that resumed is
    #: still one forward in ``executed`` / ``inner_executed``
    resumed: list[int] = dataclasses.field(default_factory=list)
    #: group key -> ``{write member: fire count}`` of the pass that ran it
    #: (§4 "Fires"): a point served this key's captures records the counts
    #: of the pass that produced them, not zeroes for a forward it never ran.
    #: Kept for the request — a count is a few bytes, and it outlives the
    #: captures it describes
    fires: dict[GroupKey, dict[str, int]] = dataclasses.field(default_factory=dict)
    #: one [`Reuse`][] per decision a counted pass made, in order — the
    #: record the parallel reuse pin compares across ranks and against
    #: world 1 (``tests/neural/engines/pytorch_hooks/test_reuse_parallel.py``).
    #: Appended beside ``executed`` (``ran``), the serving in ``_interned``
    #: (``served``) and ``resumed`` (``resumed``); never read by the executor
    decisions: list[Reuse] = dataclasses.field(default_factory=list)


@dataclasses.dataclass(frozen=True)
class Interning:
    """One point's handle on a shared [`ForwardCache`][].

    ``keys`` maps this point's ``(model, input)`` groups to the plan's
    [`GroupKey`][] they key into; the cache itself
    belongs to the whole campaign.
    An executor built without one runs every group itself and touches the
    store not at all — the unit tests' reference path, and what "no reuse"
    means.

    ``rows`` says which rows of each role this executor runs: ``None`` for the
    whole role (a campaign point), a tuple of indices for a minibatch, a split
    ref for the ``train.eval`` pass. It becomes the second coordinate of every
    [`CaptureKey`][] this executor reads or writes, so a slice is never
    served a whole-role capture or vice versa. ``counted`` says whether the
    passes this executor runs are forward groups of the campaign
    (``executed``, hence [`forwards`][causalab.protocol.engine.RunResult.forwards])
    or the inner passes of a fit (``inner_executed``). A fit's inner handles
    come from `ExecutorBase.inner_interning`; ``keys`` stays the
    point's whole map there, because a group the fit changes may not be served
    from the store (`ExecutorBase._may_intern`) yet still resumes from
    its un-intervened prefix (§4 "Resume"), and both are looked up by key."""

    keys: Mapping[tuple[str, str], GroupKey]
    cache: ForwardCache
    rows: tuple[int, ...] | str | None = None
    counted: bool = True


class InterningMixin:
    """Cross-point interning (§3) — the half an engine's ``_run_group``
    consults: this executor's group keys, whether a group may be served from
    or published to the shared [`ForwardCache`][], where its captures and
    prefixes live, and the settling that ends their lifetime.

    Composed into [`ExecutorBase`][];
    the host attributes the methods read are declared below for the type
    checker and set in the executor's constructor."""

    if TYPE_CHECKING:
        interning: Interning | None
        grad_enabled: bool
        fit_constant_models: frozenset[str]

    def _group_key(self, model: str, input_role: str) -> GroupKey | None:
        """This group's plan key, or ``None`` when the executor was built
        without a shared cache to key into."""
        if self.interning is None:
            return None
        return self.interning.keys.get((model, input_role))

    def _may_intern(self, model: str) -> bool:
        """Whether this executor may serve ``model``'s group from — or publish
        it to — the shared store at all.

        Outside a fit every group is fair game: a campaign point's captures
        are final. Inside one — a grad-enabled minibatch, or any executor whose
        passes are not counted (the grad-free eval pass) — only a model no
        trained parameter can reach qualifies ([`fit_constant_models`][]).
        Gating on grad alone would be wrong in both directions: the eval pass
        runs grad-free yet must never be served the trained model's capture
        from an earlier step, and the source forward runs grad-enabled yet is
        exactly what the fit should stop re-running.
        """
        if self.interning is None:
            return False
        if self.grad_enabled or not self.interning.counted:
            return model in self.fit_constant_models
        return True

    def inner_interning(self, rows: tuple[int, ...] | str) -> Interning | None:
        """The handle a fit's inner executor over ``rows`` of this point's
        roles gets: the same campaign cache and this point's group keys,
        keyed by the row slice, and not counted as forward groups. ``None``
        when this point interns nothing.

        The keys are **not** narrowed to the fit-constant groups: which
        groups may be served from or published to the store is
        `_may_intern`'s decision, and the trained model's group — never
        served — still needs its key to find the un-intervened prefix it
        resumes from (§4 "Resume")."""
        if self.interning is None:
            return None
        return Interning(
            keys=self.interning.keys,
            cache=self.interning.cache,
            rows=rows,
            counted=False,
        )

    def _capture_key(self, key: GroupKey) -> CaptureKey:
        """Where this executor's captures of group ``key`` live in the store:
        the bare key for the whole role, ``(key, rows)`` for a slice."""
        assert self.interning is not None
        rows = self.interning.rows
        return key if rows is None else (key, rows)

    def _prefix_plan(self, key: GroupKey | None) -> PrefixPlan | None:
        """The plan's resume arithmetic for this group (§4 "Resume"), or
        ``None`` when nothing about it may resume or store a prefix: no shared
        store, no key, or a key the campaign planned no prefix for.

        Deliberately not gated on `_may_intern`: the prefix below a
        group's first write is ``original``'s activations over these rows,
        fit-constant even when the group itself is what the fit trains."""
        if key is None or self.interning is None:
            return None
        return self.interning.cache.prefix_plans.get(key)

    def _prefix_key(self, plan: PrefixPlan, window: RowWindow, depth: int) -> PrefixKey:
        """Where the residual entering block ``depth`` of ``plan``'s prefix
        lives for this executor's rows and ``window``."""
        assert self.interning is not None
        return (
            plan.base_key,
            self.interning.rows,
            (window.start, window.stop),
            depth,
        )

    def _interned(
        self, key: GroupKey | None, taps: Iterable[TapKey]
    ) -> tuple[dict[TapKey, torch.Tensor], dict[TapKey, torch.Tensor]] | None:
        """This group's raw captures (and their routing tables) if an earlier
        pass already produced **every** address it taps under the same
        capture key, else ``None``.

        All-or-nothing on purpose: a partial hit would still have to run the
        forward for the addresses it missed, and the pass it runs captures the
        campaign's whole union anyway."""
        if key is None or self.interning is None:
            return None
        capture_key = self._capture_key(key)
        captured = self.interning.cache.captured.get(capture_key)
        if captured is None:
            return None
        wanted = set(taps)
        if not wanted or any(k not in captured for k in wanted):
            return None
        if not self.interning.counted:
            # a fit's inner pass the store saved — the number `inner_executed`
            # is measured against
            self.interning.cache.inner_served.append(key)
        else:
            self.interning.cache.decisions.append(Reuse("served", key))
        routing = self.interning.cache.routing.get(capture_key, {})
        return (
            {k: captured[k] for k in wanted},
            {k: routing[k] for k in wanted if k in routing},
        )

    def _publish(
        self,
        key: GroupKey | None,
        label: str,
        capture: Mapping[TapKey, torch.Tensor],
        routing: Mapping[TapKey, torch.Tensor],
    ) -> None:
        """Record that one forward group ran, and hand its raw captures to
        the passes that share its capture key.

        Called once per pass an engine actually executes, so
        ``len(cache.executed)`` is what the run paid against what
        [`interned_groups`][causalab.neural.shared.plan.interned_groups] says it owed. A fit's
        inner passes are tallied apart (``inner_executed``, keyed passes only)
        so that number never moves with the fit's step count.

        Captures are stored detached: a grad-enabled pass may have produced
        them, and what a later pass gathers and featurizes from the store must
        be a leaf — the trained featurizer applied *after* the capture is where
        that pass's graph begins."""
        if self.interning is None:
            return
        if not self.interning.counted:
            if key is not None:
                self.interning.cache.inner_executed.append(key)
        else:
            self.interning.cache.executed.append(key or label)
            self.interning.cache.decisions.append(Reuse("ran", key or label))
        if key is None:
            return
        # a whole-role key no *other* pass keys into has no reader to keep
        # a capture for: storing it would only pin this pass's raw activations
        # (a swept patched group tapping lm_head: the whole vocabulary, every
        # row) until the request ends
        if self._tracks(key) and self.interning.cache.owed[key] <= 1:
            return
        capture_key = self._capture_key(key)
        # only what a tap actually filled: publishing an address whose module
        # never ran would hand a later point an empty capture instead of
        # letting it run the forward
        self.interning.cache.captured.setdefault(capture_key, {}).update(
            {k: value.detach() for k, value in capture.items()}
        )
        self.interning.cache.routing.setdefault(capture_key, {}).update(
            {k: value.detach() for k, value in routing.items()}
        )

    def _tracks(self, key: GroupKey) -> bool:
        """Whether group ``key``'s whole-role captures are lifetime-tracked:
        this executor runs a counted, whole-role pass and the plan counted the
        key's instances into ``owed``."""
        assert self.interning is not None
        return (
            self.interning.counted
            and self.interning.rows is None
            and key in self.interning.cache.owed
        )

    def _settle(self, key: GroupKey | None) -> None:
        """One whole-role pass over group ``key`` is done with its captures —
        run or served, the plan owes this key one pass fewer. When no pass
        is owed any more the store drops the key's captures and routing:
        the capture's lifetime is the span between its first sharer and its
        last, not the request.

        Called by an engine's ``_run_group`` once the pass has gathered its
        reads; a ``None`` key (no interning, a decoding group, a group a
        fit trains through) settles nothing, as does a sliced pass. The
        pass's prefixes are settled at the same moment (`_settle_prefixes`)."""
        if key is None or self.interning is None:
            return
        self._settle_prefixes(key)
        if not self._tracks(key):
            return
        owed = self.interning.cache.owed
        owed[key] -= 1
        if owed[key] <= 0:
            capture_key = self._capture_key(key)
            self.interning.cache.captured.pop(capture_key, None)
            self.interning.cache.routing.pop(capture_key, None)

    def _settle_prefixes(self, key: GroupKey) -> None:
        """One whole-role pass over group ``key`` is done resuming (§4 "Resume"):
        every prefix depth it could have started from — the wanted depths up
        to its plan's ``resume_at`` — is owed one pass fewer, and a depth no
        remaining instance can reach is dropped for every rows coordinate and
        window at once. A sliced pass settles nothing: a fit's inner passes
        re-read their prefixes every step, and the point's own pass, which
        follows the fit, settles on their behalf."""
        assert self.interning is not None
        if not self.interning.counted or self.interning.rows is not None:
            return
        cache = self.interning.cache
        plan = cache.prefix_plans.get(key)
        if plan is None:
            return
        for depth in cache.wanted_prefix_depths.get(plan.base_key, ()):
            pair = (plan.base_key, depth)
            if depth > plan.resume_at or pair not in cache.prefix_owed:
                continue
            cache.prefix_owed[pair] -= 1
            if cache.prefix_owed[pair] <= 0:
                for prefix in [k for k in cache.prefixes if (k[0], k[3]) == pair]:
                    del cache.prefixes[prefix]
