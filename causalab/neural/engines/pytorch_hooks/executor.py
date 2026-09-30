"""Execute a concrete protocol point through PyTorch hooks.

Forward groups run lazily in operand dependency order. The executor
installs writes before captures at a shared address. Absolute writes apply
first, additive deltas sum, and renormalization uses the pre-write norm.
ExecutorBase handles positions, gathers, featurizers, and shared write math.

``batch_rows`` divides a group into row windows. Positions use the common
padded frame; operands and Gaussian draws are sliced by row. Captures join
in row order. Each decoding window continues immediately after its prefill.
Batching can change dtype rounding.

Input taps use pre-hooks and output taps use forward hooks. Interior taps
wrap the corresponding attention, expert, or delta functions. Ragged writes
and saves follow the declared protocol policy.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import torch

from causalab.neural.engines.pytorch_hooks.attention_interface import (
    InterfaceTap,
    attention_interface_taps,
)
from causalab.neural.engines.pytorch_hooks.delta_interface import (
    DeltaTap,
    delta_kernel_taps,
)
from causalab.neural.engines.pytorch_hooks.experts_interface import (
    ExpertsTap,
    experts_interface_taps,
)
from causalab.neural.engines.pytorch_hooks.experts_path import lean_experts_path
from causalab.neural.engines.pytorch_hooks.kernels.fused_norms import (
    fused_norm_path,
)
from causalab.neural.engines.pytorch_hooks.loading import ModelBundle, load_model
from causalab.neural.engines.pytorch_hooks.stages import (
    StageForward,
    TrainedOwner,
    hook_placement,
)
from causalab.neural.shared.encoding import (
    Continuation,
    EncodedBatch,
    continuation_frame,
    resolve_steps,
)
from causalab.neural.shared.executor import (
    ExecutorBase,
    ForwardCache,
    Interning,
    PrefixKey,
    PrefixPlan,
    RaggedValue,
    Reuse,
    RowWindow,
    TapKey,
    document_seed,
    refuse_unstackable,
    tap_key,
)
from causalab.neural.shared.fires import (
    FireTally,
    GroupFires,
    check_fires,
    group_label,
)
from causalab.neural.engines.pytorch_hooks.rows import WHOLE, RowSplit
from causalab.neural.shared.head import ReadTap, head_module, taps_head
from causalab.neural.shared.parallel.agreements import (
    Agreements,
    summed_fires,
    whole_steps,
)
from causalab.neural.shared.parallel.collective import SOLO
from causalab.neural.shared.parallel.mismatch import shared_mismatch
from causalab.neural.shared.parallel.context import SequenceFrame, activate
from causalab.neural.shared.parallel.fragments import Fragments
from causalab.neural.shared.parallel.placement import ExpertLocal, StageLocal
from causalab.neural.shared.parallel.placements import ROUTER_SCORES
from causalab.neural.shared.parallel.taps import IDENTITY_TAP, TapFragments
from causalab.protocol.registry.shapes import FeatureShape
from causalab.neural.shared.gdn_short.binding import short_seq_kernel_path
from causalab.neural.shared.kernels import torch_kernel_path
from causalab.neural.shared.layout import (
    from_contract,
    rebuild_payload,
    tap_tensor,
    to_contract,
)
from causalab.neural.shared.plan import group_reads, write_names
from causalab.neural.shared.sites import (
    ResolvedSite,
    Writeback,
    adapter_of,
    resolve_site,
)
from causalab.protocol.rules.errors import ProtocolError
from causalab.neural.shared.plan import GroupKey
from causalab.protocol.positions.encoding import generated_budget
from causalab.protocol.registry import walk
from causalab.protocol.schema import (
    LAYERLESS_COMPONENTS,
    ReadRef,
    ReadSpec,
    SiteSpec,
    WriteSpec,
    operand_reads,
)

__all__ = [
    "ForwardCache",
    "Interning",
    "PointExecutor",
    "RaggedValue",
    "Resume",
    "document_seed",
]


def has_recurrent_layers(config: Any) -> bool:
    """Whether ``config`` declares a DeltaNet (``linear_attention``) layer —
    the hybrid families, whose forward builds a second, 2-D padding mask for
    those layers beside the causal one."""
    layer_types: Sequence[str] = getattr(config, "layer_types", None) or ()
    return "linear_attention" in layer_types


#: the layer types a hybrid decoder indexes its mask mapping by
#: (``config.layer_types[i]``) that [`prompt_masks`][] has a mask for: the
#: causal mask for an attention layer, the 2-D padding mask for a DeltaNet
#: one. A closed table, not a structural guess — a family declaring a type
#: outside it (a sliding window beside its recurrent layers, say) is refused
#: by name rather than handed a mapping its block loop would ``KeyError`` on
#: or a mask with the wrong semantics for the layer
PROMPT_MASK_TYPES: frozenset[str] = frozenset({"full_attention", "linear_attention"})


def prompt_masks(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
) -> Any:
    """The attention mask(s) a prompt-only forward hands ``model`` for one
    left-padded batch, built outside the forward through transformers' own
    builders: the causal mask, and for a hybrid family the mapping its
    decoder indexes by layer type — one entry per type ``config.layer_types``
    declares, the causal mask for its attention layers and the 2-D padding
    mask its DeltaNet layers multiply their states by
    ([`PROMPT_MASK_TYPES`][]; any other declared type is refused ``P4`` by
    name, before any mask is built).

    Transformers' builder of that second mask
    (``create_recurrent_attention_mask``) returns the 2-D mask itself for a
    prompt forward — or ``None`` for an unpadded batch, decided by a
    ``torch.all`` over the mask: a device→host synchronization on every eager
    forward, before the first layer launches. Here the mask is always the
    tensor, without asking: the layers multiply by ones, the same arithmetic
    to the bit. That entry is the caller's ``attention_mask`` itself when it
    is already contiguous (a view, no copy — the usual case): the graph
    executor stages it as one more destination beside the frame's mask
    (``graph_cohort.py``), so the alias only spares a copy and nothing rests
    on it; the eager executor caches the mapping for frames it never
    mutates."""
    from transformers.masking_utils import create_causal_mask

    config = model.config
    layer_types: tuple[str, ...] = ()
    if has_recurrent_layers(config):
        layer_types = tuple(dict.fromkeys(config.layer_types))
        unknown = sorted(set(layer_types) - PROMPT_MASK_TYPES)
        if unknown:
            raise ProtocolError(
                "P4",
                f"the {getattr(config, 'model_type', 'loaded')} family declares "
                f"layer types {unknown} beside its recurrent layers; prompt masks "
                f"are built for {sorted(PROMPT_MASK_TYPES)} only",
            )
    with torch.no_grad():
        causal_mask = create_causal_mask(
            config=config,
            inputs_embeds=model.get_input_embeddings()(input_ids),
            attention_mask=attention_mask,
            past_key_values=None,
            position_ids=position_ids,
        )
    if not layer_types:
        return causal_mask
    padding_mask = attention_mask.contiguous()
    return {
        layer_type: padding_mask if layer_type == "linear_attention" else causal_mask
        for layer_type in layer_types
    }


class PointExecutor(ExecutorBase):
    """Execute one concrete document against one loaded model, over hooks."""

    #: the executor a fit's eval passes run on, built on the first pass
    #: (``train._eval_executor``) and kept for the rest of the fit. Lives here
    #: rather than on [`ExecutorBase`][causalab.neural.shared.executor.base.ExecutorBase] because this engine is the one that
    #: trains, and the point executor is the one object whose lifetime is the
    #: fit's. A class-level default rather than an ``__init__`` override, so
    #: the base constructor's keyword-only signature stays type-checked at
    #: every ``PointExecutor(...)`` call site.
    cuda_graphs = False
    # A multi-member fit may keep eager batching when graph eligibility fails.
    # This does not change the point's later inference execution policy.
    fit_cuda_graphs = True
    graph_cache: Any = None
    eval_executor: "PointExecutor | None" = None

    #: ``whole`` / ``fragment`` over the engine's collective
    #: (``docs/model_parallelism.md`` §4): every hook body makes its tensor
    #: whole through this before the shared read / write math and fragments
    #: the result after (`_tap`). The default is world 1 — the
    #: identity, no collective — set per executor by ``make_executor`` from
    #: the engine's collective. A class-level default for the same reason
    #: ``eval_executor`` is one.
    fragments: Fragments = Fragments(SOLO)

    #: This replica's share of every fit minibatch under data parallelism
    #: over rows (``docs/model_parallelism.md`` §8.3; ``rows.py``): the
    #: identity — every row — unless the engine set an active split. A
    #: class-level default for the same reason ``fragments`` is one.
    rows: RowSplit = WHOLE

    #: The position frame of the forward window in flight under context
    #: parallelism (``docs/model_parallelism.md`` §8.4; ``parallel/context.py``):
    #: bound by `_forward_window` for its duration, ``None`` at
    #: ``cp=1`` and between windows. Every tap of the window carries it, so
    #: a ``SequenceSharded`` placement gathers by the frame's chunks. (The
    #: role frames of the position resolver are `frame`, a method.)
    sequence: SequenceFrame | None = None

    #: The run's ``decoding`` block ([`decoding`][causalab.protocol.engine.RunContext.decoding]), set by the engine after construction —
    #: ``None`` (every non-behavioral run) is the greedy argmax, byte for
    #: byte the decode before the field existed; ``sampled`` draws each token
    #: through `_draw` from a generator seeded once per decode window.
    #: A class-level default for the same reason ``eval_executor`` is one.
    decoding: Mapping[str, Any] | None = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        #: prebuilt masks of recent prompt-only forwards (`_model_forward`),
        #: keyed by a window's shape and its rows' first real tokens. Bounded
        #: by the FIFO there (four entries), with no release hook: an eager
        #: executor is discarded with its frames. The graph executor's own
        #: forward never fills it — its masks live in ``_masks``/``_transient``
        self._prompt_masks: dict[Any, Any] = {}

    def continuations(self) -> dict[tuple[str, str], Continuation]:
        """The continuation frame each decoding group produced, by
        ``(model, input role)`` — what the engine publishes as
        ``continuations.json`` for a decoding request."""
        return dict(self._continuations)

    def _tap(self, site: ResolvedSite) -> TapFragments:
        """The ``whole`` / ``fragment`` pair a hook body on ``site`` runs
        through (§4): the executor's fragments at the site's placement — its
        stage-*interior* placement under a pipeline (§6.5): the hook is the
        owner stage's, and the broadcast to the other stages is the forward
        group's post-forward step — with the routing flag and the expert
        count the experts interior needs. A module-boundary tap whose
        integral tensor is the remapped routing table itself (``expert_idx``)
        is flagged ``routing_table``, so its ``whole`` is the §6.3
        reconstruction. Under context parallelism the tap carries the
        window's frame (§8.4)."""
        return TapFragments(
            self.fragments,
            hook_placement(site.placement),
            remapped_routing=site.remapped_routing,
            num_experts=self.bundle.info.num_experts,
            routing_table=(
                site.remapped_routing and site.shape.integral and site.kind != "experts"
            ),
            frame=self.sequence,
        )

    def _sequence_frame(
        self, batch: EncodedBatch, window: RowWindow
    ) -> SequenceFrame | None:
        """The position frame ``window``'s forward runs in under context
        parallelism (§8.4): the whole padded frame's mask over the window's
        rows, from which every rank reads its chunk — ``None`` at ``cp=1``.

        Raises:
            ParseError: ``P4`` naming ``--parallel.context`` — the frame is
                shorter than the context group.
        """
        if self.fragments.size("context") == 1:
            return None
        return SequenceFrame(
            self.fragments.collective, batch.attention_mask[window.slice]
        )

    def _refuse_decode(self) -> None:
        """A decode under ``cp > 1`` is refused by name (§8.4): its steps
        would each be one position on one rank with the KV cache split
        across the group. Identical on every rank — the group size is the
        geometry's.

        Raises:
            ProtocolError: ``P4`` naming ``--parallel.context``.
        """
        size = self.fragments.size("context")
        if size == 1:
            return
        raise ProtocolError(
            "P4",
            f"--parallel.context: a decode (a read at a generated position) is not "
            f"served under cp={size}: the continuation's steps would each be one "
            "position on one rank with the KV cache split across the context "
            "group (docs/model_parallelism.md §8.4); run a decoding document at cp=1",
        )

    @property
    def _stages(self) -> StageForward:
        """This rank's pipeline stage (``stages.py``, §6.5): the stage forward
        and who installs which hook. Built from the bundle and the collective
        at each use, so a collective bound after construction is honoured;
        world 1 is one stage and costs a size lookup. The placed copy is
        checked against the stage once per window (`_forward_window`)."""
        return StageForward.of(self.bundle, self.fragments.collective)

    def share_routing_mismatch(
        self,
        addresses: Mapping[
            Any, tuple[ResolvedSite, list[tuple[str, WriteSpec, ResolvedSite]]]
        ],
        stages: StageForward,
    ) -> None:
        """Make this rank's routing-mismatch record the owners' (§6.5;
        ``parallel/mismatch.py``): for every write at a stage-local address
        the owning stage's counts — made in its hook, where an expert-keyed
        write joins slots by expert — are broadcast over the pipeline and
        merged here, so the publisher's ``routing_mismatch.json`` is world
        1's whichever stage the write sits on. Called once per window where
        the fire tally is agreed, on every rank alike; a pipeline of one is
        the identity and touches neither the collective nor the pending
        counts (world 1 stays bit for bit)."""
        collective = self.fragments.collective
        if collective.size("pipeline") == 1:
            return
        owners = stages.write_owners(addresses)
        if not owners:
            return
        self._routing_mismatch = shared_mismatch(
            self.routing_mismatch, owners, collective
        )

    def _read_taps(
        self, model: str, input_role: str, reads: Iterable[tuple[str, ReadSpec]]
    ) -> dict[str, ReadTap]:
        """The base rule (`ExecutorBase._read_taps`), with a projecting
        read's head run as the pipeline runs it ([`StageForward.projected`][]):
        the head's weights live on the last stage alone — every other stage
        holds a stand-in for it (``sharding.place_stage``) — so the last
        stage projects the broadcast ``ln_final`` rows and the value is
        broadcast, the same tensor on every rank as the world-1 run's. At one
        stage the base rule's taps are returned as they are."""
        taps = super()._read_taps(model, input_role, reads)
        stages = self._stages
        if stages.stages == 1:
            return taps
        return {
            rname: (
                dataclasses.replace(tap, project=stages.projected(tap.project))
                if tap.project is not None
                else tap
            )
            for rname, tap in taps.items()
        }

    @contextlib.contextmanager
    def _attention_backend(self, sites: Iterable[ResolvedSite]) -> Iterator[None]:
        """Expose attention interiors only while the forward needs them.

        Switch before the model creates its mask: eager and FlashAttention
        consume different mask formats. Ordinary module taps, delta kernels,
        and expert taps keep the selected backend. The pattern's read is a
        module tap but still needs eager to return the attention weights.
        """
        # under context parallelism every attention forward gathers its keys
        # and values inside the eager function (§8.4), tapped or not
        needs_eager = self.fragments.size("context") > 1 or any(
            site.interface_slot is not None and site.kind not in {"delta", "experts"}
            for site in sites
        )
        if not needs_eager:
            # Family plugins can expose ordinary module taps without using
            # Transformers' attention dispatch or its configuration fields.
            yield
            return
        model = self.bundle.model
        previous = getattr(model.config, "_attn_implementation", None)
        if previous is None:
            raise ProtocolError(
                "P4",
                "attention-interior taps require a configured attention backend "
                'to restore; load the model with attn_implementation="eager"',
            )
        self.applied_requirements.add("attn_eager")
        if previous == "eager":
            yield
            return

        def restore() -> None:
            try:
                model.set_attn_implementation(previous)
            except BaseException:
                # A partially restored bundle must not survive under a stale
                # load configuration in the process-wide model cache.
                load_model.cache_clear()
                raise

        try:
            model.set_attn_implementation("eager")
            if model.config._attn_implementation != "eager":
                raise ProtocolError(
                    "P4",
                    "attention-interior taps require eager attention, but this "
                    "model cannot switch implementations dynamically; load it "
                    'with attn_implementation="eager"',
                )
            yield
        except BaseException as error:
            try:
                restore()
            except Exception as restore_error:
                # Preserve the primary error's chain without a context cycle:
                # the restore failure's implicit context is the primary error.
                restore_error.__cause__ = error.__cause__
                restore_error.__suppress_context__ = True
                raise error from restore_error
            raise
        else:
            restore()

    def _prefix_key(self, plan: PrefixPlan, window: RowWindow, depth: int) -> PrefixKey:
        key = super()._prefix_key(plan, window, depth)
        backend = getattr(self.bundle.model.config, "_attn_implementation", None)
        if not isinstance(backend, str):
            backend = json.dumps(backend, sort_keys=True)
        # Prefixes cross forward groups, whose interior taps may require
        # different implementations. Never resume eager from an SDPA/flash
        # residual (or vice versa), even with identical weights and inputs.
        return key if backend == "eager" else (key[0], key[1], key[2], key[3], backend)

    # ------------------------------------------------------------------ #
    # group execution
    # ------------------------------------------------------------------ #

    def _run_group(self, model: str, input_role: str) -> None:
        if (model, input_role) in self._groups_run:
            return
        all_taps = [
            (ref.read, self.doc.reads[ref.read])
            for ref in group_reads(self.doc, model, input_role)
        ]
        depth = 0
        taps: list[tuple[str, ReadSpec]] = []
        gen_taps: list[tuple[str, ReadSpec]] = []
        for rname, read in all_taps:
            budget = generated_budget(self.doc, read.pos)
            if budget is None:
                taps.append((rname, read))
            else:
                gen_taps.append((rname, read))
                depth = max(depth, budget)

        if depth:
            self._refuse_decode()  # not served under context parallelism (§8.4)
            self.eos_token_ids()  # Validate stopping before the first forward.

        # an `lm_head` read at named positions taps `ln_final` and projects
        # the gathered rows through the head itself (shared/head.py) — unless
        # this executor differentiates through it, or runs under a pipeline
        # above one stage (`_read_taps`); every other read taps the site it
        # names. What a stage does not serve is refused at the site the read
        # names, whichever the forward captures for it
        read_taps = self._read_taps(model, input_role, taps)
        stages = self._stages
        for rname, tap in read_taps.items():
            _refuse_interior(f"read {rname!r}", tap.site)
            stages.refuse_unserved(f"read {rname!r}", tap.site)
        capture_sites = {rname: tap.capture for rname, tap in read_taps.items()}
        # A continuation read at lm_head is served from kept ln_final
        # activations (d_model, not vocab) and projected at its addressed
        # steps — the same value, without ever building the whole vocabulary
        # for every step. Any other site is captured as itself.
        gen_sites = {
            rname: resolve_site(self.bundle, self.doc.sites[str(read.site)])
            for rname, read in gen_taps
        }
        for rname, site in gen_sites.items():
            _refuse_interior(f"read {rname!r}", site)
            stages.refuse_unserved(f"read {rname!r}", site)
        gen_capture_sites = {
            rname: (
                resolve_site(self.bundle, SiteSpec(component="ln_final"))
                if site.component == "lm_head"
                else site
            )
            for rname, site in gen_sites.items()
        }

        # Cross-point interning (§3): a group another pass already ran under
        # this capture key is served from the shared captures rather than run
        # a second, identical time. Two exemptions, both principled rather
        # than cautious: a **decoding** group's value is the continuation it
        # produced, not activations a later point can replay from a cache (§4
        # exempts it from elision for the same reason); and, inside a fit, any
        # group a **trained parameter can reach** — its activations change
        # every optimizer step, so no earlier pass's capture is this pass's
        # value (`_may_intern`). What a fit *can* share is the rest: the
        # source forward's raw capture is a leaf the trained featurizer is
        # applied to afterwards, so serving it keeps the gradient path intact.
        group_key = self._group_key(model, input_role)
        # the key this pass shares through the store under — None when it may
        # not be served or published at all
        shared_key = group_key if not depth and self._may_intern(model) else None
        interned = self._interned(
            shared_key, (tap_key(site) for site in capture_sites.values())
        )
        decoded: Decoded | None = None
        if interned is not None:
            capture, idx_capture = interned
            # the pass that produced these captures counted its writes'
            # firings (§4 "Fires"); they are this point's counts for the group
            assert shared_key is not None and self.interning is not None
            served_fires = self.interning.cache.fires.get(shared_key)
            if served_fires:
                self.fires[(model, input_role)] = dict(served_fires)
        else:
            capture, idx_capture, decoded = self._forward_group(
                model,
                input_role,
                shared_key=shared_key,
                capture_sites=capture_sites,
                depth=depth,
                gen_capture_sites=gen_capture_sites,
                # the prefix arithmetic (§4 "Resume") is keyed by the group's
                # key whether or not the group may be *served*: the blocks
                # below its first write are fit-constant even when it is not
                prefix=self._prefix_plan(group_key) if not depth else None,
            )

        batch = self._batch(input_role)
        for rname, read in taps:
            tap = read_taps[rname]
            key = tap_key(tap.capture)
            ref = ReadRef(rname, model)
            self._read_values[ref] = self._finalize_read(
                ref,
                read,
                tap.site,
                capture[key],
                batch,
                input_role,
                project=tap.project,
                expert_idx=idx_capture.get(key),
            )

        if depth:
            assert decoded is not None  # a decoding group is never interned
            self._finalize_generated(
                model,
                input_role,
                batch=batch,
                decoded=decoded,
                gen_taps=gen_taps,
                gen_sites=gen_sites,
                gen_capture_sites=gen_capture_sites,
            )
        # this pass has gathered everything it reads from the raw capture;
        # the store owes the key one pass fewer, and drops the capture
        # once no sharer is left (§3 — a capture lives with its sharers, not
        # with the request)
        self._settle(shared_key)
        self._groups_run.add((model, input_role))

    def _forward_group(
        self,
        model: str,
        input_role: str,
        *,
        shared_key: GroupKey | None,
        capture_sites: Mapping[str, ResolvedSite],
        depth: int,
        gen_capture_sites: Mapping[str, ResolvedSite],
        prefix: PrefixPlan | None = None,
    ) -> tuple[dict[TapKey, torch.Tensor], dict[TapKey, torch.Tensor], Decoded | None]:
        """Run one group's forward; return its raw captures, their routing
        tables, and — for a decoding group — what the decode produced.

        What it captures is this point's taps **unioned with every other
        address the campaign asks of the same group key** (§3): that union is
        what lets the single pass a shared key earns serve every point, and
        it is why an eliding engine has to stop at the deepest tap of the
        union rather than of the point that happened to run first.

        The group runs as one forward per row window (`_row_windows`;
        one window unless ``batch_rows`` is set), each with the same hook
        wiring over its own row slice; captures are concatenated in row
        order, so what comes back is indistinguishable from a single forward
        over every row — up to the dtype rounding of a different batch shape.
        A decoding group decodes each window right after its prefill, so the
        KV cache alive at any moment is one window's.

        ``prefix`` is the plan's resume arithmetic for this group (§4
        "Resume"): each window starts at the block below which the forward is
        ``original``'s when the store holds that block's incoming residual
        for these rows, and stores the residuals later groups will want on
        its way (`_forward_window`).

        **The write set is one transaction** (§4 "Fires"), by three
        choices this method keeps in this order: the whole set is resolved
        and built before any hook is installed, every hook edits a clone
        rather than the module's storage, and `_publish` — the only
        step that lets another point see this pass — runs after the last
        window has returned *and* after every member's fire count has been
        checked ([`check_fires`][]). A member
        that fails to resolve, mismatches its operand's shape or fires other
        than the count its kind declares therefore refuses the point with
        nothing published, memoized or written; the counts of a pass that
        did run are the group's ``fires`` record, kept with the store so a
        point served this key records them too. ``shared_key`` is that key —
        ``None`` for a pass that may neither be served nor published.
        """
        # operands first — the acyclic model graph is the schedule skeleton
        names = write_names(self.doc, model) or ()
        for ename in names:
            for ref in operand_reads(self.doc, self.doc.writes[ename].do):
                self.read_value(ref)

        batch = self._batch(input_role)
        addresses = self._resolve_write_addresses(names)
        stages = self._stages
        for site, _ in addresses.values():
            _refuse_interior(f"write at {site.component!r}", site)
        # this point's taps first, then the campaign's — the sites another
        # point will ask of this same forward, so it never has to run it again
        shared_sites: list[ResolvedSite] = []
        if shared_key is not None and self.interning is not None:
            for spec in self.interning.cache.wanted.get(shared_key, ()):
                site = resolve_site(self.bundle, spec)
                _refuse_interior(f"shared read at {site.component!r}", site)
                stages.refuse_unserved(f"shared read at {site.component!r}", site)
                shared_sites.append(site)
        tapped: list[ResolvedSite] = [*capture_sites.values(), *shared_sites]
        for name, site in gen_capture_sites.items():
            refuse_unstackable(name, site)

        # per tap, one entry per window, concatenated on the row axis below
        parts: dict[TapKey, list[torch.Tensor]] = {}
        idx_parts: dict[TapKey, list[torch.Tensor]] = {}
        token_parts: list[torch.Tensor] = []
        step_parts: dict[TapKey, list[torch.Tensor]] = {}
        idx_step_parts: dict[TapKey, list[torch.Tensor]] = {}
        fires = GroupFires()
        # One implementation for the entire KV-cache chain, including
        # prompt-only writes and continuation-only interior reads.
        with self._attention_backend(
            [
                *tapped,
                *gen_capture_sites.values(),
                *(site for site, _ in addresses.values()),
            ]
        ):
            for window in self._row_windows(int(batch.input_ids.shape[0])):
                tally = FireTally()
                write_hooks = self._build_write_hooks(
                    addresses, input_role, batch, window, tally
                )
                capture, idx_capture, prefill = self._forward_window(
                    batch,
                    window,
                    write_hooks=write_hooks,
                    tapped=tapped,
                    depth=depth,
                    prefix=prefix,
                )
                # every member fired the count its kind declares for this
                # forward, or the point is refused here — before this window's
                # captures join the group's and before anything is published;
                # the count is the whole pipeline's, summed over the stages
                # that own the module (§6.5; the identity at world 1)
                tally = summed_fires(tally, Agreements(self.fragments.collective))
                # and what the owner's hooks recorded beside their count —
                # an expert-keyed write's routing-mismatch tally — reaches
                # every stage at the same point (§6.5; the identity at world 1)
                self.share_routing_mismatch(addresses, stages)
                # a state writer's steps are the frame's positions, which the
                # context ranks partition (§8.4): its fired steps are unioned
                # over the context group before the declaration is checked
                tally = whole_steps(
                    tally,
                    self.fragments.collective,
                    _state_members(addresses),
                    padded_len=batch.padded_len,
                )
                check_fires(group_label(model, input_role), tally)
                fires.fold(tally)
                for key, value in capture.items():
                    parts.setdefault(key, []).append(value)
                for key, value in idx_capture.items():
                    idx_parts.setdefault(key, []).append(value)
                if depth:
                    decoded = self._decode_window(
                        batch,
                        window,
                        prefill,
                        depth=depth,
                        sites=gen_capture_sites,
                        # the same resolved write set, kept in force through
                        # the steps when the model declares it (§2.9)
                        writes=addresses
                        if self._writes_during_generation(model)
                        else None,
                        input_role=input_role,
                        fires=fires,
                        group=group_label(model, input_role),
                    )
                    token_parts.append(decoded.generated)
                    for key, value in decoded.steps.items():
                        step_parts.setdefault(key, []).append(value)
                    for key, value in decoded.idx_steps.items():
                        idx_step_parts.setdefault(key, []).append(value)

        capture = {key: _concat_rows(values) for key, values in parts.items()}
        idx_capture = {key: _concat_rows(values) for key, values in idx_parts.items()}
        record = fires.record()
        if record:
            self.fires[(model, input_role)] = record
            if shared_key is not None and self.interning is not None:
                self.interning.cache.fires[shared_key] = record
        # only what a tap actually filled: a placeholder whose module never ran
        # would hand a later point an empty capture instead of letting it run
        # the forward
        self._publish(
            shared_key,
            f"{model}/{input_role}",
            {key: value for key, value in capture.items() if value.numel()},
            idx_capture,
        )
        decoded = (
            Decoded(
                generated=_concat_rows(token_parts),
                steps={key: _concat_rows(v) for key, v in step_parts.items()},
                idx_steps={key: _concat_rows(v) for key, v in idx_step_parts.items()},
            )
            if depth
            else None
        )
        return capture, idx_capture, decoded

    def _forward_window(
        self,
        batch: EncodedBatch,
        window: RowWindow,
        *,
        write_hooks: list[tuple[ResolvedSite, Callable[..., Any]]],
        tapped: list[ResolvedSite],
        depth: int,
        prefix: PrefixPlan | None = None,
        resume: "Resume | Callable[[], Resume] | None" = None,
    ) -> tuple[dict[TapKey, torch.Tensor], dict[TapKey, torch.Tensor], Any]:
        """One forward over ``window``'s rows with every write and tap
        installed; return its raw captures, their routing tables, and the
        model output a decode continues from.

        ``resume`` is the resume decision for this call — the block to start
        at, the residual to hand it, the prefixes to store on the way — and is
        normally derived here from ``prefix`` through `_prefix_window`.
        A cohort forward (``cohort.py``) hands in a *callable* deriving it
        over several executors' rows; it is called inside the stack, after
        the attention backend for this forward is set, because a prefix key
        carries the backend it was computed under (`_prefix_key`) and
        a key derived outside would name the document's backend for a
        residual computed under eager.

        Hooks see contract tensors of the window's row count, and the writers
        they call were built for this window (`_build_write_hooks`), so
        the row a hook holds and the row the write math addresses agree.

        **Resume** (§4). With a ``prefix`` plan whose block the store already
        holds for this window, the forward starts there: the decoder blocks
        below it are swapped out of the model's block list for the duration
        of the call — block 0 for one that returns the cached residual, the
        rest for pass-throughs — and put back in a ``finally``
        (`_resumed`). Embeddings, the causal mask and the rotary tables
        are computed before the block loop and still are; block ``L`` then
        receives exactly the tensor it would have computed, and every hook on
        it and above fires as usual. Whatever the store lacks and a later
        group wants is captured on the way by a *first* pre-hook on the wanted
        blocks (`_storing_prefix`), so a write landing on a block's
        input never reaches the stored residual — and only down to the plan's
        ``write_depth``, below which this pass is still the un-intervened one.
        """
        capture: dict[TapKey, torch.Tensor] = {}
        # the routing table alongside each experts-interface capture — what the
        # `expert:` sub-axis joins on (executor.base._expert_selected)
        idx_capture: dict[TapKey, torch.Tensor] = {}
        # this rank's pipeline stage (§6.5): a hook is installed on the stage
        # holding its module, a write is applied there and skipped elsewhere
        # (its member stays declared in every rank's tally, so the summed count
        # meets the declaration), and a stage-local capture is broadcast from
        # its owner after the forward so every rank's cache holds it
        stages = self._stages
        stages.check_placement(self.bundle)
        with contextlib.ExitStack() as hooks:
            # the position frame this window's forward runs in under context
            # parallelism (§8.4): bound for the attention and kernel wrappers,
            # and carried by every tap built below; None at cp=1
            self.sequence = self._sequence_frame(batch, window)
            hooks.callback(setattr, self, "sequence", None)
            hooks.enter_context(activate(self.sequence))
            # a model off CUDA runs transformers' torch kernels whatever the
            # environment installed (shared/kernels.py); entered before the
            # delta taps so they wrap the implementation that will run
            hooks.enter_context(
                torch_kernel_path(
                    self.bundle.model, on_cuda=self.bundle.devices.is_cuda
                )
            )
            # short sequences on CUDA run the single-chunk kernel
            # (shared/gdn_short/); entered after that guard so it wraps the
            # kernel it bound, before the taps so they see it
            hooks.enter_context(short_seq_kernel_path(self.bundle.model))
            # the norms and the rotary embedding run as fused kernels where
            # each call's plan admits it (kernels/fused_norms.py)
            hooks.enter_context(fused_norm_path(self.bundle.model))
            # Four of the mixer's tensors — and the attention pattern's *write*
            # — are not module boundaries: transformers computes them inside one
            # `attention_interface(...)` call, so a forward hook on the mixer
            # fires after they have already been consumed. They are collected
            # first and installed together, because the interception is one
            # registry entry and nesting two of them would let the inner
            # wrapper's edits replace the outer's.
            interface: dict[int, list[InterfaceTap]] = {}
            experts: dict[int, list[ExpertsTap]] = {}
            delta: dict[Any, list[DeltaTap]] = {}
            batch_size = window.size

            self._enter_write_hooks(
                hooks, write_hooks, interface, experts, delta, batch_size, stages
            )
            for site in tapped:
                key = tap_key(site)
                if key in capture:
                    continue
                capture[key] = torch.empty(0)  # placeholder; filled by the tap
                if not stages.installs(site):
                    continue  # filled by the owner's broadcast after the forward
                if site.kind == "experts":
                    assert site.interface_slot is not None
                    experts.setdefault(id(site.module), []).append(
                        ExpertsTap(
                            slot=site.interface_slot,
                            read=_experts_capture(
                                capture,
                                idx_capture,
                                key,
                                site,
                                batch_size,
                                self._tap(site),
                            ),
                        )
                    )
                    continue
                if site.kind == "delta":
                    assert site.interface_slot is not None
                    delta.setdefault(site.module, []).append(
                        DeltaTap(
                            slot=site.interface_slot,
                            read=_interface_capture(
                                capture, key, site, batch_size, self._tap(site)
                            ),
                        )
                    )
                    continue
                if site.kind == "interface":
                    assert site.interface_slot is not None
                    interface.setdefault(id(site.module), []).append(
                        InterfaceTap(
                            slot=site.interface_slot,
                            read=_interface_capture(
                                capture, key, site, batch_size, self._tap(site)
                            ),
                        )
                    )
                    continue
                hooks.enter_context(
                    _capturing(
                        site.module,
                        site.kind,
                        capture,
                        key,
                        shape=site.shape,
                        tuple_index=site.tuple_index,
                        batch_size=batch_size,
                        tap=self._tap(site),
                    )
                )
            hooks.enter_context(
                attention_interface_taps(
                    {mid: tuple(entries) for mid, entries in interface.items()}
                )
            )
            # the engine's own grouped-experts forward (experts_path.py),
            # entered before the taps so the interior they wrap is its
            hooks.enter_context(lean_experts_path())
            hooks.enter_context(
                experts_interface_taps(
                    {mid: tuple(entries) for mid, entries in experts.items()}
                )
            )
            hooks.enter_context(
                delta_kernel_taps(
                    {mixer: tuple(entries) for mixer, entries in delta.items()},
                    model=self.bundle.model,
                )
            )
            if resume is None:
                resume = self._resume_for(prefix, window)
            elif callable(resume):
                resume = resume()
            blocks = self.bundle.blocks
            last = len(blocks) - 1
            # the residual entering a block is stored by the stage holding it;
            # every other stage records the key alone (`_ABSENT`, below), so
            # the store's keys — what the next window's start is decided from
            # — agree on every rank (§3, "never branch on rank")
            absent_prefixes: list[PrefixKey] = []
            for key, rows in resume.store.items():
                assert self.interning is not None
                if not stages.owns_block(min(key[3], last)):
                    absent_prefixes.append(key)
                    continue
                hooks.enter_context(
                    _storing_prefix(
                        blocks[min(key[3], last)],
                        self.interning.cache.prefixes,
                        key,
                        rows,
                        device=self.bundle.devices.device_of(min(key[3], last)),
                    )
                )
            if resume.cached is not None:
                # the swap makes every hook below `start` unreachable, so
                # nothing installed above may live there — the plan's depth
                # arithmetic promises it; this checks the promise
                _refuse_sites_below(
                    resume.start, [site for site, _ in write_hooks], tapped
                )
                if stages.swaps(resume.start):
                    hooks.enter_context(_resumed(blocks, resume.start, resume.cached))
            if depth == 0 and not taps_head(
                [*tapped, *(site for site, _ in write_hooks)]
            ):
                # nothing reads or writes the head on this forward — the
                # elision past the deepest tap (§4), for the one module past
                # every block: the model runs without its vocabulary
                # projection, and a projecting read (shared/head.py) does its
                # own over the rows it gathered. A decode keeps it: the
                # prefill's logits pick the first token. Under a pipeline the
                # swap lands on the last stage's head and on the stand-ins
                # every other stage holds for it (sharding.place_stage); the
                # empty logits it leaves are what the stages broadcast
                hooks.enter_context(_without_head(self.bundle))
            with torch.enable_grad() if self.grad_enabled else torch.no_grad():
                prefill = self._model_forward(
                    batch,
                    depth,
                    window,
                    resume.start if resume.cached is not None else 0,
                    # a graded forward with a write is `loss.backward()`-ed on
                    # every rank: the residual crossing a stage boundary then
                    # carries the gradient back, and what this rank did not
                    # compute is attached to its graph (stages.py, §7)
                    backward=self.grad_enabled and bool(write_hooks),
                )
            stages.broadcast_captures(tapped, capture, idx_capture)
            stages.attach_received(tapped, capture, prefill)
            if self.interning is not None:
                for key in absent_prefixes:
                    self.interning.cache.prefixes[key] = _ABSENT
            if resume.cached is not None and self.interning is not None:
                # tallied once the pass has returned: a raising pass skipped
                # nothing. A captured cohort's worker resumes from a buffer of
                # its own and has no store to tell (graph_cohort.py)
                self.interning.cache.resumed.append(resume.start)
                if self.interning.counted and prefix is not None:
                    self.interning.cache.decisions.append(
                        Reuse("resumed", f"{prefix.base_key}@{resume.start}")
                    )
        return capture, idx_capture, prefill

    def _enter_write_hooks(
        self,
        hooks: contextlib.ExitStack,
        write_hooks: Sequence[tuple[ResolvedSite, Callable[..., Any]]],
        interface: dict[int, list[InterfaceTap]],
        experts: dict[int, list[ExpertsTap]],
        delta: dict[Any, list[DeltaTap]],
        batch_size: int,
        stages: StageForward,
    ) -> None:
        """Install one forward's write hooks: module-boundary writers enter
        ``hooks`` directly; attention-interface, experts and DeltaNet writers
        are collected into the per-module tables the caller installs as one
        interception each. A site another pipeline stage holds is skipped
        (``stages.installs``); every edit runs through this rank's tap
        fragments (`_tap`). Shared by the prefill
        (`_forward_window`) and by a decode whose model keeps its
        writes in force (`_decode_window`)."""
        for site, fn in write_hooks:
            if not stages.installs(site):
                continue
            if site.kind == "experts":
                assert site.interface_slot is not None
                experts.setdefault(id(site.module), []).append(
                    ExpertsTap(
                        slot=site.interface_slot,
                        edit=_experts_edit(site, fn, batch_size, self._tap(site)),
                    )
                )
                continue
            if site.kind == "delta":
                assert site.interface_slot is not None
                if site.interface_slot == "state":
                    # a state write must feed forward, so it rides the
                    # stepwise substitution's own surface — `fn` here IS
                    # the per-step writer (_build_write_hooks)
                    delta.setdefault(site.module, []).append(
                        DeltaTap(slot="state", edit_state=fn)
                    )
                    continue
                delta.setdefault(site.module, []).append(
                    DeltaTap(
                        slot=site.interface_slot,
                        edit=_interface_edit(site, fn, batch_size, self._tap(site)),
                    )
                )
                continue
            if site.interface_slot is not None:
                interface.setdefault(id(site.module), []).append(
                    InterfaceTap(
                        slot=site.interface_slot,
                        edit=_interface_edit(site, fn, batch_size, self._tap(site)),
                    )
                )
                continue
            _refuse_unfragmentable_write(site)
            hooks.enter_context(
                _installed(
                    site.module,
                    site.kind,
                    fn,
                    shape=site.shape,
                    tuple_index=site.tuple_index,
                    batch_size=batch_size,
                    writeback=site.writeback,
                    site_name=f"{site.component} at layer {site.layer}",
                    tap=self._tap(site),
                )
            )

    def _model_forward(
        self, batch, depth, window, start: int = 0, *, backward: bool = False
    ):
        """The model call over ``window``'s rows — through this rank's
        pipeline stage (``stages.py``, §6.5), which at world 1 is the call
        itself; ``start`` is the block a resumed forward begins at (0 for a
        whole one), what decides which stages run and which receive;
        ``backward`` says the graph will be backward'ed, so the residuals
        crossing the stage boundaries carry the gradient back (§7). Under
        context parallelism (§8.4) the call receives this rank's chunk of
        the padded frame — the ids, the mask and the position ids alike, so
        rotary embeddings see the frame's positions — and the attention
        and DeltaNet wrappers bound to the window's frame do the rest.

        A prompt-only forward of a hybrid family at ``cp=1``, ``pp=1`` gets
        its masks prebuilt ([`prompt_masks`][]): transformers would
        otherwise decide per forward, with a ``torch.all`` over the mask — a
        device→host round trip before the first layer launches — whether its
        DeltaNet layers see a padding mask at all. Under a context group the
        frame's wrappers own the masks; under a pipeline the embedding the
        builder reads is a stand-in on every stage but the first, and the
        stages' mask arithmetic is the model's own either way."""
        input_ids = batch.input_ids[window.slice]
        attention_mask: Any = batch.attention_mask[window.slice]
        position_ids = batch.position_ids()[window.slice]
        if self.sequence is not None:
            input_ids = self.sequence.narrow(input_ids)
            attention_mask = self.sequence.narrow(attention_mask)
            position_ids = self.sequence.narrow(position_ids)
        elif (
            depth == 0
            and self.fragments.size("pipeline") == 1
            and has_recurrent_layers(self.bundle.model.config)
        ):
            # keyed by the window's shape and its rows' first real tokens,
            # which is what a left-padded mask is; a handful kept, since a
            # cohort frame is rebuilt per forward and repeats only by
            # composition
            key = (tuple(input_ids.shape), batch.first_reals[window.slice])
            masks = self._prompt_masks.get(key)
            if masks is None:
                masks = prompt_masks(
                    self.bundle.model, input_ids, attention_mask, position_ids
                )
                if len(self._prompt_masks) >= 4:
                    del self._prompt_masks[next(iter(self._prompt_masks))]
                self._prompt_masks[key] = masks
            attention_mask = masks
        return self._stages.forward(
            self.bundle.model,
            hidden_size=self.bundle.info.hidden_size,
            device=self.bundle.devices.embedding,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            # Enable the cache only when decode will consume it. Keeping it
            # disabled otherwise preserves the kernel path used by goldens.
            use_cache=depth > 0,
            start=start,
            backward=backward,
        )

    def trained_owner(self, names: Iterable[str]) -> TrainedOwner:
        """The pipeline stage that computes the gradients of the featurizers
        ``names`` — the owner of their sites (``stages.py``, §7); the fit
        syncs the trained parameters from it after each update. Nothing is
        owned at world 1; featurizers on two stages are refused by name."""
        return self._stages.trained_owner(names, self.featurizer_sites)

    def _resume_for(self, prefix: PrefixPlan | None, window: RowWindow) -> "Resume":
        """This window's resume decision (`_prefix_window`), as the
        forward consumes it: ``start`` is the deepest *stored* depth the window
        can use, not the plan's ceiling, and every prefix it stores covers the
        window's rows whole."""
        start, cached, store = self._prefix_window(prefix, window)
        return Resume(
            start=start, cached=cached, store={k: None for k in store.values()}
        )

    def _prefix_window(
        self, prefix: PrefixPlan | None, window: RowWindow
    ) -> tuple[int, torch.Tensor | None, dict[int, PrefixKey]]:
        """What this window does about prefixes: the block it starts at with
        the residual to hand block 0's slot (``0, None`` for a full forward),
        and the prefixes it stores — by the plan's depth, with their keys.

        The forward starts at the **deepest stored** prefix at or below the
        plan's ``resume_at`` — not only at ``resume_at`` itself. A scan with
        one point per layer never has the exact depth on hand: the layer-4
        point may store depth 4 at most (its own write bounds it), so the
        layer-8 point finds depth 4, starts there, and stores depth 8 on the
        way for the layer-12 point. Every wanted depth in ``[start,
        write_depth]`` the store lacks is captured, so each point leaves the
        next one its best possible start.

        Keys carry the plan's depth; the block a depth names is that depth
        clamped to the loaded model, so a write past every block reads as
        "start at the last block" and two such plan depths share one block —
        the [`PAST_BLOCKS`][causalab.protocol.positions.alignment.PAST_BLOCKS] key and the last
        block's key then hold the same tensor, a deliberate duplicate that
        keeps the refcount in plan coordinates.
        Only a family whose decoder loop was verified against the swap
        resumes (`_RESUMABLE_MODEL_TYPES`); any other runs whole and
        stores nothing.
        """
        if prefix is None or self.interning is None or not _resumable(self.bundle):
            return 0, None, {}
        cache = self.interning.cache
        last = len(self.bundle.blocks) - 1
        ceiling = min(prefix.resume_at, last)
        # the depths any stored prefix of this identity can have are the
        # campaign's wanted ones, so those are the candidates to look up
        depths = sorted(cache.wanted_prefix_depths.get(prefix.base_key, ()))
        start = 0
        cached: torch.Tensor | None = None
        for candidate in reversed(depths):
            if not 0 < min(candidate, last) <= ceiling:
                continue
            found = cache.prefixes.get(self._prefix_key(prefix, window, candidate))
            if found is not None:
                start, cached = min(candidate, last), found
                break
        store: dict[int, PrefixKey] = {}
        for candidate in depths:
            # a residual this pass never computes un-intervened: below where
            # it starts, or past the block its first write lands in
            if min(candidate, last) < max(start, 1) or candidate > prefix.write_depth:
                continue
            # ...or one no pass is owed any more: every group that could have
            # started from it has settled, and `_settle_prefixes` never
            # revisits a pair — storing it now would leak the residual to the
            # end of the request (the storing pass itself has not settled yet,
            # so a depth it will read is still positive here)
            if cache.prefix_owed.get((prefix.base_key, candidate), 0) <= 0:
                continue
            key = self._prefix_key(prefix, window, candidate)
            if key not in cache.prefixes:
                store[candidate] = key
        return start, cached, store

    # ------------------------------------------------------------------ #
    # generation
    # ------------------------------------------------------------------ #

    def eos_token_ids(self) -> tuple[int, ...]:
        """Explicit request, then model generation config, then tokenizer."""
        config = getattr(self.bundle.model, "generation_config", None)
        ids = (self.decoding or {}).get("eos_token_ids")
        if ids is None:
            ids = getattr(config, "eos_token_id", None)
        if ids is None:
            ids = self.bundle.tokenizer.eos_token_id
        if ids is None:
            return ()
        ids = [ids] if isinstance(ids, int) else list(ids)
        if any(
            type(t) is not int or not 0 <= t < self.bundle.info.vocab_size for t in ids
        ):
            raise ValueError("EOS IDs must be integers within the model vocabulary")
        return tuple(dict.fromkeys(ids))

    def _decode_window(
        self,
        batch: EncodedBatch,
        window: RowWindow,
        prefill: Any,
        *,
        depth: int,
        sites: Mapping[str, ResolvedSite],
        writes: Mapping[
            Any, tuple[ResolvedSite, list[tuple[str, WriteSpec, ResolvedSite]]]
        ]
        | None = None,
        input_role: str = "base",
        fires: GroupFires | None = None,
        group: str = "",
    ) -> Decoded:
        """Greedy-decode ``window``'s rows from their prefill, capturing every
        continuation tap per step.

        The prefill produced the first token; each step here consumes the
        token before it, so ``depth`` tokens need ``depth`` steps and every
        generated position has activations — including the last, whose
        ``lm_head`` value is the distribution *after* it (§2.3).

        **The prefill's write hooks are gone by now**: they lived in the
        prefill's ``ExitStack``, which closed before this runs. Without
        ``writes`` that is the whole of prefill-only writes — an intervention
        reaches the continuation through the first token's logits and through
        what it left in the KV cache, and nothing re-fires per step. With
        ``writes`` — the model's resolved write set, handed over when it
        declares ``writes_during_generation`` (§2.9) — the same writes are
        installed again for the loop and fire once per step at the token being
        decoded; each step is one forward, checked and folded into ``fires``
        like the prefill (§4 "Fires").
        """
        self._refuse_decode()  # not served under context parallelism (§8.4)
        mask = batch.attention_mask[window.slice]
        next_pos = batch.position_ids()[window.slice][:, -1:]
        # one generator per decode window, seeded from the request: the same
        # seed under the same geometry draws the same tokens (tested on CPU
        # fixtures, assumed on a GPU); `None` for a deterministic decode
        generator = _generator(self.decoding, prefill.logits.device)
        nxt = _draw(prefill.logits[:, -1:, :], self.decoding, generator)

        eos_ids = self.eos_token_ids()
        pad = self.bundle.tokenizer.pad_token_id
        pad = int(pad if pad is not None else (eos_ids[0] if eos_ids else 0))
        finished = torch.zeros_like(nxt, dtype=torch.bool)
        tokens: list[torch.Tensor] = [nxt]
        steps: dict[TapKey, list[torch.Tensor]] = {}
        idx_steps: dict[TapKey, list[torch.Tensor]] = {}
        cache = prefill.past_key_values
        with contextlib.ExitStack() as hooks:
            hooks.enter_context(
                torch_kernel_path(
                    self.bundle.model, on_cuda=self.bundle.devices.is_cuda
                )
            )
            hooks.enter_context(short_seq_kernel_path(self.bundle.model))
            # the norms and the rotary embedding run as fused kernels where
            # each call's plan admits it (kernels/fused_norms.py)
            hooks.enter_context(fused_norm_path(self.bundle.model))
            interface: dict[int, list[InterfaceTap]] = {}
            experts: dict[int, list[ExpertsTap]] = {}
            delta: dict[Any, list[DeltaTap]] = {}
            batch_size = window.size
            step_tally: list[FireTally] = [FireTally()]
            members: tuple[str, ...] = ()
            if writes:
                members = tuple(
                    ename for _, entries in writes.values() for ename, _, _ in entries
                )
                self._enter_write_hooks(
                    hooks,
                    self._build_decode_write_hooks(
                        writes, input_role, batch, window, step_tally
                    ),
                    interface,
                    experts,
                    delta,
                    batch_size,
                    self._stages,
                )
            for site in sites.values():
                key = tap_key(site)
                if key in steps:
                    continue
                steps[key] = []
                if site.kind == "experts":
                    assert site.interface_slot is not None
                    idx_steps[key] = []
                    experts.setdefault(id(site.module), []).append(
                        ExpertsTap(
                            slot=site.interface_slot,
                            read=_experts_accumulate(
                                steps[key],
                                idx_steps[key],
                                site,
                                batch_size,
                                self._tap(site),
                            ),
                        )
                    )
                    continue
                if site.kind == "delta":
                    assert site.interface_slot is not None
                    delta.setdefault(site.module, []).append(
                        DeltaTap(
                            slot=site.interface_slot,
                            read=_interface_accumulate(
                                steps[key], site, batch_size, self._tap(site)
                            ),
                        )
                    )
                    continue
                if site.kind == "interface":
                    assert site.interface_slot is not None
                    interface.setdefault(id(site.module), []).append(
                        InterfaceTap(
                            slot=site.interface_slot,
                            read=_interface_accumulate(
                                steps[key], site, batch_size, self._tap(site)
                            ),
                        )
                    )
                    continue
                hooks.enter_context(
                    _accumulating(
                        site.module,
                        site.kind,
                        steps[key],
                        shape=site.shape,
                        tuple_index=site.tuple_index,
                        batch_size=batch_size,
                        tap=self._tap(site),
                    )
                )
            hooks.enter_context(
                attention_interface_taps(
                    {mid: tuple(entries) for mid, entries in interface.items()}
                )
            )
            # the engine's own grouped-experts forward (experts_path.py),
            # entered before the taps so the interior they wrap is its
            hooks.enter_context(lean_experts_path())
            hooks.enter_context(
                experts_interface_taps(
                    {mid: tuple(entries) for mid, entries in experts.items()}
                )
            )
            hooks.enter_context(
                delta_kernel_taps(
                    {mixer: tuple(entries) for mixer, entries in delta.items()}
                )
            )
            for _ in range(depth):
                for eos in eos_ids:
                    finished |= nxt == eos
                mask = torch.cat([mask, torch.ones_like(nxt)], dim=1)
                next_pos = next_pos + 1
                if writes:
                    step_tally[0] = FireTally()
                    step_tally[0].declare(members, 1)
                with torch.enable_grad() if self.grad_enabled else torch.no_grad():
                    out = self.bundle.model(
                        input_ids=nxt,
                        attention_mask=mask,
                        position_ids=next_pos,
                        past_key_values=cache,
                        use_cache=True,
                    )
                if writes:
                    # every member fired once in this step's forward, or the
                    # point is refused here, as after the prefill
                    check_fires(group, step_tally[0])
                    if fires is not None:
                        fires.fold(step_tally[0])
                cache = out.past_key_values
                nxt = _draw(out.logits[:, -1:, :], self.decoding, generator)
                nxt = torch.where(finished, torch.full_like(nxt, pad), nxt)
                tokens.append(nxt)
                if bool(finished.all()):
                    break

        # Keep the declared frame width across microbatches. Post-stop slots
        # are padding, not emitted tokens, and never enter a metric's frame.
        while len(tokens) < depth:
            tokens.append(torch.full_like(nxt, pad))
        for values in (*steps.values(), *idx_steps.values()):
            while len(values) < depth:
                values.append(torch.zeros_like(values[0]))

        # tokens[i] entered step i; the last draw is never consumed, so the
        # generated sequence is exactly the first `depth` of them
        return Decoded(
            generated=torch.cat(tokens[:depth], dim=1),
            steps={key: torch.cat(values, 1) for key, values in steps.items()},
            idx_steps={key: torch.cat(values, 1) for key, values in idx_steps.items()},
        )

    def _finalize_generated(
        self,
        model: str,
        input_role: str,
        *,
        batch: EncodedBatch,
        decoded: Decoded,
        gen_taps: list[tuple[str, ReadSpec]],
        gen_sites: Mapping[str, ResolvedSite],
        gen_capture_sites: Mapping[str, ResolvedSite],
    ) -> None:
        """Build the continuation frame this group's decode produced and
        finalize its reads against it: a row's width is the count before its
        first EOS, positions resolve to decode steps, and an ``lm_head`` read
        is projected from the kept ``ln_final`` activations at those steps."""
        eos_ids = self.eos_token_ids()
        generated = decoded.generated
        rows, steps = int(generated.shape[0]), int(generated.shape[1])
        if eos_ids:
            # every row's first EOS step, or the full depth for a row that
            # never emitted one — one reduction and one host read for the batch
            terminal = torch.zeros_like(generated, dtype=torch.bool)
            for eos in eos_ids:
                terminal |= generated == eos
            first = terminal.int().argmax(dim=1)
            widths = tuple(
                int(w)
                for w in torch.where(
                    terminal.any(dim=1), first, torch.full_like(first, steps)
                ).tolist()
            )
        else:
            widths = (steps,) * rows
        continuation = continuation_frame(self.bundle.tokenizer, generated, widths)
        self._continuations[(model, input_role)] = continuation

        head = None
        for rname, read in gen_taps:
            ref = ReadRef(rname, model)
            site = gen_sites[rname]
            capture_site = gen_capture_sites[rname]
            stacked = decoded.steps[tap_key(capture_site)]
            stacked_idx = decoded.idx_steps.get(tap_key(capture_site))
            dataset_rows = self.role_rows[input_role]
            field = self.role_fields[input_role]
            per_row = [
                resolve_steps(
                    self._spec(read.pos),
                    continuation,
                    row,
                    dataset_row=dataset_rows[row],
                    field=field,
                )
                for row in range(rows)
            ]
            self._read_steps[ref] = per_row
            project = None
            if capture_site is not site:  # ln_final kept, lm_head owed
                head = head or head_module(self.bundle)
                project = head
                # Metrics reduce bounded projections. A saved/transformed read
                # explicitly asks for its tensor and keeps the ordinary path.
                if (
                    not read.featurizer
                    and read.dims is None
                    and ref not in self._saved_raw
                    and not self.grad_enabled
                ):
                    self._deferred_heads[ref] = head
                    project = None
            self._read_values[ref] = self._finalize_read(
                ref,
                read,
                site,
                stacked,
                batch,
                input_role,
                per_row=per_row,
                project=project,
                expert_idx=stacked_idx,
            )

    # ------------------------------------------------------------------ #
    # writes: land the shared math through hooks
    # ------------------------------------------------------------------ #

    def _build_write_hooks(
        self,
        addresses: Mapping[
            Any, tuple[ResolvedSite, list[tuple[str, WriteSpec, ResolvedSite]]]
        ],
        input_role: str,
        batch: EncodedBatch,
        rows: RowWindow,
        tally: FireTally,
    ) -> list[tuple[ResolvedSite, Callable[..., Any]]]:
        """One in-place writer per written-to tap, applying every write at that
        address in class order (the shared write math, ``executor/writes.py``), for
        the forward over ``rows``.

        Returns the *site* rather than its parts: a tap may be a module
        boundary or an attention-interface slot, and only the site knows which.

        Every writer counts its firings into ``tally`` for each member it
        applies (§4 "Fires"): a module-boundary, interface, experts or
        kernel-boundary writer is declared to fire once in this forward, a
        state writer once per distinct step its rows address
        (``executor.writes._state_step_writer``).
        """
        hooks: list[tuple[ResolvedSite, Callable[..., Any]]] = []
        for site, entries in addresses.values():
            members = tuple(ename for ename, _, _ in entries)
            if site.kind == "delta" and site.interface_slot == "state":
                # the one address whose writer is per-step (step, S) -> S:
                # a state edit feeds forward, so the whole-tensor contract
                # cannot express it (executor.writes._state_step_writer)
                hooks.append(
                    (
                        site,
                        self._state_step_writer(
                            entries, input_role, batch, rows, tally
                        ),
                    )
                )
                continue
            tally.declare(members, 1)
            hooks.append(
                (
                    site,
                    self._address_writer(
                        entries, input_role, batch, rows, tally=tally, members=members
                    ),
                )
            )
        return hooks

    def _writes_during_generation(self, model: str) -> bool:
        """§2.9: whether ``model`` keeps its writes in force through the decode
        steps. The un-intervened model has no writes and never does."""
        im = self.doc.intervened_models.get(model)
        return im is not None and bool(im.writes_during_generation)

    def _build_decode_write_hooks(
        self,
        addresses: Mapping[
            Any, tuple[ResolvedSite, list[tuple[str, WriteSpec, ResolvedSite]]]
        ],
        input_role: str,
        batch: EncodedBatch,
        rows: RowWindow,
        tally: list[FireTally],
    ) -> list[tuple[ResolvedSite, Callable[..., Any]]]:
        """One writer per written-to tap for a **decode step** (§2.9
        ``writes_during_generation``): writes land on the token being decoded,
        whatever prompt-frame form their ``pos`` took. Query-axis tensors
        carry that token alone, at position 0; key-axis tensors include the
        cached prefix, so the current token is their last position
        (rule 16 admits only the two forms that mean "this token": ``all`` and
        ``{"index": -1}``). ``tally`` is a one-element holder: the loop swaps
        in a fresh [`FireTally`][] per step, so each step is checked as the
        forward it is.
        """
        hooks: list[tuple[ResolvedSite, Callable[..., Any]]] = []
        for site, entries in addresses.values():
            members = tuple(ename for ename, _, _ in entries)
            if site.kind == "delta" and site.interface_slot == "state":
                raise ProtocolError(
                    "P4",
                    f"write(s) {list(members)} address 'delta_state' in a model "
                    "that keeps its writes in force during generation: a state "
                    "write is applied per recurrence step of the prompt, and a "
                    "decode step's state has no prompt steps to address",
                    reason="unsupported_mechanism",
                )

            def apply(
                tensor: torch.Tensor,
                routing: torch.Tensor | None = None,
                *,
                site: ResolvedSite = site,
                entries: list[tuple[str, WriteSpec, ResolvedSite]] = entries,
                members: tuple[str, ...] = members,
            ) -> None:
                tally[0].fired(members)
                key_axis = site.shape.has_contract_form and any(
                    axis.kind == "position" and axis.name == "key"
                    for axis in site.shape.axes
                )
                position = tensor.shape[1] - 1 if key_axis else 0
                self._apply_writes_to_contract(
                    entries,
                    input_role,
                    batch,
                    tensor,
                    per_row=[[position] for _ in range(rows.size)],
                    rows=rows,
                    routing=routing,
                )

            hooks.append((site, apply))
        return hooks

    def _address_writer(
        self,
        entries: list[tuple[str, WriteSpec, ResolvedSite]],
        input_role: str,
        batch: EncodedBatch,
        rows: RowWindow,
        *,
        tally: FireTally,
        members: tuple[str, ...],
    ) -> Callable[..., None]:
        def apply(tensor: torch.Tensor, routing: torch.Tensor | None = None) -> None:
            tally.fired(members)
            self._apply_writes_to_contract(
                entries, input_role, batch, tensor, rows=rows, routing=routing
            )

        return apply


@dataclasses.dataclass(frozen=True)
class Decoded:
    """What a greedy decode produced over a set of rows: the generated ids,
    ``(batch, depth)``, and per continuation tap its captures stacked on the
    step axis, ``(batch, depth, …)`` — beside the routing table an
    experts-interface tap carries, stacked the same way."""

    generated: torch.Tensor
    steps: dict[TapKey, torch.Tensor]
    idx_steps: dict[TapKey, torch.Tensor]


def _concat_rows(parts: list[torch.Tensor]) -> torch.Tensor:
    """Join one tap's per-window captures on the row axis, in window order.

    A single part is returned as is — the one-window default path hands the
    hook's tensor through untouched. A placeholder a tap never filled stays
    empty in *every* window, and that all-empty tap stays the empty placeholder
    ``_publish`` filters. All or none is the rule: a tap fires on every forward
    of a group or on none of them, so a mix of filled and empty windows means a
    hook missed a window, and dropping the empties would hand back a tensor
    with fewer rows than the group and silently misalign every later row
    index. That fails here, by window, instead.

    Raises:
        RuntimeError: some windows filled the tap and others did not.
    """
    if len(parts) == 1:
        return parts[0]
    filled = [index for index, part in enumerate(parts) if part.numel()]
    if not filled:
        return torch.empty(0)
    if len(filled) != len(parts):
        missing = [index for index in range(len(parts)) if index not in filled]
        raise RuntimeError(
            f"a tap filled {len(filled)} of {len(parts)} row windows (empty in "
            f"windows {missing}) — a tap fires on every forward of a group or on "
            "none, so its per-window captures cannot be concatenated in row order"
        )
    return torch.cat(parts, dim=0)


# --------------------------------------------------------------------------- #
# prefix resume (§4 "Resume")
# --------------------------------------------------------------------------- #

#: The families whose decoder loop the resume swap was read and verified
#: against (transformers 5.16.1): one ``ModuleList`` of blocks at
#: ``model.model.layers``, each called with ``hidden_states`` as its first
#: positional argument and returning the bare tensor, with embeddings, the
#: causal mask(s) and the rotary tables computed *before* the loop, and no
#: per-layer arithmetic outside the block itself (a residual scale, an every-k
#: cross-attention). Per-layer metadata read *off the layer object* is fine —
#: `_StandIn` falls through for it; a choice indexed off
#: ``config.layer_types[i]`` is fine too, since the swap keeps every block at
#: its own index; and the ``all_hidden_states`` collection every loop keeps
#: under ``output_hidden_states`` is harmless only because nothing in causalab
#: requests it — reads come from hooks, and below ``start`` that tuple would
#: hold the stand-ins' outputs. Spelled as [`ModelInfo.family`][causalab.protocol.registry.models.ModelInfo.family] spells
#: them — the **text** config's ``model_type`` (``registry.model_info_from_hf_config``
#: peels ``text_config``), which is why the multimodal families appear as
#: ``*_text``: a real Qwen3.5/3.6-MoE or Gemma-3 checkpoint tops its config
#: with the wrapper spelling (``qwen3_5_moe``, ``gemma3``), and keying on that
#: would switch resume off exactly on the checkpoints it was built for. An
#: allow-list, not a deny-list: a new family is opted in after its loop has
#: been read, never resumed on the strength of a module tree that happens to
#: fit the swap. Anything else runs whole and stores nothing.
_RESUMABLE_MODEL_TYPES: frozenset[str] = frozenset(
    {
        "llama",
        "qwen2",
        "qwen3",
        "qwen3_moe",
        "qwen3_5_text",
        "qwen3_5_moe_text",
        "qwen3_next",
        "mistral",
        "gemma",
        "gemma2",
        "gemma3_text",
    }
)


def _resumable(bundle: ModelBundle) -> bool:
    """Whether the loaded model's family is one the resume swap is verified
    for — the family the loader resolved ([`ModelInfo.family`][causalab.protocol.registry.models.ModelInfo.family], the text
    config's ``model_type``), never the wrapper config's own spelling, and
    not the adapter's structural family either: the allow-list names the
    decoder *loop*, code the model class carries, not the module tree, and
    ``adapter.family`` (``llama_tree``) is coarser than that question. On a
    caller-owned bundle ([`ModelBundle.from_model`][]) the family is the
    caller's assertion, so a fork with a patched loop is theirs to keep off
    the list."""
    return bundle.info.family in _RESUMABLE_MODEL_TYPES


def _state_members(
    addresses: Mapping[
        Any, tuple[ResolvedSite, list[tuple[str, WriteSpec, ResolvedSite]]]
    ],
) -> tuple[str, ...]:
    """The write members landing on a DeltaNet ``state`` slot — the per-step
    writers, whose firings are positions of the frame (``whole_steps``)."""
    return tuple(
        ename
        for site, entries in addresses.values()
        if site.kind == "delta" and site.interface_slot == "state"
        for ename, _, _ in entries
    )


def _refuse_sites_below(start: int, *groups: Iterable[ResolvedSite]) -> None:
    """A forward resumed at ``start`` never calls the blocks below it, so a
    write or tap on one of them would silently not fire. The plan's
    ``resume_at`` is the minimum over exactly these sites, so this cannot trip
    unless the plan and the resolved sites disagree — which is the failure to
    have loudly. Sites outside every block (embeddings, the final norm, the
    head) still run and are exempt.

    Raises:
        AssertionError: a write or tapped site lives inside a skipped block.
    """
    below = sorted(
        {
            (site.component, site.layer)
            for group in groups
            for site in group
            if site.component not in LAYERLESS_COMPONENTS and site.layer < start
        }
    )
    if below:
        raise AssertionError(
            f"a forward resumed at block {start} would skip sites it has to "
            f"run: {below} — the plan's resume depth and the resolved sites "
            "disagree"
        )


class _StandIn(torch.nn.Module):
    """A module standing in for one decoder block for the span of a call.

    It has to satisfy the block's **attribute** contract, not only its call
    contract: a decoder loop may read per-layer metadata off the layer object
    while building the call (``decoder_layer.attention_type``, ``is_sliding``,
    ``layer_idx``), so every attribute the stand-in lacks falls through to the
    block it replaces. That block is held in a plain list — not assigned as an
    attribute — so it is registered neither as a child module nor in the
    stand-in's ``state_dict``; the model's parameters stay where they are."""

    def __init__(self, original: torch.nn.Module) -> None:
        super().__init__()
        self._original = [original]

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            # read off ``__dict__``, not ``self._original``: were the list ever
            # missing, the fall-through would recurse on itself instead of
            # raising the AttributeError a ``hasattr`` probe is waiting for
            original = self.__dict__.get("_original")
            if original is None:
                raise
            return getattr(original[0], name)


class _CachedResidual(_StandIn):
    """Stands in for block 0 of a resumed forward: whatever it is handed, it
    returns the cached residual entering the block the forward resumes at.
    The value is a plain attribute, not a buffer — it is never a parameter of
    the model and must not appear in its state. It carries no graph, and
    neither would the tensor it replaces: the network is frozen at load
    (``loading.py``: ``model.requires_grad_(False)``), so nothing upstream of
    the first write requires grad and the fit's graph begins at the write."""

    def __init__(self, value: torch.Tensor, original: torch.nn.Module) -> None:
        super().__init__(original)
        self.value = value

    def forward(self, hidden_states: Any, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self.value


class _Passthrough(_StandIn):
    """Stands in for a skipped block above block 0: hands the residual on."""

    def forward(self, hidden_states: Any, *args: Any, **kwargs: Any) -> Any:
        return hidden_states


class _ElidedHead(_StandIn):
    """Stands in for the vocabulary head of a forward nothing reads or
    writes the head on (`_without_head`): returns an **empty** tensor
    where the logits would be. Empty rather than the residual it was handed,
    so the one legitimate consumer of a model output's ``.logits`` — the
    decode, which never runs under an elided head — would fail on its first
    index instead of drawing tokens from a plausibly shaped wrong tensor."""

    def forward(self, hidden_states: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        return hidden_states.new_empty(0)


@contextlib.contextmanager
def _without_head(bundle: ModelBundle) -> Iterator[None]:
    """Run the model without its vocabulary projection for the span of one
    forward: the head module — where the family's tree puts it
    (``adapter.tree.lm_head``) — is swapped for `_ElidedHead` and put
    back in a ``finally``. Everything before it runs as it did, every hook
    below it fires as usual, and the head's own ``[rows·seq, d_model] ×
    [d_model, vocab]`` GEMM never launches; a read that wants the head at
    its positions runs the module itself over what it gathered
    (``shared/head.py``).

    Like `_resumed` this **mutates the shared bundle** for the span of
    the call and is not re-entrant: no other forward, swap or
    ``state_dict()`` may run on the model while a window is inside it. The
    caller has checked that no tap and no write of this forward addresses
    the head ([`taps_head`][]) — a hook on
    the swapped-out module would silently never fire."""
    path = adapter_of(bundle).tree.lm_head
    parent_path, _, name = path.rpartition(".")
    parent = walk(bundle.model, parent_path)
    original = getattr(parent, name)
    setattr(parent, name, _ElidedHead(original))
    try:
        yield
    finally:
        setattr(parent, name, original)


@contextlib.contextmanager
def _resumed(blocks: Any, depth: int, cached: torch.Tensor) -> Iterator[None]:
    """Run the model's block loop from block ``depth`` on: for the duration,
    slot 0 of the block list returns ``cached`` and slots ``1..depth-1`` pass
    it through; the real blocks go back in a ``finally``.

    Swapping *entries* of the ``ModuleList`` rather than shortening it keeps
    every per-index lookup the loop does honest — a hybrid tower reads
    ``config.layer_types[i]`` to pick each block's mask — and keeps the real
    blocks' hooks unfired: a skipped block is never called, so a pre-hook on
    it (a test's block counter, a stray capture) sees nothing, which is the
    truth. The decoder loop this relies on — ``hidden_states`` as the first
    positional argument, a bare tensor returned — is the one the families in
    `_RESUMABLE_MODEL_TYPES` were read to have; every other family is
    refused by `_resumable`.

    This **mutates the shared bundle** for the span of the call and is not
    re-entrant: the loaded model is one object per process, so nothing else
    may run a forward on it, swap its blocks, or take its ``state_dict()``
    while a window is inside this context — a state dict taken mid-window
    would lack the swapped layers' parameters. The executor runs its windows
    one at a time, which is what makes the swap safe.
    """
    originals = [blocks[i] for i in range(depth)]
    try:
        blocks[0] = _CachedResidual(cached, originals[0])
        for i in range(1, depth):
            blocks[i] = _Passthrough(originals[i])
        yield
    finally:
        for i, block in enumerate(originals):
            blocks[i] = block


#: The store's entry for a prefix another pipeline stage holds (§6.5): the
#: key is present on every rank — the resume decision reads the keys — and
#: the residual lives on the stage owning the block it enters.
_ABSENT = torch.empty(0)


@dataclasses.dataclass(frozen=True)
class Resume:
    """One forward's resume decision (§4 "Resume"), resolved: the block to
    start at with the residual to hand block 0's slot (``0, None`` for a full
    forward), and the prefixes to store on the way — by key, each over the
    rows of the forward's batch it covers (``None`` for all of them). A
    cohort forward stores one member's rows under that member's key."""

    start: int
    cached: torch.Tensor | None
    store: Mapping[PrefixKey, slice | None]


@contextlib.contextmanager
def _storing_prefix(
    block: Any,
    sink: dict[PrefixKey, torch.Tensor],
    key: PrefixKey,
    rows: slice | None = None,
    *,
    device: torch.device | None = None,
) -> Iterator[None]:
    """Store the residual entering ``block`` under ``key`` — detached, as it
    arrives, before any other pre-hook on the block runs; ``rows`` narrows it
    to a row slice of the batch (a cohort member's rows).

    ``prepend=True`` is what makes an intervened pass a safe source: a write
    landing on this block's *input* rides a later pre-hook and returns edited
    arguments, and this hook has already seen the originals. A whole-batch
    prefix is kept as is rather than cloned: no hook mutates its arguments in
    place (the write hooks clone first), and the block reads it without
    writing. A row slice is cloned. A slice of the batch tensor is a view, and
    a view pins the whole window's residual for as long as the prefix lives,
    which for a minibatch slice is the fit's lifetime (sec. 4 "Fits"). A
    sixteen-row slice out of a 512-row window would pin 32 times the bytes it
    records, and a cohort stores one such slice per member per step.

    ``device`` is the block's own (``DeviceMap.device_of``): this hook runs
    before the loader's crossing moves the residual there, so on a placed
    model the stored tensor is moved itself — the resume cache holds every
    prefix on the device of the block it enters, which is where
    `_CachedResidual` hands it back and where a cohort concatenates
    its members' prefixes. On one device the move is the identity.
    """

    def hook(_module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        hidden = args[0] if args else kwargs["hidden_states"]
        kept = hidden.detach() if rows is None else hidden[rows].detach().clone()
        sink[key] = kept if device is None else kept.to(device)

    handle = block.register_forward_pre_hook(hook, prepend=True, with_kwargs=True)
    try:
        yield
    finally:
        handle.remove()


# --------------------------------------------------------------------------- #
# hook plumbing (mirrors the oracle's _install / capture helpers)
# --------------------------------------------------------------------------- #

#: The world-1 ``whole`` / ``fragment``: the identity, no collective — what a
#: hook installed without a site's tap runs through.


def _interface_edit(
    site: ResolvedSite,
    write: Callable[[torch.Tensor], None],
    batch_size: int,
    tap: TapFragments = IDENTITY_TAP,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Adapt an in-place contract-shaped writer to the interface's protocol.

    The manager hands out a clone and takes back a replacement, which is why
    this can convert, mutate and convert back without any of it reaching the
    model's own storage.

    Under tensor parallelism (§6.2) the slot is this rank's head shard: it is
    made whole before the conversion and the write — so a ``head:`` slice and
    the write math see the global head order — and the edited tensor is
    fragmented back to the rank's heads before it re-enters the library's own
    math (the value multiply, for the pattern). Replicated slots (every
    DeltaNet slot) pass through the identity.
    """

    def edit(native: torch.Tensor) -> torch.Tensor:
        native = tap.whole(native)
        contract = to_contract(native, site.shape, batch_size=batch_size)
        write(contract)
        edited = from_contract(
            contract, site.shape, batch_size=batch_size, native=native
        )
        return tap.fragment(edited)

    return edit


def _interface_capture(
    sink: dict[Any, torch.Tensor],
    key: Any,
    site: ResolvedSite,
    batch_size: int,
    tap: TapFragments = IDENTITY_TAP,
) -> Callable[[torch.Tensor], None]:
    """The read half — the same contract shape ``_capturing`` produces, made
    whole first (§6.2)."""

    def read(native: torch.Tensor) -> None:
        sink[key] = to_contract(tap.whole(native), site.shape, batch_size=batch_size)

    return read


def _interface_accumulate(
    sink: list[torch.Tensor],
    site: ResolvedSite,
    batch_size: int,
    tap: TapFragments = IDENTITY_TAP,
) -> Callable[[torch.Tensor], None]:
    """The read half for a decode: append per step, as ``_accumulating`` does."""

    def read(native: torch.Tensor) -> None:
        sink.append(to_contract(tap.whole(native), site.shape, batch_size=batch_size))

    return read


def _contract_idx(idx: torch.Tensor, batch_size: int) -> torch.Tensor:
    """The routing table in contract form: ``(tokens, top_k)`` token-major →
    ``(batch, position, top_k)`` — the same split every flat_batch shape uses."""
    return idx.reshape(batch_size, -1, idx.shape[-1])


def _experts_edit(
    site: ResolvedSite,
    write: Callable[..., None],
    batch_size: int,
    tap: TapFragments,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Adapt an in-place contract-shaped writer to the experts interface.

    Same contract as `_interface_edit`: the manager hands out a clone in
    the taps' token-major form, this converts it to ``(batch, position,
    feature)``, lets the shared write math mutate it, and converts back. The
    routing table rides along in contract form, ``(batch, position, top_k)``:
    it is what an expert-keyed gate at this address keys its parameters by
    (``executor.writes._written_value``).

    A site naming an ``expert`` writes only that expert's rows: the write math
    runs over the whole contract tensor as usual, and the merge keeps its
    result exactly where the routing table names that expert — an expert no
    token chose therefore receives a write that lands nowhere, the data-fact
    twin of the width-0 read.

    Under expert parallelism (§6.3) the routing table is made global first,
    the token-major view is made whole (this rank's slots plus every other
    rank's, an exact sum), the write runs on the global view, and the result
    is fragmented back to the slots this rank owns.
    """

    def edit(native: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        idx = tap.routing(idx)
        native = tap.whole(native)
        contract = to_contract(native, site.shape, batch_size=batch_size)
        idx_c = _contract_idx(idx, batch_size)  # (b, s, top_k)
        if site.expert is None:
            write(contract, routing=idx_c)
            edited = from_contract(
                contract, site.shape, batch_size=batch_size, native=native
            )
            return tap.fragment(edited, routing=idx)
        original = contract.clone()
        write(contract, routing=idx_c)
        top_k = idx_c.shape[-1]
        per_slot = contract.shape[-1] // top_k
        mask = (
            (idx_c == site.expert)
            .unsqueeze(-1)
            .expand(*idx_c.shape, per_slot)
            .reshape(contract.shape)
        )
        merged = torch.where(mask, contract, original)
        edited = from_contract(merged, site.shape, batch_size=batch_size, native=native)
        return tap.fragment(edited, routing=idx)

    return edit


def _experts_capture(
    sink: dict[Any, torch.Tensor],
    idx_sink: dict[Any, torch.Tensor],
    key: Any,
    site: ResolvedSite,
    batch_size: int,
    tap: TapFragments,
) -> Callable[[torch.Tensor, torch.Tensor], None]:
    """The read half — the same contract shape ``_capturing`` produces, plus
    the routing table the ``expert:`` sub-axis joins on, both global (§6.3)."""

    def read(native: torch.Tensor, idx: torch.Tensor) -> None:
        sink[key] = to_contract(tap.whole(native), site.shape, batch_size=batch_size)
        idx_sink[key] = _contract_idx(tap.routing(idx), batch_size)

    return read


def _experts_accumulate(
    sink: list[torch.Tensor],
    idx_sink: list[torch.Tensor],
    site: ResolvedSite,
    batch_size: int,
    tap: TapFragments,
) -> Callable[[torch.Tensor, torch.Tensor], None]:
    """The read half for a decode: append per step, as ``_accumulating`` does.

    Safe for every experts-interface shape: the interior is token-indexed, so a
    decode step is exactly one position per row and the steps stack on the
    position axis (unlike ``attention_key``, nothing here grows with the
    prefix). The routing table accumulates in lockstep — which experts each
    *generated* token was sent to.
    """

    def read(native: torch.Tensor, idx: torch.Tensor) -> None:
        sink.append(to_contract(tap.whole(native), site.shape, batch_size=batch_size))
        idx_sink.append(_contract_idx(tap.routing(idx), batch_size))

    return read


def _refuse_unfragmentable_write(site: ResolvedSite) -> None:
    """Refuse a module-boundary write the router's expert-parallel payload
    cannot land faithfully (§6.3, §6.6), before any forward.

    ``EpRouterParallel`` zeroes the scores of the slots this rank's experts
    do not own: a write on the scores (``router_scores``, ``ExpertLocal``)
    would need the routing table to fragment, which a module tap does not
    carry, so it is refused by name rather than landed on a guess; its read
    is served (the ``whole`` is the exact sum). A write on the indices
    (``expert_idx``) *is* served: the tap is the routing table itself, and
    ``_installed`` re-masks the scores to the edited table
    (``TapFragments.rescore``), so the two stay one routing.
    """
    placement = site.placement
    if isinstance(placement, StageLocal):
        placement = placement.inner
    if site.kind not in ("in", "out") or not isinstance(placement, ExpertLocal):
        return
    raise ProtocolError(
        "P4",
        f"a write on {site.component!r} at layer {site.layer} under expert "
        "parallelism: each rank holds the slots of its own experts and landing "
        "the edit needs the routing table, which this module tap does not carry "
        "— read the component, or run the write at ep=1 "
        "(docs/model_parallelism.md §6.3, §6.6)",
        reason="component_unavailable",
    )


def _refuse_interior(what: str, site: ResolvedSite) -> None:
    """Refuse a tap this engine has no mechanism for, naming the one that does.

    ``kind="interior"`` marks a tensor computed *inside* a fused forward (the
    per-expert MoE interior; the Gated DeltaNet interior): there is no
    module boundary for a hook and no per-family interface registry to wrap,
    so the tap belongs to the nnsight engine's ``.source`` addressing. Routing
    already keeps such documents away (the component is absent from this
    engine's declaration); this refusal is for one arriving unrouted.
    """
    if site.kind != "interior":
        return
    raise ProtocolError(
        "P4",
        f"{what} addresses {site.component!r}, which lives inside a fused "
        "forward where no pytorch hook can reach — the nnsight engine "
        "serves it (its `.source` address table, "
        "neural/engines/nnsight_tracing/addresses.py). Routing sends such "
        "documents there.",
        reason="component_unavailable",
    )


@contextlib.contextmanager
def _installed(
    module: Any,
    kind: str,
    write: Callable[[torch.Tensor], None],
    *,
    shape: FeatureShape,
    tuple_index: int | None = None,
    batch_size: int = 1,
    writeback: Writeback | None = None,
    site_name: str = "this site",
    tap: TapFragments = IDENTITY_TAP,
) -> Iterator[None]:
    """Install an in-place write hook, converting to the executor's contract.

    ``write`` always sees a ``(batch, position, feature)`` tensor and mutates it
    in place; the model always gets its native shape back. For the default
    ``native`` is handed to [`from_contract`][] because a fused tap's other
    splits live in it and have to survive the write untouched.

    An input tap may declare a ``writeback`` when the enclosing forward has
    already saved that input for a residual addition: the delta is added to
    the declared target's output, at the payload element that target's own tap
    names, without changing what reads capture at the input tap.

    ``tap`` is the site's ``whole`` / ``fragment`` (§4): the native tensor is
    made whole before the conversion and the write, and the edited tensor
    fragmented back to this rank's part before it returns to the model. The
    default is the world-1 identity. A write to the routing table itself
    (``expert_idx`` under expert parallelism) also re-masks the router's
    expert-local scores to the edited table (§6.3, ``TapFragments.rescore``):
    which experts a slot belongs to moved, and its score's owner with it.
    """
    writeback_delta: torch.Tensor | None = None
    if kind == "out":

        def out_hook(_m: Any, _i: Any, out: Any) -> Any:
            native = tap.whole(tap_tensor(out, tuple_index)).clone()
            contract = to_contract(native, shape, batch_size=batch_size)
            write(contract)
            edited = from_contract(
                contract, shape, batch_size=batch_size, native=native
            )
            if tap.routing_table:
                out = rebuild_payload(
                    out,
                    ROUTER_SCORES,
                    tap.rescore(tap_tensor(out, ROUTER_SCORES), edited),
                )
            return rebuild_payload(out, tuple_index, tap.fragment(edited))

        handle = module.register_forward_hook(out_hook)
    else:

        def pre_hook(_m: Any, args: tuple[Any, ...]) -> tuple[Any, ...]:
            nonlocal writeback_delta
            original = args[0]
            native = tap.whole(original).clone()
            contract = to_contract(native, shape, batch_size=batch_size)
            write(contract)
            rewritten = tap.fragment(
                from_contract(contract, shape, batch_size=batch_size, native=native)
            )
            if writeback is not None:
                if writeback_delta is not None:
                    raise RuntimeError(
                        f"{site_name}: the input fired twice before its write-back "
                        f"target {writeback.component!r} — a residual delta would "
                        "be lost"
                    )
                writeback_delta = rewritten - original
            return (rewritten, *args[1:])

        handle = module.register_forward_pre_hook(pre_hook)

    writeback_handle = None
    if writeback is not None:
        index = writeback.tuple_index

        def writeback_hook(_m: Any, _i: Any, out: Any) -> Any:
            nonlocal writeback_delta
            if writeback_delta is None:
                raise RuntimeError(
                    f"{site_name}: {writeback.component!r} returned before the "
                    "input write that produces its residual delta"
                )
            delta, writeback_delta = writeback_delta, None
            return rebuild_payload(out, index, tap_tensor(out, index) + delta)

        # The residual delta is part of the input site's write. Apply it before
        # any absolute write on the target, regardless of document write order.
        writeback_handle = writeback.module.register_forward_hook(
            writeback_hook, prepend=True
        )
    try:
        yield
    finally:
        if writeback_handle is not None:
            writeback_handle.remove()
        handle.remove()


@contextlib.contextmanager
def _capturing(
    module: Any,
    kind: str,
    sink: dict[Any, torch.Tensor],
    key: Any,
    *,
    shape: FeatureShape,
    tuple_index: int | None = None,
    batch_size: int = 1,
    tap: TapFragments = IDENTITY_TAP,
) -> Iterator[None]:
    """Capture a tap's tensor — made whole through ``tap`` (§4) — in the
    executor's contract shape."""
    if kind == "out":

        def out_hook(_m: Any, _i: Any, out: Any) -> None:
            sink[key] = to_contract(
                tap.whole(tap_tensor(out, tuple_index)), shape, batch_size=batch_size
            )

        handle = module.register_forward_hook(out_hook)
    else:

        def pre_hook(_m: Any, args: tuple[Any, ...]) -> None:
            sink[key] = to_contract(tap.whole(args[0]), shape, batch_size=batch_size)

        handle = module.register_forward_pre_hook(pre_hook)
    try:
        yield
    finally:
        handle.remove()


@contextlib.contextmanager
def _accumulating(
    module: Any,
    kind: str,
    sink: list[torch.Tensor],
    *,
    shape: FeatureShape,
    tuple_index: int | None = None,
    batch_size: int = 1,
    tap: TapFragments = IDENTITY_TAP,
) -> Iterator[None]:
    """Like `_capturing`, but append instead of overwrite.

    A decode calls the same modules once per step, so the single-tensor sink
    would keep only the last step. Continuation reads need every step, and
    stacking them on the sequence axis gives a ``(batch, steps, …)`` tensor
    that gathers exactly like a padded frame does.
    """
    if kind == "out":

        def out_hook(_m: Any, _i: Any, out: Any) -> None:
            sink.append(
                to_contract(
                    tap.whole(tap_tensor(out, tuple_index)),
                    shape,
                    batch_size=batch_size,
                )
            )

        handle = module.register_forward_hook(out_hook)
    else:

        def pre_hook(_m: Any, args: tuple[Any, ...]) -> None:
            sink.append(to_contract(tap.whole(args[0]), shape, batch_size=batch_size))

        handle = module.register_forward_pre_hook(pre_hook)
    try:
        yield
    finally:
        handle.remove()


# --------------------------------------------------------------------------- #
# decoding: the argmax, or a seeded draw
# --------------------------------------------------------------------------- #


def _generator(
    decoding: Mapping[str, Any] | None, device: Any
) -> "torch.Generator | None":
    """A generator seeded with the request's decode seed for a ``sampled``
    decode, ``None`` for the deterministic one (workflow spec §2.7)."""
    if decoding is None or decoding.get("mode") != "sampled":
        return None
    generator = torch.Generator(device=device)
    generator.manual_seed(int(decoding["seed"]))
    return generator


def _draw(
    logits: torch.Tensor,
    decoding: Mapping[str, Any] | None,
    generator: "torch.Generator | None",
) -> torch.Tensor:
    """The next token per row from ``logits`` ``(batch, 1, vocab)`` →
    ``(batch, 1)``: the argmax — the greedy decode every document specifies
    and every non-behavioral request keeps — or, under a ``sampled``
    decoding, one draw from ``softmax(logits / temperature)`` restricted to
    the smallest set of tokens whose probability mass reaches ``top_p``
    (nucleus sampling; ``top_p`` 1.0 keeps the whole distribution). The two
    ``argmax`` lines of the decode are this function's only callers; the
    prefill is untouched."""
    if generator is None:
        return logits.argmax(dim=-1)
    scores = logits[:, -1, :].float() / float(decoding["temperature"])  # type: ignore[index]
    probs = torch.softmax(scores, dim=-1)
    top_p = float(decoding["top_p"])  # type: ignore[index]
    if top_p < 1.0:
        sorted_probs, order = probs.sort(dim=-1, descending=True)
        cumulative = sorted_probs.cumsum(dim=-1)
        # drop every token whose mass sits entirely past top_p; the token that
        # crosses the threshold stays, so the kept set is never empty
        drop = cumulative - sorted_probs > top_p
        sorted_probs = sorted_probs.masked_fill(drop, 0.0)
        probs = torch.zeros_like(probs).scatter(-1, order, sorted_probs)
        probs = probs / probs.sum(dim=-1, keepdim=True)
    return torch.multinomial(probs, 1, generator=generator)
