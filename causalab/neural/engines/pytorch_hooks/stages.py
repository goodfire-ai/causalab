"""Pipeline stages as process ranks (``docs/model_parallelism.md`` §6.5, §7, §8.3).

Pipeline parallelism here is **placement**: the stages run in sequence,
transformers' naive schedule, one send/recv per stage boundary, transcribed
onto the engine's [`Collective`][]
(``send`` / ``recv`` / ``broadcast`` on the ``pipeline`` axis) rather than
called — the loader does not install transformers' stage forward
(``sharding.place_stage``), the executor owns it.

[`StageForward`][] describes a rank's stage and provides these operations:

- **the forward** ([`StageForward.forward`][]): the first stage embeds and
  runs its blocks; every stage but the first receives the residual from the
  stage below and feeds it as ``inputs_embeds``; every stage but the last
  sends its last hidden state on; the last stage applies the final norm and
  the head, and its logits are broadcast so every rank's executor sees the
  same ``logits`` and the metrics run everywhere identically. Masks and
  position ids are computed on every rank from the same batch. Under a
  **resume** (``start > 0``) the stage owning block ``start`` starts from the
  cached residual — its blocks below are the executor's stand-ins
  (``executor._resumed``) — and receives nothing; a stage entirely below it
  runs nothing and sends nothing; the stages above run after receiving.
- **the backward** (§7): a fit's graded forward — one with a write, the
  featurizer's math inside the owner's hook — is ``loss.backward()``-ed once
  on every rank. The residual crossing each boundary then carries the
  gradient back (``shared/parallel/autograd.py``): the stage above receives
  it through ``recv_with_grad``, whose backward sends the loss's gradient
  down; the stage below sends it through ``send_with_grad``, whose backward
  receives that gradient and continues into its blocks and its featurizer.
  What a rank did not compute — the broadcast logits, every stage-local
  capture another stage owns — is ``carry``-ed to the residual this rank
  sent ([`StageOutput.link`][]; an empty leaf on the last stage and on a
  stage that ran nothing), so a rank's ``loss.backward()`` reaches the
  boundary through whatever the loss reads, and reaches it **once**. The
  boundary runs the ``Handoff.ALWAYS`` protocol on both ends — the stage
  above sends the gradient back whatever the stage below trains — agreed
  by the link at the first graded forward and a mismatch refused by name
  on both stages (``autograd.py``). The collective sequence of a graded
  forward is therefore the same on every rank: forward ``recv``, ``send``,
  the logits broadcast, the capture broadcasts; backward the gradient's
  ``recv``, then its ``send``.
- **stage-local hooks**: a hook is installed on the stage that holds the
  module ([`installs`][causalab.neural.engines.pytorch_hooks.stages.StageForward.installs]), at the stage-*interior* placement
  ([`hook_placement`][] — the ``StageLocal`` wrapper stripped, because a
  hook that broadcast mid-forward would wait on a stage that is itself
  waiting to receive this stage's residual). A write is applied by the
  owner's hook and skipped elsewhere; its member is declared in every rank's
  fire tally, so the pipeline's summed count is compared to the declaration
  (``agreements.summed_fires``).
- **broadcast captures** ([`broadcast_captures`][causalab.neural.engines.pytorch_hooks.stages.StageForward.broadcast_captures]): after the forward,
  every ``StageLocal`` capture is broadcast from its owner in one
  deterministic order over the group's sites ([`broadcast_order`][], a
  pure function of the sites' document coordinates — never of the module
  objects, which differ per rank), so every rank's ``ForwardCache`` holds it.
- **share the owner's records** ([`write_owners`][causalab.neural.engines.pytorch_hooks.stages.StageForward.write_owners]): what a write's hook
  recorded on its owner alone — the routing-mismatch counts of an
  expert-keyed write (``parallel/mismatch.py``) — is broadcast from the
  owner at the point the window's fire tally is agreed, in the same
  deterministic order, so the publisher writes ``routing_mismatch.json`` as
  world 1 does whichever stage the write sits on.

**The trained parameters** ([`TrainedOwner`][], §7, §8.3). A featurizer's
parameters live on every rank, but only the stage owning its site — where
its hook runs — computes their gradient; every other rank's optimizer step
moves nothing (or, with a regularizer term, the wrong thing). After each
update the owner's copy is broadcast over the pipeline axis
(``agreements.sync_parameters``) so every rank — the publisher, stage 0,
included — holds the trained parameters before anything reads them: a
trajectory checkpoint, the eval's early-stop snapshot, a controller. The
owner is the ``StageLocal`` stage of the featurizer's sites
([`StageForward.trained_owner`][]); a fit whose trained featurizers sit on
**two different stages** is not supported.

**Unsupported operations**, refused by name: decode (``use_cache``, the KV
cache) under ``pipeline > 1``; a *derived* component (``attention_result``)
under ``pipeline > 1``, whose value is computed from the capture with the
module's own weights, which only the owner holds; a fit training
featurizers on two stages. World 1 is one stage: the forward is the model
call as before, nothing is broadcast, nothing synchronised, nothing refused.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Sequence

import torch

from causalab.neural.shared.executor import TapKey, tap_key
from causalab.neural.shared.parallel.agreements import sync_parameters
from causalab.neural.shared.parallel.autograd import (
    Handoff,
    carry,
    recv_with_grad,
    send_with_grad,
)
from causalab.neural.shared.parallel.collective import Collective
from causalab.neural.shared.parallel.placement import Placement, StageLocal
from causalab.neural.shared.parallel.placements import stage_of
from causalab.neural.shared.sites import ResolvedSite
from causalab.protocol.rules.errors import ProtocolError
from causalab.protocol.parallel import format_geometry

__all__ = [
    "StageForward",
    "StageOutput",
    "TrainedOwner",
    "broadcast_order",
    "hook_placement",
    "stage_hidden",
]

_AXIS = "pipeline"


def hook_placement(placement: Placement) -> Placement:
    """The placement a hook body runs at: a ``StageLocal`` stripped to its
    interior — the broadcast across stages is [`StageForward.broadcast_captures`][]'s,
    after the forward — every other placement as it is."""
    if isinstance(placement, StageLocal):
        return placement.inner
    return placement


def _order_key(site: ResolvedSite) -> tuple[Any, ...]:
    """The document coordinates of a tap — everything [`tap_key`][causalab.neural.shared.executor.cache.tap_key] keys
    on but the module object, which every rank builds for itself."""
    return (
        site.layer,
        site.component,
        site.kind,
        site.interface_slot or "",
        -1 if site.tuple_index is None else site.tuple_index,
        -1 if site.expert is None else site.expert,
        site.shape.describe(),
    )


def broadcast_order(sites: Iterable[ResolvedSite]) -> list[ResolvedSite]:
    """The order the captures of ``sites`` are broadcast in — the same on
    every rank: sorted by the taps' document coordinates, one entry per tap
    (two sites of one tap share a capture and are broadcast once)."""
    seen: set[TapKey] = set()
    unique: list[ResolvedSite] = []
    for site in sites:
        key = tap_key(site)
        if key in seen:
            continue
        seen.add(key)
        unique.append(site)
    return sorted(unique, key=_order_key)


def stage_hidden(model: Any, kwargs: Mapping[str, Any]) -> torch.Tensor:
    """The residual leaving this stage's last block: the base model's
    ``last_hidden_state`` (its final norm is an identity on every stage but
    the last, so this is the raw residual there). A function of its own so
    a test can send it one block early and show the equality break."""
    base = getattr(model, model.base_model_prefix)
    return base(**kwargs).last_hidden_state


def _leaf(device: torch.device) -> torch.Tensor:
    """An empty leaf requiring a gradient: the link of a stage that sent no
    residual — the last stage, or one below a resume that ran nothing — so
    its received values are on a graph and its ``loss.backward()`` runs and
    does nothing at the boundary."""
    return torch.zeros(0, device=device, requires_grad=True)


@dataclasses.dataclass
class StageOutput:
    """What the stage forward returns above world 1: the ``logits`` every
    rank sees — the last stage's, broadcast — and, for a graded forward, the
    ``link`` this rank's received values attach to (module docstring): the
    residual it sent on, or an empty leaf. ``None`` for an inference
    forward, whose values cross the boundary raw."""

    logits: torch.Tensor
    link: torch.Tensor | None = None


@dataclasses.dataclass(frozen=True)
class TrainedOwner:
    """The pipeline stage that computes a fit's featurizer gradients
    (module docstring): ``stage`` is ``None`` when every rank's copy is
    whole — world 1, or sites no stage owns — and nothing is synchronised."""

    collective: Collective
    stage: int | None

    def sync(self, stages: Iterable[torch.nn.Module]) -> None:
        """Every rank takes the owner's copy of each stage's state — the
        parameters and the buffers derived from them — over the pipeline
        axis, in one order everywhere."""
        if self.stage is None:
            return
        sync_parameters(
            (tensor for stage in stages for tensor in stage.state_dict().values()),
            self.stage,
            self.collective,
            axis=_AXIS,
        )


@dataclasses.dataclass(frozen=True)
class StageForward:
    """This rank's pipeline stage and the stage forward (module docstring).

    ``stages`` is the pipeline's size, ``stage`` this rank's coordinate on it
    (the collective's ``rank("pipeline")``), ``num_layers`` the tower's
    depth — what ownership of a block is a function of (``placements.stage_of``,
    the loader's ``stage_layers`` rule).
    """

    collective: Collective
    stages: int
    stage: int
    num_layers: int

    @classmethod
    def of(cls, bundle: Any, collective: Collective) -> StageForward:
        """The stage forward of ``bundle`` on this rank of ``collective``.

        Raises:
            ProtocolError: ``P4`` — the bundle was loaded under a pipeline of
                another size than the collective spans.
        """
        stages = collective.size(_AXIS)
        geometry = getattr(bundle, "geometry", None)
        loaded = int(getattr(geometry, "pipeline", 1))
        if loaded != stages:
            raise ProtocolError(
                "P4",
                f"--parallel.pipeline: the model was loaded under pp={loaded} but "
                f"the collective spans {stages} pipeline stage(s)"
                + (f" ({format_geometry(geometry)})" if geometry is not None else ""),
            )
        return cls(
            collective,
            stages=stages,
            stage=collective.rank(_AXIS),
            num_layers=len(bundle.blocks),
        )

    def check_placement(self, bundle: Any) -> None:
        """The model placed on this rank holds exactly the blocks this stage
        owns — the loader's ``stage_layers`` and the collective's coordinate
        agree. Asked once per forward window, before the resume swap puts a
        stand-in on a held block.

        Raises:
            ProtocolError: ``P4`` — a block held here that another stage owns,
                or one owned here that the placed copy does not hold.
        """
        if self.stages == 1:
            return
        holds = getattr(bundle, "holds", None)
        if holds is None:
            return
        for layer in range(self.num_layers):
            held = bool(holds(layer))
            if held != self.owns_block(layer):
                raise ProtocolError(
                    "P4",
                    f"--parallel.pipeline: this rank is stage {self.stage} of "
                    f"{self.stages}, which "
                    f"{'owns' if self.owns_block(layer) else 'does not own'} block "
                    f"{layer}, but the loaded model {'holds' if held else 'does not hold'} "
                    "it — the placed copy and the collective's coordinate disagree",
                )

    # -- ownership ------------------------------------------------------------

    @property
    def first(self) -> bool:
        return self.stage == 0

    @property
    def last(self) -> bool:
        return self.stage == self.stages - 1

    def owner_of(self, layer: int) -> int:
        """The stage owning block ``layer``."""
        return stage_of(layer, num_layers=self.num_layers, stages=self.stages)

    def owns_block(self, layer: int) -> bool:
        return self.owner_of(layer) == self.stage

    def owner(self, site: ResolvedSite) -> int | None:
        """The stage whose hook fires for ``site`` — its ``StageLocal``
        placement's — or ``None`` for a tap every rank holds."""
        placement = site.placement
        return placement.stage if isinstance(placement, StageLocal) else None

    def installs(self, site: ResolvedSite) -> bool:
        """Whether this rank installs the hook for ``site``: every rank
        installs a replicated tap, the owner alone a stage-local one."""
        owner = self.owner(site)
        return owner is None or owner == self.stage

    def write_owners(
        self,
        addresses: Mapping[Any, tuple[ResolvedSite, Sequence[tuple[str, Any, Any]]]],
    ) -> list[tuple[str, int]]:
        """``(write, owner stage)`` for every write at a stage-local address
        of ``addresses`` (the executor's ``_resolve_write_addresses``), in
        [`broadcast_order`][] of the sites and name order within one — the
        same on every rank, the addresses being the document's. What a
        write's hook recorded on its owner alone travels in this order
        (``parallel/mismatch.py``, §6.5); a write every rank holds is left
        out, its record being everywhere already."""
        by_key = {tap_key(site): entries for site, entries in addresses.values()}
        owners: list[tuple[str, int]] = []
        for site in broadcast_order(site for site, _ in addresses.values()):
            owner = self.owner(site)
            if owner is None:
                continue
            names = sorted(ename for ename, _, _ in by_key[tap_key(site)])
            owners.extend((ename, owner) for ename in names)
        return owners

    def swaps(self, start: int) -> bool:
        """Whether a forward resumed at block ``start`` swaps this rank's
        blocks below it (``executor._resumed``): the stage owning ``start``
        starts from the cached residual; every other stage keeps its blocks —
        a stage below runs nothing, a stage above runs whole after receiving."""
        return start > 0 and self.owns_block(min(start, self.num_layers - 1))

    def trained_owner(
        self,
        names: Iterable[str],
        sites_of: Callable[[str], Sequence[ResolvedSite]],
    ) -> TrainedOwner:
        """The stage that computes the gradients of the featurizers ``names``
        — the ``StageLocal`` owner of their sites, ``sites_of(name)`` being
        every read's and write's resolved site whose chain names the
        featurizer (module docstring). Nothing is owned at world 1. The
        owner is **per fit**: the members of a cohort (``cohort.py``) each
        have their own, so members on different stages fit together.

        Raises:
            ProtocolError: ``P4`` naming ``--parallel.pipeline`` and each
                featurizer with its stage — the trained featurizers' sites sit
                on two or more stages, which are not synchronised.
        """
        if self.stages == 1:
            return TrainedOwner(self.collective, None)
        owners: dict[str, list[int]] = {}
        for name in names:
            stages = {self.owner(site) for site in sites_of(name)}
            owners[name] = sorted(stage for stage in stages if stage is not None)
        found = sorted({stage for stages in owners.values() for stage in stages})
        if len(found) > 1:
            spelled = ", ".join(
                f"{name!r} on "
                + (
                    f"stage {stages[0]}"
                    if len(stages) == 1
                    else "stages " + " and ".join(str(s) for s in stages)
                )
                for name, stages in owners.items()
                if stages
            )
            raise ProtocolError(
                "P4",
                f"--parallel.pipeline: the fit trains featurizers whose sites sit on "
                f"{len(found)} pipeline stages — {spelled} — and under "
                f"pp={self.stages} the trained parameters are synchronised from one "
                "owning stage after each update in this cut "
                "(docs/model_parallelism.md §8.3); fit them in separate documents, "
                "or run this document at pp=1",
            )
        return TrainedOwner(self.collective, found[0] if found else None)

    def refuse_unserved(self, what: str, site: ResolvedSite) -> None:
        """Reject a tap that cannot run across stages: a
        derived component, whose value is computed from the capture with
        the module's own weights on the rank that holds them. A ``train``
        document is served (module docstring: the backward and the trained
        parameters' sync); its one limit, featurizers trained on two stages,
        is [`trained_owner`][]'s refusal.

        Raises:
            ProtocolError: ``P4`` naming ``--parallel.pipeline`` and the component.
        """
        if self.stages == 1 or site.derivation is None:
            return
        raise ProtocolError(
            "P4",
            f"--parallel.pipeline: {what} addresses {site.component!r}, a component "
            f"derived from its capture with the weights of the module on stage "
            f"{self.owner(site)} — under pp={self.stages} only that stage could "
            "compute it, and the reference engine does not yet broadcast a derived "
            "value (docs/model_parallelism.md §6.5); run this document at pp=1",
            reason="component_unavailable",
        )

    def projected(
        self, project: Callable[[torch.Tensor], torch.Tensor]
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        """``project`` — a read's head projection over its gathered
        ``ln_final`` rows (``shared/head.py``) — as this pipeline runs it:
        on the last stage, the one holding the head's weights (every other
        stage holds a stand-in for it, ``sharding.place_stage``), its value
        broadcast to every rank. The rows every rank hands in are the same
        broadcast capture, and the reads are finalized in document order on
        every rank, so the calls pair up. ``project`` itself at one stage."""
        if self.stages == 1:
            return project

        def staged(rows: torch.Tensor) -> torch.Tensor:
            value = project(rows) if self.last else None
            return self.collective.broadcast(value, self.stages - 1, _AXIS)

        return staged

    # -- the forward ----------------------------------------------------------

    def forward(
        self,
        model: Any,
        *,
        hidden_size: int,
        device: torch.device,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        use_cache: bool = False,
        start: int = 0,
        backward: bool = False,
    ) -> Any:
        """One forward of the model over the batch, through the stages
        (module docstring). Returns the output every rank sees — at world 1
        the model's own output of the same call, above it a
        [`StageOutput`][] whose ``logits`` are the last stage's, broadcast.

        ``backward`` says this forward's graph will be ``loss.backward()``-ed
        on every rank (a fit's graded forward with a write): the residual
        crossing each boundary then carries the gradient back, and the
        output's ``link`` is what this rank's received values attach to —
        the residual it sent, or an empty leaf on the last stage and on a
        stage that ran nothing — its logits included on every stage but the
        last. Under ``no_grad`` nothing is recorded, whatever ``backward``.

        Raises:
            ProtocolError: ``P4`` naming ``--parallel.pipeline`` — a decode
                (``use_cache``) under a pipeline above one stage.
        """
        kwargs: dict[str, Any] = {
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "use_cache": use_cache,
        }
        if self.stages == 1:
            return model(input_ids=input_ids, **kwargs)
        if use_cache:
            raise ProtocolError(
                "P4",
                f"--parallel.pipeline: a decode (the KV cache, use_cache) is not served "
                f"under pp={self.stages} in this cut: the continuation's steps would "
                "run the stage forward once per token with the cache split across "
                "stages (docs/model_parallelism.md §6.5); run a decoding document at pp=1",
            )
        graded = backward and torch.is_grad_enabled()
        resume_owner = (
            self.owner_of(min(start, self.num_layers - 1)) if start > 0 else 0
        )
        shape = (*input_ids.shape[:2], hidden_size)
        dtype = next(model.parameters()).dtype
        logits: torch.Tensor | None = None
        link: torch.Tensor | None = None
        if self.stage >= resume_owner:
            if self.first and start == 0:
                kwargs["input_ids"] = input_ids
            elif self.stage == resume_owner and start > 0:
                # the swapped-in block returns the cached residual whatever it is
                # handed: the embeddings' slot is filled and never read
                kwargs["inputs_embeds"] = torch.zeros(shape, dtype=dtype, device=device)
            elif graded:
                kwargs["inputs_embeds"] = recv_with_grad(
                    None,
                    shape,
                    dtype,
                    device,
                    self.stage - 1,
                    _AXIS,
                    self.collective,
                    handoff=Handoff.ALWAYS,
                )
            else:
                kwargs["inputs_embeds"] = self.collective.recv(
                    shape, dtype, device, self.stage - 1, _AXIS
                )
            if self.last:
                logits = model(**kwargs).logits
                if graded:
                    link = _leaf(device)
            else:
                hidden = stage_hidden(model, kwargs).contiguous()
                if graded:
                    # the stage above sends the gradient back whether or not
                    # this stage trains anything: the node exists regardless
                    link = send_with_grad(
                        hidden,
                        self.stage + 1,
                        _AXIS,
                        self.collective,
                        handoff=Handoff.ALWAYS,
                    )
                else:
                    self.collective.send(hidden, self.stage + 1, _AXIS)
        elif graded:
            link = _leaf(device)
        logits = self.collective.broadcast(logits, self.stages - 1, _AXIS)
        if link is not None and not self.last:
            logits = carry(logits, link)
        return StageOutput(logits=logits, link=link)

    # -- the captures ---------------------------------------------------------

    def broadcast_captures(
        self,
        sites: Sequence[ResolvedSite],
        capture: MutableMapping[TapKey, torch.Tensor],
        idx_capture: MutableMapping[TapKey, torch.Tensor],
    ) -> None:
        """Every stage-local capture among ``sites``, broadcast from its
        owner in [`broadcast_order`][], filled into ``capture`` on every
        rank — and, for an experts-interface tap, its routing table into
        ``idx_capture``. A tap every rank holds is not touched; at world 1
        nothing is."""
        if self.stages == 1:
            return
        for site in broadcast_order(sites):
            owner = self.owner(site)
            if owner is None:
                continue
            key = tap_key(site)
            mine = owner == self.stage
            capture[key] = self.collective.broadcast(
                capture[key] if mine else None, owner, _AXIS
            )
            if site.kind == "experts":
                idx_capture[key] = self.collective.broadcast(
                    idx_capture[key] if mine else None, owner, _AXIS
                )

    def attach_received(
        self,
        sites: Sequence[ResolvedSite],
        capture: MutableMapping[TapKey, torch.Tensor],
        output: Any,
    ) -> None:
        """After [`broadcast_captures`][] of a graded forward: every
        capture among ``sites`` another stage owns is attached to this
        rank's link (``output.link``, [`StageOutput`][]), so a loss
        reading it reaches the boundary (module docstring). No collective
        runs here; the routing tables, integers, are not touched; nothing
        is at world 1 or for an inference forward, whose output has no link."""
        link = getattr(output, "link", None)
        if self.stages == 1 or link is None:
            return
        for site in broadcast_order(sites):
            owner = self.owner(site)
            if owner is None or owner == self.stage:
                continue
            key = tap_key(site)
            capture[key] = carry(capture[key], link)
