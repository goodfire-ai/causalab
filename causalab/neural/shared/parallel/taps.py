"""One tap's ``whole`` / ``fragment`` — what a hook body holds (``docs/model_parallelism.md`` §4, §6.3).

The executor binds a [`TapFragments`][] per resolved site: the executor's
[`Fragments`][], the site's placement, whether the routing
table the tap carries is this rank's remapped one, whether the tap's tensor
*is* that table, and the model's expert count. A hook body then reads

    native = tap.whole(native)          # the global tensor, on every rank
    …the shared read / write math, unchanged…
    return tap.fragment(edited, routing=…)   # this rank's part again

with the routing table an experts-interface tap rides made global first
(`routing`, §6.3). A tap *on* the routing table — the router's indices
(``expert_idx``) under ``ep_router`` — holds the remapped table itself:
flagged ``routing_table``, its ``whole`` is the §6.3 reconstruction and its
``fragment`` the remap back to this rank's local ids and sentinel
([`remap`][causalab.neural.shared.parallel.taps.TapFragments.remap]), so a read saves the world-1 table and a ``swap`` lands this
rank's view of the swapped one. At world 1 every method is the identity and
none reaches the collective ([`Fragments`][]'s fast path).

Under context parallelism (§8.4) the tap carries the forward's
[`SequenceFrame`][], bound by the executor per forward window,
and its placement is wrapped in a ``SequenceSharded`` — so ``whole`` gathers
the chunks by position after the inner whole, ``fragment`` chunks before the
inner fragment, and the routing table a tap rides or *is* travels the same
way (`routing` gathers it by position, [`remap`][causalab.neural.shared.parallel.taps.TapFragments.remap] and
[`rescore`][causalab.neural.shared.parallel.taps.TapFragments.rescore] chunk the edited global table back first).
"""

from __future__ import annotations

import dataclasses

import torch

from causalab.neural.shared.parallel.collective import SOLO
from causalab.neural.shared.parallel.context import SequenceFrame
from causalab.neural.shared.parallel.fragments import (
    Fragments,
    PlacementError,
    reconstruct_routing,
    remap_routing,
)
from causalab.neural.shared.parallel.placement import (
    REPLICATED,
    Axis,
    ExpertLocal,
    Placement,
    SequenceSharded,
    StageLocal,
)

__all__ = ["IDENTITY_TAP", "TapFragments"]


@dataclasses.dataclass(frozen=True)
class TapFragments:
    fragments: Fragments
    placement: Placement = REPLICATED
    #: the tap's routing table is this rank's ``EpRouterParallel`` output
    remapped_routing: bool = False
    #: the model's routed-expert count, what ``ExpertLocal`` ownership and the
    #: routing reconstruction divide by
    num_experts: int | None = None
    #: the tap's tensor *is* the routing table (the router's indices output):
    #: ``whole`` reconstructs the global table, ``fragment`` remaps it back
    routing_table: bool = False
    #: the forward's position frame under context parallelism (§8.4);
    #: ``None`` at ``cp=1``, where no placement is a sequence chunk
    frame: SequenceFrame | None = None

    def whole(self, native: torch.Tensor) -> torch.Tensor:
        """The global tensor at this tap, identical on every rank."""
        if self.routing_table:
            return self.routing(native)
        return self.fragments.whole(native, self.placement, frame=self.frame)

    def fragment(
        self, edited: torch.Tensor, *, routing: torch.Tensor | None = None
    ) -> torch.Tensor:
        """This rank's part of the edited global tensor; ``routing`` is the
        **global** table ``(…, top_k)`` an ``ExpertLocal`` tap needs."""
        if self.routing_table:
            return self.remap(edited)
        return self.fragments.fragment(
            edited,
            self.placement,
            routing=routing,
            num_experts=self.num_experts,
            frame=self.frame,
        )

    @property
    def _interior(self) -> Placement:
        """The placement inside the stage and sequence wrappers."""
        placement = self.placement
        if isinstance(placement, StageLocal):
            placement = placement.inner
        if isinstance(placement, SequenceSharded):
            placement = placement.inner
        return placement

    @property
    def _sequence(self) -> SequenceSharded | None:
        """The sequence chunk this tap's positions are placed by, ``None``
        when the placement has no context wrapper (or its group is one)."""
        placement = self.placement
        if isinstance(placement, StageLocal):
            placement = placement.inner
        if not isinstance(placement, SequenceSharded):
            return None
        if self.fragments.size(placement.group) == 1:
            return None
        return dataclasses.replace(placement, inner=REPLICATED)

    def _sequence_whole(self, table: torch.Tensor) -> torch.Tensor:
        """A routing table this rank holds for its positions, gathered by
        position — the same chunking as the tap's tensor."""
        sequence = self._sequence
        if sequence is None:
            return table
        return self.fragments.whole(table, sequence, frame=self.frame)

    def _sequence_fragment(self, table: torch.Tensor) -> torch.Tensor:
        """The inverse: a global table at this rank's positions."""
        sequence = self._sequence
        if sequence is None:
            return table
        return self.fragments.fragment(table, sequence, frame=self.frame)

    @property
    def _expert_group(self) -> Axis:
        placement = self._interior
        return placement.group if isinstance(placement, ExpertLocal) else "expert"

    def _count(self, what: str) -> int:
        if self.num_experts is None:
            raise PlacementError(self.placement, f"{what} needs num_experts")
        return self.num_experts

    def routing(self, table: torch.Tensor) -> torch.Tensor:
        """The global routing table from the one this tap was handed: the
        table itself unless it is remapped and the expert group has more than
        one member (§6.3, [`reconstruct_routing`][]) — and,
        under context parallelism, gathered by position after that (§8.4)."""
        group = self._expert_group
        if self.remapped_routing and self.fragments.size(group) > 1:
            table = reconstruct_routing(
                table,
                self._count("reconstructing a remapped routing table"),
                self.fragments.collective,
                group=group,
            )
        return self._sequence_whole(table)

    def rescore(self, scores: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
        """The router's expert-local scores after its table was edited to the
        global ``table``: the scores made whole (each slot's score lives on
        exactly one rank, the sum is exact) and re-masked to the slots the
        new table gives this rank — which experts a slot belongs to moved,
        and the expert-local placement of its score follows (§6.3). The
        identity at world 1 or on an unflagged tap."""
        group = self._expert_group
        if not self.remapped_routing or self.fragments.size(group) == 1:
            return scores
        placement = ExpertLocal(group)
        whole = self.fragments.whole(scores, placement)
        return self.fragments.fragment(
            whole,
            placement,
            routing=self._sequence_fragment(table),
            num_experts=self._count("re-masking the routing scores"),
        )

    def remap(self, table: torch.Tensor) -> torch.Tensor:
        """The inverse of [`routing`][]: an edited **global** table back to
        this rank's local ids, the sentinel on every slot it does not own
        ([`remap_routing`][]) — the identity where the table
        was never remapped."""
        table = self._sequence_fragment(table)
        group = self._expert_group
        size = self.fragments.size(group)
        if not self.remapped_routing or size == 1:
            return table
        return remap_routing(
            table,
            self._count("remapping an edited routing table"),
            self.fragments.collective.rank(group),
            size,
        )


#: The world-1 ``whole`` / ``fragment``: the identity, no collective — what a
#: hook installed without a site's tap runs through.
IDENTITY_TAP = TapFragments(Fragments(SOLO))
