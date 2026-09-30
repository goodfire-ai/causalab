"""The routing-mismatch record under a pipeline (``docs/model_parallelism.md``
§6.5, §8.3).

A write through an expert-keyed gate records, per (write, layer, example),
how many of the example's written slots found no source slot holding their
expert (``executor_base._align_by_expert``; spec §2.5 ``expert_neuron``).
The record is made inside the write's hook — on the stage that owns the
module under ``pp > 1`` — and read off the **publisher's** point executor
when ``routing_mismatch.json`` is written (``shared/execution.py``). A write
on another stage must therefore share its record with the publisher.

Following the pipeline's stage-local broadcast protocol (§6.5), when each
window's tally is agreed, the owner of every stage-local write broadcasts its
record for that write over the pipeline axis and every other stage merges
it ([`shared_mismatch`][]), in one deterministic order — the writes in
``stages.broadcast_order`` of their sites, a function of the document's
coordinates and the same on every rank. The record crosses as one ``int64``
tensor of ``[layer, example, mismatched, slots]`` rows ([`pack`][] /
[`unpack`][]; the write's name is the caller's, known on every rank from
the document); a write with nothing recorded crosses as an empty one, so the
collective sequence never depends on what a write found. A group of one is
the identity with no call on the collective, so world 1 never flushes a
pending count here and stays bit for bit what it was.
"""

from __future__ import annotations

from typing import Iterable, Mapping

import torch

from causalab.neural.shared.parallel.collective import Collective
from causalab.neural.shared.parallel.placement import Axis

__all__ = [
    "COLUMNS",
    "MismatchKey",
    "MismatchCount",
    "RoutingMismatch",
    "pack",
    "shared_mismatch",
    "unpack",
]

#: ``(write, layer, example)``: the executor's key.
MismatchKey = tuple[str, int, int]
#: ``(mismatched, slots)``: of the slots the write addressed in the example,
#: how many kept their base value for want of a source.
MismatchCount = tuple[int, int]
RoutingMismatch = dict[MismatchKey, MismatchCount]

#: The packed row: ``[layer, example, mismatched, slots]``.
COLUMNS = 4


def pack(
    records: Mapping[MismatchKey, MismatchCount],
    write: str,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """``records``' entries for ``write`` as an ``(n, 4)`` ``int64`` tensor of
    ``[layer, example, mismatched, slots]`` rows on ``device``, sorted by
    ``(layer, example)`` — ``(0, 4)`` when the write recorded nothing."""
    rows = sorted(
        (layer, example, mismatched, slots)
        for (name, layer, example), (mismatched, slots) in records.items()
        if name == write
    )
    return torch.tensor(rows, dtype=torch.int64, device=device).reshape(-1, COLUMNS)


def unpack(write: str, packed: torch.Tensor) -> RoutingMismatch:
    """The inverse of [`pack`][] for ``write``."""
    if packed.dim() != 2 or packed.shape[1] != COLUMNS:
        raise ValueError(
            f"a packed routing-mismatch record is (n, {COLUMNS}); got "
            f"{tuple(packed.shape)}"
        )
    return {
        (write, layer, example): (mismatched, slots)
        for layer, example, mismatched, slots in packed.tolist()
    }


def shared_mismatch(
    records: Mapping[MismatchKey, MismatchCount],
    owners: Iterable[tuple[str, int]],
    collective: Collective,
    *,
    axis: Axis = "pipeline",
) -> RoutingMismatch:
    """``records`` with every write of ``owners`` as its owning stage holds it
    (module docstring): for each ``(write, owner)`` in order, the rank at
    ``owner`` on ``axis`` broadcasts its packed record and every other rank
    receives it; the result is ``records`` overlaid with what was received —
    the owner's copy is the authority, a receiving rank's entries for that
    write are replaced. The order is the caller's and must be the same on
    every rank. A group of one returns ``records`` as a new dict without a
    call on the collective; so does an empty ``owners``.
    """
    merged: RoutingMismatch = dict(records)
    pairs = tuple(owners)
    if collective.size(axis) == 1 or not pairs:
        return merged
    mine = collective.rank(axis)
    # the record crosses the collective, so it is born on its device
    # (``Collective.device``: NCCL refuses a CPU tensor)
    device = collective.device
    for write, owner in pairs:
        # one call site for the owner and the receivers: the ranks' collective
        # sequences are compared call by call (the simulator's Divergence)
        packed = pack(records, write, device) if owner == mine else None
        received = collective.broadcast(packed, owner, axis)
        if owner == mine:
            continue
        for key in [key for key in merged if key[0] == write]:
            del merged[key]
        merged.update(unpack(write, received))
    return merged
