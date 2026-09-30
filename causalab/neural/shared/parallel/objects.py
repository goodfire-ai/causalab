"""Python objects over the tensor collectives (``docs/model_parallelism.md``
§3, §8.3).

The one place the engine moves something other than a tensor between
ranks is the publishers' hand-off: every publishing rank of a data group
hands its shard of the run's results to the joiner. ``torch.distributed``
has ``gather_object`` for it; this module spells the same wire protocol over
the [`Collective`][] — the payload pickled to bytes, the
lengths all-gathered as one ``int64`` cell each, then each member's bytes
sent to the joiner as one ``uint8`` row of its own length — so the hand-off
runs under the tests' ``SimulatedWorld`` and is covered by the collective
contract, where a raw ``gather_object`` was reachable only from the spawned
``gloo`` smokes.

The joiner alone receives every payload; every other member sends its own
and holds nothing. The payload is a `ShardOutput` — the
replica's tensor files, its saved activations in memory before anything is
written — so the shape matters on the card: the hand-off costs the joiner
one row per member and every other member its own row, where an all-gather
would have put ``size × widest`` bytes on every member of the group. Pickle
is the codec: the payloads are the run's own records, written by the ranks
of one launch, never read from outside the process group.
"""

from __future__ import annotations

import pickle
from typing import Sequence, TypeVar

import torch

from causalab.neural.shared.parallel.collective import Collective
from causalab.neural.shared.parallel.placement import Axis

__all__ = ["gather_objects"]

T = TypeVar("T")


def gather_objects(
    payload: T, axis: Axis, collective: Collective, *, to: int = 0
) -> tuple[T, ...] | None:
    """Every member's ``payload`` on the group of ``axis``, in rank order, on
    the member at group-local index ``to``; ``None`` on every other member
    (module docstring). The tensors live on ``collective.device``. A group
    of one is the identity, with no call on the collective.

    Raises:
        ValueError: ``to`` is outside the group.
    """
    size = collective.size(axis)
    if size == 1:
        return (payload,)
    if not 0 <= to < size:
        raise ValueError(
            f"gather_objects: to={to} is outside the {axis} group of {size}"
        )
    device = collective.device
    blob = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    length = torch.tensor([len(blob)], dtype=torch.int64, device=device)
    lengths: Sequence[int] = collective.all_gather(length, 0, axis).tolist()
    mine = torch.frombuffer(bytearray(blob), dtype=torch.uint8).to(device)
    me = collective.rank(axis)
    if me != to:
        collective.send(mine, to, axis)
        return None
    rows: list[torch.Tensor] = []
    for member, length_of in enumerate(lengths):
        if member == me:
            rows.append(mine)
            continue
        rows.append(collective.recv((length_of,), torch.uint8, device, member, axis))
    return tuple(pickle.loads(row.cpu().numpy().tobytes()) for row in rows)
