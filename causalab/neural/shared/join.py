"""The data-parallel join (``docs/model_parallelism.md`` §8.3): every
replica's shard of a campaign, placed by point digest in point order and
folded into exactly what one world-1 run of the campaign accumulates before
it writes.

[`ShardOutput`][] is what ``execute_request`` holds for its points once
they have run and before anything is written — the tensor files and metric
tables, the three side tables, the summaries, the result cells, the scoring
checks, and the campaign-level facts the writer's identity stamp reads. A
replica hands its shard to the joiner through the request's
[`Publisher`][causalab.protocol.publish.Publisher]; the joiner calls
[`join_shards`][] and then runs the **same** writer the world-1 run runs
over the same object. Nothing here knows a file format, which is what makes
the joined run byte-identical to the world-1 run by construction — tables,
safetensors and the receipt's ``fires`` / ``scoring`` blocks alike.

[`ordered_shards`][] is the digest key: the shards come back in the
campaign's point order whatever order they arrived in, and a digest two
replicas both hold, a campaign point no replica holds, a digest that is no
point of the campaign, and a shard holding its points out of order are four
refusals by name (§10.3's ``data over points`` row).
"""

from __future__ import annotations

import dataclasses
from typing import Any, Mapping, Sequence

from causalab.neural.shared.results import MetricTable, TensorFile
from causalab.protocol.rules.errors import ProtocolError
from causalab.protocol.results import Resolution

__all__ = ["ShardOutput", "join_shards", "ordered_shards"]


@dataclasses.dataclass
class ShardOutput:
    """One replica's accumulated outputs for the points it ran, in its
    point order (``digests``). Every field is what ``execute_request`` had
    in hand at the moment it would have written, so a join of one shard is
    the world-1 run's own state."""

    digests: tuple[str, ...]
    tensor_files: dict[str, TensorFile]
    metric_files: dict[str, MetricTable]
    train_evals: list[Mapping[str, Any]]
    fit_diagnostics: list[Mapping[str, Any]]
    routing_mismatch: list[Mapping[str, Any]]
    summaries: list[Mapping[str, Any]]
    cells: list[Resolution]
    #: base dataset ref → the ``ScoringCheck`` record the point's pre-forward
    #: check found (``execution.record_scoring``)
    scoring: dict[str, Mapping[str, Any]]
    #: the shard's first point's model realization, as the identity stamp
    #: reads it (``key``, ``revision``, the canonical realization's ``dtype``,
    #: ``quantization``, ``attn_implementation``)
    model: Mapping[str, Any]
    #: every ``model.attn_implementation`` the shard's points declared
    attention_backends: frozenset[str | None]
    #: forward groups this shard's engine actually ran
    forwards: int


def _refuse(message: str) -> ProtocolError:
    return ProtocolError("P4", f"data-parallel join: {message}", path="--parallel.data")


def ordered_shards(
    shards: Sequence[ShardOutput], campaign: Sequence[str]
) -> tuple[ShardOutput, ...]:
    """``shards`` in the campaign's point order, keyed by point digest.

    ``campaign`` is every point digest the run selected, in point order;
    each shard is a contiguous run of it. Refused by name: a digest held by
    two replicas (or twice by one), a campaign point held by none, a digest
    that is no point of the campaign, and a shard whose digests are not in
    the campaign's order.

    Raises:
        ProtocolError: ``P4`` naming ``--parallel.data``, the digest and the
            replica(s).
    """
    index_of = {digest: index for index, digest in enumerate(campaign)}
    if len(index_of) != len(campaign):
        raise _refuse("the campaign lists a point digest twice")
    owner: dict[str, int] = {}
    for replica, shard in enumerate(shards):
        for digest in shard.digests:
            if digest in owner:
                raise _refuse(
                    f"point {digest} was run twice — by replica {owner[digest]} and "
                    f"replica {replica} — one point, one replica"
                )
            if digest not in index_of:
                raise _refuse(
                    f"replica {replica} ran point {digest}, which is not a point of "
                    "the campaign"
                )
            owner[digest] = replica
        indices = [index_of[digest] for digest in shard.digests]
        if indices != sorted(indices):
            raise _refuse(
                f"replica {replica} holds its points out of point order "
                f"(campaign indices {indices})"
            )
    for digest in campaign:
        if digest not in owner:
            raise _refuse(f"point {digest} was run by no replica")
    # each shard's first campaign index places it; an empty shard cannot
    # exist (point_shards refuses more replicas than points), so every
    # shard has one
    return tuple(sorted(shards, key=lambda shard: index_of[shard.digests[0]]))


def join_shards(shards: Sequence[ShardOutput], campaign: Sequence[str]) -> ShardOutput:
    """Fold ``shards`` — placed by [`ordered_shards`][] — into the one
    [`ShardOutput`][] a world-1 run of ``campaign`` would hold: tensor
    files and metric tables extended in point order (a tensor file's file-
    level stamp folded the way ``TensorFile.record_common`` folds it over
    points), side tables, summaries and cells concatenated, forwards summed,
    scoring records and attention backends united, the model realization
    the campaign's first point's. A join of one shard is that shard.
    """
    ordered = ordered_shards(shards, campaign)
    if len(ordered) == 1:
        return ordered[0]
    first = ordered[0]
    tensor_files: dict[str, TensorFile] = {}
    metric_files: dict[str, MetricTable] = {}
    joined = ShardOutput(
        digests=tuple(digest for shard in ordered for digest in shard.digests),
        tensor_files=tensor_files,
        metric_files=metric_files,
        train_evals=[],
        fit_diagnostics=[],
        routing_mismatch=[],
        summaries=[],
        cells=[],
        scoring={},
        model=dict(first.model),
        attention_backends=frozenset().union(*(s.attention_backends for s in ordered)),
        forwards=sum(shard.forwards for shard in ordered),
    )
    for shard in ordered:
        for rel, tensors in shard.tensor_files.items():
            tensor_files.setdefault(rel, TensorFile()).extend(tensors)
        for rel, table in shard.metric_files.items():
            metric_files.setdefault(rel, MetricTable()).rows.extend(table.rows)
        joined.train_evals.extend(shard.train_evals)
        joined.fit_diagnostics.extend(shard.fit_diagnostics)
        joined.routing_mismatch.extend(shard.routing_mismatch)
        joined.summaries.extend(shard.summaries)
        joined.cells.extend(shard.cells)
        joined.scoring.update(shard.scoring)
    return joined
