"""The data-parallel join (``docs/model_parallelism.md`` §8.3): every
replica's shard of a campaign, placed by point digest in point order and
folded into what one world-1 run of the campaign accumulates.

``unit``: the shards come back in point order whatever order they arrive
in; a duplicate digest, a missing digest and a foreign digest are three
refusals by name; a one-shard join is the shard itself; tensor files fold
their file-level stamp the way ``TensorFile.record_common`` does over
points. The mutation drops the digest key and takes shards in arrival
order, producing the wrong row order when arrivals differ from replica
order. Byte identity with world 1 must fail.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from causalab.neural.shared import join as join_module
from causalab.neural.shared.join import ShardOutput, join_shards, ordered_shards
from causalab.neural.shared.results import MetricTable, TensorFile
from causalab.protocol.rules.errors import ProtocolError
from causalab.protocol.estimand import metric_record_identity

pytestmark = pytest.mark.unit

DIGESTS = tuple(f"{i:064x}" for i in range(1, 8))


def _shard(digests: tuple[str, ...], *, model_key: str = "m") -> ShardOutput:
    """A shard whose one metric table has a row per point (``produced_by``)
    and whose one tensor file has an entry per point, stamped with a common
    identity and one per-point field."""
    table = MetricTable()
    tensors = TensorFile()
    for digest in digests:
        # the point's campaign coordinate, so two shards of one campaign
        # never collide on an entry key
        index = DIGESTS.index(digest) if digest in DIGESTS else -1
        coords = {"sites.target.layers": index}
        table.add("iia", [1.0, 0.0], coords, identity=metric_record_identity("match"))
        tensors.add("v", torch.full((2,), float(index)), coords)
        tensors.record_common({"engine": "stub", "produced_by": digest})
    return ShardOutput(
        digests=digests,
        tensor_files={"v.safetensors": tensors},
        metric_files={"iia.json": table},
        train_evals=[{"point": d} for d in digests],
        fit_diagnostics=[],
        routing_mismatch=[],
        summaries=[{"point": d, "metrics": {}} for d in digests],
        cells=[],
        scoring={"weekdays": {"digest": "x", "result": "ok"}},
        model={"key": model_key, "revision": "main", "dtype": "fp32"},
        attention_backends=frozenset({"eager"}),
        forwards=len(digests),
    )


def _rows(shard: ShardOutput) -> list[str]:
    """The point each metric row belongs to, read back from its campaign
    coordinate (a metric row carries its coordinates, not a point digest)."""
    return [
        DIGESTS[int(row["sites.target.layers"])]
        for row in shard.metric_files["iia.json"].rows
    ]


def test_shards_are_placed_in_point_order_whatever_order_they_arrive() -> None:
    first, second = _shard(DIGESTS[:4]), _shard(DIGESTS[4:])
    assert ordered_shards((second, first), DIGESTS) == (first, second)
    assert ordered_shards((first, second), DIGESTS) == (first, second)


def test_the_join_concatenates_everything_in_point_order() -> None:
    first, second = _shard(DIGESTS[:4]), _shard(DIGESTS[4:])
    joined = join_shards((second, first), DIGESTS)
    assert joined.digests == DIGESTS
    assert _rows(joined) == [d for d in DIGESTS for _ in range(2)]
    assert [r["point"] for r in joined.train_evals] == list(DIGESTS)
    assert [s["point"] for s in joined.summaries] == list(DIGESTS)
    assert joined.forwards == 7
    assert joined.scoring == {"weekdays": {"digest": "x", "result": "ok"}}
    assert joined.model == first.model
    assert joined.attention_backends == frozenset({"eager"})
    tensors = joined.tensor_files["v.safetensors"]
    assert list(tensors.entries) == list(
        first.tensor_files["v.safetensors"].entries
    ) + list(second.tensor_files["v.safetensors"].entries)
    # the file-level stamp keeps what every point agrees on, as record_common
    # does over the points of one run: the engine, never the point digest
    assert tensors.metadata == {"engine": "stub"}


def test_a_join_of_one_shard_is_the_shard_itself() -> None:
    shard = _shard(DIGESTS)
    assert join_shards((shard,), DIGESTS) is shard


def test_the_joined_tensor_file_equals_the_one_run_would_accumulate() -> None:
    """Folding the shards' tensor files is the same object the world-1 run
    builds point by point: same entry order, same entry table, same stamp."""
    whole = _shard(DIGESTS).tensor_files["v.safetensors"]
    joined = join_shards(
        (_shard(DIGESTS[:3]), _shard(DIGESTS[3:])), DIGESTS
    ).tensor_files["v.safetensors"]
    assert list(joined.entries) == list(whole.entries)
    assert all(torch.equal(joined.entries[k], whole.entries[k]) for k in whole.entries)
    assert joined.entry_meta == whole.entry_meta
    assert joined.metadata == whole.metadata


def test_attention_backends_and_scoring_are_unions() -> None:
    first, second = _shard(DIGESTS[:2]), _shard(DIGESTS[2:])
    second.attention_backends = frozenset({"sdpa"})
    second.scoring = {"other": {"digest": "y", "result": "ok"}}
    joined = join_shards((first, second), DIGESTS)
    assert joined.attention_backends == frozenset({"eager", "sdpa"})
    assert set(joined.scoring) == {"weekdays", "other"}


def test_a_duplicate_digest_is_refused_naming_it_and_both_replicas() -> None:
    first, second = _shard(DIGESTS[:4]), _shard(DIGESTS[3:])  # DIGESTS[3] twice
    with pytest.raises(ProtocolError) as err:
        ordered_shards((first, second), DIGESTS)
    message = str(err.value)
    assert DIGESTS[3] in message
    assert "replica 0" in message and "replica 1" in message
    assert "twice" in message or "duplicate" in message


def test_a_digest_declared_twice_by_one_replica_is_a_duplicate_too() -> None:
    shard = _shard((DIGESTS[0], DIGESTS[0]))
    with pytest.raises(ProtocolError) as err:
        ordered_shards((shard,), DIGESTS[:1])
    assert DIGESTS[0] in str(err.value) and "replica 0" in str(err.value)


def test_a_missing_digest_is_refused_naming_it() -> None:
    first, second = _shard(DIGESTS[:3]), _shard(DIGESTS[4:])  # DIGESTS[3] nowhere
    with pytest.raises(ProtocolError) as err:
        ordered_shards((first, second), DIGESTS)
    message = str(err.value)
    assert DIGESTS[3] in message and "no replica" in message


def test_a_foreign_digest_is_refused_naming_it_and_its_replica() -> None:
    foreign = "f" * 64
    first, second = _shard(DIGESTS[:4]), _shard((*DIGESTS[4:], foreign))
    with pytest.raises(ProtocolError) as err:
        ordered_shards((first, second), DIGESTS)
    message = str(err.value)
    assert foreign in message and "replica 1" in message
    assert "not a point of the campaign" in message


def test_a_replica_whose_shard_is_out_of_point_order_is_refused() -> None:
    """A shard is a contiguous run of the campaign in point order; one that
    holds the right digests in the wrong order is refused rather than
    reordered row by row (the rows are the replica's to order)."""
    scrambled = _shard((DIGESTS[1], DIGESTS[0]))
    with pytest.raises(ProtocolError) as err:
        ordered_shards((scrambled, _shard(DIGESTS[2:])), DIGESTS)
    assert "replica 0" in str(err.value) and "point order" in str(err.value)


def test_metric_tables_are_folded_row_for_row() -> None:
    first, second = _shard(DIGESTS[:1]), _shard(DIGESTS[1:])
    joined = join_shards((first, second), DIGESTS)
    whole = _shard(DIGESTS)
    assert joined.metric_files["iia.json"].rows == whole.metric_files["iia.json"].rows


# --------------------------------------------------------------------------- #
# the mutation: drop the digest key
# --------------------------------------------------------------------------- #


def test_mutation_dropping_the_digest_key_breaks_the_point_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A join that takes the shards as they arrive — no digest, no order —
    concatenates the second replica's rows first when the gather delivers
    them first, and the joined table is no longer the world-1 table. The
    unmutated join is."""
    first, second = _shard(DIGESTS[:4]), _shard(DIGESTS[4:])
    truth = _rows(_shard(DIGESTS))
    assert _rows(join_shards((second, first), DIGESTS)) == truth

    def as_they_arrive(shards: Any, campaign: Any) -> tuple[ShardOutput, ...]:
        return tuple(shards)

    monkeypatch.setattr(join_module, "ordered_shards", as_they_arrive)
    assert _rows(join_shards((second, first), DIGESTS)) != truth
