"""KV-head replication: the head arithmetic (``docs/model_parallelism.md``
§6.6), torch-free so ``check`` and ``dry-run`` decide it before any weights.

transformers' plan shards ``k_proj`` / ``v_proj`` colwise over the key-value
heads, which bounds the tensor axis by ``num_kv_heads``. Above that bound
the engine **replicates** the two projections — every rank of the tensor
group holds the whole weight — and each rank keeps the one KV head its
query heads read. The rank's query heads are the contiguous run
``[rank · H/tp, (rank + 1) · H/tp)`` (``Shard(0)`` of ``q_proj``), and query
head ``i`` reads KV head ``i // (H / H_kv)`` — the contiguous repeat the
library's ``repeat_kv`` spells. So the KV heads a rank needs are a contiguous
run too, and the library's own repeat on the rank (``num_key_value_groups``
divided by the replication) reproduces the rank's slice of the global repeat
exactly when either

* ``tp | H_kv`` — the run is ``H_kv / tp`` heads, the colwise shard, or
* ``H_kv | tp`` — the run is one head, held by ``tp / H_kv`` consecutive
  ranks ([`KvHeads.repeat`][]).

Otherwise a rank's query heads **straddle** two KV heads unevenly (Qwen2.5-
0.5B, 14 heads and 2 KV heads, at ``tp=7``: rank 3's query heads 6 and 7 read
KV heads 0 and 1), no contiguous repeat is right, and the geometry is refused
by name ([`kv_refusal`][]). Nothing here reads a tensor: it is the map from
``(rank, local query head)`` to a KV head, and the two facts about it.
"""

from __future__ import annotations

import dataclasses

from causalab.protocol.rules.errors import ProtocolError

__all__ = ["KvHeads", "kv_refusal"]


def fits(num_kv_heads: int, tensor: int) -> bool:
    """Whether a tensor axis of ``tensor`` fits ``num_kv_heads`` KV heads: a
    shard of them (``tensor | num_kv_heads``) or a replication of them
    (``num_kv_heads | tensor``)."""
    return num_kv_heads % tensor == 0 or tensor % num_kv_heads == 0


def kv_refusal(
    *, num_heads: int, num_kv_heads: int, tensor: int, key: str
) -> str | None:
    """The KV fact that fails for ``tensor`` on a model of ``num_heads`` query
    and ``num_kv_heads`` KV heads, as the sentence ``check`` appends to
    ``--parallel.tensor: tp=N``; ``None`` when the axis fits. Plain integers,
    so a registry entry whose KV heads do not divide its heads still gets a
    sentence rather than an exception."""
    if fits(num_kv_heads, tensor):
        return None
    straddle = ""
    if num_heads % tensor == 0 and num_heads % num_kv_heads == 0:
        straddle = (
            f"; with num_heads={num_heads} a rank's {num_heads // tensor} query "
            f"head(s) would straddle two of the KV heads, each of which serves "
            f"{num_heads // num_kv_heads} query heads, so no contiguous repeat "
            "is right"
        )
    else:
        straddle = f" (num_heads={num_heads})"
    return (
        f"neither divides num_kv_heads={num_kv_heads} of model {key!r} (a colwise "
        "shard of the KV heads) nor is a multiple of it (one KV head replicated "
        f"per rank of the tensor group){straddle}"
    )


@dataclasses.dataclass(frozen=True)
class KvHeads:
    """The KV heads of one attention layer under a tensor axis (module
    docstring): ``num_heads`` query heads in ``num_kv_heads`` groups, sharded
    over ``tensor`` ranks.

    Raises:
        ValueError: ``num_kv_heads`` does not divide ``num_heads`` (no GQA
            model), or ``tensor`` does not divide ``num_heads`` (the head
            fact ``check`` states first).
    """

    num_heads: int
    num_kv_heads: int
    tensor: int

    def __post_init__(self) -> None:
        for name in ("num_heads", "num_kv_heads", "tensor"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(
                    f"KvHeads: {name} must be a positive int, got {value!r}"
                )
        if self.num_heads % self.num_kv_heads:
            raise ValueError(
                f"KvHeads: num_kv_heads={self.num_kv_heads} does not divide "
                f"num_heads={self.num_heads}; every query head reads one KV head"
            )
        if self.num_heads % self.tensor:
            raise ValueError(
                f"KvHeads: tensor={self.tensor} does not divide num_heads={self.num_heads}"
            )

    @property
    def groups(self) -> int:
        """Query heads per KV head — the model's ``num_key_value_groups``."""
        return self.num_heads // self.num_kv_heads

    @property
    def local_heads(self) -> int:
        """Query heads per rank, ``H / tp``."""
        return self.num_heads // self.tensor

    @property
    def fits(self) -> bool:
        return fits(self.num_kv_heads, self.tensor)

    @property
    def replicates(self) -> bool:
        """Whether the axis exceeds the KV heads — the K/V projections are
        replicated rather than sharded."""
        return self.tensor > self.num_kv_heads

    @property
    def repeat(self) -> int:
        """How many consecutive ranks hold each KV head: ``tp / H_kv`` when
        replicating, else 1 (each rank holds its own shard)."""
        return self.tensor // self.num_kv_heads if self.replicates else 1

    def _check_fits(self) -> None:
        if not self.fits:
            raise ProtocolError(
                "P4",
                f"--parallel.tensor: tp={self.tensor} "
                + str(
                    kv_refusal(
                        num_heads=self.num_heads,
                        num_kv_heads=self.num_kv_heads,
                        tensor=self.tensor,
                        key="this model",
                    )
                ),
            )

    def kv_heads(self, rank: int) -> range:
        """The contiguous run of KV heads rank ``rank``'s query heads read.

        Raises:
            ProtocolError: ``P4`` — the shape straddles (module docstring).
        """
        self._check_fits()
        first = rank * self.local_heads
        last = first + self.local_heads - 1
        return range(first // self.groups, last // self.groups + 1)

    def kv_head_of(self, rank: int, local_head: int) -> int:
        """The KV head query head ``local_head`` of rank ``rank`` reads —
        ``repeat_kv``'s contiguous repeat, in global head order."""
        self._check_fits()
        return (rank * self.local_heads + local_head) // self.groups

    @property
    def local_groups(self) -> int:
        """The repeat the rank's attention runs with: query heads per KV
        head *held* — the model's ``num_key_value_groups`` when sharding,
        ``H / tp`` (every local query head reads the one head) when
        replicating. ``num_key_value_groups // repeat`` either way."""
        self._check_fits()
        return self.groups // self.repeat
