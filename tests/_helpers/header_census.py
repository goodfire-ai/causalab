"""A checkpoint's header census (``tests/golden/parallel_headers_*.json``):
every tensor's dtype and shape, compressed by wildcarding the block and
expert indices — a pattern with the ``ranges`` of its indices when they are
a full product, the explicit ``indices`` otherwise (the A3B's DeltaNet
layers are 30 of the 40) — captured off a Hub snapshot's safetensors
headers, torch-free (``protocol/checkpoint_census.py``). `load_census`
expands a census back to one ``(dtype, shape)`` per checkpoint key, which is
all the memory pre-flight reads off a header; `compress` is the
inverse, so ``compress(expand(p)) == p`` on every committed census and a
capture on a node is ``census_record`` of the cached shards.

Three censuses are committed:

- `A3B_CENSUS` — ``Qwen/Qwen3.6-35B-A3B``, 1045 tensors over 26
  shards, captured off a cached snapshot's headers (2026-09-16);
- `LLAMA70B_CENSUS` — ``meta-llama/Llama-3.1-70B``, 723 tensors over
  30 shards (131.42 GiB in bf16): the model that fits no single card
  (``docs/model_parallelism.md`` §10.6 "The large model", §11). Its patterns
  are the ``LlamaForCausalLM`` key set the checkpoint's ``config.json``
  implies, its shard count and snapshot the Hub's listing of ``main``
  (2026-09-18); the tensor bytes agree with the shards' sizes to the byte
  but for the thirty headers. ``python -m tests._helpers.header_census
  meta-llama/Llama-3.1-70B --check`` holds it to the real headers on a machine
  whose Hub cache holds the checkpoint (download it there first), and ``--out``
  rewrites it from them;
- `GEMMA2_9B_CENSUS` — ``google/gemma-2-9b``, 464 tensors over 8
  shards, **stored in fp32** (34.43 GiB on disk, 17.21 GiB held in the
  document's bf16): the second-family checkpoint whose load converts
  (``docs/model_parallelism.md`` §5.3 "the load's peak under a dtype
  conversion", §11), derived the 70B's way — the ``Gemma2ForCausalLM`` key
  set its ``config.json`` implies (a tied head, so no ``lm_head.weight``),
  the shard count and snapshot the Hub's listing of ``main`` (2026-09-18) —
  and held to the node's headers by ``--check`` the same way.

`fake_hub_cache` lays a census out as a Hub cache of **header-only**
shards (the 8-byte length, the JSON table, no data), so the torch-free
``dry-run`` estimate runs on the CPU against the real key set with no weight
on disk.
"""

from __future__ import annotations

import argparse
import itertools
import json
import struct
import sys
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Mapping

from causalab.protocol.checkpoint_census import ITEMSIZES, read_headers

__all__ = [
    "A3B_CENSUS",
    "Census",
    "CensusAmbiguity",
    "GEMMA2_9B_CENSUS",
    "LLAMA70B_CENSUS",
    "census_record",
    "compress",
    "expand",
    "fake_hub_cache",
    "load_census",
    "main",
]

_GOLDEN = Path(__file__).resolve().parents[1] / "golden"
A3B_CENSUS = _GOLDEN / "parallel_headers_a3b.json"
LLAMA70B_CENSUS = _GOLDEN / "parallel_headers_llama70b.json"
GEMMA2_9B_CENSUS = _GOLDEN / "parallel_headers_gemma2_9b.json"

#: One ``(dtype, shape)`` per checkpoint key — what `expand` yields.
Census = dict[str, tuple[str, tuple[int, ...]]]

_INDEX_NAME = "model.safetensors.index.json"


class CensusAmbiguity(ValueError):
    """Two keys of one wildcard pattern carry different dtypes or shapes, so
    the pattern cannot stand for both."""

    def __init__(self, pattern: str, first: tuple[Any, ...], second: tuple[Any, ...]):
        self.pattern = pattern
        super().__init__(
            f"the pattern {pattern!r} would stand for tensors of {first!r} and "
            f"{second!r}; a census pattern has one dtype and one shape"
        )


def expand(patterns: Mapping[str, Mapping[str, Any]]) -> Census:
    """Every checkpoint key with its ``(dtype, shape)`` from the census's
    patterns: each ``*`` replaced by the pattern's indices, in order."""
    out: Census = {}
    for pattern, entry in patterns.items():
        shape = tuple(int(n) for n in entry["shape"])
        if "ranges" in entry:
            combos: Any = itertools.product(
                *[range(lo, hi + 1) for lo, hi in entry["ranges"]]
            )
        else:
            combos = [tuple(t) for t in entry["indices"]] or [()]
        for combo in combos:
            key = pattern
            for index in combo:
                key = key.replace("*", str(index), 1)
            out[key] = (str(entry["dtype"]), shape)
    return out


def compress(census: Mapping[str, tuple[str, tuple[int, ...]]]) -> dict[str, Any]:
    """The inverse of `expand`: every purely numeric dotted segment of
    a key becomes ``*``, keys of one pattern are grouped, and the group's
    indices are written as ``ranges`` when they fill the product of their
    per-slot ranges (a single index included: ``[[0, 0]]``), as ``indices``
    otherwise (``[]`` for a key with no index).

    Raises:
        CensusAmbiguity: two keys of one pattern differ in dtype or shape.
    """
    groups: dict[str, tuple[str, list[int], set[tuple[int, ...]]]] = {}
    for key, (dtype, shape) in census.items():
        parts = key.split(".")
        pattern = ".".join("*" if part.isdigit() else part for part in parts)
        indices = tuple(int(part) for part in parts if part.isdigit())
        found = groups.get(pattern)
        if found is None:
            groups[pattern] = (dtype, list(shape), {indices})
            continue
        if found[0] != dtype or found[1] != list(shape):
            raise CensusAmbiguity(pattern, (found[0], tuple(found[1])), (dtype, shape))
        found[2].add(indices)
    out: dict[str, Any] = {}
    for pattern in sorted(groups):
        dtype, shape, index_set = groups[pattern]
        indices = sorted(index_set)
        entry: dict[str, Any] = {"dtype": dtype, "shape": shape}
        slots = len(indices[0]) if indices else 0
        if slots:
            ranges = [
                [min(i[s] for i in indices), max(i[s] for i in indices)]
                for s in range(slots)
            ]
            full = 1
            for lo, hi in ranges:
                full *= hi - lo + 1
            if full == len(indices):
                entry["ranges"] = ranges
                out[pattern] = entry
                continue
        entry["indices"] = [list(i) for i in indices] if slots else []
        out[pattern] = entry
    return out


def load_census(path: Path = A3B_CENSUS) -> Census:
    record = json.loads(path.read_text())
    keys = expand(record["patterns"])
    assert len(keys) == record["tensors"], (len(keys), record["tensors"])
    return keys


def census_record(
    files: Iterable[Path], model: str, snapshot: str | None
) -> dict[str, Any]:
    """The census of a cached checkpoint off its shards' headers alone: the
    committed files' shape (``files``, ``model``, ``snapshot``, ``tensors``,
    ``patterns``) plus when it was read. The host is not recorded: the
    headers are the same on every node, and the census is committed."""
    files = tuple(files)
    headers = read_headers(files)
    census: Census = {
        name: (header.dtype, header.shape) for name, header in headers.items()
    }
    return {
        "files": len(files),
        "model": model,
        "snapshot": snapshot,
        "tensors": len(census),
        "patterns": compress(census),
        "captured": {"date": date.today().isoformat()},
    }


# --------------------------------------------------------------------------- #
# a Hub cache of header-only shards
# --------------------------------------------------------------------------- #


def _header_only_shard(
    path: Path, tensors: Mapping[str, tuple[str, tuple[int, ...]]]
) -> None:
    table: dict[str, Any] = {"__metadata__": {"format": "pt"}}
    offset = 0
    for name, (dtype, shape) in tensors.items():
        size = ITEMSIZES[dtype]
        for n in shape:
            size *= n
        table[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + size],
        }
        offset += size
    header = json.dumps(table).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header)


def fake_hub_cache(
    root: Path, key: str, census: Mapping[str, tuple[str, tuple[int, ...]]], files: int
) -> Path:
    """``root`` as a Hub cache holding ``key`` at one snapshot (``main``
    pointing at it) whose ``files`` safetensors shards carry the census's
    tensors, headers only, the keys dealt over the shards in order and the
    index naming each. Returns the snapshot directory. The data section is
    absent: a header read (``read_header``) is all the torch-free estimate
    does, and a load of it would fail at the first tensor, by design."""
    if files < 1:
        raise ValueError(f"a checkpoint has at least one shard, got {files}")
    folder = root / f"models--{key.replace('/', '--')}"
    snapshot = folder / "snapshots" / "fake"
    snapshot.mkdir(parents=True, exist_ok=True)
    (folder / "refs").mkdir(exist_ok=True)
    (folder / "refs" / "main").write_text("fake")
    names = [f"model-{i + 1:05d}-of-{files:05d}.safetensors" for i in range(files)]
    per_shard: list[dict[str, tuple[str, tuple[int, ...]]]] = [{} for _ in names]
    weight_map: dict[str, str] = {}
    for position, (tensor, entry) in enumerate(sorted(census.items())):
        shard = position % files
        per_shard[shard][tensor] = entry
        weight_map[tensor] = names[shard]
    for name, tensors in zip(names, per_shard):
        _header_only_shard(snapshot / name, tensors)
    (snapshot / _INDEX_NAME).write_text(json.dumps({"weight_map": weight_map}))
    return snapshot


# --------------------------------------------------------------------------- #
# capture / check on a node
# --------------------------------------------------------------------------- #


def _differences(committed: Mapping[str, Any], fresh: Mapping[str, Any]) -> list[str]:
    problems = [
        f"{field}: committed {committed.get(field)!r}, the cache has {fresh[field]!r}"
        for field in ("files", "tensors", "model")
        if committed.get(field) != fresh[field]
    ]
    mine, theirs = expand(committed["patterns"]), expand(fresh["patterns"])
    for name in sorted(set(mine) | set(theirs)):
        if mine.get(name) != theirs.get(name):
            problems.append(
                f"{name}: committed {mine.get(name)!r}, cached {theirs.get(name)!r}"
            )
    return problems


def main(argv: list[str] | None = None) -> int:
    """``python -m tests._helpers.header_census KEY [--cache-dir DIR]
    (--out PATH | --check [PATH])``: the census of the cached checkpoint
    ``KEY``, written to ``--out``, or compared to the committed census
    (``--check``, the 70B's by default) — exit 1 naming every key that
    differs, so a derived census is held to the node's headers."""
    from causalab.protocol.checkpoint_census import cached_checkpoint_files

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("key")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--check", nargs="?", const=LLAMA70B_CENSUS, type=Path)
    args = parser.parse_args(argv)
    files = cached_checkpoint_files(args.key, args.revision, cache_dir=args.cache_dir)
    if files is None:
        print(
            f"refused: {args.key}@{args.revision} is not in the Hub cache",
            file=sys.stderr,
        )
        return 2
    snapshot = (
        files[0].parent.name if files[0].parent.parent.name == "snapshots" else None
    )
    fresh = census_record(files, args.key, snapshot)
    if args.check is not None:
        committed = json.loads(Path(args.check).read_text())
        problems = _differences(committed, fresh)
        if committed.get("snapshot") != fresh["snapshot"]:
            print(
                f"note: committed snapshot {committed.get('snapshot')!r}, cached "
                f"{fresh['snapshot']!r}"
            )
        if problems:
            print(f"{args.check} does not match the cached headers:", file=sys.stderr)
            for problem in problems:
                print(f"  {problem}", file=sys.stderr)
            return 1
        print(f"{args.check} matches the cached headers ({fresh['tensors']} tensors)")
        return 0
    if args.out is None:
        print("refused: pass --out PATH or --check", file=sys.stderr)
        return 2
    args.out.write_text(json.dumps(fresh, indent=1, sort_keys=True) + "\n")
    print(f"wrote {args.out} ({fresh['tensors']} tensors over {fresh['files']} shards)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
