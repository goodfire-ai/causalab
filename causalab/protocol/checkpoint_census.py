"""The checkpoint on disk as the protocol layer can read it, torch-free
(``docs/model_parallelism.md`` §2, §5.3, §11): a safetensors file's tensor
table off its header alone, the files a cached checkpoint consists of, and
which of its tensors the text model consumes.

Three facts ``dry-run`` and the memory pre-flight need before any weight is
read, none of which needs torch or transformers:

* [`read_header`][] — the 8-byte length and the JSON table a safetensors
  file opens with (https://github.com/huggingface/safetensors#format), as
  [`TensorHeader`][] entries whose ``itemsize`` comes from the format's
  dtype table. The reference engine's ``checkpoint.py`` re-exports these.
* [`cached_checkpoint_files`][] — the safetensors shards of a checkpoint
  already in the local Hub cache (or a local directory), through
  ``huggingface_hub``'s cache lookup, which never touches the network:
  ``None`` when nothing is cached, so a torch-free caller reports
  *undecided* rather than downloading 70 GB to answer a question about it.
* [`checkpoint_targets`][] — which checkpoint keys land on the text
  model's parameters, and under what name. transformers decides this with
  its conversion mapping (``weights.renamed_keys``); the torch-free rule
  here is structural, on the checkpoint's key tree the way ``family_for``
  is on the module tree: the family's [`TreeAddress`][causalab.protocol.registry.families.TreeAddress] names
  the block list (``model.layers``), the one tower whose block list has
  exactly ``num_layers`` members is the text model, its keys rename onto
  the tree's root (``model.language_model.X`` → ``model.X`` on
  ``Qwen/Qwen3.6-35B-A3B``'s multimodal checkpoint), the head stays where the tree puts it, and every
  other tower — a vision encoder, an MTP head — is left out, as the loader
  leaves them unread.
"""

from __future__ import annotations

import dataclasses
import json
import re
import struct
from pathlib import Path
from typing import Iterable, Mapping

from causalab.protocol.rules.errors import ProtocolError
from causalab.protocol.registry import FAMILIES, TreeAddress

__all__ = [
    "ITEMSIZES",
    "TensorHeader",
    "cached_checkpoint_files",
    "checkpoint_targets",
    "read_header",
    "read_headers",
    "tree_of",
]

#: Bytes per element of every dtype the safetensors format spells
#: (https://github.com/huggingface/safetensors#format, the ``dtype`` field).
ITEMSIZES: Mapping[str, int] = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E5M2": 1,
    "F8_E4M3": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}

#: The index file and the single-file spelling of a safetensors checkpoint
#: (transformers' ``SAFE_WEIGHTS_INDEX_NAME`` / ``SAFE_WEIGHTS_NAME``).
_INDEX_NAME = "model.safetensors.index.json"
_SINGLE_NAME = "model.safetensors"


@dataclasses.dataclass(frozen=True)
class TensorHeader:
    """One tensor's entry in a safetensors header: the strings the file
    carries, so a lazy stand-in can answer ``get_dtype`` / ``get_shape``
    without touching the data section."""

    dtype: str
    shape: tuple[int, ...]

    @property
    def itemsize(self) -> int:
        """Bytes per element, from the format's dtype table — what a byte
        count of a read range is made of. A dtype the format does not spell
        is a checkpoint the reader cannot describe, refused by name."""
        size = ITEMSIZES.get(self.dtype)
        if size is None:
            raise ProtocolError(
                "P2",
                f"safetensors dtype {self.dtype!r} is not one the format spells "
                f"({', '.join(ITEMSIZES)})",
            )
        return size

    @property
    def elements(self) -> int:
        """The tensor's element count (``1`` for a scalar)."""
        count = 1
        for n in self.shape:
            count *= int(n)
        return count


def read_header(path: Path) -> dict[str, TensorHeader]:
    """The tensor table of a safetensors file — a pure header read (8-byte
    little-endian length, then JSON; ``__metadata__`` is not a tensor).

    Format reference: https://github.com/huggingface/safetensors#format.

    Raises:
        ProtocolError: ``P2`` — the file is shorter than its length prefix.
    """
    with open(path, "rb") as fh:
        prefix = fh.read(8)
        if len(prefix) != 8:
            raise ProtocolError(
                "P2", f"{path} is not a safetensors file (truncated header)"
            )
        (length,) = struct.unpack("<Q", prefix)
        table = json.loads(fh.read(length))
    return {
        name: TensorHeader(dtype=str(entry["dtype"]), shape=tuple(entry["shape"]))
        for name, entry in table.items()
        if name != "__metadata__"
    }


def read_headers(files: Iterable[Path]) -> dict[str, TensorHeader]:
    """Every file's table, merged — a checkpoint's tensors are disjoint
    across its shards."""
    merged: dict[str, TensorHeader] = {}
    for path in files:
        merged.update(read_header(path))
    return merged


def cached_checkpoint_files(
    key: str, revision: str = "main", *, cache_dir: Path | None = None
) -> tuple[Path, ...] | None:
    """The safetensors shards of ``key`` **already on this machine**: a
    local directory's files as they are; a Hub id's through
    ``huggingface_hub``'s cache lookup — the index's ``weight_map`` naming
    the shards when there is one, ``model.safetensors`` otherwise — with
    no request made and nothing downloaded. ``None`` when the checkpoint,
    or any shard the index names, is not cached: the torch-free caller
    says so, and the load decides on the node that has the weights.
    ``cache_dir`` is the Hub cache root (the environment's when ``None``).
    """
    local = Path(key)
    if local.is_dir():
        index = local / _INDEX_NAME
        if index.exists():
            return _indexed(index, local)
        single = local / _SINGLE_NAME
        return (single,) if single.exists() else None
    from huggingface_hub import try_to_load_from_cache  # noqa: PLC0415 — lazy, cache only

    root = str(cache_dir) if cache_dir is not None else None
    index_hit = try_to_load_from_cache(
        key, _INDEX_NAME, cache_dir=root, revision=revision
    )
    if isinstance(index_hit, str):
        return _indexed(Path(index_hit), Path(index_hit).parent)
    single_hit = try_to_load_from_cache(
        key, _SINGLE_NAME, cache_dir=root, revision=revision
    )
    return (Path(single_hit),) if isinstance(single_hit, str) else None


def _indexed(index: Path, directory: Path) -> tuple[Path, ...] | None:
    weight_map = json.loads(index.read_text()).get("weight_map", {})
    names = sorted(set(weight_map.values()))
    files = tuple(directory / name for name in names)
    if not files or not all(path.exists() for path in files):
        return None
    return files


# --------------------------------------------------------------------------- #
# which keys the text model consumes — the tower rule
# --------------------------------------------------------------------------- #


def _towers(keys: Iterable[str], leaf: str) -> dict[str, set[int]]:
    """Per root, the block indices under ``<root>.<leaf>.<n>.`` in ``keys``."""
    towers: dict[str, set[int]] = {}
    pattern = re.compile(rf"^(?P<root>.+?)\.{re.escape(leaf)}\.(?P<n>\d+)\.")
    for key in keys:
        match = pattern.match(key)
        if match is not None:
            towers.setdefault(match["root"], set()).add(int(match["n"]))
    return towers


def _text_tower(keys: Iterable[str], tree: TreeAddress, num_layers: int) -> str | None:
    """The one root whose block list under ``tree.blocks``'s leaf has exactly
    ``range(num_layers)``; ``None`` for none or several."""
    leaf = tree.blocks.rsplit(".", 1)[-1]
    towers = _towers(keys, leaf)
    roots = [
        root for root, indices in towers.items() if indices == set(range(num_layers))
    ]
    return roots[0] if len(roots) == 1 else None


def tree_of(keys: Iterable[str], num_layers: int) -> TreeAddress | None:
    """The registered family tree whose block list the checkpoint's keys
    carry with exactly ``num_layers`` blocks — structural detection on the
    key tree, the way ``registry.family_for`` detects on the module tree;
    ``None`` when no tree matches, or more than one does.

    The answer is a tree *address*, and two families may share one: GPT-2
    and GPT-J both address ``transformer.h`` / ``transformer.wte`` /
    ``transformer.ln_f`` and differ only inside the block. The same address
    from two families is one match, not an ambiguity."""
    keys = tuple(keys)
    hits = {
        adapter.tree
        for adapter in FAMILIES.values()
        if _text_tower(keys, adapter.tree, num_layers) is not None
    }
    if len(hits) != 1:
        return None
    return next(iter(hits))


def checkpoint_targets(
    keys: Iterable[str], tree: TreeAddress, num_layers: int
) -> dict[str, str] | None:
    """The checkpoint keys the text model consumes, each with the parameter
    path it lands on (module docstring, the tower rule): the keys under the
    text tower's root renamed onto the tree's root, the head's keys as they
    are, every other tower's keys left out. ``None`` when no tower of
    ``num_layers`` blocks is found, or several are — the caller reports the
    estimate undecided rather than counting the wrong tensors.
    """
    keys = tuple(keys)
    root = _text_tower(keys, tree, num_layers)
    if root is None:
        return None
    tree_root = tree.blocks.rsplit(".", 1)[0]
    out: dict[str, str] = {}
    for key in keys:
        if key == root or key.startswith(root + "."):
            out[key] = tree_root + key[len(root) :]
        elif key == tree.lm_head or key.startswith(tree.lm_head + "."):
            out[key] = key
    return out
