"""The checkpoint on disk: which safetensors files a model key resolves to,
and each file's tensor table read off its header alone.

Split out of ``weights.py`` so the shard-on-read planner (``shard_read.py``)
and the readers share one description of a tensor on disk without importing
each other. The header read itself — [`TensorHeader`][],
[`read_header`][] — is arithmetic on the file format and lives torch-free
in [`causalab.protocol.checkpoint_census`][], where ``dry-run`` and the
memory pre-flight read it too; this module re-exports it and adds the one
resolver that needs transformers.
"""

from __future__ import annotations

from pathlib import Path

from causalab.protocol.checkpoint_census import TensorHeader, read_header

__all__ = ["TensorHeader", "checkpoint_files", "read_header"]


def checkpoint_files(key: str, revision: str) -> tuple[Path, ...] | None:
    """The safetensors shards of a checkpoint, resolved through transformers'
    own file resolver (``cached_file`` / ``get_checkpoint_shard_files``): a
    local directory as is, a Hub id through the cache (``HF_HUB_OFFLINE``
    honoured), the index's ``weight_map`` naming the shards when there is one,
    ``model.safetensors`` otherwise. The same route the stock loader takes —
    ``huggingface_hub.snapshot_download`` is not it: for Xet-backed repos it
    asks the Hub for a read token that anonymous callers are refused (CI has
    no token), while transformers' route downloads them.

    ``None`` when the checkpoint ships no safetensors — the caller then takes
    the stock loader, which knows the other formats.
    """
    from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME
    from transformers.utils.hub import cached_file, get_checkpoint_shard_files

    index = cached_file(
        key,
        SAFE_WEIGHTS_INDEX_NAME,
        revision=revision,
        _raise_exceptions_for_missing_entries=False,
    )
    if index is not None:
        shards, _metadata = get_checkpoint_shard_files(key, index, revision=revision)
        # transformers types the list as optional; an index that names no shard
        # is not a safetensors checkpoint the fast path can read
        return tuple(Path(shard) for shard in shards) if shards else None
    single = cached_file(
        key,
        SAFE_WEIGHTS_NAME,
        revision=revision,
        _raise_exceptions_for_missing_entries=False,
    )
    return (Path(single),) if single is not None else None
