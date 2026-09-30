"""Read and write the repository's safetensors files.

The public file operations use ``causalab.io.fastersafetensors``. ``TensorBundle``
and ``BundlePoint`` address entries in fitted artifacts. ``load_tensors`` and
``load_table`` provide the corresponding readers for engine inputs."""

from __future__ import annotations

import dataclasses
import functools
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from causalab.protocol.rules.errors import ProtocolError

if TYPE_CHECKING:
    from causalab.protocol.engine import RunContext

try:
    from causalab.io.fastersafetensors.torch import load_file, safe_open, save_file
except ImportError as exc:
    # seven modules reach this import; a missing extension should name its cause
    # here rather than as a bare `cannot import name '_core'` from whichever
    # script touched a tensor first
    raise ImportError(
        f"causalab.io.fastersafetensors could not be imported ({exc}). If the "
        "Rust extension (_core) is not built for this interpreter or platform, "
        "run `uv sync` with a Rust toolchain on PATH "
        "(docs/fastersafetensors.md)"
    ) from exc

__all__ = [
    "BundlePoint",
    "TensorBundle",
    "load_file",
    "load_table",
    "load_tensors",
    "safe_open",
    "save_file",
]


@dataclasses.dataclass(frozen=True)
class BundlePoint:
    """One producing point's slice of a bundle: the tensors sharing a
    coordinate suffix, plus that entry's stamped record.

    Slicing by suffix rather than selecting each slot on its own is what
    keeps a multi-slot bundle coherent — an SAE's ``enc`` and ``dec`` must
    come from the same fit, not from whichever entries each lookup found.
    """

    tensors: dict[str, torch.Tensor]
    suffix: str
    record: dict[str, Any]
    what: str
    #: the entry's ArtifactIdentity (§8): the file-level stamp, overridden by
    #: whatever the ``entries`` record says for this entry
    #: ([`causalab.io.env.entry_identity`][])
    identity: dict[str, Any] = dataclasses.field(default_factory=dict)

    def tensor(self, slot: str) -> torch.Tensor:
        key = f"{slot}{self.suffix}"
        if key not in self.tensors:
            raise ProtocolError(
                "P2",
                f"{self.what}: the bundle has no {key!r} — an entry's slots "
                f"must be complete (has {sorted(self.tensors)})",
            )
        return self.tensors[key]


@dataclasses.dataclass(frozen=True)
class TensorBundle:
    """One loaded ``.safetensors`` file: its tensors plus the ``entries``
    table from the header (§8, `causalab.protocol.bundles`).

    [`point`][] is the only way in. A bundle written by a swept document
    holds one entry per point per slot, so asking for a bare slot name would
    either ``KeyError`` or — worse — silently take whichever entry a plain
    dict lookup happened to find.
    """

    tensors: dict[str, torch.Tensor]
    entry_coords: dict[str, Any]
    #: the header's ``__metadata__`` table as written — the file-level
    #: ArtifactIdentity plus the serialized ``entries``; a hand-built bundle
    #: carries none
    header: dict[str, Any] = dataclasses.field(default_factory=dict)

    def point(
        self,
        slot: str,
        want: Any,
        *,
        what: str,
        implicit: bool = False,
    ) -> BundlePoint:
        """The entry for ``slot`` selected by ``want`` (a coordinate
        mapping; ``implicit`` when derived from the consuming point rather
        than authored), as a coherent slice of the bundle."""
        from causalab.protocol.bundles import select_entry
        from causalab.io.env import entry_identity

        key = select_entry(
            self.tensors.keys(),
            slot,
            want,
            what=what,
            coords_by_key=self.entry_coords or None,
            implicit=implicit,
        )
        record = self.entry_coords.get(key, {})
        record = record if isinstance(record, dict) else {}
        return BundlePoint(
            tensors=self.tensors,
            suffix=key[len(slot) :],
            record=record,
            what=what,
            identity=entry_identity(self.header, key),
        )


@functools.lru_cache(maxsize=32)
def _read_bundle(path: str, _stamp: tuple[int, int]) -> TensorBundle:
    """One bundle, read once. The cache matters: a write operand resolves
    its ``params`` tensor on every application, so an uncached read would
    re-open the same file for every batch of every point.

    ``_stamp`` is the file's (mtime, size), so a path rewritten in the same
    process — a step re-run into an existing run tree — is a cache miss
    rather than a stale tensor."""
    from causalab.io.env import read_safetensors_metadata

    meta = read_safetensors_metadata(Path(path)) or {}
    raw_entries = meta.get("entries")
    entry_coords: dict[str, Any] = {}
    if isinstance(raw_entries, str):
        try:
            decoded = json.loads(raw_entries)
        except json.JSONDecodeError as err:
            raise ProtocolError(
                "P2", f"{path}: unreadable 'entries' table in the header — {err}"
            ) from err
        if isinstance(decoded, dict):
            entry_coords = decoded
    return TensorBundle(
        tensors=load_file(path), entry_coords=entry_coords, header=dict(meta)
    )


def load_table(run: RunContext, file_path: str) -> tuple[list[dict[str, Any]], bytes]:
    """A saved metric table referenced by a gate's ``init.from_scores``
    (§2.5), resolved through the artifact store exactly as [`load_tensors`][]
    resolves a bundle, as ``(rows, bytes)`` — the rows to read the start off,
    the bytes to stamp its digest with."""
    from causalab.io.tables import read_table

    artifacts = run.env.artifacts
    resolve = getattr(artifacts, "resolve_path", None)
    if resolve is not None:
        target = Path(resolve(file_path))
    else:
        root = getattr(artifacts, "root", None)
        if root is None:
            raise ProtocolError("P2", "artifact store exposes no filesystem root")
        target = Path(root) / file_path
    return read_table(target), target.read_bytes()


def load_tensors(run: RunContext, file_path: str) -> TensorBundle:
    """Load a tensor bundle referenced by a featurizer/params file_path,
    resolved through the artifact store (which owns the run-tree/external
    overlay inside a workflow)."""
    artifacts = run.env.artifacts
    resolve = getattr(artifacts, "resolve_path", None)
    if resolve is not None:
        target = Path(resolve(file_path))
    else:
        root = getattr(artifacts, "root", None)
        if root is None:
            raise ProtocolError("P2", "artifact store exposes no filesystem root")
        target = Path(root) / file_path
    stat = target.stat()
    return _read_bundle(str(target), (stat.st_mtime_ns, stat.st_size))
