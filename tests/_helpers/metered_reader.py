"""A meter over a checkpoint reader (``weights.CheckpointReader``): what the
reader was asked for, and how many bytes of tensors handed out in a dtype
other than the load's — the **converting** ones, ``docs/model_parallelism.md``
§5.3 "the load's peak under a dtype conversion" — are alive at once.

The residency rule reads the CUDA allocator's counters; off CUDA there are
none, so this is the CPU tier's meter for the same fact. Every tensor the
wrapped reader returns in another dtype than ``target`` is weighed and
followed by a ``weakref.finalize``: the bytes join ``live`` on hand-out and
leave it when the tensor dies — right after ``Prefetch`` casts it on the
host and drops the on-disk copy — so ``peak`` is the most on-disk bytes the
load ever held beyond the resident weights. A loader that landed the whole
plan before casting would peak at the checkpoint's bytes; one that stages
batches peaks at a batch (the largest converting tensor). CPython frees a
tensor the moment its last reference goes, which is what makes the count
deterministic on the CPU.
"""

from __future__ import annotations

import dataclasses
import threading
import weakref
from pathlib import Path
from typing import Sequence

import torch

from causalab.neural.engines.pytorch_hooks.weights import CheckpointReader, Shard

__all__ = ["MeteredReader"]


@dataclasses.dataclass
class MeteredReader:
    """The wrapped reader's ``read_all``, metered (module docstring)."""

    inner: CheckpointReader
    #: The dtype the load holds float tensors in; a tensor handed out in
    #: another is a converting one and is weighed.
    target: torch.dtype
    #: Every call: the device asked for and, per shard, its path and keys.
    calls: list[tuple[torch.device, tuple[tuple[Path, tuple[str, ...]], ...]]] = (
        dataclasses.field(default_factory=list)
    )
    #: Converting bytes handed out and still alive / at most alive at once /
    #: handed out in all.
    live: int = 0
    peak: int = 0
    handed: int = 0
    _lock: threading.Lock = dataclasses.field(default_factory=threading.Lock)

    def read_all(
        self, shards: Sequence[Shard], device: torch.device
    ) -> dict[str, torch.Tensor]:
        out = self.inner.read_all(shards, device)
        with self._lock:
            self.calls.append((device, tuple((s.path, s.keys) for s in shards)))
            for tensor in out.values():
                if tensor.dtype == self.target:
                    continue
                nbytes = tensor.numel() * tensor.element_size()
                self.live += nbytes
                self.handed += nbytes
                self.peak = max(self.peak, self.live)
                weakref.finalize(tensor, self._release, nbytes)
        return out

    def _release(self, nbytes: int) -> None:
        with self._lock:
            self.live -= nbytes
