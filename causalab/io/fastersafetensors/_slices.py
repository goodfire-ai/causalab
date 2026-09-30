"""Lazy tensor slices: read only the bytes an index touches.

Mirrors the reference ``PySafeSlice``: ``get_shape()``, ``get_dtype()`` and
``__getitem__`` over ints, slices (positive step), ``Ellipsis`` and ``None``
(a new axis), with the reference's error messages. The index becomes a
`Selection` — a box the core resolves to
coalesced reads, so an inner-dimension slice reads its runs and not the
covering block — and the reader hands back a tensor of the result's shape.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias

import torch

from ._header import TensorEntry
from ._select import Selection, select

ReadSelection: TypeAlias = Callable[[TensorEntry, Selection], torch.Tensor]
"""``(entry, selection) -> tensor`` of the selection's result shape and the
entry's dtype, on the slice's device when the reader can land there (host,
CUDA) or on the host otherwise; the slice moves the result on afterwards."""


class SafeSlice:
    """A tensor in an open file, indexed lazily."""

    __slots__ = ("_device", "_entry", "_read")

    def __init__(
        self, entry: TensorEntry, read: ReadSelection, device: torch.device
    ) -> None:
        self._entry = entry
        self._read = read
        self._device = device

    def get_shape(self) -> list[int]:
        return list(self._entry.shape)

    def get_dtype(self) -> str:
        return self._entry.dtype

    def __getitem__(self, index: object) -> torch.Tensor:
        entry = self._entry
        selection = select(entry.shape, index)
        return self._read(entry, selection).to(self._device)
