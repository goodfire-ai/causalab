"""Value types shared by executors and results.

``RaggedValue`` holds a gather whose rows address different numbers of
positions. Results code imports it without importing the executor package.
"""

from __future__ import annotations

import dataclasses

import torch

__all__ = ["RaggedValue"]


@dataclasses.dataclass(frozen=True)
class RaggedValue:
    """A read over per-row windows of unequal width: the flat
    ``(total_positions, d)`` gather plus per-row widths, re-nestable via
    ``torch.split(flat, widths)`` — the RaggedIndex contract of the old
    resolver, kept as the protocol's ragged-read surface."""

    flat: torch.Tensor
    widths: tuple[int, ...]

    def detach_cpu(self) -> "RaggedValue":
        return RaggedValue(flat=self.flat.detach().cpu(), widths=self.widths)
