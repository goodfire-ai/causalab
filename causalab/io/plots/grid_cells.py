"""Structured grid-cell records for mask/feature-count plots.

Plot axes are joined **structurally** from [`GridCell`][] records —
never parsed out of result-key strings. The Plan-era builders that
recovered cells from a built ``SiteGrid``'s specs left with that stack;
a protocol-era consumer builds cells directly from its result tables'
sweep-coordinate columns.

Layer semantics match the legacy id-parsing exactly: a residual
``layer=-1`` grid cell was built at layer 0 and labelled ``L0``.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Optional, Sequence

__all__ = [
    "GridCell",
    "cell_grid_dimensions",
]


@dataclasses.dataclass(frozen=True)
class GridCell:
    """One plotted grid cell: a spec's structural coordinates plus its
    selection state.

    ``indices`` follows the trained-mask convention: ``None`` means every
    feature is selected (binary mask = on), a list is the selected feature
    subset (binary mask = off when empty is not constructible; mask plots
    treat any non-``None`` list as off, feature-count plots count it).
    """

    key: str
    layer: int
    head: Optional[int] = None
    position: Optional[str] = None
    indices: Optional[tuple[int, ...]] = None
    n_features: Optional[int] = None


def cell_grid_dimensions(
    component_type: str, cells: Sequence[GridCell]
) -> dict[str, list[Any]]:
    """Grid axes from structured cells — the ``extract_grid_dimensions``
    successor (id parsing retired).

    Returns ``{"layers", "heads"}`` for ``attention_head`` grids (both
    sorted) and ``{"layers", "token_position_ids"}`` otherwise (layers
    sorted, position ids in first-seen order — the axis-order contract
    ``score_heatmap`` shares).
    """
    if component_type == "attention_head":
        layers = sorted({c.layer for c in cells})
        heads = sorted({c.head for c in cells if c.head is not None})
        return {"layers": layers, "heads": heads}
    layers = sorted({c.layer for c in cells})
    position_ids: list[str] = []
    seen: set[str] = set()
    for cell in cells:
        if cell.position is not None and cell.position not in seen:
            position_ids.append(cell.position)
            seen.add(cell.position)
    return {"layers": layers, "token_position_ids": position_ids}
