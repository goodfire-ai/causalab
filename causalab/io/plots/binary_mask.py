"""
Visualization functions for binary mask heatmaps (DBM with tie_masks=True).

These visualizations show which grid cells (attention heads, residual stream
positions, or MLPs) were selected by DBM training. Selected cells have mask=1
(indices=None), unselected cells have mask=0 (indices=[]).

Cells arrive as structured [`GridCell`][]
records — component/layer/head/position joined from the grid's own specs by
`cells_from_site_grid`. The
legacy path parsed these coordinates out of unit-id strings
(`causalab.io.plots.unit_id`, retired); post-migration the per-key dicts
are keyed by opaque ``spec.key`` strings that nothing may parse.

This module consolidates all binary mask plotting for different component types:
- Attention heads: (layer, head) grid
- Residual stream: (layer, token_position) grid
- MLPs: (layer, token_position) grid
"""

from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from .grid_cells import GridCell, cell_grid_dimensions
from .utils import create_binary_mask_heatmap


# =============================================================================
# Selection Extractors
# =============================================================================


def get_selected_heads(cells: Sequence[GridCell]) -> List[Tuple[int, int]]:
    """
    Extract list of (layer, head) pairs that were selected by DBM.

    Args:
        cells: Structured grid cells (indices=None means selected).

    Returns:
        List of (layer, head) tuples for selected heads, sorted by layer then head.
    """
    selected = [
        (cell.layer, cell.head)
        for cell in cells
        if cell.head is not None and cell.indices is None
    ]
    selected.sort(key=lambda x: (x[0], x[1]))
    return selected


def get_selected_residual_positions(
    cells: Sequence[GridCell],
) -> List[Tuple[int, str]]:
    """
    Extract list of (layer, token_position_id) pairs that were selected by DBM.

    Args:
        cells: Structured grid cells (indices=None means selected).

    Returns:
        List of (layer, token_position_id) tuples for selected positions,
        sorted by layer then position.
    """
    selected = [
        (cell.layer, cell.position)
        for cell in cells
        if cell.position is not None and cell.head is None and cell.indices is None
    ]
    selected.sort(key=lambda x: (x[0], x[1]))
    return selected


def get_selected_mlps(cells: Sequence[GridCell]) -> List[Tuple[int, str]]:
    """
    Extract list of (layer, token_position_id) pairs for selected MLPs.

    Args:
        cells: Structured grid cells (indices=None means selected).

    Returns:
        List of (layer, token_position_id) tuples for selected MLPs,
        sorted by layer then position.
    """
    return get_selected_residual_positions(cells)


def get_selected_units(
    component_type: str, cells: Sequence[GridCell]
) -> List[Tuple[Any, ...]]:
    """
    Extract list of selected cells for a known component type.

    Args:
        component_type: One of ``"attention_head"``, ``"residual_stream"``,
            ``"mlp"`` (the caller's structural detection).
        cells: Structured grid cells (indices=None means selected).

    Returns:
        List of tuples for selected cells:
        - attention_head: List[Tuple[int, int]] - (layer, head)
        - residual_stream / mlp: List[Tuple[int, str]] - (layer, token_position_id)
    """
    if component_type == "attention_head":
        return get_selected_heads(cells)
    elif component_type in ("residual_stream", "mlp"):
        return get_selected_residual_positions(cells)
    else:
        raise ValueError(f"Unknown component type: {component_type}")


# =============================================================================
# Plotting Functions
# =============================================================================


def _head_mask_matrix(
    cells: Sequence[GridCell], layers: List[int], heads: List[int]
) -> np.ndarray[Any, np.dtype[Any]]:
    """(layers, heads) binary matrix; NaN where the grid has no cell."""
    mask_matrix = np.full((len(layers), len(heads)), np.nan)
    for cell in cells:
        if cell.head is None:
            continue
        if cell.layer in layers and cell.head in heads:
            layer_idx = layers.index(cell.layer)
            head_idx = heads.index(cell.head)
            # None means all features selected (mask=1); a list means a
            # (possibly empty-by-omission) subset (mask=0).
            mask_matrix[layer_idx, head_idx] = 1 if cell.indices is None else 0
    return mask_matrix


def _position_mask_matrix(
    cells: Sequence[GridCell], layers: List[int], token_position_ids: List[str]
) -> np.ndarray[Any, np.dtype[Any]]:
    """(layers, positions) binary matrix; NaN where the grid has no cell."""
    mask_matrix = np.full((len(layers), len(token_position_ids)), np.nan)
    for cell in cells:
        if cell.position is None:
            continue
        if cell.layer in layers and cell.position in token_position_ids:
            layer_idx = layers.index(cell.layer)
            pos_idx = token_position_ids.index(cell.position)
            mask_matrix[layer_idx, pos_idx] = 1 if cell.indices is None else 0
    return mask_matrix


def plot_attention_head_mask(
    cells: Sequence[GridCell],
    layers: List[int],
    heads: List[int],
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    figure_format: str = "png",
) -> None:
    """
    Plot binary mask showing which attention heads were selected by DBM.

    Args:
        cells: Structured grid cells. indices=None = selected, list = not selected.
        layers: List of layer indices (y-axis).
        heads: List of head indices (x-axis).
        title: Optional custom title.
        save_path: Optional path to save figure.
        figure_format: ``png`` or ``pdf`` for static output.
    """
    mask_matrix = _head_mask_matrix(cells, layers, heads)

    x_labels = [f"H{head}" for head in heads]
    y_labels = [f"L{layer}" for layer in layers]

    if title is None:
        num_selected = int(np.nansum(mask_matrix))
        num_total = int(np.sum(~np.isnan(mask_matrix)))
        title = f"DBM Attention Head Mask ({num_selected}/{num_total} heads selected)"

    create_binary_mask_heatmap(
        mask_matrix=mask_matrix,
        x_labels=x_labels,
        y_labels=y_labels,
        title=title,
        save_path=save_path,
        xlabel="Head",
        ylabel="Layer",
        figsize=(max(12, len(heads) * 0.6), max(6, len(layers) * 0.8)),
        figure_format=figure_format,
    )


def plot_residual_stream_mask(
    cells: Sequence[GridCell],
    layers: List[int],
    token_position_ids: List[str],
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    figure_format: str = "png",
) -> None:
    """
    Plot binary mask showing which residual stream positions were selected by DBM.

    Args:
        cells: Structured grid cells. indices=None = selected, list = not selected.
        layers: List of layer indices (y-axis).
        token_position_ids: List of token position IDs (x-axis).
        title: Optional custom title.
        save_path: Optional path to save figure.
        figure_format: ``png`` or ``pdf`` for static output.
    """
    mask_matrix = _position_mask_matrix(cells, layers, token_position_ids)

    x_labels = token_position_ids
    y_labels = [f"L{layer}" if layer >= 0 else "Emb" for layer in layers]

    if title is None:
        num_selected = int(np.nansum(mask_matrix))
        num_total = int(np.sum(~np.isnan(mask_matrix)))
        title = (
            f"DBM Residual Stream Mask ({num_selected}/{num_total} positions selected)"
        )

    create_binary_mask_heatmap(
        mask_matrix=mask_matrix,
        x_labels=x_labels,
        y_labels=y_labels,
        title=title,
        save_path=save_path,
        xlabel="Token Position",
        ylabel="Layer",
        figsize=(max(10, len(token_position_ids) * 0.8), max(6, len(layers) * 0.4)),
        figure_format=figure_format,
    )


def plot_mlp_mask(
    cells: Sequence[GridCell],
    layers: List[int],
    token_position_ids: List[str],
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    figure_format: str = "png",
) -> None:
    """
    Plot binary mask showing which MLP units were selected by DBM.

    Args:
        cells: Structured grid cells. indices=None = selected, list = not selected.
        layers: List of layer indices (y-axis).
        token_position_ids: List of token position IDs (x-axis).
        title: Optional custom title.
        save_path: Optional path to save figure.
        figure_format: ``png`` or ``pdf`` for static output.
    """
    mask_matrix = _position_mask_matrix(cells, layers, token_position_ids)

    x_labels = token_position_ids
    y_labels = [f"L{layer}" for layer in layers]

    if title is None:
        num_selected = int(np.nansum(mask_matrix))
        num_total = int(np.sum(~np.isnan(mask_matrix)))
        title = f"DBM MLP Mask ({num_selected}/{num_total} units selected)"

    create_binary_mask_heatmap(
        mask_matrix=mask_matrix,
        x_labels=x_labels,
        y_labels=y_labels,
        title=title,
        save_path=save_path,
        xlabel="Token Position",
        ylabel="Layer",
        figsize=(max(10, len(token_position_ids) * 0.8), max(6, len(layers) * 0.4)),
        figure_format=figure_format,
    )


def plot_binary_mask(
    component_type: str,
    cells: Sequence[GridCell],
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    figure_format: str = "png",
) -> None:
    """
    Plot binary mask for structured grid cells.

    Grid dimensions come from the cells' structural fields (via
    [`cell_grid_dimensions`][]); the
    ``component_type`` is the caller's structural detection — typically the
    first element of
    `cells_from_site_grid`'s return.

    Args:
        component_type: One of ``"attention_head"``, ``"residual_stream"``,
            ``"mlp"``.
        cells: Structured grid cells. indices=None = selected, list = not selected.
        title: Optional custom title.
        save_path: Optional path to save figure.
        figure_format: ``png`` or ``pdf`` for static output.
    """
    dims = cell_grid_dimensions(component_type, cells)

    if component_type == "attention_head":
        plot_attention_head_mask(
            cells=cells,
            layers=dims["layers"],
            heads=dims["heads"],
            title=title,
            save_path=save_path,
            figure_format=figure_format,
        )
    elif component_type == "residual_stream":
        plot_residual_stream_mask(
            cells=cells,
            layers=dims["layers"],
            token_position_ids=dims["token_position_ids"],
            title=title,
            save_path=save_path,
            figure_format=figure_format,
        )
    elif component_type == "mlp":
        plot_mlp_mask(
            cells=cells,
            layers=dims["layers"],
            token_position_ids=dims["token_position_ids"],
            title=title,
            save_path=save_path,
            figure_format=figure_format,
        )
    else:
        raise ValueError(f"Unknown component type: {component_type}")
