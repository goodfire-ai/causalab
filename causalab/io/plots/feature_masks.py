"""
Visualization functions for feature count heatmaps (DBM with tie_masks=False).

These visualizations show feature counts (number of selected features) for
mask-based interventions where each grid cell can select a subset of features.

Cells arrive as structured [`GridCell`][]
records — component/layer/head/position joined from the grid's own specs by
`cells_from_site_grid`. The
legacy path parsed these coordinates out of unit-id strings
(`causalab.io.plots.unit_id`, retired); post-migration the per-key dicts
are keyed by opaque ``spec.key`` strings that nothing may parse.

Component types:
- Attention heads: (layer, head) grid
- Residual stream: (layer, token_position) grid
- MLPs: (layer, token_position) grid
"""

from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from .grid_cells import GridCell, cell_grid_dimensions
from .utils import create_feature_count_heatmap


# =============================================================================
# Helper Functions
# =============================================================================


def _cell_count(cell: GridCell) -> int:
    if cell.indices is not None:
        return len(cell.indices)
    if cell.n_features is None:
        raise ValueError(
            f"cell {cell.key!r} selects all features (indices=None) but carries "
            "no n_features; pass n_features to the plot call (int or per-key "
            "mapping) so the full-selection count is defined."
        )
    return cell.n_features


# =============================================================================
# Matrix Building Functions
# =============================================================================


def _build_attention_head_matrix(
    cells: Sequence[GridCell],
    layers: List[int],
    heads: List[int],
) -> np.ndarray[Any, np.dtype[Any]]:
    """Build feature count matrix for attention heads."""
    count_matrix = np.zeros((len(layers), len(heads)))
    for cell in cells:
        if cell.head is None:
            continue
        if cell.layer in layers and cell.head in heads:
            layer_idx = layers.index(cell.layer)
            head_idx = heads.index(cell.head)
            count_matrix[layer_idx, head_idx] = _cell_count(cell)
    return count_matrix


def _build_position_based_matrix(
    cells: Sequence[GridCell],
    layers: List[int],
    token_position_ids: List[str],
) -> np.ndarray[Any, np.dtype[Any]]:
    """Build feature count matrix for residual stream or MLP."""
    count_matrix = np.zeros((len(layers), len(token_position_ids)))
    for cell in cells:
        if cell.position is None:
            continue
        if cell.layer in layers and cell.position in token_position_ids:
            layer_idx = layers.index(cell.layer)
            pos_idx = token_position_ids.index(cell.position)
            count_matrix[layer_idx, pos_idx] = _cell_count(cell)
    return count_matrix


# =============================================================================
# Public Plotting Functions
# =============================================================================


def plot_attention_head_feature_counts(
    cells: Sequence[GridCell],
    scores: Union[float, Dict[int, float]],
    layers: List[int],
    heads: List[int],
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    figure_format: str = "png",
) -> None:
    """
    Plot a full grid heatmap of feature counts for attention heads.

    Supports both single score mode (one accuracy) and per-layer score mode.

    Args:
        cells: Structured grid cells (head cells; others are ignored).
        scores: Either a float (single overall accuracy) or
            Dict[int, float] (per-layer accuracies).
        layers: List of layer indices (y-axis).
        heads: List of head indices (x-axis).
        title: Optional custom title.
        save_path: Optional path to save figure.
        figure_format: ``png`` or ``pdf`` for static output.
    """
    count_matrix = _build_attention_head_matrix(cells, layers, heads)

    x_labels = [f"H{h}" for h in heads]
    y_labels = [f"L{layer}" for layer in layers]

    has_per_layer_scores = isinstance(scores, dict)
    show_accuracy_column = has_per_layer_scores

    create_feature_count_heatmap(
        count_matrix=count_matrix,
        x_labels=x_labels,
        y_labels=y_labels,
        scores=scores,
        layers=layers,
        title=title or "Attention Heads: Features Selected",
        xlabel="Head",
        ylabel="Layer",
        colorbar_label="Feature Count",
        save_path=save_path,
        flip_vertical=True,  # Lowest layer at bottom, highest at top
        figsize=(max(12, len(heads) * 0.5 + 2), max(6, len(layers) * 0.4)),
        show_accuracy_column=show_accuracy_column,
        figure_format=figure_format,
    )


def plot_residual_stream_feature_counts(
    cells: Sequence[GridCell],
    scores: Union[float, Dict[int, float]],
    layers: List[int],
    token_position_ids: List[str],
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    score_label: str = "Acc",
    figure_format: str = "png",
) -> None:
    """
    Plot a tokens x layers heatmap for residual stream with accuracy column.

    Supports both single score mode and per-layer score mode.

    Args:
        cells: Structured grid cells (position cells; others are ignored).
        scores: Either a float (single overall accuracy) or
            Dict[int, float] (per-layer accuracies).
        layers: List of layer indices (will be displayed bottom-to-top).
        token_position_ids: List of token position IDs (x-axis).
        title: Optional custom title.
        save_path: Optional path to save figure.
        score_label: Label for the accuracy column (default: "Acc").
        figure_format: ``png`` or ``pdf`` for static output.
    """
    count_matrix = _build_position_based_matrix(cells, layers, token_position_ids)

    y_labels = [f"L{layer}" if layer >= 0 else "Emb" for layer in layers]

    create_feature_count_heatmap(
        count_matrix=count_matrix,
        x_labels=token_position_ids,
        y_labels=y_labels,
        scores=scores,
        layers=layers,
        title=title or "Residual Stream: Features Selected",
        xlabel="Token Position",
        ylabel="Layer",
        score_label=score_label,
        colorbar_label="Features Selected",
        save_path=save_path,
        flip_vertical=True,
        figsize=(max(10, len(token_position_ids) * 0.8 + 2), max(6, len(layers) * 0.4)),
        show_accuracy_column=True,
        figure_format=figure_format,
    )


def plot_mlp_feature_counts(
    cells: Sequence[GridCell],
    scores: Union[float, Dict[int, float]],
    layers: List[int],
    token_position_ids: List[str],
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    figure_format: str = "png",
) -> None:
    """
    Plot a tokens x layers heatmap of feature counts for MLPs with accuracy column.

    Supports both single score mode and per-layer score mode.

    Args:
        cells: Structured grid cells (position cells; others are ignored).
        scores: Either a float (single overall accuracy) or
            Dict[int, float] (per-layer accuracies).
        layers: List of layer indices.
        token_position_ids: List of token position IDs (x-axis).
        title: Optional custom title.
        save_path: Optional path to save figure.
        figure_format: ``png`` or ``pdf`` for static output.
    """
    count_matrix = _build_position_based_matrix(cells, layers, token_position_ids)

    y_labels = [f"L{layer}" for layer in layers]

    create_feature_count_heatmap(
        count_matrix=count_matrix,
        x_labels=token_position_ids,
        y_labels=y_labels,
        scores=scores,
        layers=layers,
        title=title or "MLPs: Features Selected",
        xlabel="Token Position",
        ylabel="Layer",
        colorbar_label="Features Selected",
        save_path=save_path,
        flip_vertical=True,
        figsize=(max(10, len(token_position_ids) * 0.8 + 2), max(6, len(layers) * 0.4)),
        show_accuracy_column=True,
        figure_format=figure_format,
    )


# =============================================================================
# Unified Dispatcher
# =============================================================================


def plot_feature_counts(
    component_type: str,
    cells: Sequence[GridCell],
    scores: Union[float, Dict[int, float]],
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    figure_format: str = "png",
) -> None:
    """
    Plot feature counts for structured grid cells.

    Grid dimensions come from the cells' structural fields (via
    [`cell_grid_dimensions`][]); the
    ``component_type`` is the caller's structural detection — typically the
    first element of
    `cells_from_site_grid`'s return.

    Args:
        component_type: One of ``"attention_head"``, ``"residual_stream"``,
            ``"mlp"``.
        cells: Structured grid cells with per-cell indices and n_features.
        scores: Either a float (single overall accuracy) or
            Dict[int, float] (per-layer accuracies).
        title: Optional custom title.
        save_path: Optional path to save figure.
        figure_format: ``png`` or ``pdf`` for static output.
    """
    dims = cell_grid_dimensions(component_type, cells)

    if component_type == "attention_head":
        plot_attention_head_feature_counts(
            cells=cells,
            scores=scores,
            layers=dims["layers"],
            heads=dims["heads"],
            title=title,
            save_path=save_path,
            figure_format=figure_format,
        )
    elif component_type == "residual_stream":
        plot_residual_stream_feature_counts(
            cells=cells,
            scores=scores,
            layers=dims["layers"],
            token_position_ids=dims["token_position_ids"],
            title=title,
            save_path=save_path,
            figure_format=figure_format,
        )
    elif component_type == "mlp":
        plot_mlp_feature_counts(
            cells=cells,
            scores=scores,
            layers=dims["layers"],
            token_position_ids=dims["token_position_ids"],
            title=title,
            save_path=save_path,
            figure_format=figure_format,
        )
    else:
        raise ValueError(f"Unknown component type: {component_type}")
