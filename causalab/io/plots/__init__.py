"""Load plotting helpers on first access through PEP 562.

The workflow runner can locate and hash a plotting script while importing only
its parent package. Deferring numerical imports keeps this operation available
during document validation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

#: Public name -> the submodule that defines it. The one place to edit when a
#: plot helper moves; ``__all__`` is derived from it so the two cannot drift.
_EXPORTS: dict[str, str] = {
    # .score_heatmap
    "plot_attention_head_heatmap": "score_heatmap",
    "plot_residual_stream_heatmap": "score_heatmap",
    "plot_variable_localization_heatmap": "score_heatmap",
    # .binary_mask
    "plot_binary_mask": "binary_mask",
    "get_selected_units": "binary_mask",
    "plot_attention_head_mask": "binary_mask",
    "plot_residual_stream_mask": "binary_mask",
    "plot_mlp_mask": "binary_mask",
    "get_selected_heads": "binary_mask",
    "get_selected_residual_positions": "binary_mask",
    "get_selected_mlps": "binary_mask",
    # .grid_cells
    "GridCell": "grid_cells",
    "cell_grid_dimensions": "grid_cells",
    # .feature_masks
    "plot_feature_counts": "feature_masks",
    "plot_attention_head_feature_counts": "feature_masks",
    "plot_residual_stream_feature_counts": "feature_masks",
    "plot_mlp_feature_counts": "feature_masks",
    # .text_analysis
    "print_residual_stream_patching_analysis": "text_analysis",
    # .pca_scatter
    "plot_pca_scatter": "pca_scatter",
    "plot_features_2d": "pca_scatter",
    # .receptive_field
    "build_receptive_field_figure": "receptive_field",
    "plot_receptive_field": "receptive_field",
    # .mds
    "mds_embed": "mds",
    # .distance_plots
    "plot_distance_scatter": "distance_plots",
    "plot_dual_mds": "distance_plots",
    # .figure_format
    "ALLOWED_FIGURE_FORMATS": "figure_format",
    "FigureFormat": "figure_format",
    "normalize_figure_format": "figure_format",
    "path_with_figure_format": "figure_format",
    # .causal_graph
    "DEFAULT_COLORS": "causal_graph",
    "build_forward_pass_app": "causal_graph",
    "build_interchange_app": "causal_graph",
    "build_setting_figure": "causal_graph",
    "build_structure_app": "causal_graph",
    "build_structure_figure": "causal_graph",
    "display_forward_pass": "causal_graph",
    "display_interchange": "causal_graph",
    "display_structure": "causal_graph",
    "print_setting": "causal_graph",
    "print_structure": "causal_graph",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Import the submodule owning ``name`` on first access (PEP 562)."""
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module(f".{module}", __name__), name)
    globals()[name] = value  # cache, so the next access is a plain lookup
    return value


def __dir__() -> list[str]:
    return __all__


if TYPE_CHECKING:  # explicit re-exports, for type checkers and IDEs only
    from .score_heatmap import (
        plot_attention_head_heatmap as plot_attention_head_heatmap,
        plot_residual_stream_heatmap as plot_residual_stream_heatmap,
        plot_variable_localization_heatmap as plot_variable_localization_heatmap,
    )
    from .binary_mask import (
        get_selected_heads as get_selected_heads,
        get_selected_mlps as get_selected_mlps,
        get_selected_residual_positions as get_selected_residual_positions,
        get_selected_units as get_selected_units,
        plot_attention_head_mask as plot_attention_head_mask,
        plot_binary_mask as plot_binary_mask,
        plot_mlp_mask as plot_mlp_mask,
        plot_residual_stream_mask as plot_residual_stream_mask,
    )
    from .grid_cells import (
        GridCell as GridCell,
        cell_grid_dimensions as cell_grid_dimensions,
    )
    from .feature_masks import (
        plot_attention_head_feature_counts as plot_attention_head_feature_counts,
        plot_feature_counts as plot_feature_counts,
        plot_mlp_feature_counts as plot_mlp_feature_counts,
        plot_residual_stream_feature_counts as plot_residual_stream_feature_counts,
    )
    from .text_analysis import (
        print_residual_stream_patching_analysis as print_residual_stream_patching_analysis,
    )
    from .pca_scatter import (
        plot_features_2d as plot_features_2d,
        plot_pca_scatter as plot_pca_scatter,
    )
    from .receptive_field import (
        build_receptive_field_figure as build_receptive_field_figure,
        plot_receptive_field as plot_receptive_field,
    )
    from .mds import (
        mds_embed as mds_embed,
    )
    from .distance_plots import (
        plot_distance_scatter as plot_distance_scatter,
        plot_dual_mds as plot_dual_mds,
    )
    from .figure_format import (
        ALLOWED_FIGURE_FORMATS as ALLOWED_FIGURE_FORMATS,
        FigureFormat as FigureFormat,
        normalize_figure_format as normalize_figure_format,
        path_with_figure_format as path_with_figure_format,
    )
    from .causal_graph import (
        DEFAULT_COLORS as DEFAULT_COLORS,
        build_forward_pass_app as build_forward_pass_app,
        build_interchange_app as build_interchange_app,
        build_setting_figure as build_setting_figure,
        build_structure_app as build_structure_app,
        build_structure_figure as build_structure_figure,
        display_forward_pass as display_forward_pass,
        display_interchange as display_interchange,
        display_structure as display_structure,
        print_setting as print_setting,
        print_structure as print_structure,
    )
