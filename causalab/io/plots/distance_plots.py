"""General-purpose distance matrix visualization tools.

Scatter plots comparing two distance matrices and side-by-side MDS
embeddings for visual geometry comparison.  Domain-agnostic — callers
supply labels, titles, and annotations.
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np

from causalab.io.plots.mds import mds_embed
from causalab.io.plots.figure_format import path_with_figure_format

# Eagerly import plot_utils so its module-level _register_custom_colormaps()
# call runs (registers dark_seismic, etc.) before any colormap lookup here.
from causalab.io.plots import plot_utils as _plot_utils  # noqa: F401  # pyright: ignore[reportUnusedImport]

logger = logging.getLogger(__name__)


def plot_distance_scatter(
    D_X: np.ndarray,
    D_Y: np.ndarray,
    output_path: str,
    *,
    figure_format: str = "png",
    x_label: str = "Distance X",
    y_label: str = "Distance Y",
    title: str = "Distance Comparison",
    annotations: dict[str, float] | None = None,
    fit_slope: float | None = None,
    dpi: int = 150,
) -> None:
    """Scatter plot comparing upper-triangle entries of two distance matrices.

    Args:
        D_X: (N, N) first distance matrix (x-axis).
        D_Y: (N, N) second distance matrix (y-axis).
        output_path: File path to save the figure (e.g. ``"out/scatter.png"``).
        figure_format: ``png`` or ``pdf`` (replaces ``output_path`` extension when saving).
        x_label: X-axis label.
        y_label: Y-axis label.
        title: Figure title.
        annotations: Key-value pairs displayed in a text box (upper-left).
        fit_slope: If provided, draws a ``y = slope * x`` reference line.
        dpi: Output resolution.
    """
    if D_X.ndim == 1:
        dx, dy = D_X, D_Y
    else:
        idx = np.triu_indices_from(D_X, k=1)
        dx = D_X[idx]
        dy = D_Y[idx]

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.scatter(dx, dy, alpha=0.3, s=8, c="steelblue", edgecolors="none")

    if fit_slope is not None and np.isfinite(fit_slope):
        x_range = np.linspace(0, dx.max() * 1.05, 100)
        ax.plot(
            x_range,
            fit_slope * x_range,
            "r-",
            linewidth=1.5,
            label=f"y = {fit_slope:.3f}x",
        )
        ax.legend(fontsize=10)

    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(title, fontsize=13)

    if annotations:
        text_lines = [
            f"{k} = {v:.4f}" if isinstance(v, float) else f"{k} = {v}"
            for k, v in annotations.items()
        ]
        ax.text(
            0.05,
            0.95,
            "\n".join(text_lines),
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.5),
        )

    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    out = path_with_figure_format(output_path, figure_format)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    logger.info("Saved distance scatter plot to %s", out)


def plot_dual_mds(
    D_left: np.ndarray,
    D_right: np.ndarray,
    output_path: str,
    *,
    color_values: np.ndarray,
    left_title: str = "Space A (MDS)",
    right_title: str = "Space B (MDS)",
    title: str = "Distance Geometry Comparison",
    subtitle: str = "",
    hover_labels: list[str] | None = None,
    colorbar_title: str = "value",
    colorscale: str | list = "Viridis",
    n_mds_components: int = 3,
    width: int = 1200,
    height: int = 600,
    n_centroids: int | None = None,
    edges: list[tuple[int, int]] | None = None,
) -> None:
    """Side-by-side MDS embeddings of two distance matrices (interactive Plotly HTML).

    Points in both panels share the same color mapping so structural
    correspondence is visually apparent.

    Args:
        D_left: (N, N) distance matrix for the left panel.
        D_right: (N, N) distance matrix for the right panel.
        output_path: File path for the HTML output.
        color_values: (N,) array for coloring points (shared across panels).
        left_title: Subplot title for the left panel.
        right_title: Subplot title for the right panel.
        title: Overall figure title.
        subtitle: Annotation line below the title (e.g. metric summary).
        hover_labels: Per-point hover text. If None, points show index only.
        colorbar_title: Label for the shared colorbar.
        colorscale: Plotly colorscale name.
        n_mds_components: 2 or 3 — controls 2D vs 3D scatter.
        width: Figure width in pixels.
        height: Figure height in pixels.
        n_centroids: Truncate centroid count for the diamond markers.
        edges: Optional list of ``(i, j)`` index pairs (into the first
            ``n_centroids`` points) to overlay as graph edges in both
            panels. Useful for graph-walk tasks where the underlying
            adjacency is informative.
    """
    import plotly.graph_objects as go  # pyright: ignore[reportMissingTypeStubs]
    from plotly.subplots import make_subplots  # pyright: ignore[reportMissingTypeStubs]

    logger.info("Computing %dD MDS embeddings", n_mds_components)
    pts_left = mds_embed(D_left, n_components=n_mds_components)
    pts_right = mds_embed(D_right, n_components=n_mds_components)

    V = pts_left.shape[0]
    n_c = V if n_centroids is None else min(int(n_centroids), V)
    cmin = float(np.nanmin(color_values))
    cmax = float(np.nanmax(color_values))

    hover = hover_labels or [f"point {i}" for i in range(len(color_values))]

    is_3d = n_mds_components == 3
    spec_type = "scene" if is_3d else "xy"

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": spec_type}, {"type": spec_type}]],
        subplot_titles=[left_title, right_title],
        horizontal_spacing=0.02,
    )

    marker_size = 4 if is_3d else 8
    centroid_size = marker_size + 4
    ScatterCls = go.Scatter3d if is_3d else go.Scatter

    shared_color_kwargs = dict(
        color=color_values,
        colorscale=colorscale,
        cmin=cmin,
        cmax=cmax,
    )

    def _add_panel(
        col_idx: int, pts: np.ndarray, panel_label: str, show_colorbar: bool
    ):
        # Interior points (everything past n_c) drawn first as small circles.
        if n_c < V:
            interior_marker = dict(
                size=marker_size,
                opacity=0.6,
                **{
                    k: v[n_c:] if k == "color" else v  # pyright: ignore[reportIndexIssue]  # only "color" entry is subscripted; dict has heterogeneous value types
                    for k, v in shared_color_kwargs.items()
                },
            )
            interior_marker["showscale"] = False
            fig.add_trace(
                ScatterCls(
                    x=pts[n_c:, 0],
                    y=pts[n_c:, 1],
                    **({"z": pts[n_c:, 2]} if is_3d else {}),
                    mode="markers",
                    marker=interior_marker,
                    text=hover[n_c:],
                    hoverinfo="text",
                    name=f"{panel_label} (interior)",
                ),
                row=1,
                col=col_idx,
            )

        # Centroids (first n_c) drawn as larger diamonds for emphasis.
        centroid_marker = dict(
            size=centroid_size,
            symbol="diamond" if is_3d else "diamond",
            opacity=1.0,
            line=dict(width=1, color="black"),
            **{
                k: v[:n_c] if k == "color" else v  # pyright: ignore[reportIndexIssue]  # only "color" entry is subscripted; dict has heterogeneous value types
                for k, v in shared_color_kwargs.items()
            },
        )
        if show_colorbar:
            centroid_marker["colorbar"] = dict(title=colorbar_title, x=1.02)
        else:
            centroid_marker["showscale"] = False
        fig.add_trace(
            ScatterCls(
                x=pts[:n_c, 0],
                y=pts[:n_c, 1],
                **({"z": pts[:n_c, 2]} if is_3d else {}),
                mode="markers",
                marker=centroid_marker,
                text=hover[:n_c],
                hoverinfo="text",
                name=f"{panel_label} (centroids)",
            ),
            row=1,
            col=col_idx,
        )

        # Optional graph edges between centroids.
        if edges:
            xs: list[float | None] = []
            ys: list[float | None] = []
            zs: list[float | None] = []
            for i, j in edges:
                if i >= n_c or j >= n_c:
                    continue
                xs.extend([pts[i, 0], pts[j, 0], None])
                ys.extend([pts[i, 1], pts[j, 1], None])
                if is_3d:
                    zs.extend([pts[i, 2], pts[j, 2], None])
            if xs:
                edge_kwargs = dict(
                    x=xs,
                    y=ys,
                    mode="lines",
                    line=dict(color="rgba(100,100,100,0.3)", width=1.5),
                    name=f"{panel_label} (edges)",
                    hoverinfo="skip",
                    showlegend=False,
                )
                if is_3d:
                    edge_kwargs["z"] = zs
                fig.add_trace(ScatterCls(**edge_kwargs), row=1, col=col_idx)

    _add_panel(1, pts_left, left_title, show_colorbar=False)
    _add_panel(2, pts_right, right_title, show_colorbar=True)

    full_title = f"{title}<br><sub>{subtitle}</sub>" if subtitle else title

    layout_kwargs: dict = {
        "title": full_title,
        "width": width,
        "height": height,
        "showlegend": False,
    }

    if is_3d:
        axis_labels = dict(
            xaxis_title="MDS 1", yaxis_title="MDS 2", zaxis_title="MDS 3"
        )
        layout_kwargs["scene"] = axis_labels
        layout_kwargs["scene2"] = axis_labels
    else:
        layout_kwargs["xaxis"] = dict(title="MDS 1")
        layout_kwargs["yaxis"] = dict(title="MDS 2")
        layout_kwargs["xaxis2"] = dict(title="MDS 1")
        layout_kwargs["yaxis2"] = dict(title="MDS 2")

    fig.update_layout(**layout_kwargs)

    # Apply consistent floor/shadow styling matching the other 3D plots.
    # plotly subplots gives us two scenes ("scene" + "scene2"), and shadow
    # traces inherit a scene assignment from the trace they shadow — so we
    # compute per-scene z_floors and per-scene shadows.
    if is_3d:
        import plotly.graph_objects as _go  # pyright: ignore[reportMissingTypeStubs]
        from causalab.io.plots.plot_utils import FigureGenerator

        fg = FigureGenerator()
        axis_common = dict(
            showgrid=False,
            zeroline=False,
            showline=False,
            backgroundcolor="white",
            showticklabels=False,
            title="",
        )
        scene_layout = dict(
            xaxis=axis_common,
            yaxis=axis_common,
            zaxis={**axis_common, "backgroundcolor": "#e5e5e5"},
            xaxis_showbackground=False,
            yaxis_showbackground=False,
            zaxis_showbackground=True,
            aspectmode="data",
        )
        fig.update_layout(
            scene=scene_layout, scene2=scene_layout, paper_bgcolor="white"
        )

        # Add per-scene shadows (skip circles per requested behavior).
        for scene_key in ("scene", "scene2"):
            scene_traces = [
                (i, t)
                for i, t in enumerate(fig.data)
                if isinstance(t, _go.Scatter3d)
                and getattr(t, "scene", None) == scene_key
                and t.z is not None
                and len(t.z) > 0
            ]
            if not scene_traces:
                continue
            z_vals = []
            for _, t in scene_traces:
                z_vals.extend([v for v in t.z if v is not None])  # pyright: ignore[reportOptionalIterable]  # filtered above by `t.z is not None`
            if not z_vals:
                continue
            z_floor = min(z_vals)
            for _, t in scene_traces:
                m = t.marker or {}
                symbol = m.symbol if m.symbol is not None else "circle"  # pyright: ignore[reportAttributeAccessIssue]  # plotly Marker dynamic attribute typed as unknown
                if symbol == "circle":
                    continue
                z_proj = [z_floor] * len(t.z)  # pyright: ignore[reportArgumentType]  # filtered above by `t.z is not None`
                shadow = _go.Scatter3d(
                    x=t.x,
                    y=t.y,
                    z=z_proj,
                    mode="markers",
                    marker=dict(
                        size=m.size if m.size is not None else 2,  # pyright: ignore[reportAttributeAccessIssue]  # plotly Marker dynamic attribute typed as unknown
                        color=fg.shadow_color,
                        symbol=symbol,
                        opacity=fg.shadow_opacity,
                    ),
                    scene=scene_key,
                    showlegend=False,
                    hoverinfo="skip",
                )
                fig.add_trace(shadow)

    from causalab.io.plots.plot_utils import PLOTLY_HTML_CONFIG

    fig.write_html(output_path, config=PLOTLY_HTML_CONFIG)
    logger.info("Saved interactive MDS plot to %s", output_path)


def plot_matrix_heatmap(
    matrix,
    row_labels: list[str],
    col_labels: list[str] | None,
    output_dir: str,
    filename: str = "ref_dists_heatmap",
    title: str = "Reference distributions",
    figure_format: str | None = None,
    xlabel: str = "Output token",
    ylabel: str = "Ground truth",
) -> None:
    """Plot a similarity/confusion matrix as a seaborn heatmap.

    Args:
        matrix (np.ndarray): (N, M) matrix to plot.
        row_labels: Labels for rows (ground-truth classes).
        col_labels: Labels for columns (output tokens). Defaults to row_labels.
        output_dir: Directory to save the plot.
        filename: Output filename.
        title: Plot title.
        figure_format: Output format (``png`` or ``pdf``); defaults to ``png`` when unset.
        xlabel: Label for x-axis.
        ylabel: Label for y-axis.
    """
    import os

    import seaborn as sns

    if col_labels is None:
        col_labels = row_labels

    fig, ax = plt.subplots(
        figsize=(max(6, len(col_labels) * 0.6), max(5, len(row_labels) * 0.5))
    )
    sns.heatmap(
        matrix.numpy(),
        xticklabels=col_labels,
        yticklabels=row_labels,
        cmap="bone_r",
        annot=False,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    out = path_with_figure_format(os.path.join(output_dir, filename), figure_format)
    fig.savefig(out)
    plt.close(fig)
