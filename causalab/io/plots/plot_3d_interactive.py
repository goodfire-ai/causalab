"""
Generic 3D interactive visualizations for the manifold steering pipeline.

Produces three incremental Plotly HTML files:
1. features_3d.html  — Training features + centroids colored by causal parameter
2. manifold_3d.html  — Features + manifold mesh surface
3. steering_results.html — Manifold surface + steering points colored by inferred stats

Each file has a single dropdown to switch between (parameter, scale) combinations
where scale is one of: linear, log, categorical.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
import plotly.graph_objects as go  # pyright: ignore[reportMissingTypeStubs]
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from torch import Tensor

from causalab.io.centroids import (
    compute_centroids,
    extract_parameters_from_dataset,
)

logger = logging.getLogger(__name__)

NDArray = np.ndarray[Any, np.dtype[Any]]

# Type alias for edge lists: each tuple is (source_idx, target_idx) into centroids
EdgeList = list[tuple[int, int]]

# Qualitative colorscale for categorical mode (plotly named colors)
_QUAL_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#aec7e8",
    "#ffbb78",
    "#98df8a",
    "#ff9896",
    "#c5b0d5",
    "#c49c94",
    "#f7b6d2",
    "#c7c7c7",
    "#dbdb8d",
    "#9edae5",
]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _get_manifold_device(manifold):
    """Get device from a manifold (works for flow and spline)."""
    p = next(manifold.parameters(), None)
    if p is not None:
        return p.device
    b = next(manifold.buffers(), None)
    if b is not None:
        return b.device
    return torch.device("cpu")


def _fit_mds(
    distance_matrix: NDArray,
    n_components: int = 3,
) -> NDArray:
    """MDS embedding from a precomputed distance matrix.

    Returns (n, n_components) embedding.
    """
    from sklearn.manifold import MDS

    mds = MDS(
        n_components=n_components,
        dissimilarity="precomputed",
        random_state=42,
        normalized_stress="auto",
    )
    return mds.fit_transform(distance_matrix)


def _fit_projector(
    features: Tensor,
    n_components: int = 3,
    pca_components: Sequence[int] | None = None,
) -> tuple[NDArray, Callable[[NDArray], NDArray]]:
    """PCA on k-dim features if k > n_components, else identity/pad.

    Returns (projected, project_fn) where project_fn maps (n, k) -> (n, n_components).

    If ``pca_components`` is provided, fits PCA with enough components to cover
    the largest requested index and selects those columns (in the given order).
    Length must equal ``n_components``.
    """
    features_np = features.detach().cpu().float().numpy()
    k = features_np.shape[1]

    if pca_components is not None:
        idx = list(pca_components)
        if len(idx) != n_components:
            raise ValueError(
                f"pca_components must have length {n_components}, got {len(idx)}"
            )
        if any(i < 0 for i in idx):
            raise ValueError(f"pca_components must be non-negative, got {idx}")
        max_needed = max(idx) + 1
        if k < max_needed:
            raise ValueError(
                f"pca_components={idx} requires at least {max_needed} feature dims, got {k}"
            )
        pca = PCA(n_components=max_needed)
        full = pca.fit_transform(features_np)
        projected = full[:, idx]

        def _proj(x: NDArray) -> NDArray:
            return pca.transform(x)[:, idx]

        return projected, _proj

    if k > n_components:
        pca = PCA(n_components=n_components)
        projected = pca.fit_transform(features_np)
        return projected, pca.transform
    if k == n_components:
        return features_np, lambda x: x
    # k < n_components: pad with zeros
    padded = np.zeros((features_np.shape[0], n_components))
    padded[:, :k] = features_np

    def _pad(x: NDArray) -> NDArray:
        out = np.zeros((x.shape[0], n_components))
        out[:, :k] = x
        return out

    return padded, _pad


def _make_edge_traces(
    centroids_3d: NDArray,
    edges: EdgeList,
) -> list[go.Scatter3d]:
    """Build a Scatter3d trace with straight-line edges between centroids.

    Args:
        centroids_3d: (n_centroids, 3) array of centroid positions.
        edges: List of (i, j) index pairs into centroids_3d.

    Returns:
        List containing a single Scatter3d trace with all edges.
    """
    xs: list[float | None] = []
    ys: list[float | None] = []
    zs: list[float | None] = []

    for i, j in edges:
        if i >= len(centroids_3d) or j >= len(centroids_3d):
            continue
        xs.extend([centroids_3d[i, 0], centroids_3d[j, 0], None])
        ys.extend([centroids_3d[i, 1], centroids_3d[j, 1], None])
        zs.extend([centroids_3d[i, 2], centroids_3d[j, 2], None])

    if not xs:
        return []

    return [
        go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="lines",
            line=dict(color="rgba(100,100,100,0.3)", width=1.5),
            name="Edges",
            hoverinfo="skip",
            showlegend=False,
        )
    ]


def _extract_param_values(
    train_dataset: list,
    intervention_variable: str | None = None,
    embeddings: dict | None = None,
) -> tuple[dict[str, NDArray], list[str]]:
    """Extract numeric causal parameter values from input traces.

    Only extracts the intervention variable numerically (for coordinates/coloring).
    Other variables are shown in hover labels directly from the dataset.

    Args:
        train_dataset: Counterfactual examples.
        intervention_variable: If set, only extract this variable (exclude all
            others by building excluded_vars from the first trace).
        embeddings: Optional dict mapping variable names to embedding functions
            for string-valued variables.

    Returns (param_dict, sorted_param_names) where param_dict maps
    param_name -> array of shape (n_examples,), one value per example,
    aligned with features from collect_features().
    """
    # Always extract only the intervention variable numerically (for coordinates/coloring).
    # Hover labels for other variables are built separately from the dataset.
    excluded_vars: set[str] | None = None
    if intervention_variable is not None and train_dataset:
        first_input = train_dataset[0].get("input", {})
        if hasattr(first_input, "_values"):
            all_vars = set(first_input._values.keys())
        elif isinstance(first_input, dict):
            all_vars = set(first_input.keys())
        else:
            all_vars = set()
        if all_vars:
            excluded_vars = all_vars - {intervention_variable}

    param_tensors = extract_parameters_from_dataset(
        train_dataset,
        excluded_vars=excluded_vars,
        embeddings=embeddings,
    )
    param_dict = {k: v.detach().cpu().numpy() for k, v in param_tensors.items()}
    sorted_names = sorted(param_dict.keys())
    return param_dict, sorted_names


def _compute_grid_stats(
    steer_result: dict,
    dist_module: Any,
    config: Any,
) -> tuple[dict[str, NDArray], list[str]]:
    """Compute output statistics for each steering grid point.

    Returns (stat_dict, stat_names) where stat_dict maps stat_name ->
    array of shape (n_grid,). Scalar stats only (multi-dim stats like
    mu=(x,y) are expanded to mu_x, mu_y).
    """
    scores_by_gp = steer_result.get("scores_by_grid_point", {})
    grid_points = steer_result["grid_points"]
    n_grid = len(grid_points)

    if not scores_by_gp or not hasattr(dist_module, "compute_output_stats"):
        return {}, []

    # Collect per-grid-point average probabilities
    all_stats: dict[str, list[float]] = {}
    for grid_idx in range(n_grid):
        scores_list = scores_by_gp.get(grid_idx, [])
        if not scores_list:
            # Will fill with NaN
            for k in all_stats:
                all_stats[k].append(float("nan"))
            continue

        # Stack scores into (n_examples, n_bins)
        tensors = []
        for s in scores_list:
            if isinstance(s, Tensor):
                tensors.append(s)
            elif isinstance(s, list) and len(s) > 0:
                tensors.append(s[0] if isinstance(s[0], Tensor) else torch.tensor(s[0]))
        if not tensors:
            for k in all_stats:
                all_stats[k].append(float("nan"))
            continue

        combined = torch.cat(tensors, dim=0)
        probs = F.softmax(combined, dim=-1)
        avg_probs = probs.mean(dim=0, keepdim=True)

        stats = dist_module.compute_output_stats(avg_probs, config.distribution_config)

        for stat_name, stat_val in stats.items():
            if not isinstance(stat_val, Tensor):
                all_stats.setdefault(stat_name, []).append(float(stat_val))
                continue
            v = stat_val.detach().cpu()
            if v.ndim == 0 or (v.ndim == 1 and v.shape[0] == 1):
                all_stats.setdefault(stat_name, []).append(float(v.squeeze()))
            elif v.ndim == 1 and v.shape[0] > 1:
                # Multi-dim stat — expand
                for j in range(v.shape[0]):
                    expanded_name = f"{stat_name}_{j}"
                    all_stats.setdefault(expanded_name, []).append(float(v[j]))
            else:
                # (1, D) from keepdim
                v_flat = v.squeeze(0)
                if v_flat.ndim == 0:
                    all_stats.setdefault(stat_name, []).append(float(v_flat))
                else:
                    for j in range(v_flat.shape[0]):
                        expanded_name = f"{stat_name}_{j}"
                        all_stats.setdefault(expanded_name, []).append(float(v_flat[j]))

    # Pad any short entries with NaN
    for k in all_stats:
        while len(all_stats[k]) < n_grid:
            all_stats[k].append(float("nan"))

    stat_dict = {k: np.array(v) for k, v in all_stats.items()}
    stat_names = sorted(stat_dict.keys())
    return stat_dict, stat_names


def _build_1d_curve_trace(
    manifold_obj: Any,
    mean_d: Tensor,
    std_d: Tensor,
    ranges: tuple,
    project_fn: Callable[[NDArray], NDArray],
    device: torch.device,
    grid_res: int = 100,
    periodic: bool = False,
    name: str = "Manifold curve",
    color: str = "darkgray",
    width: int = 3,
) -> go.Scatter3d:
    """Build a Scatter3d line trace for a 1D manifold (curve).

    If periodic, the curve is closed by appending the first point.

    The sampled u values include all control_points so the rendered polyline
    visually passes through every projected centroid.
    """
    u = np.linspace(ranges[0][0], ranges[0][1], grid_res, endpoint=not periodic)
    if hasattr(manifold_obj, "control_points"):
        cp = manifold_obj.control_points.detach().cpu().float().numpy().reshape(-1)
        # Filter to those within the rendered range
        cp = cp[(cp >= ranges[0][0]) & (cp <= ranges[0][1])]
        u = np.unique(np.concatenate([u, cp]))
    intrinsic = u[:, None]  # (n_samples, 1)

    u_tensor = torch.tensor(intrinsic, dtype=torch.float32, device=device)
    with torch.no_grad():
        decoded = manifold_obj.decode(u_tensor, r=None)
        decoded = decoded * (std_d + 1e-6) + mean_d
    points_3d = project_fn(decoded.cpu().numpy())

    if periodic:
        # Close the loop
        points_3d = np.concatenate([points_3d, points_3d[:1]], axis=0)

    return go.Scatter3d(
        x=points_3d[:, 0],
        y=points_3d[:, 1],
        z=points_3d[:, 2],
        mode="lines",
        line=dict(color=color, width=width),
        name=name,
        hoverinfo="skip",
    )


def _build_2d_mesh_slice(
    manifold_obj: Any,
    mean_d: Tensor,
    std_d: Tensor,
    ranges: tuple,
    project_fn: Callable[[NDArray], NDArray],
    device: torch.device,
    grid_res: int = 50,
    fixed_dims: dict[int, float] | None = None,
    periodic_dims: set[int] | None = None,
    name: str = "Manifold surface",
    color: str = "darkgray",
    opacity: float = 0.3,
) -> go.Mesh3d:
    """Build a single 2D mesh slice through the manifold.

    Args:
        fixed_dims: Maps dimension index → fixed value. The remaining 2 dims
            are varied on a regular grid. If None, requires len(ranges)==2.
        periodic_dims: Set of intrinsic dimension indices that are periodic.
            For periodic free dims, the grid wraps (last column stitches to
            first) to produce a closed surface.
    """
    d = len(ranges)
    if fixed_dims is None:
        fixed_dims = {}
    if periodic_dims is None:
        periodic_dims = set()
    free_dims = [i for i in range(d) if i not in fixed_dims]
    assert len(free_dims) == 2, f"Expected 2 free dims, got {len(free_dims)}"

    d0, d1 = free_dims
    d0_periodic = d0 in periodic_dims
    d1_periodic = d1 in periodic_dims

    # For periodic dims, sample grid_res points in [lo, hi) then duplicate
    # the first column/row to close the surface.
    res0 = grid_res
    res1 = grid_res
    u0 = np.linspace(ranges[d0][0], ranges[d0][1], res0, endpoint=not d0_periodic)
    u1 = np.linspace(ranges[d1][0], ranges[d1][1], res1, endpoint=not d1_periodic)
    if d0_periodic:
        u0 = np.concatenate([u0, u0[:1]])
        res0 += 1
    if d1_periodic:
        u1 = np.concatenate([u1, u1[:1]])
        res1 += 1

    g0, g1 = np.meshgrid(u0, u1, indexing="ij")

    intrinsic = np.zeros((res0 * res1, d))
    intrinsic[:, d0] = g0.ravel()
    intrinsic[:, d1] = g1.ravel()
    for dim_idx, val in fixed_dims.items():
        intrinsic[:, dim_idx] = val

    u_tensor = torch.tensor(intrinsic, dtype=torch.float32, device=device)
    with torch.no_grad():
        decoded = manifold_obj.decode(u_tensor, r=None)
        decoded = decoded * (std_d + 1e-6) + mean_d
    points_3d = project_fn(decoded.cpu().numpy())

    # Build triangle indices from grid topology: each quad → 2 triangles
    ii, jj, kk = [], [], []
    for i in range(res0 - 1):
        for j in range(res1 - 1):
            idx00 = i * res1 + j
            idx10 = (i + 1) * res1 + j
            idx01 = i * res1 + (j + 1)
            idx11 = (i + 1) * res1 + (j + 1)
            ii.extend([idx00, idx00])
            jj.extend([idx10, idx11])
            kk.extend([idx11, idx01])

    return go.Mesh3d(
        x=points_3d[:, 0],
        y=points_3d[:, 1],
        z=points_3d[:, 2],
        i=ii,
        j=jj,
        k=kk,
        color=color,
        opacity=opacity,
        name=name,
        hoverinfo="skip",
    )


def _build_manifold_mesh_traces(
    manifold_obj: Any,
    mean: Tensor,
    std: Tensor,
    ranges: tuple,
    project_fn: Callable[[NDArray], NDArray],
    grid_res: int = 50,
    param_names: list[str] | None = None,
) -> list:
    """Build Plotly traces for the manifold surface.

    For d=1: a Scatter3d curve (closed if periodic).
    For d=2: single regular-grid mesh (wrapping periodic dims).
    For d>2: 2d boundary faces of the intrinsic hypercube.

    Periodic dimensions are read from ``manifold_obj.periodic_dims``
    (if available) so that the mesh/curve wraps correctly.
    """
    d = len(ranges)
    device = _get_manifold_device(manifold_obj)
    mean_d = mean.to(device)
    std_d = std.to(device)

    # Read periodic dims from manifold if available
    periodic_dims: set[int] = set()
    if hasattr(manifold_obj, "periodic_dims"):
        periodic_dims = set(manifold_obj.periodic_dims)

    if d == 1:
        return [
            _build_1d_curve_trace(
                manifold_obj,
                mean_d,
                std_d,
                ranges,
                project_fn,
                device,
                grid_res=500,  # dense enough for visual smoothness; union with
                # control_points (in _build_1d_curve_trace) ensures
                # the polyline still passes exactly through markers
                periodic=0 in periodic_dims,
            )
        ]

    if d == 2:
        return [
            _build_2d_mesh_slice(
                manifold_obj,
                mean_d,
                std_d,
                ranges,
                project_fn,
                device,
                grid_res=grid_res,
                periodic_dims=periodic_dims,
            )
        ]

    if d == 3:
        # 6 boundary faces: fix each dim at its min/max, vary the other 2
        traces = []
        for fix_dim in range(d):
            dim_name = param_names[fix_dim] if param_names else f"dim{fix_dim}"
            lo, hi = ranges[fix_dim]
            for val, side in [(lo, "min"), (hi, "max")]:
                label = f"{dim_name}={val:.3g} ({side})"
                traces.append(
                    _build_2d_mesh_slice(
                        manifold_obj,
                        mean_d,
                        std_d,
                        ranges,
                        project_fn,
                        device,
                        grid_res=grid_res,
                        fixed_dims={fix_dim: val},
                        periodic_dims=periodic_dims,
                        name=label,
                    )
                )
    else:
        # d > 3: pick all pairs of free dims, fix the rest at midpoints
        from itertools import combinations

        traces = []
        for free_d0, free_d1 in combinations(range(d), 2):
            fixed = {
                i: (ranges[i][0] + ranges[i][1]) / 2
                for i in range(d)
                if i not in (free_d0, free_d1)
            }
            name_parts = [
                f"{(param_names[i] if param_names else f'd{i}')}={v:.2g}"
                for i, v in fixed.items()
            ]
            traces.append(
                _build_2d_mesh_slice(
                    manifold_obj,
                    mean_d,
                    std_d,
                    ranges,
                    project_fn,
                    device,
                    grid_res=grid_res,
                    fixed_dims=fixed,
                    periodic_dims=periodic_dims,
                    name=f"Slice ({', '.join(name_parts)})",
                )
            )

    return traces


def _resolve_categorical_colors(
    n: int,
    colormap: str | None = None,
) -> list[str]:
    """Return n hex colors for categorical scatter traces.

    Args:
        n: Number of distinct categories.
        colormap: Matplotlib colormap name, or None for the default palette.
    """
    if colormap is None:
        return [_QUAL_COLORS[i % len(_QUAL_COLORS)] for i in range(n)]
    import matplotlib.pyplot as _plt

    cmap = _plt.get_cmap(colormap)
    import matplotlib.colors as _mc

    return [_mc.to_hex(cmap(i / max(n - 1, 1))) for i in range(n)]


def _resolve_continuous_colorscale(
    colormap: str | None = None,
) -> str | list[list[float | str]]:
    """Return a Plotly-compatible colorscale name for continuous data.

    If the colormap is qualitative / categorical-only (tab10, Set1, etc.),
    falls back to 'Viridis'.
    """
    if colormap is None:
        return "Viridis"

    _QUALITATIVE = {
        "tab10",
        "tab20",
        "tab20b",
        "tab20c",
        "Set1",
        "Set2",
        "Set3",
        "Pastel1",
        "Pastel2",
        "Paired",
        "Accent",
        "Dark2",
    }
    if colormap in _QUALITATIVE:
        return "Viridis"

    # Try to use the matplotlib colormap as a Plotly colorscale
    # by sampling it into a list of [fraction, hex] pairs.
    try:
        import matplotlib.pyplot as _plt
        import matplotlib.colors as _mc

        cmap = _plt.get_cmap(colormap)
        steps = 256
        return [
            [i / (steps - 1), _mc.to_hex(cmap(i / (steps - 1)))] for i in range(steps)
        ]
    except Exception:
        return "Viridis"


def _make_trace_group_scatter(
    points_3d: NDArray,
    values: NDArray,
    param_name: str,
    scale: str,
    hover_extra: list[str] | None = None,
    size: int = 3,
    opacity: float = 0.3,
    symbol: str = "circle",
    show_colorbar: bool = True,
    colorbar_ticktext: list[str] | None = None,
    marker_text: list[str] | None = None,
    colormap: str | None = None,
) -> list[go.Scatter3d]:
    """Create scatter traces for one (param, scale) combo.

    Linear/log: single continuous-color Scatter3d.
    Categorical: one Scatter3d per unique value.

    Args:
        colorbar_ticktext: If provided, replace numeric colorbar ticks with
            these labels (one per unique raw value). Only used in linear/log mode.
        marker_text: If provided, show text labels on markers. For categorical
            mode the labels are distributed across per-value traces.
        colormap: Matplotlib colormap name. Used for both categorical (sampled
            at discrete intervals) and continuous (converted to Plotly colorscale).
            If None, uses the default qualitative palette for categorical and
            Viridis for continuous. If the colormap is qualitative-only (tab10,
            Set1, etc.), continuous mode falls back to Viridis.
    """
    traces: list[go.Scatter3d] = []
    mode = "markers+text" if marker_text else "markers"

    if scale == "categorical":
        unique_vals = np.unique(values[~np.isnan(values)])
        n_unique = len(unique_vals)
        cat_colors = _resolve_categorical_colors(n_unique, colormap)
        for i, uval in enumerate(unique_vals):
            mask = values == uval
            color = cat_colors[i]
            name = f"{param_name}={uval:.4g}"
            hover = [hover_extra[j] for j in np.where(mask)[0]] if hover_extra else None
            text = [marker_text[j] for j in np.where(mask)[0]] if marker_text else None
            trace = go.Scatter3d(
                x=points_3d[mask, 0],
                y=points_3d[mask, 1],
                z=points_3d[mask, 2],
                mode=mode,
                opacity=opacity,
                marker=dict(
                    size=size,
                    color=color,
                    symbol=symbol,
                    line=dict(width=0.3, color="black")
                    if symbol != "circle"
                    else dict(width=0),
                ),
                name=name,
                text=text if marker_text else hover,
                textposition="top center" if marker_text else None,
                hovertext=hover if marker_text else None,
                hovertemplate="%{hovertext}<extra></extra>"
                if (marker_text and hover)
                else ("%{text}<extra></extra>" if hover else None),
                visible=False,
            )
            traces.append(trace)
        return traces

    # Linear or log
    if scale == "log":
        display_vals = np.log(np.abs(values) + 1e-10)
        cbar_title = f"{param_name} (log)"
    else:
        display_vals = values
        cbar_title = param_name

    valid = ~np.isnan(display_vals)
    cmin = float(np.nanmin(display_vals)) if valid.any() else 0
    cmax = float(np.nanmax(display_vals)) if valid.any() else 1

    # Build colorbar dict
    colorbar_dict: dict | None = None
    if show_colorbar:
        colorbar_dict = dict(title=cbar_title, x=1.02)
        if colorbar_ticktext is not None:
            raw_unique = np.unique(values[~np.isnan(values)])
            if scale == "log":
                tick_vals = np.log(np.abs(raw_unique) + 1e-10)
            else:
                tick_vals = raw_unique
            colorbar_dict["tickvals"] = tick_vals.tolist()
            colorbar_dict["ticktext"] = list(colorbar_ticktext)

    continuous_colorscale = _resolve_continuous_colorscale(colormap)

    trace = go.Scatter3d(
        x=points_3d[:, 0],
        y=points_3d[:, 1],
        z=points_3d[:, 2],
        mode=mode,
        opacity=opacity,
        marker=dict(
            size=size,
            color=display_vals,
            colorscale=continuous_colorscale,
            cmin=cmin,
            cmax=cmax,
            symbol=symbol,
            colorbar=colorbar_dict,
            line=dict(width=0.3, color="black")
            if symbol != "circle"
            else dict(width=0),
        ),
        name=f"{param_name} ({scale})",
        text=marker_text
        if marker_text
        else ([hover_extra[i] for i in range(len(points_3d))] if hover_extra else None),
        textposition="top center" if marker_text else None,
        hovertext=[hover_extra[i] for i in range(len(points_3d))]
        if (marker_text and hover_extra)
        else None,
        hovertemplate="%{hovertext}<extra></extra>"
        if (marker_text and hover_extra)
        else ("%{text}<extra></extra>" if hover_extra else None),
        visible=False,
    )
    traces.append(trace)
    return traces


def _assemble_dropdown_figure(
    always_visible_traces: list,
    param_trace_groups: list[tuple[str, list]],
    title: str,
    axis_labels: tuple[str, str, str],
    save_path: str,
) -> go.Figure:
    """Build figure with a single combined dropdown, write to HTML.

    Args:
        always_visible_traces: Traces always shown (mesh, etc.)
        param_trace_groups: List of (dropdown_label, [traces]) pairs
        title: Plot title
        axis_labels: (x, y, z) axis labels
        save_path: Output path
    """
    from causalab.io.plots.plot_utils import FigureGenerator

    fg = FigureGenerator()

    fig = go.Figure()

    # Add all traces to a temp figure to compute global z_floor
    all_scatter_traces = list(always_visible_traces)
    for _, traces in param_trace_groups:
        all_scatter_traces.extend(traces)
    temp_fig = go.Figure(data=all_scatter_traces)
    z_floor = fg._get_z_floor(temp_fig)  # pyright: ignore[reportPrivateUsage]

    # Add always-visible traces
    for t in always_visible_traces:
        t.visible = True
        fig.add_trace(t)

    # Apply 3D styling (floor/walls only, no shadows yet)
    fg.style_plotly_3d(fig, floor_shadow=False)

    # Add shadows for always-visible traces at the global z_floor
    fg._add_floor_shadows(fig, z_floor=z_floor)  # pyright: ignore[reportPrivateUsage]
    n_always = len(fig.data)  # pyright: ignore[reportArgumentType]  # plotly Figure.data is a tuple-like Sized container; stubs widen to Figure

    # Add dropdown trace groups + their shadows
    group_ranges: list[tuple[int, int]] = []
    for _, traces in param_trace_groups:
        start = len(fig.data)  # pyright: ignore[reportArgumentType]  # plotly Figure.data is a tuple-like Sized container; stubs widen to Figure
        for t in traces:
            t.visible = False
            fig.add_trace(t)
        # Add shadows for this group's traces (also hidden)
        shadow_start = len(fig.data)  # pyright: ignore[reportArgumentType]  # plotly Figure.data is a tuple-like Sized container; stubs widen to Figure
        fg._add_floor_shadows(  # pyright: ignore[reportPrivateUsage]
            fig, only_indices=list(range(start, shadow_start)), z_floor=z_floor
        )
        for i in range(shadow_start, len(fig.data)):  # pyright: ignore[reportArgumentType]  # plotly Figure.data is a tuple-like Sized container; stubs widen to Figure
            fig.data[i].visible = False
        group_ranges.append((start, len(fig.data)))  # pyright: ignore[reportArgumentType]  # plotly Figure.data is a tuple-like Sized container; stubs widen to Figure

    total_traces = len(fig.data)  # pyright: ignore[reportArgumentType]  # plotly Figure.data is a tuple-like Sized container; stubs widen to Figure

    # Build dropdown buttons
    buttons = []
    for idx, (label, _) in enumerate(param_trace_groups):
        vis = [True] * n_always + [False] * (total_traces - n_always)
        start, end = group_ranges[idx]
        for i in range(start, end):
            vis[i] = True
        buttons.append(
            dict(
                method="update",
                label=label,
                args=[{"visible": vis}],
            )
        )

    # Make first group visible by default
    if group_ranges:
        start, end = group_ranges[0]
        for i in range(start, end):
            fig.data[i].visible = True

    fig.update_layout(
        title=title,
        updatemenus=[
            dict(
                type="dropdown",
                direction="down",
                x=0.0,
                xanchor="left",
                y=1.15,
                yanchor="top",
                buttons=buttons,
                showactive=True,
            )
        ]
        if buttons
        else [],
        margin=dict(l=0, r=300, b=0, t=60),
        legend=dict(x=1.25, y=1.0, xanchor="left", yanchor="top"),
    )

    parent = os.path.dirname(save_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    from causalab.io.plots.plot_utils import PLOTLY_HTML_CONFIG

    fig.write_html(save_path, config=PLOTLY_HTML_CONFIG)
    logger.info(f"Saved {save_path}")
    return fig


# ---------------------------------------------------------------------------
# Unified 3D plot
# ---------------------------------------------------------------------------


@dataclass
class PathTrace:
    """A 3D path to overlay on the plot."""

    # Either intrinsic coords (decoded via manifold) or k-dim ambient coords
    points: Tensor
    name: str = "Path"
    color: str = "black"
    width: int = 5
    marker_size: int = 3
    is_intrinsic: bool = (
        True  # If True, decode via manifold; if False, project directly
    )
    dash: str | None = None  # "dash", "dot", "dashdot", or None for solid
    mode: str = "lines+markers"  # "lines", "lines+markers", "markers"


def _build_readable_hover(
    param_dict: dict[str, NDArray],
    param_names: list[str],
    n: int,
    variable_values: list[str] | None = None,
    embeddings: dict | None = None,
    intervention_variable: str | None = None,
    train_dataset: list | None = None,
    detailed_hover: bool = False,
    max_hover_chars: int = 50,
) -> list[str]:
    """Build hover text with readable value names where possible."""
    # Build reverse-lookup from embedded numeric value to string name
    val_to_name: dict[str, dict[float, str]] = {}
    if variable_values and embeddings and intervention_variable:
        embed_fn = embeddings.get(intervention_variable)
        if embed_fn:
            pname_map: dict[float, str] = {}
            for vv in variable_values:
                # Skip values the embedding can't parse (e.g. graph_walk's
                # tuples after str() round-trip).
                try:
                    coords = embed_fn(vv)
                except (ValueError, TypeError):
                    continue
                if len(coords) == 1:
                    pname_map[round(coords[0], 4)] = vv
            if pname_map:
                val_to_name[intervention_variable] = pname_map

    hover = []
    for fi in range(n):
        parts = []
        # Numeric params (intervention variable coordinates)
        for pname in param_names:
            vals = param_dict[pname]
            if fi < len(vals):
                val = vals[fi]
                lookup = val_to_name.get(pname)
                if lookup:
                    name = lookup.get(round(float(val), 4))
                    if name:
                        parts.append(f"{pname}={name}")
                        continue
                parts.append(f"{pname}={val:.4g}")
        # Append all trace variables when detailed_hover is enabled
        if detailed_hover and train_dataset and fi < len(train_dataset):
            example_input = train_dataset[fi].get("input", {})
            if hasattr(example_input, "_values"):
                trace_vars = example_input._values
            elif isinstance(example_input, dict):
                trace_vars = example_input
            else:
                trace_vars = {}
            shown = set(param_names)
            for var_name in sorted(trace_vars.keys()):
                if var_name in shown:
                    continue
                val_str = str(trace_vars[var_name])
                if len(val_str) > max_hover_chars:
                    val_str = val_str[:max_hover_chars] + "..."
                parts.append(f"{var_name}={val_str}")
        hover.append("<br>".join(parts))
    return hover


def _build_centroid_hover(
    centroid_param_names: list[str],
    centroid_param_values: NDArray,
    counts: list,
    variable_values: list[str] | None = None,
    embeddings: dict | None = None,
    intervention_variable: str | None = None,
) -> list[str]:
    """Build centroid hover text with readable value names."""
    embed_fn = (
        (embeddings or {}).get(intervention_variable) if intervention_variable else None
    )
    n_centroids = centroid_param_values.shape[0]

    hover = []
    for ci in range(n_centroids):
        # Try to map to readable name
        label = None
        if len(centroid_param_names) == 1 and variable_values and embed_fn:
            val = centroid_param_values[ci, 0]
            for vv in variable_values:
                # Embedding may not accept stringified values (e.g. graph_walk's
                # tuple `(0, 0)` round-tripped through ``str()``). When parsing
                # fails, fall through to the default coordinate-based label.
                try:
                    coords = embed_fn(vv)
                except (ValueError, TypeError):
                    continue
                if len(coords) == 1 and abs(coords[0] - val) < 1e-4:
                    label = vv
                    break
        if label is None:
            label = ", ".join(
                f"{centroid_param_names[j]}={centroid_param_values[ci, j]:.4g}"
                for j in range(len(centroid_param_names))
            )
        hover.append(f"{label}<br>n={counts[ci]}")
    return hover


def plot_3d(
    features: Tensor,
    output_path: str,
    train_dataset: list | None = None,
    title: str = "Features 3D",
    intervention_variable: str | None = None,
    edges: EdgeList | None = None,
    edge_node_coords: dict[int, dict[str, float]] | None = None,
    embeddings: dict | None = None,
    colormap: str | list[str] | None = None,
    param_dict: dict[str, NDArray] | None = None,
    # Manifold (optional — enables mesh surface)
    manifold_obj: Any = None,
    mean: Tensor | None = None,
    std: Tensor | None = None,
    ranges: tuple | None = None,
    # Paths (optional — enables path line traces)
    paths: list[PathTrace] | None = None,
    # For readable hover labels
    variable_values: list[str] | None = None,
    # Distance matrix (optional — use MDS instead of PCA)
    distance_matrix: Tensor | NDArray | None = None,
    # Pre-computed centroid positions in 3D (overrides default centroid projection)
    pre_computed_centroids_3d: NDArray | None = None,
    # Override centroid color (e.g. "black") — bypasses colormap for centroids
    centroid_color: str | None = None,
    # Detailed hover: show all causal variables + raw_input in hover labels
    detailed_hover: bool = False,
    max_hover_chars: int = 50,
    # Optional explicit PCA component indices to plot (length 3). When None,
    # uses the top 3 components.
    pca_components: Sequence[int] | None = None,
    # 'raw' (default): features are taken as-is, centroids = mean(features) per
    # class. 'hellinger': features must be per-example PROBABILITIES (n, W+1);
    # the function √-transforms them internally for display + PCA fit and uses
    # √(mean(p)) per class as centroids — the correct Hellinger centroid (on
    # the unit sphere). Use 'hellinger' whenever you have probability data and
    # want a Jensen-drift-free 3D scatter.
    feature_kind: str = "raw",
) -> None:
    """Unified 3D interactive visualization.

    Incrementally adds layers based on what's provided:
    - Always: training features + centroids colored by causal parameter (dropdown)
    - If manifold_obj/mean/std/ranges: manifold mesh surface
    - If paths: path line traces overlaid on the plot

    When mean/std are provided, centroids are computed in standardized space
    (matching manifold fitting) then denormalized for plotting.

    Args:
        features: (n, k) features in ambient (PCA) space.
        output_path: Path for the output HTML file.
        train_dataset: Counterfactual examples for param extraction.
        title: Plot title.
        intervention_variable: Variable to use for coloring/centroids.
        edges: (i, j) index pairs for edge lines between centroids.
        edge_node_coords: Node ID -> {param_name: value} for edge remapping.
        embeddings: Variable name -> embedding function.
        colormap: Matplotlib colormap name.
        manifold_obj: Manifold for mesh surface and path decoding.
        mean: (k,) standardization mean.
        std: (k,) standardization std.
        ranges: Intrinsic coordinate ranges for mesh sampling.
        paths: List of PathTrace objects to overlay.
        variable_values: String names for variable values (for readable labels).
        distance_matrix: (n, n) precomputed distance matrix. When provided,
            uses MDS instead of PCA for 3D projection.
    """
    # --- Data prep ---
    if param_dict is not None:
        param_names = sorted(param_dict.keys())
    elif train_dataset is not None:
        param_dict, param_names = _extract_param_values(
            train_dataset,
            intervention_variable=intervention_variable,
            embeddings=embeddings,
        )
    else:
        logger.warning("No train_dataset or param_dict provided; skipping 3D plot.")
        return

    if not param_names:
        logger.warning("No causal parameters found; skipping 3D plot.")
        return

    if feature_kind not in ("raw", "hellinger"):
        raise ValueError(
            f"feature_kind must be 'raw' or 'hellinger', got {feature_kind!r}",
        )
    hellinger_mode = feature_kind == "hellinger"
    raw_probs: Tensor | None = None
    if hellinger_mode:
        # Caller passed per-example probabilities (n, W+1). √-transform once
        # for display + PCA fit; preserve the originals for centroid math
        # (mean must happen in probability space, not √-space, to avoid the
        # Jensen drift that makes mean(√(p)) lie inside the unit sphere).
        raw_probs = features.detach().clone()
        features = torch.sqrt(raw_probs.clamp(min=0.0)).float()

    use_mds = distance_matrix is not None
    if use_mds:
        dm_np = (
            distance_matrix.numpy()
            if isinstance(distance_matrix, Tensor)
            else np.asarray(distance_matrix)
        )
        features_3d = _fit_mds(dm_np)
        project_fn = None
    else:
        features_3d, project_fn = _fit_projector(
            features, pca_components=pca_components
        )
    n_features = features_3d.shape[0]

    # Compute centroids — standardize if mean/std provided (manifold case)
    param_tensors_input = {k: torch.tensor(v) for k, v in param_dict.items()}

    has_manifold = (
        not use_mds
        and manifold_obj is not None
        and mean is not None
        and std is not None
    )

    if use_mds:
        # MDS mode: compute centroids directly in 3D embedded space
        features_3d_t = torch.from_numpy(features_3d).float()
        control_points, centroids_3d_t, metadata = compute_centroids(
            features_3d_t,
            param_tensors_input,
        )
        centroids_3d = centroids_3d_t.numpy()
    else:
        features_cpu = features[:n_features].detach().cpu().float()
        mean_cpu: Tensor | None = None
        std_cpu: Tensor | None = None
        if has_manifold:
            assert mean is not None and std is not None
            mean_cpu = mean.detach().cpu()
            std_cpu = std.detach().cpu()
            features_for_centroids = (features_cpu - mean_cpu) / (std_cpu + 1e-6)
        else:
            features_for_centroids = features_cpu

        if hellinger_mode:
            # Aggregate raw probabilities per class, then √: gives proper
            # Hellinger centroids √(mean(p)) on the unit sphere. Standardization
            # doesn't apply (sphere_project manifolds use mean=0, std=1).
            assert raw_probs is not None
            raw_probs_cpu = raw_probs[:n_features].detach().cpu().float()
            control_points, mean_p_per_class, metadata = compute_centroids(
                raw_probs_cpu,
                param_tensors_input,
            )
            centroids_raw = torch.sqrt(mean_p_per_class.clamp(min=0.0))
        else:
            control_points, centroids_raw, metadata = compute_centroids(
                features_for_centroids,
                param_tensors_input,
            )

        if pre_computed_centroids_3d is not None:
            centroids_3d = np.asarray(pre_computed_centroids_3d)
        elif has_manifold and not hellinger_mode:
            assert (
                mean_cpu is not None and std_cpu is not None and project_fn is not None
            )
            centroids_ambient = centroids_raw * (std_cpu + 1e-6) + mean_cpu
            centroids_3d = project_fn(centroids_ambient.detach().cpu().numpy())
        else:
            assert project_fn is not None
            centroids_3d = project_fn(centroids_raw.detach().cpu().numpy())

    centroid_param_names = metadata["parameter_names"]
    centroid_param_values = control_points.detach().cpu().numpy()

    # Hover text
    centroid_hover = _build_centroid_hover(
        centroid_param_names,
        centroid_param_values,
        metadata["counts"],
        variable_values=variable_values,
        embeddings=embeddings,
        intervention_variable=intervention_variable,
    )
    feature_hover = _build_readable_hover(
        param_dict,
        param_names,
        n_features,
        variable_values=variable_values,
        embeddings=embeddings,
        intervention_variable=intervention_variable,
        train_dataset=train_dataset,
        detailed_hover=detailed_hover,
        max_hover_chars=max_hover_chars,
    )

    # --- Always-visible traces ---
    always_visible: list = []

    # Manifold mesh
    if has_manifold and ranges is not None:
        assert mean is not None and std is not None and project_fn is not None
        mesh_traces = _build_manifold_mesh_traces(
            manifold_obj,
            mean,
            std,
            ranges,
            project_fn,
            param_names=param_names,
        )
        always_visible.extend(mesh_traces)

    # Edges
    remapped_edges: EdgeList | None = None
    if edges is not None and edge_node_coords is not None:
        coord_to_centroid: dict[tuple, int] = {}
        for ci in range(centroids_3d.shape[0]):
            key = tuple(
                round(float(centroid_param_values[ci, j]), 6)
                for j in range(len(centroid_param_names))
            )
            coord_to_centroid[key] = ci
        node_to_centroid: dict[int, int] = {}
        for node_id, coords in edge_node_coords.items():
            key = tuple(
                round(float(coords[pname]), 6) for pname in centroid_param_names
            )
            ci = coord_to_centroid.get(key)
            if ci is not None:
                node_to_centroid[node_id] = ci
        remapped_edges = []
        for i, j in edges:
            ci = node_to_centroid.get(i)
            cj = node_to_centroid.get(j)
            if ci is not None and cj is not None:
                remapped_edges.append((ci, cj))

    draw_edges = remapped_edges or edges
    if draw_edges is not None:
        always_visible.extend(_make_edge_traces(centroids_3d, draw_edges))

    # Path traces
    if paths:
        for pt in paths:
            if pt.is_intrinsic:
                if not has_manifold:
                    logger.warning(
                        "Skipping intrinsic path '%s' (no manifold)", pt.name
                    )
                    continue
                assert mean is not None and std is not None and project_fn is not None
                device = _get_manifold_device(manifold_obj)
                mean_d = mean.to(device)
                std_d = std.to(device)
                with torch.no_grad():
                    decoded = manifold_obj.decode(pt.points.to(device), r=None)
                    decoded = decoded * (std_d + 1e-6) + mean_d
                pts_3d = project_fn(decoded.cpu().numpy())
            else:
                assert project_fn is not None
                pts_3d = project_fn(pt.points.detach().cpu().numpy())
            line_kwargs = dict(color=pt.color, width=pt.width)
            if pt.dash is not None:
                line_kwargs["dash"] = pt.dash
            always_visible.append(
                go.Scatter3d(
                    x=pts_3d[:, 0],
                    y=pts_3d[:, 1],
                    z=pts_3d[:, 2],
                    mode=pt.mode,
                    line=line_kwargs,
                    marker=dict(size=pt.marker_size, color=pt.color),
                    name=pt.name,
                )
            )

    # --- Dropdown trace groups ---
    scales = ["linear", "log", "categorical"]
    trace_groups: list[tuple[str, list]] = []

    # Normalize colormap to list for per-param indexing
    if isinstance(colormap, list):
        cmap_list = colormap
    elif colormap is not None:
        cmap_list = [colormap]
    else:
        cmap_list = [None]

    if param_names:
        for pi, pname in enumerate(param_names):
            cmap = cmap_list[pi] if pi < len(cmap_list) else cmap_list[-1]
            vals = param_dict[pname]
            vals_aligned = (
                vals[:n_features]
                if len(vals) >= n_features
                else np.pad(vals, (0, n_features - len(vals)), constant_values=np.nan)
            )

            if pname in centroid_param_names:
                cp_idx = centroid_param_names.index(pname)
                centroid_vals = centroid_param_values[:, cp_idx]
            else:
                centroid_vals = np.full(centroids_3d.shape[0], np.nan)

            for scale in scales:
                label = f"{pname} \u00b7 {scale}"
                traces = _make_trace_group_scatter(
                    features_3d,
                    vals_aligned,
                    pname,
                    scale,
                    hover_extra=feature_hover,
                    size=3,
                    opacity=0.15,
                    colormap=cmap,
                )
                if centroid_color is not None:
                    centroid_traces = [
                        go.Scatter3d(
                            x=centroids_3d[:, 0],
                            y=centroids_3d[:, 1],
                            z=centroids_3d[:, 2],
                            mode="markers",
                            marker=dict(
                                size=8,
                                color=centroid_color,
                                opacity=1.0,
                                symbol="diamond",
                            ),
                            name=f"{pname} (centroids)",
                            hovertext=centroid_hover,
                            hoverinfo="text",
                            visible=False,
                        )
                    ]
                else:
                    centroid_traces = _make_trace_group_scatter(
                        centroids_3d,
                        centroid_vals,
                        pname,
                        scale,
                        hover_extra=centroid_hover,
                        size=8,
                        opacity=1.0,
                        symbol="diamond",
                        show_colorbar=False,
                        colormap=cmap,
                    )
                for ct in centroid_traces:
                    ct.name = f"{ct.name} (centroids)"
                traces.extend(centroid_traces)
                trace_groups.append((label, traces))
    else:
        traces = [
            go.Scatter3d(
                x=features_3d[:, 0],
                y=features_3d[:, 1],
                z=features_3d[:, 2],
                mode="markers",
                marker=dict(size=3, color="steelblue", opacity=0.15),
                name="Features",
                visible=False,
            )
        ]
        trace_groups.append(("features", traces))

    axis_labels: tuple[str, str, str]
    if features.shape[1] > 3:
        idx = list(pca_components) if pca_components is not None else [0, 1, 2]
        axis_labels = (f"PC {idx[0]}", f"PC {idx[1]}", f"PC {idx[2]}")
    else:
        axis_labels = ("Dim 0", "Dim 1", "Dim 2")
    _assemble_dropdown_figure(
        always_visible, trace_groups, title, axis_labels, output_path
    )


# ---------------------------------------------------------------------------
# Public API — thin wrappers for backward compatibility
# ---------------------------------------------------------------------------


def _prepare_features_data(  # pyright: ignore[reportUnusedFunction]
    features: Tensor,
    n_components: int,
    train_dataset: list | None = None,
    intervention_variable: str | None = None,
    param_dict: dict[str, NDArray] | None = None,
    edges: EdgeList | None = None,
    edge_node_coords: dict[int, dict[str, float]] | None = None,
    embeddings: dict | None = None,
) -> dict[str, Any] | None:
    """Shared data prep for plot_features_3d and plot_features_2d.

    Args:
        edge_node_coords: Optional mapping from node ID to dict of
            param_name -> coordinate value. Used to remap edge node IDs
            to centroid indices.
        embeddings: Optional dict mapping variable names to embedding functions.

    Returns a dict with all computed data, or None if no params found.
    """
    if param_dict is not None:
        param_names = sorted(param_dict.keys())
    elif train_dataset is not None:
        param_dict, param_names = _extract_param_values(
            train_dataset,
            intervention_variable=intervention_variable,
            embeddings=embeddings,
        )
    else:
        logger.warning("No train_dataset or param_dict provided; skipping plot.")
        return None
    if not param_names:
        logger.warning("No causal parameters found; skipping plot.")
        return None

    projected, project_fn = _fit_projector(features, n_components=n_components)
    n_features = projected.shape[0]

    # Compute centroids
    param_tensors_input = {k: torch.tensor(v) for k, v in param_dict.items()}
    features_cpu = features[:n_features].detach().cpu().float()
    control_points, centroids, metadata = compute_centroids(
        features_cpu, param_tensors_input
    )
    centroids_proj = project_fn(centroids.detach().cpu().numpy())

    centroid_param_names = metadata["parameter_names"]
    centroid_param_values = control_points.detach().cpu().numpy()
    centroid_counts = metadata["counts"]

    # Hover text for centroids
    centroid_hover = []
    for ci in range(centroids_proj.shape[0]):
        parts = [
            f"{centroid_param_names[j]}={centroid_param_values[ci, j]:.4g}"
            for j in range(len(centroid_param_names))
        ]
        parts.append(f"n={centroid_counts[ci]}")
        centroid_hover.append("<br>".join(parts))

    # Hover text for features
    feature_hover = []
    for fi in range(n_features):
        parts = []
        for pname in param_names:
            vals = param_dict[pname]
            if fi < len(vals):
                parts.append(f"{pname}={vals[fi]:.4g}")
        feature_hover.append("<br>".join(parts))

    # Remap edges from graph node IDs to centroid indices.
    # Edges use node IDs, but centroids are reordered by torch.unique.
    # If edge_node_coords is provided, use it to map node IDs to centroids
    # via coordinate matching.
    remapped_edges: EdgeList | None = None
    if edges is not None and edge_node_coords is not None:
        # Build coordinate tuple -> centroid index
        coord_to_centroid: dict[tuple, int] = {}
        for ci in range(centroids_proj.shape[0]):
            key = tuple(
                round(float(centroid_param_values[ci, j]), 6)
                for j in range(len(centroid_param_names))
            )
            coord_to_centroid[key] = ci

        # Map node ID -> centroid index via coordinates
        node_to_centroid: dict[int, int] = {}
        for node_id, coords in edge_node_coords.items():
            key = tuple(
                round(float(coords[pname]), 6) for pname in centroid_param_names
            )
            ci = coord_to_centroid.get(key)
            if ci is not None:
                node_to_centroid[node_id] = ci

        remapped_edges = []
        for i, j in edges:
            ci = node_to_centroid.get(i)
            cj = node_to_centroid.get(j)
            if ci is not None and cj is not None:
                remapped_edges.append((ci, cj))

    return {
        "projected": projected,
        "centroids_proj": centroids_proj,
        "n_features": n_features,
        "param_names": param_names,
        "param_dict": param_dict,
        "centroid_param_names": centroid_param_names,
        "centroid_param_values": centroid_param_values,
        "centroid_counts": centroid_counts,
        "centroid_hover": centroid_hover,
        "feature_hover": feature_hover,
        "project_fn": project_fn,
        "remapped_edges": remapped_edges,
    }


def plot_features_3d(
    features: Tensor,
    output_path: str,
    train_dataset: list | None = None,
    title: str = "Features 3D",
    intervention_variable: str | None = None,
    edges: EdgeList | None = None,
    param_dict: dict[str, NDArray] | None = None,
    edge_node_coords: dict[int, dict[str, float]] | None = None,
    embeddings: dict | None = None,
    colormap: str | None = None,
) -> None:
    """Plot training features + centroids colored by causal parameter (3D).

    Thin wrapper around [`plot_3d`][] — see its docstring for details.
    """
    plot_3d(
        features=features,
        output_path=output_path,
        train_dataset=train_dataset,
        title=title,
        intervention_variable=intervention_variable,
        edges=edges,
        edge_node_coords=edge_node_coords,
        embeddings=embeddings,
        colormap=colormap,
        param_dict=param_dict,
    )


def plot_manifold_3d(
    features: Tensor,
    train_dataset: list,
    manifold_obj: Any,
    mean: Tensor,
    std: Tensor,
    ranges: tuple,
    output_path: str,
    edge_node_coords: dict[int, dict[str, float]] | None = None,
    title: str = "Manifold 3D",
    intervention_variable: str | None = None,
    edges: EdgeList | None = None,
    embeddings: dict | None = None,
    colormap: str | None = None,
) -> None:
    """Plot features + manifold mesh surface colored by causal parameter.

    Thin wrapper around [`plot_3d`][] — see its docstring for details.
    """
    plot_3d(
        features=features,
        output_path=output_path,
        train_dataset=train_dataset,
        title=title,
        intervention_variable=intervention_variable,
        edges=edges,
        edge_node_coords=edge_node_coords,
        embeddings=embeddings,
        colormap=colormap,
        manifold_obj=manifold_obj,
        mean=mean,
        std=std,
        ranges=ranges,
    )


def plot_steering_results(
    steer_result: dict,
    manifold_obj: Any,
    mean: Tensor,
    std: Tensor,
    ranges: tuple,
    dist_module: Any,
    config: Any,
    features: Tensor,
    output_path: str,
    title: str = "Steering Results",
) -> None:
    """Plot manifold surface + steering points colored by inferred output stats.

    Args:
        steer_result: Dict from steer_manifold() with grid_points, scores, etc.
        manifold_obj: SplineManifold.
        mean: Standardization mean (k,).
        std: Standardization std (k,).
        ranges: Intrinsic ranges for surface sampling.
        dist_module: Distribution module with compute_output_stats.
        config: Distribution config.
        features: (n, k) features for PCA fitting.
        output_path: Path for the output HTML file.
        title: Plot title.
    """
    _, project_fn = _fit_projector(features)

    # Decode steering grid points to feature space then project to 3D
    grid_points = steer_result["grid_points"]
    device = _get_manifold_device(manifold_obj)
    mean_d = mean.to(device)
    std_d = std.to(device)
    grid_tensor = (
        grid_points.to(device)
        if isinstance(grid_points, Tensor)
        else torch.tensor(grid_points, device=device)
    )

    with torch.no_grad():
        decoded = manifold_obj.decode(grid_tensor, r=None)
        decoded = decoded * (std_d + 1e-6) + mean_d
    steer_3d = project_fn(decoded.cpu().numpy())
    n_grid = steer_3d.shape[0]

    # Intrinsic coords for hover
    grid_np = grid_tensor.cpu().numpy()

    # Build manifold mesh (always visible)
    mesh_traces = _build_manifold_mesh_traces(
        manifold_obj, mean, std, ranges, project_fn
    )
    always_visible = list(mesh_traces)

    # Compute output stats
    stat_dict, stat_names = _compute_grid_stats(steer_result, dist_module, config)

    if not stat_names:
        logger.warning(
            "No output stats computed; steering_results plot will have no coloring."
        )
        stat_names = ["grid_index"]
        stat_dict = {"grid_index": np.arange(n_grid, dtype=float)}

    # Build hover text for steering points
    steer_hover = []
    for gi in range(n_grid):
        parts = [f"u=[{', '.join(f'{u:.3f}' for u in grid_np[gi])}]"]
        for sn in stat_names:
            val = stat_dict[sn][gi]
            parts.append(f"{sn}={val:.4g}")
        steer_hover.append("<br>".join(parts))

    # Build trace groups
    scales = ["linear", "log", "categorical"]
    trace_groups: list[tuple[str, list]] = []

    for sname in stat_names:
        vals = stat_dict[sname]
        for scale in scales:
            label = f"{sname} \u00b7 {scale}"
            traces = _make_trace_group_scatter(
                steer_3d,
                vals,
                sname,
                scale,
                hover_extra=steer_hover,
                size=6,
                opacity=0.9,
            )
            trace_groups.append((label, traces))

    axis_labels = (
        ("PC 0", "PC 1", "PC 2")
        if features.shape[1] > 3
        else ("Dim 0", "Dim 1", "Dim 2")
    )
    _assemble_dropdown_figure(
        always_visible, trace_groups, title, axis_labels, output_path
    )
