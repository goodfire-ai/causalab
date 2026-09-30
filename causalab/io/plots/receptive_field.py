"""Receptive-field decision-map plotter (analysis-neutral).

Renders a "receptive field" over a plane (2-D) or volume (3-D) of activation
space: an evenly spaced grid of sampled points, each colored by the argmax class
of the model's output distribution at that point and (optionally) shaded by
confidence. The training point cloud, class centroids, and the geometric/linear
steering paths are drawn as overlays. A dropdown selects which pair's steering
path is shown (like the dual-manifold viewer), so the paths don't all crowd the
view at once.

Dimensionality is inferred from the coordinate arrays: pass 2-column arrays for a
2-D field, 3-column for a 3-D field (the caller decides via how many PCA columns
it projects onto). This module is free of model / analysis knowledge: every input
is a plain array the caller has **already projected into the same subspace**, so
all layers (field, cloud, centroids, paths) share one coordinate system. It
imports only numpy / plotly and sibling ``causalab.io.plots`` modules (layering
invariant 3, ``tests/test_architecture_layering.py``).
"""

from __future__ import annotations

import json
import os

import numpy as np
import plotly.graph_objects as go  # pyright: ignore[reportMissingTypeStubs]
from numpy.typing import NDArray

from .figure_format import path_with_figure_format
from .plot_utils import PLOTLY_HTML_CONFIG

_FONT_FAMILY = "Avenir, Avenir Next, Helvetica Neue, sans-serif"
_CENTROID_SIZE = 14


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _rgba(hex_color: str, alpha: float) -> str:
    r, g, b = _hex_to_rgb(hex_color)
    return f"rgba({r},{g},{b},{alpha:.3f})"


def _scatter(is3d: bool, coords: NDArray, **kw):
    """Build a go.Scatter (2-D) or go.Scatter3d (3-D) from an (n, D) array."""
    if is3d:
        return go.Scatter3d(x=coords[:, 0], y=coords[:, 1], z=coords[:, 2], **kw)
    return go.Scatter(x=coords[:, 0], y=coords[:, 1], **kw)


def _cell_colors_and_hover(
    grid_argmax, grid_confidence, encode_confidence, class_colors, class_labels
):
    """Per-cell argmax classes, rgba colors (confidence->alpha), and hover text."""
    n = len(class_labels)
    conf = np.clip(np.asarray(grid_confidence, dtype=float), 0.0, 1.0)
    cls = np.asarray(grid_argmax, dtype=int)
    alphas = (0.2 + 0.8 * conf) if encode_confidence else np.ones_like(conf)
    colors = [
        _rgba(class_colors[int(cls[k]) % n], float(alphas[k])) for k in range(len(cls))
    ]
    hover = [
        f"{class_labels[int(cls[k]) % n]}<br>conf={conf[k]:.2f}"
        for k in range(len(cls))
    ]
    return cls, colors, hover


# JS injected into the 3-D HTML (via write_html post_script). Adds an orientation
# <select> + a position slider that appears only for a slice, and recomputes the
# per-class field traces' data on change (so it composes with the native class /
# path dropdowns, which set visibility). Placeholders are filled by str.replace.
_SLICE_JS = """
(function(){
  var gd=document.getElementById('rf-plot'); if(!gd) return;
  var RF=__RFJSON__;
  var bar=document.createElement('div');
  bar.style.cssText='font-family:Avenir,Helvetica,Arial,sans-serif;font-size:13px;'+
    'color:#444;margin:4px 2px 2px;display:flex;gap:18px;align-items:center;flex-wrap:wrap;';
  bar.innerHTML='<label>grid plane: <select id="rf-orient" style="font-size:13px;padding:2px 6px;">'+
    '<option value="full">Full (PC__C0__,__C1__,__C2__)</option>'+
    '<option value="01">Slice PC__C0__-PC__C1__</option>'+
    '<option value="12">Slice PC__C1__-PC__C2__</option>'+
    '<option value="20">Slice PC__C2__-PC__C0__</option>'+
    '<option value="hide">Hide grid</option></select></label>'+
    '<span id="rf-pos-wrap" style="display:none;">position <b id="rf-pos-axis"></b> '+
    '<input type="range" id="rf-pos" min="0" max="'+(RF.R-1)+'" value="'+Math.floor(RF.R/2)+
    '" style="width:240px;vertical-align:middle;"> <span id="rf-pos-val" style="color:#666;"></span></span>';
  gd.parentNode.insertBefore(bar,gd);
  var AX={'01':{dim:2,field:8,lab:'PC__C2__'},'12':{dim:0,field:6,lab:'PC__C0__'},
          '20':{dim:1,field:7,lab:'PC__C1__'}};
  var oEl=document.getElementById('rf-orient'),pEl=document.getElementById('rf-pos'),
      wrap=document.getElementById('rf-pos-wrap'),axEl=document.getElementById('rf-pos-axis'),
      vEl=document.getElementById('rf-pos-val');
  function recompute(){
    var o=oEl.value,pos=parseInt(pEl.value),isSlice=AX.hasOwnProperty(o);
    wrap.style.display=isSlice?'':'none';
    var xs=[],ys=[],zs=[],cols=[],txts=[],ids=[];
    for(var c=0;c<RF.n;c++){xs.push([]);ys.push([]);zs.push([]);cols.push([]);txts.push([]);ids.push(RF.fieldIdx[c]);}
    for(var k=0;k<RF.cells.length;k++){
      var cell=RF.cells[k],show=false;
      if(o==='full')show=true; else if(o==='hide')show=false; else show=(cell[AX[o].field]===pos);
      if(show){var c=cell[3];xs[c].push(cell[0]);ys[c].push(cell[1]);zs[c].push(cell[2]);cols[c].push(cell[4]);txts[c].push(cell[5]);}
    }
    if(isSlice){axEl.textContent=AX[o].lab+' =';vEl.textContent=RF.ticks[AX[o].dim][pos].toFixed(2);}
    Plotly.restyle(gd,{x:xs,y:ys,z:zs,'marker.color':cols,text:txts},ids);
  }
  oEl.addEventListener('change',recompute);
  pEl.addEventListener('input',recompute);
})();
"""


def _field_slice_post_script(
    grid_xy,
    grid_argmax,
    grid_confidence,
    encode_confidence,
    class_colors,
    class_labels,
    grid_res,
    pca_components,
) -> str:
    """Build the post_script JS for the 3-D slice control (empty for non-3-D)."""
    if grid_xy.shape[1] != 3:
        return ""
    comps = [int(c) for c in pca_components]
    R = max(int(grid_res), 1)
    cls, colors, hover = _cell_colors_and_hover(
        grid_argmax, grid_confidence, encode_confidence, class_colors, class_labels
    )
    cells = []
    for k in range(grid_xy.shape[0]):
        i0, i1, i2 = k // (R * R), (k // R) % R, k % R  # k = (i0*R + i1)*R + i2
        cells.append(
            [
                round(float(grid_xy[k, 0]), 5),
                round(float(grid_xy[k, 1]), 5),
                round(float(grid_xy[k, 2]), 5),
                int(cls[k]),
                colors[k],
                hover[k],
                i0,
                i1,
                i2,
            ]
        )
    ticks = [np.unique(np.round(grid_xy[:, d], 5)).tolist() for d in range(3)]
    # fieldIdx == the per-class field trace indices. Valid because
    # build_receptive_field_figure adds those traces FIRST (asserted there:
    # field_trace_idx == range(n_classes)); the slice JS restyles them by id.
    rf = {
        "R": R,
        "n": len(class_labels),
        "fieldIdx": list(range(len(class_labels))),
        "cells": cells,
        "ticks": ticks,
    }
    js = _SLICE_JS.replace("__RFJSON__", json.dumps(rf))
    for token, val in (
        ("__C0__", comps[0]),
        ("__C1__", comps[1]),
        ("__C2__", comps[2]),
    ):
        js = js.replace(token, str(val))
    return js


def build_receptive_field_figure(
    *,
    grid_xy: NDArray,  # (G, D) grid coords in PC space, D in {2, 3}
    grid_argmax: NDArray,  # (G,) int class index per cell
    grid_confidence: NDArray,  # (G,) float, max class prob in [0, 1]
    grid_res: int,  # G == grid_res ** D
    axis_ranges: list[list[float]],  # D x [lo, hi]
    scatter_xy: NDArray | None,  # (N, D) training point cloud
    scatter_classes: NDArray | None,  # (N,) int
    centroid_xy: NDArray | None,  # (W, D)
    centroid_mask: NDArray | None,  # (W,) bool
    geo_paths_xy: list[NDArray] | None,  # per pair: (n_steps, D)
    lin_paths_xy: list[NDArray] | None,  # per pair: (n_steps, D)
    class_labels: list[str],
    class_colors: list[str],  # hex, len == n_classes
    pca_components: list[int] | tuple[int, ...] = (0, 1),
    pair_labels: list[str] | None = None,  # one per path pair, for the dropdown
    encode_confidence: bool = True,
    show_scatter: bool = True,
    show_centroids: bool = True,
    show_paths: bool = True,
    field_marker_size: float | None = None,
    title: str | None = None,
) -> go.Figure:
    """Build the receptive-field figure (no I/O). Returns the ``go.Figure``.

    Every coordinate array must already be projected to the same ``pca_components``
    columns as ``grid_xy``; this function does not project. ``D = grid_xy.shape[1]``
    selects 2-D vs 3-D rendering.
    """
    comps = [int(c) for c in pca_components]
    n_dims = grid_xy.shape[1]
    is3d = n_dims == 3
    n_classes = len(class_labels)

    cls, cell_colors, hover = _cell_colors_and_hover(
        grid_argmax, grid_confidence, encode_confidence, class_colors, class_labels
    )

    fig = go.Figure()
    path_trace_idx: list[
        tuple[int, int]
    ] = []  # (geo_idx, lin_idx) per pair; -1 if none

    # --- Field: one square-marker trace PER argmax class. Splitting by class lets
    # the slice dropdown set each trace's DATA while the class dropdown sets each
    # trace's VISIBILITY — two independent properties, so the two controls compose
    # without clobbering. Markers (not a heatmap/volume) keep the plot in data
    # coordinates and let confidence bake into per-point rgba alpha; it is also the
    # honest depiction (a discrete grid of interventions).
    if field_marker_size is None:
        field_marker_size = 5.0 if is3d else max(6.0, 340.0 / max(grid_res, 1))
    cls_cell_idx = [
        np.where(cls == c)[0] for c in range(n_classes)
    ]  # full-grid, per class
    field_trace_idx: list[int] = []
    for c in range(n_classes):
        sel = cls_cell_idx[c]
        fig.add_trace(
            _scatter(
                is3d,
                grid_xy[sel],
                mode="markers",
                marker=dict(
                    symbol="square",
                    size=field_marker_size,
                    color=[cell_colors[k] for k in sel],
                    line=dict(width=0),
                ),
                text=[hover[k] for k in sel],
                hoverinfo="text",
                showlegend=False,
                legendgroup=f"field{c}",
                name=f"field {class_labels[c]}",
            )
        )
        field_trace_idx.append(len(fig.data) - 1)  # pyright: ignore[reportArgumentType]  # plotly Figure.data is a tuple-like Sized container; stubs widen to Figure

    # The per-class field traces are added FIRST and unconditionally, so they
    # occupy the leading trace indices. The 3-D slice JS (`fieldIdx` in
    # `_field_slice_post_script`) targets them by `range(n_classes)`; enforce that
    # invariant here so any future reordering fails loudly instead of silently
    # restyling the wrong traces.
    assert field_trace_idx == list(range(n_classes)), (
        "receptive-field per-class traces must be the leading traces; the 3-D "
        "slice JS (fieldIdx) targets them by range(n_classes)"
    )

    # --- Point cloud (training activations), one trace per class. Thin black edge.
    if show_scatter and scatter_xy is not None and scatter_classes is not None:
        sc = np.asarray(scatter_classes, dtype=int)
        for c in range(n_classes):
            m = sc == c
            if not m.any():
                continue
            fig.add_trace(
                _scatter(
                    is3d,
                    scatter_xy[m],
                    mode="markers",
                    marker=dict(
                        size=(3 if is3d else 4),
                        color=class_colors[c],
                        opacity=0.55,
                        line=dict(color="black", width=0.6),
                    ),
                    name=class_labels[c],
                    legendgroup=class_labels[c],
                    hoverinfo="skip",
                )
            )

    # --- Steering paths: geometric (black, solid), linear (gray, dashed). One
    # trace per pair per family so a dropdown can show a single pair at a time.
    if show_paths and (geo_paths_xy or lin_paths_xy):
        n_pairs = len(geo_paths_xy) if geo_paths_xy else len(lin_paths_xy or [])
        for p in range(n_pairs):
            lbl = (
                pair_labels[p] if pair_labels and p < len(pair_labels) else f"pair {p}"
            )
            visible = p == 0  # default: only the first pair's path is shown
            gi = li = -1
            if geo_paths_xy and p < len(geo_paths_xy):
                fig.add_trace(
                    _scatter(
                        is3d,
                        np.asarray(geo_paths_xy[p]),
                        mode="lines",
                        line=dict(color="black", width=(4 if is3d else 3)),
                        name=f"{lbl} · geometric",
                        legendgroup=f"path{p}",
                        showlegend=False,
                        hoverinfo="skip",
                        visible=visible,
                    )
                )
                gi = len(fig.data) - 1  # pyright: ignore[reportArgumentType]  # plotly Figure.data is a tuple-like Sized container; stubs widen to Figure
            if lin_paths_xy and p < len(lin_paths_xy):
                fig.add_trace(
                    _scatter(
                        is3d,
                        np.asarray(lin_paths_xy[p]),
                        mode="lines",
                        line=dict(color="#999", width=(4 if is3d else 3), dash="dash"),
                        name=f"{lbl} · linear",
                        legendgroup=f"path{p}",
                        showlegend=False,
                        hoverinfo="skip",
                        visible=visible,
                    )
                )
                li = len(fig.data) - 1  # pyright: ignore[reportArgumentType]  # plotly Figure.data is a tuple-like Sized container; stubs widen to Figure
            path_trace_idx.append((gi, li))

    # --- Centroids (diamonds, labeled), drawn last so they sit on top.
    if show_centroids and centroid_xy is not None:
        mask = (
            np.asarray(centroid_mask, dtype=bool)
            if centroid_mask is not None
            else np.ones(len(centroid_xy), dtype=bool)
        )
        _cent_first = True  # one shared "centroids" legend entry toggles them all
        for c in range(min(n_classes, len(centroid_xy))):
            if not mask[c]:
                continue
            fig.add_trace(
                _scatter(
                    is3d,
                    centroid_xy[c : c + 1],
                    mode="markers+text",
                    marker=dict(
                        size=_CENTROID_SIZE,
                        color=class_colors[c],
                        symbol="diamond",
                        line=dict(color="black", width=1.5),
                    ),
                    text=[class_labels[c]],
                    textposition="top center",
                    textfont=dict(size=11, color="black", family=_FONT_FAMILY),
                    legendgroup="centroids",
                    name="centroids",
                    showlegend=_cent_first,
                    hovertemplate=f"<b>{class_labels[c]}</b><extra></extra>",
                )
            )
            _cent_first = False

    # --- Interactive controls (native plotly updatemenus; no custom JS) ----------
    # Cloud (per class) and centroids are legend entries -> click to hide/show. The
    # dropdowns restyle ONLY their own traces (targeted by index), so they never
    # clobber legend-hidden elements or each other. The slice (sets field DATA) and
    # class (sets field VISIBILITY) dropdowns touch different properties of the
    # per-class field traces, so they compose.
    updatemenus = []
    menu_rows: list[tuple[str, dict]] = []
    _menu_style = dict(
        direction="down",
        showactive=True,
        bgcolor="white",
        bordercolor="#ccc",
        font=dict(family=_FONT_FAMILY, size=12),
    )

    # NOTE: in 3-D the field-view / slice orientation + position slider are added by
    # the JS layer (write_html post_script); see _field_slice_post_script. They set
    # the per-class field traces' DATA, composing with the class dropdown (which
    # sets their VISIBILITY).

    # Probe-class dropdown: all classes, or only one — the fraction of the field
    # assigned to that class. Sets VISIBILITY of the per-class field traces.
    class_buttons = [
        dict(
            label="All classes",
            method="restyle",
            args=[{"visible": [True] * n_classes}, field_trace_idx],
        )
    ]
    for c in range(n_classes):
        class_buttons.append(
            dict(
                label=f"Class {class_labels[c]}",
                method="restyle",
                args=[{"visible": [i == c for i in range(n_classes)]}, field_trace_idx],
            )
        )
    menu_rows.append(("class", dict(buttons=class_buttons, active=0, **_menu_style)))

    # Path-pair dropdown: isolate one pair's path (or All / None). Restyles ONLY the
    # path traces, so legend-hidden cloud/centroids stay hidden.
    if path_trace_idx:
        path_ids = [i for pr in path_trace_idx for i in pr if i >= 0]

        def _pvis(selected: set[int]) -> list[bool]:
            return [i in selected for i in path_ids]

        pbuttons = []
        for p, (gi, li) in enumerate(path_trace_idx):
            lbl = (
                pair_labels[p] if pair_labels and p < len(pair_labels) else f"pair {p}"
            )
            sel = {i for i in (gi, li) if i >= 0}
            pbuttons.append(
                dict(
                    label=lbl,
                    method="restyle",
                    args=[{"visible": _pvis(sel)}, path_ids],
                )
            )
        pbuttons.append(
            dict(
                label="All paths",
                method="restyle",
                args=[{"visible": _pvis(set(path_ids))}, path_ids],
            )
        )
        pbuttons.append(
            dict(
                label="No paths",
                method="restyle",
                args=[{"visible": _pvis(set())}, path_ids],
            )
        )
        menu_rows.append(("path", dict(buttons=pbuttons, active=0, **_menu_style)))

    # Lay the dropdowns out as labeled rows stacked at the top-left.
    control_annotations = []
    _y0 = 1.16 if is3d else 1.12
    for _r, (_lab, _menu) in enumerate(menu_rows):
        _yy = _y0 - 0.08 * _r
        _menu.update(x=0.09, xanchor="left", y=_yy, yanchor="top")
        updatemenus.append(_menu)
        control_annotations.append(
            dict(
                x=0.075,
                y=_yy - 0.012,
                xref="paper",
                yref="paper",
                xanchor="right",
                yanchor="top",
                text=_lab,
                showarrow=False,
                font=dict(size=11, family=_FONT_FAMILY, color="#666"),
            )
        )

    ttl = title or f"Receptive field — top-{n_dims} activation PCs"
    if is3d:
        c0, c1, c2 = comps[0], comps[1], comps[2]
        (lo0, hi0), (lo1, hi1), (lo2, hi2) = axis_ranges
        fig.update_layout(
            title=dict(
                text=ttl,
                font=dict(size=15, family=_FONT_FAMILY, color="#333"),
                x=0.5,
                xanchor="center",
            ),
            scene=dict(
                xaxis=dict(title=f"PC{c0}", range=[lo0, hi0]),
                yaxis=dict(title=f"PC{c1}", range=[lo1, hi1]),
                zaxis=dict(title=f"PC{c2}", range=[lo2, hi2]),
                aspectmode="data",
            ),
            font=dict(family=_FONT_FAMILY),
            margin=dict(l=0, r=0, t=105, b=0),
            updatemenus=updatemenus,
            annotations=control_annotations,
            legend=dict(
                font=dict(size=11), itemsizing="constant", groupclick="togglegroup"
            ),
        )
    else:
        c0, c1 = comps[0], comps[1]
        (lo0, hi0), (lo1, hi1) = axis_ranges
        fig.update_layout(
            title=dict(
                text=ttl,
                font=dict(size=15, family=_FONT_FAMILY, color="#333"),
                x=0.5,
                xanchor="center",
            ),
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=dict(family=_FONT_FAMILY),
            xaxis=dict(title=f"PC{c0}", range=[lo0, hi0], zeroline=False),
            yaxis=dict(
                title=f"PC{c1}",
                range=[lo1, hi1],
                zeroline=False,
                scaleanchor="x",
                scaleratio=1,
            ),
            margin=dict(l=50, r=20, t=85, b=45),
            updatemenus=updatemenus,
            annotations=control_annotations,
            legend=dict(
                font=dict(size=11), itemsizing="constant", groupclick="togglegroup"
            ),
        )
    return fig


def plot_receptive_field(
    *,
    output_path: str = "receptive_field.html",
    figure_format: str | None = None,
    **kwargs,
) -> None:
    """Render the receptive-field decision map to ``output_path`` (HTML).

    Thin I/O wrapper over [`build_receptive_field_figure`][]; all field/overlay
    arguments are forwarded as keywords. Supports 2-D and 3-D (inferred from the
    coordinate arrays). When ``figure_format`` is ``"png"``/``"pdf"`` a static twin
    is written alongside the HTML (best-effort — needs kaleido). The HTML embeds
    plotly.js so it renders offline.
    """
    fig = build_receptive_field_figure(**kwargs)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # 3-D: inject the orientation <select> + position slider (only shown for a
    # slice). The figure div is given a fixed id the injected JS targets.
    grid_xy = np.asarray(kwargs["grid_xy"])
    post = ""
    if grid_xy.shape[1] == 3:
        post = _field_slice_post_script(
            grid_xy,
            kwargs["grid_argmax"],
            kwargs["grid_confidence"],
            kwargs.get("encode_confidence", True),
            kwargs["class_colors"],
            kwargs["class_labels"],
            kwargs["grid_res"],
            kwargs.get("pca_components", (0, 1, 2)),
        )
    fig.write_html(
        output_path,
        config=PLOTLY_HTML_CONFIG,
        div_id="rf-plot",
        post_script=(post or None),
    )

    if figure_format and figure_format.lower() in ("png", "pdf"):
        static_path = path_with_figure_format(
            os.path.splitext(output_path)[0], figure_format
        )
        try:
            fig.write_image(static_path, scale=2)
        except Exception:
            # Static export needs kaleido; the HTML is the primary artifact.
            pass
