#!/usr/bin/env python3
import logging
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.colors import LinearSegmentedColormap

logger = logging.getLogger(__name__)


# Shared config for plotly HTMLs: high-resolution PNG export from the
# camera button (default plotly scale=1 is too low for figures).
PLOTLY_HTML_CONFIG: dict = {
    "toImageButtonOptions": {
        "format": "png",
        "scale": 6,
    },
}


def _register_custom_colormaps() -> None:
    """Register project-wide custom colormaps once at import time."""
    if "dark_seismic" not in matplotlib.colormaps:
        matplotlib.colormaps.register(
            LinearSegmentedColormap.from_list(
                "dark_seismic",
                [
                    (0.0, "#5090d0"),  # brighter blue
                    (0.35, "#1a3a6c"),  # dark blue
                    (0.5, "#111111"),  # dark center
                    (0.65, "#6c1a1a"),  # dark red
                    (1.0, "#d05050"),  # brighter red
                ],
            ),
        )


_register_custom_colormaps()


def resolve_task_colormap(task_cfg, default=None):
    """Return a colormap name, optionally truncated to ``task.colormap_range``.

    If ``task_cfg.colormap_range = [lo, hi]`` is set, a truncated copy of the
    base colormap is registered under a derived name (idempotent across calls)
    and that name is returned. The returned string is usable everywhere
    matplotlib's registry is consulted — ``plt.get_cmap(name)``,
    ``matplotlib.colormaps[name]``, and downstream plotly bridges that look up
    by name.
    """
    name = task_cfg.get("colormap", default) if task_cfg is not None else default
    rng = task_cfg.get("colormap_range", None) if task_cfg is not None else None
    if name is None or rng is None:
        return name
    lo, hi = float(rng[0]), float(rng[1])
    derived = f"{name}__{lo:.3f}_{hi:.3f}"
    if derived not in matplotlib.colormaps:
        base = matplotlib.colormaps[name]
        truncated = LinearSegmentedColormap.from_list(
            derived,
            base(np.linspace(lo, hi, 256)),
        )
        matplotlib.colormaps.register(truncated, name=derived)
    return derived


######### FigureGenerator class for clean plotting #########


class FigureGenerator:
    """Centralized figure generation with consistent styling and configuration."""

    def __init__(self):
        self.colors = {
            "blue": "#699AEF",
            "red": "#DF4A44",
            "offwhite": "#f5f5f5",
            "paleblue": "#b9e0fa",
            "lightgray": "#cccacc",
            "pink": "#eb9bdf",
            "green": "#abe0a4",
        }
        self.blue_to_offwhite_to_red = LinearSegmentedColormap.from_list(
            "",
            [self.colors["blue"], self.colors["offwhite"], self.colors["red"]],
        )

        self.font_sizes = {
            "title": 30,
            "axis_label": 30,
            "tick": 22,
            "legend": 16,
            "sideplot_title": 25,
            "sideplot_axis_label": 16,
            "sideplot_tick": 14,
            "sideplot_legend": 15,
            "sideplot_colorbar": 15,
        }

        self._setup_matplotlib_defaults()

    def _setup_matplotlib_defaults(self):
        """Configure global matplotlib settings."""
        plt.rcParams["axes.spines.right"] = False
        plt.rcParams["axes.spines.top"] = False
        plt.rcParams["mathtext.fontset"] = "cm"
        plt.rcParams["pdf.fonttype"] = 42
        plt.rcParams["ps.fonttype"] = 42

        # Font setup — prefer Avenir Light, fall back to any Avenir, then sans-serif
        all_fonts = fm.findSystemFonts(fontext="ttf") + fm.findSystemFonts(
            fontext="ttc"
        )

        # 1. Look for a standalone Avenir Light font file
        light_fonts = [f for f in all_fonts if "Avenir" in f and "Light" in f]
        if light_fonts:
            fm.fontManager.addfont(light_fonts[0])
            plt.rcParams["font.family"] = fm.FontProperties(
                fname=light_fonts[0]
            ).get_name()
        else:
            # 2. Try extracting Light from an Avenir .ttc collection
            avenir_ttcs = [
                f
                for f in all_fonts
                if f.endswith(".ttc")
                and "Avenir" in os.path.basename(f)
                and "Condensed" not in f
                and "Next" not in f
            ]
            extracted = False
            if avenir_ttcs:
                try:
                    from fontTools.ttLib import TTCollection  # pyright: ignore[reportMissingTypeStubs]

                    ttc = TTCollection(avenir_ttcs[0])
                    for i, face in enumerate(ttc):  # pyright: ignore[reportUnusedVariable]
                        family = face["name"].getDebugName(1) or ""
                        subfamily = face["name"].getDebugName(2) or ""
                        is_light = "Light" in family or "Light" in subfamily
                        is_oblique = "Oblique" in family or "Oblique" in subfamily
                        if is_light and not is_oblique:
                            # Patch family name so matplotlib sees it as distinct
                            for record in face["name"].names:
                                if record.nameID in (1, 16):
                                    record.string = "Avenir Light"
                            cache_dir = os.path.expanduser("~/.cache/matplotlib-fonts")
                            os.makedirs(cache_dir, exist_ok=True)
                            ttf_path = os.path.join(cache_dir, "AvenirLight.ttf")
                            face.save(ttf_path)
                            fm.fontManager.addfont(ttf_path)
                            plt.rcParams["font.family"] = "Avenir Light"
                            extracted = True
                            break
                except (ImportError, Exception):
                    pass

            if not extracted:
                # 3. Fall back to any Avenir
                avenir_any = [f for f in all_fonts if "Avenir" in os.path.basename(f)]
                if avenir_any:
                    fm.fontManager.addfont(avenir_any[0])
                    plt.rcParams["font.family"] = fm.FontProperties(
                        fname=avenir_any[0]
                    ).get_name()
                else:
                    plt.rcParams["font.family"] = "sans-serif"

    def style_axes(
        self,
        ax,
        show_spines=True,
        spine_width=3,
        tick_size=28,
        tick_length=6,
        tick_width=2,
        tick_pad=10,
        remove_minor_ticks_x=True,
        add_arrows=False,
    ):
        """Apply consistent styling to axes."""
        if show_spines:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_linewidth(spine_width)
            ax.spines["bottom"].set_linewidth(spine_width)
            ax.spines["left"].set_zorder(10)
            ax.spines["bottom"].set_zorder(10)
        else:
            for spine in ax.spines.values():
                spine.set_visible(False)

        ax.tick_params(
            axis="both",
            which="major",
            labelsize=tick_size,
            length=tick_length,
            width=tick_width,
            pad=tick_pad,
        )

        if remove_minor_ticks_x:
            ax.tick_params(axis="x", which="minor", bottom=False)

    # Shadow defaults
    shadow_opacity = 0.15
    shadow_color = "#666666"

    def style_plotly_3d(
        self, fig, floor_color="#e5e5e5", show_axes=False, floor_shadow=True
    ):
        """Apply consistent styling to a Plotly 3D figure.

        Shows only a grey floor plane (xy), hides wall panels and grid lines.

        Args:
            floor_color (str): Color of the floor (xy) plane.
            show_axes (bool): If False (default), hide axis labels, ticks, and titles.
            floor_shadow (bool): If True, project visible scatter points onto the floor.
        """
        axis_common = dict(
            showgrid=False,
            zeroline=False,
            showline=False,
            backgroundcolor="white",
            showticklabels=show_axes,
            title="" if not show_axes else None,
        )
        fig.update_layout(
            scene=dict(
                xaxis=axis_common,
                yaxis=axis_common,
                zaxis={**axis_common, "backgroundcolor": floor_color},
                xaxis_showbackground=False,
                yaxis_showbackground=False,
                zaxis_showbackground=True,
                aspectmode="data",
            ),
            paper_bgcolor="white",
        )

        if floor_shadow:
            self._add_floor_shadows(fig)

        return fig

    def _get_z_floor(self, fig):
        """Get the z_min across all Scatter3d traces in the figure."""
        import plotly.graph_objects as go  # pyright: ignore[reportMissingTypeStubs]

        z_vals = []
        for trace in fig.data:
            if isinstance(trace, go.Scatter3d) and trace.z is not None:
                z_vals.extend([v for v in trace.z if v is not None])
        return min(z_vals) if z_vals else None

    def _add_floor_shadows(self, fig, only_indices=None, z_floor=None):
        """Add floor shadow traces for Scatter3d traces in the figure.

        Args:
            fig: Plotly figure.
            only_indices: If provided, only shadow traces at these indices.
                Otherwise shadows all Scatter3d traces currently in fig.
            z_floor: Fixed floor z value. If None, computed from ALL traces.
        """
        import plotly.graph_objects as go  # pyright: ignore[reportMissingTypeStubs]

        if z_floor is None:
            z_floor = self._get_z_floor(fig)
        if z_floor is None:
            return

        traces_to_shadow = []
        for i, trace in enumerate(fig.data):
            if only_indices is not None and i not in only_indices:
                continue
            if not isinstance(trace, go.Scatter3d):
                continue
            if trace.z is None or len(trace.z) == 0:  # pyright: ignore[reportArgumentType]  # plotly Scatter3d.z is a tuple/array Sized container; stubs widen type
                continue
            traces_to_shadow.append(trace)

        shadow_traces = []
        for trace in traces_to_shadow:
            z_proj = [z_floor] * len(trace.z)  # pyright: ignore[reportArgumentType]  # filtered above by `trace.z is not None`

            if trace.mode and "lines" in trace.mode:
                shadow_traces.append(
                    go.Scatter3d(
                        x=trace.x,
                        y=trace.y,
                        z=z_proj,
                        mode="lines",
                        line=dict(
                            color=self.shadow_color,
                            width=max((trace.line.width or 3) * 0.5, 1),
                        ),
                        opacity=self.shadow_opacity,
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )
            if trace.mode and "markers" in trace.mode:
                m = trace.marker or {}
                symbol = m.symbol if m.symbol is not None else "circle"  # pyright: ignore[reportAttributeAccessIssue]  # plotly Marker dynamic attribute typed as unknown
                # Only shadow non-circle markers (centroids, etc.)
                if symbol == "circle":
                    continue
                size = m.size if m.size is not None else 2  # pyright: ignore[reportAttributeAccessIssue]  # plotly Marker dynamic attribute typed as unknown
                shadow_traces.append(
                    go.Scatter3d(
                        x=trace.x,
                        y=trace.y,
                        z=z_proj,
                        mode="markers",
                        marker=dict(
                            size=size,
                            color=self.shadow_color,
                            symbol=symbol,
                            opacity=self.shadow_opacity,
                        ),
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )

        for st in shadow_traces:
            fig.add_trace(st)

    @staticmethod
    def glow_scatter3d(trace, glow_scale=1.2, glow_alpha=0.1):
        """Add a glow effect to a Plotly Scatter3d trace.

        Returns a list of two traces: [glow (larger, transparent), inner (solid)].
        Both have marker borders removed. The glow trace inherits color/symbol
        but is larger and more transparent, creating a soft halo.
        """
        import plotly.graph_objects as go  # pyright: ignore[reportMissingTypeStubs]

        if not isinstance(trace, go.Scatter3d):
            return [trace]
        if trace.mode and "markers" not in trace.mode:
            return [trace]

        # Read marker properties (Plotly Marker object, not a dict)
        m = trace.marker
        inner_size = m.size if m.size is not None else 3  # pyright: ignore[reportAttributeAccessIssue, reportOptionalMemberAccess]  # plotly Marker dynamic attribute typed as unknown
        color = m.color if m.color is not None else "steelblue"  # pyright: ignore[reportAttributeAccessIssue, reportOptionalMemberAccess]  # plotly Marker dynamic attribute typed as unknown

        # Remove border on the inner trace
        trace.marker.line = dict(width=0)  # pyright: ignore[reportAttributeAccessIssue, reportOptionalMemberAccess]  # plotly Marker dynamic attribute typed as unknown

        # Build glow trace: larger, transparent, no border, no legend/hover
        glow_marker = dict(
            size=inner_size * glow_scale,
            color=color,
            symbol=m.symbol if m.symbol is not None else "circle",  # pyright: ignore[reportAttributeAccessIssue, reportOptionalMemberAccess]  # plotly Marker dynamic attribute typed as unknown
            opacity=glow_alpha,
            line=dict(width=0),
        )
        # Pass through colorscale for continuous-color markers
        if m.colorscale is not None:  # pyright: ignore[reportAttributeAccessIssue, reportOptionalMemberAccess]  # plotly Marker dynamic attribute typed as unknown
            glow_marker["colorscale"] = m.colorscale  # pyright: ignore[reportAttributeAccessIssue, reportOptionalMemberAccess]  # plotly Marker dynamic attribute typed as unknown
        if m.cmin is not None:  # pyright: ignore[reportAttributeAccessIssue, reportOptionalMemberAccess]  # plotly Marker dynamic attribute typed as unknown
            glow_marker["cmin"] = m.cmin  # pyright: ignore[reportAttributeAccessIssue, reportOptionalMemberAccess]  # plotly Marker dynamic attribute typed as unknown
        if m.cmax is not None:  # pyright: ignore[reportAttributeAccessIssue, reportOptionalMemberAccess]  # plotly Marker dynamic attribute typed as unknown
            glow_marker["cmax"] = m.cmax  # pyright: ignore[reportAttributeAccessIssue, reportOptionalMemberAccess]  # plotly Marker dynamic attribute typed as unknown

        glow = go.Scatter3d(
            x=trace.x,
            y=trace.y,
            z=trace.z,
            mode="markers",
            marker=glow_marker,
            showlegend=False,
            hoverinfo="skip",
            visible=trace.visible,
        )

        return [glow, trace]

    def save_figure(self, fig, filepath, format=None, dpi=None, extra_artists=None):
        """Save figure with publication-quality settings and selective rasterization.

        If ``format`` is None, it is inferred from ``filepath`` (``.png`` vs ``.pdf``).
        If ``dpi`` is None, uses 200 for PNG and 1000 for PDF.
        ``extra_artists`` are additionally included in the ``bbox_inches='tight'``
        bounding-box calculation (useful for artists with ``set_in_layout(False)``).
        """
        suffix = os.path.splitext(filepath)[1].lower()
        if format is None:
            format = "pdf" if suffix == ".pdf" else "png"
        if dpi is None:
            dpi = 200 if format == "png" else 1000
        for ax in fig.get_axes():
            for collection in ax.collections:
                if collection.__class__.__name__ in ["QuadMesh", "AxesImage"]:
                    collection.set_rasterized(True)

                if hasattr(collection, "_offsets") and hasattr(
                    collection._offsets, "shape"
                ):
                    if collection._offsets.shape[0] > 500:
                        collection.set_rasterized(True)
                        logger.debug(
                            "Rasterized %d points", collection._offsets.shape[0]
                        )

        fig.savefig(
            filepath,
            bbox_inches="tight",
            dpi=dpi,
            format=format,
            bbox_extra_artists=extra_artists,
        )
        logger.info("Saved figure to %s", filepath)
