"""Draw a desiderata-based masking (DBM) result as one static figure.

[`plot_dbm`][] draws two panels. The top panel is the sweep curve: the score
of each fitted mask against the number of units it keeps. The bottom panel
is one mask: a layer by head matrix, a grid of neurons, or one tile per
attention and MLP output of each layer. A ring marks the point on the curve
whose mask the bottom panel draws.

The colours, marks and axis follow the interactive DBM viewer of the
encyclopedia report template (its `mountDbmViewer`: `renderChart`,
`renderNeuronViewer` and `renderHeadViewer`, and the `.dbm-*` rules of its
style sheet). That viewer reads the
`causalab.analysis.export_dbm` schema; this module takes a smaller typed
input, so that a page can draw a mask the exporter does not describe, such
as one gate per component output or a count over several masks.

Matplotlib loads when a figure is drawn, so importing this module stays
cheap.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

__all__ = [
    "ComponentMask",
    "DbmLayer",
    "HeadMask",
    "NeuronMask",
    "SweepPoint",
    "plot_dbm",
]

# The encyclopedia viewer's colours, by its CSS and render functions.
#: Kept and dropped heads of a full-attention layer (`renderHeadViewer`).
ATTENTION = ("#3267a8", "#e6eef9")
#: Kept and dropped heads of a Gated DeltaNet layer (`renderHeadViewer`).
DELTA_NET = ("#b87518", "#fff1d4")
#: Kept and dropped neurons, and MLP outputs (`renderNeuronViewer`'s shade).
NEURON = ("#1e6978", "#ffffff")
#: A coordinate with no unit (`.dbm-no-unit`).
NO_UNIT = "#f5f6f7"
#: The stripes of a unit outside the experiment (`.dbm-absent`).
OUTSIDE = ("#d4dae0", "#f8fafb")
#: The border of a mask cell (`.dbm-cell`).
CELL_EDGE = "#d4dfe3"
#: The layer-label marks for full-attention and Gated DeltaNet layers
#: (`.dbm-layer-label`).
LAYER_MARK = {"normal_attention": "#6b96ab", "gated_delta_net": "#b18a63"}
#: The sweep line, the chosen point's stroke and its ring (`renderChart`).
CURVE = "#829cab"
CHOSEN_STROKE = "#172d3e"
RING = "#b6323d"
#: The outline of a head the caller marks (`HeadMask.outlined`). It takes the
#: ring's red, which the viewer keeps for the one item it marks, and which
#: stands out on both the kept and the dropped blue.
OUTLINE = RING
#: The dot fill runs from the first colour at score 0 to the second at 1
#: (`effectColor`).
DOT_FROM = (232, 240, 244)
DOT_TO = (38, 111, 127)
#: Axis text and grid lines (`.dbm-chart text`, the chart's grid stroke).
TEXT = "#536375"
GRID = "#e1e7ed"

FAMILY_NAME = {
    "normal_attention": "full attention",
    "gated_delta_net": "Gated DeltaNet",
}


@dataclass(frozen=True)
class SweepPoint:
    """One fitted mask on the sweep curve."""

    #: The number of units the mask keeps.
    selected_count: int
    #: The mask's score, a fraction in [0, 1].
    iia: float
    #: Short text beside the dot, such as the L1 weight. Empty draws no text.
    #: Points at the same place share one text, their labels joined by commas.
    label: str = ""


@dataclass(frozen=True)
class DbmLayer:
    """One layer of the model, as `model.layers` of the export schema holds it."""

    #: The layer index.
    index: int
    #: The number of attention heads, or of value heads in a Gated DeltaNet
    #: layer; 0 for a layer with none.
    heads: int = 0
    #: `normal_attention` or `gated_delta_net`: the colours of its heads.
    type: Literal["normal_attention", "gated_delta_net"] = "normal_attention"


@dataclass(frozen=True)
class HeadMask:
    """A mask over attention heads, drawn as a layer by head matrix.

    Every layer of the model gets a row, and a layer absent from `kept`
    is drawn striped, outside the experiment. The matrix has as many columns
    as the widest layer; a coordinate past a layer's heads has no unit.
    """

    #: Every layer of the model, in order.
    layers: Sequence[DbmLayer]
    #: Layer index to one value per head of that layer: 1 kept, 0 dropped,
    #: or, when `masks` is above 1, the number of masks that keep the head.
    kept: Mapping[int, Sequence[int]]
    #: The number of masks the values count over. A count shades the cell
    #: from the dropped to the kept colour.
    masks: int = 1
    #: Draw layers as columns and heads as rows, as a paper's heatmap with
    #: layer on the horizontal axis does. The default gives each layer a row,
    #: as the encyclopedia viewer does.
    layers_across: bool = False
    #: `(layer index, head)` pairs to outline, such as the heads a paper
    #: ranks highest. The figure does not name them; the title or the
    #: caption does.
    outlined: Sequence[tuple[int, int]] = ()


@dataclass(frozen=True)
class NeuronMask:
    """A mask over the neurons of one component, drawn as a grid of cells.

    The grid is 64 cells wide, and it doubles in width until it is at most a
    quarter as tall as it is wide. Neuron `i` sits at row `i // width` and
    column `i % width`.
    """

    #: One value per neuron: 1 kept, 0 dropped, or the count over `masks`.
    kept: Sequence[int]
    #: The number of masks the values count over.
    masks: int = 1


@dataclass(frozen=True)
class ComponentMask:
    """A mask with one gate per attention output and per MLP output of each
    layer, drawn as one row of layer tiles per component."""

    #: Every layer of the model, in order. Its `type` colours the attention
    #: tiles.
    layers: Sequence[DbmLayer]
    #: `attention` or `mlp` to one value per layer, in the order of `layers`.
    #: A component absent here is drawn striped, outside the experiment.
    kept: Mapping[Literal["attention", "mlp"], Sequence[int]] = field(
        default_factory=dict
    )
    #: The number of masks the values count over.
    masks: int = 1


Mask = HeadMask | NeuronMask | ComponentMask


def plot_dbm(
    mask: Mask,
    sweep: Sequence[SweepPoint] = (),
    *,
    chosen: int | None = None,
    metric: str = "held-out IIA",
    title: str = "",
) -> Figure:
    """Draw a DBM sweep curve over the mask of one of its points.

    The curve puts each point at `log1p(selected_count) / log1p(N)` on the
    horizontal axis, where `N` is the number of units in the experiment, so
    that a mask that keeps nothing still has a place. The tick labels give
    the counts. Each dot is filled by its score, and the chosen point gets a
    dark stroke and a dashed red ring. With no sweep, the figure is the mask
    panel alone.

    The figure is 7.5 inches wide. Its axes carry the labels `sweep` and
    `mask`. Each head cell and component tile carries a gid such as
    `head:L15:H2` or `component:mlp:L3`, and each outline of
    `HeadMask.outlined` a gid such as `outline:L15:H2`. The curve is `dbm-curve`, the dots
    `dbm-points` in the order of `sweep`, the chosen dot drawn again above
    them `dbm-chosen`, its ring `dbm-ring`, and the neuron grid
    `dbm-neurons`.

    Args:
        mask: The mask to draw: the chosen point's mask, or a count over
            several masks.
        sweep: The points of the curve, in any order.
        chosen: The index into `sweep` of the point whose mask is drawn.
        metric: The name of the score, for the vertical axis and the mask
            heading.
        title: Text above the figure, such as the model and the data.

    Returns:
        The figure, outside pyplot's figure registry. The caller saves it.

    Raises:
        ValueError: A value lies outside `[0, masks]`, a layer's values do not
            match its heads, an outlined head is not in the model, `chosen` is out of range or set with no sweep, or
            the chosen point's `selected_count` differs from the number of
            units the mask keeps.
    """
    from matplotlib.figure import Figure

    grid = _mask_grid(mask)
    if chosen is not None and not 0 <= chosen < len(sweep):
        raise ValueError(f"chosen={chosen} is not an index into the sweep")
    if (
        chosen is not None
        and mask.masks == 1
        and sweep[chosen].selected_count != grid.kept
    ):
        raise ValueError(
            f"the chosen point keeps {sweep[chosen].selected_count} {grid.unit}, "
            f"the mask keeps {grid.kept}"
        )

    width, left, right = 7.5, 0.8, 0.3
    heights = {
        "title": 0.35 if title else 0.1,
        "sweep": 1.3 if sweep else 0.0,
        "sweep_ticks": 0.55 if sweep else 0.0,
        "heading": 0.3,
        "mask": grid.height(width - left - right),
        "mask_ticks": grid.tick_room,
        "legend": 0.45 if mask.masks == 1 else 0.75,
    }
    height = sum(heights.values())
    # a figure outside pyplot's registry, so that no caller needs to close it
    figure = Figure(figsize=(width, height), dpi=100)

    def rect(top: float, size: float) -> tuple[float, float, float, float]:
        """An axes box from its top edge and height, in inches."""
        return (
            left / width,
            1 - (top + size) / height,
            (width - left - right) / width,
            size / height,
        )

    if title:
        figure.text(
            0.5,
            1 - 0.18 / height,
            title,
            ha="center",
            va="center",
            fontsize=8,
            color=TEXT,
        )
    top = heights["title"]
    if sweep:
        ax = figure.add_axes(rect(top, heights["sweep"]), label="sweep")
        _draw_sweep(ax, sweep, chosen, grid.eligible, grid.unit, metric)
        top += heights["sweep"] + heights["sweep_ticks"]
    heading = _heading(
        mask, grid, sweep[chosen] if chosen is not None else None, metric
    )
    figure.text(
        left / width,
        1 - (top + 0.15) / height,
        heading,
        ha="left",
        va="center",
        fontsize=8.5,
        color="#243142",
        weight="semibold",
    )
    top += heights["heading"]
    ax = figure.add_axes(rect(top, heights["mask"]), label="mask")
    grid.draw(ax)
    top += heights["mask"] + heights["mask_ticks"]
    _legend(figure, rect(top, heights["legend"]), mask, grid)
    return figure


# ---------------------------------------------------------------------------
# The sweep curve (`renderChart`)
# ---------------------------------------------------------------------------


def _dot_colour(iia: float) -> str:
    """The fill of a dot: `effectColor` for IIA, rounded per channel."""
    amount = min(1.0, max(0.0, iia))
    channels = (round(a + (b - a) * amount) for a, b in zip(DOT_FROM, DOT_TO))
    return "#" + "".join(f"{c:02x}" for c in channels)


def _draw_sweep(
    ax: Axes,
    sweep: Sequence[SweepPoint],
    chosen: int | None,
    eligible: int,
    unit: str,
    metric: str,
) -> None:
    for point in sweep:
        if not 0 <= point.selected_count <= eligible:
            raise ValueError(
                f"a sweep point keeps {point.selected_count} of {eligible} {unit}"
            )
    scale = math.log1p(max(1, eligible))
    xs = [math.log1p(p.selected_count) / scale for p in sweep]
    ys = [p.iia for p in sweep]
    order = sorted(range(len(sweep)), key=lambda i: (xs[i], i))
    for value in (0.0, 0.5, 1.0):
        ax.axhline(value, color=GRID, linewidth=0.8, zorder=0)
    ax.plot(
        [xs[i] for i in order],
        [ys[i] for i in order],
        color=CURVE,
        linewidth=1.5,
        gid="dbm-curve",
        zorder=1,
    )
    ax.scatter(
        xs,
        ys,
        s=28,
        c=[_dot_colour(p.iia) for p in sweep],
        edgecolors="#ffffff",
        linewidths=1.1,
        gid="dbm-points",
        zorder=3,
        clip_on=False,
    )
    if chosen is not None:
        # drawn again above the others, so that a point at the same place
        # cannot hide its stroke
        ax.scatter(
            [xs[chosen]],
            [ys[chosen]],
            s=56,
            c=[_dot_colour(sweep[chosen].iia)],
            edgecolors=CHOSEN_STROKE,
            linewidths=1.9,
            gid="dbm-chosen",
            zorder=3.5,
            clip_on=False,
        )
        # a marker, so that the ring stays round whatever the axes' aspect;
        # 12 points across, as the viewer's ring of radius 8 against dots of 3.5
        ax.scatter(
            [xs[chosen]],
            [ys[chosen]],
            s=144,
            facecolors="none",
            edgecolors=RING,
            linewidths=1.5,
            linestyles=[(0, (1.5, 1.0))],
            gid="dbm-ring",
            zorder=4,
            clip_on=False,
        )
    labels: dict[tuple[float, float], list[str]] = {}
    for i in order:
        if sweep[i].label:
            labels.setdefault((xs[i], ys[i]), []).append(sweep[i].label)
    for rank, ((x, y), texts) in enumerate(labels.items()):
        # alternate above and below, so that neighbouring labels do not collide
        above = rank % 2 == 0
        ax.annotate(
            ", ".join(texts),
            (x, y),
            xytext=(0, 9 if above else -9),
            textcoords="offset points",
            ha="center",
            va="bottom" if above else "top",
            fontsize=6.5,
            color=TEXT,
            annotation_clip=False,
        )
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.08, 1.12)
    ticks = [i / 4 for i in range(5)]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{round(math.expm1(t * scale)):,}" for t in ticks])
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_yticklabels(["0", "0.5", "1"])
    ax.tick_params(length=0, labelsize=7.5, labelcolor=TEXT)
    ax.set_xlabel(
        f"{unit} kept (logarithmic scale, including zero)", fontsize=8, color=TEXT
    )
    ax.set_ylabel(metric, fontsize=8, color=TEXT)
    for spine in ax.spines.values():
        spine.set_visible(False)


# ---------------------------------------------------------------------------
# The mask panel (`renderHeadViewer`, `renderNeuronViewer`)
# ---------------------------------------------------------------------------


def _mix(pair: tuple[str, str], value: int, masks: int) -> str:
    """The kept colour for `value == masks`, the dropped one for 0, and a
    linear blend between, as the viewer's `shade` does for counts."""
    from matplotlib.colors import to_hex, to_rgb

    if value == masks:
        return pair[0]
    if value == 0:
        return pair[1]
    (kr, kg, kb), (dr, dg, db) = to_rgb(pair[0]), to_rgb(pair[1])
    t = value / masks
    return to_hex((dr + (kr - dr) * t, dg + (kg - dg) * t, db + (kb - db) * t))


def _check_values(values: Sequence[int], masks: int, what: str) -> None:
    if masks < 1:
        raise ValueError(f"masks={masks}: a count needs at least one mask")
    for value in values:
        if int(value) != value or not 0 <= value <= masks:
            raise ValueError(f"{what}: {value} is not a count in [0, {masks}]")


@dataclass
class _Grid:
    """What the mask panel needs to lay itself out and to head its legend."""

    unit: str
    #: Units in the experiment, and the number the mask keeps (for one mask).
    eligible: int
    kept: int
    rows: int
    columns: int
    #: Row height in inches, or 0 for square cells.
    row_height: float
    tick_room: float
    families: tuple[str, ...]
    outside: bool
    no_unit: bool
    mask: Mask

    def height(self, width: float) -> float:
        if self.row_height:
            return self.rows * self.row_height
        return self.rows * width / self.columns

    def draw(self, ax: Axes) -> None:
        if isinstance(self.mask, HeadMask):
            _draw_heads(ax, self.mask)
        elif isinstance(self.mask, NeuronMask):
            _draw_neurons(ax, self.mask, self.rows, self.columns)
        else:
            _draw_components(ax, self.mask)


def _mask_grid(mask: Mask) -> _Grid:
    """Check a mask and measure the panel that draws it."""
    if isinstance(mask, HeadMask):
        by_index = {layer.index: layer for layer in mask.layers}
        for index, values in mask.kept.items():
            if index not in by_index:
                raise ValueError(f"layer {index} is not among the model's layers")
            if len(values) != by_index[index].heads:
                raise ValueError(
                    f"layer {index} has {by_index[index].heads} heads, "
                    f"the mask gives {len(values)} values"
                )
            _check_values(values, mask.masks, f"layer {index}")
        for index, head in mask.outlined:
            if index not in by_index or not 0 <= head < by_index[index].heads:
                raise ValueError(f"outlined head L{index}:H{head} is not in the model")
        columns = max([layer.heads for layer in mask.layers] + [1])
        inside = [by_index[i] for i in mask.kept]
        families = tuple(
            family
            for family in ("normal_attention", "gated_delta_net")
            if any(layer.type == family for layer in inside)
        )
        return _Grid(
            unit="heads",
            eligible=sum(len(v) for v in mask.kept.values()),
            kept=sum(int(v) for values in mask.kept.values() for v in values),
            rows=columns if mask.layers_across else len(mask.layers),
            columns=len(mask.layers) if mask.layers_across else columns,
            row_height=0.16 if mask.layers_across else 0.13,
            tick_room=0.45,
            families=families,
            outside=any(
                layer.heads and layer.index not in mask.kept for layer in mask.layers
            ),
            no_unit=any(layer.heads < columns for layer in mask.layers),
            mask=mask,
        )
    if isinstance(mask, NeuronMask):
        _check_values(mask.kept, mask.masks, "neurons")
        count = len(mask.kept)
        if not count:
            raise ValueError("a neuron mask needs at least one neuron")
        columns = 64
        while math.ceil(count / columns) > columns / 4:
            columns *= 2
        rows = math.ceil(count / columns)
        return _Grid(
            unit="neurons",
            eligible=count,
            kept=sum(int(v) for v in mask.kept),
            rows=rows,
            columns=columns,
            row_height=0.0,
            tick_room=0.3,
            families=(),
            outside=False,
            no_unit=rows * columns > count,
            mask=mask,
        )
    for name, values in mask.kept.items():
        if name not in ("attention", "mlp"):
            raise ValueError(f"component {name!r} is neither 'attention' nor 'mlp'")
        if len(values) != len(mask.layers):
            raise ValueError(
                f"{name} gives {len(values)} values for {len(mask.layers)} layers"
            )
        _check_values(values, mask.masks, name)
    families = tuple(
        family
        for family in ("normal_attention", "gated_delta_net")
        if "attention" in mask.kept
        and any(layer.type == family for layer in mask.layers)
    )
    return _Grid(
        unit="components",
        eligible=sum(len(v) for v in mask.kept.values()),
        kept=sum(int(v) for values in mask.kept.values() for v in values),
        rows=2,
        columns=len(mask.layers),
        row_height=0.3,
        tick_room=0.3,
        families=families + (("mlp",) if "mlp" in mask.kept else ()),
        outside=len(mask.kept) < 2,
        no_unit=False,
        mask=mask,
    )


def _heading(mask: Mask, grid: _Grid, point: SweepPoint | None, metric: str) -> str:
    if mask.masks > 1:
        return f"{mask.masks} masks over {grid.eligible:,} {grid.unit}"
    text = f"{grid.kept:,} of {grid.eligible:,} {grid.unit} kept"
    if point is None:
        return text
    return f"Mask at the ringed point: {text}, {metric} {point.iia:.3f}"


def _cell(
    ax: Axes,
    x: float,
    y: float,
    colour: str | None,
    gid: str,
    *,
    size: float = 1.0,
) -> None:
    """One matrix cell; `colour=None` draws the stripes of `.dbm-absent`."""
    from matplotlib.patches import Rectangle

    if colour is None:
        patch = Rectangle(
            (x, y),
            size,
            size,
            facecolor=OUTSIDE[1],
            edgecolor=OUTSIDE[0],
            hatch="////",
            linewidth=0.5,
        )
    else:
        patch = Rectangle(
            (x, y), size, size, facecolor=colour, edgecolor=CELL_EDGE, linewidth=0.5
        )
    patch.set_gid(gid)
    ax.add_patch(patch)


def _draw_heads(ax: Axes, mask: HeadMask) -> None:
    from matplotlib.patches import Rectangle

    columns = max([layer.heads for layer in mask.layers] + [1])
    for r, layer in enumerate(mask.layers):
        values = mask.kept.get(layer.index)
        pair = DELTA_NET if layer.type == "gated_delta_net" else ATTENTION
        for head in range(columns):
            if head >= layer.heads:
                colour: str | None = NO_UNIT
            elif values is None:
                colour = None
            else:
                colour = _mix(pair, int(values[head]), mask.masks)
            x, y = (r, head) if mask.layers_across else (head, r)
            _cell(ax, x, y, colour, f"head:L{layer.index}:H{head}")
        # the family mark beside the layer label (`.dbm-layer-label`)
        mark = LAYER_MARK.get(layer.type, LAYER_MARK["normal_attention"])
        if mask.layers_across:
            box = Rectangle((r + 0.1, -0.4), 0.8, 0.15, color=mark, clip_on=False)
        else:
            box = Rectangle((-0.4, r + 0.1), 0.15, 0.8, color=mark, clip_on=False)
        ax.add_patch(box)
    row_of = {layer.index: r for r, layer in enumerate(mask.layers)}
    for index, head in mask.outlined:
        r = row_of[index]
        x, y = (r, head) if mask.layers_across else (head, r)
        outline = Rectangle(
            (x, y),
            1,
            1,
            fill=False,
            edgecolor=OUTLINE,
            linewidth=1.5,
            zorder=3,
            # the matrix's last row and column sit on the axes' edge
            clip_on=False,
        )
        outline.set_gid(f"outline:L{index}:H{head}")
        ax.add_patch(outline)
    layers = [layer.index for layer in mask.layers]
    ax.set_xlim(-0.5, (len(layers) if mask.layers_across else columns))
    ax.set_ylim((columns if mask.layers_across else len(layers)), -0.5)
    layer_ticks = ([i + 0.5 for i in range(len(layers))], [str(i) for i in layers])
    head_ticks = ([h + 0.5 for h in range(columns)], [str(h) for h in range(columns)])
    x_ticks, y_ticks = (
        (layer_ticks, head_ticks) if mask.layers_across else (head_ticks, layer_ticks)
    )
    ax.set_xticks(*x_ticks)
    ax.set_yticks(*y_ticks)
    ax.set_xlabel("layer" if mask.layers_across else "head", fontsize=8, color=TEXT)
    ax.set_ylabel("head" if mask.layers_across else "layer", fontsize=8, color=TEXT)
    _plain_axes(ax, labelsize=6.5)


def _draw_neurons(ax: Axes, mask: NeuronMask, rows: int, columns: int) -> None:
    import numpy as np
    from matplotlib.colors import to_rgb

    image = np.tile(np.array(to_rgb(NO_UNIT)), (rows, columns, 1))
    for i, value in enumerate(mask.kept):
        image[i // columns, i % columns] = to_rgb(_mix(NEURON, int(value), mask.masks))
    ax.imshow(
        image,
        interpolation="nearest",
        aspect="auto",
        gid="dbm-neurons",
        extent=(0, columns, rows, 0),
    )
    # cell borders (`.dbm-leaf-grid` gaps), thin enough for a wide grid
    width = 0.5 if columns <= 64 else 0.25
    ax.vlines(range(columns + 1), 0, rows, color=CELL_EDGE, linewidth=width)
    ax.hlines(range(rows + 1), 0, columns, color=CELL_EDGE, linewidth=width)
    ax.set_xlim(0, columns)
    ax.set_ylim(rows, 0)
    ax.set_xticks([0.5, columns - 0.5], ["0", str(columns - 1)])
    ax.set_yticks([0.5, rows - 0.5], ["0", f"{(rows - 1) * columns:,}"])
    ax.set_xlabel(f"neuron index mod {columns}", fontsize=7.5, color=TEXT)
    ax.set_ylabel("first neuron of the row", fontsize=7.5, color=TEXT)
    _plain_axes(ax, labelsize=6.5)


def _draw_components(ax: Axes, mask: ComponentMask) -> None:
    for row, name in enumerate(("attention", "mlp")):
        values = mask.kept.get(name)
        for c, layer in enumerate(mask.layers):
            if values is None:
                colour: str | None = None
            else:
                pair = (
                    NEURON
                    if name == "mlp"
                    else (DELTA_NET if layer.type == "gated_delta_net" else ATTENTION)
                )
                colour = _mix(pair, int(values[c]), mask.masks)
            # a gap between tiles, as the viewer's layer summary has
            _cell(
                ax,
                c + 0.06,
                row + 0.06,
                colour,
                f"component:{name}:L{layer.index}",
                size=0.88,
            )
            if colour is not None:
                dark = values is not None and int(values[c]) * 2 > mask.masks
                ax.text(
                    c + 0.5,
                    row + 0.5,
                    str(layer.index),
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="#ffffff" if dark else "#243142",
                )
    ax.set_xlim(0, len(mask.layers))
    ax.set_ylim(2, 0)
    ax.set_xticks([])
    ax.set_yticks([0.5, 1.5], ["attention", "MLP"])
    ax.set_xlabel("layer", fontsize=8, color=TEXT)
    _plain_axes(ax, labelsize=7.5)


def _plain_axes(ax: Axes, labelsize: float) -> None:
    ax.tick_params(length=0, labelsize=labelsize, labelcolor=TEXT)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _legend(
    figure: Figure, box: tuple[float, float, float, float], mask: Mask, grid: _Grid
) -> None:
    """Swatches for the colours the mask panel uses, and a colour bar per
    family when the values count over several masks."""
    from matplotlib.patches import Patch

    ax = figure.add_axes(box, label="legend")
    ax.set_axis_off()
    pairs = {"normal_attention": ATTENTION, "gated_delta_net": DELTA_NET, "mlp": NEURON}
    if isinstance(mask, NeuronMask):
        families: list[tuple[str, tuple[str, str]]] = [("", NEURON)]
    elif isinstance(mask, ComponentMask):
        attention = [f for f in grid.families if f != "mlp"]
        families = [
            (
                "MLP output"
                if f == "mlp"
                else "attention output"
                if len(attention) == 1
                else f"attention output, {FAMILY_NAME[f]}",
                pairs[f],
            )
            for f in grid.families
        ]
    else:
        families = [(FAMILY_NAME[f], pairs[f]) for f in grid.families]
    handles: list[Patch] = []
    if mask.masks == 1:
        for name, (kept, dropped) in families:
            suffix = f" ({name})" if name else ""
            handles.append(
                Patch(facecolor=kept, edgecolor=CELL_EDGE, label=f"kept{suffix}")
            )
            handles.append(
                Patch(facecolor=dropped, edgecolor=CELL_EDGE, label=f"dropped{suffix}")
            )
    if isinstance(mask, HeadMask) and len({layer.type for layer in mask.layers}) > 1:
        # the layer-label marks, as the viewer's caption names them
        for kind, mark in LAYER_MARK.items():
            handles.append(
                Patch(
                    facecolor=mark, edgecolor=mark, label=f"{FAMILY_NAME[kind]} layer"
                )
            )
    if grid.outside:
        handles.append(
            Patch(
                facecolor=OUTSIDE[1],
                edgecolor=OUTSIDE[0],
                hatch="////",
                label="outside the experiment",
            )
        )
    if grid.no_unit:
        handles.append(Patch(facecolor=NO_UNIT, edgecolor=CELL_EDGE, label="no unit"))
    if handles:
        ax.legend(
            handles=handles,
            loc="upper left",
            # one row up to four swatches, else two rows
            ncol=len(handles) if len(handles) <= 4 else math.ceil(len(handles) / 2),
            frameon=False,
            fontsize=7,
            handlelength=1.2,
            handleheight=1.0,
            borderaxespad=0.0,
            labelcolor=TEXT,
        )
    if mask.masks > 1:
        _count_bars(figure, box, families, mask.masks, grid.unit)


def _count_bars(
    figure: Figure,
    box: tuple[float, float, float, float],
    families: Sequence[tuple[str, tuple[str, str]]],
    masks: int,
    unit: str,
) -> None:
    """One discrete colour bar per family, from 0 to `masks`."""
    import numpy as np
    from matplotlib.colors import ListedColormap

    x, y, w, h = box
    bar_w = min(w, 0.4) / max(1, len(families))
    for k, (name, pair) in enumerate(families):
        colours = [_mix(pair, value, masks) for value in range(masks + 1)]
        ax = figure.add_axes(
            (x + k * (bar_w + 0.05), y + h * 0.62, bar_w, h * 0.16), label=f"count-{k}"
        )
        ax.imshow(
            np.arange(masks + 1)[None, :],
            cmap=ListedColormap(colours),
            aspect="auto",
            extent=(-0.5, masks + 0.5, 0, 1),
            interpolation="nearest",
        )
        ax.set_yticks([])
        ticks = sorted({0, masks // 2, masks})
        ax.set_xticks(ticks, [str(t) for t in ticks])
        subject = f" ({name})" if name else ""
        ax.set_xlabel(
            f"masks that keep the {unit[:-1]}, of {masks}{subject}",
            fontsize=7,
            color=TEXT,
        )
        ax.tick_params(length=0, labelsize=6.5, labelcolor=TEXT)
        for spine in ax.spines.values():
            spine.set_edgecolor(CELL_EDGE)
