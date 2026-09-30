"""``causalab.io.plots.dbm_figure`` on tiny synthetic masks.

The figure copies the encyclopedia DBM viewer's colours and marks, so these
tests pin them: the fill of kept, dropped, outside and missing cells, the
outline of a marked head, the stroke and ring of the chosen point, and the
log1p position of each point. The last tests check that the same input renders the same bytes and that a
mask that disagrees with its chosen point is refused.
"""

from __future__ import annotations

import io
import math
from typing import Any, cast

import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.colors import to_hex, to_rgb
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from causalab.io.plots.dbm_figure import (
    ComponentMask,
    DbmLayer,
    HeadMask,
    NeuronMask,
    SweepPoint,
    plot_dbm,
)

pytestmark = pytest.mark.unit

#: Two full-attention layers with 2 heads around a Gated DeltaNet layer
#: with 3 value heads, as Qwen3.5 interleaves them.
LAYERS = [
    DbmLayer(0, 2),
    DbmLayer(1, 3, "gated_delta_net"),
    DbmLayer(2, 2),
]
SWEEP = [
    SweepPoint(0, 0.0, "high"),
    SweepPoint(1, 1.0, "mid"),
    SweepPoint(5, 0.5, "low"),
]


def _axes(figure: Figure, label: str) -> Axes:
    (ax,) = [ax for ax in figure.axes if ax.get_label() == label]
    return ax


def _patch(ax: Axes, gid: str) -> Rectangle:
    (patch,) = [p for p in ax.patches if p.get_gid() == gid]
    assert isinstance(patch, Rectangle)
    return patch


def _fill(ax: Axes, gid: str) -> str:
    return to_hex(_patch(ax, gid).get_facecolor())


def _marks(ax: Axes, gid: str) -> PathCollection:
    (marks,) = [c for c in ax.collections if c.get_gid() == gid]
    assert isinstance(marks, PathCollection)
    return marks


def _hexes(colours: Any) -> list[str]:
    return [to_hex(tuple(row)) for row in np.asarray(colours).tolist()]


def _offsets(marks: PathCollection) -> list[list[float]]:
    return np.asarray(marks.get_offsets()).tolist()


def _pixels(ax: Axes) -> np.ndarray:
    (image,) = [im for im in ax.images if im.get_gid() == "dbm-neurons"]
    pixels = image.get_array()
    assert pixels is not None
    return np.asarray(pixels)


def test_head_cells_take_the_viewer_colours_by_family_and_membership() -> None:
    mask = HeadMask(LAYERS, {0: [1, 0], 1: [0, 1, 0]})
    ax = _axes(plot_dbm(mask), "mask")
    assert _fill(ax, "head:L0:H0") == "#3267a8"  # kept, full attention
    assert _fill(ax, "head:L0:H1") == "#e6eef9"  # dropped, full attention
    assert _fill(ax, "head:L1:H1") == "#b87518"  # kept, Gated DeltaNet
    assert _fill(ax, "head:L1:H0") == "#fff1d4"  # dropped, Gated DeltaNet
    # layer 2 has heads but is outside the experiment: striped
    outside = _patch(ax, "head:L2:H0")
    assert outside.get_hatch() == "////"
    # layer 0 has 2 heads, the matrix 3 columns: the third has no unit
    assert _fill(ax, "head:L0:H2") == "#f5f6f7"
    assert _patch(ax, "head:L0:H2").get_hatch() is None


def test_a_count_over_several_masks_shades_between_dropped_and_kept() -> None:
    mask = HeadMask([DbmLayer(0, 3)], {0: [0, 2, 4]}, masks=4)
    ax = _axes(plot_dbm(mask), "mask")
    assert _fill(ax, "head:L0:H0") == "#e6eef9"
    assert _fill(ax, "head:L0:H2") == "#3267a8"
    # 2 of 4 masks: the linear blend halfway, channel by channel
    (kr, kg, kb), (dr, dg, db) = to_rgb("#3267a8"), to_rgb("#e6eef9")
    halfway = ((kr + dr) / 2, (kg + dg) / 2, (kb + db) / 2)
    assert _fill(ax, "head:L0:H1") == to_hex(halfway)


def test_layers_across_puts_layers_on_the_horizontal_axis() -> None:
    mask = HeadMask(LAYERS, {0: [1, 0], 1: [0, 1, 0]}, layers_across=True)
    ax = _axes(plot_dbm(mask), "mask")
    assert _patch(ax, "head:L1:H2").get_xy() == (1, 2)
    assert [t.get_text() for t in ax.get_xticklabels()] == ["0", "1", "2"]


def test_outlined_heads_get_a_red_border_over_their_cell() -> None:
    """An outline sits on its head's cell in either orientation, above the
    cell, with no fill, so that the cell's colour still shows."""
    for across in (False, True):
        mask = HeadMask(
            LAYERS, {0: [1, 0], 1: [0, 1, 0]}, layers_across=across, outlined=[(1, 2)]
        )
        ax = _axes(plot_dbm(mask), "mask")
        outline = _patch(ax, "outline:L1:H2")
        assert outline.get_xy() == _patch(ax, "head:L1:H2").get_xy()
        assert to_hex(outline.get_edgecolor()) == "#b6323d"
        assert not outline.get_fill()
        assert outline.get_zorder() > _patch(ax, "head:L1:H2").get_zorder()
        assert not [p for p in ax.patches if p.get_gid() == "outline:L0:H0"]


def test_an_outlined_head_outside_the_model_is_refused() -> None:
    with pytest.raises(ValueError, match="L0:H2 is not in the model"):
        plot_dbm(HeadMask(LAYERS, {0: [1, 0]}, outlined=[(0, 2)]))
    with pytest.raises(ValueError, match="L5:H0 is not in the model"):
        plot_dbm(HeadMask(LAYERS, {0: [1, 0]}, outlined=[(5, 0)]))


def test_neuron_cells_are_teal_when_kept_and_white_when_dropped() -> None:
    kept = [0] * 100
    kept[3] = kept[70] = 1
    ax = _axes(plot_dbm(NeuronMask(kept)), "mask")
    pixels = _pixels(ax)
    assert pixels.shape[:2] == (2, 64)  # 64 wide, row 1 holds neurons 64-99
    assert _hexes(pixels[[0, 1, 0, 1], [3, 6, 4, 40]]) == [
        "#1e6978",  # rgb(30, 105, 120)
        "#1e6978",  # neuron 70
        "#ffffff",
        "#f5f6f7",  # past the last neuron: no unit
    ]


def test_a_wide_neuron_mask_widens_the_grid_until_it_is_flat() -> None:
    ax = _axes(plot_dbm(NeuronMask([0] * 14336)), "mask")
    assert _pixels(ax).shape[:2] == (56, 256)


def test_component_tiles_colour_attention_blue_and_mlp_teal() -> None:
    mask = ComponentMask(LAYERS, {"attention": [1, 0, 0], "mlp": [0, 0, 1]})
    ax = _axes(plot_dbm(mask), "mask")
    assert _fill(ax, "component:attention:L0") == "#3267a8"
    assert _fill(ax, "component:attention:L1") == "#fff1d4"  # a DeltaNet layer
    assert _fill(ax, "component:mlp:L2") == "#1e6978"
    assert _fill(ax, "component:mlp:L0") == "#ffffff"
    only_mlp = _axes(plot_dbm(ComponentMask(LAYERS, {"mlp": [0, 1, 0]})), "mask")
    assert _patch(only_mlp, "component:attention:L0").get_hatch() == "////"


def test_the_chosen_point_has_a_dark_stroke_and_a_dashed_red_ring() -> None:
    mask = HeadMask(LAYERS, {0: [1, 0], 1: [0, 0, 0], 2: [0, 0]})
    figure = plot_dbm(mask, SWEEP, chosen=1)
    ax = _axes(figure, "sweep")
    dots = _marks(ax, "dbm-points")
    assert _hexes(dots.get_edgecolor()) == ["#ffffff"]
    fills = _hexes(dots.get_facecolor())
    # the fill runs from rgb(232, 240, 244) at score 0 to rgb(38, 111, 127) at 1
    assert fills[0] == "#e8f0f4" and fills[1] == "#266f7f"
    top = _marks(ax, "dbm-chosen")
    assert _hexes(top.get_edgecolor()) == ["#172d3e"]
    assert _offsets(top) == [_offsets(dots)[1]]
    assert top.get_zorder() > dots.get_zorder()
    ring = _marks(ax, "dbm-ring")
    assert _hexes(ring.get_edgecolor()) == ["#b6323d"]
    assert len(ring.get_facecolor()) == 0  # a ring, not a disc
    # dashed 3 on, 2 off, as the viewer's stroke-dasharray "3 2"
    ((_, (on, off)),) = cast(list[tuple[float, list[float]]], ring.get_linestyle())
    assert on / off == pytest.approx(3 / 2)
    # centred on the chosen point, and drawn above every dot
    assert _offsets(ring) == [_offsets(dots)[1]]
    assert ring.get_zorder() > top.get_zorder()
    (curve,) = [line for line in ax.lines if line.get_gid() == "dbm-curve"]
    assert to_hex(curve.get_color()) == "#829cab"


def test_the_count_axis_is_log1p_and_includes_zero() -> None:
    mask = HeadMask(LAYERS, {0: [1, 0], 1: [0, 0, 0], 2: [0, 0]})
    ax = _axes(plot_dbm(mask, SWEEP, chosen=1), "sweep")
    eligible = 7  # the units in the experiment, the right end of the axis
    xs = [x for x, _ in _offsets(_marks(ax, "dbm-points"))]
    assert xs == pytest.approx(
        [math.log1p(n) / math.log1p(eligible) for n in (0, 1, 5)]
    )
    assert xs[0] == 0.0
    labels = [t.get_text() for t in ax.get_xticklabels()]
    assert labels[0] == "0" and labels[-1] == "7"
    # the curve joins the points in order of their count
    (curve,) = [line for line in ax.lines if line.get_gid() == "dbm-curve"]
    assert np.asarray(curve.get_ydata()).tolist() == [0.0, 1.0, 0.5]


def test_points_at_one_place_share_one_label() -> None:
    sweep = [SweepPoint(1, 1.0, "0.3"), SweepPoint(1, 1.0, "1"), SweepPoint(0, 0.0)]
    ax = _axes(plot_dbm(NeuronMask([1, 0]), sweep, chosen=0), "sweep")
    assert sorted(t.get_text() for t in ax.texts) == ["0.3, 1"]


def test_mixed_layer_types_name_their_label_marks_in_the_legend() -> None:
    legend = _axes(plot_dbm(HeadMask(LAYERS, {0: [1, 0]})), "legend").get_legend()
    assert legend is not None
    labels = [t.get_text() for t in legend.get_texts()]
    assert "full attention layer" in labels and "Gated DeltaNet layer" in labels


def test_the_same_input_renders_the_same_bytes() -> None:
    mask = HeadMask(LAYERS, {0: [1, 0], 1: [0, 0, 0], 2: [0, 0]})

    def render() -> bytes:
        buffer = io.BytesIO()
        plot_dbm(mask, SWEEP, chosen=1, title="synthetic").savefig(
            buffer, format="png", dpi=60
        )
        return buffer.getvalue()

    assert render() == render()


def test_a_chosen_point_that_disagrees_with_the_mask_is_refused() -> None:
    mask = HeadMask(LAYERS, {0: [1, 1], 1: [0, 0, 0], 2: [0, 0]})
    with pytest.raises(ValueError, match="the chosen point keeps 1 heads"):
        plot_dbm(mask, SWEEP, chosen=1)
    with pytest.raises(ValueError, match="not an index"):
        plot_dbm(mask, (), chosen=0)


def test_a_mask_that_does_not_fit_the_model_is_refused() -> None:
    with pytest.raises(ValueError, match="layer 0 has 2 heads"):
        plot_dbm(HeadMask(LAYERS, {0: [1]}))
    with pytest.raises(ValueError, match="not a count"):
        plot_dbm(NeuronMask([0, 2]))
    with pytest.raises(ValueError, match="3 layers"):
        plot_dbm(ComponentMask(LAYERS, {"mlp": [1]}))
