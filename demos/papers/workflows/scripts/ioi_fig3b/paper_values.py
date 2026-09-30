"""Read the values of Figure 3b and Figure 15 (left) off the IOI paper's PDF.

Wang et al. 2022 (arXiv:2211.00593v1) draw the direct effect of each head
twice: Figure 3b is a heatmap of all 144 heads, and the left panel of
Figure 15 is a bar chart of the 15 largest. The page compares this
package's values with both, so this script writes what the PDF holds into
``artifacts/data/ioi_fig3b/fig3b_wang2022_values.json``:

* **Figure 3b** (page 6) is an embedded 320 x 320 image with one colour per
  cell. The authors' ``show_pp`` draws it with plotly's ``RdBu`` scale and
  ``color_continuous_midpoint=0`` (Easy-Transformer ``ioi_utils.py`` lines
  89-95 at ``373cd15``), so the scale runs from ``-zmax`` to ``zmax`` and
  ``zmax`` is the largest absolute value. A cell's value is
  ``zmax * (2 t - 1)``, where ``t`` is the point of the scale whose colour is
  nearest the cell's. ``zmax`` is half the colour bar's length over the
  spacing of its tick marks, both read from the page's vector drawing.
* **Figure 15** (page 20) is a vector drawing. A bar's value is the distance
  of its end from the 0% grid line, where it starts, over the grid lines'
  spacing per unit of value.

It needs poppler's ``pdfimages``, ``pdftocairo`` and ``pdftotext``. Usage,
from ``demos/papers/``::

    curl -L -o 2211.00593v1.pdf https://arxiv.org/pdf/2211.00593v1
    python workflows/scripts/ioi_fig3b/paper_values.py --pdf 2211.00593v1.pdf            # writes the values file
    python workflows/scripts/ioi_fig3b/paper_values.py --pdf 2211.00593v1.pdf --check    # exit 1 if it differs
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

__all__ = ["RDBU", "READING_ERROR", "scale_position", "extract", "main"]

#: ``demos/papers/``: this script sits in ``workflows/scripts/ioi_fig3b/``.
PAPERS = Path(__file__).resolve().parents[3]
VALUES = PAPERS / "artifacts" / "data" / "ioi_fig3b" / "fig3b_wang2022_values.json"
URL = "https://arxiv.org/pdf/2211.00593v1"
FIG3B_PAGE, FIG15_PAGE = 6, 20
LAYERS = HEADS = 12
#: plotly's ``RdBu`` (``plotly.colors.diverging.RdBu``, from ColorBrewer),
#: which ``px.imshow(color_continuous_scale="RdBu")`` spreads evenly over the
#: scale and interpolates linearly in RGB. The 257 stops of the colour bar's
#: gradient on page 6 equal this interpolation within 0.7 of 255.
RDBU = (
    (103, 0, 31),
    (178, 24, 43),
    (214, 96, 77),
    (244, 165, 130),
    (253, 219, 199),
    (247, 247, 247),
    (209, 229, 240),
    (146, 197, 222),
    (67, 147, 195),
    (33, 102, 172),
    (5, 48, 97),
)
#: The reading error of a value. Figure 3b: one 8-bit colour step near
#: white moves a value by 0.0035 to 0.0048, and ``zmax`` is known to about
#: 0.0015. Figure 15: the grid lines sit up to 0.1 of 179 units per unit of
#: value from an even spacing, so a bar reads to about 0.001.
READING_ERROR = {"fig3b": 0.005, "fig15": 0.001}
#: The fill colours of the head classes in Figure 15, as pdftocairo writes them.
_BAR_FILLS = (
    "10.588074%, 61.959839%, 46.665955%",
    "85.096741%, 37.254333%, 0.784302%",
    "72.940063%, 91.763306%, 51.763916%",
    "45.881653%, 43.920898%, 70.195007%",
)
#: The plot-area fill of both Figure 15 panels.
_PLOT_AREA = 'fill="rgb(89.802551%, 92.547607%, 96.469116%)" fill-opacity="1" d='
_NUMBER = r"-?[0-9.]+"


def scale_position(rgb: Sequence[float]) -> float:
    """The point ``t`` in ``[0, 1]`` of the `RDBU` scale nearest ``rgb``.

    Each of the ten segments between stops is a line in RGB space, and ``t``
    is the nearest point over all of them. A colour on the scale comes back
    exactly, and its 8-bit rounding moves ``t`` by less than one step."""
    best_t, best_distance = 0.0, float("inf")
    for i in range(len(RDBU) - 1):
        a, b = RDBU[i], RDBU[i + 1]
        d = [b[k] - a[k] for k in range(3)]
        f = sum((rgb[k] - a[k]) * d[k] for k in range(3)) / sum(x * x for x in d)
        f = min(1.0, max(0.0, f))
        distance = sum((a[k] + f * d[k] - rgb[k]) ** 2 for k in range(3))
        if distance < best_distance:
            best_t, best_distance = (i + f) / (len(RDBU) - 1), distance
    return best_t


def _run(*command: str) -> None:
    subprocess.run(command, check=True, capture_output=True)


def _words(pdf: Path, page: int, out: Path) -> list[tuple[float, float, str]]:
    """``(x, y, text)`` of every word on ``page``, from ``pdftotext -bbox``."""
    _run("pdftotext", "-bbox", "-f", str(page), "-l", str(page), str(pdf), str(out))
    found = re.findall(
        r'<word xMin="([0-9.]+)" yMin="([0-9.]+)"[^>]*>([^<]*)</word>',
        out.read_text(),
    )
    return [(float(x), float(y), text) for x, y, text in found]


def _percent_labels(
    words: list[tuple[float, float, str]], x_range: tuple[float, float]
) -> list[float]:
    """The values of the ``-50%`` to ``50%`` tick labels inside ``x_range``,
    top to bottom."""
    labels = [
        (y, int(text[:-1]) / 100)
        for x, y, text in words
        if x_range[0] <= x <= x_range[1] and re.fullmatch(r"-?\d+%", text)
    ]
    return [value for _, value in sorted(labels)]


def _fit(ys: list[float], values: list[float]) -> tuple[float, float]:
    """The least-squares line ``y = zero + slope * value``."""
    n = len(ys)
    mean_v, mean_y = sum(values) / n, sum(ys) / n
    slope = sum((v - mean_v) * (y - mean_y) for v, y in zip(values, ys)) / sum(
        (v - mean_v) ** 2 for v in values
    )
    return mean_y - slope * mean_v, slope


def _heatmap(pdf: Path, tmp: Path) -> dict[str, Any]:
    """The Figure 3b block: the colour bar's end and the 144 cells."""
    from PIL import Image

    page = str(FIG3B_PAGE)
    _run("pdfimages", "-f", page, "-l", page, "-png", str(pdf), str(tmp / "p6"))
    images = sorted(tmp.glob("p6-*.png"))
    assert len(images) == 1, f"page {page} holds {len(images)} images, not one"
    image = Image.open(images[0]).convert("RGB")
    assert image.size == (320, 320), image.size
    _run("pdftocairo", "-svg", "-f", page, "-l", page, str(pdf), str(tmp / "p6.svg"))
    svg = (tmp / "p6.svg").read_text()
    bars = re.findall(
        rf'fill="url\(#linear-pattern-0\)" d="M {_NUMBER} ({_NUMBER}) '
        rf"L {_NUMBER} ({_NUMBER}) L ({_NUMBER}) ",
        svg,
    )
    assert len(bars) == 1, f"expected one colour bar, found {len(bars)}"
    top, bottom, right = (float(v) for v in bars[0])
    # the tick marks: short grey strokes at the bar's right edge, in their
    # own units, which the transform maps onto the bar's
    ticks = [
        float(d) * float(y) + float(f)
        for x1, y, a, d, e, f in re.findall(
            rf'stroke="rgb\(26.66626%, 26.66626%, 26.66626%\)"[^>]* '
            rf'd="M ({_NUMBER}) ({_NUMBER}) L {_NUMBER} \2 " '
            rf'transform="matrix\(({_NUMBER}), 0, 0, ({_NUMBER}), ({_NUMBER}), ({_NUMBER})\)"',
            svg,
        )
        if abs(float(a) * float(x1) + float(e) - right) < 1.0
    ]
    labels = _percent_labels(_words(pdf, FIG3B_PAGE, tmp / "p6.html"), (310, 340))
    assert len(ticks) == len(labels) == 5, (ticks, labels)
    _, slope = _fit(sorted(ticks), labels)
    zmax = (bottom - top) / 2 / abs(slope)
    records = []
    for layer in range(LAYERS):
        for head in range(HEADS):
            x, y = int((head + 0.5) * 320 / HEADS), int((layer + 0.5) * 320 / LAYERS)
            pixel = image.getpixel((x, y))
            # an RGB image gives one (r, g, b) tuple per pixel
            assert isinstance(pixel, tuple), pixel
            rgb = [int(c) for c in pixel]
            t = scale_position(rgb)
            records.append(
                {
                    "layer": layer,
                    "head": head,
                    "rgb": rgb,
                    "t": round(t, 6),
                    "value": round(zmax * (2 * t - 1), 6),
                }
            )
    return {
        "page": FIG3B_PAGE,
        "extraction": {
            "image": f"pdfimages -f {page} -l {page} -png: the page's only image, 320 x 320",
            "image_sha256": hashlib.sha256(images[0].read_bytes()).hexdigest(),
            "cells": (
                "12 rows, layers 0 to 11 from the top, by 12 columns, heads 0 to 11 "
                "from the left; each cell is one colour, read at pixel "
                "(int((head + 0.5) * 320 / 12), int((layer + 0.5) * 320 / 12))"
            ),
            "colour_bar": (
                f"pdftocairo -svg -f {page} -l {page}: the gradient-filled rectangle "
                "and the five tick marks on its right edge, labelled 50% to -50% "
                "(pdftotext -bbox)"
            ),
        },
        "decode": {
            "scale": [list(stop) for stop in RDBU],
            "bar": {"top": top, "bottom": bottom},
            "ticks": sorted(round(t, 6) for t in ticks),
            "tick_values": labels,
            "zmax": round(zmax, 6),
            "rule": (
                "t = the point of the scale (11 evenly spaced stops, linear in RGB) "
                "nearest the cell's colour; value = zmax * (2 t - 1); zmax = "
                "(bottom - top) / 2 over the tick spacing per unit of value, from a "
                "least-squares line through the five ticks"
            ),
        },
        "reading_error": READING_ERROR["fig3b"],
        "records": records,
    }


def _bars(pdf: Path, tmp: Path) -> dict[str, Any]:
    """The Figure 15 block: the left panel's 15 bars."""
    page = str(FIG15_PAGE)
    _run("pdftocairo", "-svg", "-f", page, "-l", page, str(pdf), str(tmp / "p20.svg"))
    lines = (tmp / "p20.svg").read_text().splitlines()
    areas = [i for i, line in enumerate(lines) if _PLOT_AREA in line]
    assert len(areas) == 2, f"expected two plot areas, found {len(areas)}"
    panel = lines[areas[0] : areas[1]]
    grid = [
        float(y)
        for line in panel
        for y in re.findall(
            rf'stroke="rgb\(100%, 100%, 100%\)"[^>]* d="M {_NUMBER} ({_NUMBER}) '
            rf'L {_NUMBER} \1 "',
            line,
        )
    ]
    bars = []
    for line in panel:
        if any(f'fill="rgb({fill})"' in line for fill in _BAR_FILLS):
            corners = re.search(
                rf'd="M ({_NUMBER}) ({_NUMBER}) L {_NUMBER} \2 L {_NUMBER} ({_NUMBER}) ',
                line,
            )
            if corners:
                x, y_top, y_bottom = (float(v) for v in corners.groups())
                # a bar starts on a grid line (the zero line); a legend swatch
                # of the same colour does not
                if y_top in grid or y_bottom in grid:
                    bars.append((x, y_top, y_bottom))
    words = _words(pdf, FIG15_PAGE, tmp / "p20.html")
    labels = _percent_labels(words, (120, 135))
    assert len(grid) == len(labels) == 5 and len(bars) == 15, (grid, labels, bars)
    # every bar starts on the 0% grid line; the spacing per unit of value is
    # the least-squares slope through all five lines
    zero = sorted(grid)[labels.index(0.0)]
    _, slope = _fit(sorted(grid), labels)
    # the left panel's x labels read "(9," over "9)"; pair each opening word
    # with the closing word nearest below it
    openers = sorted(
        (x, t) for x, _, t in words if x < 262 and re.fullmatch(r"\(\d+,", t)
    )
    closers = [(x, t) for x, _, t in words if x < 262 and re.fullmatch(r"\d+\)", t)]
    heads = [
        (int(t[1:-1]), int(min(closers, key=lambda c: abs(c[0] - x))[1][:-1]))
        for x, t in openers
    ]
    assert len(heads) == 15, heads
    records = []
    for (layer, head), (_, y_top, y_bottom) in zip(heads, sorted(bars)):
        end = y_bottom if abs(y_top - zero) < abs(y_bottom - zero) else y_top
        records.append(
            {
                "layer": layer,
                "head": head,
                "end": end,
                "value": round((end - zero) / slope, 6),
            }
        )
    return {
        "page": FIG15_PAGE,
        "extraction": {
            "svg": (
                f"pdftocairo -svg -f {page} -l {page}: the left plot area's white "
                "grid lines and its 15 bar rectangles, left to right; the heads are "
                "the panel's x labels (pdftotext -bbox)"
            ),
        },
        "decode": {
            "grid": sorted(grid),
            "grid_values": labels,
            "zero": round(zero, 6),
            "per_unit": round(slope, 6),
            "rule": (
                "value = (end - zero) / per_unit, with end the bar's edge away from "
                "the 0% grid line, zero that line and per_unit the slope of a "
                "least-squares line through the five grid lines"
            ),
        },
        "reading_error": READING_ERROR["fig15"],
        "records": records,
    }


def extract(pdf: Path) -> dict[str, Any]:
    """The values file for ``pdf``, arXiv 2211.00593v1."""
    with tempfile.TemporaryDirectory() as name:
        tmp = Path(name)
        return {
            "description": (
                "The values of Figure 3b and of the left panel of Figure 15 of Wang "
                "et al. 2022, read off the PDF by "
                "workflows/scripts/ioi_fig3b/paper_values.py. A Figure 3b record is "
                "one heatmap cell: its 8-bit colour, its point t on the colour scale "
                "and its value. A Figure 15 record is one bar: the y of its end in "
                "the drawing's units and its value. A value is the relative change "
                "of the logit difference, the variation of fig3b_plotted.json."
            ),
            "source": {
                "url": URL,
                "arxiv_id": "2211.00593",
                "version": "v1",
                "sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
            },
            "fig3b": _heatmap(pdf, tmp),
            "fig15": _bars(pdf, tmp),
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--pdf", type=Path, required=True, help="arXiv 2211.00593v1")
    parser.add_argument("--out", type=Path, default=VALUES, help="the values file")
    parser.add_argument(
        "--check", action="store_true", help="exit 1 if the values file differs"
    )
    args = parser.parse_args(argv)
    text = json.dumps(extract(args.pdf), indent=1) + "\n"
    if args.check:
        if not args.out.is_file() or args.out.read_text() != text:
            print(f"{args.out}: differs from a fresh extraction", file=sys.stderr)
            return 1
        print(f"{args.out}: reproduces")
        return 0
    args.out.write_text(text)
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
