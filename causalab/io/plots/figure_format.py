"""Shared figure output format for rendered figures.

``png`` is the default and is **preferred over ``pdf`` unless a caller asks for
pdf explicitly** — it is what a reviewer can open inline, in a PR, or in a
notebook without a viewer. ``pdf`` is for print or when a vector figure is
actually needed; ``html`` for interactive figures (plotly and friends).

These are the *visualization* formats of the workflow layer's output rule
(``docs/workflow_protocol.md`` §2.5): unlike JSON and safetensors they carry no
record, so a declared figure is a rendering of an artifact rather than one
itself.
"""

from __future__ import annotations

import os
from typing import Literal

FigureFormat = Literal["png", "pdf", "html"]

#: png first: it is the one a reviewer can open anywhere (module docstring).
ALLOWED_FIGURE_FORMATS: frozenset[str] = frozenset({"png", "pdf", "html"})

#: The same set as file suffixes, for the workflow output-format check.
VISUALIZATION_SUFFIXES: tuple[str, ...] = (".png", ".pdf", ".html")


def normalize_figure_format(value: str | None, *, default: str = "png") -> str:
    """Return one of [`ALLOWED_FIGURE_FORMATS`][]; validate input.

    ``default`` is ``png`` deliberately — see the module docstring."""
    raw = default if value is None else str(value)
    fmt = raw.lower().lstrip(".")
    if fmt not in ALLOWED_FIGURE_FORMATS:
        raise ValueError(
            f"figure_format must be one of {sorted(ALLOWED_FIGURE_FORMATS)}, got {value!r}"
        )
    return fmt


def path_with_figure_format(path: str, figure_format: str | None) -> str:
    """Set or replace the file extension on ``path`` using ``figure_format``."""
    fmt = normalize_figure_format(figure_format, default="png")
    root, _ext = os.path.splitext(path)
    return f"{root}.{fmt}"
