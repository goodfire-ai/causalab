"""Unit tests for the shared figure-format helpers.

Pins the codebase default to PNG and verifies PDF stays opt-in:
the format is determined by config/argument, never baked into a hardcoded
extension, and an absent/empty config resolves to ``png``.
"""

import pytest

from causalab.io.plots.figure_format import (
    ALLOWED_FIGURE_FORMATS,
    normalize_figure_format,
    path_with_figure_format,
)

pytestmark = pytest.mark.unit


class TestNormalizeFigureFormat:
    def test_default_is_png(self):
        """No value supplied → png (the default flip to PNG)."""
        assert normalize_figure_format(None) == "png"

    def test_pdf_opt_in_preserved(self):
        assert normalize_figure_format("pdf") == "pdf"

    def test_case_and_dot_normalized(self):
        assert normalize_figure_format("PNG") == "png"
        assert normalize_figure_format(".PDF") == "pdf"

    def test_explicit_default_override(self):
        assert normalize_figure_format(None, default="pdf") == "pdf"

    def test_invalid_format_raises(self):
        with pytest.raises(ValueError):
            normalize_figure_format("svg")

    def test_allowed_set(self):
        """`html` joined png/pdf when the workflow layer began declaring figures
        as outputs (workflow spec §2.5) — interactive figures are a real third
        rendering, unlike a third *record* format."""
        assert ALLOWED_FIGURE_FORMATS == frozenset({"png", "pdf", "html"})

    def test_png_is_the_default(self):
        """The stated preference: png unless a caller asks for something else."""
        assert normalize_figure_format(None) == "png"


class TestPathWithFigureFormat:
    def test_extensionless_base_gets_png_default(self):
        """A bare basename + no format → png; format is never implicit in code."""
        assert path_with_figure_format("dir/heatmap", None) == "dir/heatmap.png"

    def test_pdf_literal_is_replaced_with_png_default(self):
        """A stale ``.pdf`` literal in code no longer leaks PDF output."""
        assert path_with_figure_format("dir/heatmap.pdf", None) == "dir/heatmap.png"

    def test_pdf_opt_in_replaces_any_extension(self):
        assert path_with_figure_format("dir/heatmap.png", "pdf") == "dir/heatmap.pdf"
        assert path_with_figure_format("dir/heatmap", "pdf") == "dir/heatmap.pdf"

    def test_png_explicit(self):
        assert path_with_figure_format("dir/heatmap.pdf", "png") == "dir/heatmap.png"
