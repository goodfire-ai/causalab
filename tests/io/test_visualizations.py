"""
Test suite for causalab/io/plots — mask/feature-count/heatmap visualizations.

Since the where-unification the mask and feature-count plots consume
structured [`GridCell`][causalab.io.plots.grid_cells.GridCell] records joined
from a grid's own specs; the unit-id string parsing they used to rely on is
retired (``causalab/io/plots/unit_id.py`` was deleted).
"""

import pytest
import numpy as np
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
from unittest.mock import patch
import tempfile
import os

from causalab.io.plots import (
    GridCell,
    cell_grid_dimensions,
    get_selected_heads,
    plot_attention_head_heatmap,
    plot_attention_head_mask,
    plot_binary_mask,
    plot_feature_counts,
    plot_residual_stream_heatmap,
)
from causalab.io.plots.utils import (
    create_heatmap,
    create_binary_mask_heatmap,
)


pytestmark = pytest.mark.unit


def _head_cell(layer: int, head: int, indices=None, n_features=None) -> GridCell:
    return GridCell(
        key=f"k.L{layer}.H{head}",
        layer=layer,
        head=head,
        indices=indices,
        n_features=n_features,
    )


def _pos_cell(layer: int, position: str, indices=None, n_features=None) -> GridCell:
    return GridCell(
        key=f"k.L{layer}.{position}",
        layer=layer,
        position=position,
        indices=indices,
        n_features=n_features,
    )


class TestGetSelectedHeads:
    """Tests for the get_selected_heads function (structured cells)."""

    def test_all_selected(self):
        cells = [_head_cell(0, 0), _head_cell(0, 1), _head_cell(1, 0)]
        assert get_selected_heads(cells) == [(0, 0), (0, 1), (1, 0)]

    def test_none_selected(self):
        cells = [_head_cell(0, 0, indices=()), _head_cell(0, 1, indices=())]
        assert get_selected_heads(cells) == []

    def test_mixed_selection(self):
        cells = [
            _head_cell(0, 0),  # selected
            _head_cell(0, 1, indices=()),  # not selected
            _head_cell(1, 0),  # selected
            _head_cell(1, 1, indices=()),  # not selected
        ]
        assert get_selected_heads(cells) == [(0, 0), (1, 0)]

    def test_sorted_output(self):
        cells = [
            _head_cell(2, 1),
            _head_cell(0, 3),
            _head_cell(1, 0),
            _head_cell(0, 1),
        ]
        assert get_selected_heads(cells) == [(0, 1), (0, 3), (1, 0), (2, 1)]

    def test_ignores_position_cells(self):
        cells = [_head_cell(0, 0), _pos_cell(0, "last"), _pos_cell(1, "last")]
        assert get_selected_heads(cells) == [(0, 0)]


# ---------------------- Tests for plot_attention_head_heatmap ---------------------- #


class TestPlotAttentionHeadHeatmap:
    """Tests for plot_attention_head_heatmap function."""

    @pytest.fixture
    def sample_scores(self):
        """Sample attention head scores."""
        return {
            (0, 0): 0.8,
            (0, 1): 0.6,
            (1, 0): 0.9,
            (1, 1): 0.4,
        }

    def test_basic_heatmap(self, sample_scores):
        """Basic heatmap creation succeeds (uses create_heatmap's xlabel/ylabel)."""
        with patch("matplotlib.pyplot.show"):
            plot_attention_head_heatmap(
                scores=sample_scores,
                layers=[0, 1],
                heads=[0, 1],
                title="Test Heatmap",
            )
        plt.close("all")

    def test_heatmap_with_save(self, sample_scores):
        """Heatmap with file saving writes the figure to disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "test_heatmap.png")
            with patch("matplotlib.pyplot.show"):
                plot_attention_head_heatmap(
                    scores=sample_scores,
                    layers=[0, 1],
                    heads=[0, 1],
                    save_path=save_path,
                    figure_format="png",
                )
            assert os.path.exists(save_path)
        plt.close("all")

    def test_heatmap_with_missing_scores(self):
        """Missing cells render as NaN gaps without error."""
        scores = {
            (0, 0): 0.8,
            (1, 1): 0.6,
            # (0, 1) and (1, 0) are missing
        }
        with patch("matplotlib.pyplot.show"):
            plot_attention_head_heatmap(
                scores=scores,
                layers=[0, 1],
                heads=[0, 1],
            )
        plt.close("all")


# ---------------------- Tests for plot_attention_head_mask ---------------------- #


class TestPlotAttentionHeadMask:
    """Tests for plot_attention_head_mask function (structured cells)."""

    @pytest.fixture
    def sample_cells(self):
        """Sample grid cells for mask plotting."""
        return [
            _head_cell(0, 0),  # selected
            _head_cell(0, 1, indices=()),  # not selected
            _head_cell(1, 0, indices=()),  # not selected
            _head_cell(1, 1),  # selected
        ]

    def test_basic_mask(self, sample_cells):
        """Test basic mask heatmap creation."""
        with patch("matplotlib.pyplot.show"):
            plot_attention_head_mask(
                cells=sample_cells,
                layers=[0, 1],
                heads=[0, 1],
                title="Test Mask",
            )
        plt.close("all")

    def test_mask_with_save(self, sample_cells):
        """Test mask heatmap with file saving."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "test_mask.png")
            with patch("matplotlib.pyplot.show"):
                plot_attention_head_mask(
                    cells=sample_cells,
                    layers=[0, 1],
                    heads=[0, 1],
                    save_path=save_path,
                    figure_format="png",
                )
            assert os.path.exists(save_path)
        plt.close("all")

    def test_unified_dispatcher_saves(self, sample_cells):
        """plot_binary_mask dispatches on the caller's component_type."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "mask.png")
            with patch("matplotlib.pyplot.show"):
                plot_binary_mask(
                    "attention_head",
                    sample_cells,
                    save_path=save_path,
                    figure_format="png",
                )
            assert os.path.exists(save_path)
        plt.close("all")


class TestPlotFeatureCounts:
    """Feature-count dispatcher over structured cells."""

    def test_residual_counts_save(self):
        cells = [
            _pos_cell(0, "last", indices=(1, 2, 3), n_features=8),
            _pos_cell(1, "last", indices=None, n_features=8),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "counts.png")
            with patch("matplotlib.pyplot.show"):
                plot_feature_counts(
                    "residual_stream",
                    cells,
                    scores=0.75,
                    save_path=save_path,
                    figure_format="png",
                )
            assert os.path.exists(save_path)
        plt.close("all")

    def test_all_selected_without_n_features_raises(self):
        cells = [_pos_cell(0, "last", indices=None, n_features=None)]
        with pytest.raises(ValueError, match="n_features"):
            plot_feature_counts("residual_stream", cells, scores=0.5)
        plt.close("all")


# ---------------------- Tests for plot_residual_stream_heatmap ---------------------- #


class TestPlotResidualStreamHeatmap:
    """Tests for plot_residual_stream_heatmap function."""

    @pytest.fixture
    def sample_residual_scores(self):
        """Sample residual stream scores."""
        return {
            (0, "pos_0"): 0.7,
            (0, "pos_1"): 0.8,
            (1, "pos_0"): 0.6,
            (1, "pos_1"): 0.9,
        }

    def test_basic_residual_heatmap(self, sample_residual_scores):
        """Test basic residual stream heatmap creation."""
        with patch("matplotlib.pyplot.show"):
            plot_residual_stream_heatmap(
                scores=sample_residual_scores,
                layers=[0, 1],
                token_position_ids=["pos_0", "pos_1"],
                title="Test Residual Heatmap",
            )
        plt.close("all")

    def test_residual_heatmap_with_embeddings(self):
        """Test residual heatmap including embedding layer (-1)."""
        scores = {
            (-1, "pos_0"): 0.5,
            (0, "pos_0"): 0.7,
            (1, "pos_0"): 0.9,
        }
        with patch("matplotlib.pyplot.show"):
            plot_residual_stream_heatmap(
                scores=scores,
                layers=[-1, 0, 1],
                token_position_ids=["pos_0"],
            )
        plt.close("all")

    def test_residual_heatmap_with_save(self, sample_residual_scores):
        """Test residual heatmap with file saving."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "test_residual.png")
            with patch("matplotlib.pyplot.show"):
                plot_residual_stream_heatmap(
                    scores=sample_residual_scores,
                    layers=[0, 1],
                    token_position_ids=["pos_0", "pos_1"],
                    save_path=save_path,
                    figure_format="png",
                )
            assert os.path.exists(save_path)
        plt.close("all")


# ---------------------- Tests for utils functions ---------------------- #


class TestCreateHeatmap:
    """Tests for the create_heatmap utility function."""

    @pytest.fixture
    def sample_matrix(self):
        """Sample score matrix."""
        return np.array([[0.8, 0.6], [0.7, 0.5]])

    def test_basic_heatmap(self, sample_matrix):
        """Test basic heatmap creation."""
        with patch("matplotlib.pyplot.show"):
            create_heatmap(
                score_matrix=sample_matrix,
                x_labels=["X0", "X1"],
                y_labels=["Y0", "Y1"],
                title="Test",
            )
        plt.close("all")

    def test_heatmap_with_nan(self):
        """Test heatmap with NaN values."""
        matrix = np.array([[0.8, np.nan], [np.nan, 0.5]])
        with patch("matplotlib.pyplot.show"):
            create_heatmap(
                score_matrix=matrix,
                x_labels=["X0", "X1"],
                y_labels=["Y0", "Y1"],
            )
        plt.close("all")

    def test_heatmap_save(self, sample_matrix):
        """Test heatmap saving."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "heatmap.png")
            with patch("matplotlib.pyplot.show"):
                create_heatmap(
                    score_matrix=sample_matrix,
                    x_labels=["X0", "X1"],
                    y_labels=["Y0", "Y1"],
                    save_path=save_path,
                    figure_format="png",
                )
            assert os.path.exists(save_path)
        plt.close("all")


class TestCreateBinaryMaskHeatmap:
    """Tests for the create_binary_mask_heatmap utility function."""

    @pytest.fixture
    def sample_mask(self):
        """Sample binary mask matrix."""
        return np.array([[1, 0], [0, 1]], dtype=float)

    def test_basic_mask_heatmap(self, sample_mask):
        """Test basic binary mask heatmap creation."""
        with patch("matplotlib.pyplot.show"):
            create_binary_mask_heatmap(
                mask_matrix=sample_mask,
                x_labels=["X0", "X1"],
                y_labels=["Y0", "Y1"],
                title="Test Mask",
            )
        plt.close("all")

    def test_mask_heatmap_with_nan(self):
        """Test mask heatmap with NaN values."""
        mask = np.array([[1, np.nan], [np.nan, 0]])
        with patch("matplotlib.pyplot.show"):
            create_binary_mask_heatmap(
                mask_matrix=mask,
                x_labels=["X0", "X1"],
                y_labels=["Y0", "Y1"],
            )
        plt.close("all")

    def test_mask_heatmap_save(self, sample_mask):
        """Test mask heatmap saving."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "mask.png")
            with patch("matplotlib.pyplot.show"):
                create_binary_mask_heatmap(
                    mask_matrix=sample_mask,
                    x_labels=["X0", "X1"],
                    y_labels=["Y0", "Y1"],
                    save_path=save_path,
                    figure_format="png",
                )
            assert os.path.exists(save_path)
        plt.close("all")


# ---------------------- Tests for cell_grid_dimensions ---------------------- #


class TestCellGridDimensions:
    """Axis extraction from structured cells (the extract_grid_dimensions
    successor — same ordering contracts, no id parsing)."""

    def test_token_position_ordering_preserved(self):
        """Token position order is first-seen, not alphabetically sorted."""
        cells = [
            _pos_cell(0, "zebra"),
            _pos_cell(0, "alpha"),
            _pos_cell(0, "middle"),
        ]
        dims = cell_grid_dimensions("residual_stream", cells)
        # Should preserve insertion order, not alphabetical
        assert dims["token_position_ids"] == ["zebra", "alpha", "middle"]
        # Layers should still be sorted
        assert dims["layers"] == [0]

    def test_token_position_ordering_with_multiple_layers(self):
        cells = [
            _pos_cell(2, "last_token"),
            _pos_cell(0, "first_token"),
            _pos_cell(1, "middle_token"),
            _pos_cell(2, "first_token"),
            _pos_cell(0, "last_token"),
        ]
        dims = cell_grid_dimensions("residual_stream", cells)
        # Token positions should preserve first-seen order
        assert dims["token_position_ids"] == [
            "last_token",
            "first_token",
            "middle_token",
        ]
        # Layers should be numerically sorted
        assert dims["layers"] == [0, 1, 2]

    def test_mlp_token_position_ordering(self):
        cells = [
            _pos_cell(0, "pos_c"),
            _pos_cell(0, "pos_a"),
            _pos_cell(0, "pos_b"),
        ]
        dims = cell_grid_dimensions("mlp", cells)
        assert dims["token_position_ids"] == ["pos_c", "pos_a", "pos_b"]

    def test_attention_head_dimensions(self):
        cells = [
            _head_cell(2, 5),
            _head_cell(0, 3),
            _head_cell(1, 1),
        ]
        dims = cell_grid_dimensions("attention_head", cells)
        # Both layers and heads should be sorted for attention
        assert dims["layers"] == [0, 1, 2]
        assert dims["heads"] == [1, 3, 5]
