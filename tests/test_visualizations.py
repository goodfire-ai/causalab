"""
Tests for the consolidated visualization functions.

These tests verify that the visualization functions in experiments/visualizations.py
work correctly for creating heatmaps, text output grids, and text-based analyses.
"""

import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock
import tempfile
import os

from experiments.visualizations import (
    create_heatmap,
    create_binary_mask_heatmap,
    create_text_output_grid,
    print_text_heatmap
)


class TestCreateHeatmap:
    """Tests for create_heatmap function."""

    @pytest.fixture
    def sample_score_matrix(self):
        """Create a sample score matrix for testing."""
        return np.array([
            [0.8, 0.6, 0.4],
            [0.7, 0.5, 0.3],
            [0.9, 0.2, 0.1]
        ])

    @pytest.fixture
    def sample_labels(self):
        """Create sample axis labels."""
        return {
            'x_labels': ['pos0', 'pos1', 'pos2'],
            'y_labels': ['L0', 'L1', 'L2']
        }

    def test_basic_heatmap_creation(self, sample_score_matrix, sample_labels):
        """Test basic heatmap creation without saving."""
        with patch('matplotlib.pyplot.show'):
            create_heatmap(
                score_matrix=sample_score_matrix,
                x_labels=sample_labels['x_labels'],
                y_labels=sample_labels['y_labels'],
                title="Test Heatmap"
            )

        # If we get here without errors, the function works
        plt.close('all')

    def test_heatmap_with_save_path(self, sample_score_matrix, sample_labels):
        """Test heatmap creation with file saving."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "test_heatmap.png")

            with patch('matplotlib.pyplot.show'):
                create_heatmap(
                    score_matrix=sample_score_matrix,
                    x_labels=sample_labels['x_labels'],
                    y_labels=sample_labels['y_labels'],
                    title="Test Heatmap",
                    save_path=save_path
                )

            # Verify file was created
            assert os.path.exists(save_path)
            plt.close('all')

    def test_heatmap_with_nan_values(self, sample_labels):
        """Test heatmap handling of NaN values."""
        score_matrix_with_nan = np.array([
            [0.8, np.nan, 0.4],
            [0.7, 0.5, np.nan],
            [np.nan, 0.2, 0.1]
        ])

        with patch('matplotlib.pyplot.show'):
            create_heatmap(
                score_matrix=score_matrix_with_nan,
                x_labels=sample_labels['x_labels'],
                y_labels=sample_labels['y_labels'],
                title="Test Heatmap with NaN"
            )

        plt.close('all')

    def test_heatmap_with_custom_bounds(self, sample_labels):
        """Test heatmap with custom bounds (for non-accuracy metrics)."""
        # Create a matrix with values outside 0-1 range
        score_matrix = np.array([
            [-0.5, 0.0, 0.5],
            [1.0, 1.5, 2.0],
            [-1.0, 0.5, 1.0]
        ])

        with patch('matplotlib.pyplot.show'):
            create_heatmap(
                score_matrix=score_matrix,
                x_labels=sample_labels['x_labels'],
                y_labels=sample_labels['y_labels'],
                title="Test Custom Bounds",
                use_custom_bounds=True
            )

        plt.close('all')

    def test_heatmap_with_diverging_colormap(self, sample_labels):
        """Test heatmap with diverging data (positive and negative values)."""
        score_matrix = np.array([
            [-0.8, -0.2, 0.1],
            [0.3, 0.5, 0.7],
            [-0.5, 0.0, 0.9]
        ])

        with patch('matplotlib.pyplot.show'):
            create_heatmap(
                score_matrix=score_matrix,
                x_labels=sample_labels['x_labels'],
                y_labels=sample_labels['y_labels'],
                title="Test Diverging",
                use_custom_bounds=True
            )

        plt.close('all')

    def test_heatmap_with_custom_labels(self, sample_score_matrix):
        """Test heatmap with custom axis labels."""
        with patch('matplotlib.pyplot.show'):
            create_heatmap(
                score_matrix=sample_score_matrix,
                x_labels=['A', 'B', 'C'],
                y_labels=['X', 'Y', 'Z'],
                title="Test Custom Labels",
                x_label="Columns",
                y_label="Rows"
            )

        plt.close('all')


class TestCreateBinaryMaskHeatmap:
    """Tests for create_binary_mask_heatmap function."""

    @pytest.fixture
    def sample_mask_matrix(self):
        """Create a sample binary mask matrix."""
        return np.array([
            [1, 0, 1],
            [0, 1, 0],
            [1, 1, 0]
        ], dtype=float)

    @pytest.fixture
    def sample_labels(self):
        """Create sample axis labels."""
        return {
            'x_labels': ['H0', 'H1', 'H2'],
            'y_labels': ['L0', 'L1', 'L2']
        }

    def test_basic_mask_heatmap(self, sample_mask_matrix, sample_labels):
        """Test basic binary mask heatmap creation."""
        with patch('matplotlib.pyplot.show'):
            create_binary_mask_heatmap(
                mask_matrix=sample_mask_matrix,
                x_labels=sample_labels['x_labels'],
                y_labels=sample_labels['y_labels'],
                title="Test Mask"
            )

        plt.close('all')

    def test_mask_heatmap_with_nan(self, sample_labels):
        """Test mask heatmap with NaN values."""
        mask_matrix = np.array([
            [1, np.nan, 0],
            [0, 1, np.nan],
            [np.nan, 0, 1]
        ])

        with patch('matplotlib.pyplot.show'):
            create_binary_mask_heatmap(
                mask_matrix=mask_matrix,
                x_labels=sample_labels['x_labels'],
                y_labels=sample_labels['y_labels'],
                title="Test Mask with NaN"
            )

        plt.close('all')

    def test_mask_heatmap_save(self, sample_mask_matrix, sample_labels):
        """Test saving mask heatmap to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "test_mask.png")

            with patch('matplotlib.pyplot.show'):
                create_binary_mask_heatmap(
                    mask_matrix=sample_mask_matrix,
                    x_labels=sample_labels['x_labels'],
                    y_labels=sample_labels['y_labels'],
                    title="Test Mask",
                    save_path=save_path
                )

            assert os.path.exists(save_path)
            plt.close('all')


class TestCreateTextOutputGrid:
    """Tests for create_text_output_grid function."""

    @pytest.fixture
    def sample_text_matrix(self):
        """Create a sample text matrix."""
        return [
            ["A", "B", "C"],
            ["B", "A", "A"],
            ["C", "C", "B"]
        ]

    @pytest.fixture
    def sample_labels(self):
        """Create sample axis labels."""
        return {
            'x_labels': ['pos0', 'pos1', 'pos2'],
            'y_labels': ['L0', 'L1', 'L2']
        }

    def test_basic_text_grid(self, sample_text_matrix, sample_labels):
        """Test basic text output grid creation."""
        with patch('matplotlib.pyplot.show'):
            create_text_output_grid(
                text_matrix=sample_text_matrix,
                x_labels=sample_labels['x_labels'],
                y_labels=sample_labels['y_labels'],
                title="Test Grid"
            )

        plt.close('all')

    def test_text_grid_with_empty_strings(self, sample_labels):
        """Test text grid with empty strings."""
        text_matrix = [
            ["A", "", "C"],
            ["", "B", ""],
            ["D", "E", ""]
        ]

        with patch('matplotlib.pyplot.show'):
            create_text_output_grid(
                text_matrix=text_matrix,
                x_labels=sample_labels['x_labels'],
                y_labels=sample_labels['y_labels'],
                title="Test with Empty Strings"
            )

        plt.close('all')

    def test_text_grid_with_long_strings(self, sample_labels):
        """Test text grid with long output strings."""
        text_matrix = [
            ["This is a very long output string", "Short", "Medium length"],
            ["Another long string that needs wrapping", "A", "B"],
            ["C", "D", "Yet another extremely long output that should wrap"]
        ]

        with patch('matplotlib.pyplot.show'):
            create_text_output_grid(
                text_matrix=text_matrix,
                x_labels=sample_labels['x_labels'],
                y_labels=sample_labels['y_labels'],
                title="Test Long Strings"
            )

        plt.close('all')

    def test_text_grid_save(self, sample_text_matrix, sample_labels):
        """Test saving text grid to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "test_grid.png")

            with patch('matplotlib.pyplot.show'):
                create_text_output_grid(
                    text_matrix=sample_text_matrix,
                    x_labels=sample_labels['x_labels'],
                    y_labels=sample_labels['y_labels'],
                    title="Test Grid",
                    save_path=save_path
                )

            assert os.path.exists(save_path)
            plt.close('all')


class TestPrintTextHeatmap:
    """Tests for print_text_heatmap function."""

    @pytest.fixture
    def sample_score_matrix(self):
        """Create a sample score matrix."""
        return np.array([
            [0.8, 0.6, 0.4],
            [0.7, 0.5, 0.3],
            [0.9, 0.2, 0.1]
        ])

    @pytest.fixture
    def sample_labels(self):
        """Create sample axis labels."""
        return {
            'x_labels': ['pos0', 'pos1', 'pos2'],
            'y_labels': ['L0', 'L1', 'L2']
        }

    def test_basic_text_heatmap(self, sample_score_matrix, sample_labels, capsys):
        """Test basic text heatmap printing."""
        output = print_text_heatmap(
            score_matrix=sample_score_matrix,
            x_labels=sample_labels['x_labels'],
            y_labels=sample_labels['y_labels'],
            title="Test Analysis"
        )

        # Verify output is a string
        assert isinstance(output, str)
        assert "TEST ANALYSIS" in output
        assert "L0" in output
        assert "pos0" in output

        # Verify it was printed
        captured = capsys.readouterr()
        assert "TEST ANALYSIS" in captured.out

    def test_text_heatmap_with_custom_formatter(self, sample_score_matrix, sample_labels):
        """Test text heatmap with custom formatting function."""
        # Custom formatter that shows raw values with 2 decimal places
        format_fn = lambda x: f"{x:.2f}" if not np.isnan(x) else "N/A"

        output = print_text_heatmap(
            score_matrix=sample_score_matrix,
            x_labels=sample_labels['x_labels'],
            y_labels=sample_labels['y_labels'],
            title="Custom Format Test",
            format_fn=format_fn
        )

        assert "0.80" in output or "0.8" in output

    def test_text_heatmap_with_nan_values(self, sample_labels):
        """Test text heatmap with NaN values."""
        score_matrix = np.array([
            [0.8, np.nan, 0.4],
            [np.nan, 0.5, 0.3],
            [0.9, 0.2, np.nan]
        ])

        output = print_text_heatmap(
            score_matrix=score_matrix,
            x_labels=sample_labels['x_labels'],
            y_labels=sample_labels['y_labels'],
            title="Test with NaN"
        )

        assert "N/A" in output
        assert "SUMMARY STATISTICS" in output

    def test_text_heatmap_save_to_file(self, sample_score_matrix, sample_labels):
        """Test saving text heatmap to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "test_analysis.txt")

            output = print_text_heatmap(
                score_matrix=sample_score_matrix,
                x_labels=sample_labels['x_labels'],
                y_labels=sample_labels['y_labels'],
                title="Test Save",
                save_path=save_path
            )

            # Verify file was created
            assert os.path.exists(save_path)

            # Verify file contents match output
            with open(save_path, 'r') as f:
                file_contents = f.read()
            assert file_contents == output

    def test_text_heatmap_without_stats(self, sample_score_matrix, sample_labels):
        """Test text heatmap without summary statistics."""
        output = print_text_heatmap(
            score_matrix=sample_score_matrix,
            x_labels=sample_labels['x_labels'],
            y_labels=sample_labels['y_labels'],
            title="No Stats Test",
            include_stats=False
        )

        assert "SUMMARY STATISTICS" not in output

    def test_text_heatmap_binary_formatter(self, sample_labels):
        """Test text heatmap with binary values."""
        binary_matrix = np.array([
            [1, 0, 1],
            [0, 1, 0],
            [1, 1, 0]
        ], dtype=float)

        # Binary formatter
        format_fn = lambda x: str(int(x)) if not np.isnan(x) else "N/A"

        output = print_text_heatmap(
            score_matrix=binary_matrix,
            x_labels=sample_labels['x_labels'],
            y_labels=sample_labels['y_labels'],
            title="Binary Test",
            format_fn=format_fn
        )

        assert "0" in output
        assert "1" in output

    def test_text_heatmap_custom_axis_labels(self, sample_score_matrix):
        """Test text heatmap with custom axis labels."""
        output = print_text_heatmap(
            score_matrix=sample_score_matrix,
            x_labels=['A', 'B', 'C'],
            y_labels=['X', 'Y', 'Z'],
            title="Custom Labels",
            x_label="Columns",
            y_label="Rows"
        )

        assert "Rows vs Columns" in output
        assert "X" in output
        assert "A" in output


class TestVisualizationIntegration:
    """Integration tests for visualization functions."""

    def test_all_functions_with_same_data(self):
        """Test that all visualization functions work with the same data structure."""
        score_matrix = np.array([
            [0.8, 0.6],
            [0.7, 0.5]
        ])
        x_labels = ['pos0', 'pos1']
        y_labels = ['L0', 'L1']

        with tempfile.TemporaryDirectory() as tmpdir:
            # Test regular heatmap
            with patch('matplotlib.pyplot.show'):
                create_heatmap(
                    score_matrix=score_matrix,
                    x_labels=x_labels,
                    y_labels=y_labels,
                    title="Test",
                    save_path=os.path.join(tmpdir, "heatmap.png")
                )
            plt.close('all')

            # Test text heatmap
            print_text_heatmap(
                score_matrix=score_matrix,
                x_labels=x_labels,
                y_labels=y_labels,
                title="Test",
                save_path=os.path.join(tmpdir, "text.txt")
            )

            # Verify both files exist
            assert os.path.exists(os.path.join(tmpdir, "heatmap.png"))
            assert os.path.exists(os.path.join(tmpdir, "text.txt"))


# Run tests when file is executed directly
if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
