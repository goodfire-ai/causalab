"""
Visualization utilities for activation patching experiments.

This module provides unified visualization functions for creating heatmaps and text representations
of intervention results across different experiment types (residual stream, attention heads, etc.).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from typing import List, Optional, Callable, Dict
from collections import Counter
import os


def create_heatmap(
    score_matrix: np.ndarray,
    x_labels: List[str],
    y_labels: List[str],
    title: str,
    save_path: Optional[str] = None,
    x_label: str = "Position",
    y_label: str = "Layer",
    use_custom_bounds: bool = False,
    cbar_label: str = "Accuracy (%)",
    figsize: tuple = None
) -> None:
    """
    Create a heatmap visualization with configurable axes and styling.

    This function creates a seaborn heatmap that can display accuracy scores,
    custom metrics, or any other grid-based data. It automatically handles:
    - NaN values (shown as blank cells)
    - Custom or default color bounds
    - Diverging colormaps for metrics with positive and negative values
    - Percentage formatting for accuracy metrics

    Args:
        score_matrix: 2D numpy array with scores for each (y, x) pair
        x_labels: List of labels for x-axis (columns)
        y_labels: List of labels for y-axis (rows)
        title: Title for the heatmap
        save_path: Optional path to save the heatmap, or None to display it
        x_label: Label for x-axis (default: "Position")
        y_label: Label for y-axis (default: "Layer")
        use_custom_bounds: If True, automatically infer vmin/vmax from data (for custom metrics).
                          If False, uses 0-1 bounds for accuracy metrics.
        cbar_label: Label for the colorbar (default: "Accuracy (%)")
        figsize: Optional tuple (width, height) for figure size. If None, auto-computed.
    """
    # Auto-compute figure size if not provided
    if figsize is None:
        figsize = (max(10, len(x_labels) * 0.8), max(6, len(y_labels) * 0.5))

    plt.figure(figsize=figsize)

    # Determine vmin, vmax, and colormap based on whether we're using custom bounds
    if use_custom_bounds:
        # For custom metrics, infer bounds from the actual data (ignoring NaN)
        data_min = np.nanmin(score_matrix)
        data_max = np.nanmax(score_matrix)

        # Check if we have both positive and negative values
        if data_min < 0 and data_max > 0:
            # Center the colormap at 0 for diverging data
            abs_max = max(abs(data_min), abs(data_max))
            # Add a small margin
            margin = abs_max * 0.05 if abs_max != 0 else 0.1
            vmin = -abs_max - margin
            vmax = abs_max + margin
            # Use a diverging colormap with white at center
            cmap = 'coolwarm_r'  # Red for negative, white at 0, blue for positive
            center = 0
        else:
            # For all-positive or all-negative data, use standard bounds
            margin = (data_max - data_min) * 0.05 if data_max != data_min else 0.1
            vmin = data_min - margin
            vmax = data_max + margin
            cmap = 'Reds'  # Use red-based colormap
            center = None

        # Don't multiply by 100 for custom metrics
        display_matrix = np.round(score_matrix, 4)
    else:
        # Default bounds for accuracy metrics (0-1)
        vmin = 0
        vmax = 1
        cmap = 'viridis'
        center = None
        # Multiply by 100 for percentage display
        display_matrix = np.round(score_matrix * 100, 2)

    # Create annotation matrix - show values for non-NaN, empty string for NaN
    annot_matrix = np.where(np.isnan(score_matrix), "", display_matrix.astype(str))

    # Create the heatmap using seaborn
    sns.heatmap(
        score_matrix,
        xticklabels=x_labels,
        yticklabels=y_labels,
        cmap=cmap,
        annot=annot_matrix,
        fmt="",  # Use string format since we're providing custom annotations
        cbar_kws={'label': cbar_label},
        vmin=vmin,
        vmax=vmax,
        cbar=True,
        center=center
    )

    plt.yticks(rotation=0)
    plt.xticks(rotation=0)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.tight_layout()

    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)

    plt.show()


def create_binary_mask_heatmap(
    mask_matrix: np.ndarray,
    x_labels: List[str],
    y_labels: List[str],
    title: str,
    save_path: Optional[str] = None,
    x_label: str = "Head",
    y_label: str = "Layer",
    figsize: tuple = None
) -> None:
    """
    Create a binary mask heatmap with custom colors.

    This function visualizes binary data (0 or 1) with distinct colors:
    - Light red for 0 (not selected)
    - Dark blue for 1 (selected)
    - Light gray for NaN (missing data)

    Args:
        mask_matrix: 2D numpy array with binary values (0, 1, or NaN)
        x_labels: List of labels for x-axis
        y_labels: List of labels for y-axis
        title: Title for the heatmap
        save_path: Optional path to save the heatmap
        x_label: Label for x-axis (default: "Head")
        y_label: Label for y-axis (default: "Layer")
        figsize: Optional tuple (width, height) for figure size. If None, auto-computed.
    """
    # Auto-compute figure size if not provided
    if figsize is None:
        figsize = (max(12, len(x_labels) * 0.6), max(6, len(y_labels) * 0.8))

    plt.figure(figsize=figsize)

    # Create custom colormap: light red for 0, dark blue for 1
    from matplotlib.colors import ListedColormap
    colors = ['#FFB3BA', '#00008B']  # Light red for 0, dark blue for 1
    cmap = ListedColormap(colors)

    # Create annotation matrix - show "0" and "1" for valid values, empty for NaN
    annot_matrix = np.where(np.isnan(mask_matrix), "", mask_matrix.astype(int).astype(str))

    # Create the heatmap
    sns.heatmap(
        mask_matrix,
        xticklabels=x_labels,
        yticklabels=y_labels,
        cmap=cmap,
        annot=annot_matrix,
        fmt="",
        cbar_kws={'label': 'Mask Value', 'ticks': [0, 1]},
        vmin=0,
        vmax=1,
        cbar=True,
        linewidths=0.5,
        linecolor='lightgray'
    )

    plt.yticks(rotation=0)
    plt.xticks(rotation=0)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)

    plt.show()
    plt.close()


def create_text_output_grid(
    text_matrix: List[List[str]],
    x_labels: List[str],
    y_labels: List[str],
    title: str,
    save_path: Optional[str] = None,
    x_label: str = "Token Position",
    y_label: str = "Layer",
    figsize: tuple = None,
    fontsize: int = 42
) -> None:
    """
    Create a grid visualization displaying text outputs with frequency-based coloring.

    This function creates a table-like visualization where:
    - Cells contain text outputs from the model
    - Top 5 most frequent outputs get unique light colors
    - A legend shows the color mapping
    - Empty or whitespace strings are shown in quotes

    Args:
        text_matrix: 2D list of strings (y x x dimensions)
        x_labels: List of labels for x-axis (columns)
        y_labels: List of labels for y-axis (rows)
        title: Title for the grid
        save_path: Optional path to save the visualization
        x_label: Label for x-axis (default: "Token Position")
        y_label: Label for y-axis (default: "Layer")
        figsize: Optional tuple (width, height) for figure size. If None, auto-computed.
        fontsize: Font size for text in cells (default: 42)
    """
    # Auto-compute figure size if not provided
    if figsize is None:
        fig_width = max(16, len(x_labels) * 2 + 4)  # Added space for legend
        fig_height = max(8, len(y_labels) * 0.8)
        figsize = (fig_width, fig_height)

    fig, ax = plt.subplots(figsize=figsize)

    # Hide axes
    ax.set_xlim(0, len(x_labels))
    ax.set_ylim(0, len(y_labels))
    ax.axis('off')

    # Count output frequencies
    output_counter = Counter()
    for row in text_matrix:
        for text in row:
            if text:
                output_counter[text] += 1

    # Define light colors for the top 5 most frequent outputs
    light_colors = [
        '#FFFFE0',  # Light yellow
        '#FFE4E1',  # Light red/pink
        '#E0FFE0',  # Light green
        '#E0E0FF',  # Light blue
        '#F5DEB3',  # Light brown/wheat
    ]

    # Get top 5 most frequent outputs and create color mapping
    top_outputs = [output for output, _ in output_counter.most_common(5)]
    output_color_map = {output: light_colors[i] for i, output in enumerate(top_outputs)}
    default_color = 'white'  # For outputs not in top 5

    # Create the table/grid
    cell_height = 0.8 / len(y_labels)
    cell_width = 0.9 / len(x_labels)

    for i, y_label_val in enumerate(y_labels):
        for j, x_label_val in enumerate(x_labels):
            # Calculate cell position
            x = 0.05 + j * cell_width
            y = 0.1 + i * cell_height

            # Get the text output for this cell
            text = text_matrix[i][j] if i < len(text_matrix) and j < len(text_matrix[i]) else ""

            # Determine the background color based on the output
            cell_color = output_color_map.get(text, default_color)

            # Create a rectangle for the cell with appropriate color
            rect = mpatches.Rectangle((x, y), cell_width, cell_height,
                                    linewidth=1, edgecolor='black',
                                    facecolor=cell_color, transform=ax.transAxes)
            ax.add_patch(rect)

            # Check if text is just whitespace and add quotes if so
            display_text = text
            if not text or text.strip() == '':
                display_text = f'"{text}"'

            # Wrap text if it's too long
            max_chars_per_line = int(cell_width * 100)  # Rough estimate
            if len(display_text) > max_chars_per_line:
                # Simple text wrapping
                words = display_text.split()
                wrapped_lines = []
                current_line = []
                current_length = 0

                for word in words:
                    if current_length + len(word) + 1 > max_chars_per_line:
                        wrapped_lines.append(' '.join(current_line))
                        current_line = [word]
                        current_length = len(word)
                    else:
                        current_line.append(word)
                        current_length += len(word) + 1

                if current_line:
                    wrapped_lines.append(' '.join(current_line))

                display_text = '\n'.join(wrapped_lines[:3])  # Limit to 3 lines
                if len(wrapped_lines) > 3:
                    display_text += '...'

            ax.text(x + cell_width/2, y + cell_height/2, display_text,
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=fontsize, wrap=True)

    # Add y-axis labels
    for i, y_label_val in enumerate(y_labels):
        ax.text(0.02, 0.1 + i * cell_height + cell_height/2, y_label_val,
               ha='right', va='center', transform=ax.transAxes, fontsize=fontsize + 6)

    # Add x-axis labels
    for j, x_label_val in enumerate(x_labels):
        # Truncate long labels
        label = str(x_label_val)
        if len(label) > 15:
            label = label[:12] + '...'

        ax.text(0.05 + j * cell_width + cell_width/2, 0.05, label,
               ha='center', va='top', transform=ax.transAxes, fontsize=fontsize,
               rotation=45 if len(label) > 5 else 0)

    # Add title
    ax.text(0.5, 0.98, title,
           ha='center', va='top', transform=ax.transAxes,
           fontsize=fontsize + 12, weight='bold')

    # Add axis labels
    ax.text(0.5, -0.1, x_label, ha='center', va='bottom',
           transform=ax.transAxes, fontsize=fontsize + 6)
    ax.text(-0.1, 0.5, y_label, ha='center', va='center', rotation=90,
           transform=ax.transAxes, fontsize=fontsize + 6)

    # Add legend to the right side if there are colored outputs
    if output_color_map:
        legend_x = 1.1  # Further right, outside the main plot area
        legend_y_start = 0.7  # Starting y position for legend
        legend_spacing = 0.08  # Spacing between legend items

        # Add legend title
        ax.text(legend_x, legend_y_start + legend_spacing, 'Top 5 Outputs:',
               ha='left', va='top', transform=ax.transAxes,
               fontsize=fontsize + 2, weight='bold')

        # Add legend items
        for i, (output, color) in enumerate(output_color_map.items()):
            y_pos = legend_y_start - i * legend_spacing

            # Create small colored rectangle
            legend_rect = mpatches.Rectangle((legend_x, y_pos - 0.03), 0.04, 0.05,
                                           linewidth=1, edgecolor='black',
                                           facecolor=color, transform=ax.transAxes)
            ax.add_patch(legend_rect)

            # Add text label - truncate if too long
            label_text = output if len(output) <= 20 else output[:17] + '...'
            # Show quotes for empty/whitespace strings
            if not output or output.strip() == '':
                label_text = f'"{output}"'

            ax.text(legend_x + 0.05, y_pos, label_text,
                   ha='left', va='center', transform=ax.transAxes, fontsize=fontsize - 2)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=150)

    plt.show()


def print_text_heatmap(
    score_matrix: np.ndarray,
    x_labels: List[str],
    y_labels: List[str],
    title: str,
    save_path: Optional[str] = None,
    x_label: str = "Position",
    y_label: str = "Layer",
    format_fn: Optional[Callable[[float], str]] = None,
    include_stats: bool = True
) -> str:
    """
    Print a text-based representation of a heatmap with full matrix display.

    This function creates a simple, readable text representation of any score matrix,
    suitable for binary heatmaps, percentage scores, or custom metrics. It displays
    the full matrix in a table format and optionally includes summary statistics.

    Args:
        score_matrix: 2D numpy array with scores for each (y, x) pair
        x_labels: List of labels for x-axis (columns)
        y_labels: List of labels for y-axis (rows)
        title: Title for the analysis
        save_path: Optional path to save the text output to a file
        x_label: Label for x-axis (default: "Position")
        y_label: Label for y-axis (default: "Layer")
        format_fn: Optional function to format cell values. If None, formats as percentages.
                  Example: lambda x: f"{x:.1%}" for percentages
                           lambda x: f"{x:.4f}" for raw values
                           lambda x: str(int(x)) for binary values
        include_stats: If True, includes summary statistics (default: True)

    Returns:
        The full output text as a string
    """
    output_lines = []
    output_lines.append("=" * 80)
    output_lines.append(title.upper())
    output_lines.append("=" * 80)

    # Use default percentage formatter if none provided
    if format_fn is None:
        format_fn = lambda x: f"{x:.1%}" if not np.isnan(x) else "N/A"

    # Calculate column widths
    max_label_width = max(len(str(label)) for label in x_labels)
    max_value_width = max(len(format_fn(val)) for row in score_matrix for val in row if not np.isnan(val))
    col_width = max(max_label_width, max_value_width, 8) + 2

    y_label_width = max(len(str(label)) for label in y_labels) + 2

    # Print header
    output_lines.append(f"\n{y_label} vs {x_label}:\n")

    # Print column headers
    header = " " * y_label_width + "|"
    for x_label_val in x_labels:
        header += f" {str(x_label_val):^{col_width}} |"
    output_lines.append(header)
    output_lines.append("-" * len(header))

    # Print each row
    for i, y_label_val in enumerate(y_labels):
        row_str = f"{str(y_label_val):>{y_label_width}}|"
        for j, x_label_val in enumerate(x_labels):
            value = score_matrix[i, j]
            formatted_value = format_fn(value) if not np.isnan(value) else "N/A"
            row_str += f" {formatted_value:^{col_width}} |"
        output_lines.append(row_str)

    # Add summary statistics if requested
    if include_stats:
        output_lines.append("\n" + "=" * 80)
        output_lines.append("SUMMARY STATISTICS")
        output_lines.append("=" * 80)

        # Filter out NaN values for statistics
        valid_values = score_matrix[~np.isnan(score_matrix)]

        if len(valid_values) > 0:
            output_lines.append(f"\nTotal cells: {score_matrix.size}")
            output_lines.append(f"Valid cells: {len(valid_values)}")
            output_lines.append(f"Missing cells (NaN): {score_matrix.size - len(valid_values)}")
            output_lines.append(f"\nMin:    {format_fn(valid_values.min())}")
            output_lines.append(f"Max:    {format_fn(valid_values.max())}")
            output_lines.append(f"Mean:   {format_fn(valid_values.mean())}")
            output_lines.append(f"Median: {format_fn(np.median(valid_values))}")
            output_lines.append(f"Std:    {format_fn(valid_values.std())}")

            # Find best (max) cell
            max_idx = np.unravel_index(np.nanargmax(score_matrix), score_matrix.shape)
            best_y = y_labels[max_idx[0]]
            best_x = x_labels[max_idx[1]]
            output_lines.append(f"\nBest cell: {y_label}={best_y}, {x_label}={best_x} ({format_fn(score_matrix[max_idx])})")

            # Find worst (min) cell
            min_idx = np.unravel_index(np.nanargmin(score_matrix), score_matrix.shape)
            worst_y = y_labels[min_idx[0]]
            worst_x = x_labels[min_idx[1]]
            output_lines.append(f"Worst cell: {y_label}={worst_y}, {x_label}={best_x} ({format_fn(score_matrix[min_idx])})")
        else:
            output_lines.append("\nNo valid data available (all values are NaN)")

    output_lines.append("\n" + "=" * 80)
    output_lines.append("END OF ANALYSIS")
    output_lines.append("=" * 80)

    # Join all lines
    full_output = "\n".join(output_lines)

    # Print to console
    print(full_output)

    # Optionally save to file
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            f.write(full_output)
        print(f"\nText analysis saved to: {save_path}")

    return full_output
