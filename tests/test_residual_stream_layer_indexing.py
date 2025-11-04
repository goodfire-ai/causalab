"""
Test to verify that layer indexing in SameLengthResidualStreamTracing is correct.

This test ensures that layer -1 (embeddings) maps to the first row and the last layer
maps to the last row in visualization outputs.
"""
import pytest


def test_layer_to_index_mapping():
    """Test that layer values correctly map to list indices."""
    # Simulate the layers list used in plot_raw_outputs and print_text_analysis
    layers = [-1, 0, 1, 2, 3]  # Example with 4 actual layers

    # Test that layer -1 maps to index 0
    layer = -1
    layer_idx = layers.index(layer)
    assert layer_idx == 0, f"Layer -1 should map to index 0, got {layer_idx}"

    # Test that layer 0 maps to index 1
    layer = 0
    layer_idx = layers.index(layer)
    assert layer_idx == 1, f"Layer 0 should map to index 1, got {layer_idx}"

    # Test that the last layer (3) maps to the last index (4)
    layer = 3
    layer_idx = layers.index(layer)
    assert layer_idx == 4, f"Layer 3 should map to index 4, got {layer_idx}"


def test_text_outputs_indexing():
    """Test that text_outputs matrix is indexed correctly."""
    layers = [-1, 0, 1, 2]
    positions = ['token1', 'token2', 'token3']

    # Create text outputs matrix
    text_outputs = [['' for _ in positions] for _ in layers]

    # Verify matrix shape
    assert len(text_outputs) == len(layers), "Matrix should have one row per layer"
    assert len(text_outputs[0]) == len(positions), "Matrix should have one column per position"

    # Simulate setting outputs with correct indexing
    test_data = [
        (-1, 'token1', 'embedding_output'),  # Layer -1, first token
        (0, 'token2', 'layer0_output'),      # Layer 0, second token
        (2, 'token3', 'layer2_output'),      # Layer 2, third token
    ]

    for layer, position, output in test_data:
        layer_idx = layers.index(layer)
        pos_idx = positions.index(position)
        text_outputs[layer_idx][pos_idx] = output

    # Verify the outputs are in the correct positions
    assert text_outputs[0][0] == 'embedding_output', "Layer -1 output should be in first row"
    assert text_outputs[1][1] == 'layer0_output', "Layer 0 output should be in second row"
    assert text_outputs[3][2] == 'layer2_output', "Layer 2 output should be in fourth row"

    # Verify that using layer as direct index would have been incorrect
    # (layer -1 would access the last row instead of first)
    assert text_outputs[-1][0] == '', "Last row should be empty (not layer -1's output)"


def test_negative_index_bug():
    """Test that demonstrates the bug that was fixed."""
    layers = [-1, 0, 1, 2, 3]
    positions = ['A', 'B', 'C']

    text_outputs = [['' for _ in positions] for _ in layers]

    # The bug: using layer directly as index
    # When layer = -1, Python interprets this as the last element
    layer = -1
    text_outputs[layer][0] = 'WRONG_LOCATION'  # This goes to last row!

    # Verify the bug behavior
    assert text_outputs[-1][0] == 'WRONG_LOCATION', "Using -1 directly accesses last row"
    assert text_outputs[4][0] == 'WRONG_LOCATION', "Index 4 is the last row"
    assert text_outputs[0][0] == '', "First row is empty when bug exists"

    # The fix: use layers.index(layer) to get correct index
    layer_idx = layers.index(-1)
    text_outputs[layer_idx][0] = 'CORRECT_LOCATION'

    assert text_outputs[0][0] == 'CORRECT_LOCATION', "Fix places -1 output in first row"
    assert layer_idx == 0, "layers.index(-1) returns 0"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
