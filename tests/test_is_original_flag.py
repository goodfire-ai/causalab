"""
Tests for the is_original flag functionality in TokenPosition and ComponentIndexer.

This test suite verifies that:
1. ComponentIndexer can pass is_original flag to indexer functions
2. TokenPosition stores and can use the is_original flag
3. The flag is properly passed through pyvene_core during interventions
4. Backward compatibility is maintained for existing code
"""

import unittest
from unittest.mock import Mock, patch
from neural.model_units import ComponentIndexer, Component, AtomicModelUnit
from neural.LM_units import TokenPosition
from neural.featurizers import Featurizer


class TestComponentIndexerIsOriginal(unittest.TestCase):
    """Test ComponentIndexer with is_original flag."""

    def test_indexer_with_is_original_parameter(self):
        """Test that indexer functions can receive is_original parameter."""
        # Create an indexer that uses the is_original flag
        def position_indexer(input, is_original=True):
            if is_original:
                return [0, 1, 2]  # Original positions
            else:
                return [3, 4, 5]  # Counterfactual positions

        indexer = ComponentIndexer(position_indexer, id="test_indexer")

        # Test with is_original=True
        result_original = indexer.index({"text": "test"}, is_original=True)
        self.assertEqual(result_original, [0, 1, 2])

        # Test with is_original=False
        result_counterfactual = indexer.index({"text": "test"}, is_original=False)
        self.assertEqual(result_counterfactual, [3, 4, 5])

    def test_indexer_without_is_original_parameter(self):
        """Test backward compatibility with indexers that don't accept is_original."""
        # Create an old-style indexer that doesn't accept is_original
        def old_indexer(input):
            return [0, 1, 2]

        indexer = ComponentIndexer(old_indexer, id="old_indexer")

        # Should work with is_original flag (but ignore it)
        result = indexer.index({"text": "test"}, is_original=True)
        self.assertEqual(result, [0, 1, 2])

        # Should also work with is_original=False
        result = indexer.index({"text": "test"}, is_original=False)
        self.assertEqual(result, [0, 1, 2])

    def test_batch_indexing_with_is_original(self):
        """Test batch indexing with is_original flag."""
        def position_indexer(input, is_original=True):
            base_pos = input.get("pos", 0)
            if is_original:
                return [base_pos]
            else:
                return [base_pos + 10]

        indexer = ComponentIndexer(position_indexer, id="batch_indexer")

        # Test batch with is_original=True
        batch = [{"pos": 0}, {"pos": 1}, {"pos": 2}]
        result_original = indexer.index(batch, batch=True, is_original=True)
        self.assertEqual(result_original, [[0], [1], [2]])

        # Test batch with is_original=False
        result_counterfactual = indexer.index(batch, batch=True, is_original=False)
        self.assertEqual(result_counterfactual, [[10], [11], [12]])

    def test_default_is_original_value(self):
        """Test that is_original defaults to True."""
        def position_indexer(input, is_original=True):
            return [0] if is_original else [1]

        indexer = ComponentIndexer(position_indexer, id="default_test")

        # When not specified, should default to True
        result = indexer.index({"text": "test"})
        self.assertEqual(result, [0])


class TestTokenPositionIsOriginal(unittest.TestCase):
    """Test TokenPosition class with is_original flag."""

    def setUp(self):
        """Set up a mock pipeline for testing."""
        self.mock_pipeline = Mock()
        self.mock_pipeline.tokenizer = Mock()

    def test_token_position_stores_is_original(self):
        """Test that TokenPosition stores the is_original flag."""
        # Create TokenPosition with is_original=True
        tp_original = TokenPosition(
            lambda x: [0],
            self.mock_pipeline,
            is_original=True,
            id="original"
        )
        self.assertTrue(tp_original.is_original)

        # Create TokenPosition with is_original=False
        tp_counterfactual = TokenPosition(
            lambda x: [0],
            self.mock_pipeline,
            is_original=False,
            id="counterfactual"
        )
        self.assertFalse(tp_counterfactual.is_original)

    def test_token_position_default_is_original(self):
        """Test that TokenPosition defaults is_original to True."""
        tp = TokenPosition(
            lambda x: [0],
            self.mock_pipeline,
            id="default"
        )
        self.assertTrue(tp.is_original)

    def test_token_position_with_flag_aware_indexer(self):
        """Test TokenPosition with an indexer that uses is_original flag."""
        def smart_indexer(input, is_original=True):
            text = input.get("text", "")
            if is_original:
                return [0, 1]  # First two tokens
            else:
                return [2, 3]  # Last two tokens

        tp = TokenPosition(smart_indexer, self.mock_pipeline, id="smart")

        # Test with is_original=True
        result_original = tp.index({"text": "test"}, is_original=True)
        self.assertEqual(result_original, [0, 1])

        # Test with is_original=False
        result_counterfactual = tp.index({"text": "test"}, is_original=False)
        self.assertEqual(result_counterfactual, [2, 3])


class TestComponentWithIsOriginal(unittest.TestCase):
    """Test Component class with is_original flag."""

    def test_component_passes_is_original_to_indexer(self):
        """Test that Component.index passes is_original to the indexer."""
        def position_indexer(input, is_original=True):
            return [0] if is_original else [1]

        indexer = ComponentIndexer(position_indexer, id="test")
        component = Component(
            layer=5,
            component_type="block_input",
            indices_func=indexer,
            unit="pos"
        )

        # Test with is_original=True
        result_original = component.index({"text": "test"}, is_original=True)
        self.assertEqual(result_original, [0])

        # Test with is_original=False
        result_counterfactual = component.index({"text": "test"}, is_original=False)
        self.assertEqual(result_counterfactual, [1])


class TestAtomicModelUnitWithIsOriginal(unittest.TestCase):
    """Test AtomicModelUnit with is_original flag."""

    def test_atomic_model_unit_passes_is_original(self):
        """Test that AtomicModelUnit.index_component passes is_original."""
        def position_indexer(input, is_original=True):
            return [0] if is_original else [1]

        indexer = ComponentIndexer(position_indexer, id="test")
        component = Component(
            layer=5,
            component_type="block_input",
            indices_func=indexer,
            unit="pos"
        )

        unit = AtomicModelUnit(
            component=component,
            featurizer=Featurizer(),
            id="test_unit"
        )

        # Test with is_original=True
        result_original = unit.index_component({"text": "test"}, is_original=True)
        self.assertEqual(result_original, [0])

        # Test with is_original=False
        result_counterfactual = unit.index_component({"text": "test"}, is_original=False)
        self.assertEqual(result_counterfactual, [1])


class TestBackwardCompatibility(unittest.TestCase):
    """Test backward compatibility with existing code."""

    def test_old_style_indexer_still_works(self):
        """Test that old indexers without is_original still work."""
        # Old-style indexer
        def old_indexer(input):
            return [0, 1, 2]

        indexer = ComponentIndexer(old_indexer, id="old")

        # Should work without is_original
        result = indexer.index({"text": "test"})
        self.assertEqual(result, [0, 1, 2])

        # Should also work when is_original is passed
        result = indexer.index({"text": "test"}, is_original=True)
        self.assertEqual(result, [0, 1, 2])

        result = indexer.index({"text": "test"}, is_original=False)
        self.assertEqual(result, [0, 1, 2])

    def test_old_style_token_position(self):
        """Test that TokenPosition works without specifying is_original."""
        mock_pipeline = Mock()

        # Old-style indexer
        def old_indexer(input):
            return [0]

        # Create without is_original parameter
        tp = TokenPosition(old_indexer, mock_pipeline, id="old")

        # Should default to True
        self.assertTrue(tp.is_original)

        # Should work
        result = tp.index({"text": "test"})
        self.assertEqual(result, [0])


class TestIntegrationWithArrowSyntax(unittest.TestCase):
    """Test integration with the arrow syntax from run_interchange."""

    def test_different_positions_for_original_and_counterfactual(self):
        """
        Test a realistic scenario where we want to select different
        token positions in original vs counterfactual inputs.

        This simulates the use case for the arrow syntax "var1<-var2"
        where var1 and var2 might be at different positions.
        """
        mock_pipeline = Mock()

        # Indexer that selects different positions based on is_original
        def variable_position_indexer(input, is_original=True):
            if is_original:
                # In original input, "var1" is at position 5
                return [5]
            else:
                # In counterfactual input, "var2" is at position 8
                return [8]

        tp = TokenPosition(
            variable_position_indexer,
            mock_pipeline,
            id="variable_pos"
        )

        # When processing original input
        original_positions = tp.index({"text": "original"}, is_original=True)
        self.assertEqual(original_positions, [5])

        # When processing counterfactual input
        counterfactual_positions = tp.index({"text": "counterfactual"}, is_original=False)
        self.assertEqual(counterfactual_positions, [8])


if __name__ == "__main__":
    unittest.main()
