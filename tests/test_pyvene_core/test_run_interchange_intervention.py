# tests/test_pyvene_core/test_run_interchange_intervention.py

import pytest
import torch
from unittest.mock import MagicMock, patch, call, ANY  # Added ANY import

from experiments.pyvene_core import (_run_interchange_interventions, 
                                    _prepare_intervenable_model, 
                                    _batched_interchange_intervention,
                                    _delete_intervenable_model)
from neural.model_units import AtomicModelUnit
from causal.counterfactual_dataset import CounterfactualDataset


class TestRunInterchangeInterventions:
    """Tests for the _run_interchange_interventions function."""
    
    @pytest.fixture
    def mock_counterfactual_dataset(self):
        """Create a mock counterfactual dataset."""
        # Create mock dataset with required features
        mock_dataset = MagicMock()
        mock_dataset.dataset = MagicMock()
        mock_dataset.dataset.__getitem__.side_effect = lambda i: {
            "input": f"input_{i}",
            "counterfactual_inputs": [f"cf_{i}_1", f"cf_{i}_2"]
        }
        mock_dataset.dataset.__len__.return_value = 10
        return mock_dataset
    
    @pytest.fixture
    def mock_intervenable_model(self):
        """Create a mock intervenable model."""
        mock_model = MagicMock()
        mock_model.generate.return_value = [
            MagicMock(sequences=torch.tensor([[1, 2, 3], [4, 5, 6]])),
        ]
        return mock_model
    
    def test_basic_intervention_run(self, mock_tiny_lm, model_units_list, mock_counterfactual_dataset):
        """Test basic functionality for running interventions."""
        # Mock _prepare_intervenable_model
        mock_model = MagicMock()
        with patch('experiments.pyvene_core._prepare_intervenable_model', return_value=mock_model) as mock_prepare, \
             patch('experiments.pyvene_core._batched_interchange_intervention') as mock_batched, \
             patch('experiments.pyvene_core._delete_intervenable_model') as mock_delete:

            # Set up mock return values for _batched_interchange_intervention
            # Return different tensors for each batch to ensure results are properly collected
            mock_batched.side_effect = [
                torch.tensor([[1, 2, 3], [4, 5, 6]]),  # First batch
                torch.tensor([[7, 8, 9]])               # Second batch
            ]

            # Call the function
            results = _run_interchange_interventions(
                pipeline=mock_tiny_lm,
                counterfactual_dataset=mock_counterfactual_dataset,
                model_units_list=model_units_list,
                verbose=False,
                batch_size=6,
                output_scores=False
            )

            # Verify that _prepare_intervenable_model was called correctly
            mock_prepare.assert_called_once_with(
                mock_tiny_lm, model_units_list, intervention_type="interchange"
            )

            # Verify that _batched_interchange_intervention was called for each batch
            assert mock_batched.call_count == 2

            # Verify that _delete_intervenable_model was called to clean up
            mock_delete.assert_called_once_with(mock_model)
            
            # Verify results
            assert len(results) == 2  # One result per batch
            # Check the shapes (should have batch_size dimension preserved)
            assert results[0].shape == (2, 3)  # First batch: 2 examples, 3 tokens each
            assert results[1].shape == (1, 3)  # Second batch: 1 example, 3 tokens each
    
    def test_with_output_scores(self, mock_tiny_lm, model_units_list, mock_counterfactual_dataset):
        """Test when output_scores=True, verifying scores are properly returned."""
        # Mock _prepare_intervenable_model
        mock_model = MagicMock()
        with patch('experiments.pyvene_core._prepare_intervenable_model', return_value=mock_model) as mock_prepare, \
             patch('experiments.pyvene_core._batched_interchange_intervention') as mock_batched, \
             patch('experiments.pyvene_core._delete_intervenable_model') as mock_delete:
            
            # For scores, we expect a list of tensors for each batch, which are then stacked
            # Each tensor has shape (batch_size, vocab_size)
            mock_batched.side_effect = [
                # First batch: 2 tensors of shape (2, 5)
                [torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5], [0.6, 0.7, 0.8, 0.9, 1.0]]),
                 torch.tensor([[1.1, 1.2, 1.3, 1.4, 1.5], [1.6, 1.7, 1.8, 1.9, 2.0]])],
                # Second batch: 2 tensors of shape (1, 5)
                [torch.tensor([[2.1, 2.2, 2.3, 2.4, 2.5]]),
                 torch.tensor([[2.6, 2.7, 2.8, 2.9, 3.0]])]
            ]
            
            # Call the function with output_scores=True
            results = _run_interchange_interventions(
                pipeline=mock_tiny_lm,
                counterfactual_dataset=mock_counterfactual_dataset,
                model_units_list=model_units_list,
                verbose=False,
                batch_size=6,
                output_scores=True
            )
            
            # Verify that _batched_interchange_intervention was called with output_scores=True
            # Fix: Use ANY instead of mock.ANY
            mock_batched.assert_called_with(mock_tiny_lm, mock_model, ANY, model_units_list, 
                                        output_scores=True)
            
            # Verify results - should be a list of stacked tensors
            assert len(results) == 2  # One result per batch
    
    def test_with_tqdm_progress(self, mock_tiny_lm, model_units_list, mock_counterfactual_dataset):
        """Test that verbose=True enables progress tracking with tqdm."""
        # Mock _prepare_intervenable_model
        mock_model = MagicMock()
        with patch('experiments.pyvene_core._prepare_intervenable_model', return_value=mock_model), \
             patch('experiments.pyvene_core._batched_interchange_intervention', return_value=torch.tensor([[1, 2, 3]])), \
             patch('experiments.pyvene_core._delete_intervenable_model'), \
             patch('experiments.pyvene_core.tqdm') as mock_tqdm:
            
            # Call the function with verbose=True
            results = _run_interchange_interventions(
                pipeline=mock_tiny_lm,
                counterfactual_dataset=mock_counterfactual_dataset,
                model_units_list=model_units_list,
                verbose=True,
                batch_size=6,
                output_scores=False
            )
            
            # Verify that tqdm was used to wrap the dataloader
            mock_tqdm.assert_called_once()
    
    def test_with_small_batch_size(self, mock_tiny_lm, model_units_list, mock_counterfactual_dataset):
        """Test behavior with a small batch size, requiring more processing batches."""
        # Mock _prepare_intervenable_model
        mock_model = MagicMock()
        with patch('experiments.pyvene_core._prepare_intervenable_model', return_value=mock_model) as mock_prepare, \
             patch('experiments.pyvene_core._batched_interchange_intervention') as mock_batched, \
             patch('experiments.pyvene_core._delete_intervenable_model') as mock_delete:
            
            # Set up return values for each batch
            mock_batched.side_effect = [
                torch.tensor([[1, 2, 3]]),  # Batch 1
                torch.tensor([[4, 5, 6]]),  # Batch 2
                torch.tensor([[7, 8, 9]]),  # Batch 3
                torch.tensor([[10, 11, 12]]),  # Batch 4
                torch.tensor([[13, 14, 15]])   # Batch 5
            ]
            
            # Call the function with a small batch size
            results = _run_interchange_interventions(
                pipeline=mock_tiny_lm,
                counterfactual_dataset=mock_counterfactual_dataset,
                model_units_list=model_units_list,
                verbose=False,
                batch_size=2,  # Small batch size
                output_scores=False
            )
            
            # Verify that _batched_interchange_intervention was called multiple times
            assert mock_batched.call_count > 2  # More calls than with larger batch size
            
            # Verify results
            assert len(results) > 2  # More batches than previous test
    
    def test_error_handling(self, mock_tiny_lm, model_units_list, mock_counterfactual_dataset):
        """Test handling of errors during intervention."""
        # Mock _prepare_intervenable_model
        mock_model = MagicMock()
        
        # Use simple mocking approach rather than trying to override the function
        with patch('experiments.pyvene_core._prepare_intervenable_model', return_value=mock_model), \
            patch('experiments.pyvene_core._batched_interchange_intervention', 
                side_effect=RuntimeError("Test error")):
            
            # Call the function - should propagate the error
            with pytest.raises(RuntimeError) as exc_info:
                results = _run_interchange_interventions(
                    pipeline=mock_tiny_lm,
                    counterfactual_dataset=mock_counterfactual_dataset,
                    model_units_list=model_units_list,
                    verbose=False,
                    batch_size=6,
                    output_scores=False
                )
            
            # Verify that the error message is as expected
            assert "Test error" in str(exc_info.value)
    
    def test_empty_dataset(self, mock_tiny_lm, model_units_list):
        """Test behavior with an empty dataset."""
        # Create an empty dataset
        empty_mock = MagicMock()
        empty_mock.dataset = MagicMock()
        empty_mock.dataset.__len__.return_value = 0
        
        # Mock _prepare_intervenable_model
        mock_model = MagicMock()
        with patch('experiments.pyvene_core._prepare_intervenable_model', return_value=mock_model) as mock_prepare, \
             patch('experiments.pyvene_core._batched_interchange_intervention') as mock_batched, \
             patch('experiments.pyvene_core._delete_intervenable_model') as mock_delete:
            
            # Call the function with an empty dataset
            results = _run_interchange_interventions(
                pipeline=mock_tiny_lm,
                counterfactual_dataset=empty_mock,
                model_units_list=model_units_list,
                verbose=False,
                batch_size=6,
                output_scores=False
            )
            
            # Verify that _prepare_intervenable_model was still called
            mock_prepare.assert_called_once()
            
            # Verify that _batched_interchange_intervention was not called
            mock_batched.assert_not_called()
            
            # Verify that _delete_intervenable_model was called for cleanup
            mock_delete.assert_called_once_with(mock_model)
            
            # Verify results - should be an empty list
            assert len(results) == 0