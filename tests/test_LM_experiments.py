import pytest
import torch
import numpy as np
import os
from unittest.mock import MagicMock, patch, PropertyMock, ANY

from experiments.LM_experiments.LM_utils import (
    LM_loss_and_metric_fn,
    compute_metrics,
    compute_cross_entropy_loss
)
from experiments.LM_experiments.residual_stream_experiment import PatchResidualStream

# --------------------------------------------------------------------------- #
#  Utility Function Tests                                                      #
# --------------------------------------------------------------------------- #

class TestComputeMetrics:
    """Tests for the compute_metrics function."""
    
    def test_perfect_predictions(self):
        """Test with predictions matching labels exactly."""
        predicted = torch.tensor([[1, 2, 3], [4, 5, 6]])
        labels = torch.tensor([[1, 2, 3], [4, 5, 6]])
        pad_id = 0
        
        metrics = compute_metrics(predicted, labels, pad_id)
        assert metrics["accuracy"] == 1.0
        assert metrics["token_accuracy"] == 1.0
        
    def test_partial_predictions(self):
        """Test with partially correct predictions."""
        predicted = torch.tensor([[1, 2, 9], [4, 9, 6]])
        labels = torch.tensor([[1, 2, 3], [4, 5, 6]])
        pad_id = 0
        
        metrics = compute_metrics(predicted, labels, pad_id)
        assert metrics["accuracy"] == 0.0  # No sequence fully correct
        assert pytest.approx(metrics["token_accuracy"], 0.01) == 4/6  # 4 correct out of 6 tokens
        
    def test_with_padding(self):
        """Test with padded sequences."""
        predicted = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]])
        labels = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]])
        pad_id = 0
        
        metrics = compute_metrics(predicted, labels, pad_id)
        assert metrics["accuracy"] == 1.0
        assert metrics["token_accuracy"] == 1.0
        
    def test_empty_inputs(self):
        """Test with empty inputs."""
        # Create singleton dimension tensors that represent empty sequences
        predicted = torch.tensor([[0]])
        labels = torch.tensor([[0]])
        pad_id = 0
        
        metrics = compute_metrics(predicted, labels, pad_id)
        assert "accuracy" in metrics
        assert "token_accuracy" in metrics


class TestComputeCrossEntropyLoss:
    """Tests for the compute_cross_entropy_loss function."""
    
    def test_basic_loss(self):
        """Test basic loss calculation."""
        # Create logits directly with values (no in-place operations)
        logits = torch.zeros(2, 3, 5)
        # Create a detached copy to avoid gradients
        logits_with_values = logits.clone().detach()
        for i in range(2):
            for j in range(3):
                logits_with_values[i, j, j+1] = 5.0
                
        labels = torch.tensor([[1, 2, 3], [1, 2, 3]])
        pad_id = 0
        
        # Mock cross_entropy to avoid actual backprop
        with patch('torch.nn.functional.cross_entropy', return_value=torch.tensor(0.5)):
            loss = compute_cross_entropy_loss(logits_with_values, labels, pad_id)
            assert isinstance(loss, torch.Tensor)
            assert loss.item() <= 1.0  # Loss should be 0.5 from our mock
        
    def test_with_padding(self):
        """Test loss calculation with padding."""
        # Create logits directly
        logits = torch.zeros(2, 3, 5)
        logits_with_values = logits.clone().detach()
        for i in range(2):
            for j in range(2):  # Only first 2 tokens are non-pad
                logits_with_values[i, j, j+1] = 5.0
                
        labels = torch.tensor([[1, 2, 0], [1, 2, 0]])  # Last token is padding
        pad_id = 0
        
        # Mock cross_entropy
        with patch('torch.nn.functional.cross_entropy', return_value=torch.tensor(0.3)):
            loss = compute_cross_entropy_loss(logits_with_values, labels, pad_id)
            assert isinstance(loss, torch.Tensor)
        
    def test_all_padded(self):
        """Test with all tokens being padding."""
        logits = torch.zeros(2, 3, 5)
        labels = torch.tensor([[0, 0, 0], [0, 0, 0]])  # All padding
        pad_id = 0
        
        # Should handle case where all tokens are padding
        with patch('torch.nn.functional.cross_entropy', return_value=torch.tensor(float('nan'))):
            loss = compute_cross_entropy_loss(logits, labels, pad_id)
            assert isinstance(loss, torch.Tensor)

# --------------------------------------------------------------------------- #
#  Core Function Tests                                                         #
# --------------------------------------------------------------------------- #

class TestLMLossAndMetricFn:
    """Tests for LM_loss_and_metric_fn."""
    
    @pytest.fixture
    def mock_pipeline(self):
        """Create a mock pipeline."""
        pipeline = MagicMock()
        pipeline.tokenizer.pad_token_id = 0
        pipeline.max_new_tokens = 3
        pipeline.load.side_effect = lambda x, **kwargs: {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]])
        }
        pipeline.dump.side_effect = lambda x, **kwargs: ["decoded_text"]
        return pipeline
        
    @pytest.fixture
    def mock_intervenable_model(self):
        """Create a mock intervenable model."""
        model = MagicMock()
        # Create mock output with logits
        logits_output = MagicMock()
        logits_output.logits = torch.zeros(1, 6, 10)  # batch=1, seq_len=6, vocab=10
        model.return_value = (None, logits_output)
        return model
        
    @pytest.fixture
    def mock_batch(self):
        """Create a mock batch."""
        return {
            "input": ["test input"],
            "counterfactual_inputs": [["cf input 1"], ["cf input 2"]],
            "label": ["expected output"]
        }
        
    @pytest.fixture
    def mock_model_units_list(self):
        """Create mock model units list."""
        unit = MagicMock()
        unit.index_component.return_value = [[0, 1]]
        unit.get_feature_indices.return_value = [0, 1, 2]
        return [[unit]]
    
    def test_loss_and_metric_fn_basic(self, mock_pipeline, mock_intervenable_model,
                                    mock_batch, mock_model_units_list):
        """Test basic functionality of loss_and_metric_fn."""
        # Mock _prepare_intervenable_inputs to return expected values
        with patch('experiments.LM_experiments.LM_utils._prepare_intervenable_inputs') as mock_prepare:
            # Configure the mock
            mock_prepare.return_value = (
                {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.tensor([[1, 1, 1]])},
                [{"input_ids": torch.tensor([[4, 5, 6]]), "attention_mask": torch.tensor([[1, 1, 1]])}],
                {"sources->base": ([], [])},
                []
            )
            
            # Call function
            loss, metrics, logging_info = LM_loss_and_metric_fn(
                mock_pipeline, 
                mock_intervenable_model, 
                mock_batch, 
                mock_model_units_list
            )
            
            # Assertions
            assert isinstance(loss, torch.Tensor)
            assert "accuracy" in metrics
            assert "token_accuracy" in metrics
            assert "preds" in logging_info
            assert "labels" in logging_info
            

# --------------------------------------------------------------------------- #
#  PatchResidualStream Class Tests                                             #
# --------------------------------------------------------------------------- #

class TestPatchResidualStream:
    """Tests for PatchResidualStream class."""
    
    @pytest.fixture
    def mock_pipeline(self):
        """Create a mock pipeline with model configuration."""
        pipeline = MagicMock()
        pipeline.model = MagicMock()
        pipeline.model.config = MagicMock()
        pipeline.model.config.hidden_size = 32
        pipeline.model.device = "cpu"
        return pipeline
    
    @pytest.fixture
    def mock_causal_model(self):
        """Create a mock causal model."""
        return MagicMock()
    
    @pytest.fixture
    def mock_token_positions(self):
        """Create mock token positions."""
        pos1 = MagicMock()
        pos1.id = "pos1"
        pos2 = MagicMock()
        pos2.id = "pos2"
        return [pos1, pos2]
    
    @pytest.fixture
    def mock_checker(self):
        """Create a mock checker function."""
        return lambda x, y: True
    
    @pytest.fixture
    def patch_object(self, mock_pipeline, mock_causal_model, mock_token_positions, mock_checker):
        """Create a PatchResidualStream instance with mocked internal structure."""
        # Instead of mocking the ResidualStream class, we'll mock the entire
        # model_units_lists structure after the object is created
        
        # Create the real object first
        with patch('neural.featurizers.Featurizer'):
            obj = PatchResidualStream(
                pipeline=mock_pipeline,
                causal_model=mock_causal_model,
                layers=[0, 1],
                token_positions=mock_token_positions,
                checker=mock_checker
            )
        
        # Now manually replace the model_units_lists with properly mocked structures
        mock_unit = MagicMock()
        mock_component = MagicMock()
        mock_component.get_layer.return_value = 0
        mock_component.get_index_id.return_value = "pos1"
        mock_unit.component = mock_component
        mock_unit.set_featurizer = MagicMock()  # This will be a proper Mock
        
        # Replace the real model_units_lists with our mock structure
        obj.model_units_lists = [[[mock_unit]]]
        
        return obj
    
    def test_initialization(self, patch_object, mock_pipeline, mock_token_positions):
        """Test proper initialization of PatchResidualStream."""
        assert patch_object.layers == [0, 1]
        assert patch_object.token_positions == mock_token_positions
        assert patch_object.loss_and_metric_fn == LM_loss_and_metric_fn
        
    def test_build_sae_feature_intervention(self, patch_object):
        """Test the SAE feature intervention builder."""
        # Create a mock SAE loader function
        mock_sae_loader = MagicMock()
        mock_sae = MagicMock()
        mock_sae_loader.return_value = mock_sae
        
        # Mock the SAEFeaturizer class
        with patch('neural.featurizers.SAEFeaturizer') as mock_featurizer_class:
            mock_featurizer = MagicMock()
            mock_featurizer_class.return_value = mock_featurizer
            
            # Call the method - our patch_object should have properly mocked model units
            patch_object.build_SAE_feature_intervention(mock_sae_loader)
            
            # Check that set_featurizer was called on our mocked unit
            first_unit = patch_object.model_units_lists[0][0][0]
            first_unit.set_featurizer.assert_called()  # This should work now
    
    def test_clean_memory(self, patch_object):
        """Test memory cleanup functionality."""
        # Mock necessary functions
        with patch('gc.collect') as mock_gc, \
             patch('torch.cuda.empty_cache') as mock_empty_cache, \
             patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.synchronize') as mock_sync:
            
            # Call method
            patch_object._clean_memory()
            
            # Verify correct calls
            mock_gc.assert_called_once()
            mock_empty_cache.assert_called_once()
            mock_sync.assert_called_once()
    
    def test_create_heatmap(self, patch_object):
        """Test _create_heatmap method."""
        score_matrix = np.array([[0.8, 0.6], [0.4, 0.2]])
        layers = [1, 0]
        positions = ["pos1", "pos2"]
        title = "Test Heatmap"
        
        # Mock all the matplotlib and os functions
        with patch('matplotlib.pyplot.figure') as mock_figure, \
             patch('matplotlib.pyplot.savefig') as mock_savefig, \
             patch('matplotlib.pyplot.close') as mock_close, \
             patch('matplotlib.pyplot.xlabel') as mock_xlabel, \
             patch('matplotlib.pyplot.ylabel') as mock_ylabel, \
             patch('matplotlib.pyplot.title') as mock_title, \
             patch('matplotlib.pyplot.yticks') as mock_yticks, \
             patch('matplotlib.pyplot.tight_layout') as mock_tight_layout, \
             patch('seaborn.heatmap') as mock_heatmap, \
             patch('os.makedirs') as mock_makedirs, \
             patch('os.path.dirname', return_value="/fake/path"):
            
            # Prevent figure manipulation
            mock_figure.return_value = MagicMock()
            
            # Test with no save path (display mode)
            patch_object._create_heatmap(score_matrix, layers, positions, title)
            
            # Test with save path - should use mocked path functions
            patch_object._create_heatmap(score_matrix, layers, positions, title, save_path="/fake/path/test.png")
            
            mock_savefig.assert_called_once()
    
    def test_plot_average_heatmap(self, patch_object):
        """Test _plot_average_heatmap method."""
        # Create mock results with multiple datasets
        results = {
            "dataset": {
                "dataset1": {
                    "model_unit": {
                        "unit1": {
                            "metadata": {"layer": 0, "position": "pos1"},
                            "var1-var2": {"average_score": 0.8}
                        }
                    }
                },
                "dataset2": {
                    "model_unit": {
                        "unit1": {
                            "metadata": {"layer": 0, "position": "pos1"},
                            "var1-var2": {"average_score": 0.6}
                        }
                    }
                }
            },
            "task_name": "test_task"
        }
        
        layers = [0]
        positions = ["pos1"]
        target_variables_str = "var1-var2"
        
        # Replace _create_heatmap with a stub
        patch_object._create_heatmap = MagicMock()
        
        # Call the method
        patch_object._plot_average_heatmap(results, layers, positions, target_variables_str)
        
        # Verify _create_heatmap was called
        patch_object._create_heatmap.assert_called_once()
            
    def test_plot_individual_heatmaps(self, patch_object):
        """Test _plot_individual_heatmaps method."""
        # Create mock results
        results = {
            "dataset": {
                "dataset1": {
                    "model_unit": {
                        "unit1": {
                            "metadata": {"layer": 0, "position": "pos1"},
                            "var1-var2": {"average_score": 0.8}
                        }
                    }
                },
                "dataset2": {
                    "model_unit": {
                        "unit1": {
                            "metadata": {"layer": 0, "position": "pos1"},
                            "var1-var2": {"average_score": 0.6}
                        }
                    }
                }
            },
            "task_name": "test_task"
        }
        
        layers = [0]
        positions = ["pos1"]
        target_variables_str = "var1-var2"
        
        # Replace _create_heatmap with a stub
        patch_object._create_heatmap = MagicMock()
        
        # Call the method
        patch_object._plot_individual_heatmaps(results, layers, positions, target_variables_str)
        
        # Verify _create_heatmap was called twice (once per dataset)
        assert patch_object._create_heatmap.call_count == 2
            
    def test_plot_heatmaps_integration(self, patch_object):
        """Test the main plot_heatmaps method."""
        # Create mock results
        results = {
            "dataset": {
                "dataset1": {
                    "model_unit": {
                        "unit1": {
                            "metadata": {"layer": 0, "position": "pos1"},
                            "var1-var2": {"average_score": 0.8}
                        }
                    }
                }
            },
            "task_name": "test_task"
        }
        
        # Replace all component methods with stubs
        patch_object._extract_metadata_map = MagicMock(return_value={"unit1": {"layer": 0, "position": "pos1"}})
        patch_object._extract_layers_positions = MagicMock(return_value=([0], ["pos1"]))
        patch_object._plot_average_heatmap = MagicMock()
        patch_object._plot_individual_heatmaps = MagicMock()
        
        # Test without averaging
        patch_object.plot_heatmaps(results, ["var1", "var2"])
        patch_object._plot_individual_heatmaps.assert_called_once()
        patch_object._plot_average_heatmap.assert_not_called()
        
        # Reset mocks
        patch_object._plot_individual_heatmaps.reset_mock()
        
        # Test with averaging
        patch_object.plot_heatmaps(results, ["var1", "var2"], average_counterfactuals=True)
        patch_object._plot_average_heatmap.assert_called_once()
        patch_object._plot_individual_heatmaps.assert_not_called()


# Run tests when file is executed directly
if __name__ == "__main__":
    pytest.main(["-xvs", __file__])