import pytest
from unittest import mock
import torch
import collections
import gc

from experiments.filter_experiment import FilterExperiment
from causal.counterfactual_dataset import CounterfactualDataset


# ---------------------- Fixtures ---------------------- #

@pytest.fixture
def mock_pipeline():
    """Create a mock pipeline with controlled outputs."""
    pipeline = mock.MagicMock()
    # Set up generate to return a consistent object
    pipeline.generate.return_value = "generated_output"
    # Set up dump to return predictable outputs
    pipeline.dump.side_effect = lambda x, **kwargs: [f"output_{i}" for i in range(3)]
    return pipeline


@pytest.fixture
def mock_causal_model():
    """Create a mock causal model with controlled outputs."""
    causal_model = mock.MagicMock()
    # Define behavior for run_forward - FIXED: use "raw_output" key
    causal_model.run_forward.return_value = {"raw_output": "expected_output"}
    return causal_model


@pytest.fixture
def mock_counterfactual_dataset_factory():
    """
    Create a factory for producing mock CounterfactualDataset objects.
    
    This fixture sets up the CounterfactualDataset.from_dict class method,
    which is crucial for the FilterExperiment's functionality.
    """
    # Create a class method mock for CounterfactualDataset.from_dict
    with mock.patch('causal.counterfactual_dataset.CounterfactualDataset.from_dict') as mock_from_dict:
        # Create a factory function that produces mock datasets
        def create_dataset(inputs, counterfactuals, has_causal=False):
            # Create the basic dataset dictionary
            data = {
                "input": inputs,
                "counterfactual_inputs": counterfactuals
            }
            
            # Add causal inputs if requested
            if has_causal:
                data["causal_input"] = [f"causal_{x}" for x in inputs]
                data["causal_counterfactual_inputs"] = [
                    [f"causal_{cf}" for cf in cfs] for cfs in counterfactuals
                ]
            
            # Create a mock dataset
            dataset = mock.MagicMock(spec=CounterfactualDataset)
            
            # Set up the dataset access methods
            dataset.dataset = mock.MagicMock()
            dataset.dataset.__getitem__.side_effect = lambda key: data.get(key, [])
            dataset.dataset.__len__.return_value = len(inputs)
            dataset.dataset.features = {k: None for k in data.keys()}
            dataset.__len__.return_value = len(inputs)
            
            # Configure the from_dict mock to return a similarly configured mock
            filtered_dataset = mock.MagicMock(spec=CounterfactualDataset)
            filtered_dataset.__len__.return_value = len(inputs)  # Default to same length
            mock_from_dict.return_value = filtered_dataset
            
            return dataset
        
        yield create_dataset


@pytest.fixture
def mock_checker_all_pass():
    """Create a checker that always returns True."""
    return lambda x, y: True


@pytest.fixture
def mock_checker_all_fail():
    """Create a checker that always returns False."""
    return lambda x, y: False


@pytest.fixture
def mock_checker_selective():
    """Create a checker that passes for specific inputs."""
    def checker(pred, expected):
        # Pass for input1 and its counterfactuals, fail for others
        return "1" in str(expected)
    return checker


@pytest.fixture
def patch_cuda(monkeypatch):
    """Patch torch.cuda functions to simulate GPU availability."""
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    monkeypatch.setattr(torch.cuda, 'empty_cache', lambda: None)
    return


# ---------------------- Tests ---------------------- #

class TestFilterExperiment:
    """Tests for the FilterExperiment class."""
    
    def test_init_valid_parameters(self, mock_pipeline, mock_causal_model, mock_checker_all_pass):
        """Test that the class initializes properly with required parameters."""
        experiment = FilterExperiment(mock_pipeline, mock_causal_model, mock_checker_all_pass)
        assert experiment.pipeline is mock_pipeline
        assert experiment.causal_model is mock_causal_model
        assert experiment.checker is mock_checker_all_pass
    
    def test_filter_all_examples_pass(self, mock_pipeline, mock_causal_model, 
                                     mock_checker_all_pass, mock_counterfactual_dataset_factory):
        """Test filtering when all examples should pass."""
        # Create dataset with 3 examples
        inputs = ["input1", "input2", "input3"]
        counterfactuals = [["cf1_1", "cf1_2"], ["cf2_1"], ["cf3_1", "cf3_2"]]
        dataset = mock_counterfactual_dataset_factory(inputs, counterfactuals)
        
        # Initialize experiment
        experiment = FilterExperiment(mock_pipeline, mock_causal_model, mock_checker_all_pass)
        
        # Mock the methods that will be called during filtering
        with mock.patch.object(experiment, '_validate_original_inputs', return_value=[True, True, True]) as mock_validate_orig, \
             mock.patch.object(experiment, '_validate_counterfactual_inputs', return_value=[True, True, True]) as mock_validate_cf, \
             mock.patch('causal.counterfactual_dataset.CounterfactualDataset.from_dict') as mock_from_dict:
            
            # Set up the from_dict mock to return a dataset
            filtered_dataset = mock.MagicMock(spec=CounterfactualDataset)
            filtered_dataset.__len__.return_value = 3
            mock_from_dict.return_value = filtered_dataset
            
            # Run filter
            filtered = experiment.filter({"test_dataset": dataset})
            
            # Verify from_dict was called
            mock_from_dict.assert_called_once()
            
            # Check results
            assert len(filtered) == 1
            assert "test_dataset" in filtered
    
    def test_filter_no_examples_pass(self, mock_pipeline, mock_causal_model, 
                                   mock_checker_all_fail, mock_counterfactual_dataset_factory):
        """Test filtering when no examples should pass."""
        # Create dataset with 3 examples
        inputs = ["input1", "input2", "input3"]
        counterfactuals = [["cf1_1", "cf1_2"], ["cf2_1"], ["cf3_1", "cf3_2"]]
        dataset = mock_counterfactual_dataset_factory(inputs, counterfactuals)
        
        # Initialize experiment
        experiment = FilterExperiment(mock_pipeline, mock_causal_model, mock_checker_all_fail)
        
        # Mock the validation methods to return all False
        with mock.patch.object(experiment, '_validate_original_inputs', return_value=[False, False, False]), \
             mock.patch.object(experiment, '_validate_counterfactual_inputs', return_value=[False, False, False]):
            
            # Run filter with no examples passing
            filtered = experiment.filter({"test_dataset": dataset}, verbose=True)
            
            # No datasets should be in the result
            assert len(filtered) == 0
    
    def test_filter_mixed_results(self, mock_pipeline, mock_causal_model, 
                            mock_counterfactual_dataset_factory):
        """Test with some examples passing and some failing."""
        # Create dataset with 3 examples
        inputs = ["input1", "input2", "input3"]
        counterfactuals = [["cf1_1", "cf1_2"], ["cf2_1"], ["cf3_1", "cf3_2"]]
        dataset = mock_counterfactual_dataset_factory(inputs, counterfactuals)
        
        # Create a checker that passes only for certain inputs
        checker = lambda x, y: "input1" in str(x) or "cf1" in str(x)
        
        # Create the experiment instance
        experiment = FilterExperiment(mock_pipeline, mock_causal_model, checker)
        
        # Mock validation to simulate mixed results (only first example passes)
        with mock.patch.object(experiment, '_validate_original_inputs', return_value=[True, False, False]), \
             mock.patch.object(experiment, '_validate_counterfactual_inputs', return_value=[True, False, False]), \
             mock.patch('causal.counterfactual_dataset.CounterfactualDataset.from_dict') as mock_from_dict:
            
            # Set up filtered dataset mock
            filtered_dataset = mock.MagicMock(spec=CounterfactualDataset)
            filtered_dataset.__len__.return_value = 1  # Only one example passes
            mock_from_dict.return_value = filtered_dataset
            
            # Run filter
            result = experiment.filter({"test_dataset": dataset})
            
            # Assert on our expected result
            assert len(result) == 1
            assert "test_dataset" in result
            assert len(result["test_dataset"]) == 1
    
    def test_filter_with_small_batch_size(self, mock_pipeline, mock_causal_model, 
                                        mock_checker_all_pass, mock_counterfactual_dataset_factory):
        """Test filtering with small batches."""
        # Create dataset with 3 examples
        inputs = ["input1", "input2", "input3"]
        counterfactuals = [["cf1_1", "cf1_2"], ["cf2_1"], ["cf3_1", "cf3_2"]]
        dataset = mock_counterfactual_dataset_factory(inputs, counterfactuals)
        
        # Initialize experiment
        experiment = FilterExperiment(mock_pipeline, mock_causal_model, mock_checker_all_pass)
        
        # Mock validation methods to pass all examples
        with mock.patch.object(experiment, '_validate_original_inputs', return_value=[True]) as mock_validate_orig, \
             mock.patch.object(experiment, '_validate_counterfactual_inputs', return_value=[True]) as mock_validate_cf, \
             mock.patch('causal.counterfactual_dataset.CounterfactualDataset.from_dict') as mock_from_dict:
            
            # Set up the from_dict mock
            filtered_dataset = mock.MagicMock(spec=CounterfactualDataset)
            filtered_dataset.__len__.return_value = 3
            mock_from_dict.return_value = filtered_dataset
            
            # Run filter with batch_size=1
            filtered = experiment.filter({"test_dataset": dataset}, batch_size=1)
            
            # Check that filtering worked correctly
            assert len(filtered) == 1
            assert "test_dataset" in filtered
            
            # Verify from_dict was called with appropriate data
            assert mock_from_dict.call_count == 1
            
            # With batch_size=1, validation should be called 3 times (once per batch)
            assert mock_validate_orig.call_count == 3
            assert mock_validate_cf.call_count == 3
    
    def test_filter_with_large_batch_size(self, mock_pipeline, mock_causal_model, 
                                        mock_checker_all_pass, mock_counterfactual_dataset_factory):
        """Test with batch size larger than dataset."""
        # Create dataset with 3 examples
        inputs = ["input1", "input2", "input3"]
        counterfactuals = [["cf1_1", "cf1_2"], ["cf2_1"], ["cf3_1", "cf3_2"]]
        dataset = mock_counterfactual_dataset_factory(inputs, counterfactuals)
        
        # Initialize experiment
        experiment = FilterExperiment(mock_pipeline, mock_causal_model, mock_checker_all_pass)
        
        # Mock validation methods
        with mock.patch.object(experiment, '_validate_original_inputs', return_value=[True, True, True]) as mock_validate_orig, \
             mock.patch.object(experiment, '_validate_counterfactual_inputs', return_value=[True, True, True]) as mock_validate_cf, \
             mock.patch('causal.counterfactual_dataset.CounterfactualDataset.from_dict') as mock_from_dict:
            
            # Set up the from_dict mock
            filtered_dataset = mock.MagicMock(spec=CounterfactualDataset)
            filtered_dataset.__len__.return_value = 3
            mock_from_dict.return_value = filtered_dataset
            
            # Run filter with batch_size=10 (larger than dataset)
            filtered = experiment.filter({"test_dataset": dataset}, batch_size=10)
            
            # Check that filtering worked correctly
            assert len(filtered) == 1
            assert "test_dataset" in filtered
            
            # Verify from_dict was called with appropriate data
            assert mock_from_dict.call_count == 1
            
            # With large batch size, validation should be called only once
            assert mock_validate_orig.call_count == 1
            assert mock_validate_cf.call_count == 1
    
    def test_filter_empty_dataset(self, mock_pipeline, mock_causal_model, 
                                mock_checker_all_pass, mock_counterfactual_dataset_factory):
        """Test behavior with empty dataset."""
        # Create empty dataset
        dataset = mock_counterfactual_dataset_factory([], [])
        
        # Initialize experiment
        experiment = FilterExperiment(mock_pipeline, mock_causal_model, mock_checker_all_pass)
        
        # Run filter
        filtered = experiment.filter({"empty_dataset": dataset})
        
        # Should return empty dictionary (no datasets)
        assert len(filtered) == 0
    
    def test_filter_empty_counterfactuals(self, mock_pipeline, mock_causal_model, 
                                        mock_checker_all_pass, mock_counterfactual_dataset_factory):
        """Test with empty counterfactual lists."""
        # Create dataset with empty counterfactuals
        inputs = ["input1", "input2"]
        counterfactuals = [[], []]
        dataset = mock_counterfactual_dataset_factory(inputs, counterfactuals)
        
        # Initialize experiment
        experiment = FilterExperiment(mock_pipeline, mock_causal_model, mock_checker_all_pass)
        
        # Mock validation methods
        with mock.patch.object(experiment, '_validate_original_inputs', return_value=[True, True]), \
             mock.patch.object(experiment, '_validate_counterfactual_inputs', return_value=[True, True]), \
             mock.patch('causal.counterfactual_dataset.CounterfactualDataset.from_dict') as mock_from_dict:
            
            # Create a filtered dataset 
            filtered_dataset = mock.MagicMock(spec=CounterfactualDataset)
            filtered_dataset.__len__.return_value = 2  # Both examples kept
            mock_from_dict.return_value = filtered_dataset
            
            # Run filter
            filtered = experiment.filter({"test_dataset": dataset})
            
            # Check results
            assert len(filtered) == 1
            assert "test_dataset" in filtered
            assert len(filtered["test_dataset"]) == 2
    
    def test_filter_with_causal_input_present(self, mock_pipeline, mock_causal_model, 
                                           mock_checker_all_pass, mock_counterfactual_dataset_factory):
        """Test when datasets already contain causal_input."""
        # Create dataset with causal inputs
        inputs = ["input1", "input2"]
        counterfactuals = [["cf1_1"], ["cf2_1"]]
        dataset = mock_counterfactual_dataset_factory(inputs, counterfactuals, has_causal=True)
        
        # Initialize experiment
        experiment = FilterExperiment(mock_pipeline, mock_causal_model, mock_checker_all_pass)
        
        # Mock validation methods
        with mock.patch.object(experiment, '_validate_original_inputs', return_value=[True, True]), \
             mock.patch.object(experiment, '_validate_counterfactual_inputs', return_value=[True, True]), \
             mock.patch('causal.counterfactual_dataset.CounterfactualDataset.from_dict') as mock_from_dict:
            
            # Create a filtered dataset
            filtered_dataset = mock.MagicMock(spec=CounterfactualDataset)
            filtered_dataset.__len__.return_value = 2  # Both examples kept
            mock_from_dict.return_value = filtered_dataset
            
            # Run filter
            filtered = experiment.filter({"test_dataset": dataset})
            
            # Check results
            assert len(filtered) == 1
            assert "test_dataset" in filtered
    
    def test_cleanup_memory(self, mock_pipeline, mock_causal_model, mock_checker_all_pass, patch_cuda):
        """Test memory cleanup with CUDA patch."""
        # Create test objects
        obj1 = ["test1", "test2"]
        obj2 = ["test3"]
        
        # Create experiment
        experiment = FilterExperiment(mock_pipeline, mock_causal_model, mock_checker_all_pass)
        
        # Mock gc.collect to track calls
        with mock.patch('gc.collect') as mock_gc:
            # Call private cleanup method directly
            experiment._cleanup_memory([obj1, obj2])
            
            # Check that gc.collect was called
            mock_gc.assert_called_once()
    
    def test_verbose_output(self, mock_pipeline, mock_causal_model, 
                          mock_checker_all_pass, mock_counterfactual_dataset_factory, capfd):
        """Test that verbose output works properly."""
        # Create dataset
        inputs = ["input1", "input2"]
        counterfactuals = [["cf1_1"], ["cf2_1"]]
        dataset = mock_counterfactual_dataset_factory(inputs, counterfactuals)
        
        # Initialize experiment
        experiment = FilterExperiment(mock_pipeline, mock_causal_model, mock_checker_all_pass)
        
        # Mock validation methods
        with mock.patch.object(experiment, '_validate_original_inputs', return_value=[True, True]), \
             mock.patch.object(experiment, '_validate_counterfactual_inputs', return_value=[True, True]), \
             mock.patch('causal.counterfactual_dataset.CounterfactualDataset.from_dict') as mock_from_dict:
            
            # Create a filtered dataset
            filtered_dataset = mock.MagicMock(spec=CounterfactualDataset)
            filtered_dataset.__len__.return_value = 2  # Both examples kept
            mock_from_dict.return_value = filtered_dataset
            
            # Run filter with verbose=True
            filtered = experiment.filter({"test_dataset": dataset}, verbose=True)
            
            # Capture printed output
            out, _ = capfd.readouterr()
            
            # Check for expected output content
            assert "test_dataset" in out
            assert "kept" in out.lower() or "example" in out.lower()

    def test_validate_original_inputs_method(self, mock_pipeline, mock_causal_model, mock_checker_all_pass):
        """Test the _validate_original_inputs method directly."""
        experiment = FilterExperiment(mock_pipeline, mock_causal_model, mock_checker_all_pass)
        
        # Mock the pipeline and causal model behavior
        mock_pipeline.dump.return_value = ["pred1", "pred2"]
        mock_causal_model.run_forward.side_effect = [
            {"raw_output": "expected1"},
            {"raw_output": "expected2"}
        ]
        
        # Test inputs
        inputs = ["input1", "input2"]
        
        # Call the method
        result = experiment._validate_original_inputs(None, inputs, 0, 32)
        
        # Check that it returns the expected results
        assert len(result) == 2
        assert all(isinstance(r, bool) for r in result)
        
        # Verify the pipeline and causal model were called correctly
        mock_pipeline.generate.assert_called_once_with(inputs)
        mock_pipeline.dump.assert_called_once()
        assert mock_causal_model.run_forward.call_count == 2

    def test_validate_counterfactual_inputs_method(self, mock_pipeline, mock_causal_model, mock_checker_all_pass):
        """Test the _validate_counterfactual_inputs method directly."""
        experiment = FilterExperiment(mock_pipeline, mock_causal_model, mock_checker_all_pass)
        
        # Mock the pipeline and causal model behavior
        mock_pipeline.dump.return_value = ["pred1", "pred2", "pred3"]
        mock_causal_model.run_forward.side_effect = [
            {"raw_output": "expected1"},
            {"raw_output": "expected2"},
            {"raw_output": "expected3"}
        ]
        
        # Test counterfactual inputs
        all_cf_inputs = [["cf1"], ["cf2_1", "cf2_2"]]
        
        # Call the method
        result = experiment._validate_counterfactual_inputs(None, all_cf_inputs, 0, 32)
        
        # Check that it returns the expected results
        assert len(result) == 2
        assert all(isinstance(r, bool) for r in result)
        
        # Verify the pipeline and causal model were called correctly
        mock_pipeline.generate.assert_called_once_with(["cf1", "cf2_1", "cf2_2"])
        mock_pipeline.dump.assert_called_once()
        assert mock_causal_model.run_forward.call_count == 3

    def test_validate_counterfactual_inputs_empty(self, mock_pipeline, mock_causal_model, mock_checker_all_pass):
        """Test _validate_counterfactual_inputs with empty counterfactuals."""
        experiment = FilterExperiment(mock_pipeline, mock_causal_model, mock_checker_all_pass)
        
        # Test with empty counterfactual inputs
        all_cf_inputs = [[], []]
        
        # Call the method
        result = experiment._validate_counterfactual_inputs(None, all_cf_inputs, 0, 32)
        
        # Should return True for all examples when there are no counterfactuals
        assert result == [True, True]
        
        # Pipeline should not be called since there are no counterfactuals
        mock_pipeline.generate.assert_not_called()
        mock_causal_model.run_forward.assert_not_called()