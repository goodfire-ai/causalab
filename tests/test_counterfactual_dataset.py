import pytest
import unittest.mock as mock

from datasets import Dataset
from causal.counterfactual_dataset import CounterfactualDataset

class TestCounterfactualDataset:
    """Tests for the CounterfactualDataset class."""
    
    def test_init_valid(self):
        """Test initialization with valid parameters."""
        # Create a simple dataset with required features
        data = {
            "input": ["input1", "input2"],
            "counterfactual_inputs": [["cf1_1", "cf1_2"], ["cf2_1"]]
        }
        dataset = Dataset.from_dict(data)
        
        # Initialize CounterfactualDataset with the dataset
        cf_dataset = CounterfactualDataset(dataset=dataset, id="test_dataset")
        
        # Check if initialization was successful
        assert cf_dataset.id == "test_dataset"
        assert len(cf_dataset) == 2
    
    def test_init_missing_features(self):
        """Test initialization with missing required features."""
        # Create a dataset missing required features
        data = {"other_feature": [1, 2, 3]}
        dataset = Dataset.from_dict(data)
        
        # Initialize should raise AssertionError
        with pytest.raises(AssertionError):
            CounterfactualDataset(dataset=dataset, id="invalid_dataset")
    def test_from_sampler(self):
        """Test generating a dataset with a simple sampling function."""
        # Define a simple counterfactual sampler
        def sampler():
            return {
                "input": "original input",
                "counterfactual_inputs": ["counterfactual 1", "counterfactual 2"]
            }
        
        # Create a dataset with required features
        data = {
            "input": ["placeholder"],
            "counterfactual_inputs": [["placeholder"]]
        }
        dataset = Dataset.from_dict(data)
        cf_dataset = CounterfactualDataset(dataset=dataset, id="generator_test")
        
        # Generate a dataset with 5 examples
        generated = cf_dataset.from_sampler(5, sampler)
        
        # Check the generated dataset
        assert len(generated) == 5
        
        # Check data directly from _data attribute
        for i in range(5):
            example = generated.dataset[i]
            assert example["input"] == "original input"
            assert len(example["counterfactual_inputs"]) == 2
        
        assert isinstance(generated, CounterfactualDataset)
    
    def test_from_sampler_with_filter(self):
        """Test generating a dataset with a filter function."""
        # Counter to generate different inputs
        counter = [0]
        
        def sampler():
            counter[0] += 1
            return {
                "input": f"input {counter[0]}",
                "counterfactual_inputs": [f"cf {counter[0]}_1", f"cf {counter[0]}_2"]
            }
        
        # Filter that only accepts even-numbered inputs
        def filter_fn(sample):
            input_num = int(sample["input"].split()[1])
            return input_num % 2 == 0
        
        # Create dataset with required features
        data = {
            "input": ["placeholder"],
            "counterfactual_inputs": [["placeholder"]]
        }
        dataset = Dataset.from_dict(data)
        cf_dataset = CounterfactualDataset(dataset=dataset, id="filter_test")
        
        # Generate dataset with filter
        generated = cf_dataset.from_sampler(3, sampler, filter=filter_fn)
        
        # Check the generated dataset
        assert len(generated) == 3
        
        # Check filtered data
        input_nums = []
        for i in range(3):
            example = generated.dataset[i]
            input_num = int(example["input"].split()[1])
            input_nums.append(input_num)
        
        assert all(num % 2 == 0 for num in input_nums)
    
    def test_display_counterfactual_data(self):
        """Test displaying examples from the dataset."""
        # Create a simple dataset
        data = {
            "input": ["input1", "input2", "input3"],
            "counterfactual_inputs": [
                ["cf1_1", "cf1_2"], 
                ["cf2_1"], 
                ["cf3_1", "cf3_2", "cf3_3"]
            ]
        }
        dataset = Dataset.from_dict(data)
        
        cf_dataset = CounterfactualDataset(dataset=dataset, id="display_test")
        
        # Capture print output for verification
        with mock.patch("builtins.print") as mock_print:
            # Display 2 examples
            result = cf_dataset.display_counterfactual_data(num_examples=2)
            
            # Verify the correct number of examples were displayed
            assert mock_print.call_count > 0
            assert len(result) == 2
            assert "input" in result[0]
            assert "counterfactual_inputs" in result[0]
            assert result[0]["input"] == "input1"
            assert result[1]["input"] == "input2"
    
    def test_display_no_verbose(self):
        """Test displaying examples without verbose output."""
        # Create a simple dataset
        data = {
            "input": ["input1", "input2"],
            "counterfactual_inputs": [["cf1_1"], ["cf2_1", "cf2_2"]]
        }
        dataset = Dataset.from_dict(data)
        
        cf_dataset = CounterfactualDataset(dataset=dataset, id="quiet_display")
        
        # With verbose=False, should not print anything
        with mock.patch("builtins.print") as mock_print:
            result = cf_dataset.display_counterfactual_data(verbose=False)
            
            # Verify no prints were made
            mock_print.assert_not_called()
            
            # But result should still contain data
            assert len(result) == 1
            assert result[0]["input"] == "input1"