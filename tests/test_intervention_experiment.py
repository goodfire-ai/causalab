"""
Test suite for intervention_experiment.py

Focuses on verifying that causal inputs, outputs without intervention,
and outputs with intervention are properly aligned and formatted.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch
from typing import Dict, List, Any

from experiments.intervention_experiment import InterventionExperiment
from neural.pipeline import Pipeline
from neural.model_units import AtomicModelUnit
from causal.causal_model import CausalModel
from causal.counterfactual_dataset import CounterfactualDataset


# Shared fixtures as standalone functions
@pytest.fixture
def mock_pipeline():
    """Create a mock pipeline for testing."""
    pipeline = Mock(spec=Pipeline)
    pipeline.model = Mock()
    pipeline.model.__class__.__name__ = "MockModel"
    pipeline.model.device = torch.device("cpu")
    pipeline.tokenizer = Mock()
    pipeline.dump = Mock(side_effect=lambda x: f"decoded_{x}")

    # Mock generate to return predictable outputs
    def generate_mock(inputs, output_scores=False):
        batch_size = len(inputs)
        sequences = torch.arange(batch_size * 10).reshape(batch_size, 10)
        result = {
            "sequences": sequences,
            "string": [f"output_{i}" for i in range(batch_size)]
        }
        if output_scores:
            # Create mock scores - 5 positions, vocab size 100
            result["scores"] = [
                torch.randn(batch_size, 100) for _ in range(5)
            ]
        return result

    pipeline.generate = Mock(side_effect=generate_mock)
    return pipeline


@pytest.fixture
def mock_causal_model():
    """Create a mock causal model."""
    model = Mock(spec=CausalModel)
    model.id = "test_task"

    # Mock label_counterfactual_data to add predictable labels
    def label_data(dataset, target_variables):
        labeled = []
        # Handle both Mock datasets and lists
        if hasattr(dataset, 'dataset'):
            examples = dataset.dataset
        elif hasattr(dataset, '__iter__'):
            examples = list(dataset)
        else:
            examples = dataset

        for i, example in enumerate(examples):
            labeled_example = dict(example)  # Use dict() instead of copy()
            labeled_example["label"] = {"string": f"label_{i}", "value": i}
            labeled.append(labeled_example)
        return labeled

    model.label_counterfactual_data = Mock(side_effect=label_data)
    return model


@pytest.fixture
def mock_model_units():
    """Create mock model units."""
    unit1 = Mock(spec=AtomicModelUnit)
    unit1.id = "unit_1"
    unit1.get_feature_indices = Mock(return_value=None)
    unit1.shape = (768,)

    unit2 = Mock(spec=AtomicModelUnit)
    unit2.id = "unit_2"
    unit2.get_feature_indices = Mock(return_value=[0, 1, 2])
    unit2.shape = (768,)

    # Structure: [[[unit1, unit2]]] - single experiment, single counterfactual group
    return [[[unit1, unit2]]]


class TestDataAlignment:
    """Test that causal inputs and outputs maintain proper alignment."""

    @pytest.fixture
    def mock_dataset(self):
        """Create a mock counterfactual dataset with known structure."""
        # Don't use spec to allow setting magic methods
        dataset = Mock()
        dataset.id = "test_dataset"

        # Create 20 examples with clear indices
        examples = []
        for i in range(20):
            examples.append({
                "input": {
                    "raw_input": f"base_input_{i}",
                    "index": i
                },
                "counterfactual_inputs": [
                    {"raw_input": f"cf_input_{i}_0", "index": i},
                    {"raw_input": f"cf_input_{i}_1", "index": i}
                ]
            })

        dataset.dataset = examples
        # Set up iteration behavior - must return fresh iterator each time
        dataset.__iter__ = Mock(side_effect=lambda: iter(examples))
        dataset.__len__ = Mock(return_value=len(examples))
        dataset.__getitem__ = Mock(side_effect=lambda idx: examples[idx])

        return dataset

    def test_alignment_with_actual_outputs(self, mock_pipeline, mock_causal_model,
                                          mock_dataset, mock_model_units):
        """Test that actual outputs align with intervention outputs and causal inputs."""

        # Create experiment
        checker = lambda x, y: x["string"] == y["string"]
        experiment = InterventionExperiment(
            pipeline=mock_pipeline,
            causal_model=mock_causal_model,
            model_units_lists=mock_model_units,
            checker=checker,
            config={"evaluation_batch_size": 5, "output_scores": True}
        )

        # Mock _run_interchange_interventions to return predictable outputs
        def mock_interventions(pipeline, counterfactual_dataset, model_units_list,
                              verbose, output_scores, batch_size):
            """Return outputs in batches matching the dataset."""
            outputs = []
            examples = list(counterfactual_dataset)

            for i in range(0, len(examples), batch_size):
                batch = examples[i:i+batch_size]
                batch_size_actual = len(batch)

                output = {
                    "sequences": torch.arange(batch_size_actual * 10).reshape(batch_size_actual, 10) + i*100,
                    "string": [f"intervention_{j}" for j in range(i, i+batch_size_actual)]
                }
                if output_scores:
                    output["scores"] = [torch.randn(batch_size_actual, 100) for _ in range(5)]
                outputs.append(output)

            return outputs

        with patch('experiments.intervention_experiment._run_interchange_interventions',
                   side_effect=mock_interventions):

            # Run experiment with actual outputs
            results = experiment.perform_interventions(
                datasets={"test": mock_dataset},
                verbose=False,
                target_variables_list=[["output"]],
                include_actual_outputs=True
            )

        # Verify structure
        assert "dataset" in results
        assert "test" in results["dataset"]
        assert "raw_outputs_no_intervention" in results["dataset"]["test"]

        # Get the model unit key (it's a string representation)
        model_unit_key = str(mock_model_units[0])
        assert model_unit_key in results["dataset"]["test"]["model_unit"]

        unit_data = results["dataset"]["test"]["model_unit"][model_unit_key]

        # Verify all three data sources exist
        assert "raw_outputs" in unit_data
        assert "causal_model_inputs" in unit_data
        actual_outputs = results["dataset"]["test"]["raw_outputs_no_intervention"]

        # Verify alignment by checking indices
        causal_inputs = unit_data["causal_model_inputs"]
        intervention_outputs = unit_data["raw_outputs"]

        # Check that we have the right number of examples
        assert len(causal_inputs) == 20  # Total dataset size

        # Count total examples in batched outputs
        total_intervention = sum(
            batch["sequences"].shape[0] for batch in intervention_outputs
        )
        total_actual = sum(
            batch["sequences"].shape[0] for batch in actual_outputs
        )

        assert total_intervention == 20
        assert total_actual == 20

        # Verify batch structure matches between actual and intervention
        assert len(actual_outputs) == len(intervention_outputs)

        for i, (actual_batch, intervention_batch) in enumerate(zip(actual_outputs, intervention_outputs)):
            # Same batch size
            actual_size = actual_batch["sequences"].shape[0]
            intervention_size = intervention_batch["sequences"].shape[0]
            assert actual_size == intervention_size, f"Batch {i} size mismatch"

            # Both should have scores if requested
            assert "scores" in actual_batch
            assert "scores" in intervention_batch
            assert len(actual_batch["scores"]) == len(intervention_batch["scores"])

        # Verify causal inputs maintain order
        for i, causal_input in enumerate(causal_inputs):
            assert causal_input["base_input"]["index"] == i
            assert causal_input["counterfactual_inputs"][0]["index"] == i

    def test_alignment_without_scores(self, mock_pipeline, mock_causal_model,
                                     mock_dataset, mock_model_units):
        """Test alignment when output_scores is False."""

        experiment = InterventionExperiment(
            pipeline=mock_pipeline,
            causal_model=mock_causal_model,
            model_units_lists=mock_model_units,
            checker=lambda x, y: True,
            config={"evaluation_batch_size": 7, "output_scores": False}
        )

        def mock_interventions(pipeline, counterfactual_dataset, model_units_list,
                              verbose, output_scores, batch_size):
            outputs = []
            examples = list(counterfactual_dataset)

            for i in range(0, len(examples), batch_size):
                batch = examples[i:i+batch_size]
                batch_size_actual = len(batch)

                output = {
                    "sequences": torch.arange(batch_size_actual * 10).reshape(batch_size_actual, 10),
                    "string": [f"intervention_{j}" for j in range(i, i+batch_size_actual)]
                }
                # No scores when output_scores=False
                outputs.append(output)

            return outputs

        with patch('experiments.intervention_experiment._run_interchange_interventions',
                   side_effect=mock_interventions):

            results = experiment.perform_interventions(
                datasets={"test_dataset": mock_dataset},
                verbose=False,
                target_variables_list=[["output"]],
                include_actual_outputs=True
            )

        # Get results
        model_unit_key = str(mock_model_units[0])
        unit_data = results["dataset"]["test_dataset"]["model_unit"][model_unit_key]
        actual_outputs = results["dataset"]["test_dataset"]["raw_outputs_no_intervention"]

        # Verify no scores in outputs
        for batch in unit_data["raw_outputs"]:
            assert "scores" not in batch or batch["scores"] is None

        for batch in actual_outputs:
            assert "scores" not in batch or batch["scores"] is None

    def test_custom_scoring_alignment(self, mock_pipeline, mock_causal_model,
                                     mock_dataset, mock_model_units):
        """Test that custom scoring functions receive properly aligned data."""

        # Track what the custom scoring function receives
        scoring_calls = []

        def custom_scorer(raw_output, label, actual_output=None):
            # The checker gets raw_output and label from causal model
            # Extract index from label if present
            if isinstance(label, dict) and "value" in label:
                index = label["value"]
            else:
                index = 0
            scoring_calls.append({
                "causal_index": index,
                "has_actual": actual_output is not None
            })
            return float(index) / 100.0

        # Create experiment with custom scoring
        experiment = InterventionExperiment(
            pipeline=mock_pipeline,
            causal_model=mock_causal_model,
            model_units_lists=mock_model_units,
            checker=custom_scorer,
            config={"evaluation_batch_size": 3, "output_scores": True}
        )

        # Prepare the attention head experiment to test custom scoring
        from experiments.LM_experiments.attention_head_experiment import PatchAttentionHeads
        from neural.LM_units import TokenPosition

        # Mock token position
        token_pos = Mock(spec=TokenPosition)
        token_pos.id = "test_position"

        with patch('experiments.intervention_experiment._run_interchange_interventions') as mock_interv:
            # Set up batched outputs
            mock_interv.return_value = [
                {
                    "sequences": torch.arange(3 * 10).reshape(3, 10),
                    "string": [f"out_{i}" for i in range(3)],
                    "scores": [torch.randn(3, 100) for _ in range(5)]
                },
                {
                    "sequences": torch.arange(3 * 10).reshape(3, 10) + 30,
                    "string": [f"out_{i}" for i in range(3, 6)],
                    "scores": [torch.randn(3, 100) for _ in range(5)]
                },
                {
                    "sequences": torch.arange(3 * 10).reshape(3, 10) + 60,
                    "string": [f"out_{i}" for i in range(6, 9)],
                    "scores": [torch.randn(3, 100) for _ in range(5)]
                },
                {
                    "sequences": torch.arange(3 * 10).reshape(3, 10) + 90,
                    "string": [f"out_{i}" for i in range(9, 12)],
                    "scores": [torch.randn(3, 100) for _ in range(5)]
                },
                {
                    "sequences": torch.arange(3 * 10).reshape(3, 10) + 120,
                    "string": [f"out_{i}" for i in range(12, 15)],
                    "scores": [torch.randn(3, 100) for _ in range(5)]
                },
                {
                    "sequences": torch.arange(3 * 10).reshape(3, 10) + 150,
                    "string": [f"out_{i}" for i in range(15, 18)],
                    "scores": [torch.randn(3, 100) for _ in range(5)]
                },
                {
                    "sequences": torch.arange(2 * 10).reshape(2, 10) + 180,
                    "string": [f"out_{i}" for i in range(18, 20)],
                    "scores": [torch.randn(2, 100) for _ in range(5)]
                }
            ]

            results = experiment.perform_interventions(
                datasets={"test_dataset": mock_dataset},
                verbose=False,
                target_variables_list=[["output"]],
                include_actual_outputs=False  # Test without actual outputs first
            )

        # Verify scoring was called with correct indices
        assert len(scoring_calls) == 20
        for i, call in enumerate(scoring_calls):
            assert call["causal_index"] == i, f"Index mismatch at position {i}"
            assert not call["has_actual"]  # No actual outputs in this test

    def test_memory_management(self, mock_pipeline, mock_causal_model,
                               mock_dataset, mock_model_units):
        """Test that tensors are properly moved to CPU."""

        experiment = InterventionExperiment(
            pipeline=mock_pipeline,
            causal_model=mock_causal_model,
            model_units_lists=mock_model_units,
            checker=lambda x, y: True,
            config={"evaluation_batch_size": 5, "output_scores": True}
        )

        # Create GPU tensors in mock outputs
        def mock_interventions_gpu(pipeline, counterfactual_dataset, model_units_list,
                                  verbose, output_scores, batch_size):
            outputs = []
            examples = list(counterfactual_dataset)

            for i in range(0, len(examples), batch_size):
                batch_size_actual = min(batch_size, len(examples) - i)

                # Create tensors (simulating GPU tensors) - use float for requires_grad
                sequences = torch.arange(batch_size_actual * 10, dtype=torch.float32).reshape(batch_size_actual, 10)
                sequences.requires_grad = True  # This will be removed by detach()

                output = {
                    "sequences": sequences,
                    "string": [f"out_{j}" for j in range(i, i+batch_size_actual)]
                }
                if output_scores:
                    scores = []
                    for _ in range(5):
                        score_tensor = torch.randn(batch_size_actual, 100)
                        score_tensor.requires_grad = True
                        scores.append(score_tensor)
                    output["scores"] = scores
                outputs.append(output)

            return outputs

        with patch('experiments.intervention_experiment._run_interchange_interventions',
                   side_effect=mock_interventions_gpu):

            results = experiment.perform_interventions(
                datasets={"test_dataset": mock_dataset},
                verbose=False,
                target_variables_list=[["output"]],
                include_actual_outputs=True
            )

        # Verify tensors are on CPU and detached
        model_unit_key = str(mock_model_units[0])
        unit_data = results["dataset"]["test_dataset"]["model_unit"][model_unit_key]

        for batch in unit_data["raw_outputs"]:
            # Check sequences
            assert not batch["sequences"].requires_grad, "Tensors should be detached"
            assert batch["sequences"].device.type == "cpu", "Tensors should be on CPU"

            # Check scores if present
            if "scores" in batch and batch["scores"]:
                for score_tensor in batch["scores"]:
                    assert not score_tensor.requires_grad, "Score tensors should be detached"
                    assert score_tensor.device.type == "cpu", "Score tensors should be on CPU"

        # Check actual outputs
        actual_outputs = results["dataset"]["test_dataset"]["raw_outputs_no_intervention"]
        for batch in actual_outputs:
            assert not batch["sequences"].requires_grad
            assert batch["sequences"].device.type == "cpu"
            if "scores" in batch and batch["scores"]:
                for score_tensor in batch["scores"]:
                    assert not score_tensor.requires_grad
                    assert score_tensor.device.type == "cpu"


class TestBatchProcessing:
    """Test various batch size scenarios."""

    @pytest.fixture
    def small_dataset(self):
        """Create a small dataset for edge case testing."""
        dataset = Mock()  # Don't use spec to allow magic methods
        dataset.id = "small_dataset"

        examples = []
        for i in range(3):  # Only 3 examples
            examples.append({
                "input": {"raw_input": f"input_{i}", "id": i},
                "counterfactual_inputs": [{"raw_input": f"cf_{i}", "id": i}]
            })

        dataset.dataset = examples
        dataset.__iter__ = Mock(side_effect=lambda: iter(examples))
        dataset.__len__ = Mock(return_value=len(examples))
        dataset.__getitem__ = Mock(side_effect=lambda idx: examples[idx])

        return dataset

    def test_batch_size_larger_than_dataset(self, mock_pipeline, mock_causal_model,
                                           small_dataset, mock_model_units):
        """Test when batch size exceeds dataset size."""

        experiment = InterventionExperiment(
            pipeline=mock_pipeline,
            causal_model=mock_causal_model,
            model_units_lists=mock_model_units,
            checker=lambda x, y: True,
            config={"evaluation_batch_size": 10, "output_scores": False}  # Batch size > dataset
        )

        with patch('experiments.intervention_experiment._run_interchange_interventions') as mock_interv:
            # Should receive all 3 examples in a single batch
            mock_interv.return_value = [{
                "sequences": torch.arange(3 * 10).reshape(3, 10),
                "string": ["out_0", "out_1", "out_2"]
            }]

            results = experiment.perform_interventions(
                datasets={"small_dataset": small_dataset},
                verbose=False,
                target_variables_list=[["output"]]
            )

        # Verify single batch was processed correctly
        model_unit_key = str(mock_model_units[0])
        unit_data = results["dataset"]["small_dataset"]["model_unit"][model_unit_key]

        assert len(unit_data["raw_outputs"]) == 1  # Single batch
        assert unit_data["raw_outputs"][0]["sequences"].shape[0] == 3

    def test_uneven_batch_sizes(self, mock_pipeline, mock_causal_model,
                               mock_model_units):
        """Test with dataset size not divisible by batch size."""

        # Create dataset with 17 examples (not divisible by common batch sizes)
        dataset = Mock()  # Don't use spec to allow magic methods
        dataset.id = "uneven_dataset"

        examples = []
        for i in range(17):
            examples.append({
                "input": {"raw_input": f"input_{i}", "id": i},
                "counterfactual_inputs": [{"raw_input": f"cf_{i}", "id": i}]
            })

        dataset.dataset = examples
        dataset.__iter__ = Mock(side_effect=lambda: iter(examples))
        dataset.__len__ = Mock(return_value=len(examples))
        dataset.__getitem__ = Mock(side_effect=lambda idx: examples[idx])

        experiment = InterventionExperiment(
            pipeline=mock_pipeline,
            causal_model=mock_causal_model,
            model_units_lists=mock_model_units,
            checker=lambda x, y: True,  # Simple checker that always passes
            config={"evaluation_batch_size": 5, "output_scores": True}
        )

        with patch('experiments.intervention_experiment._run_interchange_interventions') as mock_interv:
            # Should create 4 batches: 5, 5, 5, 2
            mock_interv.return_value = [
                {"sequences": torch.arange(5 * 10).reshape(5, 10), "string": [f"out_{i}" for i in range(5)],
                 "scores": [torch.randn(5, 100) for _ in range(3)]},
                {"sequences": torch.arange(5 * 10).reshape(5, 10), "string": [f"out_{i}" for i in range(5, 10)],
                 "scores": [torch.randn(5, 100) for _ in range(3)]},
                {"sequences": torch.arange(5 * 10).reshape(5, 10), "string": [f"out_{i}" for i in range(10, 15)],
                 "scores": [torch.randn(5, 100) for _ in range(3)]},
                {"sequences": torch.arange(2 * 10).reshape(2, 10), "string": [f"out_{i}" for i in range(15, 17)],
                 "scores": [torch.randn(2, 100) for _ in range(3)]},
            ]

            results = experiment.perform_interventions(
                datasets={"uneven_dataset": dataset},
                verbose=False,
                target_variables_list=[["output"]]
            )

        # Verify all batches processed correctly
        model_unit_key = str(mock_model_units[0])
        unit_data = results["dataset"]["uneven_dataset"]["model_unit"][model_unit_key]

        assert len(unit_data["raw_outputs"]) == 4
        assert unit_data["raw_outputs"][0]["sequences"].shape[0] == 5
        assert unit_data["raw_outputs"][1]["sequences"].shape[0] == 5
        assert unit_data["raw_outputs"][2]["sequences"].shape[0] == 5
        assert unit_data["raw_outputs"][3]["sequences"].shape[0] == 2  # Last batch is smaller

        # Verify total number of causal inputs matches
        assert len(unit_data["causal_model_inputs"]) == 17


if __name__ == "__main__":
    pytest.main([__file__, "-v"])