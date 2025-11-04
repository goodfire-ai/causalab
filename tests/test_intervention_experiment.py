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
            model_units_lists=mock_model_units,
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
            model_units_lists=mock_model_units,
            config={"output_scores": False}
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
        """Test that raw outputs are properly structured for later scoring.

        Note: In the new API, scoring happens separately via compute_interchange_scores().
        This test verifies that perform_interventions() returns raw outputs in the correct format.
        """

        # Create experiment
        experiment = InterventionExperiment(
            pipeline=mock_pipeline,
            model_units_lists=mock_model_units,
            config={"evaluation_batch_size": 3, "output_scores": True}
        )

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
                include_actual_outputs=False
            )

        # Verify raw outputs are structured correctly (20 examples total)
        model_unit_key = str(mock_model_units[0])
        unit_data = results["dataset"]["test_dataset"]["model_unit"][model_unit_key]

        # Check that raw outputs exist and are in the correct batched structure
        assert "raw_outputs" in unit_data
        assert len(unit_data["raw_outputs"]) == 7  # 7 batches (6 of size 3, 1 of size 2)

        # Verify total number of outputs across all batches is 20
        total_outputs = sum(len(batch["string"]) for batch in unit_data["raw_outputs"])
        assert total_outputs == 20

    def test_memory_management(self, mock_pipeline, mock_causal_model,
                               mock_dataset, mock_model_units):
        """Test that tensors are properly moved to CPU."""

        experiment = InterventionExperiment(
            pipeline=mock_pipeline,
            model_units_lists=mock_model_units
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
                include_actual_outputs=True
            )

        # Verify tensors are on CPU and detached
        model_unit_key = str(mock_model_units[0])
        unit_data = results["dataset"]["test_dataset"]["model_unit"][model_unit_key]

        for batch in unit_data["raw_outputs"]:
            # Check sequences
            assert not batch["sequences"].requires_grad, "Tensors should be detached"
            assert batch["sequences"].device.type == "cpu", "Tensors should be on CPU"

            # Check scores if present (now in top-K format: list of dicts)
            if "scores" in batch and batch["scores"]:
                for score_dict in batch["scores"]:
                    assert isinstance(score_dict, dict), "Scores should be in top-K dict format"
                    assert not score_dict["top_k_logits"].requires_grad, "Score tensors should be detached"
                    assert score_dict["top_k_logits"].device.type == "cpu", "Score tensors should be on CPU"
                    assert not score_dict["top_k_indices"].requires_grad, "Index tensors should be detached"
                    assert score_dict["top_k_indices"].device.type == "cpu", "Index tensors should be on CPU"

        # Check actual outputs
        actual_outputs = results["dataset"]["test_dataset"]["raw_outputs_no_intervention"]
        for batch in actual_outputs:
            assert not batch["sequences"].requires_grad
            assert batch["sequences"].device.type == "cpu"
            if "scores" in batch and batch["scores"]:
                for score_dict in batch["scores"]:
                    assert not score_dict["top_k_logits"].requires_grad
                    assert score_dict["top_k_logits"].device.type == "cpu"
                    assert not score_dict["top_k_indices"].requires_grad
                    assert score_dict["top_k_indices"].device.type == "cpu"


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
            model_units_lists=mock_model_units
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
            model_units_lists=mock_model_units
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


class TestTopKConversion:
    """Test top-K logits conversion functionality."""

    def test_convert_to_top_k_basic(self, mock_pipeline, mock_model_units):
        """Test that _convert_to_top_k correctly extracts top-K values."""
        experiment = InterventionExperiment(
            pipeline=mock_pipeline,
            model_units_lists=mock_model_units,
            config={"top_k_logits": 5}
        )

        # Create mock outputs with known logits
        vocab_size = 100
        batch_size = 3
        num_positions = 2

        # Create logits where we know the top values
        mock_outputs = [{
            "sequences": torch.arange(batch_size * 10).reshape(batch_size, 10),
            "string": ["test_output"] * batch_size,
            "scores": [
                torch.randn(batch_size, vocab_size) for _ in range(num_positions)
            ]
        }]

        # Set up mock tokenizer decode
        mock_pipeline.tokenizer.decode = Mock(side_effect=lambda x: f"token_{x[0]}")

        # Convert to top-K
        converted = experiment._convert_to_top_k(mock_outputs)

        # Verify structure
        assert len(converted) == 1
        assert "scores" in converted[0]
        assert len(converted[0]["scores"]) == num_positions

        # Check each position
        for pos_idx, score_dict in enumerate(converted[0]["scores"]):
            assert "top_k_logits" in score_dict
            assert "top_k_indices" in score_dict
            assert "top_k_tokens" in score_dict

            # Verify shapes
            assert score_dict["top_k_logits"].shape == (batch_size, 5)
            assert score_dict["top_k_indices"].shape == (batch_size, 5)
            assert len(score_dict["top_k_tokens"]) == batch_size
            assert len(score_dict["top_k_tokens"][0]) == 5

            # Verify top-K are actually the largest values
            original_logits = mock_outputs[0]["scores"][pos_idx]
            for batch_idx in range(batch_size):
                top_k_values = score_dict["top_k_logits"][batch_idx]
                top_k_indices = score_dict["top_k_indices"][batch_idx]

                # Check that indices point to correct values
                for k_idx in range(5):
                    expected_value = original_logits[batch_idx, top_k_indices[k_idx]]
                    assert torch.isclose(top_k_values[k_idx], expected_value)

                # Check that these are actually the top 5
                sorted_logits, _ = torch.sort(original_logits[batch_idx], descending=True)
                assert torch.allclose(top_k_values, sorted_logits[:5])

    def test_convert_to_top_k_with_none(self, mock_pipeline, mock_model_units):
        """Test that top_k_logits=None removes scores."""
        experiment = InterventionExperiment(
            pipeline=mock_pipeline,
            model_units_lists=mock_model_units,
            config={"top_k_logits": None}
        )

        mock_outputs = [{
            "sequences": torch.arange(10).reshape(1, 10),
            "string": ["test"],
            "scores": [torch.randn(1, 100) for _ in range(3)]
        }]

        converted = experiment._convert_to_top_k(mock_outputs)

        # Scores should be absent when k is None
        assert "scores" not in converted[0] or not converted[0].get("scores")

    def test_convert_to_top_k_with_zero(self, mock_pipeline, mock_model_units):
        """Test that top_k_logits=0 removes scores."""
        experiment = InterventionExperiment(
            pipeline=mock_pipeline,
            model_units_lists=mock_model_units,
            config={"top_k_logits": 0}
        )

        mock_outputs = [{
            "sequences": torch.arange(10).reshape(1, 10),
            "string": ["test"],
            "scores": [torch.randn(1, 100) for _ in range(3)]
        }]

        converted = experiment._convert_to_top_k(mock_outputs)

        # Scores should be absent when k is 0
        assert "scores" not in converted[0] or not converted[0].get("scores")

    def test_top_k_memory_reduction(self, mock_pipeline, mock_model_units):
        """Test that top-K actually reduces memory footprint."""
        experiment = InterventionExperiment(
            pipeline=mock_pipeline,
            model_units_lists=mock_model_units,
            config={"top_k_logits": 10}
        )

        vocab_size = 50000  # Realistic vocabulary size
        batch_size = 2
        num_positions = 5

        mock_outputs = [{
            "sequences": torch.arange(batch_size * 10).reshape(batch_size, 10),
            "string": ["test"] * batch_size,
            "scores": [torch.randn(batch_size, vocab_size) for _ in range(num_positions)]
        }]

        # Calculate original size
        original_size = sum(
            score.element_size() * score.nelement()
            for score in mock_outputs[0]["scores"]
        )

        # Set up mock tokenizer
        mock_pipeline.tokenizer.decode = Mock(side_effect=lambda x: f"token_{x[0]}")

        # Convert to top-K
        converted = experiment._convert_to_top_k(mock_outputs)

        # Calculate new size (only top-K values and indices)
        new_size = 0
        for score_dict in converted[0]["scores"]:
            new_size += score_dict["top_k_logits"].element_size() * score_dict["top_k_logits"].nelement()
            new_size += score_dict["top_k_indices"].element_size() * score_dict["top_k_indices"].nelement()

        # Verify significant reduction (should be ~5000x smaller for k=10, vocab=50000)
        reduction_factor = original_size / new_size
        assert reduction_factor > 1000, f"Expected >1000x reduction, got {reduction_factor}x"

    def test_top_k_preserves_sequences_and_strings(self, mock_pipeline, mock_model_units):
        """Test that sequences and strings are unchanged by top-K."""
        experiment = InterventionExperiment(
            pipeline=mock_pipeline,
            model_units_lists=mock_model_units,
            config={"top_k_logits": 5}
        )

        original_sequences = torch.arange(20).reshape(2, 10)
        original_strings = ["output_0", "output_1"]

        mock_outputs = [{
            "sequences": original_sequences.clone(),
            "string": original_strings.copy(),
            "scores": [torch.randn(2, 100) for _ in range(3)]
        }]

        mock_pipeline.tokenizer.decode = Mock(side_effect=lambda x: f"token_{x[0]}")

        converted = experiment._convert_to_top_k(mock_outputs)

        # Verify sequences unchanged
        assert torch.equal(converted[0]["sequences"], original_sequences)

        # Verify strings unchanged
        assert converted[0]["string"] == original_strings

    def test_top_k_with_k_larger_than_vocab(self, mock_pipeline, mock_model_units):
        """Test that k larger than vocab_size doesn't cause errors."""
        experiment = InterventionExperiment(
            pipeline=mock_pipeline,
            model_units_lists=mock_model_units,
            config={"top_k_logits": 1000}  # Larger than vocab
        )

        vocab_size = 100
        mock_outputs = [{
            "sequences": torch.arange(10).reshape(1, 10),
            "string": ["test"],
            "scores": [torch.randn(1, vocab_size) for _ in range(2)]
        }]

        mock_pipeline.tokenizer.decode = Mock(side_effect=lambda x: f"token_{x[0]}")

        converted = experiment._convert_to_top_k(mock_outputs)

        # Should return min(k, vocab_size) = 100
        for score_dict in converted[0]["scores"]:
            assert score_dict["top_k_logits"].shape[1] == vocab_size
            assert score_dict["top_k_indices"].shape[1] == vocab_size


class TestSerialization:
    """Test JSON serialization of top-K outputs."""

    def test_serialize_outputs_top_k_format(self, mock_pipeline, mock_model_units):
        """Test that _serialize_outputs converts tensors to lists."""
        experiment = InterventionExperiment(
            pipeline=mock_pipeline,
            model_units_lists=mock_model_units,
            config={"top_k_logits": 5}
        )

        # Create top-K formatted outputs (as they would be after _convert_to_top_k)
        outputs = [{
            "sequences": torch.tensor([[1, 2, 3], [4, 5, 6]]),
            "string": ["test1", "test2"],
            "scores": [
                {
                    "top_k_logits": torch.tensor([[0.9, 0.8, 0.7, 0.6, 0.5],
                                                   [0.95, 0.85, 0.75, 0.65, 0.55]]),
                    "top_k_indices": torch.tensor([[10, 20, 30, 40, 50],
                                                    [11, 21, 31, 41, 51]]),
                    "top_k_tokens": [["a", "b", "c", "d", "e"],
                                     ["f", "g", "h", "i", "j"]]
                }
            ]
        }]

        serialized = experiment._serialize_outputs(outputs)

        # Verify all tensors converted to lists
        assert isinstance(serialized[0]["sequences"], list)
        assert isinstance(serialized[0]["scores"][0]["top_k_logits"], list)
        assert isinstance(serialized[0]["scores"][0]["top_k_indices"], list)

        # Verify no tensors remain
        import json
        try:
            json.dumps(serialized)  # Should succeed if fully serializable
        except TypeError:
            pytest.fail("Serialized outputs contain non-JSON-serializable objects")

    def test_serialize_outputs_preserves_structure(self, mock_pipeline, mock_model_units):
        """Test that serialization preserves top-K structure."""
        experiment = InterventionExperiment(
            pipeline=mock_pipeline,
            model_units_lists=mock_model_units
        )

        outputs = [{
            "sequences": torch.tensor([[1, 2, 3]]),
            "string": ["test"],
            "scores": [
                {
                    "top_k_logits": torch.tensor([[0.9, 0.8, 0.7]]),
                    "top_k_indices": torch.tensor([[10, 20, 30]]),
                    "top_k_tokens": [["a", "b", "c"]]
                },
                {
                    "top_k_logits": torch.tensor([[0.6, 0.5, 0.4]]),
                    "top_k_indices": torch.tensor([[40, 50, 60]]),
                    "top_k_tokens": [["d", "e", "f"]]
                }
            ]
        }]

        serialized = experiment._serialize_outputs(outputs)

        # Verify structure preserved
        assert len(serialized[0]["scores"]) == 2
        assert "top_k_logits" in serialized[0]["scores"][0]
        assert "top_k_indices" in serialized[0]["scores"][0]
        assert "top_k_tokens" in serialized[0]["scores"][0]

        # Verify values match
        assert serialized[0]["sequences"] == [[1, 2, 3]]
        # Use approximate comparison for floats
        import numpy as np
        np.testing.assert_allclose(serialized[0]["scores"][0]["top_k_logits"], [[0.9, 0.8, 0.7]], rtol=1e-5)
        assert serialized[0]["scores"][0]["top_k_indices"] == [[10, 20, 30]]
        assert serialized[0]["scores"][0]["top_k_tokens"] == [["a", "b", "c"]]

    def test_save_and_load_top_k_results(self, mock_pipeline, mock_model_units, tmp_path):
        """Test full round-trip: save to JSON and verify format."""
        # Create a simple mock dataset for this test
        mock_dataset = Mock()
        mock_dataset.id = "test"
        examples = []
        for i in range(5):
            examples.append({
                "input": {"raw_input": f"input_{i}"},
                "counterfactual_inputs": [{"raw_input": f"cf_{i}"}]
            })
        mock_dataset.dataset = examples
        mock_dataset.__iter__ = Mock(side_effect=lambda: iter(examples))
        mock_dataset.__len__ = Mock(return_value=len(examples))

        experiment = InterventionExperiment(
            pipeline=mock_pipeline,
            model_units_lists=mock_model_units,
            config={"top_k_logits": 3, "evaluation_batch_size": 5}
        )

        def mock_interventions(pipeline, counterfactual_dataset, model_units_list,
                              verbose, output_scores, batch_size):
            return [{
                "sequences": torch.tensor([[1, 2, 3], [4, 5, 6]]),
                "string": ["out_0", "out_1"],
                "scores": [torch.randn(2, 100) for _ in range(2)]
            }]

        mock_pipeline.tokenizer.decode = Mock(side_effect=lambda x: f"token_{x[0]}")

        with patch('experiments.intervention_experiment._run_interchange_interventions',
                   side_effect=mock_interventions):
            results = experiment.perform_interventions(
                datasets={"test": mock_dataset},
                verbose=False,
                save_dir=str(tmp_path)
            )

        # Verify file was created
        saved_files = list(tmp_path.glob("*.json"))
        assert len(saved_files) == 1

        # Load and verify JSON structure
        import json
        with open(saved_files[0], 'r') as f:
            loaded = json.load(f)

        # Verify top-K format in saved file
        model_unit_key = list(loaded["dataset"]["test"]["model_unit"].keys())[0]
        outputs = loaded["dataset"]["test"]["model_unit"][model_unit_key]["outputs"]

        assert len(outputs) > 0
        if "scores" in outputs[0]:
            assert isinstance(outputs[0]["scores"], list)
            assert "top_k_logits" in outputs[0]["scores"][0]
            assert "top_k_indices" in outputs[0]["scores"][0]
            assert "top_k_tokens" in outputs[0]["scores"][0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])