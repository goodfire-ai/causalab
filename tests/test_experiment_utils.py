"""
Test suite for experiment_utils.py

Tests the compute_custom_scores() function and other utilities.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch

from experiments.experiment_utils import compute_custom_scores


@pytest.fixture
def mock_raw_results_with_top_k():
    """Create mock raw results with top-K formatted scores."""
    return {
        "method_name": "Test",
        "model_name": "MockModel",
        "experiment_id": "test_exp",
        "dataset": {
            "test_dataset": {
                "model_unit": {
                    "unit_1": {
                        "raw_outputs": [
                            {
                                "sequences": torch.tensor([[1, 2, 3], [4, 5, 6]]),
                                "string": ["output_0", "output_1"],
                                "scores": [
                                    {
                                        "top_k_logits": torch.tensor([[0.9, 0.8, 0.7], [0.6, 0.5, 0.4]]),
                                        "top_k_indices": torch.tensor([[10, 20, 30], [15, 25, 35]]),
                                        "top_k_tokens": [["a", "b", "c"], ["d", "e", "f"]]
                                    }
                                ]
                            }
                        ],
                        "causal_model_inputs": [
                            {"base_input": {"id": 0}, "counterfactual_inputs": [{}]},
                            {"base_input": {"id": 1}, "counterfactual_inputs": [{}]}
                        ],
                        "metadata": {"layer": 5},
                        "feature_indices": None
                    }
                }
            }
        }
    }


@pytest.fixture
def mock_raw_results_with_actual_outputs():
    """Create mock raw results with actual outputs for testing use_actual_outputs."""
    return {
        "method_name": "Test",
        "model_name": "MockModel",
        "experiment_id": "test_exp",
        "dataset": {
            "test_dataset": {
                "model_unit": {
                    "unit_1": {
                        "raw_outputs": [
                            {
                                "sequences": torch.tensor([[1, 2, 3]]),
                                "string": ["interv_output"],
                                "scores": [
                                    {
                                        "top_k_logits": torch.tensor([[0.9, 0.8, 0.7]]),
                                        "top_k_indices": torch.tensor([[10, 20, 30]]),
                                        "top_k_tokens": [["a", "b", "c"]]
                                    }
                                ]
                            }
                        ],
                        "causal_model_inputs": [
                            {"base_input": {"id": 0}, "counterfactual_inputs": [{}]}
                        ],
                        "metadata": {"layer": 5},
                        "feature_indices": None
                    }
                },
                "raw_outputs_no_intervention": [
                    {
                        "sequences": torch.tensor([[7, 8, 9]]),
                        "string": ["actual_output"],
                        "scores": [
                            {
                                "top_k_logits": torch.tensor([[0.95, 0.85, 0.75]]),
                                "top_k_indices": torch.tensor([[11, 21, 31]]),
                                "top_k_tokens": [["x", "y", "z"]]
                            }
                        ]
                    }
                ]
            }
        }
    }


class TestComputeCustomScores:
    """Test compute_custom_scores() function."""

    def test_basic_custom_scoring(self, mock_raw_results_with_top_k):
        """Test basic custom scoring without actual outputs."""

        def simple_scoring_fn(causal_input, intervention_output):
            """Return the sum of sequences as score."""
            return float(intervention_output["sequences"].sum())

        results = compute_custom_scores(
            mock_raw_results_with_top_k,
            custom_scoring_fn=simple_scoring_fn,
            metric_name="sum_metric"
        )

        # Verify structure
        assert "dataset" in results
        assert "test_dataset" in results["dataset"]
        assert "model_unit" in results["dataset"]["test_dataset"]
        assert "unit_1" in results["dataset"]["test_dataset"]["model_unit"]

        # Verify custom metric was added
        unit_data = results["dataset"]["test_dataset"]["model_unit"]["unit_1"]
        assert "sum_metric" in unit_data
        assert "scores" in unit_data["sum_metric"]
        assert "average_score" in unit_data["sum_metric"]

        # Verify scores computed correctly
        scores = unit_data["sum_metric"]["scores"]
        assert len(scores) == 2  # Two examples
        assert scores[0] == float(torch.tensor([1, 2, 3]).sum())
        assert scores[1] == float(torch.tensor([4, 5, 6]).sum())

        # Verify average
        assert unit_data["sum_metric"]["average_score"] == np.mean(scores)

    def test_custom_scoring_with_actual_outputs(self, mock_raw_results_with_actual_outputs):
        """Test custom scoring with actual outputs."""

        def diff_scoring_fn(causal_input, intervention_output, actual_output=None):
            """Compute difference between intervention and actual."""
            if actual_output is None:
                raise ValueError("Need actual output")

            interv_sum = float(intervention_output["sequences"].sum())
            actual_sum = float(actual_output["sequences"].sum())
            return interv_sum - actual_sum

        results = compute_custom_scores(
            mock_raw_results_with_actual_outputs,
            custom_scoring_fn=diff_scoring_fn,
            metric_name="diff_metric",
            use_actual_outputs=True
        )

        # Verify metric computed
        unit_data = results["dataset"]["test_dataset"]["model_unit"]["unit_1"]
        assert "diff_metric" in unit_data

        # Verify score is difference
        expected_diff = float(torch.tensor([1, 2, 3]).sum() - torch.tensor([7, 8, 9]).sum())
        assert unit_data["diff_metric"]["scores"][0] == expected_diff

    def test_error_when_actual_outputs_missing(self, mock_raw_results_with_top_k):
        """Test that error is raised when use_actual_outputs=True but no actual outputs."""

        def dummy_fn(causal_input, intervention_output, actual_output=None):
            return 1.0

        with pytest.raises(ValueError, match="use_actual_outputs=True requires actual outputs"):
            compute_custom_scores(
                mock_raw_results_with_top_k,
                custom_scoring_fn=dummy_fn,
                use_actual_outputs=True
            )

    def test_top_k_scores_passed_correctly(self, mock_raw_results_with_top_k):
        """Test that top-K formatted scores are correctly passed to custom function."""

        def check_top_k_format(causal_input, intervention_output):
            """Verify intervention_output has top-K format."""
            assert "scores" in intervention_output
            assert isinstance(intervention_output["scores"], list)
            assert len(intervention_output["scores"]) > 0

            # Check first position has top-K format
            score_dict = intervention_output["scores"][0]
            assert "top_k_logits" in score_dict
            assert "top_k_indices" in score_dict
            assert "top_k_tokens" in score_dict

            # Verify it's a single example (sliced from batch)
            assert score_dict["top_k_logits"].shape[0] == 1
            assert score_dict["top_k_indices"].shape[0] == 1
            assert len(score_dict["top_k_tokens"]) == 1

            return 1.0

        results = compute_custom_scores(
            mock_raw_results_with_top_k,
            custom_scoring_fn=check_top_k_format,
            metric_name="format_check"
        )

        # If we get here, the format checks in the function passed
        assert "format_check" in results["dataset"]["test_dataset"]["model_unit"]["unit_1"]

    def test_causal_inputs_alignment(self, mock_raw_results_with_top_k):
        """Test that causal inputs are correctly aligned with outputs."""

        seen_ids = []

        def track_ids(causal_input, intervention_output):
            """Track which causal inputs we see."""
            seen_ids.append(causal_input["base_input"]["id"])
            return 1.0

        compute_custom_scores(
            mock_raw_results_with_top_k,
            custom_scoring_fn=track_ids,
            metric_name="tracker"
        )

        # Verify we saw both examples in order
        assert seen_ids == [0, 1]

    def test_custom_metric_name(self, mock_raw_results_with_top_k):
        """Test that custom metric_name is used."""

        def dummy_fn(causal_input, intervention_output):
            return 0.5

        results = compute_custom_scores(
            mock_raw_results_with_top_k,
            custom_scoring_fn=dummy_fn,
            metric_name="my_custom_name"
        )

        unit_data = results["dataset"]["test_dataset"]["model_unit"]["unit_1"]
        assert "my_custom_name" in unit_data
        assert "custom_metric" not in unit_data  # Default name not used

    def test_tensor_score_conversion(self, mock_raw_results_with_top_k):
        """Test that tensor scores are converted to float."""

        def tensor_scoring_fn(causal_input, intervention_output):
            """Return a tensor score."""
            return torch.tensor(0.75)

        results = compute_custom_scores(
            mock_raw_results_with_top_k,
            custom_scoring_fn=tensor_scoring_fn,
            metric_name="tensor_metric"
        )

        scores = results["dataset"]["test_dataset"]["model_unit"]["unit_1"]["tensor_metric"]["scores"]
        assert all(isinstance(s, float) for s in scores)
        assert scores[0] == 0.75


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
