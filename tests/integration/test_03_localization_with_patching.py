"""
Integration tests for Notebook 03: Localization with Patching

Tests filtering datasets, running PatchResidualStream experiments,
and localizing answer and answer_position variables.
"""

import pytest
import torch
from tasks.MCQA.mcqa import MCQA_task
from experiments.filter_experiment import FilterExperiment
from experiments.LM_experiments.residual_stream_experiment import PatchResidualStream


pytestmark = [pytest.mark.slow, pytest.mark.gpu]


class TestDatasetFiltering:
    """Test filtering counterfactual datasets based on model performance."""

    def test_filter_experiment_creation(self, pipeline, causal_model, checker):
        """Test creating FilterExperiment."""
        filter_exp = FilterExperiment(pipeline, causal_model, checker)

        assert filter_exp is not None
        assert filter_exp.pipeline == pipeline
    def test_filter_datasets(
        self,
        pipeline,
        causal_model,
        checker,
        small_different_symbol_dataset,
        small_same_symbol_diff_position_dataset,
        small_random_dataset
    ):
        """Test filtering multiple counterfactual datasets."""
        datasets = {
            "different_symbol": small_different_symbol_dataset,
            "same_symbol_different_position": small_same_symbol_diff_position_dataset,
            "random_counterfactual": small_random_dataset
        }

        filter_exp = FilterExperiment(pipeline, causal_model, checker)
        filtered_datasets = filter_exp.filter(
            datasets,
            verbose=False,
            batch_size=8
        )

        # Verify that we get filtered datasets back
        assert isinstance(filtered_datasets, dict)
        assert len(filtered_datasets) == 3

        # Verify each filtered dataset is not larger than original
        for key in datasets.keys():
            assert key in filtered_datasets
            assert len(filtered_datasets[key]) <= len(datasets[key])

    def test_filtered_dataset_structure(
        self,
        pipeline,
        causal_model,
        checker,
        small_different_symbol_dataset
    ):
        """Test that filtered datasets maintain proper structure."""
        datasets = {"test": small_different_symbol_dataset}

        filter_exp = FilterExperiment(pipeline, causal_model, checker)
        filtered = filter_exp.filter(datasets, verbose=False, batch_size=8)

        # Check structure is preserved
        if len(filtered["test"]) > 0:
            example = filtered["test"][0]
            assert "input" in example
            assert "counterfactual_inputs" in example


class TestTokenPositions:
    """Test token position definitions for the MCQA task."""

    def test_create_token_positions(self, pipeline):
        """Test creating token positions for MCQA task."""
        token_positions = MCQA_task.create_token_positions(pipeline)

        assert isinstance(token_positions, dict)
        assert len(token_positions) > 0

        # Check expected positions exist
        expected_positions = [
            "symbol0",
            "symbol1",
            "correct_symbol",
            "last_token"
        ]
        for pos_name in expected_positions:
            assert pos_name in token_positions

    def test_token_position_selection(self, pipeline, causal_model):
        """Test that token positions correctly select tokens."""
        from tasks.MCQA.mcqa import sample_answerable_question

        token_positions = MCQA_task.create_token_positions(pipeline)
        example = sample_answerable_question()

        # Test that each position can highlight a token
        for pos_name, token_pos in token_positions.items():
            highlighted = token_pos.highlight_selected_token(example)
            assert isinstance(highlighted, str)
            assert "**" in highlighted  # Check that highlighting occurred


class TestPatchResidualStreamExperiment:
    """Test activation patching experiments on residual stream."""

    def test_create_patch_experiment(self, pipeline, causal_model, checker):
        """Test creating PatchResidualStream experiment."""
        token_positions = list(MCQA_task.create_token_positions(pipeline).values())
        layers = list(range(0, min(5, pipeline.get_num_layers())))

        experiment = PatchResidualStream(
            pipeline=pipeline,
            layers=layers,
            token_positions=token_positions[:2],  # Use first 2 positions for speed
            checker=checker,
            config={"batch_size": 8}
        )

        assert experiment is not None
        assert experiment.pipeline == pipeline        assert experiment.layers == layers

    def test_perform_interventions_structure(
        self,
        pipeline,
        causal_model,
        checker,
        small_different_symbol_dataset
    ):
        """Test performing interventions and verify results structure."""
        token_positions = list(MCQA_task.create_token_positions(pipeline).values())
        layers = list(range(0, min(3, pipeline.get_num_layers())))  # Use fewer layers for speed

        experiment = PatchResidualStream(
            pipeline=pipeline,
            layers=layers,
            token_positions=token_positions[:2],  # Use fewer positions for speed
            checker=checker,
            config={"batch_size": 8}
        )

        datasets = {"test_dataset": small_different_symbol_dataset}
        target_variables_list = [["answer"], ["answer_position"]]

        results = experiment.perform_interventions(
            datasets,
            verbose=False,
            target_variables_list=target_variables_list
        )

        # Verify results structure
        assert results is not None
        assert "dataset" in results
        assert "task_name" in results
        assert "method_name" in results
        assert "model_name" in results

        # Verify dataset results
        assert "test_dataset" in results["dataset"]
        dataset_results = results["dataset"]["test_dataset"]
        assert "model_unit" in dataset_results

        # Verify model unit results have expected structure
        model_units = dataset_results["model_unit"]
        assert len(model_units) > 0

        first_unit_key = next(iter(model_units.keys()))
        unit_result = model_units[first_unit_key]
        assert "metadata" in unit_result

        # Check that we have results for target variables
        for target_vars in target_variables_list:
            target_key = "-".join(target_vars)
            assert target_key in unit_result

    def test_perform_interventions_with_multiple_datasets(
        self,
        pipeline,
        causal_model,
        checker,
        small_different_symbol_dataset,
        small_random_dataset
    ):
        """Test performing interventions on multiple datasets."""
        token_positions = list(MCQA_task.create_token_positions(pipeline).values())
        layers = list(range(0, min(3, pipeline.get_num_layers())))

        experiment = PatchResidualStream(
            pipeline=pipeline,
            layers=layers,
            token_positions=token_positions[:2],
            config={"batch_size": 8}
        )

        datasets = {
            "different_symbol": small_different_symbol_dataset,
            "random": small_random_dataset
        }
        target_variables_list = [["answer"]]

        results = experiment.perform_interventions(
            datasets,
            verbose=False,
            target_variables_list=target_variables_list
        )

        # Verify both datasets have results
        assert "different_symbol" in results["dataset"]
        assert "random" in results["dataset"]

    def test_intervention_results_have_scores(
        self,
        pipeline,
        causal_model,
        checker,
        small_different_symbol_dataset
    ):
        """Test that intervention results contain accuracy scores."""
        token_positions = list(MCQA_task.create_token_positions(pipeline).values())
        layers = list(range(0, min(3, pipeline.get_num_layers())))

        experiment = PatchResidualStream(
            pipeline=pipeline,
            layers=layers,
            token_positions=token_positions[:2],
            config={"batch_size": 8}
        )

        datasets = {"test": small_different_symbol_dataset}
        target_variables_list = [["answer"]]

        results = experiment.perform_interventions(
            datasets,
            verbose=False,
            target_variables_list=target_variables_list
        )

        # Check that results have scores
        first_unit_key = next(iter(results["dataset"]["test"]["model_unit"].keys()))
        unit_result = results["dataset"]["test"]["model_unit"][first_unit_key]

        assert "answer" in unit_result
        answer_result = unit_result["answer"]
        assert "average_score" in answer_result
        assert isinstance(answer_result["average_score"], float)
        assert 0 <= answer_result["average_score"] <= 1


class TestIntegrationWorkflow:
    """Test the complete workflow from notebook 03."""

    def test_full_workflow(
        self,
        pipeline,
        causal_model,
        checker,
        small_different_symbol_dataset,
        small_same_symbol_diff_position_dataset
    ):
        """Test the complete workflow: filter -> create experiment -> run interventions."""
        # Step 1: Filter datasets
        datasets = {
            "different_symbol": small_different_symbol_dataset,
            "same_symbol_different_position": small_same_symbol_diff_position_dataset
        }

        filter_exp = FilterExperiment(pipeline, causal_model, checker)
        filtered_datasets = filter_exp.filter(datasets, verbose=False, batch_size=8)

        # Step 2: Create token positions
        token_positions = list(MCQA_task.create_token_positions(pipeline).values())

        # Step 3: Create patching experiment with minimal layers/positions for speed
        layers = list(range(0, min(2, pipeline.get_num_layers())))
        experiment = PatchResidualStream(
            pipeline=pipeline,
            layers=layers,
            token_positions=token_positions[:1],  # Just one position
            checker=checker,
            config={"batch_size": 8}
        )

        # Step 4: Run interventions
        target_variables_list = [["answer"]]
        results = experiment.perform_interventions(
            filtered_datasets,
            verbose=False,
            target_variables_list=target_variables_list
        )

        # Verify end-to-end results
        assert results is not None
        assert "dataset" in results
        assert len(results["dataset"]) > 0
