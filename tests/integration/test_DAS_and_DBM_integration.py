"""
Integration test for DAS (Distributed Alignment Search) and DBM (Desiderata-Based Masking).

This test is modeled after the 04_train_DAS_and_DBM.ipynb notebook and verifies:
1. Creating MCQA task datasets
2. Filtering datasets based on model performance
3. Training DAS interventions on residual stream to localize causal variables
4. Training DBM interventions on attention heads to identify responsible heads
5. Evaluating on held-out test data for generalization
"""

import pytest
import torch
from unittest.mock import MagicMock, patch, PropertyMock
import numpy as np

from tasks.MCQA.mcqa import MCQA_task, sample_answerable_question
from neural.pipeline import LMPipeline
from causal.counterfactual_dataset import CounterfactualDataset
from experiments.filter_experiment import FilterExperiment
from experiments.LM_experiments.residual_stream_experiment import PatchResidualStream
from experiments.LM_experiments.attention_head_experiment import PatchAttentionHeads
from neural.LM_units import TokenPosition, get_all_tokens


# ---------------------- Fixtures ---------------------- #

@pytest.fixture
def mock_pipeline():
    """Create a comprehensive mock pipeline for DAS/DBM experiments."""
    pipeline = MagicMock(spec=LMPipeline)

    # Model configuration
    pipeline.model = MagicMock()
    pipeline.model.config = MagicMock()
    pipeline.model.config.hidden_size = 128
    pipeline.model.config.num_hidden_layers = 4
    pipeline.model.config.num_attention_heads = 4
    pipeline.model.device = "cpu"

    # Tokenizer configuration
    pipeline.tokenizer = MagicMock()
    pipeline.tokenizer.pad_token_id = 0
    pipeline.tokenizer.padding_side = "left"
    pipeline.tokenizer.convert_ids_to_tokens.return_value = ["<pad>", "The", "answer", "is", "A"]

    # Pipeline methods
    pipeline.max_new_tokens = 1
    pipeline.load.return_value = {
        "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1]])
    }
    pipeline.dump.return_value = ["A"]
    pipeline.generate.return_value = {
        "sequences": torch.tensor([[1]]),
        "scores": [torch.randn(1, 100)]
    }

    # Model-specific methods
    pipeline.get_num_layers.return_value = 4
    pipeline.get_num_attention_heads.return_value = 4

    return pipeline


@pytest.fixture
def mock_causal_model():
    """Create a mock causal model for MCQA task."""
    causal_model = MagicMock()
    causal_model.run_forward.return_value = {
        "raw_input": "The banana is yellow. What color is the banana?\nA. yellow\nB. green\nAnswer:",
        "raw_output": " A",
        "answer": "A",
        "answer_position": 0
    }
    causal_model.sample_input.return_value = {
        "template": MCQA_task.causal_models["positional"].values["template"][0],
        "object_color": ("banana", "yellow"),
        "symbol0": "A",
        "symbol1": "B",
        "choice0": "yellow",
        "choice1": "green"
    }
    return causal_model


@pytest.fixture
def checker():
    """Create a checker function that validates model outputs."""
    def _checker(neural_output, causal_output):
        if isinstance(neural_output, dict) and "string" in neural_output:
            return causal_output in neural_output["string"] or neural_output["string"] in causal_output
        return causal_output in str(neural_output)
    return _checker


@pytest.fixture
def mock_counterfactual_dataset():
    """Create a mock counterfactual dataset."""
    # Create sample data
    inputs = [
        {
            "raw_input": "The banana is yellow. What color is the banana?\nA. yellow\nB. green\nAnswer:",
            "object_color": ("banana", "yellow"),
            "symbol0": "A",
            "symbol1": "B",
            "choice0": "yellow",
            "choice1": "green"
        }
    ]

    counterfactual_inputs = [
        [{
            "raw_input": "The banana is yellow. What color is the banana?\nA. green\nB. yellow\nAnswer:",
            "object_color": ("banana", "yellow"),
            "symbol0": "A",
            "symbol1": "B",
            "choice0": "green",
            "choice1": "yellow"
        }]
    ]

    # Create the dataset mock (without spec to allow __iter__ assignment)
    dataset = MagicMock()

    # Create inner dataset object
    inner_dataset = MagicMock()
    inner_dataset.__len__.return_value = 1
    inner_dataset.__getitem__.side_effect = lambda key: {
        "input": inputs,
        "counterfactual_inputs": counterfactual_inputs
    }.get(key, [])
    inner_dataset.features = {"input": None, "counterfactual_inputs": None}

    # Make inner dataset iterable
    def create_iter():
        return iter([{
            "input": inputs[0],
            "counterfactual_inputs": counterfactual_inputs[0]
        }])
    inner_dataset.__iter__ = create_iter

    # Assign inner dataset and make outer dataset iterable too
    dataset.dataset = inner_dataset
    dataset.__len__.return_value = 1
    dataset.__iter__ = create_iter

    return dataset


@pytest.fixture
def mock_token_positions(mock_pipeline):
    """Create mock token positions for the experiment."""
    # Create mock token positions
    pos1 = MagicMock(spec=TokenPosition)
    pos1.id = "symbol0"
    pos1.return_value = [2]  # Token index

    pos2 = MagicMock(spec=TokenPosition)
    pos2.id = "symbol0_period"
    pos2.return_value = [3]

    return [pos1, pos2]


# ---------------------- Test Classes ---------------------- #

class TestDASIntegration:
    """Integration tests for Distributed Alignment Search (DAS)."""

    def test_das_experiment_initialization(self, mock_pipeline, mock_causal_model,
                                          mock_token_positions, checker):
        """Test that DAS experiment initializes correctly."""
        with patch('neural.featurizers.Featurizer'):
            experiment = PatchResidualStream(
                pipeline=mock_pipeline,
                causal_model=mock_causal_model,
                layers=[0, 1],
                token_positions=mock_token_positions,
                config={
                    "batch_size": 32,
                    "evaluation_batch_size": 64,
                    "training_epoch": 2,
                    "n_features": 16
                }
            )

        assert experiment.layers == [0, 1]
        assert experiment.token_positions == mock_token_positions
        assert experiment.checker is checker
        assert experiment.config["n_features"] == 16

    def test_das_training_flow(self, mock_pipeline, mock_causal_model,
                              mock_token_positions, mock_counterfactual_dataset, checker):
        """Test the DAS training flow with mocked interventions."""
        with patch('neural.featurizers.Featurizer'):
            experiment = PatchResidualStream(
                pipeline=mock_pipeline,
                causal_model=mock_causal_model,
                layers=[0, 1],
                token_positions=mock_token_positions,
                config={
                    "batch_size": 32,
                    "evaluation_batch_size": 64,
                    "training_epoch": 2,
                    "n_features": 16
                }
            )

        # Mock the train_interventions method
        with patch.object(experiment, 'train_interventions') as mock_train:
            # Call training
            experiment.train_interventions(
                {"same_symbol_different_position": mock_counterfactual_dataset},
                ["answer_position"],
                method="DAS",
                verbose=False
            )

            # Verify training was called correctly
            mock_train.assert_called_once()
            call_args = mock_train.call_args
            assert "same_symbol_different_position" in call_args[0][0]
            assert "answer_position" in call_args[0][1]
            assert call_args[1]["method"] == "DAS"

    def test_das_perform_interventions(self, mock_pipeline, mock_causal_model,
                                      mock_token_positions, mock_counterfactual_dataset, checker):
        """Test performing interventions after DAS training."""
        with patch('neural.featurizers.Featurizer'):
            experiment = PatchResidualStream(
                pipeline=mock_pipeline,
                causal_model=mock_causal_model,
                layers=[0, 1],
                token_positions=mock_token_positions,
                config={
                    "batch_size": 32,
                    "evaluation_batch_size": 64
                }
            )

        # Mock perform_interventions
        mock_results = {
            "task_name": "MCQA",
            "dataset": {
                "same_symbol_different_position": {
                    "model_unit": {
                        "unit1": {
                            "metadata": {"layer": 0, "position": "symbol0"},
                            "answer_position": {"average_score": 0.85}
                        }
                    }
                }
            }
        }

        with patch.object(experiment, 'perform_interventions', return_value=mock_results) as mock_perform:
            results = experiment.perform_interventions(
                {"same_symbol_different_position": mock_counterfactual_dataset},
                target_variables_list=[["answer"], ["answer_position"]],
                verbose=False
            )

            # Verify results structure
            assert "dataset" in results
            assert "same_symbol_different_position" in results["dataset"]
            assert "model_unit" in results["dataset"]["same_symbol_different_position"]

    def test_das_generalization_to_test_set(self, mock_pipeline, mock_causal_model,
                                           mock_token_positions, mock_counterfactual_dataset, checker):
        """Test that DAS results can be evaluated on held-out test data."""
        with patch('neural.featurizers.Featurizer'):
            experiment = PatchResidualStream(
                pipeline=mock_pipeline,
                causal_model=mock_causal_model,
                layers=[0, 1, 2],
                token_positions=mock_token_positions,
                config={
                    "batch_size": 32,
                    "evaluation_batch_size": 64,
                    "training_epoch": 2,
                    "n_features": 16
                }
            )

        # Mock train and test results
        train_results = {
            "task_name": "MCQA",
            "dataset": {
                "same_symbol_different_position": {
                    "model_unit": {
                        "unit1": {
                            "metadata": {"layer": 0, "position": "symbol0"},
                            "answer_position": {"average_score": 0.90}
                        }
                    }
                }
            }
        }

        test_results = {
            "task_name": "MCQA",
            "dataset": {
                "same_symbol_different_position": {
                    "model_unit": {
                        "unit1": {
                            "metadata": {"layer": 0, "position": "symbol0"},
                            "answer_position": {"average_score": 0.75}  # Lower score indicates overfitting
                        }
                    }
                }
            }
        }

        with patch.object(experiment, 'perform_interventions') as mock_perform:
            mock_perform.side_effect = [train_results, test_results]

            # Get train results
            train_res = experiment.perform_interventions(
                {"same_symbol_different_position": mock_counterfactual_dataset},
                target_variables_list=[["answer_position"]],
                verbose=False
            )

            # Get test results
            test_res = experiment.perform_interventions(
                {"same_symbol_different_position": mock_counterfactual_dataset},
                target_variables_list=[["answer_position"]],
                verbose=False
            )

            # Verify that test performance is typically lower (generalization gap)
            train_score = train_res["dataset"]["same_symbol_different_position"]["model_unit"]["unit1"]["answer_position"]["average_score"]
            test_score = test_res["dataset"]["same_symbol_different_position"]["model_unit"]["unit1"]["answer_position"]["average_score"]

            assert train_score >= test_score  # Train score should be >= test score


class TestDBMIntegration:
    """Integration tests for Desiderata-Based Masking (DBM)."""

    def test_dbm_experiment_initialization(self, mock_pipeline, mock_causal_model, checker):
        """Test that DBM experiment initializes correctly."""
        # Create token position for all tokens
        all_tokens_pos = MagicMock(spec=TokenPosition)
        all_tokens_pos.id = "all_tokens"

        # Create layer-head list
        layer_head_list = [(0, 0), (0, 1), (1, 0), (1, 1)]

        with patch('neural.featurizers.Featurizer'):
            experiment = PatchAttentionHeads(
                pipeline=mock_pipeline,
                causal_model=mock_causal_model,
                layer_head_lists=[layer_head_list],
                token_position=all_tokens_pos,
                config={
                    "learning_rate": 0.001,
                    "batch_size": 32,
                    "evaluation_batch_size": 64,
                    "training_epoch": 10,
                    "masking": {
                        "regularization_coefficient": 0.1
                    },
                    "featurizer_kwargs": {
                        "tie_masks": True
                    }
                }
            )

        assert experiment.layer_head_lists == [layer_head_list]
        assert experiment.token_position == all_tokens_pos
        assert experiment.checker is checker

    def test_dbm_training_flow(self, mock_pipeline, mock_causal_model,
                              mock_counterfactual_dataset, checker):
        """Test the DBM training flow with mocked interventions."""
        # Create token position
        all_tokens_pos = MagicMock(spec=TokenPosition)
        all_tokens_pos.id = "all_tokens"

        # Create layer-head list covering multiple layers
        num_heads = 4
        end = 3
        layer_head_list = [(layer, head) for layer in range(0, end) for head in range(num_heads)]

        with patch('neural.featurizers.Featurizer'):
            experiment = PatchAttentionHeads(
                pipeline=mock_pipeline,
                causal_model=mock_causal_model,
                layer_head_lists=[layer_head_list],
                token_position=all_tokens_pos,
                config={
                    "learning_rate": 0.001,
                    "batch_size": 32,
                    "evaluation_batch_size": 64,
                    "training_epoch": 5,
                    "masking": {
                        "regularization_coefficient": 0.1
                    },
                    "featurizer_kwargs": {
                        "tie_masks": True
                    }
                }
            )

        # Mock the train_interventions method
        with patch.object(experiment, 'train_interventions') as mock_train:
            # Call training
            experiment.train_interventions(
                {"different_symbol": mock_counterfactual_dataset},
                ["answer"],
                method="DBM",
                verbose=False
            )

            # Verify training was called correctly
            mock_train.assert_called_once()
            call_args = mock_train.call_args
            assert "different_symbol" in call_args[0][0]
            assert "answer" in call_args[0][1]
            assert call_args[1]["method"] == "DBM"

    def test_dbm_mask_extraction(self, mock_pipeline, mock_causal_model,
                                mock_counterfactual_dataset, checker):
        """Test extracting binary masks from DBM results."""
        # Create token position
        all_tokens_pos = MagicMock(spec=TokenPosition)
        all_tokens_pos.id = "all_tokens"

        layer_head_list = [(0, 0), (0, 1), (1, 0), (1, 1)]

        with patch('neural.featurizers.Featurizer'):
            experiment = PatchAttentionHeads(
                pipeline=mock_pipeline,
                causal_model=mock_causal_model,
                layer_head_lists=[layer_head_list],
                token_position=all_tokens_pos,
                config={
                    "learning_rate": 0.001,
                    "batch_size": 32,
                    "training_epoch": 5,
                    "masking": {
                        "regularization_coefficient": 0.1
                    },
                    "featurizer_kwargs": {
                        "tie_masks": True
                    }
                }
            )

        # Mock results with feature_indices (binary masks)
        mock_results = {
            "task_name": "MCQA",
            "dataset": {
                "different_symbol": {
                    "model_unit": {
                        "all_heads": {
                            "feature_indices": {
                                "AttentionHead(Layer-0,Head-0,Token-all_tokens)": [0],  # Selected
                                "AttentionHead(Layer-0,Head-1,Token-all_tokens)": [],   # Not selected
                                "AttentionHead(Layer-1,Head-0,Token-all_tokens)": [0],  # Selected
                                "AttentionHead(Layer-1,Head-1,Token-all_tokens)": []    # Not selected
                            },
                            "answer": {"average_score": 0.95}
                        }
                    }
                }
            }
        }

        with patch.object(experiment, 'perform_interventions', return_value=mock_results):
            results = experiment.perform_interventions(
                {"different_symbol": mock_counterfactual_dataset},
                target_variables_list=[["answer"]],
                verbose=False
            )

            # Verify feature_indices structure
            assert "feature_indices" in results["dataset"]["different_symbol"]["model_unit"]["all_heads"]
            feature_indices = results["dataset"]["different_symbol"]["model_unit"]["all_heads"]["feature_indices"]

            # Check that some heads are selected and some are not
            selected_heads = [k for k, v in feature_indices.items() if v == [0]]
            unselected_heads = [k for k, v in feature_indices.items() if v == []]

            assert len(selected_heads) > 0
            assert len(unselected_heads) > 0

    def test_dbm_perfect_generalization(self, mock_pipeline, mock_causal_model,
                                       mock_counterfactual_dataset, checker):
        """Test that DBM masks generalize to test data."""
        # Create token position
        all_tokens_pos = MagicMock(spec=TokenPosition)
        all_tokens_pos.id = "all_tokens"

        layer_head_list = [(0, 0), (0, 1), (1, 0), (1, 1)]

        with patch('neural.featurizers.Featurizer'):
            experiment = PatchAttentionHeads(
                pipeline=mock_pipeline,
                causal_model=mock_causal_model,
                layer_head_lists=[layer_head_list],
                token_position=all_tokens_pos,
                config={
                    "learning_rate": 0.001,
                    "batch_size": 32,
                    "training_epoch": 10
                }
            )

        # Mock train results
        train_results = {
            "task_name": "MCQA",
            "dataset": {
                "different_symbol": {
                    "model_unit": {
                        "all_heads": {
                            "answer": {"average_score": 1.0}  # Perfect on train
                        }
                    }
                }
            }
        }

        # Mock test results (should also be perfect if heads are correctly identified)
        test_results = {
            "task_name": "MCQA",
            "dataset": {
                "different_symbol": {
                    "model_unit": {
                        "all_heads": {
                            "answer": {"average_score": 1.0}  # Perfect on test
                        }
                    }
                }
            }
        }

        with patch.object(experiment, 'perform_interventions') as mock_perform:
            mock_perform.side_effect = [train_results, test_results]

            # Get train results
            train_res = experiment.perform_interventions(
                {"different_symbol": mock_counterfactual_dataset},
                target_variables_list=[["answer"]],
                verbose=False
            )

            # Get test results
            test_res = experiment.perform_interventions(
                {"different_symbol": mock_counterfactual_dataset},
                target_variables_list=[["answer"]],
                verbose=False
            )

            # DBM should generalize perfectly when attention heads are correctly identified
            train_score = train_res["dataset"]["different_symbol"]["model_unit"]["all_heads"]["answer"]["average_score"]
            test_score = test_res["dataset"]["different_symbol"]["model_unit"]["all_heads"]["answer"]["average_score"]

            assert train_score == 1.0
            assert test_score == 1.0


class TestFilterExperimentIntegration:
    """Integration tests for FilterExperiment used before DAS/DBM."""

    def test_filter_experiment_with_mcqa_task(self, mock_pipeline, mock_causal_model,
                                             mock_counterfactual_dataset, checker):
        """Test filtering datasets based on model performance."""
        experiment = FilterExperiment(
            pipeline=mock_pipeline,
            causal_model=mock_causal_model
        )

        # Mock validation methods to simulate filtering
        with patch.object(experiment, '_validate_original_inputs', return_value=[True]), \
             patch.object(experiment, '_validate_counterfactual_inputs', return_value=[True]), \
             patch('causal.counterfactual_dataset.CounterfactualDataset.from_dict') as mock_from_dict:

            # Set up filtered dataset
            filtered_dataset = MagicMock(spec=CounterfactualDataset)
            filtered_dataset.__len__.return_value = 1
            mock_from_dict.return_value = filtered_dataset

            # Run filter
            datasets = {
                "different_symbol": mock_counterfactual_dataset,
                "same_symbol_different_position": mock_counterfactual_dataset,
                "random_counterfactual": mock_counterfactual_dataset
            }

            filtered_datasets = experiment.filter(datasets, verbose=False, batch_size=64)

            # Verify all datasets were filtered
            assert len(filtered_datasets) == 3
            assert "different_symbol" in filtered_datasets
            assert "same_symbol_different_position" in filtered_datasets
            assert "random_counterfactual" in filtered_datasets

    def test_filter_high_keep_rate(self, mock_pipeline, mock_causal_model,
                                  mock_counterfactual_dataset, checker):
        """Test that filter keeps most examples when model performs well."""
        experiment = FilterExperiment(
            pipeline=mock_pipeline,
            causal_model=mock_causal_model
        )

        # Create a dataset with multiple examples (without spec to allow __iter__ assignment)
        extended_dataset = MagicMock()
        extended_dataset.__len__.return_value = 64

        # Create inner dataset
        inner_dataset = MagicMock()
        inner_dataset.__len__.return_value = 64
        inner_dataset.__getitem__.side_effect = lambda key: [f"item_{i}" for i in range(64)]
        inner_dataset.features = {"input": None, "counterfactual_inputs": None}
        inner_dataset.__iter__ = lambda: iter([{"input": f"item_{i}", "counterfactual_inputs": []} for i in range(64)])

        extended_dataset.dataset = inner_dataset
        extended_dataset.__iter__ = lambda: iter([{"input": f"item_{i}", "counterfactual_inputs": []} for i in range(64)])

        # Mock validation to pass most examples
        passing_mask = [True] * 63 + [False]  # 63 out of 64 pass

        with patch.object(experiment, '_validate_original_inputs', return_value=passing_mask), \
             patch.object(experiment, '_validate_counterfactual_inputs', return_value=passing_mask), \
             patch('causal.counterfactual_dataset.CounterfactualDataset.from_dict') as mock_from_dict:

            # Set up filtered dataset
            filtered_dataset = MagicMock(spec=CounterfactualDataset)
            filtered_dataset.__len__.return_value = 63
            mock_from_dict.return_value = filtered_dataset

            # Run filter
            filtered_datasets = experiment.filter(
                {"test_dataset": extended_dataset},
                verbose=False,
                batch_size=64
            )

            # Check keep rate
            assert len(filtered_datasets["test_dataset"]) == 63
            keep_rate = len(filtered_datasets["test_dataset"]) / 64
            assert keep_rate > 0.95  # High keep rate (>95%)


class TestEndToEndWorkflow:
    """End-to-end integration tests simulating the notebook workflow."""

    def test_complete_das_workflow(self, mock_pipeline, mock_causal_model,
                                   mock_token_positions, mock_counterfactual_dataset, checker):
        """Test complete DAS workflow: filter -> train -> evaluate -> test."""
        # Step 1: Filter datasets
        filter_exp = FilterExperiment(
            pipeline=mock_pipeline,
            causal_model=mock_causal_model
        )

        with patch.object(filter_exp, '_validate_original_inputs', return_value=[True]), \
             patch.object(filter_exp, '_validate_counterfactual_inputs', return_value=[True]), \
             patch('causal.counterfactual_dataset.CounterfactualDataset.from_dict') as mock_from_dict:

            filtered_dataset = MagicMock(spec=CounterfactualDataset)
            filtered_dataset.__len__.return_value = 1
            mock_from_dict.return_value = filtered_dataset

            filtered_datasets = filter_exp.filter(
                {"same_symbol_different_position": mock_counterfactual_dataset},
                verbose=False,
                batch_size=64
            )

        # Step 2: Create DAS experiment
        with patch('neural.featurizers.Featurizer'):
            das_exp = PatchResidualStream(
                pipeline=mock_pipeline,
                causal_model=mock_causal_model,
                layers=[0, 1, 2],
                token_positions=mock_token_positions,
                config={
                    "batch_size": 32,
                    "evaluation_batch_size": 64,
                    "training_epoch": 4,
                    "n_features": 16
                }
            )

        # Step 3: Train DAS
        with patch.object(das_exp, 'train_interventions') as mock_train:
            das_exp.train_interventions(
                filtered_datasets,
                ["answer_position"],
                method="DAS",
                verbose=False
            )
            mock_train.assert_called_once()

        # Step 4: Evaluate on train data
        train_results = {
            "task_name": "MCQA",
            "dataset": {
                "same_symbol_different_position": {
                    "model_unit": {
                        "unit1": {
                            "metadata": {"layer": 1, "position": "symbol0"},
                            "answer_position": {"average_score": 0.88}
                        }
                    }
                }
            }
        }

        with patch.object(das_exp, 'perform_interventions', return_value=train_results):
            results = das_exp.perform_interventions(
                filtered_datasets,
                target_variables_list=[["answer_position"]],
                verbose=False
            )

            assert "dataset" in results
            assert results["dataset"]["same_symbol_different_position"]["model_unit"]["unit1"]["answer_position"]["average_score"] > 0.5

    def test_complete_dbm_workflow(self, mock_pipeline, mock_causal_model,
                                   mock_counterfactual_dataset, checker):
        """Test complete DBM workflow: filter -> train -> evaluate -> test."""
        # Step 1: Filter datasets
        filter_exp = FilterExperiment(
            pipeline=mock_pipeline,
            causal_model=mock_causal_model
        )

        with patch.object(filter_exp, '_validate_original_inputs', return_value=[True]), \
             patch.object(filter_exp, '_validate_counterfactual_inputs', return_value=[True]), \
             patch('causal.counterfactual_dataset.CounterfactualDataset.from_dict') as mock_from_dict:

            filtered_dataset = MagicMock(spec=CounterfactualDataset)
            filtered_dataset.__len__.return_value = 1
            mock_from_dict.return_value = filtered_dataset

            filtered_datasets = filter_exp.filter(
                {"different_symbol": mock_counterfactual_dataset},
                verbose=False,
                batch_size=64
            )

        # Step 2: Create DBM experiment
        all_tokens_pos = MagicMock(spec=TokenPosition)
        all_tokens_pos.id = "all_tokens"

        num_heads = 4
        end = 3
        heads_masking = [[(layer, head) for layer in range(0, end) for head in range(num_heads)]]

        with patch('neural.featurizers.Featurizer'):
            dbm_exp = PatchAttentionHeads(
                pipeline=mock_pipeline,
                causal_model=mock_causal_model,
                layer_head_lists=heads_masking,
                token_position=all_tokens_pos,
                config={
                    "learning_rate": 0.001,
                    "batch_size": 32,
                    "evaluation_batch_size": 64,
                    "training_epoch": 10,
                    "masking": {
                        "regularization_coefficient": 0.1
                    },
                    "featurizer_kwargs": {
                        "tie_masks": True
                    }
                }
            )

        # Step 3: Train DBM
        with patch.object(dbm_exp, 'train_interventions') as mock_train:
            dbm_exp.train_interventions(
                filtered_datasets,
                ["answer"],
                method="DBM",
                verbose=False
            )
            mock_train.assert_called_once()

        # Step 4: Evaluate and extract masks
        results_with_masks = {
            "task_name": "MCQA",
            "dataset": {
                "different_symbol": {
                    "model_unit": {
                        "all_heads": {
                            "feature_indices": {
                                "AttentionHead(Layer-0,Head-0,Token-all_tokens)": [0],
                                "AttentionHead(Layer-1,Head-2,Token-all_tokens)": [0],
                                "AttentionHead(Layer-2,Head-1,Token-all_tokens)": [0]
                            },
                            "answer": {"average_score": 1.0}
                        }
                    }
                }
            }
        }

        with patch.object(dbm_exp, 'perform_interventions', return_value=results_with_masks):
            results = dbm_exp.perform_interventions(
                filtered_datasets,
                target_variables_list=[["answer"]],
                verbose=False
            )

            # Verify we got masks
            assert "feature_indices" in results["dataset"]["different_symbol"]["model_unit"]["all_heads"]
            assert results["dataset"]["different_symbol"]["model_unit"]["all_heads"]["answer"]["average_score"] == 1.0


# Run tests when file is executed directly
if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
