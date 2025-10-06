"""
Integration tests for Notebook 02: Residual Stream Tracing

Tests loading the Qwen model and running SameLengthResidualStreamTracing
to trace information flow through the residual stream.
"""

import pytest
import copy
import random
import torch
from tasks.MCQA.mcqa import sample_answerable_question
from experiments.LM_experiments.residual_stream_experiment import SameLengthResidualStreamTracing


pytestmark = [pytest.mark.slow, pytest.mark.gpu]


class TestModelLoading:
    """Test loading the Qwen model via LMPipeline."""

    def test_pipeline_loaded(self, pipeline):
        """Test that pipeline is loaded correctly."""
        assert pipeline is not None
        assert pipeline.model is not None
        assert pipeline.tokenizer is not None

    def test_pipeline_device(self, pipeline, device):
        """Test that model is on expected device."""
        assert str(pipeline.model.device).startswith(device.split(":")[0])

    def test_pipeline_generation(self, pipeline):
        """Test basic generation with the pipeline."""
        test_input = "The sky is blue. What color is the sky?\nA. red\nB. blue\nAnswer:"
        output = pipeline.generate(test_input)

        # Verify output is a dict with sequences
        assert isinstance(output, dict)
        assert "sequences" in output
        assert isinstance(output["sequences"], torch.Tensor)
        assert output["sequences"].numel() > 0

    def test_pipeline_dump(self, pipeline):
        """Test decoding output with pipeline.dump."""
        test_input = "Test"
        output = pipeline.generate(test_input)
        decoded = pipeline.dump(output["sequences"])

        # Verify decoded output is a string
        assert isinstance(decoded, str)


class TestSameLengthResidualStreamTracing:
    """Test residual stream tracing experiments."""

    def test_create_tracing_experiment(self, pipeline, causal_model, checker):
        """Test creating SameLengthResidualStreamTracing experiment."""
        tracing_exp = SameLengthResidualStreamTracing(
            pipeline=pipeline,
            causal_model=causal_model,
            checker=checker
        )

        assert tracing_exp is not None
        assert tracing_exp.pipeline == pipeline
        assert tracing_exp.causal_model == causal_model
        assert tracing_exp.checker == checker

    def test_run_tracing_on_valid_pair(self, pipeline, causal_model, checker):
        """Test running tracing experiment on a valid input pair."""
        # Sample original and counterfactual
        original = sample_answerable_question()
        full_setting = causal_model.run_forward(original)

        # Create a counterfactual by changing the answer symbol
        counterfactual = copy.deepcopy(original)
        answer_symbol_key = f"symbol{full_setting['answer_position']}"
        new_symbols = list(
            {"A", "B", "C"}.difference({full_setting[answer_symbol_key]})
        )
        counterfactual[answer_symbol_key] = random.choice(new_symbols)
        del counterfactual["raw_input"]
        counterfactual_setting = causal_model.run_forward(counterfactual)
        counterfactual["raw_input"] = counterfactual_setting["raw_input"]

        # Create tracing experiment
        tracing_exp = SameLengthResidualStreamTracing(
            pipeline=pipeline,
            causal_model=causal_model,
            checker=checker
        )

        # Run tracing
        results = tracing_exp.run(
            base_input=original,
            counterfactual_input=counterfactual
        )

        # Verify results structure
        assert results is not None
        assert "dataset" in results
        assert "task_name" in results
        assert "method_name" in results
        assert "model_name" in results

        # Verify there are model_unit entries
        assert len(results["dataset"]) > 0

    def test_tracing_results_structure(self, pipeline, causal_model, checker):
        """Test that tracing results have expected structure."""
        # Sample inputs
        original = sample_answerable_question()
        full_setting = causal_model.run_forward(original)

        counterfactual = copy.deepcopy(original)
        answer_symbol_key = f"symbol{full_setting['answer_position']}"
        new_symbols = list(
            {"A", "B", "C"}.difference({full_setting[answer_symbol_key]})
        )
        counterfactual[answer_symbol_key] = random.choice(new_symbols)
        del counterfactual["raw_input"]
        counterfactual_setting = causal_model.run_forward(counterfactual)
        counterfactual["raw_input"] = counterfactual_setting["raw_input"]

        # Run tracing
        tracing_exp = SameLengthResidualStreamTracing(
            pipeline=pipeline,
            causal_model=causal_model,
            checker=checker
        )
        results = tracing_exp.run(
            base_input=original,
            counterfactual_input=counterfactual
        )

        # Check that we have results for all layers and positions
        num_layers = pipeline.get_num_layers()
        assert num_layers > 0

        # Verify dataset structure
        dataset_results = results["dataset"]
        assert len(dataset_results) > 0

        # Check that model_units have the expected keys
        first_dataset_key = next(iter(dataset_results.keys()))
        model_units = dataset_results[first_dataset_key]["model_unit"]
        assert len(model_units) > 0

        # Check structure of a model unit result
        first_unit_key = next(iter(model_units.keys()))
        unit_result = model_units[first_unit_key]
        assert "raw_outputs" in unit_result
        assert "metadata" in unit_result

    def test_tracing_with_same_length_inputs(self, pipeline, causal_model, checker):
        """Test that tracing requires same-length inputs."""
        # Sample two inputs
        original = sample_answerable_question()
        counterfactual = sample_answerable_question()

        # Tokenize using pipeline.load() to match what run() does
        base_ids = pipeline.load(original)
        cf_ids = pipeline.load(counterfactual)

        base_tokens = pipeline.tokenizer.convert_ids_to_tokens(base_ids['input_ids'][0])
        cf_tokens = pipeline.tokenizer.convert_ids_to_tokens(cf_ids['input_ids'][0])

        tracing_exp = SameLengthResidualStreamTracing(
            pipeline=pipeline,
            causal_model=causal_model,
            checker=checker
        )

        if len(base_tokens) != len(cf_tokens):
            # Expect an error when lengths don't match
            with pytest.raises(ValueError, match="must have the same length"):
                tracing_exp.run(
                    base_input=original,
                    counterfactual_input=counterfactual
                )
        else:
            # If by chance they're the same length, should succeed
            results = tracing_exp.run(
                base_input=original,
                counterfactual_input=counterfactual
            )
            assert results is not None
