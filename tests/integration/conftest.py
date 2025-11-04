"""Shared fixtures for integration tests."""

import pytest
import torch
from neural.pipeline import LMPipeline
from causal.counterfactual_dataset import CounterfactualDataset
from tasks.MCQA.causal_models import positional_causal_model
from tasks.MCQA.counterfactuals import different_symbol, same_symbol_different_position, random_counterfactual
from tasks.MCQA.token_positions import create_correct_symbol_token_position


@pytest.fixture(scope="module")
def device():
    """Get device for testing."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def pipeline(device):
    """Load Qwen model pipeline (shared across module for efficiency)."""
    model_name = "Qwen/Qwen2.5-0.5B"
    pipeline = LMPipeline(
        model_name,
        max_new_tokens=1,
        device=device,
        dtype=torch.float16,
        max_length=32
    )
    pipeline.tokenizer.padding_side = "left"
    return pipeline


@pytest.fixture(scope="module")
def causal_model():
    """Load MCQA positional causal model."""
    return positional_causal_model


@pytest.fixture
def checker():
    """Checker function for comparing neural and causal outputs."""
    def _checker(neural_output, causal_output):
        # Handle case where neural_output is a dict with 'string' key
        if isinstance(neural_output, dict) and 'string' in neural_output:
            neural_output = neural_output['string']
        return causal_output in neural_output or neural_output in causal_output
    return _checker


@pytest.fixture
def small_different_symbol_dataset():
    """Generate small different_symbol counterfactual dataset."""
    return CounterfactualDataset.from_sampler(
        8,
        different_symbol
    )


@pytest.fixture
def small_same_symbol_diff_position_dataset():
    """Generate small same_symbol_different_position counterfactual dataset."""
    return CounterfactualDataset.from_sampler(
        8,
        same_symbol_different_position
    )


@pytest.fixture
def small_random_dataset():
    """Generate small random counterfactual dataset."""
    return CounterfactualDataset.from_sampler(
        8,
        random_counterfactual
    )


@pytest.fixture
def answer_token_position(pipeline):
    """Create answer token position for the given pipeline."""
    return create_correct_symbol_token_position(pipeline)
