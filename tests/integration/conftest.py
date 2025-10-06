"""Shared fixtures for integration tests."""

import pytest
import torch
from neural.pipeline import LMPipeline
from tasks.MCQA.mcqa import MCQA_task
from causal.counterfactual_dataset import CounterfactualDataset


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
    return MCQA_task.causal_models["positional"]


@pytest.fixture
def checker():
    """Checker function for comparing neural and causal outputs."""
    def _checker(neural_output, causal_output):
        return causal_output in neural_output or neural_output in causal_output
    return _checker


@pytest.fixture
def small_different_symbol_dataset():
    """Generate small different_symbol counterfactual dataset."""
    return CounterfactualDataset.from_sampler(
        8,
        MCQA_task.dataset_generators["different_symbol"]
    )


@pytest.fixture
def small_same_symbol_diff_position_dataset():
    """Generate small same_symbol_different_position counterfactual dataset."""
    return CounterfactualDataset.from_sampler(
        8,
        MCQA_task.dataset_generators["same_symbol_different_position"]
    )


@pytest.fixture
def small_random_dataset():
    """Generate small random counterfactual dataset."""
    return CounterfactualDataset.from_sampler(
        8,
        MCQA_task.dataset_generators["random_counterfactual"]
    )
