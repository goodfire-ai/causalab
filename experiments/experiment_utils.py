"""
Shared utilities for running causal abstraction experiments.

This module provides common functionality used across different experiment types,
including memory management, directory creation, and path generation.
"""

import os
import gc
import torch


def clear_memory():
    """
    Free memory between experiments to prevent OOM errors.

    This function performs three steps:
    1. Runs Python's garbage collector to free unreferenced objects
    2. Clears CUDA cache if GPU is available
    3. Synchronizes CUDA operations to ensure memory is actually freed
    """
    # Clear Python garbage collector
    gc.collect()

    # Clear CUDA cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Force a synchronization point to ensure memory is freed
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def ensure_dir(path: str) -> None:
    """
    Create directory if it doesn't exist.

    Args:
        path: Directory path to create
    """
    if path and not os.path.exists(path):
        os.makedirs(path)


def generate_model_dir_name(method_name: str, model_name: str, target_variables: list) -> str:
    """
    Generate a standardized directory name for saving trained models.

    Args:
        method_name: Name of the intervention method (e.g., "DAS", "DBM")
        model_name: Name of the model class (e.g., "GPT2LMHeadModel")
        target_variables: List of target variable names

    Returns:
        Formatted directory name string

    Example:
        >>> generate_model_dir_name("DAS", "GPT2LMHeadModel", ["pos", "answer"])
        "DAS_GPT2LMHeadModel__pos-answer"
    """
    variables_str = "-".join(target_variables)
    return f"{method_name}_{model_name}__{variables_str}"
