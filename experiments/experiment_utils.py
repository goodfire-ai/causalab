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


def compute_custom_scores(raw_results, custom_scoring_fn, metric_name="custom_metric", use_actual_outputs=False):
    """
    Compute custom metric scores from raw intervention results.

    This function applies a custom scoring function to intervention outputs,
    returning results in the same format as compute_interchange_scores() for
    compatibility with visualization tools.

    Args:
        raw_results: Dictionary from perform_interventions() containing:
            - raw_outputs: Model generation outputs (with top-K formatted scores)
            - causal_model_inputs: Base inputs and counterfactual inputs for each example
            - metadata: Model unit metadata
            - feature_indices: Selected features
        custom_scoring_fn: Function with signature:
            (causal_input, intervention_output, actual_output=None) -> float
            where:
            - causal_input: Dict with 'base_input' and 'counterfactual_inputs'
            - intervention_output: Dict with 'sequences', 'scores' (top-K format), 'string'
            - actual_output: Same format as intervention_output (if use_actual_outputs=True)
        metric_name: Name to store scores under (default: "custom_metric")
        use_actual_outputs: Whether to pass actual outputs to scoring function.
            If True, requires perform_interventions(..., include_actual_outputs=True)

    Returns:
        Dictionary with same structure as raw_results, with added field:
            results["dataset"][dataset_name]["model_unit"][unit_str][metric_name] = {
                "scores": [...],
                "average_score": float
            }

    Raises:
        ValueError: If use_actual_outputs=True but raw_outputs_no_intervention not in results

    Example:
        >>> # Step 1: Run interventions with actual outputs
        >>> raw_results = experiment.perform_interventions(
        ...     datasets, include_actual_outputs=True
        ... )
        >>>
        >>> # Step 2: Compute custom metric (e.g., logit difference)
        >>> def logit_diff_metric(causal_in, interv_out, actual_out=None):
        ...     # Extract logits from top-K format and compute difference
        ...     return diff
        >>>
        >>> results = compute_custom_scores(
        ...     raw_results,
        ...     custom_scoring_fn=logit_diff_metric,
        ...     metric_name="logit_difference",
        ...     use_actual_outputs=True
        ... )
        >>>
        >>> # Step 3: Visualize
        >>> experiment.plot_heatmaps(results, ["logit_difference"])
    """
    import copy
    import numpy as np

    # Create a deep copy to avoid modifying the input
    results = copy.deepcopy(raw_results)

    # Process each dataset and model unit combination
    for dataset_name in results["dataset"].keys():
        # Check if actual outputs are available when requested
        if use_actual_outputs:
            if "raw_outputs_no_intervention" not in results["dataset"][dataset_name]:
                raise ValueError(
                    f"use_actual_outputs=True requires actual outputs for dataset '{dataset_name}'. "
                    "Call perform_interventions(..., include_actual_outputs=True)"
                )

        actual_outputs_batches = results["dataset"][dataset_name].get("raw_outputs_no_intervention")

        for model_units_str, model_unit_data in results["dataset"][dataset_name]["model_unit"].items():
            if model_unit_data is None:
                continue

            # Get raw outputs and causal inputs
            raw_outputs = model_unit_data.get("raw_outputs")
            causal_model_inputs = model_unit_data.get("causal_model_inputs")

            if raw_outputs is None or causal_model_inputs is None:
                continue

            # Flatten batches to individual examples and apply custom scoring
            scores = []
            example_idx = 0

            for batch_idx, intervention_batch in enumerate(raw_outputs):
                actual_batch = actual_outputs_batches[batch_idx] if actual_outputs_batches else None
                batch_size = intervention_batch["sequences"].shape[0]

                for i in range(batch_size):
                    if example_idx >= len(causal_model_inputs):
                        break

                    # Create intervention output dict for this example
                    intervention_output = {"sequences": intervention_batch["sequences"][i:i+1]}

                    # Handle string field
                    if "string" in intervention_batch:
                        if isinstance(intervention_batch["string"], list):
                            intervention_output["string"] = intervention_batch["string"][i]
                        else:
                            intervention_output["string"] = intervention_batch["string"]

                    # Handle top-K formatted scores
                    if "scores" in intervention_batch and intervention_batch["scores"]:
                        intervention_output["scores"] = []
                        for score_dict in intervention_batch["scores"]:
                            sliced_score = {
                                "top_k_logits": score_dict["top_k_logits"][i:i+1],
                                "top_k_indices": score_dict["top_k_indices"][i:i+1],
                                "top_k_tokens": [score_dict["top_k_tokens"][i]]
                            }
                            intervention_output["scores"].append(sliced_score)

                    # Create actual output dict if requested
                    actual_output = None
                    if use_actual_outputs and actual_batch is not None:
                        actual_output = {"sequences": actual_batch["sequences"][i:i+1]}

                        if "string" in actual_batch:
                            if isinstance(actual_batch["string"], list):
                                actual_output["string"] = actual_batch["string"][i]
                            else:
                                actual_output["string"] = actual_batch["string"]

                        # Handle top-K formatted scores
                        if "scores" in actual_batch and actual_batch["scores"]:
                            actual_output["scores"] = []
                            for score_dict in actual_batch["scores"]:
                                sliced_score = {
                                    "top_k_logits": score_dict["top_k_logits"][i:i+1],
                                    "top_k_indices": score_dict["top_k_indices"][i:i+1],
                                    "top_k_tokens": [score_dict["top_k_tokens"][i]]
                                }
                                actual_output["scores"].append(sliced_score)

                    # Apply custom scoring function
                    if use_actual_outputs:
                        score = custom_scoring_fn(
                            causal_model_inputs[example_idx],
                            intervention_output,
                            actual_output=actual_output
                        )
                    else:
                        score = custom_scoring_fn(
                            causal_model_inputs[example_idx],
                            intervention_output
                        )

                    # Convert tensor scores to float
                    if isinstance(score, torch.Tensor):
                        score = score.item()
                    scores.append(float(score))

                    example_idx += 1

            # Store results in the same structure as compute_interchange_scores
            results["dataset"][dataset_name]["model_unit"][model_units_str][metric_name] = {
                "scores": scores,
                "average_score": float(np.mean(scores)) if scores else 0.0
            }

    return results
