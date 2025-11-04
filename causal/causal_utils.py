"""Utility functions for working with causal models."""

from causal.counterfactual_dataset import CounterfactualDataset
from typing import List, Dict, Tuple, Optional, Callable, Union
import copy
import numpy as np
import torch


def can_distinguish_with_dataset(
    dataset,
    causal_model1,
    target_variables1,
    causal_model2=None,
    target_variables2=None
):
    """
    Check if two causal models can be distinguished using interchange interventions
    on a counterfactual dataset.

    Compares the outputs from running interchange interventions with target_variables1
    on causal_model1 against either:
    - Interchange interventions with target_variables2 on causal_model2 (if provided)
    - The forward pass output of causal_model1 (if causal_model2 is None)

    Parameters:
    -----------
    dataset : Dataset
        Dataset containing "input" and "counterfactual_inputs" fields.
    causal_model1 : CausalModel
        The first causal model to run interchange interventions on.
    target_variables1 : list
        List of variable names to use for interchange in the first model.
    causal_model2 : CausalModel, optional
        The second causal model to compare against (default is None).
    target_variables2 : list, optional
        List of variable names to use for interchange in the second model.
        Only used if causal_model2 is provided (default is None).

    Returns:
    --------
    dict
        A dictionary containing:
            - "proportion": The proportion of examples where outputs differ
            - "count": The number of examples where outputs differ
    """
    count = 0
    for example in dataset:
        input_data = example["input"]
        counterfactual_inputs = example["counterfactual_inputs"]
        assert len(counterfactual_inputs) == 1

        # Run interchange intervention on first model
        setting1 = causal_model1.run_interchange(
            input_data,
            {var: counterfactual_inputs[0] for var in target_variables1}
        )

        if causal_model2 is not None and target_variables2 is not None:
            # Run interchange intervention on second model
            setting2 = causal_model2.run_interchange(
                input_data,
                {var: counterfactual_inputs[0] for var in target_variables2}
            )
            if setting1["raw_output"] != setting2["raw_output"]:
                count += 1
        else:
            # Compare against forward pass of first model
            if setting1["raw_output"] != causal_model1.run_forward(input_data)["raw_output"]:
                count += 1

    proportion = count / len(dataset)
    print(f"Can distinguish between {target_variables1} and {target_variables2}: {count} out of {len(dataset)} examples")
    print(f"Proportion of distinguishable examples: {proportion:.2f}")
    return {"proportion": proportion, "count": count}

def statement_conjunction_function(filled_statements: List, delimiters: list) -> str:
    """
    Combine multiple filled statements into a single conjunction.

    Args:
        filled_statements: List of filled statement strings
        delimiters: List of delimiters to use between statements
    
    Returns:
        A single string combining all statements with proper punctuation, seen below:
        "Statement one delimiter one statement two delimiter two ... statement N delimiter N+1."

    """
    #Capitalize first letter and ensure it ends with a period.
    fill_index = delimiters.index("FILL")
    filler = delimiters[fill_index-1]
    new_delimiters = delimiters[:fill_index-1] + delimiters[fill_index+1:]
    for _ in range(len(filled_statements) - len(new_delimiters)):
        new_delimiters.insert(fill_index-1, filler)
    
    if len(new_delimiters) > len(filled_statements):
        new_delimiters = new_delimiters[-len(filled_statements):]

    statements = []
    for i in range(len(filled_statements)):
        statement = filled_statements[i]
        # Decompose into words
        words = statement.split()
        # Capitalize first letter and ensure it ends with a period.
        words[0] = words[0].capitalize()
        statements.append(' '.join(words).rstrip(new_delimiters[-1]))
    conjunction = statements[0] 
    for i in range(1, len(statements)):
        conjunction += new_delimiters[i-1] + statements[i]
    conjunction += new_delimiters[-1]
    return conjunction


def compute_interchange_scores(
    raw_results: Dict,
    causal_model,
    datasets: Union[Dict, 'CounterfactualDataset'],
    target_variables_list: List[List[str]],
    checker: Callable
) -> Dict:
    """
    Process raw intervention results by computing scores for target variables.

    This function takes the raw outputs from perform_interventions and adds
    target-variable-specific score fields to the results dictionary. It matches
    the exact data structure that perform_interventions would create if
    target_variables_list was passed directly, allowing all existing visualization
    code to work without changes.

    This separation allows you to:
    1. Run expensive interventions once
    2. Analyze results with different target_variables combinations
    3. Experiment with different causal model interpretations post-hoc

    Args:
        raw_results: Dictionary from perform_interventions containing:
            - raw_outputs: Model generation outputs (sequences, scores, strings)
            - causal_model_inputs: Base inputs and counterfactual inputs for each example
            - metadata: Model unit metadata (layer, position, etc.)
            - feature_indices: Selected features for each model unit
        causal_model: CausalModel used to generate expected outputs via label_counterfactual_data
        datasets: Dictionary mapping dataset names to CounterfactualDataset objects,
                 or single CounterfactualDataset (will be converted to dict)
        target_variables_list: List of target variable groups to evaluate.
                              Each group is a list of variable names to intervene on.
        checker: Function with signature (output_dict, expected_label) -> score
                Used to compare model outputs against causal model expectations.

    Returns:
        Dictionary with same structure as raw_results, but with added fields for each
        target variable group:
            results["dataset"][dataset_name]["model_unit"][unit_str][target_var_str] = {
                "scores": [...],           # List of scores for each example
                "average_score": 0.85      # Mean score across all examples
            }

    Example:
        >>> # Step 1: Run interventions once (expensive)
        >>> raw_results = experiment.perform_interventions(datasets, save_dir="./results")
        >>>
        >>> # Step 2: Try different target variable combinations (cheap)
        >>> results_A = compute_interchange_scores(
        ...     raw_results, causal_model, datasets,
        ...     target_variables_list=[["A"]], checker=exact_match
        ... )
        >>> results_AB = compute_interchange_scores(
        ...     raw_results, causal_model, datasets,
        ...     target_variables_list=[["A", "B"]], checker=exact_match
        ... )
        >>>
        >>> # Step 3: Visualize both (same visualization code)
        >>> experiment.plot_heatmaps(results_A, target_variables=["A"])
        >>> experiment.plot_heatmaps(results_AB, target_variables=["A", "B"])
    """
    # Convert single dataset to dictionary
    if isinstance(datasets, CounterfactualDataset):
        datasets = {datasets.id: datasets}

    # Create a deep copy to avoid modifying the input
    results = copy.deepcopy(raw_results)

    # Process each dataset and model unit combination
    for dataset_name in datasets.keys():
        if dataset_name not in results["dataset"]:
            continue

        for model_units_str, model_unit_data in results["dataset"][dataset_name]["model_unit"].items():
            if model_unit_data is None:
                continue

            # Get raw outputs and causal inputs
            raw_outputs = model_unit_data.get("raw_outputs")
            causal_model_inputs = model_unit_data.get("causal_model_inputs")

            if raw_outputs is None or causal_model_inputs is None:
                continue

            # Process and decode model outputs from batch dictionaries
            dumped_outputs = []
            flattened_outputs = []
            for batch_dict in raw_outputs:
                # Use the string field that's already in batch_dict
                batch_strings = batch_dict["string"]
                # Always treat as a list for consistent processing
                if not isinstance(batch_strings, list):
                    batch_strings = [batch_strings]

                dumped_outputs.extend(batch_strings)
                # Create individual output dicts for each example in the batch
                for idx, decoded_str in enumerate(batch_strings):
                    example_dict = {"sequences": batch_dict["sequences"][idx:idx+1]}

                    # Handle top-K formatted scores (list of dicts)
                    if "scores" in batch_dict and batch_dict["scores"]:
                        example_dict["scores"] = []
                        for score_dict in batch_dict["scores"]:
                            sliced_score = {
                                "top_k_logits": score_dict["top_k_logits"][idx:idx+1],
                                "top_k_indices": score_dict["top_k_indices"][idx:idx+1],
                                "top_k_tokens": [score_dict["top_k_tokens"][idx]]
                            }
                            example_dict["scores"].append(sliced_score)

                    example_dict["string"] = decoded_str
                    flattened_outputs.append(example_dict)

            # Evaluate results for each target variable group
            # This replicates the logic from intervention_experiment.py lines 219-239
            for target_variables in target_variables_list:
                target_variable_str = "-".join(target_variables)

                # Generate expected outputs from causal model
                labeled_data = causal_model.label_counterfactual_data(
                    datasets[dataset_name],
                    target_variables
                )

                # Validate alignment
                assert len(labeled_data) == len(dumped_outputs), \
                    f"Length mismatch: {len(labeled_data)} vs {len(dumped_outputs)}"
                assert len(labeled_data) == len(flattened_outputs), \
                    f"Length mismatch: {len(labeled_data)} vs {len(flattened_outputs)}"

                # Compute intervention scores - pass neural dict and expected label
                scores = []
                for example, output_dict in zip(labeled_data, flattened_outputs):
                    score = checker(output_dict, example["label"])
                    if isinstance(score, torch.Tensor):
                        score = score.item()
                    scores.append(float(score))

                # Store processed results in the same structure as perform_interventions
                results["dataset"][dataset_name]["model_unit"][model_units_str][target_variable_str] = {
                    "scores": scores,
                    "average_score": np.mean(scores)
                }

    return results