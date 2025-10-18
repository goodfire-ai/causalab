"""Utility functions for working with causal models."""


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
