"""Sample counterfactual examples and label their outputs."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence, TypedDict

if TYPE_CHECKING:
    from causalab.causal.model import CausalModel, CausalTrace


class _PairIdentity(TypedDict, total=False):
    example_id: str
    pair_id: str
    family: str
    base_id: str
    donor_id: str


class CounterfactualExample(_PairIdentity):
    """
    Type for counterfactual example dictionaries.

    Each example contains:
        input: The base input as a CausalTrace
        counterfactual_inputs: List of counterfactual inputs as CausalTraces
    """

    input: "CausalTrace"
    counterfactual_inputs: list["CausalTrace"]


class LabeledCounterfactualExample(CounterfactualExample):
    """
    Counterfactual example with a ground truth label for training.

    Training functions use the label to compute loss and accuracy.
    The label is the expected output after intervention.
    """

    label: Any


def generate_counterfactual_samples(
    size: int,
    sampler: Callable[[], CounterfactualExample],
    filter: Callable[[CounterfactualExample], bool] | None = None,
) -> list[CounterfactualExample]:
    """
    Generate a list of counterfactual examples.

    Args:
        size: Number of examples to generate.
        sampler: Function that returns a CounterfactualExample.
        filter: Function that takes a CounterfactualExample and returns
                                    a boolean indicating whether to include it.

    Returns:
        List of counterfactual examples.
    """
    examples = []
    while len(examples) < size:
        sample = sampler()
        if filter is None or filter(sample):
            examples.append(sample)

    return examples


def sample_intervention(
    model: "CausalModel", filter_func: Callable[[dict[str, Any]], bool] | None = None
) -> dict[str, Any]:
    """
    Sample a random intervention that satisfies an optional filter.

    Args:
        model: The causal model to sample from.
        filter_func: A function that takes an intervention and returns a boolean indicating
            whether it satisfies the filter (default is None).

    Returns:
        A dictionary mapping variables to their sampled intervention values.
    """
    filter_func = (
        filter_func if filter_func is not None else lambda x: len(x.keys()) > 0
    )
    intervention: dict[str, Any] = {}
    while not filter_func(intervention):
        intervention = {}
        while len(intervention.keys()) == 0:
            for var in model.variables:
                if var in model.inputs or var in model.outputs:
                    continue
                if random.choice([0, 1]) == 0:
                    intervention[var] = model.domains[var].sample(random)
    return intervention


def label_data_with_variables(
    model: "CausalModel",
    data: Sequence[Mapping[str, Any]],
    target_variables: list[str],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """
    Labels a dataset based on variable settings from running the forward model.

    Takes a dataset of inputs, runs the forward model on each input, and assigns
    a unique label ID based on the values of the specified target variables.

    Args:
        model: The causal model to use for labeling.
        data: List containing examples with "input" field.
        target_variables: List of variable names to use for labeling.

    Returns:
        A tuple containing:

            - list[dict]: A list of dicts with "input" and "label" fields.
            - dict: A mapping from concatenated target variable values to label IDs.
    """
    traces = []
    labels = []
    label_to_setting: dict[str, int] = {}

    new_id = 0
    for example in data:
        trace = example["input"]
        # Store input
        traces.append(trace)

        target_labels = [str(trace[var]) for var in target_variables]

        # Assign or create a label ID
        label_key = "".join(target_labels)
        if label_key in label_to_setting:
            id_value = label_to_setting[label_key]
        else:
            id_value = new_id
            label_to_setting[label_key] = new_id
            new_id += 1

        labels.append(id_value)

    # Return list of dicts with input and label
    labeled_data = [
        {
            "input": t.snapshot()
            if hasattr(t, "snapshot")
            else t.to_dict()
            if hasattr(t, "to_dict")
            else t,
            "label": label,
        }
        for t, label in zip(traces, labels)
    ]
    return labeled_data, label_to_setting


def get_partial_filter(
    partial_setting: dict[str, Any],
) -> Callable[[dict[str, Any]], bool]:
    """
    Get a filter function that checks if a setting matches a partial setting.

    Args:
        partial_setting: A dictionary mapping variables to their desired values.

    Returns:
        A filter function that takes a setting and returns a boolean.
    """

    def compare(total_setting: dict[str, Any]) -> bool:
        for var in partial_setting:
            if total_setting[var] != partial_setting[var]:
                return False
        return True

    return compare


def label_counterfactual_data(
    model: "CausalModel",
    examples: list[CounterfactualExample],
    target_variables: list[str],
    label_variable: str = "raw_output",
    *,
    setting_variables: Sequence[str] = (),
) -> list[dict[str, Any]]:
    """
    Labels examples with results from running interchange interventions.

    Takes examples containing inputs and counterfactual inputs, runs interchange
    interventions using the specified target variables, and returns examples
    with labeled outputs.

    Args:
        model: The causal model that runs the interventions.
        examples: List of examples with "input" and "counterfactual_inputs" fields.
        target_variables: List of variable names to use for interchange.
        label_variable: The variable whose intervened value is the label.
        setting_variables: Additional variables to compute in the exported
            setting. Unrequested lazy equations remain unevaluated.

    Returns:
        The examples with "label" and "setting" fields added.
    """
    if isinstance(setting_variables, str):
        raise TypeError("setting_variables must be a sequence of variable names")
    setting_variables = tuple(setting_variables)
    labels: list[Any] = []
    settings: list[CausalTrace] = []

    for example in examples:
        trace: CausalTrace = example["input"]
        counterfactual_traces: list[CausalTrace] = example["counterfactual_inputs"]

        # Handle target_variables element by element
        # Each element can be either a single variable name (str) or a list of variable names
        # If we have exactly one counterfactual but multiple target variables,
        # extend counterfactual_inputs by repeating the single counterfactual
        if len(counterfactual_traces) == 1 and len(target_variables) > 1:
            counterfactual_traces = counterfactual_traces * len(target_variables)

        assert len(target_variables) <= len(counterfactual_traces), (
            f"target_variables has {len(target_variables)} elements but counterfactual_traces only has {len(counterfactual_traces)}"
        )

        counterfactual_dict: dict[str, CausalTrace] = {}
        for i, var_element in enumerate(target_variables):
            cf_trace = counterfactual_traces[i]

            if isinstance(var_element, list):
                # Element is a list of variables: assign counterfactual[i] to all variables in the list
                for var in var_element:
                    counterfactual_dict[var] = cf_trace
            else:
                # Element is a single variable: assign counterfactual[i] to this variable
                counterfactual_dict[var_element] = cf_trace

        # Perform interchange using run_interchange (supports A<-B syntax)
        setting = model.run_interchange(trace, counterfactual_dict)
        labels.append(setting[label_variable])
        settings.append(setting)

    # Build result list with labels and settings added
    result: list[dict[str, Any]] = []
    for i, example in enumerate(examples):
        result.append(
            {
                **example,
                "label": labels[i],
                "setting": settings[i].snapshot(required=setting_variables),
            }
        )

    return result
