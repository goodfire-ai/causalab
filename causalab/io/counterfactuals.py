"""Save, load, and display counterfactual examples."""

from __future__ import annotations

import json
import os
import struct
from typing import TYPE_CHECKING, Any, Dict, Mapping, Sequence

from causalab.causal.counterfactuals import CounterfactualExample
from causalab.io._counterfactual_values import decode_values, encode_values

if TYPE_CHECKING:
    from causalab.causal.model import CausalModel


def _decode_value(value):
    """Decode version-one values; new saves use the version-two codec."""
    if isinstance(value, dict):
        if set(value) == {"float64"}:
            bits = value["float64"]
            if (
                not isinstance(bits, str)
                or len(bits) != 16
                or any(character not in "0123456789abcdefABCDEF" for character in bits)
            ):
                raise ValueError("Invalid saved float64 value: expected 16 hex digits")
            return struct.unpack("!d", bytes.fromhex(bits))[0]
        if set(value) == {"tuple"}:
            return tuple(_decode_value(item) for item in value["tuple"])
        if set(value) == {"mapping"}:
            return {
                _decode_value(key): _decode_value(item)
                for key, item in value["mapping"]
            }
        raise ValueError("Unknown saved value format")
    if isinstance(value, list):
        return [_decode_value(item) for item in value]
    return value


def _restore_json_value(value, domain, name):
    """Restore tuple values in older JSON files using the declared domain."""
    if domain.contains(value):
        return value
    if domain.kind == "sequence" and isinstance(value, list):
        value = domain.type(
            _restore_json_value(item, domain.element, name) for item in value
        )
    elif domain.kind == "union":
        for member in domain.members:
            try:
                return _restore_json_value(value, member, name)
            except ValueError:
                continue
    elif domain.kind == "type" and domain.type is tuple and isinstance(value, list):
        value = tuple(value)
    elif domain.kind == "finite" and not isinstance(domain.values, range):
        encoded = json.dumps(value)
        for candidate in domain.values:
            try:
                if json.dumps(candidate) == encoded:
                    return candidate
            except (TypeError, ValueError):
                continue
    domain.validate(value, name)
    return value


def _restore_trace(data, model):
    if not isinstance(data, dict):
        return data
    if set(data) == {"version", "values", "interventions"}:
        if data["version"] == 1:
            values = {
                name: _decode_value(value) for name, value in data["values"].items()
            }
        elif data["version"] == 2:
            values = decode_values(data["values"])
        else:
            raise ValueError(f"Unknown saved trace version: {data['version']!r}")
        interventions = set(data["interventions"])
        if interventions - values.keys():
            raise ValueError("Saved interventions must include their values")
    else:
        values = model._flatten(data)
        interventions = set()
        unknown = values.keys() - model.domains.keys()
        if unknown:
            raise ValueError(f"Unknown saved variables: {sorted(unknown)}")
        values = {
            name: _restore_json_value(value, model.domains[name], name)
            for name, value in values.items()
            if name in model.inputs
        }
    unknown = values.keys() - model.domains.keys()
    if unknown:
        raise ValueError(f"Unknown saved variables: {sorted(unknown)}")
    # Computed caches are rebuilt from inputs and recorded interventions.
    trace = model.new_trace(
        {
            name: value
            for name, value in values.items()
            if name in model.inputs or name in interventions
        }
    )
    for name in interventions:
        trace._override(name, values[name])
    return trace


def save_counterfactual_examples(
    examples: list[CounterfactualExample],
    path: str,
) -> None:
    """Save typed acyclic values and active interventions in version-two JSON."""

    def serialize_trace(trace):
        return {
            "version": 2,
            "values": encode_values(trace.snapshot()),
            "interventions": sorted(trace._overrides),
        }

    def serialize_example(ex: CounterfactualExample) -> dict[str, Any]:
        return {
            "input": serialize_trace(ex["input"]),
            "counterfactual_inputs": [
                serialize_trace(trace) for trace in ex["counterfactual_inputs"]
            ],
        }

    serialized = [serialize_example(ex) for ex in examples]
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w") as f:
        json.dump(serialized, f, indent=2)


def load_counterfactual_examples(
    path: str, causal_model: "CausalModel"
) -> list[CounterfactualExample]:
    """Load counterfactual examples from disk and rehydrate CausalTrace objects.

    Supports both list format ``[{"input": ..., "counterfactual_inputs": [...]}, ...]``
    and dict format ``{"input": [...], "counterfactual_inputs": [[...], ...]}``.
    Versions one and two preserve tuple values and recorded interventions;
    version two also preserves shared contents and numeric NumPy types. Plain
    value dictionaries are observations; their computed values are rebuilt.
    """
    with open(path) as f:
        data = json.load(f)

    if isinstance(data, dict) and "input" in data and "counterfactual_inputs" in data:
        data = [
            {"input": inp, "counterfactual_inputs": cf}
            for inp, cf in zip(data["input"], data["counterfactual_inputs"])
        ]

    return deserialize_counterfactual_examples(data, causal_model)


def deserialize_counterfactual_examples(
    dataset: Sequence[Mapping[str, Any]], causal_model: "CausalModel"
) -> list[CounterfactualExample]:
    """Convert dicts loaded from disk back to CausalTrace objects."""
    result = []
    for example in dataset:
        input_data = example["input"]
        cf_inputs_data = example["counterfactual_inputs"]

        input_trace = _restore_trace(input_data, causal_model)
        cf_traces = [
            _restore_trace(cf_data, causal_model) for cf_data in cf_inputs_data
        ]

        result.append({"input": input_trace, "counterfactual_inputs": cf_traces})

    return result


def display_counterfactual_examples(
    examples: Sequence[Mapping[str, Any]],
    num_examples: int = 1,
    verbose: bool = True,
    name: str = "dataset",
) -> Dict[int, Mapping[str, Any]]:
    """
    Display examples from a list of counterfactual examples.

    Args:
        examples (list): List of counterfactual example dicts.
        num_examples (int, optional): Number of examples to display. Defaults to 1.
        verbose (bool, optional): Whether to print information. Defaults to True.
        name (str, optional): Name to display for the dataset. Defaults to "dataset".

    Returns:
        dict: A dictionary mapping indices to displayed examples.
    """
    if verbose:
        print(f"Dataset '{name}':")

    displayed_examples: Dict[int, Mapping[str, Any]] = {}

    for i in range(min(num_examples, len(examples))):
        example = examples[i]

        if verbose:
            print(f"\nExample {i + 1}:")
            print(f"Input: {example['input']}")
            print(
                f"Counterfactual Inputs ({len(example['counterfactual_inputs'])} alternatives):"
            )

            for j, counterfactual_input in enumerate(example["counterfactual_inputs"]):
                print(f"  [{j + 1}] {counterfactual_input}")

        displayed_examples[i] = example

    if verbose and len(examples) > num_examples:
        print(f"\n... {len(examples) - num_examples} more examples not shown")

    return displayed_examples
