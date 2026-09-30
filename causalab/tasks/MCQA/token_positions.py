"""Token position functions for the MCQA task.

This module provides functions to locate specific tokens in MCQA prompts,
such as answer symbols, periods, and the last token.
"""

from typing import Any, Callable

from causalab.tasks.token_positions import LMPipeline
from causalab.tasks.token_positions import (
    TokenPosition,
    build_token_positions,
)

from .causal_models import NUM_CHOICES, TEMPLATES

# Type alias for token position specs - either a static dict or a dynamic function
TokenPositionSpec = dict[str, Any] | Callable[[dict[str, Any]], dict[str, Any]]


def create_token_positions(
    pipeline: LMPipeline, template: str | None = None
) -> dict[str, TokenPosition]:
    """
    Create all token positions for the MCQA task.

    Args:
        pipeline: The tokenizer pipeline
        template: The template string (optional, uses default from causal_models if not provided)

    Returns:
        dict: Dictionary mapping token position names to TokenPosition objects
    """
    # Use default template if not provided
    if template is None:
        template = TEMPLATES[0]

    def find_correct_symbol(setting: dict[str, Any]) -> str:
        for i in range(NUM_CHOICES):
            if setting[f"choices[{i}]"] == setting["color"]:
                return f"symbols[{i}]"

        choices = [setting[f"choices[{i}]"] for i in range(NUM_CHOICES)]
        raise ValueError(
            f"No correct symbol found for color {setting['color']} in choices {choices} with setting {setting}"
        )

    # Define token position specifications using the new declarative system
    token_position_specs: dict[str, TokenPositionSpec] = {
        # Dynamic position: correct answer symbol
        "correct_symbol": lambda setting: {
            "type": "variable",
            "name": find_correct_symbol(setting),
        },
        # Dynamic position: period after correct answer symbol
        "correct_symbol_period": lambda setting: {
            "type": "index",
            "position": +1,
            "relative_to": {"variable": find_correct_symbol(setting)},
        },
    }

    # Add symbol positions for each choice
    for i in range(NUM_CHOICES):
        # Symbol itself
        token_position_specs[f"symbols[{i}]"] = {
            "type": "variable",
            "name": f"symbols[{i}]",
        }

        # Period after symbol
        token_position_specs[f"symbol{i}_period"] = {
            "type": "index",
            "position": +1,
            "relative_to": {"variable": f"symbols[{i}]"},
        }

    # Add last token in the sequence last:
    token_position_specs["last_token"] = {"type": "index", "position": -1}

    # Build the declarative specs and materialize them into TokenPosition instances.
    return build_token_positions(token_position_specs, template, pipeline)
