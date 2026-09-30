"""
Token position definitions for the task.

Uses custom Python functions because this is an ICL task with repeated patterns.
The declarative system would find the LAST occurrence of quoted letters, which
may not correspond to the correct variable in the test query.
"""

import re
from typing import Dict

from causalab.tasks.token_positions import LMPipeline
from causalab.tasks.token_positions import (
    TokenPosition,
    rebase_char_range,
    get_last_token_index,
    get_tokens_in_char_range,
)


def _get_var_token_indices(input_sample, pipeline, var_idx, prompt_mode):
    """Find token indices for the var_idx-th variable (0-3) in the test query."""
    raw_input = input_sample["raw_input"]

    # Find the test query (last line)
    last_line = raw_input.rstrip().split("\n")[-1]
    line_start = raw_input.rfind(last_line)

    # Match variable values depending on prompt format
    if prompt_mode == "minimal_function":
        # f(A,B,C,D)= — bare comma-separated args inside parens
        matches = list(re.finditer(r"(?<=[(,])([^,)]+)", last_line))
    elif prompt_mode == "algorithmic":
        # A B C D: — space-separated letters before the colon
        matches = list(re.finditer(r"(\S+)(?=\s)", last_line))[:4]
    else:
        # Code mode: func("A", "B", "C", "D") — quoted values
        matches = list(re.finditer(r'"([^"]*)"', last_line))
    if var_idx >= len(matches):
        raise ValueError(
            f"Variable index {var_idx} not found in test query: {last_line}"
        )

    match = matches[var_idx]

    # Character position of the inner value (the letter) in the full text
    char_start = line_start + match.start(1)
    char_end = line_start + match.end(1)

    # Map character positions to token indices using offset_mapping
    tokenized = pipeline.load([input_sample], return_offsets_mapping=True)
    offsets = tokenized["offset_mapping"][0]

    # Under a chat template the offsets index the wrapped prompt, so rebase the
    # bare char range by the chat prefix (and verify the template kept the value).
    expected = raw_input[char_start:char_end]
    char_start, char_end = rebase_char_range(
        tokenized, char_start, char_end, expected, f"variable var_{var_idx + 1}"
    )

    return get_tokens_in_char_range(offsets, char_start, char_end)


def _make_var_factory(var_idx, var_name, prompt_mode):
    """Create a factory function for a variable token position."""

    def factory(pipeline):
        return TokenPosition(
            lambda x, p=pipeline, idx=var_idx, pm=prompt_mode: _get_var_token_indices(
                x, p, idx, pm
            ),
            pipeline,
            id=var_name,
        )

    return factory


def create_token_positions(
    pipeline: LMPipeline,
    template: str | None = None,
    prompt_mode: str | None = None,
) -> Dict[str, TokenPosition]:
    """
    Create token positions for the task.

    Returns dictionary of TokenPosition objects keyed by variable name.
    """
    if prompt_mode is None:
        from .config import PROMPT_MODE

        prompt_mode = PROMPT_MODE

    positions: Dict[str, TokenPosition] = {
        "last": TokenPosition(
            lambda x, p=pipeline: get_last_token_index(x, p), pipeline, id="last"
        ),
    }

    for idx in range(4):
        var_name = f"var_{idx + 1}"
        factory = _make_var_factory(idx, var_name, prompt_mode)
        positions[var_name] = factory(pipeline)

    return positions
