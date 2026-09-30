"""Token positions for the IOI task.

Built with the declarative spec system used by ``MCQA``: each position is a
template-variable name or an indexed offset. Returns positions for
``name_A``, ``name_B``, ``name_C``, and the final token (where the model
emits its prediction).
"""

from __future__ import annotations

from typing import Any, Callable

from causalab.tasks.token_positions import LMPipeline
from causalab.tasks.token_positions import (
    TokenPosition,
    build_token_positions,
)

from .causal_models import CANONICAL_TEMPLATE

TokenPositionSpec = dict[str, Any] | Callable[[dict[str, Any]], dict[str, Any]]


def create_token_positions(
    pipeline: LMPipeline, template: str | None = None
) -> dict[str, TokenPosition]:
    """Build the IOI token-position factories.

    Args:
        pipeline: The tokenizer pipeline.
        template: The prompt template. Defaults to the canonical IOI template
            so this can be called without a Task wrapper (e.g. from notebooks).

    Returns:
        Dict of position name → [`TokenPosition`][].
    """
    if template is None:
        template = CANONICAL_TEMPLATE

    specs: dict[str, TokenPositionSpec] = {
        "name_A": {"type": "variable", "name": "name_A"},
        "name_B": {"type": "variable", "name": "name_B"},
        "name_C": {"type": "variable", "name": "name_C"},
        "last_token": {"type": "index", "position": -1},
    }

    return build_token_positions(specs, template, pipeline)
