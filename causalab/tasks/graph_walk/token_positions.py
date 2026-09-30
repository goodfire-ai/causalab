"""Token position definitions for the graph_walk task."""

from __future__ import annotations

from causalab.tasks.token_positions import LMPipeline
from causalab.tasks.token_positions import (
    TokenPosition,
    build_token_positions,
)


def create_token_positions(
    pipeline: LMPipeline,
    template: str | None = None,
    config: object | None = None,
) -> dict[str, TokenPosition]:
    """Create token positions for the graph_walk task.

    The main position of interest is the last token in the sequence,
    since we measure the model's representation of the current graph node
    at the final position.

    Args:
        pipeline: LMPipeline for tokenization.
        template: Template string. Defaults to "{raw_input}".
        config: Optional GraphWalkConfig (unused, for API consistency).

    Returns:
        Dictionary mapping position names to TokenPosition objects.
    """
    if template is None:
        template = "{raw_input}"

    specs: dict = {
        "last": {"type": "index", "position": -1},
    }

    return build_token_positions(specs, template, pipeline)
