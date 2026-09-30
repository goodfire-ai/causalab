"""Token position functions for the identity_naming task.

Provides positions for: last_token, entity.
"""

from causalab.tasks.token_positions import (
    build_token_positions,
    TokenPosition,
)
from causalab.tasks.token_positions import LMPipeline

from typing import Any

TokenPositionSpec = dict[str, Any]


def create_token_positions(
    pipeline: LMPipeline, template: str | None = None
) -> dict[str, TokenPosition]:
    """Create all token positions for the identity_naming task."""
    if template is None:
        raise ValueError(
            "template is required for identity_naming — "
            "use task.create_token_positions(pipeline) instead of calling directly"
        )

    token_position_specs: dict[str, TokenPositionSpec] = {
        "last_token": {"type": "index", "position": -1},
        "entity": {"type": "index", "position": -1, "scope": {"variable": "entity"}},
    }

    return build_token_positions(token_position_specs, template, pipeline)  # type: ignore[arg-type]
