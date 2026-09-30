"""Token position functions for the ``hex_color`` task.

Provides two positions:

* ``last_token`` — the final prompt token (where the answer is read off).
* ``hex`` — the last token of the ``#RRGGBB`` stimulus span (the hex code is
  multi-token; this pins the readable "end of the stimulus" position, mirroring
  ``natural_domains_arithmetic``'s ``entity`` scope position).
"""

from __future__ import annotations

from typing import Any

from causalab.tasks.token_positions import LMPipeline
from causalab.tasks.token_positions import TokenPosition, build_token_positions

from .causal_models import TEMPLATE

TokenPositionSpec = dict[str, Any]


def create_token_positions(
    pipeline: LMPipeline, template: str | None = None
) -> dict[str, TokenPosition]:
    """Create all token positions for the ``hex_color`` task.

    Args:
        pipeline: The tokenizer pipeline.
        template: Prompt template (defaults to the task template).

    Returns:
        Dict mapping position names to ``TokenPosition`` objects.
    """
    if template is None:
        template = TEMPLATE

    specs: dict[str, TokenPositionSpec] = {
        "last_token": {"type": "index", "position": -1},
        "hex": {"type": "index", "position": -1, "scope": {"variable": "hex"}},
    }
    return build_token_positions(specs, template, pipeline)
