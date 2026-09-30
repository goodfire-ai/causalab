"""Token position functions for the ``subject_object_relations`` task.

Provides positions for: ``last_token`` and ``subject`` (the last token of the
``{subject}`` span). Built declaratively via ``build_token_positions`` — mirrors
``identity_naming``. The task's template carries a named ``{subject}`` placeholder
(the source's positional ``{}`` slot was rewritten at build time), which the
declarative spec's ``scope`` resolver requires.
"""

from __future__ import annotations

from typing import Any

from causalab.tasks.token_positions import LMPipeline
from causalab.tasks.token_positions import TokenPosition, build_token_positions

TokenPositionSpec = dict[str, Any]


def create_token_positions(
    pipeline: LMPipeline, template: str | None = None
) -> dict[str, TokenPosition]:
    """Create all token positions for the subject_object_relations task."""
    if template is None:
        raise ValueError(
            "template is required for subject_object_relations — "
            "use task.create_token_positions(pipeline) instead of calling directly"
        )

    token_position_specs: dict[str, TokenPositionSpec] = {
        "last_token": {"type": "index", "position": -1},
        "subject": {"type": "index", "position": -1, "scope": {"variable": "subject"}},
    }

    return build_token_positions(token_position_specs, template, pipeline)  # type: ignore[arg-type]
