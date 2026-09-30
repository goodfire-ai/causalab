"""Parse segments and validate their use in an intervention.

Segments identify parts of a plain or chat input. Rule 27 checks declarations
that depend on their relationship to pair edits."""

from __future__ import annotations

from causalab.protocol.rules.document import check_segments
from causalab.protocol.schema.positions import (
    CHAT_SEGMENTS,
    CONTINUATION_SEGMENT,
    parse_segments,
    SEGMENT_FRAMES,
    SEGMENT_SOURCE_KEYS,
    SegmentSource,
    SegmentsSpec,
)

__all__ = [
    "CHAT_SEGMENTS",
    "CONTINUATION_SEGMENT",
    "SEGMENT_FRAMES",
    "SEGMENT_SOURCE_KEYS",
    "SegmentSource",
    "SegmentsSpec",
    "check",
    "check_segments",
    "parse_segments",
]

#: Rule 27 under the name this module always exported it by.
check = check_segments
