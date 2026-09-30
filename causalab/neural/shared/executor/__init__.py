"""Shared point-executor interfaces.

``base`` supplies ExecutorBase and the read path. ``cache`` owns forward
reuse, ``writes`` applies write math, and ``ragged`` handles row windows
and variable position widths. This package re-exports their public names.
"""

from __future__ import annotations

from causalab.neural.shared.executor.base import (
    BoundRead,
    ExecutorBase,
    device_scored_reads,
    document_seed,
    refuse_unstackable,
)
from causalab.neural.shared.executor.cache import (
    CaptureKey,
    ForwardCache,
    Interning,
    PrefixKey,
    PrefixPlan,
    Reuse,
    ReuseKind,
    TapKey,
    tap_key,
)
from causalab.neural.shared.executor.ragged import RowWindow
from causalab.neural.shared.values import RaggedValue

__all__ = [
    "BoundRead",
    "CaptureKey",
    "ExecutorBase",
    "ForwardCache",
    "Interning",
    "PrefixKey",
    "PrefixPlan",
    "RaggedValue",
    "Reuse",
    "ReuseKind",
    "RowWindow",
    "TapKey",
    "device_scored_reads",
    "document_seed",
    "refuse_unstackable",
    "tap_key",
]
