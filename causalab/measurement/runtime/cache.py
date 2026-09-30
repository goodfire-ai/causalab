"""Read-only cache provenance for the selected arm, including historical sources."""

from __future__ import annotations

import sys
from typing import Literal, TypedDict

RESIDENT_RESET_POLICY = (
    "fresh engine execution, RNG and fit/optimizer for each pass; frozen resident model; "
    "retained host memos (tokenization, metric encoding, table validation, gather indices, "
    "kernel selection/installation when implemented); "
    "runtime and OS/HF caches may remain warm"
)
COLD_RESET_POLICY = (
    "fresh process, model, RNG and fit/optimizer for each execution; "
    "host memos may warm during preparation and execution; OS/HF caches not flushed"
)

_HOST_MEMOS = (
    ("causalab.neural.shared.encoding", "_TOKENIZED"),
    # the answer-id memo's home; an arm before the move keeps it in metrics
    ("causalab.protocol.answers", "_ENCODED_IDS"),
    ("causalab.neural.shared.metrics", "_ENCODED_IDS"),
    ("causalab.protocol.resolve", "_checked_table_text"),
    ("causalab.neural.shared.gather", "_dense_index"),
    ("causalab.neural.shared.gather", "_flat_index"),
    ("causalab.neural.shared.kernels", "_KERNEL_MODULES"),
    ("causalab.neural.shared.kernels", "_INSTALLED"),
    ("causalab.neural.shared.kernels", "_TORCH_PATHS"),
    ("causalab.io.env", "_TABLE_DIGESTS"),
)

MemoAvailability = Literal["available", "attribute_missing", "module_not_loaded"]


class CacheProvenance(TypedDict):
    lifetime: str
    observed_at: Literal["after_execution"]
    discovery: str
    host_memos: dict[str, MemoAvailability]
    external_caches: Literal["OS/HF caches not flushed"]


def cache_provenance(mode: Literal["resident", "cold_process"]) -> CacheProvenance:
    """Observe capabilities without importing, clearing, or reading memo contents.

    Absolute module names refer to the tested arm, even when this helper is loaded
    under the controller's private package. Availability is descriptive evidence,
    not scientific alignment: an optimization may add a memo to only one arm.
    """
    memos: dict[str, MemoAvailability] = {}
    for name, attribute in _HOST_MEMOS:
        module = sys.modules.get(name)
        memos[f"{name}.{attribute}"] = (
            "module_not_loaded"
            if module is None
            else "available"
            if attribute in vars(module)
            else "attribute_missing"
        )
    return {
        "lifetime": (
            "resident_worker"
            if mode == "resident"
            else "fresh_process; may warm during preparation and execution"
        ),
        "observed_at": "after_execution",
        "discovery": "known host memo attributes in loaded selected-arm modules; not exhaustive; occupancy not inspected",
        "host_memos": memos,
        "external_caches": "OS/HF caches not flushed",
    }
