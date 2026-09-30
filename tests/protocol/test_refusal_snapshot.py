"""Check load-time refusals against reviewed diagnostics.

Each trigger must raise the expected exception class and retain the recorded
message. Selected cases use required diagnostic details. Runtime cases use the
same checker in tests/neural/engines/nnsight_tracing/test_refusal_snapshot.py.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from tests._helpers import refusal_snapshot as table

pytestmark = pytest.mark.unit

#: Cases checked through required diagnostic details.
ALLOWED_UPGRADES: dict[str, dict[str, Any]] = {
    # The named engine must report its required and missing capabilities.
    "1": {
        "exc_class": "ValidationError",
        "pins": [
            "[V13]",
            "it requires ['component:block_output', 'component:block_output:write', "
            "'component:lm_head', 'paired_forward']",
            "lacks ['component:block_output', 'component:block_output:write', "
            "'component:lm_head']",
        ],
    },
    # The new complete neuron site also accepts expert selection.
    "18": {
        "exc_class": "ProtocolError",
        "pins": [
            "site names expert 3 on component 'router_scores'",
            "which has no per-expert axis",
            "'expert_activation', 'expert_neuron_output' and 'expert_output'",
        ],
    },
    # Probability writes must preserve normalized rows.
    "13": {
        "exc_class": "ProtocolError",
        "pins": [
            "write 'patch' applies 'add_scaled' to 'attention_probs', which only a",
            "[P4]",
            "'swap' may change: each row must sum to 1 for the following value multiply",
            "Write 'attention_scores' to use other mechanisms before softmax "
            "restores normalized probabilities",
        ],
    },
    # This model lacks the sparse-MoE component.
    "27": {
        "exc_class": "ProtocolError",
        "pins": [
            "component 'router_logits' needs a sparse-MoE block at layer 0, but "
            "this MLP (children=['act_fn', 'down_proj', 'gate_proj', 'up_proj']) "
            "is not one",
        ],
    },
    # Canonical DeltaNet sites require a linear-attention layer.
    "17": {
        "exc_class": "ProtocolError",
        "pins": [
            "component 'delta_qkv' needs a Gated DeltaNet (linear-attention) "
            "mixer, but layer 3 of 'tiny-random/qwen3.5-moe' carries "
            "'full_attention'",
            "This tower is (linear_attention, linear_attention, "
            "linear_attention, full_attention).",
        ],
    },
    # A MoE block has no dense MLP activation site.
    "29": {
        "exc_class": "ProtocolError",
        "pins": [
            "mlp_activation: this MLP (children=['experts', 'gate', "
            "'shared_expert', 'shared_expert_gate']) matches no known family",
        ],
    },
}


def _entries() -> dict[str, dict[str, Any]]:
    data = json.loads(table.SNAPSHOT.read_text())
    return {entry["id"]: entry for entry in data["entries"]}


ENTRIES = _entries()
LOAD_IDS = sorted(
    (i for i, e in ENTRIES.items() if e["layer"] == "load" and e["captured"]), key=int
)


def test_the_snapshot_covers_the_census() -> None:
    """Vacuity floor and completeness: every census row 1–34 is either captured
    or recorded as not runnable, and every trigger has an entry."""
    assert set(ENTRIES) == {str(i) for i in range(1, 35)}
    captured = {i for i, e in ENTRIES.items() if e["captured"]}
    assert captured == set(table.LOAD_TRIGGERS) | set(table.RUN_TRIGGERS)
    assert {i for i, e in ENTRIES.items() if not e["captured"]} == set(
        table.NOT_RUNNABLE
    ) | set(table.RETIRED)
    assert not set(table.NOT_RUNNABLE) & set(table.RETIRED)
    for entry_id in table.RETIRED:
        assert ENTRIES[entry_id]["reason"] == table.RETIRED[entry_id]
    # Preserve coverage of the 28 captured cases.
    assert len(captured) >= 28


def check_entry(entry: dict[str, Any], exc: BaseException) -> None:
    """Require the recorded class and diagnostic details."""
    upgrade = ALLOWED_UPGRADES.get(entry["id"])
    if upgrade is None:
        assert type(exc).__name__ == entry["exc_class"], (
            f"entry {entry['id']}: {entry['exc_class']} became {type(exc).__name__}"
        )
        assert entry["message"] in str(exc), (
            f"entry {entry['id']}: the expected message is no longer a substring "
            f"of the refusal.\n  was: {entry['message']}\n  now: {exc}"
        )
        assert getattr(exc, "code", None) == entry["code"], entry["id"]
        assert getattr(exc, "path", None) == entry["path"], entry["id"]
        return
    assert type(exc).__name__ == upgrade["exc_class"], (
        f"entry {entry['id']}: expected {upgrade['exc_class']}, got "
        f"{type(exc).__name__}"
    )
    for pin in upgrade["pins"]:
        assert pin in str(exc), f"entry {entry['id']}: {pin!r} not in {exc}"


@pytest.mark.parametrize("entry_id", LOAD_IDS)
def test_every_load_refusal_matches_the_snapshot(entry_id: str) -> None:
    entry = ENTRIES[entry_id]
    with pytest.raises(Exception) as excinfo:
        table.LOAD_TRIGGERS[entry_id]()
    check_entry(entry, excinfo.value)
