"""Check runtime refusals with the shared snapshot triggers and assertions.

Tiny model fixtures share the model loader cache.
"""

from __future__ import annotations

import pytest

from tests._helpers import refusal_snapshot as table
from tests.protocol.test_refusal_snapshot import ENTRIES, check_entry

pytestmark = pytest.mark.smoke

RUN_IDS = sorted(
    (i for i, e in ENTRIES.items() if e["layer"] == "run" and e["captured"]), key=int
)


@pytest.fixture(scope="module")
def fixtures() -> table.Fixtures:
    return table.Fixtures()


@pytest.mark.parametrize("entry_id", RUN_IDS)
def test_every_run_refusal_matches_the_snapshot(
    entry_id: str, fixtures: table.Fixtures
) -> None:
    entry = ENTRIES[entry_id]
    with pytest.raises(Exception) as excinfo:
        table.RUN_TRIGGERS[entry_id](fixtures)
    check_entry(entry, excinfo.value)
