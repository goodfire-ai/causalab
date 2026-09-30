"""CPU structural guard for the paper-golden tier (runs in default CI).

Asserts the tier's bookkeeping without loading any model: every golden
specification loads and digests to its pin, the pin file and the
protocols directory cover each other exactly, and every entry in
paper_goldens.json is claimed by exactly one golden test. The pinned
*values* live in paper_goldens.json and trace to published papers
task packages — never to a run of this stack; document identity is pinned
separately in golden_digests.json (regenerate via
``uv run python tests/golden/update_golden_digests.py``).
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from causalab.protocol.pipeline import compile_protocol

from tests.golden._env import FIXTURES, GOLDEN_PROTOCOLS, GOLDENS_FILE, build_env
from tests.protocol._env import steps_of

pytestmark = pytest.mark.unit

DIGESTS_FILE = Path(__file__).parent / "golden_digests.json"


@pytest.fixture(scope="module")
def env():
    return build_env(Path(tempfile.mkdtemp()))


def _pins() -> dict[str, dict[str, object]]:
    return json.loads(DIGESTS_FILE.read_text())


def _goldens() -> dict[str, dict[str, object]]:
    return json.loads(GOLDENS_FILE.read_text())["goldens"]


def test_every_document_is_pinned_and_every_pin_has_a_document():
    on_disk = sorted(p.name for p in GOLDEN_PROTOCOLS.glob("*_im.json"))
    assert on_disk == sorted(_pins()), (
        "tests/golden/protocols/ and golden_digests.json disagree — run "
        "update_golden_digests.py and review the diff"
    )


@pytest.mark.parametrize(
    "name", sorted(p.name for p in GOLDEN_PROTOCOLS.glob("*_im.json"))
)
def test_document_digests_match_their_pins(name, env):
    loaded = compile_protocol(GOLDEN_PROTOCOLS / name, env=env)
    pin = _pins()[name]
    assert loaded.digests.document == pin["document"]
    assert list(steps_of(loaded, env).digests) == pin["points"]


def test_every_golden_value_is_claimed_by_exactly_one_test():
    """Every non-pending goldens entry is asserted by exactly one golden
    test. ``"pending": true`` marks values with an open question — a test
    still owed, or a recorded calibration finding (see the entry's notes);
    a pending entry MAY be claimed (its test skips at runtime), so
    un-pending an entry without a test fails here."""
    from tests.golden.test_paper_goldens import COVERED

    goldens = _goldens()
    live = {g for g, e in goldens.items() if not e.get("pending")}
    claimed = [g for ids in COVERED.values() for g in ids]
    assert sorted(claimed) == sorted(set(claimed)), "a golden id is claimed twice"
    assert live <= set(claimed), f"unclaimed live: {sorted(live - set(claimed))}"
    assert set(claimed) <= set(goldens), (
        f"claimed but unknown: {sorted(set(claimed) - set(goldens))}"
    )


def test_golden_values_carry_provenance_and_never_a_stack_run():
    for gid, entry in _goldens().items():
        assert entry["source"].get("where"), gid
        assert entry["source"].get("quote"), gid
        assert entry["sidedness"] in ("two_sided", "at_least"), gid
        if entry["sidedness"] == "two_sided":
            lo, hi = entry["band"]
            assert lo < entry["value"] < hi, gid
        else:
            assert "floor" in entry, gid
        assert entry.get("paper"), gid


def test_fixture_datasets_referenced_by_documents_exist():
    for path in GOLDEN_PROTOCOLS.glob("*_im.json"):
        raw = json.loads(path.read_text())
        for role in raw.get("data", {}).values():
            ref = role["dataset"]
            assert (FIXTURES / "data" / f"{ref}.json").is_file(), (
                f"{path.name} references missing fixture dataset {ref!r}"
            )
