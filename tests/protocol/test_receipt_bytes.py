"""The run receipt's bytes did not move when the sweep moved engine-side.

``run_protocol`` used to write ``protocol.json`` off the compiled object's
points before it entered the engine; now the engine writes it
([`causalab.neural.shared.receipt.write_run_record`][]) off the steps it
enumerated and signed, before its first forward. The fixtures under
``fixtures/receipts/`` are the receipts the code before that move wrote for
four corpus documents through a stub engine — a sharded sweep, a
hand-written path patch, a fit sweep and an ``at_once`` band — and a run
today must reproduce every byte: the
same ``points[]`` (index, digest, coords), the same ``canonical``, the same
``execution`` block.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from causalab.protocol import RUN_RECORD_NAME, run_protocol
from causalab.protocol.compiled import CompiledProtocol
from causalab.protocol.engine import Engine, RunContext, RunResult
from causalab.protocol.pipeline import compile_protocol
from causalab.io.env import ResolutionEnv
from causalab.protocol.schema import COMPONENTS

from tests._helpers.stub_engine import stub_execute
from tests.protocol._env import CORPUS_DIR, FIXTURES, steps_of

pytestmark = pytest.mark.unit

RECEIPTS = FIXTURES / "receipts"

#: (document, ``--points`` shard) → the fixture the base wrote for it.
CASES: tuple[tuple[str, str | None], ...] = (
    ("07_weekdays_locate_scan_im.json", "3:7"),
    ("03_path_patching_im.json", None),
    ("08_weekdays_das_sweep_im.json", None),
    ("16_at_once_band_im.json", None),
)


def _fixture(name: str, points: str | None) -> Path:
    stem = name.removesuffix(".json")
    if points is not None:
        stem += "." + points.replace(":", "-")
    return RECEIPTS / f"{stem}.json"


class _Stub(Engine):
    """An engine that executes nothing and owes the doors the engine's half
    of the contract: the signed steps, and the receipt when asked."""

    name = "stub"
    capabilities = frozenset(
        {"grad", "paired_forward", "full_logits", "pytorch_fn_local", "generate"}
    )
    components = frozenset(COMPONENTS)
    writable_components = frozenset(COMPONENTS)
    is_local = True

    def execute(self, compiled: CompiledProtocol, run: RunContext) -> RunResult:
        return stub_execute(self, compiled, run)


@pytest.mark.parametrize("name,points", CASES, ids=[c[0] for c in CASES])
def test_the_receipt_is_byte_identical_to_the_base(
    name: str, points: str | None, env: ResolutionEnv, tmp_path: Path
) -> None:
    loaded = compile_protocol(CORPUS_DIR / name, env=env)
    result = run_protocol(loaded, env, _Stub(), tmp_path, points=points, record=True)
    written = (tmp_path / RUN_RECORD_NAME).read_bytes()
    pinned = _fixture(name, points).read_bytes()
    assert written == pinned, (
        f"{name}: the receipt's bytes moved — regenerate fixtures/receipts/ "
        "deliberately and say why in the PR body"
    )
    # the receipt's points are the steps the engine signed, in run order
    record = json.loads(written)
    assert [(p["index"], p["digest"]) for p in record["points"]] == [
        (s.index, s.digest) for s in result.steps
    ]


def test_the_fixtures_cover_exactly_the_cases() -> None:
    assert sorted(p.name for p in RECEIPTS.glob("*.json")) == sorted(
        _fixture(name, points).name for name, points in CASES
    )


def test_a_shard_selects_its_points_and_keeps_the_campaign_digest(
    env: ResolutionEnv, tmp_path: Path
) -> None:
    """The one sharded case, read back: indices 3..6 of the 64-point scan, the
    digests the engine's sweep gives those indices, the campaign digest of
    the whole document."""
    name, points = CASES[0]
    loaded = compile_protocol(CORPUS_DIR / name, env=env)
    record = json.loads(_fixture(name, points).read_text())
    steps = steps_of(loaded, env)
    assert [p["index"] for p in record["points"]] == [3, 4, 5, 6]
    assert [p["digest"] for p in record["points"]] == list(steps.digests[3:7])
    assert [p["coords"] for p in record["points"]] == [
        dict(c) for c in steps.coords[3:7]
    ]
    assert record["document_digest"] == loaded.digests.document
    assert "derived" not in record  # the receipt has no derived record
