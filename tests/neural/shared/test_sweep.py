"""The sweep is the engine's:
[`causalab.neural.shared.sweep`][] enumerates a compiled document's steps in
the canonical order and signs each with the protocol's one hasher.

What is pinned here: the order and the digests are **today's** — for every
corpus document the signed steps are, in order, the point digests
``tests/protocol/corpus_digests.json`` pinned before enumeration moved (the
pins were made by the compiler's ``expand`` stage, so agreeing with them is
agreeing with that order); the module is torch-free at import (a subprocess,
because ``tests/conftest.py`` imports torch at session scope); the named axes
enumerate as the rows they declare, slowest, never as the display form's cross
product; and ``step_records`` is exactly what a [`RunResult`][causalab.protocol.engine.RunResult] carries.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from causalab.neural.shared.sweep import (
    Expansion,
    Point,
    enumerate_steps,
    expand,
    sign_steps,
    signed_steps,
    step_records,
)
from causalab.protocol.pipeline import compile_protocol
from causalab.protocol.engine import StepRecord
from causalab.protocol.identity import step_digest
from causalab.protocol.lowering import find_axes, point_count
from causalab.io.env import ResolutionEnv
from causalab.protocol.schema import parse_document

from tests.protocol._docs import base_doc, in_order
from tests.protocol._env import CORPUS_DIR
from tests.protocol.test_axes import entity_doc, rome_doc


pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[3]
PINS = json.loads((REPO / "tests/protocol/corpus_digests.json").read_text())


# --------------------------------------------------------------------------- #
# the order is today's
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", sorted(PINS))
def test_the_signed_steps_are_the_pinned_points_in_order(
    name: str, env: ResolutionEnv
) -> None:
    """Every corpus document: ``enumerate_steps`` walks the cross product in
    the order the compiler's ``expand`` stage did (last axis fastest, the
    named axes' rows slowest), and ``sign_steps`` names each step with the
    digest the corpus pinned for that index."""
    loaded = compile_protocol(CORPUS_DIR / name, env=env)
    expansion = enumerate_steps(loaded)
    assert expansion.axes == loaded.axes
    assert len(expansion.points) == point_count(loaded.axes)
    signed = sign_steps(expansion, env)
    assert [s.digest for s in signed] == PINS[name]["points"]
    assert [s.index for s in signed] == list(range(len(expansion.points)))
    # one hasher: the digest of a step is `identity.step_digest` of its tree
    for step in signed:
        assert step.digest == step_digest(step.raw, env)
        assert step.coords == expansion.points[step.index].coords


def test_step_records_are_the_run_results_steps(env: ResolutionEnv) -> None:
    loaded = compile_protocol(CORPUS_DIR / "07_weekdays_locate_scan_im.json", env=env)
    records = step_records(loaded, env, indices=(3, 4, 5, 6))
    signed = signed_steps(loaded, env, indices=(3, 4, 5, 6))
    assert records == tuple(s.record for s in signed)
    assert [r.index for r in records] == [3, 4, 5, 6]
    assert all(isinstance(r, StepRecord) for r in records)
    assert [r.digest for r in records] == PINS["07_weekdays_locate_scan_im.json"][
        "points"
    ][3:7]


def test_the_cross_product_is_last_axis_fastest() -> None:
    raw = base_doc()
    raw["method"]["positions"] = {
        "tap": {"sweep": [{"index": -1}, {"variable": "subject"}]}
    }
    raw["method"]["sites"]["tgt"]["layers"] = {"sweep": {"range": [0, 3]}}
    raw["method"]["reads"]["v_cf"]["pos"] = "tap"
    raw["method"]["writes"]["patch"]["pos"] = "tap"
    expansion = expand(in_order(raw))
    assert isinstance(expansion, Expansion)
    assert [a.id for a in expansion.axes] == ["positions.tap", "sites.tgt.layers"]
    assert [p.coords["sites.tgt.layers"] for p in expansion.points] == [0, 1, 2] * 2
    assert [p.coords["positions.tap"] for p in expansion.points[:3]] == [
        {"index": -1}
    ] * 3
    assert all(isinstance(p, Point) for p in expansion.points)
    for point in expansion.points:
        parse_document(point.raw)  # every step is a well-formed document
        assert not find_axes(point.raw)  # and concrete


# --------------------------------------------------------------------------- #
# the named axes (§3.2) enumerate as rows, slowest
# --------------------------------------------------------------------------- #


def test_named_axes_enumerate_as_rows_slowest(env: ResolutionEnv) -> None:
    compiled = compile_protocol(
        entity_doc(), env=env, base_dir=None, overrides=None, engine=None
    )
    expansion = enumerate_steps(compiled)
    assert [a.id for a in expansion.axes] == ["axes.location", "train.seed"]
    assert [dict(p.coords) for p in expansion.points] == [
        {"axes.location": row, "train.seed": seed}
        for row in (8, 9, 10)
        for seed in (0, 1)
    ]
    assert len(expansion.points) == point_count(compiled.axes) == 6


def test_a_dependent_axis_enumerates_its_computed_windows(env: ResolutionEnv) -> None:
    compiled = compile_protocol(
        rome_doc(), env=env, base_dir=None, overrides=None, engine=None
    )
    expansion = enumerate_steps(compiled)
    assert len(expansion.points) == 48 == point_count(compiled.axes)
    bands: list[Any] = [
        p.raw["method"]["sites"]["tgt"]["layers"] for p in expansion.points
    ]
    assert bands[0] == [0, 1, 2, 3, 4] and bands[47] == [42, 43, 44, 45, 46, 47]


# --------------------------------------------------------------------------- #
# torch-free at import
# --------------------------------------------------------------------------- #


_PROBE = """
import json, sys
import causalab.neural.shared.sweep
import causalab.neural.shared.step_rules
import causalab.neural.shared.receipt
import causalab.workflow.steps

print(json.dumps(sorted(m for m in ("torch", "numpy", "pandas", "safetensors")
                        if m in sys.modules)))
"""


def test_the_enumerator_imports_no_numerics() -> None:
    """The workflow layer imports this module at load, and ``causalab
    validate`` of a workflow must stay torch-free — so the module, the
    per-step rules, the engine-side receipt and the workflow's steps view
    import no numerics."""
    completed = subprocess.run(
        [sys.executable, "-c", _PROBE],
        capture_output=True,
        text=True,
        cwd=str(REPO),
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout.strip().splitlines()[-1]) == []
