"""The real ``Qwen/Qwen3.6-35B-A3B`` across two to eight GPUs (``docs/model_parallelism.md``
§8, §10.6): the parallel golden's inference document at ``ep=4``, ``tp=4``
and ``pp=4`` (world 4), ``ep=8`` (world 8) and ``tp=2,ep=2`` (a model group
of two, world 2 — both axes sharding the same two ranks), each a
``causalab run … --device cuda`` subprocess through the two-rank harness
(``tests/golden/_parallel/``: the inference document, the runs, the
per-class measure), against the world-1 run of the same document.

``pp=4`` is held **byte for byte** (§8: placement, no reduction moves). The
tensor / expert geometries land within bands pinned from the measurements
`RECORD` holds for this node by the two-rank record's rule
(``tests/golden/_parallel/bands.py``, format 2: per class the maximum, the
world-1 scale, the dtype and ``max(3 × max, 2 × ulp(dtype, scale), 1e-3)``;
the routing fraction at twice, floor one in a hundred), the entries checked
by ``_parallel.entry_band`` as the two-rank record's are. Until a geometry
is captured its band replay skips, naming the capture, and never
passes vacuously. The loader is held through every rank's load report under
the several-axes rule (``_parallel_worlds.load_problems``): at ``ep=4``,
``tp=4`` and ``ep=8`` every sharded parameter at exactly ``1 / world`` of
its bytes (``tp=4`` above the model's two KV heads replicates the K/V
projections, which are then read whole, §6.6); at ``tp=2,ep=2`` at ``1 / 2``;
under ``pp=4`` each stage whole and disjoint, the four covering the model.

📐 Capture context: `CONTEXT`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tests.golden import _parallel as par
from tests.golden import _parallel_worlds as worlds

pytestmark = [
    pytest.mark.golden,
    worlds.needs(2),
]

EXACT: tuple[str, ...] = ("pp=4",)
BANDED: tuple[str, ...] = ("ep=4", "tp=2,ep=2", "tp=4", "ep=8")
GEOMETRIES: tuple[str, ...] = EXACT + BANDED

#: The measured entries per banded geometry and output class (the two-rank
#: record's classes: ``stream``, ``sharded_read``, ``sharded_write``,
#: ``experts``, and the ``routing`` fraction), each the format-2 entry the
#: two-rank record holds — ``max_abs_diff`` / ``scale`` / ``dtype`` / ``band``,
#: the routing ``fraction`` / ``band`` — checked by ``_parallel.entry_band``.
#: Empty until captured on a node with four (eight for ``ep=8``) CUDA devices.
RECORD: dict[str, dict[str, dict[str, Any]]] = {
    "ep=4": {
        "experts": {
            "band": 0.0234375,
            "dtype": "bf16",
            "max_abs_diff": 0.0078125,
            "scale": 0.48046875,
        },
        "routing": {"band": 0.08928571428571429, "fraction": 0.044642857142857144},
        "sharded_read": {
            "band": 0.1875,
            "dtype": "bf16",
            "max_abs_diff": 0.0625,
            "scale": 9.6875,
        },
        "sharded_write": {
            "band": 3.046875,
            "dtype": "bf16",
            "max_abs_diff": 1.015625,
            "scale": 19.375,
        },
        "stream": {
            "band": 1.828125,
            "dtype": "bf16",
            "max_abs_diff": 0.609375,
            "scale": 19.375,
        },
    },
    "ep=8": {
        "experts": {
            "band": 0.0146484375,
            "dtype": "bf16",
            "max_abs_diff": 0.0048828125,
            "scale": 0.48046875,
        },
        "routing": {"band": 0.14285714285714285, "fraction": 0.07142857142857142},
        "sharded_read": {
            "band": 0.328125,
            "dtype": "bf16",
            "max_abs_diff": 0.109375,
            "scale": 9.6875,
        },
        "sharded_write": {
            "band": 2.4609375,
            "dtype": "bf16",
            "max_abs_diff": 0.8203125,
            "scale": 19.375,
        },
        "stream": {
            "band": 2.859375,
            "dtype": "bf16",
            "max_abs_diff": 0.953125,
            "scale": 19.375,
        },
    },
    "tp=2,ep=2": {
        "experts": {
            "band": 0.0087890625,
            "dtype": "bf16",
            "max_abs_diff": 0.0029296875,
            "scale": 0.48046875,
        },
        "routing": {"band": 0.07142857142857142, "fraction": 0.03571428571428571},
        "sharded_read": {
            "band": 0.1875,
            "dtype": "bf16",
            "max_abs_diff": 0.0625,
            "scale": 9.6875,
        },
        "sharded_write": {
            "band": 2.8125,
            "dtype": "bf16",
            "max_abs_diff": 0.9375,
            "scale": 19.375,
        },
        "stream": {"band": 3.0, "dtype": "bf16", "max_abs_diff": 1.0, "scale": 19.375},
    },
    "tp=4": {
        "experts": {
            "band": 0.01318359375,
            "dtype": "bf16",
            "max_abs_diff": 0.00439453125,
            "scale": 0.48046875,
        },
        "routing": {"band": 0.125, "fraction": 0.0625},
        "sharded_read": {
            "band": 0.3310546875,
            "dtype": "bf16",
            "max_abs_diff": 0.1103515625,
            "scale": 9.6875,
        },
        "sharded_write": {
            "band": 2.4375,
            "dtype": "bf16",
            "max_abs_diff": 0.8125,
            "scale": 19.375,
        },
        "stream": {
            "band": 3.0703125,
            "dtype": "bf16",
            "max_abs_diff": 1.0234375,
            "scale": 19.375,
        },
    },
}
#: Current capture environment for `RECORD`.
CONTEXT: dict[str, Any] = {
    "node": "NVIDIA H100 80GB HBM3",
    "cuda": ["NVIDIA H100 80GB HBM3"] * 8,
    "torch": "2.9.0+cu128",
    "transformers": "5.16.1",
    "captured": "2026-09-16",
}
#: Bytes requested from the checkpoint per rank. Pipeline stages may
#: request different amounts; the other geometries have equal-sized ranks.
#: These are read counts, not device-memory peaks.
LOAD: dict[str, dict[str, int]] = {
    "ep=4": {"bytes_requested": 21002839296},
    "tp=2,ep=2": {"bytes_requested": 35699942656},
    "tp=4": {"bytes_requested": 67239142656},
    "ep=8": {"bytes_requested": 12949775616},
    "pp=4": {"bytes_requested_min": 16815289984, "bytes_requested_max": 17845318656},
}

CAPTURE = (
    "HF_HUB_OFFLINE=1 uv run pytest -m golden tests/golden/test_parallel_world4.py -s "
    "and pin the printed format-2 entries in RECORD"
)


def _needs(text: str) -> Any:
    return pytest.param(
        text, marks=worlds.needs(worlds.geometry_of(text).world), id=text
    )


def _params(texts: tuple[str, ...]) -> list[Any]:
    return [_needs(t) for t in texts]


# --------------------------------------------------------------------------- #
# the runs: the oracle once, each geometry on demand, one world resident at a time
# --------------------------------------------------------------------------- #


def _measure(solo: Path, parallel: Path) -> dict[str, float]:
    """The inference document's per-class maxima as plain floats (the
    routing fraction included), the shape `RECORD` holds."""
    measured = par.inference.measure_run(solo, parallel, par.A3B)
    return {
        kind: value.max_abs_diff if isinstance(value, par.Measurement) else value
        for kind, value in measured.classes.items()
    }


@pytest.fixture(scope="module")
def root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("parallel-world4")


@pytest.fixture(scope="module")
def document(root: Path) -> Path:
    return par.inference.author(root, par.A3B)


@pytest.fixture(scope="module")
def solo(document: Path, root: Path) -> Path:
    return par.run(document, root / "solo")


@pytest.fixture(scope="module")
def outputs() -> dict[str, Path]:
    return {}


@pytest.fixture
def parallel(
    request: pytest.FixtureRequest, document: Path, root: Path, outputs: dict[str, Path]
) -> Path:
    """The run of the test's geometry (its ``text`` parameter), once per module."""
    text = request.node.callspec.params["text"]
    if text not in outputs:
        out = root / text.replace(",", "_").replace("=", "")
        outputs[text] = par.run(document, out, "--parallel", text)
    return outputs[text]


# --------------------------------------------------------------------------- #
# exact
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("text", _params(EXACT))
def test_four_stages_are_bit_identical_to_world_one(
    solo: Path, parallel: Path, text: str
) -> None:
    assert par.exact_differences(solo, parallel) == []
    assert worlds.receipt_problems(solo, parallel, text) == []
    assert _measure(solo, parallel)[par.ROUTING] == 0.0
    assert len(par.receipt(parallel)["points"]) == len(par.inference.sweep(par.A3B))


# --------------------------------------------------------------------------- #
# banded
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("text", _params(BANDED))
def test_tensor_and_expert_geometries_land_within_the_recorded_bands(
    solo: Path, parallel: Path, text: str
) -> None:
    assert worlds.receipt_problems(solo, parallel, text) == []
    measured = _measure(solo, parallel)
    block = par.document_record(
        par.DOCUMENTS["inference"],
        par.A3B,
        {text: par.inference.measure_run(solo, parallel, par.A3B)},
        {},
        with_context=False,
    )
    print(
        text,
        {
            k: {f: v for f, v in e.items() if f != "files"}
            for k, e in block["geometries"][text].items()
        },
    )
    recorded = RECORD.get(text)
    if recorded is None:
        pytest.skip(f"{text} is not yet captured: {CAPTURE}")
    assert set(measured) == set(recorded), (text, sorted(measured), sorted(recorded))
    for kind, worst in measured.items():
        band = par.entry_band(kind, recorded[kind])
        assert worst <= band, (text, kind, worst, band, recorded[kind])


def test_the_record_obeys_the_two_rank_records_rule() -> None:
    for text, classes in RECORD.items():
        assert text in BANDED, text
        assert set(classes) == set(par.inference.CLASSES.values()) | {par.ROUTING}, text
        for kind, entry in classes.items():
            assert entry["band"] == par.entry_band(kind, entry)
            assert entry["band"] >= par.entry_value(kind, entry)
    if RECORD:
        assert {"node", "torch", "transformers", "captured"} <= set(CONTEXT)


# --------------------------------------------------------------------------- #
# the loader
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("text", _params(GEOMETRIES))
def test_every_sharded_parameter_is_read_at_its_axis_fraction(
    parallel: Path, text: str
) -> None:
    world = worlds.geometry_of(text).world
    reports = par.load_reports(parallel, world)
    assert worlds.load_problems(text, reports) == []
    totals = worlds.totals(reports)
    print(text, totals)
    requested = {t["bytes_requested"] for t in totals.values()}
    if text in EXACT:
        assert min(requested) == LOAD[text]["bytes_requested_min"]
        assert max(requested) == LOAD[text]["bytes_requested_max"]
    else:
        assert requested == {LOAD[text]["bytes_requested"]}


@pytest.mark.parametrize("text", _params(("ep=4", "tp=4", "ep=8")))
def test_a_single_model_axis_reads_one_over_world(parallel: Path, text: str) -> None:
    """With one model axis the several-axes rule is the two-rank record's:
    every sharded parameter at exactly ``1 / world``."""
    world = worlds.geometry_of(text).world
    for record in par.load_reports(parallel, world):
        denominators = set(worlds.sharded_denominators(record).values())
        assert denominators == {float(world)}, (text, denominators)


@worlds.needs(4)
def test_the_four_stages_together_hold_the_whole_model(
    document: Path, root: Path, outputs: dict[str, Path]
) -> None:
    for text in ("pp=4", "ep=4"):
        if text not in outputs:
            outputs[text] = par.run(
                document, root / text.replace("=", ""), "--parallel", text
            )
    stages = par.load_reports(outputs["pp=4"], 4)
    whole = set(par.load_reports(outputs["ep=4"], 4)[0]["bytes_on_disk"])
    assert worlds.stage_names("pp=4", stages) == whole
    on_disk = sum(sum(s["bytes_on_disk"].values()) for s in stages)
    assert on_disk == sum(
        par.load_reports(outputs["ep=4"], 4)[0]["bytes_on_disk"].values()
    )
