"""Layer-band patching as one document (spec §2.9 ``intervened_models``).

A tempting claim says the sweep language cannot make one start-layer value
expand into five or ten dependent writes, so each band needs its own
explicit specification. The claim is half right. The premise holds: a sweep
expands one axis into **independent** points, and a band is one forward with several dependent writes. The conclusion
does not: ``intervened_models`` names the *set* of writes in force for one
forward, so N bands over one table of per-layer writes are N entries in one
document — one model load instead of N, at ~1–2 min each.

Two things are asserted here, because the shipped preset is a claim about cost
and a claim about semantics:

* **cost** — the bands share one counterfactual harvest forward, so a document
  with three bands plans four forward groups, not six;
* **semantics** — a band scores exactly the writes it names. A single-layer
  band inside a multi-band document reproduces, to the bit, a document that
  carries only that write.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from causalab.cli import main
from causalab.protocol.pipeline import compile_protocol
from causalab.neural.shared.plan import plan_point
from causalab.io.env import FileArtifacts, FileDatasets, ResolutionEnv
from causalab.tasks import TASKS_ROOT

from tests.neural.engines.pytorch_hooks.conftest import TINY_LLAMA
from tests.protocol._docs import UNWRITTEN, aggregation, saved
from tests.protocol._env import FIXTURES, steps_of
from tests.tables import frame as table_frame
from tests._helpers.paths import PROTOCOLS_DIR


pytestmark = pytest.mark.smoke

REPO = Path(__file__).resolve().parents[4]
PRESET = PROTOCOLS_DIR / "attention_band_patch.json"


# --------------------------------------------------------------------------- #
# the shipped preset: shape and cost
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def env() -> ResolutionEnv:
    return ResolutionEnv(
        datasets=FileDatasets(root=FIXTURES / "data", fallback_roots=(TASKS_ROOT,)),
        artifacts=FileArtifacts(root=FIXTURES / "artifacts"),
    )


def test_the_preset_plans_one_shared_harvest_for_every_band(env) -> None:
    """Three bands, four forwards: one un-intervened harvest of all ten layers
    from the counterfactual, then one patched forward per band. Three separate
    documents would be three harvests and three model loads."""
    loaded = compile_protocol(PRESET, env=env)
    assert len(steps_of(loaded, env).points) == 1
    assert plan_point(steps_of(loaded, env).documents[0]).num_forwards == 4


def test_the_wide_band_is_the_two_narrow_ones_and_each_write_is_authored_once(
    env,
) -> None:
    """What makes this one document rather than three: overlapping bands reuse
    writes instead of restating them."""
    doc = steps_of(compile_protocol(PRESET, env=env), env).documents[0]
    models = doc.intervened_models
    narrow = set(models["band4_L9"].writes) | set(models["band4_L13"].writes)
    assert set(models["band8_L9"].writes) == narrow
    assert len(doc.writes) == 8  # one per layer, not one per (layer, band)
    assert all(len(set(m.writes)) == len(m.writes) for m in models.values())


# --------------------------------------------------------------------------- #
# the semantics, run on tiny-random
# --------------------------------------------------------------------------- #


def _band_doc(bands: dict[str, list[int]]) -> dict:
    """A band document over tiny-random's two attention layers."""
    layers = sorted({layer for span in bands.values() for layer in span})
    return {
        "header": {
            "protocol_version": "4",
            "description": "attention-output bands as intervened_models",
        },
        "model": {"key": TINY_LLAMA, "revision": "main", "dtype": "fp32"},
        "data": {
            "base": {"dataset": "weekdays/data#train", "field": "input"},
            "counterfactual": {
                "dataset": "weekdays/data#train",
                "field": "counterfactual_inputs[0]",
            },
        },
        "method": {
            "intervened_models": {
                UNWRITTEN: {
                    "input": "counterfactual",
                    "reads": [f"v_a{i}" for i in layers],
                },
                **{
                    name: {
                        "input": "base",
                        "reads": [f"logits_{name}"],
                        "writes": [f"w{i}" for i in span],
                    }
                    for name, span in bands.items()
                },
            },
            "positions": {"tap": {"index": -1}},
            "sites": {
                **{
                    f"a{i}": {"component": "attention_output", "layers": [i]}
                    for i in layers
                },
                "lm_head": {"component": "lm_head"},
            },
            "reads": {
                **{f"v_a{i}": {"site": f"a{i}", "pos": "tap"} for i in layers},
                **{f"logits_{name}": {"site": "lm_head", "pos": -1} for name in bands},
            },
            "writes": {
                f"w{i}": {"site": f"a{i}", "pos": "tap", "do": {"swap": f"v_a{i}"}}
                for i in layers
            },
            "save": [
                saved(
                    f"logits_{name}",
                    name,
                    f"iia_{name}.json",
                    aggregation(
                        "logit_diff",
                        a="cf_answer",
                        b="base_answer",
                    ),
                )
                for name in bands
            ],
        },
    }


def _run(base: Path, document: dict) -> Path:
    artifacts = base / "artifacts"
    shutil.copytree(FIXTURES / "artifacts", artifacts, dirs_exist_ok=True)
    path = base / "doc.json"
    path.write_text(json.dumps(document, indent=2))
    out = base / "run"
    code = main(
        [
            "run",
            "--engine",
            "auto",
            str(path),
            "--data-root",
            str(FIXTURES / "data"),
            "--artifacts-root",
            str(artifacts),
            "--out",
            str(out),
        ]
    )
    assert code == 0
    return out


def test_a_band_scores_exactly_the_writes_it_names(tmp_path: Path) -> None:
    """The identity that makes several bands in one document trustworthy: the
    one-layer band inside a two-band document is bit-identical to a document
    carrying only that write. Bands do not leak into each other, and the shared
    harvest does not change what any of them sees."""
    together = _run(tmp_path / "together", _band_doc({"narrow": [0], "wide": [0, 1]}))
    alone = _run(tmp_path / "alone", _band_doc({"narrow": [0]}))
    assert list(table_frame(together / "iia_narrow.json")["value"]) == pytest.approx(
        list(table_frame(alone / "iia_narrow.json")["value"]), abs=0.0
    )


def test_a_wider_band_is_a_different_intervention(tmp_path: Path) -> None:
    """The premise of scanning bands at all: patching two layers is not
    patching one. Asserted as inequality, not magnitude — tiny-random's
    weights are noise, and what is under test is that the second write lands."""
    out = _run(tmp_path, _band_doc({"narrow": [0], "wide": [0, 1]}))
    narrow = list(table_frame(out / "iia_narrow.json")["value"])
    wide = list(table_frame(out / "iia_wide.json")["value"])
    assert len(narrow) == len(wide) == 2  # the weekdays/data#train fixture rows
    assert narrow != wide
