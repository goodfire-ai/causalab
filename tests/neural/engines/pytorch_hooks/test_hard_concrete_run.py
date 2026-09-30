"""A hard-concrete DBM fit end to end through the CLI (spec §2.5
``parametrization: hard_concrete``, §2.11 ``l0``): the fit saves its bundle
with the map and the stretch stamped in the ArtifactIdentity, and an apply
document declaring the same map replays it to the fit's own number.

The failure this pins: the gate stamped ``stretch`` into its identity fields
before ``stretch`` was an ArtifactIdentity key, so the first real fit
trained to the end and raised at its first save —
``unknown ArtifactIdentity fields ['stretch']`` — a path no unit test reached because ``run_training`` alone saves nothing.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from safetensors.torch import load_file

from causalab.cli import main
from causalab.protocol.rules.errors import ValidationError
from causalab.protocol.pipeline import compile_protocol
from causalab.io.env import read_safetensors_metadata
from causalab.protocol.schema import hard_concrete_threshold
from tests.neural.engines.pytorch_hooks.conftest import TINY_QWEN35_MOE
from tests.protocol._docs import in_order
from tests.protocol._env import FIXTURES, build_env
from tests._helpers.paths import PROTOCOLS_DIR

REPO = Path(__file__).resolve().parents[4]
PROTOCOLS = PROTOCOLS_DIR
PINS = {
    "model.key": TINY_QWEN35_MOE,
    "model.dtype": "fp32",
    "sites.target.layers": 0,
    "data.base.dataset": "weekdays/train",
    "data.counterfactual.dataset": "weekdays/train",
}


def _fit_document(*, stretch: list[float] | None, fill: float) -> dict:
    """The shipped DBM fit under ``hard_concrete``, no anneal (so the
    unauthored β = 2/3 is the one the forward uses), at a ``fill`` and — when
    given — an authored ``stretch``; ``None`` leaves both constants to their
    defaults, the path the digest argument rests on."""
    fit = json.loads((PROTOCOLS / "dbm.json").read_text())
    fit["method"]["featurizers"]["gate"] = {
        "kind": "gate",
        "parametrization": "hard_concrete",
        "init": {"fill": fill},
        **({"stretch": stretch} if stretch is not None else {}),
    }
    # `l0` is the hard-concrete penalty (§2.11): it takes the `l1` term's place
    objective = fit["method"]["train"]["objective"]
    del objective["l1"]
    objective["l0"] = {"weight": 0.01, "l0": "gate"}
    fit["method"]["train"].pop("anneal", None)
    return fit


def _apply_document(*, stretch: list[float] | None) -> dict:
    apply = json.loads((PROTOCOLS / "dbm_apply.json").read_text())
    apply["method"]["featurizers"]["gate"] = {
        "kind": "gate",
        "parametrization": "hard_concrete",
        **({"stretch": stretch} if stretch is not None else {}),
        "file_path": "fit/gate.safetensors",
    }
    return apply


def _documents(
    base: Path, *, stretch: list[float] | None = None, fill: float = 0.5
) -> tuple[Path, Path]:
    fit_path, apply_path = base / "hc_fit.json", base / "hc_apply.json"
    fit_path.write_text(json.dumps(_fit_document(stretch=stretch, fill=fill), indent=2))
    apply_path.write_text(json.dumps(_apply_document(stretch=stretch), indent=2))
    return fit_path, apply_path


def _pinned(doc: dict) -> dict:
    """``PINS`` written into the document itself, for a load outside a
    workflow: a ``set`` path is section-rooted (§7) — ``model.*`` and ``data.*``
    are top-level sections, ``sites.*`` lives under ``method``."""
    for dotted, value in PINS.items():
        *parents, leaf = dotted.split(".")
        node = doc if parents[0] in doc else doc["method"]
        for key in parents:
            node = node[key]
        node[leaf] = value
    return doc


def _mean(table: Path) -> float:
    rows = json.loads(table.read_text())
    return sum(float(r["value"]) for r in rows) / len(rows)


def _run(tmp_path: Path, fit_doc: Path, apply_doc: Path, out: str) -> Path:
    workflow = {
        "version": "1",
        "output_dir": out,
        "steps": {
            "fit": {
                "type": "intervention_protocol",
                "document": str(fit_doc),
                "set": {
                    **PINS,
                    "train.eval.split": "weekdays/test",
                    "train.steps": {"epochs": 1},
                    "train.batch": {"pairs": 2},
                },
            },
            "apply": {
                "type": "intervention_protocol",
                "document": str(apply_doc),
                "set": PINS,
            },
        },
    }
    artifacts = tmp_path / "artifacts"
    shutil.copytree(FIXTURES / "artifacts", artifacts, dirs_exist_ok=True)
    path = tmp_path / "wf.json"
    path.write_text(json.dumps(workflow, indent=2))
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
            str(tmp_path / "run"),
        ]
    )
    assert code == 0
    return tmp_path / "run" / out


def _check_replay(run: Path, *, stretch: list[float]) -> None:
    """The bundle stamps the map and the stretch, the diagnostics count the hard
    mask at the threshold the stamped stretch implies — read back from the
    header, not re-derived as 0 — and the apply replays the fit's own number."""
    bundle = run / "fit/gate.safetensors"
    header = read_safetensors_metadata(bundle)
    assert header is not None
    assert header["parametrization"] == "hard_concrete"
    assert json.loads(header["stretch"]) == stretch
    theta = load_file(str(bundle))["theta"]
    diagnostics = json.loads((run / "fit/fit_diagnostics.json").read_text())
    gate = diagnostics[0]["featurizers"]["gate"]
    assert gate["parametrization"] == "hard_concrete"
    assert gate["stretch"] == stretch
    assert gate["temperature"] == pytest.approx(2 / 3)  # the unauthored β, used
    threshold = hard_concrete_threshold(tuple(json.loads(header["stretch"])))
    assert gate["hard_mask_size"] == float((theta > threshold).sum())
    # the replay scores the fit's own hard mask on the same rows: the same number
    assert _mean(run / "apply/iia.json") == pytest.approx(_mean(run / "fit/iia.json"))


@pytest.mark.smoke
def test_a_hard_concrete_fit_saves_its_stretch_and_replays_to_its_own_number(
    tmp_path: Path,
) -> None:
    """Nothing authored but the map: the defaults are used, stamped anyway
    (``stretch = [-0.1, 1.1]``), and read back through the identity check by
    an apply document that authors none. ``fill: 0.5`` starts every θ at
    exactly 0 — the value a derived threshold of ≈ −4e−16 would have counted as
    kept — so the hard-mask count is pinned where the rounding bites."""
    fit_doc, apply_doc = _documents(tmp_path, stretch=None, fill=0.5)
    run = _run(tmp_path, fit_doc, apply_doc, "hc")
    _check_replay(run, stretch=[-0.1, 1.1])


@pytest.mark.smoke
def test_a_non_default_stretch_is_part_of_the_identity(tmp_path: Path) -> None:
    """At ``[-0.1, 1.5]`` the hard split is ``θ > logit(0.6/1.6) ≈ −0.51``, not
    ``θ > 0``: the fit counts at that threshold, an apply re-authoring the
    stretch replays it, and one authoring none is refused at load (rule 15) —
    it would split the same θ at 0 and score a different mask."""
    stretch = [-0.1, 1.5]
    fit_doc, apply_doc = _documents(tmp_path, stretch=stretch, fill=0.5)
    run = _run(tmp_path, fit_doc, apply_doc, "hc_wide")
    _check_replay(run, stretch=stretch)
    env = build_env(run)  # `fit/gate.safetensors` resolves under the run tree
    with pytest.raises(ValidationError) as err:
        compile_protocol(in_order(_pinned(_apply_document(stretch=None))), env=env)
    assert err.value.rule == 15 and "fitted at stretch" in str(err.value)
    with pytest.raises(ValidationError) as err:
        compile_protocol(
            in_order(_pinned(_apply_document(stretch=[-0.2, 1.2]))), env=env
        )
    assert err.value.rule == 15 and "'stretch'" in str(err.value)
    compile_protocol(in_order(_pinned(_apply_document(stretch=stretch))), env=env)
