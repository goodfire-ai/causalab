"""A site-grouped gate end to end through the CLI (spec §2.5 ``group: site``):
one parameter over the whole ``mlp_output`` site, fitted, saved with its
``(1, width)`` map stamped, and replayed at the two poles of θ.

Why the poles and not a score: a site gate is the ``head`` map with a single
group, so the property that makes it *the* node-level unit of a circuit
benchmark is that its one parameter is exactly the on/off switch of the whole
site. At θ → +∞ the gated swap must be the plain swap of that site, at θ → −∞
the unpatched model — anything in between would mean the broadcast over the
site's coordinates is partial, or the eval split is not the hard one.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from causalab.cli import main
from tests.neural.engines.pytorch_hooks.conftest import TINY_QWEN35_MOE
from tests.protocol._env import FIXTURES
from tests._helpers.paths import PROTOCOLS_DIR

REPO = Path(__file__).resolve().parents[4]
PROTOCOLS = PROTOCOLS_DIR
PINS = {
    "model.key": TINY_QWEN35_MOE,
    "model.dtype": "fp32",
    "sites.target.component": "mlp_output",
    "sites.target.layers": 0,
    "data.base.dataset": "weekdays/train",
    "data.counterfactual.dataset": "weekdays/train",
}
FIT_SET = {
    "train.eval.split": "weekdays/test",
    "train.steps": {"epochs": 1},
    "train.batch": {"pairs": 2},
}

pytestmark = pytest.mark.smoke


def _fit_document() -> dict:
    """The shipped DBM fit with its gate grouped over the site — one θ."""
    fit = json.loads((PROTOCOLS / "dbm.json").read_text())
    fit["method"]["featurizers"]["gate"] = {"kind": "gate", "group": "site"}
    fit["method"]["train"].pop("anneal", None)
    return fit


def _apply_document(gate: dict | None, *, base_metric: bool = False) -> dict:
    """The shipped apply with the gate spelled as given, or — ``None`` — the
    plain swap of the site with no featurizer at all; optionally the same
    margin on the unpatched model (``iia_base``)."""
    apply = json.loads((PROTOCOLS / "dbm_apply.json").read_text())
    method = apply["method"]
    if gate is None:
        del method["featurizers"]
        del method["reads"]["v_cf"]["featurizer"]
        del method["writes"]["mask"]["featurizer"]
    else:
        method["featurizers"]["gate"] = {"kind": "gate", "group": "site", **gate}
    if base_metric:
        # the same margin on the un-intervened model, read on base (§2.9)
        method["reads"]["logits_base"] = {"site": "lm_head", "pos": -1}
        method["intervened_models"]["original_base"] = {
            "input": "base",
            "reads": ["logits_base"],
        }
        (iia,) = [e for e in method["save"] if e["file_path"] == "iia.json"]
        method["save"].append(
            {
                "read": "logits_base",
                "model": "original_base",
                "aggregation": iia["aggregation"],
                "file_path": "iia_base.json",
            }
        )
    return apply


def _run(tmp_path: Path, steps: dict[str, tuple[dict, dict]], out: str) -> Path:
    docs_dir = tmp_path / f"{out}_docs"
    docs_dir.mkdir(exist_ok=True)
    workflow_steps = {}
    for name, (document, extra) in steps.items():
        path = docs_dir / f"{name}.json"
        path.write_text(json.dumps(document, indent=2))
        workflow_steps[name] = {
            "type": "intervention_protocol",
            "document": str(path),
            "set": {**PINS, **extra},
        }
    workflow = {"version": "1", "output_dir": out, "steps": workflow_steps}
    artifacts = tmp_path / "artifacts"
    shutil.copytree(FIXTURES / "artifacts", artifacts, dirs_exist_ok=True)
    path = tmp_path / f"{out}.json"
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


def _values(table: Path) -> list[float]:
    return [float(r["value"]) for r in json.loads(table.read_text())]


def _with_theta(bundle: Path, target: Path, theta: float) -> Path:
    """The fitted bundle, header and all, with its one parameter set to
    ``theta`` — the stamped identity is the fit's, only the value moves."""
    with safe_open(str(bundle), framework="pt") as fh:
        metadata = dict(fh.metadata())
        tensors = {key: fh.get_tensor(key) for key in fh.keys()}
    assert tensors["theta"].numel() == 1
    tensors["theta"] = torch.full_like(tensors["theta"], theta)
    target.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(target), metadata=metadata)
    return target


def test_a_site_gate_at_the_poles_is_the_plain_swap_and_the_base(
    tmp_path: Path,
) -> None:
    run = _run(
        tmp_path,
        {
            "fit": (_fit_document(), FIT_SET),
            "plain": (_apply_document(None, base_metric=True), {}),
        },
        "fit",
    )
    bundle = run / "fit/gate.safetensors"
    with safe_open(str(bundle), framework="pt") as fh:
        header = dict(fh.metadata())
        assert fh.get_tensor("theta").numel() == 1
    assert header["group"] == "site"
    group_map = json.loads(header["group_map"])
    assert group_map[0] == 1 and group_map[1] > 1  # one group over the site
    diagnostics = json.loads((run / "fit/fit_diagnostics.json").read_text())
    gate = diagnostics[0]["featurizers"]["gate"]
    assert gate["groups"] == 1.0 and gate["width"] == float(group_map[1])
    assert gate["hard_mask_size"] in (0.0, 1.0)

    on = _with_theta(bundle, tmp_path / "on/gate.safetensors", 40.0)
    off = _with_theta(bundle, tmp_path / "off/gate.safetensors", -40.0)
    poles = _run(
        tmp_path,
        {
            "on": (_apply_document({"file_path": str(on)}), {}),
            "off": (_apply_document({"file_path": str(off)}), {}),
        },
        "poles",
    )
    swap = _values(run / "plain/iia.json")
    base = _values(run / "plain/iia_base.json")
    assert swap != base, "the identity needs a site whose swap moves the margin"
    # θ → +∞: every coordinate of the site takes the counterfactual — the swap
    assert _values(poles / "on/iia.json") == pytest.approx(swap, rel=1e-5, abs=1e-6)
    # θ → −∞: no coordinate does — the model as it was
    assert _values(poles / "off/iia.json") == pytest.approx(base, rel=1e-5, abs=1e-6)
