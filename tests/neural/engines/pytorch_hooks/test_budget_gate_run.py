"""A budget-gate DBM fit end to end through the CLI (spec §2.5
``parametrization: budget``, ``k_schedule``): the fit trains a ranking with no
sparsity penalty, its own held-out pass and diagnostics read the mask at the
schedule's ``eval`` cut, the bundle stamps the map, and the replay is a
``top_k`` readout of the same theta — at the same cut, the fit's own number;
without a cut, refused by name.

Why these and not a score: the budget map's whole claim is that ``Σ m = k`` by
construction and that the ranking is the artifact. The identities below are
what make a budget fit's curve (a ``top_k`` sweep over its bundle) comparable
with a threshold method's point.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from safetensors.torch import load_file

from causalab.cli import main
from causalab.io.env import read_safetensors_metadata
from tests.neural.engines.pytorch_hooks.conftest import TINY_QWEN35_MOE

from tests.protocol._env import FIXTURES
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
FIT_SET = {
    "train.eval.split": "weekdays/test",
    "train.steps": {"epochs": 1},
    "train.batch": {"pairs": 2},
}

pytestmark = pytest.mark.smoke


def _fit_document(k_schedule: dict) -> dict:
    """The shipped DBM fit under ``budget``: no penalty, no anneal — the
    objective is the task term alone and the sparsity is the schedule's."""
    fit = json.loads((PROTOCOLS / "dbm.json").read_text())
    fit["method"]["featurizers"]["gate"] = {
        "kind": "gate",
        "parametrization": "budget",
        "k_schedule": k_schedule,
    }
    # the task term alone: the named `ce` term stays, and the saves that name it
    del fit["method"]["train"]["objective"]["l1"]
    fit["method"]["train"].pop("anneal", None)
    fit["method"]["save"].append(
        {
            "kind": "trajectory",
            "every": {"count": 2},
            "file_path": "trajectory.safetensors",
        }
    )
    return fit


def _apply_document(gate: dict) -> dict:
    apply = json.loads((PROTOCOLS / "dbm_apply.json").read_text())
    apply["method"]["featurizers"]["gate"] = {"kind": "gate", **gate}
    return apply


def _workflow(tmp_path: Path, steps: dict[str, tuple[dict, dict]], out: str) -> int:
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
    return main(
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


def _values(table: Path) -> list[float]:
    return [float(r["value"]) for r in json.loads(table.read_text())]


def test_a_fixed_budget_fit_reads_out_at_its_cut_and_replays_through_top_k(
    tmp_path: Path,
) -> None:
    k = 3
    code = _workflow(
        tmp_path,
        {
            "fit": (_fit_document({"kind": "fixed", "k": k}), FIT_SET),
            "apply": (
                _apply_document(
                    {
                        "parametrization": "budget",
                        "file_path": "fit/gate.safetensors",
                        "top_k": k,
                    }
                ),
                {},
            ),
        },
        "budget",
    )
    assert code == 0
    run = tmp_path / "run" / "budget"
    bundle = run / "fit/gate.safetensors"
    header = read_safetensors_metadata(bundle)
    assert header is not None and header["parametrization"] == "budget"
    assert "stretch" not in header  # no constants: the map has none
    theta = load_file(str(bundle))["theta"]
    diagnostics = json.loads((run / "fit/fit_diagnostics.json").read_text())
    gate = diagnostics[0]["featurizers"]["gate"]
    assert gate["parametrization"] == "budget"
    assert gate["k_schedule"] == {"kind": "fixed", "k": k}
    assert gate["eval_k"] == k and gate["k"] == k and gate["stop_grad_shift"] == 0.0
    # the hard mask is the cut, whatever the threshold would have said
    assert gate["hard_mask_size"] == k
    # the trajectory carries the step's budget beside the count
    trajectory = read_safetensors_metadata(run / "fit/trajectory.safetensors")
    assert trajectory is not None
    entries = json.loads(trajectory["entries"])
    assert len(entries) == 2
    for record in entries.values():
        assert record["parametrization"] == "budget"
        assert record["gate.k"] == k and record["gate.hard_mask_size"] == k
    # the replay at the fit's own cut scores the fit's own number
    assert _values(run / "apply/iia.json") == pytest.approx(
        _values(run / "fit/iia.json")
    )
    # …and it is the k largest theta, ties toward the lower index
    order = sorted(range(theta.numel()), key=lambda i: (-float(theta[i]), i))
    kept = set(order[:k])
    assert len(kept) == k


def test_a_sampled_budget_needs_its_eval_cut_and_a_loaded_one_needs_top_k(
    tmp_path: Path,
) -> None:
    """A ``log_uniform`` schedule draws a different budget every step and reads
    its hard mask at ``eval``; the bundle it writes is a ranking, so an apply
    that names no ``top_k`` is refused before any forward."""
    code = _workflow(
        tmp_path,
        {
            "fit": (
                _fit_document({"kind": "log_uniform", "low": 1, "high": 6, "eval": 2}),
                FIT_SET,
            ),
        },
        "sampled",
    )
    assert code == 0
    run = tmp_path / "run" / "sampled"
    gate = json.loads((run / "fit/fit_diagnostics.json").read_text())[0]["featurizers"][
        "gate"
    ]
    assert gate["eval_k"] == 2 and gate["hard_mask_size"] == 2
    assert 1 <= gate["k"] <= 6
    code = _workflow(
        tmp_path,
        {
            "apply": (
                _apply_document(
                    {
                        "parametrization": "budget",
                        "file_path": str(run / "fit/gate.safetensors"),
                    }
                ),
                {},
            )
        },
        "no_cut",
    )
    assert code == 1
    code = _workflow(
        tmp_path,
        {
            "apply": (
                _apply_document(
                    {
                        "parametrization": "budget",
                        "file_path": str(run / "fit/gate.safetensors"),
                        "top_k": {"sweep": [0, 1, 2, 4]},
                    }
                ),
                {},
            )
        },
        "curve",
    )
    assert code == 0
    rows = json.loads((tmp_path / "run/curve/apply/iia.json").read_text())
    assert {row["featurizers.gate.top_k"] for row in rows} == {0, 1, 2, 4}
