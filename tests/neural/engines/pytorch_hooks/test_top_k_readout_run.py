"""A loaded gate read out at a count, end to end through the CLI (spec §2.5
``top_k``, §2.12 ``rank``): the same bundle replayed through the map's
threshold and through a cut at the fit's own ``hard_mask_size`` scores the
same number, a cut at ``0`` is the unpatched model, a sweep over the cut is
monotone in what it keeps, and the ``rank`` table says which unit sat where.

Why the identities and not a number: a top-k readout is only useful if it is
*the same object* as the threshold readout wherever the two coincide — a
ranking method's curve is compared against DBM's threshold point, and any
drift between the two readouts of one ``theta`` would make that comparison
about the readout rather than the mask.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from safetensors.torch import load_file

from causalab.cli import main
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

pytestmark = pytest.mark.smoke


def _fit_document() -> dict:
    """The shipped DBM fit as it is — a sigmoid gate, so ``θ > 0`` is the
    threshold the cut is compared against."""
    fit = json.loads((PROTOCOLS / "dbm.json").read_text())
    fit["method"]["train"].pop("anneal", None)
    return fit


def _apply_document(
    gate: dict, *, rank: bool = False, base_metric: bool = False
) -> dict:
    """The shipped apply with the gate spelled as given; optionally a ``rank``
    table and, beside ``iia``, the same margin on the *unpatched* model
    (``iia_base``) — what a cut of zero units must reproduce."""
    apply = json.loads((PROTOCOLS / "dbm_apply.json").read_text())
    method = apply["method"]
    method["featurizers"]["gate"] = {"kind": "gate", **gate}
    if rank:
        method["save"].append({"kind": "rank", "file_path": "rank.json"})
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
    """One workflow of ``steps`` (name → (document, extra set)), all pinned to
    the tiny model and the fixture split, run through the CLI."""
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


def _by_point(table: Path, column: str = "value") -> dict[int, list]:
    """A metric table's values grouped by the point's one coordinate, the
    swept ``top_k`` (a metric row names its point by its coordinate columns)."""
    out: dict[int, list] = {}
    for row in json.loads(table.read_text()):
        out.setdefault(row["featurizers.gate.top_k"], []).append(row[column])
    return out


FIT_SET = {
    "train.eval.split": "weekdays/test",
    "train.steps": {"epochs": 1},
    "train.batch": {"pairs": 2},
}


def test_a_cut_at_the_threshold_count_replays_the_fit_and_a_cut_at_zero_is_the_base(
    tmp_path: Path,
) -> None:
    run = _run(
        tmp_path,
        {
            "fit": (_fit_document(), FIT_SET),
            "apply": (_apply_document({"file_path": "fit/gate.safetensors"}), {}),
        },
        "plain",
    )
    theta = load_file(str(run / "fit/gate.safetensors"))["theta"]
    diagnostics = json.loads((run / "fit/fit_diagnostics.json").read_text())
    kept = int(diagnostics[0]["featurizers"]["gate"]["hard_mask_size"])
    assert kept == int((theta > 0).sum())
    assert 0 < kept < theta.numel(), "the identity needs a mask that is neither pole"
    plain = _values(run / "apply/iia.json")

    cut = _run(
        tmp_path,
        {
            "same": (
                _apply_document(
                    {"file_path": str(run / "fit/gate.safetensors"), "top_k": kept},
                    rank=True,
                ),
                {},
            ),
            "none": (
                _apply_document(
                    {"file_path": str(run / "fit/gate.safetensors"), "top_k": 0},
                    rank=True,
                    base_metric=True,
                ),
                {},
            ),
        },
        "cut",
    )
    # the same theta, cut at its own count: bit for bit the threshold replay
    assert _values(cut / "same/iia.json") == plain
    # nothing kept: the write routes nothing, so the patched model is the model
    assert _values(cut / "none/iia.json") == _values(cut / "none/iia_base.json")

    # the rank table: one row per unit, ranks a permutation, `hard` the cut
    rows = json.loads((cut / "same/rank.json").read_text())
    assert len(rows) == theta.numel()
    assert sorted(r["rank"] for r in rows) == list(range(theta.numel()))
    assert all(r["hard"] == (r["rank"] < kept) for r in rows)
    assert all(r["top_k"] == kept and r["parametrization"] == "sigmoid" for r in rows)
    assert {r["featurizer"] for r in rows} == {"gate"}
    by_unit = {r["unit"]: r for r in rows}
    assert all(by_unit[i]["theta"] == pytest.approx(float(theta[i])) for i in by_unit)
    # …and the threshold split is the same set of units the cut keeps
    assert {r["unit"] for r in rows if r["hard"]} == set(
        (theta > 0).nonzero().flatten().tolist()
    )
    none_rows = json.loads((cut / "none/rank.json").read_text())
    assert not any(r["hard"] for r in none_rows) and all(
        r["top_k"] == 0 for r in none_rows
    )


def test_a_sweep_over_the_cut_is_monotone_in_what_it_keeps(tmp_path: Path) -> None:
    """One document, one bundle, the kept-count axis: the rank table says each
    point kept exactly its ``top_k`` units and the kept sets are nested."""
    run = _run(tmp_path, {"fit": (_fit_document(), FIT_SET)}, "fit_only")
    theta = load_file(str(run / "fit/gate.safetensors"))["theta"]
    units = theta.numel()
    grid = sorted({0, 1, 2, units // 2, units})
    swept = _run(
        tmp_path,
        {
            "sweep": (
                _apply_document(
                    {
                        "file_path": str(run / "fit/gate.safetensors"),
                        "top_k": {"sweep": grid},
                    },
                    rank=True,
                ),
                {},
            )
        },
        "sweep",
    )
    rows = json.loads((swept / "sweep/rank.json").read_text())
    kept_by_k: dict[int, set[int]] = {}
    for row in rows:
        k = row["coords"]["featurizers.gate.top_k"]
        assert row["top_k"] == k
        if row["hard"]:
            kept_by_k.setdefault(k, set()).add(row["unit"])
        else:
            kept_by_k.setdefault(k, set())
    assert sorted(kept_by_k) == grid
    for k in grid:
        assert len(kept_by_k[k]) == k
    for smaller, larger in zip(grid, grid[1:]):
        assert kept_by_k[smaller] <= kept_by_k[larger]
    # every point scored, one row per example per point
    points = _by_point(swept / "sweep/iia.json")
    assert len(points) == len(grid)
