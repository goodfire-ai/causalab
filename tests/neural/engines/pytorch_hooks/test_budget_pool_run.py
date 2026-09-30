"""A budget pool end to end through the CLI (spec §2.5 ``pool``): two budget
gates at two sites of the tiny fixture fitted as ONE budget under a
``kept`` log-uniform schedule, then the two bundles replayed through one
pooled ``top_k`` axis (§3.2 ``axes``).

The identities: every checkpoint of the fit shows one ``k`` on both members;
the diagnostics name the pool and its unit count and the members' kept counts
sum to the pooled cut; a pooled cut of ``0`` scores the unpatched model and a
cut of ``N`` the swap of both sites at once; and ``rank.json``'s ``pool_rank``
is one permutation of the pool's units across the two gates.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from causalab.cli import main
from causalab.io.env import read_safetensors_metadata
from tests.neural.engines.pytorch_hooks.conftest import TINY_QWEN35_MOE
from tests.protocol._docs import UNWRITTEN
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


def _two_sites(doc: dict, gate_a: dict, gate_b: dict | None) -> dict:
    """The shipped DBM document with a second site (``mlp_output``, layer 1)
    written through gate ``gate_b`` into the same intervened model; ``None``
    for either gate spells the plain swap of that site."""
    method = doc["method"]
    method["sites"]["mlp"] = {"component": "mlp_output", "layers": [1]}
    method["reads"]["v_cf_b"] = {"site": "mlp", "pos": -1}
    method["intervened_models"][UNWRITTEN]["reads"].append("v_cf_b")
    method["writes"]["mask_b"] = {"site": "mlp", "pos": -1, "do": {"swap": "v_cf_b"}}
    method["intervened_models"]["masked"]["writes"].append("mask_b")
    method["featurizers"] = {}
    for name, gate, read, write in (
        ("gate", gate_a, "v_cf", "mask"),
        ("gate_b", gate_b, "v_cf_b", "mask_b"),
    ):
        if gate is None:
            method["reads"][read].pop("featurizer", None)
            method["writes"][write].pop("featurizer", None)
            continue
        method["featurizers"][name] = {"kind": "gate", **gate}
        method["reads"][read]["featurizer"] = name
        method["writes"][write]["featurizer"] = name
    if not method["featurizers"]:
        del method["featurizers"]
    return doc


def _fit_document(schedule: dict) -> dict:
    fit = json.loads((PROTOCOLS / "dbm.json").read_text())
    gate = {"parametrization": "budget", "k_schedule": schedule, "pool": "mib"}
    fit = _two_sites(fit, gate, gate)
    train = fit["method"]["train"]
    del train["objective"]["l1"]  # the task term alone; the pool sets the sparsity
    train["params"] = ["gate", "gate_b"]
    train.pop("anneal", None)
    fit["method"]["save"].append(
        {"value": "gate_b", "site": "mlp", "file_path": "gate_b.safetensors"}
    )
    fit["method"]["save"].append(
        {
            "kind": "trajectory",
            "every": {"count": 2},
            "file_path": "trajectory.safetensors",
        }
    )
    return fit


def _apply_document(
    run: Path | None, cuts: list[int] | None, *, base_metric: bool = False
) -> dict:
    """The replay: both bundles through one pooled `top_k` axis, or — `run`
    None — the plain swap of both sites."""
    apply = json.loads((PROTOCOLS / "dbm_apply.json").read_text())
    if run is None:
        apply = _two_sites(apply, None, None)
    else:
        top_k = {"axis": "cut"} if cuts is not None and len(cuts) > 1 else cuts[0]
        loaded = {"parametrization": "budget", "pool": "mib", "top_k": top_k}
        apply = _two_sites(
            apply,
            {**loaded, "file_path": str(run / "fit/gate.safetensors")},
            {**loaded, "file_path": str(run / "fit/gate_b.safetensors")},
        )
        if cuts is not None and len(cuts) > 1:
            apply["axes"] = {"cut": {"values": cuts}}
        apply["method"]["save"].append({"kind": "rank", "file_path": "rank.json"})
    method = apply["method"]
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


def _by_cut(table: Path) -> dict[int, list[float]]:
    out: dict[int, list[float]] = {}
    for row in json.loads(table.read_text()):
        # a metric table flattens the point's coordinates into columns
        out.setdefault(int(row["axes.cut"]), []).append(float(row["value"]))
    return out


def test_a_pool_fits_as_one_budget_and_replays_through_one_pooled_cut(
    tmp_path: Path,
) -> None:
    run = _run(
        tmp_path,
        {
            "fit": (
                _fit_document(
                    {
                        "kind": "log_uniform",
                        "low": 1,
                        "high": 15,
                        "eval": 4,
                        "of": "kept",
                    }
                ),
                FIT_SET,
            ),
            "plain": (_apply_document(None, None, base_metric=True), {}),
        },
        "fit",
    )
    diagnostics = json.loads((run / "fit/fit_diagnostics.json").read_text())[0][
        "featurizers"
    ]
    a, b = diagnostics["gate"], diagnostics["gate_b"]
    units = int(a["width"] + b["width"])  # per-coordinate gates: units = widths
    for gate in (a, b):
        assert gate["pool"] == "mib" and gate["pool_units"] == float(units)
        assert gate["k_schedule"]["of"] == "kept"
        # `eval: 4` KEPT units → the pooled cut patches N − 4
        assert gate["eval_k"] == float(units - 4)
    assert a["k"] == b["k"] and 1 <= units - a["k"] <= 15
    # the members' kept counts sum to the pooled cut
    assert a["hard_mask_size"] + b["hard_mask_size"] == units - 4
    # every checkpoint: one k on both members
    trajectory = read_safetensors_metadata(run / "fit/trajectory.safetensors")
    assert trajectory is not None
    records = json.loads(trajectory["entries"]).values()
    assert records and all(r["gate.k"] == r["gate_b.k"] for r in records)
    for bundle in ("gate", "gate_b"):
        header = read_safetensors_metadata(run / f"fit/{bundle}.safetensors")
        assert header is not None
        assert header["pool"] == "mib" and header["pool_units"] == str(units)

    cuts = _run(
        tmp_path,
        {"cuts": (_apply_document(run, [0, 3, units]), {})},
        "cuts",
    )
    by_cut = _by_cut(cuts / "cuts/iia.json")
    assert set(by_cut) == {0, 3, units}
    swap = [float(r["value"]) for r in json.loads((run / "plain/iia.json").read_text())]
    base = [
        float(r["value"]) for r in json.loads((run / "plain/iia_base.json").read_text())
    ]
    assert swap != base
    # nothing patched: the model as it was; everything patched: both swaps
    assert by_cut[0] == pytest.approx(base, rel=1e-5, abs=1e-6)
    assert by_cut[units] == pytest.approx(swap, rel=1e-5, abs=1e-6)
    # the rank table: pooled ranks are one permutation across the two gates,
    # and the cut keeps exactly the units whose pooled rank is below it
    rows = json.loads((cuts / "cuts/rank.json").read_text())
    at_3 = [r for r in rows if r["coords"]["axes.cut"] == 3]
    assert len(at_3) == units
    assert sorted(r["pool_rank"] for r in at_3) == list(range(units))
    assert all(r["pool"] == "mib" and r["top_k"] == 3 for r in at_3)
    assert all(r["hard"] == (r["pool_rank"] < 3) for r in at_3)
    assert sum(r["hard"] for r in at_3) == 3


def test_a_pooled_bundle_is_refused_alone(tmp_path: Path) -> None:
    """The stamp does its job: one member replayed with no `pool` is refused
    before any forward (rule 15)."""
    run = _run(
        tmp_path,
        {"fit": (_fit_document({"kind": "fixed", "k": 3}), FIT_SET)},
        "fit",
    )
    apply = json.loads((PROTOCOLS / "dbm_apply.json").read_text())
    apply["method"]["featurizers"]["gate"] = {
        "kind": "gate",
        "parametrization": "budget",
        "file_path": str(run / "fit/gate.safetensors"),
        "top_k": 2,
    }
    docs_dir = tmp_path / "alone_docs"
    docs_dir.mkdir()
    (docs_dir / "apply.json").write_text(json.dumps(apply))
    workflow = {
        "version": "1",
        "output_dir": "alone",
        "steps": {
            "apply": {
                "type": "intervention_protocol",
                "document": str(docs_dir / "apply.json"),
                "set": PINS,
            }
        },
    }
    path = tmp_path / "alone.json"
    path.write_text(json.dumps(workflow))
    code = main(
        [
            "run",
            "--engine",
            "auto",
            str(path),
            "--data-root",
            str(FIXTURES / "data"),
            "--artifacts-root",
            str(tmp_path / "artifacts"),
            "--out",
            str(tmp_path / "run"),
        ]
    )
    assert code == 1
