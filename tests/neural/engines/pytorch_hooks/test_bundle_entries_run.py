"""Tensor handoffs between steps, end to end on tiny-random through the CLI.

Two shapes the protocol could not express before, each the acceptance case
of one half of the seam:

* **A swept fit, applied.** The fit step really sweeps (k × seed), so its
  bundle holds four rotations; the workflow picks the winning cell and the
  apply step selects *that* entry. The pre-existing pipeline test
  (``test_workflow_run.py``) collapses the same sweep to one point with
  ``set`` — which is exactly why it never saw this gap.
* **Mean ablation.** A harvest step reduces a read to its corpus mean at
  save time, and an ablation step swaps that constant in as a ``params``
  operand. The un-reduced activations never reach disk.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from causalab.cli import main
from causalab.io.env import read_safetensors_metadata

from tests.neural.engines.pytorch_hooks.conftest import TINY_LLAMA
from tests.protocol._docs import aggregation, saved
from tests.protocol._env import FIXTURES, fixture_input_overrides
from tests.tables import frame as table_frame
from tests._helpers.paths import PROTOCOLS_DIR

pytestmark = pytest.mark.smoke

REPO = Path(__file__).resolve().parents[4]
METHODS = str(PROTOCOLS_DIR)


def _fixture_inputs(name: str) -> dict[str, str]:
    """The shipped document's dataset refs, retargeted onto the fixture tables
    (tiny-random cannot tokenize every weekday; see tests/protocol/_env.py)."""
    return fixture_input_overrides(json.loads((Path(METHODS) / name).read_text()))


# tiny-random on CPU runs fp32, while the shipped documents declare bf16 —
# and the realization is part of a fit bundle's identity (§8), so the fit
# and the apply have to name the same one
TINY = {"model.key": TINY_LLAMA, "model.dtype": "fp32"}


def _run_workflow(base: Path, document: dict) -> Path:
    """Write one workflow beside a copy of the artifact fixtures and run it
    through the CLI, exactly as a user would."""
    artifacts = base / "artifacts"
    shutil.copytree(FIXTURES / "artifacts", artifacts, dirs_exist_ok=True)
    wf_dir = base / "workflows"
    wf_dir.mkdir()
    path = wf_dir / "wf.json"
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
    # the document names its own directory under the CLI-supplied root (§1.1)
    return out / document["output_dir"]


# --------------------------------------------------------------------------- #
# a swept fit, applied at one selected coordinate
# --------------------------------------------------------------------------- #


def _swept_pipeline() -> dict:
    return {
        "version": "1",
        "description": "fit k x seed at one cell, then apply the winning fit",
        "output_dir": "swept",
        "steps": {
            "fit": {
                "type": "intervention_protocol",
                "document": f"{METHODS}/weekdays_das_sweep.json",
                "set": {
                    **TINY,
                    **_fixture_inputs("weekdays_das_sweep.json"),
                    "sites.target.layers": 0,
                    "positions.best": {"index": -1},
                    "featurizers.rot.k": {"sweep": [2, 4]},
                    "train.seed": {"sweep": [0, 1]},
                    "train.steps": {"epochs": 1},
                    "train.batch": {"pairs": 2},
                },
            },
            "best_fit": {
                "type": "script",
                "script": {"module": "causalab.workflow.scripts.select"},
                "inputs": {
                    "table": {"step": "fit", "file": "iia.json"},
                    "choose": "max",
                    "emit": {
                        "best_k": "featurizers.rot.k",
                        "best_seed": "train.seed",
                    },
                },
                "outputs": {
                    "values": {
                        "file": "values.json",
                        "keys": {"best_k": 2, "best_seed": 0},
                    }
                },
            },
            "apply": {
                "type": "intervention_protocol",
                "document": f"{METHODS}/weekdays_das_apply.json",
                "set": {
                    **TINY,
                    **_fixture_inputs("weekdays_das_apply.json"),
                    "sites.target.layers": 0,
                    "featurizers.rot.file_path": "fit/rot.safetensors",
                    "featurizers.rot.k": {"artifact": "best_fit", "key": "best_k"},
                    "featurizers.rot.entry": {
                        "k": {"artifact": "best_fit", "key": "best_k"},
                        "seed": {"artifact": "best_fit", "key": "best_seed"},
                    },
                },
            },
        },
    }


@pytest.fixture(scope="module")
def swept_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _run_workflow(tmp_path_factory.mktemp("swept"), _swept_pipeline())


def test_the_swept_fit_writes_one_entry_per_point(swept_run):
    bundle = load_file(str(swept_run / "fit/rot.safetensors"))
    assert sorted(bundle) == [
        "weight[k=2,seed=0]",
        "weight[k=2,seed=1]",
        "weight[k=4,seed=0]",
        "weight[k=4,seed=1]",
    ]


def test_the_entries_are_actually_different_fits(swept_run):
    """Selecting one entry only means something if the entries differ. Two
    seeds at one k must give different rotations (the init seed reaches the
    subspace) and a different k a different shape."""
    bundle = load_file(str(swept_run / "fit/rot.safetensors"))
    seed0, seed1 = bundle["weight[k=2,seed=0]"], bundle["weight[k=2,seed=1]"]
    assert seed0.shape == seed1.shape
    assert not torch.allclose(seed0, seed1)
    assert bundle["weight[k=4,seed=0]"].shape[1] == 4


def test_every_entry_carries_its_own_provenance(swept_run):
    """The stamp that used to be last-point-wins: each entry records the
    point that produced it, and the file level keeps only what they share."""
    stamped = read_safetensors_metadata(swept_run / "fit/rot.safetensors")
    assert stamped is not None
    entries = json.loads(stamped["entries"])
    assert entries["weight[k=2,seed=1]"]["coords"] == {"k": 2, "seed": 1}
    assert entries["weight[k=4,seed=0]"]["k"] == "4"
    coords = {
        json.dumps(record["coords"], sort_keys=True) for record in entries.values()
    }
    assert len(coords) == 4  # one entry per point, named by its coordinates
    assert "k" not in stamped  # k varies, so the file cannot claim one


def test_apply_consumed_the_selected_entry(swept_run):
    chosen = json.loads((swept_run / "best_fit/values.json").read_text())
    assert chosen["best_k"] in (2, 4)
    assert chosen["best_seed"] in (0, 1)
    manifest = json.loads((swept_run / "workflow.json").read_text())
    assert manifest["steps"]["apply"]["status"] == "completed"
    iia = table_frame(swept_run / "apply/iia.json")
    assert len(iia) == 2  # the weekdays/data#test fixture rows
    assert iia["value"].dtype.kind == "f"


def test_an_unselected_load_refuses_before_anything_runs(tmp_path):
    """The gap this closes: without a selector the load used to validate
    clean and die with a KeyError after the fit had already run."""
    document = _swept_pipeline()
    document["steps"]["apply"]["set"].pop("featurizers.rot.entry")
    document["steps"]["apply"]["set"].pop("featurizers.rot.k")
    wf_dir = tmp_path / "workflows"
    wf_dir.mkdir()
    (wf_dir / "wf.json").write_text(json.dumps(document))
    code = main(
        [
            "validate",
            "--engine",
            "auto",
            str(wf_dir / "wf.json"),
            "--data-root",
            str(FIXTURES / "data"),
            "--artifacts-root",
            str(tmp_path),
        ]
    )
    assert code != 0


# --------------------------------------------------------------------------- #
# mean ablation: reduce at save, swap as a constant
# --------------------------------------------------------------------------- #


def _harvest_doc(reduce: bool) -> dict:
    entry = saved(
        "acts", "original", "acts.safetensors", **({"reduce": "mean"} if reduce else {})
    )
    return {
        "header": {
            "protocol_version": "4",
            "description": "harvest one site over the train split",
        },
        "model": {"key": TINY_LLAMA, "revision": "main"},
        "data": {"base": {"dataset": "weekdays/data#train", "field": "input"}},
        "method": {
            "intervened_models": {"original": {"input": "base", "reads": ["acts"]}},
            "positions": {"tap": {"index": -1}},
            "sites": {"target": {"component": "block_output", "layers": [0]}},
            "reads": {"acts": {"site": "target", "pos": "tap"}},
            "save": [entry],
        },
    }


def _ablate_doc() -> dict:
    return {
        "header": {
            "protocol_version": "4",
            "description": "mean-ablate the site by swapping in the corpus mean",
        },
        "model": {"key": TINY_LLAMA, "revision": "main"},
        "data": {"base": {"dataset": "weekdays/data#test", "field": "input"}},
        "method": {
            "intervened_models": {
                "ablated": {"input": "base", "reads": ["logits"], "writes": ["ablate"]}
            },
            "positions": {"tap": {"index": -1}},
            "sites": {
                "target": {"component": "block_output", "layers": [0]},
                "lm_head": {"component": "lm_head"},
            },
            "params": {
                # the producer keyed the bundle by its read's name, not by the
                # params convention 'value' — 'slot' is how a consumer says so
                "mu": {
                    "file_path": "harvest/acts.safetensors",
                    "entry": {"slot": "acts"},
                }
            },
            "reads": {"logits": {"site": "lm_head", "pos": -1}},
            "writes": {
                "ablate": {"site": "target", "pos": "tap", "do": {"swap": "mu"}},
            },
            "save": [
                saved(
                    "logits",
                    "ablated",
                    "ld.json",
                    aggregation(
                        "logit_diff",
                        a="base_answer",
                        b="cf_answer",
                    ),
                )
            ],
        },
    }


@pytest.fixture(scope="module")
def ablation_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    base = tmp_path_factory.mktemp("ablate")
    docs = base / "docs"
    docs.mkdir()
    (docs / "harvest.json").write_text(json.dumps(_harvest_doc(reduce=True)))
    (docs / "rows.json").write_text(json.dumps(_harvest_doc(reduce=False)))
    (docs / "ablate.json").write_text(json.dumps(_ablate_doc()))
    document = {
        "version": "1",
        "description": "harvest a corpus mean, then mean-ablate with it",
        "output_dir": "ablate",
        "steps": {
            "harvest": {
                "type": "intervention_protocol",
                "document": "../docs/harvest.json",
            },
            "rows": {"type": "intervention_protocol", "document": "../docs/rows.json"},
            "ablate": {
                "type": "intervention_protocol",
                "document": "../docs/ablate.json",
            },
        },
    }
    return _run_workflow(base, document)


def test_a_reduced_read_saves_one_vector(ablation_run):
    mean = load_file(str(ablation_run / "harvest/acts.safetensors"))["acts"]
    rows = load_file(str(ablation_run / "rows/acts.safetensors"))["acts"]
    assert mean.ndim == 1
    assert mean.shape[0] == rows.shape[-1]
    assert rows.numel() > mean.numel()  # what never has to reach disk


def test_the_reduction_is_the_mean_of_the_rows(ablation_run):
    """Sanity check against the un-reduced save of the same read."""
    mean = load_file(str(ablation_run / "harvest/acts.safetensors"))["acts"]
    rows = load_file(str(ablation_run / "rows/acts.safetensors"))["acts"]
    expected = rows.to(torch.float32).reshape(-1, rows.shape[-1]).mean(dim=0)
    assert torch.allclose(mean, expected, atol=1e-6)


def test_the_ablation_consumed_the_harvested_mean(ablation_run):
    ld = table_frame(ablation_run / "ablate/ld.json")
    assert len(ld) == 2
    assert ld["value"].notna().all()
    manifest = json.loads((ablation_run / "workflow.json").read_text())
    assert manifest["steps"]["ablate"]["status"] == "completed"
