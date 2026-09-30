"""A ``script`` step end to end on tiny-random, through the real CLI.

Two directions in one pipeline, because they are the two halves of the claim
that a script step is a first-class step and not an escape from the record:

* **protocol → script → script** — a harvested activation bundle is fitted by
  ``causalab.analysis.fit_pca``; the spectrum table it writes is then ranked and drawn by
  further script steps, with no sweep axes to group by.
* **protocol → script → protocol** — the fitted basis is loaded back by a later
  protocol step as a ``pca`` featurizer through the ordinary run-tree
  ``file_path`` overlay, with its ArtifactIdentity checked. That is the handoff
  the manifold pipelines are built on, and the reason the **runner** stamps
  identity rather than trusting a script to remember.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from safetensors.torch import load_file

from causalab.cli import main
from causalab.io.env import read_safetensors_metadata

from tests.neural.engines.pytorch_hooks.conftest import TINY_LLAMA
from tests.protocol._docs import saved
from tests.protocol._env import FIXTURES
from tests.tables import frame as table_frame
from tests._helpers.paths import PROTOCOLS_DIR

pytestmark = pytest.mark.smoke

REPO = Path(__file__).resolve().parents[4]
METHODS = str(PROTOCOLS_DIR)  # absolute: the workflow is in tmp

#: A pure-read document that loads the fitted basis as a ``pca`` featurizer.
#: Its ``file_path`` names the script step's run tree, which is what makes the
#: edge a derived dependency (§3) — nothing about it is script-specific.
PROJECT_DOC = {
    "header": {
        "protocol_version": "4",
        "description": "read L0 through the fitted PCA basis",
    },
    "model": {"key": TINY_LLAMA, "revision": "main"},
    "data": {"base": {"dataset": "weekdays/data#train", "field": "input"}},
    "method": {
        "intervened_models": {"original": {"input": "base", "reads": ["coords"]}},
        "positions": {"answer_tok": {"index": -1}},
        "sites": {"L0": {"component": "block_output", "layers": [0]}},
        "featurizers": {
            "basis": {"kind": "pca", "k": 2, "file_path": "fit/weight.safetensors"}
        },
        "reads": {"coords": {"site": "L0", "pos": "answer_tok", "featurizer": "basis"}},
        "save": [saved("coords", "original", "coords.safetensors")],
    },
}


def _workflow(project_doc: Path) -> dict:
    tiny = {"model.key": TINY_LLAMA, "model.dtype": "fp32"}
    return {
        "version": "1",
        "description": "harvest -> fit_pca -> (select, plot) and back into a protocol step",
        "output_dir": "pca_pipeline",
        "steps": {
            "harvest": {
                "type": "intervention_protocol",
                "document": f"{METHODS}/harvest.json",
                "set": {**tiny, "sites.L7.layers": 0, "sites.L21.layers": 1},
            },
            "fit": {
                "type": "script",
                "script": {"module": "causalab.analysis.fit_pca"},
                "inputs": {
                    "acts": {
                        "step": "harvest",
                        "file": "acts_L7_ans.safetensors",
                        "slot": "acts_L7_ans",
                    },
                    "k": 2,
                },
                "outputs": {
                    "weight": "weight.safetensors",
                    "spectrum": {
                        "file": "spectrum.json",
                        "columns": {
                            "pc": "int64",
                            "explained_variance": "float64",
                            "explained_variance_ratio": "float64",
                        },
                    },
                },
            },
            "top_pc": {
                "type": "script",
                "script": {"module": "causalab.workflow.scripts.select"},
                "inputs": {
                    "table": {"step": "fit", "file": "spectrum.json"},
                    "choose": "max",
                    "value": "explained_variance_ratio",
                    "emit": {"best_pc": "pc"},
                },
                "outputs": {"values": {"file": "values.json", "keys": {"best_pc": 0}}},
            },
            "scree": {
                "type": "script",
                "script": {"module": "causalab.io.plots.workflow_figures"},
                "inputs": {
                    "table": {"step": "fit", "file": "spectrum.json"},
                    "plot": "lines",
                    "x": "pc",
                    "value": "explained_variance_ratio",
                },
                "outputs": {
                    "figure": "scree.png",
                    "plotted": {"file": "scree.json"},
                },
            },
            "project": {"type": "intervention_protocol", "document": str(project_doc)},
        },
    }


@pytest.fixture(scope="module")
def transform_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One full CLI run of the pipeline; the assertions below share it."""
    root = tmp_path_factory.mktemp("transform-wf")
    artifacts = root / "artifacts"
    shutil.copytree(FIXTURES / "artifacts", artifacts, dirs_exist_ok=True)
    project_doc = root / "project.json"
    project_doc.write_text(json.dumps(PROJECT_DOC, indent=2))
    wf_path = root / "wf.json"
    wf_path.write_text(json.dumps(_workflow(project_doc), indent=2))
    out = root / "run"
    code = main(
        [
            "run",
            "--engine",
            "auto",
            str(wf_path),
            "--data-root",
            str(FIXTURES / "data"),
            "--artifacts-root",
            str(artifacts),
            "--out",
            str(out),
        ]
    )
    assert code == 0
    return out / "pca_pipeline"


def test_script_step_writes_its_declared_outputs(transform_run: Path) -> None:
    assert (transform_run / "fit" / "weight.safetensors").is_file()
    assert (transform_run / "fit" / "spectrum.json").is_file()
    weight = load_file(str(transform_run / "fit" / "weight.safetensors"))["weight"]
    # (d, k): the featurizer convention, so a protocol step can load it as-is
    assert weight.shape == (16, 2)


def test_spectrum_table_matches_the_declared_columns(transform_run: Path) -> None:
    spectrum = table_frame(transform_run / "fit" / "spectrum.json")
    assert list(spectrum.columns) == [
        "pc",
        "explained_variance",
        "explained_variance_ratio",
    ]
    assert len(spectrum) == 2  # k = 2
    assert spectrum["explained_variance_ratio"].is_monotonic_decreasing


def test_select_ranks_a_script_table_as_written(transform_run: Path) -> None:
    """The rows are the producing script's, not a re-aggregation: the winning
    pc is 0, the leading component, which a collapse-to-one-row mean could not
    name. A script step publishes no axes, so `select` groups by nothing."""
    values = json.loads((transform_run / "top_pc" / "values.json").read_text())
    assert values == {"best_pc": 0}


def test_fitted_basis_carries_a_checkable_identity(transform_run: Path) -> None:
    metadata = read_safetensors_metadata(transform_run / "fit" / "weight.safetensors")
    assert metadata is not None
    # inherited from the harvested activations, so the basis is provably a fit
    # on this model at the site the read came from
    assert metadata["model_key"] == TINY_LLAMA
    assert json.loads(metadata["site"]) == {"component": "block_output", "layers": [0]}
    # from the op's params, and from the step itself
    # declared by the script, because only it knows its own parameter — a
    # consuming `pca` featurizer's identity check requires the rank
    assert metadata["k"] == "2"
    # stamped by the runner: dtype from the tensor, the engine name
    assert metadata["dtype"] == "fp32"
    assert metadata["engine"] == "script"


def test_protocol_step_consumes_the_fitted_basis(transform_run: Path) -> None:
    """The transform → protocol direction: the identity check on the loaded
    featurizer is a real one, and the read comes back in the 2-d feature
    space the basis defines."""
    coords = load_file(str(transform_run / "project" / "coords.safetensors"))["coords"]
    assert coords.shape == (2, 1, 2)  # 2 examples x 1 position x k


def test_manifest_records_the_script_step(transform_run: Path) -> None:
    manifest = json.loads((transform_run / "workflow.json").read_text())
    fit = manifest["steps"]["fit"]
    assert fit["type"] == "script"
    assert fit["script"] == "causalab.analysis.fit_pca"
    assert fit["status"] == "completed"
    assert len(fit["digest"]) == 64
    assert len(fit["script_sha256"]) == 64
    assert fit["files"] == ["spectrum.json", "weight.safetensors"]
