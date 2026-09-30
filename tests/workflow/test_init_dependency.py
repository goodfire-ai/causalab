"""A featurizer's ``init.file_path`` is a run-tree load, so a fit started from
a basis another step produces depends on that step (workflow spec §2.10).

Before this, only a featurizer's own ``file_path`` and a params entry counted
as load sites: a workflow whose ``fit`` started from the ``pca`` step's basis
was refused at load as a missing artifact, and the PCA basis had to come from
an earlier run through ``--artifacts-root`` (the shipped ``das_pca_init.json``
pattern, and the 2026-09-07 DAS experiments' two-workflow layout).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from causalab.workflow.document import _walk_run_tree_paths, load_workflow
from tests._helpers.paths import PROTOCOLS_DIR

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[2]
PROTOCOLS = PROTOCOLS_DIR


def _steps() -> dict[str, Any]:
    return {
        "harvest": {
            "type": "intervention_protocol",
            "document": str(PROTOCOLS / "harvest.json"),
        },
        "pca": {
            "type": "script",
            "script": {"module": "causalab.analysis.fit_pca"},
            "inputs": {
                "acts": {"step": "harvest", "file": "acts_L7_ans.safetensors"},
                "k": 32,
            },
            "outputs": {
                "weight": "basis.safetensors",
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
        "fit": {
            "type": "intervention_protocol",
            "document": str(PROTOCOLS / "das_pca_init.json"),
            "set": {"featurizers.rot.init.file_path": "pca/basis.safetensors"},
        },
    }


def test_a_fit_started_from_a_steps_basis_depends_on_that_step(env) -> None:
    loaded = load_workflow(
        {"version": "1", "output_dir": "init_dep", "steps": _steps()}, env
    )
    assert "pca" in loaded.dependencies["fit"]
    assert loaded.order.index("pca") < loaded.order.index("fit")
    assert loaded.order.index("harvest") < loaded.order.index("pca")


def test_the_walker_reads_the_init_load_site_and_still_not_the_save_section() -> None:
    raw = json.loads((PROTOCOLS / "das_pca_init.json").read_text())
    raw["method"]["featurizers"]["rot"]["init"]["file_path"] = "pca/basis.safetensors"
    steps = frozenset({"pca", "fit"})
    assert _walk_run_tree_paths(raw, steps) == {"pca/basis.safetensors"}
    # the document's own `save` file_paths are outputs, never loads
    assert not any(
        entry["file_path"] in _walk_run_tree_paths(raw, steps | {"rot"})
        for entry in raw["method"]["save"]
        if "file_path" in entry
    )
