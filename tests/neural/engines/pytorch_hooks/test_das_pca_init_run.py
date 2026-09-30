"""``demos/methods/protocols/das_pca_init.json`` end to end on tiny-random.

The handoff is harvest → ``fit_pca`` (a workflow through the CLI, as
``test_script_step_run.py`` runs it) → the shipped PCA-initialised DAS preset
through [`run_protocol`][causalab.protocol.pipeline.run_protocol], retargeted with ``--set``-style overrides the
way ``test_run_corpus.py`` retargets the corpus: tiny model, layer 0, and the
rank sweep cut to what a 16-wide site and a four-row harvest can hold: three
components, since centering four rows leaves three with variance. What
is asserted is the record — every fitted rotation names the data the basis it
started from was fitted over (``init_trained_on``, inherited by ``fit_pca``'s
output from the harvest) and which of its columns it took
(``init_components``) — plus the one refusal a mismatched basis has to
produce before anything runs.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from causalab.cli import main
from causalab.neural.engines.pytorch_hooks.engine import PytorchHooksEngine
from causalab.protocol.rules.errors import ValidationError
from causalab.protocol.pipeline import compile_protocol
from causalab.io.env import (
    FileArtifacts,
    FileDatasets,
    ResolutionEnv,
    read_safetensors_metadata,
)
from causalab.protocol.pipeline import run_protocol

from tests.neural.engines.pytorch_hooks.conftest import TINY_LLAMA
from tests.protocol._env import FIXTURES, steps_of
from tests.tables import frame as table_frame
from tests._helpers.paths import PROTOCOLS_DIR

pytestmark = pytest.mark.smoke

REPO = Path(__file__).resolve().parents[4]
PROTOCOLS = PROTOCOLS_DIR
TINY = {"model.key": TINY_LLAMA, "model.dtype": "fp32"}
#: The shipped presets read ``weekdays/data#train`` (two rows); a rank-3 basis
#: needs four, so both halves of the handoff are retargeted to the four-row
#: ``weekdays/train`` fixture table — the split the basis then records.
TRAIN = "weekdays/train"
DATA = {"data.base.dataset": TRAIN, "data.counterfactual.dataset": TRAIN}
#: where the workflow below leaves the fitted basis, relative to its run root
BASIS = "pca/fit/weight.safetensors"


def _pca_workflow() -> dict:
    """Harvest ``block_output`` L0 at the answer token over the four-row
    ``weekdays/train`` table, then fit its three principal components, the
    most that four centered rows carry."""
    return {
        "version": "1",
        "description": "harvest one site, fit the basis a DAS sweep starts from",
        "output_dir": "pca",
        "steps": {
            "harvest": {
                "type": "intervention_protocol",
                "document": str(PROTOCOLS / "harvest.json"),
                "set": {
                    **TINY,
                    "data.base.dataset": TRAIN,
                    "sites.L7.layers": 0,
                    "sites.L21.layers": 1,
                },
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
                    "k": 3,
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
        },
    }


def _overrides(**extra) -> dict:
    return {
        **TINY,
        **DATA,
        # The same ref for both roles: the visible train-equals-test ablation
        # (spec §2.2, rule 22). Under the tiny Llama tokenizer only Monday,
        # Friday, Saturday and Sunday are single tokens, so every Llama-legal
        # row has its entities in the four `weekdays/train` already uses — no
        # held-out table can be both Llama-legal and endpoint-disjoint from
        # it, and this smoke asserts the PCA-init record, not a held-out score.
        "train.eval.split": TRAIN,
        "sites.target.layers": 0,
        "featurizers.rot.k": {"sweep": [1, 2, 3]},
        "featurizers.rot.init.file_path": BASIS,
        "train.seed": {"sweep": [0, 1]},
        "train.steps": {"epochs": 1},
        "train.batch": {"pairs": 2},
        **extra,
    }


@pytest.fixture(scope="module")
def run_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The workflow's run tree, which is also the artifact root the DAS
    document resolves its ``init`` basis against."""
    root = tmp_path_factory.mktemp("das-pca-init")
    artifacts = root / "artifacts"
    shutil.copytree(FIXTURES / "artifacts", artifacts, dirs_exist_ok=True)
    wf = root / "wf.json"
    wf.write_text(json.dumps(_pca_workflow(), indent=2))
    code = main(
        [
            "run",
            "--engine",
            "auto",
            str(wf),
            "--data-root",
            str(FIXTURES / "data"),
            "--artifacts-root",
            str(artifacts),
            "--out",
            str(root / "run"),
        ]
    )
    assert code == 0
    return root / "run"


@pytest.fixture(scope="module")
def env(run_root: Path) -> ResolutionEnv:
    return ResolutionEnv(
        datasets=FileDatasets(root=FIXTURES / "data"),
        artifacts=FileArtifacts(root=run_root),
    )


@pytest.fixture(scope="module")
def das_out(run_root: Path, env: ResolutionEnv) -> Path:
    loaded = compile_protocol(
        PROTOCOLS / "das_pca_init.json", env=env, overrides=_overrides()
    )
    assert len(steps_of(loaded, env).points) == 6
    out = run_root / "das"
    result = run_protocol(loaded, env, PytorchHooksEngine(), out)
    assert set(result.files) >= {
        "iia.json",
        "logit_diff.json",
        "ce.json",
        "rot.safetensors",
    }
    return out


def test_every_point_fits_and_saves(das_out: Path):
    fitted = load_file(str(das_out / "rot.safetensors"))
    assert sorted(fitted) == sorted(
        f"weight[k={k},seed={s}]" for k in (1, 2, 3) for s in (0, 1)
    )
    for name, weight in fitted.items():
        k = weight.shape[1]
        assert weight.shape == (16, k), name
        torch.testing.assert_close(
            weight.T @ weight, torch.eye(k), atol=1e-5, rtol=0, msg=name
        )
    for table in ("iia.json", "logit_diff.json", "ce.json"):
        rows = table_frame(das_out / table)
        assert len(rows) == 6 * 4  # six points over the four train rows
    assert len(json.loads((das_out / "train_eval.json").read_text())) == 6


def test_the_bundle_records_the_basis_it_started_from(das_out: Path, run_root: Path):
    """The fit record (§8): the basis's data ref at file level (every point
    started from the same basis), and per entry the columns taken — which
    differ with ``k``, so they live in the entries table only."""
    basis_meta = read_safetensors_metadata(run_root / BASIS)
    assert basis_meta is not None
    stamped = read_safetensors_metadata(das_out / "rot.safetensors")
    assert stamped is not None
    # the harvest stamped the data it read, fit_pca's output inherited it, and
    # the fit record carries it as the basis's dataset ref
    assert basis_meta["trained_on"] == TRAIN
    assert stamped["init_trained_on"] == TRAIN
    assert "init_components" not in stamped  # differs with k, so per entry only
    entries = json.loads(stamped["entries"])
    for key, record in entries.items():
        k = record["coords"]["k"]
        assert json.loads(record["init_components"]) == list(range(k)), key
        assert record["init_trained_on"] == TRAIN


def test_the_fits_moved_off_the_pca_start(das_out: Path, run_root: Path):
    """One epoch is enough to leave the start; what has to stay put is the
    *record*, not the weight."""
    basis = load_file(str(run_root / BASIS))["weight"]
    fitted = load_file(str(das_out / "rot.safetensors"))
    assert not torch.equal(fitted["weight[k=3,seed=0]"], basis[:, :3])
    # two seeds complete the basis differently and order the batches
    # differently, so they are two fits
    assert not torch.equal(fitted["weight[k=2,seed=0]"], fitted["weight[k=2,seed=1]"])


def test_a_basis_from_another_layer_refuses_before_anything_runs(env):
    """The DAS document names layer 1; the basis was fitted at layer 0. The
    site record is part of the basis's identity, so the load refuses naming
    the field — no model is loaded, nothing is fitted."""
    with pytest.raises(ValidationError) as err:
        compile_protocol(
            PROTOCOLS / "das_pca_init.json",
            env=env,
            overrides=_overrides(**{"sites.target.layers": 1}),
        )
    assert err.value.rule == 15
    assert "'site'" in str(err.value) and "init" in str(err.value)
