"""A site-swept DAS fit started from a basis bundle with one entry per site.

A fit swept over sites can start each point from a basis fitted at that
point's own site: a harvest over the same sweep, turned into one start per
site by a script, is such a bundle (the random-PCA start of the
``arithmetic_fig2a`` replication). The bundle stamps the model file-wide and
the site per entry, because its entries differ in site. With no authored
``entry``, the executing point selects the entry its own coordinates name
(§2.5), and that entry's recorded site must be the point's site.

What is asserted is behaviour through ``run_protocol`` and the build: at a
learning rate that leaves the weight where it starts, each point's fitted
rotation is the first ``k`` columns of its own site's basis, under both a
``cayley`` and a ``stiefel`` map. A bundle whose entry records another site
than the coordinates that select it is refused per point before any weights
load, and the build refuses it again for a caller that builds a stack
directly. A gate has no per-point site check, so a gate start whose site is
stamped per entry still needs an authored ``entry``.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from causalab.io.env import (
    FileArtifacts,
    FileDatasets,
    ResolutionEnv,
    read_safetensors_metadata,
)
from causalab.io.tensor_files import TensorBundle
from causalab.neural.engines.pytorch_hooks import engine as hooks_engine
from causalab.neural.engines.pytorch_hooks.engine import PytorchHooksEngine
from causalab.neural.shared.featurizers import build_stack
from causalab.protocol.pipeline import compile_protocol, run_protocol
from causalab.protocol.rules.errors import ValidationError
from causalab.protocol.schema import FeaturizerSpec

from tests._helpers.paths import PROTOCOLS_DIR
from tests.neural.engines.pytorch_hooks.conftest import TINY_LLAMA
from tests.protocol._env import FIXTURES

pytestmark = pytest.mark.smoke

#: the tiny Llama's residual width and the two layers the fit sweeps
WIDTH = 16
LAYERS = (0, 1)
#: every basis holds four columns; the fit takes the first two
BASIS_COLUMNS = 4
K = 2
TRAIN = "weekdays/train"
BASIS = "starts/per_site.safetensors"


def _site(layer: int) -> str:
    """A site record as a producer stamps it per entry (§8)."""
    return json.dumps({"component": "block_output", "layers": [layer]}, sort_keys=True)


def _frames() -> dict[int, torch.Tensor]:
    """One seeded orthonormal ``(WIDTH, BASIS_COLUMNS)`` frame per layer,
    different at each layer, so a point that took the wrong entry shows."""
    generator = torch.Generator().manual_seed(11)
    return {
        layer: torch.linalg.qr(torch.randn(WIDTH, BASIS_COLUMNS, generator=generator))[
            0
        ].contiguous()
        for layer in LAYERS
    }


def _write_basis(root: Path, recorded_layer: dict[int, int]) -> dict[int, torch.Tensor]:
    """The per-site start bundle: the model stamped file-wide, one ``weight``
    entry per layer keyed by the fit's own site coordinate, each entry's site
    recorded as ``recorded_layer`` says (the true layer, unless a test lies)."""
    frames = _frames()
    tensors, entries = {}, {}
    for layer, frame in frames.items():
        key = f"weight[target.layers={layer}]"
        tensors[key] = frame
        entries[key] = {
            "slot": "weight",
            "coords": {"target.layers": layer},
            "site": _site(recorded_layer[layer]),
            "trained_on": f"{TRAIN}#train",
        }
    metadata = {
        "model_key": TINY_LLAMA,
        "model_revision": "main",
        "model_dtype": "fp32",
        "engine": "script",
        "dtype": "fp32",
        "entries": json.dumps(entries, sort_keys=True),
    }
    target = root / BASIS
    target.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(target), metadata=metadata)
    return frames


def _env(root: Path) -> ResolutionEnv:
    return ResolutionEnv(
        datasets=FileDatasets(root=FIXTURES / "data"),
        artifacts=FileArtifacts(root=root),
    )


def _overrides(parametrization: str) -> dict:
    """``das_pca_init.json`` on the tiny Llama at two block_output layers, one
    rank, a learning rate small enough that one epoch leaves each weight at
    its start to well inside the tolerance below."""
    return {
        "model.key": TINY_LLAMA,
        "model.dtype": "fp32",
        "data.base.dataset": TRAIN,
        "data.counterfactual.dataset": TRAIN,
        # the same ref for both roles, as in test_das_pca_init_run.py: this
        # smoke asserts where each fit starts, not a held-out score
        "train.eval.split": TRAIN,
        "sites.target.layers": {"sweep": list(LAYERS)},
        "featurizers.rot.k": K,
        "featurizers.rot.parametrization": parametrization,
        "featurizers.rot.init": {"file_path": BASIS},
        "train.seed": 0,
        "train.steps": {"epochs": 1},
        "train.batch": {"pairs": 2},
        "train.optimizer.lr": 1e-12,
    }


@pytest.fixture()
def root(tmp_path: Path) -> Path:
    shutil.copytree(FIXTURES / "artifacts", tmp_path / "artifacts", dirs_exist_ok=True)
    return tmp_path


@pytest.mark.parametrize("parametrization", ["cayley", "stiefel"])
def test_each_point_starts_from_its_own_sites_entry(root: Path, parametrization: str):
    frames = _write_basis(root, {layer: layer for layer in LAYERS})
    env = _env(root)
    loaded = compile_protocol(
        PROTOCOLS_DIR / "das_pca_init.json",
        env=env,
        overrides=_overrides(parametrization),
    )
    out = root / "fit"
    run_protocol(loaded, env, PytorchHooksEngine(), out)
    fitted = load_file(str(out / "rot.safetensors"))
    assert len(fitted) == len(LAYERS)
    for key, weight in fitted.items():
        layer = int(key.split("layers=")[1].rstrip("]").split(",")[0])
        torch.testing.assert_close(
            weight, frames[layer][:, :K], atol=1e-5, rtol=0, msg=key
        )
    # the two layers' starts differ, so the check above tells them apart
    assert not torch.allclose(frames[0][:, :K], frames[1][:, :K], atol=1e-2)


def test_an_entry_recording_another_site_is_refused_before_any_weights(
    root: Path, monkeypatch: pytest.MonkeyPatch
):
    """The entry the layer-1 point selects says it was fitted at layer 0.
    The engine's per-point check refuses it naming the entry and the site,
    and the monkeypatched ``load_model`` is never entered, so no cohort of
    the campaign has fitted by then."""

    def never(*args, **kwargs):
        raise AssertionError("load_model was entered")

    monkeypatch.setattr(hooks_engine, "load_model", never)
    _write_basis(root, {0: 0, 1: 0})
    env = _env(root)
    loaded = compile_protocol(
        PROTOCOLS_DIR / "das_pca_init.json",
        env=env,
        overrides=_overrides("cayley"),
    )
    with pytest.raises(ValidationError) as err:
        run_protocol(loaded, env, PytorchHooksEngine(), root / "fit")
    message = str(err.value)
    assert err.value.rule == 15
    assert "entry 'weight[target.layers=1]'" in message
    assert "mismatch on 'site'" in message
    assert "'init.entry'" in message  # the remedy is named


def _bundle_loader(path: Path):
    """What the executor's ``load_tensors`` hands the build: the file's
    tensors and its header's entries table."""
    header = dict(read_safetensors_metadata(path) or {})
    bundle = TensorBundle(
        tensors=load_file(str(path)),
        entry_coords=json.loads(header["entries"]),
        header=header,
    )
    return lambda file_path: bundle


def test_the_build_refuses_a_selected_entry_at_another_site(root: Path):
    """The second line of defense, for a caller that builds a stack without
    the engine's per-point check: given the point's site, the build holds
    the entry the point's coordinates select to it."""
    _write_basis(root, {0: 0, 1: 0})
    spec = FeaturizerSpec(
        kind="subspace", k=K, parametrization="cayley", init={"file_path": BASIS}
    )

    def build(layer: int):
        return build_stack(
            "rot",
            {"rot": spec},
            width=WIDTH,
            load_tensors=_bundle_loader(root / BASIS),
            stage_cache={},
            coords={"sites.target.layers": layer},
            site_records={"rot": {"component": "block_output", "layers": [layer]}},
        )

    build(0)  # the layer-0 entry records layer 0
    with pytest.raises(ValidationError) as err:
        build(1)
    assert "entry 'weight[target.layers=1]'" in str(err.value)
    assert "mismatch on 'site'" in str(err.value)


def _write_theta(root: Path) -> None:
    """A gate start with one ``theta`` entry per layer and the site stamped
    per entry, as a producer swept over the site writes it."""
    tensors, entries = {}, {}
    for layer in LAYERS:
        key = f"theta[target.layers={layer}]"
        tensors[key] = torch.zeros(WIDTH)
        entries[key] = {
            "slot": "theta",
            "coords": {"target.layers": layer},
            "site": _site(layer),
            "trained_on": f"{TRAIN}#train",
        }
    metadata = {
        "model_key": TINY_LLAMA,
        "model_revision": "main",
        "model_dtype": "fp32",
        "engine": "script",
        "dtype": "fp32",
        "entries": json.dumps(entries, sort_keys=True),
    }
    target = root / "starts" / "theta.safetensors"
    target.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(target), metadata=metadata)


def test_a_gate_start_with_its_site_stamped_per_entry_needs_an_authored_entry(
    root: Path,
):
    """Only a ``subspace`` start is checked per point. A gate start whose
    site differs between entries is refused at load, asking for an
    ``entry``, so no gate starts from a theta of an unchecked site."""
    _write_theta(root)
    # `--set` adds no key a document lacks, so the gate's `init` is written in
    doc = json.loads((PROTOCOLS_DIR / "dbm.json").read_text())
    doc["method"]["featurizers"]["gate"]["init"] = {
        "file_path": "starts/theta.safetensors"
    }
    path = root / "gate_fit.json"
    path.write_text(json.dumps(doc))
    with pytest.raises(ValidationError, match="must author 'init.entry'"):
        compile_protocol(
            path,
            env=_env(root),
            overrides={
                "model.key": TINY_LLAMA,
                "model.dtype": "fp32",
                "data.base.dataset": TRAIN,
                "data.counterfactual.dataset": TRAIN,
                "train.eval.split": TRAIN,
                "sites.target.layers": {"sweep": list(LAYERS)},
            },
        )
