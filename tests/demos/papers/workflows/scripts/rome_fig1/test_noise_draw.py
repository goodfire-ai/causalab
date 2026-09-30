"""``noise_draw.py``: the noise draw of the published Figure 1 (e, f, g).

The step must write the draw ROME's ``trace_with_patch`` makes,
``numpy.random.RandomState(1).randn(10, 4, 1600)``, in the layout the tracing
documents add to the table rows, and stamp it so that the documents load it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from causalab.io.env import FileArtifacts, FileDatasets, ResolutionEnv
from causalab.io.step_io import StepError, read_tensor_with_identity
from causalab.protocol.pipeline import compile_protocol
from causalab.protocol.rules.errors import ValidationError

from tests.demos.papers._scripts import PAPERS, load_script

pytestmark = pytest.mark.unit

noise_draw = load_script("rome_fig1", "noise_draw")

WORKFLOW = PAPERS / "workflows" / "rome_fig1.json"
TRACING = (
    PAPERS / "protocols" / "rome_fig1_trace_state.json",
    PAPERS / "protocols" / "rome_fig1_trace_window.json",
)
#: The first five values of NumPy's legacy Gaussian stream under seed 1. The
#: ``RandomState`` stream is frozen by NumPy's compatibility policy (NEP 19,
#: <https://numpy.org/neps/nep-0019-rng-policy.html>), so these values are
#: an independent fixed point, not a second call of the code under test.
FIRST_VALUES = (
    1.6243453636632417,
    -0.6117564136500754,
    -0.5281717522634557,
    -1.0729686221561705,
    0.8654076293246785,
)


def _write(tmp_path: Path, **overrides: object) -> Path:
    target = tmp_path / "noise_draw" / "noise.safetensors"
    noise_draw.main({**noise_draw.DEFAULTS, **overrides}, {"noise": target})
    return target


def test_the_bundle_is_numpy_randomstate_one_in_float32(tmp_path: Path) -> None:
    tensor, identity = read_tensor_with_identity(_write(tmp_path), slot="value")
    assert tensor.shape == (10, 4, 1600)
    assert tensor.dtype == torch.float32
    assert tensor.flatten()[:5].tolist() == [float(np.float32(v)) for v in FIRST_VALUES]
    # Row i is values [6400 i, 6400 (i + 1)) of the stream: the C-order layout
    # of ROME's `rs.randn(rows - 1, e - b, width)`, row i for table row i.
    stream = np.random.RandomState(1).randn(10 * 4 * 1600).astype(np.float32)
    assert np.array_equal(tensor.numpy(), stream.reshape(10, 4, 1600))
    assert identity["model_key"] == "gpt2-xl"
    assert identity["model_revision"] == noise_draw.DEFAULTS["model_revision"]


@pytest.mark.parametrize("field", ["samples", "tokens", "width"])
def test_a_shape_below_one_is_refused(tmp_path: Path, field: str) -> None:
    with pytest.raises(StepError, match=field):
        _write(tmp_path, **{field: 0})


def test_the_cli_defaults_are_the_workflow_inputs_and_the_documents_model() -> None:
    """The draw's identity must equal the model the tracing documents name,
    or they refuse it. The CLI defaults restate the workflow's inputs."""
    step = json.loads(WORKFLOW.read_text())["steps"]["noise_draw"]
    assert step["inputs"] == noise_draw.DEFAULTS
    for document in TRACING:
        model = json.loads(document.read_text())["model"]
        assert (model["key"], model["revision"]) == (
            step["inputs"]["model_key"],
            step["inputs"]["model_revision"],
        )


def test_the_cli_writes_the_same_bundle(tmp_path: Path) -> None:
    target = tmp_path / "cli.safetensors"
    assert noise_draw.cli(["--out", str(target)]) == 0
    written, _ = read_tensor_with_identity(target, slot="value")
    expected, _ = read_tensor_with_identity(_write(tmp_path), slot="value")
    assert torch.equal(written, expected)


def _env(artifacts: Path) -> ResolutionEnv:
    return ResolutionEnv(
        datasets=FileDatasets(root=PAPERS / "artifacts" / "data"),
        artifacts=FileArtifacts(root=artifacts),
    )


@pytest.mark.parametrize("document", TRACING, ids=lambda p: p.stem)
def test_the_tracing_documents_load_the_bundle(tmp_path: Path, document: Path) -> None:
    _write(tmp_path)
    compile_protocol(document, env=_env(tmp_path))


def test_a_bundle_stamped_for_another_revision_is_refused(tmp_path: Path) -> None:
    _write(tmp_path, model_revision="main")
    with pytest.raises(ValidationError, match=r"\[V15\]"):
        compile_protocol(TRACING[0], env=_env(tmp_path))
