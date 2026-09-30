"""Run the documented PCA/projection steps through the CLI, with real files."""

import json
import re
from pathlib import Path
from typing import Any

import pytest
import torch

from causalab.cli import main
from causalab.io.step_io import read_tensor_with_identity, write_table, write_tensor
from causalab.io.tensor_files import save_file
from causalab.protocol.bundles import entry_key
from causalab.io.env import build_artifact_identity, check_artifact_identity

pytestmark = pytest.mark.smoke

GUIDE = Path(__file__).resolve().parents[2] / "docs/multi_token_analysis.md"


def _example(heading: str) -> dict[str, Any]:
    section = GUIDE.read_text().split(f"### {heading}\n", 1)[1].split("\n### ", 1)[0]
    block = re.search(r"```json\n(.*?)\n```", section, re.DOTALL)
    assert block is not None, f"{heading}: missing workflow example"
    return json.loads("{" + block.group(1) + "}")


def _document(root: Path, steps: dict[str, Any]) -> Path:
    path = root / "workflow.json"
    path.write_text(json.dumps({"version": "1", "output_dir": "run", "steps": steps}))
    return path


def _run(path: Path, *extra: str) -> int:
    return main(
        [
            "run",
            str(path),
            "--out",
            str(path.parent / "runs"),
            "--engine",
            "pytorch_hooks",
            "--device",
            "cpu",
            *extra,
        ]
    )


def test_sparse_example_reruns_on_resume_when_labels_change(tmp_path: Path, capsys):
    """A `{"path": …}` input enters the step's identity as its path alone, so
    the record's `input_digests` is what holds the step to the rows it read:
    editing them makes `--resume` re-run the step (spec §8)."""
    generator = torch.Generator().manual_seed(7)
    acts = torch.randn(240, 2, 5, generator=generator, dtype=torch.float64)
    write_tensor(tmp_path / "acts.safetensors", acts, slot="acts")
    rows = [
        {
            "id": str(i),
            "split": "train" if i < 140 else "validation" if i < 190 else "evaluation",
            "output_label": "positive" if acts[i, 0, 4] > 0 else "negative",
            "operand_value": float(3 * acts[i, 0, 4]),
        }
        for i in range(len(acts))
    ]
    labels = tmp_path / "data/probe_rows.json"
    write_table(labels, rows)
    steps = {
        "pca": {
            "type": "script",
            "script": {"module": "causalab.analysis.pca_by_position"},
            "inputs": {
                "acts": {"path": "acts.safetensors"},
                "train_rows": list(range(140)),
                "k": 5,
            },
            "outputs": {
                "weight": "weight.safetensors",
                "mean": "mean.safetensors",
                "coordinates": "coordinates.safetensors",
                "spectrum": "spectrum.json",
            },
        },
        **_example("Final PCA step: sparse concept probes"),
    }
    path = _document(tmp_path, steps)
    # The CLI runs from the checkout, while relative inputs live beside the document.
    assert _run(path) == 0
    scores = tmp_path / "runs/run/sparse_probes/scores.json"
    original = scores.read_bytes()
    capsys.readouterr()
    assert _run(path, "--resume") == 0
    assert "reused sparse_probes" in capsys.readouterr().out
    rows[-1]["operand_value"] = 1000.0
    write_table(labels, rows)
    assert _run(path, "--resume") == 0
    assert "completed sparse_probes" in capsys.readouterr().out
    assert scores.read_bytes() != original


def test_projection_example_inherits_selected_bundle_identity(tmp_path: Path):
    identity = build_artifact_identity(
        model_key="test/model", model_revision="local", model_dtype="fp32"
    )
    for slot in ("acts", "weight"):
        tensors, entries = {}, {}
        for layer in (12, 13):
            coords = {"layer": layer}
            if slot == "weight":
                coords["k"] = 2
            key = entry_key(slot, ",".join(f"{k}={v}" for k, v in coords.items()))
            tensors[key] = torch.ones(2, 3) if slot == "acts" else torch.eye(3)[:, :2]
            entries[key] = {
                "slot": slot,
                "coords": coords,
                "site": json.dumps({"component": "block_output", "layers": [layer]}),
            }
        save_file(
            tensors,
            str(tmp_path / f"{slot}.safetensors"),
            metadata={**identity, "entries": json.dumps(entries)},
        )
    path = _document(tmp_path, _example("Reconstruct a saved subspace component"))
    assert _run(path) == 0
    for name in ("coordinates", "reconstructed"):
        _, saved_identity = read_tensor_with_identity(
            tmp_path / f"runs/run/projection/{name}.safetensors"
        )
        check_artifact_identity(saved_identity, identity, what=name)
        assert json.loads(saved_identity["site"]) == {
            "component": "block_output",
            "layers": [12],
        }
        assert saved_identity["engine"] == "script"


def test_fourier_example_and_frozen_apply(tmp_path: Path, capsys):
    from causalab.analysis.fourier_artifacts import load_fit
    from tests.analysis.test_fourier_probe import fixture

    inputs = fixture()
    identity = build_artifact_identity(
        model_key="test/model", model_revision="local", model_dtype="fp32"
    )
    write_tensor(
        tmp_path / "acts.safetensors",
        torch.from_numpy(inputs["acts"]),
        slot="acts",
        identity=identity,
    )
    rows = [{**r, "operand_value": r["number"]} for r in inputs["rows"]]
    labels = tmp_path / "data/probe_rows.json"
    write_table(labels, rows)
    steps = _example("Fourier probes on saved activations")
    steps["apply"] = {
        "type": "script",
        "script": {"module": "causalab.analysis.apply_fourier_probe"},
        "inputs": {
            "acts": {"path": "acts.safetensors"},
            "weight": {"step": "fourier", "file": "weight.safetensors"},
            "bias": {"step": "fourier", "file": "bias.safetensors"},
        },
        "outputs": {
            name: f"{name}.safetensors"
            for name in ("predictions", "radius", "phase", "phase_defined")
        },
    }
    path = _document(tmp_path, steps)
    assert _run(path) == 0
    for name in steps["apply"]["outputs"]:
        tensor, saved_identity = read_tensor_with_identity(
            tmp_path / f"runs/run/apply/{name}.safetensors"
        )
        check_artifact_identity(saved_identity, identity, what=name)
        assert tensor.shape[:3] == (300, 1, 149)
    fit, _ = read_tensor_with_identity(
        tmp_path / "runs/run/fourier/predictions.safetensors"
    )
    replay, _ = read_tensor_with_identity(
        tmp_path / "runs/run/apply/predictions.safetensors"
    )
    torch.testing.assert_close(fit, replay)
    saved = load_fit(
        tmp_path / "runs/run/fourier", tmp_path / "acts.safetensors", labels
    )
    check_artifact_identity(saved["identity"], identity, what="Fourier fit")
    torch.testing.assert_close(torch.from_numpy(saved["predictions"]), fit)
    capsys.readouterr()
    assert _run(path, "--resume") == 0
    assert "reused fourier" in capsys.readouterr().out
    # the rows are a `{"path": …}` input: their bytes are in the record's
    # `input_digests`, not the identity, and --resume re-runs on a change —
    # the edited row is an evaluation row, so the scores move
    scores = tmp_path / "runs/run/fourier/scores.json"
    original = scores.read_bytes()
    rows[-1]["operand_value"] += 1
    write_table(labels, rows)
    assert _run(path, "--resume") == 0
    assert "completed fourier" in capsys.readouterr().out
    assert scores.read_bytes() != original
