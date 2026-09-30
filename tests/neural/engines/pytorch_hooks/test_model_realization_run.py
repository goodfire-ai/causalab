"""The document's realization reaches the weights, and the stamp records it.

Spec §2.1 puts ``dtype``/``quantization`` in the document; §8 says an engine
reads them per point rather than being told once. These tests drive the real
CLI on tiny-random and check both ends: what the loader was asked for, and
what the artifacts say afterwards.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from causalab.cli import main
from causalab.io.env import read_safetensors_metadata

from tests.protocol._env import CORPUS_DIR, FIXTURES
from tests.neural.engines.pytorch_hooks.conftest import TINY_LLAMA

pytestmark = pytest.mark.smoke


@pytest.fixture(scope="module")
def roots(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    artifacts = tmp_path_factory.mktemp("artifacts")
    shutil.copytree(FIXTURES / "artifacts", artifacts, dirs_exist_ok=True)
    return FIXTURES / "data", artifacts


def _argv(name: str, roots: tuple[Path, Path], out: Path, *extra: str) -> list[str]:
    data_root, artifacts_root = roots
    return [
        "run",
        "--engine",
        "auto",
        str(CORPUS_DIR / name),
        "--data-root",
        str(data_root),
        "--artifacts-root",
        str(artifacts_root),
        "--out",
        str(out),
        "--set",
        f"model.key={TINY_LLAMA}",
        "--set",
        "sites.target.layers=1",  # tiny-random is 2 layers deep
        *extra,
    ]


def _argv_harvest(roots: tuple[Path, Path], out: Path, *extra: str) -> list[str]:
    """01_harvest, retargeted at tiny-random (2 layers, 4 heads)."""
    data_root, artifacts_root = roots
    return [
        "run",
        "--engine",
        "auto",
        str(CORPUS_DIR / "01_harvest_im.json"),
        "--data-root",
        str(data_root),
        "--artifacts-root",
        str(artifacts_root),
        "--out",
        str(out),
        "--set",
        f"model.key={TINY_LLAMA}",
        "--set",
        "sites.L8.layers=0",
        "--set",
        "sites.L24.layers=1",
        *extra,
    ]


@pytest.fixture()
def load_spy(monkeypatch):
    """Record what the engine asks the model loader for."""
    from causalab.neural.engines.pytorch_hooks import engine as engine_module

    calls: list[dict] = []
    real = engine_module.load_model

    def spy(key, revision="main", **kwargs):
        calls.append({"key": key, "revision": revision, **kwargs})
        return real(key, revision, **kwargs)

    monkeypatch.setattr(engine_module, "load_model", spy)
    return calls


def test_the_documents_dtype_reaches_the_loader(roots, tmp_path, load_spy):
    """No flag involved: the document says bf16, so the weights are bf16."""
    doc = json.loads((CORPUS_DIR / "02_interchange_im.json").read_text())
    doc["model"]["dtype"] = "bf16"
    document = tmp_path / "bf16_im.json"
    document.write_text(json.dumps(doc))
    data_root, artifacts_root = roots
    assert (
        main(
            [
                "run",
                "--engine",
                "auto",
                str(document),
                "--data-root",
                str(data_root),
                "--artifacts-root",
                str(artifacts_root),
                "--out",
                str(tmp_path / "out"),
                "--set",
                f"model.key={TINY_LLAMA}",
                "--set",
                "sites.target.layers=1",
            ]
        )
        == 0
    )
    assert [call["dtype"] for call in load_spy] == ["bf16"]
    assert all(call["quantization"] is None for call in load_spy)


def test_the_dtype_flag_goes_through_the_document(roots, tmp_path, load_spy):
    """``--dtype`` is ``--set model.dtype`` (§9), so the run receipt shows it."""
    out = tmp_path / "out"
    assert (
        main(_argv("02_interchange_im.json", roots, out, "--dtype", "bf16", "--record"))
        == 0
    )
    assert [call["dtype"] for call in load_spy] == ["bf16"]
    record = json.loads((out / "protocol.json").read_text())
    assert record["canonical"]["model"]["dtype"] == "bf16"


def test_an_unauthored_document_still_runs_and_records_fp32(roots, tmp_path, load_spy):
    out = tmp_path / "out"
    assert main(_argv("02_interchange_im.json", roots, out, "--record")) == 0
    assert [call["dtype"] for call in load_spy] == ["fp32"]
    record = json.loads((out / "protocol.json").read_text())
    assert record["canonical"]["model"]["dtype"] == "fp32"


def test_the_artifact_stamp_carries_the_realization(roots, tmp_path):
    """A harvested tensor bundle proves which precision produced it."""
    out = tmp_path / "out"
    assert main(_argv_harvest(roots, out, "--dtype", "bf16")) == 0
    stamped = read_safetensors_metadata(out / "acts_L8_ans.safetensors")
    assert stamped["model_dtype"] == "bf16"
    assert "model_quantization" not in stamped  # absent, not stamped as null


def test_a_quantized_document_names_what_it_needs(roots, tmp_path, capsys):
    """Quantization is document vocabulary whether or not the quantizer is
    installed: the document validates and digests, and only ``run`` needs
    bitsandbytes — saying so precisely."""
    try:
        import bitsandbytes  # noqa: F401

        pytest.skip("bitsandbytes is installed; the refusal path needs it absent")
    except ImportError:
        pass
    doc = json.loads((CORPUS_DIR / "02_interchange_im.json").read_text())
    doc["model"]["quantization"] = {"scheme": "nf4"}
    document = tmp_path / "nf4_im.json"
    document.write_text(json.dumps(doc))
    data_root, artifacts_root = roots
    argv = [
        "run",
        "--engine",
        "auto",
        str(document),
        "--data-root",
        str(data_root),
        "--artifacts-root",
        str(artifacts_root),
        "--out",
        str(tmp_path / "out"),
        "--set",
        f"model.key={TINY_LLAMA}",
        "--set",
        "sites.target.layers=1",
    ]
    assert main(argv) == 1
    refusal = capsys.readouterr().err
    assert "bitsandbytes" in refusal
    assert "nf4" in refusal


@pytest.mark.parametrize("engine", ["pytorch_hooks", "nnsight"])
@pytest.mark.parametrize("workflow", [False, True])
def test_json_attention_backend_reaches_loading_and_artifacts(
    roots, tmp_path, monkeypatch, engine, workflow
):
    import importlib
    import torch

    if engine == "nnsight":
        pytest.importorskip("nnsight")
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    module = importlib.import_module(
        "causalab.neural.engines."
        + ("nnsight_tracing" if engine == "nnsight" else engine)
        + ".engine"
    )
    real_load = module.load_model
    selected: list[str] = []

    def observe_load(*args, **kwargs):
        bundle = real_load(*args, **kwargs)
        selected.append(bundle.model.config._attn_implementation)
        return bundle

    monkeypatch.setattr(module, "load_model", observe_load)
    doc = json.loads((CORPUS_DIR / "01_harvest_im.json").read_text())
    doc["model"]["key"] = TINY_LLAMA
    doc["method"]["sites"]["L8"]["layers"] = [0]
    doc["method"]["sites"]["L24"]["layers"] = [1]
    if not workflow:
        doc["model"]["attn_implementation"] = "sdpa"
    path = tmp_path / "harvest.json"
    path.write_text(json.dumps(doc))
    if workflow:
        path = tmp_path / "workflow.json"
        path.write_text(
            json.dumps(
                {
                    "version": "1",
                    "output_dir": "attention",
                    "steps": {
                        "harvest": {
                            "type": "intervention_protocol",
                            "document": "harvest.json",
                            "set": {"model.attn_implementation": "sdpa"},
                        }
                    },
                }
            )
        )
    out = tmp_path / "out"
    assert (
        main(
            [
                "run",
                str(path),
                "--engine",
                engine,
                "--data-root",
                str(roots[0]),
                "--artifacts-root",
                str(roots[1]),
                "--out",
                str(out),
                # a workflow keeps its own records; the receipt is the
                # document run's, and only when asked
                *(() if workflow else ("--record",)),
            ]
        )
        == 0
    )
    if workflow:
        out = out / "attention" / "harvest"
    assert selected == ["sdpa"]
    stamped = read_safetensors_metadata(out / "acts_L8_ans.safetensors")
    assert stamped is not None
    assert stamped["model_attn_implementation"] == "sdpa"
    if not workflow:
        record = json.loads((out / "protocol.json").read_text())
        assert record["canonical"]["model"]["attn_implementation"] == "sdpa"


def test_backend_sweep_stamps_each_tensor_entry(roots, tmp_path, load_spy):
    out = tmp_path / "out"
    assert (
        main(
            _argv_harvest(
                roots,
                out,
                "--set",
                'model.attn_implementation={"sweep":["eager","sdpa"]}',
                "--engine",
                "pytorch_hooks",
            )
        )
        == 0
    )
    assert [call["attn_implementation"] for call in load_spy] == ["eager", "sdpa"]
    stamped = read_safetensors_metadata(out / "acts_L8_ans.safetensors")
    assert stamped is not None
    assert "model_attn_implementation" not in stamped
    entries = json.loads(stamped["entries"])
    assert {entry["model_attn_implementation"] for entry in entries.values()} == {
        "eager",
        "sdpa",
    }


def test_omitted_backend_records_loaded_implementation(roots, tmp_path):
    out = tmp_path / "out"
    assert main(_argv_harvest(roots, out, "--engine", "pytorch_hooks", "--record")) == 0
    record = json.loads((out / "protocol.json").read_text())
    assert "attn_implementation" not in record["canonical"]["model"]
    stamped = read_safetensors_metadata(out / "acts_L8_ans.safetensors")
    assert stamped is not None
    entries = json.loads(stamped["entries"])
    assert all(e["loaded_attn_implementation"] == "eager" for e in entries.values())
    from causalab.io.env import entry_identity
    from causalab.io.step_io import inherited_identity

    identities = [entry_identity(stamped, key) for key in entries]
    assert all(
        identity["loaded_attn_implementation"] == "eager" for identity in identities
    )
    assert inherited_identity(identities)["loaded_attn_implementation"] == "eager"
