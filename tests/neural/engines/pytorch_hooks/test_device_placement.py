"""Layer placement across the devices of one process (``docs/model_parallelism.md``
§5.1), on the one two-device layout this machine can offer: ``cpu,mps``.

The loader puts the embedding and the first block on the CPU, the second
block, the final norm and the head on the MPS device, and the residual
stream crosses between them inside the forward; the executor encodes onto
the embedding's device, and every capture stays where its hook produced it.

**This is not an exactness test.** CPU and MPS kernels round differently, so
the numbers of a ``cpu,mps`` run are held to the all-``cpu`` run within a
loose band and asserted finite. The exact twin — two CUDA ordinals against
one, bit for bit — is ``tests/golden/test_device_placement.py``, which needs
two CUDA devices: a placement moves no reduction, so there the claim is
equality.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
import torch

from causalab.cli import main
from causalab.neural.engines.pytorch_hooks.loading import ModelBundle, load_model
from causalab.neural.shared.devices import DeviceMap
from causalab.neural.shared.encoding import encode
from causalab.protocol.rules.capability import check_caller_bundle
from causalab.protocol import RUN_RECORD_NAME
from causalab.protocol.rules.errors import ProtocolError

from tests.neural.engines.pytorch_hooks.conftest import TINY_LLAMA
from tests.protocol._env import CORPUS_DIR, FIXTURES
from tests.tables import frame as table_frame

pytestmark = [
    pytest.mark.smoke,
    pytest.mark.skipif(
        not torch.backends.mps.is_available(),
        reason="needs an mps device beside the cpu",
    ),
]

CPU = torch.device("cpu")
MPS = torch.device("mps", 0)
TEXTS = ["the quick brown fox jumps", "a slow green turtle sleeps deeply"]
#: CPU against MPS in fp32 on the tiny fixture: a rounding band, not a claim
#: about the computation (module docstring).
BAND = {"rtol": 1e-3, "atol": 1e-3}
REALIZATION: dict[str, Any] = {
    "key": TINY_LLAMA,
    "revision": "main",
    "dtype": "fp32",
    "quantization": None,
}


@pytest.fixture(scope="module")
def placed() -> ModelBundle:
    return load_model(TINY_LLAMA, device="cpu,mps")


@pytest.fixture(scope="module")
def roots(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    from tests.protocol._env import write_rot_fixture

    artifacts = tmp_path_factory.mktemp("artifacts")
    shutil.copytree(FIXTURES / "artifacts", artifacts, dirs_exist_ok=True)
    write_rot_fixture(artifacts)
    return FIXTURES / "data", artifacts


def _run(
    name: str, roots: tuple[Path, Path], out: Path, *overrides: str, device: str
) -> int:
    """Corpus document ``name`` through the real CLI on the tiny Llama."""
    data_root, artifacts_root = roots
    argv = [
        "run",
        str(CORPUS_DIR / name),
        "--data-root",
        str(data_root),
        "--artifacts-root",
        str(artifacts_root),
        "--out",
        str(out),
        "--engine",
        "pytorch_hooks",
        "--record",
        "--device",
        device,
        "--set",
        f"model.key={TINY_LLAMA}",
        "--set",
        "model.dtype=fp32",
    ]
    for item in overrides:
        argv += ["--set", item]
    return main(argv)


def _devices_of(module: torch.nn.Module) -> set[torch.device]:
    return {p.device for p in module.parameters()} | {
        b.device for b in module.buffers()
    }


def test_every_module_sits_where_the_map_says(placed: ModelBundle) -> None:
    devices = placed.devices
    assert devices == DeviceMap.parse("cpu,mps", 2)
    assert devices.requested == "cpu,mps"
    assert devices.single is None
    for layer, block in enumerate(placed.blocks):
        assert _devices_of(block) == {devices.blocks[layer]}, layer
    assert _devices_of(placed.model.model.embed_tokens) == {CPU}
    assert _devices_of(placed.model.model.norm) == {MPS}
    assert _devices_of(placed.model.lm_head) == {MPS}
    # literal placement: nothing was offloaded to `meta`, and no accelerate
    # hook survived to execute a CPU block on the accelerator
    assert not any(p.device.type == "meta" for p in placed.model.parameters())
    assert not any(hasattr(m, "_hf_hook") for m in placed.model.modules())
    assert not placed.model.training
    assert not any(p.requires_grad for p in placed.model.parameters())


def test_the_placed_weights_are_the_cpu_weights(
    placed: ModelBundle, llama_bundle: ModelBundle
) -> None:
    """Placement moves bytes, never changes them."""
    theirs = llama_bundle.model.state_dict()
    ours = placed.model.state_dict()
    assert set(theirs) == set(ours)
    for name in theirs:
        assert torch.equal(ours[name].cpu(), theirs[name]), name


def test_a_forward_crosses_devices_and_each_capture_lands_where_it_was_produced(
    placed: ModelBundle, llama_bundle: ModelBundle
) -> None:
    produced: dict[int, torch.device] = {}

    def capture(layer: int):
        def hook(_module: Any, _args: Any, output: Any) -> None:
            tensor = output[0] if isinstance(output, tuple) else output
            produced[layer] = tensor.device

        return hook

    handles = [
        block.register_forward_hook(capture(layer))
        for layer, block in enumerate(placed.blocks)
    ]
    try:
        batch = encode(placed.tokenizer, TEXTS, device=placed.devices.embedding)
        assert batch.input_ids.device == CPU
        with torch.no_grad():
            out = placed.model(
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask,
                position_ids=batch.position_ids(),
            )
    finally:
        for handle in handles:
            handle.remove()
    assert produced == {0: CPU, 1: MPS}
    assert out.logits.device == MPS
    reference = encode(llama_bundle.tokenizer, TEXTS, device=CPU)
    with torch.no_grad():
        expected = llama_bundle.model(
            input_ids=reference.input_ids,
            attention_mask=reference.attention_mask,
            position_ids=reference.position_ids(),
        ).logits
    assert torch.isfinite(out.logits).all()
    torch.testing.assert_close(out.logits.cpu(), expected, **BAND)


def test_the_interchange_document_runs_through_the_cli_across_cpu_and_mps(
    roots: tuple[Path, Path], tmp_path: Path
) -> None:
    """Corpus 02 end to end under ``--device cpu,mps``: the counterfactual
    read at block 1 (MPS) swapped into the base run, scored at the head
    (MPS); its tables are finite and within the band of the all-CPU run,
    and the two receipts' ``execution`` blocks differ only in the recorded
    ``device``: placement is execution, so every digest is shared."""
    on_cpu, spanning = tmp_path / "cpu", tmp_path / "placed"
    assert (
        _run(
            "02_interchange_im.json",
            roots,
            on_cpu,
            "sites.target.layers=1",
            device="cpu",
        )
        == 0
    )
    assert (
        _run(
            "02_interchange_im.json",
            roots,
            spanning,
            "sites.target.layers=1",
            device="cpu,mps",
        )
        == 0
    )
    iia_cpu = table_frame(on_cpu / "iia.json")
    iia_placed = table_frame(spanning / "iia.json")
    assert len(iia_placed) == 2
    assert set(iia_placed["value"]).issubset({0.0, 1.0})
    assert list(iia_placed["value"]) == list(iia_cpu["value"])
    ld_cpu = table_frame(on_cpu / "logit_diff.json")
    ld_placed = table_frame(spanning / "logit_diff.json")
    placed_values = torch.tensor(list(ld_placed["value"]), dtype=torch.float64)
    assert torch.isfinite(placed_values).all()
    torch.testing.assert_close(
        placed_values, torch.tensor(list(ld_cpu["value"]), dtype=torch.float64), **BAND
    )
    receipt_cpu = json.loads((on_cpu / RUN_RECORD_NAME).read_text())
    receipt_placed = json.loads((spanning / RUN_RECORD_NAME).read_text())
    assert receipt_cpu["execution"].pop("device") == "cpu"
    assert receipt_placed["execution"].pop("device") == "cpu,mps"
    assert receipt_placed["execution"] == receipt_cpu["execution"]
    assert receipt_placed["document_digest"] == receipt_cpu["document_digest"]
    # the same weights on either placement: one resolved commit
    assert receipt_placed["models"] == receipt_cpu["models"]


def test_the_harvest_document_saves_each_layers_read_from_its_own_device(
    roots: tuple[Path, Path], tmp_path: Path
) -> None:
    """Corpus 01 reads block 0 (CPU) and block 1 (MPS) in one forward; both
    files are written, in the shapes the CPU run writes."""
    from safetensors.torch import load_file

    code = _run(
        "01_harvest_im.json",
        roots,
        tmp_path,
        "sites.L8.layers=0",
        "sites.L24.layers=1",
        device="cpu,mps",
    )
    assert code == 0
    low = load_file(str(tmp_path / "acts_L8_ans.safetensors"))
    high = load_file(str(tmp_path / "acts_L24_ans.safetensors"))
    assert low["acts_L8_ans"].shape == high["acts_L24_ans"].shape == (2, 1, 16)
    assert torch.isfinite(low["acts_L8_ans"]).all()
    assert torch.isfinite(high["acts_L24_ans"]).all()


def _hand_placed_llama() -> tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        TINY_LLAMA, dtype=torch.float32, attn_implementation="eager"
    )
    model.eval()
    model.requires_grad_(False)
    model.model.layers[1].to(MPS)
    model.model.norm.to(MPS)
    model.lm_head.to(MPS)
    tokenizer = AutoTokenizer.from_pretrained(TINY_LLAMA)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def test_from_model_derives_the_map_from_where_the_caller_put_the_weights() -> None:
    """A caller-owned model already spread over the two devices: the bundle
    says where every module is, and the engine's ``--device`` is checked
    against that by value."""
    model, tokenizer = _hand_placed_llama()
    bundle = ModelBundle.from_model(
        model, tokenizer, key=TINY_LLAMA, revision="main", dtype="fp32"
    )
    assert bundle.devices == DeviceMap.parse("cpu,mps", 2)
    assert bundle.devices.requested == "cpu,mps:0"
    check_caller_bundle(bundle, REALIZATION, device="cpu,mps")
    with pytest.raises(ProtocolError, match="device") as err:
        check_caller_bundle(bundle, REALIZATION, device="cpu")
    assert "'cpu'" in str(err.value) and "'cpu,mps:0'" in str(err.value)


def test_from_model_refuses_a_block_straddling_the_two_devices() -> None:
    model, tokenizer = _hand_placed_llama()
    model.model.layers[1].mlp.to(CPU)
    with pytest.raises(ProtocolError, match="block 1") as err:
        ModelBundle.from_model(
            model, tokenizer, key=TINY_LLAMA, revision="main", dtype="fp32"
        )
    assert err.value.code == "P4"
