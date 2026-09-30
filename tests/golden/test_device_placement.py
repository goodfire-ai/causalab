"""Layer placement across two CUDA ordinals is the one-ordinal run, bit for bit.

The exact twin of ``tests/neural/engines/pytorch_hooks/test_device_placement.py``
(``docs/model_parallelism.md`` §5.1, §8.1): a placement moves activations
between devices and reorders no reduction, so a document run under
``--device cuda:0,cuda:1`` writes the same tables and the same tensors as the
run under ``--device cuda:0``. Needs two CUDA devices; skipped anywhere with
fewer.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from causalab.cli import main
from causalab.neural.engines.pytorch_hooks.loading import load_model
from causalab.neural.shared.devices import DeviceMap
from causalab.protocol import RUN_RECORD_NAME

from tests.neural.engines.pytorch_hooks.conftest import TINY_LLAMA
from tests.protocol._env import CORPUS_DIR, FIXTURES
from tests.tables import frame as table_frame

pytestmark = [
    pytest.mark.golden,
    pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two CUDA devices"),
]

ONE = "cuda:0"
TWO = "cuda:0,cuda:1"


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


def test_the_placed_load_holds_the_same_bytes_on_the_devices_the_map_says() -> None:
    one = load_model(TINY_LLAMA, device=ONE)
    two = load_model(TINY_LLAMA, device=TWO)
    assert two.devices == DeviceMap.parse(TWO, 2)
    for layer, block in enumerate(two.blocks):
        assert {p.device for p in block.parameters()} == {two.devices.blocks[layer]}
    assert {p.device for p in two.model.lm_head.parameters()} == {
        torch.device("cuda", 1)
    }
    theirs, ours = one.model.state_dict(), two.model.state_dict()
    assert set(theirs) == set(ours)
    for name in theirs:
        assert torch.equal(ours[name].to("cuda:0"), theirs[name]), name


def test_the_interchange_document_is_bit_identical_across_the_two_layouts(
    roots: tuple[Path, Path], tmp_path: Path
) -> None:
    one, two = tmp_path / "one", tmp_path / "two"
    for out, device in ((one, ONE), (two, TWO)):
        code = _run(
            "02_interchange_im.json",
            roots,
            out,
            "sites.target.layers=1",
            device=device,
        )
        assert code == 0
    for table in ("iia.json", "logit_diff.json"):
        assert table_frame(one / table).equals(table_frame(two / table)), table
    receipts = [json.loads((out / RUN_RECORD_NAME).read_text()) for out in (one, two)]
    # placement is recorded and is the one execution field the layouts differ in
    assert [receipt["execution"].pop("device") for receipt in receipts] == [ONE, TWO]
    assert receipts[0]["execution"] == receipts[1]["execution"]
    assert receipts[0]["document_digest"] == receipts[1]["document_digest"]
    assert receipts[0]["models"] == receipts[1]["models"]


def test_the_harvest_document_saves_identical_tensors_from_both_devices(
    roots: tuple[Path, Path], tmp_path: Path
) -> None:
    """Block 0 sits on ``cuda:0`` and block 1 on ``cuda:1``; each read is
    saved from where it was produced and equals the one-device tensor."""
    one, two = tmp_path / "one", tmp_path / "two"
    for out, device in ((one, ONE), (two, TWO)):
        code = _run(
            "01_harvest_im.json",
            roots,
            out,
            "sites.L8.layers=0",
            "sites.L24.layers=1",
            device=device,
        )
        assert code == 0
    for name in ("acts_L8_ans", "acts_L24_ans", "acts_L8_ent"):
        theirs = load_file(str(one / f"{name}.safetensors"))
        ours = load_file(str(two / f"{name}.safetensors"))
        assert set(theirs) == set(ours), name
        for key in theirs:
            assert torch.equal(theirs[key], ours[key]), (name, key)
