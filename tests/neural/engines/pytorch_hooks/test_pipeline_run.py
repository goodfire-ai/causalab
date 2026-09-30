"""Pipeline stages on real processes over ``gloo`` (``docs/model_parallelism.md``
§6.5, §8.3, §10.6): the corpus interchange document on the tiny Llama at
``pp=2`` and on the tiny MoE at ``pp=4``, each rank loading its stage
through `load_model` with a [`Sharding`][causalab.neural.engines.pytorch_hooks.sharding.Sharding] and running the reference
engine end to end (``run_protocol``), against the world-1 run of the same
document.

The assertion is **byte identity** (§8.3: pipeline parallelism is placement,
no reduction moves): every table and every safetensors file equal to the
world-1 run's bytes on every rank, and the receipt equal minus its
``execution.parallel`` block — the ``fires`` block included. A write on
layer 0 with a read on layer 1, and the reverse, so each stage owns a write
in one of the two documents; on the MoE the write, the operand read and the
probe read sit on three different stages. And the expert-neuron preset's
swap through the expert-keyed gate on the MoE's layer 3 at ``pp=2`` — the
write on the stage that does not publish — whose ``routing_mismatch.json``,
a record made in the owner's hook and shared over the pipeline
(``parallel/mismatch.py``), is world 1's on every stage.

**Mutation.** Sending the residual one block early — the hidden state
*entering* the stage's last block instead of leaving it — moves the head's
input, and the scored tables land outside equality.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
import torch

from causalab.cli import register_model_key
from causalab.neural.engines.pytorch_hooks.engine import PytorchHooksEngine
from causalab.protocol.rules.errors import ProtocolError
from causalab.protocol.parallel import ParallelGeometry
from causalab.protocol.pipeline import run_protocol
from causalab.protocol.schema import inline_train_saves
from tests._helpers.paths import PROTOCOLS_DIR
from tests._helpers.sharded_world import WorldError, run_world
from tests.protocol._env import CORPUS_DIR, FIXTURES, build_env

from .conftest import TINY_LLAMA, TINY_QWEN35_MOE

# `parallel_world`: every test here runs a document or a fit across a spawned
# multi-rank process world — minutes on the CI runner. The PR gate deselects the
# marker; the nightly CPU job runs it (docs/TESTS.md).
pytestmark = [pytest.mark.smoke, pytest.mark.parallel_world]

INTERCHANGE = CORPUS_DIR / "02_interchange_im.json"
RECEIPT = "protocol.json"


def _document(tmp: Path, key: str, *, write_layer: int, read_layer: int) -> Path:
    """Corpus 02 retargeted to ``key``: the swap at ``write_layer``, and the
    patched residual at ``read_layer`` saved as a tensor file so the stage
    that owns *that* block is held to bytes too."""
    doc = json.loads(INTERCHANGE.read_text())
    doc["model"] = {"key": key, "revision": "main", "dtype": "fp32"}
    doc["method"]["sites"]["target"]["layers"] = [write_layer]
    doc["method"]["sites"]["probe_site"] = {
        "component": "block_output",
        "layers": [read_layer],
    }
    doc["method"]["reads"]["probe"] = {"site": "probe_site", "pos": -1}
    doc["method"]["intervened_models"]["patched"]["reads"].append("probe")
    doc["method"]["save"].append(
        {"read": "probe", "model": "patched", "file_path": "probe.safetensors"}
    )
    target = tmp / f"interchange_w{write_layer}_r{read_layer}.json"
    target.write_text(json.dumps(doc, indent=2))
    return target


def _artifacts(tmp: Path) -> Path:
    root = tmp / "artifacts"
    shutil.copytree(FIXTURES / "artifacts", root, dirs_exist_ok=True)
    return root


def _solo(document: Path, artifacts: Path, out: Path) -> Path:
    register_model_key(json.loads(document.read_text()))
    run_protocol(
        document,
        build_env(artifacts),
        PytorchHooksEngine(device="cpu"),
        out,
        record=True,
    )
    return out


def _program(rank: int, world: int, payload: Any) -> str:
    """One stage: join the mesh, load this rank's stage, run the document
    through the engine into this rank's own output directory."""
    from causalab.neural.engines.pytorch_hooks import stages as stages_module
    from causalab.neural.engines.pytorch_hooks.sharding import Sharding
    from causalab.neural.shared.parallel.collective import TorchCollective
    from causalab.neural.shared.parallel.mesh import Mesh

    document, artifacts, out_root, geometry_kwargs, mutate = payload
    register_model_key(json.loads(Path(document).read_text()))
    geometry = ParallelGeometry(**geometry_kwargs)
    if mutate:
        stages_module.stage_hidden = _one_block_early
    collective = TorchCollective(Mesh(geometry, rank))
    engine = PytorchHooksEngine(
        device="cpu",
        parallel=geometry,
        collective=collective,
        sharding=Sharding(geometry=geometry, rank=rank, meshes={}),
    )
    out = Path(out_root) / f"rank{rank}"
    run_protocol(Path(document), build_env(Path(artifacts)), engine, out, record=True)
    return str(out)


def _one_block_early(model: Any, kwargs: dict[str, Any]) -> torch.Tensor:
    """The mutation: the residual *entering* this stage's last held block
    instead of the one leaving it."""
    from causalab.neural.engines.pytorch_hooks.sharding import StageStandIn
    from causalab.protocol.registry import family_for

    blocks = family_for(model).blocks_of(model)
    last = max(i for i, b in enumerate(blocks) if not isinstance(b, StageStandIn))
    seen: list[torch.Tensor] = []

    def keep(_m: Any, args: tuple[Any, ...], kw: dict[str, Any]) -> None:
        seen.append(args[0] if args else kw["hidden_states"])

    handle = blocks[last].register_forward_pre_hook(keep, with_kwargs=True)
    try:
        getattr(model, model.base_model_prefix)(**kwargs)
    finally:
        handle.remove()
    return seen[0]


def _receipt(out: Path) -> dict[str, Any]:
    return json.loads((out / RECEIPT).read_text())


def _assert_byte_identical(solo: Path, staged: Path, geometry: dict[str, int]) -> None:
    assert sorted(p.name for p in solo.iterdir()) == sorted(
        p.name for p in staged.iterdir()
    )
    # every output but the receipt (compared below minus its parallel block)
    # and the event stream (timestamps): the tables, the tensor files, the
    # side tables such as routing_mismatch.json
    for path in solo.iterdir():
        if path.name in (RECEIPT, "events.jsonl") or not path.is_file():
            continue
        assert (staged / path.name).read_bytes() == path.read_bytes(), path.name
    a, b = _receipt(solo), _receipt(staged)
    world = 1
    for size in geometry.values():
        world *= size
    assert b["execution"]["parallel"] == {
        "data": 1,
        "data_mode": "points",
        "pipeline": 1,
        "context": 1,
        "tensor": 1,
        "expert": 1,
        **geometry,
        "world": world,
        "launcher": "solo",
    }
    assert "fires" in a and a["fires"] == b["fires"]
    del a["execution"]["parallel"], b["execution"]["parallel"]
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def _staged_run(
    document: Path,
    artifacts: Path,
    out_root: Path,
    *,
    mutate: bool = False,
    **geometry: int,
) -> dict[int, Path]:
    world = ParallelGeometry(**geometry).world
    ranks = run_world(
        world,
        _program,
        (str(document), str(artifacts), str(out_root), geometry, mutate),
    )
    return {rank: Path(path) for rank, path in ranks.items()}


@pytest.mark.parametrize(("write_layer", "read_layer"), [(0, 1), (1, 0)])
def test_pp2_llama_is_byte_identical_to_world_one_on_every_stage(
    tmp_path: Path, write_layer: int, read_layer: int
) -> None:
    artifacts = _artifacts(tmp_path)
    document = _document(
        tmp_path, TINY_LLAMA, write_layer=write_layer, read_layer=read_layer
    )
    solo = _solo(document, artifacts, tmp_path / "solo")
    ranks = _staged_run(document, artifacts, tmp_path / "pp2", pipeline=2)
    assert set(ranks) == {0, 1}
    for out in ranks.values():
        _assert_byte_identical(solo, out, {"pipeline": 2})
    receipt = _receipt(solo)
    assert receipt["fires"] and any(
        "patch" in fires
        for point in receipt["fires"].values()
        for fires in point.values()
    )


def test_pp4_moe_is_byte_identical_to_world_one_on_every_stage(tmp_path: Path) -> None:
    """Four stages, one layer each: the operand read on stage 1, the write
    on stage 1, the probe on stage 3, the head on stage 3, the embedding
    on stage 0 — every stage holds a module the document touches."""
    artifacts = _artifacts(tmp_path)
    document = _document(tmp_path, TINY_QWEN35_MOE, write_layer=1, read_layer=3)
    solo = _solo(document, artifacts, tmp_path / "solo")
    ranks = _staged_run(document, artifacts, tmp_path / "pp4", pipeline=4)
    assert set(ranks) == {0, 1, 2, 3}
    for out in ranks.values():
        _assert_byte_identical(solo, out, {"pipeline": 4})


def test_sending_the_residual_one_block_early_lands_outside_equality(
    tmp_path: Path,
) -> None:
    artifacts = _artifacts(tmp_path)
    document = _document(tmp_path, TINY_LLAMA, write_layer=0, read_layer=1)
    solo = _solo(document, artifacts, tmp_path / "solo")
    ranks = _staged_run(
        document, artifacts, tmp_path / "early", pipeline=2, mutate=True
    )
    for out in ranks.values():
        assert (out / "logit_diff.json").read_bytes() != (
            solo / "logit_diff.json"
        ).read_bytes()


def test_a_decoding_document_is_refused_by_name_under_a_pipeline(
    tmp_path: Path,
) -> None:
    """Decode under ``pp > 1`` is a named limit of this cut: the refusal
    comes from every rank, naming ``--parallel.pipeline``."""
    artifacts = _artifacts(tmp_path)
    doc = json.loads(
        _document(tmp_path, TINY_LLAMA, write_layer=0, read_layer=1).read_text()
    )
    doc["method"]["reads"]["logits"]["pos"] = {
        "generated": {"max_new_tokens": 2},
        "index": -1,
    }
    document = tmp_path / "decode.json"
    document.write_text(json.dumps(doc, indent=2))
    with pytest.raises(WorldError) as err:
        _staged_run(document, artifacts, tmp_path / "pp2", pipeline=2)
    assert "--parallel.pipeline" in str(err.value)
    # the same document runs at world 1
    try:
        _solo(document, artifacts, tmp_path / "solo")
    except ProtocolError as error:  # a document fact, not the pipeline's
        pytest.fail(f"the decoding document does not run at world 1: {error}")


EXPERT_NEURON = PROTOCOLS_DIR / "dbm_expert_neuron.json"


def _expert_write_document(tmp: Path, *, layer: int) -> Path:
    """The expert-neuron preset's swap through the expert-keyed gate at
    ``layer`` of the tiny MoE, as an inference document (the gates at their
    seed, no ``train``): every write through group ``expert_neuron``
    records its routing mismatch, which is written as
    ``routing_mismatch.json`` beside the tables."""
    doc = json.loads(EXPERT_NEURON.read_text())
    interchange = json.loads(INTERCHANGE.read_text())
    doc["model"] = {"key": TINY_QWEN35_MOE, "revision": "main", "dtype": "fp32"}
    doc["data"] = interchange["data"]
    method = doc["method"]
    for name in ("routed", "shared"):
        method["sites"][name]["layers"] = [layer]
    # the tables alone: the gate values are fitted outputs, and there is no fit;
    # the preset names its tables by training metric, so spell them inline first
    method["save"] = [entry for entry in inline_train_saves(method) if "read" in entry]
    del method["train"]
    target = tmp / f"expert_write_l{layer}.json"
    target.write_text(json.dumps(doc, indent=2))
    return target


def test_pp2_moe_expert_keyed_write_on_stage_one_records_its_routing_mismatch(
    tmp_path: Path,
) -> None:
    """📐 The record an expert-keyed write makes in its hook — on the owning
    stage alone — is shared over the pipeline where the fire tally is
    agreed (``parallel/mismatch.py``, §6.5), so with the write on layer 3
    of four (stage 1, not the publisher) every stage writes
    ``routing_mismatch.json`` byte-identical to world 1's, beside every
    other output; before the share the publisher wrote no such file."""
    artifacts = _artifacts(tmp_path)
    document = _expert_write_document(tmp_path, layer=3)
    solo = _solo(document, artifacts, tmp_path / "solo")
    records = json.loads((solo / "routing_mismatch.json").read_text())
    assert records and all(
        r["write"] == "mask_routed" and r["layer"] == 3 for r in records
    )
    ranks = _staged_run(document, artifacts, tmp_path / "pp2", pipeline=2)
    assert set(ranks) == {0, 1}
    for out in ranks.values():
        _assert_byte_identical(solo, out, {"pipeline": 2})
        assert (out / "routing_mismatch.json").read_bytes() == (
            solo / "routing_mismatch.json"
        ).read_bytes()
