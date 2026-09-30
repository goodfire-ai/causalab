"""The memory pre-flight at the loader's seam (``weights.load_planned``;
``docs/model_parallelism.md`` §2, §5.3, §11): the torch-free placement
table agrees with the loader's own accounting, and the check runs before a
read is issued.

``unit`` (CPU, meta models): the tower rule's wanted keys are exactly the
keys transformers' renamer consumes on the three tiny fixtures — the
multimodal MoE's ``model.language_model`` tower renamed, nothing else
wanted — so ``dry-run`` and the loader count the same tensors.

``smoke`` (gloo, spawned worlds): on the tiny Llama and the tiny MoE under
``tp=2``, ``ep=2``, ``ep=4``, ``pp=2`` and ``tp=2,ep=4``, every rank's
torch-free estimate equals its ``LoadReport.requested_total`` to the byte
(per on-disk dtype, since the fixture mixes bf16 and fp32); and a rank
whose device reads as full is refused ``P4`` on ``--parallel`` by
``load_model`` itself, naming the rank, before any weight is read.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from causalab.neural.engines.pytorch_hooks.weights import renamed_keys
from causalab.protocol.checkpoint_census import (
    cached_checkpoint_files,
    checkpoint_targets,
    read_headers,
    tree_of,
)
from causalab.protocol.rules.errors import ProtocolError
from causalab.protocol.parallel import ParallelGeometry
from causalab.protocol.parallel_memory import estimate_resident, placement_table
from causalab.protocol.registry import (
    LLAMA_TREE,
    family_for,
    get_model_info,
    model_info_from_hf_config,
)

from tests._helpers.sharded_world import run_world
from tests.neural.engines.pytorch_hooks.conftest import (
    TINY_GPT2,
    TINY_LLAMA,
    TINY_QWEN35_MOE,
)

FIXTURES = (TINY_LLAMA, TINY_GPT2, TINY_QWEN35_MOE)


def _meta(key: str) -> Any:
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(key)
    with torch.device("meta"):
        return AutoModelForCausalLM.from_config(config), config


def _headers(key: str) -> dict[str, Any]:
    files = cached_checkpoint_files(key)
    assert files is not None, f"{key} is not in the local Hub cache"
    return read_headers(files)


@pytest.mark.unit
@pytest.mark.parametrize("key", FIXTURES)
def test_the_tower_rule_wants_exactly_the_keys_the_renamer_consumes(key: str) -> None:
    model, config = _meta(key)
    headers = _headers(key)
    info = model_info_from_hf_config(key, config)
    tree = tree_of(headers, info.num_layers)
    assert tree == family_for(model).tree
    targets = checkpoint_targets(headers, tree, info.num_layers)
    assert targets is not None
    consumed = renamed_keys(model, headers)
    assert set(targets) == set(consumed), key
    # every key a converter does not fuse lands on the same parameter name
    parameters = set(model.state_dict())
    for source, target in targets.items():
        if consumed[source] in parameters and ".experts." not in source:
            assert target == consumed[source], source


@pytest.mark.unit
def test_the_multimodal_fixture_renames_its_text_tower_and_drops_the_rest() -> None:
    headers = _headers(TINY_QWEN35_MOE)
    assert any(key.startswith("model.language_model.") for key in headers)
    info = get_model_info(TINY_QWEN35_MOE)
    targets = checkpoint_targets(headers, LLAMA_TREE.tree, info.num_layers)
    assert targets is not None
    assert all(target.startswith(("model.", "lm_head")) for target in targets.values())
    assert not any(".language_model." in target for target in targets.values())
    assert len(targets) < len(headers) or all(
        key.startswith(("model.language_model.", "lm_head")) for key in headers
    )


# --------------------------------------------------------------------------- #
# gloo: the estimate against the loader's report, and the refusal at the seam
# --------------------------------------------------------------------------- #


def _torch_free_estimate(key: str, geometry: ParallelGeometry, rank: int) -> int:
    """The rank's bytes off the headers alone, per on-disk dtype."""
    headers = _headers(key)
    files = cached_checkpoint_files(key)
    assert files is not None
    from transformers import AutoConfig

    info = model_info_from_hf_config(key, AutoConfig.from_pretrained(key))
    assert info.parallel_plan is not None
    tree = tree_of(headers, info.num_layers)
    assert tree is not None
    targets = checkpoint_targets(headers, tree, info.num_layers)
    assert targets is not None
    total = 0
    for itemsize in sorted({h.itemsize for h in headers.values()}):
        keys = {k: v for k, v in targets.items() if headers[k].itemsize == itemsize}
        if not keys:
            continue
        table = placement_table(
            {k: headers[k].elements for k in keys}, keys, info.parallel_plan, tree, info
        )
        # a per-dtype slice of the table: the stage split is the model's
        total += estimate_resident(
            table, geometry, itemsize, num_layers=info.num_layers
        )[rank]
    return total


def _program(rank: int, world: int, payload: Any) -> dict[str, Any]:
    from causalab.neural.engines.pytorch_hooks.loading import load_model
    from causalab.neural.engines.pytorch_hooks.sharding import Sharding
    from causalab.neural.shared.parallel import memory as memory_module
    from causalab.neural.shared.parallel.mesh import Mesh

    key, geometry_kwargs, reading = payload
    geometry = ParallelGeometry(**geometry_kwargs)
    sharding = Sharding.from_mesh(Mesh.from_environment(geometry))
    if reading is not None:
        # this rank's device reads as ``reading`` says — full (``0`` free:
        # the loader must refuse by name before a read, wherever the model
        # would have gone) or plentiful (the check runs for real, on the
        # CPU, and must pass): off CUDA the query answers ``None`` and the
        # check is the identity, so a reading is what exercises it here
        free = reading
        memory_module.device_memory = lambda device: memory_module.DeviceMemory(  # type: ignore[assignment]
            free=free, total=1 << 40
        )
        import causalab.neural.engines.pytorch_hooks.weights as weights

        weights.preflight.__globals__["device_memory"] = memory_module.device_memory
        try:
            bundle = load_model(key, sharding=sharding)
        except ProtocolError as err:
            return {"refused": str(err), "code": err.code, "path": err.path}
        assert bundle.load_report is not None
        return {"refused": None, "requested": bundle.load_report.requested_total}
    bundle = load_model(key, sharding=sharding)
    assert bundle.load_report is not None
    return {"requested": bundle.load_report.requested_total}


def _world(
    key: str, reading: int | None = None, **geometry: int
) -> dict[int, dict[str, Any]]:
    world = ParallelGeometry(**geometry).world
    return run_world(world, _program, (key, geometry, reading))


# `parallel_world` is on the gloo tests alone, each by its own decorator: they
# run a load across a spawned multi-rank world, minutes on the CI runner, so
# the PR gate deselects the marker and the nightly CPU job runs it
# (docs/TESTS.md). The `unit` tests — meta models and header reads, no world,
# no cost — carry nothing and stay on the gate.
@pytest.mark.smoke
@pytest.mark.parallel_world
@pytest.mark.parametrize(
    ("key", "geometry"),
    [
        (TINY_LLAMA, {"tensor": 2}),
        (TINY_LLAMA, {"pipeline": 2}),
        (TINY_QWEN35_MOE, {"expert": 2}),
        (TINY_QWEN35_MOE, {"expert": 4}),
        (TINY_QWEN35_MOE, {"tensor": 2}),
        (TINY_QWEN35_MOE, {"pipeline": 2}),
        (TINY_QWEN35_MOE, {"tensor": 2, "expert": 4}),
    ],
    ids=lambda value: (
        ",".join(f"{k[:2]}={v}" for k, v in value.items())
        if isinstance(value, dict)
        else value.rsplit("/", 1)[-1]
    ),
)
def test_every_ranks_estimate_is_the_loaders_bytes_requested(key, geometry) -> None:
    ranks = _world(key, **geometry)
    parsed = ParallelGeometry(**geometry)
    assert set(ranks) == set(range(parsed.world))
    for rank, out in ranks.items():
        assert out["requested"] == _torch_free_estimate(key, parsed, rank), (
            rank,
            geometry,
        )


@pytest.mark.smoke
@pytest.mark.parallel_world
@pytest.mark.parametrize(
    ("key", "geometry"),
    [(TINY_LLAMA, {"pipeline": 2}), (TINY_QWEN35_MOE, {"pipeline": 4})],
    ids=["llama-pp2", "moe-pp4"],
)
def test_a_pipeline_stage_passes_a_plentiful_device_and_reads_its_bytes(
    key, geometry
) -> None:
    """Pre-flight must read the full model height, not a stage's slice.
    With plentiful memory, every stage passes and reads exactly its
    torch-free estimate."""
    ranks = _world(key, reading=1 << 40, **geometry)
    parsed = ParallelGeometry(**geometry)
    assert set(ranks) == set(range(parsed.world))
    for rank, out in ranks.items():
        assert out["refused"] is None, (rank, out)
        assert out["requested"] == _torch_free_estimate(key, parsed, rank), rank


@pytest.mark.smoke
@pytest.mark.parallel_world
def test_a_full_device_is_refused_by_the_loader_before_any_read() -> None:
    ranks = _world(TINY_QWEN35_MOE, reading=0, expert=2)
    for rank, out in ranks.items():
        assert out["code"] == "P4" and out["path"] == "--parallel", out
        assert f"rank {rank} (" in out["refused"]
        assert "Refused before any weight is read" in out["refused"]


@pytest.mark.smoke
@pytest.mark.parallel_world
def test_weights_that_fit_are_loaded_even_when_estimated_headroom_does_not() -> None:
    geometry = ParallelGeometry(tensor=2)
    resident = max(_torch_free_estimate(TINY_LLAMA, geometry, rank) for rank in (0, 1))
    ranks = _world(TINY_LLAMA, reading=resident, tensor=2)
    for out in ranks.values():
        assert out["refused"] is None
        assert out["requested"] == resident
