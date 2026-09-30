"""Loading a model over sub-meshes with the registry's plan applied and the
weights read shard by shard (``docs/model_parallelism.md`` §5.2, §5.3), on
real processes over ``gloo`` (§10.6).

Every scenario spawns a world, loads the tiny fixture on each rank through
`load_model` with a [`Sharding`][causalab.neural.engines.pytorch_hooks.sharding.Sharding], and hands back what the parent
compares against the one-process load of the same fixture:

* **tp=2 on the tiny Llama** — each rank's ``q_proj.weight.to_local()`` is
  the matching half of the whole weight; a forward on the same input gives
  logits within the measured band; the reader was asked for exactly half of
  every sharded parameter's bytes and all of every replicated one's.
* **ep=2 and ep=4 on ``tiny-random/qwen3.5-moe``** — each rank holds
  ``128 / ep`` experts (``module.num_experts``), the router remaps the global
  routing table onto local ids with the sentinel elsewhere (every slot owned
  by exactly one rank, and its global id the one-process router's), the
  expert bytes are ``1 / ep``, and the logits agree within the band.
* **tp=2, ep=4** — the mixed model group of §2 (``model=4``, world 4) loads
  and runs within the band.
* **pp=2 on the tiny Llama** — each stage holds its own layers and the
  others are identity; the embedding on the first stage, norm and head on the
  last; the reader was asked for the stage's keys and nothing else. No
  forward: the stage forward is the executor branch's.

**The band.** A sharded forward is the one-process computation up to the
reduction order its collectives introduce — a rowwise projection's
all-reduce over ``tp`` partial sums, a gathered projection's all-gather (no
rounding), the expert outputs' all-reduce over ``ep`` ranks. Measured on the
fixtures in fp32 (2026-09-14, torch 2.9.0, gloo, CPU, this file's own
program): tp=2 Llama ``8.9e-8`` (logits up to 0.38), ep=2 MoE ``4.5e-7``,
ep=4 MoE ``3.9e-7``, tp=2,ep=4 MoE ``4.3e-7``, tp=2 MoE ``4.8e-7`` (logits up
to 1.5) — maximum absolute logit difference, identical on every rank. The
band is pinned at ``1e-5``: twenty times the largest measured maximum —
room for gloo's reduction order to vary with the rank count — and four
orders below the error a wrong shard makes (experts left unsharded under a
remapping router moved the logits by 0.26–0.40 on the same fixtures).
"""

from __future__ import annotations

from typing import Any, cast

import pytest
import torch

from causalab.neural.engines.pytorch_hooks.loading import (
    LOAD_REPORT_VARIABLE,
    load_model,
    load_report_path,
)
from causalab.neural.engines.pytorch_hooks.residency import residency_problems
from causalab.neural.engines.pytorch_hooks.sharding import Sharding
from causalab.neural.shared.parallel.mesh import Mesh
from causalab.protocol.parallel import ParallelGeometry

from tests._helpers.sharded_world import run_world
from tests.neural.engines.pytorch_hooks.conftest import TINY_LLAMA, TINY_QWEN35_MOE

# `parallel_world`: every test here runs a document or a fit across a spawned
# multi-rank process world — minutes on the CI runner. The PR gate deselects the
# marker; the nightly CPU job runs it (docs/TESTS.md).
pytestmark = [pytest.mark.smoke, pytest.mark.parallel_world]

#: The measured maxima the band is pinned against (module docstring).
MEASURED = {
    "llama tp=2": 8.9e-8,
    "moe ep=2": 4.5e-7,
    "moe ep=4": 3.9e-7,
    "moe tp=2,ep=4": 4.3e-7,
    "moe tp=2": 4.8e-7,
}
BAND = 1e-5
assert BAND >= 20 * max(MEASURED.values())

IDS = torch.tensor([[5, 17, 23, 42, 8, 91, 3], [12, 4, 77, 61, 30, 2, 9]])
NUM_EXPERTS = 128


def _hidden(model: Any) -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    return torch.randn(6, model.config.hidden_size, generator=generator)


def _program(rank: int, world: int, payload: Any) -> dict[str, Any]:
    """Load ``key`` under ``geometry`` on this rank; report what the parent
    compares. Tensors leave detached and cloned (they cross a process)."""
    from torch.distributed.tensor import DTensor
    from transformers.distributed.pipeline_parallel import PipelineIdentityLayer

    import json
    import os
    import tempfile
    from pathlib import Path

    key, geometry_kwargs, forward = payload
    geometry = ParallelGeometry(**geometry_kwargs)
    sharding = Sharding.from_mesh(Mesh.from_environment(geometry))
    assert sharding.rank == rank
    # the written report carries the residency census (one copy, §5.3)
    reports = Path(tempfile.mkdtemp(prefix="load-report-"))
    os.environ[LOAD_REPORT_VARIABLE] = str(reports)
    bundle = load_model(key, sharding=sharding)
    model = bundle.model
    assert bundle.load_report is not None
    assert bundle.residency is not None
    out: dict[str, Any] = {
        "report": json.loads(load_report_path(reports, rank).read_text()),
        "geometry": bundle.geometry,
        "devices": bundle.devices.spelling,
        "local": {
            name: parameter.to_local().detach().clone()
            for name, parameter in model.named_parameters()
            if isinstance(parameter, DTensor)
        },
        # per sharded parameter, the sharded dimension of each placement
        # (``None`` for a replicated one)
        "placements": {
            name: tuple(
                cast(Any, p).dim if p.is_shard() else None for p in parameter.placements
            )
            for name, parameter in model.named_parameters()
            if isinstance(parameter, DTensor)
        },
        "parameters": sorted(name for name, _ in model.named_parameters()),
        "requested": dict(bundle.load_report.bytes_requested),
        "on_disk": dict(bundle.load_report.bytes_on_disk),
        "num_experts": {
            name: module.num_experts
            for name, module in model.named_modules()
            if name.endswith("mlp.experts")
        },
        "identity": sorted(
            name
            for name, module in model.named_modules()
            if isinstance(module, PipelineIdentityLayer)
        ),
    }
    gate = dict(model.named_modules()).get("model.layers.0.mlp.gate")
    if gate is not None:
        with torch.no_grad():
            _, scores, indices = gate(_hidden(model))
        out["router"] = (indices.clone(), scores.clone())
    if forward:
        with torch.no_grad():
            out["logits"] = model(input_ids=IDS).logits.detach().clone()
    return out


def _load(key: str, forward: bool = True, **geometry: int) -> dict[int, dict[str, Any]]:
    world = ParallelGeometry(**geometry).world
    return run_world(world, _program, (key, geometry, forward))


@pytest.fixture(scope="module")
def llama() -> Any:
    return load_model(TINY_LLAMA)


@pytest.fixture(scope="module")
def moe() -> Any:
    return load_model(TINY_QWEN35_MOE)


def _reference_logits(bundle: Any) -> torch.Tensor:
    with torch.no_grad():
        return bundle.model(input_ids=IDS).logits


def _assert_shards_are_chunks(
    ranks: dict[int, dict[str, Any]], reference: Any, world: int
) -> None:
    """Every sharded parameter's local tensor on rank ``r`` is chunk ``r`` of
    the whole parameter along the placement's dimension."""
    whole = dict(reference.model.named_parameters())
    for rank, out in ranks.items():
        for name, local in out["local"].items():
            (dim,) = out["placements"][name]
            assert dim is not None, (name, "replicated")
            expected = whole[name].chunk(world, dim=dim)[rank]
            assert torch.equal(local, expected), (rank, name)


def _assert_one_copy(ranks: dict[int, dict[str, Any]]) -> None:
    """Every rank holds each parameter once: its local tensor has exactly
    the elements the plan read, backed by its own storage, the process holds
    no unowned copy, and the rule over the written report is silent. The
    fixtures are stored in mixed precision and loaded in fp32, so the copy is
    held through elements, never bytes."""
    for rank, out in ranks.items():
        record = out["report"]
        assert residency_problems(record) == [], (rank, residency_problems(record))
        for name in record["bytes_requested"]:
            assert (
                record["elements_resident"][name] == record["elements_requested"][name]
            )
            own = record["elements_resident"][name] * record["itemsize_resident"][name]
            assert record["bytes_resident"][name] == own, name
        assert record["bytes_unowned"] is not None
        assert record["device_bytes_allocated"] is None  # gloo: no CUDA counters


def _assert_bytes(ranks: dict[int, dict[str, Any]], world: int) -> None:
    """Sharded parameters were read at ``1 / world`` of their bytes on every
    rank, replicated ones whole."""
    for out in ranks.values():
        for name, on_disk in out["on_disk"].items():
            requested = out["requested"][name]
            if name in out["local"]:
                assert requested * world == on_disk, (name, requested, on_disk)
            else:
                assert requested == on_disk, (name, requested, on_disk)


# --------------------------------------------------------------------------- #
# tensor parallelism on the dense fixture
# --------------------------------------------------------------------------- #


def test_tp2_llama_holds_half_of_every_sharded_weight_and_agrees_within_the_band(
    llama: Any,
) -> None:
    ranks = _load(TINY_LLAMA, tensor=2)
    assert set(ranks) == {0, 1}
    for out in ranks.values():
        assert out["geometry"] == ParallelGeometry(tensor=2)
        assert out["devices"] == "cpu"
        assert set(out["local"]) == {
            f"model.layers.{i}.{name}"
            for i in range(2)
            for name in (
                "self_attn.q_proj.weight",
                "self_attn.k_proj.weight",
                "self_attn.v_proj.weight",
                "self_attn.o_proj.weight",
                "mlp.gate_proj.weight",
                "mlp.up_proj.weight",
                "mlp.down_proj.weight",
            )
        }
        assert out["placements"]["model.layers.0.self_attn.q_proj.weight"] == (0,)
        assert out["placements"]["model.layers.0.self_attn.o_proj.weight"] == (1,)
    _assert_shards_are_chunks(ranks, llama, 2)
    reference = _reference_logits(llama)
    for rank, out in ranks.items():
        diff = (out["logits"] - reference).abs().max().item()
        assert diff <= BAND, (rank, diff)
        assert torch.isfinite(out["logits"]).all()
    # every rank computes the same replicated logits
    assert torch.equal(ranks[0]["logits"], ranks[1]["logits"])


def test_tp2_llama_reads_half_the_bytes_of_every_sharded_parameter(llama: Any) -> None:
    ranks = _load(TINY_LLAMA, forward=False, tensor=2)
    _assert_bytes(ranks, 2)
    _assert_one_copy(ranks)
    out = ranks[0]
    assert set(out["on_disk"]) == set(out["parameters"])
    assert (
        out["requested"]["model.embed_tokens.weight"]
        == out["on_disk"]["model.embed_tokens.weight"]
    )
    q = "model.layers.0.self_attn.q_proj.weight"
    assert out["on_disk"][q] == 16 * 16 * 4 and out["requested"][q] == 8 * 16 * 4


# --------------------------------------------------------------------------- #
# expert parallelism on the MoE fixture
# --------------------------------------------------------------------------- #


def _assert_router_remaps(ranks: dict[int, dict[str, Any]], moe: Any, ep: int) -> None:
    """The one-process router's global ids, over the ranks: each slot is
    owned by exactly one rank (its local id + rank · local count is the
    global id), and every other rank holds the sentinel with a zero score."""
    local = NUM_EXPERTS // ep
    gate = moe.model.model.layers[0].mlp.gate
    with torch.no_grad():
        _, ref_scores, ref_indices = gate(_hidden(moe.model))
    owners = torch.zeros_like(ref_indices)
    for rank, out in ranks.items():
        indices, scores = out["router"]
        assert indices.shape == ref_indices.shape
        owned = indices != local
        assert (indices[owned] >= 0).all() and (indices[owned] < local).all()
        assert torch.equal(indices[owned] + rank * local, ref_indices[owned])
        assert torch.equal(scores[owned], ref_scores[owned])
        assert (scores[~owned] == 0).all()
        owners += owned.to(owners.dtype)
    assert (owners == 1).all()


@pytest.mark.parametrize("ep", (2, 4))
def test_ep_moe_holds_its_experts_remaps_the_router_and_agrees_within_the_band(
    moe: Any, ep: int
) -> None:
    ranks = _load(TINY_QWEN35_MOE, expert=ep)
    assert set(ranks) == set(range(ep))
    reference = _reference_logits(moe)
    for rank, out in ranks.items():
        assert out["geometry"] == ParallelGeometry(expert=ep)
        assert set(out["num_experts"]) == {
            f"model.layers.{i}.mlp.experts" for i in range(4)
        }
        assert set(out["num_experts"].values()) == {NUM_EXPERTS // ep}
        assert set(out["local"]) == {
            f"model.layers.{i}.mlp.experts.{name}"
            for i in range(4)
            for name in ("gate_up_proj", "down_proj")
        }
        assert set(out["placements"].values()) == {(0,)}
        diff = (out["logits"] - reference).abs().max().item()
        assert diff <= BAND, (rank, diff)
    _assert_shards_are_chunks(ranks, moe, ep)
    _assert_router_remaps(ranks, moe, ep)
    _assert_bytes(ranks, ep)
    _assert_one_copy(ranks)
    fused = "model.layers.0.mlp.experts.gate_up_proj"
    assert ranks[0]["on_disk"][fused] == NUM_EXPERTS * 64 * 8 * 2
    assert ranks[0]["requested"][fused] == ranks[0]["on_disk"][fused] // ep


def test_tp2_ep4_moe_loads_and_runs_within_the_band(moe: Any) -> None:
    """The mixed model group: attention and the shared expert over tensor
    groups of two, the routed experts over the expert group of four."""
    ranks = _load(TINY_QWEN35_MOE, tensor=2, expert=4)
    assert set(ranks) == {0, 1, 2, 3}
    reference = _reference_logits(moe)
    whole = dict(moe.model.named_parameters())
    for rank, out in ranks.items():
        assert out["geometry"] == ParallelGeometry(tensor=2, expert=4)
        assert set(out["num_experts"].values()) == {NUM_EXPERTS // 4}
        diff = (out["logits"] - reference).abs().max().item()
        assert diff <= BAND, (rank, diff)
        # the expert shards are quarters, the attention shards halves
        experts = out["local"]["model.layers.0.mlp.experts.down_proj"]
        assert torch.equal(
            experts, whole["model.layers.0.mlp.experts.down_proj"].chunk(4)[rank]
        )
        q = out["local"]["model.layers.3.self_attn.q_proj.weight"]
        assert torch.equal(
            q, whole["model.layers.3.self_attn.q_proj.weight"].chunk(2)[rank % 2]
        )
        assert (
            out["requested"]["model.layers.0.mlp.experts.down_proj"] * 4
            == (out["on_disk"]["model.layers.0.mlp.experts.down_proj"])
        )
        assert (
            out["requested"]["model.layers.3.self_attn.q_proj.weight"] * 2
            == (out["on_disk"]["model.layers.3.self_attn.q_proj.weight"])
        )
    _assert_router_remaps(ranks, moe, 4)
    _assert_one_copy(ranks)


# --------------------------------------------------------------------------- #
# pipeline placement on the dense fixture
# --------------------------------------------------------------------------- #


def test_pp2_llama_each_stage_holds_its_layers_and_reads_only_its_keys(
    llama: Any,
) -> None:
    ranks = _load(TINY_LLAMA, forward=False, pipeline=2)
    assert set(ranks) == {0, 1}
    first, last = ranks[0], ranks[1]
    assert first["identity"] == ["lm_head", "model.layers.1", "model.norm"]
    assert last["identity"] == ["model.embed_tokens", "model.layers.0"]
    everything = {name for name, _ in llama.model.named_parameters()}
    assert set(first["parameters"]) == {
        n for n in everything if n.startswith(("model.embed_tokens", "model.layers.0."))
    }
    assert set(last["parameters"]) == {
        n
        for n in everything
        if n.startswith(("model.layers.1.", "model.norm", "lm_head"))
    }
    for out in ranks.values():
        assert out["geometry"] == ParallelGeometry(pipeline=2)
        assert out["local"] == {} and out["devices"] == "cpu"
        # the report covers the stage's own parameters, whole, and nothing else
        assert set(out["requested"]) == set(out["parameters"])
        assert out["requested"] == out["on_disk"]
    assert set(first["requested"]) | set(last["requested"]) == everything
    assert not set(first["requested"]) & set(last["requested"])
