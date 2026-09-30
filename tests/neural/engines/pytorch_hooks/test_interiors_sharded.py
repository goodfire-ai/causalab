"""The three interiors on real sharded tiny models over ``gloo``
(``docs/model_parallelism.md`` §6.2–6.4, §10.6): every attention-interior
component read and written at ``tp=2`` on both fixtures, every expert
component at ``ep=2`` and ``ep=4``, every delta slot at ``tp=2``, through the
real [`PointExecutor`][causalab.neural.engines.pytorch_hooks.executor.PointExecutor] over a [`Fragments`][causalab.neural.shared.parallel.fragments.Fragments] bound to the world's
collective, compared to the same document on the one-process model.

Each world is spawned once per geometry (a module fixture), loads the fixture
on every rank the way ``test_sharded_load.py`` does, runs the whole case list
and hands the values back; the parent runs the same cases at world 1 and
compares case by case.

**The band.** A sharded forward is the one-process computation up to the
reduction order of its collectives (``test_sharded_load.py``'s band, ``1e-5``,
measured there at ≤ ``4.8e-7`` on the fixtures' logits). The interiors add
no rounding of their own: an all-gather reorders nothing, an ``ExpertLocal``
sum adds zeros, and the routing reconstruction is integer arithmetic — so a
*read* of a gathered or expert-local slot is expected bit for bit where the
tensor feeding it is, and within the band where a sharded GEMM, a rowwise
all-reduce or the experts' output reduction feeds it. Measured maxima are in
`MEASURED`; the band is the loader's, and a gather in the wrong head
order (the mutation) lands more than twenty times outside it.
"""

from __future__ import annotations

import os
from typing import Any

import pytest
import torch

from causalab.neural.engines.pytorch_hooks.executor import PointExecutor
from causalab.neural.engines.pytorch_hooks.loading import load_model
from causalab.neural.engines.pytorch_hooks.sharding import Sharding
from causalab.neural.shared.parallel.collective import SOLO
from causalab.neural.shared.parallel.fragments import Fragments
from causalab.neural.shared.parallel.placement import REPLICATED, Sharded
from causalab.neural.shared.sites import resolve_site
from causalab.protocol.rules.errors import ProtocolError
from causalab.protocol.parallel import ParallelGeometry
from causalab.protocol.registry import DELTA_KERNEL_SLOTS
from causalab.protocol.schema import SiteSpec

from tests._helpers import a3b_sweep as sweep
from tests._helpers.sharded_world import run_world
from tests.neural.engines.pytorch_hooks.conftest import TINY_LLAMA, TINY_QWEN35_MOE

# `parallel_world`: every test here runs a document or a fit across a spawned
# multi-rank process world — minutes on the CI runner. The PR gate deselects the
# marker; the nightly CPU job runs it (docs/TESTS.md).
pytestmark = [pytest.mark.smoke, pytest.mark.parallel_world]

#: The measured maximum absolute difference per geometry (fp32, gloo, CPU,
#: torch 2.9.0, 2026-09-14, this file's own program), over every case. The
#: worst cases are reads *downstream* of a sharded GEMM's rounding: at
#: ``tp=2`` the MoE's ``attention_scores`` (the colwise shard's ``q`` and
#: ``k`` differ from the whole GEMM's by an ulp, amplified through the
#: ``q·k`` products at the scores' magnitude); under EP the layer-3
#: ``attention_query``, three expert all-reduces downstream. The gathered and
#: expert-local slots themselves are at or below ``1.5e-8``.
MEASURED = {
    "llama tp=2": 1.2e-7,
    "moe tp=2": 1.1e-6,
    "moe ep=2": 4.8e-7,
    "moe ep=4": 4.8e-7,
    "moe tp=2,ep=4": 4.8e-7,
}
#: The loader's band (``test_sharded_load.py``): pinned at ``1e-5``, nine
#: times the largest maximum measured here, and more than twenty times below
#: what the wrong head order (the mutation below) makes of the same reads and
#: writes (``2.6e-4`` at its nearest).
BAND = 1e-5
assert BAND >= 5 * max(MEASURED.values())

ROWS = [
    {
        "input": "the quick brown fox jumps",
        "counterfactual_inputs": ["a slow green turtle"],
    },
    {
        "input": "a small red hen sits",
        "counterfactual_inputs": ["the big blue whale dives"],
    },
]

Case = tuple[str, str, int | None, dict[str, Any]]

ATTENTION_LLAMA = (
    "attention_query_pre_rope",
    "attention_key_pre_rope",
    "attention_value_states",
    "attention_query",
    "attention_key",
    "attention_scores",
    "attention_probs",
    "attention_z",
    "attention_premix",
    "attention_result",
    "attention_output",
)
ATTENTION_MOE = ATTENTION_LLAMA + ("attention_gate",)
DELTA_BOUNDARIES = ("delta_qkv", "delta_gate", "delta_premix")
EXPERTS = (
    "router_logits",
    "router_scores",
    "expert_idx",
    "routed_output",
    "expert_gate_proj",
    "expert_up_proj",
    "expert_activation",
    "expert_output",
    "shared_expert_activation",
    "shared_expert_output",
)


def _case(mode: str, component: str, layer: int | None, **extra: Any) -> Case:
    return (mode, component, layer, extra)


def _name(case: Case) -> str:
    mode, component, layer, extra = case
    scope = "".join(f" {k}={v}" for k, v in extra.items())
    return f"{mode} {component}@{layer}{scope}"


def _both(components: tuple[str, ...], layer: int) -> list[Case]:
    reads = [_case("read", c, layer) for c in components]
    writes = [_case("write", c, layer) for c in sweep.write_cases(components)]
    return reads + writes


LLAMA_TP2: list[Case] = _both(ATTENTION_LLAMA, 1) + [
    # heads rank 0 never holds locally (H = 4, tp = 2 → heads 2, 3 on rank 1)
    _case("read", "attention_query", 1, head=3),
    _case("read", "attention_query_pre_rope", 1, head=3),
    _case("read", "attention_result", 1, head=3),
    _case("read", "attention_premix", 1, head=2),
    _case("write", "attention_query", 1, head=3),
    _case("read", "block_output", 1),
]
MOE_TP2: list[Case] = (
    _both(ATTENTION_MOE, 3)
    + _both(tuple(DELTA_KERNEL_SLOTS), 0)
    + [_case("read", c, 0) for c in DELTA_BOUNDARIES]
    + [_case("read", c, 0) for c in EXPERTS]
    + [
        _case("read", "attention_query", 3, head=7),
        _case("read", "attention_key", 3, head=3),
        _case("read", "attention_gate", 3, head=7),
        _case("read", "attention_result", 3, head=7),
        _case("write", "attention_probs", 3),
        _case("read", "attention_probs", 3),
    ]
)
MOE_EP: list[Case] = _both(EXPERTS, 0) + [
    _case("read", "attention_query", 3),
    _case("read", "delta_state", 0),
]
MOE_TP2_EP4: list[Case] = [
    _case("read", "attention_query", 3),
    _case("read", "attention_probs", 3),
    _case("write", "attention_probs", 3),
    _case("read", "expert_activation", 0),
    _case("write", "expert_activation", 0),
    _case("read", "expert_output", 0),
    _case("read", "expert_idx", 0),
    _case("read", "delta_state", 0),
]

#: The mutation: chunks gathered in the wrong head order (§10.7).
REVERSED_GATHER = "reversed_gather"


def _payload(value: Any) -> Any:
    """A read value as something a process boundary carries."""
    if hasattr(value, "flat") and hasattr(value, "widths"):
        return {"flat": value.flat, "widths": torch.tensor(value.widths)}
    return value


def _run_case(bundle: Any, fragments: Fragments, case: Case) -> Any:
    """One document through the executor over ``fragments``: a read's value,
    or the patched logits of an interchange; a refusal by its message."""
    mode, component, layer, extra = case
    try:
        if mode == "read":
            doc = sweep.read_doc(
                component,
                layer,
                pos=sweep.default_pos(component),
                head=extra.get("head"),
            )
            if "expert" in extra:
                doc["method"]["sites"]["tap"]["expert"] = extra["expert"]
            executor = sweep.make_executor(
                PointExecutor, doc, bundle, rows=ROWS, with_cf=False
            )
            executor.fragments = fragments
            return _payload(executor.read_value("r"))
        doc = sweep.interchange_doc(component, layer, pos=sweep.default_pos(component))
        if "head" in extra:
            doc["method"]["sites"]["tap"]["head"] = extra["head"]
        executor = sweep.make_executor(
            PointExecutor, doc, bundle, rows=ROWS, with_cf=True
        )
        executor.fragments = fragments
        return executor.dense_value("logits")
    except ProtocolError as error:
        return {"refused": str(error)}


def _reverse_gathers() -> None:
    """Gather the ranks' chunks in reverse order — a wrong head order."""
    from causalab.neural.shared.parallel import collective as collective_module

    real = collective_module.TorchCollective.all_gather

    def reversed_gather(self: Any, tensor: torch.Tensor, dim: int, axis: str) -> Any:
        out = real(self, tensor, dim, axis)
        chunks = out.chunk(self.size(axis), dim=dim)
        return torch.cat(list(reversed(chunks)), dim=dim)

    collective_module.TorchCollective.all_gather = reversed_gather  # type: ignore[method-assign]


def _program(rank: int, world: int, payload: Any) -> dict[str, Any]:
    from causalab.neural.shared.parallel.collective import TorchCollective
    from causalab.neural.shared.parallel.mesh import Mesh

    key, geometry_kwargs, cases, mutation = payload
    geometry = ParallelGeometry(**geometry_kwargs)
    os.environ["RANK"], os.environ["WORLD_SIZE"] = str(rank), str(world)
    mesh = Mesh.from_environment(geometry)
    if mutation == REVERSED_GATHER:
        _reverse_gathers()
    collective = TorchCollective(mesh)
    sharding = Sharding(
        geometry=geometry,
        rank=rank,
        meshes={
            axis: mesh.device_mesh(axis)
            for axis in ("tensor", "expert")
            if getattr(geometry, axis) > 1
        },
        collective=collective,
    )
    bundle = load_model(key, sharding=sharding)
    fragments = Fragments(collective)
    out: dict[str, Any] = {
        _name(case): _run_case(bundle, fragments, case) for case in cases
    }
    # the placement facts the parent asserts, read off the resolver
    placements: dict[str, str] = {}
    for mode, component, layer, extra in cases:
        if mode != "read" or extra:
            continue
        site = resolve_site(bundle, SiteSpec(component=component, layers=(layer,)))
        placements[component] = repr(site.placement)
    out["placements"] = placements
    if geometry.tensor > 1 and key == TINY_QWEN35_MOE:
        # 📐 the norm between colwise and rowwise emits the local heads
        shapes: dict[str, tuple[int, ...]] = {}
        attn = bundle.model.model.layers[3].self_attn
        handle = attn.q_norm.register_forward_hook(
            lambda _m, _i, o: shapes.__setitem__("q_norm", tuple(o.shape))
        )
        try:
            with torch.no_grad():
                bundle.model(input_ids=torch.tensor([[5, 17, 23, 42]]))
        finally:
            handle.remove()
        out["q_norm_shape"] = shapes["q_norm"]
        out["q_proj_out_features"] = int(attn.q_proj.out_features)
    return out


def _world(
    key: str, cases: list[Case], *, mutation: str | None = None, **geometry: int
) -> dict[int, dict[str, Any]]:
    world = ParallelGeometry(**geometry).world
    return run_world(world, _program, (key, geometry, cases, mutation))


def _reference(key: str, cases: list[Case]) -> dict[str, Any]:
    bundle = load_model(key)
    return {_name(case): _run_case(bundle, Fragments(SOLO), case) for case in cases}


def _diff(got: Any, want: Any) -> float:
    """The maximum absolute difference, ``0.0`` for equal integer tensors and
    refusals of one message, ``inf`` for a mismatch of kind."""
    if isinstance(got, dict) and isinstance(want, dict):
        if "refused" in got or "refused" in want:
            return 0.0 if got.get("refused") == want.get("refused") else float("inf")
        if not torch.equal(got["widths"], want["widths"]):
            return float("inf")
        return _diff(got["flat"], want["flat"])
    if not isinstance(got, torch.Tensor) or not isinstance(want, torch.Tensor):
        return float("inf")
    if got.shape != want.shape:
        return float("inf")
    if not got.dtype.is_floating_point:
        return 0.0 if torch.equal(got, want) else float("inf")
    return float((got.double() - want.double()).abs().max())


def _assert_within_band(
    ranks: dict[int, dict[str, Any]], reference: dict[str, Any], label: str
) -> float:
    """Every case on every rank within the band; the worst case printed
    beside the pinned maximum (``-s``)."""
    worst, worst_name = 0.0, ""
    for rank, out in ranks.items():
        for name, want in reference.items():
            diff = _diff(out[name], want)
            assert diff <= BAND, f"{label}, rank {rank}: {name} differs by {diff:.3e}"
            if diff > worst:
                worst, worst_name = diff, name
    print(
        f"\nmeasured {label}: {worst:.3e} ({worst_name}; pinned {MEASURED[label]:.1e})"
    )
    return worst


def _assert_writes_landed(reference: dict[str, Any], clean: torch.Tensor) -> None:
    """Anti-vacuity: at world 1 every interchange moved the logits."""
    for name, value in reference.items():
        if name.startswith("write") and isinstance(value, torch.Tensor):
            assert not torch.allclose(value, clean), f"{name}: the write did nothing"


@pytest.fixture(scope="module")
def clean_llama() -> torch.Tensor:
    return _reference(TINY_LLAMA, [_case("read", "lm_head", None)])["read lm_head@None"]


@pytest.fixture(scope="module")
def clean_moe() -> torch.Tensor:
    return _reference(TINY_QWEN35_MOE, [_case("read", "lm_head", None)])[
        "read lm_head@None"
    ]


# --------------------------------------------------------------------------- #
# tensor parallelism: the attention interior (§6.2) and DeltaNet (§6.4)
# --------------------------------------------------------------------------- #


def test_llama_tp2_attention_interior_reads_and_writes_agree_with_world_one(
    clean_llama: torch.Tensor,
) -> None:
    reference = _reference(TINY_LLAMA, LLAMA_TP2)
    _assert_writes_landed(reference, clean_llama)
    ranks = _world(TINY_LLAMA, LLAMA_TP2, tensor=2)
    _assert_within_band(ranks, reference, "llama tp=2")
    for out in ranks.values():
        assert out["placements"]["attention_query"] == repr(Sharded(1, "tensor"))
        assert out["placements"]["attention_z"] == repr(Sharded(2, "tensor"))
        assert out["placements"]["attention_probs"] == repr(Sharded(1, "tensor"))
        assert out["placements"]["attention_query_pre_rope"] == repr(
            Sharded(-1, "tensor")
        )
        assert out["placements"]["attention_output"] == repr(REPLICATED)


def test_moe_tp2_attention_interior_and_every_delta_slot_agree_with_world_one(
    clean_moe: torch.Tensor,
) -> None:
    reference = _reference(TINY_QWEN35_MOE, MOE_TP2)
    _assert_writes_landed(reference, clean_moe)
    ranks = _world(TINY_QWEN35_MOE, MOE_TP2, tensor=2)
    _assert_within_band(ranks, reference, "moe tp=2")
    for out in ranks.values():
        # §6.4: all ten delta slots replicated, verified on the loaded model
        for component in DELTA_KERNEL_SLOTS:
            assert out["placements"][component] == repr(REPLICATED), component
        for component in DELTA_BOUNDARIES:
            assert out["placements"][component] == repr(REPLICATED), component
        # the norm between colwise and rowwise emits the local heads (4 of 8)
        assert out["q_norm_shape"] == (1, 4, 4, 32)
        assert out["placements"]["attention_query_pre_rope"] == repr(
            Sharded(2, "tensor")
        )
        # nn.Linear keeps its global width under a DTensor weight
        assert out["q_proj_out_features"] == 512
        # the experts under this plan are on the expert axis: whole at tp=2 alone
        assert out["placements"]["expert_activation"] == repr(REPLICATED)


#: 📐 A gathered projection reorders nothing, but the GEMM over a shard of the
#: weight rounds differently from the GEMM over the whole (BLAS blocking):
#: ``delta_conv`` at layer 0 measured ``1.5e-8`` off world 1, one ulp at its
#: magnitude, and the rest at or below it. Pinned at ten times that.
DELTA_BAND = 1.5e-7


def test_moe_tp2_delta_slot_reads_are_within_one_gemm_rounding() -> None:
    """Every DeltaNet slot is replicated: the only difference from world 1 is
    the sharded GEMM's rounding, not a reordering of any axis."""
    cases = [_case("read", c, 0) for c in DELTA_KERNEL_SLOTS]
    reference = _reference(TINY_QWEN35_MOE, cases)
    ranks = _world(TINY_QWEN35_MOE, cases, tensor=2)
    worst = 0.0
    for rank, out in ranks.items():
        for name, want in reference.items():
            diff = _diff(out[name], want)
            assert diff <= DELTA_BAND, (rank, name, diff)
            worst = max(worst, diff)
    print(f"\nmeasured delta slots at tp=2: {worst:.3e}")
    # every rank holds the same whole tensor
    for name in reference:
        assert _diff(ranks[0][name], ranks[1][name]) == 0.0, name


# --------------------------------------------------------------------------- #
# expert parallelism: the experts interior (§6.3)
# --------------------------------------------------------------------------- #


def _with_chosen_expert(cases: list[Case]) -> list[Case]:
    """The case list plus an ``expert:``-scoped read of an expert the router
    chose at world 1 (an unchosen one is an ``unavailable`` cell, a data fact)."""
    idx = _reference(TINY_QWEN35_MOE, [_case("read", "expert_idx", 0)])[
        "read expert_idx@0"
    ]
    chosen = int(torch.mode(idx.reshape(-1)).values)
    return cases + [_case("read", "expert_activation", 0, expert=chosen)]


@pytest.mark.parametrize("ep", (2, 4))
def test_moe_ep_expert_components_agree_with_world_one(
    ep: int, clean_moe: torch.Tensor
) -> None:
    cases = _with_chosen_expert(MOE_EP)
    reference = _reference(TINY_QWEN35_MOE, cases)
    _assert_writes_landed(reference, clean_moe)
    ranks = _world(TINY_QWEN35_MOE, cases, expert=ep)
    label = f"moe ep={ep}"
    # the one refusal by name (§6.6): a write on the router's scores, whose
    # slots are owned per rank and whose module tap carries no routing
    # table; a write on its indices is served — the tap is the routing table
    # itself and the scores are re-masked to the edited one (``rescore``) —
    # and stays in the band comparison below
    for name in ("write router_scores@0",):
        for out in ranks.values():
            refused = out[name]
            assert (
                isinstance(refused, dict) and "expert parallelism" in refused["refused"]
            )
        assert isinstance(reference[name], torch.Tensor)  # served at world 1
        del reference[name]
    _assert_within_band(ranks, reference, label)
    for out in ranks.values():
        assert out["placements"]["expert_activation"] == "ExpertLocal(group='expert')"
        assert out["placements"]["router_scores"] == "ExpertLocal(group='expert')"
        assert out["placements"]["expert_idx"] == repr(REPLICATED)
        assert out["placements"]["attention_query"] == repr(REPLICATED)


def test_moe_tp2_ep4_mixed_group_agrees_with_world_one(clean_moe: torch.Tensor) -> None:
    reference = _reference(TINY_QWEN35_MOE, MOE_TP2_EP4)
    _assert_writes_landed(reference, clean_moe)
    ranks = _world(TINY_QWEN35_MOE, MOE_TP2_EP4, tensor=2, expert=4)
    _assert_within_band(ranks, reference, "moe tp=2,ep=4")
    for out in ranks.values():
        assert out["placements"]["attention_query"] == repr(Sharded(1, "tensor"))
        assert out["placements"]["expert_activation"] == "ExpertLocal(group='expert')"


# --------------------------------------------------------------------------- #
# the mutation: a gather in the wrong head order lands outside the band
# --------------------------------------------------------------------------- #


def test_a_gather_in_the_wrong_head_order_lands_outside_the_band() -> None:
    cases = [
        _case("read", "attention_scores", 1),
        _case("read", "attention_query", 1),
        _case("write", "attention_probs", 1),
        _case("write", "attention_query", 1),
    ]
    reference = _reference(TINY_LLAMA, cases)
    ranks = _world(TINY_LLAMA, cases, mutation=REVERSED_GATHER, tensor=2)
    nearest = float("inf")
    for rank, out in ranks.items():
        for name, want in reference.items():
            diff = _diff(out[name], want)
            assert diff > BAND, f"rank {rank}: {name} within the band ({diff:.3e})"
            nearest = min(nearest, diff)
    print(
        f"\nthe wrong head order lands no nearer than {nearest:.3e} (band {BAND:.0e})"
    )
