"""DAS fits at **sharded** sites under simulation (``docs/model_parallelism.md`` §7).

The real hook bodies — ``_installed`` at a module boundary, ``_experts_edit``
at the experts interior — over the real [`TapFragments`][causalab.neural.shared.parallel.taps.TapFragments] and
[`Fragments`][causalab.neural.shared.parallel.fragments.Fragments] on the ``SimulatedWorld`` under drawn schedules, around a
DAS-shaped write: a subspace ``R`` swaps the counterfactual's coordinates into
the base forward, trained by twelve SGD steps on a cross-entropy over
replicated logits, with the §7 guard (``average_gradients``) after every step.
The model is a fake sharded one spelled by hand per placement, as
``test_executor_parallel.py`` spells its modules: a colwise projection whose
output is ``Sharded(-1)``, a position-wise nonlinearity that keeps the shard,
a per-feature scale and the model's own gather to replicated logits; or the
experts' token-major view under ``ExpertLocal`` and the experts' own
output reduction. Every weight and input is a small dyadic rational, so
every product and short sum is exact in fp32 and the parallel fit is the
world-1 fit **bit for bit** — a fit that is only close is a gradient that is
wrong.

Three scenarios: a fit at one sharded site under ``tp=2``; a two-site fit
where the second tap sits *downstream* of the first (the case a ``whole``
off the graph would zero); a fit at the experts interior under ``ep=2``.
Each lands bit-identical across the ranks and to the world-1 twin, and the
ranks' gradients agree before the guard's mean (the ``agreement`` seam at
``0.0``).

The mutations, the repository's convention: a ``fragment`` whose backward
keeps only its own slice — plain torch math, the code before the pairing —
hands every rank a partial gradient, so the ranks disagree before the mean
(refused by the seam) and the guarded fit lands away from world 1 by the
``1/size`` an SGD step carries; a ``whole`` whose gather detaches zeroes the
first featurizer's gradient through the second tap, and the two-site fit
lands away from world 1 while the ranks still agree.
"""

from __future__ import annotations

from typing import Any, Callable

import pytest
import torch
from hypothesis import HealthCheck, example, given, settings

from causalab.neural.engines.pytorch_hooks.executor import (
    _capturing,
    _experts_capture,
    _experts_edit,
    _installed,
)
from causalab.neural.shared.parallel import fragments as fragments_module
from causalab.neural.shared.parallel.agreements import (
    GradientDisagreement,
    average_gradients,
)
from causalab.neural.shared.parallel.autograd import sum_for_edit
from causalab.neural.shared.parallel.collective import Collective
from causalab.neural.shared.parallel.fragments import Fragments, remap_routing
from causalab.neural.shared.parallel.placement import Axis, ExpertLocal, Sharded
from causalab.neural.shared.parallel.taps import TapFragments
from causalab.neural.shared.sites import ResolvedSite
from causalab.protocol.registry.shapes import bsd, flat_td
from tests._helpers import parallel_strategies as ps
from tests._helpers.simulated_world import (
    RankFailed,
    Schedule,
    SimulatedWorld,
    groups_for,
)
from tests._helpers.refusing_collective import RefusingCollective

# pyright: reportPrivateUsage=false

_SETTINGS = settings(
    deadline=None,
    max_examples=30,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)

BATCH, POSITIONS, FEATURE, K, VOCAB = 2, 3, 8, 2, 5
TP = 2
STEPS, LR = 12, 0.25
LABELS = torch.tensor([1, 3])


def _dyadic(shape: tuple[int, ...], seed: int, scale: float) -> torch.Tensor:
    """Integers in ``[-8, 8]`` times ``scale`` — exact products and sums."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(-8, 9, shape, generator=generator).to(torch.float32) * scale


class _ModelGather(torch.autograd.Function):
    """The model's own gather to replicated logits (a ``colwise_gather_output``
    style): all-gather forward, this rank's slice backward — faithful, since
    the consumer is replicated and each rank's piece appears once."""

    @staticmethod
    def forward(
        ctx: Any, tensor: torch.Tensor, c: Collective, axis: Axis
    ) -> torch.Tensor:
        ctx.rank, ctx.width = c.rank(axis), tensor.shape[-1]
        return c.all_gather(tensor.contiguous(), -1, axis)

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        return grad.narrow(-1, ctx.rank * ctx.width, ctx.width), None, None


class _Emit(torch.nn.Module):
    def __init__(self, value: torch.Tensor) -> None:
        super().__init__()
        self.value = value

    def forward(self, *args: Any) -> torch.Tensor:
        return self.value


class _Through(torch.nn.Module):
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden


def _chunk(tensor: torch.Tensor, rank: int, size: int) -> torch.Tensor:
    width = tensor.shape[-1] // size
    return tensor.narrow(-1, rank * width, width)


def _bend(tensor: torch.Tensor) -> torch.Tensor:
    """The position-wise nonlinearity between two taps: a leaky rectifier,
    exact in every element. (``tanh`` is not: torch's vectorised CPU kernel
    rounds an element differently in a 24-element chunk and a 48-element
    whole — one ulp, measured — so a transcendental would put the
    reduction-order band back into a test whose claim is bit identity.)"""
    return torch.where(tensor > 0, tensor, 0.5 * tensor)


def _subspace(seed: int = 0) -> torch.nn.Parameter:
    """The DAS projector ``P = Q Qᵀ`` onto a seeded ``k``-dimensional subspace,
    trained as a free ``(d, d)`` map: the gradient then reaches the parameter
    through **one** product per write, so a leaf fed by two writes
    accumulates two terms — an order-free sum — and the fit stays bit-exact
    across graphs whose backward order differs (``Q`` itself would appear
    twice per write, and four terms round by their order)."""
    generator = torch.Generator().manual_seed(seed)
    basis = torch.linalg.qr(torch.randn(FEATURE, K, generator=generator))[0]
    return torch.nn.Parameter(basis @ basis.T)


def _das_write(
    projector: torch.nn.Parameter, source: torch.Tensor
) -> Callable[..., None]:
    """The DAS swap: the source's coordinates under the projector replace the
    contract's, the complement is kept — ``x + (v − x) P``."""

    def write(contract: torch.Tensor, **_: Any) -> None:
        # from a clone, as the executor's write math reads ``v_pre``: the
        # contract is overwritten in place below, so nothing on the graph
        # may need its old values
        before = contract.clone()
        contract.copy_(before + (source - before) @ projector)

    return write


Outcome = tuple[torch.Tensor, torch.Tensor]  # (R after the fit, first-step ∂L/∂R)


def _fit(
    rotation: torch.nn.Parameter,
    loss_of: Callable[[], torch.Tensor],
    c: Collective,
    *,
    agreement: float | None,
) -> Outcome:
    """Twelve SGD steps with the §7 guard after each; the first step's
    gradient is kept before the mean."""
    first: torch.Tensor | None = None
    for _ in range(STEPS):
        rotation.grad = None
        loss_of().backward()
        assert rotation.grad is not None
        if first is None:
            first = rotation.grad.detach().clone()
        average_gradients([rotation], c, axis="model", agreement=agreement)
        with torch.no_grad():
            rotation -= LR * rotation.grad
    assert first is not None
    return rotation.detach().clone(), first


# --------------------------------------------------------------------------- #
# a colwise output: one site, or two in a row
# --------------------------------------------------------------------------- #


def _sharded_program(
    *, sites: int, agreement: float | None
) -> Callable[[int, Collective], Outcome]:
    """The fake tensor-parallel model: ``h_r = x W[:, chunk_r]`` (a colwise
    output, tapped), a leaky rectifier position-wise, a second tap on the
    input of a pass-through module when ``sites == 2``, a per-feature scale,
    the model's gather and a replicated head."""
    weight = _dyadic((FEATURE, FEATURE), 1, 0.5)
    base = _dyadic((BATCH, POSITIONS, FEATURE), 2, 0.125)
    counterfactual = _dyadic((BATCH, POSITIONS, FEATURE), 3, 0.125)
    scale = _dyadic((FEATURE,), 4, 0.25)
    head = _dyadic((FEATURE, VOCAB), 5, 0.25)
    placement = Sharded(-1, "tensor")

    def program(rank: int, c: Collective) -> Outcome:
        size, me = c.size("tensor"), c.rank("tensor")
        tap = TapFragments(Fragments(c), placement)
        rotation = _subspace()

        def forward(
            x: torch.Tensor, hooks: Callable[[torch.nn.Module, torch.nn.Module], Any]
        ) -> torch.Tensor:
            hidden = x @ (_chunk(weight, me, size) if size > 1 else weight)
            first, second = _Emit(hidden), _Through()
            with hooks(first, second):
                out = _bend(first())
                if sites == 2:
                    out = second(out)
            scaled = out * (_chunk(scale, me, size) if size > 1 else scale)
            whole = _ModelGather.apply(scaled, c, "tensor") if size > 1 else scaled
            return whole @ head

        # the sources: the counterfactual's values at the sites, world-1 contracts
        sink: dict[Any, torch.Tensor] = {}

        def capturing(first: torch.nn.Module, second: torch.nn.Module) -> Any:
            stack = _capturing(
                first, "out", sink, "v1", shape=bsd(FEATURE), batch_size=BATCH, tap=tap
            )
            if sites == 1:
                return stack
            return _Both(
                stack,
                _capturing(
                    second,
                    "in",
                    sink,
                    "v2",
                    shape=bsd(FEATURE),
                    batch_size=BATCH,
                    tap=tap,
                ),
            )

        with torch.no_grad():
            forward(counterfactual, capturing)
        sources = {key: value.clone() for key, value in sink.items()}

        def editing(first: torch.nn.Module, second: torch.nn.Module) -> Any:
            stack = _installed(
                first,
                "out",
                _das_write(rotation, sources["v1"]),
                shape=bsd(FEATURE),
                batch_size=BATCH,
                tap=tap,
            )
            if sites == 1:
                return stack
            return _Both(
                stack,
                _installed(
                    second,
                    "in",
                    _das_write(rotation, sources["v2"]),
                    shape=bsd(FEATURE),
                    batch_size=BATCH,
                    tap=tap,
                ),
            )

        def loss_of() -> torch.Tensor:
            logits = forward(base, editing)[:, -1]
            return torch.nn.functional.cross_entropy(logits, LABELS)

        return _fit(rotation, loss_of, c, agreement=agreement)

    return program


class _Both:
    """Two hook installations entered and left together."""

    def __init__(self, *managers: Any) -> None:
        self.managers = managers

    def __enter__(self) -> None:
        for manager in self.managers:
            manager.__enter__()

    def __exit__(self, *exc: Any) -> None:
        for manager in reversed(self.managers):
            manager.__exit__(*exc)


# --------------------------------------------------------------------------- #
# the experts interior under expert parallelism
# --------------------------------------------------------------------------- #

TOP_K, D_EXPERT, EXPERTS, EP = 2, 4, 4, 2
TOKENS = BATCH * POSITIONS
SLOTS = TOP_K * D_EXPERT


def _experts_site() -> ResolvedSite:
    return ResolvedSite(
        module=None,
        kind="experts",
        shape=flat_td(SLOTS),
        component="expert_activation",
        interface_slot="activation",
    )


def _experts_program(
    *, agreement: float | None
) -> Callable[[int, Collective], Outcome]:
    """The fake expert-parallel model: the token-major view ``(tokens,
    top_k · d_e)`` with this rank's experts' slots filled and the rest zero,
    tapped through the experts interface; a per-slot scale; the experts'
    own output reduction (an all-reduce whose backward is the identity) to
    the replicated head."""
    generator = torch.Generator().manual_seed(9)
    table = torch.stack(
        [torch.randperm(EXPERTS, generator=generator)[:TOP_K] for _ in range(TOKENS)]
    )
    base = _dyadic((TOKENS, SLOTS), 12, 0.125)
    counterfactual = _dyadic((TOKENS, SLOTS), 13, 0.125)
    scale = _dyadic((SLOTS,), 14, 0.25)
    head = _dyadic((SLOTS, VOCAB), 15, 0.25)
    site = _experts_site()

    def program(rank: int, c: Collective) -> Outcome:
        size, me = c.size("expert"), c.rank("expert")
        tap = TapFragments(
            Fragments(c),
            ExpertLocal("expert"),
            remapped_routing=True,
            num_experts=EXPERTS,
        )
        owned = torch.div(table, EXPERTS // size, rounding_mode="floor") == me
        keep = (
            owned.unsqueeze(-1).expand(TOKENS, TOP_K, D_EXPERT).reshape(TOKENS, SLOTS)
        )
        local_table = remap_routing(table, EXPERTS, me, size)
        rotation = _subspace()

        def local(view: torch.Tensor) -> torch.Tensor:
            return torch.where(keep, view, torch.zeros(()))

        sink: dict[Any, torch.Tensor] = {}
        idx_sink: dict[Any, torch.Tensor] = {}
        with torch.no_grad():
            _experts_capture(sink, idx_sink, "v", site, BATCH, tap)(
                local(counterfactual), local_table
            )
        source = sink["v"].clone()  # (batch, position, slots), the world-1 contract

        def loss_of() -> torch.Tensor:
            edited = _experts_edit(site, _das_write(rotation, source), BATCH, tap)(
                local(base).clone(), local_table
            )
            scaled = edited * scale
            combined = sum_for_edit(scaled, "expert", c) if size > 1 else scaled
            logits = (combined @ head).reshape(BATCH, POSITIONS, VOCAB)[:, -1]
            return torch.nn.functional.cross_entropy(logits, LABELS)

        return _fit(rotation, loss_of, c, agreement=agreement)

    return program


# --------------------------------------------------------------------------- #
# the scenarios
# --------------------------------------------------------------------------- #


def _solo(program: Callable[[int, Collective], Outcome]) -> Outcome:
    return program(0, RefusingCollective())


def _assert_bit_identical(results: list[Outcome], oracle: Outcome) -> None:
    for rank, (fitted, gradient) in enumerate(results):
        assert torch.equal(gradient, oracle[1]), f"rank {rank}: the first gradient"
        assert torch.equal(fitted, oracle[0]), f"rank {rank}: the fit"


def _drift(results: list[Outcome], oracle: Outcome) -> float:
    return max(float((fitted - oracle[0]).abs().max()) for fitted, _ in results)


class TestShardedSiteFits:
    @pytest.mark.property
    @given(schedule=ps.schedules())
    @example(schedule=[])
    @_SETTINGS
    def test_a_fit_at_a_colwise_output_is_the_world_one_fit(
        self, schedule: Schedule
    ) -> None:
        program = _sharded_program(sites=1, agreement=0.0)
        world = SimulatedWorld(groups_for(TP, tensor=TP), world=TP, schedule=schedule)
        _assert_bit_identical(world.run(program), _solo(program))

    @pytest.mark.property
    @given(schedule=ps.schedules())
    @_SETTINGS
    def test_a_fit_through_two_taps_in_a_row_is_the_world_one_fit(
        self, schedule: Schedule
    ) -> None:
        """The second tap's ``whole`` sits downstream of the first tap's edit:
        the featurizer's gradient through it survives the gather."""
        program = _sharded_program(sites=2, agreement=0.0)
        world = SimulatedWorld(groups_for(TP, tensor=TP), world=TP, schedule=schedule)
        _assert_bit_identical(world.run(program), _solo(program))

    @pytest.mark.property
    @given(schedule=ps.schedules())
    @_SETTINGS
    def test_a_fit_at_the_experts_interior_is_the_world_one_fit(
        self, schedule: Schedule
    ) -> None:
        program = _experts_program(agreement=0.0)
        world = SimulatedWorld(groups_for(EP, expert=EP), world=EP, schedule=schedule)
        _assert_bit_identical(world.run(program), _solo(program))

    @pytest.mark.unit
    def test_the_fits_move_the_subspace(self) -> None:
        """The scenarios are not vacuous: twelve steps move ``R`` away from
        its start by more than the mutations' bands below."""
        for program in (
            _sharded_program(sites=1, agreement=None),
            _sharded_program(sites=2, agreement=None),
            _experts_program(agreement=None),
        ):
            fitted, gradient = _solo(program)
            assert float((fitted - _subspace().detach()).abs().max()) > 1e-2
            assert float(gradient.abs().max()) > 1e-3


# --------------------------------------------------------------------------- #
# the mutations
# --------------------------------------------------------------------------- #


def _own_slice_only(
    edited: torch.Tensor, dim: int, axis: Axis, collective: Collective, **kwargs: Any
) -> torch.Tensor:
    """A ``fragment`` that is plain torch math (the code before the pairing):
    its backward scatters this rank's slice into zeros — a partial."""
    return fragments_module._chunk(
        edited, dim, collective.rank(axis), collective.size(axis), Sharded(dim, axis)
    )


def _own_slots_only(
    edited: torch.Tensor, keep: torch.Tensor | None, axis: Axis, collective: Collective
) -> torch.Tensor:
    """The same mutation at the experts interior: ``where`` alone."""
    assert keep is not None
    return torch.where(keep, edited, edited.new_zeros(()))


def _detaching_gather(
    tensor: torch.Tensor, dim: int, axis: Axis, collective: Collective, **kwargs: Any
) -> torch.Tensor:
    """A ``whole`` that is the raw all-gather: fresh buffers, off the graph."""
    return collective.all_gather(tensor, dim, axis)


@pytest.mark.unit
class TestMutations:
    def test_a_fragment_keeping_its_own_slice_makes_the_ranks_disagree(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(fragments_module, "edit_fragment", _own_slice_only)
        world = SimulatedWorld(groups_for(TP, tensor=TP), world=TP, schedule=0)
        with pytest.raises(RankFailed) as err:
            world.run(_sharded_program(sites=1, agreement=0.0))
        assert isinstance(err.value.__cause__, GradientDisagreement)

    def test_a_fragment_keeping_its_own_slice_lands_the_guarded_fit_off_world_one(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without the seam the guard averages two partials — half the
        gradient — and every SGD step is half a step."""
        monkeypatch.setattr(fragments_module, "edit_fragment", _own_slice_only)
        program = _sharded_program(sites=1, agreement=None)
        world = SimulatedWorld(groups_for(TP, tensor=TP), world=TP, schedule=0)
        results = world.run(program)
        oracle = _solo(program)
        assert _drift(results, oracle) > 1e-2, "the fit stayed within the band"
        for _, gradient in results:
            assert not torch.equal(gradient, oracle[1])

    def test_a_summand_keeping_its_own_slots_makes_the_ranks_disagree(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(fragments_module, "edit_summand", _own_slots_only)
        world = SimulatedWorld(groups_for(EP, expert=EP), world=EP, schedule=0)
        with pytest.raises(RankFailed) as err:
            world.run(_experts_program(agreement=0.0))
        assert isinstance(err.value.__cause__, GradientDisagreement)

    def test_a_whole_that_detaches_lands_the_two_tap_fit_off_world_one(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The ranks still agree — the mutation is the same on both — so the
        seam is silent and the drift from world 1 is the only witness."""
        monkeypatch.setattr(fragments_module, "gather_for_edit", _detaching_gather)
        program = _sharded_program(sites=2, agreement=0.0)
        world = SimulatedWorld(groups_for(TP, tensor=TP), world=TP, schedule=0)
        results = world.run(program)
        assert _drift(results, _solo(program)) > 1e-2, "the fit stayed within the band"

    def test_a_whole_that_detaches_is_harmless_at_a_single_tap(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Nothing trained sits upstream of a single tap's ``whole``, which is
        why every existing fit at one site never met the bug."""
        monkeypatch.setattr(fragments_module, "gather_for_edit", _detaching_gather)
        program = _sharded_program(sites=1, agreement=0.0)
        world = SimulatedWorld(groups_for(TP, tensor=TP), world=TP, schedule=0)
        _assert_bit_identical(world.run(program), _solo(program))
