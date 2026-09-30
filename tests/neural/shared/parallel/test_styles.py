"""The ``Styles`` conformance suite run against both tiers
(``docs/model_parallelism.md`` §10.8).

``styles_contract.py`` states one row per parallel style — a tiny model
sharded through ``apply_plan`` and read back — and the check against the
world-1 run of the same program. This file runs the rows

1. with the fragment tier over the simulator (``unit`` at each row's
   geometries; ``property`` over drawn schedules),
2. with the fragment tier over ``gloo`` in spawned processes, and
3. with transformers' DTensor tier over ``gloo``,

and holds the three sets of results **bit-identical** to each other and to
world 1: the simulator's fixed-order reductions, ``gloo``'s and DTensor's
redistributes agree to the bit on every row, which is what lets the
simulated scenarios stand in for the production styles. The hand-written
mutations close the tier: a colwise style that forgets to sum the input
gradient, a rowwise style that adds the bias on every rank before the
sum, a router that keeps the other ranks' scores — each fails its row.
"""

from __future__ import annotations

import functools
from typing import Any, Callable, Mapping, Sequence

import pytest
import torch
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

from causalab.neural.engines.pytorch_hooks.styles import Group, StyleError
from causalab.neural.engines.pytorch_hooks.styles import fragment as fragment_module
from causalab.neural.engines.pytorch_hooks.styles.fragment import FragmentStyles
from causalab.neural.shared.parallel.placement import AXES
from causalab.protocol.parallel import MeshLayout, ParallelGeometry, format_geometry
from tests._helpers import parallel_strategies as ps
from tests._helpers.gloo_world import GlooWorld
from tests._helpers.simulated_world import Schedule, SimulatedWorld
from tests.neural.shared.parallel.styles_contract import (
    BY_NAME,
    EXPERT_2,
    ROWS,
    TENSOR_2,
    Results,
    StyleContract,
    every_row,
    fragment_tier,
    oracle,
    run_row,
    transformers_tier,
    verify,
    verify_all,
)

_SETTINGS = settings(
    deadline=None,
    max_examples=30,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)

_ids: Callable[[Any], str] = lambda x: (  # noqa: E731 - pytest id helper
    x.name if isinstance(x, StyleContract) else format_geometry(x)
)

#: The geometries the spawned ``gloo`` worlds run: the tensor rows at
#: ``tp=2``, the expert rows at ``ep=2``.
GLOO_GEOMETRIES: tuple[ParallelGeometry, ...] = (TENSOR_2, EXPERT_2)


def _simulated(geometry: ParallelGeometry, schedule: Schedule = ()) -> SimulatedWorld:
    layout = MeshLayout(geometry)
    return SimulatedWorld(
        {axis: layout.groups(axis) for axis in AXES},
        world=geometry.world,
        schedule=schedule,
        timeout=60.0,
    )


def _cases() -> list[tuple[StyleContract, ParallelGeometry]]:
    return [(row, geometry) for row in ROWS for geometry in row.geometries]


@pytest.fixture(scope="module")
def references() -> dict[str, Results]:
    return {row.name: oracle(row) for row in ROWS}


def _equal_results(a: Any, b: Any, where: tuple[Any, ...] = ()) -> None:
    if isinstance(a, dict):
        assert set(a) == set(b), where
        for key in a:
            _equal_results(a[key], b[key], (*where, key))
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b), where
        for index, (x, y) in enumerate(zip(a, b)):
            _equal_results(x, y, (*where, index))
    elif isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor) and torch.equal(a, b), where
    else:
        assert a == b, where


# --------------------------------------------------------------------------- #
# unit: the fragment tier over the simulator
# --------------------------------------------------------------------------- #


@pytest.mark.unit
class TestFragmentStylesSimulated:
    @pytest.mark.parametrize(("row", "geometry"), _cases(), ids=_ids)
    def test_the_row_holds(
        self,
        row: StyleContract,
        geometry: ParallelGeometry,
        references: dict[str, Results],
    ) -> None:
        program = functools.partial(
            run_row, row=row, tier=fragment_tier, geometry=geometry
        )
        results = _simulated(geometry).run(program)
        verify(row, results, MeshLayout(geometry), references[row.name])

    def test_every_row_is_a_style_of_the_registry(self) -> None:
        from causalab.protocol.registry import STYLES

        covered = {style for row in ROWS for style in row.plan.rows.values().__iter__()}
        assert {row.style for row in covered} == STYLES


@pytest.mark.property
class TestFragmentStylesProperties:
    @_SETTINGS
    @given(schedule=ps.schedules(), which=st.sampled_from(GLOO_GEOMETRIES))
    @example(schedule=[], which=TENSOR_2)
    def test_every_row_holds_under_every_schedule(
        self,
        schedule: Schedule,
        which: ParallelGeometry,
        references: dict[str, Results],
    ) -> None:
        program = functools.partial(every_row, tier=fragment_tier, geometry=which)
        results = _simulated(which, schedule).run(program)
        verify_all(results, MeshLayout(which), references)


# --------------------------------------------------------------------------- #
# unit: the mutations
# --------------------------------------------------------------------------- #


def _forget_the_input_sum() -> None:
    """The mutation: a colwise projection whose input gradient stays a
    partial — each rank's share of its own columns."""

    def install(self: Any, module: Any, group: Any, *, expert_parallel: bool) -> None:
        self._check(group)

    fragment_module._Colwise.install = install  # type: ignore[method-assign]


def _bias_before_the_sum() -> None:
    """The mutation: a rowwise projection adding its bias on every rank
    before the all-reduce — the bias counted ``size`` times."""
    from causalab.neural.shared.parallel.autograd import sum_for_edit

    def install(self: Any, module: Any, group: Any, *, expert_parallel: bool) -> None:
        self._check(group)
        axis, collective = group.axis, self.collective

        def transform(original: Any, args: tuple, kwargs: dict) -> Any:
            return sum_for_edit(original(*args, **kwargs), axis, collective)

        fragment_module._wrap(module, transform)

    fragment_module._Rowwise.install = install  # type: ignore[method-assign]


def _keep_every_score() -> None:
    """The mutation: a router that remaps the ids but keeps every rank's
    scores — the scores' sum over the group doubles."""
    from causalab.neural.shared.parallel.fragments import remap_routing

    def install(self: Any, module: Any, group: Any, *, expert_parallel: bool) -> None:
        self._check(group)
        rank, size = group.rank, group.size

        def transform(original: Any, args: tuple, kwargs: dict) -> Any:
            logits, scores, indices, *extra = original(*args, **kwargs)
            num_experts = fragment_module._num_experts(module)
            return (
                logits,
                scores,
                remap_routing(indices, num_experts, rank, size),
                *extra,
            )

        fragment_module._wrap(module, transform)

    fragment_module._EpRouter.install = install  # type: ignore[method-assign]


MUTATIONS: tuple[tuple[str, Callable[[], None], ParallelGeometry], ...] = (
    ("colwise", _forget_the_input_sum, TENSOR_2),
    ("rowwise", _bias_before_the_sum, TENSOR_2),
    ("ep_router", _keep_every_score, EXPERT_2),
)


@pytest.mark.unit
class TestFragmentStylesMutations:
    @pytest.mark.parametrize(
        ("name", "mutation", "geometry"),
        MUTATIONS,
        ids=lambda x: x if isinstance(x, str) else "",
    )
    def test_the_mutation_fails_its_row(
        self,
        name: str,
        mutation: Callable[[], None],
        geometry: ParallelGeometry,
        monkeypatch: pytest.MonkeyPatch,
        references: dict[str, Results],
    ) -> None:
        row = BY_NAME[name]
        if name == "rowwise":
            # the fixture's projection has no bias: the mutation needs one
            row = _biased_rowwise(row)
            references = {**references, name: oracle(row)}
        cls_name = {
            "colwise": "_Colwise",
            "rowwise": "_Rowwise",
            "ep_router": "_EpRouter",
        }[name]
        cls = getattr(fragment_module, cls_name)
        monkeypatch.setattr(cls, "install", cls.install)  # restored after
        mutation()
        program = functools.partial(
            run_row, row=row, tier=fragment_tier, geometry=geometry
        )
        results = _simulated(geometry).run(program)
        with pytest.raises(AssertionError):
            verify(row, results, MeshLayout(geometry), references[row.name])


def _biased_rowwise(row: StyleContract) -> StyleContract:
    import dataclasses

    from tests.neural.shared.parallel import styles_contract as contract

    def build() -> Any:
        return contract._one(
            "down_proj",
            contract._linear(contract.HIDDEN, contract.INTER, 12, bias=True),
        )

    return dataclasses.replace(row, build=build)


@pytest.mark.unit
def test_a_biased_rowwise_row_holds_too(references: dict[str, Results]) -> None:
    """The bias added once after the sum (the inference path's order) is
    the world-1 value: the mutation above is a real one."""
    row = _biased_rowwise(BY_NAME["rowwise"])
    program = functools.partial(run_row, row=row, tier=fragment_tier, geometry=TENSOR_2)
    verify(row, _simulated(TENSOR_2).run(program), MeshLayout(TENSOR_2), oracle(row))


class _KeywordExperts(torch.nn.Module):
    """An experts module of the library's signature, its output a function of
    the hidden input and the routing weights."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        return hidden_states * top_k_weights.sum()


class _KeywordRouter(torch.nn.Module):
    """A router of four experts: the first four features are the logits."""

    num_experts = 4

    def forward(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = hidden_states[..., :4]
        probabilities = torch.softmax(logits, dim=-1)
        # the library's router hands back the top-k scores, ``(tokens, top_k)``
        scores, indices = probabilities.topk(2, dim=-1)
        return logits, scores, indices


class _Variadic(torch.nn.Module):
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return args[0]


def _spelling_program(rank: int, c: Any) -> tuple[Any, ...]:
    """The experts and the router wrapped by the fragment tier at ``tp=2``,
    each called positionally and by keyword on the same inputs."""
    experts, router = _KeywordExperts(), _KeywordRouter()
    group = Group(axis="tensor", rank=c.rank("tensor"), size=c.size("tensor"))
    fragment_module._MoeExperts(c).install(experts, group, expert_parallel=False)
    fragment_module._EpRouter(c).install(router, group, expert_parallel=True)
    generator = torch.Generator().manual_seed(rank)
    hidden = torch.randn(3, 4, generator=generator)
    weights = torch.randn(3, 2, generator=generator)
    index = torch.tensor([[0, 1], [2, 3], [1, 2]])

    def run(call: Callable[..., torch.Tensor]) -> tuple[torch.Tensor, ...]:
        h = hidden.clone().requires_grad_()
        w = weights.clone().requires_grad_()
        out = call(h, index, w)
        out.sum().backward()
        assert h.grad is not None and w.grad is not None
        return out.detach(), h.grad, w.grad

    positional = run(lambda h, i, w: experts(h, i, w))
    mixed = run(lambda h, i, w: experts(h, top_k_index=i, top_k_weights=w))
    keyword = run(
        lambda h, i, w: experts(hidden_states=h, top_k_index=i, top_k_weights=w)
    )
    routed = tuple(t.detach() for t in router(hidden))
    routed_keyword = tuple(t.detach() for t in router(hidden_states=hidden))
    return positional, mixed, keyword, routed, routed_keyword


@pytest.mark.unit
class TestArgumentSpelling:
    def test_a_keyword_call_is_the_positional_call_output_and_gradients(self) -> None:
        """The routing weights' gradient is summed over the group whichever
        way the caller spells the call; a wrap reading ``args[2]`` summed it
        for a positional call alone, and a keyword-spelled call kept each
        rank's own gradient with nothing to say so."""
        results = _simulated(ParallelGeometry(tensor=2)).run(_spelling_program)
        for rank, (positional, mixed, keyword, routed, routed_keyword) in enumerate(
            results
        ):
            for spelled in (mixed, keyword):
                for a, b in zip(positional, spelled):
                    assert torch.equal(a, b), (rank, "the spelling changed the call")
            for a, b in zip(routed, routed_keyword):
                assert torch.equal(a, b), (
                    rank,
                    "the router's spelling changed the call",
                )
        # Both calling conventions must sum the gradient over both ranks.
        assert torch.equal(results[0][0][2], results[1][0][2])

    def test_a_variadic_forward_is_refused_by_name(self) -> None:
        def program(rank: int, c: Any) -> str:
            group = Group(axis="tensor", rank=c.rank("tensor"), size=2)
            with pytest.raises(StyleError, match="cannot bind a variadic one"):
                fragment_module._EpRouter(c).install(
                    _Variadic(), group, expert_parallel=True
                )
            return "refused"

        assert _simulated(ParallelGeometry(tensor=2)).run(program) == ["refused"] * 2


@pytest.mark.unit
def test_fragment_styles_is_a_styles() -> None:
    from causalab.neural.engines.pytorch_hooks.styles import Styles
    from causalab.neural.shared.parallel.collective import SOLO

    assert isinstance(FragmentStyles(SOLO), Styles)


# --------------------------------------------------------------------------- #
# smoke: both tiers over gloo, held to each other and to the simulator
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module", params=GLOO_GEOMETRIES, ids=_ids)
def gloo_rows(
    request: pytest.FixtureRequest,
) -> tuple[
    ParallelGeometry, Sequence[dict[str, Results]], Sequence[dict[str, Results]]
]:
    """Every row of one geometry, once per tier, each in one spawned world."""
    geometry: ParallelGeometry = request.param
    fragment = GlooWorld(geometry).run(
        functools.partial(every_row, tier=fragment_tier, geometry=geometry)
    )
    dtensor = GlooWorld(geometry).run(
        functools.partial(every_row, tier=transformers_tier, geometry=geometry)
    )
    return geometry, fragment, dtensor


@pytest.mark.smoke
class TestBothTiersUnderGloo:
    def test_the_fragment_tier_holds(
        self, gloo_rows: tuple[Any, Any, Any], references: Mapping[str, Results]
    ) -> None:
        geometry, fragment, _ = gloo_rows
        verify_all(fragment, MeshLayout(geometry), references)

    def test_the_dtensor_tier_holds(
        self, gloo_rows: tuple[Any, Any, Any], references: Mapping[str, Results]
    ) -> None:
        geometry, _, dtensor = gloo_rows
        verify_all(dtensor, MeshLayout(geometry), references)

    def test_the_three_runs_are_bit_identical(
        self, gloo_rows: tuple[Any, Any, Any]
    ) -> None:
        """Simulator, ``gloo`` and DTensor over ``gloo``: one result."""
        geometry, fragment, dtensor = gloo_rows
        simulated = _simulated(geometry).run(
            functools.partial(every_row, tier=fragment_tier, geometry=geometry)
        )
        _equal_results(list(fragment), simulated, ("fragment/gloo vs simulated",))
        _equal_results(list(dtensor), simulated, ("dtensor/gloo vs simulated",))
