"""``experts_interface_taps`` with two managers in flight at once — the
simulated world's ranks (``docs/model_parallelism.md`` §10.2, §10.4).

The manager patches the library's ``grouped_mm`` registry entry for one
forward; under expert parallelism the taps inside it make collectives,
where a simulated rank hands off — so a second rank's manager is entered
while the first is mid-forward. One entry, many managers (the attention
registry's rule): each rank's calls reach its own table (the tables are
keyed by module identity), the entry is the one dispatch function for as
long as either is inside, and it is the library's function again once the
last has left — in the order the threads came, not the nested order.
Before it, the second manager's wrapper called the first's as its "real"
function, and the first to leave restored the library's function under the
second's feet: its later calls silently missed their taps, and the last to
leave put the first's wrapper back for the rest of the process.
"""

from __future__ import annotations

import threading
from typing import Any

import pytest
import torch
import transformers.integrations.moe as moe

from causalab.neural.engines.pytorch_hooks import experts_interface
from causalab.neural.engines.pytorch_hooks.experts_interface import (
    ExpertsInstallError,
    ExpertsTap,
    experts_interface_taps,
)
from causalab.neural.engines.pytorch_hooks.experts_path import lean_experts_path
from causalab.neural.engines.pytorch_hooks.experts_registry import EntryInstallError

from .test_interiors_parallel import D_EXPERT, EXPERTS, HIDDEN, TOKENS, TOP_K, _Experts

# pyright: reportPrivateUsage=false

pytestmark = pytest.mark.unit

TIMEOUT = 10.0


def _rank(seed: int) -> tuple[Any, dict[str, torch.Tensor]]:
    """A rank's own experts module and its own inputs."""
    generator = torch.Generator().manual_seed(seed)
    module = _Experts(
        torch.randn((EXPERTS, 2 * D_EXPERT, HIDDEN), generator=generator),
        torch.randn((EXPERTS, HIDDEN, D_EXPERT), generator=generator),
    )
    table = torch.stack(
        [torch.randperm(EXPERTS, generator=generator)[:TOP_K] for _ in range(TOKENS)]
    )
    inputs = {
        "hidden": torch.randn((TOKENS, HIDDEN), generator=generator),
        "table": table,
        "weights": torch.softmax(torch.randn((TOKENS, TOP_K), generator=generator), -1),
    }
    return module, inputs


def _forward(module: Any, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    return moe.ALL_EXPERTS_FUNCTIONS["grouped_mm"](
        module, inputs["hidden"], inputs["table"], inputs["weights"]
    )


def test_interleaved_managers_route_each_thread_to_its_own_table() -> None:
    before = moe.ALL_EXPERTS_FUNCTIONS["grouped_mm"]
    a_in, b_called, a_out, b_out = (threading.Event() for _ in range(4))
    seen: dict[str, Any] = {"a": [], "b": [], "entry": {}}
    errors: list[BaseException] = []

    def recorder(into: list[torch.Tensor]) -> Any:
        def read(value: torch.Tensor, _idx: torch.Tensor) -> None:
            into.append(value.clone())

        return read

    def rank_a() -> None:
        try:
            module, inputs = _rank(0)
            taps = {id(module): (ExpertsTap("activation", read=recorder(seen["a"])),)}
            with experts_interface_taps(taps):
                a_in.set()
                assert b_called.wait(TIMEOUT)
                _forward(module, inputs)  # B is inside: A's call still reads A's table
                seen["entry"]["while_both_inside"] = moe.ALL_EXPERTS_FUNCTIONS[
                    "grouped_mm"
                ]
            a_out.set()  # A leaves first — not the nested order
        except BaseException as error:
            errors.append(error)
            a_in.set()
            a_out.set()

    def rank_b() -> None:
        try:
            assert a_in.wait(TIMEOUT)
            module, inputs = _rank(1)
            taps = {id(module): (ExpertsTap("activation", read=recorder(seen["b"])),)}
            with experts_interface_taps(taps):
                _forward(module, inputs)
                b_called.set()
                assert a_out.wait(TIMEOUT)
                seen["entry"]["after_a_left"] = moe.ALL_EXPERTS_FUNCTIONS["grouped_mm"]
                _forward(module, inputs)  # A has left: B's taps still answer
            b_out.set()
        except BaseException as error:
            errors.append(error)
            b_called.set()
            b_out.set()

    threads = [threading.Thread(target=rank_a), threading.Thread(target=rank_b)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(TIMEOUT)
    assert not errors, errors
    assert b_out.is_set(), "the ranks did not both leave"
    # one capture per forward, each thread's own: (tokens, top_k * d_expert)
    assert len(seen["a"]) == 1 and len(seen["b"]) == 2
    assert torch.equal(seen["b"][0], seen["b"][1])
    assert not torch.equal(seen["a"][0], seen["b"][0])
    assert seen["a"][0].shape == (TOKENS, TOP_K * D_EXPERT)
    # one installation while either is inside, the library's function after
    dispatch = experts_interface._dispatch_grouped_mm
    assert seen["entry"]["while_both_inside"] is dispatch
    assert seen["entry"]["after_a_left"] is dispatch
    assert moe.ALL_EXPERTS_FUNCTIONS["grouped_mm"] is before
    assert experts_interface._TABLES == [] and experts_interface._INSTALL.active == 0
    assert moe._grouped_linear is not experts_interface._GROUPED_LINEAR


def test_a_manager_with_no_taps_installs_nothing() -> None:
    before = moe.ALL_EXPERTS_FUNCTIONS["grouped_mm"]
    with experts_interface_taps({}):
        assert moe.ALL_EXPERTS_FUNCTIONS["grouped_mm"] is before
    assert experts_interface._INSTALL.active == 0


def test_a_module_outside_the_table_runs_the_library_function() -> None:
    module, inputs = _rank(3)
    plain = _forward(module, inputs)
    with experts_interface_taps({-1: ()}):
        assert torch.equal(_forward(module, inputs), plain)


def _silent(_value: torch.Tensor, _idx: torch.Tensor) -> None:
    pass


def test_a_module_named_by_two_entered_tables_is_refused_by_name() -> None:
    """The disjointness the module docstring rests on, checked: a module
    two entered managers both tap cannot be routed, and is refused rather
    than handed to whichever manager entered last."""
    module, inputs = _rank(4)
    taps = (ExpertsTap("activation", read=_silent),)
    with experts_interface_taps({id(module): taps}):
        with experts_interface_taps({id(module): taps}):
            with pytest.raises(ExpertsInstallError, match="tapped by 2 entered"):
                _forward(module, inputs)
        # one manager left: the module routes again
        _forward(module, inputs)
    assert experts_interface._TABLES == [] and experts_interface._INSTALL.active == 0


def test_a_table_leaves_by_identity_and_a_live_addition_to_the_outer_still_routes() -> (
    None
):
    """Two managers over equal but distinct tables: the inner's leaving
    removes the inner's table, not the outer's equal one — the mapping is
    read live, so an entry added to the outer afterwards is still found."""
    module, inputs = _rank(5)
    other, other_inputs = _rank(6)
    seen: list[torch.Tensor] = []
    outer: dict[int, tuple[ExpertsTap, ...]] = {
        id(module): (ExpertsTap("activation", read=_silent),)
    }
    inner = dict(outer)
    assert inner == outer and inner is not outer
    with experts_interface_taps(outer):
        with experts_interface_taps(inner):
            pass
        assert experts_interface._TABLES[0] is outer
        outer[id(other)] = (
            ExpertsTap("activation", read=lambda value, _idx: seen.append(value)),
        )
        _forward(other, other_inputs)
    assert len(seen) == 1, "the outer manager's live addition was routed"
    assert experts_interface._TABLES == []


def test_a_window_leaving_under_an_install_made_over_it_is_refused_by_name() -> None:
    """The taps entered before the lean path and leaving before it: the
    registry then holds the lean wrapper over this dispatch, and restoring
    what the taps found would bypass it. Refused by name; the dispatch stays
    with the library's function beneath, so the lean path's own restore
    leaves a working entry and the next manager reuses it cleanly."""
    module, inputs = _rank(7)
    before = moe.ALL_EXPERTS_FUNCTIONS["grouped_mm"]
    plain = _forward(module, inputs)
    taps = experts_interface_taps(
        {id(module): (ExpertsTap("activation", read=_silent),)}
    )
    taps.__enter__()
    lean = lean_experts_path()
    lean.__enter__()
    try:
        with pytest.raises(EntryInstallError, match="has not left"):
            taps.__exit__(None, None, None)
    finally:
        lean.__exit__(None, None, None)
    dispatch = experts_interface._dispatch_grouped_mm
    assert moe.ALL_EXPERTS_FUNCTIONS["grouped_mm"] is dispatch, "left by the lean path"
    assert experts_interface._INSTALL.active == 0
    assert torch.equal(_forward(module, inputs), plain), "the library is still beneath"
    # the next manager finds the dispatch, keeps what is beneath it, and its
    # leaving puts the library's function back
    with experts_interface_taps({-1: ()}):
        assert torch.equal(_forward(module, inputs), plain)
    assert moe.ALL_EXPERTS_FUNCTIONS["grouped_mm"] is before


def test_the_dispatch_called_with_no_manager_entered_is_refused_by_name() -> None:
    """A stale reference to the dispatch after the last manager left finds
    nothing beneath it: a named error, not a ``TypeError`` on ``None``."""
    module, inputs = _rank(8)
    with experts_interface_taps({-1: ()}):
        pass
    with pytest.raises(ExpertsInstallError, match="no manager entered"):
        experts_interface._dispatch_grouped_mm(
            module, inputs["hidden"], inputs["table"], inputs["weights"]
        )


def test_the_sentinel_masks_run_only_where_a_sentinel_can_appear(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Whether the routing table can hold the EP sentinel is a rank-uniform
    fact of the module (``experts_path.may_route_to_sentinels``), decided
    once per call: a module outside any expert axis takes the plain gathers
    and pays for no mask, and the masked path is bit-identical to it where
    the mask would be all-False — so the world-1 output does not depend on
    the predicate's answer."""
    module, inputs = _rank(2)
    assert not experts_interface.may_route_to_sentinels(module)
    seen: dict[str, list[torch.Tensor]] = {"plain": [], "masked": []}

    def recorder(into: list[torch.Tensor]) -> Any:
        def read(value: torch.Tensor, _idx: torch.Tensor) -> None:
            into.append(value.clone())

        return read

    outputs: dict[str, torch.Tensor] = {}
    for label, answer in (("plain", False), ("masked", True)):
        monkeypatch.setattr(
            experts_interface, "may_route_to_sentinels", lambda _module: answer
        )
        taps = {id(module): (ExpertsTap("neuron_output", read=recorder(seen[label])),)}
        with experts_interface_taps(taps):
            outputs[label] = _forward(module, inputs)
    assert torch.equal(outputs["plain"], outputs["masked"])
    assert torch.equal(seen["plain"][0], seen["masked"][0])
    assert seen["plain"][0].shape == (TOKENS, TOP_K * D_EXPERT)
