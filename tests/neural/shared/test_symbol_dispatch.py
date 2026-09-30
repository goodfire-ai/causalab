"""``SymbolDispatch`` (``shared/symbol_dispatch.py``): one stand-in at a
patched symbol, every patch a layer on it per thread — what the three
per-forward bindings of one kernel global need for the simulated world's
ranks, threads of one process handing off *inside* the collectives the taps
make (``docs/model_parallelism.md`` §10.2).

The properties, over interleavings hypothesis draws on the test thread and
a conductor replays on the participants deterministically (a turn is one
step of one thread's script; a call is two turns, the original function
handing the turn back from *inside* the call, as a collective does):

- **own layers only**: a call runs the calling thread's own stack, top
  down — every layer sees exactly the arguments its thread called with, in
  order, and the result carries every layer on that thread's stack and no
  other's;
- **the original for the rest**: a call from a thread with no layer in
  flight — never entered, or left — is the original's result;
- **one installation**: while any thread has a layer the symbol is the
  dispatch; once the last leaves it is the original object again, in
  whatever order the threads left.

On one thread: nested layers see the layer below as ``real`` and the
original at the bottom; leaving out of LIFO order is refused. The bottom
can be **rebound** (a loader binding a family for good) or **held** (the
torch-path guard, counted, the last release restoring), installed or not.
A dispatch fully left keeps forwarding to the original it captured — a
stale reference to it keeps working — while one nothing ever installed has
no ``real`` to give. Misuse is a typed [`DispatchMisuse`][causalab.neural.shared.symbol_dispatch.DispatchMisuse] naming the
symbol and the thread.
"""

from __future__ import annotations

import threading
import types
from typing import Any, Callable

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from causalab.neural.shared.kernels import torch_implementation
from causalab.neural.shared.symbol_dispatch import (
    AttributeSymbol,
    DispatchMisuse,
    Layer,
    LazySymbol,
    SymbolDispatch,
    dispatch_for,
)

pytestmark = pytest.mark.property

_SETTINGS = settings(
    deadline=None,
    max_examples=30,
    suppress_health_check=[HealthCheck.too_slow],
)

#: The longest any participant waits for its turn before the test fails
#: rather than hangs (a conductor bug, never a slow machine).
TIMEOUT = 10.0

#: The deepest stack a drawn script builds on one thread.
MAX_DEPTH = 3


def _original(x: int) -> int:
    return 2 * x


def _mark(participant: int, depth: int) -> int:
    """What the layer at ``depth`` on ``participant``'s stack adds."""
    return 1000 * (participant + 1) + 100 * (depth + 1)


# --------------------------------------------------------------------------- #
# the conductor: a drawn interleaving, replayed
# --------------------------------------------------------------------------- #


class Conductor:
    """Hands one turn at a time to a named participant; a participant runs
    one step and hands the turn back (``done``), or hands it back from
    inside a call (``checkpoint``) and waits for its next turn there."""

    def __init__(self) -> None:
        self.cv = threading.Condition()
        self.turn: int | None = None
        self.finished = False
        self.errors: list[BaseException] = []

    def give(self, participant: int) -> None:
        with self.cv:
            self.turn = participant
            self.cv.notify_all()
            if not self.cv.wait_for(lambda: self.turn is None or self.errors, TIMEOUT):
                raise AssertionError(f"participant {participant} never took its turn")

    def wait_turn(self, participant: int) -> None:
        with self.cv:
            if not self.cv.wait_for(
                lambda: self.turn == participant or self.finished or self.errors,
                TIMEOUT,
            ):
                raise AssertionError(f"participant {participant} was never scheduled")

    def done(self) -> None:
        with self.cv:
            self.turn = None
            self.cv.notify_all()

    def checkpoint(self, participant: int) -> None:
        self.done()
        self.wait_turn(participant)

    def fail(self, error: BaseException) -> None:
        with self.cv:
            self.errors.append(error)
            self.turn = None
            self.cv.notify_all()

    def finish(self) -> None:
        with self.cv:
            self.finished = True
            self.cv.notify_all()


#: A participant's script: the steps it runs in order.
Step = str
ENTER, CALL, LEAVE = "enter", "call", "leave"

#: The thread a call is made from is what the original function reads, so
#: it can check in with the conductor for *that* thread.
_CURRENT = threading.local()


def _run_script(
    participant: int,
    script: list[Step],
    conductor: Conductor,
    dispatch: SymbolDispatch,
    module: Any,
    seen: dict[str, Any],
) -> None:
    """One participant: each of its layers records its arguments under its
    depth and adds its mark to what ``real`` returns; each call goes through
    the symbol as a caller would (``module.kernel``), never through the
    dispatch object."""
    calls = 0
    layers: list[Layer] = []
    _CURRENT.participant = participant

    def tap_at(depth: int) -> Callable[[int], int]:
        def tap(x: int) -> int:
            seen["captured"][participant][depth].append(x)
            return dispatch.real(x) + _mark(participant, depth)

        return tap

    try:
        for step in script:
            conductor.wait_turn(participant)
            if conductor.errors or conductor.finished:
                return
            if step == ENTER:
                layers.append(dispatch.enter(tap_at(len(layers))))
            elif step == LEAVE:
                dispatch.leave(layers.pop())
            else:
                argument = 10 * participant + calls
                calls += 1
                seen["results"][participant].append((argument, module.kernel(argument)))
            conductor.done()
    except BaseException as error:  # re-raised on the test thread
        conductor.fail(error)


def _original_factory(conductor: Conductor) -> Callable[[int], int]:
    def original(x: int) -> int:
        participant = getattr(_CURRENT, "participant", None)
        if participant is not None:  # the test thread's own calls do not hand off
            conductor.checkpoint(participant)
        return _original(x)

    return original


@st.composite
def interleavings(draw: st.DrawFn) -> tuple[list[list[Step]], list[int]]:
    """``n`` scripts — a layering participant walks its stack up and down
    (at most `MAX_DEPTH` deep) with calls in between, leaves every
    layer, then may call from outside; a bystander only calls — and one
    merge of them preserving each script's order. Every call is two turns
    (the original's checkpoint)."""
    n = draw(st.integers(2, 4))
    scripts: list[list[Step]] = []
    for participant in range(n):
        if participant > 0 and draw(st.booleans()):
            scripts.append([CALL] * draw(st.integers(1, 2)))
            continue
        steps: list[Step] = [ENTER]
        depth = 1
        for _ in range(draw(st.integers(0, 6))):
            choice = draw(st.sampled_from([ENTER, CALL, LEAVE]))
            if choice == ENTER and depth < MAX_DEPTH:
                depth += 1
            elif choice == LEAVE and depth > 0:
                depth -= 1
            else:
                choice = CALL
            steps.append(choice)
        steps.extend([LEAVE] * depth)
        steps.extend([CALL] * draw(st.integers(0, 1)))
        scripts.append(steps)
    turns: list[int] = []
    for participant, script in enumerate(scripts):
        turns.extend([participant] * sum(2 if s == CALL else 1 for s in script))
    order = draw(st.permutations(turns))
    return scripts, order


def _replay(
    scripts: list[list[Step]], order: list[int]
) -> tuple[dict[str, Any], list[bool], Any, SymbolDispatch]:
    """Run the interleaving; returns what the participants saw, whether the
    symbol was the dispatch after every turn, the module and the dispatch."""
    conductor = Conductor()
    module = types.ModuleType("fake_modeling")
    module.kernel = module.original = _original_factory(conductor)  # type: ignore[attr-defined]
    dispatch = SymbolDispatch(AttributeSymbol(module, "kernel"))
    seen: dict[str, Any] = {
        "captured": [[[] for _ in range(MAX_DEPTH)] for _ in scripts],
        "results": [[] for _ in scripts],
    }
    threads = [
        threading.Thread(
            target=_run_script,
            args=(p, script, conductor, dispatch, module, seen),
            name=f"participant-{p}",
            daemon=True,
        )
        for p, script in enumerate(scripts)
    ]
    for thread in threads:
        thread.start()
    installed: list[bool] = []
    try:
        for participant in order:
            conductor.give(participant)
            if conductor.errors:
                break
            installed.append(module.kernel is dispatch)
    finally:
        conductor.finish()
        for thread in threads:
            thread.join(TIMEOUT)
    if conductor.errors:
        raise conductor.errors[0]
    return seen, installed, module, dispatch


def _expected(
    scripts: list[list[Step]], order: list[int]
) -> tuple[list[list[list[int]]], list[list[tuple[int, int]]], list[bool]]:
    """The oracle, replayed on the test thread: which arguments each layer
    sees, each call's result (the original plus every mark on the calling
    thread's stack), and whether anyone has a layer after each turn."""
    position = [0] * len(scripts)
    mid_call = [False] * len(scripts)
    depth = [0] * len(scripts)
    calls = [0] * len(scripts)
    captured: list[list[list[int]]] = [[[] for _ in range(MAX_DEPTH)] for _ in scripts]
    results: list[list[tuple[int, int]]] = [[] for _ in scripts]
    anyone: list[bool] = []
    for participant in order:
        if mid_call[participant]:
            mid_call[participant] = False
            position[participant] += 1
        else:
            step = scripts[participant][position[participant]]
            if step == ENTER:
                depth[participant] += 1
                position[participant] += 1
            elif step == LEAVE:
                depth[participant] -= 1
                position[participant] += 1
            else:
                argument = 10 * participant + calls[participant]
                calls[participant] += 1
                result = _original(argument)
                for d in range(depth[participant]):
                    captured[participant][d].append(argument)
                    result += _mark(participant, d)
                results[participant].append((argument, result))
                mid_call[participant] = True
        anyone.append(any(d > 0 for d in depth))
    return captured, results, anyone


class TestInterleavedThreads:
    @_SETTINGS
    @given(interleavings())
    def test_each_call_runs_the_calling_threads_own_stack(
        self, drawn: tuple[list[list[Step]], list[int]]
    ) -> None:
        scripts, order = drawn
        seen, installed, module, dispatch = _replay(scripts, order)
        captured, results, anyone = _expected(scripts, order)
        assert seen["captured"] == captured
        assert seen["results"] == results
        # one installation: the symbol is the dispatch exactly while someone has a layer
        assert installed == anyone
        # and the original object once the last has left
        assert module.kernel is module.original
        assert not dispatch.installed
        assert module.kernel(3) == 6

    def test_the_ci_interleaving_leaves_the_original_and_runs_the_late_threads_layers(
        self,
    ) -> None:
        """Two threads each two layers deep (the short path and the tap); the
        first leaves both while the second is still inside, then the second
        calls: both of its own layers run, and the symbol is the original
        object once it has left."""
        scripts = [[ENTER, ENTER, LEAVE, LEAVE], [ENTER, ENTER, CALL, LEAVE, LEAVE]]
        order = [0, 1, 0, 1, 0, 0, 1, 1, 1, 1]
        seen, installed, module, dispatch = _replay(scripts, order)
        assert seen["results"][1] == [(10, 20 + _mark(1, 0) + _mark(1, 1))]
        assert seen["captured"][1][:2] == [[10], [10]]
        assert installed == [True] * 9 + [False]
        assert module.kernel is module.original
        assert not dispatch.installed

    def test_the_original_is_the_same_object_after_out_of_order_leaves(self) -> None:
        scripts = [[ENTER, LEAVE], [ENTER, LEAVE]]
        # 0 enters, 1 enters, 0 leaves (not the nested order), 1 leaves
        _seen, installed, module, dispatch = _replay(scripts, [0, 1, 0, 1])
        assert installed == [True, True, True, False]
        assert module.kernel is module.original
        assert not dispatch.installed


# --------------------------------------------------------------------------- #
# one thread, many layers
# --------------------------------------------------------------------------- #


def _dispatch() -> tuple[Any, SymbolDispatch]:
    module = types.ModuleType("fake_modeling")
    module.kernel = module.original = _original  # type: ignore[attr-defined]
    return module, SymbolDispatch(AttributeSymbol(module, "kernel"))


class TestLayersOnOneThread:
    def test_nested_layers_see_the_layer_below_as_real_and_the_original_at_the_bottom(
        self,
    ) -> None:
        module, dispatch = _dispatch()
        reals: dict[str, Any] = {}

        def outer(x: int) -> int:
            reals["outer"] = dispatch.real
            return dispatch.real(x) + 1

        def inner(x: int) -> int:
            reals["inner"] = dispatch.real
            return dispatch.real(x) + 10

        with dispatch.tapped(outer) as outer_layer, dispatch.tapped(inner):
            assert module.kernel(1) == 2 + 1 + 10
            assert reals["outer"] is module.original
            assert isinstance(reals["inner"], Layer)
            assert reals["inner"] is outer_layer
            assert reals["inner"].patched is outer
            # read outside a call: the view below the top layer
            assert dispatch.real is outer_layer
        assert module.kernel is module.original
        assert dispatch.real is module.original

    def test_the_tapped_context_hands_back_the_layer(self) -> None:
        _module, dispatch = _dispatch()

        def tap(x: int) -> int:  # pragma: no cover — never called
            return x

        with dispatch.tapped(tap) as layer:
            assert isinstance(layer, Layer) and layer.patched is tap
            assert layer.below is None
            with dispatch.tapped(tap) as above:
                assert above.below is layer

    def test_leaving_the_outer_layer_while_the_inner_is_up_is_refused(self) -> None:
        module, dispatch = _dispatch()
        outer = dispatch.enter(lambda x: dispatch.real(x) + 1)
        inner = dispatch.enter(lambda x: dispatch.real(x) + 10)
        with pytest.raises(DispatchMisuse) as err:
            dispatch.leave(outer)
        assert err.value.kind == "leave_out_of_order"
        assert "fake_modeling.kernel" in str(err.value)
        assert threading.current_thread().name in str(err.value)
        assert module.kernel(1) == 13  # both layers still answer
        dispatch.leave(inner)
        dispatch.leave(outer)
        assert module.kernel is module.original
        assert not dispatch.installed

    def test_leaving_a_layer_never_entered_is_refused(self) -> None:
        module, dispatch = _dispatch()
        _other_module, other = _dispatch()
        foreign = other.enter(lambda x: x)
        with pytest.raises(DispatchMisuse) as err:
            dispatch.leave(foreign)
        assert err.value.kind == "leave_without_enter"
        assert not dispatch.installed
        other.leave(foreign)
        with pytest.raises(DispatchMisuse) as err:
            other.leave(foreign)  # twice
        assert err.value.kind == "leave_without_enter"
        assert module.kernel is module.original

    def test_leaving_on_another_thread_than_entered_is_refused(self) -> None:
        """A layer belongs to the thread that entered it; another thread
        leaving would drop a count it never raised."""
        module, dispatch = _dispatch()
        errors: list[DispatchMisuse] = []

        def other(layer: Layer) -> None:
            try:
                dispatch.leave(layer)
            except DispatchMisuse as err:
                errors.append(err)

        with dispatch.tapped(lambda x: -x) as layer:
            thread = threading.Thread(target=other, args=(layer,))
            thread.start()
            thread.join(TIMEOUT)
            assert dispatch.installed and module.kernel is dispatch
        assert [err.kind for err in errors] == ["leave_without_enter"]

    def test_a_tap_that_raises_still_leaves(self) -> None:
        module, dispatch = _dispatch()
        with pytest.raises(ZeroDivisionError):
            with dispatch.tapped(lambda x: x):
                raise ZeroDivisionError
        assert not dispatch.installed
        assert module.kernel(1) == 2


class TestThreadsApart:
    def test_a_thread_with_no_layers_runs_the_original_while_another_is_inside(
        self,
    ) -> None:
        module, dispatch = _dispatch()
        seen: dict[str, Any] = {}

        def bystander() -> None:
            seen["result"] = module.kernel(2)
            seen["real"] = dispatch.real

        with dispatch.tapped(lambda x: -x):
            thread = threading.Thread(target=bystander)
            thread.start()
            thread.join(TIMEOUT)
            assert module.kernel(2) == -2
        assert seen == {"result": 4, "real": module.original}


# --------------------------------------------------------------------------- #
# after everyone left, and before anyone entered
# --------------------------------------------------------------------------- #


class TestStaleAndNeverInstalled:
    def test_a_stale_reference_to_the_dispatch_keeps_running_the_original(
        self,
    ) -> None:
        module, dispatch = _dispatch()
        with dispatch.tapped(lambda x: -x):
            stale = module.kernel
        assert stale is dispatch
        assert not dispatch.installed
        assert stale(3) == 6
        assert dispatch.real is module.original

    def test_a_dispatch_nothing_ever_installed_has_no_real(self) -> None:
        _module, dispatch = _dispatch()
        with pytest.raises(DispatchMisuse) as err:
            dispatch.real
        assert err.value.kind == "not_installed"
        with pytest.raises(DispatchMisuse) as err:
            dispatch(3)
        assert err.value.kind == "not_installed"

    def test_the_refusal_carries_the_symbol_and_the_thread_as_fields(self) -> None:
        """What the message spells is also on the error, typed, for a caller
        that routes on it."""
        _module, dispatch = _dispatch()
        with pytest.raises(DispatchMisuse) as err:
            dispatch.real
        assert err.value.symbol == "fake_modeling.kernel"
        assert err.value.thread == threading.current_thread().name
        assert err.value.kind == "not_installed"

    def test_entering_finds_the_dispatch_itself_at_the_symbol(self) -> None:
        """A stale set of the dispatch (or a wrapper over it) at the symbol
        is not captured as the original: the held original stands."""
        module, dispatch = _dispatch()
        with dispatch.tapped(lambda x: -x):
            pass
        module.kernel = dispatch  # type: ignore[attr-defined]
        with dispatch.tapped(lambda x: dispatch.real(x) + 1):
            assert module.kernel(1) == 3
        assert module.kernel is module.original

        def wrapper(x: int) -> int:  # pragma: no cover — never called
            return dispatch(x)

        wrapper.__wrapped__ = dispatch  # type: ignore[attr-defined]
        module.kernel = wrapper  # type: ignore[attr-defined]
        with dispatch.tapped(lambda x: dispatch.real(x) + 1):
            assert module.kernel(1) == 3
        assert module.kernel is module.original

    def test_a_never_installed_dispatch_finding_itself_is_refused(self) -> None:
        module, dispatch = _dispatch()
        module.kernel = dispatch  # type: ignore[attr-defined]
        with pytest.raises(DispatchMisuse) as err:
            dispatch.enter(lambda x: x)
        assert err.value.kind == "found_itself"
        assert not dispatch.installed


# --------------------------------------------------------------------------- #
# the bottom: rebind and hold
# --------------------------------------------------------------------------- #


def _triple(x: int) -> int:
    return 3 * x


class TestTheBottom:
    def test_bound_is_the_symbols_value_outside_and_the_original_inside(self) -> None:
        module, dispatch = _dispatch()
        assert dispatch.bound is module.original
        module.kernel = _triple  # type: ignore[attr-defined]
        assert dispatch.bound is _triple
        with dispatch.tapped(lambda x: -x):
            assert module.kernel is dispatch
            assert dispatch.bound is _triple
        assert module.kernel is _triple

    def test_rebind_while_installed_changes_what_the_bottom_sees_and_the_symbol_after(
        self,
    ) -> None:
        module, dispatch = _dispatch()
        with dispatch.tapped(lambda x: dispatch.real(x) + 1):
            assert module.kernel(1) == 3
            dispatch.rebind(_triple)
            assert module.kernel(1) == 4
            assert dispatch.bound is _triple and dispatch.real is _triple
            assert module.kernel is dispatch
        assert module.kernel is _triple

    def test_rebind_while_not_installed_sets_the_symbol(self) -> None:
        module, dispatch = _dispatch()
        dispatch.rebind(_triple)
        assert module.kernel is _triple
        assert dispatch.bound is _triple
        with dispatch.tapped(lambda x: dispatch.real(x) + 1):
            assert module.kernel(1) == 4
        assert module.kernel is _triple

    def test_a_hold_swaps_the_bottom_until_the_last_release(self) -> None:
        module, dispatch = _dispatch()
        dispatch.hold(_triple)
        assert module.kernel is _triple
        dispatch.hold(_triple)
        dispatch.release()
        assert module.kernel is _triple
        dispatch.release()
        assert module.kernel is module.original

    def test_a_hold_under_a_layer_is_what_the_bottom_computes_with(self) -> None:
        module, dispatch = _dispatch()
        with dispatch.tapped(lambda x: dispatch.real(x) + 1):
            with dispatch.holding(_triple):
                assert module.kernel is dispatch
                assert dispatch.bound is _triple
                assert module.kernel(1) == 4
            assert module.kernel(1) == 3
        assert module.kernel is module.original

    def test_a_layer_entered_under_a_hold_outlives_its_release(self) -> None:
        """The last release restores the bottom even while a layer stands
        on it — the layer's ``real`` follows."""
        module, dispatch = _dispatch()
        dispatch.hold(_triple)
        layer = dispatch.enter(lambda x: dispatch.real(x) + 1)
        assert module.kernel(1) == 4
        dispatch.release()
        assert module.kernel(1) == 3
        dispatch.leave(layer)
        assert module.kernel is module.original

    def test_a_second_hold_at_another_value_is_refused(self) -> None:
        _module, dispatch = _dispatch()
        dispatch.hold(_triple)
        with pytest.raises(DispatchMisuse) as err:
            dispatch.hold(lambda x: x)
        assert err.value.kind == "hold_disagrees"
        dispatch.release()

    def test_a_release_without_a_hold_is_refused(self) -> None:
        _module, dispatch = _dispatch()
        with pytest.raises(DispatchMisuse) as err:
            dispatch.release()
        assert err.value.kind == "release_without_hold"


# --------------------------------------------------------------------------- #
# reaching through: ``below`` and the ``__wrapped__`` chain
# --------------------------------------------------------------------------- #


class TestReachingThrough:
    def test_below_runs_real_as_it_is_at_call_time(self) -> None:
        """A wrapper built before its layer is entered calls ``below``; when
        it runs, that is the layer beneath it — and the rebound bottom."""
        module, dispatch = _dispatch()
        below = dispatch.below

        def wrapper(x: int) -> int:
            return below(x) + 100

        wrapper.__wrapped__ = below  # type: ignore[attr-defined]
        with dispatch.tapped(lambda x: dispatch.real(x) + 1):
            with dispatch.tapped(wrapper):
                assert module.kernel(1) == 2 + 1 + 100
                dispatch.rebind(_triple)
                assert module.kernel(1) == 3 + 1 + 100
        assert module.kernel is _triple

    def test_the_wrapped_chain_through_the_dispatch_reaches_the_original(
        self,
    ) -> None:
        module, dispatch = _dispatch()

        def wrapper(x: int) -> int:  # pragma: no cover — never called
            return dispatch.below(x)

        wrapper.__wrapped__ = dispatch.below  # type: ignore[attr-defined]
        assert torch_implementation(wrapper) is module.original
        assert dispatch.__wrapped__ is module.original
        with dispatch.tapped(wrapper) as layer:
            assert torch_implementation(module.kernel) is module.original
            assert torch_implementation(layer) is module.original
            dispatch.rebind(_triple)
            assert torch_implementation(wrapper) is _triple


class TestSymbols:
    def test_a_lazy_symbol_resolves_once_at_first_use(self) -> None:
        module = types.ModuleType("late_modeling")
        module.kernel = lambda x: x  # type: ignore[attr-defined]
        resolved = 0

        def resolve() -> AttributeSymbol:
            nonlocal resolved
            resolved += 1
            return AttributeSymbol(module, "kernel")

        dispatch = SymbolDispatch(LazySymbol(resolve))
        assert resolved == 0
        with dispatch.tapped(lambda x: -x):
            assert module.kernel is dispatch
        with dispatch.tapped(lambda x: -x):
            pass
        assert resolved == 1
        assert module.kernel(2) == 2

    def test_dispatch_for_is_one_dispatch_per_owner_and_name(self) -> None:
        module = types.ModuleType("registered_modeling")
        module.kernel = module.other = _original  # type: ignore[attr-defined]
        found = dispatch_for(module, "kernel")
        assert dispatch_for(module, "kernel") is found
        assert dispatch_for(module, "other") is not found
        assert str(found.symbol) == "registered_modeling.kernel"
        with found.tapped(lambda x: -x):
            assert module.kernel is found and module.other is _original
