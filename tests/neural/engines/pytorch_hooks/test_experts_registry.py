"""The one installation rule over the experts registry entry
(``pytorch_hooks/experts_registry.py``): the first active enterer captures
what it found and installs, the last leaver restores it, under a lock — the
shape both the interface taps and the lean path install through, so two
threads (two simulated ranks) entering in any order leave the entry as it
was and never dispatch through a stale capture."""

from __future__ import annotations

import threading

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from causalab.neural.engines.pytorch_hooks.experts_registry import EntryInstall


def _library(*args: object) -> str:
    return "library"


def _replacement(*args: object) -> str:
    return "replacement"


def _other(*args: object) -> str:
    return "other"


def _install(registry: dict[str, object], replacement=_replacement) -> EntryInstall:
    return EntryInstall("grouped_mm", lambda: replacement, registry=lambda: registry)


@pytest.mark.unit
def test_what_installs_is_read_at_the_first_entry() -> None:
    """The getter is read when the first enterer installs, so a module-level
    function rebound before the entry (a test's spy over the copy) is what
    the registry holds, under its own identity."""
    registry: dict[str, object] = {"grouped_mm": _library}
    current: list[object] = [_replacement]
    install = EntryInstall("grouped_mm", lambda: current[0], registry=lambda: registry)
    with install.installed():
        assert registry["grouped_mm"] is _replacement
    current[0] = _other
    with install.installed():
        assert registry["grouped_mm"] is _other
    assert registry["grouped_mm"] is _library


@pytest.mark.unit
def test_the_first_enterer_installs_and_the_last_leaver_restores() -> None:
    registry: dict[str, object] = {"grouped_mm": _library}
    install = _install(registry)
    assert install.active == 0 and install.previous is None
    install.enter()
    assert registry["grouped_mm"] is _replacement and install.previous is _library
    install.enter()
    assert install.active == 2 and registry["grouped_mm"] is _replacement
    install.leave()
    assert registry["grouped_mm"] is _replacement, "one enterer is still inside"
    install.leave()
    assert registry["grouped_mm"] is _library and install.previous is None


@pytest.mark.unit
def test_a_leave_under_an_install_made_over_it_is_refused_and_the_entry_stays_usable() -> (
    None
):
    """Out of LIFO order — the outer leaves while the inner is still over it:
    refused by name, the inner's function left in place; the inner's own
    leave then restores the outer's function with the original beneath, and
    the outer's next enter recognises its own function rather than capturing
    it as what lies beneath."""
    from causalab.neural.engines.pytorch_hooks.experts_registry import EntryInstallError

    def original() -> str:
        return "original"

    def outer_fn() -> str:
        return "outer"

    def inner_fn() -> str:
        return "inner"

    registry = {"k": original}
    outer = EntryInstall("k", lambda: outer_fn, registry=lambda: registry)
    inner = EntryInstall("k", lambda: inner_fn, registry=lambda: registry)
    outer.enter()
    inner.enter()
    with pytest.raises(EntryInstallError, match="has not left"):
        outer.leave()
    assert registry["k"] is inner_fn and outer.active == 0
    assert outer.previous is original, "what the outer found is kept"
    inner.leave()
    assert registry["k"] is outer_fn, "the inner restored what it found"
    outer.enter()
    assert outer.previous is original, "the outer recognised its own function"
    outer.leave()
    assert registry["k"] is original


@pytest.mark.unit
def test_a_leave_without_an_enter_is_refused() -> None:
    install = _install({"grouped_mm": _library})
    with pytest.raises(RuntimeError, match="leave without a matching enter"):
        install.leave()


@pytest.mark.unit
def test_the_context_manager_is_enter_then_leave_even_on_an_exception() -> None:
    registry: dict[str, object] = {"grouped_mm": _library}
    install = _install(registry)
    with pytest.raises(ValueError), install.installed():
        assert registry["grouped_mm"] is _replacement
        raise ValueError("inside")
    assert registry["grouped_mm"] is _library and install.active == 0


@pytest.mark.unit
def test_two_installs_nest_lifo_each_restoring_what_it_found() -> None:
    """The lean path around the forward, the taps inside it: the inner
    install finds the outer's replacement and puts it back; the outer puts
    the library's back."""
    registry: dict[str, object] = {"grouped_mm": _library}
    outer, inner = _install(registry, _other), _install(registry, _replacement)
    with outer.installed():
        assert registry["grouped_mm"] is _other
        with inner.installed():
            assert registry["grouped_mm"] is _replacement and inner.previous is _other
        assert registry["grouped_mm"] is _other
    assert registry["grouped_mm"] is _library


@pytest.mark.property
@given(st.lists(st.booleans(), min_size=1, max_size=40))
@settings(max_examples=30, deadline=None)
def test_installed_iff_somebody_is_inside(steps: list[bool]) -> None:
    """Over any enter/leave sequence (a leave with nobody inside skipped):
    the entry is the replacement exactly while the count is above zero, and
    the library's the moment it falls to zero."""
    registry: dict[str, object] = {"grouped_mm": _library}
    install = _install(registry)
    inside = 0
    for enter in steps:
        if enter:
            install.enter()
            inside += 1
        elif inside:
            install.leave()
            inside -= 1
        assert install.active == inside
        expected = _replacement if inside else _library
        assert registry["grouped_mm"] is expected
    for _ in range(inside):
        install.leave()
    assert registry["grouped_mm"] is _library and install.previous is None


@pytest.mark.unit
def test_interleaved_threads_share_one_installation() -> None:
    """Thread A enters, B enters while A is inside, A leaves while B is
    still inside — the plain capture/restore's failure case: B must keep
    dispatching through the replacement and the library must come back only
    when B leaves."""
    registry: dict[str, object] = {"grouped_mm": _library}
    install = _install(registry)
    a_in, b_in, a_out, b_out = (threading.Event() for _ in range(4))
    seen: dict[str, object] = {}

    def rank_a() -> None:
        install.enter()
        a_in.set()
        b_in.wait(5.0)
        install.leave()
        seen["after_a_left"] = registry["grouped_mm"]
        a_out.set()

    def rank_b() -> None:
        a_in.wait(5.0)
        install.enter()
        seen["while_both_inside"] = registry["grouped_mm"]
        b_in.set()
        a_out.wait(5.0)
        seen["b_still_inside"] = registry["grouped_mm"]
        install.leave()
        b_out.set()

    threads = [threading.Thread(target=rank_a), threading.Thread(target=rank_b)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(5.0)
    assert a_out.is_set() and b_out.is_set(), "a thread did not leave"
    assert seen["while_both_inside"] is _replacement
    assert seen["after_a_left"] is _replacement
    assert seen["b_still_inside"] is _replacement
    assert registry["grouped_mm"] is _library and install.active == 0


@pytest.mark.unit
def test_many_threads_entering_at_once_count_every_one() -> None:
    """The count is read-modify-write under the lock: sixteen threads
    entering through one barrier and leaving through another leave the
    entry restored — an unlocked count would lose an increment and restore
    the library while a rank was still inside."""
    registry: dict[str, object] = {"grouped_mm": _library}
    install = _install(registry)
    n = 16
    entered = threading.Barrier(n, timeout=5.0)
    leaving = threading.Barrier(n, timeout=5.0)
    peak: list[int] = []

    def rank() -> None:
        install.enter()
        entered.wait()
        peak.append(install.active)
        assert registry["grouped_mm"] is _replacement
        leaving.wait()
        install.leave()

    threads = [threading.Thread(target=rank) for _ in range(n)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10.0)
    assert peak == [n] * n
    assert registry["grouped_mm"] is _library and install.active == 0
