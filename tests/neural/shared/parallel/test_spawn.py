"""Test launcher environment inheritance, process exits and port reservation.

Set ``OMP_NUM_THREADS=1`` before children import torch to avoid CPU
oversubscription, and select the loopback interface for gloo when available.
Explicit parent settings take precedence and the parent environment is
restored afterward.

Torch-free child programs check return codes, tracebacks and signal exits.
The real model spawn is covered by
``tests/neural/engines/pytorch_hooks/test_data_parallel_run.py``.
"""

from __future__ import annotations

import logging
import os
import signal
import socket
import sys
from typing import Any, Sequence

import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from causalab.neural.shared.parallel.launcher import GROUP_VARIABLES, LAUNCHER_VARIABLE
from causalab.neural.shared.parallel.spawn import (
    TRACEBACK_LIMIT,
    CHILD_THREADS,
    GLOO_INTERFACE_VARIABLE,
    THREADS_VARIABLE,
    PortHold,
    child_environment,
    loopback_interface,
    reserve_port,
    spawn,
    spawn_environment,
)
from causalab.neural.shared.parallel import spawn as spawn_module
from causalab.protocol.parallel import ParallelGeometry, format_geometry


@pytest.mark.unit
def test_a_child_carries_the_group_variables_and_the_mark() -> None:
    assert child_environment(1, 2, 29500) == {
        "WORLD_SIZE": "2",
        "RANK": "1",
        "LOCAL_RANK": "1",
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": "29500",
        LAUNCHER_VARIABLE: "spawned",
    }


@pytest.mark.unit
def test_a_bare_parent_sets_one_intra_op_thread_for_its_children() -> None:
    assert spawn_environment({}, interfaces=()) == {THREADS_VARIABLE: CHILD_THREADS}
    assert spawn_environment({"HOME": "/x"}, interfaces=()) == {
        THREADS_VARIABLE: CHILD_THREADS
    }
    assert THREADS_VARIABLE == "OMP_NUM_THREADS" and CHILD_THREADS == "1"


@pytest.mark.unit
def test_a_count_the_parent_set_is_left_to_the_children_to_inherit() -> None:
    assert spawn_environment({THREADS_VARIABLE: "4", "HOME": "/x"}, interfaces=()) == {}


@pytest.mark.unit
@settings(max_examples=30, deadline=None)
@given(
    world=st.integers(min_value=2, max_value=8),
    data=st.data(),
    port=st.integers(min_value=1024, max_value=65535),
    threads=st.none() | st.sampled_from(["1", "2", "8", "32"]),
)
def test_the_rule_over_every_child(world: int, data, port: int, threads) -> None:
    index = data.draw(st.integers(min_value=0, max_value=world - 1))
    env = child_environment(index, world, port)
    assert env["RANK"] == env["LOCAL_RANK"] == str(index)
    assert env["WORLD_SIZE"] == str(world) and env["MASTER_PORT"] == str(port)
    assert env[LAUNCHER_VARIABLE] == "spawned"
    assert set(env) == {*GROUP_VARIABLES, LAUNCHER_VARIABLE}
    inherited = {} if threads is None else {THREADS_VARIABLE: threads}
    assert (spawn_environment(inherited, interfaces=()) != {}) == (threads is None)


# --------------------------------------------------------------------------- #
# the gloo interface: the loopback, named by the parent
# --------------------------------------------------------------------------- #


@pytest.mark.unit
@pytest.mark.parametrize(
    ("interfaces", "loopback"),
    [
        (("lo", "eth0"), "lo"),
        (("en0", "lo0", "utun0"), "lo0"),
        (("eth0", "docker0"), None),
        ((), None),
        (("local0", "lopsided"), None),
    ],
)
def test_the_loopback_is_lo_followed_by_digits(interfaces, loopback) -> None:
    assert loopback_interface(interfaces) == loopback


@pytest.mark.unit
@settings(max_examples=30, deadline=None)
@example(suffix="x")
@example(suffix="0x")
@given(
    suffix=st.text(
        alphabet=st.characters(min_codepoint=32, max_codepoint=126), max_size=4
    )
)
def test_the_loopback_rule_over_every_suffix(suffix: str) -> None:
    """``lo`` and then digits or nothing — never letters, however few."""
    name = "lo" + suffix
    expected = name if suffix == "" or suffix.isdigit() else None
    assert loopback_interface([name]) == expected
    assert loopback_interface(["eth0", name]) == expected


@pytest.mark.unit
def test_a_bare_parent_names_the_loopback_for_the_childrens_gloo_groups() -> None:
    """Selecting loopback avoids hostname resolution for local gloo groups."""
    added = spawn_environment({}, interfaces=("en0", "lo0"))
    assert added[GLOO_INTERFACE_VARIABLE] == "lo0"
    assert added[THREADS_VARIABLE] == CHILD_THREADS
    assert GLOO_INTERFACE_VARIABLE == "GLOO_SOCKET_IFNAME"


@pytest.mark.unit
def test_an_interface_the_parent_set_is_left_to_the_children_to_inherit() -> None:
    added = spawn_environment({GLOO_INTERFACE_VARIABLE: "eth0"}, interfaces=("lo",))
    assert GLOO_INTERFACE_VARIABLE not in added


@pytest.mark.unit
def test_a_node_without_a_loopback_leaves_gloo_to_its_own_choice() -> None:
    assert GLOO_INTERFACE_VARIABLE not in spawn_environment({}, interfaces=("eth0",))


@pytest.mark.unit
def test_the_nodes_own_interfaces_are_read_by_default() -> None:
    """This node has a loopback (every one does): the default is set."""
    added = spawn_environment({})
    assert loopback_interface([added[GLOO_INTERFACE_VARIABLE]]) is not None


# --------------------------------------------------------------------------- #
# the rendezvous port: held by the parent for the world's life
# --------------------------------------------------------------------------- #

LINUX = sys.platform == "linux"


def _bind(port: int, *options: int) -> socket.socket:
    """A socket bound to the loopback at ``port`` with ``options`` set —
    raises ``OSError`` when the kernel refuses the address."""
    probe = socket.socket()
    for option in options:
        probe.setsockopt(socket.SOL_SOCKET, option, 1)
    try:
        probe.bind(("127.0.0.1", port))
    except OSError:
        probe.close()
        raise
    return probe


@pytest.mark.unit
@pytest.mark.skipif(not LINUX, reason="the hold is Linux's (module docstring)")
def test_a_held_port_refuses_a_squatter_and_admits_the_stores_listener() -> None:
    """A bound, non-listening socket reserves the port against explicit
    binders without ``SO_REUSEADDR``. The store listener uses that flag
    and can bind beside the hold."""
    hold = reserve_port()
    try:
        assert hold.held and 0 < hold.port < 65536
        with pytest.raises(OSError):
            _bind(hold.port)
        listener = _bind(hold.port, socket.SO_REUSEADDR)
        try:
            listener.listen()
            client = socket.create_connection(("127.0.0.1", hold.port), timeout=5.0)
            client.close()
        finally:
            listener.close()
    finally:
        hold.release()
    assert not hold.held
    # released, the port is anyone's again
    _bind(hold.port).close()


@pytest.mark.property
@given(low=st.integers(min_value=spawn_module.RESERVED_LOW + 1, max_value=65535))
@settings(max_examples=30, deadline=None)
def test_the_reserved_port_sits_below_the_ephemeral_floor(low: int) -> None:
    """An unlistening ``SO_REUSEADDR`` socket blocks explicit binders but
    not the kernel's ephemeral allocation during ``connect``. Reserve a
    port below the ephemeral pool to avoid that collision."""
    with spawn_module.reserve_port(ephemeral=low) as hold:
        assert spawn_module.RESERVED_LOW <= hold.port < low


@pytest.mark.unit
def test_a_taken_candidate_is_skipped_and_none_free_is_refused_by_name() -> None:
    taken = _bind(spawn_module.RESERVED_LOW + 7)
    try:
        port = int(taken.getsockname()[1])
        with spawn_module.reserve_port(candidates=[port, port + 1]) as hold:
            assert hold.port == port + 1
        with pytest.raises(spawn_module.NoFreePort) as err:
            spawn_module.reserve_port(candidates=[port])
        assert err.value.tried == 1 and str(port) not in str(err.value)
        assert "no free rendezvous port among 1 candidates" in str(err.value)
    finally:
        taken.close()


@pytest.mark.unit
@pytest.mark.parametrize(
    ("platform", "text", "expected"),
    [
        ("linux", "32768\t60999\n", 32768),
        ("linux", "40000 50000", 40000),
        ("linux", "garbage", spawn_module.EPHEMERAL_FLOOR["linux"]),
        (
            "linux",
            "1000\t2000",
            spawn_module.EPHEMERAL_FLOOR["linux"],
        ),  # below RESERVED_LOW
        ("darwin", "ignored", spawn_module.EPHEMERAL_FLOOR["darwin"]),
    ],
)
def test_the_ephemeral_floor_is_the_kernels_or_the_default(
    platform: str, text: str, expected: int
) -> None:
    assert spawn_module.ephemeral_low(platform, read=lambda: text) == expected


@pytest.mark.unit
def test_an_unreadable_range_file_is_the_default() -> None:
    def unreadable() -> str:
        raise OSError("no /proc here")

    assert (
        spawn_module.ephemeral_low("linux", read=unreadable)
        == spawn_module.EPHEMERAL_FLOOR["linux"]
    )


@pytest.mark.unit
def test_a_platform_that_cannot_hold_gets_the_probe_and_the_port_alone() -> None:
    """BSD (macOS) refuses a second exact binding unless *both* sockets set
    ``SO_REUSEPORT``, which the store's listener does not: there the parent
    probes and closes, as before, and the hold is released at birth."""
    hold = reserve_port(platform="darwin")
    assert not hold.held and 0 < hold.port < 65536
    _bind(hold.port).close()
    hold.release()  # idempotent
    assert not hold.held


@pytest.mark.unit
def test_release_is_idempotent_and_the_hold_is_its_own_context() -> None:
    with reserve_port() as hold:
        port = hold.port
        assert hold.port == port
    assert not hold.held
    hold.release()
    assert isinstance(hold, PortHold)


# --------------------------------------------------------------------------- #
# the spawn itself, torch-free
# --------------------------------------------------------------------------- #


def _entry_ok(argv) -> int:
    import os

    assert (
        os.environ["WORLD_SIZE"] == "2" and os.environ[LAUNCHER_VARIABLE] == "spawned"
    )
    assert list(argv) == ["run", "doc.json"]
    return 0


#: Appended to by the parent before a spawn: a child started as a fresh
#: interpreter imports this module anew and finds it empty.
_PARENT_MARK: list[int] = []

PP2 = ParallelGeometry(pipeline=2)


def _entry_environment(argv: Sequence[str]) -> int:
    """What a spawned child finds: the group variables of its rank, a real
    rendezvous port, the parent's additions, and a fresh interpreter."""
    env = os.environ
    assert env["WORLD_SIZE"] == "2" and env[LAUNCHER_VARIABLE] == "spawned"
    assert env["RANK"] == env["LOCAL_RANK"] and env["RANK"] in ("0", "1")
    assert env["MASTER_ADDR"] == "127.0.0.1"
    assert env["MASTER_PORT"].isdigit() and 0 < int(env["MASTER_PORT"]) < 65536
    assert env[THREADS_VARIABLE] == CHILD_THREADS
    if GLOO_INTERFACE_VARIABLE in env:
        assert loopback_interface([env[GLOO_INTERFACE_VARIABLE]]) is not None
    assert _PARENT_MARK == [], "the child is a fresh interpreter, not a fork"
    return 0


def _entry_status(argv: Sequence[str]) -> int:
    return 3 if os.environ["RANK"] == "1" else 0


def _entry_raise(argv: Sequence[str]) -> int:
    if os.environ["RANK"] == "1":
        raise RuntimeError("rank one boom")
    return 0


def _entry_signal(argv: Sequence[str]) -> int:
    if os.environ["RANK"] == "1":
        os.kill(os.getpid(), signal.SIGKILL)
    return 0


def _entry_sleep(argv: Sequence[str]) -> int:
    import time

    time.sleep(60.0)
    return 0


def _entry_ignore_term(argv: Sequence[str]) -> int:
    """Rank 0 will not go on ``SIGTERM`` — a rank wedged in a driver call
    or with its own handler; rank 1 fails at once."""
    import time

    if os.environ["RANK"] == "1":
        return 3
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    time.sleep(60.0)
    return 0


def _entry_huge_traceback(argv: Sequence[str]) -> int:
    if os.environ["RANK"] == "1":
        raise RuntimeError("wide " * (1 << 18))  # ~1.3 MiB, past any pipe buffer
    return 0


@pytest.mark.smoke
class TestSpawn:
    def test_every_child_returning_zero_is_zero(self) -> None:
        assert (
            spawn(ParallelGeometry(pipeline=2), ["run", "doc.json"], entry=_entry_ok)
            == 0
        )

    def test_the_children_see_the_parents_additions_and_the_parent_is_put_back(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The parent sets the thread count and the loopback before any child
        starts (module docstring: the child's OpenMP pool is sized as torch
        is imported), says so once, and takes them back once the children
        are done; each child is a fresh interpreter on a real port."""
        monkeypatch.delenv(THREADS_VARIABLE, raising=False)
        monkeypatch.delenv(GLOO_INTERFACE_VARIABLE, raising=False)
        added = spawn_environment(os.environ)
        assert added[THREADS_VARIABLE] == CHILD_THREADS
        caplog.set_level(logging.INFO, logger="causalab.neural.shared.parallel.spawn")
        _PARENT_MARK.append(1)
        try:
            assert spawn(PP2, ["run", "doc.json"], entry=_entry_environment) == 0
        finally:
            _PARENT_MARK.clear()
        assert THREADS_VARIABLE not in os.environ
        assert GLOO_INTERFACE_VARIABLE not in os.environ
        listing = " ".join(f"{k}={v}" for k, v in sorted(added.items()))
        # the handler is the root's: only this logger's records are the claim
        spoken = [
            r.getMessage()
            for r in caplog.records
            if r.name == "causalab.neural.shared.parallel.spawn"
        ]
        assert spoken == [
            f"each of the 2 spawned ranks runs with {listing} "
            "(set a variable to choose otherwise)"
        ]

    def test_a_child_exiting_by_status_is_the_spawns_status_named_on_stderr(
        self, capfd: pytest.CaptureFixture[str]
    ) -> None:
        assert spawn(PP2, ["run"], entry=_entry_status) == 3
        err = capfd.readouterr().err
        assert (
            f"refused: rank 1 of 2 exited with status 3 "
            f"(--parallel {format_geometry(PP2)})" in err
        )

    def test_a_child_that_raised_is_status_one_with_its_traceback(
        self, capfd: pytest.CaptureFixture[str]
    ) -> None:
        assert spawn(PP2, ["run"], entry=_entry_raise) == 1
        err = capfd.readouterr().err
        assert "refused: rank 1 of 2 raised:\n" in err
        assert "RuntimeError: rank one boom" in err
        assert f"(--parallel {format_geometry(PP2)})" in err

    def test_a_child_killed_by_a_signal_is_status_one_named_by_its_signal(
        self, capfd: pytest.CaptureFixture[str]
    ) -> None:
        """The status is ``1``, as torch's spawn reports a signal; the
        refusal names the signal, since a rank the OOM killer took must
        read differently from one that called ``sys.exit(1)``."""
        assert spawn(PP2, ["run"], entry=_entry_signal) == 1
        err = capfd.readouterr().err
        assert "refused: rank 1 of 2 was killed by SIGKILL" in err
        assert "exited with status" not in err

    def test_a_parent_interrupted_while_waiting_leaves_no_child_alive(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``KeyboardInterrupt`` (or ``SIGTERM``) reaching the parent inside
        ``_wait`` is the case an interactive Ctrl-C hides — the terminal
        signals the whole group — and a scheduler's cancel does not: the
        parent reaps its children on every way out."""
        from causalab.neural.shared.parallel import spawn as module

        seen: list[Any] = []

        def interrupted(children):
            seen.extend(children)
            for child in children:
                assert child.is_alive()
            raise KeyboardInterrupt

        monkeypatch.setattr(module, "_wait", interrupted)
        with pytest.raises(KeyboardInterrupt):
            spawn(PP2, ["run"], entry=_entry_sleep)
        assert len(seen) == 2
        assert not any(child.is_alive() for child in seen)
        assert all(child.exitcode is not None for child in seen)

    def test_the_hold_on_the_rendezvous_port_is_released_when_the_world_ends(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The parent holds the port from before the first child starts to
        after the last is reaped, and the children rendezvous on exactly
        that port; the hold goes in the same ``finally`` as the reaping."""
        from causalab.neural.shared.parallel import spawn as module

        holds: list[PortHold] = []
        original = module.reserve_port

        def recorded() -> PortHold:
            hold = original()
            holds.append(hold)
            return hold

        monkeypatch.setattr(module, "reserve_port", recorded)
        assert spawn(PP2, ["run", "doc.json"], entry=_entry_environment) == 0
        assert len(holds) == 1 and not holds[0].held
        _bind(holds[0].port).close()

    def test_an_interrupted_parent_releases_the_hold_too(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from causalab.neural.shared.parallel import spawn as module

        holds: list[PortHold] = []
        original = module.reserve_port

        def recorded() -> PortHold:
            hold = original()
            holds.append(hold)
            return hold

        def interrupted(children):
            raise KeyboardInterrupt

        monkeypatch.setattr(module, "reserve_port", recorded)
        monkeypatch.setattr(module, "_wait", interrupted)
        with pytest.raises(KeyboardInterrupt):
            spawn(PP2, ["run"], entry=_entry_sleep)
        assert len(holds) == 1 and not holds[0].held

    def test_a_sibling_ignoring_sigterm_is_killed_within_the_grace(
        self, monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
    ) -> None:
        import time

        from causalab.neural.shared.parallel import spawn as module

        monkeypatch.setattr(module, "TERMINATE_GRACE_S", 1.0)
        started = time.monotonic()
        assert spawn(PP2, ["run"], entry=_entry_ignore_term) == 3
        assert time.monotonic() - started < 30.0, "the refusal path is bounded"
        assert "refused: rank 1 of 2 exited with status 3" in capfd.readouterr().err

    def test_a_traceback_wider_than_the_pipe_is_cut_to_its_tail(
        self, capfd: pytest.CaptureFixture[str]
    ) -> None:
        """The parent reads the report only after the child exits, so a
        message past the pipe's buffer would block the child in ``send`` —
        the deadlock ``torch.multiprocessing.spawn`` shares."""
        assert spawn(PP2, ["run"], entry=_entry_huge_traceback) == 1
        err = capfd.readouterr().err
        assert "refused: rank 1 of 2 raised:\n… (traceback truncated to its last" in err
        assert "wide wide" in err  # the tail of the message, not its head
        assert len(err) < 3 * TRACEBACK_LIMIT

    def test_the_parent_imports_no_torch(self) -> None:
        """The whole point: torch is not in the parent's modules after the
        launcher and the spawn module are imported, in a fresh interpreter."""
        import subprocess
        import sys

        script = (
            "import sys; import causalab.neural.shared.parallel.launcher, "
            "causalab.neural.shared.parallel.spawn; "
            "sys.exit(1 if 'torch' in sys.modules else 0)"
        )
        assert (
            subprocess.run([sys.executable, "-c", script], check=False).returncode == 0
        )
