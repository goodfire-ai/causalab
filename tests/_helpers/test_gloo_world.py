"""``GlooWorld`` rendezvous on the launcher's held port (``spawn.reserve_port``,
``docs/model_parallelism.md`` §3 "the rendezvous port is held"): one
implementation of the port choice for the production spawn and the test
worlds alike. Holding the port prevents another process from taking it
between selection and ``init_process_group``. The smoke uses two real ranks.
"""

from __future__ import annotations

import os
import socket
from typing import Any

import pytest

from causalab.neural.shared.parallel.collective import Collective
from causalab.neural.shared.parallel.spawn import PortHold
from causalab.protocol.parallel import ParallelGeometry

from tests._helpers import gloo_world

pytestmark = pytest.mark.smoke


def _port_program(rank: int, collective: Collective) -> Any:
    """Each rank reports the port its group rendezvoused on."""
    return int(os.environ["MASTER_PORT"])


def test_the_world_rendezvous_on_the_held_port_and_releases_it_after(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    holds: list[PortHold] = []
    original = gloo_world.reserve_port

    def recorded() -> PortHold:
        hold = original()
        holds.append(hold)
        return hold

    monkeypatch.setattr(gloo_world, "reserve_port", recorded)
    results = gloo_world.GlooWorld(ParallelGeometry(tensor=2)).run(_port_program)
    assert len(holds) == 1
    hold = holds[0]
    assert results == [hold.port, hold.port]
    assert not hold.held  # released in the run's own scope
    # anyone's again: a bind beside the store's TIME_WAIT connections needs
    # the flag, which a still-held port would refuse on BSD (Linux admits
    # it beside a hold too — the ``held`` line above is the pin there)
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", hold.port))
