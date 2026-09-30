"""The environment a launched world must hold alike on every rank
(``docs/model_parallelism.md`` §3 "never branch on rank").

Four settings are read from each process's own environment and steer what
that rank does inside the group: the watchdog's two durations become the
default process group's and every mesh group's timeout and the heartbeat's
grace (``watchdog.Settings``); the context waiver decides whether a hybrid
tower is refused under ``cp > 1`` before the first collective
(``protocol/parallel.py:experimental_context``); the gradient-agreement
tolerance adds an ``all_gather`` **per parameter** to the training step on
the ranks that set it (``agreements.configured_agreement``). A world whose
ranks disagree on any of them does not refuse — it diverges: the ranks reach
different collective sequences, or time out at different moments, and the
survivor's failure names the backend, not the variable. An ordinary way to
get there is a two-node ``torchrun`` join whose per-node launch scripts
export the variable on one node only.

So the raw values are **agreed at the rendezvous**: every rank writes its
own under [`environment_key`][] on the store rank 0 hosts — the one every
rank has before any process group exists, with the heartbeat already beating
over it — then reads every peer's (``store.get`` waits for a peer still
starting, bounded by the store's timeout; a peer that dies is the
heartbeat's to name) and refuses by name the first variable whose value
differs from its own, naming both ranks and both values. Every rank sees the
same disagreement, so every rank refuses. Torch-free: the spawn parent
imports this module through the launcher.
"""

from __future__ import annotations

import json
import os
from typing import Mapping

from causalab.neural.shared.parallel.heartbeat import Store
from causalab.neural.shared.parallel.watchdog import (
    COLLECTIVE_TIMEOUT_VARIABLE,
    RANK_GRACE_VARIABLE,
)
from causalab.protocol.rules.errors import ProtocolError
from causalab.protocol.parallel import CONTEXT_EXPERIMENTAL_VARIABLE

__all__ = [
    "AGREED_VARIABLES",
    "AGREEMENT_VARIABLE",
    "agree_environment",
    "environment_key",
    "read_settings",
]

#: The environment variable that switches on the per-parameter gradient
#: agreement check — a float in ``[0, 1/2)`` — read once per fit by
#: ``train.run_cohort_training`` and handed to ``agreements.average_gradients``
#: as ``agreement``. Unset or empty, the check is off and the guard is the
#: plain mean; a value outside the range or not a number is refused by name.
#: ``0`` is bit identity across the ranks. Defined here, torch-free, so the
#: launcher can name it among the agreed settings; ``agreements.py`` re-exports it.
AGREEMENT_VARIABLE = "CAUSALAB_GRADIENT_AGREEMENT"

#: The settings every rank of a launched world must spell alike (module
#: docstring), in the order a refusal names them.
AGREED_VARIABLES: tuple[str, ...] = (
    COLLECTIVE_TIMEOUT_VARIABLE,
    RANK_GRACE_VARIABLE,
    CONTEXT_EXPERIMENTAL_VARIABLE,
    AGREEMENT_VARIABLE,
)


def environment_key(rank: int) -> str:
    """The store key under which ``rank`` publishes its settings."""
    return f"causalab/environment/{rank}"


def read_settings(environ: Mapping[str, str]) -> dict[str, str | None]:
    """The raw value of every agreed variable in ``environ`` — ``None``
    where unset. Raw on purpose: the typed parsers run per rank afterwards
    and refuse a malformed value by name; what is agreed is that every rank
    parses the same text."""
    return {name: environ.get(name) for name in AGREED_VARIABLES}


def agree_environment(
    store: Store,
    *,
    rank: int,
    world: int,
    environ: Mapping[str, str] | None = None,
) -> dict[str, str | None]:
    """Publish this rank's settings on ``store`` and hold them against every
    peer's (module docstring); returns the agreed settings.

    Raises:
        ProtocolError: ``P4`` at ``--parallel`` — a peer spells an agreed
            variable differently, named with both ranks and both values.
    """
    env = os.environ if environ is None else environ
    mine = read_settings(env)
    store.set(environment_key(rank), json.dumps(mine))
    for peer in range(world):
        if peer == rank:
            continue
        theirs = json.loads(store.get(environment_key(peer)))
        for name in AGREED_VARIABLES:
            if theirs.get(name) != mine[name]:
                raise ProtocolError(
                    "P4",
                    f"rank {rank} has {_spell(name, mine[name])} and rank {peer} has "
                    f"{_spell(name, theirs.get(name))}: an execution setting read "
                    "from the environment must be the same on every rank of the "
                    "world, or the ranks diverge instead of refusing "
                    "(docs/model_parallelism.md §3)",
                    path="--parallel",
                )
    return mine


def _spell(name: str, value: str | None) -> str:
    return f"{name} unset" if value is None else f"{name}={value!r}"
