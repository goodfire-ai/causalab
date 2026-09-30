"""World 1 that refuses every collective: the probe for the fast paths.

Every seam that takes a [`Collective`][causalab.neural.shared.parallel.collective.Collective]
has a world-1 fast path — a group of one is the identity, with no call on
the collective. `RefusingCollective` is how a test proves the path
was taken: ``size`` is 1 and ``rank`` is 0 on every axis, and any other call
is the failure the fast path exists to prevent. It is not a second fake of
the seam (``docs/TESTS.md``, "one fake per seam"): it runs no collective, it
refuses them all.
"""

from __future__ import annotations

from typing import Any, Callable

import torch

from causalab.neural.shared.parallel.placement import Axis

__all__ = ["RefusingCollective"]


def _refuse(name: str) -> Callable[..., Any]:
    def method(self: Any, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError(f"world 1 touched the collective: {name}")

    return method


class RefusingCollective:
    """World 1 that refuses every collective: ``size`` is 1 and ``rank`` is 0
    on every axis, and any other call is the failure the fast path exists to
    prevent (module docstring)."""

    device = torch.device("cpu")

    def rank(self, axis: Axis) -> int:
        return 0

    def size(self, axis: Axis) -> int:
        return 1

    all_gather = _refuse("all_gather")
    all_reduce_sum = _refuse("all_reduce_sum")
    broadcast = _refuse("broadcast")
    send = _refuse("send")
    recv = _refuse("recv")
    agree_min = _refuse("agree_min")
    agree_any = _refuse("agree_any")
    agree_sum = _refuse("agree_sum")
    barrier = _refuse("barrier")
