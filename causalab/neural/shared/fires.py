"""Check how often each installed write fires.

Counts apply to one forward of a forward group, including a row window
or a forward resumed from a prefix. Module, attention, expert, and delta
boundary writes fire once. A ``delta_state`` write fires once per addressed
step. Members sharing a hook keep separate counts. Cached groups retain
the counts from the forward that produced them.

A mismatch raises before captures or result tables are published. Receipts
record the checked count for a module write and distinct addressed steps
for a state write, independent of row batching. This module is torch-free.
"""

from __future__ import annotations

import dataclasses
from typing import Iterable

from causalab.protocol.rules.errors import ProtocolError

__all__ = [
    "FireTally",
    "GroupFires",
    "check_fires",
    "group_label",
]


def group_label(model: str, input_role: str) -> str:
    """The spelling one forward group has in every record the run writes:
    ``<model> on <input>`` — the location ledger's edit group (§6)."""
    return f"{model} on {input_role}"


@dataclasses.dataclass
class FireTally:
    """One forward's count of each write member's firings against what its
    kind declares.

    [`declare`][] is called once per member while the hooks are built, with
    the count that member's kind owes this forward; [`fired`][] is called
    by the hook each time it runs. A state write declares the distinct steps
    its rows address and fires per step, so its ``steps`` are kept as well as
    counted — the receipt's number for it is the union over the group's
    forwards ([`GroupFires`][]).
    """

    expected: dict[str, int] = dataclasses.field(default_factory=dict)
    counts: dict[str, int] = dataclasses.field(default_factory=dict)
    #: the steps a state write fired at, per member (module-kind members
    #: have none: their unit is the forward itself)
    steps: dict[str, set[int]] = dataclasses.field(default_factory=dict)

    def declare(self, members: Iterable[str], times: int) -> None:
        for member in members:
            self.expected[member] = times
            self.counts.setdefault(member, 0)

    def fired(self, members: Iterable[str], *, step: int | None = None) -> None:
        for member in members:
            self.counts[member] = self.counts.get(member, 0) + 1
            if step is not None:
                self.steps.setdefault(member, set()).add(step)


@dataclasses.dataclass
class GroupFires:
    """One forward group's fire record across the forwards it ran as.

    Each forward's checked [`FireTally`][] is folded in; [`record`][] is
    the ``{member: count}`` the run receipt carries for the group — a
    module-kind member's per-forward count (every forward was checked at the
    same declared count, so the number does not depend on the row layout),
    a state write's number of distinct steps over every forward's rows.
    """

    per_forward: dict[str, int] = dataclasses.field(default_factory=dict)
    steps: dict[str, set[int]] = dataclasses.field(default_factory=dict)

    def fold(self, tally: FireTally) -> None:
        for member, count in tally.counts.items():
            if member in tally.steps:
                self.steps.setdefault(member, set()).update(tally.steps[member])
            else:
                self.per_forward[member] = count

    def record(self) -> dict[str, int]:
        return {
            **self.per_forward,
            **{member: len(steps) for member, steps in self.steps.items()},
        }


def check_fires(group: str, tally: FireTally) -> None:
    """Refuse the forward group when any member fired other than its
    declared count — a member that never fired named first.

    A count of zero is the measured shape: the forward never called the
    module the write was installed on, so the tensor the write addresses did
    not exist in this forward and the result would have been an un-intervened
    forward scored as an intervention. More than the declared count is a
    module the forward calls twice (a shared or looped block), where the
    write would land on a tensor the document did not name. Both carry
    ``component_unavailable``: what is missing is the tensor the member
    addresses, in this forward.
    """
    wrong = [
        (member, tally.counts.get(member, 0), expected)
        for member, expected in tally.expected.items()
        if tally.counts.get(member, 0) != expected
    ]
    if not wrong:
        return
    wrong.sort(key=lambda item: (item[1] != 0, item[0]))
    member, count, expected = wrong[0]
    if count == 0:
        what = (
            f"write {member!r} in forward group {group!r} fired 0 times in one "
            f"forward, not the {expected} its kind declares: the forward never "
            "called the module this write was installed on, so the tensor it "
            "addresses did not exist in this forward and the point would have "
            "scored an un-intervened forward as an intervention"
        )
    else:
        what = (
            f"write {member!r} in forward group {group!r} fired {count} times in "
            f"one forward, not the {expected} its kind declares: the forward "
            "calls the module this write was installed on more than once, so "
            "the write would land on a tensor the document did not name"
        )
    others = ", ".join(f"{m!r} ({c} of {e})" for m, c, e in wrong[1:])
    if others:
        what += f"; also off: {others}"
    raise ProtocolError(
        "P4",
        what + ". The write set is one transaction — the point is refused whole, and "
        "nothing of it was published or written.",
        reason="component_unavailable",
    )
