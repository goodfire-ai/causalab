"""Input roles (spec §2.2): which rows each role reads, paired by index, and
the ``shuffle`` permutation — protocol-side and torch-free, the rows a
position frame is encoded from (previously the ``services`` half of the
engine layer).
"""

from __future__ import annotations

import random
from typing import Any

from causalab.io.env import ResolutionEnv
from causalab.protocol.results import example_id_defect
from causalab.protocol.rules.errors import ProtocolError
from causalab.protocol.schema import DataRole, Document

__all__ = ["input_roles", "resolve_roles", "shuffle_order"]


def input_roles(doc: Document) -> dict[str, DataRole]:
    """The document's input roles under the names a read's ``input`` uses:
    ``base``, ``counterfactual``, or ``counterfactual[j]`` for a list-valued
    role (§2.2). The one place the naming rule lives, so the rows an executor
    batches, the data identity a forward group is keyed on and the stamp a
    harvested read carries all name a role the same way."""
    roles: dict[str, DataRole] = {}
    for role, value in doc.data.items():
        if isinstance(value, tuple):
            roles.update({f"{role}[{j}]": spec for j, spec in enumerate(value)})
        else:
            roles[role] = value
    return roles


def resolve_roles(
    doc: Document, env: ResolutionEnv
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str]]:
    """Dataset rows + field selector per input role, rows paired by index.

    A counterfactual role that authors ``shuffle: {seed}`` (§2.2) has its rows
    permuted by [`shuffle_order`][] before the pairing — the same rows, met
    by different base rows — so the ``shuffled_source`` control (workflow spec
    §2.2) is a document and not a serialized permuted table. The base role is
    never permuted (the parser refuses ``shuffle`` there), and the row-count
    check below runs on the permuted list, whose length is unchanged.

    ``rows`` is part of the [`DatasetResolver`][causalab.io.env.DatasetResolver]
    contract, so this reads it directly — a resolver without it is a typing
    error at construction, not a surprise at run time. ``env`` is the
    resolution environment (a [`RunContext`][]
    hands its ``env``)."""
    rows_of = env.datasets.rows
    role_rows: dict[str, list[dict[str, Any]]] = {}
    role_fields: dict[str, str] = {}
    lengths: dict[str, int] = {}
    for role_name, role_spec in input_roles(doc).items():
        rows = rows_of(str(role_spec.dataset))
        # the run's own check of what `validate --data` refuses (§2.2): a
        # label column that cannot label its rows
        defect = example_id_defect(rows)
        if defect is not None:
            raise ProtocolError(
                "P2", f"data.{role_name} dataset {role_spec.dataset!r}: {defect}"
            )
        if role_spec.shuffle is not None:
            order = shuffle_order(int(role_spec.shuffle["seed"]), len(rows))
            rows = [rows[i] for i in order]
        role_rows[role_name] = rows
        # §2.2 `draw`: outside a fit's updates a drawn role reads its fixed
        # `eval` member (`resolved_field`); the fit redraws per epoch from
        # these same rows
        role_fields[role_name] = role_spec.resolved_field
        lengths[role_name] = len(role_rows[role_name])
    if len(set(lengths.values())) > 1:
        raise ProtocolError(
            "P2",
            f"input roles have unequal row counts {lengths} — rows are paired "
            "by index (§2.2)",
        )
    return role_rows, role_fields


def shuffle_order(seed: int, n: int) -> list[int]:
    """The permutation a ``shuffle: {seed}`` role applies (§2.2): the indices
    ``0..n-1`` shuffled by ``random.Random(seed).shuffle`` — stdlib only,
    torch-free, a pure function of ``(seed, n)``, so two runs of one document
    pair the same rows and two seeds give two pairings. The permuted role's
    row ``i`` is the authored row ``order[i]``."""
    order = list(range(n))
    random.Random(seed).shuffle(order)
    return order
