"""``resolve_roles`` and the ``shuffle: {seed}`` data verb (IM spec §2.2;
the ``shuffled_source`` control's one new verb).

Against a stub resolver, no model: a counterfactual role that authors
``shuffle`` comes back permuted by exactly ``random.Random(seed).shuffle`` over
its indices (``shuffle_order``); the base role comes back in authored order;
the two lists stay the same length; seed 0 and seed 1 are two pairings for
``n >= 3``; and a list-valued role shuffles only the entry that authors it.

Without the change every test here fails: ``shuffle`` is an unknown key at
parse (``[P3] unknown key 'shuffle'``) and ``shuffle_order`` does not exist.
The mutation that permutes ``base`` too fails
``test_the_base_role_is_never_permuted``; one that ignores ``shuffle`` fails
``test_a_shuffled_counterfactual_role_is_the_seeds_permutation``.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import pytest

from causalab.protocol.positions.roles import resolve_roles, shuffle_order
from causalab.protocol.engine import RunContext
from causalab.io.env import FileArtifacts, ResolutionEnv
from causalab.protocol.schema import parse_document

from tests.protocol._docs import UNWRITTEN, base_doc, in_order


pytestmark = pytest.mark.unit

REF = "weekdays/data#train"
ROWS = [
    {"input": f"If today is day {i}, tomorrow is", "counterfactual_inputs": [f"cf {i}"]}
    for i in range(6)
]


class _Rows:
    """The ``DatasetResolver`` surface ``resolve_roles`` reads: ``rows``, in
    authored order, whatever the ref."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    def digest(self, ref: str) -> str:
        return "0" * 64

    def columns(self, ref: str) -> tuple[str, ...]:
        return tuple(self._rows[0])

    def rows(self, ref: str) -> list[dict[str, Any]]:
        return [dict(r) for r in self._rows]


def _request(tmp_path: Path, rows: list[dict[str, Any]]) -> RunContext:
    return RunContext(
        env=ResolutionEnv(
            datasets=_Rows(rows),  # pyright: ignore[reportArgumentType]
            artifacts=FileArtifacts(root=tmp_path),
        ),
        output_dir=tmp_path,
    )


def _doc(shuffle: dict[str, Any] | None, *, listed: bool = False) -> Any:
    raw = base_doc()
    cf = dict(raw["data"]["counterfactual"])
    if shuffle is not None:
        cf["shuffle"] = shuffle
    if listed:
        raw["data"]["counterfactual"] = [dict(raw["data"]["counterfactual"]), cf]
        raw["method"]["intervened_models"][UNWRITTEN]["input"] = "counterfactual[1]"
    else:
        raw["data"]["counterfactual"] = cf
    return parse_document(in_order(raw))


def test_a_drawn_role_reads_its_eval_member_outside_the_fit(tmp_path: Path) -> None:
    """§2.2 ``draw``: the resolved field of a drawn role is the fixed ``eval``
    member — ``column[0]`` unless authored — so the point's own forwards, an
    apply and ``train.eval`` read one member; the fit redraws from the rows."""
    raw = base_doc()
    raw["data"]["counterfactual"] = {
        **raw["data"]["counterfactual"],
        "field": "counterfactual_inputs",
        "draw": {"kind": "uniform", "eval": 1},
    }
    doc = parse_document(in_order(raw))
    rows = [
        dict(r, counterfactual_inputs=[f"cf {i}", f"cf' {i}"])
        for i, r in enumerate(ROWS)
    ]
    role_rows, role_fields = resolve_roles(doc, _request(tmp_path, rows).env)
    assert role_fields == {
        "base": "input",
        "counterfactual": "counterfactual_inputs[1]",
    }
    assert role_rows["counterfactual"] == rows  # the rows themselves, members intact


def test_shuffle_order_is_the_stdlib_permutation() -> None:
    """``shuffle_order(seed, n)`` is ``random.Random(seed).shuffle`` over
    ``range(n)`` and nothing else — the contract the spec states."""
    for seed in (0, 1, 7, 2**31):
        expected = list(range(5))
        random.Random(seed).shuffle(expected)
        assert shuffle_order(seed, 5) == expected
    assert shuffle_order(3, 0) == [] and shuffle_order(3, 1) == [0]
    assert shuffle_order(0, 6) == shuffle_order(0, 6)  # pure in (seed, n)


def test_an_unshuffled_document_pairs_rows_in_authored_order(tmp_path: Path) -> None:
    role_rows, role_fields = resolve_roles(_doc(None), _request(tmp_path, ROWS).env)
    assert role_rows["base"] == ROWS and role_rows["counterfactual"] == ROWS
    assert role_fields == {
        "base": "input",
        "counterfactual": "counterfactual_inputs[0]",
    }


@pytest.mark.parametrize("seed", [0, 1, 42])
def test_a_shuffled_counterfactual_role_is_the_seeds_permutation(
    tmp_path: Path, seed: int
) -> None:
    """Row ``i`` of the shuffled role is authored row ``order[i]``; the lengths
    agree, so the row-count check passes on the permuted list. Fails if
    ``resolve_roles`` ignores ``shuffle`` (the rows come back in authored
    order, and seed 0's order on six rows is not the identity)."""
    role_rows, _ = resolve_roles(_doc({"seed": seed}), _request(tmp_path, ROWS).env)
    order = shuffle_order(seed, len(ROWS))
    assert role_rows["counterfactual"] == [ROWS[i] for i in order]
    assert order != list(range(len(ROWS)))
    assert role_rows["counterfactual"] != ROWS
    assert len(role_rows["counterfactual"]) == len(role_rows["base"]) == len(ROWS)
    assert sorted(r["input"] for r in role_rows["counterfactual"]) == sorted(
        r["input"] for r in ROWS
    )  # the same rows


def test_the_base_role_is_never_permuted(tmp_path: Path) -> None:
    """The mutation that permutes ``base`` too — the population must stay in
    authored order, whatever the counterfactual role does."""
    role_rows, _ = resolve_roles(_doc({"seed": 0}), _request(tmp_path, ROWS).env)
    assert role_rows["base"] == ROWS


def test_two_seeds_are_two_pairings(tmp_path: Path) -> None:
    """For ``n >= 3`` seed 0 and seed 1 permute differently — two control
    documents, two pairings — and a re-resolution under one seed repeats it."""
    rows = ROWS[:3]
    zero, _ = resolve_roles(_doc({"seed": 0}), _request(tmp_path, rows).env)
    one, _ = resolve_roles(_doc({"seed": 1}), _request(tmp_path, rows).env)
    again, _ = resolve_roles(_doc({"seed": 0}), _request(tmp_path, rows).env)
    assert zero["counterfactual"] != one["counterfactual"]
    assert zero["counterfactual"] == again["counterfactual"]


def test_a_list_valued_role_shuffles_only_the_entry_that_authors_it(
    tmp_path: Path,
) -> None:
    role_rows, _ = resolve_roles(
        _doc({"seed": 0}, listed=True), _request(tmp_path, ROWS).env
    )
    assert set(role_rows) == {"base", "counterfactual[0]", "counterfactual[1]"}
    assert role_rows["counterfactual[0]"] == ROWS
    order = shuffle_order(0, len(ROWS))
    assert role_rows["counterfactual[1]"] == [ROWS[i] for i in order]
