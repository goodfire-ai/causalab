"""Every task package ships a table, or says why it cannot yet.

``causalab/tasks/<task>/data/<variant>.json`` is what a document names
(``<task>/data/<variant>#<split>`` under the default ``--data-root``), and it
is exactly the bytes a document names: nothing sits beside it (spec §2.2) —
no recipe sidecar, no rebuild guard. What was built how is the builder's
command line, recorded in the task's README; what a run is held to is the
table's content digest, which is in every consuming document's canonical
form (spec §7).

What this file keeps is the one invariant that survives without a recipe:
the set of task packages and the set of shipped tables agree, so a task that
ships nothing is a visible debt with its reason, not a silent gap.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from causalab.tables import SPLIT_COLUMN
from causalab.tasks import TASKS_ROOT

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[2]

#: The subdirectory of a task package that holds its tables — half of every
#: ref (``<task>/data/<variant>``).
DATA_DIR = "data"


def task_packages(root: Path = TASKS_ROOT) -> list[str]:
    """The task packages under ``root`` — every directory with a
    ``causal_models.py``, which is what ``load_task`` requires of a task.
    Hand-authored table folders (``pile/``) are not tasks."""
    return sorted(p.parent.name for p in root.glob("*/causal_models.py"))


def shipped_tables(root: Path = TASKS_ROOT) -> list[Path]:
    """Every ``<task>/data/<variant>.json`` under ``root``, sorted."""
    return sorted(root.glob(f"*/{DATA_DIR}/*.json"))


SHIPPED = shipped_tables()


def test_the_tasks_package_ships_tables() -> None:
    """The checks below are parametrized over what they find; make "found
    nothing" a failure rather than a silently green empty run."""
    assert SHIPPED, f"no */{DATA_DIR}/*.json under {TASKS_ROOT}"
    assert TASKS_ROOT == REPO / "causalab" / "tasks"


@pytest.mark.parametrize(
    "table", SHIPPED, ids=lambda p: f"{p.parents[1].name}/{p.stem}"
)
def test_a_shipped_table_is_a_split_declaring_row_table(table: Path) -> None:
    """The one structural fact a document relies on at load: a JSON array of
    row objects, every row declaring its split (spec §2.2)."""
    rows = json.loads(table.read_text())
    assert isinstance(rows, list) and rows, f"{table} is not a non-empty row table"
    assert all(isinstance(row, dict) and SPLIT_COLUMN in row for row in rows), (
        f"{table}: every row declares its {SPLIT_COLUMN!r}"
    )


def test_nothing_sits_beside_a_shipped_table() -> None:
    """A ``data/`` directory holds tables and nothing else — no sidecar of any
    kind (spec §2.2): a table is exactly the bytes a document names."""
    for data_dir in sorted(TASKS_ROOT.glob(f"*/{DATA_DIR}")):
        extras = [p.name for p in data_dir.iterdir() if p.suffix != ".json"]
        assert not extras, f"{data_dir} holds non-table files: {extras}"
        assert not [p.name for p in data_dir.glob("*.manifest.json")]


#: Task packages that ship no table yet, each with the defect that stops them.
#: A task on this list must ship nothing and a task off it must ship something,
#: so the debt is visible here and the entry is deleted the day the task is
#: fixed — not silently outlived.
NOT_YET_SHIPPABLE = {
    "entity_binding": (
        "output_tokens is declared on positional_answer (an index into the "
        "groups) with entity-name forms, so the serializer refuses every row; "
        "declare the forms on raw_output"
    ),
    "graph_walk": (
        "the answer is a set of valid neighbours (raw_output is a list), which "
        "the v1 row vocabulary — one answer string and its forms — cannot carry"
    ),
}


def test_every_task_package_ships_a_table_or_says_why() -> None:
    shipping = {t.parents[1].name for t in SHIPPED}
    for task in task_packages():
        if task in NOT_YET_SHIPPABLE:
            assert task not in shipping, (
                f"{task} ships a table now — delete its NOT_YET_SHIPPABLE entry"
            )
        else:
            assert task in shipping, (
                f"{task} ships no table: build one with scripts/build_task_dataset.py "
                f"--out causalab/tasks/{task}/{DATA_DIR}/default.json, or add it to "
                "NOT_YET_SHIPPABLE with the reason"
            )
