"""§2.2 rule 20 — ``base`` is the schema of a paired row.

Validation used to check column references against the **union** of the data
roles while the executor served them from **base** alone
(``neural/shared/executor/base.py``: ``rows_for_metrics`` returns
``role_rows["base"]``). So validation accepted a superset of what a run could
serve: a metric naming a counterfactual-only column passed ``validate --data``,
the model was loaded, and only then did the metric die looking for the column
in a base row.

Two claims are checked here, and one census:

* a reference to a column only a counterfactual carries is refused **at load**,
  by a message that names both roles — the reference resolves, just not where
  the run will look for it;
* the schema rule itself, in both halves: a non-base role's columns are a
  subset of base's, and *equal* to them when the two roles name different
  datasets. The equality half exists because a ``column`` position resolves at
  run time against the role of the read it positions, not against base, so a
  bare subset rule would leave the mirror-image hole;
* every multi-role document this repo ships names **one dataset for both
  sides**, which is the shape §3 recommends and which satisfies rule 20 for
  free. That is what makes this a rule about a shape nobody writes rather than
  a migration.

The fixture tables are the committed ones. ``weekdays/data#train`` and ``ioi/test``
happen to stand in the relation the rule is about — ioi's columns are a strict
subset of weekdays' — so pointing the two roles at them, in one order and then
the other, exercises both halves without inventing a table.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Any

import pytest

from causalab.protocol.rules.errors import ValidationError
from causalab.protocol.pipeline import compile_protocol
from causalab.protocol.rules.data import check_data_columns

from tests.protocol._docs import in_order
from tests.protocol._env import CORPUS_DIR


pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[2]

#: Every place a runnable document with a `data` section is committed.
SHIPPED = (
    "tests/protocols/*.json",
    "demos/*/protocols/*.json",
)

#: ioi/test's columns are a strict subset of weekdays/data#train's, which is the
#: relation rule 20 is about.
NARROW = "ioi/test"
WIDE = "weekdays/data#train"
#: A column `weekdays/data#train` has and `ioi/test` does not.
WIDE_ONLY = "label"


def _document(env, base: str, counterfactual: str) -> dict[str, Any]:
    """Corpus 02 (a plain interchange) with its two roles re-pointed.

    Its ``logit_diff`` metric goes with them: it names ``base_answer``, a
    column only the wider table has, which would make every test here fail on
    that instead of on what it is about. ``iia`` stays, because ``cf_answer``
    is in both tables — so the document is a clean control until a test breaks
    it deliberately.
    """
    loaded = compile_protocol(CORPUS_DIR / "02_interchange_im.json", env=env)
    raw = json.loads(json.dumps(dict(loaded.tree)))
    raw["data"] = {
        "base": {"dataset": base, "field": "input"},
        "counterfactual": {
            "dataset": counterfactual,
            "field": "counterfactual_inputs[0]",
        },
    }
    raw["method"]["save"] = [
        entry
        for entry in raw["method"]["save"]
        if entry["file_path"] != "logit_diff.json"
    ]
    return raw  # protocol-3 literal; the module shims migrate it at parse


def test_a_metric_on_a_counterfactual_only_column_is_refused_at_load(env) -> None:
    """The late failure this closes, and the message that makes it useful.

    Before: ``validate --data`` said OK, the model loaded, and
    ``metrics.py`` raised ``[P2] metric column 'label' missing from dataset
    row 0`` — a message that names neither role and reads like a bad table.
    """
    raw = _document(env, base=NARROW, counterfactual=WIDE)
    (iia,) = [e for e in raw["method"]["save"] if e["file_path"] == "iia.json"]
    iia["aggregation"]["expected"] = WIDE_ONLY

    with pytest.raises(ValidationError) as err:
        check_data_columns(compile_protocol(raw, env=env), env)

    message = str(err.value)
    assert err.value.rule == 4
    assert WIDE_ONLY in message
    # both roles, and both tables they resolve to
    assert "data.base" in message and "data.counterfactual" in message
    assert NARROW in message and WIDE in message


def test_a_position_on_a_counterfactual_only_column_is_refused_at_load(env) -> None:
    """The same anchor for a ``{"column": …}`` position."""
    raw = _document(env, base=NARROW, counterfactual=WIDE)
    raw["method"]["positions"] = {"tap": {"column": WIDE_ONLY}}
    raw["method"]["reads"]["v_cf"]["pos"] = "tap"
    raw = in_order(raw)

    with pytest.raises(ValidationError) as err:
        check_data_columns(compile_protocol(raw, env=env), env)

    assert err.value.rule == 4
    assert WIDE_ONLY in str(err.value)
    assert "data.base" in str(err.value)


def test_a_counterfactual_only_column_is_refused_even_unreferenced(env) -> None:
    """Rule 20's subset half is a schema rule, not a reachability one.

    A column only the counterfactual carries is a column no metric can read,
    whatever today's metrics happen to name — so it is refused now rather than
    the first time somebody references it.
    """
    raw = _document(env, base=NARROW, counterfactual=WIDE)

    with pytest.raises(ValidationError) as err:
        check_data_columns(compile_protocol(raw, env=env), env)

    assert err.value.rule == 20
    assert err.value.path == "data.counterfactual"
    assert WIDE_ONLY in str(err.value)


def test_two_tables_must_carry_identical_columns(env) -> None:
    """Rule 20's equality half, and the hole it closes.

    A ``column`` position resolves against the role of the read it positions
    (``executor/base.py``), not against base. Under a bare subset rule, a position
    on a counterfactual read naming a **base-only** column would pass load and
    fail at run — the mirror image of the bug this whole rule is about.
    """
    raw = _document(env, base=WIDE, counterfactual=NARROW)

    with pytest.raises(ValidationError) as err:
        check_data_columns(compile_protocol(raw, env=env), env)

    assert err.value.rule == 20
    assert "different dataset" in str(err.value)
    assert WIDE_ONLY in str(err.value)


def test_one_table_for_both_roles_costs_nothing(env) -> None:
    """The shape every shipped document has: rule 20 is satisfied trivially."""
    raw = _document(env, base=WIDE, counterfactual=WIDE)
    assert check_data_columns(compile_protocol(raw, env=env), env)


@pytest.mark.parametrize(
    "name",
    sorted(
        path.name
        for path in CORPUS_DIR.glob("*_im.json")
        if "counterfactual" in json.loads(path.read_text()).get("data", {})
    ),
)
def test_every_multi_role_corpus_document_still_loads(env, name: str) -> None:
    """The rule refuses nothing that ships."""
    assert check_data_columns(compile_protocol(CORPUS_DIR / name, env=env), env)


def test_no_shipped_document_splits_its_roles_across_tables() -> None:
    """The census behind "this refuses nothing that ships".

    A pure JSON read — no tables resolved — so it covers the shipped documents
    whose data lives outside the test fixtures too. If a document ever *does*
    want two tables, this test is where that decision gets made explicitly:
    rule 20 will then require the two to carry identical columns.
    """
    split: list[str] = []
    multi = 0
    for pattern in SHIPPED:
        for path in sorted(REPO.glob(pattern)):
            raw = json.loads(path.read_text())
            data = raw.get("data") or (raw.get("application") or {}).get("data")
            if not isinstance(data, dict) or "counterfactual" not in data:
                continue
            multi += 1
            counterfactual = data["counterfactual"]
            roles = (
                counterfactual if isinstance(counterfactual, list) else [counterfactual]
            )
            if any(
                role.get("dataset") != data["base"].get("dataset") for role in roles
            ):
                split.append(str(path.relative_to(REPO)))

    assert multi >= 20, f"expected the multi-role corpus, found {multi} documents"
    assert not split, (
        f"{split} name different datasets for base and counterfactual — legal, "
        "but rule 20 then requires identical column sets (§2.2)"
    )


def test_the_shipped_glob_actually_matches_something() -> None:
    """A census over an empty glob passes for the wrong reason."""
    for pattern in SHIPPED:
        assert glob.glob(str(REPO / pattern)), f"{pattern} matched no file"
