"""The resolution triple (spec §4.1): ``available`` / ``unavailable`` /
``invalid`` as values, and the denominator read from them.

Pure: no torch, no engine, no model. What is pinned here is the *contract*
the engines and the CLI build on — an ``unavailable`` carries a reason from
the closed vocabulary and a denominator key; an ``invalid`` never becomes a
result; a summary over 157 cells with two unavailable reads ``155 / 157`` and
names both reasons, with nothing kept beside the cells. The load half —
an entry selector that resolves to nothing is still a load error, with its
rule number, its text and now its reason code — is here too, because it is
the same decision from the other side: document-decidable stays ``invalid``.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from causalab.protocol.bundles import select_entry
from causalab.protocol.engine import RunResult
from causalab.protocol.rules.errors import REASON_CODES, ValidationError
from causalab.protocol.rules.data import _entry_identity  # pyright: ignore[reportPrivateUsage]
from causalab.protocol.results import (
    example_id_defect,
    example_labels,
    STATUS_KEY,
    UNAVAILABLE,
    Available,
    Denominator,
    Invalid,
    Unavailable,
    available,
    cell_key,
    cell_record,
    invalid,
    unavailable,
)

pytestmark = pytest.mark.unit


class TestConstructors:
    def test_available_carries_what_resolved_and_its_key(self):
        cell = available({"file_path": "iia.json", "key": "iia"}, "iia")
        assert isinstance(cell, Available)
        assert cell.mapping == {"file_path": "iia.json", "key": "iia"}
        assert cell.denominator_key == "iia"

    def test_unavailable_carries_a_reason_a_detail_and_its_key(self):
        cell = unavailable("empty_selector", "expert 7 was sent no token", "r[l=3]")
        assert isinstance(cell, Unavailable)
        assert (cell.reason, cell.detail, cell.denominator_key) == (
            "empty_selector",
            "expert 7 was sent no token",
            "r[l=3]",
        )

    def test_the_reason_vocabulary_is_closed(self):
        """`unavailable` takes exactly `errors.REASON_CODES` — the same names
        a refusal carries — so no result can invent a reason."""
        for code in REASON_CODES:
            assert unavailable(code, "x", "k").reason == code
        with pytest.raises(AssertionError, match="unknown reason code"):
            unavailable("made_up", "x", "k")  # type: ignore[arg-type]

    def test_invalid_reuses_the_existing_error_code(self):
        cell = invalid("V15", "no entry matches")
        assert isinstance(cell, Invalid)
        assert (cell.error_code, cell.detail) == ("V15", "no entry matches")

    def test_the_values_are_frozen(self):
        cell = available({}, "k")
        with pytest.raises(dataclasses.FrozenInstanceError):
            cell.denominator_key = "other"  # type: ignore[misc]


class TestResultCell:
    def test_an_available_cell_records_nothing_new(self):
        """No `status: "available"`: a result written before the value existed
        is byte-identical to one written after it (zero pins move)."""
        assert cell_record(available({"key": "r"}, "r")) == {}

    def test_an_unavailable_cell_records_the_four_fields(self):
        record = cell_record(unavailable("empty_selector", "why", "r[k=2]"))
        assert record == {
            STATUS_KEY: UNAVAILABLE,
            "reason": "empty_selector",
            "detail": "why",
            "denominator_key": "r[k=2]",
        }
        json.dumps(record)  # serializable as-is into an entries record

    def test_an_invalid_never_serializes_into_a_result(self):
        with pytest.raises(TypeError, match="never enters a result"):
            cell_record(invalid("V15", "x"))
        mixed: list[object] = [available({}, "a"), invalid("P4", "x")]
        with pytest.raises(TypeError, match="never enters a result"):
            Denominator.of(mixed)  # type: ignore[arg-type]


class TestDenominatorKey:
    def test_an_unswept_cell_is_keyed_by_its_value_name(self):
        assert cell_key("iia", {}) == "iia"

    def test_a_swept_cell_carries_the_coordinate_label(self):
        """The same key the value's tensor entry takes in a saved bundle, so a
        reader goes from the `cells` line to the entry it names."""
        key = cell_key("iia", {"sites.target.layers": 3, "positions.p.index": -1})
        assert key == "iia[target.layers=3,p.index=-1]"

    def test_coordinates_on_the_value_itself_shorten(self):
        assert cell_key("rot", {"featurizers.rot.k": 8}) == "rot[k=8]"


def _campaign(unavailable_at: dict[int, str]) -> list[Available | Unavailable]:
    """157 cells — one metric over a 157-point layer scan — with the cells at
    the given indices unavailable for the given reasons."""
    cells: list[Available | Unavailable] = []
    for layer in range(157):
        key = cell_key("iia", {"sites.target.layers": layer})
        if layer in unavailable_at:
            cells.append(unavailable(unavailable_at[layer], f"layer {layer}", key))  # type: ignore[arg-type]
        else:
            cells.append(available({"file_path": "iia.json", "key": key}, key))
    return cells


class TestDenominator:
    def test_155_of_157_with_both_reasons_named(self):
        """A synthetic 157-cell campaign with two excluded cells, read from the
        result: 155 of 157 eligible, the two cells excluded measurements rather
        than null localizations — no bookkeeping beside the cells."""
        cells = _campaign({41: "empty_selector", 99: "component_unavailable"})
        d = Denominator.of(cells)
        assert (d.eligible, d.total, d.excluded) == (155, 157, 2)
        assert d.unavailable == {
            "component_unavailable": ("iia[target.layers=99]",),
            "empty_selector": ("iia[target.layers=41]",),
        }
        line = d.render()
        assert line.startswith("155 / 157 eligible")
        assert "empty_selector ×1" in line and "component_unavailable ×1" in line
        record = d.as_record()
        assert (record["eligible"], record["total"]) == (155, 157)
        assert record["unavailable"]["empty_selector"] == {
            "count": 1,
            "cells": ["iia[target.layers=41]"],
        }

    def test_a_full_campaign_reads_n_of_n(self):
        d = Denominator.of(_campaign({}))
        assert (d.eligible, d.total) == (157, 157)
        assert d.render() == "157 / 157 eligible"
        assert d.as_record()["unavailable"] == {}

    def test_cells_under_one_reason_group(self):
        d = Denominator.of(_campaign({3: "empty_selector", 5: "empty_selector"}))
        assert d.render() == "155 / 157 eligible; 2 excluded: empty_selector ×2"

    def test_the_run_result_carries_the_cells_and_reads_its_denominator(self):
        """`RunResult.denominator` is the reduction over `RunResult.cells`; an
        engine that reports no cells (a stub) reads `0 / 0`."""
        result = RunResult(files={}, cells=tuple(_campaign({41: "empty_selector"})))
        assert result.denominator.render() == (
            "156 / 157 eligible; 1 excluded: empty_selector ×1"
        )
        assert RunResult(files={}).denominator.render() == "0 / 0 eligible"


#: a swept bundle's keys, as `tests/protocol/test_bundles.py` uses them
SWEPT = ["weight[k=2,seed=0]", "weight[k=2,seed=1]", "weight[k=4,seed=0]"]


class TestAnInvalidSelectorIsStillALoadError:
    """Document-decidable stays `invalid`: rule 15, the same text, and now a
    reason code the loader branches on instead of matching the message."""

    def test_an_authored_entry_that_matches_nothing_is_v15_empty_selector(self):
        with pytest.raises(ValidationError) as err:
            select_entry(SWEPT, "weight", {"k": 99}, what="featurizer 'rot'")
        assert err.value.rule == 15 and err.value.code == "V15"
        assert err.value.reason == "empty_selector"
        assert "no 'weight' entry matches {k=99}" in str(err.value)

    def test_no_selector_against_many_entries_is_v15_empty_selector(self):
        with pytest.raises(ValidationError) as err:
            select_entry(SWEPT, "weight", None, what="featurizer 'rot'")
        assert err.value.rule == 15 and err.value.reason == "empty_selector"
        assert "the document selects none" in str(err.value)

    def test_an_ambiguous_selection_is_not_an_empty_one(self):
        with pytest.raises(ValidationError) as err:
            select_entry(SWEPT, "weight", {"k": 2}, what="featurizer 'rot'")
        assert err.value.rule == 15 and err.value.reason is None

    def test_the_loader_defers_on_the_code_not_the_text(self):
        """Several candidates and nothing authored: the selection is the
        executing point's, so the identity check is deferred (`None`) — by
        `reason`, with no string in the message consulted."""
        stamped = {
            "entries": json.dumps(
                {key: {"slot": "weight", "coords": {}} for key in SWEPT}
            ),
        }
        assert _entry_identity(stamped, slot="weight", authored=None, what="w") is None

    def test_the_loader_still_raises_an_authored_miss(self):
        stamped = {
            "entries": json.dumps(
                {key: {"slot": "weight", "coords": {}} for key in SWEPT}
            ),
        }
        with pytest.raises(ValidationError) as err:
            _entry_identity(stamped, slot="weight", authored={"k": 99}, what="w")
        assert err.value.rule == 15 and err.value.reason == "empty_selector"


# --------------------------------------------------------------------------- #
# example labels (formerly tests/protocol/test_examples.py)
# --------------------------------------------------------------------------- #


def test_a_table_without_the_column_is_labelled_by_index_as_strings():
    rows = [
        {"prompt": "a"},
        {"prompt": "b"},
        {"prompt": "a"},
    ]  # a repeated prompt is fine
    assert example_id_defect(rows) is None
    assert example_labels(rows) == ["0", "1", "2"]


def test_an_authored_column_labels_every_row_as_a_string():
    rows = [{"example_id": "w3", "prompt": "a"}, {"example_id": 7, "prompt": "b"}]
    assert example_id_defect(rows) is None
    assert example_labels(rows) == ["w3", "7"]


def test_an_empty_table_has_no_labels_and_no_defect():
    assert example_id_defect([]) is None
    assert example_labels([]) == []


@pytest.mark.parametrize(
    ("rows", "fragment"),
    [
        ([{"example_id": "a"}, {"prompt": "b"}], "carry no example_id"),
        ([{"example_id": "a"}, {"example_id": ""}], "row 1 has an empty example_id"),
        ([{"example_id": "a"}, {"example_id": None}], "row 1 has an empty example_id"),
        (
            [{"example_id": "a"}, {"example_id": "b"}, {"example_id": "a"}],
            "'a' labels rows 0 and 2",
        ),
        ([{"example_id": 1}, {"example_id": "1"}], "'1' labels rows 0 and 1"),
    ],
)
def test_a_column_that_cannot_label_its_rows_is_named(rows, fragment):
    defect = example_id_defect(rows)
    assert defect is not None and fragment in defect
