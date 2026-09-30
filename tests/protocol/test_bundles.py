"""Addressing one entry inside a tensor bundle (§2.5, §2.6).

The producer keys entries off its *own* document (``weight[k=8,seed=0]``)
while a consumer names a slot (``weight``); these are the rules that bridge
the two without ever guessing.
"""

from __future__ import annotations

import pytest

from causalab.protocol.bundles import (
    entry_selection,
    parse_entry_key,
    select_entry,
)
from causalab.protocol.rules.errors import ValidationError

pytestmark = pytest.mark.unit

SWEPT = [
    "weight[k=2,seed=0]",
    "weight[k=2,seed=1]",
    "weight[k=4,seed=0]",
    "weight[k=4,seed=1]",
]


class TestParseEntryKey:
    def test_slot_and_coordinates(self):
        assert parse_entry_key("weight[k=8,seed=0]") == (
            "weight",
            {"k": "8", "seed": "0"},
        )

    def test_bare_slot(self):
        assert parse_entry_key("value") == ("value", {})

    def test_structured_coordinate_is_opaque(self):
        """A swept position renders as ``tap={index:-1}``; a flat split
        cannot read that back, so the key stays opaque and the entry table
        answers instead."""
        key = "v[tap={index:-1}]"
        assert parse_entry_key(key) == (key, {})


class TestSelectEntry:
    def test_single_entry_needs_no_selector(self):
        assert select_entry(["weight"], "weight", None, what="w") == "weight"

    def test_selects_by_coordinates(self):
        assert (
            select_entry(SWEPT, "weight", {"k": 4, "seed": 1}, what="w")
            == "weight[k=4,seed=1]"
        )

    def test_int_and_string_coordinates_agree(self):
        """The bundle key records ``k=4``; a document may author either."""
        assert select_entry(SWEPT, "weight", {"k": "4", "seed": 1}, what="w") == (
            select_entry(SWEPT, "weight", {"k": 4, "seed": 1}, what="w")
        )

    def test_ambiguous_selection_refuses(self):
        with pytest.raises(ValidationError) as err:
            select_entry(SWEPT, "weight", {"k": 4}, what="featurizer 'rot'")
        assert "must be unique" in str(err.value)
        assert "seed" in str(err.value)  # names what is still open

    def test_no_selector_against_many_entries_refuses(self):
        with pytest.raises(ValidationError) as err:
            select_entry(SWEPT, "weight", None, what="featurizer 'rot'")
        assert "selects none" in str(err.value)

    def test_missing_coordinate_refuses(self):
        with pytest.raises(ValidationError) as err:
            select_entry(SWEPT, "weight", {"k": 99}, what="w")
        assert "no 'weight' entry matches" in str(err.value)

    def test_unknown_slot_refuses(self):
        with pytest.raises(ValidationError) as err:
            select_entry(SWEPT, "mu", None, what="w")
        assert "holds no 'mu' entry" in str(err.value)

    def test_widths_sidecar_is_not_a_candidate(self):
        """A ragged read's ``.widths`` rides along with its parent entry."""
        keys = ["v_mean", "v_mean.widths"]
        assert select_entry(keys, "v_mean", None, what="w") == "v_mean"

    def test_a_bundle_without_coordinates_answers_anything(self):
        """An external, hand-made bundle records no coordinates: it cannot
        answer a coordinate question, and cannot contradict one either."""
        assert select_entry(["weight"], "weight", {"k": 8}, what="w") == "weight"

    def test_entry_table_beats_the_key_text(self):
        keys = ["v[tap={index:-1}]", "v[tap={index:-2}]"]
        table = {
            keys[0]: {"slot": "v", "coords": {"tap": {"index": -1}}},
            keys[1]: {"slot": "v", "coords": {"tap": {"index": -2}}},
        }
        chosen = select_entry(
            keys, "v", {"tap": {"index": -2}}, what="w", coords_by_key=table
        )
        assert chosen == keys[1]

    def test_implicit_selection_drops_axes_the_producer_never_had(self):
        """The consumer sweeps a layer the fit knew nothing about: that
        coordinate is dropped rather than matching nothing."""
        assert (
            select_entry(
                ["weight[k=2,seed=0]", "weight[k=4,seed=0]"],
                "weight",
                {"k": 4, "target.layers": 7},
                what="w",
                implicit=True,
            )
            == "weight[k=4,seed=0]"
        )

    def test_an_authored_selection_is_never_relaxed(self):
        with pytest.raises(ValidationError):
            select_entry(
                ["weight[k=2,seed=0]"],
                "weight",
                {"k": 2, "target.layers": 7},
                what="w",
            )


class TestEntrySelection:
    def test_authored_entry_wins_and_is_explicit(self):
        assert entry_selection({"k": 8}, {"featurizers.rot.k": 2}, "rot") == (
            {"k": 8},
            False,
        )

    def test_point_coordinates_stand_in_implicitly(self):
        want, implicit = entry_selection(
            None, {"featurizers.rot.k": 2, "train.seed": 1}, "rot"
        )
        assert want == {"k": 2, "seed": 1}
        assert implicit is True

    def test_an_unswept_consumer_implies_nothing(self):
        assert entry_selection(None, {}, "rot") == (None, False)
