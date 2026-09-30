"""The ``rows`` mode of the data axis (``docs/model_parallelism.md`` §8.3):
the ``dp=<count>:<mode>`` grammar with each refusal beside its valid twin,
``ParallelGeometry.data_mode``, ``format_geometry`` as the grammar's
inverse over both modes, and ``check_rows`` — the two torch-free document
rules: a ``train`` section to split, and ``train.batch.pairs`` at least the
replica count.

``unit`` for the grammar and the rules, ``property`` for the round trip over
geometries of either mode and for ``check_rows`` accepting iff both facts
hold.
"""

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from causalab.protocol.rules.errors import ProtocolError
from causalab.protocol.parallel import (
    DATA_MODES,
    ONE,
    ParallelGeometry,
    check_rows,
    format_geometry,
    parse_geometry,
)

#: The repository's hypothesis settings (``docs/model_parallelism.md`` §10).
_SETTINGS = settings(
    deadline=None,
    max_examples=30,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)


def _refusal(text: str) -> ProtocolError:
    with pytest.raises(ProtocolError) as err:
        parse_geometry(text)
    return err.value


# --------------------------------------------------------------------------- #
# the record
# --------------------------------------------------------------------------- #


@pytest.mark.unit
class TestDataMode:
    def test_the_modes_are_points_and_rows_and_points_is_the_default(self) -> None:
        assert DATA_MODES == ("points", "rows")
        assert ONE.data_mode == "points"
        assert ParallelGeometry(data=2).data_mode == "points"

    def test_a_mode_outside_the_vocabulary_is_refused_naming_the_data_axis(
        self,
    ) -> None:
        with pytest.raises(ProtocolError) as err:
            ParallelGeometry(data=2, data_mode="cols")  # type: ignore[arg-type]
        assert err.value.code == "P4"
        assert err.value.path == "--parallel.data"
        assert "points, rows" in str(err.value)

    def test_the_mode_is_part_of_equality(self) -> None:
        assert ParallelGeometry(data=2, data_mode="rows") != ParallelGeometry(data=2)
        assert ParallelGeometry(data=2, data_mode="points") == ParallelGeometry(data=2)

    def test_the_mode_leaves_world_and_model_alone(self) -> None:
        rows = ParallelGeometry(data=2, tensor=2, data_mode="rows")
        assert rows.world == 4 and rows.model == 2 and rows.data == 2


# --------------------------------------------------------------------------- #
# the grammar: each refusal beside its twin
# --------------------------------------------------------------------------- #


@pytest.mark.unit
class TestGrammar:
    def test_rows_is_spelled_after_a_colon_on_dp(self) -> None:
        assert parse_geometry("dp=2:rows") == ParallelGeometry(data=2, data_mode="rows")
        assert parse_geometry("dp=2:rows,tp=2") == ParallelGeometry(
            data=2, tensor=2, data_mode="rows"
        )
        assert parse_geometry(" dp = 2 : rows ") == ParallelGeometry(
            data=2, data_mode="rows"
        )

    def test_points_may_be_spelled_and_is_the_unspelled_mode(self) -> None:
        assert parse_geometry("dp=2:points") == parse_geometry("dp=2")
        assert parse_geometry("dp=2").data_mode == "points"

    def test_a_mode_on_another_axis_is_refused_naming_that_axis(self) -> None:
        # the twin: the same count without the mode parses
        assert parse_geometry("tp=2") == ParallelGeometry(tensor=2)
        err = _refusal("tp=2:rows")
        assert err.code == "P4" and err.path == "--parallel.tensor"
        assert "only the data axis has modes" in str(err)
        assert _refusal("pp=2:points").path == "--parallel.pipeline"

    def test_an_unknown_mode_is_refused_naming_the_modes(self) -> None:
        assert parse_geometry("dp=2:rows").data_mode == "rows"
        err = _refusal("dp=2:cols")
        assert err.code == "P4" and err.path == "--parallel.data"
        assert "'cols'" in str(err) and "points, rows" in str(err)

    def test_a_near_miss_is_suggested(self) -> None:
        assert "rows" in str(_refusal("dp=2:row"))

    def test_an_empty_mode_is_refused(self) -> None:
        err = _refusal("dp=2:")
        assert err.path == "--parallel.data" and "no mode after the colon" in str(err)

    def test_a_second_colon_is_an_unknown_mode(self) -> None:
        assert "rows:points" in str(_refusal("dp=2:rows:points"))

    def test_the_count_before_the_colon_is_still_a_rank_count(self) -> None:
        assert _refusal("dp=0:rows").path == "--parallel.data"
        assert "rank count" in str(_refusal("dp=x:rows"))

    def test_rows_at_one_replica_parses_and_is_a_world_of_one(self) -> None:
        geometry = parse_geometry("dp=1:rows")
        assert geometry.world == 1 and geometry.data_mode == "rows"
        assert geometry != ONE  # the mode is recorded, points is the default


@pytest.mark.unit
class TestFormat:
    def test_the_default_mode_is_not_spelled(self) -> None:
        assert format_geometry(ParallelGeometry(data=2)) == "dp=2,pp=1,cp=1,tp=1,ep=1"

    def test_rows_is_spelled_on_dp_alone(self) -> None:
        assert (
            format_geometry(ParallelGeometry(data=2, tensor=4, data_mode="rows"))
            == "dp=2:rows,pp=1,cp=1,tp=4,ep=1"
        )


def _geometries() -> st.SearchStrategy[ParallelGeometry]:
    axis = st.integers(min_value=1, max_value=4)
    return st.builds(
        ParallelGeometry,
        data=axis,
        pipeline=axis,
        context=axis,
        tensor=axis,
        expert=axis,
        data_mode=st.sampled_from(DATA_MODES),
    )


@pytest.mark.property
@_SETTINGS
@given(_geometries())
def test_parse_of_format_is_the_identity_over_both_modes(
    geometry: ParallelGeometry,
) -> None:
    assert parse_geometry(format_geometry(geometry)) == geometry


# --------------------------------------------------------------------------- #
# check_rows: the two document rules
# --------------------------------------------------------------------------- #

ROWS2 = ParallelGeometry(data=2, data_mode="rows")


@pytest.mark.unit
class TestCheckRows:
    def test_points_mode_has_no_document_rule(self) -> None:
        assert check_rows(ParallelGeometry(data=2), (None,)) == ()
        assert check_rows(ParallelGeometry(data=8), (1, 1)) == ()
        assert check_rows(ONE, ()) == ()

    def test_a_train_document_with_enough_pairs_is_accepted(self) -> None:
        assert check_rows(ROWS2, (2,)) == ()
        assert check_rows(ROWS2, (16, 2)) == ()

    def test_a_document_with_no_train_is_refused_naming_the_data_axis(self) -> None:
        (text,) = check_rows(ROWS2, (None,))
        assert text.startswith("--parallel.data: dp=2:rows")
        assert "declares no train" in text and "dp=2 over points" in text

    def test_a_sweep_with_a_point_lacking_train_is_refused(self) -> None:
        assert len(check_rows(ROWS2, (16, None))) == 1

    def test_an_empty_campaign_has_nothing_to_split(self) -> None:
        assert len(check_rows(ROWS2, ())) == 1

    def test_pairs_below_the_replica_count_is_refused_beside_its_twin(self) -> None:
        assert check_rows(ParallelGeometry(data=3, data_mode="rows"), (3,)) == ()
        (text,) = check_rows(ParallelGeometry(data=3, data_mode="rows"), (2,))
        assert text.startswith("--parallel.data: dp=3:rows")
        assert "train.batch.pairs=2" in text and "at least dp" in text

    def test_the_smallest_point_decides(self) -> None:
        (text,) = check_rows(ROWS2, (16, 1))
        assert "train.batch.pairs=1" in text

    def test_rows_at_one_replica_needs_a_train_document_too(self) -> None:
        """``dp=1:rows`` splits nothing, but the word still means a fit."""
        assert check_rows(ParallelGeometry(data_mode="rows"), (1,)) == ()
        assert len(check_rows(ParallelGeometry(data_mode="rows"), (None,))) == 1


@pytest.mark.property
@_SETTINGS
@given(
    data=st.integers(min_value=1, max_value=8),
    pairs=st.lists(
        st.one_of(st.none(), st.integers(min_value=1, max_value=12)),
        min_size=0,
        max_size=5,
    ),
)
def test_check_rows_accepts_iff_every_point_trains_with_enough_pairs(
    data: int, pairs: list[int | None]
) -> None:
    geometry = ParallelGeometry(data=data, data_mode="rows")
    refusals = check_rows(geometry, pairs)
    holds = bool(pairs) and all(p is not None and p >= data for p in pairs)
    assert (refusals == ()) == holds
    assert all(text.startswith("--parallel.data:") for text in refusals)
    assert len(refusals) <= 1  # the first failing rule is the refusal
    # the points mode never reads the document
    assert check_rows(ParallelGeometry(data=data), pairs) == ()
