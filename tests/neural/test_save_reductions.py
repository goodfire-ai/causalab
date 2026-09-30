"""§2.12 ``reduce`` — one test per verb, and the property they all share.

The property is the reason ``reduce`` exists: the reduction happens **where
the rows are gathered**, so the un-reduced harvest never reaches disk. An
ablation grid over an 8B model's layers is gigabytes of activations for
kilobytes of means, and a verb that returned ``(n, width)`` — or reduced
downstream of the writer — would quietly give the gigabytes back.

So every verb is checked for the shape *and* for its value. The values are
computed by hand from a table small enough to read, because each verb had a
choice to make that the name does not settle: whether ``std`` is the sample or
the population estimator, whether ``median`` interpolates, what ``count``
counts. A test that recomputed the value with the same torch call it is
testing would assert none of that.
"""

from __future__ import annotations

import pytest
import torch

from causalab.neural.shared.results import _reduce_rows
from causalab.protocol.rules.errors import ProtocolError
from causalab.protocol.schema import SAVE_REDUCTIONS

pytestmark = pytest.mark.numerical_unit

#: Four rows, two columns. Column 1 carries an outlier so `mean` and `median`
#: cannot agree by accident, and an even row count so `median`'s tie rule is
#: visible.
ROWS = torch.tensor(
    [
        [1.0, 2.0],
        [2.0, 4.0],
        [3.0, 6.0],
        [4.0, 400.0],
    ]
)

#: Hand-computed, per column of ROWS.
EXPECTED: dict[str, list[float]] = {
    # (1+2+3+4)/4 ; (2+4+6+400)/4
    "mean": [2.5, 103.0],
    "sum": [10.0, 412.0],
    # sample sd, n-1 (not the population sd, which would give 1.118 / 171.48):
    #   col 0: sqrt(((1.5)^2 + (0.5)^2 + (0.5)^2 + (1.5)^2) / 3) = sqrt(5/3)
    #   col 1: sqrt((101^2 + 99^2 + 97^2 + 297^2) / 3) = sqrt(117620/3)
    "std": [1.2909944, 198.00673],
    # even n, no interpolation: the LOWER middle value. 2.0 not 2.5; 4.0 not 5.0
    "median": [2.0, 4.0],
    "count": [4.0, 4.0],
}


def test_every_verb_in_the_vocabulary_is_covered() -> None:
    """A per-verb test file is only a guarantee while it covers the verbs."""
    assert set(EXPECTED) == set(SAVE_REDUCTIONS)


@pytest.mark.parametrize("verb", SAVE_REDUCTIONS)
def test_the_unreduced_rows_never_reach_disk(verb: str) -> None:
    """The whole point of §2.12's ``reduce``, per verb.

    ``(rows, width) -> (width,)``: what the writer receives is already the
    statistic, so there is no path by which the rows themselves are written.
    """
    out = _reduce_rows(ROWS, verb)

    assert out.shape == (ROWS.shape[-1],), (
        f"{verb} returned {tuple(out.shape)}; §2.12 requires (width,), which is "
        "what keeps the un-reduced rows off disk"
    )
    assert out.dtype is torch.float32  # fp32 accumulation, whatever the run's dtype
    assert out.is_contiguous()


@pytest.mark.parametrize("verb", SAVE_REDUCTIONS)
def test_each_verb_computes_what_the_spec_says(verb: str) -> None:
    out = _reduce_rows(ROWS, verb)
    assert torch.allclose(out, torch.tensor(EXPECTED[verb]), rtol=1e-5), (
        f"{verb}: got {out.tolist()}, §2.12's table says {EXPECTED[verb]}"
    )


def test_std_of_one_row_is_nan_not_zero() -> None:
    """The sample estimator, stated as a value.

    One observation has no spread, and ``0.0`` would read as "measured, and
    there is none". ``NaN`` is the honest answer and §2.12 says so.
    """
    out = _reduce_rows(ROWS[:1], "std")
    assert torch.isnan(out).all()


def test_count_records_a_shorter_harvest() -> None:
    """What ``count`` is for: a `mean` alone cannot tell 4 rows from 2."""
    assert _reduce_rows(ROWS[:2], "count").tolist() == [2.0, 2.0]


def test_sum_over_count_reconstructs_the_mean() -> None:
    """The composability claim in §2.12's table, checked rather than asserted:
    a mean over two shards is the shards' sums over their counts, and is *not*
    the mean of their means once the row counts differ."""
    first, second = ROWS[:1], ROWS[1:]
    total = _reduce_rows(first, "sum") + _reduce_rows(second, "sum")
    rows = _reduce_rows(first, "count") + _reduce_rows(second, "count")

    assert torch.allclose(total / rows, _reduce_rows(ROWS, "mean"))
    mean_of_means = (_reduce_rows(first, "mean") + _reduce_rows(second, "mean")) / 2
    assert not torch.allclose(mean_of_means, _reduce_rows(ROWS, "mean"))


def test_a_ragged_harvest_reduces_over_its_flat_rows() -> None:
    """A ragged read's rows differ in count; the reduction is over all of them.

    ``count`` is what makes that legible after the fact — the file says how
    many rows the number came from.
    """
    from causalab.neural.shared.executor import RaggedValue

    flat = torch.tensor([[1.0, 1.0], [3.0, 3.0], [5.0, 5.0]])
    ragged = RaggedValue(flat=flat, widths=(2, 1))

    assert _reduce_rows(ragged, "count").tolist() == [3.0, 3.0]
    assert _reduce_rows(ragged, "mean").tolist() == [3.0, 3.0]


def test_an_unknown_verb_is_refused() -> None:
    """The vocabulary is closed at the executor too, not only at the parse."""
    with pytest.raises(ProtocolError, match="unknown save reduction"):
        _reduce_rows(ROWS, "geometric_mean")
