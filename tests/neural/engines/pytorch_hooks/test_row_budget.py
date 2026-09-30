"""The rows one grad forward of a fit may cover, measured when unset (spec §8).

An authored ``fit_rows`` is a fixed bound. Unset, the cohort's first step runs
its first member alone as a probe under peak-memory tracking and the bound is
what the device's free memory holds at that slope; a window that still runs
out of memory is retried at half the rows. The bound a cohort ran under is
reported so an author can pin it. CUDA is simulated here through the meter
seam — the simulator's scripted ``SimulatedMeter`` (``docs/model_parallelism.md``
§10.1), one reading per probe, so the number of probes a budget makes is
pinned by the script's length — and the out-of-memory error raised by hand.
"""

# the cohort suite's builders are reused here
# pyright: reportPrivateUsage=false

from __future__ import annotations

from typing import Any, Sequence

import pytest
import torch

from causalab.neural.engines.pytorch_hooks import cohort as cohort_module
from causalab.neural.engines.pytorch_hooks import train as train_module
from causalab.neural.engines.pytorch_hooks.budget import MARGIN, Meter, RowBudget
from causalab.neural.engines.pytorch_hooks.loading import ModelBundle, load_model
from causalab.neural.engines.pytorch_hooks.train import run_cohort_training

from .conftest import TINY_LLAMA
from ._drive import executor_for
from .test_fit_cohort import (
    B,
    EVAL_ROWS,
    PAIRS,
    _assert_same_fit,
    _batch_sizes,
    _campaign,
    _request,
    _train_doc,
)
from .test_train import ANSWERS, BASES, COUNTERFACTUALS
from tests._helpers.simulated_world import RankMeter, SimulatedMeter

pytestmark = pytest.mark.unit


def _meter(
    per_row: int, available: int, *, rows: int = 1, probes: int = 1
) -> RankMeter:
    """A device with ``available`` bytes to give and ``per_row`` bytes per
    row, probed ``probes`` times at ``rows`` rows: the simulator's meter for
    one rank, its script one ``(peak, available)`` reading per probe. A
    probe the script has no reading for is the simulator's refusal, so
    ``probes`` pins how often the budget measures."""
    return SimulatedMeter({0: [(per_row * rows, available)] * probes}).for_rank(0)


def _size(item: tuple[str, int]) -> int:
    return item[1]


class TestRowBudget:
    def test_a_fixed_bound_packs_and_never_probes(self) -> None:
        budget = RowBudget.of(4, meter=_meter(1, 1, probes=0))
        assert budget.fixed and not budget.probing and budget.bound == 4
        items = [("a", 2), ("b", 2), ("c", 3), ("d", 2)]
        window, rest = budget.take(items, _size)
        assert window == [("a", 2), ("b", 2)] and rest == [("c", 3), ("d", 2)]
        window, rest = budget.take(rest, _size)
        assert window == [("c", 3)] and rest == [("d", 2)]
        assert not budget.can_shrink(4, 2)

    def test_auto_without_a_meter_is_unbounded(self) -> None:
        budget = RowBudget.of(None, meter=None)
        assert not budget.probing and budget.bound is None
        items = [("a", 2), ("b", 2), ("c", 2)]
        assert budget.take(items, _size) == (items, [])

    def test_auto_probes_one_member_then_packs_under_the_measured_bound(self) -> None:
        # 100 bytes a row, 1000 available: 900 usable at the margin → 9 rows
        meter = _meter(per_row=100, available=1000, rows=2)
        budget = RowBudget.of(None, meter=meter)
        items = [("a", 2), ("b", 2), ("c", 2), ("d", 2), ("e", 2), ("f", 2)]
        window, rest = budget.take(items, _size)
        assert window == [("a", 2)], "the probe is one member"
        ran: list[str] = []
        budget.run(2, lambda: ran.append("probe"))
        assert ran == ["probe"] and budget.resolved
        # 9 rows fit; floored to whole members of 2 rows: 8
        assert int(1000 * (1 - MARGIN) / 100) == 9 and budget.bound == 8
        assert budget.probe == (100, 1000)
        window, rest = budget.take(rest, _size)
        assert [name for name, _ in window] == ["b", "c", "d", "e"]  # 8 ≤ 8
        assert rest == [("f", 2)]

    def test_the_floor_is_the_cohort_s_smallest_member(self) -> None:
        """Members of 3 and 8 rows, probe 8: 23 rows fit; floored to the
        smallest member (3) the bound is 21, not 16 — a 3-row and an 8-row
        member still share a forward. Without a unit the probe's rows floor it."""
        meter = _meter(
            per_row=100, available=int(23 * 100 / (1 - MARGIN)) + 1, rows=8, probes=2
        )
        budget = RowBudget.of(None, meter=meter)
        budget.run(8, lambda: None, unit=3)
        assert budget.bound == 21
        again = RowBudget.of(None, meter=meter)
        again.run(8, lambda: None)
        assert again.bound == 16

    def test_the_measured_bound_is_never_below_the_probe(self) -> None:
        meter = _meter(per_row=10**9, available=1, rows=4)
        budget = RowBudget.of(None, meter=meter)
        budget.run(4, lambda: None)
        assert budget.bound == 4

    def test_a_window_out_of_memory_halves_but_not_below_a_member(self) -> None:
        budget = RowBudget.of(None, meter=None)
        assert budget.can_shrink(8, 2)
        budget.shrink(8, 2)
        assert budget.bound == 4 and budget.shrinks == 1
        budget.shrink(4, 2)
        assert budget.bound == 2
        assert not budget.can_shrink(2, 2)


@pytest.fixture(scope="module")
def bundle() -> ModelBundle:
    return load_model(TINY_LLAMA)


def _fit(
    raws: Sequence[dict[str, Any]],
    bundle: ModelBundle,
    meter: Meter | None,
    *,
    fit_rows: int | None = None,
    batch_rows: int | None = None,
) -> tuple[list[Any], list[int]]:
    _docs, handles = _campaign(raws)
    executors = [
        executor_for(
            raw,
            bundle,
            base_texts=BASES,
            counterfactual_texts=COUNTERFACTUALS,
            extra_columns={"label": ANSWERS},
            interning=handle,
            batch_rows=batch_rows,
        )
        for raw, handle in zip(raws, handles)
    ]
    with _batch_sizes(bundle) as sizes:
        outcomes = run_cohort_training(
            [ex.doc for ex in executors],
            executors,
            _request(),
            fit_rows=fit_rows,
            meter=meter,
        )
    return outcomes, sizes


def _flat_meter(members: int) -> RankMeter:
    """A device that affords exactly ``members`` members' rows per forward at
    the one slope it is asked about — the cohort's single probe: flat
    ``per_row``, ``available`` sized so ``members × probe rows`` fit under
    the margin."""
    return _meter(
        per_row=100, available=int(100 * members * PAIRS / (1 - MARGIN)) + 1, rows=PAIRS
    )


class TestCohortUnderAutoBudget:
    def test_the_first_step_probes_then_the_cohort_packs_under_the_bound(
        self, bundle: ModelBundle
    ) -> None:
        """Three members of ``PAIRS`` rows with a meter that affords two members
        per forward: step one is the probe (one member) plus a window of two,
        every later step two windows of two and one; the outcome carries the
        bound the cohort ran under."""
        epochs = 2
        raws = [_train_doc(k=k, epochs=epochs) for k in (2, 4, 8)]
        # eval windows pinned to one member so their sizes stay out of the count
        outcomes, sizes = _fit(raws, bundle, _flat_meter(2), batch_rows=len(EVAL_ROWS))
        assert all(o.fit_rows == 2 * PAIRS for o in outcomes)
        assert all(o.fit_rows_shrinks == 0 for o in outcomes)
        steps = B * epochs
        # grad forwards: the probe alone, then windows of 2 and 1 members
        assert sizes.count(2 * PAIRS) == steps
        assert sizes.count(3 * PAIRS) == 0
        # solo grad forwards: the probe + one per later step, beside the source slices
        assert sizes.count(PAIRS) == B + 1 + (steps - 1)

    def test_a_fixed_bound_reports_itself_and_probes_nothing(
        self, bundle: ModelBundle
    ) -> None:
        raws = [_train_doc(k=k, epochs=1) for k in (2, 4)]
        outcomes, sizes = _fit(
            raws,
            bundle,
            _meter(per_row=1, available=10**9, probes=0),
            fit_rows=2 * PAIRS,
        )
        assert all(o.fit_rows == 2 * PAIRS for o in outcomes)
        assert sizes.count(2 * PAIRS) == B  # every step batched, none probed

    def test_unbounded_off_cuda_reports_no_bound(self, bundle: ModelBundle) -> None:
        raws = [_train_doc(k=k, epochs=1) for k in (2, 4)]
        outcomes, _sizes = _fit(raws, bundle, None)
        assert all(o.fit_rows is None for o in outcomes)

    def test_the_eval_passes_pack_under_batch_rows(self, bundle: ModelBundle) -> None:
        """Three members' eval passes over a 3-row split under ``batch_rows``
        6: two forwards per eval (6 rows, then 3), never one of 9. The grad
        forwards are pinned solo so no grad window is 6 rows wide."""
        epochs = 2
        raws = [_train_doc(k=k, epochs=epochs) for k in (2, 4, 8)]
        _outcomes, sizes = _fit(raws, bundle, None, fit_rows=PAIRS, batch_rows=6)
        assert sizes.count(2 * len(EVAL_ROWS)) == epochs
        assert sizes.count(3 * len(EVAL_ROWS)) == 0

    def test_the_eval_passes_pack_under_the_fit_bound_when_batch_rows_is_unset(
        self, bundle: ModelBundle
    ) -> None:
        """With no ``batch_rows`` the eval passes share the fit's own budget —
        the grad bound measured on the first step, 2·PAIRS = 4 rows here — so
        the 3-row eval members go one per window: never two, never three. One
        number (``fit_rows_resolved``) then pins both kinds of window."""
        epochs = 2
        raws = [_train_doc(k=k, epochs=epochs) for k in (2, 4, 8)]
        outcomes, sizes = _fit(raws, bundle, _flat_meter(2))
        assert sizes.count(3 * len(EVAL_ROWS)) == 0
        assert sizes.count(2 * len(EVAL_ROWS)) == 0
        # three solo eval members per pass, plus the split's one source forward
        assert sizes.count(len(EVAL_ROWS)) == 3 * epochs + 1
        assert all(o.fit_rows == 2 * PAIRS for o in outcomes)

    def test_a_window_out_of_memory_is_retried_at_half_the_rows(
        self, bundle: ModelBundle, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A grad forward over more than two members' rows raises the
        allocator's error once; the loop leaves the handler, zeroes the
        window's gradients, halves the bound and reruns it. The fit is the one
        a fixed ``fit_rows`` would give, and the outcome counts the shrink."""
        epochs = 2
        raws = [_train_doc(k=k, epochs=epochs) for k in (2, 4, 8, 16)]
        real = cohort_module.run_groups
        raised: list[int] = []

        def fragile(entries: Sequence[Any]) -> None:
            rows = sum(len(entry.executor.rows_for_metrics()) for entry in entries)
            # a grad forward over more than two members' rows; the no-grad eval
            # passes are bounded by `batch_rows`, not `fit_rows`
            if entries[0].executor.grad_enabled and rows > 2 * PAIRS:
                raised.append(rows)
                raise torch.OutOfMemoryError("CUDA out of memory (simulated)")
            real(entries)

        monkeypatch.setattr(train_module, "run_groups", fragile)
        outcomes, sizes = _fit(raws, bundle, None)
        assert raised == [4 * PAIRS], "one window of four members, refused once"
        # 4·PAIRS halved is 2·PAIRS: two members per forward from then on
        assert all(o.fit_rows == 2 * PAIRS for o in outcomes)
        assert all(o.fit_rows_shrinks == 1 for o in outcomes)
        assert sizes.count(2 * PAIRS) == 2 * B * epochs
        monkeypatch.setattr(train_module, "run_groups", real)
        fixed, _sizes = _fit(raws, bundle, None, fit_rows=2 * PAIRS)
        for a, b in zip(outcomes, fixed):
            _assert_same_fit(a, b, atol=1e-5, rtol=1e-4)

    def test_an_eval_window_out_of_memory_leaves_the_grad_bound_alone(
        self, bundle: ModelBundle, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The eval passes pack in a budget of their own, seeded from the
        measured grad bound: an eval window that runs out of memory is retried
        at half its rows, the shrink is kept for every later pass (one doomed
        forward per fit, not per pass), and the grad windows keep their bound;
        the outcome reports the bound every window ran under — the eval
        budget's, once it shrank below the grad bound — with the shrink
        counted beside it, so what an author pins is what no window refused."""
        epochs = 3
        raws = [_train_doc(k=k, epochs=epochs) for k in (2, 4, 8)]
        real = cohort_module.run_groups
        raised: list[int] = []

        def fragile(entries: Sequence[Any]) -> None:
            rows = sum(len(entry.executor.rows_for_metrics()) for entry in entries)
            if not entries[0].executor.grad_enabled and rows > len(EVAL_ROWS):
                raised.append(rows)
                raise torch.OutOfMemoryError("CUDA out of memory (simulated)")
            real(entries)

        monkeypatch.setattr(train_module, "run_groups", fragile)
        # a measured grad bound of 3·PAIRS = 6 rows: two 3-row eval members fit
        outcomes, sizes = _fit(raws, bundle, _flat_meter(3))
        assert raised == [2 * len(EVAL_ROWS)], "one doomed eval window, then learnt"
        assert all(
            o.fit_rows == len(EVAL_ROWS) and o.fit_rows_shrinks == 1 for o in outcomes
        )
        # the grad windows never shrank: every step still packed all three members
        assert max(sizes) == 3 * PAIRS
        # after the shrink every eval member ran alone, plus the split's source
        assert sizes.count(len(EVAL_ROWS)) == 3 * epochs + 1

    def test_an_eval_shrink_under_an_unresolved_grad_bound_reports_null(
        self, bundle: ModelBundle, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Off CUDA nothing measures the grad bound and ``null`` means every
        member in one forward: an eval window that runs out of memory there
        shrinks its own budget, is counted, and leaves the reported bound
        ``None`` — the eval bound is not a number the grad windows ran under."""
        raws = [_train_doc(k=k, epochs=1) for k in (2, 4, 8)]
        real = cohort_module.run_groups
        raised: list[int] = []

        def fragile(entries: Sequence[Any]) -> None:
            rows = sum(len(entry.executor.rows_for_metrics()) for entry in entries)
            if not entries[0].executor.grad_enabled and rows > len(EVAL_ROWS):
                raised.append(rows)
                raise torch.OutOfMemoryError("CUDA out of memory (simulated)")
            real(entries)

        monkeypatch.setattr(train_module, "run_groups", fragile)
        outcomes, sizes = _fit(raws, bundle, None)
        assert raised == [3 * len(EVAL_ROWS)], "the whole eval window, once"
        assert all(o.fit_rows is None and o.fit_rows_shrinks == 1 for o in outcomes)
        assert max(sizes) == 3 * PAIRS, "the grad windows stayed whole"

    def test_an_authored_fit_rows_is_never_shrunk_for_eval_either(
        self, bundle: ModelBundle, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An authored bound is fixed for every window it governs: an eval
        window that does not fit under an authored ``fit_rows`` re-raises
        rather than quietly packing differently from the pinned geometry."""
        raws = [_train_doc(k=k, epochs=1) for k in (2, 4, 8)]
        raised: list[int] = []

        def fragile(entries: Sequence[Any]) -> None:
            rows = sum(len(entry.executor.rows_for_metrics()) for entry in entries)
            # only a window of several eval members fails: a budget that
            # shrank would retry the members alone and the fit would complete
            if not entries[0].executor.grad_enabled and rows > len(EVAL_ROWS):
                raised.append(rows)
                raise torch.OutOfMemoryError("CUDA out of memory (simulated)")
            cohort_module.run_groups(entries)

        monkeypatch.setattr(train_module, "run_groups", fragile)
        with pytest.raises(torch.OutOfMemoryError):
            _fit(raws, bundle, None, fit_rows=3 * PAIRS)
        assert raised == [2 * len(EVAL_ROWS)], "refused once, never re-packed"

    def test_a_single_member_out_of_memory_is_re_raised(
        self, bundle: ModelBundle, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        raws = [_train_doc(k=2, epochs=1)]

        def always(entries: Sequence[Any]) -> None:
            raise torch.OutOfMemoryError("CUDA out of memory (simulated)")

        monkeypatch.setattr(train_module, "run_groups", always)
        with pytest.raises(torch.OutOfMemoryError):
            _fit(raws, bundle, None)
