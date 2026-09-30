"""Direct ``is_original`` plumbing tests for [`ComponentIndexer`][causalab.tasks.token_positions.ComponentIndexer], now
homed in ``causalab.tasks.token_positions`` (relocated verbatim from the
retired ``causalab.neural.units``, and moved from ``causalab.neural`` to
``causalab.tasks`` beside its only consumers; ``TokenPosition`` subclasses it there).

These pin the contract that whether the wrapped indexer accepts
``is_original`` is decided once, from its *signature* at construction
(``_indexer_accepts_is_original``) — never by catching a ``TypeError`` at
call time. Catching the ``TypeError`` conflated "the indexer does not take
the flag" with "the indexer took the flag but a bug inside it raised",
silently re-running such an indexer *without* the flag; for a paired
position that returns base positions on a counterfactual read — a
wrong-position intervention instead of a crash.

Provenance:

* The round-trip cases (both flag values, batched propagation, the omitted
  default) were extracted from the legacy
  ``tests/neural/test_is_original_flag.py``; the ``TokenPosition``-specific
  ``is_original`` cases went to ``tests/neural/test_token_positions.py``.
* The signature-detection dispatch matrix
  (`TestComponentIndexerIsOriginalDispatch`) is re-homed here from
  ``tests/neural/test_units.py``, deleted along with the
  unit surface it otherwise covered.
* The retired ``AtomicModelUnit.index_component`` forwarding case was
  dropped with that class — the arrow-syntax interchange path
  (``var1<-var2``) now reaches ``ComponentIndexer`` through the position
  bridge, so the contract lives entirely at this level.
"""

from __future__ import annotations

import pytest

from causalab.tasks.token_positions import (
    ComponentIndexer,
    _indexer_accepts_is_original,
)


class TestComponentIndexerUnit:
    """[`ComponentIndexer`][causalab.tasks.token_positions.ComponentIndexer] wraps a per-input position-lookup callable
    used by every dynamic-indexed intervention site (re-homed from the
    deleted ``tests/neural/test_units.py``)."""

    pytestmark = pytest.mark.unit

    def test_init_stores_indexer_and_id(self) -> None:
        ci = ComponentIndexer(lambda _: [0], id="idxID")
        assert ci.id == "idxID"
        assert callable(ci.indexer)

    def test_repr_carries_id(self) -> None:
        ci = ComponentIndexer(lambda _: [0], id="idxID")
        assert "idxID" in repr(ci)

    def test_index_single_input_delegates(self) -> None:
        ci = ComponentIndexer(lambda x: [x, x + 1], id="shift")
        assert ci.index(5) == [5, 6]

    def test_index_batch_preserves_order(self) -> None:
        ci = ComponentIndexer(lambda x: [x], id="echo")
        assert ci.index([3, 1, 2], batch=True) == [[3], [1], [2]]


class TestComponentIndexerProperty:
    """Indexer wrapping must be transparent for the pipeline contract
    (re-homed from the deleted ``tests/neural/test_units.py``)."""

    pytestmark = pytest.mark.property

    @pytest.mark.parametrize(
        "inputs",
        [
            [0],
            [1, 2, 3],
            [5, 5, 5, 5],
        ],
    )
    def test_batch_index_equals_per_element_index(self, inputs: list[int]) -> None:
        ci = ComponentIndexer(lambda x: [x * 2], id="double")
        batched = ci.index(inputs, batch=True)
        per_element = [ci.index(x) for x in inputs]
        assert batched == per_element

    def test_index_returns_what_the_wrapped_callable_returns(self) -> None:
        def f(x):
            return list(range(x))

        ci = ComponentIndexer(f, id="range")
        for x in (0, 1, 4):
            assert ci.index(x) == f(x)


class TestComponentIndexerIsOriginal:
    """``ComponentIndexer`` must forward ``is_original`` to indexers that
    accept it and silently no-op for legacy indexers that do not."""

    pytestmark = pytest.mark.unit

    def test_indexer_with_is_original_parameter(self) -> None:
        """Flag-aware indexer receives both ``True`` and ``False``."""

        def position_indexer(input, is_original=True):
            return [0, 1, 2] if is_original else [3, 4, 5]

        indexer = ComponentIndexer(position_indexer, id="test_indexer")
        assert indexer.index({"text": "test"}, is_original=True) == [0, 1, 2]
        assert indexer.index({"text": "test"}, is_original=False) == [3, 4, 5]

    def test_indexer_without_is_original_parameter(self) -> None:
        """Legacy indexer (no ``is_original`` arg) still works under both
        flag values: signature detection sees no flag, so it's called without
        one (no ``TypeError``-swallowing fallback involved)."""

        def old_indexer(input):
            return [0, 1, 2]

        indexer = ComponentIndexer(old_indexer, id="old_indexer")
        assert indexer.index({"text": "test"}, is_original=True) == [0, 1, 2]
        assert indexer.index({"text": "test"}, is_original=False) == [0, 1, 2]

    def test_batch_indexing_with_is_original(self) -> None:
        """``batch=True`` must propagate ``is_original`` to every element."""

        def position_indexer(input, is_original=True):
            base_pos = input.get("pos", 0)
            return [base_pos] if is_original else [base_pos + 10]

        indexer = ComponentIndexer(position_indexer, id="batch_indexer")
        batch = [{"pos": 0}, {"pos": 1}, {"pos": 2}]
        assert indexer.index(batch, batch=True, is_original=True) == [[0], [1], [2]]
        assert indexer.index(batch, batch=True, is_original=False) == [[10], [11], [12]]

    def test_default_is_original_omits_kwarg(self) -> None:
        """When ``is_original`` is not passed, the wrapper must NOT inject
        it; the indexer's own default takes effect."""

        def position_indexer(input, is_original=True):
            return [0] if is_original else [1]

        indexer = ComponentIndexer(position_indexer, id="default_test")
        # No is_original passed → indexer's default (True) wins.
        assert indexer.index({"text": "test"}) == [0]


class TestComponentIndexerIsOriginalDispatch:
    """``is_original`` is threaded by signature detection at construction, never
    by catching a ``TypeError`` at call time.

    Tier: unit (class-scoped since the module mixes tiers).

    The old code wrapped the flagged call in ``try/except TypeError`` and, on
    *any* ``TypeError``, re-invoked the indexer without the flag. That conflated
    "indexer does not take ``is_original``" with "indexer took it but a bug
    inside raised", silently re-running a flag-aware indexer without the flag —
    for a paired position, returning base positions on a counterfactual read.
    """

    pytestmark = pytest.mark.unit

    def test_internal_typeerror_propagates(self) -> None:
        """A ``TypeError`` raised *inside* an ``is_original``-accepting indexer
        must propagate — not be swallowed and the indexer silently re-run."""
        sentinel = "boom-from-inside-the-indexer"

        def buggy(x, is_original=True):
            raise TypeError(sentinel)

        ci = ComponentIndexer(buggy, id="buggy")
        with pytest.raises(TypeError, match=sentinel):
            ci.index("inp", is_original=False)

    def test_internal_typeerror_not_masked_as_base_positions(self) -> None:
        """The exact hazard: a flag-aware indexer that raises on its CF
        branch must crash, not fall back to the (base) ``is_original=True``
        branch — which would be a silent wrong-position intervention."""

        def paired(x, is_original=True):
            if not is_original:
                raise TypeError("downstream bug on the counterfactual branch")
            return [0]  # base positions

        ci = ComponentIndexer(paired, id="paired")
        with pytest.raises(TypeError):
            ci.index("cf-input", is_original=False)

    def test_flag_threaded_when_accepted(self) -> None:
        seen: list[bool] = []

        def indexer(x, is_original=True):
            seen.append(is_original)
            return [0] if is_original else [1]

        ci = ComponentIndexer(indexer, id="aware")
        assert ci.index("x", is_original=True) == [0]
        assert ci.index("x", is_original=False) == [1]
        assert seen == [True, False]

    # -- signature detection over the four indexer shapes ------------------- #
    def test_detect_keyword_is_original(self) -> None:
        # POSITIONAL_OR_KEYWORD `is_original` → reachable by keyword.
        def kw(x, is_original=True):
            return [1] if is_original else [2]

        assert _indexer_accepts_is_original(kw) is True
        assert ComponentIndexer(kw, id="kw")._accepts_is_original is True

    def test_detect_keyword_only_is_original(self) -> None:
        def kwonly(x, *, is_original=True):
            return [1] if is_original else [2]

        assert _indexer_accepts_is_original(kwonly) is True
        assert ComponentIndexer(kwonly, id="kwonly")._accepts_is_original is True

    def test_detect_var_keyword(self) -> None:
        # A ``**kwargs`` catch-all absorbs ``is_original``.
        def varkw(x, **kwargs):
            return [1] if kwargs.get("is_original", True) else [2]

        assert _indexer_accepts_is_original(varkw) is True
        ci = ComponentIndexer(varkw, id="varkw")
        assert ci._accepts_is_original is True
        assert ci.index("x", is_original=False) == [2]

    def test_detect_positional_only_is_original_not_reachable(self) -> None:
        # A positional-only ``is_original`` cannot be passed by keyword, so it
        # counts as unsupported: the indexer is called with no kwarg (its
        # default wins) and, crucially, no ``TypeError`` from a bad keyword.
        def posonly(x, is_original=True, /):
            return [1] if is_original else [2]

        assert _indexer_accepts_is_original(posonly) is False
        ci = ComponentIndexer(posonly, id="posonly")
        assert ci._accepts_is_original is False
        assert ci.index("x", is_original=False) == [1]

    def test_detect_no_flag(self) -> None:
        def noflag(x):
            return [0]

        assert _indexer_accepts_is_original(noflag) is False
        assert ComponentIndexer(noflag, id="noflag")._accepts_is_original is False

    def test_detect_uninspectable_callable_is_unsupported(self) -> None:
        # Some C builtins have no introspectable signature (``inspect.signature``
        # raises ``ValueError``); the except branch treats them as no-flag.
        assert _indexer_accepts_is_original(print) is False
