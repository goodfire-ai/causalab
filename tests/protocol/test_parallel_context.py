"""The context axis's torch-free half (``docs/model_parallelism.md`` §8.4,
§10.3): ``sequence_chunks`` partitions the padded frame into contiguous
chunks in rank order — equal, the remainder on the last — and refuses a frame
shorter than the group by name; ``check_context`` refuses a decoding document
under ``cp > 1`` and nothing else. The mutation: a chunk table with the
remainder on the first rank is not this rule's answer for any uneven frame.
"""

from __future__ import annotations

import dataclasses
import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from causalab.protocol.rules.errors import ProtocolError
from causalab.protocol.parallel import (
    CONTEXT_EXPERIMENTAL_VARIABLE,
    experimental_context,
    hybrid_family,
    ParallelGeometry,
    check_context,
    sequence_chunks,
)

_SETTINGS = settings(
    deadline=None,
    max_examples=200,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)


@st.composite
def frames(draw: st.DrawFn) -> tuple[int, int]:
    padded_len = draw(st.integers(min_value=1, max_value=300))
    context = draw(st.integers(min_value=1, max_value=padded_len))
    return padded_len, context


@pytest.mark.property
class TestSequenceChunks:
    @_SETTINGS
    @given(frame=frames())
    def test_the_chunks_partition_the_frame_contiguously_in_rank_order(
        self, frame: tuple[int, int]
    ) -> None:
        padded_len, context = frame
        chunks = sequence_chunks(padded_len, context)
        assert len(chunks) == context
        assert chunks[0].start == 0 and chunks[-1].stop == padded_len
        for before, after in zip(chunks, chunks[1:]):
            assert before.stop == after.start  # contiguous, in order
        assert [p for chunk in chunks for p in chunk] == list(range(padded_len))
        assert all(len(chunk) >= 1 for chunk in chunks)

    @_SETTINGS
    @given(frame=frames())
    def test_the_chunks_are_equal_with_the_remainder_on_the_last(
        self, frame: tuple[int, int]
    ) -> None:
        padded_len, context = frame
        chunks = sequence_chunks(padded_len, context)
        per_rank = padded_len // context
        assert all(len(chunk) == per_rank for chunk in chunks[:-1])
        assert len(chunks[-1]) == per_rank + padded_len % context

    @_SETTINGS
    @given(frame=frames())
    def test_every_rank_computes_the_same_table(self, frame: tuple[int, int]) -> None:
        """A pure function of two integers: what lets every rank of a group
        answer without a collective (§3, "never branch on rank")."""
        padded_len, context = frame
        assert sequence_chunks(padded_len, context) == sequence_chunks(
            padded_len, context
        )

    def test_the_remainder_on_the_first_rank_is_not_this_rule(self) -> None:
        """The mutation the smoke tier's off-by-one rests on: a table that
        gives the remainder to the first rank disagrees with the rule at
        every uneven frame, so the model call on one rank and the frame's
        gather on another would slice different positions."""
        for padded_len, context in ((7, 3), (5, 2), (11, 4)):
            per_rank = padded_len // context
            mutated = (
                range(0, per_rank + padded_len % context),
                *(
                    range(
                        per_rank + padded_len % context + k * per_rank,
                        per_rank + padded_len % context + (k + 1) * per_rank,
                    )
                    for k in range(context - 1)
                ),
            )
            assert mutated != sequence_chunks(padded_len, context)


@pytest.mark.unit
class TestRefusals:
    @pytest.mark.parametrize(("padded_len", "context"), [(1, 2), (3, 4), (7, 8)])
    def test_a_frame_shorter_than_the_group_is_refused_by_name(
        self, padded_len: int, context: int
    ) -> None:
        with pytest.raises(ProtocolError) as err:
            sequence_chunks(padded_len, context)
        assert err.value.code == "P4"
        assert "--parallel.context" in str(err.value)
        assert f"cp={context}" in str(err.value)
        assert str(padded_len) in str(err.value)

    @pytest.mark.parametrize("value", [0, -1, True, 2.0, "2"])
    def test_a_non_positive_or_non_integer_argument_is_refused(
        self, value: object
    ) -> None:
        with pytest.raises(ProtocolError) as err:
            sequence_chunks(value, 1)  # type: ignore[arg-type]
        assert "--parallel.context" in str(err.value)
        with pytest.raises(ProtocolError):
            sequence_chunks(4, value)  # type: ignore[arg-type]

    def test_a_group_of_one_holds_the_whole_frame(self) -> None:
        assert sequence_chunks(9, 1) == (range(0, 9),)
        assert sequence_chunks(1, 1) == (range(0, 1),)


@pytest.mark.unit
class TestCheckContext:
    def test_a_decoding_document_is_refused_under_context_above_one(self) -> None:
        refusals = check_context(ParallelGeometry(context=2), True)
        assert len(refusals) == 1
        assert refusals[0].startswith("--parallel.context:")
        assert "cp=2" in refusals[0] and "generated" in refusals[0]
        assert "cp=1" in refusals[0]  # the remedy is named

    def test_nothing_else_is_refused(self) -> None:
        assert check_context(ParallelGeometry(context=2), False) == ()
        assert check_context(ParallelGeometry(context=1), True) == ()
        assert check_context(ParallelGeometry(), False) == ()
        assert check_context(ParallelGeometry(context=4, tensor=2), False) == ()


@pytest.mark.unit
class TestHybridFamilyRule:
    """§8.4: a hybrid tower (a ``linear_attention`` layer among the attention
    layers) buys nothing from context parallelism — the DeltaNet layers run
    in sequence across the group and every rank holds the whole model — so
    ``cp > 1`` is refused by name on it unless the experimental switch waives
    the rule; a dense tower is untouched."""

    def test_a_hybrid_tower_is_refused_by_name(self) -> None:
        refusals = check_context(ParallelGeometry(context=2), False, hybrid=True)
        assert len(refusals) == 1
        text = refusals[0]
        assert text.startswith("--parallel.context:")
        assert "cp=2" in text and "linear_attention" in text
        assert "pp or ep" in text  # the remedy is named
        assert CONTEXT_EXPERIMENTAL_VARIABLE in text  # and the waiver

    def test_the_switch_waives_the_family_rule_only(self) -> None:
        assert (
            check_context(
                ParallelGeometry(context=2), False, hybrid=True, experimental=True
            )
            == ()
        )
        both = check_context(ParallelGeometry(context=2), True, hybrid=True)
        assert len(both) == 2
        waived = check_context(
            ParallelGeometry(context=2), True, hybrid=True, experimental=True
        )
        assert len(waived) == 1 and "generated" in waived[0]

    def test_a_dense_tower_and_a_group_of_one_are_untouched(self) -> None:
        assert check_context(ParallelGeometry(context=2), False, hybrid=False) == ()
        assert check_context(ParallelGeometry(context=1), False, hybrid=True) == ()

    def test_the_facts_are_derived_over_every_model_and_the_switch_only_above_cp_one(
        self,
    ) -> None:
        """``context_refusals`` is what both verbs call: ``hybrid`` folds
        over every model of the campaign — a swept key whose hybrid tower is
        the second point is refused like the first's — and the waiver is
        read only when the context axis is above one, so a malformed switch
        refuses no world-1 or ``tp`` run and ``dry-run`` and ``run`` agree."""
        import dataclasses as dc

        from causalab.protocol.parallel import (
            CONTEXT_EXPERIMENTAL_VARIABLE,
            context_refusals,
        )
        from causalab.protocol.registry import get_model_info

        dense = get_model_info("meta-llama/Llama-3.1-8B")
        streams = ("full_attention",) * (dense.num_layers - 1) + ("linear_attention",)
        hybrid = dc.replace(dense, layer_types=streams)
        cp2 = ParallelGeometry(context=2)

        assert context_refusals(cp2, False, (dense, dense), {}) == ()
        (later,) = context_refusals(cp2, False, (dense, hybrid), {})
        assert "linear_attention" in later
        assert (
            context_refusals(
                cp2, False, (dense, hybrid), {CONTEXT_EXPERIMENTAL_VARIABLE: "1"}
            )
            == ()
        )
        # a malformed switch is read — and refused — only where it could steer
        bad = {CONTEXT_EXPERIMENTAL_VARIABLE: "maybe"}
        assert context_refusals(ParallelGeometry(), True, (hybrid,), bad) == ()
        assert context_refusals(ParallelGeometry(tensor=2), True, (hybrid,), bad) == ()
        with pytest.raises(ProtocolError) as err:
            context_refusals(cp2, False, (dense,), bad)
        assert CONTEXT_EXPERIMENTAL_VARIABLE in str(err.value)

    def test_hybrid_family_reads_the_declared_streams(self) -> None:
        from causalab.protocol.registry import get_model_info

        assert hybrid_family(get_model_info("Qwen/Qwen3.6-35B-A3B"))
        dense = dataclasses.replace(
            get_model_info("Qwen/Qwen3.6-35B-A3B"), layer_types=None
        )
        assert not hybrid_family(dense)
        full = dataclasses.replace(
            dense, layer_types=("full_attention",) * dense.num_layers
        )
        assert not hybrid_family(full)

    @pytest.mark.parametrize(
        ("environ", "expected"),
        [
            ({}, False),
            ({CONTEXT_EXPERIMENTAL_VARIABLE: ""}, False),
            ({CONTEXT_EXPERIMENTAL_VARIABLE: "0"}, False),
            ({CONTEXT_EXPERIMENTAL_VARIABLE: "1"}, True),
        ],
    )
    def test_the_switch_is_read_as_a_switch(
        self, environ: dict[str, str], expected: bool
    ) -> None:
        assert experimental_context(environ) is expected

    @given(st.text(min_size=1).filter(lambda v: v not in ("0", "1")))
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_any_other_value_is_refused_by_name(self, value: str) -> None:
        with pytest.raises(ProtocolError) as err:
            experimental_context({CONTEXT_EXPERIMENTAL_VARIABLE: value})
        assert err.value.code == "P4"
        assert CONTEXT_EXPERIMENTAL_VARIABLE in str(err.value)
