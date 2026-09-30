"""Production boundaries are planned without loading a model."""

import pytest

from causalab.workflow.behavioral_plan import plan_batches

pytestmark = pytest.mark.unit


def test_mixed_splits_and_lengths_are_grouped_before_chunking():
    rows = [
        dict(
            example_id=str(i), split=split, prefix_condition="correct", output_length=n
        )
        for i, (split, n) in enumerate(
            [
                ("confirmation", 2),
                ("diagnostic", 2),
                ("confirmation", 3),
                ("confirmation", 2),
                ("diagnostic", 2),
                ("confirmation", 2),
            ]
        )
    ]
    batches = plan_batches(rows, batch_rows=2)
    assert sorted(i for b in batches for i in b.indices) == list(range(len(rows)))
    for batch in batches:
        assert len(batch.indices) <= 2
        assert {rows[i]["split"] for i in batch.indices} == {batch.split}
        assert {rows[i]["output_length"] for i in batch.indices} == {
            batch.output_length
        }
        assert tuple(rows[i]["example_id"] for i in batch.indices) == batch.example_ids
    assert any(len(b.indices) == 1 for b in batches)


@pytest.mark.parametrize(
    "rows",
    [
        [],
        [
            dict(example_id="a", split="development"),
            dict(example_id="a", split="development"),
        ],
        [
            dict(example_id="a", split="development"),
            dict(example_id="b", split="confirmation", counterfactual_example_id="a"),
        ],
        [dict(example_id="a", split="development", output_length=2)],
    ],
)
def test_invalid_identity_or_cohort_is_rejected(rows):
    with pytest.raises(ValueError):
        plan_batches(rows, batch_rows=4)


@pytest.mark.parametrize("change", ["missing", "duplicate", "foreign", "negative"])
def test_qualification_rejects_incomplete_or_foreign_engine_rows(change):
    from causalab.workflow.behavioral import validate_continuation_coverage
    from causalab.protocol.rules.errors import ProtocolError

    labels = ["0", "1", "2"]
    rows = [
        dict(point=0, example_id=label, model="original", input="base")
        for label in labels
    ]
    validate_continuation_coverage(rows, labels, 1, unwritten={"original"})
    if change == "missing":
        rows.pop()
    elif change == "duplicate":
        rows.append(dict(rows[0]))
    else:
        rows[0]["example_id"] = "3" if change == "foreign" else -1
    with pytest.raises(ProtocolError):
        validate_continuation_coverage(rows, labels, 1, unwritten={"original"})
