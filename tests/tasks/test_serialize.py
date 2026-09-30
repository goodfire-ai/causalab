"""Task packages → dataset tables (spec §2.2): the column vocabulary, the
determinism the content digest depends on, and the loud failures.

LM-free: the serializer never touches a model or a tokenizer — that is the
property that lets a document's digest be computed without either.
"""

from __future__ import annotations

import json

import pytest

from causalab.io.env import FileDatasets
from causalab.tasks.natural_domains_arithmetic.config import NaturalDomainConfig
from causalab.tasks.serialize import (
    RESERVED_COLUMNS,
    config_class,
    serialize_counterfactual_dataset,
    table_bytes,
    write_dataset_table,
)

pytestmark = pytest.mark.unit


def _weekdays(n: int = 4, seed: int = 0):
    return serialize_counterfactual_dataset(
        "natural_domains_arithmetic",
        n=n,
        seed=seed,
        split="all",
        task_cfg=NaturalDomainConfig(domain_type="weekdays"),
        target_variables=["result"],
    )


def test_row_vocabulary_is_the_documented_one():
    dataset = _weekdays()
    assert len(dataset.rows) == 4
    row = dataset.rows[0]
    # the columns documents reference (§2.2)
    for column in (
        "input",
        "counterfactual_inputs",
        "base_answer",
        "cf_answer",
        "label",
        "counterfactual_inputs_variables",
    ):
        assert column in row, column
    # answer forms from the task's own output_tokens declaration (§2.10)
    assert row["label_forms"] == [row["label"], row["label"].strip()]
    # every causal-model variable is a per-row column (§2.3), stringified
    assert row["entity"] in row["input"]
    assert row["number"] in row["input"]
    # ... and the counterfactual side lives in the per-role sibling
    (cf_variables,) = row["counterfactual_inputs_variables"]
    assert cf_variables["entity"] in row["counterfactual_inputs"][0]


def test_label_is_the_post_intervention_answer_not_the_counterfactuals_own():
    """The trap this column vocabulary exists to avoid: `label` is what the
    causal model outputs *after* the interchange, `cf_answer` is what the
    counterfactual prompt answers on its own. They coincide only when the
    intervention replaces every variable the answer depends on — as it does
    for a full `result` swap, which is why both are serialized and documented
    rather than aliased."""
    dataset = _weekdays()
    assert all(row["label"] == row["cf_answer"] for row in dataset.rows)

    # Intervening on `number` alone: the answer recomputes from the base
    # entity and the counterfactual number, so it is neither prompt's answer.
    partial = serialize_counterfactual_dataset(
        "natural_domains_arithmetic",
        n=8,
        seed=0,
        split="all",
        task_cfg=NaturalDomainConfig(domain_type="weekdays"),
        target_variables=["number"],
    )
    assert any(
        row["label"] not in (row["base_answer"], row["cf_answer"])
        for row in partial.rows
    )


def test_same_parameters_are_byte_identical():
    """What the content digest in a canonical form (§7) rests on."""
    assert table_bytes(_weekdays().rows) == table_bytes(_weekdays().rows)
    assert table_bytes(_weekdays(seed=1).rows) != table_bytes(_weekdays(seed=0).rows)


def test_written_table_resolves_through_the_seam(tmp_path):
    """The other half of the contract: what the builder writes is exactly what
    ``FileDatasets`` reads back, digest included."""
    dataset = _weekdays()
    out = tmp_path / "weekdays" / "train.json"
    digest = write_dataset_table(dataset.rows, out)
    resolver = FileDatasets(root=tmp_path)
    assert resolver.digest("weekdays/train") == digest
    assert resolver.rows("weekdays/train") == json.loads(out.read_text())
    assert "label_forms" in resolver.columns("weekdays/train")
    # the table is the whole of what is written: no sidecar beside it (§2.2)
    assert [p.name for p in out.parent.iterdir()] == ["train.json"]


def test_missing_generator_names_the_alternatives():
    with pytest.raises(ValueError, match="generate_dataset"):
        serialize_counterfactual_dataset(
            "natural_domains_arithmetic",
            n=1,
            seed=0,
            split="all",
            task_cfg=NaturalDomainConfig(domain_type="weekdays"),
            target_variables=["result"],
            generator="generate_nonsense",
        )


def test_undeclared_answer_variable_refuses():
    with pytest.raises(ValueError, match="declares no scoring forms"):
        serialize_counterfactual_dataset(
            "natural_domains_arithmetic",
            n=1,
            seed=0,
            split="all",
            task_cfg=NaturalDomainConfig(domain_type="weekdays"),
            target_variables=["result"],
            answer_variable="entity",
        )


def test_reserved_columns_cover_the_row_vocabulary():
    """A causal-model variable named like a row column would overwrite it, so
    the guard's list has to stay in step with what ``_row`` writes."""
    dataset = _weekdays()
    variables = set(dataset.rows[0]["counterfactual_inputs_variables"][0])
    assert not variables & RESERVED_COLUMNS
    assert {"input", "label", "cf_answer"} <= RESERVED_COLUMNS


def test_config_class_follows_the_task_convention():
    assert config_class("natural_domains_arithmetic") is NaturalDomainConfig
    assert config_class("MCQA") is None  # a singleton task takes no config


def test_a_generated_table_carries_no_model_dependency():
    """The seam's load-time property, asserted where it is created: a table is
    text and variable strings — no token ids, no tokenizer, no device."""
    dataset = _weekdays()
    for row in dataset.rows:
        for value in row.values():
            assert isinstance(value, (str, list)), value


# --------------------------------------------------------------------------- #
#  serialize_examples: the seam step 3 walks into                              #
# --------------------------------------------------------------------------- #


def test_a_hand_authored_model_serializes_to_the_same_rows():
    """The gap this closes.

    `serialize_counterfactual_dataset` resolves its task through `load_task`,
    which needs a task **package** inside the library checkout. The causal
    protocol's step 3 tells an author to write `models.py` and
    `counterfactuals.py` in their own working directory, and then had nowhere
    to go: no public path from a causal model in hand to a serialized table.

    The assertion that makes the extraction meaningful is that the two entry
    points agree — the package path is now literally this function with a
    `load_task` in front of it, so the rows must be identical.
    """
    from causalab.tasks.loader import load_task, load_task_counterfactuals
    from causalab.tasks.serialize import serialize_examples

    cfg = NaturalDomainConfig(domain_type="weekdays")
    task = load_task("natural_domains_arithmetic", task_cfg=cfg)
    generators = load_task_counterfactuals("natural_domains_arithmetic")
    examples = generators.generate_dataset(task.causal_model, 4, 0)

    inline = serialize_examples(
        task.causal_model,
        examples,
        split="all",
        target_variables=["result"],
        task_label="natural_domains_arithmetic",
        generator="generate_dataset",
        n=4,
        seed=0,
    )
    assert inline.rows == _weekdays().rows
    assert table_bytes(inline.rows) == table_bytes(_weekdays().rows)


def test_serialize_examples_records_no_seed_it_cannot_vouch_for():
    """Examples that did not come from a seeded generator get `seed: None` in
    the result rather than a number nothing can reproduce."""
    from causalab.tasks.loader import load_task, load_task_counterfactuals
    from causalab.tasks.serialize import serialize_examples

    cfg = NaturalDomainConfig(domain_type="weekdays")
    task = load_task("natural_domains_arithmetic", task_cfg=cfg)
    generators = load_task_counterfactuals("natural_domains_arithmetic")
    examples = generators.generate_dataset(task.causal_model, 3, 0)

    dataset = serialize_examples(
        task.causal_model, examples, split="all", target_variables=["result"]
    )
    assert dataset.seed is None
    assert dataset.n == 3  # counted from the rows, not asserted by the caller


def test_serialize_examples_requires_the_target_variables():
    """There is no package to read a `TARGET_VARIABLE` from, and the `label`
    column — the answer *after* the interchange — is undefined without it."""
    from causalab.tasks.serialize import serialize_examples

    cfg = NaturalDomainConfig(domain_type="weekdays")
    from causalab.tasks.loader import load_task

    task = load_task("natural_domains_arithmetic", task_cfg=cfg)
    with pytest.raises(ValueError, match="target_variables is required"):
        serialize_examples(task.causal_model, [], split="all", target_variables=[])


# --------------------------------------------------------------------------- #
# the split column (§2.2)
# --------------------------------------------------------------------------- #


def test_every_row_declares_its_split():
    dataset = _weekdays()
    assert all(row["split"] == "all" for row in dataset.rows)
    assert dataset.split_counts == {"all": 4}


def test_split_is_required_with_no_default():
    """A table that forgot to say which split it is, is the failure the column
    exists to prevent — so there is nothing to forget."""
    with pytest.raises(TypeError, match="split"):
        serialize_counterfactual_dataset(
            "natural_domains_arithmetic",
            n=1,
            seed=0,
            task_cfg=NaturalDomainConfig(domain_type="weekdays"),
            target_variables=["result"],
        )


def test_a_per_row_split_partitions_one_table():
    """What a group-disjoint builder passes: the allocation decided per row,
    serialized into one table rather than split across files."""
    from causalab.tasks.loader import load_task, load_task_counterfactuals
    from causalab.tasks.serialize import serialize_examples

    cfg = NaturalDomainConfig(domain_type="weekdays")
    task = load_task("natural_domains_arithmetic", task_cfg=cfg)
    generators = load_task_counterfactuals("natural_domains_arithmetic")
    examples = generators.generate_dataset(task.causal_model, 4, 0)

    dataset = serialize_examples(
        task.causal_model,
        examples,
        split=["train", "train", "test", "val"],
        target_variables=["result"],
    )
    assert [row["split"] for row in dataset.rows] == ["train", "train", "test", "val"]
    assert dataset.split_counts == {"test": 1, "train": 2, "val": 1}


def test_a_short_split_sequence_refuses():
    """Zipping short would write rows that are individually correct and a
    partition that is a lie."""
    from causalab.tasks.loader import load_task, load_task_counterfactuals
    from causalab.tasks.serialize import serialize_examples

    cfg = NaturalDomainConfig(domain_type="weekdays")
    task = load_task("natural_domains_arithmetic", task_cfg=cfg)
    generators = load_task_counterfactuals("natural_domains_arithmetic")
    examples = generators.generate_dataset(task.causal_model, 4, 0)

    with pytest.raises(ValueError, match="2 values for 4 rows"):
        serialize_examples(
            task.causal_model,
            examples,
            split=["train", "test"],
            target_variables=["result"],
        )


def test_a_variable_named_split_collides():
    """`split` is reserved: a causal-model variable of that name would
    overwrite the row's own declaration."""
    from causalab.tasks.serialize import RESERVED_COLUMNS

    assert "split" in RESERVED_COLUMNS


def test_the_result_records_the_allocation():
    dataset = _weekdays()
    assert dataset.split_counts == {"all": 4}
