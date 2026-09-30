"""Splits are a column of one table, selected by a ref fragment (§2.2).

Both halves live here because both live in [`FileDatasets`][causalab.io.env.FileDatasets]: parsing and
selection, and the four refusals that make the declaration mean something. They
sit in the resolver rather than in a load-time pass because each is a property
of *one table*, and the resolver is the single place a ref becomes rows — so no
verb can route around them, ``run`` included.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from causalab.protocol.rules.errors import ValidationError
from causalab.io.env import FileDatasets, split_dataset_ref
from causalab.tables import INLINE_REF_PREFIX, inline_ref, inline_rows, table_bytes

pytestmark = pytest.mark.unit


def _write(root: Path, ref: str, rows: list[dict]) -> None:
    path = root / f"{ref}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(table_bytes(rows))


def _row(prompt: str, split: str) -> dict:
    return {"input": prompt, "counterfactual_inputs": [f"cf-{prompt}"], "split": split}


@pytest.fixture
def root(tmp_path: Path) -> Path:
    _write(
        tmp_path,
        "weekdays/data",
        [_row("a", "train"), _row("b", "train"), _row("c", "test")],
    )
    return tmp_path


# --------------------------------------------------------------------------- #
# ref parsing
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "ref, expected",
    [
        ("weekdays/data", ("weekdays/data", None)),
        ("weekdays/data#train", ("weekdays/data", "train")),
        # the *last* '#' wins, so a root with a '#' in a directory name resolves
        ("od#d/data#test", ("od#d/data", "test")),
        # a split value may contain anything but '#'
        ("d#fold0", ("d", "fold0")),
    ],
)
def test_split_dataset_ref_parses(ref, expected):
    assert split_dataset_ref(ref) == expected


def test_empty_fragment_is_a_typo_not_the_whole_table():
    """`weekdays#` reads as a slip of the finger. Silently returning every row
    would hand back train+test under a name that promised one split."""
    with pytest.raises(ValidationError, match="empty '#' fragment"):
        split_dataset_ref("weekdays#")


# --------------------------------------------------------------------------- #
# selection
# --------------------------------------------------------------------------- #


def test_bare_ref_to_a_partitioned_table_refuses(root: Path):
    """[R2] The anti-contamination rule. Reading train+test as one pool is the
    mistake the whole design exists to prevent, so a ref that would do it names
    no split and is refused rather than quietly obliged."""
    with pytest.raises(ValidationError, match=r"carries 2 splits.*names none"):
        FileDatasets(root).rows("weekdays/data")


def test_bare_ref_reads_an_undivided_pool(tmp_path: Path):
    """A table that is one pool says so, and a bare ref is then exactly right."""
    _write(tmp_path, "pool", [_row("a", "all"), _row("b", "all")])
    assert len(FileDatasets(tmp_path).rows("pool")) == 2


def test_fragment_selects_one_split(root: Path):
    rows = FileDatasets(root).rows("weekdays/data#train")
    assert [row["input"] for row in rows] == ["a", "b"]
    assert {row["split"] for row in rows} == {"train"}


def test_splits_of_one_table_are_disjoint_by_construction(root: Path):
    """The property the whole design buys: a row declares exactly one split, so
    two fragments of one table *cannot* share a row. No check enforces this —
    there is nothing to enforce."""
    datasets = FileDatasets(root)
    train = {row["input"] for row in datasets.rows("weekdays/data#train")}
    test = {row["input"] for row in datasets.rows("weekdays/data#test")}
    assert not train & test


def test_columns_come_from_the_selected_rows(root: Path):
    assert "split" in FileDatasets(root).columns("weekdays/data#train")


# --------------------------------------------------------------------------- #
# refusals (R1)
# --------------------------------------------------------------------------- #


def test_unknown_split_names_the_ones_that_exist(root: Path):
    """A mistyped fragment is the likeliest slip, and it is silent otherwise:
    without this, `#tets` would resolve to an empty table and the run would
    report a metric over zero rows."""
    with pytest.raises(
        ValidationError, match=r"no rows in split 'tets'.*'test', 'train'"
    ):
        FileDatasets(root).rows("weekdays/data#tets")


def test_a_table_with_no_split_column_refuses(tmp_path: Path):
    """[R4] What makes the declaration *necessary*. Without this the design is
    opt-in, and the two-file pattern it replaces survives untouched in every
    table that predates it."""
    _write(tmp_path, "old", [{"input": "a", "counterfactual_inputs": ["b"]}])
    with pytest.raises(ValidationError, match="has no 'split' column") as err:
        FileDatasets(tmp_path).rows("old")
    assert err.value.rule == 22 and err.value.rule_id == "split_declaration"
    with pytest.raises(ValidationError, match="has no 'split' column"):
        FileDatasets(tmp_path).rows("old#train")


def test_a_partly_declared_table_names_the_gap(tmp_path: Path):
    """Half a declaration is the worse failure: the rows that do declare would
    make the table look partitioned while the silent ones go wherever."""
    _write(tmp_path, "half", [_row("a", "train"), {"input": "b"}])
    with pytest.raises(ValidationError, match=r"1 of 2 rows do not.*row 1"):
        FileDatasets(tmp_path).rows("half#train")


def test_splits_that_share_a_prompt_refuse(tmp_path: Path):
    """[R3] The leak that matters, and the one the old two-file layout could not
    even express: a prompt that is a training base and a test counterfactual
    reports a training score under a held-out name."""
    _write(
        tmp_path,
        "leaky",
        [
            {"input": "p", "counterfactual_inputs": ["q"], "split": "train"},
            {"input": "r", "counterfactual_inputs": ["p"], "split": "test"},
        ],
    )
    with pytest.raises(ValidationError, match=r"leaks across splits.*'p'.*train.*test"):
        FileDatasets(tmp_path).rows("leaky#test")


def test_the_leak_is_caught_even_reading_the_clean_split(tmp_path: Path):
    """The table is what is malformed, so which split you asked for is
    irrelevant — otherwise the check would depend on the reader's luck."""
    _write(
        tmp_path,
        "leaky",
        [
            {"input": "p", "counterfactual_inputs": ["q"], "split": "train"},
            {"input": "r", "counterfactual_inputs": ["p"], "split": "test"},
        ],
    )
    with pytest.raises(ValidationError, match="leaks across splits"):
        FileDatasets(tmp_path).rows("leaky#train")


def test_one_split_used_twice_is_how_you_spell_train_equals_test(tmp_path: Path):
    """A deliberate no-holdout ablation stays possible, but it has to be said
    out loud in the document rather than smuggled in by two splits that happen
    to coincide — which is why [R3] needs no opt-out flag."""
    _write(tmp_path, "abl", [_row("a", "all"), _row("b", "all")])
    datasets = FileDatasets(tmp_path)
    assert datasets.rows("abl#all") == datasets.rows("abl")


def test_missing_table_still_refuses_by_base_ref(root: Path):
    with pytest.raises(ValidationError, match="'nope' not found"):
        FileDatasets(root).rows("nope#train")


# --------------------------------------------------------------------------- #
# digests
# --------------------------------------------------------------------------- #


def test_two_splits_of_one_table_digest_differently(root: Path):
    """Run identity survives the move to one table: a document that trains on
    #train and evaluates on #test still carries two distinct data digests."""
    datasets = FileDatasets(root)
    assert datasets.digest("weekdays/data#train") != datasets.digest(
        "weekdays/data#test"
    )


def test_digest_covers_only_the_rows_consumed(root: Path):
    """Adding a split to a table must not invalidate a run over the splits
    already in it — the digest is over the selected rows, not the file."""
    before = FileDatasets(root).digest("weekdays/data#train")
    rows = json.loads((root / "weekdays/data.json").read_text())
    _write(root, "weekdays/data", rows + [_row("d", "val")])
    assert FileDatasets(root).digest("weekdays/data#train") == before


def test_whole_table_digest_is_still_the_file_digest(tmp_path: Path):
    """`write_dataset_table` writes exactly `table_bytes`, so re-serializing a
    whole table round-trips — the move from file-bytes to row-bytes leaves
    every existing whole-table digest where it was."""
    _write(tmp_path, "pool", [_row("a", "all"), _row("b", "all")])
    on_disk = (tmp_path / "pool.json").read_bytes()
    assert FileDatasets(tmp_path).digest("pool") == hashlib.sha256(on_disk).hexdigest()


# --------------------------------------------------------------------------- #
# the committed tables
# --------------------------------------------------------------------------- #

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Every data root the repo ships tables under.
DATA_ROOTS = [
    REPO_ROOT / "demos" / "onboarding_tutorial" / "artifacts" / "data",
    REPO_ROOT / "tests" / "protocol" / "fixtures" / "data",
    REPO_ROOT / "tests" / "golden" / "fixtures" / "data",
]


def _committed_tables() -> list[tuple[Path, str]]:
    out = []
    for root in DATA_ROOTS:
        for table in sorted(root.rglob("*.json")):
            out.append((root, table.relative_to(root).with_suffix("").as_posix()))
    return out


@pytest.mark.parametrize(
    "root, ref",
    _committed_tables(),
    ids=[f"{root.parent.name}/{ref}" for root, ref in _committed_tables()],
)
def test_every_committed_table_declares_its_splits(root: Path, ref: str):
    """The ratchet. A table added without a `split` column, or one whose splits
    share a prompt, fails here rather than at whatever run first reads it —
    which is the difference between the guarantee holding for the tables that
    happened to be migrated and holding for the repo."""
    rows = json.loads((root / f"{ref}.json").read_text())
    declared = {row.get("split") for row in rows}
    assert None not in declared, f"{ref}: some row declares no split"
    # Resolving each split exercises the disjointness check on the way through.
    datasets = FileDatasets(root)
    for split in declared:
        assert datasets.rows(f"{ref}#{split}"), f"{ref}#{split} selected no rows"


def test_an_inline_ref_resolves_without_a_file(tmp_path: Path):
    """§2.2: an inline table lives in the process, never under the root; its
    rows, columns and digest follow the file formulas exactly."""
    prompts = ["a", "b"]
    ref = inline_ref(prompts)
    datasets = FileDatasets(root=tmp_path)
    rows = datasets.rows(ref)
    assert rows == inline_rows(prompts)
    assert datasets.columns(ref) == ("input", "split")
    assert datasets.digest(ref) == hashlib.sha256(table_bytes(rows)).hexdigest()
    assert datasets.digest(ref) == ref[len(INLINE_REF_PREFIX) :]


def test_inline_rows_are_fresh_copies(tmp_path: Path):
    ref = inline_ref(["a"])
    datasets = FileDatasets(root=tmp_path)
    datasets.rows(ref)[0]["input"] = "changed"
    assert datasets.rows(ref)[0]["input"] == "a"


def test_an_unregistered_inline_ref_refuses(tmp_path: Path):
    """An inline ref is derived, never authored: one nobody parsed names no
    table, and the refusal says where the rows come from."""
    with pytest.raises(ValidationError) as err:
        FileDatasets(root=tmp_path).rows(INLINE_REF_PREFIX + "0" * 64)
    assert err.value.rule == 4
    assert "'inputs'" in str(err.value)
