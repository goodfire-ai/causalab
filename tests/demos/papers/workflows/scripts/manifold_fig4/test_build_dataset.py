"""``build_dataset.py``: the pool and the paper code's 16 base prompts."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.demos.papers._scripts import PAPERS, load_script
from tests.demos.papers.workflows.scripts.manifold_fig4._fixtures import DAYS, ORACLES

pytestmark = pytest.mark.numerical_unit

build = load_script("manifold_fig4", "build_dataset")
DATA = PAPERS / "artifacts" / "data" / "manifold_fig4"


def test_the_builder_draws_the_papers_prompts() -> None:
    """``steer_prompts`` is the first 16 inputs of the paper's
    ``generate_dataset(model, 100, 142)``, duplicates kept, and ``data`` is
    every (day, number) prompt once, in enumeration order."""
    tables = build.build()
    data = tables["data"].rows
    words = ["one", "two", "three", "four", "five", "six", "seven"]
    assert [(r["entity"], r["number"]) for r in data] == [
        (d, w) for d in DAYS for w in words
    ]
    for row in data:
        shift = words.index(row["number"]) + 1
        assert row["result"] == DAYS[(DAYS.index(row["entity"]) + shift) % 7]
        assert (
            row["input"]
            == f"Q: What day is {row['number']} days after {row['entity']}?\nA:"
        )
    steer = tables["steer_prompts"].rows
    assert [[r["entity"], r["number"]] for r in steer] == ORACLES["paper_prompt_draw"]


def test_out_names_a_new_directory(tmp_path: Path) -> None:
    """``--out`` without a ``.json`` suffix is the directory of both tables,
    created when it does not exist, and ``--check`` then passes on it; a
    ``.json`` path names one table."""
    fresh = tmp_path / "fresh"
    assert build.main(["--out", str(fresh)]) == 0
    for name in ("data", "steer_prompts"):
        assert (fresh / f"{name}.json").read_bytes() == (
            DATA / f"{name}.json"
        ).read_bytes()
    assert build.main(["--out", str(fresh), "--check"]) == 0
    assert build.main(["--out", str(fresh / "steer_prompts.json"), "--check"]) == 0
    assert build.main(["--out", str(fresh / "other.json"), "--check"]) == 1
