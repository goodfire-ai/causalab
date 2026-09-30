"""``FileDatasets`` reads and checks a table once per file version and parses
every caller its own rows (``env._checked_table_text``).

A campaign resolves every point's roles against the same refs, so a scan step
re-read the same table from disk once per point per role. The memo is keyed by
the file's path, mtime and size — a table rewritten in the same process is a
miss — and holds the text, not the rows: what a caller gets is its own parse,
so annotating a row, or anything nested in it, never reaches the table the
next point reads.
"""

from __future__ import annotations

# pyright: reportPrivateUsage=false

import hashlib
import json
from pathlib import Path

import pytest

from causalab.io import env
from causalab.protocol.rules.errors import ValidationError
from causalab.io.env import FileDatasets
from causalab.tables import table_bytes

pytestmark = pytest.mark.unit


def _write(root: Path, ref: str, rows: list[dict]) -> Path:
    path = root / f"{ref}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(table_bytes(rows))
    return path


def _row(prompt: str, split: str) -> dict:
    return {"input": prompt, "counterfactual_inputs": [f"cf-{prompt}"], "split": split}


def _misses() -> int:
    return env._checked_table_text.cache_info().misses


def test_the_second_read_of_a_table_reads_nothing(tmp_path: Path) -> None:
    _write(tmp_path, "t/data", [_row("a", "train"), _row("b", "test")])
    datasets = FileDatasets(root=tmp_path)
    before = _misses()
    first = datasets.rows("t/data#train")
    assert _misses() == before + 1
    assert datasets.rows("t/data#train") == first
    assert datasets.rows("t/data#test") == [_row("b", "test")]
    assert datasets.digest("t/data#train") == datasets.digest("t/data#train")
    assert _misses() == before + 1  # one read served every call


def test_the_rows_handed_out_are_fresh(tmp_path: Path) -> None:
    _write(tmp_path, "t/data", [_row("a", "train")])
    datasets = FileDatasets(root=tmp_path)
    handed = datasets.rows("t/data#train")
    handed[0]["annotation"] = "mine"
    assert "annotation" not in datasets.rows("t/data#train")[0]
    assert datasets.rows("t/data#train") is not handed


def test_a_rows_nested_values_are_its_own(tmp_path: Path) -> None:
    """Every call parses its own rows: a consumer that edits a list or dict
    inside a row it was handed edits neither the memo nor another point's."""
    _write(tmp_path, "t/data", [_row("a", "train")])
    datasets = FileDatasets(root=tmp_path)
    handed = datasets.rows("t/data#train")
    handed[0]["counterfactual_inputs"].append("cf-mine")
    again = datasets.rows("t/data#train")
    assert again[0]["counterfactual_inputs"] == ["cf-a"]
    assert again[0]["counterfactual_inputs"] is not handed[0]["counterfactual_inputs"]


def test_a_rewritten_table_is_read_again(tmp_path: Path) -> None:
    path = _write(tmp_path, "t/data", [_row("a", "train")])
    datasets = FileDatasets(root=tmp_path)
    assert [r["input"] for r in datasets.rows("t/data#train")] == ["a"]
    # a different size makes the stamp differ whatever the filesystem's
    # mtime resolution
    _write(tmp_path, "t/data", [_row("a", "train"), _row("longer prompt", "train")])
    assert path.stat().st_size != len(table_bytes([_row("a", "train")]))
    assert [r["input"] for r in datasets.rows("t/data#train")] == ["a", "longer prompt"]


def test_a_refusal_is_raised_on_every_read(tmp_path: Path) -> None:
    path = tmp_path / "t" / "data.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"not": "a list"}))
    datasets = FileDatasets(root=tmp_path)
    for _ in range(2):
        with pytest.raises(ValidationError):
            datasets.rows("t/data")


def test_a_refs_digest_is_serialized_once_per_file_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ref's digest is a function of the file version and the ref, so it
    is serialized once per pair, and a rewritten table is a new version with
    its own digest. A workflow load names a table's digest in every point it
    compiles, which is why the count matters."""
    path = _write(tmp_path, "t/data", [_row("a", "train"), _row("b", "test")])
    datasets = FileDatasets(root=tmp_path)
    calls: list[int] = []
    real = env.table_bytes

    def counting(rows):
        calls.append(len(rows))
        return real(rows)

    monkeypatch.setattr(env, "table_bytes", counting)
    first = datasets.digest("t/data#train")
    assert [datasets.digest("t/data#train") for _ in range(3)] == [first] * 3
    assert datasets.digest("t/data#test") != first
    assert calls == [1, 1]  # one serialization per ref
    # a rewritten table is a new version, and its digest is its own
    _write(tmp_path, "t/data", [_row("a", "train"), _row("longer prompt", "train")])
    assert path.stat().st_size != len(
        table_bytes([_row("a", "train"), _row("b", "test")])
    )
    rewritten = [_row("a", "train"), _row("longer prompt", "train")]
    assert (
        datasets.digest("t/data#train") == hashlib.sha256(real(rewritten)).hexdigest()
    )
    assert datasets.digest("t/data#train") != first


def test_a_digest_refusal_is_raised_on_every_call(tmp_path: Path) -> None:
    _write(tmp_path, "t/data", [_row("a", "train")])
    datasets = FileDatasets(root=tmp_path)
    for _ in range(2):
        with pytest.raises(ValidationError):
            datasets.digest("t/data#test")
