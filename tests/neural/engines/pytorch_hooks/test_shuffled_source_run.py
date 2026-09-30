"""The ``shuffled_source`` control as a run (intervention protocol
spec §2.2 ``shuffle: {seed}``; workflow spec §2.2), on the tiny Llama fixture.

Corpus 02's interchange is run twice over the 4-row ``weekdays/train`` fixture
table (``weekdays/data#train`` is two rows here — too small for a permutation
to be visible): once as authored and once with ``data.counterfactual.shuffle:
{seed: 0}``, the one difference. The shuffled pairing is a **different
intervention**: the saved operand read ``v_cf`` carries the counterfactual
rows in the seed's order (row ``i`` is the unshuffled row ``order[i]``), the
receiver's ``logit_diff`` differs row by row, and the receipt's canonical form
shows ``shuffle`` — while the unshuffled run's shows none.

Without the change the shuffled document is refused at parse (``[P3] unknown
key 'shuffle'``). The mutation that makes ``resolve_roles`` ignore ``shuffle``
fails the two order assertions (the saved rows come back in authored order,
the metric agrees with the unshuffled run).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.torch import load_file

from causalab.cli import main
from causalab.protocol.positions.roles import shuffle_order

from .conftest import TINY_LLAMA
from tests.protocol._docs import UNWRITTEN, saved
from tests.protocol._env import CORPUS_DIR, FIXTURES

pytestmark = pytest.mark.smoke

TABLE = "weekdays/train"  # 4 rows in tests/protocol/fixtures/data


def _document(*, seed: int | None) -> dict[str, Any]:
    raw = json.loads((CORPUS_DIR / "02_interchange_im.json").read_text())
    for role in raw["data"].values():
        role["dataset"] = TABLE
    if seed is not None:
        raw["data"]["counterfactual"]["shuffle"] = {"seed": seed}
    raw["method"]["save"].append(saved("v_cf", UNWRITTEN, "v_cf.safetensors"))
    return raw


def _run(raw: dict[str, Any], root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    document = root / "doc.json"
    document.write_text(json.dumps(raw))
    artifacts = root / "artifacts"
    shutil.copytree(FIXTURES / "artifacts", artifacts, dirs_exist_ok=True)
    out = root / "run"
    code = main(
        [
            "run",
            "--engine",
            "auto",
            str(document),
            "--data-root",
            str(FIXTURES / "data"),
            "--artifacts-root",
            str(artifacts),
            "--out",
            str(out),
            "--set",
            f"model.key={TINY_LLAMA}",
            "--set",
            "sites.target.layers=1",
            "--dtype",
            "fp32",
            "--record",
        ]
    )
    assert code == 0
    return out


def _rows(out: Path, name: str) -> list[float]:
    table = json.loads((out / name).read_text())
    rows = table["rows"] if isinstance(table, dict) and "rows" in table else table
    values = [r["value"] for r in rows if isinstance(r, dict) and "value" in r]
    assert values, f"{name}: no per-row values in {str(table)[:200]}"
    return values


def test_the_shuffled_twin_is_a_different_intervention(tmp_path: Path) -> None:
    plain = _run(_document(seed=None), tmp_path / "plain")
    shuffled = _run(_document(seed=0), tmp_path / "shuffled")

    # the receipt's canonical form shows the verb exactly where authored
    plain_canonical = json.loads((plain / "protocol.json").read_text())["canonical"]
    shuffled_canonical = json.loads((shuffled / "protocol.json").read_text())[
        "canonical"
    ]
    assert "shuffle" not in plain_canonical["data"]["counterfactual"]
    assert shuffled_canonical["data"]["counterfactual"]["shuffle"] == {"seed": 0}
    assert plain_canonical["data"]["base"] == shuffled_canonical["data"]["base"]

    # the saved operand read carries the counterfactual rows in the seed's order
    v_plain = next(iter(load_file(str(plain / "v_cf.safetensors")).values()))
    v_shuffled = next(iter(load_file(str(shuffled / "v_cf.safetensors")).values()))
    order = shuffle_order(0, v_plain.shape[0])
    assert v_plain.shape[0] == 4 and order != list(range(4))
    assert torch.allclose(v_shuffled, v_plain[order], atol=1e-5, rtol=1e-4)
    assert not torch.allclose(v_shuffled, v_plain, atol=1e-5, rtol=1e-4)

    # so the receiver sees a different intervention, row by row
    ld_plain = _rows(plain, "logit_diff.json")
    ld_shuffled = _rows(shuffled, "logit_diff.json")
    assert len(ld_plain) == len(ld_shuffled) == 4
    assert ld_plain != ld_shuffled


def test_two_seeds_are_two_runs(tmp_path: Path) -> None:
    """Seed 0 and seed 1 are two pairings (two receipts, two orders); the
    unshuffled document's digest is the corpus twin's own."""
    zero = _run(_document(seed=0), tmp_path / "zero")
    one = _run(_document(seed=1), tmp_path / "one")
    d0 = json.loads((zero / "protocol.json").read_text())
    d1 = json.loads((one / "protocol.json").read_text())
    assert d0["canonical"]["data"]["counterfactual"]["shuffle"] == {"seed": 0}
    assert d1["canonical"]["data"]["counterfactual"]["shuffle"] == {"seed": 1}
    assert d0["document_digest"] != d1["document_digest"]
    v0 = next(iter(load_file(str(zero / "v_cf.safetensors")).values()))
    v1 = next(iter(load_file(str(one / "v_cf.safetensors")).values()))
    assert not torch.allclose(v0, v1, atol=1e-5, rtol=1e-4)
