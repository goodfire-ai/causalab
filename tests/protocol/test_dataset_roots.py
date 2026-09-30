"""The shipped task tables are reachable behind any ``--data-root``.

``causalab/tasks/<task>/data/<variant>.json`` is where a task ships the table a
document names, and ``FileDatasets`` searches its ``fallback_roots`` after
``root`` misses — so a private table under ``--data-root`` and a shipped
``<task>/data/<variant>`` load in one document, a private table of the same ref
shadows the shipped one (root first), and a shipped document validates with no
flag at all.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from causalab.cli import main
from causalab.protocol.rules.errors import ValidationError
from causalab.io.env import FileDatasets
from causalab.tables import table_bytes
from causalab.tasks import TASKS_ROOT
from tests._helpers.paths import PROTOCOLS_DIR

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[2]
#: The one shipped document that names a fixture table on purpose. It is the
#: standalone-install smoke document: a tiny random Llama on CPU, whose sentencepiece tokenizer cannot spell the
#: shipped weekdays answers ([P2]) — so, like every tiny-scale smoke run, it
#: reads the 4-row fixture, and the smoke run passes `--data-root` to it. Its model
#: is not in the static registry either (`run` registers it from the HF config).
SMOKE_DOCUMENTS = frozenset({"minimal_cpu.json"})

SHIPPED_DOCUMENTS = [
    path
    for path in sorted(PROTOCOLS_DIR.glob("*.json"))
    if path.name not in SMOKE_DOCUMENTS
]


def test_the_smoke_exemption_names_real_documents() -> None:
    """A retired smoke document must leave this list, not linger as a hole."""
    for name in SMOKE_DOCUMENTS:
        assert (PROTOCOLS_DIR / name).is_file(), name


def test_a_shipped_ref_resolves_behind_an_empty_root(tmp_path: Path) -> None:
    datasets = FileDatasets(root=tmp_path, fallback_roots=(TASKS_ROOT,))
    assert len(datasets.rows("pile/data/sample")) == 3
    assert datasets.digest("pile/data/sample") == FileDatasets(root=TASKS_ROOT).digest(
        "pile/data/sample"
    )


def test_a_private_table_shadows_the_shipped_one(tmp_path: Path) -> None:
    private = tmp_path / "pile" / "data" / "sample.json"
    private.parent.mkdir(parents=True)
    private.write_bytes(table_bytes([{"input": "mine", "split": "all"}]))
    datasets = FileDatasets(root=tmp_path, fallback_roots=(TASKS_ROOT,))
    assert [row["input"] for row in datasets.rows("pile/data/sample")] == ["mine"]


def test_a_miss_names_every_root_searched(tmp_path: Path) -> None:
    datasets = FileDatasets(root=tmp_path, fallback_roots=(TASKS_ROOT,))
    with pytest.raises(ValidationError) as err:
        datasets.rows("nowhere/data/none")
    assert err.value.rule == 4
    assert str(tmp_path) in str(err.value) and str(TASKS_ROOT) in str(err.value)


def test_the_same_root_twice_is_searched_once(tmp_path: Path) -> None:
    datasets = FileDatasets(root=TASKS_ROOT, fallback_roots=(TASKS_ROOT,))
    assert datasets.roots == (TASKS_ROOT,)


@pytest.mark.parametrize("document", SHIPPED_DOCUMENTS, ids=lambda p: p.name)
def test_every_shipped_document_validates_with_no_data_root(
    document: Path, artifacts_root: Path, capsys
) -> None:
    """The point of shipping the tables: `causalab validate <shipped> --data`
    with no `--data-root` — the columns each document names are checked
    against the table its ref resolves to under the default root.

    Two shipped documents are workflow steps that read an artifact a prior step
    writes (`dbm_apply`, `mean_ablation`); standalone they are refused for that
    artifact and nothing else, which is the one refusal accepted here.
    """
    code = main(
        [
            "validate",
            "--engine",
            "auto",
            str(document),
            "--data",
            "--artifacts-root",
            str(artifacts_root),
        ]
    )
    captured = capsys.readouterr()
    out = captured.out + captured.err
    assert code == 0 or ("[V15]" in out and "[V4]" not in out), out


def test_no_shipped_document_names_a_fixture_ref() -> None:
    """The regression this whole change closes: every ref a shipped document
    names must be a table a task ships, never one under tests/."""
    shipped = {
        f"{table.parents[1].name}/data/{table.stem}"
        for table in TASKS_ROOT.glob("*/data/*.json")
    }
    for document in SHIPPED_DOCUMENTS:
        raw = json.loads(document.read_text())
        data = raw.get("data") or raw.get("application", {}).get("data", {})
        for role, spec in data.items():
            ref = spec["dataset"].split("#", 1)[0]
            assert ref in shipped, (
                f"{document.name} {role}: {ref!r} is not a shipped table"
            )
