"""The ``edit_groups`` column at load (spec §2.2, §5 item 27), and the
legitimate campaign (T9) — torch-free.

**T9.** A document with an ordinary base/counterfactual pair and no pairs
declaration loads, canonicalizes to an unchanged digest, and runs: no fixture
table under ``tests/protocol/fixtures/data/`` carries the column; every corpus
document over them passes ``validate --data`` and keeps its pinned digest; both
shipped workflows still load; the serializer writes the column
**only** for an example that declares groups, and rebuilding the committed
``prepared/weekdays_n4`` table reproduces its bytes; ``build_task_dataset.py
--validate-pairs`` writes the table alone (nothing beside it) and writes
nothing for a table whose pairs do not validate. The engine half of T9 — the four-row
weekdays table running the interchange document on tiny gpt2 with no refusal
— is ``tests/neural/engines/pytorch_hooks/test_edit_groups_run.py``.

**The load-time triple** (pattern: ``test_scoring_identity.py``): a
well-formed declaration loads under ``check_data_columns``; a span outside
the text, a lopsided pair of sides or an ``atomic`` group of one is refused
as rule 27 at ``data.base`` naming the row; rows without the column are not
read.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from causalab.causal.pair_validation import EDIT_GROUPS_COLUMN
from causalab.protocol.rules.errors import ValidationError
from causalab.protocol.pipeline import compile_protocol
from causalab.protocol.rules.data import check_data_columns
from causalab.protocol.schema import PROTOCOL_VERSION
from causalab.io.env import FileArtifacts, FileDatasets, ResolutionEnv
from causalab.tasks.natural_domains_arithmetic.config import NaturalDomainConfig
from causalab.tasks.serialize import (
    RESERVED_COLUMNS,
    serialize_counterfactual_dataset,
    serialize_examples,
    table_bytes,
    write_dataset_table,
)
from causalab.tasks.loader import load_task, load_task_counterfactuals
from causalab.workflow.document import load_workflow

from tests._helpers.tiny import TINY_RANDOM_GPT2_MODEL_NAME
from tests.protocol._docs import UNWRITTEN, aggregation, saved
from tests.protocol._env import CORPUS_DIR, FIXTURES, steps_of
from tests._helpers.paths import WORKFLOWS_DIR

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[2]
BUILD_SCRIPT = REPO / "scripts" / "build_task_dataset.py"
WORKFLOW_DIR = WORKFLOWS_DIR
CORPUS_PINS = json.loads((Path(__file__).parent / "corpus_digests.json").read_text())

REF = "pairs/swap"


def _document(ref: str = REF) -> dict[str, Any]:
    """An interchange at a ``variable`` position over one table for both sides."""
    return {
        "header": {
            "protocol_version": PROTOCOL_VERSION,
            "description": "interchange at the first mapping entry",
        },
        "model": {"key": "Qwen/Qwen3-8B", "revision": "main"},
        "data": {
            "base": {"dataset": ref, "field": "input"},
            "counterfactual": {"dataset": ref, "field": "counterfactual_inputs[0]"},
        },
        "method": {
            "intervened_models": {
                UNWRITTEN: {"input": "counterfactual", "reads": ["v_cf"]},
                "patched": {"input": "base", "reads": ["logits"], "writes": ["patch"]},
            },
            "positions": {"first": {"variable": "first"}},
            "sites": {
                "target": {"component": "block_output", "layers": [1]},
                "lm_head": {"component": "lm_head"},
            },
            "reads": {
                "v_cf": {"site": "target", "pos": "first"},
                "logits": {"site": "lm_head", "pos": -1},
            },
            "writes": {
                "patch": {"site": "target", "pos": "first", "do": {"swap": "v_cf"}}
            },
            "save": [
                saved(
                    "logits",
                    "patched",
                    "iia.json",
                    aggregation("match", expected="cf_answer"),
                )
            ],
        },
    }


def _span(text: str, needle: str) -> list[int]:
    start = text.index(needle)
    return [start, start + len(needle)]


def swap_rows(*, atomic: bool = True, declare: bool = True) -> list[dict[str, Any]]:
    """Two two-mapping-entry swaps, each declaring its swap as one group."""
    rows: list[dict[str, Any]] = []
    for a, b, x, y in (("a", "b", "1", "2"), ("x", "y", "3", "4")):
        base = f"{a} maps to {x}, {b} maps to {y}. {a} maps to"
        cf = f"{a} maps to {y}, {b} maps to {x}. {a} maps to"
        row: dict[str, Any] = {
            "input": base,
            "counterfactual_inputs": [cf],
            "base_answer": f" {x}",
            "cf_answer": f" {y}",
            "label": f" {y}",
            "first": x,
            "second": y,
            "counterfactual_inputs_variables": [{"first": y, "second": x}],
            "split": "all",
        }
        if declare:
            row[EDIT_GROUPS_COLUMN] = [
                {
                    "name": "mapping_swap",
                    "atomic": atomic,
                    "spans": {
                        "base": [_span(base, x), _span(base, y)],
                        "counterfactual": [_span(cf, y), _span(cf, x)],
                    },
                }
            ]
        rows.append(row)
    return rows


def _env(root: Path, rows: list[dict[str, Any]], ref: str = REF) -> ResolutionEnv:
    write_dataset_table(rows, root / f"{ref}.json")
    return ResolutionEnv(
        datasets=FileDatasets(root=root), artifacts=FileArtifacts(root=root)
    )


# --------------------------------------------------------------------------- #
# the load-time triple
# --------------------------------------------------------------------------- #


def test_a_well_formed_declaration_loads_and_validates(tmp_path):
    env = _env(tmp_path, swap_rows())
    loaded = compile_protocol(_document(), env=env)
    assert "cf_answer" in check_data_columns(loaded, env)
    assert EDIT_GROUPS_COLUMN in env.datasets.columns(REF)


def test_the_column_is_not_in_the_canonical_form(tmp_path):
    """The declaration is data: the same document over a declaring table and
    over its undeclared twin has the same canonical form up to the dataset
    content digest — no key of the document changes."""
    with_groups = _env(tmp_path / "with", swap_rows())
    without = _env(tmp_path / "without", swap_rows(declare=False))
    a = compile_protocol(_document(), env=with_groups).canonical
    b = compile_protocol(_document(), env=without).canonical
    assert json.dumps(a, sort_keys=True).count(EDIT_GROUPS_COLUMN) == 0
    assert set(a) == set(b)


@pytest.mark.parametrize(
    ("mutate", "detail"),
    [
        (
            lambda g: g["spans"]["base"].__setitem__(1, [0, 999]),
            "not inside the side's text",
        ),
        (lambda g: g["spans"]["counterfactual"].pop(), "same number"),
        (
            lambda g: (g["spans"]["base"].pop(), g["spans"]["counterfactual"].pop()),
            "atomic with one",
        ),
        (lambda g: g.__setitem__("atomic", "yes"), "true or false"),
    ],
    ids=["span-outside-text", "lopsided-sides", "atomic-of-one", "atomic-not-bool"],
)
def test_a_malformed_declaration_is_refused_as_rule_27_at_data_base(
    tmp_path, mutate, detail
):
    rows = swap_rows()
    mutate(rows[1][EDIT_GROUPS_COLUMN][0])
    env = _env(tmp_path, rows)
    loaded = compile_protocol(
        _document(), env=env
    )  # the bare load is fine: the pass is --data
    with pytest.raises(ValidationError) as err:
        check_data_columns(loaded, env)
    assert err.value.rule == 27 and err.value.rule_id == "segment_declared"
    assert err.value.path == "data.base"
    message = str(err.value)
    assert "[V27]" in message and f"'{REF}' row 1" in message and detail in message
    assert "edit_groups" in message


def test_rows_without_the_column_are_not_read(tmp_path):
    """The fail-closed twin: a table with the column on some rows and not
    others declares groups only where it says so, and a table without it at
    all is untouched."""
    mixed = swap_rows()
    del mixed[0][EDIT_GROUPS_COLUMN]
    env = _env(tmp_path / "mixed", mixed)
    check_data_columns(compile_protocol(_document(), env=env), env)
    plain = _env(tmp_path / "plain", swap_rows(declare=False))
    check_data_columns(compile_protocol(_document(), env=plain), plain)


def test_atomic_false_loads_the_same(tmp_path):
    env = _env(tmp_path, swap_rows(atomic=False))
    check_data_columns(compile_protocol(_document(), env=env), env)


# --------------------------------------------------------------------------- #
# T9 — the legitimate campaign: nothing shipped changes
# --------------------------------------------------------------------------- #


def test_no_fixture_table_carries_the_column():
    root = FIXTURES / "data"
    tables = [t for t in sorted(root.rglob("*.json"))]
    assert len(tables) >= 5
    for table in tables:
        rows = FileDatasets(root=root)._table(  # pyright: ignore[reportPrivateUsage]
            table.relative_to(root).with_suffix("").as_posix()
        )
        assert all(EDIT_GROUPS_COLUMN not in row for row in rows), table


@pytest.mark.parametrize(
    "document", sorted(CORPUS_DIR.glob("*_im.json")), ids=lambda p: p.name
)
def test_every_corpus_document_validates_and_keeps_its_pin(document, env):
    loaded = compile_protocol(document, env=env)
    check_data_columns(loaded, env)
    assert loaded.digests.document == CORPUS_PINS[document.name]["document"]
    assert list(steps_of(loaded, env).digests) == CORPUS_PINS[document.name]["points"]


def test_the_shipped_workflows_load_with_inner_documents(env):
    """A smoke test, not a pin: both shipped workflows load and every protocol
    step compiles to an inner document with a digest. There is no
    whole-workflow pin (§7); the corpus documents' byte-identity is
    `test_every_corpus_document_validates_and_keeps_its_pin` above."""
    shipped = sorted(p.name for p in WORKFLOW_DIR.glob("*.json"))
    assert shipped == ["mean_ablation.json", "pca_basis.json", "weekdays.json"]
    for name in shipped:
        loaded = load_workflow(WORKFLOW_DIR / name, env)
        assert loaded.inner, name
        for inner in loaded.inner.values():
            assert len(inner.compiled.digests.document) == 64


def test_the_serializer_writes_the_column_only_for_a_declaring_example():
    """No shipped generator declares groups, so a shipped build carries no
    column; an example that does — the ``edit_groups`` key on what a generator
    returns — gets it, checked against the two prompts."""
    cfg = NaturalDomainConfig(domain_type="weekdays")
    dataset = serialize_counterfactual_dataset(
        "natural_domains_arithmetic", n=3, seed=0, split="all", task_cfg=cfg
    )
    assert all(EDIT_GROUPS_COLUMN not in row for row in dataset.rows)
    assert EDIT_GROUPS_COLUMN in RESERVED_COLUMNS

    task = load_task("natural_domains_arithmetic", task_cfg=cfg)
    examples = load_task_counterfactuals("natural_domains_arithmetic").generate_dataset(
        task.causal_model, 3, 0
    )
    declared: list[dict[str, Any]] = []
    for i, example in enumerate(examples):
        base = example["input"]["raw_input"]
        cf = example["counterfactual_inputs"][0]["raw_input"]
        groups = None
        if i != 1:  # the middle example declares nothing
            groups = [
                {
                    "name": "entity",
                    "atomic": False,
                    "spans": {
                        "base": [_span(base, str(example["input"]["entity"]))],
                        "counterfactual": [
                            _span(
                                cf, str(example["counterfactual_inputs"][0]["entity"])
                            )
                        ],
                    },
                }
            ]
        declared.append({**example, EDIT_GROUPS_COLUMN: groups})
    built = serialize_examples(
        task.causal_model, declared, split="all", target_variables=["result"]
    )
    assert [EDIT_GROUPS_COLUMN in row for row in built.rows] == [True, False, True]
    assert built.rows[0][EDIT_GROUPS_COLUMN][0]["name"] == "entity"
    # the rest of the row is byte-for-byte the undeclared build
    strip = lambda row: {k: v for k, v in row.items() if k != EDIT_GROUPS_COLUMN}  # noqa: E731
    assert [strip(row) for row in built.rows] == dataset.rows

    bad = [{**declared[0]}]
    bad[0][EDIT_GROUPS_COLUMN] = [
        {
            "name": "x",
            "atomic": True,
            "spans": {"base": [[0, 1]], "counterfactual": [[0, 1]]},
        }
    ]
    with pytest.raises(ValueError, match="malformed edit_groups"):
        serialize_examples(
            task.causal_model, bad, split="all", target_variables=["result"]
        )


def test_rebuilding_the_committed_prepared_table_reproduces_its_bytes():
    """The committed ``prepared/weekdays_n4`` (the same four weekday pairs as
    ``weekdays/task_n4_s0`` under a second ref) rebuilt from the command line
    that built it is byte-identical — the conditional column moved nothing.
    The recipe lives here, beside the check, not in a file beside the table."""
    table = FIXTURES / "data" / "prepared" / "weekdays_n4.json"
    dataset = serialize_counterfactual_dataset(
        "natural_domains_arithmetic",
        n=4,
        seed=0,
        split="all",
        task_cfg=NaturalDomainConfig(domain_type="weekdays"),
        target_variables=["result"],
        record_scoring=False,
    )
    assert table_bytes(dataset.rows) == table.read_bytes()
    assert not any(EDIT_GROUPS_COLUMN in row for row in dataset.rows)
    assert [p.name for p in table.parent.iterdir()] == ["weekdays_n4.json"]


# --------------------------------------------------------------------------- #
# --validate-pairs: the three row-and-tokenizer checks at build time
# --------------------------------------------------------------------------- #


def _build(out: Path, *extra: str, seed: int) -> subprocess.CompletedProcess[str]:
    argv = [
        sys.executable,
        str(BUILD_SCRIPT),
        *("--task", "natural_domains_arithmetic", "--set", "domain_type=weekdays"),
        *("--n", "3", "--seed", str(seed), "--split", "all"),
        *("--target-variable", "result", "--out", str(out)),
        *extra,
    ]
    return subprocess.run(argv, capture_output=True, text=True, cwd=REPO)


#: three weekday pairs whose answers all differ (seed 7); at seed 0 the first
#: pair's two entities land on the same weekday — a legitimate generator draw,
#: and exactly the pair an opted-in validation refuses
DISTINCT_SEED = 7
COINCIDING_SEED = 0


def test_validate_pairs_writes_the_table_alone(tmp_path):
    out = tmp_path / "weekdays" / "validated.json"
    result = _build(
        out,
        "--validate-pairs",
        "--tokenizer",
        TINY_RANDOM_GPT2_MODEL_NAME,
        "--revision",
        "main",
        seed=DISTINCT_SEED,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "validated 3 pairs" in result.stdout
    assert f"{TINY_RANDOM_GPT2_MODEL_NAME}@main" in result.stdout
    # the table is the whole of what is written: nothing beside it
    assert [p.name for p in out.parent.iterdir()] == ["validated.json"]
    # and the table itself is what a plain build writes: no column, no new byte
    plain = _build(tmp_path / "weekdays" / "plain.json", seed=DISTINCT_SEED)
    assert plain.returncode == 0, plain.stdout + plain.stderr
    assert out.read_bytes() == (tmp_path / "weekdays" / "plain.json").read_bytes()
    assert not any(EDIT_GROUPS_COLUMN in row for row in json.loads(out.read_text()))


def test_validate_pairs_needs_a_tokenizer(tmp_path):
    result = _build(tmp_path / "x.json", "--validate-pairs", seed=DISTINCT_SEED)
    assert result.returncode != 0 and "--tokenizer" in result.stderr
    assert not (tmp_path / "x.json").exists()


def test_validate_pairs_refuses_a_coinciding_pair_and_writes_nothing(tmp_path):
    """Requirement 1 at build time, fail-closed: a draw whose two entities give
    the same weekday is a pair that changes nothing the task grades — a
    legitimate row for an unvalidated table, and exactly what an opted-in
    validation refuses, writing neither table nor sidecar."""
    out = tmp_path / "weekdays" / "coinciding.json"
    result = _build(
        out,
        "--validate-pairs",
        "--tokenizer",
        TINY_RANDOM_GPT2_MODEL_NAME,
        seed=COINCIDING_SEED,
    )
    assert result.returncode == 1
    assert "refused: --validate-pairs" in result.stderr
    assert "row 0: answer change" in result.stderr
    assert not out.exists()
    # the twin: the same draw builds without validation, as it always did
    plain = _build(out, seed=COINCIDING_SEED)
    assert plain.returncode == 0, plain.stdout + plain.stderr
    assert out.exists()
