"""The table's ``string_mode`` at load (spec §2.2, §2.10): the
``validate --data`` pass holds a document's ``match`` ``mode`` to the
``string_mode`` the base table records, under the translation table.

Three tables, one document shape:

* a **recorded** table built from a ``prefix`` task
  (``subject_object_relations``) — under ``mode: first_token`` it loads and
  the check says ``ok``; under ``mode: exact`` it is refused pre-forward, as
  rule 4, naming the table's mode, the metric's mode and the derivation;
* a **recorded** table built from an ``exact`` task (weekdays) — ``exact`` is
  ``ok`` and so is ``first_token`` (a strict generalization; the over-crediting
  half is the metric's own refusal, ``test_metrics.py``);
* the **unrecorded** committed fixture table — the fail-closed twin: it loads,
  every corpus document over it validates, and the check says so.

Torch-free: tables are built by the serializer and read by ``FileDatasets``;
nothing here loads a model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from causalab.causal.scoring import (
    STRING_MODE_COLUMN,
    check_scoring,
    declared_modes,
)
from causalab.protocol.rules.errors import ValidationError
from causalab.protocol.pipeline import compile_protocol
from causalab.protocol.rules.data import check_data_columns
from causalab.io.env import FileArtifacts, FileDatasets, ResolutionEnv
from causalab.protocol.schema import PROTOCOL_VERSION
from causalab.tasks.natural_domains_arithmetic.config import NaturalDomainConfig
from causalab.tasks.serialize import (
    serialize_counterfactual_dataset,
    write_dataset_table,
)
from causalab.tasks.subject_object_relations.config import SubjectObjectRelationsConfig

from tests.protocol._docs import aggregation, by_label, saved
from tests.protocol._env import CORPUS_DIR, FIXTURES, steps_of


pytestmark = pytest.mark.unit

MODEL = "Qwen/Qwen3-8B"
PREFIX_REF = "sor/name_gender"
EXACT_REF = "weekdays/recorded"


def _document(
    ref: str, mode: str, expected: str = "base_answer_forms"
) -> dict[str, Any]:
    return {
        "header": {
            "protocol_version": PROTOCOL_VERSION,
            "description": "base accuracy over one table, scored by the declared forms",
        },
        "model": {"key": MODEL, "revision": "main"},
        "data": {"base": {"dataset": ref, "field": "input"}},
        "method": {
            "intervened_models": {"original": {"input": "base", "reads": ["logits"]}},
            "positions": {"answer_tok": {"index": -1}},
            "sites": {"lm_head": {"component": "lm_head"}},
            "reads": {"logits": {"site": "lm_head", "pos": "answer_tok"}},
            "save": [
                saved(
                    "logits",
                    "original",
                    "accuracy.json",
                    aggregation(
                        "match",
                        expected=expected,
                        mode=mode,
                    ),
                )
            ],
        },
    }


@pytest.fixture(scope="module")
def recorded(tmp_path_factory) -> ResolutionEnv:
    """Two recorded tables in a scratch data root: a ``prefix`` task's and an
    ``exact`` task's, each carrying its spec's ``string_mode`` in every row."""
    root = tmp_path_factory.mktemp("scoring_data")
    for ref, task, cfg in (
        (
            PREFIX_REF,
            "subject_object_relations",
            SubjectObjectRelationsConfig(relation="name_gender"),
        ),
        (
            EXACT_REF,
            "natural_domains_arithmetic",
            NaturalDomainConfig(domain_type="weekdays"),
        ),
    ):
        dataset = serialize_counterfactual_dataset(
            task, n=6, seed=0, split="all", task_cfg=cfg
        )
        write_dataset_table(dataset.rows, root / f"{ref}.json")
    return ResolutionEnv(
        datasets=FileDatasets(root=root), artifacts=FileArtifacts(root=root)
    )


def test_a_recorded_table_records_one_string_mode(recorded):
    env = recorded
    for ref, expected_mode in ((PREFIX_REF, "prefix"), (EXACT_REF, "exact")):
        rows = env.datasets.rows(ref)
        assert {row[STRING_MODE_COLUMN] for row in rows} == {expected_mode}
        assert STRING_MODE_COLUMN in env.datasets.columns(ref)


def test_a_prefix_table_under_first_token_validates_and_is_ok(recorded):
    env = recorded
    loaded = compile_protocol(_document(PREFIX_REF, "first_token"), env=env)
    assert "base_answer_forms" in check_data_columns(loaded, env)
    (doc,) = steps_of(loaded, env).documents
    check = check_scoring(
        env.datasets.rows(PREFIX_REF),
        declared_modes(by_label(doc)),
        where=PREFIX_REF,
    )
    assert check.as_record() == {"string_mode": "prefix", "result": "ok"}


def test_a_prefix_table_under_exact_is_refused_naming_both_modes_and_the_derivation(
    recorded,
):
    """The contradiction: the document says the answer is one token, the
    table says it is not. Rule 4 — a reference that does not resolve — at the
    metric's ``mode``, before any weights."""
    env = recorded
    loaded = compile_protocol(
        _document(PREFIX_REF, "exact"), env=env
    )  # the bare load is fine
    with pytest.raises(ValidationError) as err:
        check_data_columns(loaded, env)
    message = str(err.value)
    assert "[V4]" in message
    assert "'exact'" in message and "'prefix'" in message and "'first_token'" in message
    assert "prefix → first_token" in message
    assert "save[0].aggregation.mode" in message


def test_an_exact_table_is_ok_under_either_mode(recorded):
    """``first_token`` generalizes ``exact`` on single-token answers, so an
    exact table is not contradicted by it; the multi-token half is refused
    where the ids resolve (``_refuse_indistinct_first_tokens``), not here."""
    env = recorded
    for mode in ("exact", "first_token"):
        loaded = compile_protocol(_document(EXACT_REF, mode), env=env)
        check_data_columns(loaded, env)
        (doc,) = steps_of(loaded, env).documents
        check = check_scoring(
            env.datasets.rows(EXACT_REF),
            declared_modes(by_label(doc)),
            where=EXACT_REF,
        )
        assert check.result == "ok" and check.string_mode == "exact"


def test_a_table_whose_rows_disagree_on_their_string_mode_is_refused(
    recorded, tmp_path
):
    env = recorded
    rows = env.datasets.rows(PREFIX_REF)
    rows[0] = {**rows[0], STRING_MODE_COLUMN: "exact"}
    root = tmp_path / "broken"
    write_dataset_table(rows, root / "sor" / "name_gender.json")
    broken = ResolutionEnv(
        datasets=FileDatasets(root=root), artifacts=FileArtifacts(root=root)
    )
    with pytest.raises(ValidationError) as err:
        check_data_columns(
            compile_protocol(_document(PREFIX_REF, "first_token"), env=broken), broken
        )
    assert "[V4]" in str(err.value) and "rows disagree" in str(err.value)


# --------------------------------------------------------------------------- #
# the unrecorded twin: every committed fixture table carries no column
# --------------------------------------------------------------------------- #


def _corpus_documents() -> list[Path]:
    return sorted(CORPUS_DIR.glob("*_im.json"))


def test_the_committed_fixture_tables_are_unrecorded():
    """The fail-closed twin's premise: no fixture table carries the column —
    and every one of them still loads."""
    root = FIXTURES / "data"
    for table in sorted(root.rglob("*.json")):
        rows = FileDatasets(root=root)._table(
            table.relative_to(root).with_suffix("").as_posix()
        )
        assert all(STRING_MODE_COLUMN not in row for row in rows), table


@pytest.mark.parametrize("document", _corpus_documents(), ids=lambda p: p.name)
def test_every_corpus_document_over_an_unrecorded_table_validates(document, env):
    """T14's load half over the corpus: ``validate --data`` passes on every
    document, and for each ``match`` metric the check reports ``unrecorded``
    — nothing compared, nothing refused. ``env`` is the corpus tests' own
    (``tests/protocol/conftest.py``: the fixture tables plus the generated
    artifact bundles)."""
    loaded = compile_protocol(document, env=env)
    check_data_columns(loaded, env)
    for doc in steps_of(loaded, env).documents:
        modes = declared_modes(by_label(doc))
        if not modes:
            continue
        ref = doc.data["base"].dataset
        assert isinstance(ref, str)
        check = check_scoring(env.datasets.rows(ref), modes, where=ref)
        assert check.result == "unrecorded" and check.string_mode is None
