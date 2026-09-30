"""``token_logits`` through ``run_protocol`` (spec §2.10): the table it saves.

The unit tests pin the reduction on hand-built logits; this is the other half
of the seven-step recipe — a kind that validates, plans, executes and lands on
disk with the columns its spec row promises. The document is the corpus
interchange (``02_interchange_im``) retargeted to tiny-random Llama, plus the
new metric beside a ``token_logit`` over the same read, so the saved table can
be checked against a value the run also produced: for every example, the
logit ``token_logits`` saved under the counterfactual answer must be the logit
``token_logit`` saved for that row's ``cf_answer`` column.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from causalab.neural.shared.engine_router import route
from causalab.protocol import run_protocol
from causalab.protocol.pipeline import compile_protocol
from causalab.io.tables import read_table

from tests.neural.engines.pytorch_hooks.conftest import TINY_LLAMA
from tests.protocol._docs import aggregation
from tests.protocol._docs import saved as save_entry
from tests.protocol._env import CORPUS_DIR, build_env


pytestmark = pytest.mark.smoke

#: The fixture table's whole answer space (``cf_answer`` over every split of
#: ``weekdays/data``), each one token on tiny-random Llama. Wider than the
#: split the document reads on purpose: the kind saves the task's answer
#: space, not the answers the consumed rows happen to carry.
ANSWERS = ["Monday", "Friday", "Saturday", "Sunday"]


def _document() -> dict:
    raw = json.loads((CORPUS_DIR / "02_interchange_im.json").read_text())
    raw["model"]["key"] = TINY_LLAMA
    raw["method"]["sites"]["target"]["layers"] = 1
    raw["method"]["save"] = [
        save_entry(
            "logits",
            "patched",
            "answers.json",
            aggregation("token_logits", tokens=ANSWERS),
        ),
        save_entry(
            "logits",
            "patched",
            "cf_logit.json",
            aggregation("token_logit", token="cf_answer"),
        ),
    ]
    return raw


@pytest.fixture(scope="module")
def saved(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[list[dict], list[dict], list[dict]]:
    """``(token_logits table, token_logit table, rows the run consumed)``.

    The rows come from the same resolver the run used, under the same
    ``dataset`` ref the document names (``weekdays/data#train`` — a split
    *inside* one table, §2.2), so ``example`` indexes them the way the run
    did. Reading a fixture file by name here would pin a different table the
    day the corpus document moves.
    """
    root = tmp_path_factory.mktemp("token_logits")
    env = build_env(root / "artifacts")
    raw = _document()
    loaded = compile_protocol(raw, env=env)
    result = run_protocol(loaded, env, route("auto", device="cpu"), root / "out")
    return (
        read_table(Path(result.files["answers.json"])),
        read_table(Path(result.files["cf_logit.json"])),
        env.datasets.rows(raw["data"]["base"]["dataset"]),
    )


def test_the_table_has_one_row_per_example_with_the_metric_columns(saved) -> None:
    answers, _, rows = saved
    assert rows, "the document read no rows — the check below would be vacuous"
    # the table has no `example_id` column, so the label is the row index (§2.2)
    assert [row["example_id"] for row in answers] == [str(i) for i in range(len(rows))]
    assert all(
        set(row)
        == {
            "example_id",
            "metric",
            "value",
            "unit",
            "estimand_version",
            "eligible",  # the eligibility record (§2.10)
        }
        for row in answers
    )
    assert {row["metric"] for row in answers} == {"answers"}


def test_each_value_carries_the_three_lists_in_document_order(saved) -> None:
    answers, _, _ = saved
    for row in answers:
        value = json.loads(row["value"])
        assert set(value) == {"indices", "tokens", "values"}
        assert len(value["indices"]) == len(value["tokens"]) == len(value["values"])
        assert len(value["values"]) == len(ANSWERS)
        assert [token.strip() for token in value["tokens"]] == ANSWERS
        assert all(isinstance(i, int) for i in value["indices"])
        assert all(isinstance(v, float) for v in value["values"])


def test_the_saved_logits_agree_with_token_logit_over_the_same_read(saved) -> None:
    """The cross-check that makes this a run test rather than a shape test."""
    answers, cf_logit, rows = saved
    by_example = {row["example_id"]: row["value"] for row in cf_logit}
    assert set(by_example) == {row["example_id"] for row in answers}
    for row in answers:
        value = json.loads(row["value"])
        slot = ANSWERS.index(rows[int(row["example_id"])]["cf_answer"].strip())
        assert value["values"][slot] == pytest.approx(by_example[row["example_id"]])
