"""The engine half of multi-token scoring: a genuinely multi-token answer graded without
``first_token`` (T13), and the table's ``string_mode`` at run time — refused
before the first forward, recorded in the run receipt.

**T13.** ``" 85"`` is two content tokens on the tiny gpt2 fixture
(``[" 8", "5"]``, the docstring's own case at ``metrics.py``), so a ``match``
over it cannot be ``exact`` (the value is not one token) and must not be
``first_token`` beside an ``87`` (they share a first token, and the metric
would credit the wrong answer as correct — the half already refused by
``_refuse_indistinct_first_tokens``). The path that needs neither is the
``decode`` kind — an ``ids``-domain metric that reads the tokens the model
produced and obliges no vocabulary projection (§2.10) — plus the task's
``ScoringSpec`` grading the text: ``85`` scores 1, ``87`` scores 0, and the
grade is a metric record in the ``fraction`` unit under its own arithmetic
name (``GRADE_RECORD_IDENTITY``).

**The receipt.** A run over a recorded ``prefix`` table under ``first_token``
writes ``scoring.<ref> = {string_mode, result: ok}``; over the
unrecorded corpus fixture it writes ``result: unrecorded`` and runs exactly as
before; a ``prefix`` table under ``mode: exact`` is refused with rule 4 before
any forward, and nothing but the receipt is written.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
import torch

from causalab.causal.scoring import GRADE_RECORD_IDENTITY, ScoringSpec
from causalab.neural.engines.pytorch_hooks.engine import PytorchHooksEngine
from causalab.neural.engines.pytorch_hooks.loading import load_model
from causalab.neural.shared.execution import SCORING_KEY
from causalab.neural.shared.metrics import (
    compute_metric,
    compute_windowed_metric,
)
from causalab.protocol.answers import (
    column_first_token_id,
    column_token_id,
)
from causalab.neural.shared.results import MetricTable
from causalab.protocol import RUN_RECORD_NAME, run_protocol
from causalab.protocol.rules.errors import ProtocolError, ValidationError
from causalab.protocol.pipeline import compile_protocol
from causalab.io.env import FileArtifacts, FileDatasets, ResolutionEnv
from causalab.protocol.schema import AggregationSpec
from causalab.tasks.serialize import (
    serialize_counterfactual_dataset,
    write_dataset_table,
)
from causalab.tasks.subject_object_relations.config import SubjectObjectRelationsConfig

from tests.neural.engines.pytorch_hooks._drive import executor_for
from tests.neural.engines.pytorch_hooks.conftest import TINY_GPT2, TINY_LLAMA
from tests.protocol._docs import aggregation, by_label, saved
from tests.protocol._env import CORPUS_DIR, FIXTURES, write_rot_fixture, steps_of

pytestmark = pytest.mark.smoke

PROMPTS = ["The answer is", "The total comes to"]
BUDGET = 4


@pytest.fixture(scope="module")
def gpt2_bundle():
    return load_model(TINY_GPT2)


def _decode_document(model_key: str) -> dict[str, Any]:
    return {
        "header": {"protocol_version": "4"},
        "model": {"key": model_key, "revision": "main"},
        "data": {"base": {"dataset": "probe", "field": "input"}},
        "method": {
            "intervened_models": {"original": {"input": "base", "reads": ["cont"]}},
            "positions": {
                "window": {"generated": {"max_new_tokens": BUDGET}, "all": True}
            },
            "sites": {"lm_head": {"component": "lm_head"}},
            "reads": {"cont": {"site": "lm_head", "pos": "window"}},
            "save": [saved("cont", "original", "said.json", aggregation("decode"))],
        },
    }


# --------------------------------------------------------------------------- #
# T13 — the two-token answer
# --------------------------------------------------------------------------- #

TWO_TOKEN = ScoringSpec(forms={"total": {"85": [" 85", "85"]}})


def test_the_witness_is_two_tokens_sharing_a_first_piece(gpt2_bundle):
    """The premise, pinned on the fixture: ``" 85"`` and ``" 87"`` are each
    two content tokens and share their first one."""
    tokenizer = gpt2_bundle.tokenizer
    for value in (" 85", " 87"):
        ids = tokenizer.encode(value, add_special_tokens=False)
        assert len([t for t in ids if tokenizer.decode([t]).strip()]) == 2, (value, ids)
    first = {value: column_first_token_id(tokenizer, value) for value in (" 85", " 87")}
    assert first[" 85"] == first[" 87"]


def test_exact_cannot_score_a_two_token_answer(gpt2_bundle):
    with pytest.raises(ProtocolError, match="not a single token"):
        column_token_id(gpt2_bundle.tokenizer, " 85")


def test_first_token_over_85_and_87_is_refused_not_scored(gpt2_bundle):
    """T13's mutation: the same fixture through ``match`` with
    ``mode: first_token`` must be refused by ``_refuse_indistinct_first_tokens``
    — never silently score 1.000 for an emitted ``87``."""
    tokenizer = gpt2_bundle.tokenizer
    metric = AggregationSpec(
        kind="match",
        fields={"expected": "total", "mode": "first_token"},
    )
    logits = torch.zeros(2, 1, len(tokenizer))
    eight = column_first_token_id(tokenizer, " 85")
    logits[:, 0, eight] = 4.0  # "the model emitted ` 8`…"
    with pytest.raises(ProtocolError) as err:
        compute_metric(metric, logits, [{"total": " 85"}, {"total": " 87"}], tokenizer)
    assert err.value.code == "P2"
    assert "not first-token distinct" in str(err.value)


def test_the_spec_grades_the_two_token_answer_as_text():
    assert TWO_TOKEN.grade(" 85", "85") == 1.0
    assert TWO_TOKEN.grade("85", " 85") == 1.0
    assert TWO_TOKEN.grade("87", "85") == 0.0  # the over-credited case, scored 0
    assert TWO_TOKEN.grade(" 87", "85") == 0.0
    assert TWO_TOKEN.grade(" 850", "85") == 0.0  # exact: not a prefix match


def test_decode_plus_the_spec_yields_a_graded_metric_record(gpt2_bundle):
    """The decode-and-grade path end to end on the tiny fixture: ``decode`` returns the
    text the model produced, the spec grades it, and the grade lands in a
    metric table as a ``fraction`` under ``string_grade/v1`` — no vocabulary
    projection, no ``first_token``.

    A random-weight model says nothing meaningful, so the spec here is built
    from what it *did* say: grading each row's own text against itself is 1.0
    and against the other row's is 0.0 (the rows decode to different text),
    which is the whole arithmetic the path needs.
    """
    executor = executor_for(
        _decode_document(TINY_GPT2), gpt2_bundle, base_texts=PROMPTS
    )
    executor.run_all()
    metric = by_label(executor.doc)["said"]
    decoded = compute_windowed_metric(
        metric,
        executor.windowed_value("cont"),
        executor.rows_for_metrics(),
        gpt2_bundle.tokenizer,
        generated_ids=executor.generated_ids("cont"),
    )
    texts = [row[0] for row in decoded]
    assert len(texts) == len(PROMPTS) and all(isinstance(t, str) for t in texts)
    assert texts[0] != texts[1], (
        "the fixture decoded identical text; pick other prompts"
    )

    spec = ScoringSpec(forms={"said": {t: [t] for t in texts}}, string_mode="exact")
    grades = [spec.grade(text, expected) for text, expected in zip(texts, texts)]
    assert grades == [1.0, 1.0]
    assert spec.grade(texts[0], texts[1]) == 0.0

    table = MetricTable()
    table.add_windowed(
        "graded",
        [[g] for g in grades],
        {},
        identity=GRADE_RECORD_IDENTITY,
        steps=None,
        matched=[bool(steps) for steps in executor.addressed_steps("cont")],
    )
    assert [row["value"] for row in table.rows] == [1.0, 1.0]
    assert {row["unit"] for row in table.rows} == {"fraction"}
    assert {row["estimand_version"] for row in table.rows} == {"string_grade/v1"}


# --------------------------------------------------------------------------- #
# the receipt: ok / unrecorded / refused before the first forward
# --------------------------------------------------------------------------- #

SOR_REF = "sor/name_gender"


def _sor_document(mode: str) -> dict[str, Any]:
    return {
        "header": {"protocol_version": "4"},
        "model": {"key": TINY_LLAMA, "revision": "main"},
        "data": {"base": {"dataset": SOR_REF, "field": "input"}},
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
                        expected="base_answer_forms",
                        mode=mode,
                    ),
                )
            ],
        },
    }


@pytest.fixture(scope="module")
def recorded_env(tmp_path_factory) -> ResolutionEnv:
    root = tmp_path_factory.mktemp("recorded")
    dataset = serialize_counterfactual_dataset(
        "subject_object_relations",
        n=4,
        seed=0,
        split="all",
        task_cfg=SubjectObjectRelationsConfig(relation="name_gender"),
    )
    write_dataset_table(dataset.rows, root / f"{SOR_REF}.json")
    return ResolutionEnv(
        datasets=FileDatasets(root=root), artifacts=FileArtifacts(root=root)
    )


def _receipt(out: Path) -> dict[str, Any]:
    return json.loads((out / RUN_RECORD_NAME).read_text())


def test_a_recorded_prefix_table_under_first_token_runs_and_the_receipt_says_ok(
    recorded_env, tmp_path
):
    env = recorded_env
    result = run_protocol(
        compile_protocol(_sor_document("first_token"), env=env),
        env,
        PytorchHooksEngine(),
        tmp_path,
        record=True,
    )
    assert "accuracy.json" in result.files
    assert _receipt(tmp_path)[SCORING_KEY] == {
        SOR_REF: {"string_mode": "prefix", "result": "ok"}
    }


def test_a_recorded_prefix_table_under_exact_is_refused_before_any_forward(
    recorded_env, tmp_path
):
    env = recorded_env
    with pytest.raises(ValidationError) as err:
        run_protocol(
            compile_protocol(_sor_document("exact"), env=env),
            env,
            PytorchHooksEngine(),
            tmp_path,
            record=True,
        )
    message = str(err.value)
    assert "[V4]" in message and "'exact'" in message and "'prefix'" in message
    assert "'first_token'" in message and "prefix → first_token" in message
    # refused at the door, before the engine: nothing on disk — no receipt (the
    # engine writes it, and a refused document never reaches one), no table
    assert not (tmp_path / RUN_RECORD_NAME).exists()
    assert not (tmp_path / "accuracy.json").exists()


def test_the_unrecorded_corpus_fixture_runs_unchanged_and_the_receipt_says_so(tmp_path):
    """The fail-closed twin at run time: the committed table carries no
    string_mode column, so nothing is compared, the numbers are whatever they were, and
    the receipt records ``unrecorded`` for the base ref."""
    artifacts = tmp_path / "artifacts"
    shutil.copytree(FIXTURES / "artifacts", artifacts, dirs_exist_ok=True)
    write_rot_fixture(artifacts)
    env = ResolutionEnv(
        datasets=FileDatasets(root=FIXTURES / "data"),
        artifacts=FileArtifacts(root=artifacts),
    )
    loaded = compile_protocol(
        CORPUS_DIR / "02_interchange_im.json",
        env=env,
        overrides={"model.key": TINY_LLAMA, "sites.target.layers": 1},
    )
    out = tmp_path / "run"
    result = run_protocol(loaded, env, PytorchHooksEngine(), out, record=True)
    assert "iia.json" in result.files
    block = _receipt(out)[SCORING_KEY]
    (doc,) = steps_of(loaded, env).documents
    assert set(block) == {doc.data["base"].dataset}
    assert list(block.values()) == [{"string_mode": None, "result": "unrecorded"}]
