"""The eval pass scores from the device (``train._score``): a fit's eval
executor keeps its reads where they are (``ExecutorBase.device_reads``) when
a selecting kind gathers its answer columns there, with the token ids
resolved once per executor; every other kind reduces on a CPU copy as
before, and an eval of such kinds alone keeps the CPU reads — and the scores
are the whole-vocabulary CPU path's, to the bit."""

# the loop's internals are the subject
# pyright: reportPrivateUsage=false

from __future__ import annotations

from typing import Any

import pytest
import torch

from causalab.neural.engines.pytorch_hooks import train as train_module
from causalab.neural.engines.pytorch_hooks.loading import ModelBundle, load_model
from causalab.protocol import answers as answers_module
from causalab.neural.shared.metrics import compute_metric
from causalab.protocol.engine import RunContext
from causalab.io.env import ResolutionEnv

from tests.neural.engines.pytorch_hooks._drive import executor_for
from tests.neural.engines.pytorch_hooks.conftest import TINY_LLAMA
from tests.neural.engines.pytorch_hooks.test_fit_cohort import _InlineDatasets
from tests.neural.engines.pytorch_hooks.test_train import (
    ANSWERS,
    BASES,
    COUNTERFACTUALS,
    das_doc,
)
from tests.protocol._docs import aggregation, by_label, term

pytestmark = pytest.mark.unit

SPLIT = "inline#eval"
EVAL_ROWS = [
    {
        "input": "the tallest tree in the forest is",
        "counterfactual_inputs": ["the deepest lake in the valley is"],
        "label": " one",
        "other": " two",
    },
    {
        "input": "nine purple kites drift over",
        "counterfactual_inputs": ["two rusty bicycles lean against"],
        "label": " two",
        "other": " three",
    },
    {
        "input": "her grandmother's kitchen always smelled of",
        "counterfactual_inputs": ["his uncle's workshop always sounded like"],
        "label": " three",
        "other": None,  # an excluded row for the selecting kind
    },
]


@pytest.fixture(scope="module")
def bundle() -> ModelBundle:
    return load_model(TINY_LLAMA)


def _doc() -> dict[str, Any]:
    raw = das_doc(epochs=1)
    # the eval restates the objective's cross-entropy under the label its
    # saved table carries, beside a margin nothing saves (§2.11)
    (ce,) = [e for e in raw["method"]["save"] if e.get("file_path") == "ce.json"]
    raw["method"]["train"]["eval"] = {
        "every": {"epochs": 1},
        "split": SPLIT,
        "aggregations": {
            "iia": term(
                "logits",
                "patched",
                aggregation("logit_diff", a="label", b="other"),
            ),
            "ce": term("logits", "patched", ce["aggregation"]),
        },
    }
    return raw


def _request() -> RunContext:
    return RunContext(
        env=ResolutionEnv(
            datasets=_InlineDatasets({SPLIT: EVAL_ROWS}),
            artifacts=None,  # type: ignore[arg-type]
        ),
        output_dir=None,  # type: ignore[arg-type]
    )


def test_the_scorer_reads_from_the_device_and_matches_the_cpu_path(
    bundle: ModelBundle, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = _doc()
    point = executor_for(
        raw,
        bundle,
        base_texts=BASES,
        counterfactual_texts=COUNTERFACTUALS,
        extra_columns={"label": ANSWERS},
    )
    evaluator = train_module._eval_executor(point.doc, point, _request(), SPLIT)
    assert evaluator.device_reads
    resolved: list[str] = []
    column_token_ids = answers_module.column_token_ids

    def counted(tokenizer: Any, values: Any, **kwargs: Any) -> list[int]:
        resolved.append(str(kwargs.get("where")))
        return column_token_ids(tokenizer, values, **kwargs)

    monkeypatch.setattr(answers_module, "column_token_ids", counted)
    train_module._fresh_for_eval(evaluator)
    scores = train_module._score(point.doc, evaluator)
    # the selecting kind's ids were resolved once, over the two rows that
    # carry both answers; the third row is excluded, never scored
    assert set(evaluator.metric_token_ids) == {"iia"}
    assert {k: len(v) for k, v in evaluator.metric_token_ids["iia"].items()} == {
        "a": 2,
        "b": 2,
    }
    first = resolved.count("metric logit_diff.a") + resolved.count(
        "metric logit_diff.b"
    )
    assert first == 2
    # the whole-vocabulary CPU path over the same read, as the loop scored
    # before: the same numbers, to the bit
    rows = evaluator.rows_for_metrics()
    logits = evaluator.dense_value("logits")
    for name in ("iia", "ce"):
        values = compute_metric(
            by_label(point.doc)[name], logits.detach().cpu(), rows, bundle.tokenizer
        )
        numeric = [v for v in values if isinstance(v, float)]
        assert scores[name] == sum(numeric) / len(numeric)
    assert isinstance(logits, torch.Tensor)
    # a second pass resolves nothing more for it (the reference above did)
    since = resolved.count("metric logit_diff.a") + resolved.count(
        "metric logit_diff.b"
    )
    evaluator.reset_reads()
    train_module._fresh_for_eval(evaluator)
    again = train_module._score(point.doc, evaluator)
    assert again == scores
    assert (
        resolved.count("metric logit_diff.a") + resolved.count("metric logit_diff.b")
        == since
    )


def test_a_softmax_only_eval_keeps_its_reads_on_the_host(bundle: ModelBundle) -> None:
    """No metric selects on the device, so nothing is gained by holding the
    vocabulary there: the executor's reads move to the CPU as they are
    finalized, once, and the scorer reduces them as it always did."""
    raw = _doc()
    del raw["method"]["train"]["eval"]["aggregations"]["iia"]
    point = executor_for(
        raw,
        bundle,
        base_texts=BASES,
        counterfactual_texts=COUNTERFACTUALS,
        extra_columns={"label": ANSWERS},
    )
    evaluator = train_module._eval_executor(point.doc, point, _request(), SPLIT)
    assert not evaluator.device_reads
    train_module._fresh_for_eval(evaluator)
    scores = train_module._score(point.doc, evaluator)
    assert evaluator.metric_token_ids == {}
    logits = evaluator.dense_value("logits")
    assert isinstance(logits, torch.Tensor)
    values = compute_metric(
        by_label(point.doc)["ce"],
        logits,
        evaluator.rows_for_metrics(),
        bundle.tokenizer,
    )
    numeric = [v for v in values if isinstance(v, float)]
    assert scores == {"ce": sum(numeric) / len(numeric)}
