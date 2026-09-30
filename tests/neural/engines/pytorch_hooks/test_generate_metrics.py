"""Metrics over the continuation frame (§2.10): per-step reduction, the ids
domain, and the row that addressed nothing.

A continuation read addresses as many positions as its row generated, so a
metric over it reduces **per step** and the table says which step each value
came from. Two consequences this module pins:

* a row that addressed *nothing* — it stopped generating, or never said the
  value a ``variable`` anchor looks for — still produces one row, with a null
  value and ``matched=false``. "The model never said it" and "it said it and
  scored 0" must not look alike after a group-by;
* an ``ids``-domain metric (``decode``) reads the tokens the decode produced
  and never the vocabulary projection, which is what makes a text probe cheap
  (§8) — asserted on the plan in ``tests/protocol/test_generate_plan.py``.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from causalab.neural.shared.metrics import compute_windowed_metric
from causalab.neural.shared.results import MetricTable
from causalab.protocol.estimand import metric_record_identity
from causalab.protocol.rules.errors import ProtocolError
from causalab.protocol.schema import parse_document

from tests.neural.engines.pytorch_hooks._drive import executor_for
from tests.neural.engines.pytorch_hooks.conftest import TINY_LLAMA
from tests.protocol._docs import by_label, saved

pytestmark = pytest.mark.smoke

PROMPTS = ["the quick brown fox jumps", "a slow green turtle sleeps deeply today"]
BUDGET = 6


def _doc(aggregation: dict[str, Any], *, anchor: dict[str, Any]) -> dict[str, Any]:
    """The continuation read on the un-intervened model, reduced by
    ``aggregation`` into the table ``scored.json`` (its label: ``scored``)."""
    return {
        "header": {"protocol_version": "4"},
        "model": {"key": TINY_LLAMA, "revision": "main"},
        "data": {"base": {"dataset": "probe", "field": "input"}},
        "method": {
            "intervened_models": {"original": {"input": "base", "reads": ["cont"]}},
            "positions": {
                "window": {"generated": {"max_new_tokens": BUDGET}, **anchor},
            },
            "sites": {"lm_head": {"component": "lm_head"}},
            "reads": {"cont": {"site": "lm_head", "pos": "window"}},
            "save": [saved("cont", "original", "scored.json", aggregation)],
        },
    }


def _run(bundle, raw: dict[str, Any], *, columns: dict[str, list[Any]] | None = None):
    executor = executor_for(
        raw, bundle, base_texts=PROMPTS, extra_columns=columns or {}
    )
    executor.run_all()
    return executor


def _score(executor, name: str = "scored", read: str = "cont") -> list[list[Any]]:
    metric = by_label(executor.doc)[name]
    ids = executor.generated_ids(read) if str(metric.kind) == "decode" else None
    return compute_windowed_metric(
        metric,
        executor.windowed_value(read),
        executor.rows_for_metrics(),
        executor.bundle.tokenizer,
        generated_ids=ids,
    )


def test_top_k_scores_every_generated_step(llama_bundle):
    """One value per step, and each one is the distribution *after* that
    token — so its argmax is the token the decode emitted next (§2.3's
    off-by-one, seen from the metric side)."""
    executor = _run(
        llama_bundle,
        _doc({"kind": "top_k", "k": 1, "by": "prob"}, anchor={"all": True}),
    )
    values = _score(executor)
    (continuation,) = executor._continuations.values()
    assert [len(row) for row in values] == list(continuation.widths)
    for example, row in enumerate(values):
        assert all(len(value["tokens"]) == 1 for value in row)  # k=1
        # compare ids, not the decoded strings: a decode round-trip is lossy
        top_ids = executor.windowed_value("cont")[example].argmax(dim=-1)
        assert torch.equal(top_ids[:-1], continuation.token_ids[example][1 : len(row)])


def test_decode_returns_the_text_the_model_generated(llama_bundle):
    executor = _run(llama_bundle, _doc({"kind": "decode"}, anchor={"all": True}))
    values = _score(executor)
    (continuation,) = executor._continuations.values()
    for example, row in enumerate(values):
        assert len(row) == 1  # one string per example, not one per step
        assert row[0] == llama_bundle.tokenizer.decode(continuation.real_ids(example))


def test_decode_over_a_window_reads_only_that_window(llama_bundle):
    executor = _run(llama_bundle, _doc({"kind": "decode"}, anchor={"index": -1}))
    values = _score(executor)
    (continuation,) = executor._continuations.values()
    for example, row in enumerate(values):
        last = continuation.real_ids(example)[-1]
        assert row == [llama_bundle.tokenizer.decode([last])]


def test_a_variable_the_model_said_scores_where_it_said_it(llama_bundle):
    """Run once to learn what this model says, then ask for it by name: the
    anchor must land on the steps that produced that text."""
    seen = _run(llama_bundle, _doc({"kind": "decode"}, anchor={"all": True}))
    (continuation,) = seen._continuations.values()
    texts = [
        llama_bundle.tokenizer.decode(continuation.real_ids(row))
        for row in range(len(PROMPTS))
    ]
    said = [text[len(text) // 3 : len(text) // 3 + 4] or text[:2] for text in texts]

    executor = _run(
        llama_bundle,
        _doc({"kind": "decode"}, anchor={"variable": "said"}),
        columns={"said": said},
    )
    values = _score(executor)
    for example, row in enumerate(values):
        assert row, f"row {example} addressed no steps for {said[example]!r}"
        assert said[example] in row[0]


def test_a_variable_the_model_never_said_is_null_and_unmatched(llama_bundle):
    """The one place a loud refusal would be wrong: whether the model says
    the thing is the experiment, so it has to come back as data."""
    executor = _run(
        llama_bundle,
        _doc({"kind": "decode"}, anchor={"variable": "said"}),
        columns={"said": ["definitely-not-generated-xyzzy"] * len(PROMPTS)},
    )
    values = _score(executor)
    assert values == [[], []]

    table = MetricTable()
    table.add_windowed(
        "scored",
        values,
        {},
        identity=metric_record_identity("match"),
        steps=None,
        matched=[bool(steps) for steps in executor.addressed_steps("cont")],
    )
    assert len(table.rows) == len(PROMPTS)  # the example survives as a row
    assert all(row["value"] is None for row in table.rows)
    assert all(row["matched"] is False for row in table.rows)
    assert all(row["step"] is None for row in table.rows)


def test_rows_name_the_step_they_scored(llama_bundle):
    executor = _run(
        llama_bundle,
        _doc({"kind": "top_k", "k": 1, "by": "prob"}, anchor={"all": True}),
    )
    steps = executor.addressed_steps("cont")
    table = MetricTable()
    table.add_windowed(
        "scored",
        _score(executor),
        {},
        identity=metric_record_identity("match"),
        steps=steps,
        matched=[bool(row) for row in steps],
    )
    assert [row["step"] for row in table.rows] == [s for row in steps for s in row]
    assert all(row["matched"] is True for row in table.rows)
    assert all(row["value"] is not None for row in table.rows)


def test_a_prompt_frame_read_keeps_its_single_row_shape(llama_bundle):
    """The windowed path is for the continuation; a prompt-frame metric must
    not grow a step column just because this landed."""
    raw = _doc({"kind": "top_k", "k": 1, "by": "prob"}, anchor={"all": True})
    raw["method"]["positions"] = {"window": {"index": -1}}
    raw["method"]["reads"]["cont"]["pos"] = "window"
    executor = _run(llama_bundle, raw)
    assert executor.is_generated("cont") is False
    table = MetricTable()
    table.add("scored", [1.0, 2.0], {}, identity=metric_record_identity("top_k"))
    assert all("step" not in row for row in table.rows)


def test_kl_across_different_widths_refuses(llama_bundle):
    """A comparison needs a position-for-position pairing; two continuations
    that stopped at different places have none."""
    executor = _run(
        llama_bundle,
        _doc({"kind": "top_k", "k": 1, "by": "prob"}, anchor={"all": True}),
    )
    metric = by_label(
        parse_document(
            {
                "header": {"protocol_version": "4"},
                "model": {"key": TINY_LLAMA},
                "data": {"base": {"dataset": "probe", "field": "input"}},
                "method": {
                    "intervened_models": {
                        "original": {"input": "base", "reads": ["a", "b"]}
                    },
                    "sites": {"lm_head": {"component": "lm_head"}},
                    "reads": {
                        "a": {"site": "lm_head", "pos": -1},
                        "b": {"site": "lm_head", "pos": -1},
                    },
                    "save": [
                        saved(
                            "a",
                            "original",
                            "d.json",
                            {
                                "kind": "kl",
                                "target": {"read": "b", "model": "original"},
                            },
                        )
                    ],
                },
            }
        )
    )["d"]
    windows = executor.windowed_value("cont")
    with pytest.raises(ProtocolError, match="different position counts"):
        compute_windowed_metric(
            metric,
            windows,
            executor.rows_for_metrics(),
            executor.bundle.tokenizer,
            target_windows=[windows[0][:1], windows[1]],
        )
