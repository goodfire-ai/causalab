"""A read only a device-scored metric consumes stays on its device, and the
scorer that meets it there is ``compute_metric`` to the bit
(``metrics.DEVICE_SCORED_KINDS``, ``executor.base.device_scored_reads``).

``match`` joins the gathered kinds on the device: its argmax is a selection
over an exact, monotone fp32 upcast, and both ``torch.argmax`` kernels return
the first maximal index on a tie — so ``matched_metric`` reproduces
``compute_metric`` entry for entry over random bf16 logits *dense with ties*,
excluded rows and list-valued answer groups included. The CUDA half of the
tie claim is measured on the accelerator (the workflow parity check); the CPU
half is pinned here with the kernel the whole-vocabulary path runs.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from causalab.neural.shared.executor import device_scored_reads
from causalab.neural.shared.metrics import (
    DEVICE_SCORED_KINDS,
    GATHERED_KINDS,
    compute_metric,
    gathered_metric,
    matched_metric,
    score_metric,
)
from causalab.protocol.results import Unavailable
from causalab.protocol.schema import METRIC_KINDS, ReadRef, parse_document
from tests.protocol._docs import (
    LOGIT_DIFF,
    UNWRITTEN,
    aggregation,
    base_doc,
    in_order,
    saved,
)


pytestmark = pytest.mark.property


class _Vocabulary:
    """What ``token_form: id`` needs of a tokenizer: its size."""

    def __init__(self, size: int) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size


def _match() -> Any:
    return SimpleNamespace(kind="match", fields={"expected": "exp"}, token_form="id")


@st.composite
def _match_cases(draw: Any) -> tuple[torch.Tensor, list[dict[str, Any]], int]:
    rows = draw(st.integers(1, 6))
    vocab = draw(st.integers(2, 12))
    # a handful of distinct values only, so equal maxima — the case whose
    # tie rule the device path must share — are the norm, not the exception
    values = draw(
        st.lists(
            st.lists(
                st.sampled_from([-2.0, 0.0, 1.5, 3.0]), min_size=vocab, max_size=vocab
            ),
            min_size=rows,
            max_size=rows,
        )
    )
    logits = torch.tensor(values).to(torch.bfloat16).unsqueeze(1)  # (rows, 1, vocab)
    table: list[dict[str, Any]] = []
    for _ in range(rows):
        expected: Any = draw(
            st.one_of(
                st.integers(0, vocab - 1),
                st.lists(st.integers(0, vocab - 1), min_size=1, max_size=3),
            )
        )
        if draw(st.booleans()) and draw(st.booleans()):
            expected = draw(st.sampled_from([None, []]))  # an excluded row
        table.append({"exp": expected})
    return logits, table, vocab


def _same(got: list[Any], want: list[Any]) -> None:
    assert len(got) == len(want)
    for a, b in zip(got, want, strict=True):
        if isinstance(b, Unavailable):
            assert isinstance(a, Unavailable)
            assert a.denominator_key == b.denominator_key
        else:
            assert isinstance(a, float) and a == b


@given(case=_match_cases())
@settings(max_examples=300, deadline=None)
def test_matched_metric_is_compute_metric_entry_for_entry(case) -> None:
    logits, rows, vocab = case
    metric, tokenizer = _match(), _Vocabulary(vocab)
    want = compute_metric(metric, logits, rows, tokenizer)
    _same(matched_metric(metric, logits, rows, tokenizer), want)
    # the dispatcher on a CPU value is compute_metric itself
    _same(score_metric(metric, logits, rows, tokenizer), want)


def test_the_cpu_argmax_credits_the_first_maximal_index() -> None:
    # the tie rule matched_metric relies on, pinned on the kernel the
    # whole-vocabulary path runs: equal maxima resolve to the lowest index
    logits = torch.tensor([[[1.0, 3.0, 3.0, 0.0]], [[2.0, 2.0, 2.0, 2.0]]])
    rows = [{"exp": [2]}, {"exp": 0}]
    assert compute_metric(_match(), logits, rows, _Vocabulary(4)) == [0.0, 1.0]
    assert matched_metric(_match(), logits, rows, _Vocabulary(4)) == [0.0, 1.0]


def test_the_device_scored_kinds_are_the_gathered_kinds_and_match() -> None:
    assert DEVICE_SCORED_KINDS == GATHERED_KINDS | {"match"}
    assert DEVICE_SCORED_KINDS <= set(METRIC_KINDS)
    # a softmax or a log-sum-exp over the vocabulary rounds differently on
    # the device; those kinds keep the whole-vocabulary CPU path
    assert not DEVICE_SCORED_KINDS & {
        "cross_entropy",
        "kl",
        "js",
        "top_k",
        "class_probs",
    }
    with pytest.raises(ValueError):
        matched_metric(
            SimpleNamespace(kind="logit_diff", fields={}, token_form="id"),
            torch.zeros(1, 1, 4),
            [{}],
            _Vocabulary(4),
        )


def test_score_metric_on_a_cpu_value_is_compute_metric_for_a_gathered_kind() -> None:
    metric = SimpleNamespace(
        kind="logit_diff", fields={"a": "a", "b": "b"}, token_form="id"
    )
    logits = torch.arange(12, dtype=torch.float32).reshape(3, 1, 4)
    rows = [{"a": 1, "b": 2}, {"a": None, "b": 2}, {"a": 3, "b": 0}]
    want = compute_metric(metric, logits, rows, _Vocabulary(4))
    _same(score_metric(metric, logits, rows, _Vocabulary(4)), want)
    _same(gathered_metric(metric, logits, rows, _Vocabulary(4)), want)


# --------------------------------------------------------------------------- #
# which reads a point may leave on the device
# --------------------------------------------------------------------------- #


def _doc(**edits: Any):
    raw = base_doc()
    method = raw["method"]
    for section, value in edits.items():
        method[section] = value
    return parse_document(in_order(raw))


def test_a_read_only_a_gathered_metric_consumes_stays_on_the_device() -> None:
    doc = _doc()  # `logits` under one logit_diff; `v_cf` is the swap's operand
    # bound to the model the read is taken on, as every executor table is keyed
    assert device_scored_reads(doc) == frozenset({ReadRef("logits", "patched")})


def test_a_read_under_match_stays_too() -> None:
    doc = _doc(
        save=[
            saved(
                "logits",
                "patched",
                "iia.json",
                aggregation("match", expected="cf_answer"),
            )
        ],
    )
    assert device_scored_reads(doc) == frozenset({ReadRef("logits", "patched")})


def test_a_saved_read_is_copied_to_the_host() -> None:
    doc = _doc(
        save=[
            saved("logits", "patched", "ld.json", dict(LOGIT_DIFF)),
            saved("logits", "patched", "logits.safetensors"),
        ]
    )
    assert device_scored_reads(doc) == frozenset()


def test_a_write_operand_is_copied_to_the_host() -> None:
    # give `v_cf` an aggregation of its own: it is still the swap's operand
    doc = _doc(
        save=[
            saved("logits", "patched", "ld.json", dict(LOGIT_DIFF)),
            saved(
                "v_cf",
                UNWRITTEN,
                "peek.json",
                aggregation("token_logit", token="cf_answer"),
            ),
        ],
    )
    assert device_scored_reads(doc) == frozenset({ReadRef("logits", "patched")})


def test_a_read_a_whole_vocabulary_kind_reduces_is_copied_to_the_host() -> None:
    raw = base_doc()
    raw["method"]["reads"]["logits_orig"] = {"site": "lm_head", "pos": -1}
    raw["method"]["intervened_models"]["original_base"] = {
        "input": "base",
        "reads": ["logits_orig"],
    }
    raw["method"]["save"] = [
        saved("logits", "patched", "ld.json", dict(LOGIT_DIFF)),
        saved(
            "logits",
            "patched",
            "kl.json",
            aggregation("kl", target={"read": "logits_orig", "model": "original_base"}),
        ),
    ]
    doc = parse_document(in_order(raw))
    # `logits` feeds a kl as well; `logits_orig` is a kl target: both host copies
    assert device_scored_reads(doc) == frozenset()


class _CountingTokenizer:
    """``encode`` counts its calls; one id per character."""

    def __init__(self) -> None:
        self.calls = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        self.calls += 1
        return [ord(c) for c in text]


def test_answer_ids_are_encoded_once_per_tokenizer_and_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A metric resolves the same answer strings row after row and point after
    point: the tokenizer runs once per text, a hit refreshes its entry, and the
    least recently used entry goes first when the bound is reached."""
    from causalab.protocol import answers

    monkeypatch.setattr(answers, "_ENCODED_IDS_PER_TOKENIZER", 3)
    tokenizer = _CountingTokenizer()
    ids = answers._encoded_ids  # pyright: ignore[reportPrivateUsage]
    assert ids(tokenizer, "ab") == (97, 98)
    assert ids(tokenizer, "ab") == (97, 98)
    assert tokenizer.calls == 1
    ids(tokenizer, "c")
    ids(tokenizer, "d")
    ids(tokenizer, "ab")  # refreshed: `c` is now the least recently used
    ids(tokenizer, "e")  # evicts `c`
    calls = tokenizer.calls
    ids(tokenizer, "ab")
    assert tokenizer.calls == calls
    ids(tokenizer, "c")
    assert tokenizer.calls == calls + 1
    other = _CountingTokenizer()
    assert ids(other, "ab") == (97, 98) and other.calls == 1
