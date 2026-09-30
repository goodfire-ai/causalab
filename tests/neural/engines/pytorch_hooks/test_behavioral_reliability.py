"""Regression cases from behavioral worker failures, on a tiny real model."""

from __future__ import annotations


import pytest
import torch

from causalab.neural.engines.pytorch_hooks.engine import _write_continuations
from causalab.protocol.engine import StepRecord
from causalab.neural.shared.metrics import compute_windowed_metric
from causalab.protocol.engine import RunContext
from causalab.io.env import FileArtifacts, FileDatasets, ResolutionEnv
from causalab.io.tables import read_table
from causalab.protocol.schema import ReadRef
from tests.neural.engines.pytorch_hooks.conftest import TINY_LLAMA
from tests.neural.engines.pytorch_hooks.test_generate_frame import _doc, _executor
from tests.protocol._docs import aggregation, by_label, in_order, saved

pytestmark = pytest.mark.smoke


def test_multi_eos_stops_and_preserves_terminal_and_padding(llama_bundle, tmp_path):
    tokenizer = llama_bundle.tokenizer
    eos1 = tokenizer.eos_token_id
    eos2 = tokenizer.encode(" Friday", add_special_tokens=False)[0]
    content = tokenizer.encode(" maybe", add_special_tokens=False)[0]
    raw = _doc(TINY_LLAMA)
    executor = _executor(llama_bundle, raw)
    executor.decoding = {"mode": "deterministic", "eos_token_ids": [eos1, eos2]}
    original = llama_bundle.model.forward
    calls = []

    def forced(*args, **kwargs):
        result = original(*args, **kwargs)
        step = len(calls)
        calls.append(step)
        for row, token in enumerate([eos1, content if step == 0 else eos2]):
            result.logits[row, -1, :] = -10000
            result.logits[row, -1, token] = 10000
        return result

    llama_bundle.model.forward = forced
    try:
        executor.run_all()
    finally:
        llama_bundle.model.forward = original
    assert len(calls) == 3  # prefill, content/EOS, final consumed EOS; no budget tail
    assert executor.continuations()[("original", "base")].widths == (0, 1)
    request = RunContext(
        output_dir=tmp_path,
        env=ResolutionEnv(FileDatasets(root=tmp_path), FileArtifacts(root=tmp_path)),
        decoding=executor.decoding,
    )
    rows = read_table(
        _write_continuations(request, [executor], steps=(StepRecord(0, {}, "test"),))
    )
    assert rows[0]["token_ids"] == []
    assert rows[0]["emitted_ids"] == [eos1]
    assert rows[1]["emitted_ids"] == [content, eos2]
    for row, eos in zip(rows, [eos1, eos2]):
        assert row["terminal_eos_id"] == eos
        assert row["stop_reason"] == "eos"
        assert row["truncated"] is False
        assert set(row["padding_ids"]) == {tokenizer.pad_token_id}
        assert row["greedy_token_id"] == row["emitted_ids"][0]


def test_model_eos_configuration_precedes_tokenizer(llama_bundle):
    executor = _executor(llama_bundle, _doc(TINY_LLAMA))
    config = llama_bundle.model.generation_config
    original = config.eos_token_id
    try:
        config.eos_token_id = [8, 9]
        assert executor.eos_token_ids() == (8, 9)
        executor.decoding = {"mode": "deterministic", "eos_token_ids": [10]}
        assert executor.eos_token_ids() == (10,)
    finally:
        config.eos_token_id = original


def test_continuation_projection_is_bounded_and_matches_dense(llama_bundle):
    raw = _doc(TINY_LLAMA)
    raw["method"]["positions"]["cont"]["generated"]["max_new_tokens"] = 33
    raw["method"]["reads"]["reference"] = dict(raw["method"]["reads"]["cont"])
    raw["method"]["intervened_models"]["original"]["reads"].append("reference")
    raw["method"]["save"] = [
        saved("cont", "original", "top.json", aggregation("top_k", k=5, by="prob")),
        saved(
            "cont",
            "original",
            "kl.json",
            aggregation("kl", target={"read": "reference", "model": "original"}),
        ),
        saved("cont", "original", "text.json", aggregation("decode")),
    ]
    # Preserve protocol section order.
    executor = _executor(llama_bundle, in_order(raw))
    executor.run_all()
    # keyed by the read bound to its model (ReadRef), never by the bare name
    cont = ReadRef("cont", "original")
    assert cont in executor._deferred_heads
    calls = []
    head = executor._deferred_heads[cont]
    handle = head.register_forward_pre_hook(
        lambda _m, args: calls.append(args[0].shape[0])
    )
    # the executor scores an aggregation bound to its read: each is looked up
    # by its label (`text` and `kl` are saved under their own names)
    doc = executor.doc

    def bound(name: str):
        return next(agg for agg in doc.aggregations() if agg.label == name)

    try:
        text = executor.generated_metric(bound("text"))
        assert text and not calls
        actual = executor.generated_metric(bound("top"))
        divergence = executor.generated_metric(bound("kl"))
        assert all(abs(v) < 1e-7 for row in divergence for v in row)
        assert calls and max(calls) <= 16
    finally:
        handle.remove()
    expected = compute_windowed_metric(
        by_label(executor.doc)["top"],
        executor.windowed_value("cont"),
        executor.rows_for_metrics(),
        llama_bundle.tokenizer,
    )
    for actual_row, expected_row in zip(actual, expected):
        for a, b in zip(actual_row, expected_row):
            assert a["indices"] == b["indices"]
            assert a["values"] == pytest.approx(b["values"], abs=1e-6)
            assert a["probs"] == pytest.approx(b["probs"], abs=1e-7)


def test_last_slot_eos_and_length_cap_are_distinct(llama_bundle, tmp_path):
    eos = llama_bundle.tokenizer.eos_token_id
    content = 20
    executor = _executor(llama_bundle, _doc(TINY_LLAMA))
    executor.decoding = {"mode": "deterministic", "eos_token_ids": [eos]}
    original = llama_bundle.model.forward
    calls = 0

    def forced(*args, **kwargs):
        nonlocal calls
        out = original(*args, **kwargs)
        out.logits[:, -1, :] = -10000
        out.logits[:, -1, content] = 10000
        # Exact tie: a diagnostic top-k may order the larger ID first,
        # while the actual greedy decision must remain the lowest maximum.
        out.logits[:, -1, 30000] = 10000
        if calls == 5:
            out.logits[0, -1, eos] = 20000
        calls += 1
        return out

    llama_bundle.model.forward = forced
    try:
        executor.run_all()
    finally:
        llama_bundle.model.forward = original
    request = RunContext(
        output_dir=tmp_path,
        env=ResolutionEnv(FileDatasets(root=tmp_path), FileArtifacts(root=tmp_path)),
        decoding=executor.decoding,
    )
    rows = read_table(
        _write_continuations(request, [executor], steps=(StepRecord(0, {}, "test"),))
    )
    assert rows[0]["emitted_ids"] == [content] * 5 + [eos]
    assert rows[0]["stop_reason"] == "eos"
    assert rows[0]["padding_ids"] == []
    assert rows[1]["emitted_ids"] == [content] * 6
    assert rows[1]["stop_reason"] == "length"
    assert rows[1]["terminal_eos_id"] is None
    assert all(row["greedy_token_id"] == content for row in rows)


def test_control_token_prefix_is_not_removed(llama_bundle):
    from causalab.neural.shared.encoding import continuation_frame

    tokenizer = llama_bundle.tokenizer
    ids = [tokenizer.bos_token_id] + tokenizer.encode(
        " Friday", add_special_tokens=False
    )
    frame = continuation_frame(tokenizer, torch.tensor([ids]), (len(ids),))
    assert tokenizer.bos_token in frame.texts[0]
    assert frame.texts[0] == tokenizer.decode(
        ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
