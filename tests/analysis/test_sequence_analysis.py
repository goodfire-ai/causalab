"""Shared token analysis against actual tiny-model prefix forwards, offline."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from causalab.analysis.logit_lens import logit_lens
from causalab.analysis.sequences import (
    add_readouts,
    add_rollout_readouts,
    harvest_protocol,
    pair_sequences,
    patching_protocol,
    prepare_sequence,
    sequence_cohorts,
    write_sequence_workflow,
)
from causalab.neural.engines.pytorch_hooks.executor import PointExecutor
from causalab.neural.engines.pytorch_hooks.loading import ModelBundle
from causalab.neural.engines.pytorch_hooks.train import metric_tensor
from causalab.neural.shared.metrics import compute_metric
from causalab.protocol.rules.errors import ProtocolError
from causalab.protocol.schema import PROTOCOL_VERSION, parse_document
from causalab.protocol.rules.document import validate_document

from tests.protocol._docs import by_label, in_order, saved

pytestmark = pytest.mark.smoke


def _bundle(decoder_width: int | None = None):
    """A ten-token tokenizer over a tiny Llama; ``decoder_width`` widens the
    head past the tokenizer, as a padded vocabulary does."""
    vocabulary = {
        t: i
        for i, t in enumerate(
            ["[PAD]", "[UNK]", "[EOS]", "a", "b", "c", "d", "e", "f", "g"]
        )
    }
    backend = Tokenizer(models.WordLevel(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        eos_token="[EOS]",
        padding_side="left",
    )
    torch.manual_seed(73)
    model = (
        LlamaForCausalLM(
            LlamaConfig(
                vocab_size=decoder_width or len(tokenizer),
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=3,
                num_attention_heads=4,
                num_key_value_heads=2,
                max_position_embeddings=128,
            )
        )
        .eval()
        .requires_grad_(False)
    )
    model.set_attn_implementation("eager")
    return ModelBundle.from_model(
        model,
        tokenizer,
        key="test/sequence-llama",
        revision="local",
        dtype="fp32",
    )


@pytest.fixture(scope="module")
def bundle():
    return _bundle()


def specification(bundle) -> dict[str, Any]:
    """A bare current-protocol skeleton: header, model, data, empty method."""
    return {
        "header": {"protocol_version": PROTOCOL_VERSION, "description": "test"},
        "model": {"key": bundle.key, "revision": bundle.revision},
        "data": {"base": {"dataset": "inline", "field": "input"}},
        "method": {},
    }


def document(bundle, *, patch=False):
    raw = specification(bundle)
    if patch:
        raw["method"].update(
            sites={"write_site": {"component": "block_output", "layers": [1]}},
            reads={"donor": {"site": "write_site", "pos": -3}},
            writes={"swap": {"site": "write_site", "pos": -3, "do": {"swap": "donor"}}},
            intervened_models={
                # the donor is read un-intervened on the counterfactual; the
                # readouts land on `patched`, so it takes no read here
                "original_counterfactual": {
                    "input": "counterfactual",
                    "reads": ["donor"],
                },
                "patched": {"input": "base", "writes": ["swap"]},
            },
        )
        raw["data"]["counterfactual"] = {
            "dataset": "inline",
            "field": "counterfactual_inputs[0]",
        }
    return add_readouts(raw, 3, model="patched" if patch else "original", top_k=3)


def executor(bundle, raw, rows):
    doc = parse_document(raw)
    validate_document(doc, engine_is_local=True)
    roles = {"base": rows}
    fields = {"base": "input"}
    if "counterfactual" in doc.data:
        roles["counterfactual"] = rows
        fields["counterfactual"] = "counterfactual_inputs[0]"
    return PointExecutor(
        doc, bundle, role_rows=roles, role_fields=fields, load_tensors=lambda path: None
    )


@pytest.mark.parametrize("patch", [False, True])
def test_one_pass_scores_all_targets_like_separate_prefixes(bundle, patch):
    rows = [
        prepare_sequence(
            bundle.tokenizer, p, [6, 7, 8], example_id=str(i), split="eval"
        )
        for i, p in enumerate(["a b c", "a c"])
    ]
    for row in rows:
        donor = prepare_sequence(
            bundle.tokenizer, "b b c", [7, 6, 8], example_id="cf", split="eval"
        )
        row["counterfactual_inputs"] = [donor["input"]]
    run = executor(bundle, document(bundle, patch=patch), rows)
    calls = []
    hook = bundle.model.register_forward_pre_hook(lambda *args: calls.append(1))
    run.run_all()
    hook.remove()
    assert len(calls) == (2 if patch else 1)
    for i, row in enumerate(rows):
        # the oracle encodes the same text the engine does
        full_ids = bundle.tokenizer(row["input"])["input_ids"]
        prompt_length = len(full_ids) - row["output_length"]
        if patch:
            cf = torch.tensor(
                [bundle.tokenizer(row["counterfactual_inputs"][0])["input_ids"]]
            )
            capture = []
            capture_hook = bundle.blocks[1].register_forward_hook(
                lambda m, a, out: capture.append(out.detach())
            )
            bundle.model(cf, use_cache=False)
            capture_hook.remove()
        for j in range(3):
            length = prompt_length + j
            handles = []
            if patch:

                def swap(m, a, out):
                    out = out.clone()
                    if length > prompt_length:
                        out[:, prompt_length] = capture[0][:, -3]
                    return out

                handles.append(bundle.blocks[1].register_forward_hook(swap))
            with torch.no_grad():
                expected = bundle.model(
                    torch.tensor([full_ids[:length]]), use_cache=False
                ).logits[0, -1]
            for handle in handles:
                handle.remove()
            torch.testing.assert_close(
                run.dense_value(f"sequence_{j}")[i, 0], expected, atol=1e-6, rtol=1e-5
            )


def test_recorded_rollout_decodes_into_the_text_frame(bundle):
    """Emitted IDs become the targets; their decoding in prompt context is the
    ``input`` the engine encodes. A text completion cannot claim to be a
    recorded rollout."""
    row = prepare_sequence(
        bundle.tokenizer,
        "a b",
        [6, 6, 2],
        example_id="x",
        split="eval",
        prefix_condition="baseline_generated",
    )
    run = executor(bundle, document(bundle), [row])
    observed = []
    handle = bundle.model.register_forward_pre_hook(
        lambda module, args, kwargs: observed.append(kwargs["input_ids"].tolist()),
        with_kwargs=True,
    )
    try:
        run.dense_value("sequence_0")
    finally:
        handle.remove()
    assert row["input"] == "a b d d [EOS]"
    assert observed == [[bundle.tokenizer(row["input"])["input_ids"]]]
    assert [t["token_id"] for t in row["targets"]] == [6, 6, 2]
    assert row["targets"][-1]["is_eos"]
    with pytest.raises(ProtocolError, match="recorded token IDs"):
        prepare_sequence(
            bundle.tokenizer,
            "a",
            " b",
            example_id="x",
            split="eval",
            prefix_condition="baseline_generated",
        )


def test_alternate_eos_rejects_padding_in_a_recorded_rollout(bundle):
    alternate = 9  # "g"
    example = dict(
        prompt="a b",
        completion=[6, alternate],
        example_id="one",
        split="confirmation",
        prefix_condition="baseline_generated",
        eos_token_ids=[alternate],
    )
    row = prepare_sequence(bundle.tokenizer, **example)
    assert row["targets"][-1]["is_eos"] and not row["targets"][0]["is_eos"]
    with pytest.raises(ProtocolError, match="tokens after EOS"):
        prepare_sequence(bundle.tokenizer, **dict(example, completion=[alternate, 6]))


@pytest.mark.parametrize(
    "kind, field",
    [("cross_entropy", "target"), ("token_logit", "token"), ("match", "expected")],
)
def test_exact_id_metrics_include_whitespace_and_special_tokens(bundle, kind, field):
    raw = document(bundle)
    raw["method"]["save"].append(
        saved(
            "sequence_0",
            "original",
            "score.json",
            {"kind": kind, field: "target", "token_form": "id"},
        )
    )
    metric = by_label(parse_document(raw))["score"]
    logits = torch.tensor([[10.0] + [0.0] * 9])
    value = compute_metric(metric, logits, [{"target": 0}], bundle.tokenizer)[0]
    if kind == "cross_entropy":
        assert value == pytest.approx(float(-logits.log_softmax(-1)[0, 0]))
        differentiable = logits.clone().requires_grad_(True)
        loss = metric_tensor(metric, differentiable, [{"target": 0}], bundle.tokenizer)
        loss.sum().backward()
        assert differentiable.grad is not None and differentiable.grad[0, 0] < 0
    else:
        assert value == (1.0 if kind == "match" else 10.0)
    for invalid in ("0", True, -1, 10, 0.5):
        with pytest.raises(ProtocolError, match="integer token ID"):
            compute_metric(metric, logits, [{"target": invalid}], bundle.tokenizer)


def test_logit_lens_matches_final_layer_patching_without_a_forward(bundle):
    row = prepare_sequence(
        bundle.tokenizer, "a b c", [6, 7, 8], example_id="x", split="eval"
    )
    raw = document(bundle)
    method = raw["method"]
    method["sites"].update(
        source={"component": "block_output", "layers": [0]},
        final={"component": "block_output", "layers": [2]},
    )
    method["reads"]["residual"] = {"site": "source", "pos": -2}
    method["intervened_models"]["original"]["reads"].append("residual")
    method["writes"] = {
        "lens_write": {"site": "final", "pos": -2, "do": {"swap": "residual"}}
    }
    method["reads"]["lens_logits"] = {"site": "sequence_head", "pos": -2}
    method["intervened_models"]["lens"] = {
        "input": "base",
        "reads": ["lens_logits"],
        "writes": ["lens_write"],
    }
    method["save"].append(saved("lens_logits", "lens", "lens.safetensors"))
    run = executor(bundle, in_order(raw), [row])
    run.run_all()
    calls = []
    hook = bundle.model.register_forward_pre_hook(lambda *args: calls.append(1))
    result = logit_lens(
        bundle, run.read_value("residual"), k=3, batch_positions=1, target_ids=[8]
    )
    hook.remove()
    expected = run.dense_value("lens_logits")[0, 0]
    assert not calls
    assert result[0]["highest_indices"] == expected.topk(3).indices.tolist()
    assert result[0]["highest_tokens"] == [
        bundle.tokenizer.decode([token]) for token in expected.topk(3).indices.tolist()
    ]
    assert result[0]["highest_logits"] == pytest.approx(
        expected.topk(3).values.tolist(), abs=1e-6
    )
    assert result[0]["highest_probabilities"] == pytest.approx(
        expected.softmax(-1).topk(3).values.tolist(), abs=1e-6
    )
    assert (
        result[0]["lowest_indices"] == expected.topk(3, largest=False).indices.tolist()
    )
    assert result[0]["lowest_tokens"] == [
        bundle.tokenizer.decode([token])
        for token in expected.topk(3, largest=False).indices.tolist()
    ]
    assert result[0]["lowest_logits"] == pytest.approx(
        expected.topk(3, largest=False).values.tolist(), abs=1e-6
    )
    assert result[0]["lowest_probabilities"] == pytest.approx(
        expected.softmax(-1).topk(3, largest=False).values.tolist(), abs=1e-6
    )
    assert result[0]["target_log_probability"] == pytest.approx(
        float(expected.log_softmax(-1)[8]), abs=1e-6
    )


def test_logit_lens_bounds_target_ids_by_the_decoder_not_the_tokenizer():
    padded = _bundle(decoder_width=12)  # two logit columns no token decodes to
    assert len(padded.tokenizer) == 10 and padded.info.vocab_size == 12
    activations = torch.randn(3, padded.model.config.hidden_size)
    records = logit_lens(padded, activations, k=3, target_ids=[0, 9, 11])
    with torch.no_grad():
        logits = padded.model.lm_head(padded.model.model.norm(activations)).float()
    assert [r["target_id"] for r in records] == [0, 9, 11]
    assert records[2]["target_logit"] == pytest.approx(float(logits[2, 11]))
    # tokens added without resizing the head: the tokenizer now outruns the decoder
    padded.tokenizer.add_tokens(["h", "i", "j"])
    assert len(padded.tokenizer) == 13
    for invalid in (12, -1, True, "3", 0.5):
        with pytest.raises(ValueError, match=r"\[0, 12\), the decoder vocabulary"):
            logit_lens(padded, activations, target_ids=[0, 9, invalid])
    with pytest.raises(ValueError, match="one vocabulary ID per activation"):
        logit_lens(padded, activations, target_ids=[0])


def test_campaign_saves_every_target_with_one_shared_donor_pass(bundle, tmp_path):
    from causalab.neural.engines.pytorch_hooks.engine import PytorchHooksEngine
    from causalab.io.env import ResolutionEnv, FileDatasets, FileArtifacts
    from causalab.protocol.pipeline import run_protocol
    from causalab.io.tables import read_table, write_table

    base = prepare_sequence(
        bundle.tokenizer, "a b c", [6, 7, 8], example_id="base", split="eval"
    )
    donor = prepare_sequence(
        bundle.tokenizer, "b b c", [7, 6, 8], example_id="donor", split="eval"
    )
    pairs = [pair_sequences(base, donor)]
    assert len(sequence_cohorts(pairs)) == 1
    write_table(tmp_path / "pairs.json", pairs)
    env = ResolutionEnv(
        datasets=FileDatasets(root=tmp_path), artifacts=FileArtifacts(root=tmp_path)
    )
    model = {"key": bundle.key, "revision": bundle.revision}
    for component in ("block_output", "attention_output", "mlp_output"):
        raw = patching_protocol(
            model,
            "pairs",
            3,
            component=component,
            bands=[[0], [1]],
            positions=[{"index": -4}],
            top_k=3,
        )
        calls = []
        hook = bundle.model.register_forward_pre_hook(lambda *args: calls.append(1))
        try:
            run_protocol(
                raw, env, PytorchHooksEngine(bundle=bundle), tmp_path / component
            )
        finally:
            hook.remove()
        assert len(calls) == 4  # one donor, one null, two distinct interventions
        for band in (0, 1):
            for target in range(3):
                rows = read_table(
                    tmp_path / component / f"scores_{band}_{target}_accuracy.json"
                )
                assert len(rows) == 1
    harvest = harvest_protocol(model, "pairs", [0, 1, 2])
    calls = []
    hook = bundle.model.register_forward_pre_hook(lambda *args: calls.append(1))
    try:
        run_protocol(
            harvest, env, PytorchHooksEngine(bundle=bundle), tmp_path / "harvest"
        )
    finally:
        hook.remove()
    assert len(calls) == 1
    assert (
        len(logit_lens(bundle, tmp_path / "harvest/residual_0.safetensors", k=3)) == 6
    )
    quantized = replace(bundle, quantization={"scheme": "int8"})
    with pytest.raises(ProtocolError, match="quantization"):
        logit_lens(quantized, tmp_path / "harvest/residual_0.safetensors", k=3)


def test_rollout_readouts_share_decode_and_use_prediction_predecessors(bundle):
    raw = add_rollout_readouts(specification(bundle), 3, model="original", top_k=3)
    rows = [{"input": "a b c", "output_0": 6, "output_1": 7, "output_2": 8}]
    run = executor(bundle, raw, rows)
    calls = []
    hook = bundle.model.register_forward_pre_hook(lambda *args: calls.append(1))
    run.run_all()
    hook.remove()
    assert len(calls) == 4  # one prefill and three steps, regardless of target count
    ids = run.generated_ids("rollout_continuation")[0]
    if ids:
        assert int(run.dense_value("rollout_0")[0, 0].argmax()) == ids[0]
    for i in range(1, len(ids)):
        assert int(run.windowed_value(f"rollout_{i}")[0][0].argmax()) == ids[i]


def test_pair_alignment_and_parent_split_leakage_are_explicit(bundle):
    base = prepare_sequence(
        bundle.tokenizer, "a b", [6, 7], example_id="base", split="train"
    )
    donor = prepare_sequence(
        bundle.tokenizer, "b c", [8, 7, 6], example_id="donor", split="train"
    )
    with pytest.raises(ValueError, match="target_alignment"):
        pair_sequences(base, donor)
    with pytest.raises(ValueError, match="target_alignment"):
        pair_sequences(base, donor, target_alignment={False: 2, True: 1})
    pair = pair_sequences(base, donor, target_alignment={0: 2, 1: 1})
    assert (pair["cf_output_0"], pair["cf_output_1"]) == (6, 7)
    heldout = {**donor, "split": "eval"}
    with pytest.raises(ValueError, match="multiple splits"):
        sequence_cohorts([pair, heldout])


def test_written_workflow_runs_and_ragged_harvest_gathers_targets(bundle, tmp_path):
    from causalab.analysis import sequence_activations
    from causalab.io.step_io import read_tensor
    from causalab.neural.engines.pytorch_hooks.engine import PytorchHooksEngine
    from causalab.io.env import ResolutionEnv, FileDatasets, FileArtifacts
    from causalab.workflow.document import load_workflow
    from causalab.workflow.runner import run_workflow

    pairs = []
    for i, prompt in enumerate(["a b c", "a c"]):
        base = prepare_sequence(
            bundle.tokenizer, prompt, [6, 7, 8], example_id=f"base{i}", split="eval"
        )
        donor = prepare_sequence(
            bundle.tokenizer, "b c", [7, 6, 8], example_id=f"donor{i}", split="eval"
        )
        pairs.append(pair_sequences(base, donor))
    path = write_sequence_workflow(
        tmp_path,
        {"key": bundle.key, "revision": bundle.revision},
        pairs,
        layers=[0, 1, 2],
        bands=[[0, 1]],
        positions=[{"index": -4}],
        top_k=3,
    )
    env = ResolutionEnv(
        datasets=FileDatasets(root=tmp_path), artifacts=FileArtifacts(root=tmp_path)
    )
    loaded = load_workflow(path, env)
    run_workflow(loaded, env, tmp_path / "runs", PytorchHooksEngine(bundle=bundle))
    output = tmp_path / "aligned.safetensors"
    sequence_activations.main(
        {
            "acts": tmp_path / "runs/sequence_analysis/harvest/residual_0.safetensors",
            "rows": tmp_path / "pairs.json",
        },
        {"acts": output},
    )
    assert read_tensor(output).shape == (2, 3, 32)


def test_boundary_merge_is_refused(bundle):
    backend = Tokenizer(
        models.BPE(
            {"[UNK]": 0, "a": 1, "b": 2, "ab": 3}, [("a", "b")], unk_token="[UNK]"
        )
    )
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")
    with pytest.raises(ProtocolError, match="merges across"):
        prepare_sequence(tokenizer, "a", "b", example_id="merge", split="eval")


def test_exact_ids_refuse_first_token_semantics(bundle):
    raw = document(bundle)
    (accuracy,) = [
        e for e in raw["method"]["save"] if e["file_path"] == "sequence_0_accuracy.json"
    ]
    accuracy["aggregation"]["mode"] = "first_token"
    with pytest.raises(ProtocolError, match="exact integer token IDs"):
        parse_document(raw)


def test_readouts_refuse_a_v1_document(bundle):
    """A flat top-level ``version`` document is v1; the helper names the migrate verb."""
    flat = {
        "version": "1",
        "model": {"key": bundle.key, "revision": bundle.revision},
        "data": {"base": {"dataset": "inline", "field": "input"}},
    }
    with pytest.raises(ValueError, match="causalab migrate"):
        add_readouts(flat, 3, model="original")
    with pytest.raises(ValueError, match="unknown method sections"):
        add_readouts(
            {**specification(bundle), "method": {"sites": {}, "layer": 1}},
            3,
            model="original",
        )
