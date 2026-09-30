"""Read routing vs the raw-hook oracle — the collect contract re-driven
through intervention specifications (assertions from
tests/neural/activations/test_collect_hook_oracle.py, verbatim tolerances:
stack-vs-oracle atol=1e-5 rtol=1e-4)."""

from __future__ import annotations

import pytest
import torch

from causalab.neural.shared.encoding import encode

from tests.neural.engines.pytorch_hooks import hook_oracle_lib as oracle_lib
from tests.neural.engines.pytorch_hooks._drive import base_data_section, executor_for
from tests.neural.engines.pytorch_hooks.conftest import BASE_TEXT, OracleShim
from tests.protocol._docs import saved

pytestmark = pytest.mark.unit

TOL = dict(atol=1e-5, rtol=1e-4)


def _inputs(bundle):
    batch = encode(bundle.tokenizer, [BASE_TEXT])
    return {"input_ids": batch.input_ids, "attention_mask": batch.attention_mask}


def original_doc(
    sites: dict[str, dict], reads: dict[str, dict], **method: dict
) -> dict:
    """A harvest: every read taken on the un-intervened network on base
    (``original``, §2.9) and saved as a tensor under its own name."""
    return {
        "header": {"protocol_version": "4"},
        "model": {"key": "test", "revision": "main"},
        "data": base_data_section(with_counterfactual=False),
        "method": {
            "intervened_models": {
                "original": {"input": "base", "reads": list(reads)},
            },
            **method,
            "sites": sites,
            "reads": reads,
            "save": [saved(name, "original", f"{name}.safetensors") for name in reads],
        },
    }


def harvest_doc() -> dict:
    return original_doc(
        {
            "s_in0": {"component": "block_input", "layers": [0]},
            "s_out0": {"component": "block_output", "layers": [0]},
            "s_out1": {"component": "block_output", "layers": [1]},
        },
        {
            "r_in0": {"site": "s_in0", "pos": {"index": 0}},
            "r_out0": {"site": "s_out0", "pos": {"index": 1}},
            "r_out1": {"site": "s_out1", "pos": {"index": 2}},
        },
    )


def test_reads_match_oracle_captures(bundle, oracle: OracleShim):
    executor = executor_for(harvest_doc(), bundle, base_texts=[BASE_TEXT])
    inputs = _inputs(bundle)

    block0 = oracle_lib.decoder_block(oracle, 0)
    want_in0 = oracle_lib.capture_component(oracle, block0, "in", inputs)[:, 0, :]
    want_out0 = oracle_lib.capture_residual(oracle, 0, inputs)[:, 1, :]
    want_out1 = oracle_lib.capture_residual(oracle, 1, inputs)[:, 2, :]

    # anti-vacuity: the three captures genuinely differ (oracle contract)
    assert not torch.allclose(want_in0, want_out0, atol=1e-4)
    assert not torch.allclose(want_out0, want_out1, atol=1e-4)

    torch.testing.assert_close(executor.read_value("r_in0")[:, 0, :], want_in0, **TOL)
    torch.testing.assert_close(executor.read_value("r_out0")[:, 0, :], want_out0, **TOL)
    torch.testing.assert_close(executor.read_value("r_out1")[:, 0, :], want_out1, **TOL)


@pytest.mark.parametrize(
    "component",
    ["embeddings", "attention_output", "mlp_input", "mlp_output", "mlp_activation"],
)
def test_every_component_matches_oracle(bundle, oracle: OracleShim, component: str):
    site: dict = {"component": component}
    if component != "embeddings":
        site["layers"] = 0
    doc = original_doc({"tap": site}, {"r": {"site": "tap", "pos": {"index": 1}}})
    executor = executor_for(doc, bundle, base_texts=[BASE_TEXT])
    module, kind = oracle_lib.component_module(oracle, 0, component)
    want = oracle_lib.capture_component(oracle, module, kind, _inputs(bundle))[:, 1, :]
    torch.testing.assert_close(executor.read_value("r")[:, 0, :], want, **TOL)


def test_head_value_read_matches_oracle(llama_bundle):
    """attention_premix head H == the oracle's o_proj-input column slice
    (test_collect_hook_oracle.py's head case)."""
    doc = original_doc(
        {"h": {"component": "attention_premix", "layers": [0], "head": 1}},
        {"r": {"site": "h", "pos": {"index": 1}}},
    )
    shim = OracleShim(hf_model=llama_bundle.model)
    executor = executor_for(doc, llama_bundle, base_texts=[BASE_TEXT])
    batch = encode(llama_bundle.tokenizer, [BASE_TEXT])
    want = oracle_lib.capture_head_value(
        shim,
        0,
        1,
        {"input_ids": batch.input_ids, "attention_mask": batch.attention_mask},
    )[:, 1, :]
    torch.testing.assert_close(executor.read_value("r")[:, 0, :], want, **TOL)


def test_lm_head_read_is_the_model_logits(bundle, oracle: OracleShim):
    doc = original_doc(
        {"lm_head": {"component": "lm_head"}},
        {"logits": {"site": "lm_head", "pos": {"index": -1}}},
    )
    executor = executor_for(doc, bundle, base_texts=[BASE_TEXT])
    want = oracle_lib.next_token_logits(oracle, _inputs(bundle))
    torch.testing.assert_close(executor.read_value("logits")[:, 0, :], want, **TOL)


def test_variable_position_reads_the_substring_tokens(llama_bundle):
    """A {"variable": x} position addresses exactly the tokens of the row's
    value for x — decoded back, the gathered ids spell the value."""
    text = "the quick brown fox jumps"
    doc = original_doc(
        {"emb": {"component": "embeddings"}},
        {"r": {"site": "emb", "pos": "v"}},
        positions={"v": {"variable": "animal"}},
    )
    executor = executor_for(
        doc, llama_bundle, base_texts=[text], extra_columns={"animal": ["brown fox"]}
    )
    batch = encode(llama_bundle.tokenizer, [text])
    from causalab.neural.shared.encoding import resolve_position
    from causalab.protocol.schema import PositionSpec

    positions = resolve_position(
        PositionSpec(variable="animal"),
        batch,
        0,
        dataset_row={"input": text, "animal": "brown fox"},
        field="input",
    )
    decoded = llama_bundle.tokenizer.decode(batch.input_ids[0, positions])
    assert "brown fox" in decoded
    value = executor.read_value("r")
    embeddings = llama_bundle.model.model.embed_tokens(batch.input_ids)
    torch.testing.assert_close(value[:, :, :], embeddings[:, positions, :], **TOL)


# --------------------------------------------------------------------------- #
#  all positions — the (layer × position) read grid                            #
# --------------------------------------------------------------------------- #


def logit_lens_grid_doc() -> dict:
    """The grid the all spelling exists for: the residual stream at every
    layer (the sweep axis) × every token (the all spec), alongside the
    model's own logits at every token."""
    return original_doc(
        {
            "resid": {"component": "block_output", "layers": {"sweep": [0, 1]}},
            "lm_head": {"component": "lm_head"},
        },
        {
            "r_resid": {"site": "resid", "pos": "all"},  # bare-string sugar
            "r_logits": {"site": "lm_head", "pos": {"all": True}},  # the anchor
        },
    )


def test_all_positions_grid_matches_oracle(bundle, oracle: OracleShim):
    """Every point of the swept grid loads, validates, executes, and the
    read is the oracle's full-sequence capture — not just its last column."""
    from causalab.neural.shared.sweep import expand

    from tests.protocol._docs import in_order

    expansion = expand(in_order(logit_lens_grid_doc()))
    assert len(expansion.points) == 2  # one per layer

    inputs = _inputs(bundle)
    seq = int(inputs["input_ids"].shape[1])
    for point in expansion.points:
        layer = point.raw["method"]["sites"]["resid"]["layers"]
        executor = executor_for(point.raw, bundle, base_texts=[BASE_TEXT])
        have = executor.read_value("r_resid")
        want = oracle_lib.capture_residual(oracle, layer, inputs)
        assert have.shape[1] == seq  # every position, not the last one
        torch.testing.assert_close(have, want, **TOL)

    # anti-vacuity: the two layers genuinely differ
    l0 = oracle_lib.capture_residual(oracle, 0, inputs)
    l1 = oracle_lib.capture_residual(oracle, 1, inputs)
    assert not torch.allclose(l0, l1, atol=1e-4)


def test_all_positions_lm_head_matches_the_models_logits(bundle):
    """The other half of a logit lens: the whole logit sequence, not the
    single next-token column every other read takes."""
    from causalab.neural.shared.sweep import expand

    from tests.protocol._docs import in_order

    point = expand(in_order(logit_lens_grid_doc())).points[0]
    executor = executor_for(point.raw, bundle, base_texts=[BASE_TEXT])
    have = executor.read_value("r_logits")
    with torch.no_grad():
        want = bundle.model(**_inputs(bundle)).logits
    torch.testing.assert_close(have, want, **TOL)


def test_all_positions_is_ragged_across_rows(llama_bundle):
    """Rows of different lengths make an all read ragged; the flat gather
    and its widths are the row-wise captures, padding excluded."""
    from causalab.neural.engines.pytorch_hooks.executor import RaggedValue

    texts = ["one two three", "a much longer sentence right here"]
    doc = original_doc(
        {"emb": {"component": "embeddings"}}, {"r": {"site": "emb", "pos": "all"}}
    )
    executor = executor_for(doc, llama_bundle, base_texts=texts)
    value = executor.read_value("r")
    assert isinstance(value, RaggedValue)

    batch = encode(llama_bundle.tokenizer, texts)
    widths = tuple(
        batch.padded_len - batch.content_start(row) for row in range(len(texts))
    )
    assert value.widths == widths
    assert widths[0] != widths[1]  # the ragged case is real, not incidental

    embeddings = llama_bundle.model.model.embed_tokens(batch.input_ids)
    offset = 0
    for row, width in enumerate(widths):
        start = batch.content_start(row)
        torch.testing.assert_close(
            value.flat[offset : offset + width],
            embeddings[row, start : start + width, :],
            **TOL,
        )
        offset += width
