"""Multi-token target readouts over prepare_sequence rows agree across both
execution engines — each encodes the row's text the same way."""

import pytest
import torch

from causalab.analysis.sequences import pair_sequences, prepare_sequence
from causalab.neural.engines.nnsight_tracing.executor import TracePointExecutor
from causalab.protocol.schema import parse_document

from tests._helpers.engines import raising_loader
from tests.analysis.test_sequence_analysis import document, executor


pytestmark = pytest.mark.smoke


@pytest.mark.parametrize("patch", [False, True])
def test_sequence_targets_match_hooks(hooks_llama, trace_llama, patch):
    # continuation IDs must round-trip through the tokenizer (their decoded
    # text encodes back to the same IDs): whole word pieces of the tiny llama
    # vocabulary — " over", " the", " lazy" — do; byte-fallback IDs do not
    rows = [
        pair_sequences(
            prepare_sequence(
                hooks_llama.tokenizer,
                prompt,
                [975, 278, 17366],
                example_id=f"base-{i}",
                split="eval",
            ),
            prepare_sequence(
                hooks_llama.tokenizer,
                "the green turtle",
                [278, 975, 17366],
                example_id=f"donor-{i}",
                split="eval",
            ),
        )
        for i, prompt in enumerate(["the red fox", "a very small bird"])
    ]
    raw = document(hooks_llama, patch=patch)
    roles, fields = {"base": rows}, {"base": "input"}
    if patch:
        roles["counterfactual"] = rows
        fields["counterfactual"] = "counterfactual_inputs[0]"
    hooks = executor(hooks_llama, raw, rows)
    trace = TracePointExecutor(
        parse_document(raw),
        trace_llama,
        role_rows=roles,
        role_fields=fields,
        load_tensors=raising_loader,
    )
    for target in range(3):
        torch.testing.assert_close(
            trace.dense_value(f"sequence_{target}"),
            hooks.dense_value(f"sequence_{target}"),
            atol=1e-5,
            rtol=0,
        )
