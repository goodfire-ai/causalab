"""A minimal LMPipeline-workalike for the kept token-position tests.

The Plan-era ``LMPipeline`` is gone; the task packages' token-position
vocabularies (and their property/numerical tests) only ever used this
surface of it: a tokenizer, plain-text batch loading with optional offset
mappings, and the chat-prefix facts (always zero here — the kept tests are
plain-text). This shim satisfies
[`causalab.tasks.token_positions.EncodingPipeline`][] over the
reference engine's loader, with the same conventions (left padding,
``pad = eos``).
"""

from __future__ import annotations

from typing import Any

from causalab.neural.engines.pytorch_hooks.loading import load_model


class PipelineShim:
    """Just enough pipeline for token-position resolution on plain text."""

    def __init__(self, model_key: str) -> None:
        bundle = load_model(model_key)
        self.tokenizer = bundle.tokenizer
        self.hf_model = bundle.model
        self.max_length = None
        self.use_chat_template = False

    def _chat_prefix_token_count(self) -> int:
        return 0

    def load(self, traces: Any, **kwargs: Any) -> Any:
        texts = [
            trace if isinstance(trace, str) else trace["raw_input"] for trace in traces
        ]
        return self.tokenizer(texts, return_tensors="pt", padding=True, **kwargs)
