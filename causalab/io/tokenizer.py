"""Load the tokenizer used by position resolution and model execution.

``load_tokenizer`` sets left padding and uses EOS as the pad token when the
checkpoint has no pad token. Transformers is imported inside the loader so
services that inspect documents can remain independent of numerical libraries."""

from __future__ import annotations

from typing import Any, Callable

__all__ = ["TokenizerResolver", "load_tokenizer"]

#: ``(model key, revision) -> tokenizer``: the service a
#: [`ResolutionEnv`][causalab.io.env.ResolutionEnv] carries.
TokenizerResolver = Callable[[str, str], Any]


def load_tokenizer(key: str, revision: str = "main") -> Any:
    """The tokenizer of ``key`` at ``revision``, configured for the one
    padding convention both engines run: left padding, and ``eos`` as the
    pad token where the tokenizer declares none."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(key, revision=revision)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
