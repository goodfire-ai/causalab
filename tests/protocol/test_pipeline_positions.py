"""``pipeline.resolve_positions``: the run door's verb — every
representative's positions resolved with the model's tokenizer and attached
to the compiled object, derived and not identity; ``validate`` alone stays
torch-free and attaches nothing."""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from causalab.protocol.pipeline import build, resolve_positions, validate
from causalab.protocol.positions.resolve import positions_key
from causalab.protocol.rules.errors import ProtocolError
from causalab.io.env import ResolutionEnv

from tests._helpers.tiny import TINY_RANDOM_GPT2_MODEL_NAME
from tests.protocol._docs import base_doc


pytestmark = pytest.mark.unit


CALLS: list[tuple[str, str]] = []


@pytest.fixture()
def tokenized_env(env: ResolutionEnv) -> ResolutionEnv:
    """The fixture environment with the tiny GPT-2 tokenizer standing in for
    every model key — the document names ``gpt2`` and the tiny fixture
    carries the GPT-2 tokenizer. Every load is recorded in ``CALLS``."""
    from causalab.io.tokenizer import load_tokenizer

    tokenizer = load_tokenizer(TINY_RANDOM_GPT2_MODEL_NAME)
    CALLS.clear()

    def tokenizers(key: str, revision: str) -> Any:
        CALLS.append((key, revision))
        return tokenizer

    return dataclasses.replace(env, tokenizers=tokenizers)


def test_validate_attaches_nothing_and_resolve_positions_attaches_every_representative(
    tokenized_env: ResolutionEnv,
) -> None:
    raw = base_doc()
    raw["method"]["sites"]["tgt"]["layers"] = {"sweep": [1, 2, 3]}
    compiled = validate(build(raw, env=tokenized_env), env=tokenized_env, data=True)
    assert compiled.positions is None
    assert CALLS == []
    resolved = resolve_positions(compiled, env=tokenized_env)
    assert resolved.campaign_digest == compiled.campaign_digest  # derived, not identity
    assert resolved.positions is not None
    # three representatives (one per layer), one positions key: a layer sweep
    # moves no position, so the three share one resolution and one tokenizer load
    keys = {positions_key(pdoc) for pdoc in compiled.representatives}
    assert keys == set(resolved.positions)
    assert len(keys) == 1
    assert CALLS == [("gpt2", "main")]
    step = resolved.positions[next(iter(keys))]
    assert set(step.positions.frames) == {"base", "counterfactual"}
    assert {role for _key, role in step.positions.addresses} == {
        "base",
        "counterfactual",
    }
    assert all(
        not resolved_.problems for resolved_ in step.positions.addresses.values()
    )
    assert step.ledger is None  # the document saves no ledger
    # idempotent: resolving again finds every key and loads no tokenizer
    again = resolve_positions(resolved, env=tokenized_env)
    assert set(again.positions or {}) == keys
    assert CALLS == [("gpt2", "main")]


def test_the_ledger_rides_along_when_the_document_saves_one(
    tokenized_env: ResolutionEnv,
) -> None:
    raw = base_doc()
    raw["method"]["save"] = [
        *raw["method"].get("save", []),
        {"kind": "location_ledger", "file_path": "ledger.json"},
    ]
    compiled = validate(build(raw, env=tokenized_env), env=tokenized_env, data=True)
    resolved = resolve_positions(compiled, env=tokenized_env)
    assert resolved.positions is not None
    (step,) = resolved.positions.values()
    assert step.ledger is not None and len(step.ledger) > 0


def test_an_out_of_bounds_index_is_refused_before_any_weights(
    tokenized_env: ResolutionEnv,
) -> None:
    raw = base_doc()
    raw["method"]["reads"]["logits"]["pos"] = 500
    compiled = validate(build(raw, env=tokenized_env), env=tokenized_env, data=True)
    with pytest.raises(ProtocolError, match="out of bounds"):
        resolve_positions(compiled, env=tokenized_env)
