"""Regression guard for the IOI logit-diff readout token (#1).

The path-patching / IOI metrics score a *single* vocab row per answer — the
correct or distractor name's token id. At the readout position the IOI prompt
ends ``"... gave a drink to"``, so GPT-2 emits the **leading-space** token
``" Mary"`` (id 5335), a different BPE id than the bare ``"Mary"`` for ~half of
the name vocabulary. Reading the bare id silently scores the wrong row and
contaminates every direct-effect number.

An answer string is tokenized as written (§2.10), so the guard has two halves:
the IOI table writes its answers with the leading space
(``causal_models.raw_output`` is ``" " + IO``), and the resolver returns
exactly that row for exactly that string, for every IOI name.
"""

from __future__ import annotations

import json

import pytest

from causalab.protocol.answers import column_token_id
from causalab.tasks import TASKS_ROOT

pytestmark = pytest.mark.numerical_unit


def test_the_table_answer_resolves_to_the_emitted_leading_space_id(
    gpt2_pipeline,
) -> None:
    tokenizer = gpt2_pipeline.tokenizer
    rows = json.loads((TASKS_ROOT / "IOI" / "data" / "default.json").read_text())
    answers = sorted({row[c] for row in rows for c in ("base_answer", "cf_answer")})
    assert answers, "the shipped IOI table carries no answers"
    mismatched: list[tuple[str, int, int]] = []
    for answer in answers:
        assert answer.startswith(" "), f"{answer!r}: the table's form moved"
        emitted = tokenizer.encode(answer, add_special_tokens=False)
        assert len(emitted) == 1, f"{answer!r} is not single-token"
        resolved = column_token_id(tokenizer, answer)
        if resolved != emitted[0]:
            mismatched.append((answer, resolved, emitted[0]))

    assert not mismatched, (
        "column_token_id must return the emitted (leading-space) id for the IOI "
        f"table's answers; {len(mismatched)}/{len(answers)} read the wrong vocab "
        f"row, e.g. {mismatched[:5]}"
    )
