"""The answer check of ``tests/demos/test_papers.py`` for the packages on a
gated model: ``causalab validate --tokenizer`` over each workflow, so every
static inner document's positions and metric answers resolve with its
model's tokenizer and none is glued to its prompt.

The CPU tier has no Hub token, so these packages are checked here, on the
golden runner's offline cache with a licensed token
(``tests/test_no_gated_models.py``). Only tokenizers load: no accelerator is
needed, but the tokenizers of every gated package model must be cached at
the revisions the documents pin.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.demos.test_papers import (
    PAPERS,
    _json_files,  # pyright: ignore[reportPrivateUsage]
    _on_a_gated_model,  # pyright: ignore[reportPrivateUsage]
    check_answers_resolve,
)

pytestmark = pytest.mark.golden


@pytest.mark.parametrize(
    "workflow",
    [w for w in _json_files("workflow") if _on_a_gated_model(w)],
    ids=lambda p: str(p.relative_to(PAPERS)),
)
def test_answer_columns_resolve_on_gated_models(workflow: Path) -> None:
    check_answers_resolve(workflow)
