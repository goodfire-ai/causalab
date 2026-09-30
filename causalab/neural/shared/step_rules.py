"""Validate each concrete step before planning or loading weights.

Compiler validation checks one representative per axis value. The engine
also checks every selected combination, where swept writes can collide or
a model and layer can be incompatible. It runs the same document and
loaded-featurizer rules and aggregates distinct violations. This module
stays torch-free.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping, Sequence

from causalab.protocol.rules.errors import ValidationError
from causalab.protocol.rules.data import check_loaded_featurizers
from causalab.protocol.rules.document import validate_document
from causalab.protocol.rules.errors import raise_distinct
from causalab.protocol.schema import Document

if TYPE_CHECKING:
    from causalab.io.env import ResolutionEnv

__all__ = ["check_steps"]


def check_steps(
    steps: Sequence[Document],
    env: ResolutionEnv | None = None,
    *,
    coords: Sequence[Mapping[str, Any]] | None = None,
) -> None:
    """Hold every enumerated step — parsed — to the §5 checklist
    ([`validate_document`][], and with
    ``env`` the ``file_path`` loads and rule 29's widths through
    [`check_loaded_featurizers`][]), every
    step and not the first failing one, distinct violations raised together
    ([`ValidationErrors`][causalab.protocol.rules.errors.ValidationErrors]) and identical ones
    once — the compiler's own aggregation, so a sweep whose every step breaks
    the same rule the same way is one refusal with the text ``causalab
    validate`` gives it. Rules 13 and 30 (the engine's capabilities) are the
    compiler's: the capability union over the representatives is the union
    over the steps, so they are not decided again here.

    ``coords`` are the steps' sweep coordinates, one mapping per step. With
    them, a ``subspace`` start whose site is stamped per entry is checked
    here at the entry each step selects, rather than when the fit is built
    after the weights load."""
    at: Sequence[Mapping[str, Any] | None] = (
        [None] * len(steps) if coords is None else coords
    )
    refused: list[ValidationError] = []
    for doc, point in zip(steps, at, strict=True):
        try:
            if env is None:
                validate_document(doc)  # the registry's static facts
            else:
                validate_document(doc, model_info=env.model_info)
                check_loaded_featurizers(doc, env, coords=point)
        except ValidationError as err:
            refused.append(err)
    raise_distinct(refused)
