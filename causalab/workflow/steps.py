"""Enumerate and sign the intervention points used by a workflow step.

This module uses the shared sweep order and the protocol's step hasher. The
workflow runner uses the resulting identities to schedule, record, and reuse work."""

from __future__ import annotations

import dataclasses
from typing import Any, Mapping

from causalab.io.env import ResolutionEnv
from causalab.neural.shared.sweep import (
    Expansion,
    Point,
    enumerate_steps,
    sign_steps,
)
from causalab.protocol.compiled import CompiledProtocol
from causalab.protocol.schema import Document, parse_document

__all__ = ["InnerProtocol"]


@dataclasses.dataclass(frozen=True)
class InnerProtocol:
    """One inner document as the workflow reads it at load: the compile
    (``compiled``) plus its enumerated steps — the axes and points in the
    canonical order (``expansion``), each step parsed (``point_documents``),
    canonicalized (``canonical_points``) and signed with the load-time
    environment (``point_digests``; a step-dependent document's are the
    deferring store's, and the run-time compile legitimately moves them, which
    is why the fan-out join reads the children's records instead)."""

    compiled: CompiledProtocol
    expansion: Expansion
    point_documents: tuple[Document, ...]
    canonical_points: tuple[Mapping[str, Any], ...]
    point_digests: tuple[str, ...]

    @classmethod
    def enumerate(cls, compiled: CompiledProtocol, env: ResolutionEnv) -> InnerProtocol:
        """Enumerate and sign every step of ``compiled`` against ``env`` — the
        one spelling of the engine's sweep in the workflow layer."""
        expansion = enumerate_steps(compiled)
        signed = sign_steps(expansion, env)
        return cls(
            compiled=compiled,
            expansion=Expansion(
                axes=expansion.axes,
                points=tuple(
                    Point(coords=step.coords, raw=step.raw) for step in signed
                ),
            ),
            point_documents=tuple(parse_document(step.raw) for step in signed),
            canonical_points=tuple(step.canonical for step in signed),
            point_digests=tuple(step.digest for step in signed),
        )
