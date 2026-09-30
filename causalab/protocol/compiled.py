"""Hold a compiled intervention specification and its derived metadata.

``CompiledProtocol`` contains the parsed document, explicit canonical form,
lowered tree, axes, dependency identities, and diagnostics. It retains one
representative for each axis value for validation. Engines enumerate concrete
points and compute their digests.

Resolved positions are attached when the tokenizer service runs. They are
execution metadata. The campaign digest identifies the canonical document."""

from __future__ import annotations

import dataclasses
import functools
from typing import Any, Mapping

from causalab.io.sources import DataIdentity, Diagnostic, ResolvedArtifact
from causalab.protocol.lowering import Axes, Axis, representative_trees
from causalab.protocol.positions.resolve import StepResolution
from causalab.protocol.rules.capability import requires_campaign
from causalab.protocol.schema import Document, parse_document

__all__ = [
    "Authored",
    "CompiledProtocol",
    "Digests",
]


@dataclasses.dataclass(frozen=True)
class Digests:
    """The document digest — the campaign's identity, the one ``--resume``
    compares a workflow's protocol step by (§7). The per-point provenance
    digests are the engine's ([`steps`][causalab.protocol.engine.RunResult.steps]): a tensor is stamped with the step digest that
    produced it, signed by the same hasher."""

    document: str


@dataclasses.dataclass(frozen=True)
class CompiledProtocol:
    """What one build produced — the campaign as identity, and nothing of the
    environment it was built against.

    * ``document`` — the parsed [`Document`][causalab.protocol.schema.types.Document] of
      the explicit tree, sweep wrappers intact (the ``gate`` stage's result);
    * ``explicit`` — the materialised campaign form the campaign digest is
      over (the canonical document, §7): overrides applied, artifact-valued
      fields resolved, families lowered, every default and
      derived width written out, the dataset identities in place, the
      ``axes`` block beside the wrappers when one was authored, sweep
      wrappers kept (the module docstring says how this and ``document``
      together carry "every implicit configuration made explicit");
    * ``tree`` — the lowered tree *before* materialisation, the display form
      the axes index into: overrides applied, artifact-valued fields
      resolved, families lowered, every ``{"axis": …}``
      wrapper the ``{"sweep": [column]}`` it stands for. What the gate parsed
      into ``document``, and what the engine substitutes each step's
      coordinates into;
    * ``axes`` — the axes the campaign's steps are indexed by, in the order
      enumeration walks them ([`axes_of`][causalab.protocol.lowering.axes_of]);
    * ``named_axes`` — the parsed ``axes`` group (§3.2) when the document
      declared one, else ``None``: the rows the engine enumerates as rows,
      never as ``tree``'s cross product;
    * ``campaign_digest`` — ``sha256`` of the canonical campaign (§7);
    * ``data`` — every dataset ref the steps name, with its content digest
      and columns;
    * ``artifacts`` — every artifact reference, with what was read and what
      the store deferred;
    * ``diagnostics`` — what the build found and did not refuse on;
    * ``positions`` — what [`resolve_positions`][causalab.protocol.pipeline.resolve_positions]
      resolved with the model's tokenizer: per positions key
      ([`positions_key`][causalab.protocol.positions.resolve.positions_key]) the
      representative's frames, addresses and ledger. ``None`` until that
      verb ran. **Derived, not identity**: positions enter no canonical
      form and no digest (§7), so the object with and without them has the
      same ``campaign_digest``; the engine reads them for every step that
      shares a key and resolves the rest itself.

    ``capabilities`` — the engine capabilities the campaign requires (§8),
    derived from the registry rows through
    [`requires_campaign`][], never a
    second table — is a property of the document and is derived from the
    [`representatives`][] on first read rather than stored: the derivation
    indexes a step's ``sites`` and ``writes`` by the names its reads and
    writes spell, and a dangling name is a §5 rule-4 violation that only
    ``validate`` refuses. Deriving at build would turn that refusal into a
    ``KeyError`` on a document that has not been validated yet; ``validate``
    reads the property after the checklist has passed.

    The read-only properties ``canonical`` and ``digests`` are the spellings
    the doors read (the CLI verbs, the run receipt and the workflow runner).
    """

    document: Document
    explicit: Mapping[str, Any]
    tree: Mapping[str, Any]
    axes: tuple[Axis, ...]
    named_axes: Axes | None
    campaign_digest: str
    data: Mapping[str, DataIdentity]
    artifacts: tuple[ResolvedArtifact, ...]
    diagnostics: tuple[Diagnostic, ...]
    positions: Mapping[str, StepResolution] | None = None

    @functools.cached_property
    def representatives(self) -> tuple[Document, ...]:
        """The §5 checklist's domain, parsed: one concrete step per axis
        value, every other axis at its first value — the first is the
        campaign's first step ([`representative_trees`][]). Every one is a step the engine also
        enumerates; a document with no axes is its own one representative."""
        return tuple(
            parse_document(tree)
            for tree in representative_trees(self.tree, self.axes, self.named_axes)
        )

    @functools.cached_property
    def capabilities(self) -> frozenset[str]:
        """The engine capabilities the whole campaign requires (§8) — the
        union over the representatives, derived from the registry rows. A
        capability is charged per field (a component read or written, a
        mechanism, a training verb), and every field value appears in some
        representative, so the union over them is the union over every
        step."""
        return requires_campaign(self.representatives)

    @property
    def canonical(self) -> Mapping[str, Any]:
        """The canonical campaign — the explicit form, sweep wrappers intact;
        the spelling the run receipt and the workflow runner read."""
        return self.explicit

    @property
    def digests(self) -> Digests:
        """The campaign digest as one record — the spelling the runner
        reads."""
        return Digests(document=self.campaign_digest)


@dataclasses.dataclass(frozen=True)
class Authored:
    """The read prefix of a compile, on its own: the authored document with
    overrides applied and nothing yet resolved.

    A workflow needs this *before* it can compile a step-dependent inner
    document — the authored tree is what it walks for step references, and
    those decide which store the compile resolves against (workflow spec
    §2.3). [`read_document`][causalab.protocol.pipeline.read_document] produces it with the compiler's own first two
    stages, and [`compile_protocol`][causalab.protocol.pipeline.compile_protocol] accepts it back as the authored
    document, so the workflow's reading and the compiler's are one
    implementation and not two that agree."""

    raw: Mapping[str, Any]
