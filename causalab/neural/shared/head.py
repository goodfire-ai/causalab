"""Project the vocabulary head at selected read positions.

An eligible ``lm_head`` read gathers ``ln_final`` at its positions and
projects those rows through the model's head. The engine can then skip
the head in the model forward. The campaign cache uses the same predicate
to capture hidden states for later projection.

Whole-sequence reads, generated reads, decoding groups, and models with a
head write keep the ordinary head tap. Differentiable reads also keep it:
changing the backward GEMM shape can change bf16 gradient rounding and
training outcomes. ``ENV_PROJECT_UNDER_GRAD=1`` enables that numerical
change explicitly. CUDA parity is checked at the tested workflow shapes.

Featurizers and dimension selections apply after the gathered projection.
Saved reads contain the resulting position-resolved tensor.
"""

from __future__ import annotations

import dataclasses
import os
from typing import Any, Callable, Iterable

import torch

from causalab.neural.shared.sites import ResolvedSite, resolve_site
from causalab.protocol.positions.encoding import generated_budget
from causalab.neural.shared.plan import group_reads, write_names
from causalab.protocol.schema import Document, PositionSpec, ReadSpec, SiteSpec

__all__ = [
    "ENV_PROJECT_UNDER_GRAD",
    "HEAD",
    "HEAD_INPUT",
    "ReadTap",
    "capture_spec",
    "head_module",
    "projects_head",
    "resolve_read_taps",
    "taps_head",
]

#: The component the vocabulary projection is read at.
HEAD = "lm_head"
#: The component that is the head's input — what a projecting read taps.
HEAD_INPUT = "ln_final"
#: Set to ``1`` to project a read a gradient flows through as well — the
#: head's backward then runs at ``M = rows`` and its bf16 gradient may differ
#: from the model's in the last bit (module docstring). Unset: exact training.
ENV_PROJECT_UNDER_GRAD = "CAUSALAB_PROJECT_HEAD_UNDER_GRAD"


def projects_under_grad() -> bool:
    """Whether the environment asks for the projection under grad too."""
    return os.environ.get(ENV_PROJECT_UNDER_GRAD, "") == "1"


def _group_reads(doc: Document, model: str, input_role: str) -> list[ReadSpec]:
    return [doc.reads[ref.read] for ref in group_reads(doc, model, input_role)]


def _writes_at_head(doc: Document, model: str) -> bool:
    names = write_names(doc, model)
    if names is None:
        return True  # an unexpanded write set: decide nothing, keep the head
    return any(
        doc.sites[str(doc.writes[ename].site)].component == HEAD for ename in names
    )


def projects_head(doc: Document, model: str, input_role: str, rname: str) -> bool:
    """Whether read ``rname`` — a read of group ``(model, input_role)`` — is
    served by projecting the gathered ``ln_final`` rows through the head
    rather than by tapping the head (module docstring). ``False`` for every
    read that does not tap ``lm_head``."""
    read = doc.reads[rname]
    if doc.sites[str(read.site)].component != HEAD:
        return False
    spec = doc.positions[read.pos] if isinstance(read.pos, str) else read.pos
    if not isinstance(spec, PositionSpec) or spec.all is not None:
        return False
    if any(
        generated_budget(doc, other.pos) is not None
        for other in _group_reads(doc, model, input_role)
    ):
        return False
    return not _writes_at_head(doc, model)


def capture_spec(doc: Document, model: str, input_role: str, rname: str) -> SiteSpec:
    """The site read ``rname`` **captures** in its group's forward: its own,
    or ``ln_final`` when it projects the head itself."""
    if projects_head(doc, model, input_role, rname):
        return SiteSpec(component=HEAD_INPUT)
    return doc.sites[str(doc.reads[rname].site)]


def head_module(bundle: Any) -> Any:
    """The head module of ``bundle``'s model, as the site resolver finds it."""
    return resolve_site(bundle, SiteSpec(component=HEAD)).module


@dataclasses.dataclass(frozen=True)
class ReadTap:
    """What one read captures and how the captured slice becomes its value:
    ``site`` is the site the read names (its shape, its width, what a
    featurizer is built for), ``capture`` the site the forward taps for it,
    and ``project`` — the head module, when the two differ — runs over the
    gathered rows before anything else (``_finalize_read(project=)``)."""

    site: ResolvedSite
    capture: ResolvedSite
    project: Callable[[torch.Tensor], torch.Tensor] | None = None


def resolve_read_taps(
    bundle: Any,
    doc: Document,
    model: str,
    input_role: str,
    reads: Iterable[tuple[str, ReadSpec]],
    *,
    differentiable: bool = False,
) -> dict[str, ReadTap]:
    """Resolve the prompt-frame reads of one group to their taps: each read's
    own site, and — for a read [`projects_head`][] admits — ``ln_final``
    as the capture with the head module as the projection. The head is
    resolved once for the group.

    ``differentiable`` says a gradient flows through this group's reads (a
    grad-enabled executor reading the model it trains): such a group keeps
    the head as the model runs it, so the backward GEMM is the one the
    model's own forward would have paid and the gradient is bit-identical
    (module docstring) — unless [`ENV_PROJECT_UNDER_GRAD`][] asks
    otherwise."""
    project = not differentiable or projects_under_grad()
    out: dict[str, ReadTap] = {}
    head: Any = None
    norm: ResolvedSite | None = None
    for rname, read in reads:
        site = resolve_site(bundle, doc.sites[str(read.site)])
        if not project or not projects_head(doc, model, input_role, rname):
            out[rname] = ReadTap(site=site, capture=site)
            continue
        if head is None:
            head = head_module(bundle)
            norm = resolve_site(bundle, SiteSpec(component=HEAD_INPUT))
        assert norm is not None
        out[rname] = ReadTap(site=site, capture=norm, project=head)
    return out


def taps_head(sites: Iterable[ResolvedSite]) -> bool:
    """Whether any of ``sites`` is the head — a tap or a write there means
    the forward has to run it."""
    return any(site.component == HEAD for site in sites)
