"""Compare intervention sites for control experiments.

``SiteTuple`` describes the component, layer bands, selectors, feature shape,
and coordinate sharing of a write. Coverage compares the set of per-point
bands: a band intervention acts jointly on its layers. The comparison also
checks whether two interventions share fitted coordinates.

Controls declare any allowed differences through ``non_equivalence``. Comparison
uses canonical documents and model metadata, so it can run before weights load."""

from __future__ import annotations

import dataclasses
from typing import Any, Iterable, Sequence

from causalab.protocol.rules.errors import ValidationError
from causalab.protocol.registry import (
    COMPONENT_STREAMS,
    ModelInfo,
    _no_axis,  # pyright: ignore[reportPrivateUsage]
    _with_note,  # pyright: ignore[reportPrivateUsage]
    capability,
    component_shape,
    component_width,
)
from causalab.protocol.schema import (
    span_length,
    LAYERLESS_COMPONENTS,
    Document,
    FeaturizerSpec,
    SiteSpec,
)
from causalab.protocol.registry.shapes import FeatureShape

__all__ = [
    "EQUIVALENCE_FIELDS",
    "FeaturizerStage",
    "SiteTuple",
    "compare",
    "coverage",
    "explain",
    "sharing",
    "site_tuple",
]

#: The fields a site tuple is compared on, in the order a refusal names them
#: (workflow spec §2.2, the equivalence-field table; census
#: ``tests/workflow/test_equivalence.py``). Closed: a ``non_equivalence``
#: declaration names fields from this set and nothing else.
EQUIVALENCE_FIELDS: tuple[str, ...] = (
    "component",
    "shape",
    "layers",
    "head",
    "expert",
    "stream",
    "routed_rank",
    "featurizer",
    "dims",
    "sharing",
)

#: The fields whose per-point values are *sets* — coverage accumulates them
#: by union across points, the others by collecting the values. ``layers`` is
#: neither: a point's band is one value, and a side's coverage is the set of
#: bands (`_layer_coverage`).
_SET_FIELDS: frozenset[str] = frozenset({"stream", "sharing"})

#: The site half of the tuple — what "the same address" means before any
#: featurizer is attached.
SITE_FIELDS: tuple[str, ...] = (
    "component",
    "shape",
    "layers",
    "head",
    "expert",
    "stream",
)

#: The featurizer kinds whose output width is their ``k`` (the §2.5 chain
#: rule); ``sae`` widths are the bundle's and are unknown offline.
_RANK_KINDS: frozenset[str] = frozenset({"subspace", "pca"})


@dataclasses.dataclass(frozen=True)
class FeaturizerStage:
    """One featurizer in a write's chain, as a *shape*: kind, rank, the
    parametrization of that rank, the unit a gate parameter covers
    (``group``), the axis it indexes and how many units of it (``axis``,
    ``units``) and the parameter dtype. Not the seed, not the init basis,
    not the bundle, not a gate's ``forward`` split (§2.5: it decides which
    loss produced θ, not θ's shape — a soft-forward and a hard-forward
    fit at one site compare as equivalent) — those are values."""

    kind: str
    k: Any = None
    parametrization: Any = None
    group: Any = None
    dtype: str = "fp32"
    #: §2.5 ``axis``: a position gate's θ is over positions, a feature gate's
    #: over coordinates — two shapes at one site, so a plain-gate control is
    #: not a position-gate fit's equivalent
    axis: Any = None
    #: the number of units on that axis where the site does not fix it: a
    #: position gate's window length (§2.5), so two position gates over
    #: different windows at one site are two shapes; ``None`` for every
    #: other stage, whose unit count follows from the site
    units: Any = None

    def describe(self) -> str:
        parts = [self.kind]
        if self.k is not None:
            parts.append(f"k={self.k}")
        if self.parametrization is not None:
            parts.append(str(self.parametrization))
        if self.group is not None:
            parts.append(f"group={self.group}")
        if self.axis is not None:
            parts.append(
                f"axis={self.axis}[{self.units}]"
                if self.units is not None
                else f"axis={self.axis}"
            )
        parts.append(self.dtype)
        return "(" + ", ".join(parts) + ")"


@dataclasses.dataclass(frozen=True)
class SiteTuple:
    """Where one write acts, typed — one per (expanded point, write)."""

    #: the component name, pre/post-projection sites included
    #: (``attention_premix`` ≠ ``attention_output``, ``delta_premix``)
    component: str
    #: the component's tensor shape on the model (``component_shape``);
    #: ``None`` when no registry entry is known
    shape: FeatureShape | None
    #: the band this point's site spans (§2.4): ``(18,)`` for a one-layer
    #: site, ``(3, 4)`` for a two-layer band — one site either way; ``()``
    #: for a layerless component
    layers: tuple[int, ...]
    head: int | None
    expert: int | None
    #: the mixer stream at each covered layer — the site's declared ``stream``,
    #: else the component's bound stream, else the registry's ``layer_types``
    #: at the layer, else ``full_attention`` on a tower that declares no
    #: linear-attention mixer at all (it can carry no DeltaNet layer), else
    #: ``None`` — *unknown*, compared as such and said so
    stream: frozenset[str | None]
    #: ``(num_experts, num_experts_per_tok, moe_intermediate_size)`` on a
    #: component of the MoE block; ``None`` elsewhere
    routed_rank: tuple[int | None, int | None, int | None] | None
    #: the featurizer chain as shapes; ``()`` is the identity
    featurizer: tuple[FeaturizerStage, ...]
    #: the coordinates of the featurized value the write covers — the authored
    #: ``dims``, or every coordinate of the chain's output width when
    #: unauthored; ``None`` when that width is not decidable offline
    dims: tuple[int, ...] | None
    #: the coordinate systems this write is expressed in: ``("name", owner,
    #: featurizer)`` for a stage the document declares, ``("bundle", path)``
    #: for one it loads or saves — two tuples *share* when these intersect
    sharing: frozenset[tuple[str, ...]]


# --------------------------------------------------------------------------- #
# one write → one tuple
# --------------------------------------------------------------------------- #


def _site_layers(site: SiteSpec) -> tuple[int, ...]:
    """The band one site spans, as the parser hands it on (``SiteSpec.layers``
    is a ``tuple[int, ...]`` on an expanded point: ``(18,)`` for ``layers: 18``
    and ``layers: [18]`` alike, ``(3, 4)`` for ``layers: [3, 4]``). **The single
    seam on the field**: nothing else in this module reads it. A band is one
    site, so it is returned whole — never split into its members; ``()`` for a
    layerless component or an unexpanded leaf."""
    if site.component in LAYERLESS_COMPONENTS:
        return ()
    layers = site.layers
    if not isinstance(layers, tuple):
        return ()
    band: list[int] = []
    for layer in layers:  # pyright: ignore[reportUnknownVariableType]
        if isinstance(layer, bool) or not isinstance(layer, int):
            return ()
        band.append(layer)
    return tuple(band)


def _int_or_none(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _shape(info: ModelInfo | None, component: str) -> FeatureShape | None:
    if info is None:
        return None
    try:
        return component_shape(info, component)
    except ValidationError:
        return None


def _stream(
    component: str,
    layers: tuple[int, ...],
    declared: Any,
    info: ModelInfo | None,
) -> frozenset[str | None]:
    if isinstance(declared, str):
        return frozenset({declared})
    bound = COMPONENT_STREAMS.get(component)
    if bound is not None:
        return frozenset({bound})
    if not layers:
        return frozenset()
    if info is None:
        return frozenset({None})
    if info.layer_types is not None:
        return frozenset(
            info.layer_types[layer] if 0 <= layer < len(info.layer_types) else None
            for layer in layers
        )
    dense = (
        info.linear_num_key_heads is None
        and info.linear_num_value_heads is None
        and info.linear_key_head_dim is None
        and info.linear_value_head_dim is None
    )
    # a tower with no linear-attention mixer dimensions has no DeltaNet layer
    # to put a component on (``component_shape`` refuses every ``delta_*``
    # name there), so every layer carries the softmax mixer — decided from the
    # entry, not guessed
    return frozenset({"full_attention"}) if dense else frozenset({None})


def _routed_rank(
    component: str, info: ModelInfo | None
) -> tuple[int | None, int | None, int | None] | None:
    if "moe" not in capability(component).requires or info is None:
        return None
    return (info.num_experts, info.num_experts_per_tok, info.moe_intermediate_size)


def _chain_names(ref: Any) -> tuple[str, ...]:
    if isinstance(ref, str):
        return (ref,)
    if isinstance(ref, (tuple, list)):
        return tuple(str(name) for name in ref)  # pyright: ignore[reportUnknownVariableType]
    return ()


def _stage(doc: Document, name: str, pos: Any = None) -> FeaturizerStage:
    spec: FeaturizerSpec = doc.featurizers[name]
    kind = spec.kind if isinstance(spec.kind, str) else "identity"
    units = None
    if spec.axis is not None:
        # §2.5 `axis`: a position gate's unit count is the write's window,
        # the one gate shape the site does not size (rule 4 holds every use
        # to one length, so the write's is the set's)
        units = span_length(doc.positions.get(pos) if isinstance(pos, str) else pos)
    return FeaturizerStage(
        kind=kind,
        k=spec.k,
        parametrization=spec.parametrization,
        group=spec.group,
        dtype=spec.dtype if isinstance(spec.dtype, str) else "fp32",
        axis=spec.axis,
        units=units,
    )


def _dims(
    authored: Any,
    info: ModelInfo | None,
    component: str,
    head: int | None,
    stages: Sequence[FeaturizerStage],
) -> tuple[int, ...] | None:
    if isinstance(authored, (list, tuple)):
        return tuple(int(d) for d in authored)  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
    if info is None:
        return None
    try:
        width: int | None = component_width(info, component, head=head)
    except ValidationError:
        return None
    for stage in stages:
        if stage.kind in _RANK_KINDS:
            width = stage.k if isinstance(stage.k, int) else None
        elif stage.kind == "sae":
            width = None
        if width is None:
            return None
    return tuple(range(width)) if width is not None else None


def _sharing(
    doc: Document, names: Sequence[str], owner: str
) -> frozenset[tuple[str, ...]]:
    keys: set[tuple[str, ...]] = set()
    for name in names:
        spec = doc.featurizers.get(name)
        if spec is None:
            continue
        if isinstance(spec.file_path, str):
            keys.add(("bundle", spec.file_path))
        else:
            keys.add(("name", owner, name))
        for entry in doc.save:
            if entry.value == name and entry.site is not None:
                path = f"{owner}/{entry.file_path}" if owner else entry.file_path
                keys.add(("bundle", path))
    return frozenset(keys)


def site_tuple(
    doc: Document, write: str, info: ModelInfo | None, *, owner: str = ""
) -> SiteTuple:
    """The typed site of ``write`` in the (expanded, concrete) document
    ``doc`` on the model ``info`` describes. ``owner`` names the document
    (the workflow step) for the coordinate-sharing keys: a featurizer name is
    an identity within its owner only."""
    spec = doc.writes[write]
    site_name = str(spec.site)
    site = doc.sites[site_name]
    component = str(site.component)
    layers = _site_layers(site)
    head = _int_or_none(site.head)
    names = _chain_names(spec.featurizer)
    stages = tuple(_stage(doc, n, spec.pos) for n in names if n in doc.featurizers)
    dims = _dims(spec.dims, info, component, head, stages)
    return SiteTuple(
        component=component,
        shape=_shape(info, component),
        layers=layers,
        head=head,
        expert=_int_or_none(site.expert),
        stream=_stream(component, layers, site.stream, info),
        routed_rank=_routed_rank(component, info),
        featurizer=stages,
        dims=dims,
        sharing=_sharing(doc, names, owner),
    )


def coverage(
    documents: Iterable[Document],
    info: ModelInfo | None,
    *,
    owner: str = "",
    writes: Iterable[str] | None = None,
) -> frozenset[SiteTuple]:
    """Every (point, write) site of an expanded document: one tuple per write
    an intervened model lists, per point document. ``writes`` narrows to the
    named writes (a ``matched_random`` pairing looks at the writes through one
    featurizer)."""
    tuples: set[SiteTuple] = set()
    wanted = None if writes is None else set(writes)
    for doc in documents:
        names: set[str] = set()
        for im in doc.intervened_models.values():
            if isinstance(im.writes, tuple):
                names.update(im.writes)
        if not names:
            names = set(doc.writes)
        for name in sorted(names):
            if name not in doc.writes or (wanted is not None and name not in wanted):
                continue
            tuples.add(site_tuple(doc, name, info, owner=owner))
    return frozenset(tuples)


# --------------------------------------------------------------------------- #
# the comparison
# --------------------------------------------------------------------------- #


def _layer_coverage(cov: Iterable[SiteTuple]) -> frozenset[tuple[int, ...]]:
    """One side's layer coverage: the **set of per-point bands**. A band is one
    site (§2.4), so ``{(3, 4)}`` (one band at one point) and ``{(3,), (4,)}``
    (two one-layer points) are not the same coverage — the layers agree, the
    interventions do not, and rule 16 asks about the intervention. This is the
    one place the decision lives: to reverse it (union coverage) the body
    becomes ``frozenset((layer,) for t in cov for layer in t.layers)``."""
    return frozenset(t.layers for t in cov)


def _values(cov: Iterable[SiteTuple], field: str) -> frozenset[Any]:
    if field == "layers":
        return _layer_coverage(cov)
    out: set[Any] = set()
    for t in cov:
        value = getattr(t, field)
        if field in _SET_FIELDS:
            out.update(value)
        else:
            out.add(value)
    return frozenset(out)


def sharing(a: Iterable[SiteTuple], b: Iterable[SiteTuple]) -> str:
    """``shared`` when the two coverages are expressed in one coordinate
    system — a featurizer name inside one document, a saved bundle one of them
    loads — else ``distinct``."""
    return "shared" if _values(a, "sharing") & _values(b, "sharing") else "distinct"


def compare(a: Iterable[SiteTuple], b: Iterable[SiteTuple]) -> tuple[str, ...]:
    """The fields of [`EQUIVALENCE_FIELDS`][] the two coverages differ in,
    in vocabulary order; ``()`` when they are site-equivalent. ``sharing`` is
    listed when the two *share* a coordinate system: a control scored in its
    target's own basis is the one difference that reads as agreement."""
    a, b = frozenset(a), frozenset(b)
    differing: list[str] = []
    for field in EQUIVALENCE_FIELDS:
        if field == "sharing":
            if sharing(a, b) == "shared":
                differing.append(field)
        elif _values(a, field) != _values(b, field):
            differing.append(field)
    return tuple(differing)


# --------------------------------------------------------------------------- #
# what a refusal says
# --------------------------------------------------------------------------- #


def _show(value: Any) -> str:
    if isinstance(value, FeaturizerStage):
        return value.describe()
    if isinstance(value, FeatureShape):
        return value.describe()
    if isinstance(value, tuple):
        return "(" + ", ".join(_show(v) for v in value) + ")"  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
    return repr(value)


def _listed(values: frozenset[Any]) -> str:
    return "[" + ", ".join(_show(v) for v in sorted(values, key=_show)) + "]"


def _shapes_by_component(cov: frozenset[SiteTuple]) -> str:
    seen: dict[str, str] = {}
    for t in sorted(cov, key=lambda t: t.component):
        seen.setdefault(
            t.component, t.shape.describe() if t.shape is not None else "unknown"
        )
    return ", ".join(f"{c!r} is {s}" for c, s in seen.items())


def _band_shown(band: tuple[int, ...]) -> str:
    return str(list(band)) if len(band) != 1 else str(band[0])


def _bands_listed(bands: frozenset[Any]) -> str:
    """A set of bands: one-layer bands as their index, wider bands as lists —
    ``[0, 1]`` is two one-layer points, ``[[0, 1]]`` one two-layer band."""
    return "[" + ", ".join(_band_shown(b) for b in sorted(bands)) + "]"


def _bands_phrase(bands: frozenset[Any]) -> str:
    wide = sorted(b for b in bands if len(b) > 1)
    if wide:
        return (
            "spans the band " + ", ".join(_band_shown(b) for b in wide) + " as one site"
        )
    return (
        "visits "
        + ", ".join(_band_shown(b) for b in sorted(bands) if b)
        + " one layer per point"
    )


def _dims_summary(values: frozenset[Any]) -> str:
    parts: list[str] = []
    for dims in sorted(values, key=_show):
        if dims is None:
            parts.append("unknown")
        elif len(dims) > 8:
            parts.append(f"{len(dims)} coordinates")
        else:
            parts.append(str(list(dims)))
    return " / ".join(parts)


def explain(field: str, a: Iterable[SiteTuple], b: Iterable[SiteTuple]) -> str:
    """Why ``field`` separates the two coverages, and what that means for a
    comparison between them — the middle of a rule-16 refusal. The caller
    prefixes who is being compared and appends how to declare it."""
    a, b = frozenset(a), frozenset(b)
    va, vb = _values(a, field), _values(b, field)
    if field == "component":
        note = ""
        for t in (*a, *b):
            if t.shape is not None and t.shape.note:
                note = ". " + t.shape.note.rstrip(".")
                break
        return (
            f"one writes {_listed(va)}, the other {_listed(vb)} — so the two act on "
            f"different tensors ({_shapes_by_component(a)}; {_shapes_by_component(b)})"
            f"{note}"
        )
    if field == "shape":
        return (
            f"{_shapes_by_component(a)}; {_shapes_by_component(b)} — so the two "
            "feature spaces are not one space, and a score in one is not a score "
            "in the other"
        )
    if field == "layers":
        union_a = {layer for band in va for layer in band}
        union_b = {layer for band in vb for layer in band}
        if union_a == union_b:
            return (
                f"layers {_bands_listed(va)} vs {_bands_listed(vb)} — the same "
                f"layers, not the same sites: one {_bands_phrase(va)}, the other "
                f"{_bands_phrase(vb)} — coverage-equal, intervention-different; a "
                "band is one site (one intervention across every member), and a "
                "control that visits its layers one per point ran no intervention "
                "across the band"
            )
        return (
            f"layers {_bands_listed(va)} vs {_bands_listed(vb)} — so the two do "
            "not cover the same layers; a control pinned to one layer says "
            "nothing about the others its target sweeps"
        )
    if field == "head":
        for x, y in ((a, b), (b, a)):
            heads = _values(x, "head") - {None}
            for t in y:
                if heads and t.head is None and t.shape is not None:
                    if t.shape.head_space is None:
                        return _with_note(
                            f"one names head {sorted(heads)}, and "
                            f"{_no_axis(t.component, 'head', t.shape)} — so there is "
                            "no head of the one for the other's head to match",
                            t.shape,
                        )
        return (
            f"head {_listed(va)} vs {_listed(vb)} — so the two act on different "
            "heads (or one on the whole component)"
        )
    if field == "expert":
        return (
            f"expert {_listed(va)} vs {_listed(vb)} — so the two act inside "
            "different routed experts (or one on every routed slot)"
        )
    if field == "stream":
        unknown = (
            " (the model declares no layer_types for a hybrid tower, so the "
            "stream at some layer is not known offline and is compared as unknown)"
            if None in va or None in vb
            else ""
        )
        return (
            f"mixer streams {_listed(va)} vs {_listed(vb)} — so one covers a "
            "Gated DeltaNet layer the other does not (DeltaNet inclusion), and a "
            f"score on one mixer is not a score on the other{unknown}"
        )
    if field == "routed_rank":
        return (
            f"routed rank (num_experts, top_k, moe_inner) {_listed(va)} vs "
            f"{_listed(vb)} — so the two do not address the same routed experts"
        )
    if field == "featurizer":
        return (
            f"featurizer chains {_listed(va)} vs {_listed(vb)} — so the two are "
            "expressed in different feature spaces (kind, k, parametrization, "
            "group, axis and its unit count, or parameter dtype)"
        )
    if field == "dims":
        return (
            f"coordinates {_dims_summary(va)} vs {_dims_summary(vb)} — so the two "
            "do not cover the same coordinates of the featurized value"
        )
    if field == "sharing":
        shared = va & vb
        return (
            f"both are expressed in one coordinate system {_listed(shared)} — so "
            "the control is scored in its target's own basis and is not an "
            "independent draw"
        )
    raise ValueError(
        f"unknown equivalence field {field!r}; expected one of {EQUIVALENCE_FIELDS}"
    )
