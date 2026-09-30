"""Resolve compact document forms into explicit structure.

This module discovers sweep axes, substitutes coordinates, and expands
``at_once`` families and named axes. It also defines the coordinate labels used
by saved tensor entries. The engine uses the resulting axes to enumerate points."""

from __future__ import annotations

import dataclasses
import itertools
from collections import Counter
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

from causalab.protocol.rules.errors import (
    ParseError,
    ProtocolError,
    ValidationError,
    suggest,
)
from causalab.protocol.registry import ModelInfo
from causalab.protocol.schema import (
    Do,
    Document,
    dotted_path,
    IMSpec,
    NAMED_SECTIONS,
    OBJECTIVE_WEIGHT_PREFIX,
    ReadRef,
    ReadSpec,
    REGULARIZER_KINDS,
    RESERVED_NAMES,
    SiteSpec,
    WriteSpec,
)

__all__ = [
    # bands (from the planner)
    "band_member",
    "lower_bands",
    # the axis half of sweep.py
    "AT_ONCE_KEY",
    "DEFAULT_POINT_CAP",
    "MAX_AXIS_VALUES",
    "SWEEP_KEY",
    "Axis",
    "axis_values",
    "band_label",
    "coordinate_label",
    "find_axes",
    "label_value",
    "short_coords",
    "substitute",
    # what the protocol layer decides from the axes alone
    "CAP_MESSAGE",
    "axes_of",
    "point_count",
    "representative_trees",
    # families
    "CONSUMER_SECTIONS",
    "DO_OPERAND_FIELDS",
    "FAMILY_SECTIONS",
    "MAX_FAMILY_MEMBERS",
    "NAME_FIELDS",
    "NAMES_KEY",
    "RULE",
    "expand_families",
    "has_families",
    # named axes
    "AXES_KEY",
    "AXIS_KEY",
    "AXIS_KINDS",
    "CLIP_TARGETS",
    "RULE_KINDS",
    "Axes",
    "NamedAxis",
    "canonical_axes",
    "entries",
    "has_axes",
    "lower_axes",
    "parse_axes",
    "row_tree",
]


# --------------------------------------------------------------------------- #
# the axis half of sweep.py
# --------------------------------------------------------------------------- #


#: The wrapper that declares a sweep axis.
SWEEP_KEY = "sweep"

#: Its sibling (§3.1), which declares an axis *inside* one point rather than
#: across points. Named here rather than in [`causalab.protocol.families`][causalab.protocol.registry.families]
#: so the nested-wrapper check below can see both without importing the module
#: that expands it — and so the two keywords are declared side by side, which
#: is where a reader comparing them will look.
AT_ONCE_KEY = "at_once"

#: Refuse a cross product larger than this without an explicit override
#: (§5.14 — "may be capped without an explicit override flag").
DEFAULT_POINT_CAP = 4096

#: Hard sanity bound on one axis's value count, checked BEFORE the range
#: sugar materializes — the point cap is overridable, this is not.
MAX_AXIS_VALUES = 1_000_000


@dataclasses.dataclass(frozen=True)
class Axis:
    """One sweep axis: the path of the wrapped value and its values.

    ``path`` addresses the wrapper's location in the raw tree
    (``("method", "sites", "target", "layers")``); ``id`` is its section-rooted
    dotted spelling (``sites.target.layers``, §1) — the coordinate key in
    results and derived names, and the spelling a workflow's ``emit``,
    ``order`` and plot axes use.
    """

    path: tuple[str, ...]
    values: tuple[Any, ...]

    @property
    def id(self) -> str:
        return dotted_path(self.path)


def _is_sweep_node(node: Any) -> bool:
    return isinstance(node, dict) and SWEEP_KEY in node


def axis_values(
    spec: Any,
    *,
    path: str,
    rule: int | str = 14,
    keyword: str = SWEEP_KEY,
    article: str = "a",
) -> tuple[Any, ...]:
    """The value list an axis wrapper's payload denotes (§3).

    A list is itself; a ``{"range": [start, stop, step?]}`` object is sugar for
    the half-open range it expands to. One grammar, one implementation, two
    callers: ``{"sweep": …}`` here and ``{"at_once": …}`` in
    [`causalab.protocol.families`][causalab.protocol.registry.families], which is what keeps the two keywords
    from drifting into two dialects of "a list of values". ``rule`` and
    ``keyword`` are what differ — the checklist rule each cites, and the word
    its messages use. ``article`` goes with the keyword: "a sweep wrapper", "an
    at_once wrapper", and a reader who wrote one keyword should never be told
    about the other in bad grammar.
    """
    if isinstance(spec, Mapping):
        if set(spec) != {"range"}:
            raise ValidationError(
                rule,
                f"{article} {keyword} object form takes exactly {{'range': [...]}}",
                path=path,
            )
        rng = spec["range"]
        if (
            not isinstance(rng, list)
            or not 2 <= len(rng) <= 3
            or not all(isinstance(v, int) and not isinstance(v, bool) for v in rng)
        ):
            raise ValidationError(
                rule,
                f"{keyword} range must be [start, stop] or [start, stop, step] "
                "of integers",
                path=path,
            )
        step = rng[2] if len(rng) == 3 else 1
        if step == 0:
            raise ValidationError(
                rule, f"{keyword} range step must be non-zero", path=path
            )
        n = len(range(rng[0], rng[1], step))  # O(1) — no materialization yet
        if n > MAX_AXIS_VALUES:
            raise ValidationError(
                rule,
                f"{keyword} range denotes {n} values, over the per-axis bound of "
                f"{MAX_AXIS_VALUES} — refuse before materializing (§5.14)",
                path=path,
            )
        values: tuple[Any, ...] = tuple(range(rng[0], rng[1], step))
    elif isinstance(spec, list):
        values = tuple(spec)
    else:
        raise ValidationError(
            rule,
            f"{article} {keyword} wrapper takes a list or a range object, got "
            f"{type(spec).__name__}",
            path=path,
        )
    if not values:
        raise ValidationError(
            rule, f"{article} {keyword} axis must have at least one value", path=path
        )
    if _any_nested_wrapper(values):
        raise ValidationError(
            rule, f"{keyword} values may not contain nested axis wrappers", path=path
        )
    return values


def _sweep_values(node: Mapping[str, Any], path: tuple[str, ...]) -> tuple[Any, ...]:
    """The value list a sweep wrapper denotes (§3). Shape errors are §5.14
    rejections — the parser already checks the shapes it can see, but expansion
    may encounter wrappers in positions the schema types as free-form."""
    if len(node) != 1:
        raise ValidationError(
            14, "a sweep wrapper holds nothing but the axis", path=".".join(path)
        )
    return axis_values(node[SWEEP_KEY], path=".".join(path))


def _any_nested_wrapper(node: Any) -> bool:
    """Whether an axis wrapper of *either* kind sits inside ``node``.

    Both are refused inside an axis's values: nested sweeps because a value is
    a value, and an ``at_once`` because families expand *before* axes are
    found, so one surviving in here is one that had no name identity to expand
    against (§3.1).
    """
    if _is_sweep_node(node) or (isinstance(node, Mapping) and AT_ONCE_KEY in node):
        return True
    if isinstance(node, Mapping):
        return any(_any_nested_wrapper(v) for v in node.values())
    if isinstance(node, (list, tuple)):
        return any(_any_nested_wrapper(v) for v in node)
    return False


#: The one method section that is a **list** (§2.12): its entries are
#: addressed by index — ``save[i]`` — so a value inside one has a name
#: identity (``save[0].aggregation.k``) where a value inside any other list
#: (an authored ``dims``) has none.
INDEXED_SECTION_PATH: tuple[str, ...] = ("method", "save")


def indexed_item_paths(
    node: list[Any], path: tuple[str, ...]
) -> tuple[tuple[str, ...], ...] | None:
    """The per-item paths of a list whose items are addressable — the
    ``save`` section's ``save[i]`` segments ([`tree_path`][causalab.protocol.schema.types.tree_path] reads an
    index on a section segment) — or ``None`` for a list whose items have
    no name identity."""
    if path != INDEXED_SECTION_PATH:
        return None
    return tuple(path[:-1] + (f"{path[-1]}[{i}]",) for i in range(len(node)))


def find_axes(raw: Mapping[str, Any]) -> tuple[Axis, ...]:
    """Every sweep axis in the document, in order of first appearance.

    Wrappers are only discovered under string keys (table entries and their
    fields) and inside the indexed ``save`` list (``save[i].…``) — a wrapper
    inside any other list (e.g. inside an authored ``dims`` list) has no
    name identity and is rejected.
    """
    axes: list[Axis] = []

    def walk(node: Any, path: tuple[str, ...]) -> None:
        if _is_sweep_node(node):
            axes.append(Axis(path=path, values=_sweep_values(node, path)))
            return
        if isinstance(node, Mapping):
            for key, value in node.items():
                walk(value, path + (str(key),))
        elif isinstance(node, list):
            item_paths = indexed_item_paths(node, path)
            if item_paths is not None:
                for item, item_path in zip(node, item_paths):
                    walk(item, item_path)
                return
            for item in node:
                if _any_nested_wrapper(item):
                    raise ValidationError(
                        14,
                        "a sweep wrapper inside a list has no name identity; "
                        "declare the axis on a named field",
                        path=".".join(path),
                    )

    walk(raw, ())
    return tuple(axes)


def substitute(
    node: Any, assignment: Mapping[tuple[str, ...], Any], path: tuple[str, ...]
) -> Any:
    if _is_sweep_node(node):
        return assignment[path]
    if isinstance(node, Mapping):
        return {
            key: substitute(value, assignment, path + (str(key),))
            for key, value in node.items()
        }
    if isinstance(node, list):
        item_paths = indexed_item_paths(node, path)
        if item_paths is not None:
            return [
                substitute(item, assignment, item_path)
                for item, item_path in zip(node, item_paths)
            ]
        return [substitute(item, assignment, path) for item in node]
    return node


def short_coords(
    coords: Mapping[str, Any], *, entry: str | None = None
) -> dict[str, Any]:
    """Coordinates keyed by their **short** names — the names that appear in
    labels, in saved tensor keys, and therefore in an ``entry`` selector
    ([`causalab.protocol.bundles`][]).

    Coordinates on the named entry itself drop the entry prefix
    (``k``, not ``featurizers.rot.k``); transitive coordinates keep
    ``<entry>.<field>``; the table name is always dropped. ``train`` axes
    read best bare (``seed``)."""
    out: dict[str, Any] = {}
    for axis_id, value in coords.items():
        segments = axis_id.split(".")
        if len(segments) >= 2 and segments[0] in (
            "positions",
            "sites",
            "featurizers",
            "params",
            "code",
            "reads",
            "writes",
            "intervened_models",
            "train",
        ):
            segments = segments[1:]
        elif len(segments) >= 2 and segments[0].startswith("save["):
            # a save entry's own axis (`save[0].aggregation.k`) labels as the
            # field it sweeps: the entry index is an address, not a name
            segments = segments[1:]
            if len(segments) >= 2 and segments[0] == "aggregation":
                segments = segments[1:]
        if entry is not None and len(segments) >= 2 and segments[0] == entry:
            segments = segments[1:]
        out[".".join(segments)] = value
    return out


def coordinate_label(coords: Mapping[str, Any], *, entry: str | None = None) -> str:
    """The ``[k=8]`` / ``[target.layers=5]`` suffix for derived names (§3),
    over [`short_coords`][]."""
    parts = [
        f"{name}={label_value(value)}"
        for name, value in short_coords(coords, entry=entry).items()
    ]
    return f"[{','.join(parts)}]" if parts else ""


def label_value(value: Any) -> str:
    """One coordinate value as it appears in a label — and therefore in a
    saved tensor key, which [`causalab.protocol.bundles`][] matches
    against, so this rendering is part of the artifact contract."""
    if isinstance(value, Mapping):
        # a swept spec (e.g. a position): label by its single distinguishing pair
        pairs = ",".join(f"{k}:{v}" for k, v in value.items())
        return "{" + pairs + "}"
    if isinstance(value, (list, tuple)):
        return band_label(value)
    return str(value)


def band_label(layers: Sequence[Any]) -> str:
    """A layer band (§2.4 ``layers``) as it appears in a label: the one-layer
    band ``[3]`` is ``3`` — the scalar case is the scalar spelling, so a
    swept ``sites.target.layers`` labels ``[target.layers=3]`` exactly as the
    index does — a contiguous band ``[10, …, 19]`` is ``10..19``, and any
    other band joins its members with ``+`` (``10+12+15``). Neither the comma
    nor the brackets of the ``[k=8,seed=0]`` syntax appear, so a band label
    parses back like any other coordinate value."""
    members = [str(v) for v in layers]
    if len(members) == 1:
        return members[0]
    if all(isinstance(v, int) for v in layers) and list(layers) == list(
        range(layers[0], layers[0] + len(layers))
    ):
        return f"{layers[0]}..{layers[-1]}"
    return "+".join(members)


# --------------------------------------------------------------------------- #
# what the protocol layer decides from the axes alone
# --------------------------------------------------------------------------- #


def point_count(axes: Sequence[Axis]) -> int:
    """How many steps a campaign enumerates to — the product of the axis
    sizes — decided from the axes alone, so rule 14's cap is decided
    protocol-side without enumerating anything. ``axes`` is
    [`axes_of`][]'s: for a document with named axes the independent named
    axes are members already, one value per entry, beside the inner sweep
    axes, so the product is rows × the inner cross product and never the
    display form's."""
    total = 1
    for axis in axes:
        total *= len(axis.values)
    return total


def axes_of(tree: Mapping[str, Any], named: Axes | None) -> tuple[Axis, ...]:
    """Every axis a campaign's steps are indexed by, in the order enumeration
    walks them ([`causalab.neural.shared.sweep`][]): the document's sweep
    axes in first-appearance order, or — when it declares named axes (§3.2)
    — one ``axes.<name>`` axis per independent named axis (its entries'
    keys, first declared slowest) followed by the ordinary sweep axes the
    first row's concrete tree still carries. Exactly the ``axes`` the
    expansion reported before enumeration moved engine-side."""
    if named is None:
        return find_axes(tree)
    first = row_tree(named, next(entries(named)))
    inner = find_axes(first)
    slowest = tuple(
        Axis(path=(AXES_KEY, axis.name), values=axis.keys) for axis in named.independent
    )
    return (*slowest, *inner)


def representative_trees(
    tree: Mapping[str, Any], axes: Sequence[Axis], named: Axes | None
) -> tuple[Mapping[str, Any], ...]:
    """The §5 checklist's domain: one concrete step tree per axis value —
    that axis at the value, every other axis at its first value — so every
    value of every axis is held to the rules exactly once (the spec's "over
    the axis domain, each axis value once") and nothing is enumerated. The
    first tree is the campaign's first step (every axis at its first value),
    counted once; a document with no axes is its own one representative.
    Each tree is a step the enumeration also produces — built by the same
    [`substitute`][] and [`row_tree`][] — so a rule a representative
    breaks is broken at a real step.

    The trees come out **in enumeration order** — the position each holds
    in the cross product (last axis fastest, [`causalab.neural.shared.sweep`][]), so distinct violations aggregated over them read in the order
    a per-point checklist gives them: a refusal's text is
    byte-identical whichever way the document was checked."""
    if not axes:
        return (tree,)
    slowest = () if named is None else named.independent
    inner = axes[len(slowest) :]

    def build(at: Mapping[tuple[str, ...], int]) -> Mapping[str, Any]:
        if named is None:
            base: Mapping[str, Any] = tree
        else:
            base = row_tree(named, tuple(at[(AXES_KEY, axis.name)] for axis in slowest))
        return substitute(
            base, {axis.path: axis.values[at[axis.path]] for axis in inner}, ()
        )

    # the stride of each axis in the cross product: the product of the sizes
    # of the axes after it (the last axis is the fastest)
    strides: dict[tuple[str, ...], int] = {}
    stride = 1
    for axis in reversed(axes):
        strides[axis.path] = stride
        stride *= len(axis.values)
    first = {axis.path: 0 for axis in axes}
    placed = [(0, build(first))]
    for axis in axes:
        for index in range(1, len(axis.values)):
            placed.append(
                (index * strides[axis.path], build({**first, axis.path: index}))
            )
    placed.sort(key=lambda item: item[0])
    return tuple(tree for _, tree in placed)


# --------------------------------------------------------------------------- #
# families (§3.1) — the body of families.py, byte for byte
# --------------------------------------------------------------------------- #


#: The optional member-naming template that sits beside it.
NAMES_KEY = "names"

#: The rule every refusal here carries (§5).
RULE = "family_wrappers"

#: Tables whose entries may declare or inherit a family. `intervened_models` is
#: handled separately — it *selects* from a write family rather than joining its
#: axis — and `metrics` is absent for the reason the module docstring gives.
FAMILY_SECTIONS: tuple[str, ...] = (
    "positions",
    "sites",
    "featurizers",
    "params",
    "reads",
    "writes",
)

#: Tables a family may not reach in this version, each for the reason the
#: module docstring gives: what a per-member ``file_path`` should be.
CONSUMER_SECTIONS: tuple[str, ...] = ("save", "train")

#: Fields that hold the *name of another entry*. Fan-out rewrites only these,
#: rather than every string equal to a family name, because several
#: aggregation fields (``a``, ``b``, ``token``, ``target``) name **dataset
#: columns** and not document entities (§2.10) — a column called ``a`` must
#: not be captured by a family called ``a``.
NAME_FIELDS: frozenset[str] = frozenset({"site", "pos", "featurizer"})

#: Operand slots inside a ``do`` block (§2.8): a read name, a param name or a
#: literal. ``b`` here is ``affine``'s bias param, unrelated to ``metrics.b``.
DO_OPERAND_FIELDS: frozenset[str] = frozenset({"swap", "op", "A", "b"})

#: The most members one family may denote. `sweep`'s per-axis bound is a
#: million because [`DEFAULT_POINT_CAP`][] refuses
#: the expansion afterwards; a family materializes entries *directly*, with no
#: later cap to catch it, so it needs its own — and a table of a thousand
#: addresses in one forward is past the point where the answer is a sweep.
MAX_FAMILY_MEMBERS = 1024


def has_families(raw: Mapping[str, Any]) -> bool:
    """Whether anything in ``raw`` — a document, or its ``method`` group alone
    — declares a family.

    [`expand_families`][] is a no-op on a document without one, and every
    document written before §3.1 is such a document — so this is the cheap
    guard that keeps the new stage off the existing path entirely.
    """
    return _wrapper_path(_method_of(raw)) is not None


def _method_of(raw: Mapping[str, Any]) -> Mapping[str, Any]:
    """The tree families live in (§1, §3.1): a document's ``method`` group,
    or ``raw`` itself when it already *is* that group (no group key at all).
    A ``method`` that is not an object is left to the shape gate to refuse."""
    if "method" in raw and isinstance(raw["method"], Mapping):
        return raw["method"]
    return raw


def _wrapper_path(raw: Mapping[str, Any]) -> str | None:
    """The path of the first ``at_once`` wrapper in ``raw``, or ``None``.

    A **table key** spelled ``at_once`` is skipped: §1 reserves four names and
    this is not one of them, so an entry may legitimately be called that, and
    reading its name as a wrapper would refuse a valid document with a message
    about something the author never wrote.
    """
    for section, value in raw.items():
        if section in NAMED_SECTIONS and isinstance(value, Mapping):
            for name, entry in value.items():
                found = _find_key(entry, AT_ONCE_KEY, f"{section}.{name}")
                if found is not None:
                    return found
        else:
            found = _find_key(value, AT_ONCE_KEY, str(section))
            if found is not None:
                return found
    return None


def _find_key(node: Any, key: str, path: str = "") -> str | None:
    """The path of the first occurrence of ``key`` anywhere in ``node``."""
    if isinstance(node, Mapping):
        for name, value in node.items():
            here = f"{path}.{name}" if path else str(name)
            if name == key:
                return path or here
            found = _find_key(value, key, here)
            if found is not None:
                return found
    elif isinstance(node, list):
        for index, item in enumerate(node):
            found = _find_key(item, key, f"{path}[{index}]")
            if found is not None:
                return found
    return None


class _Family:
    """One declared or inherited family: its axis, and how members are named."""

    __slots__ = ("field", "name", "names", "section", "values")

    def __init__(
        self,
        section: str,
        name: str,
        field: str,
        values: tuple[Any, ...],
        names: str | None,
    ) -> None:
        self.section = section
        self.name = name
        self.field = field
        self.values = values
        self.names = names

    @property
    def axis(self) -> tuple[str, tuple[Any, ...]]:
        """Axis identity — the field name plus the values.

        Two families over the same field and the same values are **one** axis
        and align member for member; the same field over different values is a
        different axis, and one entry referencing both is refused, because
        which pairs are meant is exactly what alignment cannot guess.
        """
        # a member value may itself be a band (a list, on `layers`); the axis
        # is compared by value, so it is spelled hashably
        return (self.field, tuple(_label(v) for v in self.values))

    def member(self, value: Any) -> str:
        if self.names is not None:
            return self.names.replace("{" + self.field + "}", _label(value))
        return f"{self.name}[{self.field}={_label(value)}]"

    def members(self) -> tuple[tuple[str, Any], ...]:
        return tuple((self.member(value), value) for value in self.values)


def _label(value: Any) -> str:
    """One axis value as it appears in a member's name — the same rendering
    sweeps use for a coordinate ([`label_value`][]), so a family member and
    a swept name are spelled alike."""
    return label_value(value)


def _refuse_family(message: str, path: str) -> ValidationError:
    return ValidationError(RULE, message, path=path)


def _declared_axis(section: str, name: str, entry: Mapping[str, Any]) -> str | None:
    """The field this entry wraps in ``at_once``, if any."""
    wrapped = [key for key, value in entry.items() if _is_wrapper(value, AT_ONCE_KEY)]
    if not wrapped:
        return None
    if len(wrapped) > 1:
        raise _refuse_family(
            f"{len(wrapped)} fields carry an at_once wrapper "
            f"({', '.join(sorted(wrapped))}) — one axis per entry, so that a "
            "member has one index and one name",
            f"{section}.{name}",
        )
    return wrapped[0]


def _is_wrapper(node: Any, key: str) -> bool:
    return isinstance(node, Mapping) and key in node


def _template(
    entry: Mapping[str, Any], field: str, section: str, name: str
) -> str | None:
    template = entry.get(NAMES_KEY)
    if template is None:
        return None
    placeholder = "{" + field + "}"
    if not isinstance(template, str) or placeholder not in template:
        raise _refuse_family(
            f"a names template names exactly the axis field, {placeholder!r} — "
            f"got {template!r}. Without it two members would share a name",
            f"{section}.{name}.{NAMES_KEY}",
        )
    return template


def _is_name_field(key: Any, *, in_do: bool) -> bool:
    return key in NAME_FIELDS or (in_do and key in DO_OPERAND_FIELDS)


def _referenced(node: Any, *, in_do: bool = False) -> Iterable[str]:
    """Every entry name a subtree references, by [`NAME_FIELDS`][].

    A name field may hold a **list** of names — a `featurizer` composition is
    written either way (§2.5) — so both spellings of one reference are read
    here. Whether a family works must not depend on which the author picked.
    """
    if isinstance(node, Mapping):
        for key, value in node.items():
            if _is_name_field(key, in_do=in_do):
                if isinstance(value, str):
                    yield value
                elif isinstance(value, list):
                    yield from (item for item in value if isinstance(item, str))
            elif isinstance(value, (Mapping, list)):
                yield from _referenced(value, in_do=in_do or key == "do")
    elif isinstance(node, list):
        for item in node:
            yield from _referenced(item, in_do=in_do)


def _find_families(raw: Mapping[str, Any]) -> dict[str, _Family]:
    """Declared families, then inherited ones to a fixpoint."""
    families: dict[str, _Family] = {}

    for section in FAMILY_SECTIONS:
        table = raw.get(section)
        if not isinstance(table, Mapping):
            continue
        for name, entry in table.items():
            if not isinstance(entry, Mapping):
                continue
            field = _declared_axis(section, name, entry)
            if field is None:
                continue
            wrapper = entry[field]
            if len(wrapper) != 1:
                raise _refuse_family(
                    "an at_once wrapper holds nothing but the axis",
                    f"{section}.{name}.{field}",
                )
            values = axis_values(
                wrapper[AT_ONCE_KEY],
                path=f"{section}.{name}.{field}",
                rule=RULE,
                keyword=AT_ONCE_KEY,
                article="an",
            )
            _check_axis(values, f"{section}.{name}.{field}")
            families[name] = _Family(
                section,
                name,
                field,
                values,
                _template(entry, field, section, name),
            )

    # inheritance is a fixpoint over the reference graph, which §2.9 makes a
    # DAG — so it terminates on its own, and a bound would only ever
    # under-expand a chain deeper than whatever number was picked
    grew = True
    while grew:
        grew = False
        for section in FAMILY_SECTIONS:
            table = raw.get(section)
            if not isinstance(table, Mapping):
                continue
            for name, entry in table.items():
                if name in families or not isinstance(entry, Mapping):
                    continue
                donors = [
                    families[ref] for ref in _referenced(entry) if ref in families
                ]
                if not donors:
                    continue
                axes = {donor.axis for donor in donors}
                if len(axes) > 1:
                    fields = sorted({field for field, _ in axes})
                    raise _refuse_family(
                        f"this entry references families on {len(axes)} different "
                        f"axes ({', '.join(fields)}) — a cross product inside one "
                        "forward is not what a family means. Sweep the second axis "
                        "(§3), which is what a cross product is for",
                        f"{section}.{name}",
                    )
                donor = donors[0]
                families[name] = _Family(
                    section,
                    name,
                    donor.field,
                    donor.values,
                    _template(entry, donor.field, section, name),
                )
                grew = True
        if not grew:
            break
    return families


def _check_one_axis_per_entry(
    raw: Mapping[str, Any], families: Mapping[str, _Family]
) -> None:
    """Every family-capable entry sits on exactly one axis.

    The inheritance fixpoint checks the entries it *derives* an axis for, but
    an entry that declares its own was skipped — and there the mismatch is
    quieter, not louder: `_member_entry` substitutes a reference by
    *value*, so a donor family that happens to carry the same values yields the
    diagonal of a cross product silently, and one that does not yields a name
    nothing declares for rule 4 to report as a dangling reference. Neither is
    the experiment anyone wrote.
    """
    for section in FAMILY_SECTIONS:
        table = raw.get(section)
        if not isinstance(table, Mapping):
            continue
        for name, entry in table.items():
            own = families.get(name)
            if own is None or not isinstance(entry, Mapping):
                continue
            foreign = {
                families[ref].axis
                for ref in _referenced(entry)
                if ref in families and families[ref].axis != own.axis
            }
            if foreign:
                fields = sorted({field for field, _ in foreign} | {own.field})
                raise _refuse_family(
                    f"this entry is on the {own.field!r} axis and references a "
                    f"family on another ({', '.join(fields)}) — a cross product "
                    "inside one forward is not what a family means. Sweep the "
                    "second axis (§3), which is what a cross product is for",
                    f"{section}.{name}",
                )


def _check_axis(values: tuple[Any, ...], path: str) -> None:
    """The bound first, then distinctness.

    In that order on purpose: [`axis_values`][] caps
    the ``{"range": …}`` form and nothing caps an explicit list, so checking
    distinctness first meant a 200k-element list was counted before anything
    mentioned the bound.
    """
    if len(values) > MAX_FAMILY_MEMBERS:
        raise _refuse_family(
            f"this axis denotes {len(values)} members, over the bound of "
            f"{MAX_FAMILY_MEMBERS}. A family materializes entries directly, so "
            "nothing downstream caps it the way the point cap caps a sweep — and "
            "a table of this many addresses in one forward is the point where "
            "the answer is a sweep (§3.1)",
            path,
        )
    counts = Counter(_label(value) for value in values)
    duplicated = sorted(label for label, n in counts.items() if n > 1)
    if duplicated:
        raise _refuse_family(
            f"at_once values are a member index, so they must be distinct — "
            f"{', '.join(duplicated)} appears twice",
            path,
        )


def _member_entry(
    node: Any,
    families: Mapping[str, _Family],
    value: Any,
    *,
    axis: str | None = None,
    in_do: bool = False,
    top: bool = True,
) -> Any:
    """One member's copy of an authored entry.

    ``axis`` — the entry's own declared field, and *only* at the entry's top
    level — becomes this member's index value; every family reference becomes
    the member at the same index. A wrapper anywhere else is deliberately left
    in place: it has no name identity, and substituting it here would silently
    give it this member's index instead (the bug the orphan check now catches).
    """
    if isinstance(node, Mapping):
        out: dict[str, Any] = {}
        for key, child in node.items():
            if top and key == NAMES_KEY:
                continue  # the template is authoring metadata; the member is named
            if top and key == axis and _is_wrapper(child, AT_ONCE_KEY):
                out[key] = _member_value(key, value)
            elif _is_name_field(key, in_do=in_do) and isinstance(child, str):
                out[key] = families[child].member(value) if child in families else child
            elif _is_name_field(key, in_do=in_do) and isinstance(child, list):
                out[key] = [
                    families[item].member(value)
                    if isinstance(item, str) and item in families
                    else item
                    for item in child
                ]
            else:
                out[key] = _member_entry(
                    child,
                    families,
                    value,
                    in_do=in_do or key == "do",
                    top=False,
                )
        return out
    if isinstance(node, list):
        return [
            _member_entry(item, families, value, in_do=in_do, top=False)
            for item in node
        ]
    return node


def _member_value(field: str, value: Any) -> Any:
    """The value a member carries in its axis field. On ``layers`` (§2.4) an
    index denotes the one-layer band ``[n]``, so the expanded tree *is* the
    hand-written one entry for entry — the parser and the canonical form make
    the same fold, but the expansion is what an author reads back."""
    if field == "layers" and isinstance(value, int) and not isinstance(value, bool):
        return [value]
    return value


def _window(spec: Any, path: str) -> tuple[Any, ...]:
    """The values a write-list selector picks out of a family's index.

    Spelled ``{"at_once": [...]}`` or ``{"at_once": {"range": [a, b, step?]}}``
    — the wrapper that declared the family, carrying the payload grammar
    [`axis_values`][] already checks. A first draft spelled an interval
    ``{"span": [a, b]}`` after the position window (§2.3) and let a bare list
    through beside it: two dialects of one selection, the interval the weaker
    (no step), and the bare form's refusals talking about a keyword the author
    never wrote. One word per object (§11.1). The empty window — ``{"range":
    [15, 10]}`` names no write, so the band would compile as the un-intervened
    model and report no effect — is the empty axis, which the shared grammar
    already refuses.
    """
    if isinstance(spec, Mapping) and AT_ONCE_KEY in spec and len(spec) > 1:
        extra = sorted(set(spec) - {AT_ONCE_KEY})
        raise _refuse_family(
            f"a window holds nothing but its {AT_ONCE_KEY} axis — got "
            f"{', '.join(extra)} too",
            path,
        )
    if not (isinstance(spec, Mapping) and AT_ONCE_KEY in spec):
        raise _refuse_family(
            "a window is spelled like the axis it selects from: "
            f'{{"{AT_ONCE_KEY}": [...]}} or '
            f'{{"{AT_ONCE_KEY}": {{"range": [start, stop, step?]}}}}',
            path,
        )
    return axis_values(
        spec[AT_ONCE_KEY], path=path, rule=RULE, keyword=AT_ONCE_KEY, article="an"
    )


def _selected_writes(
    name: str,
    selector: Mapping[str, Any],
    families: Mapping[str, _Family],
    path: str,
) -> list[str]:
    if name not in families:
        raise _refuse_family(
            f"{name!r} is not a family, so there is nothing to select from — "
            "a write list names a write, or windows a family",
            path,
        )
    family = families[name]
    fields = sorted(selector)
    if fields != [family.field]:
        raise _refuse_family(
            f"family {name!r} is indexed by {family.field!r}, so a window is "
            f"declared on {family.field!r} — got {', '.join(fields) or 'nothing'}",
            path,
        )
    index = {_label(value): member for member, value in family.members()}
    wanted = _window(selector[family.field], f"{path}.{family.field}")
    missing = [_label(value) for value in wanted if _label(value) not in index]
    if missing:
        raise _refuse_family(
            f"the window names {family.field}={', '.join(missing)}, which family "
            f"{name!r} does not carry (it runs {_label(family.values[0])}.."
            f"{_label(family.values[-1])}) — a band that resolves to fewer writes "
            "than the interval it declares is the silent bug this form exists to "
            "refuse",
            path,
        )
    return [index[_label(value)] for value in wanted]


def _read_list(
    model: str, entry: Mapping[str, Any], families: Mapping[str, _Family]
) -> Any:
    """One model's ``reads`` (§2.9), with families resolved to member names: a
    bare family name is every member, and follows the family if it grows.
    Anything but a list of names is the parser's to refuse."""
    listed = entry.get("reads")
    if not isinstance(listed, list):
        return listed
    reads: list[Any] = []
    for item in listed:
        if isinstance(item, str) and item in families:
            reads.extend(member for member, _ in families[item].members())
        else:
            reads.append(item)
    return reads


def _write_list(
    model: str, entry: Mapping[str, Any], families: Mapping[str, _Family]
) -> list[Any]:
    """One intervened model's `writes`, with families resolved to member names.

    A bare family name is every member, and follows the family if it grows; a
    window is the explicit form, and refuses to resolve to a different number
    of writes than the interval it names.
    """
    writes: list[Any] = []
    listed = entry.get("writes")
    if not isinstance(listed, list):
        # A whole write list may be swept (`_parse_im` wraps it), and that is
        # legal and untouched — unless a family name is inside it, where the
        # member it means depends on the point and this stage has none. Refused
        # by name, like every other shape §3.1 leaves for later.
        held = next((name for name in _mentions(listed) if name in families), None)
        if held is not None:
            raise _refuse_family(
                f"this write list is swept and names the family {held!r}. A "
                "family materializes before axes are found (§3.1), so a member "
                "chosen per point is not something this stage can resolve — "
                "sweep something other than the write list, or name members",
                f"intervened_models.{model}.writes",
            )
        # otherwise absent or the wrong type: the parser's rejection to make,
        # not ours — and injecting `writes: None` would take away its message
        return listed
    for index, item in enumerate(listed):
        path = f"intervened_models.{model}.writes[{index}]"
        if isinstance(item, str):
            if item in families:
                writes.extend(member for member, _ in families[item].members())
            else:
                writes.append(item)
        elif isinstance(item, Mapping):
            if len(item) != 1:
                raise _refuse_family(
                    "a write-list entry names one family, "
                    '{"<family>": {"<field>": {"at_once": ...}}}',
                    path,
                )
            name, selector = next(iter(item.items()))
            if not isinstance(selector, Mapping):
                raise _refuse_family(
                    f"the selector for {name!r} is an object on the family's "
                    'index, e.g. {"layers": {"at_once": {"range": [10, 15]}}}',
                    path,
                )
            writes.extend(_selected_writes(name, selector, families, path))
        else:
            writes.append(item)
    counts = Counter(w for w in writes if isinstance(w, str))
    duplicated = sorted(name for name, n in counts.items() if n > 1)
    if duplicated:
        raise _refuse_family(
            f"{', '.join(duplicated)} listed twice — overlapping selectors, not "
            "overlapping bands. Bands overlap by sharing writes between models, "
            "never by naming one twice in one model",
            f"intervened_models.{model}.writes",
        )
    return writes


def expand_families(raw: Mapping[str, Any]) -> dict[str, Any]:
    """``raw`` with every ``at_once`` family materialized (§3.1).

    A pure tree edit, and a no-op on any document that declares no family — so
    the compiled intervention, and therefore the digest, is exactly what the
    same experiment written out by hand produces. Idempotent: the result
    carries no ``at_once``, so expanding twice changes nothing.

    Families live in the ``method`` group (§1): a wrapper declares one field of
    one named method entry, and every refusal path is section-rooted, the way
    every other path in the protocol is spelled. Handed a document, this
    expands its ``method`` and returns the document; handed the group alone,
    it returns the expanded group. The header, ``model`` and ``data`` never
    see a wrapper.
    """
    if "method" in raw and isinstance(raw["method"], Mapping):
        return {**raw, "method": _expand_method(raw["method"])}
    return _expand_method(raw)


def _expand_method(raw: Mapping[str, Any]) -> dict[str, Any]:
    if not has_families(raw):
        _check_orphan_names(raw, {})
        return dict(raw)

    families = _find_families(raw)
    _check_orphan_names(raw, families)
    _check_one_axis_per_entry(raw, families)
    _check_no_sweep_on_a_family(raw, families)
    declared = _declared_names(raw)

    out: dict[str, Any] = {}
    # the family tables first, whatever order the document spells: a member
    # name is checked against the declarations as its table expands, and a
    # model's windows are resolved against the families those tables carry
    ordered = [
        *((s, v) for s, v in raw.items() if s != "intervened_models"),
        *((s, v) for s, v in raw.items() if s == "intervened_models"),
    ]
    for section, value in ordered:
        if section in FAMILY_SECTIONS and isinstance(value, Mapping):
            table: dict[str, Any] = {}
            for name, entry in value.items():
                family = families.get(name)
                if family is None:
                    table[name] = entry
                    continue
                for member, index_value in family.members():
                    _check_member_name(member, name, section, table, declared)
                    table[member] = _member_entry(
                        entry, families, index_value, axis=family.field
                    )
            out[section] = table
        elif section == "intervened_models" and isinstance(value, Mapping):
            models: dict[str, Any] = {}
            for name, entry in value.items():
                if (
                    name in families
                    or _model_axis(entry, f"intervened_models.{name}") is not None
                ):
                    raise _refuse_family(
                        "a family of intervened models is not in this version: "
                        "one model per band, and its metric and save entry with "
                        "it, are what a saved family still needs a `file_path` "
                        "rule for (§3.1)",
                        f"intervened_models.{name}",
                    )
                if not isinstance(entry, Mapping):
                    models[name] = entry  # the parser's message to give, not ours
                    continue
                expanded = dict(entry)
                if "reads" in entry:
                    expanded["reads"] = _read_list(name, entry, families)
                if "writes" in entry:
                    expanded["writes"] = _write_list(name, entry, families)
                models[name] = expanded
            out[section] = models
        else:
            _check_no_consumer_reference(section, value, families)
            out[section] = value

    out = {section: out[section] for section in raw if section in out}
    orphan = _wrapper_path(out)
    if orphan is not None:
        raise _refuse_family(
            "an at_once wrapper sits where it has no name identity: it declares "
            "one field of one entry of "
            f"{', '.join(FAMILY_SECTIONS)}, and nowhere else (§3.1)",
            orphan,
        )
    return out


def _declared_names(raw: Mapping[str, Any]) -> set[str]:
    names: set[str] = set()
    for section in NAMED_SECTIONS:
        table = raw.get(section)
        if isinstance(table, Mapping):
            names.update(str(name) for name in table)
    return names


def _check_member_name(
    member: str,
    family: str,
    section: str,
    table: Mapping[str, Any],
    declared: set[str],
) -> None:
    where = f"{section}.{family}.{NAMES_KEY}"
    if member in table:
        raise _refuse_family(
            f"two members are both named {member!r} — check the names template",
            where,
        )
    if member in RESERVED_NAMES:
        raise _refuse_family(
            f"a member is named {member!r}, which is reserved (§1)",
            where,
        )
    if member in declared and member != family:
        raise _refuse_family(
            f"a member is named {member!r}, which the document already declares "
            "— one name, one entry (§1)",
            where,
        )


def _check_orphan_names(
    raw: Mapping[str, Any], families: Mapping[str, _Family]
) -> None:
    """``names`` with no axis to name is a typo worth catching by name: the
    parser would report it as an unknown key on whichever table it sits in.

    Checked against the discovered families rather than against the entry's own
    wrapper, because ``names`` is equally legal on an entry that *inherits* its
    axis by reference — which is most of them. And run on every document, not
    only family-free ones: the typo is likeliest in a document that has
    families elsewhere, which is exactly where the first version of this check
    did not look.
    """
    for section in FAMILY_SECTIONS:
        table = raw.get(section)
        if not isinstance(table, Mapping):
            continue
        for name, entry in table.items():
            if (
                isinstance(entry, Mapping)
                and NAMES_KEY in entry
                and name not in families
            ):
                raise _refuse_family(
                    f"{NAMES_KEY!r} names the members of a family, but this entry "
                    f"declares no {AT_ONCE_KEY!r} axis (§3.1)",
                    f"{section}.{name}.{NAMES_KEY}",
                )


def _model_axis(entry: Any, path: str) -> str | None:
    """The path of an ``at_once`` wrapper on an intervened model *itself* — a
    family of models, which this version refuses by name — as distinct from a
    **window** inside its write list, which `_write_list` resolves.

    The two are told apart by position, not by keyword: a window is a one-key
    object *keyed by a family name* inside a list-shaped write list,
    ``writes[i].<family>.<field>``, and ``_write_list`` consumes every such
    object and refuses anything that is not exactly that shape. A wrapper
    anywhere else in the entry — on ``input``, wrapping a write item itself —
    is an axis over models. A swept ``writes`` is left to ``_write_list`` too:
    its refusal names the sweep, which is what the author wrote.
    """
    if not isinstance(entry, Mapping):
        return _find_key(entry, AT_ONCE_KEY, path)
    for key, value in entry.items():
        if key == AT_ONCE_KEY:
            return path
        if key == "writes":
            if not isinstance(value, list):
                continue
            for index, item in enumerate(value):
                here = f"{path}.writes[{index}]"
                if isinstance(item, Mapping):
                    if AT_ONCE_KEY in item:
                        return here
                    continue
                found = _find_key(item, AT_ONCE_KEY, here)
                if found is not None:
                    return found
            continue
        found = _find_key(value, AT_ONCE_KEY, f"{path}.{key}")
        if found is not None:
            return found
    return None


def _mentions(node: Any) -> Iterable[str]:
    """Every string anywhere in a subtree, mapping keys included — for the
    check that asks whether a family is *named*, which a window does by key:
    ``{"w": {"layers": {"at_once": ...}}}`` mentions ``w``."""
    yield from _flat_strings(node)
    if isinstance(node, Mapping):
        for key, value in node.items():
            yield str(key)
            yield from _mentions(value)
    elif isinstance(node, list):
        for item in node:
            yield from _mentions(item)


def _flat_strings(node: Any) -> Iterable[str]:
    """Every string anywhere in a subtree, for the checks that only ask whether
    a name is *mentioned*."""
    if isinstance(node, str):
        yield node
    elif isinstance(node, Mapping):
        for value in node.values():
            yield from _flat_strings(value)
    elif isinstance(node, list):
        for item in node:
            yield from _flat_strings(item)


def _extra_names(entry: Any) -> Iterable[str]:
    """Name-bearing fields outside [`NAME_FIELDS`][], in the tables that only
    consume: an entry's ``read``, a featurizer entry's ``value`` and a ``kl``
    / ``js`` ``target`` (a read reference in either spelling)."""
    if not isinstance(entry, Mapping):
        return
    for field in ("read", "value", "target"):
        held = entry.get(field)
        if isinstance(held, str):
            yield held
    for nested in entry.values():
        if isinstance(nested, Mapping):
            yield from _extra_names(nested)
        elif isinstance(nested, list):
            for item in nested:
                yield from _extra_names(item)


def _train_names(block: Any) -> Iterable[str]:
    """Every entry a ``train`` block references, by the **owning entry's** name
    (§2.11).

    Four spellings, and each was its own small trap:

    * ``params`` is a plain name list — but its members are param *slots*, so
      ``"rot.weight"`` names the entry ``rot``. Compared on the first dotted
      segment, which is what `validate._check_train_references` does;
    * a **regularizer is keyed by its kind** — ``{"l1": ["rot"]}`` — which is
      why scanning for a ``names`` key found nothing: §2.11 never emits one;
    * that mapping's value may be a bare string rather than a list
      (``_parse_regularizer_names`` accepts both);
    * and in the positional objective form the whole thing sits inside a
      ``[weight, …]`` list inside the objective list, so the walk has to
      descend lists as well as mappings.

    ``anneal`` is the fourth: it is a mapping *keyed* by the slot it anneals,
    so its names are in its keys and nowhere else.
    """
    if not isinstance(block, Mapping):
        return
    params = block.get("params")
    if isinstance(params, list):
        yield from (_owner(item) for item in params if isinstance(item, str))
    anneal = block.get("anneal")
    if isinstance(anneal, Mapping):
        # a `train.objective.<name>.weight` key anneals a term, not a slot —
        # its first segment names no entry (§2.11)
        yield from (
            _owner(key)
            for key in anneal
            if isinstance(key, str) and not key.startswith(OBJECTIVE_WEIGHT_PREFIX)
        )
    phases = block.get("phases")
    if isinstance(phases, list):
        # a phase narrows `params` and may anneal — the same two spellings
        for phase in phases:
            if isinstance(phase, Mapping):
                yield from _train_names(
                    {k: v for k, v in phase.items() if k != "optimizer"}
                )
                freeze = phase.get("freeze_masks")
                if isinstance(freeze, list):
                    yield from (name for name in freeze if isinstance(name, str))
    yield from _regularizer_names(block.get("objective"))


def _owner(slot: str) -> str:
    """The entry a param slot belongs to: ``rot.weight`` is ``rot``'s."""
    return slot.split(".", 1)[0]


def _regularizer_names(node: Any) -> Iterable[str]:
    if isinstance(node, Mapping):
        for key, value in node.items():
            if key in REGULARIZER_KINDS:
                if isinstance(value, str):
                    yield _owner(value)
                elif isinstance(value, list):
                    yield from (_owner(item) for item in value if isinstance(item, str))
            else:
                yield from _regularizer_names(value)
    elif isinstance(node, list):
        for item in node:
            yield from _regularizer_names(item)


def _check_no_sweep_on_a_family(
    raw: Mapping[str, Any], families: Mapping[str, _Family]
) -> None:
    """A family entry may not also carry a sweep axis.

    Expansion copies the entry once per member, so the sweep wrapper would be
    copied too — and a sweep axis is identified by its *path* (§3), so N copies
    on N paths are N independent axes whose cross product is exponential rather
    than the one axis the author meant.
    """
    for section in FAMILY_SECTIONS:
        table = raw.get(section)
        if not isinstance(table, Mapping):
            continue
        for name, entry in table.items():
            if name not in families or not isinstance(entry, Mapping):
                continue
            # anywhere in the entry, not only on its own fields: a `sweep`
            # inside a `do` block is copied to every member exactly the same
            # way, and ten members of the shipped preset is 2**10 points —
            # under the point cap, so nothing downstream would have caught it
            swept = _find_key(entry, SWEEP_KEY, f"{section}.{name}")
            if swept is not None:
                raise _refuse_family(
                    f"this entry declares an at_once family and also sweeps "
                    f"{swept} — expansion would copy the sweep wrapper onto all "
                    f"{len(families[name].values)} members, and a sweep axis is "
                    "its path (§3), so that is one axis per member and a cross "
                    "product of them. Sweep an entry off the family instead",
                    f"{section}.{name}",
                )


def _check_no_consumer_reference(
    section: str, value: Any, families: Mapping[str, _Family]
) -> None:
    """`save` and `train` do not fan out in this version, so a
    family reaching any of them is refused by name rather than silently kept as
    a dangling reference for rule 4 to report as a missing entry.

    `train` is here for the same reason as the other two and one more: it
    references entries by *param slot* — ``params``, a regularizer's names, and
    ``anneal``'s keys — so a fit over a family is the shape most likely to look
    as though it had worked. Its names are collected by `_train_names`,
    which knows the four spellings §2.11 actually uses.
    """
    if section not in CONSUMER_SECTIONS or not isinstance(value, (Mapping, list)):
        return
    # `metrics` is a table of entries and `save` a list of them, so each can be
    # named in the path; `train` is one block, not a table, and iterating *its*
    # items would look inside `params` one element at a time and see no names
    entries: Any = (
        [(None, value)]
        if section == "train"
        else value.items()
        if isinstance(value, Mapping)
        else enumerate(value)
    )
    for key, entry in entries:
        named = list(_referenced(entry))
        named.extend(_train_names(entry) if section == "train" else _extra_names(entry))
        hit = next((name for name in named if name in families), None)
        if hit is not None:
            raise _refuse_family(
                f"this references the family {hit!r}. `save` and "
                "`train` "
                "do not fan out in this version: a saved family needs a rule for "
                "per-member `file_path`, and §2.12's answer for swept documents — "
                "coordinates become columns of one table — is the better one, so "
                "it is its own change. Name one member, or sweep the axis (§3)",
                section if key is None else f"{section}.{key}",
            )


# --------------------------------------------------------------------------- #
# named axes (§3.2) — the body of axes.py without expand_axes, byte for byte
# --------------------------------------------------------------------------- #


#: The optional fifth top-level group (§1, §3.2). Not in ``GROUP_ORDER``: the
#: stage strips it before the gate, which is what keeps ``schema.py`` untouched.
AXES_KEY = "axes"

#: The wrapper that references a named axis at a field — ``{"axis": "center"}``
#: or ``{"axis": "location.layers"}`` — legal wherever ``{"sweep": …}`` is.
AXIS_KEY = "axis"

#: The key that decides what kind of axis a declaration is — exactly one of
#: these per declaration. Closed; spec §3.2's ``key`` table is the census.
AXIS_KINDS: tuple[str, ...] = ("rows", "range", "values", "dependent_on")

#: The rules a dependent axis may be computed by. Closed; spec §3.2's ``kind``
#: table is the census. ``clipped_band`` is ROME's window: a ``width``-wide band
#: around the parent's value, clipped to the model's layer count.
RULE_KINDS: tuple[str, ...] = ("clipped_band",)

#: What a ``clipped_band`` may clip to. ``layers`` reads its bound from the
#: registry entry of the document's one ``model.key`` (``ModelInfo.num_layers``),
#: exactly as ``canonicalize`` bounds a site's layers.
CLIP_TARGETS: tuple[str, ...] = ("layers",)

#: Every key a declaration of each kind may carry.
_DECLARATION_KEYS: Mapping[str, tuple[str, ...]] = {
    "rows": ("rows", "key"),
    "range": ("range",),
    "values": ("values",),
    "dependent_on": ("dependent_on", "rule"),
}

_CLIPPED_BAND_KEYS: tuple[str, ...] = ("width", "clip_to")

#: The message the cap shares with ``sweep.expand`` — one wording, so a reader
#: who hit the cap through either path is told the same thing.
CAP_MESSAGE = (
    "sweep expands to {total} points, over the cap of {cap}; "
    "pass an explicit override to run a campaign this large"
)


@dataclasses.dataclass(frozen=True)
class NamedAxis:
    """One declared axis.

    ``keys`` are its coordinate values, one per entry — a ``rows`` axis's
    ``key`` field (or the row index), a ``range`` / ``values`` axis's values;
    empty for a dependent axis, which records nothing (its value is in the
    point, as the parent's coordinate already names it). ``columns`` hold what
    a referencing wrapper receives per entry: keyed by field for a ``rows``
    axis, by ``None`` for the axis itself otherwise. ``parent`` is the axis a
    dependent one follows; ``declared`` is the declaration as authored."""

    name: str
    kind: str
    keys: tuple[Any, ...]
    columns: Mapping[str | None, tuple[Any, ...]]
    parent: str | None
    declared: Mapping[str, Any]

    @property
    def is_dependent(self) -> bool:
        return self.kind == "dependent_on"

    @property
    def size(self) -> int:
        return len(next(iter(self.columns.values())))


@dataclasses.dataclass(frozen=True)
class Axes:
    """The parsed ``axes`` group of one document, with the authored tree it
    was parsed from — the tree with the block and the wrappers still in
    place, which is what [`expand_axes`][causalab.neural.shared.sweep.expand_axes] substitutes into.

    ``references`` are the wrappers, in first-appearance order: the tree path
    of each, the axis it names and the row field, if any."""

    source: Mapping[str, Any]
    named: tuple[NamedAxis, ...]
    references: tuple[tuple[tuple[str, ...], str, str | None], ...]

    @property
    def independent(self) -> tuple[NamedAxis, ...]:
        """The axes that are coordinates — every one that is not dependent —
        in declaration order, slowest first."""
        return tuple(axis for axis in self.named if not axis.is_dependent)

    def __getitem__(self, name: str) -> NamedAxis:
        for axis in self.named:
            if axis.name == name:
                return axis
        raise KeyError(name)


def has_axes(raw: Mapping[str, Any]) -> bool:
    """Whether ``raw`` declares an ``axes`` group — the cheap guard that keeps
    the stage off every document written before §3.2, which is every shipped
    one."""
    return AXES_KEY in raw


def _refuse(code: str, message: str, path: str) -> ParseError:
    return ParseError(code, message, path=path)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_scalar(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool))


def _wrapper_inside(node: Any) -> bool:
    """Whether a ``sweep``, ``at_once`` or ``axis`` wrapper sits anywhere in
    ``node`` — none may: an entry's value is a value."""
    if isinstance(node, Mapping):
        if SWEEP_KEY in node or AT_ONCE_KEY in node or _is_axis_node(node):
            return True
        return any(_wrapper_inside(v) for v in node.values())
    if isinstance(node, list):
        return any(_wrapper_inside(v) for v in node)
    return False


def _fold(field: str, value: Any) -> Any:
    """A row's ``layers`` spelled as an index is the one-layer band ``[n]``
    (§2.4) — the same fold the parser and the canonical form make, applied
    here so the display form's column and the substituted point both carry
    the band spelling, and the point is exactly the hand-written one."""
    if field == "layers" and _is_int(value):
        return [value]
    return value


def parse_axes(raw: Mapping[str, Any], model_info: Callable[[str], ModelInfo]) -> Axes:
    """Parse the ``axes`` group of ``raw`` and find every wrapper that
    references it. ``model_info`` is the registry lookup a ``clipped_band``
    reads its bound from. Every refusal is a [`ParseError`][] on a path
    under ``axes.<name>`` or at the wrapper's own section-rooted path."""
    block = raw[AXES_KEY]
    if not isinstance(block, Mapping) or not block:
        raise _refuse(
            "P2",
            "'axes' is a non-empty object of named axes (§3.2)"
            + ("" if isinstance(block, Mapping) else f", got {_type_name(block)}"),
            AXES_KEY,
        )
    kinds: dict[str, str] = {}
    for name, decl in block.items():
        kinds[str(name)] = _kind_of(str(name), decl)
    named: dict[str, NamedAxis] = {}
    # independent axes first, so a dependent one may name a parent declared
    # after it — declaration order is still the coordinate order
    for name, kind in kinds.items():
        if kind != "dependent_on":
            named[name] = _parse_independent(name, kind, block[name])
    for name, kind in kinds.items():
        if kind == "dependent_on":
            named[name] = _parse_dependent(name, block[name], named, raw, model_info)
    ordered = tuple(named[name] for name in kinds)
    references = _find_references(raw, ordered)
    _check_everything_is_used(ordered, references)
    return Axes(source=raw, named=ordered, references=references)


def _type_name(value: Any) -> str:
    return type(value).__name__


def _kind_of(name: str, decl: Any) -> str:
    path = f"{AXES_KEY}.{name}"
    if not isinstance(decl, Mapping):
        raise _refuse(
            "P2",
            f"axis {name!r} is an object declaration, got {_type_name(decl)}",
            path,
        )
    present = [kind for kind in AXIS_KINDS if kind in decl]
    if len(present) != 1:
        raise _refuse(
            "P2",
            f"axis {name!r} is exactly one of {list(AXIS_KINDS)} — "
            + (
                f"it declares {present}"
                if present
                else f"it declares none (keys: {sorted(str(k) for k in decl)})"
            ),
            path,
        )
    kind = present[0]
    allowed = _DECLARATION_KEYS[kind]
    for key in decl:
        if key not in allowed:
            raise _refuse(
                "P3",
                f"unknown key {key!r} in the {kind} axis {name!r}"
                f"{suggest(str(key), allowed)}",
                f"{path}.{key}",
            )
    return kind


def _parse_independent(name: str, kind: str, decl: Mapping[str, Any]) -> NamedAxis:
    path = f"{AXES_KEY}.{name}"
    if kind == "rows":
        return _parse_rows(name, decl, path)
    if kind == "range":
        values = _range_values(decl["range"], f"{path}.range")
    else:
        values = _list_values(decl["values"], f"{path}.values")
    return NamedAxis(
        name=name,
        kind=kind,
        keys=values,
        columns={None: values},
        parent=None,
        declared=decl,
    )


def _parse_rows(name: str, decl: Mapping[str, Any], path: str) -> NamedAxis:
    rows = decl["rows"]
    if not isinstance(rows, list) or not rows:
        raise _refuse(
            "P2",
            f"'rows' of axis {name!r} is a non-empty list of row objects"
            + ("" if isinstance(rows, list) else f", got {_type_name(rows)}"),
            f"{path}.rows",
        )
    fields: tuple[str, ...] | None = None
    for i, row in enumerate(rows):
        where = f"{path}.rows[{i}]"
        if not isinstance(row, Mapping):
            raise _refuse(
                "P2",
                f"row {i} of axis {name!r} is an object, got {_type_name(row)}",
                where,
            )
        if not row:
            raise _refuse("P2", f"row {i} of axis {name!r} names no field", where)
        these = tuple(str(k) for k in row)
        if fields is None:
            fields = these
        elif set(these) != set(fields):
            missing = sorted(set(fields) - set(these))
            extra = sorted(set(these) - set(fields))
            raise _refuse(
                "P2",
                f"row {i} of axis {name!r} does not cover the fields row 0 "
                f"declares ({list(fields)}): "
                + ", ".join(
                    part
                    for part in (
                        f"missing {missing}" if missing else "",
                        f"extra {extra}" if extra else "",
                    )
                    if part
                ),
                where,
            )
        for field, value in row.items():
            if _wrapper_inside(value):
                raise _refuse(
                    "P2",
                    f"a row value is a value: no sweep, at_once or axis wrapper "
                    f"inside axis {name!r}'s rows (the row *is* the axis)",
                    f"{where}.{field}",
                )
    assert fields is not None
    key = decl.get("key")
    if "key" in decl:
        if not isinstance(key, str) or key not in fields:
            raise _refuse(
                "P2",
                f"'key' of axis {name!r} names one of its row fields "
                f"{list(fields)}"
                + (suggest(key, fields) if isinstance(key, str) else ""),
                f"{path}.key",
            )
        keys = tuple(row[key] for row in rows)
        for i, value in enumerate(keys):
            if not _is_scalar(value):
                raise _refuse(
                    "P2",
                    f"the key field {key!r} of axis {name!r} is a scalar per row — "
                    f"row {i} carries {_type_name(value)}; a coordinate is a scalar",
                    f"{path}.rows[{i}].{key}",
                )
        _check_distinct(keys, name, f"{path}.key", what=f"key field {key!r}")
    else:
        keys = tuple(range(len(rows)))
    columns: dict[str | None, tuple[Any, ...]] = {
        field: tuple(_fold(field, row[field]) for row in rows) for field in fields
    }
    _check_rows_distinct(columns, fields, name, path)
    return NamedAxis(
        name=name, kind="rows", keys=keys, columns=columns, parent=None, declared=decl
    )


def _check_rows_distinct(
    columns: Mapping[str | None, tuple[Any, ...]],
    fields: tuple[str, ...],
    name: str,
    path: str,
) -> None:
    """Rows are distinct, compared folded (so ``layers: 8`` and ``[8]`` are one
    row): two identical rows would enumerate as two points with one name, keyed
    or not — a keyed pair is already caught by its repeated key, a keyless pair
    only here."""
    seen: list[tuple[Any, ...]] = []
    for i in range(len(next(iter(columns.values())))):
        row = tuple(columns[field][i] for field in fields)
        if row in seen:
            raise _refuse(
                "P2",
                f"rows {seen.index(row)} and {i} of axis {name!r} are one row "
                f"({dict(zip(fields, row, strict=True))!r}): two identical rows "
                "would be two points with one name — rows are distinct",
                f"{path}.rows[{i}]",
            )
        seen.append(row)


def _check_distinct(values: Sequence[Any], name: str, path: str, *, what: str) -> None:
    seen: list[Any] = []
    for value in values:
        if value in seen:
            raise _refuse(
                "P2",
                f"the {what} of axis {name!r} repeats {value!r}: two entries with one "
                "coordinate would be two points with one name",
                path,
            )
        seen.append(value)


def _range_values(spec: Any, path: str) -> tuple[Any, ...]:
    if (
        not isinstance(spec, list)
        or not 2 <= len(spec) <= 3
        or not all(_is_int(v) for v in spec)
    ):
        raise _refuse(
            "P2", "'range' is [start, stop] or [start, stop, step] of integers", path
        )
    step = spec[2] if len(spec) == 3 else 1
    if step == 0:
        raise _refuse("P2", "'range' step must be non-zero", path)
    n = len(range(spec[0], spec[1], step))
    if n == 0:
        raise _refuse("P2", f"'range' {spec} denotes no value", path)
    if n > MAX_AXIS_VALUES:
        raise _refuse(
            "P2",
            f"'range' denotes {n} values, over the per-axis bound of "
            f"{MAX_AXIS_VALUES} — refused before materializing (§5.14)",
            path,
        )
    return tuple(range(spec[0], spec[1], step))


def _list_values(spec: Any, path: str) -> tuple[Any, ...]:
    if not isinstance(spec, list) or not spec:
        raise _refuse(
            "P2",
            "'values' is a non-empty list of scalars"
            + ("" if isinstance(spec, list) else f", got {_type_name(spec)}"),
            path,
        )
    for i, value in enumerate(spec):
        if not _is_scalar(value):
            raise _refuse(
                "P2",
                f"a named axis's value is a scalar (it is the coordinate), got "
                f"{_type_name(value)}; a tuple of fields is a 'rows' axis",
                f"{path}[{i}]",
            )
    values = tuple(spec)
    _check_distinct(values, path.split(".")[1], path, what="values")
    return values


def _parse_dependent(
    name: str,
    decl: Mapping[str, Any],
    named: Mapping[str, NamedAxis],
    raw: Mapping[str, Any],
    model_info: Callable[[str], ModelInfo],
) -> NamedAxis:
    path = f"{AXES_KEY}.{name}"
    parent_name = decl["dependent_on"]
    if not isinstance(parent_name, str) or parent_name not in named:
        candidates = [n for n, axis in named.items() if axis.kind != "rows"]
        raise _refuse(
            "P2",
            f"axis {name!r} depends on {parent_name!r}, which is not a declared "
            f"range or values axis"
            + (
                suggest(parent_name, candidates) if isinstance(parent_name, str) else ""
            ),
            f"{path}.dependent_on",
        )
    parent = named[parent_name]
    if parent.kind == "rows":
        raise _refuse(
            "P2",
            f"axis {name!r} depends on the rows axis {parent_name!r}: a rule "
            "takes one scalar per entry, and a row's fields already move "
            "together — compute the value and write it as one more field",
            f"{path}.dependent_on",
        )
    if "rule" not in decl:
        raise _refuse(
            "P2",
            f"a dependent axis declares its 'rule' (one of {list(RULE_KINDS)})",
            path,
        )
    rule = decl["rule"]
    if not isinstance(rule, Mapping) or len(rule) != 1:
        raise _refuse(
            "P2",
            f"'rule' of axis {name!r} is an object with one key, the rule kind "
            f"({list(RULE_KINDS)})",
            f"{path}.rule",
        )
    kind = str(next(iter(rule)))
    if kind not in RULE_KINDS:
        raise _refuse(
            "P4",
            f"unknown axis rule {kind!r}; the rules are {list(RULE_KINDS)}"
            f"{suggest(kind, RULE_KINDS)}. A per-row expert, or a per-row "
            "causally-later site, is a run-time fan-out: routing is known only "
            "after a forward, and expansion is a pure function of the document "
            "(§3); declare it in the workflow (fan_out, workflow spec §2.9)",
            f"{path}.rule",
        )
    values = _clipped_band(
        name, rule[kind], parent, raw, model_info, f"{path}.rule.{kind}"
    )
    return NamedAxis(
        name=name,
        kind="dependent_on",
        keys=(),
        columns={None: values},
        parent=parent_name,
        declared=decl,
    )


def _clipped_band(
    name: str,
    spec: Any,
    parent: NamedAxis,
    raw: Mapping[str, Any],
    model_info: Callable[[str], ModelInfo],
    path: str,
) -> tuple[Any, ...]:
    """``{"width": w, "clip_to": "layers"}``: for a centre *c*, the band
    ``[max(0, c − ⌊w/2⌋) … min(L − 1, c + ⌈w/2⌉ − 1)]`` over the model's *L*
    layers — ROME's window, ten wide, so centre 0 is ``[0..4]`` and centre 47
    of 48 is ``[42..47]``."""
    if not isinstance(spec, Mapping):
        raise _refuse(
            "P2",
            f"'clipped_band' is an object {{width, clip_to}}, got {_type_name(spec)}",
            path,
        )
    for key in spec:
        if key not in _CLIPPED_BAND_KEYS:
            raise _refuse(
                "P3",
                f"unknown key {key!r} in clipped_band{suggest(str(key), _CLIPPED_BAND_KEYS)}",
                f"{path}.{key}",
            )
    for key in _CLIPPED_BAND_KEYS:
        if key not in spec:
            raise _refuse("P2", f"clipped_band declares {key!r}", f"{path}.{key}")
    width = spec["width"]
    if not _is_int(width) or width < 1:
        raise _refuse("P2", "'width' is a positive integer", f"{path}.width")
    clip_to = spec["clip_to"]
    if clip_to not in CLIP_TARGETS:
        raise _refuse(
            "P4",
            f"unknown clip target {clip_to!r}; the targets are {list(CLIP_TARGETS)}"
            + (suggest(clip_to, CLIP_TARGETS) if isinstance(clip_to, str) else ""),
            f"{path}.clip_to",
        )
    if not all(_is_int(v) for v in parent.keys):
        raise _refuse(
            "P2",
            f"clipped_band centres on the integer values of {parent.name!r}, "
            "which is not an integer axis",
            f"{path}.width",
        )
    model = raw.get("model")
    key = model.get("key") if isinstance(model, Mapping) else None
    if not isinstance(key, str):
        raise _refuse(
            "P2",
            f"clip_to 'layers' reads the layer count of one model, and "
            f"model.key is {'swept' if isinstance(key, Mapping) else 'not a string'}; "
            "clip against one model per document",
            f"{path}.clip_to",
        )
    num_layers = model_info(key).num_layers
    below, above = width // 2, width - width // 2
    bands: list[list[int]] = []
    for centre in parent.keys:
        lo = max(0, centre - below)
        hi = min(num_layers - 1, centre + above - 1)
        bands.append(list(range(lo, hi + 1)))
    assert all(bands), "a clipped band is never empty: the centre is inside the tower"
    return tuple(bands)


def _is_axis_node(node: Any) -> bool:
    """A wrapper is a mapping holding ``axis`` and nothing else — the shape
    `explicit._is_sweep` reads a sweep wrapper by. A mapping that
    carries ``axis`` beside other keys is a value (a ``gaussian`` write's
    payload, sec. 2.8, has one), so the walk needs no knowledge of the method
    vocabulary; a wrapper that meant to reference an axis and mis-spelled a
    key leaves that axis unreferenced, which is refused by name."""
    return isinstance(node, Mapping) and set(node) == {AXIS_KEY}


def _find_references(
    raw: Mapping[str, Any], named: tuple[NamedAxis, ...]
) -> tuple[tuple[tuple[str, ...], str, str | None], ...]:
    """Every ``{"axis": …}`` wrapper outside the block, in first-appearance
    order, resolved against the declarations. Like sweep wrappers, one inside
    a list has no name identity and is refused."""
    by_name = {axis.name: axis for axis in named}
    found: list[tuple[tuple[str, ...], str, str | None]] = []

    def walk(node: Any, path: tuple[str, ...], in_list: bool) -> None:
        if _is_axis_node(node):
            where = dotted_path(path)
            if in_list:
                raise _refuse(
                    "P2",
                    "an axis wrapper inside a list has no name identity; "
                    "reference the axis on a named field",
                    where,
                )
            found.append(_resolve_reference(node, where, by_name) + (path,))
            return
        if isinstance(node, Mapping):
            for key, value in node.items():
                walk(value, path + (str(key),), in_list)
        elif isinstance(node, list):
            item_paths = indexed_item_paths(node, path)
            if item_paths is not None:
                for item, item_path in zip(node, item_paths):
                    walk(item, item_path, in_list)
                return
            for item in node:
                walk(item, path, True)

    for key, value in raw.items():
        if key != AXES_KEY:
            walk(value, (str(key),), False)
    return tuple((path, name, field) for name, field, path in found)


def _resolve_reference(
    node: Mapping[str, Any], where: str, by_name: Mapping[str, NamedAxis]
) -> tuple[str, str | None]:
    ref = node[AXIS_KEY]
    if not isinstance(ref, str) or not ref:
        raise _refuse(
            "P2",
            f"'axis' names a declared axis, '<name>' or '<name>.<field>', got "
            f"{_type_name(ref)}",
            where,
        )
    name, _, field = ref.partition(".")
    axis = by_name.get(name)
    if axis is None:
        raise _refuse(
            "P2",
            f"{ref!r} names no declared axis (declared: {sorted(by_name)})"
            f"{suggest(name, by_name)}",
            where,
        )
    if axis.kind == "rows":
        fields = [str(f) for f in axis.columns if f is not None]
        if not field:
            raise _refuse(
                "P2",
                f"{name!r} is a rows axis and is referenced by field: "
                f"'{name}.<field>' with a field from {fields}",
                where,
            )
        if field not in axis.columns:
            raise _refuse(
                "P2",
                f"{ref!r}: axis {name!r} has no row field {field!r} "
                f"(fields: {fields}){suggest(field, fields)}",
                where,
            )
        return name, field
    if field:
        raise _refuse(
            "P2",
            f"{ref!r}: axis {name!r} is a {axis.kind} axis and has no fields; "
            f'write {{"axis": "{name}"}}',
            where,
        )
    return name, None


def _check_everything_is_used(
    named: tuple[NamedAxis, ...],
    references: tuple[tuple[tuple[str, ...], str, str | None], ...],
) -> None:
    """A declared axis nothing references, or a row field no wrapper reads,
    is a typo that would otherwise multiply points silently or drop a field
    the author meant to move. The ``key`` field may go unreferenced: it may
    be a label."""
    referenced = {(name, field) for _path, name, field in references}
    parents = {axis.parent for axis in named if axis.parent is not None}
    for axis in named:
        path = f"{AXES_KEY}.{axis.name}"
        if axis.kind == "rows":
            fields = [str(f) for f in axis.columns if f is not None]
            if not any((axis.name, f) in referenced for f in fields):
                raise _refuse(
                    "P2",
                    f"axis {axis.name!r} is declared but nothing references it "
                    f'({{"axis": "{axis.name}.<field>"}} on a field)',
                    path,
                )
            key = axis.declared.get("key")
            for field in fields:
                if field != key and (axis.name, field) not in referenced:
                    raise _refuse(
                        "P2",
                        f"row field {field!r} of axis {axis.name!r} is referenced "
                        f'nowhere ({{"axis": "{axis.name}.{field}"}} on the '
                        "field it moves) — or name it as the row 'key'",
                        f"{path}.rows[0].{field}",
                    )
        elif (axis.name, None) not in referenced and axis.name not in parents:
            raise _refuse(
                "P2",
                f"axis {axis.name!r} is declared but nothing references it "
                f'({{"axis": "{axis.name}"}} on a field, or a dependent axis)',
                path,
            )


def lower_axes(raw: Mapping[str, Any], axes: Axes) -> dict[str, Any]:
    """The **display form**: ``raw`` with every wrapper replaced by the
    ``{"sweep": [column]}`` it stands for and the ``axes`` group removed — a
    document the shape gate parses as it parses any swept one. Its cross
    product is *not* the expansion (the engine's ``expand_axes`` is); it is
    what ``CompiledProtocol.tree`` carries."""
    columns = {
        path: list(axes[name].columns[field]) for path, name, field in axes.references
    }
    lowered = _substitute_axis(
        raw, {path: {SWEEP_KEY: column} for path, column in columns.items()}, ()
    )
    assert isinstance(lowered, dict)
    del lowered[AXES_KEY]
    return lowered


def _substitute_axis(
    node: Any, assignment: Mapping[tuple[str, ...], Any], path: tuple[str, ...]
) -> Any:
    if _is_axis_node(node):
        return assignment[path]
    if isinstance(node, Mapping):
        return {
            str(key): _substitute_axis(value, assignment, path + (str(key),))
            for key, value in node.items()
        }
    if isinstance(node, list):
        return [_substitute_axis(item, assignment, path) for item in node]
    return node


def entries(axes: Axes) -> Iterator[tuple[int, ...]]:
    """Every combination of entry indices over the independent axes, the
    first declared slowest — the order the points come out in."""
    return itertools.product(*(range(axis.size) for axis in axes.independent))


def row_tree(axes: Axes, combo: tuple[int, ...]) -> dict[str, Any]:
    """The concrete tree one combination denotes: every wrapper replaced by
    its entry's value, the block removed — the hand-written document."""
    index = {axis.name: i for axis, i in zip(axes.independent, combo, strict=True)}
    assignment: dict[tuple[str, ...], Any] = {}
    for path, name, field in axes.references:
        axis = axes[name]
        at = index[axis.parent] if axis.is_dependent else index[name]
        assert axis.parent is None or axis.parent in index
        assignment[path] = axis.columns[field][at]
    tree = _substitute_axis(axes.source, assignment, ())
    assert isinstance(tree, dict)
    del tree[AXES_KEY]
    return tree


def canonical_axes(axes: Axes) -> dict[str, Any]:
    """The ``axes`` block as the canonical document carries it (§7): rows
    recorded folded — each field takes the value its display column received
    through `_fold`, in row 0's field order, so ``layers: 8`` and
    ``layers: [8]`` are one row and one digest — with ``key`` when authored, a
    ``range`` materialized to its values, a dependent axis with its rule as
    authored and the entries it computed — so two spellings of one campaign
    are one digest, and the campaign never reads as the display form's cross
    product."""
    out: dict[str, Any] = {}
    for axis in axes.named:
        if axis.kind == "rows":
            fields = [field for field in axis.columns if field is not None]
            entry: dict[str, Any] = {
                "rows": [
                    {field: axis.columns[field][i] for field in fields}
                    for i in range(axis.size)
                ]
            }
            if "key" in axis.declared:
                entry["key"] = axis.declared["key"]
        elif axis.is_dependent:
            entry = {
                "dependent_on": axis.parent,
                "rule": axis.declared["rule"],
                "values": list(axis.columns[None]),
            }
        else:
            entry = {"values": list(axis.keys)}
        out[axis.name] = entry
    return out


# --------------------------------------------------------------------------- #
# bands — one site across N layers, lowered to its per-layer members
# (from the planner: a document → document lowering the protocol layer
# runs before it resolves positions, and every executor runs on its document)
# --------------------------------------------------------------------------- #


def band_member(name: str, layer: int) -> str:
    """The name a band site's per-layer member — and the member of every read
    and write on it — carries once lowered: ``a[layers=10]``, §3's derived-name
    convention over the ``layers`` coordinate."""
    return f"{name}[layers={layer}]"


def lower_bands(doc: Document) -> Document:
    """``doc`` with every multi-layer band site (§2.4 ``layers``) fanned out to
    one site per member, and the reads and writes on it to one per member.

    A band is *one* address across N layers: one read on it captures N
    tensors, one write on it lands N times, and a write whose operand is a read
    on a band of the same length takes member *i*'s value at member *i* — the
    shape ROME's clipped restoration window has, "one point, N layers
    restored at once". The engines reason about one module at
    a time (``ResolvedSite`` is one module), so the executor lowers a point
    document to the N-site form the author would have written by hand before
    it resolves anything — the same form ``at_once`` (§3.1) compiles to, which
    is what lets a band and its hand-written equivalent run identically. The
    canonical form and the digest are over the authored band; only the
    execution sees the members.

    Lowering is a pure function of the document and idempotent (the members
    are one-layer sites). What the members do not cover is refused **by
    name** rather than guessed at: a band read saved, fed to a metric or
    named anywhere but as the operand of a band write of the same length has
    no single value; a featurizer on a band read or write would fit one map
    across N layers, which is a decision (a shared subspace? one per layer?)
    the document has to make explicitly, one site per layer, until the
    protocol gives it a word. Any other operand — a read on a one-layer site,
    a param, a literal — is broadcast to every member.
    """
    bands = {
        name: spec.layers
        for name, spec in doc.sites.items()
        if isinstance(spec.layers, tuple) and len(spec.layers) > 1
    }
    if not bands:
        return doc

    def label(site: str) -> str:
        from causalab.protocol.lowering import band_label  # one rendering

        return f"site {site!r} (layers {band_label(bands[site])})"

    sites: dict[str, SiteSpec] = {}
    for name, spec in doc.sites.items():
        if name in bands:
            for layer in bands[name]:
                sites[band_member(name, layer)] = dataclasses.replace(
                    spec, layers=(layer,)
                )
        else:
            sites[name] = spec

    band_reads: dict[str, str] = {}  # read → its band site
    reads: dict[str, ReadSpec] = {}
    for rname, read in doc.reads.items():
        site = str(read.site)
        if site not in bands:
            reads[rname] = read
            continue
        if read.featurizer is not None:
            raise ProtocolError(
                "P4",
                f"read {rname!r} at {label(site)} names a featurizer — a "
                "featurizer on a band would be one map fitted across every "
                "layer of it, a choice this document has to make explicitly, "
                "one site per layer",
                reason="unsupported_mechanism",
            )
        band_reads[rname] = site
        for layer in bands[site]:
            reads[band_member(rname, layer)] = dataclasses.replace(
                read, site=band_member(site, layer)
            )

    band_writes: dict[str, str] = {}  # write → its band site
    writes: dict[str, WriteSpec] = {}
    for ename, write in doc.writes.items():
        site = str(write.site)
        operands = tuple(
            name for name in _operand_names(write.do) if name in band_reads
        )
        if site not in bands:
            if operands:
                raise ProtocolError(
                    "P4",
                    f"write {ename!r} at site {site!r} takes its operand from "
                    f"read {operands[0]!r} on {label(band_reads[operands[0]])} — "
                    "a band read is N tensors, and a one-layer write lands "
                    "one; read the operand at a one-layer site, or make the "
                    "write a band of the same length",
                    reason="unsupported_mechanism",
                )
            writes[ename] = write
            continue
        if write.featurizer is not None:
            raise ProtocolError(
                "P4",
                f"write {ename!r} at {label(site)} names a featurizer — a "
                "featurizer on a band would be one map fitted across every "
                "layer of it, a choice this document has to make explicitly, "
                "one site per layer",
                reason="unsupported_mechanism",
            )
        for operand in operands:
            if len(bands[band_reads[operand]]) != len(bands[site]):
                raise ProtocolError(
                    "P4",
                    f"write {ename!r} at {label(site)} takes its operand from "
                    f"read {operand!r} on {label(band_reads[operand])} — a "
                    "band operand feeds a band write member by member, so the "
                    "two bands must have the same number of layers",
                    reason="unsupported_mechanism",
                )
        band_writes[ename] = site
        for index, layer in enumerate(bands[site]):
            substitution = {
                operand: band_member(operand, bands[band_reads[operand]][index])
                for operand in operands
            }
            writes[band_member(ename, layer)] = dataclasses.replace(
                write,
                site=band_member(site, layer),
                do=_substitute_operands(write.do, substitution),
            )

    intervened_models: dict[str, IMSpec] = {}
    for mname, im in doc.intervened_models.items():
        lowered_reads: list[str] = []
        for rname in im.reads:
            if rname in band_reads:
                lowered_reads.extend(
                    band_member(rname, layer) for layer in bands[band_reads[rname]]
                )
            else:
                lowered_reads.append(rname)
        if not isinstance(im.writes, tuple):
            intervened_models[mname] = dataclasses.replace(
                im, reads=tuple(lowered_reads)
            )
            continue
        lowered: list[str] = []
        for ename in im.writes:
            if ename in band_writes:
                lowered.extend(
                    band_member(ename, layer) for layer in bands[band_writes[ename]]
                )
            else:
                lowered.append(ename)
        intervened_models[mname] = dataclasses.replace(
            im, reads=tuple(lowered_reads), writes=tuple(lowered)
        )

    # every other mention of a band read or write has no member to go to
    named = {**band_reads, **band_writes}
    for entry in doc.save:
        if (
            entry.read is not None
            and entry.aggregation is None
            and entry.read.read in named
        ):
            raise ProtocolError(
                "P4",
                f"save entry {entry.read.read!r} names a read on "
                f"{label(named[entry.read.read])} — a band read is N tensors and "
                "the manifest saves one per entry; save each layer as its own "
                "read, or sweep the layer (§3) to get one table",
                reason="unsupported_mechanism",
            )
    for agg in doc.aggregations():
        of = agg.read.read
        if of in named:
            raise ProtocolError(
                "P4",
                f"aggregation {agg.label!r} reduces read {of!r} on "
                f"{label(named[of])} — an aggregation reads one tensor, and a "
                "band read is N",
                reason="unsupported_mechanism",
            )
    elsewhere = sorted(
        set(
            _strings(
                doc.raw.get("method", {}),
                skip=(
                    "sites",
                    "reads",
                    "writes",
                    "intervened_models",
                    "save",
                ),
            )
        )
        & set(named)
    )
    if elsewhere:
        raise ProtocolError(
            "P4",
            f"{elsewhere[0]!r} is a read or write on {label(named[elsewhere[0]])} "
            "and is named outside the tables a band lowers (reads, writes, "
            "intervened_models) — a band member has no value there",
            reason="unsupported_mechanism",
        )

    method = dict(doc.raw.get("method", {}))
    method["sites"] = {name: _site_raw(spec) for name, spec in sites.items()}
    raw = {**doc.raw, "method": method}
    return dataclasses.replace(
        doc,
        sites=sites,
        reads=reads,
        writes=writes,
        intervened_models=intervened_models,
        raw=raw,
    )


def _site_raw(spec: SiteSpec) -> dict[str, Any]:
    return {
        key: list(value) if isinstance(value, tuple) else value
        for key, value in dataclasses.asdict(spec).items()
        if value is not None
    }


def _operand_names(do: Do) -> tuple[str, ...]:
    """The read or param names a mechanism's payload references — the
    validator's rule (``validate._operand_names``) over the same payload
    shapes: ``swap``'s one operand, the ``op``/``alpha`` of ``add_scaled`` and
    ``lerp``, ``affine``'s ``A`` and ``b``. A bound read reference
    ([`ReadRef`][causalab.protocol.schema.types.ReadRef]) names its read."""
    payload = do.payload
    if isinstance(payload, ReadRef):
        return (payload.read,)
    if isinstance(payload, str):
        return (payload,)
    if isinstance(payload, Mapping):
        return tuple(
            v.read if isinstance(v, ReadRef) else v
            for v in payload.values()
            if isinstance(v, (str, ReadRef))
        )
    return ()


def _substitute_operands(do: Do, substitution: Mapping[str, str]) -> Do:
    """The payload with each named operand renamed — a [`ReadRef`][causalab.protocol.schema.types.ReadRef] keeps
    its model and renames its read."""
    if not substitution:
        return do

    def rename(value: Any) -> Any:
        if isinstance(value, ReadRef):
            return dataclasses.replace(
                value, read=substitution.get(value.read, value.read)
            )
        if isinstance(value, str):
            return substitution.get(value, value)
        return value

    payload = do.payload
    if isinstance(payload, (str, ReadRef)):
        return dataclasses.replace(do, payload=rename(payload))
    if isinstance(payload, Mapping):
        return dataclasses.replace(
            do, payload={key: rename(value) for key, value in payload.items()}
        )
    return do


def _strings(node: Any, *, skip: tuple[str, ...] = ()) -> Iterable[str]:
    """Every string value in a subtree (mapping keys included), the sections
    in ``skip`` left out at the top level."""
    if isinstance(node, Mapping):
        for key, value in node.items():
            if key in skip:
                continue
            yield str(key)
            yield from _strings(value)
    elif isinstance(node, (list, tuple)):
        for item in node:
            yield from _strings(item)
    elif isinstance(node, str):
        yield node
