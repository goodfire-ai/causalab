"""Addressing one entry inside a ``.safetensors`` bundle (§2.5, §2.6).

A run writes **one file per ``save`` entry**, across every point of a swept
document: the engine suffixes each tensor key with the point's coordinate
label (``weight[k=8,seed=0]``, ``v_mean[target.layers=12]`` —
[`causalab.protocol.lowering.coordinate_label`][]). Consumers, on the other
side, name a *slot* (``weight`` for a subspace/pca bundle, ``value`` for a
``params`` constant). Without a selector the two vocabularies only coincide
for an un-swept producer, so every swept handoff fails at run time with a
``KeyError`` — the gap this module closes.

The selector is the optional ``entry`` field on a spec that also carries
``file_path``: a mapping of coordinate name to value
(``{"k": 8, "seed": 0}``), matched against the coordinates parsed back out
of the bundle's keys. Matching is on **(name, value) pairs**, never on the
rendered label string as a whole — the label's field order follows the
producing document's axis order, which the consumer has no way to know.

Three resolution rules, in order:

1. exactly one entry for the slot ⇒ that one (an un-swept producer keeps
   working, and an external bundle needs no ``entry``);
2. otherwise, the entries matching every requested pair — exactly one is
   the answer;
3. anything else (no match, or several) is an error that lists what the
   bundle actually holds. Never first-hit-wins: silently applying the wrong
   fit is the failure mode this whole seam exists to prevent.

``entry`` may be omitted on a swept consumer: the requested pairs then come
from the consuming point's own sweep coordinates, so an ``apply`` document
swept on ``featurizers.rot.k`` selects the fit at *its* ``k`` — axis
identity is name identity (§3), so the same axis name means the same axis
on both sides, and the sweeps zip instead of crossing.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Mapping

from causalab.protocol.rules.errors import ValidationError
from causalab.protocol.lowering import label_value, short_coords

__all__ = [
    "RAGGED_SUFFIX",
    "SLOT_KEY",
    "entry_key",
    "entry_selection",
    "parse_entry_key",
    "select_entry",
]

#: Sidecar key holding a ragged read's per-row widths (``outputs.TensorFile``).
RAGGED_SUFFIX = ".widths"

#: Reserved key inside an ``entry`` selector: the producer's *name* for the
#: tensor, as opposed to its coordinates. A ``params`` constant defaults to
#: ``value``, but a bundle harvested from a read is keyed by the read's name,
#: so the consumer has to be able to say which.
SLOT_KEY = "slot"

_KEY = re.compile(r"^(?P<slot>[^\[]+)\[(?P<coords>.*)\]$")


def entry_key(slot: str, label: str) -> str:
    """The bundle key for ``slot`` under a coordinate ``label`` (possibly
    empty, for an un-swept document)."""
    return f"{slot}{label}"


def parse_entry_key(key: str) -> tuple[str, dict[str, str]]:
    """Split a bundle key into its slot and its coordinate pairs, as
    strings — ``"weight[k=8,seed=0]"`` → ``("weight", {"k": "8",
    "seed": "0"})``.

    This is the *fallback* reader, for bundles written before the entry
    table existed or by hand. It deliberately gives up (whole key as slot,
    no coordinates) on a label carrying a structured value — a swept
    position renders as ``tap={index:-1}`` and a swept list keeps its own
    brackets and commas, neither of which survives a flat split. Those
    bundles are addressed through the ``__metadata__`` entry table instead,
    which stores coordinates as data.
    """
    match = _KEY.match(key)
    if match is None:
        return key, {}
    coords_text = match.group("coords")
    if not coords_text:
        return match.group("slot"), {}
    if any(ch in coords_text for ch in "{}[]"):
        return key, {}
    coords: dict[str, str] = {}
    for part in coords_text.split(","):
        name, sep, value = part.partition("=")
        if not sep:  # not a coordinate pair — treat the key as opaque
            return key, {}
        coords[name] = value
    return match.group("slot"), coords


def select_entry(
    keys: Iterable[str],
    slot: str,
    want: Mapping[str, Any] | None,
    *,
    what: str,
    coords_by_key: Mapping[str, Mapping[str, Any]] | None = None,
    implicit: bool = False,
) -> str:
    """The single bundle key for ``slot`` matching every requested
    coordinate (module docstring).

    ``coords_by_key`` is the producer's ``__metadata__`` entry table when it
    has one; it is authoritative, because it stores coordinates as data
    rather than as a rendered label. ``what`` names the consumer in errors.

    ``implicit`` marks a selection derived from the consuming point's own
    coordinates rather than authored: axes the producer never had are then
    dropped instead of matching nothing, so a consumer swept on a layer it
    shares with nobody still resolves a bundle keyed only by ``k``.
    """
    candidates: list[tuple[str, Mapping[str, Any]]] = []
    for key in keys:
        if key.endswith(RAGGED_SUFFIX):
            continue  # a sidecar, addressed via its parent key
        if coords_by_key is not None and key in coords_by_key:
            stored = coords_by_key[key]
            parsed_slot = str(stored.get("slot", parse_entry_key(key)[0]))
            coords: Mapping[str, Any] = stored.get("coords", {})
        else:
            parsed_slot, coords = parse_entry_key(key)
        if parsed_slot == slot:
            candidates.append((key, coords))
    if not candidates:
        raise ValidationError(
            15,
            f"{what}: the bundle holds no {slot!r} entry "
            f"(has {sorted(str(k) for k in keys)})",
        )
    if len(candidates) == 1 and (not want or not candidates[0][1]):
        # one entry, and either nothing asked or nothing recorded to ask
        # about: a hand-made or un-swept bundle carries no coordinates, so it
        # cannot answer a coordinate question — and cannot contradict one
        # either. What the entry actually *is* stays guarded by its
        # ArtifactIdentity.
        return candidates[0][0]
    requested = {name: label_value(value) for name, value in (want or {}).items()}
    if implicit:
        known = {name for _, coords in candidates for name in coords}
        requested = {name: value for name, value in requested.items() if name in known}
    matched = [
        key
        for key, coords in candidates
        if all(
            name in coords and label_value(coords[name]) == value
            for name, value in requested.items()
        )
    ]
    if len(matched) == 1:
        return matched[0]
    available = sorted(key for key, _ in candidates)
    if not requested:
        raise ValidationError(
            15,
            f"{what}: the bundle holds {len(candidates)} {slot!r} entries "
            f"({available}) and the document selects none — add an 'entry' "
            "selector, or sweep the consumer on the producer's axis (§2.5)",
            reason="empty_selector",
        )
    shown = ", ".join(f"{name}={value}" for name, value in sorted(requested.items()))
    if not matched:
        raise ValidationError(
            15,
            f"{what}: no {slot!r} entry matches {{{shown}}} (has {available})",
            reason="empty_selector",
        )
    missing = sorted(
        {name for key, coords in candidates if key in matched for name in coords}
        - set(requested)
    )
    naming = f" ({', '.join(missing)})" if missing else ""
    raise ValidationError(
        15,
        f"{what}: {len(matched)} {slot!r} entries match {{{shown}}} "
        f"({sorted(matched)}) — the selection must be unique, so name the "
        f"remaining coordinates{naming} in 'entry'. An authored 'entry' is "
        "used exactly as written and is NOT completed from the consuming "
        "point's coordinates, so pinning a coordinate elsewhere in the "
        "document does not narrow this; the other way out is to drop 'entry' "
        "entirely and let the point's own coordinates imply the selection "
        "(§2.5)",
    )


def entry_selection(
    authored: Any,
    coords: Mapping[str, Any] | None,
    name: str,
) -> tuple[Mapping[str, Any] | None, bool]:
    """What one loaded spec selects, and whether it was implied.

    An authored ``entry`` wins as written — **all of it, and only it**. It is
    deliberately *not* completed from the point's coordinates for the keys it
    omits: the two are alternative ways to say the same thing, and mixing them
    would make a selector's meaning depend on which axes the consuming document
    happens to sweep. A partial ``entry`` against a multi-axis bundle therefore
    resolves only if it is already unique, and otherwise refuses naming the
    coordinates that would disambiguate ([`select_entry`][]). Pinning ``k``
    elsewhere in the document does not narrow an ``entry`` that omits ``k``.

    With no authored ``entry`` the consuming point's own coordinates stand in —
    that is what lets an ``apply`` document swept on ``featurizers.rot.k`` pick
    up the fit at *its* ``k`` without naming a single entry, and what keeps the
    two sweeps zipped instead of crossed (§3: axis identity is name identity).
    An un-swept consumer implies nothing, so a single-entry bundle still
    resolves by slot alone.
    """
    if authored:
        return {
            key: value for key, value in authored.items() if key != SLOT_KEY
        } or None, False
    if coords:
        return short_coords(coords, entry=name), True
    return None, False


def selector_slot(authored: Any, default: str) -> str:
    """The slot an ``entry`` selector names, or ``default``."""
    if isinstance(authored, Mapping):
        slot = authored.get(SLOT_KEY)
        if isinstance(slot, str):
            return slot
    return default
