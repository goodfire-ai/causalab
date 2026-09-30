"""What the document generators share.

A generator reads a template intervention specification, adds or rewrites entries, and
writes an ordinary intervention specification — nothing at run time knows it was
generated. The pieces every generator needs are here, so ``expand_layers``,
``add_routing_reads`` and ``joint_dbm`` cannot disagree about them:

* reading and writing a document as a plain tree with its key order intact —
  the four groups of spec §1, and a generator's edits land in ``method``;
* the model's static facts from the protocol registry — layer count, the mixer
  stream at each layer, whether the model routes experts;
* deterministic entry names, and the refusal that keeps a generated name from
  colliding with anything the method already declares (the named sections
  share one namespace, spec §1);
* adding a site, a read and the save entry that lets the read leave the run;
* the ``(model, input)`` pairs a document executes — the forward groups of
  spec §4, restated over the authored tree.

Everything here is torch-free: it imports the protocol layer only, which is
what lets a generator run on a machine that has never loaded a model.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

from causalab.io.sources import load_text
from causalab.protocol.registry import ModelInfo, get_model_info
from causalab.protocol.schema import (
    GROUP_ORDER,
    METHOD_SECTIONS,
    NAMED_SECTIONS,
    RESERVED_NAMES,
    Stream,
    check_protocol_version,
)

__all__ = [
    "AuthoringError",
    "add_read",
    "attach_read",
    "add_tap",
    "dump_document",
    "ensure_site",
    "entry_name",
    "executed_pairs",
    "layer_stream",
    "layer_types",
    "load_document",
    "method_of",
    "model_info",
    "moe_layers",
    "ordered_document",
    "parse_layer_range",
    "position_label",
    "read_positions",
    "reserve_name",
    "routes_experts",
]


class AuthoringError(Exception):
    """A template the generator refuses to expand. The message says why; the
    script prints it as ``refused: …`` and exits non-zero, the way the
    protocol CLI reports a load error."""


# --------------------------------------------------------------------------- #
# reading and writing
# --------------------------------------------------------------------------- #


def load_document(path: Path) -> dict[str, Any]:
    """One authored document as a plain tree, key order preserved: the four
    groups of spec §1. The version gate runs first, so a ``protocol_version``
    1 file is refused by name with ``causalab migrate`` as the answer, before
    a generator reaches for a ``method`` group the file does not have."""
    raw = load_text(path)
    check_protocol_version(raw)
    return dict(raw)


def method_of(doc: dict[str, Any]) -> dict[str, Any]:
    """The document's ``method`` group — where every generated entry lands —
    created empty when the tree has none yet. A generator edits this mapping
    in place; `dump_document` writes it in the §1 order."""
    method = doc.setdefault("method", {})
    if not isinstance(method, dict):
        raise AuthoringError(
            f"'method' must be an object, got {type(method).__name__} — the "
            "group that holds the experiment's sections (§1)"
        )
    return method


def _ordered(tree: Mapping[str, Any], order: tuple[str, ...]) -> dict[str, Any]:
    """``tree`` with the keys in ``order`` first, in that order, and anything
    outside the vocabulary after them — so the loader, not this function, is
    what refuses an unknown key."""
    out = {key: tree[key] for key in order if key in tree}
    out.update((key, value) for key, value in tree.items() if key not in out)
    return out


def ordered_document(doc: Mapping[str, Any]) -> dict[str, Any]:
    """The document with its groups in the §1 order and the method's sections
    in theirs; every entry inside a section keeps the order it was added."""
    out = _ordered(doc, GROUP_ORDER)
    method = out.get("method")
    if isinstance(method, Mapping):
        out["method"] = _ordered(method, METHOD_SECTIONS)
    return out


def dump_document(doc: Mapping[str, Any], path: Path) -> None:
    """Write a document as `ordered_document` lays it out."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(ordered_document(doc), indent=2, ensure_ascii=False)
    path.write_text(text + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# model facts
# --------------------------------------------------------------------------- #


def model_info(doc: Mapping[str, Any]) -> ModelInfo:
    """The registry entry for the document's model. The key must be one
    literal string: a generator sizes its output by the model's layer count
    and expert table, and a swept or missing key has neither. An unregistered
    key raises the loader's own ``[V4]``."""
    model = doc.get("model")
    key = model.get("key") if isinstance(model, Mapping) else None
    if not isinstance(key, str):
        raise AuthoringError(
            "model.key must be one literal model key — a generator needs the "
            "model's layer count and expert table, and a swept or missing key "
            "has neither"
        )
    return get_model_info(key)


def layer_types(info: ModelInfo) -> tuple[Stream, ...] | None:
    """The per-layer stream pattern the registry entry declares, ``None``
    when it declares none. A registry whose ``ModelInfo`` has no
    ``layer_types`` field at all reports ``None`` too — the field arrives with
    the Qwen3.6-35B-A3B row — so a generator that needs the pattern refuses
    by naming the missing registry field instead of failing on an attribute
    it assumed. Read through ``getattr`` for exactly that reason: this is the
    one place the helpers touch a field the registry may not carry yet."""
    declared = getattr(info, "layer_types", None)
    if declared is None:
        return None
    return tuple(declared)


def layer_stream(info: ModelInfo, layer: int) -> Stream | None:
    """The mixer stream at ``layer`` (``full_attention`` or
    ``linear_attention``), or ``None`` when the model declares no per-layer
    pattern — a dense tower, where every layer is softmax attention and the
    registry leaves ``layer_types`` unset (see [`layer_types`][causalab.protocol.registry.models.ModelInfo.layer_types])."""
    if not 0 <= layer < info.num_layers:
        raise AuthoringError(
            f"layer {layer} is out of range for the {info.num_layers}-layer "
            f"model {info.key!r}"
        )
    pattern = layer_types(info)
    if pattern is None:
        return None
    return pattern[layer]


def routes_experts(info: ModelInfo) -> bool:
    """Whether the model has a routed-expert table at all — the fact
    ``router_scores`` and ``expert_idx`` exist on it."""
    return info.num_experts is not None


def moe_layers(info: ModelInfo) -> list[int]:
    """The layers whose block routes experts: every layer when the model
    declares an expert table, none otherwise. The registry carries no
    per-layer MoE pattern — on Qwen3.6-35B-A3B and on the tiny MoE fixture
    every block is sparse — so this is the one place to teach a generator a
    hybrid dense/MoE tower if one arrives."""
    return list(range(info.num_layers)) if routes_experts(info) else []


def parse_layer_range(spec: str | None, info: ModelInfo) -> range:
    """``START:STOP`` (half-open, like ``--points``) into a range of layers,
    defaulting to every layer of the model. Refuses an empty or out-of-range
    slice by name rather than emitting a document with no entries."""
    if spec is None:
        return range(info.num_layers)
    match = re.fullmatch(r"(\d*):(\d*)", spec)
    if match is None:
        raise AuthoringError(f"--layers takes START:STOP (half-open), got {spec!r}")
    start = int(match.group(1)) if match.group(1) else 0
    stop = int(match.group(2)) if match.group(2) else info.num_layers
    if not 0 <= start < stop <= info.num_layers:
        raise AuthoringError(
            f"--layers {spec!r} is not a non-empty slice of the "
            f"{info.num_layers}-layer model {info.key!r}"
        )
    return range(start, stop)


# --------------------------------------------------------------------------- #
# names
# --------------------------------------------------------------------------- #

_UNSAFE = re.compile(r"[^A-Za-z0-9_]")
_COUNTERFACTUAL_INDEXED = re.compile(r"^counterfactual\[(\d+)\]$")


def entry_name(*parts: object) -> str:
    """A deterministic name from its parts, joined with ``_``. Anything but a
    letter, digit or underscore is dropped from each part, so the input role
    ``counterfactual[0]`` spells ``counterfactual0`` — legal in a name and in
    the file path derived from it."""
    return "_".join(_UNSAFE.sub("", str(part)) for part in parts)


def position_label(pos: Any) -> str:
    """The name a position contributes to a generated entry: a
    ``positions``-table name as itself, the ``all`` sugar as ``all``, and the
    integer sugar as ``i<n>`` with a leading ``m`` for a negative index
    (``-1`` is ``im1``). An inline mapping has no name to give — the
    generator asks for it to be declared in ``positions`` instead of inventing
    one from its fields."""
    if isinstance(pos, bool):
        raise AuthoringError(f"{pos!r} is not a position")
    if isinstance(pos, int):
        return f"i{pos}" if pos >= 0 else f"im{-pos}"
    if isinstance(pos, str):
        return pos  # a positions-table name, or the "all" sugar
    raise AuthoringError(
        f"inline position {json.dumps(pos)} has no name to build an entry name "
        "from — declare it in `positions` and reference it by name"
    )


def reserve_name(method: Mapping[str, Any], name: str, section: str) -> None:
    """Refuse ``name`` if the method already declares it anywhere in the
    shared namespace (spec §1: method sections 2–10 are one namespace) or if
    it is reserved. Called before every generated entry, so a generator applied
    twice refuses instead of silently producing a document the loader would
    refuse — or worse, one it would not."""
    if name in RESERVED_NAMES or _COUNTERFACTUAL_INDEXED.match(name):
        raise AuthoringError(f"{name!r} is a reserved name (§5.3)")
    for declared_in in NAMED_SECTIONS:
        table = method.get(declared_in)
        if isinstance(table, Mapping) and name in table:
            raise AuthoringError(
                f"cannot add {section}.{name}: {name!r} is already declared in "
                f"{declared_in!r} — the named sections share one namespace, and "
                "a generated document is refused rather than renamed"
            )


# --------------------------------------------------------------------------- #
# adding entries
# --------------------------------------------------------------------------- #


def ensure_site(method: dict[str, Any], name: str, spec: Mapping[str, Any]) -> None:
    """Declare site ``name`` as ``spec``. A site already declared under that
    name with the same spec is left alone — several reads share one site,
    and the second generated read at a layer meets the first one's site — but
    the same name with a different spec is a collision and refuses."""
    sites = method.setdefault("sites", {})
    existing = sites.get(name)
    if existing is not None:
        if existing == dict(spec):
            return
        raise AuthoringError(
            f"site {name!r} is already declared as {json.dumps(existing)}, not "
            f"{json.dumps(dict(spec))}"
        )
    reserve_name(method, name, "sites")
    sites[name] = dict(spec)


def add_read(
    method: dict[str, Any],
    name: str,
    *,
    site: str,
    pos: Any,
    model: str,
    input: str,
    file_path: str | None = None,
    reduce: str | None = None,
    dims: Any = None,
) -> None:
    """Declare read ``name`` at ``site`` and the save entry that lets it leave
    the run (spec §2.7, §2.12). ``file_path`` defaults to
    ``<name>.safetensors``; ``reduce`` is the save-time statistic (a mean
    harvest), ``dims`` the read's static feature index. Refuses a name the
    document already declares, a site it does not, and a file another save
    entry already writes."""
    if site not in method.get("sites", {}):
        raise AuthoringError(f"read {name!r} names undeclared site {site!r}")
    reserve_name(method, name, "reads")
    path = file_path if file_path is not None else f"{name}.safetensors"
    save = method.setdefault("save", [])
    taken = {entry.get("file_path") for entry in save if isinstance(entry, Mapping)}
    if path in taken:
        raise AuthoringError(
            f"a save entry already writes {path!r} — one file per entry (§2.12)"
        )
    read: dict[str, Any] = {"site": site, "pos": pos}
    if dims is not None:
        read["dims"] = dims
    method.setdefault("reads", {})[name] = read
    attach_read(method, name, model=model, input=input)
    entry: dict[str, Any] = {"read": name, "model": model}
    if reduce is not None:
        entry["reduce"] = reduce
    entry["file_path"] = path
    save.append(entry)


def attach_read(method: dict[str, Any], name: str, *, model: str, input: str) -> None:
    """List read ``name`` on ``model`` (spec §2.9): an existing model gains
    the read and must run on ``input``; a new name is declared as the
    un-intervened model on ``input`` — no writes, only reads."""
    models = method.setdefault("intervened_models", {})
    entry = models.get(model)
    if entry is None:
        entry = models[model] = {"input": input, "reads": []}
    elif not isinstance(entry, Mapping):
        raise AuthoringError(f"intervened_models.{model} is not an object")
    if str(entry.get("input")) != input:
        raise AuthoringError(
            f"model {model!r} runs on {entry.get('input')!r}, not {input!r} — a "
            "read is taken on the model's own input (§2.9)"
        )
    listed = entry.setdefault("reads", [])
    if not isinstance(listed, list):
        raise AuthoringError(f"intervened_models.{model}.reads is not a list")
    if name not in listed:
        listed.append(name)


def add_tap(
    method: dict[str, Any],
    *,
    site: str,
    component: str,
    layer: int,
    read: str,
    pos: Any,
    model: str,
    input: str,
    reduce: str | None = None,
    dims: Any = None,
) -> None:
    """One site/read/save triple: the site ``{component, layer}`` under
    ``site`` (shared with any earlier tap at the same address), the read
    ``read`` on it for ``(model, input)`` at ``pos``, and its save entry."""
    ensure_site(method, site, {"component": component, "layers": [layer]})
    add_read(
        method,
        read,
        site=site,
        pos=pos,
        model=model,
        input=input,
        reduce=reduce,
        dims=dims,
    )


# --------------------------------------------------------------------------- #
# what a document executes
# --------------------------------------------------------------------------- #


def executed_pairs(method: Mapping[str, Any]) -> list[tuple[str, str]]:
    """The ``(model, input)`` pairs the method runs a forward for, in the
    planner's order: every declared model on its own input, in declaration
    order — the un-intervened models included, since protocol 4 declares
    them like any other (spec §2.9, §4;
    [`causalab.neural.shared.plan.plan_point`][] derives the same list from
    the parsed document, and the tests hold the two equal over the corpus).
    A generator that must observe every executed condition — routing capture
    — iterates this."""
    return [
        (name, str(im.get("input")))
        for name, im in _entries(method, "intervened_models").items()
    ]


def read_positions(method: Mapping[str, Any], model: str, input: str) -> list[Any]:
    """Every ``pos`` the method's reads on ``(model, input)`` use, in
    declaration order, deduplicated by value — the positions a generator adds
    its own reads at when told nothing else."""
    out: list[Any] = []
    entry = _entries(method, "intervened_models").get(model)
    if entry is None or str(entry.get("input")) != input:
        return out
    reads = _entries(method, "reads")
    listed = entry.get("reads")
    for rname in listed if isinstance(listed, list) else ():
        read = reads.get(str(rname))
        if read is None:
            continue
        pos = read.get("pos")
        if pos not in out:
            out.append(pos)
    return out


def _entries(method: Mapping[str, Any], section: str) -> dict[str, Mapping[str, Any]]:
    """A named section's mapping-valued entries — the authored tree has not
    been parsed yet, so a malformed entry is skipped here and refused by the
    loader, which says what is wrong with it."""
    table = method.get(section, {})
    if not isinstance(table, Mapping):
        return {}
    return {name: entry for name, entry in table.items() if isinstance(entry, Mapping)}
