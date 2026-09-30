"""The explicit form of a document: every default and derived value written out (spec §7).

The authored file is for humans; the canonical form is the record. It
materializes every default (optimizer betas, dtypes, the implicit
``revision``), every resolved reference (dataset content digests, artifact
file hashes), every derived width, expands sugar (int and ``"all"``
positions), sorts unordered lists (IM write lists), and drops the header's
authoring metadata (``title``, ``description`` — §1). Whether a site's
address exists on the model is rule 4's, decided by the checklist
(`causalab.protocol.rules.document`) before any document reaches here.

This is the *materialising* half of what ``protocol/canonical.py`` used to
be; the *hashing* half — `canonical_bytes`
and `digest` — is
`causalab.protocol.identity`'s, re-exported here so the old path keeps
naming both. It lives in the ``schema`` package because it is the last word
on the object model: what a parsed document *is* once nothing is left
implicit. The package ``__init__`` does not import it — the registry imports
the schema package and this module imports the registry, so an eager import
here would be a cycle.

Two granularities share one implementation:

* [`canonicalize`][] on a *concrete* raw tree (a compiled intervention, or an
  un-swept document) materializes everything — the **point digest** names
  the point in the receipt (``points[].digest``).
* On a swept document the sweep wrappers stay in place (they are the
  campaign's identity) and any derived value that depends on a swept field
  is left unmaterialized — the **document digest** names the campaign.

One deliberate interpretation: a compiled intervention's
canonical bytes contain no campaign metadata (no coordinates, no parent
digest), so a point re-authored standalone digests identically to the same
point reached by expansion. Campaign linkage is recorded in run outputs,
never in the canonical bytes.
"""

from __future__ import annotations

import dataclasses
import math
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from causalab.protocol.rules.errors import ValidationError
from causalab.protocol.identity import (
    canonical_bytes,
    closure_sha256,
    digest,
    import_closure,
    is_installed_module,
    resolve_locator_or_refuse,
    source_root,
    source_sha256,
)
from causalab.protocol.registry import (
    ModelInfo,
    component_shape,
    component_width,
    gate_param_shape,
    site_group_map,
)
from causalab.tables import INPUT_COLUMN, inline_ref
from causalab.protocol.schema import (
    ALL_POSITIONS,
    CONTROL_DEFAULTS,
    DEPRECATED_COMPONENTS,
    FEATURIZER_SLOTS,
    GATE_DEFAULT_MAP,
    inline_train_saves,
    METHOD_SECTIONS,
    METRIC_FIELD_DEFAULTS,
    READ_TARGET_METRIC_KINDS,
    MODEL_DTYPE_DEFAULT,
    ModelRef,
    OPTIMIZER_DEFAULTS,
    OPTIONAL_METRIC_FIELDS,
    parse_document,
    REGULARIZER_KINDS,
    SCORES_INIT_DEFAULTS,
    to_base_form,
    WRITES_DURING_GENERATION_FIELD,
)

if TYPE_CHECKING:
    from causalab.io.env import ResolutionEnv

__all__ = [
    "canonical_bytes",
    "canonical_model",
    "canonical_model_ref",
    "canonicalize",
    "digest",
]


def _is_sweep(node: Any) -> bool:
    return isinstance(node, Mapping) and set(node) == {"sweep"}


def canonicalize(
    raw: Mapping[str, Any],
    env: ResolutionEnv,
    *,
    axes: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The canonical form of one raw document tree (artifact fields already
    resolved). Concrete documents materialize fully; swept documents keep
    their wrappers and skip sweep-dependent derivations.

    ``axes`` is the campaign's named-axes block (§3.2), as the compiler's
    ``axes`` stage parsed it (``axes.canonical_axes``): emitted between
    ``data`` and ``method`` exactly when given — the ``unit`` / ``shuffle`` /
    ``draw`` "digest-bearing when authored" precedent (§7) — and never read
    from ``raw``, whose gate below knows the four groups alone. Every
    document without the group keeps its canonical bytes."""
    doc = parse_document(raw)  # shape-checks the tree we are about to walk
    del doc  # only the raw tree is transformed; parse is the gate

    model_raw = raw["model"]
    model_key = model_raw.get("key")
    info: ModelInfo | None = None
    if isinstance(model_key, str):
        info = env.model_info(model_key)

    # The four groups (§1), each canonicalized in place. The header keeps only
    # `protocol_version`: `title` and `description` say what a file is for,
    # never what the experiment is, so a rename or a reworded intent moves no
    # digest (§7). The method's sections are walked in their recommended
    # order, and the lookups a section's canonicalization needs (a
    # featurizer's sites, a chain's members) resolve inside the method group.
    normalized: Mapping[str, Any] = raw["method"]
    attached = _attached_reads(normalized)
    out: dict[str, Any] = {
        "header": {"protocol_version": raw["header"]["protocol_version"]},
        "model": canonical_model(model_raw),
        "data": _canon_data(raw["data"], env),
    }
    if axes is not None:
        out["axes"] = dict(axes)
    method_out: dict[str, Any] = {}
    out["method"] = method_out
    for section in METHOD_SECTIONS:
        if section not in normalized:
            continue
        value = normalized[section]
        if section == "positions":
            method_out["positions"] = {
                name: _canon_position_entry(entry) for name, entry in value.items()
            }
        elif section == "sites":
            method_out["sites"] = {
                name: _canon_site(name, entry) for name, entry in value.items()
            }
        elif section == "featurizers":
            method_out["featurizers"] = {
                name: _canon_featurizer(name, entry, normalized, info, env)
                for name, entry in value.items()
            }
        elif section == "reads":
            method_out["reads"] = {
                name: _canon_read_or_edit(entry) for name, entry in value.items()
            }
        elif section == "writes":
            method_out["writes"] = {
                name: _canon_write(entry, attached) for name, entry in value.items()
            }
        elif section == "intervened_models":
            method_out["intervened_models"] = {
                name: _canon_intervened_model(entry) for name, entry in value.items()
            }
        elif section == "params":
            method_out["params"] = {
                name: _canon_param(entry, env) for name, entry in value.items()
            }
        elif section == "code":
            method_out["code"] = {
                name: _canon_code(name, entry, env) for name, entry in value.items()
            }
        elif section == "train":
            method_out["train"] = _canon_train(value, info, env, attached)
        elif section == "save" and isinstance(value, list):
            # a `train` save is the term it names, spelled out (§2.12), so a
            # reference and its inline twin are one canonical form
            method_out["save"] = [
                _canon_save_entry(entry, attached)
                for entry in inline_train_saves(normalized)
            ]
        else:
            method_out[section] = value
    return out


def _attached_reads(method: Mapping[str, Any]) -> dict[str, list[str]]:
    """Read name → the models that list it (§2.9), off the authored tree —
    what a bare read reference binds to in the canonical form."""
    out: dict[str, list[str]] = {}
    models = method.get("intervened_models")
    if isinstance(models, Mapping):
        for name, entry in models.items():
            reads = entry.get("reads") if isinstance(entry, Mapping) else None
            if isinstance(reads, list):
                for rname in reads:
                    if isinstance(rname, str):
                        out.setdefault(rname, []).append(name)
    return out


def _canon_read_ref(value: Any, attached: Mapping[str, list[str]]) -> Any:
    """A read reference in its one canonical spelling, the object form with
    the model filled (§2.7): a bare name bound to the one model that lists
    the read; a name no model or several models list is left as authored
    (validation refuses it); the object form passes through."""
    if isinstance(value, str) and len(attached.get(value, ())) == 1:
        return {"read": value, "model": attached[value][0]}
    return value


def _canon_operand(value: Any, attached: Mapping[str, list[str]]) -> Any:
    """A write operand: a read reference in object form, inside a sweep
    wrapper per value; params and literals as authored."""
    if _is_sweep(value) and isinstance(value["sweep"], list):
        return {"sweep": [_canon_operand(v, attached) for v in value["sweep"]]}
    return _canon_read_ref(value, attached)


def _canon_write(entry: Mapping[str, Any], attached: Mapping[str, list[str]]) -> Any:
    out = _canon_read_or_edit(entry)
    do = out.get("do")
    if isinstance(do, Mapping) and len(do) == 1:
        ((mech, payload),) = do.items()
        if mech == "swap":
            out["do"] = {mech: _canon_operand(payload, attached)}
        elif mech in ("add_scaled", "lerp") and isinstance(payload, Mapping):
            out["do"] = {
                mech: {
                    key: (_canon_operand(v, attached) if key == "op" else v)
                    for key, v in payload.items()
                }
            }
    return out


def _canon_save_entry(entry: Any, attached: Mapping[str, list[str]]) -> Any:
    if isinstance(entry, Mapping) and isinstance(entry.get("aggregation"), Mapping):
        return {
            **entry,
            "aggregation": _canon_aggregation(entry["aggregation"], attached),
        }
    return entry


def _canon_aggregation(entry: Any, attached: Mapping[str, list[str]]) -> Any:
    """Materialize an aggregation's optional fields to their defaults (§2.10),
    so a document that spells out ``"mode": "exact"`` and one that omits it
    are one canonical form — the same treatment ``train.optimizer`` defaults
    get — and spell a ``kl`` / ``js`` ``target`` in the object form.

    The identity fields ``unit`` / ``estimand_version`` are **not** in
    ``OPTIONAL_METRIC_FIELDS`` on purpose: they ride through from ``entry``
    only when authored (§7), so a document that states neither keeps the
    digest it had before they existed. Their derived values land on the
    metric's rows, never here. An optional field with **no** default
    (``js.restrict``) is left absent for the same reason: absent means
    "unrestricted", and materializing a spelling of that would move the
    digest of every unrestricted document."""
    if not isinstance(entry, Mapping):
        return entry
    kind = entry.get("kind")
    if not isinstance(kind, str):
        return entry
    out = dict(entry)
    for field in OPTIONAL_METRIC_FIELDS.get(kind, ()):
        if (kind, field) in METRIC_FIELD_DEFAULTS:
            out.setdefault(field, METRIC_FIELD_DEFAULTS[(kind, field)])
    if kind in READ_TARGET_METRIC_KINDS and "target" in out:
        out["target"] = _canon_read_ref(out["target"], attached)
    return out


def canonical_model(value: Mapping[str, Any]) -> dict[str, Any]:
    """§2.1 — the network *and* how it is realized numerically. ``revision``
    and ``dtype`` are materialized here, so no canonical form is silent about
    the precision its numbers came out of; a ``quantization`` block
    materializes the scheme's own defaults for the same reason. An explicit
    attention backend is preserved; omission leaves the engine default."""
    out: dict[str, Any] = {
        "key": value["key"],
        "revision": value.get("revision", "main"),
        "dtype": value.get("dtype", MODEL_DTYPE_DEFAULT),
    }
    # Unlike precision, the historical attention default is engine-specific.
    # Preserve omission, but hash an explicit backend in every model identity.
    if "attn_implementation" in value:
        out["attn_implementation"] = value["attn_implementation"]
    quantization = value.get("quantization")
    if quantization is not None:
        quant = dict(quantization)
        quant.setdefault("method", "bitsandbytes")
        if quant.get("scheme") in ("nf4", "fp4"):
            # 📐 `compute_dtype` is a 4-bit knob: BitsAndBytesConfig takes it
            # as bnb_4bit_compute_dtype, and the int8 path
            # (`load_in_8bit=True`) has nowhere to put it — LLM.int8()
            # accumulates in fp16 by construction. Materializing it for every
            # scheme gave two int8 documents that differ only there two
            # different digests for numerically identical runs, which inverts
            # the rule this whole function exists to keep: every field in the
            # canonical form is one that moves a number. Rule 17 refuses an
            # authored one, so this only has to stop inventing it.
            quant.setdefault("compute_dtype", out["dtype"])
            quant.setdefault("double_quant", False)
        if quant.get("scheme") == "int8":
            # bitsandbytes' LLM.int8() outlier threshold: a number that moves
            # numbers, so it is materialized like any other (arXiv:2208.07339)
            quant.setdefault("int8_threshold", 6.0)
        out["quantization"] = quant
    return out


def canonical_model_ref(model: ModelRef) -> dict[str, Any]:
    """[`canonical_model`][] over the *parsed* form (§2.1).

    The planner holds a [`ModelRef`][causalab.protocol.schema.types.ModelRef], not the
    raw mapping, and its interning digests have to agree with the canonical
    form field for field. Routing through [`canonical_model`][] rather than
    re-listing the defaults is the point: a default added there — a new
    quantization knob, say — reaches the interning digests without anyone
    remembering to copy it, which is exactly the drift that let fp32 and nf4
    realizations share a forward group.
    """
    value: dict[str, Any] = {"key": model.key, "revision": model.revision}
    if model.dtype is not None:
        value["dtype"] = model.dtype
    if model.attn_implementation is not None:
        value["attn_implementation"] = model.attn_implementation
    if model.quantization is not None:
        value["quantization"] = {
            field.name: getattr(model.quantization, field.name)
            for field in dataclasses.fields(model.quantization)
            if getattr(model.quantization, field.name) is not None
        }
    return canonical_model(value)


def _canon_param(entry: Mapping[str, Any], env: ResolutionEnv) -> dict[str, Any]:
    """§7: "each param replaced by its content hash" — a loaded constant's
    bytes are its identity (this also makes a missing file a load error)."""
    out = dict(entry)
    file_path = out.get("file_path")
    if isinstance(file_path, str):
        out["content_digest"] = env.artifacts.file_digest(file_path)
    return out


def _canon_code(
    name: str, entry: Mapping[str, Any], env: ResolutionEnv
) -> dict[str, Any]:
    """§2.8.1 — a code reference's identity is the content of what it names.

    Four derivations, all of them the same move ``_canon_param`` makes for a
    loaded constant: the resolved module (so the record says *which file* was
    hashed), that file's ``source_sha256`` (so editing the code moves the
    document digest), the manifest and hash of its declared import closure
    (``closure`` / ``closure_sha256`` — the sibling modules *beside* the code
    that nothing else covers, written only when there are any), and a content
    digest per declared data input (so the externally selected noise-scale
    file that ROME's corruption function read is part of the protocol, not
    beside it). ``source_sha256`` keeps its meaning — the defining module's
    bytes alone — and the closure never contains the module itself. A module
    inside the ``causalab`` package declares no closure: the package's bytes
    are runtime identity, the ``tree_digest`` ``--resume`` compares (§7).

    ``env_inputs`` is sorted here and its **values are not read**: the digest
    names the variables a function is allowed to consult, which is a property
    of the document, while their values are a property of the machine and
    belong in a run's execution record instead.
    """
    out = dict(entry)
    locator = out.get("locator")
    if isinstance(locator, str):
        resolved = resolve_locator_or_refuse(locator, path=f"code.{name}.locator")
        out["source_module"] = resolved.module
        out["source_sha256"] = source_sha256(resolved.path)
        # a locator into an installed third-party package or the stdlib names
        # a file the document hashes — but that module's imports are runtime
        # identity, never document identity (§2.8.1), so it declares no closure;
        # the identity walk (`repository=False`) likewise skips the package's
        # own modules, so a repository locator carries the keys only when its
        # module imports a sibling outside the package
        closure = (
            import_closure(resolved.path, root=source_root(resolved), repository=False)
            if not is_installed_module(resolved.path)
            else {}
        )
        if closure:
            out["closure"] = closure
            out["closure_sha256"] = closure_sha256(closure)
    env_inputs = out.get("env_inputs")
    if isinstance(env_inputs, list):
        out["env_inputs"] = sorted(env_inputs)
    data_inputs = out.get("data_inputs")
    if isinstance(data_inputs, Mapping) and data_inputs:
        out["data_input_digests"] = {
            name: env.artifacts.file_digest(path)
            for name, path in sorted(data_inputs.items())
            if isinstance(path, str)
        }
    return out


def _canon_data(data: Mapping[str, Any], env: ResolutionEnv) -> dict[str, Any]:
    # a role-less block is the base role (§2.2): converted here as well as at
    # parse, so the short and the explicit spelling digest identically
    data = to_base_form(data)

    def one(role: Mapping[str, Any]) -> dict[str, Any]:
        stamped = dict(role)
        if "inputs" in role:
            # an inline role (§2.2): the authored prompts stay, and the two
            # derived fields every reader of a canonical role expects — the
            # column it reads and the ref its rows resolve by — materialize,
            # so a canonical inline role and a canonical file role have one
            # shape and their digests compare by content alone
            ref = inline_ref(list(role["inputs"]))
            stamped["field"] = INPUT_COLUMN
            stamped["dataset"] = ref
            stamped["digest"] = env.datasets.digest(ref)
            return stamped
        ref = role["dataset"]
        if isinstance(ref, str):
            stamped["digest"] = env.datasets.digest(ref)
        return stamped

    out: dict[str, Any] = {"base": one(data["base"])}
    if "counterfactual" in data:
        cf = data["counterfactual"]
        out["counterfactual"] = (
            [one(s) for s in cf] if isinstance(cf, list) else one(cf)
        )
    return out


def _canon_intervened_model(entry: Mapping[str, Any]) -> dict[str, Any]:
    """§2.9: the input, the sorted reads, the sorted writes **only when
    non-empty** (an authored ``[]`` and an absent field are the same
    un-intervened model), and ``writes_during_generation`` **only when
    true** — an authored ``false`` and an absent field are the same
    intervention (prefill-only), so they digest identically."""
    out: dict[str, Any] = {"input": entry["input"]}
    reads = entry.get("reads")
    out["reads"] = sorted(reads) if isinstance(reads, list) else reads
    writes = entry.get("writes")
    if writes is not None and writes != []:
        out["writes"] = _canon_write_list(writes)
    if entry.get(WRITES_DURING_GENERATION_FIELD) is True:
        out[WRITES_DURING_GENERATION_FIELD] = True
    return out


def _canon_write_list(writes: Any) -> Any:
    """IM write lists are unordered (§6.8) — sorted concrete, and sorted
    per-value inside a sweep wrapper, so one campaign has one spelling."""
    if isinstance(writes, list):
        return sorted(writes)
    if _is_sweep(writes) and isinstance(writes["sweep"], list):
        return {
            "sweep": [sorted(v) if isinstance(v, list) else v for v in writes["sweep"]]
        }
    return writes


def _canon_position_spec(value: Any) -> Any:
    if isinstance(value, int) and not isinstance(value, bool):
        return {"index": value}  # §6.1 sugar
    if value == ALL_POSITIONS:
        return {"all": True}  # §6.1 sugar
    return value


def _canon_position_entry(entry: Any) -> Any:
    if _is_sweep(entry):
        spec = entry["sweep"]
        if isinstance(spec, list):
            return {"sweep": [_canon_position_spec(v) for v in spec]}
        return entry
    return _canon_position_spec(entry)


def _canon_site(name: str, entry: Mapping[str, Any]) -> dict[str, Any]:
    entry = dict(entry)
    component = entry.get("component")
    if isinstance(component, str) and component in DEPRECATED_COMPONENTS:
        # A retired spelling canonicalizes to its replacement, so both digest
        # identically and the canonical form is in one vocabulary. Folding only
        # at parse would leave the *canonical* document carrying a name no table
        # downstream knows — `component_shape` would refuse it, which is the
        # opposite of what an alias is for.
        component = DEPRECATED_COMPONENTS[component]
        entry["component"] = component
    layers = entry.get("layers")
    if isinstance(layers, int) and not isinstance(layers, bool):
        # A bare index is the one-layer band `[n]` (§2.4): an axis over
        # `layers` and a workflow `emit` hand a point the index, and the
        # canonical form writes the list either way, so both spellings carry
        # one digest — the same fold the parser makes (`schema._band`).
        layers = [layers]
        entry["layers"] = layers
    # Whether the address exists on the model — each layer of the band inside
    # the tower and on the declared stream, the component present on this
    # entry, a `head` inside the component's head space — is rule 4's address
    # half (`rules.document._check_site_addresses`), decided by the validate
    # stage before any point reaches here; the folds above are all this
    # function keeps.
    return dict(entry)


def _canon_read_or_edit(entry: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(entry)
    if "pos" in out:
        pos = out["pos"]
        # a string pos is a positions-table name and stays one — except the
        # reserved "all", which is sugar and expands like a bare int
        out["pos"] = (
            pos
            if isinstance(pos, str) and pos != ALL_POSITIONS
            else _canon_position_entry(pos)  # sugar expands inside sweeps too
        )
    return out


def _featurizer_chains_raw(
    name: str, normalized: Mapping[str, Any]
) -> list[tuple[Any, list[str]]]:
    """Every (site, chain) a featurizer participates in."""
    used: list[tuple[Any, list[str]]] = []
    for section in ("reads", "writes"):
        for entry in normalized.get(section, {}).values():
            ref = entry.get("featurizer")
            chain = [ref] if isinstance(ref, str) else list(ref or [])
            if name in chain:
                pair = (entry.get("site"), chain)
                if pair not in used:
                    used.append(pair)
    return used


def _featurizer_entries_raw(
    name: str, normalized: Mapping[str, Any]
) -> list[Mapping[str, Any]]:
    """Every read/write entry whose chain names the featurizer."""
    used: list[Mapping[str, Any]] = []
    for section in ("reads", "writes"):
        for entry in normalized.get(section, {}).values():
            ref = entry.get("featurizer")
            chain = [ref] if isinstance(ref, str) else list(ref or [])
            if name in chain:
                used.append(entry)
    return used


def _window_length_raw(pos: Any, normalized: Mapping[str, Any]) -> int | None:
    """``schema.span_length`` over the canonical (raw-mapping) form: the number
    of positions a fixed prompt-frame ``span`` of two or more addresses on
    every row, or ``None`` — a named position is looked up, a sweep, an
    ``index``, ``all``, a variable/column, a span set, a ``generated`` or a
    ``scope``d / ``relative_to`` span all answer ``None``. The twin must move
    with ``span_length``: both answers are ``int``, so a change to what counts
    as a window that reaches one file and not the other is caught by no type
    checker, only by the offline and online widths disagreeing."""
    spec = normalized.get("positions", {}).get(pos) if isinstance(pos, str) else pos
    if not isinstance(spec, Mapping) or _is_sweep(spec):
        return None
    if spec.get("generated") is not None:
        return None
    if spec.get("scope") is not None or spec.get("relative_to") is not None:
        return None
    span = spec.get("span")
    if not isinstance(span, (list, tuple)) or len(span) != 2:
        return None
    a, b = span
    if not all(isinstance(v, int) and not isinstance(v, bool) for v in (a, b)):
        return None
    # `b - a` is the length because an unscoped span is non-negative (the
    # parser's rule); a right-anchored span, if §2.3 ever admits one, would
    # need the straddling case refused here as in `span_length`
    return b - a if b - a >= 2 else None


def _derived_window(name: str, normalized: Mapping[str, Any]) -> int | None:
    """A position gate's θ length (§2.5 ``axis``): the fixed window every
    entry using it — at every site it is named from — addresses. ``None``
    when any window is not derivable here (a swept or non-fixed ``pos``: rule
    4's ``_check_position_gates`` is the refusal for those); two lengths is
    the offline twin of that rule's "one gate, one window"."""
    lengths: set[int] = set()
    for entry in _featurizer_entries_raw(name, normalized):
        length = _window_length_raw(entry.get("pos"), normalized)
        if length is None:
            return None
        lengths.add(length)
    if len(lengths) > 1:
        raise ValidationError(
            4,
            f"position gate {name!r} is used over windows of lengths "
            f"{sorted(lengths)} — one gate, one window (§2.5 axis)",
            path=f"featurizers.{name}.axis",
        )
    return lengths.pop() if lengths else None


#: Featurizer kinds whose parameters are a **basis** over the feature axis, and
#: which therefore need that axis to be one. The rest (identity, standardize,
#: gate) act per column and are meaningful on any axis with a width.
_BASIS_FITTING_KINDS: frozenset[str] = frozenset({"subspace", "pca", "sae"})


def _raw_stage_output_width(spec: Mapping[str, Any], input_width: int) -> int | None:
    """The §2.5 chain rule on a raw featurizer entry (mirrors
    ``pytorch_hooks.featurizers.stage_output_width``)."""
    kind = spec.get("kind", "identity")
    if kind in ("subspace", "pca"):
        k = spec.get("k")
        return k if isinstance(k, int) else None
    if kind == "sae":
        return None
    return input_width


def _gate_map_arms(value: Any) -> list[str]:
    """The map names a raw gate entry's ``parametrization`` authors: the
    default when absent, the ``backward`` of the mapping form, every named arm
    of a sweep. An arm that is not a name (artifact-valued) is resolved at run
    time and reads as the default here, as it always has."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        if "backward" in value:
            return _gate_map_arms(value["backward"])
        if "sweep" in value and isinstance(value["sweep"], (list, tuple)):
            named = [v for v in value["sweep"] if isinstance(v, str)]
            return named or [GATE_DEFAULT_MAP]
    return [GATE_DEFAULT_MAP]


def _theta_shape(
    group: str | None,
    group_map: tuple[int, int] | None,
    width: int,
    arms: Sequence[str],
) -> list[int] | None:
    """``theta``'s shape under every authored map, when the arms agree on
    one — a sweep mixing an indexed map (``[1]``) with a per-unit one has no
    single shape to materialize, as a swept site has no single width."""
    shapes = {
        tuple(gate_param_shape(group, group_map, width, parametrization=arm))
        for arm in arms
    }
    return list(shapes.pop()) if len(shapes) == 1 else None


def _canon_featurizer(
    name: str,
    entry: Mapping[str, Any],
    normalized: Mapping[str, Any],
    info: ModelInfo | None,
    env: ResolutionEnv,
) -> dict[str, Any]:
    out = dict(entry)
    kind = out.setdefault("kind", "identity")
    out.setdefault("dtype", "fp32")
    if not isinstance(kind, str):
        return out  # swept kind: nothing derivable
    positional = kind == "gate" and out.get("axis") == "position"
    # §5.23 (group legality) is decided HERE, above the ``file_path`` return:
    # none of its clauses needs the width, and a loaded grouped gate — the
    # whole apply/replay surface — would otherwise reach the run unchecked.
    group = out.get("group")
    group_map: tuple[int, int] | None = None
    if kind == "gate" and isinstance(group, str) and info is not None:
        group_map = _derived_group_map(name, group, normalized, info)
    if isinstance(out.get("file_path"), str):
        # a loaded bundle: its params are its bytes — hash them (§7). No width
        # is recorded, but rule 4's one-width check over every site the name
        # is used at (§2.5, one name at several sites) runs for it as for a
        # fitted one — here, with no weights read, not at the build after
        # they are. The ranking-axis basis refusal stays the fitted path's: a
        # loaded basis applies
        # one it did not fit on the ranking axis. A position gate's one width
        # is one window (§2.5 `axis`), at every site it is named from.
        if info is not None and positional:
            _derived_window(name, normalized)
        elif info is not None:
            _derived_width(name, normalized, info, basis_check=False)
        out["content_digest"] = env.artifacts.file_digest(out["file_path"])
        return out
    init = out.get("init")
    if isinstance(init, Mapping) and isinstance(init.get("file_path"), str):
        # the basis a fit starts from is part of what the fit *is*: two runs
        # from two bases are two experiments, so its bytes enter the digest
        # exactly as a loaded featurizer's do (§7)
        out["init"] = {
            **init,
            "content_digest": env.artifacts.file_digest(init["file_path"]),
        }
    scores = init.get("from_scores") if isinstance(init, Mapping) else None
    if isinstance(scores, Mapping) and isinstance(scores.get("file_path"), str):
        # the same reasoning for a start read off a score table (§2.5
        # `init.from_scores`): a different table is a different start
        # the two column names are materialized to their defaults here, as
        # the parser does, so an authored default and an omitted one digest
        # identically (the METRIC_FIELD_DEFAULTS treatment)
        out["init"] = {
            **init,
            "from_scores": {
                **SCORES_INIT_DEFAULTS,
                **scores,
                "content_digest": env.artifacts.file_digest(scores["file_path"]),
            },
        }
    if info is None:
        return out
    if positional:
        # §2.5 `axis`: a position gate's θ is one entry per addressed position,
        # so its width is the window, not the site's feature width — which
        # may differ across the sites it is used at without contradiction
        width = _derived_window(name, normalized)
    else:
        width = _derived_width(name, normalized, info)
    if width is None:
        return out
    out["width"] = width
    arms = _gate_map_arms(out.get("parametrization")) if kind == "gate" else []
    grouped = isinstance(group, str) and group_map is not None
    theta_shape = (
        _theta_shape(
            group if grouped else None, group_map if grouped else None, width, arms
        )
        if kind == "gate" and (grouped or group is None)
        else None
    )
    if isinstance(scores, Mapping) and kind == "gate":
        # rule 32, the half decidable before any file is read: `keep` is a
        # count of this gate's units, which the width and the group map fix
        units = math.prod(theta_shape) if theta_shape is not None else None
        keep = scores.get("keep")
        if units is not None and isinstance(keep, int) and keep > units:
            raise ValidationError(
                32,
                f"featurizer {name!r}: init.from_scores.keep={keep} exceeds the "
                f"gate's {units} units",
                path=f"featurizers.{name}.init.from_scores.keep",
            )
    k = out.get("k")
    shapes: dict[str, list[int]] = {}
    if kind == "subspace" and isinstance(k, int):
        if not 0 < k <= width:
            raise ValidationError(
                4,
                f"featurizer {name!r}: k={k} exceeds the width {width}",
                path=f"featurizers.{name}.k",
            )
        shapes["weight"] = [width, k]
    elif kind == "pca" and isinstance(k, int):
        if not 0 < k <= width:
            raise ValidationError(
                4,
                f"featurizer {name!r}: k={k} exceeds the width {width}",
                path=f"featurizers.{name}.k",
            )
        shapes["weight"] = [width, k]
    elif kind == "gate":
        # a grouped gate has one θ per unit, not per coordinate (§2.5): the
        # map is derived above; ``None`` there means a swept site, and then —
        # as for the width — nothing is materialized. An indexed map's θ is
        # the one β, ``[1]``, and a sweep over maps that disagree on the shape
        # materializes none
        if theta_shape is not None:
            shapes["theta"] = theta_shape
    elif kind == "standardize":
        shapes["mu"] = [width]
        shapes["sigma"] = [width]
    if shapes and set(shapes) <= set(FEATURIZER_SLOTS.get(kind, ())):
        out["params"] = shapes
    return out


def _derived_width(
    name: str,
    normalized: Mapping[str, Any],
    info: ModelInfo,
    *,
    basis_check: bool = True,
) -> int | None:
    """The feature width of one featurizer, from the sites its reads/writes
    use (§2.5). Unmaterializable (None) when a needed field is swept;
    ambiguous multi-width use is an error — rule 4's "one name, one width",
    which a name used at several sites (one parameter set) is held to. Also
    the ranking-axis basis refusal (a basis *fitted* on a ranking component), unless
    ``basis_check`` is off: a loaded basis applies one it did not fit here."""
    widths: set[int] = set()
    for site_name, chain in _featurizer_chains_raw(name, normalized):
        if not isinstance(site_name, str):
            return None
        site = normalized.get("sites", {}).get(site_name)
        if site is None:
            return None
        component = site.get("component")
        head = site.get("head")
        if _is_sweep(component) or _is_sweep(head):
            return None
        if not isinstance(component, str):
            return None
        shape = component_shape(info, component)
        if shape.ranking and basis_check:
            # The ranking-axis basis refusal: dimensionally the axis has a width, so every featurizer used
            # to be accepted here — but column *k* is the *k*-th ranked expert,
            # a different expert for different tokens. A basis fitted across
            # positions is fitted across a basis that is itself shuffled per
            # position, so the fit means nothing even though it converges. Only
            # the kinds that *fit a basis* are refused; identity, standardize
            # and gate act per column and stay meaningful.
            declared = normalized.get("featurizers", {})
            fitting = [
                (member, declared.get(member, {}).get("kind"))
                for member in chain
                if declared.get(member, {}).get("kind") in _BASIS_FITTING_KINDS
            ]
            if fitting:
                member, kind = fitting[0]
                raise ValidationError(
                    4,
                    f"featurizer {member!r} ({kind!r}) fits a basis on site "
                    f"{site_name!r}, whose component {component!r} is not one: "
                    + shape.refusal(f"component {component!r}"),
                    path=f"featurizers.{member}",
                )
        running: int | None = component_width(
            info, component, head=head if isinstance(head, int) else None
        )
        for member in chain:
            if member == name:
                break
            member_spec = normalized.get("featurizers", {}).get(member, {})
            if running is None or _is_sweep(member_spec.get("k")):
                return None
            running = _raw_stage_output_width(member_spec, running)
        if running is None:
            return None
        widths.add(running)
    if not widths:
        return None
    if len(widths) > 1:
        raise ValidationError(
            4,
            f"featurizer {name!r} is used at sites of different widths "
            f"{sorted(widths)} — one featurizer, one width",
            path=f"featurizers.{name}",
        )
    return widths.pop()


def _derived_group_map(
    name: str, group: str, normalized: Mapping[str, Any], info: ModelInfo
) -> tuple[int, int] | None:
    """The group map of a grouped gate (§2.5) — ``(heads, head_dim)`` under
    ``head``, ``(num_experts, d_expert)`` under ``expert_neuron`` — from the
    sites its reads/writes use: the offline twin of the map the executor
    builds from the resolved site, and where rule 23 (group legality) is
    decided. ``None`` when nothing is derivable — a swept component, head or
    stage kind somewhere in the way, exactly as `_derived_width` treats a
    sweep.

    Refuses (§5.23) when the group *is* authored and a site it is used at
    cannot honour it: the gate is not the first stage of its chain (a grouped
    gate acts on the component's own coordinates — after any other stage, a
    rotation and a standardize alike, a coordinate no longer names a unit of
    the component), the site already selects a single member of what the group groups
    over, the component has no such axis, or two sites would give it different
    maps. The whole resolution reads the registry's declared axes, so it
    happens here, in the torch-free layer, with no model loaded.
    """
    derived: tuple[int, int] | None = None
    for site_name, chain in _featurizer_chains_raw(name, normalized):
        if chain[0] != name:
            raise ValidationError(
                23,
                f"featurizer {name!r} is grouped by {group!r} but follows "
                f"{chain[0]!r} in the chain {list(chain)} at site {site_name!r} — "
                "a grouped gate acts on the component's own coordinates, so it "
                "must be the first stage of its chain",
                path=f"featurizers.{name}.group",
            )
        if not isinstance(site_name, str):
            return None
        site = normalized.get("sites", {}).get(site_name)
        if site is None:
            return None
        component, head, expert = (
            site.get("component"),
            site.get("head"),
            site.get("expert"),
        )
        if _is_sweep(component) or _is_sweep(head) or _is_sweep(expert):
            return None
        if not isinstance(component, str):
            return None
        try:
            group_map = site_group_map(
                info,
                group,
                component,
                head=head if isinstance(head, int) else None,
                expert=expert if isinstance(expert, int) else None,
            )
        except ValidationError as err:
            raise ValidationError(
                23, err.message, path=f"featurizers.{name}.group"
            ) from err
        if derived is not None and group_map != derived:
            raise ValidationError(
                23,
                f"featurizer {name!r} is grouped by {group!r} at sites whose units "
                f"are laid out differently {sorted((derived, group_map))} — one "
                "featurizer, one parameter set, so one group map",
                path=f"featurizers.{name}.group",
            )
        derived = group_map
    return derived


def _canon_regularizer_names(names: Any) -> Any:
    """A regularizer's featurizer list is a set: sorted, and a one-name list
    is the name itself — so a list that names one featurizer canonicalizes to
    the form every earlier document wrote, and no digest moves (§7)."""
    if isinstance(names, list):
        ordered = sorted(names)
        return ordered[0] if len(ordered) == 1 else ordered
    return names


def _canon_term_fields(
    term: Mapping[str, Any], attached: Mapping[str, list[str]]
) -> dict[str, Any]:
    return {
        key: (
            _canon_regularizer_names(value)
            if key in REGULARIZER_KINDS
            else _canon_aggregation(value, attached)
            if key == "aggregation"
            else value  # `reduce` / `costs` / `read` / `model`: as authored
        )
        for key, value in term.items()
    }


def _canon_objective(objective: Any, attached: Mapping[str, list[str]]) -> Any:
    """Both spellings of ``train.objective`` (§2.11) with their regularizer
    lists and aggregations canonicalized; weights (possibly swept) pass
    through untouched."""
    if isinstance(objective, list):
        return [
            [term[0], _canon_term_fields(term[1], attached)]
            if isinstance(term, list)
            and len(term) == 2
            and isinstance(term[1], Mapping)
            else term
            for term in objective
        ]
    # the named form is the one a `constraint` reaches (the positional form
    # refuses it at parse): its nested block — `target`, `dual` — passes
    # through as authored, like `reduce` / `costs`, so an unauthored `dual.init`
    # materializes nothing and only a document that authors the block moves
    return {
        name: _canon_term_fields(term, attached) if isinstance(term, Mapping) else term
        for name, term in objective.items()
    }


def _canon_anneal(entry: Any) -> Any:
    if not isinstance(entry, Mapping):
        return entry
    shape = entry.get("shape", "linear")
    if shape == "linear" and set(entry) <= {"from", "to", "frac", "shape"}:
        return [entry["from"], entry["to"], entry["frac"]]
    return {
        "from": entry["from"],
        "to": entry["to"],
        "frac": entry["frac"],
        "shape": shape,
    }


def _canon_control(entry: Any) -> Any:
    if not isinstance(entry, Mapping):
        return entry
    out = dict(entry)
    if isinstance(out.get("signal"), Mapping):
        # one gate or several: the list is the canonical spelling, so `"g"` and
        # `["g"]` are one controller — the `layers` fold (§2.4), for a signal
        out["signal"] = {
            name: [target] if isinstance(target, str) else list(target)
            for name, target in out["signal"].items()
        }
    gains = dict(out.get("gains", {}))
    gains.setdefault("kd", CONTROL_DEFAULTS["kd"])
    out["gains"] = gains
    for field in ("space", "bounds", "d_clip"):
        out.setdefault(field, CONTROL_DEFAULTS[field])
    return out


def _canon_train(
    train: Mapping[str, Any],
    info: ModelInfo | None,
    env: ResolutionEnv,
    attached: Mapping[str, list[str]],
) -> dict[str, Any]:
    out = dict(train)
    out["objective"] = _canon_objective(out["objective"], attached)
    optimizer = dict(out["optimizer"])
    name = optimizer.get("name")
    if isinstance(name, str):
        for field, default in OPTIMIZER_DEFAULTS[name].items():
            optimizer.setdefault(field, default)
    out["optimizer"] = optimizer
    precision = dict(out.get("precision", {}))
    precision.setdefault("feature", "fp32")
    precision.setdefault("loss", "fp32")
    out["precision"] = precision
    if isinstance(out.get("anneal"), Mapping):
        # a linear schedule has two spellings (§2.11); the list is the canonical
        # one, so `{"from", "to", "frac"}` (shape unauthored or `linear`) digests
        # as `[from, to, frac]` and no document authored before the mapping
        # form existed moves. A geometric schedule keeps the mapping, `shape`
        # spelled, since the list has no place for it.
        out["anneal"] = {
            target: _canon_anneal(entry) for target, entry in out["anneal"].items()
        }
    if isinstance(out.get("phases"), list):
        # only when authored (a one-phase fit has no `phases`, so no digest
        # moves); a phase's anneal gets the top-level treatment above
        out["phases"] = [
            {
                **phase,
                **(
                    {
                        "anneal": {
                            target: _canon_anneal(entry)
                            for target, entry in phase["anneal"].items()
                        }
                    }
                    if isinstance(phase.get("anneal"), Mapping)
                    else {}
                ),
            }
            if isinstance(phase, Mapping)
            else phase
            for phase in out["phases"]
        ]
    if isinstance(out.get("control"), Mapping):
        # a controller's optional fields materialize like an optimizer's:
        # two spellings of one controller are one canonical form (§2.11)
        out["control"] = {
            target: _canon_control(entry) for target, entry in out["control"].items()
        }
    out.setdefault("seed", 0)
    if "eval" in out and isinstance(out["eval"], Mapping):
        eval_spec = dict(out["eval"])
        split = eval_spec.get("split")
        if isinstance(split, str):
            eval_spec["digest"] = env.datasets.digest(split)  # a dataset ref too (§2.2)
        if isinstance(eval_spec.get("aggregations"), Mapping):
            eval_spec["aggregations"] = {
                label: _canon_term_fields(entry, attached)
                if isinstance(entry, Mapping)
                else entry
                for label, entry in eval_spec["aggregations"].items()
            }
        out["eval"] = eval_spec
    return out
