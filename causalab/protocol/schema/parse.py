"""The strict parser: raw mapping → [`Document`][].

The raw loader ([`load_raw`][]), the parse helpers, every per-section
parser and the gate ([`check_protocol_version`][], [`parse_document`][]).
The shape rules it owns — strict keys, section order, the ``pos`` and
``"all"`` sugar, the ``sweep`` / ``artifact`` value wrappers — are stated on
the package ([`causalab.protocol.schema`][]); the vocabularies defined beside
their parsers here ([`SAVE_REDUCTIONS`][], [`DRAW_KINDS`][],
[`REGULARIZER_KINDS`][], …) are re-exported by it.
"""

from __future__ import annotations

import copy
import dataclasses

import json
import math
import re
import warnings
from typing import Any, Callable, Iterable, Mapping, Sequence

from causalab.tables import INPUT_COLUMN, inline_ref, is_inline_ref
from causalab.protocol.rules.errors import (
    ParseError,
    ProtocolWarning,
    ValidationError,
    suggest,
)
from causalab.protocol.estimand import (
    EstimandError,
    IDENTITY_COLUMNS,
    METRIC_UNITS,
    UNITS,
    metric_identity,
    parse_identifier,
)
from causalab.protocol.schema.types import (
    _QUANT_4BIT_FIELDS,  # pyright: ignore[reportPrivateUsage]
    _QUANT_INT8_FIELDS,  # pyright: ignore[reportPrivateUsage]
    ALL_POSITIONS,
    ArtifactRef,
    ATTENTION_IMPLEMENTATIONS,
    CodeSpec,
    COMPONENTS,
    ConstraintSpec,
    DataRole,
    DEPRECATED_COMPONENTS,
    Do,
    Document,
    GROUP_ORDER,
    HEADER_FIELDS,
    IMSpec,
    WRITES_DURING_GENERATION_FIELD,
    LAYERLESS_COMPONENTS,
    MATCH_MODES,
    MECHANISMS,
    METHOD_SECTIONS,
    METRIC_FIELD_DEFAULTS,
    METRIC_FIELDS,
    METRIC_KINDS,
    AggregationSpec,
    ReadRef,
    READ_TARGET_METRIC_KINDS,
    MIGRATABLE_PROTOCOL_VERSIONS,
    MINIMUM_COUNT_FIELD,
    ModelRef,
    ObjectiveTerm,
    OPTIMIZER_DEFAULTS,
    OPTIMIZER_FIELDS,
    OPTIMIZER_SCHEDULES,
    OPTIONAL_METRIC_FIELDS,
    ParamSpec,
    PRECISION_DTYPES,
    PROTOCOL_VERSION,
    QUANT_METHODS,
    QUANT_SCHEMES,
    QuantizationSpec,
    RAGGED_FIELD,
    RAGGED_POLICIES,
    ReadSpec,
    REQUIRED_METHOD_SECTIONS,
    RowRole,
    SAVE_KINDS,
    SaveEntry,
    SiteSpec,
    STREAMS,
    Sweep,
    TOKEN_COLUMN_METRIC_KINDS,
    RETIRED_TOKEN_FORM_REWRITES,
    RETIRED_TOKEN_FORMS,
    TOKEN_FORMS,
    TOP_K_RANKINGS,
    TrainSpec,
    TRAJECTORY_EVERY_UNITS,
    VOCAB_TOP_K_RANKING,
    WriteSpec,
    answer_columns,
)
from causalab.protocol.schema.featurizers import (
    ANNEAL_SHAPES,
    AnnealSchedule,
    CONTROL_KINDS,
    CONTROL_SIGNALS,
    CONTROL_SPACES,
    FEATURIZER_FIELD_CONDITIONS,
    FEATURIZER_FIELDS,
    FEATURIZER_KINDS,
    FeaturizerSpec,
    FORWARD_MASKS,
    GATE_AXES,
    GATE_DEAD_RULES,
    GATE_DEFAULT_MAP,
    GATE_GROUPS,
    GATE_MAPS,
    GATE_PARAMETRIZATIONS,
    K_SCHEDULE_KINDS,
    K_SCHEDULE_OF,
    PARAMETRIZATIONS,
    PHASE_UNTIL_UNITS,
    PhaseSpec,
)
from causalab.protocol.schema.positions import (
    is_span_object,
    parse_segments,
    parse_span_spec,
    PositionSpec,
)


# --------------------------------------------------------------------------- #
# raw loading
# --------------------------------------------------------------------------- #


#: The tail every protocol-3 spelling's refusal ends with (§1, §7).
_MIGRATE_HINT: str = "`causalab migrate <file>` rewrites a protocol-3 document"


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ParseError("P2", f"duplicate key {key!r} in one object")
        out[key] = value
    return out


def load_raw(text: str) -> dict[str, Any]:
    """Parse strict JSON text into an order-preserving mapping.

    YAML is accepted at the CLI surface (it parses to the same object model);
    this function is the JSON path and the normative behavior: duplicate keys
    and non-object top levels are errors, NaN/Infinity are rejected.
    """
    try:
        raw = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except ParseError:
        raise
    except json.JSONDecodeError as err:
        raise ParseError("P1", f"not valid JSON: {err}") from err
    if not isinstance(raw, dict):
        raise ParseError("P1", "the top level must be a JSON object")
    return raw


def _reject_constant(name: str) -> Any:
    raise ParseError("P1", f"non-finite JSON constant {name!r} is not allowed")


# --------------------------------------------------------------------------- #
# parse helpers
# --------------------------------------------------------------------------- #


def _require_mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ParseError(
            "P2", f"expected an object, got {type(value).__name__}", path=path
        )
    return value


def _check_keys(mapping: Mapping[str, Any], allowed: Iterable[str], path: str) -> None:
    allowed_set = set(allowed)
    for key in mapping:
        if key not in allowed_set:
            raise ParseError(
                "P3",
                f"unknown key {key!r}{suggest(key, allowed_set)}",
                path=path,
            )


def _enum(value: Any, options: Sequence[str], path: str) -> str:
    if not isinstance(value, str) or value not in options:
        raise ParseError(
            "P4",
            f"{value!r} is not one of {list(options)}"
            + (suggest(value, options) if isinstance(value, str) else ""),
            path=path,
        )
    return value


def _wrapped(
    value: Any,
    elem: Callable[[Any, str], Any],
    path: str,
    *,
    allow_sweep: bool = True,
) -> Any:
    """Parse a leaf that may be a ``{"sweep": …}`` wrapper around
    ``elem``-typed values (§3). Artifact references (§1) resolve *before*
    the parse gate (the pipeline's ``resolve`` stage runs ahead of ``gate``),
    so one reaching the parser is a pipeline misuse, not an authoring
    surface."""
    if isinstance(value, dict) and "sweep" in value:
        if not allow_sweep:
            raise ValidationError(14, "a sweep wrapper is not allowed here", path=path)
        _check_keys(value, ("sweep",), path)
        return _parse_sweep(value["sweep"], elem, path)
    if isinstance(value, dict) and isinstance(value.get("artifact"), str):
        raise ParseError(
            "P2",
            "unresolved artifact reference reached the parser — compile through "
            "causalab.protocol.pipeline.compile_protocol, which resolves artifact "
            "fields first",
            path=path,
        )
    return elem(value, path)


def _parse_sweep(spec: Any, elem: Callable[[Any, str], Any], path: str) -> Sweep:
    if isinstance(spec, dict):
        _check_keys(spec, ("range",), f"{path}.sweep")
        rng = spec.get("range")
        if (
            not isinstance(rng, list)
            or not 2 <= len(rng) <= 3
            or not all(isinstance(v, int) and not isinstance(v, bool) for v in rng)
        ):
            raise ValidationError(
                14,
                "sweep range must be [start, stop] or [start, stop, step] of integers",
                path=f"{path}.sweep",
            )
        start, stop = rng[0], rng[1]
        step = rng[2] if len(rng) == 3 else 1
        if step == 0:
            raise ValidationError(
                14, "sweep range step must be non-zero", path=f"{path}.sweep"
            )
        if len(range(start, stop, step)) > 1_000_000:  # O(1); before materializing
            raise ValidationError(
                14,
                "sweep range denotes over 1,000,000 values — refuse before "
                "materializing (§5.14)",
                path=f"{path}.sweep",
            )
        values = list(range(start, stop, step))
    elif isinstance(spec, list):
        values = spec
    else:
        raise ValidationError(
            14,
            f"a sweep wrapper takes a list or a range object, got {type(spec).__name__}",
            path=f"{path}.sweep",
        )
    if not values:
        raise ValidationError(
            14, "a sweep axis must have at least one value", path=path
        )
    return Sweep(
        values=tuple(elem(v, f"{path}.sweep[{i}]") for i, v in enumerate(values))
    )


#: Reductions a ``save`` entry may apply to a read (§2.12). Closed: the
#: vocabulary grows by PR, with a §2.12 row and a test, which is what the
#: docs↔code guard in ``tests/protocol/test_vocabulary_census.py`` enforces.
#:
#: All five collapse ``(rows, width)`` to ``(width,)``, so the un-reduced
#: harvest never reaches disk — that is the whole reason ``reduce`` exists.
#: ``mean`` and ``sum`` are the pair a sharded run needs (a weighted mean
#: across points is ``sum``s over ``count``s); ``std`` reports the spread the
#: mean hides; ``median`` survives an outlier row that the mean does not.
SAVE_REDUCTIONS: tuple[str, ...] = ("mean", "sum", "std", "median", "count")


def _entry_selector(
    value: Any, path: str, *, allow_slot: bool = False
) -> dict[str, Any]:
    """§2.5/§2.6 ``entry``: a mapping of coordinate name to scalar value,
    naming one entry inside a loaded bundle
    ([`causalab.protocol.bundles`][]). Names are the coordinate names as
    they appear in the producer's keys (``k``, ``seed``,
    ``target.layers``) — not full axis ids, which the consuming document has
    no reason to know."""
    obj = _require_mapping(value, path)
    if not obj:
        raise ParseError(
            "P2", "an 'entry' selector names at least one coordinate", path=path
        )
    selector: dict[str, Any] = {}
    for name, coord in obj.items():
        if name == "slot":
            if not allow_slot:
                raise ParseError(
                    "P2",
                    "a featurizer bundle's slots are fixed by its kind — "
                    "'slot' selects only inside a params bundle",
                    path=f"{path}.slot",
                )
            if not isinstance(coord, str):
                raise ParseError("P2", "'slot' names one tensor", path=f"{path}.slot")
            selector[name] = coord
            continue
        if isinstance(coord, (dict, list)):
            raise ParseError(
                "P2",
                f"entry coordinate {name!r} must be a scalar — a bundle key "
                "records one value per coordinate",
                path=f"{path}.{name}",
            )
        selector[name] = coord
    return selector


def _scalar_str(value: Any, path: str) -> str:
    if not isinstance(value, str):
        raise ParseError(
            "P2", f"expected a string, got {type(value).__name__}", path=path
        )
    return value


def _scalar_int(value: Any, path: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ParseError(
            "P2", f"expected an integer, got {type(value).__name__}", path=path
        )
    return value


def _scalar_number(value: Any, path: str) -> float | int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ParseError(
            "P2", f"expected a number, got {type(value).__name__}", path=path
        )
    return value


def _scalar_bool(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise ParseError("P2", f"expected true or false (got {value!r})", path=path)
    return value


def _int_list(value: Any, path: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not all(
        isinstance(v, int) and not isinstance(v, bool) for v in value
    ):
        raise ParseError("P2", "expected a list of integers", path=path)
    return tuple(value)


def _str_list(value: Any, path: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ParseError("P2", "expected a list of strings", path=path)
    return tuple(value)


def _any_leaf(value: Any, path: str) -> Any:
    return value


def _token_list(value: Any, path: str) -> tuple[str, ...]:
    """``token_logits.tokens`` (§2.10): a non-empty list of literal token
    strings, each listed once and none of them empty.

    Every string is tokenized as written, so ``"X"`` and ``" X"`` are two
    answers (two gpt2 rows) and both may be listed; only a string repeated
    letter for letter is refused here, torch-free. An empty string names no
    token. Two *different* strings that a tokenizer maps to one id can only
    be seen with the tokenizer in hand, and are refused where the metric
    resolves them ([`causalab.neural.shared.metrics`][]).
    """
    tokens = _str_list(value, path)
    if not tokens:
        raise ParseError(
            "P2", "expected at least one token string — nothing to save", path=path
        )
    seen: set[str] = set()
    for token in tokens:
        if token == "":
            raise ParseError(
                "P2",
                "an empty string names no token — a token_logits entry is the "
                "answer's text as the model emits it (§2.10)",
                path=path,
            )
        if token in seen:
            raise ParseError(
                "P2",
                f"{token!r} is listed twice — list each answer once (§2.10)",
                path=path,
            )
        seen.add(token)
    return tokens


# --------------------------------------------------------------------------- #
# section parsers
# --------------------------------------------------------------------------- #


def _parse_model(raw: Any, path: str) -> ModelRef:
    obj = _require_mapping(raw, path)
    _check_keys(
        obj, ("key", "revision", "dtype", "quantization", "attn_implementation"), path
    )
    if "key" not in obj:
        raise ParseError("P2", "model needs a 'key'", path=path)
    key = _wrapped(obj["key"], _scalar_str, f"{path}.key")
    revision = (
        _wrapped(obj["revision"], _scalar_str, f"{path}.revision")
        if "revision" in obj
        else "main"
    )
    dtype = (
        _wrapped(
            obj["dtype"],
            lambda v, p: _enum(v, PRECISION_DTYPES, p),
            f"{path}.dtype",
        )
        if "dtype" in obj
        else None
    )
    quantization = (
        _parse_quantization(obj["quantization"], f"{path}.quantization")
        if "quantization" in obj
        else None
    )
    attn_implementation = (
        _wrapped(
            obj["attn_implementation"],
            lambda v, p: _enum(v, ATTENTION_IMPLEMENTATIONS, p),
            f"{path}.attn_implementation",
        )
        if "attn_implementation" in obj
        else None
    )
    return ModelRef(
        key=key,
        revision=revision,
        dtype=dtype,
        quantization=quantization,
        attn_implementation=attn_implementation,
    )


def _parse_quantization(raw: Any, path: str) -> QuantizationSpec:
    obj = _require_mapping(raw, path)
    _check_keys(
        obj,
        ("scheme", "method", "compute_dtype", *_QUANT_4BIT_FIELDS, *_QUANT_INT8_FIELDS),
        path,
    )
    if "scheme" not in obj:
        raise ParseError(
            "P2",
            f"quantization needs a 'scheme' — one of {list(QUANT_SCHEMES)}",
            path=path,
        )
    return QuantizationSpec(
        scheme=_wrapped(
            obj["scheme"], lambda v, p: _enum(v, QUANT_SCHEMES, p), f"{path}.scheme"
        ),
        method=_wrapped(
            obj["method"], lambda v, p: _enum(v, QUANT_METHODS, p), f"{path}.method"
        )
        if "method" in obj
        else "bitsandbytes",
        compute_dtype=_wrapped(
            obj["compute_dtype"],
            lambda v, p: _enum(v, PRECISION_DTYPES, p),
            f"{path}.compute_dtype",
        )
        if "compute_dtype" in obj
        else None,
        double_quant=_wrapped(obj["double_quant"], _scalar_bool, f"{path}.double_quant")
        if "double_quant" in obj
        else None,
        int8_threshold=_wrapped(
            obj["int8_threshold"], _scalar_number, f"{path}.int8_threshold"
        )
        if "int8_threshold" in obj
        else None,
    )


#: §2.2 ``draw.kind``: how a fit picks one member of a list-valued
#: counterfactual column per row per epoch. Closed; censused against §2.2.
DRAW_KINDS: tuple[str, ...] = ("uniform",)

_INDEXED_FIELD = re.compile(r"\[\d+\]$")


def _parse_inputs(value: Any, path: str) -> tuple[str, ...]:
    """A role's ``inputs`` (§2.2): a non-empty list of prompt strings, none of
    them empty. Not sweepable, and not a wrapper — the rows are what the
    document *is*, so the list is read as a literal."""
    if isinstance(value, dict) and "sweep" in value:
        raise ValidationError(
            14,
            "'inputs' is not swept — an inline table is the document's data, "
            "not a knob; put the prompts in a table and sweep 'dataset' instead",
            path=path,
        )
    inputs = _str_list(value, path)
    if not inputs:
        raise ParseError("P2", "'inputs' names at least one prompt", path=path)
    for i, text in enumerate(inputs):
        if not text:
            raise ParseError(
                "P2", "an empty prompt names no input", path=f"{path}[{i}]"
            )
    return inputs


def _parse_data_role(raw: Any, path: str, *, shuffleable: bool) -> DataRole:
    obj = _require_mapping(raw, path)
    _check_keys(obj, ("dataset", "inputs", "field", "shuffle", "draw"), path)
    if ("dataset" in obj) == ("inputs" in obj):
        raise ParseError(
            "P2",
            "a data role reads either a 'dataset' (with its 'field') or inline "
            "'inputs', not both and not neither (§2.2)",
            path=path,
        )
    if "inputs" in obj:
        return _parse_inline_role(obj, path, shuffleable=shuffleable)
    if "field" not in obj:
        raise ParseError("P2", "data role needs a 'field'", path=path)
    if isinstance(obj["dataset"], str) and is_inline_ref(obj["dataset"]):
        raise ParseError(
            "P2",
            f"{obj['dataset']!r} is a derived ref (§7); a document inlines its "
            "inputs as 'inputs': [...], never by naming the ref",
            path=f"{path}.dataset",
        )
    shuffle: dict[str, Any] | None = None
    if "shuffle" in obj:
        if not shuffleable:
            raise ParseError(
                "P2",
                "'shuffle' permutes a counterfactual role; the base role is the "
                "population and is never permuted (§2.2)",
                path=f"{path}.shuffle",
            )
        shuffle = _parse_shuffle(obj["shuffle"], f"{path}.shuffle")
    field_leaf = _wrapped(obj["field"], _scalar_str, f"{path}.field")
    draw: dict[str, Any] | None = None
    if "draw" in obj:
        if not shuffleable:
            raise ParseError(
                "P2",
                "'draw' samples a counterfactual role's members; the base role is "
                "the population and has one input per row (§2.2)",
                path=f"{path}.draw",
            )
        draw = _parse_draw(obj["draw"], f"{path}.draw")
        if isinstance(field_leaf, str) and _INDEXED_FIELD.search(field_leaf):
            raise ParseError(
                "P2",
                f"a drawn role names its list column bare — {field_leaf!r} already "
                "picks one member, so there is nothing to draw; drop the index "
                "(the fixed member every non-training forward reads is 'draw.eval')",
                path=f"{path}.field",
            )
    return DataRole(
        dataset=_wrapped(obj["dataset"], _scalar_str, f"{path}.dataset"),
        field=field_leaf,
        shuffle=shuffle,
        draw=draw,
    )


def _parse_inline_role(
    obj: Mapping[str, Any], path: str, *, shuffleable: bool
) -> DataRole:
    """An inline role (§2.2): ``{"inputs": [...]}`` denotes a one-column table
    ([`INPUT_COLUMN`][], split ``"all"``) registered under
    its content digest, and the role reads that column. ``field`` is implied
    and so refused; ``draw`` needs a list-valued column an inline table does
    not have; ``shuffle`` permutes rows as it does for any counterfactual."""
    if "field" in obj:
        raise ParseError(
            "P2",
            f"an inline role has one column, {INPUT_COLUMN!r}, and reads it; "
            "drop 'field'",
            path=f"{path}.field",
        )
    if "draw" in obj:
        raise ParseError(
            "P2",
            "'draw' samples a list-valued column; an inline role has one prompt "
            "per row — put the members in a table column to draw from them",
            path=f"{path}.draw",
        )
    shuffle: dict[str, Any] | None = None
    if "shuffle" in obj:
        if not shuffleable:
            raise ParseError(
                "P2",
                "'shuffle' permutes a counterfactual role; the base role is the "
                "population and is never permuted (§2.2)",
                path=f"{path}.shuffle",
            )
        shuffle = _parse_shuffle(obj["shuffle"], f"{path}.shuffle")
    inputs = _parse_inputs(obj["inputs"], f"{path}.inputs")
    return DataRole(dataset=inline_ref(inputs), field=INPUT_COLUMN, shuffle=shuffle)


def _parse_draw(raw: Any, path: str) -> dict[str, Any]:
    """``draw: {kind: uniform, eval?: j}`` (§2.2). ``kind`` is closed and not
    sweepable (what is drawn is what the document *is*, not a knob); ``eval``
    is the member index every non-training forward reads, a non-negative
    integer that is not a ``bool``, absent for ``0``."""
    obj = _require_mapping(raw, path)
    _check_keys(obj, ("kind", "eval"), path)
    if "kind" not in obj:
        raise ParseError(
            "P2", f"'draw' needs a 'kind' — one of {list(DRAW_KINDS)}", path=path
        )
    kind = _wrapped(
        obj["kind"],
        lambda v, p: _enum(_scalar_str(v, p), DRAW_KINDS, p),
        f"{path}.kind",
        allow_sweep=False,
    )
    out: dict[str, Any] = {"kind": kind}
    if "eval" in obj:
        member = _wrapped(obj["eval"], _scalar_int, f"{path}.eval", allow_sweep=False)
        if int(member) < 0:
            raise ParseError(
                "P2",
                f"'eval' indexes the list column — a non-negative integer, got {member!r}",
                path=f"{path}.eval",
            )
        out["eval"] = member
    return out


def _parse_shuffle(raw: Any, path: str) -> dict[str, Any]:
    """``shuffle: {seed: <int>}`` (§2.2) — the seed is the permutation's only
    input, an integer that is not a ``bool``, and not sweepable: one document
    is one pairing, and a swept seed would make the control document differ
    from its target by an axis rather than by one field."""
    obj = _require_mapping(raw, path)
    _check_keys(obj, ("seed",), path)
    if "seed" not in obj:
        raise ParseError(
            "P2", "'shuffle' needs a 'seed' — the permutation's only input", path=path
        )
    seed = _wrapped(obj["seed"], _scalar_int, f"{path}.seed", allow_sweep=False)
    return {"seed": seed}


def to_base_form(raw: Any, path: str = "data") -> dict[str, Any]:
    """§2.2: a ``data`` block that names no role *is* the base role.

    Role names earn their place only when a counterfactual is present, so
    ``{"dataset": "weekdays/data#train", "field": "input"}`` reads as
    ``{"base": {...}}``. The conversion runs before the parse and before
    canonicalization (``explicit._canon_data``), so both spellings share one
    canonical form and one digest — the int-position sugar precedent (§6.1,
    §7). A block that names ``counterfactual`` without ``base`` is refused
    rather than wrapped: the author started naming roles and stopped."""
    obj = _require_mapping(raw, path)
    if "base" in obj:
        return obj
    if "counterfactual" in obj:
        raise ParseError(
            "P2",
            "a data block with a counterfactual names its roles: put the "
            "original input under 'base'",
            path=path,
        )
    return {"base": obj}


def _parse_data(raw: Any, path: str) -> dict[str, DataRole | tuple[DataRole, ...]]:
    obj = to_base_form(raw, path)
    _check_keys(obj, ("base", "counterfactual"), path)
    if "base" not in obj:
        raise ParseError("P2", "data needs a 'base' role", path=path)
    out: dict[str, DataRole | tuple[DataRole, ...]] = {
        "base": _parse_data_role(obj["base"], f"{path}.base", shuffleable=False)
    }
    if "counterfactual" in obj:
        cf = obj["counterfactual"]
        if isinstance(cf, list):
            out["counterfactual"] = tuple(
                _parse_data_role(s, f"{path}.counterfactual[{j}]", shuffleable=True)
                for j, s in enumerate(cf)
            )
        else:
            out["counterfactual"] = _parse_data_role(
                cf, f"{path}.counterfactual", shuffleable=True
            )
    return out


def _parse_position_spec(raw: Any, path: str) -> PositionSpec:
    if isinstance(raw, int) and not isinstance(raw, bool):
        return PositionSpec(index=raw)  # §6.1 int sugar
    if raw == ALL_POSITIONS:
        return PositionSpec(all=True)  # §6.1 "all" sugar
    obj = _require_mapping(raw, path)
    # §2.3 spans: any span key dispatches to schema/positions.py's
    # parse_span_spec, which owns the algebra's grammar (sets, unions,
    # intersections, predicates, `atomic`).
    if is_span_object(obj):
        return parse_span_spec(
            obj,
            path,
            parse_position=_parse_position_spec,
            parse_anchor_ref=_parse_anchor_ref,
        )
    _check_keys(
        obj,
        (
            "index",
            "span",
            "variable",
            "column",
            "all",
            "scope",
            "relative_to",
            "generated",
            "alignment",
        ),
        path,
    )
    anchors = [k for k in ("index", "span", "variable", "column", "all") if k in obj]
    if len(anchors) != 1:
        raise ParseError(
            "P2",
            "a position spec needs exactly one of "
            f"index/span/variable/column/all, got {anchors}",
            path=path,
        )
    if "all" in obj and obj["all"] is not True:
        raise ParseError(
            "P2",
            f'all is the flag {{"all": true}} — got {obj["all"]!r}; there is no '
            "other all-positions selection to spell",
            path=path,
        )
    index = (
        _wrapped(obj["index"], _scalar_int, f"{path}.index") if "index" in obj else None
    )
    span = _wrapped(obj["span"], _parse_span, f"{path}.span") if "span" in obj else None
    variable = (
        _wrapped(obj["variable"], _scalar_str, f"{path}.variable")
        if "variable" in obj
        else None
    )
    column = (
        _wrapped(obj["column"], _scalar_str, f"{path}.column")
        if "column" in obj
        else None
    )
    every = True if "all" in obj else None
    scope_ref = (
        _wrapped(obj["scope"], _parse_anchor_ref, f"{path}.scope")
        if "scope" in obj
        else None
    )
    relative_ref = (
        _wrapped(obj["relative_to"], _parse_anchor_ref, f"{path}.relative_to")
        if "relative_to" in obj
        else None
    )
    anchor_ref = scope_ref if scope_ref is not None else relative_ref
    anchor_source = anchor_ref[0] if anchor_ref is not None else "variable"
    scope = scope_ref[1] if scope_ref is not None else None
    relative_to = relative_ref[1] if relative_ref is not None else None
    if (scope is not None or relative_to is not None) and (
        variable is not None or column is not None or every is not None
    ):
        raise ParseError(
            "P2",
            "scope/relative_to modify an index or span, not a variable/column/all spec",
            path=path,
        )
    if scope is not None and relative_to is not None:
        raise ParseError(
            "P2", "scope and relative_to are mutually exclusive", path=path
        )
    generated = (
        _parse_generated(obj["generated"], f"{path}.generated")
        if "generated" in obj
        else None
    )
    # A declared cardinality is a string and never swept (a sweep over how an
    # address pairs is not a sweep over anything the model sees). Membership
    # in ALIGNMENT_CARDINALITIES and fit to the address are rule 26's
    # (`validate`), so one rule names the field for every way it can be wrong.
    alignment = (
        _wrapped(obj["alignment"], _scalar_str, f"{path}.alignment", allow_sweep=False)
        if "alignment" in obj
        else None
    )
    if generated is not None:
        # The continuation frame carries no prompt-frame notions: a `column`
        # holds a substring of the *input* text, and scope/relative_to anchor
        # on a prompt variable's token run. Both are meaningless in a frame
        # the prompt does not contain (§2.3).
        # …except a scope naming the `continuation` segment, which *is* the
        # frame `generated` selects (§2.2.1): the name and the frame agree.
        in_continuation = anchor_source == "segment" and scope == "continuation"
        offenders = [
            key
            for key, present in (
                ("column", column is not None),
                ("scope", scope is not None and not in_continuation),
                ("relative_to", relative_to is not None),
            )
            if present
        ]
        if offenders:
            raise ParseError(
                "P2",
                f"{offenders} resolve against the prompt, so they cannot combine "
                "with 'generated' — anchor inside the continuation with "
                "index/span/variable/all instead",
                path=path,
            )
    if isinstance(span, tuple):
        lo, hi = span
        if scope is None and (lo < 0 or hi <= lo):
            raise ParseError(
                "P2",
                f"span [{lo}, {hi}) is not a forward window — unscoped spans are "
                "content-frame, non-negative, non-empty",
                path=path,
            )
        if (
            scope is not None
            and (lo < 0) == (hi < 0 or hi == 0 and lo < 0)
            and lo >= hi
        ):
            raise ParseError(
                "P2", f"scoped span [{lo}, {hi}) is statically empty", path=path
            )
    return PositionSpec(
        index=index,
        span=span,
        variable=variable,
        column=column,
        all=every,
        scope=scope,
        relative_to=relative_to,
        anchor_source=anchor_source,
        generated=generated,
        alignment=alignment,
    )


def _parse_generated(value: Any, path: str) -> dict[str, Any]:
    """§2.3 — the continuation frame selector: ``{"max_new_tokens": n}``.

    A mapping rather than a bare int on purpose: stopping conditions
    (``stop``, ``min_new_tokens``) join this object later without any
    ambiguity about what a bare number would have meant.
    """
    obj = _require_mapping(value, path)
    _check_keys(obj, ("max_new_tokens",), path)
    if "max_new_tokens" not in obj:
        raise ParseError(
            "P2", "generated needs 'max_new_tokens' — the decode budget", path=path
        )
    budget = _wrapped(obj["max_new_tokens"], _scalar_int, f"{path}.max_new_tokens")
    if isinstance(budget, int) and budget < 1:
        raise ParseError(
            "P2",
            f"max_new_tokens is {budget} — a continuation frame needs at least "
            "one generated token",
            path=f"{path}.max_new_tokens",
        )
    return {"max_new_tokens": budget}


def _parse_span(value: Any, path: str) -> tuple[int, int]:
    ints = _int_list(value, path)
    if len(ints) != 2:
        raise ParseError("P2", "a span is [a, b) — exactly two integers", path=path)
    return (ints[0], ints[1])


def _parse_anchor_ref(value: Any, path: str) -> tuple[str, str]:
    """A ``scope``/``relative_to`` anchor: ``(source, name)`` where source is
    ``"variable"`` (per-role prompt variable), ``"column"`` (a top-level row
    column) or ``"segment"`` (a declared segment, §2.2.1) — §2.3."""
    obj = _require_mapping(value, path)
    _check_keys(obj, ("variable", "column", "segment"), path)
    named = [key for key in ("variable", "column", "segment") if key in obj]
    if len(named) != 1 or not isinstance(obj[named[0]], str):
        raise ParseError(
            "P2",
            'expected {"variable": "<name>"}, {"column": "<name>"} or '
            '{"segment": "<name>"}',
            path=path,
        )
    return named[0], obj[named[0]]


def _parse_positions(
    raw: Any, path: str
) -> dict[str, PositionSpec | Sweep | ArtifactRef]:
    obj = _require_mapping(raw, path)
    return {
        name: _wrapped(value, _parse_position_spec, f"{path}.{name}")
        for name, value in obj.items()
    }


def _current_component(value: Any) -> Any:
    """Fold a retired component spelling onto the name that replaced it.

    Done here rather than in canonicalization so that *nothing* downstream ever
    sees the old name: the canonical form, the digest and every table are in one
    vocabulary, and the alias is a parse-time courtesy with no second code path
    behind it.
    """
    return DEPRECATED_COMPONENTS.get(value, value) if isinstance(value, str) else value


def _band(value: Any, path: str) -> tuple[int, ...]:
    """§2.4 ``layers``: the band a site spans, as a tuple of layer indices.

    Authored as a non-empty list of integers in strictly increasing order —
    a band is a *set* of layers, so a repeated or unsorted index is a typo,
    not a second spelling. A bare index is the one-layer band ``[n]``: it is
    what an axis over ``layers`` (``{"sweep": {"range": [0, 32]}}``,
    ``{"at_once": …}``) and a workflow ``emit`` hand a point, and the
    canonical form writes the list either way (``explicit._canon_site``), so
    the two spellings carry one digest. ``true``/``false`` are refused as
    they are everywhere an integer is expected.
    """
    if isinstance(value, int) and not isinstance(value, bool):
        return (value,)
    if not isinstance(value, list):
        raise ParseError(
            "P2",
            "'layers' is a band: a list of layer indices, [18] for one layer "
            f"(got {type(value).__name__})",
            path=path,
        )
    if not value:
        raise ParseError(
            "P2",
            "'layers' names at least one layer — an empty band is no address",
            path=path,
        )
    for i, item in enumerate(value):
        if not isinstance(item, int) or isinstance(item, bool):
            raise ParseError(
                "P2",
                f"expected an integer layer index, got {type(item).__name__}",
                path=f"{path}[{i}]",
            )
    if any(b <= a for a, b in zip(value, value[1:])):
        raise ParseError(
            "P2",
            f"'layers' is a band, listed in strictly increasing order — got "
            f"{value}" + (" (a layer repeats)" if len(set(value)) < len(value) else ""),
            path=path,
        )
    return tuple(value)


def _parse_site(raw: Any, path: str) -> SiteSpec:
    obj = _require_mapping(raw, path)
    if "layer" in obj:
        # the protocol_version 2 spelling — named, with the rename and the
        # verb that carries it, rather than left to `suggest`'s guess
        raise ParseError(
            "P3",
            "unknown key 'layer' — did you mean 'layers'? protocol_version 3 "
            "renamed a site's depth index to 'layers', a band of layer "
            "indices ([18] for one layer, §2.4); `causalab migrate <file>` "
            "rewrites a protocol_version 2 document",
            path=path,
        )
    _check_keys(obj, ("component", "layers", "head", "expert", "stream"), path)
    if "component" not in obj:
        raise ParseError("P2", "a site needs a 'component'", path=path)
    component = _wrapped(
        obj["component"],
        lambda v, p: _enum(_current_component(v), COMPONENTS, p),
        f"{path}.component",
    )
    layers = (
        _wrapped(obj["layers"], _band, f"{path}.layers") if "layers" in obj else None
    )
    if isinstance(component, str):  # un-swept: layer presence is checkable now
        if component in LAYERLESS_COMPONENTS and layers is not None:
            raise ParseError("P2", f"{component} is layer-less", path=path)
        if component not in LAYERLESS_COMPONENTS and layers is None:
            raise ParseError("P2", f"{component} needs 'layers'", path=path)
    return SiteSpec(
        component=component,
        layers=layers,
        head=_wrapped(obj["head"], _scalar_int, f"{path}.head")
        if "head" in obj
        else None,
        expert=_wrapped(obj["expert"], _scalar_int, f"{path}.expert")
        if "expert" in obj
        else None,
        stream=_wrapped(
            obj["stream"], lambda v, p: _enum(v, STREAMS, p), f"{path}.stream"
        )
        if "stream" in obj
        else None,
    )


def _parse_sites(raw: Any, path: str) -> dict[str, SiteSpec]:
    obj = _require_mapping(raw, path)
    return {name: _parse_site(value, f"{path}.{name}") for name, value in obj.items()}


def _authored_gate_maps(parametrization: Any) -> list[str]:
    """The concrete map names a gate's ``parametrization`` authors —
    [`GATE_DEFAULT_MAP`][] when absent, every string arm of a sweep — so a
    legality check holds each arm. An artifact-valued arm is resolved at run
    time and is not checked here."""
    if parametrization is None:
        return [GATE_DEFAULT_MAP]
    values = (
        list(parametrization.values)
        if isinstance(parametrization, Sweep)
        else [parametrization]
    )
    return [v for v in values if isinstance(v, str)]


def _is_sweep_like(value: Mapping[str, Any]) -> bool:
    """A `{"sweep": …}` wrapper (§3), which `_wrapped` owns — as opposed to a
    field's own mapping form. `{"axis": …}` is lowered to a sweep before the
    parse gate (§3.2), so the second test is defensive: one that reached here
    is refused by `_wrapped`'s callback as not swept, not read as a form."""
    return "sweep" in value or "axis" in value


def _parse_featurizer(raw: Any, path: str) -> FeaturizerSpec:
    obj = _require_mapping(raw, path)
    kind_raw = obj.get("kind", "identity")
    kind = _wrapped(
        kind_raw,
        lambda v, p: _enum(v, FEATURIZER_KINDS, p),
        f"{path}.kind",
    )
    kind_key = kind if isinstance(kind, str) else "identity"
    allowed = {"kind", "file_path", "entry", "dtype", "description"} | set(
        FEATURIZER_FIELDS.get(kind_key, frozenset())
    )
    _check_keys(obj, allowed, path)
    if "entry" in obj and "file_path" not in obj:
        raise ParseError(
            "P2",
            "'entry' selects inside a loaded bundle — it needs a file_path",
            path=path,
        )
    parametrization = None
    forward = None
    if (
        "parametrization" in obj
        and isinstance(obj["parametrization"], Mapping)
        and not _is_sweep_like(obj["parametrization"])
    ):
        # §2.5 the mapping form: {"forward": "hard", "backward": <map>} — a
        # gate's straight-through split of the forward and backward masks
        p_param = f"{path}.parametrization"
        if kind_key != "gate":
            raise ParseError(
                "P2",
                "the mapping form of 'parametrization' splits a gate's forward and "
                "backward masks — "
                + (
                    f"a {kind_key!r} has one rotation map"
                    if isinstance(kind, str)
                    else "a swept 'kind' cannot take it (the form is a gate's)"
                ),
                path=p_param,
            )
        mapping = obj["parametrization"]
        _check_keys(mapping, ("forward", "backward"), p_param)
        for field in ("forward", "backward"):
            if field not in mapping:
                raise ParseError(
                    "P2",
                    'the mapping form is {"forward": "hard", "backward": <map>} — '
                    f"{field!r} is missing",
                    path=p_param,
                )
        forward_raw = mapping["forward"]
        if forward_raw in ("soft", "sampled"):
            raise ParseError(
                "P2",
                f"forward {forward_raw!r} is the map's own forward — spell the map "
                "alone ('sampled' is hard_concrete itself); the mapping form is for "
                "'hard'",
                path=f"{p_param}.forward",
            )
        forward = _enum(
            _scalar_str(forward_raw, f"{p_param}.forward"),
            FORWARD_MASKS,
            f"{p_param}.forward",
        )
        if isinstance(mapping["backward"], Mapping) and "sweep" in mapping["backward"]:
            raise ParseError(
                "P2",
                'the mapping form of \'parametrization\' ({"forward", "backward"}) '
                "is not swept — author one document per arm (§2.5)",
                path=f"{p_param}.backward",
            )
        parametrization = _enum(
            _scalar_str(mapping["backward"], f"{p_param}.backward"),
            GATE_PARAMETRIZATIONS,
            f"{p_param}.backward",
        )
    elif "parametrization" in obj:
        # one field, one meaning, an enum per kind: a subspace's rotation map
        # or a gate's theta→mask map (§2.5); a swept kind admits either
        vocabulary = (
            GATE_PARAMETRIZATIONS
            if kind_key == "gate"
            else PARAMETRIZATIONS
            if isinstance(kind, str)
            else (*PARAMETRIZATIONS, *GATE_PARAMETRIZATIONS)
        )

        def _one_map(v: Any, p: str) -> str:
            if isinstance(v, Mapping):
                # the mapping form is not a leaf, so a sweep arm may not be one:
                # an ablation grid is one document per forward/backward pair
                raise ParseError(
                    "P2",
                    "the mapping form of 'parametrization' ({\"forward\", "
                    '"backward"}) is not swept — author one document per arm '
                    "(§2.5)",
                    path=p,
                )
            return _enum(v, vocabulary, p)

        parametrization = _wrapped(
            obj["parametrization"], _one_map, f"{path}.parametrization"
        )
    # §2.5's conditional legality, read off FEATURIZER_FIELD_CONDITIONS: a
    # field authored in a state (fitted or loaded) or under a map the table
    # does not list is refused in the table's own words. A swept map is legal
    # only if every arm is — compile would refuse the whole run on the
    # offending arm anyway (every expanded point is validated before weights
    # load), and naming the arm here is the better message.
    loaded = "file_path" in obj
    maps = _authored_gate_maps(parametrization) if kind_key == "gate" else None
    for field, legality in FEATURIZER_FIELD_CONDITIONS.items():
        if field not in obj:
            continue
        legal = legality.legal(loaded=loaded)
        if legal is None:
            continue
        if not legal:
            why = legality.why_loaded if loaded else legality.why_fit
            raise ParseError("P2", why, path=f"{path}.{field}")
        if maps is not None:
            outside = [m for m in maps if m not in legal]
            if outside:
                named = ", ".join(repr(m) for m in outside)
                raise ParseError(
                    "P2", legality.why_map.format(maps=named), path=f"{path}.{field}"
                )
    init_raw = obj.get("init")
    if (
        maps is not None
        and isinstance(init_raw, Mapping)
        and "from_scores" in init_raw
        and (indexed := [m for m in maps if GATE_MAPS[m].indexed])
    ):
        # a score table ranks theta's units and puts each at a pole; an
        # indexed map has one β and no unit to place (§2.5)
        named = ", ".join(repr(m) for m in indexed)
        raise ParseError(
            "P2",
            "init.from_scores places theta's units at the map's poles by score "
            f"— this gate maps theta through {named}, one β over an ordered "
            "basis with no unit to place; start it with a fill or a saved theta "
            "(§2.5)",
            path=f"{path}.init.from_scores",
        )
    group = None
    if "group" in obj:
        group = _wrapped(
            obj["group"],
            lambda v, p: _enum(v, GATE_GROUPS, p),
            f"{path}.group",
        )
    axis = None
    if "axis" in obj:
        axis = _wrapped(
            obj["axis"],
            lambda v, p: _enum(v, GATE_AXES, p),
            f"{path}.axis",
            allow_sweep=False,
        )
        if "group" in obj:
            raise ParseError(
                "P2",
                "a position gate is one θ per addressed position over every "
                "coordinate — already the scalar `group: site` would name — so "
                "`axis` and `group` do not combine (§2.5)",
                path=f"{path}.group",
            )
        if "pool" in obj:
            raise ParseError(
                "P2",
                "a budget pool over position gates is not written down (§2.5) — "
                "pool feature gates, or budget one position gate alone",
                path=f"{path}.pool",
            )
    temperature = (
        _wrapped(obj["temperature"], _positive_number, f"{path}.temperature")
        if "temperature" in obj
        else None
    )
    stretch = (
        _parse_stretch(obj["stretch"], f"{path}.stretch") if "stretch" in obj else None
    )
    k_schedule = None
    stop_grad_shift = None
    pool = None
    if kind_key == "gate":
        assert maps is not None
        if "pool" in obj and loaded and "top_k" not in obj:
            raise ParseError(
                "P2",
                "'pool' on a loaded gate is a pooled readout — one 'top_k' cut "
                "through the members' joint ranking — so it needs a top_k (§2.5)",
                path=f"{path}.pool",
            )
        ranked = [m for m in maps if GATE_MAPS[m].ranked]
        if ranked and not loaded and "k_schedule" not in obj:
            raise ParseError(
                "P2",
                "a budget gate draws its per-step budget from 'k_schedule' (§2.5) "
                "— {'kind': 'fixed', 'k': n} or {'kind': 'uniform' | "
                "'log_uniform', 'low': a, 'high': b, 'eval': n}",
                path=path,
            )
        if "k_schedule" in obj:
            k_schedule = _parse_k_schedule(obj["k_schedule"], f"{path}.k_schedule")
        if "stop_grad_shift" in obj:
            stop_grad_shift = _wrapped(
                obj["stop_grad_shift"], _scalar_bool, f"{path}.stop_grad_shift"
            )
        if "pool" in obj:
            # a name, never swept: the members of a pool are whoever authors
            # the same string, and a sweep over names would be a sweep over
            # which gates share a budget — a different document per arm
            pool = _scalar_str(obj["pool"], f"{path}.pool")
            if not pool:
                raise ParseError(
                    "P2", "a budget pool needs a name", path=f"{path}.pool"
                )
    dead = _parse_gate_dead(obj["dead"], f"{path}.dead") if "dead" in obj else None
    return FeaturizerSpec(
        kind=kind,
        k=_wrapped(obj["k"], _scalar_int, f"{path}.k") if "k" in obj else None,
        parametrization=parametrization,
        init=_parse_featurizer_init(obj["init"], f"{path}.init", kind_key)
        if "init" in obj
        else None,
        seed=_wrapped(obj["seed"], _scalar_int, f"{path}.seed")
        if "seed" in obj
        else None,
        group=group,
        temperature=temperature,
        stretch=stretch,
        dead=dead,
        top_k=_wrapped(obj["top_k"], _non_negative_int, f"{path}.top_k")
        if "top_k" in obj
        else None,
        k_schedule=k_schedule,
        stop_grad_shift=stop_grad_shift,
        pool=pool,
        axis=axis,
        forward=forward,
        dtype=_wrapped(
            obj["dtype"], lambda v, p: _enum(v, PRECISION_DTYPES, p), f"{path}.dtype"
        )
        if "dtype" in obj
        else None,
        file_path=_wrapped(obj["file_path"], _scalar_str, f"{path}.file_path")
        if "file_path" in obj
        else None,
        entry=_wrapped(obj["entry"], _entry_selector, f"{path}.entry")
        if "entry" in obj
        else None,
        description=obj.get("description"),
    )


def _scalar_bool(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise ParseError(
            "P2", f"expected true or false, got {type(value).__name__}", path=path
        )
    return value


def _parse_k_schedule(raw: Any, path: str) -> dict[str, Any]:
    """§2.5 ``k_schedule`` of a ``budget`` gate: how each optimizer step draws
    its budget ``k``. ``{"kind": "fixed", "k": n}`` — the same cut every step;
    ``{"kind": "uniform" | "log_uniform", "low": a, "high": b}`` — an integer
    drawn from ``[a, b]`` per step (``log_uniform`` needs ``a ≥ 1``: the draw
    is ``round(exp(U(log a, log b)))``). ``eval`` is the cut the fit's own
    held-out pass scores and its ``hard_mask_size`` counts: defaults to ``k``
    under ``fixed`` and is **required** under a sampled kind, since a sampled
    schedule implies no single number. ``k`` and ``eval`` may be swept; the
    bounds are one schedule. ``of`` ([`K_SCHEDULE_OF`][]) says what every
    number here counts — patched units (absent, the default) or kept ones."""
    obj = _require_mapping(raw, path)
    _check_keys(obj, ("kind", "k", "low", "high", "eval", "of"), path)
    if "kind" not in obj:
        raise ParseError(
            "P2",
            f"k_schedule needs a 'kind': one of {list(K_SCHEDULE_KINDS)}",
            path=path,
        )
    kind = _enum(obj["kind"], K_SCHEDULE_KINDS, f"{path}.kind")
    out: dict[str, Any] = {"kind": kind}
    if kind == "fixed":
        if "k" not in obj or "low" in obj or "high" in obj:
            raise ParseError(
                "P2", "a fixed k_schedule names 'k' and no bounds", path=path
            )
        out["k"] = _wrapped(obj["k"], _non_negative_int, f"{path}.k")
    else:
        if "k" in obj or "low" not in obj or "high" not in obj:
            raise ParseError(
                "P2",
                f"a {kind} k_schedule names 'low' and 'high' and no 'k'",
                path=path,
            )
        low = _non_negative_int(obj["low"], f"{path}.low")
        high = _non_negative_int(obj["high"], f"{path}.high")
        if low > high:
            raise ParseError(
                "P2",
                f"k_schedule bounds are ordered, got low={low} > high={high}",
                path=path,
            )
        if kind == "log_uniform" and low < 1:
            raise ParseError(
                "P2",
                "a log_uniform k_schedule draws round(exp(U(log low, log high))) — "
                f"low must be at least 1, got {low}",
                path=f"{path}.low",
            )
        out["low"], out["high"] = low, high
        if "eval" not in obj:
            raise ParseError(
                "P2",
                f"a {kind} k_schedule samples its budget, so it names 'eval': the "
                "cut the fit's held-out pass scores and its hard_mask_size counts",
                path=path,
            )
    if "eval" in obj:
        out["eval"] = _wrapped(obj["eval"], _non_negative_int, f"{path}.eval")
    if "of" in obj:
        out["of"] = _enum(obj["of"], K_SCHEDULE_OF, f"{path}.of")
    return out


def _non_negative_int(value: Any, path: str) -> int:
    number = _scalar_int(value, path)
    if number < 0:
        raise ParseError(
            "P2", f"expected a non-negative integer, got {number}", path=path
        )
    return number


def _positive_number(value: Any, path: str) -> float | int:
    number = _scalar_number(value, path)
    if number <= 0:
        raise ParseError("P2", f"expected a positive number, got {number}", path=path)
    return number


def _parse_stretch(raw: Any, path: str) -> tuple[float, float]:
    """``stretch: [γ, ζ]`` of a hard-concrete gate: the interval the concrete
    sample is stretched onto before clipping to ``[0, 1]``, so ``γ < 0 < 1 < ζ``
    — a stretch that does not cover the unit interval has no mass on the poles
    and its hard split would never be reached."""
    if (
        not isinstance(raw, list)
        or len(raw) != 2
        or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in raw)
    ):
        raise ParseError("P2", "stretch is a two-number list [γ, ζ]", path=path)
    lo, hi = float(raw[0]), float(raw[1])
    if not lo < 0.0 < 1.0 < hi:
        raise ParseError(
            "P2",
            f"stretch [γ, ζ] must satisfy γ < 0 < 1 < ζ, got [{lo}, {hi}]",
            path=path,
        )
    return (lo, hi)


def _parse_gate_dead(raw: Any, path: str) -> dict[str, float | int]:
    """§2.5 ``dead``: one of [`GATE_DEAD_RULES`][], never both — a frozen
    unit takes no gradient and a leaking one exists to keep taking it, so the
    two answers to a dead unit contradict each other on the same gate.
    ``freeze_after`` is a positive integer count of consecutive hard-off
    steps; ``leak`` is a gradient slope strictly inside ``(0, 1)`` — ``0`` is
    no rule and ``1`` makes the mask's backward that of an unsquashed θ.
    Neither is sweepable: a
    sweep over how a unit dies is a sweep over the training *procedure*, and
    the two rules' results are not points on one axis."""
    obj = _require_mapping(raw, path)
    if not obj:
        raise ParseError(
            "P2",
            '\'dead\' names a rule: {"freeze_after": n} or {"leak": eps} (§2.5)',
            path=path,
        )
    _check_keys(obj, set(GATE_DEAD_RULES), path)
    if len(obj) != 1:
        raise ParseError(
            "P2",
            "'dead' names exactly one rule — a frozen unit takes no gradient and "
            "a leaking one exists to keep taking it, so 'freeze_after' and 'leak' "
            "cannot both hold on one gate (§2.5)",
            path=path,
        )
    if "freeze_after" in obj:
        n = obj["freeze_after"]
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            raise ParseError(
                "P2",
                f"'freeze_after' counts consecutive hard-off optimizer steps — a "
                f"positive integer, got {n!r}",
                path=f"{path}.freeze_after",
            )
        return {"freeze_after": int(n)}
    eps = obj["leak"]
    if (
        isinstance(eps, bool)
        or not isinstance(eps, (int, float))
        or not 0.0 < eps < 1.0
    ):
        raise ParseError(
            "P2",
            f"'leak' is a gradient slope strictly inside (0, 1), got {eps!r} — "
            "0 is no rule and 1 is the backward of an unsquashed theta",
            path=f"{path}.leak",
        )
    return {"leak": float(eps)}


def _parse_featurizer_init(raw: Any, path: str, kind: str) -> dict[str, Any]:
    """§2.5 ``init`` — where a fit **starts**.

    On a ``subspace``: ``{"file_path": <basis bundle>, "entry": <selector>}``,
    ``entry`` optional with the same semantics as the featurizer's own.
    Neither field is sweepable — a basis is one artifact, and which of its
    columns seed the fit follows from ``k``.

    On a ``gate``: either ``{"fill": p}`` — every unit starts at **mask value**
    ``p ∈ [0, 1]``, which is ``θ = logit(p)`` under ``sigmoid`` and ``θ = p``
    under ``clamp``, the one number that means the same under both maps, and
    sweepable (whether the start decides the mask is a real question) — or
    the ``file_path`` form, a saved ``theta`` taken verbatim as the start (an
    earlier fit, a trajectory checkpoint, a hand-built prior); or
    ``{"from_scores": …}``, a start read off a per-unit **score table**
    (`_parse_scores_init`). Exactly one of the three."""
    obj = _require_mapping(raw, path)
    keys = (
        ("file_path", "entry", "fill", "from_scores")
        if kind == "gate"
        else ("file_path", "entry")
    )
    _check_keys(obj, keys, path)
    starts = [key for key in ("fill", "file_path", "from_scores") if key in obj]
    if len(starts) > 1 or ("entry" in obj and starts and "file_path" not in obj):
        # `entry` belongs to the file_path start; beside another start it is
        # two starts spelled at once
        raise ParseError(
            "P2",
            "'init' names one start: {'fill': p}, {'file_path': …, 'entry': …} "
            "or {'from_scores': …} — not both",
            path=path,
        )
    if "fill" in obj:
        return {"fill": _wrapped(obj["fill"], _mask_value, f"{path}.fill")}
    if "from_scores" in obj:
        return {
            "from_scores": _parse_scores_init(obj["from_scores"], f"{path}.from_scores")
        }
    if "file_path" not in obj:
        raise ParseError(
            "P2",
            "'init' names the start to fit from — it needs a file_path"
            + (" (or, on a gate, a fill or from_scores)" if kind == "gate" else ""),
            path=path,
        )
    init: dict[str, Any] = {
        "file_path": _scalar_str(obj["file_path"], f"{path}.file_path")
    }
    if "entry" in obj:
        init["entry"] = _entry_selector(obj["entry"], f"{path}.entry")
    return init


#: §2.5 ``init.from_scores`` — the keys, and the two the table is read by
#: when unauthored. ``unit`` and ``value`` default to the column names a
#: per-unit table conventionally carries; naming them is what lets a
#: ``head_stats.json`` (``head`` / ``mean``) seed a head gate without a
#: rewrite. They are materialized into the parsed form, so an authored
#: default and an omitted one digest identically.
SCORES_INIT_KEYS: tuple[str, ...] = (
    "file_path",
    "unit",
    "value",
    "where",
    "keep",
    "scale",
)
SCORES_INIT_DEFAULTS: dict[str, str] = {"unit": "unit", "value": "value"}


def _parse_scores_init(raw: Any, path: str) -> dict[str, Any]:
    """§2.5 ``init.from_scores`` — a gate's start read off a **score table**:
    a saved metric table (one row per unit, a JSON list of row objects) such
    as ``causalab.analysis.head_stats`` writes, or an attribution scan's own
    output. ``file_path`` names it; ``unit`` is the column (or, for a
    two-axis theta such as ``expert_neuron``'s, the list of columns) holding
    each row's unit index; ``value`` the column holding its score; ``where``
    an equality filter (``{"layer": 15}``) that picks this gate's rows out of
    a table over several sites. Exactly one of ``keep`` — the top-``keep``
    units by score start on the kept pole of the gate's map, the rest on the
    dropped pole (the ``random_mask`` convention), the attribution- or
    magnitude-pruning baseline as one document — or ``scale`` — ``theta`` is
    the table's z-scored values times ``scale``, centred on the midpoint mask,
    an attribution-initialised fit (the first SGD step of a mask *is*
    path-weighted IG). Both are sweepable: how many units
    a ranking needs is the question a ``keep`` sweep asks. The coverage checks
    — ``keep`` at most the unit count, every unit named exactly once — are
    rule 32, decided where the width and the table are known."""
    obj = _require_mapping(raw, path)
    _check_keys(obj, SCORES_INIT_KEYS, path)
    if "file_path" not in obj:
        raise ParseError(
            "P2", "from_scores names the score table: it needs a file_path", path=path
        )
    out: dict[str, Any] = {
        "file_path": _scalar_str(obj["file_path"], f"{path}.file_path")
    }
    unit = obj.get("unit", SCORES_INIT_DEFAULTS["unit"])
    if isinstance(unit, list):
        if not unit or not all(isinstance(column, str) for column in unit):
            raise ParseError(
                "P2",
                "from_scores.unit is a column name, or a non-empty list of them "
                "(one per axis of the gate's theta)",
                path=f"{path}.unit",
            )
        out["unit"] = list(unit)
    else:
        out["unit"] = _scalar_str(unit, f"{path}.unit")
    out["value"] = _scalar_str(
        obj.get("value", SCORES_INIT_DEFAULTS["value"]), f"{path}.value"
    )
    if "where" in obj:
        where = _require_mapping(obj["where"], f"{path}.where")
        for column, literal in where.items():
            if isinstance(literal, bool) or not isinstance(literal, (str, int, float)):
                raise ParseError(
                    "P2",
                    "from_scores.where maps column names to the scalar each row "
                    f"must equal, got {literal!r}",
                    path=f"{path}.where.{column}",
                )
        out["where"] = dict(where)
    modes = [key for key in ("keep", "scale") if key in obj]
    if len(modes) != 1:
        raise ParseError(
            "P2",
            "from_scores reads the table one way: 'keep' (the top-k units start "
            "kept) or 'scale' (theta is the z-scored score times scale) — exactly "
            "one of the two",
            path=path,
        )
    if "keep" in obj:
        out["keep"] = _wrapped(obj["keep"], _positive_int, f"{path}.keep")
    else:
        out["scale"] = _wrapped(obj["scale"], _positive_number, f"{path}.scale")
    return out


def _positive_int(value: Any, path: str) -> int:
    number = _scalar_int(value, path)
    if number < 1:
        raise ParseError("P2", f"expected a positive integer, got {number}", path=path)
    return number


def _mask_value(value: Any, path: str) -> float:
    """A gate's ``init.fill`` (§2.5): a number in ``[0, 1]``, the mask value
    every unit starts at. The endpoints are legal here — a clamp gate may
    start fully patched, and so may a hard-concrete gate, whose start
    ``logit((fill − γ)/(ζ − γ))`` is finite at both poles — and a sigmoid gate,
    whose start is ``logit(fill)``, refuses them at build where the map is
    known."""
    number = _scalar_number(value, path)
    if not 0.0 <= float(number) <= 1.0:
        raise ParseError(
            "P2",
            f"a gate's init.fill is a mask value in [0, 1], got {number!r}",
            path=path,
        )
    return float(number)


def _parse_featurizers(raw: Any, path: str) -> dict[str, FeaturizerSpec]:
    obj = _require_mapping(raw, path)
    return {
        name: _parse_featurizer(value, f"{path}.{name}") for name, value in obj.items()
    }


def _parse_param(raw: Any, path: str) -> ParamSpec:
    obj = _require_mapping(raw, path)
    _check_keys(obj, ("file_path", "entry", "shape", "init", "description"), path)
    loaded = "file_path" in obj
    trainable = "shape" in obj or "init" in obj
    if "entry" in obj and not loaded:
        raise ParseError(
            "P2",
            "'entry' selects inside a loaded bundle — it needs a file_path",
            path=path,
        )
    if loaded == trainable:
        raise ParseError(
            "P2",
            "a params entry is either loaded (file_path) or trainable (shape + init)",
            path=path,
        )
    if trainable and not ("shape" in obj and "init" in obj):
        raise ParseError(
            "P2", "a trainable params entry needs both shape and init", path=path
        )
    return ParamSpec(
        file_path=_wrapped(obj["file_path"], _scalar_str, f"{path}.file_path")
        if loaded
        else None,
        entry=_wrapped(
            obj["entry"],
            lambda v, p: _entry_selector(v, p, allow_slot=True),
            f"{path}.entry",
        )
        if "entry" in obj
        else None,
        shape=_wrapped(obj["shape"], _int_list, f"{path}.shape")
        if "shape" in obj
        else None,
        init=_wrapped(obj["init"], _scalar_str, f"{path}.init")
        if "init" in obj
        else None,
        description=obj.get("description"),
    )


def _parse_params(raw: Any, path: str) -> dict[str, ParamSpec]:
    obj = _require_mapping(raw, path)
    return {name: _parse_param(value, f"{path}.{name}") for name, value in obj.items()}


#: Fields of a ``code`` entry the loader derives and no one may author (§6):
#: the resolved module and its source hash, the manifest and hash of its
#: declared import closure (present only when the module imports a sibling
#: outside the ``causalab`` package), and the content digests of the declared
#: data inputs.
DERIVED_CODE_FIELDS: tuple[str, ...] = (
    "source_module",
    "source_sha256",
    "closure",
    "closure_sha256",
    "data_input_digests",
)


def _parse_row_roles(raw: Any, path: str) -> tuple[RowRole, ...]:
    """§2.8.1 — the batch's row convention, in batch order. A list, not a
    mapping: ``[clean, corrupted]`` and ``[corrupted, clean]`` are different
    conventions and JSON objects are unordered."""
    if not isinstance(raw, list):
        raise ParseError(
            "P2",
            "row_roles is a list of {'role': …, 'rows': n} in batch order",
            path=path,
        )
    roles: list[RowRole] = []
    seen: set[str] = set()
    for index, item in enumerate(raw):
        where = f"{path}[{index}]"
        obj = _require_mapping(item, where)
        _check_keys(obj, ("role", "rows"), where)
        for field in ("role", "rows"):
            if field not in obj:
                raise ParseError("P2", f"a row role needs {field!r}", path=where)
        role = _scalar_str(obj["role"], f"{where}.role")
        rows = _scalar_int(obj["rows"], f"{where}.rows")
        if rows < 1:
            raise ParseError(
                "P2",
                f"row role {role!r} covers {rows} rows; it must be ≥ 1",
                path=f"{where}.rows",
            )
        if role in seen:
            raise ParseError("P2", f"duplicate row role {role!r}", path=where)
        seen.add(role)
        roles.append(RowRole(role=role, rows=rows))
    if not roles:
        raise ParseError(
            "P2",
            "row_roles is empty — omit it to say nothing about the rows, "
            "rather than saying there are none",
            path=path,
        )
    return tuple(roles)


def _parse_code_entry(raw: Any, path: str) -> CodeSpec:
    obj = _require_mapping(raw, path)
    for field in DERIVED_CODE_FIELDS:
        if field in obj:
            raise ParseError(
                "P5",
                f"{field!r} is derived from the resolved source and stamped at "
                "load, never authored (§6)",
                path=f"{path}.{field}",
            )
    _check_keys(
        obj,
        ("locator", "args", "data_inputs", "env_inputs", "row_roles", "description"),
        path,
    )
    if "locator" not in obj:
        raise ParseError("P2", "a code entry needs a 'locator'", path=path)
    args = obj.get("args", {})
    if not isinstance(args, Mapping):
        raise ParseError(
            "P2", "args is a JSON object of keyword values", path=f"{path}.args"
        )
    data_inputs = obj.get("data_inputs", {})
    if not isinstance(data_inputs, Mapping) or not all(
        isinstance(v, str) for v in data_inputs.values()
    ):
        raise ParseError(
            "P2",
            "data_inputs maps a name to a file path",
            path=f"{path}.data_inputs",
        )
    description = obj.get("description")
    if description is not None and not isinstance(description, str):
        raise ParseError("P2", "description is free text", path=f"{path}.description")
    return CodeSpec(
        locator=_wrapped(obj["locator"], _scalar_str, f"{path}.locator"),
        args=dict(args),
        data_inputs=dict(data_inputs),
        env_inputs=tuple(_str_list(obj["env_inputs"], f"{path}.env_inputs"))
        if "env_inputs" in obj
        else (),
        row_roles=_parse_row_roles(obj["row_roles"], f"{path}.row_roles")
        if "row_roles" in obj
        else (),
        description=description,
    )


def _parse_code(raw: Any, path: str) -> dict[str, CodeSpec]:
    obj = _require_mapping(raw, path)
    return {
        name: _parse_code_entry(value, f"{path}.{name}") for name, value in obj.items()
    }


def _parse_pos_field(value: Any, path: str) -> Any:
    """A read/write ``pos``: a positions-table name or an inline spec. The
    bare string ``"all"`` is the all-positions sugar, never a name — it is
    reserved (§5.3), so no entry can be declared under it."""
    if isinstance(value, str) and value != ALL_POSITIONS:
        return value
    return _parse_position_spec(value, path)


def _parse_featurizer_ref(value: Any, path: str) -> Any:
    """A ``featurizer`` reference: one name or a composition list (§2.5)."""
    if isinstance(value, str):
        return value
    return _str_list(value, path)


def _parse_read(raw: Any, path: str) -> ReadSpec:
    obj = _require_mapping(raw, path)
    for retired in ("model", "input"):
        if retired in obj:
            raise ParseError(
                "P3",
                f"a read carries no {retired!r} under protocol 4: a read is an "
                "address, and the models that take it list it in "
                "intervened_models.<model>.reads (§2.7, §2.9) — " + _MIGRATE_HINT,
                path=f"{path}.{retired}",
            )
    _check_keys(obj, ("site", "pos", "featurizer", "dims"), path)
    for field in ("site", "pos"):
        if field not in obj:
            raise ParseError("P2", f"a read needs {field!r}", path=path)
    return ReadSpec(
        site=_wrapped(obj["site"], _scalar_str, f"{path}.site"),
        pos=_wrapped(obj["pos"], _parse_pos_field, f"{path}.pos"),
        featurizer=_wrapped(
            obj["featurizer"], _parse_featurizer_ref, f"{path}.featurizer"
        )
        if "featurizer" in obj
        else None,
        dims=_wrapped(obj["dims"], _int_list, f"{path}.dims")
        if "dims" in obj
        else None,
    )


def _parse_reads(raw: Any, path: str) -> dict[str, ReadSpec]:
    obj = _require_mapping(raw, path)
    return {name: _parse_read(value, f"{path}.{name}") for name, value in obj.items()}


def _parse_read_ref(value: Any, path: str) -> ReadRef:
    """A reference to a read as taken on a model (§2.7): the bare name, legal
    when exactly one model lists the read (bound by [`resolve_read_refs`][]),
    or ``{"read": name, "model": model}``."""
    if isinstance(value, str):
        return ReadRef(value, None)
    obj = _require_mapping(value, path)
    _check_keys(obj, ("read", "model"), path)
    for field in ("read", "model"):
        if field not in obj:
            raise ParseError(
                "P2",
                f'a read reference is a read name or {{"read": …, "model": …}} '
                f"— this one lacks {field!r}",
                path=path,
            )
    return ReadRef(
        _scalar_str(obj["read"], f"{path}.read"),
        _scalar_str(obj["model"], f"{path}.model"),
    )


def _parse_operand(value: Any, path: str) -> Any:
    """A write operand (§2.8): a read reference (a bare read or param name,
    or ``{"read", "model"}``), or a literal scalar. A bare name stays a
    string here — whether it is a read, a param or a slot is resolved once
    the whole document is parsed ([`resolve_read_refs`][])."""
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        return _parse_read_ref(value, path)
    return _scalar_number(value, path)


def _parse_do(raw: Any, path: str) -> Do:
    obj = _require_mapping(raw, path)
    if len(obj) != 1:
        raise ParseError("P2", "'do' has exactly one mechanism key", path=path)
    ((mech, payload),) = obj.items()
    if mech not in MECHANISMS:
        raise ParseError(
            "P4", f"unknown mechanism {mech!r}{suggest(mech, MECHANISMS)}", path=path
        )
    p = f"{path}.{mech}"
    if mech == "swap":
        return Do(mechanism=mech, payload=_wrapped(payload, _parse_operand, p))
    if mech in ("add_scaled", "lerp"):
        options = _require_mapping(payload, p)
        _check_keys(options, ("op", "alpha"), p)
        for field in ("op", "alpha"):
            if field not in options:
                raise ParseError("P2", f"{mech} needs {field!r}", path=p)
        return Do(
            mechanism=mech,
            payload={
                "op": _wrapped(options["op"], _parse_operand, f"{p}.op"),
                "alpha": _wrapped(options["alpha"], _parse_operand, f"{p}.alpha"),
            },
        )
    if mech == "affine":
        options = _require_mapping(payload, p)
        _check_keys(options, ("A", "b"), p)
        for field in ("A", "b"):
            if field not in options:
                raise ParseError("P2", f"affine needs {field!r}", path=p)
        return Do(
            mechanism=mech,
            payload={
                "A": _wrapped(options["A"], _scalar_str, f"{p}.A"),
                "b": _wrapped(options["b"], _scalar_str, f"{p}.b"),
            },
        )
    if mech == "gaussian":
        options = _require_mapping(payload, p)
        _check_keys(options, ("seed", "scale", "axis"), p)
        for field in ("seed", "scale", "axis"):
            if field not in options:
                raise ParseError("P2", f"gaussian needs {field!r}", path=p)
        return Do(
            mechanism=mech,
            payload={
                "seed": _wrapped(options["seed"], _scalar_int, f"{p}.seed"),
                "scale": _wrapped(options["scale"], _scalar_number, f"{p}.scale"),
                "axis": _wrapped(
                    options["axis"],
                    lambda v, pp: _enum(v, ("tp_duplicated", "tp_split"), pp),
                    f"{p}.axis",
                ),
            },
        )
    if mech == "renormalize":
        if payload is not True:
            raise ParseError(
                "P2", 'renormalize is written {"renormalize": true}', path=p
            )
        return Do(mechanism=mech, payload=True)
    if mech == "clamp":
        options = _require_mapping(payload, p)
        _check_keys(options, ("lo", "hi"), p)
        for field in ("lo", "hi"):
            if field not in options:
                raise ParseError("P2", f"clamp needs {field!r}", path=p)
        return Do(
            mechanism=mech,
            payload={
                "lo": _wrapped(options["lo"], _scalar_number, f"{p}.lo"),
                "hi": _wrapped(options["hi"], _scalar_number, f"{p}.hi"),
            },
        )
    # pytorch_fn — names a `code` declaration, never a bare qualname (§2.8.1).
    # A qualname alone left the function's body, arguments, file reads,
    # environment reads and row convention outside the digest; the
    # declaration is what carries them.
    options = _require_mapping(payload, p)
    if "qualname" in options:
        raise ParseError(
            "P3",
            "pytorch_fn no longer takes a bare 'qualname': write "
            '{"pytorch_fn": {"code": "<name>"}} and declare the function in '
            "the document's 'code' section, so its source, arguments, "
            "declared inputs and row roles are in the digest (§2.8.1)",
            path=f"{p}.qualname",
        )
    _check_keys(options, ("code",), p)
    if "code" not in options:
        raise ParseError(
            "P2",
            "pytorch_fn names a 'code' declaration — {\"pytorch_fn\": "
            '{"code": "<name>"}} — so the function\'s source, arguments, '
            "declared inputs and row roles are in the digest (§2.8.1)",
            path=p,
        )
    return Do(
        mechanism=mech,
        payload={"code": _wrapped(options["code"], _scalar_str, f"{p}.code")},
    )


def _parse_ragged(raw: Any, path: str) -> str:
    """``{"policy": <RAGGED_POLICIES>}`` (§2.8): the one-key object a write's
    ``ragged`` field holds. Vocabulary only — whether the window *is* ragged
    is the executor's to decide on the encoded batch (§5 rule 19)."""
    obj = _require_mapping(raw, path)
    _check_keys(obj, ("policy",), path)
    if "policy" not in obj:
        raise ParseError(
            "P2",
            f"a {RAGGED_FIELD!r} declaration needs 'policy' "
            f"(one of {list(RAGGED_POLICIES)})",
            path=path,
        )
    return _enum(obj["policy"], RAGGED_POLICIES, f"{path}.policy")


def _parse_write(raw: Any, path: str) -> WriteSpec:
    obj = _require_mapping(raw, path)
    _check_keys(obj, ("site", "pos", "featurizer", "dims", "do", RAGGED_FIELD), path)
    for field in ("site", "pos", "do"):
        if field not in obj:
            raise ParseError("P2", f"a write needs {field!r}", path=path)
    ragged: str | None = None
    if RAGGED_FIELD in obj:
        # how a ragged window lands is an execution strategy, fixed per
        # campaign like `minimum_count` — never a research variable, so a
        # sweep wrapper is rule 14 here
        ragged = _wrapped(
            obj[RAGGED_FIELD],
            _parse_ragged,
            f"{path}.{RAGGED_FIELD}",
            allow_sweep=False,
        )
    return WriteSpec(
        site=_wrapped(obj["site"], _scalar_str, f"{path}.site"),
        pos=_wrapped(obj["pos"], _parse_pos_field, f"{path}.pos"),
        do=_parse_do(obj["do"], f"{path}.do"),
        featurizer=_wrapped(
            obj["featurizer"], _parse_featurizer_ref, f"{path}.featurizer"
        )
        if "featurizer" in obj
        else None,
        dims=_wrapped(obj["dims"], _int_list, f"{path}.dims")
        if "dims" in obj
        else None,
        ragged=ragged,
    )


def _parse_writes(raw: Any, path: str) -> dict[str, WriteSpec]:
    obj = _require_mapping(raw, path)
    return {name: _parse_write(value, f"{path}.{name}") for name, value in obj.items()}


def _parse_im(raw: Any, path: str) -> IMSpec:
    obj = _require_mapping(raw, path)
    _check_keys(obj, ("input", "reads", "writes", WRITES_DURING_GENERATION_FIELD), path)
    for field in ("input", "reads"):
        if field not in obj:
            raise ParseError(
                "P2",
                f"an intervened_model needs {field!r}"
                + (
                    " — the reads taken on it, at least one: a model nobody "
                    "reads runs a forward nobody observes (§2.9)"
                    if field == "reads"
                    else ""
                ),
                path=path,
            )
    # the reads a model takes are what it *is for*; never swept — two read
    # sets are two models
    reads = _wrapped(obj["reads"], _str_list, f"{path}.reads", allow_sweep=False)
    if not reads:
        raise ParseError(
            "P2",
            "an intervened_model lists at least one read — a model nobody "
            "reads runs a forward nobody observes (§2.9)",
            path=f"{path}.reads",
        )
    during_generation = False
    if WRITES_DURING_GENERATION_FIELD in obj:
        # all-or-nothing for the model, and never swept: whether the writes
        # stay in force through the decode is a property of the intervention,
        # so two arms are two documents (or two intervened models), not one
        # document with a wrapper here
        during_generation = _wrapped(
            obj[WRITES_DURING_GENERATION_FIELD],
            _scalar_bool,
            f"{path}.{WRITES_DURING_GENERATION_FIELD}",
            allow_sweep=False,
        )
    return IMSpec(
        input=_wrapped(obj["input"], _scalar_str, f"{path}.input"),
        reads=reads,
        writes=_wrapped(obj["writes"], _str_list, f"{path}.writes")
        if "writes" in obj
        else (),
        writes_during_generation=during_generation,
    )


def _parse_intervened_models(raw: Any, path: str) -> dict[str, IMSpec]:
    obj = _require_mapping(raw, path)
    return {name: _parse_im(value, f"{path}.{name}") for name, value in obj.items()}


def _token_form(value: Any, path: str) -> str:
    """``token_form`` (§2.10): ``"id"`` is the one value. A retired value is
    refused before this runs (`_retired_token_form`)."""
    return _enum(value, TOKEN_FORMS, path)


def _retired_token_form(
    form: str, kind: str, obj: Mapping[str, Any], path: str
) -> ParseError:
    """The refusal of a retired ``token_form`` (§2.10) on the aggregation
    ``obj`` of ``kind``. A document that carries one was written when the key
    rewrote the answer's leading space. Now the answer string is tokenized as
    written, so the space belongs in the string. The refusal states the
    rewrite of each answer string that keeps the tokens the retired value
    scored. Where the answers are dataset columns it names them, because the
    strings to rewrite are in the table."""
    rewrite = RETIRED_TOKEN_FORM_REWRITES[form]
    columns = answer_columns(kind, obj)
    if columns:
        named = "; ".join(f"{field}: {spelled}" for field, spelled in columns.items())
        how = (
            f"This aggregation reads its answers from dataset columns ({named}), "
            "so the strings to rewrite are in the table. To keep the tokens the "
            "retired value scored, rewrite each answer string s of those "
            f"columns, each member of a list of forms included, {rewrite}; then "
            "drop the key"
        )
    else:
        how = (
            "Write ' Seattle' for the token the model emits after a space and "
            "'Seattle' for the one it emits glued to the text before it. To "
            f"keep the tokens the retired value scored, rewrite each answer "
            f"string s {rewrite}; then drop the key"
        )
    return ParseError(
        "P4",
        f"token_form {form!r} was retired: an answer string is tokenized as "
        f"written, so a leading space is part of the answer. {how}. The one "
        "remaining value is 'id', for a column of integer vocabulary ids "
        "(§2.10)",
        path=path,
    )


def _parse_aggregation(raw: Any, path: str) -> AggregationSpec:
    """§2.10: a reduction over the read the owning entry names — a
    ``AggregationSpec`` without its ``of``."""
    obj = _require_mapping(raw, path)
    if "of" in obj:
        raise ParseError(
            "P3",
            "an aggregation names no 'of': the read it reduces is the entry's "
            "('read' + 'model' on the save entry, objective term or eval entry "
            "that carries it, §2.10) — " + _MIGRATE_HINT,
            path=f"{path}.of",
        )
    kind = obj.get("kind")
    if not isinstance(kind, str) or kind not in METRIC_KINDS:
        raise ParseError(
            "P4",
            f"unknown metric kind {kind!r}{suggest(str(kind), METRIC_KINDS)}",
            path=f"{path}.kind",
        )
    extra = METRIC_FIELDS[kind]
    takes_token_form = kind in TOKEN_COLUMN_METRIC_KINDS
    optional = OPTIONAL_METRIC_FIELDS.get(kind, ())
    # a kind that may carry `restrict` resolves answer strings only when it
    # does, so the key is *allowed* here and *required or refused* below
    _check_keys(
        obj,
        (
            "kind",
            *extra,
            *optional,
            *(("token_form",) if takes_token_form or "restrict" in optional else ()),
            *IDENTITY_COLUMNS,
            MINIMUM_COUNT_FIELD,
        ),
        path,
    )
    unit, estimand_version = _parse_metric_identity(obj, kind, path)
    minimum_count: int | None = None
    if MINIMUM_COUNT_FIELD in obj:
        # a decision threshold is a preregistered bar, not a research
        # variable — fixed per campaign like `token_form`, never swept
        minimum_count = _wrapped(
            obj[MINIMUM_COUNT_FIELD],
            _scalar_int,
            f"{path}.{MINIMUM_COUNT_FIELD}",
            allow_sweep=False,
        )
        if minimum_count < 1:
            raise ParseError(
                "P2",
                f"{MINIMUM_COUNT_FIELD} must be a positive integer, got "
                f"{minimum_count} — a decision rule that needs no eligible row "
                "declares no threshold",
                path=f"{path}.{MINIMUM_COUNT_FIELD}",
            )
    token_form: Any = None
    if "token_form" in obj:
        retired = obj["token_form"]
        if isinstance(retired, str) and retired in RETIRED_TOKEN_FORMS:
            raise _retired_token_form(retired, kind, obj, f"{path}.token_form")
        # optional, and never materialized: absent means the answers are
        # strings tokenized as written (§2.10), and an absent key keeps the
        # digest. Not sweepable: how a column is read is not a research
        # variable, the same reasoning as `top_k.by`
        token_form = _wrapped(
            obj["token_form"],
            _token_form,
            f"{path}.token_form",
            allow_sweep=False,
        )
    fields: dict[str, Any] = {}
    for field in extra:
        if field not in obj:
            hint = (
                " — the ranking rule is mandatory because the read decides it: "
                f"{VOCAB_TOP_K_RANKING!r} (softmax the vocabulary, lm_head reads "
                "only), 'value' (largest signed entries) or 'abs_value' (largest "
                "magnitude). A pre-'by' document that scored logits meant "
                f"{VOCAB_TOP_K_RANKING!r}"
                if (kind, field) == ("top_k", "by")
                else ""
            )
            raise ParseError(
                "P2", f"metric kind {kind!r} needs {field!r}{hint}", path=path
            )
        if field == "k":
            fields[field] = _wrapped(obj[field], _scalar_int, f"{path}.{field}")
        elif field == "by":
            # ranking rule, not a dataset column — and not sweepable: a sweep
            # over `by` would fork a campaign on how a plot is read rather
            # than on a research variable (same reasoning as `token_form`).
            fields[field] = _wrapped(
                obj[field],
                lambda v, p: _enum(v, TOP_K_RANKINGS, p),
                f"{path}.{field}",
                allow_sweep=False,
            )
        elif field == "groups":
            fields[field] = _wrapped(obj[field], _any_leaf, f"{path}.{field}")
        elif field == "tokens":
            # the run's answer space, not a research variable — fixed per
            # campaign like `token_form` and `top_k.by`: a sweep over it would
            # fork the campaign on what gets saved rather than on a hypothesis
            fields[field] = _wrapped(
                obj[field], _token_list, f"{path}.{field}", allow_sweep=False
            )
        elif field == "target" and kind in READ_TARGET_METRIC_KINDS:
            # the second distribution: a read reference, bare or qualified
            # (§2.7); never swept — two targets are two aggregations
            fields[field] = _parse_read_ref(obj[field], f"{path}.{field}")
        else:
            fields[field] = _wrapped(obj[field], _scalar_str, f"{path}.{field}")
    for field in optional:
        if field == "restrict":
            # no default: absent means unrestricted, and stays absent (§2.10)
            if field in obj:
                fields[field] = _parse_restrict(obj[field], f"{path}.{field}")
            continue
        value = obj.get(field, METRIC_FIELD_DEFAULTS[(kind, field)])
        fields[field] = _wrapped(value, _scalar_str, f"{path}.{field}")
        if (kind, field) == ("match", "mode") and fields[field] not in MATCH_MODES:
            raise ParseError(
                "P4",
                f"unknown match mode {fields[field]!r} — one of "
                f"{list(MATCH_MODES)}{suggest(str(fields[field]), MATCH_MODES)}. "
                "(A task's 'prefix' string_mode is 'first_token' here — the "
                "translation table in sec. 2.10.)",
                path=f"{path}.{field}",
            )
    if token_form == "id" and (
        kind in {"class_probs", "token_logits"} or fields.get("mode") == "first_token"
    ):
        raise ParseError(
            "P2",
            "token_form='id' scores exact integer token IDs from dataset columns; "
            "it does not accept literal token lists or first_token matching",
            path=f"{path}.token_form",
        )
    if "restrict" in optional:
        # `restrict` is what makes the kind resolve an answer to a token id,
        # so only a restricted `js` may say its answers are ids — the rule
        # the token-column kinds and `kl` each follow by kind, applied here
        # by field
        if "restrict" not in fields and "token_form" in obj:
            raise ParseError(
                "P3",
                f"metric kind {kind!r} without 'restrict' compares two whole "
                "distributions and resolves no string — 'token_form' has nothing "
                "to decide; drop it, or add 'restrict'",
                path=f"{path}.token_form",
            )
    return AggregationSpec(
        kind=kind,
        fields=fields,
        token_form=token_form,
        unit=unit,
        estimand_version=estimand_version,
        minimum_count=minimum_count,
    )


def _parse_restrict(value: Any, path: str) -> Any:
    """``js.restrict`` (§2.10): the answer set the two distributions are
    restricted to and renormalised over — a **column name** (a string), whose
    per-row value is a list of answer strings, or a **literal list** of answer
    strings for an answer space that is one for the whole run (the
    ``token_logits.tokens`` shape). The shape decides, as it does for
    ``class_probs.groups``. Not sweepable in either form: an answer space is
    not a research variable, so a sweep wrapper is refused as a shape."""
    if isinstance(value, str):
        return _scalar_str(value, path)
    if isinstance(value, list):
        return _token_list(value, path)
    raise ParseError(
        "P2",
        "restrict is a column name (whose per-row value is a list of answer "
        "strings) or a literal list of answer strings — not a sweep: an answer "
        "space is not a research variable",
        path=path,
    )


def _parse_metric_identity(
    obj: Mapping[str, Any], kind: str, path: str
) -> tuple[str | None, str | None]:
    """The optional ``unit`` / ``estimand_version`` of a metric (§2.10).

    Both are parse-level rejections of the existing kinds, no §5 rule: an
    off-vocabulary unit or a malformed identifier is ``P4`` with suggestions
    (``estimand.py`` owns both vocabularies), and a value the kind does not
    compute — ``percentage_points`` on a ``match``, ``ratio_of_sums/v1`` on a
    ``kl`` — is ``P4`` too, naming what the kind computes: the admissible set
    for a kind is exactly its own unit and its own identifier. Neither field
    is sweepable — an identity is not a research variable."""
    unit: str | None = None
    if "unit" in obj:
        unit = _wrapped(
            obj["unit"],
            lambda v, p: _enum(v, UNITS, p),
            f"{path}.unit",
            allow_sweep=False,
        )
        own = METRIC_UNITS[kind]
        if own is None:
            raise ParseError(
                "P4",
                f"metric kind {kind!r} produces no scalar and has no unit — "
                f"drop 'unit' ({unit!r})",
                path=f"{path}.unit",
            )
        if unit != own:
            raise ParseError(
                "P4",
                f"metric kind {kind!r} computes a value in {own!r}, not {unit!r} — "
                "a document may state a kind's unit, not change it",
                path=f"{path}.unit",
            )
    estimand_version: str | None = None
    if "estimand_version" in obj:
        authored = _wrapped(
            obj["estimand_version"],
            lambda v, _p: v,
            f"{path}.estimand_version",
            allow_sweep=False,
        )
        try:
            parse_identifier(authored)
        except EstimandError as err:
            raise ParseError("P4", str(err), path=f"{path}.estimand_version") from err
        own_identity = metric_identity(kind)
        if authored != own_identity:
            raise ParseError(
                "P4",
                f"metric kind {kind!r} computes {own_identity!r}, not {authored!r} — "
                "a kind is one arithmetic, so its identifier is derived; a "
                "reduction over the saved table names its own (workflow spec §2.6)",
                path=f"{path}.estimand_version",
            )
        estimand_version = str(authored)
    return unit, estimand_version


def _parse_aggregated_ref(
    obj: Mapping[str, Any], path: str, *, aggregation_required: bool
) -> tuple[ReadRef, AggregationSpec | None]:
    """The ``read`` + ``model`` (+ ``aggregation``) of a save entry, an
    objective term or an eval entry (§2.10–§2.12): the read as taken on the
    model, and the reduction over it when one is authored."""
    for field in ("read", "model"):
        if field not in obj:
            raise ParseError(
                "P2",
                f"{'this entry' if field == 'read' else 'a read entry'} needs "
                f"{field!r} — a read is measured on one model (§2.7)",
                path=path,
            )
    ref = ReadRef(
        _scalar_str(obj["read"], f"{path}.read"),
        _scalar_str(obj["model"], f"{path}.model"),
    )
    aggregation = None
    if "aggregation" in obj:
        aggregation = _parse_aggregation(obj["aggregation"], f"{path}.aggregation")
    elif aggregation_required:
        raise ParseError(
            "P2",
            "this entry reduces its read: it needs 'aggregation' (§2.10)",
            path=path,
        )
    return ref, aggregation


#: ``l0`` is the expected kept fraction of a ``hard_concrete`` gate's sampled
#: mask (§2.11), the Louizos et al. expected-L0 ``σ(θ − β·log(−γ/ζ))`` — legal
#: only under that map, as ``l1`` is legal only under the deterministic ones
#: (rule 4): on a deterministic map the relaxed mask is already the kept
#: probability and its mean is ``l1``, so ``l0`` there would be a second
#: spelling of one computation. Gates only.
REGULARIZER_KINDS: tuple[str, ...] = ("l1", "l2", "l0")
#: §2.11 ``reduce`` on a regularizer term: how the per-unit penalized
#: quantities, concatenated over the term's featurizers, become one number.
REGULARIZER_REDUCTIONS: tuple[str, ...] = ("mean", "sum")
#: §2.11 ``costs`` on a regularizer term, the word form: a rule for the
#: per-target multiplier instead of a table. ``parameter_count`` is ``1 /
#: (the target's penalized element count)``.
REGULARIZER_COSTS: tuple[str, ...] = ("parameter_count",)


def _parse_regularizer_names(value: Any, path: str) -> tuple[str, ...]:
    """The featurizers one regularizer penalizes together (§2.11): a name, or
    a non-empty list of distinct names. The list is one penalty over the
    concatenation of their parameters, so a repeated name would count a
    featurizer twice and an empty list would penalize nothing — both refuse
    here rather than fit quietly."""
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ParseError(
            "P2",
            "a regularizer names one featurizer (or dotted slot) or a list of "
            "featurizer names",
            path=path,
        )
    if not value:
        raise ParseError(
            "P2", "a regularizer list names at least one featurizer", path=path
        )
    seen: set[str] = set()
    for name in value:
        if name in seen:
            raise ParseError(
                "P2",
                f"a regularizer list names each featurizer once ({name!r} repeats)",
                path=path,
            )
        seen.add(name)
    return tuple(value)


def _parse_reduce(value: Any, path: str) -> str:
    return _enum(_scalar_str(value, path), REGULARIZER_REDUCTIONS, path)


#: The optional fields a regularizer term carries beside its kind.
_REGULARIZER_OPTIONS: tuple[str, ...] = ("reduce", "costs")


def _parse_costs(value: Any, path: str) -> Mapping[str, float] | str:
    """§2.11 ``costs``: ``{target: c}`` with every ``c`` a finite positive
    number — a cost of 0 would name a target and penalize nothing, so name
    fewer targets instead — or one word of [`REGULARIZER_COSTS`][]. Whether
    the keys are the term's own targets is a reference and rule 4's.

    The two sweep refusals below are reachable at the authoring gate
    because ``compile.STAGES`` runs ``gate`` (the strict parse, sweep
    wrappers intact) before ``expand``; were the stages ever reordered, a
    ``{"sweep": …}`` cost would be expanded into points and swept silently
    instead of refused here."""
    if isinstance(value, str):
        return _enum(value, REGULARIZER_COSTS, path)
    if not isinstance(value, Mapping):
        raise ParseError(
            "P2",
            "'costs' is {target: positive number} — a multiplier per penalized "
            f"featurizer — or one of {list(REGULARIZER_COSTS)}",
            path=path,
        )
    if not value:
        raise ParseError("P2", "'costs' names at least one target", path=path)
    out: dict[str, float] = {}
    for key, cost in value.items():
        if not isinstance(key, str) or not key:
            raise ParseError(
                "P2", f"'costs' keys are target names, got {key!r}", path=path
            )
        if key == "sweep":
            raise ParseError(
                "P2",
                "'costs' is not swept — a cost is a literal number per target; "
                "sweep the term's weight instead",
                path=path,
            )
        # convert inside the refusal: an integer too large for a double
        # (JSON keeps it an int) must leave as P2, not as OverflowError
        try:
            scaled = float(cost)
        except (OverflowError, TypeError, ValueError):
            scaled = float("nan")
        if (
            isinstance(cost, bool)
            or not isinstance(cost, (int, float))
            or not math.isfinite(scaled)
            or scaled <= 0.0
        ):
            swept = isinstance(cost, Mapping) and "sweep" in cost
            raise ParseError(
                "P2",
                f"a cost is a finite positive number, got {cost!r} — "
                + (
                    "a cost is a literal, not swept; sweep the term's weight instead"
                    if swept
                    else "a target that should cost nothing is a target to leave out"
                ),
                path=f"{path}.{key}",
            )
        out[key] = scaled
    return out


def _parse_regularizer(
    value: Any, path: str
) -> tuple[tuple[str, tuple[str, ...]], str | None, Mapping[str, float] | str | None]:
    """The positional regularizer, ``{"l1"|"l2"|"l0": names}`` with optional
    ``"reduce"`` and ``"costs"`` (§2.11): returns ``((kind, names), reduce,
    costs)``."""
    reg = _require_mapping(value, path)
    if "constraint" in reg:
        raise ParseError(
            "P2",
            "a 'constraint' term is addressed by name (its duals are traced under "
            "it) — spell the objective in its named form",
            path=f"{path}.constraint",
        )
    kinds = [key for key in reg if key in REGULARIZER_KINDS]
    extra = [
        key
        for key in reg
        if key not in REGULARIZER_KINDS and key not in _REGULARIZER_OPTIONS
    ]
    if len(kinds) != 1 or extra:
        odd = extra[0] if extra else (next(iter(reg)) if len(reg) == 1 else None)
        raise ParseError(
            "P2",
            'a regularizer is {"l1": names}, {"l2": names} or {"l0": names}, '
            'optionally with "reduce" and "costs"'
            + (suggest(odd, REGULARIZER_KINDS) if isinstance(odd, str) else ""),
            path=path,
        )
    (kind,) = kinds
    reduce = _parse_reduce(reg["reduce"], f"{path}.reduce") if "reduce" in reg else None
    costs = _parse_costs(reg["costs"], f"{path}.costs") if "costs" in reg else None
    return (kind, _parse_regularizer_names(reg[kind], f"{path}.{kind}")), reduce, costs


def _parse_objective_term(value: Any, path: str) -> ObjectiveTerm:
    """The positional form: ``[weight, {read, model, aggregation}]`` or
    ``[weight, {<regularizer kind>: names}]`` ([`REGULARIZER_KINDS`][])."""
    if not isinstance(value, list) or len(value) != 2:
        raise ParseError(
            "P2", "an objective term is [weight, aggregation-or-regularizer]", path=path
        )
    weight = _wrapped(value[0], _scalar_number, f"{path}[0]")
    if isinstance(value[1], str):
        raise ParseError(
            "P3",
            f"an objective term names no metric ({value[1]!r}): under protocol 4 "
            'it carries its aggregation, [weight, {"read": …, "model": …, '
            '"aggregation": {…}}] (§2.11) — ' + _MIGRATE_HINT,
            path=f"{path}[1]",
        )
    term = _require_mapping(value[1], f"{path}[1]")
    if "read" in term or "aggregation" in term or "model" in term:
        _check_keys(term, ("read", "model", "aggregation"), f"{path}[1]")
        read, aggregation = _parse_aggregated_ref(
            term, f"{path}[1]", aggregation_required=True
        )
        return ObjectiveTerm(weight=weight, read=read, aggregation=aggregation)
    regularizer, reduce, costs = _parse_regularizer(term, f"{path}[1]")
    return ObjectiveTerm(
        weight=weight, regularizer=regularizer, reduce=reduce, costs=costs
    )


def _refuse_sweep(value: Any, path: str, what: str) -> None:
    """A §2.11 ``constraint`` field is not an axis: one constraint, one target.
    A ``{"sweep": …}`` wrapper anywhere in the block is refused by name, not
    as ``_scalar_number``'s "expected a number, got dict" or ``_check_keys``'s
    "unknown key 'sweep'", so the author learns the rule, not the type.
    Reachable for the reason ``_parse_costs`` records: the gate parses before
    ``expand``, wrappers intact.

    The ``{"axis": …}`` spelling of §3.2 reaches here two ways. A document
    that declares an ``axes`` group has it lowered to the sweep it stands for
    by ``compile._axes`` *before* the gate, so the message names ``axis`` too
    — the only word that tells that author what was refused. A document with
    no ``axes`` group (``has_axes`` is the stage's guard) hands the wrapper to
    the gate intact, in a real compile as much as in a test that skips it —
    that is what the ``"axis" in value`` check is for."""
    if isinstance(value, Mapping) and ("sweep" in value or "axis" in value):
        raise ParseError(
            "P2",
            f"a constraint's {what} is not swept (nor an `axis`) — one constraint, "
            "one target; author one document per value, or override it per run "
            "with `set` (§2.11)",
            path=path,
        )


def _parse_constraint(raw: Any, path: str) -> ConstraintSpec:
    """§2.11 ``constraint``: ``{"target": t, "dual": {"lr": η, "init"?: [λ₁,
    λ₂]}}``. The target is a density — a fraction in (0, 1); the dual lr is
    a positive number; ``init`` is two finite numbers with ``λ₂ ≥ 0``, absent
    for ``(0, 0)``."""
    _refuse_sweep(raw, path, "block")
    obj = _require_mapping(raw, path)
    _check_keys(obj, ("target", "dual"), path)
    for field in ("target", "dual"):
        if field not in obj:
            raise ParseError("P2", f"a constraint needs {field!r}", path=path)
    _refuse_sweep(obj["target"], f"{path}.target", "'target'")
    _refuse_sweep(obj["dual"], f"{path}.dual", "'dual' pair")
    target = _scalar_number(obj["target"], f"{path}.target")
    if not 0.0 < float(target) < 1.0:
        raise ParseError(
            "P2",
            f"a target density is a fraction in (0, 1) — the mask mean the fit is "
            f"held to — got {target!r}",
            path=f"{path}.target",
        )
    dual = _require_mapping(obj["dual"], f"{path}.dual")
    _check_keys(dual, ("lr", "init"), f"{path}.dual")
    if "lr" not in dual:
        raise ParseError(
            "P2",
            "the dual pair needs 'lr' — the rate (λ₁, λ₂) ascend at",
            path=f"{path}.dual",
        )
    _refuse_sweep(dual["lr"], f"{path}.dual.lr", "'dual.lr'")
    lr = _scalar_number(dual["lr"], f"{path}.dual.lr")
    if not math.isfinite(float(lr)) or float(lr) <= 0.0:
        raise ParseError(
            "P2", f"dual.lr is a positive number, got {lr!r}", path=f"{path}.dual.lr"
        )
    init: tuple[float, float] | None = None
    if "init" in dual:
        raw_init = dual["init"]
        _refuse_sweep(raw_init, f"{path}.dual.init", "'dual.init'")
        if (
            not isinstance(raw_init, list)
            or len(raw_init) != 2
            or any(
                isinstance(v, bool)
                or not isinstance(v, (int, float))
                or not math.isfinite(float(v))
                for v in raw_init
            )
        ):
            raise ParseError(
                "P2",
                "dual.init is [λ₁, λ₂] — two finite numbers — or absent for [0, 0]",
                path=f"{path}.dual.init",
            )
        if float(raw_init[1]) < 0.0:
            # λ₂ is the quadratic penalty's coefficient: negative, the term is
            # concave and the mask gradient points away from the target until
            # the ascent carries λ₂ back above zero. λ₁ ranges over ℝ — an
            # equality multiplier — and the fit takes it negative itself
            raise ParseError(
                "P2",
                "dual.init's λ₂ is the quadratic penalty's coefficient — negative "
                "inverts it (the gate driven away from the target); λ₁ may be "
                "negative, the fit takes it there itself (§2.11)",
                path=f"{path}.dual.init",
            )
        init = (float(raw_init[0]), float(raw_init[1]))
    return ConstraintSpec(target=float(target), dual_lr=float(lr), dual_init=init)


def _parse_named_objective_term(name: str, value: Any, path: str) -> ObjectiveTerm:
    """The named form: ``{"weight": w, "metric": name}``, ``{"weight": w,
    "l1"|"l2"|"l0": names}`` or — on a mask term — ``{"l1"|"l0": names,
    "constraint": {…}}``, which carries its dual pair instead of a weight
    (§2.11). A named term's weight sits under mapping keys, so it is a field
    of a named entry and may be swept (§3); the positional form's weight is
    inside a list and may not, and nothing inside ``constraint`` is swept
    (`_refuse_sweep`)."""
    obj = _require_mapping(value, path)
    if "metric" in obj:
        raise ParseError(
            "P3",
            "an objective term names no 'metric': under protocol 4 it carries "
            "its aggregation — 'read', 'model' and 'aggregation' (§2.11) — "
            + _MIGRATE_HINT,
            path=f"{path}.metric",
        )
    _check_keys(
        obj,
        (
            "weight",
            "read",
            "model",
            "aggregation",
            *REGULARIZER_KINDS,
            *_REGULARIZER_OPTIONS,
            "constraint",
        ),
        path,
    )
    constraint = (
        _parse_constraint(obj["constraint"], f"{path}.constraint")
        if "constraint" in obj
        else None
    )
    if constraint is None and "weight" not in obj:
        raise ParseError("P2", "an objective term needs 'weight'", path=path)
    weight = (
        _wrapped(obj["weight"], _scalar_number, f"{path}.weight")
        if "weight" in obj
        else None
    )
    aggregated = any(key in obj for key in ("read", "model", "aggregation"))
    kinds = [key for key in obj if key in REGULARIZER_KINDS]
    if len(kinds) + int(aggregated) != 1:
        raise ParseError(
            "P2",
            "an objective term is a weight and exactly one of an aggregation "
            "('read' + 'model' + 'aggregation') or a regularizer ('l1', 'l2', 'l0')",
            path=path,
        )
    reduce = _parse_reduce(obj["reduce"], f"{path}.reduce") if "reduce" in obj else None
    costs = _parse_costs(obj["costs"], f"{path}.costs") if "costs" in obj else None
    if aggregated:
        # each option refused for its own reason: P2 text is the interface
        for option, parsed, why in (
            (
                "reduce",
                reduce,
                "a metric term is already one number per row, reduced by the "
                "metric's own rule",
            ),
            (
                "costs",
                costs,
                "a metric term has no per-target quantities to scale — scaling "
                "one metric is what its weight is",
            ),
            (
                "constraint",
                constraint,
                "a metric has no mask density to hold to a target",
            ),
        ):
            if parsed is not None:
                raise ParseError(
                    "P2",
                    f"'{option}' is a regularizer's field — {why}",
                    path=f"{path}.{option}",
                )
        read, aggregation = _parse_aggregated_ref(obj, path, aggregation_required=True)
        return ObjectiveTerm(
            weight=weight, read=read, aggregation=aggregation, name=name
        )
    (kind,) = kinds
    if constraint is not None:
        if weight is not None:
            raise ParseError(
                "P2",
                "a constraint term has no weight — its multipliers are the dual pair "
                "(λ₁, λ₂) the fit ascends (§2.11); drop 'weight' or drop 'constraint'",
                path=f"{path}.weight",
            )
        if kind not in ("l1", "l0"):
            raise ParseError(
                "P2",
                "a target density is a mask quantity — 'l1' (the soft mask's mean) "
                f"or 'l0' (the expected kept fraction), not {kind!r}",
                path=f"{path}.constraint",
            )
        if reduce == "sum":
            raise ParseError(
                "P2",
                "a constraint's target is a density (a fraction), and under 'sum' "
                "the term is a count — spell 'mean' or leave reduce unauthored",
                path=f"{path}.reduce",
            )
        if costs == "parameter_count":
            # the same category error as `reduce: sum`, and worse: with the
            # density divided by N the gap is negative from the first update,
            # λ₁ descends and the fit drives the gate toward *full* density
            raise ParseError(
                "P2",
                "a constraint's target is a density, and 'parameter_count' divides "
                "each target's quantities by its element count — the term is no "
                "longer a fraction; drop 'costs' or drop 'constraint' (a costs "
                "table is fine: the target is then held on the cost-weighted "
                "density)",
                path=f"{path}.costs",
            )
    names = _parse_regularizer_names(obj[kind], f"{path}.{kind}")
    return ObjectiveTerm(
        weight=weight,
        regularizer=(kind, names),
        name=name,
        reduce=reduce,
        costs=costs,
        constraint=constraint,
    )


def _parse_objective(raw: Any, path: str) -> tuple[ObjectiveTerm, ...]:
    if isinstance(raw, list) and raw:
        return tuple(
            _parse_objective_term(t, f"{path}[{i}]") for i, t in enumerate(raw)
        )
    if isinstance(raw, dict) and raw and "sweep" not in raw:
        return tuple(
            _parse_named_objective_term(name, term, f"{path}.{name}")
            for name, term in raw.items()
        )
    raise ParseError(
        "P2",
        "train.objective is a non-empty list of [weight, term] pairs or a "
        "non-empty object of named {weight, term} entries",
        path=path,
    )


def _parse_counter(value: Any, path: str) -> dict[str, Any]:
    obj = _require_mapping(value, path)
    _check_keys(obj, ("epochs", "updates"), path)
    if len(obj) != 1:
        raise ParseError("P2", "expected exactly one of epochs/updates", path=path)
    ((unit, count),) = obj.items()
    return {unit: _wrapped(count, _scalar_int, f"{path}.{unit}")}


#: The optimizer fields a fit may set **per trained parameter** (§2.11): one
#: number for everything in ``train.params``, or a mapping keyed by the entries
#: of ``train.params`` — a rotation at 1e-3 beside a gate at 0.1 in one fit.
PER_PARAMS_OPTIMIZER_FIELDS: tuple[str, ...] = ("lr", "weight_decay")


def _parse_per_params_number(value: Any, params: Sequence[str], path: str) -> None:
    """``lr`` / ``weight_decay``: a scalar (or a sweep of one) for every trained
    parameter, or a mapping ``{<params entry>: number}`` naming **every** entry
    of ``train.params`` exactly once — a key that is not a trained parameter
    is refused (it would silently apply to nothing), and an entry left out is
    refused rather than given a hidden default (there is none to give)."""
    if isinstance(value, Mapping) and "sweep" not in value:
        keys = set(value)
        declared = set(params)
        unknown = sorted(keys - declared)
        if unknown:
            raise ParseError(
                "P2",
                f"per-parameter optimizer setting names {unknown}, which train.params "
                f"does not train — keys are entries of train.params ({sorted(declared)})",
                path=path,
            )
        missing = sorted(declared - keys)
        if missing:
            raise ParseError(
                "P2",
                f"per-parameter optimizer setting leaves {missing} without a value — "
                "name every entry of train.params, or give one number for all",
                path=path,
            )
        for key, number in value.items():
            _wrapped(number, _scalar_number, f"{path}.{key}")
        return
    _wrapped(value, _scalar_number, path)


def _parse_train(raw: Any, path: str) -> TrainSpec:
    obj = _require_mapping(raw, path)
    _check_keys(
        obj,
        (
            "objective",
            "params",
            "optimizer",
            "steps",
            "batch",
            "anneal",
            "control",
            "phases",
            "precision",
            "eval",
            "early_stop",
            "checkpoint",
            "seed",
        ),
        path,
    )
    for field in ("objective", "params", "optimizer", "steps", "batch"):
        if field not in obj:
            raise ParseError("P2", f"train needs {field!r}", path=path)
    objective = _parse_objective(obj["objective"], f"{path}.objective")
    params = _str_list(obj["params"], f"{path}.params")
    optimizer = _require_mapping(obj["optimizer"], f"{path}.optimizer")
    _check_keys(optimizer, OPTIMIZER_FIELDS, f"{path}.optimizer")
    if "name" not in optimizer or "lr" not in optimizer:
        raise ParseError(
            "P2", "train.optimizer needs 'name' and 'lr'", path=f"{path}.optimizer"
        )
    _enum(optimizer["name"], tuple(OPTIMIZER_DEFAULTS), f"{path}.optimizer.name")
    for field in ("lr", "weight_decay"):
        if field in optimizer:
            _parse_per_params_number(
                optimizer[field], params, f"{path}.optimizer.{field}"
            )
    for field in ("eps", "momentum", "clip_grad_norm"):
        if field in optimizer:
            _wrapped(optimizer[field], _scalar_number, f"{path}.optimizer.{field}")
    if "betas" in optimizer:
        betas = optimizer["betas"]
        if (
            not isinstance(betas, list)
            or len(betas) != 2
            or not all(
                isinstance(b, (int, float)) and not isinstance(b, bool) for b in betas
            )
        ):
            raise ParseError(
                "P2",
                "optimizer betas is a two-number list",
                path=f"{path}.optimizer.betas",
            )
    schedule = "constant"
    if "schedule" in optimizer:
        schedule = _enum(
            optimizer["schedule"], OPTIMIZER_SCHEDULES, f"{path}.optimizer.schedule"
        )
    if "warmup_frac" in optimizer:
        if schedule != "linear_warmup_decay":
            raise ParseError(
                "P2",
                "'warmup_frac' belongs to schedule 'linear_warmup_decay' — a constant "
                "schedule warms nothing up (§2.11)",
                path=f"{path}.optimizer.warmup_frac",
            )
        frac = _scalar_number(optimizer["warmup_frac"], f"{path}.optimizer.warmup_frac")
        if not 0.0 <= float(frac) < 1.0:
            raise ParseError(
                "P2",
                f"warmup_frac is a fraction of the updates in [0, 1), got {frac}",
                path=f"{path}.optimizer.warmup_frac",
            )
    steps = _parse_counter(obj["steps"], f"{path}.steps")
    batch = _require_mapping(obj["batch"], f"{path}.batch")
    _check_keys(batch, ("pairs",), f"{path}.batch")
    if "pairs" not in batch:
        raise ParseError(
            "P2",
            "train.batch counts base+counterfactual pairs: {'pairs': n}",
            path=f"{path}.batch",
        )
    anneal = None
    if "anneal" in obj:
        anneal_raw = _require_mapping(obj["anneal"], f"{path}.anneal")
        anneal = {
            key: _wrapped(
                value,
                lambda v, p: _parse_anneal_entry(v, p),
                f"{path}.anneal.{key}",
            )
            for key, value in anneal_raw.items()
        }
    control = None
    if "control" in obj:
        control_raw = _require_mapping(obj["control"], f"{path}.control")
        if not control_raw:
            raise ParseError(
                "P2", "train.control names at least one target", path=f"{path}.control"
            )
        control = {
            key: _parse_control(value, f"{path}.control.{key}")
            for key, value in control_raw.items()
        }
    phases = None
    if "phases" in obj:
        phases = _parse_phases(obj["phases"], params, f"{path}.phases")
    precision = None
    if "precision" in obj:
        precision_raw = _require_mapping(obj["precision"], f"{path}.precision")
        _check_keys(precision_raw, ("feature", "loss"), f"{path}.precision")
        precision = {
            key: _wrapped(
                value,
                lambda v, p: _enum(v, PRECISION_DTYPES, p),
                f"{path}.precision.{key}",
            )
            for key, value in precision_raw.items()
        }
    eval_spec = None
    if "eval" in obj:
        eval_raw = _require_mapping(obj["eval"], f"{path}.eval")
        if "metrics" in eval_raw:
            raise ParseError(
                "P3",
                "train.eval names no 'metrics': under protocol 4 it carries "
                "'aggregations', {label: {read, model, aggregation}} (§2.11) — "
                + _MIGRATE_HINT,
                path=f"{path}.eval.metrics",
            )
        _check_keys(eval_raw, ("every", "split", "aggregations"), f"{path}.eval")
        for field in ("every", "split", "aggregations"):
            if field not in eval_raw:
                raise ParseError(
                    "P2", f"train.eval needs {field!r}", path=f"{path}.eval"
                )
        aggregations_raw = _require_mapping(
            eval_raw["aggregations"], f"{path}.eval.aggregations"
        )
        if not aggregations_raw:
            raise ParseError(
                "P2",
                "train.eval.aggregations names at least one aggregation",
                path=f"{path}.eval.aggregations",
            )
        aggregations: dict[str, tuple[ReadRef, AggregationSpec]] = {}
        for label, entry in aggregations_raw.items():
            entry_path = f"{path}.eval.aggregations.{label}"
            entry_obj = _require_mapping(entry, entry_path)
            _check_keys(entry_obj, ("read", "model", "aggregation"), entry_path)
            read, aggregation = _parse_aggregated_ref(
                entry_obj, entry_path, aggregation_required=True
            )
            assert aggregation is not None
            aggregations[label] = (read, aggregation)
        eval_spec = {
            "every": _parse_counter(eval_raw["every"], f"{path}.eval.every"),
            "split": _wrapped(eval_raw["split"], _scalar_str, f"{path}.eval.split"),
            "aggregations": aggregations,
        }
    early_stop = None
    if "early_stop" in obj:
        es_raw = _require_mapping(obj["early_stop"], f"{path}.early_stop")
        if "metric" in es_raw:
            raise ParseError(
                "P3",
                "train.early_stop names no 'metric': under protocol 4 it names "
                "the eval label it watches as 'on' (§2.11) — " + _MIGRATE_HINT,
                path=f"{path}.early_stop.metric",
            )
        _check_keys(es_raw, ("on", "patience", "mode"), f"{path}.early_stop")
        for field in ("on", "patience", "mode"):
            if field not in es_raw:
                raise ParseError(
                    "P2", f"train.early_stop needs {field!r}", path=f"{path}.early_stop"
                )
        early_stop = {
            "on": _wrapped(es_raw["on"], _scalar_str, f"{path}.early_stop.on"),
            "patience": _wrapped(
                es_raw["patience"], _scalar_int, f"{path}.early_stop.patience"
            ),
            "mode": _wrapped(
                es_raw["mode"],
                lambda v, p: _enum(v, ("min", "max"), p),
                f"{path}.early_stop.mode",
            ),
        }
    checkpoint = None
    if "checkpoint" in obj:
        ck_raw = _require_mapping(obj["checkpoint"], f"{path}.checkpoint")
        _check_keys(ck_raw, ("every", "file_path"), f"{path}.checkpoint")
        checkpoint = {
            key: (
                _parse_counter(value, f"{path}.checkpoint.every")
                if key == "every"
                else _wrapped(value, _scalar_str, f"{path}.checkpoint.file_path")
            )
            for key, value in ck_raw.items()
        }
    seed: Any = 0
    if "seed" in obj:
        seed = _wrapped(obj["seed"], _scalar_int, f"{path}.seed")
    return TrainSpec(
        objective=objective,
        params=params,
        optimizer=dict(optimizer),
        steps=steps,
        batch=dict(batch),
        anneal=anneal,
        control=control,
        phases=phases,
        precision=precision,
        eval=eval_spec,
        early_stop=early_stop,
        checkpoint=checkpoint,
        seed=seed,
    )


def _parse_signal_target(raw: Any, path: str) -> str | list[str]:
    """What a control signal observes (§2.11): one gate's name, or a
    **list** of gate names whose kept-unit counts are summed — one signal over
    several layers' gates, as a list-valued ``l1`` is one penalty over them.
    Kept as authored here; the canonical form writes the list either way, so
    ``"g"`` and ``["g"]`` are one controller (the ``layers`` fold)."""
    if isinstance(raw, str):
        return _scalar_str(raw, path)
    if not isinstance(raw, list) or not raw:
        raise ParseError(
            "P2",
            "a control signal names a trained gate, or a non-empty list of them "
            "whose kept counts are summed",
            path=path,
        )
    names = [_scalar_str(item, f"{path}[{i}]") for i, item in enumerate(raw)]
    if len(set(names)) != len(names):
        raise ParseError(
            "P2",
            f"a control signal lists each gate once — got {names}",
            path=path,
        )
    return names


def _parse_control(raw: Any, path: str) -> dict[str, Any]:
    """One ``train.control`` entry (§2.11): a closed-loop schedule on the
    hyperparameter the key names. ``kind`` is the controller
    ([`CONTROL_KINDS`][]), ``signal`` the one fit quantity it observes —
    ``{<signal>: <featurizer>}`` over [`CONTROL_SIGNALS`][] — ``setpoint``
    the ramp the signal should follow (the ``anneal`` schedule shape, in the
    signal's units) and ``gains`` the ``kp`` / ``ki`` / optional ``kd``. The
    optional ``space``, ``bounds`` and ``d_clip`` default per
    [`CONTROL_DEFAULTS`][] and are materialized in the canonical form. The
    gains are sweepable; the kind, the signal and the setpoint are not — they
    say *what* is controlled, not how hard."""
    obj = _require_mapping(raw, path)
    _check_keys(
        obj, ("kind", "signal", "setpoint", "gains", "space", "bounds", "d_clip"), path
    )
    for field in ("kind", "signal", "setpoint", "gains"):
        if field not in obj:
            raise ParseError("P2", f"a control entry needs {field!r}", path=path)
    kind = _wrapped(
        obj["kind"],
        lambda v, p: _enum(v, CONTROL_KINDS, p),
        f"{path}.kind",
        allow_sweep=False,
    )
    signal_raw = _require_mapping(obj["signal"], f"{path}.signal")
    if len(signal_raw) != 1:
        raise ParseError(
            "P2",
            "a control observes exactly one signal: {<signal>: <featurizer>}",
            path=f"{path}.signal",
        )
    ((signal_name, signal_target),) = signal_raw.items()
    _enum(signal_name, CONTROL_SIGNALS, f"{path}.signal")
    signal = {
        signal_name: _parse_signal_target(signal_target, f"{path}.signal.{signal_name}")
    }
    setpoint_raw = _require_mapping(obj["setpoint"], f"{path}.setpoint")
    _check_keys(setpoint_raw, ("ramp",), f"{path}.setpoint")
    if "ramp" not in setpoint_raw:
        raise ParseError(
            "P2",
            "a control setpoint is {'ramp': [start, end, frac]}",
            path=f"{path}.setpoint",
        )
    ramp = _parse_anneal_schedule(setpoint_raw["ramp"], f"{path}.setpoint.ramp")
    if not 0.0 < ramp[2] <= 1.0:
        raise ParseError(
            "P2",
            f"a setpoint ramp's frac is the fraction of the run it spans, in (0, 1]; got {ramp[2]}",
            path=f"{path}.setpoint.ramp",
        )
    gains_raw = _require_mapping(obj["gains"], f"{path}.gains")
    _check_keys(gains_raw, ("kp", "ki", "kd"), f"{path}.gains")
    for gain in ("kp", "ki"):
        if gain not in gains_raw:
            raise ParseError("P2", f"control gains need {gain!r}", path=f"{path}.gains")
    gains = {
        gain: _wrapped(value, _scalar_number, f"{path}.gains.{gain}")
        for gain, value in gains_raw.items()
    }
    out: dict[str, Any] = {
        "kind": kind,
        "signal": signal,
        "setpoint": {"ramp": list(ramp)},
        "gains": gains,
    }
    if "space" in obj:
        out["space"] = _enum(obj["space"], CONTROL_SPACES, f"{path}.space")
    if "bounds" in obj:
        bounds = obj["bounds"]
        if (
            not isinstance(bounds, list)
            or len(bounds) != 2
            or not all(
                isinstance(b, (int, float)) and not isinstance(b, bool) for b in bounds
            )
            or not bounds[0] < bounds[1]
        ):
            raise ParseError(
                "P2",
                "control bounds are an increasing two-number list",
                path=f"{path}.bounds",
            )
        out["bounds"] = [float(bounds[0]), float(bounds[1])]
    if "d_clip" in obj:
        d_clip = _scalar_number(obj["d_clip"], f"{path}.d_clip")
        if d_clip <= 0:
            raise ParseError(
                "P2", "control d_clip is a positive number", path=f"{path}.d_clip"
            )
        out["d_clip"] = float(d_clip)
    return out


def _parse_anneal_schedule(value: Any, path: str) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ParseError("P2", "an anneal schedule is [start, end, frac]", path=path)
    start, end, frac = (_scalar_number(v, f"{path}[{i}]") for i, v in enumerate(value))
    return (float(start), float(end), float(frac))


def _parse_anneal_entry(value: Any, path: str) -> AnnealSchedule:
    """§2.11 ``anneal.<target>``: the list ``[start, end, frac]`` (linear), or
    the mapping ``{"from", "to", "frac", "shape"?}`` whose ``shape`` is one of
    [`ANNEAL_SHAPES`][]. A geometric schedule multiplies by a constant per
    step, so its endpoints must share a sign and neither may be zero — a ramp
    through zero has no ratio to walk."""
    if isinstance(value, list):
        return AnnealSchedule(*_parse_anneal_schedule(value, path))
    if not isinstance(value, Mapping):
        raise ParseError(
            "P2",
            'an anneal schedule is [start, end, frac] or {"from", "to", "frac", "shape"}',
            path=path,
        )
    _check_keys(value, ("from", "to", "frac", "shape"), path)
    for field in ("from", "to", "frac"):
        if field not in value:
            raise ParseError("P2", f"an anneal schedule needs {field!r}", path=path)
    start = float(_scalar_number(value["from"], f"{path}.from"))
    end = float(_scalar_number(value["to"], f"{path}.to"))
    frac = float(_scalar_number(value["frac"], f"{path}.frac"))
    shape = "linear"
    if "shape" in value:
        shape = _enum(value["shape"], ANNEAL_SHAPES, f"{path}.shape")
    if shape == "geometric" and start * end <= 0:
        raise ParseError(
            "P2",
            f"a geometric anneal multiplies by a constant per step, so 'from' and "
            f"'to' share a sign and neither is zero; got {start!r} → {end!r}",
            path=path,
        )
    return AnnealSchedule(start, end, frac, shape)


def _parse_phases(raw: Any, params: Sequence[str], path: str) -> tuple[PhaseSpec, ...]:
    """§2.11 ``phases``: a non-empty list of windows that **partition** the
    run. Each names ``until`` — one of [`PHASE_UNTIL_UNITS`][], the same
    unit for every phase — strictly increasing, the last ``frac`` exactly
    ``1.0`` (an ``updates``-counted last phase is checked against the run's
    update count by the loop, which alone knows it); and ``params``, a
    non-empty subset of ``train.params`` by entry. What ``params`` leaves out
    is frozen for the phase. A phase's ``optimizer`` may carry only the
    per-params fields, keyed by the phase's own params; its ``anneal`` is
    parsed as the top-level one; ``freeze_masks`` is a name list the
    validator resolves to gates (rule 4)."""
    if not isinstance(raw, list) or not raw:
        raise ParseError("P2", "train.phases is a non-empty list of phases", path=path)
    declared = set(params)
    out: list[PhaseSpec] = []
    unit: str | None = None
    last_end: float | None = None
    for i, entry in enumerate(raw):
        p = f"{path}[{i}]"
        obj = _require_mapping(entry, p)
        _check_keys(obj, ("until", "params", "optimizer", "anneal", "freeze_masks"), p)
        for field in ("until", "params"):
            if field not in obj:
                raise ParseError("P2", f"a phase needs {field!r}", path=p)
        until = _require_mapping(obj["until"], f"{p}.until")
        _check_keys(until, PHASE_UNTIL_UNITS, f"{p}.until")
        if len(until) != 1:
            raise ParseError(
                "P2",
                "a phase ends at exactly one of {'frac': f} | {'updates': n}",
                path=f"{p}.until",
            )
        ((this_unit, end_raw),) = until.items()
        if unit is None:
            unit = this_unit
        elif this_unit != unit:
            raise ParseError(
                "P2",
                f"every phase counts its end in one unit; phase 0 used {unit!r}, this one {this_unit!r}",
                path=f"{p}.until",
            )
        end = (
            float(_scalar_number(end_raw, f"{p}.until.frac"))
            if unit == "frac"
            else float(_scalar_int(end_raw, f"{p}.until.updates"))
        )
        if unit == "frac" and not 0.0 < end <= 1.0:
            raise ParseError(
                "P2",
                f"a phase's frac is a fraction of the run in (0, 1]; got {end}",
                path=f"{p}.until",
            )
        if unit == "updates" and end < 1:
            raise ParseError(
                "P2", "a phase spans at least one update", path=f"{p}.until"
            )
        if last_end is not None and end <= last_end:
            raise ParseError(
                "P2",
                f"phases are consecutive: this phase ends at {end}, the previous at {last_end}",
                path=f"{p}.until",
            )
        last_end = end
        phase_params = _str_list(obj["params"], f"{p}.params")
        if not phase_params:
            raise ParseError(
                "P2",
                "a phase trains at least one entry — a phase that trains nothing is a wait, not a fit",
                path=f"{p}.params",
            )
        outside = sorted(set(phase_params) - declared)
        if outside:
            raise ParseError(
                "P2",
                f"phase params {outside} are not entries of train.params ({sorted(declared)}) — "
                "a phase narrows the trained set, it never widens it",
                path=f"{p}.params",
            )
        if len(set(phase_params)) != len(phase_params):
            raise ParseError("P2", "a phase names each entry once", path=f"{p}.params")
        optimizer = None
        if "optimizer" in obj:
            optimizer = _require_mapping(obj["optimizer"], f"{p}.optimizer")
            _check_keys(optimizer, PER_PARAMS_OPTIMIZER_FIELDS, f"{p}.optimizer")
            for field, value in optimizer.items():
                _parse_per_params_number(value, phase_params, f"{p}.optimizer.{field}")
            optimizer = dict(optimizer)
        anneal = None
        if "anneal" in obj:
            anneal_raw = _require_mapping(obj["anneal"], f"{p}.anneal")
            anneal = {
                key: _wrapped(
                    value, lambda v, q: _parse_anneal_entry(v, q), f"{p}.anneal.{key}"
                )
                for key, value in anneal_raw.items()
            }
        freeze = ()
        if "freeze_masks" in obj:
            freeze = _str_list(obj["freeze_masks"], f"{p}.freeze_masks")
            if len(set(freeze)) != len(freeze):
                raise ParseError(
                    "P2", "freeze_masks names each gate once", path=f"{p}.freeze_masks"
                )
        out.append(
            PhaseSpec(
                until={unit: end if unit == "frac" else int(end)},
                params=phase_params,
                optimizer=optimizer,
                anneal=anneal,
                freeze_masks=freeze,
            )
        )
    if unit == "frac" and last_end != 1.0:
        raise ParseError(
            "P2",
            f"the last phase ends at frac {last_end}; phases partition the run, so it ends at 1.0",
            path=path,
        )
    return tuple(out)


def _parse_trajectory_every(value: Any, path: str) -> dict[str, int]:
    """``trajectory.every`` (§2.12): exactly one of
    [`TRAJECTORY_EVERY_UNITS`][], a positive integer. Not sweepable — how
    often a fit is photographed is not a research variable."""
    obj = _require_mapping(value, path)
    _check_keys(obj, TRAJECTORY_EVERY_UNITS, path)
    if len(obj) != 1:
        raise ParseError(
            "P2", f"expected exactly one of {list(TRAJECTORY_EVERY_UNITS)}", path=path
        )
    ((unit, count),) = obj.items()
    n = _scalar_int(count, f"{path}.{unit}")
    if n < 1:
        raise ParseError(
            "P2", f"every.{unit} is a positive integer, got {n}", path=path
        )
    return {unit: n}


@dataclasses.dataclass(frozen=True)
class _TrainSave:
    """A ``{"train": name, "file_path": …}`` save entry as `_parse_save`
    reads it: the name stays unresolved until ``train`` is parsed, and
    [`parse_document`][] replaces it with the [`SaveEntry`][] it names
    (`_resolve_train_save`)."""

    name: str
    file_path: str
    path: str


#: The keys a ``train`` save entry copies from the term it names (§2.12), so
#: authoring one of them beside ``train`` would be a second definition.
_TRAIN_SAVE_COPIED = ("read", "model", "aggregation", "reduce")


def _refuse_bare_save_target(aggregation: AggregationSpec | None, path: str) -> None:
    """Inside a save every read reference is ``{"read", "model"}`` (§2.7):
    a ``kl`` / ``js`` target is the one reference a save's aggregation can
    hold, and the bare-name sugar is legal only outside a save."""
    target = None if aggregation is None else aggregation.fields.get("target")
    if isinstance(target, ReadRef) and target.model is None:
        raise ParseError(
            "P2",
            'inside a save every read reference is {"read", "model"}: spell '
            f'the target {{"read": {json.dumps(target.read)}, "model": …}} (§2.7) — the '
            "bare name is a write operand's sugar, and a kl/js target's "
            "outside a save",
            path=f"{path}.target",
        )


def _parse_save(raw: Any, path: str) -> tuple[SaveEntry | _TrainSave, ...]:
    if not isinstance(raw, list):
        raise ParseError("P2", "save is a list of entries", path=path)
    entries: list[SaveEntry | _TrainSave] = []
    for i, entry_raw in enumerate(raw):
        p = f"{path}[{i}]"
        obj = _require_mapping(entry_raw, p)
        if "train" in obj:
            # a training metric by name (§2.12): the entry the term's read,
            # model and aggregation spell, resolved once `train` is parsed
            for key in _TRAIN_SAVE_COPIED:
                if key in obj:
                    raise ParseError(
                        "P2",
                        "a train save entry copies its term's read, model and "
                        f"aggregation, and takes no {key!r} of its own — save "
                        "the read inline to reduce it another way (§2.12)",
                        path=f"{p}.{key}",
                    )
            _check_keys(obj, ("train", "file_path"), p)
            if "file_path" not in obj:
                raise ParseError("P2", "a save entry needs 'file_path'", path=p)
            entries.append(
                _TrainSave(
                    name=_scalar_str(obj["train"], f"{p}.train"),
                    file_path=_scalar_str(obj["file_path"], f"{p}.file_path"),
                    path=p,
                )
            )
            continue
        if "kind" in obj:
            # a non-value entry (§2.12): the kind is the whole binding
            kind = _enum(obj["kind"], SAVE_KINDS, f"{p}.kind")
            _check_keys(
                obj,
                ("kind", "file_path", *(("every",) if kind == "trajectory" else ())),
                p,
            )
            if "file_path" not in obj:
                raise ParseError("P2", "a save entry needs 'file_path'", path=p)
            every = None
            if kind == "trajectory":
                if "every" not in obj:
                    raise ParseError(
                        "P2",
                        "a trajectory entry says how its checkpoints are spaced: "
                        "'every': {'count': n} | {'updates': n} | {'epochs': n}",
                        path=p,
                    )
                every = _parse_trajectory_every(obj["every"], f"{p}.every")
            entries.append(
                SaveEntry(
                    value=kind,
                    file_path=_scalar_str(obj["file_path"], f"{p}.file_path"),
                    kind=kind,
                    every=every,
                )
            )
            continue
        if "input" in obj or ("value" in obj and "site" not in obj):
            raise ParseError(
                "P3",
                "a save entry names the read it saves as 'read' + 'model' (with "
                "'aggregation' for a table, §2.12); 'value' with 'model'/'input' "
                "is the protocol-3 spelling — " + _MIGRATE_HINT,
                path=p,
            )
        if "file_path" not in obj:
            raise ParseError("P2", "a save entry needs 'file_path'", path=p)
        file_path = _scalar_str(obj["file_path"], f"{p}.file_path")
        if "site" in obj or "value" in obj:
            # a trained featurizer's bundle, bound by the site it is used at
            _check_keys(obj, ("value", "site", "file_path"), p)
            for field in ("value", "site"):
                if field not in obj:
                    raise ValidationError(
                        10,
                        "a featurizer save entry binds with 'value' and 'site'",
                        path=p,
                    )
            entries.append(
                SaveEntry(
                    file_path=file_path,
                    value=_scalar_str(obj["value"], f"{p}.value"),
                    site=_scalar_str(obj["site"], f"{p}.site"),
                )
            )
            continue
        # a read on a model: its rows as a tensor, or a reduction over it
        _check_keys(obj, ("read", "model", "aggregation", "file_path", "reduce"), p)
        read, aggregation = _parse_aggregated_ref(obj, p, aggregation_required=False)
        _refuse_bare_save_target(aggregation, f"{p}.aggregation")
        reduce = None
        if "reduce" in obj:
            if aggregation is not None:
                raise ValidationError(
                    10,
                    "a save entry carries 'reduce' or 'aggregation', never both — "
                    "an aggregation is already a reduction over its read (§2.12)",
                    path=f"{p}.reduce",
                )
            reduce = _enum(obj["reduce"], SAVE_REDUCTIONS, f"{p}.reduce")
        entries.append(
            SaveEntry(
                file_path=file_path, read=read, aggregation=aggregation, reduce=reduce
            )
        )
    return tuple(entries)


def _holds_sweep(node: Any) -> bool:
    if isinstance(node, Sweep):
        return True
    if isinstance(node, Mapping):
        return any(_holds_sweep(v) for v in node.values())
    if isinstance(node, (list, tuple)):
        return any(_holds_sweep(v) for v in node)
    return False


def _resolve_train_save(ref: _TrainSave, train: TrainSpec | None) -> SaveEntry:
    """The [`SaveEntry`][] a ``{"train": name}`` entry names (§2.12): a
    named objective term that reduces a read, or an eval aggregation label —
    exactly one of them. What the entry shares is the term's *definition*;
    the save computes its table over the run's rows like any save entry."""
    path = f"{ref.path}.train"
    if train is None:
        raise ValidationError(
            4,
            f"a train save entry names a training metric ({ref.name!r}), but the "
            "document has no 'train' section — spell the entry inline (§2.12)",
            path=path,
        )
    named = {term.name: term for term in train.objective if term.name is not None}
    labels: Mapping[str, tuple[ReadRef, AggregationSpec]] = (
        train.eval["aggregations"] if train.eval is not None else {}
    )
    term = named.get(ref.name)
    if term is not None and ref.name in labels:
        raise ValidationError(
            4,
            f"{ref.name!r} is both an objective term and an eval label, so a "
            "train save cannot say which it copies — rename one (§2.12)",
            path=path,
        )
    if term is not None:
        if term.read is None or term.aggregation is None:
            raise ValidationError(
                4,
                f"{ref.name!r} is a regularizer term: it reduces no read, so "
                "there is no table to save — a train save names a term with "
                "'read', 'model' and 'aggregation' (§2.12)",
                path=path,
            )
        read, spec = term.read, term.aggregation
        source = f"train.objective.{ref.name}"
    elif ref.name in labels:
        read, spec = labels[ref.name]
        source = f"train.eval.aggregations.{ref.name}"
    else:
        candidates = sorted(
            [n for n, t in named.items() if t.aggregation is not None] + list(labels)
        )
        positional = any(t.name is None for t in train.objective)
        raise ValidationError(
            4,
            f"a train save names {ref.name!r}, which names no objective term or "
            f"eval label ({candidates}){suggest(ref.name, candidates)}"
            + (
                " — positional objective terms have no name: spell the "
                "objective in the named form to save one of its terms (§2.11)"
                if positional
                else ""
            ),
            path=path,
        )
    if _holds_sweep(spec.fields):
        # the reference moves with the term point by point, while the inline
        # spelling the canonical form writes would be a second, independent
        # axis — two campaigns under one digest
        raise ValidationError(
            14,
            f"{source} carries a sweep, and a train save copies one term — spell "
            "the save entry inline and share the swept value through a named "
            "axis (§3.2)",
            path=path,
        )
    target = spec.fields.get("target")
    if isinstance(target, ReadRef) and target.model is None:
        raise ParseError(
            "P2",
            'inside a save every read reference is {"read", "model"}, and a '
            f"train save copies {source}'s bare target {target.read!r}: spell "
            f'the term\'s target {{"read": {json.dumps(target.read)}, "model": …}} to '
            "save it (§2.7)",
            path=f"{source}.aggregation.target",
        )
    return SaveEntry(file_path=ref.file_path, read=read, aggregation=spec)


def inline_train_saves(method: Mapping[str, Any]) -> list[Any]:
    """A raw ``method`` group's ``save`` list with every ``{"train": name}``
    entry spelled inline (§2.12): the named term's ``read``, ``model`` and
    ``aggregation``, deep-copied, beside the entry's ``file_path``.

    For a tree that has passed [`parse_document`][], which refuses a name
    that resolves to nothing, to both namespaces or to a swept term; an
    entry this function cannot resolve is returned as authored. The
    canonical form and the readers of a run's raw tree go through here, so a
    reference and its inline twin are one entry to all of them."""
    save = method.get("save")
    if not isinstance(save, list):
        return []
    train = method.get("train")
    train = train if isinstance(train, Mapping) else {}
    objective = train.get("objective")
    eval_spec = train.get("eval")
    labels = eval_spec.get("aggregations") if isinstance(eval_spec, Mapping) else None
    out: list[Any] = []
    for entry in save:
        if not isinstance(entry, Mapping) or "train" not in entry:
            out.append(entry)
            continue
        name = entry["train"]
        term = objective.get(name) if isinstance(objective, Mapping) else None
        if not isinstance(term, Mapping) and isinstance(labels, Mapping):
            term = labels.get(name)
        copied = ("read", "model", "aggregation")
        if not isinstance(term, Mapping) or not all(key in term for key in copied):
            out.append(entry)
            continue
        inline = {key: copy.deepcopy(term[key]) for key in copied}
        out.append({**inline, "file_path": entry.get("file_path")})
    return out


# --------------------------------------------------------------------------- #
# the document parser
# --------------------------------------------------------------------------- #


def check_protocol_version(raw: Mapping[str, Any]) -> None:
    """The first thing asked of any tree handed to the parser: is it an
    intervention specification of the version this loader reads (§1)?

    Answered *before* anything addresses the tree by path, so a ``--set`` or a
    workflow ``set`` on a document of the wrong shape is refused as that,
    rather than as a path that "does not exist". A **workflow** document is
    recognised by its ``steps`` section (workflow spec §1); the v1 spelling by
    its top-level ``version``, and the refusal names the verb that rewrites it.
    """
    if "steps" in raw:
        raise ParseError(
            "P2",
            "this is a workflow document (it has a 'steps' section), not an "
            "intervention specification — the workflow verbs read it",
            path="steps",
        )
    if "header" not in raw and ("version" in raw or "application" in raw):
        raise ParseError(
            "P2",
            "this is an intervention protocol v1 document (top-level 'version'); "
            f"this loader reads protocol_version {PROTOCOL_VERSION!r} — rewrite it "
            "with `causalab migrate <file>`",
            path="version",
        )
    header = raw.get("header")
    if not isinstance(header, Mapping):
        raise ParseError("P2", "missing required group 'header'", path="header")
    if "protocol_version" not in header:
        raise ParseError(
            "P2", "header needs 'protocol_version'", path="header.protocol_version"
        )
    version = header["protocol_version"]
    if version in MIGRATABLE_PROTOCOL_VERSIONS:
        raise ParseError(
            "P2",
            f"this is a protocol_version {version!r} document; this loader "
            f"reads protocol_version {PROTOCOL_VERSION!r} — rewrite it with "
            "`causalab migrate <file>` (version 4 lists reads on the models "
            "that take them, declares the un-intervened model, and carries "
            "each aggregation on the entry that consumes it; the verb carries "
            "the rewrite)",
            path="header.protocol_version",
        )
    if version != PROTOCOL_VERSION:
        raise ParseError(
            "P2",
            f"unsupported protocol_version {version!r}; this loader reads "
            f"protocol_version {PROTOCOL_VERSION!r}",
            path="header.protocol_version",
        )


def _warn_unconventional_order(
    keys: Sequence[str], order: Sequence[str], *, what: str
) -> None:
    """§5.2 — the §1 order is recommended, not required.

    Order carries no meaning downstream: `canonical.canonicalize` walks
    the recommended order and emits what it finds however it was authored, so
    a document written in another order has the same canonical bytes, the same
    digest and the same run as one written conventionally. Refusing it
    therefore refused a document that was already, byte for byte, the same
    experiment — most often one that had been through
    ``json.dumps(..., sort_keys=True)`` or a YAML round-trip on the way.

    What the order is *for* is reading, so what is left of the rule is a
    warning that names the order it recommends — once for the four groups,
    once for the method's sections.

    The retired half of this rule — ``save`` last — is subsumed: ``save`` is
    last in [`METHOD_SECTIONS`][], and its presence is required separately
    ([`REQUIRED_METHOD_SECTIONS`][]) and its contents by rule 10.
    """
    ranks = {name: i for i, name in enumerate(order)}
    if [ranks[k] for k in keys] == sorted(ranks[k] for k in keys):
        return
    recommended = [name for name in order if name in keys]
    warnings.warn(
        f"{what} are not in the recommended docs/intervention_protocol.md §1 "
        f"order: got {list(keys)}, recommended {recommended} — this parses, "
        f"digests and runs identically either way (§5 rule 2)",
        ProtocolWarning,
        stacklevel=3,
    )


def _free_text(header: Mapping[str, Any], field: str) -> str | None:
    value = header.get(field)
    if value is not None and not isinstance(value, str):
        raise ParseError("P2", f"{field} is free text", path=f"header.{field}")
    return value


def parse_document(raw: Mapping[str, Any]) -> Document:
    """Strict-parse a raw mapping into a [`Document`][].

    Owns §5.1 (strict keys, closed enums, no authored derived fields) and
    §5.2 (group and section order, which it warns about rather than refuses);
    cross-reference rules are
    `causalab.protocol.rules.document.validate_document`'s job. ``raw`` may
    be in any order; the order it *is* in is only used to warn (§5 rule 2).

    The tree is the four groups of §1. The header is checked first
    ([`check_protocol_version`][]), then the groups' key sets, then each
    section — every refusal names its path from the section, which is how
    every other dotted path in the protocol is spelled (§1).
    """
    check_protocol_version(raw)
    for key in raw:
        if key not in GROUP_ORDER:
            raise ParseError(
                "P3", f"unknown group {key!r}{suggest(key, GROUP_ORDER)}", path=key
            )
    _warn_unconventional_order(list(raw), GROUP_ORDER, what="groups")
    for group in GROUP_ORDER:
        if group not in raw:
            raise ParseError("P2", f"missing required group {group!r}", path=group)
    header = _require_mapping(raw["header"], "header")
    _check_keys(header, HEADER_FIELDS, "header")
    method = _require_mapping(raw["method"], "method")
    if "metrics" in method:
        raise ParseError(
            "P3",
            "the 'metrics' section is gone under protocol 4: an aggregation "
            "lives on the save entry, objective term or eval entry that "
            "consumes it (§2.10, §2.12) — " + _MIGRATE_HINT,
            path="metrics",
        )
    for key in method:
        if key not in METHOD_SECTIONS:
            raise ParseError(
                "P3",
                f"unknown section {key!r}{suggest(key, METHOD_SECTIONS)}",
                path=key,
            )
    _warn_unconventional_order(list(method), METHOD_SECTIONS, what="method sections")
    for section in METHOD_SECTIONS:
        if section in REQUIRED_METHOD_SECTIONS and section not in method:
            raise ParseError(
                "P2", f"missing required section {section!r}", path=section
            )
    authored_save = _parse_save(method["save"], "save")
    if not authored_save:
        raise ValidationError(10, "save must be non-empty", path="save")

    segments = (
        parse_segments(method["segments"], "segments") if "segments" in method else None
    )
    doc = Document(
        protocol_version=header["protocol_version"],
        title=_free_text(header, "title"),
        description=_free_text(header, "description"),
        model=_parse_model(raw["model"], "model"),
        data=_parse_data(raw["data"], "data"),
        segments=segments,
        positions=_parse_positions(method.get("positions", {}), "positions"),
        sites=_parse_sites(method["sites"], "sites"),
        featurizers=_parse_featurizers(method.get("featurizers", {}), "featurizers"),
        params=_parse_params(method.get("params", {}), "params"),
        code=_parse_code(method.get("code", {}), "code"),
        reads=_parse_reads(method["reads"], "reads"),
        writes=_parse_writes(method.get("writes", {}), "writes"),
        intervened_models=_parse_intervened_models(
            method["intervened_models"], "intervened_models"
        ),
        train=_parse_train(method["train"], "train") if "train" in method else None,
        save=(),
        raw=dict(raw),
    )
    # a `train` save entry resolves against the parsed fit, and before
    # `resolve_read_refs`, so a bare target it would copy is still bare
    save = tuple(
        _resolve_train_save(entry, doc.train)
        if isinstance(entry, _TrainSave)
        else entry
        for entry in authored_save
    )
    return resolve_read_refs(dataclasses.replace(doc, save=save))


def resolve_read_refs(doc: Document) -> Document:
    """The last step of [`parse_document`][]: bind every bare read
    reference to the one model that lists the read (§2.7).

    A write operand that names a read — a bare string, or a
    [`ReadRef`][] with no model — and a ``kl`` / ``js`` ``target`` are
    rewritten to the qualified [`ReadRef`][] when exactly one model lists
    the read (inside sweep wrappers too, per value). A reference that names
    a read listed by several models, or by none, keeps ``model=None`` and is
    validation's to refuse by name (rule 5); a bare string that names no
    read (a param, a slot) stays a string. Exported so
    `lower_bands` can rebind the members it
    fans a band read out to."""

    def bind(value: Any) -> Any:
        if isinstance(value, ReadRef):
            if value.model is None:
                return doc.bound(value.read)
            return value
        if isinstance(value, str) and value in doc.reads:
            return doc.bound(value)
        if isinstance(value, Sweep):
            return Sweep(values=tuple(bind(v) for v in value.values))
        return value

    def bind_do(do: Do) -> Do:
        payload = do.payload
        if do.mechanism == "swap":
            return dataclasses.replace(do, payload=bind(payload))
        if do.mechanism in ("add_scaled", "lerp") and isinstance(payload, Mapping):
            # both operand slots may name a read (§2.8): the value and its
            # coefficient
            return dataclasses.replace(
                do,
                payload={
                    **payload,
                    **{
                        slot: bind(payload[slot])
                        for slot in ("op", "alpha")
                        if slot in payload
                    },
                },
            )
        return do

    def bind_aggregation(spec: AggregationSpec | None) -> AggregationSpec | None:
        if spec is None or spec.kind not in READ_TARGET_METRIC_KINDS:
            return spec
        target = spec.fields.get("target")
        if not isinstance(target, ReadRef):
            return spec
        return dataclasses.replace(spec, fields={**spec.fields, "target": bind(target)})

    writes = {
        name: dataclasses.replace(write, do=bind_do(write.do))
        for name, write in doc.writes.items()
    }
    save = tuple(
        dataclasses.replace(entry, aggregation=bind_aggregation(entry.aggregation))
        if entry.aggregation is not None
        else entry
        for entry in doc.save
    )
    train = doc.train
    if train is not None:
        objective = tuple(
            dataclasses.replace(term, aggregation=bind_aggregation(term.aggregation))
            if term.aggregation is not None
            else term
            for term in train.objective
        )
        eval_spec = train.eval
        if eval_spec is not None and "aggregations" in eval_spec:
            eval_spec = {
                **eval_spec,
                "aggregations": {
                    label: (read, bind_aggregation(spec))
                    for label, (read, spec) in eval_spec["aggregations"].items()
                },
            }
        train = dataclasses.replace(train, objective=objective, eval=eval_spec)
    return dataclasses.replace(doc, writes=writes, save=save, train=train)
