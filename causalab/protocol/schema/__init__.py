"""The intervention-protocol object model and its strict parser.

This package is the authoring surface of ``docs/intervention_protocol.md``:
it turns a raw JSON/YAML mapping into a typed, frozen
[`Document`][] — or refuses with a structured
`causalab.protocol.rules.errors.ParseError` /
`causalab.protocol.rules.errors.ValidationError`. Everything here is
engine-free and torch-free: sites, positions, featurizers and writes are
pure data records; an engine interprets them (spec §8).

Four modules import in one direction at run time: ``types`` first, then
``featurizers`` and ``positions``, then ``parse``. A fifth module,
[`explicit`][] (the canonical form, §7), is
imported by its own path. The registry imports this package and ``explicit``
imports the registry, so the root does not import ``explicit``. The four:

* [`types`][] — the closed vocabularies, the value
  wrappers ([`Sweep`][],
  [`ArtifactRef`][causalab.protocol.schema.types.ArtifactRef]) and the §2 section
  records through [`Document`][];
* [`featurizers`][] — the featurizer vocabulary
  (§2.5), the gate maps and legality tables, and [`FeaturizerSpec`][];
* [`positions`][] — [`PositionSpec`][], the
  span grammar (§2.3) and the ``segments`` section (§2.2.1);
* [`parse`][] — the raw loader, the parse helpers,
  the per-section parsers and [`parse_document`][].

The package root exports the public names that other modules, scripts,
tests, and generated docs blocks import from ``causalab.protocol.schema``.
``tests/protocol/test_schema_package.py`` pins that surface. Import a private
helper from the submodule that defines it. A new root export goes into
``__all__`` here and into the pin.

Parsing owns the *shape* rules of the spec:

* strict keys — an unknown field anywhere is an error with suggestions
  (§5.1); closed enums reject with suggestions; derived fields (§6) may not
  be authored;
* section order — a *recommendation* (§1), not a rule: an unconventional
  order warns and parses on (§5.2);
* sugar — a bare int ``pos`` means ``{"index": n}`` and the bare string
  ``"all"`` means ``{"all": true}`` (§2.3); sugar is expanded here, so the
  object model only ever holds the canonical spelling;
* the two value wrappers — ``{"sweep": …}`` (§3) and
  ``{"artifact": …, "key": …}`` (§1) — are accepted anywhere a scalar-,
  list- or spec-typed *leaf* is expected and preserved as [`Sweep`][] /
  [`ArtifactRef`][causalab.protocol.schema.types.ArtifactRef] values; expansion and
  resolution happen in `causalab.protocol.lowering` /
  [`causalab.io.sources`][].

Cross-reference and semantic checks (the §5 checklist items that need the
whole document) live in `causalab.protocol.rules.document`.
"""

from __future__ import annotations

from causalab.protocol.schema.types import (
    ADDITIVE_MECHANISMS,
    ALIGNMENT_CARDINALITIES,
    AlignmentCardinality,
    ALL_POSITIONS,
    AggregationSpec,
    BoundAggregation,
    CodeSpec,
    COMPONENTS,
    concrete_int,
    concrete_str,
    ConstraintSpec,
    DataRole,
    DEPRECATED_COMPONENTS,
    DEPRECATED_IN,
    Do,
    do_operand_slots,
    Document,
    dotted_path,
    GROUP_ORDER,
    HEADER_FIELDS,
    IMSpec,
    WRITES_DURING_GENERATION_FIELD,
    LAYERLESS_COMPONENTS,
    MATCH_MODES,
    MECHANISMS,
    METHOD_SECTIONS,
    metric_column_fields,
    METRIC_DOMAINS,
    METRIC_FIELD_DEFAULTS,
    METRIC_FIELDS,
    METRIC_KINDS,
    MIGRATABLE_PROTOCOL_VERSIONS,
    MINIMUM_COUNT_FIELD,
    MODEL_DTYPE_DEFAULT,
    ModelRef,
    NAMED_SECTIONS,
    ObjectiveTerm,
    operand_params,
    operand_reads,
    OPTIMIZER_DEFAULTS,
    OPTIMIZER_SCHEDULES,
    OPTIONAL_METRIC_FIELDS,
    PRECISION_DTYPES,
    PROTOCOL_VERSION,
    RAGGED_FIELD,
    RAGGED_POLICIES,
    READ_TARGET_METRIC_KINDS,
    read_is_vocabulary,
    ReadRef,
    ReadSpec,
    REQUIRED_METHOD_SECTIONS,
    RESERVED_NAMES,
    RowRole,
    SAVE_KINDS,
    SaveEntry,
    SECTION_ORDER,
    SiteSpec,
    Stream,
    STREAMS,
    Sweep,
    RETIRED_TOKEN_FORMS,
    TOKEN_FORMS,
    TrainSpec,
    tree_path,
    VOCAB_TOP_K_RANKING,
    WHOLE_WINDOW_METRIC_KINDS,
    WriteSpec,
)
from causalab.protocol.schema.featurizers import (
    AnnealSchedule,
    CONTROL_DEFAULTS,
    FEATURIZER_FAMILIES,
    FEATURIZER_FIELD_CONDITIONS,
    FEATURIZER_FIELDS,
    FEATURIZER_KINDS,
    FEATURIZER_SLOTS,
    FeaturizerSpec,
    FORWARD_MASKS,
    GATE_AXES,
    GATE_DEAD_RULES,
    GATE_DEFAULT_MAP,
    GATE_GROUP_AXES,
    GATE_GROUPS,
    GATE_MAPS,
    GATE_PARAMETRIZATIONS,
    HARD_CONCRETE_STRETCH,
    HARD_CONCRETE_TEMPERATURE,
    hard_concrete_theta,
    hard_concrete_threshold,
    K_SCHEDULE_OF,
    OBJECTIVE_WEIGHT_PREFIX,
    PhaseSpec,
    render_featurizer_kind_table,
    render_field_legality_table,
    render_field_refusals,  # read by the `generated: call` blocks in docs/methods/*.md
    render_gate_map_table,
    TRAINABLE_KINDS,
)
from causalab.protocol.schema.positions import (
    CONTINUATION_SEGMENT,
    PositionSpec,
    SegmentsSpec,
    span_length,
    SpanSpec,
)
from causalab.protocol.schema.parse import (
    check_protocol_version,
    DRAW_KINDS,
    inline_train_saves,
    load_raw,
    parse_document,
    PER_PARAMS_OPTIMIZER_FIELDS,
    REGULARIZER_COSTS,
    REGULARIZER_KINDS,
    SAVE_REDUCTIONS,
    SCORES_INIT_DEFAULTS,
    SCORES_INIT_KEYS,
    to_base_form,
)

__all__ = [
    "ADDITIVE_MECHANISMS",
    "ALIGNMENT_CARDINALITIES",
    "AlignmentCardinality",
    "ALL_POSITIONS",
    "AggregationSpec",
    "AnnealSchedule",
    "BoundAggregation",
    "check_protocol_version",
    "CodeSpec",
    "COMPONENTS",
    "concrete_int",
    "concrete_str",
    "ConstraintSpec",
    "CONTINUATION_SEGMENT",
    "CONTROL_DEFAULTS",
    "DataRole",
    "DEPRECATED_COMPONENTS",
    "DEPRECATED_IN",
    "Do",
    "do_operand_slots",
    "Document",
    "dotted_path",
    "DRAW_KINDS",
    "FEATURIZER_FAMILIES",
    "FEATURIZER_FIELD_CONDITIONS",
    "FEATURIZER_FIELDS",
    "FEATURIZER_KINDS",
    "FEATURIZER_SLOTS",
    "FeaturizerSpec",
    "FORWARD_MASKS",
    "GATE_AXES",
    "GATE_DEAD_RULES",
    "GATE_DEFAULT_MAP",
    "GATE_GROUP_AXES",
    "GATE_GROUPS",
    "GATE_MAPS",
    "GATE_PARAMETRIZATIONS",
    "GROUP_ORDER",
    "HARD_CONCRETE_STRETCH",
    "HARD_CONCRETE_TEMPERATURE",
    "hard_concrete_theta",
    "hard_concrete_threshold",
    "HEADER_FIELDS",
    "IMSpec",
    "inline_train_saves",
    "WRITES_DURING_GENERATION_FIELD",
    "K_SCHEDULE_OF",
    "LAYERLESS_COMPONENTS",
    "load_raw",
    "MATCH_MODES",
    "MECHANISMS",
    "METHOD_SECTIONS",
    "metric_column_fields",
    "METRIC_DOMAINS",
    "METRIC_FIELD_DEFAULTS",
    "METRIC_FIELDS",
    "METRIC_KINDS",
    "MIGRATABLE_PROTOCOL_VERSIONS",
    "MINIMUM_COUNT_FIELD",
    "MODEL_DTYPE_DEFAULT",
    "ModelRef",
    "NAMED_SECTIONS",
    "OBJECTIVE_WEIGHT_PREFIX",
    "ObjectiveTerm",
    "operand_params",
    "operand_reads",
    "OPTIMIZER_DEFAULTS",
    "OPTIMIZER_SCHEDULES",
    "OPTIONAL_METRIC_FIELDS",
    "parse_document",
    "PER_PARAMS_OPTIMIZER_FIELDS",
    "PhaseSpec",
    "PositionSpec",
    "PRECISION_DTYPES",
    "PROTOCOL_VERSION",
    "RAGGED_FIELD",
    "RAGGED_POLICIES",
    "READ_TARGET_METRIC_KINDS",
    "read_is_vocabulary",
    "ReadRef",
    "ReadSpec",
    "REGULARIZER_COSTS",
    "REGULARIZER_KINDS",
    "render_featurizer_kind_table",
    "render_field_legality_table",
    "render_field_refusals",
    "render_gate_map_table",
    "REQUIRED_METHOD_SECTIONS",
    "RESERVED_NAMES",
    "RowRole",
    "SAVE_KINDS",
    "SAVE_REDUCTIONS",
    "SaveEntry",
    "SCORES_INIT_DEFAULTS",
    "SCORES_INIT_KEYS",
    "SECTION_ORDER",
    "SegmentsSpec",
    "SiteSpec",
    "span_length",
    "SpanSpec",
    "Stream",
    "STREAMS",
    "Sweep",
    "RETIRED_TOKEN_FORMS",
    "to_base_form",
    "TOKEN_FORMS",
    "TRAINABLE_KINDS",
    "TrainSpec",
    "tree_path",
    "VOCAB_TOP_K_RANKING",
    "WHOLE_WINDOW_METRIC_KINDS",
    "WriteSpec",
]
