"""The protocol object model: closed vocabularies, value wrappers and the §2
section records.

The runtime leaf of [`causalab.protocol.schema`][]: it imports no sibling of
the package at run time, so a consumer that wants only the typed records
([`Document`][], [`SiteSpec`][], [`ModelRef`][], …) or a vocabulary
([`COMPONENTS`][], [`MECHANISMS`][], [`METRIC_KINDS`][], …) pays for no
parser. [`FeaturizerSpec`][] and
[`PositionSpec`][] /
[`SegmentsSpec`][] are referenced in
annotations only. Everything here is engine-free and torch-free: pure data
records an engine interprets (spec §8).
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any, Literal, Mapping, Sequence, TYPE_CHECKING, Union, get_args

from causalab.protocol.rules.errors import ParseError

if TYPE_CHECKING:
    from causalab.protocol.schema.featurizers import (
        AnnealSchedule,
        FeaturizerSpec,
        PhaseSpec,
    )
    from causalab.protocol.schema.positions import PositionSpec, SegmentsSpec


# --------------------------------------------------------------------------- #
# closed vocabularies (spec §2.4, §2.5, §2.8, §2.10)
# --------------------------------------------------------------------------- #

Component = Literal[
    "input_ids",
    "embeddings",
    "block_input",
    "attention_input_norm",
    "delta_qkv",
    "delta_gate",
    "delta_conv",
    "delta_query",
    "delta_key",
    "delta_value",
    "delta_beta",
    "delta_decay",
    "delta_kv_mem",
    "delta_state_update",
    "delta_state",
    "delta_kernel_output",
    "attention_query_pre_rope",
    "attention_key_pre_rope",
    "attention_value_states",
    "attention_gate",
    "attention_query",
    "attention_key",
    "attention_scores",
    "attention_z",
    "deltanet_query",
    "deltanet_key",
    "deltanet_state",
    "attention_result",
    "delta_premix",
    "attention_output",
    "attention_premix",
    "attention_probs",
    "block_mid",
    "mlp_input_norm",
    "mlp_input",
    "mlp_output",
    "mlp_activation",
    "mlp_neuron_output",
    "router_logits",
    "router_scores",
    "expert_idx",
    "expert_gate_proj",
    "expert_up_proj",
    "expert_activation",
    "expert_neuron_output",
    "expert_permutation",
    "expert_output",
    "routed_output",
    "shared_expert_gate_proj",
    "shared_expert_up_proj",
    "shared_expert_activation",
    "shared_expert_output",
    "shared_expert_gate",
    "block_output",
    "ln_final",
    "lm_head",
]

#: The closed site component vocabulary (§2.4). The order here is the
#: literal's, not §2.4's: the spec lists the same 56 names in a reading
#: order that walks a block, and nothing depends on either ordering: the
#: census in tests/protocol/test_vocabulary_census.py compares the *sets*.
COMPONENTS: tuple[Component, ...] = get_args(Component)

#: Retired spellings, mapped to the name that replaced them. Applied at parse,
#: so ``SiteSpec.component`` is always current and nothing downstream: the
#: shape table, the rank table, the tap table: carries a second name.
#:
#: 🔤 ``attention_value`` named the o-projection's **input**: on a gated
#: attention family (Qwen3.5/3.6) that is ``z · σ(gate)``, on an ungated one it
#: is ``z``, and on neither is it the value vectors. Those are exposed
#: separately, as ``attention_value_states``: so the old name had to move
#: before the collision, not after it (nnterp issue #51,
#: https://github.com/ndif-team/nnterp/issues/51, is the same mistake, made
#: after).
#:
#: The replacement name is **not** reused for the new box, deliberately: a
#: document written against the old vocabulary would then load, and silently
#: mean a different tensor. An alias that redirects is safe; an alias that
#: rebinds is the failure this whole rename exists to avoid.
#:
#: 🔤 The eight ``deltanet_*`` spellings below named the nnsight engine's
#: ``.source`` reach into the Gated DeltaNet forward, while ``delta_*`` named
#: the reference engine's kernel-boundary reach: two vocabularies for **the
#: same physical tensors** (📐 measured identical in shape and value on
#: ``tiny-random/qwen3.5-moe``, ``tests/_helpers/a3b_sweep.py``'s pair table).
#: That was a leak: a document had to name the *backend*
#: to name the tensor. The vocabulary keeps one canonical, engine-neutral spelling
#: per tensor: ``delta_*`` names the delta rule the tensor belongs to, where
#: ``deltanet_core_out`` / ``deltanet_gated_out`` / ``deltanet_qkv_conv`` echo
#: the modeling file's variable names: and each engine translates it to its
#: own mechanism. Only pairs whose two spellings agree in **shape and timing**
#: are aliased: ``deltanet_query`` / ``deltanet_key`` (pre GVA tiling,
#: key-head space) and ``deltanet_state`` (per 64-token chunk, not per step)
#: stay their own names with a typed backend requirement
#: (``registry.BACKEND_PAIRS``), because an alias there would *rebind*, not
#: redirect: ``registry.alias_would_rebind`` is the guard, and the census
#: holds every entry here to it.
DEPRECATED_COMPONENTS: dict[str, Component] = {
    "attention_value": "attention_premix",
    "deltanet_qkv": "delta_qkv",
    "deltanet_qkv_conv": "delta_conv",
    "deltanet_gate": "delta_gate",
    "deltanet_value": "delta_value",
    "deltanet_beta": "delta_beta",
    "deltanet_decay": "delta_decay",
    "deltanet_core_out": "delta_kernel_output",
    "deltanet_gated_out": "delta_premix",
}

#: The protocol ``version`` under which each retired spelling became an alias
#:: the version *from which* a document may still author it and canonicalize
#: to the replacement. Every entry is ``"1"`` today: the one protocol version
#: there is, and both folds were made within it. The field is
#: per alias so that a spelling retired under a later version records that
#: version, not the table's oldest.
DEPRECATED_IN: dict[str, str] = {alias: "1" for alias in DEPRECATED_COMPONENTS}

#: The mixer a layer carries. A hybrid tower has both: 📐 on
#: ``tiny-random/qwen3.5-moe`` three of four layers are Gated DeltaNet
#: (``linear_attention``) and one is ``full_attention``: so a site may name the
#: stream it means and be refused at load if the layer it names carries the
#: other one.
#:
#: 🐞 This once parsed as an **integer**, which made the field
#: unusable from either side: ``model_tree._check_stream`` only reads a *string*
#: (``bundle.stream_at`` returns one), so an authored ``"full_attention"`` was
#: rejected by the parser before the check could see it, and an authored ``0``
#: parsed and was then silently ignored: precisely the failure ``_moe_site``'s
#: ``expert`` refusal was written to avoid. The earlier tests missed it because
#: they construct ``SiteSpec(stream="full_attention")`` directly, exercising a
#: path no document can reach.
Stream = Literal["full_attention", "linear_attention"]

#: Every stream a site may name.
STREAMS: tuple[Stream, ...] = get_args(Stream)

#: Components that carry no ``layers`` field.
#: ``input_ids`` joins these because it is the model's *input* (§5.4), not an
#: activation inside a block: there is no layer at which to read it.
LAYERLESS_COMPONENTS: frozenset[str] = frozenset(
    {"input_ids", "embeddings", "ln_final", "lm_head"}
)

Mechanism = Literal[
    "swap",
    "add_scaled",
    "lerp",
    "affine",
    "gaussian",
    "renormalize",
    "clamp",
    "pytorch_fn",
]
#: The closed ``do`` mechanism set (§2.8).
MECHANISMS: tuple[Mechanism, ...] = get_args(Mechanism)

#: Mechanisms whose write is a delta added after the absolute write (§2.8).
ADDITIVE_MECHANISMS: frozenset[str] = frozenset({"add_scaled", "gaussian"})

MetricKind = Literal[
    "logit_diff",
    "soft_accuracy",
    "token_logit",
    "cross_entropy",
    "kl",
    "js",
    "class_probs",
    "token_logits",
    "top_k",
    "match",
    "decode",
]
METRIC_KINDS: tuple[MetricKind, ...] = get_args(MetricKind)

#: Value fields per metric kind beyond ``of`` (§2.10). ``kl.target`` and
#: ``js.target`` name a read ([`READ_TARGET_METRIC_KINDS`][]); every other
#: value field names a dataset column (checked at run time by
#: ``validate --data``, §2.2): except the fields in
#: [`NON_COLUMN_METRIC_FIELDS`][]. [`metric_column_fields`][] is the one
#: function that applies those exceptions, so the loader, the eligibility
#: predicate and the run cannot disagree about which fields are columns.
METRIC_FIELDS: dict[str, tuple[str, ...]] = {
    "logit_diff": ("a", "b"),
    "soft_accuracy": ("a", "b"),
    "token_logit": ("token",),
    "cross_entropy": ("target",),
    "kl": ("target",),
    "js": ("target",),
    "class_probs": ("groups",),
    "token_logits": ("tokens",),
    "top_k": ("k", "by"),
    "match": ("expected",),
    "decode": (),
}

#: Mandatory value fields that are *not* dataset column names: ``top_k.k`` is
#: an integer, ``top_k.by`` is a closed enum, and ``token_logits.tokens`` is a
#: list of literal token strings: an answer space is a property of the run,
#: exactly as ``class_probs.groups`` is (a mapping, which the column check
#: already skips by shape). Everything else in [`METRIC_FIELDS`][] (bar
#: ``kl.target``, a read) is checked against the resolved datasets' columns.
NON_COLUMN_METRIC_FIELDS: frozenset[str] = frozenset({"k", "by", "tokens"})

#: Metric kinds whose ``target`` is a **read**, not a column (§2.10): ``kl``
#: and ``js`` compare two reads' distributions against each other. The
#: planner materializes the target read for them, the executor hands its
#: value to the reduction, and the column checks skip the field.
READ_TARGET_METRIC_KINDS: frozenset[str] = frozenset({"kl", "js"})

#: What each metric kind consumes from its read (§2.10). ``distribution``
#: kinds reduce the read's dense value at the addressed positions (the
#: vocabulary projection for an ``lm_head`` read, which is every kind but
#: ``top_k``); ``ids`` kinds consume only the tokens the decode produced. The
#: split is what lets the planner (§8) tell a text probe: which obliges no
#: vocabulary projection at all: from a scoring one, so it is a property of
#: the kind, never of the document.
MetricDomain = Literal["distribution", "ids"]
METRIC_DOMAINS: dict[str, MetricDomain] = {
    "logit_diff": "distribution",
    "soft_accuracy": "distribution",
    "token_logit": "distribution",
    "cross_entropy": "distribution",
    "kl": "distribution",
    "js": "distribution",
    "class_probs": "distribution",
    "token_logits": "distribution",
    "top_k": "distribution",
    "match": "distribution",
    "decode": "ids",
}

#: Metric kinds that reduce the whole addressed window to one value per
#: example rather than one per position: ``decode`` joins its tokens into a
#: string, so a per-step row would be a per-character-ish lie.
WHOLE_WINDOW_METRIC_KINDS: frozenset[str] = frozenset({"decode"})


TokenForm = Literal["id"]

#: The one authored token form (§2.10). A metric's answer strings are
#: tokenized **as written**: ``" Seattle"`` and ``"Seattle"`` name the two
#: gpt2 rows 7312 and 34007, and the row that fixes the prompt fixes the
#: answer's form with it. ``token_form: "id"`` is the only key a document may
#: set: the column holds integer vocabulary ids (the decode readouts of
#: ``causalab.analysis.sequences.add_readouts`` write ids, not text). The
#: retired values ``auto`` / ``bare`` / ``space_prefixed`` used to rewrite the
#: leading space for the author; [`RETIRED_TOKEN_FORMS`][] keeps their names
#: so the parser can say what replaced them.
TOKEN_FORMS: tuple[TokenForm, ...] = get_args(TokenForm)
RETIRED_TOKEN_FORMS: tuple[str, ...] = ("auto", "bare", "space_prefixed")

#: Per retired ``token_form``, how to rewrite one answer string ``s`` so it
#: keeps the token the retired value scored, as a clause a refusal completes
#: with "rewrite each answer string s ..." (§2.10). ``bare`` and
#: ``space_prefixed`` fixed one rewrite. ``auto`` chose a form per string with
#: the tokenizer, so its rewrite needs the tokenizer. The parser and
#: ``causalab migrate`` both state it.
RETIRED_TOKEN_FORM_REWRITES: Mapping[str, str] = {
    "bare": "as s.lstrip(' ')",
    "space_prefixed": "as ' ' + s.lstrip(' ')",
    "auto": (
        "as the model emits it. 'auto' used ' ' + s.lstrip(' ') when that was "
        "one token and s.lstrip(' ') otherwise, so the rewrite needs the tokenizer"
    ),
}

#: How one position maps across a pair's inputs (§2.3): the cardinality of
#: the token runs the same address resolves to on the base input and on a
#: counterfactual input. Authored as the optional ``alignment`` key of a
#: position entry: the member the author *declares*: and derived at encode
#: time as the *observed* one by ``causalab.protocol.positions.alignment.alignment_of``,
#: the one function planning and execution both call. An ``index`` and
#: an unscoped ``span`` are ``one_to_one`` by construction (one token, or one
#: joint span of the same width, on every input: two ``index`` specs are two
#: locations, one ``span`` is one joint address); a ``variable`` or ``column``
#: window is whatever the tokenizer says. ``absent`` and ``ambiguous`` are the
#: two unalignable values, and they are §2.4's ``alignment_missing`` and
#: ``alignment_ambiguous`` reason codes. Optional with **no default**: an
#: unauthored key stays absent through the canonical form, so no digest moves.
AlignmentCardinality = Literal[
    "one_to_one", "one_to_many", "many_to_one", "absent", "ambiguous"
]
ALIGNMENT_CARDINALITIES: tuple[AlignmentCardinality, ...] = get_args(
    AlignmentCardinality
)

#: Metric kinds whose value fields carry answers that resolve to token ids:
#: the kinds that accept ``token_form: "id"``. ``kl`` compares two reads'
#: distributions and ``top_k`` reports indices it found (decoding them only
#: when the read happens to tap ``lm_head``), so neither resolves an authored
#: answer and neither accepts the key.
TOKEN_COLUMN_METRIC_KINDS: frozenset[str] = frozenset(
    {
        "logit_diff",
        "soft_accuracy",
        "token_logit",
        "cross_entropy",
        "class_probs",
        "token_logits",
        "match",
    }
)

#: Optional value fields per metric kind (§2.10). An omitted optional field
#: **with a default** ([`METRIC_FIELD_DEFAULTS`][]) is materialized to it in
#: the canonical form (§7), so an authored default and an omitted one digest
#: identically: the same treatment ``train.optimizer`` defaults get. An
#: optional field with **no** default (``js.restrict``) stays absent when
#: unauthored, like ``minimum_count``: absent means "unrestricted", and an
#: absent field keeps the digest.
OPTIONAL_METRIC_FIELDS: dict[str, tuple[str, ...]] = {
    "match": ("mode",),
    "js": ("restrict",),
}

#: Defaults for the optional fields above, by ``(kind, field)``.
METRIC_FIELD_DEFAULTS: dict[tuple[str, str], Any] = {
    ("match", "mode"): "exact",
}

#: The optional decision threshold on any metric kind (§2.10 "Eligibility"):
#: the fewest eligible rows the metric's decision rule needs. Not in
#: [`OPTIONAL_METRIC_FIELDS`][] because it has no default to materialize :
#: absent means "no threshold", and an absent field keeps the digest (§7).
#: Checked by ``validate --data`` against the resolved base table's maximum
#: eligible count (``loader.check_data_columns``, rule 4).
MINIMUM_COUNT_FIELD = "minimum_count"

#: The optional ragged-window policy of a write (§2.8, §5 rule 19): how a
#: write whose rows address different numbers of positions (an ``all``,
#: ``variable``, ``column`` or span window) lands. Spelled as a one-key
#: object, ``{"policy": <one of RAGGED_POLICIES>}``, so a later knob has a
#: place beside the policy. Like [`MINIMUM_COUNT_FIELD`][] it has no
#: default to materialize: absent means ``refuse`` (rule 19's refusal, the
#: behaviour every document had before the field existed), and an absent
#: field keeps the digest (§7). Vocabulary-checked at parse; resolved by the
#: run on the encoded batch, where the tokenizer is (never by the pure verbs).
RAGGED_FIELD = "ragged"

#: The closed policy set (§2.8): ``refuse``: rule 19 as before;
#: ``exact_length_buckets``: the rows land grouped by width, one gather per
#: width; ``padded_masked``: the rows land through one padded gather and a
#: mask, padding never written. Both non-refusing policies land every row at
#: its own width before any forward and change no batch geometry.
RAGGED_POLICIES: tuple[str, ...] = ("refuse", "exact_length_buckets", "padded_masked")

#: How ``match`` compares the argmax token to an expected form (§2.10):
#: ``exact`` needs the form to be one token; ``first_token`` credits the
#: form's first token, which is what "prefix" means with logits at one
#: position (a multi-token answer's first piece).
MATCH_MODES: tuple[str, ...] = ("exact", "first_token")

#: How ``top_k`` ranks the entries of its read's last axis (§2.10). Mandatory,
#: because the right answer depends on what the axis *is* and only the author
#: knows: a vocabulary projection has no meaningful negative entries, while a
#: residual stream and a signed feature code do: ranking a 100k-latent SAE
#: code by signed value and by magnitude give different top-k sets, and
#: silently picking one for the author is how a plot ends up meaning something
#: other than its caption.
#:
#: * ``value``: the k largest signed entries. Any read.
#: * ``abs_value``: the k largest by ``|x|``; the reported value stays signed.
#:   Any read.
#: * ``prob``: softmax the last axis, then take the k largest probabilities.
#:   Legal **only** on an ``lm_head`` read: a softmax across neurons or SAE
#:   latents normalizes over an axis that is not an event space, and the
#:   resulting "probabilities" would mean nothing (validation refuses it).
TOP_K_RANKINGS: tuple[str, ...] = ("value", "abs_value", "prob")

#: The ``top_k.by`` ranking that normalizes, and so is vocabulary-only.
VOCAB_TOP_K_RANKING: str = "prob"

#: The bare-string spelling of an all-positions spec (§2.3 sugar). Reserved as
#: a name so a ``positions`` entry can never shadow the sugar.
ALL_POSITIONS: str = "all"

#: Names no section may declare (§1): the input roles, the
#: indexed-counterfactual family (checked by prefix for ``counterfactual[``),
#: and the all-positions sugar. ``original`` is **not** reserved: the
#: un-intervened model is a declared model with no writes (§2.9), named by
#: the author like any other.
RESERVED_NAMES: frozenset[str] = frozenset({"base", "counterfactual", ALL_POSITIONS})

#: The one value ``header.protocol_version`` may hold (§1). A string, as v1's
#: ``version`` was: the loader compares it, never orders it, and a JSON
#: integer would make ``"4"`` versus ``4`` a refusal of its own.
PROTOCOL_VERSION: str = "4"

#: Earlier ``protocol_version`` values ``causalab migrate`` rewrites (§7, §9):
#: a document declaring one is refused by name and told the verb. ``"2"`` is
#: the four-group form whose sites spelled a scalar ``layer``; version 3
#: renamed the field to ``layers``, a band of layer indices (§2.4); version 4
#: reorganized the method block reads-first — reads are listed on the models
#: that take them, the un-intervened model is declared, and the ``metrics``
#: section dissolved into ``aggregation`` blocks on the entries that consume
#: them (§2.9, §2.10, §2.12).
MIGRATABLE_PROTOCOL_VERSIONS: tuple[str, ...] = ("2", "3")

#: The four groups of an intervention specification, in recommended order
#: (§1): the header names the file, ``model`` and ``data`` name what the
#: experiment ran on, and ``method`` is the experiment itself.
GROUP_ORDER: tuple[str, ...] = ("header", "model", "data", "method")

#: What the ``header`` group may hold (§1). ``title`` and ``description`` are
#: authoring metadata: they say what a file is *for*: and canonicalization
#: drops them; ``protocol_version`` is content, and stays.
HEADER_FIELDS: tuple[str, ...] = ("protocol_version", "title", "description")

#: The eleven sections of the ``method`` group, in recommended order (§1):
#: the models first — what runs, and which reads and writes each carries —
#: then the addresses and maps they name, the reads, the writes, the fit,
#: the manifest.
METHOD_SECTIONS: tuple[str, ...] = (
    "intervened_models",
    "segments",
    "positions",
    "sites",
    "featurizers",
    "params",
    "code",
    "reads",
    "writes",
    "train",
    "save",
)

#: The ``save`` entry kinds that are not a saved read / metric / featurizer
#: (§2.12): ``location_ledger``: the run's resolved token indices as a
#: table (``protocol/positions/ledger.py``); ``trajectory``: a fit's checkpoints as a
#: bundle; ``rank``: every gate's units ordered by ``theta``, as a table
#: (``neural/shared/execution.py``), the object a top-k readout
#: ([`FeaturizerSpec.top_k`][]) reads off and the record that makes a
#: ranking method's result inspectable without the bundle. Opt-in: an entry
#: of a kind is the only thing that makes a run write one.
SAVE_KINDS: tuple[str, ...] = ("location_ledger", "trajectory", "rank")

#: How a ``trajectory`` entry (§2.12) spaces its checkpoints: ``count``: n
#: checkpoints equally spaced over the run, the last at its end: or every n
#: ``updates`` / ``epochs`` (the ``_parse_counter`` units), the last update
#: always included.
TRAJECTORY_EVERY_UNITS: tuple[str, ...] = ("count", "updates", "epochs")

REQUIRED_METHOD_SECTIONS: frozenset[str] = frozenset(
    {"intervened_models", "sites", "reads", "save"}
)

#: Every addressable section, in recommended order: the two input sections and
#: the method's eleven. A section name is unique across the groups, so a dotted
#: path (``--set``, a sweep axis id, a workflow's ``emit``) starts at the
#: section and never spells the group: [`tree_path`][] finds it (§1).
SECTION_ORDER: tuple[str, ...] = ("model", "data", *METHOD_SECTIONS)

REQUIRED_SECTIONS: frozenset[str] = (
    frozenset({"model", "data"}) | REQUIRED_METHOD_SECTIONS
)


def tree_path(dotted: str) -> tuple[str, ...]:
    """A section-rooted dotted path as a path into the document tree (§1):
    ``sites.target.layers`` → ``("method", "sites", "target", "layers")``,
    ``model.dtype`` → ``("model", "dtype")``. A path that starts at a group or
    at nothing known is returned as written, so the caller's "does not exist"
    refusal names what was typed. An index on the first segment
    (``save[0].file_path``: ``save`` is the method's one list) is part of
    that segment, not of the section's name."""
    parts = tuple(dotted.split("."))
    if parts and parts[0].split("[", 1)[0] in METHOD_SECTIONS:
        return ("method", *parts)
    return parts


def dotted_path(path: Sequence[str]) -> str:
    """The inverse of [`tree_path`][]: a tree path as its section-rooted
    dotted spelling, the group dropped."""
    parts = tuple(path)
    if (
        len(parts) >= 2
        and parts[0] == "method"
        and parts[1].split("[", 1)[0] in METHOD_SECTIONS
    ):
        parts = parts[1:]
    return ".".join(parts)


#: The name-bearing sections sharing one global namespace (§1: method sections 1–8).
NAMED_SECTIONS: tuple[str, ...] = (
    "intervened_models",
    "positions",
    "sites",
    "featurizers",
    "params",
    "code",
    "reads",
    "writes",
)

PRECISION_DTYPES: tuple[str, ...] = ("fp32", "bf16", "fp16")

#: The compute dtype a document runs in when it authors none (§2.1). This is
#: the value canonicalization materializes, so every canonical form names a
#: dtype and every digest covers it; ``fp32`` keeps an unauthored document
#: running exactly as it did when dtype was an execution flag.
MODEL_DTYPE_DEFAULT: str = "fp32"

#: Full-attention backends supported by the document vocabulary. Omission
#: retains the engine's default; an explicit choice is part of the experiment.
ATTENTION_IMPLEMENTATIONS: tuple[str, ...] = ("eager", "sdpa", "flash_attention_2")

#: Weight-quantization schemes (§2.1). Each names one *load-time* scheme the
#: reference engine can realize through bitsandbytes: ``int8`` is LLM.int8()
#: mixed-precision decomposition, ``nf4`` / ``fp4`` are the two 4-bit
#: quantization types. There is no bare ``int4``: bitsandbytes' 4-bit is one
#: of these two datatypes, and "int4" would not say which: the whole point
#: of putting the field in the record is that it names one realization.
#: Weights quantized *ahead of time* (GPTQ, AWQ) are a property of the
#: checkpoint, so they are named by ``model.key``/``revision``, not here.
QUANT_SCHEMES: tuple[str, ...] = ("int8", "nf4", "fp4")

#: Quantizers (§2.1). One entry in v1: the library the reference engine
#: calls; naming it keeps a document honest when a second one appears.
QUANT_METHODS: tuple[str, ...] = ("bitsandbytes",)

#: Quantization fields that only make sense for some schemes (rule 17).
_QUANT_4BIT_FIELDS: tuple[str, ...] = ("double_quant",)
_QUANT_INT8_FIELDS: tuple[str, ...] = ("int8_threshold",)

#: Optimizer field vocabulary and per-name defaults, materialized into the
#: canonical form (§7: "every default (constant LR, optimizer betas, dtypes)").
#: The vocabulary is closed like every other enum here; extending it is a
#: schema change, not a free-form pass-through.
OPTIMIZER_FIELDS: frozenset[str] = frozenset(
    {
        "name",
        "lr",
        "weight_decay",
        "betas",
        "eps",
        "momentum",
        "clip_grad_norm",
        "schedule",
        "warmup_frac",
    }
)
#: The learning-rate schedules an optimizer may follow (§2.11 ``optimizer.schedule``):
#: ``constant``: the authored ``lr`` at every update, the default and the only
#: value the field had before 2026-09-10 (it was parsed and ignored); and
#: ``linear_warmup_decay``: HF's ``get_linear_schedule_with_warmup``: the lr
#: climbs linearly from 0 over the first ``warmup_frac`` of the updates (0.1
#: unless authored) and decays linearly to 0 at the last, the schedule pyvene's
#: sigmoid-mask recipe (DBM) trains under. ``warmup_frac`` is legal only with it,
#: and enters the canonical form only when authored.
OPTIMIZER_SCHEDULES: tuple[str, ...] = ("constant", "linear_warmup_decay")

OPTIMIZER_DEFAULTS: dict[str, dict[str, Any]] = {
    # torch.optim.AdamW defaults (torch 2.x): betas=(0.9, 0.999), eps=1e-8,
    # weight_decay=1e-2: but the protocol default is 0.0: a regularizer is an
    # objective term here (§2.11), never an optimizer side-effect.
    "adamw": {
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "weight_decay": 0.0,
        "schedule": "constant",
    },
    "adam": {
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "weight_decay": 0.0,
        "schedule": "constant",
    },
    "sgd": {"momentum": 0.0, "weight_decay": 0.0, "schedule": "constant"},
}


# --------------------------------------------------------------------------- #
# value wrappers
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class Sweep:
    """An explicit sweep axis (§3): ``{"sweep": [v1, v2]}`` or
    ``{"sweep": {"range": [start, stop, step?]}}``. ``values`` holds the
    expanded value list either way (a range is expanded eagerly: it is sugar
    for the list it denotes)."""

    values: tuple[Any, ...]


@dataclasses.dataclass(frozen=True)
class ArtifactRef:
    """An artifact-valued field (§1): one value read from a prior run's
    artifact at load. Unresolved in the authored object model; resolution
    (and the missing-artifact load error, §5.15) is
    [`causalab.io.sources`][]'s job."""

    artifact: str
    key: str


#: A leaf that may still be swept or artifact-valued in the authored model.
Leaf = Union[Any, Sweep, ArtifactRef]


def concrete_int(value: Leaf, what: str) -> int:
    """Narrow a leaf that must be concrete by now (a point document) to int."""
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    raise ParseError("P2", f"{what} is not a concrete integer (got {value!r})")


def concrete_str(value: Leaf, what: str) -> str:
    """Narrow a leaf that must be concrete by now (a point document) to str."""
    if isinstance(value, str):
        return value
    raise ParseError("P2", f"{what} is not a concrete string (got {value!r})")


# --------------------------------------------------------------------------- #
# section records
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class QuantizationSpec:
    """§2.1: how the weights are quantized at load.

    Quantization changes what the network computes, so it is document
    vocabulary, not an execution flag: two runs of the same protocol at
    ``nf4`` and at ``bf16`` are two experiments, and their digests say so.
    Every field that moves a number is here: the scheme, the quantizer, the
    dtype the dequantized matmuls run in, and the scheme's own knobs.
    """

    scheme: Leaf
    method: Leaf = "bitsandbytes"
    compute_dtype: Leaf | None = None
    double_quant: Leaf | None = None
    int8_threshold: Leaf | None = None


@dataclasses.dataclass(frozen=True)
class ModelRef:
    """§2.1: the network as a name, plus how it is realized numerically.

    ``revision`` defaults to ``"main"`` and ``dtype`` to
    [`MODEL_DTYPE_DEFAULT`][] (both materialized by canonicalization when
    unauthored, §7).
    """

    key: Leaf
    revision: Leaf = "main"
    dtype: Leaf | None = None
    quantization: QuantizationSpec | None = None
    attn_implementation: Leaf | None = None


@dataclasses.dataclass(frozen=True)
class DataRole:
    """§2.2: one input-row column: a dataset ref (local path or HF key, no
    digest: the content digest is stamped at load) plus the column selector
    ``field`` (``[j]`` indexes list-valued columns). A role authored as
    ``inputs: [...]`` arrives here already resolved: ``dataset`` is the
    derived ``inline:<digest>`` ref (`causalab.tables.inline_ref`) and
    ``field`` is `causalab.tables.INPUT_COLUMN`, so no consumer of a
    role tells the two spellings apart.

    ``shuffle`` is the first data verb: ``{"seed": <int>}`` on a **counterfactual**
    role permutes that role's rows by ``random.Random(seed)`` over their indices
    before rows are paired, so the same rows meet different base rows: the
    ``shuffled_source`` control (workflow spec §2.2). The base role is the
    population and is never permuted (refused at parse). ``None`` when
    unauthored, and in the canonical form exactly when authored (§7), so no
    unshuffled document's digest moves.

    ``draw`` is the second data verb, on a **counterfactual** role whose
    ``field`` names a list-valued column bare (no ``[j]``): ``{"kind":
    "uniform", "eval"?: j}``. A fit redraws one member per row from its own
    seeded generator at every epoch (each row is visited once per epoch, so
    that is once per update it takes part in); every other forward: the
    point's reads, ``train.eval``, an apply document: reads the fixed member
    ``eval`` (``0`` unless authored). ``None`` when unauthored, canonical only
    when authored."""

    dataset: Leaf
    field: Leaf
    shuffle: Mapping[str, Any] | None = None
    draw: Mapping[str, Any] | None = None

    @property
    def draw_column(self) -> str | None:
        """The bare list column a drawn role reads, or ``None``."""
        return str(self.field) if self.draw is not None else None

    @property
    def eval_member(self) -> int:
        """The member every non-training forward of a drawn role reads."""
        if self.draw is None:
            return 0
        return int(self.draw.get("eval", 0))

    @property
    def resolved_field(self) -> str:
        """The field a forward tokenizes out of this role's rows: the authored
        field, or ``<column>[eval]`` for a drawn role (§2.2). The one spelling
        every mirror of "the field this role reads" asks: ``resolve_roles``,
        the fit's own minibatch executors, the loader's prompt-variable check
        and the data identity: so a drawn role cannot be one thing to the
        engine and another to a checker."""
        if self.draw is None:
            return str(self.field)
        return f"{self.draw_column}[{self.eval_member}]"


@dataclasses.dataclass(frozen=True)
class SiteSpec:
    """§2.4: a named activation address: pure data, no behavior.

    ``layers`` is the **band** the site spans: a non-empty, strictly
    increasing tuple of layer indices, ``(18,)`` for the ordinary one-layer
    site. A band is one site: one read, one write, one operand: across
    every layer it names (ROME's clipped restoration window is the shape);
    the engines fan it out to one module per member
    (`causalab.protocol.lowering.lower_bands`). Distinct from ``at_once``
    (§3.1), which declares N one-layer sites in one point. ``None`` on the
    layer-less trunk components ([`LAYERLESS_COMPONENTS`][]).
    """

    component: Leaf
    layers: Leaf | None = None
    head: Leaf | None = None
    expert: Leaf | None = None
    stream: Leaf | None = None


@dataclasses.dataclass(frozen=True)
class ParamSpec:
    """§2.6: a free tensor owned by no featurizer: either a loaded constant
    (``file_path``, optionally narrowed to one bundle entry by ``entry``) or
    a trainable free tensor (``shape`` + ``init``, which must then appear in
    ``train.params``)."""

    file_path: Leaf | None = None
    entry: Any = None
    shape: Leaf | None = None
    init: Leaf | None = None
    description: str | None = None


@dataclasses.dataclass(frozen=True)
class RowRole:
    """§2.8.1: one named block of rows in the batch a referenced function
    receives, in batch order. ``rows`` is how many; the roles' order is the
    physical order, so ``[clean:1, corrupted:10]`` *is* ROME's eleven-row
    convention, written down."""

    role: str
    rows: int


@dataclasses.dataclass(frozen=True)
class CodeSpec:
    """§2.8.1: a user function, declared rather than merely named.

    ``locator`` is the importable dotted path. Everything else is what a bare
    qualname left outside the digest: the arguments it is called with, the
    files it may read (content-digested at load), the environment variables
    it is allowed to read, and what the rows of its batch are. The source
    hash is derived, never authored (§6).
    """

    locator: Leaf
    args: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    data_inputs: Mapping[str, str] = dataclasses.field(default_factory=dict)
    env_inputs: tuple[str, ...] = ()
    row_roles: tuple[RowRole, ...] = ()
    description: str | None = None

    @property
    def declared_rows(self) -> int | None:
        """How many rows the declaration says the batch has, or ``None`` when
        it says nothing about rows."""
        return sum(r.rows for r in self.row_roles) if self.row_roles else None


@dataclasses.dataclass(frozen=True)
class ReadSpec:
    """§2.7: a value producer — an **address**, bound to no model. ``pos`` is
    a positions-table name or an inline [`PositionSpec`][] (int sugar
    already expanded). ``featurizer`` is a single name or a left-to-right
    composition tuple. The models that take the read list it (§2.9); one
    read listed by two models is two values ([`ReadRef`][])."""

    site: Leaf
    pos: Leaf
    featurizer: Leaf | None = None
    dims: Leaf | None = None


@dataclasses.dataclass(frozen=True)
class ReadRef:
    """One read **as taken on one model**: the unit the runtime keys every
    read value by (§2.7, §2.9).

    A read declares an address; the model that lists it decides which
    forward the address is gathered from, so the same read on two models is
    two tensors. ``model`` is ``None`` only for a reference authored bare
    (``"swap": "v_cf"``) that resolution has not yet bound; after
    ``parse_document`` a ``None`` survives only on an ambiguous or dangling
    reference, which validation refuses (§5 rule 5)."""

    read: str
    model: str | None = None

    def __str__(self) -> str:
        return self.read if self.model is None else f"{self.model}/{self.read}"


@dataclasses.dataclass(frozen=True)
class Do:
    """§2.8: one mechanism from the closed set. ``payload`` holds the
    mechanism's single value exactly as authored (operand name / literal
    scalar for ``swap``; the option mapping for the structured mechanisms;
    ``True`` for ``renormalize``)."""

    mechanism: Leaf
    payload: Any


@dataclasses.dataclass(frozen=True)
class WriteSpec:
    """§2.8: an inert effect definition: no model, no input, no conditions;
    it executes inside every intervened model that lists it."""

    site: Leaf
    pos: Leaf
    do: Do
    featurizer: Leaf | None = None
    dims: Leaf | None = None
    #: The ragged-window policy, one of [`RAGGED_POLICIES`][], or ``None``
    #: when the document authors none: which the executor reads as
    #: ``refuse`` (§5 rule 19). Never swept: how a window lands is an
    #: execution strategy, not a research variable.
    ragged: str | None = None


#: §2.9: the intervened model's one optional flag. ``true`` keeps every write
#: of the model in force through the decode steps of a continuation (§2.3), at
#: the token each step processes; absent or ``false`` is the prefill-only
#: behaviour every document had before the flag existed.
WRITES_DURING_GENERATION_FIELD: str = "writes_during_generation"


@dataclasses.dataclass(frozen=True)
class IMSpec:
    """§2.9: a model ℒ_{b∪𝕀}: a mandatory input role, the reads taken on
    it (mandatory, non-empty: a model nobody reads runs a forward nobody
    observes) and the writes in force (unordered; canonical form sorts). A
    model with no writes is the un-intervened model on its input.

    ``writes_during_generation`` is all-or-nothing for the model: ``True``
    fires every listed write at every decode step as well as in the prefill
    (rule 16 holds the writes to the forms that mean "the token being
    processed"); ``False`` — the default, and the canonical form's absent
    field — is prefill-only."""

    input: Leaf
    reads: tuple[str, ...]
    writes: tuple[str, ...] | Sweep | ArtifactRef = ()
    writes_during_generation: bool = False

    @property
    def write_names(self) -> tuple[str, ...] | None:
        """The writes in force, or ``None`` while the list is still under a
        sweep wrapper — a caller that needs a point document refuses."""
        return tuple(self.writes) if isinstance(self.writes, tuple) else None

    @property
    def is_unwritten(self) -> bool:
        """Whether the model lands no write: an empty, *known* write list.
        A wrapped list is not known to be unwritten and answers False."""
        return isinstance(self.writes, tuple) and not self.writes


@dataclasses.dataclass(frozen=True)
class AggregationSpec:
    """§2.10: a closed-vocabulary reduction over one read plus dataset
    columns, **without** the read it reduces — that binding is the consumer's
    ([`BoundAggregation`][]). ``fields`` holds the kind's extra value
    fields; for the two-read kinds ([`READ_TARGET_METRIC_KINDS`][])
    ``fields["target"]`` holds a [`ReadRef`][]. ``top_k`` additionally
    carries a mandatory ``by`` (``TOP_K_RANKINGS``) in ``fields``, because a
    top-k over a signed feature code and one over a vocabulary projection are
    different questions.

    ``token_form`` is ``None`` when the aggregation's answers are strings
    tokenized as written (the default, and absent from the canonical form)
    and ``"id"`` when its columns hold integer vocabulary ids
    (``TOKEN_FORMS``). Only the kinds in ``TOKEN_COLUMN_METRIC_KINDS`` accept
    the key.

    ``unit`` and ``estimand_version`` are the record's identity
    (``causalab/protocol/estimand.py``): optional, in the canonical form
    **only when authored**, and: because every kind is one arithmetic in
    one unit: checked at parse to be the kind's own (``METRIC_UNITS``,
    ``<kind>/v1``). A document may state what its aggregation is; it may not
    declare it to be something else. Unauthored, both are derived onto every
    row the aggregation writes (``neural/shared/results.py``) and never into
    the digest.

    ``minimum_count`` is the aggregation's **decision threshold** (§2.10
    "Eligibility"): the fewest eligible rows its decision rule needs.
    Optional, never sweepable, in the canonical form only when authored;
    ``validate --data`` refuses one above the resolved base table's maximum
    eligible count (rule 4), and the cell's derived ``n_eligible`` is what a
    consumer holds it against. The count itself is derived at run time
    (``neural/shared/results.py``), never authored: this field is only the bar."""

    kind: Leaf
    fields: Mapping[str, Leaf]
    token_form: str | None = None
    unit: str | None = None
    estimand_version: str | None = None
    minimum_count: int | None = None


@dataclasses.dataclass(frozen=True)
class BoundAggregation:
    """One aggregation **where it is consumed**: a reduction ``spec`` over
    the bound read ``read``, owned by the document path ``owner``
    (``save[2]``, ``train.objective.ce``, ``train.eval.aggregations.iia``)
    and going by ``label`` in every record it writes — the ``metric`` column
    of its table, its key in the point summary, the name on its ``metric``
    event."""

    owner: str
    label: str
    read: ReadRef
    spec: AggregationSpec

    @property
    def target(self) -> ReadRef | None:
        """The second read of a ``kl`` / ``js`` aggregation, else ``None``."""
        target = self.spec.fields.get("target")
        return target if isinstance(target, ReadRef) else None

    @property
    def kind(self) -> Leaf:
        return self.spec.kind


@dataclasses.dataclass(frozen=True)
class ConstraintSpec:
    """§2.11 ``constraint`` on a named mask-density regularizer: Edge
    Pruning's Lagrangian target. The term's value ``s`` (the mask mean under
    ``l1``, the expected kept fraction under ``l0``) is held to ``target`` by
    ``λ₁·(s − t) + λ₂·(s − t)²`` added to the loss, with the dual pair
    ``(λ₁, λ₂)`` **ascended**: stepped against its gradient: at ``dual_lr``
    from ``dual_init`` (``(0, 0)`` when unauthored, and then absent from the
    canonical form). The term has no ``weight``: the duals are its weight.
    ``λ₂`` is non-negative: the parser refuses a negative one, and the ascent
    never makes one (its gradient ``(s − t)²`` is non-negative): a
    constructor that bypasses the parser owes the same bound, since
    ``_add_dual_groups`` reads ``init`` into a tensor without re-checking."""

    target: float
    dual_lr: float
    dual_init: tuple[float, float] | None = None

    @property
    def init(self) -> tuple[float, float]:
        return self.dual_init if self.dual_init is not None else (0.0, 0.0)


@dataclasses.dataclass(frozen=True)
class ObjectiveTerm:
    """One weighted term of ``train.objective`` (§2.11): a differentiable
    aggregation over a bound read (``read`` + ``aggregation``), or a
    regularizer ``(kind, names)``: ``kind`` is one of
    [`REGULARIZER_KINDS`][] (``l1``, ``l2``, ``l0``) and ``names`` the
    featurizers (or the one dotted slot) it penalizes together. ``name`` is the
    term's key in the named mapping form and ``None`` in the positional list
    form; it is what the term's ``weight`` is addressed by when swept
    (``train.objective.<name>.weight``). A ``constraint`` term (named form
    only) has no weight: ``None``: and carries its [`ConstraintSpec`][]
    instead."""

    weight: Leaf | None
    read: ReadRef | None = None
    aggregation: AggregationSpec | None = None
    regularizer: tuple[str, tuple[str, ...]] | None = None
    name: str | None = None
    #: a regularizer's reduction over the concatenated per-unit quantities
    #: ([`REGULARIZER_REDUCTIONS`][causalab.protocol.schema.parse.REGULARIZER_REDUCTIONS]): ``mean`` when unauthored: a kept
    #: unit then costs ``weight / units``: or ``sum``, where it costs
    #: ``weight`` whatever the unit count (NeuroSurgeon's ``λ · Σ``, with λ
    #: "scaled with parameter count" by the author instead of by the mean).
    #: Kept ``None`` when unauthored so no canonical form materializes it.
    reduce: str | None = None
    #: §2.11 ``costs``: a per-target multiplier on the penalized quantities
    #: before they are concatenated: ``{target: c}`` (an unlisted target
    #: costs 1), or a word from [`REGULARIZER_COSTS`][]:
    #: ``"parameter_count"`` divides each target's quantities by its own
    #: element count, NeuroSurgeon's λ scaled with the parameter count, so
    #: under ``reduce: sum`` the term is the sum of per-featurizer means.
    #: ``None`` when unauthored.
    costs: Mapping[str, float] | str | None = None
    #: §2.11 ``constraint``: a Lagrangian target density on a named ``l1`` /
    #: ``l0`` term ([`ConstraintSpec`][]); ``None`` for every other term.
    constraint: ConstraintSpec | None = None

    def path(self, index: int) -> str:
        """The term's address in error messages: its name, or its list index."""
        if self.name is not None:
            return f"train.objective.{self.name}"
        return f"train.objective[{index}]"


@dataclasses.dataclass(frozen=True)
class TrainSpec:
    """§2.11: the fit, declared. ``objective`` is the weighted terms, in
    authored order, whichever form spelled them."""

    objective: tuple[ObjectiveTerm, ...]
    params: tuple[str, ...]
    optimizer: Mapping[str, Leaf]
    steps: Mapping[str, Leaf]
    batch: Mapping[str, Leaf]
    #: Open-loop schedules keyed by ``<name>.<slot>.<hyperparameter>`` or a
    #: named objective term's ``train.objective.<name>.weight`` (§2.11).
    #:
    anneal: Mapping[str, AnnealSchedule] | None = None
    #: Closed-loop schedules: ``{<target>: {kind, signal, setpoint, gains,
    #: …}}`` (§2.11). A target is a named objective weight or an anneal-
    #: style hyperparameter path. Its declared value initializes the
    #: controller.
    #:
    control: Mapping[str, Mapping[str, Any]] | None = None
    #: Consecutive training windows (§2.11). Each selects trainable
    #: parameters and annealing schedules. Defaults to one phase.
    #:
    phases: tuple[PhaseSpec, ...] | None = None
    precision: Mapping[str, Leaf] | None = None
    #: ``{every, split, aggregations}`` (§2.11): ``aggregations`` maps each
    #: eval label to its ``(ReadRef, AggregationSpec)`` pair.
    eval: Mapping[str, Any] | None = None
    #: ``{on, patience, mode}``: ``on`` names an eval label.
    early_stop: Mapping[str, Leaf] | None = None
    checkpoint: Mapping[str, Leaf] | None = None
    seed: Leaf = 0


@dataclasses.dataclass(frozen=True)
class SaveEntry:
    """§2.12: one manifest entry, in one of three shapes. A **read** entry
    names a bound read (``read``: a [`ReadRef`][] with its model) and
    saves its rows as a tensor, or — with ``aggregation`` — the reduction
    over it as a table; ``reduce`` (tensor entries only) saves a statistic
    over the gathered rows instead of the rows themselves. A **featurizer**
    entry names a trained featurizer (``value``) and the ``site`` it is used
    at, cross-checked at validation. A **derived-record** entry names a
    ``kind`` ([`SAVE_KINDS`][])."""

    file_path: str
    read: ReadRef | None = None
    aggregation: AggregationSpec | None = None
    reduce: str | None = None
    value: str | None = None
    site: str | None = None
    #: A non-value entry kind ([`SAVE_KINDS`][], §2.12): ``location_ledger``
    #: saves the run's resolved token indices, ``trajectory`` the trained
    #: featurizers at checkpoints along the fit. ``value`` then repeats the
    #: kind, so the one-entry-per-value rule holds for it too.
    kind: str | None = None
    #: ``trajectory`` only: how the checkpoints are spaced :
    #: ``{"count": n}`` | ``{"updates": n}`` | ``{"epochs": n}``
    #: ([`TRAJECTORY_EVERY_UNITS`][]).
    every: Mapping[str, int] | None = None

    @property
    def label(self) -> str:
        """What the entry's output goes by: the file stem for a read entry —
        an aggregation's ``metric`` column, summary key and ``metric`` event,
        a tensor's cell key — the featurizer name, or the kind. Two entries
        never share a file, so the stem is the one label a reader can key by
        when one read is saved on two models."""
        if self.read is not None:
            stem = self.file_path.rsplit("/", 1)[-1]
            return stem.rsplit(".", 1)[0] if "." in stem else stem
        return str(self.value if self.value is not None else self.kind)


@dataclasses.dataclass(frozen=True)
class Document:
    """One parsed intervention specification (authored form, sugar
    expanded, wrappers preserved). The four groups of the file (§1) flatten
    to attributes here: a consumer reads ``document.sites``, never
    ``document.method.sites``: while ``raw`` keeps the mapping the parser
    consumed, grouped as authored: the substrate sweep expansion and
    canonicalization operate on."""

    protocol_version: str
    model: ModelRef
    data: Mapping[str, DataRole | tuple[DataRole, ...]]
    sites: Mapping[str, SiteSpec]
    reads: Mapping[str, ReadSpec]
    intervened_models: Mapping[str, IMSpec]
    save: tuple[SaveEntry, ...]
    title: str | None = None
    description: str | None = None
    positions: Mapping[str, PositionSpec | Sweep | ArtifactRef] = dataclasses.field(
        default_factory=dict
    )
    featurizers: Mapping[str, FeaturizerSpec] = dataclasses.field(default_factory=dict)
    params: Mapping[str, ParamSpec] = dataclasses.field(default_factory=dict)
    code: Mapping[str, CodeSpec] = dataclasses.field(default_factory=dict)
    writes: Mapping[str, WriteSpec] = dataclasses.field(default_factory=dict)
    train: TrainSpec | None = None
    #: §2.2.1: the row's named segments and their frame (``protocol/
    #: segments.py``); ``None`` is plain text, which is every document that
    #: does not author the section.
    segments: "SegmentsSpec | None" = None
    raw: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def named_entries(self) -> dict[str, str]:
        """Every declared name → the section that declares it. Duplicates are
        a validation concern (§5.3); the parser reports the first section."""
        seen: dict[str, str] = {}
        for section in NAMED_SECTIONS:
            table: Mapping[str, Any] = getattr(self, section)
            for name in table:
                seen.setdefault(name, section)
        return seen

    # ------------------------------------------------------------------ #
    # the reads-first surface the runtime consumes (§2.7, §2.9, §2.10)
    # ------------------------------------------------------------------ #

    def models_of(self, read: str) -> tuple[str, ...]:
        """The models that list ``read`` (§2.9), in model declaration order
        — empty for a name that is no read or that no model takes."""
        if read not in self.reads:
            return ()
        return tuple(
            name for name, im in self.intervened_models.items() if read in im.reads
        )

    def bound(self, read: str) -> ReadRef:
        """``read`` bound to the one model that lists it — the bare-string
        sugar (§2.7). ``model`` is ``None`` when the read is listed by no
        model or by several, which validation refuses (rule 5)."""
        models = self.models_of(read)
        return ReadRef(read, models[0] if len(models) == 1 else None)

    def read_refs(self) -> tuple[ReadRef, ...]:
        """Every read **as taken on a model**: one [`ReadRef`][] per
        ``(read, model)`` pair, reads in declaration order, then the models
        that list each. The runtime's unit of value."""
        return tuple(
            ReadRef(name, model)
            for name in self.reads
            for model in self.models_of(name)
        )

    def is_unwritten(self, model: str) -> bool:
        """Whether ``model`` lands no write (§2.9): a declared model whose
        write list is empty. A write list still under a sweep wrapper is not
        known to be unwritten and answers False; a name no model carries is
        unwritten by construction (nothing lands a write in it)."""
        im = self.intervened_models.get(model)
        return True if im is None else im.is_unwritten

    def input_of(self, model: str) -> str:
        """The input role ``model`` runs on (§2.9)."""
        return str(self.intervened_models[model].input)

    def group_of(self, ref: ReadRef) -> tuple[str, str]:
        """The ``(model, input role)`` forward group a bound read is gathered
        from (§4)."""
        if ref.model is None:
            raise KeyError(f"unbound read reference {ref.read!r}")
        return ref.model, self.input_of(ref.model)

    def aggregations(self) -> tuple[BoundAggregation, ...]:
        """Every aggregation the document consumes, with its owner: the
        ``save`` entries in order, then the objective terms, then the eval
        entries (§2.10, §2.11, §2.12)."""
        out: list[BoundAggregation] = []
        for i, entry in enumerate(self.save):
            if entry.aggregation is not None and entry.read is not None:
                out.append(
                    BoundAggregation(
                        owner=f"save[{i}]",
                        label=entry.label,
                        read=entry.read,
                        spec=entry.aggregation,
                    )
                )
        train = self.train
        if train is not None:
            for i, term in enumerate(train.objective):
                if term.aggregation is not None and term.read is not None:
                    out.append(
                        BoundAggregation(
                            owner=term.path(i),
                            label=term.name
                            if term.name is not None
                            else f"objective[{i}]",
                            read=term.read,
                            spec=term.aggregation,
                        )
                    )
            if train.eval is not None:
                for label, (read, spec) in train.eval.get("aggregations", {}).items():
                    out.append(
                        BoundAggregation(
                            owner=f"train.eval.aggregations.{label}",
                            label=label,
                            read=read,
                            spec=spec,
                        )
                    )
        return tuple(out)

    def saved_aggregations(self) -> tuple[BoundAggregation, ...]:
        """The aggregations a ``save`` entry writes as a table, in entry order."""
        return tuple(a for a in self.aggregations() if a.owner.startswith("save["))

    def objective_aggregations(self) -> tuple[BoundAggregation, ...]:
        """The aggregations the fit's objective terms reduce, in term order."""
        return tuple(
            a for a in self.aggregations() if a.owner.startswith("train.objective")
        )

    def eval_aggregations(self) -> tuple[BoundAggregation, ...]:
        """The aggregations the fit's eval pass scores, in authored order."""
        return tuple(
            a
            for a in self.aggregations()
            if a.owner.startswith("train.eval.aggregations.")
        )

    def aggregation_at(self, owner: str) -> BoundAggregation | None:
        """The aggregation owned by document path ``owner``, else ``None``."""
        for agg in self.aggregations():
            if agg.owner == owner:
                return agg
        return None

    def early_stop_label(self) -> str | None:
        """The eval label ``train.early_stop`` watches, else ``None``."""
        train = self.train
        if train is None or train.early_stop is None:
            return None
        return str(train.early_stop["on"])

    def saved_raw_reads(self) -> frozenset[ReadRef]:
        """The bound reads a ``save`` entry writes **as tensors** (§2.12)."""
        return frozenset(
            entry.read
            for entry in self.save
            if entry.read is not None and entry.aggregation is None
        )


def do_operand_slots(do: Do) -> dict[str, Any]:
    """The operand slots of one mechanism payload (§2.8): ``{"": payload}``
    for a bare operand (``swap``), one slot per key for a structured one,
    nothing for a flag (``renormalize``)."""
    payload = do.payload
    if isinstance(payload, Mapping):
        return dict(payload)
    if payload is True or payload is None:
        return {}
    return {"": payload}


def operand_reads(doc: Document, do: Do) -> tuple[ReadRef, ...]:
    """The read operands of one mechanism payload, bound to the model each
    is taken on (§2.8): an authored [`ReadRef`][], or a bare name that
    names a read — bound through [`Document.bound`][causalab.protocol.schema.types.Document.bound]. Params and literals
    are left out ([`operand_params`][])."""
    out: list[ReadRef] = []
    for value in do_operand_slots(do).values():
        if isinstance(value, ReadRef):
            out.append(value)
        elif isinstance(value, str) and value in doc.reads:
            out.append(doc.bound(value))
    return tuple(out)


def operand_params(doc: Document, do: Do) -> tuple[str, ...]:
    """The operand names that are **not** reads: ``params`` entries and
    ``<featurizer>.<slot>`` addresses (§2.8)."""
    return tuple(
        value
        for value in do_operand_slots(do).values()
        if isinstance(value, str) and value not in doc.reads
    )


def read_is_vocabulary(doc: Document, read: str) -> bool:
    """Whether the last axis of ``read``'s value is the vocabulary: it taps
    ``lm_head`` **and hands the projection on unchanged** — no ``featurizer``
    re-expressing it and no ``dims`` re-indexing it (§2.10). A dangling name
    is validation's error to report, so it answers False here."""
    spec = doc.reads.get(read)
    if spec is None:
        return False
    if spec.featurizer is not None or spec.dims is not None:
        return False
    site = doc.sites.get(str(spec.site))
    return site is not None and site.component == "lm_head"


def metric_column_fields(metric: AggregationSpec) -> dict[str, str]:
    """The value fields of ``metric`` that name **dataset columns**, as
    ``field → column`` (§2.10).

    One predicate for three callers: ``validate --data``'s column check and
    its maximum-eligible count (``rules/data.py``) and the run-time eligibility
    predicate (``answers.excluded_rows``): so a row the loader counts as
    eligible is a row the run scores, and a field the loader skips is a field
    the run never looks up. Skipped: the non-column fields (``k``, ``by``,
    ``tokens``), ``groups`` (a mapping of literals), a ``target`` that is a
    read ([`READ_TARGET_METRIC_KINDS`][]), the enum-valued optional fields
    (``match.mode``), and any field whose value is not a string: a literal
    ``restrict`` list, or a swept field the caller resolves per point. A
    string-valued ``restrict`` **is** a column: its per-row value is the
    answer list the comparison is restricted to.
    """
    kind = str(metric.kind)
    optional = OPTIONAL_METRIC_FIELDS.get(kind, ())
    out: dict[str, str] = {}
    for field, value in metric.fields.items():
        if field in NON_COLUMN_METRIC_FIELDS or field == "groups":
            continue
        if field == "target" and kind in READ_TARGET_METRIC_KINDS:
            continue
        if field in optional and field != "restrict":
            continue
        if not isinstance(value, str):
            continue
        out[field] = value
    return out


def answer_columns(kind: str, raw: Mapping[str, Any]) -> dict[str, str]:
    """The fields of an authored metric or aggregation ``raw`` of ``kind``
    whose answers are dataset columns, each with the columns it names,
    spelled for a refusal (§2.10).

    [`metric_column_fields`][] decides which fields are columns, the
    predicate the loader and the run use. Each present field is passed its
    own name as its value, so a swept field is judged by the field it sits
    on and not by its wrapper. A sweep is spelled as its column names. A
    ``restrict`` keeps its value, because a literal list there is not a
    column. Any other value, such as an artifact reference, is shown as
    JSON.

    Args:
        kind: The metric kind. An unknown kind has no column fields.
        raw: The authored metric (protocol 3) or aggregation (protocol 4).

    Returns:
        ``field -> spelled columns`` in the kind's field order: ``'io'`` for
        one column, ``'cf_answer', 'base_answer'`` for a sweep.
    """
    fields = METRIC_FIELDS.get(kind, ()) + OPTIONAL_METRIC_FIELDS.get(kind, ())
    present = {
        field: raw[field] if field == "restrict" else field
        for field in fields
        if field in raw
    }
    columns = metric_column_fields(AggregationSpec(kind=kind, fields=present))
    out: dict[str, str] = {}
    for field in columns:
        value = raw[field]
        swept = value.get("sweep") if isinstance(value, Mapping) else None
        if isinstance(value, str):
            out[field] = repr(value)
        elif isinstance(swept, list) and all(isinstance(v, str) for v in swept):
            out[field] = ", ".join(repr(v) for v in swept)
        else:
            out[field] = json.dumps(value, sort_keys=True)
    return out
