"""The featurizer vocabulary (spec §2.5) and [`FeaturizerSpec`][].

The closed kinds and their families, the gate's parametrizations and maps
([`GATE_MAPS`][]), the per-field legality tables the validator and the
method pages read ([`FEATURIZER_FIELD_CONDITIONS`][]), the hard-concrete
constants, the control / anneal / phase records of a fit, and the
``render_*`` functions that write the spec's and the method pages' tables from
these rows. Imports [`causalab.protocol.schema.types`][] only.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Literal, Mapping, get_args

from causalab.protocol.rules.errors import RULES
from causalab.protocol.schema.types import (
    Leaf,
)


FeaturizerKind = Literal["identity", "subspace", "pca", "sae", "standardize", "gate"]
FEATURIZER_KINDS: tuple[FeaturizerKind, ...] = get_args(FeaturizerKind)

#: Auto-declared param slots per featurizer kind (§2.5): ``<name>.<slot>``.
FEATURIZER_SLOTS: dict[str, tuple[str, ...]] = {
    "identity": (),
    "subspace": ("weight",),
    "pca": ("weight",),
    "sae": ("enc", "dec", "b_enc", "b_dec"),
    "standardize": ("mu", "sigma"),
    "gate": ("theta",),
}

#: Authorable choice fields per featurizer kind (§2.5): everything else about
#: a featurizer (width, param shapes, slots) is derived and may not be
#: authored (§6). ``file_path`` (load a fitted artifact) and ``dtype`` are
#: legal on every kind; ``description`` is legal everywhere.
FEATURIZER_FIELDS: dict[str, frozenset[str]] = {
    "identity": frozenset(),
    "subspace": frozenset({"k", "parametrization", "init", "seed"}),
    "pca": frozenset({"k"}),
    "sae": frozenset(),
    "standardize": frozenset(),
    "gate": frozenset(
        {
            "group",
            "parametrization",
            "init",
            "temperature",
            "stretch",
            "top_k",
            "k_schedule",
            "stop_grad_shift",
            "pool",
            "dead",
            "axis",
        }
    ),
}

#: The kinds with gradient-trainable slots (§5.12): the ones ``train.params``
#: may name. ``pca`` and ``standardize`` are computed from data and loaded,
#: ``identity`` has nothing to fit.
TRAINABLE_KINDS: frozenset[str] = frozenset({"subspace", "gate", "sae"})


@dataclasses.dataclass(frozen=True)
class FeaturizerFamily:
    "Description and feature-map formula for one featurizer kind. Method pages and reference tables render these fields."

    sentence: str
    featurize: str


#: One entry per [`FEATURIZER_KINDS`][] (the census holds the two to one
#: set): what each kind does in one line, and the ``featurize`` cell of the
#: §2.5 kinds table ([`render_featurizer_kind_table`][]).
FEATURIZER_FAMILIES: dict[str, FeaturizerFamily] = {
    "identity": FeaturizerFamily(
        "The site's coordinates, with ``f = x``. This is the default map.",
        "`(x, 0)`",
    ),
    "subspace": FeaturizerFamily(
        "An orthonormal basis whose first ``k`` columns define the subspace "
        "for an interchange intervention. DAS learns this basis.",
        "`(Qᵀx, 0)`",
    ),
    "pca": FeaturizerFamily(
        "A PCA basis fitted to saved activations and loaded from a bundle. "
        "Its first ``k`` components define the intervention subspace.",
        "`(Pᵀx, 0)`",
    ),
    "sae": FeaturizerFamily(
        "An encoder/decoder pair that maps activations to latent features. "
        "The saved reconstruction error ``err`` makes inversion exact.",
        "`(enc(x), x − dec(enc(x)))`",
    ),
    "standardize": FeaturizerFamily(
        "A loaded per-coordinate ``(μ, σ)``: the site in z-scored units, so a "
        "write in feature space is a write in standard deviations.",
        "`((x−μ)/σ, 0)`",
    ),
    "gate": FeaturizerFamily(
        "A mask learned with supervision that specifies desiderata. An "
        "interchange uses counterfactual activations at selected units "
        "and preserves the original activations elsewhere.",
        "`(m⊙x, (1−m)⊙x)`, `m` the soft mask in training and the hard mask "
        "at eval, by `parametrization` (the table below)",
    ),
}

#: Maps from stored parameters to an orthonormal ``(d, k)`` basis ``Q``
#: (§2.5). ``cayley`` uses the Cayley transform from the initial basis, with
#: ``O(d k²)`` work per access. ``matrix_exp`` uses a skew-symmetric matrix
#: exponential; ``stiefel`` uses Householder reflections through torch's
#: ``orthogonal`` maps. A loaded document must match the bundle's
#: parametrization (rule 15).
#:
PARAMETRIZATIONS: tuple[str, ...] = ("cayley", "matrix_exp", "stiefel")

#: Rules for units whose training mask becomes hard-off (§2.5). Choose one:
#: ``freeze_after: n`` fixes a unit after ``n`` consecutive hard-off
#: optimizer steps; ``leak: ε`` adds ``ε`` to ``∂m/∂θ`` so saturated units
#: can receive gradients. The leak preserves the forward value and
#: evaluation mask. An omitted rule is absent from the canonical form.
#:
GATE_DEAD_RULES: tuple[str, ...] = ("freeze_after", "leak")


@dataclasses.dataclass(frozen=True)
class GateMap:
    "A gate parametrization and its documentation (§2.5). The formulas describe soft and hard masks, projection, and regularization. The validator reads penalty and annealing support. Ranked maps need a training budget and a loaded top-k cut. Indexed maps use a scalar boundary over an ordered basis and exclude per-unit options."

    name: str
    soft_mask: str
    post_step: str
    hard_mask: str
    penalty: str | None
    penalty_formula: str
    anneals_temperature: bool
    default_start: str
    no_temperature_because: str = ""
    ranked: bool = False
    indexed: bool = False

    def __post_init__(self) -> None:
        if self.anneals_temperature == bool(self.no_temperature_because):
            raise ValueError(
                f"gate map {self.name!r}: a map that does not anneal its "
                "temperature says why, and one that does says nothing"
            )
        if (self.penalty is None) == bool(self.penalty_formula):
            raise ValueError(
                f"gate map {self.name!r}: a penalty comes with its formula"
            )
        if self.ranked and self.indexed:
            raise ValueError(
                f"gate map {self.name!r}: a ranked map cuts theta's units at a "
                "count and an indexed map has no units to rank: not both"
            )


#: Gate parametrizations map ``theta`` to a mask (§2.5). The table gives
#: their formulas and allowed training options. The default is ``sigmoid``;
#: its omitted spelling stays absent from the canonical form.
#:
#: ``hard_concrete`` uses the stochastic L0 relaxation of Louizos, Welling
#: and Kingma (2018, arXiv 1712.01312). Each optimizer step samples one mask
#: shared across the gate's reads and writes. Evaluation uses the
#: deterministic stretched sigmoid and a threshold of ½. Its ``l0`` penalty
#: is the expected fraction kept.
#:
#: ``budget`` learns a ranking with a fixed-sum mask ``σ(θ + c_k)``.
#: Bisection solves the shift ``c_k`` for the step's ``k_schedule`` budget.
#: Evaluation requires a cut: ``k_schedule.eval`` during fitting and
#: ``top_k`` after loading. ``stop_grad_shift`` treats the solved shift as
#: constant in the gradient.
#:
#: ``boundary`` learns a prefix of an ordered basis, as in Boundless DAS (Wu
#: et al. 2023, arXiv 2305.08809). Its single parameter ``θ ∈ [0, 1]`` gives
#: boundary ``β = θ · width`` and retained rank ``⌈β⌉``. It must follow a
#: ``subspace`` or ``pca`` stage in every chain. The parameter has shape
#: ``[1]``; per-unit options are invalid. ``init.fill p`` starts at ``θ =
#: p``; the default is ½. Projection keeps θ in bounds after each optimizer
#: step.
#:
GATE_MAPS: dict[str, GateMap] = {
    "sigmoid": GateMap(
        "sigmoid",
        soft_mask="`σ(θ / T)`",
        post_step="nothing",
        hard_mask="`θ > 0`",
        penalty="l1",
        penalty_formula="`mean σ(θ/T)`",
        anneals_temperature=True,
        default_start="`θ = 0`, i.e. `m = ½`",
    ),
    "clamp": GateMap(
        "clamp",
        soft_mask="`θ` itself",
        post_step="`θ ← clip(θ, 0, 1)`",
        hard_mask="`θ > ½` (`round`)",
        penalty="l1",
        penalty_formula="`mean θ`",
        anneals_temperature=False,
        no_temperature_because=(
            "a clamp gate uses θ directly, projected into [0, 1] after each step"
        ),
        default_start="`θ = ½`",
    ),
    "hard_concrete": GateMap(
        "hard_concrete",
        soft_mask=(
            "**sampled**, once per optimizer step: `u ~ U(0,1)`, "
            "`s = σ((log u − log(1−u) + θ)/β)`, then `clip(s·(ζ−γ)+γ, 0, 1)`"
        ),
        post_step="nothing",
        hard_mask=(
            "`clip(σ(θ)·(ζ−γ)+γ, 0, 1) > ½`, i.e. `θ > logit((½−γ)/(ζ−γ))`: "
            "exactly `θ > 0` at the default stretch"
        ),
        penalty="l0",
        penalty_formula=(
            "`mean σ(θ − β·log(−γ/ζ))`, the expected kept fraction of the sampled mask"
        ),
        anneals_temperature=True,
        default_start="`θ = 0`, i.e. `m = ½`",
    ),
    "budget": GateMap(
        "budget",
        soft_mask=(
            "`σ(θ + c_k)` with the step's budget `k` drawn from `k_schedule` "
            "and the scalar `c_k` solved so `Σ m = k` exactly"
        ),
        post_step="nothing",
        hard_mask=(
            "largest `θ` values: `k_schedule.eval` during fitting, "
            "`top_k` after loading"
        ),
        penalty=None,
        penalty_formula="",
        anneals_temperature=False,
        no_temperature_because=(
            "a budget gate fixes sharpness through its budget and solved shift"
        ),
        default_start="`θ = 0` (`fill` ½)",
        ranked=True,
    ),
    "boundary": GateMap(
        "boundary",
        soft_mask=(
            "`σ((β − i) / T)` over the coordinate index `i = 0 … width−1` of "
            "the stage's input: the rotation's columns or the PCA components; "
            "`θ ∈ [0, 1]` is the one scalar, the boundary as a fraction of the "
            "width, `β = θ · width`"
        ),
        post_step="`θ ← clip(θ, 0, 1)`",
        hard_mask=("`i < β`: the first `⌈β⌉` coordinates"),
        penalty="l1",
        penalty_formula=(
            "`mean σ((β − i)/T)`, the kept fraction (`⌈β⌉ / width` as `T → 0`)"
        ),
        anneals_temperature=True,
        default_start="`θ = ½`, the half prefix (`fill` ½)",
        indexed=True,
    ),
}

#: The maps whose ``theta`` is one entry per masked unit: every map but the
#: indexed ones: and so the maps a per-unit field (``group``, ``axis``,
#: ``dead``, ``top_k``, a pooled readout) is legal under.
_UNIT_MAPS: frozenset[str] = frozenset(
    name for name, gate_map in GATE_MAPS.items() if not gate_map.indexed
)

#: The map an unauthored ``parametrization`` means on a gate. It has no
#: spelling in the canonical form, so no existing document's digest moves.
GATE_DEFAULT_MAP: str = "sigmoid"

#: The closed vocabulary of gate maps, in [`GATE_MAPS`][]' order.
GATE_PARAMETRIZATIONS: tuple[str, ...] = tuple(GATE_MAPS)


@dataclasses.dataclass(frozen=True)
class FieldLegality:
    "Allowed maps for a field during fitting and after loading. None allows every map; an empty set forbids the field. Diagnostic strings explain restrictions. The parser and documentation renderer use this record."

    fit: frozenset[str] | None
    loaded: frozenset[str] | None
    why_fit: str = ""
    why_loaded: str = ""
    why_map: str = ""

    def __post_init__(self) -> None:
        for state, legal, why in (
            ("fit", self.fit, self.why_fit),
            ("loaded", self.loaded, self.why_loaded),
        ):
            if legal is not None and not legal <= frozenset(GATE_MAPS):
                raise ValueError(f"{state}: {sorted(legal)} are not gate maps")
            if (legal == frozenset()) != bool(why):
                raise ValueError(
                    f"{state}: a state the field is refused in says why, and "
                    "only such a state does"
                )
        restricted = any(
            legal is not None and legal and legal != frozenset(GATE_MAPS)
            for legal in (self.fit, self.loaded)
        )
        if restricted != bool(self.why_map):
            raise ValueError(
                "a field legal under some maps and not others says why, and "
                "only such a field does"
            )

    def legal(self, *, loaded: bool) -> frozenset[str] | None:
        """The maps the field is legal under in one state."""
        return self.loaded if loaded else self.fit


_HARD_CONCRETE: frozenset[str] = frozenset({"hard_concrete"})
_TEMPERED: frozenset[str] = frozenset({"hard_concrete", "boundary"})
_BUDGET: frozenset[str] = frozenset({"budget"})
_NEVER: frozenset[str] = frozenset()
_TEMPERED_ONLY = (
    "'temperature' requires hard_concrete or boundary; the selected map "
    "is {maps} (§2.5)"
)
_HARD_CONCRETE_ONLY = (
    "'stretch' requires hard_concrete; the selected map is {maps} (§2.5)"
)
_BUDGET_ONLY = (
    "'k_schedule' and 'stop_grad_shift' require budget; the selected map "
    "is {maps} (§2.5)"
)

#: §2.5's conditional legality, one row per authorable featurizer field
#: (every field of [`FEATURIZER_FIELDS`][]), in the order the spec's tables
#: and the method pages list them. "Fit only", "with ``file_path`` only" and
#: "under ``hard_concrete`` only" used to be ``if`` branches in the parser
#: and prose in the attribute docs; they are this table, which the parser
#: applies (`_parse_featurizer`) and the generator renders.
FEATURIZER_FIELD_CONDITIONS: dict[str, FieldLegality] = {
    "k": FieldLegality(None, None),
    "parametrization": FieldLegality(None, None),
    "group": FieldLegality(
        _UNIT_MAPS,
        _UNIT_MAPS,
        why_map=(
            "'group' requires one theta entry per unit; {maps} uses one scalar "
            "boundary (§2.5)"
        ),
    ),
    "axis": FieldLegality(
        _UNIT_MAPS,
        _UNIT_MAPS,
        why_map=(
            "'axis' requires theta entries per position; {maps} defines a prefix of "
            "an ordered feature basis (§2.5)"
        ),
    ),
    "init": FieldLegality(
        None,
        _NEVER,
        why_loaded=(
            "'init' sets the starting point of a fit; a loaded featurizer uses the "
            "weights in file_path"
        ),
    ),
    "seed": FieldLegality(
        None,
        _NEVER,
        why_loaded=(
            "'seed' selects an initial rotation; a loaded featurizer uses its saved "
            "rotation"
        ),
    ),
    "temperature": FieldLegality(_TEMPERED, _TEMPERED, why_map=_TEMPERED_ONLY),
    "stretch": FieldLegality(
        _HARD_CONCRETE, _HARD_CONCRETE, why_map=_HARD_CONCRETE_ONLY
    ),
    "dead": FieldLegality(
        _UNIT_MAPS,
        _NEVER,
        why_loaded=(
            "'dead' applies during training; a loaded gate uses a fixed mask (§2.5)"
        ),
        why_map=(
            "'dead' requires theta entries per unit; {maps} uses one scalar boundary "
            "(§2.5)"
        ),
    ),
    "top_k": FieldLegality(
        _NEVER,
        _UNIT_MAPS,
        why_fit=(
            "'top_k' requires file_path and selects the largest saved theta entries; "
            "training uses the map's own evaluation mask (§2.5)"
        ),
        why_map=(
            "'top_k' requires a ranking of units; {maps} selects a prefix using i < β "
            "(§2.5)"
        ),
    ),
    "k_schedule": FieldLegality(
        _BUDGET,
        _NEVER,
        why_loaded=(
            "'k_schedule' supplies training budgets; a loaded gate uses top_k (§2.5)"
        ),
        why_map=_BUDGET_ONLY,
    ),
    "stop_grad_shift": FieldLegality(
        _BUDGET,
        _NEVER,
        why_loaded=(
            "'stop_grad_shift' controls the solved shift's training gradient; a "
            "loaded gate uses top_k (§2.5)"
        ),
        why_map=_BUDGET_ONLY,
    ),
    "pool": FieldLegality(
        _BUDGET,
        _UNIT_MAPS,
        why_map=(
            "'pool' requires budget during fitting. A loaded pool requires per-unit "
            "maps, file_path, and a shared top_k; the selected map is {maps} (§2.5)"
        ),
    ),
}


def _legality_note(legality: FieldLegality) -> str:
    """The parenthetical the kinds table hangs on a conditionally legal field:
    where it is legal, in the spec's words, or ``""`` when everywhere."""
    every = frozenset(GATE_MAPS)

    def maps(legal: frozenset[str]) -> str:
        return " \\| ".join(f"`{m}`" for m in GATE_PARAMETRIZATIONS if m in legal)

    def state(legal: frozenset[str] | None) -> str:
        if legal is None or legal == every:
            return "any"
        if not legal:
            return "none"
        if len(every - legal) == 1:
            return f"any map but {maps(every - legal)}"
        return maps(legal)

    fit, loaded = state(legality.fit), state(legality.loaded)
    if fit == "any" and loaded == "any":
        return ""
    if loaded == "none":
        return "on a fit" if fit == "any" else f"under {fit}, on a fit"
    if fit == "none":
        return (
            "with `file_path`"
            if loaded == "any"
            else f"under {loaded}, with `file_path`"
        )
    if fit == loaded:
        return f"under {fit}"
    fit_part = "any map" if fit == "any" else fit
    loaded_part = "any map" if loaded == "any" else loaded
    return f"under {fit_part} to fit; {loaded_part} with `file_path`"


def _field_name_cell(kind: str, field: str) -> str:
    """One authored field's name and, when it has one, its closed vocabulary."""
    domain: tuple[str, ...] | None = None
    if field == "parametrization":
        domain = GATE_PARAMETRIZATIONS if kind == "gate" else PARAMETRIZATIONS
    elif field == "group":
        domain = GATE_GROUPS
    elif field == "axis":
        domain = GATE_AXES
    elif field == "dead":
        domain = GATE_DEAD_RULES
    cell = f"`{field}`"
    if domain is not None:
        cell += " ∈ " + " \\| ".join(f"`{v}`" for v in domain)
    return cell


def _field_cell(kind: str, field: str) -> str:
    """One authored field as the kinds table spells it: its name, its closed
    vocabulary when it has one, and where it is legal when that is not
    everywhere."""
    cell = _field_name_cell(kind, field)
    note = _legality_note(FEATURIZER_FIELD_CONDITIONS[field])
    return f"{cell} ({note})" if note else cell


def render_featurizer_kind_table() -> str:
    "Render feature maps, parameter slots, and allowed fields by kind (§2.5)."
    lines = [
        "| kind | featurize | param slots | authored fields |",
        "|---|---|---|---|",
    ]
    for kind in FEATURIZER_KINDS:
        family = FEATURIZER_FAMILIES[kind]
        name = f"`{kind}` (default)" if kind == "identity" else f"`{kind}`"
        slots = ", ".join(f"`{s}`" for s in FEATURIZER_SLOTS[kind]) or "none"
        fields = ", ".join(
            _field_cell(kind, f)
            for f in FEATURIZER_FIELD_CONDITIONS
            if f in FEATURIZER_FIELDS[kind]
        )
        lines.append(f"| {name} | {family.featurize} | {slots} | {fields or 'none'} |")
    return "\n".join(lines) + "\n"


def render_gate_map_table() -> str:
    "Render gate formulas and training options from GATE_MAPS (§2.5)."
    lines = [
        "| parametrization | soft mask (train) | after every optimizer step "
        "| hard mask (eval, apply) | mask penalty (`train.objective`) "
        "| `anneal` on `theta.temperature` | default start |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, gate_map in GATE_MAPS.items():
        label = f"`{name}` (absent)" if name == GATE_DEFAULT_MAP else f"`{name}`"
        if gate_map.penalty is None:
            others = " and ".join(f"`{k}`" for k in ("l1", "l0"))
            penalty = f"none; the budget fixes mask size. {others} are invalid (rule 4)"
        else:
            other = "l0" if gate_map.penalty == "l1" else "l1"
            penalty = (
                f"**`{gate_map.penalty}`** = {gate_map.penalty_formula}; "
                f"`{other}` is **refused** (rule 4)"
            )
        anneal = (
            "legal"
            if gate_map.anneals_temperature
            else f"**refused** (rule 4): {gate_map.no_temperature_because}"
        )
        cells = (
            label,
            gate_map.soft_mask,
            gate_map.post_step,
            gate_map.hard_mask,
            penalty,
            anneal,
            gate_map.default_start,
        )
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _kind_fields(kind: str) -> list[str]:
    """One kind's authorable fields, in the legality table's order."""
    if kind not in FEATURIZER_FIELDS:
        raise KeyError(f"{kind!r} is not a featurizer kind: {FEATURIZER_KINDS}")
    return [f for f in FEATURIZER_FIELD_CONDITIONS if f in FEATURIZER_FIELDS[kind]]


def _legality_cell(legal: frozenset[str] | None, *, has_maps: bool) -> str:
    if legal is None:
        return "any map" if has_maps else "legal"
    if not legal:
        return "**refused**"
    every = frozenset(GATE_MAPS)
    if legal == every:
        return "any map"
    if len(every - legal) == 1:
        (excluded,) = every - legal
        return f"any map but `{excluded}`"
    return ", ".join(f"`{m}`" for m in GATE_PARAMETRIZATIONS if m in legal)


def render_field_legality_table(kind: str) -> str:
    "Render allowed fields and parametrizations for fitting and loading."
    fields = _kind_fields(kind)
    if not fields:
        return (
            f"A `{kind}` uses the common fields `file_path`, `entry`, "
            "`dtype`, and `description` (§2.5).\n"
        )
    has_maps = kind == "gate"
    lines = ["| field | without `file_path` | with `file_path` |", "|---|---|---|"]
    for field in fields:
        legality = FEATURIZER_FIELD_CONDITIONS[field]
        lines.append(
            f"| {_field_name_cell(kind, field)} "
            f"| {_legality_cell(legality.fit, has_maps=has_maps)} "
            f"| {_legality_cell(legality.loaded, has_maps=has_maps)} |"
        )
    return "\n".join(lines) + "\n"


def render_field_refusals(kind: str) -> str:
    "Render field restrictions and validation rules for one featurizer kind."
    bullets: list[str] = []
    for field in _kind_fields(kind):
        legality = FEATURIZER_FIELD_CONDITIONS[field]
        if legality.why_fit:
            bullets.append(f"- `{field}` without `file_path`: {legality.why_fit}")
        if legality.why_loaded:
            bullets.append(f"- `{field}` with `file_path`: {legality.why_loaded}")
        if legality.why_map:
            bullets.append(
                f"- `{field}` under another map: "
                + legality.why_map.format(maps="the authored map")
            )
    for rule in RULES.values():
        if kind in rule.kinds:
            bullets.append(
                f"- [{rule.code}] `{rule.slug}`: {rule.title} (§5.{rule.number})"
            )
    if not bullets:
        bullets.append(f"- A `{kind}` follows the shared validation rules (§5).")
    return "\n".join(bullets) + "\n"


#: §2.5 the mapping form of a gate's ``parametrization``: ``{"forward":
#: "hard", "backward": <map>}``: the forward pass uses the map's training
#: mask **thresholded at ½** (0/1) while the backward pass sees the map's own
#: gradient, the straight-through idiom ``hard + (soft − soft.detach())``,
#: a hard-forward ablation of the soft fit. The map is any of
#: [`GATE_PARAMETRIZATIONS`][]. ``forward`` has the one value ``hard``: a
#: soft forward is the string form, and the sampled forward is
#: ``hard_concrete`` itself: either would be a second spelling of one
#: computation, digesting apart, so both are refused.
FORWARD_MASKS: tuple[str, ...] = ("hard",)

#: The kinds a ``budget`` gate's ``k_schedule`` draws its per-step budget by
#: (§2.5): ``fixed`` (``k`` every step: the plain sigmoid-top-k mask),
#: ``uniform`` over the integers ``[low, high]``, ``log_uniform`` over them
#: (``low ≥ 1``: a budget curriculum that spends as many
#: steps between 1 and 2 units as between 24 and 48).
K_SCHEDULE_KINDS: tuple[str, ...] = ("fixed", "uniform", "log_uniform")

#: What a ``k_schedule``'s numbers count (§2.5 ``k_schedule.of``): ``patched``
#:: units that take the counterfactual, the gate's own count and the default
#: when absent (no canonical-form change): or ``kept``, the units left clean.
#: A log-uniform draw on one is not log-uniform on the other, so a
#: ``k ~ LogUniform{1, N − 1}`` curriculum over kept units needs ``kept``; every internal reader works
#: in patched units, so a ``kept`` schedule is complemented exactly once, at
#: the gate (``eval`` included; ``top_k`` is not a schedule number and always
#: counts patched units).
K_SCHEDULE_OF: tuple[str, ...] = ("patched", "kept")

#: The hard-concrete constants when unauthored (§2.5): Louizos et al.'s
#: ``β = 2/3``, ``(γ, ζ) = (−0.1, 1.1)``: NeuroSurgeon's defaults verbatim.
#: Not materialized into the canonical form (the ``group`` precedent: no
#: spelling of the default), so an authored default and an absent one are
#: two spellings that digest apart: write neither, or both.
HARD_CONCRETE_TEMPERATURE: float = 2.0 / 3.0
HARD_CONCRETE_STRETCH: tuple[float, float] = (-0.1, 1.1)


def hard_concrete_theta(mask: float, stretch: tuple[float, float]) -> float:
    """The θ whose **deterministic** hard-concrete mask is ``mask`` (§2.5): the
    inverse of ``clip(σ(θ)·(ζ − γ) + γ, 0, 1)`` strictly inside the poles,
    ``logit((mask − γ) / (ζ − γ)) = log(mask − γ) − log(ζ − mask)``. It is
    both what ``init.fill p`` starts a unit at and, at ``mask = ½``, the
    eval-mode threshold ([`hard_concrete_threshold`][]): one map, so a fill
    of ½ starts exactly on the split.

    The balanced case: ``mask`` at the midpoint of the stretch, which is ½ at
    every symmetric stretch, the default included: is answered with an exact
    ``0.0`` rather than derived: in binary floating point ``1.1 − (−0.1)`` is
    not ``1.2`` and the derived value is about ``−4e−16``, which would count a
    θ of exactly ``0``: every unit at the default start: as kept, where the
    sigmoid gate's ``θ > 0`` does not."""
    lo, hi = float(stretch[0]), float(stretch[1])
    if not lo < mask < hi:
        raise ValueError(f"a mask value of {mask} is outside the stretch ({lo}, {hi})")
    below, above = mask - lo, hi - mask
    if math.isclose(below, above, rel_tol=1e-9, abs_tol=0.0):
        return 0.0
    return math.log(below) - math.log(above)


def hard_concrete_threshold(stretch: tuple[float, float]) -> float:
    """The θ above which a hard-concrete gate keeps a unit in eval mode (§2.5):
    where the stretched σ(θ) crosses ½, ``logit((½ − γ) / (ζ − γ))``: exactly
    ``0`` at a symmetric stretch ([`hard_concrete_theta`][]). One function
    for the two readers of a fitted bundle: the gate's own hard mask
    (``Gate.hard_threshold``) and the size-matched control
    (``analysis.random_mask``): so the two counts agree bit for bit."""
    return hard_concrete_theta(0.5, stretch)


#: §2.11 ``train.control``: the closed-loop counterpart of ``anneal``. A
#: closed vocabulary of controllers, of the fit signals one may observe, and
#: of the spaces it may move the controlled value in; and the defaults the
#: canonical form materializes so two spellings of one controller digest
#: identically (the ``train.optimizer`` treatment).
CONTROL_KINDS: tuple[str, ...] = ("pid",)
#: ``hard_mask_size``: a trained gate's kept-unit count through its hard mask
#:: the number ``fit_diagnostics.json`` reports under the same name.
#: ``hard_mask_fraction``: the same count divided by the number of units the
#: named gates have: kept units over all units: so a setpoint ramp
#: ``[1, 0, frac]`` and one set of gains mean the same thing over 8, 48 or
#: 2048 units, where a count-valued signal needs its ``ki`` rescaled per unit
#: count (the integral term is ``ki · (kept − setpoint)`` in the signal's units).
CONTROL_SIGNALS: tuple[str, ...] = ("hard_mask_size", "hard_mask_fraction")
CONTROL_SPACES: tuple[str, ...] = ("log", "linear")
CONTROL_DEFAULTS: dict[str, Any] = {
    "kd": 0.0,
    "space": "log",
    "bounds": [1e-8, 1e8],
    "d_clip": 5.0,
}
#: The prefix a control target uses to name an objective term's weight:
#: ``train.objective.<name>.weight``: the same address a sweep uses (§3).
OBJECTIVE_WEIGHT_PREFIX = "train.objective."

#: §2.11 ``train.anneal``: the shapes an open-loop schedule may take between
#: its endpoints. ``linear`` is the list spelling ``[start, end, frac]`` and
#: the default; ``geometric`` multiplies by a constant factor per step
#: (continuous sparsification's ``T ← T · r``, Savarese et al. 2020), so it
#: needs endpoints of one sign and neither zero.
ANNEAL_SHAPES: tuple[str, ...] = ("linear", "geometric")


@dataclasses.dataclass(frozen=True)
class AnnealSchedule:
    """One §2.11 ``anneal`` entry, parsed: the value walks from ``start`` to
    ``end`` over the first ``frac`` of the run and holds, along ``shape``.
    Two spellings: ``[start, end, frac]`` and ``{"from", "to", "frac",
    "shape"}``: land here alike, and the canonical form writes the list
    whenever the shape is linear ([`causalab.protocol.schema.explicit`][]), so the
    longer spelling of a linear schedule digests as the shorter one."""

    start: float
    end: float
    frac: float
    shape: str = "linear"

    def value_at(self, step: int, total_steps: int) -> float:
        """The scheduled value before update ``step`` of ``total_steps``:
        ``start`` at step 0, ``end`` from ``frac · total_steps`` on."""
        ramp_steps = max(1, int(self.frac * total_steps))
        progress = min(1.0, step / ramp_steps)
        if self.shape == "geometric":
            return self.start * (self.end / self.start) ** progress
        return self.start + (self.end - self.start) * progress


#: §2.11 ``train.phases[i].until``: how a phase's end is counted: a fraction
#: of the run's updates, or an absolute update count. One form per document.
PHASE_UNTIL_UNITS: tuple[str, ...] = ("frac", "updates")


@dataclasses.dataclass(frozen=True)
class PhaseSpec:
    """One §2.11 ``train.phases`` entry: a window of the run ending at
    ``until`` (``{"frac": f}`` or ``{"updates": n}``) inside which only
    ``params`` (a subset of ``train.params``) receive gradients, the phase's
    own ``anneal`` schedules run over the phase's steps, ``freeze_masks``
    names gates whose *hard* mask is snapshotted at the phase's start and used
    by every forward inside it, and ``optimizer`` may override ``lr`` /
    ``weight_decay`` for the phase's params. Everything not spelled is
    inherited from the top-level ``train``."""

    until: Mapping[str, int | float]
    params: tuple[str, ...]
    optimizer: Mapping[str, Any] | None = None
    anneal: Mapping[str, AnnealSchedule] | None = None
    freeze_masks: tuple[str, ...] = ()


#: Units that share one gate parameter (§2.5). ``head`` assigns one
#: parameter per head on a component with a head axis. ``expert_neuron``
#: assigns one per ``(expert, neuron)`` on ``expert_activation`` or
#: ``expert_neuron_output``; lookup through ``expert_idx`` preserves that
#: identity across routing choices. ``site`` assigns one parameter to the
#: whole site, with a ``(1, width)`` group map.
#:
GATE_GROUPS: tuple[str, ...] = ("head", "expert_neuron", "site")

#: A gate's parameter axis (§2.5). The default assigns one θ per feature
#: coordinate or group. ``position`` assigns one θ per addressed token
#: position, broadcast across its features. Every use must address a fixed
#: ``span`` window. A position gate accepts neither ``group`` nor ``pool``.
#: Chaining it before a feature gate applies the outer product ``m_t ·
#: m_j``.
#:
GATE_AXES: tuple[str, ...] = ("position",)

#: The axis of the site's declared shape each group groups over (§2.5's
#: ``group`` table, third column; §5.23): ``head`` needs a head axis: a
#: head-major component: and ``expert_neuron`` the routed-expert slot axis
#: (``topk``) on ``expert_activation`` or ``expert_neuron_output``. ``site``
#: uses the feature axis itself. Every site a featurizer may attach to has
#: one, so the map always exists and is one group wide.
#: Spelled in `causalab.protocol.registry.shapes`' ``AxisKind``
#: vocabulary, kept as plain strings here because ``schema`` sits below
#: ``shapes`` in the import order; the census guard asserts the two agree and
#: that every group has a row, so a group value with no axis behind it fails
#: CI rather than resolving to "no grouping".
GATE_GROUP_AXES: dict[str, str] = {
    "head": "head",
    "expert_neuron": "topk",
    "site": "feature",
}


@dataclasses.dataclass(frozen=True)
class FeaturizerSpec:
    """§2.5: a named feature-space map. Only choices are authored; widths
    and param shapes derive from (model, site). ``file_path`` loads a fitted
    artifact (its ``ArtifactIdentity`` is checked; a loaded featurizer may
    not be trained)."""

    #: Feature-space map from [`FEATURIZER_KINDS`][]. Defaults to
    #: ``identity``. Sweepable.
    #:
    kind: Leaf = "identity"
    #: Width of a ``subspace`` or ``pca`` feature space. Interchanges act in
    #: the first ``k`` basis columns and preserve the complementary ``d −
    #: k`` directions. Sweepable.
    #:
    k: Leaf | None = None
    #: Parameter map: [`PARAMETRIZATIONS`][] for ``subspace`` or
    #: [`GATE_PARAMETRIZATIONS`][] for ``gate``. Gates default to
    #: ``sigmoid``. A ``boundary`` gate learns one scalar over the preceding
    #: ordered basis; its allowed fields are listed in
    #: [`FEATURIZER_FIELD_CONDITIONS`][].
    #:
    parametrization: Leaf | None = None
    #: Initial values for a fit (§2.5). A subspace accepts ``{"file_path":
    #: …, "entry": …}`` and uses the saved basis's first ``k`` columns. A
    #: gate accepts ``{"fill": p}``, a saved ``theta`` through
    #: ``file_path``, or ``{"from_scores": …}``. ``fill`` maps p to θ for
    #: the chosen parametrization; under ``boundary``, p is the retained
    #: fraction. ``from_scores`` initializes the top ``keep`` units or
    #: scales z-scored values. ``entry`` selects a bundle entry. A loaded
    #: featurizer cannot declare ``init``.
    #:
    init: Mapping[str, Any] | None = None
    #: Seed for a subspace's initial rotation. Defaults to ``train.seed``,
    #: or 0 without training. Sweep seeds on an untrained rank-k subspace to
    #: construct matched random controls.
    #:
    seed: Leaf | None = None
    #: Unit covered by one gate parameter: ``head``, ``expert_neuron``, or
    #: ``site`` (§2.5). The default is one parameter per coordinate and is
    #: omitted from the canonical form. The model and component determine
    #: the coordinate-to-group map.
    #:
    group: Leaf | None = None
    #: Temperature β for ``hard_concrete`` or T for ``boundary``. Defaults
    #: to [`HARD_CONCRETE_TEMPERATURE`][] or 1, respectively. Sweepable.
    #: Use an ``anneal`` schedule to change the temperature during fitting;
    #: declaring both a constant and a schedule for the same gate is
    #: invalid. The sigmoid temperature is controlled through ``anneal``.
    #:
    temperature: Leaf | None = None
    #: Fixed ``[γ, ζ]`` bounds for ``hard_concrete``, defaulting to
    #: [`HARD_CONCRETE_STRETCH`][]. These bounds determine the evaluation
    #: threshold and enter ``ArtifactIdentity``. This field cannot be swept.
    #:
    stretch: tuple[float, float] | None = None
    #: Rule for hard-off units in a trained gate: exactly one of
    #: ``{"freeze_after": n}`` or ``{"leak": ε}`` (§2.5). The gate must
    #: appear in ``train.params`` and use a per-unit map. This training rule
    #: is excluded from the saved bundle's ``ArtifactIdentity``.
    #:
    dead: Mapping[str, Any] | None = None
    #: ``"position"`` assigns θ to addressed token positions (§2.5). The
    #: default ``None`` assigns θ to features.
    #:
    axis: Leaf | None = None
    #: ``"hard"`` uses a training mask thresholded at ½ in the forward pass,
    #: with the selected map's gradient ([`FORWARD_MASKS`][]). In this
    #: mapping form, ``parametrization`` stores the backward map. Defaults
    #: to the plain map.
    #:
    forward: str | None = None  # never swept: the mapping form is not a leaf
    #: Number of units selected from a loaded gate's largest ``theta``
    #: values. Requires ``file_path`` and a per-unit map. Sweepable over
    #: integers in ``[0, units]``; 0 keeps the original activations. A
    #: loaded ``budget`` gate requires this field. Other maps use their
    #: threshold when it is omitted.
    #:
    top_k: Leaf | None = None
    #: Training budget for a ``budget`` gate, required during fitting
    #: (§2.5). Use ``{"kind": "fixed", "k": n}`` or ``{"kind": "uniform" |
    #: "log_uniform", "low": a, "high": b}``. ``eval`` sets the evaluation
    #: cut and the reported ``hard_mask_size``; it defaults to ``k`` for a
    #: fixed schedule and is required for a sampled schedule. ``k`` and
    #: ``eval`` are sweepable. Loaded gates use ``top_k``.
    #:
    k_schedule: Mapping[str, Any] | None = None
    #: Training option for ``budget`` gates. When true, the solved shift is
    #: constant in backpropagation, leaving only the direct ``σ'`` gradient.
    #: By default the shift carries ``∂c/∂θ_i = −σ'_i / Σ σ'_j``, preserving
    #: ``Σ m = k`` to first order under an update.
    #:
    stop_grad_shift: Leaf | None = None
    #: Name of a shared budget or readout pool (§2.5). During fitting,
    #: ``budget`` gates with this name share one ``k_schedule``, solved
    #: shift, and ranking across their units. Loaded per-unit gates may
    #: share one ``top_k`` cut under any map. Members must agree on the
    #: schedule or cut. The name cannot be swept. Saved budget pools stamp
    #: ``pool`` and ``pool_units``; loading must match that identity. An
    #: unstamped bundle may join a readout pool.
    #:
    pool: Leaf | None = None
    #: Precision used to hold and save featurizer parameters
    #: ([`PRECISION_DTYPES`][]). Defaults to the model precision. A loaded
    #: document must match the bundle's dtype.
    #:
    dtype: Leaf | None = None
    #: Path to a fitted artifact. The loaded featurizer uses its saved
    #: parameters and accepts no training, initialization, or training-rule
    #: fields. A loaded budget gate requires ``top_k``. ``ArtifactIdentity``
    #: is checked at load and build (rule 15). Sweep bundle paths to compare
    #: fits.
    #:
    file_path: Leaf | None = None
    #: Coordinate selector for a bundle loaded through ``file_path``.
    #: Required when the bundle contains several entries; otherwise the sole
    #: entry is used.
    #:
    entry: Any = None
    #: Description for readers. Excluded from the canonical form and digest.
    #:
    description: str | None = None
