"""Parallel plans: module pattern → (style, axis) tables for tensor and
expert sharding (``docs/model_parallelism.md`` §5.2).

A plan is a fact of the HF ``model_type`` (the config class), declared on the
built-in entries and derived from a loaded config otherwise, so ``dry-run``
checks a geometry from the registry entry alone, torch-free and offline.
Never in any canonical form: geometry is execution, not identity.
"""

from __future__ import annotations

import dataclasses
import re
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Literal, Mapping, get_args

from causalab.protocol.kv_replication import kv_refusal
from causalab.protocol.rules.errors import ProtocolError

if TYPE_CHECKING:
    from causalab.protocol.parallel import ParallelGeometry
    from causalab.protocol.registry.models import ModelInfo

__all__ = [
    "GEMMA2_PLAN",
    "KV_PROJECTIONS",
    "KV_REPLICATED",
    "LLAMA_PLAN",
    "NO_PLAN",
    "PLAN_AXES",
    "QWEN3_PLAN",
    "QWEN35_MOE_PLAN",
    "QWEN35_PLAN",
    "STYLES",
    "VOCABULARY_STYLES",
    "ParallelPlan",
    "PlanAxis",
    "PlanRow",
    "parallel_plan_from_hf_config",
    "wildcard_layers",
]


# --------------------------------------------------------------------------- #
# the parallel plan — module pattern → (style, axis), a registry table
# (docs/model_parallelism.md §5.2)
# --------------------------------------------------------------------------- #

#: The two axes a plan row shards over (``docs/model_parallelism.md`` §2):
#: attention and dense MLPs over the tensor group, routed experts over the
#: expert group. A subset of the geometry's ``Axis`` vocabulary
#: (``protocol/parallel.py``), which asserts the inclusion.
PlanAxis = Literal["tensor", "expert"]
PLAN_AXES: tuple[PlanAxis, ...] = get_args(PlanAxis)

#: The parallel styles this repository serves this phase — the rows of the
#: placement table (§4): eight of transformers' and one of its own,
#: ``kv_replicated`` (§6.6) — the K/V projection whole on every rank of the
#: tensor group, its output the one KV head this rank's query heads read;
#: never on a config's plan, written by [`ParallelPlan.for_geometry`][] when
#: the tensor axis exceeds the KV heads. A config's plan may name others
#: (``mla_kv_a_proj``, …); such a row is kept on the plan, marked
#: [`ParallelPlan.unserved`][], and refused **by name** by the geometry
#: check the moment the axis it sits on is above one — before any weights,
#: never as a wrong number — except the vocabulary styles below, which are
#: dropped. Three buckets, then: served, unserved (refused), dropped.
STYLES: frozenset[str] = frozenset(
    {
        "colwise",
        "rowwise",
        "packed_colwise",
        "colwise_gather_output",
        "replicated_with_grad_allreduce",
        "moe_tp_experts",
        "grouped_gemm",
        "ep_router",
        "kv_replicated",
    }
)

#: Styles a config's plan may name that shard the **vocabulary** — the
#: embedding's rows over the tensor group (``embedding_rowwise``; the tied
#: head with it). Not applied, not refused: the embedding and the head are
#: whole on every rank (§6.1's "full vocabulary"), so the derivation drops
#: the row and the module is placed like any module in no row, replicated.
#: Keyed on the **style**, not on a match against the tree's
#: embedding: the style names what it shards, and a config naming it on
#: another module would be sharding a vocabulary-shaped table there too.
VOCABULARY_STYLES: frozenset[str] = frozenset({"embedding_rowwise"})

#: The one style the repository writes itself (above).
KV_REPLICATED = "kv_replicated"

#: The pattern leaves of the key and value projections in every served
#: transformers plan — the rows [`ParallelPlan.for_geometry`][] replaces
#: with [`KV_REPLICATED`][] when the tensor axis exceeds the KV heads.
KV_PROJECTIONS: frozenset[str] = frozenset({"k_proj", "v_proj"})

_LAYER_INDEX = re.compile(r"\.\d+(\.|$)")


def wildcard_layers(name: str) -> str:
    """transformers' ``replace_layer_number_by_wildcard``, transcribed: a
    number between two dots, or between a dot and the end, is a ModuleList
    index and becomes ``*``; a digit inside a name (``w1``) is not."""
    return _LAYER_INDEX.sub(lambda m: ".*" + m.group(1), name)


@dataclasses.dataclass(frozen=True)
class PlanRow:
    """One row of a parallel plan: the style applied at the module (or
    parameter) the pattern names, over the group of ``axis``. ``repeat`` is
    [`KV_REPLICATED`][]'s alone: how many consecutive ranks of the group
    hold the same chunk of the module's output — ``tp / num_kv_heads``, one
    KV head per rank (``docs/model_parallelism.md`` §6.6); 1 on every other
    row, where each rank's chunk is its own."""

    style: str
    axis: PlanAxis
    repeat: int = 1

    def __post_init__(self) -> None:
        if not self.style:
            raise ValueError(
                f"plan row: style must be a non-empty name, got {self.style!r}"
            )
        if self.axis not in PLAN_AXES:
            raise ValueError(
                f"plan row for {self.style!r}: axis {self.axis!r} is not one of "
                f"{list(PLAN_AXES)}"
            )
        if isinstance(self.repeat, bool) or not isinstance(self.repeat, int):
            raise ValueError(
                f"plan row for {self.style!r}: repeat must be an int, got {self.repeat!r}"
            )
        if self.repeat < 1:
            raise ValueError(
                f"plan row for {self.style!r}: repeat must be at least 1, got "
                f"{self.repeat}"
            )
        if self.repeat > 1 and self.style != KV_REPLICATED:
            raise ValueError(
                f"plan row for {self.style!r}: repeat={self.repeat} belongs to the "
                f"{KV_REPLICATED!r} style alone; every other style holds one chunk "
                "per rank"
            )

    @property
    def served(self) -> bool:
        return self.style in STYLES


@dataclasses.dataclass(frozen=True, eq=False)
class ParallelPlan:
    """Module pattern (``layers.*.self_attn.q_proj``, the config's spelling,
    without the base-model prefix) → [`PlanRow`][]. A value: equal and
    hashed by its rows, whatever order they were declared in. The empty plan
    is a family whose transformers config ships no plan at all
    (``PLANLESS_FAMILIES`` in ``protocol/parallel.py`` is the same fact by
    family name)."""

    rows: Mapping[str, PlanRow]
    #: The config's rows the derivation declined — pattern → style, today
    #: the [`VOCABULARY_STYLES`][] (``embed_tokens: embedding_rowwise``):
    #: the embedding and the head stay whole on every rank (§6.1), so the
    #: row is neither applied nor refused. Provenance, not the plan's value:
    #: ``dry-run`` names it under the ``parallel`` fact so the choice is
    #: visible; equality and the hash read ``rows`` alone.
    unapplied: Mapping[str, str] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        for pattern in self.rows:
            if not pattern:
                raise ValueError(f"plan: pattern {pattern!r} is not a module pattern")
        # plain dicts, not read-only views: a ``ModelInfo`` is deep-copied and
        # ``dataclasses.asdict``-ed by its readers, and a ``mappingproxy`` is
        # neither copyable nor picklable; the plan is frozen by convention
        # (equality and the hash read the rows' items)
        object.__setattr__(self, "rows", dict(self.rows))
        object.__setattr__(self, "unapplied", dict(self.unapplied))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ParallelPlan):
            return NotImplemented
        return dict(self.rows) == dict(other.rows)

    def __hash__(self) -> int:
        return hash(frozenset(self.rows.items()))

    @property
    def empty(self) -> bool:
        return not self.rows

    @property
    def unserved(self) -> Mapping[str, PlanRow]:
        """The rows whose style is outside [`STYLES`][]."""
        return {pattern: row for pattern, row in self.rows.items() if not row.served}

    def rows_on(self, axis: PlanAxis) -> Mapping[str, PlanRow]:
        return {pattern: row for pattern, row in self.rows.items() if row.axis == axis}

    def style_for(self, path: str, *, prefix: str | None = None) -> PlanRow | None:
        """The row for a concrete module or parameter path
        (``model.layers.3.self_attn.q_proj``): its layer indices wildcarded,
        looked up as is and then with the base-model prefix stripped —
        exactly ``prefix`` when given (``model``, ``transformer``), else the
        first dotted component, the way transformers adds the prefix to its
        plan before matching module names. ``None`` off the table."""
        generic = wildcard_layers(path)
        row = self.rows.get(generic)
        if row is not None:
            return row
        if prefix is not None:
            stripped = (
                generic[len(prefix) + 1 :] if generic.startswith(prefix + ".") else None
            )
        else:
            _, dot, rest = generic.partition(".")
            stripped = rest if dot else None
        return self.rows.get(stripped) if stripped else None

    def for_geometry(self, geometry: ParallelGeometry, info: ModelInfo) -> ParallelPlan:
        """The plan ``apply_plan`` applies and the placements are read from
        under ``geometry`` (``docs/model_parallelism.md`` §5.2, §6.6): this
        very plan — a fact of the model type — unless the tensor axis
        exceeds ``info.num_kv_heads``, when the colwise key and value
        projections (the rows whose pattern ends in one of
        [`KV_PROJECTIONS`][]) become [`KV_REPLICATED`][] with
        ``repeat = tp / num_kv_heads`` and every other row stays. A plan
        with no tensor row, or a tensor axis of one, is returned as is: the
        rows rule refuses the former, and the latter applies nothing.

        Raises:
            ProtocolError: ``P4`` naming ``--parallel.tensor`` — the head
                facts ``check`` states, restated at the seam that would
                otherwise shard a wrong number: the axis does not divide the
                heads, or the shape straddles the KV heads
                ([`kv_refusal`][]); or
                the plan has tensor rows but no colwise key / value
                projection to replicate, or a key / value row is not
                ``colwise``.
        """
        tensor = geometry.tensor
        if tensor == 1 or not self.rows_on("tensor"):
            return self
        if info.num_heads % tensor:
            raise ProtocolError(
                "P4",
                f"--parallel.tensor: tp={tensor} does not divide num_heads="
                f"{info.num_heads} of model {info.key!r}",
            )
        refusal = kv_refusal(
            num_heads=info.num_heads,
            num_kv_heads=info.num_kv_heads,
            tensor=tensor,
            key=info.key,
        )
        if refusal is not None:
            raise ProtocolError("P4", f"--parallel.tensor: tp={tensor} {refusal}")
        if tensor <= info.num_kv_heads:
            return self
        kv = {
            pattern: row
            for pattern, row in self.rows_on("tensor").items()
            if pattern.rsplit(".", 1)[-1] in KV_PROJECTIONS
        }
        if not kv:
            raise ProtocolError(
                "P4",
                f"--parallel.tensor: tp={tensor} exceeds num_kv_heads="
                f"{info.num_kv_heads} of model {info.key!r}, and the parallel plan "
                f"of family {info.family!r} has no colwise "
                f"{' / '.join(sorted(KV_PROJECTIONS))} row on the tensor axis to "
                "replicate",
            )
        odd = {
            pattern: row.style for pattern, row in kv.items() if row.style != "colwise"
        }
        if odd:
            named = ", ".join(f"{p} → {s}" for p, s in sorted(odd.items()))
            raise ProtocolError(
                "P4",
                f"--parallel.tensor: tp={tensor} exceeds num_kv_heads="
                f"{info.num_kv_heads} of model {info.key!r}, and replicating the KV "
                f"heads replaces colwise key / value rows only; the plan of family "
                f"{info.family!r} has {named}",
            )
        repeat = tensor // info.num_kv_heads
        rows = dict(self.rows)
        for pattern in kv:
            rows[pattern] = PlanRow(KV_REPLICATED, "tensor", repeat=repeat)
        # the rows change, the provenance does not: what the derivation
        # declined is still declined under this geometry
        return dataclasses.replace(self, rows=rows)


def parallel_plan_from_hf_config(config: Any) -> ParallelPlan:
    """The plan a transformers text config ships, as one table
    (``docs/model_parallelism.md`` §5.2): every ``base_model_tp_plan`` row
    on the tensor axis, then the ``base_model_ep_plan`` rows on the expert
    axis **replacing** the TP rows at their patterns and under them (the
    experts' parameters and the router: ``grouped_gemm`` where the TP plan
    said ``packed_colwise`` / ``rowwise``); the attention and shared-expert
    rows stay on the tensor axis. Reads two attributes and nothing else, so
    it is total: a style outside [`STYLES`][] is kept and marked
    unserved, and refused by the geometry check by name when its axis is
    above one — except a row sharding the **vocabulary**
    ([`VOCABULARY_STYLES`][]), which is not applied at all: the embedding
    and the head are whole on every rank by design (§6.1; a head tied to
    its embedding is one replicated tensor under ``tp``). A config with
    neither attribute derives to the empty plan."""
    tp = dict(getattr(config, "base_model_tp_plan", None) or {})
    ep = dict(getattr(config, "base_model_ep_plan", None) or {})
    rows: dict[str, PlanRow] = {
        pattern: PlanRow(str(style), "tensor")
        for pattern, style in tp.items()
        if str(style) not in VOCABULARY_STYLES
    }
    unapplied = {
        pattern: str(style)
        for pattern, style in tp.items()
        if str(style) in VOCABULARY_STYLES
    }
    for pattern in ep:
        for replaced in [
            p for p in rows if p == pattern or p.startswith(pattern + ".")
        ]:
            del rows[replaced]
    for pattern, style in ep.items():
        rows[pattern] = PlanRow(str(style), "expert")
    return ParallelPlan(rows, unapplied)


#: The plans the built-in entries declare, spelled from the transformers
#: 5.16.1 config classes (``base_model_tp_plan`` / ``base_model_ep_plan``) so
#: ``dry-run`` reads them torch-free and offline;
#: ``tests/protocol/test_parallel_plan.py`` holds each to the derivation from
#: ``AutoConfig.for_model(family)``.
NO_PLAN = ParallelPlan({})

_DENSE_ROWS: dict[str, str] = {
    "layers.*.self_attn.q_proj": "colwise",
    "layers.*.self_attn.k_proj": "colwise",
    "layers.*.self_attn.v_proj": "colwise",
    "layers.*.self_attn.o_proj": "rowwise",
    "layers.*.mlp.gate_proj": "colwise",
    "layers.*.mlp.up_proj": "colwise",
    "layers.*.mlp.down_proj": "rowwise",
}


def _tensor_rows(
    styles: Mapping[str, str], *, unapplied: Mapping[str, str] | None = None
) -> ParallelPlan:
    """Tensor-axis rows, with the config's declined rows as ``unapplied`` —
    the same provenance [`parallel_plan_from_hf_config`][] records, so a
    hand-spelled plan reads under ``dry-run`` as the derived one does."""
    return ParallelPlan(
        {k: PlanRow(v, "tensor") for k, v in styles.items()}, unapplied or {}
    )


LLAMA_PLAN = _tensor_rows(_DENSE_ROWS)
QWEN3_PLAN = _tensor_rows(
    {
        **_DENSE_ROWS,
        "layers.*.self_attn.q_norm": "replicated_with_grad_allreduce",
        "layers.*.self_attn.k_norm": "replicated_with_grad_allreduce",
    }
)
#: Gemma2's config class ships an ``embed_tokens → embedding_rowwise`` row,
#: which the derivation drops ([`VOCABULARY_STYLES`][]): its plan is the
#: dense rows, the vocabulary whole on every rank, and the declined row its
#: ``unapplied`` provenance, so ``dry-run … --parallel tp=2`` on a gemma2
#: document names it. (Qwen3's *class* ships no such row — the
#: Qwen3-4B-Instruct-2507 checkpoint's ``config.json`` does, and the load
#: derives from the checkpoint's config, so the run declines it there.)
GEMMA2_PLAN = _tensor_rows(_DENSE_ROWS, unapplied={"embed_tokens": "embedding_rowwise"})
QWEN35_MOE_PLAN = parallel_plan_from_hf_config(
    SimpleNamespace(
        base_model_tp_plan={
            "layers.*.self_attn.q_proj": "colwise",
            "layers.*.self_attn.k_proj": "colwise",
            "layers.*.self_attn.v_proj": "colwise",
            "layers.*.self_attn.o_proj": "rowwise",
            "layers.*.self_attn.q_norm": "replicated_with_grad_allreduce",
            "layers.*.self_attn.k_norm": "replicated_with_grad_allreduce",
            "layers.*.mlp.experts.gate_up_proj": "packed_colwise",
            "layers.*.mlp.experts.down_proj": "rowwise",
            "layers.*.mlp.experts": "moe_tp_experts",
            "layers.*.mlp.shared_expert.gate_proj": "colwise",
            "layers.*.mlp.shared_expert.up_proj": "colwise",
            "layers.*.mlp.shared_expert.down_proj": "rowwise",
            "layers.*.linear_attn.in_proj_qkv": "colwise_gather_output",
            "layers.*.linear_attn.in_proj_z": "colwise_gather_output",
            "layers.*.linear_attn.in_proj_b": "colwise_gather_output",
            "layers.*.linear_attn.in_proj_a": "colwise_gather_output",
            "layers.*.linear_attn.out_proj": "colwise_gather_output",
        },
        base_model_ep_plan={
            "layers.*.mlp.gate": "ep_router",
            "layers.*.mlp.experts.gate_up_proj": "grouped_gemm",
            "layers.*.mlp.experts.down_proj": "grouped_gemm",
            "layers.*.mlp.experts": "moe_tp_experts",
        },
    )
)
#: The dense Qwen3.5 text tower (``Qwen3_5TextConfig``, ``model_type``
#: ``qwen3_5_text``): the A3B's attention and Gated DeltaNet rows around a
#: dense SwiGLU MLP. Spelled from the config class, which ships no vocabulary
#: row; the checkpoint's config.json adds ``embed_tokens: embedding_rowwise``,
#: which the load declines as ``unapplied`` (the same split as Qwen3's).
QWEN35_PLAN = parallel_plan_from_hf_config(
    SimpleNamespace(
        base_model_tp_plan={
            "layers.*.self_attn.q_proj": "colwise",
            "layers.*.self_attn.k_proj": "colwise",
            "layers.*.self_attn.v_proj": "colwise",
            "layers.*.self_attn.o_proj": "rowwise",
            "layers.*.self_attn.q_norm": "replicated_with_grad_allreduce",
            "layers.*.self_attn.k_norm": "replicated_with_grad_allreduce",
            "layers.*.mlp.gate_proj": "colwise",
            "layers.*.mlp.up_proj": "colwise",
            "layers.*.mlp.down_proj": "rowwise",
            "layers.*.linear_attn.in_proj_qkv": "colwise_gather_output",
            "layers.*.linear_attn.in_proj_z": "colwise_gather_output",
            "layers.*.linear_attn.in_proj_b": "colwise_gather_output",
            "layers.*.linear_attn.in_proj_a": "colwise_gather_output",
            "layers.*.linear_attn.out_proj": "colwise_gather_output",
        },
    )
)
