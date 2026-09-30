"""Classify how an address aligns across a counterfactual pair.

Each address resolves on the original input and the counterfactual input.
Cardinality records whether the runs align one to one, have unequal widths,
are ambiguous, or are missing. Validators compare the observed cardinality
with the document's declared alignment and check pair edits where required.

The executor uses this record to decide whether a write can address the pair."""

from __future__ import annotations

import dataclasses
import difflib
from typing import Any, Sequence

from causalab.protocol.positions.spans import static_indices
from causalab.protocol.results import Unavailable, unavailable
from causalab.protocol.rules.errors import ProtocolError, ReasonCode
from causalab.protocol.schema import (
    AlignmentCardinality,
    Document,
    PositionSpec,
    SpanSpec,
)

__all__ = [
    "COMPONENT_RANK",
    "Hunk",
    "PAST_BLOCKS",
    "PairDifferences",
    "UNRANKED",
    "UnalignableError",
    "alignment_of",
    "check_declared",
    "pair_differences",
    "refuse_unalignable",
    "site_depth",
    "site_depths",
    "static_alignment",
    "token_runs",
    "unalignable",
    "unalignable_reason",
]


def alignment_of(
    base: Sequence[Sequence[int]],
    counterfactual: Sequence[Sequence[int]] | None = None,
) -> AlignmentCardinality:
    """The observed cardinality of one address across a pair.

    ``base`` and ``counterfactual`` are the **candidate** runs the address
    resolved to on each side — one entry per occurrence of the value in the
    row's text, each a run of token indices. ``counterfactual`` is ``None``
    when the address is resolved on one input alone (a document with one
    role, or the per-input half of a pair), in which case the classification
    is over the candidates only: none is ``absent``, several is
    ``ambiguous``, one is ``one_to_one``.
    """
    sides = [base] if counterfactual is None else [base, counterfactual]
    for side in sides:
        if len(side) == 0:
            return "absent"
    for side in sides:
        if len(side) > 1:
            return "ambiguous"
    widths = [len(side[0]) for side in sides]
    if any(width == 0 for width in widths):
        return "absent"
    if len(widths) == 1 or widths[0] == widths[1]:
        return "one_to_one"
    if widths[0] == 1:
        return "one_to_many"
    if widths[1] == 1:
        return "many_to_one"
    return "ambiguous"


def token_runs(
    needle: Sequence[int], haystack: Sequence[int]
) -> tuple[tuple[int, ...], ...]:
    """Every occurrence of ``needle`` as a contiguous run inside ``haystack``,
    as runs of ``haystack`` indices — the candidate runs of one token
    sequence addressed inside another (a variable's value inside the row's
    text)."""
    n = len(needle)
    if n == 0:
        return ()
    return tuple(
        tuple(range(start, start + n))
        for start in range(len(haystack) - n + 1)
        if tuple(haystack[start : start + n]) == tuple(needle)
    )


def unalignable_reason(observed: AlignmentCardinality) -> ReasonCode | None:
    """The reason code an unalignable cardinality carries (§2.4), or ``None``
    for the three that pair."""
    if observed == "absent":
        return "alignment_missing"
    if observed == "ambiguous":
        return "alignment_ambiguous"
    return None


def unalignable(
    observed: AlignmentCardinality, detail: str, denominator_key: str
) -> Unavailable | None:
    """The ``unavailable`` cell an unalignable *read* row becomes (§4.1), or
    ``None`` when the cardinality pairs. ``detail`` names the value, how many
    times it occurred and the row; the cell is counted in the denominator
    under ``denominator_key``."""
    if observed == "absent":
        return unavailable("alignment_missing", detail, denominator_key)
    if observed == "ambiguous":
        return unavailable("alignment_ambiguous", detail, denominator_key)
    return None


class UnalignableError(ProtocolError):
    """A row an address could not be aligned on, as a refusal.

    Raised where the row is not a *read's* to record as an unavailable cell
    — position resolution itself, a write's positions, a metric's answer
    form — with the reason code the cardinality maps to. ``cardinality`` is
    the observed value, so an executor catching this for a read can build the
    cell with [`unalignable`][] rather than re-deriving it.
    """

    def __init__(self, cardinality: AlignmentCardinality, message: str) -> None:
        reason = unalignable_reason(cardinality)
        if reason is None:
            raise AssertionError(f"{cardinality!r} is not an unalignable cardinality")
        self.cardinality: AlignmentCardinality = cardinality
        super().__init__("P2", message, reason=reason)


def refuse_unalignable(observed: AlignmentCardinality, message: str) -> None:
    """Raise [`UnalignableError`][] when ``observed`` is ``absent`` or
    ``ambiguous``; return otherwise. The refusal carries reason
    ``alignment_missing`` for the first and ``alignment_ambiguous`` for the
    second — the two spellings are written here and nowhere else."""
    if observed == "absent":
        raise UnalignableError(observed, message)
    if observed == "ambiguous":
        raise UnalignableError(observed, message)


def check_declared(
    declared: str | None, observed: AlignmentCardinality, *, where: str
) -> None:
    """Refuse a declared ``alignment`` the observed cardinality contradicts.

    ``declared`` is the position's authored value (``None`` declares
    nothing and is never refused). A contradiction is an encode-time refusal
    (the V19 boundary, §2.3): the document *said* how the address pairs and
    the tokenizer says otherwise, which is a wrong number waiting to happen
    if the declaration were honored over the observation, and a silent
    override the other way. The message names both.
    """
    if declared is None or declared == observed:
        return
    reason = unalignable_reason(observed)
    raise ProtocolError(
        "P2",
        f"{where} declares alignment {declared!r}, but the pair resolves it as "
        f"{observed!r} — a declared cardinality is checked against the "
        "tokenizer's, never honored over it. Correct the declaration, or drop "
        "it to let the observed cardinality stand",
        reason=reason,
    )


# --------------------------------------------------------------------------- #
# the pair-difference validator
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class Hunk:
    """One differing stretch between two token sequences: ``op`` is
    difflib's ``replace`` / ``delete`` / ``insert``; ``base`` and
    ``counterfactual`` are the decoded tokens each side has there."""

    op: str
    base: tuple[str, ...]
    counterfactual: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class PairDifferences:
    """The three difference sets of one counterfactual pair, kept apart.

    ``prompt`` compares the two prompts; ``teacher_forced_prefix`` the two
    answer prefixes; ``full_context`` the two ``prompt + prefix`` strings,
    each tokenized as one string. The three are reported separately because
    the error class this exists for is reading a prefix change as a prompt
    edit: [`prompt_edited`][] is the question "did the *prompt* change",
    and it is answered by the first set alone.
    """

    prompt: tuple[Hunk, ...]
    teacher_forced_prefix: tuple[Hunk, ...]
    full_context: tuple[Hunk, ...]

    @property
    def prompt_edited(self) -> bool:
        """Whether the pair differs in its prompt — the intended edit of a
        counterfactual pair, and the one thing a recomputed prefix is not."""
        return bool(self.prompt)

    @property
    def prefix_recomputed(self) -> bool:
        """Whether the pair differs in its teacher-forced answer prefix."""
        return bool(self.teacher_forced_prefix)


def _tokens(tokenizer: Any, text: str) -> list[str]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    return [str(t) for t in tokenizer.convert_ids_to_tokens(list(ids))]


def _hunks(tokenizer: Any, base: str, counterfactual: str) -> tuple[Hunk, ...]:
    a, b = _tokens(tokenizer, base), _tokens(tokenizer, counterfactual)
    matcher = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    return tuple(
        Hunk(op=op, base=tuple(a[i1:i2]), counterfactual=tuple(b[j1:j2]))
        for op, i1, i2, j1, j2 in matcher.get_opcodes()
        if op != "equal"
    )


def pair_differences(
    tokenizer: Any,
    base_prompt: str,
    counterfactual_prompt: str,
    *,
    base_prefix: str = "",
    counterfactual_prefix: str = "",
) -> PairDifferences:
    """The three difference sets of one pair (module docstring).

    ``*_prefix`` is each input's teacher-forced answer prefix — the part of
    the context the task recomputes per input and the model is forced through
    before the position a metric reads. The default (no prefix on either
    side) is every v1 corpus document, whose rows end at the prompt.
    """
    return PairDifferences(
        prompt=_hunks(tokenizer, base_prompt, counterfactual_prompt),
        teacher_forced_prefix=_hunks(tokenizer, base_prefix, counterfactual_prefix),
        full_context=_hunks(
            tokenizer,
            base_prompt + base_prefix,
            counterfactual_prompt + counterfactual_prefix,
        ),
    )


# --------------------------------------------------------------------------- #
# the planning half: what the document alone decides
# --------------------------------------------------------------------------- #


def static_alignment(doc: Document, pos: Any) -> AlignmentCardinality | None:
    """The cardinality a position has **by construction**, from the document
    alone (§2.3) — or ``None`` when only the tokenizer can say.

    An ``index`` (bare, scoped or relative) is one token per row on every
    input; an unscoped ``span [a, b)`` is one joint window of width ``b − a``
    on every input. Both pair ``one_to_one`` whatever the text says — two
    ``index`` specs are two locations, one ``span`` is one joint address — so
    a declaration that says otherwise is refusable at load (rule 26). This is
    the planning half of [`alignment_of`][]'s
    two callers: what the plan knows about the pairing shape before any
    encode. A ``variable`` or ``column`` window, a scoped span (clipped to
    its anchor's width) and ``all`` are as wide as the tokenizer makes them,
    and the executor decides those at encode time.

    Takes the spelling a read or write carries: a ``positions`` name or an
    inline spec.
    """
    spec = doc.positions[pos] if isinstance(pos, str) else pos
    if not isinstance(spec, PositionSpec) or spec.generated is not None:
        return None
    if isinstance(spec, SpanSpec):
        # A span the document alone fixes (an unscoped `indices` set, an
        # atomic unscoped `span`, a union of such) is one set of one width on
        # every input — an atomic span as one joint address, a non-atomic set
        # as constituents that are each one token. Anything text-located
        # (`segment`, `variable`, a predicate) is the tokenizer's to decide.
        fixed = static_indices(spec)
        return alignment_of((fixed,), (fixed,)) if fixed is not None else None
    if spec.index is not None:
        run: tuple[int, ...] = (0,)
    elif (
        spec.span is not None
        and spec.scope is None
        and isinstance(spec.span, tuple)
        and len(spec.span) == 2
    ):
        lo, hi = (int(v) for v in spec.span)
        run = tuple(range(lo, hi))
    else:
        return None
    return alignment_of((run,), (run,))


# --------------------------------------------------------------------------- #
# site depth — the forward-pass order rule 21 and the planner read
# (from the planner, now ``neural/shared/plan.py``; protocol-side because
# rule 21 reads it and the planner is engine-side)
# --------------------------------------------------------------------------- #

COMPONENT_RANK: dict[str, int] = {
    "input_ids": -10,  # the model's input: before every activation
    "embeddings": 0,
    "block_input": 100,
    "attention_input_norm": 150,  # input_layernorm, between resid_pre and mixer
    # The DeltaNet mixer's interior interleaves numerically with the
    # full-attention band below: a layer carries one stream or the other, so
    # only relative order *within* a stream is ever compared, and the numbers
    # avoid every attention slot so that neither band renumbers the other.
    "delta_qkv": 152,  # in_proj_qkv's fused [q|k|v] output, pre-conv
    "delta_gate": 154,  # in_proj_z's output — the output gate, produced early
    "delta_conv": 156,  # causal_conv1d_fn's return, channels-first
    "delta_query": 158,  # kernel arg 0: post-conv, post-tiling, PRE-l2norm
    # The two pre-tiling faces only the nnsight engine serves (the typed
    # backend pairs, ``registry.BACKEND_PAIRS``), ranked where the forward computes them — the
    # q/k splits of the conv output, before the value split — because the
    # `.source` interiors refuse out-of-order requests and the one-name
    # `delta_*` band is requested by the same engine in the same forward.
    "deltanet_query": 159,
    # The mixer's interior, in the order the forward computes it. All four are
    # module boundaries: q_norm/k_norm run BEFORE RoPE and are nn.Modules, so
    # the pre-RoPE projections are ordinary forward hooks rather than taps
    # inside the attention function.
    "attention_query_pre_rope": 160,
    "deltanet_key": 161,
    "delta_key": 162,  # kernel arg 1
    "delta_value": 164,  # kernel arg 2
    "delta_beta": 166,  # kernel kwarg beta — sigmoid(in_proj_b), per head
    "delta_decay": 168,  # kernel kwarg g — the log-decay, negative reals
    "attention_key_pre_rope": 170,
    # the per-step interior, in loop order: readout, update, state
    "delta_kv_mem": 172,  # (S_{t-1}·exp(g_t) · k̂_t).sum — what the state recalls
    "delta_state_update": 174,  # (v_t − kv_mem_t)·β_t — the diagram's `delta`
    "delta_state": 176,  # S_t, one d_k × d_v matrix per head per step
    # the per-chunk state the nnsight engine reads inside the chunked kernel,
    # before its return (the third typed pair)
    "deltanet_state": 177,
    "delta_kernel_output": 178,  # kernel return[0]: pre-norm, pre-gate
    "attention_value_states": 180,
    # the DeltaNet post-norm, post-gate mixer input — the exact analogue of
    # attention_premix, which is why the name
    "delta_premix": 182,
    # produced with q (one fused projection) and consumed at the very end, at
    # `attn_output * sigmoid(gate)` — ranked where it is produced
    "attention_gate": 190,
    # ...then RoPE rotates q and k, and the attention function runs: scores,
    # softmax, and the weighted sum of values. These four are taps *inside* that
    # function rather than module boundaries — see pytorch_hooks/attention_interface.py.
    "attention_query": 200,
    "attention_key": 210,
    "attention_scores": 220,
    "attention_probs": 230,
    "attention_z": 240,
    # 🔤 `attention_premix` was once `attention_value`. It is the
    # o-projection's INPUT — on a gated family `z · σ(gate)`, on an ungated one
    # `z` — which is the mixer's output just before it is mixed back into the
    # residual stream, and is not the value vectors that name suggested. Those
    # are a separate component (`attention_value_states`), and two components a letter apart in meaning
    # and identical in name is nnterp issue #51's cautionary tale
    # (https://github.com/ndif-team/nnterp/issues/51) happening to us.
    "attention_premix": 300,
    # derived, not computed: the model never forms it, so it sorts where it
    # would be if it did — between the tensor it is a function of and the sum
    # of its own heads
    "attention_result": 350,
    "attention_output": 400,
    # resid_mid is post_attention_layernorm's INPUT and mlp_input_norm its
    # OUTPUT, so the two straddle that one module in this order
    "block_mid": 450,
    "mlp_input_norm": 470,
    "mlp_input": 500,
    # The MoE interior, between the block's input and its output: the router
    # fires first, then the experts, then the combine.
    "router_logits": 510,
    "router_scores": 520,
    "expert_idx": 530,
    # The per-expert interior, ranked where its ops fire inside
    # the fused experts forward: the fused [gate | up] projection's two halves
    # land at 532/534, the activation between them and the down-projection at
    # 536, then — just before the weighted combine — the kernel's inverse
    # permutation (538), which is what `expert_permutation` reads, and the
    # down-projection's (pre-routing-weight) output keeps its reserved 540.
    # The late permutation rank is deliberate: ranks are execution order, and
    # the `.source` interiors refuse out-of-order taps.
    "expert_gate_proj": 532,
    "expert_up_proj": 534,
    "expert_activation": 536,
    "expert_neuron_output": 537,
    "expert_permutation": 538,
    "expert_output": 540,
    "routed_output": 550,
    "mlp_activation": 600,
    "mlp_neuron_output": 605,
    # the shared expert runs beside the routed ones; its gate is *consumed*
    # last, at the multiply that produces the (derived) gated output
    "shared_expert_gate_proj": 610,
    "shared_expert_up_proj": 620,
    "shared_expert_activation": 630,
    "shared_expert_output": 640,
    "shared_expert_gate": 650,
    "mlp_output": 700,
    "block_output": 800,
    "ln_final": 900,
    "lm_head": 1000,
}

#: The rank of a component the table does not know. Deliberately past
#: ``lm_head``: an unranked tap sorts last, so it is treated as the deepest and
#: nothing is elided behind it. Being wrong in the other direction would elide a
#: forward that a later tap still needed.
UNRANKED = 10_000

#: The block index [`site_depth`][] gives the two trunk components past every
#: block (``ln_final``, ``lm_head``). Also what [`ForwardGroup.write_depth`][causalab.neural.shared.plan.ForwardGroup.write_depth]
#: reads when no write of the group lands in a block, so a resume bound can be
#: compared with tap depths in one arithmetic; an engine clamps it to the model.
PAST_BLOCKS = 1_000_000


def site_depth(doc: Document, site_name: str) -> tuple[int, int]:
    """One site's position in the forward pass, as a sortable ``(layer, rank)``.

    The total order the whole vocabulary shares: block depth first, then
    [`COMPONENT_RANK`][] inside the block, with the two layer-less trunk
    components sorting after every block. Three readers depend on it, and on
    nothing finer:

    * group elision (§4) — a forward may stop after its deepest tap;
    * operand reachability (§5.21) — a write's operand may not be read from
      strictly deeper than the address it lands on;
    * resume (§4, `_write_depth`) — the shallowest block a write lands
      in.

    A **band** site (§2.4 ``layers`` with several members) answers with its
    **shallowest** member — the first block the site touches, which is the
    block a write on it first lands in (resume) and the depth a write's
    operand has to be at or above (rule 21's "at or above the write" reads
    the band's min). The deepest member — where a *read* on the band is
    complete, and the tap depth group elision stops after — is the last of
    [`site_depths`][]; the planner never sees a band in either role because
    [`plan_point`][causalab.neural.shared.plan.plan_point] lowers bands to their members first
    ([`lower_bands`][causalab.protocol.lowering.lower_bands]), and rule 21 compares member depths pairwise.

    An unresolved ``layers`` (an unexpanded sweep or artifact wrapper) reads
    as 0 and a non-string ``component`` as the trunk. The first two callers
    run on a *point* document, where every site is concrete, so for them that
    fallback is a type-narrowing convenience and not a semantic; the third
    can be asked about a template and refuses a site it cannot read before
    calling here, since the trunk fallback is the permissive direction for it.
    """
    return site_depths(doc, site_name)[0]


def site_depths(doc: Document, site_name: str) -> tuple[tuple[int, int], ...]:
    """Every member of a site's band as a [`site_depth`][] coordinate, in
    band order — one entry for a one-layer site or a trunk component. Rule 21
    (``validate._check_operand_reachability``) reads a band's members from
    here so a band operand feeding a band write is checked member by member."""
    site = doc.sites[site_name]
    layers = site.layers if isinstance(site.layers, tuple) and site.layers else (0,)
    component = site.component if isinstance(site.component, str) else "lm_head"
    rank = COMPONENT_RANK.get(component, UNRANKED)
    if component in ("ln_final", "lm_head"):
        return ((PAST_BLOCKS, rank),)  # after every block
    return tuple((int(layer), rank) for layer in layers)
