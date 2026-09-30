"""Reduce read values to per-example metrics.

Vocabulary metrics gather logits using dataset columns. ``top_k`` can rank
any read axis, including residual or SAE features. Its ``by`` field defines
the ranking; ``vocab_axis`` controls token decoding.

Every answer a metric names resolves to token ids through
[`metric_token_ids`][causalab.protocol.answers.metric_token_ids], the function
the run door's pass before the weights load also calls
(``pipeline.resolve_answers``): strings as written (§2.10), integer ids under
``token_form: "id"``. Multi-token answers require an explicit supported
scoring mode. Results are plain floats or small structures ready for metric
tables.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import torch

from causalab.protocol.answers import excluded_rows, metric_token_ids, names_answers
from causalab.protocol.rules.errors import ProtocolError
from causalab.protocol.schema import (
    VOCAB_TOP_K_RANKING,
    WHOLE_WINDOW_METRIC_KINDS,
    AggregationSpec,
)

__all__ = [
    "compute_metric",
    "compute_windowed_metric",
    "DEVICE_SCORED_KINDS",
    "GATHERED_KINDS",
    "gathered_metric",
    "js_divergence",
    "matched_metric",
    "score_metric",
]


def _last_pos_rows(value: torch.Tensor) -> torch.Tensor:
    """A read at one position arrives as (batch, 1, width); squeeze it.

    ``width`` is the vocabulary for an ``lm_head`` read and the site's own
    width otherwise — only ``top_k`` reduces the latter (every other kind is
    bound to a vocabulary projection by validation)."""
    if value.dim() == 3:
        if value.shape[1] != 1:
            raise ProtocolError(
                "P2",
                f"metric read spans {value.shape[1]} positions — metrics reduce "
                "one position per example",
            )
        return value[:, 0, :]
    return value


def _top_k(
    metric: AggregationSpec,
    dense: torch.Tensor,
    tokenizer: Any,
    *,
    vocab_axis: bool,
) -> list[dict[str, Any]]:
    """``top_k`` over one read's rows — the reduction that happens **where the
    rows are gathered**, so a 100k-latent SAE code never reaches disk.

    ``by`` (mandatory, §2.10) is the ranking rule, and it is the author's call
    because only the author knows what the axis is: a vocabulary projection
    has no meaningful negative entries, a residual stream and a signed feature
    code do.

    The emitted columns have **fixed identities**, so a column never means one
    thing in one document and another in the next; a column is absent rather
    than reinterpreted:

    ==========  =====================================  ====================
    column      meaning                                emitted when
    ==========  =====================================  ====================
    ``indices`` index along the read's last axis       always
    ``tokens``  that index decoded as a token string   the read is a plain lm_head tap
    ``values``  the **raw** read value at that index   always
    ``probs``   softmax probability over the vocab     ``by == "prob"``
    ==========  =====================================  ====================

    ``values`` is always the raw value — a logit under ``by: "prob"``, not the
    probability — so a downstream reader never has to know the ranking rule to
    know what it is holding. The normalized number lives in its own column.
    """
    k = metric.fields["k"]
    assert isinstance(k, int)  # parse guarantees the shape
    by = str(metric.fields["by"])
    width = int(dense.shape[-1])
    if k < 1 or k > width:
        raise ProtocolError(
            "P2",
            f"top_k asks for k={k} of a read {width} wide — k must be in [1, width]",
        )
    if by == VOCAB_TOP_K_RANKING:
        # validation binds `prob` to an lm_head read (§2.10): a softmax across
        # neurons or latents normalizes over an axis that is not an event space
        scores = torch.softmax(dense, dim=-1)
    elif by == "abs_value":
        scores = dense.abs()
    else:
        scores = dense
    top = scores.topk(k, dim=-1)
    out: list[dict[str, Any]] = []
    for i in range(dense.shape[0]):
        indices = [int(j) for j in top.indices[i]]
        entry: dict[str, Any] = {"indices": indices}
        if vocab_axis:
            entry["tokens"] = [tokenizer.decode([j]) for j in indices]
        entry["values"] = [float(dense[i, j]) for j in indices]
        if by == VOCAB_TOP_K_RANKING:
            entry["probs"] = [float(p) for p in top.values[i]]
        out.append(entry)
    return out


def js_divergence(
    of_logits: torch.Tensor,
    target_logits: torch.Tensor,
    restrict_ids: Sequence[Sequence[int]] | None = None,
) -> torch.Tensor:
    """Per-row Jensen–Shannon divergence, in nats, between the distributions
    two ``(batch, vocab)`` logit tensors define (§2.10 ``js``)::

        JS(p, q) = ½ KL(p ‖ m) + ½ KL(q ‖ m),   m = ½ (p + q)

    Symmetric, and bounded by ``ln 2``. With ``restrict_ids`` each row's two
    distributions are first **restricted to its answer ids and renormalised**
    — a ``log_softmax`` over the sliced logits, which is exact and needs no
    ``eps``. Differentiable in both arguments, so the same function is the
    objective term (``pytorch_hooks.train.metric_tensor``) and the saved
    record — one arithmetic, one unit."""
    if restrict_ids is None:
        return _js_from_log_probs(
            torch.log_softmax(of_logits, dim=-1),
            torch.log_softmax(target_logits, dim=-1),
        )
    values: list[torch.Tensor] = []
    for i, ids in enumerate(restrict_ids):
        index = torch.as_tensor(list(ids), dtype=torch.long, device=of_logits.device)
        p = torch.log_softmax(of_logits[i].index_select(-1, index), dim=-1)
        q = torch.log_softmax(target_logits[i].index_select(-1, index), dim=-1)
        values.append(_js_from_log_probs(p, q))
    return torch.stack(values)


def _js_from_log_probs(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """JS over the last axis of two log-probability tensors; ``m`` is formed in
    log space (``logaddexp − ln 2``) so a near-zero probability never
    underflows the log."""
    m = torch.logaddexp(p, q) - math.log(2.0)
    return 0.5 * (p.exp() * (p - m)).sum(dim=-1) + 0.5 * (q.exp() * (q - m)).sum(dim=-1)


#: The kinds whose per-example value **selects** entries of the projection at
#: the answer token ids and computes on nothing else. For these, gathering
#: the entries where the value sits — the device — and reducing the gathered
#: values on the CPU in float is [`compute_metric`][]'s arithmetic to the
#: bit: an upcast commutes with a selection, and the per-example operation is
#: the same 0-d float op. A kind with a softmax or a log-sum-exp over the
#: vocabulary is not in the set — reduced on the device it rounds differently
#: — and keeps the whole-vocabulary CPU path. ``match`` is not gathered (its
#: argmax ranges over the vocabulary) but is scored on the device all the
#: same: [`matched_metric`][], [`DEVICE_SCORED_KINDS`][].
GATHERED_KINDS = frozenset({"logit_diff", "soft_accuracy", "token_logit"})

#: The kinds [`score_metric`][] reduces where the read's value sits, copying
#: a few columns — or the argmax indices — to the host rather than the
#: vocabulary: the gathered kinds, and ``match``. An argmax is a selection,
#: not a rounding reduction: the fp32 upcast is exact and monotone, so the
#: maximal entry is the same on either device, and both ``torch.argmax``
#: kernels return the *first* maximal index on a tie (CPU: a strict-greater
#: update; CUDA: ``ArgMaxOps`` prefers the smaller index at equal values). A
#: read consumed only by these kinds may stay on the device
#: (``executor.base.device_scored_reads``).
DEVICE_SCORED_KINDS = GATHERED_KINDS | {"match"}

_GATHERED_FIELDS: Mapping[str, tuple[str, ...]] = {
    "logit_diff": ("a", "b"),
    "soft_accuracy": ("a", "b"),
    "token_logit": ("token",),
}


def gathered_metric(
    metric: AggregationSpec,
    of_value: torch.Tensor,
    rows: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    *,
    token_ids: Mapping[str, Sequence[int]] | None = None,
    denominator_key: str | None = None,
) -> list[Any]:
    """[`compute_metric`][] for a [`GATHERED_KINDS`][] kind over a value
    wherever it sits: the excluded rows as [`compute_metric`][] excludes
    them, the answer entries gathered from the read on its own device, the
    one or two columns copied to the CPU, and the kind's per-example
    arithmetic in float exactly as [`compute_metric`][] writes it — so the
    list is the same, entry for entry, without the vocabulary ever leaving
    the device. ``token_ids`` is
    [`metric_token_ids`][causalab.protocol.answers.metric_token_ids]' result when the
    caller resolved it once. No ``vocab_axis``: validation holds a gathered
    kind's input read to a plain ``lm_head`` tap (``read_is_vocabulary``
    — no featurizer, no ``dims``; ``protocol/validate.py`` exempts only
    ``kl`` and ``top_k``), so the value's last axis is the vocabulary and
    a raw token id indexes it."""
    kind = str(metric.kind)
    if kind not in GATHERED_KINDS:
        raise ValueError(f"metric kind {kind!r} is not gathered at token ids")
    excluded = excluded_rows(metric, rows, denominator_key or kind)
    keep = [i for i in range(len(rows)) if i not in excluded]
    if not keep:  # every row excluded: nothing to reduce, nothing raised
        return [excluded[i] for i in range(len(rows))]
    ids = (
        token_ids
        if token_ids is not None
        else metric_token_ids(metric, rows, tokenizer)
    )
    dense = _last_pos_rows(of_value)
    if len(keep) != len(rows):
        dense = dense[torch.tensor(keep, dtype=torch.long, device=dense.device)]
    fields = _GATHERED_FIELDS[kind]
    for field in fields:
        if len(ids[field]) != len(keep):
            raise ValueError(
                f"metric {kind}.{field}: {len(ids[field])} token ids for "
                f"{len(keep)} rows"
            )
    with torch.no_grad():
        # every answer column in one gather and one copy — one host wait per
        # metric, not per column — then the CPU path's upcast: the fp32 value
        # `compute_metric` reads at `logits[i, id]`, entry for entry
        index = torch.tensor(
            [list(ids[field]) for field in fields],
            dtype=torch.long,
            device=dense.device,
        )  # (fields, keep)
        # (keep, fields) off the device contiguous, then split by column
        taken = dense.gather(1, index.t()).cpu().t().float()  # (fields, keep)
    columns = {field: taken[i] for i, field in enumerate(fields)}
    if kind == "logit_diff":
        a, b = columns["a"], columns["b"]
        scored = [float(a[i] - b[i]) for i in range(len(keep))]
    elif kind == "soft_accuracy":
        # `_compute_metric`'s line, `.float()` included (already fp32 here)
        a, b = columns["a"], columns["b"]
        scored = [
            float(torch.sigmoid(a[i].float() - b[i].float())) for i in range(len(keep))
        ]
    else:
        token = columns["token"]
        scored = [float(token[i]) for i in range(len(keep))]
    it = iter(scored)
    return [excluded[i] if i in excluded else next(it) for i in range(len(rows))]


def _match_scores(argmax: torch.Tensor, resolved: Sequence[set[int]]) -> list[float]:
    """``1.0`` where the row's argmax is one of its credited ids."""
    return [float(int(argmax[i]) in ids) for i, ids in enumerate(resolved)]


def matched_metric(
    metric: AggregationSpec,
    of_value: torch.Tensor,
    rows: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    *,
    denominator_key: str | None = None,
) -> list[Any]:
    """[`compute_metric`][] for ``match`` over a value wherever it sits:
    the excluded rows as [`compute_metric`][] excludes them, the ids
    resolved as it resolves them, and the argmax taken on the read's own
    device over the same fp32 upcast — one ``(rows,)`` index vector copied
    to the host, never the vocabulary. The same list, entry for entry
    ([`DEVICE_SCORED_KINDS`][] says why the argmax may move)."""
    if str(metric.kind) != "match":
        raise ValueError(f"metric kind {metric.kind!r} is not 'match'")
    excluded = excluded_rows(metric, rows, denominator_key or "match")
    keep = [i for i in range(len(rows)) if i not in excluded]
    if not keep:  # every row excluded: nothing to reduce, nothing raised
        return [excluded[i] for i in range(len(rows))]
    dense = _last_pos_rows(of_value)
    if len(keep) != len(rows):
        dense = dense[torch.tensor(keep, dtype=torch.long, device=dense.device)]
    resolved = metric_token_ids(metric, [rows[i] for i in keep], tokenizer)["expected"]
    with torch.no_grad():
        argmax = dense.float().argmax(dim=-1).cpu()
    it = iter(_match_scores(argmax, resolved))
    return [excluded[i] if i in excluded else next(it) for i in range(len(rows))]


def score_metric(
    metric: AggregationSpec,
    of_value: torch.Tensor,
    rows: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    *,
    target_value: torch.Tensor | None = None,
    vocab_axis: bool = True,
    denominator_key: str | None = None,
) -> list[Any]:
    """[`compute_metric`][] over a read's value wherever it sits.

    A value on an accelerator is reduced there when its kind allows
    ([`DEVICE_SCORED_KINDS`][]: [`gathered_metric`][],
    [`matched_metric`][]) and copied to the host whole otherwise — the
    one host copy [`compute_metric`][] always made, taken here instead of
    in the read's finalization. A CPU value is [`compute_metric`][]
    itself. The list is the same either way, entry for entry."""
    kind = str(metric.kind)
    if of_value.device.type != "cpu" and kind in DEVICE_SCORED_KINDS:
        # `target_value` and `vocab_axis` are not consulted on this branch:
        # `_compute_metric` reads `target_value` for `kl`/`js` and
        # `vocab_axis` for `top_k`/`class_probs` only, none of them device-
        # scored kinds
        if kind in GATHERED_KINDS:
            return gathered_metric(
                metric, of_value, rows, tokenizer, denominator_key=denominator_key
            )
        return matched_metric(
            metric, of_value, rows, tokenizer, denominator_key=denominator_key
        )
    return compute_metric(
        metric,
        of_value.cpu(),
        rows,
        tokenizer,
        target_value=None if target_value is None else target_value.cpu(),
        vocab_axis=vocab_axis,
        denominator_key=denominator_key,
    )


def compute_metric(
    metric: AggregationSpec,
    of_value: torch.Tensor,
    rows: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    *,
    target_value: torch.Tensor | None = None,
    vocab_axis: bool = True,
    denominator_key: str | None = None,
) -> list[Any]:
    """One metric over one read's value, per example.

    ``vocab_axis`` says whether the read's last axis is the vocabulary — i.e.
    whether it is a plain ``lm_head`` tap, with no featurizer or ``dims``
    taking the value out of token-id space ([`read_is_vocabulary`][causalab.protocol.schema.types.read_is_vocabulary]). Every kind but ``top_k`` is bound to a
    vocabulary projection by validation, so the default is ``True``; ``top_k``
    is the one kind that also runs over a residual stream, an MLP activation
    or a featurizer's latents, and it needs to know because a token id is
    worth decoding and a neuron index is not.

    A row the table carries no answer for
    ([`excluded_rows`][causalab.protocol.answers.excluded_rows]) comes back
    as the typed [`Unavailable`][causalab.protocol.results.Unavailable] in its
    place — an excluded measurement, keyed under ``denominator_key`` (the
    metric's cell key; its kind when the caller has no coordinates) — and
    the kind is computed over the other rows only, so no excluded row ever
    reaches a mean (§2.10 "Eligibility").
    """
    excluded = excluded_rows(metric, rows, denominator_key or str(metric.kind))
    if excluded:
        keep = [i for i in range(len(rows)) if i not in excluded]
        if not keep:  # every row excluded: nothing to reduce, nothing raised
            return [excluded[i] for i in range(len(rows))]
        index = torch.tensor(keep, dtype=torch.long, device=of_value.device)
        scored = _compute_metric(
            metric,
            _last_pos_rows(of_value)[index],
            [rows[i] for i in keep],
            tokenizer,
            target_value=(
                _last_pos_rows(target_value)[index]
                if target_value is not None
                else None
            ),
            vocab_axis=vocab_axis,
        )
        it = iter(scored)
        return [excluded[i] if i in excluded else next(it) for i in range(len(rows))]
    return _compute_metric(
        metric,
        of_value,
        rows,
        tokenizer,
        target_value=target_value,
        vocab_axis=vocab_axis,
    )


def _compute_metric(
    metric: AggregationSpec,
    of_value: torch.Tensor,
    rows: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    *,
    target_value: torch.Tensor | None,
    vocab_axis: bool,
) -> list[Any]:
    """The kinds themselves, over rows that all carry their answers."""
    # `dense` is the read's value at the addressed position, (batch, width).
    # Every kind but `top_k` is bound to an lm_head read, so for those it is
    # the vocabulary projection and reads as `logits` below.
    dense = _last_pos_rows(of_value).float()
    logits = dense
    kind = str(metric.kind)
    # §2.10: strings are tokenized as written; `id` columns hold the ids. The
    # resolution is the run door's own (`protocol/answers.py`), over rows
    # that all carry answers

    def answers() -> dict[str, list[Any]]:
        return metric_token_ids(metric, rows, tokenizer)

    if kind == "logit_diff":
        ids = answers()
        return [
            float(logits[i, a] - logits[i, b])
            for i, (a, b) in enumerate(zip(ids["a"], ids["b"]))
        ]
    if kind == "soft_accuracy":
        # σ of the same margin: the differentiable stand-in for "a beats b",
        # bounded so a runaway margin on
        # one row cannot dominate a mean the way a raw logit_diff can. Computed
        # in float so the saved value and the objective twin agree to the bit.
        ids = answers()
        return [
            float(torch.sigmoid(logits[i, a].float() - logits[i, b].float()))
            for i, (a, b) in enumerate(zip(ids["a"], ids["b"]))
        ]
    if kind == "token_logit":
        return [float(logits[i, t]) for i, t in enumerate(answers()["token"])]
    if kind == "cross_entropy":
        targets = answers()["target"]
        log_probs = torch.log_softmax(logits, dim=-1)
        return [float(-log_probs[i, t]) for i, t in enumerate(targets)]
    if kind == "kl":
        if target_value is None:
            raise ProtocolError("P2", "kl needs its target read's value")
        p = torch.log_softmax(logits, dim=-1)
        q = torch.log_softmax(_last_pos_rows(target_value).float(), dim=-1)
        kl = (p.exp() * (p - q)).sum(dim=-1)
        return [float(v) for v in kl]
    if kind == "js":
        if target_value is None:
            raise ProtocolError("P2", "js needs its target read's value")
        values = js_divergence(
            logits,
            _last_pos_rows(target_value).float(),
            answers().get("restrict"),
        )
        return [float(v) for v in values]
    if kind == "match":
        return _match_scores(logits.argmax(dim=-1), answers()["expected"])
    if kind == "top_k":
        return _top_k(metric, dense, tokenizer, vocab_axis=vocab_axis)
    if kind == "class_probs":
        group_ids = {
            field.removeprefix("groups."): ids for field, ids in answers().items()
        }
        probs = torch.softmax(logits, dim=-1)
        return [
            {name: float(probs[i, ids].sum()) for name, ids in group_ids.items()}
            for i in range(logits.shape[0])
        ]
    if kind == "token_logits":
        # The result mirrors `top_k`'s fixed column identities — `indices` are
        # the ids, `tokens` the ids decoded, `values` the raw logits — so a
        # reader of either table holds the same three things under the same
        # names.
        ids = answers()["tokens"]
        decoded = [tokenizer.decode([t]) for t in ids]
        return [
            {
                "indices": ids,
                "tokens": decoded,
                "values": [float(logits[i, t]) for t in ids],
            }
            for i in range(logits.shape[0])
        ]
    if kind == "decode":
        raise ProtocolError(
            "P2",
            "'decode' reduces the tokens a decode produced, so it binds to a "
            "read in the continuation frame (§2.3) — validation refuses it "
            "anywhere else",
        )
    raise ProtocolError("P4", f"unknown metric kind {kind!r}")


def compute_windowed_metric(
    metric: AggregationSpec,
    windows: Sequence[torch.Tensor],
    rows: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    *,
    target_windows: Sequence[torch.Tensor] | None = None,
    generated_ids: Sequence[Sequence[int]] | None = None,
    vocab_axis: bool = True,
) -> list[list[Any]]:
    """One metric over a read that addresses **several** positions per row.

    ``windows[i]`` is example ``i``'s value at the positions it addresses,
    ``(positions_i, vocab)`` — empty when the row addressed none, which in
    the continuation frame is a result (§2.3), not a misalignment. Returns
    the same shape: one list of values per example.

    Every ``distribution`` kind reduces **per position**, and does so
    through [`compute_metric`][] on the flattened positions — one
    implementation of the kinds, not two. ``ids`` kinds never look at
    ``windows`` at all: they consume ``generated_ids``, which is why a text
    probe obliges no vocabulary projection (§8).
    """
    kind = str(metric.kind)
    if kind in WHOLE_WINDOW_METRIC_KINDS:
        if generated_ids is None:
            raise ProtocolError(
                "P2", f"metric kind {kind!r} needs the decode's token ids"
            )
        if kind == "decode":
            return [
                [
                    tokenizer.decode(
                        list(ids),
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    )
                ]
                if len(ids)
                else []
                for ids in generated_ids
            ]
        raise ProtocolError("P4", f"unhandled whole-window metric kind {kind!r}")

    counts = [int(window.shape[0]) for window in windows]
    if not any(counts):
        return [[] for _ in windows]
    if names_answers(metric):
        # the answers of each row that addresses a position, once per row:
        # the flattened rows below repeat a row once per position, so a
        # refusal raised over them would count positions, not rows. The
        # pass over the flattened rows then resolves from the encode memo
        metric_token_ids(
            metric, [row for row, count in zip(rows, counts) if count], tokenizer
        )
    flat = torch.cat([w for w in windows if w.shape[0]], dim=0)
    flat_rows = [rows[i] for i, count in enumerate(counts) for _ in range(count)]
    flat_target = None
    if target_windows is not None:
        target_counts = [int(w.shape[0]) for w in target_windows]
        if target_counts != counts:
            raise ProtocolError(
                "P2",
                f"{kind} compares reads addressing different position counts "
                f"({counts} vs {target_counts}) — a comparison needs a "
                "position-for-position pairing",
            )
        flat_target = torch.cat([w for w in target_windows if w.shape[0]], dim=0)
    values = compute_metric(
        metric,
        flat,
        flat_rows,
        tokenizer,
        target_value=flat_target,
        vocab_axis=vocab_axis,
    )
    out: list[list[Any]] = []
    cursor = 0
    for count in counts:
        out.append(values[cursor : cursor + count])
        cursor += count
    return out
