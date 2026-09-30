"""Resolve positions in generated continuations.

``Continuation`` stores one step per generated token, indexed from zero.
Rows stop at their first EOS. ``index``, ``all``, and ``variable`` anchors
refer to these steps. A variable anchor selects its first occurrence and
returns zero positions when absent.

Windows clip at each row's end. Empty generations contribute zero positions,
so continuation reads can have different widths across rows. Prompt-frame
positions keep their own strict bounds checks in the protocol layer.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Mapping

import torch

from causalab.protocol.positions.encoding import variable_value
from causalab.protocol.rules.errors import ProtocolError
from causalab.protocol.schema import PositionSpec, concrete_int, concrete_str

__all__ = ["Continuation", "continuation_frame", "resolve_steps"]


@dataclasses.dataclass(frozen=True)
class Continuation:
    """One batch's greedy continuation: the frame ``generated`` addresses.

    ``token_ids`` is ``(batch, steps)`` as decoded — every row runs the same
    number of steps, because a batched decode has no way not to — and
    ``widths`` says how much of each row is real: the count before its first
    EOS, or every step for a row that never emitted one. Positions resolve
    against ``widths``, so a row that stopped early simply contributes fewer
    of them.

    ``texts`` and ``offsets`` describe the same tokens as characters (per
    row: the decoded continuation, and each token's ``[start, end)`` span
    inside it), which is what a ``variable`` anchor searches. They come from
    incremental detokenization rather than a tokenizer's offset mapping:
    re-encoding ``prompt + continuation`` is not the same token sequence the
    decode produced (merges cross the boundary), so the spans have to be
    built as the tokens arrive.
    """

    token_ids: torch.Tensor
    widths: tuple[int, ...]
    texts: tuple[str, ...] = ()
    offsets: tuple[tuple[tuple[int, int], ...], ...] = ()

    @property
    def steps(self) -> int:
        """How many steps the decode ran — the same for every row."""
        return int(self.token_ids.shape[1])

    def real_ids(self, row: int) -> list[int]:
        """``row``'s generated ids up to its first EOS."""
        return [int(t) for t in self.token_ids[row, : self.widths[row]]]


def _generated_variable_run(
    continuation: Continuation, row: int, value: str
) -> list[int]:
    """Step indices covering the **first** occurrence of ``value`` in the
    row's generated text, or ``[]`` when the model never said it.

    Char spans come from the decode's incremental detokenization
    ([`Continuation`][]), so a match that starts mid-piece still lands on
    the steps that produced it — the sentencepiece case a post-hoc
    ``offset_mapping`` cannot resolve.
    """
    if row >= len(continuation.texts):
        return []
    text = continuation.texts[row]
    start = text.find(value)
    if start < 0:
        return []
    lo, hi = start, start + len(value)
    width = continuation.widths[row]
    return [
        step
        for step, (a, b) in enumerate(continuation.offsets[row][:width])
        if a < hi and b > lo
    ]


def resolve_steps(
    spec: PositionSpec,
    continuation: Continuation,
    row: int,
    *,
    dataset_row: Mapping[str, Any] | None = None,
    field: str | None = None,
) -> list[int]:
    """Resolve one ``generated`` spec for one row into **decode-step** indices.

    Indices are 0-based into the decode, bounded by the row's real width —
    see the module docstring on why a window past a row's end clips and a
    row that generated nothing yields nothing.
    """
    width = continuation.widths[row]
    if width == 0:
        return []
    if spec.all is not None:
        return list(range(width))
    if spec.index is not None:
        n = concrete_int(spec.index, "position index")
        step = width + n if n < 0 else n
        return [step] if 0 <= step < width else []
    if spec.span is not None:
        span = spec.span
        if not isinstance(span, tuple) or len(span) != 2:
            raise ProtocolError("P2", f"span is not concrete: {span!r}")
        a, b = (int(v) for v in span)
        return list(range(min(a, width), min(b, width)))
    if spec.variable is not None:
        if dataset_row is None or field is None:
            raise ProtocolError(
                "P2",
                "a generated 'variable' position needs its dataset row — the "
                "value the model may have said comes from the table",
            )
        variable = concrete_str(spec.variable, "position variable")
        value = variable_value(dataset_row, field, variable)
        return _generated_variable_run(continuation, row, value)
    raise ProtocolError(
        "P2",
        f"anchor {spec!r} has no continuation-frame resolution — v1 addresses "
        "generated tokens by index, span, variable or all",
    )


def continuation_frame(
    tokenizer: Any, generated: torch.Tensor, widths: tuple[int, ...]
) -> Continuation:
    """Build the frame the decode produced, characters included.

    Token spans come from incremental detokenization — decode the row's
    first ``k`` tokens, then ``k + 1``, and the growth is token ``k``'s
    span. Re-encoding the finished text would not do: a tokenizer is free
    to merge across a boundary the decode never saw, and the spans have to
    describe the tokens the model actually emitted.
    """
    texts: list[str] = []
    offsets: list[tuple[tuple[int, int], ...]] = []
    for row, width in enumerate(widths):
        ids = [int(t) for t in generated[row, :width]]
        spans: list[tuple[int, int]] = []
        text = ""
        for k in range(width):
            grown = tokenizer.decode(
                ids[: k + 1],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            spans.append((len(text), len(grown)))
            text = grown
        texts.append(text)
        offsets.append(tuple(spans))
    return Continuation(
        token_ids=generated.detach().cpu(),
        widths=widths,
        texts=tuple(texts),
        offsets=tuple(offsets),
    )
