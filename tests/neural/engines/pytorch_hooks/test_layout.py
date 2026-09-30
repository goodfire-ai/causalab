"""Tap-shape conversions: the shape contract, made explicit and reversible.

These are pure-function tests on small tensors — no model, no hooks. What they
pin is that a tap can declare a native shape other than
``(batch, position, feature)`` without any component-specific branching, and
that a declared shape which does not match reality fails loudly instead of
being silently reinterpreted.

The conversions are *computed* from a
[`FeatureShape`][causalab.protocol.registry.shapes.FeatureShape], so these run over every shape
the vocabulary can build rather than over an enumerated list of layout strings:
adding a shape adds a row to ``SHAPES`` and it is covered by all four generic
properties below.
"""

from __future__ import annotations

import pytest
import torch

from causalab.protocol.registry import shapes as sh
from causalab.protocol.registry.shapes import FeatureShape

from causalab.neural.shared.layout import (
    LayoutError,
    from_contract,
    rebuild_payload,
    tap_tensor,
    to_contract,
)

pytestmark = pytest.mark.unit

BATCH, SEQ, FEATURE = 2, 3, 5
HEADS, HEAD_DIM = 4, 2

#: ``name -> (shape, native shape, contract feature width)``. Every shape the
#: constructors build is listed, so that a new descriptor cannot be added
#: without the generic properties covering it.
SHAPES: dict[str, tuple[FeatureShape, tuple[int, ...], int]] = {
    "bsd": (sh.bsd(FEATURE), (BATCH, SEQ, FEATURE), FEATURE),
    "flat_td": (sh.flat_td(FEATURE), (BATCH * SEQ, FEATURE), FEATURE),
    "bds": (sh.bds(FEATURE), (BATCH, FEATURE, SEQ), FEATURE),
    # no native feature axis at all: the contract's width is 1 by definition
    # rather than by choice
    "bs": (sh.bs(integral=True), (BATCH, SEQ), 1),
    "flat_topk": (sh.flat_topk(FEATURE), (BATCH * SEQ, FEATURE), FEATURE),
    "bs_flat_heads": (
        sh.bs_flat_heads(HEADS, HEAD_DIM),
        (BATCH, SEQ, HEADS * HEAD_DIM),
        HEADS * HEAD_DIM,
    ),
    "bshd": (
        sh.bshd(HEADS, HEAD_DIM),
        (BATCH, SEQ, HEADS, HEAD_DIM),
        HEADS * HEAD_DIM,
    ),
    "bhsd": (
        sh.bhsd(HEADS, HEAD_DIM),
        (BATCH, HEADS, SEQ, HEAD_DIM),
        HEADS * HEAD_DIM,
    ),
    "bs_fused_heads": (
        sh.bs_fused_heads(HEADS, 2, 1, HEAD_DIM),
        (BATCH, SEQ, HEADS * 2 * HEAD_DIM),
        HEADS * HEAD_DIM,
    ),
}

#: The shape with no contract form — an attention pattern, whose feature axis is
#: a position axis. Excluded from the generic properties on purpose: there is no
#: contract shape to assert it reaches. It gets its own test below.
PATTERN = sh.attention_pattern(HEADS)


def _native(name: str) -> torch.Tensor:
    _, native_shape, _ = SHAPES[name]
    count = 1
    for dim in native_shape:
        count *= dim
    return torch.arange(count, dtype=torch.float32).reshape(native_shape)


@pytest.mark.parametrize("name", sorted(SHAPES))
def test_to_contract_yields_the_executor_shape(name: str) -> None:
    """Whatever the tap's native shape, the executor sees (batch, pos, feature)."""
    shape, _, width = SHAPES[name]
    got = to_contract(_native(name), shape, batch_size=BATCH)
    assert got.shape == (BATCH, SEQ, width)
    assert width == (shape.width if shape.width is not None else 1)


@pytest.mark.parametrize("name", sorted(SHAPES))
def test_round_trip_restores_the_native_tensor(name: str) -> None:
    """The write path must hand the model back the shape it expected.

    ``from_contract(to_contract(x)) == x`` for every descriptor — the property
    the five hand-written layout branches used to assert one at a time.
    """
    shape, _, _ = SHAPES[name]
    native = _native(name)
    contract = to_contract(native, shape, batch_size=BATCH)
    back = from_contract(contract, shape, batch_size=BATCH, native=native.clone())
    assert back.shape == native.shape
    assert torch.equal(back, native)


@pytest.mark.parametrize("name", sorted(SHAPES))
def test_an_edit_through_the_contract_reaches_the_native_tensor(name: str) -> None:
    """This is the property the write path depends on.

    ``_address_writer`` mutates the contract-shaped tensor in place, then the
    hook converts back. If the conversion copied and the copy were discarded,
    writes would silently no-op — so assert the edit survives the round trip.
    """
    shape, _, width = SHAPES[name]
    native = _native(name)
    contract = to_contract(native, shape, batch_size=BATCH)
    cell = (0, 1, width - 1)
    contract[cell] = -99.0
    back = from_contract(contract, shape, batch_size=BATCH, native=native)
    assert to_contract(back, shape, batch_size=BATCH)[cell] == -99.0


@pytest.mark.parametrize("name", sorted(SHAPES))
def test_the_declared_native_rank_is_the_real_one(name: str) -> None:
    """``native_rank`` is what the rank check refuses against, so pin it against
    the tensors the fixtures actually build."""
    shape, native_shape, _ = SHAPES[name]
    assert shape.native_rank == len(native_shape)


# ------------------------------------------------------------------ #
# the shapes that need more than the generic properties
# ------------------------------------------------------------------ #


def test_a_shape_with_no_contract_form_is_the_identity_both_ways() -> None:
    """An attention pattern passes through untouched, in both directions.

    This is the generated replacement for the ``"native"`` marker: the axes are
    fully described — ``(batch, head, position[query], key_position[key])`` —
    and it is *that description* that says there is no contract to convert to,
    rather than a magic string saying "undescribed".
    """
    probs = torch.zeros(BATCH, HEADS, SEQ, SEQ)
    assert to_contract(probs, PATTERN, batch_size=BATCH) is probs
    assert from_contract(probs, PATTERN, batch_size=BATCH) is probs
    assert not PATTERN.has_contract_form
    assert not PATTERN.is_feature_space
    assert PATTERN.width is None


def test_bsd_does_not_copy() -> None:
    """The default shape must not reshape anything the executor can notice."""
    native = _native("bsd")
    contract = to_contract(native, sh.bsd(FEATURE), batch_size=BATCH)
    assert contract.data_ptr() == native.data_ptr()


def test_flat_positions_are_recovered_in_order() -> None:
    """Row r of a flattened tap is (batch r // seq, position r % seq)."""
    native = _native("flat_td")
    contract = to_contract(native, sh.flat_td(FEATURE), batch_size=BATCH)
    for b in range(BATCH):
        for pos in range(SEQ):
            assert torch.equal(contract[b, pos], native[b * SEQ + pos])


def test_bds_moves_the_feature_axis_last() -> None:
    native = _native("bds")
    contract = to_contract(native, sh.bds(FEATURE), batch_size=BATCH)
    assert torch.equal(contract[:, :, 0], native[:, 0, :])


def test_head_axes_flatten_head_major() -> None:
    """Head ``h``'s block of the contract is ``[h*d, (h+1)*d)`` — the same
    space ``attention_premix``'s ``head`` slice already addressed, which is why
    a per-head tap needs no new sub-axis in the executor."""
    native = _native("bshd")  # (batch, seq, heads, head_dim)
    contract = to_contract(native, sh.bshd(HEADS, HEAD_DIM), batch_size=BATCH)
    for head in range(HEADS):
        block = contract[..., head * HEAD_DIM : (head + 1) * HEAD_DIM]
        assert torch.equal(block, native[:, :, head, :])


def test_bhsd_transposes_before_flattening() -> None:
    """The attention interface hands back ``(batch, heads, pos, head_dim)``; the
    contract's position axis must be the *position* one, not the head one."""
    native = _native("bhsd")
    contract = to_contract(native, sh.bhsd(HEADS, HEAD_DIM), batch_size=BATCH)
    for head in range(HEADS):
        block = contract[..., head * HEAD_DIM : (head + 1) * HEAD_DIM]
        assert torch.equal(block, native[:, head, :, :])


def test_a_fused_tap_reads_only_its_own_split() -> None:
    """``[q_h | gate_h]`` per head: split 1 is the gate, and reading it must not
    pick up any of ``q``."""
    native = _native("bs_fused_heads").reshape(BATCH, SEQ, HEADS, 2, HEAD_DIM)
    shape = sh.bs_fused_heads(HEADS, 2, 1, HEAD_DIM)
    contract = to_contract(native.reshape(BATCH, SEQ, -1), shape, batch_size=BATCH)
    for head in range(HEADS):
        block = contract[..., head * HEAD_DIM : (head + 1) * HEAD_DIM]
        assert torch.equal(block, native[:, :, head, 1, :])


def test_a_fused_write_leaves_the_other_split_alone() -> None:
    """The reason ``from_contract`` takes ``native``: editing the gate must not
    disturb ``q``, which shares the projection's output tensor."""
    shape = sh.bs_fused_heads(HEADS, 2, 1, HEAD_DIM)
    native = _native("bs_fused_heads")
    before = native.clone()
    contract = to_contract(native, shape, batch_size=BATCH)
    contract[:] = -99.0
    back = from_contract(contract, shape, batch_size=BATCH, native=native)
    unpacked, was = (
        back.reshape(BATCH, SEQ, HEADS, 2, HEAD_DIM),
        before.reshape(BATCH, SEQ, HEADS, 2, HEAD_DIM),
    )
    assert torch.equal(
        unpacked[:, :, :, 1, :], torch.full_like(was[:, :, :, 1, :], -99.0)
    )
    assert torch.equal(unpacked[:, :, :, 0, :], was[:, :, :, 0, :])


def test_a_fused_write_without_the_native_tensor_is_refused() -> None:
    """Rather than silently dropping the other split."""
    shape = sh.bs_fused_heads(HEADS, 2, 1, HEAD_DIM)
    contract = torch.zeros(BATCH, SEQ, HEADS * HEAD_DIM)
    with pytest.raises(LayoutError, match="needs the native tensor"):
        from_contract(contract, shape, batch_size=BATCH)


# ------------------------------------------------------------------ #
# a declared shape that contradicts the tensor must fail loudly
# ------------------------------------------------------------------ #


def test_a_flat_tap_rejects_a_batch_size_that_does_not_divide() -> None:
    with pytest.raises(LayoutError, match="multiple of the batch size"):
        to_contract(_native("flat_td"), sh.flat_td(FEATURE), batch_size=4)


def test_a_flat_tap_rejects_a_tensor_of_the_wrong_rank() -> None:
    with pytest.raises(LayoutError, match="expects a 2-D tensor"):
        to_contract(_native("bsd"), sh.flat_td(FEATURE), batch_size=BATCH)


def test_bds_rejects_a_tensor_of_the_wrong_rank() -> None:
    with pytest.raises(LayoutError, match="expects a 3-D tensor"):
        to_contract(_native("flat_td"), sh.bds(FEATURE), batch_size=BATCH)


def test_a_static_width_that_does_not_match_is_refused() -> None:
    """The width is part of the declaration, so a tap pointed at the wrong
    module fails here instead of producing plausible numbers downstream."""
    with pytest.raises(LayoutError, match="declares feature of width"):
        to_contract(_native("bsd"), sh.bsd(FEATURE + 1), batch_size=BATCH)


def test_a_head_packing_that_does_not_multiply_out_is_refused() -> None:
    with pytest.raises(LayoutError, match="packs 4·2 = 8"):
        to_contract(_native("bsd"), sh.bs_flat_heads(HEADS, HEAD_DIM), batch_size=BATCH)


# ------------------------------------------------------------------ #
# tuple payloads
# ------------------------------------------------------------------ #


def test_default_tuple_rule_is_unchanged() -> None:
    """No tuple_index keeps the historical behaviour exactly."""
    t0, t1 = torch.zeros(2), torch.ones(2)
    assert tap_tensor((t0, t1), None) is t0
    assert tap_tensor(t0, None) is t0


def test_tuple_index_addresses_a_later_element() -> None:
    """A router returning (logits, scores, indices) needs element 1 and 2."""
    logits, scores, indices = torch.zeros(2), torch.ones(2), torch.full((2,), 7.0)
    payload = (logits, scores, indices)
    assert tap_tensor(payload, 0) is logits
    assert tap_tensor(payload, 1) is scores
    assert tap_tensor(payload, 2) is indices
    assert tap_tensor(payload, -1) is indices


def test_tuple_index_on_a_bare_tensor_is_refused() -> None:
    """Silently returning the whole tensor would read the wrong thing."""
    with pytest.raises(LayoutError, match="not a tuple"):
        tap_tensor(torch.zeros(2), 1)


def test_tuple_index_out_of_range_is_refused() -> None:
    with pytest.raises(LayoutError, match="2-tuple"):
        tap_tensor((torch.zeros(2), torch.ones(2)), 5)


def test_rebuild_preserves_the_rest_of_the_payload() -> None:
    """A write must not drop the cache or weights a module also returned."""
    hidden, cache = torch.zeros(2), object()
    out = rebuild_payload((hidden, cache), None, torch.ones(2))
    assert isinstance(out, tuple) and len(out) == 2
    assert torch.equal(out[0], torch.ones(2))
    assert out[1] is cache


def test_rebuild_targets_the_declared_element() -> None:
    a, b, c = torch.zeros(2), torch.ones(2), torch.full((2,), 3.0)
    out = rebuild_payload((a, b, c), 1, torch.full((2,), -1.0))
    assert out[0] is a and out[2] is c
    assert torch.equal(out[1], torch.full((2,), -1.0))


def test_rebuild_on_a_bare_payload_returns_the_value() -> None:
    assert torch.equal(
        rebuild_payload(torch.zeros(2), None, torch.ones(2)), torch.ones(2)
    )
