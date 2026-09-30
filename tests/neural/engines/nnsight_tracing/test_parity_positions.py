"""Positions and frames through both engines (group F).

Position resolution is shared above ``_run_group``
(``causalab/protocol/positions/{encoding,spans,framing,alignment,ledger}.py``
and the executor base in ``causalab/neural/shared/executor/base.py``), so
every selector must gather the *same* tokens on both engines and every
position refusal must be the *same* words. The tests here drive the reference
tests' own documents (imported, not copied) through
``tests._helpers.engines.both_executors`` and, where a row is an
engine-level claim (the ledger save, the path-patching receipt), through
``tests._helpers.engines.run_both``.

Cases, by id (each test names the one it covers):

* **F1** ``index >= 0`` counts from the content start, past the left pad;
* **F2** the anchor vocabulary (``variable`` / ``column`` / ``span`` /
  ``scope`` / ``relative_to`` / ``segment`` / ``all`` / ``indices``) and the
  two unalignable cardinalities as *cells* (``alignment_missing`` /
  ``alignment_ambiguous``) on a read;
* **F3** the span algebra (``union`` / ``intersection`` / ``between`` /
  ``before`` / ``after``), ``atomic`` addresses, and rule 27's edit-group
  atomicity (P21's ``check_edit_groups`` leg);
* **F4** a declared ``alignment`` under each pairing cardinality, and the
  identical refusal when the declaration contradicts the rows (P21's
  declared-alignment leg);
* **F5** ``segments.frame: chat``. Both fixtures ship a template, so the
  positive case runs on both. The ``chat_template_missing`` refusal is
  produced by removing the template from the tokenizer object;
* **F6** the ``location_ledger`` save, row for row;
* **F7** path patching in the hand form: the corpus document and the two
  reference documents of ``test_path_patching_run.py`` (joint receivers, two
  freeze sets). Logits, tables and receipts are compared, with the known
  ``fires`` asymmetry asserted as the current state;
* **F8** a ``head``-sliced write on ``attention_premix`` and ``attention_z``;
* **P2** (read half) ``lm_head`` at ``pos: "all"``;
* **P14** the out-of-bounds refusal on a module boundary and on the DeltaNet
  interior.

The two P21 legs are the executor's pre-forward checks
(``ExecutorBase.check_edit_groups`` and the width and alignment checks in
``causalab/neural/shared/executor/base.py``). Both engines inherit them, so
the refusal text must match byte for byte.

Anti-vacuity on every write: each engine's patched value differs from its own
unpatched one, so "both agree" cannot be satisfied by "neither landed".
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch

from causalab.neural.engines.nnsight_tracing.executor import TracePointExecutor
from causalab.neural.engines.pytorch_hooks.executor import PointExecutor
from causalab.io.env import read_safetensors_metadata
from causalab.neural.shared.values import RaggedValue
from causalab.protocol import RUN_RECORD_NAME
from causalab.protocol.positions.ledger import LEDGER_COLUMNS
from causalab.protocol.receipt import FIRES_KEY
from causalab.protocol.results import Unavailable
from causalab.protocol.rules.errors import ProtocolError
from causalab.protocol.schema import PROTOCOL_VERSION

from tests._helpers import engines
from tests._helpers import a3b_sweep as sweep
from tests.neural.engines.nnsight_tracing.conftest import TINY_LLAMA
from tests.neural.engines.nnsight_tracing.test_parity_module_boundaries import (
    BASE_TEXTS,
    CF_TEXTS,
)
from tests.neural.engines.pytorch_hooks import test_alignment_run as alignment_ref
from tests.neural.engines.pytorch_hooks import test_edit_groups_run as edit_groups_ref
from tests.neural.engines.pytorch_hooks import test_location_ledger as ledger_ref
from tests.neural.engines.pytorch_hooks import test_path_patching_run as paths_ref
from tests.protocol._docs import UNWRITTEN, in_order, saved
from tests.protocol._env import CORPUS_DIR

pytestmark = pytest.mark.smoke

ATOL = engines.ATOL

#: The weekdays shape (`test_positions_frame.py`, `test_ragged_writes.py`):
#: two prompts of different token length whose entity is one piece on one row
#: and several on the other for a sentencepiece tokenizer.
DAY_TEXTS = ["If today is Thursday, tomorrow is", "If today is Friday, tomorrow is"]
DAY_ENTITY = ["Thursday", "Friday"]
DAY_COLUMNS = {"entity": DAY_ENTITY, "tail": ["tomorrow is", "tomorrow is"]}

FIXTURES = [("hooks_llama", "trace_llama"), ("hooks_qwen", "trace_qwen")]
FIXTURE_IDS = ["llama", "qwen"]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _bundles(request: pytest.FixtureRequest, hooks_name: str, trace_name: str):
    return request.getfixturevalue(hooks_name), request.getfixturevalue(trace_name)


def _same(a: Any, b: Any, what: str) -> None:
    """Shape-then-value agreement, ragged values by widths then flat."""
    if isinstance(a, RaggedValue) or isinstance(b, RaggedValue):
        assert isinstance(a, RaggedValue) and isinstance(b, RaggedValue), (
            f"{what}: {type(a).__name__} on pytorch_hooks, {type(b).__name__} on nnsight"
        )
        assert a.widths == b.widths, f"{what}: widths {a.widths} != {b.widths}"
        a, b = a.flat, b.flat
    if a.numel() == 0 or b.numel() == 0:
        # an empty gather (every row addressed nothing) has no values to
        # compare; the shape is the whole claim
        assert a.shape == b.shape, f"{what}: {tuple(a.shape)} != {tuple(b.shape)}"
        return
    sweep.assert_same(a, b, what)


def _read_both(doc: dict[str, Any], hooks: Any, trace: Any, name: str, **kw: Any):
    hooked, traced = engines.both_executors(doc, hooks, trace, **kw)
    return hooked, traced, hooked.read_value(name), traced.read_value(name)


def _refusal(doc: dict[str, Any], hooks: Any, trace: Any, **kw: Any) -> str:
    """The refusal both executors raise on ``doc`` — asserted byte-identical,
    returned for the test to name what it must say."""
    texts: list[str] = []
    for cls, bundle in ((PointExecutor, hooks), (TracePointExecutor, trace)):
        with pytest.raises(ProtocolError) as err:
            engines.executor_for(cls, doc, bundle, **kw).run_all()
        texts.append(str(err.value))
    assert texts[0] == texts[1], f"refusals differ:\n{texts[0]}\n{texts[1]}"
    return texts[0]


def _unavailable(executor: Any, name: str) -> Unavailable:
    cell = executor.resolution(name)
    assert isinstance(cell, Unavailable), cell
    return cell


def _positions_doc(
    positions: dict[str, Any],
    reads: dict[str, str],
    *,
    segments: dict[str, Any] | None = None,
    counterfactual: bool = False,
) -> dict[str, Any]:
    """`test_location_ledger._doc`: one `block_output` layer-0 site, a read per
    named position, saved."""
    return ledger_ref._doc(  # pyright: ignore[reportPrivateUsage]
        positions, reads, segments=segments, counterfactual=counterfactual
    )


# --------------------------------------------------------------------------- #
# F1 — index >= 0 is the content frame, past the pad
# --------------------------------------------------------------------------- #

#: Rows of different token length (`test_positions_frame.py`), so the shorter
#: one is left-padded and a wrong frame would address the pad.
PADDED_TEXTS = ["one two three", "a much longer sentence right here"]


@pytest.mark.parametrize("hooks_name,trace_name", FIXTURES, ids=FIXTURE_IDS)
@pytest.mark.parametrize("index", [0, 2])
def test_f1_a_nonnegative_index_counts_from_the_content_start(
    request, hooks_name, trace_name, index
):
    """F1. `pos: 0` and `pos: 2` on rows of different length agree, and on
    the row that is padded they address a real token: the read at `pos: 0`
    is the same vector on both rows on a BOS-adding tokenizer (the first
    content token attends to itself alone, whatever the pad in front of
    it), which a frame that counted from the padded start would break."""
    hooks, trace = _bundles(request, hooks_name, trace_name)
    doc = sweep.read_doc("block_output", 1, pos=index)
    hooked, traced, a, b = _read_both(doc, hooks, trace, "r", base_texts=PADDED_TEXTS)
    _same(a, b, f"block_output at pos {index}")
    assert a.shape[:2] == (2, 1)
    batch = hooked.frame("base")
    assert batch.content_start(0) > batch.content_start(1)  # row 0 is padded
    if index == 0 and hooks.tokenizer.bos_token_id is not None:
        for label, value in (("pytorch_hooks", a), ("nnsight", b)):
            assert torch.allclose(value[0], value[1], atol=ATOL), (
                f"{label}: pos 0 is not the same BOS vector on both rows — the "
                "padded row's index is not counted from its content start"
            )


def test_f1_pos_0_and_pos_2_are_different_tokens(hooks_llama, trace_llama):
    """Anti-vacuity for F1: the two indices address different tokens."""
    for cls, bundle in (
        (PointExecutor, hooks_llama),
        (TracePointExecutor, trace_llama),
    ):
        first = engines.executor_for(
            cls,
            sweep.read_doc("block_output", 1, pos=0),
            bundle,
            base_texts=PADDED_TEXTS,
        ).read_value("r")
        third = engines.executor_for(
            cls,
            sweep.read_doc("block_output", 1, pos=2),
            bundle,
            base_texts=PADDED_TEXTS,
        ).read_value("r")
        assert not torch.allclose(first, third, atol=ATOL), cls.__name__


# --------------------------------------------------------------------------- #
# F2 — the anchor vocabulary, and the unalignable cells
# --------------------------------------------------------------------------- #

#: (selector, the authored position) — `test_positions_frame.py`'s forms as
#: document positions; the `segment` rows declare the entity column as one.
SELECTORS: dict[str, dict[str, Any]] = {
    "index": {"index": -1},
    "variable": {"variable": "entity"},
    "column": {"column": "tail"},
    "span": {"span": [1, 3]},
    "scope": {"index": -1, "scope": {"variable": "entity"}},
    "relative_to": {"index": 1, "relative_to": {"variable": "entity"}},
    "segment": {"segment": "ent"},
    "segment_scope": {"index": 0, "scope": {"segment": "ent"}},
    "all": {"all": True},
    "indices": {"indices": [0, 2]},
}
SEGMENTS = {"declare": {"ent": {"column": "entity"}}}


def _selector_doc(spec: dict[str, Any]) -> dict[str, Any]:
    segments = SEGMENTS if "segment" in json.dumps(spec) else None
    return _positions_doc({"p": spec}, {"r": "p"}, segments=segments)


@pytest.mark.parametrize("hooks_name,trace_name", FIXTURES, ids=FIXTURE_IDS)
@pytest.mark.parametrize("selector", sorted(SELECTORS))
def test_f2_each_selector_gathers_the_same_tokens(
    request, hooks_name, trace_name, selector
):
    """F2. The same anchor, the same rows, the same tokens on both engines —
    dense or ragged as the rows make it."""
    hooks, trace = _bundles(request, hooks_name, trace_name)
    hooked, traced, a, b = _read_both(
        _selector_doc(SELECTORS[selector]),
        hooks,
        trace,
        "r",
        base_texts=DAY_TEXTS,
        extra_columns=DAY_COLUMNS,
    )
    _same(a, b, f"selector {selector!r}")
    assert not isinstance(hooked.resolution("r"), Unavailable)
    assert not isinstance(traced.resolution("r"), Unavailable)


def test_f2_the_selectors_are_not_all_the_same_window(hooks_llama):
    """Anti-vacuity: the ten selectors resolve to at least six distinct
    windows on the reference engine (some coincide by construction: `index`
    and `scope` both end on the entity's last token when it ends the clause)."""
    windows: set[tuple[tuple[int, ...], ...]] = set()
    for spec in SELECTORS.values():
        executor = engines.executor_for(
            PointExecutor,
            _selector_doc(spec),
            hooks_llama,
            base_texts=DAY_TEXTS,
            extra_columns=DAY_COLUMNS,
        )
        batch = executor.frame("base")
        per_row = executor._positions("p", batch, "base")  # pyright: ignore[reportPrivateUsage]
        windows.add(tuple(tuple(int(i) for i in row) for row in per_row))
    assert len(windows) >= 6, windows


@pytest.mark.parametrize("hooks_name,trace_name", FIXTURES, ids=FIXTURE_IDS)
def test_f2_an_absent_variable_is_the_same_unavailable_cell(
    request, hooks_name, trace_name
):
    """F2 (`alignment_missing`). A value that occurs 0 times in its row is a
    read's unavailable cell, not a refusal: the same reason, the same detail
    text, the same zero width on both engines."""
    hooks, trace = _bundles(request, hooks_name, trace_name)
    hooked, traced, a, b = _read_both(
        _selector_doc(SELECTORS["variable"]),
        hooks,
        trace,
        "r",
        base_texts=DAY_TEXTS,
        extra_columns={"entity": ["Thursday", "Sunday"]},
    )
    ha, ta = _unavailable(hooked, "r"), _unavailable(traced, "r")
    assert ha.reason == ta.reason == "alignment_missing"
    assert ha.detail == ta.detail and "'Sunday' occurs 0 times" in ha.detail
    _same(a, b, "a read with one absent row")
    assert isinstance(a, RaggedValue) and a.widths[1] == 0 and a.widths[0] > 0


@pytest.mark.parametrize("hooks_name,trace_name", FIXTURES, ids=FIXTURE_IDS)
def test_f2_an_ambiguous_variable_is_the_same_unavailable_cell(
    request, hooks_name, trace_name
):
    """F2 (`alignment_ambiguous`). Several occurrences: the same cell."""
    hooks, trace = _bundles(request, hooks_name, trace_name)
    hooked, traced, a, b = _read_both(
        _selector_doc(SELECTORS["variable"]),
        hooks,
        trace,
        "r",
        base_texts=["day after day", "one two three"],
        extra_columns={"entity": ["day", "two"]},
    )
    ha, ta = _unavailable(hooked, "r"), _unavailable(traced, "r")
    assert ha.reason == ta.reason == "alignment_ambiguous"
    assert ha.detail == ta.detail and "'day' occurs 2 times" in ha.detail
    _same(a, b, "a read with one ambiguous row")
    assert isinstance(a, RaggedValue) and a.widths[0] == 0 and a.widths[1] > 0


def test_f2_an_ambiguous_row_under_a_write_is_the_same_refusal(
    hooks_llama, trace_llama
):
    """F2, the write half (`test_alignment_run.py`): a write cannot skip a
    row, so the same fact is a pre-forward refusal — identical text."""
    message = _refusal(
        alignment_ref._variable_write_doc(),  # pyright: ignore[reportPrivateUsage]
        hooks_llama,
        trace_llama,
        base_texts=["day after day", "one two three"],
        counterfactual_texts=["one day only", "one two three"],
        extra_columns={"entity": ["day", "two"]},
    )
    assert "'day' occurs 2 times" in message


# --------------------------------------------------------------------------- #
# F3 — the span algebra, atomic addresses, edit-group atomicity
# --------------------------------------------------------------------------- #

ALGEBRA: dict[str, dict[str, Any]] = {
    "union": {"union": [{"index": 0}, {"index": -1}]},
    "intersection": {"intersection": [{"span": [0, 4]}, {"span": [2, 6]}]},
    "between": {"between": [{"variable": "entity"}, {"index": -1}]},
    "before": {"before": {"variable": "entity"}},
    "after": {"after": {"variable": "entity"}},
    "atomic_variable": {"variable": "entity", "atomic": True},
    "atomic_span": {"span": [1, 3], "atomic": True},
    "atomic_indices": {"indices": [0, 2], "atomic": True},
}


@pytest.mark.parametrize("hooks_name,trace_name", FIXTURES, ids=FIXTURE_IDS)
@pytest.mark.parametrize("selector", sorted(ALGEBRA))
def test_f3_the_span_algebra_gathers_the_same_tokens(
    request, hooks_name, trace_name, selector
):
    """F3. `union` / `intersection` / `between` / `before` / `after` and the
    `atomic` forms, as reads, agree on both engines."""
    hooks, trace = _bundles(request, hooks_name, trace_name)
    _, _, a, b = _read_both(
        _selector_doc(ALGEBRA[selector]),
        hooks,
        trace,
        "r",
        base_texts=DAY_TEXTS,
        extra_columns=DAY_COLUMNS,
    )
    _same(a, b, f"span algebra {selector!r}")


def test_f3_an_atomic_span_is_the_plain_anchor_read_as_one_address(
    hooks_llama, trace_llama
):
    """The `atomic` flag changes how the address is *classified*, not which
    tokens it gathers: the atomic variable reads the same tokens as the bare
    one on both engines."""
    kw: dict[str, Any] = {"base_texts": DAY_TEXTS, "extra_columns": DAY_COLUMNS}
    _, _, plain_h, plain_t = _read_both(
        _selector_doc(SELECTORS["variable"]), hooks_llama, trace_llama, "r", **kw
    )
    _, _, atomic_h, atomic_t = _read_both(
        _selector_doc(ALGEBRA["atomic_variable"]), hooks_llama, trace_llama, "r", **kw
    )
    _same(plain_h, atomic_h, "atomic vs plain (pytorch_hooks)")
    _same(plain_t, atomic_t, "atomic vs plain (nnsight)")


def test_f3_an_empty_selector_is_the_same_refusal(hooks_llama, trace_llama):
    """`between` two runs that touch selects nothing: refused rather than
    gathered silently — `empty_selector`, the same words."""
    message = _refusal(
        _selector_doc({"between": [{"index": 0}, {"index": 1}]}),
        hooks_llama,
        trace_llama,
        base_texts=DAY_TEXTS,
        extra_columns=DAY_COLUMNS,
    )
    assert "resolves to no token" in message


def _atomic_write_doc(pos: dict[str, Any]) -> dict[str, Any]:
    """`test_location_ledger._write_doc` — swap the counterfactual's value at
    `pos` into base, read `block_output` at -1 in the patched model — plus a
    read of the patched logits, which is where a write at other positions
    of the same layer output shows up (the same-site read at -1 cannot see
    it: nothing downstream of the write feeds that position at that layer)."""
    doc = ledger_ref._write_doc(pos)  # pyright: ignore[reportPrivateUsage]
    doc["method"]["sites"]["lm_head"] = {"component": "lm_head"}
    doc["method"]["reads"]["logits"] = {"site": "lm_head", "pos": -1}
    doc["method"]["intervened_models"]["patched"]["reads"].append("logits")
    doc["method"]["save"].append(saved("logits", "patched", "logits.safetensors"))
    return doc


RAGGED_ATOMIC_COLUMNS = {
    "entity": ["cat", "caterpillar"],
    "counterfactual_inputs_variables": [{"entity": "dog"}, {"entity": "doghouse"}],
}


def test_f3_a_ragged_atomic_span_write_is_the_same_refusal(hooks_llama, trace_llama):
    """Rule 19 on an atomic span whose width differs across rows (`cat` is
    one piece, `caterpillar` several): refused before any forward with the
    same words on both engines."""
    tok = hooks_llama.tokenizer
    widths = {
        len(tok.encode(e, add_special_tokens=False)) for e in ("cat", "caterpillar")
    }
    assert len(widths) == 2, "the premise: the two entities tokenize to two widths"
    message = _refusal(
        _atomic_write_doc({"variable": "entity", "atomic": True}),
        hooks_llama,
        trace_llama,
        base_texts=ledger_ref.RAGGED_BASE,
        counterfactual_texts=ledger_ref.RAGGED_CF,
        extra_columns=RAGGED_ATOMIC_COLUMNS,
    )
    assert "[V19]" in message


def test_f3_twin_a_fixed_width_atomic_span_write_agrees_and_lands(
    hooks_llama, trace_llama
):
    """The twin: `indices: [1, 2]` atomic is one address of width two on
    every row (the tokens after BOS, where the pair differs); the write lands
    on both engines and the patched values agree."""
    doc = _atomic_write_doc({"indices": [1, 2], "atomic": True})
    clean = sweep.read_doc("lm_head", None)
    hooked, traced = engines.both_executors(
        doc,
        hooks_llama,
        trace_llama,
        base_texts=ledger_ref.RAGGED_BASE,
        counterfactual_texts=ledger_ref.RAGGED_CF,
    )
    _same(
        hooked.read_value("out"),
        traced.read_value("out"),
        "block_output at -1 after an atomic two-token swap",
    )
    a, b = hooked.read_value("logits"), traced.read_value("logits")
    _same(a, b, "logits after an atomic two-token swap")
    for cls, bundle, patched in (
        (PointExecutor, hooks_llama, a),
        (TracePointExecutor, trace_llama, b),
    ):
        unpatched = engines.executor_for(
            cls, clean, bundle, base_texts=ledger_ref.RAGGED_BASE
        ).read_value("r")
        assert not torch.allclose(patched, unpatched, atol=ATOL), (
            f"{cls.__name__}: the atomic swap left the logits unchanged"
        )


def _edit_group_doc(*positions: str) -> dict[str, Any]:
    """`test_edit_groups_run._doc` on the sentencepiece fixture: the shipped
    `match` aggregation is dropped (its space-prefixed answers are two pieces
    here, the aggregation's own [P2]); the patched logits are saved raw
    instead. Rule 27 is about the write set, which is unchanged but for the
    target layer."""
    doc = edit_groups_ref._doc(*positions)  # pyright: ignore[reportPrivateUsage]
    doc["model"]["key"] = TINY_LLAMA
    # the target moves to layer 0: on the two-layer fixture a swap at the
    # last layer's output at a non-final position cannot reach the logits
    doc["method"]["sites"]["target"]["layers"] = [0]
    doc["method"]["save"] = [saved("logits", "patched", "logits.safetensors")]
    return doc


def _edit_group_kw(atomic: bool | None) -> dict[str, Any]:
    return {
        "base_texts": edit_groups_ref.BASES,
        "counterfactual_texts": edit_groups_ref.CFS,
        "extra_columns": edit_groups_ref._columns(atomic),  # pyright: ignore[reportPrivateUsage]
    }


def test_f3_a_constituent_addressed_alone_is_the_same_rule_27_refusal(
    hooks_llama, trace_llama
):
    """F3 / P21 (`check_edit_groups`). An interchange at the first entry of
    an `atomic` two-entry swap is refused before any forward — rule 27,
    naming the group and the sibling — identically on both engines."""
    message = _refusal(
        _edit_group_doc("first"), hooks_llama, trace_llama, **_edit_group_kw(True)
    )
    assert "[V27]" in message and "'mapping_swap'" in message
    assert "constituent(s) 0 ('1')" in message and "sibling(s) 1 ('2')" in message


def test_f3_twin_the_whole_atomic_group_runs_and_agrees(hooks_llama, trace_llama):
    """Both constituents addressed in one intervened model: the writes land
    (patched logits move off the clean ones on each engine) and agree."""
    doc = _edit_group_doc("first", "second")
    hooked, traced = engines.both_executors(
        doc, hooks_llama, trace_llama, **_edit_group_kw(True)
    )
    a, b = hooked.read_value("logits"), traced.read_value("logits")
    _same(a, b, "patched logits after the whole atomic group")
    clean = sweep.read_doc("lm_head", None)
    for cls, bundle, patched in (
        (PointExecutor, hooks_llama, a),
        (TracePointExecutor, trace_llama, b),
    ):
        unpatched = engines.executor_for(
            cls, clean, bundle, base_texts=edit_groups_ref.BASES
        ).read_value("r")
        assert not torch.allclose(patched, unpatched, atol=ATOL), cls.__name__


def test_f3_twin_a_non_atomic_group_runs_the_partial_address(hooks_llama, trace_llama):
    """`atomic: false` on the same rows: the first entry alone runs, and the
    two engines agree on the patched logits."""
    hooked, traced = engines.both_executors(
        _edit_group_doc("first"), hooks_llama, trace_llama, **_edit_group_kw(False)
    )
    _same(
        hooked.read_value("logits"),
        traced.read_value("logits"),
        "patched logits under a non-atomic group",
    )


# --------------------------------------------------------------------------- #
# F4 — a declared alignment under each cardinality
# --------------------------------------------------------------------------- #

#: (declared cardinality, base text, base entity, cf text, cf entity) — the
#: pairs whose observed cardinality the declaration matches; the widths are
#: asserted against the tokenizer in the test, not assumed.
CARDINALITY_PAIRS: dict[str, tuple[str, str, str, str]] = {
    "one_to_one": ("the cat sat", "cat", "the dog sat", "dog"),
    "one_to_many": ("the cat sat", "cat", "the caterpillar sat", "caterpillar"),
    "many_to_one": ("the caterpillar sat", "caterpillar", "the cat sat", "cat"),
    "absent": ("the cat sat", "cat", "the dog sat", "cat"),
    "ambiguous": ("day after day", "day", "night after night", "night"),
}


def _declared_kw(pair: tuple[str, str, str, str]) -> dict[str, Any]:
    base, entity, cf, cf_entity = pair
    return {
        "base_texts": [base],
        "counterfactual_texts": [cf],
        "extra_columns": {
            "entity": [entity],
            "counterfactual_inputs_variables": [{"entity": cf_entity}],
        },
    }


def _observed(bundle: Any, pair: tuple[str, str, str, str]) -> str:
    """The pair's cardinality as the tokenizer makes it (`alignment_of` over
    the candidate runs), so the test asserts against the tokenizer."""
    from causalab.neural.shared.encoding import candidate_runs, encode
    from causalab.protocol.positions.alignment import alignment_of
    from causalab.protocol.schema import PositionSpec

    spec = PositionSpec(variable="entity")
    base, entity, cf, cf_entity = pair
    runs = [
        candidate_runs(
            spec,
            encode(bundle.tokenizer, [text]),
            0,
            dataset_row={"input": text, "entity": value},
            field="input",
        )
        for text, value in ((base, entity), (cf, cf_entity))
    ]
    return alignment_of(*runs)


@pytest.mark.parametrize("declared", sorted(CARDINALITY_PAIRS))
def test_f4_a_matching_declaration_runs_the_same_on_both_engines(
    hooks_llama, trace_llama, declared
):
    """F4 / P21 (declared alignment). Each cardinality declared on a pair
    that has it: the read runs on both engines with the same value (or the
    same unavailable cell, for the two that do not pair)."""
    pair = CARDINALITY_PAIRS[declared]
    observed = _observed(hooks_llama, pair)
    assert observed == declared, f"the fixture tokenizer makes this pair {observed!r}"
    doc = alignment_ref._declared_doc(declared)  # pyright: ignore[reportPrivateUsage]
    hooked, traced, a, b = _read_both(
        doc, hooks_llama, trace_llama, "r", **_declared_kw(pair)
    )
    _same(a, b, f"a read declared {declared!r}")
    cell_h, cell_t = hooked.resolution("r"), traced.resolution("r")
    if declared == "ambiguous":
        assert isinstance(cell_h, Unavailable) and isinstance(cell_t, Unavailable)
        assert cell_h.reason == cell_t.reason == "alignment_ambiguous"
        assert cell_h.detail == cell_t.detail
    else:
        assert not isinstance(cell_h, Unavailable) and not isinstance(
            cell_t, Unavailable
        )


@pytest.mark.parametrize(
    "declared,pair_name",
    [
        ("one_to_one", "one_to_many"),
        ("one_to_many", "many_to_one"),
        ("one_to_one", "absent"),
    ],
)
def test_f4_a_contradicted_declaration_is_the_same_refusal(
    hooks_llama, trace_llama, declared, pair_name
):
    """F4 / P21. The declaration the pair contradicts: refused before any
    forward, naming both cardinalities, identically."""
    pair = CARDINALITY_PAIRS[pair_name]
    observed = _observed(hooks_llama, pair)
    assert observed != declared
    message = _refusal(
        alignment_ref._declared_doc(declared),  # pyright: ignore[reportPrivateUsage]
        hooks_llama,
        trace_llama,
        **_declared_kw(pair),
    )
    assert f"declares alignment {declared!r}" in message
    assert f"resolves it as {observed!r}" in message


# --------------------------------------------------------------------------- #
# F5 — the chat frame
# --------------------------------------------------------------------------- #

CHAT = {"frame": "chat"}


def _chat_doc(pos: dict[str, Any]) -> dict[str, Any]:
    return _positions_doc({"p": pos}, {"r": "p"}, segments=CHAT)


@pytest.mark.parametrize("hooks_name,trace_name", FIXTURES, ids=FIXTURE_IDS)
def test_f5_the_user_turn_is_the_same_window_under_the_chat_frame(
    request, hooks_name, trace_name
):
    """F5. Both fixtures ship a chat template. `segment: user` and the
    rebased `index: 0` read the same tokens on both engines, and the frame
    is real: a non-zero prefix on every row."""
    hooks, trace = _bundles(request, hooks_name, trace_name)
    assert getattr(hooks.tokenizer, "chat_template", None)
    for pos in ({"segment": "user"}, {"index": 0}, {"all": True}):
        hooked, traced, a, b = _read_both(
            _chat_doc(pos), hooks, trace, "r", base_texts=DAY_TEXTS
        )
        _same(a, b, f"chat frame, position {pos}")
        for executor in (hooked, traced):
            prefix = executor.frame("base").prefix_lengths
            assert len(prefix) == 2 and all(p > 0 for p in prefix), prefix


def test_f5_the_chat_frame_moves_the_read_off_the_plain_one(hooks_llama, trace_llama):
    """Anti-vacuity: under the template the user turn sits after a real
    prefix, so `index: 0` reads a different vector than the plain frame's."""
    for cls, bundle in (
        (PointExecutor, hooks_llama),
        (TracePointExecutor, trace_llama),
    ):
        framed = engines.executor_for(
            cls, _chat_doc({"index": 0}), bundle, base_texts=DAY_TEXTS
        ).read_value("r")
        plain = engines.executor_for(
            cls,
            _positions_doc({"p": {"index": 0}}, {"r": "p"}),
            bundle,
            base_texts=DAY_TEXTS,
        ).read_value("r")
        assert not torch.allclose(framed, plain, atol=ATOL), cls.__name__


def test_f5_the_assistant_prefix_reads_where_the_template_has_one(
    hooks_qwen, trace_qwen
):
    """The qwen template appends a generation prompt: `assistant_prefix`
    is a real window, read identically."""
    _, _, a, b = _read_both(
        _chat_doc({"index": -1, "scope": {"segment": "assistant_prefix"}}),
        hooks_qwen,
        trace_qwen,
        "r",
        base_texts=DAY_TEXTS,
    )
    _same(a, b, "the last token of the assistant prefix")
    assert a.shape[:2] == (2, 1)


def test_f5_an_absent_assistant_prefix_is_the_same_cell(hooks_llama, trace_llama):
    """The Llama-2 template carries no generation prompt, so the segment is
    `absent` on every row — the same `alignment_missing` cell on both."""
    hooked, traced, a, b = _read_both(
        _chat_doc({"segment": "assistant_prefix"}),
        hooks_llama,
        trace_llama,
        "r",
        base_texts=DAY_TEXTS,
    )
    ha, ta = _unavailable(hooked, "r"), _unavailable(traced, "r")
    assert ha.reason == ta.reason == "alignment_missing"
    assert ha.detail == ta.detail
    _same(a, b, "an absent segment")


def test_f5_frame_chat_without_a_template_is_the_same_refusal(hooks_llama, trace_llama):
    """The fail-closed twin: the template is data on the tokenizer object;
    with it removed, `frame: chat` is `chat_template_missing` on both engines,
    word for word, before any forward."""
    tokenizers = {id(b.tokenizer): b.tokenizer for b in (hooks_llama, trace_llama)}
    saved = {key: tok.chat_template for key, tok in tokenizers.items()}
    for tok in tokenizers.values():
        tok.chat_template = None
    try:
        message = _refusal(
            _chat_doc({"segment": "user"}),
            hooks_llama,
            trace_llama,
            base_texts=DAY_TEXTS,
        )
    finally:
        for key, tok in tokenizers.items():
            tok.chat_template = saved[key]
    assert "carries no chat template" in message


# --------------------------------------------------------------------------- #
# F6 — the location ledger, saved (engine level)
# --------------------------------------------------------------------------- #


#: 📐 FINDING (F6/F7): at the engine seam the
#: two engines load different attention implementations by default. The
#: reference engine loads `eager`; the nnsight engine keeps the checkpoint's
#: own default (`sdpa`). Each stamps it into every saved tensor's entry as
#: `loaded_attn_implementation`, so one document yields artifacts with two
#: provenance stamps (the values agree at ATOL regardless).
#: `KNOWN_ENTRY_RECORD_DIFFERENCES` in the harness lists the
#: field, and `test_f6_the_default_attention_implementation_stamp_is_the_known_pair`
#: asserts the pair as the current state so the entry cannot outlive the
#: asymmetry. The other engine-level tests that save tensors pin the
#: implementation, so what they claim is tested like for like.
PINNED_ATTENTION = {"model.attn_implementation": "eager"}


def _ledger_runs(base: Path, *, pinned: bool) -> engines.BothRuns:
    env = engines.corpus_env(base / "artifacts")
    doc = ledger_ref._run_doc(base, ledger=True)  # pyright: ignore[reportPrivateUsage]
    return engines.run_both(
        doc, env, base / "out", overrides=PINNED_ATTENTION if pinned else None
    )


@pytest.fixture(scope="module")
def ledger_runs(tmp_path_factory: pytest.TempPathFactory) -> engines.BothRuns:
    return _ledger_runs(tmp_path_factory.mktemp("ledger"), pinned=True)


def test_f6_the_default_attention_implementation_stamp_is_the_known_pair(
    tmp_path: Path,
):
    """The finding above as the current state. The same document, no
    `attn_implementation` authored, saves tensors whose provenance stamps
    differ in exactly this field and this way: the reference loader's
    `eager` against the checkpoint default the nnsight loader keeps. The
    rest of the two directories agrees. When the loaders agree this fails,
    and `KNOWN_ENTRY_RECORD_DIFFERENCES` loses its one entry."""
    runs = _ledger_runs(tmp_path, pinned=False)
    stamps = []
    for out in (runs.hooks_dir, runs.trace_dir):
        metadata = read_safetensors_metadata(out / "r.safetensors")
        assert metadata is not None
        (record,) = json.loads(str(metadata["entries"])).values()
        stamps.append(record["loaded_attn_implementation"])
    assert stamps == ["eager", "sdpa"], stamps
    runs.compare()


def test_f6_the_location_ledger_agrees_row_for_row(ledger_runs):
    """F6. The saved `location_ledger` table: the seven columns, the same
    rows in the same order, and the file set and receipt agree."""
    ledger_runs.compare()
    hooks_rows = json.loads((ledger_runs.hooks_dir / "ledger.json").read_text())
    trace_rows = json.loads((ledger_runs.trace_dir / "ledger.json").read_text())
    assert hooks_rows and hooks_rows == trace_rows
    assert all(set(LEDGER_COLUMNS) <= set(row) for row in hooks_rows)
    assert {row["constituent"] for row in hooks_rows} == {"ent"}


# --------------------------------------------------------------------------- #
# F7 — path patching in the hand form
# --------------------------------------------------------------------------- #

#: `test_write_set_fires.py::MULTI_WRITE_CORPUS`'s retargeting onto the
#: two-layer fixture: sender head 1 at layer 0, receiver at layer 1, the two
#: frozen attention outputs on the two layers.
PATH_OVERRIDES = {
    "model.key": TINY_LLAMA,
    "model.dtype": "fp32",
    "sites.sender.layers": 0,
    "sites.sender.head": 1,
    "sites.receiver.layers": 1,
    "sites.a10.layers": 0,
    "sites.a11.layers": 1,
}


def _record(out: Path) -> dict[str, Any]:
    return json.loads((out / RUN_RECORD_NAME).read_text())


def _fires(out: Path) -> dict[str, dict[str, int]]:
    """The one point's fire tally in the receipt ``out`` holds."""
    record = _record(out)
    (point,) = record["points"]
    return record[FIRES_KEY][point["digest"]]


def _assert_no_trace_fires(runs: engines.BothRuns) -> None:
    """The `fires` asymmetry as the current state: the nnsight executor
    tallies no fires, so its receipt carries an empty block where the reference names every
    write. `KNOWN_RECORD_DIFFERENCES` lists `fires` for this reason; this
    fails the day the nnsight engine starts counting."""
    assert _record(runs.trace_dir)[FIRES_KEY] == {}


def test_f7_the_corpus_path_patching_document_agrees(tmp_path: Path):
    """F7. `03_path_patching_im.json` through both engines: the
    `logit_diff` table and the receipt agree, modulo the `fires`
    asymmetry, which is asserted as the current state."""
    env = engines.corpus_env(tmp_path / "artifacts")
    runs = engines.run_both(
        CORPUS_DIR / "03_path_patching_im.json",
        env,
        tmp_path / "out",
        overrides=PATH_OVERRIDES,
    )
    runs.compare()
    fires = _fires(runs.hooks_dir)
    assert fires == {
        "patched on base": {"swap_sender": 1, "freeze_10": 1, "freeze_11": 1},
        "final on base": {"inject": 1},
    }
    _assert_no_trace_fires(runs)


#: `test_path_patching_run.py`'s joint-pass documents: two nested receivers
#: of layer 1 in one `final` model, and each receiver alone.
#: `_document` builds the method in the reference file's order, so each one
#: is rebuilt in the §1 order here to keep the order warning out of the run.
RECEIVER_DOCS: dict[str, dict[str, Any]] = {
    "joint": paths_ref.JOINT,
    "upstream": paths_ref.UPSTREAM_ONLY,
    "downstream": paths_ref.DOWNSTREAM_ONLY,
}


@pytest.fixture(scope="module")
def receiver_runs(
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, engines.BothRuns]:
    base = tmp_path_factory.mktemp("receivers")
    env = engines.corpus_env(base / "artifacts")
    return {
        name: engines.run_both(
            in_order(
                paths_ref._document(  # pyright: ignore[reportPrivateUsage]
                    TINY_LLAMA, method, paths_ref.METRIC_READOUT
                )
            ),
            env,
            base / name,
        )
        for name, method in RECEIVER_DOCS.items()
    }


@pytest.mark.parametrize("name", sorted(RECEIVER_DOCS))
def test_f7_the_receiver_documents_agree(receiver_runs, name):
    """F7. Each receiver document: both `logit_diff` tables and the receipt
    agree. The reference engine counts each injection once in the one
    `final` group; the nnsight receipt is empty."""
    runs = receiver_runs[name]
    runs.compare()
    injected = {"inject_0": 1, "inject_1": 1} if name == "joint" else {"inject": 1}
    assert _fires(runs.hooks_dir) == {
        "patched on base": {"swap_sender": 1},
        "final on base": injected,
    }
    _assert_no_trace_fires(runs)


def test_f7_the_joint_pass_is_not_a_sum_on_both_engines(receiver_runs):
    """The reference test's claim, held on each engine: the joint pass equals
    the upstream-only run (the nested MLP recomputes to its harvested value)
    and differs from the sum of the two separate effects by more than
    `paths_ref.GAP`. Each receiver moves the metric on its own."""
    gap = paths_ref.GAP
    for side in ("hooks_dir", "trace_dir"):

        def values(name: str, table: str) -> list[float]:
            out = getattr(receiver_runs[name], side)
            return [row["value"] for row in json.loads((out / table).read_text())]

        clean = values("joint", "ld_clean.json")
        joint = values("joint", "logit_diff.json")
        up = values("upstream", "logit_diff.json")
        down = values("downstream", "logit_diff.json")
        summed = [u + d - c for u, d, c in zip(up, down, clean)]
        assert max(abs(j - s) for j, s in zip(joint, summed)) > gap, side
        assert max(abs(d - c) for d, c in zip(down, clean)) > gap, side
        assert max(abs(u - c) for u, c in zip(up, clean)) > gap, side
        assert max(abs(j - u) for j, u in zip(joint, up)) <= 1e-6, side


#: `test_path_patching_run.py`'s freeze-set documents on the five-layer GPT-2
#: fixture: attention output of layer 1 frozen, and that plus the MLP
#: outputs of layers 0 and 1.
FREEZE_DOCS: dict[str, dict[str, Any]] = {
    "attention": paths_ref.FREEZE_ATTENTION,
    "attention_and_mlp": paths_ref.FREEZE_ATTENTION_AND_MLP,
}


@pytest.fixture(scope="module")
def freeze_runs(
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, engines.BothRuns]:
    base = tmp_path_factory.mktemp("freeze")
    env = engines.corpus_env(base / "artifacts")
    return {
        name: engines.run_both(
            in_order(
                paths_ref._document(  # pyright: ignore[reportPrivateUsage]
                    paths_ref.TINY_GPT2, method, paths_ref.LOGITS_READOUT
                )
            ),
            env,
            base / name,
            overrides=PINNED_ATTENTION,
        )
        for name, method in FREEZE_DOCS.items()
    }


@pytest.mark.parametrize("name", sorted(FREEZE_DOCS))
def test_f7_each_freeze_set_agrees(freeze_runs, name):
    """F7. The saved logits (patched and clean) and the receipt agree under
    each freeze set. The reference engine counts every freeze once; the
    nnsight receipt is empty."""
    runs = freeze_runs[name]
    runs.compare()
    patched = {"swap_sender": 1, "freeze_1": 1}
    if name == "attention_and_mlp":
        patched |= {"freeze_m0": 1, "freeze_m1": 1}
    assert _fires(runs.hooks_dir)["patched on base"] == patched
    _assert_no_trace_fires(runs)


def test_f7_the_freeze_set_changes_the_numbers_on_both_engines(freeze_runs):
    """Anti-vacuity: the two freeze sets run different write sets and land on
    different logits on each engine, and neither equals the clean logits."""
    from causalab.io.tensor_files import load_file

    attention, both = freeze_runs["attention"], freeze_runs["attention_and_mlp"]
    assert (
        _record(attention.hooks_dir)["points"][0]["digest"]
        != _record(both.hooks_dir)["points"][0]["digest"]
    )
    for side in ("hooks_dir", "trace_dir"):
        (la,) = load_file(str(getattr(attention, side) / "logits.safetensors")).values()
        (lb,) = load_file(str(getattr(both, side) / "logits.safetensors")).values()
        (clean,) = load_file(
            str(getattr(attention, side) / "logits_clean.safetensors")
        ).values()
        assert not torch.allclose(la, lb, atol=ATOL), side
        assert not torch.allclose(la, clean, atol=ATOL), side
        assert not torch.allclose(lb, clean, atol=ATOL), side


# --------------------------------------------------------------------------- #
# F8 — a head slice on a write
# --------------------------------------------------------------------------- #

HEAD = 1


def _head_write_doc(component: str, layer: int) -> dict[str, Any]:
    """Swap head `HEAD` of `component` from the counterfactual into base at
    the last token; read the patched logits, the clean logits, and the whole
    component clean and patched (so the untouched heads can be compared)."""
    return {
        "header": {"protocol_version": PROTOCOL_VERSION},
        "model": {"key": "test", "revision": "main"},
        "data": sweep._data(with_cf=True),  # pyright: ignore[reportPrivateUsage]
        "method": {
            "intervened_models": {
                UNWRITTEN: {"input": "counterfactual", "reads": ["v_cf"]},
                "original": {"input": "base", "reads": ["whole_clean", "clean"]},
                "patched": {
                    "input": "base",
                    "reads": ["whole_patched", "logits"],
                    "writes": ["patch"],
                },
            },
            "sites": {
                "slice": {"component": component, "layers": layer, "head": HEAD},
                "whole": {"component": component, "layers": layer},
                "lm_head": {"component": "lm_head"},
            },
            "reads": {
                "v_cf": {"site": "slice", "pos": -1},
                "whole_clean": {"site": "whole", "pos": -1},
                "whole_patched": {"site": "whole", "pos": -1},
                "clean": {"site": "lm_head", "pos": -1},
                "logits": {"site": "lm_head", "pos": -1},
            },
            "writes": {"patch": {"site": "slice", "pos": -1, "do": {"swap": "v_cf"}}},
            "save": [
                saved(name, model, f"{name}.safetensors")
                for name, model in (
                    ("whole_clean", "original"),
                    ("whole_patched", "patched"),
                    ("clean", "original"),
                    ("logits", "patched"),
                )
            ],
        },
    }


@pytest.mark.parametrize("component", ["attention_premix", "attention_z"])
def test_f8_a_head_sliced_swap_agrees_and_touches_one_head(
    hooks_qwen, trace_qwen, component
):
    """F8. `head: 1` on the o-projection input and on the attention-function
    output, at the fixture's full-attention layer: the patched logits agree,
    the write landed (logits moved off clean on each engine), and the whole
    component read after the write shows the other heads unchanged and head
    1 carrying the counterfactual's value — on both engines."""
    _, full_layer = sweep.stream_layers(hooks_qwen)
    doc = _head_write_doc(component, full_layer)
    hooked, traced = engines.both_executors(
        doc,
        hooks_qwen,
        trace_qwen,
        base_texts=BASE_TEXTS,
        counterfactual_texts=CF_TEXTS,
    )
    _same(
        hooked.read_value("logits"),
        traced.read_value("logits"),
        f"logits after a head swap at {component}",
    )
    _same(
        hooked.read_value("whole_patched"),
        traced.read_value("whole_patched"),
        f"{component} after a head swap",
    )
    info = hooks_qwen.info
    head_dim = info.head_dim
    for label, executor in (("pytorch_hooks", hooked), ("nnsight", traced)):
        assert not torch.allclose(
            executor.read_value("logits"), executor.read_value("clean"), atol=ATOL
        ), f"{label}: the head swap left the logits unchanged"
        clean = executor.read_value("whole_clean")
        patched = executor.read_value("whole_patched")
        v_cf = executor.read_value("v_cf")
        assert clean.shape[-1] == info.num_heads * head_dim
        assert v_cf.shape[-1] == head_dim
        touched = slice(HEAD * head_dim, (HEAD + 1) * head_dim)
        untouched = [h for h in range(info.num_heads) if h != HEAD]
        for h in untouched:
            s = slice(h * head_dim, (h + 1) * head_dim)
            assert torch.allclose(patched[..., s], clean[..., s], atol=ATOL), (
                f"{label}: head {h} moved under a head-{HEAD} write"
            )
        assert torch.allclose(patched[..., touched], v_cf, atol=ATOL), (
            f"{label}: head {HEAD} does not carry the swapped value"
        )
        assert not torch.allclose(clean[..., touched], v_cf, atol=ATOL), (
            f"{label}: the counterfactual head equals the clean one — the swap is vacuous"
        )


# --------------------------------------------------------------------------- #
# P2 (read half) — lm_head at every position
# --------------------------------------------------------------------------- #

#: Two rows of equal token length per fixture, so the whole-vocabulary read
#: is dense (the metric half of P2 belongs to the metrics group).
EQUAL_LENGTH_TEXTS = {
    "llama": BASE_TEXTS,
    "qwen": ["the quick brown fox jumps", "a small red hen sits"],
}


@pytest.mark.parametrize("hooks_name,trace_name", FIXTURES, ids=FIXTURE_IDS)
def test_p2_the_full_logits_read_agrees(request, hooks_name, trace_name):
    """P2 (read half). `lm_head` at `pos: "all"` — every position, the whole
    vocabulary — is one dense tensor on both engines and agrees."""
    hooks, trace = _bundles(request, hooks_name, trace_name)
    texts = EQUAL_LENGTH_TEXTS[FIXTURE_IDS[FIXTURES.index((hooks_name, trace_name))]]
    hooked, _, a, b = _read_both(
        sweep.read_doc("lm_head", None, pos="all"), hooks, trace, "r", base_texts=texts
    )
    assert isinstance(a, torch.Tensor), "pick equal-length rows: the read went ragged"
    assert a.shape[0] == 2 and a.shape[-1] == hooks.info.vocab_size
    assert a.shape[1] == hooked.frame("base").padded_len - hooked.frame(
        "base"
    ).content_start(0)
    _same(a, b, "lm_head at every position")


# --------------------------------------------------------------------------- #
# P14 — the bounds-check refusal
# --------------------------------------------------------------------------- #

STALE_INDEX = 99


@pytest.mark.parametrize("hooks_name,trace_name", FIXTURES, ids=FIXTURE_IDS)
def test_p14_a_stale_index_on_a_module_boundary_is_the_same_refusal(
    request, hooks_name, trace_name
):
    """P14. `block_output` at `pos: 99` on a short row: out of bounds,
    refused rather than addressing the wrong token, the same words."""
    hooks, trace = _bundles(request, hooks_name, trace_name)
    message = _refusal(
        sweep.read_doc("block_output", 1, pos=STALE_INDEX),
        hooks,
        trace,
        base_texts=BASE_TEXTS,
    )
    assert "out of bounds" in message


@pytest.mark.parametrize("component", sweep.SHARED_LINEAR_ONLY)
def test_p14_a_stale_index_on_the_deltanet_interior_is_the_same_refusal(
    hooks_qwen, trace_qwen, component
):
    """P14. The same refusal on a `delta_*` component at a DeltaNet layer —
    the interior the two engines reach by unrelated mechanisms, refused
    before either reaches it."""
    delta_layer, _ = sweep.stream_layers(hooks_qwen)
    message = _refusal(
        sweep.read_doc(component, delta_layer, pos=STALE_INDEX),
        hooks_qwen,
        trace_qwen,
        base_texts=BASE_TEXTS,
    )
    assert "out of bounds" in message
