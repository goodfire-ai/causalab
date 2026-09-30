"""The generated frame and band sites through both engines
(cases P3, P12 and P13).

``test_generate_frame_nnsight.py`` already pins the greedy decode itself:
ids, steps, seven components and one prefill write agree. This module adds
what the continuation frame does *around* the decode — the anchors that
resolve against it and the rows that stop early — and the band-site shape:

* **P3** a ``variable`` anchor inside ``generated``: the steps that said the
  value, and the rows that never said it (zero occurrences is a result, not
  an error — both engines must agree it is *nothing*); ``lm_head`` and
  ``block_output`` at a DeltaNet and a full-attention layer read in one
  generated document on the hybrid fixture;
* **P12** the generated window clipped by the budget (a ``span`` past the
  depth) and by a row's EOS (a row that stops at step *k* has width *k* on
  both engines while its neighbour runs to the depth);
* **P13** band sites: a band read compared per lowered member, a band write
  on the hybrid fixture spanning a DeltaNet and a full-attention layer, and
  the corpus ``at_once`` band document at the engine seam.

The reference engine hand-rolls its decode with hooks; the nnsight engine
walks ``model.generate`` with ``tracer.iter``. Every agreement here is
therefore two mechanisms landing on one frame.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from causalab.neural.engines.nnsight_tracing.executor import TracePointExecutor
from causalab.neural.engines.pytorch_hooks.executor import PointExecutor
from causalab.neural.shared.values import RaggedValue
from causalab.protocol.lowering import band_member
from causalab.protocol.schema import PROTOCOL_VERSION

from tests._helpers import engines
from tests._helpers import a3b_sweep as sweep
from tests.neural.engines.nnsight_tracing.conftest import TINY_LLAMA
from tests.neural.engines.nnsight_tracing.test_generate_frame_nnsight import (
    DEPTH,
    _gen_doc,
)
from tests.neural.engines.nnsight_tracing.test_parity_module_boundaries import (
    BASE_TEXTS,
    CF_TEXTS,
    _data,
)
from tests.neural.engines.nnsight_tracing.test_parity_positions import _same
from tests.protocol._docs import UNWRITTEN, saved
from tests.protocol._env import CORPUS_DIR

pytestmark = pytest.mark.smoke

ATOL = engines.ATOL

FIXTURES = [("hooks_llama", "trace_llama"), ("hooks_qwen", "trace_qwen")]
FIXTURE_IDS = ["llama", "qwen"]


def _bundles(request: pytest.FixtureRequest, hooks_name: str, trace_name: str):
    return request.getfixturevalue(hooks_name), request.getfixturevalue(trace_name)


def _generated(anchor: dict[str, Any], *, depth: int = DEPTH) -> dict[str, Any]:
    return {"generated": {"max_new_tokens": depth}, **anchor}


def _frame_agrees(hooked: Any, traced: Any, name: str, what: str) -> None:
    """The read's value, the steps it addressed and the ids at them."""
    _same(hooked.read_value(name), traced.read_value(name), what)
    assert hooked.addressed_steps(name) == traced.addressed_steps(name), what
    assert hooked.generated_ids(name) == traced.generated_ids(name), what


# --------------------------------------------------------------------------- #
# P3 — the variable anchor inside the continuation
# --------------------------------------------------------------------------- #

NEVER_SAID = "definitely-not-generated-xyzzy"


@pytest.mark.parametrize("hooks_name,trace_name", FIXTURES, ids=FIXTURE_IDS)
def test_p3_a_variable_the_model_never_said_addresses_nothing_on_both(
    request, hooks_name, trace_name
):
    """P3. Zero occurrences on every row: no steps, no ids, and the same
    (empty) value on both engines — the run continues rather than refusing."""
    hooks, trace = _bundles(request, hooks_name, trace_name)
    doc = _gen_doc("block_output", layer=1, pos=_generated({"variable": "said"}))
    hooked, traced = engines.both_executors(
        doc,
        hooks,
        trace,
        base_texts=BASE_TEXTS,
        extra_columns={"said": [NEVER_SAID] * len(BASE_TEXTS)},
    )
    _frame_agrees(hooked, traced, "r", "a generated variable never said")
    assert hooked.addressed_steps("r") == [[], []]
    assert hooked.generated_ids("r") == [[], []]
    value = hooked.read_value("r")
    if isinstance(value, RaggedValue):
        assert value.widths == (0, 0)
    else:
        assert value.shape[1] == 0


def _said_pieces(hooks: Any) -> list[str]:
    """Run the plain decode once on the reference engine and take a slice of
    what each row said, so the anchor below asks for text the model does
    produce (`test_generate_metrics.py`'s recipe)."""
    seen = engines.executor_for(
        PointExecutor,
        _gen_doc("lm_head", layer=None, pos=_generated({"all": True})),
        hooks,
        base_texts=BASE_TEXTS,
    )
    seen.read_value("r")
    (continuation,) = seen._continuations.values()  # pyright: ignore[reportPrivateUsage]
    pieces = []
    for row in range(len(BASE_TEXTS)):
        text = continuation.texts[row]
        pieces.append(text[len(text) // 3 : len(text) // 3 + 4] or text[:2])
    assert all(pieces)
    return pieces


@pytest.mark.parametrize("hooks_name,trace_name", FIXTURES, ids=FIXTURE_IDS)
def test_p3_a_variable_the_model_said_lands_on_the_same_steps(
    request, hooks_name, trace_name
):
    """P3. The steps that produced the value — located through incremental
    detokenization on both engines — agree, and so do the values there."""
    hooks, trace = _bundles(request, hooks_name, trace_name)
    said = _said_pieces(hooks)
    doc = _gen_doc("block_output", layer=1, pos=_generated({"variable": "said"}))
    hooked, traced = engines.both_executors(
        doc, hooks, trace, base_texts=BASE_TEXTS, extra_columns={"said": said}
    )
    _frame_agrees(hooked, traced, "r", f"a generated variable {said!r}")
    steps = hooked.addressed_steps("r")
    assert all(steps), f"{said!r}: a row addressed no steps"
    assert any(len(row) < DEPTH for row in steps), "the anchor is the whole window"


def test_p3_lm_head_and_block_output_at_both_block_types_in_one_generated_document(
    hooks_qwen, trace_qwen
):
    """P3. One decode, three continuation reads on the hybrid fixture: the
    head, a DeltaNet layer's boundary and a full-attention layer's boundary
    — the same frame, all three values agreeing."""
    delta_layer, full_layer = sweep.stream_layers(hooks_qwen)
    doc = _gen_doc("lm_head", layer=None)
    method = doc["method"]
    method["sites"]["delta"] = {"component": "block_output", "layers": delta_layer}
    method["sites"]["full"] = {"component": "block_output", "layers": full_layer}
    for name in ("delta", "full"):
        method["reads"][f"r_{name}"] = {"site": name, "pos": "window"}
        method["intervened_models"]["original"]["reads"].append(f"r_{name}")
        method["save"].append(saved(f"r_{name}", "original", f"{name}.safetensors"))
    hooked, traced = engines.both_executors(
        doc, hooks_qwen, trace_qwen, base_texts=BASE_TEXTS
    )
    for name in ("r", "r_delta", "r_full"):
        _frame_agrees(hooked, traced, name, f"generated read {name!r}")
    head = hooked.read_value("r")
    assert head.shape == (len(BASE_TEXTS), DEPTH, hooks_qwen.info.vocab_size)
    assert hooked.read_value("r_delta").shape == (
        len(BASE_TEXTS),
        DEPTH,
        hooks_qwen.info.hidden_size,
    )
    # the two boundaries are different tensors (anti-vacuity for the sites)
    assert not torch.allclose(
        hooked.read_value("r_delta"), hooked.read_value("r_full"), atol=ATOL
    )


# --------------------------------------------------------------------------- #
# P12 — the window clipped by the budget and by a row's EOS
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("hooks_name,trace_name", FIXTURES, ids=FIXTURE_IDS)
def test_p12_a_span_past_the_budget_clips_identically(request, hooks_name, trace_name):
    """P12. `span: [1, 10]` under a depth of 4 is steps 1..3 on every row."""
    hooks, trace = _bundles(request, hooks_name, trace_name)
    doc = _gen_doc("block_output", layer=1, pos=_generated({"span": [1, 10]}))
    hooked, traced = engines.both_executors(doc, hooks, trace, base_texts=BASE_TEXTS)
    _frame_agrees(hooked, traced, "r", "a span past the budget")
    assert hooked.addressed_steps("r") == [[1, 2, 3], [1, 2, 3]]


def _plain_ids(bundle: Any, cls: type, depth: int) -> list[list[int]]:
    executor = engines.executor_for(
        cls,
        _gen_doc("lm_head", layer=None, pos=_generated({"all": True}, depth=depth)),
        bundle,
        base_texts=BASE_TEXTS,
    )
    executor.read_value("r")
    return executor.generated_ids("r")


def _early_stop_token(ids: list[list[int]]) -> tuple[int, int]:
    """A token row 0 emits at step k >= 1 (first occurrence) that row 1 never
    emits — declared EOS below, it ends row 0 at width k and leaves row 1 at
    the depth."""
    for k in range(1, len(ids[0])):
        token = ids[0][k]
        if token not in ids[0][:k] and token not in ids[1]:
            return token, k
    raise AssertionError(f"no token separates the rows: {ids}")


def test_p12_a_row_that_hits_eos_early_clips_identically(hooks_llama, trace_llama):
    """P12. A row that stops at step k has width k on both engines while
    its neighbour runs to the depth; `all` and `index: -1` read the row's
    *own* last step. The EOS is chosen from what the model actually says
    (a token row 0 emits and row 1 does not) and declared on the tokenizer
    and the generation config for the duration — the engines read the
    stopping set from there (`eos_token_ids()`; the trace's width pass)."""
    depth = 6
    plain = _plain_ids(hooks_llama, PointExecutor, depth)
    assert plain == _plain_ids(trace_llama, TracePointExecutor, depth)
    eos, width = _early_stop_token(plain)
    patched: list[tuple[Any, str, Any]] = []
    for bundle in (hooks_llama, trace_llama):
        for holder, attr in (
            (bundle.tokenizer, "eos_token_id"),
            (bundle.model.generation_config, "eos_token_id"),
        ):
            patched.append((holder, attr, getattr(holder, attr)))
            setattr(holder, attr, eos)
    try:
        for anchor, want in (
            ({"all": True}, [list(range(width)), list(range(depth))]),
            ({"index": -1}, [[width - 1], [depth - 1]]),
            ({"span": [0, depth]}, [list(range(width)), list(range(depth))]),
        ):
            doc = _gen_doc("block_output", layer=1, pos=_generated(anchor, depth=depth))
            hooked, traced = engines.both_executors(
                doc, hooks_llama, trace_llama, base_texts=BASE_TEXTS
            )
            _frame_agrees(hooked, traced, "r", f"an early EOS under {anchor}")
            assert hooked.addressed_steps("r") == want, anchor
            assert eos not in [t for row in hooked.generated_ids("r") for t in row]
        value = hooked.read_value("r")
        assert isinstance(value, RaggedValue) and value.widths == (width, depth)
    finally:
        for holder, attr, saved in patched:
            setattr(holder, attr, saved)


# --------------------------------------------------------------------------- #
# P13 — band sites
# --------------------------------------------------------------------------- #


def _band_write_doc(component: str, layers: list[int], *, one_site: bool) -> dict:
    """Swap the counterfactual's `component` at every layer of `layers` into
    the base forward: as one band site, or as the hand-written per-layer
    twin (`test_band_site_parity._band_doc`, generalized over the site)."""
    if one_site:
        sites = {"a": {"component": component, "layers": layers}}
        reads = {"v": {"site": "a", "pos": -1}}
        writes = {"w": {"site": "a", "pos": -1, "do": {"swap": "v"}}}
        in_force = ["w"]
    else:
        sites = {f"a{i}": {"component": component, "layers": [i]} for i in layers}
        reads = {f"v{i}": {"site": f"a{i}", "pos": -1} for i in layers}
        writes = {
            f"w{i}": {"site": f"a{i}", "pos": -1, "do": {"swap": f"v{i}"}}
            for i in layers
        }
        in_force = [f"w{i}" for i in layers]
    return {
        "header": {"protocol_version": PROTOCOL_VERSION},
        "model": {"key": "test", "revision": "main"},
        "data": _data(with_cf=True),
        "method": {
            "intervened_models": {
                UNWRITTEN: {"input": "counterfactual", "reads": list(reads)},
                "patched": {"input": "base", "reads": ["logits"], "writes": in_force},
            },
            "sites": {**sites, "head": {"component": "lm_head"}},
            "reads": {**reads, "logits": {"site": "head", "pos": -1}},
            "writes": writes,
            "save": [saved("logits", "patched", "l.safetensors")],
        },
    }


def _both_band(doc: dict, hooks: Any, trace: Any):
    return engines.both_executors(
        doc, hooks, trace, base_texts=BASE_TEXTS, counterfactual_texts=CF_TEXTS
    )


def test_p13_a_band_read_agrees_per_lowered_member(hooks_llama, trace_llama):
    """P13. A band read is N tensors, one per lowered member
    (`a[layers=L]`, `lowering.band_member`): each member agrees across engines,
    and equals the one-layer read of the same site on the same engine."""
    doc = _band_write_doc("attention_output", [0, 1], one_site=True)
    hooked, traced = _both_band(doc, hooks_llama, trace_llama)
    members = [band_member("v", layer) for layer in (0, 1)]
    assert sorted(n for n in hooked.doc.reads if n.startswith("v")) == members
    assert sorted(n for n in traced.doc.reads if n.startswith("v")) == members
    for layer, member in zip((0, 1), members):
        _same(hooked.read_value(member), traced.read_value(member), member)
        one_layer = sweep.read_doc("attention_output", layer)
        for cls, bundle, executor in (
            (PointExecutor, hooks_llama, hooked),
            (TracePointExecutor, trace_llama, traced),
        ):
            scalar = engines.executor_for(
                cls, one_layer, bundle, base_texts=CF_TEXTS
            ).read_value("r")
            sweep.assert_same(
                executor.read_value(member), scalar, f"{member} vs the one-layer site"
            )
    assert not torch.allclose(
        hooked.read_value(members[0]), hooked.read_value(members[1]), atol=ATOL
    )


def test_p13_a_band_write_across_the_two_block_types_agrees_and_lands(
    hooks_qwen, trace_qwen
):
    """P13. `block_output` over `[full - 1, full]` — a Gated DeltaNet layer
    and the full-attention layer in one band: the patched logits agree
    across engines, each engine runs the band as its hand-written twin to
    the bit, and the write landed."""
    _, full_layer = sweep.stream_layers(hooks_qwen)
    layers = [full_layer - 1, full_layer]
    assert hooks_qwen.stream_at(layers[0]) == "linear_attention"
    assert hooks_qwen.stream_at(layers[1]) == "full_attention"
    band = _band_write_doc("block_output", layers, one_site=True)
    hand = _band_write_doc("block_output", layers, one_site=False)
    hooked, traced = _both_band(band, hooks_qwen, trace_qwen)
    hooked_hand, traced_hand = _both_band(hand, hooks_qwen, trace_qwen)
    assert sorted(hooked.doc.sites) == [
        band_member("a", layers[0]),
        band_member("a", layers[1]),
        "head",
    ]
    _same(hooked.read_value("logits"), traced.read_value("logits"), "band logits")
    clean = sweep.read_doc("lm_head", None)
    for cls, bundle, executor, twin in (
        (PointExecutor, hooks_qwen, hooked, hooked_hand),
        (TracePointExecutor, trace_qwen, traced, traced_hand),
    ):
        assert torch.equal(executor.read_value("logits"), twin.read_value("logits")), (
            f"{cls.__name__}: the band is not its hand-written twin to the bit"
        )
        unpatched = engines.executor_for(
            cls, clean, bundle, base_texts=BASE_TEXTS
        ).read_value("r")
        assert not torch.allclose(
            executor.read_value("logits"), unpatched, atol=ATOL
        ), f"{cls.__name__}: the band write left the logits unchanged"


#: `16_at_once_band_im.json` retargeted onto the two-layer fixture: the
#: `at_once` range over both layers, the two narrow bands one layer each, the
#: wide band both — the same nesting the shipped ranges have.
AT_ONCE_OVERRIDES = {
    "model.key": TINY_LLAMA,
    "model.dtype": "fp32",
    "sites.a.layers.at_once.range": [0, 2],
    "intervened_models.band5_L10.writes[0].w.layers.at_once.range": [0, 1],
    "intervened_models.band5_L15.writes[0].w.layers.at_once.range": [1, 2],
    "intervened_models.band10_L10.writes[0].w.layers.at_once.range": [0, 2],
}


def test_p13_the_at_once_band_corpus_document_agrees_at_the_engine_seam(
    tmp_path: Path,
):
    """P13. The compiled family form: three bands over ten (here: two) sites
    from one authored entry, through both engines — the three `logit_diff`
    tables, the file set and the receipt agree; and the bands differ from
    each other on both engines (the second write lands)."""
    import json

    env = engines.corpus_env(tmp_path / "artifacts")
    runs = engines.run_both(
        CORPUS_DIR / "16_at_once_band_im.json",
        env,
        tmp_path / "out",
        overrides=AT_ONCE_OVERRIDES,
    )
    runs.compare()
    tables = ["iia_band5_L10.json", "iia_band5_L15.json", "iia_band10_L10.json"]
    assert set(tables) <= set(runs.hooks_result.files)
    for out in (runs.hooks_dir, runs.trace_dir):
        values = [
            [row["value"] for row in json.loads((out / name).read_text())]
            for name in tables
        ]
        assert all(len(v) == 2 for v in values)
        assert len({tuple(v) for v in values}) == 3, values
