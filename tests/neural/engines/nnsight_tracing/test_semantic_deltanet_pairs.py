"""A black-box test of one semantic intervention across two engines.

    Execute one semantic DeltaNet-state intervention through two compatible
    engines. The document must remain unchanged while each adapter translates
    it to its own tensor representation.

Run over the eight tensors the two engines used to reach under two spellings
(``delta_*`` / ``deltanet_*``), one name each now: the **same parsed
document** — one interchange on the DeltaNet interior, read on the
counterfactual and swapped into the base — is driven through the reference
engine (hooks and kernel-global swaps) and the nnsight engine (envoys and
``.source`` lines), and the patched logits agree; the retired spelling parses
and digests to the same document. *Mutation:* the two ``gva_tile`` pairs and
the ``chunk_boundary`` pair are **not** aliases — ``deltanet_query`` stays its
own name, the reference engine refuses it by name, and its shape differs from
``delta_query``'s (a tile, never silent) — which is the alias boundary biting.
"""

from __future__ import annotations

import pytest
import torch

from causalab.neural.engines.nnsight_tracing.executor import TracePointExecutor
from causalab.neural.engines.pytorch_hooks.executor import PointExecutor
from causalab.neural.shared.sites import resolve_site
from causalab.protocol.schema.explicit import canonicalize
from causalab.protocol.rules.errors import ProtocolError
from causalab.protocol.registry import BACKEND_PAIRS, register_model
from causalab.protocol.schema import DEPRECATED_COMPONENTS, parse_document

from tests._helpers import a3b_sweep as sweep
from tests.protocol._docs import base_doc, in_order
from tests.protocol._env import FIXTURES, build_env

from .conftest import TINY_QWEN35_MOE


pytestmark = pytest.mark.smoke

ENV = build_env(FIXTURES / "artifacts")
ROWS = [
    {
        "input": "the quick brown fox jumps over the lazy dog",
        "counterfactual_inputs": ["a slow green turtle sleeps under the old tree"],
    },
    {
        "input": "one two three four five six seven",
        "counterfactual_inputs": ["red orange yellow green blue indigo violet"],
    },
]

ALIASED = tuple(p for p in BACKEND_PAIRS if p.aliased)
TYPED = tuple(p for p in BACKEND_PAIRS if not p.aliased)


@pytest.fixture(scope="module")
def delta_layer(hooks_qwen) -> int:
    layer, _ = sweep.stream_layers(hooks_qwen)
    return layer


def _intervention(component: str, layer: int) -> dict:
    doc = sweep.interchange_doc(component, layer, pos=sweep.default_pos(component))
    doc["model"]["key"] = TINY_QWEN35_MOE
    return doc


def _logits(executor_cls, doc_raw, bundle) -> torch.Tensor:
    return sweep.make_executor(
        executor_cls, doc_raw, bundle, rows=ROWS, with_cf=True
    ).read_value("logits")


@pytest.mark.parametrize("pair", ALIASED, ids=lambda p: p.hooks)
def test_one_document_one_intervention_two_engines(
    hooks_qwen, trace_qwen, delta_layer, pair
):
    """The document is one object, parsed once; each engine translates the one
    name to its own mechanism; the patched logits agree, and the intervention
    is not a no-op."""
    register_model(hooks_qwen.info)
    raw = _intervention(pair.hooks, delta_layer)
    doc = parse_document(in_order(raw))
    assert doc.sites["tap"].component == pair.hooks
    hooked = _logits(PointExecutor, raw, hooks_qwen)
    traced = _logits(TracePointExecutor, raw, trace_qwen)
    sweep.assert_same(hooked, traced, f"{pair.hooks!r} interchange, both engines")
    clean = sweep.make_executor(
        PointExecutor,
        sweep.read_doc("lm_head", None) | {"model": raw["model"]},
        hooks_qwen,
        rows=ROWS,
        with_cf=False,
    ).read_value("r")
    assert float((hooked - clean).abs().max()) > 1e-6, "the interchange moved nothing"


@pytest.mark.parametrize("pair", ALIASED, ids=lambda p: p.hooks)
def test_each_engine_translates_the_name_to_its_own_mechanism(
    hooks_qwen, trace_qwen, delta_layer, pair
):
    site = resolve_site(hooks_qwen, doc_site(pair.hooks, delta_layer))
    trace_site = resolve_site(trace_qwen, doc_site(pair.hooks, delta_layer))
    assert site.component == trace_site.component == pair.hooks
    assert site.kind == trace_site.kind  # the shared resolution is one
    executor = sweep.make_executor(
        TracePointExecutor,
        _intervention(pair.hooks, delta_layer),
        trace_qwen,
        rows=ROWS,
        with_cf=True,
    )
    tap = executor._wrap(trace_site)
    if site.kind == "delta":
        # the reference engine swaps a kernel global; nnsight drills a line
        assert tap.source is not None and tap.source.module == "linear_attn"
    else:
        # a module boundary: a hook on one engine, an envoy on the other
        assert site.kind in ("in", "out") and tap.source is None


def doc_site(component: str, layer: int):
    from causalab.protocol.schema import SiteSpec

    return SiteSpec(component=component, layers=(layer,))


@pytest.mark.parametrize("pair", ALIASED, ids=lambda p: p.nnsight)
def test_the_retired_spelling_is_the_same_document(hooks_qwen, delta_layer, pair):
    """A document authored with the nnsight spelling parses to the canonical
    name and canonicalizes to the same bytes — the alias is a courtesy at the
    door, and the document that runs is one document."""
    register_model(hooks_qwen.info)
    old = _intervention(pair.nnsight, delta_layer)
    new = _intervention(pair.hooks, delta_layer)
    # the parsed sites are one site (`raw` keeps the authored text, by design)
    assert parse_document(in_order(old)).sites == parse_document(in_order(new)).sites
    # the canonical form, on a document whose data the test env resolves
    old_c, new_c = base_doc(), base_doc()
    for raw, component in ((old_c, pair.nnsight), (new_c, pair.hooks)):
        raw["model"]["key"] = TINY_QWEN35_MOE
        raw["method"]["sites"]["tgt"] = {
            "component": component,
            "layers": [delta_layer],
        }
    assert canonicalize(in_order(old_c), ENV) == canonicalize(in_order(new_c), ENV)
    assert DEPRECATED_COMPONENTS[pair.nnsight] == pair.hooks


# --------------------------------------------------------------------------- #
# the mutation: the typed pairs are two names, and the boundary bites
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("pair", TYPED, ids=lambda p: p.nnsight)
def test_a_typed_pair_is_not_redirected_and_would_not_tile_silently(
    hooks_qwen, trace_qwen, delta_layer, pair
):
    register_model(hooks_qwen.info)
    raw = _intervention(pair.nnsight, delta_layer)
    doc = parse_document(in_order(raw))
    assert doc.sites["tap"].component == pair.nnsight  # not folded
    # the reference engine refuses the nnsight face by name — the seam holds
    with pytest.raises(ProtocolError, match="nnsight engine"):
        sweep.make_executor(
            PointExecutor, raw, hooks_qwen, rows=ROWS[:1], with_cf=True
        ).read_value("logits")
    # and the two faces are different tensors: shapes disagree, so an alias
    # would have handed one engine's tensor to the other engine's math
    hooked = sweep.make_executor(
        PointExecutor,
        sweep.read_doc(pair.hooks, delta_layer, pos="all") | {"model": raw["model"]},
        hooks_qwen,
        rows=ROWS[:1],  # one row: a whole-sequence read is dense, not ragged
        with_cf=False,
    ).read_value("r")
    traced = sweep.make_executor(
        TracePointExecutor,
        sweep.read_doc(pair.nnsight, delta_layer, pos="all") | {"model": raw["model"]},
        trace_qwen,
        rows=ROWS[:1],
        with_cf=False,
    ).read_value("r")
    assert hooked.shape != traced.shape, (pair, hooked.shape, traced.shape)
    # the declared relation, and only it, lines them up
    left, right = sweep.align_delta_pair(hooked, traced, pair.hooks, hooks_qwen.info)
    sweep.assert_same(
        left, right, f"{pair.hooks!r} vs {pair.nnsight!r} after {pair.relation}"
    )
