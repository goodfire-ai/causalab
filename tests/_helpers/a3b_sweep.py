"""The Qwen3.6-35B-A3B hookpoint sweep: one table, two tiers.

The smoke tier runs it on ``tiny-random/qwen3.5-moe`` (the A3B architecture in
miniature — a hybrid Gated-DeltaNet/full-attention tower with a sparse MoE plus
shared expert in every layer) and the golden tier on the real checkpoint. Both
read the partition and the document builders from here, so "which hookpoints
exist" is answered once and a component cannot be exercised at one tier and
quietly skipped at the other.

The partition is read off the capability registry (``registry.CAPABILITIES``),
the same rows the engines' own declarations are generated from:

* **shared** — both engines serve it, so the agreement claim is the ordinary
  one: same document, same numbers;
* **hooks-only** — ``delta_query`` / ``delta_key`` (post GVA tiling) and the
  per-step ``delta_state``, which the reference engine reaches by swapping the
  modeling file's kernel globals and the nnsight engine serves in another
  shape or at another time;
* **nnsight-only** — their pre-tiling / per-chunk faces ``deltanet_query`` /
  ``deltanet_key`` / ``deltanet_state`` and ``expert_permutation``, ``.source``
  lines inside a fused forward that no hook can reach.

📐 The two single-engine sets are not two blind spots: measured on the fixture,
they name the *same physical tensors* through the two different mechanisms, and
the registry declares how each pair lines up (``registry.BACKEND_PAIRS``, read
here as `DELTA_FAMILY_PAIRS`). The other eight DeltaNet tensors carry one
name served by both engines (`SHARED_LINEAR_ONLY`), which is
ordinary cross-engine agreement for 30 of the target's 40 layers.

``mlp_activation`` and ``mlp_neuron_output`` are absent from the A3B.
Each MLP is a sparse MoE block. ``expert_neuron_output`` exposes its routed
neuron products, and ``shared_expert_activation`` exposes its shared products.
"""

from __future__ import annotations

from typing import Any

import torch

from causalab.protocol.rules.errors import ValidationError
from causalab.protocol.registry import (
    BACKEND_PAIRS,
    CAPABILITIES,
    DOCS_TABLE_MODEL,
    backend_pair,
    component_shape,
    get_model_info,
)
from causalab.protocol.schema import COMPONENTS, LAYERLESS_COMPONENTS

__all__ = [
    "ABSENT_ON_A3B",
    "ATOL",
    "DELTA_FAMILY_PAIRS",
    "HOOKS_ONLY",
    "NNSIGHT_ONLY",
    "SHARED_ANY_STREAM",
    "SHARED_FULL_ONLY",
    "SHARED_LAYERLESS",
    "SHARED_LINEAR_ONLY",
    "READ_ONLY",
    "SWAP_ONLY_WRITES",
    "WHOLE_TENSOR_ONLY",
    "align_delta_pair",
    "assert_same",
    "default_pos",
    "interchange_doc",
    "make_executor",
    "read_doc",
    "stream_layers",
    "write_cases",
]

#: Reads agree to this tolerance when both engines run fp32 eager on the same
#: device: the same kernels in a different order of capture, so anything larger
#: is an executor bug rather than float noise.
ATOL = 1e-5

# --------------------------------------------------------------------------- #
# the partition — read off the capability registry, not restated
# --------------------------------------------------------------------------- #
#
# Every bucket below is a query over ``registry.CAPABILITIES`` (which engines
# serve a component, which stream it needs, its write policy) and over
# ``component_shape`` on the A3B entry (whether the architecture has the tensor
# at all, and whether it has a contract form). The census guard in
# tests/protocol/test_vocabulary_census.py asserts the buckets equal the rows;
# ``test_the_buckets_match_what_the_engines_declare`` asserts them equal to the
# engines' declarations — which are themselves generated from the rows.

_A3B = get_model_info(DOCS_TABLE_MODEL)
_BOTH = frozenset({"pytorch_hooks", "nnsight"})


def _exists_on_a3b(component: str) -> bool:
    """Whether the A3B has the tensor at all: its entry sizes it, or refuses."""
    try:
        component_shape(_A3B, component)
    except ValidationError:
        return False
    return True


def _rows(*, served: frozenset[str], stream: object = "any") -> tuple[str, ...]:
    return tuple(
        c
        for c in COMPONENTS
        if CAPABILITIES[c].reads == served
        and (stream == "any" or CAPABILITIES[c].stream == stream)
    )


#: Layer-less components both engines serve.
SHARED_LAYERLESS: tuple[str, ...] = tuple(
    c for c in _rows(served=_BOTH) if c in LAYERLESS_COMPONENTS
)

#: Both engines, and the component exists in **either** block type — the
#: residual-stream boundaries and the whole MoE surface, which every layer of
#: the A3B carries.
SHARED_ANY_STREAM: tuple[str, ...] = tuple(
    c
    for c in _rows(served=_BOTH, stream=None)
    if c not in LAYERLESS_COMPONENTS and _exists_on_a3b(c)
)

#: Both engines, but only at a full-attention layer — 10 of the target's 40.
SHARED_FULL_ONLY: tuple[str, ...] = _rows(served=_BOTH, stream="full_attention")

#: Both engines, but only at a Gated DeltaNet layer — 30 of the target's 40:
#: the DeltaNet module boundaries and kernel boundary under their one name,
#: which the reference engine reaches by hooks and kernel-global swaps
#: and the nnsight engine by envoys and `.source` lines. Same document, same
#: numbers — a black-box test of the shared name, run as ordinary parity.
SHARED_LINEAR_ONLY: tuple[str, ...] = _rows(served=_BOTH, stream="linear_attention")

#: The reference engine's Gated DeltaNet interior — linear-attention layers only.
HOOKS_ONLY: tuple[str, ...] = _rows(served=frozenset({"pytorch_hooks"}))

#: The nnsight engine's fused-forward interiors.
NNSIGHT_ONLY: tuple[str, ...] = _rows(served=frozenset({"nnsight"}))

#: In the vocabulary, absent from this architecture — see the module docstring.
ABSENT_ON_A3B: tuple[str, ...] = tuple(c for c in COMPONENTS if not _exists_on_a3b(c))

#: Components no write may target (the rows' ``writes is None``), so the
#: sweep's write half skips them by table rather than by exception.
READ_ONLY: frozenset[str] = frozenset(
    c for c, row in CAPABILITIES.items() if row.writes is None
)

#: Components a write may only **replace** — the integer routing table and the
#: normalized attention pattern. The sweep writes `swap` everywhere, so these
#: need no special case; the set is here because the docs table cites it.
SWAP_ONLY_WRITES: frozenset[str] = frozenset(
    c for c, row in CAPABILITIES.items() if row.writes == frozenset({"swap"})
)

#: Components that can only be addressed whole. 📐 The attention matrix has
#: **two** position axes (query and key), so an integer position is ambiguous
#: between them and the executor refuses it by shape — no component name
#: appears in that refusal, which is what makes it a rule rather than a case.
#: The sweep honours the rule rather than skipping the components.
WHOLE_TENSOR_ONLY: frozenset[str] = frozenset(
    c
    for c in COMPONENTS
    if _exists_on_a3b(c) and not component_shape(_A3B, c).has_contract_form
)


def default_pos(component: str) -> object:
    """The position spec the sweep addresses ``component`` with."""
    return "all" if component in WHOLE_TENSOR_ONLY else -1


#: The DeltaNet tensors the two engines reach by **different** captures — the
#: typed backend pairs of the registry (``registry.BACKEND_PAIRS``), read here
#: rather than declared: ``(hooks spelling, nnsight spelling, relation)`` for
#: every pair that is *not* an alias. 📐 Measured on ``tiny-random/qwen3.5-moe``
#: (2026-08-28); the relations and the chunk length are the registry's rows.
#: The eight ``identical`` pairs are one name each and are
#: exercised as ordinary shared components (`SHARED_LINEAR_ONLY`).
DELTA_FAMILY_PAIRS: tuple[tuple[str, str, str], ...] = tuple(
    (pair.hooks, pair.nnsight, pair.relation)
    for pair in BACKEND_PAIRS
    if not pair.aliased
)


# --------------------------------------------------------------------------- #
# documents
# --------------------------------------------------------------------------- #


def _data(with_cf: bool) -> dict[str, Any]:
    data: dict[str, Any] = {"base": {"dataset": "inline", "field": "input"}}
    if with_cf:
        data["counterfactual"] = {
            "dataset": "inline",
            "field": "counterfactual_inputs[0]",
        }
    return data


def read_doc(
    component: str, layer: int | None, *, pos: object = -1, head: int | None = None
) -> dict[str, Any]:
    """Read one site on the base input and save it."""
    site: dict[str, Any] = {"component": component}
    if layer is not None:
        site["layers"] = layer
    if head is not None:
        site["head"] = head
    return {
        "header": {"protocol_version": "4"},
        "model": {"key": "test", "revision": "main"},
        "data": _data(with_cf=False),
        "method": {
            "intervened_models": {"original": {"input": "base", "reads": ["r"]}},
            "sites": {"tap": site},
            "reads": {"r": {"site": "tap", "pos": pos}},
            "save": [{"read": "r", "model": "original", "file_path": "a.safetensors"}],
        },
    }


def interchange_doc(
    component: str, layer: int | None, *, pos: object = -1
) -> dict[str, Any]:
    """Read the site on the counterfactual, swap it into the base forward, read
    the patched logits — the intervention whose downstream effect must agree."""
    site: dict[str, Any] = {"component": component}
    if layer is not None:
        site["layers"] = layer
    return {
        "header": {"protocol_version": "4"},
        "model": {"key": "test", "revision": "main"},
        "data": _data(with_cf=True),
        "method": {
            "intervened_models": {
                "original_counterfactual": {
                    "input": "counterfactual",
                    "reads": ["v_cf"],
                },
                "patched": {"input": "base", "reads": ["logits"], "writes": ["patch"]},
            },
            "sites": {"tap": site, "head": {"component": "lm_head"}},
            "reads": {
                "v_cf": {"site": "tap", "pos": pos},
                "logits": {"site": "head", "pos": -1},
            },
            "writes": {"patch": {"site": "tap", "pos": pos, "do": {"swap": "v_cf"}}},
            "save": [
                {"read": "logits", "model": "patched", "file_path": "l.safetensors"}
            ],
        },
    }


def make_executor(executor_cls, doc_raw, bundle, *, rows, with_cf: bool):
    """The same document driven through either engine's executor."""
    from causalab.protocol.schema import parse_document
    from causalab.protocol.rules.document import validate_document

    from tests.protocol._docs import in_order

    doc = parse_document(in_order(doc_raw))
    validate_document(doc, engine_is_local=True)
    role_rows = {"base": rows}
    role_fields = {"base": "input"}
    if with_cf:
        role_rows["counterfactual"] = rows
        role_fields["counterfactual"] = "counterfactual_inputs[0]"
    return executor_cls(
        doc,
        bundle,
        role_rows=role_rows,
        role_fields=role_fields,
        load_tensors=lambda path: (_ for _ in ()).throw(KeyError(path)),
    )


# --------------------------------------------------------------------------- #
# comparison
# --------------------------------------------------------------------------- #


def assert_same(a: torch.Tensor, b: torch.Tensor, what: str, *, atol: float = ATOL):
    """Shape-then-value agreement; integers must match exactly.

    Integer components (``input_ids``, ``expert_idx``, ``expert_permutation``)
    carry labels, not measurements: a tolerance on them would let a routing
    table that sends a token to a different expert pass.
    """
    assert a.shape == b.shape, f"{what}: {tuple(a.shape)} != {tuple(b.shape)}"
    if not a.dtype.is_floating_point:
        assert torch.equal(a, b), f"{what}: integer values differ"
        return
    diff = (a.double() - b.double()).abs().max().item()
    assert torch.allclose(a, b, atol=atol, rtol=0), (
        f"{what}: max abs diff {diff:.3e} exceeds {atol}"
    )


def align_delta_pair(hooks_value, trace_value, hooks_component: str, info):
    """Bring the two engines' captures of one DeltaNet tensor into one frame.

    The relation is the registry's (``registry.backend_pair``), never this
    helper's: it owns the two tensor transforms and reads which one applies —
    so a wrong address cannot be massaged into agreement here, and a pair the
    registry calls ``identical`` is compared as-is.
    """
    pair = backend_pair(hooks_component)
    relation = pair.relation
    if relation == "identical":
        return hooks_value, trace_value
    if relation == "gva_tile":
        # `delta_*` is the kernel's argument, already tiled to the value-head
        # count; `deltanet_*` is the pre-`repeat_interleave` projection in
        # key-head space. The tile is over the HEAD axis of an unflattened view.
        h_k, h_v = info.linear_num_key_heads, info.linear_num_value_heads
        d_k = info.linear_key_head_dim
        b, s = trace_value.shape[0], trace_value.shape[1]
        tiled = (
            trace_value.reshape(b, s, h_k, d_k)
            .repeat_interleave(h_v // h_k, dim=2)
            .reshape(b, s, h_v * d_k)
        )
        return hooks_value, tiled
    if relation == "chunk_boundary":
        # `delta_state` is per step; `deltanet_state` per 64-token chunk. The
        # chunk's state is the step-state at the chunk's last position (the
        # final chunk may be partial, hence the clamp).
        n_chunks, seq = trace_value.shape[1], hooks_value.shape[1]
        assert pair.chunk is not None
        idx = [min(pair.chunk * (i + 1) - 1, seq - 1) for i in range(n_chunks)]
        return hooks_value[:, idx].reshape(trace_value.shape), trace_value
    raise AssertionError(f"unknown relation {relation!r}")


def stream_layers(bundle) -> tuple[int, int]:
    """``(first linear-attention layer, first full-attention layer)`` of the
    loaded tower — read off the model rather than hardcoded, because the
    fixture's hybrid schedule and the real A3B's are different orders of the
    same two block types."""
    n = len(bundle.model.model.layers)
    streams = [bundle.stream_at(i) for i in range(n)]
    assert "linear_attention" in streams, f"no Gated DeltaNet layer in {streams}"
    assert "full_attention" in streams, f"no full-attention layer in {streams}"
    return streams.index("linear_attention"), streams.index("full_attention")


def write_cases(components: tuple[str, ...]) -> tuple[str, ...]:
    """The sweep's write half: everything a write may target."""
    return tuple(c for c in components if c not in READ_ONLY)


def coverage_partition() -> dict[str, tuple[str, ...]]:
    """Every component in the vocabulary, claimed by exactly one bucket — what
    the completeness guard checks the sweep against."""
    return {
        "shared_layerless": SHARED_LAYERLESS,
        "shared_any_stream": SHARED_ANY_STREAM,
        "shared_full_only": SHARED_FULL_ONLY,
        "shared_linear_only": SHARED_LINEAR_ONLY,
        "hooks_only": HOOKS_ONLY,
        "nnsight_only": NNSIGHT_ONLY,
        "absent_on_a3b": ABSENT_ON_A3B,
    }


def unclaimed_components() -> tuple[str, ...]:
    claimed: set[str] = set()
    for group in coverage_partition().values():
        claimed |= set(group)
    return tuple(c for c in COMPONENTS if c not in claimed)


def double_claimed_components() -> tuple[str, ...]:
    seen: set[str] = set()
    twice: set[str] = set()
    for group in coverage_partition().values():
        for component in group:
            if component in seen:
                twice.add(component)
            seen.add(component)
    return tuple(sorted(twice))


def layerless(component: str) -> bool:
    return component in LAYERLESS_COMPONENTS
