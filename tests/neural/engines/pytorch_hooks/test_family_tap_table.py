"""The per-family tap table.

GPT-2 fuses q, k and v into one ``c_attn``; llama keeps three projections;
qwen3.5-moe normalizes q and k before RoPE and packs a gate beside q. Before
this table the interior q/k/v were *refused by name* on the fused family. Now
each interior component's row carries, per family, the mixer child to tap and
how that child's tensor packs the value (``registry.Capability.overrides``),
and the site resolver reads it — so a split-projection model and a
fused-projection model expose **equivalent** logical sites.

What each group of tests is for:

* **oracle equivalence (the acceptance)** — on ``tiny-gpt2`` (fused) and
  ``tiny-llama`` (split) the logical ``query|key|value`` read equals an oracle
  computed from the module *weights* (``x @ W_q`` on llama;
  ``(x @ W_c_attn + b)[..., block]`` on gpt2), and a write through the logical
  site moves the logits exactly as the oracle's direct write to the underlying
  tensor does — both within the write oracle's declared tolerance
  (``test_write_oracle.py``: atol 1e-5, rtol 1e-4);
* **same site, both families** — same component names, same logical shape
  ``(b, s, H·d)``, same read/write mechanism (a module-output tap);
* **the qwen norms** stay as measured: ``(b, s, H, d)``, a kept head axis;
* **load and run agree** on every (family × interior component) cell of the
  three fixtures — the offline predicates decide what the run decides;
* **fail-closed, valid work still passes**: a
  fused-projection family *without* a row keeps the earlier refusal, text
  pinned (the refusal snapshot's retired entry 23); a family without a row
  whose mixer is unambiguous is served by measurement, exactly as before.

📐 Measured on the fixtures: tiny-gpt2 H 4, d 8, ``c_attn`` ``(1, s, 96)``
= ``[q | k | v]`` in 32-wide blocks (``split_size = 32``); tiny-llama H 4, d 4,
``q_proj`` ``(1, s, 16)``; qwen3.5-moe H 8, H_kv 4, d 32, ``q_norm`` ``(1, 5, 8, 32)``.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest
import torch

from causalab.neural.engines.pytorch_hooks.loading import ModelBundle, load_model
from causalab.neural.shared.encoding import encode
from causalab.neural.shared.sites import resolve_site
from causalab.protocol.rules.errors import ProtocolError
from causalab.protocol.registry import (
    CAPABILITIES,
    INTERIOR_ROWS,
    component_shape,
    component_width,
    unavailable_at_load,
)
from causalab.protocol.schema import PROTOCOL_VERSION, SiteSpec

from tests.neural.engines.pytorch_hooks._drive import base_data_section, executor_for
from tests.neural.engines.pytorch_hooks.conftest import TINY_GPT2, TINY_LLAMA
from tests.protocol._docs import saved

pytestmark = pytest.mark.smoke

#: The write oracle's declared tolerance (test_write_oracle.py).
TOL = dict(atol=1e-5, rtol=1e-4)

TEXT = "the quick brown fox jumps"
CF_TEXT = "a slow green turtle sleeps deeply"

QKV = ("attention_query_pre_rope", "attention_key_pre_rope", "attention_value_states")

#: 📐 GPT-2's ``GPT2Attention.forward``: ``query, key, value =
#: self.c_attn(hidden_states).split(self.split_size, dim=2)`` — the oracle's
#: own statement of the block order, independent of the registry row.
GPT2_BLOCK = {"attention_query_pre_rope": 0, "attention_key_pre_rope": 1}
GPT2_BLOCK["attention_value_states"] = 2
#: The llama tree's separate projections — the oracle's, not the row's.
LLAMA_PROJECTION = {
    "attention_query_pre_rope": "q_proj",
    "attention_key_pre_rope": "k_proj",
    "attention_value_states": "v_proj",
}

#: Layer 0 on both fixtures, deliberately: a swap of the *queries* at position
#: p changes only position p's mixer output, which reaches the last position's
#: logits only through later layers — on the 2-layer tiny-llama a layer-1
#: query swap is invisible to the next-token logits (measured: max |Δ| = 0.0).
LAYER = 0
QWEN_FULL_ATTENTION_LAYER = 3


def _bundle(family: str) -> ModelBundle:
    return load_model(TINY_GPT2 if family == "gpt2" else TINY_LLAMA)


def _inputs(bundle: ModelBundle, text: str) -> dict[str, torch.Tensor]:
    batch = encode(bundle.tokenizer, [text])
    return {"input_ids": batch.input_ids, "attention_mask": batch.attention_mask}


def _mixer_input(bundle: ModelBundle, layer: int, text: str) -> torch.Tensor:
    """What the mixer consumes at ``layer`` — captured with a raw pre-hook
    (``hidden_states`` arrives positionally on GPT-2, as a keyword on llama)."""
    seen: dict[str, torch.Tensor] = {}

    def pre_hook(_m: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        x = args[0] if args else kwargs["hidden_states"]
        seen["x"] = x.detach().clone()

    handle = bundle.mixer_at(layer).register_forward_pre_hook(
        pre_hook, with_kwargs=True
    )
    try:
        with torch.no_grad():
            bundle.model(**_inputs(bundle, text))
    finally:
        handle.remove()
    return seen["x"]


def _oracle_value(
    bundle: ModelBundle, layer: int, component: str, x: torch.Tensor
) -> torch.Tensor:
    """The logical q/k/v from the module **weights** — the independent oracle."""
    attn = bundle.mixer_at(layer)
    width = bundle.info.num_heads * bundle.info.head_dim
    if bundle.is_gpt2_family:
        # Conv1D: weight is (in, out); the fused output is [q | k | v] blocks
        fused = x @ attn.c_attn.weight + attn.c_attn.bias
        k = GPT2_BLOCK[component]
        return fused[..., k * width : (k + 1) * width]
    projection = getattr(attn, LLAMA_PROJECTION[component])
    return torch.nn.functional.linear(x, projection.weight, projection.bias)


def _oracle_columns(bundle: ModelBundle, component: str) -> slice:
    """Where the logical value sits in the underlying module's output."""
    if not bundle.is_gpt2_family:
        return slice(None)
    width = bundle.info.num_heads * bundle.info.head_dim
    k = GPT2_BLOCK[component]
    return slice(k * width, (k + 1) * width)


def _underlying_module(bundle: ModelBundle, layer: int, component: str) -> Any:
    attn = bundle.mixer_at(layer)
    if bundle.is_gpt2_family:
        return attn.c_attn
    return getattr(attn, LLAMA_PROJECTION[component])


def _read_doc(component: str, layer: int, *, head: int | None = None) -> dict:
    site: dict[str, Any] = {"component": component, "layers": [layer]}
    if head is not None:
        site["head"] = head
    return {
        "header": {"protocol_version": PROTOCOL_VERSION},
        "model": {"key": "test", "revision": "main"},
        "data": base_data_section(with_counterfactual=False),
        "method": {
            "intervened_models": {"original": {"input": "base", "reads": ["r"]}},
            "sites": {"tap": site},
            "reads": {"r": {"site": "tap", "pos": "all"}},
            "save": [saved("r", "original", "r.safetensors")],
        },
    }


def _swap_doc(component: str, layer: int, pos: int) -> dict:
    """Swap the counterfactual's logical value into base at ``pos``; read the
    patched and the clean next-token logits."""
    return {
        "header": {"protocol_version": PROTOCOL_VERSION},
        "model": {"key": "test", "revision": "main"},
        "data": base_data_section(with_counterfactual=True),
        "method": {
            "intervened_models": {
                "original_counterfactual": {
                    "input": "counterfactual",
                    "reads": ["v_cf"],
                },
                "original_base": {"input": "base", "reads": ["clean"]},
                "patched": {"input": "base", "reads": ["after"], "writes": ["patch"]},
            },
            "sites": {
                "tap": {"component": component, "layers": [layer]},
                "lm_head": {"component": "lm_head"},
            },
            "reads": {
                "v_cf": {"site": "tap", "pos": {"index": pos}},
                "clean": {"site": "lm_head", "pos": {"index": -1}},
                "after": {"site": "lm_head", "pos": {"index": -1}},
            },
            "writes": {
                "patch": {"site": "tap", "pos": {"index": pos}, "do": {"swap": "v_cf"}}
            },
            "save": [
                saved("after", "patched", "p.safetensors"),
                saved("clean", "original_base", "c.safetensors"),
            ],
        },
    }


def _next_token_logits(
    bundle: ModelBundle, text: str, write: Any = None
) -> torch.Tensor:
    """Last-position logits, optionally under one raw forward hook ``write``
    (a ``(module, fn)`` pair) — the oracle's direct write."""
    handle = None
    if write is not None:
        module, fn = write
        handle = module.register_forward_hook(fn)
    try:
        with torch.no_grad():
            logits = bundle.model(**_inputs(bundle, text)).logits
    finally:
        if handle is not None:
            handle.remove()
    return logits[:, -1, :]


# --------------------------------------------------------------------------- #
# the acceptance: oracle equivalence on a fused and a split projection
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("family", ["gpt2", "llama"])
@pytest.mark.parametrize("component", QKV)
def test_the_logical_read_equals_the_weight_oracle(family: str, component: str):
    """The logical site's read is the projection of what the mixer consumes,
    computed from the weights by hand — on GPT-2 the right block of the fused
    output, on llama the bare projection. Without the table this refused on
    GPT-2 (snapshot entry 23)."""
    bundle = _bundle(family)
    executor = executor_for(_read_doc(component, LAYER), bundle, base_texts=[TEXT])
    have = executor.read_value("r")
    x = _mixer_input(bundle, LAYER, TEXT)
    want = _oracle_value(bundle, LAYER, component, x)
    assert tuple(have.shape) == tuple(want.shape)  # (1, s, H·d): the same shape
    torch.testing.assert_close(have, want, **TOL)


@pytest.mark.parametrize("family", ["gpt2", "llama"])
@pytest.mark.parametrize("component", QKV)
def test_the_logical_write_equals_the_oracles_direct_write(family: str, component: str):
    """A swap through the logical site moves the logits exactly as the oracle's
    direct write into the underlying tensor (the fused output's block on
    GPT-2, the projection's output on llama) — and moves them at all."""
    bundle = _bundle(family)
    pos = 1
    executor = executor_for(
        _swap_doc(component, LAYER, pos),
        bundle,
        base_texts=[TEXT],
        counterfactual_texts=[CF_TEXT],
    )
    have = executor.read_value("after")[:, 0, :]

    cf_value = _oracle_value(
        bundle, LAYER, component, _mixer_input(bundle, LAYER, CF_TEXT)
    )[:, pos, :]
    columns = _oracle_columns(bundle, component)

    def direct_write(_m: Any, _i: Any, out: torch.Tensor) -> torch.Tensor:
        out = out.clone()
        out[:, pos, columns] = cf_value
        return out

    module = _underlying_module(bundle, LAYER, component)
    want = _next_token_logits(bundle, TEXT, (module, direct_write))
    clean = _next_token_logits(bundle, TEXT)
    assert float((want - clean).abs().max()) > 1e-6, "the oracle write is vacuous"
    torch.testing.assert_close(have, want, **TOL)
    torch.testing.assert_close(executor.read_value("clean")[:, 0, :], clean, **TOL)


def test_a_head_on_the_fused_projection_is_one_heads_block_of_the_right_split():
    """``head`` slices inside the logical value, not inside ``c_attn``'s whole
    output: head 1 of the *keys* is columns ``[H·d + d, H·d + 2d)`` of the fused
    tensor, which is what the oracle says and what the fused-blocks packing
    gives. Wrong packing (fused axis inside the head axis) would pass a shape
    check and read the wrong columns."""
    bundle = _bundle("gpt2")
    d = bundle.info.head_dim
    executor = executor_for(
        _read_doc("attention_key_pre_rope", LAYER, head=1), bundle, base_texts=[TEXT]
    )
    have = executor.read_value("r")
    x = _mixer_input(bundle, LAYER, TEXT)
    keys = _oracle_value(bundle, LAYER, "attention_key_pre_rope", x)
    torch.testing.assert_close(have, keys[..., d : 2 * d], **TOL)


# --------------------------------------------------------------------------- #
# the same logical site on both families
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("component", QKV)
def test_the_two_families_expose_the_same_logical_site(component: str):
    """Same name, same logical shape, same mechanism: a module-output tap whose
    contract form is ``(b, s, heads·d)`` in the component's own head space.
    Only the *native packing* differs, and that is the row's to say."""
    gpt2, llama = _bundle("gpt2"), _bundle("llama")
    sites = [
        resolve_site(b, SiteSpec(component=component, layers=(LAYER,)))
        for b in (gpt2, llama)
    ]
    assert [s.kind for s in sites] == ["out", "out"]
    for bundle, site in zip((gpt2, llama), sites, strict=True):
        value = component_shape(bundle.info, component)
        assert (
            site.shape.width == value.width == component_width(bundle.info, component)
        )
        assert site.shape.head_space == value.head_space
        executor = executor_for(_read_doc(component, LAYER), bundle, base_texts=[TEXT])
        read = executor.read_value("r")
        assert read.dim() == 3 and read.shape[-1] == value.width
    # the family-independent half really is: identical axis kinds once the
    # fused axis (GPT-2's packing) is set aside
    kinds = [[a.kind for a in s.shape.axes if a.kind != "fused"] for s in sites]
    assert kinds[0] == kinds[1] == ["batch", "position", "head", "feature"]
    assert sites[0].shape.fused_index is not None  # GPT-2: one of three blocks
    assert sites[1].shape.fused_index is None  # llama: the whole projection


def test_the_qwen_norms_keep_their_head_axis_as_measured(qwen35moe_bundle):
    """📐 ``q_norm``/``k_norm`` emit ``(b, s, H, d)`` before RoPE; the row says
    ``head_axis`` and the resolved shape is rank 4 with the head axis kept —
    unchanged by this table."""
    for component, module_name in (
        ("attention_query_pre_rope", "q_norm"),
        ("attention_key_pre_rope", "k_norm"),
    ):
        address = CAPABILITIES[component].address_on(qwen35moe_bundle.info)
        assert address is not None
        assert address["module"] == module_name and address["packing"] == "head_axis"
        site = resolve_site(
            qwen35moe_bundle,
            SiteSpec(component=component, layers=(QWEN_FULL_ATTENTION_LAYER,)),
        )
        attn = qwen35moe_bundle.mixer_at(QWEN_FULL_ATTENTION_LAYER)
        assert site.module is getattr(attn, module_name)
        assert site.shape.native_rank == 4 and site.shape.flat_inner is False
    gate = CAPABILITIES["attention_gate"].address_on(qwen35moe_bundle.info)
    assert gate == {
        "module": "q_proj",
        "packing": "fused_heads",
        "splits": 2,
        "split": 1,
    }


# --------------------------------------------------------------------------- #
# load and run agree on every cell
# --------------------------------------------------------------------------- #


def _run_refuses(bundle: ModelBundle, component: str, layer: int) -> bool:
    try:
        resolve_site(bundle, SiteSpec(component=component, layers=(layer,)))
    except ProtocolError as err:
        assert err.reason == "component_unavailable", err
        return True
    return False


def test_load_and_run_agree_on_every_family_by_component_cell(qwen35moe_bundle):
    """With the rows carrying per-family facts, ``validate``
    (``unavailable_at_load``) refuses offline exactly what the run refuses —
    on every (fixture × interior component) cell. Without the table
    ``predicate_holds`` returned ``None`` for ``gated_attention`` and a
    document naming ``attention_gate`` on GPT-2 validated, then failed on the
    accelerator."""
    cells: dict[tuple[str, str], tuple[bool, bool]] = {}
    for bundle, layer in (
        (_bundle("gpt2"), LAYER),
        (_bundle("llama"), LAYER),
        (qwen35moe_bundle, QWEN_FULL_ATTENTION_LAYER),
    ):
        for component in INTERIOR_ROWS:
            load = unavailable_at_load(bundle.info, component) is not None
            run = _run_refuses(bundle, component, layer)
            cells[(bundle.info.family or "?", component)] = (load, run)
    assert all(load == run for load, run in cells.values()), cells
    refused = {cell for cell, (load, _run) in cells.items() if load}
    assert refused == {("gpt2", "attention_gate"), ("llama", "attention_gate")}
    assert len(cells) == 3 * len(INTERIOR_ROWS)


# --------------------------------------------------------------------------- #
# fail-closed: the refusal survives for a fused family without a row, and a
# family without a row is served by measurement where that is unambiguous
# --------------------------------------------------------------------------- #


def _without_a_row(bundle: ModelBundle, family: str | None) -> ModelBundle:
    """The same loaded modules under a family the table has not met."""
    return dataclasses.replace(
        bundle, info=dataclasses.replace(bundle.info, family=family)
    )


@pytest.mark.parametrize("family", [None, "gpt2-lookalike"])
@pytest.mark.parametrize("component", QKV)
def test_a_fused_family_without_a_row_keeps_the_refusal(
    family: str | None, component: str
):
    """GPT-2's module tree under a family the table has not met: which block
    of ``c_attn`` is q is not inferred — refused, with the pre-PR text (the
    snapshot's retired entry 23) plus where the row goes."""
    bundle = _without_a_row(_bundle("gpt2"), family)
    with pytest.raises(ProtocolError) as excinfo:
        resolve_site(bundle, SiteSpec(component=component, layers=(LAYER,)))
    message = str(excinfo.value)
    assert excinfo.value.reason == "component_unavailable"
    assert (
        f"component {component!r} needs separate q/k/v projections, and this "
        "mixer fuses them into one 'c_attn'"
    ) in message
    assert "per-family tap table" in message
    assert (
        "'attention_premix' and 'attention_output' read on this family today" in message
    )
    assert f"family {family!r} has no row" in message


def test_the_same_modules_with_the_row_resolve():
    """The valid twin of the refusal above: the identical loaded modules, keyed
    by the family the config declares, resolve — the row is the only difference."""
    bundle = _bundle("gpt2")
    assert bundle.info.family == "gpt2"
    for component in QKV:
        site = resolve_site(bundle, SiteSpec(component=component, layers=(LAYER,)))
        assert site.module is bundle.mixer_at(LAYER).c_attn


def test_a_split_family_without_a_row_is_served_by_measurement():
    """Today's behaviour for a family the table has not met, kept exactly:
    llama's bare projections are unambiguous, so the same site resolves with
    or without the row, to the same module with the same shape."""
    bundle = _bundle("llama")
    unknown = _without_a_row(bundle, None)
    for component in QKV:
        with_row = resolve_site(bundle, SiteSpec(component=component, layers=(LAYER,)))
        measured = resolve_site(unknown, SiteSpec(component=component, layers=(LAYER,)))
        assert measured.module is with_row.module
        assert measured.shape == with_row.shape
    with pytest.raises(ProtocolError, match="computes no output gate"):
        resolve_site(unknown, SiteSpec(component="attention_gate", layers=(LAYER,)))


def test_a_normed_gated_family_without_a_row_is_served_by_measurement(
    qwen35moe_bundle,
):
    """And the qwen tree without its row: the norm after the projection wins
    (its output is the pre-RoPE tensor), and the gate is measured off the
    doubled q-projection — the same sites the row gives."""
    unknown = _without_a_row(qwen35moe_bundle, None)
    for component in INTERIOR_ROWS:
        spec = SiteSpec(component=component, layers=(QWEN_FULL_ATTENTION_LAYER,))
        with_row = resolve_site(qwen35moe_bundle, spec)
        measured = resolve_site(unknown, spec)
        assert measured.module is with_row.module, component
        assert measured.shape == with_row.shape, component


def test_a_row_that_disagrees_with_the_module_is_refused_by_name():
    """Fail-closed on the table itself: a llama tree keyed as ``gpt2`` has no
    ``c_attn``; a gpt2 tree keyed as ``llama`` has no ``q_proj``. Neither is
    served with a guess."""
    for source, wrong_family in (("llama", "gpt2"), ("gpt2", "llama")):
        bundle = _without_a_row(_bundle(source), wrong_family)
        with pytest.raises(ProtocolError) as err:
            resolve_site(
                bundle, SiteSpec(component="attention_query_pre_rope", layers=(LAYER,))
            )
        assert err.value.reason == "component_unavailable"
        assert "has no child of that name" in str(err.value)
