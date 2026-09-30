"""The GPT-J tree on the reference engine, against a raw-hook oracle.

GPT-J shares GPT-2's root layout (``transformer.h``, ``transformer.wte``,
``transformer.ln_f``) and differs inside the block: one norm (``ln_1``) feeds
both the attention and the MLP, and the block adds both outputs to its input
in one expression (the parallel residual,
``transformers/models/gptj/modeling_gptj.py``, ``GPTJBlock.forward``)::

    residual = hidden_states
    hidden_states = self.ln_1(hidden_states)
    attn_outputs, _ = self.attn(hidden_states=hidden_states, ...)
    feed_forward_hidden_states = self.mlp(hidden_states)
    hidden_states = attn_outputs + feed_forward_hidden_states + residual

So there is no ``block_mid`` and no ``mlp_input_norm`` on this tree, and the
GPT-2 taps (``ln_2``, ``attn.c_proj``, ``mlp.c_proj``) name modules it does
not have. The oracle below names the HF modules directly and never reads the
registry, so a wrong tap in the family adapter fails here.

📐 The fixture ``hf-internal-testing/tiny-random-GPTJForCausalLM`` (public,
ungated): 5 layers, ``n_embd`` 32, 4 heads of 8, ``rotary_dim`` 4,
``n_inner`` null (so the MLP is 4 · 32 = 128 wide), vocab 1024, fp32.
"""

from __future__ import annotations

from typing import Any, Callable

import pytest
import torch

from causalab.neural.engines.pytorch_hooks.loading import ModelBundle, load_model
from causalab.neural.shared.encoding import encode
from causalab.neural.shared.sites import inventory as loaded_inventory
from causalab.neural.shared.sites import resolve_site
from causalab.protocol.registry import (
    CAPABILITIES,
    GPTJ_TREE,
    family_for,
    identities_for,
    site_group_map,
)
from causalab.protocol.rules.errors import ProtocolError
from causalab.protocol.schema import LAYERLESS_COMPONENTS, PROTOCOL_VERSION, SiteSpec

from tests.neural.engines.pytorch_hooks._drive import base_data_section, executor_for
from tests.protocol._docs import saved

pytestmark = pytest.mark.unit

TINY_GPTJ = "hf-internal-testing/tiny-random-GPTJForCausalLM"

TEXT = "the quick brown fox jumps over"
CF_TEXT = "a slow green turtle sleeps deeply"

#: A middle layer of five, so a write there reaches the logits through later
#: layers' attention as well as through the residual.
LAYER = 2

#: What the oracle taps, per component: the HF module under the block and
#: which side of it. Written from ``modeling_gptj.py``, not from the registry.
ORACLE_TAPS: dict[str, tuple[str, str]] = {
    "block_input": ("", "in"),
    "attention_input_norm": ("ln_1", "out"),
    "attention_query_pre_rope": ("attn.q_proj", "out"),
    "attention_key_pre_rope": ("attn.k_proj", "out"),
    "attention_value_states": ("attn.v_proj", "out"),
    "attention_premix": ("attn.out_proj", "in"),
    "attention_output": ("attn", "out"),
    "mlp_input": ("mlp", "in"),
    "mlp_activation": ("mlp.fc_out", "in"),
    "mlp_neuron_output": ("mlp.fc_out", "in"),
    "mlp_output": ("mlp", "out"),
    "block_output": ("", "out"),
}
ORACLE_LAYERLESS: dict[str, str] = {
    "embeddings": "transformer.wte",
    "ln_final": "transformer.ln_f",
    "lm_head": "lm_head",
}

#: The components the GPT-J tree does not have, or that its attention does
#: not compute through the transformers attention interface.
NOT_ON_GPTJ = (
    "block_mid",
    "mlp_input_norm",
    "attention_probs",
    "attention_query",
    "attention_key",
    "attention_scores",
    "attention_z",
)


@pytest.fixture(scope="module")
def gptj() -> ModelBundle:
    return load_model(TINY_GPTJ)


def _inputs(bundle: ModelBundle, text: str) -> dict[str, torch.Tensor]:
    batch = encode(bundle.tokenizer, [text])
    return {"input_ids": batch.input_ids, "attention_mask": batch.attention_mask}


def _module(model: Any, path: str) -> torch.nn.Module:
    module = model
    for part in path.split(".") if path else ():
        module = getattr(module, part)
    return module


def _first(value: Any) -> torch.Tensor:
    return value[0] if isinstance(value, tuple) else value


def _hidden_arg(args: tuple[Any, ...], kwargs: dict[str, Any]) -> torch.Tensor:
    # GPTJBlock passes the attention its input by keyword, everything else
    # positionally
    return args[0] if args else kwargs["hidden_states"]


def _oracle_forward(
    bundle: ModelBundle,
    text: str,
    layer: int,
    *,
    patch: tuple[str, int, torch.Tensor] | None = None,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Every oracle tap at ``layer`` and the logits, by raw hooks on the HF
    modules. ``patch`` = (component, position, value) overwrites that
    component's tensor at one position before the model consumes it."""
    model = bundle.model
    block = model.transformer.h[layer]
    seen: dict[str, torch.Tensor] = {}
    handles = []

    def reader(name: str, kind: str) -> Callable[..., Any]:
        if kind == "in":

            def pre(_m: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
                seen[name] = _hidden_arg(args, kwargs).detach().clone()

            return pre

        def post(_m: Any, _args: Any, output: Any) -> None:
            seen[name] = _first(output).detach().clone()

        return post

    def writer(kind: str, pos: int, value: torch.Tensor) -> Callable[..., Any]:
        if kind == "in":

            def pre(_m: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
                x = _hidden_arg(args, kwargs).clone()
                x[:, pos] = value
                if args:
                    return (x, *args[1:]), kwargs
                return args, {**kwargs, "hidden_states": x}

            return pre

        def post(_m: Any, _args: Any, output: Any) -> Any:
            x = _first(output).clone()
            x[:, pos] = value
            return (x, *output[1:]) if isinstance(output, tuple) else x

        return post

    try:
        if patch is not None:
            component, pos, value = patch
            path, kind = ORACLE_TAPS[component]
            target = _module(block, path)
            if kind == "in":
                handles.append(
                    target.register_forward_pre_hook(
                        writer(kind, pos, value), with_kwargs=True
                    )
                )
            else:
                handles.append(target.register_forward_hook(writer(kind, pos, value)))
        for name, (path, kind) in ORACLE_TAPS.items():
            target = _module(block, path)
            if kind == "in":
                handles.append(
                    target.register_forward_pre_hook(
                        reader(name, kind), with_kwargs=True
                    )
                )
            else:
                handles.append(target.register_forward_hook(reader(name, kind)))
        for name, path in ORACLE_LAYERLESS.items():
            handles.append(
                _module(model, path).register_forward_hook(reader(name, "out"))
            )
        with torch.no_grad():
            logits = model(**_inputs(bundle, text)).logits
    finally:
        for handle in handles:
            handle.remove()
    return seen, logits


def _read_doc(components: dict[str, int | None]) -> dict[str, Any]:
    """Read every named component (at its layer; ``None`` for a layer-less
    one) over all positions on the un-intervened base."""
    sites: dict[str, Any] = {}
    reads: dict[str, Any] = {}
    for component, layer in components.items():
        site: dict[str, Any] = {"component": component}
        if layer is not None:
            site["layers"] = [layer]
        sites[f"s_{component}"] = site
        reads[f"r_{component}"] = {"site": f"s_{component}", "pos": "all"}
    return {
        "header": {"protocol_version": PROTOCOL_VERSION},
        "model": {"key": "test", "revision": "main"},
        "data": base_data_section(with_counterfactual=False),
        "method": {
            "intervened_models": {"original": {"input": "base", "reads": list(reads)}},
            "sites": sites,
            "reads": reads,
            "save": [saved(r, "original", f"{r}.safetensors") for r in reads],
        },
    }


def _swap_doc(
    component: str, layer: int, pos: int, *, also_read: tuple[str, ...] = ()
) -> dict[str, Any]:
    """Swap the counterfactual's ``component`` into base at ``pos``; read the
    patched and clean next-token logits, and ``also_read`` at ``layer`` on
    both."""
    sites: dict[str, Any] = {
        "tap": {"component": component, "layers": [layer]},
        "lm_head": {"component": "lm_head"},
    }
    reads: dict[str, Any] = {
        "v_cf": {"site": "tap", "pos": {"index": pos}},
        "logits": {"site": "lm_head", "pos": {"index": -1}},
    }
    for other in also_read:
        sites[f"s_{other}"] = {"component": other, "layers": [layer]}
        reads[f"r_{other}"] = {"site": f"s_{other}", "pos": "all"}
    extra = [f"r_{other}" for other in also_read]
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
                "original_base": {"input": "base", "reads": ["logits", *extra]},
                "patched": {
                    "input": "base",
                    "reads": ["logits", *extra],
                    "writes": ["patch"],
                },
            },
            "sites": sites,
            "reads": reads,
            "writes": {
                "patch": {"site": "tap", "pos": {"index": pos}, "do": {"swap": "v_cf"}}
            },
            "save": [
                saved("logits", "patched", "p.safetensors"),
                saved("logits", "original_base", "c.safetensors"),
                *(saved(r, "patched", f"p_{r}.safetensors") for r in extra),
                *(saved(r, "original_base", f"c_{r}.safetensors") for r in extra),
            ],
        },
    }


def _swap(bundle: ModelBundle, doc: dict[str, Any]) -> Any:
    return executor_for(doc, bundle, base_texts=[TEXT], counterfactual_texts=[CF_TEXT])


# --------------------------------------------------------------------------- #
# detection
# --------------------------------------------------------------------------- #


def test_gptj_is_detected_as_its_own_tree_not_gpt2(gptj: ModelBundle):
    """🐞 GPT-2's predicate was ``transformer.h`` alone, so GPT-J loaded with
    GPT-2's taps (``ln_2``, ``attn.c_proj``) on a tree that has neither."""
    assert family_for(gptj.model).family == "gptj_tree"
    assert gptj.adapter.family == "gptj_tree"
    assert not gptj.is_gpt2_family
    assert gptj.info.family == "gptj"
    assert gptj.streams == ("full_attention",) * 5


def test_the_gpt2_and_llama_fixtures_keep_their_trees():
    assert load_model("hf-internal-testing/tiny-random-gpt2").adapter.family == (
        "gpt2_tree"
    )
    llama = load_model("hf-internal-testing/tiny-random-LlamaForCausalLM")
    assert llama.adapter.family == "llama_tree"


# --------------------------------------------------------------------------- #
# the tap oracle, per component
# --------------------------------------------------------------------------- #


def test_every_block_tap_reads_what_the_oracle_captures(gptj: ModelBundle):
    """Every per-layer component the tree serves, read by the engine over all
    positions, is bit-identical to the raw hook on the HF module the oracle
    names. Anti-vacuity: the captures differ from each other, so a tap that
    resolved to a neighbouring module could not pass."""
    want, _ = _oracle_forward(gptj, TEXT, LAYER)
    executor = executor_for(
        _read_doc(dict.fromkeys(ORACLE_TAPS, LAYER)), gptj, base_texts=[TEXT]
    )
    for component in ORACLE_TAPS:
        got = executor.read_value(f"r_{component}")
        assert torch.equal(got, want[component]), component
    distinct = [
        "block_input",
        "attention_input_norm",
        "attention_output",
        "mlp_output",
        "block_output",
        "attention_premix",
    ]
    for i, a in enumerate(distinct):
        for b in distinct[i + 1 :]:
            assert not torch.allclose(want[a], want[b], atol=1e-4), (a, b)
    # the parallel block: the MLP consumes ln_1's output, not a second norm
    assert torch.equal(want["mlp_input"], want["attention_input_norm"])
    assert want["mlp_activation"].shape[-1] == 4 * gptj.info.hidden_size


def test_every_model_boundary_tap_reads_what_the_oracle_captures(gptj: ModelBundle):
    want, logits = _oracle_forward(gptj, TEXT, LAYER)
    executor = executor_for(
        _read_doc(dict.fromkeys(ORACLE_LAYERLESS)), gptj, base_texts=[TEXT]
    )
    for component in ORACLE_LAYERLESS:
        assert torch.equal(executor.read_value(f"r_{component}"), want[component])
    assert torch.equal(want["lm_head"], logits)


def test_attention_result_is_each_heads_share_of_the_attention_output(
    gptj: ModelBundle,
):
    """``attention_result`` is derived, not tapped: head ``h``'s slice of the
    ``out_proj`` input times that head's columns of ``out_proj``'s weight
    (an ``nn.Linear``, so ``(out, in)``). GPT-J's ``out_proj`` has no bias, so
    the heads sum to ``attention_output``."""
    want, _ = _oracle_forward(gptj, TEXT, LAYER)
    out_proj = gptj.model.transformer.h[LAYER].attn.out_proj
    assert out_proj.bias is None
    heads, d = gptj.info.num_heads, gptj.info.head_dim
    premix = want["attention_premix"]
    oracle = torch.cat(
        [
            premix[..., h * d : (h + 1) * d] @ out_proj.weight[:, h * d : (h + 1) * d].T
            for h in range(heads)
        ],
        dim=-1,
    )
    got = executor_for(
        _read_doc({"attention_result": LAYER}), gptj, base_texts=[TEXT]
    ).read_value("r_attention_result")
    assert got.shape[-1] == heads * gptj.info.hidden_size
    torch.testing.assert_close(got, oracle, atol=1e-6, rtol=1e-5)
    summed = got.reshape(*got.shape[:-1], heads, -1).sum(dim=-2)
    torch.testing.assert_close(summed, want["attention_output"], atol=1e-6, rtol=1e-5)


def test_a_head_of_the_premix_is_its_head_dim_slice(gptj: ModelBundle):
    want, _ = _oracle_forward(gptj, TEXT, LAYER)
    d = gptj.info.head_dim
    doc = _read_doc({"attention_premix": LAYER})
    doc["method"]["sites"]["s_attention_premix"]["head"] = 3
    got = executor_for(doc, gptj, base_texts=[TEXT]).read_value("r_attention_premix")
    assert torch.equal(got, want["attention_premix"][..., 3 * d : 4 * d])


def test_the_head_gate_group_map_is_one_group_per_head(gptj: ModelBundle):
    """§2.5 ``group: head`` on the premix: one group per head, ``head_dim``
    wide, derived from the entry alone (4 heads of 8 on the fixture)."""
    info = gptj.info
    assert site_group_map(info, "head", "attention_premix") == (
        info.num_heads,
        info.head_dim,
    )
    assert (info.num_heads, info.head_dim) == (4, 8)


# --------------------------------------------------------------------------- #
# the parallel residual
# --------------------------------------------------------------------------- #


def test_the_parallel_residual_identity_is_exact_in_fp32(gptj: ModelBundle):
    """The one identity the family declares, read from the declaration: which
    component, from which inputs, in which order, to which tolerance. fp32
    addition is not associative, so the declared order is the block's own
    (``attn_outputs + feed_forward_hidden_states + residual``) and the
    identity is exact there. The sum in the order the spec writes it,
    ``block_input + attention_output + mlp_output``, is held to one fp32 ulp
    of the residual's magnitude."""
    (identity,) = identities_for("gptj_tree")
    assert identity.name == "residual_parallel"
    assert identity.component == "block_output"
    assert identity.inputs == ("attention_output", "mlp_output", "block_input")
    names = (identity.component, *identity.inputs)
    for layer in range(gptj.info.num_layers):
        executor = executor_for(
            _read_doc(dict.fromkeys(names, layer)), gptj, base_texts=[TEXT]
        )
        values = {n: executor.read_value(f"r_{n}") for n in names}
        atol, rtol = identity.tolerance_for(gptj.dtype)
        first, *rest = identity.inputs
        total = values[first]
        for name in rest:
            total = total + values[name]
        torch.testing.assert_close(total, values["block_output"], atol=atol, rtol=rtol)
        reordered = (
            values["block_input"] + values["attention_output"] + values["mlp_output"]
        )
        ulp = torch.finfo(torch.float32).eps * values["block_output"].abs().max()
        assert (reordered - values["block_output"]).abs().max() <= ulp, layer


def test_a_write_at_mlp_input_reaches_the_mlp_alone(gptj: ModelBundle):
    """On the parallel block ``mlp_input`` is ``ln_1``'s output as the MLP
    sees it: a write there moves ``mlp_output`` and leaves the same layer's
    ``attention_output`` bit-identical. A write at ``attention_input_norm``
    reaches both, because both branches read ``ln_1``."""
    also = ("attention_output", "mlp_output")
    executor = _swap(gptj, _swap_doc("mlp_input", LAYER, 1, also_read=also))
    reads = {
        (r, m): executor.dense_value(_ref(f"r_{r}", m))
        for r in also
        for m in ("patched", "original_base")
    }
    assert torch.equal(
        reads[("attention_output", "patched")],
        reads[("attention_output", "original_base")],
    )
    assert not torch.allclose(
        reads[("mlp_output", "patched")],
        reads[("mlp_output", "original_base")],
        atol=1e-6,
    )
    executor = _swap(gptj, _swap_doc("attention_input_norm", LAYER, 1, also_read=also))
    for component in also:
        patched = executor.dense_value(_ref(f"r_{component}", "patched"))
        clean = executor.dense_value(_ref(f"r_{component}", "original_base"))
        assert not torch.allclose(patched, clean, atol=1e-6), component


def _ref(read: str, model: str) -> Any:
    from causalab.protocol.schema import ReadRef

    return ReadRef(read, model)


# --------------------------------------------------------------------------- #
# writes, against the oracle
# --------------------------------------------------------------------------- #

WRITABLE = [c for c in ORACLE_TAPS if CAPABILITIES[c].writes is not None]


@pytest.mark.parametrize("component", WRITABLE)
def test_a_swap_lands_where_the_oracle_lands_it(gptj: ModelBundle, component: str):
    """The engine's swap of the counterfactual's value at position 1 moves the
    next-token logits exactly as the raw hook writing the same value into the
    same HF tensor does. Anti-vacuity: the swap moves the logits (past the
    1e-6 the synthetic-family test uses; 📐 the smallest move here is the
    query swap's, 8.0e-5 on the random-weight fixture)."""
    pos = 1
    cf, _ = _oracle_forward(gptj, CF_TEXT, LAYER)
    _, clean = _oracle_forward(gptj, TEXT, LAYER)
    _, patched = _oracle_forward(
        gptj, TEXT, LAYER, patch=(component, pos, cf[component][:, pos])
    )
    executor = _swap(gptj, _swap_doc(component, LAYER, pos))
    got = executor.dense_value(_ref("logits", "patched"))
    torch.testing.assert_close(got[:, 0], patched[:, -1], atol=1e-6, rtol=1e-5)
    assert (patched[:, -1] - clean[:, -1]).abs().max() > 1e-6, component


# --------------------------------------------------------------------------- #
# what the tree does not have is refused by name
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("component", NOT_ON_GPTJ)
def test_a_component_the_tree_lacks_is_refused_by_name(
    gptj: ModelBundle, component: str
):
    with pytest.raises(ProtocolError, match="declares no tap") as refused:
        resolve_site(gptj, SiteSpec(component=component, layers=(LAYER,)))
    assert "gptj_tree" in str(refused.value) and component in str(refused.value)


def test_a_document_naming_block_mid_is_refused_before_a_forward(gptj: ModelBundle):
    forwards: list[int] = []
    handle = gptj.model.register_forward_pre_hook(lambda *_: forwards.append(1))
    try:
        executor = executor_for(
            _read_doc({"block_mid": LAYER}), gptj, base_texts=[TEXT]
        )
        with pytest.raises(ProtocolError, match="block_mid"):
            executor.read_value("r_block_mid")
    finally:
        handle.remove()
    assert forwards == []


def test_the_attention_gate_is_refused_by_its_predicate(gptj: ModelBundle):
    """GPT-J's attention has no gate: the ``gated_attention`` probe refuses it
    by the architecture, before the family is asked for a tap."""
    with pytest.raises(ProtocolError, match="gate"):
        resolve_site(gptj, SiteSpec(component="attention_gate", layers=(LAYER,)))


def test_the_loaded_inventory_is_the_declared_taps(gptj: ModelBundle):
    """Per layer, exactly the per-layer components the adapter declares that
    the architecture has (the gate is refused by its predicate); at the model
    boundary, the four layer-less ones."""
    inventory = loaded_inventory(gptj)
    per_layer = set(GPTJ_TREE.taps) - set(LAYERLESS_COMPONENTS) - {"attention_gate"}
    for layer in inventory.layers:
        assert set(layer.components) == per_layer, layer.layer
    assert set(inventory.layerless) == set(LAYERLESS_COMPONENTS)
