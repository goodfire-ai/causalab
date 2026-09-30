"""Static model metadata — the widths canonicalization derives from.

Deriving featurizer widths and param shapes (spec §6) needs the model's
*static configuration* (hidden size, depth, head counts), never its weights.
This registry keeps that metadata deterministic and offline: entries for the
models the repo actually uses are declared here as data, tests register
their tiny-random models, and an HF config can be adapted explicitly with
[`model_info_from_hf_config`][] when a caller opts in — the protocol layer
itself never touches the network.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from causalab.protocol.registry.plans import (
    GEMMA2_PLAN,
    LLAMA_PLAN,
    NO_PLAN,
    QWEN3_PLAN,
    QWEN35_MOE_PLAN,
    QWEN35_PLAN,
    ParallelPlan,
    parallel_plan_from_hf_config,
)
from causalab.protocol.rules.errors import ValidationError
from causalab.protocol.schema import STREAMS, Stream


@dataclasses.dataclass(frozen=True)
class ModelInfo:
    """The static facts canonicalization needs about one model."""

    key: str
    hidden_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    #: The dense MLP's inner width — ``None`` on a tower whose every block is a
    #: sparse-MoE block (Qwen3.6-35B-A3B): there is no dense MLP, so
    #: ``mlp_activation`` names no tensor and [`component_shape`][] refuses
    #: it at load, the same refusal the run makes from the module tree. 🐞 The
    #: adapter used to fall back to ``4 · hidden`` there, so ``validate`` sized
    #: an ``mlp_activation`` featurizer at 8192 for a tensor that does not
    #: exist and the run refused with a ``NotImplementedError``.
    intermediate_size: int | None
    vocab_size: int
    native_dtype: str = "fp32"
    #: The HF ``model_type`` of the config the entry describes, exactly as
    #: [`model_info_from_hf_config`][] reads it off the *text* config
    #: (``llama``, ``gpt2``, ``qwen3_5_moe_text`` — 📐 measured on the tiny
    #: fixtures and on ``Qwen/Qwen3.6-35B-A3B``'s own config: the Qwen3.5-MoE *wrapper* config
    #: says ``qwen3_5_moe``, its text config ``qwen3_5_moe_text``, and the
    #: adapter reads the text config, so that is the key). It is the key of
    #: the per-family tap table ([`Capability.overrides`][]): the address of
    #: each attention-interior component on this family, and what lets
    #: [`predicate_holds`][] decide ``split_qkv`` / ``gated_attention``
    #: offline. Not in any canonical form. An entry that leaves it unset is a
    #: family the table has not met: the run serves it by measurement and
    #: ``validate`` decides nothing about its mixer interior.
    family: str | None = None
    num_experts: int | None = None
    #: top-k: how many of ``num_experts`` each token is routed to. The width of
    #: ``router_scores``, whose axis is that top-k list.
    num_experts_per_tok: int | None = None
    #: The shared expert's inner width. Deliberately separate from
    #: ``intermediate_size``: a MoE checkpoint can carry three different inner
    #: widths (dense ``intermediate_size``, ``moe_intermediate_size`` per routed
    #: expert, and this one), and reading the wrong one is silent.
    shared_expert_intermediate_size: int | None = None
    #: The *routed* experts' inner width (``moe_intermediate_size``) — the third
    #: of those three inner widths, and the feature width of the per-expert
    #: interior (``expert_gate_proj`` and friends). ⚠️ On
    #: ``tiny-random/qwen3.5-moe`` all three widths are 32, so the fixture
    #: cannot tell a wrong choice from a right one — which is exactly why this
    #: is its own field rather than a fallback through one of the others.
    moe_intermediate_size: int | None = None
    #: The Gated DeltaNet mixer's dimensions. Its q/k live in
    #: *key-head* space and its v/gate/state in *value-head* space — two
    #: different head counts, the linear-attention analogue of GQA, and the
    #: same silent-empty-slice hazard if one bound is used for the other.
    #: ⚠️ Four independent numbers, deliberately not derived from each other:
    #: the fixture has 2× GVA tiling (``num_value_heads == 2 · num_key_heads``)
    #: and equal head dims, and a table that assumed either coupling would be
    #: silently wrong on a family that breaks it.
    linear_num_key_heads: int | None = None
    linear_num_value_heads: int | None = None
    linear_key_head_dim: int | None = None
    linear_value_head_dim: int | None = None
    #: The mixer stream at each layer, ``num_layers`` long, on a family whose
    #: config declares it (HF ``layer_types``). A hybrid tower alternates
    #: streams per layer, and a component that exists on only one of them
    #: (its capability row's ``stream``; ``COMPONENT_STREAMS`` is the view) is
    #: refused at load against this table —
    #: which is what lets the pure verbs refuse ``attention_premix`` at a Gated
    #: DeltaNet layer offline, instead of the run doing it against the loaded
    #: modules. ``None`` means the model declares no pattern the protocol can
    #: read; the run-time check still applies, so a dense family loses nothing
    #: by leaving it unset. Values are the protocol's ``STREAMS`` — an HF
    #: config's own spelling is mapped by [`model_info_from_hf_config`][]
    #: (``_HF_LAYER_STREAMS``), never copied in.
    layer_types: tuple[Stream, ...] | None = None
    #: The experts implementation the loaded model dispatches on
    #: (``experts_implementation``: ``"grouped_mm"``, ``"eager"``,
    #: ``"batched_mm"``) — a **load-time knob**, not a config fact, so a
    #: hand-declared entry leaves it ``None`` and only an entry adapted from a
    #: *loaded* model's config carries it. It is what lets ``validate`` decide
    #: the ``grouped_mm`` predicate (the routed interior's dispatch pin) during
    #: model-capability validation instead of at tap time; the tap-time
    #: probe stays as the last-line check. Not in
    #: any canonical form.
    experts_implementation: str | None = None
    #: The family's parallel plan (``docs/model_parallelism.md`` §5.2): module
    #: pattern → ``(style, axis)``, derived from the config class's
    #: ``base_model_tp_plan`` / ``base_model_ep_plan`` when the entry is
    #: adapted from a loaded config and declared on the built-in entries, so
    #: ``dry-run`` checks a geometry from the entry alone, torch-free and
    #: offline. On the entry rather than the family adapter because the plan
    #: is a fact of the HF ``model_type`` (the config class), not of the
    #: module tree. ``None`` is an entry the table has not met (a
    #: hand-declared one): the plan rules of the geometry check say nothing
    #: and the load derives it. Not in any canonical form.
    parallel_plan: ParallelPlan | None = None

    def __post_init__(self) -> None:
        if self.layer_types is None:
            return
        if len(self.layer_types) != self.num_layers:
            raise ValueError(
                f"model {self.key!r}: layer_types has {len(self.layer_types)} "
                f"entries for a {self.num_layers}-layer model"
            )
        unknown = sorted(set(self.layer_types) - set(STREAMS))
        if unknown:
            raise ValueError(
                f"model {self.key!r}: layer_types names {unknown}, not in "
                f"{list(STREAMS)}"
            )


_REGISTRY: dict[str, ModelInfo] = {}


def register_model(info: ModelInfo) -> None:
    """Register (or replace) one model's static metadata."""
    _REGISTRY[info.key] = info


def get_model_info(key: str) -> ModelInfo:
    """Look up a model key; a missing entry is a load error (the alternative
    — fetching a config from the network mid-canonicalization — would make
    digests depend on connectivity)."""
    info = _REGISTRY.get(key)
    if info is None:
        raise ValidationError(
            4,
            f"model {key!r} is not in the protocol model registry — register "
            "its static config (causalab.protocol.registry.register_model, or "
            "model_info_from_hf_config on a loaded HF config)",
            path="model.key",
        )
    return info


def model_info_from_hf_config(key: str, config: Any) -> ModelInfo:
    """Adapt a loaded HF config object (its text config, on multimodal
    wrappers) into a [`ModelInfo`][]. The caller owns where the config
    came from; this function only reads attributes."""
    text = getattr(config, "text_config", None) or config
    num_heads = int(getattr(text, "num_attention_heads"))
    hidden = int(getattr(text, "hidden_size"))
    head_dim = int(getattr(text, "head_dim", None) or hidden // num_heads)
    # transformers 5 renamed ``torch_dtype`` to ``dtype`` and maps the old
    # config.json key onto it at load time (configuration_utils.py, "BC for the
    # torch_dtype argument"); reading ``torch_dtype`` here would only warn.
    dtype = str(getattr(text, "dtype", None) or "float32")
    return ModelInfo(
        key=key,
        hidden_size=hidden,
        num_layers=int(getattr(text, "num_hidden_layers")),
        num_heads=num_heads,
        num_kv_heads=int(getattr(text, "num_key_value_heads", None) or num_heads),
        head_dim=head_dim,
        intermediate_size=_intermediate_size(text, hidden),
        vocab_size=int(getattr(text, "vocab_size")),
        family=(
            str(getattr(text, "model_type"))
            if getattr(text, "model_type", None)
            else None
        ),
        native_dtype={"bfloat16": "bf16", "float16": "fp16"}.get(
            dtype.removeprefix("torch."), "fp32"
        ),
        # Two spellings in the wild, and neither is universal: mixtral and
        # qwen3_moe carry both, while qwen2_moe and qwen3_5_moe carry only
        # ``num_experts``. Reading ``num_local_experts`` alone silently left
        # num_experts=None on those, which makes component_width refuse
        # router_logits on a model that plainly has a router.
        num_experts=(
            getattr(text, "num_experts", None)
            or getattr(text, "num_local_experts", None)
        ),
        num_experts_per_tok=getattr(text, "num_experts_per_tok", None),
        # ⚠️ Three spellings, and on `tiny-random/qwen3.5-moe` all three are 32,
        # so the fixture CANNOT tell a wrong choice from a right one. Ordered
        # most-specific first and never silently defaulted to the dense
        # `intermediate_size`, because that is the one that would be wrong on a
        # real checkpoint while still producing a plausible number.
        shared_expert_intermediate_size=(
            getattr(text, "shared_expert_intermediate_size", None)
            or getattr(text, "moe_intermediate_size", None)
        ),
        moe_intermediate_size=getattr(text, "moe_intermediate_size", None),
        linear_num_key_heads=getattr(text, "linear_num_key_heads", None),
        linear_num_value_heads=getattr(text, "linear_num_value_heads", None),
        linear_key_head_dim=getattr(text, "linear_key_head_dim", None),
        linear_value_head_dim=getattr(text, "linear_value_head_dim", None),
        layer_types=_layer_types(text),
        # set on a loaded model's config by the modeling code's dispatch
        # (``from_pretrained(experts_implementation=...)``, default grouped_mm);
        # absent on a bare text config, which is the honest ``None``
        experts_implementation=(
            str(getattr(text, "_experts_implementation"))
            if getattr(text, "_experts_implementation", None) is not None
            else None
        ),
        parallel_plan=parallel_plan_from_hf_config(text),
    )


#: HF ``layer_types`` spellings and the protocol stream each one is. HF's
#: vocabulary names attention *variants* — a Gemma2/Gemma3 tower alternates
#: ``sliding_attention`` with ``full_attention``, Llama4 has
#: ``chunked_attention``, the sparse-attention families each have their own —
#: while the protocol's ``stream`` names the *mixer*: softmax attention or a
#: linear-attention kernel. A sliding window is still a ``self_attn`` child
#: computing an attention matrix, which is what the run-time probe
#: (``neural/shared/model_tree.py``) answers for it, so both halves of the stream
#: check agree on ``full_attention``. 🐞 The adapter used to copy the HF strings
#: straight into the ``STREAMS``-validated field, so every engine load of
#: ``google/gemma-2-2b-it`` — a built-in entry and a golden-protocol model —
#: raised on its ``sliding_attention`` layers.
_HF_LAYER_STREAMS: dict[str, Stream] = {
    "full_attention": "full_attention",
    "sliding_attention": "full_attention",
    "linear_attention": "linear_attention",
}


def _layer_types(text: Any) -> tuple[Stream, ...] | None:
    """The per-layer stream pattern an HF text config declares, in the
    protocol's vocabulary — or ``None`` when it declares none the protocol can
    read.

    The pinned transformers' llama and gpt2 configs carry no ``layer_types``;
    qwen3_5_moe and gemma2/gemma3 do. A pattern naming any spelling outside
    ``_HF_LAYER_STREAMS`` (``mamba``, ``chunked_attention``, a sparse-attention
    kind) is left unset as a whole rather than guessed at per layer: an
    unmapped kind is a family this table has not met, and the documented
    fallback for an entry without a pattern — the run-time check against the
    module the layer actually carries — is the honest answer for it. Deferring
    beats a wrong offline refusal, or a wrong offline pass.
    """
    declared = getattr(text, "layer_types", None)
    if declared is None:
        return None
    kinds = [str(kind) for kind in declared]
    if any(kind not in _HF_LAYER_STREAMS for kind in kinds):
        return None
    return tuple(_HF_LAYER_STREAMS[kind] for kind in kinds)


def _intermediate_size(text: Any, hidden: int) -> int | None:
    """The dense MLP's inner width, resolved the way the modeling code resolves
    it — or ``None`` when the config declares none, which is what an all-MoE
    text config does (``Qwen3_5MoeTextConfig`` has no ``intermediate_size``
    attribute at all; the ``4304`` in ``Qwen/Qwen3.6-35B-A3B``'s config.json is
    ``vision_config.intermediate_size``). 🐞 Falling back to ``4 · hidden``
    there gave ``mlp_activation`` a width on a tower with no dense MLP.

    🐞 Reading ``intermediate_size`` unconditionally is wrong on the GPT-2
    family: ``GPT2Config`` spells the field ``n_inner`` and the block computes
    ``config.n_inner if config.n_inner is not None else 4 * hidden_size``
    (transformers ``models/gpt2/modeling_gpt2.py:250``), ignoring any
    ``intermediate_size`` in the config. 📐 ``hf-internal-testing/tiny-random-gpt2``
    carries a stray ``intermediate_size: 37`` next to ``n_inner: null`` and a
    128-wide MLP, so the adapter reported 37 for a tensor that is 128 wide — a
    featurizer on ``mlp_activation`` would have been sized against nothing. It
    went unnoticed because no code compared a declared width to a real tensor
    until [`component_shape`][] did.
    """
    if hasattr(text, "n_inner"):  # the GPT-2 family's spelling, authoritative
        n_inner = getattr(text, "n_inner")
        return int(n_inner) if n_inner is not None else 4 * hidden
    declared = getattr(text, "intermediate_size", None)
    return int(declared) if declared else None


# --------------------------------------------------------------------------- #
# built-in entries — the models the repo's configs and corpus name.
# Sources: HF config.json of each checkpoint (static metadata, no weights).
# ``family`` is the config class's ``model_type`` (``GPT2Config.model_type ==
# "gpt2"``, ``LlamaConfig`` → ``llama``, ``Qwen3Config`` → ``qwen3``,
# ``Gemma2Config`` → ``gemma2``), i.e. what ``model_info_from_hf_config`` reads
# at run — so ``validate`` and the run key the per-family tap table identically.
# --------------------------------------------------------------------------- #

register_model(
    ModelInfo(
        key="meta-llama/Llama-3.1-8B",
        hidden_size=4096,
        num_layers=32,
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        intermediate_size=14336,
        vocab_size=128256,
        native_dtype="bf16",
        family="llama",
        parallel_plan=LLAMA_PLAN,
    )
)
register_model(
    ModelInfo(
        # The model of Prakash et al. 2025 (demos/papers/lookbacks.md),
        # registered so the replication's documents validate offline. Source:
        # the checkpoint's own config.json.
        key="meta-llama/Meta-Llama-3-70B-Instruct",
        hidden_size=8192,
        num_layers=80,
        num_heads=64,
        num_kv_heads=8,
        head_dim=128,
        intermediate_size=28672,
        vocab_size=128256,
        native_dtype="bf16",
        family="llama",
    )
)
register_model(
    ModelInfo(
        key="meta-llama/Llama-3.1-8B-Instruct",
        hidden_size=4096,
        num_layers=32,
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        intermediate_size=14336,
        vocab_size=128256,
        native_dtype="bf16",
        family="llama",
        parallel_plan=LLAMA_PLAN,
    )
)
register_model(
    ModelInfo(
        # Metadata lets validation, dry-run, and memory preflight size this
        # checkpoint offline. Source: its config.json at revision
        # 349b2ddb53ce8f2849a6c168a81980ab25258dac.
        key="meta-llama/Llama-3.1-70B",
        hidden_size=8192,
        num_layers=80,
        num_heads=64,
        num_kv_heads=8,
        head_dim=128,
        intermediate_size=28672,
        vocab_size=128256,
        native_dtype="bf16",
        family="llama",
        parallel_plan=LLAMA_PLAN,
    )
)
register_model(
    ModelInfo(
        # Registered so `validate` and `explain` can size documents naming it
        # offline — the pure verbs read this table rather than fetching a
        # config, so a document naming an unregistered key is checkable only
        # by running it. Source: the checkpoint's own config.json, revision
        # 9213176726f574b556790deb65791e0c5aa438b6.
        key="meta-llama/Llama-3.2-1B-Instruct",
        hidden_size=2048,
        num_layers=16,
        num_heads=32,
        num_kv_heads=8,
        head_dim=64,
        intermediate_size=8192,
        vocab_size=128256,
        native_dtype="bf16",
        family="llama",
        parallel_plan=LLAMA_PLAN,
    )
)
register_model(
    ModelInfo(
        # The pretrained sibling of the entry above — same architecture,
        # same tokenizer, different weights, so a document that reads one
        # checkpoint and writes the other can size both offline.
        # Source: the checkpoint's own config.json, revision
        # 4e20de362430cd3b72f300e6b0f18e50e7166e08.
        key="meta-llama/Llama-3.2-1B",
        hidden_size=2048,
        num_layers=16,
        num_heads=32,
        num_kv_heads=8,
        head_dim=64,
        intermediate_size=8192,
        vocab_size=128256,
        native_dtype="bf16",
        family="llama",
        parallel_plan=LLAMA_PLAN,
    )
)
register_model(
    ModelInfo(
        # The onboarding demos' model (demos/onboarding_tutorial/). Registered
        # so `validate` and `explain` can size its documents offline.
        # Source: the checkpoint's own config.json (``Qwen2Config``: hidden
        # 1536, 28 layers, 12 query / 2 KV heads of 128, intermediate 8960,
        # vocab 151936, tied embeddings).
        key="Qwen/Qwen2.5-1.5B-Instruct",
        hidden_size=1536,
        num_layers=28,
        num_heads=12,
        num_kv_heads=2,
        head_dim=128,
        intermediate_size=8960,
        vocab_size=151936,
        native_dtype="bf16",
        family="qwen2",
    )
)
register_model(
    ModelInfo(
        # The pretrained sibling of the demo model above — same architecture,
        # same tokenizer, different weights. 10_cross_model names both, which
        # is the only reason a second 1.5B entry exists: a document that reads
        # one checkpoint and writes another has to size both offline.
        # Source: the checkpoint's own config.json.
        key="Qwen/Qwen2.5-1.5B",
        hidden_size=1536,
        num_layers=28,
        num_heads=12,
        num_kv_heads=2,
        head_dim=128,
        intermediate_size=8960,
        vocab_size=151936,
        native_dtype="bf16",
        family="qwen2",
    )
)
register_model(
    ModelInfo(
        # The method library's model (demos/methods/): the smallest Qwen2.5
        # base checkpoint that mostly does the weekdays task the documents run
        # on (clean accuracy 0.68 base / 0.63 counterfactual on the test split;
        # 1.5B and 3B sit at 0.2, see demos/methods/README.md). Same depth as
        # the 1.5B, so the documents' layer literals hold for both. Source: the
        # checkpoint's own config.json (``Qwen2Config``: hidden 3584, 28
        # layers, 28 query / 4 KV heads of 128, intermediate 18944, vocab
        # 152064, untied embeddings).
        key="Qwen/Qwen2.5-7B",
        hidden_size=3584,
        num_layers=28,
        num_heads=28,
        num_kv_heads=4,
        head_dim=128,
        intermediate_size=18944,
        vocab_size=152064,
        native_dtype="bf16",
        family="qwen2",
    )
)
register_model(
    ModelInfo(
        key="gpt2",
        hidden_size=768,
        num_layers=12,
        num_heads=12,
        num_kv_heads=12,
        head_dim=64,
        intermediate_size=3072,
        vocab_size=50257,
        native_dtype="fp32",
        family="gpt2",
        parallel_plan=NO_PLAN,
    )
)
register_model(
    ModelInfo(
        key="gpt2-xl",
        hidden_size=1600,
        num_layers=48,
        num_heads=25,
        num_kv_heads=25,
        head_dim=64,
        intermediate_size=6400,
        vocab_size=50257,
        native_dtype="fp32",
        family="gpt2",
        parallel_plan=NO_PLAN,
    )
)
register_model(
    ModelInfo(
        # An alias row, not a second model: the same checkpoint under the Hub
        # id that carries its organization (`gpt2-xl` redirects to
        # `openai-community/gpt2-xl`). The onboarding tutorial's 00 demos name
        # it that way, and a key is looked up as spelled, so both spellings
        # need a row for a document naming either to validate offline. The
        # two spellings hash to two document digests, since `model.key` is in
        # the canonical form. `tests/protocol/test_capability_registry.py`
        # holds the two rows identical field for field.
        key="openai-community/gpt2-xl",
        hidden_size=1600,
        num_layers=48,
        num_heads=25,
        num_kv_heads=25,
        head_dim=64,
        intermediate_size=6400,
        vocab_size=50257,
        native_dtype="fp32",
        family="gpt2",
        parallel_plan=NO_PLAN,
    )
)
register_model(
    ModelInfo(
        # The MLP-steering paper replication's model (demos/papers/
        # mlp_steering.md, Geva et al. 2022), named by its Hub id with the
        # organization as the documents spell it. A registry row lets its
        # documents validate offline, which the standalone smoke
        # (scripts/standalone_smoke.py) requires. Source: the checkpoint's own
        # config.json (n_embd 1024, n_layer 24, n_head 16, vocab 50257; GPT-2's
        # MLP is 4 * n_embd), the same values model_info_from_hf_config derives.
        key="openai-community/gpt2-medium",
        hidden_size=1024,
        num_layers=24,
        num_heads=16,
        num_kv_heads=16,
        head_dim=64,
        intermediate_size=4096,
        vocab_size=50257,
        native_dtype="fp32",
        family="gpt2",
        parallel_plan=NO_PLAN,
    )
)
register_model(
    ModelInfo(
        # GPT-J 6B, the model of the function-vector replication (Todd et al.
        # 2024, arXiv:2310.15213). Registered so its documents validate
        # offline. Source: the checkpoint's config.json at revision
        # ``float16`` (b71ae8bc86cac13154e03e92b5855203086b722e), the fp16
        # weights the runs load: ``GPTJConfig`` n_embd 4096, n_layer 28,
        # n_head 16 (so head_dim 4096 / 16 = 256; no KV grouping), rotary_dim
        # 64 (rotary on the first 64 of each head's 256 dims), n_inner absent
        # (the class default None, so the MLP is 4 · 4096), vocab 50400,
        # torch_dtype float16, untied embeddings. The ``main`` revision
        # (47e169305d2e8376be1d31e765533382721b2cc1) has the same shape with
        # fp32 weights and no torch_dtype. transformers ships no
        # ``base_model_tp_plan`` for GPT-J, so the plan is empty.
        key="EleutherAI/gpt-j-6b",
        hidden_size=4096,
        num_layers=28,
        num_heads=16,
        num_kv_heads=16,
        head_dim=256,
        intermediate_size=16384,
        vocab_size=50400,
        native_dtype="fp16",
        family="gptj",
        parallel_plan=NO_PLAN,
    )
)
register_model(
    ModelInfo(
        key="Qwen/Qwen3-4B-Instruct-2507",
        hidden_size=2560,
        num_layers=36,
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        intermediate_size=9728,
        vocab_size=151936,
        native_dtype="bf16",
        family="qwen3",
        parallel_plan=QWEN3_PLAN,
    )
)
# The CPU-tier fixture corpus's model (``tests/protocols/*_im.json``): an
# ungated 8B in the shape the corpus was authored for — hidden 4096, a layer
# 18, (4096, k) bundles — so CI without a Hub token can load its tokenizer
# at the run door. Source: the checkpoint's config.json (``Qwen3Config``:
# hidden 4096, 36 layers, 32 query / 8 KV heads of 128, intermediate 12288,
# vocab 151936, untied embeddings).
register_model(
    ModelInfo(
        key="Qwen/Qwen3-8B",
        hidden_size=4096,
        num_layers=36,
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        intermediate_size=12288,
        vocab_size=151936,
        native_dtype="bf16",
        family="qwen3",
        parallel_plan=QWEN3_PLAN,
    )
)
register_model(
    ModelInfo(
        key="google/gemma-2-2b-it",
        hidden_size=2304,
        num_layers=26,
        num_heads=8,
        num_kv_heads=4,
        head_dim=256,
        intermediate_size=9216,
        vocab_size=256000,
        native_dtype="bf16",
        family="gemma2",
        parallel_plan=GEMMA2_PLAN,
    )
)
# The two MIB circuit-track models (Mueller et al. 2025, ``MIB_circuit_track/
# utils.py``: ``qwen2.5 → Qwen/Qwen2.5-0.5B``, ``gemma2 → google/gemma-2-2b``),
# registered so a document naming them validates offline. Source: each
# checkpoint's config.json (``Qwen2Config``: hidden 896, 24 layers, 14 query /
# 2 KV heads of 64, intermediate 4864, vocab 151936; ``Gemma2Config`` for the
# base model equals the ``-it`` row above — same architecture, different
# weights). ``tests/protocol/test_registry_shapes.py`` cross-checks each row
# against ``model_info_from_hf_config`` on the cached config when present.
register_model(
    ModelInfo(
        key="Qwen/Qwen2.5-0.5B",
        hidden_size=896,
        num_layers=24,
        num_heads=14,
        num_kv_heads=2,
        head_dim=64,
        intermediate_size=4864,
        vocab_size=151936,
        native_dtype="bf16",
        family="qwen2",
        parallel_plan=LLAMA_PLAN,
    )
)
register_model(
    ModelInfo(
        key="google/gemma-2-2b",
        hidden_size=2304,
        num_layers=26,
        num_heads=8,
        num_kv_heads=4,
        head_dim=256,
        intermediate_size=9216,
        vocab_size=256000,
        native_dtype="bf16",
        family="gemma2",
        parallel_plan=GEMMA2_PLAN,
    )
)
# The second dense family the parallel golden runs on a card
# (``tests/golden/_parallel/families.py``; docs/model_parallelism.md §10.6,
# §11): gemma2's 9B base checkpoint. Source: its config.json (``Gemma2Config``:
# hidden 3584, 42 layers, 16 query / 8 KV heads of 256, intermediate 14336,
# vocab 256000, ``tie_word_embeddings`` true — so ``pp`` is refused by name
# at the load and the vocabulary row is ``unapplied`` like the 2b's; its
# ``torch_dtype`` is **float32**: the checkpoint is stored in fp32 — its
# header census ``tests/golden/parallel_headers_gemma2_9b.json`` — and the
# parallel golden's bf16 realization is a converting load, §5.3).
# ``tests/golden/test_parallel_families_record.py`` cross-checks the row
# against ``model_info_from_hf_config`` on the cached config when present.
register_model(
    ModelInfo(
        key="google/gemma-2-9b",
        hidden_size=3584,
        num_layers=42,
        num_heads=16,
        num_kv_heads=8,
        head_dim=256,
        intermediate_size=14336,
        vocab_size=256000,
        native_dtype="fp32",
        family="gemma2",
        parallel_plan=GEMMA2_PLAN,
    )
)
register_model(
    ModelInfo(
        # A hybrid Gated
        # DeltaNet / gated full-attention tower with a sparse MoE block in every
        # layer, loaded as the Qwen3.5-MoE text tower (the same class as the
        # ``tiny-random/qwen3.5-moe`` fixture). Registered so a document naming
        # it validates and digests offline; without this entry a run on it
        # pre-flights through ``--register-from-hf``.
        # Source: the checkpoint's own config.json, revision
        # 995ad96eacd98c81ed38be0c5b274b04031597b0, read through
        # ``model_info_from_hf_config`` — every value here equals what the run
        # re-registers from the loaded config, so validate and run size the
        # same document identically. Cross-checked against
        # docs/qwen36-35b-a3b-architecture.html.
        key="Qwen/Qwen3.6-35B-A3B",
        hidden_size=2048,
        num_layers=40,
        num_heads=16,
        num_kv_heads=2,
        head_dim=256,
        # The text config carries no dense ``intermediate_size`` at all: the
        # block is MoE at every layer, so ``mlp_activation`` names no tensor
        # here, and the adapter reads ``None`` for it too (validate and run
        # agree). Was the adapter's ``4 · hidden`` fallback, 8192, which sized
        # a featurizer against a tensor that does not exist.
        intermediate_size=None,
        vocab_size=248320,
        native_dtype="bf16",
        # 📐 what the adapter reads off this checkpoint's config: the wrapper
        # ``Qwen3_5MoeConfig`` says ``qwen3_5_moe``, but the adapter reads the
        # *text* config, whose ``model_type`` is ``qwen3_5_moe_text`` — the same
        # string the ``tiny-random/qwen3.5-moe`` fixture loads with. 🐞 Was
        # ``qwen3_5_moe`` (the wrapper's spelling), which nothing keyed on;
        # the per-family tap table does, and validate and run must agree.
        family="qwen3_5_moe_text",
        num_experts=256,
        num_experts_per_tok=8,
        # the routed and the shared inner widths happen to coincide on this
        # checkpoint (``d_expert 512`` for both); they are two fields because
        # they are two config keys
        shared_expert_intermediate_size=512,
        moe_intermediate_size=512,
        # Gated DeltaNet: q/k in 16 key heads of 128, v/gate/state in 32 value
        # heads of 128 — the 2× GVA tiling the fixture also has
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        # ``full_attention_interval: 4``: layers 3, 7, …, 39 carry the gated
        # full-attention mixer, the other 30 the DeltaNet one
        layer_types=(("linear_attention",) * 3 + ("full_attention",)) * 10,
        parallel_plan=QWEN35_MOE_PLAN,
    )
)
register_model(
    ModelInfo(
        # The dense Qwen3.5 tower the addition-heads DBM package runs on
        # (demos/papers/addition_heads_dbm.md), registered so its documents
        # validate offline. Source: the checkpoint's own config.json, revision
        # 15852e8c16360a2fea060d615a32b45270f8a8fc, read through
        # ``model_info_from_hf_config``: the text config ``Qwen3_5TextConfig``
        # (``model_type`` ``qwen3_5_text``), hidden 2048, 24 layers, 8 query /
        # 2 KV heads of 256, a dense MLP of 6144, vocab 248320, tied
        # embeddings, gated full attention (``attn_output_gate``).
        key="Qwen/Qwen3.5-2B",
        hidden_size=2048,
        num_layers=24,
        num_heads=8,
        num_kv_heads=2,
        head_dim=256,
        intermediate_size=6144,
        vocab_size=248320,
        native_dtype="bf16",
        family="qwen3_5_text",
        # Gated DeltaNet: 16 key heads and 16 value heads of 128 (no GVA
        # tiling on this checkpoint, unlike the A3B's 2x)
        linear_num_key_heads=16,
        linear_num_value_heads=16,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        # ``full_attention_interval: 4``: layers 3, 7, ..., 23 carry the gated
        # full-attention mixer, the other 18 the DeltaNet one
        layer_types=(("linear_attention",) * 3 + ("full_attention",)) * 6,
        parallel_plan=QWEN35_PLAN,
    )
)
