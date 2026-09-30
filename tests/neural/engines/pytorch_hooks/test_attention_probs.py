"""``attention_probs``: read + write, and the one component `layout` won't describe.

The three checks at the top are nnterp's, ported to this backend
(``nnterp/rename_utils.py`` ``check_source``): the pattern must have shape
``(batch, heads, seq, seq)``, its rows must sum to 1, and **writing it must
change the logits**. The third is the one that matters, because it is the one a
plausible-looking implementation fails.

The scope is the whole pattern. Addressing one query row, featurizing, or
slicing ``dims`` all need the typed feature-shape descriptor — the feature axis
here *is* a position axis — so each is refused by name rather than
approximated.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from causalab.neural.engines.pytorch_hooks.attention_interface import (
    InterfaceTap,
    attention_interface_taps,
)
from causalab.neural.engines.pytorch_hooks.engine import PytorchHooksEngine
from causalab.neural.engines.pytorch_hooks.loading import load_model
from causalab.neural.shared.sites import resolve_site
from causalab.protocol.rules.errors import ProtocolError
from causalab.protocol.schema import SiteSpec

from ._drive import base_data_section, executor_for
from .conftest import TINY_GPT2
from tests.protocol._docs import UNWRITTEN, saved

pytestmark = pytest.mark.smoke


def eager_attention_writes(edits: dict) -> Any:
    """The pattern-edit spelling, over the attention interface manager.

    Kept as a test-local shim because these tests pin the *pattern-write*
    behaviour specifically, and phrasing them in terms of "an in-place edit to
    the pattern for these modules" is what they are about. The manager's own
    contract (hand out a clone, take back a replacement) is exercised by the
    interface manager's own tests.
    """

    def as_tap(edit: Any) -> tuple[InterfaceTap, ...]:
        def run(probs: torch.Tensor) -> torch.Tensor:
            edit(probs)
            return probs

        return (InterfaceTap(slot="probs", edit=run),)

    return attention_interface_taps(
        {module_id: as_tap(edit) for module_id, edit in edits.items()}
    )


#: 📐 the fixture's only full-attention layer; 0-2 are Gated DeltaNet.
FULL_ATTENTION_LAYER = 3
DELTANET_LAYER = 0

#: Same token count, so an attention-pattern interchange is shape-compatible.
BASE_TEXT = "the quick brown fox jumps"
CF_TEXT = "a slow green turtle sleeps"


def _read_doc(
    *,
    layer: int = FULL_ATTENTION_LAYER,
    pos: object = "all",
    featurizer: bool = False,
    dims: list[int] | None = None,
) -> dict:
    read: dict = {"site": "tap", "pos": pos}
    doc: dict = {
        "header": {"protocol_version": "4"},
        "model": {"key": "test", "revision": "main"},
        "data": base_data_section(with_counterfactual=False),
        "method": {
            "intervened_models": {"original": {"input": "base", "reads": ["r"]}},
            "sites": {"tap": {"component": "attention_probs", "layers": [layer]}},
            "reads": {"r": read},
            "save": [saved("r", "original", "a.safetensors")],
        },
    }
    if featurizer:
        doc["method"]["featurizers"] = {"f": {"kind": "subspace", "k": 1}}
        read["featurizer"] = "f"
    if dims is not None:
        read["dims"] = dims
    return doc


def _swap_doc(*, layer: int = FULL_ATTENTION_LAYER) -> dict:
    """The counterfactual's whole pattern (read on the un-intervened network
    on the counterfactual) swapped into base; the logits before and after,
    the clean ones off the un-intervened network on base."""
    return {
        "header": {"protocol_version": "4"},
        "model": {"key": "test", "revision": "main"},
        "data": base_data_section(with_counterfactual=True),
        "method": {
            "intervened_models": {
                UNWRITTEN: {"input": "counterfactual", "reads": ["v_cf"]},
                "original_base": {"input": "base", "reads": ["clean"]},
                "patched": {"input": "base", "reads": ["after"], "writes": ["patch"]},
            },
            "sites": {
                "tap": {"component": "attention_probs", "layers": [layer]},
                "lm_head": {"component": "lm_head"},
            },
            "reads": {
                "v_cf": {"site": "tap", "pos": "all"},
                "clean": {"site": "lm_head", "pos": {"index": -1}},
                "after": {"site": "lm_head", "pos": {"index": -1}},
            },
            "writes": {"patch": {"site": "tap", "pos": "all", "do": {"swap": "v_cf"}}},
            "save": [
                saved("after", "patched", "p.safetensors"),
                saved("clean", "original_base", "c.safetensors"),
            ],
        },
    }


@pytest.fixture(scope="module")
def pattern(qwen35moe_bundle) -> torch.Tensor:
    return executor_for(
        _read_doc(), qwen35moe_bundle, base_texts=[BASE_TEXT]
    ).read_value("r")


# --------------------------------------------------------------------------- #
# nnterp's three-part check
# --------------------------------------------------------------------------- #


def test_the_pattern_has_the_attention_shape(qwen35moe_bundle, pattern):
    """nnterp check 1: (batch, heads, query, key)."""
    batch, heads, query, key = pattern.shape
    assert batch == 1
    assert heads == qwen35moe_bundle.info.num_heads
    assert query == key, "both axes are positions — that is the whole difficulty"


def test_the_pattern_rows_sum_to_one(pattern):
    """nnterp check 2: it is a distribution over key positions."""
    sums = pattern.sum(-1)
    torch.testing.assert_close(sums, torch.ones_like(sums), atol=1e-5, rtol=1e-5)


def test_writing_the_pattern_changes_the_logits(qwen35moe_bundle):
    """nnterp check 3, and the reason the pattern write is not a one-line tap.

    A ``register_forward_hook`` on the mixer CAN rewrite element 1 of its output
    tuple — and it would change nothing, because ``attn_output`` was computed
    from the pattern inside the attention function before the hook fires. That
    is the same silent-no-op shape as writing ``router_logits``. The write
    goes through the eager attention function instead, and this test is what
    says the difference mattered.
    """
    executor = executor_for(
        _swap_doc(),
        qwen35moe_bundle,
        base_texts=[BASE_TEXT],
        counterfactual_texts=[CF_TEXT],
    )
    clean = executor.read_value("clean")
    after = executor.read_value("after")
    assert float((after - clean).abs().max()) > 0.0, (
        "swapping the whole attention pattern left the logits unchanged — the "
        "write is not reaching the value multiply"
    )


# --------------------------------------------------------------------------- #
# the property that makes duplicating the post-softmax math safe
# --------------------------------------------------------------------------- #


def test_an_identity_edit_is_bit_identical(qwen35moe_bundle):
    """The wrapper redoes the two lines that follow the softmax
    (``repeat_kv`` + ``matmul(...).transpose(1, 2).contiguous()``). Duplicating
    library internals is how a backend rots silently, so it is pinned: with an
    edit that changes nothing, the recomputed logits must equal the unpatched
    logits **exactly**. 📐 Measured max difference 0.0.

    If a future transformers changes what happens after the softmax, this fails
    instead of the numbers quietly drifting.
    """
    encoded = qwen35moe_bundle.tokenizer(BASE_TEXT, return_tensors="pt")
    attn = qwen35moe_bundle.mixer_at(FULL_ATTENTION_LAYER)

    with torch.no_grad():
        clean = qwen35moe_bundle.model(**encoded).logits.clone()

    seen: dict[str, object] = {}

    def identity(probs: torch.Tensor) -> None:
        seen["shape"] = tuple(probs.shape)  # mutate nothing

    with eager_attention_writes({id(attn): identity}):
        with torch.no_grad():
            recomputed = qwen35moe_bundle.model(**encoded).logits.clone()

    assert seen["shape"], "the wrapper never ran — the registry was not consulted"
    assert torch.equal(recomputed, clean), (
        "recomputing the attention output from an unmodified pattern did not "
        "reproduce the model's own result — the post-softmax math has drifted"
    )


def test_the_wrapper_leaves_a_deltanet_mixer_alone(qwen35moe_bundle):
    """Scoping against a mixer that is not an attention module at all.

    ⚠️ Weak on its own, and kept only as the cheap half: a DeltaNet layer never
    consults ``ALL_ATTENTION_FUNCTIONS``, so this passes even for a wrapper
    that ignored its edit map entirely and edited every module it saw. The
    test below is the one that can fail.
    """
    encoded = qwen35moe_bundle.tokenizer(BASE_TEXT, return_tensors="pt")
    with torch.no_grad():
        clean = qwen35moe_bundle.model(**encoded).logits.clone()

    def wreck(probs: torch.Tensor) -> None:
        probs.zero_()

    # a module that never runs an eager attention forward (DeltaNet layer)
    other = qwen35moe_bundle.mixer_at(DELTANET_LAYER)
    with eager_attention_writes({id(other): wreck}):
        with torch.no_grad():
            same = qwen35moe_bundle.model(**encoded).logits.clone()
    assert torch.equal(same, clean)


def test_the_wrapper_only_touches_the_modules_it_was_given(llama_bundle):
    """Scoping, on a model where an unedited module DOES reach the wrapper.

    The registry entry is global while installed, so every full-attention
    forward in the model goes through it — the edit map is the only thing
    keeping one layer's edit off another's pattern. 📐 tiny-random-Llama has
    two full-attention layers, so an implementation that ignored the map would
    call the edit twice, and one that keyed it wrongly would call it zero
    times. Both are visible here; neither is visible against a DeltaNet mixer.
    """
    assert len(llama_bundle.blocks) == 2, "this test needs two attention layers"
    encoded = llama_bundle.tokenizer(BASE_TEXT, return_tensors="pt")

    seen: list[tuple[int, ...]] = []

    def record(probs: torch.Tensor) -> None:
        seen.append(tuple(probs.shape))

    edited = llama_bundle.mixer_at(0)
    with eager_attention_writes({id(edited): record}):
        with torch.no_grad():
            llama_bundle.model(**encoded)

    assert len(seen) == 1, (
        f"the edit ran {len(seen)} times for a map naming one of two attention "
        "layers — the wrapper is not scoping by module"
    )


def test_an_edit_to_one_layer_leaves_the_other_layers_pattern_intact(llama_bundle):
    """The same scoping property, checked on the tensor rather than the count.

    Layer 0's pattern is zeroed while layer 1's is merely observed. If the
    wrapper shared one edit across modules, layer 1's rows would come back
    zeroed too; they must still sum to 1, because layer 1 is a *reader* in the
    map. That both entries are honoured separately is the point — the map is
    per module, not a single global callback.
    """
    encoded = llama_bundle.tokenizer(BASE_TEXT, return_tensors="pt")
    later: list[torch.Tensor] = []

    def wreck(probs: torch.Tensor) -> None:
        probs.zero_()

    def observe(probs: torch.Tensor) -> None:
        later.append(probs.clone())

    with eager_attention_writes(
        {
            id(llama_bundle.mixer_at(0)): wreck,
            id(llama_bundle.mixer_at(1)): observe,
        }
    ):
        with torch.no_grad():
            llama_bundle.model(**encoded)

    (pattern,) = later
    rows = pattern.sum(dim=-1)
    assert torch.allclose(rows, torch.ones_like(rows), atol=1e-5), (
        "layer 1's pattern did not sum to 1 — layer 0's zeroing edit reached it"
    )


# --------------------------------------------------------------------------- #
# registry hygiene
# --------------------------------------------------------------------------- #


def test_the_registry_is_restored_afterwards(qwen35moe_bundle):
    """Leaving the key installed would keep the wrapper in force for the rest of
    the process — quietly, and for every later model."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    assert "eager" not in ALL_ATTENTION_FUNCTIONS
    attn = qwen35moe_bundle.mixer_at(FULL_ATTENTION_LAYER)
    with eager_attention_writes({id(attn): lambda p: None}):
        assert "eager" in ALL_ATTENTION_FUNCTIONS
    assert "eager" not in ALL_ATTENTION_FUNCTIONS


def test_the_registry_is_restored_after_an_exception(qwen35moe_bundle):
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    attn = qwen35moe_bundle.mixer_at(FULL_ATTENTION_LAYER)
    with pytest.raises(RuntimeError):
        with eager_attention_writes({id(attn): lambda p: None}):
            raise RuntimeError("boom")
    assert "eager" not in ALL_ATTENTION_FUNCTIONS


def test_no_edits_installs_nothing(qwen35moe_bundle):
    """A run with no pattern writes must not pay for, or risk, the patch."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    with eager_attention_writes({}):
        assert "eager" not in ALL_ATTENTION_FUNCTIONS


def test_the_backend_now_declares_the_capability():
    """The capability check refused these documents before; it must accept
    them now."""
    assert "writable_attention_probs" in PytorchHooksEngine.capabilities


# --------------------------------------------------------------------------- #
# refusals: what this component will not approximate
# --------------------------------------------------------------------------- #


def test_the_tap_declares_two_position_axes_and_so_has_no_contract(
    qwen35moe_bundle,
):
    """The shape names all four axes, and it is that description — not a magic
    marker — that says there is no ``(batch, position, feature)`` form. Every
    refusal below follows from it."""
    site = resolve_site(
        qwen35moe_bundle,
        SiteSpec(component="attention_probs", layers=(FULL_ATTENTION_LAYER,)),
    )
    assert [a.kind for a in site.shape.axes] == [
        "batch",
        "head",
        "position",
        "key_position",
    ]
    assert not site.shape.has_contract_form
    assert not site.shape.is_feature_space
    assert site.tuple_index == 1  # (attn_output, attn_weights)


@pytest.mark.parametrize(
    "kwargs, needle",
    [
        ({"pos": {"index": 1}}, "addresses positions"),
        ({"pos": {"span": [0, 2]}}, "addresses positions"),
        ({"featurizer": True}, "featurizes"),
        ({"dims": [0, 1]}, "slices 'dims'"),
    ],
    ids=["pos_index", "pos_span", "featurizer", "dims"],
)
def test_the_forms_the_shape_cannot_support_are_refused(
    qwen35moe_bundle, kwargs, needle
):
    """Each of these would silently read the wrong axis: ``_gather`` would index
    heads with positions, and ``dims`` would select key positions as features.

    Every message is *generated* from the tap's declared axes rather than
    written per component, so each one quotes the shape it is refusing on
    behalf of."""
    with pytest.raises(ProtocolError) as excinfo:
        executor_for(
            _read_doc(**kwargs), qwen35moe_bundle, base_texts=[BASE_TEXT]
        ).read_value("r")
    message = str(excinfo.value)
    assert needle in message
    # the refusal names the axes it is refusing on behalf of
    assert "key_position[key]" in message


def test_a_generated_frame_read_refuses(llama_bundle):
    """The decode path needs the same refusal, and used to say so in torch's words.

    📐 Before this refusal the same document raised a bare
    ``RuntimeError: Sizes of tensors must match except in dimension 1.
    Expected size 9 but got size 10`` out of ``torch.cat`` — a KV-cached step
    attends over the whole cache, so the key axis grows by one per step while
    the query axis stays 1, and the per-step patterns do not stack. The author
    of the document learns nothing from that message; they learn the shape of
    our sink.

    Refused now on the same grounds as the prompt-frame forms above: stacking
    decode steps on the position axis needs a tap that *has* one position axis,
    and this one has two.
    """
    doc = {
        "header": {"protocol_version": "4"},
        "model": {"key": "test", "revision": "main"},
        "data": base_data_section(with_counterfactual=False),
        "method": {
            "intervened_models": {"original": {"input": "base", "reads": ["r"]}},
            "positions": {"window": {"generated": {"max_new_tokens": 4}, "all": True}},
            "sites": {"tap": {"component": "attention_probs", "layers": [0]}},
            "reads": {"r": {"site": "tap", "pos": "window"}},
            "save": [saved("r", "original", "a.safetensors")],
        },
    }
    with pytest.raises(ProtocolError) as excinfo:
        executor_for(doc, llama_bundle, base_texts=[BASE_TEXT]).read_value("r")
    message = str(excinfo.value)
    assert "generated frame" in message
    assert "no single axis the decode steps stack along" in message
    assert "key_position[key]" in message


def test_a_deltanet_layer_refuses_on_the_architecture(qwen35moe_bundle):
    """The stream check still owns this, and it must keep owning it: at a
    Gated DeltaNet layer there is no attention matrix, which stays true now that
    the component is implemented."""
    with pytest.raises(ProtocolError) as excinfo:
        executor_for(
            _read_doc(layer=DELTANET_LAYER),
            qwen35moe_bundle,
            base_texts=[BASE_TEXT],
        ).read_value("r")
    assert "full-attention mixer" in str(excinfo.value)


def test_an_interchange_across_different_lengths_refuses(qwen35moe_bundle):
    """A whole-pattern swap needs both inputs to have the same position count —
    the pattern is (seq, seq), so a mismatch is not broadcastable. Refuse rather
    than let a broadcast produce a plausible tensor."""
    with pytest.raises(ProtocolError) as excinfo:
        executor_for(
            _swap_doc(),
            qwen35moe_bundle,
            base_texts=[BASE_TEXT],
            counterfactual_texts=["a slow green turtle sleeps very deeply indeed"],
        ).read_value("after")
    assert "same" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# the wrapper's math belongs to the module's own family
# --------------------------------------------------------------------------- #


def test_the_eager_math_is_the_modules_own(qwen35moe_bundle, llama_bundle):
    """The registry entry intercepts every attention forward while installed,
    so the wrapper must resolve `eager_attention_forward` from each module's
    own modeling file — borrowing one family's function would silently replace
    another's math (gemma-2's eager soft-caps the logits, for instance)."""
    from causalab.neural.engines.pytorch_hooks.attention_interface import (
        module_eager_attention,
    )

    qwen_eager = module_eager_attention(qwen35moe_bundle.mixer_at(3))
    llama_eager = module_eager_attention(llama_bundle.mixer_at(0))
    assert "qwen3_5" in qwen_eager.__module__
    assert "llama" in llama_eager.__module__
    assert qwen_eager is not llama_eager


def test_a_family_without_eager_math_refuses_by_name():
    """A modeling file that exports no eager function cannot be wrapped, and
    approximating it with another family's math is exactly the silent drift
    this module exists to prevent — so it refuses, naming what is missing."""
    from causalab.neural.engines.pytorch_hooks.attention_interface import (
        module_eager_attention,
    )

    class FakeMixer(torch.nn.Module):
        pass

    with pytest.raises(ProtocolError) as excinfo:
        module_eager_attention(FakeMixer())
    assert "eager_attention_forward" in str(excinfo.value)


def test_the_pattern_is_writable_on_a_family_with_no_repeat_kv():
    """The capability that dropping the recompute bought, as a behaviour
    rather than a grep.

    📐 The recompute this module used to carry needed ``repeat_kv``, and GPT-2's
    modeling file does not export it — it has no GQA, so it has nothing to
    repeat. A pattern write on gpt2 therefore refused by name. Intercepting the
    softmax's *output* hands the edited pattern back to the model's own value
    multiply, so no second symbol is needed and the family that could not be
    written now can.
    """
    import transformers.models.gpt2.modeling_gpt2 as gpt2_modeling

    assert hasattr(gpt2_modeling, "eager_attention_forward")
    assert not hasattr(gpt2_modeling, "repeat_kv"), "fixture assumption"

    bundle = load_model(TINY_GPT2)
    encoded = bundle.tokenizer(BASE_TEXT, return_tensors="pt")
    attn = bundle.mixer_at(1)
    with torch.no_grad():
        clean = bundle.model(**encoded).logits.clone()

    def wreck(probs: torch.Tensor) -> None:
        probs.zero_()

    with eager_attention_writes({id(attn): wreck}):
        with torch.no_grad():
            wrecked = bundle.model(**encoded).logits.clone()
    assert float((wrecked - clean).abs().max()) > 1e-4


def test_an_identity_edit_is_bit_identical_on_gpt2_too():
    """And the identity pin holds there, which is what says the write landed in
    the right place rather than merely somewhere."""
    bundle = load_model(TINY_GPT2)
    encoded = bundle.tokenizer(BASE_TEXT, return_tensors="pt")
    attn = bundle.mixer_at(1)
    with torch.no_grad():
        clean = bundle.model(**encoded).logits.clone()
    seen: dict[str, object] = {}
    with eager_attention_writes({id(attn): lambda p: seen.__setitem__("ran", True)}):
        with torch.no_grad():
            again = bundle.model(**encoded).logits.clone()
    assert seen.get("ran"), "the wrapper never ran"
    assert torch.equal(again, clean)


def test_an_identity_edit_is_bit_identical_on_llama(llama_bundle):
    """The identity pin, on a second family: this exercises the per-module
    resolution end to end — the wrapper must find and defer to *llama's*
    modeling file, not the qwen one the fixture above uses."""
    encoded = llama_bundle.tokenizer(BASE_TEXT, return_tensors="pt")
    attn = llama_bundle.mixer_at(0)

    with torch.no_grad():
        clean = llama_bundle.model(**encoded).logits.clone()

    seen: dict[str, object] = {}

    def identity(probs: torch.Tensor) -> None:
        seen["shape"] = tuple(probs.shape)

    with eager_attention_writes({id(attn): identity}):
        with torch.no_grad():
            recomputed = llama_bundle.model(**encoded).logits.clone()

    assert seen["shape"], "the wrapper never ran — the registry was not consulted"
    assert torch.equal(recomputed, clean)


def test_writing_the_pattern_changes_the_logits_on_llama(llama_bundle):
    """nnterp check 3 on the second family, through a real document.

    The counterfactual is 8 tokens under *llama's* tokenizer, matching
    BASE_TEXT — the module-level CF_TEXT is length-matched for the qwen
    fixture and comes out one token long here."""
    executor = executor_for(
        _swap_doc(layer=0),
        llama_bundle,
        base_texts=[BASE_TEXT],
        counterfactual_texts=["a big red dog barks loud"],
    )
    clean = executor.read_value("clean")
    after = executor.read_value("after")
    assert float((after - clean).abs().max()) > 0.0


# --------------------------------------------------------------------------- #
# pattern writes are interchanges: only `swap` means "replace the pattern"
# --------------------------------------------------------------------------- #


def test_a_non_swap_pattern_write_is_refused(qwen35moe_bundle):
    """A delta or clamp on the pattern would leave rows that no longer sum
    to 1, and the whole-pattern branch would misread its payload as a
    replacement anyway — refused by name."""
    doc = _swap_doc()
    doc["method"]["writes"]["patch"] = {
        "site": "tap",
        "pos": "all",
        "do": {"clamp": {"lo": 0.0, "hi": 0.0}},
    }
    del doc["method"]["reads"]["v_cf"]
    del doc["method"]["intervened_models"][UNWRITTEN]  # nothing reads it now
    with pytest.raises(ProtocolError) as excinfo:
        executor_for(
            doc,
            qwen35moe_bundle,
            base_texts=[BASE_TEXT],
            counterfactual_texts=[CF_TEXT],
        ).read_value("after")
    message = str(excinfo.value)
    assert "clamp" in message and "swap" in message
