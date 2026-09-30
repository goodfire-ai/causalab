"""A fourth model family, registered from outside ``causalab/neural/``.

The plugin contract's acceptance: *a new family is added without
touching* ``causalab/neural/``. This module is that family — a tiny decoder
whose tree shares **no child name** with the two built-in trees (``tower.stack``
for the blocks, ``chan_mix`` for the mixer, ``norm_a`` / ``norm_b`` / ``ffn``
inside a block, ``tok`` / ``final`` / ``head`` at the root) — and its
[`FamilyAdapter`][causalab.protocol.registry.families.FamilyAdapter]: a structural detection
predicate, the tree address, the mixer child and its stream, the taps for the
residual and MLP sites, and the two residual identities every block satisfies.
Nothing here imports ``causalab.neural``; the engine finds the family through
the registry alone.

The mixer is a per-position channel mix, not attention, and the stream
vocabulary is closed to the two mixers the protocol knows — so the adapter
declares the child as ``full_attention`` and declares **no attention-interior
tap**: naming ``attention_premix`` on this family is refused by the registry,
by name, which is the point (the ``o_proj`` ``AttributeError`` of the pre-fix
behaviour cannot happen: no module is looked up before the family has said it
serves the component).
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from transformers.modeling_outputs import CausalLMOutputWithPast

from causalab.protocol.registry import (
    FamilyAdapter,
    Identity,
    ModelInfo,
    Tap,
    TreeAddress,
    register_family,
    register_model,
    walk,
)

__all__ = [
    "FAMILY",
    "KEY",
    "INFO",
    "SYNTHETIC_TREE",
    "SynthModel",
    "build_model",
]

KEY = "synthetic/third-tree"
FAMILY = "synthetic_tree"
HIDDEN, INNER, LAYERS, VOCAB = 8, 16, 2, 32000  # the tiny-llama tokenizer's vocab


class SynthMixer(nn.Module):
    """A per-position channel mix standing where a mixer stands."""

    def __init__(self) -> None:
        super().__init__()
        self.mix = nn.Linear(HIDDEN, HIDDEN)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.mix(x))


class SynthFFN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.up = nn.Linear(HIDDEN, INNER)
        self.act = nn.GELU()
        self.down = nn.Linear(INNER, HIDDEN)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(self.act(self.up(x)))


class SynthBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm_a = nn.LayerNorm(HIDDEN)
        self.chan_mix = SynthMixer()
        self.norm_b = nn.LayerNorm(HIDDEN)
        self.ffn = SynthFFN()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mid = x + self.chan_mix(self.norm_a(x))
        return mid + self.ffn(self.norm_b(mid))


class SynthTower(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.tok = nn.Embedding(VOCAB, HIDDEN)
        self.stack = nn.ModuleList(SynthBlock() for _ in range(LAYERS))
        self.final = nn.LayerNorm(HIDDEN)


class SynthModel(nn.Module):
    """The causal LM: embeds, runs the stack, norms, projects to the vocabulary.
    Accepts the keyword arguments the executor passes an HF model and returns
    the HF output object, so the engine drives it like any other."""

    def __init__(self) -> None:
        super().__init__()
        self.tower = SynthTower()
        self.head = nn.Linear(HIDDEN, VOCAB, bias=False)
        self.config = _Config()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        use_cache: bool = False,
        **_: Any,
    ) -> CausalLMOutputWithPast:
        x = self.tower.tok(input_ids)
        for block in self.tower.stack:
            x = block(x)
        return CausalLMOutputWithPast(logits=self.head(self.tower.final(x)))


class _Config:
    """The one attribute anything reads off the config here."""

    _experts_implementation = None


def build_model(seed: int = 0) -> SynthModel:
    torch.manual_seed(seed)
    model = SynthModel()
    model.eval()
    model.requires_grad_(False)
    return model


INFO = ModelInfo(
    key=KEY,
    hidden_size=HIDDEN,
    num_layers=LAYERS,
    num_heads=1,
    num_kv_heads=1,
    head_dim=HIDDEN,
    intermediate_size=INNER,
    vocab_size=VOCAB,
)
register_model(INFO)

_EXACT = {"fp32": (0.0, 0.0)}

#: The family's plugin. Residual stream and MLP sites only; no attention
#: interior, no MoE, no DeltaNet — and no probe overrides (the shared
#: module-tree probes refuse the MoE components by the block's children).
SYNTHETIC_TREE = FamilyAdapter(
    family=FAMILY,
    detect=lambda model: walk(model, "tower.stack") is not None
    and walk(model, "tower.tok") is not None,
    tree=TreeAddress(
        blocks="tower.stack",
        embedding="tower.tok",
        final_norm="tower.final",
        lm_head="head",
        mlp="ffn",
    ),
    mixers={"chan_mix": "full_attention"},
    taps={
        "input_ids": Tap("embedding", kind="in"),
        "embeddings": Tap("embedding"),
        "ln_final": Tap("final_norm"),
        "lm_head": Tap("lm_head"),
        "block_input": Tap("block", kind="in"),
        "block_output": Tap("block"),
        "attention_input_norm": Tap("block", "norm_a"),
        "attention_output": Tap("mixer"),
        "block_mid": Tap("block", "norm_b", "in", writeback="block_output"),
        "mlp_input_norm": Tap("block", "norm_b"),
        "mlp_input": Tap("mlp", kind="in"),
        "mlp_output": Tap("mlp"),
        "mlp_activation": Tap("mlp", "act"),
    },
    identities=(
        Identity(
            "residual_mid",
            "block_mid",
            ("block_input", "attention_output"),
            "block_mid == block_input + attention_output",
            _EXACT,
            additive=True,
        ),
        Identity(
            "residual_out",
            "block_output",
            ("block_mid", "mlp_output"),
            "block_output == block_mid + mlp_output",
            _EXACT,
            additive=True,
        ),
    ),
)
register_family(SYNTHETIC_TREE)
