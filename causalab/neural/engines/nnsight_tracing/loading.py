"""Load nnsight models behind the shared bundle interface.

The bundle exposes model, tokenizer, metadata, blocks, and mixer access.
The common site resolver addresses envoys through the model tree.
Trace execution reads and assigns their inputs and outputs.

The document can select an attention backend. Omission uses the model's
Transformers default, usually SDPA. Interior taps temporarily enable eager
attention and restore the selection afterward. Declare eager explicitly
for comparisons with the hooks engine's default.
"""

from __future__ import annotations

import dataclasses
import functools
from typing import Any

import torch

from causalab.neural.shared.devices import DeviceMap
from causalab.neural.shared.compile_cache import configure as configure_compile_cache
from causalab.neural.shared.kernels import bind_kernel_path
from causalab.neural.shared.normalized_cache import normalized_cache
from causalab.neural.shared import model_tree
from causalab.protocol.parallel import ONE, ParallelGeometry
from causalab.protocol.rules.errors import ProtocolError
from causalab.protocol.registry import (
    FamilyAdapter,
    ModelInfo,
    family_for,
    model_info_from_hf_config,
    register_model,
)

__all__ = ["NnsightBundle", "load_model", "torch_module"]

_DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


@dataclasses.dataclass(frozen=True)
class NnsightBundle:
    """One loaded nnsight model with everything the executor needs.

    ``model`` is the `TransformersModel`: attribute access on it
    yields envoys mirroring the HF module tree, which is what lets the
    shared site map resolve against it unchanged.
    """

    key: str
    revision: str
    model: Any
    tokenizer: Any
    info: ModelInfo
    #: The device inputs are sent to. nnsight's dispatch places the *model*
    #: itself on first trace; a disagreement surfaces as a loud device
    #: mismatch at the first forward, never as quiet wrong numbers (the same
    #: trade the reference bundle documents).
    device: str
    dtype: str
    quantization: dict[str, Any] | None = None
    #: The nnsight engine is single-device (``docs/model_parallelism.md``
    #: §8.5): every bundle is a world-1 load, which is what the caller-bundle
    #: check compares an engine's geometry against.
    geometry: ParallelGeometry = ONE

    @functools.cached_property
    def adapter(self) -> FamilyAdapter:
        """The registered family whose predicate recognizes this model's
        module tree (``registry.family_for``) — detected once per
        bundle, structurally, never off the config. Everything below that
        used to be a ``hasattr`` on the tree routes through it."""
        return family_for(self.model)

    @functools.cached_property
    def devices(self) -> DeviceMap:
        """The one device as the map the shared services read
        ([`DeviceMap`][]): the nnsight engine is single-device
        (``docs/model_parallelism.md`` §8.5), so it is ``device`` everywhere."""
        return DeviceMap.parse(self.device, self.info.num_layers)

    @property
    def is_gpt2_family(self) -> bool:
        return self.adapter.family == "gpt2_tree"

    @property
    def blocks(self) -> Any:
        """The decoder-layer list (envoys), whichever tree this family uses."""
        return self.adapter.blocks_of(self.model)

    def stream_at(self, layer: int) -> str:
        """Which mixer stream ``layer`` carries — the shared table's answer
        ([`causalab.neural.shared.model_tree`][]), read off the envoy tree."""
        return model_tree.stream_at(
            self.blocks, layer, key=self.key, mixers=self.adapter.mixers
        )

    def mixer_at(self, layer: int) -> Any:
        """The attention/mixer envoy at ``layer``, whichever stream it is."""
        return model_tree.mixer_at(
            self.blocks, layer, key=self.key, mixers=self.adapter.mixers
        )

    @property
    def streams(self) -> tuple[str, ...]:
        """``stream_at`` for every layer — the whole tower's shape at a glance."""
        return tuple(self.stream_at(i) for i in range(len(self.blocks)))


def torch_module(envoy: Any) -> torch.nn.Module:
    """The torch module an nnsight envoy wraps (``Envoy._module``) — what the
    kernel-path binding inspects, and what a caller needing the plain module
    behind a bundle reads."""
    module = getattr(envoy, "_module", None)
    if not isinstance(module, torch.nn.Module):
        raise AssertionError("an nnsight model envoy wraps a torch module")
    return module


@normalized_cache(maxsize=4)
def load_model(
    key: str,
    revision: str = "main",
    *,
    dtype: str = "fp32",
    device: str = "cpu",
    attn_implementation: str | None = None,
) -> NnsightBundle:
    """Load (and cache) one nnsight bundle.

    The tokenizer is set to the engines' single padding convention: left
    padding with ``pad = eos`` when the checkpoint ships no pad token — the
    same contract the reference bundle loads under, so both engines encode
    identical batches.

    Four bundles stay resident, keyed on the *bound* arguments
    ([`normalized_cache`][]), so
    two spellings of one realization are one entry; ``load_model.cache_clear()``
    and ``cache_info()`` manage the cache.

    ``attn_implementation=None`` keeps the checkpoint's own default; the
    executor switches on demand for the traces that need another one. Passing
    one pins it — what the parity suite does to compare like against like.
    """
    from nnsight.modeling.transformers import TransformersModel

    # the compilers a CUDA model will use are pointed at the shared cache
    # root, when one is set, before the first kernel is built (the reference
    # loader does the same; shared/compile_cache.py says why the loader is
    # the seam)
    configure_compile_cache(device)
    model = TransformersModel(
        key,
        task="text-generation",
        revision=revision,
        dtype=_DTYPES[dtype],
        **(
            {}
            if attn_implementation is None
            else {"attn_implementation": attn_implementation}
        ),
    )
    tokenizer = model.tokenizer
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    info = model_info_from_hf_config(key, model.config)
    register_model(info)
    devices = DeviceMap.parse(device, info.num_layers)
    if devices.single is None:
        raise ProtocolError(
            "P4",
            f"device {device!r} places the layers across several devices, and "
            "the nnsight engine is single-device (docs/model_parallelism.md "
            "§8.5); name one device, or run the document on the reference engine",
        )
    # a DeltaNet family's kernel globals follow the device this bundle was
    # asked for — the torch path off CUDA, the installed kernels on it — so a
    # bare trace works wherever the weights land at dispatch (shared/kernels.py)
    bind_kernel_path(torch_module(model), on_cuda=devices.is_cuda)
    return NnsightBundle(
        key=key,
        revision=revision,
        model=model,
        tokenizer=tokenizer,
        info=info,
        device=device,
        dtype=dtype,
    )
