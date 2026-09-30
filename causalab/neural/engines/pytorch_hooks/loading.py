"""Load models, tokenizers, and bundles for the hooks engine.

The cache key includes model configuration and attention backend.
Tokenizers use left padding and EOS as a pad token when needed. The loader
registers static model metadata. Eager attention is the default; documents
can choose another backend, with temporary eager forwards for interior taps.

``load_model`` prepares and caches a library-owned model.
``ModelBundle.from_model`` wraps a caller-owned model, checks required
preparation, and names any action the caller must take. Caller bundles
stay outside the loader cache.
"""

from __future__ import annotations

import dataclasses
import functools
import json
import os
from pathlib import Path
from typing import Any, Mapping

import torch

from causalab.neural.engines.pytorch_hooks.shard_read import LoadReport
from causalab.neural.engines.pytorch_hooks.sharding import Sharding
from causalab.neural.engines.pytorch_hooks.residency import Residency
from causalab.neural.engines.pytorch_hooks.weights import load_planned, load_pretrained
from causalab.neural.shared import model_tree
from causalab.neural.shared.compile_cache import configure as configure_compile_cache
from causalab.neural.shared.devices import DeviceMap
from causalab.neural.shared.kernels import bind_kernel_path
from causalab.neural.shared.normalized_cache import normalized_cache
from causalab.neural.shared.parallel.standin import Shadowed
from causalab.io.tensor_files import BundlePoint, TensorBundle
from causalab.protocol.parallel import ONE, ParallelGeometry, format_geometry
from causalab.protocol.rules.errors import ProtocolError
from causalab.protocol.registry import (
    FamilyAdapter,
    ModelInfo,
    family_for,
    model_info_from_hf_config,
    register_model,
    walk,
)

__all__ = [
    "LOAD_REPORT_VARIABLE",
    "BundlePoint",
    "DeviceMap",
    "ModelBundle",
    "Sharding",
    "TensorBundle",
    "load_model",
    "load_report_path",
    "write_load_report",
]

#: Debug output, opt in: a directory into which every **sharded** load
#: writes its [`LoadReport`][] — the bytes the reader was asked for
#: against the bytes on disk, per parameter — as
#: ``load_report.rank<N>.json`` ([`load_report_path`][]), one file per
#: rank of the world. The fact belongs to one rank's process, which is why
#: it is not in the run receipt (the joiner's, one per campaign): the
#: parallel golden reads every rank's file and holds a sharded parameter on
#: a real checkpoint to exactly ``1 / world`` (``docs/model_parallelism.md``
#: §5.3, §10.6). Unset, nothing is written.
LOAD_REPORT_VARIABLE = "CAUSALAB_LOAD_REPORT_DIR"

_DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


@dataclasses.dataclass(frozen=True)
class ModelBundle:
    """One loaded model with everything the executor needs."""

    key: str
    revision: str
    model: Any
    tokenizer: Any
    info: ModelInfo
    #: Where the model's layers are: the device of the embedding, of every
    #: block and of the head (``shared/devices.py``). On a load this is read
    #: off the placed parameters with the user's ``--device`` word kept as
    #: ``requested``; on a caller-owned model it is derived from the model
    #: alone. A quantized load records the **requested** map instead —
    #: bitsandbytes places the weights itself and moving them afterwards is
    #: refused — so a disagreement there surfaces as a loud device mismatch
    #: at the first forward, never as quiet wrong numbers. Inputs are encoded
    #: onto ``devices.embedding``; a capture stays where its block produced
    #: it; a write's operand meets the written tensor on its block's device.
    devices: DeviceMap
    dtype: str
    quantization: dict[str, Any] | None = None
    #: The geometry this bundle was loaded under (``docs/model_parallelism.md``
    #: §2): all ones for a one-process load and for every caller-owned model.
    #: Above world 1 the sharded parameters are DTensors carrying their own
    #: placements, so nothing per parameter is recorded here. Execution,
    #: never identity.
    geometry: ParallelGeometry = ONE
    #: What the weight reader was asked for against what is on disk, per
    #: parameter — recorded by a sharded load (§5.3); ``None`` for a world-1
    #: load, a quantized one and a caller-owned model.
    load_report: LoadReport | None = None
    #: What this rank holds against that report (``residency.py``); ``None``
    #: at world 1, with the report.
    residency: Residency | None = None

    @functools.cached_property
    def adapter(self) -> FamilyAdapter:
        """The registered family whose predicate recognizes this model's
        module tree (``registry.family_for``) — detected once per
        bundle, structurally, never off the config. Everything below that
        used to be a ``hasattr`` on the tree routes through it."""
        return family_for(self.model)

    @property
    def is_gpt2_family(self) -> bool:
        return self.adapter.family == "gpt2_tree"

    @property
    def blocks(self) -> Any:
        """The decoder-layer ModuleList, whichever tree this family uses."""
        return self.adapter.blocks_of(self.model)

    def stream_at(self, layer: int) -> str:
        """Which mixer stream ``layer`` actually carries — delegated to the
        shared table ([`causalab.neural.shared.model_tree`][]), because the
        per-layer hybrid answer must never diverge between engines. A block
        another pipeline stage holds is answered from the entry's
        ``layer_types`` (``docs/model_parallelism.md`` §6.5)."""
        return model_tree.stream_at(
            self.blocks,
            layer,
            key=self.key,
            mixers=self.adapter.mixers,
            layer_types=self.info.layer_types,
        )

    def mixer_at(self, layer: int) -> Any:
        """The attention/mixer module at ``layer``, whichever stream it is."""
        return model_tree.mixer_at(
            self.blocks,
            layer,
            key=self.key,
            mixers=self.adapter.mixers,
            layer_types=self.info.layer_types,
        )

    def holds(self, layer: int) -> bool:
        """Whether this rank holds block ``layer`` — false for the stand-in a
        pipeline stage keeps for another stage's block
        (``docs/model_parallelism.md`` §6.5; ``sharding.place_stage``): the
        shadowing identity, or a bare parameterless identity. A structural
        fact of the loaded tree; the resume swap's stand-ins for a held
        block (``executor._resumed``) are neither."""
        block = self.blocks[layer]
        if isinstance(block, Shadowed):
            return False
        return not (
            isinstance(block, torch.nn.Identity)
            and next(iter(block.parameters()), None) is None
        )

    @property
    def streams(self) -> tuple[str, ...]:
        """``stream_at`` for every layer — the whole tower's shape at a glance."""
        return tuple(self.stream_at(i) for i in range(len(self.blocks)))

    @classmethod
    def from_model(
        cls,
        model: Any,
        tokenizer: Any,
        *,
        key: str,
        revision: str,
        dtype: str,
        quantization: Mapping[str, Any] | None = None,
    ) -> "ModelBundle":
        """Wrap a model the **caller** owns — the supported way in for a model
        that is already loaded (spec §9, the ownership contract).

        ``info`` is derived exactly as [`load_model`][] derives it
        ([`model_info_from_hf_config`][causalab.protocol.registry.models.model_info_from_hf_config] + [`register_model`][causalab.protocol.registry.models.register_model]), so the
        engine's tap table reads the same registry row either way. ``key``
        and ``revision`` are the caller's *assertion*: nothing here can check
        them against the weights, and the run receipt stamps them as given.
        The device map is **not** asserted: it is read off the model's own
        parameters ([`DeviceMap.of_modules`][]), and a block whose
        parameters straddle devices is refused by index. A caller model
        spread over several devices must carry its own crossings (the hooks
        accelerate's ``device_map`` installs, or the engine's); the bundle
        installs none, because it mutates nothing.

        Nothing about ``model`` or ``tokenizer`` is mutated. [`load_model`][]
        prepares what it loads — ``.eval()``,
        ``.requires_grad_(False)``, left padding with a pad token — and each
        of those changes the numbers or what the hooks see, so an object that
        lacks one is **refused** with the exact call to make rather than
        quietly re-moded, moved or re-configured behind the caller's back
        (`_refuse_unprepared`). The bundle is never inserted into
        [`load_model`][]'s cache.

        One thing this method does reach beyond the model: when
        ``CAUSALAB_COMPILE_CACHE`` is set, the process's compiler cache
        variables are pointed at the shared root here, as [`load_model`][]
        does for a model it loads (``shared/compile_cache.py``) — a
        caller-owned model compiles its kernels on first use like any other.

        The caller's attention backend is accepted unchanged. During execution,
        forwards needing attention-function interiors temporarily use eager;
        the executor restores the caller's backend on every exit path.

        Raises:
            ProtocolError: the model or tokenizer is not prepared the way a
                loaded one is (one refusal, one call named), or ``dtype`` is
                not a precision the engine knows.
        """
        if dtype not in _DTYPES:
            raise ProtocolError(
                "P4",
                f"dtype {dtype!r} is not one of {sorted(_DTYPES)} — the same "
                "closed set a document's model.dtype draws from (§2.1)",
            )
        _refuse_unprepared(
            model, tokenizer, dtype=dtype, quantized=quantization is not None
        )
        info = model_info_from_hf_config(key, model.config)
        register_model(info)
        devices = _placement_of(model)
        # a caller-owned model compiles its kernels on first use like a loaded
        # one; the shared cache root applies the same way
        configure_compile_cache(devices.requested)
        return cls(
            key=key,
            revision=revision,
            model=model,
            tokenizer=tokenizer,
            info=info,
            devices=devices,
            dtype=dtype,
            quantization=dict(quantization) if quantization is not None else None,
        )


def _placement_of(
    model: Any, requested: str | None = None, *, empty: torch.device | None = None
) -> DeviceMap:
    """The map a model realizes, read off its parameters: the family's tree
    names the embedding, the blocks and the head, and each is on one device
    or refused by name ([`DeviceMap.of_modules`][]). ``requested`` is the
    user's spelling to keep for the record; ``None`` keeps the canonical one.
    ``empty`` is the device a module with no tensors is recorded on — a
    pipeline stage's identity layers, on the rank's device."""
    adapter = family_for(model)
    embedding = walk(model, adapter.tree.embedding)
    head = walk(model, adapter.tree.lm_head)
    if embedding is None or head is None:
        raise ProtocolError(
            "P4",
            f"family {adapter.family!r} addresses its embedding at "
            f"{adapter.tree.embedding!r} and its head at {adapter.tree.lm_head!r}, "
            f"but this model ({type(model).__name__}) lacks one of them",
        )
    placed = DeviceMap.of_modules(
        embedding, list(adapter.blocks_of(model)), head, empty=empty
    )
    if requested is None:
        return placed
    return dataclasses.replace(placed, requested=requested)


def _refuse_unprepared(
    model: Any, tokenizer: Any, *, dtype: str, quantized: bool
) -> None:
    """Refuse a caller-owned model the loader would have had to prepare.

    One check per preparation [`load_model`][] makes on a model it owns, in
    the loader's own order, each naming the one expression the caller runs
    instead — fail closed on an object the library does not own, and only
    where a loaded run's numbers would differ:

    * **eval mode** — dropout and other train-time branches change the
      numbers on every forward;
    * **frozen weights** — a train document accumulates gradients into any
      parameter that requires them (§2.11 trains featurizers only);
    * **the padding convention** — the position frame is built for left
      padding, and encoding needs a pad token;
    * **the declared precision** — a weight whose dtype is not the declared
      one produces a run the record would misdescribe (skipped on a quantized
      model, whose weights are integer tensors by construction).
    """
    how = "prepare the model the way load_model does"
    if getattr(model, "training", False):
        raise ProtocolError(
            "P4",
            f"the caller-owned model is in train mode — {how}: call "
            "model.eval() before handing it over",
        )
    if any(p.requires_grad for p in model.parameters()):
        raise ProtocolError(
            "P4",
            "the caller-owned model has parameters that require grad, and a "
            "train document would accumulate gradients into them — "
            f"{how}: call model.requires_grad_(False) before handing it over",
        )
    if getattr(tokenizer, "padding_side", None) != "left":
        raise ProtocolError(
            "P4",
            f"the caller-owned tokenizer pads on the {tokenizer.padding_side!r}, "
            "and the engine's position frame is built for left padding — "
            f'{how}: set tokenizer.padding_side = "left" before handing it over',
        )
    if getattr(tokenizer, "pad_token", None) is None:
        raise ProtocolError(
            "P4",
            "the caller-owned tokenizer has no pad token, and batches are "
            f"padded — {how}: set tokenizer.pad_token = tokenizer.eos_token "
            "before handing it over",
        )
    if not quantized:
        wanted = _DTYPES[dtype]
        found = {p.dtype for p in model.parameters() if p.is_floating_point()} - {
            wanted
        }
        if found:
            raise ProtocolError(
                "P4",
                f"the bundle declares dtype {dtype!r} ({wanted}) but the "
                f"caller-owned model holds parameters in {sorted(map(str, found))}"
                " — the run receipt would describe a precision that did not "
                f"run; declare the model's actual dtype, or call model.to({wanted}) "
                "before handing it over",
            )


def quantization_key(
    quantization: Mapping[str, Any] | None,
) -> tuple[tuple[str, Any], ...] | None:
    """The one canonical form of a document's materialized ``model.quantization``
    block, for the cache key: its fields as pairs sorted by name, ``None`` for
    an unquantized realization. Equal blocks give equal keys whatever their
    order, so a realization is one cache entry (§2.1: quantization is a
    document fact, part of identity)."""
    if quantization is None:
        return None
    return tuple(sorted(quantization.items()))


@normalized_cache(maxsize=4, keys={"quantization": quantization_key})
def load_model(
    key: str,
    revision: str = "main",
    *,
    dtype: str = "fp32",
    device: str = "cpu",
    quantization: Mapping[str, Any] | None = None,
    attn_implementation: str | None = "eager",
    sharding: Sharding | None = None,
) -> ModelBundle:
    """Load (and cache) one model bundle.

    The tokenizer is set to the engine's single padding convention: left
    padding with ``pad = eos`` when the checkpoint ships no pad token — the
    inherited pipeline contract the oracle tests were captured under.

    Four bundles stay resident, keyed on the *bound* arguments
    ([`normalized_cache`][]): an
    omitted default and its explicit spelling, positional and keyword, are one
    entry; revision, precision, device, quantization and the attention
    selection are each part of identity, and ``attn_implementation=None`` is
    distinct from ``"eager"``. ``load_model.cache_clear()`` and
    ``cache_info()`` manage the cache.

    ``quantization`` is the document's materialized ``model.quantization``
    block; [`quantization_key`][] is the one place its cache identity is
    defined. The realization is a document fact, not an engine flag (§2.1).

    ``attn_implementation`` selects a Transformers attention backend. Eager
    remains the reproducible default; ``None`` uses Transformers' default.
    Interior taps temporarily switch a forward to eager and restore this
    selection afterward.

    ``device`` is one device or a comma list placing the layers across the
    devices of this process ([`DeviceMap.parse`][]; ``weights.py``). The
    cache ([`normalized_cache`][],
    keyed on the bound arguments) holds the string as given —
    ``DeviceMap.requested`` — so ``cuda`` and ``cuda:0`` are two cache
    entries of one placement, which the caller-bundle check nonetheless
    compares as equal.

    ``sharding`` is this rank's place in a geometry above world 1
    (``docs/model_parallelism.md`` §5.2–5.3): the registry's plan is applied
    over its meshes and each rank reads its own shard of the weights
    (``weights.load_planned``). It is part of the cache key by ``(geometry,
    rank)`` — ``Sharding``'s equality, its meshes outside it; ``None`` is
    today's one-process load, byte for byte.
    """
    from transformers import AutoModelForCausalLM

    from causalab.io.tokenizer import load_tokenizer

    # the compilers a CUDA model will use (Triton, TileLang, Inductor) are
    # pointed at the shared cache root, when one is set, before the first
    # kernel is built (shared/compile_cache.py). The loader is the seam rather
    # than the CLI's entry: the library is imported at least as often as it is
    # run from the command line, and the device is only known once a model is
    # placed — so every path a model can land on calls this, and the setting
    # is process-wide (the last call wins)
    configure_compile_cache(device)

    report: LoadReport | None = None
    residency: Residency | None = None
    if sharding is not None and quantization is not None:
        raise ProtocolError(
            "P4",
            f"a geometry above world 1 ({format_geometry(sharding.geometry)}) and "
            "quantized weights do not compose: bitsandbytes places its own "
            "weights (docs/model_parallelism.md §6.6, §11)",
        )
    if sharding is not None:
        report_dir = os.environ.get(LOAD_REPORT_VARIABLE)
        loaded = load_planned(
            key,
            revision,
            dtype=_DTYPES[dtype],
            device=device,
            attn_implementation=attn_implementation,
            sharding=sharding,
            # the live-tensor census is for the written report alone
            census=bool(report_dir),
        )
        model, report, residency = loaded.model, loaded.report, loaded.residency
        if report_dir:
            write_load_report(
                report,
                sharding,
                Path(report_dir),
                key=key,
                revision=revision,
                residency=residency,
            )
        # a pipeline stage's identity layers have no tensors: they run on
        # the rank's one device, which is what the map records for them
        devices = _placement_of(
            model, requested=device, empty=DeviceMap.parse(device, 1).single
        )
    elif quantization is None:
        # weights read straight onto their devices, several shards at once
        # (``weights.py``); the stock CPU load + ``.to(device)`` is one
        # thread end to end, and slower for it
        model = load_pretrained(
            key,
            revision,
            dtype=_DTYPES[dtype],
            device=device,
            attn_implementation=attn_implementation,
        )
        devices = _placement_of(model, requested=device)
    else:
        # bitsandbytes quantizes on the way in and places the weights itself;
        # moving them afterwards is refused, so the requested map is only
        # recorded (``ModelBundle.devices``) — and quantized weights are
        # single-device (docs/model_parallelism.md §11)
        devices = DeviceMap.parse(
            device, model_info_from_hf_config(key, _config_of(key, revision)).num_layers
        )
        if devices.single is None:
            raise ProtocolError(
                "P4",
                f"device {device!r} places the layers across several devices, "
                "and quantized weights are placed by bitsandbytes on one; run a "
                "quantized document on one device",
            )
        model = AutoModelForCausalLM.from_pretrained(
            key,
            revision=revision,
            dtype=_DTYPES[dtype],
            **(
                {"attn_implementation": attn_implementation}
                if attn_implementation is not None
                else {}
            ),
            quantization_config=_bitsandbytes_config(dict(quantization)),
        )
    model.eval()
    # only featurizer/free params ever train (§2.11); freezing the network
    # keeps training graphs from accumulating gradients into model weights
    model.requires_grad_(False)
    # a DeltaNet family's kernel globals follow the devices this load put the
    # weights on: the torch path off CUDA, the installed kernels on it — one
    # answer for the tower, since the map refuses mixing — so a bare forward
    # of the model works wherever it lives (shared/kernels.py)
    bind_kernel_path(model, on_cuda=devices.is_cuda)
    tokenizer = load_tokenizer(key, revision)
    info = model_info_from_hf_config(key, model.config)
    register_model(info)
    return ModelBundle(
        key=key,
        revision=revision,
        model=model,
        tokenizer=tokenizer,
        info=info,
        devices=devices,
        dtype=dtype,
        quantization=dict(quantization) if quantization is not None else None,
        geometry=ONE if sharding is None else sharding.geometry,
        load_report=report,
        residency=residency,
    )


def load_report_path(directory: Path, rank: int) -> Path:
    """Where rank ``rank`` of a sharded load writes its report under
    [`LOAD_REPORT_VARIABLE`][]."""
    return directory / f"load_report.rank{rank}.json"


def write_load_report(
    report: LoadReport,
    sharding: Sharding,
    directory: Path,
    *,
    key: str,
    revision: str,
    residency: Residency | None = None,
) -> Path:
    """Write one rank's [`LoadReport`][] as JSON ([`LOAD_REPORT_VARIABLE`][]):
    the model, the geometry, this rank, per parameter the bytes requested,
    the bytes on disk and the elements requested, and — with ``residency``
    — what the rank holds against them (``Residency.record``, the block
    ``residency_problems`` reads) — ``indent=2, sort_keys=True`` and a
    trailing newline, like every other record this package writes."""
    directory.mkdir(parents=True, exist_ok=True)
    target = load_report_path(directory, sharding.rank)
    record: dict[str, Any] = {
        "key": key,
        "revision": revision,
        "geometry": format_geometry(sharding.geometry),
        "world": sharding.geometry.world,
        "rank": sharding.rank,
        "bytes_requested": dict(report.bytes_requested),
        "bytes_on_disk": dict(report.bytes_on_disk),
        "elements_requested": dict(report.elements_requested),
        "dtype_on_disk": dict(report.dtype_on_disk),
    }
    if residency is not None:
        record.update(residency.record())
    target.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return target


def _config_of(key: str, revision: str) -> Any:
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(key, revision=revision)


def _bitsandbytes_config(quantization: dict[str, Any]) -> Any:
    """Lower a materialized ``model.quantization`` block to a
    ``BitsAndBytesConfig``.

    bitsandbytes is an optional extra: quantization is in the *document*
    vocabulary so that a shared protocol says which realization produced its
    numbers, and a reader without the library still gets a document that
    validates, digests and explains — only ``run`` needs the quantizer, and
    it says so precisely.

    Field mapping: https://huggingface.co/docs/transformers/main_classes/quantization
    """
    method = quantization.get("method", "bitsandbytes")
    if method != "bitsandbytes":
        raise ProtocolError(
            "P4", f"quantization method {method!r} has no reference implementation"
        )
    try:
        from transformers import BitsAndBytesConfig
        import bitsandbytes  # noqa: F401 — the config is inert without it
    except ImportError as err:
        raise ProtocolError(
            "P2",
            f"this document declares {quantization.get('scheme')!r} weight "
            "quantization, which the reference engine realizes through "
            f"bitsandbytes — not installed ({err}). Install the extra, or run "
            "the document at its unquantized precision by setting "
            "model.quantization out of it (a different experiment, and its "
            "digest says so).",
        ) from err

    scheme = quantization["scheme"]
    compute_dtype = _DTYPES[quantization.get("compute_dtype", "fp32")]
    if scheme == "int8":
        return BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=float(quantization.get("int8_threshold", 6.0)),
        )
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=scheme,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=bool(quantization.get("double_quant", False)),
    )
