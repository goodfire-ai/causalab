"""The model-family readout adapter — final normalization, unembedding in a
declared accumulation dtype, and centering — keyed by the registry entry's
``family``, and the **logit lens** built on it.

**An analysis over a loaded model, not an engine service.** No engine and no
``neural/shared`` module imports this one: a readout is what an *analysis*
computes from a loaded bundle (``Readout.from_bundle``) once the engine has
produced the residual — the same relation ``fit_pca`` has to a saved read. It
therefore lives beside the other analyses and keeps their discipline: torch and
the neural layer are imported inside the functions that need them, so listing
or hashing a shipped script never pays for numerics
(``tests/test_architecture_layering.py``, ``tests/protocol/test_load_is_torch_free.py``).

**What a readout is.** ``logits = lm_head(final_norm(h))`` — the two module
calls the model itself makes after its last block. Both engines already
serve the two tensors (``ln_final``, ``lm_head`` are module-output taps at the
model root, ``registry.TreeAddress.final_norm`` / ``.lm_head``); what no
landed code declared is the *kind* of norm, its **gain convention**, where its
epsilon lives, and the dtype a downstream analysis should accumulate the
unembedding in. Those are exactly the facts a residual decomposition needs —
``N(v) = g · s · v`` linearises the final norm at the fixed scale ``s`` of the
whole residual, and ``g`` is ``weight`` on a Llama-style RMSNorm but
``1 + weight`` on Qwen3.5-MoE's — and code that assumes them ends up with
architecture branches. Here they are a declaration per family, measured against the module's own forward
([`Readout.certify`][]) rather than assumed.

**Module application, never weight slicing.** [`Readout.normalize`][],
[`Readout.logits`][] and [`Readout.unembed`][] *call the modules the
family's tree addresses*. That is the principle ``executor/base.py`` states for
the head-wise projection: running the projection the model's own module
defines cannot be wrong about its own layout (``nn.Linear`` vs ``Conv1D``,
tied vs untied embeddings). The unembedding in the declared accumulation dtype
is the same module forward with its parameters cast
(``torch.func.functional_call``) — no ``.weight`` is read here or by any
consumer.

**Keyed by ``ModelInfo.family``, not ``FamilyAdapter.family``.** The tree
family is too coarse: ``llama_tree`` detects both Llama and Qwen3.5-MoE, whose
final RMSNorms differ in the one fact the linearisation needs (the gain
convention). The HF ``model_type`` the registry entry records is the right
key — 📐 measured on the tiny fixtures (``gpt2``: ``LayerNorm``, gain
``weight``, epsilon ``eps``; ``llama``: ``LlamaRMSNorm``, ``weight``,
``variance_epsilon``; ``qwen3_5_moe_text``: ``Qwen3_5MoeRMSNorm``,
``one_plus_weight``, ``eps`` — the same class the real Qwen3.6-35B-A3B runs).

**Deliberately outside the protocol layer.** Nothing hashed imports this
module, and no ``SHARED`` member of the shipped scripts' closure may — so no
pinned digest depends on it. The readout is not document vocabulary
either: centering is invisible to every softmax-based metric (spec §2.9, a
uniform shift is a no-op) and shows only in raw ``token_logit`` values, so a
``center`` field or a ``centered_logits`` component is a schema decision
batched with the next legitimate ``schema.py`` change, not smuggled in here.
Likewise the run receipt records no ``execution.readout`` block yet.

**Refusals, by name.** A family with no declaration, an accumulation dtype
outside ``{fp32, fp64}``, a declared epsilon attribute the module lacks, and a
gain convention the module's forward contradicts are each a ``ValueError``
naming the thing and the fix — the ``Identity.tolerance_for`` precedent, never
a bare ``KeyError`` / ``AttributeError``. Every refusal has a passing twin in
``tests/analysis/test_readout.py``.

**The logit lens** ([`logit_lens`][]) projects a saved ``block_output``
harvest through the readout as run — ``logits(h)``, the modules called as the
model calls them — in bounded position chunks, without a transformer forward,
and keeps the highest-k, lowest-k and optional exact-target scores; a file input has its
``ArtifactIdentity`` checked against the bundle before anything is projected.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Mapping, Sequence, get_args

from causalab.protocol.rules.errors import ProtocolError
from causalab.protocol.registry import walk

#: ``torch.Tensor`` / ``torch.dtype`` in signatures. ``analysis/`` is torch-free
#: at module level (the layering guard reads every module-level import, a
#: ``TYPE_CHECKING`` block included), so torch is imported where it is called
#: and the annotations are the open alias.
Tensor = Any
DType = Any

__all__ = [
    "ACCUMULATION_DTYPES",
    "CERTIFICATION_ULPS",
    "GAINS",
    "NORMS",
    "READOUT_SPECS",
    "UNIT_ROUNDOFF",
    "AccumulationDtype",
    "Certificate",
    "Gain",
    "GainMismatch",
    "Norm",
    "Readout",
    "ReadoutSpec",
    "logit_lens",
    "readout_spec",
    "register_readout",
    "unit_roundoff",
]

#: The final normalization's kind: a root-mean-square norm (no centering) or a
#: layer norm (centered, and with an additive term the module may carry).
Norm = Literal["rmsnorm", "layernorm"]
NORMS: tuple[Norm, ...] = get_args(Norm)

#: How the norm's ``weight`` enters: Llama-style ``weight · normed`` or the
#: zero-centred ``(1 + weight) · normed`` of Qwen3.5-MoE (``Qwen3_5MoeRMSNorm``
#: in transformers' ``models/qwen3_5_moe/modeling_qwen3_5_moe.py``). 📐 Which
#: one a family uses is measured, not assumed: [`Readout.certify`][] refuses
#: a declaration the module's forward contradicts.
Gain = Literal["weight", "one_plus_weight"]
GAINS: tuple[Gain, ...] = get_args(Gain)

#: The dtype the *reference* unembedding accumulates in. Only the two that
#: never round a bf16/fp16/fp32 forward's values coarser than the forward did:
#: an analysis that attributes rounding residuals to declared terms cannot be
#: run in a dtype that adds its own.
AccumulationDtype = Literal["fp32", "fp64"]
ACCUMULATION_DTYPES: tuple[AccumulationDtype, ...] = get_args(AccumulationDtype)
#: The torch dtype each accumulation spelling names, by attribute name —
#: resolved on ``torch`` at call time (`_torch_dtype`).
_TORCH_DTYPE_NAMES: Mapping[str, str] = MappingProxyType(
    {"fp32": "float32", "fp64": "float64"}
)

#: Unit roundoff (half an ulp at 1.0) per dtype a readout may run in — the
#: relative size of one rounding of a forward's value — keyed by the torch
#: dtype's name (``str(dtype)`` without the ``torch.`` prefix). The three
#: protocol ``native_dtype`` spellings' dtypes plus fp64.
UNIT_ROUNDOFF: Mapping[str, float] = MappingProxyType(
    {
        "float64": 2.0**-53,
        "float32": 2.0**-24,
        "float16": 2.0**-11,
        "bfloat16": 2.0**-8,
    }
)

#: The certification band, in units of ``unit_roundoff(dtype) · max|norm(x)|``.
#: 📐 The right convention measures ≈ 2 such units on every tiny fixture in
#: fp32 (gpt2 4.3e-7, llama 3.1e-7, qwen3.5-moe 2.1e-7 at |ln_final| 2.4–3.5);
#: the wrong one measures ≈ |ln_final| itself (2.2–3.5 on the fixtures) — six
#: orders of magnitude apart in fp32, with 8 units as the band (the table in
#: ``tests/analysis/test_readout.py``'s docstring).
CERTIFICATION_ULPS = 8


def _torch_dtype(accumulation_dtype: str) -> DType:
    """The ``torch.dtype`` an accumulation spelling (``fp32`` / ``fp64``) names."""
    import torch

    return getattr(torch, _TORCH_DTYPE_NAMES[accumulation_dtype])


def unit_roundoff(dtype: DType) -> float:
    """The unit roundoff of ``dtype`` (a ``torch.dtype``, or its name), or a
    refusal naming the dtypes a readout is certified in — a tolerance is
    declared per dtype, never interpolated."""
    try:
        return UNIT_ROUNDOFF[str(dtype).removeprefix("torch.")]
    except KeyError:
        raise ValueError(
            f"no unit roundoff is declared for {dtype} (declared: "
            f"{[f'torch.{d}' for d in UNIT_ROUNDOFF]}) — a readout is certified "
            "in a dtype whose rounding it can name"
        ) from None


@dataclasses.dataclass(frozen=True)
class ReadoutSpec:
    """One family's readout declaration: the final norm's kind, its gain
    convention, the attribute its epsilon lives at on the module, and the
    dtype the reference unembedding accumulates in. Every field is a closed
    vocabulary or a module attribute name, refused by name otherwise."""

    norm: Norm
    gain: Gain
    eps_attr: str
    accumulation_dtype: AccumulationDtype

    def __post_init__(self) -> None:
        if self.norm not in NORMS:
            raise ValueError(f"readout norm {self.norm!r} is not in {list(NORMS)}")
        if self.gain not in GAINS:
            raise ValueError(f"readout gain {self.gain!r} is not in {list(GAINS)}")
        if not self.eps_attr.isidentifier():
            raise ValueError(
                f"readout eps_attr {self.eps_attr!r} is not an attribute name"
            )
        if self.accumulation_dtype not in ACCUMULATION_DTYPES:
            raise ValueError(
                f"readout accumulation dtype {self.accumulation_dtype!r} is not in "
                f"{list(ACCUMULATION_DTYPES)} — a reference unembedding accumulates "
                "in a dtype at least as wide as the forward's, so the rounding it "
                "attributes is the model's and not its own"
            )


_READOUT_SPECS: dict[str, ReadoutSpec] = {}

#: The registered declarations, keyed by ``ModelInfo.family`` (the HF
#: ``model_type`` of the entry's text config) — read-only view;
#: [`register_readout`][] is the one way in, from any module.
READOUT_SPECS: Mapping[str, ReadoutSpec] = MappingProxyType(_READOUT_SPECS)


def register_readout(family: str, spec: ReadoutSpec) -> None:
    """Register (or replace) the readout declaration of ``family`` — the
    ``register_family`` precedent: a third-party family declares its readout
    beside its adapter, from its own module."""
    if not isinstance(family, str) or not family.isidentifier():
        raise ValueError(f"readout family {family!r} is not an identifier")
    if not isinstance(spec, ReadoutSpec):
        raise ValueError(
            f"family {family!r}: the readout declaration is not a ReadoutSpec"
        )
    _READOUT_SPECS[family] = spec


def readout_spec(family: str | None, *, key: str | None = None) -> ReadoutSpec:
    """The declaration of ``family``, or a refusal naming the family, the
    declared ones and how to declare a new one. ``key`` names the model in the
    refusal of an entry that records no family at all."""
    where = f" (model {key!r})" if key else ""
    if family is None:
        raise ValueError(
            f"the registry entry{where} records no family (ModelInfo.family), so no "
            f"readout declaration can be looked up (declared: {sorted(_READOUT_SPECS)}) "
            "— set the entry's family and declare its readout with "
            "causalab.analysis.logit_lens.register_readout"
        )
    try:
        return _READOUT_SPECS[family]
    except KeyError:
        raise ValueError(
            f"family {family!r}{where} declares no readout (declared: "
            f"{sorted(_READOUT_SPECS)}) — declare its final norm, gain convention, "
            "epsilon attribute and accumulation dtype with "
            "causalab.analysis.logit_lens.register_readout"
        ) from None


# 📐 Measured on the tiny fixtures (2026-09-03, transformers 5.16 lock), norm
# module forward against both conventions at the fixed scale — the numbers are
# in tests/analysis/test_readout.py's docstring. The accumulation dtype is
# the reference projection's: the residual accounting accumulates response
# moments in float64 and is exact only there.
register_readout(
    "gpt2",
    ReadoutSpec(
        norm="layernorm", gain="weight", eps_attr="eps", accumulation_dtype="fp64"
    ),
)
register_readout(
    "llama",
    ReadoutSpec(
        norm="rmsnorm",
        gain="weight",
        eps_attr="variance_epsilon",
        accumulation_dtype="fp64",
    ),
)
register_readout(
    "qwen3_5_moe_text",
    ReadoutSpec(
        norm="rmsnorm",
        gain="one_plus_weight",
        eps_attr="eps",
        accumulation_dtype="fp64",
    ),
)


@dataclasses.dataclass(frozen=True)
class Certificate:
    """What [`Readout.certify`][] measured: the declared gain's gap between
    the module forward and the linearisation at the fixed scale, the band it
    was held to, and every convention's gap beside it — so a test pins the
    separation rather than trusting the verdict."""

    family: str
    dtype: str
    gain: Gain
    gap: float
    tolerance: float
    #: ``max|norm(x)|`` as run — the magnitude the band is relative to
    scale: float
    gaps: Mapping[Gain, float]


class GainMismatch(ValueError):
    """The declared gain convention is not the one the module computes."""

    def __init__(self, message: str, certificate: Certificate) -> None:
        super().__init__(message)
        self.certificate = certificate


@dataclasses.dataclass(frozen=True)
class Readout:
    """One loaded model's readout: the final-norm and head modules the family's
    tree addresses, and the family's declaration.

    [`normalize`][] and [`logits`][] are the readout **as run** — the
    modules called as the model calls them, at the model's dtype, so
    ``logits(block_output@last)`` is the engine's ``lm_head`` read bit for bit.
    [`unembed`][] is the **reference** projection: the head's own forward in
    the declared accumulation dtype. [`fixed_rms_scale`][],
    [`linearized_norm`][] and [`norm_offset`][] are the linearisation a
    residual decomposition uses (``norm(x) ≈ g · s · x + b`` at the fixed
    ``s`` of the whole residual), and [`certify`][] holds the declaration to
    the module.
    """

    family: str
    spec: ReadoutSpec
    norm: Any
    head: Any

    @classmethod
    def from_bundle(cls, bundle: Any) -> "Readout":
        """Build from a loaded bundle (either engine's): the family adapter's
        ``tree.final_norm`` / ``tree.lm_head`` walked on the model, and the
        declaration of ``bundle.info.family``. Refuses by name a family with
        no declaration, a tree address the model lacks, and a declared epsilon
        attribute or weight the norm module does not have."""
        info = bundle.info
        adapter = bundle.adapter
        spec = readout_spec(info.family, key=info.key)
        norm = walk(bundle.model, adapter.tree.final_norm)
        if norm is None:
            raise ValueError(
                f"family {adapter.family!r} addresses its final norm at "
                f"{adapter.tree.final_norm!r}, but this model "
                f"({type(bundle.model).__name__}) has no such child"
            )
        head = walk(bundle.model, adapter.tree.lm_head)
        if head is None:
            raise ValueError(
                f"family {adapter.family!r} addresses its head at "
                f"{adapter.tree.lm_head!r}, but this model "
                f"({type(bundle.model).__name__}) has no such child"
            )
        if not hasattr(norm, spec.eps_attr):
            have = sorted(
                a for a in ("eps", "variance_epsilon", "epsilon") if hasattr(norm, a)
            )
            raise ValueError(
                f"family {info.family!r} declares its final norm's epsilon at "
                f"attribute {spec.eps_attr!r}, but the module "
                f"({type(norm).__name__}) has no such attribute (it has {have}) — "
                "the declaration and the loaded module disagree; fix the "
                "declaration (register_readout)"
            )
        if getattr(norm, "weight", None) is None:
            raise ValueError(
                f"family {info.family!r} declares a {spec.gain!r} gain on its final "
                f"norm, but the module ({type(norm).__name__}) has no weight"
            )
        return cls(family=info.family, spec=spec, norm=norm, head=head)

    def at(self, accumulation_dtype: str) -> "Readout":
        """The same readout with another accumulation dtype for the reference
        unembedding — validated as a declaration is, so ``bf16`` is refused by
        name here too."""
        return dataclasses.replace(
            self,
            spec=dataclasses.replace(
                self.spec,
                accumulation_dtype=accumulation_dtype,  # pyright: ignore[reportArgumentType]
            ),
        )

    @property
    def eps(self) -> float:
        """The final norm's epsilon, read at the declared attribute."""
        return float(getattr(self.norm, self.spec.eps_attr))

    @property
    def accumulation_dtype(self) -> DType:
        """The declared accumulation dtype, as a ``torch.dtype``."""
        return _torch_dtype(self.spec.accumulation_dtype)

    @property
    def head_width(self) -> int | None:
        """The width of the head's output axis — the decoder vocabulary a
        target ID indexes, which is **not** the tokenizer's: a padded
        vocabulary has more columns than tokens, added tokens without a
        resized head fewer. ``nn.Linear``'s ``out_features``, GPT-2's
        ``Conv1D.nf``; ``None`` for a head that declares neither."""
        for attr in ("out_features", "nf"):
            width = getattr(self.head, attr, None)
            if isinstance(width, int):
                return width
        return None

    # --- as run ------------------------------------------------------------ #

    def normalize(self, h: Tensor) -> Tensor:
        """``final_norm(h)`` — the module call, at ``h``'s dtype."""
        return self.norm(h)

    def logits(self, h: Tensor) -> Tensor:
        """``lm_head(final_norm(h))`` — the readout as the model runs it. On
        the last block's output this is the ``lm_head`` read, bit for bit."""
        return self.head(self.norm(h))

    # --- the reference ----------------------------------------------------- #

    def unembed(self, z: Tensor) -> Tensor:
        """The head's own forward on ``z`` in the declared accumulation dtype,
        cast back to ``z``'s dtype. The module's parameters are cast (and
        moved to ``z``'s device) for the call, never read or sliced: whatever
        layout the head has, its forward knows it. When ``z`` and the head are
        already in the accumulation dtype this is the plain module call."""
        from torch.func import functional_call

        acc = self.accumulation_dtype
        params = dict(self.head.named_parameters())
        if z.dtype == acc and all(p.dtype == acc for p in params.values()):
            return self.head(z)
        tensors: dict[str, Tensor] = {
            name: p.detach().to(device=z.device, dtype=acc)
            for name, p in params.items()
        }
        for name, buffer in self.head.named_buffers():
            tensors[name] = (
                buffer.to(device=z.device, dtype=acc)
                if buffer.is_floating_point()
                else buffer.to(device=z.device)
            )
        out = functional_call(self.head, tensors, (z.to(acc),))
        return out.to(z.dtype)

    @staticmethod
    def center(logits: Tensor) -> Tensor:
        """The centered readout: ``logits − mean over the vocabulary``, per
        position. A uniform shift, so every softmax-based metric is invariant
        to it (spec §2.9); it is visible only to raw logit values, which is
        why it is a Python method here and not yet a document field."""
        return logits - logits.mean(dim=-1, keepdim=True)

    # --- the linearisation ------------------------------------------------- #

    def fixed_rms_scale(self, x: Tensor) -> Tensor:
        """The norm's own scale ``s = rsqrt(mean(x²) + eps)`` over the last
        axis of ``x`` — the **fixed** scale of the whole residual, at which the
        decomposition linearises the norm. For a layer norm the mean is of the
        centered ``x`` (its variance), as the module computes it."""
        import torch

        if self.spec.norm == "layernorm":
            x = x - x.mean(dim=-1, keepdim=True)
        return torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def linearized_norm(self, v: Tensor, s: Tensor) -> Tensor:
        """``N(v) = g · s · v`` with ``g`` from the declared gain convention
        (``weight`` → ``w``; ``one_plus_weight`` → ``1 + w``), ``v`` centered
        first under a layer norm. Linear in ``v`` at fixed ``s``, which is what
        lets a sum of components be normalised term by term."""
        return self._linearized(v, s, self.spec.gain)

    def norm_offset(self, like: Tensor) -> Tensor:
        """The norm's additive term — a layer norm's bias, zero for a norm
        without one — in ``like``'s dtype and device. Added **once** to a
        decomposition, never per component."""
        import torch

        bias = getattr(self.norm, "bias", None)
        if bias is None:
            return torch.zeros((), dtype=like.dtype, device=like.device)
        return bias.detach().to(device=like.device, dtype=like.dtype)

    def _linearized(self, v: Tensor, s: Tensor, gain: str) -> Tensor:
        w = self.norm.weight.detach().to(device=v.device, dtype=v.dtype)
        g = w if gain == "weight" else 1.0 + w
        if self.spec.norm == "layernorm":
            v = v - v.mean(dim=-1, keepdim=True)
        return g * (s * v)

    def certify(self, x: Tensor) -> Certificate:
        """Hold the declaration to the module: one forward of the norm module
        on ``x`` against ``linearized_norm(x, fixed_rms_scale(x)) +
        norm_offset`` in float64, for **every** gain convention. The declared
        one must land within [`CERTIFICATION_ULPS`][] units of
        ``x``'s dtype's roundoff at ``max|norm(x)|`` — measured, so a wrong
        declaration is refused naming both conventions and the gap
        ([`GainMismatch`][]), and the returned [`Certificate`][] carries
        every gap for a test to pin the separation."""
        import torch

        with torch.no_grad():
            as_run = self.normalize(x).detach()
            reference_dtype = torch.float64
            x_ref = x.detach().to(reference_dtype)
            s = self.fixed_rms_scale(x_ref)
            offset = self.norm_offset(x_ref)
            as_run_ref = as_run.to(reference_dtype)
            gaps: dict[Gain, float] = {
                gain: float(
                    (as_run_ref - (self._linearized(x_ref, s, gain) + offset))
                    .abs()
                    .max()
                )
                for gain in GAINS
            }
            scale = float(as_run.abs().max())
        tolerance = CERTIFICATION_ULPS * unit_roundoff(as_run.dtype) * scale
        gap = gaps[self.spec.gain]
        certificate = Certificate(
            family=self.family,
            dtype=str(as_run.dtype).removeprefix("torch."),
            gain=self.spec.gain,
            gap=gap,
            tolerance=tolerance,
            scale=scale,
            gaps=MappingProxyType(gaps),
        )
        if not gap <= tolerance:
            others = ", ".join(
                f"{name!r} measures {value:.3e}"
                for name, value in gaps.items()
                if name != self.spec.gain
            )
            raise GainMismatch(
                f"family {self.family!r} declares its final norm's gain as "
                f"{self.spec.gain!r}, but the module ({type(self.norm).__name__}) "
                f"disagrees: the gap between its forward and g·s·x is {gap:.3e} in "
                f"{certificate.dtype}, over the band {tolerance:.3e} "
                f"({CERTIFICATION_ULPS} units of roundoff at max|norm(x)| = "
                f"{scale:.3e}); {others} — fix the declaration (register_readout)",
                certificate,
            )
        return certificate


# --------------------------------------------------------------------------- #
# The logit lens
# --------------------------------------------------------------------------- #


def logit_lens(
    bundle: Any,
    activations: Any,
    *,
    k: int = 10,
    batch_positions: int = 128,
    target_ids: Sequence[int] | None = None,
    slot: str | None = None,
    entry: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """The logit lens over a saved residual harvest, without a transformer
    forward: one record per flattened activation, preserving its leading-axis
    index, holding the top-``k`` tokens and the optional exact target score.

    The projection is the readout **as run** — [`Readout.logits`][], the
    loaded bundle's actual final normalization and vocabulary head called as
    the model calls them — in chunks of ``batch_positions`` rows. A tensor is
    caller-owned data. A file must identify the matching model, revision,
    precision, quantization and ``block_output`` site
    (``read_tensor_with_identity`` + ``check_artifact_identity``): another
    model's harvest fails before projecting. This API accepts a PyTorch
    ``ModelBundle``; a tracing envoy is not an executable decoder module
    outside its trace.
    """
    import itertools

    import torch

    from causalab.io.step_io import read_tensor_with_identity
    from causalab.io.env import check_artifact_identity

    if (
        type(k) is not int
        or k < 1
        or type(batch_positions) is not int
        or batch_positions < 1
    ):
        raise ValueError("k and batch_positions must be positive integers")
    if isinstance(activations, (str, Path)):
        activations, identity = read_tensor_with_identity(
            Path(activations), slot=slot, entry=entry
        )
        check_artifact_identity(
            identity,
            {
                "model_key": bundle.key,
                "model_revision": bundle.revision,
                "model_dtype": bundle.dtype,
            },
            what="logit lens harvest",
        )
        quantization = getattr(bundle, "quantization", None)
        expected_quantization = (
            json.dumps(quantization, sort_keys=True)
            if quantization is not None
            else None
        )
        if identity.get("model_quantization") != expected_quantization:
            raise ProtocolError(
                "P2", "logit lens harvest quantization does not match model"
            )
        site = json.loads(identity.get("site", "{}"))
        if site.get("component") != "block_output":
            raise ProtocolError("P2", "logit lens needs a block_output harvest")
    if not isinstance(activations, torch.Tensor) or activations.ndim < 2:
        raise ValueError("activations must have shape (..., hidden_size)")
    readout = Readout.from_bundle(bundle)
    if not isinstance(readout.norm, torch.nn.Module) or not isinstance(
        readout.head, torch.nn.Module
    ):
        raise ValueError("logit lens requires executable PyTorch decoder modules")
    parameter = next(readout.head.parameters())
    rows = activations.reshape(-1, activations.shape[-1])
    if target_ids is not None:
        if len(target_ids) != len(rows):
            raise ValueError("target_ids must supply one vocabulary ID per activation")
        # a target indexes the decoder's logit axis, whose width is the head's
        # — not the tokenizer's (Readout.head_width)
        width = readout.head_width or bundle.info.vocab_size
        if any(type(t) is not int or not 0 <= t < width for t in target_ids):
            raise ValueError(
                f"target_ids must be integers in [0, {width}), the decoder vocabulary"
            )
    coordinates = list(itertools.product(*(range(n) for n in activations.shape[:-1])))
    records: list[dict[str, Any]] = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_positions):
            values = rows[start : start + batch_positions].to(
                parameter.device, parameter.dtype
            )
            logits = readout.logits(values).float()
            if k > logits.shape[-1]:
                raise ValueError("k exceeds the decoder vocabulary")
            normalizers = logits.logsumexp(dim=-1)
            highest, highest_indices = logits.topk(k, dim=-1)
            lowest, lowest_indices = logits.topk(k, dim=-1, largest=False)
            for local in range(len(values)):
                index = start + local
                highest_ids = highest_indices[local].tolist()
                lowest_ids = lowest_indices[local].tolist()
                record = {
                    "index": list(coordinates[index]),
                    "highest_indices": highest_ids,
                    "highest_tokens": [
                        bundle.tokenizer.decode([t]) for t in highest_ids
                    ],
                    "highest_logits": highest[local].tolist(),
                    "highest_probabilities": (highest[local] - normalizers[local])
                    .exp()
                    .tolist(),
                    "lowest_indices": lowest_ids,
                    "lowest_tokens": [bundle.tokenizer.decode([t]) for t in lowest_ids],
                    "lowest_logits": lowest[local].tolist(),
                    "lowest_probabilities": (lowest[local] - normalizers[local])
                    .exp()
                    .tolist(),
                    "log_normalizer": float(normalizers[local]),
                }
                if target_ids is not None:
                    token = target_ids[index]
                    logit = float(logits[local, token])
                    log_probability = logit - float(normalizers[local])
                    record.update(
                        target_id=token,
                        target_logit=logit,
                        target_log_probability=log_probability,
                    )
                records.append(record)
    return records
