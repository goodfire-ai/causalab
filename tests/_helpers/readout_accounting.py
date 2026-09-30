"""The Hydra-effect residual accounting, rebuilt **through the readout adapter**.

The direct-effect accounting behind the Hydra-effect replication (McGrath et
al., arXiv:2307.15771; ``demos/papers/hydra_fig1.md``) decomposes the logits
into per-component contributions through a fixed-RMS linearisation of the
final norm and attributes the closure residual to two declared rounding terms
— the normalization's and the LM head's — with the acceptance that removing
either term must break the closure on a fixture that exhibits that residual.
An earlier version on the real Qwen3.6-35B-A3B needed architecture branches (a
``final_norm_of`` probing three attribute names, a gain convention detected by
fit, ``lm_head.weight`` read by hand). This module is the same accounting with
**none of them**: every family fact comes from [`Readout`][causalab.analysis.logit_lens.Readout], every tensor
from the engine's own reads, and the reference projection from the head's own
forward in the declared accumulation dtype.

The accounting, references in float64 on CPU (the earlier version's, verbatim)::

    x        = block_output @ last layer          (the final residual, as run)
    c_i      = block_input @ 0, then attention_output@L, mlp_output@L for every L
    s        = fixed_rms_scale(x)                 (the FIXED scale, from x)
    N(v)     = linearized_norm(v, s)              (g · s · v, g by declaration)
    b        = norm_offset                        (a layer norm's bias; 0 otherwise)

    additive residual        = max| Σ_i c_i − x |
    fixed-RMS reconstruction = max| Σ_i N(c_i) − N(x) |
    normalization rounding   = max| ln_final_as_run − (N(x) + b) |          (norm_term)
    LM-head rounding         = max| logits_as_run − W(ln_final_as_run) |    (lmhead_term)

with ``W`` the reference unembedding. Closure is exact by linearity of ``W``::

    logits_as_run = W(Σ_i N(c_i) + b) + W(N(x) − Σ_i N(c_i)) + W(norm_term) + lmhead_term

and what each dropped term costs is the size of that term: without the
normalization term the residual is ``max|W(norm_term)|``, without the LM-head
term it is ``max|lmhead_term|``.

The first component is ``block_input`` of layer 0 rather than ``embeddings``:
on GPT-2 the residual entering the tower is token **plus position**
embedding, and ``block_input@0`` is what enters the tower on every family —
no branch. Both tiers (the tiny fixtures on CPU, the A3B under ``-m golden``)
read this one module, so the assertion has one shape.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Mapping

import torch

from causalab.neural.engines.pytorch_hooks.executor import PointExecutor
from causalab.analysis.logit_lens import (
    CERTIFICATION_ULPS,
    Certificate,
    Readout,
    unit_roundoff,
)

from tests._helpers import a3b_sweep as sweep

__all__ = ["ROWS", "Accounting", "account", "problems", "read"]

#: The earlier version's two rows — a batch axis, and two lengths so the reads are ragged.
ROWS: list[dict[str, Any]] = [
    {"input": "The Eiffel Tower stands in the city of Paris"},
    {"input": "The capital city of Japan is Tokyo"},
]


def read(
    bundle: Any, component: str, layer: int | None, *, rows: list[dict[str, Any]] = ROWS
) -> torch.Tensor:
    """One engine read of ``component`` at every position of every row, as the
    flat ``(positions, features)`` tensor — the same document shape the A3B
    sweep and the earlier version drive. The executor hands it back on the host, whatever
    device the model runs on."""
    doc = sweep.read_doc(component, layer, pos="all")
    executor = sweep.make_executor(PointExecutor, doc, bundle, rows=rows, with_cf=False)
    value = executor.read_value("r")
    tensor = value.flat if hasattr(value, "flat") else value
    return tensor.detach()


@dataclasses.dataclass(frozen=True)
class Accounting:
    """Every measured term of one fixture's accounting."""

    family: str
    dtype: str
    n_components: int
    additive_reconstruction: float
    fixed_rms_reconstruction: float
    normalization_rounding: float
    lm_head_rounding: float
    #: ``max|W(norm_term)|`` — what dropping the normalization term costs
    projected_normalization_rounding: float
    closure: float
    projection_noise_floor: float
    closure_without_normalization_term: float
    closure_without_lm_head_term: float
    residual_absmax: float
    ln_final_absmax: float
    logits_absmax: float
    #: ``readout.logits(x) == lm_head read``, ``torch.equal``
    bit_exact_logits: bool
    certificate: Certificate

    def as_dict(self) -> Mapping[str, Any]:
        """Plain dicts and floats (the certificate's read-only gap table
        copied), for a log line or an assertion message."""
        d: dict[str, Any] = {
            f.name: getattr(self, f.name)
            for f in dataclasses.fields(self)
            if f.name != "certificate"
        }
        d["certificate"] = {
            f.name: getattr(self.certificate, f.name)
            for f in dataclasses.fields(self.certificate)
            if f.name != "gaps"
        }
        d["certificate"]["gaps"] = dict(self.certificate.gaps)
        return d


def _maxabs(t: torch.Tensor) -> float:
    return float(t.abs().max())


def account(bundle: Any, *, rows: list[dict[str, Any]] = ROWS) -> Accounting:
    """Run the accounting on ``bundle`` through [`Readout`][causalab.analysis.logit_lens.Readout]."""
    readout = Readout.from_bundle(bundle)
    n_layers = len(bundle.blocks)

    components = [read(bundle, "block_input", 0, rows=rows)]
    for layer in range(n_layers):
        components.append(read(bundle, "attention_output", layer, rows=rows))
        components.append(read(bundle, "mlp_output", layer, rows=rows))
    x = read(bundle, "block_output", n_layers - 1, rows=rows)
    ln_actual = read(bundle, "ln_final", None, rows=rows)
    logits_actual = read(bundle, "lm_head", None, rows=rows)

    # The as-run calls are module forwards, so they need ``x`` where the
    # module's parameters are; the reads arrive on the host. The tiny tier runs
    # on the CPU and never told the two apart — the A3B golden tier did.
    run_device = next(readout.norm.parameters()).device
    x_run = x.to(run_device)
    with torch.no_grad():
        bit_exact = torch.equal(readout.logits(x_run), logits_actual.to(run_device))
        certificate = readout.certify(x_run)

        f64 = torch.float64
        x64 = x.to("cpu", f64)
        ln64 = ln_actual.to("cpu", f64)
        logits64 = logits_actual.to("cpu", f64)
        reference = readout.at("fp64")

        s = reference.fixed_rms_scale(x64)
        offset = reference.norm_offset(x64)
        n_x = reference.linearized_norm(x64, s)

        total = torch.zeros_like(x64)
        normalized_total = torch.zeros_like(x64)
        for c in components:
            c64 = c.to("cpu", f64)
            total = total + c64
            normalized_total = normalized_total + reference.linearized_norm(c64, s)

        norm_term = ln64 - (n_x + offset)
        lmhead_term = logits64 - reference.unembed(ln64)

        w_components = reference.unembed(normalized_total + offset)
        w_reconstruction = reference.unembed(n_x - normalized_total)
        w_norm_term = reference.unembed(norm_term)
        closure = logits64 - (
            w_components + w_reconstruction + w_norm_term + lmhead_term
        )
        floor = _maxabs(
            reference.unembed(n_x + offset) - (w_components + w_reconstruction)
        )
        without_norm = logits64 - (w_components + w_reconstruction + lmhead_term)
        without_head = logits64 - (w_components + w_reconstruction + w_norm_term)

    return Accounting(
        family=readout.family,
        dtype=str(x.dtype).removeprefix("torch."),
        n_components=len(components),
        additive_reconstruction=_maxabs(total - x64),
        fixed_rms_reconstruction=_maxabs(normalized_total - n_x),
        normalization_rounding=_maxabs(norm_term),
        lm_head_rounding=_maxabs(lmhead_term),
        projected_normalization_rounding=_maxabs(w_norm_term),
        closure=_maxabs(closure),
        projection_noise_floor=floor,
        closure_without_normalization_term=_maxabs(without_norm),
        closure_without_lm_head_term=_maxabs(without_head),
        residual_absmax=_maxabs(x64),
        ln_final_absmax=_maxabs(ln64),
        logits_absmax=_maxabs(logits64),
        bit_exact_logits=bit_exact,
        certificate=certificate,
    )


def problems(acc: Accounting, *, ulps: int = CERTIFICATION_ULPS) -> list[str]:
    """What the accounting must satisfy, in one shape for both tiers.

    * the adapter's readout is the engine's ``lm_head`` read bit for bit;
    * the closure residual is at the reference projection's own noise floor
      (the earlier version's ``10 · floor``, floor ≥ 1e-12);
    * the fixture **exhibits** both rounding terms (each ≫ the floor) — so
      dropping either breaks the closure, by exactly that term;
    * each rounding term is what its dtype's roundoff allows at the value's
      magnitude (``ulps · roundoff · max|value|``) — the tolerance policy,
      declared per dtype rather than absorbed into one band;
    * the components sum to the residual, and their normalised sum to the
      normalised residual, within that same policy.
    """
    out: list[str] = []
    floor = max(acc.projection_noise_floor, 1e-12)
    roundoff = unit_roundoff(getattr(torch, acc.dtype))
    if not acc.bit_exact_logits:
        out.append(
            "readout.logits(block_output@last) is not the lm_head read bit for bit"
        )
    if not acc.closure <= 10 * floor:
        out.append(f"closure {acc.closure:.3e} > 10 · floor {floor:.3e}")
    for name, value in (
        ("normalization rounding", acc.normalization_rounding),
        ("LM-head rounding", acc.lm_head_rounding),
    ):
        if not value > 100 * floor:
            out.append(
                f"the fixture does not exhibit {name}: {value:.3e} ≤ 100 · floor"
            )
    for name, dropped, term in (
        (
            "normalization",
            acc.closure_without_normalization_term,
            acc.projected_normalization_rounding,
        ),
        ("LM-head", acc.closure_without_lm_head_term, acc.lm_head_rounding),
    ):
        if not dropped >= 100 * floor:
            out.append(f"dropping the {name} term leaves only {dropped:.3e}")
        if not abs(dropped - term) <= 10 * floor:
            out.append(
                f"dropping the {name} term leaves {dropped:.3e}, not the term's own "
                f"{term:.3e}"
            )
    norm_band = ulps * roundoff * acc.ln_final_absmax
    if not acc.normalization_rounding <= norm_band:
        out.append(
            f"normalization rounding {acc.normalization_rounding:.3e} exceeds "
            f"{ulps} units of {acc.dtype} roundoff at |ln_final| ({norm_band:.3e})"
        )
    head_band = ulps * roundoff * acc.logits_absmax
    if not acc.lm_head_rounding <= head_band:
        out.append(
            f"LM-head rounding {acc.lm_head_rounding:.3e} exceeds {ulps} units of "
            f"{acc.dtype} roundoff at |logits| ({head_band:.3e})"
        )
    sum_band = ulps * roundoff * acc.n_components
    if not acc.additive_reconstruction <= sum_band * acc.residual_absmax:
        out.append(
            f"the {acc.n_components} components do not sum to the residual: "
            f"{acc.additive_reconstruction:.3e}"
        )
    if not acc.fixed_rms_reconstruction <= sum_band * acc.ln_final_absmax:
        out.append(
            f"fixed-RMS reconstruction {acc.fixed_rms_reconstruction:.3e} exceeds "
            f"{sum_band * acc.ln_final_absmax:.3e}"
        )
    return out
