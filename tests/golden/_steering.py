"""Fourier steering for the arithmetic golden (arXiv:2605.01148 Eq. 4 / Alg. 1).

The paper's hours steering correction is per-prompt — it reads each
residual's *current* sine/cosine coefficients and radius before patching —
so it cannot be a constant ``add_scaled`` vector, and an ``affine`` edit
would need an unauthorable 4096x4096 matrix param. It is exactly what the
spec's local-only ``pytorch_fn`` mechanism exists for: the generated
steering documents declare ``apply_target_<n>`` in their ``code`` section
(§2.8.1) and their write is ``{"pytorch_fn": {"code": "steer_fn"}}``,
routing only to local engines (``pytorch_fn_local``).

The declaration is what puts *this file's bytes* in those documents'
digests. It also shows the boundary of the static half of §5 rule 24:
``apply_target_<n>`` is built by `_make` at import time, so there is no
``def`` of that name to read a signature from, and the argument and
undeclared-read checks stand down while the source hash still covers the
function.

The probe weights are process-local state set by the golden test after it
fits the probes (``configure``); the documents are generated per run and
not digest-pinned. Algorithm 1, per period T in the fixed steered set:
read the current coefficients s = w_s.h + b_s and c = w_c.h + b_c, form
the radius r = sqrt(s^2 + c^2) once, then patch the sine and cosine
directions in sequence, re-reading the coefficient each time:

    h <- h + ((alpha*r*sin(theta) - s) / ||w_s||^2) * w_s      (then cos)

with theta = 2*pi*n'/T and alpha = 10.
"""

from __future__ import annotations

import math
from typing import Any

import torch

ALPHA = 10.0
PERIODS = (2, 5, 10, 20, 50)  # fixed for hours; T=100 excluded per the paper

_PROBES: dict[int, dict[str, Any]] = {}


def configure(probes: dict[int, dict[str, Any]]) -> None:
    """Install fitted probes: {period: {w_sin, b_sin, w_cos, b_cos}} with
    weight vectors shaped (d,) and float biases."""
    _PROBES.clear()
    _PROBES.update(probes)


def _steer(f: torch.Tensor, target: int) -> torch.Tensor:
    assert _PROBES, "tests.golden._steering.configure() was not called"
    h = f.clone().float()
    for period in PERIODS:
        p = _PROBES[period]
        w_s = p["w_sin"].to(h.device, h.dtype)
        w_c = p["w_cos"].to(h.device, h.dtype)
        s = h @ w_s + p["b_sin"]
        c = h @ w_c + p["b_cos"]
        r = torch.sqrt(s * s + c * c)
        theta = 2 * math.pi * target / period
        s_star = ALPHA * r * math.sin(theta)
        c_star = ALPHA * r * math.cos(theta)
        h = h + ((s_star - s) / float(w_s.norm() ** 2)).unsqueeze(-1) * w_s
        c = h @ w_c + p["b_cos"]  # re-read after the sine patch (Alg. 1)
        h = h + ((c_star - c) / float(w_c.norm() ** 2)).unsqueeze(-1) * w_c
    return h.to(f.dtype)


def _make(target: int):
    def apply(f: torch.Tensor) -> torch.Tensor:
        return _steer(f, target)

    apply.__name__ = f"apply_target_{target}"
    return apply


for _n in range(24):
    globals()[f"apply_target_{_n}"] = _make(_n)
