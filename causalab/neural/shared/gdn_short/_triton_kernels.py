"""Triton programs for the single-chunk gated delta rule.

One program handles each sequence and value head. Inputs, normalization,
and intermediate values use float32. Dots use the caller's ``PREC``;
stores round to the output dtype. The triangular inverse uses the finite
Neumann product ``(I+L)^-1 = (I+M)(I+M^2)(I+M^4)...`` with ``M=-L``.
Nilpotence makes it exact after ``log2(BT)`` squarings in exact arithmetic.
``triton_kernel`` imports this module lazily because it requires Triton.
"""

# The kernels are Triton programs: their bodies are traced, not executed, by
# Python, and their `tl` types are opaque to the type checker.
# pyright: basic

from __future__ import annotations

try:
    import triton  # type: ignore[import-not-found]
    import triton.language as tl  # type: ignore[import-not-found]
except ImportError as error:  # the CPU tiers, macOS: the binding never gets here
    raise ImportError(
        "the single-chunk gated-delta kernel needs triton, which the "
        "flash-linear-attention extra installs on Linux"
    ) from error


@triton.jit
def _unit_lower_inverse(b_M, o_t, BT: tl.constexpr, PREC: tl.constexpr):
    """``(I - M)^-1`` for a strictly lower-triangular ``[BT, BT]`` ``M``
    (nilpotent: ``M^BT = 0``), as ``Π_{j < log2 BT} (I + M^(2^j))``."""
    eye = tl.where(o_t[:, None] == o_t[None, :], 1.0, 0.0)
    b_A = eye + b_M
    b_P = tl.dot(b_M, b_M, input_precision=PREC)  # M^2
    b_A = tl.dot(b_A, eye + b_P, input_precision=PREC)
    b_P = tl.dot(b_P, b_P, input_precision=PREC)  # M^4
    b_A = tl.dot(b_A, eye + b_P, input_precision=PREC)
    b_P = tl.dot(b_P, b_P, input_precision=PREC)  # M^8
    b_A = tl.dot(b_A, eye + b_P, input_precision=PREC)
    if BT > 16:
        b_P = tl.dot(b_P, b_P, input_precision=PREC)  # M^16
        b_A = tl.dot(b_A, eye + b_P, input_precision=PREC)
    return b_A


@triton.jit
def single_chunk_fwd_kernel(
    q,
    k,
    v,
    g,
    beta,
    o,
    u,
    T,
    scale,
    eps,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    L2NORM: tl.constexpr,
    STORE_U: tl.constexpr,
    PREC: tl.constexpr,
):
    i_bh = tl.program_id(0)
    i_b = i_bh // HV
    i_h = i_bh % HV
    i_hq = i_h // (HV // H)

    o_t = tl.arange(0, BT)
    o_k = tl.arange(0, K)
    o_v = tl.arange(0, V)
    m_t = o_t < T

    p_q = q + ((i_b * T + o_t[:, None]) * H + i_hq) * K + o_k[None, :]
    p_k = k + ((i_b * T + o_t[:, None]) * H + i_hq) * K + o_k[None, :]
    p_v = v + ((i_b * T + o_t[:, None]) * HV + i_h) * V + o_v[None, :]
    p_g = g + (i_b * T + o_t) * HV + i_h
    p_b = beta + (i_b * T + o_t) * HV + i_h

    b_q = tl.load(p_q, mask=m_t[:, None], other=0.0).to(tl.float32)
    b_k = tl.load(p_k, mask=m_t[:, None], other=0.0).to(tl.float32)
    b_v = tl.load(p_v, mask=m_t[:, None], other=0.0).to(tl.float32)
    b_g = tl.load(p_g, mask=m_t, other=0.0).to(tl.float32)
    b_beta = tl.load(p_b, mask=m_t, other=0.0).to(tl.float32)

    if L2NORM:
        # padded rows are all-zero: 0 * rsqrt(eps) = 0, no NaN
        b_q = b_q * tl.rsqrt(tl.sum(b_q * b_q, 1) + eps)[:, None]
        b_k = b_k * tl.rsqrt(tl.sum(b_k * b_k, 1) + eps)[:, None]

    # decay: D_ij = exp(gc_i - gc_j) on the causal triangle of the real rows
    b_gc = tl.cumsum(b_g, 0)
    m_c = (o_t[:, None] >= o_t[None, :]) & m_t[:, None] & m_t[None, :]
    m_s = (o_t[:, None] > o_t[None, :]) & m_t[:, None] & m_t[None, :]
    b_D = tl.where(m_c, tl.exp(b_gc[:, None] - b_gc[None, :]), 0.0)

    # A = (I + L)^-1,  L = strict_lower((beta k) k^T ⊙ D)
    b_kb = b_k * b_beta[:, None]
    b_kk = tl.dot(b_kb, tl.trans(b_k), input_precision=PREC)
    b_M = tl.where(m_s, -(b_kk * b_D), 0.0)
    b_A = _unit_lower_inverse(b_M, o_t, BT, PREC)

    # u = A (beta v);  o = scale * (q k^T ⊙ D) u
    b_vb = b_v * b_beta[:, None]
    b_u = tl.dot(b_A, b_vb, input_precision=PREC)
    b_qk = tl.dot(b_q, tl.trans(b_k), input_precision=PREC)
    b_P = tl.where(m_c, b_qk * b_D, 0.0) * scale
    b_o = tl.dot(b_P, b_u, input_precision=PREC)

    p_o = o + ((i_b * T + o_t[:, None]) * HV + i_h) * V + o_v[None, :]
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=m_t[:, None])
    if STORE_U:
        p_u = u + ((i_b * T + o_t[:, None]) * HV + i_h) * V + o_v[None, :]
        tl.store(p_u, b_u, mask=m_t[:, None])


@triton.jit
def single_chunk_bwd_kernel(
    q,
    k,
    v,
    g,
    beta,
    do,
    du,
    dq,
    dk,
    dv,
    dg,
    dbeta,
    T,
    scale,
    eps,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    L2NORM: tl.constexpr,
    HAS_DU: tl.constexpr,
    PREC: tl.constexpr,
):
    """Recomputes the forward's tiles, then walks the closed form backwards.
    ``dq``/``dk`` are written per value head (``[B, T, HV, K]``); the caller
    sums the group when ``H < HV``."""
    i_bh = tl.program_id(0)
    i_b = i_bh // HV
    i_h = i_bh % HV
    i_hq = i_h // (HV // H)

    o_t = tl.arange(0, BT)
    o_k = tl.arange(0, K)
    o_v = tl.arange(0, V)
    m_t = o_t < T

    p_q = q + ((i_b * T + o_t[:, None]) * H + i_hq) * K + o_k[None, :]
    p_k = k + ((i_b * T + o_t[:, None]) * H + i_hq) * K + o_k[None, :]
    p_v = v + ((i_b * T + o_t[:, None]) * HV + i_h) * V + o_v[None, :]
    p_do = do + ((i_b * T + o_t[:, None]) * HV + i_h) * V + o_v[None, :]
    p_g = g + (i_b * T + o_t) * HV + i_h
    p_b = beta + (i_b * T + o_t) * HV + i_h

    b_q = tl.load(p_q, mask=m_t[:, None], other=0.0).to(tl.float32)
    b_k = tl.load(p_k, mask=m_t[:, None], other=0.0).to(tl.float32)
    b_v = tl.load(p_v, mask=m_t[:, None], other=0.0).to(tl.float32)
    b_do = tl.load(p_do, mask=m_t[:, None], other=0.0).to(tl.float32)
    b_g = tl.load(p_g, mask=m_t, other=0.0).to(tl.float32)
    b_beta = tl.load(p_b, mask=m_t, other=0.0).to(tl.float32)

    if L2NORM:
        b_rq = tl.rsqrt(tl.sum(b_q * b_q, 1) + eps)
        b_rk = tl.rsqrt(tl.sum(b_k * b_k, 1) + eps)
        b_q = b_q * b_rq[:, None]
        b_k = b_k * b_rk[:, None]

    # ---- forward recomputation ----
    b_gc = tl.cumsum(b_g, 0)
    m_c = (o_t[:, None] >= o_t[None, :]) & m_t[:, None] & m_t[None, :]
    m_s = (o_t[:, None] > o_t[None, :]) & m_t[:, None] & m_t[None, :]
    b_D = tl.where(m_c, tl.exp(b_gc[:, None] - b_gc[None, :]), 0.0)
    b_kb = b_k * b_beta[:, None]
    b_kk = tl.dot(b_kb, tl.trans(b_k), input_precision=PREC)
    b_M = tl.where(m_s, -(b_kk * b_D), 0.0)
    b_A = _unit_lower_inverse(b_M, o_t, BT, PREC)
    b_vb = b_v * b_beta[:, None]
    b_u = tl.dot(b_A, b_vb, input_precision=PREC)
    b_qk = tl.dot(b_q, tl.trans(b_k), input_precision=PREC)
    b_P = tl.where(m_c, b_qk * b_D, 0.0) * scale

    # ---- o = P u ----
    b_du = tl.dot(tl.trans(b_P), b_do, input_precision=PREC)
    if HAS_DU:
        p_du = du + ((i_b * T + o_t[:, None]) * HV + i_h) * V + o_v[None, :]
        b_du += tl.load(p_du, mask=m_t[:, None], other=0.0).to(tl.float32)
    b_dP = tl.where(m_c, tl.dot(b_do, tl.trans(b_u), input_precision=PREC), 0.0)
    # P = scale * qk ⊙ D  (on the causal triangle)
    b_dqk = b_dP * b_D * scale
    b_dD = b_dP * b_qk * scale
    b_dq = tl.dot(b_dqk, b_k, input_precision=PREC)
    b_dk = tl.dot(tl.trans(b_dqk), b_q, input_precision=PREC)

    # ---- u = A vb ----
    b_dA = tl.dot(b_du, tl.trans(b_vb), input_precision=PREC)
    b_dvb = tl.dot(tl.trans(b_A), b_du, input_precision=PREC)
    b_dv = b_dvb * b_beta[:, None]
    b_dbeta = tl.sum(b_dvb * b_v, 1)

    # ---- A = (I + L)^-1  ->  dL = -A^T dA A^T, strictly lower ----
    b_dL = tl.dot(
        tl.dot(tl.trans(b_A), b_dA, input_precision=PREC),
        tl.trans(b_A),
        input_precision=PREC,
    )
    b_dL = tl.where(m_s, -b_dL, 0.0)
    # L = (kb k^T) ⊙ D
    b_dkk = b_dL * b_D
    b_dD += b_dL * b_kk
    b_dkb = tl.dot(b_dkk, b_k, input_precision=PREC)
    b_dk += tl.dot(tl.trans(b_dkk), b_kb, input_precision=PREC)
    # kb = k beta
    b_dk += b_dkb * b_beta[:, None]
    b_dbeta += tl.sum(b_dkb * b_k, 1)

    # ---- D_ij = exp(gc_i - gc_j);  gc = cumsum(g) ----
    b_dDD = b_dD * b_D
    b_dgc = tl.sum(b_dDD, 1) - tl.sum(b_dDD, 0)
    b_dg = tl.sum(tl.where(o_t[None, :] >= o_t[:, None], b_dgc[None, :], 0.0), 1)

    # ---- l2norm: y = x r  ->  dx = r (dy - y (y . dy)) ----
    if L2NORM:
        b_dq = b_rq[:, None] * (b_dq - b_q * tl.sum(b_q * b_dq, 1)[:, None])
        b_dk = b_rk[:, None] * (b_dk - b_k * tl.sum(b_k * b_dk, 1)[:, None])

    p_dq = dq + ((i_b * T + o_t[:, None]) * HV + i_h) * K + o_k[None, :]
    p_dk = dk + ((i_b * T + o_t[:, None]) * HV + i_h) * K + o_k[None, :]
    p_dv = dv + ((i_b * T + o_t[:, None]) * HV + i_h) * V + o_v[None, :]
    p_dg = dg + (i_b * T + o_t) * HV + i_h
    p_db = dbeta + (i_b * T + o_t) * HV + i_h
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), mask=m_t[:, None])
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), mask=m_t[:, None])
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), mask=m_t[:, None])
    tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), mask=m_t)
    tl.store(p_db, b_dbeta.to(p_db.dtype.element_ty), mask=m_t)
