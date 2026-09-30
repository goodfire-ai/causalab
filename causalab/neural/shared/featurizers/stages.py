"""Featurizer stage implementations.

The Stage interface supports identity, Cayley and other subspace maps,
loaded linear maps, standardization, and sparse autoencoders. Frame checks
use ``ORTHONORMAL_TOLERANCE`` and ``orthonormality_deviation``.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn.utils.parametrize

from causalab.neural.shared.featurizers.sharing import _SCOPE, _Shared, _once  # pyright: ignore[reportPrivateUsage]

#: How far ``PᵀP`` may sit from the identity before a saved basis is refused
#: as a ``subspace`` start (`_init_basis`). Loose enough for the fp32
#: bases the pipeline writes — ``fit_pca`` (~1e-6) and a fitted ``subspace``
#: (~1e-6 in the regime fits live in; [`Cayley`][], *Conditioning*, for
#: the degenerate one that can exceed this) — tight enough that a basis which
#: is merely *close* to a frame — a bf16-rounded one (≈ 3e-3), a hand-scaled
#: one — is caught. The writer side records every fitted rotation's deviation
#: and its verdict against this bar as ``orthonormality_deviation`` /
#: ``within_tolerance`` in ``fit_diagnostics.json``
#: ([`fit_diagnostics`][causalab.neural.engines.pytorch_hooks.train.fit_diagnostics]),
#: so a start refused here can be traced to the fit that produced it.
ORTHONORMAL_TOLERANCE = 1e-4


def orthonormality_deviation(q: torch.Tensor) -> float:
    """``max|QᵀQ − I|`` in fp32 — the one quantity [`ORTHONORMAL_TOLERANCE`][]
    is the bar for, so it lives beside it rather than being respelled at each
    of its call sites: `_init_basis` at load, [`Cayley.right_inverse`][]
    on assignment, and the train loop's ``fit_diagnostics`` at save."""
    with torch.no_grad():
        columns = q.detach().to(torch.float32)
        gram = columns.mT @ columns
        eye = torch.eye(gram.shape[-1], dtype=gram.dtype, device=gram.device)
        return float((gram - eye).abs().max())


class Stage(torch.nn.Module):
    """One featurizer stage. Subclasses implement ``featurize`` /
    ``inverse``; parameters registered here are what ``train.params``
    optimizes."""

    kind: str = "identity"

    #: Whether ``featurize`` takes the routing table beside the activation —
    #: true of the expert-keyed gate alone ([`Gate`][], ``expert_neuron``),
    #: whose parameters a token's slots find through ``expert_idx``.
    needs_routing: bool = False

    def featurize(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        return x, None

    def inverse(self, f: torch.Tensor, err: torch.Tensor | None) -> torch.Tensor:
        return f

    def slot_params(self) -> dict[str, torch.Tensor]:
        """The auto-declared slots (§2.5), for saving and identity checks."""
        return {}

    def project(self) -> None:
        """Put the parameters back on their feasible set after an optimizer
        step — a no-op for every stage whose parameters are unconstrained or
        constrained by construction (a ``subspace``'s orthogonal
        parametrization). The train loop calls it after every step; a
        ``clamp`` gate clips its mask into ``[0, 1]`` here (§2.5)."""

    def identity_fields(self) -> dict[str, Any]:
        """ArtifactIdentity fields (§8) a bundle this stage is saved into
        carries *beyond* what the document implies — empty unless the stage
        was built from something the document only points at."""
        return {}


class Identity(Stage):
    kind = "identity"


class Cayley(torch.nn.Module):
    """The Cayley map onto the Stiefel manifold, computed at the rank of the
    tangent vector rather than at ``d×d``.

    Given the base frame ``Q₀`` (``(d, k)``, orthonormal) and a free
    ``X ∈ ℝ^{d×k}``, the skew-symmetric ``A = X Q₀ᵀ − Q₀ Xᵀ`` and

        Q(X) = cayley(A) Q₀,   cayley(A) = (I − A/2)⁻¹ (I + A/2).

    ``A Q₀`` ranges over the whole tangent space at ``Q₀``, so this reaches
    every frame the d×d map reaches. Torch's ``orthogonal(...,
    orthogonal_map="cayley")`` computes exactly this map — but by embedding
    ``X`` into a d×d skew matrix, solving a d×d system, and multiplying by a
    d×d ``base`` (completed once, at build time, from the *global* RNG), on
    every ``.weight`` access. At ``d = 4096, k = 8`` that is a 4096³ solve
    to move eight vectors, no cheaper than the matrix exponential.

    **The low-rank form.** Split ``X`` against the frame: ``B = Q₀ᵀX`` and
    ``X⊥ = X − Q₀B``. Only the skew half ``Ω = B − Bᵀ`` of ``B`` reaches
    ``A`` (its symmetric half cancels — those ``k(k+1)/2`` directions of
    ``X`` are redundant, and a loss has zero gradient along them), so

        A = Q₀ Ω Q₀ᵀ + X⊥ Q₀ᵀ − Q₀ X⊥ᵀ = P C Pᵀ,
        P = [X⊥, Q₀] (d×2k),   C = [[0, I], [−I, Ω]].

    The Woodbury identity gives ``(I − A/2)⁻¹ A Q₀ = −2 P (PᵀP − 2C⁻¹)⁻¹
    PᵀQ₀``; with ``X⊥ ⊥ Q₀`` the Gram is block-diagonal and ``PᵀQ₀ = [0; I]``,
    so ``Q(X) = Q₀ − 2 P v`` with ``v = [v₁; v₂]`` solving

        [[X⊥ᵀX⊥ − 2Ω, 2I], [−2I, I]] v = [0; I].

    Eliminating the second block row (``v₂ = I + 2v₁``) leaves one ``k×k``
    system on the Schur complement ``S``:

        S v₁ = −2I,   S = X⊥ᵀX⊥ − 2Ω + 4I.

    ``sym(S) = X⊥ᵀX⊥ + 4I`` is positive definite, so ``S`` is never singular
    — that is the whole invertibility argument. Cost ``O(d k²)`` plus a
    ``k×k`` inverse; nothing larger than ``(d, k)`` is built or saved for
    backward, and the map is a function of ``Q₀`` alone — no RNG.

    **Conditioning.** Scaling ``X⊥``'s columns to unit norm, ``X⊥ = X̃ D``
    with ``D = diag(‖x⊥ⱼ‖)`` clamped at 1 and detached (the value does not
    depend on ``D``, so neither should its gradient), gives

        S = X̃ᵀX̃ − 2D⁻¹ΩD⁻¹ + 4D⁻²,   Q(X) = Q₀ − 2 (X̃ v₁ + Q₀ v₂),
        v₂ = I + 2D⁻¹v₁.

    With generic (near-orthogonal) columns ``X̃ᵀX̃ ≈ I`` and ``κ(S) = O(1)``
    for any ``‖X‖`` — measured at ``d = 4096, k = 32`` the fp32
    orthonormality error stays near 2e-6 up to column norms of 900, where the
    dense fp32 solve has drifted to 3e-6. The limit is **rank-deficient**
    ``X⊥``: two (near-)parallel columns of length ``s`` put an eigenvalue of
    ``≈ 4/s²`` in ``S``, so ``κ(S) ~ s²/2`` — quadratic in ``‖X‖`` where the
    dense ``I − A/2`` degrades only linearly. Measured (fp32, low-rank /
    dense): ``s = 10`` → 3e-6 / 2e-6, ``s = 100`` → 1e-4 / 2e-6, ``s = 900``
    → 1e-2 / 1e-5. At ``k = 1`` a column of length ``s`` is a rotation of the
    base vector by ``2·atan(s/2)`` — 178° at ``s = 100``, the chart near
    saturation. At ``k ≥ 2`` the degenerate case is several columns rotating
    toward one direction, which no such one-plane picture bounds, so rather
    than argue it cannot happen the train loop records each fitted rotation's
    deviation in ``fit_diagnostics.json`` (``orthonormality_deviation``, and
    ``within_tolerance`` against [`ORTHONORMAL_TOLERANCE`][]); a rotation
    past that bar is one a later document cannot name as an ``init``. A pure
    in-frame ``X`` (``X⊥ = 0``) is exact:
    ``S = 4I − 2Ω``.

    ``right_inverse`` is what ``stage.weight = Q`` calls: it rebases the map
    at ``Q`` and returns the zero ``X`` — the identity in ``X`` is the base.

    **Launches.** The map is a run of ~25 tiny kernels, issued on every
    ``.weight`` access unless a [`featurizer_cache`][] scope is open, so
    its spelling minds the count where that costs no bit: the ``k×k``
    identity is a buffer rather than a fresh ``eye`` per call, and the
    inverse is ``inv_ex`` with its host error check off — ``torch.linalg.inv``
    *is* ``inv_ex`` followed by a device-to-host copy of the info tensor, a
    synchronization per call, and the Schur system is nonsingular by the
    argument above. The products with the diagonal ``D⁻¹`` stay GEMMs on
    purpose: spelled as the row and column scalings they are, the *forward*
    is bit-identical but the gradient is not — the autograd engine then sums
    ``X̃``'s three contributions in another order — and the count is the
    same either way. ``tests/neural/shared/test_featurizer_cache.py`` holds
    this spelling bit-identical, forward and backward, to the one it
    replaced."""

    base: torch.Tensor
    eye: torch.Tensor

    def __init__(self, base: torch.Tensor) -> None:
        super().__init__()
        # row-major on purpose: `torch.linalg.qr` hands back a column-major Q
        # and `clone` would keep that layout, so the materialized weight would
        # be a transposed-layout matrix while its saved copy is row-major —
        # and a matmul takes a different kernel path for each, landing one ulp
        # apart. A fit and its reloaded artifact must featurize bit-identically.
        self.register_buffer("base", base.detach().contiguous().clone())
        # the k×k identity `v₂ = I + 2D⁻¹v₁` adds: a constant, so a buffer
        # that follows the module's device and dtype — and not saved state,
        # so a bundle's keys and the train loop's snapshot are unchanged
        self.register_buffer(
            "eye",
            torch.eye(base.shape[-1], dtype=base.dtype, device=base.device),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.map(x, x)

    def map(self, x_frame: torch.Tensor, x_perp: torch.Tensor) -> torch.Tensor:
        """The map with ``X`` named once per place it enters: ``x_frame``
        where the frame coordinates ``Q₀ᵀX`` are taken, ``x_perp`` where the
        complement ``X − Q₀(Q₀ᵀX)`` is. [`forward`][] passes the one ``X``
        to both; `_Shared.rotation` passes two aliases of it so the
        parameter's two gradient contributions come back apart."""
        base = self.base
        in_frame = base.mT @ x_frame
        omega = in_frame - in_frame.mT
        perp = x_perp - base @ in_frame
        scale = torch.linalg.vector_norm(perp.detach(), dim=-2).clamp(min=1.0)
        perp = perp / scale
        inv_scale = torch.diag_embed(1.0 / scale)
        schur = (
            perp.mT @ perp
            - 2.0 * inv_scale @ omega @ inv_scale
            + 4.0 * inv_scale @ inv_scale
        )
        # `inv` rather than `solve`: the matrix is k×k, so the cost is the
        # same, and solve's backward (`linalg_lu_solve`) has no MPS kernel in
        # torch 2.9 while inv's is plain matmuls — the map trains on every
        # device the engine accepts. `inv_ex` without the error check: the
        # check is a host synchronization per call, and the system is
        # nonsingular (class docstring).
        inverse = torch.linalg.inv_ex(schur, check_errors=False).inverse
        v1 = inverse @ (-2.0 * inv_scale)
        v2 = self.eye + (2.0 * inv_scale @ v1)
        return base - 2.0 * (perp @ v1 + base @ v2)

    def right_inverse(self, q: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            deviation = orthonormality_deviation(q)
            if deviation > ORTHONORMAL_TOLERANCE:
                raise ValueError(
                    "a subspace weight must be orthonormal (QᵀQ = I; max |QᵀQ − I| "
                    f"= {deviation:.3g}, tolerance {ORTHONORMAL_TOLERANCE:g}); "
                    "refusing to re-orthonormalize silently"
                )
            self.base.copy_(q)
            return torch.zeros_like(q, memory_format=torch.contiguous_format)


class Subspace(Stage):
    """An orthonormal ``(d, k)`` map ``Q``; features are the coordinates in
    its column space, ``err`` the complement (module docstring).

    ``parametrization`` picks how the optimizer's free tensor becomes a
    Stiefel point. ``cayley`` is [`Cayley`][], ``O(d k²)`` per access.
    ``matrix_exp`` and ``stiefel`` (householder products) are torch's
    ``orthogonal`` maps: ``stiefel`` is also ``O(d k²)``, but ``matrix_exp``
    exponentiates a d×d matrix on every access and is impractical at model
    width — it stays in the vocabulary because the spec (§2.5) and every
    saved rotation's ArtifactIdentity name it.

    ``seed`` picks the initial rotation and is kept on the instance so a
    cached stage can be checked against the seed a later use site asks for
    ([`build_stack`][]).

    ``init`` is an optional orthonormal ``(d, k)`` matrix ``P`` the fit starts
    from (§2.5 ``init`` — the first ``k`` columns of a saved basis). Every
    parametrization here is a *trivialization*: the weight is a map of a free
    parameter that starts at the identity, applied to a fixed base frame, so
    before any step the weight *is* the start (``qr(randn(d, k))`` from the
    seeded generator, or ``P`` verbatim) and the read is ``Pᵀx`` bit-for-bit,
    while the trainable surface is unchanged. For ``cayley`` the base is the
    ``(d, k)`` start itself ([`Cayley`][]). For ``matrix_exp`` and
    ``stiefel`` torch's ``orthogonal`` needs a d×d base: absent ``init`` torch
    completes it from the global RNG, exactly as before; with ``init`` the
    base is ``[P | N]`` orthonormalized by QR, ``N`` drawn from the same
    seeded generator, with its first ``k`` columns set to ``P``, so the
    start is a function of the document alone. ``init_identity`` is what a
    bundle this stage is saved into records about that start
    ([`identity_fields`][])."""

    kind = "subspace"

    def __init__(
        self,
        width: int,
        k: int,
        parametrization: str,
        *,
        seed: int = 0,
        init: torch.Tensor | None = None,
        init_identity: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.k = k
        self.seed = seed
        self.init_identity: dict[str, Any] = dict(init_identity or {})
        generator = torch.Generator().manual_seed(seed)
        if init is None:
            start = torch.linalg.qr(torch.randn(width, k, generator=generator))[0]
        else:
            if tuple(init.shape) != (width, k):
                raise ValueError(
                    f"init must be a ({width}, {k}) matrix, got {tuple(init.shape)}"
                )
            start = init.detach().to(torch.float32).clone()
        self.weight = torch.nn.Parameter(start)
        if parametrization == "cayley":
            # the base *is* the start — no d×d completion to seed or replace
            torch.nn.utils.parametrize.register_parametrization(
                self, "weight", Cayley(start)
            )
            return
        orthogonal_map = {
            "matrix_exp": "matrix_exp",
            # a direct Stiefel point via householder products — the map torch
            # provides for rectangular orthogonal parametrizations
            "stiefel": "householder",
        }[parametrization]
        torch.nn.utils.parametrizations.orthogonal(
            self, "weight", orthogonal_map=orthogonal_map
        )
        if init is not None:
            # torch completed the base from the global RNG; replace it with the
            # seeded completion so the start is a function of the document alone
            complement = torch.randn(width, width - k, generator=generator)
            full = torch.linalg.qr(torch.cat([start, complement], dim=1))[0]
            self.parametrizations.weight[0].base = torch.cat(
                [start, full[:, k:]], dim=1
            )

    def identity_fields(self) -> dict[str, Any]:
        return dict(self.init_identity)

    def _q(self) -> torch.Tensor:
        """The rotation — the parametrization evaluated once per open
        [`featurizer_cache`][] scope where that is exact: a trained
        ``cayley`` map shares its forward and replays its backward per access
        (`_Shared.rotation`); every other map shares under ``no_grad``
        only and is recomputed on a grad access. The entry is keyed without
        the stage's mode on purpose: ``Q`` is a function of ``base`` and the
        parameter alone, the same in ``train()`` and ``eval()``."""
        entries = _SCOPE.entries
        if entries is None:
            return self.weight
        parametrization = self.parametrizations.weight  # type: ignore[union-attr]
        original: torch.Tensor = parametrization.original
        # one map composes `weight`; a second would make `[0]` a wrong
        # rotation inside a scope alone, so the assumption is loud
        assert len(parametrization) == 1, "the rotation is one map; see `_Shared`"
        cayley = parametrization[0]  # type: ignore[index]
        if isinstance(cayley, Cayley) and original.requires_grad:
            key = (self, "rotation")
            shared = entries.get(key)
            if shared is None:
                shared = _Shared.rotation(cayley, original)
                entries[key] = shared
            return shared.access()
        return _once(
            self, "weight", lambda: self.weight, trainable=original.requires_grad
        )

    def featurize(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        q = self._q()
        f = x.to(q.dtype) @ q
        return f, x - f @ q.T

    def inverse(self, f: torch.Tensor, err: torch.Tensor | None) -> torch.Tensor:
        q = self._q()
        x = f @ q.T
        return x if err is None else x + err

    def slot_params(self) -> dict[str, torch.Tensor]:
        return {"weight": self._q()}


class LoadedLinear(Stage):
    """A fixed ``(d, k)`` map loaded from an artifact (``pca``, or an
    applied ``subspace`` fit): same math as [`Subspace`][], no
    parametrization, never trainable."""

    weight: torch.Tensor

    def __init__(self, kind: str, weight: torch.Tensor) -> None:
        super().__init__()
        self.kind = kind
        self.register_buffer("weight", weight)

    def featurize(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        weight = self.weight
        f = x.to(weight.dtype) @ weight
        return f, x - f @ weight.T

    def inverse(self, f: torch.Tensor, err: torch.Tensor | None) -> torch.Tensor:
        x = f @ self.weight.T
        return x if err is None else x + err


class Standardize(Stage):
    kind = "standardize"

    mu: torch.Tensor
    sigma: torch.Tensor

    def __init__(self, mu: torch.Tensor, sigma: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("mu", mu)
        self.register_buffer("sigma", sigma)

    def featurize(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        return (x - self.mu) / self.sigma, None

    def inverse(self, f: torch.Tensor, err: torch.Tensor | None) -> torch.Tensor:
        return f * self.sigma + self.mu


class Sae(Stage):
    """A loaded sparse autoencoder: ``(enc(x), x − dec(enc(x)))``."""

    kind = "sae"

    enc: torch.Tensor
    dec: torch.Tensor
    b_enc: torch.Tensor
    b_dec: torch.Tensor

    def __init__(
        self,
        enc: torch.Tensor,
        dec: torch.Tensor,
        b_enc: torch.Tensor,
        b_dec: torch.Tensor,
    ) -> None:
        super().__init__()
        self.register_buffer("enc", enc)
        self.register_buffer("dec", dec)
        self.register_buffer("b_enc", b_enc)
        self.register_buffer("b_dec", b_dec)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu((x - self.b_dec) @ self.enc + self.b_enc)

    def _decode(self, f: torch.Tensor) -> torch.Tensor:
        return f @ self.dec + self.b_dec

    def featurize(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        f = self._encode(x)
        return f, x - self._decode(f)

    def inverse(self, f: torch.Tensor, err: torch.Tensor | None) -> torch.Tensor:
        x = self._decode(f)
        return x if err is None else x + err
