"""The ``subspace`` featurizer's ``cayley`` parametrization (spec §2.5).

The bug this guards: ``torch.nn.utils.parametrizations.orthogonal`` with
``orthogonal_map="cayley"`` embeds a ``(d, k)`` weight into a **d×d**
skew-symmetric matrix, solves a d×d linear system, and multiplies by a d×d
``base`` buffer it completes from the *global* RNG at build time — the solve
and the multiply on every ``.weight`` access, so twice per forward group. For
a DAS fit at ``d = 4096, k = 8`` that is a 4096³ solve and 64 MB of
intermediates to move eight 4096-vectors, and it is no cheaper than the
matrix exponential it was meant to replace. The map here is the same Cayley
transform written through the Woodbury identity on the rank-``2k`` skew
matrix, so every tensor it touches is ``(d, k)`` or smaller.

Two contracts pinned below, beyond the numerics:

* the map is a **function of the seed alone** — no global-RNG draw anywhere,
  not in the init (already pinned by ``test_featurizer_seed.py``) and not in
  the parametrization's basis, so the same ``original`` under two global
  states materializes the same ``weight``;
* the optimizer-facing shape is unchanged: the trained tensor is still
  ``parametrizations.weight.original`` of shape ``(d, k)``, which is what the
  train loop's snapshot/restore and its tests address.
"""

from __future__ import annotations

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from causalab.neural.shared.featurizers import ORTHONORMAL_TOLERANCE, Cayley, Subspace

TRAINED_KEY = "parametrizations.weight.original"

#: The map's own fp32 guarantee: ``max|QᵀQ − I|`` for any ``original`` the
#: property test draws. Measured worst cases over 9000 draws (k ≤ 6, d ≤ 46,
#: entries up to N(0, 3²)): 1.3e-5 for generic columns — the ``k = 1`` draw
#: whose ``x ∥ Q₀`` leaves ``X⊥`` as pure cancellation noise, 1.15e-5 on CI's
#: BLAS — and 1.35e-5 for a collinear pair of length ≈ 20. A 3× margin on
#: both; a 10× regression in either regime trips it.
FP32_STIEFEL_BOUND = 5e-5

DEVICES = [
    "cpu",
    *(["cuda"] if torch.cuda.is_available() else []),
    *(["mps"] if torch.backends.mps.is_available() else []),
]


def _dense_cayley(x: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
    """The textbook map, in fp64, as the oracle: ``(I − A/2)⁻¹ (I + A/2) Q₀``
    with ``A = X Q₀ᵀ − Q₀ Xᵀ``."""
    x, base = x.double(), base.double()
    a = x @ base.T - base @ x.T
    eye = torch.eye(a.shape[0], dtype=a.dtype)
    return torch.linalg.solve(eye - 0.5 * a, eye + 0.5 * a) @ base


def _subspace(width: int, k: int, seed: int = 0) -> Subspace:
    return Subspace(width, k, "cayley", seed=seed)


def _original(stage: Subspace) -> torch.nn.Parameter:
    return dict(stage.named_parameters())[TRAINED_KEY]  # type: ignore[return-value]


def _set_original(stage: Subspace, value: torch.Tensor) -> None:
    with torch.no_grad():
        _original(stage).copy_(value)


def _base(stage: Subspace) -> torch.Tensor:
    return stage.parametrizations.weight[0].base  # type: ignore[union-attr]


def _orthonormality_error(q: torch.Tensor) -> float:
    k = q.shape[-1]
    return float((q.T @ q - torch.eye(k, dtype=q.dtype)).detach().abs().max())


def _draw(width: int, k: int, seed: int, scale: float, collinear: bool) -> torch.Tensor:
    """An ``original``: iid Gaussian columns, or with the second column a copy
    of the first — the rank-deficient ``X⊥`` that is the map's hard case."""
    x = scale * torch.randn(width, k, generator=torch.Generator().manual_seed(seed))
    if collinear and k > 1:
        x[:, 1] = x[:, 0]
    return x


# --------------------------------------------------------------------- #
# the numerics
# --------------------------------------------------------------------- #


@pytest.mark.property
class TestCayleyMap:
    @given(
        k=st.integers(min_value=1, max_value=6),
        extra=st.integers(min_value=0, max_value=40),
        seed=st.integers(min_value=0, max_value=2**16),
        scale=st.sampled_from([0.01, 0.3, 3.0]),
        collinear=st.booleans(),
    )
    @settings(max_examples=60, deadline=None)
    def test_the_weight_is_orthonormal_for_any_original(
        self, k: int, extra: int, seed: int, scale: float, collinear: bool
    ) -> None:
        """The whole point of a parametrization: wherever the optimizer puts
        ``original``, ``weight`` is a Stiefel point — including ``k = d``
        (``extra = 0``), a rotation of the whole space; ``scale = 3``, columns
        of ``original`` far longer than any fit drifts to; the redundant
        direction (``seed`` shared with the base draws ``x ∥ Q₀`` at ``k = 1``,
        pure cancellation in ``X⊥``); and **collinear** columns, the
        rank-deficient ``X⊥`` where the Woodbury form's conditioning is
        quadratic in ``‖X‖`` ([`Cayley`][causalab.neural.shared.featurizers.stages.Cayley], *Conditioning*).

        In fp64 the residual is rounding — the formula is exact. In fp32 the
        bound is `FP32_STIEFEL_BOUND`, the map's own guarantee; measured,
        the two regimes share the same worst case at these sizes, so one bound
        serves both (see the constant)."""
        width = k + extra
        stage = _subspace(width, k, seed=seed)
        x = _draw(width, k, seed, scale, collinear)
        _set_original(stage, x)
        assert _orthonormality_error(stage.weight) < FP32_STIEFEL_BOUND
        base = torch.linalg.qr(
            torch.randn(
                width,
                k,
                dtype=torch.float64,
                generator=torch.Generator().manual_seed(seed),
            )
        )[0]
        assert _orthonormality_error(Cayley(base)(x.double())) < 1e-12

    @given(
        k=st.integers(min_value=1, max_value=6),
        extra=st.integers(min_value=0, max_value=40),
        seed=st.integers(min_value=0, max_value=2**16),
        collinear=st.booleans(),
    )
    @settings(max_examples=40, deadline=None)
    def test_matches_the_dense_cayley_transform(
        self, k: int, extra: int, seed: int, collinear: bool
    ) -> None:
        """Same map as the d×d one, just not computed at d×d: the low-rank
        form is an identity, not an approximation."""
        width = k + extra
        stage = _subspace(width, k, seed=seed)
        x = _draw(width, k, seed, 1.0, collinear)
        _set_original(stage, x)
        torch.testing.assert_close(
            stage.weight.double(),
            _dense_cayley(x, _base(stage)),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_gradients_are_exact(self) -> None:
        """The optimizer steps ``original`` through this map, so its Jacobian
        has to be the map's — checked against finite differences in fp64."""
        base = torch.linalg.qr(
            torch.randn(
                9, 3, dtype=torch.float64, generator=torch.Generator().manual_seed(2)
            )
        )[0]
        cayley = Cayley(base)
        x = 0.5 * torch.randn(
            9, 3, dtype=torch.float64, generator=torch.Generator().manual_seed(2)
        )
        x.requires_grad_(True)
        assert torch.autograd.gradcheck(cayley, (x,), atol=1e-6, rtol=1e-4)

    def test_the_map_does_not_read_the_global_rng(self) -> None:
        """Torch's version completed a d×d basis with ``torch.randn`` at build
        time, so the *trajectory* of a fit depended on the global RNG even
        though its starting point did not. Here two stages built under
        different global states are the same function."""
        x = torch.randn(24, 3, generator=torch.Generator().manual_seed(5))

        def weight_under(global_seed: int) -> torch.Tensor:
            torch.manual_seed(global_seed)
            torch.randn(64)
            stage = _subspace(24, 3, seed=1)
            _set_original(stage, x)
            return stage.weight.detach().clone()

        assert torch.equal(weight_under(0), weight_under(999_999))

    def test_nothing_it_holds_is_wider_than_the_weight(self) -> None:
        """The cost contract: no d×d state and no d×d autograd intermediate.
        At ``d = 4096`` torch's map kept a 64 MB ``base`` and saved a d×d LU
        factorization for backward; here every state tensor is ``(d, k)`` and
        nothing saved for backward exceeds ``(d, k)`` — the ``(d, 2k)`` ``P``
        of the derivation is never materialized, and this is what would catch
        a rewrite that concatenates it."""
        width, k = 4096, 8
        stage = _subspace(width, k)
        for key, tensor in stage.state_dict().items():
            assert tuple(tensor.shape) == (width, k), key

        saved: list[int] = []

        def pack(t: torch.Tensor) -> torch.Tensor:
            saved.append(t.numel())
            return t

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
            loss = stage.weight.sum()
        loss.backward()
        assert saved, "the map saved nothing — is it differentiable?"
        assert max(saved) <= width * k

    def test_the_weight_is_row_major_like_its_saved_copy(self) -> None:
        """``torch.linalg.qr`` returns a column-major ``Q``, and the map's
        output inherits its base's layout; had the base kept that layout the
        live weight and the row-major tensor a bundle reloads would go through
        different matmul kernels and differ by an ulp — the drift
        ``test_rotation_round_trip.py`` exists to rule out."""
        stage = _subspace(64, 4)
        _set_original(
            stage, 0.1 * torch.randn(64, 4, generator=torch.Generator().manual_seed(9))
        )
        hazard = torch.linalg.qr(
            torch.randn(64, 4, generator=torch.Generator().manual_seed(0))
        )[0]
        if hazard.is_contiguous():
            pytest.skip("this LAPACK returns a row-major Q — the hazard is gone")
        assert _base(stage).is_contiguous()
        # measured, not assumed: `base - 2.0 * (matmul)` keeps a column-major
        # base's strides, and `x @ q` then differs from `x @ q.contiguous()`
        assert stage.weight.is_contiguous()

    @pytest.mark.parametrize("device", DEVICES)
    def test_trains_on_every_device_the_engine_accepts(self, device: str) -> None:
        """Forward *and* backward on the run's device. ``torch.linalg.solve``
        has no MPS backward kernel (its ``linalg_lu_solve``) in torch 2.9 —
        which is also why torch's own ``cayley`` could not fit on a developer
        Mac; the ``inv`` form here can. ``cpu`` always runs; ``mps``/``cuda``
        prove the foreign device when the box has one."""
        stage = _subspace(64, 4).to(device)
        _set_original(stage, 0.3 * torch.randn(64, 4, device=device))
        q = stage.weight
        assert q.device.type == device
        assert _orthonormality_error(q.cpu()) < FP32_STIEFEL_BOUND
        q.sum().backward()
        grad = _original(stage).grad
        assert grad is not None and bool(torch.isfinite(grad).all())

    def test_the_trained_tensor_keeps_its_name_and_shape(self) -> None:
        """What ``train.py`` snapshots and its tests address — and the same
        ``(d, k)`` torch's ``cayley`` trained, so the trainable surface is
        unchanged."""
        stage = _subspace(16, 4)
        original = _original(stage)
        assert original.requires_grad
        assert tuple(original.shape) == (16, 4)
        assert TRAINED_KEY in stage.state_dict()

    def test_each_parametrization_still_yields_a_stiefel_point(self) -> None:
        """The vocabulary is a spec commitment (§2.5); the other two maps stay
        on torch's implementation. Read at ``original = 0``, where every map is
        exact, so the residual is the QR init's — tighter than
        `FP32_STIEFEL_BOUND`, which is about the chart away from it."""
        for parametrization in ("cayley", "matrix_exp", "stiefel"):
            stage = Subspace(12, 3, parametrization, seed=0)
            assert _orthonormality_error(stage.weight) < 1e-5


@pytest.mark.numerical_unit
class TestTheStart:
    def test_the_init_is_the_seeded_qr_frame(self) -> None:
        """``original = 0`` maps to the base, so the starting rotation is the
        seeded QR init exactly — the same start ``test_featurizer_seed.py``
        pins, and what an untrained ``random_subspace_control`` document
        evaluates."""
        stage = _subspace(32, 4, seed=11)
        expected = torch.linalg.qr(
            torch.randn(32, 4, generator=torch.Generator().manual_seed(11))
        )[0]
        assert torch.equal(_original(stage), torch.zeros(32, 4))
        torch.testing.assert_close(stage.weight, expected, atol=1e-6, rtol=0.0)


@pytest.mark.numerical_unit
class TestConditioning:
    def test_the_conditioning_limit_is_where_the_docstring_says(self) -> None:
        """Pins the measured shape of the one regime the low-rank form loses
        to the dense solve, so the *Conditioning* section cannot silently go
        stale: two parallel columns of length ``s`` cost ``κ(S) ~ s²/2``.
        Measured at this size: 4.8e-7 at ``s = 10``, 1.5e-4 at ``s = 100``.
        The ratio is the claim itself — quadratic, not linear — and is what
        catches a fix to the conditioning that leaves this section describing
        a limit that no longer exists."""
        d, k = 256, 4
        base = torch.linalg.qr(
            torch.randn(d, k, generator=torch.Generator().manual_seed(1))
        )[0]
        x = torch.randn(d, k, generator=torch.Generator().manual_seed(2))
        x[:, 1] = x[:, 0]
        x = x / x.norm(dim=0, keepdim=True)
        err_10 = _orthonormality_error(Cayley(base)(10.0 * x))
        err_100 = _orthonormality_error(Cayley(base)(100.0 * x))
        assert err_10 < 1e-5
        assert err_100 < 3e-4  # the docstring's ~1e-4, one BLAS of slack
        assert err_100 > 10 * err_10


@pytest.mark.unit
class TestPolicy:
    def test_the_maps_bound_stays_within_the_load_tolerance(self) -> None:
        """The two policies are distinct — this map's own fp32 guarantee
        (`FP32_STIEFEL_BOUND`, over the regimes the property test draws)
        and what ``_init_basis`` accepts as a start — but the first must not
        drift past the second, or tightening one would silently retune the
        other. Outside those regimes nothing static bounds the deviation, which
        is why the fit *records* it
        ([`fit_diagnostics`][causalab.neural.engines.pytorch_hooks.train.fit_diagnostics];
        the runtime check is ``test_train.py``'s subspace-diagnostic test)."""
        assert FP32_STIEFEL_BOUND <= ORTHONORMAL_TOLERANCE


# --------------------------------------------------------------------- #
# assigning a weight
# --------------------------------------------------------------------- #


@pytest.mark.unit
class TestAssignment:
    def test_assigning_an_orthonormal_weight_rebases_the_map(self) -> None:
        """``stage.weight = Q`` means "start from Q": the base moves, the
        original returns to zero, and the weight reads back as Q."""
        stage = _subspace(20, 4)
        target = torch.linalg.qr(
            torch.randn(20, 4, generator=torch.Generator().manual_seed(3))
        )[0]
        setattr(stage, "weight", target)
        torch.testing.assert_close(stage.weight, target, atol=1e-6, rtol=0.0)
        assert torch.equal(_original(stage), torch.zeros(20, 4))

    def test_assigning_a_non_orthonormal_weight_is_refused(self) -> None:
        """A silent re-orthonormalization would make the loaded frame differ
        from the one a reader asked for; refuse instead."""
        stage = _subspace(20, 4)
        skewed = torch.randn(20, 4, generator=torch.Generator().manual_seed(0))
        with pytest.raises(ValueError, match="orthonormal"):
            setattr(stage, "weight", skewed)


@pytest.mark.unit
class TestClampGate:
    """§2.5 ``parametrization: clamp`` on a gate — the DCM relaxation.

    The parameter *is* the mask: soft ``m = θ``, projected into ``[0, 1]``
    after every optimizer step, hard ``θ > ½``. Pinned here against the
    sigmoid gate it sits beside, and the identity check that keeps a bundle
    fitted under one map from being replayed under the other."""

    def test_the_soft_mask_is_theta_and_the_hard_mask_rounds(self) -> None:
        from causalab.neural.shared.featurizers import Gate

        gate = Gate(4, parametrization="clamp")
        with torch.no_grad():
            gate.theta.copy_(torch.tensor([0.2, 0.5, 0.51, 0.9]))
        x = torch.ones(2, 4)
        gate.train(True)
        f, err = gate.featurize(x)
        assert torch.allclose(f, gate.theta * x)
        assert err is not None and torch.allclose(err, (1 - gate.theta) * x)
        gate.eval()
        f, _ = gate.featurize(x)
        assert torch.equal(f[0], torch.tensor([0.0, 0.0, 1.0, 1.0]))  # ½ is off
        assert torch.equal(gate.hard_mask(), torch.tensor([0.0, 0.0, 1.0, 1.0]))
        assert torch.equal(gate.soft_mask(), gate.theta)

    def test_project_clips_into_the_unit_interval(self) -> None:
        from causalab.neural.shared.featurizers import Gate

        gate = Gate(4, parametrization="clamp")
        with torch.no_grad():
            gate.theta.copy_(torch.tensor([-3.0, 0.25, 1.0, 7.0]))
        gate.project()
        assert torch.equal(gate.theta, torch.tensor([0.0, 0.25, 1.0, 1.0]))
        sigmoid = Gate(4)
        with torch.no_grad():
            sigmoid.theta.copy_(torch.tensor([-3.0, 0.25, 1.0, 7.0]))
        sigmoid.project()  # a no-op for the sigmoid gate
        assert torch.equal(sigmoid.theta, torch.tensor([-3.0, 0.25, 1.0, 7.0]))

    def test_both_maps_start_at_the_midpoint_mask(self) -> None:
        from causalab.neural.shared.featurizers import Gate

        assert torch.equal(Gate(3).theta, torch.zeros(3))
        assert torch.equal(
            Gate(3, parametrization="clamp").theta, torch.full((3,), 0.5)
        )
        for gate in (Gate(3), Gate(3, parametrization="clamp")):
            assert torch.allclose(gate.soft_mask(), torch.full((3,), 0.5))
            assert float(gate.hard_mask().sum()) == 0.0  # the midpoint is "off"

    def test_a_grouped_clamp_gate_expands_per_head(self) -> None:
        from causalab.neural.shared.featurizers import Gate

        gate = Gate(8, group="head", groups=(2, 4), parametrization="clamp")
        with torch.no_grad():
            gate.theta.copy_(torch.tensor([0.9, 0.1]))
        gate.eval()
        f, _ = gate.featurize(torch.ones(1, 8))
        assert torch.equal(f[0], torch.tensor([1.0] * 4 + [0.0] * 4))

    def test_a_site_gate_is_one_parameter_broadcast_over_the_site(self) -> None:
        """§2.5 `group: site`: the `head` map with one group. At the poles the
        gate is the plain swap (θ → +∞ routes every coordinate) or the plain
        base (θ → −∞ routes none); the hard split is one 0/1 entry."""
        from causalab.neural.shared.featurizers import Gate

        gate = Gate(8, group="site", groups=(1, 8))
        assert gate.theta.shape == (1,)
        x = torch.arange(8.0).view(1, 8)
        for theta, want_kept in ((40.0, x), (-40.0, torch.zeros_like(x))):
            with torch.no_grad():
                gate.theta.fill_(theta)
            gate.train(True)
            kept, rest = gate.featurize(x)
            assert torch.allclose(kept, want_kept, atol=1e-12)
            assert torch.allclose(kept + rest, x)
            gate.eval()
            kept, _ = gate.featurize(x)
            assert torch.equal(kept, want_kept)
            assert gate.hard_mask().shape == (1,)
            assert float(gate.hard_mask().sum()) == (1.0 if theta > 0 else 0.0)
        loaded = Gate.from_theta(
            torch.tensor([0.5]), group="site", groups=(1, 8), width=8, top_k=1
        )
        loaded.eval()  # the executor's mode for a loaded gate: the hard split
        assert loaded.theta.shape == (1,)
        assert torch.equal(loaded.featurize(x)[0], x)
        with pytest.raises(ValueError, match="does not tile"):
            Gate(8, group="site", groups=(1, 4))

    def test_from_theta_carries_the_parametrization(self) -> None:
        from causalab.neural.shared.featurizers import Gate

        gate = Gate.from_theta(torch.tensor([0.7, 0.2]), parametrization="clamp")
        assert gate.parametrization == "clamp"
        assert torch.equal(gate.hard_mask(), torch.tensor([1.0, 0.0]))
        assert Gate.from_theta(torch.tensor([0.7, -0.2])).parametrization == "sigmoid"

    def test_an_unknown_parametrization_is_refused(self) -> None:
        from causalab.neural.shared.featurizers import Gate

        with pytest.raises(ValueError, match="sigmoid"):
            Gate(4, parametrization="softplus")

    def test_the_entry_identity_check_holds_the_map_in_both_directions(self) -> None:
        """A bundle fitted under one map is not a mask under the other, and an
        unstamped record is a sigmoid gate (fitted before the field existed)."""
        from causalab.neural.shared.featurizers.build import (
            _check_entry_identity,  # pyright: ignore[reportPrivateUsage]
        )
        from causalab.protocol.rules.errors import ProtocolError
        from causalab.protocol.schema import FeaturizerSpec

        clamp = FeaturizerSpec(kind="gate", parametrization="clamp")
        plain = FeaturizerSpec(kind="gate")
        _check_entry_identity({"parametrization": "clamp"}, clamp, "w")
        _check_entry_identity({"parametrization": "sigmoid"}, plain, "w")
        _check_entry_identity({}, plain, "w")  # pre-field bundle, sigmoid document
        with pytest.raises(ProtocolError, match="fitted 'sigmoid'"):
            _check_entry_identity({}, clamp, "w")
        with pytest.raises(ProtocolError, match="fitted 'clamp'"):
            _check_entry_identity({"parametrization": "clamp"}, plain, "w")


@pytest.mark.unit
class TestHardConcreteGate:
    """§2.5 ``parametrization: hard_concrete`` — Louizos et al. 2018's
    stochastic L0 relaxation: sampled in training from a draw the loop makes
    once per step, deterministic at eval, with the sigmoid gate's ``θ > 0``
    split at the default stretch."""

    def test_defaults_start_and_split_like_a_sigmoid_gate(self) -> None:
        from causalab.neural.shared.featurizers import Gate

        gate = Gate(4, parametrization="hard_concrete")
        assert torch.equal(gate.theta, torch.zeros(4))
        assert gate.temperature == pytest.approx(2.0 / 3.0)
        assert gate.stretch == (-0.1, 1.1)
        # exactly 0, not the ≈ −4e−16 the derived logit((½−γ)/(ζ−γ)) rounds
        # to: at the default start every θ is exactly 0 and must NOT be kept,
        # as under sigmoid — the whole point of "reloads through the same check"
        assert gate.hard_threshold() == 0.0
        assert torch.equal(gate.hard_mask(), torch.zeros(4))
        assert float(gate.hard_mask().sum()) == 0.0
        with torch.no_grad():
            gate.theta.copy_(torch.tensor([-2.0, -1e-3, 1e-3, 3.0]))
        assert torch.equal(gate.hard_mask(), torch.tensor([0.0, 0.0, 1.0, 1.0]))

    def test_the_threshold_is_one_number_computed_in_one_place(self) -> None:
        import math

        from causalab.neural.shared.featurizers import Gate
        from causalab.protocol.schema import hard_concrete_threshold

        # symmetric stretches are exactly 0; an asymmetric one is the logit
        for stretch in ((-0.1, 1.1), (-0.3, 1.3), (-1.0, 2.0)):
            assert hard_concrete_threshold(stretch) == 0.0
            assert (
                Gate(
                    2, parametrization="hard_concrete", stretch=stretch
                ).hard_threshold()
                == 0.0
            )
        q = (0.5 + 0.1) / 1.6
        expected = math.log(q / (1 - q))
        gate = Gate(2, parametrization="hard_concrete", stretch=(-0.1, 1.5))
        assert gate.hard_threshold() == pytest.approx(expected)
        assert gate.hard_threshold() == hard_concrete_threshold((-0.1, 1.5))
        # and the split is where the deterministic mask crosses ½
        with torch.no_grad():
            gate.theta.copy_(torch.tensor([expected - 1e-3, expected + 1e-3]))
        assert torch.equal(gate.hard_mask(), torch.tensor([0.0, 1.0]))
        soft = gate.soft_mask().detach()
        assert float(soft[0]) < 0.5 < float(soft[1])

    def test_the_deterministic_mask_is_the_stretched_clipped_sigmoid(self) -> None:
        from causalab.neural.shared.featurizers import Gate

        gate = Gate(3, parametrization="hard_concrete")
        with torch.no_grad():
            gate.theta.copy_(torch.tensor([-6.0, 0.0, 6.0]))
        expected = (torch.sigmoid(gate.theta) * 1.2 - 0.1).clamp(0.0, 1.0)
        assert torch.allclose(gate.soft_mask(), expected)
        soft = gate.soft_mask().detach()
        assert float(soft[0]) == 0.0  # clipped onto the pole
        assert float(soft[2]) == 1.0
        assert float(soft[1]) == pytest.approx(0.5)

    def test_training_samples_a_mask_on_the_unit_interval_around_its_mean(self) -> None:
        from causalab.neural.shared.featurizers import Gate

        gate = Gate(1, parametrization="hard_concrete")
        with torch.no_grad():
            gate.theta.fill_(0.8)
        gate.train()
        generator = torch.Generator().manual_seed(0)
        draws = []
        for _ in range(4000):
            gate.resample(generator)
            draws.append(gate.sampled_mask().detach())
        stacked = torch.stack(draws)
        assert float(stacked.min()) >= 0.0 and float(stacked.max()) <= 1.0
        assert float(stacked.std()) > 0.05  # it is a sample, not the mean
        # a training forward uses the step's draw; eval mode uses the
        # deterministic mask and then the hard split
        f_train, _ = gate.featurize(torch.ones(1, 1))
        assert 0.0 <= float(f_train.detach()) <= 1.0
        gate.eval()
        f_eval, _ = gate.featurize(torch.ones(1, 1))
        assert float(f_eval) == 1.0  # θ = 0.8 > 0

    def test_one_draw_per_step_is_shared_by_every_featurize_of_the_step(self) -> None:
        """The DBM interchange puts one gate on the read and the write: two
        independent draws would double-write or erase units. One ``resample``,
        then every ``featurize`` reads the same mask until the next."""
        from causalab.neural.shared.featurizers import Gate

        gate = Gate(64, parametrization="hard_concrete")
        gate.train()
        gate.resample(torch.Generator().manual_seed(0))
        read, _ = gate.featurize(torch.ones(2, 64))
        _, err = gate.featurize(torch.ones(2, 64))
        assert torch.equal(read + err, torch.ones(2, 64))  # m·x + (1−m)·x = x
        assert torch.equal(gate.sampled_mask(), gate.sampled_mask())
        before = gate.sampled_mask().detach().clone()
        gate.resample(torch.Generator().manual_seed(1))
        assert not torch.equal(gate.sampled_mask(), before)  # the next step draws anew

    def test_the_draw_is_a_function_of_the_generator_not_the_global_rng(self) -> None:
        from causalab.neural.shared.featurizers import Gate

        gate = Gate(8, parametrization="hard_concrete")
        gate.resample(torch.Generator().manual_seed(3))
        a = gate.sampled_mask()
        torch.manual_seed(99)
        torch.rand(5)  # the global stream moves; the gate's does not care
        gate.resample(torch.Generator().manual_seed(3))
        assert torch.equal(a, gate.sampled_mask())
        gate.resample(torch.Generator().manual_seed(4))
        assert not torch.equal(a, gate.sampled_mask())

    def test_a_training_forward_without_a_draw_refuses(self) -> None:
        from causalab.neural.shared.featurizers import Gate

        gate = Gate(4, parametrization="hard_concrete")
        gate.train()
        with pytest.raises(RuntimeError, match="resample"):
            gate.featurize(torch.ones(1, 4))
        gate.eval()
        gate.featurize(torch.ones(1, 4))  # eval needs no draw

    def test_expected_l0_is_louizos_closed_form_and_hard_concretes_alone(
        self,
    ) -> None:
        import math

        from causalab.neural.shared.featurizers import Gate

        gate = Gate(
            3, parametrization="hard_concrete", temperature=0.5, stretch=[-0.2, 1.2]
        )
        with torch.no_grad():
            gate.theta.copy_(torch.tensor([-1.0, 0.0, 2.0]))
        expected = torch.sigmoid(gate.theta - 0.5 * math.log(0.2 / 1.2))
        assert torch.allclose(gate.expected_l0(), expected)
        # a deterministic gate's relaxed mask is already its kept probability
        # — its mean is the l1 term, so l0 there is refused, not aliased
        for other in (Gate(3), Gate(3, parametrization="clamp")):
            with pytest.raises(ValueError, match="l1"):
                other.expected_l0()

    def test_the_constants_are_hard_concretes_alone_and_are_checked(self) -> None:
        from causalab.neural.shared.featurizers import Gate

        with pytest.raises(ValueError, match="hard_concrete"):
            Gate(3, temperature=0.5)
        with pytest.raises(ValueError, match="hard_concrete"):
            Gate(3, parametrization="clamp", stretch=(-0.1, 1.1))
        with pytest.raises(ValueError, match="positive"):
            Gate(3, parametrization="hard_concrete", temperature=0.0)
        with pytest.raises(ValueError, match="γ < 0 < 1 < ζ"):
            Gate(3, parametrization="hard_concrete", stretch=(0.0, 1.0))
        with pytest.raises(ValueError, match="only a hard_concrete"):
            Gate(3).sampled_mask()
        with pytest.raises(ValueError, match="only a hard_concrete"):
            Gate(3).resample(torch.Generator().manual_seed(0))

    def test_a_fill_is_the_start_mask_exactly_and_the_bundle_stamps_the_stretch(
        self,
    ) -> None:
        """``init.fill p`` means "every unit starts at mask value p" under every
        map (§2.5): under ``hard_concrete`` that is the deterministic mask, so
        θ inverts the stretch — ``logit((p−γ)/(ζ−γ))`` — rather than being
        ``logit(p)``, which would start ``fill: 0.99`` fully clipped at 1."""
        import json
        import math

        from causalab.neural.shared.featurizers import Gate

        for fill in (0.0, 0.05, 0.5, 0.9, 0.99, 1.0):
            # the poles are legal: the inverted stretch is finite there, unlike
            # the sigmoid gate's bare logit
            gate = Gate(3, parametrization="hard_concrete", init=fill)
            assert torch.allclose(gate.soft_mask(), torch.full((3,), fill), atol=1e-6)
            assert gate.init_fill == fill
        with pytest.raises(ValueError, match="sigmoid gate"):
            Gate(3, init=0.0)
        gate = Gate(3, parametrization="hard_concrete", init=0.99)
        p = (0.99 + 0.1) / 1.2
        assert torch.allclose(gate.theta, torch.full((3,), math.log(p / (1 - p))))
        assert float(gate.hard_mask().sum()) == 3.0
        # ½ ↔ θ = 0 survives at the default stretch, as under sigmoid
        assert torch.equal(
            Gate(2, parametrization="hard_concrete", init=0.5).theta, torch.zeros(2)
        )
        assert json.loads(gate.identity_fields()["stretch"]) == [-0.1, 1.1]
        loaded = Gate.from_theta(
            torch.tensor([0.3, -0.3]),
            parametrization="hard_concrete",
            stretch=(-0.2, 1.2),
        )
        assert loaded.stretch == (-0.2, 1.2)
        assert loaded.parametrization == "hard_concrete"


def _theta_bundle(theta: torch.Tensor, **header):
    """A ``load_tensors`` over one hand-built fitted-gate bundle: ``theta`` and
    the two header keys every gate load requires, plus ``header``."""
    from causalab.io.tensor_files import TensorBundle

    bundle = TensorBundle(
        tensors={"theta": theta},
        entry_coords={},
        header={
            "trained_on": "weekdays/data#train",
            "parametrization": "sigmoid",
            **header,
        },
    )
    return lambda path: bundle


@pytest.mark.unit
class TestGateInit:
    """§2.5 ``init`` on a gate: a mask value every unit starts at, or a saved
    theta taken verbatim — and the checks that keep a start honest."""

    def test_a_fill_is_a_mask_value_under_both_maps(self) -> None:
        import math

        from causalab.neural.shared.featurizers import Gate

        sigmoid = Gate(3, init=0.99)
        clamp = Gate(3, parametrization="clamp", init=0.99)
        assert torch.allclose(sigmoid.theta, torch.full((3,), math.log(0.99 / 0.01)))
        assert torch.allclose(torch.sigmoid(sigmoid.theta), torch.full((3,), 0.99))
        assert torch.allclose(clamp.theta, torch.full((3,), 0.99))
        for gate in (sigmoid, clamp):
            assert gate.init_fill == 0.99
            assert float(gate.hard_mask().sum()) == 3.0  # "everything patched"
        assert Gate(3).init_fill is None

    def test_a_sigmoid_gate_refuses_a_start_at_a_pole_and_a_clamp_gate_takes_it(
        self,
    ) -> None:
        from causalab.neural.shared.featurizers import Gate

        for pole in (0.0, 1.0):
            with pytest.raises(ValueError, match="logit"):
                Gate(3, init=pole)
        assert torch.equal(
            Gate(3, parametrization="clamp", init=1.0).theta, torch.ones(3)
        )
        with pytest.raises(ValueError, match="in \\[0, 1\\]"):
            Gate(3, init=1.5)

    def test_a_saved_theta_is_taken_verbatim_at_the_gates_layout(self) -> None:
        from causalab.neural.shared.featurizers import Gate

        theta = torch.tensor([0.3, -2.0, 4.0])
        gate = Gate(3, init=theta)
        assert torch.equal(gate.theta, theta) and gate.init_fill is None
        assert gate.theta.requires_grad  # a start, not a load: it trains
        with pytest.raises(ValueError, match="needs 3 parameters"):
            Gate(3, init=torch.zeros(4))

    def _bundle(self, theta: torch.Tensor, **header):
        return _theta_bundle(theta, **header)

    def test_a_gate_starts_from_a_saved_theta_and_records_where(self) -> None:
        from causalab.neural.shared.featurizers import Gate, build_stack
        from causalab.protocol.schema import FeaturizerSpec

        theta = torch.tensor([1.0, -1.0, 2.0, -2.0])
        stack = build_stack(
            "g",
            {"g": FeaturizerSpec(kind="gate", init={"file_path": "fit/g.safetensors"})},
            width=4,
            load_tensors=self._bundle(theta),
            stage_cache={},
        )
        gate = stack.stages[0]
        assert isinstance(gate, Gate)
        assert torch.equal(gate.theta, theta) and gate.theta.requires_grad
        assert gate.identity_fields() == {"init_trained_on": "weekdays/data#train"}

    def test_a_saved_start_must_be_a_theta_of_this_gate(self) -> None:
        """Same map, same layout: the checks a loaded gate passes, applied to
        a start."""
        from causalab.neural.shared.featurizers import build_stack
        from causalab.protocol.rules.errors import ProtocolError
        from causalab.protocol.schema import FeaturizerSpec

        def build(spec: FeaturizerSpec, loader) -> None:
            build_stack("g", {"g": spec}, width=4, load_tensors=loader, stage_cache={})

        plain = FeaturizerSpec(kind="gate", init={"file_path": "fit/g.safetensors"})
        with pytest.raises(ProtocolError, match="fitted 'clamp'"):
            build(plain, self._bundle(torch.zeros(4), parametrization="clamp"))
        with pytest.raises(ProtocolError, match="has 4 units"):
            build(plain, self._bundle(torch.zeros(6)))


@pytest.mark.unit
def test_every_field_a_stage_can_stamp_is_an_artifact_identity_key() -> None:
    """``Stage.identity_fields()`` is splatted into ``build_artifact_identity``
    on the bundle-stamp path (``execution.featurizer_identity``), which refuses
    a key outside ``ARTIFACT_IDENTITY_KEYS`` — after the whole fit has run. The
    census in ``tests/protocol/test_model_realization.py`` reads the literal
    keys a writer stamps — ``execution.py``'s, and since the round that
    registered ``axis`` and ``forward``, the ``fields[...]`` assignments of
    ``Gate.identity_fields`` too, which is the census closed over *fields*.
    This one is closed over *classes*: ``stretch`` walked through a green
    suite before either existed and the first real hard-concrete fit died at
    its first save, so every stage producer that stamps anything is built
    here and its runtime keys checked against the closed schema, with no
    model and no run — the instance list is hand-maintained."""
    from causalab.neural.shared.featurizers import Gate, Subspace
    from causalab.io.env import ARTIFACT_IDENTITY_KEYS

    start_identity = {
        "init_trained_on": "weekdays/data#train",
        "init_components": [0, 1],
    }
    producers = [
        Gate(4),
        Gate(4, parametrization="clamp"),
        Gate(4, parametrization="hard_concrete"),
        Gate(4, parametrization="hard_concrete", stretch=(-0.2, 1.2)),
        Gate(4, init=torch.zeros(4), init_identity=start_identity),
        # the two stamps added since: a position gate's `axis` and a
        # straight-through fit's `forward` — each walked through a green
        # suite before it was registered, exactly as `stretch` had. This list
        # is closed over classes (below) and open over fields: a fourth stamp
        # needs its instance here; the field-level census is
        # test_model_realization.test_every_key_a_stage_stamps_is_an_artifact_identity_key
        Gate(3, axis="position"),
        Gate(4, forward="hard"),
        Gate(4, parametrization="clamp", forward="hard"),
        Subspace(8, 2, "cayley"),
        Subspace(
            8, 2, "cayley", init=torch.eye(8)[:, :2], init_identity=start_identity
        ),
    ]
    for stage in producers:
        undeclared = sorted(set(stage.identity_fields()) - set(ARTIFACT_IDENTITY_KEYS))
        assert not undeclared, (
            f"{stage.kind} stamps {undeclared}, which build_artifact_identity refuses"
        )
    # and the list is closed over the CLASS, not just these instances: every
    # Stage subclass that overrides identity_fields must have a producer here
    from causalab.neural.shared.featurizers import Stage

    overriding = {
        cls for cls in Stage.__subclasses__() if "identity_fields" in cls.__dict__
    }
    # isinstance, not type(): torch's `register_parametrization` rewrites a
    # Subspace instance's class to a dynamic subclass
    missing = sorted(
        cls.__name__
        for cls in overriding
        if not any(isinstance(stage, cls) for stage in producers)
    )
    assert not missing, f"{missing} override identity_fields; add a producer for each"


@pytest.mark.unit
class TestEntryIdentityStretch:
    """The build-time check is the one that reaches a swept producer whose
    entry the executing point selects (the load-time check bails on that case
    before comparing anything), so it compares ``stretch`` like the map — in
    both directions — else a layer-swept fit at ``[-0.1, 1.5]`` replayed by a
    swept apply authoring none would be split at 0, silently."""

    def _check(self, stamped: str | None, authored: tuple[float, float] | None):
        from causalab.neural.shared.featurizers.build import (
            _check_entry_identity,  # pyright: ignore[reportPrivateUsage]
        )
        from causalab.protocol.schema import FeaturizerSpec

        record = {"parametrization": "hard_concrete"}
        if stamped is not None:
            record["stretch"] = stamped
        spec = FeaturizerSpec(
            kind="gate", parametrization="hard_concrete", stretch=authored
        )
        _check_entry_identity(record, spec, "featurizer 'g' (fit/g.safetensors)")

    def test_a_non_default_stamp_is_refused_by_a_spec_authoring_none(self) -> None:
        from causalab.protocol.rules.errors import ProtocolError

        with pytest.raises(ProtocolError, match="fitted at stretch \\[-0.1, 1.5\\]"):
            self._check("[-0.1, 1.5]", None)

    def test_an_authored_stretch_is_compared_to_the_stamp(self) -> None:
        from causalab.protocol.rules.errors import ProtocolError

        with pytest.raises(ProtocolError, match="different thresholds"):
            self._check("[-0.1, 1.5]", (-0.2, 1.2))
        self._check("[-0.1, 1.5]", (-0.1, 1.5))

    def test_the_default_agrees_with_an_unauthored_spec_and_a_spelled_one(
        self,
    ) -> None:
        self._check("[-0.1, 1.1]", None)
        self._check("[-0.1, 1.1]", (-0.1, 1.1))
        self._check(None, None)  # a stamp from a fit that could not record one


@pytest.mark.unit
class TestDeadUnits:
    """§2.5 ``dead`` on a gate: ``freeze_after`` freezes a unit hard-off for n
    consecutive post-step projections and restores its θ thereafter; ``leak``
    adds ε to the training mask's derivative and leaves every forward value
    alone. Both are bookkept per unit and read back by ``dead_diagnostics``."""

    def test_a_unit_freezes_after_n_hard_off_projections_and_its_theta_is_restored(
        self,
    ) -> None:
        from causalab.neural.shared.featurizers import Gate

        gate = Gate(4, dead={"freeze_after": 2})
        with torch.no_grad():
            gate.theta.copy_(torch.tensor([-1.0, 1.0, -1.0, 1.0]))
        gate.project()  # streak 1: nothing frozen yet
        assert not gate._frozen.any()
        with torch.no_grad():
            gate.theta[0] = 2.0  # unit 0 comes back before the count is reached
        gate.project()  # unit 2 is at streak 2 and freezes at θ = −1; unit 0 resets
        assert gate._frozen.tolist() == [False, False, True, False]
        assert gate.dead_diagnostics() == {
            "frozen_units": 1.0,
            "reawakened_units": 1.0,  # unit 0 was off and is kept
            "dead": {"freeze_after": 2},
        }
        # an optimizer step that would reopen the frozen unit is undone by the
        # projection, to the bit; the other units keep their new values
        with torch.no_grad():
            gate.theta.copy_(torch.tensor([3.0, 3.0, 3.0, 3.0]))
        gate.project()
        assert gate.theta.tolist() == [3.0, 3.0, -1.0, 3.0]
        assert torch.equal(gate.hard_mask(), torch.tensor([1.0, 1.0, 0.0, 1.0]))

    def test_the_streak_reads_the_maps_own_threshold_after_the_clamp(self) -> None:
        """A clamp gate's split is θ > ½ and its projection runs first, so a
        step that overshoots below 0 is clipped to 0 and counted as off."""
        from causalab.neural.shared.featurizers import Gate

        gate = Gate(2, parametrization="clamp", dead={"freeze_after": 1})
        with torch.no_grad():
            gate.theta.copy_(torch.tensor([-0.3, 0.6]))
        gate.project()
        assert gate.theta.tolist() == pytest.approx([0.0, 0.6])
        assert gate._frozen.tolist() == [True, False]

    def test_a_leak_changes_the_backward_and_not_the_forward(self) -> None:
        """At θ = −50 the sigmoid has no gradient left; with ``leak: 0.5`` the
        unit's θ receives ``0.5 · ∂L/∂m`` while every mask value — training
        soft, eval hard — is the plain gate's to the bit."""
        from causalab.neural.shared.featurizers import Gate

        theta = torch.tensor([-50.0, 0.0, 50.0])
        plain, leaky = Gate(3), Gate(3, dead={"leak": 0.5})
        for gate in (plain, leaky):
            gate.train()
            with torch.no_grad():
                gate.theta.copy_(theta)
            f, err = gate.featurize(torch.ones(1, 3))
            (f * torch.tensor([[1.0, 2.0, 3.0]])).sum().backward()
        f_plain, _ = plain.featurize(torch.ones(1, 3))
        f_leaky, _ = leaky.featurize(torch.ones(1, 3))
        assert torch.equal(f_plain, f_leaky)
        assert plain.theta.grad is not None and leaky.theta.grad is not None
        assert plain.theta.grad[0] == pytest.approx(0.0, abs=1e-12)
        # ∂L/∂m = 1 at unit 0, 2 at unit 1 (plain σ' = ¼ there), 3 at unit 2
        assert torch.allclose(
            leaky.theta.grad, plain.theta.grad + 0.5 * torch.tensor([1.0, 2.0, 3.0])
        )
        plain.eval(), leaky.eval()
        assert torch.equal(plain.hard_mask(), leaky.hard_mask())
        assert torch.equal(
            plain.featurize(torch.ones(1, 3))[0], leaky.featurize(torch.ones(1, 3))[0]
        )
        assert leaky.dead_diagnostics()["dead"] == {"leak": 0.5}

    def test_the_bookkeeping_is_reported_for_a_gate_with_no_rule(self) -> None:
        from causalab.neural.shared.featurizers import Gate

        gate = Gate(2)
        with torch.no_grad():
            gate.theta.copy_(torch.tensor([-1.0, 1.0]))
        gate.project()
        with torch.no_grad():
            gate.theta.copy_(torch.tensor([1.0, 1.0]))
        assert gate.dead_diagnostics() == {"frozen_units": 0.0, "reawakened_units": 1.0}

    def test_the_rule_is_checked_at_construction(self) -> None:
        from causalab.neural.shared.featurizers import Gate

        with pytest.raises(ValueError, match="exactly one"):
            Gate(2, dead={"freeze_after": 1, "leak": 0.1})
        with pytest.raises(ValueError, match="exactly one"):
            Gate(2, dead={"freeze": 1})
        with pytest.raises(ValueError, match="positive"):
            Gate(2, dead={"freeze_after": 0})
        with pytest.raises(ValueError, match="strictly inside"):
            Gate(2, dead={"leak": 1.0})


@pytest.mark.unit
class TestTopKReadout:
    """§2.5 ``top_k``: a loaded gate read out at a count instead of through
    its map's threshold — the readout every ranking method needs, and the
    kept-count → score curve every mask method reports."""

    def test_the_hard_mask_is_the_k_largest_units_of_theta(self) -> None:
        from causalab.neural.shared.featurizers import Gate

        theta = torch.tensor([0.1, -2.0, 3.0, 0.5, -0.5, 2.0])
        gate = Gate.from_theta(theta, top_k=3)
        assert gate.top_k == 3
        assert torch.equal(gate.hard_mask(), torch.tensor([0.0, 0, 1, 1, 0, 1]))
        # the threshold split would keep four; the cut keeps exactly k
        assert int((theta > 0).sum()) == 4 and int(gate.hard_mask().sum()) == 3
        # eval-mode featurize routes through the same cut
        gate.eval()
        kept, dropped = gate.featurize(torch.ones(1, 6))
        assert torch.equal(kept[0], gate.hard_mask())
        assert torch.equal(dropped[0], 1.0 - gate.hard_mask())

    def test_zero_keeps_nothing_and_the_unit_count_keeps_everything(self) -> None:
        from causalab.neural.shared.featurizers import Gate

        theta = torch.tensor([0.1, -2.0, 3.0, 0.5])
        assert torch.equal(Gate.from_theta(theta, top_k=0).hard_mask(), torch.zeros(4))
        assert torch.equal(Gate.from_theta(theta, top_k=4).hard_mask(), torch.ones(4))

    def test_at_the_threshold_count_the_two_readouts_are_one_mask(self) -> None:
        """The identity the replay tests rest on: cutting the ranking at the
        fit's own ``hard_mask_size`` reproduces the threshold split exactly,
        under every map."""
        from causalab.neural.shared.featurizers import Gate

        theta = torch.randn(32, generator=torch.Generator().manual_seed(0))
        for parametrization, stretch in (
            ("sigmoid", None),
            ("hard_concrete", (-0.1, 1.1)),
            ("hard_concrete", (-0.1, 1.5)),
        ):
            plain = Gate.from_theta(
                theta, parametrization=parametrization, stretch=stretch
            )
            cut = Gate.from_theta(
                theta,
                parametrization=parametrization,
                stretch=stretch,
                top_k=int(plain.hard_mask().sum()),
            )
            assert torch.equal(plain.hard_mask(), cut.hard_mask()), parametrization
        clamp_theta = torch.rand(32, generator=torch.Generator().manual_seed(1))
        plain = Gate.from_theta(clamp_theta, parametrization="clamp")
        cut = Gate.from_theta(
            clamp_theta, parametrization="clamp", top_k=int(plain.hard_mask().sum())
        )
        assert torch.equal(plain.hard_mask(), cut.hard_mask())

    def test_the_ranking_is_by_theta_with_ties_toward_the_lower_index(self) -> None:
        """Ranking ``theta`` rather than the soft mask keeps units the
        hard-concrete clip saturates to exactly 1.0 apart, and a tie in
        ``theta`` is broken by index so the cut is a function of ``theta``."""
        from causalab.neural.shared.featurizers import Gate

        # both 8.0 and 6.0 saturate the stretched σ(θ) at 1.0; θ still orders them
        theta = torch.tensor([6.0, 8.0, -1.0, 2.0, 2.0])
        gate = Gate.from_theta(theta, parametrization="hard_concrete", top_k=1)
        assert gate.soft_mask()[0] == gate.soft_mask()[1] == 1.0
        assert torch.equal(gate.hard_mask(), torch.tensor([0.0, 1, 0, 0, 0]))
        assert gate.ranking().tolist() == [1, 0, 3, 4, 2]
        assert gate.rank().tolist() == [1, 0, 4, 2, 3]
        assert torch.equal(
            Gate.from_theta(
                theta, parametrization="hard_concrete", top_k=3
            ).hard_mask(),
            torch.tensor([1.0, 1, 0, 1, 0]),
        )

    def test_a_grouped_gate_counts_units_not_coordinates(self) -> None:
        from causalab.neural.shared.featurizers import Gate

        theta = torch.tensor([0.2, 0.9, -0.3])  # three heads of four coordinates
        gate = Gate.from_theta(theta, group="head", groups=(3, 4), width=12, top_k=1)
        gate.eval()
        kept, _ = gate.featurize(torch.ones(1, 12))
        assert torch.equal(kept[0], torch.tensor([0.0] * 4 + [1.0] * 4 + [0.0] * 4))
        assert gate.rank().tolist() == [1, 0, 2]

    def test_a_count_above_the_units_is_refused(self) -> None:
        from causalab.neural.shared.featurizers import Gate

        with pytest.raises(ValueError, match="top_k=5 on a gate of 4 units"):
            Gate.from_theta(torch.zeros(4), top_k=5)
        with pytest.raises(ValueError, match="top_k=-1"):
            Gate.from_theta(torch.zeros(4), top_k=-1)

    def test_a_fitted_gate_has_no_cut(self) -> None:
        from causalab.neural.shared.featurizers import Gate

        gate = Gate(4)
        assert gate.top_k is None
        assert torch.equal(gate.hard_mask(), (gate.theta > 0).float())


@pytest.mark.unit
class TestBudgetGate:
    """§2.5 ``parametrization: budget`` — a mask with no sparsity loss:
    ``σ(θ + c_k)`` with the shift solved so the mask sums to the step's budget,
    ``θ`` learned as a ranking, read out at a count."""

    @staticmethod
    def _gate(schedule: dict, units: int = 16, seed: int = 0, **kw):
        from causalab.neural.shared.featurizers import Gate

        gate = Gate(units, parametrization="budget", k_schedule=schedule, **kw)
        with torch.no_grad():
            gate.theta.copy_(
                torch.randn(units, generator=torch.Generator().manual_seed(seed))
            )
        return gate

    def test_the_shift_makes_the_mask_sum_to_the_budget(self) -> None:
        gate = self._gate({"kind": "fixed", "k": 5}, units=24)
        for k in (1, 2, 5, 12, 23):
            mask = gate.budget_mask(k)
            assert float(mask.sum()) == pytest.approx(k, abs=1e-4), k
            assert mask.shape == gate.theta.shape
            assert float(mask.min()) > 0.0 and float(mask.max()) < 1.0
        # the poles: no finite root, so the mask sits on the pole
        assert float(gate.budget_mask(0).sum()) == pytest.approx(0.0, abs=1e-6)
        assert float(gate.budget_mask(24).sum()) == pytest.approx(24.0, abs=1e-6)

    def test_a_fixed_budget_cuts_the_ranking_where_top_k_does(self) -> None:
        """The training mask's k largest entries are the eval-mode hard mask,
        which is the same cut a loaded gate makes at ``top_k = k`` (G3): one
        ranking, three readers."""
        from causalab.neural.shared.featurizers import Gate

        gate = self._gate({"kind": "fixed", "k": 4})
        gate.train()
        gate.resample(torch.Generator().manual_seed(0))
        soft_top = set(torch.topk(gate.budget_mask(4).detach(), 4).indices.tolist())
        gate.eval()
        hard = gate.hard_mask()
        assert int(hard.sum()) == 4
        assert set(hard.nonzero().flatten().tolist()) == soft_top
        loaded = Gate.from_theta(gate.theta.detach(), parametrization="budget", top_k=4)
        assert torch.equal(loaded.hard_mask(), hard)
        # `eval` overrides `k` for the in-fit readout
        wide = self._gate({"kind": "fixed", "k": 4, "eval": 6})
        with torch.no_grad():
            wide.theta.copy_(gate.theta)
        assert int(wide.hard_mask().sum()) == 6 and wide.eval_k() == 6

    def test_the_implicit_gradient_keeps_the_sum_fixed_and_stop_grad_frees_it(
        self,
    ) -> None:
        """Through the shift, ``Σ m`` is a constant of ``θ`` and its gradient
        vanishes; with ``stop_grad_shift`` the shift is a constant and the sum's
        gradient is the plain ``σ'``."""
        for stop, expect_zero in ((False, True), (True, False)):
            gate = self._gate({"kind": "fixed", "k": 5}, stop_grad_shift=stop)
            total = gate.budget_mask(5).sum()
            (grad,) = torch.autograd.grad(total, gate.theta)
            assert (float(grad.abs().max()) < 1e-5) is expect_zero, stop
            # either way the mask itself moves θ: a unit's own mask value
            # has a non-zero derivative in its θ
            gate.zero_grad()
            single = gate.budget_mask(5)[3]
            (grad,) = torch.autograd.grad(single, gate.theta)
            assert float(grad[3]) > 0.0

    def test_the_draw_follows_the_schedule_and_the_generator(self) -> None:
        fixed = self._gate({"kind": "fixed", "k": 3})
        assert fixed._draw_budget(torch.Generator().manual_seed(0)) == 3
        uniform = self._gate({"kind": "uniform", "low": 2, "high": 5, "eval": 3})
        gen = torch.Generator().manual_seed(0)
        draws = [uniform._draw_budget(gen) for _ in range(200)]
        assert set(draws) == {2, 3, 4, 5}
        log = self._gate({"kind": "log_uniform", "low": 1, "high": 15, "eval": 3})
        gen = torch.Generator().manual_seed(0)
        draws = [log._draw_budget(gen) for _ in range(400)]
        assert min(draws) >= 1 and max(draws) <= 15
        # log-uniform: as many draws in [1, 4) as in [4, 16) (each a factor of 4)
        small = sum(d < 4 for d in draws)
        assert 120 < small < 280
        again = [log._draw_budget(torch.Generator().manual_seed(0)) for _ in range(3)]
        assert again == [
            log._draw_budget(torch.Generator().manual_seed(0)) for _ in range(3)
        ]
        # resample keeps the step's k for the forward
        log.train()
        log.resample(torch.Generator().manual_seed(7))
        assert log._k in range(1, 16)
        assert float(log.featurize(torch.ones(1, 16))[0].sum()) == pytest.approx(
            log._k, abs=1e-3
        )

    def test_a_budget_gate_has_no_threshold_no_penalty_no_noise(self) -> None:
        from causalab.neural.shared.featurizers import Gate

        gate = self._gate({"kind": "fixed", "k": 2})
        with pytest.raises(ValueError, match="no threshold"):
            gate.hard_threshold()
        with pytest.raises(ValueError, match="expected L0"):
            gate.expected_l0()
        with pytest.raises(ValueError, match="samples nothing"):
            Gate(
                4,
                parametrization="budget",
                k_schedule={"kind": "fixed", "k": 1},
                temperature=0.5,
            )
        with pytest.raises(ValueError, match="draws no budget"):
            Gate(4, k_schedule={"kind": "fixed", "k": 1})
        with pytest.raises(ValueError, match="names 'eval'"):
            Gate(
                4,
                parametrization="budget",
                k_schedule={"kind": "uniform", "low": 1, "high": 3},
            )
        with pytest.raises(ValueError, match="starts at 1"):
            Gate(
                4,
                parametrization="budget",
                k_schedule={"kind": "log_uniform", "low": 0, "high": 3, "eval": 1},
            )
        # a training forward without the loop's draw is a broken step
        gate.train()
        with pytest.raises(RuntimeError, match="resample"):
            gate.featurize(torch.ones(1, 16))

    def test_a_loaded_budget_gate_needs_a_cut(self) -> None:
        from causalab.neural.shared.featurizers import Gate

        theta = torch.randn(8, generator=torch.Generator().manual_seed(0))
        with pytest.raises(ValueError, match="read out at 'top_k'"):
            Gate.from_theta(theta, parametrization="budget")
        loaded = Gate.from_theta(theta, parametrization="budget", top_k=3)
        assert loaded.k_schedule is None and loaded.eval_k() == 3
        assert int(loaded.hard_mask().sum()) == 3
        # soft mask at the cut: sums to the cut, orders like theta
        assert float(loaded.soft_mask().sum()) == pytest.approx(3.0, abs=1e-4)


@pytest.mark.unit
class TestForwardBackwardSplit:
    """§2.5 the mapping form: in training the mask's *value* is the map's
    training mask thresholded at ½ and its *gradient* is the map's; in eval
    nothing changes; the bundle stamps `forward` as provenance."""

    def test_the_split_does_not_occupy_nn_modules_forward_slot(self) -> None:
        """`forward` is `nn.Module`'s callable slot; the split is stored as
        `forward_mask`, so the slot holds torch's default on a plain gate and a
        split gate alike. No `Stage` defines `forward` — stages run through
        `featurize` / `inverse` — so calling a gate as a module raises torch's
        own `NotImplementedError`, not `TypeError: 'str' is not callable`; a
        `forward` added to a stage later would be reachable."""
        from causalab.neural.shared.featurizers import Gate

        for gate in (Gate(4), Gate(4, forward="hard")):
            with pytest.raises(NotImplementedError):  # torch's, not a str in the slot
                gate(torch.zeros(1, 4))
        assert Gate(4, forward="hard").forward_mask == "hard"
        assert Gate(4).forward_mask is None

    def test_the_training_mask_is_hard_with_the_soft_gradient(self) -> None:
        from causalab.neural.shared.featurizers import Gate

        theta0 = torch.tensor([2.0, -1.0, 0.5, -0.25])
        split, plain = Gate(4, forward="hard"), Gate(4)
        for gate in (split, plain):
            with torch.no_grad():
                gate.theta.copy_(theta0)
            gate.train(True)
        x = torch.ones(1, 4)
        kept_split, _ = split.featurize(x)
        kept_plain, _ = plain.featurize(x)
        assert kept_split.detach().tolist() == [[1.0, 0.0, 1.0, 0.0]]
        assert not torch.equal(kept_plain.detach(), kept_split.detach())
        kept_split.sum().backward()
        kept_plain.sum().backward()
        assert torch.allclose(split.theta.grad, plain.theta.grad)  # the soft gradient
        split.eval(), plain.eval()
        assert torch.equal(
            split.featurize(x)[0], plain.featurize(x)[0]
        )  # eval: the map's split
        assert split.identity_fields()["forward"] == "hard"
        assert "forward" not in plain.identity_fields()

    def test_the_split_follows_the_maps_own_threshold(self) -> None:
        from causalab.neural.shared.featurizers import Gate

        clamp = Gate(3, parametrization="clamp", forward="hard")
        with torch.no_grad():
            clamp.theta.copy_(torch.tensor([0.9, 0.5, 0.1]))
        clamp.train(True)
        assert clamp.featurize(torch.ones(1, 3))[0].detach().tolist() == [
            [1.0, 0.0, 0.0]
        ]
        with pytest.raises(ValueError, match="forward"):
            Gate(3, forward="soft")


@pytest.mark.unit
class TestPositionGate:
    """§2.5 ``axis: position``: θ runs over the window's positions and the
    mask broadcasts over every coordinate of a ``(rows, positions, width)``
    value; an all-on gate is the plain window; a value without a positions
    axis of that length is refused; the identity stamps the axis; a position
    gate composes with a grouped one as the outer product positions ⊗ groups
    in that order only; ``group`` and ``pool`` beside ``axis`` are refused at
    parse; a saved gate reloads as one; ``top_k`` counts positions, and the
    loader sizes and bounds a loaded gate in positions. ``init.from_scores``
    counts positions too — the ``keep`` bound offline (rule 32,
    ``tests/protocol/test_scores_init.py``) and the table's coverage at the
    build (``tests/neural/shared/test_scores_init.py``)."""

    def test_the_mask_broadcasts_along_the_position_axis(self) -> None:
        from causalab.neural.shared.featurizers import Gate

        gate = Gate(3, axis="position")
        with torch.no_grad():
            gate.theta.copy_(torch.tensor([5.0, -5.0, 5.0]))
        gate.eval()
        x = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
        kept, dropped = gate.featurize(x)
        assert kept.shape == x.shape
        assert torch.equal(kept[:, 0], x[:, 0]) and torch.equal(kept[:, 2], x[:, 2])
        assert torch.equal(kept[:, 1], torch.zeros_like(x[:, 1]))
        assert dropped is not None and torch.equal(kept + dropped, x)
        assert gate.hard_mask().tolist() == [1.0, 0.0, 1.0]
        assert gate.identity_fields()["axis"] == "position"
        assert "axis" not in Gate(3).identity_fields()

    def test_top_k_on_a_position_gate_keeps_the_highest_ranked_positions(self) -> None:
        """§2.5: `top_k` reads a loaded gate out at a count of **units**, which
        on a position gate are positions — the apply-side claim of the spec."""
        from causalab.neural.shared.featurizers import Gate

        gate = Gate.from_theta(torch.tensor([1.0, 3.0, 2.0]), axis="position", top_k=1)
        assert gate.hard_mask().tolist() == [0.0, 1.0, 0.0]
        # where "a unit is a position" shows: the kept value is the window
        # with every coordinate of the two other positions dropped
        gate.eval()
        x = torch.randn(2, 3, 5)
        kept, dropped = gate.featurize(x)
        assert torch.equal(kept[:, 1], x[:, 1]) and torch.equal(
            dropped[:, 1], torch.zeros(2, 5)
        )
        assert torch.equal(kept[:, [0, 2]], torch.zeros(2, 2, 5))
        assert torch.equal(dropped[:, [0, 2]], x[:, [0, 2]])

    def test_a_loaded_position_gate_is_sized_and_bounded_in_positions(self) -> None:
        """The two `_build_stage` messages this branch reworded count what θ
        counts: a four-position gate under a three-position window is refused
        naming positions and the window (not coordinates and the site), and a
        `top_k` beyond the window names positions. The sized twin builds."""
        from causalab.neural.shared.featurizers import build_stack
        from causalab.protocol.rules.errors import ProtocolError
        from causalab.protocol.schema import FeaturizerSpec

        def build(spec: FeaturizerSpec, theta: torch.Tensor) -> None:
            build_stack(
                "g",
                {"g": spec},
                width=8,
                position_width=3,
                load_tensors=_theta_bundle(theta, axis="position"),
                stage_cache={},
            )

        spec = FeaturizerSpec(
            kind="gate", file_path="fit/g.safetensors", axis="position"
        )
        with pytest.raises(
            ProtocolError, match="is 4 positions but the window here is 3"
        ):
            build(spec, torch.zeros(4))
        bounded = FeaturizerSpec(
            kind="gate", file_path="fit/g.safetensors", axis="position", top_k=5
        )
        with pytest.raises(ProtocolError, match="top_k=5 but the gate has 3 positions"):
            build(bounded, torch.zeros(3))
        build(spec, torch.zeros(3))

    def test_one_position_gate_over_two_windows_is_refused_at_the_build(self) -> None:
        """A document rule 4 never saw: the stack cache's `one featurizer, one
        width` counts the window and says so, before any forward — the guard a
        second window meets, `Gate.featurize`'s positions check being the
        backstop for a shape the sizing did not predict."""
        from causalab.neural.shared.featurizers import build_stack
        from causalab.protocol.rules.errors import ProtocolError
        from causalab.protocol.schema import FeaturizerSpec

        cache: dict = {}
        spec = FeaturizerSpec(
            kind="gate", file_path="fit/g.safetensors", axis="position"
        )
        kw = dict(width=8, load_tensors=_theta_bundle(torch.zeros(3), axis="position"))
        build_stack("g", {"g": spec}, position_width=3, stage_cache=cache, **kw)
        with pytest.raises(
            ProtocolError,
            match="used at width 4 here but was built for width 3 — one featurizer, "
            r"one width \(a position gate's width is its window's length\)",
        ):
            build_stack("g", {"g": spec}, position_width=4, stage_cache=cache, **kw)

    def test_an_all_on_position_gate_is_the_plain_window(self) -> None:
        from causalab.neural.shared.featurizers import Gate

        gate = Gate(3, axis="position", init=0.999)
        gate.eval()
        x = torch.randn(2, 3, 5)
        kept, dropped = gate.featurize(x)
        assert torch.equal(kept, x) and torch.equal(dropped, torch.zeros_like(x))

    def test_a_value_without_the_window_is_refused(self) -> None:
        from causalab.neural.shared.featurizers import Gate
        from causalab.protocol.rules.errors import ProtocolError

        gate = Gate(3, axis="position")
        with pytest.raises(ProtocolError, match="positions axis"):
            gate.featurize(torch.zeros(2, 4))  # a single position, no window
        with pytest.raises(ProtocolError, match="positions axis"):
            gate.featurize(torch.zeros(2, 5, 4))  # another window length

    def test_a_position_gate_takes_no_group_and_no_pool(self) -> None:
        from causalab.neural.shared.featurizers import Gate

        with pytest.raises(ValueError, match="no group"):
            Gate(4, axis="position", group="site", groups=(1, 4))
        with pytest.raises(ValueError, match="no pool"):
            Gate(
                4,
                axis="position",
                parametrization="budget",
                k_schedule={"kind": "fixed", "k": 1},
                pool="p",
            )
        with pytest.raises(ValueError, match="axis"):
            Gate(4, axis="feature")

    def test_a_position_gate_before_a_feature_gate_is_the_outer_product(self) -> None:
        """The headline use: `["posgate", "gate"]` — the position mask and
        the feature mask multiply, so the per-position neuron mask is their
        outer product, tied across positions instead of one gate per
        position; the feature width flows through the position gate."""
        from causalab.neural.shared.featurizers import FeaturizerStack, Gate

        pos, feat = Gate(3, axis="position"), Gate(4)
        with torch.no_grad():
            pos.theta.copy_(torch.tensor([5.0, -5.0, 5.0]))
            feat.theta.copy_(torch.tensor([5.0, 5.0, -5.0, -5.0]))
        pos.eval()
        feat.eval()
        x = torch.ones(2, 3, 4)
        kept, _ = FeaturizerStack(names=("pos", "feat"), stages=(pos, feat)).featurize(
            x
        )
        outer = pos.hard_mask()[:, None] * feat.hard_mask()[None, :]
        assert kept.shape == x.shape
        assert torch.equal(kept[0], outer) and torch.equal(kept[1], outer)
        assert outer.sum() == 4  # two kept positions × two kept coordinates

    def test_a_head_grouped_gate_before_a_position_gate_is_positions_by_heads(
        self,
    ) -> None:
        """The order rule 23 permits, `["gate", "posgate"]` with `gate` grouped
        by head: two heads of four coordinates, three positions — the mask is
        the outer product of the head mask (spread over each head's
        coordinates) and the position mask: positions ⊗ heads."""
        from causalab.neural.shared.featurizers import FeaturizerStack, Gate

        feat, pos = Gate(8, group="head", groups=(2, 4)), Gate(3, axis="position")
        with torch.no_grad():
            feat.theta.copy_(torch.tensor([5.0, -5.0]))
            pos.theta.copy_(torch.tensor([5.0, -5.0, 5.0]))
        feat.eval()
        pos.eval()
        x = torch.ones(2, 3, 8)
        kept, _ = FeaturizerStack(names=("feat", "pos"), stages=(feat, pos)).featurize(
            x
        )
        heads = feat.hard_mask().repeat_interleave(4)  # each head's four coordinates
        outer = pos.hard_mask()[:, None] * heads[None, :]
        assert kept.shape == x.shape
        assert torch.equal(kept[0], outer) and torch.equal(kept[1], outer)
        assert outer.sum() == 8  # two kept positions × one kept head of four

    def test_a_saved_position_gate_reloads_only_under_a_document_that_spells_it(
        self,
    ) -> None:
        """The identity stamp, both directions: a bundle fitted over
        positions refuses a document without `axis`, and an unstamped (or
        coordinate-fitted) bundle refuses a document with it."""
        from causalab.neural.shared.featurizers.build import (
            _check_entry_identity,  # pyright: ignore[reportPrivateUsage]
        )
        from causalab.protocol.rules.errors import ProtocolError
        from causalab.protocol.schema import parse_document
        from tests.protocol._docs import base_doc, in_order

        def spec(**fields):
            raw = base_doc()
            raw["method"]["featurizers"] = {"g": {"kind": "gate", **fields}}
            raw["method"]["writes"]["patch"]["featurizer"] = "g"
            return parse_document(in_order(raw)).featurizers["g"]

        positional, plain = spec(axis="position"), spec()
        stamped = {"axis": "position", "parametrization": "sigmoid"}
        unstamped = {"parametrization": "sigmoid"}
        _check_entry_identity(stamped, positional, "g")
        _check_entry_identity(unstamped, plain, "g")
        with pytest.raises(ProtocolError, match="mask over positions"):
            _check_entry_identity(stamped, plain, "g")
        with pytest.raises(ProtocolError, match="mask over positions"):
            _check_entry_identity(unstamped, positional, "g")


@pytest.mark.unit
class TestBudgetPool:
    """§2.5 ``pool``: several budget gates, one budget. The identities that
    make a pool *the same object* as one gate over the concatenation of its
    members' θ — one shift, one sum, one ranking, one cut — and the two ways a
    schedule's ``of`` counts."""

    @staticmethod
    def _pool(schedule: dict, widths=(7, 5), seed: int = 0, **kw):
        from causalab.neural.shared.featurizers import BudgetPool, Gate

        gen = torch.Generator().manual_seed(seed)
        members = []
        for width in widths:
            gate = Gate(width, parametrization="budget", k_schedule=schedule, **kw)
            with torch.no_grad():
                gate.theta.copy_(torch.randn(width, generator=gen))
            members.append(gate)
        pool = BudgetPool("p", members)
        for gate in members:
            gate.pool = pool
        return pool, members

    def test_the_pooled_shift_makes_the_pooled_mask_sum_to_the_budget(self) -> None:
        pool, members = self._pool({"kind": "fixed", "k": 4})
        for k in (1, 4, 11):
            masks = [gate.budget_mask(k) for gate in members]
            assert float(sum(m.sum() for m in masks)) == pytest.approx(k, abs=1e-4), k
            # one scalar shift, every member's mask is σ(θ_m + c)
            shift = pool.shift(k)
            for gate, mask in zip(members, masks):
                assert torch.allclose(mask, torch.sigmoid(gate.theta + shift))
        assert pool.units == 12

    def test_a_pool_is_one_gate_over_the_concatenation(self) -> None:
        """The cut and the training mask of the members, side by side, equal a
        single budget gate whose θ is the members' θ concatenated."""
        from causalab.neural.shared.featurizers import Gate

        pool, members = self._pool({"kind": "fixed", "k": 4})
        whole = Gate.from_theta(
            pool.theta().detach(), parametrization="budget", top_k=4
        )
        whole.eval()
        hard = torch.cat([gate.hard_mask() for gate in members])
        assert torch.equal(hard, whole.hard_mask())
        assert int(hard.sum()) == 4
        # the training mask, with its implicit gradient
        alone = Gate(12, parametrization="budget", k_schedule={"kind": "fixed", "k": 4})
        with torch.no_grad():
            alone.theta.copy_(pool.theta())
        soft = torch.cat([gate.budget_mask(4) for gate in members])
        assert torch.allclose(soft, alone.budget_mask(4), atol=1e-6)
        # Σ over the POOL is a constant of every member's θ to first order
        total = sum(gate.budget_mask(4).sum() for gate in members)
        grads = torch.autograd.grad(total, [gate.theta for gate in members])
        assert all(float(g.abs().max()) < 1e-5 for g in grads)
        # the pooled ranking is a permutation, split back per member
        ranks = torch.cat([pool.member_rank(gate) for gate in members])
        assert sorted(ranks.tolist()) == list(range(12))

    def test_a_kept_schedule_is_the_complement_of_a_patched_one(self) -> None:
        """``of: kept`` with ``fixed k`` and ``of: patched`` with ``fixed N − k``
        are one gate: the same draw, the same shift, the same masks — on a lone
        gate and on a pool."""
        from causalab.neural.shared.featurizers import Gate

        gen = torch.Generator().manual_seed(3)
        theta = torch.randn(9, generator=gen)
        kept = Gate(
            9,
            parametrization="budget",
            k_schedule={"kind": "fixed", "k": 2, "of": "kept"},
        )
        patched = Gate(
            9, parametrization="budget", k_schedule={"kind": "fixed", "k": 7}
        )
        for gate in (kept, patched):
            with torch.no_grad():
                gate.theta.copy_(theta)
            gate.train()
            gate.resample(torch.Generator().manual_seed(0))
        assert kept._k == patched._k == 7
        assert kept.eval_k() == patched.eval_k() == 7
        assert torch.equal(kept.budget_mask(kept._k), patched.budget_mask(patched._k))
        kept.eval(), patched.eval()
        assert torch.equal(kept.hard_mask(), patched.hard_mask())
        assert int(kept.hard_mask().sum()) == 7  # patched units, as every reader counts
        # sampled: a kept draw k' is the patched draw N − k' of the same u
        lo = {"kind": "log_uniform", "low": 1, "high": 8, "eval": 3}
        a, b = (
            Gate(9, parametrization="budget", k_schedule={**lo, "of": "kept"}),
            Gate(9, parametrization="budget", k_schedule=lo),
        )
        a.resample(torch.Generator().manual_seed(5))
        b.resample(torch.Generator().manual_seed(5))
        assert a._k == 9 - b._k and a.eval_k() == 9 - b.eval_k()
        # …and on a pool, complemented against the POOL's units
        pool, _ = self._pool({"kind": "fixed", "k": 2, "of": "kept"})
        pool.resample(torch.Generator().manual_seed(0))
        assert pool.k == pool.eval_k() == 12 - 2

    def test_a_pooled_member_draws_no_budget_of_its_own(self) -> None:
        pool, members = self._pool(
            {"kind": "log_uniform", "low": 1, "high": 11, "eval": 3}
        )
        with pytest.raises(RuntimeError, match="resamples the pool"):
            members[0].resample(torch.Generator().manual_seed(0))
        members[0].train()
        with pytest.raises(RuntimeError, match="no budget for this step"):
            members[0].featurize(torch.ones(1, 7))
        pool.resample(torch.Generator().manual_seed(0))
        assert 1 <= pool.k <= 11
        kept, rest = members[0].featurize(torch.ones(1, 7))
        assert torch.allclose(kept + rest, torch.ones(1, 7))
        # the pool's draw is the lone gate's draw at the same seed: one scalar
        from causalab.neural.shared.featurizers import Gate

        alone = Gate(12, parametrization="budget", k_schedule=pool._schedule())
        alone.resample(torch.Generator().manual_seed(0))
        assert alone._k == pool.k

    def test_the_link_builds_missing_members_and_refuses_a_non_pool(self) -> None:
        from causalab.neural.shared.featurizers import Gate, link_budget_pools
        from causalab.protocol.rules.errors import ProtocolError
        from causalab.protocol.schema import FeaturizerSpec

        def spec(**kw) -> FeaturizerSpec:
            return FeaturizerSpec(kind="gate", parametrization="budget", pool="p", **kw)

        schedule = {"kind": "fixed", "k": 3}
        specs = {"a": spec(k_schedule=schedule), "b": spec(k_schedule=schedule)}
        stages: dict = {}

        def build(name: str) -> Gate:
            stages[name] = Gate(
                4, parametrization="budget", k_schedule=schedule, pool="p"
            )
            return stages[name]

        link_budget_pools(specs, stages, build)
        assert set(stages) == {"a", "b"}  # `b` was built by the link
        assert stages["a"].pool is stages["b"].pool
        assert stages["a"].pool.units == 8
        link_budget_pools(specs, stages, build)  # idempotent
        assert stages["a"].pool is stages["b"].pool
        # members that disagree on the schedule are not a pool
        stages2 = {
            "a": Gate(4, parametrization="budget", k_schedule=schedule, pool="p"),
            "b": Gate(
                4,
                parametrization="budget",
                k_schedule={"kind": "fixed", "k": 2},
                pool="p",
            ),
        }
        with pytest.raises(ProtocolError, match="disagree"):
            link_budget_pools(specs, stages2, build)
        # a schedule number above the pool's units
        big = {"kind": "fixed", "k": 9}
        stages3 = {
            n: Gate(4, parametrization="budget", k_schedule=big, pool="p") for n in "ab"
        }
        with pytest.raises(ProtocolError, match="pool has 8 units"):
            link_budget_pools(
                {"a": spec(k_schedule=big), "b": spec(k_schedule=big)}, stages3, build
            )
        # loaded members read out at one pooled cut, or refused
        theta = torch.randn(4, generator=torch.Generator().manual_seed(0))
        loaded = {
            n: Gate.from_theta(theta, parametrization="budget", top_k=k, pool="p")
            for n, k in (("a", 3), ("b", 5))
        }
        with pytest.raises(ProtocolError, match="disagree"):
            link_budget_pools({"a": spec(top_k=3), "b": spec(top_k=5)}, loaded, build)
        # a pooled cut above the pool's units, and a lone gate's own bound
        too_many = {
            n: Gate.from_theta(theta, parametrization="budget", top_k=9, pool="p")
            for n in "ab"
        }
        with pytest.raises(ProtocolError, match="pool has 8 units"):
            link_budget_pools({"a": spec(top_k=9), "b": spec(top_k=9)}, too_many, build)
        with pytest.raises(ValueError, match="top_k=5 on a gate of 4 units"):
            Gate.from_theta(theta, parametrization="budget", top_k=5)
        same = {
            n: Gate.from_theta(theta, parametrization="budget", top_k=3, pool="p")
            for n in "ab"
        }
        link_budget_pools({"a": spec(top_k=3), "b": spec(top_k=3)}, same, build)
        assert same["a"].pool.eval_k() == 3
        assert int(sum(g.hard_mask().sum() for g in same.values())) == 3
        assert same["a"].identity_fields() == {"pool": "p", "pool_units": "8"}
        # a pooled READOUT of sigmoid gates: the joint top-3, not θ > 0 per gate
        sig = {
            n: Gate.from_theta(t, top_k=3, pool="p")
            for n, t in (("a", theta), ("b", theta + 10.0))
        }
        for g in sig.values():
            g.eval()
        readout_specs = {
            "a": FeaturizerSpec(kind="gate", pool="p", top_k=3, file_path="a.st"),
            "b": FeaturizerSpec(kind="gate", pool="p", top_k=3, file_path="b.st"),
        }
        link_budget_pools(readout_specs, sig, build)
        assert int(sig["a"].hard_mask().sum()) == 0  # b's θ are all larger
        assert int(sig["b"].hard_mask().sum()) == 3
        with pytest.raises(ValueError, match="pooled 'top_k'"):
            Gate.from_theta(theta, pool="p")

    def test_an_unlinked_pooled_gate_refuses_to_be_read(self) -> None:
        """`pool_name` has a job: a gate that authors a pool but never reached
        the link must not fall back to a lone gate's budget or cut."""
        from causalab.neural.shared.featurizers import Gate

        gate = Gate(
            4, parametrization="budget", k_schedule={"kind": "fixed", "k": 1}, pool="p"
        )
        for call in (
            lambda: gate.resample(torch.Generator().manual_seed(0)),
            gate.eval_k,
            gate.hard_mask,
            gate.identity_fields,
            lambda: gate.featurize(torch.ones(1, 4)),
        ):
            with pytest.raises(RuntimeError, match="never linked"):
                call()
        loaded = Gate.from_theta(torch.randn(4), top_k=2, pool="p")
        loaded.eval()
        with pytest.raises(RuntimeError, match="never linked"):
            loaded.featurize(torch.ones(1, 4))

    def test_the_pool_memoizes_per_theta_version(self) -> None:
        """One solve and one argsort per forward for M members; an optimizer
        step (an in-place θ update) invalidates both."""
        pool, members = self._pool({"kind": "fixed", "k": 4})
        s1 = pool.shift(4)
        assert pool.shift(4) is s1  # memoized: the same tensor object
        r1 = pool.pooled_rank()
        assert pool.pooled_rank() is r1
        masks = [g.budget_mask(4) for g in members]
        assert pool.shift(4) is s1  # the members' solves shared it
        with torch.no_grad():
            members[0].theta.add_(1.0)  # bumps theta._version
        s2 = pool.shift(4)
        assert s2 is not s1 and float(s2) != float(s1)
        assert float(sum(g.budget_mask(4).sum() for g in members)) == pytest.approx(
            4, abs=1e-4
        )
        assert masks[0].shape == members[0].theta.shape

    def test_the_link_is_flat_and_refuses_mixed_maps(self) -> None:
        """A member's build re-enters the link; the guard returns at once, so
        the depth is 2 whatever M is — and a sigmoid beside a clamp is refused."""
        from causalab.neural.shared.featurizers import Gate, link_budget_pools
        from causalab.protocol.rules.errors import ProtocolError
        from causalab.protocol.schema import FeaturizerSpec

        names = [f"g{i}" for i in range(12)]
        specs = {
            n: FeaturizerSpec(kind="gate", top_k=3, pool="p", file_path=f"{n}.st")
            for n in names
        }
        stages: dict = {}
        depth = {"now": 0, "max": 0}

        def build(name: str):
            depth["now"] += 1
            depth["max"] = max(depth["max"], depth["now"])
            stages[name] = Gate.from_theta(torch.randn(3), top_k=3, pool="p")
            link_budget_pools(specs, stages, build)  # what executor.stage() does
            depth["now"] -= 1
            return stages[name]

        link_budget_pools(specs, stages, build)
        assert len(stages) == 12 and len({id(g.pool) for g in stages.values()}) == 1
        assert depth["max"] == 1  # never nested: the re-entered link returned
        mixed = {
            "a": Gate.from_theta(
                torch.rand(3), top_k=2, pool="q", parametrization="clamp"
            ),
            "b": Gate.from_theta(torch.randn(3), top_k=2, pool="q"),
        }
        mixed_specs = {
            "a": FeaturizerSpec(
                kind="gate",
                parametrization="clamp",
                top_k=2,
                pool="q",
                file_path="a.st",
            ),
            "b": FeaturizerSpec(kind="gate", top_k=2, pool="q", file_path="b.st"),
        }
        with pytest.raises(ProtocolError, match="parametrization"):
            link_budget_pools(mixed_specs, mixed, lambda n: mixed[n])


@pytest.mark.unit
class TestBoundaryGate:
    """§2.5 ``parametrization: boundary`` — Boundless DAS (Wu et al. 2023): one
    θ in ``[0, 1]``, the boundary as a fraction of the width (``β = θ·width``),
    over the coordinate index of an ordered input. Soft ``σ((β − i)/T)``, hard
    ``i < β`` (the soft mask at ½, as under every map), θ projected into
    ``[0, 1]``; nothing per unit exists."""

    def _gate(self, width: int, beta: float, **kw):
        """A boundary gate at β coordinates (θ = β / width)."""
        from causalab.neural.shared.featurizers import Gate

        return Gate(
            width, parametrization="boundary", init=torch.tensor([beta / width]), **kw
        )

    def test_the_soft_mask_is_a_sigmoid_over_the_coordinate_index(self) -> None:
        gate = self._gate(5, 2.5)
        assert gate.theta.shape == (1,)
        logits = torch.tensor([2.5, 1.5, 0.5, -0.5, -1.5])
        assert torch.allclose(gate.soft_mask(), torch.sigmoid(logits))
        gate.temperature = 0.5  # the anneal target, the paper's T
        assert torch.allclose(gate.soft_mask(), torch.sigmoid(logits / 0.5))
        # the kept fraction the l1 term reads: ⌈β⌉ / width as T → 0
        sharp = self._gate(10, 3.5)
        sharp.temperature = 0.01
        assert float(sharp.soft_mask().mean()) == pytest.approx(0.4, abs=1e-6)

    def test_the_hard_mask_is_the_prefix_below_beta(self) -> None:
        gate = self._gate(5, 2.5)
        assert gate.hard_mask().tolist() == [1.0, 1.0, 1.0, 0.0, 0.0]
        with torch.no_grad():
            # the soft mask at ½: a β just past an integer keeps one more
            gate.theta.fill_(0.4)  # β = 2.0
            assert gate.hard_mask().tolist() == [1.0, 1.0, 0.0, 0.0, 0.0]
            gate.theta.fill_(0.42)  # β = 2.1
            assert gate.hard_mask().tolist() == [1.0, 1.0, 1.0, 0.0, 0.0]
            gate.theta.fill_(0.0)
            assert gate.hard_mask().sum() == 0
            gate.theta.fill_(1.0)  # β = width
        # β = width is the ungated rotation: featurize is the identity
        gate.eval()
        x = torch.randn(3, 5)
        f, err = gate.featurize(x)
        assert (
            torch.equal(f, x)
            and err is not None
            and torch.equal(err, torch.zeros_like(x))
        )

    def test_fill_is_a_kept_fraction_and_the_default_start_is_the_half_prefix(
        self,
    ) -> None:
        from causalab.neural.shared.featurizers import Gate

        half = Gate(8, parametrization="boundary")
        assert half.theta.item() == 0.5 and half.boundary() == 4.0
        quarter = Gate(8, parametrization="boundary", init=0.25)
        assert quarter.theta.item() == 0.25 and quarter.init_fill == 0.25
        assert quarter.boundary() == 2.0 and float(quarter.hard_mask().sum()) == 2.0
        # the poles are legal starts: β = 0 and β = width are real prefixes
        assert Gate(8, parametrization="boundary", init=1.0).boundary() == 8.0
        assert Gate(8, parametrization="boundary", init=0.0).boundary() == 0.0

    def test_project_clips_theta_into_the_unit_interval(self) -> None:
        gate = self._gate(4, 2.0)
        with torch.no_grad():
            gate.theta.fill_(9.0)
        gate.project()
        assert gate.theta.item() == 1.0 and gate.boundary() == 4.0
        with torch.no_grad():
            gate.theta.fill_(-3.0)
        gate.project()
        assert gate.theta.item() == 0.0
        assert gate.dead_diagnostics() == {}  # no unit to freeze or reawaken

    def test_the_hard_forward_keeps_the_prefix_with_the_soft_gradient(self) -> None:
        """§2.5 the mapping form: ``forward: hard`` is ``mask > ½``, which is
        ``i < β`` — the eval split itself — and the backward is the soft
        mask's, ``width · Σ_i σ'((β − i)/T) / T`` (β = θ·width)."""
        gate = self._gate(5, 2.5, forward="hard")
        gate.train()
        f, _ = gate.featurize(torch.ones(1, 5))
        assert f.detach().tolist() == [[1.0, 1.0, 1.0, 0.0, 0.0]]
        f.sum().backward()
        s = torch.sigmoid(torch.tensor([2.5, 1.5, 0.5, -0.5, -1.5]))
        assert gate.theta.grad is not None
        assert float(gate.theta.grad) == pytest.approx(5 * float((s * (1 - s)).sum()))

    def test_every_per_unit_reader_and_field_refuses_by_name(self) -> None:
        from causalab.neural.shared.featurizers import Gate

        gate = self._gate(4, 2.0)
        with pytest.raises(ValueError, match="no per-unit threshold"):
            gate.hard_threshold()
        with pytest.raises(ValueError, match="no ranking"):
            gate.ranking()
        with pytest.raises(ValueError, match="no ranking"):
            gate.rank()
        with pytest.raises(ValueError, match="takes no group"):
            Gate(4, parametrization="boundary", group="head", groups=(2, 2))
        with pytest.raises(ValueError, match="takes no axis"):
            Gate(4, parametrization="boundary", axis="position")
        with pytest.raises(ValueError, match="no dead rule"):
            Gate(4, parametrization="boundary", dead={"leak": 0.1})
        with pytest.raises(ValueError, match="samples nothing"):
            Gate(4, parametrization="boundary", stretch=(-0.1, 1.1))
        with pytest.raises(ValueError, match="joins no pool"):
            Gate(4, parametrization="boundary", pool="p")
        with pytest.raises(ValueError, match="positive"):
            Gate(4, parametrization="boundary", temperature=0.0)
        with pytest.raises(ValueError, match="no ranking"):
            Gate.from_theta(
                torch.tensor([0.5]), width=4, parametrization="boundary", top_k=2
            )
        with pytest.raises(ValueError, match="needs the site width"):
            Gate.from_theta(torch.tensor([0.5]), parametrization="boundary")
        with pytest.raises(ValueError, match="needs 1 parameters"):
            Gate.from_theta(torch.zeros(4), width=4, parametrization="boundary")

    def test_a_loaded_boundary_gate_reproduces_the_mask_and_reloads_only_as_one(
        self,
    ) -> None:
        from causalab.neural.shared.featurizers import Gate
        from causalab.neural.shared.featurizers.build import (
            _check_entry_identity,  # pyright: ignore[reportPrivateUsage]
        )
        from causalab.protocol.rules.errors import ProtocolError
        from causalab.protocol.schema import FeaturizerSpec

        fitted = self._gate(6, 3.7)
        fitted.eval()
        loaded = Gate.from_theta(
            fitted.theta.detach(), width=6, parametrization="boundary"
        )
        loaded.eval()
        assert torch.equal(loaded.hard_mask(), fitted.hard_mask())
        assert float(loaded.hard_mask().sum()) == 4.0
        assert loaded.boundary() == pytest.approx(3.7)
        assert not loaded.theta.requires_grad
        # rule 15: `parametrization` is stamped, so a boundary bundle reloads
        # only under `boundary`, and a per-unit bundle never as a boundary
        plain, bnd = (
            FeaturizerSpec(kind="gate"),
            FeaturizerSpec(kind="gate", parametrization="boundary"),
        )
        _check_entry_identity({"parametrization": "boundary"}, bnd, "w")
        with pytest.raises(ProtocolError, match="fitted 'boundary'"):
            _check_entry_identity({"parametrization": "boundary"}, plain, "w")
        with pytest.raises(ProtocolError, match="parametrization='sigmoid'"):
            _check_entry_identity({"parametrization": "sigmoid"}, bnd, "w")
