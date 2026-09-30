"""Featurizers evaluated once per step ([`featurizer_cache`][causalab.neural.shared.featurizers.sharing.featurizer_cache]), and the
Cayley map's launch diet — both without a change of result.

The finding this answers (profiling campaign ``fullprof0910``, the DAS step
of the standard A3B workflow): the ``cayley`` parametrization was the single
largest issuer of CUDA launches in an optimizer step — the rotation is
recomputed on **every** ``.weight`` access, and a member's featurizer is
accessed for its read, and twice per write (featurize and inverse), on every
hooked layer; each evaluation also ran ``torch.linalg.inv``, whose error
check is a hidden host synchronization.

Two contracts, pinned separately because they fail separately:

* **the map's numbers are unchanged** — the rewritten [`Cayley.forward`][causalab.neural.shared.featurizers.stages.Cayley.forward]
  is compared *bit for bit*, forward and backward, against the previous
  spelling (kept verbatim here as the oracle) over the property test's draws,
  including the zero ``original`` a fit starts from and the collinear columns
  that are the map's hard case. A fit and its reloaded artifact must featurize
  bit-identically, and fp32 goldens compare to four decimals, so an ulp is a
  real change;
* **the cache changes nothing but the count** — inside a scope a stage's
  derived quantity (a subspace's ``Q``, a gate's mask) is computed once and
  serves every access, so the forward is bit-identical to the uncached one
  by construction. The gradient is bit-identical too, and that takes care:
  one tensor for every consumer would make autograd sum the consumers'
  cotangents at ``Q`` and run the map's backward once — ``Jᵀ(Σcᵢ)`` for
  ``Σ Jᵀcᵢ``, roundoff a bf16 model amplifies into a different fit — so a
  trained ``cayley`` map shares its forward and *replays* its backward per
  access, handing the parameter the same contributions in the same order a
  map per access does, and every other trainable quantity is recomputed on
  a grad access and shared under ``no_grad`` alone.
"""

from __future__ import annotations

import contextlib
from typing import Any

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from torch.utils._python_dispatch import TorchDispatchMode

from causalab.neural.shared.featurizers import (
    Cayley,
    Gate,
    Subspace,
    featurizer_cache,
)
from causalab.neural.shared.featurizers.sharing import (
    _leaf_edges,  # pyright: ignore[reportPrivateUsage]
)

TRAINED_KEY = "parametrizations.weight.original"


class _OpCounter(TorchDispatchMode):
    """Every aten op dispatched inside the block, by overload-packet name."""

    def __init__(self) -> None:
        super().__init__()
        self.counts: dict[str, int] = {}

    def __torch_dispatch__(
        self, func: Any, types: Any, args: Any = (), kwargs: Any = None
    ) -> Any:
        name = str(func.overloadpacket).split(".")[-1]
        self.counts[name] = self.counts.get(name, 0) + 1
        return func(*args, **(kwargs or {}))

    def total(self) -> int:
        return sum(self.counts.values())


def _reference_cayley(x: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
    """The map as it was spelled before this change — ``diag_embed`` and
    k×k matmuls for the scalings, a fresh ``eye``, and ``torch.linalg.inv``
    with its host error check. The oracle every bit-identity claim is
    against."""
    in_frame = base.mT @ x
    omega = in_frame - in_frame.mT
    perp = x - base @ in_frame
    scale = torch.linalg.vector_norm(perp.detach(), dim=-2).clamp(min=1.0)
    perp = perp / scale
    inv_scale = torch.diag_embed(1.0 / scale)
    schur = (
        perp.mT @ perp
        - 2.0 * inv_scale @ omega @ inv_scale
        + 4.0 * inv_scale @ inv_scale
    )
    inverse = torch.linalg.inv(schur)
    v1 = inverse @ (-2.0 * inv_scale)
    v2 = torch.eye(base.shape[-1], dtype=x.dtype, device=x.device) + (
        2.0 * inv_scale @ v1
    )
    return base - 2.0 * (perp @ v1 + base @ v2)


def _subspace(
    width: int, k: int, parametrization: str = "cayley", seed: int = 0
) -> Subspace:
    return Subspace(width, k, parametrization, seed=seed)


def _original(stage: Subspace) -> torch.nn.Parameter:
    return dict(stage.named_parameters())[TRAINED_KEY]  # type: ignore[return-value]


def _cayley(stage: Subspace) -> Cayley:
    """The parametrization module a ``cayley`` stage computes its weight
    through."""
    module = dict(stage.named_modules())["parametrizations.weight.0"]
    assert isinstance(module, Cayley)
    return module


def _set_original(stage: Subspace, value: torch.Tensor) -> None:
    with torch.no_grad():
        _original(stage).copy_(value)


def _draw(width: int, k: int, seed: int, scale: float, collinear: bool) -> torch.Tensor:
    x = scale * torch.randn(width, k, generator=torch.Generator().manual_seed(seed))
    if collinear and k > 1:
        x[:, 1] = x[:, 0]
    return x


# --------------------------------------------------------------------- #
# the map's numbers are unchanged
# --------------------------------------------------------------------- #


@pytest.mark.property
class TestCayleyMapIsBitIdentical:
    @given(
        k=st.integers(min_value=1, max_value=6),
        extra=st.integers(min_value=0, max_value=40),
        seed=st.integers(min_value=0, max_value=2**16),
        scale=st.sampled_from([0.0, 0.01, 0.3, 3.0, 30.0]),
        collinear=st.booleans(),
    )
    @settings(max_examples=80, deadline=None)
    def test_forward_and_backward_match_the_previous_spelling_bit_for_bit(
        self, k: int, extra: int, seed: int, scale: float, collinear: bool
    ) -> None:
        """The ``eye`` became a buffer and the inverse ``inv_ex`` without
        its error check: the same arithmetic in the same order, so the
        weight — and the gradient through it — is the same tensor to the
        bit. ``scale = 0`` is the zero ``original`` a fit starts from.

        What this guards against is real: spelling the products with the
        diagonal ``D⁻¹`` as broadcast scalings keeps the forward bit-identical
        and moves the gradient by an ulp — the autograd engine then
        accumulates ``X̃``'s three contributions in another order — which is
        why that rewrite is *not* in the map."""
        width = k + extra
        stage = _subspace(width, k, seed=seed)
        cayley = _cayley(stage)
        x = _draw(width, k, seed, scale, collinear)
        cotangent = torch.randn(
            width, k, generator=torch.Generator().manual_seed(seed + 1)
        )

        x_new = x.clone().requires_grad_(True)
        q_new = cayley(x_new)
        (q_new * cotangent).sum().backward()

        x_ref = x.clone().requires_grad_(True)
        q_ref = _reference_cayley(x_ref, cayley.base)
        (q_ref * cotangent).sum().backward()

        assert torch.equal(q_new, q_ref), "the rewritten map moved the weight"
        assert x_new.grad is not None and x_ref.grad is not None
        assert torch.equal(x_new.grad, x_ref.grad), (
            "the rewritten map moved the gradient"
        )

    def test_the_inverse_skips_the_host_error_check(self) -> None:
        """``torch.linalg.inv`` is ``inv_ex`` followed by
        ``_linalg_check_errors``, a device-to-host copy of the info tensor on
        every call; the Schur system is nonsingular by construction
        ([`Cayley`][causalab.neural.shared.featurizers.stages.Cayley]), so the check is dropped unconditionally — not only
        inside CUDA-graph capture, where it used to be."""
        stage = _subspace(32, 4)
        _set_original(stage, _draw(32, 4, 1, 0.3, False))
        with _OpCounter() as ops:
            stage.weight
        assert ops.counts.get("linalg_inv_ex", 0) == 1
        assert ops.counts.get("_linalg_check_errors", 0) == 0
        # non-vacuous: the previous spelling did check
        with _OpCounter() as ref:
            _reference_cayley(_original(stage).detach(), _cayley(stage).base)
        assert ref.counts.get("_linalg_check_errors", 0) == 1

    def test_the_map_issues_fewer_ops_than_before(self) -> None:
        """The launch count is the whole point; a rewrite that kept the bits
        but not the diet would pass everything else here."""
        stage = _subspace(64, 8)
        _set_original(stage, _draw(64, 8, 2, 0.3, False))
        with _OpCounter() as new:
            stage.weight
        with _OpCounter() as old:
            _reference_cayley(_original(stage).detach(), _cayley(stage).base)
        assert new.total() < old.total(), (new.counts, old.counts)
        assert new.counts.get("eye", 0) == 0, (
            "the identity is a buffer, not a per-call op"
        )

    def test_the_identity_buffer_is_not_saved_state(self) -> None:
        """A fit's artifact and the train loop's snapshot both go through
        ``state_dict``; the constant ``eye`` must not appear there, or every
        saved rotation would grow a k×k tensor and every reload would look
        for one."""
        stage = _subspace(16, 4)
        assert set(stage.state_dict()) == {
            TRAINED_KEY,
            "parametrizations.weight.0.base",
        }
        moved = _subspace(16, 4).to(torch.float64)
        assert _cayley(moved).eye.dtype == torch.float64, (
            "the buffer follows the module"
        )


# --------------------------------------------------------------------- #
# the cache changes nothing but the count
# --------------------------------------------------------------------- #


def _uses(stage: Subspace, x: torch.Tensor, f: torch.Tensor) -> list[torch.Tensor]:
    """The access pattern of one member in one step: a read's featurize, a
    write's featurize and inverse, twice over, and the regularizer's read of
    the weight."""
    out: list[torch.Tensor] = []
    for _ in range(2):
        feat, err = stage.featurize(x)
        out.extend([feat, err])  # type: ignore[list-item]
        out.append(stage.inverse(f, err))
    out.append(stage.slot_params()["weight"].abs().sum().reshape(1))
    return out


def _fit_step(
    stage: Subspace, x: torch.Tensor, f: torch.Tensor, scope: Any
) -> tuple[list[torch.Tensor], torch.Tensor, _OpCounter]:
    """One step's forward and backward over `_uses`; the outputs, the
    parameter's gradient and the ops the backward issued."""
    with scope:
        outputs = _uses(stage, x, f)
        loss = torch.stack([(t * t).sum() for t in outputs]).sum()
        with _OpCounter() as backward:
            loss.backward()
    grad = _original(stage).grad
    assert grad is not None
    return [t.detach().clone() for t in outputs], grad.clone(), backward


@pytest.mark.unit
class TestSubspaceCache:
    def test_a_scope_evaluates_the_map_once_and_the_results_are_the_same(self) -> None:
        stage = _subspace(32, 4)
        _set_original(stage, _draw(32, 4, 3, 0.3, False))
        x = torch.randn(5, 32, generator=torch.Generator().manual_seed(4))
        f = torch.randn(5, 4, generator=torch.Generator().manual_seed(5))

        with _OpCounter() as plain:
            uncached = _uses(stage, x, f)
        with featurizer_cache(), _OpCounter() as scoped:
            cached = _uses(stage, x, f)

        assert plain.counts["linalg_inv_ex"] == 5, "five accesses, five maps — uncached"
        assert scoped.counts["linalg_inv_ex"] == 1, "five accesses, one map — cached"
        for a, b in zip(uncached, cached, strict=True):
            assert torch.equal(a, b)

    @given(
        k=st.integers(min_value=1, max_value=6),
        extra=st.integers(min_value=0, max_value=30),
        seed=st.integers(min_value=0, max_value=2**16),
        scale=st.sampled_from([0.0, 0.3, 3.0]),
    )
    @settings(max_examples=40, deadline=None)
    def test_the_gradient_is_bit_identical_and_the_backward_is_replayed_per_access(
        self, k: int, extra: int, seed: int, scale: float
    ) -> None:
        """The parameter's gradient under the scope is the uncached one to
        the bit: each access's cotangent goes through its own replay of the
        map's backward and reaches the parameter as the same two
        contributions, in the same order. The backward issues the same ops
        as before — the scope removes forward evaluations, not backward
        ones — which is what pins the replay as a replay."""
        width = k + extra
        x = torch.randn(6, width, generator=torch.Generator().manual_seed(seed))
        f = torch.randn(6, k, generator=torch.Generator().manual_seed(seed + 1))
        results = []
        for scope in (contextlib.nullcontext(), featurizer_cache()):
            stage = _subspace(width, k, seed=seed)
            _set_original(stage, _draw(width, k, seed + 2, scale, False))
            results.append(_fit_step(stage, x, f, scope))
        (plain_out, plain_grad, plain_bwd), (cached_out, cached_grad, cached_bwd) = (
            results
        )
        for a, b in zip(plain_out, cached_out, strict=True):
            assert torch.equal(a, b)
        assert torch.equal(plain_grad, cached_grad), (
            (plain_grad - cached_grad).abs().max()
        )
        assert cached_bwd.counts.get("linalg_inv_ex", 0) == 0, (
            "no forward in the backward"
        )
        assert cached_bwd.counts.get("mm", 0) == plain_bwd.counts.get("mm", 0)

    @pytest.mark.parametrize("parametrization", ["matrix_exp", "stiefel"])
    def test_torch_orthogonal_maps_share_without_grad_only(
        self, parametrization: str
    ) -> None:
        """Torch's ``orthogonal`` maps have no replay, so a grad access gets
        its own evaluation (exactness) and a no-grad access shares one."""
        torch.manual_seed(0)
        stage = _subspace(16, 4, parametrization)
        x = torch.randn(3, 16)
        marker = {
            "matrix_exp": "linalg_matrix_exp",
            "stiefel": "linalg_householder_product",
        }[parametrization]
        with featurizer_cache():
            with _OpCounter() as with_grad:
                stage.featurize(x)
                stage.featurize(x)
            with _OpCounter() as without, torch.no_grad():
                stage.featurize(x)
                stage.featurize(x)
                stage.inverse(x[:, :4], None)
        assert with_grad.counts.get(marker, 0) == 2, with_grad.counts
        assert without.counts.get(marker, 0) == 1, without.counts

    def test_the_cache_ends_with_the_scope(self) -> None:
        stage = _subspace(16, 4)
        x = torch.randn(2, 16, generator=torch.Generator().manual_seed(9))
        with featurizer_cache():
            before, _ = stage.featurize(x)
        _set_original(stage, _draw(16, 4, 10, 0.5, False))
        after, _ = stage.featurize(x)
        assert not torch.equal(before, after), "a stale weight survived the scope"

    def test_a_no_grad_access_before_a_grad_one_costs_it_nothing(self) -> None:
        """The order a CUDA-graph executor produces: its frozen-input pass
        reads the weight under ``no_grad`` on the same stages the grad
        forward then trains. One map; the gradient is the uncached one."""
        x = torch.randn(3, 16, generator=torch.Generator().manual_seed(11))
        stage = _subspace(16, 4)
        _set_original(stage, _draw(16, 4, 12, 0.3, False))
        with _OpCounter() as ops, featurizer_cache():
            with torch.no_grad():
                frozen, _ = stage.featurize(x)
            assert not frozen.requires_grad
            live, _ = stage.featurize(x)
            assert live.requires_grad
            assert torch.equal(frozen, live)
            live.sum().backward()
        assert ops.counts["linalg_inv_ex"] == 1
        cached_grad = _original(stage).grad
        assert cached_grad is not None

        plain = _subspace(16, 4)
        _set_original(plain, _draw(16, 4, 12, 0.3, False))
        plain.featurize(x)[0].sum().backward()
        assert torch.equal(cached_grad, _original(plain).grad)  # type: ignore[arg-type]

    def test_a_grad_access_before_a_no_grad_one_costs_it_nothing(self) -> None:
        """The eval pass's order: the read's featurize runs with grad
        enabled, the write hooks under ``no_grad`` — one map, not two."""
        stage = _subspace(16, 4)
        x = torch.randn(3, 16, generator=torch.Generator().manual_seed(13))
        with _OpCounter() as ops, featurizer_cache():
            stage.featurize(x)
            with torch.no_grad():
                stage.featurize(x)
                stage.inverse(x[:, :4], None)
        assert ops.counts["linalg_inv_ex"] == 1

    def test_a_frozen_stage_is_computed_once_under_grad_too(self) -> None:
        """A stage whose parameter is not trained has no gradient to keep
        exact, so its value is shared whatever the grad mode."""
        stage = _subspace(16, 4)
        _original(stage).requires_grad_(False)
        x = torch.randn(3, 16, generator=torch.Generator().manual_seed(14))
        with _OpCounter() as ops, featurizer_cache():
            stage.featurize(x)
            stage.featurize(x)
        assert ops.counts["linalg_inv_ex"] == 1

    def test_scopes_nest(self) -> None:
        stage = _subspace(16, 4)
        x = torch.randn(3, 16, generator=torch.Generator().manual_seed(15))
        with _OpCounter() as ops, featurizer_cache():
            stage.featurize(x)
            with featurizer_cache():
                stage.featurize(x)
            stage.featurize(x)  # the inner exit must not have emptied the cache
        assert ops.counts["linalg_inv_ex"] == 1

    def test_slot_params_reads_the_shared_rotation(self) -> None:
        """The regularizer reads the weight through ``slot_params`` inside the
        same step; it must be the one evaluation the forward used."""
        stage = _subspace(16, 4)
        x = torch.randn(3, 16, generator=torch.Generator().manual_seed(16))
        with featurizer_cache():
            stage.featurize(x)
            with _OpCounter() as ops:
                weight = stage.slot_params()["weight"]
        assert ops.counts.get("linalg_inv_ex", 0) == 0 and ops.counts.get("mm", 0) == 0
        assert weight.shape == (16, 4) and weight.requires_grad


def _gate(parametrization: str = "sigmoid", **kw: Any) -> Gate:
    gate = Gate(8, parametrization=parametrization, **kw)
    with torch.no_grad():
        gate.theta.copy_(
            torch.linspace(-2, 2, gate.theta.numel()).view(gate.theta.shape)
        )
    return gate


@pytest.mark.unit
class TestGateCache:
    def test_an_eval_mask_is_computed_once_per_scope(self) -> None:
        gate = _gate().eval()
        x = torch.randn(3, 8, generator=torch.Generator().manual_seed(17))
        with torch.no_grad():
            plain = [gate.featurize(x) for _ in range(3)]
            with _OpCounter() as ops, featurizer_cache():
                cached = [gate.featurize(x) for _ in range(3)]
        assert ops.counts["gt"] == 1, ops.counts  # the hard split θ > 0, once
        for (f1, e1), (f2, e2) in zip(plain, cached, strict=True):
            assert torch.equal(f1, f2) and torch.equal(e1, e2)  # type: ignore[arg-type]

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
    @pytest.mark.parametrize(
        "spec",
        [
            pytest.param(dict(parametrization="sigmoid"), id="sigmoid"),
            pytest.param(dict(parametrization="hard_concrete"), id="hard_concrete"),
            pytest.param(
                dict(parametrization="budget", k_schedule={"kind": "fixed", "k": 3}),
                id="budget",
            ),
            pytest.param(
                dict(parametrization="sigmoid", group="head", groups=(4, 2)), id="head"
            ),
        ],
    )
    def test_a_training_mask_is_computed_once_and_its_gradient_is_bit_identical(
        self, spec: dict[str, Any], dtype: torch.dtype, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Under grad the mask's graph reaches ``theta`` by one edge, so one
        replay per access is the standalone backward exactly — including the
        cast to the activation's dtype, which sits on the gradient path: a
        bf16 consumer's cotangent is cast back per access, never summed in
        bf16 with another's."""
        calls = {"n": 0}
        real = Gate._table_mask  # pyright: ignore[reportPrivateUsage]

        def counting(self: Gate) -> torch.Tensor:
            calls["n"] += 1
            return real(self)

        monkeypatch.setattr(Gate, "_table_mask", counting)
        x = torch.randn(3, 8, generator=torch.Generator().manual_seed(18)).to(dtype)
        grads: list[torch.Tensor] = []
        for scope in (contextlib.nullcontext(), featurizer_cache()):
            gate = _gate(**spec).train()
            if spec["parametrization"] in ("hard_concrete", "budget"):
                gate.resample(torch.Generator().manual_seed(0))
            calls["n"] = 0
            with scope:
                outs = [gate.featurize(x)[0] for _ in range(3)]
                torch.stack([o.float().pow(2).sum() for o in outs]).sum().backward()
            assert gate.theta.grad is not None
            grads.append(gate.theta.grad.clone())
        assert calls["n"] == 1, "three accesses, one mask — cached"
        assert torch.equal(grads[0], grads[1])

    def test_a_leaky_mask_is_computed_per_access(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A ``dead.leak`` adds ``ε(θ − θ.detach())``: a second edge into
        ``theta``, whose two contributions a single replay could not hand
        back in the standalone order — so each access computes its own, and
        the gradient is exactly the uncached one either way."""
        calls = {"n": 0}
        real = Gate._table_mask  # pyright: ignore[reportPrivateUsage]

        def counting(self: Gate) -> torch.Tensor:
            calls["n"] += 1
            return real(self)

        monkeypatch.setattr(Gate, "_table_mask", counting)
        x = torch.randn(3, 8, generator=torch.Generator().manual_seed(22))
        grads: list[torch.Tensor] = []
        for scope in (contextlib.nullcontext(), featurizer_cache()):
            gate = _gate(dead={"leak": 0.1}).train()
            calls["n"] = 0
            with scope:
                outs = [gate.featurize(x)[0] for _ in range(3)]
                torch.stack([o.pow(2).sum() for o in outs]).sum().backward()
            assert calls["n"] == 3
            assert gate.theta.grad is not None
            grads.append(gate.theta.grad.clone())
        assert torch.equal(grads[0], grads[1])

    def test_a_leaky_mask_is_one_value_under_no_grad(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """What one replay cannot serve is still exact as a value: under
        ``no_grad`` a leaky gate's mask is evaluated once per scope — whether
        the scope learns it cannot be replayed from a no-grad access or from
        a grad one — and only a grad access computes its own."""
        calls = {"n": 0}
        real = Gate._table_mask  # pyright: ignore[reportPrivateUsage]

        def counting(self: Gate) -> torch.Tensor:
            calls["n"] += 1
            return real(self)

        monkeypatch.setattr(Gate, "_table_mask", counting)
        x = torch.randn(3, 8, generator=torch.Generator().manual_seed(23))
        gate = _gate(dead={"leak": 0.1}).train()
        with featurizer_cache():
            calls["n"] = 0
            with torch.no_grad():
                outs = [gate.featurize(x)[0] for _ in range(3)]
            assert calls["n"] == 1, "no-grad first: the probe is the value, kept"
            assert all(torch.equal(o, outs[0]) for o in outs)
            live, _ = gate.featurize(x)
            assert calls["n"] == 2 and torch.equal(live, outs[0]), (
                "a grad access computes its own"
            )
            with torch.no_grad():
                gate.featurize(x)
            assert calls["n"] == 2, "the no-grad value from before, not a third"
        gate = _gate(dead={"leak": 0.1}).train()
        with featurizer_cache():
            calls["n"] = 0
            gate.featurize(x)  # grad first: the probe, recorded unshareable
            with torch.no_grad():
                gate.featurize(x)
                gate.featurize(x)
            assert calls["n"] == 2, "grad first: one value for both no-grad accesses"

    def test_a_no_grad_access_before_a_grad_one_costs_the_gate_nothing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """As for the subspace: the evaluation a no-grad access makes is the
        one a later grad access in the scope replays — one mask, and the
        gradient is the uncached one."""
        calls = {"n": 0}
        real = Gate._table_mask  # pyright: ignore[reportPrivateUsage]

        def counting(self: Gate) -> torch.Tensor:
            calls["n"] += 1
            return real(self)

        monkeypatch.setattr(Gate, "_table_mask", counting)
        x = torch.randn(3, 8, generator=torch.Generator().manual_seed(24))
        grads: list[torch.Tensor] = []
        for scoped in (False, True):
            gate = _gate().train()
            calls["n"] = 0
            with featurizer_cache() if scoped else contextlib.nullcontext():
                with torch.no_grad():
                    frozen, _ = gate.featurize(x)
                live, _ = gate.featurize(x)
                assert torch.equal(frozen, live) and live.requires_grad
                live.pow(2).sum().backward()
            assert calls["n"] == (1 if scoped else 2)
            assert gate.theta.grad is not None
            grads.append(gate.theta.grad.clone())
        assert torch.equal(grads[0], grads[1])

    def test_train_and_eval_masks_do_not_share_an_entry(self) -> None:
        """The key carries the mode: an eval pass's hard mask and a training
        step's soft one are different quantities even inside one scope."""
        gate = _gate()
        x = torch.ones(1, 8)
        with torch.no_grad(), featurizer_cache():
            gate.train()
            soft, _ = gate.featurize(x)
            gate.eval()
            hard, _ = gate.featurize(x)
        assert torch.equal(hard, (gate.theta > 0).to(x.dtype) * x)
        assert not torch.equal(soft, hard)

    def test_the_routing_lookup_stays_per_call(self) -> None:
        """An expert-keyed gate's table is shared; the slots a token fills are
        looked up per call, so two calls with different routing tables get
        different masks — and the same ones as without the cache."""
        gate = Gate(6, group="expert_neuron", groups=(4, 3)).eval()
        with torch.no_grad():
            gate.theta.copy_(torch.arange(12, dtype=torch.float32).view(4, 3) / 6 - 1)
        x = torch.randn(2, 6, generator=torch.Generator().manual_seed(19))
        r1 = torch.tensor([[0, 1], [2, 3]])
        r2 = torch.tensor([[3, 2], [1, 0]])
        with torch.no_grad():
            plain = [gate.featurize(x, routing=r)[0] for r in (r1, r2)]
            with _OpCounter() as ops, featurizer_cache():
                cached = [gate.featurize(x, routing=r)[0] for r in (r1, r2)]
        assert ops.counts["gt"] == 1
        assert torch.equal(plain[0], cached[0]) and torch.equal(plain[1], cached[1])
        assert not torch.equal(cached[0], cached[1])

    def test_an_expert_keyed_gate_casts_to_the_activation_after_the_lookup(
        self,
    ) -> None:
        """The table is shared in ``theta``'s dtype and the cast to ``x``'s
        follows the routing lookup per call, so ``mask[routing]``'s backward
        — a scatter-add in the indexed tensor's dtype — sums the slots'
        cotangents in fp32 as it always did. The rows here are built so a
        bf16 sum would differ (``1 + 2⁻¹⁰`` rounds back to ``1`` in bf16):
        the gradient is the cast-after-lookup one exactly, scope or no
        scope, and not the cast-before-lookup one."""
        x = torch.full((5, 6), 2.0**-10, dtype=torch.bfloat16)
        x[0, :] = 1.0
        routings = (torch.tensor([[0, 1]] * 5), torch.tensor([[2, 3]] * 5))

        def gate() -> Gate:
            g = Gate(6, group="expert_neuron", groups=(4, 3)).train()
            with torch.no_grad():
                g.theta.copy_(torch.arange(12, dtype=torch.float32).view(4, 3) / 6 - 1)
            return g

        def by_hand(cast_first: bool) -> torch.Tensor:
            g = gate()
            loss = torch.zeros(())
            for r in routings:
                table = g._table_mask()  # pyright: ignore[reportPrivateUsage]
                if cast_first:
                    mask = g._route(table.to(x.dtype), r)  # pyright: ignore[reportPrivateUsage]
                else:
                    mask = g._route(table, r).to(x.dtype)  # pyright: ignore[reportPrivateUsage]
                loss = loss + (mask * x).sum()
            loss.backward()
            assert g.theta.grad is not None
            return g.theta.grad.clone()

        reference, wrong_order = by_hand(False), by_hand(True)
        assert not torch.equal(reference, wrong_order), "the rows tell the orders apart"
        for scope in (contextlib.nullcontext(), featurizer_cache()):
            g = gate()
            with scope:
                loss = torch.zeros(())
                for r in routings:
                    f, _ = g.featurize(x, routing=r)
                    assert f.dtype == x.dtype
                    loss = loss + f.sum()
                loss.backward()
            assert g.theta.grad is not None
            assert torch.equal(g.theta.grad, reference)

    def test_a_budget_gate_cuts_its_ranking_once_per_scope(self) -> None:
        gate = _gate("budget", k_schedule={"kind": "fixed", "k": 3}).eval()
        x = torch.randn(2, 8, generator=torch.Generator().manual_seed(20))
        with torch.no_grad():
            plain, _ = gate.featurize(x)
            with _OpCounter() as twice:
                gate.featurize(x)
                gate.featurize(x)
            with _OpCounter() as once, featurizer_cache():
                first, _ = gate.featurize(x)
                second, _ = gate.featurize(x)
        assert once.total() < twice.total()
        assert torch.equal(first, plain) and torch.equal(second, plain)

    def test_the_gradient_reaches_theta_after_a_no_grad_first_access(self) -> None:
        gate = _gate().train()
        x = torch.randn(2, 8, generator=torch.Generator().manual_seed(21))
        with featurizer_cache():
            with torch.no_grad():
                gate.featurize(x)
            f, _ = gate.featurize(x)
            f.sum().backward()
        assert gate.theta.grad is not None and bool((gate.theta.grad != 0).any())


@pytest.mark.unit
class TestLeafEdges:
    """``_leaf_edges`` decides whether one replay node per access can stand
    in for a graph per access: the parameter's contributions must be tellable
    apart (one edge) and nothing else trainable may be reached."""

    def test_the_cayley_map_reaches_each_alias_once_and_sees_the_other(self) -> None:
        stage = _subspace(16, 4)
        frame = _original(stage).detach().requires_grad_(True)
        complement = _original(stage).detach().requires_grad_(True)
        q = _cayley(stage).map(frame, complement)
        assert _leaf_edges(q, frame) == (1, True)
        assert _leaf_edges(q, complement) == (1, True)

    def test_a_gate_mask_reaches_theta_once_a_leak_twice(self) -> None:
        plain = _gate().train()
        assert _leaf_edges(plain.featurize(torch.ones(1, 8))[0], plain.theta) == (
            1,
            False,
        )
        leaky = _gate(dead={"leak": 0.1}).train()
        assert _leaf_edges(leaky.featurize(torch.ones(1, 8))[0], leaky.theta) == (
            2,
            False,
        )

    def test_uses_through_one_view_are_one_edge(self) -> None:
        """A budget mask uses ``theta.view(-1)`` twice; both go through the
        view's node, which sums them itself — one edge into ``theta``."""
        theta = torch.zeros(6, requires_grad=True)
        flat = theta.view(-1)
        out = (flat * 2).sum() + torch.sigmoid(flat).sum()
        assert _leaf_edges(out, theta) == (1, False)

    def test_a_deep_graph_is_walked_to_the_leaf(self) -> None:
        """The wrappers ``next_functions`` returns are fresh objects; a walk
        that let them die would see recycled ids and stop early."""
        theta = torch.zeros(6, requires_grad=True)
        out = theta
        for _ in range(40):
            out = torch.sigmoid(out) * 1.5
        assert _leaf_edges(out.to(torch.bfloat16), theta) == (1, False)


@pytest.mark.unit
class TestIsolatedScope:
    """A captured pass runs under an isolated scope: it must evaluate every
    stage inside the pass — a value shared in from outside would be baked
    into the graph and replayed stale — and must leave the enclosing scope's
    store as it found it."""

    def test_an_isolated_scope_shares_nothing_with_the_enclosing_one(self) -> None:
        stage = _subspace(16, 4)
        x = torch.randn(2, 16, generator=torch.Generator().manual_seed(23))
        with featurizer_cache():
            stage.featurize(x)
            with _OpCounter() as inner, featurizer_cache(isolated=True):
                stage.featurize(x)
                stage.featurize(x)
            with _OpCounter() as outer:
                stage.featurize(x)
        assert inner.counts["linalg_inv_ex"] == 1, "its own evaluation, once"
        assert outer.counts.get("linalg_inv_ex", 0) == 0, "the enclosing entry survived"

    def test_an_isolated_scope_sees_the_parameter_as_it_stands(self) -> None:
        """Two captured passes around a parameter update — the warmup pass
        and the capture, or two captures — each see the current weight."""
        stage = _subspace(16, 4)
        x = torch.randn(2, 16, generator=torch.Generator().manual_seed(24))
        with featurizer_cache():
            with featurizer_cache(isolated=True):
                before, _ = stage.featurize(x)
            _set_original(stage, _draw(16, 4, 25, 0.5, False))
            with featurizer_cache(isolated=True):
                after, _ = stage.featurize(x)
        assert not torch.equal(before, after)

    def test_an_isolated_scope_outside_any_scope_leaves_none_behind(self) -> None:
        from causalab.neural.shared.featurizers.sharing import (
            _SCOPE,  # pyright: ignore[reportPrivateUsage]
        )

        stage = _subspace(16, 4)
        with featurizer_cache(isolated=True):
            stage.featurize(torch.ones(1, 16))
            assert _SCOPE.entries is not None
        assert _SCOPE.entries is None and _SCOPE.depth == 0
