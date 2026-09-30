"""Gate stages and shared budgets for Desiderata-Based Masking.

``Gate`` supplies the parametrization and forward/backward split.
``BudgetPool`` and ``link_budget_pools`` coordinate schedules across gates.
"""

from __future__ import annotations

import json
import math
from typing import Any, Callable, Mapping, Sequence

import torch

from causalab.protocol.rules.errors import ProtocolError
from causalab.protocol.registry import gate_param_shape
from causalab.protocol.schema import (
    FeaturizerSpec,
    GATE_DEAD_RULES,
    GATE_PARAMETRIZATIONS,
    HARD_CONCRETE_STRETCH,
    HARD_CONCRETE_TEMPERATURE,
    hard_concrete_theta,
    hard_concrete_threshold,
)
from causalab.neural.shared.featurizers.sharing import _SCOPE, _UNSHARED, _Shared, _once  # pyright: ignore[reportPrivateUsage]
from causalab.neural.shared.featurizers.stages import Stage


class Gate(Stage):
    """The DBM gate: soft ``σ(θ/T) ⊙ x`` in training, hard ``θ > 0`` in
    eval. ``temperature`` is the anneal target ``<name>.theta.temperature``.

    ``parametrization`` is how ``theta`` maps to the mask (§2.5,
    ``GATE_PARAMETRIZATIONS``). ``sigmoid`` is the description above. Under
    ``clamp`` the parameter *is* the mask: soft ``m = θ``, projected into
    ``[0, 1]`` after every optimizer step ([`project`][]), hard ``θ > ½``
    — i.e. ``round`` — and no temperature: the anneal target is refused at
    load and again here. The L1 term and the decisiveness diagnostic read the
    mask through [`soft_mask`][], the hard count through [`hard_mask`][],
    so neither spells a parametrization out.

    Under ``hard_concrete`` (Louizos, Welling & Kingma 2018) the training
    mask is a *sample* of the stretched, clipped concrete distribution and the
    eval mask its deterministic mean above ½ ([`hard_concrete_threshold`][causalab.protocol.schema.featurizers.hard_concrete_threshold]).
    The draw is not made here: the train loop calls [`resample`][] once per
    optimizer step with the fit's generator, and every ``featurize`` of that
    step — the read of the counterfactual *and* the write into the base — reads
    the one cached draw, so ``m·v_cf + (1 − m)·v_base`` is an interchange and
    not two masks ([`sampled_mask`][]). ``temperature`` is β and ``stretch``
    the ``(γ, ζ)`` of the relaxation; ``init.fill p`` starts the deterministic
    mask at exactly ``p`` by inverting the stretch.

    ``group`` is the unit kind the document authored (§2.5) and ``groups``
    the derived map that goes with it; ``None`` for both is the per-coordinate
    gate. Under ``head``, ``groups`` is ``(heads, head_dim)``: ``theta`` has
    one entry per head and every coordinate of a head receives that entry's
    value, soft or hard. Under ``expert_neuron``, ``groups`` is
    ``(num_experts, d_expert)``: ``theta`` is the whole expert table,
    ``theta[expert, neuron]``, and the site is token-major with ``top_k``
    slots of ``d_expert`` — so ``featurize`` needs the routing table beside
    the activation, and slot *k* of a token receives the row of the expert
    ``expert_idx[..., k]`` names. An expert a token did not activate has no
    slot, so its parameters do not touch that token. Either way the L1 term
    (the mean of the soft mask over ``theta``) and the hard-mask count are
    over units, not coordinates.

    ``dead`` (§2.5 ``GATE_DEAD_RULES``) is what a *training* gate does about a
    unit whose hard mask has closed. Under ``{"freeze_after": n}`` a unit
    hard-off for ``n`` consecutive optimizer steps is frozen: its ``theta`` is
    photographed and restored after every later step ([`project`][]), so
    the optimizer's momentum cannot reopen it and a pruned unit stays pruned
    during DBM training. The kept sets form a nested sequence by
    construction. Under ``{"leak": ε}`` the training mask's *derivative* is
    floored: the forward value is the map's own ``m``, unchanged, and the
    backward pass sees ``∂m/∂θ + ε`` (`_mask`, the leaky-ReLU idiom
    ``m + ε·(θ − θ.detach())``), so a unit whose map has saturated at the
    zero pole — ``σ'(θ/T) ≈ 0``, or a concrete sample clipped to 0 — still
    receives ``ε·∂L/∂m``, which is nonzero whenever the loss would rather have
    the unit's counterfactual, and can come back. A *value* floor would not
    do this: ``ε + (1 − ε)·m`` leaves ``∂/∂θ`` scaled by the same vanishing
    ``σ'``, and it routes ``ε`` of the counterfactual through a unit the eval
    mask drops — the co-adaptation a hard mask exists to prevent. The
    eval-mode hard mask is untouched. Both rules are bookkept per
    unit here because [`project`][] is the one hook the loop calls after
    every step for every stage; ``fit_diagnostics`` reads
    [`dead_diagnostics`][] — how many units froze, how many were ever
    hard-off and are kept at the end (``reawakened_units``).

    Under ``boundary`` (Boundless DAS, Wu et al. 2023) ``theta`` is **one**
    scalar in ``[0, 1]``, the boundary as a fraction of the width — the
    paper's spelling, so a default learning rate crosses the whole width and
    the saved θ reads the same at any width — and ``β = θ · width`` is the
    boundary in coordinates ([`boundary`][]). The mask runs over the
    coordinate *index* of the input: soft ``σ((β − i)/T)`` for ``i = 0 …
    width−1`` ([`soft_mask`][]), hard ``i < β`` ([`hard_mask`][]) — the
    soft mask at ½, as under every map, so the first ``⌈β⌉`` coordinates, a
    prefix, and exactly the split a ``forward: hard`` training forward uses
    — with θ projected into ``[0, 1]`` after every step ([`project`][]).
    The order is the input's, so the gate sits behind a ``subspace`` (its
    column order) or a ``pca`` (variance order); the validator holds that at
    load. ``temperature`` is the paper's ``T`` in coordinates, set and
    annealed as the sigmoid gate's; ``init.fill p`` is ``θ = p``, the kept
    fraction, as under ``clamp``.
    Nothing per unit exists — no [`hard_threshold`][], no [`ranking`][],
    no ``top_k``, ``group``, ``axis``, ``dead`` or pool — and each refuses by
    name rather than reading a one-entry θ as one unit. The mask is kept a
    function of ``(θ, arange(width), T)`` so a two-sided ``[β_lo, β_hi]``
    could slot in as a second θ entry without touching the rest."""

    kind = "gate"

    def __init__(
        self,
        width: int,
        *,
        group: str | None = None,
        groups: tuple[int, int] | None = None,
        parametrization: str = "sigmoid",
        init: float | torch.Tensor | None = None,
        init_identity: Mapping[str, Any] | None = None,
        temperature: float | None = None,
        stretch: tuple[float, float] | None = None,
        k_schedule: Mapping[str, Any] | None = None,
        stop_grad_shift: bool = False,
        pool: str | None = None,
        dead: Mapping[str, Any] | None = None,
        axis: str | None = None,
        forward: str | None = None,
    ) -> None:
        super().__init__()
        if forward not in (None, "hard"):
            raise ValueError(f"a gate's forward is 'hard' or absent, got {forward!r}")
        #: §2.5 the mapping form of ``parametrization``: ``"hard"`` when the
        #: training forward uses the map's mask thresholded at ½ with the
        #: map's gradient behind it (straight-through); ``None`` is the map's
        #: own forward. Eval is the map's hard split either way.
        #: stored as ``forward_mask``: ``forward`` is ``nn.Module``'s callable
        #: slot, and a string there would make every gate uncallable as a
        #: module (hooks, compile, DDP). The document key and the stamp key
        #: stay ``forward``
        self.forward_mask: str | None = forward
        if axis not in (None, "position"):
            raise ValueError(f"a gate's axis is 'position' or absent, got {axis!r}")
        if axis == "position" and group is not None:
            raise ValueError("a position gate takes no group: it is one θ per position")
        if axis == "position" and pool is not None:
            # the parser refuses this too; a second line of defense, as `group`
            raise ValueError("a position gate takes no pool: no budget over positions")
        #: §2.5 ``axis``: ``"position"`` when θ runs over the addressed token
        #: positions — ``width`` is then the window's length and the mask
        #: broadcasts along the position axis of a ``(rows, positions,
        #: width)`` value — else ``None``, the feature gate
        self.axis: str | None = axis
        if parametrization not in GATE_PARAMETRIZATIONS:
            raise ValueError(
                f"unknown gate parametrization {parametrization!r}; one of "
                f"{list(GATE_PARAMETRIZATIONS)}"
            )
        #: ``hard_concrete`` only: the stretch ``(γ, ζ)`` the concrete sample
        #: is mapped onto before clipping; ``None`` under the other maps
        self.stretch: tuple[float, float] | None = None
        beta: float | None = None
        if parametrization == "hard_concrete":
            beta = (
                HARD_CONCRETE_TEMPERATURE if temperature is None else float(temperature)
            )
            lo, hi = HARD_CONCRETE_STRETCH if stretch is None else stretch
            if beta <= 0.0:
                raise ValueError(f"a hard-concrete temperature is positive, got {beta}")
            if not float(lo) < 0.0 < 1.0 < float(hi):
                raise ValueError(
                    f"stretch [γ, ζ] must satisfy γ < 0 < 1 < ζ, got {stretch}"
                )
            self.stretch = (float(lo), float(hi))
        elif parametrization == "boundary":
            if stretch is not None:
                raise ValueError(
                    "stretch is the hard_concrete relaxation's constant; a boundary "
                    "gate samples nothing"
                )
            if temperature is not None and float(temperature) <= 0.0:
                raise ValueError(
                    f"a boundary temperature is positive, got {temperature}"
                )
            beta = 1.0 if temperature is None else float(temperature)
            if group is not None:
                raise ValueError(
                    "a boundary gate takes no group: its one θ is a boundary over "
                    "the ordered coordinates of the stage before it, not an entry "
                    "per unit"
                )
            if axis is not None:
                raise ValueError(
                    "a boundary gate takes no axis: a window's positions have no "
                    "order for a boundary to be a prefix of"
                )
            if dead is not None:
                raise ValueError(
                    "a boundary gate takes no dead rule: one β has no unit to "
                    "freeze or leak"
                )
            if pool is not None:
                raise ValueError(
                    "a boundary gate joins no pool: it has no ranking to share or cut"
                )
        elif temperature is not None or stretch is not None:
            raise ValueError(
                "temperature and stretch are hard_concrete's constants; a "
                f"{parametrization!r} gate samples nothing"
            )
        #: ``budget`` only (§2.5 ``k_schedule``): how each step draws its
        #: budget, and the cut the eval-mode split is read at (``eval``, or
        #: ``k`` when fixed). A budget gate being fitted needs one; a loaded
        #: one is read out through [`top_k`][causalab.neural.shared.featurizers.gate.Gate.top_k] instead and carries none.
        self.k_schedule: dict[str, Any] | None = None
        self.stop_grad_shift: bool = bool(stop_grad_shift)
        #: ``budget`` only (§2.5 ``pool``): the name of the budget pool this
        #: gate shares one ``k``, one shift and one ranking with, and — once
        #: [`link_budget_pools`][] has run over the point's stages — the
        #: [`BudgetPool`][] itself. ``None`` for a gate that budgets alone.
        self.pool_name: str | None = pool
        self.pool: BudgetPool | None = None
        #: a loaded pooled gate's stamped ``pool_units`` (§8), compared with the
        #: pool the document assembles at the link — ``None`` when unstamped
        self.stamped_pool_units: int | None = None
        if parametrization == "budget":
            if k_schedule is not None:
                self.k_schedule = _checked_k_schedule(k_schedule)
        elif k_schedule is not None or stop_grad_shift:
            raise ValueError(
                "k_schedule and stop_grad_shift belong to the budget "
                f"parametrization; a {parametrization!r} gate draws no budget"
            )
        # `pool` on a non-budget gate is legal only as a pooled READOUT of a
        # loaded gate (`from_theta` with `top_k`); a fit under another map has
        # no budget to share, which `from_theta`'s caller and the parser hold
        if (group is None) != (groups is None):
            raise ValueError("a gate's group kind and group map come together")
        if (
            group in ("head", "site")
            and groups is not None
            and groups[0] * groups[1] != width
        ):
            raise ValueError(f"group map {groups} does not tile a {width}-wide gate")
        if group == "expert_neuron" and groups is not None and width % groups[1]:
            raise ValueError(
                f"expert map {groups}: a {width}-wide gate is not a whole number "
                f"of {groups[1]}-wide expert slots"
            )
        self.width = width
        self.group = group
        self.groups = groups
        self.parametrization = parametrization
        #: the ``init.fill`` the gate started at, when it did (§2.5) —
        #: recorded by ``fit_diagnostics`` so the record says where it began
        self.init_fill: float | None = None
        #: what a bundle this gate is saved into records about a saved start
        #: (``_gate_start``): the ``init_*`` provenance keys, as a subspace's
        self.init_identity: dict[str, Any] = dict(init_identity or {})
        #: the ``init.from_scores`` a start was read from, when it was (§2.5)
        #: — the resolved table path, the mode and the units it put on the
        #: kept pole — recorded by ``fit_diagnostics`` beside ``init_fill``
        self.init_scores: dict[str, Any] | None = None
        shape = gate_param_shape(group, groups, width, parametrization=parametrization)
        if init is None:
            # the midpoint mask under every map: σ(0) = ½, or ½ itself — under
            # `boundary` the half prefix, θ = ½
            start = torch.full(
                shape, 0.5 if parametrization in ("clamp", "boundary") else 0.0
            )
        elif isinstance(init, torch.Tensor):
            # a saved theta of this very gate, verbatim (§2.5 init.file_path)
            if init.numel() != math.prod(shape):
                raise ValueError(
                    f"a gate over {list(shape)} needs {math.prod(shape)} parameters "
                    f"to start from, got {init.numel()}"
                )
            start = init.detach().to(torch.float32).reshape(shape).clone()
        else:
            # a mask value, mapped into theta by the parametrization
            fill = float(init)
            if not 0.0 <= fill <= 1.0:
                raise ValueError(f"init fill is a mask value in [0, 1], got {fill}")
            if parametrization in ("sigmoid", "hard_concrete", "budget"):
                if self.stretch is None and not 0.0 < fill < 1.0:
                    # only the sigmoid start is a bare logit; the hard-concrete
                    # start inverts the stretch and is finite at both poles
                    raise ValueError(
                        f"a {parametrization} gate starts at θ = logit(fill), which "
                        f"is undefined at fill={fill}; start strictly inside (0, 1), "
                        "or use parametrization 'clamp' or 'hard_concrete' for a "
                        "start at a pole"
                    )
                if self.stretch is not None:
                    # the deterministic mask is clip(σ(θ)·(ζ−γ)+γ), so the θ
                    # whose mask is exactly `fill` inverts the stretch — the
                    # same map that puts the eval split at mask ½
                    theta_0 = hard_concrete_theta(fill, self.stretch)
                else:
                    theta_0 = math.log(fill / (1.0 - fill))
                start = torch.full(shape, theta_0)
            else:
                # `clamp`: the mask itself; `boundary`: the kept fraction
                start = torch.full(shape, fill)
            self.init_fill = fill
        self.theta = torch.nn.Parameter(start)
        #: §2.5 ``dead``: ``freeze_after`` steps, or ``None``
        self.freeze_after: int | None = None
        #: §2.5 ``dead``: the gradient leak ``ε`` added to ``∂m/∂θ``, or ``None``
        self.leak: float | None = None
        if dead is not None:
            rules = dict(dead)
            unknown = sorted(set(rules) - set(GATE_DEAD_RULES))
            if unknown or len(rules) != 1:
                raise ValueError(
                    f"a gate's dead-unit rule is exactly one of {list(GATE_DEAD_RULES)}, "
                    f"got {sorted(rules)}"
                )
            if "freeze_after" in rules:
                n = int(rules["freeze_after"])
                if n < 1:
                    raise ValueError(
                        f"freeze_after counts steps, a positive integer; got {n}"
                    )
                self.freeze_after = n
            else:
                eps = float(rules["leak"])
                if not 0.0 < eps < 1.0:
                    raise ValueError(
                        f"leak is a gradient slope strictly inside (0, 1); got {eps}"
                    )
                self.leak = eps
        # per-unit bookkeeping for `dead` and for `reawakened_units`, kept as
        # buffers so a `.to(device)` moves them with theta; none is a slot
        self.register_buffer("_off_streak", torch.zeros(shape, dtype=torch.long))
        self.register_buffer("_frozen", torch.zeros(shape, dtype=torch.bool))
        self.register_buffer("_frozen_theta", torch.zeros(shape))
        self.register_buffer("_ever_off", torch.zeros(shape, dtype=torch.bool))
        #: the sigmoid gate's anneal target ``T``; under ``hard_concrete`` the
        #: concrete temperature β (annealable the same way)
        self.temperature: float = 1.0 if beta is None else beta
        self.hard_eval: bool = True
        self.capture_temperature: torch.Tensor | None = None
        #: §2.11 ``phases[i].freeze_masks``: the hard mask a phase pinned at its
        #: start, used by every forward while set — training mode included —
        #: so a featurizer trained behind this gate learns under the split it
        #: will be scored through, not under a mask still moving. ``None``
        #: outside such a phase; never saved (a bundle holds ``theta``).
        self.frozen_mask: torch.Tensor | None = None
        #: ``hard_concrete`` only: the uniform draw of the current optimizer
        #: step ([`resample`][]), ``None`` until the loop makes one
        self._draw: torch.Tensor | None = None
        #: ``budget`` only: the budget ``k`` of the current optimizer step
        #: ([`resample`][]) and the shift ``c_k`` of the last solve
        #: ([`budget_shift`][]), the mask's second coordinate beside ``θ``
        self._k: int | None = None
        self._shift: torch.Tensor | None = None
        #: a *loaded* gate's top-k readout (§2.5 ``top_k``): when set, the
        #: eval-mode split is the ``top_k`` largest units of ``theta`` rather
        #: than the map's threshold ([`hard_mask`][]). Only
        #: [`from_theta`][] sets it — a gate being fitted has none
        self.top_k: int | None = None

    @property
    def needs_routing(self) -> bool:  # type: ignore[override]
        return self.group == "expert_neuron"

    def _require_linked(self) -> None:
        """A gate that authors a pool is only ever read *through* it. The link
        ([`link_budget_pools`][]) runs after every stack build, so a gate
        reaching a mask unlinked is a broken construction, not a lone gate —
        left alone it would draw its own budget, cut its own ranking and stamp
        no pool, wrong numbers with no trace."""
        if self.pool_name is not None and self.pool is None:
            raise RuntimeError(
                f"gate authors pool {self.pool_name!r} but was never linked — "
                "link_budget_pools runs over the point's stages before any mask "
                "is computed (§2.5 pool)"
            )

    @property
    def samples_per_step(self) -> bool:
        """Whether the training mask depends on a per-step draw the loop makes
        through [`resample`][] — the hard-concrete noise, or the budget's
        ``k`` — so a read and a write the gate sits on share one mask."""
        return self.parametrization in ("hard_concrete", "budget")

    def _stretched(self, s: torch.Tensor) -> torch.Tensor:
        assert self.stretch is not None
        lo, hi = self.stretch
        return (s * (hi - lo) + lo).clamp(0.0, 1.0)

    def soft_mask(self) -> torch.Tensor:
        """The relaxed mask over ``theta``'s units, the quantity the L1 term
        and the decisiveness diagnostic are about: ``σ(θ/T)`` under
        ``sigmoid``, ``θ`` itself under ``clamp``, and under ``hard_concrete``
        the **deterministic** stretched-and-clipped ``σ(θ)`` — the mask the
        eval-mode split is taken from, with the concrete noise at its mean.
        The mask a training forward *uses* under ``hard_concrete`` is
        [`sampled_mask`][]."""
        if self.parametrization == "clamp":
            return self.theta
        if self.parametrization == "boundary":
            return self._boundary_mask()
        if self.parametrization == "hard_concrete":
            return self._stretched(torch.sigmoid(self.theta))
        if self.parametrization == "budget":
            # the mask the last forward used, or — before any step, and on a
            # loaded gate — the mask at the eval cut: `Σ m = k` either way
            # (over the pool, when pooled: the shift is the pool's)
            self._require_linked()
            last = self._shift if self.pool is None else self.pool.last_shift
            shift = last if last is not None else self.budget_shift(self.eval_k())
            return torch.sigmoid(self.theta.view(-1) + shift).view(self.theta.shape)
        if self.capture_temperature is not None:
            # CUDA scalar division lowers to reciprocal-multiply. Tensor
            # division rounds differently; preserve the eager CUDA arithmetic.
            return torch.sigmoid(self.theta * self.capture_temperature.reciprocal())
        return torch.sigmoid(self.theta / self.temperature)

    def _index(self) -> torch.Tensor:
        """The coordinate index ``i = 0 … width−1`` a ``boundary`` mask is a
        function of, on ``theta``'s device and in its dtype."""
        return torch.arange(
            self.width, device=self.theta.device, dtype=self.theta.dtype
        )

    def _boundary_mask(self) -> torch.Tensor:
        """The ``boundary`` soft mask ``σ((β − i) / T)`` over the coordinate
        index, ``β = θ · width`` — one θ broadcast against ``arange(width)``,
        so the mask's graph reaches ``theta`` by one edge and the cache can
        replay it. Honours a CUDA-graph capture's temperature slot as the
        sigmoid gate does (reciprocal-multiply, the eager CUDA arithmetic)."""
        logits = self.theta * self.width - self._index()
        if self.capture_temperature is not None:
            return torch.sigmoid(logits * self.capture_temperature.reciprocal())
        return torch.sigmoid(logits / self.temperature)

    def boundary(self) -> float:
        """A ``boundary`` gate's β in coordinates, ``θ · width`` — the learned
        rank is ``⌈β⌉``, the number of coordinates [`hard_mask`][] keeps
        (``hard_mask_size``)."""
        if self.parametrization != "boundary":
            raise ValueError("only a boundary gate has a boundary β")
        return float(self.theta.detach().view(-1)[0]) * self.width

    def resample(self, generator: torch.Generator) -> None:
        """Draw this step's concrete noise, ``u ~ U(0, 1)`` per unit, from
        ``generator`` — the fit's own CPU generator, never the global RNG — and
        keep it for every [`featurize`][] of the step. The train loop calls
        this once per optimizer step: the read of the counterfactual and the
        write into the base then share one mask, so the training intervention
        is the interchange ``m·v_cf + (1 − m)·v_base`` and not two independent
        draws that would double-write or erase a unit. Drawn on CPU and moved
        to ``theta``'s device, so a seeded fit is bit-identical across devices
        (the ``subspace`` init's rule). ``u`` is kept off the endpoints as
        NeuroSurgeon does (``1e-4``), where the logit is undefined."""
        if self.parametrization == "budget":
            self._require_linked()
            if self.pool is not None:
                raise RuntimeError(
                    f"gate in pool {self.pool.name!r} draws no budget of its "
                    "own — the train loop resamples the pool once per step "
                    "(BudgetPool.resample), so every member shares the draw"
                )
            self._k = self._draw_budget(generator)
            return
        if self.parametrization != "hard_concrete":
            raise ValueError("only a hard_concrete or budget gate samples its mask")
        u = torch.rand(self.theta.shape, generator=generator, dtype=torch.float32)
        self._draw = u.clamp(1e-4, 1.0 - 1e-4).to(self.theta.device)

    # ------------------------------------------------------------------ #
    # the budget map (§2.5 `parametrization: budget`)
    # ------------------------------------------------------------------ #

    def _draw_budget(self, generator: torch.Generator) -> int:
        """This step's budget from the schedule (§2.5 ``k_schedule``), as the
        number of units that **take the counterfactual** — the count every
        internal reader ([`budget_mask`][], [`hard_mask`][]) works in.
        The draw itself is `_draw_from_schedule`; under ``of: "kept"``
        the schedule counts the units left clean, so the draw is complemented
        against the unit count (`_as_patched`)."""
        schedule = self._budget_schedule()
        return _as_patched(
            schedule, _draw_from_schedule(schedule, generator), self.theta.numel()
        )

    def _budget_schedule(self) -> dict[str, Any]:
        if self.parametrization != "budget":
            raise ValueError("only a budget gate has a k_schedule")
        if self.k_schedule is None:
            raise ValueError(
                "a budget gate being fitted needs a k_schedule; a loaded one is "
                "read out at top_k"
            )
        return self.k_schedule

    def eval_k(self) -> int:
        """The count the eval-mode split of a ``budget`` gate is cut at: the
        document's ``top_k`` on a loaded gate, else the schedule's ``eval``
        (``k`` when fixed). A budget gate has no threshold — ``θ`` is a
        ranking — so one of the two must name the cut."""
        self._require_linked()
        if self.pool is not None:
            return self.pool.eval_k()
        if self.top_k is not None:
            return self.top_k
        schedule = self._budget_schedule()
        raw = int(schedule["eval"]) if "eval" in schedule else int(schedule["k"])
        return _as_patched(schedule, raw, self.theta.numel())

    def budget_shift(self, k: int) -> torch.Tensor:
        """The scalar ``c_k`` with ``Σ σ(θ + c_k) = k``, by
        bisection in float64 on a bracket that contains it: ``Σσ`` is strictly
        increasing in ``c`` from 0 to the unit count, so for ``0 < k < units``
        the root is unique and 60 halvings of ``[−max θ − 40, −min θ + 40]``
        pin it well below float32 resolution; at the poles ``k = 0`` /
        ``k = units`` there is no finite root and the bracket's end is returned
        (a mask within ``1e−17`` of the pole). Not differentiated through: the
        gradient the shift carries is added in [`budget_mask`][] from the
        implicit-function rule."""
        if self.pool is not None:
            return self.pool.shift(k)
        # the bisection runs on CPU in float64: a few hundred units, and MPS
        # has no float64 at all
        theta = self.theta.detach().cpu().to(torch.float64).view(-1)
        return torch.tensor(
            _solve_shift(theta, k), dtype=self.theta.dtype, device=self.theta.device
        )

    def budget_mask(self, k: int) -> torch.Tensor:
        """The budget mask at ``k``: ``σ(θ + c_k)`` over ``theta``'s units.
        The shift is solved without gradient ([`budget_shift`][]); unless
        ``stop_grad_shift``, its implicit gradient is attached — from
        ``F(θ, c) = Σ σ(θ + c) − k = 0``, ``∂c/∂θ_i = −σ'_i / Σ_j σ'_j`` — as
        ``c = c₀ + (w·θ).detach() − w·θ`` with ``w_i = σ'_i / Σ σ'_j`` held
        constant, which is exact to first order and keeps ``Σ m = k`` under any
        step. With ``stop_grad_shift`` the shift is the constant ``c₀``
        (an ablation that drops its gradient). The solve is kept as `_shift` so
        [`soft_mask`][] reports the mask the last forward used. In a pool the
        whole computation is the pool's ([`BudgetPool.mask_for`][]): one
        shift over every member's θ, the weights normalised over the pool."""
        if self.pool is not None:
            return self.pool.mask_for(self, k)
        shift = self.budget_shift(k)
        self._shift = shift
        theta = self.theta.view(-1)
        if not self.stop_grad_shift:
            with torch.no_grad():
                s = torch.sigmoid(theta + shift)
                weights = s * (1.0 - s)
                weights = weights / weights.sum().clamp_min(1e-30)
            correction = (weights * theta).sum()
            shift = shift + correction.detach() - correction
        return torch.sigmoid(theta + shift).view(self.theta.shape)

    def sampled_mask(self) -> torch.Tensor:
        """The hard-concrete mask of the current step (Louizos et al. 2018,
        eq. 10–11): with the step's draw ``u`` ([`resample`][]),
        ``s = σ((log u − log(1 − u) + θ) / β)``, stretched to ``(γ, ζ)`` and
        clipped to ``[0, 1]`` — a reparametrized sample, so the gradient reaches
        ``θ`` through the sigmoid. Refuses when no draw was made: a training
        forward without the loop's ``resample`` is a broken step, not a mask."""
        if self.parametrization != "hard_concrete":
            raise ValueError("only a hard_concrete gate samples its mask")
        if self._draw is None:
            raise RuntimeError(
                "a hard_concrete gate in training mode has no draw for this step — "
                "the train loop calls Gate.resample(generator) once per optimizer "
                "step before the forward"
            )
        u = self._draw
        s = torch.sigmoid(
            (torch.log(u) - torch.log1p(-u) + self.theta) / self.temperature
        )
        return self._stretched(s)

    def expected_l0(self) -> torch.Tensor:
        """The expected kept fraction per unit of a ``hard_concrete`` gate — the
        ``l0`` penalty's quantity (§2.11): Louizos et al.'s closed form
        ``P(mask ≠ 0) = σ(θ − β · log(−γ/ζ))`` (eq. 12). ``hard_concrete`` only:
        under a deterministic map the relaxed mask is itself the kept
        probability and its mean is the ``l1`` term, so ``l0`` there would be a
        second spelling of ``l1`` — refused at validation (rule 4) and here."""
        if self.parametrization != "hard_concrete":
            raise ValueError(
                "only a hard_concrete gate has an expected L0; the relaxed mask of a "
                f"{self.parametrization!r} gate is deterministic and its mean is 'l1'"
            )
        assert self.stretch is not None
        lo, hi = self.stretch
        return torch.sigmoid(self.theta - self.temperature * math.log(-lo / hi))

    def hard_threshold(self) -> float:
        """The value of ``theta`` above which a unit is kept in eval mode:
        ``0`` under ``sigmoid``; ``½`` under ``clamp``; under ``hard_concrete``
        the θ whose stretched-and-clipped ``σ(θ)`` crosses ½ —
        ``logit((½ − γ) / (ζ − γ))``, exactly ``0`` at the default (symmetric)
        stretch. The arithmetic lives in
        [`hard_concrete_threshold`][causalab.protocol.schema.featurizers.hard_concrete_threshold], shared with
        ``analysis.random_mask`` so a control's count is the fit's own."""
        if self.parametrization == "budget":
            raise ValueError(
                "a budget gate has no threshold — theta is a ranking, read out at "
                "a count (k_schedule.eval in a fit, top_k on a loaded gate)"
            )
        if self.parametrization == "boundary":
            raise ValueError(
                "a boundary gate has no per-unit threshold — its split is the "
                "prefix i < β of the coordinate index, read through hard_mask()"
            )
        if self.parametrization == "clamp":
            return 0.5
        if self.parametrization == "hard_concrete":
            assert self.stretch is not None
            return hard_concrete_threshold(self.stretch)
        return 0.0

    def hard_mask(self) -> torch.Tensor:
        """The eval-mode split over ``theta``'s units, as a 0/1 tensor of
        ``theta``'s dtype: ``θ > 0`` under ``sigmoid`` (and ``hard_concrete``
        at its default stretch), ``θ > ½`` under ``clamp`` — the one number a
        localization claim is about ([`hard_threshold`][]`). Under a
        [`top_k`][] readout the split is the ``top_k`` first units of
        [`ranking`][] instead: the same ``theta``, cut at a count rather
        than at the map's threshold."""
        self._require_linked()
        if self.pool is not None:
            # the cut is through the POOLED ranking: this member keeps the
            # units whose pooled rank is below the pool's cut — a budget pool's
            # cut, or the one `top_k` a pooled readout of loaded gates names
            return self.pool.hard_for(self)
        if self.parametrization == "boundary":
            # the soft mask at ½ — σ((β − i)/T) > ½ ⟺ i < β = θ·width — the
            # first ⌈β⌉ coordinates; the same rule every map's hard split
            # follows, so `forward: hard` and eval agree to the coordinate
            return (self._index() < self.theta * self.width).to(self.theta.dtype)
        cut = self.eval_k() if self.parametrization == "budget" else self.top_k
        if cut is not None:
            mask = torch.zeros_like(self.theta)
            mask.view(-1)[self.ranking()[:cut]] = 1.0
            return mask
        return (self.theta > self.hard_threshold()).to(self.theta.dtype)

    def ranking(self) -> torch.Tensor:
        """``theta``'s units (flat indices) from the most to the least kept —
        the object a top-k readout cuts and a ``rank`` save records (§2.12).
        Ordered by ``theta`` itself: every map's relaxed mask is monotone in
        ``θ`` (``σ(θ/T)``, ``θ``, the stretched-and-clipped ``σ(θ)``), so the
        order is the soft mask's, and ranking the parameter rather than the
        mask keeps units the clip has saturated to exactly 0 or 1 apart.
        Ties break toward the lower index, so the cut is a function of
        ``theta`` alone. A ``boundary`` gate has none: its order *is* the
        coordinate index, and a one-entry θ ranked would silently read as one
        unit."""
        if self.parametrization == "boundary":
            raise ValueError(
                "a boundary gate has no ranking — its coordinates are already in "
                "order (the rotation's columns, the PCA components) and its split "
                "is the prefix i < β, so there is nothing to rank, cut at top_k "
                "or save as rank"
            )
        return _stable_ranking(self.theta.detach().view(-1))

    def rank(self) -> torch.Tensor:
        """Each unit's position in [`ranking`][] (``0`` = kept first), in
        ``theta``'s layout."""
        order = self.ranking()
        rank = torch.empty_like(order)
        rank[order] = torch.arange(order.numel(), device=order.device)
        return rank.view(self.theta.shape)

    def project(self) -> None:  # type: ignore[override]
        """After every optimizer step (the loop's one post-step hook): a
        ``clamp`` gate back onto ``[0, 1]``; then the dead-unit bookkeeping
        (§2.5 ``dead``). The hard split is read *after* the projection so a
        clamp gate's streak counts what its mask does. A frozen unit's
        ``theta`` is restored from the photograph taken when it froze — the
        step that just happened is undone for that unit, and so is every
        later one — which is what makes the freeze hold under Adam, whose
        momentum would keep moving a unit whose gradient was merely zeroed."""
        with torch.no_grad():
            if self.parametrization == "clamp":
                self.theta.clamp_(0.0, 1.0)
            if self.parametrization == "boundary":
                # the fraction into [0, 1]: the readout ⌈θ·width⌉ stays a count
                # of the coordinates the gate has. No dead-unit bookkeeping —
                # the buffers are per θ entry and a boundary has no unit to freeze
                self.theta.clamp_(0.0, 1.0)
                return
            # "off" is the eval-mode split's complement, read through
            # `hard_mask` rather than the threshold: a budget gate has no
            # threshold (its split is the cut at `k_schedule.eval`), and under
            # the threshold maps the two readings are the same tensor
            off = self.hard_mask() == 0
            self._ever_off |= off
            if self.freeze_after is None:
                return
            self._off_streak = torch.where(
                off, self._off_streak + 1, torch.zeros_like(self._off_streak)
            )
            newly = (self._off_streak >= self.freeze_after) & ~self._frozen
            # branch-free on purpose: with nothing newly frozen the photograph
            # and the mask are unchanged, with nothing frozen theta is copied
            # onto itself — the same values a conditional would write, without
            # two device→host reads per gate per update
            self._frozen_theta = torch.where(
                newly, self.theta.detach(), self._frozen_theta
            )
            self._frozen |= newly
            self.theta.copy_(torch.where(self._frozen, self._frozen_theta, self.theta))

    def dead_diagnostics(self) -> dict[str, Any]:
        """What the dead-unit bookkeeping can say at the end of a fit, for
        ``fit_diagnostics.json``: the rule authored (as authored), how many
        units are frozen, and ``reawakened_units`` — units that were hard-off
        after some step and are kept by the final hard mask. The last is
        bookkept under every rule and none: it is the observable a ``leak``
        exists to move, and a frozen gate reports exactly ``0.0``. A
        ``boundary`` gate reports nothing here: one β has no unit to freeze,
        and ``dead`` is refused under it."""
        if self.parametrization == "boundary":
            return {}
        with torch.no_grad():
            kept = self.hard_mask().bool()
            out: dict[str, Any] = {
                "frozen_units": float(self._frozen.sum()),
                "reawakened_units": float((self._ever_off & kept).sum()),
            }
        if self.freeze_after is not None:
            out["dead"] = {"freeze_after": self.freeze_after}
        elif self.leak is not None:
            out["dead"] = {"leak": self.leak}
        return out

    def _mask(self, routing: torch.Tensor | None = None) -> torch.Tensor:
        """The mask over ``x``'s coordinates: the table mask, looked up per
        routed slot when the gate is expert-keyed."""
        return self._route(self._table_mask(), routing)

    def _table_mask(self) -> torch.Tensor:
        """The mask over the gate's own units, expanded over a ``head`` /
        ``site`` group's coordinates — everything about the mask that does
        not depend on the activation it is applied to, which is what a
        [`featurizer_cache`][] scope shares across the step's accesses."""
        self._require_linked()
        if self.frozen_mask is not None:
            # a phase's photograph: 0/1 already, no gradient — the split below
            # has nothing to do to it, so it is not applied
            mask = self.frozen_mask.to(self.theta.dtype)
        else:
            if self.training and self.parametrization == "hard_concrete":
                mask = self.sampled_mask()
            elif self.training and self.parametrization == "budget":
                k = self._k if self.pool is None else self.pool.k
                if k is None:
                    raise RuntimeError(
                        "a budget gate in training mode has no budget for this step "
                        "— the train loop calls Gate.resample(generator) (or, in a "
                        "pool, BudgetPool.resample) once per optimizer step before "
                        "the forward"
                    )
                mask = self.budget_mask(k)
            elif self.training or not self.hard_eval:
                mask = self.soft_mask()
            else:
                mask = self.hard_mask()
            if self.training and self.forward_mask == "hard":
                # §2.5 forward/backward split: the value is the training mask
                # thresholded at ½ (the map's own split: θ > 0 under sigmoid,
                # θ > ½ under clamp, the sample's or the shifted mask's ½),
                # the gradient the map's — `hard + (soft − soft.detach())`
                mask = (mask > 0.5).to(mask.dtype) + (mask - mask.detach())
        if self.training and self.leak is not None and self.theta.requires_grad:
            # §2.5 `dead.leak`: a gradient leak, not a value floor — the forward
            # mask is the map's own, the backward sees ∂m/∂θ + ε, so a unit
            # saturated at the zero pole still gets ε·∂L/∂m and can climb back;
            # the eval split (and the value the loss sees) is untouched
            mask = mask + self.leak * (self.theta - self.theta.detach())
        if self.group in ("head", "site"):
            # one entry per group, each `groups[1]` coordinates wide; under
            # `site` that is a single entry over the whole width
            assert self.groups is not None
            mask = mask.repeat_interleave(self.groups[1])
        return mask

    def _route(self, mask: torch.Tensor, routing: torch.Tensor | None) -> torch.Tensor:
        """An expert-keyed gate's table rows at the slots ``routing`` names;
        every other gate's mask is already over ``x``'s coordinates."""
        if self.group == "expert_neuron":
            assert self.groups is not None
            if routing is None:
                raise ProtocolError(
                    "P2",
                    "an expert-keyed gate (group 'expert_neuron') needs the "
                    "routing table beside the activation — its parameters are "
                    "looked up per slot through expert_idx, and this path handed "
                    "it none",
                )
            top_k = self.width // self.groups[1]
            if routing.shape[-1] != top_k:
                raise ProtocolError(
                    "P2",
                    f"an expert-keyed gate over {top_k} routed slots was handed a "
                    f"routing table with {routing.shape[-1]} — the two come from "
                    "one tap and cannot disagree",
                )
            # (..., top_k) expert ids -> (..., top_k, d_expert) rows of the
            # table -> the token-major (..., top_k · d_expert) the site has
            mask = mask[routing.long()].reshape(*routing.shape[:-1], self.width)
        return mask

    def featurize(  # type: ignore[override]
        self, x: torch.Tensor, routing: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # the table mask once per scope, in `theta`'s dtype; the routing
        # lookup and the cast to `x`'s dtype stay per call — the lookup is
        # the one part that depends on the activation, and casting *after* it
        # keeps the lookup's backward accumulating in `theta`'s dtype as it
        # always has: `mask[routing]`'s backward is a scatter-add in the
        # indexed tensor's dtype, and a bf16 sum of the slots' cotangents is
        # not the fp32 one
        mask = self._route(self._shared_table(), routing).to(x.dtype)
        if self.axis == "position":
            # one scalar per addressed position, over every coordinate: the
            # value is `(…, positions, width)` and θ runs over `positions`
            if x.dim() < 3 or x.shape[-2] != self.width:
                raise ProtocolError(
                    "P2",
                    f"a position gate over {self.width} positions was handed a value "
                    f"of shape {tuple(x.shape)} — it applies to a (rows, positions, "
                    "width) window whose positions axis is its own; a ragged window "
                    "arrives flattened and cannot carry one, and a window of another "
                    "length is another gate's (§2.5 axis)",
                )
            mask = mask.unsqueeze(-1)
        return mask * x, (1.0 - mask) * x

    def _shared_table(self) -> torch.Tensor:
        """The table mask, in ``theta``'s dtype, evaluated once per open
        [`featurizer_cache`][] scope where that is exact: shared as a value
        under ``no_grad``; under grad through one replay node per access when
        the mask's graph reaches ``theta`` by a single edge and nothing else
        trainable (`_Shared.single`) — a ``leak`` or a pool makes each
        grad access compute its own instead, the value still shared under
        ``no_grad``. Like the subspace's rotation the evaluation is made
        whatever the grad mode of the access that makes it, so a no-grad
        access followed by a grad one costs one evaluation, not two."""
        entries = _SCOPE.entries
        compute = self._table_mask
        if entries is None:
            return compute()
        # the graph entry; the no-grad value `_once` keeps sits under its own tag
        key = (self, "mask graph", self.training)
        shared = entries.get(key)
        if shared is None:
            shared, probe = _Shared.single(self.theta, compute)
            if shared is None:
                entries[key] = _UNSHARED
                if torch.is_grad_enabled():
                    return probe  # this access's own graph; the rest compute theirs
                # the probe is this no-grad access's value: kept for the next
                return _once(self, "mask value", lambda: probe.detach(), trainable=True)
            entries[key] = shared
        if shared is _UNSHARED:
            # what one replay cannot serve is still one exact value under
            # no_grad; a grad access computes its own
            return _once(self, "mask value", compute, trainable=True)
        return shared.access()

    def inverse(self, f: torch.Tensor, err: torch.Tensor | None) -> torch.Tensor:
        return f if err is None else f + err

    def slot_params(self) -> dict[str, torch.Tensor]:
        return {"theta": self.theta}

    def identity_fields(self) -> dict[str, Any]:
        self._require_linked()
        fields = dict(self.init_identity)
        if self.stretch is not None:
            # the hard split depends on the stretch (`hard_threshold`), so a
            # reader of the bundle — `analysis.random_mask` — needs it
            fields["stretch"] = json.dumps(list(self.stretch))
        if self.pool is not None:
            # a pooled member's θ is a ranking only relative to its co-members
            # (§2.5 `pool`): the name is compared by the loader, the unit
            # count is provenance a reader can size the pool's cut by
            fields["pool"] = self.pool.name
            fields["pool_units"] = str(self.pool.units)
        if self.axis is not None:
            # a position gate's θ is one entry per token position — a bundle
            # of it is not a mask over coordinates (§2.5 axis)
            fields["axis"] = self.axis
        if self.forward_mask is not None:
            # provenance, not identity: a straight-through fit's θ reads out
            # through the same hard split as the map's own, so an apply under
            # either spelling is the same mask (§2.5)
            fields["forward"] = self.forward_mask
        return fields

    @classmethod
    def from_theta(
        cls,
        theta: torch.Tensor,
        *,
        group: str | None = None,
        groups: tuple[int, int] | None = None,
        width: int | None = None,
        parametrization: str = "sigmoid",
        temperature: float | None = None,
        stretch: tuple[float, float] | None = None,
        top_k: int | None = None,
        pool: str | None = None,
        axis: str | None = None,
        forward: str | None = None,
    ) -> "Gate":
        """A gate reconstituted from a fitted ``theta`` (§2.5 ``file_path``).

        The number a DBM fit *reports* is scored through its eval-mode hard
        split, so an apply document only reproduces the fit if the reloaded
        stage is the same object a trained gate is after ``stage.eval()``:
        same ``theta``, same ``θ > 0`` mask. ``theta`` is therefore copied
        verbatim — no re-init, no thresholding here — and left untrainable,
        because applying a mask is not resuming a fit (a ``file_path``
        featurizer may not appear in ``train.params``).

        ``groups`` is the map the gate is applied through and ``theta`` must
        hold exactly the parameters that map implies (``gate_param_shape``);
        the caller has already checked it against the map the bundle was
        stamped with. ``width`` is the site width, which only an expert-keyed
        gate cannot recover from its table (the routed slot count is the
        site's, not the table's); for the other gates it is derived when
        omitted.

        ``top_k`` (§2.5) replaces the map's threshold with a count: the hard
        mask becomes the ``top_k`` largest units of ``theta``
        ([`ranking`][]), ``0`` keeps nothing. A count above the unit count
        names units the gate does not have and is refused; the caller
        reports it with the document's words.
        """
        if width is None:
            if group == "expert_neuron":
                raise ValueError("an expert-keyed gate needs the site width")
            if parametrization == "boundary":
                raise ValueError(
                    "a boundary gate needs the site width: its theta is one β, "
                    "not one entry per coordinate"
                )
            width = int(theta.numel()) if groups is None else groups[0] * groups[1]
        if parametrization == "boundary" and top_k is not None:
            raise ValueError(
                "a boundary gate is read out at the prefix i < β its β names — "
                "its coordinates have no ranking for a top_k to cut (§2.5)"
            )
        shape = gate_param_shape(group, groups, width, parametrization=parametrization)
        count = int(theta.numel())
        if count != math.prod(shape):
            raise ValueError(
                f"a gate over {list(shape)} needs {math.prod(shape)} parameters, "
                f"got {count}"
            )
        if top_k is not None and top_k < 0:
            raise ValueError(
                f"top_k={top_k} — a top-k readout keeps a non-negative count"
            )
        if top_k is not None and pool is None and top_k > count:
            # in a pool the cut is the POOLED count, bounded by the pool's
            # units at the link (`link_budget_pools`), not by this member's
            raise ValueError(
                f"top_k={top_k} on a gate of {count} units — a top-k readout keeps "
                f"between 0 and {count} of them"
            )
        if parametrization == "budget" and top_k is None:
            raise ValueError(
                "a budget gate's theta is a ranking with no threshold — a loaded "
                "one is read out at 'top_k' (§2.5)"
            )
        if pool is not None and top_k is None:
            raise ValueError(
                "a loaded gate in a pool is read out at one pooled 'top_k' — a "
                "pool of loaded gates without a cut has nothing to share (§2.5)"
            )
        gate = cls(
            width,
            group=group,
            groups=groups,
            parametrization=parametrization,
            temperature=temperature,
            stretch=stretch,
            pool=pool,
            axis=axis,
            forward=forward,
        )
        gate.theta = torch.nn.Parameter(
            theta.detach().clone().reshape(shape), requires_grad=False
        )
        gate.top_k = top_k
        return gate


# --------------------------------------------------------------------------- #
# the budget pool (§2.5 `pool`)
# --------------------------------------------------------------------------- #


def _schedule_of(schedule: Mapping[str, Any]) -> str:
    """What a schedule's numbers count (§2.5 ``k_schedule.of``): ``patched``
    — units that take the counterfactual, the gate's own count — unless the
    schedule says ``kept``."""
    return str(schedule.get("of", "patched"))


def _as_patched(schedule: Mapping[str, Any], value: int, units: int) -> int:
    """A schedule number as a **patched** count: itself, or its complement
    against ``units`` when the schedule counts kept units. Every internal reader
    of a budget works in patched units, so a ``kept`` schedule is complemented
    exactly once, here."""
    return units - int(value) if _schedule_of(schedule) == "kept" else int(value)


def _draw_from_schedule(schedule: Mapping[str, Any], generator: torch.Generator) -> int:
    """One draw from a ``k_schedule`` (§2.5), in the schedule's own units:
    ``k`` itself when fixed; an integer uniform on ``[low, high]``; or
    ``round(exp(U(log low, log high)))`` clipped to the bounds — a
    log-uniform curriculum, which spends as many steps between 1 and 2 units as
    between 24 and 48, so the ranking is learned at every scale. One scalar
    from the fit's own generator, so the sequence of budgets is a function of
    ``train.seed`` — and the same sequence whether the gate budgets alone or
    for a pool, since a pool draws exactly one scalar per step too."""
    kind = schedule["kind"]
    if kind == "fixed":
        return int(schedule["k"])
    low, high = int(schedule["low"]), int(schedule["high"])
    u = float(torch.rand((), generator=generator, dtype=torch.float64))
    if kind == "uniform":
        return min(high, low + int(u * (high - low + 1)))
    k = int(round(math.exp(math.log(low) + u * (math.log(high) - math.log(low)))))
    return max(low, min(high, k))


def _stable_ranking(theta: torch.Tensor) -> torch.Tensor:
    """Flat unit indices from the largest θ down, ties toward the lower index,
    **sorted on CPU**: ``stable=True`` is load-bearing (a pooled ranking over
    separately fitted bundles has real ties), and a tie-break that varied by
    backend would make one document cut differently on different machines.
    The order comes back on θ's device."""
    order = torch.argsort(theta.detach().cpu(), descending=True, stable=True)
    return order.to(theta.device)


def _solve_shift(theta: torch.Tensor, k: int) -> float:
    """The scalar ``c_k`` with ``Σ σ(θ + c_k) = k`` over a flat float64
    ``theta``, by bisection on a bracket that contains it:
    ``Σσ`` is strictly increasing in ``c`` from 0 to the unit count, so for
    ``0 < k < units`` the root is unique and 60 halvings of
    ``[−max θ − 40, −min θ + 40]`` pin it well below float32 resolution; at the
    poles ``k = 0`` / ``k = units`` there is no finite root and the bracket's
    end is returned (a mask within ``1e−17`` of the pole)."""
    units = theta.numel()
    if not 0 <= k <= units:
        raise ValueError(f"a budget of {k} on a gate of {units} units")
    lo = -float(theta.max()) - 40.0
    hi = -float(theta.min()) + 40.0
    if k == 0:
        return lo
    if k == units:
        return hi
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if float(torch.sigmoid(theta + mid).sum()) < k:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


class BudgetPool:
    """One ranking over several gates (§2.5 ``pool``), in two roles. On a
    **fit** it is the one budget a set of ``budget`` gates share: one
    ``k`` per optimizer step, one shift ``c_k`` solved over the **concatenation**
    of every member's θ, one ranking cut at one count. It owns no parameters
    and is never saved — a bundle stays one gate's θ, stamped with the pool's
    name and unit count — and it exists so that a mask over units that live at
    different sites (every head, every MLP block and the embedding of a model)
    is *one* budget fit over ``N`` units rather than ``2L + 1`` fits over
    their own budgets: the draw ``k ~ LogUniform{1, N − 1}`` is over ``N``.

    On a **readout** — every member loaded through ``file_path`` with one
    ``top_k`` — it is the pooled cut alone: the members' θ concatenated, ranked
    once, and each member keeps the units whose pooled rank falls below
    ``top_k``. That role takes gates of any map, since a ranking method's
    curve (MIB's node-level CPR) cuts the *joint* ranking of every unit of a
    model, whether the units were fitted as one budget or as one L1-penalised
    sigmoid mask; nothing is drawn and no shift is solved.

    Members are the point's gates authoring this ``pool`` name, in featurizer
    name order, with one ``k_schedule`` (a fit, budget gates only) or one
    ``top_k`` (loaded, any map), linked by [`link_budget_pools`][] once the
    executor has built them all. Every quantity below is in **patched** units — the count a member's
    [`Gate.hard_mask`][] sums to — with a ``kept`` schedule complemented once
    against the pool's units, exactly as a lone gate complements against its
    own.

    The implicit gradient of the shared shift is the pool's: ``w_i = σ'_i / Σ_j
    σ'_j`` normalised over *every* member's unit, so a member's mask carries a
    gradient into its co-members' θ through the correction term — which is the
    true derivative of ``c_k`` under ``Σ_pool σ(θ + c_k) = k``."""

    def __init__(self, name: str, members: Sequence["Gate"]) -> None:
        if not members:
            raise ValueError(f"pool {name!r} has no members")
        self.name = name
        self.members: tuple[Gate, ...] = tuple(members)
        #: this step's budget, patched units ([`resample`][]), or ``None``
        self.k: int | None = None
        #: the shift of the last solve, so [`Gate.soft_mask`][] on a member
        #: reports the mask the last forward used
        self.last_shift: torch.Tensor | None = None
        #: memo of the quantities every member asks for in one forward — the
        #: pooled θ, the shift at a budget, the pooled ranks — keyed by the
        #: members' ``theta._version`` counters (an optimizer step bumps them),
        #: so M members share one solve and one argsort, exactly: θ cannot
        #: move inside a forward
        self._memo: dict[tuple[Any, ...], Any] = {}

    @property
    def units(self) -> int:
        return sum(member.theta.numel() for member in self.members)

    def _version(self) -> tuple[int, ...]:
        return tuple(int(member.theta._version) for member in self.members)

    def _memoized(self, key: tuple[Any, ...], compute: "Callable[[], Any]") -> Any:
        version = self._version()
        if self._memo and next(iter(self._memo))[0] != version:
            self._memo.clear()  # θ moved: everything below is stale
        full = (version, *key)
        if full not in self._memo:
            self._memo[full] = compute()
        return self._memo[full]

    def theta(self) -> torch.Tensor:
        """The pooled ranking's parameter: every member's θ, flat, in member
        order — the object one shift is solved over and one cut is made in.
        Not memoized: it carries the members' autograd graph, and the caller
        that needs the detached copy memoizes that."""
        return torch.cat([member.theta.view(-1) for member in self.members])

    def _theta_detached(self) -> torch.Tensor:
        return self._memoized(
            ("theta",),
            lambda: torch.cat([m.theta.detach().view(-1) for m in self.members]),
        )

    def _schedule(self) -> dict[str, Any]:
        schedule = self.members[0].k_schedule
        if schedule is None:
            raise ValueError(
                f"pool {self.name!r} is loaded (no k_schedule) and is read "
                "out at its members' top_k"
            )
        return schedule

    def resample(self, generator: torch.Generator) -> None:
        """The pool's one draw for the step (`_draw_from_schedule`),
        complemented against the pool's units under a ``kept`` schedule."""
        schedule = self._schedule()
        self.k = _as_patched(
            schedule, _draw_from_schedule(schedule, generator), self.units
        )

    def eval_k(self) -> int:
        """The pooled count the eval-mode split is cut at: the members' one
        ``top_k`` when loaded, else the schedule's ``eval`` (``k`` when fixed),
        complemented under ``kept``."""
        tops = {member.top_k for member in self.members}
        if tops != {None}:
            if len(tops) != 1:
                raise ValueError(
                    f"pool {self.name!r}: its members are read out at "
                    f"different top_k values {sorted(t for t in tops if t is not None)} "
                    "— one pool, one cut through one ranking"
                )
            return int(next(iter(tops)))
        schedule = self._schedule()
        raw = int(schedule["eval"]) if "eval" in schedule else int(schedule["k"])
        return _as_patched(schedule, raw, self.units)

    def shift(self, k: int) -> torch.Tensor:
        """``c_k`` over the pooled θ (`_solve_shift`), as a scalar tensor
        on the members' dtype and device."""
        anchor = self.members[0].theta

        def solve() -> torch.Tensor:
            # on CPU in float64: device-independent to the bit (a seeded fit is
            # the same fit on CPU, CUDA and MPS) and free of the 60 device syncs
            # the on-device loop paid; MPS has no float64 at all
            theta = self._theta_detached().cpu().to(torch.float64)
            return torch.tensor(
                _solve_shift(theta, k), dtype=anchor.dtype, device=anchor.device
            )

        return self._memoized(("shift", int(k)), solve)

    def mask_for(self, member: "Gate", k: int) -> torch.Tensor:
        """``member``'s training mask at the pooled budget ``k``: ``σ(θ_m + c_k)``
        with the shift solved over the pool and its implicit gradient attached
        over the pool (``Gate.budget_mask``'s rule with the sum over every
        member's units), unless the pool's ``stop_grad_shift``."""
        shift = self.shift(k)
        self.last_shift = shift
        if not member.stop_grad_shift:
            pooled = self.theta()
            with torch.no_grad():
                s = torch.sigmoid(pooled + shift)
                weights = s * (1.0 - s)
                weights = weights / weights.sum().clamp_min(1e-30)
            correction = (weights * pooled).sum()
            shift = shift + correction.detach() - correction
        theta = member.theta.view(-1)
        return torch.sigmoid(theta + shift).view(member.theta.shape)

    def ranking(self) -> torch.Tensor:
        """Pooled unit indices from the most to the least kept (``Gate.ranking``
        over the concatenation, ties toward the lower pooled index)."""
        return self._memoized(
            ("ranking",), lambda: _stable_ranking(self._theta_detached())
        )

    def pooled_rank(self) -> torch.Tensor:
        """Each pooled unit's position in [`ranking`][] (``0`` = kept first)."""

        def invert() -> torch.Tensor:
            order = self.ranking()
            rank = torch.empty_like(order)
            rank[order] = torch.arange(order.numel(), device=order.device)
            return rank

        return self._memoized(("rank",), invert)

    def offset(self, member: "Gate") -> int:
        """Where ``member``'s units start in the pooled layout."""
        start = 0
        for candidate in self.members:
            if candidate is member:
                return start
            start += candidate.theta.numel()
        raise ValueError(f"gate is not a member of pool {self.name!r}")

    def member_rank(self, member: "Gate") -> torch.Tensor:
        """``member``'s units' pooled ranks, in θ's layout."""
        start = self.offset(member)
        count = member.theta.numel()
        return self.pooled_rank()[start : start + count].view(member.theta.shape)

    def hard_for(self, member: "Gate") -> torch.Tensor:
        """``member``'s eval-mode split: the units whose pooled rank is below the
        pool's cut ([`eval_k`][]), as 0/1 in θ's dtype."""
        return (self.member_rank(member) < self.eval_k()).to(member.theta.dtype)


def link_budget_pools(
    featurizers: Mapping[str, FeaturizerSpec],
    stages: Mapping[str, Stage],
    build: Callable[[str], Stage],
) -> None:
    """Attach one [`BudgetPool`][] to every gate of each ``pool`` the
    document authors (§2.5), building the members not yet built through
    ``build`` — the executor's own ``stage(name)`` — so a pool is complete
    before any member's mask is computed. Idempotent: a pool whose members all
    carry the same pool object is left alone, which is what makes the recursion
    through ``build`` (a sibling's build calls back here) terminate.

    Refuses (P2) a pool that is not a pool: a fitted member under another map
    (only the budget map has a budget to share; a *loaded* member of any map
    joins a pooled readout), members that disagree on ``k_schedule``,
    ``stop_grad_shift`` or ``top_k``, a mix of fitted and loaded members, or a
    schedule number above the pool's units."""
    by_pool: dict[str, list[str]] = {}
    for name in sorted(featurizers):
        spec = featurizers[name]
        if spec.kind == "gate" and isinstance(spec.pool, str):
            by_pool.setdefault(spec.pool, []).append(name)
    linking = _LINKING.setdefault(id(stages), set())
    for pool_name, names in by_pool.items():
        if pool_name in linking:
            continue  # a sibling's build re-entered us: the outer frame finishes
        linking.add(pool_name)
        try:
            _link_pool(pool_name, names, stages, build)
        finally:
            linking.discard(pool_name)
            if not linking:
                _LINKING.pop(id(stages), None)


#: Pools being linked right now, per stage map (``id(stages)``): a member's
#: build calls back into [`link_budget_pools`][], which would otherwise
#: recurse once per member — depth ``M`` and ``O(M·F)`` re-scans on a 53-gate
#: pool. With the guard the nested call returns at once and the outer frame,
#: which is building the members in order anyway, links the pool: depth 2.
_LINKING: dict[int, set[str]] = {}


def _link_pool(
    pool_name: str,
    names: Sequence[str],
    stages: Mapping[str, Stage],
    build: Callable[[str], Stage],
) -> None:
    members: list[Gate] = []
    for name in names:
        stage = stages[name] if name in stages else build(name)
        if not isinstance(stage, Gate) or (
            stage.parametrization != "budget" and stage.top_k is None
        ):
            raise ProtocolError(
                "P2",
                f"featurizer {name!r} is in pool {pool_name!r} but is neither a "
                "budget gate nor a loaded gate read out at top_k — a pool "
                "shares one budget (which only the budget map draws) or one "
                "pooled cut (§2.5)",
            )
        members.append(stage)
    pools = {id(member.pool) for member in members if member.pool is not None}
    if len(pools) == 1 and all(member.pool is not None for member in members):
        return  # linked already
    loaded = {member.k_schedule is None for member in members}
    if len(loaded) != 1:
        raise ProtocolError(
            "P2",
            f"pool {pool_name!r} mixes fitted and loaded gates — a pool is one "
            "ranking, fitted together or read out together (§2.5)",
        )
    schedules = {json.dumps(member.k_schedule, sort_keys=True) for member in members}
    stops = {member.stop_grad_shift for member in members}
    tops = {member.top_k for member in members}
    maps = {member.parametrization for member in members}
    if len(schedules) != 1 or len(stops) != 1 or len(tops) != 1 or len(maps) != 1:
        raise ProtocolError(
            "P2",
            f"pool {pool_name!r}: its members {list(names)} disagree on "
            "parametrization, k_schedule, stop_grad_shift or top_k — one pool "
            "is one ranking on one scale, draws one budget and is cut at one "
            "count (§2.5)",
        )
    pool = BudgetPool(pool_name, members)
    for name, member in zip(names, members):
        stamped = member.stamped_pool_units
        if stamped is not None and stamped != pool.units:
            raise ProtocolError(
                "P2",
                f"featurizer {name!r}: its bundle was fitted in a pool of "
                f"{stamped} units but pool {pool_name!r} here has {pool.units} "
                "— a pooled theta ranks against exactly its co-members, and "
                "these are not them (§2.5, rule 15)",
            )
    top_k = members[0].top_k
    if top_k is not None and top_k > pool.units:
        raise ProtocolError(
            "P2",
            f"pool {pool_name!r}: top_k={top_k} but the pool has {pool.units} "
            "units — a pooled cut keeps at most every unit (§2.5)",
        )
    schedule = members[0].k_schedule
    if schedule is not None:
        for key in ("k", "low", "high", "eval"):
            if key in schedule and int(schedule[key]) > pool.units:
                raise ProtocolError(
                    "P2",
                    f"pool {pool_name!r}: k_schedule.{key}={schedule[key]} but "
                    f"the pool has {pool.units} units — a budget keeps at most "
                    "every unit (§2.5)",
                )
    for member in members:
        member.pool = pool


def _checked_k_schedule(schedule: Mapping[str, Any]) -> dict[str, Any]:
    """The schedule as the gate keeps it: kind, ``k`` or the bounds, and the
    eval cut — every value a non-negative integer, the bounds ordered,
    ``log_uniform`` from 1, a sampled kind naming its ``eval``."""
    kind = schedule.get("kind")
    if kind not in ("fixed", "uniform", "log_uniform"):
        raise ValueError(f"unknown k_schedule kind {kind!r}")
    out: dict[str, Any] = {"kind": kind}
    if "of" in schedule:
        if schedule["of"] not in ("patched", "kept"):
            raise ValueError(
                f"k_schedule.of is 'patched' or 'kept', got {schedule['of']!r}"
            )
        out["of"] = schedule["of"]
    keys = ("k",) if kind == "fixed" else ("low", "high")
    for key in (*keys, "eval"):
        if key in schedule:
            value = schedule[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(
                    f"k_schedule.{key} is a non-negative integer, got {value!r}"
                )
            out[key] = value
        elif key != "eval":
            raise ValueError(f"a {kind} k_schedule needs {key!r}")
    if kind != "fixed":
        if out["low"] > out["high"]:
            raise ValueError("k_schedule bounds are ordered")
        if kind == "log_uniform" and out["low"] < 1:
            raise ValueError("a log_uniform k_schedule starts at 1 or above")
        if "eval" not in out:
            raise ValueError(
                f"a {kind} k_schedule names 'eval', the cut the hard mask is read at"
            )
    return out


def _gate_parametrization(spec: FeaturizerSpec) -> str:
    """A gate spec's theta→mask map, ``sigmoid`` when unauthored (§2.5)."""
    return spec.parametrization if isinstance(spec.parametrization, str) else "sigmoid"


def gate_poles(
    parametrization: str, stretch: tuple[float, float] | None
) -> tuple[float, float]:
    """``(dropped, kept)`` — the two values of ``theta`` a decisive start is
    written on under one θ→mask map (§2.5): ``(0, 1)`` under ``clamp``, whose
    parameter lives on the unit interval and splits at ½; one unit either
    side of the hard threshold under ``sigmoid`` (``∓1`` around 0) and
    ``hard_concrete`` (around ``logit((½−γ)/(ζ−γ))``, the same arithmetic
    [`Gate.hard_threshold`][] uses). The convention ``analysis.random_mask``
    writes its controls on, stated once so a ranking start and a size-matched
    control are the same kind of object."""
    if parametrization == "clamp":
        return (0.0, 1.0)
    threshold = 0.0
    if parametrization == "hard_concrete":
        lo, hi = HARD_CONCRETE_STRETCH if stretch is None else stretch
        threshold = hard_concrete_threshold((float(lo), float(hi)))
    return (threshold - 1.0, threshold + 1.0)
