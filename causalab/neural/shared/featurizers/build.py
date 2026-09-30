"""Build and compose featurizer stages on the run device.

``FeaturizerStack`` tracks stage widths and composition. Builders check
saved artifact identity and initialize gates or subspaces from the
specified parameters, scores, or basis.
"""

from __future__ import annotations

import dataclasses
import json
import math
from typing import Any, Mapping, Sequence

import torch

from causalab.protocol.bundles import entry_selection
from causalab.protocol.rules.data import check_start_site
from causalab.protocol.rules.errors import ProtocolError, ValidationError
from causalab.protocol.registry import ModelInfo, gate_group_map, gate_param_shape
from causalab.protocol.schema import (
    FEATURIZER_SLOTS,
    FeaturizerSpec,
    HARD_CONCRETE_STRETCH,
)
from causalab.protocol.registry.shapes import FeatureShape
from causalab.neural.shared.featurizers.stages import (
    Identity,
    LoadedLinear,
    ORTHONORMAL_TOLERANCE,
    Sae,
    Stage,
    Standardize,
    Subspace,
    orthonormality_deviation,
)
from causalab.neural.shared.featurizers.gate import (
    Gate,
    _gate_parametrization,  # pyright: ignore[reportPrivateUsage]
    gate_poles,
)


@dataclasses.dataclass
class FeaturizerStack:
    """A left-to-right composition of stages with a per-stage ``err`` list
    (§2.5). ``names`` aligns with ``stages`` for train-param addressing."""

    names: tuple[str, ...]
    stages: tuple[Stage, ...]

    def featurize(
        self, x: torch.Tensor, *, routing: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, list[torch.Tensor | None]]:
        """``routing`` is the routing table gathered at the same rows and
        positions as ``x`` (``(..., top_k)`` expert ids), which only a stage
        with ``needs_routing`` consumes; every other stage ignores it."""
        errs: list[torch.Tensor | None] = []
        for stage in self.stages:
            if stage.needs_routing:
                x, err = stage.featurize(x, routing=routing)  # type: ignore[call-arg]
            else:
                x, err = stage.featurize(x)
            errs.append(err)
        return x, errs

    def inverse(
        self, f: torch.Tensor, errs: Sequence[torch.Tensor | None]
    ) -> torch.Tensor:
        for stage, err in zip(reversed(self.stages), reversed(list(errs))):
            f = stage.inverse(f, err)
        return f

    @property
    def is_identity(self) -> bool:
        return all(isinstance(s, Identity) for s in self.stages)

    @property
    def needs_routing(self) -> bool:
        """Whether any stage joins its parameters to the activation through
        the routing table (an expert-keyed gate)."""
        return any(stage.needs_routing for stage in self.stages)


def stage_output_width(spec: FeaturizerSpec, input_width: int) -> int | None:
    """The feature width a stage emits, given its input width — the chain
    rule of §2.5's composition: subspace/pca project to ``k``, the
    width-preserving kinds pass ``input_width`` through, and a loaded SAE's
    dictionary size is unknowable from the spec alone (``None``)."""
    kind = spec.kind if isinstance(spec.kind, str) else "identity"
    if kind in ("subspace", "pca"):
        return spec.k if isinstance(spec.k, int) else None
    if kind == "sae":
        return None  # the dictionary size lives in the bundle, not the spec
    return input_width


def _stage_width(stage: Stage) -> int | None:
    """The input width a built stage was sized for (for cache-reuse checks)."""
    params = stage.slot_params()
    if isinstance(stage, (Subspace, LoadedLinear)):
        weight = stage.weight if isinstance(stage, LoadedLinear) else params["weight"]
        return int(weight.shape[0])
    if isinstance(stage, Gate):
        return stage.width
    if isinstance(stage, Standardize):
        return int(stage.mu.shape[0])
    return None


def build_stack(
    ref: Any,
    specs: dict[str, FeaturizerSpec],
    *,
    width: int,
    load_tensors: Any,
    stage_cache: dict[str, Stage],
    device: str | torch.device = "cpu",
    seed: int = 0,
    coords: Mapping[str, Any] | None = None,
    site_shape: FeatureShape | None = None,
    site_component: str | None = None,
    model_info: ModelInfo | None = None,
    load_table: Any = None,
    position_width: int | None = None,
    site_records: Mapping[str, Mapping[str, Any]] | None = None,
) -> FeaturizerStack:
    """Build (or reuse from ``stage_cache``) the stack a read/write
    references. ``width`` is the SITE width; each later stage in a
    composition is sized to the *previous stage's output* (the §2.5 chain —
    a gate after a k=3 rotation is a 3-wide gate). ``load_tensors`` supplies
    loaded bundles; caching by name keeps one stage instance per declared
    featurizer, so training one featurizer updates every use site — a name
    reused at a different chain width is a contradiction and refuses.

    ``site_shape`` and ``site_component`` describe the site the chain starts
    at, and ``model_info`` the model; a grouped gate derives its map from them
    ([`gate_group_map`][causalab.protocol.registry.components.gate_group_map] — the head layout for
    ``group: head``, the expert table for ``group: expert_neuron``) and
    refuses without them, or after a stage that changed the coordinate basis
    (§5.23) — it groups the component's own coordinates, and after a rotation
    "head" names nothing; a ``standardize`` before it is per coordinate and
    legal.

    ``device`` is the run's device; stages are built on CPU and moved there
    (module docstring). The ``"cpu"`` default leaves CPU-only callers alone.

    ``seed`` is the document's featurizer-init seed (``train.seed``, 0 with no
    fit — ``executor.document_seed``). Explicit rather than read from the global
    RNG because this also runs on apply/inference paths, where a global-RNG init
    would make a rotation depend on construction order. The cache is keyed by
    name, so a cached stage built from a different seed refuses, as with width.

    A ``subspace`` spec may name its **own** ``seed`` (§2.5), which wins. That is
    what makes an *untrained* subspace a random rank-k basis a document can
    sweep — the matched-k random-subspace control, which otherwise needs a
    ``train`` block it has nothing to train.

    ``coords`` are the executing point's sweep coordinates: they select the
    matching entry of a swept bundle when the spec authored no ``entry``
    (§2.5).

    ``site_records`` maps a featurizer used at exactly one site to that
    site's record, as ArtifactIdentity stamps it (§8,
    [`single_site_featurizers`][causalab.protocol.identity.single_site_featurizers]).
    For a ``subspace`` with an ``init`` basis, the build checks the site
    recorded on the entry it selects against this record (``_init_basis``).
    The engine makes the same check for every point before any weights
    load; this one covers a caller that builds a stack directly."""
    if ref is None:
        return FeaturizerStack(names=(), stages=(Identity(),))
    chain = (ref,) if isinstance(ref, str) else tuple(ref)
    stages: list[Stage] = []
    running: int | None = width
    for index, name in enumerate(chain):
        spec = specs[name]
        # a `subspace` may author its own seed (§2.5); absent, the document's
        # seed stands, so nothing about an existing document changes
        stage_seed = spec.seed if isinstance(spec.seed, int) else seed
        groups = _gate_groups(
            name,
            spec,
            width=running,
            before=tuple(
                (
                    member,
                    specs[member].kind
                    if isinstance(specs[member].kind, str)
                    else "identity",
                )
                for member in chain[:index]
            ),
            site_shape=site_shape,
            site_component=site_component,
            model_info=model_info,
        )
        # §2.5 `axis`: a position gate is sized by the addressed window's
        # length, which the executor derives from the entry's `span`, not by
        # the feature width flowing through the chain (which it passes on)
        positional = spec.kind == "gate" and spec.axis == "position"
        stage_width = position_width if positional else running
        if positional and position_width is None:
            raise ProtocolError(
                "P2",
                f"featurizer {name!r} is a position gate but the entry using it "
                "addresses no fixed window — its `pos` must be a `span` [a, b) "
                "of two or more positions (§2.5 axis)",
            )
        if name in stage_cache:
            stage = stage_cache[name]
            built_for = _stage_width(stage)
            if (
                built_for is not None
                and stage_width is not None
                and built_for != stage_width
            ):
                raise ProtocolError(
                    "P2",
                    f"featurizer {name!r} is used at width {stage_width} here but was "
                    f"built for width {built_for} — one featurizer, one width"
                    + (
                        " (a position gate's width is its window's length)"
                        if positional
                        else ""
                    ),
                )
            built_groups = getattr(stage, "groups", None)
            if isinstance(stage, Gate) and built_groups != groups:
                raise ProtocolError(
                    "P2",
                    f"featurizer {name!r} is grouped {groups} here but the cached "
                    f"stage was built over {built_groups} — one featurizer, one "
                    "group map",
                )
            built_seed = getattr(stage, "seed", None)
            if built_seed is not None and built_seed != stage_seed:
                raise ProtocolError(
                    "P2",
                    f"featurizer {name!r} is used at init seed {stage_seed} here "
                    f"but the "
                    f"cached stage was initialised from seed {built_seed} — one "
                    "featurizer, one seed; a stage cache belongs to one point, so "
                    "two points differing in train.seed must not share one",
                )
        else:
            if stage_width is None:
                raise ProtocolError(
                    "P2",
                    f"cannot size featurizer {name!r}: the preceding stage's "
                    "output width is not derivable from its spec",
                )
            stage = _build_stage(
                name,
                spec,
                width=stage_width,
                load_tensors=load_tensors,
                seed=stage_seed,
                coords=coords,
                groups=groups,
                load_table=load_table,
                site_record=(site_records or {}).get(name),
            )
            stage.to(device)  # parameters and registered buffers alike
            # inference documents get eval semantics (a gate's hard split);
            # the train loop flips modes around its steps explicitly
            stage.eval()
            stage_cache[name] = stage
        stages.append(stage)
        if isinstance(stage, Sae):
            running = int(stage.enc.shape[1])
        elif running is not None:
            running = stage_output_width(spec, running)
    return FeaturizerStack(names=chain, stages=tuple(stages))


def _gate_groups(
    name: str,
    spec: FeaturizerSpec,
    *,
    width: int | None,
    before: Sequence[tuple[str, str]],
    site_shape: FeatureShape | None,
    site_component: str | None,
    model_info: ModelInfo | None,
) -> tuple[int, int] | None:
    """The group map a grouped gate is built over — ``(heads, head_dim)`` or
    ``(num_experts, d_expert)`` — or ``None`` for every other stage (module
    docstring, ``group``). ``before`` is ``(name, kind)`` of every stage ahead
    of this one in its chain: rule 23 already refused a grouped gate that is
    not first at load, so a refusal here is the executor holding the same line
    rather than a document problem."""
    if spec.kind != "gate" or not isinstance(spec.group, str):
        return None
    if before:
        member, kind = before[0]
        raise ProtocolError(
            "P2",
            f"featurizer {name!r} is grouped by {spec.group} but follows {member!r} "
            f"({kind!r}) in its chain — a grouped gate acts on the component's own "
            "coordinates, so it must be the first stage of its chain",
        )
    if site_shape is None or site_component is None or width is None:
        raise ProtocolError(
            "P2",
            f"featurizer {name!r} is grouped by {spec.group} but the site it is "
            "built for declares no shape to derive the groups from",
        )
    try:
        return gate_group_map(
            spec.group, site_shape, width, component=site_component, info=model_info
        )
    except ValidationError as err:
        raise ProtocolError("P2", f"featurizer {name!r}: {err.message}") from err


def _check_entry_identity(
    record: Mapping[str, Any],
    spec: FeaturizerSpec,
    what: str,
    *,
    groups: tuple[int, int] | None = None,
) -> None:
    """Refuse an entry whose stamped fit contradicts the spec that selected
    it (§2.5).

    The load-time check (``rules.data.check_loaded_featurizers``) covers a
    bundle whose entry is knowable there; when the selection is the
    executing point's — implicit matching against a swept producer — this is
    where the claim is finally tested, so "apply the k=8 fit" cannot quietly
    apply the k=32 one. Only the per-entry fields are compared: everything
    file-level was already checked at load.

    A hard-concrete gate's ``stretch`` is compared the same way in both
    directions, since the hard split's threshold is derived from it.

    A gate's grouping is compared in both directions: a bundle fitted per
    head or per expert neuron is not applicable through a per-coordinate gate,
    and the stamped ``group_map`` must be the one this site derives
    (``groups``) — the same head count and head width, or the same expert
    table, not merely the same parameter count.
    """
    for field, value in (
        ("k", spec.k),
        ("parametrization", spec.parametrization),
    ):
        if value is None or not isinstance(value, (int, str)):
            continue
        stamped = record.get(field)
        if stamped is not None and str(stamped) != str(value):
            raise ProtocolError(
                "P2",
                f"{what}: the document says {field}={value!r} but the selected "
                f"entry was fitted with {field}={stamped!r}",
            )
    if spec.kind != "gate":
        return
    # a position gate's θ is one entry per token position; a feature gate's
    # one per coordinate — compared both ways, an unstamped bundle being a
    # feature gate (§2.5 axis)
    declared_axis = spec.axis if isinstance(spec.axis, str) else None
    stamped_axis = record.get("axis")
    if (stamped_axis or None) != declared_axis:
        raise ProtocolError(
            "P2",
            f"{what}: the document's gate runs over "
            f"{declared_axis or 'coordinates'} but the selected entry was fitted "
            f"over {stamped_axis or 'coordinates'} — a mask over positions is not "
            "a mask over coordinates",
        )
    # a gate's map from theta to mask decides its hard split (θ > 0 against
    # θ > ½), so a bundle fitted under one map is not a mask under the other.
    # Unstamped means fitted before the field existed, i.e. sigmoid — both
    # directions are compared, like `group` below
    declared = (
        spec.parametrization if isinstance(spec.parametrization, str) else "sigmoid"
    )
    stamped_param = record.get("parametrization") or "sigmoid"
    if str(stamped_param) != declared:
        raise ProtocolError(
            "P2",
            f"{what}: the document's gate is parametrized {declared!r} but the "
            f"selected entry was fitted {stamped_param!r} — the two hard masks "
            "differ (θ > 0 against θ > ½), so a mask under one map is not a "
            "mask under the other",
        )
    if declared == "hard_concrete":
        # the stretch decides WHERE the hard split falls, θ > logit((½−γ)/(ζ−γ)),
        # so it is compared like the map, in both directions: an authored
        # stretch against the stamped one, and a non-default stamp against a
        # spec authoring none (which implies the default). As JSON, not
        # str(tuple): the stamp is '[-0.1, 1.1]'. This is the check that
        # reaches a swept producer whose entry the executing point selects —
        # the load-time check bails on that case before comparing anything
        stamped_stretch = record.get("stretch")
        if stamped_stretch is not None:
            fitted = [
                float(v)
                for v in (
                    json.loads(stamped_stretch)
                    if isinstance(stamped_stretch, str)
                    else stamped_stretch
                )
            ]
            wanted = (
                [float(v) for v in spec.stretch]
                if spec.stretch is not None
                else list(HARD_CONCRETE_STRETCH)
            )
            if fitted != wanted:
                authored = (
                    f"stretch {wanted}"
                    if spec.stretch is not None
                    else f"no stretch (the default {wanted})"
                )
                raise ProtocolError(
                    "P2",
                    f"{what}: the document's gate declares {authored} but the "
                    f"selected entry was fitted at stretch {fitted} — the two hard "
                    "masks split θ at different thresholds, so declare the same "
                    "stretch (§2.5)",
                )
    group = spec.group if isinstance(spec.group, str) else None
    stamped_group = record.get("group")
    if stamped_group is not None and str(stamped_group) != str(group):
        raise ProtocolError(
            "P2",
            f"{what}: the document declares group={group!r} on the gate but the "
            f"selected entry was fitted with group={stamped_group!r} — a mask "
            "over one kind of unit is not a mask over another",
        )
    stamped_map = record.get("group_map")
    if group is not None and groups is not None and stamped_map is not None:
        want = list(groups)
        got = json.loads(stamped_map) if isinstance(stamped_map, str) else stamped_map
        if list(got) != want:
            raise ProtocolError(
                "P2",
                f"{what}: the site here has {_describe_map(group, want)} but the "
                f"fitted gate was grouped over {_describe_map(group, got)} — same "
                "group kind, different units",
            )


def _describe_map(group: str, group_map: Sequence[int]) -> str:
    """A group map in words: ``8 heads of 32 coordinates``, ``128 experts
    of 32 neurons`` or ``one site of 768 coordinates``."""
    if group == "expert_neuron":
        return f"{group_map[0]} experts of {group_map[1]} neurons"
    if group == "site":
        return f"one site of {group_map[1]} coordinates"
    return f"{group_map[0]} heads of {group_map[1]} coordinates"


def _build_stage(
    name: str,
    spec: FeaturizerSpec,
    *,
    width: int,
    load_tensors: Any,
    seed: int = 0,
    coords: Mapping[str, Any] | None = None,
    groups: tuple[int, int] | None = None,
    load_table: Any = None,
    site_record: Mapping[str, Any] | None = None,
) -> Stage:
    kind = spec.kind if isinstance(spec.kind, str) else "identity"
    group = spec.group if isinstance(spec.group, str) else None
    if isinstance(spec.file_path, str):
        slots = FEATURIZER_SLOTS.get(kind, ())
        if not slots:
            raise ProtocolError(
                "P2", f"featurizer kind {kind!r} cannot be loaded from a file"
            )
        want, implicit = entry_selection(spec.entry, coords, name)
        what = f"featurizer {name!r} ({spec.file_path})"
        point = load_tensors(spec.file_path).point(
            slots[0], want, what=what, implicit=implicit
        )
        # the entry's record carries what a swept producer stamped per entry;
        # the header identity carries what it stamped file-wide (a
        # single-point fit's `parametrization`, `group`) — the check reads
        # both, entry over file, as `entry_identity` does
        _check_entry_identity(
            {**point.identity, **point.record}, spec, what, groups=groups
        )
        slot = point.tensor
        if kind in ("subspace", "pca"):
            return LoadedLinear(kind, slot("weight"))
        if kind == "standardize":
            return Standardize(slot("mu"), slot("sigma"))
        if kind == "sae":
            return Sae(slot("enc"), slot("dec"), slot("b_enc"), slot("b_dec"))
        if kind == "gate":
            theta = slot("theta")
            expected = math.prod(
                gate_param_shape(
                    group, groups, width, parametrization=_gate_parametrization(spec)
                )
            )
            positional = spec.axis == "position"
            if theta.numel() != expected and _gate_parametrization(spec) == "boundary":
                raise ProtocolError(
                    "P2",
                    f"{what}: a boundary gate's theta is one β, but the bundle "
                    f"holds {theta.numel()} parameters — it was not fitted as a "
                    "boundary gate (§2.5)",
                )
            if theta.numel() != expected:
                # §2.5 `axis`: a position gate's units are positions and its
                # layout is the window, so the message counts what θ counts
                unit = (
                    "positions"
                    if positional
                    else {
                        None: "wide",
                        "head": "heads",
                        "expert_neuron": "expert neurons",
                        "site": "site-wide unit(s)",
                    }[group]
                )
                where = "window" if positional else "site"
                raise ProtocolError(
                    "P2",
                    f"{what}: the fitted gate is {theta.numel()} {unit} but the "
                    f"{where} here is {expected} — a mask is a set of units of "
                    "one activation, so it only applies at the layout it was "
                    "fitted at",
                )
            top_k = _concrete_top_k(spec)
            if _gate_parametrization(spec) == "budget" and top_k is None:
                raise ProtocolError(
                    "P2",
                    f"{what}: the fitted gate is a budget gate — its theta is a "
                    "ranking with no threshold, so the document names the cut: "
                    "author 'top_k' (§2.5)",
                )
            if (
                top_k is not None
                and top_k > expected
                and not isinstance(spec.pool, str)
            ):
                unit = (
                    "positions"
                    if positional
                    else {
                        None: "coordinates",
                        "head": "heads",
                        "expert_neuron": "expert neurons",
                        "site": "site-wide unit(s)",
                    }[group]
                )
                raise ProtocolError(
                    "P2",
                    f"{what}: top_k={top_k} but the gate has {expected} {unit} — a "
                    "top-k readout keeps at most every unit (§2.5)",
                )
            gate = Gate.from_theta(
                theta,
                group=group,
                groups=groups,
                width=width,
                parametrization=_gate_parametrization(spec),
                temperature=_concrete_float(spec.temperature),
                stretch=spec.stretch,
                top_k=top_k,
                pool=spec.pool if isinstance(spec.pool, str) else None,
                axis=spec.axis if isinstance(spec.axis, str) else None,
                forward=spec.forward,
            )
            stamped_units = {**point.identity, **point.record}.get("pool_units")
            if stamped_units is not None:
                # compared with the pool the document assembles, at the link
                gate.stamped_pool_units = int(stamped_units)
            return gate
        raise ProtocolError(
            "P2", f"featurizer kind {kind!r} cannot be loaded from a file"
        )
    if kind == "identity":
        return Identity()
    if kind == "subspace":
        k = spec.k if isinstance(spec.k, int) else None
        parametrization = (
            spec.parametrization if isinstance(spec.parametrization, str) else "cayley"
        )
        if k is None:
            raise ProtocolError("P2", f"subspace featurizer {name!r} needs k")
        if spec.init is None:
            return Subspace(width, k, parametrization, seed=seed)
        basis, identity = _init_basis(
            name,
            spec,
            load_tensors,
            width=width,
            k=k,
            coords=coords,
            site_record=site_record,
        )
        return Subspace(
            width, k, parametrization, seed=seed, init=basis, init_identity=identity
        )
    if kind == "gate":
        # θ starts at the midpoint mask, at a declared fill, or at a saved
        # theta — no draw anywhere, so nothing for `seed` to influence
        start, start_identity, start_scores = _gate_start(
            name,
            spec,
            load_tensors,
            group=group,
            groups=groups,
            width=width,
            coords=coords,
            load_table=load_table,
        )
        gate = Gate(
            width,
            group=group,
            groups=groups,
            axis=spec.axis if isinstance(spec.axis, str) else None,
            forward=spec.forward,
            parametrization=_gate_parametrization(spec),
            init=start,
            init_identity=start_identity,
            temperature=_concrete_float(spec.temperature),
            stretch=spec.stretch,
            k_schedule=_concrete_k_schedule(
                name,
                spec,
                # a pooled schedule counts the POOL's units, which only the
                # link (`link_budget_pools`) can check once every member exists
                units=None
                if isinstance(spec.pool, str)
                else math.prod(
                    gate_param_shape(
                        group,
                        groups,
                        width,
                        parametrization=_gate_parametrization(spec),
                    )
                ),
            ),
            stop_grad_shift=spec.stop_grad_shift is True,
            pool=spec.pool if isinstance(spec.pool, str) else None,
            dead=spec.dead,
        )
        gate.init_scores = start_scores
        return gate
    raise ProtocolError(
        "P2",
        f"featurizer {name!r} of kind {kind!r} needs a file_path — this engine "
        "does not fit it from data at run start",
    )


def _concrete_k_schedule(
    name: str, spec: FeaturizerSpec, *, units: int | None
) -> dict[str, Any] | None:
    """A budget gate's resolved ``k_schedule`` (§2.5), checked against the unit
    count the site gives the gate — a budget above the units names units the
    gate does not have (``units`` is ``None`` for a pooled gate, whose count is
    the pool's and is checked at the link) — and refused when a sweep reaches
    the build, like an unresolved temperature: a different cut is a different
    fit."""
    if spec.k_schedule is None:
        return None
    out: dict[str, Any] = {}
    for key, value in spec.k_schedule.items():
        if key in ("kind", "of"):
            out[key] = value
            continue
        if not isinstance(value, int) or isinstance(value, bool):
            raise ProtocolError(
                "P2",
                f"featurizer {name!r}: k_schedule.{key} must be one integer by the "
                f"time the stage is built, got {value!r}",
            )
        if units is not None and value > units:
            raise ProtocolError(
                "P2",
                f"featurizer {name!r}: k_schedule.{key}={value} but the gate has "
                f"{units} units — a budget keeps at most every unit (§2.5)",
            )
        out[key] = value
    return out


def _concrete_float(value: Any) -> float | None:
    """A resolved numeric field, or ``None`` when unauthored. Anything else —
    an unresolved sweep reaching the build — is refused rather than silently
    read as the default: a quietly different β is a different fit."""
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    raise ProtocolError(
        "P2",
        f"a gate's temperature must be one number by the time the stage is built, "
        f"got {value!r} — a swept value is resolved per point before the build",
    )


def _concrete_top_k(spec: FeaturizerSpec) -> int | None:
    """A loaded gate's resolved ``top_k`` (§2.5), or ``None`` when unauthored;
    a sweep reaching the build is refused like an unresolved temperature —
    a different cut is a different mask."""
    if spec.top_k is None:
        return None
    if isinstance(spec.top_k, int) and not isinstance(spec.top_k, bool):
        return spec.top_k
    raise ProtocolError(
        "P2", f"gate top_k did not resolve to an integer (got {spec.top_k!r})"
    )


def _gate_start(
    name: str,
    spec: FeaturizerSpec,
    load_tensors: Any,
    *,
    group: str | None,
    groups: tuple[int, int] | None,
    width: int,
    coords: Mapping[str, Any] | None,
    load_table: Any = None,
) -> tuple[float | torch.Tensor | None, dict[str, Any] | None, dict[str, Any] | None]:
    """Where a gate fit starts (§2.5 ``init``), and what a bundle records
    about it: ``(None, None, None)`` for the midpoint mask, ``(fill, None,
    None)`` for a declared mask value, a saved ``theta`` with the ``init_*``
    provenance keys — the `_init_basis` contract, for a gate — or a
    theta read off a score table (`_scores_start`), the third member
    being the record ``fit_diagnostics`` keeps of it.

    A saved start has to be a theta **of this gate**: same group and map, same
    parametrization (``θ > 0`` and ``θ > ½`` split a start as differently as
    they split a mask), the unit count this layout has — the same
    `_check_entry_identity` a loaded gate passes, read over the entry's
    record and the header identity. The bundle the fit writes records the
    start's data ref as ``init_trained_on``."""
    if spec.init is None:
        return None, None, None
    if "from_scores" in spec.init and _gate_parametrization(spec) == "boundary":
        # the parser refuses this too; a second line of defense
        raise ProtocolError(
            "P2",
            f"featurizer {name!r}: init.from_scores places theta's units at the "
            "map's poles by score, and a boundary gate has one β and no unit to "
            "place (§2.5)",
        )
    if "fill" in spec.init:
        fill = spec.init["fill"]
        if isinstance(fill, bool) or not isinstance(fill, (int, float)):
            raise ProtocolError(
                "P2",
                f"featurizer {name!r}: init.fill is unresolved ({fill!r}) — a "
                "swept fill is a coordinate, resolved per point before the build",
            )
        return float(fill), None, None
    if "from_scores" in spec.init:
        return _scores_start(
            name,
            spec,
            spec.init["from_scores"],
            load_table,
            group=group,
            groups=groups,
            width=width,
        )
    (slot,) = FEATURIZER_SLOTS["gate"]
    init_path = str(spec.init["file_path"])
    want, implicit = entry_selection(spec.init.get("entry"), coords, name)
    what = f"featurizer {name!r} init ({init_path})"
    point = load_tensors(init_path).point(slot, want, what=what, implicit=implicit)
    _check_entry_identity({**point.identity, **point.record}, spec, what, groups=groups)
    theta = point.tensor(slot)
    expected = math.prod(
        gate_param_shape(
            group, groups, width, parametrization=_gate_parametrization(spec)
        )
    )
    if theta.numel() != expected:
        units = (
            "one β"
            if _gate_parametrization(spec) == "boundary"
            else f"{expected} units"
        )
        raise ProtocolError(
            "P2",
            f"{what}: the saved theta holds {theta.numel()} parameters but the "
            f"gate here has {units} — a start is a theta of this gate, "
            "at the layout it is fitted at",
        )
    start = theta.detach().to("cpu", torch.float32).contiguous()
    identity = {"init_trained_on": point.identity.get("trained_on")}
    return start, identity, None


def _scores_start(
    name: str,
    spec: FeaturizerSpec,
    scores: Mapping[str, Any],
    load_table: Any,
    *,
    group: str | None,
    groups: tuple[int, int] | None,
    width: int,
) -> tuple[torch.Tensor, dict[str, Any], dict[str, Any]]:
    """A gate's start read off a per-unit score table (§2.5
    ``init.from_scores``; rule 32 is what is checked here).

    The table is a saved metric table — a list of row objects — filtered by
    ``where`` (every named column equal to its literal), then read by its
    ``unit`` column(s) and ``value`` column. After the filter it has to name
    **every unit of this gate exactly once**: a unit index is a position in
    ``theta`` — ``[0, units)`` for a one-axis theta, an ``(i, j)`` pair for a
    two-axis one (``expert_neuron``'s expert table) — and a table that skips
    or repeats one would seed a mask nobody authored. Under ``keep`` the top
    ``keep`` units by score (ties by unit index, so the start is a function
    of the table alone) go on the kept pole of the map and the rest on the
    dropped one ([`gate_poles`][]): a decisive start, and the attribution-
    or magnitude-pruning baseline when the gate is not trained. Under
    ``scale`` theta is ``midpoint + scale · z`` with ``z`` the population
    z-score of the values (a constant column is all-midpoint), clipped to the
    unit interval under ``clamp`` — the midpoint being the θ whose mask is ½,
    so a zero score is exactly the untouched start.

    The table's bytes are the canonical form's to record
    (``init.from_scores.content_digest``); a score table has no
    ``ArtifactIdentity`` of its own — it is a JSON table, not a tensor
    bundle — so the bundle stamps no ``init_*`` field for it."""
    if load_table is None:
        raise ProtocolError(
            "P2",
            f"featurizer {name!r}: init.from_scores needs a table loader and this "
            "build has none — the executor resolves the table through the run's "
            "artifact store",
        )
    what = f"featurizer {name!r} init.from_scores ({scores['file_path']})"
    rows, _bytes = load_table(str(scores["file_path"]))
    where = scores.get("where") or {}
    selected = [
        row for row in rows if all(row.get(col) == lit for col, lit in where.items())
    ]
    shape = gate_param_shape(
        group, groups, width, parametrization=_gate_parametrization(spec)
    )
    unit_columns = scores["unit"]
    unit_columns = (
        [unit_columns] if isinstance(unit_columns, str) else list(unit_columns)
    )
    where_path = f"featurizers.{name}.init.from_scores"
    if len(unit_columns) != len(shape):
        raise ValidationError(
            32,
            f"{what}: the gate's theta has {len(shape)} axis/axes "
            f"{list(shape)} but 'unit' names {len(unit_columns)} column(s) "
            f"{unit_columns} — one unit column per axis",
            path=f"{where_path}.unit",
        )
    value_column = str(scores["value"])
    values = torch.full(shape, float("nan"), dtype=torch.float32)
    seen: set[tuple[int, ...]] = set()
    for row in selected:
        try:
            index = tuple(int(row[col]) for col in unit_columns)
            value = float(row[value_column])
        except (KeyError, TypeError, ValueError) as err:
            raise ValidationError(
                32,
                f"{what}: a row lacks an integer unit under {unit_columns} or a "
                f"number under {value_column!r}: {row!r} ({err})",
                path=where_path,
            ) from None
        if not all(0 <= i < n for i, n in zip(index, shape)):
            raise ValidationError(
                32,
                f"{what}: unit {list(index)} is outside the gate's theta {list(shape)}",
                path=f"{where_path}.unit",
            )
        if index in seen:
            raise ValidationError(
                32,
                f"{what}: unit {list(index)} is named twice — narrow the table "
                "with 'where' so each unit has one score",
                path=where_path,
            )
        if math.isnan(value):
            raise ValidationError(
                32,
                f"{what}: unit {list(index)} has no score (NaN)",
                path=f"{where_path}.value",
            )
        seen.add(index)
        values[index] = value
    units = math.prod(shape)
    if len(seen) != units:
        raise ValidationError(
            32,
            f"{what}: the table names {len(seen)} of the gate's {units} units "
            + (f"after where={dict(where)} " if where else "")
            + "— a start needs a score for every unit",
            path=where_path,
        )
    parametrization = _gate_parametrization(spec)
    dropped, kept = gate_poles(parametrization, spec.stretch)
    record: dict[str, Any] = {
        "file_path": str(scores["file_path"]),
        "units": units,
        **({"where": dict(where)} if where else {}),
    }
    flat = values.flatten()
    if "keep" in scores:
        keep = scores["keep"]
        if isinstance(keep, bool) or not isinstance(keep, int):
            raise ProtocolError(
                "P2",
                f"{what}: keep is unresolved ({keep!r}) — a swept keep is a "
                "coordinate, resolved per point before the build",
            )
        if keep > units:
            raise ValidationError(
                32,
                f"{what}: keep={keep} exceeds the gate's {units} units",
                path=f"{where_path}.keep",
            )
        # descending by score, ascending by index on ties: a stable sort on
        # the negated scores keeps the index order among equals
        order = torch.sort(-flat, stable=True).indices[:keep]
        theta = torch.full((units,), dropped, dtype=torch.float32)
        theta[order] = kept
        record.update(
            {"keep": keep, "kept_units": sorted(int(i) for i in order.tolist())}
        )
    else:
        scale = scores["scale"]
        if isinstance(scale, bool) or not isinstance(scale, (int, float)):
            raise ProtocolError(
                "P2",
                f"{what}: scale is unresolved ({scale!r}) — a swept scale is a "
                "coordinate, resolved per point before the build",
            )
        std = float(flat.std(unbiased=False))
        z = (flat - flat.mean()) / std if std > 0.0 else torch.zeros_like(flat)
        midpoint = 0.5 if parametrization == "clamp" else (dropped + kept) / 2.0
        theta = midpoint + float(scale) * z
        if parametrization == "clamp":
            theta = theta.clamp(0.0, 1.0)
        record["scale"] = float(scale)
    identity: dict[str, Any] = {}
    return theta.reshape(shape).contiguous(), identity, record


def _init_basis(
    name: str,
    spec: FeaturizerSpec,
    load_tensors: Any,
    *,
    width: int,
    k: int,
    coords: Mapping[str, Any] | None,
    site_record: Mapping[str, Any] | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """The first ``k`` columns of the basis a ``subspace`` spec's ``init``
    names, and what a fit seeded from them records about that start (§2.5,
    §8): the basis's data ref and the component indices taken.

    The basis is fitted at one site of one model, so it has to be as wide as
    the site here and hold at least ``k`` components; the identity fields
    that say *which* site and model were checked at load
    (``rules.data.check_loaded_featurizers``), where the header is readable
    without the tensor. ``site_record`` is this point's site, as a stamp
    records it. The selected entry's recorded site must equal it
    ([`check_start_site`][causalab.protocol.rules.data.check_start_site]),
    which matters for a bundle with one entry per site: the engine checks
    that entry per point before the weights load, and this is the same check
    for a caller that builds a stack directly. What only the tensor can
    show is checked here: the shape, and that the columns taken are an
    orthonormal frame. The parametrization installs them as its base
    *verbatim* ([`Subspace`][]), so a basis that is not orthonormal would
    make every weight the fit ever produces non-orthonormal, silently. The
    bundle the fit writes records the basis's data ref (``init_trained_on``)
    and the columns taken (``init_components``); a swept basis stamps
    ``trained_on`` per entry, so the record is read off the selected entry."""
    assert spec.init is not None
    init_path = str(spec.init["file_path"])
    want, implicit = entry_selection(spec.init.get("entry"), coords, name)
    what = f"featurizer {name!r} init ({init_path})"
    point = load_tensors(init_path).point("weight", want, what=what, implicit=implicit)
    if site_record is not None:
        # the stamp's site (entry over file) against this point's site: the
        # comparison the engine's per-point check makes before the weights load
        check_start_site(
            {**point.identity, **point.record},
            site_record,
            what=f"{what} entry {'weight' + point.suffix!r}",
        )
    basis = point.tensor("weight")
    if basis.ndim != 2 or int(basis.shape[0]) != width:
        raise ProtocolError(
            "P2",
            f"{what}: the basis has shape {tuple(basis.shape)} but the site here "
            f"is {width} wide — a starting subspace lives in the activation "
            "space the fit is trained in, so the basis must be (width, ≥ k)",
        )
    if int(basis.shape[1]) < k:
        raise ProtocolError(
            "P2",
            f"{what}: the basis holds {int(basis.shape[1])} components but the "
            f"fit needs k={k} of them — a rank-k fit starts from the first k "
            "columns, so the basis must hold at least that many",
        )
    columns = basis[:, :k].detach().to("cpu", torch.float32).contiguous()
    deviation = orthonormality_deviation(columns)
    if deviation > ORTHONORMAL_TOLERANCE:
        raise ProtocolError(
            "P2",
            f"{what}: the first {k} columns are not orthonormal (max |PᵀP − I| = "
            f"{deviation:.3g}, tolerance {ORTHONORMAL_TOLERANCE:g}) — a "
            "subspace's start is installed as the base of an orthogonal "
            "parametrization, which only stays orthonormal if the base is",
        )
    identity = {
        "init_trained_on": point.identity.get("trained_on"),
        "init_components": list(range(k)),
    }
    return columns, identity
