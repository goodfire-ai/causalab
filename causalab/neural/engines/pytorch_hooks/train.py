"""Fit featurizer parameters under the declared training protocol.

Each member has separate stages, optimizer state, minibatch order, schedules,
and stopping state. Local generators drive initialization, ordering, and
hard-concrete masks. Setup also seeds the global generator for Torch's
orthogonal basis completion. The loop uses per-member generators while
model weights remain frozen in evaluation mode.

Objective terms reduce differentiable reads and apply signed weights.
Regularizers act on the declared featurizer parameters. Gate L1 measures
the mean soft mask for Desiderata-Based Masking; a list of gates uses one
mean across their units. Hard-concrete L0 measures expected kept fraction.
Each update projects stages into their feasible sets.

Annealing and controllers update their declared targets. Trajectory saves
record scheduled parameters, objective values, gate sizes, and controlled
values. Evaluation uses the named split with hard gates and gradients off;
early stopping follows its metric and patience. ``batch.pairs`` slices all
roles together.

Compatible fits share cohort forwards over concatenated minibatches.
Each member keeps its own rows and parameter gradients, and stops when
its budget or patience ends. ``fit_rows`` bounds gradient windows while
keeping member minibatches whole. Evaluation also honors ``batch_rows``.
"""

from __future__ import annotations

import contextlib
import dataclasses
import copy
import hashlib
import logging
from typing import Any, Callable, Iterator, Mapping, Sequence

import torch

from causalab.neural.engines.pytorch_hooks.budget import (
    DistributedOutOfMemory,
    Meter,
    OOMPolicy,
    RowBudget,
    cuda_meter,
)
from causalab.neural.engines.pytorch_hooks.rows import RowSplit
from causalab.neural.engines.pytorch_hooks.stages import TrainedOwner
from causalab.neural.shared.parallel.agreements import (
    average_gradients,
    configured_agreement,
)
from causalab.neural.shared.parallel.collective import SOLO, Collective
from causalab.neural.engines.pytorch_hooks.cohort import (
    cohort_entries,
    groups_read_by,
    run_groups,
)
from causalab.neural.engines.pytorch_hooks.control import (
    PidController,
    build_controller,
    ramp_setpoint,
)
from causalab.neural.engines.pytorch_hooks.executor import PointExecutor, document_seed
from causalab.neural.shared.encoding import EncodedBatch
from causalab.neural.shared.execution import (
    Checkpoint,
    TrainEvalScore,
    TrainOutcome,
)
from causalab.neural.shared.results import MASK_DECISIVE_MARGIN
from causalab.neural.shared.encoding import encode
from causalab.neural.shared.executor import ForwardCache
from causalab.neural.shared.featurizers import (
    ORTHONORMAL_TOLERANCE,
    Gate,
    Stage,
    Subspace,
    featurizer_cache,
    orthonormality_deviation,
)
from causalab.neural.shared.plan import write_names
from causalab.protocol.positions.roles import input_roles
from causalab.neural.shared.metrics import (
    GATHERED_KINDS,
    compute_metric,
    gathered_metric,
    js_divergence,
)
from causalab.protocol.answers import (
    column_token_ids,
    metric_token_ids,
    restrict_token_ids,
)
from causalab.protocol.engine import RunContext
from causalab.protocol.rules.errors import ProtocolError
from causalab.protocol.schema import (
    OBJECTIVE_WEIGHT_PREFIX,
    PER_PARAMS_OPTIMIZER_FIELDS,
    AnnealSchedule,
    BoundAggregation,
    ConstraintSpec,
    DataRole,
    Document,
    PhaseSpec,
    AggregationSpec,
    ReadRef,
    TrainSpec,
    concrete_int,
    concrete_str,
    operand_reads,
    read_is_vocabulary,
)

from causalab.neural.engines.pytorch_hooks.cuda_graphs import (
    GraphExecutor,
    GraphPool,
    TrainingGraphs,
    copy_executor_stages,
    make_executor,
)
from causalab.neural.engines.pytorch_hooks.graph_cohort import (
    CohortGraphs,
    EvaluationGraphs,
    Member,
    WindowItem,
    cohort_graph_reason,
)
from causalab.neural.engines.pytorch_hooks.graph_reuse import FitGraphCache

__all__ = ["fit_diagnostics", "metric_tensor", "run_cohort_training", "run_training"]


def metric_tensor(
    metric: AggregationSpec,
    of_value: torch.Tensor,
    rows: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    *,
    target_value: torch.Tensor | None = None,
    token_ids: Mapping[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    """A differentiable per-example metric (the objective-side twin of
    ``metrics.compute_metric``). Only the reduction kinds with a gradient
    are usable in an objective."""
    logits = of_value[:, 0, :] if of_value.dim() == 3 else of_value
    logits = logits.float()
    kind = str(metric.kind)

    form = metric.token_form  # §2.10: strings as written, or "id"

    def ids(field: str) -> torch.Tensor:
        if token_ids is not None:
            return token_ids[field]
        column = concrete_str(metric.fields[field], f"metric field {field}")
        return torch.tensor(
            column_token_ids(
                tokenizer,
                [row[column] if form == "id" else str(row[column]) for row in rows],
                token_form=form,
                where=f"metric {kind}.{field}",
            ),
            dtype=torch.long,
            device=logits.device,
        )

    if kind == "cross_entropy":
        log_probs = torch.log_softmax(logits, dim=-1)
        return -log_probs.gather(1, ids("target").unsqueeze(1)).squeeze(1)
    if kind == "logit_diff":
        return logits.gather(1, ids("a").unsqueeze(1)).squeeze(1) - logits.gather(
            1, ids("b").unsqueeze(1)
        ).squeeze(1)
    if kind == "soft_accuracy":
        # the same margin through a sigmoid (metrics.py's twin): its gradient
        # vanishes once a row is decided, so the fit spends its updates on the
        # rows still near the boundary, which is why a mask fit trains on it
        margin = logits.gather(1, ids("a").unsqueeze(1)).squeeze(1) - logits.gather(
            1, ids("b").unsqueeze(1)
        ).squeeze(1)
        return torch.sigmoid(margin)
    if kind == "kl":
        if target_value is None:
            raise ProtocolError("P2", "kl needs its target read's value")
        target = target_value[:, 0, :] if target_value.dim() == 3 else target_value
        p = torch.log_softmax(logits, dim=-1)
        q = torch.log_softmax(target.float(), dim=-1)
        return (p.exp() * (p - q)).sum(dim=-1)
    if kind == "js":
        if target_value is None:
            raise ProtocolError("P2", "js needs its target read's value")
        target = target_value[:, 0, :] if target_value.dim() == 3 else target_value
        # the arithmetic the saved metric table reports (metrics.js_divergence),
        # so the objective and the record cannot disagree; differentiable in
        # `logits`, and the target is a constant model's read
        return js_divergence(
            logits, target.float(), restrict_token_ids(metric, rows, tokenizer)
        )
    raise ProtocolError(
        "P2",
        f"metric kind {kind!r} has no gradient — objectives compose from "
        "cross_entropy / logit_diff / soft_accuracy / kl / js (§2.11)",
    )


def _cost(
    piece: torch.Tensor, target: str, costs: Mapping[str, float] | str | None
) -> torch.Tensor:
    """§2.11 ``costs``: one target's penalized quantities, scaled before the
    concatenation. A table costs the target its entry (1 when unlisted);
    ``parameter_count`` divides by the target's own element count — ``theta``
    as stored, so on a grouped gate its *units* (heads, experts; positions
    on a position gate), not the
    coordinates they span: the same per-unit reading `_regularizer`
    gives the quantity — so under ``reduce: sum`` the term is a sum of
    per-featurizer means — NeuroSurgeon's λ scaled with the parameter count,
    one weight meaning one thing across featurizers of different sizes
    without the author doing the division. Under the default ``reduce: mean``
    the same word gives ``(1/N) Σ_f S_f/n_f``: the per-featurizer equality
    kept, the whole term scaled by the total unit count ``N`` — the width
    coupling the word exists to remove — so ``parameter_count`` pairs with
    ``sum`` (the other pairing is legal, and is that coupling)."""
    if costs is None:
        return piece
    if costs == "parameter_count":
        # never empty: a gate has theta, a param target matched a slot
        return piece / piece.numel()
    assert isinstance(costs, Mapping)
    return piece * float(costs.get(target, 1.0))


def _regularizer(
    kind: str,
    targets: Sequence[str],
    stages: Mapping[str, Stage],
    reduce: str = "mean",
    costs: Mapping[str, float] | str | None = None,
) -> torch.Tensor:
    """One penalty over everything ``targets`` name (§2.11): the ``reduce``
    — the mean when unauthored, or the sum — over the **concatenation** of
    their penalized quantities, not a mean of per-featurizer means — so under
    the mean a gate with more units weighs more, and one weight over forty
    layers' gates counts selected units across all of them; under the sum a
    kept unit costs the term's weight whatever the unit count, NeuroSurgeon's
    ``λ · Σ`` convention, which is what lets one weight mean one thing across
    gates of different sizes.

    For a ``gate`` under ``l1`` the quantity is the SOFT mask ``σ(θ/T)``, not
    ``|θ|`` — DBM sparsity pushes mask mass toward zero features,
    temperature-annealed — over ``theta`` as stored, so on a grouped gate it
    is per unit: selected heads, not the coordinates they span; the whole
    (expert, neuron) table, not the slots a token happens to fill; on a
    position gate (§2.5 ``axis``) the addressed positions. Under
    ``l0`` it is the gate's **expected kept fraction** per unit
    ([`Gate.expected_l0`][causalab.neural.shared.featurizers.gate.Gate.expected_l0]): Louizos et al.'s closed form for a
    ``hard_concrete`` gate, whose sampled training mask makes the soft mask
    the wrong surrogate.
    For every other featurizer (or a gate under ``l2``) it is ``|p|`` or ``p²``
    over its params; a dotted target restricts to that one slot. The pairing
    is by map, not by kind: ``l0`` on a deterministic gate would be ``l1``
    under a second name, and ``l1`` on a ``hard_concrete`` gate would penalize
    a mask its training forward never uses — both, and ``l0`` on a featurizer
    with no mask, are refused at validation (rule 4) and again here.

    ``costs`` scales each target's quantities before the concatenation
    (`_cost`), so a table ``{"gate_15": 0.25}`` makes a kept unit of
    that gate a quarter as expensive as one elsewhere, and
    ``"parameter_count"`` makes every target weigh its mean.
    """
    pieces: list[torch.Tensor] = []
    for target in targets:
        fname, _, slot_name = target.partition(".")
        stage = stages[fname]
        if isinstance(stage, Gate) and stage.parametrization == "budget":
            raise ProtocolError(
                "P2",
                f"regularizer target {target!r}: a budget gate's mask sums to the "
                "step's budget by construction — it takes no sparsity penalty (§2.5)",
            )
        if isinstance(stage, Gate) and kind in ("l1", "l0"):
            sampled = stage.parametrization == "hard_concrete"
            if sampled != (kind == "l0"):
                raise ProtocolError(
                    "P2",
                    f"regularizer target {target!r}: {kind!r} does not pair with a "
                    f"{stage.parametrization!r} gate — 'l0' is the expected kept "
                    "fraction of a sampled (hard_concrete) mask, 'l1' the mean of a "
                    "deterministic one (§2.11); validation refuses this at load",
                )
            # the gate's own relaxed mask — σ(θ/T), or θ itself under `clamp`
            # — or, under `hard_concrete`, its expected L0
            quantity = stage.soft_mask() if kind == "l1" else stage.expected_l0()
            pieces.append(_cost(quantity.flatten(), target, costs))
            continue
        if kind == "l0":
            raise ProtocolError(
                "P2",
                f"regularizer target {target!r}: 'l0' is the expected kept fraction "
                "of a gate's mask; this featurizer has no mask",
            )
        own: list[torch.Tensor] = []
        for slot, param in stage.slot_params().items():
            if slot_name and slot != slot_name:
                continue
            own.append((param.abs() if kind == "l1" else param.pow(2)).flatten())
        if not own:
            raise ProtocolError("P2", f"regularizer target {target!r} matches no slot")
        # one target, one cost: `parameter_count` counts every slot it matched
        pieces.append(_cost(torch.cat(own), target, costs))
    cat = torch.cat(pieces)
    return cat.sum() if reduce == "sum" else cat.mean()


def _slice_rows(
    role_rows: Mapping[str, list[dict[str, Any]]], indices: list[int]
) -> dict[str, list[dict[str, Any]]]:
    return {role: [rows[i] for i in indices] for role, rows in role_rows.items()}


class _Drawn:
    """§2.2 ``draw``: the drawn roles of one fit — every member of every row
    encoded once into one expanded frame per role, so a minibatch under any
    draw is a **selection** of that frame (the same padded width, and a
    cohort's members encode the same texts, so they still concatenate — by
    construction), and the per-epoch redraw that rebuilds the fit's
    minibatch executors with one member per row.

    The draw runs on its own stream: the document seed hashed with the word
    ``draw`` (`_stream_seed`), not the raw seed the batch order and the
    mask samples share — both of those call ``torch.rand`` too, and a member
    drawn from the same numbers as a gate's first resample would be
    correlated with the mask noise.

    Refusals at prepare, two codes: a row with no members, an ``eval`` past a
    row's count and a per-member sibling of another length are the *data*'s
    shape (P2); ``segments`` is a declared frame this engine cannot combine
    with a re-encoded role (P4). Graph capture is neither: a
    run asked to capture fit graphs runs a drawn point eager
    (``cuda_graphs.unsupported_reason``), because the shapes *are*
    epoch-invariant (every minibatch is a ``select`` of one frame) and it is
    the per-epoch rebuild of the minibatch executors, not the draw, that a
    captured graph cannot follow; `of` asserts that routing rather
    than refusing a second time."""

    def __init__(
        self,
        doc: Document,
        executor: PointExecutor,
        roles: Mapping[str, DataRole],
        seed: int,
    ) -> None:
        self.doc = doc
        self.executor = executor
        self.roles = dict(roles)
        self.rng = torch.Generator().manual_seed(_stream_seed(seed, "draw"))
        self.trace: dict[str, list[list[int]]] = {role: [] for role in roles}
        self.members: dict[str, list[list[Any]]] = {}
        self.offsets: dict[str, list[int]] = {}
        self.expanded: dict[str, EncodedBatch] = {}
        for role, spec in roles.items():
            column = spec.draw_column
            assert column is not None
            members: list[list[Any]] = []
            for i, row in enumerate(executor.role_rows[role]):
                values = row.get(column)
                if not isinstance(values, list) or not values:
                    raise ProtocolError(
                        "P2",
                        f"data.{role}: row {i} has no non-empty list in column "
                        f"{column!r} to draw from",
                    )
                if spec.eval_member >= len(values):
                    raise ProtocolError(
                        "P2",
                        f"data.{role}.draw.eval = {spec.eval_member} but row {i} "
                        f"holds {len(values)} member(s) in {column!r}",
                    )
                members.append(list(values))
                # the per-member siblings are checked here, once, with the other
                # data-shape refusals — not on every epoch's redraw
                _check_member_siblings(row, column, len(values), role=role, index=i)
            offsets, total = [], 0
            for values in members:
                offsets.append(total)
                total += len(values)
            self.members[role] = members
            self.offsets[role] = offsets
            self.expanded[role] = encode(
                executor.bundle.tokenizer,
                [str(t) for values in members for t in values],
                device=executor.bundle.devices.embedding,
            )

    @classmethod
    def of(cls, doc: Document, executor: PointExecutor, seed: int) -> "_Drawn | None":
        roles = {
            role: spec
            for role, spec in input_roles(doc).items()
            if spec.draw is not None and role in executor.role_rows
        }
        if not roles:
            return None
        # `cuda_graphs.unsupported_reason` sends a drawn document down the eager
        # path, so a drawn point is a plain `PointExecutor`; this is the
        # invariant, not a second refusal
        assert not (executor.cuda_graphs and executor.fit_cuda_graphs), (
            "unsupported_reason routes a drawn point to the eager executor"
        )
        if doc.segments is not None:
            raise ProtocolError(
                "P4",
                "data.*.draw: a drawn role is encoded as plain text; it does not "
                "combine with `segments` (a declared frame) in this engine",
            )
        for role, spec in roles.items():
            # every index of the collapsed lists reads the drawn member, so the
            # field the rest of the document reads must be the one the fit
            # uses: `resolve_roles` hands the engine `spec.resolved_field`, and
            # the minibatch executors take the executor's fields as they are.
            # An epoch-invariant — checked here once, like the siblings, not
            # on every redraw
            assert executor.role_fields[role] == spec.resolved_field, (
                f"data.{role}: the executor reads {executor.role_fields[role]!r}, "
                f"the draw {spec.resolved_field!r}"
            )
        return cls(doc, executor, roles, seed)

    def record(self) -> dict[str, dict[str, Any]]:
        """§2.2: what ``fit_diagnostics.json`` carries under ``draws`` — per
        drawn role the kind, ``eval`` and the member each row took, one list
        per epoch. Built here, from the roles that drew, so a role the executor
        does not carry gets no entry."""
        return {
            role: {
                "kind": str(spec.draw["kind"]),  # type: ignore[index]  # `of` filtered on draw
                "eval": spec.eval_member,
                "members": [list(epoch) for epoch in self.trace[role]],
            }
            for role, spec in self.roles.items()
        }

    def draw(self) -> dict[str, list[int]]:
        """One member index per row per drawn role, uniform over the row's
        members, from this fit's own generator."""
        out: dict[str, list[int]] = {}
        for role, members in self.members.items():
            u = torch.rand(len(members), generator=self.rng)
            out[role] = [
                min(int(u[i].item() * len(values)), len(values) - 1)
                for i, values in enumerate(members)
            ]
        return out

    def minibatches(self) -> list[PointExecutor]:
        """One epoch's minibatch executors over the partition and frames
        `bind` set: a fresh draw, the drawn rows, and per batch a
        selection of the expanded frame. ``interning=None``: the
        store is not consulted by these executors at all — a source forward
        over a drawn role is constant for no two epochs, and the store is
        per executor, so the base role's forwards are recomputed with it
        (narrowing the bypass to the drawn role is a follow-up)."""
        picks = self.draw()
        for role, chosen in picks.items():
            self.trace[role].append(list(chosen))
        executor = self.executor
        role_rows: dict[str, list[dict[str, Any]]] = dict(executor.role_rows)
        for role, spec in self.roles.items():
            column = spec.draw_column
            assert column is not None
            role_rows[role] = [
                _drawn_row(row, column, picks[role][i])
                for i, row in enumerate(executor.role_rows[role])
            ]
        out: list[PointExecutor] = []
        for indices in self.batches:
            selected = {
                role: (
                    self.expanded[role].select(
                        [self.offsets[role][i] + picks[role][i] for i in indices]
                    )
                    if role in self.roles
                    else frame.select(indices)
                )
                for role, frame in self.frames.items()
            }
            out.append(
                _derived(
                    executor,
                    make_executor(
                        self.doc,
                        executor.bundle,
                        cuda_graphs=False,
                        role_rows=_slice_rows(role_rows, indices),
                        # the outer executor's fields as they are: `role_fields[role]`
                        # is the drawn member's (`of` checks it once, at prepare)
                        role_fields=executor.role_fields,
                        load_tensors=executor.load_tensors,
                        load_table=executor.load_table,
                        # shared, not per executor: `_prepare_fit` built the
                        # optimizer's groups over these stages, so a rebuilt
                        # minibatch's `stage(name)` is a cache hit on the tensors
                        # being stepped — a fresh cache would re-initialise every
                        # featurizer each epoch while the optimizer stepped the
                        # originals (the identity test diverges at the first
                        # checkpoint)
                        stage_cache=executor.stage_cache,
                        grad_enabled=True,
                        coords=executor.coords,
                        interning=None,
                        batches=selected,
                    ),
                )
            )
        return out

    def bind(
        self, batches: Sequence[list[int]], frames: Mapping[str, EncodedBatch]
    ) -> None:
        """The fit's minibatch partition and its per-role frames — what every
        epoch's `minibatches` selects from; known once the frames are
        encoded, and bound before the first draw is taken, so no reader of
        ``fit.drawn`` can find them unset."""
        self.batches = list(batches)
        self.frames = dict(frames)

    def redraw(self, fit: "_Fit") -> None:
        """A new member per row for a new epoch: the fit's minibatch
        executors are rebuilt over this draw, and its captured objectives
        (keyed by executor) fall with them."""
        fit.minibatch_executors = self.minibatches()
        fit.graph_objectives.clear()


#: The per-member siblings of a list column (§2.2): the prompt-variable table
#: (``encoding.variable_value`` reads ``<column>_variables``). Named, not a
#: prefix: another ``<column>_…`` column is the author's own and is left alone.
_MEMBER_SIBLINGS: tuple[str, ...] = ("_variables",)


def _member_siblings(row: Mapping[str, Any], column: str) -> list[str]:
    return [column + suffix for suffix in _MEMBER_SIBLINGS if column + suffix in row]


def _check_member_siblings(
    row: Mapping[str, Any], column: str, count: int, *, role: str, index: int
) -> None:
    """A per-member sibling holds one entry per member: a shorter or longer
    list is refused rather than passed through half-collapsed — the document
    would read member 0 of it against member ``m`` of the text. (The
    fixed-member path tolerates a short ``_variables`` table by falling back
    to the plain column; under a draw there is no one member to fall back
    to.) A prepare-time check: it depends on the row, never on the draw."""
    for key in _member_siblings(row, column):
        value = row[key]
        if isinstance(value, list) and len(value) != count:
            raise ProtocolError(
                "P2",
                f"data.{role}: row {index} holds {len(value)} entries in {key!r} for "
                f"the {count} member(s) of {column!r} — a per-member sibling holds "
                "one entry per member",
            )


def _drawn_row(row: Mapping[str, Any], column: str, member: int) -> dict[str, Any]:
    """The row a minibatch executor reads for a drawn role. The list column
    and its per-member siblings (`_MEMBER_SIBLINGS`) hold the drawn
    member at **every** index, so ``<column>[eval]`` and ``<column>[0]`` alike
    read it inside the fit (``eval`` need not be 0). Every other column — a
    ``<column>_…`` list of the author's own included — is untouched. Pure
    rewriting: the shape was checked at prepare (`_check_member_siblings`),
    so every list here has one entry per member and its own length is the
    count."""
    out = dict(row)
    for key in (column, *_member_siblings(row, column)):
        value = row[key]
        if isinstance(value, list):
            out[key] = [value[member]] * len(value)
    return out


def _stream_seed(seed: int, purpose: str) -> int:
    """A generator seed for one purpose derived from the document seed, so
    two purposes that both call ``torch.rand`` do not consume the same
    numbers; stable across processes (sha256, not ``hash``)."""
    digest = hashlib.sha256(f"{int(seed)}:{purpose}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def _inner_interning(executor: PointExecutor, rows: tuple[int, ...] | str) -> Any:
    """The store handle a fit's inner executor gets. A solo graph fit runs on
    the graph executor's own frozen cache instead of the store (its constants
    must sit in storage the captures own); a captured cohort keeps the store
    (``GraphExecutor.keep_store``) and copies what its graph needs out of it."""
    if executor.cuda_graphs and not getattr(executor, "keep_store", False):
        return None
    return executor.inner_interning(rows)


def _slot_rows(doc: Document, executor: PointExecutor) -> int:
    """The largest minibatch this dataset can produce, before inner executors
    are built and their graph eligibility is chosen."""
    assert doc.train is not None
    return min(
        concrete_int(doc.train.batch["pairs"], "train.batch.pairs"),
        len(executor.rows_for_metrics()),
    )


# --------------------------------------------------------------------------- #
# the loop
# --------------------------------------------------------------------------- #


@dataclasses.dataclass
class _Fit:
    """One point's fit, as the cohort loop advances it: what the document
    declared, what the loop built for it, and where it stands."""

    doc: Document
    executor: PointExecutor
    seed: int
    stages: dict[str, Stage]
    trained_names: tuple[str, ...]
    optimizer: torch.optim.Optimizer
    #: the minibatches: contiguous blocks of ``train.batch.pairs`` row indices
    batches: list[list[int]]
    #: this replica's rows of each minibatch — the whole minibatch unless the
    #: data axis is split over rows (``rows``; docs/model_parallelism.md §8.3)
    slices: list[list[int]]
    rows: RowSplit
    #: the pipeline stage that computes this fit's featurizer gradients, from
    #: which every rank takes the trained parameters after each update
    #: (``stages.TrainedOwner``; docs/model_parallelism.md §7, §8.3)
    owner: TrainedOwner
    minibatch_executors: list[PointExecutor]
    epochs: int
    total_steps: int
    anneals: dict[str, AnnealSchedule]
    eval_every_epochs: int | None
    order_rng: torch.Generator
    #: the generator a ``hard_concrete`` gate's per-step mask draw comes from
    #: — this member's own, so a cohort cannot mix members' samples
    mask_rng: torch.Generator
    #: this fit's reads the objective needs, in order — the groups a step runs
    objective_reads: tuple[ReadRef, ...]
    step: int = 0
    epoch: int = 0
    #: the current epoch's minibatch order and the next position in it
    order: list[int] = dataclasses.field(default_factory=list)
    position: int = 0
    active: bool = True
    best: float | None = None
    stale: int = 0
    eval_passes: int = 0
    last_score: dict[str, float] | None = None
    #: the pool the fit's held-out inference replays capture into — the
    #: cohort's, or the solo bank's — handed to the eval executor when it is
    #: built; None when the fit runs eagerly
    graph_pool: GraphPool | None = None
    # `early_stop` selects a fit by its eval score, so the fit it selected is
    # the one that must be saved. Without a snapshot the loop returned the
    # *last* stages — after `patience` non-improving evals, the worst of the
    # tail — and nothing in the saved bundle said which you had.
    best_state: dict[str, dict[str, torch.Tensor]] | None = None
    best_score: dict[str, float] | None = None
    #: what this fit's inner passes paid for constant groups and were served
    run: int = 0
    served: int = 0
    #: the block each forward this fit took part in resumed at
    resumed: list[int] = dataclasses.field(default_factory=list)
    #: §2.11 ``control``: the live value of every named term's weight (the
    #: authored weight is a controller's start), the controllers bound to
    #: this fit's stages, and each controller's per-update trace
    live_weights: dict[str, float] = dataclasses.field(default_factory=dict)
    controls: dict[str, "_Control"] = dataclasses.field(default_factory=dict)
    #: §2.11 ``phases``: the windows in update terms, the optimizer group
    #: each ``train.params`` entry owns (its base ``lr`` / ``weight_decay``
    #: kept so a phase can restore them), and which window the fit is in —
    #: ``-1`` before the first update
    phases: tuple["_Phase", ...] = ()
    groups_by_entry: dict[str, int] = dataclasses.field(default_factory=dict)
    phase_index: int = -1
    control_trace: dict[str, list[dict[str, float]]] = dataclasses.field(
        default_factory=dict
    )
    #: §2.11 ``constraint``: per constraint term, its dual pair ``(λ₁, λ₂)``
    #: — a parameter in the fit's optimizer under a ``maximize`` group — and
    #: the per-update trace of the duals after their ascent
    duals: dict[str, torch.nn.Parameter] = dataclasses.field(default_factory=dict)
    #: the authored constraint per term name — the target and the initial
    #: duals are read from here, not recovered from the trace
    constraint_specs: dict[str, ConstraintSpec] = dataclasses.field(
        default_factory=dict
    )
    constraint_trace: dict[str, list[dict[str, float]]] = dataclasses.field(
        default_factory=dict
    )
    #: §2.2 ``draw``: the fit's drawn roles (``None`` when no role draws) —
    #: the redraw that rebuilds the minibatch executors with one fresh member
    #: per row each epoch, and per drawn role the member each row took
    drawn: "_Drawn | None" = None
    #: §2.12 ``trajectory``: the updates after which the fit is photographed,
    #: and the photographs
    checkpoint_steps: frozenset[int] = frozenset()
    checkpoints: list[Checkpoint] = dataclasses.field(default_factory=list)
    #: the last update's loss and its terms (``term.<name>``, ``weight.<name>``),
    #: what a checkpoint taken after that update records — kept as the
    #: detached device scalars (a weight is the float the update used) and
    #: read to the host together by `loss_record` only when a
    #: checkpoint asks, so an update pays no round trip for a record it
    #: does not take
    last_loss: torch.Tensor | None = None
    term_values: dict[str, torch.Tensor | float] = dataclasses.field(
        default_factory=dict
    )

    def loss_record(self) -> tuple[float, dict[str, float]]:
        """The last update's ``(loss, term_values)`` as floats: every tensor
        of the record in one host read, the values ``float(tensor)`` would
        give — ``stack`` promotes to the widest dtype among them, and a float
        widened is the same number; ``nan`` and no terms before the first
        update."""
        if self.last_loss is None:
            return float("nan"), {}
        tensors = [self.last_loss] + [
            value
            for value in self.term_values.values()
            if isinstance(value, torch.Tensor)
        ]
        read = iter(
            torch.stack(
                [t.reshape(()).to(self.last_loss.device) for t in tensors]
            ).tolist()
        )
        loss = float(next(read))
        return loss, {
            key: float(next(read)) if isinstance(value, torch.Tensor) else value
            for key, value in self.term_values.items()
        }

    graph_objectives: dict[PointExecutor, TrainingObjective] = dataclasses.field(
        default_factory=dict
    )

    @property
    def train(self) -> Any:
        assert self.doc.train is not None
        return self.doc.train

    @property
    def store(self) -> ForwardCache | None:
        return self.executor.interning.cache if self.executor.interning else None

    @property
    def exhausted(self) -> bool:
        return self.step >= self.total_steps or self.epoch >= self.epochs


class TrainingObjective:
    """The normal objective, with token-column resolution outside replay."""

    def __init__(
        self,
        executor: PointExecutor,
        stages: dict[str, Stage],
        *,
        weight: torch.Tensor | None = None,
    ) -> None:
        self.executor = executor
        self.stages = stages
        #: per-row loss weights in place of the plain mean over the executor's
        #: rows — a captured cohort slot's padding rows carry zero
        #: (``graph_cohort.py``); ``None`` is the mean
        self.weight = weight
        self.labels: dict[str, dict[str, torch.Tensor]] = {}
        train = executor.doc.train
        assert train is not None
        for index, term in enumerate(train.objective):
            agg = executor.doc.aggregation_at(term.path(index))
            if agg is None:
                continue
            metric = agg.spec
            fields = {"cross_entropy": ("target",), "logit_diff": ("a", "b")}.get(
                str(metric.kind), ()
            )
            self.labels[agg.owner] = {
                field: torch.tensor(
                    column_token_ids(
                        executor.bundle.tokenizer,
                        [
                            row[
                                concrete_str(
                                    metric.fields[field], f"metric field {field}"
                                )
                            ]
                            if metric.token_form == "id"
                            else str(
                                row[
                                    concrete_str(
                                        metric.fields[field], f"metric field {field}"
                                    )
                                ]
                            )
                            for row in executor.rows_for_metrics()
                        ],
                        token_form=metric.token_form,
                        where=f"metric {metric.kind}.{field}",
                    ),
                    dtype=torch.long,
                    # beside the logits they index: the head's device
                    device=executor.bundle.devices.head,
                )
                for field in fields
            }

    def for_executor(self, executor: PointExecutor) -> TrainingObjective:
        return TrainingObjective(
            executor, {name: executor.stage(name) for name in self.stages}
        )

    def copy_labels(self, other: TrainingObjective) -> None:
        for name, fields in self.labels.items():
            for field, value in fields.items():
                value.copy_(other.labels[name][field])

    def __call__(self) -> torch.Tensor:
        mb = self.executor
        train = mb.doc.train
        assert train is not None
        # the loss scalar follows the ``lm_head`` read: the head's device;
        # a regularizer over a stage placed on another block's device joins
        # it there (a 0-d move, differentiable)
        loss = torch.zeros((), device=mb.bundle.devices.head)
        for index, term_spec in enumerate(train.objective):
            if term_spec.constraint is not None:
                # refused before capture in `_prepare_fit`; said again here so a
                # graph objective can never silently drop the duals
                raise ProtocolError(
                    "P4",
                    f"{term_spec.path(0)}: a constraint term's duals step on the "
                    "eager loop, not inside a captured graph",
                )
            weight = term_spec.weight
            agg = (
                mb.doc.aggregation_at(term_spec.path(index))
                if term_spec.aggregation is not None
                else None
            )
            target = agg if agg is not None else term_spec.regularizer
            w = float(weight) if isinstance(weight, (int, float)) else 1.0
            if isinstance(target, BoundAggregation):
                metric = target.spec
                of_value = mb.dense_value(target.read)
                target_value = (
                    mb.dense_value(target.target)
                    if metric.kind == "kl" and target.target is not None
                    else None
                )
                per_row = metric_tensor(
                    metric,
                    of_value,
                    mb.rows_for_metrics(),
                    mb.bundle.tokenizer,
                    target_value=target_value,
                    token_ids=self.labels[target.owner],
                )
                if self.weight is None:
                    term = per_row.mean()
                else:
                    term = (per_row * self.weight).sum() / self.weight.sum()
            elif isinstance(target, tuple):
                reg_kind, reg_target = target
                term = _regularizer(
                    reg_kind,
                    reg_target,
                    self.stages,
                    term_spec.reduce or "mean",
                    term_spec.costs,
                )
            else:
                raise ProtocolError("P2", f"unresolvable objective term {target!r}")
            loss = loss + w * term.to(loss.device)
        return loss


def run_training(
    doc: Document,
    executor: PointExecutor,
    run: RunContext,
    *,
    graph_cache: FitGraphCache | None = None,
) -> TrainOutcome:
    """Fit one point: [`run_cohort_training`][] over a cohort of one."""
    return run_cohort_training([doc], [executor], run, graph_cache=graph_cache)[0]


def run_cohort_training(
    docs: Sequence[Document],
    executors: Sequence[PointExecutor],
    run: RunContext,
    *,
    fit_rows: int | None = None,
    meter: Meter | None = None,
    graph_cache: FitGraphCache | None = None,
    graph_pool: GraphPool | None = None,
) -> list[TrainOutcome]:
    """Fit the points of one cohort together (module docstring); one outcome
    per point, in order.

    Each ``executor`` is its point's full-data executor — its stage cache is
    shared with every training minibatch, so the stages it later evaluates
    are the fitted ones. Every point's outcome carries the trained stages by
    featurizer name (for the save manifest) **and** the ``train.eval`` score.

    The eval score is returned rather than dropped because it used to be
    computed and then consumed only inside the ``early_stop`` branch, so a fit
    document's saved metric table was the **train** score under a name a reader
    took for the eval one. Spec §2.12 says every metric a document declares is
    saved, and a fit is reported with its train and eval scores together. Both
    were unsatisfiable. [`TrainOutcome`][]
    carries it to the run tree as a sibling record — never as a column in the
    metric table, whose rows are a different split.

    **Which weights the returned stages hold.** With ``train.early_stop`` the
    loop selects a fit by its eval score, so the returned stages are the
    *best-scoring* ones, restored from a snapshot taken at each improvement.
    Without ``early_stop`` there is nothing selecting, and they are the last
    ones. ``TrainOutcome.eval_score.selected`` says which, and its ``metrics``
    always describe the weights actually returned.

    ``fit_rows`` bounds the rows of one grad forward: members are packed into
    forwards under it in cohort order, a member's minibatch never split.
    ``None`` measures the bound on the cohort's first step (``budget.py``: a
    one-member probe under peak-memory tracking, unbounded off CUDA), and any
    window that still runs out of memory is retried at half the rows.
    The batched eval passes pack under the engine's ``batch_rows`` when it is
    authored and otherwise under the fit's own bound, in a budget of their own
    (`_advance_eval_budget`): a measured bound's eval window that runs
    out of memory shrinks and retries without touching the grad bound, and
    the shrink is kept for every later pass; an authored bound is fixed for
    eval windows too and re-raises. The outcome reports the smaller of the two
    budgets' bounds as ``fit_rows`` — the number every window of the fit ran
    under, so pinning it is safe — and both budgets' shrinks as one
    ``fit_rows_shrinks``. A grad shrink lowers the grad bound in place, so a
    re-run pinned at the reported number packs exactly as this run did; only
    an eval shrink (``batch_rows`` unauthored) leaves the report below the
    grad bound, and a re-run pinned there packs its grad windows smaller —
    a different rounding, not a different fit. ``meter`` is the device the
    probes read — the tests' seam for a simulated one; the engine leaves it
    to be found from the model (``cuda_meter``).

    ``graph_pool`` is the [`GraphPool`][] every graph of a captured fit
    is captured into — the engine's, kept across fits and requests so the
    segments one fit's graphs grew serve the next (``cuda_graphs.GraphPool``);
    the loop borrows it and leaves it open. Without one (or once it is
    closed by an out-of-memory fallback) the loop opens a pool for this fit
    and releases it in the ``finally``, after every graph holder.
    """
    if len(docs) != len(executors):
        raise ValueError(
            f"{len(docs)} documents but {len(executors)} executors — one per point"
        )
    if len(docs) > 1 and graph_cache is not None:
        # A solo bank cannot serve this layout; release its pool before either
        # a captured or eager cohort allocates its working set.
        graph_cache.close()
    # A cohort of graph-eligible members is captured whole, on a fixed slot
    # layout (graph_cohort.py). If eligibility or the row bound refuses that
    # layout, keep the eager cohort's batching and shared constants.
    cohort_reason: str | None = None
    slot_rows = [_slot_rows(doc, ex) for doc, ex in zip(docs, executors)]
    if (
        any(isinstance(executor, GraphExecutor) for executor in executors)
        and len(docs) > 1
    ):
        cohort_reason = cohort_graph_reason(executors, fit_rows, slot_rows)
    if cohort_reason is not None:
        logging.getLogger(__name__).info("cohort runs eagerly: %s", cohort_reason)
    captured_cohort = (
        cohort_reason is None
        and len(docs) > 1
        and all(isinstance(executor, GraphExecutor) for executor in executors)
    )
    for executor in executors:
        executor.fit_cuda_graphs = cohort_reason is None
        if isinstance(executor, GraphExecutor):
            # a captured cohort's minibatch and eval executors keep the store
            # (shared sources, prefix resume); decided before they are built
            executor.keep_store = len(docs) > 1
    fits = [_prepare_fit(doc, executor) for doc, executor in zip(docs, executors)]
    # the §7 gradient agreement check's tolerance, read once per fit from
    # ``CAUSALAB_GRADIENT_AGREEMENT`` (``agreements.AGREEMENT_VARIABLE``): a
    # malformed value is refused here, before any forward, on every rank
    # alike; unset, the guard is the plain mean and costs nothing
    agreement = configured_agreement()
    if meter is None and executors:
        meter = cuda_meter(executors[0].bundle.devices)
    # the cohort's collective is its executors' (docs/model_parallelism.md
    # §3): the budget's two agreements and the gradient guard run through it
    # — over the data replicas too when the fit's rows are split (§8.3)
    collective: Collective = executors[0].fragments.collective if executors else SOLO
    axes = fits[0].rows.budget_axes if fits else RowBudget.axes
    oom_policy = (
        OOMPolicy.ABORT
        if any(
            OOMPolicy.for_geometry(ex.bundle.geometry) is OOMPolicy.ABORT
            for ex in executors
        )
        else OOMPolicy.RETRY
    )
    budget = RowBudget.of(fit_rows, meter, collective, axes, oom_policy=oom_policy)
    batch_rows = executors[0].batch_rows if executors else None
    eval_budget: RowBudget | None = None
    # the floor of a measured bound: the smallest minibatch any member will
    # ever step — known before the loop, so it does not depend on which
    # minibatch the shuffle draws first (an epoch's last one is a remainder)
    unit = min((len(rows) for fit in fits for rows in fit.slices), default=None)
    cohort_graphs: CohortGraphs | None = None
    evaluation_graphs: EvaluationGraphs | None = None
    # one allocator pool for every graph the fit captures (GraphPool): a
    # cohort's step and evaluation graphs (graph_cohort.py, "Memory") and its
    # members' inference replays here; a solo fit's buckets and inference
    # replays on its bank's, below. The loop owns the pools it opens and
    # releases them in the finally, after every graph holder; a cached bank's
    # pool belongs to the FitGraphCache.
    cohort_pool: GraphPool | None = None
    # the engine's pool while it is open; otherwise `own_pool`, the one pool
    # this call opens — and the only one it releases, in the finally
    borrowed = graph_pool if graph_pool is not None and not graph_pool.closed else None
    own_pool: GraphPool | None = None

    def fit_pool() -> GraphPool:
        nonlocal own_pool
        if borrowed is not None:
            return borrowed
        if own_pool is None:
            own_pool = GraphPool()
        return own_pool

    if captured_cohort:
        cohort_pool = fit_pool()
        # every graph holder takes the fit's policy: under a sharded geometry
        # an out-of-memory capture or replay ends the run, as a window's does
        evaluation_graphs = EvaluationGraphs(pool=cohort_pool, oom_policy=oom_policy)
        for fit in fits:
            fit.graph_pool = cohort_pool
        cohort_graphs = CohortGraphs(
            [
                Member(
                    key=id(fit),
                    executor=fit.executor,
                    stages=fit.stages,
                    parameters=[
                        p
                        for group in fit.optimizer.param_groups
                        for p in group["params"]
                    ],
                    objective_reads=fit.objective_reads,
                    pairs=pairs,
                )
                for fit, pairs in zip(fits, slot_rows, strict=True)
            ],
            make_objective=TrainingObjective,
            pool=cohort_pool,
            oom_policy=oom_policy,
        )
    for fit in fits:
        # the held-out capture cache is a solo-fit affair: the members of a
        # captured cohort evaluate together, each on its own eval executor
        fit.executor.graph_cache = (
            graph_cache
            if isinstance(fit.executor, GraphExecutor) and len(fits) == 1
            else None
        )
    graphs = None
    success = False
    if len(fits) == 1 and isinstance(fits[0].executor, GraphExecutor):
        parameters = [
            p for group in fits[0].optimizer.param_groups for p in group["params"]
        ]
        if graph_cache:
            graphs = graph_cache.begin(
                fits[0].executor, parameters, oom_policy=oom_policy
            )
        else:
            graphs = TrainingGraphs(parameters, pool=fit_pool(), oom_policy=oom_policy)
        fits[0].graph_pool = graphs.pool
    try:
        while True:
            stepping = [fit for fit in fits if fit.active]
            if not stepping:
                break
            current: list[tuple[_Fit, PointExecutor]] = []
            for fit in stepping:
                if fit.position >= len(fit.order):
                    fit.order = torch.randperm(
                        len(fit.batches), generator=fit.order_rng
                    ).tolist()
                    fit.position = 0
                    if fit.drawn is not None and fit.step > 0:
                        # §2.2 `draw`: a new member per row for the new epoch
                        # (the first epoch's draw was taken at prepare)
                        fit.drawn.redraw(fit)
                pools_drawn: set[int] = set()
                for stage in fit.stages.values():
                    stage.train(True)
                    if isinstance(stage, Gate) and stage.samples_per_step:
                        # One draw per step (concrete noise or budget k), shared
                        # by the read and write, from this member's generator.
                        # A budget pool draws once for all its gates (§2.5).
                        if stage.pool is not None:
                            if id(stage.pool) not in pools_drawn:
                                stage.pool.resample(fit.mask_rng)
                                pools_drawn.add(id(stage.pool))
                            continue
                        stage.resample(fit.mask_rng)
                _advance_phase(fit)
                for dotted, schedule in fit.anneals.items():
                    _set_anneal(fit, dotted, schedule)
                if fit.phase_index >= 0:
                    phase = fit.phases[fit.phase_index]
                    for dotted, schedule in phase.anneals.items():
                        # a phase's schedule runs over the phase's own steps
                        _set_anneal(
                            fit,
                            dotted,
                            schedule,
                            step=fit.step - phase.start,
                            total_steps=phase.end - phase.start,
                        )
                minibatch = fit.minibatch_executors[fit.order[fit.position]]
                minibatch.reset_reads()
                fit.optimizer.zero_grad()
                current.append((fit, minibatch))
            _run_step_windows(
                current,
                budget,
                unit,
                graphs=graphs,
                cohort_graphs=cohort_graphs,
                agreement=agreement,
            )
            due: list[_Fit] = []
            for fit, minibatch in current:
                _apply_lr_schedule(fit)
                fit.optimizer.step()
                for stage in fit.stages.values():
                    stage.project()  # back onto the feasible set (a clamp gate)
                # under a pipeline the featurizer's gradient is made on the
                # stage owning its site alone (§6.5, §7): every rank — the
                # publisher included — takes the owner's parameters after the
                # step, before anything reads them (a checkpoint, the eval's
                # snapshot, a controller). The identity at world 1.
                fit.owner.sync(fit.stages.values())
                fit.step += 1
                fit.position += 1
                # Release this step's captured reads now, not when the minibatch
                # comes round again an epoch later. A member's last-token logits
                # over the vocabulary are about 4 MB bf16 per minibatch, and one
                # parked per member per step is members x minibatches of them by
                # the end of the epoch: at 80 members x 224 minibatches that is
                # about 70 GB, which exhausts an 80 GB device before the first
                # eval.
                minibatch.reset_reads()
            # the members' controllers observe their gates in one host read,
            # after every member has stepped (the members are independent, so
            # the order of stepping and observing across them changes nothing)
            signals = _read_signals([fit for fit, _ in current])
            for fit, _ in current:
                _after_update(fit, signals)
                if fit.position >= len(fit.order) or fit.step >= fit.total_steps:
                    # an epoch ended — complete, or cut short by an `updates`
                    # budget; either way the eval it owes runs now
                    fit.epoch += 1
                    if (
                        fit.eval_every_epochs is not None
                        and fit.epoch % fit.eval_every_epochs == 0
                    ):
                        due.append(fit)
            if due:
                eval_budget = _advance_eval_budget(batch_rows, budget, eval_budget)
                _evaluate(due, run, eval_budget, graphs=evaluation_graphs)
            for fit, _ in current:
                if fit.exhausted:
                    fit.active = False
        # the bound reported is the one every window of the fit ran under — the
        # grad budget's, or the eval budget's once an eval window shrank below it
        # — so the number an author pins is one no window of the run refused; an
        # authored `batch_rows` governs the eval windows on its own and says
        # nothing about `fit_rows`, and a grad budget that never resolved (off
        # CUDA) stays `None`, since `null` means every member in one forward and
        # an eval shrink there changed no grad window. The shrinks of both budgets
        # are one count: the bound was too loose by that many, whichever kind of
        # window found out (an eval budget under an authored `batch_rows` is fixed
        # and never shrinks). A grad shrink lowered the grad bound in place, so
        # a re-run pinned at the report packs as this run did; only an eval shrink
        # leaves the report below the grad bound, and a re-run pinned there packs
        # its grad windows smaller
        reported = budget.bound
        if (
            reported is not None
            and batch_rows is None
            and eval_budget is not None
            and eval_budget.bound is not None
        ):
            reported = min(reported, eval_budget.bound)
        shrinks = budget.shrinks + (
            eval_budget.shrinks if eval_budget is not None else 0
        )
        outcomes = [
            dataclasses.replace(
                _finish(fit), fit_rows=reported, fit_rows_shrinks=shrinks
            )
            for fit in fits
        ]

        success = True
        return outcomes
    finally:
        if evaluation_graphs is not None:
            evaluation_graphs.close()
        if cohort_graphs is not None:
            cohort_graphs.close()
        if graph_cache is not None:
            graph_cache.finish(success=success)
        elif graphs is not None:
            graphs.close()
        for fit in fits:
            evaluation = fit.executor.eval_executor
            if isinstance(evaluation, GraphExecutor) and (
                graph_cache is None or evaluation is not graph_cache.eval_executor
            ):
                evaluation.close()
        # every graph holder of the fit is closed: release the pool the loop
        # opened. A cached bank's pool stays with the bank (FitGraphCache),
        # and the engine's stays with the engine, its blocks free for the
        # next fit's captures.
        if own_pool is not None:
            own_pool.close()


def _advance_eval_budget(
    batch_rows: int | None, budget: RowBudget, previous: RowBudget | None
) -> RowBudget:
    """The budget the batched eval passes pack under, built on the first
    pass and advanced on every later one — one object for the whole fit, so
    what an eval window learns by running out of memory is kept for every
    later pass rather than re-learnt each time.

    An authored ``batch_rows`` bounds every no-grad forward of the run, so it
    is the fixed bound here. Otherwise the eval passes pack under the fit's
    own bound, so one number (``fit_rows_resolved``) pins both kinds of
    window: a budget seeded from the grad bound as it stands and re-seeded
    **down** to it on every pass (the grad bound only ever shrinks), never
    up past what an eval window already failed at. It is fixed exactly when
    the grad bound is — an authored ``fit_rows`` is never shrunk for eval
    windows either (§8) — and shrinks on its own otherwise: the grad bound
    never moves on an eval window's account, while ``fit_rows_shrinks`` counts
    the windows of both kinds that packed under the fit's bound and did not
    fit.
    """
    # both eval budgets agree their out-of-memory windows over the same group
    # the grad budget does (§3)
    if batch_rows is not None:
        return (
            previous
            if previous is not None
            else RowBudget.of(
                batch_rows,
                meter=None,
                collective=budget.agreements.collective,
                axes=budget.axes,
                oom_policy=budget.oom_policy,
            )
        )
    if previous is None:
        return RowBudget(
            bound=budget.bound,
            fixed=budget.fixed,
            meter=None,
            resolved=True,
            agreements=budget.agreements,
            axes=budget.axes,
            oom_policy=budget.oom_policy,
        )
    if budget.bound is not None and (
        previous.bound is None or budget.bound < previous.bound
    ):
        previous.bound = budget.bound
    return previous


def _run_step_windows(
    current: Sequence[tuple[_Fit, PointExecutor]],
    budget: RowBudget,
    unit: int | None,
    *,
    graphs: TrainingGraphs | None = None,
    cohort_graphs: CohortGraphs | None = None,
    agreement: float | None = None,
) -> None:
    """One optimizer step's grad forwards: the members packed into windows
    under the budget (``budget.py``), each window one forward, one summed
    loss and one backward. A window the device cannot hold is retried at a
    smaller bound: its members have not stepped, so their gradients are
    zeroed and the window re-packed (`_run_windows`).

    A captured cohort (``cohort_graphs``) takes the whole step first — every
    stepping member in its slot, one replay — and the budget is not consulted;
    a step it declines runs as the eager cohort below. Either way the step's
    reductions follow: the sum over a rows split's replicas, the record's
    agreement and the mean over the model group. Captured work also
    bypasses the eager per-member tally brackets, so receipt forward/reuse
    counts are not comparable to eager counts; the graph benchmark reports
    captures and replays separately.

    Each eager window runs under one [`featurizer_cache`][causalab.neural.shared.featurizers.sharing.featurizer_cache] scope: a member's
    rotation or mask is evaluated once and serves its read, every write's
    featurize and inverse and — for a subspace, whose penalty reads the
    rotation through ``slot_params`` — the regularizer, with the gradient
    reaching the parameter through that one evaluation. A gate's ``l1``
    penalty reads ``soft_mask()`` itself, one evaluation beside the shared
    table: under ``hard_concrete`` the table is the sampled mask and the
    penalty the deterministic one, and a ``head`` / ``site`` group's penalty
    wants the mask before its expansion over the group's coordinates. The
    scope closes before the optimizer moves the parameter.

    ``agreement`` is the §7 guard's runtime check (``agreements.py``): the
    tolerance the ranks' gradients are held to before the mean, ``None``
    for no check."""
    replayed = False
    if cohort_graphs is not None and not cohort_graphs.disabled:
        items: list[WindowItem] = []
        for fit, minibatch in current:
            if minibatch not in fit.graph_objectives:
                fit.graph_objectives[minibatch] = TrainingObjective(
                    minibatch, fit.stages
                )
            items.append(
                WindowItem(
                    key=id(fit),
                    indices=fit.slices[fit.order[fit.position]],
                    minibatch=minibatch,
                    objective=fit.graph_objectives[minibatch],
                )
            )
        # a replayed step leaves the gradients where an eager one does, and
        # the reductions below follow it the same way
        replayed = cohort_graphs.backward(items)

    def body(window: Sequence[tuple[_Fit, PointExecutor]]) -> None:
        if graphs is not None and not graphs.disabled and len(window) == 1:
            # no scope of ours around the graph path: every captured pass
            # opens its own, isolated one (`cuda_graphs.captured_pass`), which
            # is what keeps the map inside the graph rather than baked in
            # stale — so sitting outside the eager scope below is a choice,
            # not a necessity; the eval round's `graphs.forward` runs inside
            # one (`_evaluate`) on the same guarantee
            fit, minibatch = window[0]
            if minibatch not in fit.graph_objectives:
                fit.graph_objectives[minibatch] = TrainingObjective(
                    minibatch, fit.stages
                )
            objective = fit.graph_objectives[minibatch]
            if graphs.backward(minibatch, objective):
                return
        with featurizer_cache():
            _run_batched(
                [(fit, minibatch, fit.objective_reads) for fit, minibatch in window]
            )
            loss = torch.zeros(())
            for fit, minibatch in window:
                with _tally(fit):
                    # under a rows split each replica's loss carries its share
                    # of the minibatch's rows, so the replicas' gradients sum
                    # to the unsplit minibatch's (rows.py); the loss itself at
                    # world 1
                    loss = loss + fit.rows.weigh(
                        _loss(fit, minibatch), *_step_rows(fit, minibatch)
                    )
            loss.backward()

    def abandon(window: Sequence[tuple[_Fit, PointExecutor]]) -> None:
        for fit, minibatch in window:
            fit.optimizer.zero_grad()
            minibatch.reset_reads()

    if not replayed:
        _run_windows(current, budget, body, abandon, unit)
    # §7: every trained parameter's gradient is the mean over the
    # model-parallel group — identical already at a replicated site, a guard
    # against a future sharded one; the identity at world 1. After the
    # step's windows, not inside one, so solo/data-row bodies can retry OOMs
    # before these reductions. Sharded model bodies contain other collectives
    # and abort instead. The mean is linear, so averaging accumulated gradients
    # equals averaging each window's gradients.
    # Under a rows split (§8.3) the gradient is first summed over the data
    # replicas — the unsplit minibatch's gradient, the shares having carried
    # the 1/N — and the update's record (the loss and its terms) agreed the
    # same way, before any controller reads it (``_after_update``).
    for fit, minibatch in current:
        parameters = [
            p for group in fit.optimizer.param_groups for p in group["params"]
        ]
        fit.rows.reduce_gradients(parameters)
        _agree_record(fit, minibatch)
        average_gradients(
            parameters,
            minibatch.fragments.collective,
            axis="model",
            agreement=agreement,
        )


def _step_rows(fit: _Fit, minibatch: PointExecutor) -> tuple[int, int]:
    """``minibatch`` — one of the fit's minibatch executors — as ``(rows this
    replica steps, rows the minibatch has)``; equal unless the data axis is
    split over rows. Read off the executor, not the fit's position: a step's
    windows are driven with the minibatch in hand."""
    index = fit.minibatch_executors.index(minibatch)
    return len(fit.slices[index]), len(fit.batches[index])


def _device(fit: _Fit) -> torch.device:
    """The device a fit's agreed scalars are reduced on: the model's head
    device (one device per rank under a launched world)."""
    return fit.executor.bundle.devices.head


def _agree_record(fit: _Fit, minibatch: PointExecutor) -> None:
    """The update's record — ``last_loss`` and the ``term.*`` values — as
    the means over every replica's rows ([`RowSplit.agree_record`][]), so
    a trajectory checkpoint and a controller read one number on every
    replica. Nothing at an inactive split: the record is already the fit's."""
    if not fit.rows.active:
        return
    names = [name for name in fit.term_values if name.startswith("term.")]
    agreed = fit.rows.agree_record(
        [fit.last_loss, *(fit.term_values[name] for name in names)],
        *_step_rows(fit, minibatch),
        device=_device(fit),
    )
    fit.last_loss = agreed[0]
    for name, value in zip(names, agreed[1:], strict=True):
        fit.term_values[name] = value


def _run_windows(
    items: Sequence[tuple[_Fit, PointExecutor]],
    budget: RowBudget,
    body: Callable[[Sequence[tuple[_Fit, PointExecutor]]], None],
    abandon: Callable[[Sequence[tuple[_Fit, PointExecutor]]], None],
    unit: int | None,
) -> None:
    """Run rank-agreed windows, retrying only collective-free auto budgets.

    ``abandon`` clears a failed window before releasing cached memory and
    retrying at a smaller agreed bound. Fixed budgets and single members
    cannot shrink. Bodies with model collectives abort immediately on OOM,
    without entering a retry agreement that peers may never reach.

    The release happens *after* the ``except`` block, deliberately: while the
    handler runs, the exception's traceback still holds the failed body's
    frames — the loss at the root of the window's autograd graph, the capture
    dict — so every activation the forward allocated is still referenced, and
    ``empty_cache`` inside the handler would return nothing. Leaving the
    handler drops the traceback first.

    ``unit`` is the floor of a measured bound (``RowBudget.run``): the
    smallest window any member will bring, decided by the caller over the
    whole fit rather than over this step's items. Only a budget still probing
    reads it; a resolved or fixed one (every eval budget) ignores it.

    A failed attempt's tallies (``fit_forwards``, ``prefix_reuse``) stay: its
    forwards did run. A fit whose windows shrank — of either kind, under the
    fit's bound — says so in its receipt (``fit_rows_shrinks``).

    For retryable bodies, all ranks agree failure, shrink eligibility and
    the new bound. A successful peer abandons the same window too. When no
    smaller window exists, each rank raises its own or its peer's OOM.
    """
    pending = list(items)
    while pending:
        window, rest = budget.take(pending, _window_rows)
        rows = sum(_window_rows(item) for item in window)
        largest = max(_window_rows(item) for item in window)
        failure: torch.OutOfMemoryError | None = None
        try:
            budget.run(rows, lambda: body(window), unit)
        except DistributedOutOfMemory:
            raise
        except torch.OutOfMemoryError as error:
            failure = error.with_traceback(None)
        if not budget.out_of_memory(failure is not None):
            pending = rest
            continue
        if not budget.can_shrink(rows, largest):
            if failure is not None:
                raise failure
            raise torch.OutOfMemoryError(
                f"a rank of the model-parallel group ran a window of {rows} rows "
                "out of memory, and no smaller window exists"
            )
        abandon(window)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        budget.shrink(rows, largest)


def _derived(source: PointExecutor, built: PointExecutor) -> PointExecutor:
    """An executor built for a slice or a split of ``source``'s point runs
    over the same ranks: it inherits the point executor's ``fragments``
    (``docs/model_parallelism.md`` §4) — set after construction, as the
    executor's other class-level defaults are, so ``make_executor``'s
    signature stays the base constructor's — and the same rows split
    (§8.3), so every executor of one fit agrees which replica it is."""
    built.fragments = source.fragments
    built.rows = source.rows
    return built


def _window_rows(item: tuple[_Fit, PointExecutor]) -> int:
    """The rows one member brings to a packed window: its executor's."""
    return len(item[1].rows_for_metrics())


def _prepare_fit(doc: Document, executor: PointExecutor) -> _Fit:
    train = doc.train
    assert train is not None
    # one reader of train.seed, shared with the featurizer inits the executor
    # builds below — the loop and the init cannot disagree about the seed.
    # Seeded per member, right before its stages are built: a member's init
    # is a function of its own document, whatever fitted beside it
    seed = document_seed(doc)
    torch.manual_seed(seed)

    trained_names = tuple(sorted({p.split(".", 1)[0] for p in train.params}))
    stages: dict[str, Stage] = {}
    parameters: list[torch.nn.Parameter] = []
    # one optimizer parameter group per `train.params` entry, so a per-parameter
    # `lr` / `weight_decay` (§2.11, a mapping keyed by those entries) lands on
    # exactly the tensors its entry names; with scalar settings the groups
    # share every hyperparameter and the optimizer's arithmetic is that of
    # one flat list (Adam's state is per tensor either way)
    groups: list[dict[str, Any]] = []
    for pname in train.params:
        fname, _, slot = pname.partition(".")
        if fname not in doc.featurizers:
            # The compile refuses this under §5 rule 30 (this engine declares
            # no 'train_free_params'); here for a document that arrived
            # unvalidated, so routing should not have sent it here.
            raise ProtocolError(
                "P4",
                f"free params entries ({pname!r}) are not trainable in this "
                "engine — featurizer slots only; validation refuses this at "
                "load (rule 30), so this document arrived unvalidated",
            )
        stage = executor.stage(fname)
        stages[fname] = stage
        if slot:
            group_params = [stage.slot_params()[slot]]
        else:
            group_params = [p for p in stage.parameters() if p.requires_grad]
        parameters.extend(group_params)  # type: ignore[arg-type]
        group: dict[str, Any] = {"params": group_params}
        for field in PER_PARAMS_OPTIMIZER_FIELDS:
            value = train.optimizer.get(field)
            if isinstance(value, Mapping):
                # the schema checked every entry of train.params is a key
                group[field] = float(value[pname])
        groups.append(group)
    if not parameters:
        raise ProtocolError("P2", "train.params resolved to no trainable tensors")

    optimizer = _build_optimizer(train.optimizer, groups)
    groups_by_entry = {pname: i for i, pname in enumerate(train.params)}
    duals = _add_dual_groups(train, optimizer, executor)
    constraint_specs = {
        term.name: term.constraint
        for term in train.objective
        if term.constraint is not None and term.name is not None
    }

    n_examples = len(executor.rows_for_metrics())
    pairs = concrete_int(train.batch["pairs"], "train.batch.pairs")
    batches = [
        list(range(start, min(start + pairs, n_examples)))
        for start in range(0, n_examples, pairs)
    ]
    # data parallelism over rows (docs/model_parallelism.md §8.3): this
    # replica steps its contiguous slice of every minibatch — the whole
    # minibatch at world 1 and over points; a minibatch with fewer rows than
    # replicas (an epoch's remainder) is refused by name, on every rank alike
    rows = executor.rows
    slices = [rows.slice_for(indices) for indices in batches]
    # under a pipeline (§7, §8.3) the stage owning the trained featurizers'
    # sites computes their gradients; two stages are refused here, before
    # any forward, identically on every rank
    owner = executor.trained_owner(trained_names)
    if "epochs" in train.steps:
        epochs = concrete_int(train.steps["epochs"], "train.steps.epochs")
        total_steps = epochs * len(batches)
    else:
        total_steps = concrete_int(train.steps["updates"], "train.steps.updates")
        epochs = -(-total_steps // len(batches))

    eval_every_epochs = None
    if train.eval is not None:
        if "epochs" not in train.eval["every"]:
            # This loop only reaches an eval on an epoch boundary. Accepting an
            # `updates` counter here would run *no* eval at all and still save
            # the fit, which is the silent-wrong-number this commit exists to
            # remove — so refuse instead of pretending. The compile refuses
            # this under §5 rule 30 (this engine declares no
            # 'train_eval_updates'); here for a document that arrived
            # unvalidated.
            raise ProtocolError(
                "P4",
                "train.eval.every must count epochs in this engine — "
                f"got {sorted(train.eval['every'])}; an update counter would "
                "silently never evaluate",
            )
        eval_every_epochs = concrete_int(
            train.eval["every"]["epochs"], "train.eval.every.epochs"
        )

    # A minibatch's rows are a *slice* of the campaign's, so its captures live
    # under `(key, indices)` in the shared ForwardCache — never under the
    # whole role's key — and only the groups this fit cannot change are ever
    # read from or written to it (`inner_interning`, §4 "Fits"). For the
    # shipped methods that is the source forward: run once per slice here,
    # then served on every step, epoch and point that shares the group key.
    # The minibatch is a row *selection* of the point's frame, not a fresh
    # encode of its rows: every minibatch of every point in a cohort is then
    # in one padded frame, which is what lets their forwards concatenate.
    # No `batch_rows=`: `train.batch.pairs` is the document's own batching
    # knob for grad forwards, so the execution bound applies to the no-grad
    # passes and to `train.eval` (below), not to a minibatch.
    # the draw's refusals first: they read the rows, not the frames, and a
    # refused document should not pay for an encode
    drawn = _Drawn.of(doc, executor, seed)
    frames = {role: executor.frame(role) for role in executor.role_rows}
    if drawn is not None:
        # §2.2 `draw`: this fit's minibatches read a freshly drawn member per
        # row each epoch, encoded as selections of one expanded frame; the
        # inner store is not consulted for them (a source forward over a
        # drawn role changes every epoch, so nothing about it is constant).
        # Under `dp=N:rows` (§8.3) the draw is the same on every replica —
        # one seeded stream — and each replica selects its slice of it
        drawn.bind(slices, frames)
        minibatch_executors = drawn.minibatches()
    else:
        minibatch_executors = [
            _derived(
                executor,
                make_executor(
                    doc,
                    executor.bundle,
                    cuda_graphs=executor.cuda_graphs and executor.fit_cuda_graphs,
                    role_rows=_slice_rows(executor.role_rows, indices),
                    role_fields=executor.role_fields,
                    load_tensors=executor.load_tensors,
                    load_table=executor.load_table,
                    stage_cache=executor.stage_cache,  # shared: one stage per name
                    grad_enabled=True,
                    coords=executor.coords,
                    interning=_inner_interning(executor, tuple(indices)),
                    batches={
                        role: frame.select(indices) for role, frame in frames.items()
                    },
                ),
            )
            for indices in slices
        ]
    objective_reads: list[ReadRef] = []
    for agg in doc.objective_aggregations():
        objective_reads.append(agg.read)
        if agg.target is not None:
            objective_reads.append(agg.target)
    # §2.11 `control`: the closed-loop schedules, and the live value of every
    # named term's weight (the authored weight is a controller's start)
    live_weights: dict[str, float] = {
        term.name: float(term.weight)
        for term in train.objective
        if term.name is not None and isinstance(term.weight, (int, float))
    }
    controls = _build_controls(train.control, stages, live_weights)
    anneals = _parse_anneals(train.anneal, stages, live_weights)
    phases = _build_phases(train.phases, total_steps, stages, live_weights)
    fit = _Fit(
        doc=doc,
        executor=executor,
        seed=seed,
        stages=stages,
        trained_names=trained_names,
        optimizer=optimizer,
        batches=batches,
        slices=slices,
        rows=rows,
        owner=owner,
        minibatch_executors=minibatch_executors,
        epochs=epochs,
        total_steps=total_steps,
        anneals=anneals,
        eval_every_epochs=eval_every_epochs,
        order_rng=torch.Generator().manual_seed(seed),
        # its own generator object at the same seed, as the batch order and
        # the subspace init have theirs: a local stream, never the global one
        mask_rng=torch.Generator().manual_seed(seed),
        objective_reads=tuple(objective_reads),
        live_weights=live_weights,
        controls=controls,
        control_trace={target: [] for target in controls},
        duals=duals,
        constraint_specs=constraint_specs,
        constraint_trace={name: [] for name in duals},
        phases=phases,
        drawn=drawn,
        groups_by_entry=groups_by_entry,
        # §2.12 `trajectory`: the updates after which the fit is photographed
        checkpoint_steps=frozenset(_checkpoint_steps(doc, total_steps, len(batches))),
    )
    return fit


@contextlib.contextmanager
def _tally(fit: _Fit) -> Iterator[None]:
    """Attribute the passes run inside to ``fit``: the constant groups the
    store ran and served, and the forwards that resumed from a prefix, as a
    difference over the shared tallies — per member, since a cohort's passes
    interleave several points'. The one forward several members share (a
    cohort forward) is credited to each of them by `_run_batched`
    instead, outside any member's bracket."""
    store = fit.store
    if store is None:
        yield
        return
    run_before, served_before = len(store.inner_executed), len(store.inner_served)
    resumed_before = len(store.resumed)
    try:
        yield
    finally:
        fit.run += len(store.inner_executed) - run_before
        fit.served += len(store.inner_served) - served_before
        fit.resumed.extend(store.resumed[resumed_before:])


def _run_batched(
    members: Sequence[tuple[_Fit, PointExecutor, Sequence[ReadRef]]],
    *,
    frames: Mapping[str, EncodedBatch] | None = None,
) -> None:
    """Run the groups ``reads`` need on every member's executor, batched
    across members where the group admits it (``cohort.batchable``): one
    forward per input role over the members' rows together. Each member's
    operand reads — the source groups the store serves — are run first, on
    the member's own tally; a group left out runs on its executor's own path
    when the read is asked for. ``frames`` is the members' concatenated batch
    per role when the caller holds a prepared one (an eval layout's,
    ``EvaluationGraphs.frames_for``); by default it is built per forward."""
    for fit, executor, reads in members:
        with _tally(fit):
            for model, _role in groups_read_by(executor.doc, reads):
                for ename in write_names(executor.doc, model) or ():
                    for ref in operand_reads(
                        executor.doc, executor.doc.writes[ename].do
                    ):
                        executor.read_value(ref)
    by_role = cohort_entries([(executor, reads) for _, executor, reads in members])
    owner = {id(executor): fit for fit, executor, _ in members}
    for role, entries in by_role.items():
        store = members[0][0].store
        before = len(store.resumed) if store is not None else 0
        frame = frames.get(role) if frames else None
        if frame is None:
            run_groups(entries)
        else:
            run_groups(entries, frame=frame)
        if store is not None:
            resumed = store.resumed[before:]
            for entry in entries:
                owner[id(entry.executor)].resumed.extend(resumed)


def _loss(fit: _Fit, minibatch: PointExecutor) -> torch.Tensor:
    """This update's objective for one member, and — on the member — the
    record of it (``last_loss``, ``term_values``) a checkpoint taken after
    the update carries. A named term's weight is its *live* value: the
    authored one until a controller moves it (§2.11)."""
    loss = torch.zeros(())
    term_values: dict[str, torch.Tensor | float] = {}
    for index, term in enumerate(fit.train.objective):
        w = float(term.weight) if isinstance(term.weight, (int, float)) else 1.0
        if term.name is not None:
            w = fit.live_weights.get(term.name, w)  # a controlled weight moves
        agg = (
            fit.doc.aggregation_at(term.path(index))
            if term.aggregation is not None
            else None
        )
        if agg is not None:
            metric = agg.spec
            of_value = minibatch.dense_value(agg.read)
            target_value = (
                minibatch.dense_value(agg.target) if agg.target is not None else None
            )
            value = metric_tensor(
                metric,
                of_value,
                minibatch.rows_for_metrics(),
                minibatch.bundle.tokenizer,
                target_value=target_value,
            ).mean()
        else:
            assert term.regularizer is not None
            kind, targets = term.regularizer
            value = _regularizer(
                kind, targets, fit.stages, term.reduce or "mean", term.costs
            )
        if term.constraint is not None:
            # §2.11 `constraint`: λ₁(s − t) + λ₂(s − t)², the duals ascended by
            # their own optimizer group — the term has no weight
            assert term.name is not None
            lam = fit.duals[term.name]
            gap = value - term.constraint.target
            loss = loss + lam[0] * gap + lam[1] * gap * gap
            term_values[f"term.{term.name}"] = float(value.detach())
            term_values[f"lambda1.{term.name}"] = float(lam[0].detach())
            term_values[f"lambda2.{term.name}"] = float(lam[1].detach())
            continue
        loss = loss + w * value
        term_values[f"term.{term.name or index}"] = value.detach()
        term_values[f"weight.{term.name or index}"] = w
    fit.last_loss = loss.detach()
    fit.term_values = term_values
    return loss


def _evaluate(
    due: Sequence[_Fit],
    run: RunContext,
    budget: RowBudget,
    *,
    graphs: EvaluationGraphs | None = None,
) -> None:
    """One eval pass for every fit in ``due`` (§2.11), then each fit's
    early-stop bookkeeping. One fit runs its own pass (`_run_eval`);
    several run the trained groups batched across the fits whose split
    agrees, packed under ``budget`` (`_advance_eval_budget`), and are
    scored on the result.

    Under a captured cohort, ``graphs`` serves the pass: the fits due on one
    split that the budget would run as one window are one replay of the
    cohort's eval layout (``EvaluationGraphs``) — the layout's first pass
    runs eagerly on its prepared frame, and a single fit still due is served
    from its slot rather than a pass of its own.
    """
    scores: dict[int, dict[str, float]] = {}
    if len(due) == 1 and (graphs is None or graphs.bank is None):
        fit = due[0]
        assert fit.train.eval is not None
        split = concrete_str(fit.train.eval["split"], "train.eval.split")
        with _tally(fit):
            if fit.graph_pool is not None:
                # build the fit's eval executor on the fit's pool first; the
                # pass finds it kept on the point executor (`eval_executor`)
                _eval_executor(fit.doc, fit.executor, run, split, pool=fit.graph_pool)
            scores[id(fit)] = _run_eval(fit.doc, fit.executor, run, split)
    else:
        prepared: list[tuple[_Fit, PointExecutor]] = []
        by_split: dict[str, list[tuple[_Fit, PointExecutor]]] = {}
        for fit in due:
            assert fit.train.eval is not None
            split = concrete_str(fit.train.eval["split"], "train.eval.split")
            eval_executor = _eval_executor(
                fit.doc, fit.executor, run, split, pool=fit.graph_pool
            )
            _fresh_for_eval(eval_executor)
            prepared.append((fit, eval_executor))
            by_split.setdefault(split, []).append((fit, eval_executor))

        def members_of(
            window: Sequence[tuple[_Fit, PointExecutor]],
        ) -> list[tuple[PointExecutor, Sequence[str]]]:
            return [(executor, _eval_reads(fit)) for fit, executor in window]

        def body(window: Sequence[tuple[_Fit, PointExecutor]]) -> None:
            _run_batched(
                [
                    (fit, eval_executor, _eval_reads(fit))
                    for fit, eval_executor in window
                ],
                frames=graphs.frames_for(members_of(window))
                if graphs is not None
                else None,
            )

        def abandon(window: Sequence[tuple[_Fit, PointExecutor]]) -> None:
            for _fit, eval_executor in window:
                eval_executor.reset_reads()

        # the replays first, outside any featurizer scope: a capture's
        # warm-up and recording passes run under their own isolated scope
        # (`cuda_graphs.captured_pass`), so every stage is evaluated inside
        # the graph and nothing shared in from eager code is baked into it.
        # The replay is the whole layout's frame, so it stands in for the
        # pass only where the budget would run the group as one window
        replayed: set[str] = set()
        if graphs is not None:
            for split, group in by_split.items():
                _whole, rest = budget.take(group, _window_rows)
                if not rest:
                    if graphs.forward(members_of(group)):
                        replayed.add(split)
                elif graphs.holds(members_of(group)):
                    # the bound fell below this group. It only falls because
                    # the device ran short (`RowBudget.shrink`,
                    # `_advance_eval_budget`; never back up), and the
                    # layout's replay is the fit's widest eval frame — so the
                    # capture is given back now rather than held against a
                    # later due set small enough to fit again (members stop),
                    # and not recaptured at that smaller size: the recapture
                    # churn `CohortGraphs` declines too
                    graphs.invalidate()
        # one scope for the eager remainder of the round: the stages are in
        # eval mode and do not move until the next optimizer step, so every
        # window's forwards and the scoring reads share one evaluation of
        # each stage
        try:
            with featurizer_cache():
                for split, group in by_split.items():
                    if split not in replayed:
                        _run_windows(group, budget, body, abandon, None)
                for fit, eval_executor in prepared:
                    with _tally(fit):
                        scores[id(fit)] = _score(fit.doc, eval_executor)
        finally:
            # scored (or not), released — as `_run_eval` releases its own:
            # device storage nothing needs until the next pass, and a
            # replay's values alias the graph pool the next training replay
            # overwrites, which nothing may still hold by then
            # (cuda_graphs.Replay, ``pool``)
            for _fit, eval_executor in prepared:
                eval_executor.reset_reads()
    for fit in due:
        score = scores[id(fit)]
        fit.eval_passes += 1
        fit.last_score = score
        if fit.train.early_stop is None:
            continue
        metric_name = fit.doc.early_stop_label()
        assert metric_name is not None
        mode = str(fit.train.early_stop["mode"])
        value = score[metric_name]
        improved = (
            fit.best is None
            or (mode == "max" and value > fit.best)
            or (mode == "min" and value < fit.best)
        )
        if improved:
            fit.best, fit.stale = value, 0
            fit.best_state = _snapshot(fit.stages)
            fit.best_score = dict(score)
        else:
            fit.stale += 1
            if fit.stale > concrete_int(fit.train.early_stop["patience"], "patience"):
                fit.active = False


def _run_eval(
    doc: Document,
    executor: PointExecutor,
    run: RunContext,
    split: str,
) -> dict[str, float]:
    """One eval pass over ``split`` for one fit: the declared eval metrics,
    hard-gate eval mode. ``executor`` is the point's full-data executor; the
    pass runs on its `_eval_executor` (built beforehand on the fit's
    pool when the fit captures graphs)."""
    eval_executor = _eval_executor(doc, executor, run, split)
    copy_executor_stages(eval_executor, executor)
    _fresh_for_eval(eval_executor)
    try:
        # after the copy and the mode switch, so the shared evaluation is of
        # the stages this pass scores
        with featurizer_cache():
            return _score(doc, eval_executor)
    finally:
        eval_executor.reset_reads()


def _fresh_for_eval(eval_executor: PointExecutor) -> None:
    for stage in eval_executor.stage_cache.values():
        stage.eval()
    # the stages moved since the last pass, so every read is stale; the
    # encoded split and the interned constant captures are not (reset_reads)
    eval_executor.reset_reads()


def _score(doc: Document, eval_executor: PointExecutor) -> dict[str, float]:
    """The declared eval metrics over ``eval_executor``'s reads — run on its
    own path unless a batched pass or a replay already filled them.

    A kind that only selects entries of the projection at the answer ids
    (``metrics.GATHERED_KINDS`` — the presets' ``iia``) gathers them where
    the read sits — the device, on an executor whose ``device_reads`` is on
    — and copies the one or two columns, never the vocabulary, with the ids
    resolved once per executor (its rows never change); every other kind
    reduces a CPU copy of the whole value in float, as it always did, one
    copy per read however many metrics read it. Either way the numbers are
    the ones the whole-vocabulary CPU path computes, to the bit
    (``metrics.gathered_metric``). A replay's values are graph-owned
    storage: they are consumed here and released by the caller before the
    next replay.

    Under a rows split (the executor's ``rows``, inherited from its point's)
    the executor holds this replica's rows of the split, and each metric's
    ``(sum, count)`` is agreed over the replicas
    ([`RowSplit.agree_means`][]) so the score is one number on every
    replica; the local mean otherwise."""
    assert doc.train is not None and doc.train.eval is not None
    aggregations = doc.eval_aggregations()
    names = [agg.label for agg in aggregations]
    sums: list[float] = []
    counts: list[int] = []
    rows = eval_executor.rows_for_metrics()
    tokenizer = eval_executor.bundle.tokenizer
    host: dict[ReadRef, torch.Tensor] = {}

    def on_host(read: ReadRef) -> torch.Tensor:
        if read not in host:
            host[read] = eval_executor.dense_value(read).detach().cpu()
        return host[read]

    for agg in aggregations:
        name, metric = agg.label, agg.spec
        if str(metric.kind) in GATHERED_KINDS:
            ids = eval_executor.metric_token_ids.get(name)
            if ids is None:
                ids = metric_token_ids(metric, rows, tokenizer)
                eval_executor.metric_token_ids[name] = ids
            values = gathered_metric(
                metric,
                eval_executor.dense_value(agg.read),
                rows,
                tokenizer,
                token_ids=ids,
            )
        else:
            values = compute_metric(
                metric,
                on_host(agg.read),
                rows,
                tokenizer,
                target_value=on_host(agg.target) if agg.target is not None else None,
                vocab_axis=read_is_vocabulary(doc, agg.read.read),
            )
        numeric = [v for v in values if isinstance(v, (int, float))]
        sums.append(sum(numeric))
        counts.append(len(numeric))
    means = eval_executor.rows.agree_means(
        sums, counts, eval_executor.bundle.devices.head
    )
    return dict(zip(names, means, strict=True))


def _eval_reads(fit: _Fit) -> tuple[ReadRef, ...]:
    assert fit.train.eval is not None
    reads: list[ReadRef] = []
    for agg in fit.doc.eval_aggregations():
        reads.append(agg.read)
        if agg.target is not None:
            reads.append(agg.target)
    return tuple(reads)


def _finish(fit: _Fit) -> TrainOutcome:
    selected = "last"
    last_score = fit.last_score
    if fit.best_state is not None:
        _restore(fit.stages, fit.best_state)
        last_score = fit.best_score
        selected = "early_stop.best"
    for stage in fit.stages.values():
        stage.eval()
        if isinstance(stage, Gate):
            # a pinned mask is a phase's, not the bundle's: the saved gate is
            # θ, and an apply reads its own hard split from it (§2.5)
            stage.frozen_mask = None
    eval_score = None
    if fit.train.eval is not None and last_score is not None:
        eval_score = TrainEvalScore(
            split=concrete_str(fit.train.eval["split"], "train.eval.split"),
            metrics=dict(last_score),
            passes=fit.eval_passes,
            featurizers=fit.trained_names,
            selected=selected,
        )
    trained = {name: fit.stages[name] for name in fit.trained_names}
    return TrainOutcome(
        stages=trained,
        eval_score=eval_score,
        diagnostics=fit_diagnostics(trained),
        fit_forwards=(
            {"run": fit.run, "served": fit.served} if fit.store is not None else None
        ),
        resumed=tuple(fit.resumed),
        controls={
            target: {
                "initial": fit.controls[target].initial,
                "final": fit.controls[target].controller.value,
                "signal_final": trace[-1]["signal"] if trace else float("nan"),
                "setpoint_final": trace[-1]["setpoint"] if trace else float("nan"),
                "updates": float(len(trace)),
            }
            for target, trace in fit.control_trace.items()
        },
        control_trace=fit.control_trace,
        constraints={
            name: {
                "target": fit.constraint_specs[name].target,
                # the duals the first update stepped with are the authored
                # init by construction: nothing steps them before it
                "lambda1_initial": fit.constraint_specs[name].init[0],
                "lambda2_initial": fit.constraint_specs[name].init[1],
                "lambda1_final": float(fit.duals[name][0].detach()),
                "lambda2_final": float(fit.duals[name][1].detach()),
                "value_final": trace[-1]["value"] if trace else float("nan"),
                "updates": float(len(trace)),
            }
            for name, trace in fit.constraint_trace.items()
        },
        constraint_trace=fit.constraint_trace,
        draws=fit.drawn.record() if fit.drawn is not None else {},
        anneals={
            target: {
                "start": schedule.start,
                "end": schedule.end,
                "final": schedule.value_at(fit.step, fit.total_steps),
                "shape": schedule.shape,
            }
            for target, schedule in fit.anneals.items()
        },
        phases=tuple(
            {
                "start": phase.start,
                "end": phase.end,
                "params": list(phase.params),
                "freeze_masks": list(phase.freeze_masks),
            }
            for phase in fit.phases
        ),
        checkpoints=tuple(fit.checkpoints),
    )


def _read_signals(fits: Sequence[_Fit]) -> dict[tuple[int, str], float]:
    """Every controller's signal after this update, for every member of the
    step, keyed by ``(id(fit), target)`` — the gates' kept-unit counts
    brought to the host together, one read for the step rather than one per
    gate per controller per member (``stack`` promotes to the widest dtype
    among them, and a float widened is the same number)."""
    pending: list[tuple[int, str, _Control, list[tuple[torch.Tensor, int]]]] = []
    tensors: list[torch.Tensor] = []
    for fit in fits:
        for target, control in fit.controls.items():
            counts = control.signal_counts()
            pending.append((id(fit), target, control, counts))
            tensors.extend(count for count, _ in counts)
    if not tensors:
        return {}
    device = tensors[0].device
    values = iter(torch.stack([t.reshape(()).to(device) for t in tensors]).tolist())
    return {
        (key, target): control.read_signal(
            [float(next(values)) for _ in counts], [units for _, units in counts]
        )
        for key, target, control, counts in pending
    }


def _after_update(fit: _Fit, signals: Mapping[tuple[int, str], float]) -> None:
    """What follows one member's optimizer step (``fit.step`` already counts
    it): every controller observes the fit — its signal read by
    `_read_signals` — and moves its target for the *next* update
    (§2.11), and a scheduled ``trajectory`` checkpoint (§2.12) photographs
    the slots — so a checkpoint's ``weight.<term>`` is the weight the update
    used and its ``control.<target>`` the value set after it."""
    for target, control in fit.controls.items():
        signal = signals[(id(fit), target)]
        setpoint = ramp_setpoint(*control.ramp, fit.step, fit.total_steps)
        value = control.controller.step(signal, setpoint)
        control.apply(value, fit.stages, fit.live_weights)
        fit.control_trace[target].append(
            {
                "step": float(fit.step),
                "value": value,
                "signal": signal,
                "setpoint": setpoint,
            }
        )
    for name, lam in fit.duals.items():
        # the density this update saw and where the ascent left the duals
        # for the next one (the duals it stepped *with* are the previous
        # row's, or the authored init on the first)
        fit.constraint_trace[name].append(
            {
                "step": float(fit.step),
                "value": fit.term_values.get(f"term.{name}", float("nan")),
                "target": fit.constraint_specs[name].target,
                "lambda1": float(lam[0].detach()),
                "lambda2": float(lam[1].detach()),
            }
        )
    if fit.step in fit.checkpoint_steps:
        loss, term_values = fit.loss_record()
        fit.checkpoints.append(
            _checkpoint(
                fit.step,
                fit.epoch,
                {name: fit.stages[name] for name in fit.trained_names},
                loss=loss,
                term_values=term_values,
                controls={
                    target: fit.control_trace[target][-1]["value"]
                    for target in fit.controls
                },
                phase=fit.phase_index if fit.phases else None,
            )
        )


def _checkpoint_steps(
    doc: Document, total_steps: int, batches_per_epoch: int
) -> set[int]:
    """The updates after which a ``trajectory`` entry (§2.12) photographs
    the fit, or the empty set. ``count: n`` spaces n checkpoints equally over
    the run — ``floor(i · total / n)`` for ``i = 1..n``, so the last is the
    final update and a run shorter than ``n`` updates yields as many distinct
    checkpoints as it has updates; ``updates: n`` / ``epochs: n`` photograph
    every n of them, and the final update always."""
    entry = next((e for e in doc.save if e.kind == "trajectory"), None)
    if entry is None or entry.every is None or total_steps <= 0:
        return set()
    ((unit, n),) = entry.every.items()
    if unit == "count":
        return {max(1, (i * total_steps) // n) for i in range(1, n + 1)}
    stride = n if unit == "updates" else n * batches_per_epoch
    return {step for step in range(stride, total_steps + 1, stride)} | {total_steps}


def _checkpoint(
    step: int,
    epoch: int,
    stages: Mapping[str, Stage],
    *,
    loss: float,
    term_values: Mapping[str, float],
    controls: Mapping[str, float],
    phase: int | None = None,
) -> Checkpoint:
    """One photograph of the fit after ``step`` updates: every trained slot,
    detached to the CPU, and the record a reader wants beside it — the loss
    and its terms, each gate's mask numbers, each controlled value."""
    slots = {
        name: {
            slot: param.detach().to("cpu").clone()
            for slot, param in stage.slot_params().items()
        }
        for name, stage in stages.items()
    }
    record: dict[str, Any] = {"step": step, "epoch": epoch, "loss": loss, **term_values}
    for name, values in fit_diagnostics(stages).items():
        for key in ("hard_mask_size", "decisive_fraction", "k"):
            if key in values:
                record[f"{name}.{key}"] = values[key]
    for target, value in controls.items():
        record[f"control.{target}"] = value
    if phase is not None:
        record["phase"] = phase  # §2.11 `phases`: the window this update ran in
    return Checkpoint(step=step, epoch=epoch, slots=slots, record=record)


@dataclasses.dataclass
class _Control:
    """One §2.11 ``control`` entry, bound to the loop: the controller, the
    gate or gates whose kept-unit count it observes (a list of gates is one
    signal — their counts summed — as a list-valued ``l1`` is one penalty),
    and how the controlled value is written back — into a named term's live
    weight, or onto a stage attribute the way an anneal writes one."""

    controller: PidController
    ramp: tuple[float, float, float]
    initial: float
    signal_stages: Sequence[Stage]
    term: str | None = None
    hyper: tuple[str, str] | None = None  # (featurizer, attribute)
    signal: str = "hard_mask_size"

    def signal_counts(self) -> list[tuple[torch.Tensor, int]]:
        """Per observed gate, its kept-unit count through the hard mask as
        the device scalar it is, and its unit count — what
        `_read_signals` brings to the host for every controller of a
        step in one read."""
        counts: list[tuple[torch.Tensor, int]] = []
        for stage in self.signal_stages:
            assert isinstance(stage, Gate)
            hard = stage.hard_mask()
            counts.append((hard.sum(), int(hard.numel())))
        return counts

    def read_signal(self, kept_counts: Sequence[float], units: Sequence[int]) -> float:
        # kept-unit counts through the hard mask, summed over the named gates
        # (`hard_mask_size`), or that sum over the gates' total unit count
        # (`hard_mask_fraction`) — CONTROL_SIGNALS; ``kept_counts`` are the
        # gates' counts as floats, in `signal_counts` order
        kept = 0.0
        for count in kept_counts:
            kept += count
        total = sum(units)
        if self.signal == "hard_mask_fraction":
            return kept / total if total else 0.0
        return kept

    def apply(
        self, value: float, stages: Mapping[str, Stage], live_weights: dict[str, float]
    ) -> None:
        if self.term is not None:
            live_weights[self.term] = value
        else:
            assert self.hyper is not None
            setattr(stages[self.hyper[0]], self.hyper[1], value)


def _build_controls(
    control: Mapping[str, Mapping[str, Any]] | None,
    stages: Mapping[str, Stage],
    live_weights: Mapping[str, float],
) -> dict[str, _Control]:
    """The loop's controllers, one per ``train.control`` target (§2.11).
    Validation has already resolved every target and signal; what is decided
    here is the binding — which live weight or which attribute the value
    lands on — and the initial value it starts from."""
    if not control:
        return {}
    out: dict[str, _Control] = {}
    for target, spec in control.items():
        ((signal, signal_target),) = spec["signal"].items()
        names = (
            [signal_target] if isinstance(signal_target, str) else list(signal_target)
        )
        signal_stages = [stages[str(name)] for name in names]
        ramp = tuple(float(v) for v in spec["setpoint"]["ramp"])
        if target.startswith(OBJECTIVE_WEIGHT_PREFIX):
            name = target[len(OBJECTIVE_WEIGHT_PREFIX) :].rpartition(".")[0]
            initial = live_weights[name]
            out[target] = _Control(
                controller=build_controller(spec, initial=initial),
                ramp=ramp,  # type: ignore[arg-type]
                initial=initial,
                signal_stages=signal_stages,
                term=name,
                signal=str(signal),
            )
        else:
            fname, _, tail = target.partition(".")
            hyper = tail.rsplit(".", 1)[-1]
            stage = stages[fname]
            if not hasattr(stage, hyper):
                raise ProtocolError("P2", f"{fname!r} has no controllable {hyper!r}")
            initial = float(getattr(stage, hyper))
            out[target] = _Control(
                controller=build_controller(spec, initial=initial),
                ramp=ramp,  # type: ignore[arg-type]
                initial=initial,
                signal_stages=signal_stages,
                hyper=(fname, hyper),
                signal=str(signal),
            )
    return out


def fit_diagnostics(stages: Mapping[str, Stage]) -> dict[str, dict[str, Any]]:
    """What each fit can say about *itself*, saved beside the bundle.

    The case this exists for: the shipped DBM preset
    (``demos/methods/protocols/dbm.json``) records, in its committed run
    ``demos/methods/results/protocols/dbm.json``, a ``decisive_fraction`` of
    0.0: **no** dimension outside [0.1, 0.9]. A fit like that can still score
    **1.000** at a readout layer, as the preset's description states. The
    mechanism is that `Gate._mask` returns a *hard* ``θ > 0`` mask in eval
    mode, and ``run_training`` puts the stages in eval mode before returning —
    so the 1.000 is the hard mask, and with θ never separated ``θ > 0`` is a
    coin flip on gradient noise. Roughly half the dimensions swap, which at the
    readout layer scores 1.000.

    So the preset produced a **meaningless mask and a perfect number**, and
    nothing in the run's saved outputs said so. These two numbers say so:

    ``decisive_fraction``
        The fraction of dimensions where σ(θ) is outside
        [0.5 − [`MASK_DECISIVE_MARGIN`][], 0.5 + …]. Near 0 means the gate
        never committed and the score below it describes noise, whatever it
        says.
    ``hard_mask_size``
        How many dimensions ``θ > 0`` keeps — the mask the eval-mode score was
        actually computed through, which is the number a localization claim is
        about.
    ``frozen_units`` / ``reawakened_units`` (with ``dead`` when authored)
        The dead-unit bookkeeping (§2.5 ``dead``): how many units a
        ``freeze_after`` rule froze, and how many units were hard-off after
        some step yet are kept by the final hard mask — the count a ``leak``
        exists to raise, and exactly 0 on a frozen gate. Recorded for every
        gate, rule or none, so the two fits are comparable.

    Under a ``clamp`` gate (§2.5 ``parametrization``) the soft mask is ``θ``
    itself and the hard split ``θ > ½``, so both numbers are read through
    [`Gate.soft_mask`][causalab.neural.shared.featurizers.gate.Gate.soft_mask] / [`Gate.hard_mask`][causalab.neural.shared.featurizers.gate.Gate.hard_mask] rather than spelled here;
    ``parametrization`` is recorded so the record says which split it counted.
    Under ``hard_concrete`` the soft mask is the *deterministic* stretched and
    clipped ``σ(θ)`` — the mean of the sampled training mask — and ``stretch``
    is recorded beside it because the hard split depends on it. Two
    consequences for ``decisive_fraction`` there: the stretch scales the
    distance from ½ by ``ζ − γ`` (1.2 at the default), so a unit is decisive
    at a smaller ``|θ|`` than a ``sigmoid`` gate's and *saturates* at exactly
    0 or 1 once ``|θ| ≳ logit((1 − γ)/(ζ − γ))``; and the temperature does not
    enter the deterministic mask, so an anneal of β leaves the number alone
    where a ``sigmoid`` anneal of ``T → 0`` drives it to 1. The number is the
    right one for the gate it describes, but it is not comparable across maps
    at one [`MASK_DECISIVE_MARGIN`][].

    Both count the gate's *units* — one ``theta`` entry each. On a grouped gate
    (§2.5 ``group``) a unit is a head, or one ``(expert, neuron)`` of the
    expert table, so ``hard_mask_size`` is a head or expert-neuron count and
    ``groups`` records how many units there were; ``width`` stays the site's
    coordinate width. On a position gate (§2.5 ``axis``) a unit is an
    addressed token position, so ``width`` is the window's length rather
    than the site's coordinate width — which is recorded nowhere — and
    ``axis`` is recorded to say so.

    This is the half of the DBM finding that needs no GPU. Retuning `l1` and
    the anneal so θ *does* separate needs a run to validate and is deliberately
    not asserted here.

    A fitted ``subspace`` reports ``orthonormality_deviation`` — ``max|QᵀQ − I|``
    of the rotation the bundle saves — and ``within_tolerance``, its verdict
    against [`ORTHONORMAL_TOLERANCE`][causalab.neural.shared.featurizers.stages.ORTHONORMAL_TOLERANCE]
    (``1.0``/``0.0``). The verdict is recorded beside the value because the
    value is raw roundoff, whose last digits are accumulation order and differ
    across BLAS and device, while the verdict is what a reader acts on: a
    ``0.0`` is a rotation no later document can name as an ``init``. The
    ``cayley`` map's fp32 error is ~1e-6 where fits live but grows quadratically
    in ``‖X‖`` once ``X⊥`` goes rank-deficient
    ([`Cayley`][causalab.neural.shared.featurizers.stages.Cayley], *Conditioning*), and
    nothing else checks orthonormality at save time — the next check is
    ``_init_basis`` refusing the rotation as a start in a later run, a
    diagnostic that would otherwise arrive one run away from its cause.
    """
    out: dict[str, dict[str, Any]] = {}
    for name, stage in stages.items():
        if isinstance(stage, Subspace):
            with torch.no_grad():
                deviation = orthonormality_deviation(stage.weight)
            out[name] = {
                "k": float(stage.k),
                "orthonormality_deviation": deviation,
                "within_tolerance": float(deviation <= ORTHONORMAL_TOLERANCE),
            }
            continue
        theta = getattr(stage, "theta", None)
        if not isinstance(stage, Gate) or theta is None:
            continue
        with torch.no_grad():
            soft = stage.soft_mask().detach().float()
            decisive = (soft - 0.5).abs() > MASK_DECISIVE_MARGIN
            # a budget's last drawn k: the gate's own, or its pool's
            budget_k = stage._k if stage.pool is None else stage.pool.k
            out[name] = {
                "width": float(stage.width),
                **({"groups": float(theta.numel())} if stage.groups else {}),
                # §2.5 `axis`: `width` is then the window length and every count
                # above is over positions — the record says so, as `groups` does
                **({"axis": stage.axis} if stage.axis is not None else {}),
                "decisive_fraction": float(decisive.float().mean()),
                "hard_mask_size": float(stage.hard_mask().sum()),
                "temperature": float(stage.temperature),
                "parametrization": stage.parametrization,
                # §2.5 `boundary`: the learned β itself, beside the rank ⌈β⌉ it
                # keeps (`hard_mask_size`), so a report can put the fit on the
                # `k` sweep it replaces
                **(
                    {"boundary": stage.boundary()}
                    if stage.parametrization == "boundary"
                    else {}
                ),
                # §2.5 the mapping form: with it, `decisive_fraction` reports the
                # backward map's confidence (the forward is 0/1 by construction)
                # and `hard_mask_size` is the eval split, which the forward's
                # count above ½ equals under sigmoid and clamp only — so the
                # record says which regime its numbers were taken in
                **(
                    {"forward": stage.forward_mask}
                    if stage.forward_mask is not None
                    else {}
                ),
                **(
                    {"stretch": list(stage.stretch)}
                    if stage.stretch is not None
                    else {}
                ),
                **(
                    {"init_fill": stage.init_fill}
                    if stage.init_fill is not None
                    else {}
                ),
                **(
                    {"init_from_scores": stage.init_scores}
                    if stage.init_scores is not None
                    else {}
                ),
                # §2.5 `dead`: the rule as authored, `frozen_units`, and
                # `reawakened_units` — the latter under every rule and none,
                # since it is the observable a leak exists to move
                **stage.dead_diagnostics(),
                # a loaded gate read out at a count (§2.5 `top_k`): the split
                # the numbers above were counted through is a cut, not the
                # map's threshold, and the record says so
                **({"top_k": float(stage.top_k)} if stage.top_k is not None else {}),
                # a budget gate (§2.5): the schedule, the cut `hard_mask_size`
                # is read at, and the budget of the last step — the number a
                # trajectory checkpoint's `k` column carries
                **(
                    {
                        "k_schedule": dict(stage.k_schedule),  # carries `of`
                        "eval_k": float(stage.eval_k()),
                        **({"k": float(budget_k)} if budget_k is not None else {}),
                        "stop_grad_shift": float(stage.stop_grad_shift),
                        # a pooled gate (§2.5 `pool`): the pool, its unit
                        # count, and how many of THIS gate's units the pooled
                        # cut keeps — `hard_mask_size` above is that number,
                        # `eval_k` the pool's
                        **(
                            {
                                "pool": stage.pool.name,
                                "pool_units": float(stage.pool.units),
                            }
                            if stage.pool is not None
                            else {}
                        ),
                    }
                    if stage.parametrization == "budget"
                    and stage.k_schedule is not None
                    else {}
                ),
            }
    return out


def _snapshot(stages: Mapping[str, Stage]) -> dict[str, dict[str, torch.Tensor]]:
    """Detached copies of every trained stage's parameters.

    ``state_dict`` rather than ``slot_params`` on purpose: a ``subspace``
    stage's ``weight`` is *computed* by an orthogonal parametrization, so the
    tensor the optimizer actually steps is
    ``parametrizations.weight.original``. Restoring the materialized weight
    would restore nothing.
    """
    return {
        name: {key: value.detach().clone() for key, value in stage.state_dict().items()}
        for name, stage in stages.items()
    }


def _restore(
    stages: Mapping[str, Stage], snapshot: Mapping[str, Mapping[str, torch.Tensor]]
) -> None:
    """Put the snapshotted parameters back, in place."""
    for name, stage in stages.items():
        state = snapshot.get(name)
        if state is not None:
            stage.load_state_dict(dict(state))


def lr_factor(step: int, total_steps: int, warmup_frac: float) -> float:
    """HF's ``get_linear_schedule_with_warmup`` at update ``step`` (0-based):
    ``step / warmup`` while warming up, then ``(total − step) / (total − warmup)``
    down to 0 at the last update. ``warmup = warmup_frac · total`` (a float, as
    HF takes it). The first update runs at lr 0 when there is any warm-up, as
    in the reference implementation."""
    warmup = warmup_frac * total_steps
    if step < warmup:
        return step / max(1.0, warmup)
    return max(0.0, (total_steps - step) / max(1.0, total_steps - warmup))


#: The lr a param group was built with, kept beside the group so the schedule
#: multiplies the authored value and never its own previous output.
_BASE_LR = "_schedule_base_lr"


def _apply_lr_schedule(fit: "_Fit") -> None:
    """Set every group's lr for this update under §2.11 ``optimizer.schedule``:
    a no-op under ``constant``. Per-entry lrs are each scaled by the same factor.
    Refused beside ``phases`` at load (rule 4), so no other writer of ``lr``
    runs in the same fit."""
    spec = fit.doc.train.optimizer if fit.doc.train is not None else {}
    schedule = str(spec.get("schedule", "constant"))
    if schedule == "constant":
        return
    warmup_frac = float(spec.get("warmup_frac", 0.1))
    factor = lr_factor(fit.step, fit.total_steps, warmup_frac)
    for group in fit.optimizer.param_groups:
        if _DUAL_GROUP in group:
            continue  # a constraint's duals ascend at their own authored rate
        if _BASE_LR not in group:
            group[_BASE_LR] = group["lr"]
        group["lr"] = group[_BASE_LR] * factor


#: The key marking a constraint term's dual-pair parameter group (§2.11), by
#: the term's name — what the lr schedule and the phase machinery skip.
_DUAL_GROUP = "_constraint_duals_of"


def _add_dual_groups(
    train: TrainSpec, optimizer: torch.optim.Optimizer, executor: PointExecutor
) -> dict[str, torch.nn.Parameter]:
    """§2.11 ``constraint``: one ``(λ₁, λ₂)`` parameter per constraint term,
    appended to the fit's optimizer as its own group — ``maximize`` so the
    step is an ascent on ``λ₁(s − t) + λ₂(s − t)²``, the authored ``dual.lr``,
    and none of what the group would otherwise inherit from the constructor:
    no weight decay and no momentum, so under ``sgd`` the step is exactly
    ``lr · gradient`` whatever the document's ``momentum`` (under ``adamw``
    it is Adam's, as §2.11 says). The duals share the optimizer's
    arithmetic and nothing else: not its schedule, not its phases.

    A run capturing fit graphs never reaches here with a constraint —
    ``cuda_graphs.unsupported_reason`` sends such a point down the eager
    path, as it does for ``control`` and ``phases`` — so the refusal below
    is a backstop: the cohort and single-fit graphs map optimizer parameters
    onto worker stages by identity, and a dual is no stage's parameter."""
    constrained = [term for term in train.objective if term.constraint is not None]
    if constrained and executor.cuda_graphs and executor.fit_cuda_graphs:
        raise ProtocolError(
            "P4",
            f"train.objective.{constrained[0].name}.constraint: the Lagrangian "
            "duals step on the eager loop — this run captures fit graphs "
            "(fit_cuda_graphs); run it eager",
        )
    duals: dict[str, torch.nn.Parameter] = {}
    for term in constrained:
        assert term.constraint is not None and term.name is not None
        lam = torch.nn.Parameter(
            torch.tensor(
                list(term.constraint.init),
                dtype=torch.float32,
                device=executor.bundle.devices.head,
            )
        )
        optimizer.add_param_group(
            {
                "params": [lam],
                "lr": float(term.constraint.dual_lr),
                "weight_decay": 0.0,
                "momentum": 0.0,
                "maximize": True,
                _DUAL_GROUP: term.name,
            }
        )
        duals[term.name] = lam
    return duals


def _optimizer_default(spec: Mapping[str, Any], field: str, fallback: float) -> float:
    """The constructor-level value of ``lr`` / ``weight_decay``: the scalar the
    document gave, or — when the field is a per-parameter mapping (§2.11) —
    the largest of its values. Torch needs one default even when every
    parameter group overrides it; every group *does* override it here
    (`_prepare_fit` writes the field on each group), so the default is
    never the value any tensor is stepped with."""
    value = spec.get(field, fallback)
    if isinstance(value, Mapping):
        return max(float(v) for v in value.values())
    return float(value)


def _build_optimizer(
    spec: Mapping[str, Any], groups: Sequence[Mapping[str, Any]]
) -> torch.optim.Optimizer:
    """The optimizer over ``groups`` — one ``{"params": [...], lr?,
    weight_decay?}`` per ``train.params`` entry; a group's own ``lr`` /
    ``weight_decay`` override the constructor's, which is how a rotation and a
    gate step at different rates inside one fit."""
    name = str(spec["name"])
    lr = _optimizer_default(spec, "lr", 0.0)
    weight_decay = _optimizer_default(spec, "weight_decay", 0.0)
    params = [dict(g) for g in groups]
    if name in ("adamw", "adam"):
        raw_betas = spec.get("betas", (0.9, 0.999))
        betas = (float(raw_betas[0]), float(raw_betas[1]))
        eps = float(spec.get("eps", 1e-8))
        cls = torch.optim.AdamW if name == "adamw" else torch.optim.Adam
        return cls(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(
            params,
            lr=lr,
            momentum=float(spec.get("momentum", 0.0)),
            weight_decay=weight_decay,
        )
    raise ProtocolError("P4", f"unknown optimizer {name!r}")


@dataclasses.dataclass(frozen=True)
class _Phase:
    """One §2.11 ``phases`` entry in update terms: the half-open window
    ``[start, end)``, the ``train.params`` entries that step in it, the
    per-entry ``lr`` / ``weight_decay`` it overrides (absent = the top-level
    group's), its own schedules, and the gates whose hard mask it pins."""

    start: int
    end: int
    params: frozenset[str]
    optimizer: Mapping[str, Mapping[str, float]]
    anneals: dict[str, AnnealSchedule]
    freeze_masks: tuple[str, ...]


def _build_phases(
    phases: Sequence[PhaseSpec] | None,
    total_steps: int,
    stages: Mapping[str, Stage],
    live_weights: Mapping[str, float],
) -> tuple[_Phase, ...]:
    """Resolve ``train.phases`` against this fit's update count (§2.11). A
    ``frac`` end is ``round(frac · total_steps)``, so consecutive phases meet
    without a gap; an ``updates`` end is taken as written and the last one
    must be the run's length — the parser could not know it, so a document
    whose phases do not partition the run is refused here, before any step.
    A phase that resolves to zero updates (a frac too small for the run) is
    refused rather than silently skipped: its params would never train."""
    if not phases:
        return ()
    out: list[_Phase] = []
    start = 0
    for i, phase in enumerate(phases):
        ((unit, value),) = phase.until.items()
        end = int(round(float(value) * total_steps)) if unit == "frac" else int(value)
        if i == len(phases) - 1 and unit == "frac":
            end = total_steps  # rounding never leaves the tail unowned
        if end <= start:
            raise ProtocolError(
                "P2",
                f"train.phases[{i}] spans no update: it would end at {end} of "
                f"{total_steps} after the previous phase ended at {start}",
            )
        if end > total_steps:
            raise ProtocolError(
                "P2",
                f"train.phases[{i}] ends at update {end}, past the run's {total_steps}",
            )
        if i == len(phases) - 1 and end != total_steps:
            raise ProtocolError(
                "P2",
                f"train.phases partition the run: the last phase ends at update "
                f"{end}, the run at {total_steps}",
            )
        optimizer: dict[str, dict[str, float]] = {}
        for field, setting in (phase.optimizer or {}).items():
            if isinstance(setting, Mapping):
                optimizer[field] = {k: float(v) for k, v in setting.items()}
            else:
                optimizer[field] = {p: float(setting) for p in phase.params}
        out.append(
            _Phase(
                start=start,
                end=end,
                params=frozenset(phase.params),
                optimizer=optimizer,
                anneals=_parse_anneals(phase.anneal, stages, live_weights),
                freeze_masks=tuple(phase.freeze_masks),
            )
        )
        start = end
    return tuple(out)


def _advance_phase(fit: _Fit) -> None:
    """Before an update: enter the phase that owns ``fit.step``, if the fit is
    not in it yet. Entering sets, per ``train.params`` entry, whether its
    tensors take gradients and its optimizer group's ``lr`` /
    ``weight_decay`` — the phase's override, the top-level value, or ``0``
    for an entry the phase leaves out (the group stays, so Adam's moments
    survive the boundary and a later phase resumes rather than restarts) —
    and pins each ``freeze_masks`` gate's hard mask as of this step, clearing
    any pin a previous phase set on a gate this one does not name."""
    if not fit.phases:
        return
    index = fit.phase_index
    while index + 1 < len(fit.phases) and fit.step >= fit.phases[index + 1].start:
        index += 1
    if index == fit.phase_index:
        return
    fit.phase_index = index
    phase = fit.phases[index]
    for entry, gi in fit.groups_by_entry.items():
        group = fit.optimizer.param_groups[gi]
        active = entry in phase.params
        for param in group["params"]:
            param.requires_grad_(active)
        for field in PER_PARAMS_OPTIMIZER_FIELDS:
            base_key = f"_phase_base_{field}"
            if base_key not in group:
                group[base_key] = group.get(field, 0.0)  # the top-level value
            if not active:
                group[field] = 0.0
            else:
                group[field] = phase.optimizer.get(field, {}).get(
                    entry, group[base_key]
                )
    pinned = set(phase.freeze_masks)
    for name, stage in fit.stages.items():
        if not isinstance(stage, Gate):
            continue
        if name in pinned:
            with torch.no_grad():
                stage.frozen_mask = stage.hard_mask().detach().clone()
        else:
            stage.frozen_mask = None


def _parse_anneals(
    anneal: Mapping[str, AnnealSchedule] | None,
    stages: Mapping[str, Stage],
    live_weights: Mapping[str, float],
) -> dict[str, AnnealSchedule]:
    """The loop's open-loop schedules (§2.11), each bound to what it moves: a
    trained featurizer's hyperparameter, or a named objective term's live
    weight — the same dict a controller writes and `_loss` reads, so an
    annealed weight and a controlled one move through one path. Validation
    resolved both target forms at load; this re-checks for a document that
    arrived unvalidated."""
    if anneal is None:
        return {}
    out: dict[str, AnnealSchedule] = {}
    for dotted, schedule in anneal.items():
        if dotted.startswith(OBJECTIVE_WEIGHT_PREFIX):
            name = dotted[len(OBJECTIVE_WEIGHT_PREFIX) :].rpartition(".")[0]
            if name not in live_weights:
                raise ProtocolError(
                    "P2",
                    f"anneal target {dotted!r} is not a named objective term's weight",
                )
        elif dotted.split(".", 1)[0] not in stages:
            raise ProtocolError("P2", f"anneal target {dotted!r} is not being trained")
        if not isinstance(schedule, AnnealSchedule):
            raise ProtocolError(
                "P2", f"anneal schedule for {dotted!r} is [start, end, frac]"
            )
        out[dotted] = schedule
    return out


def _set_anneal(
    fit: _Fit,
    dotted: str,
    schedule: AnnealSchedule,
    *,
    step: int | None = None,
    total_steps: int | None = None,
) -> None:
    """Write this update's scheduled value where ``dotted`` points: into the
    named term's live weight, or onto the stage attribute the path names.
    ``step`` / ``total_steps`` default to the run's; a phase passes its own
    window so its schedule spans the phase (§2.11)."""
    value = schedule.value_at(
        fit.step if step is None else step,
        fit.total_steps if total_steps is None else total_steps,
    )
    if dotted.startswith(OBJECTIVE_WEIGHT_PREFIX):
        # `_loss` reads the live weight of a named term (§2.11), so the
        # schedule lands there and the checkpoint's `weight.<name>` is the
        # value the update used
        name = dotted[len(OBJECTIVE_WEIGHT_PREFIX) :].rpartition(".")[0]
        fit.live_weights[name] = value
        return
    stages = fit.stages
    fname, _, tail = dotted.partition(".")
    hyper = tail.rsplit(".", 1)[-1]
    stage = stages[fname]
    if not hasattr(stage, hyper):
        raise ProtocolError("P2", f"{fname!r} has no annealable {hyper!r}")
    if (
        isinstance(stage, Gate)
        and stage.parametrization in ("clamp", "budget")
        and hyper == "temperature"
    ):
        # validation refuses this at load (rule 4); a document that reached
        # the loop unvalidated must not have its schedule silently ignored
        raise ProtocolError(
            "P2",
            f"{fname!r} is a {stage.parametrization} gate: its mask has no "
            "temperature to anneal — θ itself under clamp, σ(θ + c_k) with the "
            "shift solved per step under budget (§2.5)",
        )
    setattr(stage, hyper, value)


def _eval_executor(
    doc: Document,
    executor: PointExecutor,
    run: RunContext,
    split: str,
    *,
    pool: GraphPool | None = None,
) -> PointExecutor:
    """The executor every eval pass of this fit runs on — built on the first
    pass and kept on the point executor (``eval_executor``) for the rest.

    One per fit, not one per pass: the split's rows are read and tokenized
    once, and its captures live under ``(key, split)`` in the shared store
    — the eval split is not the campaign's rows, so it gets its own key —
    with only the fit-constant groups eligible (``inner_interning``). The
    trained model's group is re-run on every pass, as it must be.
    The row bound (``batch_rows``) carries over from the point executor: an
    eval pass is a no-grad forward over the whole split, exactly what
    microbatching is for.

    Built lazily, on the first pass, rather than before the epoch loop: an
    ``updates`` budget shorter than one epoch never evaluates, and the eval
    split is only guaranteed readable through the eval path."""
    if executor.eval_executor is not None:
        return executor.eval_executor
    cache = executor.graph_cache

    def build() -> PointExecutor:
        # under a rows split (§8.3) this replica scores its slice of the
        # eval split — every row at world 1 — and the score is agreed
        all_rows = run.env.datasets.rows(split)
        split_rows = [
            all_rows[index] for index in executor.rows.slice_for(range(len(all_rows)))
        ]
        # Retained captures must not mutate a prior fit's saved stages.
        stages = (
            copy.deepcopy(executor.stage_cache)
            if cache is not None
            else executor.stage_cache
        )
        return _derived(
            executor,
            make_executor(
                doc,
                executor.bundle,
                cuda_graphs=executor.cuda_graphs and executor.fit_cuda_graphs,
                role_rows={role: split_rows for role in executor.role_rows},
                role_fields=executor.role_fields,
                load_tensors=executor.load_tensors,
                load_table=executor.load_table,
                stage_cache=stages,
                grad_enabled=False,
                coords=executor.coords,
                batch_rows=executor.batch_rows,
                interning=_inner_interning(executor, split),
            ),
        )

    built = cache.evaluation(split, build) if cache is not None else build()
    # its reads stay on the device when a metric selects from them there:
    # `_score` gathers the answer columns and copies those, not the
    # vocabulary (ExecutorBase.device_reads). A softmax-class eval keeps the
    # one host copy `_finalize_read` makes
    built.device_reads = any(
        str(agg.spec.kind) in GATHERED_KINDS for agg in doc.eval_aggregations()
    )
    if isinstance(built, GraphExecutor):
        # its inference replays join the fit's pool (GraphPool); the point
        # executor itself never captures during a fit and carries no pool
        built.graph_pool = pool
    executor.eval_executor = built
    return built
