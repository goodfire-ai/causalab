"""Resolve reads and writes in the shared point executor.

``ExecutorBase`` works with ``(batch, position, feature)`` tensors. It
resolves positions, gathers values, builds featurizers, and runs checks
before a forward. ``InterningMixin`` supplies cache access and
``WriteMathMixin`` applies writes through the ragged landing helpers.

An engine implements ``_run_group`` to capture tensors and land writes.
``refuse_unstackable`` checks continuation reads whose steps cannot stack.
"""

from __future__ import annotations

import contextlib
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch

from causalab.causal.pair_validation import (
    EDIT_GROUPS_COLUMN,
    EditGroup,
    EditGroupError,
    check_component_wise,
    parse_edit_groups,
    row_texts,
)
from causalab.causal.scoring import (
    ScoringCheck,
    ScoringError,
    ScoringMismatch,
    check_scoring,
    declared_modes,
)
from causalab.neural.shared.encoding import (
    Continuation,
    EncodedBatch,
    select_field,
)
from causalab.neural.shared.executor.cache import Interning, InterningMixin
from causalab.neural.shared.executor.ragged import (
    RowWindow,
    ragged_write_error,
    ragged_geometry_of,
)
from causalab.neural.shared.executor.writes import (
    WriteMathMixin,
    whole_native_tensor,
)
from causalab.neural.shared.featurizers import (
    FeaturizerStack,
    Stage,
    build_stack,
    link_budget_pools,
)
from causalab.neural.shared.gather import dense_index, flat_index, gather_positions
from causalab.neural.shared.head import ReadTap, resolve_read_taps
from causalab.neural.shared.parallel.taps import IDENTITY_TAP, TapFragments
from causalab.neural.shared.metrics import DEVICE_SCORED_KINDS
from causalab.neural.shared.plan import (
    bound_read,
    fit_constant_models,
    saved_raw_reads,
    write_names,
)
from causalab.neural.shared.sites import ResolvedSite, resolve_site
from causalab.neural.shared.values import RaggedValue
from causalab.protocol.identity import single_site_featurizers
from causalab.protocol.lowering import lower_bands
from causalab.protocol.positions.alignment import unalignable
from causalab.protocol.positions.encoding import PositionFrame, generated_budget
from causalab.protocol.positions.ledger import LocationLedger
from causalab.protocol.positions.resolve import (
    StepPositions,
    build_ledger,
    encode_role,
    spec_of,
)
from causalab.protocol.registry import component_width
from causalab.protocol.results import (
    Resolution,
    Unavailable,
    available,
    cell_key,
    unavailable,
)
from causalab.protocol.rules.errors import ProtocolError, ValidationError
from causalab.protocol.schema import (
    BoundAggregation,
    Document,
    PositionSpec,
    ReadRef,
    ReadSpec,
    SpanSpec,
    WriteSpec,
    operand_reads,
    read_is_vocabulary,
    span_length,
)

#: What every read-keyed table of an executor is keyed by (§2.7): a read
#: **bound to the model it is taken on**. The read name alone is not a key —
#: one read listed by two models is two values (``plan.bound_read``,
#: ``plan.read_label``).
BoundRead = ReadRef


def device_scored_reads(doc: Document) -> frozenset[BoundRead]:
    """The prompt-frame reads whose **only** consumers are metrics scored on
    the device (``metrics.DEVICE_SCORED_KINDS``) — a scan's ``lm_head`` read
    under an ``iia`` or ``logit_diff``.

    Such a read's finalized value may stay where the forward left it: the
    metric loop (``execution._execute_point`` → ``metrics.score_metric``)
    gathers the answer columns or the argmax indices there and copies those,
    never the vocabulary. Every other consumer takes the host copy
    `ExecutorBase._finalize_read` has always made, so a read is left out
    when anything else reads it: a ``save`` entry writes it to a tensor file,
    a write names it as an operand, a ``kl``/``js`` compares against it as
    ``target``, or it addresses the continuation frame (whose metrics are
    windowed and reduced per step)."""
    consumers: dict[BoundRead, set[str]] = {}
    for agg in doc.aggregations():
        consumers.setdefault(agg.read, set()).add(str(agg.spec.kind))
        if agg.target is not None:
            consumers.setdefault(agg.target, set()).add(str(agg.spec.kind))
    saved = saved_raw_reads(doc)
    operands = {
        ref for write in doc.writes.values() for ref in operand_reads(doc, write.do)
    }
    return frozenset(
        ref
        for ref, kinds in consumers.items()
        if ref.model is not None
        and ref.read in doc.reads
        and kinds <= DEVICE_SCORED_KINDS
        and ref not in saved
        and ref not in operands
        and generated_budget(doc, doc.reads[ref.read].pos) is None
    )


def document_seed(doc: Document) -> int:
    """The one seed a document implies: ``train.seed``, or **0** when it
    declares no fit.

    Read in one place so the three consumers cannot drift apart: the
    ``subspace`` featurizer's initial rotation ([`build_stack`][causalab.neural.shared.featurizers.build.build_stack]),
    ``torch.manual_seed`` at train-loop entry, and the batch-order RNG.

    The 0 for a document with no ``train`` block is deliberate rather than
    accidental: an apply/inference document has no seed to name, and pinning
    it means the same document builds the same (unfitted) featurizer whether
    or not a fit is running — which a global-RNG init could not promise."""
    train = doc.train
    if train is None:
        return 0
    return int(train.seed) if isinstance(train.seed, int) else 0


def _derive(
    site: ResolvedSite,
    value: torch.Tensor,
    rname: str,
    tap: TapFragments = IDENTITY_TAP,
) -> torch.Tensor:
    """Compute a derived component from the tensor its tap captured."""
    if site.derivation == "attention_result":
        return _attention_result(site, value, tap)
    raise ProtocolError("P2", f"read {rname!r}: unknown derivation {site.derivation!r}")


def _attention_result(
    site: ResolvedSite, premix: torch.Tensor, tap: TapFragments = IDENTITY_TAP
) -> torch.Tensor:
    """Head ``h``'s contribution to the residual stream.

    ``result[..., h, :] = premix[..., h·d:(h+1)·d] @ W_o[:, h·d:(h+1)·d].T`` —
    the part of the block's attention output that head ``h`` is responsible for.
    The model never forms it: it projects the whole premix at once, so what it
    computes is the *sum* over heads (plus the o-projection's bias, if it has
    one). ``sum_h result == attention_output - bias`` is the identity that
    defines this component, and the test suite pins it.

    Computed by **masking and re-projecting** rather than by slicing the weight
    matrix. That is deliberate: ``nn.Linear`` stores ``(out, in)`` and
    transformers' ``Conv1D`` (GPT-2's ``c_proj``) stores ``(in, out)``, so a
    weight-slicing implementation has to know which family it is looking at and
    is silently wrong if it guesses. Running the projection the model's own
    module defines cannot be wrong about its own layout, and the bias — which is
    *not* attributable to any head — is subtracted back off explicitly.

    ⚠️ Calls ``site.module`` directly, so it needs a real ``nn.Module``. That is
    why the nnsight engine, whose ``site.module`` is an envoy, does not declare
    this component.

    Under tensor parallelism (``docs/model_parallelism.md`` §6.2) ``premix``
    is the gathered, global o-projection input and the module is the
    ``rowwise`` projection expecting this rank's chunk of it: the head mask
    is applied in the global head order, ``tap.fragment`` hands the module its
    chunk, and the module's own all-reduce completes the product — a head
    another rank holds contributes exactly zero from this one. The bias is
    added once, after the all-reduce, so subtracting it here is unchanged.
    """
    module = site.module
    bias = _local_tensor(getattr(module, "bias", None))
    heads = site.shape.head_space
    assert heads is not None  # the premix tap always has a head axis
    per_head = premix.shape[-1] // heads

    def contribution(head: int) -> torch.Tensor:
        masked = torch.zeros_like(premix)
        window = slice(head * per_head, (head + 1) * per_head)
        masked[..., window] = premix[..., window]
        out = module(tap.fragment(masked))
        return out if bias is None else out - bias

    if site.head is not None:
        return contribution(site.head)
    # The whole tensor: `heads` times wider than `attention_output`. On a
    # model with many heads that is many times the residual width, which is
    # why naming a `head` is encouraged — but it is a documented cost, not a refusal.
    return torch.cat([contribution(head) for head in range(heads)], dim=-1)


class _LazyFrames(dict):  # type: ignore[type-arg]
    """The executor's frames as a mapping, each role encoded on first
    request through `ExecutorBase._batch` — so a step resolved
    engine-side pays for a role's frame only when a position on it is asked
    for, as before."""

    def __init__(self, executor: "ExecutorBase") -> None:
        super().__init__()
        self._executor = executor

    def __missing__(self, role: str) -> EncodedBatch:
        batch = self._executor._batch(role)  # pyright: ignore[reportPrivateUsage]
        self[role] = batch
        return batch

    def get(self, role: str, default: Any = None) -> Any:  # type: ignore[override]
        # a plain lookup, never a load: `_batch` asks a handed-in resolution for
        # its frame through `get`, and the executor's own lazy mapping is filled
        # by `_batch` itself — loading here would recurse
        return dict.get(self, role, default)


def _local_tensor(parameter: Any) -> torch.Tensor | None:
    """A parameter as a plain tensor: a sharded model's ``DTensor`` bias is
    replicated, so its local tensor is the bias itself."""
    if parameter is None:
        return None
    to_local = getattr(parameter, "to_local", None)
    return to_local() if callable(to_local) else parameter


def _pair_offsets(
    batch: EncodedBatch, row: int, text: str, *, role: str
) -> tuple[tuple[int, int], ...]:
    """The row's token char offsets in the coordinates of ``text`` — the pair
    side the ``edit_groups`` spans are declared over. The plain frame encodes
    the text verbatim; a chat frame renders it inside a template, so the
    offsets shift by where the text sits in what was encoded. Refused as rule
    27 when the encoded text does not contain the declared text exactly once
    — spans over text the frame did not encode name nothing."""
    encoded = batch.texts[row]
    offsets = batch.offset_mapping[row]
    if encoded == text:
        return offsets
    start = encoded.find(text)
    if start < 0 or encoded.find(text, start + 1) >= 0:
        raise ValidationError(
            27,
            f"data.{role} row {row}: the row's edit_groups spans are declared over "
            f"{text!r}, which the frame did not encode verbatim ({encoded!r}) — the "
            "spans name nothing in what the model reads (sec. 2.2 `edit_groups`)",
            path=f"data.{role}",
        )
    return tuple(
        (0, 0) if (a == 0 and b == 0) else (a - start, b - start) for a, b in offsets
    )


def refuse_unstackable(name: str, site: ResolvedSite) -> None:
    """Refuse a continuation read whose steps do not stack into a frame.

    📐 A decode step attends over the whole KV cache, so a tensor indexed by the
    positions being attended *to* is ``prompt + step`` long at step ``step``
    while the query axis stays 1. The accumulating sink concatenates steps on
    dim 1, which for an ordinary tap is the position axis, so such a tap either
    raises a bare torch size error (measured: "Expected size 9 but got size 10")
    or, for a single-step budget, silently returns one step shaped like a frame.
    Neither is a continuation read.

    Both conditions are read off the declared axes rather than a component list:

    * **two position axes** — the attention pattern and the scores. There is no
      non-arbitrary answer to which of them the steps stack along;
    * **one position axis, over the keys** — ``attention_key``. Its own length
      is what grows, so consecutive steps are different lengths.

    ``attention_query`` and ``attention_z`` are query-axis-shaped and accumulate
    correctly, which is why this is a property of the axes and not of "anything
    inside the attention function".
    """
    shape = site.shape
    if shape.has_contract_form and not any(
        axis.kind == "position" and axis.name == "key" for axis in shape.axes
    ):
        return
    why = (
        "it has two position axes, so there is no single axis the decode steps "
        "stack along"
        if not shape.has_contract_form
        else "its position axis runs over the positions being attended to, "
        "which under a KV cache is the whole prefix and grows by one per step"
    )
    raise ProtocolError(
        "P4",
        f"read {name!r} reads {site.component!r} in the generated frame, whose "
        f"shape is {shape.describe()}: {why}, so the steps do not stack into "
        "one tensor. Read it in the prompt frame.",
    )


class ExecutorBase(InterningMixin, WriteMathMixin):
    """Execute one concrete document against one loaded model.

    Subclasses implement `_run_group` — everything else is shared."""

    #: Whether a no-grad read's finalized value stays on its device instead
    #: of moving to the CPU (``_finalize_read``) — detached either way, only
    #: the placement is the flag's: a fit's eval executor when a metric
    #: selects from the read on the device and its scorer copies the columns
    #: rather than the vocabulary (``train._score``), and a CUDA evaluation
    #: capture (``graph_cohort.EvaluationGraphs``). Off by default: a point's
    #: own passes hand CPU values to the writers.
    device_reads = False

    def _tap(self, site: ResolvedSite) -> TapFragments:
        """The ``whole`` / ``fragment`` pair a derivation on ``site`` runs
        through (``docs/model_parallelism.md`` §4): the world-1 identity here;
        an engine that runs above world 1 binds its collective."""
        return IDENTITY_TAP

    def __init__(
        self,
        doc: Document,
        bundle: Any,
        *,
        role_rows: Mapping[str, list[dict[str, Any]]],
        role_fields: Mapping[str, str],
        load_tensors: Callable[[str], Any],
        load_table: Callable[[str], Any] | None = None,
        stage_cache: dict[str, Stage] | None = None,
        grad_enabled: bool = False,
        coords: Mapping[str, Any] | None = None,
        interning: Interning | None = None,
        batch_rows: int | None = None,
        batches: Mapping[str, EncodedBatch] | None = None,
        resolved: StepPositions | None = None,
    ) -> None:
        #: the document in its execution form: a band site (§2.4 ``layers``)
        #: is one address across N layers, and every consumer below reasons
        #: about one module at a time, so a band is lowered to its per-layer
        #: members here — the N-site document the author would have written by
        #: hand ([`lower_bands`][]). Names the
        #: caller asks for (``read_value``, ``resolution``) are unchanged:
        #: what a band read has no single value for, lowering refuses by name
        authored = doc
        self.doc = doc = lower_bands(doc)
        #: each featurizer used at exactly one site, mapped to that site's
        #: stamp record: what the build holds a subspace start entry to
        #: (`build_stack`). Read off the document as authored, before
        #: lowering, because that is the form the load checks
        self._site_records = single_site_featurizers(authored)
        self.bundle = bundle
        self.role_rows = dict(role_rows)
        self.role_fields = dict(role_fields)
        self.load_tensors = load_tensors
        #: the score-table loader a gate's ``init.from_scores`` reads through
        #: (§2.5); ``None`` where no artifact store backs the executor
        self.load_table = load_table
        self.stage_cache: dict[str, Stage] = (
            stage_cache if stage_cache is not None else {}
        )
        self.grad_enabled = grad_enabled
        #: at most this many rows per forward (§8, execution scale): a group
        #: over more rows runs as several forwards over row windows whose
        #: captures are concatenated in row order. ``None`` is one forward per
        #: group. An execution parameter — it never enters a digest or a stamp
        #: (the run receipt records it, ``protocol/run.py``). Checked positive
        #: where it enters (the engine constructor and the CLI's argparse
        #: type), not again here
        self.batch_rows = batch_rows
        #: the campaign's shared forward groups, or ``None`` to run every
        #: group this point declares (§3 interning is opt-in per executor
        #: because only the campaign loop knows the points share one row set)
        self.interning = interning
        #: the models no trained parameter can reach (§4 "Fits") — the only
        #: groups an executor that runs *inside* a fit may serve from, or
        #: publish to, the shared store
        self.fit_constant_models = fit_constant_models(doc)
        # the seed every freshly built featurizer initialises from; the stage
        # cache is keyed by name alone, so it belongs to this one point
        self.seed = document_seed(doc)
        #: this point's sweep coordinates — they select the matching entry of
        #: a swept bundle a loaded featurizer/param points at (§2.5)
        self.coords = dict(coords or {})
        #: every read-keyed table below is keyed by [`BoundRead`][] — the
        #: read bound to the model it is taken on — never by the bare name
        self._read_values: dict[BoundRead, torch.Tensor | RaggedValue] = {}
        #: the reads whose value stays on its device after finalization
        #: because only a device-scored metric consumes it
        #: ([`device_scored_reads`][]); every other no-grad read is
        #: copied to the host as before
        self._device_scored: frozenset[BoundRead] = device_scored_reads(doc)
        #: the reads a ``save`` entry writes as tensors (``plan.saved_raw_reads``)
        self._saved_raw: frozenset[BoundRead] = saved_raw_reads(doc)
        #: per eval metric, the token ids its answer columns resolve to over
        #: this executor's rows (``answers.metric_token_ids``) — the rows never
        #: change, so a fit's eval executor resolves them once, not per pass
        self.metric_token_ids: dict[str, dict[str, list[int]]] = {}
        self._deferred_heads: dict[
            BoundRead, Callable[[torch.Tensor], torch.Tensor]
        ] = {}
        #: per dense read at a routed-interior site, the routing table
        #: gathered at the read's own rows and positions, ``(batch, position,
        #: top_k)`` — what lets a write through an expert-keyed gate align that
        #: read's slots to its own by expert id (`_align_by_expert`)
        self._read_routing: dict[BoundRead, torch.Tensor] = {}
        #: ``(write, layer, example) -> (mismatched, slots)``: per write through
        #: an expert-keyed gate, how many of the example's base slots held an
        #: expert inactive on the operand's side and so kept their base value,
        #: out of the slots the write addressed. Saved as
        #: ``routing_mismatch.json`` beside the fit (§2.5 ``expert_neuron``).
        #: Keyed rather than appended so a re-run of the group overwrites
        self._routing_mismatch: dict[tuple[str, int, int], tuple[int, int]] = {}
        #: what the writes have not yet read back: per (write, layer, the
        #: window's examples) the latest per-example count of unmatched
        #: slots, still on the device (`routing_mismatch` flushes)
        self._routing_mismatch_pending: dict[
            tuple[str, int, tuple[int, ...]], tuple[torch.Tensor, int]
        ] = {}
        #: reads whose scoped slice selected no rows at all — legal cells with
        #: nothing to measure, reported as ``unavailable`` values rather than
        #: raised (spec §4.1). The read's value is still the width-0 gather.
        self._unavailable: dict[BoundRead, Unavailable] = {}
        #: per read, the rows an address could not be aligned on (§4.1): the
        #: row-level half of ``_unavailable``, so a metric over the read can
        #: exclude those rows and score the rest (§2.10 "Eligibility")
        self._row_unavailable: dict[BoundRead, dict[int, Unavailable]] = {}
        self._groups_run: set[tuple[str, str]] = set()
        self._write_widths_checked = False
        #: ``(intervened model, write) -> {"policy", "widths", "buckets"}``
        #: for every write whose rows address different numbers of positions
        #: and whose declared ``ragged`` policy lands them (§5 rule 19):
        #: filled by [`check_write_widths`][] before any forward, the run
        #: receipt's ``execution.ragged`` entry for the point
        #: ([`ragged_geometry_of`][]). Empty for every document that
        #: authors no policy — such a write is refused there instead.
        self.ragged_geometry: dict[tuple[str, str], dict[str, Any]] = {}
        self._edit_groups_checked = False
        #: what [`check_scoring`][] found for the base table — ``None``
        #: until it has run; the run receipt's ``scoring`` block per point
        self.scoring_check: ScoringCheck | None = None
        #: the step's resolved positions (§2.3): handed in by the engine
        #: when the protocol layer resolved this step before any weights
        #: loaded (``pipeline.resolve_positions``, keyed by
        #: ``positions_key``), else built here on first use from this
        #: executor's own frames through the same protocol functions
        #: (`_step_positions`). Every position a gather, a write or the
        #: ledger reads comes off it; the executor resolves nothing itself
        self._resolved: StepPositions | None = resolved
        #: the location ledger (§6), built on first request and only when the
        #: document saves one ([`location_ledger`][])
        self._ledger: LocationLedger | None = None
        #: the encoded frame per input role — built on first use from the
        #: role's rows, or handed in: a fit's minibatch executor runs a row
        #: *selection* of its point's frame (``EncodedBatch.select``) so every
        #: minibatch of every point in a cohort shares one padded width and
        #: their forwards concatenate (§4 "Cohorts"). What is handed in must
        #: describe exactly ``role_rows``, row for row; the frame's texts are
        #: checked against the rows' field on first use
        self._batches: dict[str, EncodedBatch] = dict(batches or {})
        self._batches_checked: set[str] = set()
        self._continuations: dict[tuple[str, str], Continuation] = {}
        #: per generate read, the decode steps each row addresses — the
        #: same list the gather used, kept because metrics need to know
        #: *which* steps a value covers (and that a row covered none)
        self._read_steps: dict[BoundRead, list[list[int]]] = {}
        #: implementation requirements the run's addresses imposed (§7.3,
        #: e.g. "attn_eager") — execution metadata, stamped into the artifact
        #: identity next to the engine name, never canonical form
        self.applied_requirements: set[str] = set()
        #: per forward group this executor ran or was served, ``{write
        #: member: fire count}`` (§4 "Fires", ``neural/shared/fires.py``):
        #: the run receipt's ``fires`` entry for the point. Only groups with
        #: writes appear; a group whose member fired other than its declared
        #: count never gets here — the point is refused before anything is
        #: published
        self.fires: dict[tuple[str, str], dict[str, int]] = {}

    # ------------------------------------------------------------------ #
    # public surface
    # ------------------------------------------------------------------ #

    def _group_of(self, ref: BoundRead) -> tuple[str, str]:
        """The ``(model, input role)`` forward group one bound read is
        gathered from."""
        return self.doc.group_of(ref)

    def _ref(self, ref: "BoundRead | str") -> BoundRead:
        """A read reference as the tables key it. The public surface also
        takes a bare read name — the sugar a document itself allows (§2.7) —
        which binds to the one model the read is taken on."""
        return bound_read(self.doc, ref) if isinstance(ref, str) else ref

    def read_value(self, ref: "BoundRead | str") -> "torch.Tensor | RaggedValue":
        """The (featurized, dims-selected) value of one bound read; runs its
        group (and, transitively, operand groups) on first use."""
        ref = self._ref(ref)
        if ref not in self._read_values:
            self.check_write_widths()
            self.check_scoring()
            self.check_edit_groups()
            self._run_group(*self._group_of(ref))
        value = self._read_values[ref]
        if ref in self._deferred_heads:
            value = (
                RaggedValue(self._project_deferred(ref, value.flat), value.widths)
                if isinstance(value, RaggedValue)
                else self._project_deferred(ref, value)
            )
            self._read_values[ref] = value
            del self._deferred_heads[ref]
        return value

    def _project_deferred(
        self, ref: BoundRead, value: "torch.Tensor"
    ) -> "torch.Tensor":
        """Run read ``name``'s owed ``lm_head`` over ``value`` and hand the
        logits back where the read's value lives.

        A head is only deferred when gradients are off (the registration in
        the engine's generate tail), and that is exactly when
        `_finalize_read` has detached the kept ``ln_final`` rows to the
        CPU — so the projection runs on the head's device, where its weight
        is, and its result comes back to the CPU like every other stored read
        value. On a CPU bundle both moves are no-ops."""
        head = self._deferred_heads[ref]
        with torch.no_grad():
            return head(value.to(self.bundle.devices.head)).detach().cpu()

    def generated_metric(self, agg: BoundAggregation) -> list[list[Any]]:
        """Reduce ordinary continuation aggregations before releasing each
        projection.

        Only untransformed, unsaved lm_head reads are deferred. Explicit tensor
        consumers still get full logits through read_value. Sixteen positions
        bound each vocabulary projection independently of forward batch size.
        """
        from causalab.neural.shared.metrics import compute_windowed_metric
        from causalab.protocol.schema import METRIC_DOMAINS

        metric = agg.spec
        name = agg.read
        target = agg.target
        rows = self.rows_for_metrics()
        ids = (
            self.generated_ids(name)
            if METRIC_DOMAINS.get(str(metric.kind)) == "ids"
            else None
        )
        if ids is not None:
            return compute_windowed_metric(
                metric, [], rows, self.bundle.tokenizer, generated_ids=ids
            )
        if name not in self._deferred_heads or (
            target is not None and target not in self._deferred_heads
        ):
            return compute_windowed_metric(
                metric,
                self.windowed_value(name),
                rows,
                self.bundle.tokenizer,
                target_windows=self.windowed_value(target) if target else None,
                vocab_axis=read_is_vocabulary(self.doc, name.read),
            )
        value = self._read_values[name]
        windows = (
            list(torch.split(value.flat, list(value.widths)))
            if isinstance(value, RaggedValue)
            else list(value)
        )
        target_windows = None
        if target is not None:
            target_value = self._read_values[target]
            target_windows = (
                list(torch.split(target_value.flat, list(target_value.widths)))
                if isinstance(target_value, RaggedValue)
                else list(target_value)
            )
            if [len(w) for w in windows] != [len(w) for w in target_windows]:
                raise ProtocolError(
                    "P2",
                    "continuation metric and target windows have different lengths",
                )
        out = []
        for index, (row, window) in enumerate(zip(rows, windows)):
            values = []
            for start in range(0, len(window), 16):
                chunk = window[start : start + 16]
                if chunk.shape[0]:
                    logits = self._project_deferred(name, chunk)
                    target_logits = (
                        self._project_deferred(
                            target, target_windows[index][start : start + 16]
                        )
                        if target is not None and target_windows is not None
                        else None
                    )
                    values.extend(
                        compute_windowed_metric(
                            metric,
                            [logits],
                            [row],
                            self.bundle.tokenizer,
                            target_windows=[target_logits]
                            if target_logits is not None
                            else None,
                            vocab_axis=True,
                        )[0]
                    )
                    del logits, target_logits
            out.append(values)
        return out

    def resolution(self, ref: "BoundRead | str") -> Resolution:
        """How one read resolved: ``Available`` (the ordinary case) or the
        ``Unavailable`` a scoped slice that selected nothing produced. Runs the
        read's group on first use, like [`read_value`][]. The denominator
        key is [`cell_key`][] over this point's
        coordinates — the same key the saved entry takes."""
        ref = self._ref(ref)
        value = self.read_value(ref)
        if ref in self._unavailable:
            return self._unavailable[ref]
        rows = sum(value.widths) if isinstance(value, RaggedValue) else value.shape[0]
        return available(
            {"read": ref.read, "rows": rows}, cell_key(ref.read, self.coords)
        )

    def row_resolutions(self, ref: "BoundRead | str") -> list[Unavailable | None]:
        """Per row of one read's input role, the ``Unavailable`` the row became
        when its address could not be aligned (§4.1), else ``None``.

        The row-level half of [`resolution`][]: a read whose cell is
        unavailable because *some* rows failed to align still has rows that
        did, and a metric over it scores those and reports the rest as
        excluded measurements (§2.10 "Eligibility"). A cell unavailable for a
        reason that is not a row's — an ``expert:`` face the router sent no
        token — has no row-level record, and every entry is ``None``.
        """
        ref = self._ref(ref)
        self.read_value(ref)
        per_row = self._row_unavailable.get(ref, {})
        rows = len(self.role_rows[self._group_of(ref)[1]])
        return [per_row.get(i) for i in range(rows)]

    def dense_rows(self, ref: "BoundRead | str", rows: Sequence[int]) -> torch.Tensor:
        """A read's value at the given rows, as the dense ``(len(rows), …)``
        tensor a metric reduces — the eligible-row form of [`dense_value`][].

        A read some of whose rows aligned on nothing is ragged (those rows
        have width zero), and [`dense_value`][] rightly refuses to reduce
        it. Over the rows that *did* align it is not ragged at all: each has
        exactly the one position a metric reduces, so selecting them gives
        the same dense value the read would have had without the excluded
        rows. A row of any other width is still the ragged refusal.
        """
        ref = self._ref(ref)
        value = self.read_value(ref)
        if not isinstance(value, RaggedValue):
            index = torch.tensor(list(rows), dtype=torch.long, device=value.device)
            return value[index]
        widths = list(value.widths)
        if any(widths[i] != 1 for i in rows):
            raise ProtocolError(
                "P2",
                f"read {ref.read!r} is ragged (unequal per-row position widths) — "
                "metrics reduce one aligned position per example",
            )
        offsets = [sum(widths[:i]) for i in rows]
        index = torch.tensor(offsets, dtype=torch.long, device=value.flat.device)
        return value.flat[index].unsqueeze(1)

    def check_write_widths(self) -> None:
        """Rule 19, checked **before any forward pass**.

        A ragged write used to surface from the landing path — i.e. on the
        accelerator, after the weights had loaded, with no rule number. On a
        35 B model that is minutes of wasted compute for a fact the encoded
        batch already knows, and it shaped whole corpora: a dataset had to end
        every request in the same token (a ``.``) so negative indices aligned
        across rows.

        Only the tokenizer can say how wide a row is, which is why this cannot
        live in ``validate --data`` — that verb reads the dataset but holds no
        tokenizer, and giving it one would break the pure verbs' network- and
        torch-free contract. Encoding is the earliest point the question has an
        answer.

        What a ragged write meets here is its declared ``ragged`` policy
        (§2.8): ``refuse`` — and an absent field — is the refusal above;
        ``exact_length_buckets`` and ``padded_masked`` land every row at its
        own width instead (`_land_ragged`), and this check records what
        they will land under ([`ragged_geometry`][]) for the run receipt.
        The policy is authored in the pure layer and *resolved* here, on the
        encoded batch — the V19 boundary is unchanged.
        """
        if self._write_widths_checked:
            return
        self._write_widths_checked = True
        for model, im in self.doc.intervened_models.items():
            input_role = str(im.input)
            for ename in write_names(self.doc, model) or ():
                write = self.doc.writes[ename]
                spec = self._spec(write.pos)
                if (
                    spec.all is None
                    and spec.variable is None
                    and spec.column is None
                    and not isinstance(spec, SpanSpec)
                ):
                    continue  # an index (scoped or not) is uniform by shape
                # a `variable` or `column` window is as wide as the row's
                # value tokenizes, a span — atomic or not — as wide as its
                # members make it on each row; a ragged one is rule 19's
                # business exactly as an `all` window is (§2.3)
                if spec.generated is not None:
                    continue  # rule 16 already refuses a generated write
                site = resolve_site(self.bundle, self.doc.sites[str(write.site)])
                if not site.shape.has_contract_form:
                    # This tap's last axis is positions, not features, so the
                    # landing path edits the whole tensor and never gathers —
                    # there are no per-row widths to be ragged. Mirroring that
                    # skip here is what keeps the pre-flight from refusing an
                    # `attention_scores` write, which is the point of the
                    # component.
                    continue
                batch = self._batch(input_role)
                per_row = self._positions(write.pos, batch, input_role)
                widths = {len(row) for row in per_row}
                if len(widths) == 1:
                    continue
                policy = write.ragged or "refuse"
                if policy == "refuse":
                    raise ragged_write_error(ename, sorted(widths), model=model)
                # a declared policy lands every row at its own width (the
                # landing path); what it lands under is recorded here, before
                # any forward, for the receipt's `execution.ragged` (§8)
                self.ragged_geometry[(model, ename)] = ragged_geometry_of(
                    policy, per_row
                )

    def check_scoring(self) -> ScoringCheck:
        """The table's recorded ``string_mode`` against every ``match``
        ``mode`` this document declares, **before any forward pass** — the
        same comparison ``validate --data`` makes
        ([`causalab.causal.scoring.check_scoring`][], §2.2, §2.10), run
        again here because a run may start from rows the pure verbs never saw.
        A ``prefix`` table under ``mode: exact`` is refused under rule 4,
        naming both modes and the derivation; an unrecorded table compares
        nothing and the receipt says so. Torch-free arithmetic over the rows,
        so this costs nothing a forward would have paid for.
        """
        if self.scoring_check is not None:
            return self.scoring_check
        rows = self.rows_for_metrics()
        try:
            self.scoring_check = check_scoring(
                rows,
                declared_modes(
                    {
                        label: agg.spec
                        for label, agg in self._aggregations_by_label().items()
                    }
                ),
                where="data.base",
            )
        except ScoringMismatch as err:
            owner = self._aggregations_by_label()[err.metric].owner
            raise ValidationError(
                4, str(err), path=f"{owner}.aggregation.mode"
            ) from err
        except ScoringError as err:
            raise ValidationError(4, f"data.base: {err}", path="data.base") from err
        return self.scoring_check

    def _aggregations_by_label(self) -> dict[str, BoundAggregation]:
        """The document's aggregations, one per label (a label consumed in
        several places is one reduction, checked once)."""
        out: dict[str, BoundAggregation] = {}
        for agg in self.doc.aggregations():
            out.setdefault(agg.label, agg)
        return out

    def check_edit_groups(self) -> None:
        """Rule 27's pair half, checked **before any forward pass** (§2.2,
        §5 item 27): an ``atomic`` edit group a row declares
        ([`causalab.causal.pair_validation`][]) is addressed whole or not at all by
        each intervened model — the positions of its writes on its input, and
        of the reads its writes take their operands from, on theirs. A
        constituent addressed without its siblings is refused naming the
        group and the missing siblings; a non-atomic group, a fully addressed
        atomic group and a table without the column all run.

        Encode-time for the reason rule 19 is: which tokens a char span
        covers is the tokenizer's to say, and the pure verbs hold none. The
        char → token reading is the one ``variable`` positions take
        (``encoding._chars_to_tokens``), so a declared span and a ``variable``
        anchor over the same characters name the same tokens.
        """
        if self._edit_groups_checked:
            return
        self._edit_groups_checked = True
        base_rows = self.role_rows.get("base", [])
        if not any(row.get(EDIT_GROUPS_COLUMN) is not None for row in base_rows):
            return
        groups_by_row: list[tuple[EditGroup, ...]] = []
        for index, row in enumerate(base_rows):
            try:
                groups_by_row.append(parse_edit_groups(row))
            except EditGroupError as err:
                raise ValidationError(
                    27, f"data.base row {index}: {err}", path="data.base"
                ) from err
        for mname, im in self.doc.intervened_models.items():
            addressed: dict[str, list[set[int]]] = {}
            for ename in write_names(self.doc, mname) or ():
                write = self.doc.writes[ename]
                self._address(addressed, write.pos, str(im.input))
                for ref in operand_reads(self.doc, write.do):
                    read = self.doc.reads[ref.read]
                    self._address(
                        addressed, read.pos, self.doc.group_of(ref)[1], cell=ref
                    )
            for role, per_row in addressed.items():
                side = "counterfactual" if role == "counterfactual" else "base"
                batch = self._batch(role)
                for index, groups in enumerate(groups_by_row):
                    if not any(group.atomic for group in groups):
                        continue
                    text = row_texts(base_rows[index])[side]
                    offsets = _pair_offsets(batch, index, text, role=role)
                    try:
                        check_component_wise(
                            groups,
                            offsets,
                            per_row[index],
                            side=side,
                            where=f"intervened_models.{mname} (input {role!r}), row {index}",
                            text=text,
                        )
                    except EditGroupError as err:
                        raise ValidationError(
                            27,
                            f"{err} (sec. 2.2 `edit_groups`)",
                            path=f"intervened_models.{mname}",
                        ) from err

    def _address(
        self,
        addressed: dict[str, list[set[int]]],
        pos: Any,
        role: str,
        *,
        cell: BoundRead | None = None,
    ) -> None:
        """Fold one position's resolved runs on ``role`` into ``addressed``
        (per row, the padded-frame indices). A generated position addresses
        the continuation, not the prompt the spans are declared over, and is
        skipped. ``cell`` names a read, whose unalignable rows become its
        ``unavailable`` cell as they would in the run; a write's propagate."""
        spec = self._spec(pos)
        if spec.generated is not None:
            return
        rows = self._positions(pos, self._batch(role), role, cell=cell)
        per_row = addressed.setdefault(role, [set() for _ in rows])
        for index, run in enumerate(rows):
            per_row[index].update(run)

    def dense_value(self, ref: "BoundRead | str") -> torch.Tensor:
        """A read value that must be a dense tensor (metric inputs): a
        ragged read has no per-example position alignment to reduce."""
        ref = self._ref(ref)
        value = self.read_value(ref)
        if isinstance(value, RaggedValue):
            raise ProtocolError(
                "P2",
                f"read {ref.read!r} is ragged (unequal per-row position widths) — "
                "metrics reduce one aligned position per example",
            )
        return value

    def is_generated(self, ref: "BoundRead | str") -> bool:
        """Whether this read addresses the continuation frame (§2.3)."""
        ref = self._ref(ref)
        return generated_budget(self.doc, self.doc.reads[ref.read].pos) is not None

    def windowed_value(self, ref: "BoundRead | str") -> list[torch.Tensor]:
        """One read's value split per example: ``(positions_i, …)`` each.

        The metric surface for a continuation read. Unlike
        [`dense_value`][] it welcomes ragged widths, because in the
        continuation frame they are the answer rather than a
        misalignment — a row that stopped early, or never said the value a
        ``variable`` anchor looks for, contributes an **empty** tensor.
        """
        ref = self._ref(ref)
        value = self.read_value(ref)
        if isinstance(value, RaggedValue):
            widths = list(value.widths)
            return list(torch.split(value.flat, widths)) if widths else []
        if value.dim() == 2:  # one position per row, already squeezed
            return [value[i].unsqueeze(0) for i in range(value.shape[0])]
        return [value[i] for i in range(value.shape[0])]

    def addressed_steps(self, ref: "BoundRead | str") -> list[list[int]]:
        """Per example, the decode steps one generate read covers."""
        ref = self._ref(ref)
        if ref not in self._read_steps:
            self._run_group(*self._group_of(ref))
        return self._read_steps[ref]

    def generated_ids(self, ref: "BoundRead | str") -> list[list[int]]:
        """Per example, the token ids at a generate read's addressed steps.

        The ``ids`` metric domain (§2.10): these come from the decode
        itself, so a metric that only needs them obliges no vocabulary
        projection anywhere.
        """
        ref = self._ref(ref)
        steps = self.addressed_steps(ref)
        continuation = self._continuations[self._group_of(ref)]
        return [
            [int(continuation.token_ids[row, step]) for step in row_steps]
            for row, row_steps in enumerate(steps)
        ]

    def run_all(self) -> None:
        """Run every group the document implies (all reads materialize)."""
        self.check_write_widths()
        self.check_scoring()
        self.check_edit_groups()
        for ref in self.doc.read_refs():
            self._run_group(*self.doc.group_of(ref))

    def stage(self, name: str) -> Stage:
        """The (shared) featurizer stage instance for one declared name."""
        if name not in self.stage_cache:
            width, site, entry = self._featurizer_input(name)
            # §2.5 `axis`: the window the entry addresses sizes a position gate;
            # `build_stack` decides whether it matters and refuses a `None` for a
            # position gate — the same line `_read_stack` hands it, so a gate
            # reached through either path gets one message. Rule 4 already held
            # every use of the gate to one fixed span, so the first entry's
            # window is the gate's
            build_stack(
                name,
                dict(self.doc.featurizers),
                width=width,
                load_tensors=self.load_tensors,
                load_table=self.load_table,
                stage_cache=self.stage_cache,
                device=self._site_device(site),
                seed=self.seed,
                coords=self.coords,
                site_shape=site.shape,
                site_component=site.component,
                model_info=self.bundle.info,
                position_width=span_length(self._spec(entry.pos)),
                site_records=self._site_records,
            )
            # a budget pool's members are built together (§2.5 `pool`): a
            # member's mask is solved over the whole pool, so none may be
            # computed before every co-member exists
            link_budget_pools(self.doc.featurizers, self.stage_cache, self.stage)
        return self.stage_cache[name]

    def _site_device(self, site: ResolvedSite) -> torch.device:
        """The device a site's tensor lives on, from the bundle's device map
        (``shared/devices.py``): its block's for a block-scoped component,
        the head's for the final norm and the head, the embedding's for the
        embedding — the family's tap declares which scope a component is on.
        A featurizer stage is built where its site is, so it meets the value
        it featurizes without a move."""
        devices = self.bundle.devices
        tap = self.bundle.adapter.taps.get(site.component)
        scope = tap.scope if tap is not None else "block"
        if scope in ("final_norm", "lm_head"):
            return devices.head
        if scope == "embedding":
            return devices.embedding
        return devices.device_of(site.layer)

    def featurizer_sites(self, name: str) -> list[ResolvedSite]:
        """The resolved site of every read and write whose featurizer chain
        names ``name`` — where the featurizer's math runs, and so, under a
        pipeline, which stage computes its gradient
        (``stages.StageForward.trained_owner``)."""
        sites: list[ResolvedSite] = []
        for entry in (*self.doc.reads.values(), *self.doc.writes.values()):
            ref = entry.featurizer
            chain = (
                (ref,)
                if isinstance(ref, str)
                else tuple(ref)
                if isinstance(ref, tuple)
                else ()
            )
            if name in chain:
                sites.append(resolve_site(self.bundle, self.doc.sites[str(entry.site)]))
        return sites

    def _featurizer_input(
        self, name: str
    ) -> "tuple[int, ResolvedSite, ReadSpec | WriteSpec]":
        """The input of one declared featurizer: the width it is sized to —
        the site width of a chain that uses it, folded through the stages
        before it (§2.5 composition — a gate after a k=3 rotation is 3-wide)
        — the resolved site that chain starts at, whose shape a grouped gate
        derives its map from, and the read or write entry itself, whose
        window sizes a position gate (§2.5 ``axis``). One scan, the first
        entry that uses the name."""
        from causalab.neural.shared.featurizers import stage_output_width

        for entry in (*self.doc.reads.values(), *self.doc.writes.values()):
            ref = entry.featurizer
            chain = (
                (ref,)
                if isinstance(ref, str)
                else tuple(ref)
                if isinstance(ref, tuple)
                else ()
            )
            if name not in chain:
                continue
            site = resolve_site(self.bundle, self.doc.sites[str(entry.site)])
            running = (
                site.feature_slice.stop - site.feature_slice.start
                if site.feature_slice is not None
                else component_width(self.bundle.info, site.component, head=None)
            )
            for member in chain:
                if member == name:
                    return running, site, entry
                out = stage_output_width(self.doc.featurizers[member], running)
                if out is None:
                    raise ProtocolError(
                        "P2",
                        f"cannot size {name!r}: {member!r} before it in the "
                        "chain has no spec-derivable output width",
                    )
                running = out
        raise ProtocolError("P2", f"featurizer {name!r} is used by no read or write")

    def input_token_ids(self, input_role: str) -> list[list[int]]:
        """Exact unpadded inputs for output provenance, without retokenizing."""
        batch = self._batch(input_role)
        # one host copy of each tensor, then per-row slicing on the host
        ids, mask = batch.input_ids.cpu(), batch.attention_mask.cpu()
        return [tokens[real.bool()].tolist() for tokens, real in zip(ids, mask)]

    def rows_for_metrics(self) -> list[dict[str, Any]]:
        """Metric columns resolve against the base rows — the pairing
        anchor (§2.2: rows are paired; one base row + its counterfactuals form one
        example)."""
        return self.role_rows["base"]

    def reset_reads(self) -> None:
        """Drop cached read values and group state (training steps re-run
        forwards with updated featurizer parameters).

        The encoded batches and the shared interning store are untouched: the
        rows have not changed, and a constant group's raw capture is exactly
        what the next step should be served rather than re-run."""
        self._read_values.clear()
        self._deferred_heads.clear()
        self._read_routing.clear()
        self._groups_run.clear()
        self._write_widths_checked = False
        self.ragged_geometry.clear()
        self._continuations.clear()
        self._read_steps.clear()

    # ------------------------------------------------------------------ #
    # what an engine implements
    # ------------------------------------------------------------------ #

    def _run_group(self, model: str, input_role: str) -> None:
        """Run one (model, input role) forward group: land its writes and
        fill ``self._read_values`` for its reads."""
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # shared plumbing
    # ------------------------------------------------------------------ #

    def frame(self, role: str) -> EncodedBatch:
        """The encoded frame of one input role — every row of the role, left
        padded to one width. What a fit's minibatch executors are built as
        selections of (``EncodedBatch.select``, §4 "Cohorts")."""
        return self._batch(role)

    def _batch(self, role: str) -> EncodedBatch:
        if role not in self._batches and self._resolved is not None:
            # the protocol layer encoded this role before any weights loaded:
            # its frame, as ints, becomes this executor's device frame — the
            # same rows, checked below like any handed-in frame, and the same
            # *tokens*: the positions it carries are indices into that frame,
            # so a frame this bundle's tokenizer does not reproduce would make
            # every one of them address the wrong token, silently. Re-encoding
            # is a memo hit when the tokenizers are the same object and one
            # tokenizer call otherwise; the comparison is host ints
            frame = self._resolved.frames.get(role)
            if isinstance(frame, PositionFrame):
                own = self._encode(role)
                if (
                    own.token_ids != frame.token_ids
                    or own.attention_mask != frame.attention_mask
                    or own.offset_mapping != frame.offset_mapping
                    or own.prefix_lengths != frame.prefix_lengths
                ):
                    raise ProtocolError(
                        "P2",
                        f"the positions handed to this executor for input {role!r} "
                        "were resolved on a frame this model's tokenizer does not "
                        "reproduce — a different tokenizer, or different rows — so "
                        "every index in them would address the wrong token. The run "
                        "door resolves positions with the engine's own tokenizer "
                        "(causalab.io.tokenizer.load_tokenizer); hand the engine a "
                        "resolution built with it, or none",
                    )
                self._batches[role] = EncodedBatch.from_frame(
                    frame, self.bundle.devices.embedding
                )
            elif isinstance(frame, EncodedBatch):
                self._batches[role] = frame
        if role in self._batches and role not in self._batches_checked:
            # a handed-in frame (a minibatch's selection of its point's frame)
            # must be these rows, row for row: the row count always, and under
            # the plain frame the texts themselves — a chat frame's texts are
            # the rendered template, so there only the count is comparable
            handed = self._batches[role]
            rows = self.role_rows[role]
            expected = tuple(
                str(select_field(row, self.role_fields[role])) for row in rows
            )
            if len(handed.texts) != len(rows) or (
                self.doc.segments is None and handed.texts != expected
            ):
                raise AssertionError(
                    f"the frame handed in for input {role!r} encodes "
                    f"{len(handed.texts)} rows that are not this executor's "
                    f"{len(rows)} — a selection must be built from the "
                    "same rows it is handed to"
                )
            self._batches_checked.add(role)
        if role not in self._batches:
            # inputs are encoded onto the embedding's device: the first
            # module that reads them
            self._batches[role] = EncodedBatch.from_frame(
                self._encode(role), self.bundle.devices.embedding
            )
        return self._batches[role]

    def _encode(self, role: str) -> PositionFrame:
        """This role's rows through this bundle's tokenizer, as the protocol
        layer encodes them ([`encode_role`][]): the document's ``segments`` frame when it declares one
        (§2.2.1 — the chat frame renders through the tokenizer's own template
        and sets the real prefix; a plain frame with declared column segments
        locates them on the text), else the plain-text frame. Memoized per
        tokenizer by the protocol's ``encode``."""
        return encode_role(
            self.bundle.tokenizer,
            self.doc,
            self.role_rows[role],
            self.role_fields[role],
        )

    def location_ledger(self) -> LocationLedger:
        """The location ledger (§6, ``protocol/positions/ledger.py``): one row
        per (example, edit group, constituent, side, token index, token id,
        decoded token) for every prompt-frame position every read and write
        of this point resolves, on every input it resolves it on — the same
        indices `_positions` hands the gathers, recorded once.

        Built on first use and cached; the caller (``execution.py``) asks for
        it only when the document saves a ``location_ledger`` entry (§2.12)
        and the protocol layer did not build it already. The builder is the
        protocol's ([`build_ledger`][])
        over this executor's `_step_positions`; whether a site's tap has
        a position axis is read off the resolved tap here, off the registry
        there — the same answer by construction (``FeatureShape``).
        """
        if self._ledger is not None:
            return self._ledger

        def has_positions(site_name: str) -> bool:
            site = resolve_site(self.bundle, self.doc.sites[site_name])
            return site.shape.has_contract_form

        self._ledger = build_ledger(
            self.doc,
            self._step_positions(),
            self.bundle.tokenizer,
            has_positions=has_positions,
        )
        return self._ledger

    def _step_positions(self) -> StepPositions:
        """This step's positions: the protocol layer's, when the engine was
        handed them, else one built here over this executor's frames — each
        role encoded on first request (`_batch`) — through the same
        protocol functions. What the protocol never resolved (a constituent,
        an operand read the ledger asks for, a step no representative
        covered) is resolved on demand, so a consumer never sees a missing
        address."""
        if self._resolved is None:
            self._resolved = StepPositions(
                frames=_LazyFrames(self),
                role_rows=self.role_rows,
                role_fields=self.role_fields,
            )
        return self._resolved

    def _row_windows(self, total: int) -> list[RowWindow]:
        """The microbatches one group over ``total`` rows runs as: ceil(total /
        batch_rows) windows of at most ``batch_rows`` rows, in row order — or
        the one whole window when no bound is set."""
        if self.batch_rows is None or self.batch_rows >= total:
            return [RowWindow(0, total, total)]
        return [
            RowWindow(start, min(start + self.batch_rows, total), total)
            for start in range(0, total, self.batch_rows)
        ]

    def _spec(self, pos: Any) -> PositionSpec:
        return spec_of(self.doc, pos)

    def _positions(
        self,
        pos: Any,
        batch: EncodedBatch,
        input_role: str,
        *,
        cell: BoundRead | None = None,
    ) -> list[list[int]]:
        """Every row's positions for one spec on one input — read off the
        step's resolution (`_step_positions`; ``batch`` is that role's
        frame, kept on the signature for the engines' overrides).

        ``cell`` names the *read* whose rows these are. A row the address
        cannot be aligned on — its ``variable`` / ``column`` value occurs zero
        or several times — then becomes that read's ``unavailable`` cell
        (spec §4.1, reason ``alignment_missing`` / ``alignment_ambiguous``,
        the detail naming the value, its count and the row) and contributes
        no positions, so the row is an excluded measurement in the
        denominator rather than a refusal of the run. Without ``cell`` (a
        write, the width pre-flight) the typed refusal propagates before any
        forward: a write that skipped a row would report a number for an
        intervention that did not happen.

        A spec with a declared ``alignment`` is checked against the pair's
        observed cardinality on first use
        ([`check_declared_alignment`][causalab.protocol.positions.resolve.check_declared_alignment],
        once per address per step).
        """
        spec = self._spec(pos)
        resolved = self._step_positions().address(pos, spec, input_role)
        if resolved.problems:
            if cell is None:
                raise resolved.first_problem().error()
            rows = self.role_rows[input_role]
            key = cell_key(cell.read, self.coords)
            # the row-level record (§2.10 "Eligibility"): each excluded row
            # under its own reason, so a metric over this read scores the
            # rows that aligned and reports these as excluded measurements
            per_row = self._row_unavailable.setdefault(cell, {})
            problems = sorted(resolved.problems.items())
            for i, problem in problems:
                row_cell = unalignable(problem.cardinality, problem.message, key)
                assert row_cell is not None
                per_row.setdefault(i, row_cell)
            first = problems[0][1]
            excluded = unalignable(
                first.cardinality,
                f"read {cell.read!r} at position {pos if isinstance(pos, str) else spec!r} "
                f"on input {input_role!r}: {len(problems)} of {len(rows)} rows "
                "could not be aligned and contribute no positions — "
                + "; ".join(problem.message for _, problem in problems),
                key,
            )
            assert excluded is not None
            self._unavailable[cell] = excluded
        self._step_positions().check_alignment(pos, spec, input_role)
        return resolved.rows()

    @staticmethod
    def _gather(
        tensor: torch.Tensor, per_row: list[list[int]], what: str
    ) -> "torch.Tensor | RaggedValue":
        widths = {len(row) for row in per_row}
        if len(widths) == 1:
            return gather_positions(tensor, dense_index(per_row, tensor.device))
        # ragged: one flat advanced index, (total_positions, ...) + widths
        return RaggedValue(
            flat=gather_positions(tensor, flat_index(per_row, tensor.device)),
            widths=tuple(len(row) for row in per_row),
        )

    def _read_taps(
        self, model: str, input_role: str, reads: Iterable[tuple[str, ReadSpec]]
    ) -> "dict[str, ReadTap]":
        """The taps of one group's prompt-frame reads on this executor
        ([`resolve_read_taps`][]): an
        ``lm_head`` read at named positions projects the head over its
        gathered rows, except where this executor differentiates through
        it — a grad-enabled executor reading a model a trained parameter can
        reach — which keeps the head as the model runs it, so the training
        gradient is the model's to the bit. A fit-constant group's read on
        the same executor carries no graph (the network is frozen) and
        projects."""
        differentiable = self.grad_enabled and model not in self.fit_constant_models
        return resolve_read_taps(
            self.bundle,
            self.doc,
            model,
            input_role,
            reads,
            differentiable=differentiable,
        )

    def _read_stack(
        self, read: ReadSpec | WriteSpec, site: ResolvedSite
    ) -> FeaturizerStack:
        # Lazily: `build_stack` ignores the width entirely when no featurizer is
        # referenced (it returns an Identity stack), and some components have no
        # width to give — `input_ids` carries integer ids on a position axis and
        # `expert_idx` a routing table, so both refuse rather than invent one
        # (§5.4). Asking for the width up front made an unfeaturized read of
        # those impossible, which is not what the refusal is for: it exists to
        # reject a *featurizer*, not a read.
        if read.featurizer is None:
            width = 0
        elif site.feature_slice is not None:
            width = site.feature_slice.stop - site.feature_slice.start
        else:
            # `head=site.head` matters only for a *derived* component, which
            # carries no feature_slice: a head there narrows the value's width
            # without narrowing the captured tensor's.
            width = component_width(self.bundle.info, site.component, head=site.head)
        stack = build_stack(
            read.featurizer,
            dict(self.doc.featurizers),
            width=width,
            load_tensors=self.load_tensors,
            load_table=self.load_table,
            stage_cache=self.stage_cache,
            device=self._site_device(site),
            seed=self.seed,
            coords=self.coords,
            site_shape=site.shape,
            site_component=site.component,
            model_info=self.bundle.info,
            # §2.5 `axis`: a position gate in this chain is sized by the
            # entry's fixed span; `None` for every other chain
            position_width=span_length(self._spec(read.pos)),
            site_records=self._site_records,
        )
        link_budget_pools(self.doc.featurizers, self.stage_cache, self.stage)
        return stack

    def _finalize_read(
        self,
        ref: BoundRead,
        read: ReadSpec,
        site: ResolvedSite,
        raw: torch.Tensor,
        batch: EncodedBatch,
        input_role: str,
        *,
        per_row: list[list[int]] | None = None,
        project: Callable[[torch.Tensor], torch.Tensor] | None = None,
        expert_idx: torch.Tensor | None = None,
        to_cpu: bool | None = None,
    ) -> "torch.Tensor | RaggedValue":
        """One bound read's value: gather at its positions, then featurize.

        ``per_row`` overrides position resolution — the continuation frame
        resolves to decode steps, which the caller has already worked out
        against the decode. ``project`` runs on the gathered slice before
        anything else, which is how an ``lm_head`` continuation read is
        served from kept ``ln_final`` activations: the vocabulary projection
        happens at the addressed positions and nowhere else.
        ``to_cpu`` defaults to ``not self.device_reads``: an eval executor and
        a CUDA evaluation capture keep their read values on the device, and
        their scorer copies out only what a metric selects. A read that only
        a device-scored metric consumes ([`device_scored_reads`][]) stays on
        the device on any executor, for the same scorer's sake.
        """
        rname = ref.read
        if to_cpu is None:
            to_cpu = not self.device_reads and ref not in self._device_scored
        if not site.shape.has_contract_form:
            return whole_native_tensor(rname, read, raw, site)
        if per_row is None:
            per_row = self._positions(read.pos, batch, input_role, cell=ref)
        if site.expert is not None:
            return self._expert_selected(ref, read, site, raw, expert_idx, per_row)
        gathered = self._gather(raw, per_row, f"read {rname!r}")
        if project is not None:
            if isinstance(gathered, RaggedValue):
                gathered = RaggedValue(
                    flat=project(gathered.flat), widths=gathered.widths
                )
            else:
                gathered = project(gathered)
        ragged = isinstance(gathered, RaggedValue)
        value = gathered.flat if isinstance(gathered, RaggedValue) else gathered
        if site.shape.state_axes:
            return self._state_read(rname, read, site, value, gathered)
        if site.derivation is not None:
            # After the gather, deliberately: the value is `heads` times wider
            # than the tensor it comes from, so deriving it before the gather
            # would cost `seq · H · hidden` where this costs
            # `n_positions · H · hidden`.
            value = _derive(site, value, rname, self._tap(site))
        if site.feature_slice is not None:
            value = value[..., site.feature_slice]
        routing = None
        if expert_idx is not None:
            # the routing table at the same rows and positions as the value —
            # what an expert-keyed gate keys its parameters by, and what a
            # later write through one aligns this read's slots to its own by
            idx_gathered = self._gather(expert_idx, per_row, f"read {rname!r}")
            if isinstance(idx_gathered, RaggedValue):
                routing = idx_gathered.flat
            else:
                routing = idx_gathered
                self._read_routing[ref] = routing.detach()
        stack = self._read_stack(read, site)
        if not stack.is_identity:
            # a read whose executor keeps no gradients is detached below, so
            # its featurize builds no graph — and shares the no-grad
            # evaluation the write hooks of the same pass use (featurizer_cache);
            # a grad-enabled executor's read inherits the ambient mode rather
            # than forcing grad on as the write hooks do: a read never trains
            with contextlib.nullcontext() if self.grad_enabled else torch.no_grad():
                value, _errs = stack.featurize(value, routing=routing)
        if isinstance(read.dims, tuple):
            dims = torch.tensor(list(read.dims), dtype=torch.long, device=value.device)
            value = value.index_select(-1, dims)
        if not self.grad_enabled:
            value = value.detach()
            if to_cpu:
                value = value.cpu()
        if ragged:
            assert isinstance(gathered, RaggedValue)
            return RaggedValue(flat=value, widths=gathered.widths)
        return value

    def _expert_selected(
        self,
        ref: BoundRead,
        read: ReadSpec,
        site: ResolvedSite,
        raw: torch.Tensor,
        expert_idx: torch.Tensor | None,
        per_row: list[list[int]],
    ) -> RaggedValue:
        """The ragged face of the routed interior: the (position, slot) pairs
        the router sent to ``site.expert``, as flat ``(selected, d)`` rows plus
        per-example widths.

        An expert no token chose returns width-0 rows — **a data fact, not an
        error** (there is no per-expert hook to have not fired; the router
        simply sent it nothing at these positions). When that is true of
        *every* addressed position, the read is recorded as an
        ``unavailable`` cell with reason ``empty_selector`` (spec §4.1): the
        document was legal, the selector selected nothing here, and the cell
        belongs in the result and in the denominator rather than in a
        refusal. Partial emptiness stays data — the per-row widths say which
        rows the expert served.

        ``featurizer`` and ``dims`` are refused here rather than resized: the
        document sized them against the token-major form (``top_k · d``), and
        these rows are ``d``-wide — silently applying either would index a
        different space than the author named.
        """
        rname = ref.read
        if expert_idx is None:
            raise ProtocolError(
                "P2",
                f"read {rname!r} selects expert {site.expert}, but the engine "
                "captured no routing table alongside the tap — an executor bug, "
                "not a document error",
            )
        if read.featurizer is not None:
            raise ProtocolError(
                "P4",
                f"read {rname!r} featurizes the 'expert: {site.expert}' face of "
                f"{site.component!r}, whose rows are d_expert-wide while the "
                "component (and any featurizer sized against it) is top_k·d "
                "wide. Featurize the token-major form — drop 'expert' — or read "
                "this face raw.",
            )
        if isinstance(read.dims, tuple):
            raise ProtocolError(
                "P4",
                f"read {rname!r} slices 'dims' on the 'expert: {site.expert}' "
                f"face of {site.component!r}: 'dims' indexes the token-major "
                "top_k·d axis, and these rows are d-wide. Drop 'expert' or "
                "drop 'dims'.",
            )
        gathered = self._gather(raw, per_row, f"read {rname!r}")
        idx_gathered = self._gather(expert_idx, per_row, f"read {rname!r}")
        if isinstance(gathered, RaggedValue):
            assert isinstance(idx_gathered, RaggedValue)
            flat_value, pos_widths = gathered.flat, gathered.widths
            flat_idx = idx_gathered.flat
        else:
            assert isinstance(idx_gathered, torch.Tensor)
            rows, n_pos = gathered.shape[0], gathered.shape[1]
            flat_value = gathered.reshape(rows * n_pos, gathered.shape[-1])
            flat_idx = idx_gathered.reshape(rows * n_pos, idx_gathered.shape[-1])
            pos_widths = (n_pos,) * rows
        top_k = flat_idx.shape[-1]
        per_slot = flat_value.shape[-1] // top_k
        mask = flat_idx == site.expert  # (positions, top_k)
        selected = flat_value.reshape(-1, top_k, per_slot)[mask]
        # hits per (example, position) row — read to the host once, then
        # summed per example there rather than one device read per row
        counts = mask.sum(dim=-1).tolist()
        widths: list[int] = []
        offset = 0
        for width in pos_widths:
            widths.append(sum(counts[offset : offset + width]))
            offset += width
        if sum(widths) == 0:
            self._unavailable[ref] = unavailable(
                "empty_selector",
                f"read {rname!r} selects the 'expert: {site.expert}' face of "
                f"{site.component!r} at layer {site.layer}, and the router sent "
                f"expert {site.expert} no token at the addressed positions "
                f"({len(pos_widths)} rows, {sum(pos_widths)} positions) — a "
                "fact of this batch's routing, not of the document",
                cell_key(rname, self.coords),
            )
        if not self.grad_enabled:
            selected = selected.detach().cpu()
        return RaggedValue(flat=selected, widths=tuple(widths))

    def _state_read(
        self,
        rname: str,
        read: ReadSpec,
        site: ResolvedSite,
        value: torch.Tensor,
        gathered: "torch.Tensor | RaggedValue",
    ) -> "torch.Tensor | RaggedValue":
        """The tail of a read whose trailing axes form a state matrix.

        The tensor keeps its native layout — ``(batch, steps, heads, d_k,
        d_v)`` after the position gather — because there is no feature vector
        to flatten to. ``head:`` selects on the head axis directly;
        ``featurizer`` and ``dims`` are refused off the declared axes (the same
        generated refusals the attention pattern gets, with the position gather
        kept, which is what distinguishes the two shapes).
        """
        what = f"{site.component!r} ({site.shape.describe()})"
        if read.featurizer is not None:
            raise ProtocolError(
                "P4",
                f"read {rname!r} featurizes {what}: {site.shape.refusal('it')}",
            )
        if isinstance(read.dims, tuple):
            raise ProtocolError(
                "P4",
                f"read {rname!r} slices 'dims' on {what}: that would select "
                "d_v columns of a matrix as though they were features.",
            )
        if site.head is not None:
            # dim 0 of a ragged flat is the gathered rows; dense keeps
            # (batch, steps) in front — the head axis is right after either way
            value = (
                value[:, site.head]
                if isinstance(gathered, RaggedValue)
                else value[:, :, site.head]
            )
        if not self.grad_enabled:
            value = value.detach().cpu()
        if isinstance(gathered, RaggedValue):
            return RaggedValue(flat=value, widths=gathered.widths)
        return value
