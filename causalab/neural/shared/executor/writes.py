"""Apply writes in protocol class order.

``WriteMathMixin`` resolves operands and addresses, checks policy, and
applies each write through featurize, mechanism, and inverse operations.
It handles expert-ID joins and per-step DeltaNet state writes through the
ragged landing helpers. ``whole_native_tensor`` identifies taps read or
written in their native shape; ``schema.operand_reads`` finds payload dependencies.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any, Callable, Sequence

import torch

from causalab.neural.shared.encoding import EncodedBatch
from causalab.neural.shared.executor.cache import tap_key
from causalab.neural.shared.executor.ragged import (
    RaggedLandingMixin,
    RowWindow,
    operand_width_error,
    ragged_operand_error,
    ragged_write_error,
)
from causalab.neural.shared.fires import FireTally
from causalab.neural.shared.gather import (
    dense_index,
    gather_positions,
    splice_features,
)
from causalab.neural.shared.mechanisms import (
    apply_absolute,
    apply_delta,
    apply_renormalize,
    is_additive,
)
from causalab.neural.shared.sites import ResolvedSite, resolve_site
from causalab.neural.shared.values import RaggedValue
from causalab.protocol.bundles import entry_selection, selector_slot
from causalab.protocol.registry import write_policy_refusal
from causalab.protocol.rules.errors import ProtocolError
from causalab.neural.shared.plan import bound_read
from causalab.protocol.schema import ALL_POSITIONS, ReadRef, ReadSpec, WriteSpec

if TYPE_CHECKING:
    from causalab.neural.shared.featurizers import FeaturizerStack, Stage


def whole_native_tensor(
    rname: str, read: "ReadSpec | WriteSpec", raw: torch.Tensor, site: ResolvedSite
) -> torch.Tensor:
    """A read of a tap with **no contract form**: the whole native tensor.

    The one such tap is the attention pattern, ``(batch, heads, query, key)``,
    and the reason this is a bypass rather than a gather is its second position
    axis: the gather (dim 0 batch, dim 1 position) would index the head axis
    with position indices, and ``dims`` would slice the key axis as though it
    were features. Both produce plausible numbers from the wrong tensor.

    All three refusals below are *generated* from
    [`FeatureShape`][causalab.protocol.registry.shapes.FeatureShape] — position addressing needs
    one position axis, a featurizer needs a feature space, ``dims`` needs a
    feature axis to index — so a later tap with the same problem is refused by
    declaring its axes rather than by adding a branch here. What remains — the
    whole tensor, at ``pos: "all"`` — is exactly what an interchange on
    attention needs, and what nnterp's own check exercises
    (``self[layer] = rnd``).
    """
    shape = site.shape
    what = f"{site.component!r} ({shape.describe()})"
    pos = read.pos
    whole = getattr(pos, "all", None) is True or pos == ALL_POSITIONS
    if not whole:
        axes = ", ".join(a.label for a in shape.position_axes)
        raise ProtocolError(
            "P4",
            f"read {rname!r} addresses positions on {what}, which has "
            f"{len(shape.position_axes)} position axes ({axes}) — a position "
            "index would be ambiguous between them. Read the whole tensor with "
            'pos: "all".',
        )
    if read.featurizer is not None:
        raise ProtocolError(
            "P4",
            f"read {rname!r} featurizes {what}: {shape.refusal('it')} A "
            "featurizer would be fitted across an axis that is not a basis.",
        )
    if isinstance(read.dims, tuple):
        raise ProtocolError(
            "P4",
            f"read {rname!r} slices 'dims' on {what}: that would select "
            f"{shape.axes[-1].label} entries as though they were features.",
        )
    return raw


def _on_device(
    lookup: "Callable[[Any], torch.Tensor | float]", device: torch.device
) -> "Callable[[Any], torch.Tensor | float]":
    """Operand resolution whose tensor results live on ``device`` — the
    written tensor's (`WriteMathMixin._written_value`). A literal passes
    through; a tensor already there is not copied."""

    def resolve(value: Any) -> "torch.Tensor | float":
        operand = lookup(value)
        if isinstance(operand, torch.Tensor) and operand.device != device:
            return operand.to(device)
        return operand

    return resolve


def _renormalizes(entry: tuple[str, WriteSpec, ResolvedSite]) -> bool:
    """Whether a landing entry is a ``renormalize`` write, the one mechanism
    whose result depends on the address's pre-write value (§2.8)."""
    return str(entry[1].do.mechanism) == "renormalize"


class WriteMathMixin(RaggedLandingMixin):
    """The write math (§2.8): featurize → class-ordered ``do`` → inverse at
    one address, with operands resolved against this executor's reads,
    featurizer slots and ``params`` entries.

    Composed into [`ExecutorBase`][];
    the host surface the methods read — beyond ``doc`` and ``bundle``, which
    [`RaggedLandingMixin`][] declares — is declared below for the type
    checker and provided by the executor."""

    if TYPE_CHECKING:
        coords: dict[str, Any]
        load_tensors: Callable[[str], Any]
        _read_values: dict[ReadRef, torch.Tensor | RaggedValue]
        _read_routing: dict[ReadRef, torch.Tensor]
        _routing_mismatch: dict[tuple[str, int, int], tuple[int, int]]
        _routing_mismatch_pending: dict[
            tuple[str, int, tuple[int, ...]], tuple[torch.Tensor, int]
        ]

        def _positions(
            self,
            pos: Any,
            batch: EncodedBatch,
            input_role: str,
            *,
            cell: str | None = None,
        ) -> list[list[int]]: ...

        def _read_stack(
            self, read: ReadSpec | WriteSpec, site: ResolvedSite
        ) -> FeaturizerStack: ...

        def stage(self, name: str) -> Stage: ...

    def _state_step_writer(
        self,
        entries: list[tuple[str, WriteSpec, ResolvedSite]],
        input_role: str,
        batch: EncodedBatch,
        rows: RowWindow,
        tally: FireTally | None = None,
    ) -> Callable[[int, torch.Tensor], torch.Tensor]:
        """Per-step application of ``delta_state`` writes, for the stepwise
        substitution.

        A state edit must feed forward — step ``t``'s replacement is what step
        ``t+1`` decays and writes into — so the shared whole-tensor write math
        cannot land it. Instead each addressed step applies the same
        class-ordered mechanisms to that step's matrix, flattened to one
        row: ``v_pre`` is ``S_t`` as ``(1, 1, heads·d_k·d_v)``, and a tensor
        operand (a ``delta_state`` read) is sliced to the same (row, step)
        before the mechanism sees it, so ``swap`` interchanges step-for-step.

        ``dims`` is refused (a matrix has no feature columns); ``featurizer``
        refuses through the width lookup, as every state read does.

        ``rows`` is the window this forward covers: ``state`` holds its rows
        only, so the window offsets them back to role rows when a tensor
        operand is sliced.

        ``tally`` is this forward's fire count (§4 "Fires"): a state write
        declares one firing per distinct step its rows address and fires
        once per step it edits, so a kernel path that never reached the
        substitution — or skipped a step — is refused after the forward.
        """

        def class_rank(entry: tuple[str, WriteSpec, ResolvedSite]) -> int:
            do = entry[1].do
            if str(do.mechanism) == "renormalize":
                return 2
            return 1 if is_additive(do) else 0

        prepared: list[tuple[str, WriteSpec, ResolvedSite, list[list[int]]]] = []
        for ename, write, site in sorted(entries, key=class_rank):
            if isinstance(write.dims, tuple):
                raise ProtocolError(
                    "P4",
                    f"write {ename!r} slices 'dims' on {site.component!r} "
                    f"({site.shape.describe()}): that would select d_v columns "
                    "of a matrix as though they were features.",
                )
            per_row = self._positions(write.pos, batch, input_role)[rows.slice]
            prepared.append((ename, write, site, per_row))
            if tally is not None:
                tally.declare((ename,), len({p for row in per_row for p in row}))

        def edit_state(step: int, state: torch.Tensor) -> torch.Tensor:
            edited = state
            for ename, write, site, per_row in prepared:
                if tally is not None and any(step in row for row in per_row):
                    tally.fired((ename,), step=step)
                for row, positions in enumerate(per_row):
                    if step not in positions:
                        continue
                    j = positions.index(step)
                    if edited is state:
                        edited = state.clone()
                    v_pre = edited[row : row + 1].reshape(1, 1, -1)
                    v_new = self._written_value(
                        ename,
                        write,
                        site,
                        v_pre,
                        lookup=self._state_operand(
                            ename, rows.start + row, j, len(positions), v_pre
                        ),
                        # `state` is the step's matrix before any edit, and
                        # what a renormalize restores the norm of (§2.8)
                        v_ref=state[row : row + 1].reshape(1, 1, -1),
                    )
                    edited[row] = v_new.reshape(edited.shape[1:]).to(edited.dtype)
            return edited

        return edit_state

    def _state_operand(
        self, ename: str, row: int, step_index: int, n_steps: int, v_pre: torch.Tensor
    ) -> Callable[[Any], "torch.Tensor | float"]:
        """Operand lookup for one (row, addressed-step) state application: a
        tensor operand must be a state read — ``(batch, steps, heads, d_k,
        d_v)`` — covering **exactly the write's addressed steps** (the standard
        write path's elementwise rule, stated rather than broadcast), and is
        sliced to this row and step so the mechanism math sees two aligned
        single-step rows."""

        def lookup(value: Any) -> "torch.Tensor | float":
            operand = self._operand_lookup(value)
            if not isinstance(operand, torch.Tensor):
                return operand
            if operand.dim() != 5:
                raise ProtocolError(
                    "P2",
                    f"write {ename!r} hands {value!r} to a 'delta_state' "
                    f"write, but its shape is {tuple(operand.shape)} — a state "
                    "operand is a 'delta_state' read, (batch, steps, heads, "
                    "d_k, d_v), applied step for step",
                )
            if operand.shape[1] != n_steps:
                raise ProtocolError(
                    "P2",
                    f"write {ename!r}: operand {value!r} covers "
                    f"{operand.shape[1]} steps, but the write addresses "
                    f"{n_steps} — the j-th operand step lands on the j-th "
                    "addressed step, so both sides must cover the same steps "
                    "(read the operand at the write's own positions)",
                )
            sliced = operand[row : row + 1, step_index : step_index + 1]
            return sliced.reshape(1, 1, -1).to(v_pre.device)

        return lookup

    # ------------------------------------------------------------------ #
    # writes (the math; landing them is the engine's job)
    # ------------------------------------------------------------------ #

    def _operand_lookup(
        self,
        value: Any,
        *,
        rows: RowWindow | None = None,
        ragged: Sequence[int] | None = None,
    ) -> torch.Tensor | float:
        """Resolve one operand: a literal, a read's value, a featurizer slot,
        or a ``params`` entry.

        Only a **read** is row-indexed (dim 0 is the example), so it alone is
        sliced to ``rows`` — the window (or width bucket) of the forward that
        consumes it. A featurizer slot or a params tensor is one value for
        every row and a scalar has no rows; both pass through whole.

        ``ragged`` is the consuming write's per-row position widths over
        ``rows`` when that write lands under a ``ragged`` policy (§5 rule 19).
        A [`RaggedValue`][] operand is then re-nested row by row to those
        widths — refused when any row's widths disagree
        ([`operand_width_error`][]) — and padded to the widest with zeros,
        which the landing never writes back; a **dense** read (one width for
        every row) is held to the same rule (`_check_dense_operand`):
        its width must be each row's own, or one — a uniform operand as wide
        as the widest row would otherwise satisfy the padded frame's broadcast
        and land truncated into every narrower row. Without ``ragged`` (a
        write under ``refuse``, the absent-field behaviour), a ragged operand
        is rule 19's refusal as before ([`ragged_operand_error`][])."""
        if isinstance(value, ReadRef) or (
            isinstance(value, str) and value in self.doc.reads
        ):
            ref = bound_read(self.doc, value) if isinstance(value, str) else value
            value = ref.read
            stored = self._read_values[ref]
            if isinstance(stored, RaggedValue):
                if ragged is None:
                    raise ragged_operand_error(value)
                return self._nest_ragged_operand(value, stored, rows, ragged)
            # left where it is stored (the CPU, or the device its block
            # produced it on under grad); `_written_value` moves every
            # tensor operand to the written tensor's device
            operand = stored
            if rows is not None and not rows.whole:
                operand = operand[rows.index]
            if ragged is not None:
                self._check_dense_operand(value, operand, rows, ragged)
            return operand
        if "." in value:
            fname, slot = value.split(".", 1)
            if fname in self.doc.featurizers:
                params = self.stage(fname).slot_params()
                if slot in params:
                    return params[slot]
        if value in self.doc.params:
            spec = self.doc.params[value]
            if isinstance(spec.file_path, str):
                want, implicit = entry_selection(spec.entry, self.coords, value)
                slot = selector_slot(spec.entry, "value")
                what = f"params entry {value!r} ({spec.file_path})"
                point = self.load_tensors(spec.file_path).point(
                    slot, want, what=what, implicit=implicit
                )
                return point.tensor(slot)
            raise NotImplementedError(
                f"trainable free params ({value!r}) arrive with the train loop"
            )
        raise ProtocolError("P2", f"operand {value!r} did not resolve at run time")

    def _operand_routing(
        self, value: Any, rows: RowWindow | None = None
    ) -> torch.Tensor | None:
        """The routing table a tensor operand was read beside — ``None`` when
        the operand is not a read at a routed-interior site (a literal, a
        params tensor, a read anywhere else) and so carries no expert ids."""
        if isinstance(value, ReadRef):
            ref = value
        elif isinstance(value, str) and value in self.doc.reads:
            ref = bound_read(self.doc, value)
        else:
            return None
        if ref not in self._read_routing:
            return None
        routing = self._read_routing[ref]
        # sliced to the consuming forward's window — or width bucket — like
        # the operand itself (§8 microbatching): the two are joined row by row;
        # the consumer moves it beside the tensor it joins
        return routing if rows is None or rows.whole else routing[rows.index]

    def _check_dense_operand(
        self,
        value: str,
        operand: torch.Tensor,
        rows: RowWindow | None,
        widths: Sequence[int],
    ) -> None:
        """A dense read as a write operand under a landing policy (§5 rule
        19): ``operand`` is ``(rows, width, …)`` already sliced to ``rows``,
        and its one ``width`` must be every row's own landed width in
        ``widths``, or one (a single position broadcasts over a row, as it
        always has). Any other row is [`operand_width_error`][], naming the
        row and both widths — the refusal a [`RaggedValue`][] operand's
        disagreeing row gets, so the two landing policies refuse one document
        identically. Only a positioned read is held to it: a whole-tensor
        read (a tap with no contract form) or a state read has no position
        axis at dim 1, and pairs into a positioned write by broadcast alone."""
        if operand.dim() < 2:
            return
        site = resolve_site(
            self.bundle, self.doc.sites[str(self.doc.reads[value].site)]
        )
        if not site.shape.has_contract_form or site.shape.state_axes:
            return
        got = int(operand.shape[1])
        if got == 1:
            return
        examples = list(range(len(widths))) if rows is None else rows.examples
        mismatches = [
            (row, got, int(width))
            for row, width in zip(examples, widths)
            if got != int(width)
        ]
        if mismatches:
            raise operand_width_error(value, mismatches)

    def _resolve_write_addresses(
        self, write_names: tuple[str, ...]
    ) -> dict[Any, tuple[ResolvedSite, list[tuple[str, WriteSpec, ResolvedSite]]]]:
        """Resolve and policy-check this group's writes, grouped by address.

        Addresses are keyed by [`tap_key`][], so two components that share a
        module but mean different tensors (a different tuple element, or a
        different shape) get their own application rather than one
        overwriting the other's view."""
        by_address: dict[
            Any, tuple[ResolvedSite, list[tuple[str, WriteSpec, ResolvedSite]]]
        ] = {}
        for ename in write_names:
            write = self.doc.writes[ename]
            site = resolve_site(self.bundle, self.doc.sites[str(write.site)])
            # The write policy is the capability row's (read-only, or a closed
            # mechanism set) — the same function `validate` applies at load,
            # here for a document that arrived unvalidated. One check, one
            # text: the read-only, routing-table and attention-pattern
            # refusals used to be three tables at two sites.
            refusal = write_policy_refusal(
                ename, site.component, str(write.do.mechanism)
            )
            if refusal is not None:
                raise ProtocolError("P4", refusal, reason="unsupported_mechanism")
            key = tap_key(site)
            if key not in by_address:
                by_address[key] = (site, [])
            by_address[key][1].append((ename, write, site))
        return by_address

    def _apply_writes_to_contract(
        self,
        entries: list[tuple[str, WriteSpec, ResolvedSite]],
        input_role: str,
        batch: EncodedBatch,
        tensor: torch.Tensor,
        *,
        per_row: list[list[int]] | None = None,
        rows: RowWindow | None = None,
        routing: torch.Tensor | None = None,
    ) -> None:
        """Apply every write at one address, in class order, mutating the
        contract-shaped ``tensor`` in place — absolute first, additive deltas
        summed, renormalize last against the pre-write norm (§2.8).

        ``per_row`` overrides position resolution, the same override
        `_finalize_read` takes: a caller whose position axis is not the
        token axis (a per-chunk state) has already worked the indices out.

        ``rows`` is the window of the role's rows ``tensor`` holds — the
        microbatch. Positions resolve against the whole padded frame (a
        window is a row slice of it, so the indices coincide), then both they
        and any tensor operand are sliced to the window; ``None`` is the whole
        batch.

        ``routing`` is the routing table of a routed-interior address,
        ``(batch, position, top_k)`` over the same rows as ``tensor`` — the
        expert ids an expert-keyed gate keys its parameters by, gathered at
        the write's positions alongside the value."""
        if rows is None:
            rows = RowWindow(0, tensor.shape[0], tensor.shape[0])
        lookup = functools.partial(self._operand_lookup, rows=rows)

        def class_rank(entry: tuple[str, WriteSpec, ResolvedSite]) -> int:
            do = entry[1].do
            if str(do.mechanism) == "renormalize":
                return 2  # after the deltas — the only order where it acts (§2.8 note)
            return 1 if is_additive(do) else 0  # absolute first, then additive

        ordered = sorted(entries, key=class_rank)
        # a renormalize rescales to the norm the address had before any of
        # its writes (§2.8), and the loop below re-gathers the running value
        # between writes, so the pre-write value is kept here. Alone,
        # a renormalize's running value is its pre-write value: no copy
        pre = (
            tensor.clone()
            if len(ordered) > 1 and any(_renormalizes(entry) for entry in ordered)
            else None
        )
        for ename, write, site in ordered:
            reference = (
                pre if pre is not None and _renormalizes((ename, write, site)) else None
            )
            if not site.shape.has_contract_form:
                # Symmetric with the read (see whole_native_tensor): this
                # tensor's feature axis is a position axis, so the position
                # gather below would index heads with positions and `dims`
                # would slice key positions as features. Both are refused
                # there; what is left is the whole tensor, edited whole.
                whole_native_tensor(ename, write, tensor, site)
                mechanism = str(write.do.mechanism)
                if mechanism == "swap":
                    replacement = lookup(write.do.payload)
                    if not isinstance(replacement, torch.Tensor):
                        raise ProtocolError(
                            "P2",
                            f"write {ename!r} swaps {site.component!r} with "
                            "a scalar; a whole-tensor interchange needs a "
                            "tensor operand read from elsewhere",
                        )
                    if replacement.shape != tensor.shape:
                        raise ProtocolError(
                            "P2",
                            f"write {ename!r} replaces the whole "
                            f"{site.component!r} tensor, but its operand has "
                            f"shape {tuple(replacement.shape)} and the tap is "
                            f"{tuple(tensor.shape)} — an interchange needs "
                            "both inputs to have the same number of positions",
                        )
                    tensor.copy_(replacement.to(tensor.dtype))
                elif mechanism == "gaussian":
                    # 📐 The noise is drawn as (batch, position, feature) and
                    # its `axis` names the feature axis' tensor-parallel
                    # semantics. This tap has no feature axis — its last axis
                    # is key positions — so there is nothing for either to
                    # mean, and the draw does not even fit (measured: "shape
                    # '[1, 8, 5, 5]' is invalid for input of size 40").
                    # Refused by name rather than reshaped into something that
                    # would run.
                    raise ProtocolError(
                        "P4",
                        f"write {ename!r} applies 'gaussian' to "
                        f"{site.component!r}, whose shape is "
                        f"{site.shape.describe()}: the noise is drawn per "
                        "(batch, position, feature) and its 'axis' names how "
                        "the feature axis is sharded, and this tap has no "
                        "feature axis at all. Swap in a noise tensor of the "
                        "tap's own shape instead.",
                    )
                else:
                    # 📐 Arithmetic on the whole tensor, with no gather: for
                    # `attention_scores` this is the point of the component.
                    # `_written_value` broadcasts a scalar operand over any
                    # rank, and `dims` and featurizers are already refused
                    # above, so there is no feature axis for it to mis-slice.
                    tensor.copy_(
                        self._written_value(
                            ename,
                            write,
                            site,
                            tensor,
                            lookup=lookup,
                            rows=rows,
                            v_ref=reference,
                        ).to(tensor.dtype)
                    )
                continue
            if per_row is not None:
                positions = per_row
                pad_to = max((len(row) for row in positions), default=0)
            else:
                # resolved against the whole padded frame, then sliced to the
                # window; the widest row of the *whole* batch is what a masked
                # landing pads to, so the `gaussian` draw a row receives does
                # not depend on how the batch was cut (§8)
                every = self._positions(write.pos, batch, input_role)
                positions = every[rows.slice]
                pad_to = max((len(row) for row in every), default=0)
            widths = {len(row) for row in positions}
            policy = write.ragged or "refuse"
            if len(widths) != 1:
                if policy == "refuse":
                    raise ragged_write_error(ename, sorted(widths))
                self._land_ragged(
                    ename,
                    write,
                    site,
                    tensor,
                    positions,
                    policy=policy,
                    pad_to=pad_to,
                    rows=rows,
                    routing=routing,
                    reference=reference,
                )
                continue
            (width,) = widths
            # this write's lookup alone: the ragged binding below must not
            # leak into a later write of the same landing call, whose operands
            # would then be held to *this* write's width (or, under `refuse`,
            # refused with the width message instead of the operand one)
            write_lookup = lookup
            if policy != "refuse":
                # uniform on this window (or on the whole batch): the dense
                # landing below, with a ragged operand welcome at this width
                write_lookup = functools.partial(
                    self._operand_lookup, rows=rows, ragged=[width] * len(positions)
                )
            index = dense_index(positions, tensor.device)
            fslice = site.feature_slice or slice(None)
            # one gather of the landed positions, sort-free backward when the
            # table repeats no element (gather.py); the write-back splices
            # the new features into it rather than gathering again
            landed = gather_positions(tensor, index)
            v_new = self._written_value(
                ename,
                write,
                site,
                landed[..., fslice],
                lookup=write_lookup,
                rows=rows,
                routing=None if routing is None else routing[index.pair],
                v_ref=None
                if reference is None
                else gather_positions(reference, index)[..., fslice],
            )
            tensor[index.pair] = splice_features(landed, fslice, v_new.to(tensor.dtype))

    def _written_value(
        self,
        ename: str,
        write: WriteSpec,
        site: ResolvedSite,
        v_pre: torch.Tensor,
        *,
        lookup: "Callable[[Any], torch.Tensor | float] | None" = None,
        rows: RowWindow | None = None,
        routing: torch.Tensor | None = None,
        v_ref: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """featurize → class-ordered do → inverse, honoring dims and the
        error-term contract.

        ``v_ref`` is the address's value before any of its writes landed, at
        ``v_pre``'s rows and positions: what a ``renormalize`` restores the
        norm of (§2.8, ``f₀``). ``None`` means ``v_pre`` is that value, which
        holds when no earlier write landed at the address.

        ``lookup`` overrides operand resolution — the state-write path slices
        tensor operands to one (row, step) so the same mechanism math applies
        per step; everything else uses `_operand_lookup` unchanged.

        ``rows`` is the window ``v_pre`` covers, which only a ``gaussian``
        write needs: its draw is made over the whole batch and sliced, so the
        noise a row receives does not depend on how the batch was cut.

        ``routing`` is the routing table at ``v_pre``'s rows and positions,
        ``(batch, position, top_k)``, when the address is the routed interior.
        A write through an expert-keyed gate featurizes with it and joins
        every tensor operand to it by expert id (`_align_by_expert`), so
        a slot receives the operand's value for the *same expert* and a slot
        whose expert the operand never activated is left unchanged.

        Every tensor operand is moved to ``v_pre``'s device — the written
        tensor's, which is its block's (the loader's crossings put it there):
        a read value stored on the CPU, or captured on another block's device
        under grad, meets the write where it lands.
        """
        lookup = _on_device(
            lookup if lookup is not None else self._operand_lookup, v_pre.device
        )
        stack = self._read_stack(write, site)
        f0, errs = stack.featurize(v_pre, routing=routing)
        dims = None
        if isinstance(write.dims, tuple):
            if stack.needs_routing:
                raise ProtocolError(
                    "P4",
                    f"write {ename!r} slices 'dims' through an expert-keyed gate: "
                    "the token-major axis is joined to experts per token, so a "
                    "fixed coordinate subset names different neurons on "
                    "different tokens. Select neurons with the gate instead.",
                )
            dims = torch.tensor(list(write.dims), dtype=torch.long, device=f0.device)

        def select(f: torch.Tensor) -> torch.Tensor:
            return f if dims is None else f.index_select(-1, dims)

        def aligned(fill: torch.Tensor) -> "Callable[[Any], torch.Tensor | float]":
            """Operand resolution that joins a tensor operand's slots to
            ``v_pre``'s by expert; ``fill`` is what a slot with no source
            receives, chosen so the mechanism leaves it unchanged."""
            assert lookup is not None and routing is not None

            def resolve(value: Any) -> torch.Tensor | float:
                operand = lookup(value)
                if not isinstance(operand, torch.Tensor):
                    return operand
                return self._align_by_expert(
                    ename, value, operand, routing, fill, layer=site.layer, rows=rows
                )

            return resolve

        # `f` is written into in place only where a `dims` slice lands in it
        # (the three `index_copy_` below), and then into its own copy: `f0` is
        # the featurizer's output, which its error term saved for backward (a
        # subspace's `x - f @ Qᵀ`), and through the identity stack the gathered
        # slice itself. A whole-axis result is a fresh tensor already (the sum,
        # the broadcast operand, the renormalized product), and every consumer
        # copies it into the model's tensor rather than mutating it — so no
        # defensive clone there
        f = f0 if dims is None else f0.clone()
        do = write.do
        batch_size, n_pos = v_pre.shape[0], v_pre.shape[1]
        if str(do.mechanism) == "renormalize":
            pass  # applied last, below
        elif is_additive(do):
            delta = apply_delta(
                do,
                select(f0),
                aligned(torch.zeros_like(f0)) if stack.needs_routing else lookup,
                batch=batch_size if rows is None else rows.total,
                n_pos=n_pos,
                rows=None if rows is None else rows.index,
            )
            if dims is None:
                f = f0 + delta
            else:
                f.index_copy_(-1, dims, select(f0) + delta)
        else:
            written = apply_absolute(
                do,
                select(f0),
                aligned(f0) if stack.needs_routing else lookup,
                code=self.doc.code,
            )
            written = written.broadcast_to(select(f0).shape).to(f0.dtype)
            if dims is None:
                f = written
            else:
                f.index_copy_(-1, dims, written)
        if str(do.mechanism) == "renormalize":
            f_ref = f0 if v_ref is None else stack.featurize(v_ref, routing=routing)[0]
            if dims is None:
                f = apply_renormalize(f, f_ref)
            else:
                f.index_copy_(-1, dims, apply_renormalize(select(f), select(f_ref)))
        return stack.inverse(f, errs)

    @property
    def routing_mismatch(self) -> dict[tuple[str, int, int], tuple[int, int]]:
        """Per (write, layer, example), how many of the example's written
        slots found no source slot holding their expert, of how many slots —
        what ``routing_mismatch.json`` records (§2.5 ``expert_neuron``). The
        writes leave their counts on the device (`_align_by_expert`);
        reading this brings every pending count over in one host read."""
        pending, self._routing_mismatch_pending = self._routing_mismatch_pending, {}
        if pending:
            device = next(iter(pending.values()))[0].device
            counts = iter(
                torch.cat(
                    [missing.to(device).reshape(-1) for missing, _ in pending.values()]
                ).tolist()
            )
            for (ename, layer, examples), (_missing, per_example) in pending.items():
                for example in examples:
                    self._routing_mismatch[(ename, layer, example)] = (
                        int(next(counts)),
                        per_example,
                    )
        return self._routing_mismatch

    def _align_by_expert(
        self,
        ename: str,
        operand_name: Any,
        operand: torch.Tensor,
        routing: torch.Tensor,
        fill: torch.Tensor,
        *,
        layer: int | None,
        rows: RowWindow | None = None,
    ) -> torch.Tensor:
        """Join a tensor operand's routed slots to the written slots by expert
        id (§2.5 ``expert_neuron``).

        Slot *k* of a token holds its *k*-th ranked expert, so the same slot
        on the operand's side may hold a different expert. For every written
        slot holding expert ``e``, the source is the operand's slot holding
        ``e`` at the same row and position when ``e`` is active there, and
        ``fill`` otherwise — the pre-write feature value for an absolute
        write, zero for an additive one, so a slot with no source keeps its
        base value. The count of such slots per example is recorded in
        [`routing_mismatch`][], keyed by write, layer and example.

        The operand must have been read at a routed-interior site (it carries
        expert ids) over the same rows and positions as the write: a
        broadcast operand has no slot-to-expert map to join on, and is
        refused rather than landed slot for slot.
        """
        source_routing = self._operand_routing(operand_name, rows)
        if source_routing is None:
            raise ProtocolError(
                "P2",
                f"write {ename!r} hands {operand_name!r} to an expert-keyed gate, "
                "but that operand carries no routing table — the source of a "
                "write through group 'expert_neuron' is a read at the routed "
                "interior, whose expert ids say which of its slots matches which "
                "of the written ones",
            )
        source_routing = source_routing.to(fill.device)
        operand = operand.to(device=fill.device, dtype=fill.dtype)
        if operand.shape != fill.shape or source_routing.shape != routing.shape:
            raise ProtocolError(
                "P2",
                f"write {ename!r}: operand {operand_name!r} covers "
                f"{tuple(operand.shape)} with routing {tuple(source_routing.shape)}, "
                f"but the write addresses {tuple(fill.shape)} with routing "
                f"{tuple(routing.shape)} — slots are joined by expert per (example, "
                "position), so both sides must address the same positions",
            )
        top_k = routing.shape[-1]
        per_slot = operand.shape[-1] // top_k
        # (…, written slot, operand slot): does the operand's slot hold the
        # written slot's expert? An expert appears at most once per token, so
        # at most one operand slot matches
        match = routing.unsqueeze(-1) == source_routing.unsqueeze(-2)
        found = match.any(-1)
        source_slot = match.to(torch.int8).argmax(-1)
        slots = operand.reshape(*operand.shape[:-1], top_k, per_slot)
        picked = slots.gather(
            -2, source_slot.unsqueeze(-1).expand(*source_slot.shape, per_slot)
        )
        aligned = torch.where(found.unsqueeze(-1), picked, fill.reshape(slots.shape))
        assert layer is not None  # the routed interior is a layered component
        missing = (~found).reshape(found.shape[0], -1).sum(-1)
        per_example = found[0].numel()
        examples = list(range(len(missing))) if rows is None else rows.examples
        # keyed by the role's rows, not the window's (nor the bucket's), so a
        # microbatched layout names the same examples as a whole-batch one;
        # the counts stay on the device — a host read inside a layer hook
        # would stall the launch stream once per layer, for a record only the
        # point's full-data pass is ever asked for. Re-inserted at the end so
        # a flush lands the calls in order (the last write of a row wins).
        key = (ename, layer, tuple(examples))
        self._routing_mismatch_pending.pop(key, None)
        self._routing_mismatch_pending[key] = (missing, per_example)
        return aligned.reshape(operand.shape)
