"""Execute a protocol point through nnsight traces.

TracePointExecutor inherits position resolution, featurizers, and write
math from ExecutorBase. Each group runs in one trace. Writes assign envoy
values before reads save them. Function interiors use the address table
and the same contract-layout conversions.

Operations follow model execution order. At one depth, writes precede
reads. A ``block_mid`` edit carries its residual delta to the declared
write-back target and applies it before an absolute write there.

A trace temporarily enables implementations required by its interior taps
and restores them afterward. Applied requirements enter execution metadata;
the authored document retains its declared backend identity.

Generation uses one ``model.generate`` trace. Prompt operations bind the
prefill occurrence; ``tracer.iter`` walks later decode occurrences. Each
generated interior needs an address for the decode path, which can use
different kernels from prefill.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, replace
from typing import Any, Iterator

import torch

from causalab.neural.engines.nnsight_tracing.addresses import (
    ADDRESSES,
    GENERATED_ADDRESSES,
    MOE_EXPERTS,
    AddressResolutionError,
    SourceAddress,
    match_op,
)
from causalab.neural.engines.nnsight_tracing.loading import torch_module
from causalab.neural.shared.kernels import torch_kernel_path
from causalab.neural.shared.encoding import (
    EncodedBatch,
    continuation_frame,
    resolve_steps,
)
from causalab.neural.shared.executor import (
    ExecutorBase,
    TapKey,
    refuse_unstackable,
    tap_key,
)
from causalab.neural.shared.layout import (
    from_contract,
    rebuild_payload,
    tap_tensor,
    to_contract,
)
from causalab.neural.shared.plan import group_reads, write_names
from causalab.neural.shared.sites import ResolvedSite, resolve_site
from causalab.protocol.rules.errors import ProtocolError
from causalab.protocol.positions.encoding import generated_budget
from causalab.protocol.registry import walk
from causalab.protocol.schema import ReadRef, SiteSpec, WriteSpec, operand_reads

__all__ = ["TracePointExecutor"]


@dataclass(frozen=True)
class ResolvedTap:
    """One tap as this engine lands it: the shared resolution (policy, layout,
    layer — [`ResolvedSite`][] is the cross-engine vocabulary and stays
    untouched) plus, for an interior component, its ``.source`` address."""

    site: ResolvedSite
    #: ``None`` = a module boundary (served by ``envoy.input``/``.output``).
    source: SourceAddress | None = None


def _writeback_name(tap: Any) -> str:
    """``block_mid at layer 11 -> block_output`` — a refusal about a deferred
    delta is useless without which site and which layer it belonged to."""
    site = tap.site
    target = site.writeback.component if site.writeback is not None else "?"
    return f"{site.component} at layer {site.layer} -> {target}"


class TracePointExecutor(ExecutorBase):
    """Execute one concrete document against one nnsight-loaded model."""

    def _run_group(self, model: str, input_role: str) -> None:
        if (model, input_role) in self._groups_run:
            return
        if self.grad_enabled:
            raise ProtocolError(
                "P4",
                "the nnsight engine does not run gradient-enabled groups — "
                "training documents route to an engine with the 'grad' "
                "capability",
            )
        declared = self.doc.intervened_models.get(model)
        if declared is not None and declared.writes_during_generation:
            # the trace binds each write's prefill occurrence only, so the
            # flag would be dropped without a word: routing refuses this
            # document (rule 13), and this is the backstop for a caller that
            # builds the executor directly
            raise ProtocolError(
                "P4",
                f"intervened model {model!r} declares writes_during_generation, "
                "and the nnsight engine lands every write on the prefill only — "
                "its 'generation_writes' capability is absent, so routing should "
                "not have sent it here; the reference engine serves it",
            )
        # operands first — the acyclic model graph is the schedule skeleton
        names = write_names(self.doc, model) or ()
        for ename in names:
            for ref in operand_reads(self.doc, self.doc.writes[ename].do):
                self.read_value(ref)

        batch = self._batch(input_role)
        taps = []
        gen_taps: list[tuple[str, Any]] = []
        depth = 0
        for ref in group_reads(self.doc, model, input_role):
            rname, read = ref.read, self.doc.reads[ref.read]
            budget = generated_budget(self.doc, read.pos)
            if budget is None:
                taps.append((rname, read))
            else:
                gen_taps.append((rname, read))
                depth = max(depth, budget)

        read_taps = {
            rname: self._wrap(resolve_site(self.bundle, self.doc.sites[str(read.site)]))
            for rname, read in taps
        }
        write_taps = {
            key: (self._wrap(site), entries)
            for key, (site, entries) in self._resolve_write_addresses(names).items()
        }
        # A continuation read at lm_head is served from kept ln_final
        # activations and projected at its addressed steps — the reference
        # engine's own trick, shared here so both serve the same value.
        gen_sites = {
            rname: resolve_site(self.bundle, self.doc.sites[str(read.site)])
            for rname, read in gen_taps
        }
        gen_wrapped: dict[str, ResolvedTap] = {}
        for rname, site in gen_sites.items():
            refuse_unstackable(rname, site)
            capture_site = (
                resolve_site(self.bundle, SiteSpec(component="ln_final"))
                if site.component == "lm_head"
                else site
            )
            gen_wrapped[rname] = self._wrap_generated(capture_site)

        group_taps = [
            *read_taps.values(),
            *(tap for tap, _ in write_taps.values()),
            *gen_wrapped.values(),
        ]
        required = frozenset(
            requirement
            for tap in group_taps
            if tap.source is not None
            for requirement in tap.source.requires
        )
        self.applied_requirements |= required

        # one trace per group, operations in forward-execution order; a write
        # lands before a read at the same depth so the read sees it, and the
        # `.source` interiors demand the order outright
        operations: list[tuple[int, tuple[int, int], str, ResolvedTap, list]] = []
        for tap, entries in write_taps.values():
            operations.append((0, tap.site.depth, "write", tap, entries))
            if tap.site.writeback is not None:
                # A write-back site is a module boundary (`block_mid` is the
                # norm's input), so it has no `.source` and cannot be per-fire.
                # That is what makes one deferred delta per key sound: a
                # per-fire write would take `_land_per_fire_writes`, return no
                # delta, and be misreported as an ordering failure.
                assert tap.source is None, tap.site.component
                # The input rewrite produces this delta at its own site, but
                # touching the target envoy's output there would advance the
                # trace past every same-layer operation between the two.
                # Issue a synthetic landing at the target's own depth instead.
                operations.append(
                    (
                        -1,
                        replace(tap.site, component=tap.site.writeback.component).depth,
                        "writeback",
                        tap,
                        [],
                    )
                )
        seen_keys: set = set()
        for tap in read_taps.values():
            key = tap_key(tap.site, tap.source)
            if key not in seen_keys:
                seen_keys.add(key)
                operations.append((1, tap.site.depth, "read", tap, []))
        operations.sort(key=lambda op: (op[1], op[0]))

        import nnsight

        batch_size = int(batch.input_ids.shape[0])
        saves: dict = {}
        #: navigated `.source` values, shared within one trace so two taps on
        #: one op (q and k are one rope call) request it once
        self._trace_sources: dict = {}
        from nnsight.intervention.interleaver import OutOfOrderError

        self._fire_checks: list[tuple[str, int, Any]] = []
        # An exception inside a trace body does not survive the tracer's exit
        # (measured: it is swallowed and the trace completes on the fires it
        # reached), so a fire-index miss detected at build time is carried out
        # by hand and raised after.
        fire_miss: list[tuple[ResolvedTap, OutOfOrderError]] = []
        writeback_errors: list[str] = []
        gen_sinks: dict = {}
        gen_result: Any = None
        inputs = {
            "input_ids": batch.input_ids,
            "attention_mask": batch.attention_mask,
        }

        def trace_body(tracer: Any) -> Any:
            # prompt-frame operations first: everything here binds occurrence
            # 0 of its location — the prefill — which is the whole of "writes
            # are prefill-only"
            deferred_writebacks: dict[TapKey, Any] = {}
            for _, _, op_kind, tap, entries in operations:
                per_fire = tap.source is not None and tap.source.fires != "once"
                try:
                    if op_kind == "writeback":
                        key = tap_key(tap.site, tap.source)
                        if key not in deferred_writebacks:
                            writeback_errors.append(
                                f"{_writeback_name(tap)}: the target was reached "
                                "before the input write that produces its delta"
                            )
                            continue
                        self._land_writeback(tap, deferred_writebacks.pop(key))
                    elif op_kind == "write" and per_fire:
                        self._land_per_fire_writes(
                            tracer, tap, entries, input_role, batch
                        )
                    elif op_kind == "write":
                        delta = self._land_writes(tap, entries, input_role, batch)
                        if delta is not None:
                            key = tap_key(tap.site, tap.source)
                            if key in deferred_writebacks:
                                writeback_errors.append(
                                    f"{_writeback_name(tap)}: the input fired twice "
                                    "before the target — a residual delta would be "
                                    "lost"
                                )
                            else:
                                deferred_writebacks[key] = delta
                    elif per_fire:
                        saves[tap_key(tap.site, tap.source)] = self._collect_per_fire(
                            tracer, tap
                        )
                    else:
                        saves[tap_key(tap.site, tap.source)] = nnsight.save(
                            self._tap_contract(tap, batch_size)
                        )
                except OutOfOrderError as error:
                    if not per_fire:
                        raise
                    fire_miss.append((tap, error))
                    break
            if deferred_writebacks and not fire_miss:
                writeback_errors.extend(
                    f"{_writeback_name(pending)}: the delta was never applied — "
                    "the target did not run"
                    for pending in (
                        tap
                        for _, _, op_kind, tap, _ in operations
                        if op_kind == "writeback"
                        and tap_key(tap.site, tap.source) in deferred_writebacks
                    )
                )
            if not depth:
                return None
            # decode steps: occurrence j of a per-forward location is the
            # step consuming generated token j-1, so fires 1..depth are
            # generated positions 0..depth-1
            gen_order = sorted(
                {
                    tap_key(tap.site, tap.source): tap for tap in gen_wrapped.values()
                }.items(),
                key=lambda item: item[1].site.depth,
            )
            for key, _tap in gen_order:
                gen_sinks[key] = nnsight.save([])
            # 📐 fire-counting probes show occurrences are counted
            # per location, and a decode-only op (the recurrent delta kernel)
            # has no prefill occurrence — so its occurrence j is step j+1, one
            # off. Anchoring each body on a location that fires every forward
            # makes the iteration index mean the forward; the ops after it
            # follow the model in order. Skipped when the first tap already is
            # such a location (every module boundary and interface slot is).
            needs_anchor = gen_order and (
                gen_order[0][1].source is not None
                and gen_order[0][1].source.fires != "once"
            )
            # the family's own tree names the embedding. 🐞 Was GPT-2's
            # ``transformer.wte`` or else llama's ``model.embed_tokens``, so
            # every generated read on a third tree (GPT-J) raised here
            embedding = walk(self.bundle.model, self.bundle.adapter.tree.embedding)
            for _step in tracer.iter[1 : depth + 1]:
                if needs_anchor:
                    _ = embedding.input
                for key, tap in gen_order:
                    gen_sinks[key].append(self._generated_step_value(tap, batch_size))
            # ⚠️ last: `tracer.result` consumes the whole run, so a request
            # after it is never reached (measured)
            return nnsight.save(tracer.result)

        with torch.no_grad():
            # a model off CUDA runs transformers' torch kernels whatever the
            # environment installed (shared/kernels.py)
            with (
                torch_kernel_path(torch_module(self.bundle.model)),
                self._switched_implementations(required),
            ):
                if depth:
                    # depth+1 forwards give every generated position its
                    # activations, the last token's included (the extra draw
                    # is never consumed — the reference engine's own decode
                    # loop, §2.3); eos_token_id=None keeps decoding past an
                    # eos exactly as that loop does, and widths truncate
                    # reads post-hoc. `generate` must be called in the with
                    # expression itself: it is @traceable, and a plain call
                    # just generates.
                    with self.bundle.model.generate(
                        inputs,
                        max_new_tokens=depth + 1,
                        do_sample=False,
                        eos_token_id=None,
                    ) as tracer:
                        gen_result = trace_body(tracer)
                else:
                    with self.bundle.model.trace(
                        inputs, position_ids=batch.position_ids()
                    ) as tracer:
                        trace_body(tracer)
        self._trace_sources = {}
        if fire_miss:
            tap, error = fire_miss[0]
            raise ProtocolError(
                "P2",
                f"a write or read addresses a fire of {tap.site.component!r} "
                "past the kernel's last — a tracer.iter body there never "
                f"runs, so it is refused rather than silently skipped ({error})",
            )
        if writeback_errors:
            # internal invariants, not a document error — but all of them, so a
            # group that mis-sequenced twice does not read as one failure
            raise RuntimeError("; ".join(writeback_errors))
        for component, wanted, count in self._fire_checks:
            if wanted >= int(count):
                raise ProtocolError(
                    "P2",
                    f"a write addresses fire {wanted} of {component!r}, but the "
                    f"kernel fired {int(count)} times on this batch — a "
                    "tracer.iter body past the last fire never runs, so this "
                    "is refused rather than silently skipped",
                )
        self._fire_checks = []

        for rname, read in taps:
            tap = read_taps[rname]
            raw = saves[tap_key(tap.site, tap.source)]
            ref = ReadRef(rname, model)
            if tap.source is not None and tap.source.fires != "once":
                self._read_values[ref] = self._finalize_per_fire_read(
                    ref, read, tap, raw, batch, input_role
                )
            else:
                self._read_values[ref] = self._finalize_read(
                    ref, read, tap.site, raw, batch, input_role
                )
        if depth:
            self._finalize_generated(
                model,
                input_role,
                batch=batch,
                depth=depth,
                gen_taps=gen_taps,
                gen_sites=gen_sites,
                gen_wrapped=gen_wrapped,
                gen_sinks=gen_sinks,
                gen_result=gen_result,
            )
        self._groups_run.add((model, input_role))

    # ------------------------------------------------------------------ #
    # interior addressing
    # ------------------------------------------------------------------ #

    def _wrap(self, site: ResolvedSite) -> ResolvedTap:
        """Pair the shared resolution with this engine's landing.

        An ``interface_slot`` marks a component with no module boundary (the
        four function-interior slots, the delta kernel's boundary — kind
        ``"delta"``, the reference engine's global swap, which this engine
        lands as a ``.source`` line of the same name — and the pattern, whose
        *write* has no boundary; this engine serves its read from the same
        op); those are looked up in the per-stream address table. ``kind="interior"`` marks
        a fused-forward interior keyed by component, and ``kind="experts"``
        is the routed-expert interior's shared resolution (the dispatch
        wrapper on the reference engine) — this engine lands the same
        components through the ``MOE_EXPERTS`` addresses. Everything else is
        the envoy path unchanged.
        """
        if site.kind in ("interior", "experts"):
            if site.expert is not None:
                # the ragged `expert:` face is served by the reference
                # engine's dispatch wrapper; nothing captures the routing
                # table alongside a `.source` read here yet
                raise ProtocolError(
                    "P4",
                    f"site names expert {site.expert!r} on "
                    f"{site.component!r}, and the nnsight engine does not "
                    "serve the ragged 'expert:' face — read the token-major "
                    "form here, or route to the reference engine.",
                    reason="component_unavailable",
                )
            # the expert interior lives under mlp; the DeltaNet interior on
            # the mixer, keyed by its stream — one lookup covers both
            address = MOE_EXPERTS.get(site.component) or ADDRESSES.get(
                self.bundle.stream_at(site.layer), {}
            ).get(site.component)
            where = "the interior tables"
        elif site.interface_slot is not None:
            stream = self.bundle.stream_at(site.layer)
            address = ADDRESSES.get(stream, {}).get(site.component)
            where = f"the {stream!r} table"
        else:
            return ResolvedTap(site=site)
        if address is None:
            raise ProtocolError(
                "P4",
                f"component {site.component!r} has no interior address in "
                f"{where} of the nnsight engine "
                "(neural/engines/nnsight_tracing/addresses.py) — extend the "
                "table, or route to the reference engine.",
                reason="component_unavailable",
            )
        return ResolvedTap(site=site, source=address)

    def _source_value(self, tap: ResolvedTap) -> tuple[Any, Any]:
        """The traced tensor an interior address names, navigated in-trace —
        plus, for an ``align`` address, the kernel's sorted→token permutation.

        Memoized per trace by the address' op identity (before
        ``tuple_index``), so two components on one op — q and k are the two
        elements of one rope call — request it once; ``.source`` ops refuse a
        second request after the model has run past them. The permutation is
        requested *first*, matching op order (the sort fires before anything
        it sorted for).
        """
        address = tap.source
        assert address is not None
        site = tap.site
        key = (
            id(site.module),
            address.op_pattern,
            address.peel,
            address.field,
            address.arg,
        )
        if key not in self._trace_sources:
            self._trace_sources[key] = self._navigate_value(
                tap, perm_cache=self._trace_sources
            )
        value, perm = self._trace_sources[key]
        if address.tuple_index is not None:
            value = value[address.tuple_index]
        return value, perm

    def _navigate_value(
        self, tap: ResolvedTap, perm_cache: dict | None = None
    ) -> tuple[Any, Any]:
        """The uncached navigation: (pre-``tuple_index`` value, permutation).

        The step loop of the generated frame calls this directly — there each
        request must bind the *current* occurrence, so a cached proxy would
        hand back an earlier step's tensor.
        """
        address = tap.source
        assert address is not None
        site = tap.site
        root = self._drill(site.module.source, address.op_pattern, site)
        perm = None
        if address.align is not None:
            # its own cache slot: several addresses share one sort op, and
            # the permutation must be requested exactly once per trace
            perm_key = (id(site.module), address.op_pattern, "align", address.align)
            if perm_cache is not None and perm_key in perm_cache:
                perm = perm_cache[perm_key]
            else:
                perm = self._drill(root.source, address.align, site).output[1]
                if perm_cache is not None:
                    perm_cache[perm_key] = perm
        op = root
        for entry in address.peel:
            op = self._drill(op.source, entry, site)
        if address.field is not None:
            op = self._drill(op.source, address.field, site)
        if address.arg is not None:
            index, keyword = address.arg
            return op.inputs[index][keyword], perm
        return op.output, perm

    def _present_native(self, tap: ResolvedTap, value: Any, perm: Any) -> Any:
        """The value as the declared shape describes it — semantic order.

        An ``align`` address' rows are in the kernel's expert-sorted order:
        un-sort them (a gather, so the result is a copy — writes go back
        through `_land_writes`' restore path, never through this).
        ``expert_rows`` re-packs ``(batch·position·top_k, …)`` rows into the
        declared 2-D native ``(batch·position, top_k·…)``.
        """
        address = tap.source
        assert address is not None
        if address.align is not None:
            value = value[torch.argsort(perm)]
        if address.expert_rows:
            # (batch·position·top_k, d) → (batch·position, top_k·d). The row
            # width is the NATIVE packed width — every declared axis behind
            # the position, fused split included (a fused capture's native row
            # is top_k·splits·d; the split is selected later, in to_contract)
            # — so it is read off the axes, not off `shape.width` (the
            # contract width, which a fused shape is narrower than). An
            # integral tap has no feature axis, so its row is the top-k
            # itself.
            value = value.reshape(-1, self._native_row_width(tap.site))
        return value

    @staticmethod
    def _native_row_width(site: ResolvedSite) -> int:
        widths = [
            axis.width
            for axis in site.shape.axes
            if axis.kind in ("topk", "fused", "feature")
        ]
        assert widths  # expert_rows implies at least a top-k axis
        row = 1
        for width in widths:
            row *= width
        return row

    def _drill(self, source: Any, pattern: str, site: ResolvedSite) -> Any:
        """One matched op on one ``.source``, with the refusal made legible."""

        def line_of(name: str) -> str:
            op = getattr(source, name)
            text, line = getattr(op, "text", None), getattr(op, "line", None)
            if not isinstance(text, str) or not isinstance(line, int):
                return ""
            return text.splitlines()[line - 1]

        try:
            return getattr(source, match_op(pattern, source.names, line_of))
        except AddressResolutionError as error:
            import transformers

            raise ProtocolError(
                "P4",
                f"addressing {site.component!r} at layer {site.layer} of "
                f"{self.bundle.key!r} (transformers "
                f"{transformers.__version__}): {error}",
            ) from error

    # ------------------------------------------------------------------ #
    # per-fire taps: ops that fire once per kernel chunk
    # ------------------------------------------------------------------ #

    def _navigate_fire_ops(self, tap: ResolvedTap) -> tuple[Any, Any]:
        """The (value op, trip-count op) of a per-fire address — both in the
        same drilled source, memoized per trace."""
        address, site = tap.source, tap.site
        assert address is not None
        assert address.field is not None and address.trip is not None, (
            "a per-fire address names its value op and its loop's range"
        )
        key = (id(site.module), address.op_pattern, address.peel, "per-fire")
        if key not in self._trace_sources:
            op = self._drill(site.module.source, address.op_pattern, site)
            for entry in address.peel:
                op = self._drill(op.source, entry, site)
            parent = op.source
            self._trace_sources[key] = (
                self._drill(parent, address.field, site),
                self._drill(parent, address.trip, site),
            )
        return self._trace_sources[key]

    def _collect_per_fire(self, tracer: Any, tap: ResolvedTap) -> Any:
        """Every fire's tensor, as a saved list — one entry per kernel chunk.

        📐 The trip count is the loop's own ``range(...)`` output (never
        config: the kernel pads the sequence to a chunk multiple), requested
        before the loop — its op fires first.
        """
        import nnsight

        value_op, trip_op = self._navigate_fire_ops(tap)
        count = len(trip_op.output)
        sink = nnsight.save([])
        for _ in tracer.iter[:count]:
            sink.append(value_op.output)
        return sink

    def _land_per_fire_writes(
        self,
        tracer: Any,
        tap: ResolvedTap,
        entries: list[tuple[str, WriteSpec, ResolvedSite]],
        input_role: str,
        batch: EncodedBatch,
    ) -> None:
        """Land writes on specific fires — one ``tracer.iter[k]`` body each.

        A fire index past the last fire would simply never run (the loop body
        binds to a fire that does not happen), so the count is saved and the
        miss refused after the trace rather than silently skipped.
        """
        import nnsight

        value_op, trip_op = self._navigate_fire_ops(tap)
        count = nnsight.save(len(trip_op.output))
        batch_size = int(batch.input_ids.shape[0])
        fires = self._write_fire_indices(tap, entries)
        for k in fires:
            for _ in tracer.iter[k]:
                value = value_op.output
                # in value-space throughout: a reshape of the traced tensor is
                # a view, so the final fill reaches the kernel's own storage
                site = tap.site
                width = site.shape.width
                heads = site.shape.head_space
                assert width is not None and heads is not None
                native = value.reshape(-1, 1, heads, width // heads).clone()
                contract = to_contract(native, site.shape, batch_size=batch_size)
                self._apply_writes_to_contract(
                    entries,
                    input_role,
                    batch,
                    contract,
                    per_row=[[0]] * batch_size,
                )
                new_native = from_contract(
                    contract, site.shape, batch_size=batch_size, native=native
                )
                value[:] = new_native.reshape(value.shape)
        self._fire_checks.append((tap.site.component, max(fires), count))

    def _write_fire_indices(
        self,
        tap: ResolvedTap,
        entries: list[tuple[str, WriteSpec, ResolvedSite]],
    ) -> list[int]:
        """Which fires this address' writes target: plain non-negative
        indices only — the fire count is a runtime fact of the kernel, so
        nothing anchored, spanned, or counted from the end can resolve before
        the trace runs."""
        fires: set[int] = set()
        for ename, write, _site in entries:
            spec = self._spec(write.pos)
            index = spec.index if isinstance(spec.index, int) else None
            anchored = any(
                getattr(spec, field) is not None
                for field in ("span", "variable", "column", "scope", "relative_to")
            )
            if index is None or index < 0 or anchored:
                raise ProtocolError(
                    "P4",
                    f"write {ename!r} targets {tap.site.component!r}, whose "
                    f"position axis is the kernel's fire index "
                    f"({tap.site.shape.describe()}): only a plain non-negative "
                    "integer index resolves there — the fire count is the "
                    "kernel's runtime fact, so text anchors, spans and "
                    "negative indices have nothing to resolve against before "
                    "the trace runs.",
                )
            fires.add(index)
        return sorted(fires)

    def _finalize_per_fire_read(
        self,
        ref: ReadRef,
        read: Any,
        tap: ResolvedTap,
        sink: Any,
        batch: EncodedBatch,
        input_role: str,
    ) -> torch.Tensor:
        """One per-fire read's value: stack the fires into the declared
        position axis, then the shared gather/featurize path with positions
        resolved against the fire count."""
        site = tap.site
        fires = list(sink)
        stacked = torch.stack(fires, dim=1)  # (b, fires, *native tail)
        width = site.shape.width
        heads = site.shape.head_space
        assert width is not None and heads is not None
        native = stacked.reshape(
            stacked.shape[0], stacked.shape[1], heads, width // heads
        )
        raw = to_contract(native, site.shape, batch_size=int(batch.input_ids.shape[0]))
        per_row = self._fire_positions(ref.read, read.pos, len(fires), raw.shape[0])
        return self._finalize_read(
            ref, read, site, raw, batch, input_role, per_row=per_row
        )

    def _fire_positions(
        self, rname: str, pos: Any, n_fires: int, n_rows: int
    ) -> list[list[int]]:
        """Positions on a fire axis: ``all``, or one integer index (negative
        counts from the last fire). Anything anchored refuses — there is no
        text on a chunk axis."""
        spec = self._spec(pos)
        anchored = any(
            getattr(spec, field) is not None
            for field in ("span", "variable", "column", "scope", "relative_to")
        )
        if not anchored and getattr(spec, "all", None) is True:
            return [list(range(n_fires))] * n_rows
        index = spec.index if isinstance(spec.index, int) else None
        if anchored or index is None:
            raise ProtocolError(
                "P4",
                f"read {rname!r} addresses positions on a per-fire component, "
                "whose position axis is the kernel's chunk index: only "
                '"all" or a plain integer index resolves there — text '
                "anchors and spans have nothing to resolve against.",
            )
        if not -n_fires <= index < n_fires:
            raise ProtocolError(
                "P2",
                f"read {rname!r} addresses fire {index}, but the kernel fired "
                f"{n_fires} times on this batch",
            )
        return [[index % n_fires]] * n_rows

    # ------------------------------------------------------------------ #
    # the generated frame: step-anchored reads under model.generate
    # ------------------------------------------------------------------ #

    def _wrap_generated(self, site: ResolvedSite) -> ResolvedTap:
        """A continuation tap's landing.

        Module boundaries and the stackable interface slots are the prompt
        frame's own landings, one occurrence per decode step. An *interior*
        component must appear in the generated-frame table — decode dispatches
        different code than prefill (the recurrent delta kernel replaces the
        chunked one), so a prompt-frame address is not evidence the tensor
        exists per step.
        """
        if site.kind in ("interior", "experts", "delta"):
            # a kernel-boundary component (kind "delta") is served in the
            # prompt frame through the chunked kernel's address; decode runs
            # the recurrent kernel, so it too needs a verified decode address
            stream = self.bundle.stream_at(site.layer)
            address = GENERATED_ADDRESSES.get(stream, {}).get(site.component)
            if address is None:
                raise ProtocolError(
                    "P4",
                    f"component {site.component!r} has no generated-frame "
                    "address in the nnsight engine's tables "
                    "(neural/engines/nnsight_tracing/addresses.py): the decode "
                    "path dispatches different kernels than prefill, so an "
                    "interior tensor is only served per step once its decode "
                    "address is verified. Read it in the prompt frame.",
                )
            return ResolvedTap(site=site, source=address)
        if site.interface_slot is not None:
            return self._wrap(site)
        return ResolvedTap(site=site)

    def _generated_step_value(self, tap: ResolvedTap, batch_size: int) -> Any:
        """One decode step's contract tensor for one continuation tap.

        Navigated fresh inside the step body — ``tracer.iter`` binds each
        request to the current occurrence, so the per-trace cache must not
        hand back an earlier step's value.
        """
        site = tap.site
        if tap.source is None:
            envoy = site.module
            if site.kind == "out":
                native = tap_tensor(envoy.output, site.tuple_index)
            else:
                native = envoy.input
            return to_contract(native, site.shape, batch_size=batch_size)
        value, perm = self._navigate_value(tap)
        if tap.source.tuple_index is not None:
            value = value[tap.source.tuple_index]
        native = self._present_native(tap, value, perm)
        if tap.source.fires != "once":
            # a per-step state has no position axis of its own: give it the
            # declared shape's one-step form, (b, 1, heads, feature)
            heads = site.shape.head_space
            width = site.shape.width
            assert heads is not None and width is not None
            native = native.reshape(-1, 1, heads, width // heads)
        return to_contract(native, site.shape, batch_size=batch_size)

    def _finalize_generated(
        self,
        model: str,
        input_role: str,
        *,
        batch: EncodedBatch,
        depth: int,
        gen_taps: list[tuple[str, Any]],
        gen_sites: dict[str, ResolvedSite],
        gen_wrapped: dict[str, ResolvedTap],
        gen_sinks: dict,
        gen_result: Any,
    ) -> None:
        """Build the continuation frame and finalize its reads — the same
        step-space semantics as the reference engine's decode."""
        prompt_len = int(batch.input_ids.shape[1])
        generated = gen_result[:, prompt_len:][:, :depth].detach().cpu()
        rows_n = int(generated.shape[0])
        eos = self.bundle.tokenizer.eos_token_id
        widths: list[int] = []
        for row in range(rows_n):
            width = depth
            if eos is not None:
                hit = (generated[row] == eos).nonzero()
                if hit.numel():
                    width = int(hit[0].item())
            widths.append(width)
        continuation = continuation_frame(
            self.bundle.tokenizer, generated, tuple(widths)
        )
        self._continuations[(model, input_role)] = continuation

        dataset_rows = self.role_rows[input_role]
        field = self.role_fields[input_role]
        head = None
        for rname, read in gen_taps:
            ref = ReadRef(rname, model)
            site = gen_sites[rname]
            tap = gen_wrapped[rname]
            sink = gen_sinks[tap_key(tap.site, tap.source)]
            frame = torch.cat(list(sink), dim=1)  # (b, steps, feature)
            per_row = [
                resolve_steps(
                    self._spec(read.pos),
                    continuation,
                    row,
                    dataset_row=dataset_rows[row],
                    field=field,
                )
                for row in range(rows_n)
            ]
            self._read_steps[ref] = per_row
            project = None
            if tap.site.component != site.component:  # ln_final kept, lm_head owed
                head = (
                    head
                    or resolve_site(self.bundle, SiteSpec(component="lm_head")).module
                )
                project = head
            self._read_values[ref] = self._finalize_read(
                ref,
                read,
                site,
                frame,
                batch,
                input_role,
                per_row=per_row,
                project=project,
            )

    @contextlib.contextmanager
    def _switched_implementations(self, required: frozenset[str]) -> Iterator[None]:
        """Run one trace under the implementations its addresses require.

        📐 The runtime switch is verified on the real A3B and the fixture
        (`set_attn_implementation` both directions). The model default is
        restored afterwards, so a document that never touches the pattern
        keeps whatever the checkpoint prefers (sdpa); the switch is on demand.
        """
        model = self.bundle.model
        previous_attn: str | None = None
        previous_experts: str | None = None
        if required:
            # 📐 nnsight dispatches the real model on first trace, and dispatch
            # rebuilds the config — a switch applied before it is silently
            # reset (measured: eager set pre-dispatch, sdpa ran). Dispatch
            # first, so the switch lands on the model that will run.
            if not getattr(model, "dispatched", True):
                model.dispatch()
        if "attn_eager" in required and model.config._attn_implementation != "eager":
            previous_attn = model.config._attn_implementation
            model.set_attn_implementation("eager")
        if "experts_grouped" in required:
            # grouped_mm is the checkpoint default (an unset value means it),
            # so this only ever switches back from an explicit "eager"
            current = getattr(model.config, "_experts_implementation", None)
            if current is not None and current != "grouped_mm":
                previous_experts = current
                model.set_experts_implementation("grouped_mm")
        try:
            yield
        finally:
            if previous_attn is not None:
                model.set_attn_implementation(previous_attn)
            if previous_experts is not None:
                model.set_experts_implementation(previous_experts)

    # ------------------------------------------------------------------ #
    # in-trace plumbing
    # ------------------------------------------------------------------ #

    def _tap_contract(self, tap: ResolvedTap, batch_size: int) -> torch.Tensor:
        """One tap's tensor in the contract shape — read off its envoy, or
        off the op its interior address names."""
        site = tap.site
        if tap.source is not None:
            raw, perm = self._source_value(tap)
            native = self._present_native(tap, raw, perm)
            return to_contract(native, site.shape, batch_size=batch_size)
        envoy = site.module
        if site.kind == "out":
            native = tap_tensor(envoy.output, site.tuple_index)
        else:
            native = envoy.input
        return to_contract(native, site.shape, batch_size=batch_size)

    def _land_writes(
        self,
        tap: ResolvedTap,
        entries: list[tuple[str, WriteSpec, ResolvedSite]],
        input_role: str,
        batch: EncodedBatch,
    ) -> Any:
        """Apply every write at one address (shared class-ordered math) and
        assign the result back — to the envoy, or into the interior op's
        tensor. An input site with a declared ``writeback`` returns its rewrite
        delta so the operation loop can land it at the enclosing output's own
        forward depth.

        The interior landing is an in-place fill (``value[:] = new``): the op
        has already produced its tensor when the write runs, and everything
        downstream — the next op in the same forward, a later read of the
        same address — consumes that same object. 📐 Verified consumed on the
        fixture and the real A3B (zeroing the softmax input moved the logits
        by 1.78; a pattern write is #53's finding).
        """
        site = tap.site
        batch_size = int(batch.input_ids.shape[0])
        if tap.source is not None:
            address = tap.source
            raw, perm = self._source_value(tap)
            native = self._present_native(tap, raw, perm).clone()
            contract = to_contract(native, site.shape, batch_size=batch_size)
            self._apply_writes_to_contract(entries, input_role, batch, contract)
            new_native = from_contract(
                contract, site.shape, batch_size=batch_size, native=native
            )
            if address.expert_rows:
                # back to one row per (token, slot), then to the kernel's own
                # order — the exact inverse of _present_native. The per-slot
                # row is the NATIVE row over top_k (a fused capture's slot row
                # is splits·d, and from_contract has already written the edit
                # back into the whole fused native).
                topk = next(
                    axis.width for axis in site.shape.axes if axis.kind == "topk"
                )
                rows = new_native.reshape(-1, self._native_row_width(site) // topk)
                if address.align is not None:
                    rows = rows[perm]
                raw[:] = rows
            else:
                raw[:] = new_native
            return
        envoy = site.module
        if site.kind == "out":
            payload = envoy.output
            original_native = tap_tensor(payload, site.tuple_index)
        else:
            payload = None
            original_native = envoy.input
        native = original_native.clone()
        contract = to_contract(native, site.shape, batch_size=batch_size)
        self._apply_writes_to_contract(entries, input_role, batch, contract)
        new_native = from_contract(
            contract, site.shape, batch_size=batch_size, native=native
        )
        if site.kind == "out":
            envoy.output = rebuild_payload(payload, site.tuple_index, new_native)
            return None
        envoy.input = new_native
        if site.writeback is not None:
            return new_native - original_native
        return None

    @staticmethod
    def _land_writeback(tap: ResolvedTap, delta: Any) -> None:
        """Add a deferred input rewrite to its declared target's output, at the
        payload element that target's own tap names."""
        writeback = tap.site.writeback
        assert writeback is not None
        payload = writeback.module.output
        native = tap_tensor(payload, writeback.tuple_index)
        writeback.module.output = rebuild_payload(
            payload, writeback.tuple_index, native + delta
        )
