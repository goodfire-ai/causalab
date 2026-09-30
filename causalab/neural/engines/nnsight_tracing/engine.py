"""Connect the nnsight loader and executor to shared execution.

Registry rows declare supported components and capabilities. Training and
quantized weights require the hooks engine. Validation checks the named
engine before execution.
"""

from __future__ import annotations

import functools
from typing import Any, Mapping

from causalab.neural.engines.nnsight_tracing.executor import TracePointExecutor
from causalab.neural.engines.nnsight_tracing.loading import NnsightBundle, load_model
from causalab.neural.shared.execution import execute_request
from causalab.protocol.positions.resolve import StepResolution
from causalab.io.tensor_files import load_table, load_tensors
from causalab.protocol.rules.capability import check_caller_bundle
from causalab.protocol.positions.roles import resolve_roles
from causalab.protocol.schema.explicit import canonical_model
from causalab.protocol.compiled import CompiledProtocol
from causalab.protocol.engine import Engine, RunContext, RunResult
from causalab.protocol.rules.errors import ProtocolError
from causalab.protocol.registry import components_served_by, declared_capabilities
from causalab.protocol.schema import Document

__all__ = ["NnsightEngine"]


class NnsightEngine(Engine):
    name = "nnsight"
    # The engine-level verbs are a row of the capability registry
    # (`registry.ENGINE_VERBS`), plus the write verbs the rows charge for the
    # components this engine serves — not a literal here.
    capabilities = declared_capabilities("nnsight")
    # Which components this engine serves is a row in the capability registry
    # (`registry.CAPABILITIES`, the `reads` cell): the whole vocabulary but
    # the `delta_*` set, like the reference engine — module boundaries land
    # on envoys, the attention interior through the `.source` address table,
    # and 'attention_result' (derived by re-invoking the o-projection) works
    # because an envoy outside a trace calls its underlying module. The
    # per-expert MoE interior and the DeltaNet interior (`deltanet_*`) are the
    # vocabulary only this engine serves; routing lands them here by name.
    # Read-only / swap-only components and stream constraints are *protocol
    # policy* (the rows' `writes` and `stream` cells), not capability gaps —
    # the same argument the reference engine's declaration makes, and why
    # `writable_components` is the same set.
    #
    # The `delta_*` vocabulary is the *reference engine's* DeltaNet
    # interior: the kernel boundary is reached by swapping the modeling
    # file's module globals for the dynamic extent of one mixer forward, and
    # the per-step interior by stepping the recurrent kernel inside the
    # swapped globals — pytorch_hooks mechanisms with no nnsight equivalent
    # (this engine's DeltaNet interior is the `deltanet_*` set). The
    # `delta_*` module-boundary taps (qkv, gate, premix) are ordinary envoy
    # reads and would very likely work here unchanged — but nothing exercises
    # them on this engine, and declaring support this engine has never been
    # tested for is the claim worth not making.
    components = components_served_by("nnsight")
    writable_components = components
    is_local = True

    def __init__(
        self, *, device: str = "cpu", bundle: NnsightBundle | None = None
    ) -> None:
        # placement is execution (the engine's call, §8); precision is not —
        # dtype comes from each point's own `model` section. `bundle` is a
        # caller-owned model run instead of loading one (spec §9): checked
        # against each document's realization before any forward, never
        # loaded, moved, freed or re-moded by this engine
        self.device = device
        self.bundle = bundle

    @property
    def model_source(self) -> str:
        """``"caller"`` when this engine runs a bundle handed to it, else
        ``"loaded"`` — execution provenance for the run receipt (§8)."""
        return "caller" if self.bundle is not None else "loaded"

    # ------------------------------------------------------------------ #

    def execute(self, compiled: CompiledProtocol, run: RunContext) -> RunResult:
        return execute_request(
            compiled,
            run,
            engine_name=self.name,
            # the trace executor does not consult the shared ForwardCache, so
            # it takes the campaign's interning handle and drops it; §3's
            # cross-point sharing is unclaimed here and RunResult.forwards
            # stays 0 rather than reporting a count nothing measured
            executor_factory=lambda doc, ctx, coords, _interning, resolution: (
                self._executor(doc, ctx, coords=coords, resolution=resolution)
            ),
            train_runner=None,
            engine=self,
        )

    # ------------------------------------------------------------------ #

    def _executor(
        self,
        doc: Document,
        run: RunContext,
        *,
        coords: Mapping[str, Any] | None = None,
        resolution: StepResolution | None = None,
    ) -> TracePointExecutor:
        realization = canonical_model(doc.raw["model"])
        if realization.get("quantization") is not None:
            raise ProtocolError(
                "P4",
                "this document declares weight quantization, which the "
                "nnsight engine has not verified through its loader — its "
                "'quantized_weights' capability is absent, so routing should "
                "not have sent it here; the reference engine serves it",
            )
        if self.bundle is not None:
            check_caller_bundle(self.bundle, realization, device=self.device)
            bundle = self.bundle
        else:
            bundle = load_model(
                str(doc.model.key),
                str(doc.model.revision),
                dtype=str(realization["dtype"]),
                device=self.device,
                **(
                    {"attn_implementation": realization["attn_implementation"]}
                    if "attn_implementation" in realization
                    else {}
                ),
            )
        role_rows, role_fields = resolve_roles(doc, run.env)
        return TracePointExecutor(
            doc,
            bundle,
            role_rows=role_rows,
            role_fields=role_fields,
            load_tensors=functools.partial(load_tensors, run),
            load_table=functools.partial(load_table, run),
            coords=coords,
            resolved=resolution.positions if resolution is not None else None,
        )
