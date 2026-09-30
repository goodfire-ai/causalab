"""Check a document against the selected engine's capabilities.

The registry declares available operations and components. These checks report
unsupported requirements before the engine loads a model."""

from __future__ import annotations

import functools
from typing import Any, Mapping, Sequence

from causalab.protocol.positions.encoding import generated_budget
from causalab.protocol.parallel import ONE, ParallelGeometry, format_geometry
from causalab.protocol.registry import capability, write_capabilities
from causalab.protocol.registry.engines import component_capability
from causalab.protocol.rules.errors import (
    ProtocolError,
    ValidationError,
    ValidationErrors,
)
from causalab.protocol.schema import Document, BoundAggregation, operand_reads

__all__ = [
    "CAPABILITIES",
    "check_caller_bundle",
    "check_engine_support",
    "refuse_shortfall",
    "requires",
    "requires_campaign",
    "train_capabilities",
]

#: The closed capability vocabulary (§8). Component capabilities
#: (``component:<name>``, ``component:<name>:write``) are *generated*, one per
#: entry of the closed [`Component`][causalab.protocol.schema.types.Component] vocabulary —
#: two engines with different site surfaces route on them, and the vocabulary
#: stays closed because ``Component`` already is. The one component-shaped
#: verb (``writable_attention_probs``: the pattern's write goes through the
#: attention function, not a hook) is likewise generated — from the
#: ``write_capability`` cell of the capability rows — so the verb exists
#: because a row says a write there costs more than the component entry.
CAPABILITIES: tuple[str, ...] = (
    "grad",
    "paired_forward",
    "full_logits",
    *sorted(write_capabilities()),
    "pytorch_fn_local",
    "generate",
    "generation_writes",
    "quantized_weights",
    # the three training facts a fit can author that an engine's loop may not
    # honour (§2.11, rule 30): a free ``params`` tensor in ``train.params``, a
    # ``train.precision`` other than fp32, an ``updates``-counted ``eval``.
    # Neither shipped engine offers them, and each used to be refused inside
    # the train loop, after the weights had loaded.
    "train_free_params",
    "train_loss_precision",
    "train_eval_updates",
)


#: Metric kinds that need the whole vocabulary materialized (§8) — but only
#: when their read actually taps ``lm_head``. ``class_probs`` always does
#: (validation binds it to a vocabulary projection); ``top_k`` ranks whatever
#: axis its read has, and a top-k over a 4k-wide residual stream or a 100k-wide
#: SAE code obliges no vocabulary projection at all. Charging it ``full_logits``
#: would route such a document onto a full-vocab engine for nothing.
_FULL_VOCAB_METRICS = frozenset({"top_k", "class_probs"})


def _metric_read_obliges_full_projection(doc: Document, agg: BoundAggregation) -> bool:
    """Whether serving ``metric``'s read means materializing the vocabulary.

    Deliberately NOT [`read_is_vocabulary`][causalab.protocol.schema.types.read_is_vocabulary]:
    that predicate asks what the read *hands the metric* (token ids, or a
    featurizer's latents / a ``dims`` re-index), which governs softmaxing and
    decoding. This one asks what the engine must *compute upstream* — and a
    featurized ``lm_head`` read still consumes the whole projection, its
    featurizer merely re-expresses it. The two questions diverge exactly
    there. A ``dims`` slice is the one transform that needs only its named
    rows, matching the saved-read rule above.
    """
    read = doc.reads.get(agg.read.read)
    if read is None:
        return False
    site = doc.sites.get(str(read.site))
    return site is not None and site.component == "lm_head" and read.dims is None


def requires(doc: Document) -> frozenset[str]:
    """The capability set one concrete document needs — derived, never
    authored (§6).

    Component needs are part of the set: every site a read or write
    references contributes ``component:<name>`` (writes also
    ``component:<name>:write``), so a document is routed by *what it touches*,
    not only by the coarse §8 verbs — the honest answer once two engines with
    different site surfaces exist. Stream- and layer-level constraints stay
    engine-internal: they depend on the loaded model, which routing never
    sees."""
    needed: set[str] = set()
    if doc.train is not None:
        needed.add("grad")
        needed.update(train_capabilities(doc))
    for read in doc.reads.values():
        needed.add(component_capability(doc.sites[str(read.site)].component))
    for write in doc.writes.values():
        component = doc.sites[str(write.site)].component
        needed.add(component_capability(component))
        needed.add(component_capability(component, write=True))
    if doc.model.quantization is not None:
        needed.add("quantized_weights")
    for im in doc.intervened_models.values():
        if not isinstance(im.writes, tuple):
            raise AssertionError(
                "requires() takes a concrete point document — expand sweeps first"
            )
        for ename in im.writes:
            write = doc.writes[ename]
            for ref in operand_reads(doc, write.do):
                if ref.model is not None and doc.input_of(ref.model) != str(im.input):
                    needed.add("paired_forward")
            if write.do.mechanism == "pytorch_fn":
                needed.add("pytorch_fn_local")
            site = doc.sites[str(write.site)]
            if isinstance(site.component, str):
                verb = capability(site.component).write_capability
                if verb is not None:
                    needed.add(verb)
    for ref in doc.saved_raw_reads():
        read = doc.reads[ref.read]
        if read.dims is None:
            site = doc.sites[str(read.site)]
            if site.component == "lm_head":
                needed.add("full_logits")
    for agg in doc.aggregations():
        if (
            agg.spec.kind in _FULL_VOCAB_METRICS
            and _metric_read_obliges_full_projection(doc, agg)
        ):
            needed.add("full_logits")
    for read in doc.reads.values():
        if generated_budget(doc, read.pos) is not None:
            # a continuation to address means the engine must decode one
            needed.add("generate")
            break
    if any(im.writes_during_generation for im in doc.intervened_models.values()):
        # the writes stay in force through the decode steps (§2.9): a hook
        # that fires per step, which only an engine that walks the steps
        # itself can install
        needed.add("generation_writes")
    return frozenset(needed)


def train_capabilities(doc: Document) -> frozenset[str]:
    """The training verbs a fit's own fields oblige (§2.11): each is a fact
    the document decides and an engine's loop may not implement, so it is
    routed on here and refused by name under rule 30 when the routed engine
    lacks it ([`causalab.protocol.rules.document.check_engine_support`][causalab.protocol.rules.capability.check_engine_support]) — the
    one derivation both read, so routing and the rule cannot disagree.

    * ``train_free_params`` — a ``train.params`` entry names a ``params``
      entry (a free tensor, §2.6) rather than a featurizer or a slot;
    * ``train_loss_precision`` — ``train.precision.feature`` or ``.loss`` is
      authored as anything but ``fp32``;
    * ``train_eval_updates`` — ``train.eval.every`` counts ``updates``.
    """
    train = doc.train
    if train is None:
        return frozenset()
    needed: set[str] = set()
    if any(pname in doc.params for pname in train.params):
        needed.add("train_free_params")
    if train.precision is not None and any(
        isinstance(value, str) and value != "fp32" for value in train.precision.values()
    ):
        needed.add("train_loss_precision")
    if train.eval is not None and "updates" in train.eval["every"]:
        needed.add("train_eval_updates")
    return frozenset(needed)


def requires_campaign(docs: Sequence[Document]) -> frozenset[str]:
    """The union of every point's capability needs — a heterogeneous sweep
    routes on the whole campaign, not its first point."""
    needed: frozenset[str] = frozenset()
    for doc in docs:
        needed |= requires(doc)
    return needed


# --------------------------------------------------------------------------- #
# rules 13 + 30 — the rules that need the named engine
# --------------------------------------------------------------------------- #


def check_engine_support(doc: Document, engine_capabilities: frozenset[str]) -> None:
    """The engine-aware half of the checklist — rule 13 (a ``pytorch_fn``
    needs a local engine, §2.8) and rule 30 (the fit as authored needs the
    engine's training verbs, §2.11) — against one engine's capability set.

    The same two rule functions [`validate_document`][causalab.protocol.rules.document.validate_document] runs when it is
    handed ``engine_capabilities``; public so the pipeline can re-enter them
    for the engine ``--engine`` named (``pipeline.check_engine``), which no
    door knows when it builds. Independent violations are reported together,
    as the checklist's are.
    """
    failures: list[ValidationError] = []
    for check in (
        functools.partial(
            _check_pytorch_fn, doc, "pytorch_fn_local" in engine_capabilities
        ),
        functools.partial(_check_train_engine_supported, doc, engine_capabilities),
    ):
        try:
            check()
        except ValidationError as err:
            failures.append(err)
    if len(failures) == 1:
        raise failures[0]
    if failures:
        raise ValidationErrors(failures)


def _check_train_engine_supported(doc: Document, capabilities: frozenset[str]) -> None:
    """Rule 30 — the named engine can execute the fit as authored (§2.11).

    Three facts a fit authors are decided by the document and honoured, or
    not, by the engine's loop: a free ``params`` tensor in ``train.params``,
    a ``train.precision`` other than fp32, and an ``eval`` counted in
    ``updates``. Each maps to one §8 verb the engine declares or does not
    (``engine.train_capabilities`` is the one derivation the check shares), and
    each used to be discovered inside the loop — after the weights had
    loaded, or not at all: a bf16 loss was digested as bf16 and run at fp32
    with nothing said. The refusal names the field and the missing verb; the
    engine's twin refusals stay for a document that arrived unvalidated.
    """
    train = doc.train
    if train is None:
        return
    if "train_free_params" not in capabilities:
        for i, pname in enumerate(train.params):
            if pname in doc.params:
                raise ValidationError(
                    30,
                    f"train.params names the free tensor {pname!r}, which the "
                    "routed engine cannot train: it declares no "
                    "'train_free_params' capability — featurizer slots only "
                    "(sec. 2.11)",
                    path=f"train.params[{i}]",
                )
    if "train_loss_precision" not in capabilities and train.precision is not None:
        for key in ("feature", "loss"):
            value = train.precision.get(key)
            if isinstance(value, str) and value != "fp32":
                raise ValidationError(
                    30,
                    f"train.precision.{key} is {value!r}, but the routed engine "
                    "computes its loss path in fp32 only: it declares no "
                    "'train_loss_precision' capability, so the authored "
                    "precision would be digested and not executed (sec. 2.11)",
                    path=f"train.precision.{key}",
                )
    if (
        "train_eval_updates" not in capabilities
        and train.eval is not None
        and "updates" in train.eval["every"]
    ):
        raise ValidationError(
            30,
            "train.eval.every counts updates, but the routed engine evaluates "
            "on epoch boundaries only: it declares no 'train_eval_updates' "
            "capability, so the fit would run no eval at all and still save "
            "(sec. 2.11)",
            path="train.eval.every",
        )


# --------------------------------------------------------------------------- #
# rule 13 — pytorch_fn is local-only
# --------------------------------------------------------------------------- #


def _check_pytorch_fn(doc: Document, engine_is_local: bool) -> None:
    if engine_is_local:
        return
    for ename, write in doc.writes.items():
        if write.do.mechanism == "pytorch_fn":
            raise ValidationError(
                13,
                f"write {ename!r} uses pytorch_fn, which only a local engine may "
                "run (§2.8) — the selected engine is not local",
                path=f"writes.{ename}.do",
            )


# --------------------------------------------------------------------------- #
# rule 13, the shortfall face — the engine ``--engine`` named lacks a required verb
# --------------------------------------------------------------------------- #


def refuse_shortfall(required: frozenset[str], offered: frozenset[str]) -> None:
    """Rule 13's capability-shortfall refusal (§8, ``[V13]``), generated from
    the missing entries — never hand-written per case. The engine is the
    caller's explicit choice (``--engine``; ``auto`` is the reference engine),
    so the text names the one engine's shortfall and nothing falls through to
    a second engine: the user pins the one that serves the document."""
    missing = required - offered
    if missing:
        raise ValidationError(
            13,
            f"the engine does not support this document: it requires "
            f"{sorted(required)} and lacks {sorted(missing)} (sec. 8)",
        )


# --------------------------------------------------------------------------- #
# the caller-owned bundle (spec §9) — held to the document before any forward
# --------------------------------------------------------------------------- #


def check_caller_bundle(
    bundle: Any,
    realization: Mapping[str, Any],
    *,
    device: str,
    geometry: ParallelGeometry = ONE,
) -> None:
    """Refuse a caller-owned bundle that does not realize the document's model.

    An engine built with ``bundle=`` (spec §9, the ownership contract) runs
    that bundle instead of loading — so before any forward, the document's
    canonical ``model`` block ([`canonical_model`][causalab.protocol.schema.explicit.canonical_model]:
    ``key``, ``revision``, ``dtype``, the materialized ``quantization``, and
    any explicit ``attn_implementation``) is
    compared field by field with what the bundle says it is, the engine's
    ``device`` word with the bundle's device map, and the engine's parallel
    ``geometry`` with the one the bundle was loaded under
    (``docs/model_parallelism.md`` §3: a bundle at ``world > 1`` is this
    rank's sharded load, or the receipt's ``execution.parallel`` would
    describe shards that do not exist). A disagreement refuses, naming both
    sides: the run receipt and every ``ArtifactIdentity`` stamp would
    otherwise describe a model that did not run. The comparison is by value
    (a materialized ``quantization`` block is a mapping, order-free; the
    device word is parsed over the bundle's tower, so ``cuda`` and
    ``cuda:<current>`` are one placement), and the refusal names each side's
    own spelling.
    """
    disagreements = [
        f"{what}: the document says {theirs!r}, the bundle {ours!r}"
        for what, theirs, ours in (
            ("model.key", str(realization["key"]), bundle.key),
            ("model.revision", str(realization["revision"]), bundle.revision),
            ("model.dtype", str(realization["dtype"]), bundle.dtype),
            (
                "model.quantization",
                realization.get("quantization"),
                bundle.quantization,
            ),
        )
        if theirs != ours
    ]
    # the device word is parsed over the bundle's tower (``cuda`` and
    # ``cuda:<current>`` are one placement), through the bundle's own map
    # class so this module stays torch-free
    placed = bundle.devices
    if type(placed).parse(device, len(placed.blocks)) != placed:
        disagreements.append(
            f"device: the engine was built for {device!r}, the bundle is placed "
            f"on {placed.requested!r}"
        )
    if "attn_implementation" in realization:
        wanted = realization["attn_implementation"]
        actual = getattr(bundle.model.config, "_attn_implementation", None)
        if wanted != actual:
            disagreements.append(
                f"model.attn_implementation: the document says {wanted!r}, "
                f"the bundle {actual!r}"
            )
    loaded_under: ParallelGeometry = bundle.geometry
    if loaded_under != geometry:
        disagreements.append(
            f"parallel geometry: the engine runs under {format_geometry(geometry)}, "
            f"the bundle was loaded under {format_geometry(loaded_under)}"
        )
    if disagreements:
        raise ProtocolError(
            "P4",
            "the caller-owned bundle does not realize this document's model, "
            "so the run receipt and every artifact stamp would describe a "
            "model that did not run — " + "; ".join(disagreements) + ". Hand "
            "the engine a bundle built for this document (or edit the "
            "document to say what actually runs)",
        )
