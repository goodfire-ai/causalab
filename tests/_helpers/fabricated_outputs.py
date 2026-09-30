"""Fabricate the files a protocol step saves, through the engine's own writers.

A workflow's script steps read what its protocol steps save. The CPU tier
cannot run the models of the paper packages, so a test fabricates those
files instead. Each save entry of each point document goes through the
accumulators the engine uses (``causalab.neural.shared.results.MetricTable``
and ``TensorFile``) and through ``causalab.io.results_io.write_outputs``.
Each metric value comes from ``causalab.neural.shared.metrics.compute_metric``
or ``compute_windowed_metric`` over a fabricated read. A featurizer bundle,
what a fit step saves, holds the slots of the stage the engine trains, built
by the engine's own ``build_stack`` at the site width ``env.model_info``
gives (the model registry by default), and stamped by the engine's
``featurizer_identity``.

What follows the engine. ``tests/_helpers/test_fabricated_outputs.py`` pins
it against a real run on a tiny model:

* the file names, the metric-table columns (``example_id``, the axis ids,
  ``unit``, ``eligible``) and the tensor keys, including the ``.widths``
  sidecar of a ragged read and the per-point keys of a swept fit;
* the rank and dtype of every tensor. A read at one position is
  ``(rows, 1, width)``, the executor's ``(batch, position, feature)``
  contract (``causalab/protocol/registry/shapes.py``). A read over a fixed
  set of positions is ``(rows, positions, width)``. For a position whose
  width per row only the tokenizer decides (a ``variable``, ``column`` or
  ``all`` position), a read is the flat gather plus its widths when the
  widths differ, as ``causalab/neural/shared/executor/base.py`` gathers
  it. Values are in the document's model dtype. A bundle's slots have the
  whole shape and the dtype a real fit writes: a rotation is ``(site
  width, k)``, a gate has one ``theta`` per coordinate, one per head under
  ``group: head``, and one under ``boundary``;
* the fields of the file-level identity and of each ``entries`` record:
  ``slot``, ``coords``, ``trained_on``, ``site`` and the model fields, and
  on a bundle the featurizer fields (``k``, ``parametrization``, ``group``,
  ``group_map``, ``dtype``).

What is fabricated or left out:

* The numbers. A read is standard normal, seeded by what decides it in a
  real run. A read on a model that lands no write depends only on its
  address (site, position, featurizer, dims) and its rows. Every point that
  shares the address then shares the value, as the clean run does at every
  point of a scan. A read on any other model depends on the whole point.
  A bundle's slot depends on the whole point too. A rotation has
  orthonormal columns, a ``clamp`` or ``boundary`` gate lies in ``[0, 1]``,
  and any other gate is standard normal.
* The widths. A vocabulary read is ``len(tokenizer)`` wide, a read with
  ``dims`` is as wide as its dims, and every other read is `WIDTH` wide. A
  tokenizer-decided position gives each row a seeded width from 1 to
  `SPAN_MAX`. A metric reads one position per row, as a real run must.
* The tokenizer is `StandInTokenizer`, one token per distinct string, so
  every answer is one token.
* A decode generates ``max_new_tokens`` ids per row.
* ``loaded_attn_implementation``. The engine stamps the attention backend
  of the model it loaded. No model is loaded here, so the field is absent,
  as it is for an engine run whose executor has no model config.
* ``engine`` is ``fabricated``.

What the helper refuses rather than guesses: the derived records of a
``kind`` entry, a windowed metric of a kind other than ``decode``, a saved
tensor at a generated position, and a read on an input other than ``base``.
It also refuses a saved tensor at a ``span`` or ``indices`` set inside a
``scope`` or ``relative_to`` anchor, whose width follows from the set and
the anchor, not from the tokenizer alone. Of the bundles, it refuses a
featurizer that starts from a saved file, a position gate, a pooled gate, a
featurizer at a site that selects a head or an expert, and any stage other
than a subspace or a gate.
A test that needs one of them extends this module.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch

from causalab.io.env import ResolutionEnv, build_artifact_identity
from causalab.io.results_io import write_outputs
from causalab.neural.shared.execution import featurizer_identity
from causalab.neural.shared.executor.base import document_seed
from causalab.neural.shared.featurizers import (
    Gate,
    Subspace,
    build_stack,
    stage_output_width,
)
from causalab.neural.shared.featurizers.stages import Stage
from causalab.neural.shared.metrics import compute_metric, compute_windowed_metric
from causalab.neural.shared.results import MetricTable, TensorFile
from causalab.neural.shared.values import RaggedValue
from causalab.protocol.estimand import metric_record_identity
from causalab.protocol.identity import site_identity
from causalab.protocol.positions.encoding import generated_budget
from causalab.protocol.positions.resolve import spec_of
from causalab.protocol.positions.roles import input_roles
from causalab.protocol.positions.spans import static_indices
from causalab.protocol.registry import ModelInfo, component_shape, component_width
from causalab.protocol.results import example_labels
from causalab.protocol.schema import WHOLE_WINDOW_METRIC_KINDS, Document
from causalab.protocol.schema.explicit import canonical_model
from causalab.protocol.schema.positions import SpanSpec
from causalab.protocol.schema.types import ReadRef, SaveEntry, read_is_vocabulary
from causalab.provenance import runtime_identity
from causalab.workflow.steps import InnerProtocol
from tests.step_scripts import put_sidecar

__all__ = [
    "SPAN_MAX",
    "VOCAB",
    "WIDTH",
    "StandInTokenizer",
    "fabricate_protocol_outputs",
]

#: The width of every fabricated read that is not a vocabulary projection
#: and has no ``dims``. It is at least the largest PCA rank a shipped script
#: fits (48), so a fit over a fabricated harvest is not refused for its width.
WIDTH = 64
#: The number of ids `StandInTokenizer` holds, and so the width of every
#: fabricated vocabulary read. It is above the number of distinct answer
#: strings a shipped package names.
VOCAB = 1024
#: The largest number of positions a tokenizer-decided position gives one
#: row. It is above 1, so a table of several rows gives a ragged read.
SPAN_MAX = 4
#: The model dtypes a document can author (``PRECISION_DTYPES``), as the
#: engines load them (``causalab/neural/engines/pytorch_hooks/loading.py``).
_DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


class StandInTokenizer:
    """A tokenizer that gives each distinct string its own id, in first-seen order.

    It stands in for a model's tokenizer where the CPU tier has none, such as
    a gated Llama checkpoint. Every answer is one token under it, so the
    multi-token refusal of ``causalab.neural.shared.metrics`` never fires;
    that check needs the real tokenizer. An id that no string has taken
    decodes to a placeholder piece. A test passes one instance to every step
    of a workflow, because a real run has one tokenizer for all of them.
    """

    def __init__(self, size: int = VOCAB) -> None:
        self.size = size
        self._ids: dict[str, int] = {}
        self._pieces: dict[int, str] = {}

    def __len__(self) -> int:
        return self.size

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """The one id of ``text``; a new string takes the next free id.

        Raises:
            AssertionError: when every id is taken.
        """
        del add_special_tokens  # a stand-in has no special tokens
        if text not in self._ids:
            if len(self._ids) >= self.size:
                raise AssertionError(
                    f"StandInTokenizer holds {self.size} strings; raise VOCAB"
                )
            self._ids[text] = len(self._ids)
            self._pieces[self._ids[text]] = text
        return [self._ids[text]]

    def decode(self, ids: Iterable[int], **_: Any) -> str:
        """The pieces of ``ids``, joined."""
        return "".join(self._pieces.get(int(i), f" <{int(i)}>") for i in ids)


def fabricate_protocol_outputs(
    name: str,
    inner: InnerProtocol,
    files: Iterable[str],
    env: ResolutionEnv,
    out_dir: Path,
    tokenizer: StandInTokenizer | None = None,
) -> None:
    """Write the saved ``files`` of one protocol step into ``out_dir``.

    Walks the step's points in order, as the engine does, and writes each
    requested file once with every point's rows or entries. Also writes the
    step's ``_step.json`` with its sweep axes, which scripts group a table by
    (``causalab.io.step_record.axes_for``).

    Args:
        name: The step's name, for refusals.
        inner: The step as the workflow loaded it.
        files: The saved files to write, by their ``file_path``.
        env: Resolves the documents' datasets, and the model facts that
            size a featurizer bundle (``env.model_info``).
        out_dir: The step's directory in the run tree.
        tokenizer: Resolves answer strings; a fresh `StandInTokenizer` by
            default.

    Raises:
        AssertionError: when a requested file is saved by no entry, or by an
            entry this helper does not fabricate (module docstring).
    """
    tokenizer = tokenizer or StandInTokenizer()
    wanted = set(files)
    rows_of: dict[str, list[dict[str, Any]]] = {}
    tensor_files: dict[str, TensorFile] = {}
    metric_files: dict[str, MetricTable] = {}
    for point, doc in zip(inner.expansion.points, inner.point_documents, strict=True):
        dataset = str(input_roles(doc)["base"].dataset)
        if dataset not in rows_of:
            rows_of[dataset] = env.datasets.rows(dataset)
        rows = rows_of[dataset]
        labels = example_labels(rows)
        coords = dict(point.coords)
        reads = _Reads(doc, dataset, coords, len(rows), tokenizer)
        for entry in doc.save:
            if entry.file_path not in wanted:
                continue
            what = f"step {name!r}: {entry.file_path}"
            if entry.kind is not None:
                raise AssertionError(
                    f"{what}: this helper fabricates read, metric and featurizer "
                    f"entries only, not the derived {entry.kind!r} record"
                )
            if entry.read is None:
                _add_bundle(
                    tensor_files.setdefault(entry.file_path, TensorFile()),
                    doc,
                    entry,
                    dataset,
                    coords,
                    env.model_info(str(doc.model.key)),
                    what,
                )
                continue
            role = doc.group_of(entry.read)[1]
            if role != "base":
                raise AssertionError(
                    f"{what}: this helper fabricates reads on the 'base' input "
                    f"only, not on {role!r}"
                )
            if entry.aggregation is None:
                read_site = site_identity(doc, str(doc.reads[entry.read.read].site))
                tensor_files.setdefault(entry.file_path, TensorFile()).add(
                    entry.read.read,
                    reads.saved(entry.read, what),
                    coords,
                    reduce=entry.reduce,
                    # the fields the engine stamps on a read entry
                    # (`execution._execute_point`), less the backend a loaded
                    # model reports; an available cell adds no status fields
                    identity={
                        **build_artifact_identity(
                            model_attn_implementation=doc.model.attn_implementation
                        ),
                        "trained_on": dataset,
                        **(
                            {"site": json.dumps(read_site, sort_keys=True)}
                            if read_site
                            else {}
                        ),
                    },
                )
                continue
            spec = entry.aggregation
            kind = str(spec.kind)
            identity = metric_record_identity(
                kind, unit=spec.unit, estimand_version=spec.estimand_version
            )
            table = metric_files.setdefault(entry.file_path, MetricTable())
            budget = generated_budget(doc, doc.reads[entry.read.read].pos)
            if budget is None:
                target = spec.fields.get("target")
                values = compute_metric(
                    spec,
                    reads.scored(entry.read),
                    rows,
                    tokenizer,
                    target_value=(
                        reads.scored(target) if isinstance(target, ReadRef) else None
                    ),
                    vocab_axis=read_is_vocabulary(doc, entry.read.read),
                )
                table.add(entry.label, values, coords, identity=identity, labels=labels)
                continue
            if kind not in WHOLE_WINDOW_METRIC_KINDS:
                raise AssertionError(
                    f"{what}: this helper fabricates windowed "
                    f"{sorted(WHOLE_WINDOW_METRIC_KINDS)} metrics only, not {kind!r}"
                )
            generated = torch.randint(
                len(tokenizer),
                (len(rows), budget),
                generator=reads.generator(entry.read),
            ).tolist()
            table.add_windowed(
                entry.label,
                compute_windowed_metric(
                    spec, [], rows, tokenizer, generated_ids=generated
                ),
                coords,
                identity=identity,
                steps=None,
                matched=[bool(ids) for ids in generated],
                labels=labels,
            )
    missing = wanted - set(tensor_files) - set(metric_files)
    if missing:
        raise AssertionError(f"step {name!r}: no save entry writes {sorted(missing)}")
    first = inner.point_documents[0]
    model = canonical_model(first.raw["model"])
    # the file-level fields the engine stamps (`execution._publish`)
    write_outputs(
        out_dir,
        tensor_files,
        metric_files,
        identity_base={
            "model_key": str(first.model.key),
            "model_revision": str(first.model.revision),
            "model_dtype": str(model["dtype"]),
            "model_quantization": model.get("quantization"),
            "model_attn_implementation": model.get("attn_implementation"),
            "engine": "fabricated",
            "commit": runtime_identity().short_revision,
        },
    )
    put_sidecar(out_dir, [axis.id for axis in inner.compiled.axes])


@dataclasses.dataclass(frozen=True)
class _Reads:
    """The fabricated reads of one point (module docstring)."""

    doc: Document
    dataset: str
    coords: Mapping[str, Any]
    n_rows: int
    tokenizer: StandInTokenizer

    def saved(self, ref: ReadRef, what: str) -> torch.Tensor | RaggedValue:
        """A saved read: ``(rows, positions, width)``, or the flat gather
        and its widths when the rows address different numbers of positions."""
        generator = self.generator(ref)
        widths = self._positions(ref, generator, what)
        if len(set(widths)) == 1:
            return self._values((self.n_rows, widths[0]), ref, generator)
        return RaggedValue(
            flat=self._values((sum(widths),), ref, generator), widths=tuple(widths)
        )

    def scored(self, ref: ReadRef) -> torch.Tensor:
        """A read a metric reduces: one position per row, ``(rows, 1, width)``."""
        return self._values((self.n_rows, 1), ref, self.generator(ref))

    def generator(self, ref: ReadRef) -> torch.Generator:
        """A generator seeded by what the value of ``ref`` depends on."""
        return _generator(_read_seed(self.doc, ref, self.dataset, self.coords))

    def _values(
        self, lead: tuple[int, ...], ref: ReadRef, generator: torch.Generator
    ) -> torch.Tensor:
        read = self.doc.reads[ref.read]
        if isinstance(read.dims, tuple):
            width = len(read.dims)
        elif read_is_vocabulary(self.doc, ref.read):
            width = len(self.tokenizer)
        else:
            width = WIDTH
        dtype = str(canonical_model(self.doc.raw["model"])["dtype"])
        if dtype not in _DTYPES:
            raise AssertionError(f"no torch dtype for the model dtype {dtype!r}")
        return torch.randn((*lead, width), generator=generator).to(_DTYPES[dtype])

    def _positions(
        self, ref: ReadRef, generator: torch.Generator, what: str
    ) -> list[int]:
        """How many positions each row's read addresses: one for an
        ``index``, the set's size for a fixed set, and a seeded width from 1
        to `SPAN_MAX` where only the tokenizer can say."""
        spec = spec_of(self.doc, self.doc.reads[ref.read].pos)
        if spec.generated is not None:
            raise AssertionError(
                f"{what}: this helper does not fabricate a saved tensor at a "
                "generated position"
            )
        if spec.index is not None and not isinstance(spec, SpanSpec):
            return [1] * self.n_rows  # bare, scoped or relative: one token
        anchored = spec.scope is not None or spec.relative_to is not None
        if anchored and (
            spec.span is not None
            or (isinstance(spec, SpanSpec) and spec.indices is not None)
        ):
            # the engine's width here follows from the set, and under a scope
            # from the anchor's length too (`resolve_position`), so a seeded
            # width per row would be a ragged shape the engine does not write
            raise AssertionError(
                f"{what}: this helper does not fabricate a saved tensor at a span "
                "or indices set inside a scope or relative_to anchor"
            )
        fixed = static_indices(spec)
        if fixed is not None:
            return [len(fixed)] * self.n_rows
        return torch.randint(
            1, SPAN_MAX + 1, (self.n_rows,), generator=generator
        ).tolist()


def _add_bundle(
    bundle: TensorFile,
    doc: Document,
    entry: SaveEntry,
    dataset: str,
    coords: Mapping[str, Any],
    info: ModelInfo,
    what: str,
) -> None:
    """Add one point's trained featurizer to ``bundle`` as the engine's
    featurizer entry does (`execution._execute_point`): one entry per slot
    of the stage, keyed by the featurizer, then the file-level stamp. The
    values are seeded by the whole point, as a fit's are decided by it."""
    featurizer = str(entry.value)
    stage = _trained_stage(doc, featurizer, coords, info, what)
    slots = stage.slot_params()
    if not isinstance(stage, (Subspace, Gate)) or not slots:
        raise AssertionError(
            f"{what}: this helper fabricates a subspace or a gate, not a "
            f"{type(stage).__name__} stage"
        )
    # the fields the engine stamps on a bundle, less the backend a loaded
    # model reports
    identity = featurizer_identity(
        doc,
        featurizer,
        entry.site,
        stage=stage,
        group_map=stage.groups if isinstance(stage, Gate) else None,
    )
    for slot, param in slots.items():
        generator = _generator(
            json.dumps(
                [entry.file_path, featurizer, slot, dataset, coords],
                sort_keys=True,
                default=str,
            )
        )
        bundle.add(
            slot,
            _trained_value(stage, tuple(param.shape), generator).to(param.dtype),
            coords,
            label_entry=featurizer,
            identity=identity,
        )
    bundle.record_common(identity)


def _trained_stage(
    doc: Document,
    featurizer: str,
    coords: Mapping[str, Any],
    info: ModelInfo,
    what: str,
) -> Stage:
    """The stage the engine builds for ``featurizer``, through its own
    ``build_stack``, at the width it sizes it to
    (``executor.base._featurizer_input``): the site width of the first read
    or write whose chain uses it, carried through the stages before it."""
    spec = doc.featurizers[featurizer]
    if spec.axis is not None or spec.pool is not None:
        raise AssertionError(
            f"{what}: this helper does not fabricate a position gate or a pooled gate"
        )

    def refuse(*_: Any, **__: Any) -> Any:
        raise AssertionError(
            f"{what}: featurizer {featurizer!r} starts from a saved file, which "
            "this helper does not load"
        )

    for use in (*doc.reads.values(), *doc.writes.values()):
        ref = use.featurizer
        chain: tuple[str, ...] = (
            (ref,)
            if isinstance(ref, str)
            else tuple(ref)
            if isinstance(ref, tuple)
            else ()
        )
        if featurizer not in chain:
            continue
        site = doc.sites[str(use.site)]
        if site.head is not None or site.expert is not None:
            # the engine sizes a stage there from the resolved site's own
            # slice and shape, which the parity test does not cover
            raise AssertionError(
                f"{what}: this helper does not fabricate a featurizer at a site "
                "that selects a head or an expert"
            )
        component = str(site.component)
        width: int | None = component_width(info, component)
        for member in chain[: chain.index(featurizer)]:
            assert width is not None
            width = stage_output_width(doc.featurizers[member], width)
        if width is None:
            raise AssertionError(
                f"{what}: a stage before {featurizer!r} has no width the spec gives"
            )
        cache: dict[str, Stage] = {}
        build_stack(
            featurizer,
            dict(doc.featurizers),
            width=width,
            load_tensors=refuse,
            load_table=refuse,
            stage_cache=cache,
            seed=document_seed(doc),
            coords=coords,
            site_shape=component_shape(info, component),
            site_component=component,
            model_info=info,
        )
        return cache[featurizer]
    raise AssertionError(f"{what}: no read or write uses {featurizer!r}")


def _trained_value(
    stage: Stage, shape: tuple[int, ...], generator: torch.Generator
) -> torch.Tensor:
    """A seeded value in the set a trained slot lies in: orthonormal columns
    for a rotation, the unit interval for a ``clamp`` or ``boundary`` gate
    (projected there after every step, ``Gate.project``), any real number
    for the other gates."""
    if isinstance(stage, Subspace):
        return torch.linalg.qr(torch.randn(shape, generator=generator))[0]
    assert isinstance(stage, Gate)
    if stage.parametrization in ("clamp", "boundary"):
        return torch.rand(shape, generator=generator)
    return torch.randn(shape, generator=generator)


def _read_seed(
    doc: Document, ref: ReadRef, dataset: str, coords: Mapping[str, Any]
) -> str:
    """What the value of ``ref`` depends on, as one string (module docstring):
    its address on a model that lands no write, else the whole point."""
    model = doc.intervened_models[str(ref.model)]
    if not model.is_unwritten:
        return json.dumps([str(ref), dataset, coords], sort_keys=True, default=str)
    read = doc.reads[ref.read]
    if read.featurizer is None:
        featurizers: tuple[Any, ...] = ()
    elif isinstance(read.featurizer, tuple):
        featurizers = read.featurizer
    else:
        featurizers = (read.featurizer,)
    address = {
        "site": doc.sites[str(read.site)],
        "pos": doc.positions.get(read.pos, read.pos)
        if isinstance(read.pos, str)
        else read.pos,
        "featurizers": [doc.featurizers[str(name)] for name in featurizers],
        "dims": read.dims,
    }
    return json.dumps([str(ref), dataset, address], sort_keys=True, default=_plain)


def _plain(value: Any) -> Any:
    """A schema object as JSON: a dataclass by its fields, anything else by
    its text."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    return str(value)


def _generator(seed: str) -> torch.Generator:
    """A generator seeded by the sha256 of ``seed``, the same in every process."""
    return torch.Generator().manual_seed(
        int.from_bytes(hashlib.sha256(seed.encode()).digest()[:8], "little")
    )
