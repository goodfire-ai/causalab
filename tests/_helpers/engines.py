"""Drive one document through **both** execution engines and compare.

The two engines share everything above ``_run_group`` — position resolution,
featurizer stacks, mechanism arithmetic, ragged handling, aggregation
lowering, receipts (``causalab/neural/shared/executor/``, ``execution.py``) — so
a parity test is almost always "the reference engine's existing document,
second executor". This module is the one place that spells that out:

* ``executor_for`` / ``both_executors`` — the executor seam. Rows are
  handed to the executor directly (its own seam) while the document still
  parses and validates through the real loader path. The reference tests'
  ``tests/neural/engines/pytorch_hooks/_drive.py`` is a thin wrapper.
* ``run_both`` — the engine seam: one compiled document through
  ``run_protocol`` with ``PytorchHooksEngine()`` and with ``NnsightEngine()``,
  two output directories. Both runs pass ``record=True`` by default, so the
  run receipt is part of the comparison (receipts are opt-in).
* ``compare_run_dirs`` — walks both directories: tensors ``allclose`` at
  ``ATOL``, JSON tables row by row (floats at ``ATOL``, everything
  else exact), the run receipt whole after ``KNOWN_RECORD_DIFFERENCES``
  are removed, safetensors ``__metadata__`` after ``KNOWN_METADATA_DIFFERENCES``.

The allowed differences live in ``KNOWN_RECORD_DIFFERENCES``,
``KNOWN_METADATA_DIFFERENCES`` and ``SKIPPED_FILES``, one comment per
entry. Anything not listed there is a parity claim.
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch

from causalab.neural.engines.pytorch_hooks.engine import PytorchHooksEngine
from causalab.neural.engines.pytorch_hooks.executor import PointExecutor
from causalab.io.env import FileArtifacts, FileDatasets, ResolutionEnv
from causalab.protocol import RUN_RECORD_NAME, run_protocol
from causalab.protocol.compiled import CompiledProtocol
from causalab.protocol.engine import RunResult
from causalab.protocol.pipeline import compile_protocol
from causalab.protocol.rules.document import validate_document
from causalab.protocol.schema import parse_document

from tests.protocol._docs import in_order
from tests.protocol._env import FIXTURES

__all__ = [
    "ATOL",
    "BothRuns",
    "KNOWN_ENTRY_RECORD_DIFFERENCES",
    "KNOWN_METADATA_DIFFERENCES",
    "KNOWN_RECORD_DIFFERENCES",
    "SKIPPED_FILES",
    "both_executors",
    "bundle_loader",
    "compile_for_both",
    "compare_json",
    "compare_records",
    "compare_run_dirs",
    "compare_safetensors",
    "corpus_env",
    "executor_for",
    "raising_loader",
    "run_both",
    "strip_known_record_differences",
]

#: Reads and metric values agree to this tolerance when both engines run fp32
#: eager on one device: the same kernels in a different order of capture, so
#: anything larger is an executor bug rather than float noise.
ATOL = 1e-5


# --------------------------------------------------------------------------- #
# the executor seam
# --------------------------------------------------------------------------- #


def raising_loader(path: str) -> Any:
    """The ``load_tensors`` a document without artifacts gets: any request is
    a test bug, named by path, rather than ``None`` flowing into a stack."""
    raise KeyError(path)


def bundle_loader(files: Mapping[str, Mapping[str, Any]]) -> Callable[[str], Any]:
    """A ``load_tensors`` over in-memory bundles: path -> {slot: tensor}.

    Tests that hand-build bundles carry no ``entries`` table, which is the
    same shape an external (hand-made) artifact has — selection then falls
    back to the entry keys themselves. ``TensorBundle`` is the shared type
    both executors accept through ``load_tensors``."""
    from causalab.io.tensor_files import TensorBundle

    def load(path: str) -> TensorBundle:
        return TensorBundle(tensors=dict(files[path]), entry_coords={})

    return load


def executor_for(
    executor_cls: type,
    doc_raw: Mapping[str, Any],
    bundle: Any,
    *,
    base_texts: Sequence[str],
    counterfactual_texts: Sequence[str] | None = None,
    extra_columns: Mapping[str, Sequence[Any]] | None = None,
    load_tensors: Callable[[str], Any] | None = None,
    load_table: Callable[[str], Any] | None = None,
    **executor_kwargs: Any,
) -> Any:
    """``executor_cls`` over ``doc_raw`` on ``bundle``, rows built from the
    texts. ``extra_columns`` are per-row columns (``{column: [value per row]}``)
    a position selector or a metric may name. ``executor_kwargs`` reach the
    executor's constructor unchanged (``grad_enabled``, ``interning``,
    ``batch_rows`` — the reference engine's seams)."""
    doc = parse_document(in_order(dict(doc_raw)))
    validate_document(doc, engine_is_local=True)
    rows: list[dict[str, Any]] = []
    for i, text in enumerate(base_texts):
        row: dict[str, Any] = {"input": text}
        if counterfactual_texts is not None:
            row["counterfactual_inputs"] = [counterfactual_texts[i]]
        for column, values in (extra_columns or {}).items():
            row[column] = values[i]
        rows.append(row)
    role_rows: dict[str, list[dict[str, Any]]] = {"base": rows}
    role_fields = {"base": "input"}
    if counterfactual_texts is not None:
        role_rows["counterfactual"] = rows
        role_fields["counterfactual"] = "counterfactual_inputs[0]"
    return executor_cls(
        doc,
        bundle,
        role_rows=role_rows,
        role_fields=role_fields,
        load_tensors=load_tensors or raising_loader,
        load_table=load_table,
        **executor_kwargs,
    )


def both_executors(
    doc_raw: Mapping[str, Any], hooks_bundle: Any, trace_bundle: Any, **kwargs: Any
) -> tuple[PointExecutor, Any]:
    """``(reference executor, nnsight executor)`` over one document — the
    shape every executor-level parity test takes. ``kwargs`` are
    ``executor_for``'s. The nnsight import is function-local so the
    reference tests' ``_drive`` can wrap ``executor_for`` without the
    ``nnsight`` extra."""
    from causalab.neural.engines.nnsight_tracing.executor import TracePointExecutor

    hooks = executor_for(PointExecutor, doc_raw, hooks_bundle, **kwargs)
    trace = executor_for(TracePointExecutor, doc_raw, trace_bundle, **kwargs)
    return hooks, trace


# --------------------------------------------------------------------------- #
# the engine seam
# --------------------------------------------------------------------------- #


def corpus_env(artifacts_root: Path) -> ResolutionEnv:
    """The corpus documents' resolution environment: the committed fixture
    tables for datasets, ``artifacts_root`` (a fresh copy of the fixture
    artifacts, so a run may write beside them) for artifacts."""
    import shutil

    shutil.copytree(FIXTURES / "artifacts", artifacts_root, dirs_exist_ok=True)
    return ResolutionEnv(
        datasets=FileDatasets(root=FIXTURES / "data"),
        artifacts=FileArtifacts(root=artifacts_root),
    )


@dataclasses.dataclass(frozen=True)
class BothRuns:
    """One document run through each engine: where each wrote, what each
    returned."""

    hooks_dir: Path
    trace_dir: Path
    hooks_result: RunResult
    trace_result: RunResult

    def compare(self, **kwargs: Any) -> None:
        compare_run_dirs(self.hooks_dir, self.trace_dir, **kwargs)


def compile_for_both(
    document: Path | Mapping[str, Any],
    env: ResolutionEnv,
    *,
    overrides: Mapping[str, Any] | None = None,
) -> CompiledProtocol:
    """``document`` compiled once for both engines. ``overrides`` are
    ``--set``-style dotted paths, the way the corpus documents are retargeted
    onto the tiny fixtures. The model the document names is registered in the
    protocol registry first, as the CLI does, so the compile does not depend
    on a fixture having loaded it already."""
    from causalab.cli import register_model_key

    raw = (
        json.loads(Path(document).read_text())
        if isinstance(document, Path)
        else dict(document)
    )
    model_key = (overrides or {}).get("model.key") or raw.get("model", {}).get("key")
    register_model_key({"model": {"key": model_key, "revision": "main"}})
    return compile_protocol(document, env=env, overrides=dict(overrides or {}))


def run_both(
    document: CompiledProtocol | Path | Mapping[str, Any],
    env: ResolutionEnv,
    out: Path,
    *,
    overrides: Mapping[str, Any] | None = None,
    record: bool = True,
    hooks_engine: PytorchHooksEngine | None = None,
    trace_engine: Any = None,
) -> BothRuns:
    """``run_protocol`` on ``document`` through the reference engine into
    ``out / "hooks"`` and through the nnsight engine into ``out / "trace"``.

    A path or a raw tree is compiled here through ``compile_for_both``;
    a [`CompiledProtocol`][causalab.protocol.compiled.CompiledProtocol] is run
    as given. ``record`` reaches both runs: on (the default) each engine
    writes its receipt and event stream, so the comparison covers them.
    """
    from causalab.neural.engines.nnsight_tracing.engine import NnsightEngine

    compiled = (
        document
        if isinstance(document, CompiledProtocol)
        else compile_for_both(document, env, overrides=overrides)
    )
    hooks_dir, trace_dir = out / "hooks", out / "trace"
    hooks_result = run_protocol(
        compiled, env, hooks_engine or PytorchHooksEngine(), hooks_dir, record=record
    )
    trace_result = run_protocol(
        compiled, env, trace_engine or NnsightEngine(), trace_dir, record=record
    )
    return BothRuns(hooks_dir, trace_dir, hooks_result, trace_result)


# --------------------------------------------------------------------------- #
# the comparer
# --------------------------------------------------------------------------- #

#: Files the comparison skips outright, by name.
SKIPPED_FILES: frozenset[str] = frozenset(
    {
        # the event stream is a timestamped sidecar (workflow spec §4.3), an
        # input to nothing — the one file two runs of one document may differ in
        "events.jsonl",
    }
)

#: Files only the reference engine writes: present on one side, absent on the
#: other, and that asymmetry is the accepted state rather than a defect.
HOOKS_ONLY_FILES: frozenset[str] = frozenset(
    {
        # the nnsight engine does no sampled decoding and writes no
        # continuations (the decoding request's behavioral side is n/a there)
        "continuations.json",
    }
)

#: Dotted paths into the run receipt (``protocol.json``) removed from both
#: sides before the whole-record comparison — each one an engine property the
#: receipt records rather than a claim about the document.
KNOWN_RECORD_DIFFERENCES: tuple[str, ...] = (
    # the nnsight engine runs one batch per forward group and measures no
    # rows-per-forward bound: null in its receipt, a number in the reference's
    "execution.batch_rows",
    "execution.fit_rows",
    # 📐 FINDING (2026-09-18): the nnsight executor
    # keeps no per-group fire tally — `ExecutorBase.fires` stays empty, so
    # `record_fires` writes no `fires` block for it, while the reference
    # engine records `{point: {"<model> on <input>": {write: count}}}`. The
    # receipt asymmetry is asserted as the current state by
    # tests/neural/engines/nnsight_tracing/test_parity_harness.py and
    # test_parity_receipts.py — remove this entry when the nnsight executor
    # tallies fires.
    "fires",
)

#: Keys removed from a saved tensor file's ``__metadata__`` before comparison.
KNOWN_METADATA_DIFFERENCES: frozenset[str] = frozenset(
    {
        # the engine's own name — the one field the identity stamps by engine
        "engine",
    }
)

#: Keys removed from every record of a saved tensor file's ``entries`` table
#: (``__metadata__["entries"]``, a JSON object of entry -> record) before
#: comparison.
KNOWN_ENTRY_RECORD_DIFFERENCES: frozenset[str] = frozenset(
    {
        # 📐 FINDING (2026-09-18): the attention backend
        # the engine's loader realized, stamped on every entry as runtime
        # provenance (`ARTIFACT_IDENTITY_KEYS`). The reference loader pins
        # `eager`; the nnsight loader keeps the checkpoint's own default
        # (`sdpa` on the tiny fixtures) and switches to eager on demand,
        # so the two engines stamp different backends for the same document
        # while the saved numbers agree at `ATOL`. Asserted as the current
        # state by tests/neural/engines/nnsight_tracing/
        # test_parity_receipts.py and test_parity_positions.py.
        "loaded_attn_implementation",
    }
)


def _pop_path(tree: dict[str, Any], dotted: str) -> None:
    node: Any = tree
    parts = dotted.split(".")
    for part in parts[:-1]:
        if not isinstance(node, dict) or part not in node:
            return
        node = node[part]
    if isinstance(node, dict):
        node.pop(parts[-1], None)


def strip_known_record_differences(
    record: Mapping[str, Any], extra: Sequence[str] = ()
) -> dict[str, Any]:
    """A deep copy of ``record`` without ``KNOWN_RECORD_DIFFERENCES``
    (and ``extra``, a test's own additions — each one should say why)."""
    out = json.loads(json.dumps(record))
    for dotted in (*KNOWN_RECORD_DIFFERENCES, *extra):
        _pop_path(out, dotted)
    return out


def compare_json(a: Any, b: Any, what: str, *, atol: float = ATOL) -> None:
    """Structural equality with floats at ``atol``, everything else exact —
    ints stay ints (token ids and ranks are labels, not measurements). A
    string cell holding a JSON object or list is compared decoded."""
    if isinstance(a, dict) and isinstance(b, dict):
        assert sorted(a) == sorted(b), f"{what}: keys {sorted(a)} != {sorted(b)}"
        for key in a:
            compare_json(a[key], b[key], f"{what}.{key}", atol=atol)
        return
    if isinstance(a, list) and isinstance(b, list):
        assert len(a) == len(b), f"{what}: {len(a)} rows != {len(b)} rows"
        for i, (x, y) in enumerate(zip(a, b)):
            compare_json(x, y, f"{what}[{i}]", atol=atol)
        return
    if (
        isinstance(a, float)
        and isinstance(b, (float, int))
        and not isinstance(b, bool)
        or isinstance(b, float)
        and isinstance(a, (float, int))
        and not isinstance(a, bool)
    ):
        if math.isnan(a) or math.isnan(b):
            assert math.isnan(a) and math.isnan(b), f"{what}: {a!r} != {b!r}"
            return
        assert abs(float(a) - float(b)) <= atol, (
            f"{what}: {a!r} != {b!r} (abs diff {abs(float(a) - float(b)):.3e} > {atol})"
        )
        return
    if isinstance(a, str) and isinstance(b, str) and a != b:
        # a structured metric value (`top_k`, `token_logits`, `class_probs`)
        # is a JSON object encoded into the table's `value` cell: compare it
        # decoded, so its floats meet the same tolerance as a bare cell's
        try:
            ja, jb = json.loads(a), json.loads(b)
        except ValueError:
            ja = jb = None
        if isinstance(ja, (dict, list)) and isinstance(jb, (dict, list)):
            compare_json(ja, jb, f"{what}<decoded>", atol=atol)
            return
    assert type(a) is type(b) and a == b, f"{what}: {a!r} != {b!r}"


def compare_records(
    hooks_record: Mapping[str, Any],
    trace_record: Mapping[str, Any],
    *,
    extra_ignored: Sequence[str] = (),
    atol: float = ATOL,
) -> None:
    """The run receipt whole — campaign digest, canonical document, per-point
    digests, ``execution``, ``scoring``, ``fires`` — after
    the known differences are removed from both."""
    compare_json(
        strip_known_record_differences(hooks_record, extra_ignored),
        strip_known_record_differences(trace_record, extra_ignored),
        RUN_RECORD_NAME,
        atol=atol,
    )


def compare_safetensors(a: Path, b: Path, what: str, *, atol: float = ATOL) -> None:
    """Every tensor by key (shape, dtype, values — integers exact) and the
    ``__metadata__`` header without ``KNOWN_METADATA_DIFFERENCES``."""
    from causalab.io.tensor_files import load_file, safe_open

    ta, tb = load_file(str(a)), load_file(str(b))
    assert sorted(ta) == sorted(tb), f"{what}: tensor keys {sorted(ta)} != {sorted(tb)}"
    for key in ta:
        x, y = ta[key], tb[key]
        assert x.shape == y.shape, (
            f"{what}[{key}]: {tuple(x.shape)} != {tuple(y.shape)}"
        )
        assert x.dtype == y.dtype, f"{what}[{key}]: {x.dtype} != {y.dtype}"
        if not x.dtype.is_floating_point:
            assert torch.equal(x, y), f"{what}[{key}]: integer values differ"
            continue
        diff = (x.double() - y.double()).abs().max().item() if x.numel() else 0.0
        assert torch.allclose(x, y, atol=atol, rtol=0), (
            f"{what}[{key}]: max abs diff {diff:.3e} exceeds {atol}"
        )
    with (
        safe_open(str(a), framework="pt") as fa,
        safe_open(str(b), framework="pt") as fb,
    ):
        ma, mb = dict(fa.metadata() or {}), dict(fb.metadata() or {})
    for key in KNOWN_METADATA_DIFFERENCES:
        ma.pop(key, None)
        mb.pop(key, None)
    for key in sorted(set(ma) | set(mb)):
        assert key in ma and key in mb, f"{what}: metadata key {key!r} on one side only"
        va, vb = ma[key], mb[key]
        try:
            ja, jb = json.loads(va), json.loads(vb)
        except (ValueError, TypeError):
            assert va == vb, f"{what}.__metadata__.{key}: {va!r} != {vb!r}"
            continue
        if key == "entries" and isinstance(ja, dict) and isinstance(jb, dict):
            for table in (ja, jb):
                for record in table.values():
                    if isinstance(record, dict):
                        for field in KNOWN_ENTRY_RECORD_DIFFERENCES:
                            record.pop(field, None)
        compare_json(ja, jb, f"{what}.__metadata__.{key}")


def _files(root: Path) -> dict[str, Path]:
    return {
        str(p.relative_to(root)): p
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.name not in SKIPPED_FILES
    }


def compare_run_dirs(
    hooks_dir: Path,
    trace_dir: Path,
    *,
    atol: float = ATOL,
    extra_ignored_record_fields: Sequence[str] = (),
) -> None:
    """Both output directories, file by file: the same set of files (modulo
    ``HOOKS_ONLY_FILES``), tensors at ``atol``, JSON tables row by row,
    the receipt through ``compare_records``, anything else byte-equal."""
    a, b = _files(hooks_dir), _files(trace_dir)
    only_a = {k for k in set(a) - set(b) if Path(k).name not in HOOKS_ONLY_FILES}
    only_b = set(b) - set(a)
    assert not only_a and not only_b, (
        f"file sets differ: only pytorch_hooks wrote {sorted(only_a)}, "
        f"only nnsight wrote {sorted(only_b)}"
    )
    for rel in sorted(set(a) & set(b)):
        pa, pb = a[rel], b[rel]
        if pa.name == RUN_RECORD_NAME:
            compare_records(
                json.loads(pa.read_text()),
                json.loads(pb.read_text()),
                extra_ignored=extra_ignored_record_fields,
                atol=atol,
            )
        elif pa.suffix == ".safetensors":
            compare_safetensors(pa, pb, rel, atol=atol)
        elif pa.suffix == ".json":
            compare_json(
                json.loads(pa.read_text()), json.loads(pb.read_text()), rel, atol=atol
            )
        else:
            assert pa.read_bytes() == pb.read_bytes(), f"{rel}: bytes differ"
