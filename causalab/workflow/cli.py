"""The CLI verbs for **workflow** documents (docs/workflow_protocol.md §9).

Split out of ``protocol/cli.py`` so the protocol package carries no workflow
code: someone who wants only the intervention protocol imports only that.
Dispatch between the two document types is [`causalab.cli`][].
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

from causalab.protocol.lockstep import SOLO as SOLO_LOCKSTEP
from causalab.protocol.parallel import ONE, ParallelGeometry, format_geometry
from causalab.protocol.publish import SOLO, is_joiner
from causalab.protocol.rules.errors import ProtocolError
from causalab.io.env import ResolutionEnv
from causalab.protocol.rules.data import check_data_columns

if TYPE_CHECKING:
    from causalab.protocol.pipeline import AnswerCheck

__all__ = ["check_geometry", "main"]


def check_geometry(geometry: ParallelGeometry) -> None:
    """The one axis a workflow run cannot serve (``docs/model_parallelism.md``
    §11): the runner is one process with one engine list, and a data-parallel
    geometry would need every replica to run a different point shard of each
    step — what ``fan_out.over.shards`` declares on a step instead. Refused
    before any rank is spawned or joined, on every rank alike.

    Raises:
        ProtocolError: ``P4`` naming ``--parallel.data``.
    """
    if geometry.data > 1:
        raise ProtocolError(
            "P4",
            f"--parallel {format_geometry(geometry)} asks for {geometry.data} "
            "data-parallel replicas, and a workflow runs its steps in one process "
            "with one engine list, so the data axis is not served under a workflow "
            "(docs/model_parallelism.md §11): shard a step with "
            "fan_out.over.shards, or run its document itself under dp",
            path="--parallel.data",
        )


def main(args: argparse.Namespace, env: ResolutionEnv) -> int:
    from causalab.protocol.pipeline import read_document
    from causalab.io.sources import apply_overrides as _apply
    from causalab.io.sources import load_text as _load_text
    from causalab.cli import register_model_key, wants_hf_registration
    from causalab.workflow.document import load_workflow

    if args.verb == "run":
        if getattr(args, "dtype", None) is not None:
            print(
                "refused: --dtype sets model.dtype on one intervention "
                "specification; a workflow's steps each declare their own "
                "realization — set it in the step's document, or with that "
                "step's own `set` block",
                file=sys.stderr,
            )
            return 1
        if args.points is not None:
            print(
                "refused: --points shards a single document's expanded "
                "campaign; a workflow schedules whole steps — shard the "
                "inner document runs instead",
                file=sys.stderr,
            )
            return 1

    if wants_hf_registration(args):
        # `run` touches models anyway, and `--register-from-hf` is the author
        # asking for it: pre-register **every inner** model key BEFORE loading,
        # because canonicalization derives widths from the registry. A workflow
        # names several documents, so registering only the outer one would
        # pre-flight nothing — which is why the three A3B runs' hand-rolled
        # wrappers had to be workflow-aware too.
        raw_wf = _apply(dict(_load_text(args.document)), dict(args.parsed_set))
        for doc_path, overrides in _inner_documents(
            raw_wf, args.document.resolve().parent, _load_text
        ):
            try:
                # the compiler's own read prefix (IM spec §9), so the
                # key this pre-registers is the one the compile will read
                inner = read_document(doc_path, doc_path.parent, overrides)
            except ProtocolError:
                continue
            register_model_key(dict(inner.raw))

    loaded = load_workflow(
        args.document.resolve(),
        env,
        overrides=dict(args.parsed_set),
    )
    if args.verb == "validate":
        if getattr(args, "data", False):
            for name in loaded.order:
                inner = loaded.inner.get(name)
                if inner is not None:
                    check_data_columns(inner.compiled, env)
        checked: dict[str, AnswerCheck] | None = None
        if getattr(args, "tokenizer", False):
            # the tokenizer passes of every static inner document, positions
            # and answers; a step-dependent one resolves at its step
            from causalab.workflow.runner import check_tokenization

            checked = check_tokenization(loaded, env, positions=True)
        print(f"OK: {args.document} — {len(loaded.document.steps)} steps")
        if checked is not None:
            deferred = [
                name
                for name in loaded.order
                if name in loaded.inner and loaded.inner_digest_kind[name] != "campaign"
            ]
            # a metric over generated tokens is scored on the rows that
            # generated a step, known only after the decode: its answers are
            # checked when scored, and the line names them apart
            when_scored = {
                name: list(check.when_scored)
                for name, check in checked.items()
                if check.when_scored
            }
            print(
                f"tokenizer: positions and metric answers resolve in {list(checked)}"
                + (
                    f"; the answers of {when_scored} are checked when scored, "
                    "because they are read over generated tokens"
                    if when_scored
                    else ""
                )
                + (
                    f"; {deferred} depend on earlier steps and resolve at their step"
                    if deferred
                    else ""
                )
            )
        return 0
    if args.verb == "digest":
        # the identities `--resume` compares, one per step in schedule order
        # (§7): a script, behavioral, decision, conditional or fanned-out step's
        # entry digest; an unfanned protocol step's inner document digest.
        # There is no whole-workflow digest to print — nothing compares one.
        for name in loaded.order:
            identity = loaded.step_digests.get(name) or loaded.inner_digests[name]
            print(f"{name}  {identity}")
        return 0
    if args.verb == "explain":
        print(f"schedule  {len(loaded.levels)} levels")
        for i, level in enumerate(loaded.levels):
            print(f"  level {i}: {', '.join(level)}")
        _explain_steps(loaded, "  ")
        if loaded.nondeterministic:
            # §7: explain names the steps that make a run unreplayable, so the
            # gap is visible before anyone trusts a rerun
            print(
                "not replayable: "
                + ", ".join(loaded.nondeterministic)
                + " (is_deterministic: false)"
            )
        if loaded.unchecked_paths:
            # rule 4: an absolute path is not existence-checked at load,
            # because validation and execution routinely run on different hosts
            print("unchecked absolute paths (verified at run time):")
            for item in loaded.unchecked_paths:
                print(f"  {item}")
        return 0
    # run — the one verb that constructs the engine: a lazily-imported extra;
    # --engine named it, and every protocol step is held to it
    from causalab.neural.shared.engine_router import route
    from causalab.workflow import run_workflow

    # this process's place in a launched world (docs/model_parallelism.md
    # §3, §11; causalab.cli sets both for a world above 1): the publisher every
    # step's run context carries, and the lockstep the joiner's decisions
    # travel on. World 1 — SOLO — publishes, joins, and decides for itself
    publisher = getattr(args, "publisher", SOLO)
    lockstep = getattr(args, "lockstep", SOLO_LOCKSTEP)
    joins = is_joiner(publisher)
    result = run_workflow(
        loaded,
        env,
        args.out,
        route(
            args.engine,
            device=args.device,
            cuda_graphs=getattr(args, "cuda_graphs", False),
            batch_rows=getattr(args, "batch_rows", None),
            fit_rows=getattr(args, "fit_rows", None),
            # the same geometry as a document run's (docs/model_parallelism.md
            # §2): every rank builds the same engine list
            parallel=getattr(args, "parallel_geometry", ONE),
        ),
        resume=getattr(args, "resume", False),
        reuse_nondeterministic=getattr(args, "reuse_nondeterministic", False),
        publisher=publisher,
        lockstep=lockstep,
    )
    if not joins:
        # a rank that does not publish wrote nothing and says nothing; the
        # joiner's lines below are the run's
        return 0
    for name, entry in sorted(result.manifest["steps"].items()):
        files = ", ".join(entry.get("files", ()))
        print(f"{entry.get('status', 'completed')} {name}: {files}")
    print(f"manifest {result.run_root / 'workflow.json'}")
    return 0


def _inner_documents(
    raw_wf: Any,
    workflow_dir: Path,
    load_text: Any,
    *,
    outer_set: Any = None,
    seen: tuple[Path, ...] = (),
) -> Iterator[tuple[Path, dict[str, Any]]]:
    """Every intervention specification a workflow names, with the ``set``
    the compile will read — through nested ``workflow`` steps too (§2.10),
    whose ``set`` is the nested form laid over the inner step's own. Malformed
    shapes and loops are left to ``load_workflow`` to refuse properly."""
    steps_raw = raw_wf.get("steps", {}) if isinstance(raw_wf, dict) else {}
    if not isinstance(steps_raw, dict):
        return
    laid_over = outer_set if isinstance(outer_set, dict) else {}
    for name, step_raw in steps_raw.items():
        if not isinstance(step_raw, dict):
            continue
        document = step_raw.get("document")
        if not isinstance(document, str):
            continue
        doc_path = (workflow_dir / document).resolve()
        if not doc_path.is_file():
            continue
        authored = step_raw.get("set", {}) or {}
        authored = dict(authored) if isinstance(authored, dict) else {}
        kind = step_raw.get("type")
        if kind in ("intervention_protocol", "behavioral"):
            laid = laid_over.get(str(name), {})
            yield doc_path, {**authored, **(laid if isinstance(laid, dict) else {})}
        elif kind == "workflow" and doc_path not in seen:
            try:
                inner_raw = dict(load_text(doc_path))
            except ProtocolError:
                continue
            yield from _inner_documents(
                inner_raw,
                doc_path.parent,
                load_text,
                outer_set=authored,
                seen=(*seen, doc_path),
            )


def _explain_steps(loaded: Any, indent: str) -> None:
    """One line per step of the derived order (§9); a nested workflow's steps
    (§2.10) under their ``workflow`` step's own line, indented two more
    spaces and in their own order — so depth reads as indentation."""
    from causalab.workflow import fan_out, nested
    from causalab.workflow.document import (
        BehavioralStep,
        ConditionalStep,
        DecisionStep,
        ProtocolStep,
        WorkflowStep,
    )

    shown: set[str] = set()
    for name in loaded.order:
        head = name.split(nested.SEPARATOR, 1)[0]
        container = loaded.document.steps.get(head)
        if isinstance(container, WorkflowStep):
            if head not in shown:
                shown.add(head)
                print(
                    f"{indent}{nested.describe(head, container, loaded.nested[head])}"
                )
                _explain_steps(loaded.nested[head], indent + "  ")
            continue
        step = loaded.document.steps[name]
        # §2.9: a fanned-out step reports its width and join; each child
        # its shard — the schedule above already lists them as steps
        shard = getattr(step, "shard", None)
        fanned = (
            f" — {fan_out.describe(step, loaded.children[name])}"
            if isinstance(step, (ProtocolStep, BehavioralStep))
            and step.fan_out is not None
            else ""
        )
        sharded = (
            f" — shard {shard['index']} of {shard['of']}: "
            f"{len(shard['points'])} point(s)"
            if isinstance(shard, dict)
            else ""
        )
        if isinstance(step, ProtocolStep):
            inner = loaded.inner[name]
            kind = loaded.inner_digest_kind[name]
            print(
                f"{indent}{name}: intervention_protocol {step.document} — "
                f"{len(inner.expansion.points)} point(s), "
                f"{kind} digest {loaded.inner_digests[name][:16]}…"
                f"{fanned}{sharded}"
            )
        elif isinstance(step, BehavioralStep):
            inner = loaded.inner[name]
            print(
                f"{indent}{name}: behavioral {step.document} — "
                f"{len(inner.expansion.points)} point(s), "
                f"decoding {step.decoding['mode']}, split {step.split}, "
                f"step digest {loaded.step_digests[name][:16]}…"
                f"{fanned}{sharded}"
            )
        elif isinstance(step, DecisionStep):
            print(
                f"{indent}{name}: decision over {step.values.target} — rule on "
                f"{', '.join(sorted(step.rule))}, "
                f"step digest {loaded.step_digests[name][:16]}…"
            )
        elif isinstance(step, ConditionalStep):
            comparator = next(c for c in ("eq", "ne", "in") if c in step.predicate)
            print(
                f"{indent}{name}: conditional on "
                f"{step.predicate['decision']['step']}.{step.predicate['field']} "
                f"{comparator} {step.predicate[comparator]!r} -> "
                f"on_true [{', '.join(step.on_true)}] / "
                f"on_false [{', '.join(step.on_false)}] (scope {step.scope})"
            )
        else:
            marks = []
            if not step.is_deterministic:
                marks.append("non-deterministic")
            if step.runtime and step.runtime.get("isolate"):
                marks.append("isolated")
            suffix = f" [{', '.join(marks)}]" if marks else ""
            print(
                f"{indent}{name}: script {step.script} -> "
                f"{', '.join(sorted(d.file for d in step.outputs.values()))}"
                f"{suffix}"
            )
