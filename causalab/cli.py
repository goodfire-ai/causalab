"""CLI for intervention specifications, workflows and measurements.

Documents with ``steps`` dispatch to [`causalab.workflow.cli`][]; intervention
specifications dispatch to [`causalab.protocol.reports`][]. Measurement commands use
[`causalab.measurement`][]. Argument parsing and resolution are shared.
Dispatch stays here so ``protocol/`` remains independent of ``workflow/``.
Execution engines load lazily so validation and other pure verbs stay torch-free.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

from causalab.neural.shared.engine_router import ENGINE_CHOICES, route_name
from causalab.protocol.parallel import ONE, ParallelGeometry, parse_geometry
from causalab.protocol.rules.errors import ProtocolError
from causalab.io.env import ResolutionEnv, file_env
from causalab.protocol.schema import PRECISION_DTYPES

from causalab.tasks import TASKS_ROOT

__all__ = ["ensure_model_registered", "main", "register_model_key"]

logger = logging.getLogger(__name__)

#: The one logger ``--verbose`` enables: the shared execution loop's progress
#: lines ([`causalab.neural.shared.execution`][]). The flag configures this
#: name, never the root logger, so no other module gains output from it.
VERBOSE_LOGGER = "causalab.neural.shared.execution"


def _configure_verbose(stream=sys.stderr) -> None:
    """Attach one stderr handler to [`VERBOSE_LOGGER`][] at INFO, and raise
    Hugging Face Hub's own logger to INFO so its download progress and
    ``Still waiting to acquire lock`` lines show. Calling this twice adds no
    second handler: the CLI's ``main`` runs in-process more than once in
    tests, and a second run should print each line once."""
    log = logging.getLogger(VERBOSE_LOGGER)
    log.setLevel(logging.INFO)
    if not any(getattr(h, "_causalab_verbose", False) for h in log.handlers):
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S"))
        handler._causalab_verbose = True  # pyright: ignore[reportAttributeAccessIssue]
        log.addHandler(handler)
    # `huggingface_hub` installs its own stderr handler and sets its level
    # from HF_HUB_VERBOSITY at import; import it here so this call comes
    # after that and is not overwritten when the engine imports it later
    from huggingface_hub.utils import logging as hf_logging

    hf_logging.set_verbosity_info()


def _parse_set(values: Sequence[str]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for item in values:
        if "=" not in item:
            raise SystemExit(f"--set takes path=value, got {item!r}")
        dotted, _, raw_value = item.partition("=")
        try:
            overrides[dotted] = json.loads(raw_value)
        except json.JSONDecodeError:
            overrides[dotted] = raw_value  # a bare word is a string
    return overrides


def _overrides(args: argparse.Namespace) -> dict[str, Any]:
    """``--set`` overrides plus the ``--dtype`` shorthand, which is one of
    them: dtype belongs to the document, so the only way to change it from
    the command line is the way every other field changes (§9)."""
    overrides = _parse_set(args.set)
    dtype = getattr(args, "dtype", None)
    if dtype is None:
        return overrides
    already = overrides.get("model.dtype")
    if already is not None and already != dtype:
        raise SystemExit(
            f"--dtype {dtype} contradicts --set model.dtype={already} — "
            "they set the same field"
        )
    overrides["model.dtype"] = dtype
    return overrides


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="causalab",
        description="Intervention specifications and workflows: run, validate, "
        "explain, dry-run, digest, migrate (docs/intervention_protocol.md, "
        "docs/workflow_protocol.md).",
    )
    sub = parser.add_subparsers(dest="verb", required=True)
    measure = sub.add_parser(
        "measure",
        help="measure and profile a workflow at one or two source revisions",
        description="Measure workflows on one CPU or CUDA device with one rank. "
        "Multi-GPU profiling artifact collection is not supported.",
    )
    measure.add_argument("document", type=Path)
    measure.add_argument("--bindings", type=Path, required=True)
    measure.add_argument("--out", type=Path, required=True)
    measure.add_argument("--resume", action="store_true")
    migrate = sub.add_parser(
        "migrate",
        help="Update older JSON specifications and embedded Markdown examples to the current format.",
    )
    migrate.add_argument(
        "paths",
        type=Path,
        nargs="+",
        help="JSON documents or Markdown files to update. Convert YAML documents manually.",
    )
    migrate.add_argument(
        "--check",
        action="store_true",
        help="Check for required changes; exit 1 if a file needs an update.",
    )
    for verb, help_text in (
        ("run", "validate, expand, plan, execute, stamp"),
        ("validate", "the §5 load-error checklist"),
        (
            "explain",
            "models, forward plan, point count, requires, digest, save products",
        ),
        (
            "dry-run",
            "resolve everything a run decides before weights load, and report it",
        ),
        ("digest", "the campaign digest"),
    ):
        p = sub.add_parser(verb, help=help_text)
        p.add_argument(
            "document",
            type=Path,
            help="Intervention or workflow document in JSON or YAML.",
        )
        p.add_argument("--set", action="append", default=[], metavar="PATH=VALUE")
        p.add_argument(
            "--data-root",
            type=Path,
            default=TASKS_ROOT,
            help="Dataset root for <ref>.json references. Defaults to the installed task data.",
        )
        p.add_argument("--artifacts-root", type=Path, default=Path("."))
        p.add_argument(
            "--max-points",
            type=int,
            default=None,
            help="Maximum number of points in a sweep.",
        )
        p.add_argument(
            "--register-from-hf",
            action="store_true",
            help="Read an unregistered model configuration from Hugging Face. The run command does this automatically.",
        )
        if verb == "validate":
            p.add_argument(
                "--data",
                action="store_true",
                help="also check the resolved tables — column references, "
                "thresholds, row roles, fit splits (the default; kept for "
                "compatibility, it changes nothing)",
            )
        if verb in ("run", "validate", "explain", "dry-run"):
            p.add_argument(
                "--engine",
                choices=ENGINE_CHOICES,
                required=True,
                help="Execution engine. 'auto' selects pytorch_hooks. Validation and reports check registered capabilities before an engine loads.",
            )
        if verb == "dry-run":
            p.add_argument(
                "--data",
                action="store_true",
                help="also check column and prompt-variable references and the "
                "declared row roles at every point (the `validate --data` pass); "
                "a refusal is reported and exits 1",
            )
        if verb in ("validate", "dry-run"):
            # each verb its own text: only `validate` takes a workflow
            scope = (
                ". On a workflow, check every inner document that does not "
                "depend on an earlier step"
                if verb == "validate"
                else ""
            )
            p.add_argument(
                "--tokenizer",
                action="store_true",
                help="load the model's tokenizer, never its weights, and check "
                "token positions and every metric's answer tokens as a run "
                "checks them before the weights load; a refusal exits 1. The "
                "tokenizer loads as a run loads it: from the Hugging Face "
                "cache, or a download of its files when the Hub is reachable" + scope,
            )
        if verb in ("run", "dry-run"):
            p.add_argument(
                "--parallel",
                default=None,
                metavar="AXES",
                help="the reference engine's parallel geometry, `tp=4,ep=8` over "
                "the axes dp, pp, cp, tp, ep (data, pipeline, context, tensor, "
                "expert), each 1 unless named (docs/model_parallelism.md §2). "
                "The data axis has two modes (§8.3): `dp=2` (or `dp=2:points`) "
                "shards the campaign's points across the replicas, exact by "
                "construction; `dp=2:rows` runs every point on every replica "
                "and splits each fit minibatch's rows across them, the loss "
                "mean and the featurizer gradient averaged — a train document "
                "only, with train.batch.pairs at least dp. "
                "Execution, never identity: digests and stamps are unaffected "
                "and the run receipt records it as execution.parallel. "
                "`dry-run` checks the geometry against the model's registry "
                "entry and the rows mode against the document before any "
                "weights; `run` at a world above 1 spawns or joins the ranks — "
                "for a workflow too, whose steps every rank then runs in "
                "lockstep with the joiner alone writing the ROOT (§11); the "
                "data axis is refused under a workflow, whose runner is one "
                "process — shard a step with fan_out.over.shards instead",
            )
        if verb == "run":
            p.add_argument(
                "--cuda-graphs",
                action="store_true",
                help="Use CUDA replay for supported pytorch_hooks workloads.",
            )
            p.add_argument(
                "--out",
                type=Path,
                required=True,
                help="Output directory. A workflow creates its declared output_dir beneath this path.",
            )
            p.add_argument(
                "--resume",
                action="store_true",
                help="Reuse workflow steps whose identity, runtime, and verified outputs match their saved records.",
            )
            p.add_argument(
                "--reuse-nondeterministic",
                action="store_true",
                help="Allow --resume to reuse steps with is_deterministic: false.",
            )
            p.add_argument(
                "--device",
                default="cpu",
                help="PyTorch device for the reference engine (cpu, cuda, cuda:1, "
                "mps), or a comma list (cuda:0,cuda:1) placing the model's layers "
                "across the devices of this process — contiguous even block "
                "ranges in order, embedding first, head last; memory, not speed.",
            )
            p.add_argument(
                "--dtype",
                choices=PRECISION_DTYPES,
                default=None,
                help="Set model.dtype in the document. The selected precision enters its digest.",
            )
            p.add_argument(
                "--record",
                action="store_true",
                help="Also write the run receipt protocol.json and the event stream events.jsonl into --out. Off by default: a run writes its saved tables only.",
            )
            p.add_argument(
                "--points",
                default=None,
                metavar="START:STOP",
                help="Execute the point indices in [START, STOP) for an intervention document. Point digests stay the same.",
            )
            p.add_argument(
                "--batch-rows",
                type=_positive_int,
                default=None,
                metavar="N",
                help="Maximum rows per inference forward in pytorch_hooks, including training evaluation. Results can vary with floating-point rounding.",
            )
            p.add_argument(
                "--fit-rows",
                type=_positive_int,
                default=None,
                metavar="N",
                help="Maximum rows per training forward in pytorch_hooks. Each member's minibatch stays whole. Omit to measure a bound from available device memory. With --record the receipt records the bound.",
            )
            p.add_argument(
                "-v",
                "--verbose",
                action="store_true",
                help="Report progress on stderr: point selection, each point's model load, cohort fits, each point's run, and the output write. Also shows Hugging Face Hub download and lock-wait messages. Changes no output file.",
            )
    return parser


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"expected a positive row count, got {text}")
    return value


def _parallel(args: argparse.Namespace) -> ParallelGeometry:
    """The ``--parallel`` geometry, or the default (world 1) when the flag is
    absent or the verb has none. Raises the grammar's ``P4``."""
    text = getattr(args, "parallel", None)
    return ONE if text is None else parse_geometry(text)


def wants_hf_registration(args: argparse.Namespace) -> bool:
    """Whether this invocation may resolve an unregistered key over the network.

    ``run`` allows registration; other verbs require ``--register-from-hf``.
    Registration validates the requested model's geometry; substituting another
    registered model can falsely reject valid layer indices.
    """
    return args.verb == "run" or bool(getattr(args, "register_from_hf", False))


def ensure_model_registered(args: argparse.Namespace) -> None:
    """Resolve an unregistered model key from its HF config and register it
    before canonicalization.

    Called for ``run`` unconditionally and for the pure verbs only under
    ``--register-from-hf``, so the invariant survives: without the flag a
    digest never depends on the network."""
    from causalab.protocol.pipeline import read_document

    # read through the compiler's own prefix, so `--set model.key=…` is applied
    # and the key read here is the one the compile will read
    try:
        authored = read_document(
            args.document, args.document.resolve().parent, dict(args.parsed_set)
        )
    except ProtocolError:
        return  # a malformed document refuses properly in the real compile
    register_model_key(dict(authored.raw))


def register_model_key(raw: dict[str, Any]) -> None:
    from causalab.protocol.registry import (
        get_model_info,
        model_info_from_hf_config,
        register_model,
    )

    model = raw.get("model", {})
    key = model.get("key") if isinstance(model, dict) else None
    if not isinstance(key, str):
        return
    try:
        get_model_info(key)
    except ProtocolError:
        from transformers import AutoConfig

        revision = model.get("revision", "main") if isinstance(model, dict) else "main"
        config = AutoConfig.from_pretrained(key, revision=revision)
        register_model(model_info_from_hf_config(key, config))


def main(argv: Sequence[str] | None = None) -> int:
    """Parse, build the environment, and dispatch on document type."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.verb == "measure":
        from causalab.measurement.study.controller import run

        try:
            reports = run(args.document, args.bindings, args.out, resume=args.resume)
        except (ValueError, RuntimeError, ProtocolError) as err:
            print(f"refused: {err}", file=sys.stderr)
            return 1
        print(reports)
        return 0
    if args.verb == "migrate":
        # a rewrite of files, not a compile: no environment, no dispatch
        from causalab.protocol.migrate import main as migrate_main

        return migrate_main(args)
    if getattr(args, "engine", None) == "nnsight":
        # fail closed: both bounds are the reference engine's, and a pinned
        # nnsight run would drop them silently while the receipt said null
        if getattr(args, "batch_rows", None) is not None:
            parser.error(
                "--batch-rows bounds only the reference engine, and --engine "
                "nnsight pins an engine that runs every group as one batch, so "
                "the two flags cannot be combined"
            )
        if getattr(args, "fit_rows", None) is not None:
            parser.error(
                "--fit-rows bounds only the reference engine's grad forwards, "
                "and --engine nnsight pins an engine with no grad path, so the "
                "two flags cannot be combined"
            )
    if getattr(args, "verbose", False):
        # before any dispatch, so a workflow run's protocol steps report too:
        # they run through the same shared execution loop
        _configure_verbose()
    if getattr(args, "engine", None) is not None:
        # the router's one decision, made once here: 'auto' becomes the engine
        # name every verb below reads — the pure verbs hand it to
        # pipeline.validate as the engine to hold the document to, `run`
        # constructs it (causalab.neural.shared.engine_router)
        args.engine = route_name(args.engine)
    args.parsed_set = _overrides(args)
    # the shipped task tables stay reachable behind any --data-root
    env = file_env(args.data_root, args.artifacts_root, fallback_roots=(TASKS_ROOT,))
    try:
        # the geometry, parsed once here so both document types and both
        # verbs read one ParallelGeometry; a malformed one is the grammar's
        # P4 refusal, printed below like every other
        args.parallel_geometry = _parallel(args)
        if (
            getattr(args, "engine", None) == "nnsight"
            and args.parallel_geometry.world > 1
        ):
            # fail closed, like the two bounds: the nnsight engine is
            # single-device (docs/model_parallelism.md §8.5), and a pinned
            # nnsight run would drop the geometry while the receipt kept it
            parser.error(
                "--parallel is the reference engine's geometry, and --engine "
                "nnsight pins a single-device engine, so the two flags cannot "
                "be combined above a world of 1"
            )
        from causalab.io.sources import load_text
        from causalab.workflow.document import is_workflow

        if is_workflow(load_text(args.document)):
            if args.verb == "dry-run":
                print(
                    "refused: dry-run is per intervention specification; the "
                    "workflow-level dry run is a follow-up — validate or explain "
                    "the workflow, and dry-run its documents one by one",
                    file=sys.stderr,
                )
                return 1
            from causalab.workflow import cli as workflow_cli

            if args.verb == "run" and args.parallel_geometry.world > 1:
                # docs/model_parallelism.md §3, §11: a workflow at a world
                # above 1 launches like a document — after the one refusal
                # the launch cannot serve, the data axis, refused by name
                # here before any child starts and on every rank alike
                workflow_cli.check_geometry(args.parallel_geometry)
                return _launched(
                    args, env, sys.argv[1:] if argv is None else argv, workflow_cli.main
                )
            return workflow_cli.main(args, env)
        if getattr(args, "resume", False):
            print(
                "refused: --resume is a workflow flag — it reuses a published "
                "step whose identity and files still match, and an intervention "
                "specification run has no step boundaries to resume at (IM spec "
                "§9). Wrap the document in a workflow step, or shard it with "
                "--points",
                file=sys.stderr,
            )
            return 1
        from causalab.protocol import reports as protocol_cli

        if args.verb == "run" and args.parallel_geometry.world > 1:
            return _launched(
                args, env, sys.argv[1:] if argv is None else argv, protocol_cli.main
            )
        return protocol_cli.main(args, env)
    except ProtocolError as err:
        print(f"refused: {err}", file=sys.stderr)
        return 1


def _launched(
    args: argparse.Namespace,
    env: ResolutionEnv,
    argv: Sequence[str],
    run: Callable[[argparse.Namespace, ResolutionEnv], int],
) -> int:
    """A run at a world above 1 (``docs/model_parallelism.md`` §3) — of an
    intervention specification or of a workflow, ``run`` being that document
    type's ``main``.

    [`detect`][causalab.neural.shared.parallel.launcher.detect] says what this
    process is. The **parent** of a spawn — no ``WORLD_SIZE`` in the
    environment — starts ``world`` children re-entering this very ``main``
    with the same ``argv`` and exits with their status; it builds no engine
    and loads no model. A **spawned** child or a **joined** rank
    (``torchrun``, Slurm) pins its device (``cuda:LOCAL_RANK`` under
    ``--device cuda``), joins the process group for the device's backend,
    and runs the document as that rank: its publisher rides on every
    engine request, so exactly one process writes the campaign — and, for a
    workflow, its lockstep carries the joiner's per-process decisions to
    every rank (§11; [`causalab.protocol.lockstep`][]), so the ranks run
    the steps together and the joiner alone writes the ROOT.

    Imported here, not at module scope, and torch-free on the parent's
    path: the parent of a spawn imports no torch (its import would sit
    serially ahead of every child's start), only a rank does, in ``enter``.
    """
    from causalab.neural.shared.parallel import launcher

    geometry = args.parallel_geometry
    detected = launcher.detect(geometry)
    visible = launcher.visible_devices(args.device)
    if isinstance(detected, launcher.Parent):
        launcher.check_spawn_devices(geometry, args.device, visible=visible)
        return launcher.spawn(geometry, argv, device=args.device)
    args.device = launcher.device_for(detected, args.device, visible=visible)
    publisher = launcher.enter(detected, geometry, args.device)
    # the status the peers hear of through the rank's heartbeat (§3 "when a
    # rank dies"): the run's, or a failure when it raised
    status = 1
    try:
        args.publisher = publisher
        args.lockstep = launcher.lockstep(publisher)
        status = run(args, env)
        return status
    except RuntimeError as error:
        # a backend failure outside this package's collectives (transformers'
        # own dist calls in its styles): a dead peer is refused by name here
        # and the process ends; torch's own distributed error with nobody
        # lost is a refusal naming the rank; anything else is the traceback
        # it was
        refusal = launcher.hold_for_peer(error)
        if refusal is not None:
            raise refusal from error
        raise
    finally:
        launcher.leave(publisher, status)


if __name__ == "__main__":
    # `python -m causalab.cli …` is what a launcher that runs a module per
    # rank (`torchrun -m causalab.cli run …`) needs; the console script is
    # the same `main`
    raise SystemExit(main())
