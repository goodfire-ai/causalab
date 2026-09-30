"""Run every method document into ``demos/methods/runs/<name>/``.

The documents are a dependency graph, not a list: a fit's apply half loads the
fit's bundle through ``--artifacts-root``, ``mean_ablation`` loads
``mean_harvest``'s mean, ``das_pca_init`` starts from the basis
``workflows/pca_basis.json`` fits. `GROUPS` spells that order — one
group is one independent chain, so the groups can run as separate jobs
(``--group``) and the documents inside one run in sequence. A workflow runs as
a workflow (no ``--dtype``: its steps declare their own precision); an
intervention specification runs as authored, so one that declares no
``dtype`` runs in the engine's default precision.

Every run is one ``causalab run`` subprocess, so the command the sidecar
records is the command a reader can paste.

    uv run python demos/methods/scripts/run_all.py --device cuda
    uv run python demos/methods/scripts/run_all.py --device cuda --group dbm
    uv run python demos/methods/scripts/run_all.py --list

Each run directory gets a ``_run_all.json`` sidecar (the command, the model,
the device, the dtype the document declared, wall time) that
``summarize.py`` reduces into ``results/<name>.json``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

METHODS = Path(__file__).resolve().parents[1]
REPO = METHODS.parents[1]
PROTOCOLS = METHODS / "protocols"
WORKFLOWS = METHODS / "workflows"
RUNS = METHODS / "runs"
#: Where ``das_pca_init.json`` expects its basis (a repo-relative artifact
#: path, resolved against ``--artifacts-root``), copied out of the
#: ``pca_basis`` workflow's ``pca`` step.
PCA_BASIS_TARGET = "artifacts/weekdays/qwen25_7b/pca/block_output_L23.safetensors"


@dataclass(frozen=True)
class Run:
    """One document to run; ``artifacts`` is the run directory whose tree the
    document's ``file_path`` entries name (``fit/gate.safetensors`` resolves
    under it), ``after`` a callable to run once the document has finished."""

    name: str
    kind: str  # "protocol" | "workflow"
    artifacts: str | None = None
    copy_basis: bool = False

    @property
    def document(self) -> Path:
        return (
            WORKFLOWS if self.kind == "workflow" else PROTOCOLS
        ) / f"{self.name}.json"

    @property
    def out(self) -> Path:
        """Protocol runs sit at ``runs/<name>``, workflow runs at
        ``runs/workflows/<name>`` — the two kinds share names
        (``mean_ablation``), and so do their results files."""
        return (
            RUNS / self.name
            if self.kind == "protocol"
            else RUNS / "workflows" / self.name
        )


#: Group -> the runs of that chain, in order. A fit and its apply half share a
#: group; the apply's ``fit/<bundle>`` path resolves under a directory whose
#: ``fit/`` is the fit's run tree, which ``artifacts_dir`` arranges.
GROUPS: dict[str, tuple[Run, ...]] = {
    "interchange": (Run("interchange", "protocol"),),
    "weekdays_interchange": (Run("weekdays_interchange", "protocol"),),
    "weekdays_locate_scan": (Run("weekdays_locate_scan", "protocol"),),
    "multi_position_patch": (Run("multi_position_patch", "protocol"),),
    "attention_band_patch": (Run("attention_band_patch", "protocol"),),
    "path_patching": (Run("path_patching", "protocol"),),
    "hydra_effect": (Run("hydra_effect", "protocol"),),
    "harvest": (Run("harvest", "protocol"),),
    "probe_generate": (Run("probe_generate", "protocol"),),
    "probe_variable": (Run("probe_variable", "protocol"),),
    "random_subspace_control": (Run("random_subspace_control", "protocol"),),
    "das": (Run("das", "protocol"),),
    "das_boundless": (Run("das_boundless", "protocol"),),
    "dbm": (Run("dbm", "protocol"), Run("dbm_apply", "protocol", artifacts="dbm")),
    "mean_ablation": (
        Run("mean_harvest", "protocol"),
        Run("mean_ablation", "protocol", artifacts="mean_harvest"),
        Run("mean_ablation", "workflow"),
    ),
    "weekdays": (Run("weekdays", "workflow"),),
    "pca": (
        Run("pca_basis", "workflow", copy_basis=True),
        Run("das_pca_init", "protocol"),
    ),
    # the four Qwen3.6-35B-A3B documents: one 80 GB GPU each
    "dbm_head": (
        Run("dbm_head", "protocol"),
        Run("dbm_head_apply", "protocol", artifacts="dbm_head"),
    ),
    "dbm_expert_neuron": (
        Run("dbm_expert_neuron", "protocol"),
        Run("dbm_expert_neuron_apply", "protocol", artifacts="dbm_expert_neuron"),
    ),
    # CPU, tiny-random: the CI smoke document, over the protocol fixture rows
    "minimal_cpu": (Run("minimal_cpu", "protocol"),),
}
#: ``minimal_cpu.json`` names a 4-row fixture table the shipped tasks do not
#: carry, and its model registers from the HF config rather than the registry.
MINIMAL_CPU_ARGS = (
    "--data-root",
    str(REPO / "tests/protocol/fixtures/data"),
    "--artifacts-root",
    str(REPO / "tests/protocol/fixtures/artifacts"),
    "--device",
    "cpu",
    "--register-from-hf",
)
A3B_GROUPS = ("dbm_head", "dbm_expert_neuron")


def artifacts_dir(run: Run) -> Path:
    """A directory under which ``fit/<bundle>`` (or ``harvest/<bundle>``) is
    the producing run's tree: the apply document's ``file_path`` first segment
    is the *step name* its workflow would use, so the driver builds that
    layout with one symlink rather than rewriting the document."""
    assert run.artifacts is not None
    step = {
        "dbm_apply": "fit",
        "dbm_head_apply": "fit",
        "dbm_expert_neuron_apply": "fit",
        "mean_ablation": "harvest",
    }[run.name]
    root = RUNS / f"_artifacts_{run.name}"
    root.mkdir(parents=True, exist_ok=True)
    link = root / step
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to((RUNS / run.artifacts).resolve())
    return root


def command(run: Run, device: str, engine: str, resume: bool) -> list[str]:
    causalab = (
        Path(sys.executable).parent / "causalab"
    )  # the environment's console script
    cmd = [
        str(causalab),
        "run",
        str(run.document),
        "--engine",
        engine,
        "--out",
        str(run.out),
    ]
    if run.name == "minimal_cpu":
        cmd += list(MINIMAL_CPU_ARGS)
    else:
        cmd += ["--device", device]
    if run.kind == "workflow":
        # a workflow creates its declared output_dir under --out, and every
        # shipped workflow's output_dir is its file name, so its tree lands at
        # run.out
        cmd[cmd.index("--out") + 1] = str(RUNS / "workflows")
        cmd += ["--artifacts-root", str(RUNS)]
        if resume:
            cmd.append("--resume")
    else:
        cmd.append("--record")
        if run.artifacts is not None:
            cmd += ["--artifacts-root", str(artifacts_dir(run))]
        elif run.name == "das_pca_init":
            cmd += ["--artifacts-root", str(RUNS)]
    return cmd


def place_basis() -> None:
    """Copy the ``pca_basis`` workflow's basis to the address
    ``das_pca_init.json`` names, under the runs root."""
    source = RUNS / "workflows" / "pca_basis" / "pca" / "basis.safetensors"
    target = RUNS / PCA_BASIS_TARGET
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    print(f"placed {source.relative_to(REPO)} -> {target.relative_to(REPO)}")


def execute(run: Run, device: str, engine: str, resume: bool) -> int:
    cmd = command(run, device, engine, resume)
    if run.out.exists() and not resume:
        shutil.rmtree(run.out)
    run.out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    print("+", " ".join(cmd), flush=True)
    completed = subprocess.run(cmd, cwd=REPO)
    elapsed = time.time() - started
    raw = json.loads(run.document.read_text())
    model = raw.get("model", {})
    sidecar = {
        "document": str(run.document.relative_to(METHODS)),
        "kind": run.kind,
        "model": model.get("key"),
        "engine": engine,
        "device": "cpu" if run.name == "minimal_cpu" else device,
        "dtype": model.get("dtype", "engine default"),
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "elapsed_s": round(elapsed, 1),
        "returncode": completed.returncode,
        "command": cmd,
    }
    (run.out / "_run_all.json").write_text(json.dumps(sidecar, indent=2) + "\n")
    if completed.returncode == 0 and run.copy_basis:
        place_basis()
    return completed.returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--engine", default="auto")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="pass --resume to workflows; keep finished protocol runs",
    )
    parser.add_argument(
        "--group", action="append", help="only these groups (repeatable)"
    )
    parser.add_argument(
        "--skip-a3b", action="store_true", help="skip the Qwen3.6-35B-A3B groups"
    )
    parser.add_argument("--list", action="store_true", help="print the groups and stop")
    args = parser.parse_args(argv)
    groups = args.group or list(GROUPS)
    if args.skip_a3b:
        groups = [g for g in groups if g not in A3B_GROUPS]
    unknown = [g for g in groups if g not in GROUPS]
    if unknown:
        parser.error(f"unknown group(s) {unknown}; known: {sorted(GROUPS)}")
    if args.list:
        for g in groups:
            print(g, "->", ", ".join(r.name for r in GROUPS[g]))
        return 0
    failures = 0
    for g in groups:
        for run in GROUPS[g]:
            if args.resume and (run.out / "_run_all.json").exists():
                if (
                    json.loads((run.out / "_run_all.json").read_text()).get(
                        "returncode"
                    )
                    == 0
                ):
                    print(f"= {run.name}: finished, kept")
                    continue
            code = execute(run, args.device, args.engine, args.resume)
            if code != 0:
                failures += 1
                print(
                    f"! {run.name} failed ({code}); the rest of group {g} is skipped",
                    file=sys.stderr,
                )
                break
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
