"""Local controller and isolated source installation for authored measurements.

Local and remote execution use private wheel targets over prepared dependency
environments, leaving the source checkouts and dependency environments unchanged.
"""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from typing import Any
import uuid

from ..collection import file_hash, write_record
from ..device import require_single_device
from ..paths import controller_root, source_identity, worker_bootstrap
from ..plan import execution_plan
from ..deployment.bindings import target_bindings
from ..runtime.cache import COLD_RESET_POLICY
from ..runtime.benchmark import benchmark_identity
from causalab.measurement.deployment.installation import (
    build_arm,
    build_source,
    load_installation,
    load_source,
)
from causalab.measurement.study.scheduler import run_schedule


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=repository, text=True
    ).strip()


class ProcessSession:
    def __init__(self, python: str, config: dict[str, Any], logs: Path):
        require_single_device(config["device"])
        self.python, self.config, self.logs = python, config, logs
        logs.mkdir(parents=True, exist_ok=True)
        self.child = None
        self.log = None
        self.start()

    def start(self):
        require_single_device(self.config["device"])
        ticket = uuid.uuid4().hex
        path = self.logs / f"{ticket}.json"
        write_record(path, self.config)
        self.log = (self.logs / f"{ticket}.log").open("w")
        bootstrap = worker_bootstrap(Path(self.config["controller_root"]))
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1")
        self.child = subprocess.Popen(
            [self.python, str(bootstrap), str(path)],
            cwd=self.config["package_root"],
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.log,
            text=True,
            pass_fds=tuple(
                config_fd
                for config_fd in [self.config.get("lease_fd")]
                if config_fd is not None
            ),
        )
        try:
            self.identity = self.read()["identity"]
            self.config["execution_probe"] = self.identity["execution_probe"]
            # Isolated cold/capture workers must reproduce this exact census,
            # not accept a newly resolved source or external input contract.
            if "source_pins" in self.identity:
                self.config["resolved_pins"] = {
                    category: {
                        **self.identity["shared_pins"].get(category, {}),
                        **self.identity["source_pins"].get(category, {}),
                    }
                    for category in (
                        "documents",
                        "datasets",
                        "files",
                        "code",
                        "scripts",
                    )
                }
        except BaseException:
            self.close()
            raise

    def read(self) -> dict[str, Any]:
        assert self.child is not None and self.child.stdout is not None
        line = self.child.stdout.readline()
        if not line:
            try:
                code = self.child.wait(timeout=1)
            except subprocess.TimeoutExpired:
                status = "closed its output while still running"
            else:
                status = (
                    f"exited with code {code}"
                    if code >= 0
                    else f"terminated by {signal.Signals(-code).name} ({code})"
                )
            raise RuntimeError(
                f"measurement worker {status}; inspect logs under {self.logs}"
            )
        value = json.loads(line)
        if "error" in value:
            raise RuntimeError(value["error"])
        return value

    def sample(self, case, seed, repeat, directory):
        if self.child is None:
            self.start()
        assert self.child is not None and self.child.stdin is not None
        self.child.stdin.write(
            json.dumps(
                {
                    "action": "sample",
                    "case": case,
                    "seed": seed,
                    "directory": str(directory),
                }
            )
            + "\n"
        )
        self.child.stdin.flush()
        receipt = Path(self.read()["receipt"])
        if self.config["plan"]["cases"][case]["cold_process"]:
            self.close()  # Release the resident model before the cold process.
            self._cold(case, seed, directory, receipt)
        return receipt

    def capture(self, case, seed, repeat, directory):
        """Collect diagnostics only after the scheduler completes clean timings."""
        # The initialized session attests this arm before releasing GPU memory.
        self.close()
        from ..capture.controller import run_captures

        directory.mkdir(parents=True, exist_ok=False)
        receipt = directory / "measurement.json"
        write_record(receipt, {"captures": []})
        run_captures(
            self.python, self.config, self.identity, case, seed, directory, receipt
        )
        return receipt

    def evaluate(self, steps, seed, fit_root, directory):
        if self.child is None:
            self.start()
        assert self.child is not None and self.child.stdin is not None
        self.child.stdin.write(
            json.dumps(
                {
                    "action": "evaluate",
                    "steps": steps,
                    "seed": seed,
                    "fit_root": str(fit_root),
                    "directory": str(directory),
                }
            )
            + "\n"
        )
        self.child.stdin.flush()
        return self.read()

    def _cold(self, case: str, seed: int, directory: Path, receipt: Path) -> None:
        config: dict[str, Any] = {
            **self.config,
            "verified_model_files": {
                key: value["files"] for key, value in self.identity["models"].items()
            },
            "cold": {"case": case, "seed": seed, "directory": str(directory / "cold")},
        }
        bootstrap = worker_bootstrap(Path(config["controller_root"]))

        def execute_cold(cold_config):
            path = self.logs / f"{uuid.uuid4().hex}.cold.json"
            write_record(path, cold_config)
            with path.with_suffix(".log").open("w") as log:
                started = time.perf_counter_ns()
                process = subprocess.run(
                    [self.python, str(bootstrap), str(path)],
                    cwd=config["package_root"],
                    env=dict(
                        os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1"
                    ),
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=log,
                    text=True,
                    check=True,
                    pass_fds=tuple(
                        config_fd
                        for config_fd in [self.config.get("lease_fd")]
                        if config_fd is not None
                    ),
                )
                seconds = (time.perf_counter_ns() - started) / 1e9
            response = json.loads(process.stdout)
            if (
                response["identity"] != self.identity
                or response["status"] != "completed"
            ):
                raise ValueError("cold process did not execute the same attested arm")
            return seconds, response

        seconds, response = execute_cold(config)
        record = json.loads(receipt.read_text())
        record["context"]["resident_cache_policy"] = record["context"]["cache_policy"]
        record["context"]["cache_policy"] = response["cache_policy"]
        record["reset_policy"] = COLD_RESET_POLICY
        sample = record["samples"][0]
        from causalab.io.sources import load_text

        workflow = load_text(Path(self.config["workflow"]))
        cold_outputs = directory / "cold" / workflow["output_dir"]
        sample["resident_output_files"] = sample.pop("output_files", {})
        sample["output_files"] = {
            path.relative_to(directory).as_posix(): file_hash(path)
            for path in sorted(cold_outputs.rglob("*"))
            if path.is_file()
        }
        sample["workflow_outputs"] = "cold/" + workflow["output_dir"]
        if self.config["plan"].get("observation_policy", "required") == "required":
            # Read the cold process's required outputs after exit so observations
            # describe the timed execution without adding an observer to it.
            from safetensors.torch import save_file

            from causalab.measurement.runtime.observations import (
                observation_specs,
                observations,
            )

            cold_observations = directory / "cold.safetensors"
            cold_values = observations(
                cold_outputs, self.config["plan"]["observations"]
            )
            save_file(cold_values, str(cold_observations))
            for field in (
                "diagnostic_observations",
                "diagnostic_observation_specs",
                "diagnostic_workflow_outputs",
            ):
                if field in sample:
                    sample["resident_" + field] = sample.pop(field)
            if (sample.get("numerics_context") or {}).get("fits"):
                from causalab.measurement.runtime.training import output_check

                diagnostic_config = {
                    **config,
                    "cold": {
                        **config["cold"],
                        "directory": str(directory / "cold_diagnostic"),
                        "observe_training": True,
                    },
                }
                _, diagnostic = execute_cold(diagnostic_config)
                diagnostic_root = directory / "cold_diagnostic" / workflow["output_dir"]
                diagnostic_values = observations(
                    diagnostic_root, self.config["plan"]["observations"]
                )
                diagnostic_file = directory / "cold_diagnostic.safetensors"
                save_file(diagnostic_values, str(diagnostic_file))
                sample["resident_numerics_context"] = sample["numerics_context"]
                sample["resident_observer_check"] = sample["observer_check"]
                sample["diagnostic_observations"] = {
                    "file": diagnostic_file.name,
                    "sha256": file_hash(diagnostic_file),
                }
                sample["diagnostic_workflow_outputs"] = (
                    "cold_diagnostic/" + workflow["output_dir"]
                )
                sample["diagnostic_observation_specs"] = observation_specs(
                    diagnostic_root, self.config["plan"]["observations"]
                )
                sample["numerics_context"] = diagnostic["numerics_context"]
                sample["numerics_context"]["scope"] = (
                    "separate cold-process numerical diagnostic; excluded from primary cold wall time"
                )
                sample["observer_check"] = output_check(cold_values, diagnostic_values)
                sample["observer_check"]["observation_specs_match"] = observation_specs(
                    cold_outputs, self.config["plan"]["observations"]
                ) == observation_specs(
                    diagnostic_root, self.config["plan"]["observations"]
                )
            sample["resident_observations"] = sample["observations"]
            sample["resident_observation_specs"] = sample.get("observation_specs", {})
            sample["observation_specs"] = observation_specs(
                cold_outputs, self.config["plan"]["observations"]
            )
            sample["observations"] = {
                "file": cold_observations.name,
                "sha256": file_hash(cold_observations),
            }
            sample["workflow_outputs"] = "cold/" + workflow["output_dir"]
            sample["observation_origin"] = "cold_timing_pass"
            if "diagnostic_observations" not in sample:
                sample["resident_observer_check"] = sample.pop("observer_check", {})
            # Even non-fitting cold cases retain the corresponding clean reference.
            sample["resident_unobserved_observations"] = sample["resident_observations"]
            sample["resident_unobserved_observation_specs"] = sample[
                "resident_observation_specs"
            ]
            sample["unobserved_observations"] = dict(sample["observations"])
            sample["unobserved_observation_specs"] = dict(sample["observation_specs"])
        sample["resident_seconds"] = sample["seconds"]
        sample["seconds"] = seconds
        sample["resident_peak_memory"] = sample["peak_memory"]
        sample["peak_memory"] = None
        record["scope"] = (
            "cold-process workflow worker wall: process creation through exit; source/environment verification, imports, model load, workflow and required publication included; checkpoint hashing and execution probes excluded; OS/HF caches not flushed"
        )
        record["passes"] = (
            "cold timed workflow with required saved outputs; observations extracted after process exit; separate profiling subprocesses and resident diagnostic passes"
        )
        if self.config["plan"].get("observation_policy") == "not_requested":
            record["passes"] = (
                "cold timed workflow with required saved outputs; no numerical-only passes; separate profiling subprocesses"
            )
        if "diagnostic_observations" in sample:
            record["passes"] += (
                "; separate cold-process training-state diagnostic checked against the timed workflow outputs"
            )
        sample["cold_attestation"] = (
            "checkpoint hashes verified in preceding session outside the cold timer; source and realized model configuration checked in cold process"
        )
        write_record(receipt, record)

    def close(self):
        if self.child is not None:
            if self.child.poll() is None:
                try:
                    if self.child.stdin is not None:
                        self.child.stdin.write('{"action":"shutdown"}\n')
                        self.child.stdin.flush()
                    self.child.wait(timeout=5)
                except (BrokenPipeError, subprocess.TimeoutExpired):
                    self.child.terminate()
                    try:
                        self.child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        self.child.kill()
                        self.child.wait()
            for stream in (self.child.stdin, self.child.stdout):
                if stream is not None:
                    try:
                        stream.close()
                    except BrokenPipeError:
                        # A failed worker may leave buffered shutdown bytes.
                        # Preserve its original startup/execution error.
                        pass
            self.child = None
        if self.log is not None:
            self.log.close()
            self.log = None


def run(
    document: Path, bindings_path: Path, output: Path, *, resume: bool = False
) -> Path:
    """Own deployment, collection and reporting under one process lock."""
    # Resumes acquire the existing worker lease before inspecting study inputs.
    require_single_device("cpu")
    if not resume:
        bindings = json.loads(bindings_path.read_text())
        require_single_device(
            bindings.get("device") if isinstance(bindings, dict) else None
        )
    output = output.resolve()
    if resume and not output.is_dir():
        raise ValueError("resume requires an existing measurement directory")
    output.mkdir(parents=True, exist_ok=resume)
    with (output / ".experiment.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(
                "another controller owns this measurement directory"
            ) from exc
        # Children inherit this lease. A dead controller cannot authorize a
        # restart while one of its orphaned workers still holds GPU state.
        with (output / ".workers.lock").open("a") as workers:
            try:
                fcntl.flock(workers, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ValueError(
                    "previous measurement workers are still running"
                ) from exc
            fcntl.flock(workers, fcntl.LOCK_SH)
            return _run(
                document,
                bindings_path,
                output,
                resume=resume,
                lease_fd=workers.fileno(),
            )


def _run(
    document: Path, bindings_path: Path, output: Path, *, resume: bool, lease_fd: int
) -> Path:
    from causalab.io.sources import load_text
    from causalab.workflow.document import parse_workflow
    from ..census import strip_pins
    from .definitions import load_definitions
    from .contrast import WorkflowContrast, definition_changes

    document, bindings_path, output = (
        document.resolve(),
        bindings_path.resolve(),
        output.resolve(),
    )
    # ``raw`` keeps the census (a study record, part of the contract below);
    # the workflow loader reads the document without it.
    raw = load_text(document)
    workflow = parse_workflow(strip_pins(dict(raw))[0])
    plan = workflow.measurement
    if plan is None:
        raise ValueError(
            "measurement requires a workflow with an authored measurement block"
        )
    plan = execution_plan(plan)
    bindings = target_bindings(json.loads(bindings_path.read_text()), plan)
    require_single_device(bindings["device"])
    for name in ("data_root", "artifacts_root"):
        path = Path(bindings[name])
        bindings[name] = str((bindings_path.parent / path).resolve())
    for arm, binding in bindings["arms"].items():
        for name in binding:
            # Resolving a venv's Python symlink selects its base interpreter and
            # silently loses the prepared dependencies. Preserve that path.
            binding[name] = os.path.abspath(bindings_path.parent / binding[name])
    definitions = load_definitions(document, workflow.measurement)
    authored = definitions[plan["source_pin_anchor"]].protocols
    fitting = any(definition.fitting for definition in definitions.values())
    if (
        fitting
        and len({arm["execution"]["batch_rows"] for arm in plan["arms"].values()}) > 1
    ):
        raise ValueError(
            "fitting comparisons require identical batch_rows in v1; "
            "preservation of optimizer updates across changed layouts is not attested"
        )
    contract = {
        "workflow": raw,
        "protocols": authored,
        "bindings": bindings,
        "source_bundles": {
            arm: load_source(Path(binding["source"]), plan["arms"][arm]["revision"])
            for arm, binding in bindings["arms"].items()
            if "source" in binding
        },
        "controller": source_identity(),
    }
    workflow_mode = plan.get("comparison", "code") == "workflow"
    if workflow_mode:
        contract["workflows"] = {
            arm: {"workflow": definition.raw, "protocols": definition.protocols}
            for arm, definition in definitions.items()
        }
    preparation = output / "preparation.json"
    if resume:
        if not preparation.is_file():
            raise ValueError("resume requires a preparation receipt")
        prepared = json.loads(preparation.read_text())
        if prepared["contract"] != contract:
            raise ValueError("resume refused: authored study or deployment changed")
        pins = prepared["source_commits"]
    else:
        pins = {
            arm: spec["revision"]
            if "installation" in bindings["arms"][arm]
            or "source" in bindings["arms"][arm]
            else _git(
                Path(bindings["arms"][arm]["repository"]),
                "rev-parse",
                "--verify",
                "--end-of-options",
                f"{spec['revision']}^{{commit}}",
            )
            for arm, spec in plan["arms"].items()
        }
        if workflow_mode:
            WorkflowContrast.create(
                pins,
                {arm: value.benchmark_identity for arm, value in definitions.items()},
            )
        write_record(preparation, {"contract": contract, "source_commits": pins})
    contrast = (
        WorkflowContrast.create(
            pins, {arm: value.benchmark_identity for arm, value in definitions.items()}
        )
        if workflow_mode
        else None
    )
    installations = {}
    for arm in plan["arms"]:
        binding = bindings["arms"][arm]
        if "installation" in binding:
            installation = load_installation(
                Path(binding["installation"]), pins[arm], python=binding["python"]
            )
        elif "source" in binding:
            source = Path(binding["source"])
            load_source(source, pins[arm])
            installation = build_source(
                source, output / "deployment" / arm, python=binding["python"]
            )
        else:
            installation = build_arm(
                Path(binding["repository"]),
                pins[arm],
                output / "deployment" / arm,
                python=binding["python"],
            )
        installations[arm] = installation
    contract["sources"] = installations
    input_identity = (
        contrast.identity
        if contrast
        else benchmark_identity(raw, authored, plan["observations"])
    )

    def configured_definition(arm: str) -> dict[str, Any]:
        definition = definitions[arm]
        return {
            "workflow": {
                key: value
                for key, value in definition.raw.items()
                if key not in {"measurement", "pins"}
            },
            "protocols": definition.protocols,
            "execution": plan["arms"][arm]["execution"],
        }

    opened = set()
    probes = {}

    @contextmanager
    def open_session(arm):
        config = {
            "arm": arm,
            "plan": plan,
            "workflow": str(definitions[arm].path),
            "input_identity": input_identity,
            "controller_root": str(controller_root()),
            **{key: bindings[key] for key in ("device", "data_root", "artifacts_root")},
            "package_root": installations[arm]["package_root"],
            "source_commit": installations[arm]["source_commit"],
            # package_root is always <deployment>/installed (installation.py)
            "installation": str(
                Path(installations[arm]["package_root"]).parent / "installation.json"
            ),
            "checkpoint_cache": str(output / "checkpoint-verification" / arm),
            "rehash_models": arm not in opened,
            "lease_fd": lease_fd,
            "probe_directory": str(
                output / "execution-probes" / arm / uuid.uuid4().hex
            ),
            "execution_probe": probes.get(arm),
        }
        if contrast is not None:
            config.update(
                benchmark_identity=definitions[arm].benchmark_identity,
                comparison_contract=contrast.receipt(),
                configuration_changes=definition_changes(
                    configured_definition("before"), configured_definition(arm)
                ),
            )
        print(f"[{arm}] preparing source, model and execution identity", flush=True)
        session = ProcessSession(
            bindings["arms"][arm]["python"], config, output / "workers" / arm
        )
        opened.add(arm)
        probes[arm] = session.identity["execution_probe"]
        print(f"[{arm}] ready", flush=True)
        try:
            yield session
        finally:
            session.close()

    from causalab.measurement.study.evaluation import finish_evaluations

    result = run_schedule(
        plan,
        contract,
        output / "collections",
        open_session,
        resume=resume,
        finish_block=(
            lambda cell, attempt, identities: finish_evaluations(
                plan, open_session, cell, attempt, identities
            )
        )
        if plan["evaluation"]
        else None,
    )
    from causalab.measurement.analysis.reports import write_reports

    reports = output / "reports"
    write_reports(result, plan, reports)
    return reports
