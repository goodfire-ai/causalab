"""Execute the authored workflow using only the selected source arm's engine.

Loaded under a private package by the standalone bootstrap. Imports beginning
with causalab deliberately resolve to the arm; relative imports are the current
measurement controller. Timing never includes IPC or observation extraction.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext, redirect_stdout
from collections import Counter
from dataclasses import asdict
import hashlib
import inspect
import json
from pathlib import Path
import random
import sys
import traceback
from typing import Any, Mapping

from ..collection import (
    Operation,
    collect,
    execution_identity,
    file_hash,
    write_record,
)
from ..device import require_single_device_runtime
from .cache import RESIDENT_RESET_POLICY, cache_provenance
from .observations import observations
from ..census import CensusError, check_pins, collect_pins
from .pins import PinContract
from ..study.scheduler import manifest
from .benchmark import BenchmarkIdentityError, benchmark_identity
from .comparison import (
    check_workflow_contract,
    checkpoint_aliases,
    logical_input_rows,
    workflow_inputs,
)


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False, default=str).encode()
    ).hexdigest()


def model_files(
    key: str, revision: str, *, cache: Path | None = None, rehash: bool = True
) -> dict[str, str]:
    """Hash at startup/resume; reuse within a study only under unchanged file identity.

    The identity includes inode, device, size, mtime and ctime, following checkpoint
    symlinks to their targets. A content change with a restored mtime still changes
    ctime. Rechecking before/after hashing refuses files modified during the read.
    """
    root = Path(key)
    if not root.is_dir():
        from huggingface_hub import snapshot_download

        root = Path(snapshot_download(key, revision=revision, local_files_only=True))
    suffixes = {".safetensors", ".bin", ".json", ".txt", ".model", ".tiktoken"}
    paths = {
        path.relative_to(root).as_posix(): path
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and path.suffix in suffixes
        and ".cache" not in path.relative_to(root).parts
    }
    if not any(name.endswith((".safetensors", ".bin")) for name in paths):
        raise ValueError("model snapshot contains no attested weight files")

    def signatures():
        result = {}
        for name, path in paths.items():
            stat = path.stat()
            result[name] = [
                stat.st_dev,
                stat.st_ino,
                stat.st_size,
                stat.st_mtime_ns,
                stat.st_ctime_ns,
            ]
        return result

    before = signatures()
    if cache is not None and cache.is_file() and not rehash:
        stored = json.loads(cache.read_text())
        if (
            stored.get("root") == str(root.resolve())
            and stored.get("signatures") == before
            and set(stored.get("files", {})) == set(paths)
        ):
            return stored["files"]
    files = {name: file_hash(path) for name, path in paths.items()}
    if signatures() != before:
        raise ValueError("checkpoint files changed during content attestation")
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        write_record(
            cache, {"root": str(root.resolve()), "signatures": before, "files": files}
        )
    return files


def parameter_elements(model) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for parameter in model.parameters():
        counts[f"{parameter.device}/{parameter.dtype}"] += parameter.numel()
    return dict(sorted(counts.items()))


def attest_installation(config: Mapping[str, Any], package: Path) -> str:
    """The commit the imported arm was built from, from the deployment record.

    The runtime identity states what is installed (its tree digest) and not
    which git revision produced it; the controller's ``installation.json``
    is the one record binding the installed bytes to ``source_commit``. Hold
    the record to the study's commit, to the package the worker imports, and
    to the bytes on disk, and return the attested commit.
    """
    record = json.loads(Path(config["installation"]).read_text())
    commit = record["source_commit"]
    if commit != config["source_commit"]:
        raise ValueError(
            f"arm revision mismatch: expected {config['source_commit']}, "
            f"installed {commit}"
        )
    if Path(record["package_root"]).resolve() != package.resolve():
        raise ValueError(
            f"arm installation mismatch: record installs {record['package_root']}, "
            f"worker imports {package}"
        )
    if manifest(package) != record["installed_files"]:
        raise ValueError("deployed source installation changed")
    return commit


class Worker:
    def __init__(self, config: dict[str, Any]):
        require_single_device_runtime(config["device"])
        import causalab
        import torch

        self.identity = execution_identity(torch.device(config["device"]))
        actual = Path(causalab.__file__).resolve().parent
        installed = Path(self.identity["implementation"]["location"]).resolve()
        expected = Path(config["package_root"]).resolve() / "causalab"
        if actual != installed or actual != expected:
            raise ValueError(
                f"arm import/installation mismatch: imported {actual}, metadata {installed}, expected {expected}"
            )
        revision = attest_installation(config, expected.parent)

        from causalab.cli import register_model_key
        from causalab.neural.engines.pytorch_hooks.engine import PytorchHooksEngine
        from causalab.neural.engines.pytorch_hooks.loading import load_model
        from causalab.protocol.schema.explicit import canonical_model
        from causalab.protocol.pipeline import read_document
        from causalab.io.sources import load_text
        from causalab.io.env import FileArtifacts, FileDatasets, ResolutionEnv

        self.config = config
        self.plan = config["plan"]
        self.path = Path(config["workflow"])
        self.raw = dict(load_text(self.path))
        self.raw.pop("measurement", None)
        self.pin_contract: PinContract | None = None
        self.env = ResolutionEnv(
            datasets=FileDatasets(root=Path(config["data_root"])),
            artifacts=FileArtifacts(root=Path(config["artifacts_root"])),
        )
        self.fitting = set()
        authored = {}
        for name, step in self.raw["steps"].items():
            if step["type"] == "intervention_protocol":
                path = (self.path.parent / step["document"]).resolve()
                inner = read_document(path, path.parent, step.get("set", {}))
                authored[name] = dict(inner.raw)
                register_model_key(dict(inner.raw))
                if inner.raw.get("method", {}).get("train") is not None:
                    self.fitting.add(name)
        observed = benchmark_identity(self.raw, authored, self.plan["observations"])
        expected_benchmark = (
            config["benchmark_identity"]
            if self.plan.get("comparison", "code") == "workflow"
            else config["input_identity"]
        )
        if observed != expected_benchmark:
            raise BenchmarkIdentityError(expected_benchmark, observed)
        if self.plan.get("comparison", "code") == "workflow":
            check_workflow_contract(
                config["comparison_contract"],
                arm=config["arm"],
                source_commit=revision,
                benchmark_identity=observed,
                input_identity=config["input_identity"],
            )
        self.loaded = self.load(self.plan["seeds"][0])
        assert self.pin_contract is not None
        execution = self.plan["arms"][config["arm"]]["execution"]
        arguments = {"device": config["device"], "batch_rows": execution["batch_rows"]}
        if execution["cuda_graphs"]:
            if "cuda_graphs" not in inspect.signature(PytorchHooksEngine).parameters:
                raise ValueError(
                    "selected arm does not support the authored cuda_graphs setting"
                )
            arguments["cuda_graphs"] = True
        self.engine = PytorchHooksEngine(**arguments)
        # Full compiler/routing checks have already run before loading weights.
        from causalab.protocol.pipeline import compile_protocol, route_engine
        from causalab.protocol.lowering import DEFAULT_POINT_CAP
        from causalab.workflow.document import ProtocolStep

        # load_workflow already validates deferred entries. Independently compile
        # only dependency-free steps here; dependent execution routes at runtime.
        models = {}
        data = {}
        artifacts = {}
        for name, inner in self.loaded.inner.items():
            step = self.loaded.document.steps[name]
            if not isinstance(step, ProtocolStep):
                continue
            if not self.loaded.dependencies[name]:
                path = (self.path.parent / step.document).resolve()
                compiled = compile_protocol(
                    path,
                    env=self.env,
                    overrides=step.set,
                    point_cap=step.max_points or DEFAULT_POINT_CAP,
                )
                route_engine(compiled, self.engine)
                data.update(
                    {ref: asdict(identity) for ref, identity in compiled.data.items()}
                )
                artifacts.update(
                    {
                        a.reference: self.env.artifacts.file_digest(a.reference)
                        for a in compiled.artifacts
                        if not a.deferred
                    }
                )
            for point in inner.expansion.points:
                model = canonical_model(point.raw["model"])
                models[_digest(model)] = model
                # Attest external evaluation data before loading models, even
                # when fitted-artifact dependencies defer full compilation.
                refs = {
                    role["dataset"]
                    for role in point.raw["data"].values()
                    if isinstance(role.get("dataset"), str)
                }
                fit = point.raw.get("method", {}).get("train", {})
                split = fit.get("eval", {}).get("split") if fit else None
                if isinstance(split, str):
                    refs.add(split)
                for ref in refs:
                    data[ref] = {
                        "digest": self.env.datasets.digest(ref),
                        "columns": tuple(self.env.datasets.columns(ref)),
                    }
        self.bundles = {}
        if len(models) > 4:
            raise ValueError(
                "resident measurement exceeds the engine's four-model cache"
            )
        for key, model in models.items():
            bundle = load_model(
                model["key"],
                model["revision"],
                dtype=model["dtype"],
                device=config["device"],
                quantization=model.get("quantization"),
            )
            self.bundles[key] = bundle
        # Model/import setup may select precision policy. Capture runtime flags
        # after setup as well as verifying the initial source installation.
        self.identity = execution_identity(torch.device(config["device"]))
        self.identity.update(
            source_commit=revision,
            benchmark_identity=expected_benchmark,
            source_pins=self.pin_contract.source,
            shared_pins=self.pin_contract.shared,
            execution=execution,
            workflow=self.loaded.digest,
            data=data,
            artifacts=artifacts,
            models={
                key: {
                    "realization": models[key],
                    "resolved_revision": getattr(
                        bundle.model.config, "_commit_hash", None
                    ),
                    "configuration": _digest(bundle.model.config.to_dict()),
                    "tokenizer": _digest(bundle.tokenizer.get_vocab()),
                    "files": config["verified_model_files"][key]
                    if "verified_model_files" in config
                    else model_files(
                        models[key]["key"],
                        models[key]["revision"],
                        cache=Path(config["checkpoint_cache"]) / f"{key}.json"
                        if config.get("checkpoint_cache")
                        else None,
                        rehash=config.get("rehash_models", True),
                    ),
                    "model_class": type(bundle.model).__module__
                    + "."
                    + type(bundle.model).__qualname__,
                    "attention_configuration": getattr(
                        bundle.model.config, "_attn_implementation", None
                    ),
                    "parameter_elements": parameter_elements(bundle.model),
                }
                for key, bundle in self.bundles.items()
            },
        )
        self.identity["checkpoint_verification"] = (
            "full content hashes on startup and resume; same-study reuse guarded by inode/device/size/mtime/ctime"
        )
        self.identity["comparison_identity"] = _digest(
            {
                "shared_pins": self.pin_contract.shared,
                "data": data,
                "artifacts": artifacts,
                "models": {
                    key: {"realization": value["realization"], "files": value["files"]}
                    for key, value in self.identity["models"].items()
                },
            }
        )
        if self.plan.get("comparison", "code") == "workflow":
            self.identity.update(
                comparison_kind="workflow",
                input_identity=config["input_identity"],
                comparison_contract=config["comparison_contract"],
                configuration_changes=config["configuration_changes"],
            )
            observed_steps = {
                observation["step"]
                for observation in self.plan["observations"].values()
            }
            data_bindings = {
                name: [
                    point.raw["data"]
                    for point in self.loaded.inner[name].expansion.points
                ]
                for name in sorted(observed_steps)
            }
            self.identity["comparison_identity"] = _digest(
                workflow_inputs(self.identity, self.pin_contract.shared, data_bindings)
            )
        self.attest_execution()

    def attest_execution(self):
        """Re-probe on initial startup/resume; reuse only within an unchanged study."""
        from .probe import probe_case, reuse_evidence

        if self.config["device"].startswith("cuda") and (
            not self.identity["hardware"].get("driver_versions")
            or self.identity["hardware"]["uuid"] == "unknown"
        ):
            raise ValueError(
                "CUDA measurement requires observed driver and GPU UUID evidence"
            )
        base = _digest(self.identity)
        cached = self.config.get("execution_probe")
        evidence: dict[str, Any]
        if cached is not None:
            if cached["base_identity"] != base:
                raise ValueError(
                    "execution probe no longer matches the worker identity"
                )
            for evidence in cached["cases"].values():
                for file, expected in evidence["source_files"].items():
                    if file_hash(Path(file)) != expected:
                        raise ValueError(
                            "observed backend source changed since execution probe"
                        )
            evidence = cached
        else:
            directory = Path(self.config["probe_directory"])
            directory.mkdir(parents=True, exist_ok=False)
            records = {}
            for case in self.plan["cases"]:
                print(
                    f"[{self.config['arm']}] probing execution: {case}",
                    file=sys.stderr,
                    flush=True,
                )
                records[case] = probe_case(
                    self.prepare,
                    self.bundles,
                    directory / case,
                    case=case,
                    seed=self.plan["seeds"][0],
                    device=self.config["device"],
                    profile=(
                        case in self.plan["profile"]["cases"]
                        and "torch" in self.plan["profile"]["backends"]
                    ),
                )
            evidence = {
                "base_identity": base,
                "cases": {
                    case: reuse_evidence(record) for case, record in records.items()
                },
                "scope": "untimed resident probes; dynamic counts retained in probe.json; startup/cold-process dispatch is not profiled",
            }
            write_record(
                directory / "probe.json", {"evidence": evidence, "records": records}
            )
        self.identity["execution_probe"] = evidence
        rows = {
            case: value["logical_token_rows"]
            for case, value in evidence["cases"].items()
        }
        if self.plan.get("comparison", "code") == "workflow":
            aliases = checkpoint_aliases(self.identity["models"])
            rows = {
                case: logical_input_rows(value, aliases) for case, value in rows.items()
            }
        self.identity["comparison_identity"] = _digest(
            {
                "resolved_inputs": self.identity["comparison_identity"],
                "logical_token_rows": rows,
            }
        )
        self.identity["dispatch"] = {
            "status": "observed_untimed_probe",
            "coverage": evidence["scope"],
            "cases": list(evidence["cases"]),
        }

    def load(self, seed: int):
        from causalab.workflow.document import load_workflow

        raw = json.loads(json.dumps(self.raw))
        for name in self.fitting:
            raw["steps"][name].setdefault("set", {})["train.seed"] = seed
        authored = raw.pop("pins", None)
        loaded = load_workflow(raw, self.env, workflow_dir=self.path.parent)
        census = collect_pins(loaded, self.env.datasets)
        file_digests = self.deferred_file_digests(census, self.env)
        if self.pin_contract is None:
            contract = PinContract.resolve(
                authored,
                census,
                arm=self.config["arm"],
                source_pin_anchor=self.config.get("plan", {}).get(
                    "source_pin_anchor", "before"
                ),
                package_root=Path(self.config["package_root"]),
                file_digests=file_digests,
                comparison=getattr(self, "plan", {}).get("comparison", "code"),
            )
            if "resolved_pins" in self.config:
                check_pins(self.config["resolved_pins"], contract.pins)
            self.pin_contract = contract
            return loaded
        self.pin_contract.check(census, file_digests=file_digests)
        return loaded

    @staticmethod
    def deferred_file_digests(census, env) -> dict[str, str]:
        # The workflow loader uses placeholders for external files in dependent
        # protocols. They are real benchmark inputs even before a fit exists.
        return {
            ref: env.artifacts.file_digest(ref)
            for ref, value in census.get("files", {}).items()
            if value == "0" * 64
        }

    def load_subset(self, raw, env, *, fit_root: Path | None = None):
        """Check used resources against the full frozen arm, plus saved fits."""
        from causalab.workflow.document import load_workflow, producer_of
        from causalab.workflow.runner import OverlayArtifacts

        assert self.pin_contract is not None
        raw = {key: value for key, value in raw.items() if key != "pins"}
        loaded = load_workflow(raw, env, workflow_dir=self.path.parent)
        census = collect_pins(loaded, env.datasets)
        produced_files: set[str] = set()
        if fit_root is not None:
            assert isinstance(env.artifacts, OverlayArtifacts)
            assert env.artifacts.run_root.resolve() == fit_root.resolve()
            for ref in census.get("files", {}):
                if producer_of(ref, self.raw["steps"]) is not None:
                    path = env.artifacts.resolve_path(ref).resolve()
                    if (
                        not path.is_relative_to(fit_root.resolve())
                        or not path.is_file()
                    ):
                        raise CensusError(
                            "evaluation input escapes the saved fit tree",
                            path=f"pins.files.{ref}",
                        )
                    produced_files.add(ref)
        self.pin_contract.check_subset(
            census,
            produced_files=produced_files,
            file_digests=self.deferred_file_digests(census, env),
        )
        return loaded

    @contextmanager
    def prepare(self, case: str, seed: int, directory: Path):
        import numpy as np
        import torch

        from causalab.protocol.pipeline import compile_protocol, handoff, route_engine
        from causalab.protocol.engine import RunContext
        from causalab.io.env import ResolutionEnv
        from causalab.protocol.lowering import DEFAULT_POINT_CAP
        from causalab.workflow.runner import OverlayArtifacts, run_workflow

        random.seed(seed)
        np.random.seed(seed % 2**32)
        torch.manual_seed(seed)
        loaded = self.load(seed)
        spec = self.plan["cases"][case]
        if spec["kind"] == "workflow":
            yield Operation(
                lambda: run_workflow(
                    loaded, self.env, directory, self.engine, resume=False
                ),
                lambda result: observations(result.run_root, self.plan["observations"]),
            )
            return
        name = spec["step"]
        ancestors = set()

        def include(node):
            for dependency in loaded.dependencies[node]:
                if dependency not in ancestors:
                    ancestors.add(dependency)
                    include(dependency)

        include(name)
        root = directory / loaded.document.output_dir
        if ancestors:
            raw = json.loads(json.dumps(self.raw))
            raw["steps"] = {
                key: value for key, value in raw["steps"].items() if key in ancestors
            }
            for key in self.fitting & ancestors:
                raw["steps"][key].setdefault("set", {})["train.seed"] = seed
            prefix = self.load_subset(raw, self.env)
            run_workflow(prefix, self.env, directory, self.engine, resume=False)
        from causalab.workflow.document import ProtocolStep

        step = loaded.document.steps[name]
        if not isinstance(step, ProtocolStep):
            raise ValueError(
                "operation measurement requires an intervention_protocol step"
            )
        env = ResolutionEnv(
            datasets=self.env.datasets,
            artifacts=OverlayArtifacts(
                root, self.env.artifacts, frozenset(loaded.document.steps)
            ),
            model_info=self.env.model_info,
        )
        path = (self.path.parent / step.document).resolve()
        compiled = compile_protocol(
            path,
            env=env,
            overrides=step.set,
            point_cap=step.max_points or DEFAULT_POINT_CAP,
        )
        engine = route_engine(compiled, self.engine)
        destination = root / name
        destination.mkdir(parents=True)
        run = RunContext(env=env, output_dir=destination)
        yield Operation(
            lambda: handoff(compiled, engine, run),
            lambda result: observations(root, self.plan["observations"], step=name),
        )

    def sample(self, command: dict[str, Any]) -> Path:
        case = command["case"]
        kind = self.plan["cases"][case]["kind"]
        selected = self.plan["cases"][case].get("step")
        fitting = bool(self.fitting) if kind == "workflow" else selected in self.fitting
        scope = (
            "protocol operation: resident model; compilation/dependencies excluded; engine execution and required saves included"
            if kind == "operation"
            else "workflow wall: resident model; document load excluded; steps and required publication included"
        )

        @contextmanager
        def prepare(seed, directory):
            from .training import training_evidence

            with self.prepare(case, seed, directory) as operation:
                yield Operation(
                    operation.run,
                    operation.observe,
                    numerics_context=(
                        lambda: training_evidence(
                            device=self.config["device"],
                            required=True,
                            operation_step=selected,
                        )
                    )
                    if fitting
                    else None,
                )

        receipt = collect(
            prepare,
            Path(command["directory"]),
            case=case,
            input_identity=self.config["input_identity"],
            scope=scope,
            reset_policy=RESIDENT_RESET_POLICY,
            seeds=[command["seed"]],
            repeats=1,
            warmups=self.plan["warmups"],
            device=self.config["device"],
            profile=False,
            observation_specs=self.plan["observations"],
            mode=self.plan.get("mode", "comparison"),
            observation_policy=self.plan.get("observation_policy", "required"),
            context={"worker": self.identity},
        )
        record = json.loads(receipt.read_text())
        record["context"]["cache_policy"] = cache_provenance("resident")
        for sample in record["samples"]:
            bind_timing_outputs(
                sample,
                receipt.parent,
                self.loaded.document.output_dir,
                self.plan["observations"],
                step=selected,
                observation_policy=self.plan.get("observation_policy", "required"),
            )
        record["passes"] = (
            "required saved timing-pass outputs; no numerical-only passes"
            if self.plan.get("observation_policy") == "not_requested"
            else "required saved timing-pass outputs are primary; extracted after timing; "
            "separate diagnostic and optional profiling passes"
        )
        write_record(receipt, record)
        return receipt

    def evaluate(self, command: dict[str, Any]) -> dict[str, Any]:
        """Replay authored evaluation steps against one arm's saved fit tree."""
        import numpy as np
        import torch
        from safetensors.torch import save_file

        from causalab.io.env import ResolutionEnv
        from causalab.workflow.runner import OverlayArtifacts, run_workflow

        selected = set(command["steps"])
        if (
            not selected
            or selected & self.fitting
            or not selected <= set(self.raw["steps"])
        ):
            raise ValueError("common evaluation must select existing non-fitting steps")
        raw = json.loads(json.dumps(self.raw))
        raw["steps"] = {
            name: step for name, step in raw["steps"].items() if name in selected
        }
        for step in raw["steps"].values():
            if "after" in step:
                step["after"] = [name for name in step["after"] if name in selected]
        fit_root = Path(command["fit_root"])
        env = ResolutionEnv(
            datasets=self.env.datasets,
            artifacts=OverlayArtifacts(
                fit_root, self.env.artifacts, frozenset(self.raw["steps"])
            ),
            model_info=self.env.model_info,
        )
        loaded = self.load_subset(raw, env, fit_root=fit_root)
        seed = command["seed"]
        random.seed(seed)
        np.random.seed(seed % 2**32)
        torch.manual_seed(seed)
        directory = Path(command["directory"])
        directory.mkdir(parents=True, exist_ok=False)
        result = run_workflow(loaded, env, directory, self.engine, resume=False)
        specs = {
            name: spec
            for name, spec in self.plan["observations"].items()
            if spec["step"] in selected
        }
        values = observations(result.run_root, specs)
        from .observations import observation_specs

        specs = observation_specs(result.run_root, specs)
        artifact = directory / "observations.safetensors"
        save_file(values, str(artifact))
        return {
            "observations": {"file": str(artifact), "sha256": file_hash(artifact)},
            "observation_specs": specs,
            "identity": self.identity,
        }


def bind_timing_outputs(
    sample, directory, output_dir, specs, *, step=None, observation_policy="required"
):
    """Bind numerics and subsequent evaluation to the actual timed execution."""
    timing_outputs = f"{sample['timing_directory']}/{output_dir}"
    timing_root = directory / timing_outputs
    sample["output_files"] = {
        path.relative_to(directory).as_posix(): file_hash(path)
        for path in sorted(timing_root.rglob("*"))
        if path.is_file()
    }
    if sample["output_files"]:
        sample["workflow_outputs"] = timing_outputs
    else:
        sample.pop("workflow_outputs", None)
    if observation_policy == "not_requested":
        sample["observer_check"] = {"status": "not_requested"}
        return
    from safetensors.torch import load_file, save_file

    from .observations import observation_specs
    from .training import output_check

    sample["diagnostic_observations"] = sample["observations"]
    sample["diagnostic_workflow_outputs"] = (
        f"{sample['numerics_directory']}/{output_dir}"
    )
    sample["diagnostic_observation_specs"] = observation_specs(
        directory / sample["diagnostic_workflow_outputs"], specs, step=step
    )
    values = observations(timing_root, specs, step=step)
    artifact = directory / f"{sample['timing_directory']}.safetensors"
    save_file(values, str(artifact))
    sample["observations"] = {"file": artifact.name, "sha256": file_hash(artifact)}
    sample["observation_specs"] = observation_specs(timing_root, specs, step=step)
    sample["observation_origin"] = "timing_pass"
    # Keep the native timing reference when common evaluation extends observations.
    sample["unobserved_observations"] = dict(sample["observations"])
    sample["unobserved_observation_specs"] = dict(sample["observation_specs"])
    sample["observer_check"] = output_check(
        values, load_file(str(directory / sample["diagnostic_observations"]["file"]))
    )
    sample["observer_check"]["observation_specs_match"] = (
        sample["observation_specs"] == sample["diagnostic_observation_specs"]
    )


def serve(config: dict[str, Any]) -> None:
    protocol = sys.stdout
    with redirect_stdout(sys.stderr):
        worker = Worker(config)
        if "cold" in config:
            import torch

            command = config["cold"]
            directory = Path(command["directory"])
            directory.mkdir(parents=True, exist_ok=False)
            with worker.prepare(
                command["case"], command["seed"], directory
            ) as operation:
                from .training import training_evidence

                with (
                    training_evidence(
                        device=config["device"],
                        required=True,
                    )
                    if command.get("observe_training")
                    else nullcontext(None)
                ) as numerics_context:
                    operation.run()
                    if config["device"].startswith("cuda"):
                        torch.cuda.synchronize(config["device"])
            protocol.write(
                json.dumps(
                    {
                        "identity": worker.identity,
                        "status": "completed",
                        "numerics_context": numerics_context,
                        "cache_policy": cache_provenance("cold_process"),
                    },
                    allow_nan=False,
                )
                + "\n"
            )
            protocol.flush()
            return
    protocol.write(json.dumps({"identity": worker.identity}, allow_nan=False) + "\n")
    protocol.flush()
    for line in sys.stdin:
        command = json.loads(line)
        if command["action"] == "shutdown":
            return
        try:
            with redirect_stdout(sys.stderr):
                if command["action"] == "sample":
                    result = {"receipt": str(worker.sample(command))}
                elif command["action"] == "evaluate":
                    result = worker.evaluate(command)
                else:
                    raise ValueError("unknown measurement worker action")
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            result = {"error": f"{type(exc).__name__}: {exc}"}
        protocol.write(json.dumps(result, allow_nan=False) + "\n")
        protocol.flush()
