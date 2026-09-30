"""Strict, torch-free authoring for single-source and comparison measurements.

Machine paths and Python executables are deployment bindings, not study choices.
The normalized plan is part of the workflow digest only when it is authored.
"""

from __future__ import annotations

from collections.abc import Mapping
import math
import re
from typing import Any


class MeasurementSpecError(ValueError):
    pass


def _object(
    raw: Any, allowed: set[str], required: set[str], path: str
) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise MeasurementSpecError(f"{path}: expected an object")
    if set(raw) - allowed:
        raise MeasurementSpecError(
            f"{path}: unknown fields {sorted(set(raw) - allowed)}"
        )
    if required - set(raw):
        raise MeasurementSpecError(
            f"{path}: missing fields {sorted(required - set(raw))}"
        )
    return dict(raw)


def _integer(value: Any, minimum: int, path: str) -> int:
    if type(value) is not int or value < minimum:
        raise MeasurementSpecError(f"{path}: expected integer >= {minimum}")
    return value


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MeasurementSpecError(f"{path}: expected a nonempty string")
    return value


def _name(value: Any, path: str) -> str:
    value = _string(value, path)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", value):
        raise MeasurementSpecError(f"{path}: expected a filesystem-safe name")
    return value


def _workflow_path(value: Any, path: str) -> str:
    value = _string(value, path)
    if any(part in {"", ".", ".."} for part in value.split("/")) or any(
        character in '\\:*?"<>|' or ord(character) < 32 for character in value
    ):
        raise MeasurementSpecError(
            f"{path}: expected a portable relative workflow path"
        )
    return value


def _strings(
    value: Any, path: str, *, empty: bool = False, unique: bool = True
) -> list[str]:
    if not isinstance(value, list) or (not value and not empty):
        raise MeasurementSpecError(f"{path}: expected a list of strings")
    result = [_string(v, path) for v in value]
    if unique and len(set(result)) != len(result):
        raise MeasurementSpecError(f"{path}: duplicate values")
    return result


def parse_measurement(raw: Any, steps: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize a finite study; no imports, model loading, or external I/O."""
    single = isinstance(raw, Mapping) and raw.get("mode") == "single"
    shared_fields = {
        "version",
        "cases",
        "seeds",
        "repeats",
        "warmups",
        "order_seed",
        "profile",
        "observations",
    }
    value = _object(
        raw,
        shared_fields
        | (
            {"mode", "source"}
            if single
            else {
                "arms",
                "comparison",
                "evaluation",
                "acceptance",
                "bootstrap_draws",
            }
        ),
        {"version", "cases", "seeds", "repeats"}
        | ({"mode", "source"} if single else {"arms", "observations"}),
        "measurement",
    )
    if type(value["version"]) is not int or value["version"] != 1:
        raise MeasurementSpecError("measurement.version: expected 1")
    comparison = value.get("comparison", "code")
    if comparison not in ("code", "workflow"):
        raise MeasurementSpecError("measurement.comparison: expected code or workflow")
    seeds = value["seeds"]
    if not isinstance(seeds, list) or not seeds:
        raise MeasurementSpecError("measurement.seeds: expected a nonempty list")
    seeds = [_integer(s, 0, "measurement.seeds") for s in seeds]
    if len(set(seeds)) != len(seeds):
        raise MeasurementSpecError("measurement.seeds: duplicates are not replicates")
    arms_raw = (
        {"source": value["source"]}
        if single
        else _object(
            value["arms"],
            {"before", "after", "eager"},
            {"before", "after"},
            "measurement.arms",
        )
    )
    arms = {}
    for name, raw_arm in arms_raw.items():
        arm_path = "measurement.source" if single else f"measurement.arms.{name}"
        arm = _object(
            raw_arm,
            {"revision", "execution", "workflow"},
            {"revision"},
            arm_path,
        )
        if single and "workflow" in arm:
            raise MeasurementSpecError(
                f"{arm_path}.workflow: single-source measurements use the authored workflow"
            )
        if "workflow" in arm and (comparison == "code" or name == "before"):
            raise MeasurementSpecError(
                f"{arm_path}.workflow: only candidate or eager workflow comparisons may select a workflow"
            )
        if comparison == "workflow" and name == "after" and "workflow" not in arm:
            raise MeasurementSpecError(
                "measurement.arms.after.workflow: required for workflow comparisons"
            )
        revision = _string(arm["revision"], f"{arm_path}.revision")
        if revision.startswith("-") or any(c in revision for c in "\n\r\0"):
            raise MeasurementSpecError(
                f"{arm_path}.revision: not a valid revision argument"
            )
        execution = _object(
            arm.get("execution", {}),
            {"engine", "batch_rows", "cuda_graphs"},
            set(),
            f"{arm_path}.execution",
        )
        engine = execution.get("engine", "pytorch_hooks")
        if engine != "pytorch_hooks":
            raise MeasurementSpecError(
                f"{arm_path}.execution.engine: measurement currently requires the pytorch_hooks engine"
            )
        batch_rows = execution.get("batch_rows")
        if batch_rows is not None:
            batch_rows = _integer(batch_rows, 1, f"{arm_path}.execution.batch_rows")
        graphs = execution.get("cuda_graphs", False)
        if type(graphs) is not bool:
            raise MeasurementSpecError(
                f"{arm_path}.execution.cuda_graphs: cuda_graphs must be boolean"
            )
        arms[name] = {
            "revision": revision,
            "execution": {
                "engine": engine,
                "batch_rows": batch_rows,
                "cuda_graphs": graphs,
            },
        }
        if "workflow" in arm:
            arms[name]["workflow"] = _workflow_path(
                arm["workflow"], f"{arm_path}.workflow"
            )

    def step_name(name: Any, path: str) -> str:
        name = _string(name, path)
        if name not in steps:
            raise MeasurementSpecError(f"{path}: unknown workflow step {name!r}")
        step = steps[name]
        kind = step.get("type") if isinstance(step, Mapping) else step.type
        if kind != "intervention_protocol":
            raise MeasurementSpecError(
                f"{path}: requires an intervention_protocol step"
            )
        return name

    if not isinstance(value["cases"], Mapping) or not value["cases"]:
        raise MeasurementSpecError("measurement.cases: expected a nonempty object")
    cases = {}
    for name, raw_case in value["cases"].items():
        _name(name, "measurement.cases")
        case = _object(
            raw_case,
            {"kind", "step", "cold_process"},
            {"kind"},
            f"measurement.cases.{name}",
        )
        kind = _string(case["kind"], f"measurement.cases.{name}.kind")
        if kind not in {"operation", "workflow"}:
            raise MeasurementSpecError(
                "measurement case kind must be operation or workflow"
            )
        cold = case.get("cold_process", kind == "workflow")
        if type(cold) is not bool or (cold and kind != "workflow"):
            raise MeasurementSpecError(
                "cold_process must be boolean and requires a workflow case"
            )
        if kind == "operation":
            name_step = step_name(case.get("step"), f"measurement.cases.{name}.step")
            cases[name] = {"kind": kind, "step": name_step, "cold_process": False}
        else:
            if "step" in case:
                raise MeasurementSpecError("workflow case cannot select one step")
            cases[name] = {"kind": kind, "cold_process": cold}

    observations = {}
    requested = "observations" in value
    if requested and (
        not isinstance(value["observations"], Mapping) or not value["observations"]
    ):
        raise MeasurementSpecError(
            "measurement.observations: expected a nonempty object"
        )
    for name, raw_observation in value.get("observations", {}).items():
        _name(name, "measurement.observations")
        obs = _object(
            raw_observation,
            {"step", "file", "kind", "row_keys", "value", "temperature", "featurizer"},
            {"step", "file", "kind"},
            f"measurement.observations.{name}",
        )
        step = step_name(obs["step"], f"measurement.observations.{name}.step")
        file = _string(obs["file"], f"measurement.observations.{name}.file")
        if file.startswith(("/", "\\")) or ".." in file.replace("\\", "/").split("/"):
            raise MeasurementSpecError(
                "observation file must stay within the step output directory"
            )
        kind = _string(obs["kind"], f"measurement.observations.{name}.kind")
        if kind not in {"tensor", "subspace", "gate", "table"}:
            raise MeasurementSpecError(
                "observation kind must be tensor, subspace, gate or table"
            )
        normalized: dict[str, Any] = {"step": step, "file": file, "kind": kind}
        if kind == "table":
            if not file.endswith(".json"):
                raise MeasurementSpecError("table observation requires a JSON artifact")
            normalized["row_keys"] = _strings(
                obs.get("row_keys"), f"measurement.observations.{name}.row_keys"
            )
            normalized["value"] = _string(
                obs.get("value"), f"measurement.observations.{name}.value"
            )
        else:
            if not file.endswith(".safetensors"):
                raise MeasurementSpecError(
                    "tensor/subspace/gate observation requires safetensors"
                )
            if "row_keys" in obs or "value" in obs:
                raise MeasurementSpecError(
                    "row_keys and value only apply to table observations"
                )
        if kind == "gate":
            temperature = obs.get("temperature", "fit")
            if temperature != "fit" and (
                type(temperature) not in (int, float)
                or not math.isfinite(temperature)
                or temperature <= 0
            ):
                raise MeasurementSpecError(
                    "gate temperature must be 'fit' or finite and positive"
                )
            normalized["temperature"] = (
                "fit" if temperature == "fit" else float(temperature)
            )
            if "featurizer" in obs:
                normalized["featurizer"] = _string(obs["featurizer"], "gate featurizer")
        elif "temperature" in obs or "featurizer" in obs:
            raise MeasurementSpecError(
                "temperature and featurizer only apply to gate observations"
            )
        observations[name] = normalized
    for name, case in cases.items():
        if (
            requested
            and case["kind"] == "operation"
            and not any(o["step"] == case["step"] for o in observations.values())
        ):
            raise MeasurementSpecError(
                f"measurement.cases.{name}: no observations for its operation"
            )

    profile_value = value.get("profile", {})
    profile = _object(
        {"cases": []} if profile_value is False else profile_value,
        {
            "cases",
            "reason",
            "with_stack",
            "record_shapes",
            "backends",
            "timeout_seconds",
        },
        set(),
        "measurement.profile",
    )
    profiled = _strings(
        profile.get("cases", list(cases)), "measurement.profile.cases", empty=True
    )
    if set(profiled) - set(cases):
        raise MeasurementSpecError("measurement.profile.cases: unknown case")
    for field in ("with_stack", "record_shapes"):
        if type(profile.get(field, False)) is not bool:
            raise MeasurementSpecError(f"measurement.profile.{field}: expected boolean")
    from causalab.profiling import get_backend

    backend_options = profile.get("backends", {"torch": {}})
    if not isinstance(backend_options, Mapping) or not backend_options:
        raise MeasurementSpecError(
            "measurement.profile.backends: expected a nonempty object"
        )
    if any(not isinstance(name, str) for name in backend_options):
        raise MeasurementSpecError(
            "measurement.profile.backends: names must be strings"
        )
    backends = {}
    for name, options in sorted(backend_options.items()):
        if not isinstance(options, Mapping):
            raise MeasurementSpecError(
                f"measurement.profile.backends.{name}: expected an object"
            )
        options = dict(options)
        if name == "torch":
            for flag in ("record_shapes", "with_stack"):
                if (
                    flag in profile
                    and flag in options
                    and profile[flag] != options[flag]
                ):
                    raise MeasurementSpecError(
                        f"measurement.profile: conflicting {flag} settings"
                    )
                options.setdefault(flag, profile.get(flag, False))
        try:
            backends[name] = get_backend(name).normalize_options(options)
        except ValueError as exc:
            raise MeasurementSpecError(
                f"measurement.profile.backends.{name}: {exc}"
            ) from exc
    # Preserve the legacy Torch option surface and its canonical round trip.
    torch_options = backends.get("torch", {})
    if "torch" not in backends and any(
        profile.get(flag, False) for flag in ("record_shapes", "with_stack")
    ):
        raise MeasurementSpecError(
            "measurement.profile: record_shapes/with_stack require the torch backend"
        )

    evaluation = value.get("evaluation")
    if evaluation is not None:
        evaluation = _object(
            evaluation,
            {"arm", "crossed", "cases"},
            {"arm", "cases"},
            "measurement.evaluation",
        )
        if _string(evaluation["arm"], "measurement.evaluation.arm") not in arms:
            raise MeasurementSpecError("measurement.evaluation.arm: unknown arm")
        if type(evaluation.get("crossed", True)) is not bool:
            raise MeasurementSpecError(
                "measurement.evaluation.crossed: expected boolean"
            )
        selections = evaluation["cases"]
        if (
            not isinstance(selections, Mapping)
            or not selections
            or set(selections) - set(cases)
        ):
            raise MeasurementSpecError(
                "measurement.evaluation.cases must select known measurement cases"
            )
        selected = {}
        for case, names in selections.items():
            names = _strings(names, f"measurement.evaluation.cases.{case}")
            selected[case] = [
                step_name(name, f"measurement.evaluation.cases.{case}")
                for name in names
            ]
            if not any(obs["step"] in names for obs in observations.values()):
                raise MeasurementSpecError(
                    f"measurement.evaluation.cases.{case} has no declared observations"
                )
        evaluation = {
            "arm": evaluation["arm"],
            "crossed": evaluation.get("crossed", True),
            "cases": selected,
        }

    acceptance = value.get("acceptance", [])
    if not isinstance(acceptance, list):
        raise MeasurementSpecError("measurement.acceptance: expected a list")
    criteria = []
    for index, raw_criterion in enumerate(acceptance):
        criterion = _object(
            raw_criterion,
            {"name", "path", "maximum", "minimum"},
            {"name", "path"},
            f"measurement.acceptance[{index}]",
        )
        _name(criterion["name"], "acceptance name")
        _strings(criterion["path"], "acceptance path", unique=False)
        if ("minimum" in criterion) == ("maximum" in criterion):
            raise MeasurementSpecError(
                "acceptance criterion needs exactly one minimum or maximum"
            )
        bound = criterion["maximum"] if "maximum" in criterion else criterion["minimum"]
        if type(bound) not in (int, float) or not math.isfinite(bound):
            raise MeasurementSpecError("acceptance bound must be finite")
        criteria.append(criterion)
    if len({c["name"] for c in criteria}) != len(criteria):
        raise MeasurementSpecError("duplicate acceptance criterion name")

    result = {
        "version": 1,
        **({"comparison": "workflow"} if comparison == "workflow" else {}),
        "arms": arms,
        "cases": cases,
        "seeds": seeds,
        "repeats": _integer(value["repeats"], 1, "measurement.repeats"),
        "warmups": _integer(value.get("warmups", 1), 0, "measurement.warmups"),
        "order_seed": _integer(value.get("order_seed", 0), 0, "measurement.order_seed"),
        "bootstrap_draws": _integer(
            value.get("bootstrap_draws", 1000), 100, "measurement.bootstrap_draws"
        ),
        "observations": observations,
        "profile": {
            "cases": profiled,
            "reason": _string(
                profile.get("reason", "whole selected case"),
                "measurement.profile.reason",
            ),
            "with_stack": torch_options.get("with_stack", False),
            "record_shapes": torch_options.get("record_shapes", False),
            "backends": backends,
            "timeout_seconds": _integer(
                profile.get("timeout_seconds", 3600),
                1,
                "measurement.profile.timeout_seconds",
            ),
        },
        "evaluation": evaluation,
        "acceptance": criteria,
    }
    if single:
        result["mode"] = "single"
        result["source"] = result.pop("arms")["source"]
        for field in ("evaluation", "acceptance", "bootstrap_draws"):
            result.pop(field)
        if not requested:
            result.pop("observations")
    return result
