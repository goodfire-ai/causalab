"""Validate deployment targets separately from the authored measurement plan."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(eq=False)
class BindingError(ValueError):
    field: str
    reason: str

    def __str__(self) -> str:
        return f"bindings.{self.field}: {self.reason}"


def target_bindings(bindings: Any, plan: Mapping[str, Any]) -> dict[str, Any]:
    """Return independent internal arm bindings for a validated execution plan."""
    target_field = "source" if plan.get("mode") == "single" else "arms"
    required = {target_field, "device", "data_root", "artifacts_root"}
    if not isinstance(bindings, dict) or set(bindings) != required:
        raise BindingError(target_field, f"need exactly {', '.join(sorted(required))}")
    targets = (
        {"source": bindings["source"]} if target_field == "source" else bindings["arms"]
    )
    if not isinstance(targets, dict) or set(targets) != set(plan["arms"]):
        raise BindingError(target_field, "target names must match the study")
    for name, binding in targets.items():
        if not isinstance(binding, dict) or set(binding) not in (
            {"repository", "python"},
            {"installation", "python"},
            {"source", "python"},
        ):
            raise BindingError(
                f"{target_field}.{name}",
                "needs python and exactly one repository, source bundle or prepared installation",
            )
        for key, value in binding.items():
            if not isinstance(value, str) or not value:
                raise BindingError(
                    f"{target_field}.{name}.{key}", "must be a nonempty path"
                )
    for key in ("device", "data_root", "artifacts_root"):
        if not isinstance(bindings[key], str) or not bindings[key]:
            raise BindingError(key, "must be a nonempty string")
    if bindings["device"] != "cpu" and not bindings["device"].startswith("cuda"):
        raise BindingError("device", "v1 currently measures one CPU or CUDA device")
    return {
        **{key: bindings[key] for key in ("device", "data_root", "artifacts_root")},
        "arms": {name: dict(binding) for name, binding in targets.items()},
    }


def authored_bindings(
    bindings: Mapping[str, Any], plan: Mapping[str, Any]
) -> dict[str, Any]:
    """Serialize prepared deployment bindings in their public mode-specific shape."""
    if plan.get("mode") != "single":
        return dict(bindings)
    return {
        **{key: bindings[key] for key in ("device", "data_root", "artifacts_root")},
        "source": bindings["arms"]["source"],
    }
