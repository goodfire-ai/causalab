"""Typed views over normalized authored plans and their execution targets."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, TypedDict


class ExecutionOptions(TypedDict):
    engine: Literal["pytorch_hooks"]
    batch_rows: int | None
    cuda_graphs: bool


class SourceSpec(TypedDict):
    revision: str
    execution: ExecutionOptions


@dataclass(eq=False)
class MeasurementPlanError(ValueError):
    reason: str

    def __str__(self) -> str:
        return f"measurement plan: {self.reason}"


@dataclass(frozen=True)
class NoObservations:
    status: Literal["not_requested"] = field(default="not_requested", init=False)


@dataclass(frozen=True)
class RequiredObservations:
    specs: Mapping[str, Mapping[str, Any]]
    status: Literal["required"] = field(default="required", init=False)

    def __post_init__(self) -> None:
        if not self.specs:
            raise ValueError("requested observations must be nonempty")


ObservationPolicy = NoObservations | RequiredObservations


@dataclass(frozen=True)
class SharedMeasurementPlan:
    cases: Mapping[str, Mapping[str, Any]]
    seeds: tuple[int, ...]
    repeats: int
    warmups: int
    order_seed: int
    profile: Mapping[str, Any]


@dataclass(frozen=True)
class SingleMeasurementPlan:
    source: SourceSpec
    shared: SharedMeasurementPlan
    observations: ObservationPolicy
    mode: Literal["single"] = field(default="single", init=False)


@dataclass(frozen=True)
class ComparisonMeasurementPlan:
    before: SourceSpec
    after: SourceSpec
    eager: SourceSpec | None
    shared: SharedMeasurementPlan
    observations: RequiredObservations
    mode: Literal["comparison"] = field(default="comparison", init=False)


MeasurementPlan = SingleMeasurementPlan | ComparisonMeasurementPlan


def plan_view(plan: Mapping[str, Any]) -> MeasurementPlan:
    """Read a parsed authored plan or its normalized execution representation."""
    timing_only = plan.get("observation_policy") == "not_requested"
    if timing_only and (plan.get("mode") != "single" or plan.get("observations") != {}):
        raise MeasurementPlanError(
            "not_requested observations require a single-source execution plan "
            "with an empty observation table"
        )
    shared = SharedMeasurementPlan(
        cases=plan["cases"],
        seeds=tuple(plan["seeds"]),
        repeats=plan["repeats"],
        warmups=plan["warmups"],
        order_seed=plan["order_seed"],
        profile=plan["profile"],
    )
    if plan.get("mode") == "single":
        return SingleMeasurementPlan(
            plan["source"],
            shared,
            RequiredObservations(plan["observations"])
            if "observations" in plan and not timing_only
            else NoObservations(),
        )
    arms = plan["arms"]
    return ComparisonMeasurementPlan(
        arms["before"],
        arms["after"],
        arms.get("eager"),
        shared,
        RequiredObservations(plan["observations"]),
    )


def execution_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Idempotently normalize a parsed plan without changing authored JSON."""
    view = plan_view(plan)
    result = dict(plan)
    if isinstance(view, SingleMeasurementPlan):
        result.update(
            arms={"source": view.source},
            source_pin_anchor="source",
            observations=dict(view.observations.specs)
            if isinstance(view.observations, RequiredObservations)
            else {},
            evaluation=None,
        )
    else:
        result["source_pin_anchor"] = "before"
    result["observation_policy"] = view.observations.status
    return result
