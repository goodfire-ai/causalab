"""Compare observed fit starts and schedules without treating seeds as proof."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from typing import Any, Literal


@dataclass(eq=False)
class TrainingIdentityError(ValueError):
    identity: Any
    reason: str

    def __str__(self) -> str:
        return f"observed training fit identity: {self.reason}"


def _pairing_identity(identity: Any, comparison: Literal["code", "workflow"]) -> Any:
    if comparison == "code":
        return identity
    required = {"step", "protocol", "coords", "train_params"}
    if not isinstance(identity, Mapping) or not required.issubset(identity):
        raise TrainingIdentityError(
            identity, "missing step, protocol, coords or train_params"
        )
    if (
        not isinstance(identity["step"], str)
        or not identity["step"]
        or not isinstance(identity["protocol"], str)
        or not identity["protocol"]
        or not isinstance(identity["coords"], Mapping)
        or not all(isinstance(key, str) for key in identity["coords"])
        or not isinstance(identity["train_params"], list)
        or not identity["train_params"]
        or not all(
            isinstance(param, str) and param for param in identity["train_params"]
        )
        or len(set(identity["train_params"])) != len(identity["train_params"])
    ):
        raise TrainingIdentityError(
            identity, "malformed step, protocol, coords or train_params"
        )
    return {key: value for key, value in identity.items() if key != "protocol"}


def _digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _fit_summary(fit):
    initial = fit["initial"]
    return {
        "identity": fit["identity"],
        "seed": fit["seed"],
        "initial_parameters_sha256": _digest(
            {key: initial[key] for key in ("parameters", "stages")}
        ),
        "initial_optimizer_sha256": _digest(
            {key: initial[key] for key in ("optimizer_class", "optimizer_state")}
        ),
        "schedule_sha256": _digest(
            [
                {
                    key: value
                    for key, value in batch.items()
                    if key
                    not in (
                        "order_rng_seed",
                        "order_rng_state",
                        "mask_rng_state",
                        "rng_before_update",
                    )
                }
                for batch in fit["batches"]
            ]
        ),
        "rng_sha256": _digest(
            {
                "start": fit["rng_after_initialization"],
                "batches": [
                    {
                        key: batch[key]
                        for key in (
                            "order_rng_seed",
                            "order_rng_state",
                            "mask_rng_state",
                            "rng_before_update",
                        )
                        if key in batch
                    }
                    for batch in fit["batches"]
                ],
                "end": fit["rng_at_return"],
            }
        ),
        "optimizer_steps": fit["optimizer_steps"],
    }


def compare_training(
    before, after, identities, *, comparison: Literal["code", "workflow"] = "code"
):
    paired, starts = [], {}
    for seed, repeat in identities:
        row = {"seed": seed, "repeat": repeat, "fits": []}
        arms = {}
        for arm, samples in (("before", before), ("after", after)):
            sample = samples[seed, repeat]
            context = sample.get("numerics_context")
            if not context or not context.get("fits"):
                arms[arm] = None
                continue
            fits = {}
            for fit in context["fits"]:
                key = _digest(_pairing_identity(fit.get("identity"), comparison))
                if key in fits:
                    raise TrainingIdentityError(fit["identity"], "ambiguous pairing")
                summary = _fit_summary(fit)
                fits[key] = summary
                starts.setdefault((arm, seed, key), []).append(summary)
            arms[arm] = fits
            row[arm] = {
                "scope": context["scope"],
                "observer_check": sample.get(
                    "observer_check", {"status": "not_collected"}
                ),
            }
        if any(fits is None for fits in arms.values()):
            row["status"] = (
                "not_collected"
                if all(fits is None for fits in arms.values())
                else "incomplete"
            )
        elif set(arms["before"]) != set(arms["after"]):
            row["status"] = "unaligned_fit_identities"
        else:
            row["status"] = "compared"
            for key, reference in arms["before"].items():
                candidate = arms["after"][key]
                row["fits"].append(
                    {
                        "before": reference,
                        "after": candidate,
                        "initial_parameters_match": reference[
                            "initial_parameters_sha256"
                        ]
                        == candidate["initial_parameters_sha256"],
                        "initial_optimizer_match": reference["initial_optimizer_sha256"]
                        == candidate["initial_optimizer_sha256"],
                        "logical_schedule_matches": reference["schedule_sha256"]
                        == candidate["schedule_sha256"],
                        "observed_rng_states_match": reference["rng_sha256"]
                        == candidate["rng_sha256"],
                        "optimizer_steps_match": reference["optimizer_steps"]
                        == candidate["optimizer_steps"],
                    }
                )
        paired.append(row)
    return {
        "policy": "observed diagnostic fit starts, actual logical batches and RNG boundary states; differences are confounds to inspect, not automatic rejection; matching states do not attest every random draw",
        "per_sample": paired,
        "within_seed": [
            {
                "arm": arm,
                "seed": seed,
                "fit": values[0]["identity"],
                "repeats": len(values),
                "initial_parameters_match": len(
                    {value["initial_parameters_sha256"] for value in values}
                )
                == 1
                if len(values) > 1
                else None,
                "initial_optimizer_match": len(
                    {value["initial_optimizer_sha256"] for value in values}
                )
                == 1
                if len(values) > 1
                else None,
                "logical_schedules_match": len(
                    {value["schedule_sha256"] for value in values}
                )
                == 1
                if len(values) > 1
                else None,
                "observed_rng_states_match": len(
                    {value["rng_sha256"] for value in values}
                )
                == 1
                if len(values) > 1
                else None,
                "optimizer_steps_match": len(
                    {value["optimizer_steps"] for value in values}
                )
                == 1
                if len(values) > 1
                else None,
            }
            for (arm, seed, _), values in sorted(starts.items())
        ],
    }
