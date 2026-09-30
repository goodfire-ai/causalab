"""Observe fit initialization and actual batches in a separate numerical pass.

The observer reads existing objects at the engine's training boundaries. It never
replaces initialization, RNG draws, batches, gradients, or optimizer calls.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import random
import sys
from typing import Any


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def snapshot(value: Any) -> Any:
    """Hash tensor bytes now, before the optimizer can mutate them."""
    import torch

    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        return {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "sha256": hashlib.sha256(
                tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
            ).hexdigest(),
        }
    if isinstance(value, dict):
        if len({str(key) for key in value}) != len(value):
            raise ValueError("ambiguous optimizer state keys")
        return {str(key): snapshot(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [snapshot(item) for item in value]
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise ValueError(f"unsupported training state value: {type(value).__name__}")


def rng_snapshot(device: str) -> dict[str, Any]:
    import numpy as np
    import torch

    numpy_state = np.random.get_state(legacy=True)
    if not isinstance(numpy_state, tuple):
        raise ValueError("expected the legacy NumPy global RNG state")
    result = {
        "python": _digest(random.getstate()),
        "numpy": _digest([numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]]),
        "torch_cpu": snapshot(torch.get_rng_state()),
    }
    if device.startswith("cuda"):
        result["torch_device"] = snapshot(torch.cuda.get_rng_state(device))
    return result


def _campaign_digest(frame) -> str | None:
    """The campaign digest of the compiled protocol a fit runs under. The
    cohort entry receives a [`RunContext`][causalab.protocol.engine.RunContext],
    which carries no identity by design; the identity is the compiled
    protocol ``execute_request`` holds in the frame above. A fit entered
    directly (the trainer's own tests) runs under no campaign: ``None``."""
    parent = frame.f_back
    while parent is not None:
        if (
            parent.f_code.co_name == "execute_request"
            and parent.f_globals.get("__name__") == "causalab.neural.shared.execution"
        ):
            compiled = parent.f_locals.get("compiled")
            return getattr(compiled, "campaign_digest", None)
        parent = parent.f_back
    return None


@contextmanager
def training_evidence(
    *, device: str, required: bool, operation_step: str | None = None
):
    from causalab.neural.engines.pytorch_hooks import train
    from causalab.neural.engines.pytorch_hooks.executor import document_seed

    result: dict[str, Any] = {
        "scope": "separate numerical pass; training-state reads and Python observer excluded from primary timing",
        "coverage": "pytorch_hooks training entry, per-fit initialization, minibatch selection and optimizer calls on the calling thread",
        "fits": [],
    }
    active: dict[int, dict[str, Any]] = {}
    optimizer_ids: dict[int, dict[str, Any]] = {}
    steps: dict[int, str] = {}
    cohorts: dict[int, dict[str, Any]] = {}
    members: dict[int, dict[str, Any]] = {}
    cohort_entry = hasattr(train, "run_cohort_training")
    previous = sys.getprofile()
    train_module = "causalab.neural.engines.pytorch_hooks.train"

    def profile(frame, event, arg):
        if previous is not None:
            previous(frame, event, arg)
        name = frame.f_code.co_name
        if name not in (
            "_run_protocol_step",
            "run_training",
            "run_cohort_training",
            "_prepare_fit",
            "_build_optimizer",
            "reset_reads",
            "step",
        ):
            return
        namespace = frame.f_globals.get("__name__", "")
        if name == "_run_protocol_step" and namespace == "causalab.workflow.runner":
            if event == "call":
                steps[id(frame)] = frame.f_locals["name"]
            elif event == "return":
                steps.pop(id(frame), None)
            return
        if namespace == train_module and name == "run_cohort_training":
            if event == "call":
                step = operation_step or next(reversed(steps.values()), None)
                if step is None:
                    raise ValueError(
                        "training evidence needs an observed workflow step identity"
                    )
                cohorts[id(frame)] = {
                    "step": step,
                    "protocol": _campaign_digest(frame),
                    "fits": [],
                }
            elif event == "return":
                cohort = cohorts.pop(id(frame))
                for record in cohort["fits"]:
                    record["completed"] = arg is not None and len(arg) == len(
                        cohort["fits"]
                    )
                    record["rng_at_return"] = rng_snapshot(device)
                optimizer_ids.clear()
                members.clear()
            return
        if (
            namespace == train_module
            and name == "_prepare_fit"
            and event == "return"
            and arg is not None
        ):
            parent = frame.f_back
            while parent is not None and id(parent) not in cohorts:
                parent = parent.f_back
            if parent is None:
                return
            cohort = cohorts[id(parent)]
            record = {
                "identity": {
                    "step": cohort["step"],
                    "protocol": cohort["protocol"],
                    "coords": dict(arg.executor.coords),
                    "train_params": list(arg.doc.train.params),
                },
                "seed": arg.seed,
                "batches": [],
                "optimizer_steps": 0,
                "completed": False,
                "initial": {
                    "parameters": snapshot(
                        [
                            p
                            for group in arg.optimizer.param_groups
                            for p in group["params"]
                        ]
                    ),
                    "stages": {
                        key: snapshot(stage.state_dict())
                        for key, stage in arg.stages.items()
                    },
                    "optimizer_class": type(arg.optimizer).__module__
                    + "."
                    + type(arg.optimizer).__qualname__,
                    "optimizer_state": snapshot(arg.optimizer.state_dict()),
                },
                "rng_after_initialization": rng_snapshot(device),
            }
            cohort["fits"].append(record)
            result["fits"].append(record)
            members[id(arg)] = record
            optimizer_ids[id(arg.optimizer)] = record
            return
        parent = frame.f_back
        if (
            name == "reset_reads"
            and event == "call"
            and parent is not None
            and id(parent) in cohorts
        ):
            state = parent.f_locals
            member = state["fit"]
            minibatch = frame.f_locals["self"]
            if minibatch is not state.get("minibatch"):
                raise ValueError("cannot identify the actual training minibatch")
            # The engine also resets a member's reads after its update, to free
            # them (train.py, run_cohort_training). By then `position` names the
            # next minibatch, or is past the epoch's end; only the reset before
            # the update is the batch this step trains on.
            if not (
                member.position < len(member.order)
                and member.minibatch_executors[member.order[member.position]]
                is minibatch
            ):
                return
            members[id(member)]["batches"].append(
                {
                    "update": member.step,
                    "epoch": member.epoch,
                    "logical_indices": list(
                        member.batches[member.order[member.position]]
                    ),
                    "role_rows_sha256": {
                        role: _digest(rows)
                        for role, rows in minibatch.role_rows.items()
                    },
                    "order_rng_seed": member.order_rng.initial_seed(),
                    "order_rng_state": snapshot(member.order_rng.get_state()),
                    "mask_rng_state": snapshot(member.mask_rng.get_state()),
                    "rng_before_update": rng_snapshot(device),
                }
            )
            return
        if namespace == train_module and name == "run_training":
            if cohort_entry:
                return  # The wrapper delegates to the observed cohort entry.
            if event == "call":
                doc = frame.f_locals["doc"]
                executor = frame.f_locals["executor"]
                step = operation_step or next(reversed(steps.values()), None)
                if step is None:
                    raise ValueError(
                        "training evidence needs an observed workflow step identity"
                    )
                fit = {
                    "identity": {
                        "step": step,
                        "protocol": _campaign_digest(frame),
                        "coords": dict(executor.coords),
                        "train_params": list(doc.train.params),
                    },
                    "seed": document_seed(doc),
                    "batches": [],
                    "optimizer_steps": 0,
                    "completed": False,
                }
                active[id(frame)] = fit
                result["fits"].append(fit)
            elif event == "return":
                fit = active.pop(id(frame), None)
                if fit is not None:
                    fit["completed"] = arg is not None
                    fit["rng_at_return"] = rng_snapshot(device)
                    for key in [
                        key for key, value in optimizer_ids.items() if value is fit
                    ]:
                        del optimizer_ids[key]
            return
        parent = frame.f_back
        fit = active.get(id(parent)) if parent is not None else None
        if (
            namespace == train_module
            and name == "_build_optimizer"
            and event == "return"
            and fit is not None
            and arg is not None
        ):
            assert parent is not None
            fit["initial"] = {
                "parameters": snapshot(parent.f_locals["parameters"]),
                "stages": {
                    key: snapshot(stage.state_dict())
                    for key, stage in parent.f_locals["stages"].items()
                },
                "optimizer_class": type(arg).__module__ + "." + type(arg).__qualname__,
                "optimizer_state": snapshot(arg.state_dict()),
            }
            fit["rng_after_initialization"] = rng_snapshot(device)
            optimizer_ids[id(arg)] = fit
        elif name == "reset_reads" and event == "call" and fit is not None:
            assert parent is not None
            state = parent.f_locals
            mb = frame.f_locals.get("self")
            if mb is not state.get("mb"):
                raise ValueError("cannot identify the actual training minibatch")
            order_rng = state["order_rng"]
            fit["batches"].append(
                {
                    "update": state["step"],
                    "epoch": state["_epoch"],
                    "logical_indices": list(state["batches"][state["batch_index"]]),
                    "role_rows_sha256": {
                        role: _digest(rows) for role, rows in mb.role_rows.items()
                    },
                    "order_rng_seed": order_rng.initial_seed(),
                    "order_rng_state": snapshot(order_rng.get_state()),
                    "rng_before_update": rng_snapshot(device),
                }
            )
        elif (
            name == "step" and event == "call" and namespace.startswith("torch.optim.")
        ):
            fit = optimizer_ids.get(id(frame.f_locals.get("self")))
            if fit is not None:
                fit["optimizer_steps"] += 1

    try:
        sys.setprofile(profile)
        yield result
        if required and not result["fits"]:
            raise ValueError(
                "required fit initialization and schedule were not observed"
            )
        for fit in result["fits"]:
            if (
                not fit["completed"]
                or "initial" not in fit
                or not fit["batches"]
                or fit["optimizer_steps"] != len(fit["batches"])
            ):
                raise ValueError("incomplete training-state observer coverage")
    finally:
        sys.setprofile(previous)


def output_check(reference, observed) -> dict[str, Any]:
    """Raw saved-output drift against an unobserved run, not an acceptance test."""
    import torch

    if reference.keys() != observed.keys() or any(
        reference[key].shape != observed[key].shape for key in reference
    ):
        return {"status": "unaligned"}
    if any(
        not torch.isfinite(value).all()
        for values in (reference, observed)
        for value in values.values()
    ):
        return {"status": "nonfinite"}
    drift = {}
    for key, value in reference.items():
        delta = observed[key].to(torch.float64) - value.to(torch.float64)
        drift[key] = {
            "max_abs": float(delta.abs().max()),
            "rms": float(delta.square().mean().sqrt()),
        }
    return {
        "status": "compared",
        "policy": "raw saved tensor drift against the uninstrumented timed execution at the same seed; execution variability may also contribute",
        "exactly_equal": all(value["max_abs"] == 0 for value in drift.values()),
        "drift": drift,
    }
