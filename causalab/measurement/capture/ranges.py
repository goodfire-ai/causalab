"""Profile-only phase ranges observed without replacing engine functions.

Python call/return boundaries provide Torch/NVTX annotations on the calling
thread; unavailable boundaries are reported. Native profiler events retain
device work and autograd activity on other threads.
"""

from __future__ import annotations

from contextlib import contextmanager
import inspect
import sys
from typing import Any


@contextmanager
def phase_ranges(engine: Any, *, bundles, operation_step: str | None = None):
    import torch

    from causalab.neural.engines.pytorch_hooks import train
    from causalab.neural.engines.pytorch_hooks.executor import PointExecutor
    from causalab.workflow import runner
    from ..runtime.probe import observe_execution
    from ..runtime.training import training_evidence

    coverage: dict[str, Any] = {
        "policy": "profile-only Python call/return observation; no function replacement; NVTX and Torch user annotations",
        "thread_coverage": "calling thread; backward worker-thread Python boundaries may be absent",
        "available": [],
        "unavailable": [],
        "calls": {},
        "optimizer_step_range": "optimizer call only; not a full forward/backward update window",
    }
    bindings = {}
    active = []
    previous = sys.getprofile()
    device = str(next(iter(bundles.values())).model.device)
    nvtx = torch.cuda.nvtx if device.startswith("cuda") else None

    def bind(owner, attr, name, *, label=None, instance=None, cls=None):
        method = getattr(owner, attr, None)
        function = (
            inspect.unwrap(getattr(method, "__func__", method))
            if callable(method)
            else None
        )
        code = getattr(function, "__code__", None)
        if code is None:
            coverage["unavailable"].append(name)
            return
        coverage["available"].append(name)
        coverage["calls"][name] = 0
        bindings.setdefault(code, []).append((name, label, instance, cls))

    if operation_step is None:
        bind(
            runner,
            "_run_protocol_step",
            "step",
            label=lambda frame: f"step:{frame.f_locals['name']}",
        )
        bind(engine, "execute", "engine", instance=engine)
    else:
        bind(
            engine,
            "execute",
            "step",
            instance=engine,
            label=lambda frame: f"step:{operation_step}",
        )
    bind(
        train,
        "run_cohort_training"
        if hasattr(train, "run_cohort_training")
        else "run_training",
        "cohort_training",
    )
    bind(PointExecutor, "_forward_group", "forward")
    bind(train, "metric_tensor", "metric")
    bind(train, "_regularizer", "regularizer")
    bind(train, "_run_eval", "eval")
    bind(torch.Tensor, "backward", "backward")
    for cls in (torch.optim.AdamW, torch.optim.Adam, torch.optim.SGD):
        bind(cls, "step", f"optimizer:{cls.__name__}", cls=cls)

    def close_range():
        _, context = active.pop()
        try:
            context.__exit__(None, None, None)
        finally:
            if nvtx is not None:
                nvtx.range_pop()

    def profile(frame, event, arg):
        if previous is not None:
            previous(frame, event, arg)
        if event == "return":
            if active and active[-1][0] == id(frame):
                close_range()
            return
        if event != "call":
            return
        for name, label, instance, cls in bindings.get(frame.f_code, []):
            receiver = frame.f_locals.get("self")
            if instance is not None and receiver is not instance:
                continue
            if cls is not None and not isinstance(receiver, cls):
                continue
            tag = label(frame) if label else name
            context = torch.profiler.record_function(tag)
            if nvtx is not None:
                nvtx.range_push(tag)
            try:
                context.__enter__()
            except BaseException:
                if nvtx is not None:
                    nvtx.range_pop()
                raise
            active.append((id(frame), context))
            coverage["calls"][name] += 1
            break

    try:
        sys.setprofile(profile)
        with observe_execution(bundles) as observed:
            coverage["execution"] = observed
            with training_evidence(
                device=device, required=False, operation_step=operation_step
            ) as training:
                training["scope"] = (
                    "separate profiling pass; observed fit identities, starts and complete executed batch/update sequence"
                )
                coverage["training"] = training
                yield coverage
    finally:
        sys.setprofile(previous)
        while active:
            close_range()
