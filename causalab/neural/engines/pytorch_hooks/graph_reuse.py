"""Request-owned, single-fit CUDA capture cache for compatible seed sweeps."""

from __future__ import annotations

import dataclasses
import json
from typing import Any, Callable

import torch

from causalab.neural.engines.pytorch_hooks.budget import OOMPolicy
from causalab.neural.engines.pytorch_hooks.cuda_graphs import (
    GraphExecutor,
    GraphPool,
    TrainingGraphs,
    graph_device,
    stage_layout,
)
from causalab.neural.engines.pytorch_hooks.executor import PointExecutor


def fit_signature(executor: PointExecutor) -> tuple[Any, ...]:
    """Only the training seed may differ; resolved data and layouts must agree.

    The cache cannot cross requests, so a model object and resolved rows also
    identify the tokenizer/artifact environment. Shape and mask differences
    within training remain guarded by TrainingGraphs' existing bucket keys.
    """
    # Shuffle RNG stays in Python; stochastic gates, phases and controllers
    # are ineligible. Featurizer init seeds remain part of the document.
    doc = executor.doc
    parsed = dataclasses.asdict(
        dataclasses.replace(
            doc,
            raw={},
            train=dataclasses.replace(doc.train, seed=0) if doc.train else None,
        )
    )
    return (
        id(executor.bundle.model),
        executor.batch_rows,
        json.dumps((parsed, executor.role_rows, executor.role_fields), sort_keys=True),
        tuple(
            (name, stage_layout(executor.stage(name)))
            for name in sorted(executor.doc.featurizers)
        ),
    )


class FitGraphCache:
    """Own at most one training bank and one held-out executor per request.

    begin/finish bracket a fit. A mismatch, fallback, or exception releases
    both banks; close also runs after request output saving, even on failure.
    There is no global cache and no retained optimizer state. The bank and
    the held-out executor capture into ``pool`` — the engine's
    [`GraphPool`][], borrowed and left open, so its segments outlive this
    request's graphs and serve the next request's — or, without one (or once
    it is closed), into a pool the cache opens itself and releases after
    both of them.
    """

    def __init__(self, pool: GraphPool | None = None) -> None:
        self.signature: tuple[Any, ...] | None = None
        self.training: TrainingGraphs | None = None
        self.pool: GraphPool | None = None
        self._shared = pool
        self.eval_executor: GraphExecutor | None = None
        self.eval_split: str | None = None
        self.reused_training_fits = 0
        self.reused_eval_fits = 0

    def begin(
        self,
        executor: PointExecutor,
        parameters: list[torch.nn.Parameter],
        *,
        oom_policy: OOMPolicy = OOMPolicy.RETRY,
    ) -> TrainingGraphs:
        signature = (
            fit_signature(executor) if isinstance(executor, GraphExecutor) else None
        )
        if signature is None or signature != self.signature:
            self.close()
        self.signature = signature
        if self.training is None:
            shared = self._shared
            self.pool = GraphPool() if shared is None or shared.closed else shared
            self.training = TrainingGraphs(
                parameters, pool=self.pool, oom_policy=oom_policy
            )
        else:
            self.training.parameters = parameters
            self.training.oom_policy = oom_policy
            self.training.keys.clear()
            self.reused_training_fits += 1
        return self.training

    def evaluation(
        self, split: str, factory: Callable[[], PointExecutor]
    ) -> PointExecutor:
        if self.eval_executor is not None and self.eval_split == split:
            self.reused_eval_fits += 1
            return self.eval_executor
        if self.eval_executor is not None:
            self.eval_executor.close()
            self.eval_executor = None
        result = factory()
        if isinstance(result, GraphExecutor):
            self.eval_executor, self.eval_split = result, split
        return result

    def finish(self, *, success: bool) -> None:
        training = self.training
        if not success or training is None or training.disabled or not training.buckets:
            self.close()
            return
        for worker, _, _ in training.buckets.values():
            torch.cuda.synchronize(graph_device(worker.bundle))
            worker.reset_reads()
        training.keys.clear()
        for parameter in training.parameters:
            parameter.grad = None
        if self.eval_executor is not None:
            self.eval_executor.reset_reads()

    def close(self) -> None:
        training, evaluation, pool = self.training, self.eval_executor, self.pool
        self.training = self.eval_executor = self.pool = None
        self.signature = self.eval_split = None
        try:
            if training is not None:
                training.close()
        finally:
            try:
                if evaluation is not None:
                    evaluation.close()
            finally:
                # both graph holders are closed: a pool the cache opened goes
                # with them; the engine's stays open for the next fit
                if pool is not None and pool is not self._shared:
                    pool.close()
