"""Real CUDA replay across fresh optimizers and a newly encountered bucket."""

import pytest
import torch

from causalab.neural.engines.pytorch_hooks.cuda_graphs import (
    GraphExecutor,
    graph_device,
)
from causalab.neural.engines.pytorch_hooks.graph_reuse import FitGraphCache
from causalab.neural.engines.pytorch_hooks.loading import load_model
from tests.neural.engines.pytorch_hooks.test_graph_reuse import executor

pytestmark = [
    pytest.mark.golden,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


class StageObjective:
    """A small differentiable oracle; capture uses the real stage and optimizer."""

    def __init__(self, point):
        self.executor = point
        self.stages = {"rot": point.stage("rot")}
        self.x = (
            torch.arange(
                16, device=graph_device(point.bundle), dtype=torch.float32
            ).reshape(1, 16)
            / 16
        )

    def for_executor(self, point):
        return StageObjective(point)

    def copy_labels(self, other):
        pass

    def __call__(self):
        features, _ = self.stages["rot"].featurize(self.x)
        return features.square().mean()


def test_reuse_then_new_bucket_preserves_fresh_optimizer_trajectory():
    # The objective tests capture/storage mechanics independently of transformer
    # kernels. Full Qwen production sweeps separately validate model numerics.
    bundle = load_model(
        "hf-internal-testing/tiny-random-LlamaForCausalLM", device="cuda"
    )
    cache = FitGraphCache()
    retained = []
    try:
        for seed in (0, 1):
            point = executor(bundle, seed=seed)
            reference = executor(bundle, seed=seed)
            torch.manual_seed(seed)
            parameters = list(point.stage("rot").parameters())
            torch.manual_seed(seed)
            expected = list(reference.stage("rot").parameters())
            optimizer = torch.optim.AdamW(parameters, lr=0.01)
            eager_optimizer = torch.optim.AdamW(expected, lr=0.01)
            bank = cache.begin(point, parameters)
            for index in [0, 0] if seed == 0 else [0, 1, 0]:
                # Distinct row counts force a second shape/mask bucket only
                # after captures have been reused by a different seed's fit.
                minibatch = GraphExecutor(
                    point.doc,
                    bundle,
                    role_rows={
                        role: rows[: index + 1]
                        for role, rows in point.role_rows.items()
                    },
                    role_fields=point.role_fields,
                    load_tensors=point.load_tensors,
                    stage_cache=point.stage_cache,
                    grad_enabled=True,
                )
                objective = StageObjective(minibatch)
                assert bank.backward(minibatch, objective)
                eager_optimizer.zero_grad(set_to_none=True)
                StageObjective(reference)().backward()
                for actual, oracle in zip(parameters, expected, strict=True):
                    torch.testing.assert_close(actual.grad, oracle.grad, rtol=0, atol=0)
                optimizer.step()
                eager_optimizer.step()
                torch.testing.assert_close(
                    point.stage("rot").weight,
                    reference.stage("rot").weight,
                    rtol=0,
                    atol=0,
                )
            if seed == 1:
                assert len(bank.buckets) == 2
                assert cache.reused_training_fits == 1
            cache.finish(success=True)
            retained.append(
                (
                    point.stage("rot"),
                    {
                        name: value.clone()
                        for name, value in point.stage("rot").state_dict().items()
                    },
                )
            )
            # The cohort runner saves outcomes after all fits have finished.
            # Reusing capture storage must never rewrite a previous fit's result.
            for stage, saved in retained:
                torch.testing.assert_close(stage.state_dict(), saved, rtol=0, atol=0)
    finally:
        cache.close()
    assert cache.training is None
