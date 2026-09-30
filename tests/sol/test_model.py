"""Analytical invariants with hand-computable resource bounds; no accelerator."""

import unittest

import pytest

from causalab.sol.model import Hardware, Phase, Workload, catalog, reference
from causalab.sol.recipes import DenseTransformer, dense_catalog_workloads


pytestmark = pytest.mark.unit


class TestReferences(unittest.TestCase):
    def setUp(self):
        self.hw = Hardware("test", {"fp32": 100}, 100, 1000, 100, 0.01, "test fixture")

    def work(self, phase):
        return Workload("test", 8, [phase], ["test"], ["test"])

    def test_roofline_and_repeats(self):
        row = reference(
            self.work(Phase("a", 3, {"fp32": 200}, activation_bytes=100)), self.hw, 1, 1
        )
        self.assertEqual(row["sol_seconds"], 6)
        self.assertEqual(row["serialized_resources_seconds"], 9)

    def test_dp_does_not_shard_weights(self):
        row = reference(self.work(Phase("a", 1, weight_bytes=400)), self.hw, 4, 1)
        self.assertEqual(row["sol_seconds"], 4)

    def test_ring_and_gradient_sharding(self):
        row = reference(self.work(Phase("a", 1, gradient_bytes=100)), self.hw, 2, 2)
        self.assertAlmostEqual(row["sol_seconds"], 0.52)

    def test_capacity(self):
        work = self.work(Phase("a", 1, resident_sharded_bytes=1500))
        self.assertIsNone(reference(work, self.hw, 2, 1)["sol_seconds"])
        self.assertIsNone(reference(work, self.hw, 1, 2)["fits"])
        self.assertEqual(reference(work, self.hw, 1, 2)["memory_status"], "unknown")
        # An incomplete transient estimate must not reject a runnable case.
        estimate = self.work(Phase("a", 1, {"fp32": 100}, transient_bytes=10000))
        row = reference(estimate, self.hw, 1, 1)
        self.assertEqual(row["sol_seconds"], 1)
        self.assertIsNone(row["fits"])

    def test_layouts(self):
        rows = catalog([self.work(Phase("a", 1))], self.hw)["references"]
        self.assertTrue(any(r["gpus"] == 8 for r in rows))
        self.assertTrue(all(r["gpus"] <= 8 and 8 % r["dp"] == 0 for r in rows))
        with self.assertRaises(ValueError):
            reference(self.work(Phase("a", 1)), self.hw, 3, 1)

    def test_invalid_values(self):
        for value in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                Phase("a", 1, activation_bytes=value)
        with self.assertRaises(ValueError):
            Phase("a", 1, tp_payload_bytes=10)
        with self.assertRaises(ValueError):
            Phase("a", 1.5)  # pyright: ignore[reportArgumentType] -- exercise runtime rejection

    def test_recipes(self):
        model = DenseTransformer(2, 8, 16, 32, 4, 8)
        works = dense_catalog_workloads(
            model, updates=10, source_batches=2, eval_batches=3, suffix_layers=1, rank=2
        )
        self.assertEqual(len(works), 7)
        train = next(w for w in works if w.name == "dbm_train")
        self.assertEqual(train.phases[0].repeats, 2)
        self.assertEqual(train.phases[1].repeats, 10)
        self.assertEqual(train.phases[4].gradient_bytes, 32)
        self.assertLess(
            train.phases[2].flops["bf16_dense"], train.phases[1].flops["bf16_dense"]
        )
