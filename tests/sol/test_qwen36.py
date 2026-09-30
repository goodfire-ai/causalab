"""Architecture arithmetic and hardware/unit regressions, independent of CUDA."""

from dataclasses import replace
import unittest

import pytest

from causalab.sol.hardware import h100_sxm, b200_sxm
from causalab.sol.model import Phase, Workload, catalog, reference
from causalab.sol.qwen36 import Qwen36A3B, qwen36_workloads


pytestmark = pytest.mark.unit


class TestQwen36(unittest.TestCase):
    def test_hardware_dense_and_per_gpu_conversions(self):
        h, b = h100_sxm(), b200_sxm()
        self.assertEqual(h.flops_per_second["bf16_dense"], 989.5e12)
        self.assertEqual(b.flops_per_second["bf16_dense"], 2250e12)
        self.assertEqual(b.flops_per_second["fp32"], 75e12)
        self.assertEqual(h.link_bytes_per_second, 450e9)
        self.assertEqual(b.link_bytes_per_second, 900e9)
        self.assertEqual(b.memory_bytes, 180e9)
        self.assertEqual(
            (h.collective_latency_seconds, b.collective_latency_seconds), (0, 0)
        )

    def test_parameter_shapes(self):
        p = Qwen36A3B().parameter_breakdown()
        # Routed expert matrices alone: 40*256*(1024*2048 + 2048*512).
        self.assertEqual(p["routed_experts"], 32212254720)
        self.assertEqual(p["embedding"], 508559360)
        self.assertEqual(p["lm_head"], p["embedding"])
        # Qwen tap diagram independently records 34.66B text parameters.
        self.assertAlmostEqual(sum(p.values()) / 1e9, 34.66, places=2)
        self.assertEqual(
            p["full_attention_projections"],
            10 * (8192 * 2048 + 2 * 512 * 2048 + 2048 * 4096),
        )
        self.assertEqual(p["delta_convolution"], 30 * 8192 * 4)

    def test_actual_suffix_layer_types(self):
        m = Qwen36A3B()
        # Last layer is full attention; last four are three delta + one full.
        p1, p4 = m.parameter_breakdown(39), m.parameter_breakdown(36)
        self.assertEqual(p1["delta_projections"], 0)
        self.assertEqual(
            p1["full_attention_projections"], p4["full_attention_projections"]
        )
        self.assertEqual(p4["delta_projections"], 3 * 2048 * 16448)
        self.assertEqual(
            m.suffix_backward(1, 1).resident_sharded_bytes,
            m.forward("f").resident_sharded_bytes,
        )
        self.assertEqual(
            m.operation_breakdown(39, backward=True)["bf16_attention_flops"],
            2 * m.operation_breakdown(39)["bf16_attention_flops"],
        )

    def test_expected_expert_traffic_depends_on_local_tokens(self):
        # Two experts, top-1, two global tokens: E[touched]=1.5; with DP2 -> 1.
        phase = Phase(
            "experts",
            1,
            routed_weight_bytes=200,
            routed_experts=2,
            routed_top_k=1,
            routed_tokens=2,
        )
        work = Workload("w", 2, [phase], ["uniform independent routing"], ["test"])
        single = reference(work, h100_sxm(), 1, 1)["phases"][0]
        dp = reference(work, h100_sxm(), 2, 1)["phases"][0]
        tp = reference(work, h100_sxm(), 1, 2)["phases"][0]
        self.assertEqual(single["expected_experts_per_layer"], 1.5)
        self.assertEqual(dp["weight_bytes_per_gpu"], 100)
        self.assertEqual(tp["weight_bytes_per_gpu"], 75)

    def test_qwen_layouts_and_capacities(self):
        works = qwen36_workloads(Qwen36A3B())
        h = catalog(works, h100_sxm())
        self.assertEqual(len(h["references"]), 28)
        self.assertTrue(all(r["tp"] == 1 for r in h["references"]))
        with self.assertRaises(ValueError):
            reference(works[0], h100_sxm(), 1, 3)
        small = replace(h100_sxm(), memory_bytes=60e9)
        self.assertFalse(reference(works[0], small, 1, 1)["fits"])
        with self.assertRaises(ValueError):
            reference(works[0], small, 1, 2)
        for work in works:
            self.assertLess(
                reference(work, b200_sxm(), 1, 1)["sol_seconds"],
                reference(work, h100_sxm(), 1, 1)["sol_seconds"],
            )

    def test_bad_routing_and_shapes(self):
        with self.assertRaises(ValueError):
            Phase(
                "bad",
                1,
                routed_experts=2,
                routed_top_k=3,
                routed_tokens=1,
                routed_weight_bytes=100,
            )
        with self.assertRaises(ValueError):
            Qwen36A3B(delta_algorithm="unknown")
        with self.assertRaises(ValueError):
            Qwen36A3B(batch=0)

    def test_chunk_padding_and_linear_recurrence(self):
        a, b = Qwen36A3B(sequence=64), Qwen36A3B(sequence=65)
        self.assertEqual(
            b.operation_breakdown()["fp32_delta_core_flops"],
            2 * a.operation_breakdown()["fp32_delta_core_flops"],
        )
        a, b = (
            replace(a, delta_algorithm="recurrent"),
            replace(b, delta_algorithm="recurrent"),
        )
        self.assertAlmostEqual(
            b.operation_breakdown()["fp32_delta_core_flops"]
            / a.operation_breakdown()["fp32_delta_core_flops"],
            65 / 64,
        )
