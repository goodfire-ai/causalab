"""``sites.py`` under a geometry above world 1 (``docs/model_parallelism.md``
§6.2, §6.6): the projection-width rule accepts exactly the global width and,
when the module carries a tensor-parallel colwise row, the local shard's
width — and refuses every other width naming both — and ``_placed`` reads
the plan off ``bundle.info.parallel_plan`` and the bundle's geometry.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from causalab.neural.shared import sites
from causalab.protocol.rules.errors import ProtocolError
from causalab.protocol.parallel import ONE, ParallelGeometry
from causalab.protocol.registry import QWEN35_MOE_PLAN, ModelInfo

# pyright: reportPrivateUsage=false

pytestmark = pytest.mark.unit

MOE_INFO = ModelInfo(
    key="tiny-random/qwen3.5-moe",
    hidden_size=8,
    num_layers=4,
    num_heads=8,
    num_kv_heads=4,
    head_dim=32,
    intermediate_size=None,
    vocab_size=2048,
    native_dtype="fp32",
    family="qwen3_5_moe_text",
    num_experts=128,
    num_experts_per_tok=10,
    shared_expert_intermediate_size=32,
    moe_intermediate_size=32,
    linear_num_key_heads=4,
    linear_num_value_heads=8,
    linear_key_head_dim=16,
    linear_value_head_dim=16,
    layer_types=("linear_attention",) * 3 + ("full_attention",),
    parallel_plan=QWEN35_MOE_PLAN,
)

#: ``attention_gate`` is split 1 of 2 of ``q_proj``: ``H · 2 · d = 512`` on
#: the fixture, ``256`` per rank at ``tp=2``.
GATE_ADDRESS = {"module": "q_proj", "packing": "fused_heads", "splits": 2, "split": 1}
GLOBAL, LOCAL = 512, 256


class _Tree(torch.nn.Module):
    """``model.layers.3.self_attn.q_proj``, the path the plan row names."""

    base_model_prefix = "model"

    def __init__(self, out_features: int) -> None:
        super().__init__()
        attn = torch.nn.Module()
        attn.q_proj = torch.nn.Linear(8, out_features, bias=False)
        block = torch.nn.Module()
        block.self_attn = attn
        inner = torch.nn.Module()
        inner.layers = torch.nn.ModuleList(
            [torch.nn.Module(), torch.nn.Module(), torch.nn.Module(), block]
        )
        self.model = inner


class _Bundle:
    def __init__(self, out_features: int, geometry: ParallelGeometry) -> None:
        self.key = MOE_INFO.key
        self.info = MOE_INFO
        self.geometry = geometry
        self.model = _Tree(out_features)

    @property
    def q_proj(self) -> Any:
        return self.model.model.layers[3].self_attn.q_proj


def _check(bundle: _Bundle) -> None:
    sites._check_projection_width(
        bundle, bundle.q_proj, GATE_ADDRESS, "attention_gate", 3
    )


class TestProjectionWidthUnderTensorParallelism:
    def test_the_global_width_is_accepted_at_every_geometry(self) -> None:
        _check(_Bundle(GLOBAL, ONE))
        _check(_Bundle(GLOBAL, ParallelGeometry(tensor=2)))

    def test_the_local_width_is_accepted_only_under_a_tensor_row_on_an_active_axis(
        self,
    ) -> None:
        _check(_Bundle(LOCAL, ParallelGeometry(tensor=2)))
        with pytest.raises(ProtocolError) as err:
            _check(_Bundle(LOCAL, ONE))
        # at world 1 the message names the one accepted width, not a shard
        assert "512" in str(err.value) and "shard" not in str(err.value)

    def test_any_other_width_is_refused_naming_both_accepted_widths(self) -> None:
        with pytest.raises(ProtocolError) as err:
            _check(_Bundle(128, ParallelGeometry(tensor=2)))
        message = str(err.value)
        assert "512" in message and "256" in message and "128" in message
        assert err.value.reason == "component_unavailable"

    def test_a_tensor_group_the_width_does_not_divide_is_refused_by_name(
        self,
    ) -> None:
        with pytest.raises(ProtocolError, match="tp=3"):
            _check(_Bundle(GLOBAL, ParallelGeometry(tensor=3)))

    def test_under_expert_parallelism_alone_the_tensor_row_is_inactive(self) -> None:
        with pytest.raises(ProtocolError):
            _check(_Bundle(LOCAL, ParallelGeometry(expert=2)))
