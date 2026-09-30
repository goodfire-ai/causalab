"""A sharded load that converts the checkpoint's dtype
(``docs/model_parallelism.md`` §5.3 "the load's peak under a dtype
conversion"; ``weights.Prefetch`` "Converting"), on real processes over
``gloo``: the tiny Llama — stored in fp32 — held in bf16 at ``tp=2``, the
shape of loading ``google/gemma-2-9b``'s fp32 checkpoint in bf16.

Every rank's local shard is the matching chunk of the stock bf16 model,
byte for byte; the written load report names the conversion (every
parameter ``F32`` on disk, ``bfloat16`` resident) and the residency rule
over it is silent; the reader — the real planned reader behind a
`MeteredReader` — was asked for the
rank's pieces in batches, the fp32 copies alive at once never above the
largest piece it read, and every byte the plan requested went through a
converting batch (the whole fixture converts).
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from causalab.neural.engines.pytorch_hooks.loading import (
    LOAD_REPORT_VARIABLE,
    load_model,
    load_report_path,
)
from causalab.neural.engines.pytorch_hooks.residency import residency_problems
from causalab.neural.engines.pytorch_hooks.sharding import Sharding
from causalab.neural.shared.parallel.mesh import Mesh
from causalab.protocol.parallel import ParallelGeometry

from tests._helpers.sharded_world import run_world
from tests.neural.engines.pytorch_hooks.conftest import TINY_LLAMA

pytestmark = pytest.mark.smoke


def _program(rank: int, world: int, payload: Any) -> dict[str, Any]:
    import json
    import os
    import tempfile
    from pathlib import Path

    from torch.distributed.tensor import DTensor

    from causalab.neural.engines.pytorch_hooks import weights
    from tests._helpers.metered_reader import MeteredReader

    key, geometry_kwargs = payload
    geometry = ParallelGeometry(**geometry_kwargs)
    sharding = Sharding.from_mesh(Mesh.from_environment(geometry))
    assert sharding.rank == rank
    reports = Path(tempfile.mkdtemp(prefix="load-report-"))
    os.environ[LOAD_REPORT_VARIABLE] = str(reports)
    metered = MeteredReader(weights.FastersafetensorsReader(), target=torch.bfloat16)
    # the loader takes its reader from this seam; the meter wraps the real one
    weights.default_reader = lambda: metered
    bundle = load_model(key, dtype="bf16", sharding=sharding)
    model = bundle.model
    # the tensors cross a process as numpy, which has no bf16: they travel
    # widened to fp32, which every bf16 value survives exactly
    return {
        "report": json.loads(load_report_path(reports, rank).read_text()),
        "local": {
            name: parameter.to_local().detach().float().clone()
            for name, parameter in model.named_parameters()
            if isinstance(parameter, DTensor)
        },
        "placements": {
            name: tuple(p.dim if p.is_shard() else None for p in parameter.placements)  # type: ignore[attr-defined]
            for name, parameter in model.named_parameters()
            if isinstance(parameter, DTensor)
        },
        "whole": {
            name: parameter.detach().float().clone()
            for name, parameter in model.named_parameters()
            if not isinstance(parameter, DTensor)
        },
        "dtypes": sorted({str(p.dtype) for p in model.parameters()}),
        "peak": metered.peak,
        "handed": metered.handed,
        "live": metered.live,
        "calls": [
            (str(device), [(path.name, list(keys)) for path, keys in shards])
            for device, shards in metered.calls
        ],
    }


def test_tp2_llama_held_in_bf16_from_an_fp32_checkpoint_stages_the_cast_on_the_host() -> (
    None
):
    from transformers import AutoModelForCausalLM

    ranks = run_world(2, _program, (TINY_LLAMA, {"tensor": 2}))
    assert set(ranks) == {0, 1}
    stock = AutoModelForCausalLM.from_pretrained(
        TINY_LLAMA, dtype=torch.bfloat16, attn_implementation="eager"
    )
    whole = dict(stock.named_parameters())
    for rank, out in ranks.items():
        record = out["report"]
        assert out["dtypes"] == ["torch.bfloat16"]
        # the report names the conversion, and the rule over it is silent
        assert residency_problems(record) == [], (rank, residency_problems(record))
        requested = record["bytes_requested"]
        assert set(record["dtype_on_disk"]) == set(requested)
        assert set(record["dtype_on_disk"].values()) == {"F32"}
        assert {record["dtype_resident"][name] for name in requested} == {"bfloat16"}
        # every local shard is the stock bf16 model's chunk, byte for byte
        assert out["local"], "nothing was sharded"
        for name, local in out["local"].items():
            (dim,) = out["placements"][name]
            assert dim is not None
            expected = whole[name].chunk(2, dim=dim)[rank].float()
            assert torch.equal(local, expected), (rank, name)
        for name, tensor in out["whole"].items():
            assert torch.equal(tensor, whole[name].float()), (rank, name)
        # the meter: every requested byte went through a converting batch,
        # the fp32 copies alive at once never above the largest piece read,
        # nothing left alive, and more than one batch
        assert out["handed"] == sum(requested.values())
        assert out["peak"] <= max(requested.values()), (
            out["peak"],
            max(requested.values()),
        )
        assert out["live"] == 0
        assert len(out["calls"]) > 1
        assert {device for device, _ in out["calls"]} == {"cpu"}
