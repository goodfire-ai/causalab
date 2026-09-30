"""CPU guard for the world-4 / world-8 tiers' shared half
(``tests/golden/_parallel_worlds.py``; runs in default CI, loads no model):
the receipt block is the runner's, the loader rule under several axes
holds on a synthetic loader that shards by the plan row's axis and
partitions by stage — a hypothesis property over geometries — and each of
its named mutations (a rank reading a sharded parameter whole, two stages
sharing a parameter, a replica sharding under ``dp`` alone, a report of the
wrong world) is refused by name; the drift rule for pinned maxima never
passes an unpinned label.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from causalab.protocol.parallel import (
    MeshLayout,
    ParallelGeometry,
    format_geometry,
)
from causalab.protocol.receipt import parallel_record

from tests.golden import _parallel_worlds as worlds

pytestmark = pytest.mark.unit

#: The repository's hypothesis settings (``docs/model_parallelism.md`` §10).
_HYPOTHESIS_SETTINGS = settings(
    deadline=None,
    max_examples=30,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)

#: Bytes on disk of every synthetic parameter: divisible by every axis size
#: the strategy draws, so a contiguous shard is whole bytes.
BYTES = 2 * 3 * 4 * 5 * 8

_MODEL_PAIRS = [
    (1, 1),
    (2, 1),
    (1, 2),
    (2, 2),
    (4, 1),
    (1, 4),
    (2, 4),
    (4, 2),
    (8, 1),
    (1, 8),
]


@st.composite
def geometries(draw: st.DrawFn) -> ParallelGeometry:
    tensor, expert = draw(st.sampled_from(_MODEL_PAIRS))
    return ParallelGeometry(
        data=draw(st.integers(1, 2)),
        pipeline=draw(st.integers(1, 3)),
        context=draw(st.integers(1, 2)),
        tensor=tensor,
        expert=expert,
        data_mode=draw(st.sampled_from(["points", "rows"])),
    )


def _reports(geometry: ParallelGeometry, layers: int) -> list[dict[str, Any]]:
    """A synthetic loader's per-rank reports: ``layers`` blocks of an
    attention weight (sharded on the tensor axis), an expert weight (sharded
    on the expert axis) and a norm (replicated), the embedding on the first
    stage and the head on the last, each stage holding a contiguous run of
    blocks — the plan rows' axes and the stage partition of §5.3 / §6.5."""
    layout = MeshLayout(geometry)
    stages = geometry.pipeline
    out: list[dict[str, Any]] = []
    for rank in range(geometry.world):
        stage = layout.rank_in(rank, "pipeline")
        mine = range(stage * layers // stages, (stage + 1) * layers // stages)
        requested: dict[str, int] = {}
        for layer in mine:
            requested[f"layers.{layer}.attn"] = BYTES // geometry.tensor
            requested[f"layers.{layer}.experts"] = BYTES // geometry.expert
            requested[f"layers.{layer}.norm"] = BYTES
        if stage == 0:
            requested["embed"] = BYTES
        if stage == stages - 1:
            requested["head"] = BYTES
        out.append(
            {
                "rank": rank,
                "world": geometry.world,
                "bytes_requested": requested,
                "bytes_on_disk": {name: BYTES for name in requested},
                **_resident(requested),
            }
        )
    return out


def _resident(requested: dict[str, int], itemsize: int = 4) -> dict[str, Any]:
    """The residency block of a report whose parameters landed once, at
    ``itemsize`` bytes an element, on a device with no counters (gloo)."""
    elements = {name: nbytes // itemsize for name, nbytes in requested.items()}
    return {
        "device": "cpu",
        "elements_requested": dict(elements),
        "elements_resident": dict(elements),
        "itemsize_resident": {name: itemsize for name in requested},
        "bytes_resident": dict(requested),
        "bytes_total": sum(requested.values()),
        "shared": [],
        "device_bytes_allocated": None,
        "device_bytes_reserved": None,
        "bytes_unowned": None,
    }


class TestBlock:
    def test_the_block_is_the_runners_record_of_the_parsed_geometry(self) -> None:
        assert worlds.block("dp=2:rows,pp=2") == parallel_record(
            ParallelGeometry(data=2, pipeline=2, data_mode="rows"), "spawned"
        )
        assert worlds.block("tp=2,ep=4")["world"] == 4
        assert worlds.block("tp=2,ep=2", "joined")["launcher"] == "joined"

    @given(geometry=geometries())
    @_HYPOTHESIS_SETTINGS
    def test_the_block_round_trips_through_the_cli_grammar(
        self, geometry: ParallelGeometry
    ) -> None:
        text = format_geometry(geometry)
        assert worlds.geometry_of(text) == geometry
        assert worlds.block(text)["world"] == geometry.world


class TestLoadRule:
    @given(geometry=geometries(), layers=st.integers(3, 6))
    @_HYPOTHESIS_SETTINGS
    def test_a_loader_that_shards_by_axis_and_partitions_by_stage_passes(
        self, geometry: ParallelGeometry, layers: int
    ) -> None:
        reports = _reports(geometry, layers)
        assert worlds.load_problems(format_geometry(geometry), reports) == []
        assert worlds.stage_names(format_geometry(geometry), reports) == set(
            _reports(ParallelGeometry(), layers)[0]["bytes_requested"]
        )

    def test_a_rank_reading_a_sharded_parameter_whole_is_named(self) -> None:
        reports = _reports(ParallelGeometry(tensor=2, expert=4), 4)
        reports[3]["bytes_requested"]["layers.1.experts"] = BYTES
        problems = worlds.load_problems("tp=2,ep=4", reports)
        assert any("ranks 0 and 3" in p and "shard different" in p for p in problems), (
            problems
        )

    def test_a_shard_on_no_axis_of_the_geometry_is_named(self) -> None:
        reports = _reports(ParallelGeometry(tensor=2), 4)
        reports[0]["bytes_requested"]["layers.0.attn"] = BYTES // 3
        problems = worlds.load_problems("tp=2", reports)
        assert any("layers.0.attn" in p and "1/3" in p for p in problems), problems

    def test_two_stages_sharing_a_parameter_are_named(self) -> None:
        reports = _reports(ParallelGeometry(pipeline=4), 4)
        reports[2]["bytes_requested"]["layers.0.norm"] = BYTES
        reports[2]["bytes_on_disk"]["layers.0.norm"] = BYTES
        problems = worlds.load_problems("pp=4", reports)
        assert any("stages 0 and 2" in p for p in problems), problems

    def test_a_replica_sharding_under_dp_alone_is_named(self) -> None:
        reports = _reports(ParallelGeometry(data=2), 2)
        reports[1]["bytes_requested"]["layers.0.attn"] = BYTES // 2
        problems = worlds.load_problems("dp=2", reports)
        assert any("shards a parameter" in p for p in problems), problems

    def test_a_report_of_the_wrong_world_is_named(self) -> None:
        reports = _reports(ParallelGeometry(expert=4), 4)
        reports[1]["world"] = 2
        assert any("report 1" in p for p in worlds.load_problems("ep=4", reports))
        assert worlds.load_problems("ep=4", reports[:2]) == [
            "2 reports for a world of 4"
        ]

    def test_nothing_sharded_under_a_model_axis_is_named(self) -> None:
        reports = _reports(ParallelGeometry(), 2)
        for record in reports:
            record["world"] = 2
        reports = [dict(reports[0]), {**reports[0], "rank": 1}]
        assert "no parameter is sharded" in worlds.load_problems("tp=2", reports)

    def test_totals_sum_each_ranks_tables(self) -> None:
        reports = _reports(ParallelGeometry(expert=2), 2)
        totals = worlds.totals(reports)
        assert set(totals) == {"rank0", "rank1"}
        for record, total in zip(reports, totals.values()):
            assert total["bytes_requested"] == sum(record["bytes_requested"].values())
            assert total["bytes_on_disk"] == sum(record["bytes_on_disk"].values())
            assert total["bytes_requested"] < total["bytes_on_disk"]


class TestPins:
    PINNED = {"moe ep=4": {"stream": 1e-6, "experts": 1e-8}}

    def test_within_the_pin_and_the_band_passes(self) -> None:
        measured = {"stream": 1.5e-6, "experts": 5e-9}
        assert (
            worlds.pin_problems(
                "moe ep=4", measured, self.PINNED, band=2e-5, floor=1e-7
            )
            == []
        )

    def test_an_unpinned_label_never_passes(self) -> None:
        problems = worlds.pin_problems(
            "moe tp=8", {"stream": 0.0}, self.PINNED, band=2e-5, floor=1e-7
        )
        assert problems == ["moe tp=8: no pinned maxima; measured {'stream': 0.0}"]

    def test_past_twice_the_pin_or_the_band_is_named(self) -> None:
        problems = worlds.pin_problems(
            "moe ep=4",
            {"stream": 3e-6, "experts": 3e-5},
            self.PINNED,
            band=2e-5,
            floor=1e-7,
        )
        assert any("stream" in p and "twice" in p for p in problems), problems
        assert any("experts" in p and "band" in p for p in problems), problems

    def test_a_class_the_pin_does_not_name_is_named(self) -> None:
        problems = worlds.pin_problems(
            "moe ep=4", {"stream": 0.0}, self.PINNED, band=2e-5, floor=1e-7
        )
        assert any("classes" in p for p in problems), problems


class TestDocuments:
    def test_swept_moves_the_target_to_a_sweep_beside_the_document(
        self, tmp_path: Path
    ) -> None:
        document = tmp_path / "parity.json"
        document.write_text(
            json.dumps({"method": {"sites": {"target": {"layers": [3]}}}})
        )
        target = worlds.swept(document, [1, 3])
        assert target == tmp_path / "parity_swept.json"
        assert json.loads(target.read_text())["method"]["sites"]["target"] == {
            "layers": {"sweep": [1, 3]}
        }
        assert json.loads(document.read_text())["method"]["sites"]["target"] == {
            "layers": [3]
        }

    def test_receipt_problems_name_the_block_and_the_rest(self, tmp_path: Path) -> None:
        solo, parallel = tmp_path / "solo", tmp_path / "par"
        solo.mkdir(), parallel.mkdir()
        a = {"execution": {"parallel": parallel_record(ParallelGeometry()), "x": 1}}
        b = {
            "execution": {
                "parallel": parallel_record(ParallelGeometry(expert=4), "spawned"),
                "x": 1,
            }
        }
        (solo / "protocol.json").write_text(json.dumps(a))
        (parallel / "protocol.json").write_text(json.dumps(b))
        assert worlds.receipt_problems(solo, parallel, "ep=4") == []
        b["execution"]["fit_rows_resolved"] = 8
        (parallel / "protocol.json").write_text(json.dumps(b))
        problems = worlds.receipt_problems(solo, parallel, "ep=4")
        assert any("fit_rows_resolved" in p for p in problems), problems
        assert any(
            p.startswith("execution.parallel is")
            for p in worlds.receipt_problems(solo, parallel, "tp=4")
        )
