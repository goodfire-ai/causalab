"""CPU guard for the large model's golden (``tests/golden/_parallel/large.py``,
``tests/golden/test_parallel_large.py``; runs in default CI, loads no model):
every rule the GPU replay applies, on hand-built inputs — the documents and
their oracle, the census estimate, the memory rules (the estimate a bound,
the weights a floor, the replay within 5 %), the device gates that skip by
name, the two-node case's layout and its exact ``torchrun`` line, the
oracle-aware run plan and receipt comparison, and the record block a
capture of an oracle document yields, replaying against itself.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest
from huggingface_hub import constants as hf_constants
from hypothesis import HealthCheck, given, settings, strategies as st

from tests.golden import _parallel as par
from tests.golden import _parallel_worlds as worlds
from tests.golden._parallel import fit, inference, large, recorder
from tests.golden._parallel.bands import Measurement

pytestmark = pytest.mark.unit

_SETTINGS = settings(
    deadline=None,
    max_examples=30,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)

GIB = 1 << 30


# --------------------------------------------------------------------------- #
# the documents
# --------------------------------------------------------------------------- #


class TestDocuments:
    def test_the_two_documents_stand_on_the_pipeline_oracle(self) -> None:
        assert (
            set(large.DOCUMENTS) == {"large", "das_large"} == set(par.LARGE_DOCUMENTS)
        )
        assert set(par.DOCUMENTS).isdisjoint(large.DOCUMENTS)
        for document in large.DOCUMENTS.values():
            assert document.oracle == "pp=4" and document.oracle in document.exact
            assert document.exact == ("pp=4", "pp=8")
            assert document.realization == par.LARGE
            assert par.realization_of(document) == par.LARGE
            assert document.recorded
            assert document.load_rule is worlds.load_problems
            assert document.estimate is large.estimate
            assert document.compared == tuple(
                g for g in document.geometries if g != "pp=4"
            )
        assert large.INFERENCE.banded == ("tp=4", "tp=2,pp=2", "tp=8")
        assert large.INFERENCE.classes == inference.DENSE_CLASSES
        assert not large.INFERENCE.trains
        assert large.DAS.banded == ("tp=4",)
        assert large.DAS.classes == fit.CLASSES and large.DAS.trains
        assert large.DAS.argv == ("--fit-rows", str(fit.FIT_ROWS))
        assert large.DAS.author is fit.das_document

    def test_the_realization_is_the_registered_70b_in_bf16_at_a_front_layer(
        self,
    ) -> None:
        from causalab.protocol.registry import get_model_info

        assert par.LARGE == par.Realization(par.LARGE_MODEL, "bf16", "cuda", 3)
        info = get_model_info(par.LARGE_MODEL)
        assert info.num_layers == 80 and info.family == "llama"
        assert inference.sweep(par.LARGE) == (3, 43)
        # on different stages of every pipeline the golden runs
        for pipeline in (2, 4, 8):
            per_stage = 80 // pipeline
            assert 3 // per_stage != 43 // per_stage
        # the A3B's sweep is untouched
        assert inference.sweep(par.A3B) == (3, 7)

    def test_an_oracle_outside_the_exact_set_is_refused(self) -> None:
        with pytest.raises(ValueError, match="oracle 'tp=4' must be one of"):
            dataclasses.replace(large.INFERENCE, oracle="tp=4")

    def test_the_documents_of_the_two_rank_record_have_no_oracle(self) -> None:
        for document in par.DOCUMENTS.values():
            assert document.oracle is None and document.estimate is None
            assert document.compared == document.geometries
        assert par.DOCUMENTS["das"].trains and not par.DOCUMENTS["inference"].trains

    def test_the_dense_measure_has_no_routing_class(self, tmp_path: Path) -> None:
        """The boundary document authored from the dense template writes no
        routing table; the measure then yields the dense classes alone."""
        from tests.golden._parallel.bands import ROUTING

        assert ROUTING not in inference.DENSE_CLASSES
        assert inference.DENSE_CLASSES < frozenset(inference.CLASSES.values())
        for side in ("a", "b"):
            (tmp_path / side).mkdir()
            (tmp_path / side / par.RECEIPT).write_text("{}")
            for table in inference.TABLES:
                (tmp_path / side / table).write_text(json.dumps({"value": 1.0}))
        measured = inference.measure_run(tmp_path / "a", tmp_path / "b", par.LARGE)
        assert ROUTING not in measured.classes
        assert set(measured.classes) == {"stream"}  # the tables alone, no tensor file


# --------------------------------------------------------------------------- #
# the census estimate
# --------------------------------------------------------------------------- #


class TestEstimate:
    def test_the_estimate_is_the_pre_flights_arithmetic_per_rank(self) -> None:
        from causalab.protocol.parallel_memory import format_bytes

        pp4 = large.estimate("pp=4")
        assert set(pp4) == {"rank0", "rank1", "rank2", "rank3"}
        assert format_bytes(pp4["rank0"]["resident"]) == "33.83 GiB"
        assert format_bytes(pp4["rank0"]["footprint"]) == "53.55 GiB"
        tp8 = large.estimate("tp=8")
        assert len(tp8) == 8 and len({json.dumps(v) for v in tp8.values()}) == 1
        assert format_bytes(tp8["rank0"]["footprint"]) == "39.57 GiB"
        mixed = large.estimate("tp=2,pp=2")
        assert mixed["rank0"] == mixed["rank1"] and mixed["rank2"] == mixed["rank3"]
        for ranks in (pp4, tp8, mixed):
            for entry in ranks.values():
                assert entry["footprint"] > entry["resident"] > 0

    def test_the_table_is_the_whole_model(self) -> None:
        from causalab.protocol.parallel_memory import whole_bytes

        assert whole_bytes(large.table(), 2) == 141107412992
        assert len(large.table()) == 723


# --------------------------------------------------------------------------- #
# the memory rules
# --------------------------------------------------------------------------- #


def _peak(allocated: int, reserved: int, rank: int = 0) -> dict[str, Any]:
    return {
        "device": f"cuda:{rank}",
        "peak_bytes_allocated": allocated,
        "peak_bytes_reserved": reserved,
        "rank": rank,
        "steps": 0,
    }


ESTIMATE = {"rank0": {"resident": 30 * GIB, "footprint": 50 * GIB}}


class TestMemoryRules:
    def test_a_peak_between_the_weights_and_the_footprint_passes(self) -> None:
        assert par.memory_problems(ESTIMATE, {"rank0": _peak(34 * GIB, 36 * GIB)}) == []
        assert par.memory_problems(ESTIMATE, {"rank0": _peak(30 * GIB, 50 * GIB)}) == []

    def test_a_reserved_peak_above_the_footprint_is_named(self) -> None:
        problems = par.memory_problems(ESTIMATE, {"rank0": _peak(34 * GIB, 51 * GIB)})
        assert len(problems) == 1 and "above the estimated footprint" in problems[0]
        assert "rank0" in problems[0] and "bound does not hold" in problems[0]

    def test_an_allocated_peak_below_the_weights_is_named(self) -> None:
        problems = par.memory_problems(ESTIMATE, {"rank0": _peak(29 * GIB, 36 * GIB)})
        assert len(problems) == 1 and "below the resident weights" in problems[0]

    def test_a_missing_rank_or_a_peak_without_a_device_is_named(self) -> None:
        assert par.memory_problems(ESTIMATE, {}) == [
            "estimated ranks ['rank0'] but measured []"
        ]
        none = {
            "rank0": {
                "device": None,
                "peak_bytes_allocated": None,
                "peak_bytes_reserved": None,
            }
        }
        problems = par.memory_problems(ESTIMATE, none)
        assert problems == ["rank0: recorded no device peak (None)"]

    def test_the_replay_holds_the_allocated_peak_within_five_percent(self) -> None:
        recorded = {"rank0": _peak(40 * GIB, 41 * GIB)}
        assert (
            par.memory_replay_problems(recorded, {"rank0": _peak(42 * GIB, 60 * GIB)})
            == []
        )
        problems = par.memory_replay_problems(
            recorded, {"rank0": _peak(43 * GIB, 43 * GIB)}
        )
        assert (
            len(problems) == 1
            and "+7.5%" in problems[0]
            and "tolerance 5%" in problems[0]
        )
        assert par.memory_replay_problems(recorded, {}) == [
            "recorded ranks ['rank0'] but measured []"
        ]
        assert (
            par.memory_replay_problems(
                recorded, {"rank0": _peak(43 * GIB, 43 * GIB)}, tolerance=0.1
            )
            == []
        )

    def test_the_slack_is_the_footprint_over_the_reserved_peak(self) -> None:
        slack = par.estimate_slack(ESTIMATE, {"rank0": _peak(34 * GIB, 40 * GIB)})
        assert slack == {"rank0": 0.25}
        assert par.estimate_slack(ESTIMATE, {"rank0": _peak(1, 0)}) == {}

    @_SETTINGS
    @given(
        resident=st.integers(min_value=1, max_value=1 << 40),
        headroom=st.integers(min_value=0, max_value=1 << 40),
        allocated=st.integers(min_value=0, max_value=1 << 41),
        reserved=st.integers(min_value=0, max_value=1 << 41),
    )
    def test_the_rule_passes_exactly_the_peaks_inside_the_interval(
        self, resident, headroom, allocated, reserved
    ) -> None:
        estimate = {"rank0": {"resident": resident, "footprint": resident + headroom}}
        problems = par.memory_problems(estimate, {"rank0": _peak(allocated, reserved)})
        inside = allocated >= resident and reserved <= resident + headroom
        assert (problems == []) == inside


# --------------------------------------------------------------------------- #
# device gates, by name
# --------------------------------------------------------------------------- #


class TestGates:
    @pytest.mark.parametrize(
        ("devices", "case", "expected"),
        [
            (8, "pp=8", None),
            (8, "tp=4", None),
            (4, "tp=4", None),
            (4, "pp=8", "pp=8 needs 8 CUDA devices; this node shows 4"),
            (2, "pp=4", "pp=4 needs 4 CUDA devices; this node shows 2"),
            (2, "tp=2,pp=2", "tp=2,pp=2 needs 4 CUDA devices; this node shows 2"),
            (0, "pp=8@2x4", None),
        ],
    )
    def test_a_case_skips_by_name_below_its_world(
        self, devices, case, expected
    ) -> None:
        assert large.skip_reason(devices, case) == expected

    def test_a_case_needs_the_oracles_world_too(self) -> None:
        narrow = dataclasses.replace(
            large.INFERENCE, exact=("pp=8", "pp=4"), banded=("tp=2",), oracle="pp=8"
        )
        assert large.skip_reason(4, "tp=2", narrow) == (
            "tp=2 needs 8 CUDA devices; this node shows 4"
        )
        assert large.skip_reason(8, "tp=2", narrow) is None
        assert large.skip_reason(2, "tp=2") is None  # no document: its own world alone

    def test_devices_needed_is_the_largest_world(self) -> None:
        assert large.devices_needed(large.INFERENCE) == 8
        assert large.devices_needed(large.DAS) == 8
        assert large.devices_needed(par.DOCUMENTS["inference"]) == 2

    @_SETTINGS
    @given(
        devices=st.integers(min_value=0, max_value=16),
        case=st.sampled_from(
            ("pp=4", "pp=8", "tp=4", "tp=8", "tp=2,pp=2", "pp=8@2x4", "tp=8@2x4")
        ),
    )
    def test_the_gate_is_exactly_the_world_against_the_count(
        self, devices, case
    ) -> None:
        reason = large.skip_reason(devices, case)
        if large.is_two_node(case):
            assert reason is None
        else:
            assert (reason is None) == (devices >= par.world_of(case))


# --------------------------------------------------------------------------- #
# across two nodes
# --------------------------------------------------------------------------- #


class TestTwoNodes:
    def test_the_case_spelling(self) -> None:
        assert large.TWO_NODE_CASES == ("pp=8@2x4", "tp=8@2x4")
        assert large.is_two_node("pp=8@2x4") and not large.is_two_node("pp=8")
        assert large.geometry_of_case("pp=8@2x4") == "pp=8"
        assert large.geometry_of_case("tp=4") == "tp=4"
        assert large.two_node_out(Path("/root"), large.INFERENCE, "pp=8@2x4") == Path(
            "/root/large/pp8_2x4"
        )
        assert large.two_node_out(Path("/root"), large.DAS, "tp=8@2x4") == Path(
            "/root/das_large/tp8_2x4"
        )

    def test_the_torchrun_line_is_the_spawns_arguments_under_a_c10d_join(self) -> None:
        authored = Path("/root/large/parity.json")
        out = Path("/root/large/pp8_2x4")
        argv = large.torchrun_argv(
            large.INFERENCE,
            authored,
            out,
            "pp=8@2x4",
            node_rank=1,
            endpoint="192.0.2.1:29500",
        )
        head = argv[: argv.index("run")]
        assert head == [
            ".venv/bin/python",
            "-m",
            "torch.distributed.run",
            "--nnodes",
            "2",
            "--nproc-per-node",
            "4",
            "--node-rank",
            "1",
            "--rdzv-backend",
            "c10d",
            "--rdzv-endpoint",
            "192.0.2.1:29500",
            "-m",
            recorder.__name__,
        ]
        tail = argv[argv.index("run") :]
        assert tail == [
            "run",
            str(authored),
            "--engine",
            "pytorch_hooks",
            "--record",
            "--data-root",
            str(par.runs.DATA),
            "--artifacts-root",
            "/root/large",
            "--out",
            str(out),
            "--device",
            "cuda",
            "--parallel",
            "pp=8",
        ]
        fit_argv = large.torchrun_argv(
            large.DAS, authored, out, "tp=8@2x4", node_rank=0, endpoint="h:1"
        )
        assert fit_argv[-4:] == ["--fit-rows", "16", "--parallel", "tp=8"]
        assert (
            "--node-rank" in fit_argv
            and fit_argv[fit_argv.index("--node-rank") + 1] == "0"
        )

    def test_a_plain_case_or_a_third_node_is_refused(self) -> None:
        with pytest.raises(ValueError, match="not a two-node case"):
            large.torchrun_argv(
                large.INFERENCE,
                Path("d"),
                Path("o"),
                "pp=8",
                node_rank=0,
                endpoint="h:1",
            )
        with pytest.raises(ValueError, match="node_rank must be 0 or 1"):
            large.torchrun_argv(
                large.INFERENCE,
                Path("d"),
                Path("o"),
                "pp=8@2x4",
                node_rank=2,
                endpoint="h:1",
            )

    def test_the_environment_is_the_spawns_plus_the_nodes_interface_and_cache(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HF_HUB_CACHE", "/cache/hub")
        out = Path("/root/large/pp8_2x4")
        env = large.torchrun_environment(out, large.INFERENCE)
        assert env["HF_HUB_OFFLINE"] == env["TRANSFORMERS_OFFLINE"] == "1"
        assert env["HF_HUB_CACHE"] == large.hub_cache() == "/cache/hub"
        assert env["NCCL_SOCKET_IFNAME"] == env["GLOO_SOCKET_IFNAME"] == "eth0"
        assert (
            env["CUDA_VISIBLE_DEVICES"] == "0,1,2,3" and env["OMP_NUM_THREADS"] == "1"
        )
        assert env["CAUSALAB_LOAD_REPORT_DIR"] == str(out / par.REPORTS)
        assert env[recorder.GRADIENTS_VARIABLE] == str(out / par.GRADIENTS)
        assert par.GRADIENT_AGREEMENT_VARIABLE not in env  # inference: no §7 check
        fit_env = large.torchrun_environment(out, large.DAS)
        assert fit_env[par.GRADIENT_AGREEMENT_VARIABLE] == par.gradient_agreement()

    def test_without_hf_hub_cache_the_ranks_read_huggingface_hubs_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("HF_HUB_CACHE", raising=False)
        env = large.torchrun_environment(Path("/root/large/pp8_2x4"), large.INFERENCE)
        assert env["HF_HUB_CACHE"] == hf_constants.HF_HUB_CACHE

    def test_an_absent_two_node_run_skips_by_name_and_a_present_one_does_not(
        self, tmp_path: Path
    ) -> None:
        reason = large.two_node_skip(None, large.INFERENCE, "pp=8@2x4")
        assert reason is not None and "CAUSALAB_PARALLEL_GOLDENS_ROOT" in reason
        reason = large.two_node_skip(tmp_path, large.INFERENCE, "pp=8@2x4")
        assert (
            reason is not None and "commands --root" in reason and "pp8_2x4" in reason
        )
        out = large.two_node_out(tmp_path, large.INFERENCE, "pp=8@2x4")
        out.mkdir(parents=True)
        (out / par.RECEIPT).write_text("{}")
        assert large.two_node_skip(tmp_path, large.INFERENCE, "pp=8@2x4") is None

    def test_the_commands_entry_prints_both_nodes_lines_for_every_two_node_case(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert (
            large.main(
                ["commands", "--root", str(tmp_path), "--endpoint", "192.0.2.1:29500"]
            )
            == 0
        )
        out = capsys.readouterr().out
        # inference: pp=8 and tp=8, two nodes each; the fit: pp=8 alone (tp=8 is not its geometry)
        assert out.count("# large pp=8@2x4, node") == 2
        assert out.count("# large tp=8@2x4, node") == 2
        assert out.count("# das_large pp=8@2x4, node") == 2
        assert "# das_large tp=8@2x4" not in out
        assert out.count("--rdzv-endpoint 192.0.2.1:29500") == 6
        assert (tmp_path / "large" / "parity.json").exists()
        assert (tmp_path / "das_large" / "das_attention_query.json").exists()
        authored = json.loads((tmp_path / "large" / "parity.json").read_text())
        assert authored["model"] == {
            "key": par.LARGE_MODEL,
            "revision": "main",
            "dtype": "bf16",
        }
        assert authored["method"]["sites"]["target"]["layers"] == {"sweep": [3, 43]}
        assert (
            "idx" not in authored["method"]["sites"]
        )  # the dense template: no routing


# --------------------------------------------------------------------------- #
# the oracle-aware runs and receipts
# --------------------------------------------------------------------------- #


def _receipt(launcher: str, geometry: str | None, **rest: Any) -> dict[str, Any]:
    block = (
        par.parallel_block(geometry, launcher)
        if geometry is not None
        else {"launcher": "solo", "geometry": "dp=1,pp=1,cp=1,tp=1,ep=1", "world": 1}
    )
    return {"execution": {"parallel": block, **rest}, "points": [1, 2]}


class TestOracleRuns:
    def test_run_all_runs_no_solo_and_names_the_oracles_run_as_the_reference(
        self, tmp_path: Path
    ) -> None:
        seen: list[str] = []

        def author(base: Path, realization: par.Realization) -> Path:
            seen.append(realization.model)
            for geometry in ("pp=4", "pp=8", "tp=4"):
                (base / par.out_name(geometry)).mkdir(exist_ok=True)
                (base / par.out_name(geometry) / par.RECEIPT).write_text("{}")
            return base / "doc.json"

        document = dataclasses.replace(
            large.INFERENCE, author=author, exact=("pp=4", "pp=8"), banded=("tp=4",)
        )
        outputs = par.run_all(tmp_path, document)
        assert set(outputs) == {"solo", "pp=4", "pp=8", "tp=4"}
        assert outputs["solo"] == outputs["pp=4"] == tmp_path / "large" / "pp4"
        assert not (tmp_path / "large" / "solo").exists()
        assert seen == [par.LARGE_MODEL]

    def test_receipts_agree_against_an_oracle_and_under_a_join(
        self, tmp_path: Path
    ) -> None:
        reference, parallel = tmp_path / "ref", tmp_path / "par"
        reference.mkdir()
        parallel.mkdir()
        (reference / par.RECEIPT).write_text(json.dumps(_receipt("spawned", "pp=4")))
        (parallel / par.RECEIPT).write_text(json.dumps(_receipt("spawned", "pp=8")))
        assert (
            par.receipts_agree(reference, parallel, "pp=8", reference_geometry="pp=4")
            == []
        )
        # the world-1 rule still applies without an oracle
        problems = par.receipts_agree(reference, parallel, "pp=8")
        assert problems == ["the oracle was not a solo run"]
        # a reference that is not the oracle's run is named
        problems = par.receipts_agree(
            reference, parallel, "pp=8", reference_geometry="tp=4"
        )
        assert len(problems) == 1 and "not the oracle's" in problems[0]
        # a torchrun join carries its own launcher
        (parallel / par.RECEIPT).write_text(json.dumps(_receipt("joined", "pp=8")))
        assert (
            par.receipts_agree(
                reference,
                parallel,
                "pp=8",
                reference_geometry="pp=4",
                launcher="joined",
            )
            == []
        )
        problems = par.receipts_agree(
            reference, parallel, "pp=8", reference_geometry="pp=4"
        )
        assert len(problems) == 1 and "execution.parallel is" in problems[0]
        # anything else that differs is named
        (parallel / par.RECEIPT).write_text(
            json.dumps(_receipt("spawned", "pp=8", fit_rows_resolved=3))
        )
        problems = par.receipts_agree(
            reference, parallel, "pp=8", reference_geometry="pp=4"
        )
        assert problems == ["the receipts differ beyond execution.parallel"]


# --------------------------------------------------------------------------- #
# the record block of an oracle document
# --------------------------------------------------------------------------- #


def _measured(value: float, scale: float, dtype: str) -> par.Measured:
    measured = par.Measured()
    for kind in sorted(large.INFERENCE.classes):
        measured.add(kind, f"{kind}.safetensors", Measurement(value, scale, dtype))
    return measured


def _block() -> dict[str, Any]:
    peaks = {
        g: {
            f"rank{r}": _peak(e["resident"] + GIB, e["resident"] + 2 * GIB, r)
            for r, e in enumerate(large.estimate(g).values())
        }
        for g in large.INFERENCE.geometries
    }
    return par.document_record(
        large.INFERENCE,
        par.LARGE,
        {g: _measured(0.5, 20.0, "bf16") for g in large.INFERENCE.banded},
        {"pp=8": 0.0},
        {
            g: {"rank0": {"bytes_requested": 1, "bytes_on_disk": 2}}
            for g in large.INFERENCE.geometries
        },
        peaks,
        {},
        with_context=False,
        estimates={g: large.estimate(g) for g in large.INFERENCE.geometries},
    )


class TestRecordBlock:
    def test_the_block_names_the_oracle_the_realization_and_the_estimate(self) -> None:
        block = _block()
        assert block["oracle"] == "pp=4"
        assert block["realization"] == {
            "model": par.LARGE_MODEL,
            "dtype": "bf16",
            "device": "cuda",
        }
        assert set(block["exact"]) == {"pp=8"}  # the oracle is compared to nothing
        assert set(block["geometries"]) == set(large.INFERENCE.banded)
        assert (
            set(block["estimate"])
            == set(block["memory"])
            == set(large.INFERENCE.geometries)
        )
        assert block["estimate"]["tp=8"] == large.estimate("tp=8")
        assert "gradient_agreement" not in block  # an inference document trains nothing
        for geometry in large.INFERENCE.geometries:
            assert (
                par.memory_problems(
                    block["estimate"][geometry], block["memory"][geometry]
                )
                == []
            )
        fit_block = par.document_record(
            large.DAS, par.LARGE, {}, {"pp=8": 0.0}, with_context=False
        )
        assert fit_block["oracle"] == "pp=4" and "gradient_agreement" in fit_block

    def test_the_block_replays_against_itself_beside_the_a3b_record(self) -> None:
        record = par.make_record(par.load_record(), par.A3B, {"large": _block()})
        assert record["model"] == par.MODEL  # the record's realization stays the A3B's
        assert set(par.DOCUMENTS) <= set(record["documents"]) or not par.captured(
            record, "das"
        )
        assert par.compare_records(record, record, "large") == []
        moved = json.loads(par.render(record))
        moved["documents"]["large"]["oracle"] = "pp=8"
        problems = par.compare_records(record, moved, "large")
        assert problems == ["large: oracle 'pp=8' != committed 'pp=4'"]
        drifted = json.loads(par.render(record))
        drifted["documents"]["large"]["geometries"]["tp=4"]["stream"][
            "max_abs_diff"
        ] = 2.5
        problems = par.compare_records(record, drifted, "large")
        assert any("large tp=4 stream: 2.5 > band" in p for p in problems), problems

    def test_the_committed_record_holds_the_large_blocks_to_these_rules_once_captured(
        self,
    ) -> None:
        record = par.load_record()
        for name, document in large.DOCUMENTS.items():
            if not par.captured(record, name):
                continue
            block = record["documents"][name]
            assert block["oracle"] == document.oracle
            assert set(block["exact"]) == {
                g for g in document.exact if g != document.oracle
            }
            for geometry in document.geometries:
                assert block["estimate"][geometry] == large.estimate(geometry)
                assert (
                    par.memory_problems(
                        block["estimate"][geometry], block["memory"][geometry]
                    )
                    == []
                )
            assert par.compare_records(record, record, name) == []
