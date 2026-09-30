"""CPU guard for the memory soak (``docs/model_parallelism.md`` §10.6 "the
soak"; runs in default CI, loads no model on the GPU tiers' path).

The rule (``causalab/neural/shared/parallel/soak.py``, torch-free) is held
as hypothesis properties on hand-built traces: a flat series passes, a
leak of one tensor per point fails naming the slope, a sawtooth within the
slack passes, a reserved pool that climbs every point fails naming the
rise, a tail too short to fit is refused by name rather than passed. The
trace line round-trips through its JSON spelling and a line missing a
field is refused by name. The recorder's point-loop wrapper writes one
line per point per rank (in process, off any device: zero bytes). The
soak document authored for the tiny MoE expands to layers × two
positions, eight points, and — the harness's own smoke, over ``gloo`` —
runs recorded at ``ep=2`` through the production spawn path, every rank's
trace one line per point, the rule passing on a device-less trace.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from causalab.neural.shared import execution as execution_module
from causalab.neural.shared.parallel import soak as rule
from causalab.neural.shared.parallel.soak import (
    Sample,
    TraceLine,
    fit,
    flat_after_warmup,
    read_trace,
    slope,
)
from causalab.neural.shared.sweep import enumerate_steps
from causalab.protocol.pipeline import compile_protocol
from tests.golden import _parallel as par
from tests.golden._parallel import recorder, soak
from tests.neural.engines.pytorch_hooks.conftest import TINY_QWEN35_MOE
from tests.protocol._env import FIXTURES, build_env

_SETTINGS = settings(
    deadline=None,
    max_examples=30,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)

BYTES = 1 << 40
bytes_ = st.integers(min_value=0, max_value=BYTES)
warmups = st.integers(min_value=0, max_value=8)
slacks = st.integers(min_value=0, max_value=1 << 24)
lengths = st.integers(min_value=2, max_value=40)


def _flat(base: int, reserved: int, n: int) -> list[Sample]:
    return [(base, reserved)] * n


# --------------------------------------------------------------------------- #
# the rule: properties
# --------------------------------------------------------------------------- #


@pytest.mark.property
class TestRule:
    @_SETTINGS
    @given(base=bytes_, reserved=bytes_, warmup=warmups, n=lengths, slack=slacks)
    def test_a_flat_series_passes(
        self, base: int, reserved: int, warmup: int, n: int, slack: int
    ) -> None:
        samples = _flat(base, reserved, warmup + n)
        assert flat_after_warmup(samples, warmup=warmup, slack=slack) == []
        measured = fit(samples, warmup=warmup)
        assert measured.allocated_slope == 0.0 and measured.reserved_rise == 0

    @_SETTINGS
    @given(
        base=bytes_,
        reserved=bytes_,
        warmup=warmups,
        n=lengths,
        slack=slacks,
        extra=st.integers(min_value=1, max_value=1 << 30),
    )
    def test_a_leak_of_one_tensor_per_point_fails_naming_the_slope(
        self, base: int, reserved: int, warmup: int, n: int, slack: int, extra: int
    ) -> None:
        """Every point after warm-up leaves ``slack + extra`` more bytes
        allocated: the slope is exactly that, above the slack."""
        leak = slack + extra
        samples = _flat(base, reserved, warmup) + [
            (base + i * leak, reserved) for i in range(n)
        ]
        problems = flat_after_warmup(samples, warmup=warmup, slack=slack)
        assert len(problems) == 1 and "allocated bytes grow" in problems[0]
        assert f"{leak:.0f} B/point" in problems[0] or f"{leak} B/point" in problems[0]

    @_SETTINGS
    @given(
        base=bytes_,
        reserved=bytes_,
        warmup=warmups,
        n=lengths,
        slack=st.integers(min_value=1, max_value=1 << 24),
        period=st.integers(min_value=2, max_value=7),
    )
    def test_a_sawtooth_within_the_slack_passes(
        self, base: int, reserved: int, warmup: int, n: int, slack: int, period: int
    ) -> None:
        """Peaks ``slack`` above the troughs, any period: the slope of a
        series is bounded by its range, so it stays within the slack."""
        samples = _flat(base, reserved, warmup) + [
            (base + (slack if i % period else 0), reserved) for i in range(n)
        ]
        assert flat_after_warmup(samples, warmup=warmup, slack=slack) == []

    @_SETTINGS
    @given(
        base=bytes_,
        reserved=bytes_,
        warmup=warmups,
        n=lengths,
        slack=slacks,
        step=st.integers(min_value=1, max_value=1 << 30),
    )
    def test_a_reserved_pool_that_climbs_every_point_fails_naming_the_rise(
        self, base: int, reserved: int, warmup: int, n: int, slack: int, step: int
    ) -> None:
        climb = slack + step
        samples = _flat(base, reserved, warmup) + [
            (base, reserved + i * climb) for i in range(n)
        ]
        problems = flat_after_warmup(samples, warmup=warmup, slack=slack)
        assert (
            len(problems) == 1 and "reserved bytes climb monotonically" in problems[0]
        )
        assert str(climb * (n - 1)) in problems[0]

    @_SETTINGS
    @given(base=bytes_, reserved=bytes_, warmup=warmups, n=lengths, slack=slacks)
    def test_a_pool_that_opened_one_segment_within_the_slack_passes(
        self, base: int, reserved: int, warmup: int, n: int, slack: int
    ) -> None:
        """One step up of at most ``slack × steps`` and flat after: not a
        climb every point."""
        rise = slack * (n - 1)
        samples = (
            _flat(base, reserved, warmup)
            + [(base, reserved)]
            + [(base, reserved + rise) for _ in range(n - 1)]
        )
        assert flat_after_warmup(samples, warmup=warmup, slack=slack) == []

    @_SETTINGS
    @given(base=bytes_, reserved=bytes_, warmup=warmups, short=st.integers(0, 1))
    def test_too_short_a_tail_is_refused_by_name_never_passed(
        self, base: int, reserved: int, warmup: int, short: int
    ) -> None:
        samples = _flat(base, reserved, warmup + short)
        problems = flat_after_warmup(samples, warmup=warmup, slack=0)
        assert len(problems) == 1 and "at least two samples" in problems[0]
        with pytest.raises(ValueError, match="at least two samples"):
            fit(samples, warmup=warmup)

    @_SETTINGS
    @given(values=st.lists(st.integers(-BYTES, BYTES), min_size=2, max_size=40))
    def test_the_slope_is_bounded_by_the_range(self, values: list[int]) -> None:
        assert abs(slope(values)) <= max(values) - min(values) + 1e-6

    @_SETTINGS
    @given(a=st.integers(-BYTES, BYTES), b=st.integers(-(1 << 30), 1 << 30), n=lengths)
    def test_the_slope_of_a_line_is_its_step(self, a: int, b: int, n: int) -> None:
        assert slope([a + i * b for i in range(n)]) == pytest.approx(b)


# --------------------------------------------------------------------------- #
# the trace line, the recorder's wrapper, the document
# --------------------------------------------------------------------------- #


@pytest.mark.unit
class TestTrace:
    def test_a_line_round_trips_and_a_missing_field_is_refused_by_name(
        self, tmp_path: Path
    ) -> None:
        line = TraceLine(rank=1, point=7, point_digest="d" * 8, allocated=5, reserved=9)
        assert TraceLine.parse(line.render()) == line
        assert line.sample() == (5, 9)
        path = tmp_path / "rank1.jsonl"
        path.write_text(line.render() + "\n" + line.render() + "\n")
        assert read_trace(path) == [line, line]
        with pytest.raises(ValueError, match="lacks \\['reserved'\\]"):
            TraceLine.parse(
                json.dumps({"rank": 1, "point": 0, "point_digest": "", "allocated": 1})
            )
        with pytest.raises(ValueError, match="not an int"):
            TraceLine.parse(
                json.dumps(
                    {
                        "rank": 1,
                        "point": 0,
                        "point_digest": "",
                        "allocated": 1.5,
                        "reserved": 0,
                    }
                )
            )

    def test_the_wrapper_writes_one_line_per_point_for_this_rank(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Installed once; every call of the point loop's seam appends a
        line with the rank read at write time and the point's digest;
        off a device the sample is zero bytes."""
        calls: list[str] = []

        def fake_point(member: Any, request: Any, **kwargs: Any) -> dict[str, str]:
            calls.append(member.point_digest)
            return {"point": member.point_digest}

        monkeypatch.setattr(execution_module, "_execute_point", fake_point)
        monkeypatch.setenv(recorder.MEMORY_TRACE_VARIABLE, str(tmp_path))
        monkeypatch.setenv("RANK", "1")
        recorder.install_trace()
        recorder.install_trace()  # a second install (the child's twin import) is the identity

        class Member:
            def __init__(self, digest: str) -> None:
                self.point_digest = digest

        for digest in ("a", "b", "c"):
            assert execution_module._execute_point(Member(digest), None) == {
                "point": digest
            }  # pyright: ignore[reportPrivateUsage]
        assert calls == ["a", "b", "c"]
        lines = read_trace(recorder.trace_path(tmp_path, 1))
        assert [(line.rank, line.point, line.point_digest) for line in lines] == [
            (1, 0, "a"),
            (1, 1, "b"),
            (1, 2, "c"),
        ]
        assert all(line.sample() == (0, 0) and line.device is None for line in lines)
        assert not recorder.trace_path(tmp_path, 0).exists()
        assert soak.problems([rule.samples_of(lines)], warmup=1) == []

    def test_the_flag_is_taken_off_the_cli_arguments(self) -> None:
        assert recorder.split_trace_flag(["run", "doc.json"]) == (
            ["run", "doc.json"],
            None,
        )
        assert recorder.split_trace_flag(
            [recorder.MEMORY_TRACE_FLAG, "/t", "run", "doc.json"]
        ) == (["run", "doc.json"], "/t")
        with pytest.raises(SystemExit):
            recorder.split_trace_flag([recorder.MEMORY_TRACE_FLAG])

    def test_the_soak_document_is_layers_times_two_positions(
        self, tmp_path: Path
    ) -> None:
        """On the tiny MoE: four layers × two positions, eight points; on
        the A3B the budget bounds it to fifty; every other fact the
        inference document's."""
        realization = par.Realization(TINY_QWEN35_MOE, "fp32", "cpu", 3)
        from causalab.cli import register_model_key

        authored = soak.author(tmp_path, realization)
        raw = json.loads(authored.read_text())
        register_model_key(raw)
        root = tmp_path / "artifacts"
        shutil.copytree(FIXTURES / "artifacts", root, dirs_exist_ok=True)
        compiled = compile_protocol(authored, env=build_env(root))
        assert (
            len(enumerate_steps(compiled).points)
            == 8
            == soak.describe(realization)["soak"]["points"]
        )
        assert soak.layers_swept(realization) == 4
        assert soak.layers_swept(par.A3B) == 25
        assert soak.describe(par.A3B)["soak"]["points"] == soak.POINTS == 50
        assert raw["method"]["writes"]["patch"]["pos"] == "tap"
        assert raw["method"]["positions"]["tap"] == {"sweep": list(soak.POSITIONS)}
        document = soak.DOCUMENT
        assert document.recorded and document.exact == ()
        assert document.geometries == soak.GEOMETRIES == ("ep=2", "tp=2")
        assert document.classes == par.DOCUMENTS["inference"].classes
        assert "soak" not in par.DOCUMENTS  # the parity replay never runs it
        assert par.TRACES == "memory_trace"


# --------------------------------------------------------------------------- #
# the harness over gloo: the soak on the tiny MoE at ep=2
# --------------------------------------------------------------------------- #


@pytest.mark.smoke
class TestSoakHarness:
    def test_every_rank_traces_one_line_per_point_and_the_rule_passes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        monkeypatch.setenv("OMP_NUM_THREADS", "1")
        monkeypatch.delenv(recorder.MEMORY_TRACE_VARIABLE, raising=False)
        monkeypatch.delenv(recorder.GRADIENTS_VARIABLE, raising=False)
        realization = par.Realization(TINY_QWEN35_MOE, "fp32", "cpu", 3)
        outputs = soak.run_geometries(tmp_path, realization, ("ep=2",))
        out = outputs["ep=2"]
        receipt = par.receipt(out)
        assert receipt["execution"]["parallel"]["launcher"] == "spawned"
        assert len(receipt["points"]) == 8
        traces = par.traces(out)
        assert len(traces) == par.WORLD
        for rank, lines in enumerate(traces):
            assert [line.point for line in lines] == list(range(8)), rank
            assert {line.rank for line in lines} == {rank}
            assert len({line.point_digest for line in lines}) == 8, rank
            assert all(
                line.device is None and line.sample() == (0, 0) for line in lines
            )
        assert soak.problems([rule.samples_of(lines) for lines in traces]) == []
        # the gradient recorder wrote nothing: an inference document trains nothing
        assert not recorder.gradients_path(out / par.GRADIENTS, 0).exists()
        assert os.environ.get(recorder.MEMORY_TRACE_VARIABLE) is None
