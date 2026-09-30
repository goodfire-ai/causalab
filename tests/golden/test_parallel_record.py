"""CPU guard for the parallel golden's band rule, record and comparison
(``docs/model_parallelism.md`` §10.6; runs in default CI, loads no model).

The band rule (``tests/golden/_parallel/bands.py``) is held as hypothesis
properties — monotone in both measured inputs, never below the floor, always
holding the measured maximum, an injected fivefold regression outside
whenever the maximum is the binding term, ``ulp`` agreeing with torch's own
spacing — with the twenty-fold rule and a rule without the resolution term
as their hand-written mutations. The record ``tests/golden/parallel_goldens.json``
is either pending (format 2, no documents) or a capture whose every entry
obeys the rule and replays against itself; a record of an earlier format,
or an entry missing a field the rule needs, is refused **by name**
(``StaleRecord``), never as a ``KeyError``. The comparison, measurement and
loader rules the GPU replay applies are held on hand-built inputs.
"""

from __future__ import annotations

import dataclasses
import json
import math
import platform
from pathlib import Path
from typing import Any

import pytest
import torch
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from causalab.neural.engines.pytorch_hooks.residency import RESERVED_SLACK
from tests.golden import _parallel as par
from tests.golden._parallel import bands, fit, measure
from tests.neural.engines.pytorch_hooks import (
    test_tensor_expert_parallel_run as boundary,
)
from tests.neural.engines.pytorch_hooks import test_train_parallel_run as smoke

_SETTINGS = settings(
    deadline=None,
    max_examples=30,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)

finite = st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False)
dtypes = st.sampled_from(sorted(bands.PRECISION))


# --------------------------------------------------------------------------- #
# the band rule: properties
# --------------------------------------------------------------------------- #


class TestBandRule:
    pytestmark = pytest.mark.property

    @_SETTINGS
    @given(a=finite, b=finite, scale=finite, dtype=dtypes)
    def test_monotone_in_the_measured_maximum(
        self, a: float, b: float, scale: float, dtype: str
    ) -> None:
        low, high = sorted((a, b))
        assert bands.band_for(low, scale, dtype) <= bands.band_for(high, scale, dtype)

    @_SETTINGS
    @given(worst=finite, a=finite, b=finite, dtype=dtypes)
    def test_monotone_in_the_scale(
        self, worst: float, a: float, b: float, dtype: str
    ) -> None:
        low, high = sorted((a, b))
        assert bands.band_for(worst, low, dtype) <= bands.band_for(worst, high, dtype)
        assert bands.ulp(dtype, low) <= bands.ulp(dtype, high)

    @_SETTINGS
    @given(worst=finite, scale=finite, dtype=dtypes)
    def test_never_below_the_floor_and_always_holds_the_maximum(
        self, worst: float, scale: float, dtype: str
    ) -> None:
        band = bands.band_for(worst, scale, dtype)
        assert band >= bands.FLOOR
        assert band >= worst

    @_SETTINGS
    @given(worst=finite, scale=finite, dtype=dtypes)
    def test_a_fivefold_regression_falls_outside_when_the_maximum_binds(
        self, worst: float, scale: float, dtype: str
    ) -> None:
        resolution = bands.ULPS * bands.ulp(dtype, scale)
        assume(worst > 0.0 and bands.FACTOR * worst >= max(resolution, bands.FLOOR))
        assert bands.band_for(worst, scale, dtype) < 5 * worst

    @_SETTINGS
    @given(scale=st.floats(min_value=1e-30, max_value=1e30, allow_nan=False))
    def test_ulp_is_the_spacing_torch_shows_in_bf16_and_fp32(
        self, scale: float
    ) -> None:
        for dtype, torch_dtype, bits in (
            ("bf16", torch.bfloat16, torch.int16),
            ("fp32", torch.float32, torch.int32),
        ):
            x = torch.tensor(scale, dtype=torch_dtype)
            above = (x.view(bits) + 1).view(torch_dtype)
            assert bands.ulp(dtype, float(x)) == float(above.double() - x.double())

    @_SETTINGS
    @given(a=finite, b=finite, s=finite, t=finite, d=dtypes, e=dtypes)
    def test_a_join_of_two_measurements_has_the_larger_band(
        self, a: float, b: float, s: float, t: float, d: str, e: str
    ) -> None:
        x, y = bands.Measurement(a, s, d), bands.Measurement(b, t, e)
        joined = x.join(y)
        assert joined.band == max(x.band, y.band) == y.join(x).band
        assert joined.max_abs_diff == max(a, b)

    @_SETTINGS
    @given(fraction=st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
    def test_the_routing_band_is_twice_the_fraction_between_its_floor_and_one(
        self, fraction: float
    ) -> None:
        band = bands.routing_band(fraction)
        assert bands.ROUTING_FLOOR <= band <= 1.0
        assert band >= fraction


class TestBandRuleExamples:
    pytestmark = pytest.mark.unit

    def test_the_a3b_maxima_get_bands_of_a_few_units_not_twenty_times(self) -> None:
        """A bf16 sharded-write difference of 1.22 at logit scale 40."""
        assert bands.ulp("bf16", 40.0) == 0.25
        assert bands.band_for(1.21875, 40.0, "bf16") == pytest.approx(3.65625)
        assert bands.band_for(0.0, 40.0, "bf16") == 0.5  # two ulps, not the floor
        assert bands.band_for(0.0, 0.0, "bf16") == bands.FLOOR
        assert bands.band_for(2.4e-5, 1.0, "fp32") == bands.FLOOR  # the floor binds
        assert bands.band_for(2.4e-3, 1.0, "fp32") == pytest.approx(7.2e-3)

    def test_an_unknown_dtype_is_refused_by_name(self) -> None:
        with pytest.raises(bands.UnknownDtype, match="int64"):
            bands.ulp("int64", 1.0)
        with pytest.raises(bands.UnknownDtype):
            bands.dtype_name(torch.int64)
        assert bands.dtype_name(torch.bfloat16) == "bf16"
        assert bands.dtype_name(torch.float32) == "fp32"

    def test_mutation_the_twenty_fold_rule_lets_a_fivefold_regression_through(
        self,
    ) -> None:
        """The rule this one replaces: twenty times the maximum. On the A3B's
        sharded_write maximum it passes a regression five times the size."""

        def twenty(worst: float, scale: float, dtype: str) -> float:
            return max(20 * worst, bands.FLOOR)

        worst, scale = 1.21875, 40.0
        assert 5 * worst <= twenty(worst, scale, "bf16")  # the mutation passes it
        assert 5 * worst > bands.band_for(worst, scale, "bf16")  # the rule refuses it

    def test_mutation_dropping_the_resolution_term_pins_a_zero_class_to_the_floor(
        self,
    ) -> None:
        """Without the ulp term a class that measured zero at logit scale
        would be held to 1e-3 — below what bf16 can express there, so any
        recapture's rounding would fail it."""

        def unresolved(worst: float, scale: float, dtype: str) -> float:
            return max(bands.FACTOR * worst, bands.FLOOR)

        one_ulp = bands.ulp("bf16", 40.0)
        assert one_ulp > unresolved(0.0, 40.0, "bf16")
        assert one_ulp <= bands.band_for(0.0, 40.0, "bf16")

    def test_the_routing_examples(self) -> None:
        assert bands.routing_band(0.0) == bands.ROUTING_FLOOR
        assert bands.routing_band(4 / 112) == pytest.approx(8 / 112)
        assert bands.routing_band(0.7) == 1.0


# --------------------------------------------------------------------------- #
# the record
# --------------------------------------------------------------------------- #


def _measured(
    worst: float, scale: float = 40.0, dtype: str = "bf16"
) -> bands.Measurement:
    return bands.Measurement(worst, scale, dtype)


def _banded(**classes: bands.Measurement | float) -> measure.Measured:
    measured = measure.Measured()
    for kind, value in classes.items():
        if isinstance(value, bands.Measurement):
            measured.add(kind, f"{kind}.safetensors", value)
        else:
            measured.classes[kind] = value
    return measured


def _record(
    banded: dict[str, measure.Measured],
    exact: dict[str, float],
    name: str = "inference",
) -> dict[str, Any]:
    document = par.DOCUMENTS[name]
    block = par.document_record(document, par.A3B, banded, exact, with_context=False)
    return par.make_record(None, par.A3B, {name: block})


class TestRecord:
    pytestmark = pytest.mark.unit

    def test_the_record_exists_is_this_format_and_names_the_capture_command(
        self,
    ) -> None:
        record = par.load_record()
        par.check_format(record)
        assert record["recapture"] == par.CAPTURE_COMMAND
        assert record["model"] == par.MODEL and record["dtype"] == par.A3B.dtype
        assert record["tolerance"] == par.tolerance()
        assert str(bands.FACTOR) in record["tolerance"]["rule"]
        assert "ulp" in record["tolerance"]["rule"]

    def test_a_pending_record_is_captured_for_no_document(self) -> None:
        pending = par.pending_record(par.A3B)
        assert not any(par.captured(pending, name) for name in par.DOCUMENTS)
        assert par.render(pending).endswith("\n")
        with pytest.raises(par.StaleRecord, match="no capture of the 'das'"):
            par.compare_records(pending, pending, "das")

    @pytest.mark.parametrize("name", sorted(par.DOCUMENTS))
    def test_a_captured_document_obeys_the_rule_and_replays_against_itself(
        self, name: str
    ) -> None:
        record = par.load_record()
        if not par.captured(record, name):
            pytest.skip(f"pending: no capture of {name!r} to hold to the rule yet")
        document = par.DOCUMENTS[name]
        block = record["documents"][name]
        held_exact = [g for g in document.exact if g not in block["inexact"]]
        assert set(block["exact"]) == set(held_exact)
        assert all(worst == 0.0 for worst in block["exact"].values())
        assert set(block["geometries"]) == set(document.banded) | set(block["inexact"])
        for classes in block["geometries"].values():
            assert set(classes) == set(document.classes)
            for kind, entry in classes.items():
                assert entry["band"] == par.entry_band(kind, entry)
                if kind != par.ROUTING:
                    assert entry["files"], kind
        assert par.compare_records(record, record, name) == []
        assert {"torch", "transformers", "causalab", "node"} <= set(block["context"])
        # the node is described by its hardware, never named by its host
        assert block["context"]["node"] in block["context"]["cuda"]
        assert set(block["load"]) == set(document.geometries)
        if document.recorded:
            assert set(block["memory"]) == set(document.geometries) | {"solo"}

    def test_an_injected_fivefold_regression_of_a_binding_class_is_refused(
        self,
    ) -> None:
        """On the committed capture, whichever document carries one."""
        record = par.load_record()
        checked = 0
        for name in par.DOCUMENTS:
            if not par.captured(record, name):
                continue
            for geometry, classes in record["documents"][name]["geometries"].items():
                for kind, entry in classes.items():
                    value = par.entry_value(kind, entry)
                    binding = (
                        kind != par.ROUTING
                        and value > 0.0
                        and bands.FACTOR * value >= max(entry["band"] / 1.0000001, 0.0)
                    )
                    if not binding:
                        continue
                    fresh = json.loads(json.dumps(record))
                    fresh["documents"][name]["geometries"][geometry][kind][
                        "max_abs_diff"
                    ] = 5 * value
                    problems = par.compare_records(record, fresh, name)
                    assert any(f"{geometry} {kind}" in p for p in problems), (
                        name,
                        kind,
                    )
                    checked += 1
        if checked == 0:
            pytest.skip("pending: no binding class captured yet")

    def test_a_format_one_record_is_refused_by_name_never_as_a_key_error(
        self,
    ) -> None:
        """The record as it was written before the rule carried scales."""
        old = {
            "model": par.MODEL,
            "dtype": "bf16",
            "device": "cuda",
            "document": {"layer": 3, "sweep": [3, 7]},
            "geometries": {
                "tp=2": {"stream": {"max_abs_diff": 0.634765625, "band": 12.6953125}}
            },
            "exact": {"dp=2": 0.0, "pp=2": 0.0},
            "tolerance": {"rule": "band = max(20 * max_abs_diff, 0.001)"},
        }
        fresh = _record({"tp=2": _banded(stream=_measured(0.5))}, {})
        with pytest.raises(
            par.StaleRecord, match="format None; the band rule needs format 2"
        ):
            par.compare_records(old, fresh, "inference")
        assert not par.captured(old, "inference")

    def test_an_entry_without_its_scale_or_with_a_disobedient_band_is_refused(
        self,
    ) -> None:
        with pytest.raises(par.StaleRecord, match="lacks \\['scale'\\]"):
            par.entry_band(
                "stream", {"max_abs_diff": 0.5, "dtype": "bf16", "band": 1.5}
            )
        with pytest.raises(
            par.StaleRecord, match="records band 20.0 but the rule yields"
        ):
            par.entry_band(
                "stream",
                {"max_abs_diff": 0.5, "scale": 40.0, "dtype": "bf16", "band": 20.0},
            )
        with pytest.raises(par.StaleRecord, match="knows no dtype 'int8'"):
            par.entry_band(
                "stream",
                {"max_abs_diff": 0.5, "scale": 40.0, "dtype": "int8", "band": 1.5},
            )
        with pytest.raises(par.StaleRecord, match="lacks 'fraction'"):
            par.entry_value(par.ROUTING, {"max_abs_diff": 0.03})
        committed = _record({"tp=2": _banded(stream=_measured(0.5))}, {})
        del committed["documents"]["inference"]["geometries"]["tp=2"]["stream"]["scale"]
        with pytest.raises(par.StaleRecord):
            par.compare_records(committed, committed, "inference")

    def test_a_fresh_value_outside_the_band_is_a_problem_naming_it(self) -> None:
        committed = _record(
            {"tp=2": _banded(stream=_measured(0.5), routing=4 / 112)},
            {"pp=2": 0.0, "dp=2": 0.0},
        )
        entry = committed["documents"]["inference"]["geometries"]["tp=2"]["stream"]
        assert entry["band"] == bands.band_for(0.5, 40.0, "bf16") == 1.5
        inside = _record({"tp=2": _banded(stream=_measured(1.4), routing=5 / 112)}, {})
        assert par.compare_records(committed, inside, "inference") == []
        outside = _record({"tp=2": _banded(stream=_measured(1.6), routing=5 / 112)}, {})
        problems = par.compare_records(committed, outside, "inference")
        assert len(problems) == 1 and "inference tp=2 stream" in problems[0]
        assert "scale 40.0 bf16" in problems[0]
        flipped = _record({"tp=2": _banded(stream=_measured(0.5), routing=0.5)}, {})
        assert any(
            "routing" in p for p in par.compare_records(committed, flipped, "inference")
        )
        not_exact = _record(
            {"tp=2": _banded(stream=_measured(0.5), routing=0.0)}, {"pp=2": 1e-9}
        )
        assert any(
            "must be exact" in p
            for p in par.compare_records(committed, not_exact, "inference")
        )
        other_model = {**inside, "model": "somebody/else"}
        assert any(
            p.startswith("model")
            for p in par.compare_records(committed, other_model, "inference")
        )
        missing = _record({"tp=2": _banded(stream=_measured(0.5))}, {})
        assert any(
            "classes" in p for p in par.compare_records(committed, missing, "inference")
        )

    def test_a_capture_of_one_document_keeps_the_others_of_this_format(self) -> None:
        first = _record(
            {"tp=2": _banded(stream=_measured(0.5), routing=0.0)}, {"pp=2": 0.0}
        )
        das = par.document_record(
            par.DOCUMENTS["das"],
            par.A3B,
            {"tp=2": _banded(bundle=_measured(1e-3, 1.0, "fp32"))},
            {"pp=2": 0.0},
            with_context=False,
        )
        merged = par.make_record(first, par.A3B, {"das": das})
        assert set(merged["documents"]) == {"inference", "das"}
        assert par.captured(merged, "das") and par.captured(merged, "inference")
        # an earlier format's documents are not carried into the new record
        stale = {**first, "format": 1}
        assert set(par.make_record(stale, par.A3B, {"das": das})["documents"]) == {
            "das"
        }

    def test_the_record_carries_per_file_detail_beside_each_class(self) -> None:
        measured = measure.Measured()
        measured.add("stream", "logits.safetensors", _measured(0.5, 40.0))
        measured.add("stream", "r_attn.safetensors", _measured(0.1, 2.0))
        block = par.document_record(
            par.DOCUMENTS["inference"],
            par.A3B,
            {"tp=2": measured},
            {},
            with_context=False,
        )
        entry = block["geometries"]["tp=2"]["stream"]
        assert entry["max_abs_diff"] == 0.5 and entry["scale"] == 40.0
        assert set(entry["files"]) == {"logits.safetensors", "r_attn.safetensors"}
        assert entry["files"]["r_attn.safetensors"] == {
            "max_abs_diff": 0.1,
            "scale": 2.0,
            "dtype": "bf16",
        }

    def test_the_documents_and_their_geometries(self) -> None:
        assert set(par.DOCUMENTS) == {"inference", "das", "dbm", "das_dense"}
        assert par.DOCUMENTS["inference"].geometries == ("dp=2", "pp=2", "tp=2", "ep=2")
        assert par.DOCUMENTS["das"].exact == ("pp=2",)
        assert set(par.DOCUMENTS["das"].banded) == {"tp=2", "dp=2:rows"}
        assert par.DOCUMENTS["dbm"].banded == ("ep=2",)
        assert par.DOCUMENTS["das"].recorded and par.DOCUMENTS["dbm"].recorded
        assert not par.DOCUMENTS["inference"].recorded
        assert par.DOCUMENTS["das"].argv == ("--fit-rows", str(fit.FIT_ROWS))
        assert par.GRADIENTS_VARIABLE == smoke.GRADIENTS_VARIABLE
        assert par.out_name("dp=2:rows") == "dp2-rows"

    def test_the_dense_fit_is_the_das_fit_on_its_own_fp32_realization(self) -> None:
        """§10.6: the fit pin tight to the fp32 ulp — the DAS document on a
        dense model in fp32, ``tp=2`` and ``dp=2:rows`` banded (no exact
        geometry: the dense model ties its head, which ``pp`` refuses by
        name), the realization the document's own and written into its
        block."""
        dense = par.DOCUMENTS["das_dense"]
        assert dense.exact == () and dense.banded == ("tp=2", "dp=2:rows")
        assert dense.recorded and dense.author is fit.das_document
        assert dense.realization == par.DENSE
        assert par.DENSE.dtype == "fp32" and par.DENSE.device == "cuda"
        assert par.DENSE.model == par.DENSE_MODEL == "Qwen/Qwen3-4B-Instruct-2507"
        assert par.realization_of(dense) == par.DENSE
        assert par.realization_of(par.DOCUMENTS["das"]) == par.A3B
        for name in ("inference", "das", "dbm"):
            assert par.DOCUMENTS[name].realization is None
        # the registry knows the dense model as a served dense family
        from causalab.protocol.registry import get_model_info

        info = get_model_info(par.DENSE_MODEL)
        assert info.family == "qwen3" and info.num_layers > par.DENSE.layer

    def test_a_documents_own_realization_is_in_its_block_and_compared(self) -> None:
        dense = par.DOCUMENTS["das_dense"]
        measured = _banded(**{k: _measured(1e-4, 1.0, "fp32") for k in fit.CLASSES})
        block = par.document_record(
            dense, par.DENSE, {"tp=2": measured}, {}, with_context=False
        )
        assert block["realization"] == {
            "model": par.DENSE_MODEL,
            "dtype": "fp32",
            "device": "cuda",
        }
        record = par.make_record(None, par.A3B, {"das_dense": block})
        assert record["model"] == par.MODEL  # the record's realization stays the A3B's
        assert par.compare_records(record, record, "das_dense") == []
        moved = json.loads(par.render(record))
        moved["documents"]["das_dense"]["realization"]["dtype"] = "bf16"
        problems = par.compare_records(record, moved, "das_dense")
        assert any("das_dense: realization" in p for p in problems), problems
        plain = par.document_record(
            par.DOCUMENTS["das"], par.A3B, {}, {"pp=2": 0.0}, with_context=False
        )
        assert "realization" not in plain

    def test_the_context_describes_the_nodes_hardware_not_its_host(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
        monkeypatch.setattr(torch.cuda, "get_device_name", lambda i: "H100 80GB")
        gpu = par.context()
        assert gpu["node"] == "H100 80GB" and gpu["cuda"] == ["H100 80GB"] * 2
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
        assert par.context()["node"] == platform.machine()


# --------------------------------------------------------------------------- #
# the measurements, on hand-built inputs
# --------------------------------------------------------------------------- #


class TestMeasure:
    pytestmark = pytest.mark.unit

    def test_a_table_measures_numeric_cells_and_refuses_a_label_that_differs(
        self, tmp_path: Path
    ) -> None:
        solo, other = tmp_path / "solo", tmp_path / "other"
        solo.mkdir(), other.mkdir()
        rows = [{"metric": "iia", "value": 10.0, "passes": 5, "nested": {"x": 1.5}}]
        (solo / "t.json").write_text(json.dumps(rows))
        (other / "t.json").write_text(
            json.dumps(
                [{"metric": "iia", "value": 10.5, "passes": 5, "nested": {"x": 1.0}}]
            )
        )
        got = measure.table(solo, other, "t.json", "bf16")
        assert got == bands.Measurement(0.5, 10.0, "bf16")
        (other / "t.json").write_text(
            json.dumps(
                [{"metric": "ce", "value": 10.0, "passes": 5, "nested": {"x": 1.5}}]
            )
        )
        assert measure.table(solo, other, "t.json", "bf16").max_abs_diff == math.inf
        (other / "t.json").write_text(json.dumps(rows + rows))
        assert measure.table(solo, other, "t.json", "bf16").max_abs_diff == math.inf

    def test_a_tensor_file_measures_floats_with_their_dtype_and_scale(
        self, tmp_path: Path
    ) -> None:
        from causalab.io.tensor_files import save_file

        solo, other = tmp_path / "solo", tmp_path / "other"
        solo.mkdir(), other.mkdir()
        a = {
            "x": torch.tensor([1.0, -8.0], dtype=torch.bfloat16),
            "i": torch.tensor([1, 2]),
        }
        b = {
            "x": torch.tensor([1.5, -8.0], dtype=torch.bfloat16),
            "i": torch.tensor([1, 2]),
        }
        save_file(a, str(solo / "x.safetensors"))
        save_file(b, str(other / "x.safetensors"))
        assert measure.tensor_file(solo, other, "x.safetensors") == bands.Measurement(
            0.5, 8.0, "bf16"
        )
        b["i"] = torch.tensor([1, 3])
        save_file(b, str(other / "x.safetensors"))
        assert (
            measure.tensor_file(solo, other, "x.safetensors").max_abs_diff == math.inf
        )

    def test_gradients_are_relative_and_a_stage_without_the_featurizer_is_skipped(
        self,
    ) -> None:
        g = torch.tensor([2.0, -4.0])
        solo = [[g], [g * 2]]
        owner = [[g + 0.004], [g * 2 + 0.008]]
        other = [[], []]  # the pipeline stage that owns no featurizer
        against, across = measure.gradient_measurements(solo, [owner, other])
        assert against.max_abs_diff == pytest.approx(0.001, rel=1e-3)
        assert (
            across.max_abs_diff == 0.0
            and against.dtype == "fp32"
            and against.scale == 1.0
        )
        _, across = measure.gradient_measurements(solo, [owner, [[g], [g * 2]]])
        assert across.max_abs_diff == pytest.approx(0.001, rel=1e-3)
        against, _ = measure.gradient_measurements(solo, [[[g]]])
        assert against.max_abs_diff == math.inf
        against, _ = measure.gradient_measurements(solo, [other, other])
        assert against.max_abs_diff == math.inf

    def test_measured_joins_classes_over_files_and_knows_its_worst(self) -> None:
        measured = measure.Measured()
        measured.classes[par.ROUTING] = 0.02
        measured.add("stream", "a", _measured(0.1, 1.0))
        measured.add("stream", "b", _measured(0.3, 40.0))
        assert measured.classes["stream"] == _measured(0.3, 40.0)
        assert measured.worst() == 0.3
        assert measure.Measured().worst() == 0.0


# --------------------------------------------------------------------------- #
# the loader rule and the receipt block
# --------------------------------------------------------------------------- #


def _report(
    rank: int, world: int, requested: dict[str, int], on_disk: dict[str, int]
) -> dict[str, Any]:
    return {
        "rank": rank,
        "world": world,
        "bytes_requested": requested,
        "bytes_on_disk": on_disk,
        **_resident(requested),
    }


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


class TestLoadRule:
    pytestmark = pytest.mark.unit
    ON_DISK = {"q": 800, "experts": 1600, "norm": 40}

    def test_a_correct_tensor_parallel_pair_passes(self) -> None:
        requested = {"q": 400, "experts": 800, "norm": 40}
        reports = [_report(r, 2, requested, self.ON_DISK) for r in range(2)]
        assert par.load_problems("tp=2", reports) == []

    def test_a_rank_that_reserved_twice_what_it_holds_is_named(self) -> None:
        """Reject a second reserved segment as large as the model weights."""
        requested = {"q": 400, "experts": 800, "norm": 40}
        reports = [_report(r, 2, requested, self.ON_DISK) for r in range(2)]
        held = reports[1]["bytes_total"]
        reports[1]["device_bytes_allocated"] = held
        reports[1]["device_bytes_reserved"] = 2 * held + RESERVED_SLACK
        (problem,) = par.load_problems("ep=2", reports)
        assert problem.startswith("rank 1: the allocator reserves")

    def test_a_third_of_a_parameter_is_named(self) -> None:
        wrong = {"q": 266, "experts": 800, "norm": 40}
        reports = [
            _report(0, 2, wrong, self.ON_DISK),
            _report(1, 2, {"q": 400, "experts": 800, "norm": 40}, self.ON_DISK),
        ]
        problems = par.load_problems("tp=2", reports)
        assert any("q requested 266" in p for p in problems)

    def test_nothing_sharded_under_tensor_parallelism_is_a_problem(self) -> None:
        reports = [_report(r, 2, dict(self.ON_DISK), self.ON_DISK) for r in range(2)]
        assert any(
            "no parameter is sharded" in p for p in par.load_problems("tp=2", reports)
        )

    def test_pipeline_stages_read_whole_and_disjoint(self) -> None:
        first = {"q": 800, "norm": 40}
        last = {"experts": 1600}
        reports = [_report(0, 2, first, first), _report(1, 2, last, last)]
        assert par.load_problems("pp=2", reports) == []
        overlapping = [
            _report(0, 2, first, first),
            _report(1, 2, {**last, "q": 800}, {**last, "q": 800}),
        ]
        assert any(
            "same parameter" in p for p in par.load_problems("pp=2", overlapping)
        )
        sharded = [
            _report(0, 2, {"q": 400, "norm": 40}, first),
            _report(1, 2, last, last),
        ]
        assert any(
            "shards a parameter" in p for p in par.load_problems("pp=2", sharded)
        )

    def test_a_replica_reads_the_whole_model_over_points_and_over_rows(self) -> None:
        reports = [_report(r, 2, dict(self.ON_DISK), self.ON_DISK) for r in range(2)]
        assert par.load_problems("dp=2", reports) == []
        assert par.load_problems("dp=2:rows", reports) == []
        assert par.load_totals(reports)["rank1"] == {
            "bytes_requested": 2440,
            "bytes_on_disk": 2440,
        }

    def test_the_parallel_block_is_the_smoke_tiers(self) -> None:
        assert par.parallel_block("tp=2") == {
            **boundary._block(tensor=2),  # pyright: ignore[reportPrivateUsage]
            "launcher": "spawned",
        }
        assert par.parallel_block("dp=2")["data"] == 2
        assert par.parallel_block("pp=2")["pipeline"] == 2
        rows = par.parallel_block("dp=2:rows")
        assert rows["data"] == 2 and rows["data_mode"] == "rows" and rows["world"] == 2

    def test_the_inference_classes_are_the_smoke_tiers(self) -> None:
        assert set(boundary.CLASSES.values()) == {
            "stream",
            "sharded_read",
            "sharded_write",
            "experts",
        }
        assert par.ROUTING not in boundary.CLASSES.values()  # measured as a fraction


class TestRun:
    pytestmark = pytest.mark.unit

    def test_a_failed_run_leaves_no_receipt_for_a_kept_root_to_resume(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A publishing rank may write the receipt before another rank dies;
        the resume rule trusts the receipt, so a failure removes it."""
        import subprocess

        out = tmp_path / "out"
        out.mkdir()
        (out / par.RECEIPT).write_text("{}")
        document = tmp_path / "doc.json"
        document.write_text("{}")

        class Completed:
            returncode = 1
            stderr = "rank 1 of 2 raised"

        monkeypatch.setattr(subprocess, "run", lambda *a, **k: Completed())
        # the receipt exists, so run() would resume: stage a failure instead
        (out / par.RECEIPT).unlink()
        with pytest.raises(par.RunFailed, match="rank 1 of 2"):
            par.run(document, out, realization=par.A3B)
        assert not (out / par.RECEIPT).exists()
        (out / par.RECEIPT).write_text("{}")
        assert par.run(document, out, realization=par.A3B) == out  # resumes

    @pytest.mark.parametrize("own", [None, par.DENSE])
    def test_run_all_authors_a_document_on_its_own_realization(
        self, tmp_path: Path, own: par.Realization | None
    ) -> None:
        """``run_all`` given no realization runs a document on the one it
        names, else the record's — the replay must not author ``das_dense``
        on the A3B (``run_all``'s docstring). Every run's receipt is staged
        by the author, so nothing is launched and the author alone sees the
        realization."""
        seen: list[par.Realization] = []

        def author(base: Path, realization: par.Realization) -> Path:
            seen.append(realization)
            for name in ("solo", *map(par.out_name, ("pp=2", "tp=2"))):
                (base / name).mkdir(exist_ok=True)
                (base / name / par.RECEIPT).write_text("{}")
            return base / "doc.json"

        document = dataclasses.replace(
            fit.DAS, author=author, exact=("pp=2",), banded=("tp=2",), realization=own
        )
        outputs = par.run_all(tmp_path, document)
        assert set(outputs) == {"solo", "pp=2", "tp=2"}
        assert seen == [par.realization_of(document)]
        assert seen == [own if own is not None else par.A3B]
