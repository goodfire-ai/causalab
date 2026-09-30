"""Authoring errors identify the source field that the user can edit."""

import re
from typing import Any

import pytest
from hypothesis import given, strategies as st

from causalab.measurement.spec import MeasurementSpecError, parse_measurement
from tests.measurement.test_single_spec import single_study
from tests.measurement.test_spec import study

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("single", [False, True])
@pytest.mark.parametrize(
    "source,field",
    [
        (None, ""),
        ({}, ""),
        ({"revision": "HEAD", "unknown": True}, ""),
        ({"revision": ""}, ".revision"),
        ({"revision": "-option"}, ".revision"),
        ({"revision": "HEAD", "execution": None}, ".execution"),
        ({"revision": "HEAD", "execution": {"engine": "unknown"}}, ".execution.engine"),
        ({"revision": "HEAD", "execution": {"batch_rows": 0}}, ".execution.batch_rows"),
        (
            {"revision": "HEAD", "execution": {"cuda_graphs": "yes"}},
            ".execution.cuda_graphs",
        ),
    ],
)
def test_source_errors_use_authored_path(single: bool, source: Any, field: str):
    raw = single_study() if single else study()
    if single:
        raw["measurement"]["source"] = source
        path = "measurement.source"
    else:
        raw["measurement"]["arms"]["before"] = source
        path = "measurement.arms.before"
    with pytest.raises(MeasurementSpecError, match="^" + re.escape(path + field + ":")):
        parse_measurement(raw["measurement"], raw["steps"])


def test_single_workflow_selection_explains_single_source_constraint():
    raw = single_study()
    raw["measurement"]["source"]["workflow"] = "candidate.yaml"
    with pytest.raises(
        MeasurementSpecError,
        match=r"^measurement\.source\.workflow:.*single-source",
    ):
        parse_measurement(raw["measurement"], raw["steps"])


@given(st.integers(max_value=0))
def test_invalid_batch_size_always_points_to_authored_source(batch_rows: int):
    raw = single_study()
    raw["measurement"]["source"]["execution"] = {"batch_rows": batch_rows}
    with pytest.raises(
        MeasurementSpecError,
        match=r"^measurement\.source\.execution\.batch_rows:",
    ):
        parse_measurement(raw["measurement"], raw["steps"])
