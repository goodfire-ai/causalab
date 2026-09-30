"""Execution-affecting environment choices participate in resume identity."""

import os
from unittest.mock import patch

from hypothesis import given, strategies as st
import pytest
import torch

from causalab.measurement.collection import execution_identity

pytestmark = pytest.mark.property

SWITCHES = (
    ("CAUSALAB_MOE_GLUE", "off", "all"),
    ("CAUSALAB_FUSED_NORMS", "off", "all"),
    ("CAUSALAB_GDN_SHORT_SEQ", "0", "16"),
    ("CAUSALAB_PROJECT_HEAD_UNDER_GRAD", "0", "1"),
    ("CAUSALAB_COMPILE_CACHE", "/tmp/cache-a", "/tmp/cache-b"),
)


@given(switch=st.sampled_from(SWITCHES), reverse=st.booleans())
def test_runtime_switch_changes_are_attested_without_recording_secrets(switch, reverse):
    name, first, second = switch
    if reverse:
        first, second = second, first
    with patch.dict(os.environ, {"CAUSALAB_PRIVATE_TOKEN": "not-for-the-receipt"}):
        os.environ.pop(name, None)
        absent = execution_identity(torch.device("cpu"))["environment"][
            "runtime_variables"
        ]
        assert name in absent and absent[name] is None
        os.environ[name] = first
        before = execution_identity(torch.device("cpu"))["environment"][
            "runtime_variables"
        ]
        os.environ[name] = second
        after = execution_identity(torch.device("cpu"))["environment"][
            "runtime_variables"
        ]
    assert before[name] == first and after[name] == second
    assert absent != before != after
    assert all(
        "CAUSALAB_PRIVATE_TOKEN" not in record for record in (absent, before, after)
    )
