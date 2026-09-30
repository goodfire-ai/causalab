"""Native identities ignore local paths and bind the exact saved tensors."""

import copy

import pytest

from causalab.analysis.hypothesis_artifacts import frozen_dbm_identity, resolve_position
from causalab.analysis.export_dbm import position

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("value", ["all", {"all": True}, "every_token"])
def test_named_and_literal_all_token_positions_share_the_native_form(value):
    method = {"positions": {"every_token": {"all": True}}}
    assert resolve_position(value, method) == position(value, method) == "all"


def test_dbm_identity_binds_inventory_mask_and_entry():
    gates = [{"id": "heads", "component": "attention_head", "position": "all"}]
    point = {
        "masks": {"heads": [1, 0]},
        "provenance": {
            "fits": {
                "heads": {"sha256": "fit", "entry": "theta", "file_path": "/original"}
            }
        },
    }
    expected = frozen_dbm_identity(gates, point)
    moved = copy.deepcopy(point)
    moved["provenance"]["fits"]["heads"]["file_path"] = "/copied"
    assert frozen_dbm_identity(gates, moved) == expected
    moved["provenance"]["fits"]["heads"]["entry"] = "theta[seed=1]"
    assert frozen_dbm_identity(gates, moved) != expected
    moved = copy.deepcopy(point)
    moved["masks"]["heads"] = [0, 1]
    assert frozen_dbm_identity(gates, moved) != expected
    assert frozen_dbm_identity([{**gates[0], "position": 0}], point) != expected
