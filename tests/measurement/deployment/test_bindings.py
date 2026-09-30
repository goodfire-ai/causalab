"""Deployment accepts only the targets declared by the measurement mode."""

from copy import deepcopy

import pytest

from causalab.measurement.deployment.bindings import (
    BindingError,
    authored_bindings,
    target_bindings,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("mode", ["single", "comparison"])
@pytest.mark.parametrize("kind", ["repository", "source", "installation"])
def test_target_bindings_roundtrip_without_mutating_authored_paths(mode, kind):
    targets = {"source"} if mode == "single" else {"before", "after", "eager"}
    plan = {"mode": mode, "arms": dict.fromkeys(targets)}
    bindings = {
        "device": "cpu",
        "data_root": "data",
        "artifacts_root": "artifacts",
    }
    target = {kind: "relative-path", "python": "venv/bin/python"}
    if mode == "single":
        bindings["source"] = target
    else:
        bindings["arms"] = {name: dict(target) for name in targets}
    original = deepcopy(bindings)
    normalized = target_bindings(bindings, plan)
    assert set(normalized["arms"]) == targets
    assert authored_bindings(normalized, plan) == original
    for target in normalized["arms"].values():
        target["python"] = "different"
    assert bindings == original


@pytest.mark.parametrize(
    "damage",
    [
        "comparison_shape",
        "both_shapes",
        "missing_source",
        "empty_source",
        "non_object",
        "unknown",
        "non_path",
        "empty_path",
        "bad_device",
    ],
)
def test_single_invalid_bindings_are_structured(damage):
    plan = {"mode": "single", "arms": {"source": {}}}
    bindings = {
        "source": {"repository": "repo", "python": "venv/bin/python"},
        "device": "cpu",
        "data_root": "data",
        "artifacts_root": "artifacts",
    }
    match damage:
        case "comparison_shape":
            bindings["arms"] = {"source": bindings.pop("source")}
        case "both_shapes":
            bindings["arms"] = {"source": bindings["source"]}
        case "missing_source":
            bindings.pop("source")
        case "empty_source":
            bindings["source"] = {}
        case "non_object":
            bindings = []
        case "unknown":
            bindings["source"]["extra"] = "unexpected"
        case "non_path":
            bindings["source"]["python"] = None
        case "empty_path":
            bindings["source"]["repository"] = ""
        case "bad_device":
            bindings["device"] = 42
    with pytest.raises(BindingError) as caught:
        target_bindings(bindings, plan)
    assert caught.value.field
    assert caught.value.reason
