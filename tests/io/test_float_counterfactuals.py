"""Counterfactual JSON preserves the float bits observable by model equations."""

import json
import math
import struct

import pytest

from causalab.causal import CausalModel, Dom, V, mechanism
from causalab.io.counterfactuals import (
    _decode_value,
    load_counterfactual_examples,
    save_counterfactual_examples,
)

pytestmark = pytest.mark.unit


def _float(bits):
    return struct.unpack("!d", bytes.fromhex(bits))[0]


def _bits(value):
    return struct.pack("!d", value).hex()


def _roundtrip(tmp_path, model, trace):
    path = tmp_path / "examples.json"
    save_counterfactual_examples(
        [{"input": trace, "counterfactual_inputs": [trace.copy()]}], str(path)
    )
    json.loads(
        path.read_text(),
        parse_constant=lambda value: pytest.fail(f"Nonstandard JSON number: {value}"),
    )
    return load_counterfactual_examples(str(path), model)[0]


def test_saved_nan_retains_its_signed_outcome(tmp_path):
    @mechanism
    def equations(x: Dom(float)):
        raw_input = V(str(x), domain=Dom(str))  # noqa: F841
        raw_output = V(
            "negative" if math.copysign(1.0, x) < 0 else "positive",
            domain=Dom(str),
        )
        return raw_output

    model = CausalModel(equations)
    original = model.new_trace({"x": -float("nan")})
    example = _roundtrip(tmp_path, model, original)
    assert original["raw_output"] == "negative"
    assert example["input"]["raw_output"] == "negative"


@pytest.mark.parametrize("finite_domain", [False, True])
@pytest.mark.parametrize(
    "bits",
    [
        "7ff8000000000001",  # Distinct quiet NaN payloads and signs.
        "7ff8000000000002",
        "fff8000000000001",
        "7ff0000000000001",  # Signaling NaN payloads must also survive.
        "fff0000000000002",
        "7ff0000000000000",  # Positive and negative infinity.
        "fff0000000000000",
        "0000000000000000",  # Positive and negative zero.
        "8000000000000000",
        "0000000000000001",  # Smallest subnormal and largest finite value.
        "7fefffffffffffff",
    ],
)
def test_float_bits_and_outcomes_survive_json(tmp_path, finite_domain, bits):
    value = _float(bits)
    domain = Dom([value]) if finite_domain else Dom(float)

    @mechanism
    def equations(x: domain):
        raw_input = V("float", domain=Dom(str))  # noqa: F841
        raw_output = V(_bits(x), domain=Dom(str))
        return raw_output

    model = CausalModel(equations)
    example = _roundtrip(tmp_path, model, model.new_trace({"x": value}))
    for trace in [example["input"], *example["counterfactual_inputs"]]:
        assert _bits(trace["x"]) == bits
        assert trace["raw_output"] == bits


def test_special_floats_in_nested_values_and_mapping_keys(tmp_path):
    negative_nan = _float("fff8000000000042")
    payload_nan = _float("7ff8000000000043")
    value = (
        {"float64": "ordinary mapping key", "items": [negative_nan, -0.0]},
        {(payload_nan,): [float("inf"), -float("inf"), (payload_nan,)]},
    )

    @mechanism
    def equations(x: Dom([value])):
        raw_input = V("nested floats", domain=Dom(str))  # noqa: F841
        raw_output = V(_bits(x[0]["items"][0]), domain=Dom(str))
        return raw_output

    model = CausalModel(equations)
    trace = _roundtrip(tmp_path, model, model.new_trace({"x": value}))["input"]
    loaded = trace["x"]
    assert type(loaded) is tuple
    assert loaded[0]["float64"] == "ordinary mapping key"
    assert _bits(loaded[0]["items"][0]) == "fff8000000000042"
    assert _bits(loaded[0]["items"][1]) == "8000000000000000"
    key, items = next(iter(loaded[1].items()))
    assert type(key) is tuple
    assert _bits(key[0]) == "7ff8000000000043"
    assert items[:2] == [float("inf"), -float("inf")]
    assert type(items[2]) is tuple
    assert _bits(items[2][0]) == "7ff8000000000043"
    assert trace["raw_output"] == "fff8000000000042"


def test_saved_nan_interventions_remain_active_after_loading(tmp_path):
    replacement = _float("fff8000000000042")

    @mechanism
    def equations(x: Dom(float)):
        middle = V(x, domain=Dom(float))
        raw_input = V("intervention", domain=Dom(str))  # noqa: F841
        raw_output = V(_bits(middle), domain=Dom(str))
        return raw_output

    model = CausalModel(equations)
    original = model.new_trace({"x": 1.0})
    original.intervene_many({"x": replacement, "middle": replacement})
    example = _roundtrip(tmp_path, model, original)
    for trace in [example["input"], *example["counterfactual_inputs"]]:
        del trace["x"]
        assert _bits(trace["x"]) == "fff8000000000042"
        trace["x"] = 2.0
        assert trace["raw_output"] == "fff8000000000042"


@pytest.mark.parametrize(
    "value", [1.25, -0.0, float("inf"), -float("inf"), float("nan")]
)
def test_version_one_native_float_values_still_load(tmp_path, value):
    @mechanism
    def equations(x: Dom(float)):
        raw_input = V("legacy", domain=Dom(str))  # noqa: F841
        raw_output = V(_bits(x), domain=Dom(str))
        return raw_output

    model = CausalModel(equations)
    path = tmp_path / "legacy.json"
    path.write_text(
        json.dumps(
            [
                {
                    "input": {
                        "version": 1,
                        "values": {"x": value},
                        "interventions": [],
                    },
                    "counterfactual_inputs": [],
                }
            ]
        )
    )
    trace = load_counterfactual_examples(str(path), model)[0]["input"]
    assert trace["raw_output"] == _bits(value)


@pytest.mark.parametrize(
    "encoded",
    [
        {"float64": None},
        {"float64": 0},
        {"float64": ["7ff8000000000001"]},
        {"float64": "nan"},
        {"float64": "7ff800000000000"},
        {"float64": "7ff80000000000001"},
        {"float64": "7ff800000000000g"},
        {"float64": "7ff80000000000  "},
        {"float64": "7ff8000000000001", "extra": True},
    ],
)
def test_malformed_float_tags_are_rejected(encoded):
    with pytest.raises(ValueError):
        _decode_value(encoded)
