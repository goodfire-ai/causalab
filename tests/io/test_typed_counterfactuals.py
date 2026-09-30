"""Saved traces preserve supported numeric types and shared value contents."""

import copy
import json
import struct

import numpy as np
import pytest

from causalab.causal import CausalModel, Dom, V, mechanism
from causalab.causal.model import CausalTrace
from causalab.io._counterfactual_values import decode_values, encode_values
from causalab.io.counterfactuals import (
    load_counterfactual_examples,
    save_counterfactual_examples,
)

pytestmark = pytest.mark.unit


def _type_name(value):
    return type(value).__name__


def _membership(value):
    return str(value[0] in [value[1]])


def _model(domain, outcome=_type_name):
    @mechanism
    def equations(x: domain):
        raw_input = V("typed values", domain=Dom(str))  # noqa: F841
        raw_output = V(outcome(x), domain=Dom(str))
        return raw_output

    return CausalModel(equations)


def _roundtrip(tmp_path, model, trace):
    path = tmp_path / "values.json"
    save_counterfactual_examples(
        [{"input": trace, "counterfactual_inputs": [trace.copy()]}], str(path)
    )
    saved = json.loads(
        path.read_text(),
        parse_constant=lambda token: pytest.fail(f"Nonstandard JSON token: {token}"),
    )
    assert saved[0]["input"]["version"] == 2
    return load_counterfactual_examples(str(path), model)[0]


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("scalar", [float, np.float32, np.float64, complex])
def test_shared_and_distinct_nan_membership_survives_json(tmp_path, shared, scalar):
    first = scalar("nan")
    second = first if shared else scalar("nan")
    model = _model(Dom(tuple), _membership)
    original = model.new_trace({"x": (first, second)})
    assert original["raw_output"] == str(shared)
    example = _roundtrip(tmp_path, model, original)
    for trace in [example["input"], *example["counterfactual_inputs"]]:
        assert trace["raw_output"] == str(shared)
        assert (trace["x"][0] is trace["x"][1]) is shared
        assert type(trace["x"][0]) is scalar


def test_nan_sharing_across_nested_containers_and_mapping_keys():
    shared = struct.unpack("!d", bytes.fromhex("fff8000000000042"))[0]
    distinct = struct.unpack("!d", bytes.fromhex("fff8000000000042"))[0]
    value = ({shared: [shared, distinct]}, {"ref": shared, "numpy": distinct})
    decoded = decode_values(json.loads(json.dumps(encode_values({"x": value}))))["x"]
    key, items = next(iter(decoded[0].items()))
    assert key is items[0] is decoded[1]["ref"]
    assert items[1] is decoded[1]["numpy"]
    assert items[0] is not items[1]
    assert struct.pack("!d", key).hex() == "fff8000000000042"


def test_shared_values_survive_reordered_json_object_members():
    nan = float("nan")
    encoded = encode_values({"z": nan, "a": nan})
    restored = decode_values(json.loads(json.dumps(encoded, sort_keys=True)))
    assert restored["z"] is restored["a"]


def test_shared_arrays_and_containers_preserve_membership(tmp_path):
    array = np.array([1, 2])
    model = _model(Dom(tuple), _membership)
    original = model.new_trace({"x": (array, array)})
    trace = _roundtrip(tmp_path, model, original)["input"]
    assert trace["raw_output"] == "True"
    assert trace["x"][0] is trace["x"][1]
    shared = [array]
    decoded = decode_values(encode_values({"x": [shared, shared]}))["x"]
    assert decoded[0] is decoded[1]


@pytest.mark.parametrize(
    "value",
    [
        np.bool_(True),
        np.int8(-1),
        np.uint8(255),
        np.int16(-2),
        np.uint16(65535),
        np.int32(-3),
        np.uint32(2**32 - 1),
        np.int64(-4),
        np.uint64(2**64 - 1),
        np.longlong(-5),
        np.ulonglong(2**64 - 1),
        np.float16(-0.0),
        np.float32(-float("inf")),
        np.float64(1.0),
        np.frombuffer(bytes.fromhex("0100c07f"), dtype="<f4")[0],
        np.frombuffer(bytes.fromhex("420000000000f8ff"), dtype="<f8")[0],
        np.complex64(complex(-0.0, 2)),
        np.complex128(complex(1, float("nan"))),
    ],
)
def test_numpy_scalar_types_and_bits_survive_exact_domains(tmp_path, value):
    model = _model(Dom([value]))
    original = model.new_trace({"x": value})
    restored = _roundtrip(tmp_path, model, original)["input"]
    assert type(restored["x"]) is type(value)
    assert restored["x"].dtype.type is value.dtype.type
    assert restored["x"].tobytes() == value.tobytes()
    assert restored["raw_output"] == original["raw_output"]


def test_numpy_float_subclass_of_python_float_stays_numpy(tmp_path):
    model = _model(Dom(float))
    trace = _roundtrip(tmp_path, model, model.new_trace({"x": np.float64(1.0)}))[
        "input"
    ]
    assert trace["raw_output"] == "float64"
    assert type(trace["x"]) is np.float64


@pytest.mark.parametrize(
    "value",
    [
        np.array([1, 2], dtype=np.int64),
        np.array([1, 2], dtype=np.longlong),
        np.array([1, 2], dtype=np.ulonglong),
        np.array([[1, 2], [3, 4]], dtype="int16", order="F"),
        np.array([1, 2], dtype="int32")[::-1],
        np.empty((0, 2), dtype="float32"),
        np.array(3, dtype="uint32"),
        np.array([True, False]),
        np.frombuffer(bytes.fromhex("fff8000000000042"), dtype=">f8"),
        np.array([complex(-0.0, 2)], dtype=">c16"),
    ],
)
def test_numpy_array_dtype_shape_and_bytes_survive_json(tmp_path, value):
    model = _model(Dom([value]))
    trace = _roundtrip(tmp_path, model, model.new_trace({"x": value}))["input"]
    assert type(trace["x"]) is np.ndarray
    assert trace["x"].dtype.type is value.dtype.type
    assert trace["x"].dtype.str == value.dtype.str
    assert trace["x"].shape == value.shape
    assert trace["x"].tobytes(order="C") == value.tobytes(order="C")


def test_shared_numpy_intervention_remains_active(tmp_path):
    @mechanism
    def equations(x: Dom(tuple)):
        middle = V(x, domain=Dom(tuple))
        raw_input = V("override", domain=Dom(str))  # noqa: F841
        raw_output = V(_membership(middle), domain=Dom(str))
        return raw_output

    model = CausalModel(equations)
    trace = model.new_trace({"x": (1, 2)})
    nan = np.float64("nan")
    trace["middle"] = (nan, nan)
    del trace["middle"]
    example = _roundtrip(tmp_path, model, trace)
    for restored in [example["input"], *example["counterfactual_inputs"]]:
        restored["x"] = (3, 4)
        assert restored["raw_output"] == "True"
        assert type(restored["middle"][0]) is np.float64
        assert restored["middle"][0] is restored["middle"][1]


def test_version_one_tags_still_load_without_inventing_shared_nan_identity(tmp_path):
    model = _model(Dom(tuple), _membership)
    path = tmp_path / "v1.json"
    nan = {"float64": "fff8000000000042"}
    path.write_text(
        json.dumps(
            [
                {
                    "input": {
                        "version": 1,
                        "values": {"x": {"tuple": [nan, nan]}},
                        "interventions": [],
                    },
                    "counterfactual_inputs": [],
                }
            ]
        )
    )
    trace = load_counterfactual_examples(str(path), model)[0]["input"]
    assert trace["raw_output"] == "False"
    assert struct.pack("!d", trace["x"][0]).hex() == "fff8000000000042"


def test_legacy_tuple_restoration_retains_dictionary_order(tmp_path):
    choices = [{"a": (1,), "b": 2}, {"b": 2, "a": (1,)}]

    @mechanism
    def equations(x: Dom(choices)):
        raw_input = V(str(x), domain=Dom(str))  # noqa: F841
        raw_output = V(list(x)[0], domain=Dom(str))
        return raw_output

    model = CausalModel(equations)
    original = model.new_trace({"x": choices[1]})
    path = tmp_path / "legacy.json"
    path.write_text(
        json.dumps([{"input": original.to_dict(), "counterfactual_inputs": []}])
    )
    trace = load_counterfactual_examples(str(path), model)[0]["input"]
    assert list(trace["x"]) == ["b", "a"]
    assert type(trace["x"]["a"]) is tuple
    assert trace["raw_output"] == original["raw_output"] == "b"


def test_legacy_values_already_in_the_domain_keep_existing_interpretation(tmp_path):
    model = _model(Dom([[1], (1,)]))
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps([{"input": {"x": [1]}, "counterfactual_inputs": []}]))
    assert (
        load_counterfactual_examples(str(path), model)[0]["input"]["raw_output"]
        == "list"
    )


@pytest.mark.parametrize(
    "value",
    [
        np.array([1], dtype=np.dtype("int64", metadata={"note": [1]})),
        np.array([object()], dtype=object),
        np.array([(1,)], dtype=[("field", "int32")]),
        np.datetime64("2020-01-01"),
        np.longdouble(1),
    ],
)
def test_unsupported_numpy_values_are_rejected_before_writing(tmp_path, value):
    trace = CausalTrace.from_values({"x": value})
    path = tmp_path / "unsupported.json"
    with pytest.raises(ValueError, match="supported numeric dtype without metadata"):
        save_counterfactual_examples(
            [{"input": trace, "counterfactual_inputs": []}], str(path)
        )
    assert not path.exists()


def test_cycles_are_rejected_without_losing_supported_dag_sharing(tmp_path):
    value = []
    value.append(value)
    trace = CausalTrace.from_values({"x": value})
    with pytest.raises(ValueError, match="cyclic"):
        save_counterfactual_examples(
            [{"input": trace, "counterfactual_inputs": []}],
            str(tmp_path / "cycle.json"),
        )


@pytest.mark.parametrize(
    "value",
    [
        {"ref": 0},
        {"ref": True},
        {"ref": -1},
        {"id": 0, "value": {"ref": 0}},
        [{"id": 0, "value": []}, {"id": 0, "value": []}],
        {"id": 0, "value": [], "extra": 1},
        {"tuple": None},
        {"mapping": [[1]]},
        {"mapping": [["x", 1], ["x", 2]]},
        {"mapping": [[[1], 2]]},
        {"complex128": "00"},
    ],
)
def test_malformed_value_tags_are_rejected(value):
    with pytest.raises(ValueError):
        decode_values({"x": value})


@pytest.mark.parametrize(
    "changes",
    [
        {"scalar_type": "eval"},
        {"scalar_type": "float32"},
        {"dtype": "int64"},
        {"dtype": "<f8"},
        {"shape": [-1]},
        {"shape": [True]},
        {"shape": [2]},
        {"shape": "1"},
        {"kind": "scalar"},
        {"data": "g" * 16},
        {"extra": True},
    ],
)
def test_malformed_numpy_tags_are_rejected(changes):
    value = encode_values({"x": np.array([1], dtype=np.int64)})["x"]
    value = copy.deepcopy(value)
    value["numpy"].update(changes)
    with pytest.raises(ValueError):
        decode_values({"x": value})
