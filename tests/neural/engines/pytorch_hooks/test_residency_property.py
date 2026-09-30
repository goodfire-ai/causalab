"""Properties of the residency census (``residency.py``): for any small
model whose parameters are their own storages the census reads back exactly
the parameters, the rule over its own record is silent, and re-backing any
one parameter by a ``k``-times larger allocation (the shape of a ``1 / k``
view of a whole read) is named for that parameter alone; and for any set
of storages alive before the load and any set appearing during it, the
census counts exactly the latter, each once, however many views of them
are live."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from causalab.neural.engines.pytorch_hooks import residency
from causalab.neural.engines.pytorch_hooks.residency import (
    Census,
    Residency,
    residency_problems,
)

# pyright: reportPrivateUsage=false

pytestmark = pytest.mark.property

_DEVICES = st.tuples(
    st.sampled_from(("cpu", "cuda", "meta")), st.sampled_from((None, 0, 1))
)


def _device(kind: str, index: int | None) -> torch.device:
    return torch.device(kind) if index is None else torch.device(kind, index)


@settings(max_examples=40, deadline=None)
@given(held=_DEVICES, wanted=_DEVICES)
def test_a_tensor_is_on_the_device_when_the_types_agree_and_no_index_disagrees(
    held: tuple[str, int | None], wanted: tuple[str, int | None]
) -> None:
    """``cuda`` against ``cuda:1`` is the device (an unindexed device names
    the current one), ``cuda:0`` against ``cuda:1`` is not, another type
    never is. Devices are compared, not tensors: the tensor is a stand-in
    carrying one, since a CPU process holds no CUDA tensor to ask."""
    tensor = SimpleNamespace(device=_device(*held))
    expected = held[0] == wanted[0] and (
        held[1] is None or wanted[1] is None or held[1] == wanted[1]
    )
    assert residency._same_device(tensor, _device(*wanted)) is expected  # type: ignore[arg-type]


CPU = torch.device("cpu")
_SETTINGS = settings(max_examples=40, deadline=None)

_shapes = st.lists(
    st.tuples(st.integers(1, 6), st.integers(1, 6)), min_size=1, max_size=4
)
_dtypes = st.sampled_from([torch.float32, torch.float16, torch.bfloat16, torch.int64])


def _model(shapes: list[tuple[int, int]], dtype: torch.dtype) -> torch.nn.Module:
    module = torch.nn.Module()
    for i, (rows, cols) in enumerate(shapes):
        tensor = torch.zeros(rows, cols, dtype=dtype)
        module.register_parameter(
            f"w{i}", torch.nn.Parameter(tensor, requires_grad=dtype.is_floating_point)
        )
    return module


def _record(residency: Residency) -> dict:
    record = residency.record()
    record["elements_requested"] = dict(residency.elements_resident)
    return record


@_SETTINGS
@given(shapes=_shapes, dtype=_dtypes)
def test_own_storages_read_back_as_the_parameters_and_the_rule_is_silent(
    shapes: list[tuple[int, int]], dtype: torch.dtype
) -> None:
    module = _model(shapes, dtype)
    residency = Residency.of(module, CPU)
    itemsize = torch.empty((), dtype=dtype).element_size()
    for i, (rows, cols) in enumerate(shapes):
        assert residency.elements_resident[f"w{i}"] == rows * cols
        assert residency.bytes_resident[f"w{i}"] == rows * cols * itemsize
    assert residency.shared == ()
    assert residency.bytes_total == sum(r * c for r, c in shapes) * itemsize
    assert residency_problems(_record(residency)) == []


@_SETTINGS
@given(shapes=_shapes, dtype=_dtypes, k=st.integers(2, 8), data=st.data())
def test_a_parameter_backed_k_times_larger_is_named_alone(
    shapes: list[tuple[int, int]], dtype: torch.dtype, k: int, data: st.DataObject
) -> None:
    module = _model(shapes, dtype)
    which = data.draw(st.integers(0, len(shapes) - 1))
    rows, cols = shapes[which]
    whole = torch.zeros(k * rows * cols, dtype=dtype)
    module.register_parameter(
        f"w{which}",
        torch.nn.Parameter(
            whole[: rows * cols].view(rows, cols),
            requires_grad=dtype.is_floating_point,
        ),
    )
    residency = Residency.of(module, CPU)
    problems = residency_problems(_record(residency))
    assert len(problems) == 1
    assert problems[0].startswith(f"w{which}: backed by a storage of")
    assert "a view of a larger allocation" in problems[0]


_sizes = st.lists(st.integers(1, 64), min_size=0, max_size=6)


@_SETTINGS
@given(before=_sizes, during=_sizes, views=st.integers(0, 3))
def test_the_census_counts_what_appeared_since_the_baseline_once_each(
    before: list[int], during: list[int], views: int
) -> None:
    """Over the ``gc`` seam: ``before`` storages alive at the baseline and
    ``during`` appearing after it — plus ``views`` extra views of each —
    the unowned bytes are the ``during`` storages' bytes, each counted once,
    and a baseline that saw nothing counts every storage."""
    module = torch.nn.Module()
    module.register_parameter("w", torch.nn.Parameter(torch.zeros(2, 2)))
    old = [torch.zeros(n) for n in before]
    live: list = [module.w, *old]
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            residency,
            "gc",
            SimpleNamespace(collect=lambda: 0, get_objects=lambda: list(live)),
        )
        baseline = Census.take(CPU)
        assert baseline.storages.keys() == {residency._storage_key(t) for t in live}
        new = [torch.zeros(n) for n in during]
        live.extend(new)
        for tensor in new:
            for _ in range(views):
                live.append(tensor.view(-1))
        measured = Residency.of(module, CPU, since=baseline)
        assert measured.bytes_unowned == sum(n * 4 for n in during)
        everything = Residency.of(module, CPU, since=Census(CPU, {}))
        assert everything.bytes_unowned == sum(n * 4 for n in before + during)
