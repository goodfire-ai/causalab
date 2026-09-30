"""The loader under a dtype conversion (``weights.py`` "Converting";
``docs/model_parallelism.md`` §5.3 "the load's peak under a dtype
conversion"): a tensor stored in another dtype than the model holds it in
is read onto the host in batches no larger than the largest converting
tensor, cast there, and copied to the device already in its dtype — never
landed on the device as stored.

``unit``: [`target_dtypes`][causalab.neural.engines.pytorch_hooks.weights.target_dtypes] is transformers' own per-key rule (the
model's dtype; a keep-in-fp32 plan's entry where its pattern matches);
[`conversions`][causalab.neural.engines.pytorch_hooks.weights.conversions] names exactly the keys stored otherwise; a
[`Prefetch`][causalab.neural.engines.pytorch_hooks.weights.Prefetch] plans the converting keys of a group as batches within
its budget that span the files, reads them on the host and lands them on
the device (``meta`` stands in) in the target dtype while the straight keys
stay one whole-group read on the device; a same-dtype plan has no batch and
the reader is asked once, as before; the residency rule's device clauses
name a conversion when the report shows one and read as before when it
does not.

``property``: over drawn key tables — files, shapes, stored dtypes — and
both reader kinds, every tensor leaves exactly once in its target dtype
with its value intact, every converting job is read on the host within the
budget, every straight job on the device.

And the loads: the tiny Llama (fp32 on disk) held in bf16 and the tiny MoE
(bf16 on disk) held in fp32 — the two directions of gemma-2-9b's case —
through a `MeteredReader` over the
real planned reader are bit-identical to the stock loader's, and the
on-disk copies alive at once never exceed the largest converting tensor.
"""

from __future__ import annotations

import dataclasses
import threading
from pathlib import Path
from typing import Any, Sequence

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from causalab.io.fastersafetensors._dtypes import (  # pyright: ignore[reportPrivateUsage]
    torch_dtype,
)
from causalab.neural.engines.pytorch_hooks.checkpoint import (
    TensorHeader,
    checkpoint_files,
    read_header,
)
from causalab.neural.engines.pytorch_hooks.residency import (
    RESERVED_SLACK,
    Residency,
    conversion_note,
    residency_problems,
)
from causalab.neural.engines.pytorch_hooks.weights import (
    FastersafetensorsReader,
    Prefetch,
    ReadGroup,
    Shard,
    conversions,
    load_pretrained,
    renamed_keys,
    target_dtypes,
    wanted_keys,
)

from tests._helpers.metered_reader import MeteredReader
from tests.neural.engines.pytorch_hooks.conftest import TINY_LLAMA, TINY_QWEN35_MOE

# pyright: reportPrivateUsage=false

CPU = torch.device("cpu")
META = torch.device("meta")
_SETTINGS = settings(max_examples=30, deadline=None)


def _meta(key: str, dtype: torch.dtype) -> Any:
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(key)
    with torch.device("meta"):
        return AutoModelForCausalLM.from_config(
            config, dtype=dtype, attn_implementation="eager"
        )


def _headers(key: str) -> dict[str, TensorHeader]:
    files = checkpoint_files(key, "main")
    assert files is not None
    out: dict[str, TensorHeader] = {}
    for path in files:
        out.update(read_header(path))
    return out


# --------------------------------------------------------------------------- #
# which keys convert, and to what
# --------------------------------------------------------------------------- #


@pytest.mark.unit
class TestTargetDtypes:
    def test_every_float_parameter_lands_in_the_models_dtype(self) -> None:
        meta = _meta(TINY_LLAMA, torch.bfloat16)
        headers = _headers(TINY_LLAMA)
        targets = renamed_keys(meta, headers)
        assert set(targets) == wanted_keys(meta, headers)
        wanted = target_dtypes(meta, targets, torch.bfloat16)
        assert set(wanted.values()) == {torch.bfloat16}
        # the fixture is stored in fp32: every key converts
        assert conversions(meta, targets, headers, torch.bfloat16) == wanted
        # held in fp32 (a meta model built as the loader builds it, in the
        # load's dtype) nothing does
        in_fp32 = _meta(TINY_LLAMA, torch.float32)
        assert conversions(in_fp32, targets, headers, torch.float32) == {}
        # transformers' rule, mirrored: the meta parameter's own dtype wins
        # over a load dtype it disagrees with, so a bf16 instance asked for
        # fp32 still lands every key in bf16
        assert set(target_dtypes(meta, targets, torch.float32).values()) == {
            torch.bfloat16
        }

    def test_a_keep_in_fp32_plan_wins_where_its_pattern_matches(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """transformers' dtype plan (``_keep_in_fp32_modules_strict`` under
        a half precision) casts the matching parameters to fp32 whatever
        the model's dtype; the rule here reads the same plan, so those keys
        are not conversions of an fp32 checkpoint while the rest are."""
        meta = _meta(TINY_LLAMA, torch.bfloat16)
        # transformers copies the class flag onto the instance at init, and
        # ``_get_dtype_plan`` reads the instance's
        monkeypatch.setattr(meta, "_keep_in_fp32_modules_strict", ["norm"])
        assert meta._get_dtype_plan(torch.bfloat16) == {"norm": torch.float32}
        headers = _headers(TINY_LLAMA)
        targets = renamed_keys(meta, headers)
        wanted = target_dtypes(meta, targets, torch.bfloat16)
        norms = {key for key, parameter in targets.items() if "norm" in parameter}
        assert norms and all(wanted[key] == torch.float32 for key in norms)
        assert all(wanted[key] == torch.bfloat16 for key in wanted if key not in norms)
        cast = conversions(meta, targets, headers, torch.bfloat16)
        assert set(cast) == set(wanted) - norms

    def test_the_mixed_fixture_converts_its_bf16_tensors_alone_under_fp32(self) -> None:
        """The tiny MoE stores 2028 tensors in bf16 and six in fp32; held in
        fp32 the bf16 ones convert and the fp32 ones take the straight path."""
        meta = _meta(TINY_QWEN35_MOE, torch.float32)
        headers = _headers(TINY_QWEN35_MOE)
        targets = renamed_keys(meta, headers)
        cast = conversions(meta, targets, headers, torch.float32)
        assert cast and set(cast.values()) == {torch.float32}
        assert all(headers[key].dtype == "BF16" for key in cast)
        straight = set(targets) - set(cast)
        assert straight and all(headers[key].dtype == "F32" for key in straight)


# --------------------------------------------------------------------------- #
# the batches a prefetch plans and runs
# --------------------------------------------------------------------------- #


@dataclasses.dataclass
class _RecordingCheckpointReader:
    """A ``CheckpointReader`` with no disk: every tensor is filled with its
    key's index, in its header's dtype, on the device asked for; every
    call is recorded with the device it was asked to land on."""

    table: dict[str, int]
    calls: list[tuple[torch.device, tuple[Shard, ...]]] = dataclasses.field(
        default_factory=list
    )
    _lock: threading.Lock = dataclasses.field(default_factory=threading.Lock)

    def read_all(
        self, shards: Sequence[Shard], device: torch.device
    ) -> dict[str, torch.Tensor]:
        with self._lock:
            self.calls.append((device, tuple(shards)))
        return {
            key: _filled(shard.headers[key], self.table[key], device)
            for shard in shards
            for key in shard.keys
        }


@dataclasses.dataclass
class _RecordingShardReader:
    """The ``ShardReader`` twin: one call per shard."""

    table: dict[str, int]
    concurrency: int = 3
    calls: list[tuple[torch.device, Path, tuple[str, ...]]] = dataclasses.field(
        default_factory=list
    )
    _lock: threading.Lock = dataclasses.field(default_factory=threading.Lock)

    def read(
        self,
        path: Path,
        keys: Sequence[str],
        device: torch.device,
        select: Any = None,
    ) -> dict[str, torch.Tensor]:
        with self._lock:
            self.calls.append((device, path, tuple(keys)))
        headers = self._headers[path]
        return {key: _filled(headers[key], self.table[key], device) for key in keys}

    _headers: dict[Path, dict[str, TensorHeader]] = dataclasses.field(
        default_factory=dict
    )


def _filled(header: TensorHeader, value: int, device: torch.device) -> torch.Tensor:
    dtype = torch_dtype(header.dtype)
    if device.type == "meta":
        return torch.empty(header.shape, dtype=dtype, device=device)
    return torch.full(header.shape, float(value), dtype=dtype)


def _nbytes(header: TensorHeader) -> int:
    return header.elements * header.itemsize


#: Two files, the tensors stored in fp32 but for one bf16 and one int64.
_TABLES: list[tuple[Path, dict[str, TensorHeader]]] = [
    (
        Path("a.safetensors"),
        {
            "embed": TensorHeader("F32", (8, 4)),
            "l0.q": TensorHeader("F32", (4, 4)),
            "l0.norm": TensorHeader("F32", (4,)),
            "l0.bias": TensorHeader("BF16", (4,)),
        },
    ),
    (
        Path("b.safetensors"),
        {
            "l1.q": TensorHeader("F32", (4, 4)),
            "l1.norm": TensorHeader("F32", (4,)),
            "steps": TensorHeader("I64", (2,)),
            "head": TensorHeader("F32", (8, 4)),
        },
    ),
]


def _shards(tables=_TABLES) -> tuple[Shard, ...]:
    return tuple(
        Shard(path=path, keys=tuple(headers), headers=headers)
        for path, headers in tables
    )


def _cast_for(tables, dtype: torch.dtype) -> dict[str, torch.dtype]:
    """The converting keys as [`conversions`][causalab.neural.engines.pytorch_hooks.weights.conversions] would name them: every
    float tensor stored in another dtype (an int tensor keeps its own)."""
    return {
        key: dtype
        for _, headers in tables
        for key, header in headers.items()
        if torch_dtype(header.dtype).is_floating_point
        and torch_dtype(header.dtype) != dtype
    }


@pytest.mark.unit
class TestPrefetchBatches:
    def test_converting_keys_are_batched_within_the_budget_across_the_files(
        self,
    ) -> None:
        shards = _shards()
        cast = _cast_for(_TABLES, torch.bfloat16)
        assert set(cast) == {"embed", "l0.q", "l0.norm", "l1.q", "l1.norm", "head"}
        reader = _RecordingCheckpointReader(
            {k: i for i, k in enumerate(sorted(cast) + ["l0.bias", "steps"])}
        )
        prefetch = Prefetch([ReadGroup(META, shards)], reader, cast=cast)
        # the budget is the largest converting tensor: the 8 × 4 fp32 embedding
        assert prefetch.budget == 8 * 4 * 4
        straight = [job for job in prefetch.jobs if not job.converting]
        converting = [job for job in prefetch.jobs if job.converting]
        # one whole-group read on the device for the keys stored as held
        assert len(straight) == 1
        assert straight[0].read_device == META and straight[0].device == META
        assert set(straight[0].keys) == {"l0.bias", "steps"}
        # every converting job reads on the host, lands on the device, fits
        # the budget, and together they cover every converting key once
        assert converting and all(
            job.read_device == CPU and job.device == META for job in converting
        )
        assert all(job.disk_bytes <= prefetch.budget for job in converting)
        seen = [key for job in converting for key in job.keys]
        assert sorted(seen) == sorted(cast)
        # the batches span the files: some job reads from both shards
        assert any(len(job.shards) == 2 for job in converting)
        # the embedding fills a batch by itself; the norms travel together
        alone = [job for job in converting if job.keys == ("embed",)]
        assert len(alone) == 1

    def test_converting_tensors_leave_on_the_device_in_the_target_dtype(self) -> None:
        shards = _shards()
        cast = _cast_for(_TABLES, torch.bfloat16)
        keys = [key for _, headers in _TABLES for key in headers]
        reader = _RecordingCheckpointReader({k: i for i, k in enumerate(keys)})
        with Prefetch([ReadGroup(META, shards)], reader, cast=cast) as prefetch:
            out = {key: prefetch.take(key) for key in keys}
        for key, tensor in out.items():
            assert tensor.device == META, key
            header = (
                dict(_TABLES)[Path("a.safetensors")].get(key)
                or dict(_TABLES)[Path("b.safetensors")][key]
            )
            expected = cast.get(key, torch_dtype(header.dtype))
            assert tensor.dtype == expected, key
            assert tuple(tensor.shape) == header.shape
        # the reader was asked for the host exactly for the converting jobs
        devices = [device for device, _ in reader.calls]
        assert devices.count(META) == 1
        assert devices.count(CPU) == len([j for j in prefetch.jobs if j.converting])

    def test_a_same_dtype_plan_has_no_batch_and_asks_the_reader_once(self) -> None:
        shards = _shards()
        keys = [key for _, headers in _TABLES for key in headers]
        reader = _RecordingCheckpointReader({k: i for i, k in enumerate(keys)})
        prefetch = Prefetch([ReadGroup(CPU, shards)], reader)
        assert prefetch.budget == 0
        assert len(prefetch.jobs) == 1 and not prefetch.jobs[0].converting
        assert prefetch.jobs[0].shards == shards
        with prefetch:
            for key in keys:
                prefetch.take(key)
        assert len(reader.calls) == 1

    def test_a_shard_reader_reads_straight_shards_one_by_one_and_batches_the_rest(
        self,
    ) -> None:
        shards = _shards()
        cast = _cast_for(_TABLES, torch.bfloat16)
        keys = [key for _, headers in _TABLES for key in headers]
        reader = _RecordingShardReader({k: i for i, k in enumerate(keys)})
        reader._headers = {path: headers for path, headers in _TABLES}
        with Prefetch([ReadGroup(CPU, shards)], reader, cast=cast) as prefetch:
            straight = [job for job in prefetch.jobs if not job.converting]
            assert [tuple(job.keys) for job in straight] == [("l0.bias",), ("steps",)]
            out = {key: prefetch.take(key) for key in keys}
        assert out["embed"].dtype == torch.bfloat16 and out["embed"][0, 0].item() == 0.0
        assert out["l0.bias"].dtype == torch.bfloat16  # stored so, not cast
        assert out["steps"].dtype == torch.int64
        # a converting job of two shards is two reader calls on the host
        host_calls = [call for call in reader.calls if call[0] == CPU]
        assert len(host_calls) == len(
            reader.calls
        )  # the target device is the host here
        assert {path for _, path, _ in reader.calls} == {
            Path("a.safetensors"),
            Path("b.safetensors"),
        }


# --------------------------------------------------------------------------- #
# property: over drawn tables and both reader kinds
# --------------------------------------------------------------------------- #

_DTYPES = st.sampled_from(("F32", "BF16", "F16"))
_TARGETS = st.sampled_from((torch.float32, torch.bfloat16, torch.float16))


@st.composite
def _drawn_tables(draw: st.DrawFn) -> list[tuple[Path, dict[str, TensorHeader]]]:
    files = draw(st.integers(min_value=1, max_value=3))
    tables: list[tuple[Path, dict[str, TensorHeader]]] = []
    n = 0
    for f in range(files):
        count = draw(st.integers(min_value=1, max_value=5))
        headers: dict[str, TensorHeader] = {}
        for _ in range(count):
            shape = tuple(draw(st.lists(st.integers(1, 4), min_size=1, max_size=2)))
            headers[f"t{n}"] = TensorHeader(draw(_DTYPES), shape)
            n += 1
        tables.append((Path(f"f{f}.safetensors"), headers))
    return tables


@pytest.mark.property
class TestPrefetchProperties:
    @_SETTINGS
    @given(tables=_drawn_tables(), target=_TARGETS, whole=st.booleans())
    def test_every_tensor_leaves_once_in_its_dtype_within_the_budget(
        self, tables, target, whole
    ) -> None:
        shards = _shards(tables)
        keys = [key for _, headers in tables for key in headers]
        cast = _cast_for(tables, target)
        values = {key: i + 1 for i, key in enumerate(keys)}
        reader: Any
        if whole:
            reader = _RecordingCheckpointReader(values)
        else:
            reader = _RecordingShardReader(values)
            reader._headers = {path: headers for path, headers in tables}
        headers_of = {
            key: header for _, headers in tables for key, header in headers.items()
        }
        with Prefetch([ReadGroup(CPU, shards)], reader, cast=cast) as prefetch:
            budget = max((_nbytes(headers_of[k]) for k in cast), default=0)
            assert prefetch.budget == budget
            for job in prefetch.jobs:
                if job.converting:
                    assert job.read_device == CPU and job.disk_bytes <= budget
                    assert set(job.cast) == set(job.keys) <= set(cast)
                else:
                    assert not set(job.keys) & set(cast)
            handed = [key for job in prefetch.jobs for key in job.keys]
            assert sorted(handed) == sorted(keys)
            for key in keys:
                tensor = prefetch.take(key)
                expected = cast.get(key, torch_dtype(headers_of[key].dtype))
                assert tensor.dtype == expected
                assert tuple(tensor.shape) == headers_of[key].shape
                # small integers survive any of the three float dtypes exactly
                assert torch.equal(
                    tensor.float(),
                    torch.full(headers_of[key].shape, float(values[key])),
                )
            for key in keys:
                with pytest.raises(KeyError):
                    prefetch.take(key)


# --------------------------------------------------------------------------- #
# the residency rule names the conversion
# --------------------------------------------------------------------------- #


@pytest.mark.unit
class TestResidencyNote:
    def _record(self, on_disk: str | None) -> dict[str, Any]:
        module = torch.nn.Module()
        module.w = torch.nn.Parameter(torch.zeros(4, dtype=torch.bfloat16))
        residency = Residency.of(module, CPU)
        assert residency.dtype_resident == {"w": "bfloat16"}
        record = residency.record()
        record["elements_requested"] = dict(residency.elements_resident)
        if on_disk is not None:
            record["dtype_on_disk"] = {"w": on_disk}
        total = residency.bytes_total
        record["device_bytes_allocated"] = total
        record["device_bytes_reserved"] = total + RESERVED_SLACK + 1
        return record

    def test_a_reserved_excess_of_a_converting_load_names_the_conversion(self) -> None:
        record = self._record("F32")
        assert conversion_note(record) == (
            "; 1 parameter read as fp32, held as bf16 — a conversion's staging "
            "belongs on the host, never in the device's pool"
        )
        (problem,) = residency_problems(record)
        assert problem.startswith("the allocator reserves")
        assert problem.endswith(
            "read as fp32, held as bf16 — a conversion's staging belongs on the host, never in the device's pool"
        )

    def test_a_same_dtype_load_reads_as_before(self) -> None:
        for on_disk in ("BF16", None):
            record = self._record(on_disk)
            assert conversion_note(record) == ""
            (problem,) = residency_problems(record)
            assert problem.endswith(": a segment the weights did not reuse")

    def test_several_parameters_and_directions_are_counted(self) -> None:
        """Each direction counted and worded once, in word order; a
        parameter the residency does not list says nothing; an integer
        tensor is never a conversion (transformers keeps its dtype), nor is
        one stored in the dtype it is held in."""
        record = {
            "dtype_on_disk": {
                "a": "F32",
                "b": "F32",
                "c": "BF16",
                "d": "BF16/F32",
                "e": "I64",
                "f": "BF16",
            },
            "dtype_resident": {
                "a": "bfloat16",
                "b": "bfloat16",
                "c": "float32",
                "e": "int64",
                "f": "bfloat16",
            },
        }
        assert conversion_note(record) == (
            "; 1 parameter read as bf16, held as fp32, 2 parameters read as fp32, "
            "held as bf16 — a conversion's staging belongs on the host, never in the "
            "device's pool"
        )


# --------------------------------------------------------------------------- #
# the loads: bit-identical to stock, the on-disk copies bounded
# --------------------------------------------------------------------------- #


def _stock(key: str, dtype: torch.dtype) -> Any:
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(
        key, dtype=dtype, attn_implementation="eager"
    )


def _largest_converting(key: str, dtype: torch.dtype) -> int:
    meta = _meta(key, dtype)
    headers = _headers(key)
    cast = conversions(meta, renamed_keys(meta, headers), headers, dtype)
    return max(_nbytes(headers[k]) for k in cast)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("key", "dtype", "stored"),
    [(TINY_LLAMA, torch.bfloat16, "F32"), (TINY_QWEN35_MOE, torch.float32, "BF16")],
    ids=["llama-fp32-on-disk-held-bf16", "moe-bf16-on-disk-held-fp32"],
)
def test_a_converting_load_is_stock_bit_for_bit_and_stages_one_batch_at_a_time(
    key: str, dtype: torch.dtype, stored: str
) -> None:
    metered = MeteredReader(FastersafetensorsReader(), target=dtype)
    model = load_pretrained(
        key,
        "main",
        dtype=dtype,
        device="cpu",
        attn_implementation="eager",
        reader=metered,
    )
    stock = _stock(key, dtype)
    ours, theirs = model.state_dict(), stock.state_dict()
    assert set(ours) == set(theirs)
    for name in ours:
        assert ours[name].dtype == theirs[name].dtype, name
        assert torch.equal(ours[name], theirs[name]), name
    # every converting tensor went through the meter in its stored dtype,
    # and the copies alive at once never exceeded the largest of them
    budget = _largest_converting(key, dtype)
    assert metered.handed > 0 and metered.peak <= budget, (metered.peak, budget)
    assert metered.live == 0
    assert len(metered.calls) > 1  # batches, not one read
    headers = _headers(key)
    for _, shards in metered.calls:
        for _, keys in shards:
            assert all(headers[k].dtype == stored for k in keys) or all(
                headers[k].dtype != stored for k in keys
            ), "a call mixed converting and straight keys"
    # the dtype the meter weighed against is the one the model holds
    assert {p.dtype for p in model.parameters()} <= {dtype, torch.float32}
