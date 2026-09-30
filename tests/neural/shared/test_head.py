"""The vocabulary head where the document reads it (``neural/shared/head.py``).

Three claims. The **decision** is a pure function of the document: an
``lm_head`` read at named positions projects the head itself; a read of the
whole sequence, a continuation read, a read in a group that decodes and a
read in a model that writes at the head keep the head as a tap. The
**numerics**: the head module over the rows a gather selected is the head
over every row, sliced — ``torch.equal``, fp32 and bf16, dense and ragged
position tables, on the three tiny families (a property over random tables).
The **store**: the campaign's tap union names ``ln_final`` for a projecting
read, so a shared pass stores the head's input and not the vocabulary.
"""

from __future__ import annotations

import dataclasses
import functools
from typing import Any

import pytest
import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from causalab.neural.engines.pytorch_hooks.loading import load_model
from causalab.neural.shared.execution import campaign_cache
from causalab.neural.shared.executor import ExecutorBase, RaggedValue, tap_key
from causalab.neural.shared.head import (
    HEAD,
    HEAD_INPUT,
    capture_spec,
    head_module,
    projects_head,
    resolve_read_taps,
    taps_head,
)
from causalab.neural.shared.sites import resolve_site
from causalab.neural.shared.plan import plan_point
from causalab.protocol.schema import (
    PROTOCOL_VERSION,
    Document,
    SiteSpec,
    parse_document,
)

from tests.neural.engines.pytorch_hooks._drive import base_data_section
from tests.neural.engines.pytorch_hooks.conftest import (
    TINY_GPT2,
    TINY_LLAMA,
    TINY_QWEN35_MOE,
)
from tests.protocol._docs import in_order, saved


unit = pytest.mark.unit
prop = pytest.mark.property

FAMILIES = [TINY_LLAMA, TINY_GPT2, TINY_QWEN35_MOE]


@dataclasses.dataclass(frozen=True)
class _Read:
    """A read's address and the model that takes it — the binding a document
    spells on the model (§2.9), not on the read. The un-intervened models go
    by the migrator's names: ``original_base`` / ``original_counterfactual``."""

    address: dict[str, Any]
    model: str = "patched"

    @property
    def input(self) -> str:
        return "counterfactual" if self.model == "original_counterfactual" else "base"


def _raw(**reads: _Read) -> dict[str, Any]:
    """An inference document with a swap at block 0 and the given reads —
    ``v_cf`` (the operand) always present. A model a read names and the
    document does not declare is declared un-intervened on the read's input;
    a test that writes in it sets its ``writes`` afterwards."""
    models: dict[str, dict[str, Any]] = {
        "original_counterfactual": {"input": "counterfactual", "reads": ["v_cf"]},
        "patched": {"input": "base", "reads": [], "writes": ["patch"]},
    }
    for name, read in reads.items():
        entry = models.setdefault(read.model, {"input": read.input, "reads": []})
        entry["reads"].append(name)
    return {
        "header": {"protocol_version": PROTOCOL_VERSION},
        "model": {"key": TINY_LLAMA, "revision": "main"},
        "data": base_data_section(with_counterfactual=True),
        "method": {
            "intervened_models": models,
            "sites": {
                "tgt": {"component": "block_output", "layers": [0]},
                "head": {"component": "lm_head"},
                "norm": {"component": "ln_final"},
            },
            "reads": {
                "v_cf": {"site": "tgt", "pos": {"index": -1}},
                **{name: read.address for name, read in reads.items()},
            },
            "writes": {
                "patch": {"site": "tgt", "pos": {"index": -1}, "do": {"swap": "v_cf"}}
            },
            "save": [
                saved(name, read.model, f"{name}.safetensors")
                for name, read in reads.items()
            ],
        },
    }


def _doc(**reads: _Read) -> Document:
    return parse_document(in_order(_raw(**reads)))


def _head_read(pos: Any, model: str = "patched") -> _Read:
    return _Read({"site": "head", "pos": pos}, model)


class TestTheDecision:
    @unit
    def test_a_read_at_named_positions_projects_the_head(self) -> None:
        doc = _doc(
            last=_head_read({"index": -1}),
            first=_head_read({"index": 0}, model="original_base"),
            second_last=_head_read({"index": -2}, model="original_base"),
        )
        assert projects_head(doc, "patched", "base", "last")
        assert projects_head(doc, "original_base", "base", "first")
        assert projects_head(doc, "original_base", "base", "second_last")
        assert capture_spec(doc, "patched", "base", "last") == SiteSpec(
            component=HEAD_INPUT
        )

    @unit
    def test_a_named_positions_table_projects_too(self) -> None:
        raw = _raw(named=_head_read("late"))
        raw["method"]["positions"] = {"late": {"index": -1}}
        doc = parse_document(in_order(raw))
        assert projects_head(doc, "patched", "base", "named")

    @unit
    def test_a_read_of_the_whole_sequence_keeps_the_head(self) -> None:
        doc = _doc(whole=_head_read("all"), last=_head_read({"index": -1}))
        assert not projects_head(doc, "patched", "base", "whole")
        assert capture_spec(doc, "patched", "base", "whole") == doc.sites["head"]
        # its neighbour in the same group still projects: the decision is per read
        assert projects_head(doc, "patched", "base", "last")

    @unit
    def test_a_read_off_the_head_never_projects(self) -> None:
        doc = _doc(norm=_Read({"site": "norm", "pos": {"index": -1}}))
        assert not projects_head(doc, "patched", "base", "norm")
        assert not projects_head(
            doc, "original_counterfactual", "counterfactual", "v_cf"
        )
        assert capture_spec(doc, "patched", "base", "norm") == doc.sites["norm"]

    @unit
    def test_a_continuation_read_and_its_group_keep_the_head(self) -> None:
        doc = _doc(
            gen=_head_read(
                {"index": 0, "generated": {"max_new_tokens": 2}}, model="original_base"
            ),
            prompt=_head_read({"index": -1}, model="original_base"),
            other=_head_read({"index": -1}),
        )
        assert not projects_head(doc, "original_base", "base", "gen")
        # the prompt-frame read shares the decoding group: its prefill's
        # logits are consumed, so the head runs there anyway
        assert not projects_head(doc, "original_base", "base", "prompt")
        # a group that does not decode is unaffected
        assert projects_head(doc, "patched", "base", "other")

    @unit
    def test_a_write_at_the_head_keeps_the_head_in_that_model_only(self) -> None:
        raw = _raw(
            head_cf=_head_read({"index": -1}, model="original_counterfactual"),
            bumped=_head_read({"index": -1}, model="bumped"),
            last=_head_read({"index": -1}),
        )
        raw["method"]["writes"]["swap_head"] = {
            "site": "head",
            "pos": {"index": -1},
            "do": {"swap": "head_cf"},
        }
        raw["method"]["intervened_models"]["bumped"]["writes"] = ["swap_head"]
        doc = parse_document(in_order(raw))
        assert not projects_head(doc, "bumped", "base", "bumped")
        assert projects_head(doc, "patched", "base", "last")
        assert projects_head(
            doc, "original_counterfactual", "counterfactual", "head_cf"
        )

    @unit
    def test_a_write_below_the_head_does_not_keep_it(self) -> None:
        """A write at ``ln_final`` is seen by the ``ln_final`` tap (writes
        install before captures at one module), so the projection sees the
        written input exactly as the head would."""
        raw = _raw(
            norm_cf=_Read(
                {"site": "norm", "pos": {"index": -1}}, "original_counterfactual"
            ),
            after=_head_read({"index": -1}, model="normed"),
            probe=_head_read({"index": -1}),  # a model nobody reads is refused
        )
        raw["method"]["writes"]["swap_norm"] = {
            "site": "norm",
            "pos": {"index": -1},
            "do": {"swap": "norm_cf"},
        }
        raw["method"]["intervened_models"]["normed"]["writes"] = ["swap_norm"]
        doc = parse_document(in_order(raw))
        assert projects_head(doc, "normed", "base", "after")

    @unit
    def test_a_featurizer_or_dims_on_the_read_changes_nothing(self) -> None:
        raw = _raw(
            sliced=_Read({"site": "head", "pos": {"index": -1}, "dims": [0, 1, 2]})
        )
        doc = parse_document(in_order(raw))
        assert projects_head(doc, "patched", "base", "sliced")


class TestResolvedTaps:
    @unit
    def test_a_projecting_read_captures_ln_final_with_the_head_as_projection(
        self,
    ) -> None:
        bundle = load_model(TINY_LLAMA, device="cpu")
        doc = _doc(last=_head_read({"index": -1}), whole=_head_read("all"))
        reads = [(n, doc.reads[n]) for n in ("last", "whole")]
        taps = resolve_read_taps(bundle, doc, "patched", "base", reads)
        norm = resolve_site(bundle, SiteSpec(component=HEAD_INPUT))
        head = resolve_site(bundle, SiteSpec(component=HEAD))
        assert taps["last"].site.component == HEAD
        assert tap_key(taps["last"].capture) == tap_key(norm)
        assert taps["last"].project is head_module(bundle)
        assert tap_key(taps["whole"].capture) == tap_key(head)
        assert taps["whole"].project is None
        assert taps_head([taps["whole"].capture])
        assert not taps_head([taps["last"].capture])


@functools.lru_cache(maxsize=None)
def _head_and_width(key: str) -> tuple[Any, int]:
    bundle = load_model(key, device="cpu")
    head = head_module(bundle)
    return head, int(head.weight.shape[1])


def _head_copy(
    key: str, dtype: torch.dtype, device: str
) -> tuple[torch.nn.Linear, int]:
    """The family's real head weights as a plain ``Linear`` in ``dtype`` on
    ``device`` — the module the projection runs, where the test says."""
    head, d_model = _head_and_width(key)
    weight = head.weight.detach().to(dtype)
    linear = torch.nn.Linear(d_model, weight.shape[0], bias=head.bias is not None)
    with torch.no_grad():
        linear.weight.copy_(weight)
        if head.bias is not None:
            linear.bias.copy_(head.bias.detach().to(dtype))
    return linear.to(device=device, dtype=dtype), d_model


@st.composite
def _tables(draw: st.DrawFn) -> tuple[int, int, list[list[int]]]:
    """A row count, a padded width and a per-row table of **distinct**
    positions — dense (one width) or ragged (a width per row), rows with
    no position included."""
    rows = draw(st.integers(1, 5))
    seq = draw(st.integers(1, 7))
    ragged = draw(st.booleans())
    width = draw(st.integers(0 if ragged else 1, seq))
    per_row: list[list[int]] = []
    for _ in range(rows):
        w = draw(st.integers(0, seq)) if ragged else width
        per_row.append(draw(st.permutations(range(seq)))[:w])
    return rows, seq, per_row


@prop
@settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(table=_tables(), dtype=st.sampled_from([torch.float32, torch.bfloat16]))
@pytest.mark.parametrize("key", FAMILIES)
def test_the_head_over_gathered_rows_is_the_full_head_sliced(
    key: str, table: tuple[int, int, list[list[int]]], dtype: torch.dtype
) -> None:
    """The invariant the projection rests on: for the executor's own gather
    (``ExecutorBase._gather`` — an advanced index, contiguous), ``head(x)``
    gathered agrees with ``head(x gathered)`` to a few ulps of the dtype,
    dense or ragged, fp32 and bf16, on every tiny family's real head, on the
    CPU. Each logit is the same dot product over ``d_model`` — what can move
    is the reduction order, when the BLAS picks another kernel for the
    gathered ``M = rows`` than for the full ``M = rows·seq``, and that is
    per library and shape: 📐 MKL fp32 at ``M = 1`` (GEMV), MKL/oneDNN bf16
    at ``M = 2`` vs 8 (x86 node), never on arm64 Accelerate for ``M ≥ 2``,
    and cuBLAS bf16 on the tiny shapes at ``M = 4`` vs 8. The same class of
    rounding a row window or a cohort's batch shape introduces, so the
    property is tolerance-based: ``4·eps`` relative, ``4·eps·|logits|max``
    absolute.

    **Where the projection is exact** is established elsewhere, not by this
    test: measured on an H100 with the ``Qwen/Qwen3.6-35B-A3B`` head at
    M ∈ {42, 96, 900} against ``M·13``, bf16 and fp32, 0 of 223 488 000
    entries differ; and end to end by the
    standard workflow's **zero differing metric rows** against the base tree
    in eager and graphs mode. A gradient never flows through a projection
    (``shared/head.py``), so training is the model's own head, exact by
    construction."""
    rows, seq, per_row = table
    linear, d_model = _head_copy(key, dtype, "cpu")
    x = torch.randn(rows, seq, d_model, generator=torch.Generator().manual_seed(1)).to(
        dtype
    )
    with torch.no_grad():
        full = linear(x)
        gathered = ExecutorBase._gather(x, per_row, "x")  # pyright: ignore[reportPrivateUsage]
        sliced = ExecutorBase._gather(full, per_row, "full")  # pyright: ignore[reportPrivateUsage]
        if isinstance(gathered, RaggedValue):
            assert isinstance(sliced, RaggedValue)
            assert gathered.widths == sliced.widths
            projected, want = linear(gathered.flat), sliced.flat
        else:
            assert isinstance(sliced, torch.Tensor)
            projected, want = linear(gathered), sliced
    _assert_within_ulps(projected, want)


def _assert_within_ulps(got: torch.Tensor, want: torch.Tensor) -> None:
    """``got`` agrees with ``want`` to a few ulps of their dtype: ``4·eps``
    relative, and ``4·eps`` of the largest entry absolute (a logit near zero
    has no meaningful relative error)."""
    eps = torch.finfo(want.dtype).eps
    scale = float(want.abs().max()) if want.numel() else 0.0
    torch.testing.assert_close(got, want, rtol=4 * eps, atol=4 * eps * scale)


@unit
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the CUDA twin of the CPU property"
)
@pytest.mark.parametrize("key", FAMILIES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_on_cuda_the_tiny_head_agrees_to_bf16_ulps(
    key: str, dtype: torch.dtype
) -> None:
    """The CUDA twin of the CPU property at the shape cuBLAS was seen to
    round differently (📐 tiny qwen3.5-moe, bf16, M = 4 vs 8), to the same
    few ulps. The A3B's workflow shapes were bit-identical there; that claim
    is the GPU parity script's, not this test's."""
    linear, d_model = _head_copy(key, dtype, "cuda")
    x = torch.randn(4, 2, d_model, generator=torch.Generator().manual_seed(1)).to(
        "cuda", dtype
    )
    per_row = [[0], [0], [0], [0]]
    with torch.no_grad():
        full = linear(x)
        gathered = ExecutorBase._gather(x, per_row, "x")  # pyright: ignore[reportPrivateUsage]
        sliced = ExecutorBase._gather(full, per_row, "full")  # pyright: ignore[reportPrivateUsage]
        assert isinstance(gathered, torch.Tensor) and isinstance(sliced, torch.Tensor)
        projected = linear(gathered)
    _assert_within_ulps(projected, sliced)


class TestTheTapUnion:
    @unit
    def test_a_shared_group_wants_the_heads_input_for_a_projecting_read(
        self,
    ) -> None:
        """Two points sharing ``original`` on ``base``: one reads the head at
        the last token, the other every position. The union names
        ``ln_final`` for the first and ``lm_head`` for the second — the two
        captures the shared pass has to leave behind — and the intervened
        group's own union is ``ln_final`` alone."""
        docs = [
            _doc(
                last=_head_read({"index": -1}, model="original_base"),
                patched=_head_read({"index": -1}),
            ),
            _doc(
                whole=_head_read("all", model="original_base"),
                patched=_head_read({"index": -1}),
            ),
        ]
        plans = [plan_point(doc) for doc in docs]
        cache = campaign_cache(docs, plans)
        shared = next(
            g
            for g in plans[0].groups
            if g.model == "original_base" and g.input == "base"
        )
        assert shared.key == next(
            g.key
            for g in plans[1].groups
            if g.model == "original_base" and g.input == "base"
        )
        wanted = {spec.component for spec in cache.wanted[shared.key]}
        assert wanted == {HEAD_INPUT, HEAD}
        patched = next(g for g in plans[0].groups if g.model == "patched")
        assert {spec.component for spec in cache.wanted[patched.key]} == {HEAD_INPUT}
