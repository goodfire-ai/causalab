"""Every read value the executor holds is keyed by the read **bound to the
model it is taken on** (``executor.base.BoundRead``, spec §2.7), never by the
bare read name.

A read declares an address; the model that lists it decides which forward
the address is gathered from. Keying by the pair is what lets one read be
listed by several models — the shape the reads-first method block (§2.9)
authors — and it is checked here on the runtime alone, before the schema
learns to spell it."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from causalab.neural.engines.pytorch_hooks.executor import PointExecutor
from causalab.neural.engines.pytorch_hooks.loading import ModelBundle
from causalab.neural.shared.executor import BoundRead
from causalab.neural.shared.plan import bound_read, read_label, saved_raw_reads
from causalab.protocol.schema import ReadRef

from tests.neural.engines.pytorch_hooks._drive import executor_for
from tests.neural.engines.pytorch_hooks.test_prefix_resume import swap_doc
from tests.neural.engines.pytorch_hooks.test_train import (
    ANSWERS,
    BASES,
    COUNTERFACTUALS,
)

pytestmark = pytest.mark.unit


def _executor(raw: dict[str, Any], bundle: ModelBundle) -> PointExecutor:
    return executor_for(
        raw,
        bundle,
        base_texts=BASES,
        counterfactual_texts=COUNTERFACTUALS,
        extra_columns={"label": ANSWERS},
    )


def test_read_values_are_keyed_by_the_read_bound_to_its_model(
    llama_bundle: ModelBundle,
) -> None:
    executor = _executor(swap_doc(), llama_bundle)
    executor.run_all()
    assert set(executor._read_values) == {
        ReadRef("v_cf", "original_counterfactual"),
        ReadRef("logits", "patched"),
    }
    assert all(isinstance(key, BoundRead) for key in executor._read_values)
    # the bare name is sugar for the one binding the read has
    bound = executor.read_value(ReadRef("logits", "patched"))
    assert torch.equal(executor.read_value("logits"), bound)
    assert executor.resolution("logits") == executor.resolution(
        ReadRef("logits", "patched")
    )


def test_bound_read_helpers_spell_the_binding() -> None:
    from causalab.protocol.schema import parse_document

    from tests.protocol._docs import in_order

    raw = swap_doc()
    raw["method"]["save"].append(
        {
            "read": "v_cf",
            "model": "original_counterfactual",
            "file_path": "v_cf.safetensors",
        }
    )
    doc = parse_document(in_order(raw))
    assert bound_read(doc, "v_cf") == ReadRef("v_cf", "original_counterfactual")
    assert (
        read_label(ReadRef("v_cf", "original_counterfactual"))
        == "original_counterfactual/v_cf"
    )
    # a `.json` entry reduces a read; only the tensor entry saves one raw
    assert saved_raw_reads(doc) == frozenset(
        {ReadRef("v_cf", "original_counterfactual")}
    )
