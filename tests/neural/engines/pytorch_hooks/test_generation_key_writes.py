"""Decode writes address the newest key without changing its cached prefix."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from hypothesis import given, settings, strategies as st

from causalab.neural.engines.pytorch_hooks import attention_interface
from causalab.neural.engines.pytorch_hooks.executor import PointExecutor
from causalab.neural.engines.pytorch_hooks.loading import ModelBundle
from causalab.neural.shared.executor.ragged import RowWindow
from causalab.neural.shared.fires import FireTally
from tests.analysis.test_sequence_analysis import _bundle, executor
from tests.protocol._docs import in_order


@pytest.fixture(scope="module")
def local_bundle() -> ModelBundle:
    return _bundle()


def _run(bundle: ModelBundle, components: tuple[str, ...]) -> PointExecutor:
    raw: dict[str, Any] = {
        "header": {"protocol_version": "4"},
        "model": {"key": bundle.key, "revision": bundle.revision},
        "data": {"base": {"dataset": "inline", "field": "input"}},
        "method": {
            "positions": {
                "generated": {"generated": {"max_new_tokens": 6}, "all": True}
            },
            "sites": {
                "head": {"component": "lm_head"},
                **{name: {"component": name, "layers": [1]} for name in components},
            },
            "writes": {
                f"write_{name}": {
                    "site": name,
                    "pos": -1,
                    "do": {"add_scaled": {"op": 5.0, "alpha": 1.0}},
                }
                for name in components
            },
            "intervened_models": {
                "steered": {
                    "input": "base",
                    "reads": ["continuation"],
                    "writes": [f"write_{name}" for name in components],
                    "writes_during_generation": True,
                }
            },
            "reads": {"continuation": {"site": "head", "pos": "generated"}},
            "save": [
                {
                    "read": "continuation",
                    "model": "steered",
                    "file_path": "out.safetensors",
                }
            ],
        },
    }
    result = executor(bundle, in_order(raw), [{"input": "a b c"}, {"input": "a b"}])
    result.decoding = {"mode": "deterministic", "eos_token_ids": []}
    return result


@pytest.mark.property
@settings(max_examples=20, deadline=None)
@given(
    prefix_length=st.integers(min_value=1, max_value=32), offset=st.integers(-100, 100)
)
def test_decode_writers_preserve_every_cached_prefix(
    local_bundle: ModelBundle, prefix_length: int, offset: int
) -> None:
    # Build both writers together: neither may capture the other site's axes.
    run = _run(
        local_bundle,
        (
            "attention_key",
            "attention_query",
            "attention_key_pre_rope",
            "attention_value_states",
        ),
    )
    addresses = run._resolve_write_addresses(tuple(run.doc.writes))
    hooks = run._build_decode_write_hooks(
        addresses, "base", run._batch("base"), RowWindow(0, 2, 2), [FireTally()]
    )
    for site, write in hooks:
        positions = prefix_length + 1 if site.component == "attention_key" else 1
        width = site.shape.width
        assert width is not None
        values = (
            torch.arange(2 * positions * width, dtype=torch.float32).reshape(
                2, positions, width
            )
            + offset
        )
        before = values.clone()
        write(values)
        assert torch.equal(values[:, :-1], before[:, :-1])
        assert torch.equal(values[:, -1], before[:, -1] + 5.0)


@pytest.mark.smoke
@pytest.mark.parametrize("batch_rows", [None, 1])
def test_attention_key_edits_the_newest_token_at_every_decode_step(
    local_bundle: ModelBundle, monkeypatch: pytest.MonkeyPatch, batch_rows: int | None
) -> None:
    run = _run(local_bundle, ("attention_key",))
    run.batch_rows = batch_rows
    original = attention_interface._apply
    observed: list[tuple[int, list[int]]] = []

    def observe(taps: Any, slot: str, value: torch.Tensor) -> torch.Tensor:
        before = value.clone()
        after = original(taps, slot, value)
        if slot == "key" and any(tap.edit is not None for tap in taps):
            changed = (after - before).abs().sum(dim=(0, 1, 3))
            observed.append((value.shape[2], changed.nonzero().flatten().tolist()))
        return after

    monkeypatch.setattr(attention_interface, "_apply", observe)
    run.run_all()
    windows = 1 if batch_rows is None else 2
    assert observed == [(length, [length - 1]) for length in range(3, 10)] * windows
    assert run.fires == {("steered", "base"): {"write_attention_key": 1}}
