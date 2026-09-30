"""Selected attention backends survive ordinary hooks and interior fallbacks.

Real tiny models run SDPA on CPU so these checks need no FlashAttention build.
Model pre-hooks observe the implementation at forward time, before mask creation.
"""

from __future__ import annotations

# These integration checks reuse fixture builders and inspect cache lifecycle.
# pyright: reportPrivateUsage=false

import contextlib
from typing import Iterator

import pytest
import torch

from causalab.neural.engines.pytorch_hooks.loading import ModelBundle, load_model
from causalab.neural.shared.executor import RowWindow
from causalab.neural.shared.sites import resolve_site
from causalab.protocol.registry import ATTENTION_FUNCTION_SLOTS
from causalab.protocol.schema import SiteSpec

from ._drive import executor_for
from .conftest import TINY_GPT2, TINY_LLAMA, TINY_QWEN35_MOE
from .test_prefix_resume import _campaign, _executor, _scan
from .test_sites_round2_attention import _write_doc
from .test_sites_round2_function import _read_doc

pytestmark = pytest.mark.smoke

TEXT = "1 2 3 4"
CF_TEXT = "4 3 2 1"


@pytest.fixture(params=[TINY_LLAMA, TINY_GPT2, TINY_QWEN35_MOE])
def accelerated(request: pytest.FixtureRequest) -> Iterator[ModelBundle]:
    bundle = load_model(request.param, attn_implementation="sdpa")
    assert bundle.model.config._attn_implementation == "sdpa"
    yield bundle
    bundle.model.set_attn_implementation("sdpa")


def _layer(bundle: ModelBundle) -> int:
    return bundle.streams.index("full_attention")


@contextlib.contextmanager
def observed(bundle: ModelBundle) -> Iterator[list[str]]:
    seen: list[str] = []
    handle = bundle.model.register_forward_pre_hook(
        lambda model, args: seen.append(model.config._attn_implementation)
    )
    try:
        yield seen
    finally:
        handle.remove()


@pytest.mark.parametrize(
    "component", ["block_output", "attention_output", "attention_value_states"]
)
def test_module_interventions_keep_the_selected_backend(accelerated, component):
    doc = _write_doc(component, _layer(accelerated))
    executor = executor_for(
        doc, accelerated, base_texts=[TEXT], counterfactual_texts=[CF_TEXT]
    )
    with observed(accelerated) as seen:
        after = executor.read_value("after")
        clean = executor.read_value("clean")
    assert seen and set(seen) == {"sdpa"}
    assert isinstance(after, torch.Tensor) and isinstance(clean, torch.Tensor)
    assert not torch.equal(after, clean)
    assert accelerated.model.config._attn_implementation == "sdpa"


@pytest.mark.parametrize(
    "component",
    [
        "attention_query",
        "attention_key",
        "attention_scores",
        "attention_probs",
        "attention_z",
    ],
)
def test_interior_reads_match_eager_and_restore_the_backend(accelerated, component):
    doc = _read_doc(component, _layer(accelerated))
    with observed(accelerated) as seen:
        actual = executor_for(doc, accelerated, base_texts=[TEXT]).read_value("r")
    assert seen == ["eager"]
    assert accelerated.model.config._attn_implementation == "sdpa"
    accelerated.model.set_attn_implementation("eager")
    expected = executor_for(doc, accelerated, base_texts=[TEXT]).read_value("r")
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "component", ["attention_scores", "attention_probs", "attention_z"]
)
def test_interior_writes_match_eager_and_other_groups_keep_sdpa(accelerated, component):
    doc = _write_doc(component, _layer(accelerated))
    executor = executor_for(
        doc, accelerated, base_texts=[TEXT], counterfactual_texts=[CF_TEXT]
    )
    with observed(accelerated) as seen:
        actual = executor.read_value("after")
        clean = executor.read_value("clean")
    assert seen == ["eager", "eager", "sdpa"]
    assert isinstance(actual, torch.Tensor) and isinstance(clean, torch.Tensor)
    assert not torch.equal(actual, clean)
    assert accelerated.model.config._attn_implementation == "sdpa"
    accelerated.model.set_attn_implementation("eager")
    expected = executor_for(
        doc, accelerated, base_texts=[TEXT], counterfactual_texts=[CF_TEXT]
    ).read_value("after")
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "component,decode_backend", [("block_output", "sdpa"), ("attention_z", "eager")]
)
def test_decode_selects_its_own_backend(accelerated, component, decode_backend):
    doc = _read_doc(component, _layer(accelerated))
    doc["method"]["positions"] = {
        "cont": {"generated": {"max_new_tokens": 2}, "all": True}
    }
    doc["method"]["reads"]["r"]["pos"] = "cont"
    with observed(accelerated) as seen:
        value = executor_for(doc, accelerated, base_texts=[TEXT]).read_value("r")
    assert isinstance(value, torch.Tensor)
    assert value.numel() > 0
    assert seen == [decode_backend, decode_backend, decode_backend]
    assert accelerated.model.config._attn_implementation == "sdpa"
    if decode_backend == "eager":
        accelerated.model.set_attn_implementation("eager")
        expected = executor_for(doc, accelerated, base_texts=[TEXT]).read_value("r")
        torch.testing.assert_close(value, expected, rtol=0, atol=0)


def test_failed_interior_forward_restores_backend_and_registry(accelerated):
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    previous = ALL_ATTENTION_FUNCTIONS.get("eager")
    had_key = "eager" in ALL_ATTENTION_FUNCTIONS

    def fail(model, args):
        assert model.config._attn_implementation == "eager"
        raise RuntimeError("injected forward failure")

    handle = accelerated.model.register_forward_pre_hook(fail)
    try:
        with pytest.raises(RuntimeError, match="injected forward failure"):
            executor_for(
                _read_doc("attention_scores", _layer(accelerated)),
                accelerated,
                base_texts=[TEXT],
            ).read_value("r")
    finally:
        handle.remove()
    assert accelerated.model.config._attn_implementation == "sdpa"
    assert ("eager" in ALL_ATTENTION_FUNCTIONS) == had_key
    assert ALL_ATTENTION_FUNCTIONS.get("eager") is previous
    assert not any(
        m._forward_hooks or m._forward_pre_hooks for m in accelerated.model.modules()
    )


def test_prefixes_do_not_cross_backends():
    bundle = load_model(TINY_LLAMA, attn_implementation="sdpa")
    raws = _scan()
    _, handles, cache = _campaign(raws)
    executor = _executor(raws[0], bundle, interning=handles[0])
    key = handles[0].keys[("patched", "base")]
    plan = cache.prefix_plans[key]
    window = RowWindow(0, 1, 1)
    interior = resolve_site(bundle, SiteSpec(component="attention_scores", layers=[1]))
    sdpa_key = executor._prefix_key(plan, window, 1)
    cache.prefixes[sdpa_key] = torch.ones(1)
    with executor._attention_backend([interior]):
        eager_key = executor._prefix_key(plan, window, 1)
        assert eager_key not in cache.prefixes
        cache.prefixes[eager_key] = torch.zeros(1)
    assert sdpa_key != eager_key
    assert executor._prefix_key(plan, window, 1) == sdpa_key
    # Both variants must be released by the existing reference-count policy.
    while cache.prefix_owed[(plan.base_key, 1)] > 0:
        executor._settle_prefixes(key)
    assert not cache.prefixes


def test_sdpa_prefix_resume_matches_a_whole_forward():
    bundle = load_model(TINY_LLAMA, attn_implementation="sdpa")
    raws = _scan()
    _, handles, cache = _campaign(raws)
    for raw, handle in zip(raws, handles):
        actual = _executor(raw, bundle, interning=handle).read_value("logits")
        expected = _executor(raw, bundle, interning=None).read_value("logits")
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert cache.resumed
    assert not cache.prefixes


@pytest.mark.parametrize("component", ["delta_kernel_output", "expert_output"])
def test_other_kernel_interiors_do_not_force_eager(component):
    bundle = load_model(TINY_QWEN35_MOE, attn_implementation="sdpa")
    with observed(bundle) as seen:
        value = executor_for(
            _read_doc(component, 0), bundle, base_texts=[TEXT]
        ).read_value("r")
    assert isinstance(value, torch.Tensor) and value.numel() > 0
    assert seen == ["sdpa"]


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("component", ["attention_probs", *ATTENTION_FUNCTION_SLOTS])
def test_campaign_interior_taps_do_not_change_module_only_values(reverse, component):
    bundle = load_model(TINY_LLAMA, attn_implementation="sdpa")
    raws = [_read_doc(kind, 1) for kind in ("block_output", component)]
    for raw in raws:
        raw["model"]["attn_implementation"] = "sdpa"
    if reverse:
        raws.reverse()
    _, handles, _ = _campaign(raws)
    assert handles[0].keys != handles[1].keys
    for raw, handle in zip(raws, handles):
        with observed(bundle) as seen:
            actual = executor_for(
                raw, bundle, interning=handle, base_texts=[TEXT]
            ).read_value("r")
        expected_backend = (
            "eager"
            if raw["method"]["sites"]["tap"]["component"] == component
            else "sdpa"
        )
        assert seen == [expected_backend]
        expected = executor_for(raw, bundle, base_texts=[TEXT]).read_value("r")
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_switch_failure_remains_primary_when_restoration_also_fails(monkeypatch):
    bundle = load_model(TINY_LLAMA, attn_implementation="sdpa")
    executor = executor_for(_read_doc("attention_probs", 1), bundle, base_texts=[TEXT])

    def broken_switch(backend):
        raise RuntimeError("initial switch" if backend == "eager" else "restore")

    monkeypatch.setattr(bundle.model, "set_attn_implementation", broken_switch)
    with pytest.raises(RuntimeError, match="initial switch") as err:
        executor.read_value("r")
    assert str(err.value.__cause__) == "restore"
    assert load_model.cache_info().currsize == 0


def test_composite_backend_prefix_keys_are_stable_and_distinct(monkeypatch):
    bundle = load_model(TINY_LLAMA, attn_implementation="sdpa")
    raws = _scan()
    _, handles, cache = _campaign(raws)
    executor = _executor(raws[0], bundle, interning=handles[0])
    plan = cache.prefix_plans[handles[0].keys[("patched", "base")]]
    window = RowWindow(0, 1, 1)
    from types import SimpleNamespace

    monkeypatch.setattr(
        bundle.model, "config", SimpleNamespace(_attn_implementation=None)
    )
    monkeypatch.setattr(
        bundle.model.config, "_attn_implementation", {"text": "sdpa", "vision": "eager"}
    )
    first = executor._prefix_key(plan, window, 1)
    monkeypatch.setattr(
        bundle.model.config, "_attn_implementation", {"vision": "eager", "text": "sdpa"}
    )
    assert executor._prefix_key(plan, window, 1) == first
    monkeypatch.setattr(
        bundle.model.config,
        "_attn_implementation",
        {"text": "eager", "vision": "eager"},
    )
    second = executor._prefix_key(plan, window, 1)
    assert len({first, second}) == 2


@pytest.mark.parametrize("missing", [False, True])
def test_unknown_backend_refuses_before_switching(monkeypatch, missing):
    from types import SimpleNamespace
    from causalab.protocol.rules.errors import ProtocolError

    bundle = load_model(TINY_LLAMA, attn_implementation="sdpa")
    executor = executor_for(_read_doc("attention_probs", 1), bundle, base_texts=[TEXT])
    site = resolve_site(bundle, SiteSpec(component="attention_probs", layers=[1]))
    config = (
        SimpleNamespace() if missing else SimpleNamespace(_attn_implementation=None)
    )
    monkeypatch.setattr(bundle.model, "config", config)
    switches = []
    monkeypatch.setattr(bundle.model, "set_attn_implementation", switches.append)
    with pytest.raises(ProtocolError, match="configured attention backend"):
        with executor._attention_backend([site]):
            pytest.fail("must refuse before forwarding")
    assert switches == []
    assert vars(config) == ({} if missing else {"_attn_implementation": None})


def test_plugin_attention_slots_partition_forward_groups(monkeypatch):
    import dataclasses
    from causalab.neural.shared import plan
    from causalab.protocol.registry import Tap

    raw = _read_doc("attention_value_states", 1)
    _, before, _ = _campaign([raw])
    adapter = next(iter(plan.FAMILIES.values()))
    plugin = dataclasses.replace(
        adapter,
        taps={
            **adapter.taps,
            "attention_value_states": Tap("mixer", kind="interface", slot="value"),
        },
    )
    monkeypatch.setattr(plan, "FAMILIES", {**plan.FAMILIES, "test_plugin": plugin})
    _, after, _ = _campaign([raw])
    assert before[0].keys != after[0].keys
