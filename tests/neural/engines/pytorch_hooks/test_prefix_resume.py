"""Resuming an intervened forward from a cached prefix (spec §4, "Resume").

An intervened model whose shallowest write lands in block ``L`` computes
blocks ``0..L-1`` exactly as ``original`` would on the same rows — no write
reaches them and no trained parameter is upstream. Yet every such forward
used to run the whole stack: every optimizer step, eval pass and later point
of a fit at layer 20 re-ran twenty blocks of un-intervened prefix. Now the
residual entering block ``L`` is stored once per (prefix identity, rows,
window, depth) and every later forward that may start at ``L`` starts there.

Ground truth for "a block ran" is a pre-hook on each decoder block of the
loaded model, not the engine's own tally — the tally (``blocks_skipped``) is
then checked against it. Parity is bit-exact: the resumed forward hands block
``L`` the very tensor it would have computed, so nothing rounds differently.
"""

from __future__ import annotations

import contextlib
import dataclasses
import shutil
import types
from pathlib import Path
from typing import Any, Iterator, Sequence

import pandas as pd
import pytest
import torch

from causalab.cli import register_model_key
from causalab.neural.engines.pytorch_hooks.engine import PytorchHooksEngine
from causalab.neural.engines.pytorch_hooks import executor as executor_module
from causalab.neural.engines.pytorch_hooks.executor import PointExecutor
from causalab.neural.engines.pytorch_hooks.loading import ModelBundle, load_model
from causalab.neural.engines.pytorch_hooks.train import run_training
from causalab.neural.shared import execution
from causalab.neural.shared.execution import campaign_cache
from causalab.neural.shared.executor import ForwardCache, Interning
from causalab.protocol.compiled import CompiledProtocol
from causalab.protocol.engine import RunContext
from causalab.protocol.pipeline import compile_protocol
from causalab.neural.shared.plan import plan_point
from causalab.protocol.registry import model_info_from_hf_config
from causalab.io.env import ResolutionEnv
from causalab.protocol.schema import Document, parse_document

from tests.neural.engines.pytorch_hooks._drive import base_data_section, executor_for
from tests.neural.engines.pytorch_hooks.conftest import TINY_LLAMA, TINY_QWEN35_MOE
from tests.neural.engines.pytorch_hooks.test_train import (
    ANSWERS,
    BASES,
    CE,
    COUNTERFACTUALS,
    ce_term,
    das_doc,
    dbm_doc,
)
from tests.protocol._docs import UNWRITTEN, in_order, saved
from tests.protocol._env import (
    CORPUS_DIR,
    FIXTURES,
    build_env,
    write_rot_fixture,
    steps_of,
)
from tests.tables import frame as table_frame

DATA_IDENTITY = {
    "base": "inline#input",
    "counterfactual": "inline#counterfactual_inputs[0]",
}
EVAL_SPLIT = "inline#eval"
EVAL_ROWS = [
    {
        "input": "the tallest tree in the forest is",
        "counterfactual_inputs": ["the deepest lake in the valley is"],
        "label": " one",
    },
    {
        "input": "nine purple kites drift over",
        "counterfactual_inputs": ["two rusty bicycles lean against"],
        "label": " two",
    },
    {
        "input": "her grandmother's kitchen always smelled of",
        "counterfactual_inputs": ["his uncle's workshop always sounded like"],
        "label": " three",
    },
]
B = 2  # minibatches: 4 training rows at pairs=2


# --------------------------------------------------------------------------- #
# documents
# --------------------------------------------------------------------------- #


def swap_doc(
    key: str = TINY_LLAMA,
    layer: int = 1,
    component: str = "block_output",
    pos: int = -1,
) -> dict[str, Any]:
    """One swap at ``(component, layer)`` scored at lm_head — the shape of
    every interchange, control and ablation point.

    ``pos`` is the write's position. Two documents differing only in it are
    two points of one scan (corpus 07's position axis): different patched
    closures — so neither is served the other's captures (§3) — over one
    un-intervened prefix."""
    return {
        "header": {"protocol_version": "4"},
        "model": {"key": key, "revision": "main"},
        "data": base_data_section(with_counterfactual=True),
        "method": {
            "intervened_models": {
                UNWRITTEN: {"input": "counterfactual", "reads": ["v_cf"]},
                "patched": {"input": "base", "reads": ["logits"], "writes": ["patch"]},
            },
            "sites": {
                "tgt": {"component": component, "layers": [layer]},
                "lm_head": {"component": "lm_head"},
            },
            "reads": {
                "v_cf": {"site": "tgt", "pos": {"index": -1}},
                "logits": {"site": "lm_head", "pos": {"index": -1}},
            },
            "writes": {
                "patch": {"site": "tgt", "pos": {"index": pos}, "do": {"swap": "v_cf"}}
            },
            "save": [saved("logits", "patched", "ce.json", dict(CE))],
        },
    }


def _train_doc(kind: str, *, layer: int = 1, epochs: int = 3) -> dict[str, Any]:
    raw = das_doc(epochs=epochs) if kind == "das" else dbm_doc()
    raw["method"]["sites"]["tgt"]["layers"] = layer
    raw["method"]["train"]["steps"] = {"epochs": epochs}
    raw["method"]["train"]["batch"] = {"pairs": 2}
    raw["method"]["train"]["eval"] = {
        "every": {"epochs": 1},
        "split": EVAL_SPLIT,
        "aggregations": {"ce": ce_term()},
    }
    return raw


class _InlineDatasets:
    def __init__(self, splits: dict[str, list[dict[str, Any]]]) -> None:
        self._splits = splits

    def digest(self, ref: str) -> str:
        return "0" * 64

    def columns(self, ref: str) -> tuple[str, ...]:
        return tuple(self._splits[ref][0]) if self._splits.get(ref) else ()

    def rows(self, ref: str) -> list[dict[str, Any]]:
        return self._splits[ref]


def _train_request() -> RunContext:
    return RunContext(
        env=ResolutionEnv(
            datasets=_InlineDatasets({EVAL_SPLIT: EVAL_ROWS}), artifacts=None
        ),  # type: ignore[arg-type]
        output_dir=None,  # type: ignore[arg-type]
    )


# --------------------------------------------------------------------------- #
# plumbing
# --------------------------------------------------------------------------- #


def _scan(
    layer: int = 1, *, key: str = TINY_LLAMA, component: str = "block_output"
) -> list[dict[str, Any]]:
    """Two points of a scan at one layer: the write at the last position and
    at the one before it."""
    return [
        swap_doc(key=key, layer=layer, component=component, pos=pos) for pos in (-1, -2)
    ]


def _campaign(
    raws: Sequence[dict[str, Any]], cache: ForwardCache | None = None
) -> tuple[list[Document], list[Interning], ForwardCache]:
    """The points' handles on one campaign cache, built the way
    ``execute_request`` builds them: plan keys, the tap union and the
    prefix plans of the whole point set."""
    docs = [parse_document(in_order(raw)) for raw in raws]
    plans = [plan_point(doc, data_identity=DATA_IDENTITY) for doc in docs]
    if cache is None:
        cache = campaign_cache(docs, plans)
    handles = [
        Interning(keys={(g.model, g.input): g.key for g in plan.groups}, cache=cache)
        for plan in plans
    ]
    return docs, handles, cache


def _executor(
    raw: dict[str, Any],
    bundle: ModelBundle,
    *,
    interning: Interning | None,
    batch_rows: int | None = None,
) -> PointExecutor:
    return executor_for(
        raw,
        bundle,
        base_texts=BASES,
        counterfactual_texts=COUNTERFACTUALS,
        extra_columns={"label": ANSWERS},
        interning=interning,
        batch_rows=batch_rows,
    )


def _reads(executor: PointExecutor) -> dict[str, torch.Tensor]:
    executor.run_all()
    return {name: executor.dense_value(name) for name in executor.doc.reads}


def _assert_same_reads(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> None:
    assert set(a) == set(b)
    for name in a:
        assert torch.equal(a[name], b[name]), name


@contextlib.contextmanager
def _block_fires(bundle: ModelBundle) -> Iterator[list[list[int]]]:
    """Per decoder block, the batch size of every forward it ran — a pre-hook
    on the block module itself, which a skipped block never reaches."""
    fires: list[list[int]] = [[] for _ in bundle.blocks]
    handles = []
    for index, block in enumerate(bundle.blocks):

        def hook(_m: Any, args: tuple[Any, ...], _kw: Any, *, i: int = index) -> None:
            hidden = args[0] if args else _kw["hidden_states"]
            fires[i].append(int(hidden.shape[0]))

        handles.append(block.register_forward_pre_hook(hook, with_kwargs=True))
    try:
        yield fires
    finally:
        for handle in handles:
            handle.remove()


def _weights(outcome: Any) -> dict[str, torch.Tensor]:
    return {
        f"{name}.{slot}": param.detach().clone()
        for name, stage in outcome.stages.items()
        for slot, param in stage.slot_params().items()
    }


@pytest.fixture(scope="module")
def bundle() -> ModelBundle:
    return load_model(TINY_LLAMA)


@pytest.fixture(scope="module")
def moe_bundle() -> ModelBundle:
    return load_model(TINY_QWEN35_MOE)


# --------------------------------------------------------------------------- #
# unit: the mechanism, one executor at a time
# --------------------------------------------------------------------------- #


@pytest.mark.unit
class TestResume:
    def test_a_later_point_on_the_same_prefix_skips_the_blocks_below_its_write(
        self, bundle: ModelBundle
    ) -> None:
        """Two points writing at layer 1 on the same rows: the first runs the
        whole stack and leaves the residual entering block 1 behind; the
        second starts there. Results are bit-identical to no reuse at all."""
        raw_a, raw_b = _scan(layer=1)
        _docs, (first, second), cache = _campaign([raw_a, raw_b])
        plain_a = _reads(_executor(raw_a, bundle, interning=None))
        plain_b = _reads(_executor(raw_b, bundle, interning=None))

        with _block_fires(bundle) as fires:
            a = _reads(_executor(raw_a, bundle, interning=first))
        # the harvest and the patched forward both ran every block
        assert fires[0] == [4, 4] and fires[1] == [4, 4]
        assert cache.resumed == []
        _assert_same_reads(a, plain_a)

        with _block_fires(bundle) as fires:
            b = _reads(_executor(raw_b, bundle, interning=second))
        # the harvest is interned (§3); the patched forward resumed at block 1
        assert fires[0] == []
        assert fires[1] == [4]
        assert cache.resumed == [1]
        _assert_same_reads(b, plain_b)

    def test_the_prefix_is_keyed_by_the_un_intervened_identity(
        self, bundle: ModelBundle
    ) -> None:
        """Two intervened models writing at different layers on the same rows
        share one prefix identity; a point running the deeper one after the
        shallower one still finds nothing *below* the shallower write stored
        from the intervened pass — that residual is post-write."""
        shallow = swap_doc(layer=0)
        deep, deep_again = _scan(layer=1)
        _docs, (h_shallow, h_deep, h_again), cache = _campaign(
            [shallow, deep, deep_again]
        )
        plain_deep = _reads(_executor(deep, bundle, interning=None))
        plain_again = _reads(_executor(deep_again, bundle, interning=None))

        _reads(_executor(shallow, bundle, interning=h_shallow))
        # the shallow pass writes at block 0: everything entering block 1 is
        # intervened, so it may store nothing at depth 1
        assert cache.prefixes == {}
        first = _reads(_executor(deep, bundle, interning=h_deep))
        assert cache.resumed == []
        assert len(cache.prefixes) == 1
        _assert_same_reads(first, plain_deep)
        # ...and the deep pass stored it: the next point at layer 1 resumes
        again = _reads(_executor(deep_again, bundle, interning=h_again))
        assert cache.resumed == [1]
        _assert_same_reads(again, plain_again)

    def test_an_original_group_that_runs_anyway_contributes_the_prefix(
        self, bundle: ModelBundle
    ) -> None:
        """A read on the un-intervened model on ``base`` plans the
        un-intervened forward over the very rows the patched forward reads;
        its pass stores the prefix, so even the *first* patched forward
        resumes. Declared first, since groups run lazily in read order."""
        raw = swap_doc(layer=1)
        clean = {"site": "lm_head", "pos": {"index": -1}}
        raw["method"]["reads"] = {"logits_clean": clean, **raw["method"]["reads"]}
        raw["method"]["intervened_models"] = {
            "original_base": {"input": "base", "reads": ["logits_clean"]},
            **raw["method"]["intervened_models"],
        }
        raw["method"]["save"].append(
            saved(
                "logits",
                "patched",
                "kl.json",
                {
                    "kind": "kl",
                    "target": {"read": "logits_clean", "model": "original_base"},
                },
            )
        )
        _docs, (handle,), cache = _campaign([raw])
        plain = _reads(_executor(raw, bundle, interning=None))
        with _block_fires(bundle) as fires:
            reads = _reads(_executor(raw, bundle, interning=handle))
        _assert_same_reads(reads, plain)
        assert cache.resumed == [1]
        # three groups; block 0 ran for the two original ones only
        assert fires[0] == [4, 4]
        assert fires[1] == [4, 4, 4]

    def test_no_resume_when_a_tap_sits_below_the_write(
        self, bundle: ModelBundle
    ) -> None:
        """A read of block 0's output on the intervened model needs block 0
        to run, whatever the write's depth: every pass runs it, and the
        values are the un-resumed ones."""
        raws = _scan(layer=1)
        for raw in raws:
            raw["method"]["sites"]["probe"] = {
                "component": "block_output",
                "layers": [0],
            }
            raw["method"]["reads"]["below"] = {"site": "probe", "pos": {"index": -1}}
            raw["method"]["intervened_models"]["patched"]["reads"].append("below")
            raw["method"]["save"].append(saved("below", "patched", "below.safetensors"))
        raw_a, raw_b = raws
        _docs, (first, second), cache = _campaign(raws)
        plain_a = _reads(_executor(raw_a, bundle, interning=None))
        plain_b = _reads(_executor(raw_b, bundle, interning=None))
        _assert_same_reads(_reads(_executor(raw_a, bundle, interning=first)), plain_a)
        with _block_fires(bundle) as fires:
            _assert_same_reads(
                _reads(_executor(raw_b, bundle, interning=second)), plain_b
            )
        assert fires[0] == [4]  # the patched forward ran block 0 again
        assert cache.resumed == []
        assert cache.prefixes == {}

    @pytest.mark.parametrize("tapped_first", [False, True])
    def test_a_pass_stores_no_prefix_nobody_can_still_read(
        self, bundle: ModelBundle, tapped_first: bool
    ) -> None:
        """Two points on one prefix identity: one resumes at 1, the other has
        a tap below its write and never resumes, yet its pass is un-intervened
        to depth 1 and could store it. Once the resuming point has settled,
        nobody is left to read that depth, so the other pass must not store it
        — else the residual leaks to the end of the request, in one order of
        the two and not the other."""
        resuming, tapped = _scan(layer=1)
        tapped["method"]["sites"]["probe"] = {
            "component": "block_output",
            "layers": [0],
        }
        tapped["method"]["reads"]["below"] = {"site": "probe", "pos": {"index": -1}}
        tapped["method"]["intervened_models"]["patched"]["reads"].append("below")
        tapped["method"]["save"].append(saved("below", "patched", "below.safetensors"))
        order = [tapped, resuming] if tapped_first else [resuming, tapped]
        _docs, handles, cache = _campaign(order)
        assert cache.prefix_owed == {
            (
                handles[0]
                .cache.prefix_plans[handles[0].keys[("patched", "base")]]
                .base_key,
                1,
            ): 1
        }
        for raw, handle in zip(order, handles):
            _reads(_executor(raw, bundle, interning=handle))
        # tapped first: its pass stores depth 1 for the resuming point, which
        # resumes and settles it. Resuming first: it stores and settles depth 1
        # itself; the tapped pass then finds nobody owed it and stores nothing
        assert cache.resumed == ([1] if tapped_first else [])
        assert cache.prefixes == {}
        assert all(count == 0 for count in cache.prefix_owed.values())

    def test_row_windows_key_their_own_prefixes(self, bundle: ModelBundle) -> None:
        """Under ``batch_rows`` each window is its own forward, so each has its
        own prefix; parity is against the same windowing without reuse."""
        raw_a, raw_b = _scan(layer=1)
        _docs, (first, second), cache = _campaign([raw_a, raw_b])
        plain_a = _reads(_executor(raw_a, bundle, interning=None, batch_rows=2))
        plain_b = _reads(_executor(raw_b, bundle, interning=None, batch_rows=2))
        _assert_same_reads(
            _reads(_executor(raw_a, bundle, interning=first, batch_rows=2)), plain_a
        )
        windows = {key[2] for key in cache.prefixes}
        assert windows == {(0, 2), (2, 4)}
        assert {key[3] for key in cache.prefixes} == {1}
        with _block_fires(bundle) as fires:
            _assert_same_reads(
                _reads(_executor(raw_b, bundle, interning=second, batch_rows=2)),
                plain_b,
            )
        assert fires[0] == []
        assert fires[1] == [2, 2]
        assert cache.resumed == [1, 1]

    def test_a_window_of_the_whole_role_and_a_minibatch_never_share(
        self, bundle: ModelBundle
    ) -> None:
        """The same two rows, once as a fit's minibatch ``(0, 1)`` and once as
        window ``[0, 2)`` of the whole role, are padded to different frames:
        their keys differ by the rows coordinate, never by luck. In a fit's
        order — sliced passes first, the point's own pass after — the sliced
        prefix is stored, the whole-role windows resume nothing from it, and
        the point's settle drops both."""
        raw = swap_doc(layer=1)
        _docs, (handle,), cache = _campaign([raw])
        rows = [
            {"input": BASES[i], "counterfactual_inputs": [COUNTERFACTUALS[i]]}
            for i in (0, 1)
        ]
        inner = PointExecutor(
            parse_document(in_order(raw)),
            bundle,
            role_rows={"base": rows, "counterfactual": rows},
            role_fields={"base": "input", "counterfactual": "counterfactual_inputs[0]"},
            load_tensors=lambda path: (_ for _ in ()).throw(KeyError(path)),
            interning=Interning(
                keys=handle.keys, cache=cache, rows=(0, 1), counted=False
            ),
        )
        inner.run_all()
        assert {key[1] for key in cache.prefixes} == {(0, 1)}
        assert cache.resumed == []
        _reads(_executor(raw, bundle, interning=handle, batch_rows=2))
        # window [0, 2) of the whole role found nothing under its own key
        assert cache.resumed == []
        # ...and the point's pass, the last sharer, took the sliced key with it
        assert cache.prefixes == {}

    def test_an_unverified_family_runs_the_whole_forward(
        self, bundle: ModelBundle
    ) -> None:
        """Resume is allow-listed by the resolved family (``ModelInfo.family``):
        one whose decoder loop was not verified runs whole and stores nothing,
        even on a module tree the swap would mechanically fit."""
        unverified = dataclasses.replace(
            bundle, info=dataclasses.replace(bundle.info, family="unverified_family")
        )
        assert not executor_module._resumable(unverified)
        raw_a, raw_b = _scan(layer=1)
        _docs, (first, second), cache = _campaign([raw_a, raw_b])
        plain_b = _reads(_executor(raw_b, unverified, interning=None))
        _reads(_executor(raw_a, unverified, interning=first))
        with _block_fires(unverified) as fires:
            _assert_same_reads(
                _reads(_executor(raw_b, unverified, interning=second)), plain_b
            )
        assert fires[0] == [4]
        assert cache.resumed == []
        assert cache.prefixes == {}

    def test_the_gpt2_tree_runs_the_whole_forward(self) -> None:
        """The resume mechanism is verified on the Llama tree's decoder loop;
        the other tree falls back to a full forward and stores nothing."""
        from tests.neural.engines.pytorch_hooks.conftest import TINY_GPT2

        gpt2 = load_model(TINY_GPT2)
        raw_a, raw_b = _scan(layer=1, key=TINY_GPT2)
        _docs, (first, second), cache = _campaign([raw_a, raw_b])
        plain_a = _reads(_executor(raw_a, gpt2, interning=None))
        plain_b = _reads(_executor(raw_b, gpt2, interning=None))
        _assert_same_reads(_reads(_executor(raw_a, gpt2, interning=first)), plain_a)
        with _block_fires(gpt2) as fires:
            _assert_same_reads(
                _reads(_executor(raw_b, gpt2, interning=second)), plain_b
            )
        assert fires[0] == [4]
        assert cache.resumed == []
        assert cache.prefixes == {}


@pytest.mark.unit
class TestFitResume:
    def test_each_minibatch_and_the_eval_split_store_once_and_resume_after(
        self, bundle: ModelBundle
    ) -> None:
        """A DAS fit at layer 1: block 0 runs for the ``B`` source slices and
        the eval source (frozen, interned, §4 "Fits") and once per slice for
        the *patched* forward — never once per step. Block 1 runs on every
        forward, as the model-level tally already pins."""
        epochs = 3
        raw = _train_doc("das", epochs=epochs)
        _docs, (handle,), cache = _campaign([raw])
        executor = _executor(raw, bundle, interning=handle)
        with _block_fires(bundle) as fires:
            run_training(executor.doc, executor, _train_request())
        assert len(fires[1]) == B + B * epochs + 1 + epochs
        # source forwards: B minibatches + 1 eval; patched first passes: the same
        assert len(fires[0]) == 2 * (B + 1)
        assert sorted(fires[0]) == sorted([2] * (2 * B) + [3] * 2)
        assert len(cache.resumed) == (B + 1) * (epochs - 1)
        assert all(depth == 1 for depth in cache.resumed)
        assert {key[1] for key in cache.prefixes} == {(0, 1), (2, 3), EVAL_SPLIT}
        # the sliced prefixes outlive the fit's inner passes — they are settled
        # with the point's own whole-role pass, which comes after the fit —
        # and a one-point campaign has no one left to keep them for after it
        executor.run_all()
        assert cache.prefixes == {}

    @pytest.mark.parametrize("kind", ["das", "dbm"])
    def test_resuming_changes_no_weight(self, bundle: ModelBundle, kind: str) -> None:
        """The resumed residual is the leaf block 1 would have received, so
        the fitted weights and the eval scores are bit-identical to a fit
        that re-runs the prefix every step."""
        raw = _train_doc(kind)
        _docs, (handle,), cache = _campaign([raw])
        executor = _executor(raw, bundle, interning=handle)
        resumed = run_training(executor.doc, executor, _train_request())
        assert cache.resumed, "the fit never resumed — the test measures nothing"
        executor = _executor(raw, bundle, interning=None)
        plain = run_training(executor.doc, executor, _train_request())
        a, b = _weights(resumed), _weights(plain)
        assert set(a) == set(b) == ({"rot.weight"} if kind == "das" else {"gate.theta"})
        for name in a:
            torch.testing.assert_close(a[name], b[name], atol=0.0, rtol=0.0)
        assert resumed.eval_score is not None and plain.eval_score is not None
        assert resumed.eval_score.metrics == plain.eval_score.metrics
        assert resumed.eval_score.passes == plain.eval_score.passes


@pytest.mark.unit
class TestHybridMoe:
    """``tiny-random/qwen3.5-moe``: three Gated DeltaNet blocks under one
    full-attention block, a routed MoE in every block."""

    @pytest.mark.parametrize(
        ("component", "layer"), [("block_output", 3), ("block_input", 2)]
    )
    def test_blocks_below_the_write_are_skipped(
        self, moe_bundle: ModelBundle, component: str, layer: int
    ) -> None:
        raw_a, raw_b = _scan(layer=layer, key=TINY_QWEN35_MOE, component=component)
        _docs, (first, second), cache = _campaign([raw_a, raw_b])
        plain_a = _reads(_executor(raw_a, moe_bundle, interning=None))
        plain_b = _reads(_executor(raw_b, moe_bundle, interning=None))
        _assert_same_reads(
            _reads(_executor(raw_a, moe_bundle, interning=first)), plain_a
        )
        with _block_fires(moe_bundle) as fires:
            _assert_same_reads(
                _reads(_executor(raw_b, moe_bundle, interning=second)), plain_b
            )
        for below in range(layer):
            assert fires[below] == [], f"block {below} ran under a resume at {layer}"
        for above in range(layer, len(moe_bundle.blocks)):
            assert fires[above] == [4]
        assert cache.resumed == [layer]

    def test_the_fixture_resumes_by_its_resolved_family(
        self, moe_bundle: ModelBundle
    ) -> None:
        """The allow-list holds the *text* config's ``model_type`` — the one
        ``ModelInfo.family`` resolves to. The fixture's config is text-topped
        (a ``Qwen3_5MoeTextConfig`` with no ``text_config``), so wrapper and
        text spellings coincide on it; the test below covers the wrapper."""
        assert not hasattr(moe_bundle.model.config, "text_config")
        assert moe_bundle.info.family == "qwen3_5_moe_text"
        assert moe_bundle.info.family in executor_module._RESUMABLE_MODEL_TYPES
        assert executor_module._resumable(moe_bundle)

    def test_a_wrapper_topped_config_still_resumes(
        self, moe_bundle: ModelBundle, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A real ``Qwen/Qwen3.6-35B-A3B`` checkpoint tops its config with a
        ``Qwen3_5MoeConfig`` whose ``model_type`` is ``qwen3_5_moe`` and whose
        ``text_config.model_type`` is ``qwen3_5_moe_text``. The family is read
        the way the registry reads it — off the text config — so the wrapper
        spelling on the loaded model's config does not switch resume off."""
        text = moe_bundle.model.config
        wrapper = types.SimpleNamespace(model_type="qwen3_5_moe", text_config=text)
        info = model_info_from_hf_config(moe_bundle.key, wrapper)
        assert info.family == "qwen3_5_moe_text"
        wrapped = dataclasses.replace(moe_bundle, info=info)
        # the model object's own config now carries the wrapper spelling too
        monkeypatch.setattr(text, "model_type", "qwen3_5_moe")
        assert executor_module._resumable(wrapped)
        raw_a, raw_b = _scan(layer=3, key=TINY_QWEN35_MOE)
        _docs, (first, second), cache = _campaign([raw_a, raw_b])
        plain_b = _reads(_executor(raw_b, wrapped, interning=None))
        _reads(_executor(raw_a, wrapped, interning=first))
        with _block_fires(wrapped) as fires:
            _assert_same_reads(
                _reads(_executor(raw_b, wrapped, interning=second)), plain_b
            )
        assert fires[:3] == [[], [], []] and fires[3] == [4]
        assert cache.resumed == [3]

    def test_stand_ins_expose_the_replaced_blocks_attributes(
        self, moe_bundle: ModelBundle
    ) -> None:
        """A decoder loop may read per-layer metadata off the layer object
        while building its call (``decoder_layer.attention_type`` and the
        like). The stand-ins fall through to the block they replace for every
        attribute they lack, while registering it neither as a child module
        nor in their state."""
        block = moe_bundle.blocks[0]
        value = torch.zeros(1)
        for stand_in in (
            executor_module._CachedResidual(value, block),
            executor_module._Passthrough(block),
        ):
            assert stand_in.block_type == block.block_type == "linear_attention"
            assert stand_in.input_layernorm is block.input_layernorm
            assert list(stand_in.children()) == []
            assert stand_in.state_dict() == {}
            with pytest.raises(AttributeError):
                stand_in.no_such_attribute
        assert executor_module._CachedResidual(value, block).value is value

    def test_a_layer_scan_resumes_from_the_deepest_prefix_below_its_write(
        self, moe_bundle: ModelBundle
    ) -> None:
        """One point per layer ``0..n-1`` (corpus 07's layer axis): no point
        ever finds a prefix at exactly its own depth, since the point before
        it could store at most its *own* write depth. Each point therefore
        starts at the deepest stored depth below its write — the previous
        point's — and stores its own on the way for the next. The layer-``L``
        point runs blocks ``>= L-1`` only, and ``blocks_skipped`` is
        ``sum(L-1 for L >= 2)``.

        Four blocks are the least that show it: on the two-block Llama the
        layer-1 point is the first that could store anything and nothing
        follows it, so that fixture never resumes in this shape."""
        n = len(moe_bundle.blocks)
        raws = [swap_doc(key=TINY_QWEN35_MOE, layer=layer) for layer in range(n)]
        _docs, handles, cache = _campaign(raws)
        plain = [_reads(_executor(raw, moe_bundle, interning=None)) for raw in raws]
        fired: list[list[list[int]]] = []
        for raw, handle, reference in zip(raws, handles, plain):
            with _block_fires(moe_bundle) as fires:
                _assert_same_reads(
                    _reads(_executor(raw, moe_bundle, interning=handle)), reference
                )
            fired.append(fires)
        # the layer-0 point's pass is the harvest plus a whole patched forward
        assert fired[0] == [[4, 4]] * n
        # the layer-1 point: no prefix yet (layer 0 stored nothing), whole forward
        assert fired[1] == [[4]] * n
        # every later point starts at the block below its write
        for layer in range(2, n):
            for block in range(n):
                assert fired[layer][block] == ([] if block < layer - 1 else [4]), (
                    f"layer {layer}, block {block}"
                )
        assert cache.resumed == [layer - 1 for layer in range(2, n)]
        assert sum(cache.resumed) == sum(layer - 1 for layer in range(2, n))
        # the last point settled the last sharer of every depth: nothing left
        assert cache.prefixes == {}

    def test_a_prefix_dies_with_its_last_sharer(self, moe_bundle: ModelBundle) -> None:
        """The same scan run deepest-first. The layer-3 point runs whole and
        stores depths 1, 2, 3; it is the only group that resumes at 3, so
        settling it drops depth 3 at once. Each shallower point then resumes
        from the depth below its write and takes that depth with it — the
        store shrinks by one depth per point and is empty at the end."""
        n = len(moe_bundle.blocks)
        raws = [swap_doc(key=TINY_QWEN35_MOE, layer=layer) for layer in range(n)]
        _docs, handles, cache = _campaign(raws)
        depths_after: list[set[int]] = []
        for layer in reversed(range(n)):
            _reads(_executor(raws[layer], moe_bundle, interning=handles[layer]))
            depths_after.append({key[3] for key in cache.prefixes})
        assert depths_after == [{1, 2}, {1}, set(), set()]
        assert cache.resumed == [2, 1]
        assert all(count == 0 for count in cache.prefix_owed.values())


# --------------------------------------------------------------------------- #
# smoke: a corpus-07 scan through the engine
# --------------------------------------------------------------------------- #

OVERRIDES = {
    "model.key": TINY_LLAMA,
    "sites.target.layers": {"sweep": [0, 1]},
    "positions.tap": {"sweep": [{"index": -1}, {"index": -2}]},
}


@pytest.fixture(scope="module")
def scan_env(tmp_path_factory: pytest.TempPathFactory) -> ResolutionEnv:
    root = tmp_path_factory.mktemp("resume-artifacts")
    shutil.copytree(FIXTURES / "artifacts", root, dirs_exist_ok=True)
    write_rot_fixture(root)
    register_model_key({"model": {"key": TINY_LLAMA, "revision": "main"}})
    return build_env(root)


@pytest.fixture(scope="module")
def scan(scan_env: ResolutionEnv) -> CompiledProtocol:
    return compile_protocol(
        CORPUS_DIR / "07_weekdays_locate_scan_im.json",
        env=scan_env,
        overrides=OVERRIDES,
    )


def _run(
    loaded: CompiledProtocol,
    env: ResolutionEnv,
    out: Path,
    selected: Sequence[int] | None = None,
) -> tuple[CompiledProtocol, RunContext]:
    """The compiled campaign and a run context over a chosen subset of its
    points — the two arguments of ``Engine.execute``."""
    points = None if selected is None else tuple(selected)
    return loaded, RunContext(output_dir=out, env=env, points=points)


def _engine_bundle() -> ModelBundle:
    """The very model object the engine runs — ``load_model`` is memoized on
    this exact call form (see ``test_forward_interning``)."""
    return load_model(TINY_LLAMA, "main", dtype="fp32", device="cpu", quantization=None)


def _expected_scan(
    loaded: CompiledProtocol, env: ResolutionEnv
) -> tuple[list[int], list[int], int]:
    """Walk the campaign's points in order and predict, per block, how many
    forwards run it. The shared harvest runs once (§3, every block). A
    patched forward at layer 0 runs every block and may store nothing (its
    residual entering block 1 is post-write); the first at layer 1 runs every
    block and stores the prefix; every later one at layer 1 resumes."""
    block0, block1, skipped = 1, 1, 0
    stored = False
    for point in steps_of(loaded, env).points:
        layer = point.coords["sites.target.layers"]
        if layer == 0:
            block0 += 1
            block1 += 1
        elif stored:
            block1 += 1
            skipped += 1
        else:
            block0 += 1
            block1 += 1
            stored = True
    return [block0], [block1], skipped


@pytest.mark.smoke
def test_the_scan_skips_the_prefix_below_every_later_write(
    scan: CompiledProtocol, scan_env: ResolutionEnv, tmp_path: Path
) -> None:
    bundle = _engine_bundle()
    (block0,), (block1,), skipped = _expected_scan(scan, scan_env)
    with _block_fires(bundle) as fires:
        result = PytorchHooksEngine().execute(*_run(scan, scan_env, tmp_path))
    assert len(fires[0]) == block0
    assert len(fires[1]) == block1
    assert skipped > 0, "the tiny scan never resumed — the test measures nothing"
    tallies = [s["prefix_reuse"] for s in result.summaries if "prefix_reuse" in s]
    assert sum(t["blocks_skipped"] for t in tallies) == skipped
    assert sum(t["resumed"] for t in tallies) == skipped  # every resume was at 1
    # the model-level forward count is untouched by design
    assert result.forwards == len(fires[1])


@pytest.mark.smoke
def test_the_store_is_empty_when_the_run_returns(
    scan: CompiledProtocol,
    scan_env: ResolutionEnv,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every prefix is refcounted by the plan's group instances and dropped
    with its last sharer, so a finished campaign holds none."""
    caches: list[Any] = []

    def spy(docs: Any, plans: Any) -> Any:
        cache = campaign_cache(docs, plans)
        caches.append(cache)
        return cache

    monkeypatch.setattr(execution, "campaign_cache", spy)
    result = PytorchHooksEngine().execute(*_run(scan, scan_env, tmp_path))
    (cache,) = caches
    assert cache.resumed, "nothing resumed — the test measures nothing"
    assert cache.prefixes == {}
    assert all(count == 0 for count in cache.prefix_owed.values())
    # a point that resumed nothing reports nothing; one that did, its tally
    for summary in result.summaries:
        if "prefix_reuse" in summary:
            assert summary["prefix_reuse"]["resumed"] > 0


@pytest.mark.smoke
def test_resuming_changes_no_number(
    scan: CompiledProtocol, scan_env: ResolutionEnv, tmp_path: Path
) -> None:
    """One request per point has nothing to resume against — each shard's
    patched forward is the first on its prefix — so the sharded tables are
    the reference, and the whole run must reproduce them exactly."""
    whole = PytorchHooksEngine().execute(*_run(scan, scan_env, tmp_path / "whole"))
    shards = [
        PytorchHooksEngine().execute(*_run(scan, scan_env, tmp_path / f"point{i}", [i]))
        for i in range(len(steps_of(scan, scan_env).points))
    ]
    assert (
        sum(s.get("prefix_reuse", {}).get("blocks_skipped", 0) for s in whole.summaries)
        > 0
    )
    for shard in shards:
        assert all("prefix_reuse" not in s for s in shard.summaries)
    for name in ("iia.json", "logit_diff.json"):
        resumed = table_frame(tmp_path / "whole" / name)
        sharded = pd.concat(
            [table_frame(tmp_path / f"point{i}" / name) for i in range(len(shards))],
            ignore_index=True,
        )
        pd.testing.assert_frame_equal(resumed, sharded, check_exact=True)
