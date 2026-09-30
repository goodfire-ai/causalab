"""An ``lm_head`` read at named positions runs the head over the rows it
gathered, and the forward runs without the head when nothing else needs it
(``neural/shared/head.py``, ``executor._without_head``).

Held against the **full path** — every ``lm_head`` read tapping the head as
the model runs it — which the tests reach by refusing the projection
(``projects_head`` monkeypatched to ``False``): every read value within a
few ulps of the dtype, every metric table the same to ``1e-4`` relative, a
DAS and a DBM fit with every loss and parameter bit-identical over three
epochs (a gradient never flows through a projection, so training is the
model's own head) and every eval score within ``1e-5``, a cohort fit the
same, the eager and the graph-eligible executor agreeing; the head invoked
once per projecting read over ``[rows, width, d_model]`` and never by the
model on a forward nothing else needs it on; the head kept when a read
spans the whole sequence, when a write lands on it, when the group decodes
or when a gradient flows through it; the campaign store holding ``ln_final``
for a shared positional read; the model's head module back in place after
every forward. A property over left-padding patterns and position indices
closes it.

Tolerance rather than ``torch.equal``: each logit is the same dot product,
but the BLAS may pick another kernel for the gathered ``M = rows`` than for
the full ``M = rows·seq`` and change the reduction order (MKL/oneDNN on x86
does at small M, Accelerate on arm64 does not, cuBLAS does on the tiny
shapes and did not on the A3B's). Where the projection is bit-identical is
the GPU parity script's and the end-to-end zero-digest-diff's claim
(``tests/neural/shared/test_head.py`` docstring), not this file's.
"""

# the loop's internals and the sibling suites' builders are used directly
# pyright: reportPrivateUsage=false

from __future__ import annotations

import contextlib
from typing import Any, Callable, Iterator

import pytest
import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from causalab.neural.engines.pytorch_hooks import cuda_graphs as cuda_graphs_module
from causalab.neural.engines.pytorch_hooks import train as train_module
from causalab.neural.engines.pytorch_hooks.cuda_graphs import (
    GraphExecutor,
    make_executor,
)
from causalab.neural.engines.pytorch_hooks.loading import ModelBundle, load_model
from causalab.neural.shared import head as head_mod
from causalab.neural.shared.executor import RaggedValue, tap_key
from causalab.neural.shared.head import HEAD, HEAD_INPUT, head_module
from causalab.neural.shared.metrics import compute_metric
from causalab.neural.shared.sites import resolve_site
from causalab.protocol.schema import SiteSpec, read_is_vocabulary

from tests.neural.engines.pytorch_hooks._drive import base_data_section, executor_for
from tests.neural.engines.pytorch_hooks.conftest import TINY_LLAMA, TINY_QWEN35_MOE
from tests.neural.engines.pytorch_hooks.test_featurizer_cache_loop import _with_eval
from tests.neural.engines.pytorch_hooks.test_fit_cohort import (
    _assert_same_fit,
    _campaign,
    _fit_cohort,
    _train_doc,
)
from tests.neural.engines.pytorch_hooks.test_rotation_round_trip import _fit, das_doc
from tests.neural.engines.pytorch_hooks.test_train import (
    ANSWERS,
    BASES,
    COUNTERFACTUALS,
    WRONG,
    dbm_doc,
)
from tests.protocol._docs import UNWRITTEN, saved

unit = pytest.mark.unit
prop = pytest.mark.property

FAMILIES = [TINY_LLAMA, TINY_QWEN35_MOE]
COLUMNS = {"label": ANSWERS, "wrong": WRONG}


@contextlib.contextmanager
def full_path(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """The reference: no read projects the head, so every ``lm_head`` read
    taps the head as the model runs it and the forward keeps it."""
    with monkeypatch.context() as patch:
        patch.setattr(head_mod, "projects_head", lambda *args: False)
        yield


@contextlib.contextmanager
def head_calls(bundle: ModelBundle) -> Iterator[list[tuple[tuple[int, ...], int]]]:
    """Every run of the head module — the model's or a read's projection —
    as ``(input shape, output numel)``."""
    seen: list[tuple[tuple[int, ...], int]] = []
    handle = head_module(bundle).register_forward_hook(
        lambda _m, args, out: seen.append((tuple(args[0].shape), int(out.numel())))
    )
    try:
        yield seen
    finally:
        handle.remove()


#: The un-intervened network read on base, beside `UNWRITTEN` on the
#: counterfactual (§2.9's names for a network read un-intervened on both).
ORIGINAL_BASE = "original_base"

#: Every model a document here may read, with its input; ``patched`` lands
#: the swap, and a test declaring ``bumped`` adds its own write.
INPUT_OF = {
    UNWRITTEN: "counterfactual",
    ORIGINAL_BASE: "base",
    "patched": "base",
    "bumped": "base",
}

#: The five distribution aggregations over the patched ``logits`` (§2.10),
#: by the label each one's table carries.
AGGREGATIONS: dict[str, dict[str, Any]] = {
    "ld": {
        "kind": "logit_diff",
        "a": "label",
        "b": "wrong",
    },
    "sa": {
        "kind": "soft_accuracy",
        "a": "label",
        "b": "wrong",
    },
    "ce": {"kind": "cross_entropy", "target": "label"},
    # a save holds a read reference in the object form (§2.7)
    "kl": {"kind": "kl", "target": {"read": "clean", "model": ORIGINAL_BASE}},
    "tk": {"kind": "top_k", "k": 3, "by": "prob"},
}


def _head_read(pos: Any, **extra: Any) -> dict:
    """The address of a read at the head (§2.7); the model that takes it is
    the caller's to name."""
    return {"site": "head", "pos": pos, **extra}


def _doc(
    reads: dict[str, tuple[str, dict[str, Any]]], *, metrics: bool = True
) -> dict[str, Any]:
    """A swap at block 0 read at the head: ``logits`` (patched) and ``clean``
    (the un-intervened network on base) at the last token, plus ``reads`` —
    ``name: (model, address)``, every model listing the reads taken on it;
    the five distribution aggregations over ``logits`` when ``metrics``."""
    all_reads: dict[str, tuple[str, dict[str, Any]]] = {
        "v_cf": (UNWRITTEN, {"site": "tgt", "pos": {"index": -1}}),
        "logits": ("patched", _head_read({"index": -1})),
        "clean": (ORIGINAL_BASE, _head_read({"index": -1})),
        **reads,
    }
    models: dict[str, dict[str, Any]] = {}
    for name, (model, _) in all_reads.items():
        models.setdefault(model, {"input": INPUT_OF[model], "reads": []})
        models[model]["reads"].append(name)
    models["patched"]["writes"] = ["patch"]
    method: dict[str, Any] = {
        "intervened_models": models,
        "sites": {
            "tgt": {"component": "block_output", "layers": [0]},
            "head": {"component": "lm_head"},
        },
        "reads": {name: address for name, (_, address) in all_reads.items()},
        "writes": {
            "patch": {"site": "tgt", "pos": {"index": -1}, "do": {"swap": "v_cf"}}
        },
        "save": [
            saved(name, model, f"{name}.safetensors")
            for name, (model, _) in all_reads.items()
            if name != "v_cf"
        ],
    }
    if metrics:
        method["save"] += [
            saved("logits", "patched", f"{label}.json", dict(spec))
            for label, spec in AGGREGATIONS.items()
        ]
    return {
        "header": {"protocol_version": "4"},
        "model": {"key": TINY_LLAMA, "revision": "main"},
        "data": base_data_section(with_counterfactual=True),
        "method": method,
    }


def _executor(raw: dict[str, Any], bundle: ModelBundle, **kwargs: Any) -> Any:
    return executor_for(
        raw,
        bundle,
        base_texts=BASES,
        counterfactual_texts=COUNTERFACTUALS,
        extra_columns=COLUMNS,
        **kwargs,
    )


def _values(executor: Any) -> dict[str, torch.Tensor | RaggedValue]:
    return {name: executor.read_value(name) for name in executor.doc.reads}


def _assert_close_values(got: dict[str, Any], want: dict[str, Any]) -> None:
    """Every read value within a few ulps of its dtype (module docstring)."""
    assert got.keys() == want.keys()
    for name in want:
        a, b = got[name], want[name]
        if isinstance(a, RaggedValue):
            assert isinstance(b, RaggedValue) and a.widths == b.widths, name
            a, b = a.flat, b.flat
        assert isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor), name
        a, b = a.detach(), b.detach()
        eps = torch.finfo(b.dtype).eps
        scale = float(b.abs().max()) if b.numel() else 0.0
        torch.testing.assert_close(a, b, rtol=4 * eps, atol=4 * eps * scale, msg=name)


def _assert_close_tables(got: Any, want: Any, where: str = "") -> None:
    """Metric tables entry for entry: floats to ``1e-4`` relative (they are
    reductions of logits that may differ by ulps), everything else exact."""
    if isinstance(got, float) or isinstance(want, float):
        assert got == pytest.approx(want, rel=1e-4, abs=1e-6), where
    elif isinstance(want, dict):
        assert isinstance(got, dict) and got.keys() == want.keys(), where
        for key in want:
            _assert_close_tables(got[key], want[key], f"{where}/{key}")
    elif isinstance(want, (list, tuple)):
        assert isinstance(got, (list, tuple)) and len(got) == len(want), where
        for i, (a, b) in enumerate(zip(got, want)):
            _assert_close_tables(a, b, f"{where}[{i}]")
    else:
        assert got == want, where


def _tables(executor: Any) -> dict[str, list[Any]]:
    doc, rows, tokenizer = (
        executor.doc,
        executor.rows_for_metrics(),
        executor.bundle.tokenizer,
    )
    out: dict[str, list[Any]] = {}
    for agg in doc.saved_aggregations():
        target = executor.dense_value(agg.target) if agg.target is not None else None
        out[agg.label] = compute_metric(
            agg.spec,
            executor.dense_value(agg.read),
            rows,
            tokenizer,
            target_value=target,
            vocab_axis=read_is_vocabulary(doc, agg.read.read),
        )
    return out


@pytest.fixture(scope="module", params=FAMILIES)
def bundle(request: pytest.FixtureRequest) -> ModelBundle:
    return load_model(request.param, device="cpu")


@pytest.fixture(scope="module")
def llama() -> ModelBundle:
    return load_model(TINY_LLAMA, device="cpu")


class TestTheProjection:
    @unit
    def test_reads_and_metrics_equal_the_full_path_and_the_model_never_runs_the_head(
        self, bundle: ModelBundle, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Positional reads at the last token, at content position 1 (a
        different padded column per row) and through ``dims``: every value
        within ulps of the full path's, every metric table the same. The head
        ran exactly once per projecting read, over ``[rows, 1, d_model]``,
        and never over the sequence; the biggest logits tensor of the run
        is ``rows × vocab`` where the full path built ``rows × seq × vocab``."""
        raw = _doc(
            {
                "early": (ORIGINAL_BASE, _head_read({"index": 1})),
                "sliced": ("patched", _head_read({"index": -1}, dims=[0, 1, 2, 3])),
            }
        )
        with head_calls(bundle) as projected:
            executor = _executor(raw, bundle)
            got_values, got_tables = _values(executor), _tables(executor)
        with full_path(monkeypatch), head_calls(bundle) as full:
            reference = _executor(raw, bundle)
            want_values, want_tables = _values(reference), _tables(reference)
        _assert_close_values(got_values, want_values)
        _assert_close_tables(got_tables, want_tables)
        rows = len(BASES)
        vocab = int(head_module(bundle).weight.shape[0])
        seq = int(executor.frame("base").padded_len)
        assert seq > 1
        # four projecting reads over two groups, each its own head call
        assert len(projected) == 4
        assert all(shape[:2] == (rows, 1) for shape, _ in projected), projected
        assert max(numel for _, numel in projected) == rows * vocab
        # the full path: the model ran the head over every position of every
        # forward that read it — the two groups tapping it, none at width 1
        # (the counterfactual forward taps nothing at the head and runs
        # without it in both runs: the elision is the forward's, not the
        # read's)
        assert [shape[1] for shape, _ in full] == [seq] * 2
        assert max(numel for _, numel in full) == rows * seq * vocab

    @unit
    def test_a_whole_sequence_read_keeps_the_head_beside_a_projected_one(
        self, llama: ModelBundle, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        raw = _doc({"whole": ("patched", _head_read("all"))}, metrics=False)
        with head_calls(llama) as calls:
            executor = _executor(raw, llama)
            got = _values(executor)
        with full_path(monkeypatch):
            want = _values(_executor(raw, llama))
        _assert_close_values(got, want)
        seq = int(executor.frame("base").padded_len)
        # the patched group ran the model's head (the whole-sequence read);
        # the original group did not and projected `clean` itself
        assert sorted(shape[1] for shape, _ in calls) == [1, 1, seq]
        # and the two reads of one group agree: the projected last token is
        # the whole-sequence read's last column
        whole = got["whole"]
        assert isinstance(whole, RaggedValue)
        last = torch.stack([w[-1] for w in torch.split(whole.flat, list(whole.widths))])
        logits = got["logits"]
        assert isinstance(logits, torch.Tensor)
        _assert_close_values({"last": logits[:, 0, :]}, {"last": last})

    @unit
    def test_a_write_at_the_head_keeps_the_head_in_that_model(
        self, llama: ModelBundle, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        raw = _doc(
            {
                "head_cf": (UNWRITTEN, _head_read({"index": -1})),
                "after_bump": ("bumped", _head_read({"index": -1})),
            },
            metrics=False,
        )
        raw["method"]["writes"]["swap_head"] = {
            "site": "head",
            "pos": {"index": -1},
            "do": {"swap": "head_cf"},
        }
        raw["method"]["intervened_models"]["bumped"]["writes"] = ["swap_head"]
        with head_calls(llama) as calls:
            executor = _executor(raw, llama)
            got = _values(executor)
        with full_path(monkeypatch):
            want = _values(_executor(raw, llama))
        _assert_close_values(got, want)
        seq = int(executor.frame("base").padded_len)
        # exactly one forward ran the model's head: the one written at it
        assert sorted(shape[1] for shape, _ in calls) == [1, 1, 1, seq]
        # and the write landed: the bumped read is the counterfactual's head
        _assert_close_values({"bumped": got["after_bump"]}, {"bumped": got["head_cf"]})

    @unit
    def test_a_decoding_group_keeps_the_head_for_its_prompt_reads(
        self, llama: ModelBundle, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        raw = _doc(
            {
                "gen": (
                    ORIGINAL_BASE,
                    _head_read({"index": 0, "generated": {"max_new_tokens": 2}}),
                )
            },
            metrics=False,
        )
        with head_calls(llama) as calls:
            executor = _executor(raw, llama)
            got = _values(executor)
        with full_path(monkeypatch):
            want = _values(_executor(raw, llama))
        _assert_close_values(got, want)
        seq = int(executor.frame("base").padded_len)
        # the original group decodes: its prefill ran the head over the
        # prompt and each decode step over one token; only the patched
        # group's `logits` projected
        assert seq in [shape[1] for shape, _ in calls]
        assert [shape[:2] for shape, _ in calls].count((len(BASES), 1)) >= 1

    @unit
    def test_the_head_module_is_back_after_every_forward(
        self, llama: ModelBundle
    ) -> None:
        head = head_module(llama)
        enc = llama.tokenizer(BASES, return_tensors="pt", padding=True)
        with torch.no_grad():
            before = llama.model(**enc).logits
        executor = _executor(_doc({}, metrics=False), llama)
        executor.run_all()
        assert head_module(llama) is head
        assert isinstance(head, torch.nn.Linear)
        with torch.no_grad():
            after = llama.model(**enc).logits
        assert torch.equal(before, after)

    @unit
    def test_a_shared_pass_stores_the_heads_input_not_the_vocabulary(
        self, llama: ModelBundle
    ) -> None:
        """Two points sharing ``original`` on ``base`` with a last-token head
        read: the first pass stores ``ln_final`` under the shared digest and
        no ``lm_head`` capture; the second point is served from it, its
        value equal to the first's, and runs no forward for the group."""
        raws = [
            _doc({}, metrics=False),
            _doc({"extra": (ORIGINAL_BASE, _head_read({"index": -1}))}, metrics=False),
        ]
        _docs, handles = _campaign(raws)
        store = handles[0].cache
        first = _executor(raws[0], llama, interning=handles[0])
        second = _executor(raws[1], llama, interning=handles[1])
        clean = first.read_value("clean")
        key = handles[0].keys[(ORIGINAL_BASE, "base")]
        captured = store.captured[key]
        norm_key = tap_key(resolve_site(llama, SiteSpec(component=HEAD_INPUT)))
        head_key = tap_key(resolve_site(llama, SiteSpec(component=HEAD)))
        assert norm_key in captured and head_key not in captured
        assert captured[norm_key].shape[-1] == head_module(llama).weight.shape[1]
        executed = len(store.executed)
        assert torch.equal(second.read_value("clean"), clean)
        assert torch.equal(second.read_value("extra"), clean)
        assert len(store.executed) == executed  # served, not run

    @unit
    def test_the_graph_eligible_executor_agrees_with_the_eager_one(
        self, llama: ModelBundle, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``GraphExecutor`` gathers the last column by a slice rather than
        an index; its projected read is the eager executor's to the bit."""
        monkeypatch.setattr(
            cuda_graphs_module,
            "unsupported_reason",
            lambda doc, bundle, collective: None,
        )
        raw = _doc({}, metrics=False)
        eager = _executor(raw, llama)
        reference = executor_for(
            raw,
            llama,
            base_texts=BASES,
            counterfactual_texts=COUNTERFACTUALS,
            extra_columns=COLUMNS,
        )
        graphed = make_executor(
            reference.doc,
            llama,
            cuda_graphs=True,
            role_rows=reference.role_rows,
            role_fields=reference.role_fields,
            load_tensors=reference.load_tensors,
        )
        assert isinstance(graphed, GraphExecutor)
        with head_calls(llama) as calls:
            got = _values(graphed)
        _assert_close_values(got, _values(eager))
        assert all(shape[1] == 1 for shape, _ in calls)


def _record_fit(
    doc: dict[str, Any], monkeypatch: pytest.MonkeyPatch, *, projected: bool
) -> dict[str, Any]:
    """Everything the loop computes that a moved logit would move: the loss
    of every update, every eval score, the selected fit, the parameters."""
    losses: list[float] = []
    scores: list[dict[str, float]] = []
    real_loss, real_score = train_module._loss, train_module._score

    def recording_loss(*args: Any, **kwargs: Any) -> torch.Tensor:
        loss = real_loss(*args, **kwargs)
        losses.append(float(loss.detach()))
        return loss

    def recording_score(*args: Any, **kwargs: Any) -> dict[str, float]:
        score = real_score(*args, **kwargs)
        scores.append(dict(score))
        return score

    with monkeypatch.context() as patch:
        patch.setattr(train_module, "_loss", recording_loss)
        patch.setattr(train_module, "_score", recording_score)
        if not projected:
            patch.setattr(head_mod, "projects_head", lambda *args: False)
        outcome = _fit(doc)
    assert outcome.eval_score is not None
    return {
        "losses": losses,
        "scores": scores,
        "selected": (outcome.eval_score.selected, dict(outcome.eval_score.metrics)),
        "state": {
            name: {k: v.detach().clone() for k, v in stage.state_dict().items()}
            for name, stage in outcome.stages.items()
        },
    }


class TestFits:
    @unit
    @pytest.mark.parametrize(
        "build",
        [
            pytest.param(lambda: _with_eval(das_doc(epochs=3)), id="das-cayley"),
            pytest.param(lambda: _with_eval(dbm_doc()), id="dbm-gate"),
        ],
    )
    def test_a_fit_is_bit_identical_with_and_without_the_projection(
        self, build: Callable[[], dict], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Three epochs of updates, an eval after each: every loss and every
        parameter is the full path's to the bit, every score within ``1e-5``. The
        training steps read the head **as the model runs it** — a gradient
        flows through it, and the head's backward GEMM at another ``M`` is
        another cuBLAS problem (``shared/head.py``) — while every eval pass
        projects: the head calls of the run are the trained group's at the
        full width and the eval reads' at width 1, nothing else."""
        # the bundle `_fit` loads — `load_model` caches by its arguments, so
        # the hook has to sit on the same object (its default device is the CPU)
        llama = load_model(TINY_LLAMA)
        with head_calls(llama) as calls:
            got = _record_fit(build(), monkeypatch, projected=True)
        want = _record_fit(build(), monkeypatch, projected=False)
        assert len(want["losses"]) >= 6 and len(want["scores"]) >= 3
        # the training steps never project: every loss is the model's, to the bit
        assert got["losses"] == want["losses"]
        # the eval passes project: within ulps, hence the scores within 1e-5
        assert len(got["scores"]) == len(want["scores"])
        for a, b in zip(got["scores"], want["scores"], strict=True):
            assert a.keys() == b.keys()
            for name in b:
                assert a[name] == pytest.approx(b[name], rel=1e-5), name
        assert got["selected"][0] == want["selected"][0]
        for name, value in want["selected"][1].items():
            assert got["selected"][1][name] == pytest.approx(value, rel=1e-5), name
        assert got["state"].keys() == want["state"].keys()
        for name in want["state"]:
            for key, value in want["state"][name].items():
                assert torch.equal(value, got["state"][name][key]), (name, key)
        widths = sorted({shape[1] for shape, _ in calls})
        assert len(widths) == 2 and widths[0] == 1, widths
        # the eval passes projected (width 1); the training steps ran the
        # model's head over the frame (the padded width, > 1)
        assert sum(shape[1] == 1 for shape, _ in calls) >= len(want["scores"])
        assert sum(shape[1] > 1 for shape, _ in calls) >= len(want["losses"])

    @unit
    def test_a_gradient_keeps_the_head_and_a_constant_group_beside_it_projects(
        self, llama: ModelBundle, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """On one grad-enabled executor: the trained model's head read taps
        the head (its gradient must be the model's), the fit-constant
        ``original`` group's head read projects (it carries no graph — the
        network is frozen), and their values are the full path's to the
        bit. ``CAUSALAB_PROJECT_HEAD_UNDER_GRAD=1`` projects the trained
        read too."""
        raw = das_doc(epochs=1)
        raw["method"]["reads"]["clean"] = {
            **_head_read({"index": -1}),
            "site": "lm_head",  # das_doc's name for the head site
        }
        raw["method"]["intervened_models"][ORIGINAL_BASE] = {
            "input": "base",
            "reads": ["clean"],
        }
        raw["method"]["save"].append(
            saved(
                "logits",
                "patched",
                "kl.json",
                {"kind": "kl", "target": {"read": "clean", "model": ORIGINAL_BASE}},
            )
        )
        reference = _executor(raw, llama, grad_enabled=True)
        # the un-intervened network, once per input it is read on
        assert reference.fit_constant_models == {ORIGINAL_BASE, UNWRITTEN}
        taps = reference._read_taps(
            "patched", "base", [("logits", reference.doc.reads["logits"])]
        )
        assert (
            taps["logits"].capture.component == HEAD and taps["logits"].project is None
        )
        taps = reference._read_taps(
            ORIGINAL_BASE, "base", [("clean", reference.doc.reads["clean"])]
        )
        assert (
            taps["clean"].capture.component == HEAD_INPUT
            and taps["clean"].project is not None
        )
        with head_calls(llama) as calls:
            got = {name: reference.read_value(name) for name in ("logits", "clean")}
        seq = int(reference.frame("base").padded_len)
        assert sorted(shape[1] for shape, _ in calls) == [1, seq]
        assert got["logits"].requires_grad and not got["clean"].requires_grad
        with full_path(monkeypatch):
            want_ex = _executor(raw, llama, grad_enabled=True)
            want = {name: want_ex.read_value(name) for name in ("logits", "clean")}
        # the trained read is the model's head on both sides: exact; the
        # constant group's projected read within ulps
        assert torch.equal(got["logits"].detach(), want["logits"].detach())
        _assert_close_values({"clean": got["clean"]}, {"clean": want["clean"]})
        monkeypatch.setenv(head_mod.ENV_PROJECT_UNDER_GRAD, "1")
        knobbed = _executor(raw, llama, grad_enabled=True)
        taps = knobbed._read_taps(
            "patched", "base", [("logits", knobbed.doc.reads["logits"])]
        )
        assert taps["logits"].capture.component == HEAD_INPUT
        with head_calls(llama) as calls:
            projected = knobbed.read_value("logits")
        assert [shape[1] for shape, _ in calls] == [1]
        assert projected.requires_grad
        _assert_close_values({"logits": projected}, {"logits": want["logits"]})

    @unit
    def test_a_cohort_fit_is_bit_identical_with_and_without_the_projection(
        self, llama: ModelBundle, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        raws = [_train_doc(seed=0, k=4), _train_doc(seed=1, k=2)]
        got, _e, sizes, _c = _fit_cohort(raws, llama)
        with full_path(monkeypatch):
            want, _e, want_sizes, _c = _fit_cohort(raws, llama)
        assert sizes == want_sizes
        for a, b in zip(got, want, strict=True):
            # weights exact (training never projects); eval metrics to 1e-4
            _assert_same_fit(a, b, atol=0.0, rtol=0.0)


#: Texts of three to seven tokens on the tiny Llama tokenizer, so a content
#: index up to 2 from either end resolves on every row.
POOL = [
    "the quick brown fox jumps over",
    "a slow green turtle sleeps",
    "every shiny robot dances tonight in",
    "some ancient rivers flow",
    "cold silver mountains echo loudly at",
    "seven broken clocks",
    "warm quiet valleys rest gently",
]


def _both_paths(rows: list[int], index: int) -> tuple[dict[str, Any], dict[str, Any]]:
    """The read values of a document reading the head at content ``index``
    over ``POOL``'s rows ``rows``, projected and on the full path."""
    bundle = load_model(TINY_LLAMA, device="cpu")
    raw = _doc({"at": (ORIGINAL_BASE, _head_read({"index": index}))}, metrics=False)
    texts = [POOL[i] for i in rows]
    counterfactuals = [POOL[(i + 1) % len(POOL)] for i in rows]

    def build() -> Any:
        return executor_for(
            raw, bundle, base_texts=texts, counterfactual_texts=counterfactuals
        )

    got = _values(build())
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(head_mod, "projects_head", lambda *args: False)
        want = _values(build())
    return got, want


@prop
@settings(max_examples=12, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    rows=st.lists(st.sampled_from(range(len(POOL))), min_size=1, max_size=4),
    index=st.sampled_from([-3, -2, -1, 0, 1, 2]),
)
def test_any_padding_pattern_and_position_equals_the_full_path(
    rows: list[int], index: int
) -> None:
    """Random rows of different lengths — a different left padding per row,
    hence a different gathered column per row — read at a random content
    index: the projected read is the full path's within ulps (a one-row
    document is the ``M = 1`` GEMV case on MKL)."""
    got, want = _both_paths(rows, index)
    _assert_close_values(got, want)
