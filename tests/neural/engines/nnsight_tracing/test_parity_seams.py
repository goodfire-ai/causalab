"""Engine parity at the execution seams around a forward: the attention
backend, the caller-owned bundle, the CPU kernel path, the run-time and
pre-forward refusal batteries, and the workflow step kinds.

Each seam is one of the reference engine's existing documents through both
engines — the executor seam via ``tests._helpers.engines.both_executors``, the
engine seam via ``engines.run_both`` and ``compare_run_dirs`` — with the
reference tests' document builders imported rather than copied.

* **P16** — both engines loaded ``sdpa``: a boundary read forwards under sdpa
  on both; an ``attention_scores`` read switches both to eager for the
  interior, agrees with the eager-pinned fixtures, and both restore ``sdpa``.
* **P17** — the corpus interchange on caller-owned bundles
  (``PytorchHooksEngine(bundle=…)`` / ``NnsightEngine(bundle=…)``): the two
  output directories agree and both receipts say ``model_source: caller``.
* **P19** — the simulated CUDA-only DeltaNet kernel globals: both engines run
  the torch path, agree with each other and bit-for-bit with the unguarded run.
* **P20** — every run-time refusal of the snapshot table whose trigger both
  engines can run, run on the other engine: the same message.
* **P21** — the pre-forward battery (``check_scoring``, the artifact entry
  identity): byte-identical refusals, as they should be for ``ExecutorBase``
  methods. The per-step rules of ``causalab/neural/shared/step_rules.py``
  run inside both engines' ``execute`` before any forward: a violation at a
  combination of axis values that no representative reaches is refused with
  the same text by both, and nothing is written.
* **R4** — the workflow step kinds: a ``fan_out`` scan (two shards joined)
  and the ``shuffled_source`` control through ``NnsightEngine()`` agree
  with the reference run per step directory.
* **generation writes** — ``writes_during_generation`` is reference-engine
  only. Routing refuses a flagged document for nnsight before any weights load, and that
  refusal is pinned here. The executor's own backstop is pinned in
  ``test_generate_frame_nnsight.py``.
* **parallelism** — multi-GPU parallelism is reference-engine only. The
  nnsight engine takes no ``sharding`` or ``parallel`` argument. The CLI and
  ``load_engine`` refusals are pinned in ``tests/protocol/test_cli_parallel.py``.

``check_answer_forms`` left with the ``token_form`` retirement: answers
are tokenized as written, so the old answer-form refusal has no counterpart.
"""

from __future__ import annotations

import contextlib
import copy
import importlib
import inspect
import json
import shutil
from pathlib import Path
from typing import Any, Callable, Iterator

import pytest
import torch

from causalab.neural.engines.nnsight_tracing.engine import NnsightEngine
from causalab.neural.engines.nnsight_tracing.loading import (
    load_model as load_trace_model,
)
from causalab.neural.engines.pytorch_hooks.engine import PytorchHooksEngine
from causalab.neural.engines.pytorch_hooks.loading import (
    ModelBundle,
    load_model as load_hooks_model,
)
from causalab.io.env import FileArtifacts, FileDatasets, ResolutionEnv
from causalab.io.tensor_files import TensorBundle
from causalab.neural.shared.kernels import KERNEL_GLOBALS, torch_implementation
from causalab.protocol import RUN_RECORD_NAME, run_protocol
from causalab.protocol.pipeline import route_engine, validate
from causalab.protocol.registry.engines import ENGINE_VERBS
from causalab.protocol.rules.errors import ValidationError, ValidationErrors
from causalab.tasks import TASKS_ROOT
from causalab.workflow import run_workflow
from causalab.workflow.document import load_workflow

from tests._helpers import engines
from tests._helpers import refusal_snapshot as table
from tests._helpers.a3b_sweep import assert_same, interchange_doc, read_doc
from tests.neural.engines.nnsight_tracing.conftest import TINY_LLAMA, TINY_QWEN35_MOE
from tests.neural.engines.nnsight_tracing.test_parity_module_boundaries import (
    BASE_TEXTS,
    CF_TEXTS,
)
from tests.neural.engines.pytorch_hooks.test_alignment_run import (
    PROMPTS,
    SPACED_ANSWERS,
    _match_doc,
)
from tests.neural.engines.pytorch_hooks.test_shuffled_source_run import (
    _document as shuffled_source_document,
)
from tests.protocol._docs import in_order
from tests.protocol._env import CORPUS_DIR, FIXTURES
from tests.protocol.test_refusal_snapshot import ENTRIES, check_entry

pytestmark = pytest.mark.smoke

#: the qwen fixture's one full-attention layer (0–2 are Gated DeltaNet)
FULL_ATTENTION_LAYER = 3
DELTANET_LAYER = 0

DOCUMENT = CORPUS_DIR / "02_interchange_im.json"
#: tiny-random is two layers deep, so the shipped L18 site is retargeted
OVERRIDES = {"model.key": TINY_LLAMA, "sites.target.layers": 1}


def _refusal_text(run: Callable[[], Any]) -> str:
    """``type: message`` of the exception ``run`` raises — the comparison
    unit for refusal parity (a refusal that changes class is a change)."""
    with pytest.raises(Exception) as excinfo:
        run()
    return f"{type(excinfo.value).__name__}: {excinfo.value}"


# --------------------------------------------------------------------------- #
# P16 — the attention backend
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def hooks_qwen_sdpa() -> ModelBundle:
    """The reference engine loaded ``sdpa``. This is the module's one extra
    model load: ``load_model`` keys its cache on ``attn_implementation``, so
    the eager session bundle cannot stand in for it (the nnsight side has the
    session fixture ``trace_qwen_default_impl``, whose checkpoint default is
    sdpa). Loaded once here, shared by the two tests below."""
    bundle = load_hooks_model(TINY_QWEN35_MOE, attn_implementation="sdpa")
    assert bundle.model.config._attn_implementation == "sdpa"
    return bundle


@contextlib.contextmanager
def _observed(model: torch.nn.Module) -> Iterator[list[str]]:
    """The attention implementation in force at each forward of ``model``,
    read by a pre-hook the test owns (``test_attention_backends.py``'s shape)."""
    seen: list[str] = []
    handle = model.register_forward_pre_hook(
        lambda module, args: seen.append(module.config._attn_implementation)
    )
    try:
        yield seen
    finally:
        handle.remove()


def _torch_module(bundle: Any) -> torch.nn.Module:
    """The plain torch module behind either bundle (an nnsight envoy wraps
    it as ``_module``). An nnsight model is dispatched first: until its first
    trace the envoy wraps a meta-device skeleton that dispatch *replaces*, so
    a hook registered before it would never fire (the executor dispatches
    before switching backends for the same reason)."""
    if not getattr(bundle.model, "dispatched", True):
        bundle.model.dispatch()
    module = getattr(bundle.model, "_module", bundle.model)
    assert isinstance(module, torch.nn.Module)
    return module


def test_a_boundary_read_under_sdpa_switches_nothing_on_either_engine(
    hooks_qwen_sdpa, trace_qwen_default_impl
):
    """(a) a module-boundary read: both forwards run under the selected
    backend, neither executor applies ``attn_eager``, both agree."""
    doc = read_doc("block_output", FULL_ATTENTION_LAYER)
    hooks, trace = engines.both_executors(
        doc, hooks_qwen_sdpa, trace_qwen_default_impl, base_texts=BASE_TEXTS
    )
    with _observed(_torch_module(hooks_qwen_sdpa)) as hooks_seen:
        hooked = hooks.read_value("r")
    with _observed(_torch_module(trace_qwen_default_impl)) as trace_seen:
        traced = trace.read_value("r")
    assert hooks_seen == ["sdpa"] and trace_seen == ["sdpa"]
    assert hooks.applied_requirements == trace.applied_requirements == set()
    assert_same(hooked, traced, "block_output read under sdpa")
    assert hooks_qwen_sdpa.model.config._attn_implementation == "sdpa"
    assert trace_qwen_default_impl.model.config._attn_implementation == "sdpa"


def test_an_interior_read_under_sdpa_switches_both_to_eager_and_restores(
    hooks_qwen_sdpa, trace_qwen_default_impl, hooks_qwen, trace_qwen
):
    """(b) ``attention_scores`` exists only inside the eager function: both
    engines switch the forward to eager, stamp ``attn_eager``, agree with each
    other and with the eager-pinned fixtures, and put ``sdpa`` back."""
    doc = read_doc("attention_scores", FULL_ATTENTION_LAYER, pos="all")
    hooks, trace = engines.both_executors(
        doc, hooks_qwen_sdpa, trace_qwen_default_impl, base_texts=BASE_TEXTS
    )
    with _observed(_torch_module(hooks_qwen_sdpa)) as hooks_seen:
        hooked = hooks.read_value("r")
    with _observed(_torch_module(trace_qwen_default_impl)) as trace_seen:
        traced = trace.read_value("r")
    assert hooks_seen == ["eager"] and trace_seen == ["eager"]
    assert hooks.applied_requirements == trace.applied_requirements == {"attn_eager"}
    assert hooked.dim() == 4  # (batch, head, query, key)
    assert_same(hooked, traced, "attention_scores through the runtime switch")
    # restored on both: the selection outlives the interior forward
    assert hooks_qwen_sdpa.model.config._attn_implementation == "sdpa"
    assert trace_qwen_default_impl.model.config._attn_implementation == "sdpa"
    # and the switched forward computed what the pinned-eager fixtures compute
    pinned_hooks, pinned_trace = engines.both_executors(
        doc, hooks_qwen, trace_qwen, base_texts=BASE_TEXTS
    )
    assert_same(hooked, pinned_hooks.read_value("r"), "switched vs pinned (hooks)")
    assert_same(traced, pinned_trace.read_value("r"), "switched vs pinned (nnsight)")


# --------------------------------------------------------------------------- #
# P17 — the caller-owned bundle at the engine seam
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def caller_runs(
    hooks_llama, trace_llama, tmp_path_factory: pytest.TempPathFactory
) -> engines.BothRuns:
    """The corpus interchange through each engine on the bundle handed in,
    never loaded (reference: ``test_caller_owned_model.py``,
    ``test_caller_owned_bundle.py``)."""
    base = tmp_path_factory.mktemp("caller")
    env = engines.corpus_env(base / "artifacts")
    hooks_before = load_hooks_model.cache_info()
    trace_before = load_trace_model.cache_info()
    runs = engines.run_both(
        DOCUMENT,
        env,
        base / "out",
        overrides=OVERRIDES,
        hooks_engine=PytorchHooksEngine(bundle=hooks_llama),
        trace_engine=NnsightEngine(bundle=trace_llama),
    )
    assert load_hooks_model.cache_info() == hooks_before, "the hooks run loaded a model"
    assert load_trace_model.cache_info() == trace_before, (
        "the nnsight run loaded a model"
    )
    return runs


def test_caller_owned_bundles_produce_the_same_run(caller_runs):
    assert caller_runs.hooks_result.files and caller_runs.trace_result.files
    caller_runs.compare()


def test_both_receipts_say_the_model_was_the_callers(caller_runs):
    for run_dir in (caller_runs.hooks_dir, caller_runs.trace_dir):
        record = json.loads((run_dir / RUN_RECORD_NAME).read_text())
        assert record["execution"]["model_source"] == "caller", run_dir


# --------------------------------------------------------------------------- #
# P19 — the CPU kernel path under simulated CUDA-only kernels
# --------------------------------------------------------------------------- #


def _modeling(bundle: Any) -> Any:
    """The modeling module exporting the DeltaNet kernel globals, found the
    way the guard finds it: a submodule class whose module has them all."""
    for module in _torch_module(bundle).modules():
        modeling = importlib.import_module(type(module).__module__)
        if all(hasattr(modeling, name) for name in KERNEL_GLOBALS):
            return modeling
    raise AssertionError("the fixture has no DeltaNet mixer")


def _cuda_only(torch_fn: Callable[..., Any]) -> Callable[..., Any]:
    """What the ``flash-linear-attention`` extra binds: a global refusing CPU
    tensors, wrapping the torch implementation as ``functools.wraps`` would
    (``tests/neural/engines/*/test_kernel_path.py``)."""

    def kernel(x: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        if not x.is_cuda:
            raise RuntimeError("Expected x.is_cuda() to be true, but got false.")
        return torch_fn(x, *args, **kwargs)  # pragma: no cover — CPU tests

    kernel.__wrapped__ = torch_fn  # type: ignore[attr-defined]
    return kernel


def test_both_engines_run_the_torch_kernel_path_and_agree(
    hooks_qwen, trace_qwen, monkeypatch: pytest.MonkeyPatch
):
    """One DeltaNet kernel-boundary read (``delta_qkv``, served by both)
    with the installed-kernel globals simulated: both engines rebind to
    the torch path for the forward, agree with each other, are bit-identical
    to their own unguarded read, and leave the "installed" globals in place."""
    modeling = _modeling(hooks_qwen)
    assert _modeling(trace_qwen) is modeling  # one modeling module, both engines
    doc = read_doc("delta_qkv", DELTANET_LAYER)

    hooks, trace = engines.both_executors(
        doc, hooks_qwen, trace_qwen, base_texts=BASE_TEXTS
    )
    plain_hooks, plain_trace = hooks.read_value("r"), trace.read_value("r")
    assert_same(plain_hooks, plain_trace, "delta_qkv, unguarded")

    installed = {
        name: _cuda_only(torch_implementation(getattr(modeling, name)))
        for name in KERNEL_GLOBALS
    }
    for name, kernel in installed.items():
        monkeypatch.setattr(modeling, name, kernel)
    hooks, trace = engines.both_executors(
        doc, hooks_qwen, trace_qwen, base_texts=BASE_TEXTS
    )
    guarded_hooks, guarded_trace = hooks.read_value("r"), trace.read_value("r")
    assert_same(guarded_hooks, guarded_trace, "delta_qkv under the simulated kernels")
    assert torch.equal(plain_hooks, guarded_hooks), (
        "pytorch_hooks: the guard changed the numbers"
    )
    assert torch.equal(plain_trace, guarded_trace), (
        "nnsight: the guard changed the numbers"
    )
    for name in KERNEL_GLOBALS:
        assert getattr(modeling, name) is installed[name], name


# --------------------------------------------------------------------------- #
# P20 — the run-time refusal snapshot, mirrored onto the other engine
# --------------------------------------------------------------------------- #

#: Snapshot rows whose trigger names something only one engine has, so there
#: is no "other engine" to run it on.
MIRROR_NOT_APPLICABLE: dict[str, str] = {
    "20": (
        "the trigger's fixture is a hand-built reference bundle (a raw HF model "
        "loaded with experts_implementation='eager'); there is no nnsight "
        "counterpart to hand the trigger, and rebuilding one would test the "
        "shared resolver, not the engine"
    ),
    "31": (
        "the refusal is the nnsight engine's own: it does not serve the ragged "
        "'expert:' face, which the reference engine does"
    ),
    "33": (
        "'dims' on the 'expert:' face — the nnsight engine refuses the face "
        "itself first (row 31), so the row's refusal is unreachable there"
    ),
}

MIRRORED_IDS = sorted(set(table.RUN_TRIGGERS) - set(MIRROR_NOT_APPLICABLE), key=int)


class _Mirrored(table.Fixtures):
    """The snapshot's fixtures with each engine's bundle swapped for the
    other's — the triggers then reach the same refusal sites on the other
    engine unchanged."""

    def __init__(self, *, hooks_qwen: Any, hooks_llama: Any, trace_qwen: Any) -> None:
        self._hooks_qwen = hooks_qwen
        self._hooks_llama = hooks_llama
        self._trace_qwen = trace_qwen

    @property
    def hooks_qwen(self) -> Any:  # type: ignore[override]
        return self._trace_qwen

    @property
    def hooks_llama(self) -> Any:  # type: ignore[override]
        return self._hooks_llama

    @property
    def trace_qwen(self) -> Any:  # type: ignore[override]
        return self._hooks_qwen


@pytest.fixture(scope="module")
def snapshot_fixtures() -> table.Fixtures:
    # shares load_model's cache with the session bundles: nothing loads twice
    return table.Fixtures()


@pytest.fixture(scope="module")
def mirrored_fixtures(hooks_qwen, trace_qwen, trace_llama) -> _Mirrored:
    return _Mirrored(
        hooks_qwen=hooks_qwen, hooks_llama=trace_llama, trace_qwen=trace_qwen
    )


def test_every_run_trigger_is_mirrored_or_named_not_applicable():
    assert set(MIRRORED_IDS) | set(MIRROR_NOT_APPLICABLE) == set(table.RUN_TRIGGERS)
    assert not set(MIRRORED_IDS) & set(MIRROR_NOT_APPLICABLE)


@pytest.mark.parametrize("entry_id", MIRRORED_IDS)
def test_a_run_time_refusal_is_the_same_on_the_other_engine(
    entry_id: str,
    snapshot_fixtures: table.Fixtures,
    mirrored_fixtures: _Mirrored,
    monkeypatch: pytest.MonkeyPatch,
):
    """The snapshot's trigger, run as recorded and run with the engines
    swapped (the executor class flipped, the bundles exchanged): both still
    match the pinned message (``check_entry``) and are the same text."""
    trigger = table.RUN_TRIGGERS[entry_id]
    reference = _refusal_text(lambda: trigger(snapshot_fixtures))
    original_executor = table._executor

    def other_engine(doc_raw: Any, bundle: Any, *, trace: bool = False) -> Any:
        return original_executor(doc_raw, bundle, trace=not trace)

    monkeypatch.setattr(table, "_executor", other_engine)
    with pytest.raises(Exception) as excinfo:
        trigger(mirrored_fixtures)
    check_entry(ENTRIES[entry_id], excinfo.value)
    assert f"{type(excinfo.value).__name__}: {excinfo.value}" == reference


# --------------------------------------------------------------------------- #
# P21 — the pre-forward refusal battery (ExecutorBase methods)
# --------------------------------------------------------------------------- #


def test_check_scoring_refuses_identically(hooks_llama, trace_llama):
    """A table recording ``string_mode: prefix`` under a ``match`` metric
    declaring ``mode: exact`` — rule 4, before any forward (reference:
    ``test_scoring_run.py``)."""
    doc = _match_doc()
    doc["method"]["save"][0]["aggregation"]["mode"] = "exact"
    rows = len(PROMPTS)
    hooks, trace = engines.both_executors(
        doc,
        hooks_llama,
        trace_llama,
        base_texts=PROMPTS,
        extra_columns={
            "cf_answer": SPACED_ANSWERS,
            "scoring_digest": ["a" * 64] * rows,
            "string_mode": ["prefix"] * rows,
        },
    )
    hooks_text = _refusal_text(hooks.check_scoring)
    assert hooks_text == _refusal_text(trace.check_scoring)
    assert "[V4]" in hooks_text and "'exact'" in hooks_text and "'prefix'" in hooks_text


def _stamped_subspace_loader(
    weight: torch.Tensor, **stamped: Any
) -> Callable[[str], Any]:
    """A ``file_path`` bundle whose one ``weight`` entry carries ``stamped``
    as its per-entry fit record (what a swept producer writes)."""

    def load(_path: str) -> TensorBundle:
        return TensorBundle(
            tensors={"weight": weight},
            entry_coords={"weight": {"slot": "weight", "coords": {}, **stamped}},
        )

    return load


def _featurized_interchange_doc(k: int) -> dict[str, Any]:
    doc = interchange_doc("block_output", 1)
    doc["method"]["featurizers"] = {
        "rot": {
            "kind": "subspace",
            "k": k,
            "parametrization": "cayley",
            "file_path": "fit/rot.safetensors",
        }
    }
    doc["method"]["reads"]["v_cf"]["featurizer"] = "rot"
    doc["method"]["writes"]["patch"]["featurizer"] = "rot"
    return doc


def test_an_entry_identity_mismatch_refuses_identically(hooks_llama, trace_llama):
    """The document applies the k=8 fit; the selected entry says it was fitted
    at k=32 — refused by name on both engines (``_check_entry_identity``,
    reached from the shared featurizer build). The twin: the honestly stamped
    entry runs, and the featurized interchange agrees."""
    torch.manual_seed(0)
    weight = torch.linalg.qr(torch.randn(hooks_llama.info.hidden_size, 8))[0]
    doc = _featurized_interchange_doc(k=8)
    hooks, trace = engines.both_executors(
        doc,
        hooks_llama,
        trace_llama,
        base_texts=BASE_TEXTS,
        counterfactual_texts=CF_TEXTS,
        load_tensors=_stamped_subspace_loader(weight, k=32, parametrization="cayley"),
    )
    hooks_text = _refusal_text(hooks.run_all)
    assert hooks_text == _refusal_text(trace.run_all)
    assert "k=8" in hooks_text and "k=32" in hooks_text

    hooks, trace = engines.both_executors(
        doc,
        hooks_llama,
        trace_llama,
        base_texts=BASE_TEXTS,
        counterfactual_texts=CF_TEXTS,
        load_tensors=_stamped_subspace_loader(weight, k=8, parametrization="cayley"),
    )
    assert_same(
        hooks.dense_value("logits"),
        trace.dense_value("logits"),
        "patched logits through the honestly stamped subspace",
    )


# --------------------------------------------------------------------------- #
# P21 — the per-step rules, inside each engine's execute
# --------------------------------------------------------------------------- #

#: ``check_steps`` (``causalab/neural/shared/step_rules.py``) runs two rule
#: functions over every selected step and aggregates what they raise:
#: ``validate_document`` and ``check_loaded_featurizers``. The compiler runs the
#: same two over the representatives, so a step refusal is reachable at the
#: engine seam only at a combination of axis values no representative holds.
#: The first and the aggregation are tested below. The featurizer leg is
#: absent because its identity fields compare one value per axis on each
#: side: when (1, 1), (1, 2) and (2, 1) pass, (2, 2) passes too. Rule 29's
#: width is the exception in principle (it depends on a swept model and a
#: swept component together), but no two tiny fixtures have widths that
#: agree in three of the four cells.
STEP_RULE_LEGS_NOT_REACHABLE = {
    "check_loaded_featurizers": (
        "every identity field is one value per axis on each side, so a "
        "violation at a non-representative step implies one at a representative"
    ),
}


def _reachability_doc(read_layers: list[Any], write_layers: list[Any]) -> dict:
    """The corpus interchange with the operand read and the write on two
    swept sites. Rule 21 refuses a step whose read is strictly deeper than
    its write (``tests/neural/shared/test_step_rules.py``'s document, on the
    tiny fixtures' depths)."""
    raw = copy.deepcopy(json.loads(DOCUMENT.read_text()))
    method = raw["method"]
    method["sites"] = {
        "src": {"component": "block_output", "layers": {"sweep": read_layers}},
        "dst": {"component": "block_output", "layers": {"sweep": write_layers}},
        "lm_head": {"component": "lm_head"},
    }
    method["reads"]["v_cf"]["site"] = "src"
    method["writes"]["patch"]["site"] = "dst"
    return raw


def _step_refusals(
    raw: dict, model_key: str, tmp_path: Path
) -> dict[str, ValidationError]:
    """Each engine's refusal of ``raw`` through ``run_protocol``, after the
    compiler's checks passed for that engine. Nothing may be written."""
    env = engines.corpus_env(tmp_path / "artifacts")
    compiled = engines.compile_for_both(raw, env, overrides={"model.key": model_key})
    refusals: dict[str, ValidationError] = {}
    for engine in (PytorchHooksEngine(), NnsightEngine()):
        # the representatives are legal for this engine: what refuses below
        # is the per-step pass, not the compiler's
        assert validate(compiled, engine, env=env, data=True) is compiled
        out = tmp_path / engine.name
        with pytest.raises(ValidationError) as excinfo:
            run_protocol(compiled, env, engine, out, record=True)
        assert not out.exists() or not any(out.iterdir()), (
            f"{engine.name} wrote before refusing"
        )
        refusals[engine.name] = excinfo.value
    return refusals


def test_a_step_rule_refusal_is_the_same_on_both_engines(tmp_path: Path):
    """``validate_document`` per step: read layers [0, 1] against write layers
    [1, 0] on tiny llama. The three representatives are legal; (1, 0) is
    refused under rule 21 with one text on both engines."""
    refusals = _step_refusals(
        _reachability_doc([[0], [1]], [[1], [0]]), TINY_LLAMA, tmp_path
    )
    hooks, trace = refusals["pytorch_hooks"], refusals["nnsight"]
    assert f"{type(hooks).__name__}: {hooks}" == f"{type(trace).__name__}: {trace}"
    assert not isinstance(hooks, ValidationErrors)
    assert hooks.rule == 21
    assert "block_output layer 1" in str(hooks) and "block_output layer 0" in str(hooks)


def test_distinct_step_rule_refusals_aggregate_the_same_on_both_engines(
    tmp_path: Path,
):
    """``raise_distinct``: read layers [0, 2, 3] against write layers [3, 1]
    on the four-layer qwen fixture. The representatives are legal; (2, 1) and
    (3, 1) are two distinct rule-21 refusals, raised together in the same
    order with the same text on both engines."""
    refusals = _step_refusals(
        _reachability_doc([[0], [2], [3]], [[3], [1]]), TINY_QWEN35_MOE, tmp_path
    )
    hooks, trace = refusals["pytorch_hooks"], refusals["nnsight"]
    assert isinstance(hooks, ValidationErrors) and isinstance(trace, ValidationErrors)
    assert [str(e) for e in hooks.errors] == [str(e) for e in trace.errors]
    assert str(hooks) == str(trace)
    assert len(hooks.errors) == 2 and {e.rule for e in hooks.errors} == {21}


# --------------------------------------------------------------------------- #
# writes_during_generation: the routing refusal is the only guard
# --------------------------------------------------------------------------- #


def _generation_writes_doc() -> dict:
    """The corpus interchange read at the last generated token, its write a
    literal zero kept installed across the decode steps (§2.9
    ``writes_during_generation``). The shape is
    ``tests/protocol/test_engine_routing.py``'s, on the corpus document."""
    raw = copy.deepcopy(json.loads(DOCUMENT.read_text()))
    method = raw["method"]
    del raw["data"]["counterfactual"]
    del method["reads"]["v_cf"]
    del method["intervened_models"]["original_counterfactual"]
    method["positions"] = {"tail": {"generated": {"max_new_tokens": 2}, "index": -1}}
    method["reads"]["logits"]["pos"] = "tail"
    method["writes"]["patch"]["do"] = {"swap": 0.0}
    method["intervened_models"]["patched"]["writes_during_generation"] = True
    method["save"] = [
        {"read": "logits", "model": "patched", "file_path": "l.safetensors"}
    ]
    return in_order(raw)  # `positions` sits before `sites` in the §1 order


def test_writes_during_generation_is_refused_for_nnsight_before_any_weights(
    tmp_path: Path,
):
    """``ENGINE_VERBS["nnsight"]`` has no ``generation_writes``, so routing
    refuses the flagged document under rule 13, naming the verb, before any
    weights load. The trace binds only the prefill occurrence of every write,
    so the executor refuses the flag too
    (``test_generate_frame_nnsight.py``); this is the earlier refusal. The
    reference engine accepts the same compiled document."""
    env = engines.corpus_env(tmp_path / "artifacts")
    compiled = engines.compile_for_both(
        _generation_writes_doc(), env, overrides=OVERRIDES
    )
    assert "generation_writes" in compiled.capabilities
    assert "generation_writes" in ENGINE_VERBS["pytorch_hooks"]
    assert "generation_writes" not in ENGINE_VERBS["nnsight"]
    hooks_engine = PytorchHooksEngine()
    assert route_engine(compiled, hooks_engine) is hooks_engine
    with pytest.raises(ValidationError) as routed:
        route_engine(compiled, NnsightEngine())
    assert routed.value.rule == 13
    assert "lacks ['generation_writes']" in str(routed.value)
    out = tmp_path / "trace"
    with pytest.raises(ValidationError) as ran:
        run_protocol(compiled, env, NnsightEngine(), out, record=True)
    assert str(ran.value) == str(routed.value)
    assert not out.exists() or not any(out.iterdir())


# --------------------------------------------------------------------------- #
# multi-GPU parallelism is the reference engine's alone
# --------------------------------------------------------------------------- #


def test_the_nnsight_engine_takes_no_parallel_geometry():
    """The reference engine's constructor takes the geometry, its collective
    and a sharded load. The nnsight engine is single-device and takes none of
    them, so a caller cannot hand it a geometry it would drop."""
    hooks_params = set(inspect.signature(PytorchHooksEngine).parameters)
    trace_params = set(inspect.signature(NnsightEngine).parameters)
    assert {"parallel", "collective", "sharding"} <= hooks_params
    assert trace_params == {"device", "bundle"}


# --------------------------------------------------------------------------- #
# R4 — workflow step kinds
# --------------------------------------------------------------------------- #

#: Step kinds not run here, and why: both fixtures' inner step is
#: ``behavioral`` (a sampled decode scored into ``decision.json``), which the
#: nnsight engine does not serve (no ``continuations.json``; the accepted
#: difference in ``engines.HOOKS_ONLY_FILES``).
SKIPPED_WORKFLOWS: dict[str, str] = {
    "nested (test_nested_run.py)": "the nested document's one step is behavioral",
    "conditional (test_conditional_run.py)": (
        "the conditional gates on a behavioral step's decision record"
    ),
}


#: The two loaders' defaults differ — the reference engine pins ``eager``,
#: the nnsight engine keeps the checkpoint's default (``sdpa``) — and a
#: saved tensor's ``__metadata__`` stamps ``loaded_attn_implementation``, so
#: a document that saves a read must name the backend for the two stamps to
#: agree. 📐 Measured here first: with the defaults, ``v_cf.safetensors``
#: agreed in every value and differed only in that stamp (``'eager'`` vs
#: ``'sdpa'``); the metric tables agreed either way.
PINNED_EAGER = {**OVERRIDES, "model.attn_implementation": "eager"}


@pytest.fixture(scope="module")
def shuffled_runs(tmp_path_factory: pytest.TempPathFactory) -> engines.BothRuns:
    """The ``shuffled_source`` control (``data.counterfactual.shuffle``) over
    the 4-row fixture table, both engines, the backend pinned so the saved
    operand's stamp compares like against like."""
    base = tmp_path_factory.mktemp("shuffled")
    env = engines.corpus_env(base / "artifacts")
    return engines.run_both(
        shuffled_source_document(seed=0), env, base / "out", overrides=PINNED_EAGER
    )


def test_the_shuffled_source_control_agrees_at_the_engine_seam(shuffled_runs):
    record = json.loads((shuffled_runs.trace_dir / RUN_RECORD_NAME).read_text())
    assert record["canonical"]["data"]["counterfactual"]["shuffle"] == {"seed": 0}
    assert "v_cf.safetensors" in shuffled_runs.trace_result.files
    shuffled_runs.compare()


WORKFLOW_FIXTURES = (
    Path(__file__).resolve().parents[4] / "tests/workflow/fixtures/fan_out"
)
PROTOCOL_DATA = FIXTURES / "data"
FAN_OUT_STEPS = ("scan", "scan@0", "scan@1")


def _fan_out_tree(root: Path) -> Path:
    """The fan-out fixtures, the scan retargeted to tiny llama's two layers
    (the tree ``tests/workflow/test_fan_out.py`` runs on a stub engine), at
    the session's ``main`` revision so the engines' session bundles are the
    ones that run."""
    shutil.copytree(WORKFLOW_FIXTURES, root)
    scan = root / "protocols" / "scan.json"
    doc = json.loads(scan.read_text())
    doc["model"] = {"key": TINY_LLAMA, "revision": "main", "dtype": "fp32"}
    doc["method"]["sites"]["target"]["layers"] = {"sweep": [0, 1]}
    scan.write_text(json.dumps(doc, indent=2))
    return root


@pytest.fixture(scope="module")
def fan_out_runs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """The two-shard scan and its join, run once per engine."""
    from causalab.cli import register_model_key

    base = tmp_path_factory.mktemp("fan_out")
    root = _fan_out_tree(base / "tree")
    env = ResolutionEnv(
        datasets=FileDatasets(root=root, fallback_roots=(PROTOCOL_DATA, TASKS_ROOT)),
        artifacts=FileArtifacts(root=root),
    )
    register_model_key({"model": {"key": TINY_LLAMA, "revision": "main"}})
    loaded = load_workflow(root / "scan_wf.json", env, workflow_dir=root)
    return {
        name: run_workflow(loaded, env, base / name, engine).run_root
        for name, engine in (
            ("hooks", PytorchHooksEngine()),
            ("trace", NnsightEngine()),
        )
    }


#: Step-record fields that name the engine or what only one engine measures:
#: ``engine`` itself; ``forwards`` (the nnsight engine does no interning and
#: reports 0); ``execution`` (the receipt's engine block, the same allowances
#: as ``engines.KNOWN_RECORD_DIFFERENCES``); and the content digests of
#: the metric tables — those tables agree at ``engines.ATOL``, not byte
#: for byte, so their digests (and the join's record of them) differ.
STEP_RECORD_ENGINE_FIELDS = ("engine", "forwards", "execution", "digests")


def _engine_neutral(record: dict[str, Any]) -> dict[str, Any]:
    out = {k: v for k, v in record.items() if k not in STEP_RECORD_ENGINE_FIELDS}
    join = out.get("join")
    if isinstance(join, dict):
        out["join"] = {
            **join,
            "consumed": {
                child: {k: v for k, v in entry.items() if k != "digests"}
                for child, entry in join["consumed"].items()
            },
        }
    return out


@pytest.mark.parametrize("step", FAN_OUT_STEPS)
def test_a_fan_out_step_agrees_per_directory(fan_out_runs, step):
    """Each shard and the join: metric tables row by row at ``ATOL``, the
    step record whole after the engine-naming fields are removed."""
    hooks_dir, trace_dir = fan_out_runs["hooks"] / step, fan_out_runs["trace"] / step
    assert sorted(p.name for p in hooks_dir.iterdir()) == sorted(
        p.name for p in trace_dir.iterdir()
    )
    for table_name in ("iia.json", "logit_diff.json"):
        engines.compare_json(
            json.loads((hooks_dir / table_name).read_text()),
            json.loads((trace_dir / table_name).read_text()),
            f"{step}/{table_name}",
        )
    hooks_record = json.loads((hooks_dir / "_step.json").read_text())
    trace_record = json.loads((trace_dir / "_step.json").read_text())
    engines.compare_json(
        _engine_neutral(hooks_record),
        _engine_neutral(trace_record),
        f"{step}/_step.json",
    )
    if step != "scan":  # the join's record carries no engine of its own
        assert (hooks_record["engine"], trace_record["engine"]) == (
            "pytorch_hooks",
            "nnsight",
        )


def test_the_select_over_the_joined_table_reads_the_same_best_cell(fan_out_runs):
    """The shipped ``select`` script downstream of the join is engine-blind:
    the same best layer out of either engine's joined table, and every step
    completed on both."""
    values = [
        json.loads((run_root / "best" / "values.json").read_text())
        for run_root in fan_out_runs.values()
    ]
    assert values[0] == values[1]
    statuses = [
        {
            name: entry["status"]
            for name, entry in json.loads((run_root / "workflow.json").read_text())[
                "steps"
            ].items()
        }
        for run_root in fan_out_runs.values()
    ]
    assert statuses[0] == statuses[1]
    assert set(statuses[0].values()) == {"completed"}
