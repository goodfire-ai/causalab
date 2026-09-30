"""A caller-owned model enters the reference engine unmutated (spec §9, the
ownership contract).

``ModelBundle.from_model`` is the supported way in for a model the caller
already holds. The contract it is held to here:

* it **refuses** rather than prepares — a train-mode model, a grad-enabled
  one, a right-padding or pad-less
  tokenizer and a precision that is not the declared one are each refused
  naming the exact call the caller makes, and the object is left as it was;
* a bundle handed to ``PytorchHooksEngine(bundle=…)`` runs **instead of**
  ``load_model`` — never inserted into its cache — after the document's
  realization (``key``/``revision``/``dtype``/``quantization``) and the
  engine's ``device`` are checked against it, *before any forward*;
* raising during evaluation removes every hook **without unloading the
  caller-owned model**: same object, same device, same parameters;
* the same weights loaded and handed in write byte-identical files, and run
  receipts that differ in ``execution.model_source`` and nowhere else.

Every refusal has its valid-work twin in this file, so a check that refuses
everything cannot pass.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import pytest
import torch

from causalab.neural.engines.pytorch_hooks.engine import PytorchHooksEngine
from causalab.neural.engines.pytorch_hooks.loading import ModelBundle, load_model
from causalab.io.events import EVENTS_FILE
from causalab.protocol import RUN_RECORD_NAME, run_protocol
from causalab.protocol.rules.errors import ProtocolError
from causalab.protocol.pipeline import compile_protocol
from causalab.io.env import FileArtifacts, FileDatasets, ResolutionEnv

from tests.neural.engines.pytorch_hooks.conftest import TINY_GPT2, TINY_LLAMA
from tests.protocol._docs import UNWRITTEN, base_doc, in_order
from tests.protocol._env import CORPUS_DIR, FIXTURES


pytestmark = pytest.mark.smoke

#: The same document `test_run_protocol_api.py` compares the CLI and the
#: Python entry on: a paired interchange with two metric tables.
DOCUMENT = CORPUS_DIR / "02_interchange_im.json"
OVERRIDES = {"model.key": TINY_LLAMA, "sites.target.layers": 1}

#: What a `pytorch_fn` write raises from inside the forward (T1).
BOOM = "raised mid-forward on purpose"


def raise_mid_forward(f: torch.Tensor) -> torch.Tensor:
    """The declared write function of the T1 document: it fires inside the
    write hook, i.e. in the middle of the model's forward, and raises."""
    raise RuntimeError(BOOM)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _prepared_llama(dtype: torch.dtype = torch.float32) -> tuple[Any, Any]:
    """A tiny Llama prepared by hand exactly as ``load_model`` prepares what it
    loads — the four settings the caller is responsible for."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        TINY_LLAMA, dtype=dtype, attn_implementation="eager"
    )
    model.eval()
    model.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(TINY_LLAMA)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def _from_model(model: Any, tokenizer: Any, **fields: Any) -> ModelBundle:
    fields = {
        "key": TINY_LLAMA,
        "revision": "main",
        "dtype": "fp32",
        **fields,
    }
    return ModelBundle.from_model(model, tokenizer, **fields)


def _checksum(model: Any) -> str:
    flat = torch.cat([p.detach().flatten().float() for p in model.parameters()])
    return hashlib.sha256(flat.numpy().tobytes()).hexdigest()


def _modules_with_hooks(model: Any) -> list[str]:
    return [
        name
        for name, module in model.named_modules()
        if module._forward_hooks or module._forward_pre_hooks
    ]


def _files(root: Path) -> dict[str, bytes]:
    # every file but the event stream: a timestamped sidecar (workflow spec
    # §4.3) that is an input to nothing, and the one file two runs of one
    # document are allowed to differ in
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != EVENTS_FILE
    }


class _ForwardCounter:
    """Counts the model's own forwards through a pre-hook the test owns (and
    removes) — so "refused before any forward" is a measured claim."""

    def __init__(self, model: Any) -> None:
        self.calls = 0
        self._handle = model.register_forward_pre_hook(self._count)

    def _count(self, _module: Any, _args: Any) -> None:
        self.calls += 1

    def remove(self) -> None:
        self._handle.remove()


@pytest.fixture(scope="module")
def env(tmp_path_factory: pytest.TempPathFactory) -> ResolutionEnv:
    artifacts = tmp_path_factory.mktemp("artifacts")
    shutil.copytree(FIXTURES / "artifacts", artifacts, dirs_exist_ok=True)
    return ResolutionEnv(
        datasets=FileDatasets(root=FIXTURES / "data"),
        artifacts=FileArtifacts(root=artifacts),
    )


@pytest.fixture(scope="module")
def loaded_02(env: ResolutionEnv):
    return compile_protocol(DOCUMENT, env=env, overrides=OVERRIDES)


# --------------------------------------------------------------------------- #
# from_model refuses, naming the call — and leaves the object alone (T3)
# --------------------------------------------------------------------------- #


class TestFromModelRefusesUnpreparedModels:
    """Each refusal asserts the *refusal* and the un-mutated state, never a
    resulting mode: were ``from_model`` to call ``.eval()`` itself (mutating the
    caller's model), ``pytest.raises`` fails and the mode assertion after it would
    never have been the guard."""

    def test_a_train_mode_model_is_refused_naming_eval(self) -> None:
        from transformers import AutoConfig, AutoTokenizer, LlamaForCausalLM

        torch.manual_seed(0)
        model = LlamaForCausalLM(AutoConfig.from_pretrained(TINY_LLAMA))
        model.set_attn_implementation("eager")
        model.requires_grad_(False)
        assert model.training  # a model built from a config starts in train mode
        with pytest.raises(ProtocolError, match=r"model\.eval\(\)") as err:
            _from_model(model, AutoTokenizer.from_pretrained(TINY_LLAMA))
        assert err.value.code == "P4"
        assert model.training, "from_model re-moded a model it does not own"

    def test_a_grad_enabled_model_is_refused_naming_requires_grad(self) -> None:
        model, tokenizer = _prepared_llama()
        model.requires_grad_(True)
        with pytest.raises(ProtocolError, match=r"model\.requires_grad_\(False\)"):
            _from_model(model, tokenizer)
        assert all(p.requires_grad for p in model.parameters())

    def test_a_non_eager_attention_is_accepted_without_mutation(self) -> None:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(TINY_LLAMA)  # the default: sdpa
        model.eval()
        model.requires_grad_(False)
        before = model.config._attn_implementation
        assert before != "eager"
        bundle = _from_model(model, _prepared_llama()[1])
        assert bundle.model is model
        assert model.config._attn_implementation == before

    def test_a_right_padding_tokenizer_is_refused_naming_the_side(self) -> None:
        from transformers import AutoTokenizer

        model, _ = _prepared_llama()
        tokenizer = AutoTokenizer.from_pretrained(TINY_GPT2)  # pads right, no pad
        assert tokenizer.padding_side == "right"
        with pytest.raises(ProtocolError, match=r'tokenizer\.padding_side = "left"'):
            _from_model(model, tokenizer)
        assert tokenizer.padding_side == "right"

    def test_a_pad_less_tokenizer_is_refused_naming_the_pad_token(self) -> None:
        from transformers import AutoTokenizer

        model, _ = _prepared_llama()
        tokenizer = AutoTokenizer.from_pretrained(TINY_GPT2)
        tokenizer.padding_side = "left"
        assert tokenizer.pad_token is None
        with pytest.raises(
            ProtocolError, match=r"tokenizer\.pad_token = tokenizer\.eos_token"
        ):
            _from_model(model, tokenizer)
        assert tokenizer.pad_token is None

    def test_a_precision_the_weights_do_not_have_is_refused(self) -> None:
        model, tokenizer = _prepared_llama()  # fp32 weights
        with pytest.raises(ProtocolError, match="bf16") as err:
            _from_model(model, tokenizer, dtype="bf16")
        assert "torch.float32" in str(err.value)
        assert next(model.parameters()).dtype is torch.float32

    def test_an_unknown_dtype_is_refused(self) -> None:
        model, tokenizer = _prepared_llama()
        with pytest.raises(ProtocolError, match="fp8"):
            _from_model(model, tokenizer, dtype="fp8")


class TestFromModelAcceptsAPreparedModel:
    """The valid-work twin of every refusal above."""

    def test_the_same_object_the_same_registry_row_and_no_load(self) -> None:
        model, tokenizer = _prepared_llama()
        before = load_model.cache_info()
        bundle = _from_model(model, tokenizer)
        assert load_model.cache_info() == before, "from_model touched the loader"
        assert bundle.model is model and bundle.tokenizer is tokenizer
        assert (
            bundle.key,
            bundle.revision,
            bundle.devices.requested,
            bundle.dtype,
        ) == (
            TINY_LLAMA,
            "main",
            "cpu",
            "fp32",
        )
        # the map is read off the model's own parameters, not asserted
        assert bundle.devices == load_model(TINY_LLAMA).devices
        assert bundle.quantization is None
        # the tap table reads the registry row: it is the loader's row exactly
        assert bundle.info == load_model(TINY_LLAMA).info
        assert bundle.streams == load_model(TINY_LLAMA).streams

    def test_a_bf16_model_declared_bf16_is_accepted(self) -> None:
        model, tokenizer = _prepared_llama(torch.bfloat16)
        bundle = _from_model(model, tokenizer, dtype="bf16")
        assert bundle.dtype == "bf16" and bundle.model is model

    def test_a_declared_quantization_is_kept_by_value(self) -> None:
        """A quantized model's weights are integer tensors, so the precision
        check steps aside and the block is recorded as given."""
        model, tokenizer = _prepared_llama()
        block = {"method": "bitsandbytes", "scheme": "nf4"}
        bundle = _from_model(model, tokenizer, quantization=block)
        assert bundle.quantization == block and bundle.quantization is not block


# --------------------------------------------------------------------------- #
# the engine checks the realization before any forward (T3)
# --------------------------------------------------------------------------- #


class TestTheEngineChecksTheRealization:
    def _refused(
        self, loaded: Any, env: ResolutionEnv, engine: PytorchHooksEngine, out: Path
    ) -> ProtocolError:
        counter = _ForwardCounter(engine.bundle.model)
        try:
            with pytest.raises(ProtocolError) as err:
                run_protocol(loaded, env, engine, out)
        finally:
            counter.remove()
        assert counter.calls == 0, "the model ran before the realization check"
        assert err.value.code == "P4"
        return err.value

    def test_a_dtype_disagreement_is_refused_before_any_forward(
        self, loaded_02, env, tmp_path
    ) -> None:
        model, tokenizer = _prepared_llama(torch.bfloat16)
        bundle = _from_model(model, tokenizer, dtype="bf16")
        err = self._refused(loaded_02, env, PytorchHooksEngine(bundle=bundle), tmp_path)
        assert "model.dtype" in str(err)
        assert "'fp32'" in str(err) and "'bf16'" in str(err)  # both sides named

    def test_an_attention_disagreement_is_refused_before_any_forward(
        self, env, tmp_path
    ) -> None:
        loaded = compile_protocol(
            DOCUMENT,
            env=env,
            overrides={**OVERRIDES, "model.attn_implementation": "sdpa"},
        )
        model, tokenizer = _prepared_llama()
        bundle = _from_model(model, tokenizer)
        err = self._refused(loaded, env, PytorchHooksEngine(bundle=bundle), tmp_path)
        assert "model.attn_implementation" in str(err)
        assert "'eager'" in str(err) and "'sdpa'" in str(err)
        assert model.config._attn_implementation == "eager"

    @pytest.mark.parametrize(
        "field, value",
        [
            ("key", "caller/asserts-another-checkpoint"),
            ("revision", "not-main"),
            ("quantization", {"method": "bitsandbytes", "scheme": "nf4"}),
        ],
    )
    def test_a_key_revision_or_quantization_disagreement_is_refused(
        self, loaded_02, env, tmp_path, field: str, value: Any
    ) -> None:
        model, tokenizer = _prepared_llama()
        bundle = _from_model(model, tokenizer, **{field: value})
        err = self._refused(loaded_02, env, PytorchHooksEngine(bundle=bundle), tmp_path)
        assert f"model.{field}" in str(err)
        assert repr(value) in str(err)

    def test_a_device_disagreement_is_refused(self, loaded_02, env, tmp_path) -> None:
        model, tokenizer = _prepared_llama()
        bundle = _from_model(model, tokenizer)
        err = self._refused(
            loaded_02, env, PytorchHooksEngine(device="cuda:1", bundle=bundle), tmp_path
        )
        assert "device" in str(err) and "'cuda:1'" in str(err) and "'cpu'" in str(err)


# --------------------------------------------------------------------------- #
# T2 — the same weights, loaded and handed in, are the same run
# --------------------------------------------------------------------------- #


class TestACallerModelIsTheSameRun:
    @pytest.fixture(scope="class")
    def both_runs(self, loaded_02, env, tmp_path_factory) -> tuple[Path, Path]:
        base = tmp_path_factory.mktemp("same-run")
        via_loader, via_caller = base / "loaded", base / "caller"
        run_protocol(loaded_02, env, PytorchHooksEngine(), via_loader, record=True)
        model, tokenizer = _prepared_llama()
        bundle = _from_model(model, tokenizer)
        before = load_model.cache_info()
        run_protocol(
            loaded_02, env, PytorchHooksEngine(bundle=bundle), via_caller, record=True
        )
        assert load_model.cache_info() == before, "the caller run loaded a model"
        return via_loader, via_caller

    def test_every_saved_file_is_byte_identical(self, both_runs) -> None:
        via_loader, via_caller = both_runs
        loaded_files, caller_files = _files(via_loader), _files(via_caller)
        assert set(loaded_files) == set(caller_files)
        differing = sorted(
            name
            for name in loaded_files
            if loaded_files[name] != caller_files[name] and name != RUN_RECORD_NAME
        )
        assert not differing, f"same weights, different bytes: {differing}"

    def test_the_run_records_differ_only_in_model_source(self, both_runs) -> None:
        via_loader, via_caller = both_runs
        loaded = json.loads((via_loader / RUN_RECORD_NAME).read_text())
        caller = json.loads((via_caller / RUN_RECORD_NAME).read_text())
        assert loaded["execution"] == {
            "batch_rows": None,
            "device": "cpu",
            "fit_rows": None,
            "model_source": "loaded",
            "parallel": {
                "data": 1,
                "data_mode": "points",
                "pipeline": 1,
                "context": 1,
                "tensor": 1,
                "expert": 1,
                "world": 1,
                "launcher": "solo",
            },
        }
        assert caller["execution"] == {
            "batch_rows": None,
            "device": "cpu",
            "fit_rows": None,
            "model_source": "caller",
            "parallel": {
                "data": 1,
                "data_mode": "points",
                "pipeline": 1,
                "context": 1,
                "tensor": 1,
                "expert": 1,
                "world": 1,
                "launcher": "solo",
            },
        }
        # execution provenance, not identity: in no canonical form, no digest
        assert "model_source" not in json.dumps(loaded["canonical"])
        assert "model_source" not in json.dumps(caller["points"])
        assert loaded["document_digest"] == caller["document_digest"]
        loaded["execution"].pop("model_source")
        caller["execution"].pop("model_source")
        assert loaded == caller


# --------------------------------------------------------------------------- #
# T1 — raising during evaluation removes every hook, unloads nothing
# --------------------------------------------------------------------------- #


def _raising_write_doc() -> dict[str, Any]:
    """`base_doc` retargeted at tiny Llama with its write a declared
    `pytorch_fn` that raises: no counterfactual role, since the swap is gone
    and a dead read — and the model nobody reads — would be refused (§2.9)."""
    doc = base_doc()
    doc["model"]["key"] = TINY_LLAMA
    doc["method"]["sites"]["tgt"]["layers"] = 1
    del doc["method"]["reads"]["v_cf"]
    del doc["method"]["intervened_models"][UNWRITTEN]
    del doc["data"]["counterfactual"]
    doc["method"]["code"] = {"boom": {"locator": f"{__name__}.raise_mid_forward"}}
    doc["method"]["writes"]["patch"]["do"] = {"pytorch_fn": {"code": "boom"}}
    return in_order(doc)


def test_raising_mid_forward_removes_every_hook_and_unloads_nothing(
    env, tmp_path
) -> None:
    """The cleanup contract, measured from outside the hook scopes:
    after a `pytorch_fn` write raises inside the forward, no module carries a
    forward or pre-forward hook, the model is the same object on the same
    device with the same parameters, and the loader's cache saw nothing.

    *Mutation:* dropping one ``finally: handle.remove()`` in executor.py's
    ``_installed`` / ``_capturing`` / ``_accumulating`` leaves that hook on
    its module, and the hook census below names it.
    """
    model, tokenizer = _prepared_llama()
    bundle = _from_model(model, tokenizer)
    assert not _modules_with_hooks(model)
    checksum, device = _checksum(model), next(model.parameters()).device
    cache_before = load_model.cache_info()

    loaded = compile_protocol(_raising_write_doc(), env=env)
    with pytest.raises(RuntimeError, match=BOOM):
        run_protocol(loaded, env, PytorchHooksEngine(bundle=bundle), tmp_path)

    assert _modules_with_hooks(model) == [], "a hook survived the raise"
    assert bundle.model is model
    assert next(model.parameters()).device == device
    assert _checksum(model) == checksum
    assert not model.training and not any(p.requires_grad for p in model.parameters())
    assert load_model.cache_info() == cache_before, "the engine loaded a model"


def test_the_same_document_runs_when_the_write_does_not_raise(env, tmp_path) -> None:
    """The twin: the T1 document with an identity `pytorch_fn` runs to
    completion on the caller's model and says so in the record."""
    model, tokenizer = _prepared_llama()
    doc = _raising_write_doc()
    doc["method"]["code"]["boom"]["locator"] = f"{__name__}.identity"
    loaded = compile_protocol(doc, env=env)
    result = run_protocol(
        loaded,
        env,
        PytorchHooksEngine(bundle=_from_model(model, tokenizer)),
        tmp_path,
        record=True,
    )
    assert result.files
    record = json.loads((tmp_path / RUN_RECORD_NAME).read_text())
    assert record["execution"]["model_source"] == "caller"
    assert _modules_with_hooks(model) == []


def identity(f: torch.Tensor) -> torch.Tensor:
    return f
