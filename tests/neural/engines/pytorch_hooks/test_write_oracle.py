"""Write semantics vs the raw-hook oracle — interchange, feature space +
error term, additive steer, gaussian determinism, cross-model two-pass
path patching. Assertions and tolerances follow the oracle suites
(stack-vs-oracle atol=1e-5 rtol=1e-4; wrapper-level cases atol=1e-4
rtol=1e-3; determinism byte-identical)."""

from __future__ import annotations

import pytest
import torch

from causalab.neural.shared.encoding import encode
from causalab.protocol.rules.errors import ValidationError

from tests.neural.engines.pytorch_hooks import hook_oracle_lib as oracle_lib
from tests.neural.engines.pytorch_hooks._drive import base_data_section, executor_for
from tests.neural.engines.pytorch_hooks.conftest import (
    BASE_TEXT,
    COUNTERFACTUAL_TEXT,
    OracleShim,
)
from tests.protocol._docs import UNWRITTEN, saved

pytestmark = pytest.mark.unit

TOL = dict(atol=1e-5, rtol=1e-4)


def _inputs(bundle, text: str):
    batch = encode(bundle.tokenizer, [text])
    return {"input_ids": batch.input_ids, "attention_mask": batch.attention_mask}


def interchange_doc(pos: int = 1) -> dict:
    """The counterfactual's residual at (L0, ``pos``), read on the
    un-intervened network (`UNWRITTEN`), swapped into base in
    ``patched``, whose logits are saved."""
    return {
        "header": {"protocol_version": "4"},
        "model": {"key": "test", "revision": "main"},
        "data": base_data_section(with_counterfactual=True),
        "method": {
            "intervened_models": {
                UNWRITTEN: {"input": "counterfactual", "reads": ["v_cf"]},
                "patched": {"input": "base", "reads": ["logits"], "writes": ["patch"]},
            },
            "sites": {
                "tgt": {"component": "block_output", "layers": [0]},
                "lm_head": {"component": "lm_head"},
            },
            "reads": {
                "v_cf": {"site": "tgt", "pos": {"index": pos}},
                "logits": {"site": "lm_head", "pos": {"index": -1}},
            },
            "writes": {
                "patch": {"site": "tgt", "pos": {"index": pos}, "do": {"swap": "v_cf"}}
            },
            "save": [saved("logits", "patched", "l.safetensors")],
        },
    }


def base_only(doc: dict) -> dict:
    """`interchange_doc` with no counterfactual: the read of it and the
    model that took it go with the role (the write's operand is the caller's
    to replace)."""
    del doc["method"]["reads"]["v_cf"]
    del doc["method"]["intervened_models"][UNWRITTEN]
    doc["data"] = base_data_section(with_counterfactual=False)
    return doc


def read_on(doc: dict, name: str, model: str, address: dict, file_path: str) -> None:
    """Add a read at ``address`` taken on ``model`` and save it as a tensor."""
    doc["method"]["reads"][name] = address
    doc["method"]["intervened_models"][model]["reads"].append(name)
    doc["method"]["save"].append(saved(name, model, file_path))


def test_interchange_matches_oracle(bundle, oracle: OracleShim):
    """The corpus-02 shape: swap the counterfactual residual into base at (L0, p1);
    the patched logits equal the oracle's hand-rolled patch, and differ
    from clean (non-vacuity)."""
    executor = executor_for(
        interchange_doc(),
        bundle,
        base_texts=[BASE_TEXT],
        counterfactual_texts=[COUNTERFACTUAL_TEXT],
    )
    have = executor.read_value("logits")[:, 0, :]

    base_inputs = _inputs(bundle, BASE_TEXT)
    cf_resid = oracle_lib.capture_residual(
        oracle, 0, _inputs(bundle, COUNTERFACTUAL_TEXT)
    )
    want = oracle_lib.next_token_logits(
        oracle, base_inputs, layer=0, positions=[1], patch_values=cf_resid[:, 1:2, :]
    )
    clean = oracle_lib.next_token_logits(oracle, base_inputs)
    assert not torch.allclose(want, clean, atol=1e-4)  # non-vacuity
    torch.testing.assert_close(have, want, **TOL)


def test_feature_space_swap_keeps_the_complement(llama_bundle):
    """The error-term contract (§2.5): a swap through a k=d/2 subspace
    replaces only the in-subspace coordinates; the complement of the BASE
    value survives (the oracle's lossy-split case 3)."""
    d = llama_bundle.info.hidden_size
    k = d // 2
    doc = interchange_doc()
    doc["method"]["featurizers"] = {
        "rot": {"kind": "subspace", "k": k, "parametrization": "cayley"}
    }
    doc["method"]["reads"]["v_cf"]["featurizer"] = "rot"
    doc["method"]["writes"]["patch"]["featurizer"] = "rot"
    executor = executor_for(
        doc,
        llama_bundle,
        base_texts=[BASE_TEXT],
        counterfactual_texts=[COUNTERFACTUAL_TEXT],
    )
    have = executor.read_value("logits")[:, 0, :]

    # the SAME Q both sides: extract the parametrized weight from the stage
    q = executor.stage("rot").slot_params()["weight"].detach()
    shim = OracleShim(hf_model=llama_bundle.model)
    base_inputs = _inputs(llama_bundle, BASE_TEXT)
    base_resid = oracle_lib.capture_residual(shim, 0, base_inputs)[:, 1:2, :]
    cf_resid = oracle_lib.capture_residual(
        shim, 0, _inputs(llama_bundle, COUNTERFACTUAL_TEXT)
    )[:, 1:2, :]
    patch = base_resid - (base_resid @ q) @ q.T + (cf_resid @ q) @ q.T
    want = oracle_lib.next_token_logits(
        shim, base_inputs, layer=0, positions=[1], patch_values=patch
    )
    clean = oracle_lib.next_token_logits(shim, base_inputs)
    assert not torch.allclose(want, clean, atol=1e-4)
    torch.testing.assert_close(have, want, **TOL)


def test_dims_swap_is_a_subspace_swap(llama_bundle):
    """A dims selection swaps only those feature coordinates — raw space,
    dims [0, 2]: untouched dims come from the pre-write value."""
    doc = interchange_doc()
    doc["method"]["writes"]["patch"]["dims"] = [0, 2]
    doc["method"]["reads"]["v_cf"]["dims"] = [0, 2]
    executor = executor_for(
        doc,
        llama_bundle,
        base_texts=[BASE_TEXT],
        counterfactual_texts=[COUNTERFACTUAL_TEXT],
    )
    have = executor.read_value("logits")[:, 0, :]

    shim = OracleShim(hf_model=llama_bundle.model)
    base_inputs = _inputs(llama_bundle, BASE_TEXT)
    base_resid = oracle_lib.capture_residual(shim, 0, base_inputs)[:, 1:2, :].clone()
    cf_resid = oracle_lib.capture_residual(
        shim, 0, _inputs(llama_bundle, COUNTERFACTUAL_TEXT)
    )[:, 1:2, :]
    patch = base_resid.clone()
    patch[..., [0, 2]] = cf_resid[..., [0, 2]]
    want = oracle_lib.next_token_logits(
        shim, base_inputs, layer=0, positions=[1], patch_values=patch
    )
    torch.testing.assert_close(have, want, **TOL)


def test_add_scaled_matches_oracle_steer(bundle, oracle: OracleShim):
    """add_scaled with a literal-alpha scalar operand == the oracle's
    additive steer of a constant (broadcast scalar) vector."""
    doc = base_only(interchange_doc())
    doc["method"]["writes"]["patch"]["do"] = {"add_scaled": {"op": 2.5, "alpha": 1.0}}
    executor = executor_for(doc, bundle, base_texts=[BASE_TEXT])
    have = executor.read_value("logits")[:, 0, :]

    base_inputs = _inputs(bundle, BASE_TEXT)

    def write_fn(hidden: torch.Tensor) -> None:
        hidden[:, 1, :] = hidden[:, 1, :] + 2.5

    block = oracle_lib.decoder_block(oracle, 0)
    want = oracle_lib.component_written_logits(
        oracle, base_inputs, block, "out", write_fn
    )
    torch.testing.assert_close(have, want, **TOL)


def test_gaussian_contract(llama_bundle):
    """The noise-oracle contract: scale 0 == clean; same seed byte-identical
    across runs; different seed differs."""

    def run(seed: int, scale: float) -> torch.Tensor:
        doc = base_only(interchange_doc())
        doc["method"]["writes"]["patch"]["do"] = {
            "gaussian": {"seed": seed, "scale": scale, "axis": "tp_duplicated"}
        }
        executor = executor_for(doc, llama_bundle, base_texts=[BASE_TEXT])
        return executor.read_value("logits")[:, 0, :]

    shim = OracleShim(hf_model=llama_bundle.model)
    clean = oracle_lib.next_token_logits(shim, _inputs(llama_bundle, BASE_TEXT))
    torch.testing.assert_close(run(7, 0.0), clean, **TOL)  # scale 0 == clean
    noisy = run(7, 3.0)
    assert not torch.allclose(noisy, clean, atol=1e-4)  # noise moves logits
    torch.testing.assert_close(run(7, 3.0), noisy, atol=0.0, rtol=0.0)  # same seed
    assert not torch.allclose(run(8, 3.0), noisy, atol=1e-4)  # different seed


def test_gaussian_draw_realization(llama_bundle):
    """The RNG realization the parity goldens pin: the draw is
    Generator().manual_seed(seed) → randn((batch, n_pos, width)), made
    outside the model."""
    doc = base_only(interchange_doc())
    doc["method"]["writes"]["patch"]["do"] = {
        "gaussian": {"seed": 7, "scale": 3.0, "axis": "tp_duplicated"}
    }
    executor = executor_for(doc, llama_bundle, base_texts=[BASE_TEXT])
    have = executor.read_value("logits")[:, 0, :]

    d = llama_bundle.info.hidden_size
    draw = torch.randn((1, 1, d), generator=torch.Generator().manual_seed(7))
    shim = OracleShim(hf_model=llama_bundle.model)
    base_inputs = _inputs(llama_bundle, BASE_TEXT)
    base_resid = oracle_lib.capture_residual(shim, 0, base_inputs)[:, 1:2, :]
    want = oracle_lib.next_token_logits(
        shim, base_inputs, layer=0, positions=[1], patch_values=base_resid + 3.0 * draw
    )
    torch.testing.assert_close(have, want, **TOL)


def test_renormalize_restores_the_pre_write_norm(llama_bundle):
    """add_scaled + renormalize at one address: the delta applies, then the
    feature vector is rescaled to the norm it had before either write (§2.8,
    ``f ← f·‖f₀‖/‖f‖`` with ``f₀`` the pre-write value).

    The written site is checked against the oracle directly, not only through
    the logits: at L0 the RMSNorm downstream is nearly scale-invariant, so the
    rescaled and the merely bumped values give logits within the tolerance,
    and a logits-only check passed while ``renormalize`` was the identity.
    The anti-vacuity check pins that the two candidates differ at
    the site by far more than the tolerance."""
    doc = base_only(interchange_doc())
    doc["method"]["writes"] = {
        "nudge": {
            "site": "tgt",
            "pos": {"index": 1},
            "do": {"add_scaled": {"op": 2.5, "alpha": 1.0}},
        },
        "renorm": {"site": "tgt", "pos": {"index": 1}, "do": {"renormalize": True}},
    }
    doc["method"]["intervened_models"]["patched"]["writes"] = [
        "nudge",
        "renorm",
    ]
    doc["method"]["intervened_models"]["original"] = {"input": "base", "reads": []}
    read_on(
        doc, "written", "patched", {"site": "tgt", "pos": {"index": 1}}, "w.safetensors"
    )
    read_on(
        doc, "v_base", "original", {"site": "tgt", "pos": {"index": 1}}, "b.safetensors"
    )
    executor = executor_for(doc, llama_bundle, base_texts=[BASE_TEXT])
    have = executor.read_value("logits")[:, 0, :]

    shim = OracleShim(hf_model=llama_bundle.model)
    base_inputs = _inputs(llama_bundle, BASE_TEXT)
    base_resid = oracle_lib.capture_residual(shim, 0, base_inputs)[:, 1:2, :]
    bumped = base_resid + 2.5
    rescaled = bumped * (
        base_resid.norm(dim=-1, keepdim=True) / bumped.norm(dim=-1, keepdim=True)
    )
    want = oracle_lib.next_token_logits(
        shim, base_inputs, layer=0, positions=[1], patch_values=rescaled
    )
    torch.testing.assert_close(have, want, **TOL)

    written, v_base = executor.read_value("written"), executor.read_value("v_base")
    torch.testing.assert_close(v_base, base_resid, **TOL)
    torch.testing.assert_close(written, rescaled, **TOL)
    torch.testing.assert_close(written.norm(dim=-1), v_base.norm(dim=-1), **TOL)
    # anti-vacuity: the identity landing would leave the bumped norm
    assert float((bumped.norm(dim=-1) - base_resid.norm(dim=-1)).abs().min()) > 1.0


def test_renormalize_on_dims_through_a_featurizer_writes_into_its_own_copy(
    llama_bundle,
):
    """``renormalize`` on a ``dims`` subset through a subspace featurizer,
    under grad: the write math lands its ``dims`` slice in a copy of the
    featurized value, never in the featurizer's output itself — that output
    is what the error term (``x - f @ Qᵀ``) saved for backward, so an
    in-place ``index_copy_`` on it (the regression) raised on ``backward``.
    The value is the no-grad landing's to the bit."""
    doc = base_only(interchange_doc())
    doc["method"]["featurizers"] = {
        "rot": {"kind": "subspace", "k": 4, "parametrization": "cayley"}
    }
    doc["method"]["writes"] = {
        "nudge": {
            "site": "tgt",
            "pos": {"index": 1},
            "do": {"add_scaled": {"op": 2.5, "alpha": 1.0}},
        },
        "renorm": {
            "site": "tgt",
            "pos": {"index": 1},
            "do": {"renormalize": True},
            "dims": [0, 2],
            "featurizer": "rot",
        },
    }
    doc["method"]["intervened_models"]["patched"]["writes"] = ["nudge", "renorm"]
    read_on(
        doc, "written", "patched", {"site": "tgt", "pos": {"index": 1}}, "w.safetensors"
    )
    executor = executor_for(
        doc, llama_bundle, base_texts=[BASE_TEXT], grad_enabled=True
    )
    written = executor.read_value("written")
    parameters = tuple(executor.stage("rot").parameters())
    assert parameters and written.requires_grad
    grads = torch.autograd.grad(written.sum(), parameters)
    assert all(torch.isfinite(g).all() for g in grads)

    plain = executor_for(doc, llama_bundle, base_texts=[BASE_TEXT]).read_value(
        "written"
    )
    assert torch.equal(written.detach(), plain)


def test_two_pass_path_patching_matches_oracle(bundle, oracle: OracleShim):
    """The spec's worked example (corpus 03) at tiny scale: sender L0 →
    receiver L1, v* read under the sender swap in 'patched', injected clean
    in 'final' — vs the two-pass hook oracle (wrapper tolerance atol=1e-4
    rtol=1e-3)."""
    doc = {
        "header": {"protocol_version": "4"},
        "model": {"key": "test", "revision": "main"},
        "data": base_data_section(with_counterfactual=True),
        "method": {
            "intervened_models": {
                UNWRITTEN: {"input": "counterfactual", "reads": ["v_sender"]},
                "patched": {
                    "input": "base",
                    "reads": ["v_receiver"],
                    "writes": ["swap_sender"],
                },
                "final": {"input": "base", "reads": ["logits"], "writes": ["inject"]},
            },
            "sites": {
                "sender": {"component": "block_output", "layers": [0]},
                "receiver": {"component": "block_output", "layers": [1]},
                "lm_head": {"component": "lm_head"},
            },
            "reads": {
                "v_sender": {"site": "sender", "pos": {"index": -1}},
                "v_receiver": {"site": "receiver", "pos": {"index": -1}},
                "logits": {"site": "lm_head", "pos": {"index": -1}},
            },
            "writes": {
                "swap_sender": {
                    "site": "sender",
                    "pos": {"index": -1},
                    "do": {"swap": "v_sender"},
                },
                "inject": {
                    "site": "receiver",
                    "pos": {"index": -1},
                    "do": {"swap": "v_receiver"},
                },
            },
            "save": [saved("logits", "final", "l.safetensors")],
        },
    }
    executor = executor_for(
        doc, bundle, base_texts=[BASE_TEXT], counterfactual_texts=[COUNTERFACTUAL_TEXT]
    )
    have = executor.read_value("logits")[:, 0, :]

    base_inputs = _inputs(bundle, BASE_TEXT)
    cf_resid = oracle_lib.capture_residual(
        oracle, 0, _inputs(bundle, COUNTERFACTUAL_TEXT)
    )
    sender_block = oracle_lib.decoder_block(oracle, 0)
    receiver_block = oracle_lib.decoder_block(oracle, 1)
    patch_last = cf_resid[:, -1, :]

    def sender_write(hidden: torch.Tensor) -> None:
        hidden[:, -1, :] = patch_last

    v_star = oracle_lib.capture_with_writes(
        oracle,
        base_inputs,
        receiver_block,
        "out",
        [(sender_block, "out", sender_write)],
    )[:, -1:, :]
    want = oracle_lib.next_token_logits(
        oracle, base_inputs, layer=1, positions=[-1], patch_values=v_star
    )
    clean = oracle_lib.next_token_logits(oracle, base_inputs)
    assert not torch.allclose(want, clean, atol=1e-4)
    torch.testing.assert_close(have, want, atol=1e-4, rtol=1e-3)


def test_reads_see_the_fully_written_state(bundle, oracle: OracleShim):
    """§2.7: a read in model M at the written address sees the write."""
    doc = interchange_doc()
    read_on(
        doc, "v_at", "patched", {"site": "tgt", "pos": {"index": 1}}, "v.safetensors"
    )
    executor = executor_for(
        doc, bundle, base_texts=[BASE_TEXT], counterfactual_texts=[COUNTERFACTUAL_TEXT]
    )
    cf_resid = oracle_lib.capture_residual(
        oracle, 0, _inputs(bundle, COUNTERFACTUAL_TEXT)
    )[:, 1, :]
    torch.testing.assert_close(executor.read_value("v_at")[:, 0, :], cf_resid, **TOL)


def test_composed_chain_sizes_stages_by_chain_width(llama_bundle):
    """§2.5 composition: a gate after a k=3 rotation is a 3-wide gate, and
    the composed swap equals the hand-rolled two-stage math — the counterfactual's
    on-features replace the base's inside the rotated subspace, everything
    else (the gate's off-features AND the rotation's complement) survives
    from the base."""
    d = llama_bundle.info.hidden_size
    doc = interchange_doc()
    doc["method"]["featurizers"] = {
        "rot": {"kind": "subspace", "k": 3, "parametrization": "cayley"},
        "gate": {"kind": "gate"},
    }
    doc["method"]["reads"]["v_cf"]["featurizer"] = ["rot", "gate"]
    doc["method"]["writes"]["patch"]["featurizer"] = ["rot", "gate"]
    executor = executor_for(
        doc,
        llama_bundle,
        base_texts=[BASE_TEXT],
        counterfactual_texts=[COUNTERFACTUAL_TEXT],
    )
    gate = executor.stage("gate")
    assert tuple(gate.theta.shape) == (3,)  # chain width, not the site width d
    with torch.no_grad():  # a non-trivial hard-eval mask: on for the last dim
        gate.theta.copy_(torch.tensor([-2.0, -1.0, 2.0]))
    rot_q = executor.stage("rot").slot_params()["weight"].detach()
    assert tuple(rot_q.shape) == (d, 3)

    have = executor.read_value("logits")[:, 0, :]
    shim = OracleShim(hf_model=llama_bundle.model)
    base_inputs = _inputs(llama_bundle, BASE_TEXT)
    base_resid = oracle_lib.capture_residual(shim, 0, base_inputs)[:, 1:2, :]
    cf_resid = oracle_lib.capture_residual(
        shim, 0, _inputs(llama_bundle, COUNTERFACTUAL_TEXT)
    )[:, 1:2, :]
    mask = (gate.theta > 0).float()
    z_base, z_cf = base_resid @ rot_q, cf_resid @ rot_q
    z_new = mask * z_cf + (1.0 - mask) * z_base  # gate swap keeps off-features
    patch = base_resid - z_base @ rot_q.T + z_new @ rot_q.T  # rot err survives
    want = oracle_lib.next_token_logits(
        shim, base_inputs, layer=0, positions=[1], patch_values=patch
    )
    clean = oracle_lib.next_token_logits(shim, base_inputs)
    assert not torch.allclose(want, clean, atol=1e-4)  # non-vacuity
    torch.testing.assert_close(have, want, **TOL)


# --------------------------------------------------------------------------- #
#  all positions — writes                                                      #
# --------------------------------------------------------------------------- #


def zero_ablate_all_doc() -> dict:
    """Zero-ablate a whole layer: the write case the all spelling makes
    expressible without naming every index."""
    return {
        "header": {"protocol_version": "4"},
        "model": {"key": "test", "revision": "main"},
        "data": base_data_section(with_counterfactual=False),
        "method": {
            "intervened_models": {
                "ablated": {"input": "base", "reads": ["logits"], "writes": ["zero"]}
            },
            "sites": {
                "tgt": {"component": "block_output", "layers": [0]},
                "lm_head": {"component": "lm_head"},
            },
            "reads": {"logits": {"site": "lm_head", "pos": {"index": -1}}},
            "writes": {"zero": {"site": "tgt", "pos": "all", "do": {"swap": 0.0}}},
            "save": [saved("logits", "ablated", "l.safetensors")],
        },
    }


def test_all_positions_write_matches_oracle(bundle, oracle: OracleShim):
    """An all-positions swap writes every content token — the oracle patch
    covers the whole sequence, and the result differs from clean."""
    executor = executor_for(zero_ablate_all_doc(), bundle, base_texts=[BASE_TEXT])
    have = executor.read_value("logits")[:, 0, :]

    base_inputs = _inputs(bundle, BASE_TEXT)
    seq = int(base_inputs["input_ids"].shape[1])
    resid = oracle_lib.capture_residual(oracle, 0, base_inputs)
    want = oracle_lib.next_token_logits(
        oracle,
        base_inputs,
        layer=0,
        positions=list(range(seq)),
        patch_values=torch.zeros_like(resid),
    )
    clean = oracle_lib.next_token_logits(oracle, base_inputs)
    assert not torch.allclose(want, clean, atol=1e-4)  # non-vacuity
    torch.testing.assert_close(have, want, **TOL)


def test_ragged_all_positions_write_refuses(llama_bundle, monkeypatch):
    """The documented v1 limit (spec §5.19): rows of unequal length make an
    `all` write ragged, which the engine refuses rather than silently writing
    a rectangle over the padding.

    It refuses **by rule number, before the forward pass**. It used to be a
    bare `NotImplementedError` raised while landing the write — i.e. on the
    accelerator after the weights had loaded, with no rule to look up. That is
    minutes of a 35 B load for a fact the encoded batch already knows, and it
    shaped a whole corpus: every request had to end in a `.` token so negative
    indices aligned across rows.
    """
    texts = ["one two", "a much longer sentence right here indeed and then some more"]
    batch = encode(llama_bundle.tokenizer, texts)
    assert batch.content_start(0) != batch.content_start(1)  # genuinely ragged

    executor = executor_for(zero_ablate_all_doc(), llama_bundle, base_texts=texts)

    def no_forward(*args, **kwargs):
        raise AssertionError("refused too late: a forward pass already ran")

    monkeypatch.setattr(type(executor), "_run_group", no_forward)

    with pytest.raises(ValidationError) as err:
        executor.read_value("logits")
    assert err.value.rule == 19
    assert "ragged position widths" in str(err.value)
    assert "all-positions or variable write" in str(err.value)
