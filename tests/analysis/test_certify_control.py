"""``causalab.analysis.certify_control`` — the certification legs of a
``self_swap`` control (``docs/workflow_protocol.md`` §2.2). A bit-exact no-op is
**necessary but insufficient**.

**T5, the four-leg bar.** The self-swap twin of corpus document 02
(``tests/protocols/02_interchange_im.json``; the twin is
``tests/workflow/fixtures/controls/self_swap_twin.json``) is run on both tiny
fixtures through the real CLI and its five saved reads are handed to the
script the way the runner hands them: (a) the *valid* twin — interchange
beside its no-op — passes all three legs: identity **bit-exact**
(``torch.equal``), a positive sender effect, a changed receiver; (b) the
*vacuous* variant, the operand read on ``base`` (a self-swap that
"passes for free"), passes leg (i) alone with legs (ii) and (iii) at exactly
``0.0`` and is ``failed``; (c) the *receiver-unchanged* variant — the write at
the last layer, the receiver read at the layer above it, legal under rule 21
of ``docs/intervention_protocol.md`` §5 because the operand is read at the write's own depth — has a positive
sender effect and a receiver the write cannot reach, so leg (iii) alone fails
it. That third case is the **mutation witness**: a script that dropped leg
(iii) would certify it. Leg (iv), agreement with an independent oracle, is
checked **here** against ``hook_oracle_lib`` at the oracle suites' tolerance
and nowhere in shipped code, as ``docs/workflow_protocol.md`` §2.2 states.

Without the change nothing under ``causalab.analysis`` decides any of this:
the module does not exist and every test here fails at import.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
import torch

from causalab.analysis import certify_control
from causalab.io.step_io import StepError, write_tensor
from tests.neural.engines.pytorch_hooks.conftest import TINY_GPT2, TINY_LLAMA
from tests.protocol._env import FIXTURES
from tests.step_scripts import run_step

TWIN = (
    Path(__file__).resolve().parents[1]
    / "workflow/fixtures/controls/self_swap_twin.json"
)
TOL = dict(atol=1e-5, rtol=1e-4)  # the write oracle's own bar (test_write_oracle.py)


def _twin(
    model: str,
    *,
    operand_input: str = "counterfactual",
    receiver_layer: int | None = None,
) -> dict[str, Any]:
    """The fixture twin, retargeted: ``operand_input`` is the target's operand
    role (``base`` makes it vacuous); ``receiver_layer`` moves the three
    receiver reads from ``lm_head`` to a ``block_output`` *above* the write."""
    raw = json.loads(TWIN.read_text())
    raw["model"]["key"] = model
    method = raw["method"]
    method["sites"]["target"]["layers"] = 1  # tiny-random is two layers deep
    # the operand read is measured on the un-intervened model of its role
    method["intervened_models"]["original_counterfactual"]["input"] = operand_input
    if receiver_layer is not None:
        method["sites"]["receiver"] = {
            "component": "block_output",
            "layers": [receiver_layer],
        }
        del method["sites"]["lm_head"]
        for name in ("recv_target", "recv_control", "recv_original"):
            method["reads"][name]["site"] = "receiver"
    return raw


def _run(raw: dict[str, Any], tmp_path: Path) -> Path:
    from causalab.cli import main

    document = tmp_path / "twin.json"
    document.write_text(json.dumps(raw))
    artifacts = tmp_path / "artifacts"
    shutil.copytree(FIXTURES / "artifacts", artifacts, dirs_exist_ok=True)
    out = tmp_path / "run"
    code = main(
        [
            "run",
            "--engine",
            "auto",
            str(document),
            "--data-root",
            str(FIXTURES / "data"),
            "--artifacts-root",
            str(artifacts),
            "--out",
            str(out),
        ]
    )
    assert code == 0
    return out


def _certify(out: Path, **extra: Any) -> list[dict[str, Any]]:
    inputs: dict[str, Any] = {
        name: out / f"{name}.safetensors" for name in certify_control.REQUIRED_INPUTS
    }
    inputs.update(extra)
    run_step(certify_control, inputs, {"controls": out / "controls.json"})
    return json.loads((out / "controls.json").read_text())


# --------------------------------------------------------------------------- #
# T5 — on the tiny fixtures
# --------------------------------------------------------------------------- #


@pytest.mark.smoke
@pytest.mark.parametrize("model", [TINY_LLAMA, TINY_GPT2], ids=["llama", "gpt2"])
def test_t5_the_valid_twin_passes_all_three_legs(model: str, tmp_path: Path) -> None:
    """The interchange beside its no-op: identity bit-exact, sender effect and
    receiver change both above the floor. Fails without the change: no
    module, no legs, no ``controls.json``."""
    (row,) = _certify(_run(_twin(model), tmp_path))
    assert row["status"] == "passed"
    assert row["identity_exact"] is True and row["identity_max_abs"] == 0.0
    assert row["sender_effect"] > row["atol"] and row["receiver_change"] > row["atol"]
    assert row["coords"] == {} and row["label"] == ""
    assert row["kind"] == "self_swap" and row["seam"] is None


@pytest.mark.smoke
@pytest.mark.parametrize("model", [TINY_LLAMA, TINY_GPT2], ids=["llama", "gpt2"])
def test_t5_the_vacuous_self_swap_passes_identity_alone_and_fails(
    model: str, tmp_path: Path
) -> None:
    """The self-swap that passes for free: the target's operand is its
    own ``base`` value, so leg (i) is exact and legs (ii)/(iii) are exactly
    zero — the point is ``failed``. A bit-exact no-op is necessary, not
    sufficient. Fails without the change: a certifier reading identity alone
    would pass it."""
    (row,) = _certify(_run(_twin(model, operand_input="base"), tmp_path))
    assert row["status"] == "failed"
    assert row["identity_exact"] is True and row["identity_max_abs"] == 0.0
    assert row["sender_effect"] == 0.0 and row["receiver_change"] == 0.0


@pytest.mark.smoke
@pytest.mark.parametrize("model", [TINY_LLAMA, TINY_GPT2], ids=["llama", "gpt2"])
def test_t5_mutation_a_receiver_unchanged_intervention_fails_leg_iii_alone(
    model: str, tmp_path: Path
) -> None:
    """The write lands at the last layer, the receiver is read one layer
    *above* it (rule 21 admits the operand at equal depth): the sender effect
    is positive, the receiver cannot change. Only leg (iii) fails it — so a
    script that dropped leg (iii) would certify a receiver-unchanged
    intervention, and this test would fail."""
    (row,) = _certify(_run(_twin(model, receiver_layer=0), tmp_path))
    assert row["status"] == "failed"
    assert row["identity_exact"] is True
    assert row["sender_effect"] > row["atol"]
    assert row["receiver_change"] == 0.0


@pytest.mark.smoke
def test_t5_leg_iv_the_saved_receivers_agree_with_the_raw_hook_oracle(
    tmp_path: Path,
) -> None:
    """Leg (iv), test-side: the receiver under ``original`` the engine saved
    equals the raw-hook oracle's clean next-token logits for each row, and
    the receiver under the *target* equals the oracle's patched forward
    (counterfactual residual swapped in at the write address), at the write
    oracle's tolerance. No shipped code claims this leg."""
    from causalab.neural.engines.pytorch_hooks.loading import load_model
    from causalab.neural.shared.encoding import encode
    from safetensors.torch import load_file
    from tests.neural.engines.pytorch_hooks import hook_oracle_lib as oracle_lib
    from tests.neural.engines.pytorch_hooks.conftest import OracleShim

    out = _run(_twin(TINY_LLAMA), tmp_path)
    original = load_file(str(out / "receiver_original.safetensors"))["recv_original"]
    target = load_file(str(out / "receiver_target.safetensors"))["recv_target"]
    rows = [
        row
        for row in json.loads(
            (FIXTURES / "data" / "weekdays" / "data.json").read_text()
        )
        if row["split"] == "train"
    ]
    assert len(rows) == original.shape[0]
    bundle = load_model(TINY_LLAMA)
    shim = OracleShim(hf_model=bundle.model)

    def inputs(text: str) -> dict[str, Any]:
        batch = encode(bundle.tokenizer, [text])
        return {"input_ids": batch.input_ids, "attention_mask": batch.attention_mask}

    for index, row in enumerate(rows):
        base = inputs(row["input"])
        clean = oracle_lib.next_token_logits(shim, base).reshape(1, -1)
        torch.testing.assert_close(original[index], clean, **TOL)  # (rows, 1, vocab)
        cf_resid = oracle_lib.capture_residual(
            shim, 1, inputs(row["counterfactual_inputs"][0])
        )
        last = base["input_ids"].shape[1] - 1
        want = oracle_lib.next_token_logits(
            shim, base, layer=1, positions=[last], patch_values=cf_resid[:, -1:, :]
        ).reshape(1, -1)
        torch.testing.assert_close(target[index], want, **TOL)


# --------------------------------------------------------------------------- #
# the contract, without a model
# --------------------------------------------------------------------------- #


def _bundle(path: Path, tensor: torch.Tensor, slot: str) -> Path:
    write_tensor(path, tensor, slot=slot)
    return path


def _five(
    tmp_path: Path,
    *,
    receiver: torch.Tensor,
    control: torch.Tensor,
    target: torch.Tensor,
    operand: torch.Tensor,
    overwritten: torch.Tensor,
) -> dict[str, Any]:
    return {
        "receiver_original": _bundle(
            tmp_path / "ro.safetensors", receiver, "recv_original"
        ),
        "receiver_control": _bundle(
            tmp_path / "rc.safetensors", control, "recv_control"
        ),
        "receiver_target": _bundle(tmp_path / "rt.safetensors", target, "recv_target"),
        "operand": _bundle(tmp_path / "op.safetensors", operand, "v_cf"),
        "overwritten": _bundle(tmp_path / "ow.safetensors", overwritten, "v_self"),
    }


@pytest.mark.unit
def test_a_missing_input_is_refused_naming_the_five() -> None:
    with pytest.raises(StepError, match="are required"):
        run_step(
            certify_control, {"receiver_original": Path("x")}, {"controls": Path("c")}
        )


@pytest.mark.unit
def test_a_selected_tensor_input_is_refused(tmp_path: Path) -> None:
    """The runner hands an ``entry``-selected input over as a tensor; the legs
    pair entries by the header's coordinates, so the bundle itself is required."""
    torch.manual_seed(0)
    x = torch.randn(2, 4)
    inputs = _five(
        tmp_path, receiver=x, control=x, target=x + 1, operand=x, overwritten=x - 1
    )
    inputs["operand"] = x
    with pytest.raises(StepError, match="must reference the saved bundle itself"):
        run_step(certify_control, inputs, {"controls": tmp_path / "c.json"})


@pytest.mark.unit
def test_a_shape_mismatch_is_refused(tmp_path: Path) -> None:
    x = torch.zeros(2, 4)
    inputs = _five(
        tmp_path,
        receiver=x,
        control=x,
        target=x,
        operand=x,
        overwritten=torch.zeros(2, 3),
    )
    with pytest.raises(StepError, match="differ in shape"):
        run_step(certify_control, inputs, {"controls": tmp_path / "c.json"})


@pytest.mark.unit
def test_entries_that_pair_with_nothing_are_refused(tmp_path: Path) -> None:
    """A swept bundle beside an un-swept one shares no coordinates — refused,
    never silently paired by order."""
    x = torch.zeros(2, 4)
    inputs = _five(tmp_path, receiver=x, control=x, target=x, operand=x, overwritten=x)
    from safetensors.torch import save_file

    save_file({"v_cf[rot.seed=0]": x}, str(tmp_path / "op.safetensors"))
    with pytest.raises(StepError, match="pairs with nothing"):
        run_step(certify_control, inputs, {"controls": tmp_path / "c.json"})


@pytest.mark.unit
def test_the_floor_is_the_campaigns_to_raise_and_the_rows_repeat_the_declaration(
    tmp_path: Path,
) -> None:
    """A sender effect of 1e-3 is positive at the default floor and vacuous at
    an authored ``atol`` of 1e-2; the rows carry the ``kind`` and ``seam`` the
    runner hands in under ``control``."""
    torch.manual_seed(0)
    x = torch.randn(3, 5)
    inputs = _five(
        tmp_path,
        receiver=x,
        control=x.clone(),
        target=x + 1.0,
        operand=x + 1e-3,
        overwritten=x,
    )
    (row,) = _rows(
        inputs, tmp_path, control={"step": "ctl", "kind": "self_swap", "seam": "B"}
    )
    assert (
        row["status"] == "passed" and row["seam"] == "B" and row["kind"] == "self_swap"
    )
    assert row["sender_effect"] == pytest.approx(1e-3, rel=1e-3)
    (row,) = _rows(inputs, tmp_path, atol=1e-2)
    assert row["status"] == "failed"
    with pytest.raises(StepError, match="'atol' must be a non-negative number"):
        _rows(inputs, tmp_path, atol=True)


@pytest.mark.unit
def test_identity_is_bit_exact_not_close(tmp_path: Path) -> None:
    """One ulp on the control receiver fails leg (i): ``torch.equal``, never
    ``allclose`` — the no-op leg is exact by definition."""
    torch.manual_seed(0)
    x = torch.randn(2, 4)
    nudged = x.clone()
    nudged[0, 0] = torch.nextafter(nudged[0, 0], torch.tensor(float("inf")))
    inputs = _five(
        tmp_path, receiver=x, control=nudged, target=x + 1, operand=x + 1, overwritten=x
    )
    (row,) = _rows(inputs, tmp_path)
    assert row["identity_exact"] is False and row["status"] == "failed"
    assert 0 < row["identity_max_abs"] < 1e-6


def _rows(inputs: dict[str, Any], tmp_path: Path, **extra: Any) -> list[dict[str, Any]]:
    run_step(
        certify_control, {**inputs, **extra}, {"controls": tmp_path / "controls.json"}
    )
    return json.loads((tmp_path / "controls.json").read_text())
