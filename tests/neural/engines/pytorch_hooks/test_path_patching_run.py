"""Path patching in the hand form, executed on the tiny fixtures.

Path patching is authored as ``sites``, ``reads``, ``writes`` and
``intervened_models``. The shipped ``configs/protocols/path_patching.json``
is written this way, and each document in this file is written out in full
the same way. Two properties of that form are asserted here.

**One joint pass.** Two receivers are two absolute swaps in the one
``final`` intervened model. The receipt's ``fires`` block lists both writes
under one forward-group label. The joint effect on the metric differs from
the sum of the two single-receiver effects. The receivers nest: the block
input of layer 1 and the MLP output of the same layer. With the block input
injected, the MLP recomputes to the harvested value, so the joint pass equals
the upstream-only run and a sum of separate runs counts the downstream
effect twice. Each half is asserted on its own, so a mutation that emits one
intervened model per receiver and adds the effects afterwards fails. A
tiny-random model is close to linear. Nothing here claims a nonlinearity,
which is why the sender is a residual site and the tolerance comes from the
probed gap.

**The freeze set changes the numbers.** On the five-layer GPT-2 fixture a
sender at layer 0 and a receiver at layer 2 leave one layer between them.
Freezing the attention output alone and freezing attention and MLP outputs
run to different logits and different digests. Dropping the ``freeze_m*``
writes collapses the two.

The corpus document ``03_path_patching_im.json`` runs in
``test_run_corpus.py::test_03_path_patching_runs`` and
``test_write_set_fires.py``.
"""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from safetensors.torch import load_file

from causalab.cli import register_model_key
from causalab.io.env import FileArtifacts, FileDatasets, ResolutionEnv
from causalab.neural.engines.pytorch_hooks.engine import PytorchHooksEngine
from causalab.neural.shared.fires import group_label
from causalab.protocol import RUN_RECORD_NAME, run_protocol
from causalab.protocol.receipt import FIRES_KEY
from causalab.protocol.registry import get_model_info

from tests.neural.engines.pytorch_hooks.conftest import TINY_GPT2, TINY_LLAMA
from tests.protocol._env import FIXTURES
from tests.tables import frame as table_frame

pytestmark = pytest.mark.smoke

#: The joint-versus-sum gap below which the one-joint-pass mutation would
#: pass vacuously: a hundred times the fp32 noise of a tiny-random logit
#: difference (~1e-7), and a fifteenth of the gap the fixture shows (~1.6e-4,
#: probed).
GAP = 1e-5

#: The readout every document below shares: ``logits`` in the injection model
#: and a clean twin on ``original_base``, both at the last position.
LM_HEAD_SITE: dict[str, Any] = {"component": "lm_head"}
LOGITS_READS: dict[str, Any] = {
    "logits": {"site": "lm_head", "pos": -1},
    "logits_clean": {"site": "lm_head", "pos": -1},
}
#: The model that takes each readout read.
LOGITS_MODELS: dict[str, str] = {"logits": "final", "logits_clean": "original_base"}

#: The shipped preset's ``logit_diff`` aggregation.
LOGIT_DIFF: dict[str, Any] = {"kind": "logit_diff", "a": "answer", "b": "cf_answer"}

#: Two ``logit_diff`` tables, one on each readout.
METRIC_READOUT: list[dict[str, Any]] = [
    {
        "read": "logits",
        "model": "final",
        "aggregation": LOGIT_DIFF,
        "file_path": "logit_diff.json",
    },
    {
        "read": "logits_clean",
        "model": "original_base",
        "aggregation": LOGIT_DIFF,
        "file_path": "ld_clean.json",
    },
]

#: The raw logits of both readouts.
LOGITS_READOUT: list[dict[str, Any]] = [
    {"read": "logits", "model": "final", "file_path": "logits.safetensors"},
    {
        "read": "logits_clean",
        "model": "original_base",
        "file_path": "logits_clean.safetensors",
    },
]


def _env(tmp_path: Path) -> ResolutionEnv:
    artifacts = tmp_path / "artifacts"
    shutil.copytree(FIXTURES / "artifacts", artifacts, dirs_exist_ok=True)
    for key in (TINY_LLAMA, TINY_GPT2):
        register_model_key({"model": {"key": key, "revision": "main"}})
    return ResolutionEnv(
        datasets=FileDatasets(root=FIXTURES / "data"),
        artifacts=FileArtifacts(root=artifacts),
    )


def _document(
    model_key: str, method: dict[str, Any], readout: list[dict[str, Any]]
) -> dict[str, Any]:
    """The envelope around one hand-written intervention over the IOI fixture
    rows: the header, the model, the two data roles, and the shared readout
    merged into the method's tables."""
    sites = {**method["sites"], "lm_head": LM_HEAD_SITE}
    reads = {**method["reads"], **LOGITS_READS}
    models = copy.deepcopy(method["intervened_models"])
    for read, model in LOGITS_MODELS.items():
        # the clean readout declares the un-intervened model on base (§2.9)
        models.setdefault(model, {"input": "base", "reads": []})
        models[model].setdefault("reads", []).append(read)
    return {
        "header": {"protocol_version": "4"},
        "model": {"key": model_key, "revision": "main", "dtype": "fp32"},
        "data": {
            "base": {"dataset": "ioi/test", "field": "input"},
            "counterfactual": {
                "dataset": "ioi/test",
                "field": "counterfactual_inputs[0]",
            },
        },
        "method": {
            "sites": sites,
            "reads": reads,
            "writes": method["writes"],
            "intervened_models": models,
            "save": readout,
        },
    }


def _run(doc: dict[str, Any], env: ResolutionEnv, out: Path) -> dict[str, Any]:
    run_protocol(doc, env, PytorchHooksEngine(), out, record=True)
    return json.loads((out / RUN_RECORD_NAME).read_text())


def _values(out: Path, name: str) -> np.ndarray:
    return table_frame(out / name)["value"].to_numpy(dtype=float)


def _fires(receipt: dict[str, Any]) -> dict[str, dict[str, int]]:
    (point,) = receipt["points"]
    return receipt[FIRES_KEY][point["digest"]]


# --------------------------------------------------------------------------- #
# one joint pass, not a sum
# --------------------------------------------------------------------------- #

#: Sender ``block_output@0`` swapped to its counterfactual value in ``patched``.
#: Adjacent layers leave nothing to restore.
SENDER_L0: dict[str, Any] = {
    "sites": {"sender": {"component": "block_output", "layers": [0]}},
    "reads": {"v_sender": {"site": "sender", "pos": -1}},
    "writes": {
        "swap_sender": {"site": "sender", "pos": -1, "do": {"swap": "v_sender"}}
    },
    "intervened_models": {
        "original_counterfactual": {"input": "counterfactual", "reads": ["v_sender"]},
    },
}

#: Both receivers of layer 1 injected together in ``final``.
JOINT: dict[str, Any] = {
    "sites": {
        **SENDER_L0["sites"],
        "receiver_0": {"component": "block_input", "layers": [1]},
        "receiver_1": {"component": "mlp_output", "layers": [1]},
    },
    "reads": {
        **SENDER_L0["reads"],
        "v_receiver_0": {"site": "receiver_0", "pos": -1},
        "v_receiver_1": {"site": "receiver_1", "pos": -1},
    },
    "writes": {
        **SENDER_L0["writes"],
        "inject_0": {"site": "receiver_0", "pos": -1, "do": {"swap": "v_receiver_0"}},
        "inject_1": {"site": "receiver_1", "pos": -1, "do": {"swap": "v_receiver_1"}},
    },
    "intervened_models": {
        **SENDER_L0["intervened_models"],
        "patched": {
            "input": "base",
            "reads": ["v_receiver_0", "v_receiver_1"],
            "writes": ["swap_sender"],
        },
        "final": {"input": "base", "writes": ["inject_0", "inject_1"]},
    },
}

#: The upstream receiver alone: ``block_input@1``.
UPSTREAM_ONLY: dict[str, Any] = {
    "sites": {
        **SENDER_L0["sites"],
        "receiver": {"component": "block_input", "layers": [1]},
    },
    "reads": {
        **SENDER_L0["reads"],
        "v_receiver": {"site": "receiver", "pos": -1},
    },
    "writes": {
        **SENDER_L0["writes"],
        "inject": {"site": "receiver", "pos": -1, "do": {"swap": "v_receiver"}},
    },
    "intervened_models": {
        **SENDER_L0["intervened_models"],
        "patched": {
            "input": "base",
            "reads": ["v_receiver"],
            "writes": ["swap_sender"],
        },
        "final": {"input": "base", "writes": ["inject"]},
    },
}

#: The downstream receiver alone: ``mlp_output@1``.
DOWNSTREAM_ONLY: dict[str, Any] = {
    "sites": {
        **SENDER_L0["sites"],
        "receiver": {"component": "mlp_output", "layers": [1]},
    },
    "reads": UPSTREAM_ONLY["reads"],
    "writes": UPSTREAM_ONLY["writes"],
    "intervened_models": UPSTREAM_ONLY["intervened_models"],
}


def test_two_receivers_are_one_joint_pass_and_not_a_sum(tmp_path: Path) -> None:
    """The two receivers nest: ``mlp_output@1`` is downstream of
    ``block_input@1`` on the same path. With the block input injected the MLP
    recomputes to the harvested value, so the joint pass equals the
    upstream-only run and a sum of separate runs counts the downstream
    receiver twice. Mutation: one intervened model per receiver (``final_0``,
    ``final_1``) with the two effects added. The fires block then shows two
    ``final`` groups and the joint number equals the sum, so both halves
    below fail."""
    env = _env(tmp_path)
    joint = _run(_document(TINY_LLAMA, JOINT, METRIC_READOUT), env, tmp_path / "joint")
    only_upstream = _run(
        _document(TINY_LLAMA, UPSTREAM_ONLY, METRIC_READOUT), env, tmp_path / "upstream"
    )
    only_downstream = _run(
        _document(TINY_LLAMA, DOWNSTREAM_ONLY, METRIC_READOUT),
        env,
        tmp_path / "downstream",
    )

    # anti-vacuity, structural: the receivers' layer exists in the model, so
    # the downstream receiver is recomputed from the injected upstream one
    assert (
        JOINT["sites"]["receiver_0"]["layers"][0]
        < get_model_info(TINY_LLAMA).num_layers
    )

    # one forward group carries both injections, each firing once; the
    # patched model is the sender swap alone
    fires = _fires(joint)
    assert fires == {
        group_label("patched", "base"): {"swap_sender": 1},
        group_label("final", "base"): {"inject_0": 1, "inject_1": 1},
    }
    assert [g for g in fires if g.startswith("final")] == [group_label("final", "base")]
    for single in (only_upstream, only_downstream):
        assert _fires(single) == {
            group_label("patched", "base"): {"swap_sender": 1},
            group_label("final", "base"): {"inject": 1},
        }

    # the joint effect is not the sum of the separate effects
    clean = _values(tmp_path / "joint", "ld_clean.json")
    np.testing.assert_allclose(
        _values(tmp_path / "upstream", "ld_clean.json"), clean, atol=1e-6
    )
    joint_value = _values(tmp_path / "joint", "logit_diff.json")
    up = _values(tmp_path / "upstream", "logit_diff.json")
    down = _values(tmp_path / "downstream", "logit_diff.json")
    summed = up + down - clean
    gap = float(np.max(np.abs(joint_value - summed)))
    assert gap > GAP, f"joint {joint_value} vs sum-of-separate {summed}: gap {gap}"
    # anti-vacuity, numerical: each receiver has an effect of its own, and the
    # nesting prediction holds: the joint pass is the upstream run
    assert float(np.max(np.abs(down - clean))) > GAP
    assert float(np.max(np.abs(up - clean))) > GAP
    np.testing.assert_allclose(joint_value, up, atol=1e-6)


# --------------------------------------------------------------------------- #
# the freeze set changes the numbers
# --------------------------------------------------------------------------- #

#: Sender ``attention_premix@0`` head 1, receiver ``block_input@2``, and the
#: attention output of layer 1 swapped back to its clean value.
FREEZE_ATTENTION: dict[str, Any] = {
    "sites": {
        "sender": {"component": "attention_premix", "layers": [0], "head": 1},
        "receiver": {"component": "block_input", "layers": [2]},
        "a1": {"component": "attention_output", "layers": [1]},
    },
    "reads": {
        "v_sender": {"site": "sender", "pos": -1},
        "v_a1": {"site": "a1", "pos": -1},
        "v_receiver": {"site": "receiver", "pos": -1},
    },
    "writes": {
        "swap_sender": {"site": "sender", "pos": -1, "do": {"swap": "v_sender"}},
        "freeze_1": {"site": "a1", "pos": -1, "do": {"swap": "v_a1"}},
        "inject": {"site": "receiver", "pos": -1, "do": {"swap": "v_receiver"}},
    },
    "intervened_models": {
        "original_counterfactual": {"input": "counterfactual", "reads": ["v_sender"]},
        "original_base": {"input": "base", "reads": ["v_a1"]},
        "patched": {
            "input": "base",
            "reads": ["v_receiver"],
            "writes": ["swap_sender", "freeze_1"],
        },
        "final": {"input": "base", "writes": ["inject"]},
    },
}

#: The same path with the MLP outputs of layers 0 and 1 frozen as well.
FREEZE_ATTENTION_AND_MLP: dict[str, Any] = {
    "sites": {
        **FREEZE_ATTENTION["sites"],
        "m0": {"component": "mlp_output", "layers": [0]},
        "m1": {"component": "mlp_output", "layers": [1]},
    },
    "reads": {
        **FREEZE_ATTENTION["reads"],
        "v_m0": {"site": "m0", "pos": -1},
        "v_m1": {"site": "m1", "pos": -1},
    },
    "writes": {
        **FREEZE_ATTENTION["writes"],
        "freeze_m0": {"site": "m0", "pos": -1, "do": {"swap": "v_m0"}},
        "freeze_m1": {"site": "m1", "pos": -1, "do": {"swap": "v_m1"}},
    },
    "intervened_models": {
        "original_counterfactual": {"input": "counterfactual", "reads": ["v_sender"]},
        "original_base": {"input": "base", "reads": ["v_a1", "v_m0", "v_m1"]},
        "patched": {
            "input": "base",
            "reads": ["v_receiver"],
            "writes": ["swap_sender", "freeze_m0", "freeze_1", "freeze_m1"],
        },
        "final": {"input": "base", "writes": ["inject"]},
    },
}


def test_the_freeze_set_changes_the_numbers(tmp_path: Path) -> None:
    """On the five-layer GPT-2 fixture the two freeze sets run to different
    logits and different digests. Mutation: drop the ``freeze_m*`` writes
    from the second document. The two runs then execute the same write set
    and the logits coincide."""
    env = _env(tmp_path)
    assert get_model_info(TINY_GPT2).num_layers == 5
    attention = _run(
        _document(TINY_GPT2, FREEZE_ATTENTION, LOGITS_READOUT),
        env,
        tmp_path / "attention",
    )
    both = _run(
        _document(TINY_GPT2, FREEZE_ATTENTION_AND_MLP, LOGITS_READOUT),
        env,
        tmp_path / "both",
    )

    assert _fires(attention)[group_label("patched", "base")] == {
        "swap_sender": 1,
        "freeze_1": 1,
    }
    assert _fires(both)[group_label("patched", "base")] == {
        "swap_sender": 1,
        "freeze_1": 1,
        "freeze_m0": 1,
        "freeze_m1": 1,
    }
    # different estimand identities: the point digest is over the write set
    assert attention["points"][0]["digest"] != both["points"][0]["digest"]
    assert attention["document_digest"] != both["document_digest"]

    # different numbers
    la = load_file(str(tmp_path / "attention" / "logits.safetensors"))
    lb = load_file(str(tmp_path / "both" / "logits.safetensors"))
    assert set(la) == set(lb) and la
    assert any(not torch.allclose(la[key], lb[key], atol=1e-5) for key in la)
    # and both patched something: neither equals the clean logits (each saved
    # bundle carries its read's one tensor under the read's own name)
    (clean,) = load_file(
        str(tmp_path / "attention" / "logits_clean.safetensors")
    ).values()
    for patched in (la, lb):
        (tensor,) = patched.values()
        assert not torch.allclose(tensor, clean, atol=1e-5)
