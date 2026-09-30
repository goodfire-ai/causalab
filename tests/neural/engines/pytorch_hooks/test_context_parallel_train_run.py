"""Training parity under context parallelism through the real CLI
(``docs/model_parallelism.md`` §7, §8.4, §10.6): the corpus DAS fit
(``tests/protocols/04_das_im.json``, retargeted as ``test_train_parallel_run.py``
retargets it) at ``--parallel cp=2 --device cpu`` on the tiny Llama and on
``tiny-random/qwen3.5-moe``, and at ``cp=2,tp=2`` on the MoE, over ``gloo``,
against the world-1 fit of the same document.

**The write is at the weekday token** (`ENTITY` — the first position
where a base row and its counterfactual differ; before it the residuals
are identical and a swap is a no-op whose gradient is rounding noise) and
the loss at the last token. On the tiny Llama the frame is eleven positions
(``[0, 5)`` and ``[5, 11)`` at ``cp=2``) and the weekday sits at position 4
of the eleven-token row and 6 of the nine-token rows, so **that row's
gradient crosses the chunk boundary**: through the logits gathered at the
head (the tap pair — the loss on every rank is the same scalar over the same
whole tensor, its gradient this rank's chunk), the second chunk's queries
over the first chunk's keys and values (``gather_reduce_scatter``), and
back to the write through its fragment's all-gathered gradient, the same
tensor on every rank. On the MoE every row is seven tokens without a BOS
and the weekday is position 3 — the first position of the second chunk —
so no write on this data crosses at ``cp=2``; the DeltaNet handoffs still
run with their gradient (the state's gradient travels back to a chunk that
holds no parameter), the KV all-reduce and the fragment's all-gather run,
and the loop's agreements run for real: the out-of-memory ``any`` over the
model *and* context groups, the §7 guard's mean over the model group.

**The band.** What separates the parallel fit from the world-1 fit is the
forward's re-association (the attention GEMM over a shorter query axis, the
chunked kernel's blocks on the MoE) and the backward's — the key and value
gradients summed over the ranks' GEMMs — carried through the fp32 AdamW
updates. Measured on the fixtures in fp32 (torch 2.9.0, gloo, CPU, this
file's own program): `MEASURED`, the maximum absolute difference per
output class — the fitted bundle (absolute) and the saved tables (relative
to a magnitude of at least one). The band is pinned at `BAND`, at
least twenty times the largest measured maximum.

**Mutation**, applied in a ``torchrun``-style child before the CLI runs on
the Llama: the tap pair's backward at the KV gather (this rank's own chunk
in place of the all-reduce), which drops the eleven-token row's gradient —
its weekday's keys are attended from the other chunk — and lands the fit
outside the band. The DeltaNet handoff without its gradient is held by the
simulated tier (``test_context_parallel_train.py``), where a chunk boundary
can be placed before the state a write moves.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pytest
import torch.multiprocessing as mp

from causalab.cli import main
from causalab.neural.shared.parallel.spawn import reserve_port

from tests.neural.engines.pytorch_hooks.conftest import TINY_LLAMA, TINY_QWEN35_MOE
from tests.neural.engines.pytorch_hooks.test_train_parallel_run import (
    DAS,
    K,
    LAYER,
    _argv,
    _diffs,
    _mutated_entry,
    _receipt,
    never_load,
    offline,
)
from tests._helpers.parity_band import fit_band

__all__ = ["never_load", "offline"]

# the §7 gradient agreement check on at the measured band (conftest.py):
# under ``cp=2,tp=2`` the model group is two and the check runs for real
#
# `parallel_world`: every test here runs a document or a fit across a spawned
# multi-rank process world — minutes on the CI runner. The PR gate deselects the
# marker; the nightly CPU job runs it (docs/TESTS.md).
pytestmark = [
    pytest.mark.smoke,
    pytest.mark.parallel_world,
    pytest.mark.usefixtures("checked_gradients_gloo"),
]

#: The weekday token's index in every row of the fixture data under each
#: tokenizer (module docstring): after the Llama's BOS, ``If today is``;
#: the MoE adds no BOS.
ENTITY = {TINY_LLAMA: 4, TINY_QWEN35_MOE: 3}

#: The measured maxima (module docstring), per geometry and output class.
MEASURED: dict[str, dict[str, float]] = {
    "llama cp=2": {"bundle": 1.5e-8, "tables": 7.5e-9},
    "moe cp=2": {"bundle": 6.0e-8, "tables": 2.1e-7},
    "moe cp=2,tp=2": {"bundle": 1.5e-8, "tables": 1.8e-7},
}
#: At least twenty times the largest measured maximum; the mutation below
#: lands the Llama's bundle at ``3.7e-3`` (the tables at ``2.8e-5``), two
#: orders outside.
BAND = 1e-5
assert BAND >= 20 * max(max(v.values()) for v in MEASURED.values())


def _document(tmp: Path, key: str) -> Path:
    """The corpus DAS fit retargeted to ``key`` — its layer and its width,
    fp32 — with the read and the swap at the weekday token."""
    doc = json.loads(DAS.read_text())
    doc["model"] = {"key": key, "revision": "main", "dtype": "fp32"}
    method = doc["method"]
    method["sites"]["target"]["layers"] = [LAYER[key]]
    method["featurizers"]["rot"]["k"] = K[key]
    method["reads"]["v_cf"]["pos"] = ENTITY[key]
    method["writes"]["patch"]["pos"] = ENTITY[key]
    target = tmp / "das_entity.json"
    target.write_text(json.dumps(doc, indent=2))
    return target


def _block(**axes: int) -> dict[str, Any]:
    geometry = {"data": 1, "pipeline": 1, "context": 1, "tensor": 1, "expert": 1}
    geometry.update(axes)
    world = geometry["context"] * max(geometry["tensor"], geometry["expert"])
    return {**geometry, "data_mode": "points", "world": world, "launcher": "spawned"}


def _assert_parity(
    solo: Path, parallel: Path, block: dict[str, Any], label: str
) -> dict[str, float]:
    a, b = _receipt(solo), _receipt(parallel)
    assert b["execution"]["parallel"] == block
    assert a["execution"]["parallel"]["launcher"] == "solo"
    for key in ("fit_rows_resolved", "fit_rows_shrinks"):
        assert a["execution"].get(key) == b["execution"].get(key), key
    del a["execution"]["parallel"], b["execution"]["parallel"]
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    measured = _diffs(solo, parallel)
    print(label, measured)
    for kind, worst in measured.items():
        assert worst <= BAND, (label, kind, worst)
        # a drift past the record is a re-measurement, not a silent pass
        assert worst <= fit_band(MEASURED[label][kind]), (label, kind, worst)
    return measured


@pytest.fixture(autouse=True)
def experimental_context(monkeypatch: pytest.MonkeyPatch) -> None:
    """The tiny MoE is a hybrid tower: ``cp`` is refused on it unless waived
    (§8.4); the spawned ranks inherit the switch."""
    monkeypatch.setenv("CAUSALAB_EXPERIMENTAL_CONTEXT", "1")


@pytest.fixture(scope="module")
def llama_solo(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    tmp = tmp_path_factory.mktemp("cp-train-llama")
    document = _document(tmp, TINY_LLAMA)
    out = tmp / "solo"
    assert main(_argv(document, out)) == 0
    return document, out


@pytest.fixture(scope="module")
def moe_solo(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    tmp = tmp_path_factory.mktemp("cp-train-moe")
    document = _document(tmp, TINY_QWEN35_MOE)
    out = tmp / "solo"
    assert main(_argv(document, out)) == 0
    return document, out


def test_cp2_fit_on_the_llama_lands_within_the_band(
    llama_solo: tuple[Path, Path], tmp_path: Path, never_load: None
) -> None:
    document, solo = llama_solo
    out = tmp_path / "cp2"
    assert main(_argv(document, out, "--parallel", "cp=2")) == 0
    _assert_parity(solo, out, _block(context=2), "llama cp=2")


@pytest.mark.parametrize(
    "geometry, axes",
    [("cp=2", {"context": 2}), ("cp=2,tp=2", {"context": 2, "tensor": 2})],
)
def test_cp2_fit_on_the_moe_lands_within_the_band(
    moe_solo: tuple[Path, Path],
    tmp_path: Path,
    never_load: None,
    geometry: str,
    axes: dict[str, int],
) -> None:
    document, solo = moe_solo
    out = tmp_path / geometry
    assert main(_argv(document, out, "--parallel", geometry)) == 0
    _assert_parity(solo, out, _block(**axes), f"moe {geometry}")


# --------------------------------------------------------------------------- #
# the mutation, in a torchrun-style child
# --------------------------------------------------------------------------- #


def _own_chunk_kv() -> None:
    """The tap pair's gather at the KV gather: its backward keeps this rank's
    own queries' contribution and drops the other chunk's."""
    from causalab.neural.engines.pytorch_hooks import attention_interface

    def own_chunk(frame: Any, key: Any, value: Any, dtype: Any) -> tuple[Any, Any, Any]:
        return frame.gather(key, 2), frame.gather(value, 2), frame.attention_mask(dtype)

    attention_interface._gather_kv = own_chunk  # type: ignore[assignment]  # pyright: ignore[reportPrivateUsage]


MUTATIONS: dict[str, Callable[[], None]] = {"own_chunk_kv": _own_chunk_kv}


def _mutated(rank: int, argv: list[str], world: int, port: int, mutation: str) -> None:
    from tests.neural.engines.pytorch_hooks import test_train_parallel_run as sibling

    sibling.MUTATIONS[mutation] = MUTATIONS[mutation]
    _mutated_entry(rank, argv, world, port, mutation)


def test_the_tap_pairs_backward_at_the_kv_gather_leaves_the_band_on_the_llama(
    llama_solo: tuple[Path, Path], tmp_path: Path
) -> None:
    document, solo = llama_solo
    out = tmp_path / "own_chunk"
    with reserve_port() as hold:
        mp.spawn(
            _mutated,
            args=(
                _argv(document, out, "--parallel", "cp=2"),
                2,
                hold.port,
                "own_chunk_kv",
            ),
            nprocs=2,
            join=True,
        )
    diffs = _diffs(solo, out)
    print("llama cp=2 own-chunk kv", diffs)
    assert diffs["bundle"] > BAND, diffs
