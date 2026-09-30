"""Data parallelism over rows through the real CLI (``docs/model_parallelism.md``
§8.3, §9): ``--parallel dp=2:rows --device cpu`` on the corpus DAS document
retargeted to the tiny Llama.

The document (``tests/protocols/04_das_im.json``: one point, a Cayley
subspace fit over the four ``weekdays`` training rows with an eval on the two
test rows and an early stop) runs once at world 1 and once at ``dp=2:rows``
under ``gloo``: the parent spawns two children, each runs the whole point,
each optimizer step's four-row minibatch is split two rows to a replica, and
the joiner alone writes. The fitted bundle lands within the pinned training
band of the world-1 run (``test_train_rows.py``: ``2.5e-7`` relative to a
magnitude of at least one — four fp32 ulps; the rotation's entries are at
most one, the saved cross-entropies of order ten) and so do the saved metric
tables — the tables differ by one ulp of a value near ten; the receipt records
``"data": 2, "data_mode": "rows"`` and ``"launcher": "spawned"``. A second
run hands every replica a publisher that publishes — the debug hook — and
the two replicas' would-be outputs are **byte-identical**: the summed
gradient is one tensor on both. A document with no ``train`` is refused by
the children before any weights load.
"""

from __future__ import annotations

import dataclasses
import functools
import json
from pathlib import Path
from typing import Any, Sequence, TypeVar

import pytest
from safetensors.torch import load_file

from causalab.cli import main
from causalab.neural.engines.pytorch_hooks import engine as hooks_engine
from causalab.neural.shared.parallel.collective import Collective
from causalab.protocol.parallel import ParallelGeometry
from causalab.io.env import FileArtifacts, FileDatasets, ResolutionEnv
from causalab.protocol.pipeline import run_protocol
from causalab.tasks import TASKS_ROOT

from tests._helpers.gloo_world import GlooWorld
from tests.tables import frame as table_frame

# the §7 gradient agreement check on at the measured band (conftest.py):
# the model group is one here, so the check is the identity — set all
# the same, so every training smoke runs under one rule
pytestmark = [pytest.mark.smoke, pytest.mark.usefixtures("checked_gradients_gloo")]

REPO = Path(__file__).resolve().parents[4]
DAS = REPO / "tests" / "protocols" / "04_das_im.json"
SCAN = REPO / "tests" / "workflow" / "fixtures" / "fan_out" / "protocols" / "scan.json"
DATA = REPO / "tests" / "protocol" / "fixtures" / "data"
TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"
TINY_REVISION = "9fb191250dd56d0ba7ec9785a025ed29c03d5998"

GEOMETRY = ParallelGeometry(data=2, data_mode="rows")
#: the training band, measured and pinned in ``test_train_rows.py``
BAND = 2.5e-7
TABLES = ("iia.json", "ce.json")
TENSOR = "rot.safetensors"
RECEIPT = "protocol.json"
EVENTS = "events.jsonl"

T = TypeVar("T")


def _document(tmp: Path) -> Path:
    """The corpus DAS fit on tiny Llama's first layer, fp32."""
    doc = json.loads(DAS.read_text())
    doc["model"] = {"key": TINY, "revision": TINY_REVISION, "dtype": "fp32"}
    doc["method"]["sites"]["target"]["layers"] = [0]
    target = tmp / "das.json"
    target.write_text(json.dumps(doc, indent=2))
    return target


def _argv(document: Path, out: Path, *extra: str, device: str = "cpu") -> list[str]:
    """The CLI ``run`` of ``document`` into ``out`` on ``device`` (``cpu``: the
    gloo tier; the CUDA twin, ``tests/golden/test_parallel_worlds.py``, passes
    ``cuda``)."""
    return [
        "run",
        str(document),
        "--engine",
        "pytorch_hooks",
        "--record",
        "--data-root",
        str(DATA),
        "--artifacts-root",
        str(document.parent),
        "--out",
        str(out),
        "--device",
        device,
        # Set the row budget explicitly: free memory can differ across ranks,
        # but the receipts must compare exactly.
        "--fit-rows",
        "16",
        *extra,
    ]


def _receipt(out: Path) -> dict[str, Any]:
    return json.loads((out / RECEIPT).read_text())


def _values(table: Path) -> list[float]:
    return [float(v) for v in table_frame(table)["value"]]


def _assert_within_band(solo: Path, other: Path) -> None:
    # the outputs alone: the receipt and the event stream are the recorded
    # run's sidecars (`--record`), not what the engine published
    assert sorted(
        p.name for p in solo.iterdir() if p.name not in (RECEIPT, EVENTS)
    ) == sorted(p.name for p in other.iterdir() if p.name not in (RECEIPT, EVENTS))
    a, b = load_file(str(solo / TENSOR)), load_file(str(other / TENSOR))
    assert set(a) == set(b)
    for name in a:
        # a rotation's entries are at most one: the band is absolute here
        assert float((a[name] - b[name]).abs().max()) <= BAND, name
    for name in TABLES:
        # a cross-entropy of order ten: the band relative to its magnitude
        for x, y in zip(_values(solo / name), _values(other / name), strict=True):
            assert abs(x - y) <= BAND * max(1.0, abs(x)), name


@pytest.fixture
def never_load(monkeypatch: pytest.MonkeyPatch) -> None:
    """The parent of a spawn never loads a model: in *this* process the
    loader raises; the children are fresh interpreters and load normally."""

    def never(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the spawn parent entered load_model")

    monkeypatch.setattr(hooks_engine, "load_model", never)


@pytest.fixture(scope="module")
def solo_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    tmp = tmp_path_factory.mktemp("rows-solo")
    document = _document(tmp)
    out = tmp / "solo"
    assert main(_argv(document, out)) == 0
    return document, out


def test_dp2_rows_spawned_lands_within_the_band_of_world_one(
    solo_run: tuple[Path, Path], tmp_path: Path, never_load: None
) -> None:
    document, solo = solo_run
    out = tmp_path / "rows"
    assert main(_argv(document, out, "--parallel", "dp=2:rows")) == 0
    receipt = _receipt(out)
    assert receipt["execution"]["parallel"] == {
        "data": 2,
        "data_mode": "rows",
        "pipeline": 1,
        "context": 1,
        "tensor": 1,
        "expert": 1,
        "world": 2,
        "launcher": "spawned",
    }
    assert len(receipt["points"]) == 1
    _assert_within_band(solo, out)
    # the receipt is the world-1 receipt but for the geometry
    a, b = _receipt(solo), receipt
    assert a["execution"]["parallel"]["launcher"] == "solo"
    del a["execution"]["parallel"], b["execution"]["parallel"]
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


# --------------------------------------------------------------------------- #
# every replica computes the same outputs: the debug hook
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class _EveryReplica:
    """A publisher that lets every replica publish into its own directory:
    what replica 1 *would* write, made visible."""

    replica: int
    replicas: int = 2
    launcher: str = "joined"
    publish: bool = True

    def gather(self, payload: T) -> Sequence[T] | None:
        return (payload,)


def _rank_run(
    rank: int, collective: Collective, *, document: str, out: str
) -> dict[str, bytes]:
    from causalab.cli import register_model_key
    from causalab.neural.engines.pytorch_hooks import PytorchHooksEngine

    path = Path(document)
    # what `causalab run` does before compiling: the tiny fixture's entry
    register_model_key(json.loads(path.read_text()))
    env = ResolutionEnv(
        datasets=FileDatasets(root=DATA, fallback_roots=(TASKS_ROOT,)),
        artifacts=FileArtifacts(root=path.parent),
    )
    engine = PytorchHooksEngine(device="cpu", parallel=GEOMETRY, collective=collective)
    target = Path(out) / f"rank{rank}"
    run_protocol(path, env, engine, target, publisher=_EveryReplica(rank))
    # the receipt and the event stream are the joiner's alone (replica 0);
    # what is compared is what the engine published: the tables, the bundle
    return {
        p.name: p.read_bytes()
        for p in target.iterdir()
        if p.name not in (RECEIPT, EVENTS)
    }


def test_every_replica_holds_the_same_outputs_byte_for_byte(
    solo_run: tuple[Path, Path], tmp_path: Path
) -> None:
    document, solo = solo_run
    results = GlooWorld(GEOMETRY).run(
        functools.partial(_rank_run, document=str(document), out=str(tmp_path))
    )
    assert results[0] == results[1]
    assert TENSOR in results[0] and all(name in results[0] for name in TABLES)
    assert "train_eval.json" in results[0]  # the agreed eval scores too
    _assert_within_band(solo, tmp_path / "rank0")


# --------------------------------------------------------------------------- #
# refused by the children: no train to split
# --------------------------------------------------------------------------- #


def test_rows_on_a_document_without_train_is_refused_by_the_children(
    tmp_path: Path, never_load: None, capsys
) -> None:
    doc = json.loads(SCAN.read_text())
    doc["model"] = {"key": TINY, "revision": TINY_REVISION, "dtype": "fp32"}
    doc["method"]["sites"]["target"]["layers"] = {"sweep": [0, 1]}
    document = tmp_path / "scan.json"
    document.write_text(json.dumps(doc, indent=2))
    out = tmp_path / "rows"
    assert main(_argv(document, out, "--parallel", "dp=2:rows")) == 1
    assert "rank" in capsys.readouterr().err
    assert not (out / RECEIPT).exists()
