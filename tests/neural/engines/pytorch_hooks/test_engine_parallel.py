"""``PytorchHooksEngine(parallel=…, collective=…)`` (``docs/model_parallelism.md`` §2, §3, §8).

The engine stores the geometry it was built with and the receipt reads it
off the engine. At ``world > 1`` ``execute`` decides everything it can
**before** ``load_model`` is entered: an axis no branch serves yet is refused
by name, a world with no ``torch.distributed`` group and no collective handed
in is refused by name, a collective whose group sizes disagree with the
geometry is refused by name, a collective with no mesh and no ``Sharding``
beside it is refused by name (the loader cannot shard without the meshes),
a caller-owned bundle loaded under another geometry is refused by name — and
a collective that fits, with its sharding, reaches the loader, so a geometry
never runs on one device as if it were sharded. At world 1 nothing changes:
the loader is the first thing the executor reaches, exactly as today.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from causalab.neural.engines.pytorch_hooks import engine as hooks_engine
from causalab.neural.engines.pytorch_hooks.engine import PytorchHooksEngine
from causalab.neural.engines.pytorch_hooks.loading import load_model
from causalab.neural.engines.pytorch_hooks.sharding import Sharding
from causalab.protocol.rules.errors import ProtocolError
from causalab.protocol.parallel import ONE, ParallelGeometry
from causalab.protocol.receipt import execution_record
from causalab.protocol.pipeline import run_protocol

from tests.neural.engines.pytorch_hooks.conftest import TINY_LLAMA
from tests._helpers.device_mesh import FakeDeviceMesh

from tests.protocol._env import (
    CORPUS_DIR,
    FIXTURES,
    build_env,
    write_pca_fixture,
    write_rot_fixture,
)

pytestmark = pytest.mark.unit

INTERCHANGE = CORPUS_DIR / "02_interchange_im.json"


class _Never(Exception):
    """The loader was entered."""


@pytest.fixture(scope="module")
def env(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("engine-parallel-artifacts")
    shutil.copytree(FIXTURES / "artifacts", root, dirs_exist_ok=True)
    write_rot_fixture(root)
    write_pca_fixture(root)
    return build_env(root)


def test_the_engine_stores_its_geometry_and_defaults_to_world_one() -> None:
    assert PytorchHooksEngine().parallel == ONE
    geometry = ParallelGeometry(tensor=2, expert=4)
    assert PytorchHooksEngine(parallel=geometry).parallel is geometry
    assert execution_record(PytorchHooksEngine(parallel=geometry))["parallel"] == {
        "data": 1,
        "data_mode": "points",
        "pipeline": 1,
        "context": 1,
        "tensor": 2,
        "expert": 4,
        "world": 4,
        "launcher": "solo",
    }


def test_a_world_above_one_is_refused_before_the_loader_is_entered(
    env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``P4`` naming ``--parallel`` and the geometry: no ``torch.distributed``
    group is initialised in the test process and no collective was handed
    in — and the monkeypatched ``load_model`` is never called, so the
    refusal is shown to come before any weights."""

    def never(*args, **kwargs):
        raise _Never("load_model was entered")

    monkeypatch.setattr(hooks_engine, "load_model", never)
    engine = PytorchHooksEngine(parallel=ParallelGeometry(tensor=2))
    with pytest.raises(ProtocolError) as err:
        run_protocol(INTERCHANGE, env, engine, tmp_path)
    assert err.value.code == "P4"
    message = str(err.value)
    assert "--parallel" in message and "tp=2" in message
    assert "no torch.distributed process group" in message


class _Sized:
    """A collective with the given group sizes and nothing else: what the
    engine inspects before the loader."""

    def __init__(self, **sizes: int) -> None:
        self._sizes = sizes

    def rank(self, axis: str) -> int:
        return 0

    def size(self, axis: str) -> int:
        return self._sizes.get(axis, 1)


def test_a_decoding_document_is_refused_by_name_under_context_before_the_loader(
    env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Context parallelism's one document rule (§8.4): a read at a
    ``generated`` position under ``cp > 1`` is refused naming
    ``--parallel.context`` before any weights — identically on every rank,
    since it is a fact of the document and the geometry alone."""
    import json

    def never(*args, **kwargs):
        raise _Never("load_model was entered")

    monkeypatch.setattr(hooks_engine, "load_model", never)
    doc = json.loads(INTERCHANGE.read_text())
    doc["method"]["reads"]["logits"]["pos"] = {
        "generated": {"max_new_tokens": 2},
        "index": -1,
    }
    document = tmp_path / "decode.json"
    document.write_text(json.dumps(doc))
    geometry = ParallelGeometry(context=2)
    engine = PytorchHooksEngine(
        parallel=geometry,
        collective=_Sized(context=2),
        sharding=Sharding(geometry=geometry, rank=0, meshes={}),
    )
    with pytest.raises(ProtocolError) as err:
        run_protocol(document, env, engine, tmp_path / "out")
    assert err.value.code == "P4"
    assert "--parallel.context" in str(err.value) and "cp=2" in str(err.value)
    # the same document without the decode passes the check and reaches the loader
    with pytest.raises(_Never):
        run_protocol(INTERCHANGE, env, engine, tmp_path / "plain")


def test_a_collective_that_disagrees_with_the_geometry_is_refused(
    env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def never(*args, **kwargs):
        raise _Never("load_model was entered")

    monkeypatch.setattr(hooks_engine, "load_model", never)
    engine = PytorchHooksEngine(
        parallel=ParallelGeometry(tensor=2), collective=_Sized(tensor=4, model=4)
    )
    with pytest.raises(ProtocolError) as err:
        run_protocol(INTERCHANGE, env, engine, tmp_path)
    assert err.value.code == "P4"
    assert "--parallel.tensor" in str(err.value) or "--parallel" in str(err.value)
    assert "4 ranks" in str(err.value)


def test_a_fitting_collective_with_its_sharding_reaches_the_loader(
    env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With a collective whose sizes are the geometry's and this rank's
    ``Sharding`` beside it, ``execute`` goes on to the loader — the sentinel
    surfaces, not a geometry refusal. (Before tensor parallelism was served
    the collective alone reached the loader; now the loader needs the meshes
    the sharding carries, so a collective without one is the refusal below.)"""

    def never(*args, **kwargs):
        raise _Never("load_model was entered")

    monkeypatch.setattr(hooks_engine, "load_model", never)
    geometry = ParallelGeometry(tensor=2)
    collective = _Sized(tensor=2, model=2)
    engine = PytorchHooksEngine(
        parallel=geometry,
        collective=collective,
        sharding=Sharding(
            geometry,
            0,
            meshes={"tensor": FakeDeviceMesh((0, 1))},  # type: ignore[dict-item]
            collective=collective,
        ),
    )
    with pytest.raises(_Never):
        run_protocol(INTERCHANGE, env, engine, tmp_path)


def test_a_collective_without_a_mesh_and_no_sharding_is_refused_before_the_loader(
    env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A handed-in collective that carries no ``Mesh`` gives the loader
    nothing to apply the plan over: refused by name, never an unsharded
    load behind a geometry that says ``tp=2``."""

    def never(*args, **kwargs):
        raise _Never("load_model was entered")

    monkeypatch.setattr(hooks_engine, "load_model", never)
    engine = PytorchHooksEngine(
        parallel=ParallelGeometry(tensor=2), collective=_Sized(tensor=2, model=2)
    )
    with pytest.raises(ProtocolError) as err:
        run_protocol(INTERCHANGE, env, engine, tmp_path)
    assert err.value.code == "P4"
    message = str(err.value)
    assert "--parallel" in message and "Sharding" in message


def test_a_caller_bundle_loaded_at_world_one_is_refused_above_world_one(
    env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``bundle=`` at ``world > 1``: the bundle's recorded geometry must be
    the engine's, or the receipt would say ``tp=2`` of a model that was
    loaded whole (``check_caller_bundle``)."""

    def never(*args, **kwargs):
        raise _Never("load_model was entered")

    bundle = load_model(TINY_LLAMA)
    monkeypatch.setattr(hooks_engine, "load_model", never)
    engine = PytorchHooksEngine(
        parallel=ParallelGeometry(tensor=2),
        collective=_Sized(tensor=2, model=2),
        bundle=bundle,
    )
    with pytest.raises(ProtocolError) as err:
        run_protocol(INTERCHANGE, env, engine, tmp_path)
    assert err.value.code == "P4"
    message = str(err.value)
    assert "tp=2" in message and "tp=1" in message and "geometry" in message


def test_at_world_one_the_loader_is_the_first_thing_reached(
    env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The twin: with every axis one the engine takes today's path, and the
    first call the executor makes is the loader — the sentinel is what
    surfaces, not a geometry refusal."""

    def never(*args, **kwargs):
        raise _Never("load_model was entered")

    monkeypatch.setattr(hooks_engine, "load_model", never)
    with pytest.raises(_Never):
        run_protocol(INTERCHANGE, env, PytorchHooksEngine(parallel=ONE), tmp_path)
