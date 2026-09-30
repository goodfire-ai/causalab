"""The nnsight engine's ``bundle=`` entry (spec §9, the ownership contract).

The reference engine's constructor is the one with ``ModelBundle.from_model``;
this engine takes an [`NnsightBundle`][causalab.neural.engines.nnsight_tracing.loading.NnsightBundle] the same way and holds it to the
same realization check before any trace — the document's ``key`` /
``revision`` / ``dtype`` and the engine's ``device`` against the bundle's —
and the run receipt says ``model_source: caller``.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from causalab.neural.engines.nnsight_tracing.engine import NnsightEngine
from causalab.protocol import RUN_RECORD_NAME, run_protocol
from causalab.protocol.rules.errors import ProtocolError
from causalab.protocol.pipeline import compile_protocol
from causalab.io.env import FileArtifacts, FileDatasets, ResolutionEnv

from tests.neural.engines.nnsight_tracing.conftest import TINY_LLAMA
from tests.protocol._env import CORPUS_DIR, FIXTURES

pytestmark = pytest.mark.smoke

DOCUMENT = CORPUS_DIR / "02_interchange_im.json"
OVERRIDES = {"model.key": TINY_LLAMA, "sites.target.layers": 1}


@pytest.fixture(scope="module")
def env(tmp_path_factory: pytest.TempPathFactory) -> ResolutionEnv:
    artifacts = tmp_path_factory.mktemp("artifacts")
    shutil.copytree(FIXTURES / "artifacts", artifacts, dirs_exist_ok=True)
    return ResolutionEnv(
        datasets=FileDatasets(root=FIXTURES / "data"),
        artifacts=FileArtifacts(root=artifacts),
    )


def test_a_dtype_disagreement_is_refused_before_any_trace(
    trace_llama, env, tmp_path: Path
) -> None:
    loaded = compile_protocol(
        DOCUMENT, env=env, overrides={**OVERRIDES, "model.dtype": "bf16"}
    )
    with pytest.raises(ProtocolError, match="model.dtype") as err:
        run_protocol(loaded, env, NnsightEngine(bundle=trace_llama), tmp_path)
    assert "'bf16'" in str(err.value) and "'fp32'" in str(err.value)
    assert not (tmp_path / "iia.json").exists()


def test_a_device_disagreement_is_refused(trace_llama, env, tmp_path: Path) -> None:
    loaded = compile_protocol(DOCUMENT, env=env, overrides=OVERRIDES)
    with pytest.raises(ProtocolError, match="device"):
        run_protocol(
            loaded, env, NnsightEngine(device="cuda:1", bundle=trace_llama), tmp_path
        )


def test_a_matching_bundle_runs_and_the_record_says_caller(
    trace_llama, env, tmp_path: Path
) -> None:
    """The valid-work twin: the document's realization and the bundle agree,
    the trace runs, and the receipt is the loaded run's save for the source."""
    loaded = compile_protocol(DOCUMENT, env=env, overrides=OVERRIDES)
    via_loader, via_caller = tmp_path / "loaded", tmp_path / "caller"
    run_protocol(loaded, env, NnsightEngine(), via_loader, record=True)
    result = run_protocol(
        loaded, env, NnsightEngine(bundle=trace_llama), via_caller, record=True
    )
    assert result.files
    loaded_record = json.loads((via_loader / RUN_RECORD_NAME).read_text())
    caller_record = json.loads((via_caller / RUN_RECORD_NAME).read_text())
    assert loaded_record["execution"] == {
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
    assert caller_record["execution"] == {
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
    loaded_record["execution"].pop("model_source")
    caller_record["execution"].pop("model_source")
    assert loaded_record == caller_record
    assert sorted(p.name for p in via_loader.iterdir()) == sorted(
        p.name for p in via_caller.iterdir()
    )
