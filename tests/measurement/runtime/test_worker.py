"""Checkpoint attestation is tested as file I/O, without constructing a model."""

import json
import os

import pytest

from causalab.measurement.collection import file_hash
from causalab.measurement.runtime.worker import model_files

pytestmark = pytest.mark.unit


def test_same_study_hash_reuse_checks_ctime_even_when_mtime_is_restored(
    tmp_path, monkeypatch
):
    model = tmp_path / "model"
    model.mkdir()
    weights = model / "model.safetensors"
    weights.write_bytes(b"opaque checkpoint bytes 1")
    cache = tmp_path / "verified.json"
    first = model_files(str(model), "local", cache=cache)
    with monkeypatch.context() as patch:

        def no_read(path):
            raise AssertionError(
                "unchanged checkpoint should not be read again within this study"
            )

        patch.setattr("causalab.measurement.runtime.worker.file_hash", no_read)
        assert model_files(str(model), "local", cache=cache, rehash=False) == first
    stat = weights.stat()
    weights.write_bytes(b"opaque checkpoint bytes 2")
    os.utime(weights, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    changed = model_files(str(model), "local", cache=cache, rehash=False)
    assert changed != first
    assert changed[weights.name] == file_hash(weights)


def test_startup_and_resume_rehash_instead_of_trusting_cached_values(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    weights = model / "model.safetensors"
    weights.write_bytes(b"opaque checkpoint")
    cache = tmp_path / "verified.json"
    expected = model_files(str(model), "local", cache=cache)
    stored = json.loads(cache.read_text())
    stored["files"][weights.name] = "0" * 64
    cache.write_text(json.dumps(stored))
    assert model_files(str(model), "local", cache=cache, rehash=True) == expected


def test_changing_a_checkpoint_during_hashing_is_refused(tmp_path, monkeypatch):
    model = tmp_path / "model"
    model.mkdir()
    weights = model / "model.safetensors"
    weights.write_bytes(b"before")

    def changed_during_read(path):
        value = file_hash(path)
        path.write_bytes(b"after, longer")
        return value

    monkeypatch.setattr(
        "causalab.measurement.runtime.worker.file_hash", changed_during_read
    )
    with pytest.raises(ValueError, match="changed during"):
        model_files(str(model), "local", cache=tmp_path / "verified.json")
    assert not (tmp_path / "verified.json").exists()
