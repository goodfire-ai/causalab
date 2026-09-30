"""``dry-run``'s memory estimate under ``--parallel`` (``docs/model_parallelism.md``
§2, §11; ``protocol/dry_run.py:memory_estimate``, the ``parallel`` fact's
two extra lines in ``protocol/cli.py``): per rank the resident weights and
the card footprint off the cached headers, the rule in words, and
*undecided* — a status, never a green — when the checkpoint is not cached
or the geometry is refused.

``unit`` throughout, torch-free: the API on a document naming the cached
tiny MoE fixture (registered by the session fixture), the real CLI in a
fresh offline interpreter on the corpus interchange document (its
Llama-3.1-8B behind an empty Hub cache: the undecided line, and torch never
imported), and the in-process CLI on the fixture for the estimate lines.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from causalab.cli import main
from causalab.protocol import reports as dry_run_module
from causalab.protocol.reports import MemoryEstimate, dry_run, memory_estimate
from causalab.protocol.parallel import ParallelGeometry
from causalab.protocol.registry import get_model_info

from tests.protocol._docs import base_doc
from tests.protocol.test_cli_parallel import INTERCHANGE
from tests.protocol.test_dry_run import _argv, _compile, _offline

pytestmark = pytest.mark.unit

#: A public tiny-random Qwen3.5-MoE checkpoint on the Hub
#: (https://huggingface.co/tiny-random/qwen3.5-moe).
TINY_MOE = "tiny-random/qwen3.5-moe"


def _moe_doc() -> dict[str, Any]:
    doc = base_doc()
    doc["model"] = {"key": TINY_MOE, "revision": "main", "dtype": "bf16"}
    return doc


@pytest.fixture
def moe_compiled(env):
    pytest.importorskip("transformers")
    try:
        get_model_info(TINY_MOE)
    except Exception:  # noqa: BLE001 — the fixture is registered by the session fixture when cached
        pytest.skip(f"{TINY_MOE} is not registered (not in the local Hub cache)")
    return _compile(_moe_doc(), env)


def test_the_estimate_off_the_cached_headers_is_per_rank_and_names_the_rule(
    moe_compiled, env
) -> None:
    report = dry_run(moe_compiled, env, parallel=ParallelGeometry(expert=2))
    assert report.parallel is not None and report.parallel.refusals == ()
    estimate = report.parallel.memory
    assert isinstance(estimate, MemoryEstimate)
    assert report.parallel.memory_undecided is None
    assert estimate.dtype == "bf16" and len(estimate.resident) == 2
    assert estimate.resident[0] == estimate.resident[1] < estimate.whole
    assert all(f > r for f, r in zip(estimate.footprint, estimate.resident))
    assert "15%" in estimate.rule and "1.8x" not in estimate.rule  # one copy now
    assert estimate.source.startswith("the headers of 1 cached safetensors file")
    # world 1 in bf16 holds the whole model, twice the bytes of nothing shard
    whole = dry_run(moe_compiled, env, parallel=ParallelGeometry(data=2)).parallel
    assert whole is not None and whole.memory is not None
    assert whole.memory.resident == (estimate.whole, estimate.whole)


def test_a_refused_geometry_carries_no_estimate(moe_compiled, env) -> None:
    report = dry_run(moe_compiled, env, parallel=ParallelGeometry(tensor=3))
    assert report.parallel is not None and report.parallel.refusals
    assert report.parallel.memory is None
    assert report.parallel.memory_undecided is None


def test_an_uncached_checkpoint_is_undecided_naming_the_cache(
    moe_compiled, env, monkeypatch
) -> None:
    monkeypatch.setattr(
        dry_run_module,
        "memory_estimate",
        lambda *a, **k: "the checkpoint x is not cached",
    )
    report = dry_run(moe_compiled, env, parallel=ParallelGeometry(expert=2))
    assert report.parallel is not None
    assert report.parallel.memory is None
    assert report.parallel.memory_undecided == "the checkpoint x is not cached"
    # the reason is a fact of the report's undecided vocabulary, so a reader
    # of ``undecided_topics`` never takes an uncomputed estimate for a green
    assert "memory" in report.undecided_topics
    (memory,) = [u for u in report.undecided if u.topic == "memory"]
    assert memory.detail == "the checkpoint x is not cached"


def test_a_computed_estimate_leaves_memory_decided(moe_compiled, env) -> None:
    report = dry_run(moe_compiled, env, parallel=ParallelGeometry(expert=2))
    assert report.parallel is not None and report.parallel.memory is not None
    assert "memory" not in report.undecided_topics


def test_memory_estimate_says_why_when_it_cannot(monkeypatch) -> None:
    info = get_model_info("meta-llama/Llama-3.1-8B")
    monkeypatch.setattr(
        "causalab.protocol.checkpoint_census.cached_checkpoint_files",
        lambda key, revision: None,
    )
    reason = memory_estimate(info, "main", "bf16", ParallelGeometry(tensor=2))
    assert isinstance(reason, str) and "not in the local Hub cache" in reason
    monkeypatch.setattr(
        "causalab.protocol.checkpoint_census.cached_checkpoint_files",
        lambda key, revision: (Path("/nonexistent"),),
    )
    reason = memory_estimate(info, "main", "int8", ParallelGeometry(tensor=2))
    assert isinstance(reason, str) and "int8" in reason


def _data_root(artifacts_root: Path) -> str:
    """The corpus data root, read off the shared argv (``--data-root <dir>``)
    rather than by position, which ``--engine`` shifted."""
    argv = _argv("x", artifacts_root)
    return argv[argv.index("--data-root") + 1]


def test_the_cli_prints_the_estimate_lines_for_the_cached_fixture(
    moe_compiled, artifacts_root: Path, tmp_path: Path, capsys
) -> None:
    document = tmp_path / "moe.json"
    document.write_text(json.dumps(_moe_doc()))
    argv = [
        "dry-run",
        "--engine",
        "auto",
        str(document),
        "--data-root",
        _data_root(artifacts_root),
        "--artifacts-root",
        str(artifacts_root),
        "--parallel",
        "ep=2",
    ]
    assert main(argv) == 0
    out = capsys.readouterr().out
    assert "parallel  dp=1,pp=1,cp=1,tp=1,ep=2 (world 2): accepted" in out
    assert "  resident weights (bf16; the model is " in out
    assert "  estimated card footprint: rank0 " in out and "rank1 " in out
    assert "  headroom rule: 15% of the model's bytes" in out


def test_the_offline_cli_prints_undecided_for_an_uncached_checkpoint(
    artifacts_root: Path, tmp_path: Path
) -> None:
    """The corpus interchange document names Qwen/Qwen3-8B; with the Hub
    cache pointed at an empty directory (a machine with the model cached
    reports it otherwise) the
    ``parallel`` fact prints its accepted line, then the memory status, and
    torch is never imported."""
    empty_cache = tmp_path / "hub"
    empty_cache.mkdir()
    result = _offline(
        _argv(INTERCHANGE, artifacts_root, "--parallel", "tp=2"), hub_cache=empty_cache
    )
    assert result["code"] == 0, result["err"]
    assert "parallel  dp=1,pp=1,cp=1,tp=2,ep=1 (world 2): accepted" in result["out"]
    assert (
        "  memory: undecided — the checkpoint Qwen/Qwen3-8B@main is not in the local Hub cache"
        in result["out"]
    )
    assert not result["torch"]
