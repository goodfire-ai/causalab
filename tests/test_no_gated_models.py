"""The CPU tier names no gated Hugging Face repo.

GitHub-hosted CI runs ``uv run pytest -m "not golden"`` with no ``HF_TOKEN``,
and the run door
([`causalab.protocol.pipeline.resolve_positions`][], from ``handoff``) loads
the document's tokenizer **before** any engine runs — so a CPU test whose
document names a gated repo 401s there even under a stub engine that loads no
weights (18 CLI tests once did). A warm local cache hides the reliance
completely: a machine with the Llama tokenizer cached passes every such test,
which is how the tier came to be built around a gated model without anyone
noticing. This census is the guard that does not depend on
what is cached: it walks ``tests/`` and refuses any file whose text spells a
gated key outside `ALLOWED`, each entry of which says why the mention
loads nothing.

The fixture corpus's model is ``Qwen/Qwen3-8B`` (``tests/protocol/_env.py::
CORPUS_MODEL``). The paper goldens (``tests/golden/``) are the documented
exception — they pin published values on Llama and Gemma and run offline from
a Hub cache that already holds those weights — and are outside the walk.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from tests._helpers.paths import PROTOCOLS_DIR

pytestmark = pytest.mark.unit

TESTS = Path(__file__).resolve().parent
REPO = TESTS.parent
PRESETS_DIR = PROTOCOLS_DIR

#: Key prefixes of the gated organisations the repo's registry knows.
GATED = ("meta-llama/", "google/gemma")

#: Files that may spell a gated key, and why none of them loads one. Paths are
#: relative to ``tests/``. Keep it minimal: an entry whose file no longer
#: names a gated key is refused below, so the list cannot rot.
ALLOWED: dict[str, str] = {
    "protocol/test_site_layers.py": (
        "the model the shipped documents named at the base revision the "
        "layer-rename round trip replays from: a literal compared against git "
        "history, nothing loaded"
    ),
    "protocol/test_capability_registry.py": (
        "the registry's Llama rows by name — static metadata, nothing loaded"
    ),
    "protocol/test_registry_shapes.py": (
        "the registry's gemma-2-2b MIB row by name — static metadata, nothing loaded"
    ),
    "test_no_gated_models.py": "this census",
    # the parallelism tests (docs/model_parallelism.md): registry rows and
    # committed header censuses of the gated checkpoints by name — static
    # metadata and JSON headers, never a weight, a config fetch or a tokenizer
    "_helpers/header_census.py": (
        "the committed safetensors header censuses of Llama-3.1-70B and "
        "gemma-2-9b, read as JSON into a fake Hub cache; no repo is touched"
    ),
    "neural/engines/pytorch_hooks/test_sharded_load_cast.py": (
        "names gemma-2-9b's capture shape in its docstring; the converting "
        "load it exercises runs on a tiny-random fixture"
    ),
    "protocol/test_dry_run_memory.py": (
        "the registry's Llama-3.1-8B row by name, and the dry run's own "
        "'not in the local Hub cache' text — nothing loaded"
    ),
    "protocol/test_header_census.py": (
        "the committed header censuses by key — the census reader is held to "
        "the JSON, never to the Hub"
    ),
    "protocol/test_kv_replication.py": (
        "the registry's Llama and gemma rows by name — head counts, nothing loaded"
    ),
    "protocol/test_parallel.py": (
        "the registry's Llama-3.1-8B row by name — the geometry checks read heads "
        "and layers, nothing loaded"
    ),
    "protocol/test_parallel_context.py": (
        "the registry's Llama-3.1-8B row by name — the hybrid-family switch "
        "reads layer types, nothing loaded"
    ),
    "protocol/test_parallel_memory.py": (
        "the registry's Llama-3.1-8B row by name — placement arithmetic over "
        "the entry, nothing loaded"
    ),
    "protocol/test_parallel_memory_conversion.py": (
        "gemma-2-9b's committed header census and registry row — the dtype "
        "conversion estimate off headers, nothing loaded"
    ),
    "protocol/test_parallel_memory_large.py": (
        "Llama-3.1-70B's committed header census and registry row — the "
        "memory estimate off headers, nothing loaded"
    ),
    "protocol/test_parallel_plan.py": (
        "the registry's Llama and gemma rows by name, and AutoConfig.for_model "
        "on the model TYPE (a config class, no repo) — nothing loaded"
    ),
}


def _is_shipped_canonical_pin(rel: Path) -> bool:
    """A byte pin of a *shipped* preset's canonical form (``test_shipped_digests
    .py``): the preset may name a gated model — the tier loads it offline
    against the registry and never runs it — and the pin is its bytes."""
    return (
        rel.parent == Path("protocol/canonical") and (PRESETS_DIR / rel.name).is_file()
    )


def _mentions_gated(path: Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return False  # a binary fixture, or a socket/pyc the walk met
    return any(key in text for key in GATED)


def _walk() -> list[Path]:
    """The committed files under ``tests/`` outside ``tests/golden/`` — what
    git tracks, so a neighbouring xdist worker's transient scratch file is not
    censused (a directory walk raced one once); ``rglob`` where the tree is
    not a checkout (an export)."""
    try:
        listed = subprocess.run(
            ["git", "-C", str(REPO), "ls-files", "-z", "--", "tests"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        files = [REPO / name for name in listed.split("\0") if name]
    except (subprocess.CalledProcessError, OSError):
        files = [p for p in TESTS.rglob("*") if "__pycache__" not in p.parts]
    return sorted(
        p.relative_to(TESTS)
        for p in files
        if p.is_file() and p.relative_to(TESTS).parts[0] != "golden"
    )


def test_no_cpu_tier_file_names_a_gated_model() -> None:
    offenders = [
        str(rel)
        for rel in _walk()
        if str(rel) not in ALLOWED
        and not _is_shipped_canonical_pin(rel)
        and _mentions_gated(TESTS / rel)
    ]
    assert not offenders, (
        "these files under tests/ (outside tests/golden/) name a gated Hugging "
        "Face repo, which CI cannot load without a token — retarget the fixture "
        f"at the corpus model (tests/protocol/_env.py::CORPUS_MODEL): {offenders}"
    )


def test_every_allowlist_entry_is_still_needed() -> None:
    stale = [rel for rel in ALLOWED if not _mentions_gated(TESTS / rel)]
    assert not stale, f"ALLOWED entries whose file names no gated key any more: {stale}"


def test_the_shipped_pins_the_walk_skips_are_pins_of_shipped_presets() -> None:
    """The one pattern rule covers exactly the presets' pins: every skipped
    canonical file has a preset of the same name, and no corpus pin
    (``NN_*_im.json``) is skipped."""
    skipped = [rel for rel in _walk() if _is_shipped_canonical_pin(rel)]
    assert skipped, (
        "no shipped canonical pin found — did tests/protocol/canonical move?"
    )
    assert not [rel for rel in skipped if rel.name.endswith("_im.json")]
