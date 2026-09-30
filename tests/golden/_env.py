"""Resolution environment for the paper-golden tier.

Mirrors tests/protocol/_env.py but roots at this directory's own fixtures,
so the paper-golden documents and their digest pins stay decoupled from the
corpus pins in tests/protocol/corpus_digests.json.

The fixture datasets are committed JSON tables produced by the seeded
generators in tests/golden/fixtures/generators/ — dataset content digests
are part of each document's canonical form, so fixtures and pins move
together (regenerate both, review both diffs).
"""

from __future__ import annotations

from pathlib import Path

from causalab.io.env import FileArtifacts, FileDatasets, ResolutionEnv

FIXTURES = Path(__file__).parent / "fixtures"
GOLDEN_PROTOCOLS = Path(__file__).parent / "protocols"
GOLDENS_FILE = Path(__file__).parent / "paper_goldens.json"


def build_env(artifacts_root: Path) -> ResolutionEnv:
    return ResolutionEnv(
        datasets=FileDatasets(root=FIXTURES / "data"),
        artifacts=FileArtifacts(root=artifacts_root),
    )
