"""Checks over the method library, ``demos/methods/``.

A method document is not a seven-section demo (``docs/demos.md``): it is one
intervention specification per method, indexed by one README table, with the
run each row reports reduced to a small results file. So ``test_demos.py``
leaves the directory alone (``OTHER_GENRES``) and this module checks the
library's own contract:

* every document under ``protocols/`` and ``workflows/`` validates offline
  against the default data root — the shipped task tables, which every data
  root falls back to, so the library ships no ``data/``. The apply half of a
  fit-and-apply pair names a run-tree bundle and is refused standalone under
  rule 15; ``test_protocol_presets.RUN_TREE_ONLY`` is the one list of those,
  and each is validated through the workflow that supplies its artifact when
  one exists;
* the README table has exactly one row per document, and every row's
  ``result`` link is a results file that parses;
* each results file quotes the digest of the document it reports on — a
  intervention specification's from ``tests/protocol/shipped_digests.json`` (which
  ``test_shipped_digests.py`` holds to the documents), a workflow's computed
  the way ``causalab validate`` reports it — so a stale results file is a
  test failure rather than a quiet lie. The results files are the future
  golden pins; there is no golden test over them yet.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import pytest

from causalab.io.env import FileArtifacts, FileDatasets, ResolutionEnv
from causalab.protocol.pipeline import compile_protocol
from causalab.protocol.rules.data import check_data_columns
from causalab.protocol.rules.errors import ValidationError
from causalab.tasks import TASKS_ROOT
from causalab.workflow.document import load_workflow

from tests._helpers.paths import METHODS_DIR, PROTOCOLS_DIR, WORKFLOWS_DIR
from tests.protocol._env import FIXTURES, write_pca_fixture, write_rot_fixture
from tests.protocol.test_protocol_presets import RUN_TREE_ONLY

pytestmark = pytest.mark.unit

README = METHODS_DIR / "README.md"
RESULTS = METHODS_DIR / "results"
REPO = METHODS_DIR.parents[1]

#: A README table row: ``| [`name.json`](protocols/name.json) | ... | [x](results/...) |``.
ROW = re.compile(
    r"^\|\s*\[`([a-z0-9_]+\.json)`\]\(((?:protocols|workflows)/[a-z0-9_]+\.json)\)"
)
RESULT_LINK = re.compile(r"\]\((results/(?:protocols|workflows)/[a-z0-9_]+\.json)\)")
#: The one document with no static registry row: its model registers only
#: when an engine loads the HF config (``test_shipped_digests.EXCLUDED``), and
#: its table is a 4-row fixture the shipped tasks do not carry.
SMOKE_ONLY = frozenset({"minimal_cpu.json"})


def _documents() -> list[Path]:
    return sorted(PROTOCOLS_DIR.glob("*.json")) + sorted(WORKFLOWS_DIR.glob("*.json"))


def _ids(paths: list[Path]) -> list[str]:
    return [p.relative_to(METHODS_DIR).as_posix() for p in paths]


@pytest.fixture(scope="module")
def env(tmp_path_factory: pytest.TempPathFactory) -> ResolutionEnv:
    """The CLI's default environment with no ``--data-root`` — the shipped task
    tables — over the protocol tests' artifact fixtures: the committed locate
    artifact plus the two generated bundles (``weekdays_das_apply.json``'s
    rotation, ``das_pca_init.json``'s basis), stamped for the shipped model
    the way ``tests/protocol/update_shipped_digests.py`` makes them."""
    root = tmp_path_factory.mktemp("artifacts")
    shutil.copytree(FIXTURES / "artifacts", root, dirs_exist_ok=True)
    write_rot_fixture(root)
    write_pca_fixture(root)
    return ResolutionEnv(
        datasets=FileDatasets(root=TASKS_ROOT, fallback_roots=(TASKS_ROOT,)),
        artifacts=FileArtifacts(root=root),
    )


def _workflows_naming(document: Path) -> list[Path]:
    found = []
    for workflow in sorted(WORKFLOWS_DIR.glob("*.json")):
        steps = json.loads(workflow.read_text()).get("steps", {})
        if any(
            str(s.get("document", "")).endswith(document.name) for s in steps.values()
        ):
            found.append(workflow)
    return found


def _rows() -> dict[str, tuple[str, str | None]]:
    """document relpath -> (line, results link) for every README table row."""
    rows: dict[str, tuple[str, str | None]] = {}
    for line in README.read_text().splitlines():
        match = ROW.match(line)
        if match is None:
            continue
        link = RESULT_LINK.search(line)
        assert match.group(2) not in rows, f"{match.group(2)} has two rows"
        rows[match.group(2)] = (line, link.group(1) if link else None)
    return rows


@pytest.mark.parametrize("document", _documents(), ids=_ids(_documents()))
def test_validates_with_the_default_data_root(
    document: Path, env: ResolutionEnv
) -> None:
    if document.name in SMOKE_ONLY:
        pytest.skip(
            "registered from the HF config at run time; the CI smoke job runs it"
        )

    raw = json.loads(document.read_text())
    if "steps" in raw:
        loaded = load_workflow(document, env)
        for name in loaded.order:
            inner = loaded.inner.get(name)
            if inner is not None:
                check_data_columns(inner.compiled, env)
        return
    try:
        check_data_columns(compile_protocol(document, env=env), env)
    except ValidationError as error:
        if error.rule != 15 or document.name not in RUN_TREE_ONLY:
            raise
        # an apply half: validated through its workflow when one names it,
        # else through the fit it replays (its bundle exists only in a run tree)
        assert _workflows_naming(document) or document.name in RUN_TREE_ONLY


def test_every_document_has_exactly_one_readme_row() -> None:
    rows = _rows()
    expected = {p.relative_to(METHODS_DIR).as_posix() for p in _documents()}
    assert set(rows) == expected, (
        f"missing rows: {sorted(expected - set(rows))}; "
        f"rows without a document: {sorted(set(rows) - expected)}"
    )


def test_every_row_links_a_results_file_that_parses() -> None:
    for rel, (line, link) in _rows().items():
        assert link, f"{rel}: its README row has no results link"
        target = METHODS_DIR / link
        assert target.is_file(), f"{rel}: {link} does not exist"
        payload = json.loads(target.read_text())
        assert payload["document"] == rel, (
            f"{link} reports on {payload['document']}, not {rel}"
        )
        for key in (
            "document_digest",
            "model",
            "engine",
            "device",
            "dtype",
            "date",
            "elapsed_s",
            "metrics",
        ):
            assert key in payload, f"{link} lacks {key}"
        assert target.stat().st_size < 8000, (
            f"{link} is not small: {target.stat().st_size} bytes"
        )


PINS = json.loads((REPO / "tests/protocol/shipped_digests.json").read_text())


@pytest.mark.parametrize("document", _documents(), ids=_ids(_documents()))
def test_the_results_file_pins_the_current_digest(document: Path) -> None:
    rel = document.relative_to(METHODS_DIR).as_posix()
    results = RESULTS / rel
    assert results.is_file(), f"no results file {results.relative_to(METHODS_DIR)}"
    pinned = json.loads(results.read_text())["document_digest"]
    if "steps" in json.loads(document.read_text()):
        env = ResolutionEnv(
            datasets=FileDatasets(root=TASKS_ROOT, fallback_roots=(TASKS_ROOT,)),
            artifacts=FileArtifacts(root=REPO),
        )
        current = load_workflow(document, env).digest
    elif document.name in SMOKE_ONLY:
        assert pinned.startswith("unpinned:"), pinned  # no canonical form offline
        return
    else:
        current = PINS[document.name]["document"]
    assert pinned == current, (
        f"{rel}: results file quotes {pinned[:16]}… but the document digests to "
        f"{current[:16]}… — re-run it (demos/methods/scripts/run_all.py) and "
        "demos/methods/scripts/summarize.py"
    )
