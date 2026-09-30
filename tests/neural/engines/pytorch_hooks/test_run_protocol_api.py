"""``run_protocol`` and ``causalab run`` are the same run.

Every other verb in the protocol layer has been a Python function all along —
load, validate, expand, canonicalize, plan, digest — so ``run`` was the one a
notebook, a script step or a test could reach only by shelling out and parsing
stdout. The fix is not "add a function that does something similar": it is that
the CLI *calls* the function, so there is exactly one run.

That is what this test pins, and it pins it the only way that means anything —
by running one document both ways and comparing everything that left the
process:

* the **run receipt**, which carries the campaign digest, the canonical
  document, and the per-point provenance digests. Canonical points and output
  identities are in there, so comparing it whole is stronger than picking
  fields out of it;
* every **file** either run wrote, byte for byte, and the fact that the two
  runs wrote the same set of them;
* the **manifest** ``RunResult`` reports, keyed by save-manifest path;
* the **terminal status** — the CLI's exit code, and the function returning
  rather than raising.

Byte equality is the right bar because the run receipt and the metric tables are
both deterministic functions of the document: a divergence would mean the two
paths disagree about what the run *was*, which is the failure mode a duplicated
implementation produces and a shared one cannot.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from causalab.cli import main
from causalab.neural.shared.engine_router import route
from causalab.io.events import EVENTS_FILE
from causalab.protocol import RUN_RECORD_NAME, run_protocol
from causalab.io.env import FileArtifacts, FileDatasets, ResolutionEnv

from tests.neural.engines.pytorch_hooks.conftest import TINY_LLAMA
from tests.protocol._env import CORPUS_DIR, FIXTURES

pytestmark = pytest.mark.smoke

#: A paired interchange with two metrics — so the comparison covers a save
#: manifest with more than one entry, and a document with a counterfactual
#: role rather than a bare harvest.
DOCUMENT = CORPUS_DIR / "02_interchange_im.json"

#: tiny-random is two layers deep, so the shipped L18 site is retargeted. The
#: same overrides go into both paths: the point is the run, not the document.
OVERRIDES = {"model.key": TINY_LLAMA, "sites.target.layers": 1}


@pytest.fixture(scope="module")
def artifacts_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("artifacts")
    shutil.copytree(FIXTURES / "artifacts", root, dirs_exist_ok=True)
    return root


@pytest.fixture(scope="module")
def both_runs(tmp_path_factory: pytest.TempPathFactory, artifacts_root: Path):
    """One document, run through the CLI and through ``run_protocol``."""
    base = tmp_path_factory.mktemp("parity")
    via_cli, via_api = base / "cli", base / "api"

    argv = [
        "run",
        "--engine",
        "auto",
        str(DOCUMENT),
        "--data-root",
        str(FIXTURES / "data"),
        "--artifacts-root",
        str(artifacts_root),
        "--out",
        str(via_cli),
        "--record",
    ]
    for path, value in OVERRIDES.items():
        argv += ["--set", f"{path}={value}"]
    status = main(argv)

    # the Python path, built the way any caller would: an environment, a
    # loaded document, and the engines it wants to offer
    from causalab.protocol.pipeline import compile_protocol

    env = ResolutionEnv(
        datasets=FileDatasets(root=FIXTURES / "data"),
        artifacts=FileArtifacts(root=artifacts_root),
    )
    loaded = compile_protocol(DOCUMENT, env=env, overrides=OVERRIDES)
    result = run_protocol(
        loaded, env, route("auto", device="cpu"), via_api, record=True
    )

    return status, via_cli, via_api, result


def _files(root: Path) -> dict[str, bytes]:
    # every file but the event stream: a timestamped sidecar (workflow spec
    # §4.3) that is an input to nothing, and the one file two runs of one
    # document are allowed to differ in
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != EVENTS_FILE
    }


def test_the_terminal_status_matches(both_runs) -> None:
    """The CLI succeeded, and the function returned rather than raising."""
    status, _, _, result = both_runs
    assert status == 0
    assert result.files, "run_protocol reported no saved files"


def test_the_run_record_is_identical(both_runs) -> None:
    """The canonical document, the campaign digest and every point digest.

    Comparing the record whole rather than field by field is deliberate: if
    the two paths ever disagree about *anything* they claim the run was, this
    fails, including fields added later.
    """
    _, via_cli, via_api, _ = both_runs
    assert (via_cli / RUN_RECORD_NAME).read_bytes() == (
        via_api / RUN_RECORD_NAME
    ).read_bytes()


def test_the_canonical_points_and_digests_match_explicitly(both_runs) -> None:
    """The criterion for one run, named rather than implied by the bytes above —
    so a future record-format change cannot quietly drop it."""
    _, via_cli, via_api, _ = both_runs
    cli = json.loads((via_cli / RUN_RECORD_NAME).read_text())
    api = json.loads((via_api / RUN_RECORD_NAME).read_text())

    assert cli["canonical"] == api["canonical"]
    assert cli["document_digest"] == api["document_digest"]
    assert [point["digest"] for point in cli["points"]] == [
        point["digest"] for point in api["points"]
    ]


def test_the_receipt_holds_the_table_to_nothing_beside_it(both_runs) -> None:
    """A run reads the table's bytes and nothing beside them (§2.2): the
    receipt carries no ``preparation`` block, because no sidecar is compared
    before a forward — the table's content digest in the canonical form is
    the whole of its identity to the run."""
    _, via_cli, _, _ = both_runs
    record = json.loads((via_cli / RUN_RECORD_NAME).read_text())
    assert "preparation" not in record


def test_the_same_files_were_written_with_the_same_bytes(both_runs) -> None:
    _, via_cli, via_api, _ = both_runs
    cli, api = _files(via_cli), _files(via_api)

    assert set(cli) == set(api), (
        f"only via the CLI: {sorted(set(cli) - set(api))}; "
        f"only via run_protocol: {sorted(set(api) - set(cli))}"
    )
    differing = sorted(name for name in cli if cli[name] != api[name])
    assert not differing, f"same document, different bytes: {differing}"


def test_the_reported_manifest_names_the_files_on_disk(both_runs) -> None:
    """``RunResult.files`` is what a Python caller uses instead of parsing the
    CLI's ``saved … -> …`` lines, so it has to name real files."""
    _, _, via_api, result = both_runs
    for manifest_path, disk_path in result.files.items():
        assert Path(disk_path).is_file(), (
            f"{manifest_path} -> {disk_path} is not a file"
        )
        assert via_api in Path(disk_path).parents


def test_a_bad_point_shard_is_refused_not_clamped(both_runs, artifacts_root) -> None:
    """``points`` is the sharding seam, and it moved with the run.

    A shard that silently became a different shard is worse than one that
    failed: its artifacts would still stamp as members of the whole campaign.

    Real engines are passed even though nothing executes: routing is decided
    before the shard is parsed — the order the CLI has always had — so an
    empty engine list would refuse on routing and prove nothing about
    ``points``.
    """
    from causalab.protocol.rules.errors import ProtocolError
    from causalab.protocol.pipeline import compile_protocol

    env = ResolutionEnv(
        datasets=FileDatasets(root=FIXTURES / "data"),
        artifacts=FileArtifacts(root=artifacts_root),
    )
    loaded = compile_protocol(DOCUMENT, env=env, overrides=OVERRIDES)
    with pytest.raises(ProtocolError, match="outside the campaign"):
        run_protocol(
            loaded,
            env,
            route("auto", device="cpu"),
            Path("never-created"),
            points="0:99",
            record=True,
        )
    assert not Path("never-created").exists(), (
        "the run receipt was written before the shard was validated"
    )
