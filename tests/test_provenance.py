"""The runtime can say what it is, and never says "unknown".

`code_commit()` — the stub this replaces — was a `git rev-parse --short HEAD`
that returned the string `"unknown"` on failure and was stamped into every
output identity. So a run whose provenance could not be resolved recorded a
value that reads like a value, and a run could produce numerically excellent
results on a package that was not the requested branch with nothing to notice
it.

Two tiers here, and the split is deliberate.

The **unit** tier covers the mapping and the hashing as pure functions: PEP 610
`direct_url.json` → source kind, and files → tree digest. Those are where the
logic is, and they are testable from synthetic metadata with no install.

The **smoke** tier does the thing that cannot be faked: it *actually installs*
causalab three ways — editable (the live environment), a wheel, and a git URL —
and asks each install to describe itself. That is the acceptance test, and it is
cheap only because `causalab.provenance` is torch-free: each install is
`--no-deps` into a bare venv, so no wheel over 200 kB is downloaded.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from causalab.provenance import (
    SOURCE_KINDS,
    ModuleLocation,
    ProvenanceError,
    RuntimeIdentity,
    runtime_identity,
)
from causalab.provenance import (
    _digest_of,
    _direct_url,
    _file_digests,
    _files,
    _fold,
    _modules,
    _package_root,
    _source,
)

REPO = Path(__file__).resolve().parents[1]


def _run_git(tree: Path, *args: str) -> str:
    """git in ``tree`` with an identity configured, so a commit works on a
    runner that has none."""
    return subprocess.run(
        [
            "git",
            "-C",
            str(tree),
            "-c",
            "user.email=t@example.invalid",
            "-c",
            "user.name=t",
            "-c",
            "commit.gpgsign=false",
            *args,
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


# --------------------------------------------------------------------------- #
# the PEP 610 mapping, from synthetic metadata
# --------------------------------------------------------------------------- #


class _Dist:
    """Just enough of an `importlib.metadata.Distribution` for `_source`."""

    def __init__(self, *, wheel: bool = True) -> None:
        self._wheel = wheel

    def read_text(self, name: str) -> str | None:
        if name == "WHEEL" and self._wheel:
            return "Wheel-Version: 1.0\n"
        return None


class TestSourceKind:
    pytestmark = pytest.mark.unit

    def test_an_editable_install_is_editable(self, tmp_path: Path) -> None:
        kind, tree = _source(
            _Dist(),
            {"url": tmp_path.as_uri(), "dir_info": {"editable": True}},
        )
        assert kind == "editable"
        assert tree == tmp_path

    def test_a_local_directory_build_is_an_sdist(self, tmp_path: Path) -> None:
        """A build over sources, not a published artifact — which is what
        `sdist` names. The tree still comes back, so the identity can say
        where the install came from."""
        kind, tree = _source(_Dist(), {"url": tmp_path.as_uri(), "dir_info": {}})
        assert kind == "sdist"
        assert tree == tmp_path

    def test_a_vcs_install_is_git(self) -> None:
        kind, tree = _source(
            _Dist(),
            {
                "url": "https://github.com/goodfire-ai/causalab",
                "vcs_info": {"vcs": "git", "commit_id": "a" * 40},
            },
        )
        assert kind == "git"
        assert tree is None  # a remote URL has no local tree to consult

    @pytest.mark.parametrize(
        ("url", "expected"),
        [
            ("https://example.invalid/causalab-1-py3-none-any.whl", "wheel"),
            ("https://example.invalid/causalab-1.tar.gz", "sdist"),
        ],
    )
    def test_an_archive_install_is_read_from_its_extension(
        self, url: str, expected: str
    ) -> None:
        kind, _tree = _source(_Dist(), {"url": url, "archive_info": {}})
        assert kind == expected

    def test_a_local_archive_has_no_source_tree(self, tmp_path: Path) -> None:
        """An archive is a file, not a tree: there is no source directory the
        install reads, so none is offered — a `file:///x.tar.gz` install must
        not be described by the directory the tarball sits in."""
        archive = tmp_path / "causalab-1.tar.gz"
        kind, tree = _source(_Dist(), {"url": archive.as_uri(), "archive_info": {}})
        assert kind == "sdist"
        assert tree is None

    def test_no_direct_url_is_an_index_install(self) -> None:
        """The common case, and the one with no origin to record: PEP 610 is
        only written for a *direct* URL."""
        assert _source(_Dist(wheel=True), None)[0] == "wheel"
        assert _source(_Dist(wheel=False), None)[0] == "sdist"

    def test_metadata_that_says_nothing_raises(self) -> None:
        """The "unknown" this module exists to remove: a `direct_url.json` with
        none of the three info blocks cannot be summarized, so it raises."""
        with pytest.raises(ProvenanceError, match="cannot say what kind"):
            _source(_Dist(), {"url": "https://example.invalid/x"})

    def test_a_url_less_record_raises(self) -> None:
        with pytest.raises(ProvenanceError, match="carries no 'url'"):
            _source(_Dist(), {"dir_info": {}})

    def test_the_kinds_are_the_declared_ones(self) -> None:
        assert set(SOURCE_KINDS) == {"git", "editable", "sdist", "wheel"}


# --------------------------------------------------------------------------- #
# the tree digest
# --------------------------------------------------------------------------- #


class TestTreeDigest:
    pytestmark = pytest.mark.unit

    def _tree(self, root: Path, files: dict[str, str]) -> Path:
        for name, body in files.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(body)
        return root

    def test_content_changes_the_digest(self, tmp_path: Path) -> None:
        root = self._tree(tmp_path / "pkg", {"a.py": "x = 1"})
        before, _ = _digest_of(root, sorted(root.rglob("*.py")))
        (root / "a.py").write_text("x = 2")
        after, _ = _digest_of(root, sorted(root.rglob("*.py")))
        assert before != after

    def test_a_rename_changes_the_digest(self, tmp_path: Path) -> None:
        """Path *and* content are hashed. A digest over bytes alone would call
        two different layouts of the same code identical, which is not what
        "the tree that will execute" means."""
        one = self._tree(tmp_path / "one", {"a.py": "x = 1"})
        two = self._tree(tmp_path / "two", {"b.py": "x = 1"})
        assert (
            _digest_of(one, sorted(one.rglob("*.py")))[0]
            != (_digest_of(two, sorted(two.rglob("*.py")))[0])
        )

    def test_the_digest_is_stable_across_calls(self, tmp_path: Path) -> None:
        root = self._tree(tmp_path / "pkg", {"a.py": "x = 1", "b/c.py": "y = 2"})
        files = sorted(p for p in root.rglob("*") if p.is_file())
        assert _digest_of(root, files) == _digest_of(root, files)

    def test_modules_localize_a_difference(self, tmp_path: Path) -> None:
        """The reason `modules` exists: two trees differing in one subpackage
        say *which*, instead of differing in one 64-hexit number."""
        root = self._tree(
            tmp_path / "pkg", {"protocol/a.py": "x = 1", "neural/b.py": "y = 2"}
        )
        files = sorted(p for p in root.rglob("*") if p.is_file())
        before = {m.name: m.digest for m in _modules(root, _file_digests(root, files))}
        (root / "protocol" / "a.py").write_text("x = 99")
        after = {m.name: m.digest for m in _modules(root, _file_digests(root, files))}
        changed = [name for name in before if before[name] != after[name]]
        assert changed == ["pkg.protocol"]

    def test_the_digest_does_not_depend_on_the_callers_order(
        self, tmp_path: Path
    ) -> None:
        """Sorted inside the digest, on the relative POSIX string, so the value
        is the same whatever order — or platform-specific `PurePath` order —
        the caller happened to walk the tree in."""
        root = self._tree(
            tmp_path / "pkg", {"a.py": "1", "a/b.py": "2", "a.b.py": "3", "B.py": "4"}
        )
        files = sorted(p for p in root.rglob("*") if p.is_file())
        forward, _ = _digest_of(root, files)
        backward, _ = _digest_of(root, list(reversed(files)))
        assert forward == backward
        relatives = [relative for relative, _ in _file_digests(root, reversed(files))]
        assert relatives == sorted(relatives)

    def test_the_tree_digest_is_the_fold_of_the_per_file_digests(
        self, tmp_path: Path
    ) -> None:
        """One hash per file: the tree digest and the module digests are the
        same fold over the same entries, so `_digest_of` is just the two
        steps composed."""
        root = self._tree(tmp_path / "pkg", {"a.py": "x = 1", "b/c.py": "y = 2"})
        files = sorted(p for p in root.rglob("*") if p.is_file())
        entries = _file_digests(root, files)
        assert _digest_of(root, files) == _fold(entries)
        assert _fold(entries)[1] == 2


class TestShippedFiles:
    """`_files` describes what ships — not whatever is present."""

    pytestmark = pytest.mark.unit

    def _package(self, root: Path) -> Path:
        for name, body in {
            "__init__.py": "",
            "core.py": "x = 1",
            "types.pyi": "x: int",
            "py.typed": "",
            "tasks/t/data/table.json": "{}",
            "tasks/t/config.yaml": "a: 1",
            "tasks/t/README.md": "# t",
        }.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(body)
        return root

    def test_the_allowlist_is_what_ships(self, tmp_path: Path) -> None:
        root = self._package(tmp_path / "pkg")
        (root / "tasks" / "t" / "demo.ipynb").write_text("{}")
        (root / "tasks" / "t" / "notes.txt").write_text("scratch")
        (root / "core.pyc").write_bytes(b"\0")
        relatives = sorted(p.relative_to(root).as_posix() for p in _files(root))
        assert relatives == [
            "__init__.py",
            "core.py",
            "py.typed",
            "tasks/t/README.md",
            "tasks/t/config.yaml",
            "tasks/t/data/table.json",
            "types.pyi",
        ]

    @pytest.mark.parametrize(
        "written",
        [
            # the four runtime output shapes `.gitignore` names under
            # /causalab/tasks/**; a task run materializes them beside its code
            "tasks/t/outputs/scores.json",
            "tasks/t/sweep_results/iia.json",
            "tasks/t/task_datasets/table.json",
            "tasks/t/run_logs/events.md",
            # caches and build residue
            "tasks/t/.ipynb_checkpoints/demo-checkpoint.json",
            ".pytest_cache/v/cache/nodeids.json",
            "causalab.egg-info/PKG-INFO.md",
            "__pycache__/core.cpython-310.json",
        ],
    )
    def test_a_run_writing_under_the_package_does_not_move_the_digest(
        self, tmp_path: Path, written: str
    ) -> None:
        """The digest attests the package a run executes; a run that writes a
        dataset or its results under that package must not change it, or two
        runs on identical code disagree because one ran a task first."""
        root = self._package(tmp_path / "pkg")
        before = _digest_of(root, _files(root))
        path = root / written
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("written by a run")
        assert _digest_of(root, _files(root)) == before

    def test_no_shipped_directory_is_mistaken_for_runtime_output(self) -> None:
        """The inverse guard: the directory-name rules must not swallow a
        subpackage that actually ships. Checked against what git tracks in this
        repo, so adding a `causalab/.../results/` package fails here rather than
        silently vanishing from every tree digest."""
        tracked = subprocess.run(
            ["git", "-C", str(REPO), "ls-files", "--", "causalab"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split()
        shipped = {p for p in _files(REPO / "causalab")}
        missing = [
            path
            for path in tracked
            if path.endswith((".py", ".pyi", ".json", ".yaml", ".yml", ".md"))
            and REPO / path not in shipped
        ]
        assert missing == []


# --------------------------------------------------------------------------- #
# where the bytes live, and what the metadata says about them
# --------------------------------------------------------------------------- #


class _LocatedDist:
    """A distribution whose `locate_file` names a site-packages copy."""

    def __init__(self, site: Path, *, direct_url: str | None = None) -> None:
        self._site = site
        self._direct_url = direct_url
        self.metadata = {"Name": "causalab"}

    def locate_file(self, name: str) -> Path:
        return self._site / name

    def read_text(self, name: str) -> str | None:
        return self._direct_url if name == "direct_url.json" else None


class TestPackageRoot:
    pytestmark = pytest.mark.unit

    @pytest.fixture
    def site_and_tree(self, tmp_path: Path) -> tuple[Path, Path]:
        """An installed copy in a fake site-packages, *and* the clone it was
        installed from — both hold a `causalab/` directory."""
        site = tmp_path / "site-packages"
        (site / "causalab").mkdir(parents=True)
        tree = tmp_path / "clone"
        (tree / "causalab").mkdir(parents=True)
        return site, tree

    def test_an_editable_install_runs_from_its_tree(
        self, site_and_tree: tuple[Path, Path]
    ) -> None:
        site, tree = site_and_tree
        assert (
            _package_root(_LocatedDist(site), "causalab", "editable", tree)
            == tree / "causalab"
        )

    @pytest.mark.parametrize("kind", ["git", "sdist", "wheel"])
    def test_a_copied_install_runs_from_site_packages(
        self, site_and_tree: tuple[Path, Path], kind: str
    ) -> None:
        """A `git+file:///clone@sha` install records `file:///clone` as its
        URL, so the clone is a real directory holding a `causalab/` — and it is
        not what runs. The installer copied the bytes; the identity must
        describe the copy, or editing the clone afterwards moves the digest of
        an install whose bytes did not change."""
        site, tree = site_and_tree
        assert (
            _package_root(_LocatedDist(site), "causalab", kind, tree)
            == site / "causalab"
        )

    def test_metadata_and_files_disagreeing_raises(self, tmp_path: Path) -> None:
        """The raise branch: `locate_file` names a directory that is not there.
        Nothing to hash is not an empty digest — it is the environment being
        unable to say what would execute."""
        with pytest.raises(ProvenanceError, match="metadata and its files disagree"):
            _package_root(_LocatedDist(tmp_path / "gone"), "causalab", "wheel", None)


class TestDirectUrl:
    pytestmark = pytest.mark.unit

    def test_malformed_json_raises(self, tmp_path: Path) -> None:
        """A `direct_url.json` that cannot be parsed is a broken install, not
        an index install: returning `None` here would quietly reclassify it as
        `wheel` with no origin."""
        with pytest.raises(ProvenanceError, match="not valid JSON"):
            _direct_url(_LocatedDist(tmp_path, direct_url="{not json"))

    def test_an_absent_record_is_an_index_install(self, tmp_path: Path) -> None:
        assert _direct_url(_LocatedDist(tmp_path)) is None


# --------------------------------------------------------------------------- #
# the live environment
# --------------------------------------------------------------------------- #


class TestThisInstall:
    pytestmark = pytest.mark.unit

    @pytest.fixture(autouse=True)
    def _fresh(self) -> None:
        # each test reads the process for itself: the cache would otherwise let
        # whichever test ran first fix the value every later one sees
        runtime_identity.cache_clear()

    def test_it_describes_itself(self) -> None:
        identity = runtime_identity()
        assert isinstance(identity, RuntimeIdentity)
        assert identity.distribution == "causalab"
        assert identity.source_kind in SOURCE_KINDS
        assert identity.tree_digest and len(identity.tree_digest) == 64
        assert identity.modules and all(
            isinstance(m, ModuleLocation) for m in identity.modules
        )

    def test_unknown_is_never_a_value(self) -> None:
        """The property that distinguishes this from `code_commit()`. Checked
        over the serialized form so a field added later is covered too."""
        blob = json.dumps(runtime_identity().to_dict())
        assert "unknown" not in blob.lower()

    def test_short_revision_always_identifies_content(self) -> None:
        """What replaces `code_commit()`'s return value in the output identity:
        the same shape — a short hex string — with no placeholder branch."""
        identity = runtime_identity()
        short = identity.short_revision
        assert len(short) == 12
        assert all(c in "0123456789abcdef" for c in short)
        assert identity.tree_digest.startswith(short)

    def test_the_receipt_form_round_trips_through_json(self) -> None:
        """The run receipt carries this, so it has to serialize."""
        assert json.loads(json.dumps(runtime_identity().to_dict()))

    def test_a_missing_distribution_raises(self) -> None:
        with pytest.raises(ProvenanceError, match="not an installed distribution"):
            runtime_identity("no-such-distribution-8f3a")


# --------------------------------------------------------------------------- #
# three real installs — the acceptance test
# --------------------------------------------------------------------------- #

#: The one-liner each install is asked to run. It imports only
#: `causalab.provenance`, which is torch-free, so a `--no-deps` install answers.
PROBE = (
    "import json;"
    "from causalab.provenance import runtime_identity;"
    "print(json.dumps(runtime_identity().to_dict()))"
)


def _venv(root: Path) -> Path:
    """A bare venv, and the python inside it.

    `uv venv` rather than the stdlib's: `ensurepip` aborts on uv-managed
    standalone interpreters, which is what this repo runs on. `uv` is the
    project's own tool and is on PATH wherever the suite runs.
    """
    subprocess.run(
        ["uv", "venv", "--python", sys.executable, str(root)],
        capture_output=True,
        text=True,
        check=True,
    )
    python = root / ("Scripts" if sys.platform == "win32" else "bin") / "python"
    assert python.exists(), f"uv venv produced no interpreter at {python}"
    return python


def _install(python: Path, *spec: str) -> subprocess.CompletedProcess[str]:
    """`uv pip install --no-deps` into one venv.

    `--no-deps` is what makes this tier cheap: `causalab.provenance` is
    torch-free, so describing an install needs none of the runtime's
    dependencies.
    """
    return subprocess.run(
        ["uv", "pip", "install", "--python", str(python), "--no-deps", *spec],
        capture_output=True,
        text=True,
    )


#: stderr fragments that name a precondition the git-install test cannot
#: supply itself: no `git` on PATH, or no network for the build backend's
#: isolated environment. Anything else is a failure, not a skip.
_INSTALL_PRECONDITIONS = (
    "git: command not found",
    "No such file or directory (os error 2)",
    "failed to resolve address",
    "Temporary failure in name resolution",
    "Could not connect",
    "Network is unreachable",
    "dns error",
)


def _describe(python: Path) -> dict:
    result = subprocess.run(
        [str(python), "-c", PROBE], capture_output=True, text=True, check=True
    )
    return json.loads(result.stdout)


@pytest.fixture(scope="module")
def wheel(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("wheel")
    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(out)],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    )
    built = sorted(out.glob("*.whl"))
    assert built, "uv build produced no wheel"
    return built[0]


@pytest.mark.smoke
class TestRealInstalls:
    """`runtime_identity()` on installs it did not choose.

    The editable case is the live environment and is covered above. These two
    are the ones a report actually hit: a wheel (which records no request) and
    a git URL (which records the requested ref, and needs no checkout to say
    so).
    """

    def test_the_live_environment_is_an_editable_or_wheel_install(self) -> None:
        """Stated rather than assumed, because every claim below is relative to
        it: `uv sync` installs this repo editable."""
        assert runtime_identity().source_kind in ("editable", "wheel", "sdist")

    def test_a_wheel_install_describes_itself_as_a_wheel(
        self, wheel: Path, tmp_path: Path
    ) -> None:
        python = _venv(tmp_path / "venv")
        installed = _install(python, str(wheel))
        assert installed.returncode == 0, installed.stderr
        identity = _describe(python)
        assert identity["source_kind"] == "wheel"
        # a wheel from a local path records its origin but no request — and
        # that absence is a fact, so the tree digest is what identifies it
        assert identity["requested_revision"] is None
        assert len(identity["tree_digest"]) == 64
        assert "unknown" not in json.dumps(identity).lower()

    def test_a_git_install_names_the_requested_revision(self, tmp_path: Path) -> None:
        """The requested ref, on the install kind that records one.

        A local source repository is used as the VCS URL so the test needs no
        history objects from the developer's potentially partial clone. The
        ref comes from the installer's own record, which is why it is readable
        without importing the package or having a checkout.
        """
        source = tmp_path / "source"
        source.mkdir()
        # Snapshot tracked package/build inputs, including working-tree fixes.
        # A complete one-commit repository keeps uv's real Git install local.
        tracked = subprocess.run(
            [
                "git",
                "-C",
                str(REPO),
                "ls-files",
                "-z",
                "--",
                "causalab",
                "pyproject.toml",
                "README.md",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split("\0")
        for name in filter(None, tracked):
            target = source / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPO / name, target)
        _run_git(source, "init", "-q")
        _run_git(source, "add", "-A")
        _run_git(source, "commit", "-q", "-m", "package under test")
        head = _run_git(source, "rev-parse", "HEAD")
        python = _venv(tmp_path / "venv")
        install = _install(python, f"causalab @ git+{source.as_uri()}@{head}")
        if install.returncode != 0:
            # Skip only on a named precondition this test cannot supply — no
            # `git`, or no network for the isolated build environment. Any
            # other failure (uv changing how it records VCS metadata, the build
            # backend breaking) is exactly what this test exists to hear about,
            # and a blanket skip would turn the headline property optional.
            unavailable = [m for m in _INSTALL_PRECONDITIONS if m in install.stderr]
            if unavailable:
                pytest.skip(f"git install precondition missing ({unavailable[0]!r})")
            raise AssertionError(install.stderr)
        identity = _describe(python)
        assert identity["source_kind"] == "git"
        assert identity["requested_revision"] == head
        # the installer copied the bytes into the venv: that copy is what runs,
        # so the identity describes it and not this working tree
        location = Path(identity["location"]).resolve()
        assert python.parent.parent.resolve() in location.parents
        assert REPO not in location.parents
        assert len(identity["tree_digest"]) == 64
        assert identity["origin"] and identity["origin"].startswith("file://")
        assert "unknown" not in json.dumps(identity).lower()

    def test_an_uncommitted_edit_moves_the_tree_digest(self, tmp_path: Path) -> None:
        """The tree digest says *the bytes that will run* are different — no
        revision does, which is what this module is about: a report needs the
        digest to compare two runs of the same checkout.
        """
        clone = tmp_path / "clone"
        subprocess.run(
            ["git", "clone", "-q", "--depth", "1", f"file://{REPO}", str(clone)],
            capture_output=True,
            text=True,
            check=True,
        )
        python = _venv(tmp_path / "venv")
        installed = _install(python, "-e", str(clone))
        assert installed.returncode == 0, installed.stderr
        clean = _describe(python)
        assert clean["source_kind"] == "editable"
        assert len(clean["tree_digest"]) == 64

        # any tracked module of the clone; picked at runtime rather than named,
        # so the test does not depend on this branch's own files existing at the
        # cloned revision
        target = sorted((clone / "causalab").rglob("*.py"))[0]
        target.write_text(target.read_text() + "\n# an edit\n")
        edited = _describe(python)
        assert edited["tree_digest"] != clean["tree_digest"]
