"""Identify the installed runtime and its source files.

``RuntimeIdentity.tree_digest`` hashes the package files selected by the shipped
file rules. It is available for both wheel and checkout installs. Git revision
metadata describes the checkout when available. Reuse checks compare runtime
identities to detect changes to installed code."""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Literal

__all__ = [
    "ModuleLocation",
    "ProvenanceError",
    "RuntimeIdentity",
    "SOURCE_KINDS",
    "SourceKind",
    "runtime_identity",
]

#: How the running copy got here. Four kinds, and the mapping from PEP 610 is
#: stated rather than inferred (see `_source`):
#:
#: * ``git`` — installed from a VCS URL; the requested ref is recorded by the
#:   installer and needs no local checkout to read;
#: * ``editable`` — a source tree on ``sys.path``; the bytes that run are
#:   hashed from that tree;
#: * ``sdist`` — built from a source tree or archive, so there was a build step
#:   over sources rather than a published wheel;
#: * ``wheel`` — a built artifact, from an index or a local ``.whl``.
SourceKind = Literal["git", "editable", "sdist", "wheel"]
SOURCE_KINDS: tuple[SourceKind, ...] = ("git", "editable", "sdist", "wheel")

#: What a distribution *ships* and the runtime executes or loads: code,
#: extension modules, the typing marker, and the data files causalab reads
#: beside its code (task tables, method and workflow configs, docs). An
#: allowlist rather than a denylist, because a denylist describes "every file
#: present" and rots the first time a run writes something new under the
#: package; the allowlist describes the claim the digest makes.
_SHIPPED_SUFFIXES = frozenset(
    {".py", ".pyi", ".so", ".pyd", ".json", ".yaml", ".yml", ".md"}
)
_SHIPPED_NAMES = frozenset({"py.typed"})

#: Never hashed into a tree digest, whatever they contain. Caches are a property
#: of the interpreter that ran, not of the code that will run; build residue
#: (``*.egg-info``) is metadata about the code; and ``outputs`` is a runtime
#: output directory a task run materializes *inside its own package*
#: (``.gitignore``: ``/causalab/tasks/**/outputs/``). A run that wrote there
#: would otherwise change the digest of the very package it attests.
_IGNORED_DIRS = frozenset(
    {
        "__pycache__",
        ".git",
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
        ".ipynb_checkpoints",
        "outputs",
    }
)
#: The other runtime output shapes ``.gitignore`` names under ``causalab/tasks``
#: — ``*results``, ``*datasets``, ``*logs`` — are directory-name *suffixes*.
_IGNORED_DIR_SUFFIXES = ("results", "datasets", "logs", ".egg-info")


class ProvenanceError(RuntimeError):
    """The runtime cannot describe itself.

    Raised rather than summarized. Every caller of this module is asking the
    question "may I trust a number this run produces", and a placeholder answer
    to that question is worse than no answer — it is the ``"unknown"`` this
    module exists to remove.
    """


@dataclasses.dataclass(frozen=True)
class ModuleLocation:
    """One subpackage of the running distribution, and the digest of its bytes.

    Present so that a tree-digest mismatch can be *localized*: two installs
    differing in one module say so, instead of differing in a single 64-hexit
    number with no way to narrow it.
    """

    name: str
    #: Relative to the package root, POSIX-separated, so the value is the same
    #: on every platform.
    path: str
    digest: str
    files: int


@dataclasses.dataclass(frozen=True)
class RuntimeIdentity:
    """What is installed, where it came from, and what will execute."""

    distribution: str
    source_kind: SourceKind
    #: Absolute path of the package root whose bytes will run.
    location: str
    #: The URL the install came from (PEP 610 ``url``), or ``None`` for an
    #: install from an index, which records no origin.
    origin: str | None
    #: What the install *asked for* — a branch, a tag, a ref. ``None`` when
    #: the install records no request, which an index install does not. What
    #: is *installed* is [`tree_digest`][], never a revision.
    requested_revision: str | None
    #: Deterministic digest over every file that will execute. Always present.
    tree_digest: str
    modules: tuple[ModuleLocation, ...]
    dependencies: tuple[tuple[str, str], ...]

    @property
    def short_revision(self) -> str:
        """A short content identity that always exists: the first 12 hex of
        [`tree_digest`][], for every install kind. This is what replaces
        ``code_commit()``'s return value in the output identity, so that field
        keeps its shape — a short hex string — while losing its ability to say
        ``"unknown"``.
        """
        return self.tree_digest[:12]

    def to_dict(self) -> dict[str, Any]:
        """A JSON-serializable form, for the run receipt.

        Sorted and free of timestamps, hostnames and absolute paths *except*
        ``location``, which the receipt's `observed` section carries and its
        `verification` section does not — the shard-identical requirement lands
        on the latter.
        """
        return {
            "distribution": self.distribution,
            "source_kind": self.source_kind,
            "location": self.location,
            "origin": self.origin,
            "requested_revision": self.requested_revision,
            "tree_digest": self.tree_digest,
            "modules": [dataclasses.asdict(m) for m in self.modules],
            "dependencies": dict(self.dependencies),
        }


# --------------------------------------------------------------------------- #
# reading the install, without importing it
# --------------------------------------------------------------------------- #


def _distribution(name: str) -> Any:
    from importlib.metadata import PackageNotFoundError, distribution

    try:
        return distribution(name)
    except PackageNotFoundError as error:
        raise ProvenanceError(
            f"{name!r} is not an installed distribution, so its provenance "
            "cannot be read. An import-path install (a bare `sys.path` entry, "
            "a `PYTHONPATH` shim) has no metadata to attest — install it, "
            "editable is enough"
        ) from error


def _direct_url(dist: Any) -> dict[str, Any] | None:
    """PEP 610's record of where this install came from.

    Read as *metadata text*: this is the whole reason the check does not import
    the package. Absent for an install from an index, which is a fact about the
    install and not a failure.
    """
    try:
        raw = dist.read_text("direct_url.json")
    except (OSError, KeyError):  # pragma: no cover - unreadable dist-info
        return None
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ProvenanceError(
            f"{dist.metadata['Name']}'s direct_url.json is not valid JSON, so "
            "the install cannot say where it came from"
        ) from error
    return parsed if isinstance(parsed, dict) else None


def _url_path(url: str) -> Path | None:
    """The local path a ``file://`` URL names, or ``None`` for a remote URL."""
    if not url.startswith("file://"):
        return None
    from urllib.parse import unquote, urlparse

    return Path(unquote(urlparse(url).path))


def _source(dist: Any, direct: dict[str, Any] | None) -> tuple[SourceKind, Path | None]:
    """``(source_kind, source_tree)`` — the mapping from PEP 610, stated.

    ``source_tree`` is the local directory this install came from, or ``None``
    when there is none. Only an editable install *reads* it; for the other
    kinds it says where the bytes were copied from and nothing more.
    """
    if direct is None:
        # No origin recorded: an install from an index. The artifact is a wheel
        # when the dist-info says so, and otherwise an sdist that was built here.
        try:
            wheel = dist.read_text("WHEEL")
        except (OSError, KeyError):  # pragma: no cover
            wheel = None
        return ("wheel" if wheel else "sdist"), None

    url = direct.get("url")
    if not isinstance(url, str):
        raise ProvenanceError(
            "direct_url.json carries no 'url', so the install records an "
            "origin it cannot name (PEP 610 requires one)"
        )
    if isinstance(direct.get("vcs_info"), dict):
        # A VCS install records what it asked for, and the installer *copied*
        # the bytes out of the clone — so the local tree, if the URL is a local
        # clone, is not what runs. It is returned so the identity can say where
        # the install came from; nothing about the running copy is read from it
        # (see `_package_root`).
        return "git", _url_path(url)
    if isinstance(direct.get("dir_info"), dict):
        tree = _url_path(url)
        if direct["dir_info"].get("editable"):
            return "editable", tree
        # built from a source tree: a build step over sources, not a published
        # artifact — which is what `sdist` names here
        return "sdist", tree
    if isinstance(direct.get("archive_info"), dict):
        # An archive is a file, not a tree: there is no source directory this
        # install reads, even when the archive is local.
        return ("wheel" if url.endswith(".whl") else "sdist"), None
    raise ProvenanceError(
        f"direct_url.json for {url!r} carries none of vcs_info / dir_info / "
        "archive_info, so PEP 610 cannot say what kind of install this is"
    )


def _module_name(distribution: str) -> str:
    """The import package a distribution name conventionally installs."""
    return distribution.replace("-", "_")


def _package_root(dist: Any, name: str, kind: SourceKind, tree: Path | None) -> Path:
    """Where the bytes that will execute actually live.

    For an editable install that is the source tree, not ``site-packages`` —
    which is the distinction a tree digest has to get right, because an editable
    install's ``site-packages`` holds a link and no code. For every *other*
    kind the installer copied files out of the tree (a ``git+file://`` clone, a
    ``pip install .`` directory), so the tree is where the revision came from
    and hashing it would describe a copy that will not run — and would move
    the identity every time someone edits the clone afterwards.
    """
    module = _module_name(name)
    if kind == "editable" and tree is not None:
        candidate = tree / module
        if candidate.is_dir():
            return candidate
    located = dist.locate_file(module)
    root = Path(str(located))
    if root.is_dir():
        return root
    raise ProvenanceError(
        f"cannot find the {module!r} package under {root} — the install's "
        "metadata and its files disagree, so there is nothing to hash"
    )


# --------------------------------------------------------------------------- #
# hashing what will run
# --------------------------------------------------------------------------- #


def _is_ignored_dir(name: str) -> bool:
    return name in _IGNORED_DIRS or name.endswith(_IGNORED_DIR_SUFFIXES)


def _files(root: Path) -> list[Path]:
    """Every file that ships, sorted: what executes or loads, and nothing a run
    wrote under the package afterwards.

    Membership is by the allowlist (`_SHIPPED_SUFFIXES`,
    `_SHIPPED_NAMES`) and the excluded directories, so the same tree
    digests the same before and after a task run materializes its datasets,
    results or outputs beside the task's code.
    """
    out: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix not in _SHIPPED_SUFFIXES and path.name not in _SHIPPED_NAMES:
            continue
        if any(_is_ignored_dir(part) for part in path.relative_to(root).parts[:-1]):
            continue
        out.append(path)
    return out


#: ``(relative POSIX path, content digest)`` — one file, hashed once.
FileDigest = tuple[str, bytes]


def _file_digests(root: Path, paths: Iterable[Path]) -> list[FileDigest]:
    """Each file under ``root`` hashed once, sorted by its relative POSIX path.

    Sorted *here*, on the string, rather than inherited from the caller's
    iteration order: ``PurePath`` ordering compares path components and is
    case-normalized on Windows, so a digest that inherited it would differ
    between platforms for an identical tree. The string order is the same
    everywhere, which is what makes the tree digest platform-independent
    rather than incidentally so.
    """
    out = [
        (path.relative_to(root).as_posix(), hashlib.sha256(path.read_bytes()).digest())
        for path in paths
    ]
    out.sort(key=lambda entry: entry[0])
    return out


def _fold(entries: Iterable[FileDigest]) -> tuple[str, int]:
    """``(digest, count)`` over pre-hashed files.

    Path *and* content are folded in, so a renamed file changes the digest: a
    tree digest that only covered bytes would call two different layouts
    identical. The tree digest and each module digest are this same fold over
    subsets of the same entries, so the relationship between them is by
    construction rather than by re-reading every file.
    """
    digest = hashlib.sha256()
    count = 0
    for relative, content in entries:
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(content)
        count += 1
    return digest.hexdigest(), count


def _digest_of(root: Path, paths: Iterable[Path]) -> tuple[str, int]:
    """``(digest, count)`` over ``paths``, relative to ``root``: hash then fold."""
    return _fold(_file_digests(root, paths))


def _modules(root: Path, entries: Iterable[FileDigest]) -> tuple[ModuleLocation, ...]:
    """One entry per top-level member of the package, so a mismatch localizes."""
    groups: dict[str, list[FileDigest]] = {}
    for relative, content in entries:
        groups.setdefault(relative.split("/", 1)[0], []).append((relative, content))
    out: list[ModuleLocation] = []
    for member, members in sorted(groups.items()):
        digest, count = _fold(members)
        out.append(
            ModuleLocation(
                name=f"{root.name}.{member[:-3] if member.endswith('.py') else member}",
                path=member,
                digest=digest,
                files=count,
            )
        )
    return tuple(out)


def _requested_revision(direct: dict[str, Any] | None) -> str | None:
    if not direct:
        return None
    info = direct.get("vcs_info")
    if isinstance(info, dict):
        requested = info.get("requested_revision")
        return str(requested) if requested else None
    return None


def _dependencies(dist: Any) -> tuple[tuple[str, str], ...]:
    """Installed versions of everything this distribution requires.

    Names only, resolved against the live environment — the *requirement* text
    is in the metadata already, and what a run needs recorded is what is
    actually importable beside it.
    """
    from importlib.metadata import PackageNotFoundError, version

    import re

    out: dict[str, str] = {}
    for requirement in dist.requires or ():
        match = re.match(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)", str(requirement))
        if not match:
            continue
        name = match.group(1)
        if name in out:
            continue
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            continue  # an extra's dependency, not installed here
    return tuple(sorted(out.items()))


# --------------------------------------------------------------------------- #
# the one entry point
# --------------------------------------------------------------------------- #


@functools.cache
def runtime_identity(distribution: str = "causalab") -> RuntimeIdentity:
    """What is installed and running, as data.

    Cached: the answer is a property of the process, and a run that consulted it
    twice and got two answers would be recording something other than what it
    executed. Call ``runtime_identity.cache_clear()`` in a test that changes the
    tree underneath it.
    """
    dist = _distribution(distribution)
    direct = _direct_url(dist)
    kind, tree = _source(dist, direct)
    root = _package_root(dist, distribution, kind, tree)
    files = _files(root)
    if not files:
        raise ProvenanceError(
            f"the {distribution!r} package at {root} contains no files, so "
            "there is nothing that could execute"
        )
    entries = _file_digests(root, files)
    tree_digest, _ = _fold(entries)
    return RuntimeIdentity(
        distribution=distribution,
        source_kind=kind,
        location=str(root),
        origin=str(direct["url"]) if direct else None,
        requested_revision=_requested_revision(direct),
        tree_digest=tree_digest,
        modules=_modules(root, entries),
        dependencies=_dependencies(dist),
    )
