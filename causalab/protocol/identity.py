"""Compute document, source, and artifact identities from bytes.

Canonical JSON uses sorted keys, normalized numbers, and finite values.
``sign_step`` canonicalizes a concrete point through the schema before hashing.
``source_sha256`` hashes a referenced module's full source file, including prose.
A static import walk records the sibling dependencies of external scripts.
The installed package has its own runtime tree digest.

Artifact identity records the fields that fitted tensor files must match when
loaded. Static code checks inspect literal reads in the AST; dynamic reads
remain outside their coverage. Module lookup may import parent packages, which
must support import during document validation."""

from __future__ import annotations

import ast
import dataclasses
import functools
import hashlib
import importlib.util
import json
import site
import sysconfig
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from causalab.protocol.rules.errors import ValidationError
from causalab.protocol.rules.code import (
    READER_CALLS,
    _attribute_tail,  # pyright: ignore[reportPrivateUsage]
    function_def,
    signature_problems,
    undeclared_reads,
)

if TYPE_CHECKING:
    from causalab.protocol.schema import Document, SiteSpec

__all__ = [
    "ARTIFACT_IDENTITY_KEYS",
    "CODE_RULE",
    "READER_CALLS",
    "ROW_ROLE_RULE",
    "RUNTIME_KEYWORDS",
    "CodeResolutionError",
    "ResolvedCode",
    "build_artifact_identity",
    "canonical_bytes",
    "check_artifact_identity",
    "closure_sha256",
    "digest",
    "function_def",
    "import_closure",
    "is_installed_module",
    "package_root",
    "repo_root",
    "resolve_locator",
    "resolve_locator_or_refuse",
    "sign_step",
    "signature_problems",
    "single_site_featurizers",
    "site_identity",
    "source_root",
    "source_sha256",
    "spec_identity",
    "step_digest",
    "undeclared_reads",
]


# --------------------------------------------------------------------------- #
# canonical bytes and the digest (§7)
# --------------------------------------------------------------------------- #


def canonical_bytes(canonical: Mapping[str, Any]) -> bytes:
    """Serialize a canonical form to its digestable bytes: sorted keys,
    minimal separators, UTF-8, no NaN/Inf, numbers normalized (an integral
    float digests as its integer — ``1`` and ``1.0`` are one value across
    the JSON and YAML surfaces; ``-0.0`` digests as ``0``)."""
    try:
        return json.dumps(
            _normalize_numbers(canonical),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except ValueError as err:
        raise ValidationError(
            15, f"canonical form holds a non-finite number: {err}"
        ) from err


def _normalize_numbers(node: Any) -> Any:
    if isinstance(node, bool):
        return node
    if isinstance(node, float):
        if node != node or node in (float("inf"), float("-inf")):
            raise ValidationError(15, "canonical form holds a non-finite number")
        if node.is_integer() and abs(node) <= 2**53:
            return int(node)
        return node
    if isinstance(node, Mapping):
        return {key: _normalize_numbers(value) for key, value in node.items()}
    if isinstance(node, list):
        return [_normalize_numbers(item) for item in node]
    return node


def digest(canonical: Mapping[str, Any]) -> str:
    """``sha256`` of the canonical bytes, hex."""
    return hashlib.sha256(canonical_bytes(canonical)).hexdigest()


def sign_step(step_tree: Mapping[str, Any], env: Any) -> tuple[dict[str, Any], str]:
    """The one hasher of a step: a
    concrete step tree — every axis at one value — canonicalized against
    ``env`` ([`causalab.protocol.schema.explicit.canonicalize`][], no
    ``axes`` block: exactly what a point got from the compiler before
    enumeration moved engine-side) and the ``sha256`` of those bytes. Returns
    both, because the executor reads the canonical form too; the engine-side
    sweep ([`causalab.neural.shared.sweep`][]) and the workflow layer call
    this and nothing else to name a step, so the step digest is the point
    digest the corpus, golden and demo pins carry. ``env`` is the
    [`ResolutionEnv`][causalab.io.env.ResolutionEnv] the canonical form resolves
    against — typed ``Any`` because ``io/env.py`` imports this module and the
    module-level graph keeps no edge back, not even for a type; the canonical
    form's module imports this one too, so it is reached function-locally."""
    from causalab.protocol.schema.explicit import canonicalize

    canonical = canonicalize(step_tree, env)
    return canonical, digest(canonical)


def step_digest(step_tree: Mapping[str, Any], env: Any) -> str:
    """The provenance digest of one step (§7): [`sign_step`][]'s digest."""
    return sign_step(step_tree, env)[1]


# --------------------------------------------------------------------------- #
# code identity (§2.8.1)
# --------------------------------------------------------------------------- #


#: §5 rule 24 — a code reference agrees with the source it names. Every
#: refusal raised out of this module carries it, so a test pins the rule and
#: not the sentence.
CODE_RULE: int = 24


#: §5 rule 25 — declared row roles match the resolved data. Needs the tables,
#: so it lives in the ``validate --data`` pass (like rules 4's column half and
#: 20), and is raised from [`causalab.protocol.rules.data`][].
ROW_ROLE_RULE: int = 25


#: Keyword arguments the runtime supplies to a referenced function when the
#: declaration asks for them, so the signature check does not demand that the
#: author also list them under ``args``.
#:
#: Only ``row_roles``. It is the one declaration a function cannot act on
#: without being *told*: the alternative is inferring roles from physical
#: batch positions, which is the inference this exists to remove.
#: ``data_inputs`` and ``env_inputs`` are the other kind of declaration —
#: identity plus an allowlist. The function opens its own file and reads its
#: own variable, and what the declaration adds is that the file is
#: content-digested into the protocol and that an undeclared one is refused.
RUNTIME_KEYWORDS: tuple[str, ...] = ("row_roles",)


class CodeResolutionError(Exception):
    """A locator does not name Python source. Translated to the caller's
    rule number (a [`ValidationError`][]) at the
    boundary — this module knows nothing about the checklist."""


@dataclasses.dataclass(frozen=True)
class ResolvedCode:
    """Where a locator landed: the defining module, its file, and the dotted
    attribute path inside it (empty when the locator *is* a module)."""

    module: str
    path: Path
    attr: tuple[str, ...]


def source_sha256(path: Path) -> str:
    """The sha256 of a Python source file's bytes.

    One function, two callers: a workflow script step (§4.2) and a protocol
    code reference (§2.8.1) hash the same quantity the same way, so a module
    that is both cannot get two identities.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def repo_root() -> Path:
    """The directory the ``causalab`` package sits in.

    One thing resolves against it: the repository-wide import-closure walk
    ([`import_closure`][] with ``repository=True``), the layering tool, which
    only a checkout runs. The identity walk (``repository=False``) never
    resolves here, and neither does a workflow document's ``path`` reference —
    that is relative to the document (workflow spec §3), because an installed
    wheel has no repository root to offer.
    """
    return Path(__file__).resolve().parents[2]


def package_root() -> Path:
    """The ``causalab`` package directory — the tree whose bytes are the
    ``tree_digest`` every step record carries ([`causalab.provenance.runtime_identity`][])
    and ``--resume`` compares. A module under it is runtime identity and is
    excluded from every declared import closure that enters a digest."""
    return Path(__file__).resolve().parents[1]


@functools.cache
def _installation_roots() -> tuple[Path, ...]:
    """Where installed Python lives on this interpreter: the stdlib and every
    site-packages directory (``sysconfig`` and ``site`` agree on the venv's,
    the base interpreter's and the user's)."""
    paths = sysconfig.get_paths()
    roots = {paths[key] for key in ("stdlib", "platstdlib", "purelib", "platlib")}
    roots.update(site.getsitepackages())
    roots.add(site.getusersitepackages())
    return tuple(sorted(Path(root).resolve() for root in roots))


def is_installed_module(path: Path) -> bool:
    """Whether the module at ``path`` is *installed* — lives under the stdlib
    or a site-packages directory — as opposed to being repository code or a
    user's own code on ``sys.path``.

    The one predicate behind "a hashed module that itself lives outside the
    repository declares no closure; its imports are runtime identity". A
    ``code`` locator (§2.8.1) or a ``{"module": …}`` script step (workflow
    spec §4.2) may legitimately name a stdlib or third-party file — that file
    is still ``source_sha256``, because the document named it — but walking
    *its* imports with the stdlib or site-packages as the own root would admit
    everything reachable there, which is exactly the third-party code the spec
    excludes. The test is by installation directory rather than by
    ``repo_root``, for two reasons: a project venv conventionally sits *inside*
    the repository (``<repo>/.venv``), so "under the repo root" would still walk
    ``numpy``; and a user's own module outside the repository — a session-local
    package, a test's ``tmp_path`` — is the author's code, whose imported
    siblings [`import_closure`][] admits through the module's own root. Both
    callers ask here rather than each spelling the test.
    """
    resolved = path.resolve()
    return any(resolved.is_relative_to(root) for root in _installation_roots())


def source_root(resolved: ResolvedCode) -> Path:
    """The directory ``resolved.module``'s dotted name was resolved from — the
    root a sibling ``import`` inside that module resolves against."""
    depth = len(resolved.module.split("."))
    if resolved.path.name != "__init__.py":
        depth -= 1
    return resolved.path.resolve().parents[depth]


def closure_sha256(closure: Mapping[str, str]) -> str:
    """One hash over a closure manifest: ``sha256`` of the newline-joined
    ``"<path> <sha256>"`` lines, sorted by path. Empty manifest, one fixed
    value; the loaders write neither key for an empty manifest, so a module
    with no sibling imports carries no closure fields at all."""
    lines = "\n".join(f"{path} {sha}" for path, sha in sorted(closure.items()))
    return hashlib.sha256(lines.encode("utf-8")).hexdigest()


@dataclasses.dataclass(frozen=True)
class _ImportRef:
    """One ``import`` statement, as the walk needs it: the dotted module (``""``
    for ``from . import x``), the relative level, and the names a ``from``
    form binds (empty for a plain ``import``)."""

    module: str
    level: int
    names: tuple[str, ...]


#: Parsed import statements per source sha256. Content-addressed, so a file
#: rewritten in place — which the digest-moving tests do many times a second —
#: can never be served a stale parse; and a load that hashes the same seven
#: thousand lines of ``protocol/`` for its second script step parses nothing.
_IMPORTS_BY_SHA: dict[str, tuple[_ImportRef, ...]] = {}


def import_closure(
    path: Path,
    *,
    root: Path | None = None,
    include_parents: bool = False,
    repository: bool = True,
) -> dict[str, str]:
    """The declared import closure of the module at ``path``, as a manifest
    ``{relative_path: sha256}`` sorted by path — every module the module
    reaches transitively through its ``import`` statements, **not** counting
    the module itself (its own hash is [`source_sha256`][], right beside
    this in every record).

    ``root`` is the directory the module's own dotted name resolves from
    ([`source_root`][]; the file's directory for a ``{"path": …}`` script).
    A name is probed there by filesystem shape alone — ``x/y.py`` or
    ``x/y/__init__.py`` — exactly as [`resolve_locator`][] probes, and never
    through ``sys.path``: a module that lives anywhere else is third-party and
    stays out. Manifest keys are relative to the root a member was found
    under, so moving a workflow tree moves no digest.

    ``repository`` selects which of two walks this is. The default, the
    **layering** walk, probes [`repo_root`][] after ``root`` and admits every
    repository module: it is how the test suite proves a module reaches no
    engine and no numerics, and it enters no digest. ``repository=False`` is
    the **identity** walk the loaders use (workflow spec §4.2, IM spec
    §2.8.1): only ``root`` is probed, and a member under [`package_root`][]
    is dropped, because the package's bytes are runtime identity — the
    ``tree_digest`` every step record carries and ``--resume`` compares — so a
    document naming them would only move on every edit to the package. What
    remains is the code beside the hashed module that nothing else covers: a
    ``{"path": …}`` script's sibling helpers, a user package's siblings on
    ``sys.path``. A hashed module that itself lives in an installation
    directory declares no closure at all ([`is_installed_module`][]), because
    with ``root`` at the stdlib or site-packages every module reachable there
    would be admitted.

    ``include_parents`` also admits every parent package ``__init__.py``
    Python executes on the way to an imported name — the *execution* closure.
    Off by default: a parent is executed, not declared, and admitting it would
    make every parent package's ``__init__.py`` — bytes no document names —
    part of every script's identity. The switch exists so the acceptance test
    can pin the choice either way rather than leave it implicit.

    Nothing is imported. Each member is read, hashed and parsed with
    ``ast.parse``; a member that does not parse contributes its bytes and no
    edges.
    """
    start = path.resolve()
    own = (root if root is not None else start.parent).resolve()
    if repository:
        roots = [own] if own == repo_root() else [own, repo_root()]
    else:
        roots = [own]
    package = package_root()
    manifest: dict[str, str] = {}
    seen: set[Path] = {start}
    pending: list[tuple[Path, Path]] = [(start, own)]
    while pending:
        file, file_root = pending.pop()
        data = file.read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        if file != start:
            manifest[file.relative_to(file_root).as_posix()] = sha
        refs = _IMPORTS_BY_SHA.get(sha)
        if refs is None:
            refs = _import_refs(data, file)
            _IMPORTS_BY_SHA[sha] = refs
        for ref in refs:
            for target, target_root in _resolve_ref(
                ref, file, file_root, roots, include_parents
            ):
                if not repository and target.is_relative_to(package):
                    continue  # runtime identity: the package's own bytes
                if target not in seen:
                    seen.add(target)
                    pending.append((target, target_root))
    return dict(sorted(manifest.items()))


def _import_refs(data: bytes, file: Path) -> tuple[_ImportRef, ...]:
    """Every import statement in ``data`` outside ``if TYPE_CHECKING:`` bodies,
    function-local ones included."""
    try:
        tree = ast.parse(data, filename=str(file))
    except (SyntaxError, ValueError):
        return ()
    out: list[_ImportRef] = []
    stack: list[ast.AST] = [tree]
    while stack:
        node = stack.pop()
        if isinstance(node, ast.If) and _is_type_checking(node.test):
            stack.extend(node.orelse)  # the run-time branch still counts
            continue
        if isinstance(node, ast.Import):
            out.extend(_ImportRef(alias.name, 0, ()) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            out.append(
                _ImportRef(
                    node.module or "",
                    node.level,
                    tuple(alias.name for alias in node.names),
                )
            )
        stack.extend(ast.iter_child_nodes(node))
    return tuple(out)


def _is_type_checking(test: ast.expr) -> bool:
    """``if TYPE_CHECKING:`` / ``if typing.TYPE_CHECKING:`` — a block the
    interpreter never runs, so nothing in it is a dependence."""
    return _attribute_tail(test) == "TYPE_CHECKING"


def _resolve_ref(
    ref: _ImportRef,
    file: Path,
    file_root: Path,
    roots: Sequence[Path],
    include_parents: bool,
) -> list[tuple[Path, Path]]:
    """The files one import statement declares, each with the root it was
    found under; empty when the name resolves under no root (third-party)."""
    if ref.level:
        # relative: anchored in the importing module's own package (a
        # module's directory; for an ``__init__.py`` the package it defines)
        package = file.parent
        for _ in range(ref.level - 1):
            package = package.parent
        try:
            base = list(package.relative_to(file_root).parts)
        except ValueError:
            return []
        base += ref.module.split(".") if ref.module else []
        candidates: list[tuple[list[str], Path]] = [(base, file_root)]
    else:
        base = ref.module.split(".")
        candidates = [(base, root) for root in roots]

    for parts, root in candidates:
        found = _probe(root, parts)
        if found is None:
            continue
        out: list[tuple[Path, Path]] = []
        if ref.names and found.name == "__init__.py":
            # ``from pkg import name``: a submodule when one exists on disk,
            # otherwise an attribute the package's ``__init__`` defines
            for name in ref.names:
                child = _probe(root, [*parts, name]) if name != "*" else None
                out.append((child if child is not None else found, root))
        else:
            out.append((found, root))
        if include_parents:
            out.extend(
                (init, root)
                for depth in range(1, len(parts) + 1)
                if (init := root.joinpath(*parts[:depth], "__init__.py")).is_file()
            )
        return out
    return []


def _probe(root: Path, parts: Sequence[str]) -> Path | None:
    """``root/a/b/c.py`` or ``root/a/b/c/__init__.py`` for ``a.b.c``, or
    ``None`` — the shape `_child_module` probes, from a fixed root."""
    if not parts or not all(part.isidentifier() for part in parts):
        return None
    current = root
    for index, part in enumerate(parts):
        package_init = current / part / "__init__.py"
        if package_init.is_file():
            current = current / part
            continue
        module_file = current / f"{part}.py"
        if module_file.is_file() and index == len(parts) - 1:
            return module_file
        return None
    return current / "__init__.py"


def resolve_locator(locator: str) -> ResolvedCode:
    """Resolve a dotted locator to its defining module's file, **without
    importing it**.

    ``find_spec`` on a *submodule* imports its parent, and on
    ``pkg.mod.attr`` it would import ``pkg.mod`` itself — the module holding
    the user's torch code. So only the top-level name goes through
    ``find_spec`` (which imports nothing), and every further dotted part is a
    filesystem probe against the package's search locations. The walk stops at
    the first part that is not a file on disk; what remains is the attribute
    path.
    """
    parts = locator.split(".")
    if not all(part.isidentifier() for part in parts):
        raise CodeResolutionError(f"{locator!r} is not a dotted importable name")
    try:
        spec = importlib.util.find_spec(parts[0])
    except (ImportError, ValueError) as err:
        # a missing or unimportable top-level package raises rather than
        # returning None
        raise CodeResolutionError(f"{parts[0]!r} does not import: {err}") from err
    if spec is None:
        raise CodeResolutionError(f"no module named {parts[0]!r}")
    origin = spec.origin
    search = [Path(root) for root in (spec.submodule_search_locations or [])]
    consumed = 1
    while consumed < len(parts) and search:
        found = _child_module(search, parts[consumed])
        if found is None:
            break
        origin, search = str(found[0]), found[1]
        consumed += 1
    if origin is None or not origin.endswith(".py"):
        raise CodeResolutionError(
            f"{locator!r} resolves to {origin or 'a namespace package'}, which is "
            "not Python source — a code reference is identified by its source "
            "bytes, so there has to be some"
        )
    return ResolvedCode(
        module=".".join(parts[:consumed]),
        path=Path(origin),
        attr=tuple(parts[consumed:]),
    )


def _child_module(search: Sequence[Path], name: str) -> tuple[Path, list[Path]] | None:
    """``(origin, next search locations)`` for a child module found on disk."""
    for root in search:
        package_init = root / name / "__init__.py"
        if package_init.is_file():
            return package_init, [root / name]
        module_file = root / f"{name}.py"
        if module_file.is_file():
            return module_file, []
    return None


def resolve_locator_or_refuse(locator: str, *, path: str) -> ResolvedCode:
    """[`resolve_locator`][], with the failure spelled as §5 rule 24.

    Canonicalization and validation both need the resolved source and both
    have to refuse the same way, so the translation lives here rather than
    being written out twice.
    """
    try:
        return resolve_locator(locator)
    except CodeResolutionError as err:
        raise ValidationError(CODE_RULE, str(err), path=path) from err


# --------------------------------------------------------------------------- #
# ArtifactIdentity (§8)
# --------------------------------------------------------------------------- #


#: The stamped-identity schema for featurizer bundles: these keys live in the
#: safetensors ``__metadata__`` table (string-valued, per the format). The
#: engine stamps them at save; the loader refuses a ``file_path`` load whose
#: stamped values contradict the document (§2.5).
#:
#: ⚠️ **Migration.** ``model_dtype`` and ``model_quantization`` joined this
#: schema when precision entered the record (§2.1). A bundle fitted before
#: that carries neither, so it no longer matches a document that names them
#: and ``_check_loaded_featurizers`` refuses it. That is intended and not a
#: bug to route around — a rotation fitted in bf16 is not the same artifact as
#: one fitted in fp32, and pretending otherwise is what the stamp exists to
#: prevent. Locally kept fitted artifacts must be re-fitted once; nothing in
#: the repo ships one.
ARTIFACT_IDENTITY_KEYS: tuple[str, ...] = (
    "model_key",
    "model_revision",
    "model_dtype",
    "model_quantization",
    "model_attn_implementation",
    "tokenizer",
    "site",
    "k",
    "parametrization",
    # a gate's grouping (§2.5): the unit kind and its derived
    # ``[groups, group_width]`` map, so a mask fitted over 16 heads of 256 is
    # refused against a site whose heads are laid out any other way
    "group",
    "group_map",
    # a hard-concrete gate's stretch ``[γ, ζ]`` (§2.5 ``parametrization``):
    # the hard split is ``θ > logit((½−γ)/(ζ−γ))``, so a reader of the bundle
    # (``analysis.random_mask``) needs it to match the fit's own threshold, and
    # the loader compares it — an authored stretch is expected of the bundle,
    # a non-default stamp refused by a document authoring none
    # (``rules/data.py``).
    # Registering a key here also admits it per entry (``entry_identity``) and
    # carries it forward on a re-stamp (``step_io.inherited_identity``)
    "stretch",
    # a budget gate's pool (§2.5 ``pool``): a pooled member's θ is a ranking
    # only relative to its co-members, so the pool's name is compared by the
    # loader in both directions (like ``group``) and ``pool_units`` — the
    # pool's unit count — rides along as provenance a reader sizes a cut by
    "pool",
    "pool_units",
    # a position gate's ``axis`` (§2.5): θ is one entry per token position,
    # not per coordinate. Compared both ways at compile (``rules/data.py``, the
    # ``group`` shape) and per entry at the build
    # (``featurizers._check_entry_identity``); registering it here also
    # carries it forward on a re-stamp (``step_io.inherited_identity``), so a
    # script step's output inherits the axis its inputs were fitted over
    "axis",
    # a straight-through fit's ``forward`` (§2.5, the mapping form of
    # ``parametrization``): provenance only — the loader's expectation never
    # carries it and ``check_artifact_identity`` compares only the keys the
    # expectation has, so it gates no reload; registered so the stamp path
    # (``build_artifact_identity``, which refuses unknown keys) admits it.
    # Unlike ``axis`` above, deliberately no load-time clause: ``axis``
    # changes what θ's entries *index* (W positions or W coordinates),
    # ``forward`` only which loss produced them — the readout is the same
    # map's hard split either way
    "forward",
    "dtype",
    "trained_on",
    "engine",
    # Which optional engine implementations a run actually applied — e.g.
    # `attn_eager` when the nnsight engine forces eager attention to reach the
    # pattern interior. Runtime provenance of the same kind as `engine`, and
    # `neural/shared/execution.py` has always stamped it; it was missing here,
    # so any nnsight run that wrote a tensor file raised on its own stamp.
    "implementations",
    "loaded_attn_implementation",  # observed backend, inherited as runtime provenance
    "commit",
    # a fit initialised from a saved basis or theta (§2.5 ``init``): the data
    # ref the start was fitted over and, for a subspace, which columns of it
    # seeded the fit — so the record says where the fit *started*, not only
    # where it ended
    "init_trained_on",
    "init_components",
)


def build_artifact_identity(**fields: Any) -> dict[str, str]:
    """Stringify identity fields for a safetensors ``__metadata__`` table
    (the format only carries ``str -> str``). Unknown keys are refused so
    the schema stays closed; absent fields are simply not stamped."""
    unknown = set(fields) - set(ARTIFACT_IDENTITY_KEYS)
    if unknown:
        raise AssertionError(f"unknown ArtifactIdentity fields {sorted(unknown)}")
    return {
        key: value if isinstance(value, str) else json.dumps(value, sort_keys=True)
        for key, value in fields.items()
        if value is not None
    }


#: Extra guidance for the mismatches whose *cause* is not where the reader
#: looks first, as ``str.format`` templates over ``want`` (what the document
#: implies) and ``got`` (what the bundle was stamped with).
#:
#: ``model_dtype`` is the one whose cause is least obvious. A ``model`` block
#: with no ``dtype`` **implies fp32**, so an apply document that simply omits
#: the field is refused against a bf16 fit — and the fix is in the document,
#: not on the command line: ``--dtype`` exists only on a *document* run
#: (it is a ``--set model.dtype=…`` shorthand), and a workflow run has no such
#: flag at all, so a chained fit → apply can only be repaired in the file.
_MISMATCH_HINTS: dict[str, str] = {
    "model_attn_implementation": (
        '. Write "attn_implementation": "{got}" inside the document\'s "model" '
        'object, or set "model.attn_implementation" in the workflow step\'s "set" object'
    ),
    "model_dtype": (
        '. Precision is a document fact: write "dtype": "{got}" into this '
        "document's 'model' block, next to the fit's. A 'model' with no "
        "'dtype' implies 'fp32', which is what an apply document usually gets "
        "wrong. The --dtype flag is not the fix: it is a --set shorthand on a "
        "document run, and a workflow run does not accept it at all"
    ),
}


def check_artifact_identity(
    stamped: Mapping[str, Any] | None,
    expected: Mapping[str, Any],
    *,
    what: str,
) -> None:
    """Refuse a loaded bundle whose stamped identity contradicts the
    document (§2.5). A bundle with no identity at all is refused too — an
    unverifiable artifact is a provenance hole, not a pass."""
    if stamped is None:
        raise ValidationError(
            15,
            f"{what}: the artifact carries no ArtifactIdentity metadata — "
            "nothing to check, so the load refuses (§2.5)",
        )
    normalized_expected = build_artifact_identity(**expected)
    for key, want in normalized_expected.items():
        got = stamped.get(key)
        if got is not None and str(got) != want:
            raise ValidationError(
                15,
                f"{what}: ArtifactIdentity mismatch on {key!r} — the document "
                f"implies {want!r} but the bundle was stamped {got!r} (§2.5)"
                + _MISMATCH_HINTS.get(key, "").format(want=want, got=got),
            )
        if got is None:
            raise ValidationError(
                15,
                f"{what}: ArtifactIdentity is missing {key!r} — the bundle "
                "cannot prove it matches the document (§2.5)",
            )


# --------------------------------------------------------------------------- #
# site identity (§8) — the site half of an artifact stamp
# --------------------------------------------------------------------------- #


def site_identity(doc: Document, site_name: str | None) -> dict[str, Any] | None:
    """One site as the ArtifactIdentity records it — the non-null address
    fields only, the shape ``rules/data.py`` builds its expectation in."""
    if site_name is None or site_name not in doc.sites:
        return None
    return spec_identity(doc.sites[site_name])


def single_site_featurizers(doc: Document) -> dict[str, dict[str, Any]]:
    """Each featurizer that the document's reads and writes use at exactly
    one site, mapped to that site's record as a stamp carries it
    ([`site_identity`][]).

    A bundle fitted for such a featurizer must record that site (§8). A
    featurizer used at several sites is left out, because no one site is
    the one its bundle was fitted at. The load
    ([`check_loaded_featurizers`][causalab.protocol.rules.data.check_loaded_featurizers])
    and the build read this one map, so they agree on which featurizers get
    a site check."""
    used: dict[str, list[str]] = {}
    for entry in (*doc.reads.values(), *doc.writes.values()):
        ref = entry.featurizer
        chain = (
            (ref,)
            if isinstance(ref, str)
            else tuple(ref)
            if isinstance(ref, tuple)
            else ()
        )
        for name in chain:
            sites = used.setdefault(name, [])
            if str(entry.site) not in sites:
                sites.append(str(entry.site))
    records: dict[str, dict[str, Any]] = {}
    for name, sites in used.items():
        if len(sites) == 1:
            record = site_identity(doc, sites[0])
            if record is not None:
                records[name] = record
    return records


def spec_identity(record: SiteSpec) -> dict[str, Any]:
    """[`site_identity`][] of a site record itself — for a site the
    executor addresses without the document naming it (the ``ln_final``
    capture of a projecting ``lm_head`` read, ``execution._tap_union``)."""
    # the band as a JSON list — the stamp is serialized, and the loader's
    # expectation (`rules/data._init_expectation`) is spelled the same way
    layers = list(record.layers) if isinstance(record.layers, tuple) else record.layers
    return {
        key: value
        for key, value in {
            "component": record.component,
            "layers": layers,
            "head": record.head,
            "expert": record.expert,
            "stream": record.stream,
        }.items()
        if value is not None
    }
