"""Read documents, overrides, and referenced artifacts.

The compiler uses these readers to resolve inputs and record dependency
metadata. File paths are interpreted relative to the document or the configured
resolution environment. Keep this layer importable with the standard library."""

from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal, Mapping, Sequence, get_args

from causalab.protocol.rules.errors import ParseError, ValidationError
from causalab.protocol.schema import Document, dotted_path, load_raw, tree_path

if TYPE_CHECKING:
    from causalab.io.env import ResolutionEnv

__all__ = [
    "CREATABLE_PATHS",
    "DIAGNOSTIC_KINDS",
    "DataIdentity",
    "Diagnostic",
    "DiagnosticKind",
    "ResolvedArtifact",
    "apply_overrides",
    "check_json_values",
    "identify",
    "load_text",
    "resolve_artifact_fields",
]


# --------------------------------------------------------------------------- #
# the authored file
# --------------------------------------------------------------------------- #


def load_text(path: Path) -> dict[str, Any]:
    """Read one authored file. JSON is the normative surface; ``.yaml`` /
    ``.yml`` parse through a duplicate-key-rejecting SafeLoader into the
    same object model (strict keys hold on both surfaces, §5.1)."""
    text = path.read_text()
    if path.suffix in (".yaml", ".yml"):
        raw = _load_yaml(text)
        if not isinstance(raw, dict):
            raise ParseError("P1", "the top level must be a mapping")
        check_json_values(raw)
        return raw
    return load_raw(text)


def _load_yaml(text: str) -> Any:
    import yaml  # the optional authoring surface — not a load-path dependency

    class _StrictLoader(yaml.SafeLoader):
        pass

    def _mapping(loader: Any, node: Any) -> dict[Any, Any]:
        out: dict[Any, Any] = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node)
            if key in out:
                raise ParseError("P2", f"duplicate key {key!r} in one object")
            out[key] = loader.construct_object(value_node)
        return out

    _StrictLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping
    )
    try:
        return yaml.load(text, Loader=_StrictLoader)  # noqa: S506 — SafeLoader subclass
    except yaml.YAMLError as err:
        raise ParseError("P1", f"not valid YAML: {err}") from err


def check_json_values(raw: Any, *, _path: str = "") -> None:
    """The JSON object model is normative (§0): every mapping key is a
    string and every number is finite — whatever surface (YAML, an artifact
    store) produced the tree."""
    if isinstance(raw, Mapping):
        for key, value in raw.items():
            if not isinstance(key, str):
                raise ParseError(
                    "P2",
                    f"mapping key {key!r} is not a string — quote it at the "
                    "authoring surface",
                    path=_path or None,
                )
            check_json_values(value, _path=f"{_path}.{key}" if _path else str(key))
    elif isinstance(raw, list):
        for item in raw:
            check_json_values(item, _path=_path)
    elif isinstance(raw, float) and (
        raw != raw or raw in (float("inf"), float("-inf"))
    ):
        raise ParseError(
            "P2",
            f"non-finite number at {_path or '<root>'} — the object model is JSON",
            path=_path or None,
        )


# --------------------------------------------------------------------------- #
# --set overrides (§9)
# --------------------------------------------------------------------------- #


_INDEX = re.compile(r"^(.*)\[(\d+)\]$")


#: Dotted paths an override may *create*. An override normally has to hit a
#: field that exists — inventing structure is how a typo becomes an
#: experiment. Revision and dtype fill materialized defaults (§7); the
#: optional attention backend lets a workflow select an implementation even
#: when its referenced campaign leaves the engine default in place.
CREATABLE_PATHS: frozenset[str] = frozenset(
    {"model.dtype", "model.revision", "model.attn_implementation"}
)


def apply_overrides(
    raw: dict[str, Any], overrides: Mapping[str, Any]
) -> dict[str, Any]:
    """Apply ``--set path=value`` overrides (§9): section-rooted dotted paths
    ([`tree_path`][causalab.protocol.schema.types.tree_path] finds the group), ``[i]`` for
    list entries, values as JSON (bare words fall back to strings). The
    path must exist — an override that would *create* structure is a typo,
    not an experiment — except for [`CREATABLE_PATHS`][], the model's
    defaulted fields and its optional attention backend."""
    out = json.loads(json.dumps(raw))  # deep copy, stays plain JSON types
    for dotted, value in overrides.items():
        node: Any = out
        head, _, rest = dotted.partition(".")
        if head == "method":
            raise ParseError(
                "P2",
                f"--set {dotted}: paths start at the section, never the group — "
                + (f"write {rest!r} (§1)" if rest else "name a section (§1)"),
                path=dotted,
            )
        parts = tree_path(dotted)
        for i, part in enumerate(parts):
            last = i == len(parts) - 1
            match = _INDEX.match(part)
            key, index = (
                (match.group(1), int(match.group(2))) if match else (part, None)
            )
            if not isinstance(node, dict) or (
                key not in node and not (last and dotted in CREATABLE_PATHS)
            ):
                raise ParseError(
                    "P2",
                    f"--set {dotted}: {key!r} does not exist in the document",
                    path=dotted,
                )
            if last and index is None:
                node[key] = value
            elif index is not None:
                target = node[key]
                if not isinstance(target, list) or index >= len(target):
                    raise ParseError(
                        "P2",
                        f"--set {dotted}: {key}[{index}] is out of range",
                        path=dotted,
                    )
                if last:
                    target[index] = value
                else:
                    node = target[index]
            else:
                node = node[key]
    return out


# --------------------------------------------------------------------------- #
# artifact-valued fields (§1, §5.15)
# --------------------------------------------------------------------------- #


def resolve_artifact_fields(
    raw: Any,
    env: ResolutionEnv,
    *,
    _path: str = "",
    _seen: frozenset[tuple[str, str]] = frozenset(),
) -> Any:
    """Replace every ``{"artifact": …, "key": …}`` node in a raw tree with
    the value it reads — recursively, so an artifact may itself store a
    reference (a cycle is a load error, not a hang). Runs before the parse
    gate, so a ref is legal anywhere a value is (§1); a mapping that
    *looks* like a ref but is malformed refuses rather than loading as a
    literal dict."""
    if isinstance(raw, Mapping):
        if isinstance(raw.get("artifact"), str):
            if set(raw) != {"artifact", "key"} or not isinstance(raw.get("key"), str):
                raise ValidationError(
                    15,
                    f"malformed artifact reference {dict(raw)!r} — the shape is "
                    '{"artifact": "<ref>", "key": "<field>"} exactly (§1)',
                    path=_path,
                )
            pair = (str(raw["artifact"]), str(raw["key"]))
            if pair in _seen:
                raise ValidationError(
                    15, f"artifact reference cycle through {pair!r}", path=_path
                )
            try:
                value = env.artifacts.read_value(*pair)
            except (FileNotFoundError, KeyError) as err:
                raise ValidationError(
                    15,
                    f"artifact-valued field did not resolve: {err} — a missing "
                    "artifact is a load error, never a default (§1)",
                    path=_path,
                ) from err
            return resolve_artifact_fields(
                value, env, _path=_path, _seen=_seen | {pair}
            )
        return {
            key: resolve_artifact_fields(
                value, env, _path=f"{_path}.{key}" if _path else key, _seen=_seen
            )
            for key, value in raw.items()
        }
    if isinstance(raw, list):
        return [
            resolve_artifact_fields(item, env, _path=_path, _seen=_seen) for item in raw
        ]
    return raw


# --------------------------------------------------------------------------- #
# what a document reached outside itself (the compile's identify stage)
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class DataIdentity:
    """One resolved dataset ref: the content digest stamped into the canonical
    form (§2.2) and the table's columns — the schema ``validate --data``
    checks references against."""

    digest: str
    columns: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class ResolvedArtifact:
    """One reference the document makes outside itself, as the compile resolved
    it.

    A *value* reference (``{"artifact": ref, "key": k}``, §1) names ``key``;
    its value is in the explicit document. A *file* reference (a featurizer's
    or a param's ``file_path``, §2.5/§2.6) names none, and carries the
    stamped ArtifactIdentity the compile read from the file's header (§8) —
    or ``None`` when the store deferred the check, or the file is unstamped.
    ``deferred`` is the store's word: a workflow validating a step-dependent
    document answers with declared representatives and defers every file
    check to run time (workflow spec §2.3), and the compiled result says so
    instead of looking identical to a real resolution."""

    path: str
    reference: str
    key: str | None
    deferred: bool
    identity: Mapping[str, Any] | None


#: What a compile can report without refusing. Closed, and tabulated in spec
#: §9 (``test_compile_protocol.py`` holds the two together):
#:
#: * ``deferred_check`` — the artifact resolver deferred a file's existence
#:   and identity check to run time (workflow validation of a step-dependent
#:   document). The base did this silently; the compiled result now says it;
#: * ``capability_shortfall`` — the document requires a capability an engine
#:   lacks. A compile handed ``engine_capabilities`` *refuses* on it (rule
#:   13, the ``route`` stage), as does `causalab.protocol.pipeline.check_engine` for the routed
#:   engine; the kind is produced by `causalab.protocol.reports.dry_run`,
#:   per candidate engine, from what ``check_engine`` would refuse — without
#:   refusing.
DiagnosticKind = Literal["deferred_check", "capability_shortfall"]
DIAGNOSTIC_KINDS: tuple[DiagnosticKind, ...] = get_args(DiagnosticKind)


@dataclasses.dataclass(frozen=True)
class Diagnostic:
    """One non-refusing finding of a compile. Violations are not diagnostics:
    they are raised, one as itself and several as
    `causalab.protocol.rules.errors.ValidationErrors`, so a returned
    `causalab.protocol.compiled.CompiledProtocol` is always a valid one."""

    kind: DiagnosticKind
    message: str
    path: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in DIAGNOSTIC_KINDS:
            raise AssertionError(
                f"unknown diagnostic kind {self.kind!r}; expected one of "
                f"{DIAGNOSTIC_KINDS}"
            )


def identify(
    authored: Mapping[str, Any],
    documents: Sequence[Document],
    env: ResolutionEnv,
) -> tuple[dict[str, DataIdentity], list[ResolvedArtifact], list[Diagnostic]]:
    """What the document reached outside itself, as resolved: every dataset
    ref's identity and schema, every artifact reference and whether the store
    deferred its check.

    ``authored`` is the tree *before* artifact-valued fields were resolved —
    the value references are found there, since resolution replaced them —
    and ``documents`` are the compiled points' parses, where the data roles
    and the ``file_path`` loads are read. The compiler's ``identify`` stage is
    this function over its build; the three results are the compiled
    protocol's ``data``, ``artifacts`` and ``diagnostics``."""
    data: dict[str, DataIdentity] = {}
    artifacts: list[ResolvedArtifact] = []
    diagnostics: list[Diagnostic] = []
    for doc in documents:
        for role in _data_roles(doc):
            ref = role.dataset
            if isinstance(ref, str) and ref not in data:
                data[ref] = DataIdentity(
                    digest=env.datasets.digest(ref),
                    columns=tuple(env.datasets.columns(ref)),
                )
    defers: Callable[[str], bool] | None = getattr(env.artifacts, "defers", None)
    seen: set[tuple[str, str]] = set()
    for path, reference, key in _value_references(authored):
        if (path, reference) in seen:
            continue
        seen.add((path, reference))
        artifacts.append(
            ResolvedArtifact(
                path=path,
                reference=reference,
                key=key,
                deferred=defers is not None and defers(reference),
                identity=None,
            )
        )
    for doc in documents:
        for path, file_path in _file_references(doc):
            if (path, file_path) in seen:
                continue
            seen.add((path, file_path))
            deferred = defers is not None and defers(file_path)
            artifacts.append(
                ResolvedArtifact(
                    path=path,
                    reference=file_path,
                    key=None,
                    deferred=deferred,
                    identity=None
                    if deferred
                    else env.artifacts.read_identity(file_path),
                )
            )
            if deferred:
                diagnostics.append(
                    Diagnostic(
                        "deferred_check",
                        f"{file_path!r} is loaded from a run tree: its existence "
                        "and identity are checked at run time, not here",
                        path=path,
                    )
                )
    return data, artifacts, diagnostics


def _data_roles(doc: Document) -> list[Any]:
    """Every data role of a point, base first (§2.2)."""
    out: list[Any] = []
    for value in doc.data.values():
        out.extend(value if isinstance(value, tuple) else (value,))
    return out


def _value_references(node: Any, *, _path: str = "") -> list[tuple[str, str, str]]:
    """``(path, artifact, key)`` for every ``{"artifact": …, "key": …}`` node
    of the authored document — the references [`resolve_artifact_fields`][]
    replaced by their values."""
    found: list[tuple[str, str, str]] = []
    if isinstance(node, Mapping):
        artifact = node.get("artifact")
        if isinstance(artifact, str) and isinstance(node.get("key"), str):
            found.append((dotted_path(_path.split(".")), artifact, str(node["key"])))
            return found
        for key, value in node.items():
            found.extend(
                _value_references(value, _path=f"{_path}.{key}" if _path else key)
            )
    elif isinstance(node, list):
        for index, item in enumerate(node):
            found.extend(_value_references(item, _path=f"{_path}[{index}]"))
    return found


def _file_references(doc: Document) -> list[tuple[str, str]]:
    """``(path, file_path)`` for the two places a document *loads* a file:
    featurizer bundles and params (§2.5, §2.6). ``save`` also carries
    ``file_path`` keys, but those are outputs, never loads."""
    found: list[tuple[str, str]] = []
    for name, spec in doc.featurizers.items():
        if isinstance(spec.file_path, str):
            found.append((f"featurizers.{name}.file_path", spec.file_path))
    for name, pspec in doc.params.items():
        if isinstance(pspec.file_path, str):
            found.append((f"params.{name}.file_path", pspec.file_path))
    return found
