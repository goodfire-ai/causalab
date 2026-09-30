"""Freeze each arm's source census while holding benchmark inputs fixed."""

from __future__ import annotations

from collections.abc import Mapping, Set
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Literal

Pins = Mapping[str, Mapping[str, str]]
FrozenPins = tuple[tuple[str, tuple[tuple[str, str], ...]], ...]
DEFERRED_DIGEST = "0" * 64


@dataclass(eq=False)
class DeferredFilePinError(ValueError):
    """A loader placeholder has no concrete external-file attestation."""

    key: str
    reason: str

    def __str__(self) -> str:
        return f"pins.files.{self.key}: {self.reason}"


@dataclass(eq=False)
class SourceOwnershipError(ValueError):
    """A commit-varying source pin is not backed by the selected installation."""

    arm: str
    category: str
    key: str
    reason: str

    def __str__(self) -> str:
        return f"arm {self.arm!r}, pins.{self.category}.{self.key}: {self.reason}"


def _owned(key: str) -> bool:
    parts = key.split(".")
    return parts[0] == "causalab" and all(part.isidentifier() for part in parts)


def _partition(
    pins: Pins,
) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    source: dict[str, dict[str, str]] = {}
    shared: dict[str, dict[str, str]] = {}
    for category, entries in pins.items():
        for key, value in entries.items():
            target = (
                source if category in {"code", "scripts"} and _owned(key) else shared
            )
            target.setdefault(category, {})[key] = value
    return source, shared


def _freeze(pins: Pins) -> FrozenPins:
    return tuple(
        (category, tuple(sorted(entries.items())))
        for category, entries in sorted(pins.items())
        if entries
    )


def _thaw(pins: FrozenPins) -> dict[str, dict[str, str]]:
    return {category: dict(entries) for category, entries in pins}


def _resolved_files(
    pins: Pins, file_digests: Mapping[str, str] | None
) -> dict[str, dict[str, str]]:
    from ..census import parse_pins

    result = parse_pins(pins, "pins")
    supplied = parse_pins({"files": dict(file_digests or {})}, "pins").get("files", {})
    for key, value in result.get("files", {}).items():
        if value == DEFERRED_DIGEST:
            concrete = supplied.get(key)
            if concrete is None or concrete == DEFERRED_DIGEST:
                raise DeferredFilePinError(
                    key, "deferred external input requires its actual file digest"
                )
            result["files"][key] = concrete
        elif key in supplied and supplied[key] != value:
            raise DeferredFilePinError(
                key, "file digest changed after workflow loading"
            )
    return result


@dataclass(frozen=True)
class PinContract:
    """Source varies only for direct modules in the selected Causalab package.

    External modules, local scripts and closure members remain shared alongside
    documents, datasets and input files. Returned dictionaries are independent
    copies; callers cannot mutate the frozen contract through a receipt.
    """

    _source: FrozenPins
    _shared: FrozenPins

    @classmethod
    def resolve(
        cls,
        authored: Pins | None,
        actual: Pins,
        *,
        arm: str,
        package_root: Path,
        source_pin_anchor: str = "before",
        file_digests: Mapping[str, str] | None = None,
        comparison: Literal["code", "workflow"] = "code",
    ) -> PinContract:
        # These imports deliberately resolve against the selected arm, including
        # when this controller module is loaded under the private bootstrap.
        from causalab.protocol.identity import CodeResolutionError, resolve_locator
        from ..census import check_pins, parse_pins

        loader_pins = actual
        actual = _resolved_files(actual, file_digests)
        source, shared = _partition(actual)
        package = (package_root / "causalab").resolve()
        for category, entries in source.items():
            for key, expected in entries.items():
                try:
                    resolved = resolve_locator(key)
                except CodeResolutionError as error:
                    raise SourceOwnershipError(
                        arm,
                        category,
                        key,
                        f"source cannot be resolved in the selected installation: {error}",
                    ) from error
                path = resolved.path
                if (
                    resolved.module != key
                    or resolved.attr
                    or path.is_symlink()
                    or not path.is_file()
                    or not path.resolve().is_relative_to(package)
                ):
                    raise SourceOwnershipError(
                        arm,
                        category,
                        key,
                        "source is not a regular module in the selected installation",
                    )
                if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                    raise SourceOwnershipError(
                        arm,
                        category,
                        key,
                        "resolved pin does not match selected source bytes",
                    )
        if authored is not None:
            authored = parse_pins(authored, "pins")
            # Accept stamped placeholders only for loader-deferred files;
            # concrete authored hashes must match the resolved bytes.
            for key, value in authored.get("files", {}).items():
                if (
                    value == DEFERRED_DIGEST
                    and loader_pins.get("files", {}).get(key) == DEFERRED_DIGEST
                ):
                    authored["files"][key] = actual["files"][key]
            if comparison == "workflow" or arm == source_pin_anchor:
                check_pins(authored, actual)
            else:
                _, authored_shared = _partition(authored)
                check_pins(authored_shared, shared)
        return cls(_freeze(source), _freeze(shared))

    @property
    def source(self) -> dict[str, dict[str, str]]:
        return _thaw(self._source)

    @property
    def shared(self) -> dict[str, dict[str, str]]:
        return _thaw(self._shared)

    @property
    def pins(self) -> dict[str, dict[str, str]]:
        result = self.shared
        for category, entries in self.source.items():
            result.setdefault(category, {}).update(entries)
        return result

    def check(
        self, actual: Pins, *, file_digests: Mapping[str, str] | None = None
    ) -> None:
        """A later full load must reproduce the complete frozen census."""
        from ..census import check_pins

        check_pins(self.pins, _resolved_files(actual, file_digests))

    def check_subset(
        self,
        actual: Pins,
        *,
        produced_files: Set[str] = frozenset(),
        file_digests: Mapping[str, str] | None = None,
    ) -> None:
        """Hold selected steps; allow explicitly attested run-tree products.

        The caller identifies generated files using the original workflow and
        its contained artifact overlay. Existing external pins are never exempt.
        """
        from ..census import check_pins

        frozen = self.pins
        expected: dict[str, dict[str, str]] = {}
        observed: dict[str, dict[str, str]] = {}
        for category, entries in _resolved_files(actual, file_digests).items():
            known = frozen.get(category, {})
            for key, value in entries.items():
                if category == "files" and key in produced_files and key not in known:
                    continue
                observed.setdefault(category, {})[key] = value
                if key in known:
                    expected.setdefault(category, {})[key] = known[key]
        check_pins(expected, observed)
