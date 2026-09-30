"""Regenerate ``fixtures/resolved_sites_census.json`` — the resolver census.

For each of the three tiny fixtures (``tiny-random-gpt2``,
``tiny-random-LlamaForCausalLM``, ``tiny-random/qwen3.5-moe``), every
``(component, layer, head)`` the census walks is handed to
[`causalab.neural.shared.sites.resolve_site`][] and the answer is recorded:
the [`ResolvedSite`][causalab.neural.shared.sites.ResolvedSite] (module path, io side,
feature slice, interface slot, declared shape, derivation) where it resolves,
the refusal (class, code, reason, message) where it refuses. The site is
built the way the parser builds one — a retired spelling folds onto its
replacement first (``schema.DEPRECATED_COMPONENTS``) — so the census records
what a *document* naming the component gets.

The census is the proof of a refactor whose acceptance is that **nothing
changes** (moving ``resolve_site`` onto rows — snapshot before, diff after).
It was captured on the resolver before that move (its ``base`` field says so)
and is compared, not regenerated, by ``test_resolved_sites_census.py``; the
comparison names the one set of entries a deliberate decision moved (the
eight ``identical`` DeltaNet pairs that became aliases). Regenerate only to
extend the census — a fourth fixture, another layer — and say so in the PR::

    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run python \\
        tests/neural/engines/pytorch_hooks/update_resolved_sites_census.py [--check]

``--check`` re-captures and exits non-zero if the committed file differs.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any

from causalab.neural.engines.pytorch_hooks.loading import load_model
from causalab.neural.shared.sites import ResolvedSite, resolve_site
from causalab.protocol.rules.errors import ProtocolError, ValidationError
from causalab.protocol.registry import component_shape
from causalab.protocol.schema import (
    COMPONENTS,
    DEPRECATED_COMPONENTS,
    LAYERLESS_COMPONENTS,
    SiteSpec,
)

CENSUS = Path(__file__).parent / "fixtures" / "resolved_sites_census.json"

#: The fixtures and the layers the census walks on each. The two dense
#: families are uniform across depth, so their first and last layer suffice;
#: the hybrid fixture alternates mixers, so every layer is walked.
FIXTURES: dict[str, tuple[int, ...]] = {
    "hf-internal-testing/tiny-random-gpt2": (0, 4),
    "hf-internal-testing/tiny-random-LlamaForCausalLM": (0, 1),
    "tiny-random/qwen3.5-moe": (0, 1, 2, 3),
}


def _module_path(bundle: Any, module: Any) -> str:
    """The qualified name of ``module`` inside the loaded model."""
    for name, candidate in bundle.model.named_modules():
        if candidate is module:
            return name or "<model>"
    raise AssertionError(f"module {type(module).__name__} is not in the tree")


def _record_site(bundle: Any, site: ResolvedSite) -> dict[str, Any]:
    feature_slice = site.feature_slice
    return {
        "module": _module_path(bundle, site.module),
        "kind": site.kind,
        "shape": dataclasses.asdict(site.shape),
        "feature_slice": (
            None
            if feature_slice is None
            else [feature_slice.start, feature_slice.stop, feature_slice.step]
        ),
        "layer": site.layer,
        "component": site.component,
        "tuple_index": site.tuple_index,
        "interface_slot": site.interface_slot,
        "head": site.head,
        "expert": site.expert,
        "derivation": site.derivation,
    }


def _record_refusal(exc: BaseException) -> dict[str, Any]:
    protocol = isinstance(exc, (ProtocolError, ValidationError))
    return {
        "refusal": {
            "exc_class": type(exc).__name__,
            "code": exc.code if protocol else None,
            "reason": getattr(exc, "reason", None) if protocol else None,
            "message": str(exc),
        }
    }


def _has_head_space(bundle: Any, component: str) -> bool:
    try:
        return component_shape(bundle.info, component).head_space is not None
    except ValidationError:
        return False


def _heads(bundle: Any, component: str) -> tuple[int | None, ...]:
    """``None`` always; ``0`` too where the component has a head axis (the
    slice) or names one the resolver bound-checks itself (the state)."""
    return (None, 0) if _has_head_space(bundle, component) else (None,)


def _site(component: str, layer: int | None, head: int | None) -> SiteSpec:
    # the parser's fold: a retired spelling names its replacement
    return SiteSpec(
        component=DEPRECATED_COMPONENTS.get(component, component),
        layers=(layer,) if layer is not None else None,
        head=head,
    )


def census() -> dict[str, Any]:
    out: dict[str, Any] = {"base": "the resolver before rows", "fixtures": {}}
    for key, layers in FIXTURES.items():
        bundle = load_model(key)
        entries: list[dict[str, Any]] = []
        names = tuple(COMPONENTS) + tuple(
            alias for alias in DEPRECATED_COMPONENTS if alias not in COMPONENTS
        )
        for component in names:
            per_layer: tuple[int | None, ...] = (
                (None,) if component in LAYERLESS_COMPONENTS else layers
            )
            for layer in per_layer:
                for head in _heads(bundle, component):
                    entry: dict[str, Any] = {
                        "component": component,
                        "layer": layer,
                        "head": head,
                    }
                    try:
                        site = resolve_site(bundle, _site(component, layer, head))
                    except Exception as exc:  # noqa: BLE001 - recorded, not raised
                        entry.update(_record_refusal(exc))
                    else:
                        entry["resolved"] = _record_site(bundle, site)
                    entries.append(entry)
        out["fixtures"][key] = {
            "family": bundle.info.family,
            "streams": list(bundle.streams),
            "entries": entries,
        }
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true", help="verify, do not write")
    args = parser.parse_args(argv)
    text = json.dumps(census(), indent=1, ensure_ascii=False) + "\n"
    if args.check:
        current = CENSUS.read_text() if CENSUS.exists() else ""
        if current != text:
            print(f"{CENSUS} is stale", file=sys.stderr)
            return 1
        print(f"{CENSUS} is current")
        return 0
    CENSUS.parent.mkdir(parents=True, exist_ok=True)
    CENSUS.write_text(text)
    total = sum(len(f["entries"]) for f in json.loads(text)["fixtures"].values())
    print(f"wrote {CENSUS} ({total} entries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
