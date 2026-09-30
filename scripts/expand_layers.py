"""Expand a one-layer harvest template over every layer.

The template is a pure-read document with one site, its reads and their save
entries: the shape of ``tests/protocols/01_harvest_im.json`` at one layer, or
of ``demos/methods/protocols/mean_harvest.json``. The output declares, for
every layer in the range and for each of the two residual sites, ``block_mid``
(the residual after the mixer, letter ``A``) and ``block_output`` (the
residual after the MLP, letter ``M``), one site and — at every kept
position — one read and one save entry::

    sites:  L{layer}A  {"component": "block_mid",    "layers": [layer]}
            L{layer}M  {"component": "block_output", "layers": [layer]}
    reads:  acts_L{layer}{A|M}_{position}
    save:   acts_L{layer}{A|M}_{position}.safetensors

A position's name is its ``positions``-table name, or the integer sugar's
``i<n>`` / ``im<n>`` spelling (``-1`` is ``im1``). A save-time ``reduce`` and
a read's ``dims`` carry over from the template read at that position, so a
``reduce: mean`` template yields the per-layer means. Every other section of
the template's ``method`` is preserved, and its header, model and data
groups (spec §1) are carried over untouched.

Usage::

    uv run python scripts/expand_layers.py harvest_template.json \\
        --out harvest.json                       # every layer, every position
    uv run python scripts/expand_layers.py harvest_template.json \\
        --out harvest_early.json --layers 0:8 --positions answer_tok

Refused: a template with ``writes`` or ``intervened_models`` (a harvest is a
pure read), a read through a featurizer (the harvest is the raw residual),
more than one site, a site with a sub-axis, or reads on more than one input
role (run once per role — the names carry no role).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

from protocol_authoring import (  # noqa: E402 — the sibling module, found above
    AuthoringError,
    add_tap,
    dump_document,
    load_document,
    method_of,
    model_info,
    parse_layer_range,
    position_label,
)

from causalab.cli import register_model_key  # noqa: E402
from causalab.protocol.rules.errors import ProtocolError  # noqa: E402

#: The two residual sites per layer and the letter each contributes to a
#: generated name: ``A`` is the residual after the attention or DeltaNet
#: mixer, ``M`` the residual after the MLP.
RESIDUAL_SITES: tuple[tuple[str, str], ...] = (
    ("block_mid", "A"),
    ("block_output", "M"),
)


def _template_reads(method: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """The template method's reads keyed by position label, each carrying the
    ``pos``, the ``dims`` and the save-time ``reduce`` the expansion repeats
    at every layer — after the refusals the docstring lists."""
    if "writes" in method:
        raise AuthoringError(
            "the template declares 'writes' — a harvest is a pure read, so the "
            "template carries one un-intervened model, reads and save entries only"
        )
    sites = method.get("sites", {})
    if len(sites) != 1:
        raise AuthoringError(
            f"the template declares {len(sites)} sites; expand_layers expands "
            "exactly one — the site whose reads it repeats at every layer"
        )
    (site_name, site), *_ = sites.items()
    if set(site) != {"component", "layers"}:
        raise AuthoringError(
            f"site {site_name!r} must be {{component, layers}} alone — the "
            "expansion replaces the component by the two residual sites at "
            f"each layer, and {sorted(set(site) - {'component', 'layers'})} "
            "would not survive that"
        )
    saves = {
        entry["read"]: entry
        for entry in method.get("save", [])
        if isinstance(entry, dict) and "read" in entry
    }
    models = method.get("intervened_models", {})
    if len(models) != 1:
        raise AuthoringError(
            f"the template declares {len(models)} models — a harvest reads one "
            "un-intervened model on one input role"
        )
    ((model_name, model_entry),) = models.items()
    if model_entry.get("writes"):
        raise AuthoringError(
            f"model {model_name!r} lands writes — a harvest reads the un-intervened "
            "model only"
        )
    taken = set(model_entry.get("reads", []))
    role = str(model_entry.get("input"))
    by_label: dict[str, dict[str, Any]] = {}
    for name, read in method.get("reads", {}).items():
        if "featurizer" in read:
            raise AuthoringError(
                f"read {name!r} goes through a featurizer — a harvest reads the "
                "raw residual, and a featurizer at one layer has no meaning at "
                "another"
            )
        if name not in taken:
            raise AuthoringError(
                f"read {name!r} is taken on no model — the template's one model "
                f"{model_name!r} lists {sorted(taken)}"
            )
        label = position_label(read.get("pos"))
        if label in by_label:
            raise AuthoringError(
                f"reads {by_label[label]['read']!r} and {name!r} are both at "
                f"position {label!r} — one read per position"
            )
        by_label[label] = {
            "read": name,
            "pos": read["pos"],
            "model": model_name,
            "input": role,
            "dims": read.get("dims"),
            "reduce": saves.get(name, {}).get("reduce"),
        }
    if not by_label:
        raise AuthoringError("the template declares no reads")
    return by_label


def expand(
    doc: dict[str, Any], *, layers: str | None, positions: Sequence[str]
) -> dict[str, Any]:
    """The expanded document: the template's groups, with the method's
    ``positions``, ``sites``, ``reads`` and ``save`` rebuilt over ``layers`` ×
    the two residual sites × the kept positions."""
    info = model_info(doc)
    layer_range = parse_layer_range(layers, info)
    template = method_of(doc)
    reads = _template_reads(template)
    kept = list(positions) or list(reads)
    unknown = [label for label in kept if label not in reads]
    if unknown:
        raise AuthoringError(
            f"--positions names {unknown}, which the template's reads do not "
            f"use; the template positions are {list(reads)}"
        )
    out = {key: value for key, value in doc.items() if key != "method"}
    method: dict[str, Any] = {
        key: value
        for key, value in template.items()
        if key not in ("intervened_models", "positions", "sites", "reads", "save")
    }
    out["method"] = method
    first = next(iter(reads.values()))
    method["intervened_models"] = {
        first["model"]: {"input": first["input"], "reads": []}
    }
    declared = template.get("positions", {})
    kept_named = {label: declared[label] for label in kept if label in declared}
    if kept_named:
        method["positions"] = kept_named
    method["sites"], method["reads"], method["save"] = {}, {}, []
    for layer in layer_range:
        for component, letter in RESIDUAL_SITES:
            site = f"L{layer}{letter}"
            for label in kept:
                spec = reads[label]
                add_tap(
                    method,
                    site=site,
                    component=component,
                    layer=layer,
                    read=f"acts_L{layer}{letter}_{label}",
                    pos=spec["pos"],
                    model=spec["model"],
                    input=spec["input"],
                    reduce=spec["reduce"],
                    dims=spec["dims"],
                )
    return out


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="expand_layers",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("template", type=Path, help="the one-layer harvest document")
    parser.add_argument("--out", required=True, type=Path, help="document to write")
    parser.add_argument(
        "--layers",
        default=None,
        metavar="START:STOP",
        help="half-open layer range (default: every layer of the model)",
    )
    parser.add_argument(
        "--positions",
        nargs="+",
        default=[],
        metavar="NAME",
        help="positions to keep, by the name their template read uses "
        "(default: every position the template reads at)",
    )
    parser.add_argument(
        "--register-from-hf",
        action="store_true",
        help="resolve an unregistered model key from its HF config before "
        "sizing the output (the CLI's opt-in; the only thing here that can "
        "touch the network)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        doc = load_document(args.template)
        if args.register_from_hf:
            register_model_key(doc)
        out = expand(doc, layers=args.layers, positions=args.positions)
    except (AuthoringError, ProtocolError) as err:
        raise SystemExit(f"refused: {err}") from err
    dump_document(out, args.out)
    method = out["method"]
    print(
        f"wrote {args.out} ({len(method['sites'])} sites, {len(method['reads'])} "
        f"reads, {len(method['save'])} save entries)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
