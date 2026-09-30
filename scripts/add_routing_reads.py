"""Add routing capture to a document.

Expert ids and router weights are declared reads, never implicit saves. For
every MoE layer of the document's model and every ``(model, input)`` pair the
document executes — every declared model on its own input, in declaration
order — this adds one ``expert_idx`` read and one
``router_scores`` read at each position, and the save entry each needs::

    sites:  routing_L{layer}_idx     {"component": "expert_idx",    "layers": [layer]}
            routing_L{layer}_scores  {"component": "router_scores", "layers": [layer]}
    reads:  routing_L{layer}_{model}_{position}_{idx|scores}
    save:   routing_L{layer}_{model}_{position}_{idx|scores}.safetensors

The positions default to the ones the document's own reads on that pair
already use, deduplicated; ``--positions`` names them instead (a ``positions``
entry, the integer sugar, or ``all``) and applies to every pair. An input
role's brackets are dropped in a name (``counterfactual[0]`` spells
``counterfactual0``). Existing entries are left untouched, and a generated
name that already exists refuses — so applying the script twice is an error,
never a silent duplicate.

Usage::

    uv run python scripts/add_routing_reads.py patch.json --out patch_routed.json
    uv run python scripts/add_routing_reads.py patch.json --out patch_routed.json \\
        --positions answer_tok -1

Refused: a model whose registry entry declares no experts (there is no routing
to capture), a read whose inline position is a mapping (name it in
``positions`` so the entry has a name), and any name collision.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

from protocol_authoring import (  # noqa: E402 — the sibling module, found above
    AuthoringError,
    add_tap,
    dump_document,
    entry_name,
    executed_pairs,
    load_document,
    method_of,
    model_info,
    moe_layers,
    position_label,
    read_positions,
    routes_experts,
)

from causalab.cli import register_model_key  # noqa: E402
from causalab.protocol.rules.errors import ProtocolError  # noqa: E402
from causalab.protocol.schema import ALL_POSITIONS  # noqa: E402

#: The two routing reads per (layer, pair, position) and the suffix each
#: contributes to its name.
ROUTING_COMPONENTS: tuple[tuple[str, str], ...] = (
    ("expert_idx", "idx"),
    ("router_scores", "scores"),
)


def parse_positions(values: Sequence[str], method: Mapping[str, Any]) -> list[Any]:
    """``--positions`` into read ``pos`` values: an integer is the index
    sugar, ``all`` the all-positions sugar, anything else must be a declared
    ``positions`` entry of the method."""
    out: list[Any] = []
    declared = method.get("positions", {})
    for value in values:
        if value == ALL_POSITIONS:
            out.append(value)
            continue
        try:
            out.append(int(value))
            continue
        except ValueError:
            pass
        if value not in declared:
            raise AuthoringError(
                f"--positions names {value!r}, which is neither an integer, "
                f"'all', nor one of the declared positions {list(declared)}"
            )
        out.append(value)
    return out


def add_routing_reads(
    doc: dict[str, Any], *, positions: Sequence[Any] | None = None
) -> dict[str, Any]:
    """The document with routing reads added, as the module docstring
    describes. ``positions`` (already parsed) applies to every executed pair;
    ``None`` takes each pair's own read positions."""
    info = model_info(doc)
    if not routes_experts(info):
        raise AuthoringError(
            f"model {info.key!r} declares no experts — there is no routing to "
            "capture on a dense model"
        )
    template = method_of(doc)
    pairs = executed_pairs(template)
    if not pairs:
        raise AuthoringError("the document executes nothing: it declares no reads")
    per_pair: dict[tuple[str, str], list[Any]] = {}
    for pair in pairs:
        chosen = (
            list(positions)
            if positions is not None
            else read_positions(template, *pair)
        )
        if not chosen:
            raise AuthoringError(
                f"{pair[0]} on {pair[1]} has no reads to take positions from — "
                "pass --positions"
            )
        per_pair[pair] = chosen
    out = dict(doc)
    method = dict(template)
    out["method"] = method
    for section in ("sites", "reads"):
        method[section] = dict(template.get(section, {}))
    # the models gain reads: copy each entry so the template is untouched
    method["intervened_models"] = {
        name: {**entry, "reads": list(entry.get("reads", []))}
        if isinstance(entry, dict)
        else entry
        for name, entry in template.get("intervened_models", {}).items()
    }
    method["save"] = list(template.get("save", []))
    for layer in moe_layers(info):
        for model, input_role in pairs:
            for pos in per_pair[(model, input_role)]:
                label = position_label(pos)
                for component, suffix in ROUTING_COMPONENTS:
                    add_tap(
                        method,
                        site=f"routing_L{layer}_{suffix}",
                        component=component,
                        layer=layer,
                        # a model runs on one input (§2.9), so the model alone
                        # names the forward the read is taken on
                        read=entry_name(f"routing_L{layer}", model, label, suffix),
                        pos=pos,
                        model=model,
                        input=input_role,
                    )
    return out


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="add_routing_reads",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("document", type=Path, help="the document to add reads to")
    parser.add_argument("--out", required=True, type=Path, help="document to write")
    parser.add_argument(
        "--positions",
        nargs="+",
        default=None,
        metavar="POS",
        help="positions for every pair: a `positions` name, an integer, or "
        "'all' (default: the positions the pair's own reads use)",
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
        doc = load_document(args.document)
        if args.register_from_hf:
            register_model_key(doc)
        positions = (
            parse_positions(args.positions, method_of(doc))
            if args.positions is not None
            else None
        )
        out = add_routing_reads(doc, positions=positions)
    except (AuthoringError, ProtocolError) as err:
        raise SystemExit(f"refused: {err}") from err
    dump_document(out, args.out)
    reads = out["method"]["reads"]
    added = len(reads) - len(method_of(doc).get("reads", {}))
    print(f"wrote {args.out} ({added} routing reads added over {len(reads)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
