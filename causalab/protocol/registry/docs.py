"""The docs tables rendered from the registry: never hand-written twice.

``docs/qwen36_35b_a3b.md``'s component table, spec §8's component
row, the per-family tap table and the gate-group table are all drawn from
the capability rows and the shapes; the census guards in
``tests/protocol/test_vocabulary_census.py`` hold the committed text to
what these render.
"""

from __future__ import annotations

from causalab.protocol.rules.errors import ValidationError
from causalab.protocol.registry import shapes
from causalab.protocol.registry.components import (
    CAPABILITIES,
    ENGINES,
    INTERIOR_ROWS,
    Capability,
    ModelInfo,
    _ANY_MECHANISM,  # pyright: ignore[reportPrivateUsage]
    _BOTH,  # pyright: ignore[reportPrivateUsage]
    _FUSED_PACKINGS,  # pyright: ignore[reportPrivateUsage]
    component_shape,
    component_width,
    families_in_table,
    native_shape,
    site_group_map,
)
from causalab.protocol.registry.engines import DOCS_TABLE_MODEL, components_served_by
from causalab.protocol.registry.models import get_model_info
from causalab.protocol.schema import (
    COMPONENTS,
    GATE_GROUP_AXES,
    GATE_GROUPS,
    LAYERLESS_COMPONENTS,
)


# --------------------------------------------------------------------------- #
# Tables rendered from the capability registry.
# --------------------------------------------------------------------------- #


_TABLE_HEADER = (
    "| component | blocks | shape | tap | engines | write |\n|---|---|---|---|---|---|"
)

#: The five groups the table draws, in order, with their headings.
_TABLE_GROUPS: tuple[tuple[str, str], ...] = (
    ("boundary", "**Model boundary (no `layer`)**"),
    ("residual", "**Residual stream and dense MLP: every layer**"),
    (
        "full_attention",
        "**Full-attention mixer interior: the {full} `full_attention` layers**",
    ),
    (
        "linear_attention",
        "**Gated DeltaNet mixer interior: the {linear} `linear_attention` layers**",
    ),
    ("moe", "**Sparse MoE + shared expert: every layer**"),
)


def _table_group(row: Capability) -> str:
    if row.component in LAYERLESS_COMPONENTS:
        return "boundary"
    if row.stream is not None:
        return row.stream
    if "moe" in row.requires:
        return "moe"
    return "residual"


def _write_cell(row: Capability) -> str:
    if row.writes is None:
        return f"read-only: {row.why}"
    if row.writes == _ANY_MECHANISM:
        return "any mechanism"
    allowed = " ".join(f"`{m}`" for m in sorted(row.writes))
    return f"{allowed} only: {row.why}"


def _engines_cell(row: Capability) -> str:
    if row.reads == _BOTH:
        return "both"
    return " ".join(f"`{e}`" for e in ENGINES if e in row.reads)


def render_component_tables(info: ModelInfo | None = None) -> str:
    "Render component shapes, engine support, and write policies for the selected model. Defaults to DOCS_TABLE_MODEL."
    info = get_model_info(DOCS_TABLE_MODEL) if info is None else info
    assert info.layer_types is not None  # the docs model declares its pattern
    n_full = info.layer_types.count("full_attention")
    n_linear = info.layer_types.count("linear_attention")
    blocks = {
        "boundary": "layer-less",
        "residual": f"every layer ({info.num_layers})",
        "full_attention": f"full-attn ({n_full})",
        "linear_attention": f"DeltaNet ({n_linear})",
        "moe": f"every layer ({info.num_layers})",
    }
    sections: list[str] = []
    for group, heading in _TABLE_GROUPS:
        lines = [heading.format(full=n_full, linear=n_linear), "", _TABLE_HEADER]
        for component in COMPONENTS:
            row = CAPABILITIES[component]
            if _table_group(row) != group:
                continue
            try:
                shape = f"`{component_shape(info, component).describe()}`"
                where = blocks[group]
            except ValidationError:
                # no tensor on this architecture: the row exists, the box does
                # not (``Qwen/Qwen3.6-35B-A3B``'s MLP is a sparse-MoE block
                # at every layer)
                shape = "n/a"
                where = "unavailable on this architecture"
            lines.append(
                f"| `{component}` | {where} | {shape} | {row.tap} | "
                f"{_engines_cell(row)} | {_write_cell(row)} |"
            )
        sections.append("\n".join(lines))
    return "\n\n".join(sections) + "\n"


def engine_component_summary(engine: str) -> str:
    """The ``N of M`` cell of spec §8's generated component row."""
    return f"{len(components_served_by(engine))} of {len(COMPONENTS)}"


def _address_cell(component: str, family: str) -> str:
    address = CAPABILITIES[component].address_on(
        # any widths: the rendered shape carries axis names, not numbers
        ModelInfo(
            key="render",
            hidden_size=1,
            num_layers=1,
            num_heads=1,
            num_kv_heads=1,
            head_dim=1,
            intermediate_size=None,
            vocab_size=1,
            family=family,
        )
    )
    if address is None:
        return "unavailable on this family"
    shape = native_shape(address, shapes.bs_flat_heads(1, 1)).describe()
    cell = f"`{address['module']}` output `{shape}`"
    if address["packing"] in _FUSED_PACKINGS:
        cell += f", split {address['split']} of {address['splits']}"
    return cell


def render_family_table() -> str:
    "Render module addresses for attention components across registered model families."
    families = families_in_table()
    header = "| component | " + " | ".join(f"`{f}`" for f in families) + " |"
    rule = "|---|" + "---|" * len(families)
    lines = [header, rule]
    for component in INTERIOR_ROWS:
        cells = " | ".join(_address_cell(component, f) for f in families)
        lines.append(f"| `{component}` | {cells} |")
    return "\n".join(lines) + "\n"


def render_widthless_components(info: ModelInfo | None = None) -> str:
    "List components without a feature width on the selected model."
    info = get_model_info(DOCS_TABLE_MODEL) if info is None else info
    without: list[str] = []
    for component in COMPONENTS:
        try:
            component_width(info, component)
        except (ValidationError, ValueError):
            without.append(f"`{component}`")
    return (
        f"On `{info.key}` a featurizer attaches to "
        f"{len(COMPONENTS) - len(without)} of {len(COMPONENTS)} components. "
        "These components lack a feature width: " + ", ".join(without) + ".\n"
    )


def render_gate_group_table(info: ModelInfo | None = None) -> str:
    "Render each gate group, its parameter axis, and supported components from site_group_map."
    info = get_model_info(DOCS_TABLE_MODEL) if info is None else info
    lines = ["| group | axis | components it is legal on |", "|---|---|---|"]
    for group in GATE_GROUPS:
        legal: list[str] = []
        for component in COMPONENTS:
            try:
                site_group_map(info, group, component)
            except (ValidationError, ValueError):
                continue
            legal.append(f"`{component}`")
        lines.append(f"| `{group}` | `{GATE_GROUP_AXES[group]}` | {', '.join(legal)} |")
    return "\n".join(lines) + "\n"
