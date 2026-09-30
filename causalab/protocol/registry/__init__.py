"""The protocol registry: static model metadata, per-component shapes and
capability rows, the per-engine view, the family plugin contract, and the
rendered docs tables — as data, torch-free and offline.

Deriving featurizer widths and param shapes (spec §6) needs a model's *static
configuration* (hidden size, depth, head counts), never its weights, and
deciding what a document may address needs one row of truth per component.
This package keeps both deterministic and offline: entries for the models the
repo actually uses are declared here as data, tests register their
tiny-random models, and an HF config can be adapted explicitly with
[`model_info_from_hf_config`][] when a caller opts in — the protocol layer
itself never touches the network.

The package is split by what each module owns; ``causalab.protocol.registry``
re-exports the whole namespace, so ``from causalab.protocol.registry import X``
is the canonical spelling for every ``X`` below:

``shapes``
    [`FeatureShape`][] — a tapped tensor's native axes and which one is
    features, as data rather than a layout string.
``models``
    [`ModelInfo`][], the model registry ([`register_model`][],
    [`get_model_info`][], [`model_info_from_hf_config`][]) and the
    built-in entries for the models the repo's configs and corpus name.
``components``
    per-component truth: [`component_shape`][] (the ``(model, site) → d``
    rule of §2.5 and its axes), the grouped-gate maps ([`site_group_map`][],
    [`gate_param_shape`][]), and **the capability registry** —
    [`CAPABILITIES`][], one row per component (which engines serve it, its
    write policy and refusal text, its mixer stream, the predicates it
    requires, its per-family address) with the offline refusals derived from
    the rows.
``engines``
    the per-engine view of the rows ([`components_served_by`][],
    [`write_capabilities`][], [`declared_capabilities`][],
    [`ENGINE_VERBS`][]), the typed backend pairs ([`BACKEND_PAIRS`][],
    [`alias_would_rebind`][]) and the alias check the vocabulary is held to
    at import.
``families``
    **the family plugin contract** — [`FamilyAdapter`][], [`Tap`][],
    [`Identity`][], [`register_family`][] / [`family_for`][] with the
    two built-in trees, the function-slot tables, and the [`inventory`][]
    of what exists at which layer.
``docs``
    the rendered docs tables ([`render_component_tables`][],
    [`render_family_table`][], [`render_gate_group_table`][], …).
"""

# pyright: reportUnusedImport=false, reportPrivateUsage=false

from __future__ import annotations

# The package re-exports every public name of its submodules, so
# ``from causalab.protocol.registry import X`` keeps working for every ``X``
# the old single module defined. Of the old module's private names, only the
# ones something outside the package still reads are carried (each is marked
# with its readers below); the module's own incidental imports (``dataclasses``,
# ``Mapping``, the ``schema`` and ``errors`` names) are not — nothing read them
# through the registry (``tests/protocol/test_registry_package.py`` holds the
# package to this surface).
from causalab.protocol.registry import shapes  # noqa: F401
from causalab.protocol.registry.shapes import FeatureShape  # noqa: F401

# Submodules in dependency order, which fixes the order of the import-time
# side effects: the built-in model entries register (models), the three
# built-in trees register (families), and the alias table is checked last
# (engines). The old single module ran register_family(LLAMA_TREE) /
# register_family(GPT2_TREE) *before* its register_model calls; the two
# registries are independent, so the inversion is inert.
from causalab.protocol.registry.plans import (  # noqa: F401
    GEMMA2_PLAN,
    KV_PROJECTIONS,
    KV_REPLICATED,
    LLAMA_PLAN,
    NO_PLAN,
    PLAN_AXES,
    QWEN3_PLAN,
    QWEN35_MOE_PLAN,
    QWEN35_PLAN,
    STYLES,
    VOCABULARY_STYLES,
    ParallelPlan,
    PlanAxis,
    PlanRow,
    parallel_plan_from_hf_config,
    wildcard_layers,
)
from causalab.protocol.registry.models import (  # noqa: F401
    get_model_info,
    _HF_LAYER_STREAMS,  # read by tests/protocol/test_registry_shapes.py
    model_info_from_hf_config,
    ModelInfo,
    register_model,
)

from causalab.protocol.registry.components import (  # noqa: F401
    CAPABILITIES,
    Capability,
    capability,
    component_shape,
    COMPONENT_STREAMS,
    component_width,
    ENGINES,
    expert_axis_refusal,
    expert_neuron_group_map,
    families_in_table,
    family_in_table,
    gate_group_map,
    gate_param_shape,
    GROUP_SITE_SELECTORS,
    head_group_map,
    head_space_refusal,
    INTERIOR_ROWS,
    native_shape,
    _no_axis,  # read by protocol/equivalence.py
    OVERRIDE_KEYS,
    OverrideKey,
    Packing,
    PACKINGS,
    Predicate,
    predicate_holds,
    _PREDICATE_MEANS,  # read by scripts/generate_support_tables.py
    PREDICATES,
    site_group_map,
    site_group_map_whole,
    TAP_KINDS,
    TapKind,
    unavailable_at_load,
    _with_note,  # read by protocol/equivalence.py
    write_policy_refusal,
)

from causalab.protocol.registry.families import (  # noqa: F401
    ATTENTION_FUNCTION_SLOTS,
    DELTA_KERNEL_SLOTS,
    EXPERTS_FUNCTION_SLOTS,
    FAMILIES,
    _FAMILIES,  # read by tests/protocol/test_family_contract.py, tests/neural/…/test_family_plugin.py
    family,
    family_for,
    FamilyAdapter,
    GPT2_TREE,
    GPTJ_TREE,
    HOOK_KINDS,
    HookKind,
    identities_for,
    Identity,
    identity,
    Inventory,
    inventory,
    LayerInventory,
    LLAMA_TREE,
    mixer_children,
    Probe,
    register_family,
    Tap,
    TAP_SCOPES,
    TapScope,
    TreeAddress,
    walk,
)

from causalab.protocol.registry.engines import (  # noqa: F401
    alias_would_rebind,
    backend_pair,
    BACKEND_PAIRS,
    BackendPair,
    _check_aliases,  # read by tests/protocol/test_family_contract.py
    components_served_by,
    declared_capabilities,
    DOCS_TABLE_MODEL,
    effective_capabilities,  # read by protocol/reports.py
    ENGINE_VERBS,
    Relation,
    RELATIONS,
    write_capabilities,
)

from causalab.protocol.registry.docs import (  # noqa: F401
    engine_component_summary,
    render_component_tables,
    render_family_table,
    render_gate_group_table,
    render_widthless_components,
    _write_cell,  # read by scripts/generate_support_tables.py
)

__all__ = [
    "BACKEND_PAIRS",
    "CAPABILITIES",
    "COMPONENT_STREAMS",
    "DOCS_TABLE_MODEL",
    "ENGINE_VERBS",
    "ENGINES",
    "FAMILIES",
    "GEMMA2_PLAN",
    "GPT2_TREE",
    "GPTJ_TREE",
    "HOOK_KINDS",
    "INTERIOR_ROWS",
    "LLAMA_PLAN",
    "LLAMA_TREE",
    "NO_PLAN",
    "OVERRIDE_KEYS",
    "PACKINGS",
    "KV_PROJECTIONS",
    "KV_REPLICATED",
    "PLAN_AXES",
    "PREDICATES",
    "QWEN3_PLAN",
    "QWEN35_MOE_PLAN",
    "QWEN35_PLAN",
    "RELATIONS",
    "STYLES",
    "TAP_KINDS",
    "VOCABULARY_STYLES",
    "TAP_SCOPES",
    "BackendPair",
    "Capability",
    "FamilyAdapter",
    "HookKind",
    "Identity",
    "LayerInventory",
    "ModelInfo",
    "OverrideKey",
    "Packing",
    "ParallelPlan",
    "PlanAxis",
    "PlanRow",
    "Predicate",
    "Relation",
    "Tap",
    "TapKind",
    "TapScope",
    "TreeAddress",
    "alias_would_rebind",
    "backend_pair",
    "capability",
    "component_shape",
    "component_width",
    "components_served_by",
    "declared_capabilities",
    "effective_capabilities",
    "engine_component_summary",
    "expert_axis_refusal",
    "expert_neuron_group_map",
    "families_in_table",
    "family",
    "family_for",
    "family_in_table",
    "gate_group_map",
    "gate_param_shape",
    "get_model_info",
    "GROUP_SITE_SELECTORS",
    "head_group_map",
    "head_space_refusal",
    "identities_for",
    "identity",
    "inventory",
    "mixer_children",
    "model_info_from_hf_config",
    "native_shape",
    "parallel_plan_from_hf_config",
    "predicate_holds",
    "register_family",
    "register_model",
    "render_component_tables",
    "render_family_table",
    "site_group_map",
    "unavailable_at_load",
    "wildcard_layers",
    "write_capabilities",
    "write_policy_refusal",
]
