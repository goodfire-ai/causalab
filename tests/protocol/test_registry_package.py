"""The registry package keeps the namespace ``causalab.protocol.registry`` and
the old ``causalab.protocol.shapes`` had before the split (the ``shapes``
names live on ``registry/shapes.py``; its one-beat shim is deleted), imports
torch-free module by module, and is what
the engine classes read their capability set from.

The name lists below are the namespaces of the two old modules as recorded
just before the split (``dir(module)`` minus dunders, and each module's
``__all__``) — the committed baseline every consumer's import was
written against. They are the proof the surface survived, so they are spelled
out rather than derived. ``REGISTRY_NAMES`` is that ``dir`` less the old
module's own imports (``dataclasses``, ``Mapping``, the ``schema`` and
``errors`` names) and less its private names: nothing read those through the
registry, so the package does not carry them — except the seven in
``PRIVATE_NAMES_READ``, each pinned to the file that reaches it.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

import causalab.protocol.registry as registry
import causalab.protocol.registry.shapes as moved_shapes
from causalab.protocol.engine import Engine, component_capability
from causalab.protocol.registry.engines import effective_capabilities
from causalab.protocol.registry import (
    ENGINE_VERBS,
    ENGINES,
    components_served_by,
    declared_capabilities,
    write_capabilities,
)

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[2]
PACKAGE = REPO / "causalab" / "protocol" / "registry"

#: The submodules in the order ``registry/__init__.py`` imports them: each may
#: import only the ones before it (``schema`` and ``errors`` sit outside the
#: package), and the order fixes the import-time side effects — the built-in
#: model entries register (``models``), the two built-in trees register
#: (``families``), the alias table is checked last (``engines``). The old
#: single module ran ``register_family`` before ``register_model``; the two
#: registries are independent, so the inversion is inert.
SUBMODULES: tuple[str, ...] = (
    "shapes",
    "plans",
    "models",
    "components",
    "families",
    "engines",
    "docs",
)

#: The module-scope side effect each submodule carries, as the call it makes.
SIDE_EFFECTS: dict[str, str] = {
    "models": "register_model",
    "families": "register_family",
    "engines": "_check_aliases",
}

REGISTRY_NAMES: tuple[str, ...] = (
    "ATTENTION_FUNCTION_SLOTS",
    "BACKEND_PAIRS",
    "BackendPair",
    "CAPABILITIES",
    "COMPONENT_STREAMS",
    "Capability",
    "DELTA_KERNEL_SLOTS",
    "DOCS_TABLE_MODEL",
    "ENGINES",
    "EXPERTS_FUNCTION_SLOTS",
    "FAMILIES",
    "FamilyAdapter",
    "FeatureShape",
    "GPT2_TREE",
    "GROUP_SITE_SELECTORS",
    "HOOK_KINDS",
    "HookKind",
    "INTERIOR_ROWS",
    "Identity",
    "Inventory",
    "LLAMA_TREE",
    "LayerInventory",
    "ModelInfo",
    "OVERRIDE_KEYS",
    "OverrideKey",
    "PACKINGS",
    "PREDICATES",
    "Packing",
    "Predicate",
    "Probe",
    "RELATIONS",
    "Relation",
    "TAP_KINDS",
    "TAP_SCOPES",
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
    "head_group_map",
    "head_space_refusal",
    "identities_for",
    "identity",
    "inventory",
    "mixer_children",
    "model_info_from_hf_config",
    "native_shape",
    "predicate_holds",
    "register_family",
    "register_model",
    "render_component_tables",
    "render_family_table",
    "render_gate_group_table",
    "render_widthless_components",
    "site_group_map",
    "site_group_map_whole",
    "unavailable_at_load",
    "walk",
    "write_capabilities",
    "write_policy_refusal",
)

#: The old module's private names something outside the package still reads,
#: and where. A private name with no reader is not re-exported: the package
#: is held to exactly this set below, so narrowing it means editing a reader,
#: not deleting a guard.
PRIVATE_NAMES_READ: dict[str, tuple[str, ...]] = {
    "_FAMILIES": (
        "tests/protocol/test_family_contract.py",
        "tests/neural/engines/pytorch_hooks/test_family_plugin.py",
    ),
    "_HF_LAYER_STREAMS": ("tests/protocol/test_registry_shapes.py",),
    "_PREDICATE_MEANS": ("scripts/generate_support_tables.py",),
    "_check_aliases": ("tests/protocol/test_family_contract.py",),
    "_no_axis": ("causalab/protocol/equivalence.py",),
    "_with_note": ("causalab/protocol/equivalence.py",),
    "_write_cell": ("scripts/generate_support_tables.py",),
}

REGISTRY_ALL: tuple[str, ...] = (
    "BACKEND_PAIRS",
    "CAPABILITIES",
    "COMPONENT_STREAMS",
    "DOCS_TABLE_MODEL",
    "ENGINES",
    "FAMILIES",
    "GPT2_TREE",
    "HOOK_KINDS",
    "INTERIOR_ROWS",
    "LLAMA_TREE",
    "OVERRIDE_KEYS",
    "PACKINGS",
    "PREDICATES",
    "RELATIONS",
    "TAP_KINDS",
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
    "predicate_holds",
    "register_family",
    "register_model",
    "render_component_tables",
    "render_family_table",
    "site_group_map",
    "unavailable_at_load",
    "write_capabilities",
    "write_policy_refusal",
)

SHAPES_NAMES: tuple[str, ...] = (
    "Axis",
    "AxisKind",
    "FeatureShape",
    "INNER_KINDS",
    "Literal",
    "OUTER_KINDS",
    "_ALL_KINDS",
    "_BATCH",
    "_POSITION",
    "annotations",
    "attention_pattern",
    "bds",
    "bhsd",
    "bs",
    "bs_flat_heads",
    "bs_fused_blocks",
    "bs_fused_heads",
    "bsd",
    "bsh",
    "bshd",
    "chunked_state",
    "dataclasses",
    "flat_td",
    "flat_topk",
    "flat_topk_features",
    "flat_topk_fused_features",
    "get_args",
    "math",
    "state_matrix",
)

SHAPES_ALL: tuple[str, ...] = (
    "Axis",
    "AxisKind",
    "FeatureShape",
    "INNER_KINDS",
    "OUTER_KINDS",
    "attention_pattern",
    "bds",
    "bs",
    "bsh",
    "bs_flat_heads",
    "bs_fused_blocks",
    "bs_fused_heads",
    "bshd",
    "bhsd",
    "bsd",
    "chunked_state",
    "flat_td",
    "flat_topk",
    "flat_topk_features",
    "flat_topk_fused_features",
)


# --------------------------------------------------------------------------- #
# namespace preservation
# --------------------------------------------------------------------------- #


def test_every_old_registry_name_is_a_package_attribute() -> None:
    missing = [name for name in REGISTRY_NAMES if not hasattr(registry, name)]
    assert not missing, f"causalab.protocol.registry lost {missing}"


def test_the_package_carries_exactly_the_private_names_something_reads() -> None:
    """``dir()``-parity was never the contract — only specific attributes
    were, and these are them. Both directions: a private name nothing reads
    is gone from the package, and every pinned reader still reads its name
    (so the table above cannot outlive the imports it describes)."""
    carried = {
        name
        for name in dir(registry)
        if name.startswith("_") and not name.startswith("__")
    }
    assert carried == set(PRIVATE_NAMES_READ), (
        f"unread private names re-exported: {sorted(carried - set(PRIVATE_NAMES_READ))}; "
        f"read but missing: {sorted(set(PRIVATE_NAMES_READ) - carried)}"
    )
    for name, readers in PRIVATE_NAMES_READ.items():
        for reader in readers:
            assert re.search(rf"\b{name}\b", (REPO / reader).read_text()), (
                name,
                reader,
            )


def test_the_old_modules_incidental_imports_are_not_package_attributes() -> None:
    """The old module's ``dataclasses``, ``Mapping``, ``ValidationError``,
    ``COMPONENTS``, … were its imports, not its surface; nothing read them
    through the registry and the package does not re-export them."""
    incidental = {
        "dataclasses",
        "MappingProxyType",
        "Mapping",
        "ValidationError",
        "COMPONENTS",
        "STREAMS",
    }
    leaked = {name for name in incidental if hasattr(registry, name)}
    assert not leaked, f"causalab.protocol.registry re-exports {sorted(leaked)}"


def test_the_old_registry_all_is_a_subset_of_the_new() -> None:
    assert set(REGISTRY_ALL) <= set(registry.__all__)
    assert all(hasattr(registry, name) for name in registry.__all__)


def test_every_old_shapes_name_is_on_the_moved_module() -> None:
    missing = [name for name in SHAPES_NAMES if not hasattr(moved_shapes, name)]
    assert not missing, f"causalab.protocol.registry.shapes lost {missing}"


# The one-beat star-import shim ``causalab/protocol/shapes.py`` and its pin
# are deleted.


def test_the_package_exposes_the_same_objects_as_the_submodules() -> None:
    """A re-export is the object itself, so a test that mutates
    ``registry._FAMILIES`` or calls ``registry._check_aliases`` reaches the
    submodule's own."""
    for name in SUBMODULES:
        module = getattr(registry, name)
        for attr in dir(module):
            if attr.startswith("__") or not hasattr(registry, attr):
                continue
            assert getattr(registry, attr) is getattr(module, attr), (name, attr)


# --------------------------------------------------------------------------- #
# import graph
# --------------------------------------------------------------------------- #


def _package_imports(module: str) -> set[str]:
    """The registry submodules ``module`` imports at module scope."""
    tree = ast.parse((PACKAGE / f"{module}.py").read_text())
    prefix = "causalab.protocol.registry"
    found: set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom) or node.module is None:
            continue
        if node.module == prefix:
            found.update(alias.name for alias in node.names)
        elif node.module.startswith(prefix + "."):
            found.add(node.module.removeprefix(prefix + "."))
    return found & set(SUBMODULES)


@pytest.mark.parametrize("module", SUBMODULES)
def test_a_submodule_imports_only_what_precedes_it(module: str) -> None:
    allowed = set(SUBMODULES[: SUBMODULES.index(module)])
    forward = _package_imports(module) - allowed
    assert not forward, f"{module} imports {sorted(forward)}, which sit after it"


def test_the_package_imports_the_submodules_in_the_pinned_order() -> None:
    """``SUBMODULES`` is what ``__init__.py`` does, not a wish: its
    ``from causalab.protocol.registry.<m> import`` statements run in exactly
    this order (the ``shapes`` module import counts as its first)."""
    tree = ast.parse((PACKAGE / "__init__.py").read_text())
    prefix = "causalab.protocol.registry"
    seen: list[str] = []
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom) or node.module is None:
            continue
        if node.module == prefix:
            names = [alias.name for alias in node.names]
        elif node.module.startswith(prefix + "."):
            names = [node.module.removeprefix(prefix + ".")]
        else:
            continue
        for name in names:
            if name in SUBMODULES and name not in seen:
                seen.append(name)
    assert tuple(seen) == SUBMODULES


def _module_scope_calls(module: str) -> set[str]:
    """The names called as bare statements at module scope of ``module``."""
    tree = ast.parse((PACKAGE / f"{module}.py").read_text())
    found: set[str] = set()
    for node in tree.body:
        if (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
        ):
            found.add(node.value.func.id)
    return found


def test_the_import_time_side_effects_run_in_the_pinned_order() -> None:
    """Each side effect lives in the module ``SIDE_EFFECTS`` names and nowhere
    else in the package, so the import order above is the order they run:
    ``register_model`` calls before ``register_family``, ``_check_aliases``
    last."""
    for module in SUBMODULES:
        calls = _module_scope_calls(module) & set(SIDE_EFFECTS.values())
        expected = {SIDE_EFFECTS[module]} if module in SIDE_EFFECTS else set()
        assert calls == expected, (module, calls)
    order = [m for m in SUBMODULES if m in SIDE_EFFECTS]
    assert order == ["models", "families", "engines"]


_PROBE = """
import importlib, json, sys
importlib.import_module(sys.argv[1])
print(json.dumps({"torch": "torch" in sys.modules}))
"""


@pytest.mark.parametrize(
    "module",
    [
        "causalab.protocol.registry",
        *(f"causalab.protocol.registry.{m}" for m in SUBMODULES),
    ],
)
def test_a_module_imports_standalone_without_torch(module: str) -> None:
    """In a subprocess: ``tests/conftest.py`` imports torch at session scope,
    so an in-process check would be false regardless."""
    result = subprocess.run(
        [sys.executable, "-c", _PROBE, module],
        capture_output=True,
        text=True,
        cwd=REPO,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip().splitlines()[-1]) == {"torch": False}


# --------------------------------------------------------------------------- #
# the engine rows
# --------------------------------------------------------------------------- #


#: The two engines' verbs as the classes spelled them by hand before the rows
#: took over — pinned so the rows are held to the old literals rather than
#: the classes to the rows (which would be tautological).
PYTORCH_HOOKS_VERBS = frozenset(
    {
        "grad",
        "paired_forward",
        "full_logits",
        "generate",
        "generation_writes",
        "pytorch_fn_local",
        "quantized_weights",
    }
)
NNSIGHT_VERBS = frozenset(
    {"paired_forward", "full_logits", "pytorch_fn_local", "generate"}
)
OLD_VERBS: dict[str, frozenset[str]] = {
    "pytorch_hooks": PYTORCH_HOOKS_VERBS,
    "nnsight": NNSIGHT_VERBS,
}


def test_engine_verbs_name_exactly_the_engines() -> None:
    assert ENGINE_VERBS.keys() == set(ENGINES) == set(OLD_VERBS)


def _engine_classes() -> list[type[Engine]]:
    from causalab.neural.engines.nnsight_tracing.engine import NnsightEngine
    from causalab.neural.engines.pytorch_hooks.engine import PytorchHooksEngine

    return [PytorchHooksEngine, NnsightEngine]


def test_the_engine_rows_are_the_old_literals() -> None:
    """The rows say what the classes used to say by hand: the verbs are the
    old literals, ``declared_capabilities`` is those plus the write verbs the
    rows charge, the class attribute is that call, and the routed set — the
    ``effective_capabilities`` property — is that plus one read and one write
    entry per component the engine's row serves. The property stays on
    ``Engine``: an engine need not be a row (the stubs in
    ``test_legality_before_weights.py`` are not). Its name-keyed twin,
    ``registry.engines.effective_capabilities``, is what ``pipeline._offered``
    reads for a bare engine name before an instance exists, and the two real
    engines get one answer from both."""
    classes = _engine_classes()
    assert {cls.name for cls in classes} == set(ENGINES)
    prop = Engine.effective_capabilities
    assert isinstance(prop, property) and prop.fget is not None
    for cls in classes:
        name = cls.name
        literal = OLD_VERBS[name]
        assert ENGINE_VERBS[name] == literal, name
        assert declared_capabilities(name) == literal | write_capabilities(name), name
        assert cls.capabilities == declared_capabilities(name), name
        served = components_served_by(name)
        assert prop.fget(cls) == (
            declared_capabilities(name)
            | {component_capability(c) for c in served}
            | {component_capability(c, write=True) for c in served}
        ), name
        assert effective_capabilities(name) == prop.fget(cls), name


def test_declared_capabilities_refuses_an_unknown_engine() -> None:
    with pytest.raises(AssertionError, match="unknown engine"):
        declared_capabilities("megatron")
