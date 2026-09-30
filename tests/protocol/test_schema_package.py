"""Pin the exported surface of ``causalab/protocol/schema/``.

The package root exports the public names that modules, scripts, tests, and
``generated:`` docs blocks import from it. ``SURFACE`` records that census at
the time of the narrowing. The tests hold ``__all__`` to that surface, check
that each export is its submodule's own object, and refuse private names on
the root. Removing a read name fails its reader at collection. The reader
census itself is a review-time check: an unread name added to both lists
passes.

Each submodule imports on its own in a fresh interpreter without ``torch``.
At run time ``types`` imports no package sibling, ``featurizers`` and
``positions`` import only ``types``, and ``parse`` imports the three. The
tests pin this on the import statements.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

import causalab.protocol.schema as schema

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[2]
PACKAGE = REPO / "causalab" / "protocol" / "schema"
SUBMODULES = ("types", "featurizers", "positions", "parse")

#: The names read through the package root at the time of the narrowing, by
#: the submodule that defines each. The tests hold ``__all__`` to this surface,
#: each export to its submodule's object, and the root free of private names.
#: Whether every entry still has a reader is a review-time census.
SURFACE: dict[str, tuple[str, ...]] = {
    "types": (
        "ADDITIVE_MECHANISMS",
        "ALIGNMENT_CARDINALITIES",
        "ALL_POSITIONS",
        "AlignmentCardinality",
        "AggregationSpec",
        "BoundAggregation",
        "COMPONENTS",
        "CodeSpec",
        "ConstraintSpec",
        "DEPRECATED_COMPONENTS",
        "DEPRECATED_IN",
        "DataRole",
        "Do",
        "do_operand_slots",
        "Document",
        "GROUP_ORDER",
        "HEADER_FIELDS",
        "IMSpec",
        "LAYERLESS_COMPONENTS",
        "MATCH_MODES",
        "MECHANISMS",
        "METHOD_SECTIONS",
        "METRIC_DOMAINS",
        "METRIC_FIELDS",
        "METRIC_FIELD_DEFAULTS",
        "METRIC_KINDS",
        "MIGRATABLE_PROTOCOL_VERSIONS",
        "MINIMUM_COUNT_FIELD",
        "MODEL_DTYPE_DEFAULT",
        "ModelRef",
        "NAMED_SECTIONS",
        "OPTIMIZER_DEFAULTS",
        "OPTIMIZER_SCHEDULES",
        "OPTIONAL_METRIC_FIELDS",
        "ObjectiveTerm",
        "operand_params",
        "operand_reads",
        "PRECISION_DTYPES",
        "PROTOCOL_VERSION",
        "RAGGED_FIELD",
        "RAGGED_POLICIES",
        "READ_TARGET_METRIC_KINDS",
        "REQUIRED_METHOD_SECTIONS",
        "RESERVED_NAMES",
        "ReadRef",
        "read_is_vocabulary",
        "ReadSpec",
        "RowRole",
        "SAVE_KINDS",
        "SECTION_ORDER",
        "STREAMS",
        "SaveEntry",
        "SiteSpec",
        "Stream",
        "Sweep",
        "RETIRED_TOKEN_FORMS",
        "TOKEN_FORMS",
        "TrainSpec",
        "VOCAB_TOP_K_RANKING",
        "WHOLE_WINDOW_METRIC_KINDS",
        "WRITES_DURING_GENERATION_FIELD",
        "WriteSpec",
        "concrete_int",
        "concrete_str",
        "dotted_path",
        "metric_column_fields",
        "tree_path",
    ),
    "featurizers": (
        "AnnealSchedule",
        "CONTROL_DEFAULTS",
        "FEATURIZER_FAMILIES",
        "FEATURIZER_FIELDS",
        "FEATURIZER_FIELD_CONDITIONS",
        "FEATURIZER_KINDS",
        "FEATURIZER_SLOTS",
        "FORWARD_MASKS",
        "FeaturizerSpec",
        "GATE_AXES",
        "GATE_DEAD_RULES",
        "GATE_DEFAULT_MAP",
        "GATE_GROUPS",
        "GATE_GROUP_AXES",
        "GATE_MAPS",
        "GATE_PARAMETRIZATIONS",
        "HARD_CONCRETE_STRETCH",
        "HARD_CONCRETE_TEMPERATURE",
        "K_SCHEDULE_OF",
        "OBJECTIVE_WEIGHT_PREFIX",
        "PhaseSpec",
        "TRAINABLE_KINDS",
        "hard_concrete_theta",
        "hard_concrete_threshold",
        "render_featurizer_kind_table",
        "render_field_legality_table",
        "render_field_refusals",
        "render_gate_map_table",
    ),
    "positions": (
        "CONTINUATION_SEGMENT",
        "PositionSpec",
        "SegmentsSpec",
        "SpanSpec",
        "span_length",
    ),
    "parse": (
        "DRAW_KINDS",
        "PER_PARAMS_OPTIMIZER_FIELDS",
        "REGULARIZER_COSTS",
        "REGULARIZER_KINDS",
        "SAVE_REDUCTIONS",
        "SCORES_INIT_DEFAULTS",
        "SCORES_INIT_KEYS",
        "check_protocol_version",
        "inline_train_saves",
        "load_raw",
        "parse_document",
        "to_base_form",
    ),
}


# --------------------------------------------------------------------------- #
# the surface
# --------------------------------------------------------------------------- #


def test_the_package_exports_exactly_the_surface() -> None:
    expected = {name for names in SURFACE.values() for name in names}
    assert set(schema.__all__) == expected, (
        f"exported but not in the census: {sorted(set(schema.__all__) - expected)}; "
        f"in the census but not exported: {sorted(expected - set(schema.__all__))}"
    )
    assert all(hasattr(schema, name) for name in expected)


def test_every_exported_name_is_the_submodules_own() -> None:
    """A re-export is the object itself, and it comes from the submodule the
    census says defines it."""
    for module, names in SURFACE.items():
        sub = getattr(schema, module)
        for name in names:
            assert getattr(schema, name) is getattr(sub, name), (module, name)


def test_no_private_name_is_carried() -> None:
    """The parse helpers, the legality-table cells and the span predicates are
    read from their homes; the package root carries none of them."""
    carried = sorted(
        name
        for name, value in vars(schema).items()
        if name.startswith("_")
        and not name.startswith("__")
        and not isinstance(value, ModuleType)
    )
    assert carried == [], f"private names re-exported: {carried}"


def test_nothing_but_the_surface_is_a_package_attribute() -> None:
    """``__all__`` is the whole public namespace — a name imported by the
    ``__init__`` but left out of ``__all__`` would be a silent export."""
    public = {
        name
        for name, value in vars(schema).items()
        if not name.startswith("_")
        and not isinstance(value, ModuleType)
        and name != "annotations"  # the ``__future__`` flag
    }
    assert public == set(schema.__all__), sorted(public ^ set(schema.__all__))


def test_the_root_imports_only_from_the_four_submodules() -> None:
    init = ast.parse((PACKAGE / "__init__.py").read_text())
    modules = {
        node.module
        for node in init.body
        if isinstance(node, ast.ImportFrom) and node.module != "__future__"
    }
    assert modules == {f"causalab.protocol.schema.{m}" for m in SUBMODULES}


# --------------------------------------------------------------------------- #
# standalone import, torch-free
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("submodule", SUBMODULES)
def test_a_submodule_imports_standalone_without_torch(submodule: str) -> None:
    """A fresh interpreter imports the submodule alone; ``torch`` is not
    reached. (Importing a submodule executes the package ``__init__`` first,
    so this also proves the package's re-export list resolves.)"""
    code = (
        "import sys\n"
        f"import causalab.protocol.schema.{submodule}\n"
        "print('torch' in sys.modules)\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False", "torch was imported"


# --------------------------------------------------------------------------- #
# the runtime graph
# --------------------------------------------------------------------------- #


def _runtime_package_imports(path: Path) -> set[str]:
    """The ``causalab.protocol.schema.<x>`` submodules ``path`` imports at
    module level outside ``if TYPE_CHECKING:``."""
    out: set[str] = set()
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.If) and ast.unparse(node.test) == "TYPE_CHECKING":
            continue
        if isinstance(node, ast.ImportFrom) and node.module:
            prefix = "causalab.protocol.schema."
            if node.module.startswith(prefix):
                out.add(node.module[len(prefix) :])
    return out


def test_the_runtime_import_graph_is_acyclic() -> None:
    edges = {
        name: _runtime_package_imports(PACKAGE / f"{name}.py") for name in SUBMODULES
    }
    assert edges["types"] == set()
    assert edges["featurizers"] == {"types"}
    assert edges["positions"] == {"types"}
    assert edges["parse"] == {"types", "featurizers", "positions"}
    # the package __init__ is a leaf consumer: it re-exports and defines nothing
    init = ast.parse((PACKAGE / "__init__.py").read_text()).body
    assert all(
        isinstance(node, (ast.ImportFrom, ast.Expr, ast.Assign)) for node in init
    )
    # and segments.py / positions/spans.py hang below positions, never the other
    # way round (spans live in the positions package)
    for sibling in (
        PACKAGE.parent / "segments.py",
        PACKAGE.parent / "positions" / "spans.py",
    ):
        assert "positions" in _runtime_package_imports(sibling)
    for name in SUBMODULES:
        imported = {
            node.module
            for node in ast.walk(ast.parse((PACKAGE / f"{name}.py").read_text()))
            if isinstance(node, ast.ImportFrom)
        }
        assert not imported & {"causalab.protocol.segments", "causalab.protocol.spans"}
