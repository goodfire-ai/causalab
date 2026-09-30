"""The shipped script modules' repository import closures, frozen (workflow
spec §4.2) — the layering census.

The repository-wide walk (``import_closure`` with ``repository=True``) is how
the suite proves what a shipped script reaches: the protocol core and the step
I/O, never an engine, never numerics, never a layer the script has no business
importing (``events.py``, ``derived.py``, ``fan_out.py``, ``nested.py`` —
each has a guard below or in its own test module). An import added to
``causalab/io/step_io.py`` widens what every shipped script pulls into a
torch-free ``validate``, so each hashed module's closure is listed here, member
by member, and compared. A change is a deliberate edit to this table, never a
drift.

None of it is identity. The *identity* walk (``repository=False``) drops every
member under the ``causalab`` package — those bytes are runtime identity, the
``tree_digest`` every step record carries and ``--resume`` compares (§7) — so
a ``{"module": …}`` step carries no closure keys at all, and no edit to a
module in this table moves a shipped or demo workflow's digest. The pins
below hold that: every shipped and demo script entry names a module and
carries neither key; the parent-package switch is **off** (a parent
``__init__`` is executed, not declared); and the isolated-runtime script with
no repository imports carries no closure keys either (valid work is not
refused and gets an identity that says so).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

from causalab.protocol.identity import closure_sha256, import_closure
from causalab.io.env import ResolutionEnv
from causalab.workflow.document import load_workflow

from tests.protocol.test_vocabulary_census import HASHED_SCRIPTS
from tests.workflow.test_isolation import (
    SCRIPT as ISOLATED_SCRIPT,
    _document as isolated_document,  # pyright: ignore[reportPrivateUsage]
)
from tests._helpers.paths import WORKFLOWS_DIR

from tests._helpers.demos import demo_env, demo_workflows

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[2]
WORKFLOWS = WORKFLOWS_DIR
DEMOS = REPO / "demos"

#: The modules every shipped script reaches: ``io/step_io.py`` and, through
#: it, the env → registry → schema chain of the protocol core
#: (``schema/parse.py`` imports ``estimand.py``, so the estimand
#: vocabulary is a member; the span grammar is one through
#: ``schema/positions.py``, its home since the schema split — ``spans.py`` /
#: ``segments.py`` themselves, the load-time checks and rule 27, are reached
#: by no shipped script). Every member is named directly: the one-beat
#: star-import shims of the protocol refactor (``protocol/resolve.py``,
#: ``protocol/tables.py``, ``protocol/errors.py``, …) were deleted
#: and every importer repointed to the home, so the homes alone are members.
SHARED: tuple[str, ...] = (
    # the io half of a load: the resolution
    # environment — torch-free, which ``tests/test_architecture_layering.py``
    # holds it to
    "causalab/io/env.py",
    # the safetensors implementation (the Python surface over the Rust
    # extension `_core`, which has no `.py` and so is not a member — imported
    # as `import …._core as _core`, since `from . import _core` would resolve
    # to the package `__init__`): reached through tensor_files.py below,
    # torch-importing modules every one — which is why step_io imports
    # tensor_files function-locally, so a torch-free `validate` still executes
    # none of it
    "causalab/io/fastersafetensors/_buffers.py",
    "causalab/io/fastersafetensors/_distributed.py",
    "causalab/io/fastersafetensors/_dtypes.py",
    "causalab/io/fastersafetensors/_files.py",
    "causalab/io/fastersafetensors/_header.py",
    "causalab/io/fastersafetensors/_select.py",
    "causalab/io/fastersafetensors/_serialize.py",
    "causalab/io/fastersafetensors/_slices.py",
    "causalab/io/fastersafetensors/errors.py",
    "causalab/io/fastersafetensors/torch.py",
    # ``sources.py`` and ``tables.py``: the readers ``env.py`` re-exports
    # (``resolve_artifact_fields``) and the metric-table format — torch-free,
    # like ``env.py`` above
    "causalab/io/sources.py",
    "causalab/io/step_io.py",
    "causalab/io/tables.py",
    # the one import for the repository's tensor files; reached
    # function-locally from step_io
    "causalab/io/tensor_files.py",
    "causalab/protocol/bundles.py",
    "causalab/protocol/estimand.py",
    # hashing: ``env.py`` imports the artifact
    # identity schema from here, and ``identity`` re-exports the check half of
    # the old ``code.py`` from ``rules/code.py`` — which makes ``rules/code.py``
    # (AST checks, stdlib only) the one rules module beside ``rules/errors``
    # a shipped script reaches
    "causalab/protocol/identity.py",
    # the KV-head replication arithmetic (docs/model_parallelism.md §6.6):
    # ``registry/plans.py`` imports it for ``ParallelPlan.for_geometry``,
    # torch-free and importing nothing but the rule table
    "causalab/protocol/kv_replication.py",
    # the axis vocabulary: ``bundles.py``,
    # ``results.py`` and ``rules/document.py`` import it
    # directly; the enumeration (``neural/shared/sweep.py``) is reached by no
    # shipped script
    "causalab/protocol/lowering.py",
    # the registry package: the package
    # ``__init__`` is the module ``io/env.py`` imports, and it imports every
    # submodule
    "causalab/protocol/registry/__init__.py",
    "causalab/protocol/registry/components.py",
    "causalab/protocol/registry/docs.py",
    "causalab/protocol/registry/engines.py",
    "causalab/protocol/registry/families.py",
    "causalab/protocol/registry/models.py",
    "causalab/protocol/registry/plans.py",
    "causalab/protocol/registry/shapes.py",
    # the rules package: ``rules/errors.py`` is
    # the rule table every consumer imports; ``rules/code.py`` arrives through
    # ``identity.py`` (above); no other rules module (``document``, ``data``,
    # ``capability`` — the checklist itself) is reached by any shipped script
    "causalab/protocol/rules/code.py",
    "causalab/protocol/rules/errors.py",
    "causalab/protocol/schema/__init__.py",
    # the canonical form: ``identity.sign_step`` — the one hasher of a
    # step, called by the engine and the workflow layer — reaches it
    # function-locally, and the walk counts function-local imports
    "causalab/protocol/schema/explicit.py",
    "causalab/protocol/schema/featurizers.py",
    "causalab/protocol/schema/parse.py",
    "causalab/protocol/schema/positions.py",
    "causalab/protocol/schema/types.py",
    "causalab/tables.py",
)

#: The built-in reduction step. Its closure is what a `reduce` step pulls into
#: a torch-free load; the intro demos' amplification workflow names it, so it
#: sits in the frozen table with the other shipped step scripts.
REDUCE = "causalab/workflow/scripts/reduce.py"
REDUCE_CLOSURE: tuple[str, ...] = tuple(
    sorted((*SHARED, "causalab/workflow/reduction.py"))
)

#: hashed module → its declared closure, sorted by path. Exactly the set the
#: vocabulary census exempts (``HASHED_SCRIPTS``), asserted below.
CLOSURES: dict[str, tuple[str, ...]] = {
    "causalab/analysis/fit_pca.py": SHARED,
    "causalab/analysis/harvest_difference.py": SHARED,
    # the size-matched random gate of demos/papers/workflows/arithmetic_neurons.json
    "causalab/analysis/random_mask.py": SHARED,
    "causalab/io/plots/workflow_figures.py": tuple(
        sorted(
            (
                *SHARED,
                "causalab/io/plots/figure_format.py",
                "causalab/io/step_record.py",
            )
        )
    ),
    REDUCE: REDUCE_CLOSURE,
    "causalab/workflow/scripts/select.py": tuple(
        sorted(
            (
                *SHARED,
                # Shared selection uses only the standard library.
                "causalab/analysis/selection.py",
                "causalab/io/step_record.py",
            )
        )
    ),
}


def _closure(module: str, **kwargs: Any) -> dict[str, str]:
    return import_closure(REPO / module, root=REPO, **kwargs)


def test_fan_out_is_in_no_hashed_closure() -> None:
    """The fan-out layer (§2.9) is a member of no hashed script's
    closure, of the reduce script's and of SHARED — the same guard
    `test_conditional.py` holds for conditional.py, so no digest moves."""
    module = "causalab/workflow/fan_out.py"
    assert module not in SHARED
    for hashed in CLOSURES:
        assert module not in _closure(hashed), hashed


def test_nested_is_in_no_hashed_closure() -> None:
    """The nested-workflow layer (§2.10) is a member of no hashed
    script's closure, of the reduce script's and of SHARED — the same guard as
    for conditional.py and fan_out.py, so no digest moves."""
    module = "causalab/workflow/nested.py"
    assert module not in SHARED
    for hashed in CLOSURES:
        assert module not in _closure(hashed), hashed


def _demo_env(document: Path) -> ResolutionEnv:
    """A demo carries its own tables (``tests/_helpers/demos.py``)."""
    return demo_env(document)


# --------------------------------------------------------------------------- #
# the frozen table
# --------------------------------------------------------------------------- #


def test_the_frozen_table_is_the_hashed_set() -> None:
    assert set(CLOSURES) == HASHED_SCRIPTS


@pytest.mark.parametrize("module", sorted(CLOSURES), ids=lambda m: Path(m).stem)
def test_a_hashed_module_has_the_frozen_closure(module: str) -> None:
    got = tuple(_closure(module))
    want = CLOSURES[module]
    assert got == want, (
        f"{module}'s repository import closure moved — what a torch-free "
        "`validate` pulls in changed. If the import is right, update this table "
        "deliberately (docs/workflow_protocol_internals.md §4.2); no digest moves with it.\n"
        f"new members: {sorted(set(got) - set(want))}\n"
        f"gone: {sorted(set(want) - set(got))}"
    )


@pytest.mark.parametrize("module", sorted(CLOSURES), ids=lambda m: Path(m).stem)
def test_the_identity_walk_admits_nothing_from_the_package(module: str) -> None:
    """The two walks, side by side on every hashed module: the layering walk
    reaches the frozen table; the identity walk — the one the loaders call —
    reaches nothing, because every member is under the package and the
    package is runtime identity (§4.2, §7)."""
    assert _closure(module), "the layering walk found nothing"
    assert _closure(module, repository=False) == {}


def test_the_reduce_script_has_the_frozen_closure() -> None:
    assert tuple(_closure(REDUCE)) == REDUCE_CLOSURE


def test_every_member_is_a_tracked_module_under_the_repo_root() -> None:
    """The manifest keys are repo-relative paths to files that exist, so a
    reader can open the file a moved digest names."""
    for members in CLOSURES.values():
        for member in members:
            assert (REPO / member).is_file(), member
            assert member.startswith("causalab/") and member.endswith(".py")


# --------------------------------------------------------------------------- #
# the switch, pinned off
# --------------------------------------------------------------------------- #


def test_parent_packages_are_not_declared(env: Any) -> None:
    """The declared closure has no *parent* ``__init__.py`` in it — a package
    imported by name is a member like any module, its parents are not; the
    switch adds them, and the loader does not use the switch. Two
    ``__init__``s are declared: ``causalab/protocol/registry/__init__.py``
    (the module ``io/env.py`` imports) and
    ``causalab/protocol/schema/__init__.py`` (every ``from
    causalab.protocol.schema import …`` names the package itself). The parents
    the switch adds are *only* the parents: ``causalab/protocol/__init__.py``
    is lazy (PEP 562), so following it pulls in none of the compiler, loader
    or runner it re-exports."""
    select = "causalab/workflow/scripts/select.py"
    declared = _closure(select)
    inits = {member for member in declared if member.endswith("__init__.py")}
    assert inits == {
        "causalab/protocol/registry/__init__.py",
        "causalab/protocol/schema/__init__.py",
    }
    assert "causalab/protocol/__init__.py" not in declared

    executed = _closure(select, include_parents=True)
    assert set(declared) < set(executed)
    assert "causalab/protocol/__init__.py" in executed
    added = set(executed) - set(declared)
    assert all(member.endswith("__init__.py") for member in added), sorted(added)
    assert "causalab/protocol/pipeline.py" not in executed, (
        "protocol/__init__.py is lazy; following the parents must add no fan-out"
    )

    loaded = load_workflow(WORKFLOWS / "weekdays.json", env)
    assert "closure" not in loaded.canonical["steps"]["best"]


# --------------------------------------------------------------------------- #
# valid work
# --------------------------------------------------------------------------- #


def test_an_isolated_script_carries_no_closure_keys(tmp_path: Path, env: Any) -> None:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "count.py").write_text(ISOLATED_SCRIPT)
    document = isolated_document(tmp_path, deps=["packaging"])
    loaded = load_workflow(document, env, workflow_dir=tmp_path)
    entry = loaded.canonical["steps"]["count"]
    assert "closure" not in entry and "closure_sha256" not in entry
    assert closure_sha256({}) == hashlib.sha256(b"").hexdigest()  # the fixed value
    assert load_workflow(document, env, workflow_dir=tmp_path).digest == loaded.digest


SHIPPED = sorted(WORKFLOWS.glob("*.json"))
DEMO_WORKFLOWS = demo_workflows()


def test_the_workflow_census_found_something() -> None:
    assert len(SHIPPED) >= 2 and len(DEMO_WORKFLOWS) >= 9


@pytest.mark.parametrize(
    "path", SHIPPED + DEMO_WORKFLOWS, ids=[p.stem for p in SHIPPED + DEMO_WORKFLOWS]
)
def test_every_shipped_and_demo_workflow_loads_with_no_closure_keys(
    path: Path, env: Any
) -> None:
    """Every script step names a module in the frozen table and carries neither
    closure key — so no edit to the package moves its digest — and no other
    kind of entry carries one either."""
    loaded = load_workflow(path, env if path.parent == WORKFLOWS else _demo_env(path))
    for entry in loaded.canonical["steps"].values():
        assert "closure" not in entry and "closure_sha256" not in entry
        if entry["type"] == "script" and "module" in entry["script"]:
            # a path-located script (a demo's own plotting script) is hashed
            # by content and is no member of the package's frozen table
            module = entry["script"]["module"].replace(".", "/") + ".py"
            assert module in CLOSURES, module
