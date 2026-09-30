"""Pin the exported surface of ``causalab/neural/shared/featurizers/``.

``SURFACE`` records the names that the engines, the analysis scripts, and the
tests import from the package root. The tests hold ``__all__`` to that
surface, check that each export is its submodule's own object, and refuse
private names on the root. Removing a read name fails its reader at
collection. The reader census itself is a review-time check: an unread name
added to both lists passes. Import a private name such as ``sharing._SCOPE``
or ``build._stage_width`` from its submodule.
"""

from __future__ import annotations

from types import ModuleType

import pytest

import causalab.neural.shared.featurizers as featurizers

pytestmark = pytest.mark.unit

SURFACE: dict[str, tuple[str, ...]] = {
    "sharing": ("featurizer_cache",),
    "stages": (
        "Cayley",
        "LoadedLinear",
        "ORTHONORMAL_TOLERANCE",
        "Sae",
        "Stage",
        "Standardize",
        "Subspace",
        "orthonormality_deviation",
    ),
    "gate": ("BudgetPool", "Gate", "gate_poles", "link_budget_pools"),
    "build": ("FeaturizerStack", "build_stack", "stage_output_width"),
}


def test_the_package_exports_exactly_the_surface() -> None:
    expected = {name for names in SURFACE.values() for name in names}
    assert set(featurizers.__all__) == expected, (
        f"exported but not in the census: {sorted(set(featurizers.__all__) - expected)}; "
        f"in the census but not exported: {sorted(expected - set(featurizers.__all__))}"
    )
    for module, names in SURFACE.items():
        sub = getattr(featurizers, module)
        for name in names:
            assert getattr(featurizers, name) is getattr(sub, name), (module, name)


def test_nothing_but_the_surface_is_a_package_attribute() -> None:
    public = {
        name
        for name, value in vars(featurizers).items()
        if not name.startswith("_") and not isinstance(value, ModuleType)
    }
    assert public == set(featurizers.__all__), sorted(public ^ set(featurizers.__all__))
    carried = sorted(
        name
        for name, value in vars(featurizers).items()
        if name.startswith("_")
        and not name.startswith("__")
        and not isinstance(value, ModuleType)
    )
    assert carried == [], f"private names re-exported: {carried}"
