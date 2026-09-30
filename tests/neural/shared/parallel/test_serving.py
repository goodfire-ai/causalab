"""What the reference engine serves at ``world > 1`` (``docs/model_parallelism.md``
§3, §8.2): the two checks it makes before any weights load, in one process.

``unit``: [`check_collective`][causalab.neural.shared.parallel.serving.check_collective] accepts a collective whose group sizes are
the geometry's on every axis and refuses the first that disagrees — a ``P4``
at ``--parallel.<axis>``, at ``--parallel`` for the derived ``model`` axis
no flag names — naming both sizes; [`process_mesh`][causalab.neural.shared.parallel.serving.process_mesh] hands back the
launcher's mesh from its [`RankPublisher`][causalab.neural.shared.parallel.launcher.RankPublisher] (built here over a fake
group factory, as ``test_mesh.py`` builds it) and refuses one built for
another geometry by name, and for any other publisher builds the mesh from
the environment over the initialised default group — refused by name where
none is. The default group's presence and the environment mesh are the two
seams faked: a unit process holds no group, and ``Mesh.from_environment``
is ``test_mesh.py``'s to hold.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch.distributed as dist

from causalab.neural.shared.parallel import serving
from causalab.neural.shared.parallel.launcher import Launch, RankPublisher
from causalab.neural.shared.parallel.mesh import Mesh
from causalab.neural.shared.parallel.serving import check_collective, process_mesh
from causalab.protocol.rules.errors import ParseError
from causalab.protocol.parallel import AXES, ParallelGeometry, format_geometry
from causalab.protocol.publish import SOLO
from tests.neural.shared.parallel.test_mesh import Recorder

pytestmark = pytest.mark.unit

TP2 = ParallelGeometry(tensor=2)
EP2 = ParallelGeometry(expert=2)


class _Sized:
    """A collective reporting the geometry's group sizes, ``overrides`` on
    the named axes."""

    def __init__(self, geometry: ParallelGeometry, **overrides: int) -> None:
        self._geometry = geometry
        self._overrides = overrides

    def rank(self, axis: str) -> int:
        return 0

    def size(self, axis: str) -> int:
        if axis in self._overrides:
            return self._overrides[axis]
        return int(getattr(self._geometry, axis))


class TestCheckCollective:
    @pytest.mark.parametrize(
        "geometry",
        (
            ParallelGeometry(),
            TP2,
            ParallelGeometry(data=2, pipeline=2, context=2, tensor=2, expert=2),
        ),
        ids=("solo", "tp2", "everything"),
    )
    def test_the_geometrys_sizes_on_every_axis_pass(
        self, geometry: ParallelGeometry
    ) -> None:
        check_collective(geometry, _Sized(geometry))  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "axis", ("data", "pipeline", "context", "tensor", "expert")
    )
    def test_a_flagged_axis_that_disagrees_is_refused_at_its_flag(
        self, axis: str
    ) -> None:
        geometry = ParallelGeometry(**{axis: 2})
        with pytest.raises(ParseError) as err:
            check_collective(geometry, _Sized(geometry, **{axis: 1}))  # type: ignore[arg-type]
        assert err.value.code == "P4"
        assert err.value.path == f"--parallel.{axis}"
        text = str(err.value)
        assert f"the collective has 1 ranks on the {axis} axis" in text
        assert format_geometry(geometry) in text and "asks for 2" in text

    def test_the_derived_model_axis_is_refused_at_the_flag_itself(self) -> None:
        """``model`` is ``tensor × expert``: no ``--parallel.model`` names it,
        so the refusal sits at ``--parallel``."""
        assert "model" in AXES
        with pytest.raises(ParseError) as err:
            check_collective(TP2, _Sized(TP2, model=1))  # type: ignore[arg-type]
        assert err.value.code == "P4" and err.value.path == "--parallel"
        assert "on the model axis" in str(err.value)


class TestProcessMesh:
    def test_the_launchers_mesh_is_the_publishers(self) -> None:
        mesh = Mesh(TP2, 0, groups=Recorder())
        publisher = RankPublisher(Launch("spawned", 0, 2, 0), mesh)
        assert process_mesh(TP2, publisher) is mesh

    def test_a_mesh_built_for_another_geometry_is_refused_by_both_names(self) -> None:
        publisher = RankPublisher(
            Launch("spawned", 0, 2, 0), Mesh(TP2, 0, groups=Recorder())
        )
        with pytest.raises(ParseError) as err:
            process_mesh(EP2, publisher)
        assert err.value.code == "P4" and err.value.path == "--parallel"
        text = str(err.value)
        assert "the launcher built this rank's mesh for" in text
        assert f"for {format_geometry(TP2)}," in text
        assert f"runs under {format_geometry(EP2)}" in text

    def test_without_a_process_group_any_other_publisher_is_refused_by_name(
        self,
    ) -> None:
        assert not dist.is_initialized()
        with pytest.raises(ParseError) as err:
            process_mesh(TP2, SOLO)
        assert err.value.code == "P4" and err.value.path == "--parallel"
        text = str(err.value)
        assert f"{format_geometry(TP2)} asks for a world of 2 ranks" in text
        assert "no torch.distributed process group is initialised" in text
        assert "launch through the SPMD launcher (docs/model_parallelism.md §3)" in text
        assert "or hand the engine a collective" in text

    def test_over_an_initialised_group_the_mesh_comes_from_the_environment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The two seams faked: the default group is there, and the
        environment mesh is whatever ``Mesh.from_environment`` builds for
        exactly this geometry."""
        built: list[ParallelGeometry] = []
        sentinel = object()

        class _EnvironmentMesh:
            @classmethod
            def from_environment(cls, geometry: ParallelGeometry) -> Any:
                built.append(geometry)
                return sentinel

        monkeypatch.setattr(dist, "is_initialized", lambda: True)
        monkeypatch.setattr(serving, "Mesh", _EnvironmentMesh)
        assert process_mesh(TP2, SOLO) is sentinel
        assert built == [TP2]
