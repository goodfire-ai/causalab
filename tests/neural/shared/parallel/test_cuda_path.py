"""The CUDA launch path, pinned without CUDA (``docs/model_parallelism.md`` §3).

``causalab run … --parallel tp=2 --device cuda`` on a 2-GPU node runs as two
ranks, each on ``cuda:LOCAL_RANK`` over ``nccl``. The decisions along that
path are made from strings and integers — the device word, the rank, the
backend the group was initialised with — and this module holds every one of
them on a machine with no CUDA device, so that a change that would only
break on the GPU node is caught here:

- the CLI's ``_launched`` re-spells ``--device cuda`` as ``cuda:LOCAL_RANK``
  before the engine is built, and enters the group with that device;
- ``join_group`` pins the CUDA device (``torch.cuda.set_device``) **before**
  ``init_process_group("nccl")`` — NCCL binds a communicator to the device
  current at initialisation;
- the mesh's device type follows the backend (``nccl`` → ``cuda``), the
  collective's device is *this* rank's ordinal (``torch.cuda.current_device``,
  what ``set_device`` made current), and every header, cell and receive
  buffer is allocated on it — a tensor on another ordinal is refused by name
  rather than handed to NCCL;
- the loader places a rank's tensors on the one device the rank was given
  (a device list is refused above world 1), and the fit's memory meter
  reads ``mem_get_info`` on that ordinal, not on ``cuda:0``.

The real-backend half — the same path with two H100s — is
``tests/golden/test_parallel_parity.py``.
"""

from __future__ import annotations

import argparse
import datetime
import types
from typing import Any

import pytest
import torch

from causalab import cli
from causalab.neural.engines.pytorch_hooks import budget
from causalab.neural.shared.devices import DeviceMap
from causalab.neural.shared.parallel import collective as collective_module
from causalab.neural.shared.parallel import launcher
from causalab.neural.shared.parallel.collective import (
    HEADER_WIDTH,
    CollectiveError,
    TorchCollective,
    encode_header,
)
from causalab.neural.shared.parallel.mesh import device_type_of
from causalab.neural.shared.parallel.watchdog import Settings
from causalab.protocol.parallel import ParallelGeometry

pytestmark = pytest.mark.unit

TP2 = ParallelGeometry(tensor=2)


class _FakeMesh:
    """A mesh with a device type and no process group: enough for the
    collective's device decision, which is made at construction."""

    def __init__(self, device_type: str) -> None:
        self.device_type = device_type
        self.geometry = TP2
        self.rank = 0

    def group(self, axis: Any) -> None:
        return None

    def local_rank(self, axis: Any) -> int:
        return 0

    def size(self, axis: Any) -> int:
        return 1


# --------------------------------------------------------------------------- #
# the CLI: --device cuda becomes cuda:LOCAL_RANK before the engine exists
# --------------------------------------------------------------------------- #


class TestLaunchedCli:
    def test_a_joined_cuda_rank_enters_on_its_local_ordinal(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``torchrun``'s environment for rank 1 of 2 on ordinal 1: the CLI
        enters the group with ``cuda:1``, the protocol run sees ``cuda:1``
        and the publisher the launcher built, and leaves afterwards."""
        for name, value in {
            "WORLD_SIZE": "2",
            "RANK": "1",
            "LOCAL_RANK": "1",
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": "29500",
        }.items():
            monkeypatch.setenv(name, value)
        monkeypatch.delenv(launcher.LAUNCHER_VARIABLE, raising=False)
        calls: list[tuple[str, Any]] = []
        publisher = object()

        def enter(launch: launcher.Launch, geometry: Any, device: str) -> Any:
            calls.append(("enter", (launch, geometry, device)))
            return publisher

        def run(args: argparse.Namespace, env: Any) -> int:
            calls.append(("run", (args.device, args.publisher)))
            return 0

        monkeypatch.setattr(launcher, "enter", enter)
        monkeypatch.setattr(
            launcher, "leave", lambda p, status: calls.append(("leave", (p, status)))
        )
        args = argparse.Namespace(parallel_geometry=TP2, device="cuda")
        assert cli._launched(args, env=None, argv=["run", "doc.json"], run=run) == 0
        assert [name for name, _ in calls] == ["enter", "run", "leave"]
        launch, geometry, device = calls[0][1]
        assert launch == launcher.Launch("joined", 1, 2, 1)
        assert geometry == TP2 and device == "cuda:1"
        assert calls[1][1] == ("cuda:1", publisher)
        # the run's status reaches the peers through leave (§3 "when a rank dies")
        assert calls[2][1] == (publisher, 0)
        assert args.device == "cuda:1"

    def test_the_parent_spawns_and_never_enters(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("WORLD_SIZE", raising=False)
        spawned: list[Any] = []
        monkeypatch.setattr(
            launcher,
            "spawn",
            # the parent hands the device word on, so a child's SIGABRT under
            # NCCL is named as NCCL's watchdog (launcher.describe_child_exit)
            lambda geometry, argv, device: spawned.append((geometry, argv, device))
            or 3,
        )
        monkeypatch.setattr(
            launcher, "enter", lambda *a: pytest.fail("the parent entered a group")
        )
        args = argparse.Namespace(parallel_geometry=TP2, device="cuda")
        never = lambda *a: pytest.fail("the parent ran the document")  # noqa: E731
        assert cli._launched(args, env=None, argv=["run", "doc.json"], run=never) == 3
        assert spawned == [(TP2, ["run", "doc.json"], "cuda")]
        assert args.device == "cuda"  # the children re-spell it, each for itself


# --------------------------------------------------------------------------- #
# join_group: set_device before init_process_group("nccl")
# --------------------------------------------------------------------------- #


class TestJoinGroup:
    """The store is built (``launcher.rendezvous``) between the device pin
    and the group's initialisation, which takes it with the collective
    timeout (§3 "when a rank dies")."""

    SETTINGS = Settings(timeout=90.0, grace=9.0)

    class _AgreeingStore:
        """The rendezvous store as ``join_group``'s environment agreement
        sees it: every peer spelled its settings as this rank did."""

        def __init__(self) -> None:
            self.values: dict[str, str] = {}

        def set(self, key: str, value: str) -> None:
            self.values[key] = value

        def get(self, key: str) -> bytes:
            (mine,) = set(self.values.values())
            return mine.encode()

        def check(self, keys: Any) -> bool:
            return True

    def _rendezvous(self, order: list[Any], store: object, heartbeat: object):
        def fake(launch, geometry, settings):
            order.append(("rendezvous", launch.rank, settings))
            return store, heartbeat

        return fake

    def test_nccl_pins_the_local_ordinal_before_initialising(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        order: list[tuple[str, Any]] = []
        store, heartbeat = self._AgreeingStore(), object()
        monkeypatch.setattr(
            torch.cuda,
            "set_device",
            lambda device: order.append(("set_device", device)),
        )
        monkeypatch.setattr(
            torch.distributed,
            "init_process_group",
            lambda backend, **kwargs: order.append(("init", (backend, kwargs))),
        )
        monkeypatch.setattr(
            launcher, "rendezvous", self._rendezvous(order, store, heartbeat)
        )
        returned = launcher.join_group(
            launcher.Launch("joined", 1, 2, 1),
            "nccl",
            geometry=ParallelGeometry(tensor=2),
            settings=self.SETTINGS,
        )
        assert returned is heartbeat
        assert order == [
            ("set_device", 1),
            ("rendezvous", 1, self.SETTINGS),
            (
                "init",
                (
                    "nccl",
                    {
                        "store": store,
                        "rank": 1,
                        "world_size": 2,
                        "timeout": datetime.timedelta(seconds=90),
                    },
                ),
            ),
        ]

    def test_gloo_touches_no_cuda_device(self, monkeypatch: pytest.MonkeyPatch) -> None:
        order: list[Any] = []
        monkeypatch.setattr(
            torch.cuda, "set_device", lambda device: pytest.fail("set_device on gloo")
        )
        monkeypatch.setattr(
            torch.distributed,
            "init_process_group",
            lambda backend, **kwargs: order.append(backend),
        )
        monkeypatch.setattr(
            launcher,
            "rendezvous",
            self._rendezvous(order, self._AgreeingStore(), object()),
        )
        launcher.join_group(
            launcher.Launch("spawned", 0, 2, 0),
            "gloo",
            geometry=ParallelGeometry(data=2),
            settings=self.SETTINGS,
        )
        assert order == [("rendezvous", 0, self.SETTINGS), "gloo"]

    def test_solo_initialises_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            torch.distributed,
            "init_process_group",
            lambda *a, **k: pytest.fail("a solo launch initialised a group"),
        )
        monkeypatch.setattr(
            launcher,
            "rendezvous",
            lambda *a, **k: pytest.fail("a solo launch built a store"),
        )
        assert (
            launcher.join_group(
                launcher.SOLO_LAUNCH,
                "nccl",
                geometry=ParallelGeometry(),
                settings=self.SETTINGS,
            )
            is None
        )

    def test_the_backend_follows_the_device_word_the_rank_was_given(self) -> None:
        launch = launcher.Launch("joined", 1, 2, 1)
        device = launcher.device_for(launch, "cuda")
        assert device == "cuda:1"
        assert launcher.backend_for(device) == "nccl"
        assert device_type_of(launcher.backend_for(device)) == "cuda"
        assert device_type_of(launcher.backend_for("cpu")) == "cpu"


# --------------------------------------------------------------------------- #
# the collective's device: this rank's ordinal, and nothing else
# --------------------------------------------------------------------------- #


class TestCollectiveDevice:
    def test_the_cuda_device_is_the_current_ordinal(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
        assert collective_module._device_of("cuda") == torch.device("cuda", 1)
        assert collective_module._device_of("cpu") == torch.device("cpu")

    def test_the_collective_runs_on_the_meshs_device(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
        assert TorchCollective(_FakeMesh("cuda")).device == torch.device("cuda", 1)  # type: ignore[arg-type]
        assert TorchCollective(_FakeMesh("cpu")).device == torch.device("cpu")  # type: ignore[arg-type]

    def test_the_header_is_encoded_on_the_collectives_device(self) -> None:
        """With a fake device (``meta``: allocates, holds no bytes): the
        header travels on the device the collective was built for, which
        under ``nccl`` is where NCCL expects it."""
        header = encode_header(torch.empty(2, 3), torch.device("meta"))
        assert header.device.type == "meta"
        assert header.shape == (HEADER_WIDTH,) and header.dtype == torch.int64
        c = TorchCollective(_FakeMesh("meta"))  # type: ignore[arg-type]
        assert c.device == torch.device("meta")
        assert encode_header(torch.empty(4), c.device).device == c.device

    def test_a_tensor_on_another_ordinal_is_refused_by_name(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Rank 1's collective runs on ``cuda:1``; a tensor on ``cuda:0`` —
        the same device *type* — is refused before NCCL sees it, naming both."""
        monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
        c = TorchCollective(_FakeMesh("cuda"))  # type: ignore[arg-type]
        elsewhere = types.SimpleNamespace(device=torch.device("cuda", 0))
        with pytest.raises(CollectiveError) as err:
            c._on_device(elsewhere, "all_gather")  # type: ignore[arg-type]
        assert "cuda:0" in str(err.value) and "cuda:1" in str(err.value)
        c._on_device(
            types.SimpleNamespace(device=torch.device("cuda", 1)), "all_gather"
        )  # type: ignore[arg-type]

    def test_a_cpu_tensor_is_refused_on_a_cuda_collective(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
        c = TorchCollective(_FakeMesh("cuda"))  # type: ignore[arg-type]
        with pytest.raises(CollectiveError):
            c._on_device(torch.empty(2), "send")


# --------------------------------------------------------------------------- #
# the loader's device and the fit's memory meter follow the rank
# --------------------------------------------------------------------------- #


class TestRankDevice:
    def test_the_ranks_one_device_is_the_ordinal_it_was_given(self) -> None:
        devices = DeviceMap.parse("cuda:1", 4)
        assert devices.single == torch.device("cuda", 1)
        assert devices.is_cuda
        assert (
            DeviceMap.parse("cuda:0,cuda:1", 4).single is None
        )  # refused above world 1

    def test_the_meter_reads_the_ranks_ordinal(self) -> None:
        meter = budget.cuda_meter(DeviceMap.parse("cuda:1", 4))
        assert isinstance(meter, budget._CudaMeter)
        assert meter.devices == (torch.device("cuda", 1),)
        assert budget.cuda_meter(DeviceMap.parse("cpu", 4)) is None


@pytest.mark.unit
def test_the_meter_orders_several_devices_by_name() -> None:
    """Sort devices by spelling because ``torch.device`` has no ordering."""
    meter = budget.cuda_meter(DeviceMap.parse("cuda:1,cuda:0", 4))
    assert isinstance(meter, budget._CudaMeter)
    assert meter.devices == (torch.device("cuda", 0), torch.device("cuda", 1))
