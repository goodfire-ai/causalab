"""Exercise clean shutdown skew through the real gloo launcher."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from causalab.neural.shared.parallel.spawn import loopback_interface, reserve_port

pytestmark = pytest.mark.smoke


@pytest.mark.parametrize("store_owner", ["rank", "agent"])
def test_store_host_stays_alive_until_the_slow_rank_finishes(store_owner: str) -> None:
    env = {
        **os.environ,
        "CAUSALAB_RANK_GRACE": "1",
        "CAUSALAB_COLLECTIVE_TIMEOUT": "30",
        "OMP_NUM_THREADS": "1",
    }
    interface = loopback_interface(name for _, name in socket.if_nameindex())
    if interface:
        env["GLOO_SOCKET_IFNAME"] = interface
    with reserve_port() as agent_port, reserve_port() as rank_port:
        store_args = (
            ["--rank-store-port", str(rank_port.port)] if store_owner == "rank" else []
        )
        run = subprocess.run(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--nnodes=1",
                "--nproc-per-node=2",
                "--master-addr=127.0.0.1",
                f"--master-port={agent_port.port}",
                "-m",
                "tests._helpers.watchdog_lifecycle",
                "--device",
                "cpu",
                "--mode",
                "skew",
                "--delay-seconds",
                "3",
                *store_args,
            ],
            cwd=Path(__file__).resolve().parents[4],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    assert run.returncode == 0, run.stdout + run.stderr
    events = [
        json.loads(line) for line in run.stdout.splitlines() if line.startswith("{")
    ]
    ready = [event for event in events if event["event"] == "ready"]
    assert len(ready) == 2
    assert all(event["store_host"] == store_owner for event in ready)
    left = {event["rank"]: event for event in events if event["event"] == "left"}
    assert set(left) == {0, 1}
    assert all(event["status"] == 0 for event in left.values())
    finishing = {
        event["rank"]: event for event in events if event["event"] == "finishing"
    }
    assert left[0]["time"] >= finishing[1]["time"]
    assert "refused:" not in run.stderr
