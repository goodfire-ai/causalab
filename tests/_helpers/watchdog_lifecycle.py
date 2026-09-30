"""Run heartbeat lifecycle checks under torchrun on gloo or NCCL.

Use --mode skew with --delay-seconds greater than CAUSALAB_RANK_GRACE to
check that rank 0 keeps the store available while another rank finishes.
In failure mode the selected rank exits nonzero while peers enter a collective.
--rank-store-port uses a separate rank-0-owned store instead of the torchrun
agent's store; pass the same reachable port to every worker.
"""

from __future__ import annotations

import argparse
import json
import os
import time

from causalab.neural.shared.parallel import launcher
from causalab.protocol.parallel import parse_geometry


def _report(rank: int, event: str, **fields: int | float | str) -> None:
    # One pipe write keeps concurrent ranks' JSON records on separate lines.
    record = {"rank": rank, "event": event, "time": time.time(), **fields}
    os.write(1, (json.dumps(record) + "\n").encode())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--mode", choices=("skew", "failure"), default="skew")
    parser.add_argument("--delay-seconds", type=float, default=6.0)
    parser.add_argument("--selected-rank", type=int)
    parser.add_argument("--rank-store-port", type=int)
    args = parser.parse_args()
    world = int(os.environ["WORLD_SIZE"])
    geometry = parse_geometry(f"tp={world}")
    launch = launcher.detect(geometry)
    if not isinstance(launch, launcher.Launch):
        parser.error("launch this helper through torchrun")
    if args.rank_store_port is not None:
        if not 1 <= args.rank_store_port <= 65535:
            parser.error("--rank-store-port must be between 1 and 65535")
        if str(args.rank_store_port) == os.environ.get("MASTER_PORT"):
            parser.error("--rank-store-port must differ from the agent's port")
        os.environ["MASTER_PORT"] = str(args.rank_store_port)
        os.environ[launcher.AGENT_STORE_VARIABLE] = "False"
    publisher = launcher.enter(launch, geometry, args.device)
    assert isinstance(publisher, launcher.RankPublisher)
    assert publisher.heartbeat is not None
    import torch

    from causalab.neural.shared.parallel.collective import TorchCollective

    collective = TorchCollective(publisher.mesh)
    value = torch.ones(1, device=launcher.device_for(launch, args.device))
    selected = world - 1 if args.selected_rank is None else args.selected_rank
    status = 1
    try:
        collective.all_reduce_sum(value, "model")
        _report(launch.rank, "ready", store_host=publisher.heartbeat.store_host)
        if args.mode == "skew":
            if launch.rank == selected:
                time.sleep(args.delay_seconds)
            status = 0
        elif launch.rank == selected:
            status = 3
        else:
            collective.all_reduce_sum(value, "model")
            raise AssertionError("a collective completed without the failed rank")
        return status
    finally:
        _report(launch.rank, "finishing", status=status)
        launcher.leave(publisher, status)
        _report(launch.rank, "left", status=status)


if __name__ == "__main__":
    raise SystemExit(main())
