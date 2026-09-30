"""Measure Qwen operation % SOL on H100/B200; launch with python or torchrun."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from typing import Any

from causalab.sol.benchmark import fingerprint, markdown_report, measure
from causalab.sol.hardware import b200_sxm, h100_sxm
from causalab.sol.qwen36_v2 import CONTRACT
from causalab.sol.qwen36 import Qwen36A3B, REVISION, TEXT_CONFIG, qwen36_workloads


def device_profile(name: str, memory_bytes: int) -> str:
    """Recognize only the SKU families represented by the published profiles."""
    name = name.upper()
    if any(tag in name for tag in ("MIG", "NVL", "PCIE", "PCI-E")):
        raise ValueError(f"unsupported GPU variant: {name}")
    if (
        "H100" in name
        and ("SXM" in name or "HBM3" in name)
        and 70e9 <= memory_bytes <= 90e9
    ):
        return "h100"
    if "B200" in name and 160e9 <= memory_bytes <= 210e9:
        return "b200"
    raise ValueError(
        f"GPU {name} ({memory_bytes} bytes) is not a recognized H100 SXM/B200 profile"
    )


class CudaRuntime:
    def __init__(self) -> None:
        import torch
        import torch.distributed as dist

        if not torch.cuda.is_available():
            raise RuntimeError(
                "This runner requires an H100/B200 CUDA GPU; no CUDA device is available"
            )
        self.torch, self.dist = torch, dist
        self.world = int(os.environ.get("WORLD_SIZE", "1"))
        self.rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if (
            not 1 <= self.world <= 8
            or int(os.environ.get("LOCAL_WORLD_SIZE", str(self.world))) != self.world
        ):
            raise ValueError("runner supports one node with 1–8 data-parallel ranks")
        torch.cuda.set_device(self.local_rank)
        self.device = f"cuda:{self.local_rank}"
        if self.world > 1:
            dist.init_process_group("nccl", timeout=timedelta(minutes=5))

    def synchronize(self) -> None:
        self.torch.cuda.synchronize(self.local_rank)

    def barrier(self) -> None:
        if self.world > 1:
            self.dist.barrier(device_ids=[self.local_rank])

    def reset_peak_memory(self) -> None:
        self.torch.cuda.reset_peak_memory_stats(self.local_rank)

    def gather(self, value) -> list:
        if self.world == 1:
            return [value]
        values = [None] * self.world
        self.dist.all_gather_object(values, value)
        return values

    def sample(self, seconds: float) -> dict:
        # Peak counters are read before the timing-aggregation allocations.
        cuda = self.torch.cuda
        rank_data = self.gather(
            {
                "seconds": seconds,
                "peak_allocated_bytes": cuda.max_memory_allocated(self.local_rank),
                "peak_reserved_bytes": cuda.max_memory_reserved(self.local_rank),
            }
        )
        return {
            "rank_seconds": [r["seconds"] for r in rank_data],
            "rank_peak_allocated_bytes": [r["peak_allocated_bytes"] for r in rank_data],
            "rank_peak_reserved_bytes": [r["peak_reserved_bytes"] for r in rank_data],
        }

    def sync_gradients(self, stage) -> None:
        if self.world > 1:
            for parameter in stage.parameters():
                if parameter.grad is None:
                    raise RuntimeError(
                        "missing trainable gradient before DP synchronization"
                    )
                self.dist.all_reduce(parameter.grad)
                parameter.grad.div_(self.world)

    def hardware(self, selected: str):
        props = self.torch.cuda.get_device_properties(self.local_rank)
        detected = device_profile(props.name, props.total_memory)
        if selected != "auto" and selected != detected:
            raise ValueError(f"GPU {props.name} does not match hardware={selected}")
        records = self.gather(
            {
                "rank": self.rank,
                "hostname": socket.gethostname(),
                "gpu": props.name,
                "total_memory_bytes": props.total_memory,
                "sm_count": props.multi_processor_count,
                "compute_capability": list(
                    self.torch.cuda.get_device_capability(self.local_rank)
                ),
                "profile": detected,
            }
        )
        if (
            len({r["hostname"] for r in records}) != 1
            or len({r["gpu"] for r in records}) != 1
        ):
            raise ValueError("all ranks must use identical GPUs on one node")
        profile = h100_sxm() if detected == "h100" else b200_sxm()
        profile = replace(
            profile,
            memory_bytes=min(r["total_memory_bytes"] for r in records),
            provenance=profile.provenance
            + " Runtime capacity uses CUDA-reported device memory; available allocator headroom remains unknown.",
        )
        return profile, records

    def close(self) -> None:
        if self.world > 1 and self.dist.is_initialized():
            self.dist.destroy_process_group()


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="JSON report; Markdown is written beside it",
    )
    p.add_argument("--hardware", choices=["auto", "h100", "b200"], default="auto")
    p.add_argument(
        "--batch",
        type=int,
        default=8,
        help="global batch across all data-parallel ranks",
    )
    p.add_argument("--sequence", type=int, default=128)
    p.add_argument("--updates", type=int, default=100)
    p.add_argument("--source-batches", type=int, default=10)
    p.add_argument(
        "--eval-batches",
        type=int,
        default=20,
        help="total source+intervened evaluation forwards (even)",
    )
    p.add_argument("--suffix-layers", type=int, default=13)
    p.add_argument("--rank", type=int, default=8, help="subspace rank")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--warmup", type=int, default=1, help="full workload trials to discard"
    )
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument(
        "--operations",
        nargs="+",
        choices=[
            "inference",
            "activation_harvest",
            "interchange",
            "subspace_apply",
            "dbm_apply",
            "subspace_train",
            "dbm_train",
        ],
        help="default: all seven",
    )
    p.add_argument(
        "--allow-download",
        action="store_true",
        help="allow downloading the pinned ~69GB checkpoint; default is cache only",
    )
    return p


def _software() -> dict:
    versions = {}
    for package in ("torch", "transformers", "flash-linear-attention", "causal-conv1d"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    source_commit = os.environ.get("CAUSALAB_SOL_SOURCE_COMMIT")
    if source_commit:
        versions["git_commit"] = source_commit
        versions["git_dirty"] = False
        versions["source_kind"] = "remote_git_archive"
    else:
        root = Path(__file__).resolve().parents[2]
        try:
            versions["git_commit"] = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip()
            versions["git_dirty"] = bool(
                subprocess.check_output(
                    ["git", "status", "--porcelain"], cwd=root, text=True
                )
            )
        except (OSError, subprocess.CalledProcessError):
            versions["git_commit"] = None
    versions["python"] = sys.version
    return versions


def prepare_workloads(args: argparse.Namespace):
    """Validate the job before loading weights or submitting remote compute."""
    if args.output.suffix != ".json":
        raise ValueError("--output must end in .json")
    if args.warmup < 0 or args.repeats < 1:
        raise ValueError("warmup must be nonnegative and repeats positive")
    spec = Qwen36A3B(batch=args.batch, sequence=args.sequence)
    works = qwen36_workloads(
        spec,
        updates=args.updates,
        source_batches=args.source_batches,
        eval_batches=args.eval_batches,
        suffix_layers=args.suffix_layers,
        rank=args.rank,
    )
    if (
        args.source_batches > args.updates
        or args.eval_batches % 2
        or args.suffix_layers >= 40
    ):
        raise ValueError(
            "require source_batches <= updates, even eval_batches, suffix_layers < 40"
        )
    if args.operations:
        works = [w for w in works if w.name in args.operations]
    return spec, works


def main() -> None:
    args = parser().parse_args()
    spec, works = prepare_workloads(args)
    runtime = CudaRuntime()
    report: dict = {
        "schema_version": 2,
        "execution_contract": CONTRACT,
        "reference_kind": "conditional_major_work_estimate",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "results": [],
        "command": sys.argv,
        "definition": "percent_sol = 100 * analytical reference seconds / measured median seconds",
        "scope": "synthetic-token compute-path microbenchmark, not protocol/dataset/I/O end-to-end",
        "limitations": [
            "Published peaks and conditional analytical costs, not GPU utilization.",
            "Uniform-routing reference is compared with actual routing on synthetic tokens.",
            "Installed Delta kernels may differ from the analytical chunk64 algorithm.",
            "Setup, featurizer reset and timing aggregation excluded; losses, Python/hooks and gradient synchronization included.",
            "Built-in driver supports DP only; TP requires an actual backend through the library API.",
        ],
    }

    def save():
        if runtime.rank == 0:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            # Keep completed operations if a later operation fails or OOMs.
            temporary = args.output.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            temporary.replace(args.output)
            args.output.with_suffix(".md").write_text(markdown_report(report))

    try:
        if args.batch % runtime.world:
            raise ValueError(
                "global batch must be divisible by the data-parallel world size"
            )
        hardware, devices = runtime.hardware(args.hardware)
        import torch
        from transformers import AutoModelForCausalLM
        from causalab.sol.benchmark_torch import TorchOperations

        torch.manual_seed(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        report.update(
            hardware=asdict(hardware),
            devices=devices,
            software=_software(),
            model=spec.metadata(),
            execution={
                **vars(args),
                "output": str(args.output),
                "dp": runtime.world,
                "tp": 1,
            },
            cuda_version=torch.version.cuda,
            allow_tf32=False,
            attention_implementation="eager",
            driver_sha256=fingerprint(
                Path(__file__).with_name("benchmark_torch.py").read_text()
            ),
        )
        report["model"]["forward_operation_breakdown_scope"] = (
            "whole-forward architecture diagnostic; operation-specific counts are in each result's workload_ledger"
        )
        if runtime.rank == 0:
            print(
                "Loading pinned Qwen text weights (excluded from timing)...", flush=True
            )
        model: Any = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3.6-35B-A3B",
            revision=REVISION,
            dtype=torch.bfloat16,
            attn_implementation="eager",
            local_files_only=not args.allow_download,
        )
        model = model.to(runtime.device)
        report["experts_implementation"] = getattr(
            model.config, "_experts_implementation", None
        )
        text_config = model.config.to_dict()
        for key, expected in TEXT_CONFIG.items():
            if text_config.get(key) != expected:
                raise ValueError(
                    f"loaded model config differs from pinned reference: {key}"
                )
        report["loaded_parameter_count"] = sum(p.numel() for p in model.parameters())
        expected_params = sum(spec.parameter_breakdown().values())
        if report["loaded_parameter_count"] != expected_params:
            raise ValueError(
                f"loaded text parameter count differs: {report['loaded_parameter_count']} != {expected_params}"
            )
        driver = TorchOperations(
            model,
            batch=args.batch // runtime.world,
            sequence=args.sequence,
            updates=args.updates,
            source_batches=args.source_batches,
            eval_batches=args.eval_batches,
            suffix_layers=args.suffix_layers,
            rank=args.rank,
            device=runtime.device,
            seed=args.seed,
            data_rank=runtime.rank,
            sync_gradients=runtime.sync_gradients,
        )
        for work in works:
            report["active_operation"] = work.name
            save()
            result = measure(
                driver.case(work),
                hardware,
                runtime,
                dp=runtime.world,
                warmup=args.warmup,
                repeats=args.repeats,
                progress=lambda message: (
                    print(message, flush=True) if runtime.rank == 0 else None
                ),
            )
            report["results"].append(result)
            report.pop("active_operation", None)
            save()
    except Exception as exc:
        report["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "rank": runtime.rank,
        }
        save()
        raise
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
