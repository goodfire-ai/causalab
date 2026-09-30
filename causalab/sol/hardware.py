"""Published peak profiles in decimal units.

SXM GPUs in an HGX/DGX NVSwitch domain, not PCIe cards or GB200 NVL72.
No sustained efficiency is implied. Latency=0 is an idealization, not a spec.
"""

from causalab.sol.model import Hardware

H100_SOURCE = "https://www.nvidia.com/en-us/data-center/h100/"
H100_LINK_SOURCE = (
    "https://docs.nvidia.com/cuda/archive/12.1.1/hopper-tuning-guide/index.html"
)
B200_SOURCE = "https://www.nvidia.com/en-us/data-center/hgx/"
B200_MEMORY_SOURCE = (
    "https://docs.nvidia.com/dgx/dgxb200-user-guide/introduction-to-dgxb200.html"
)
B200_BANDWIDTH_SOURCE = "https://lenovopress.lenovo.com/lp2226.pdf"


def h100_sxm() -> Hardware:
    return Hardware(
        name="NVIDIA H100 SXM 80GB (HGX/NVSwitch, published peaks)",
        flops_per_second={"bf16_dense": 1979e12 / 2, "fp32": 67e12},
        hbm_bytes_per_second=3.35e12,
        memory_bytes=80e9,
        link_bytes_per_second=900e9 / 2,
        collective_latency_seconds=0,
        provenance=f"Verified 2026-09-08: {H100_SOURCE}; {H100_LINK_SOURCE}. "
        "BF16 sparse 1979 TF/s divided by 2; FP32 is non-Tensor-Core. "
        "NVLink bidirectional 900 GB/s divided by 2 for one-way injection. "
        "Zero ring latency is an unmeasured idealization; capacity has no runtime reserve.",
    )


def b200_sxm() -> Hardware:
    return Hardware(
        name="NVIDIA B200 SXM 180GB (HGX/NVSwitch, published peaks)",
        flops_per_second={"bf16_dense": 36e15 / 8 / 2, "fp32": 600e12 / 8},
        hbm_bytes_per_second=8e12,
        memory_bytes=1440e9 / 8,
        link_bytes_per_second=1.8e12 / 2,
        collective_latency_seconds=0,
        provenance=f"Verified 2026-09-08: {B200_SOURCE}; {B200_MEMORY_SOURCE}; "
        f"HBM bandwidth: {B200_BANDWIDTH_SOURCE}. HGX 8-GPU sparse BF16 36 PF/s "
        "divided by 8 GPUs and 2 for dense; FP32 600 TF/s divided by 8. "
        "NVIDIA FP32 75 TF/s selected over OEM's 80 TF/s. "
        "NVLink bidirectional 1.8 TB/s divided by 2. Zero ring latency is an "
        "unmeasured idealization; 180 GB shipping capacity, no runtime reserve.",
    )
