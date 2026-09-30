"""Compute the gated delta rule for sequences that fit one chunk.

The package includes a float32 Torch reference, a Triton forward and
backward, and a per-forward dispatcher. Eligible short sequences use the
single-chunk form with zero initial state. Other calls keep the bound
kernel. ``options`` supplies the sequence threshold.
"""

from causalab.neural.shared.gdn_short.binding import (
    selects_single_chunk,
    short_seq_kernel_path,
)
from causalab.neural.shared.gdn_short.options import ShortSeqKernelOptions
from causalab.neural.shared.gdn_short.reference import (
    recurrent_gated_delta_rule_reference,
    single_chunk_gated_delta_rule_torch,
)
from causalab.neural.shared.gdn_short.triton_kernel import (
    MAX_SEQ_LEN,
    single_chunk_gated_delta_rule,
    triton_available,
)

__all__ = [
    "MAX_SEQ_LEN",
    "ShortSeqKernelOptions",
    "recurrent_gated_delta_rule_reference",
    "selects_single_chunk",
    "short_seq_kernel_path",
    "single_chunk_gated_delta_rule",
    "single_chunk_gated_delta_rule_torch",
    "triton_available",
]
