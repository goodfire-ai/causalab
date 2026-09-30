"""The noise draw of the published Figure 1 (e, f, g), as a ``params`` bundle.

ROME's ``trace_with_patch`` (lines 167-171 and 187-197 of
<https://github.com/kmeng01/rome/blob/0874014cd9837e4365f3e6f3c71400ef11509e04/experiments/causal_trace.py>) builds a new
``numpy.random.RandomState(1)`` on every call and adds
``noise * rs.randn(samples, subject_tokens, width)`` to the subject's token
embeddings of the corrupted rows. So every restoration and the corrupted run
itself see the same draw. The figure's ``plot_all_flow`` (line 586) keeps the
default ``noise=0.1`` of ``plot_hidden_flow`` (line 512).

This step writes that draw once, under the slot ``value``, rounded to
float32. The write casts its operand to the fp32 slice it writes
(``causalab/neural/shared/mechanisms.py``, ``_coerce``), so float64 storage
would keep nothing the write uses, and an MPS device cannot hold a float64
tensor at all. The two tracing documents load it through ``params`` and add it
with ``add_scaled`` at ``alpha`` 0.1. Row ``i`` of the draw goes to row ``i``
of the table, which is ROME's batch row ``i + 1`` (its row 0 is the clean
run). The shape is the step's inputs: ``samples`` rows (the table's ten
identical prompts), ``tokens`` subject tokens (``The Space Needle`` is four
GPT-2 tokens) and ``width`` (GPT-2 XL's 1600). A loaded ``params`` bundle
must name the model of the document that loads it
(``causalab/protocol/rules/data.py``, ``check_loaded_featurizers``), and a
draw has no tensor input to inherit that from, so ``model_key`` and
``model_revision`` are inputs too. A document that names another model or
revision refuses the bundle with ``[V15]``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

from causalab.io.step_io import StepError, write_tensor

__all__ = ["main", "cli", "draw"]

#: ``demos/papers/``: this script sits in ``workflows/scripts/rome_fig1/``.
PAPERS = Path(__file__).resolve().parents[3]
RUN = PAPERS / "artifacts" / "output" / "rome_fig1"
#: The slot a loaded ``params`` constant is read from by convention
#: (``docs/intervention_protocol.md``, sec. 2.6).
SLOT = "value"
#: The workflow's inputs, as the CLI's defaults: ROME's seed and ten samples,
#: the four GPT-2 tokens of ``The Space Needle``, GPT-2 XL's width, and the
#: model the tracing documents name.
DEFAULTS = {
    "seed": 1,
    "samples": 10,
    "tokens": 4,
    "width": 1600,
    "model_key": "gpt2-xl",
    "model_revision": "15ea56dee5df4983c59b2538573817e1667135e2",
}


def draw(seed: int, samples: int, tokens: int, width: int) -> Any:
    """``numpy.random.RandomState(seed).randn(samples, tokens, width)`` as a
    float64 torch tensor, the values ROME's ``prng`` returns.

    Raises:
        StepError: ``samples``, ``tokens`` or ``width`` is below 1.
    """
    import numpy as np
    import torch

    for name, value in (("samples", samples), ("tokens", tokens), ("width", width)):
        if int(value) < 1:
            raise StepError(
                f"noise_draw: {name} must be a positive integer, got {value!r}"
            )
    values = np.random.RandomState(int(seed)).randn(
        int(samples), int(tokens), int(width)
    )
    return torch.from_numpy(values)


def main(inputs: Mapping[str, Any], outputs: Mapping[str, Path]) -> None:
    """Write the draw of ``inputs`` to ``outputs["noise"]`` as float32 under
    ``SLOT``, stamped with the model identity of ``inputs``."""
    write_tensor(
        Path(outputs["noise"]),
        draw(
            inputs["seed"], inputs["samples"], inputs["tokens"], inputs["width"]
        ).float(),
        slot=SLOT,
        identity={
            "model_key": str(inputs["model_key"]),
            "model_revision": str(inputs["model_revision"]),
        },
    )


def cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    for name, default in DEFAULTS.items():
        parser.add_argument(
            f"--{name.replace('_', '-')}", type=type(default), default=default
        )
    parser.add_argument(
        "--out", type=Path, default=RUN / "noise_draw" / "noise.safetensors"
    )
    args = parser.parse_args(argv)
    main({name: getattr(args, name) for name in DEFAULTS}, {"noise": args.out})
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
