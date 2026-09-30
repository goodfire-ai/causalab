"""Capture (or check) a per-family oracle certification record.

Writes ``tests/neural/parity/goldens/<family>.json`` — the family's pinned
values in the frozen goldens' shape plus the ``certification`` block carrying
the eight certification fields — by running every case of
`tests.neural.parity.family_certification` through the reference engine
and the raw-hook oracle on the family's realization::

    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run python \\
        tests/neural/parity/update_family_goldens.py --family qwen35moe [--check]

``qwen35moe`` is ``tiny-random/qwen3.5-moe`` in fp32 on CPU (seconds);
``qwen36_a3b`` is ``Qwen/Qwen3.6-35B-A3B`` in bf16 on cuda (a GPU job, ~70 GB
resident, run it in its own process). ``--check`` re-captures and exits
non-zero if the committed record does not replay — the same comparison
``test_family_certification.py`` and ``tests/golden/test_family_certification_
a3b.py`` make (provenance equal, measured differences inside the committed
band, pins within tolerance) — without writing. ``--out`` writes elsewhere
(an A/B capture to ``cmp`` against the committed file).

The three frozen goldens beside these files (``gpt2``, ``gqa``, ``llama``) are
**not** produced by this script and must not be regenerated (docs/TESTS.md).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tests.neural.parity import family_certification as fc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--family", required=True, choices=sorted(fc.FAMILIES), help="which record"
    )
    parser.add_argument("--check", action="store_true", help="verify, do not write")
    parser.add_argument(
        "--device",
        default=None,
        help="override the family's device (e.g. cuda:0); the dtype never changes",
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="write here instead of goldens/"
    )
    args = parser.parse_args(argv)

    bundle = fc.make_bundle(args.family, device=args.device)
    capture = fc.capture_family(bundle, args.family)
    band = fc.BANDS[fc.FAMILIES[args.family].dtype]
    failures = fc.certification_failures(capture, band)
    if failures:
        print(
            f"the engine and the oracle disagree outside the {bundle.dtype} band "
            f"({len(failures)} comparison(s)); the record is not certifiable:",
            file=sys.stderr,
        )
        for name, diff in failures:
            print(f"  {diff!r}  {name}", file=sys.stderr)
        return 2
    text = fc.render_record(capture.record)
    target = args.out or fc.record_path(args.family)
    if args.check:
        if not target.exists():
            print(f"{target} does not exist", file=sys.stderr)
            return 1
        committed = json.loads(target.read_text())
        problems = fc.compare_records(committed, capture.record)
        if problems:
            print(f"{target} does not replay:", file=sys.stderr)
            for problem in problems:
                print(f"  {problem}", file=sys.stderr)
            return 1
        print(f"{target} replays ({len(capture.record['values'])} pins)")
        return 0
    target.write_text(text)
    cert = capture.record["certification"]
    print(
        f"wrote {target} ({len(capture.record['values'])} pins, "
        f"{len(cert['hook_names'])} hooks, max activation diff "
        f"{cert['max_activation_diff']['max']!r}, max logit diff "
        f"{cert['max_logit_diff']['max']!r})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
