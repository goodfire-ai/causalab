"""Capture (or refresh) the chat-coherent drift pins on the canonical GPU.

    uv run python tests/golden/drift/update_drift_goldens.py \\
        --device cuda --i-have-reviewed-the-diff

Runs both drift documents through the real CLI, extracts values with the
same code path the replay test uses (tests/golden/drift/_extract.py),
prints a per-key diff against the current pins, and refuses to write
without the review flag — the idiom inherited from the retired tier's
update_goldens.py and tests/protocol/update_corpus_digests.py. The
baseline-accuracy gate (>= 0.9) must hold at capture; a capture that
fails it is not a pin candidate but a task/model regression to
investigate.

Record which device the pins were captured on: the replay compares
against tolerance, and cross-device bf16 numerics (cuda vs mps) are not
guaranteed to sit inside it — the canonical capture device is cuda.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

# repo root (NOT tests/ — putting tests/ itself on sys.path would shadow the
# top-level `tasks` package with tests/tasks and break task resolution)
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tests.golden.drift._extract import (  # noqa: E402
    ACCURACY_GATE,
    DOCS,
    PINS,
    extract_values,
    load_pins,
    run_drift_documents,
)


def _document_dtype() -> str:
    """The dtype the drift documents declare — they agree, and the pins
    record it as the precision the values were measured at."""
    from causalab.io.sources import load_text
    from causalab.protocol.schema import MODEL_DTYPE_DEFAULT

    from tests.golden._env import GOLDEN_PROTOCOLS

    dtypes = {
        load_text(GOLDEN_PROTOCOLS / name)["model"].get("dtype", MODEL_DTYPE_DEFAULT)
        for name in DOCS
    }
    if len(dtypes) != 1:
        raise SystemExit(f"the drift documents disagree on dtype: {sorted(dtypes)}")
    return dtypes.pop()


def non_finite_keys(values: Mapping[str, Any]) -> list[str]:
    """The keys whose value is not finite — a coordinate with every row
    excluded means a ``NaN`` mean. A list value is the shape pin
    (``interchange.acts_mid.shape``): its elements are checked only when
    they are floats, never a str. One census for the gate path and for
    ``serialise_pins`` so the two cannot drift."""
    bad: list[str] = []
    for key, value in values.items():
        scalars = value if isinstance(value, list) else [value]
        if any(isinstance(v, float) and not math.isfinite(v) for v in scalars):
            bad.append(key)
    return sorted(bad)


def gate_refuses(acc: float) -> bool:
    """The baseline-accuracy gate. A non-finite accuracy — every example
    excluded — is refused like one below the gate: ``nan < gate`` is False
    and would pass it."""
    return not math.isfinite(acc) or acc < ACCURACY_GATE


def serialise_pins(pins: dict[str, Any]) -> str:
    """The pins file's bytes, or a ``ValueError`` naming every non-finite
    value in ``pins["values"]`` — pure, so test_extract_labels.py calls it
    on the CPU instead of grepping this file's source."""
    bad = non_finite_keys(pins["values"])
    if bad:
        raise ValueError(
            f"non-finite pin value(s): {', '.join(bad)} — a coordinate with "
            "every row excluded; refusing to write"
        )
    # a NaN value — a coordinate with every row excluded — is refused at
    # capture rather than written as a token no other JSON parser accepts
    # (tables.py); compare() would name it on replay, the pin never gets it
    return json.dumps(pins, indent=2, allow_nan=False) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", required=True)
    parser.add_argument("--i-have-reviewed-the-diff", action="store_true")
    args = parser.parse_args()

    dirs = run_drift_documents(Path(tempfile.mkdtemp()), args.device)
    values = extract_values(dirs)

    acc = values["interchange.acc.mean"]
    if gate_refuses(acc):
        print(f"REFUSED: baseline accuracy {acc:.4f} < gate {ACCURACY_GATE}")
        return 1
    # before the key diff, so the refusal names the coordinate rather than
    # dying in json.dumps after the wall of - / + lines
    bad = non_finite_keys(values)
    if bad:
        print(
            f"REFUSED: non-finite value(s) {bad} — a coordinate with every row excluded"
        )
        return 1

    pins = load_pins(PINS)
    old = pins.get("values") or {}
    for key in sorted(set(old) | set(values)):
        before, after = old.get(key), values.get(key)
        if before != after:
            print(f"  - {key}: {before}")
            print(f"  + {key}: {after}")

    if not args.i_have_reviewed_the_diff:
        print("dry run: pass --i-have-reviewed-the-diff to write the pins")
        return 1

    pins["values"] = values
    # precision is the documents' own (§2.1) — only placement is the
    # capture's, and only placement can differ run to run
    pins["capture"] = {"device": args.device, "dtype": _document_dtype()}
    PINS.write_text(serialise_pins(pins))
    print(f"wrote {PINS} ({len(values)} keys, acc {acc:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
