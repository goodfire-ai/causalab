"""Chat-coherent drift tier: replay the pinned Qwen3-4B runs (GPU).

The counterpart of tests/golden/test_paper_goldens.py with the opposite
provenance: these values ARE pinned from a reviewed run of this stack
(tests/golden/drift/update_drift_goldens.py on the canonical cuda box) —
their job is run-to-run drift detection on a real model, the role the
retired tests/end_to_end golden tier held. Until the first capture lands,
the replay skips ("pins not yet captured").

Run: uv run pytest tests/golden/drift -m golden   (needs cuda; bf16)
"""

from __future__ import annotations

import json

import pytest
import torch

from tests.golden.drift._extract import (
    DOCS,
    GOLDEN_PROTOCOLS,
    ACCURACY_GATE,
    PINS,
    compare,
    extract_values,
    load_pins,
    run_drift_documents,
)

pytestmark = pytest.mark.golden


def test_drift_pins_replay(tmp_path):
    pins = load_pins(PINS)
    if not pins.get("values"):
        pytest.skip("drift pins not yet captured (see update_drift_goldens.py)")
    capture = pins.get("capture") or {}
    device = capture.get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("pins were captured on cuda; no cuda available")

    # precision is the document's own (§2.1): the CLI takes no
    # --dtype, so the pins' captured dtype is checked against what each drift
    # document declares rather than passed in
    dtype = capture.get("dtype", "bf16")
    for name in DOCS:
        declared = json.loads((GOLDEN_PROTOCOLS / name).read_text())["model"]["dtype"]
        assert declared == dtype, (name, declared, dtype)
    dirs = run_drift_documents(tmp_path, device)
    values = extract_values(dirs)

    assert values["interchange.acc.mean"] >= ACCURACY_GATE
    problems = compare(pins["values"], values, pins["tolerance"])
    assert not problems, "drift vs pins:\n" + "\n".join(problems)
