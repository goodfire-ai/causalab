"""The task-driven end-to-end IIA pin: task package → serialized table →
interchange document → one IIA number per layer, through the real CLI.

This is the anchor the protocol refactor retired with `test_walking_skeleton.py`:
the only test
that runs the whole chain a real analysis runs — a task's own causal model and
counterfactual generator, its answer declaration, position resolution against
real prompts, a cross-model swap, and scoring — rather than hand-written
fixture text. What was missing was the dataset-serialization seam; the table
under test is built by ``scripts/build_task_dataset.py`` and committed
(``tests/protocol/fixtures/data/weekdays/task_n4_s0.json``).

**The pins are captured fresh, not carried over.** The retired test scored
generated *strings* through ``task.checker``; a document scores an argmax
against the answer's declared forms, and the retired coherent-model half used
a chat template v1 cannot express. So the old numbers are a cross-check on the
mechanism, not a target: what is pinned here is this stack's own
tiny-random-CPU behaviour, with a semantic guard (the patch must move the
logits) so an inert interchange cannot pass by matching zeros — the same guard
the retired test carried as ``logits_delta``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from safetensors.torch import load_file

from causalab.neural.engines.pytorch_hooks.loading import load_model
from causalab.protocol.answers import column_first_token_id
from causalab.cli import main

from tests.neural.engines.pytorch_hooks.conftest import TINY_LLAMA
from tests.protocol._env import CORPUS_DIR, FIXTURES
from tests.tables import frame as table_frame

pytestmark = pytest.mark.numerical_unit

DOCUMENT = "10_task_table_iia_im.json"
TABLE_REF = "weekdays/task_n4_s0"
REPO_ROOT = Path(__file__).resolve().parents[4]

#: Per patched layer on tiny-random (CPU, fp32), captured 2026-08-20:
#: ``iia`` per example, the argmax token the patched model puts at the answer
#: position, and the largest absolute logit change the swap causes.
#:
#: IIA zero everywhere is the *expected* shape for a randomly-initialized
#: model — it answers nothing correctly, patched or not, and the retired
#: walking-skeleton pins were flat at tiny scale for the same reason. The
#: numbers that actually discriminate are the argmax and the logit delta: a
#: silently-inert interchange would keep the delta at zero, and a change
#: anywhere in the chain (table bytes, position resolution, swap semantics,
#: first-token grading) moves one of the three.
PINS = {
    0: {"iia": [0.0, 0.0, 0.0, 0.0], "argmax": 21027, "logit_delta": 0.029653},
    1: {"iia": [0.0, 0.0, 0.0, 0.0], "argmax": 21027, "logit_delta": 0.036646},
}

#: The subject read's per-row window widths — the point of a ``column``
#: position: each row's window comes from *its own* ``entity`` value
#: (``Sunday``/``Monday`` are one sentencepiece piece, ``Thursday`` is three),
#: so a flat or role-wide resolution could not produce this.
PINNED_SUBJECT_WIDTHS = [1, 1, 3, 3]


@pytest.fixture(scope="module")
def tokenizer():
    return load_model(TINY_LLAMA).tokenizer


def _run(out: Path, layer: int) -> int:
    return main(
        [
            "run",
            "--engine",
            "auto",
            str(CORPUS_DIR / DOCUMENT),
            "--data-root",
            str(FIXTURES / "data"),
            "--artifacts-root",
            str(out),
            "--out",
            str(out),
            "--set",
            f"model.key={TINY_LLAMA}",
            "--set",
            "model.dtype=fp32",
            "--set",
            f"sites.target.layers={layer}",
        ]
    )


@pytest.mark.parametrize("layer", sorted(PINS))
def test_task_table_interchange_iia_matches_pins(tmp_path, layer):
    assert _run(tmp_path, layer) == 0
    pins = PINS[layer]

    iia = table_frame(tmp_path / "iia.json")
    assert list(iia["value"]) == pins["iia"]

    patched = load_file(str(tmp_path / "logits.safetensors"))["logits"]
    clean = load_file(str(tmp_path / "logits_clean.safetensors"))["logits_clean"]
    assert patched.shape == (4, 1, 32000)
    assert patched.squeeze(1).argmax(-1).tolist() == [pins["argmax"]] * 4

    # Not inert: the swap has to *change* the answer-position logits, or the
    # flat IIA above would be a pin on a no-op (the retired test's own guard,
    # which it kept for exactly this reason).
    delta = float((patched - clean).abs().max())
    assert delta > 0.0
    assert delta == pytest.approx(pins["logit_delta"], rel=1e-3)


def test_column_position_resolves_per_row(tmp_path):
    """The Q13 mechanism, end to end: ``{"column": "entity"}`` reads each row's
    own subject span, so the harvest is ragged with per-row widths."""
    assert _run(tmp_path, 0) == 0
    subject = load_file(str(tmp_path / "subject_acts.safetensors"))
    widths = subject["subject_acts.widths"]
    assert widths.tolist() == PINNED_SUBJECT_WIDTHS
    assert subject["subject_acts"].shape[0] == sum(PINNED_SUBJECT_WIDTHS)


def test_answer_space_is_first_token_distinct(tokenizer):
    """The honesty guard on ``first_token`` grading (§2.10): it credits a
    prefix, so it only means "the model answered" when the answer space's
    first tokens are distinct. Weekdays are, under the tokenizer this pin
    runs on — assert it rather than assume it."""
    rows = json.loads((FIXTURES / "data" / f"{TABLE_REF}.json").read_text())
    answers = {form for row in rows for form in row["label_forms"]}
    by_first: dict[int, set[str]] = {}
    for answer in answers:
        by_first.setdefault(column_first_token_id(tokenizer, answer), set()).add(
            answer.strip()
        )
    collisions = {k: v for k, v in by_first.items() if len(v) > 1}
    assert not collisions, f"answers share a first token: {collisions}"


#: The command line that built the committed fixture table — the recipe lives
#: here, beside the test that rebuilds it, not in a file beside the table
#: (spec §2.2: nothing sits beside a table).
TABLE_RECIPE = {
    "task": "natural_domains_arithmetic",
    "n": 4,
    "seed": 0,
    "split": "all",
    "target_variables": ["result"],
    "task_cfg": {"domain_type": "weekdays"},
}


def test_committed_table_is_reproducible_from_its_recipe():
    """The table is a *build product*: rebuilding it from the command line
    that built it has to give back the same bytes — that is what makes the
    content digest in the document's canonical form (§7) meaningful."""
    argv = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "build_task_dataset.py"),
        "--task",
        TABLE_RECIPE["task"],
        "--n",
        str(TABLE_RECIPE["n"]),
        "--seed",
        str(TABLE_RECIPE["seed"]),
        "--split",
        TABLE_RECIPE["split"],
        "--out",
        str(FIXTURES / "data" / f"{TABLE_REF}.json"),
        "--check",
    ]
    for variable in TABLE_RECIPE["target_variables"]:
        argv += ["--target-variable", variable]
    for key, value in TABLE_RECIPE["task_cfg"].items():
        argv += ["--set", f"{key}={value}"]
    result = subprocess.run(argv, capture_output=True, text=True, cwd=REPO_ROOT)
    assert result.returncode == 0, result.stdout + result.stderr
