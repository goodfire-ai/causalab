"""The behavioral runner on the tiny fixture (``docs/workflow_protocol.md`` §2.7; T5–T7).

The declarative half — parsing, rule 17, the censuses, the identity facts —
is ``tests/workflow/test_behavioral.py``. This file is the run: one
``causalab run`` of a one-step workflow over a no-intervention document,
with no experiment-specific generation or scoring code anywhere (T5); the
four terminal outcomes from four forced-logit rows, each asserted on its own
(T6); and the decode's reproducibility contract — deterministic twice is
byte-identical, sampled under one seed twice is byte-identical, two seeds
differ, and both records say which seed ran (T7). The last test is the
``golden`` canary for batch geometry: sampled decoding on a real model under one
seed and one ``--batch-rows`` is byte-identical, and a differing geometry is
*recorded*, never gated.

The forced-logit pattern is ``test_generate_frame.py``'s: the bundle's
``forward`` is wrapped so that each decode step's last-position logits put
all their mass on one chosen token per row, so the random-weight model says
exactly what the outcome under test needs it to say.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any

import pytest

from causalab.cli import main
from causalab.neural.engines.pytorch_hooks import PytorchHooksEngine
from causalab.neural.engines.pytorch_hooks.loading import load_model
from causalab.protocol.engine import CONTINUATIONS_FILE
from causalab.io.env import FileArtifacts, FileDatasets, ResolutionEnv
from causalab.io.tables import read_table
from causalab.workflow import run_workflow
from causalab.workflow.behavioral import DECISION_FILE, OUTCOMES, OUTCOMES_FILE
from causalab.workflow.document import BehavioralStep, WorkflowError, load_workflow
from tests.neural.engines.pytorch_hooks.conftest import TINY_LLAMA

REPO = Path(__file__).resolve().parents[4]
FIXTURES = REPO / "tests" / "workflow" / "fixtures" / "behavioral"
STEP = "qualify"
OUTPUT_DIR = "qualify_weekdays"
#: the development split's rows, in table order: (entity, the answer)
DEVELOPMENT = (
    ("Thursday", "Friday"),
    ("Monday", "Tuesday"),
    ("Saturday", "Sunday"),
    ("Tuesday", "Wednesday"),
)
BUDGET = 4  # qa_probe.json's max_new_tokens
#: the fixture document's own `save` files
SAVES = ("per_step.json", "said.json", "said_at.json")
CHECKER = {
    "task": "natural_domains_arithmetic",
    "task_cfg": {"domain_type": "weekdays"},
}


def _tree(tmp: Path) -> Path:
    """A private copy of the fixture tree: the workflow, its document and its
    table, so a test may rewrite the workflow without touching the fixture."""
    root = tmp / "behavioral"
    shutil.copytree(FIXTURES, root)
    return root


def _workflow(root: Path, **step_changes: Any) -> Path:
    """The fixture workflow with ``step_changes`` applied to its one step
    (a ``None`` value deletes the key), written beside the fixture copy."""
    raw = json.loads((root / "qualify.json").read_text())
    step = raw["steps"][STEP]
    for key, value in step_changes.items():
        if value is None:
            step.pop(key, None)
        else:
            step[key] = value
    target = root / "variant.json"
    target.write_text(json.dumps(raw, indent=2))
    return target


def _env(root: Path) -> ResolutionEnv:
    return ResolutionEnv(
        datasets=FileDatasets(root=root), artifacts=FileArtifacts(root=root)
    )


def _run_cli(root: Path, workflow: Path, out: Path, *extra: str) -> int:
    return main(
        [
            "run",
            "--engine",
            "auto",
            str(workflow),
            "--data-root",
            str(root),
            "--artifacts-root",
            str(root),
            "--out",
            str(out),
            *extra,
        ]
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- #
# T5 — one document, one command, no experiment code
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def t5_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """One CLI run of the fixture workflow; the T5 assertions share it."""
    base = tmp_path_factory.mktemp("behavioral-t5")
    root = _tree(base)
    out = base / "runs"
    assert _run_cli(root, root / "qualify.json", out) == 0
    return out / OUTPUT_DIR, root


@pytest.mark.smoke
def test_t5_one_command_publishes_the_step(t5_run: tuple[Path, Path]) -> None:
    """The step's outputs are the document's own `save` files plus the three
    behavioral files, each verified and content-digested into the record."""
    out, _root = t5_run
    step = out / STEP
    for name in ("_step.json", CONTINUATIONS_FILE, OUTCOMES_FILE, DECISION_FILE):
        assert (step / name).is_file(), name
    for name in SAVES:  # the document's own saves
        assert (step / name).is_file(), name
    record = json.loads((step / "_step.json").read_text())
    assert record["type"] == "behavioral" and record["status"] == "completed"
    assert set(record["files"]) == {
        *SAVES,
        CONTINUATIONS_FILE,
        OUTCOMES_FILE,
        DECISION_FILE,
    }
    assert set(record["digests"]) == set(record["files"])
    manifest = json.loads((out / "workflow.json").read_text())
    assert manifest["steps"][STEP]["status"] == "completed"


@pytest.mark.smoke
def test_t5_no_experiment_code_anywhere(t5_run: tuple[Path, Path]) -> None:
    """A free-form generation task executes without experiment-specific
    generation or scoring code: the workflow has one step and it is
    declarative; the fixture tree holds no Python at all."""
    _out, root = t5_run
    loaded = load_workflow(root / "qualify.json", _env(root))
    assert all(isinstance(s, BehavioralStep) for s in loaded.document.steps.values())
    assert not list(root.rglob("*.py"))


@pytest.mark.smoke
def test_t5_the_record_carries_the_behavioral_facts(t5_run: tuple[Path, Path]) -> None:
    out, _root = t5_run
    sidecar = out / STEP / "_step.json"
    record = json.loads(sidecar.read_text())
    # the one execution block: the engine's geometry and source through the
    # one recorder, the decode spec beside them (§8)
    assert record["execution"]["model_source"] == "loaded"
    assert record["execution"]["batch_rows"] is None
    assert record["execution"]["device"] == "cpu"  # the CLI's default placement
    assert record["execution"]["decoding"] == {"mode": "deterministic"}
    # the commit the loader resolved `main` to, beside the execution block
    (model,) = record["models"]
    assert (model["key"], model["revision"]) == (TINY_LLAMA, "main")
    assert re.fullmatch(r"[0-9a-f]{40}", model["resolved_revision"])
    # the checker's string mode is read from the spec, never authored
    assert record["checker"]["string_mode"] == "exact"
    assert record["checker"]["task"] == "natural_domains_arithmetic"
    assert record["split"] == "development"
    counts = record["outcomes"]
    assert counts["n"] == len(DEVELOPMENT)
    assert sum(counts[outcome] for outcome in OUTCOMES) == counts["n"]
    assert counts["correct"] <= counts["valid"]
    # a cohort below the default minimum size is recorded and warned about,
    # not refused
    assert record["cohort"] == {
        "default_min_examples": 10_000,
        "n": len(DEVELOPMENT),
        "below_default": True,
    }
    # retention: the fixture keeps two of the four raw generations
    assert record["retain"] == {
        "generations": {"max_rows": 2},
        "retained": 2,
        "n": len(DEVELOPMENT),
    }
    assert len(read_table(out / STEP / CONTINUATIONS_FILE)) == 2
    assert record["decision"] == DECISION_FILE


@pytest.mark.smoke
def test_t5_the_decision_record_has_the_six_fields(t5_run: tuple[Path, Path]) -> None:
    """The `DecisionRecord`, produced here: six fields plus `split` and
    `step`; the rule is the thresholds block verbatim, the outcome follows
    from counts over n, and the evidence identity ties the decision to the
    outcomes table's bytes and the step identity."""
    out, _root = t5_run
    step = out / STEP
    record = json.loads((step / "_step.json").read_text())
    decision = json.loads((step / DECISION_FILE).read_text())
    assert set(decision) == {
        "decision_type",
        "schema_version",
        "measured_inputs",
        "rule",
        "outcome",
        "evidence_identity",
        "split",
        "step",
    }
    assert decision["schema_version"] == 1
    assert decision["rule"] == record["thresholds"]
    assert decision["split"] == "development" and decision["step"] == STEP
    n = decision["measured_inputs"]["n"]
    assert n == len(DEVELOPMENT)
    assert decision["measured_inputs"]["counts"] == {
        **{outcome: record["outcomes"][outcome] for outcome in OUTCOMES},
        "correct": record["outcomes"]["correct"],
    }
    assert decision["measured_inputs"]["valid_rate"] == record["outcomes"]["valid"] / n
    # the fixture's thresholds hold on any four rows: min_examples 4, rates 0
    assert decision["outcome"] == "pass" and decision["decision_type"] == "advance"
    outcomes_sha, identity = decision["evidence_identity"].split(":")
    assert outcomes_sha == _sha256(step / OUTCOMES_FILE)
    assert identity == record["identity"]


@pytest.mark.smoke
def test_t5_mutation_a_checker_less_workflow_fails_at_load_not_at_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Delete the checker binding: the workflow does not load (rule 17, the
    path naming `checker`), and the reference engine's loader — replaced by
    one that raises — is never entered (`test_legality_before_weights.py`'s
    never-called-loader assertion)."""
    import causalab.neural.engines.pytorch_hooks.engine as engine_module

    def never(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("load_model was entered — the refusal came too late")

    monkeypatch.setattr(engine_module, "load_model", never)
    root = _tree(tmp_path)
    workflow = _workflow(root, checker=None)
    with pytest.raises(WorkflowError) as err:
        load_workflow(workflow, _env(root))
    assert err.value.rule == 17 and "checker" in str(err.value)
    assert _run_cli(root, workflow, tmp_path / "runs") == 1
    assert "W17" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# T6 — the four terminal outcomes, one forced row each
# --------------------------------------------------------------------------- #


def _ids(tokenizer: Any, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)


@pytest.mark.smoke
def test_t6_the_four_outcomes_from_four_forced_rows(tmp_path: Path) -> None:
    """Row 0 says "maybe Friday" — the answer appears but the string is not a
    declared form under `exact`: `invalid_format`. Row 1 never emits EOS
    inside the budget: `truncated`, whatever it said. Row 2 says "Sunday" and
    stops: `valid`, graded `correct`. Row 3 says "maybe" and stops — no
    answer anywhere: `no_final_answer`. Collapsing `truncated` into
    `invalid_format` (the mutation) makes row 1 read `invalid_format`."""
    bundle = load_model(TINY_LLAMA)
    tokenizer = bundle.tokenizer
    eos = int(tokenizer.eos_token_id)
    friday, sunday = _ids(tokenizer, " Friday"), _ids(tokenizer, " Sunday")
    assert len(friday) == 1 and len(sunday) == 1  # single pieces on this tokenizer
    (maybe,) = _ids(tokenizer, " maybe")
    (the,) = _ids(tokenizer, " the")
    plan: list[list[int]] = [
        [maybe, friday[0], eos, eos],  # "maybe Friday" then EOS → invalid_format
        [the, the, the, the],  # the whole budget, no EOS → truncated
        [sunday[0], eos, eos, eos],  # "Sunday" then EOS → valid, correct
        [maybe, eos, eos, eos],  # "maybe" then EOS → no_final_answer
    ]
    assert len(plan) == len(DEVELOPMENT)
    original = bundle.model.forward
    calls = {"n": 0}

    def forced(*args: Any, **kwargs: Any) -> Any:
        out = original(*args, **kwargs)
        step = calls["n"]  # 0 is the prefill; each later call is one decode step
        calls["n"] += 1
        for row, tokens in enumerate(plan):
            if step < len(tokens):
                out.logits[row, -1, :] = -1e4
                out.logits[row, -1, tokens[step]] = 1e4
        return out

    root = _tree(tmp_path)
    workflow = _workflow(
        root, decoding={"mode": "deterministic", "eos_token_ids": [eos]}
    )
    loaded = load_workflow(workflow, _env(root))
    engine = PytorchHooksEngine(bundle=bundle)  # caller-owned, so the wrap is seen
    bundle.model.forward = forced  # type: ignore[method-assign]
    try:
        run_workflow(loaded, _env(root), tmp_path / "runs", engine)
    finally:
        bundle.model.forward = original  # type: ignore[method-assign]
    step_dir = tmp_path / "runs" / OUTPUT_DIR / STEP
    rows = {row["example_id"]: row for row in read_table(step_dir / OUTCOMES_FILE)}
    assert rows["0"]["outcome"] == "invalid_format" and rows["0"]["grade"] is None
    assert rows["1"]["outcome"] == "truncated" and rows["1"]["width"] == BUDGET
    assert rows["2"]["outcome"] == "valid" and rows["2"]["grade"] == "correct"
    assert rows["3"]["outcome"] == "no_final_answer" and rows["3"]["grade"] is None
    record = json.loads((step_dir / "_step.json").read_text())
    assert record["outcomes"] == {
        "n": 4,
        "truncated": 1,
        "no_final_answer": 1,
        "invalid_format": 1,
        "valid": 1,
        "correct": 1,
    }
    assert record["execution"]["model_source"] == "caller"
    # what the model said is on the record, raw: the fixture retains two rows
    said = {row["example_id"]: row for row in read_table(step_dir / CONTINUATIONS_FILE)}
    assert said["0"]["text"] == "maybe Friday" and said["0"]["truncated"] is False
    assert said["1"]["truncated"] is True and said["1"]["width"] == BUDGET


# --------------------------------------------------------------------------- #
# T7 — same seed, same bytes; different seeds differ; the seed is recorded
# --------------------------------------------------------------------------- #


def _run_under(
    root: Path, out: Path, decoding: dict[str, Any], engine: PytorchHooksEngine
) -> tuple[str, dict[str, Any]]:
    """Run the fixture under ``decoding`` into ``out``: the sha256 of the
    continuation file and the step record."""
    workflow = _workflow(root, decoding=decoding, retain={"generations": "all"})
    loaded = load_workflow(workflow, _env(root))
    run_workflow(loaded, _env(root), out, engine)
    step_dir = out / OUTPUT_DIR / STEP
    record = json.loads((step_dir / "_step.json").read_text())
    return _sha256(step_dir / CONTINUATIONS_FILE), record


@pytest.mark.smoke
def test_t7_deterministic_twice_and_sampled_under_one_seed_are_byte_identical(
    tmp_path: Path,
) -> None:
    root = _tree(tmp_path)
    engine = PytorchHooksEngine()
    plain_a, _ = _run_under(root, tmp_path / "a", {"mode": "deterministic"}, engine)
    plain_b, _ = _run_under(root, tmp_path / "b", {"mode": "deterministic"}, engine)
    assert plain_a == plain_b
    seven = {"mode": "sampled", "seed": 7}
    seven_a, record_a = _run_under(root, tmp_path / "c", seven, engine)
    seven_b, record_b = _run_under(root, tmp_path / "d", seven, engine)
    assert seven_a == seven_b
    eight_a, record_c = _run_under(
        root, tmp_path / "e", {"mode": "sampled", "seed": 8}, engine
    )
    assert eight_a != seven_a, "two seeds drew the same continuation"
    # both runs record their seed — without it the third case could not be
    # told from a bug (the mutation: drop the seed from the record)
    for record, seed in ((record_a, 7), (record_b, 7), (record_c, 8)):
        assert record["execution"]["decoding"] == {
            "mode": "sampled",
            "seed": seed,
            "temperature": 1.0,
            "top_p": 1.0,
        }
        # the batch geometry is recorded beside the decode spec
        assert "batch_rows" in record["execution"]
    # a sampled step is another step: its identity carries the seed (§7)
    assert record_a["identity"] == record_b["identity"] != record_c["identity"]


@pytest.mark.smoke
def test_t7_a_sampled_decode_is_not_the_greedy_one(tmp_path: Path) -> None:
    """Sanity on the sampler: at temperature 1 over a random-weight vocabulary
    the draw differs from the argmax, so `sampled` really samples."""
    root = _tree(tmp_path)
    engine = PytorchHooksEngine()
    plain, _ = _run_under(root, tmp_path / "a", {"mode": "deterministic"}, engine)
    sampled, _ = _run_under(
        root, tmp_path / "b", {"mode": "sampled", "seed": 7}, engine
    )
    assert plain != sampled


# --------------------------------------------------------------------------- #
# the golden canary — written here, run on a CUDA host
# --------------------------------------------------------------------------- #

CANARY_MODEL = "Qwen/Qwen3-4B"


@pytest.mark.golden
def test_golden_same_seed_same_geometry_is_byte_identical_on_a_real_model(
    tmp_path: Path,
) -> None:
    """On a real model, sampled decoding under one seed and one `--batch-rows`
    is byte-identical across two runs — the claim T7 makes on the CPU
    fixture, on the accelerator. A third run under another geometry is
    **recorded** in `canary.json` beside the runs, never asserted: sampled
    decoding is not bit-reproducible across batch geometries on a GPU
    (batch-dependent kernels can flip the top-1 token), and that is a fact a reader of two receipts
    sees, not something a run refuses over.

    Run as: ``uv run pytest -m golden
    tests/neural/engines/pytorch_hooks/test_behavioral_run.py -q`` on a CUDA
    host with the model cached (``HF_HUB_OFFLINE=1``)."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("the canary needs CUDA")
    root = _tree(tmp_path)
    workflow = _workflow(
        root,
        set={
            "data.base.dataset": "qa/data#development",
            "model.key": CANARY_MODEL,
            "model.dtype": "bf16",
        },
        decoding={"mode": "sampled", "seed": 7, "temperature": 1.0, "top_p": 1.0},
        retain={"generations": "all"},
    )

    def run(name: str, batch_rows: int) -> tuple[str, dict[str, Any]]:
        out = tmp_path / name
        code = _run_cli(
            root, workflow, out, "--device", "cuda", "--batch-rows", str(batch_rows)
        )
        assert code == 0
        step_dir = out / OUTPUT_DIR / STEP
        record = json.loads((step_dir / "_step.json").read_text())
        return _sha256(step_dir / CONTINUATIONS_FILE), record

    same_a, record_a = run("same_a", 4)
    same_b, record_b = run("same_b", 4)
    assert same_a == same_b, "same seed, same geometry, different bytes"
    assert (
        record_a["execution"]["batch_rows"] == record_b["execution"]["batch_rows"] == 4
    )
    assert record_a["execution"]["decoding"]["seed"] == 7
    other, record_c = run("other_geometry", 1)
    canary = {
        "model": CANARY_MODEL,
        "seed": 7,
        "same_geometry_identical": same_a == same_b,
        "other_geometry_identical": other == same_a,
        "batch_rows": {"same": 4, "other": record_c["execution"]["batch_rows"]},
        "recorded_not_gated": True,
    }
    (tmp_path / "canary.json").write_text(json.dumps(canary, indent=2) + "\n")
    print(json.dumps(canary))
