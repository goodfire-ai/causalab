"""``--tokenizer`` on ``validate`` and ``dry-run``: the tokenizer's half of
the run door's checks, before any weights.

``run`` resolves a document's positions and its metrics' answers with the
model's tokenizer before the weights load (``pipeline.resolve_positions``,
``pipeline.resolve_answers``). The pure verbs load no tokenizer, so a
column value the tokenizer splits passed ``validate`` and ``dry-run`` and
was refused only by ``run``. With ``--tokenizer`` both verbs load the
tokenizer (never the weights) and run the same two passes. Without the flag
they are what they were: ``validate`` prints OK and ``dry-run`` leaves
``tokenization`` undecided.

On the base the flag does not exist, and argparse exits 2.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from causalab.cli import main

pytestmark = pytest.mark.unit

#: Two IOI prompts. Under the gpt2 tokenizer ``' Tiffany'`` is one token and
#: ``'Tiffany'`` is three; ``'Jennifer'`` and ``' Jennifer'`` are one each.
PROMPTS = (
    "Then, Jennifer and Kevin went to the store. Kevin gave a drink to",
    "Then, Tiffany and Sean went to the store. Sean gave a drink to",
)
TABLES: dict[str, tuple[str, str]] = {
    "spaced": (" Jennifer", " Tiffany"),
    "split": (" Jennifer", "Tiffany"),
    "glued": ("Jennifer", " Tiffany"),
}


def _document(table: str) -> dict[str, Any]:
    """The un-intervened gpt2 on ``names/<table>``, its last-position logits
    reduced to ``logit_diff`` of the ``io`` and ``s`` columns."""
    return {
        "header": {"protocol_version": "4"},
        "model": {"key": "gpt2", "revision": "main"},
        "data": {"base": {"dataset": f"names/{table}", "field": "input"}},
        "method": {
            "intervened_models": {
                "original_base": {"input": "base", "reads": ["logits"]}
            },
            "sites": {"lm_head": {"component": "lm_head"}},
            "reads": {"logits": {"site": "lm_head", "pos": -1}},
            "save": [
                {
                    "read": "logits",
                    "model": "original_base",
                    "aggregation": {"kind": "logit_diff", "a": "io", "b": "s"},
                    "file_path": "ld.json",
                }
            ],
        },
    }


@pytest.fixture()
def root(tmp_path: Path) -> Path:
    """Every table under ``data/names/``, one document per table, and a
    workflow running the spaced document and then the split one."""
    names = tmp_path / "data" / "names"
    names.mkdir(parents=True)
    for table, io in TABLES.items():
        rows = [
            {"input": prompt, "io": name, "s": " Kevin", "split": "all"}
            for prompt, name in zip(PROMPTS, io)
        ]
        (names / f"{table}.json").write_text(json.dumps(rows))
        (tmp_path / f"{table}.json").write_text(json.dumps(_document(table)))
    workflow = {
        "version": "1",
        "output_dir": "names",
        "steps": {
            "first": {"type": "intervention_protocol", "document": "spaced.json"},
            "second": {
                "type": "intervention_protocol",
                "document": "split.json",
                "after": ["first"],
            },
        },
    }
    (tmp_path / "workflow.json").write_text(json.dumps(workflow))
    return tmp_path


def _argv(verb: str, root: Path, document: str, *extra: str) -> list[str]:
    return [
        verb,
        str(root / f"{document}.json"),
        "--data-root",
        str(root / "data"),
        "--artifacts-root",
        str(root),
        "--engine",
        "auto",
        *extra,
    ]


def test_validate_without_the_flag_loads_no_tokenizer(
    root: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The flag is opt-in: plain ``validate`` accepts the split table and
    never calls the tokenizer loader."""
    import causalab.io.tokenizer as tokenizer_module

    def never(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("tokenizer loaded")

    monkeypatch.setattr(tokenizer_module, "load_tokenizer", never)
    assert main(_argv("validate", root, "split")) == 0
    assert "OK" in capsys.readouterr().out


def test_validate_tokenizer_refuses_a_multi_token_answer(
    root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(_argv("validate", root, "split", "--tokenizer")) == 1
    err = capsys.readouterr().err
    assert "refused: [P2]" in err and "'Tiffany'" in err, err
    assert "metric logit_diff.a" in err, err


def test_validate_tokenizer_accepts_the_spaced_answers(
    root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(_argv("validate", root, "spaced", "--tokenizer")) == 0
    out = capsys.readouterr().out
    assert "OK" in out and "tokenizer gpt2@main" in out, out


def test_validate_tokenizer_checks_every_inner_document_of_a_workflow(
    root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A workflow's static inner documents are checked in step order; the
    second step's split table is refused, naming the step."""
    argv = _argv("validate", root, "workflow", "--tokenizer")
    assert main(argv) == 1
    err = capsys.readouterr().err
    assert "refused: [P2]" in err and "steps.second" in err and "'Tiffany'" in err


def test_validate_tokenizer_warns_on_a_glued_answer(
    root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``'Jennifer'`` after a prompt ending in ``to`` is one gpt2 token, so
    nothing refuses it, but the model emits ``' Jennifer'`` there: the verb
    warns and still exits 0."""
    with pytest.warns(Warning, match="Jennifer"):
        assert main(_argv("validate", root, "glued", "--tokenizer")) == 0


def test_a_tokenizer_that_cannot_load_is_refused_by_name(
    root: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tokenizer outside the cache offline, or a gated one without a
    token, is refused with the key, the revision and what to do, and exit 1
    rather than a traceback."""
    import causalab.io.tokenizer as tokenizer_module

    def offline(key: str, revision: str = "main") -> Any:
        raise OSError(
            "We couldn't connect to 'https://huggingface.co' to load the files, "
            "and couldn't find them in the cached files."
        )

    monkeypatch.setattr(tokenizer_module, "load_tokenizer", offline)
    for verb in ("validate", "dry-run"):
        assert main(_argv(verb, root, "spaced", "--tokenizer")) == 1
        err = capsys.readouterr().err
        assert "refused: [P4]" in err, err
        assert "gpt2@main" in err and "tokenizer" in err and "cached" in err, err


def test_dry_run_tokenizer_decides_the_tokenization(
    root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """With the flag, ``tokenization`` leaves the ``undecided`` line and the
    report says what the tokenizer decided; without it, the line names it."""
    assert main(_argv("dry-run", root, "spaced")) == 0
    plain = capsys.readouterr().out
    assert "tokenization" in plain.strip().splitlines()[-1]
    assert main(_argv("dry-run", root, "spaced", "--tokenizer")) == 0
    decided = capsys.readouterr().out
    assert "tokenization" not in decided.strip().splitlines()[-1], decided
    assert "tokenizer gpt2@main" in decided, decided


def test_dry_run_tokenizer_reports_a_multi_token_answer(
    root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(_argv("dry-run", root, "split", "--tokenizer")) == 1
    captured = capsys.readouterr()
    assert "refused: [P2]" in captured.err and "'Tiffany'" in captured.err


def test_validate_tokenizer_lists_every_step_that_shares_a_document(
    root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Two steps run ``spaced.json``: the document resolves once, and both
    steps are in the list of steps that resolve."""
    workflow = {
        "version": "1",
        "output_dir": "names",
        "steps": {
            "first": {"type": "intervention_protocol", "document": "spaced.json"},
            "same": {
                "type": "intervention_protocol",
                "document": "spaced.json",
                "after": ["first"],
            },
            "third": {
                "type": "intervention_protocol",
                "document": "glued.json",
                "after": ["same"],
            },
        },
    }
    (root / "shared.json").write_text(json.dumps(workflow))
    with pytest.warns(Warning, match="Jennifer"):
        assert main(_argv("validate", root, "shared", "--tokenizer")) == 0
    out = capsys.readouterr().out
    assert "resolve in ['first', 'same', 'third']" in out, out


def test_validate_tokenizer_says_when_no_metric_names_answers(
    root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``top_k`` names no answer: the line says so rather than listing an
    empty set of metrics."""
    doc = _document("split")
    doc["method"]["save"][0]["aggregation"] = {"kind": "top_k", "k": 2, "by": "value"}
    (root / "topk.json").write_text(json.dumps(doc))
    assert main(_argv("validate", root, "topk", "--tokenizer")) == 0
    out = capsys.readouterr().out
    assert "positions resolve; no metric names answers" in out, out


@pytest.mark.parametrize("verb", ["validate", "dry-run"])
def test_the_flag_help_names_workflows_only_where_the_verb_takes_one(
    verb: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """``dry-run`` refuses a workflow, so its help does not promise a
    workflow check; ``validate``'s does."""
    with pytest.raises(SystemExit):
        main([verb, "--help"])
    help_text = " ".join(capsys.readouterr().out.split())
    assert "never its weights" in help_text
    assert ("On a workflow" in help_text) is (verb == "validate"), help_text


@pytest.mark.parametrize("verb", ["validate", "dry-run", "workflow"])
def test_a_metric_over_generated_tokens_is_reported_as_checked_when_scored(
    verb: str, root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A read over the generated steps scores the rows that generated a
    step, which is known only after the decode, so the pass leaves its
    answers to the score. The split ``'Tiffany'`` is not resolved here, so
    the report lists the metric as checked when scored and not among the
    metrics whose answers resolve. On the base every verb listed it as
    resolved."""
    doc = _document("split")
    doc["method"]["positions"] = {
        "window": {"generated": {"max_new_tokens": 2}, "all": True}
    }
    doc["method"]["reads"]["logits"]["pos"] = "window"
    (root / "generated.json").write_text(json.dumps(doc))
    workflow = {
        "version": "1",
        "output_dir": "names",
        "steps": {
            "first": {"type": "intervention_protocol", "document": "spaced.json"},
            "decode": {
                "type": "intervention_protocol",
                "document": "generated.json",
                "after": ["first"],
            },
        },
    }
    (root / "decoding.json").write_text(json.dumps(workflow))
    argv = (
        _argv("validate", root, "decoding", "--tokenizer")
        if verb == "workflow"
        else _argv(verb, root, "generated", "--tokenizer")
    )
    assert main(argv) == 0
    line = next(
        line for line in capsys.readouterr().out.splitlines() if "tokenizer" in line
    )
    listed = "{'decode': ['ld']}" if verb == "workflow" else "['ld']"
    assert f"; the answers of {listed} are checked when scored" in line, line
    resolved = line.partition("; the answers of")[0]
    assert "'ld'" not in resolved, line
