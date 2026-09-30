"""Check workflow CLI refusals and every matching quote in tracked Markdown.

Quotes are found by a shared signature and checked against actual CLI stderr.
A quote may contain any contiguous part of the message.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from causalab.cli import main
from tests._helpers.tracked import tracked_files
from tests._helpers.paths import PROTOCOLS_DIR

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]

#: A shipped intervention specification — any document without a ``steps``
#: section reaches the `--resume` refusal, which fires before the compile.
LOCATE = PROTOCOLS_DIR / "weekdays_locate_scan.json"

#: The run of words a quote of the refusal is recognized by. The head of the
#: message (`docs/running_experiments.md`) and its tail (the demos) overlap on
#: exactly this, and it is asserted to be *in* the message too — so a rewording
#: that drops it fails here instead of letting the finder go quiet.
SIGNATURE = "each declare their own realization"

#: A quoted run in markdown: backticked, or italicised inside double quotes
#: (``*"…"*``), which is how the demos quote a message they are not showing as
#: a terminal line. Either may span a soft wrap but never a blank line: a
#: paragraph boundary ends a quote, so one stray backtick in a paragraph cannot
#: shift the pairing for the rest of the file.
QUOTED = re.compile(
    r"`((?:[^`\n]|\n(?![ \t]*\n))+)`" r"|" r"\*\"((?:[^\n]|\n(?![ \t]*\n))+?)\"\*"
)

#: A fenced code block. Its three backticks are an odd count, which would
#: pair the fence with the next inline run and swallow the paragraph after it —
#: and what a doc shows in a fence is a command or its output, which the
#: `docs/running_experiments.md` example quotes *outside* the fence.
FENCE = re.compile(r"```.*?```", re.S)

#: The line-lead marker of a blockquote. A wrapped quote repeats it on every
#: line, and the CLI never printed it.
BLOCKQUOTE = re.compile(r"^[ \t]*>[ \t]?", re.M)

#: Two demo pages quote the refusal (the demos README and the weekdays
#: geometry replication; the onboarding tutorial's quote left with its
#: rewrite). Keep a floor so a broken finder cannot pass by returning no
#: quotes.
QUOTING_FILES_FLOOR = 2


def _refusal(tmp_path: Path, capsys) -> str:
    """The `--dtype` refusal as the CLI actually prints it.

    A workflow document is recognized by its ``steps`` section alone
    (``is_workflow``), and the refusal is the first thing ``run`` does, so an
    empty one is enough to reach it without a model or a dataset.
    """
    document = tmp_path / "workflow.json"
    document.write_text(json.dumps({"version": "1", "steps": {}}))
    code = main(
        [
            "run",
            "--engine",
            "auto",
            str(document),
            "--out",
            str(tmp_path / "out"),
            "--dtype",
            "bf16",
        ]
    )
    assert code == 1, "--dtype on a workflow is refused (§9)"
    return capsys.readouterr().err.strip()


def _normalized(text: str) -> str:
    return " ".join(text.split())


def _quotes() -> dict[str, list[str]]:
    """Every quoted run carrying `SIGNATURE`, whitespace-normalized, by
    file — from the tracked markdown tree, so a new copy is found and an
    untracked one (a worktree, a build) is not."""
    out: dict[str, list[str]] = {}
    for path in tracked_files(ROOT, "*.md"):
        text = BLOCKQUOTE.sub("", FENCE.sub("", path.read_text(errors="replace")))
        for match in QUOTED.finditer(text):
            quote = _normalized(match.group(1) or match.group(2))
            if SIGNATURE in quote:
                out.setdefault(path.relative_to(ROOT).as_posix(), []).append(quote)
    return out


def test_dtype_is_refused_on_a_workflow(tmp_path, capsys) -> None:
    """A workflow's steps each declare their own realization, so one
    `--dtype` for the whole schedule has no meaning to give it."""
    message = _refusal(tmp_path, capsys)
    assert message.startswith("refused: --dtype sets model.dtype")
    assert "intervention specification" in message, (
        "the refusal names the object §11.1 calls an intervention specification"
    )


def test_resume_is_refused_on_an_intervention_specification(tmp_path, capsys) -> None:
    """`--resume` reuses a published step whose identity and files still
    match; an intervention specification run has no step boundaries to
    resume at (IM spec §9), so `run --resume` on one is refused."""
    code = main(
        [
            "run",
            "--engine",
            "auto",
            str(LOCATE),
            "--out",
            str(tmp_path / "out"),
            "--resume",
            "--artifacts-root",
            str(tmp_path),
        ]
    )
    assert code == 1
    assert "--resume is a workflow flag" in capsys.readouterr().err


def test_every_doc_that_quotes_the_refusal_quotes_it_verbatim(tmp_path, capsys) -> None:
    """Each quoted run is a substring of the message.

    A substring rather than the whole thing: a doc quotes as far as the reader
    needs and then explains, which is the right thing for it to do — the head
    in one place, the tail in another. What must not happen is the quoted part
    saying something the CLI does not.
    """
    message = _normalized(_refusal(tmp_path, capsys))
    assert SIGNATURE in message, (
        "the refusal no longer carries the words the finder recognizes a quote "
        f"by ({SIGNATURE!r}) — move SIGNATURE with the message"
    )
    quotes = _quotes()
    assert len(quotes) >= QUOTING_FILES_FLOOR, (
        f"only {sorted(quotes)} quote the --dtype refusal; the finder used to "
        f"see {QUOTING_FILES_FLOOR} files, so either a quote was dropped or the "
        "finder stopped seeing one"
    )
    wrong = [
        f"  {name}: {quote}"
        for name, found in sorted(quotes.items())
        for quote in found
        if quote not in message
    ]
    assert not wrong, (
        "a doc quotes a --dtype refusal the CLI does not print:\n"
        + "\n".join(wrong)
        + f"\n  cli: {message}"
    )
