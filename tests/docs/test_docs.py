"""Check documentation against the current files and CLI.

Parse commands with the real argument parser, load complete JSON examples, and
check links, quoted digests, and module paths. Coverage includes nested reference
pages. Architecture tables may select the entry points useful to a reader."""

from __future__ import annotations

import ast
import contextlib
import functools
import io
import json
import re
import shlex
from pathlib import Path
from typing import NamedTuple

import pytest

from tests.protocol._env import steps_of
from tests._helpers.paths import METHODS_DIR

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[2]
DOCS = REPO / "docs"

#: Any fenced block, with its info string. Anchored at line starts so a lazy
#: body stops at the first closing fence.
#:
#: Up to **three** spaces of indentation are allowed before either fence,
#: because CommonMark allows them and every fence inside a numbered list has
#: them. Anchoring at column 0 made `README.md` — whose three fences all sit in
#: its Quick Start list — contribute zero commands while the module docstring
#: named it, and hid two `running_experiments.md` blocks carrying exactly the
#: flags most worth checking (`--register-from-hf`, `--set model.key=…`).
#: `_commands_in` strips each line, so an indented body needs nothing else.
FENCE = re.compile(
    r"^ {0,3}```([^\n]*)\n(.*?)^ {0,3}```[ \t]*$", re.DOTALL | re.MULTILINE
)


def _prose(text: str) -> str:
    """``text`` with every fenced block removed.

    Links inside a fence are *examples*, not links: ``docs/demos.md`` is the
    demo format's own spec, so its fenced ```markdown template shows a header
    table whose ``[protocols/…](protocols/…)`` is relative to a demo directory
    that does not exist from ``docs/``. Checking those would make the spec
    unable to show its own format.
    """
    return FENCE.sub("", text)


def _fenced(text: str, language: str | None = None) -> list[str]:
    """The bodies of ``text``'s fenced blocks, optionally only one language's.

    The info string is compared by its first word, which is where CommonMark
    puts the language (`json`, `json title=…`); anything after it is not.
    """
    return [
        body
        for info, body in FENCE.findall(text)
        if language is None or info.split()[:1] == [language]
    ]


#: A markdown link whose target is a path rather than a URL or an anchor.
LINK = re.compile(r"\[[^\]]*\]\((?!https?://|#|mailto:)([^)\s]+)\)")

#: `digest <hex>` — the shape the demo suite checks. `running_experiments.md`
#: quotes four of them in its `weekdays_8b` `explain` block; `_checkable_blocks`
#: decides which are recomputable here.
QUOTED_DIGEST = re.compile(r"digest\s+`?([0-9a-f]{8,64})")

#: Placeholders a documented command legitimately carries. They stand where a
#: reader supplies a path, so for parsing purposes they are just a token.
PLACEHOLDER = re.compile(r"<[^>\s]+>|\$\{[^}]*\}|\$\([^)]*\)")

#: Where a documented command line stops being the command: a trailing
#: `# comment` (README.md annotates one that way), a pipe, a redirect (`>`,
#: `2>`), or a `;`/`&&` chain. `shlex` keeps all of these as tokens, so without
#: the cut a perfectly good `explain x.json   # what must be bound` parses as
#: four extra positionals. Whitespace before the metacharacter is required so
#: that `<name>`, `ref#split` and `--set a=b` survive.
SHELL_TAIL = re.compile(r"\s+(?:\d?[>|&;]|#)")


@functools.cache
def _live_docs() -> tuple[Path, ...]:
    """Every doc the checks apply to.

    Cached: the parametrize decorators below call it at collection, several
    times, and `intervention_protocol.md` alone is ~1.7k lines.
    """
    out = tuple(sorted(DOCS.rglob("*.md")))
    assert out, "no docs found — the glob is wrong"
    return out


def _guides() -> list[Path]:
    """The docs whose *examples* nobody else loads: `docs/` and the README.

    The demos' documents and digests are the demo suite's — it inlines each
    beside its link and loads it — so their fenced JSON is not re-parsed here.
    """
    return [REPO / "README.md", *_live_docs()]


def _markdown() -> list[Path]:
    """Every markdown file whose commands, links and rooted paths are checked:
    the guides plus the demos (whose *documents* the demo suite already loads,
    but whose *commands* nothing parsed)."""
    return [*_guides(), *sorted((REPO / "demos").glob("**/*.md"))]


def _ids(paths: list[Path]) -> list[str]:
    return [str(p.relative_to(REPO)) for p in paths]


# --------------------------------------------------------------------------- #
# the documented command line, against the parser that defines it
# --------------------------------------------------------------------------- #


def _commands(text: str) -> list[str]:
    """Every `causalab …` invocation in a fenced code block.

    Backslash continuations are joined first, so a multi-line command is one
    command and its later flags are not silently dropped — which is where the
    flags most worth checking tend to sit. A leading `$ ` prompt and a
    `uv run ` prefix are stripped.
    """
    return [c for block in _fenced(text) for c in _commands_in(block)]


def _commands_in(block: str) -> list[str]:
    """Every `causalab …` invocation inside one already-unfenced block."""
    out: list[str] = []
    for line in re.sub(r"\\\n\s*", " ", block).splitlines():
        line = SHELL_TAIL.split(line.strip(), maxsplit=1)[0].strip()
        line = re.sub(r"^\$\s+", "", line)
        line = re.sub(r"^uv run\s+", "", line)
        # A leading venv path is stripped so that the standalone-install
        # recipe's `/path/to/venv/bin/causalab …` is checked like any other
        # documented command.
        line = re.sub(r"^\S*/bin/causalab(?=\s|$)", "causalab", line)
        # `causalab` followed by whitespace or nothing — not `causalab/`, which
        # is the package directory in a tree diagram, and not
        # `causalab.protocol`, which is a module path
        if re.match(r"^causalab(\s|$)", line):
            out.append(line)
    return out


def _argv(command: str) -> list[str]:
    """A documented command as the argument list the CLI would receive.

    Placeholders (`<doc>`, `${SLURM_ARRAY_TASK_ID}`, `$(…)`) become a single
    token: they stand where a reader supplies a value, and what is under test is
    the *flags*, not the paths. One tokenization for every check here, so the
    parse test and the digest filter cannot drift apart.
    """
    return shlex.split(PLACEHOLDER.sub("PLACEHOLDER", command))[1:]


def _documented_commands() -> list[tuple[str, str]]:
    """(source, command) for every documented invocation."""
    return [
        (str(path.relative_to(REPO)), command)
        for path in _markdown()
        for command in _commands(path.read_text())
    ]


COMMANDS = _documented_commands()

#: A line that *is* a `causalab` command, wherever it sits — fenced or not,
#: prompt or `uv run` prefix or not. Deliberately independent of `FENCE`: it is
#: the check on the extractor, so it must not share the extractor's blind spot.
COMMAND_LINE = re.compile(
    r"^\s*(?:\$\s+)?(?:uv run\s+|\S*/bin/)?causalab (?:run|validate|explain|dry-run|digest)\b",
    re.MULTILINE,
)


def test_the_commands_were_found() -> None:
    """A parametrization over an empty list passes vacuously — the one way a
    guard like this fails silently."""
    assert len(COMMANDS) > 20, f"only found {len(COMMANDS)} documented commands"


def test_every_file_that_documents_a_command_yielded_one() -> None:
    """A global floor cannot see a whole file going quiet.

    A fence the extractor stops recognizing takes its file's commands with it
    and the count above barely moves — which is how `README.md` contributed
    nothing for as long as `FENCE` was anchored at column 0. Per source, the
    same regression is a failure naming the file.
    """
    found = {source for source, _ in COMMANDS}
    silent = [
        str(p.relative_to(REPO))
        for p in _markdown()
        if COMMAND_LINE.search(p.read_text()) and str(p.relative_to(REPO)) not in found
    ]
    assert not silent, f"{silent} document commands the extractor never saw"


@pytest.mark.parametrize(
    ("source", "command"),
    COMMANDS,
    ids=[f"{s}::{c[:60]}" for s, c in COMMANDS],
)
def test_a_documented_command_parses(source: str, command: str) -> None:
    """The real parser, on the real argument list. `--help` exits 0 and is a
    pass; an unknown verb fails here with argparse's own message, which lists
    the verbs that exist."""
    from causalab.cli import _build_parser

    argv = _argv(command)
    parser = _build_parser()
    stderr = io.StringIO()
    try:
        with (
            contextlib.redirect_stderr(stderr),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            parser.parse_args(argv)
    except SystemExit as exit_code:
        if exit_code.code in (0, None):
            return  # --help
        pytest.fail(
            f"{source} documents a command the CLI refuses:\n"
            f"    causalab {' '.join(argv)}\n"
            f"{stderr.getvalue().strip()}"
        )


# --------------------------------------------------------------------------- #
# the fenced JSON
# --------------------------------------------------------------------------- #

#: A fenced JSON block in the docs is one of three things, and all three are
#: legitimate — what is not legitimate is a fourth: malformed.
#:
#: * a **whole document** (it has `version`) — loads and validates;
#: * a **fragment**: one or more top-level pairs shown without their enclosing
#:   braces (`"model": {...}`), which is how §2 shows one section at a time;
#: * an **elision**: a fragment with `{...}` standing for content the example
#:   is not about, or `//` comments annotating alternatives.
#:
#: A fragment is normalized (braces added, elisions filled, comments stripped)
#: and must then parse. That is a real check: it caught nothing here, but a
#: trailing comma or an unbalanced brace in an example is exactly the defect a
#: reader copies.
#:
#: A comment starts at a line start or after whitespace, so that the `//` of a
#: `"https://…"` string value is left alone.
COMMENT = re.compile(r"(?:^|(?<=\s))//[^\n]*", re.MULTILINE)
BARE_ELLIPSIS = re.compile(r"\{\s*\.\.\.\s*\}")


def _normalized(block: str) -> str:
    text = COMMENT.sub("", block)
    text = BARE_ELLIPSIS.sub("{}", text)
    stripped = text.strip()
    if not stripped.startswith("{") and not stripped.startswith("["):
        text = "{" + text + "}"
    return text


def _json_blocks() -> list[tuple[Path, int, str]]:
    out: list[tuple[Path, int, str]] = []
    for path in _guides():
        blocks = _fenced(path.read_text(), "json")
        out.extend((path, index, block) for index, block in enumerate(blocks))
    return out


JSON_BLOCKS = _json_blocks()


def test_the_json_blocks_were_found() -> None:
    assert len(JSON_BLOCKS) > 10, f"only found {len(JSON_BLOCKS)} json blocks"


@pytest.mark.parametrize(
    ("path", "index", "block"),
    JSON_BLOCKS,
    ids=[f"{p.name}[{i}]" for p, i, _ in JSON_BLOCKS],
)
def test_a_fenced_json_block_parses(path: Path, index: int, block: str) -> None:
    """Every fenced example is well-formed JSON once its elisions are filled.

    An example a reader cannot paste is worse than no example: it reads as
    authoritative and fails on their machine, not ours.
    """
    try:
        json.loads(_normalized(block))
        return
    except json.JSONDecodeError as error:
        first = error
    # A fence may hold several *alternatives* rather than one object — §2.4
    # shows three sites side by side, each annotated with what the loader says
    # about it. Then every line has to be valid on its own, which is the same
    # claim one level down rather than a weaker one. A trailing comma is the
    # one thing an alternative may carry that a standalone value may not.
    lines = [line for line in COMMENT.sub("", block).splitlines() if line.strip()]
    if len(lines) > 1:
        try:
            for line in lines:
                json.loads(_normalized(line.rstrip().rstrip(",")))
            return
        except json.JSONDecodeError:
            pass
    pytest.fail(f"{path.name}[{index}] is not valid JSON ({first}):\n{block[:400]}")


def _whole_documents() -> list[tuple[Path, int, dict]]:
    out: list[tuple[Path, int, dict]] = []
    for path, index, block in JSON_BLOCKS:
        try:
            raw = json.loads(_normalized(block))
        except json.JSONDecodeError:
            continue  # the parse test above owns that failure
        if isinstance(raw, dict) and "header" in raw and "model" in raw:
            out.append((path, index, raw))
    return out


WHOLE_DOCUMENTS = _whole_documents()

#: The fixture tables the docs' example documents name (`weekdays/train`).
#: Where a documented *command* names its own `--data-root`, that root is used
#: instead (`_checkable_blocks`); this is for the inline examples, which have
#: no command beside them.
DATA_ROOT = REPO / "tests" / "protocol" / "fixtures" / "data"


def test_whole_documents_were_found() -> None:
    """If the classifier stops recognizing whole documents, the load test below
    silently checks nothing."""
    assert len(WHOLE_DOCUMENTS) >= 3, (
        f"only classified {len(WHOLE_DOCUMENTS)} fenced blocks as whole "
        "documents — the docs show more than that"
    )


@pytest.mark.parametrize(
    ("path", "index", "raw"),
    WHOLE_DOCUMENTS,
    ids=[f"{p.name}[{i}]" for p, i, _ in WHOLE_DOCUMENTS],
)
def test_a_documented_document_loads(path: Path, index: int, raw: dict) -> None:
    """A whole document shown in the docs loads and validates.

    Elided documents are skipped by the classifier above, not here: a document
    with `{...}` where its `reads` should be is an *illustration of the split*,
    and asserting it validates would be asserting something the example never
    claimed. What is checked is every example that is complete enough to run.
    """
    from causalab.protocol.rules.errors import ValidationError
    from causalab.protocol.pipeline import compile_protocol
    from causalab.io.env import FileArtifacts, FileDatasets, ResolutionEnv
    from causalab.tasks import TASKS_ROOT

    env = ResolutionEnv(
        datasets=FileDatasets(root=DATA_ROOT, fallback_roots=(TASKS_ROOT,)),
        artifacts=FileArtifacts(root=REPO),
    )
    try:
        compile_protocol(raw, env=env)
    except ValidationError as error:
        # A doc may legitimately show a model the static registry does not
        # carry — `running_experiments.md` documents exactly that case, and the
        # `--register-from-hf` flag for it. Registry membership is an
        # environment fact, not a property of the document. What has *already
        # passed* when this V4 fires is everything up to it in
        # `compile_protocol`: the strict parse, and the §5 checklist over one
        # representative per axis value, which `build`'s guard consults before
        # a refusal raised in `canonicalize` gets out. What is out of reach is
        # canonicalization and so the digest — the one step that asks the
        # registry for the model's widths — and only for a
        # document whose model this tree cannot size. Every violation in an
        # aggregate has to be that one, or something real is being skipped.
        # (`ResolutionEnv(model_info=…)` is the seam that would put these
        # examples through the full pipeline; it costs a hardcoded `ModelInfo`
        # fixture for the A3B, so it is a hint, not this test.)
        violations = getattr(error, "errors", (error,))
        if not all(e.rule == 4 and e.path == "model.key" for e in violations):
            raise


# --------------------------------------------------------------------------- #
# the layout of a documented command
# --------------------------------------------------------------------------- #

#: The fence languages whose bodies are shell commands.
SHELL_LANGUAGES = frozenset({"bash", "sh"})

#: A flag: one or two dashes, then a letter. `--` alone ends the options and
#: `-1` is a value.
FLAG = re.compile(r"^--?[A-Za-z]")

#: The operators that end one command and start the next on the same line.
CHAIN = re.compile(r"\s(?:&&|\|\||\||;)\s")

#: The start of a heredoc (`<<'PY'`, `<<EOF`); its body is not shell.
HEREDOC = re.compile(r"<<-?\s*['\"]?(\w+)['\"]?")

#: Programs whose `-m` names the module to run, so it is part of the command.
MODULE_RUNNERS = frozenset({"python", "python3", "torchrun"})


def _flag_count(tokens: list[str], module_runner: bool) -> int:
    return sum(
        1
        for token in tokens
        if FLAG.match(token) and not (module_runner and token == "-m")
    )


def _shell_commands(block: str) -> list[list[str]]:
    """The commands of one shell fence, each as its physical lines.

    Continuation lines join their command, comment lines and heredoc bodies
    drop out, and a chain (`a && b`) splits into one command per program.
    """
    commands: list[list[str]] = []
    current: list[str] = []
    heredoc_end: str | None = None
    for line in block.splitlines():
        if heredoc_end is not None:
            heredoc_end = None if line.strip() == heredoc_end else heredoc_end
            continue
        code = re.split(r"(?:^|\s)#", line, maxsplit=1)[0].rstrip()
        if not code and not current:
            continue
        if match := HEREDOC.search(code):
            heredoc_end = match.group(1)
        current.append(code.removesuffix("\\").rstrip())
        if not code.endswith("\\"):
            commands.append(current)
            current = []
    if current:
        commands.append(current)
    out: list[list[str]] = []
    for command in commands:
        segments: list[list[str]] = [[]]
        for line in command:
            parts = CHAIN.split(line)
            segments[-1].append(parts[0])
            segments.extend([part] for part in parts[1:])
        out.extend(segments)
    return out


def _flag_layout_errors(block: str) -> list[str]:
    """The commands in ``block`` that break the one-flag-per-line layout.

    A command with at most one flag may sit on one line. A command with two or
    more keeps its first line free of flags and has one flag per continuation
    line (docs/STYLE_GUIDE.md, Shell commands).
    """
    errors: list[str] = []
    for command in _shell_commands(block):
        lines: list[list[str]] = []
        for line in command:
            text = PLACEHOLDER.sub("PLACEHOLDER", line)
            try:
                lines.append(shlex.split(text))
            except ValueError:
                lines.append(text.split())
        words = [
            word
            for tokens in lines
            for word in tokens
            if "=" not in word or FLAG.match(word)
        ]
        program = next((w for w in words if w not in ("uv", "run")), "")
        module_runner = program.rsplit("/", 1)[-1] in MODULE_RUNNERS
        counts = [_flag_count(tokens, module_runner) for tokens in lines]
        if sum(counts) >= 2 and (counts[0] > 0 or max(counts) > 1):
            errors.append(" \\ ".join(command).strip())
    return errors


def test_the_flag_layout_check_catches_flags_on_one_line() -> None:
    """The check itself, on the layouts it must refuse and accept."""
    assert _flag_layout_errors("causalab run a.json --engine auto --out o\n")
    assert _flag_layout_errors("causalab run a.json --engine auto \\\n    --out o\n")
    assert not _flag_layout_errors(
        "uv run causalab run a.json \\\n    --engine auto \\\n    --out o\n"
    )
    assert not _flag_layout_errors("uv sync --extra notebook\n")
    assert not _flag_layout_errors(
        "python3 -m venv .venv && source .venv/bin/activate\n"
    )
    assert not _flag_layout_errors("uv run python - <<'PY'\nx = f(--a, --b)\nPY\n")


@pytest.mark.parametrize("path", _markdown(), ids=_ids(_markdown()))
def test_each_flag_of_a_documented_command_has_its_own_line(path: Path) -> None:
    """A reader copies a command and edits one flag. One flag per line makes the
    flag easy to find and the edit a one-line diff."""
    errors = [
        error
        for info, body in FENCE.findall(path.read_text())
        if info.split()[:1] and info.split()[0] in SHELL_LANGUAGES
        for error in _flag_layout_errors(body)
    ]
    assert not errors, (
        f"{path.relative_to(REPO)}: put each flag on its own continuation line "
        f"(docs/STYLE_GUIDE.md, Shell commands):\n" + "\n".join(errors)
    )


#: A colour set inside a Mermaid diagram (`fill:#fff`, `color:white`).
MERMAID_COLOUR = re.compile(
    r"\b(?:fill|color|stroke)\s*:\s*(?!transparent\b|none\b)[#\w]"
)


@pytest.mark.parametrize("path", _markdown(), ids=_ids(_markdown()))
def test_a_mermaid_diagram_takes_its_colours_from_the_theme(path: Path) -> None:
    """The site's theme sets the label colour for each scheme. A fixed node
    colour keeps its value in both schemes, so the labels lose contrast in one
    of them (docs/STYLE_GUIDE.md, Diagrams)."""
    coloured = [
        line.strip()
        for body in _fenced(path.read_text(), "mermaid")
        for line in body.splitlines()
        if re.match(r"\s*(?:classDef|style)\s", line) and MERMAID_COLOUR.search(line)
    ]
    assert not coloured, f"{path.relative_to(REPO)}: colours in a diagram {coloured}"


def test_the_readme_protocol_is_the_file_it_runs() -> None:
    """The quick start shows a protocol and runs a file. Without its comments,
    the copy must be that file, or the reader reads one experiment and runs
    another."""
    readme = (REPO / "README.md").read_text()
    shown = [
        json.loads(COMMENT.sub("", block))
        for block in _fenced(readme, "json")
        if '"header"' in block
    ]
    runs = {
        match
        for command in _commands(readme)
        for match in re.findall(r"\S+\.json", command)
    }
    assert len(shown) == 1 and len(runs) == 1, (shown, runs)
    assert shown[0] == json.loads((REPO / runs.pop()).read_text())


# --------------------------------------------------------------------------- #
# links and digests
# --------------------------------------------------------------------------- #


def _link_checked() -> list[Path]:
    """Every covered markdown file.

    `README.md` and `demos/README.md` are here because no other suite reads
    them: the demo suite globs demo *directories*, which leaves `demos/README.md`
    outside it.
    """
    return _markdown()


@pytest.mark.parametrize("path", _link_checked(), ids=_ids(_link_checked()))
def test_links_resolve(path: Path) -> None:
    """A dead relative link in the docs is the most-clicked kind of rot.

    A target is resolved against the file's own directory and nothing else —
    that is how markdown resolves it, on GitHub and everywhere. A repo-root
    fallback used to sit beside it and could only weaken the check: nothing in
    the tree needed it, and what it admitted was `[x](docs/TESTS.md)` written
    *inside* `docs/` — dead where it is read, green here.
    """
    missing = [
        target
        for target in LINK.findall(_prose(path.read_text()))
        if not (path.parent / target.split("#", 1)[0]).exists()
    ]
    assert not missing, f"{path.relative_to(REPO)}: dead links {missing}"


class _Checkable(NamedTuple):
    """A fenced block whose quoted digests this tree can recompute, and the
    roots its own command resolves against."""

    block: str
    data_root: Path
    artifacts_root: Path


def _checkable_blocks(text: str) -> list[_Checkable]:
    """The fenced blocks whose digests this tree can actually recompute.

    A pasted digest is only verifiable if everything it is a function of is *in
    the tree* — the document and the resolved data alike, since a dataset's
    content digest is part of a document's identity (§7).
    `running_experiments.md` quotes both kinds: the `weekdays_8b` block runs a
    committed workflow against the committed fixture tables, so its digests are
    checkable and were in fact **stale**; the tutorial blocks above it name
    `patch.json` and `data/`, which the reader creates, so no digest over them
    can be recomputed here.

    The test is the block's own command rather than a marker an author has to
    remember: a block is checkable when it runs a `causalab` verb whose
    document argument and `--data-root` both exist in the tree. Both are read
    off the **real parser**, not a hand-rolled argv: option values are not
    positionals, the document need not come first, and the roots the digests
    are then verified against are the ones the command names — a block
    documenting a different data root is checked against *its* tables, not a
    module constant's.
    """
    from causalab.cli import _build_parser

    out: list[_Checkable] = []
    for block in _fenced(text):
        if not QUOTED_DIGEST.search(block):
            continue
        for command in _commands_in(block):
            try:
                with (
                    contextlib.redirect_stderr(io.StringIO()),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    args = _build_parser().parse_args(_argv(command))
            except SystemExit:
                continue  # the parse test owns that failure
            data_root = REPO / args.data_root
            if not (REPO / args.document).is_file() or not data_root.is_dir():
                continue
            out.append(_Checkable(block, data_root, REPO / args.artifacts_root))
            break
    return out


def test_at_least_one_doc_has_checkable_digests() -> None:
    """Without this the digest check can go vacuous — every doc skipping reads
    as green, which is how the stale `weekdays_8b` digests survived a first
    draft of this very test."""
    checked = [p.name for p in _guides() if _checkable_blocks(p.read_text())]
    assert checked, "no doc's digests are checkable — the block filter is wrong"


@pytest.mark.parametrize("path", _guides(), ids=_ids(_guides()))
def test_quoted_digests_are_real(path: Path) -> None:
    """Any digest the docs paste has to be one a document in this tree has.

    This found four stale digests in `running_experiments.md`'s `explain`
    output on its first run — the block claims to be pasted output and had
    drifted from what the command prints.
    """
    blocks = _checkable_blocks(path.read_text())
    if not blocks:
        pytest.skip(f"{path.name} quotes no digest over in-tree paths")
    stale = sorted(
        {
            hexits
            for block, data_root, artifacts_root in blocks
            for hexits in QUOTED_DIGEST.findall(block)
            if not any(
                d.startswith(hexits) for d in _real_digests(data_root, artifacts_root)
            )
        }
    )
    assert not stale, (
        f"{path.name}: digests {stale} match no document this tree has — "
        "re-paste the explain output"
    )


@functools.cache
def _real_digests(data_root: Path, artifacts_root: Path) -> frozenset[str]:
    """Every digest the docs could legitimately be quoting, under these roots.

    Two sources, because the docs quote both: a document shown inline in a
    fence, and a **shipped** document the docs run by path
    (`demos/methods/workflows/weekdays.json` and its steps — a workflow
    contributes its own digest plus each step's inner and stamped ones, which
    is where §9's example output comes from).

    Cached per pair of roots: it loads every shipped config and workflow, and
    each doc that quotes a digest would otherwise redo that.
    """
    from causalab.protocol.pipeline import compile_protocol
    from causalab.io.env import FileArtifacts, FileDatasets, ResolutionEnv
    from causalab.workflow.document import load_workflow
    from causalab.tasks import TASKS_ROOT

    env = ResolutionEnv(
        datasets=FileDatasets(root=data_root, fallback_roots=(TASKS_ROOT,)),
        artifacts=FileArtifacts(root=artifacts_root),
    )
    real: set[str] = set()
    for _p, _i, raw in WHOLE_DOCUMENTS:
        with contextlib.suppress(Exception):
            loaded = compile_protocol(raw, env=env)
            real.add(loaded.digests.document)
            real.update(steps_of(loaded, env).digests)
    for shipped in sorted(METHODS_DIR.glob("*/*.json")):
        raw_text = shipped.read_text()
        with contextlib.suppress(Exception):
            if '"steps"' in raw_text:
                workflow = load_workflow(shipped, env)
                real.add(workflow.digest)
                real.update(workflow.inner_digests.values())
                real.update(workflow.step_digests.values())
            else:
                loaded = compile_protocol(shipped, env=env)
                real.add(loaded.digests.document)
                real.update(steps_of(loaded, env).digests)
    return frozenset(real)


# --------------------------------------------------------------------------- #
# a quoted module path names a file that is there
# --------------------------------------------------------------------------- #
#
# The defect class these two checks close: *a doc names a module or directory
# that moved.* An audit of `docs/` found it five times over — `steps/` and
# `tests/steps/` (never existed under those names), `protocol/workflow.py`
# (moved to `workflow/document.py`), `neural/pytorch_hooks/…` (moved under
# `engines/`), and two per-module tables that had drifted from their own
# packages in both directions at once. None of it was visible to any test,
# because a stale path reads exactly like a live one.

#: Directories a rooted path may start with — the repo's own top level. A token
#: starting with one of these is a claim about *this tree*, which is what makes
#: it checkable; `neural/shared/` or `schema.py` on its own is a claim about
#: the section it sits in, and resolving those would need a table's context.
REPO_ROOTS = ("causalab/", "tests/", "docs/", "demos/", "scripts/")

#: A backtick-quoted token. Deliberately not anchored to a path shape: the
#: filter below is what decides, and it is easier to read than one regex.
BACKTICKED = re.compile(r"`([^`\n]+)`")


def _rooted_paths(text: str) -> list[str]:
    """Every repo-rooted path quoted in ``text``'s prose.

    Three exclusions, each for a token that is not a claim about the tree:

    * **fenced blocks** — example content, handled like `test_links_resolve`
      (a document body may name `scripts/probe.py` that no one shipped);
    * **a token that is itself a quoted string** (`"scripts/prompt.txt"`) — a
      JSON *value* shown in prose, which §3 of the workflow spec uses to make
      exactly the point that a string is not a path. Tested explicitly rather
      than falling out of the leading `"`. Note the converse is deliberate: an
      example path written in prose *without* the quotes is checked, because
      prose that names a path without marking it as a value is making a claim
      about the tree;
    * **globs and placeholders** (`tests/tasks/<task>/pinned_samples.json`,
      `demos/methods/protocols/*.json`) — a shape, not a file.
    """
    out = []
    for raw in BACKTICKED.findall(_prose(text)):
        token = raw.strip()
        if token.startswith('"') and token.endswith('"'):
            continue
        if not token.startswith(REPO_ROOTS):
            continue
        if any(ch in token for ch in "*<>$ "):
            continue
        # a pytest nodeid's `::selector` is not part of the filename, and a
        # `path:12` / `path:12-14` line citation is a claim about the file, not
        # about a file named with a colon
        token = re.sub(r":\d+(?:-\d+)?$", "", token.split("::")[0])
        out.append(token.rstrip("/"))
    return out


def _roots(path: Path) -> tuple[Path, ...]:
    """Where a document's rooted paths resolve: the repository root for every
    document, and for a replication package's page also ``demos/papers/``.
    Every package's commands run from that directory, so a page's
    `workflows/scripts/<name>/figure.py` is the folder's file, not the
    repository's."""
    if path.is_relative_to(REPO / "demos" / "papers"):
        return (REPO, path.parent)
    return (REPO,)


@pytest.mark.parametrize("path", _markdown(), ids=_ids(_markdown()))
def test_rooted_paths_resolve(path: Path) -> None:
    """A path written from the repo root has to exist at the repo root, or for
    a replication package's page in ``demos/papers/`` (`_roots`)."""
    roots = _roots(path)
    missing = sorted(
        {
            p
            for p in _rooted_paths(path.read_text())
            if not any((root / p).exists() for root in roots)
        }
    )
    assert not missing, (
        f"{path.relative_to(REPO)} names paths that are not in the tree: {missing}. "
        "If one moved, follow it; if it is gone, say so without spelling it as a "
        "live path."
    )


def test_the_rooted_paths_were_found() -> None:
    """The vacuous-pass guard, as for the commands."""
    found = sum(len(_rooted_paths(p.read_text())) for p in _markdown())
    assert found > 50, f"only {found} rooted paths found — the pattern is wrong"


#: A heading naming the package a table under it maps, e.g.
#: `## 2. The protocol layer (`causalab/protocol/`)`.
PACKAGE_HEADING = re.compile(
    r"^#{2,3} .*\(`(causalab/[a-z0-9_/]+)/`\)\s*$", re.MULTILINE
)

#: Expected tables of selected module entry points. Pin this set so a changed
#: heading cannot silently remove a table from the checks.
GUARDED_PACKAGES = frozenset(
    {
        "causalab/causal",
        "causalab/protocol",
        "causalab/workflow",
        "causalab/measurement",
    }
)

#: Rows that name modules relative to the package heading.
MODULE_ROW = re.compile(r"^\|\s*((?:`[a-z0-9_/]+\.py`[,\s]*)+)\|", re.MULTILINE)


def _is_star_import_forwarder(module: Path) -> bool:
    """A module whose only statement is a star import forwards another module.

    The docstring and ``from __future__`` imports are ignored, so a shim copied
    from a neighbour's header is still detected.
    """
    body = [
        node
        for node in ast.parse(module.read_text()).body
        if not (
            (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant))
            or (isinstance(node, ast.ImportFrom) and node.module == "__future__")
        )
    ]
    return (
        len(body) == 1
        and isinstance(body[0], ast.ImportFrom)
        and any(alias.name == "*" for alias in body[0].names)
    )


def test_no_module_is_a_star_import_forwarder() -> None:
    """The mechanical half of CODEBASE.md's rule that no module exists only to
    forward another module's names. A forwarder that lists its names is a
    review-time check."""
    forwarders = sorted(
        path.relative_to(REPO).as_posix()
        for path in (REPO / "causalab").rglob("*.py")
        if _is_star_import_forwarder(path)
    )
    assert forwarders == [], f"star-import forwarders on the tree: {forwarders}"


def _module_tables() -> list[tuple[str, set[str], set[str]]]:
    """(package, documented modules, live modules) for each package module map.

    A section is only checked if it actually draws one — a heading that names a
    package but introduces prose or a capability table is skipped rather than
    being asserted to document zero modules.
    """
    text = (DOCS / "CODEBASE.md").read_text()
    headings = list(PACKAGE_HEADING.finditer(text))
    assert headings, "no package-titled sections found in CODEBASE.md"
    out = []
    for i, match in enumerate(headings):
        end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
        package = match.group(1)
        directory = REPO / package
        if not directory.is_dir():
            # `test_rooted_paths_resolve` already owns this claim — a heading's
            # `causalab/…/` is a rooted path in prose — and reports it as one
            # clean failure. Asserting here instead raises during *collection*
            # (MODULE_TABLES is built at import) and takes down every test in
            # this file, including the one that would explain why.
            continue
        documented = {
            name
            for cell in MODULE_ROW.findall(text[match.end() : end])
            for name in re.findall(r"`([a-z0-9_/]+\.py)`", cell)
        }
        if not documented:
            continue
        live = {
            p.relative_to(directory).as_posix()
            for p in directory.rglob("*.py")
            if p.name != "__init__.py"
        }
        out.append((package, documented, live))
    guarded = {package for package, _documented, _live in out}
    assert guarded == GUARDED_PACKAGES, (
        "the set of guarded module tables changed — "
        f"no longer guarded: {sorted(GUARDED_PACKAGES - guarded)}; "
        f"newly guarded: {sorted(guarded - GUARDED_PACKAGES)}. If a section "
        "legitimately gained or lost its module map, update GUARDED_PACKAGES "
        "deliberately."
    )
    return out


MODULE_TABLES = _module_tables()


@pytest.mark.parametrize(
    ("package", "documented", "live"), MODULE_TABLES, ids=[t[0] for t in MODULE_TABLES]
)
def test_documented_modules_exist(
    package: str, documented: set[str], live: set[str]
) -> None:
    """Architecture tables describe selected entry points that must exist."""
    assert documented <= live, (
        f"{package}: documented modules missing from the package: "
        f"{sorted(documented - live)}"
    )
