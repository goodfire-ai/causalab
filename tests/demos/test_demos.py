"""The demos are checked, not trusted (``docs/demos.md`` §9).

A demo is prose wrapped around documents, and prose rots silently: a renamed
component, a moved script module or a retired metric kind leaves the markdown
reading exactly as before and the JSON beside it dead. So the mechanical half
of the format's checklist runs here —

* every document under a demo loads and validates against its own demo's
  data root, ``--data`` included, so a column a metric names has to exist;
* every workflow's steps resolve, which reaches into each inner document;
* every relative link in a demo's markdown points at a file that exists, and
  every committed figure is actually shown;
* every document is inlined in its demo's markdown beside a link to the file:
  a workflow *byte for byte* (§5.4), an intervention specification with ``//``
  comments whose copy minus the comments is the file's JSON (§4.6, §4.9), so
  the copy a reader sees and the bytes a run reads cannot drift apart;
* every demo has the layout of ``docs/demos.md`` §1 and §2 — the two-row top
  table under an ``Overview`` header, the five sections in order, no
  ``Reproduced`` field (§8.8). A demo in an earlier layout is refused until
  it is rewritten; the harness carries no list of exceptions;
* every digest a demo quotes is one its own documents and tables really have,
  which is what makes "pasted output, not typed by hand" a check.

What is deliberately *not* checked is the prose: whether a number has a floor,
whether a figure caption is honest, whether a verdict answers its question.
Those are review, and the checklist says so.

The tier is ``unit``: validation is pure — no weights, no network, no
accelerator — so this costs a second and runs in the CPU tier CI selects.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

import pytest

from tests._helpers.demos import (
    DEMOS,
    OTHER_GENRES,
    REPO,
    data_root,
    demo_dirs,
    demo_env,
    figures_dir,
)
from tests.protocol._env import steps_of

pytestmark = pytest.mark.unit

#: §1 — the five sections, in this order. The top table sits above the first
#: of them, so it is checked separately.
SECTIONS = (
    "Research question",
    "Method",
    "Execution",
    "Results",
    "Next steps",
)

#: §2 — the two rows of the top table, in this order.
TOP_TABLE = ("Question", "Method")

#: §2 — the top table's header row: ``Overview`` above the labels, and no
#: title above the values.
TOP_TABLE_HEADER = "| Overview | |"

#: §1 — where text that fits no section waits during a rewrite. Allowed only
#: as the last section.
MANUAL_REVIEW = "Manual review"

#: A markdown link whose target is a path rather than a URL or an anchor.
LINK = re.compile(r"\[[^\]]*\]\((?!https?://|#)([^)\s]+)\)")

#: A fenced ``json`` block. Anchored at line starts so the lazy body stops at
#: the first closing fence rather than the last one in the file, which is what
#: lets a demo carry six of them.
JSON_FENCE = re.compile(r"^```json\n(.*?)^```$", re.DOTALL | re.MULTILINE)

#: Any fenced block. A ``## `` line or a ``| **X** |`` row inside a paste of
#: terminal output or of another document is content, not structure.
FENCE = re.compile(r"^ {0,3}```.*?^ {0,3}```[ \t]*$", re.DOTALL | re.MULTILINE)

#: A ``## `` heading's text.
HEADING = re.compile(r"^## (.+?)\s*$", re.MULTILINE)

#: A table row whose first cell is a bold label: ``| **Question** | … |``.
BOLD_ROW = re.compile(r"^\|\s*\*\*([^*|]+?)\*\*\s*\|", re.MULTILINE)


def _markdown() -> list[Path]:
    """Every demo file: the markdown under a demo directory, minus its index.

    A directory's ``README.md`` is the landing page of its track, such as the
    onboarding tutorial's, and follows no demo format.
    """
    return sorted(
        p for d in demo_dirs() for p in d.glob("*.md") if p.name != "README.md"
    )


def _documents() -> list[Path]:
    return sorted(p for d in demo_dirs() for p in d.glob("protocols/*.json")) + sorted(
        p for d in demo_dirs() for p in d.glob("workflows/*.json")
    )


def _ids(paths: list[Path]) -> list[str]:
    return [str(p.relative_to(REPO)) for p in paths]


def strip_json_comments(text: str) -> str:
    """``text`` with every ``//`` comment removed (§4.6).

    A comment runs from ``//`` outside a string to the end of its line. Inside
    a string (``"https://…"``, a prompt that happens to hold two slashes) the
    two characters are content and stay. Trailing whitespace a removed comment
    leaves behind is not JSON-significant, so it is left alone.

    ``scripts/repin_demo_digests.py`` carries the same function: the tool
    has to recognize a commented copy to leave it alone, and the two must
    agree on what a comment is.
    """
    out: list[str] = []
    in_string = False
    escaped = False
    index = 0
    while index < len(text):
        char = text[index]
        if in_string:
            out.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            index += 1
            continue
        if char == '"':
            in_string = True
            out.append(char)
            index += 1
            continue
        if text.startswith("//", index):
            end = text.find("\n", index)
            index = len(text) if end == -1 else end
            continue
        out.append(char)
        index += 1
    return "".join(out)


def _as_json(fence: str) -> object | None:
    """The JSON a fenced copy parses to once its comments are gone, or
    ``None`` when it is not a JSON document at all (a table excerpt, say)."""
    try:
        return json.loads(strip_json_comments(fence))
    except json.JSONDecodeError:
        return None


def _carries(fence: str, document: Path) -> bool:
    """Whether ``fence`` is a copy of ``document`` (§4.9, §5.4).

    A workflow is inlined byte for byte. An intervention specification is
    inlined with ``//`` comments, so the copy minus its comments has to parse
    to the file's JSON; a byte-equal copy passes the same test.
    """
    want = document.read_text()
    if fence == want:
        return True
    if document.parent.name != "protocols":
        return False
    return _as_json(fence) == json.loads(want)


def _carriers(document: Path) -> list[Path]:
    """The demo markdown files that carry ``document`` in a fence."""
    return [
        demo
        for demo in sorted(document.parents[1].glob("*.md"))
        if any(
            _carries(fence, document) for fence in JSON_FENCE.findall(demo.read_text())
        )
    ]


def _run_tree_load(document: Path) -> str | None:
    """The first ``file_path`` this document LOADS whose leading segment looks
    like a step name rather than a repo path — i.e. one that resolves only in a
    run tree. Featurizers and ``params`` are the IM spec's two load sites."""
    raw = json.loads(document.read_text())
    method = raw.get("method") or {}  # the two load sites live in the method group
    for section in ("featurizers", "params"):
        for entry in (method.get(section) or {}).values():
            path = entry.get("file_path") if isinstance(entry, dict) else None
            if isinstance(path, str) and not (REPO / path).exists():
                return path
    return None


def _workflows_naming(document: Path) -> list[Path]:
    """Workflows in the same demo that run ``document`` as a step."""
    want = document.name
    found = []
    for workflow in sorted((document.parents[1] / "workflows").glob("*.json")):
        steps = json.loads(workflow.read_text()).get("steps", {})
        if any(str(s.get("document", "")).endswith(want) for s in steps.values()):
            found.append(workflow)
    return found


def _prose(text: str) -> str:
    """``text`` with its fenced blocks blanked, line count preserved."""
    return FENCE.sub(lambda match: "\n" * match.group().count("\n"), text)


def _headings(text: str) -> list[str]:
    return HEADING.findall(_prose(text))


def _head(text: str) -> str:
    """The text above the first ``## `` heading: the title and the top table."""
    prose = _prose(text)
    match = HEADING.search(prose)
    return prose[: match.start()] if match else prose


def _top_table_rows(text: str) -> list[str]:
    return BOLD_ROW.findall(_head(text))


def _top_table_header(text: str) -> str | None:
    """The first table row above the first section, or ``None`` without one."""
    for line in _head(text).splitlines():
        if line.startswith("|"):
            return line.strip()
    return None


class TestGenres:
    def test_the_other_genres_are_directories(self) -> None:
        """An excluded name that no longer exists is a stale exclusion; a demo
        directory that is really another genre has to be named here, or the
        format checks read it as a demo."""
        for name in OTHER_GENRES:
            assert (DEMOS / name).is_dir(), (
                f"demos/{name} is excluded but does not exist"
            )
        for demo in demo_dirs():
            assert list(demo.glob("*.md")), (
                f"demos/{demo.name} has no markdown, so it is not a demo — name it "
                "in OTHER_GENRES (tests/_helpers/demos.py) with its own suite"
            )


class TestDocuments:
    @pytest.mark.parametrize("document", _documents(), ids=_ids(_documents()))
    def test_validates(self, document: Path) -> None:
        """Load and validate, columns included.

        ``--data`` is the half that catches the drift a demo is most likely to
        acquire: a metric naming a column the table stopped emitting reads as
        valid structurally and produces nothing at run time.

        One document shape cannot be loaded on its own, and the exception is
        narrow on purpose. An **apply** document — the second half of every
        fit→apply pair — names its fitted artifact by a *run-tree* path
        (``"fit/rot.safetensors"``), where the first segment is a step name
        that only means something inside the workflow that declares it. Loaded
        standalone it raises ``[V15] artifact file not found``, which is the
        loader being right rather than the demo being wrong. Such a document is
        covered by its workflow's own entry in this same parametrization, and
        covered *better*: that load applies the step's ``set`` block and checks
        the producing step really writes the file. So the exception is allowed
        only when the workflow exists and names this document.
        """
        from causalab.protocol.rules.errors import ValidationError
        from causalab.protocol.pipeline import compile_protocol
        from causalab.protocol.rules.data import check_data_columns

        raw = json.loads(document.read_text())
        env = demo_env(document)
        if "steps" in raw:
            from causalab.workflow.document import load_workflow

            loaded = load_workflow(document, env)
            for name in loaded.order:
                inner = loaded.inner.get(name)
                if inner is not None:
                    check_data_columns(inner.compiled, env)
            return
        try:
            check_data_columns(compile_protocol(document, env=env), env)
        except ValidationError as error:
            if error.rule != 15 or not _run_tree_load(document):
                raise
            carriers = _workflows_naming(document)
            assert carriers, (
                f"{document.name} loads from a run tree "
                f"({_run_tree_load(document)}) but no workflow in "
                f"{document.parents[1].name}/ names it as a step — an apply "
                "document is only loadable inside the workflow that supplies "
                "its artifact"
            )

    @pytest.mark.parametrize("document", _documents(), ids=_ids(_documents()))
    def test_has_a_description(self, document: Path) -> None:
        """JSON has no comments, which is why ``description`` exists
        (spec §7). A demo document without one is a document whose reason to
        exist lives only in the markdown beside it."""
        raw = json.loads(document.read_text())
        header = raw.get("header", raw)  # a workflow keeps its description at the top
        assert header.get("description"), f"{document.name} declares no description"


class TestFormat:
    """§1, §2 and §8.8. There is no list of exceptions: a demo in an earlier
    layout fails here until it is rewritten."""

    @pytest.mark.parametrize("demo", _markdown(), ids=_ids(_markdown()))
    def test_sections_in_order(self, demo: Path) -> None:
        """Exactly the five sections of §1, in order. ``## Manual review`` may
        follow them during a rewrite, and nothing else may."""
        headings = _headings(demo.read_text())
        allowed = (list(SECTIONS), [*SECTIONS, MANUAL_REVIEW])
        assert headings in allowed, (
            f"{demo.name}: expected the sections of docs/demos.md §1 in order, "
            f"{list(SECTIONS)}, with '{MANUAL_REVIEW}' allowed last during a "
            f"rewrite; got {headings}"
        )

    @pytest.mark.parametrize("demo", _markdown(), ids=_ids(_markdown()))
    def test_top_table_is_complete(self, demo: Path) -> None:
        """§2: two rows, ``Question`` then ``Method``, above the first section."""
        rows = _top_table_rows(demo.read_text())
        assert rows == list(TOP_TABLE), (
            f"{demo.name}: docs/demos.md §2 wants a two-row top table, "
            f"**Question** then **Method**, got {rows}"
        )

    @pytest.mark.parametrize("demo", _markdown(), ids=_ids(_markdown()))
    def test_top_table_header(self, demo: Path) -> None:
        """§2: the header row names the label column ``Overview`` and leaves
        the value column untitled."""
        header = _top_table_header(demo.read_text())
        assert header == TOP_TABLE_HEADER, (
            f"{demo.name}: docs/demos.md §2 wants the top table's header row "
            f"{TOP_TABLE_HEADER!r}, got {header!r}"
        )

    @pytest.mark.parametrize("demo", _markdown(), ids=_ids(_markdown()))
    def test_no_reproduced_row(self, demo: Path) -> None:
        """§8.8: there is no ``Reproduced`` field; the Execution section states
        where and when the artifacts were produced."""
        assert "**Reproduced**" not in _prose(demo.read_text()), (
            f"{demo.name}: docs/demos.md §8.8 has no Reproduced field — state "
            "hardware, time and date in the Execution section"
        )

    @pytest.mark.parametrize("demo", _markdown(), ids=_ids(_markdown()))
    def test_links_resolve(self, demo: Path) -> None:
        missing = [
            target
            for target in LINK.findall(demo.read_text())
            if not (demo.parent / target.split("#", 1)[0]).exists()
        ]
        assert not missing, f"{demo.name}: dead links {missing}"

    def test_a_fence_is_not_structure(self, tmp_path: Path) -> None:
        """A ``## `` line or a bold table row pasted inside a fence is content:
        a demo that quotes terminal output or another document keeps its
        five sections and its two-row table."""
        demo = tmp_path / "x.md"
        body = "\n".join(
            [
                "# Title",
                "| Overview | |",
                "|---|---|",
                "| **Question** | q |",
                "| **Method** | m |",
                *(f"## {section}" for section in SECTIONS),
                "```",
                "## Limits",
                "| **Reproduced** | ✓ pasted |",
                "```",
                "",
            ]
        )
        demo.write_text(body)
        assert _headings(body) == list(SECTIONS)
        assert _top_table_rows(body) == list(TOP_TABLE)
        assert _top_table_header(body) == TOP_TABLE_HEADER
        assert "**Reproduced**" not in _prose(body)
        assert "**Reproduced**" in body


class TestPastedOutput:
    """§5.3 — a ``validate`` or ``explain`` block is pasted output, digest
    included, not typed by hand.

    A digest is the one thing in a demo that cannot be *nearly* right: it is a
    function of the document's canonical bytes, so a stale one is proof that
    the prose and the JSON have diverged. Editing a document's ``description``
    is enough to move it, which is exactly the edit a careful author makes
    without thinking to re-paste.
    """

    @pytest.mark.parametrize("demo_dir", demo_dirs(), ids=_ids(demo_dirs()))
    def test_quoted_digests_are_current(self, demo_dir: Path) -> None:
        from causalab.protocol.rules.errors import ValidationError
        from causalab.protocol.pipeline import compile_protocol
        from causalab.workflow.document import load_workflow

        real: set[str] = set()
        for document in sorted(demo_dir.glob("*/*.json")):
            if document.parent.name not in ("protocols", "workflows"):
                continue
            env = demo_env(document)
            if "steps" in json.loads(document.read_text()):
                workflow = load_workflow(document, env)
                # no whole-workflow digest is quotable (§7): the identities are
                # per step. A step's digest is its document's *with `set`
                # applied*, so it differs from the same document loaded standalone
                real.update(workflow.inner_digests.values())
                real.update(workflow.step_digests.values())
            else:
                try:
                    loaded = compile_protocol(document, env=env)
                except ValidationError as error:
                    # an apply document does not load on its own — its artifact
                    # lives in a run tree. Its digests are already in `real`
                    # via the workflow that runs it (inner_digests /
                    # step_digests above), so nothing is lost by skipping it.
                    if error.rule != 15 or not _run_tree_load(document):
                        raise
                    continue
                real.add(loaded.digests.document)
                real.update(steps_of(loaded, env).digests)
        # a demo also quotes the content digest a table was built at — the
        # same sha256 over the file's bytes that a ref resolves to
        # (causalab.io.env.FileDatasets.digest)
        real.update(
            hashlib.sha256(table.read_bytes()).hexdigest()
            for table in data_root(demo_dir).glob("*/*.json")
        )

        quoted = {
            hexits
            for demo in demo_dir.glob("*.md")
            for hexits in re.findall(r"digest\s+([0-9a-f]{8,64})", demo.read_text())
        }
        stale = sorted(q for q in quoted if not any(d.startswith(q) for d in real))
        assert not stale, (
            f"{demo_dir.name}: digests {stale} match no document in this demo — "
            "re-paste the explain output"
        )


class TestInlinedDocuments:
    """§4.9 and §5.4 — every document is inlined beside a link to it.

    The copy and the file are both load-bearing. The copy is what a reader on
    GitHub sees without a second click, so a demo whose thesis lives in another
    file is a demo nobody reads; the file is what ``causalab run`` reads, so it
    is the copy that can be wrong. That is two places for one fact, which is
    exactly the arrangement that needs a test rather than a habit — the same
    argument as the digests above, one level up.

    Drift here is worse than never inlining: the prose reads as authoritative
    while the run uses the other bytes, and nothing in the markdown says so.
    A specification's copy carries ``//`` comments (§4.6), so it is compared
    as JSON once the comments are gone; a workflow's copy is compared byte for
    byte (§5.4).
    """

    @pytest.mark.parametrize("document", _documents(), ids=_ids(_documents()))
    def test_is_inlined(self, document: Path) -> None:
        how = (
            "with // comments, the copy minus its comments equal to the file's JSON"
            if document.parent.name == "protocols"
            else "byte for byte"
        )
        assert _carriers(document), (
            f"no markdown in {document.parents[1].name}/ carries {document.name} "
            f"({how}) — inline it in a ```json block (docs/demos.md §4.9, §5.4), "
            "or update the copy if the file has changed"
        )

    @pytest.mark.parametrize("document", _documents(), ids=_ids(_documents()))
    def test_the_inlined_copy_links_the_file(self, document: Path) -> None:
        """A copy with no path beside it is a copy the reader cannot get back
        to, and cannot run."""
        rel = document.relative_to(document.parents[1]).as_posix()
        for demo in _carriers(document):
            assert rel in LINK.findall(demo.read_text()), (
                f"{demo.name} inlines {rel} but never links it"
            )

    def test_comments_are_stripped_and_strings_are_kept(self) -> None:
        """The comparison behind ``test_is_inlined``, on one commented copy:
        a ``//`` after whitespace, one flush against a value, and two inside
        string values that have to survive."""
        fence = (
            "{\n"
            '    "header": {                        // Metadata\n'
            '        "description": "see https://example.org/a//b"\n'
            "    },\n"
            '    "prompt": "a // b",// flush comment\n'
            '    "n": 1                             // the last field\n'
            "}\n"
        )
        assert _as_json(fence) == {
            "header": {"description": "see https://example.org/a//b"},
            "prompt": "a // b",
            "n": 1,
        }

    def test_a_changed_value_is_not_a_copy(self, tmp_path: Path) -> None:
        """The mutation the check exists for: the same fields, one value
        edited in the file and not in the copy."""
        protocols = tmp_path / "protocols"
        protocols.mkdir()
        document = protocols / "x.json"
        document.write_text('{\n  "a": 1,\n  "b": [1, 2]\n}\n')
        commented = '{\n    "a": 1,                     // one\n    "b": [1, 2]\n}\n'
        assert _carries(commented, document)
        document.write_text('{\n  "a": 2,\n  "b": [1, 2]\n}\n')
        assert not _carries(commented, document)
        # a workflow's copy is byte for byte: the commented form is not one
        workflows = tmp_path / "workflows"
        workflows.mkdir()
        workflow = workflows / "w.json"
        workflow.write_text('{\n  "a": 1,\n  "b": [1, 2]\n}\n')
        assert not _carries(commented, workflow)
        assert _carries(workflow.read_text(), workflow)


class TestFigures:
    @pytest.mark.parametrize("demo_dir", demo_dirs(), ids=_ids(demo_dirs()))
    def test_every_figure_is_shown(self, demo_dir: Path) -> None:
        """A figure carries no record (workflow spec §2.5), so an unreferenced
        one is a binary nobody can date or explain. §6.4 asks every figure to
        carry a caption; the cheap half of that is checking it is shown at
        all."""
        shown = "\n".join(p.read_text() for p in demo_dir.glob("*.md"))
        orphans = [
            p.name
            for p in sorted(figures_dir(demo_dir).glob("*"))
            if p.name not in shown
        ]
        assert not orphans, f"{demo_dir.name}: figures never shown {orphans}"


class TestCommittedOutputs:
    """``docs/demos.md`` §5.7: a demo's ``.gitignore`` says which outputs are
    committed, and the tracked set has to agree with it."""

    def test_no_tracked_file_is_ignored(self) -> None:
        """An ignore rule does not untrack a file committed before it, so a
        run tree committed once stays tracked after its demo ignores it.
        ``git ls-files --cached --ignored`` has to come back empty."""
        tracked_and_ignored = subprocess.run(
            ["git", "ls-files", "--cached", "--ignored", "--exclude-standard"],
            cwd=REPO,
            capture_output=True,
            check=True,
            text=True,
        ).stdout.split()
        assert not tracked_and_ignored, (
            f"tracked files match an ignore rule {tracked_and_ignored}: "
            "`git rm --cached` them or change the rule"
        )

    def test_paper_packages_ignore_their_run_trees(self) -> None:
        """Every package runs into ``demos/papers/artifacts/output/``. Without
        that line in the joint ``.gitignore``, every file a run writes is
        committed by the next ``git add``."""
        ignore = DEMOS / "papers" / ".gitignore"
        assert ignore.is_file(), "demos/papers: no .gitignore"
        assert "artifacts/output/" in ignore.read_text().splitlines(), (
            "demos/papers/.gitignore does not ignore artifacts/output/"
        )


class TestIndex:
    def test_every_demo_is_indexed(self) -> None:
        """``demos/README.md`` is where a reader looks first, directly or
        through a track's README that it links, so a demo missing from both
        is a demo nobody finds."""
        index = (DEMOS / "README.md").read_text()
        # A track's own README indexes its demos, if demos/README.md links it.
        for track in demo_dirs():
            readme = track / "README.md"
            if readme.is_file():
                assert f"{track.name}/README.md" in index, (
                    f"demos/README.md does not link {track.name}/README.md"
                )
                index += readme.read_text().replace("](", f"]({track.name}/")
        missing = [
            str(p.relative_to(DEMOS))
            for p in _markdown()
            if str(p.relative_to(DEMOS)) not in index
        ]
        assert not missing, f"no demo index links {missing}"
