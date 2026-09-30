"""Re-pin the digests a demo quotes, and re-inline any document that moved.

A demo is prose wrapped around documents (``docs/demos.md``), and two of the
things it says about them are mechanical: the ``digest <hex>`` values pasted
out of ``causalab explain``/``validate``, and the copy of each document
inlined beside its link — a workflow byte for byte, an intervention
specification with ``//`` comments (§4.6). ``tests/demos/test_demos.py``
checks both,
so any change that moves a digest — editing a demo document, or changing what
the canonical form hashes — turns the demo suite red, and the fix is a hunt
through the hexits every demo quotes.

This script is that hunt, done mechanically.

The hard part is not computing today's digests; it is knowing *which* digest a
stale quotation meant. A demo writes ``digest b9c21dcd…`` with no machine-
readable statement of what it is the digest *of*, and a stale truncated hex
matches nothing, so it cannot be looked up. The mapping therefore runs through
a **baseline**: every digest is computed twice — once for the tree as it was
at ``--baseline`` (default ``HEAD``), once for the working tree — each one
labelled by what produced it (``demos/x/protocols/y.json#document``,
``…#point[3]``, ``…#step[fit]``, ``demos/x/data/d/t.json#content``, …). A
quotation that matches no current digest but does match a baseline one is
rewritten to that label's current digest, truncated to the length the demo
quoted: ``test_quoted_digests_are_current`` matches by prefix, so a demo
quoting 12 hexits has to keep quoting 12.

The baseline tree is exported to a temporary directory and read by a child
process with ``PYTHONPATH`` pointed at it, so the baseline digests come from
the baseline *code* as well as the baseline files. That is what lets the
script serve a change which moves every digest without touching a single demo
file — dtype and quantization entering the interning digests, say.

Usage::

    # Rewrite the demos' markdown in place.
    uv run python scripts/repin_demo_digests.py

    # Say what would change, write nothing, exit 1 if anything would.
    uv run python scripts/repin_demo_digests.py --check

    # Re-pin against something other than HEAD.
    uv run python scripts/repin_demo_digests.py --baseline origin/main

On a clean tree this produces no diff, which is what ``--check`` is for. A
quotation that matches neither tree is left alone and reported: it predates
the baseline, and re-running with an earlier ``--baseline`` is the fix.

A byte-for-byte copy of a document that changed is re-inlined. A commented
copy cannot be: the comments are the author's, so the tool cannot regenerate
them, and a commented copy whose JSON no longer matches its file is reported
for the author to update by hand.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path

# Repo root is two levels up: scripts/repin_demo_digests.py.
REPO_ROOT = Path(__file__).resolve().parents[1]

#: The quotation shape ``tests/demos/test_demos.py`` scans for, split so the
#: hexits can be rewritten without disturbing the word in front of them.
QUOTED = re.compile(r"(digest\s+)([0-9a-f]{8,64})")

#: A fenced ``json`` block — where an inlined document lives. Same pattern as
#: the demo suite's, anchored at line starts so the lazy body stops at the
#: first closing fence.
JSON_FENCE = re.compile(r"^```json\n(.*?)^```$", re.DOTALL | re.MULTILINE)

#: What the baseline export has to carry for its digests to compute: the demos
#: themselves and the package that hashes them.
BASELINE_PATHS = ("demos", "causalab")

#: The two directories under a demo that hold documents (the demo suite's rule).
DOCUMENT_DIRS = ("protocols", "workflows")

#: Directories under ``demos/`` that are not demos, so quote no digests:
#: ``papers/`` (replication packages) and ``methods/`` (the method library,
#: which indexes its documents in one README table; ``tests/demos/
#: test_methods.py`` holds its results files to the documents instead, and two
#: of its documents load the protocol tests' fixture bundles, which no demo
#: tree carries). The same set as ``tests/_helpers/demos.py``.
OTHER_GENRES = ("papers", "methods")

#: Where a document may name a file it *loads* (the IM spec's two load sites).
LOAD_SITES = ("featurizers", "params")


# --------------------------------------------------------------------------- #
# the digests a demo may quote, each one labelled by what produced it
# --------------------------------------------------------------------------- #


def demo_dirs(root: Path) -> list[Path]:
    return sorted(
        p
        for p in (root / "demos").iterdir()
        if p.is_dir() and p.name not in OTHER_GENRES
    )


def data_root(demo_dir: Path) -> Path:
    """``artifacts/data`` when the demo keeps an ``artifacts/`` directory, else
    ``data`` — the rule ``tests/_helpers/demos.py`` states, restated here
    because a baseline export of ``demos/`` and ``causalab/`` has no tests."""
    artifacts = demo_dir / "artifacts"
    return (artifacts if artifacts.is_dir() else demo_dir) / "data"


def strip_json_comments(text: str) -> str:
    """``text`` with every ``//`` comment removed (``docs/demos.md`` §4.6):
    from a ``//`` outside a string to the end of its line. The same function
    as ``tests/demos/test_demos.py``'s, so the checker and this tool agree on
    what a commented copy is."""
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


def is_commented_copy(fence: str, document_text: str) -> bool:
    """Whether ``fence`` is ``document_text`` with ``//`` comments added."""
    if fence == document_text:
        return False
    try:
        return json.loads(strip_json_comments(fence)) == json.loads(document_text)
    except json.JSONDecodeError:
        return False


def documents(demo_dir: Path) -> list[Path]:
    return [
        p for p in sorted(demo_dir.glob("*/*.json")) if p.parent.name in DOCUMENT_DIRS
    ]


def tables(demo_dir: Path) -> list[Path]:
    """The data tables a ref resolves to — every JSON file under the data root."""
    return sorted(data_root(demo_dir).glob("*/*.json"))


def run_tree_load(root: Path, document: Path) -> str | None:
    """The first loaded ``file_path`` that resolves only inside a run tree.

    Mirrors ``tests/demos/test_demos.py::_run_tree_load``. An **apply**
    document — the second half of a fit→apply pair — names its fitted artifact
    by a run-tree path, so it cannot be loaded on its own; the demo suite
    tolerates exactly that, because such a document's digests reach the census
    through the workflow that runs it (``inner_digests`` / ``step_digests``).
    This tool has to tolerate the same shape or it cannot read a demo that has
    one.
    """
    raw = json.loads(document.read_text())
    # the load sites live in the method group (§1); a baseline exported from a
    # pre-v2 revision still carries them at the top level, and this function
    # reads both trees
    method = raw.get("method", raw)
    for section in LOAD_SITES:
        for entry in (method.get(section) or {}).values():
            path = entry.get("file_path") if isinstance(entry, dict) else None
            if isinstance(path, str) and not (root / path).exists():
                return path
    return None


def compute_labels(root: Path) -> dict[str, str]:
    """``label -> digest`` for every digest a demo could legitimately quote.

    The digest *set* mirrors ``test_quoted_digests_are_current``; the labels
    are this script's own addition, and they are what makes a stale quotation
    rewritable rather than merely detectable.
    """
    from causalab.protocol.rules.errors import ValidationError
    from causalab.protocol.pipeline import compile_protocol

    # the sweep is the engine's: the point
    # digests are signed off the compile, by the baseline revision's code when
    # this runs as the baseline child
    from causalab.neural.shared.sweep import step_records
    from causalab.io.env import FileArtifacts, FileDatasets, ResolutionEnv
    from causalab.tasks import TASKS_ROOT
    from causalab.workflow.document import load_workflow

    labels: dict[str, str] = {}
    for demo_dir in demo_dirs(root):
        for document in documents(demo_dir):
            rel = document.relative_to(root).as_posix()
            # a demo carries its own tables, with the shipped task tables
            # behind them as the CLI has it; artifacts resolve against the root
            env = ResolutionEnv(
                datasets=FileDatasets(
                    root=data_root(demo_dir), fallback_roots=(TASKS_ROOT,)
                ),
                artifacts=FileArtifacts(root=root),
            )
            if "steps" in json.loads(document.read_text()):
                workflow = load_workflow(document, env)
                # a step's digest is its document's *with `set` applied*, so it
                # differs from the same document loaded standalone
                for name, digest in workflow.inner_digests.items():
                    labels[f"{rel}#inner[{name}]"] = digest
                for name, digest in workflow.step_digests.items():
                    labels[f"{rel}#step[{name}]"] = digest
            else:
                try:
                    compiled = compile_protocol(document, env=env)
                except ValidationError as error:
                    # `rule` and `path` on an aggregate report its first
                    # violation only, so every violation has to be the one
                    violations = getattr(error, "errors", (error,))
                    if all(v.rule == 4 and v.path == "model.key" for v in violations):
                        # this tree's registry cannot size the model, so the
                        # document has no digest here: a registry row added
                        # after the baseline leaves the baseline unable to
                        # read the document, and that is not a reason to
                        # abort the run. The demo suite holds the working
                        # tree to the stricter rule.
                        print(
                            f"{rel}: {error} — no digests from this tree",
                            file=sys.stderr,
                        )
                        continue
                    # the apply-document exception, as the demo suite has it
                    if error.rule != 15 or not run_tree_load(root, document):
                        raise
                    continue
                labels[f"{rel}#document"] = compiled.digests.document
                digests = [s.digest for s in step_records(compiled, env)]
                for index, digest in enumerate(digests):
                    labels[f"{rel}#point[{index}]"] = digest
        for table in tables(demo_dir):
            rel = table.relative_to(root).as_posix()
            labels[f"{rel}#content"] = hashlib.sha256(table.read_bytes()).hexdigest()
    return labels


def document_bytes(root: Path) -> dict[str, str]:
    """``repo-relative path -> text`` for every demo document under ``root``."""
    return {
        document.relative_to(root).as_posix(): document.read_text()
        for demo_dir in demo_dirs(root)
        for document in documents(demo_dir)
    }


# --------------------------------------------------------------------------- #
# the baseline: the same two maps, from another revision's files *and* code
# --------------------------------------------------------------------------- #


def export_revision(rev: str, destination: Path, repo: Path = REPO_ROOT) -> None:
    """Check ``rev``'s demos and package out into ``destination``.

    ``git archive`` rather than a worktree: this needs a few megabytes of two
    directories, not a second checkout of a 1.7 GB tree.
    """
    archive = destination.parent / f"{destination.name}.tar"
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "archive",
            "--format=tar",
            "--output",
            str(archive),
            rev,
            "--",
            *BASELINE_PATHS,
        ],
        check=True,
    )
    destination.mkdir(parents=True, exist_ok=True)
    subprocess.run(["tar", "-xf", str(archive), "-C", str(destination)], check=True)
    archive.unlink()


def labels_of_tree(root: Path) -> dict[str, str]:
    """``compute_labels(root)`` run by *that tree's* code, in a child process.

    ``PYTHONPATH`` precedes site-packages, so the child's ``causalab`` is the
    one under ``root`` and not the installed one. The child re-checks that
    before it hashes anything.
    """
    env = dict(os.environ, PYTHONPATH=str(root))
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--emit-labels",
            "--root",
            str(root),
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    if completed.returncode != 0:
        raise SystemExit(
            f"reading the baseline at {root} failed:\n{completed.stderr.strip()}"
        )
    return json.loads(completed.stdout)


# --------------------------------------------------------------------------- #
# rewriting
# --------------------------------------------------------------------------- #


def _protected_spans(text: str, verbatim: set[str]) -> list[tuple[int, int]]:
    """The fenced blocks that hold a document — byte for byte, or with ``//``
    comments over the same JSON.

    Those copies are checked against the file (``test_is_inlined``), so a
    digest that happens to appear inside one is the document's own content
    and must not be rewritten.
    """
    return [
        match.span(1)
        for match in JSON_FENCE.finditer(text)
        if match.group(1) in verbatim
        or any(is_commented_copy(match.group(1), copy) for copy in verbatim)
    ]


def _segments(text: str, spans: list[tuple[int, int]]) -> Iterator[tuple[str, bool]]:
    """``text`` split into ``(chunk, protected)`` pieces, in order."""
    cursor = 0
    for start, end in sorted(spans):
        if start > cursor:
            yield text[cursor:start], False
        yield text[start:end], True
        cursor = end
    yield text[cursor:], False


def repin_quotations(
    text: str,
    *,
    current: dict[str, str],
    baseline: dict[str, str],
    verbatim: set[str],
) -> tuple[str, list[str], list[str]]:
    """Rewrite this demo's stale quotations. Returns ``(text, done, stuck)``.

    ``current`` and ``baseline`` are the label→digest maps restricted to the
    demo the markdown belongs to: the suite checks a quotation against its own
    demo's digests, and restricting the lookup keeps two demos that happen to
    share a prefix from resolving to each other.
    """
    live = set(current.values())
    done: list[str] = []
    stuck: list[str] = []

    def rewrite(match: re.Match[str]) -> str:
        word, hexits = match.group(1), match.group(2)
        if any(digest.startswith(hexits) for digest in live):
            return match.group(0)
        labels = [
            label for label, digest in baseline.items() if digest.startswith(hexits)
        ]
        moved = {current[label] for label in labels if label in current}
        if len(moved) != 1:
            stuck.append(hexits)
            return match.group(0)
        replacement = moved.pop()[: len(hexits)]
        done.append(f"{hexits} -> {replacement}  ({', '.join(sorted(labels))})")
        return word + replacement

    rebuilt = "".join(
        chunk if protected else QUOTED.sub(rewrite, chunk)
        for chunk, protected in _segments(text, _protected_spans(text, verbatim))
    )
    return rebuilt, done, stuck


def reinline_documents(
    text: str, *, current: dict[str, str], baseline: dict[str, str]
) -> tuple[str, list[str]]:
    """Replace each document's byte-for-byte baseline copy with the bytes it
    has today."""
    done: list[str] = []
    for rel, was in baseline.items():
        now = current.get(rel)
        if now is None or now == was or was not in text:
            continue
        text = text.replace(was, now)
        done.append(rel)
    return text, done


def stale_commented_copies(
    text: str, *, current: dict[str, str], baseline: dict[str, str]
) -> list[str]:
    """Documents whose commented copy in ``text`` is the baseline's JSON and
    not today's. The tool cannot rewrite these — the comments are the
    author's — so it names them."""
    stale: list[str] = []
    for rel, was in baseline.items():
        now = current.get(rel)
        if now is None or now == was:
            continue
        for fence in JSON_FENCE.findall(text):
            if is_commented_copy(fence, was) and not is_commented_copy(fence, now):
                stale.append(rel)
                break
    return stale


def repin(
    root: Path,
    *,
    current_labels: dict[str, str],
    baseline_labels: dict[str, str],
    current_documents: dict[str, str],
    baseline_documents: dict[str, str],
    write: bool,
) -> tuple[list[str], list[str]]:
    """Re-pin every demo under ``root``. Returns ``(changed files, warnings)``."""
    changed: list[str] = []
    warnings: list[str] = []
    for demo_dir in demo_dirs(root):
        prefix = f"{demo_dir.relative_to(root).as_posix()}/"
        current = {k: v for k, v in current_labels.items() if k.startswith(prefix)}
        baseline = {k: v for k, v in baseline_labels.items() if k.startswith(prefix)}
        verbatim = {
            text
            for source in (current_documents, baseline_documents)
            for rel, text in source.items()
            if rel.startswith(prefix)
        }
        for markdown in sorted(demo_dir.glob("*.md")):
            before = markdown.read_text()
            text, inlined = reinline_documents(
                before,
                current={
                    k: v for k, v in current_documents.items() if k.startswith(prefix)
                },
                baseline={
                    k: v for k, v in baseline_documents.items() if k.startswith(prefix)
                },
            )
            text, repinned, stuck = repin_quotations(
                text, current=current, baseline=baseline, verbatim=verbatim
            )
            rel = markdown.relative_to(root).as_posix()
            for document in stale_commented_copies(
                text,
                current={
                    k: v for k, v in current_documents.items() if k.startswith(prefix)
                },
                baseline={
                    k: v for k, v in baseline_documents.items() if k.startswith(prefix)
                },
            ):
                warnings.append(
                    f"{rel}: the commented copy of {document} is the baseline's "
                    "JSON, not the file's — update the copy by hand (the comments "
                    "are yours to keep)"
                )
            for message in inlined:
                print(f"{rel}: re-inlined {message}")
            for message in repinned:
                print(f"{rel}: {message}")
            for hexits in stuck:
                warnings.append(
                    f"{rel}: quoted digest {hexits} matches neither the working "
                    "tree nor the baseline — re-run with an earlier --baseline, "
                    "or re-paste the block"
                )
            if text != before:
                changed.append(rel)
                if write:
                    markdown.write_text(text)
    return changed, warnings


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--root",
        type=Path,
        default=REPO_ROOT,
        help="tree to rewrite (default: the repo)",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--baseline", default="HEAD", help="revision the demos were last pinned against"
    )
    source.add_argument(
        "--baseline-root",
        type=Path,
        help="an already-checked-out baseline tree, instead of exporting a revision",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="write nothing; exit 1 if any demo would change",
    )
    parser.add_argument(
        "--emit-labels",
        action="store_true",
        help=argparse.SUPPRESS,  # internal: how the baseline child reports
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    root = args.root.resolve()

    if args.emit_labels:
        import causalab

        # resolved on both sides: a temporary directory is reached through a
        # symlink on macOS, and ``causalab.__file__`` records the way in
        package = Path(causalab.__file__).resolve()
        if root.resolve() not in package.parents:
            raise SystemExit(
                f"--emit-labels: imported causalab from {package}, not from {root}"
            )
        json.dump(compute_labels(root), sys.stdout)
        return 0

    scratch: Path | None = None
    try:
        if args.baseline_root is not None:
            baseline_root = args.baseline_root.resolve()
        else:
            scratch = Path(tempfile.mkdtemp(prefix="causalab-repin-"))
            baseline_root = scratch / "baseline"
            export_revision(args.baseline, baseline_root, repo=root)
        baseline_labels = labels_of_tree(baseline_root)
        baseline_documents = document_bytes(baseline_root)
    finally:
        if scratch is not None:
            shutil.rmtree(scratch, ignore_errors=True)

    changed, warnings = repin(
        root,
        current_labels=compute_labels(root),
        baseline_labels=baseline_labels,
        current_documents=document_bytes(root),
        baseline_documents=baseline_documents,
        write=not args.check,
    )

    for warning in warnings:
        print(warning, file=sys.stderr)
    if not changed:
        print("demos are pinned; nothing to do")
        return 1 if warnings else 0
    verb = "would change" if args.check else "re-pinned"
    print(f"{verb}: {', '.join(changed)}")
    return 1 if (args.check or warnings) else 0


if __name__ == "__main__":
    raise SystemExit(main())
