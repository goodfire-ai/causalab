"""Exercise an installed ``causalab`` on every demo and shipped workflow, from
outside the checkout.

The install checks in ``docs/standalone_install.md`` prove the
wheel imports, the CLI is on PATH and one shipped specification validates. What
they did not exercise is the surface a *user* meets first: a workflow document
with ``{"module": …}`` script locators, whose ``validate`` imports the named
modules' parent packages — which is how an undeclared dependency
(``plotly``, 2026-09) was a clean ``uv run causalab`` from every checkout and a
``ModuleNotFoundError`` from every install. So this runs the two pure verbs,
``validate --data`` and ``explain``, over every demo workflow under ``demos/``
and every shipped method workflow under ``demos/methods/workflows/``, with a
working directory that is neither
the checkout nor a document's directory: whatever passes here does not lean on
a repository root.

Stdlib only, on purpose — it drives the *installed* interpreter through its
``causalab`` executable and must not import the package into its own process::

    python scripts/standalone_smoke.py --bin /tmp/standalone/bin/causalab
    uv run python scripts/standalone_smoke.py --bin .venv/bin/causalab --select mcqa_locate

The data roots are the checkout's: a demo's own tables (``artifacts/data/``,
or ``data/`` in the earlier layout — the rule ``tests/_helpers/demos.py``
states), and the fixture rows for the shipped method workflows — documents
and data are inputs, the install is what is under test. Exit status is the
number of failing verbs.
"""

from __future__ import annotations

import argparse
import dataclasses
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Sequence

REPO = Path(__file__).resolve().parents[1]

#: The two verbs that need no weights, no network and no accelerator.
VERBS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("validate", ("--data", "--engine", "auto")),
    ("explain", ("--engine", "auto")),
)


@dataclasses.dataclass(frozen=True)
class Document:
    """A workflow document and the data root its rows come from."""

    path: Path
    data_root: Path

    def label(self, repo: Path) -> str:
        return str(self.path.relative_to(repo))


@dataclasses.dataclass(frozen=True)
class Outcome:
    document: Document
    verb: str
    returncode: int
    tail: str


def _demo_data_root(demo: Path) -> Path:
    """``artifacts/data`` when the demo keeps an ``artifacts/`` directory, else
    ``data`` — the same rule as ``tests/_helpers/demos.py``, restated here
    because this script must not import the checkout."""
    artifacts = demo / "artifacts"
    return (artifacts if artifacts.is_dir() else demo) / "data"


def discover(repo: Path) -> list[Document]:
    """Every demo workflow beside its demo's tables, then every shipped method
    workflow (``demos/methods/``) over the protocol fixture rows."""
    methods = repo / "demos" / "methods"
    documents = [
        Document(path=path, data_root=_demo_data_root(path.parents[1]))
        for path in sorted((repo / "demos").glob("*/workflows/*.json"))
        if path.parents[1] != methods
    ]
    fixtures = repo / "tests" / "protocol" / "fixtures" / "data"
    documents += [
        Document(path=path, data_root=fixtures)
        for path in sorted((methods / "workflows").glob("*.json"))
    ]
    return documents


def run(
    binary: Path, documents: Sequence[Document], cwd: Path, select: str | None
) -> list[Outcome]:
    outcomes: list[Outcome] = []
    for document in documents:
        if select and select not in str(document.path):
            continue
        for verb, extra in VERBS:
            command = [
                str(binary),
                verb,
                str(document.path),
                "--data-root",
                str(document.data_root),
                *extra,
            ]
            completed = subprocess.run(command, capture_output=True, text=True, cwd=cwd)
            tail = (completed.stdout + completed.stderr).strip().splitlines()
            outcomes.append(
                Outcome(document, verb, completed.returncode, tail[-1] if tail else "")
            )
    return outcomes


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument(
        "--bin",
        type=Path,
        default=Path("causalab"),
        help="the installed `causalab` executable to drive (default: on PATH)",
    )
    parser.add_argument(
        "--repo", type=Path, default=REPO, help="the checkout holding the documents"
    )
    parser.add_argument(
        "--cwd",
        type=Path,
        default=None,
        help="working directory for every verb (default: a fresh temporary one)",
    )
    parser.add_argument(
        "--select", default=None, help="only documents whose path contains this"
    )
    args = parser.parse_args(argv)
    repo = args.repo.resolve()
    documents = discover(repo)
    cwd = args.cwd or Path(tempfile.mkdtemp(prefix="causalab-smoke-"))
    try:
        outcomes = run(args.bin, documents, cwd, args.select)
    except FileNotFoundError as err:
        print(f"cannot run {args.bin}: {err}", file=sys.stderr)
        return 1
    if not outcomes:
        print(f"no document matches --select {args.select!r}", file=sys.stderr)
        return 1
    failures = 0
    for outcome in outcomes:
        status = "OK  " if outcome.returncode == 0 else "FAIL"
        failures += outcome.returncode != 0
        print(
            f"{status} {outcome.verb:8} {outcome.document.label(repo):55} {outcome.tail}"
        )
    documents_seen = len({o.document for o in outcomes})
    print(
        f"{documents_seen} documents, {len(outcomes)} verbs, {failures} failing; "
        f"cwd {cwd}"
    )
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
