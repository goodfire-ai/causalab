"""An mkdocs hook that makes the repository's relative links work on the site.

The markdown in this repository is written for GitHub, where every file sits
in one tree: `docs/demos.md` links to `../demos/README.md`, a method guide
links to `../../causalab/configs/protocols/das.json`, the README links to
`docs/intervention_protocol.md`. The site has a different shape. `docs/` is
its root, the README is its home page, `demos/` is mirrored at the top level
by `scripts/gen_ref_pages.py`, and source files are not served at all.

`on_page_markdown` runs on every page before rendering and rewrites each
relative link once, from the page's location in the repository to the target's
location on the site:

* a target the site serves (anything under `docs/` or `demos/`, and the README)
  becomes a relative link to where the site puts it;
* a target that exists in the repository but is not served (a source file, a
  test fixture, a workflow) becomes a link to the file on GitHub, on the
  public mirror's ``main`` branch;
* a target that exists nowhere is left as written, so `mkdocs` reports it.

The pure function `rewrite_links` is what `tests/docs/test_mkdocs.py`
exercises; the hook only supplies it with the repository's facts.
"""

from __future__ import annotations

import posixpath
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent

#: The branch source links point at on the public mirror named in `mkdocs.yml`.
PUBLIC_BRANCH = "main"

#: Markdown link targets: the part inside ``](...)``. Images share the syntax.
LINK = re.compile(r"\]\(([^)\s]+)\)")

#: A ``<details>`` opening tag at the start of a line, without ``markdown``.
DETAILS = re.compile(r"^<details(?![^>]*\bmarkdown\b)([^>]*)>", re.MULTILINE)

#: A target with a scheme (``https:``, ``mailto:``) or a bare anchor.
EXTERNAL = re.compile(r"^(?:[a-z][a-z0-9+.-]*:|#)")


def served_path(repo_path: str) -> str | None:
    """Where the site serves ``repo_path``, or ``None`` if it does not.

    ``docs/`` is the site root, ``README.md`` is ``index.md``, and ``demos/``
    is mirrored at the same path.
    """
    if repo_path == "README.md":
        return "index.md"
    if repo_path.startswith("docs/"):
        return repo_path[len("docs/") :]
    if repo_path == "demos" or repo_path.startswith("demos/"):
        return repo_path
    return None


def repo_path_of(page_src: str) -> str | None:
    """The repository file a site page was built from, or ``None`` for pages
    that have none (the generated API reference)."""
    if page_src == "index.md":
        return "README.md"
    if page_src.startswith("demos/"):
        return page_src
    if page_src.startswith("reference/"):
        return None
    return f"docs/{page_src}"


def rewrite_links(
    text: str,
    source: str,
    page: str,
    exists: Callable[[str], bool],
    blob_url: str,
) -> str:
    """``text``'s relative links, rewritten for a page at ``page`` on the site.

    ``source`` is the file's path in the repository; ``page`` is its path on
    the site. ``exists`` says whether a repository path is a real file, and
    ``blob_url`` is the GitHub prefix (ending in ``/``) for files the site does
    not serve. External URLs, anchors, and links that escape the repository
    are returned unchanged.
    """
    source_dir = posixpath.dirname(source)
    page_dir = posixpath.dirname(page) or "."

    def sub(match: re.Match[str]) -> str:
        target = match.group(1)
        if EXTERNAL.match(target):
            return match.group(0)
        path, sep, fragment = target.partition("#")
        if not path:
            return match.group(0)
        resolved = posixpath.normpath(posixpath.join(source_dir, path))
        if resolved.startswith("../"):
            return match.group(0)
        served = served_path(resolved)
        if served is not None:
            return f"]({posixpath.relpath(served, page_dir)}{sep}{fragment})"
        if exists(resolved):
            return f"]({blob_url}{resolved}{sep}{fragment})"
        return match.group(0)

    return LINK.sub(sub, text)


def mark_details(text: str) -> str:
    """``text`` with ``markdown`` added to each ``<details>`` tag that lacks it.

    `md_in_html` then parses the block's content as Markdown, so a table or a
    list inside a dropdown renders. A tag that already has the attribute, or
    that does not start its line, is left as written.
    """
    return DETAILS.sub(r"<details markdown\1>", text)


def _blob_url(repo_url: str) -> str:
    """``<repo_url>/blob/<branch>/`` for links to files the site does not serve.

    A branch, not the commit being built: ``repo_url`` is the public mirror,
    whose history is a tree sync of this repository, so an internal commit hash
    would not resolve there. Line anchors in such links may drift as ``main``
    moves; a link to the file stays valid.
    """
    return f"{repo_url.rstrip('/')}/blob/{PUBLIC_BRANCH}/"


def on_page_markdown(markdown: str, page: Any, config: Any, files: Any) -> str:
    """mkdocs hook: mark the page's dropdowns and rewrite its links before rendering."""
    source = repo_path_of(page.file.src_uri)
    if source is None:
        return markdown
    return rewrite_links(
        mark_details(markdown),
        source,
        page.file.src_uri,
        exists=lambda p: (REPO / p).exists(),
        blob_url=_blob_url(config["repo_url"]),
    )
