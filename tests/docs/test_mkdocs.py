"""The documentation site's build inputs behave as `mkdocs.yml` assumes.

Three scripts stand between the repository and the rendered site:
`scripts/griffe_doc_comments.py`, which teaches griffe to read `#:`
attribute docs; `scripts/gen_ref_pages.py`, which adds the README, the
demos and the API reference pages to the build; and `scripts/mkdocs_links.py`,
which rewrites relative links for the site's layout. All three are exercised
here without running mkdocs; the full build runs in the `Docs` CI workflow.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path, PurePosixPath
from types import ModuleType

import griffe
import pytest

from tests._helpers.mkdocs import mkdocs_config

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[2]
FEATURIZERS = "causalab.protocol.schema.featurizers"
BLOB = "https://example.test/blob/abc/"


def _load_script(name: str) -> ModuleType:
    """Import ``scripts/<name>.py`` as a module, without the ``scripts`` package."""
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def doc_comments() -> ModuleType:
    return _load_script("griffe_doc_comments")


@pytest.fixture(scope="module")
def gen_ref_pages() -> ModuleType:
    return _load_script("gen_ref_pages")


@pytest.fixture(scope="module")
def links() -> ModuleType:
    return _load_script("mkdocs_links")


def _load(extension: griffe.Extension | None) -> griffe.Module:
    exts = griffe.load_extensions(extension) if extension is not None else None
    module = griffe.load(
        FEATURIZERS, search_paths=[REPO], docstring_parser="google", extensions=exts
    )
    assert isinstance(module, griffe.Module)
    return module


# --------------------------------------------------------------------------- #
# the griffe extension
# --------------------------------------------------------------------------- #


def test_plain_griffe_drops_doc_comments() -> None:
    """The reason the extension exists: `#:` docs are invisible to griffe."""
    module = _load(None)
    assert module["GATE_GROUPS"].docstring is None


def test_doc_comment_block_becomes_the_docstring(doc_comments: ModuleType) -> None:
    module = _load(doc_comments.DocComments())
    docstring = module["GATE_GROUPS"].docstring
    assert docstring is not None
    assert docstring.value.startswith("Units that share one gate parameter")
    # the last `#:` line before the assignment is blank, and is trimmed
    assert docstring.value.endswith("group map.")


def test_dataclass_fields_get_their_doc_comments(doc_comments: ModuleType) -> None:
    module = _load(doc_comments.DocComments())
    spec = module["FeaturizerSpec"]
    documented = [
        name
        for name, member in spec.members.items()
        if member.is_attribute and member.docstring
    ]
    assert "kind" in documented and "parametrization" in documented


def test_an_existing_docstring_is_kept(doc_comments: ModuleType) -> None:
    module = _load(doc_comments.DocComments())
    docstring = module["FeaturizerSpec"].docstring
    assert docstring is not None
    assert docstring.value.startswith("§2.5")


# --------------------------------------------------------------------------- #
# the page generator
# --------------------------------------------------------------------------- #


def test_every_public_module_gets_a_reference_page(gen_ref_pages: ModuleType) -> None:
    pages = {
        parts: page for parts, _, page in gen_ref_pages.module_pages(REPO / "causalab")
    }
    featurizers = ("causalab", "protocol", "schema", "featurizers")
    assert pages[featurizers] == PurePosixPath(
        "reference/causalab/protocol/schema/featurizers.md"
    )
    assert pages[("causalab", "protocol")] == PurePosixPath(
        "reference/causalab/protocol/index.md"
    )
    assert not any(parts[-1].startswith("_") for parts in pages)
    assert all("__pycache__" not in parts for parts in pages)


def test_a_directory_without_an_init_is_not_documented(
    gen_ref_pages: ModuleType, tmp_path: Path
) -> None:
    """griffe cannot collect a module in a directory that is not a package."""
    pkg = tmp_path / "pkg"
    (pkg / "sub").mkdir(parents=True)
    (pkg / "loose").mkdir()
    for f in ("__init__.py", "a.py", "sub/__init__.py", "sub/b.py", "loose/c.py"):
        (pkg / f).write_text('"""Doc."""\n')
    parts = {p for p, _, _ in gen_ref_pages.module_pages(pkg)}
    assert parts == {("pkg",), ("pkg", "a"), ("pkg", "sub"), ("pkg", "sub", "b")}


def test_an_empty_init_gets_no_page_but_its_modules_do(
    gen_ref_pages: ModuleType, tmp_path: Path
) -> None:
    """`causalab/__init__.py` is empty; a blank page would be the landing page."""
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "a.py").write_text('"""A."""\n')
    parts = {p for p, _, _ in gen_ref_pages.module_pages(pkg)}
    assert parts == {("pkg", "a")}


def test_landing_page_lists_every_subpackage(gen_ref_pages: ModuleType) -> None:
    rows = {
        name: (summary, page)
        for name, summary, page in gen_ref_pages.package_summaries(REPO / "causalab")
    }
    protocol = rows["causalab.protocol"]
    assert protocol[1] == "causalab/protocol/index.md"
    assert protocol[0]  # the protocol package has a docstring
    assert not any(name.split(".")[-1].startswith("_") for name in rows)
    text = gen_ref_pages.landing_page(REPO / "causalab")
    assert text.startswith("# API reference")
    assert "[`causalab.protocol`](causalab/protocol/index.md)" in text


def test_listed_packages_select_their_modules(gen_ref_pages: ModuleType) -> None:
    parts = {
        p
        for p, _, _ in gen_ref_pages.module_pages(
            REPO / "causalab", ["causalab.protocol.schema"]
        )
    }
    assert ("causalab", "protocol", "schema") in parts
    assert ("causalab", "protocol", "schema", "featurizers") in parts
    # neither the parent package nor a sibling whose name shares the prefix
    assert ("causalab", "protocol") not in parts
    assert all(p[:3] == ("causalab", "protocol", "schema") for p in parts)


def test_reference_nav_gives_each_listed_package_its_own_section(
    gen_ref_pages: ModuleType,
) -> None:
    packages = ["causalab.tasks", "causalab.analysis"]
    nav = gen_ref_pages.reference_nav(REPO / "causalab", packages)
    # listed order, and each section opens on its package's own page
    assert [keys[0] for keys, _ in nav][0] == "causalab.tasks"
    first = {}
    for keys, page in nav:
        first.setdefault(keys[0], (keys, page))
    assert first["causalab.analysis"] == (
        ("causalab.analysis",),
        PurePosixPath("reference/causalab/analysis/index.md"),
    )
    assert set(first) == set(packages)


def test_landing_page_lists_the_configured_packages(gen_ref_pages: ModuleType) -> None:
    text = gen_ref_pages.landing_page(
        REPO / "causalab", ["causalab.protocol.schema", "causalab.causal"]
    )
    assert "[`causalab.protocol.schema`](causalab/protocol/schema/index.md)" in text
    assert "`causalab.neural`" not in text
    assert text.index("causalab.protocol.schema") < text.index("`causalab.causal`")


def test_the_configured_reference_packages_exist() -> None:
    config = mkdocs_config()
    packages = config["extra"]["api_reference"]
    assert packages
    for dotted in packages:
        assert (REPO / Path(*dotted.split(".")) / "__init__.py").is_file(), dotted
    (mkdocstrings,) = [
        p["mkdocstrings"]
        for p in config["plugins"]
        if isinstance(p, dict) and "mkdocstrings" in p
    ]
    (extension,) = mkdocstrings["handlers"]["python"]["options"]["extensions"]
    assert extension == "scripts/griffe_doc_comments.py:DocComments"


def _nav_pages(nav: object) -> set[str]:
    """Every page path in a ``nav`` list from `mkdocs.yml`, at any depth."""
    if isinstance(nav, str):
        return {nav}
    if isinstance(nav, list):
        return set().union(*(_nav_pages(item) for item in nav))
    if isinstance(nav, dict):
        return set().union(*(_nav_pages(value) for value in nav.values()))
    return set()


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_every_demo_page_is_in_the_tutorials_or_the_nav(
    gen_ref_pages: ModuleType,
) -> None:
    """A page no subsection lists and no nav entry names drops out of the site."""
    config = mkdocs_config()
    sections = gen_ref_pages.tutorial_sections(
        REPO / "demos", config["extra"]["tutorials"]
    )
    listed = {"README.md"}
    for _, index, entries in sections:
        listed |= {index} if index else set()
        listed |= {page for _, page in entries}
    listed |= {
        page.removeprefix("demos/")
        for page in _nav_pages(config["nav"])
        if page.startswith("demos/")
    }
    pages = {
        md.relative_to(REPO / "demos").as_posix()
        for md in (REPO / "demos").rglob("*.md")
    }
    assert pages <= listed, sorted(pages - listed)
    # The method library is a how-to guide, not a tutorial.
    tutorial_pages = {page for _, _, entries in sections for _, page in entries}
    assert "methods/README.md" not in tutorial_pages


def test_file_numbering_keeps_file_numbers_and_letters_the_rest(
    gen_ref_pages: ModuleType, tmp_path: Path
) -> None:
    _write(tmp_path / "tut/01_a.md", "# Alpha\n")
    _write(tmp_path / "tut/03_c.md", "# 03 — Gamma\n")
    _write(tmp_path / "extra/x.md", "# Extra\n")
    _write(tmp_path / "extra/y.md", "no heading\n")
    sections = [
        {
            "title": "Onboarding",
            "numbering": "file",
            "pages": ["tut/01_*.md", "extra/*.md", "tut/*.md"],
        }
    ]
    ((title, index, entries),) = gen_ref_pages.tutorial_sections(tmp_path, sections)
    assert (title, index) == ("Onboarding", None)
    assert entries == [
        ("01 — Alpha", "tut/01_a.md"),
        ("01a — Extra", "extra/x.md"),
        ("01b — y", "extra/y.md"),
        ("03 — Gamma", "tut/03_c.md"),
    ]


def test_order_numbering_counts_pages_and_skips_the_index(
    gen_ref_pages: ModuleType, tmp_path: Path
) -> None:
    _write(tmp_path / "papers/README.md", "# Papers\n")
    _write(tmp_path / "papers/b/README.md", "# Bee\n")
    _write(tmp_path / "papers/a/README.md", "# Ay\n")
    sections = [
        {
            "title": "Papers",
            "numbering": "order",
            "index": "papers/README.md",
            "pages": ["papers/*/*.md", "papers/*.md"],
        }
    ]
    ((_, index, entries),) = gen_ref_pages.tutorial_sections(tmp_path, sections)
    assert index == "papers/README.md"
    assert entries == [
        ("01 — Ay", "papers/a/README.md"),
        ("02 — Bee", "papers/b/README.md"),
    ]


def test_a_page_an_earlier_subsection_lists_is_not_listed_again(
    gen_ref_pages: ModuleType, tmp_path: Path
) -> None:
    """A curated subsection can pick pages out of a glob that a later
    subsection keeps, so the nav names each page once."""
    _write(tmp_path / "papers/README.md", "# Papers\n")
    _write(tmp_path / "papers/a.md", "# Ay\n")
    _write(tmp_path / "papers/b.md", "# Bee\n")
    _write(tmp_path / "papers/c.md", "# Cee\n")
    sections = [
        {
            "title": "Series",
            "numbering": "order",
            "pages": ["papers/c.md", "papers/a.md"],
        },
        {
            "title": "Papers",
            "numbering": "order",
            "index": "papers/README.md",
            "pages": ["papers/*.md"],
        },
    ]
    series, papers = gen_ref_pages.tutorial_sections(tmp_path, sections)
    assert series[2] == [("01 — Cee", "papers/c.md"), ("02 — Ay", "papers/a.md")]
    assert papers[2] == [("01 — Bee", "papers/b.md")]


def test_every_tutorial_pattern_matches_a_page() -> None:
    """A glob of a missing literal path matches nothing, so a misspelt or
    renamed entry in a subsection (Series) would drop its page without an
    error: the page stays under Papers and the build does not warn."""
    unmatched = [
        (section["title"], pattern)
        for section in mkdocs_config()["extra"]["tutorials"]
        for pattern in section["pages"]
        if not any((REPO / "demos").glob(pattern))
    ]
    assert not unmatched, f"extra.tutorials patterns that match no page: {unmatched}"


def test_an_unknown_numbering_is_refused(
    gen_ref_pages: ModuleType, tmp_path: Path
) -> None:
    _write(tmp_path / "a.md", "# A\n")
    with pytest.raises(ValueError, match="numbering"):
        gen_ref_pages.tutorial_sections(
            tmp_path, [{"title": "T", "numbering": "alpha", "pages": ["*.md"]}]
        )


def test_demo_summary_opens_on_the_readme_and_nests_each_subsection(
    gen_ref_pages: ModuleType,
) -> None:
    """section-index makes an untitled first page, and a linked section title,
    the section's own page."""
    config = mkdocs_config()
    lines = gen_ref_pages.demo_summary(
        REPO / "demos", config["extra"]["tutorials"]
    ).splitlines()
    assert lines[0] == "* [](README.md)"
    # Each subsection opens on its own landing page.
    assert "* [Onboarding](onboarding_tutorial/README.md)" in lines
    assert "* [Papers](papers/README.md)" in lines
    # The series has no landing page of its own; papers/README.md lists it.
    assert "* Series" in lines
    children = [line for line in lines if line.startswith("    * [")]
    assert children and all(line.split("[", 1)[1][:2].isdigit() for line in children), (
        "every tutorial title starts with its number"
    )
    assert (
        "    * [03a — Saved hypothesis comparisons]"
        "(hypothesis_testing/hypothesis_testing.md)" in lines
    )


def test_opening_sentence_skips_headings_and_drops_link_targets(
    gen_ref_pages: ModuleType, tmp_path: Path
) -> None:
    page = tmp_path / "guide.md"
    _write(
        page,
        "# Guide\n\n<!-- note -->\n\n## 1. Part\n\nThis guide covers the\n"
        "[layout](other.md#x) of `x`. Second sentence.\n\nNext paragraph.\n",
    )
    assert (
        gen_ref_pages.opening_sentence(page) == "This guide covers the layout of `x`."
    )


def test_section_index_lists_each_titled_page_of_the_section(
    gen_ref_pages: ModuleType, tmp_path: Path
) -> None:
    _write(tmp_path / "a.md", "# A\n\nA says what it is for.\n")
    _write(tmp_path / "b.md", "# B\n\nB is for this. More.\n")
    nav = [{"Home": "index.md"}, {"Dev": ["dev.md", {"Aa": "a.md"}, {"Bb": "b.md"}]}]
    children = gen_ref_pages.nav_section(nav, "Dev")
    assert gen_ref_pages.section_index(tmp_path, "Dev", children) == (
        "# Dev\n\n| Guide | Purpose |\n|---|---|\n"
        "| [Aa](a.md) | A says what it is for. |\n"
        "| [Bb](b.md) | B is for this. |\n"
    )
    assert gen_ref_pages.nav_section(nav, "Missing") is None


def test_a_nested_section_gets_its_own_heading_and_table(
    gen_ref_pages: ModuleType, tmp_path: Path
) -> None:
    """A group inside the section is listed under its own name, not dropped."""
    _write(tmp_path / "a.md", "# A\n\nA says what it is for.\n")
    _write(tmp_path / "s.md", "# S\n\nS sets the style.\n")
    _write(tmp_path / "b.md", "# B\n\nB is for this.\n")
    children = ["dev.md", {"Aa": "a.md"}, {"Style": [{"Ss": "s.md"}]}, {"Bb": "b.md"}]
    assert gen_ref_pages.section_index(tmp_path, "Dev", children) == (
        "# Dev\n\n| Guide | Purpose |\n|---|---|\n"
        "| [Aa](a.md) | A says what it is for. |\n"
        "| [Bb](b.md) | B is for this. |\n"
        "\n## Style\n\n| Guide | Purpose |\n|---|---|\n"
        "| [Ss](s.md) | S sets the style. |\n"
    )


def _titled_pages(children: list) -> list[str]:
    """Every titled page under ``children``, nested sections included."""
    out: list[str] = []
    for child in children:
        if isinstance(child, dict):
            for value in child.values():
                out.extend([value] if isinstance(value, str) else _titled_pages(value))
    return out


def test_every_generated_section_index_has_a_purpose_for_each_guide(
    gen_ref_pages: ModuleType,
) -> None:
    """A guide's opening sentence is its row in the generated table."""
    config = mkdocs_config()
    for title, page in config["extra"]["section_index"].items():
        children = gen_ref_pages.nav_section(config["nav"], title)
        assert children is not None, title
        assert page in children, f"{page} must open the {title} section"
        text = gen_ref_pages.section_index(REPO / "docs", title, children)
        rows = [line for line in text.splitlines() if line.startswith("| [")]
        assert len(rows) == len(_titled_pages(children))
        for row in rows:
            purpose = row.rsplit(" | ", 1)[1].removesuffix(" |")
            assert purpose.endswith("."), row


def test_importing_the_generator_does_not_build(gen_ref_pages: ModuleType) -> None:
    """`mkdocs-gen-files` runs the script; a plain import must stay inert."""
    assert callable(gen_ref_pages.main)


# --------------------------------------------------------------------------- #
# the link hook
# --------------------------------------------------------------------------- #


def _rewrite(
    links: ModuleType, text: str, source: str, page: str, existing: set[str]
) -> str:
    return links.rewrite_links(
        text, source, page, exists=lambda p: p in existing, blob_url=BLOB
    )


def test_readme_links_into_docs_lose_the_docs_segment(links: ModuleType) -> None:
    text = "[spec](docs/intervention_protocol.md#s1) and [demo](demos/README.md)"
    out = _rewrite(links, text, "README.md", "index.md", set())
    assert out == "[spec](intervention_protocol.md#s1) and [demo](demos/README.md)"


def test_docs_pages_reach_the_mirrored_demos(links: ModuleType) -> None:
    text = (
        "[demo](../demos/README.md) [tut](../demos/onboarding_tutorial/04_subspace.md)"
    )
    out = _rewrite(
        links, text, "docs/running_experiments.md", "running_experiments.md", set()
    )
    assert (
        out == "[demo](demos/README.md) [tut](demos/onboarding_tutorial/04_subspace.md)"
    )


def test_demo_links_are_rewritten_relative_to_the_mirrored_page(
    links: ModuleType,
) -> None:
    source = "demos/onboarding_tutorial/01_define.md"
    text = "[spec](../../docs/intervention_protocol.md#s2) [own](protocols/x.json) [up](../README.md)"
    out = _rewrite(links, text, source, source, set())
    assert (
        out
        == "[spec](../../intervention_protocol.md#s2) [own](protocols/x.json) [up](../README.md)"
    )


def test_source_files_link_to_github(links: ModuleType) -> None:
    text = "[cli](../causalab/cli.py) [tmpl](../../causalab/configs/protocols/das.json)"
    existing = {"causalab/cli.py", "causalab/configs/protocols/das.json"}
    out = _rewrite(
        links, "[cli](../causalab/cli.py)", "docs/CODEBASE.md", "CODEBASE.md", existing
    )
    assert out == f"[cli]({BLOB}causalab/cli.py)"
    out = _rewrite(
        links, text.split(" ")[1], "docs/methods/das.md", "methods/das.md", existing
    )
    assert out == f"[tmpl]({BLOB}causalab/configs/protocols/das.json)"


def test_a_dead_link_is_left_for_mkdocs_to_report(links: ModuleType) -> None:
    text = "[gone](../causalab/nowhere.py) [page](missing.md)"
    assert _rewrite(links, text, "docs/CODEBASE.md", "CODEBASE.md", set()) == text


def test_urls_anchors_and_escapes_are_left_alone(links: ModuleType) -> None:
    text = "[a](https://x.y/z.md) [b](#anchor) [c](mailto:x@y.z) [d](../../outside.md)"
    assert _rewrite(links, text, "demos/README.md", "demos/README.md", set()) == text


def test_details_blocks_are_marked_for_md_in_html(links: ModuleType) -> None:
    text = "<details>\n<summary>S</summary>\n\n| a |\n|---|\n\n</details>\n"
    assert links.mark_details(text).startswith("<details markdown>\n<summary>")
    kept = '<details markdown="1">\n  <details>\n<details open>\n'
    assert (
        links.mark_details(kept)
        == '<details markdown="1">\n  <details>\n<details markdown open>\n'
    )


def test_reference_pages_are_not_rewritten(links: ModuleType) -> None:
    assert links.repo_path_of("reference/causalab/cli.md") is None
    assert links.repo_path_of("index.md") == "README.md"
    assert links.repo_path_of("methods/das.md") == "docs/methods/das.md"
    assert links.repo_path_of("demos/README.md") == "demos/README.md"
