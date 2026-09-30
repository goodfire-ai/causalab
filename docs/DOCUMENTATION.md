# Documentation maintenance

This guide explains how to build, check and extend the documentation site and
the pages it renders. The [README](../README.md) is the reader's map: it names
the guide to start with for each objective.

## The rendered site

```bash
uv run mkdocs serve
```

This serves the whole repository's documentation at `http://127.0.0.1:8000`.
`uv run mkdocs build` writes it to `site/`, and the result also works opened
as files, search included, so a copy of `site/` needs no server. The
configuration is [mkdocs.yml](../mkdocs.yml).

The `nav` block in `mkdocs.yml` sets the site's tabs:

| Tab | Contents |
|---|---|
| Home | The README |
| Demos | A landing page with two cards, then tutorials in the subsections that `extra.tutorials` lists, each with its own landing page, and how-to guides behind `howtos.md` in four groups: general protocol structure, interp methods, performance and models |
| Development | A generated table of the tab's guides, then the guides themselves, with the three writing guides grouped under Style guides |
| API reference | One page per module in the packages listed under `extra.api_reference` |

A new page under `docs/` needs a `nav` entry, or the build warns. A new page
under `demos/` joins the tutorials when a pattern in `extra.tutorials` matches
it. The subsection's `numbering` sets its number in the sidebar: the number
its file name starts with, or its position. A page that two subsections
match is listed once, in the first of them, so Series takes its pages out of
the Papers glob. A page that no pattern matches needs a `nav` entry of its
own, as the method library has.

The Development tab opens on a table that the build writes from the tab's
`nav` list. Each row is a guide's nav title and the first sentence of its
first paragraph. A nested group, such as Style guides, gets a heading and a
table of its own below the first table. Open a new guide with a sentence that says what it is for.
`extra.section_index` names the generated page.

`docs/` is the only directory mkdocs reads. Three scripts add the rest during
the build. `scripts/gen_ref_pages.py` adds the README as the home page, every
file under `demos/` at its own path, the tutorials' navigation, the
Development table, and one API reference page per module in the packages that
`extra.api_reference` lists. `scripts/mkdocs_links.py` rewrites each page's
relative links from the repository's layout to the site's, and points links to
source files at GitHub. It also adds the `markdown` attribute to each
`<details>` block, so a table inside a dropdown renders on the site as it does
on GitHub. `scripts/griffe_doc_comments.py` teaches the docstring reader
the `#:` attribute docs, which it would otherwise drop.
`tests/docs/test_mkdocs.py` covers all three.

## Maintaining the documentation

The [general writing guide](STYLE_GUIDE.md) covers language, Markdown layout,
code examples, docstrings, generated references, and other documentation
surfaces. The [tutorial guide](demos.md) and the
[paper replication guide](paper_replications.md) add the format of their pages.

Write for researchers who run experiments and contributors who extend the
library. Put the steps needed to complete a task at the start of its guide.
Keep field definitions and implementation contracts in the reference pages.

| Content | Location |
|---|---|
| First experiment | `README.md` and the onboarding tutorial |
| Method instructions | `docs/methods/` |
| Worked research question and evidence | `demos/` |
| Protocol fields and execution semantics | `docs/intervention_protocol.md`, `docs/workflow_protocol.md` |
| Canonical form, engine and runner contracts | `docs/intervention_protocol_internals.md`, `docs/workflow_protocol_internals.md` |
| Package purpose and Python contracts | Package, module, and public symbol docstrings |
| JSON field meaning | Attribute docs in `causalab/protocol/schema/` |
| Module map and shared invariants | `docs/CODEBASE.md` |

Explain a rule where it is defined, then link to it. Remove obsolete plans and
development history when updating a page. Describe the behavior in the current
checkout.

### Generated reference blocks

Method pages mix prose with blocks generated from code. Each block starts with
`<!-- generated: begin ... -->` and names a reader (`doc`, `attrs`, `value` or
`call`) and an object under `causalab`. Edit the source object, then
regenerate:

```bash
uv run python scripts/generate_support_tables.py
uv run python scripts/generate_support_tables.py --check
```

Generated blocks must match their sources byte for byte. A new method template
also needs a link on every method page for the featurizer kinds it uses.

### Checking an edit

```bash
uv run pytest tests/docs
uv run python scripts/generate_support_tables.py --check
uv run mkdocs build
```

`tests/docs/test_docs.py` checks links, quoted paths, documented commands and
JSON examples across every markdown file. `tests/protocol/test_vocabulary_census.py`
checks documented vocabularies against the code. The site build reports a link
to a page that does not exist, and a docstring whose parameter list does not
match its signature.

### Terminology

Use **Desiderata-Based Masking (DBM)** for learning masks with supervision that
specifies the desired behavior. A method that learns a rotation with DAS and a
mask with DBM is **DBM-DAS**. Use **original input** and **counterfactual
input** in scientific explanations. Keep code identifiers exact, including
serialized fields such as `base`.
