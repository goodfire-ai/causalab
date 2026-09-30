# General writing guide

This guide sets the rules for language, layout, examples and docstrings on
every documentation surface.

Write so a reader can find an answer, take the next step, and understand the
limits of the result. Prefer direct language, visible structure, and examples
that work. Keep the detail needed to use or change the system correctly.

This guide records the editorial approach used in the documentation
rewrite and the method-library README edit.
It turns those choices into reusable guidance; existing files may still need
work to follow it. A reduction in word count is a useful outcome, not a quota.

The [documentation maintenance guide](DOCUMENTATION.md) explains where content belongs.
The [tutorial guide](demos.md) and the [paper replication guide](paper_replications.md)
set the required structure for their respective documents. Follow those
formats while applying the language rules here.

| Surface | Guidance |
|---|---|
| All explanatory text | [Language](#language), [terminology](#terminology-and-precision) |
| Markdown pages | [Layout](#markdown-layout), [page types](#structure-by-page-type) |
| Commands and examples | [Code blocks](#code-blocks-and-inline-code) |
| Research results and benchmarks | [Evidence](#results-figures-and-evidence) |
| Python modules, classes, and functions | [Docstrings](#python-docstrings) |
| Schema fields and generated references | [Source documentation](#attribute-docs-and-generated-references) |
| Code and configuration comments | [Comments](#comments-and-configuration) |
| CLI help, errors, and logs | [Runtime text](#cli-help-errors-and-logs) |
| Tests and snapshots | [Test documentation](#test-docstrings-and-snapshots) |
| PRs, commits, and operational guidance | [Contributor documents](#prs-commits-and-operational-guidance) |
| Editing and verification | [Workflow](#editing-and-verification) |

## Language

### Lead with the useful fact

Open with what the thing does, what the reader should do, or what the evidence
shows. Introduce background when it explains that fact. A page should not
require a reader to reconstruct the development process before using the
current implementation.

**Before:** “One intervention specification per method. Each document is the
smallest complete application of one method to a shipped task, written to be
copied from.”

**After:** “Copy a specification from the table below to try a method on a
shipped task.”

The revision gives the reader an action and a destination. The linked
specification can supply the details of the experiment.

### Give each sentence a clear job

Use concrete subjects and active verbs: “The loader checks the dtype,”
“Evaluate the saved mask on held-out pairs,” or “The scan selected block 23.”
Use passive voice when the actor is irrelevant and the result deserves focus:
“Nonfinite values are saved as JSON null.”

A sentence usually states one behavior, instruction, condition, or consequence.
Keep a condition next to the behavior it qualifies. Split a sentence when it
accumulates unrelated clauses, several parenthetical explanations, or a second
argument after a dash.

**Before:** “The apply half of a fit-and-apply pair loads the fit's bundle by a
run-tree path and runs after the fit.”

**After:** “Run fits before the documents that apply their saved bundles.”

Keep the path rule as a separate sentence when readers must supply that path.
Shortening an instruction must not remove a prerequisite.

### Prefer familiar words and precise verbs

| Prefer | Usually replace |
|---|---|
| use | utilize, leverage |
| run | execute the invocation of |
| check | perform a verification of |
| save | persist to disk, when persistence is not the distinction |
| before, after | prior to, subsequent to |
| because | due to the fact that |
| requires | necessitates the presence of |
| “The mask retained 8 heads.” | “The resulting mask exhibits retention of 8 heads.” |

Technical terms remain useful when they name a real distinction. Keep
“canonical form,” “orthonormal,” or “counterfactual” where that meaning matters.
Explain an unfamiliar term at its first useful appearance rather than replacing
it with a less accurate everyday word.

### State definitions directly

Define the thing before discussing nearby alternatives.

**Before:** “This is not a seven-section demo; it is one intervention
specification per method.”

**After:** “The method library contains runnable specifications indexed by
method.”

Use a comparison when the distinction changes a decision. For example,
“Clean accuracy measures the unmodified model; interchange accuracy measures
the patched model against the counterfactual answer” distinguishes two metrics
that readers could otherwise confuse.

Avoid habitual “not X, but Y” framing, rhetorical questions, and claims that an
approach is “simple,” “obvious,” “robust,” or “powerful.” Name the property or
measurement instead. Replace “the gate never committed” with its diagnostic,
such as “The gate's decisive fraction was 0.0,” and explain the consequence.

### Keep rationale; remove the diary

Preserve why a constraint exists when that reason helps someone avoid a mistake.
Remove obsolete plans, resolved review discussions, migration inventories,
and accounts of how many attempts preceded the current design.

**Before:** “The formatter is pinned because an upstream release turned a
green branch red during the earlier refactor.”

**After:** “Pin the formatter so local checks and CI produce the same output.”

Keep history when it remains operationally necessary: a supported migration,
an unresolved upstream bug, a compatibility boundary, or the provenance of a
measurement. Give it a short explanation and a durable reference.

### Connect paragraphs

Each paragraph develops one point. Its opening sentence states the point;
subsequent sentences explain the condition, mechanism, or consequence. Put the
reader's next action after the information needed to choose it.

A short paragraph often needs two to four sentences. This is a useful default,
not a maximum. Avoid both a wall of text and a series of disconnected
single-sentence paragraphs. Remove conclusions that merely repeat the opening.

## Terminology and precision

Use the same name for the same concept across guides, code descriptions, and
results. Follow the repository's domain vocabulary:

- **Distributed alignment search (DAS)** learns a basis for interchange
  interventions. Expand the abbreviation at first use in a standalone guide.
- **Desiderata-Based Masking (DBM)** learns a mask using supervision that
  specifies the desired behavior.
- **DBM-DAS** combines a learned mask with a basis learned by DAS. Name the
  gate parametrization when it matters, such as a boundary gate. Retain a
  recognizable method name such as Boundless DAS where it helps identification.
- Use **original input** and **counterfactual input** in scientific explanations.
  Preserve `base` when referring to that exact serialized role or identifier.
- Distinguish the intervention procedure from the causal hypothesis it tests.
  A successful intervention supports a claim under the stated conditions.

Never rewrite a field name, flag, enum value, import path, exception code, or
filename merely to improve its English. Explain the identifier in prose.
Use inline code for literal names and ordinary text for their meaning.

Use **must** or **requires** for enforced requirements, **should** for a
recommendation with possible exceptions, and **may** for permission. Use **can**
for capability. Avoid “always,” “never,” “every,” and “only” unless the scope
and implementation support them.

Keep distinctions between absent, empty, zero, `false`, and `null`. In field
descriptions, say what happens when a field is omitted. Do not replace “uses
the model's dtype” with “uses the default” if the default has several owners.

Use sentence case for headings and labels. Preserve proper names and code
capitalization. Prefer full stops to chains of semicolons and em dashes.
Parentheses work for a brief definition or unit; put substantive conditions in
the sentence. Hyphenate established terms where useful, such as “held-out,”
but avoid inventing compound labels for ordinary actions.

## Markdown layout

### Headings and navigation

Use one `#` title naming the subject. Organize major topics with `##` and
subtopics with `###`; do not skip levels. A subsection should contain a
coherent topic, not just exist to give every paragraph a heading.

Choose headings a reader would look for: “Training,” “Applying a fitted
subspace,” “Results,” or “Rerunning.” Avoid promotional headings and vague
labels such as “Some important things.” Use numbered headings when a reference
already has stable section numbers or an ordered tutorial needs them.

Long references benefit from a short contents list or a table mapping tasks to
sections. Short guides can rely on their headings. Preserve existing anchors
when changing titles, or update all incoming links. An explicit HTML anchor
can retain a widely used old target.

### Paragraphs, lists, and whitespace

Put blank lines between paragraphs and around headings, lists, tables, and
fenced blocks. Wrap source prose at roughly 80 characters where convenient.
Keep URLs, code, and table rows intact rather than breaking their syntax to
meet a width target.

Use paragraphs for explanation, numbered lists for an ordered procedure, and
bullets for independent alternatives or requirements. Start parallel list items
with parallel grammar. Use complete sentences for instructions and consistent
punctuation within a list. Keep nesting shallow; promote a complicated branch
to a subsection.

Avoid manual line breaks, repeated horizontal rules, and HTML used only for
spacing. A page should read well in both the repository and the rendered site.

### Tables

Use a table when readers need to compare entries along the same dimensions:
methods, fields, allowed combinations, resource requirements, or results.
Make each column answer one question. State shared assumptions immediately
before the table instead of repeating them in every cell.

Keep cells compact. Put a procedure or a paragraph of qualifications below the
table and link to it when necessary. Split an unwieldy table by a meaningful
category, while retaining the fields readers need to compare entries.

Use explicit values such as “none,” “not measured,” or “not applicable” when
an empty cell would be ambiguous. Keep those meanings distinct. Put units in
headers when all entries share them; otherwise label values individually.

In an index, link the method or document name to its source and the reported
result to its artifact. Keep identifiers exact so readers can locate files.
An index row should describe the use and observation without claiming more
than the measurement establishes.

### Emphasis, links, and expandable material

Use **bold** for a small number of important findings or labels. Use inline
code for literal syntax. Avoid bolding entire paragraphs or using italics to
carry essential qualifications.

Choose link text that names its destination: “gate field table,” “saved
results,” or “testing guide.” Use paths relative to the Markdown file for
repository links. Link to the section that defines a rule instead of copying
its full explanation into each caller's guide.

Use expandable blocks for optional derivations or repeated configuration when
the renderer supports them. Keep prerequisites, essential commands, findings,
and interpretation limits visible without expansion.

### Diagrams

Leave colour out of a Mermaid diagram. The site's theme colours the nodes, the
edges and the label text for the light and the dark scheme. A fixed `fill` or
`color` in a `classDef` or `style` line keeps its value in both schemes while
the theme changes the text colour, so one of the two schemes loses contrast.
Show a difference between kinds of node with shape or stroke width, as the
pipeline diagram in the
[intervention protocol internals](intervention_protocol_internals.md) does.
`tests/docs/test_docs.py` refuses a colour in a Mermaid fence.

## Structure by page type

These are starting structures. Omit a section that has no useful content,
except where an existing repository format requires it.

| Page | Recommended order |
|---|---|
| Repository README | Purpose and audience; installation; first working example; links to guides |
| Task guide | Outcome; prerequisites; steps; expected output; relevant failure recovery |
| Method guide | Method and hypothesis; sites and fields; fitting; applying; validation; templates |
| Method library index | First command; shared model/data assumptions; document/result table; evidence limits; rerunning |
| Reference specification | Scope and navigation; definitions; field and execution rules; validation; examples |
| Architecture guide | Responsibilities and dependency boundaries; entry points; invariants; extension points |
| Benchmark report | Question; hardware and workload; procedure; measurements and controls; limits; reproduction |

A method guide explains how to use a method. A library index helps readers
choose a runnable example. Keep the full field contract in the reference or
its generated table, and link to it from either page.

A worked demo follows the [tutorial guide](demos.md): question and method table,
research question, method, execution, results, and useful next steps. Preserve
the question-to-answer structure. A method-library entry does not need to
repeat a full research narrative.

Reference prose should state conditions and outcomes precisely. For a field,
cover meaning, default, allowed values, dependencies, and failure conditions
where applicable. Keep an exhaustive vocabulary in its maintained source;
architecture tables may instead select useful entry points if labeled as such.

## Code blocks and inline code

### Choose the right presentation

Use inline code for a name or short expression inside a sentence. Use a fenced
block when line breaks, indentation, several fields, or a complete command
matter. Give each fence an accurate language label: `bash`, `python`, `json`,
`yaml`, `toml`, `diff`, `markdown`, or `text`.

Use `text` for terminal output and deliberately non-executable sketches. Label
an excerpt or schematic example in the preceding sentence. A language label
helps rendering; it does not establish that a snippet is complete or runnable.

Introduce a block by explaining its purpose. Follow it with the important
outcome or the meaning of unfamiliar parts. Avoid narrating every obvious line.
If a long example hides the teaching point, show a clearly labeled excerpt
and link the complete file.

### Shell commands

State the working directory and required environment before the command.
Repository development commands use `uv run` from the repository root.
Keep copyable commands free of prompt characters and terminal output.

A command with at most one flag stays on one line. A command with two or more
flags puts the command and its positional arguments on the first line, then
each flag with its value on its own continuation line, indented four spaces:

```bash
uv run causalab run demos/methods/protocols/minimal_cpu.json \
    --engine auto \
    --data-root tests/protocol/fixtures/data \
    --out runs/minimal_cpu
```

A backslash must be the last character on the line. Avoid comments after it.
`tests/docs/test_docs.py` checks the flag layout in every `bash` and `sh`
fence. Explain flags on first use; later examples can link that explanation.

Show output separately. Include only the lines needed to recognize success,
understand the result, or diagnose a problem. Label illustrative output as
illustrative. Exact quoted output must come from the stated version and run.

Prefer a real shipped file when the reader can run it. When substitution is
necessary, identify each placeholder and its expected value before the block.
Angle-bracket placeholders are notation and must be replaced before pasting
into a shell. Do not present a placeholder-filled command as ready to run.

Keep a copyable block focused on one intended action. Identify commands that
change or delete files and their scope. Never include credentials or personal
machine paths. Document resource requirements when executing the example
loads weights, uses an accelerator, or performs a costly run.

### Configuration examples

A complete example must parse and satisfy the contract it claims to illustrate.
Keep JSON valid: double-quoted keys and strings, no trailing commas, and no
comments. JSON fragments should still be valid objects when practical.

For annotated excerpts, use a suitable label such as `jsonc` or `text`, explain
that comments must be removed for JSON input, and link the runnable document.
The [tutorial guide](demos.md) permits annotated specifications; a complete inline
copy must match the linked JSON after comments are removed. Existing tooling
may accept annotated blocks, but label new examples honestly.

This fragment sets a subspace's rank and parametrization; it is not a complete
intervention specification:

```json
{
  "kind": "subspace",
  "k": 8,
  "parametrization": "cayley"
}
```

Do not use `...` inside a block advertised as runnable JSON. Use a labeled
excerpt, a `diff` showing an edit, or a complete linked template. Explain what
a reader must replace and which surrounding structure is omitted.

### Python and mathematical examples

Include imports and setup for a runnable Python example. Show output only
when it teaches the behavior. Use doctest prompts only for examples intended
as an interactive transcript or maintained as doctests.

Keep pseudocode visibly separate from executable Python. Avoid placeholder
functions in an example described as ready to run. Keep tensor shapes, axes,
units, and indexing conventions when they are needed to interpret a formula.
Use inline notation for a small expression; use the site's established math
format for a derivation. Do not use a code block merely to frame prose.

When demonstrating Markdown fences, use a longer outer fence or another
supported delimiter. Check the rendered nesting. Repository documentation
checks may recognize fewer fence forms than the renderer, so inspect coverage
rather than assuming every nested example is automatically validated.

## Results, figures, and evidence

State the observation first, then interpret it within the experiment's scope.
Preserve the conditions that make a number meaningful: model and revision,
data split, sample count, site or intervention, precision, metric, units,
and relevant baseline. Put shared conditions near a table and link the full
run receipt for details.

Distinguish a training score, a score used to select a fit, and a final
held-out evaluation. Keep selection and evaluation roles explicit when the
same split participates in both. Do not imply generalization from a fitted
training result or independence from a repeatedly consulted test split.

For intervention metrics, define which answer is scored and which model state
produces the logits. A logit difference of 0.87 is not an accuracy of 0.87.
State whether an aggregate is over pairs, prompts, eligible rows, cells, or
seeds. Record exclusions and uncertainty when relevant.

Report measured outcomes even when the method fails. Prefer “The patch left a
negative answer-logit difference at these positions” to “These positions carry
nothing.” A limited intervention does not establish the absence of a
representation. A fitted mask with poor diagnostics does not become convincing
because its score is positive.

Use enough digits to support the comparison, with full precision available in
the artifact. Preserve the source's units. Follow the demo convention of
fractions for accuracy; do not mix fractions and percentages in a comparison
without explicit labels.

A figure needs a caption naming the quantity, axes, conditions, and source
artifact. Use semantic token labels when they explain the experiment better
than raw indices. Keep indices in the specification or a key. Choose a figure
for a pattern and a table for exact comparisons; avoid reproducing the same
result in several formats without a reason.

For performance claims, include hardware, workload, precision, cache state,
and what the timing includes. Distinguish a measured memory use from a
recommended device capacity. Keep cold and warm timings separate. State which
comparison supports a claimed improvement.

Limitations belong beside the claims they qualify. A short shared limitations
section can collect issues that affect an entire library, such as a small
training set or weak clean accuracy. Keep future experiments separate from
observations and describe the question each would resolve.

## Python docstrings

### General form

The first sentence states the purpose or contract and ends with a full stop.
Use an imperative for an operation (“Read a metric table”) and a descriptive
phrase for an object (“A saved intervention result”). Put a blank line between
the summary and additional detail. Use triple double quotes.

Write enough for callers to use the object correctly without tracing its
implementation. Describe semantic constraints beyond the signature:
shapes, units, mutation, ownership, ordering, state, side effects, failure
conditions, and resource use when applicable.

Use Google-style sections such as `Args:`, `Returns:`, `Raises:`, and
`Examples:`. The API reference parses every docstring as Google style
(`docstring_style` in [mkdocs.yml](../mkdocs.yml)), so other section syntax
renders as plain text. Put types in the signature. Write a type in the
docstring only when the signature has none: `name (type):` for a parameter
and `(type):` for a return value. The parentheses matter in `Returns:`,
where a bare `type:` reads as the name of the returned value. Mark a
deprecated object with a `Deprecated:` section, which the reference renders as
an admonition, and show code in it with a fenced block. The reference shows
reStructuredText directives such as `.. deprecated::` as plain text.

### Package and module introductions

Start with the responsibility. Add the public entry points and boundaries a
reader needs to navigate the module. Preserve dependency constraints when
they matter, such as importing tensor libraries only when a run requires them.

Avoid a full inventory of every helper, a retelling of a refactor, or a
step-by-step description of implementation already visible below. A complex
module can justify a longer introduction when it has process-wide state,
concurrency guarantees, or important security assumptions.

An example based on the table I/O module:

```python
"""Read and write metric tables as JSON arrays of row objects.

Writes preserve row order and convert nonfinite floats to JSON null.
"""
```

### Functions and methods

A one-line docstring is sufficient for a straightforward operation. Add
parameter, return, exception, or example sections only when they supply useful
information. Do not restate a type annotation as a parameter description.

Explain behavior callers cannot infer from the name: whether a path's parent
is created, whether a value is copied, whether an iterator is consumed, whether
input order survives, or when an error is raised. Document exceptions that
callers are expected to handle rather than every incidental failure an
implementation dependency could emit.

For example, this illustrative Google-style docstring explains the behavior
of a writer whose signature already names `target` and `rows`:

```python
"""Write rows as a JSON metric table.

Creates the parent directory if needed and preserves row order.
Nonfinite float values are stored as null.

Args:
    target: Destination file, replaced if it already exists.
    rows: Metric records to serialize.
"""
```

### Cross-references

Link to a public object under `causalab/` with an autorefs link. When the
name is defined or imported in the same module, write it on its own:

```python
"""Return the [`Rule`][] a number or slug names."""
```

The docs build looks the name up where the docstring sits: the object's own
members, then its module's members and imports (`scoped_crossrefs` in
[mkdocs.yml](../mkdocs.yml)). A dotted name such as ``[`ProtocolError.code`][]``
works the same way through its first segment. For any other target, write
the full path:

```python
"""Write rows with [`serialize_examples`][causalab.tasks.serialize.serialize_examples]."""
```

Write a private object, an external object, or code outside `causalab/` as
inline code, because the site has no page to link to. The API reference
renders only the packages that `extra.api_reference` in
[mkdocs.yml](../mkdocs.yml) lists. A docstring in one of those packages writes
a target in any other package as inline code with its full path, such as
`` `causalab.protocol.pipeline.check_engine` ``. Do not use Sphinx roles
such as `:class:` or `:func:`; the docs build does not read them.
`tests/docs/test_docstring_format.py` refuses roles, checks that every link
in the package resolves to a public object, and checks that every link in a
rendered docstring targets a rendered package.

### Classes and dataclasses

Explain what an instance represents, its lifecycle, and any invariants a
caller must preserve. Document immutability or shared state when it affects
use. Put field-specific meaning beside the field instead of maintaining a
second exhaustive field list in the class introduction.

Keep a class docstring concise when the fields already explain the record.
An abstraction that owns resources or coordinates execution needs its
ownership and cleanup rules as well.

## Attribute docs and generated references

Document a module constant or a class field with a `#:` block immediately
above it, not with an `Attributes:` list in the class docstring. The block
documents the constant or field immediately below it. Keep the block adjacent
to the assignment. Use a blank `#:` line for a paragraph break; an ordinary
blank line can detach the block from the field in the documentation reader.

Write these descriptions as public reference text because they may appear in
several generated pages. Start with meaning, then add defaults, allowed
values, dependencies, and validation conditions as needed. Avoid references
such as “the table below” when the text may render in another location.

For a field that chooses a saved bundle, explain the selection rule and when
selection is required. Do not replace that contract with “Bundle selector.”
For a closed vocabulary, keep the authoritative entries in code and generate
the listing used by reference pages.

Generated blocks are identified by HTML markers naming a source and reader.
The readers `doc`, `attrs`, `value`, and `call` select docstrings, attribute
descriptions, values, or rendered output. Edit the source and regenerate;
hand-editing the generated copy will be lost on the next build.

From the repository root:

```bash
uv run python scripts/generate_support_tables.py
uv run python scripts/generate_support_tables.py --check
```

Review every changed generated page. A short source edit can affect several
field tables or diagnostics. Preserve marker syntax and supported values.
The [documentation maintenance guide](DOCUMENTATION.md) describes the generation workflow.

## Comments and configuration

Use comments to explain intent, invariants, and non-obvious constraints.
Avoid translating the next statement into English. Put a comment near the
code it qualifies so it is likely to be updated with that code.

For example, a comment can explain why lint dependencies have their own group:

```toml
[dependency-groups]
# Keep lint checks independent of the model runtime dependencies.
lint = ["pre-commit>=4.4.0"]
```

Keep actual configuration keys, values, versions, and semantics unchanged in
an editorial pass. Use the comment syntax supported by the file format.

Retain explanations of numerical stability, intentional precision choices,
platform restrictions, synchronization, and performance tradeoffs. They can
prevent plausible but incorrect simplifications. Replace a long account of a
bug with the condition that triggers it and a reference if the issue remains
relevant. Remove commented-out abandoned implementations.

Treat `#:` attribute docs as public documentation, ordinary `#` comments as
local implementation guidance, and tool directives as functional syntax.
Preserve type-checker suppressions, lint directives, coverage markers,
licenses, and generation markers exactly unless changing their behavior is
part of the task.

## CLI help, errors, and logs

CLI help should name the action, accepted value, meaningful default, and effect.
Begin an option's help with a verb where natural. Keep a short help entry
self-contained; link longer explanations from the guide.

Prefer “Check for required changes; exit 1 if a file needs an update” to
“Perform the migration check procedure.” Say whether an option writes files,
loads weights, or overrides another setting when that affects the decision to
use it.

An error should identify the object or field, explain the failed condition,
and give a remedy when one is known. Keep structured codes and paths stable.
Quote actual offending values carefully and avoid exposing secrets or large
payloads. Do not blame the reader or replace useful diagnostics with “invalid
configuration.”

This is an illustrative diagnostic, not an exact current error message:

```text
file_path loads a fitted artifact. Remove init or omit file_path to fit a new one.
```

Logs should describe observed events, state changes, or failures at the
appropriate severity. Preserve machine-consumed keys. Keep verbose debugging
history out of routine user output, and do not describe a planned action as
completed before it succeeds.

Wording in executable strings can affect snapshots, consumers, or source
fingerprints. Review those effects even when the algorithm is unchanged.
If documentation quotes an error verbatim, update it with the real diagnostic
and retain the test that checks the quote.

## Test docstrings and snapshots

A test introduction should state the behavior or invariant it protects.
Explain an unusual fixture or assertion when its purpose is otherwise unclear.
Do not make readers reconstruct a sequence of past PRs to understand what
would constitute a regression today.

**Before:** “After the earlier move, this page stopped linking to the right
file, so this test was added to stop that from silently happening again.”

**After:** “Check that every documented relative link resolves from its page.”

When prose changes, distinguish checks of wording from checks of behavior.
A structured error code, complete vocabulary, link target, metric unit, or
serialization rule remains a meaningful assertion. Adapt a heading lookup or
text expectation when needed, while preserving what the test proves.

Do not delete an assertion simply because it fails after editing. A reader
that discovers zero examples can pass vacuously, so preserve discovery checks.
Review snapshot changes individually and do not accept a fresh snapshot as
evidence that the new behavior is correct.

## PRs, commits, and operational guidance

### PR descriptions and commit messages

Write for a reviewer who has not read the conversation. Lead with the concrete
problem and resulting behavior. Give a before/after example when it makes the
change easier to assess, then state the relevant design, validation, and risk.

A stacked PR identifies its base at the start. Keep the title and
body aligned with the final implementation. Remove abandoned approaches
unless they explain a decision the reviewer must assess.

Distinguish checks that passed, checks that failed, and checks that were not
run. State which failures are pre-existing only with evidence. Do not claim
unchanged runtime behavior solely because a diff looks editorial: error text,
generated values, and source-derived digests can still change.

A commit subject names the completed change, such as “docs: condense method
library README.” Use a body for rationale or compatibility details when needed.
Avoid vague subjects such as “cleanup” and claims about the author's effort.

### Runbooks, security guidance, and contributor instructions

Organize operational instructions by the task or failure a reader faces.
State prerequisites, commands, expected effects, and recovery where relevant.
Keep the scope of a destructive command visible before the reader runs it.
Use stable resource names and documented placeholders rather than personal
paths or an obsolete incident's machine names.

Preserve trust boundaries, access requirements, and consequences in security
text. Concision must not erase who can perform an action or why a restriction
exists. Separate an enforced rule from a recommendation and from background
rationale.

Contributor and agent instructions should give actionable rules and point to
their detailed guides. Avoid duplicating the architecture or testing manual
inside an instruction file. An example, quotation, or historical note should
be clearly distinguishable from an instruction to follow now.

## Editing and verification

### Make the edit in a deliberate order

1. Identify the audience and the decision or action the text supports.
2. Read the source of truth: implementation, schema, runnable example, or
   recorded result. Establish which details are contractual.
3. Move the useful fact or first action to the opening. Group prerequisites,
   procedures, results, and explanations by their role.
4. Remove repetition, obsolete history, filler, and implementation detail
   that does not help this reader. Keep relevant rationale and limitations.
5. Shorten sentences and table cells without changing scope, causality,
   units, defaults, or failure conditions.
6. Check terminology, code identifiers, links, headings, examples, and
   generated sources. Review the diff for accidental semantic changes.
7. Run the checks appropriate to the changed surfaces and inspect the
   rendered result when layout or docstring rendering changes.

### Match verification to the surface

| Change | Relevant verification |
|---|---|
| Markdown links, paths, commands, or JSON | Documentation checks; inspect the edited page |
| Method-library index | One row per document; result links and digest checks |
| Schema descriptions or generated blocks | Regenerate; check generated output; vocabulary checks |
| Docstring syntax or cross-references | Build the API reference and inspect warnings and rendering |
| CLI help or diagnostics | CLI/parser tests, quoted-message checks, relevant snapshots |
| Runtime source text or executable examples | Relevant behavior tests and any affected provenance checks |

The repository's standard documentation commands are:

```bash
uv run pytest tests/docs
uv run python scripts/generate_support_tables.py --check
uv run mkdocs build
```

Follow the [testing guide](TESTS.md) for broader gates and the contributor
conventions for pre-commit checks. A documentation test may check syntax or
existence without proving that an example runs, a fragment anchor resolves,
or a numerical interpretation is sound. Verify those claims separately.
Do not change unrelated files to conceal an existing failure.

For visual review, inspect heading hierarchy, table width, wrapping, nested
lists, syntax highlighting, code-copy behavior, and placement of qualifications.
Check both source readability and the rendered page when their behavior differs.

### Final review questions

- Can the intended reader find the action or answer in the opening?
- Does each paragraph, list, table, and example serve a distinct purpose?
- Are prerequisites, defaults, identifiers, units, and failure conditions exact?
- Are runnable examples complete and excerpts clearly labeled?
- Do results retain their conditions, controls, and limitations?
- Have links, anchors, generated sources, and affected checks been verified?
- Does the page describe the current system without making the reader follow
  its development history?
