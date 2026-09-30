# Writing a paper replication

This guide gives the layout of `demos/papers/`, the rules for a paper
replication package, and the format of its page.

A package reproduces one published figure with causalab as an installed
library. A few packages teach a method on a figure of the onboarding tutorial
or on a measurement no paper draws (Page variants, below). A package ships its
own tables, documents and figure script, and no task package under
`causalab/tasks/`. The [paper replications index](../demos/papers/README.md)
gives the commands that run a package, and the
[demo index](../demos/README.md#paper-replications) lists the packages. The
checks that `tests/demos/test_papers.py` runs are in the
[testing guide](TESTS.md#paper-replication-packages).

## The series

Six pages make a tutorial series, one method per page, in this order. The
site lists them under Demos, Tutorials, Series (`extra.tutorials` in
`mkdocs.yml`), and the other packages under Papers.

| Slot | H1 | Page |
|---|---|---|
| 1 | Zero ablation | [`rome_fig1_knockout.md`](../demos/papers/rome_fig1_knockout.md) |
| 2 | Causal tracing | [`rome_fig1.md`](../demos/papers/rome_fig1.md) |
| 3 | Causal models and distributed alignment search | `mcqa_symbol.md` or `mcqa_pointer.md` |
| 4 | Desiderata-based masking | one of `mcqa_components_dbm.md`, `function_vectors_fig3a_dbm.md`, `addition_heads_dbm.md`, `arithmetic_neurons.md` |
| 5 | Path patching | [`ioi_fig3b.md`](../demos/papers/ioi_fig3b.md) |
| 6 | Interchange at several sites at once | [`lookbacks.md`](../demos/papers/lookbacks.md) |

Slots 3 and 4 have candidate packages and no page yet. A page defines a
term on its first use in the series, and a later page links back to it.

## Layout

`demos/papers/` has the layout of the
[onboarding tutorial](../demos/onboarding_tutorial/README.md): one page per
package beside one shared `protocols/`, `workflows/` and `artifacts/` folder.
Every file of a package carries the package's name.

```
demos/papers/
├── README.md                          the index: how to run a package
├── <name>.md                          the package page (The page, below)
├── .gitignore                         ignores artifacts/output/ (below)
├── protocols/
│   └── <name>_<step>.json             every intervention specification, prefixed with its package
├── workflows/
│   ├── <name>.json                    one workflow per package
│   └── scripts/<name>/
│       ├── build_dataset.py           optional; the recipe for every committed table
│       ├── <step>.py                  workflow `script` steps
│       ├── <fig>_figure.py            draws the figures from the run tree
│       └── jobs/                      optional cluster launchers; not committed
└── artifacts/
    ├── data/<name>/                   every external input of one package
    │   ├── <fig>_<author><year>_original.png   the paper's figure, cropped from the PDF
    │   ├── <fig>_onboarding_original.json      or: the values an onboarding figure draws, with source path and sha256
    │   ├── <fig>_<author><year>_values.json    optional; the paper's values, read from its figure
    │   ├── <dataset>.json             a download, wrapped as {source url, sha256 of the original bytes, records}
    │   └── <table>.json               the tables the specifications name ("dataset": "<name>/data" is data.json)
    ├── figures/<name>/                the committed figures and the values behind them
    └── output/<name>/                 the run tree, the workflow's output_dir; not committed
```

| Path | Role | Committed |
|---|---|---|
| `<name>.md` | the package page | yes |
| `protocols/<name>_<step>.json` | the package's intervention specifications | yes |
| `workflows/<name>.json` | the one workflow, holding only the steps the figures read | yes |
| `workflows/scripts/<name>/` | the builder, the step scripts, the figure scripts | yes, except `jobs/` |
| `artifacts/data/<name>/` | tables, the paper's crop or the copied onboarding values, wrapped downloads | yes |
| `artifacts/figures/<name>/` | the produced figures and their values | yes |
| `artifacts/output/<name>/` | the run tree | no |

## Rules

- **Name.** `<paper_short_name>_fig<N>` for one figure (`rome_fig1`,
  `arithmetic_fig2a`), `<paper_short_name>` alone when the package covers
  several figures (`lookbacks`). A package that reproduces no paper figure
  is named after its task and its method (`mcqa_symbol`,
  `addition_heads_dbm`). The page, the protocol prefix, the workflow
  and the folders under `workflows/scripts/`, `artifacts/data/` and
  `artifacts/figures/` all carry the name. A second page on the same workflow,
  such as an experiment the paper does not draw, is `<name>_<topic>.md`
  (`rome_fig1_knockout.md`).
- **One workflow.** `workflows/<name>.json` holds only the steps the figures
  read. Fan-out, script steps and several figures are steps inside it. A
  variant that changes a few lines of one document is a step of the same
  document with a `set` override (`sites.target.component`), not a second
  protocol file. A check the figures do not draw is a number on the page,
  not a step and not a second workflow.
- **Protocols.** `protocols/<name>_<step>.json`, named after the workflow step
  or the measurement (`rome_fig1_trace_state.json`,
  `ioi_fig3b_direct_effect.json`). A workflow names them as
  `../protocols/<file>`. Name a site after its role (`target`, `restore`), so
  that a variant changes the component and not the name.
- **Model.** Every `model.key` a package names has a row in the static
  registry (`causalab/protocol/registry/models.py`), taken from the
  checkpoint's `config.json`. With that row, `validate` and `explain` run
  offline, which `tests/demos/test_papers.py` and the standalone-install
  smoke (`scripts/standalone_smoke.py`) both require.
- **Sources.** Everything the specifications consume that the run does not
  produce lives in `artifacts/data/<name>/`: the tables, the paper's crop
  `<fig>_<firstauthor><year>_original.png`, and downloads wrapped in an object
  that names the source URL and the sha256 of the original bytes. A package
  never reads another demo's files at run time. When its Original is an
  onboarding figure, the builder copies the values that figure draws into
  `<fig>_onboarding_original.json`, an object that names the source path in
  the repository and the sha256 of the source file's bytes. The values
  a page reads off the paper's figure can be kept as
  `<fig>_<firstauthor><year>_values.json`, with the PDF's URL and sha256 and
  the rule that turns a mark of the figure into a value
  ([`rome_fig1`](../demos/papers/artifacts/data/rome_fig1/fig1efg_meng2022_values.json)).
  The data root of every command is `artifacts/data`, so a specification
  names its table `<name>/<table>`.
- **Tables.** `workflows/scripts/<name>/build_dataset.py` writes them into
  `artifacts/data/<name>/`, and
  `python workflows/scripts/<name>/build_dataset.py --out artifacts/data/<name> --check`
  exits 0 on every committed table. A package without a builder treats its
  table as given.
- **Script steps.** A script reads a protocol step's tables by the columns
  the engine writes (`example_id`, the axis ids). `tests/demos/test_papers.py`
  runs every script step on fabricated outputs of the protocol steps it
  reads, before any cluster run. It fails when the script reads a column
  the engine does not write. It also fails when the script does not write a
  declared column or key, or writes no rows to a table declared with
  columns. It does not refuse a column that the script writes and does not
  declare. A script that loads a model of its own, such as a classifier,
  needs a stand-in for that object in the test's `SCRIPT_STAND_INS`.
- **Launchers.** Files under `workflows/scripts/<name>/jobs/` name machines,
  environments and branches, so they stay out of git. The page records the
  run they made.
- **Figures.** `workflows/scripts/<name>/<fig>_figure.py` reads
  `artifacts/output/<name>/` and writes into `artifacts/figures/<name>/`: the
  replication figure, one image per panel, and `<fig>_plotted.json`, the
  values drawn (The figures, below).
- **Page.** The parts, their order and their length are fixed in
  [The page](#the-page). There is no `Reproduced` field and no header table:
  the Execution block names the run the committed figures come from.

## The page

The page is a tutorial for a practising interpretability researcher, read in
about three minutes. It shows the source figure and ours, then the
intervention specification that produced ours, cut into small commented
chunks. Everything else is collapsed. Copy the layout from
[`rome_fig1.md`](../demos/papers/rome_fig1.md), a replication, and
[`rome_fig1_knockout.md`](../demos/papers/rome_fig1_knockout.md), an
experiment the paper does not draw. A page with `## Method`, `## Results` and
`## Execution` sections and one whole specification in a commented fence is in
the earlier layout and is not a model.

The [general writing guide](STYLE_GUIDE.md) sets the language, and the
[tutorial guide](demos.md) the vocabulary (§8), the caption (§6.4) and the
number format (§6.3). This page fixes only what is specific to a replication
package.

### Parts, in order

1. **Title and citation.** The H1 names the method or the content that the
   page teaches, in sentence case: `# Causal tracing`, `# Path patching`. A
   qualifier can narrow it: `# Desiderata-based masking over attention
   heads`. The H1 does not name the paper or the figure. The rule holds for
   every page, in the series or not (`# Manifold steering`), and
   `tests/demos/test_papers.py` checks it. Below the title, one
   blockquote names the paper. It holds the first author, the title in bold
   and the arXiv link, and nothing else:

   ```markdown
   > Meng et al. **Locating and Editing Factual Associations in GPT.**
   > [[arXiv]](https://arxiv.org/abs/2202.05262)
   ```

   A page that reproduces no paper figure cites the paper that introduced
   its method. A page whose task comes from another paper adds a second
   citation of that paper in the same format, after a `>` line that
   separates the two. A page whose Original is an onboarding figure adds one
   line to the blockquote that links the onboarding page:

   ```markdown
   > Original figure: [onboarding 06](../onboarding_tutorial/06_localize.md)
   ```

2. **Figure context.** `**Figure context:**` and a short bullet list:
   - the first bullet describes the behaviour under study in one sentence;
   - the second states the research question, as a question;
   - the third, and optionally a fourth, give the methodology: the high-level
     idea first, then the detail the reader needs;
   - an optional last bullet links a related replication page.

   Leave out ranks, splits, training settings and counts.

3. **The figures.** A replication shows two images under two subheadings, so
   that the reader sees two results from two systems:
   - `### Original`: the source figure, with no caption.
   - `### Replication`: this run's figure, in the style the figure script
     draws, panel for panel with the original, then the caption.

   The source figure depends on the page variant (Page variants, below). A
   page with no source figure shows one figure and no subheadings. The
   caption opens with a numbered, descriptive title of the setting, then
   says what the image shows: "*Figure 1: Localization of factual recall in GPT2-XL: `The Space Needle
   is in downtown` --> ` Seattle`. …*"

4. **`## CausaLab implementation`.** In order:
   - One sentence that says which experiment and which panel the walkthrough
     builds, and points to the dropdown: "Let's walk through the
     specification for running the MLP ablation experiment (Figure 1,
     center) in CausaLab. Expand the dropdown to see the full
     implementation." When a block sits between that sentence and the
     dropdown, such as the equations fence of a causal model (The chunks,
     below), the pointer to the dropdown moves to the line directly above
     the Full JSON block.
   - A collapsed block `<details><summary><b>Full JSON</b></summary>` holding
     the whole protocol file, header included, in one `json` fence.
   - The chunks, each under a `###` subheading (The chunks, below).
   - Only when the specification draws part of the replication figure:
     "Given the specification, CausaLab produces:", then that part alone, with
     its caption. When the specification draws the whole figure, the page
     does not show the figure again.

5. **`## <Other panels>: change these lines`.** This section shows which
   variation of the specification produces the other parts of the
   replication figure. When those parts come from the same specification
   with a few lines changed, or from a sibling document, one paragraph says
   which document or workflow override carries the change.
   Then one `###` subheading per panel, each with a `diff` fence of the
   changed lines only and that panel alone with its caption. A page whose
   figure has one panel has no such section. A page may show the paper's
   crop of each panel beside that panel. The two images then share one row
   of a table with the header `| Original | Replication |`, above the
   caption, since GitHub and the site both draw a table row side by side.
   The page's `### Original` and `### Replication` then show only the panel
   that the implementation section builds (`lookbacks`).

6. **`## Further Details`.** `<details>` blocks, collapsed by default, each
   with a bold `<summary>` label, in this order:
   - **Method**, only when the method needs one: a fit and
     its selection, a measured noise scale, a filter on correct pairs, an
     answer-spelling rule, or a data departure from the paper. A plain table
     of columns is not a reason for this block.
   - **Execution**: five paragraphs, each opening with a bold label that ends
     in a period. `**Environment.**` names the checkpoint, its license and the
     environment variables in a `bash` block. `**Run.**` gives the command
     from `demos/papers/`, `causalab run workflows/<name>.json` first, one
     flag per line, then the figure scripts. `**Flags.**` explains the flags
     whose values differ from [Run a package](../demos/papers/README.md#run-a-package),
     `--resume`, and `validate` in place of `run`.
     `**Resources and reproducibility.**` gives the hardware, the wall time,
     the laptop alternative, and the run the committed figures come from:
     hardware, precision, engine and date or commit. A protocol step's
     `_step.json` records the device (`execution.device`) and the model
     snapshot (`models[].resolved_revision`) to quote. `**Workflow.**` names the
     steps in order, each document linked once, what each hands to the next,
     and every departure from the paper's setup in one sentence.
   - **Intervention protocol parameters**: one table of the fields a reader
     changes for another fact, model or grid, with the value here and what to
     put instead, and a link to
     [the intervention reference](intervention_protocol.md).

There is no other `##` section: no `## Method` (the Method block sits
collapsed under Further Details), no `Results`, `Next steps`, `Limits` or
`Layout`. Write in a plain voice, with "we" and short sentences.

### Page variants

The parts above stay in every variant. The source of the Original changes.

| Variant | `### Original` | Example |
|---|---|---|
| A paper figure | the paper's crop, `artifacts/data/<name>/<fig>_<firstauthor><year>_original.png` | `rome_fig1` |
| An onboarding figure | an image that the package's figure script draws from `artifacts/data/<name>/<fig>_onboarding_original.json` into `artifacts/figures/<name>/<fig>_original.png` | `mcqa_components_dbm` |
| Values the paper prints, such as table cells | an image that the package's figure script draws from `artifacts/data/<name>/<fig>_<firstauthor><year>_values.json` into `artifacts/figures/<name>/<fig>_original.png`, in the style of the replication; the Replication caption names the table that the plot adapts | `mlp_steering` |
| No source figure | none: one figure, and no `### Original` or `### Replication` subheading | `rome_fig1_knockout`, `addition_heads_dbm`, `mcqa_symbol` |
| Verification | the paper's crop; `### Verification with <method>` replaces `### Replication` | `function_vectors_fig3a_dbm` |

A verification page checks the paper's result with a method the paper does
not use, so its figure is that method's answer to the paper's question and
its caption compares the two.

The onboarding variant cites the onboarding page in its blockquote. Its
builder copies the values the onboarding figure draws, cut to what the page
compares (a row, every third layer), from that page's committed results into
`artifacts/data/<name>/`. The copy is an object with the source path, the
sha256 of the source file's bytes and the values, so that `--check` fails when
the onboarding result changes. The figure script draws the Original from the
copy in the style of the replication.

### What the page leaves out

- A results paragraph or a results table. Each panel's caption states what
  the panel shows, against the paper's values where the paper has them, with
  the limits beside the claim: seeds, a best-of selection, a substitute model.
  A `change these lines` panel of a few numbers can be a small markdown table
  of values from `<fig>_plotted.json` in place of an image, with its caption.
- Prose between chunks, beyond one sentence under a subheading. A chunk's
  subheading says what it does, and its `//` comments carry the few details
  the reader needs. A few chunks take one sentence (The chunks, below).
- Pasted `validate` or `explain` output and digest inventories. The demo
  suite checks the documents. The page does not prove they load.
- The paper-to-field table, the development history of the package, and
  measurements the current workflow does not make.
- Install instructions, PR numbers and the history of fixes. Run a package
  covers the install, and the page describes the package as it is now.

### The figures

`workflows/scripts/<name>/<fig>_figure.py` writes into
`artifacts/figures/<name>/`:

- `<fig>_replication.png`, every panel of this run, without the paper's crop,
  for `### Replication` (`<topic>_all.png` for a page with no source figure).
  A page that shows each crop beside its panel (Parts, item 5) has no such
  image. Its `### Replication` shows the panel image of the implementation
  section;
- one image per panel, `<fig>_<panel>.png`, for a page whose specification
  draws only part of the figure: the implementation section shows that part,
  and each `change these lines` subheading shows its own. When every panel
  mixes the lines of two specifications, the parts are the specifications'
  lines, `<fig>_<specification>.png`, each drawn across all panels
  (`mlp_steering`: `table5_amplified.png` and `table5_baseline.png`);
- `<fig>_original.png`, the Original of a page on an onboarding figure,
  drawn from the copied values;
- `<fig>_plotted.json`, the values drawn.

Where the paper colours a component, the figure keeps the colour: purple for
the residual stream, green for MLP outputs, red for attention outputs, as in
the ROME figures. The image carries its axes and colour bar, so the caption
need not repeat them.

A desiderata-based masking result is drawn by
`causalab.io.plots.dbm_figure.plot_dbm`: the sweep with the reported point
ringed, then that point's mask. It keeps the colours of the DBM viewer the
encyclopedia reports use, so every DBM page reads the same way
(`addition_heads_dbm`).

A caption follows each image as one italic paragraph of one to four
sentences. The first caption on the page names the quantity and the axes. A
panel caption states what the panel shows, with the numbers that compare with
the paper.

### The chunks

The implementation section shows its specification whole, cut into chunks:
one fence per key, or per few keys that belong together (`model` and `data`;
`featurizers` and `train`; `positions` and `sites`). A chunk is a fragment,
top-level pairs without the enclosing braces, so `"model": {...},\n"data": {...}`
with no trailing comma. A section of `method` (`intervened_models`,
`positions`, `sites`, `featurizers`, `reads`, `writes`, `train`, `save`) is
shown as its own key, without the `method` wrapper. The `header` is shown
only in the Full JSON block.

- **Order.** The causal model, when the page shows it, then model and data,
  the intervened models, positions and sites, any `axes`, reads, writes, the
  featurizer and training, then save.
- **Subheadings.** Each subheading says in plain words what the chunk does
  in this experiment: "Define the knocked-out model, with reads and writes as
  placeholders", "Select every token position and the MLP layers", "Define
  reads: the final logits of the knocked-out run". No bullets under a
  subheading.
- **One sentence.** A few chunks take one sentence between the subheading
  and the fence, when the reader needs a fact that no single line carries,
  such as why the table repeats one prompt. Most chunks take none, and no
  chunk takes two. The sentence counts in the page's length, and the Length
  table does not change.
- **Comments.** A `//` comment at the end of a line carries a detail the
  reader needs: one base and one counterfactual sample, what a sweep means
  (`sweep: a separate intervention per layer`), where a value comes from in
  the paper (`Appendix D.1: lr 1e-4, 8 epochs, batch 16`), or the line that a
  `change these lines` section replaces. Two or three comments serve most
  chunks, and many need none.
- **Repeated structure** stays in full. A 33-row axis is 33 rows, so that the
  chunks can be checked against the file. Where the format has a sweep
  spelling for a list, `{"sweep": {"range": [0, 48]}}`, `{"axis": …}` or a
  dependent axis, the document uses it, and the chunk is one line.
- **The file.** The Full JSON block is the file, so the chunks after it are
  checked against the protocol file it equals. The fit half of a fit and
  apply pair is the document shown; the apply half is one sentence in the
  Workflow paragraph.
- **Only what the figure needs.** A part of the specification stays only if
  the figure needs it: a clean or baseline run, a read or a save that no
  figure input consumes is left out. A number the page only quotes, such as
  a clean probability, is stated as given ("Clean p(Seattle) is 0.976" on
  [the ROME knockout page](../demos/papers/rome_fig1_knockout.md)).
- `header.description` is one sentence that says what the document measures
  and which script or document consumes its output.

Every `python` fence on a package's page is a verbatim copy of one
top-level definition, decorators included, and names the file it copies in
its `title`
(```` ```python title="causalab/tasks/MCQA/causal_models.py" ````).
`tests/demos/test_papers.py` requires each `python` fence to equal the
source of that definition, so a usage snippet or any other python fails the
check. A page whose causal model comes from the library shows its equations
function this way, under its own subheading before the Full JSON block. The
fence is not a chunk: the merge check below reads only `json` fences.

`tests/demos/test_papers.py` merges the chunks, the `method` sections into
`method` and the rest at the root, and requires the result to equal the file
minus its `header`. The chunks are the fences after a Full JSON block up to
the next `##` heading. The Full JSON block must equal a protocol
file, header included, once its comments are gone.

### Length

The counts are prose outside fences, images and `<details>` blocks. The two
ROME pages have about 480 and 550 words.

| Part | Words |
|---|---|
| Figure context | four or five bullets, 60 to 120 |
| One caption | 20 to 80 |
| Chunk subheadings together | 60 to 150 |
| Whole page outside the collapsed blocks | 350 to 650 |

A page over the upper bound is carrying material from the list above of what
the page leaves out. Shorten by removing it, not by cutting a condition on a
number or a departure from the paper.

## `.gitignore`

One file for every package, `demos/papers/.gitignore`:

```gitignore
# `causalab run workflows/<name>.json --data-root artifacts/data --out artifacts/output`
# writes each run tree under artifacts/output/<name>/, which is not kept. The
# tables and paper crops under artifacts/data/ and the committed figures and
# their values under artifacts/figures/ are kept.
artifacts/output/
__pycache__/
workflows/scripts/*/jobs/
```
