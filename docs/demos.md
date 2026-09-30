# Writing a tutorial

This guide gives the format of a tutorial: its sections, header table, voice
and checklist.

A tutorial, or demo, tests a research question through a runnable experiment. It states the
question, explains which intervention answers it, and reports the measured
result with the conditions that limit the claim. The reader knows Python and
neural networks and learns each causal concept where the experiment first
needs it.

The role models are the onboarding tutorials
[02_ablation_attention](../demos/onboarding_tutorial/02_ablation_attention.md)
and [05_trace](../demos/onboarding_tutorial/05_trace.md). See the
[demo index](../demos/README.md) for the current examples.

## 1. Layout

Use these parts in order, keeping each section focused on what this demo adds:

1. A top table with the question and the method.
2. `## Research question`
3. `## Method`
4. `## Execution`
5. `## Results`
6. `## Next steps`

Keep Next steps short and include it only when there is a useful follow-up.
Resolve editorial comments in the text and drop obsolete material. Do not add
a Manual review section or preserve discarded text as a checklist. The layout
supports the explanation; it does not require repeating material from earlier
tutorials.

## 2. Top table

1. Every demo opens with a two-row table, `**Question**` and `**Method**`.
   Its header row is `| Overview | |`: the label column carries the title
   `Overview`, and the value column has no title.
2. The question is one sentence. It links the model card and the dataset file.
3. The method is a bold name followed by one sentence that names the input, the
   write, and the read-out, in that order.

## 3. Research question

1. When building on an earlier tutorial, open with the concrete finding that
   motivates the new question: name the component, positions, layers, and
   tested inputs when relevant. Then state what remains unknown. A first
   tutorial can open directly with its research question; a generalization
   tutorial asks whether the earlier finding holds across other examples.
2. Show representative inputs here. For counterfactual experiments, put each
   prompt on one line and each base/counterfactual pair on consecutive lines,
   labelled as in 05_trace. Flatten line breaks for this compact display and
   link the exact dataset. A population study should show a few varied pairs.
3. State clean accuracy over the dataset used by the experiment, for both
   input roles when relevant. One prompt's probability is not a population
   baseline. Use measured values and link their recorded output.
4. Cite the task or hypothesis source when it is introduced. Later tutorials
   can link that introduction.
5. Define a concept on its first use in the series. Later tutorials build on
   that explanation and focus on the new addition.
6. When token positions matter, show their semantic roles alongside indices.
7. End with a small set of distinct, bold numbered questions, `**Q1 — …**`.
   Prefer questions about the behavior or mechanism to requests for the
   maximum score and its coordinates. A population study should distinguish
   the shared pattern from variation between examples.
8. Give each question at most one or two sentences of clarification. A demo
   with one question still calls it Q1.

## 4. Method

1. Start with the procedure: prompt the model with X, replace Y, and measure
   Z. Do not repeat the research problem or motivation from the preceding
   section.
2. Explain what changes from the previous tutorial. Link familiar concepts
   and avoid reproducing the same prompts or dataset rows in several sections.
3. Show an annotated dataset row only when its structure is new or needed to
   understand the change. Familiar counterfactual pairs need no second display.
4. State how to interpret the measurement before presenting results. Separate
   a zero response to this intervention from evidence that no representation
   exists. Do not treat a dataset's slot balance as a proven causal ceiling.
5. Inline the intervention specification when it helps explain the new
   experiment. Comment on new fields; collapse repeated structure. Align
   comments in one column, as in the role models.
6. Keep `description` short: say what the document measures. Put numerical
   baselines beside the results that use them.
7. A full inlined copy, minus comments, must parse to the same JSON as its
   linked file. Clearly label partial examples as excerpts or edits.
8. Explain only the parameters readers need to change for this experiment.
   Link the reference for established fields; a parameter table or expandable
   block is optional, not repeated boilerplate.
9. Use "base" and "counterfactual" for the specification's input roles and
   explain how they correspond to the experimental inputs.
10. For interchange interventions, state the expected output supplied by the
    causal model and how the tested pairs were selected. Build on the dataset
    design explained earlier in the series.

## 5. Execution

1. The command runs from the repository root. Every flag is on its own line.
2. A flag or command is explained on its first appearance in the series and
   not again. 01_ablation_MLP explains `--engine`, `--out`, and `--device`;
   02_ablation_attention explains `--data-root`; 05_trace explains `explain` and
   the workflow.
3. Show the `run` command once. Mention that readers can replace `run` with
   `validate` to check without loading weights, omitting run-only flags. Do not
   add repeated `validate` and `explain` command/output blocks unless those
   commands are themselves the lesson.
4. Describe a workflow by what it does: run the experiment, then plot its
   results. Include only steps the questions require. A best-point selection
   or values file belongs only in a tutorial that uses it. Inline the workflow
   with a link when useful; keep scheduling internals out of the explanation
   unless they are the topic.
5. Resources are stated: forward count, memory, wall time, hardware, and the
   date the shipped artifacts were produced.
6. Shipped artifacts are what the command writes. `--out` is the demo's
   `artifacts/output`, and a workflow's `output_dir` carries the demo's file
   prefix, so nothing is copied or renamed by hand.
7. Each demo directory has a `.gitignore` that says which outputs are
   committed. Commit the inputs, the figures, the values each figure draws,
   and the small result files the text links. Do not commit per-example
   tables, superseded attempts (`.attempts/`) or the rest of the run tree. A reader
   regenerates them with the demo's command. Git keeps every committed version
   of a file, so a large table stays in the history after its removal.
   - The paper packages share one `.gitignore` in `demos/papers/`. It ignores
     every run tree under `artifacts/output/` and keeps the tables under
     `artifacts/data/` and the figures under `artifacts/figures/`
     ([the paper replication guide](paper_replications.md#gitignore)):

     ```gitignore
     artifacts/output/
     __pycache__/
     workflows/scripts/*/jobs/
     ```

   - The onboarding tutorial ignores the per-example tables of its scan steps
     (steps named `scan` or ending in `_scan`), every step's `_step.json`,
     which the run's committed `workflow.json` repeats field for field, and
     every `.safetensors` file, the harvested activations and fitted
     featurizers no tutorial links. It commits the rest of its run trees.

   `tests/demos/test_demos.py` fails when a tracked file matches an ignore
   rule and when a paper package has no `.gitignore`.

## 6. Results

1. The first demo that produces a table shows the raw output file with one
   comment per field. Later demos show a table or a figure instead.
2. Give the relevant clean baseline once near the intervention results and
   make clear when baseline accuracy and the intervention metric target
   different answers.
3. Numbers are fractions, `0.522`, not percent.
4. A figure has a caption naming the quantity, both axes, and the file with
   the drawn values. Label token positions by semantic role, such as symbol 0,
   symbol 1, and answer slot. Keep raw indices in the specification or a key.
   For dense layer scans, label every third layer on the x axis.
5. Show only plots and tables needed to answer the questions. When a question
   asks about variation between examples, prefer one clearly defined variance
   statistic alongside the aggregate plot. State the measured quantity, units,
   denominator, and exclusions. Add another plot only when it answers something
   the statistic cannot. First success need not mean persistent success.
6. Each question gets a `### Qn:` heading whose text is the answer, not the
   question.
7. Explain relevant controls and limitations where they inform the answer.
   Avoid forcing the same control discussion into every subsection.
8. An answer connects to a previous demo's finding when it can.
9. Claims stay within the tested inputs and interventions. When this demo
   tests generalization, report how broadly the earlier finding holds and
   where it differs. Changing a question requires updating its plots and
   results discussion, not just its heading.

## 7. Next steps

Keep at most one or two follow-ups that arise directly from the results.
For an experiment, state the concrete edit and the question it would resolve;
do not predict its outcome as certain. A link to the next tutorial can stand
alone when it supplies the natural continuation. Omit speculative variations,
implementation chores, and unrelated paper links.

## 8. Across sections

1. Terms follow the vocabulary census in
   [the intervention reference](intervention_protocol.md#111-terms):
   "intervention specification", not "protocol".
2. Say "model component", or name the component, such as "residual stream" or
   "attention output". Do not say "cell".
3. Learned masks are **Desiderata-Based Masking**; the combination with
   distributed alignment search is **DBM-DAS**.
4. No `TODO`, unresolved `{{comments}}`, or Manual review section in finished text.
5. Artifacts carry the demo's file prefix: `05_trace_p_q_grid.png` belongs to
   `05_trace.md`.
6. Links are relative to the demo file. Every link resolves.
7. The voice is first person plural: "we", "let's".
8. There is no `Reproduced` field. The Execution section states where and
   when the shipped artifacts were produced.

## 9. Checks

Run `uv run pytest tests/demos tests/docs`. The docs suite checks links and
figure references in every demo. The demos suite checks that each document
validates against its demo's data root, that each inlined JSON copy matches
its file, and that each quoted digest is current. Review scientific claims
against the recorded measurements.

`tests/demos/test_demos.py` checks the layout above: the two-row top table,
exactly the five sections in order, and no `Reproduced` field. A demo in an
earlier layout fails until it is rewritten; the suite carries no list of
exceptions. Data resolves under
`artifacts/data/` when a demo keeps an `artifacts/` directory and under
`data/` otherwise, with the shipped task tables behind it
(`tests/_helpers/demos.py`). An inlined specification is compared as JSON
after its `//` comments are removed; an inlined workflow is compared byte for
byte.

When an experiment document changes, update its embedded copy and digest with:

```bash
uv run python scripts/repin_demo_digests.py
uv run python scripts/repin_demo_digests.py --check
```

The script compares the working tree with `HEAD` by default. Use `--baseline`
to select an earlier revision when needed. It reports a quotation it cannot
associate with a document. Updating a digest records the current document;
results require a corresponding run before their reproduction status changes.
