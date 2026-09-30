# Demos

One markdown file per research question, with the documents that answer it. The
format — sections, header table, voice, checklist — is
[`docs/demos.md`](../docs/demos.md).

These replace the notebooks the protocol refactor retired. A notebook had to
carry its own execution; a document does not, so what is left is the experiment.

## Onboarding

The [onboarding tutorial](onboarding_tutorial/README.md) is the place to start.
It runs from two zero-ablation warm-ups and a causal model with no network in
it to trained subspaces on `Qwen/Qwen2.5-1.5B-Instruct`. Its landing page lists
every demo in order, with what each one needs and how long it took.
[Saved hypothesis comparisons](hypothesis_testing/hypothesis_testing.md), 03a in
that order, checks the handoff from exact pairs to symbolic predictions and
saved neural outputs.

## Causal models

[Function-based causal models](causal_models/README.md) define each bundled
task's causal model with Python equations. The page covers indexed families,
explicit noise inputs and interventions. [Onboarding 03](onboarding_tutorial/03_causal_model.md)
uses the same syntax on two small algorithms.

## Worked research

| demo | question | needs | measured |
|---|---|---|---|
| [Weekdays geometry](onboarding_tutorial/weekdays_geometry.md) | where and how is the answer day represented, and what does the model say between two answers? | one GPU ≥40 GB | 132 s, 1×H100 |

A **workflow demo**: four research questions as ten steps, each RQ's answer
feeding the next one's document.

## Paper replications

Each package under [`papers/`](papers/README.md) reproduces one published
figure with causalab installed as a library. A package ships its own tables,
documents and figure script and no task package under `causalab/tasks/`. Its
page says how to run it from a pip install, and
[the paper replication guide](../docs/paper_replications.md) gives the page format. `tests/demos/test_papers.py`
validates every document against the package's tables and rebuilds the tables
of every package that ships a builder. It also runs every script step on
fabricated outputs of the protocol steps it reads, and holds each output to
its declared columns and keys.

Six of the pages make a tutorial series, one method per page, in this order
([the series](../docs/paper_replications.md#the-series)):

1. [Zero ablation](papers/rome_fig1_knockout.md), on GPT-2 XL
2. [Causal tracing](papers/rome_fig1.md), on GPT-2 XL
3. Causal models and distributed alignment search, pending
   (`mcqa_symbol` or `mcqa_pointer`)
4. Desiderata-based masking, pending (one of four candidates)
5. [Path patching](papers/ioi_fig3b.md), on GPT-2 small
6. [Interchange at several sites at once](papers/lookbacks.md), on
   Llama-3-70B-Instruct

| Package | Reproduces | Model | Needs |
|---|---|---|---|
| [Desiderata-based masking over attention heads](papers/addition_heads_dbm.md) | Davies et al. 2023, not a paper figure: a head mask over all 48 attention heads of the six full-attention layers finds the heads that carry the tens digit of `NN+MM=`, with a layer-15 fit, exact swaps and random masks as checks | Qwen3.5-2B | one GPU that holds 4.3 GB of bf16 weights; 75 s on one H100 |
| [Arithmetic in the wild, Figure 2a](papers/arithmetic_fig2a.md) | Feucht et al. 2026 ([arXiv:2605.01148](https://arxiv.org/abs/2605.01148)): DAS for the first operand of `a+b=` at every sublayer; the addition curve | Llama-3.1-8B | one 80 GB GPU; 1 h 17 min on one H100 |
| [Arithmetic in the wild, Figure 15](papers/arithmetic_fig15.md) | Feucht et al. 2026 ([arXiv:2605.01148](https://arxiv.org/abs/2605.01148)): whole-residual patching on the weekdays task, IIA over depth and token position for three variables | Llama-3.1-8B | one GPU (23.0 GiB peak at `--batch-rows 1024`); 10 min on one H100 |
| [Desiderata-based masking over MLP neurons, Figure 8a](papers/arithmetic_neurons.md) | Feucht et al. 2026 ([arXiv:2605.01148](https://arxiv.org/abs/2605.01148)): a DBM mask over the 14336 neurons of layer 18's MLP on the weekdays sum, against the paper's 28 addition neurons, and the mask's activations over output sums | Llama-3.1-8B | one H100; 6 min; peak memory not recorded |
| [Average indirect effect per attention head, Figure 3a](papers/function_vectors_fig3a.md) | Todd et al. 2024: the average indirect effect of every attention head when it is set to its task mean in shuffled-label prompts, over 18 tasks | GPT-J 6B, fp16 | one GPU that holds 12 GB of fp16 weights; about 50 min on one H100 for the whole workflow |
| [Desiderata-based masking for function vectors](papers/function_vectors_fig3a_dbm.md) | Todd et al. 2024, a verification with desiderata-based masking (Davies et al. 2023): one head mask per task on the same workflow, against the paper's ten heads | GPT-J 6B, fp16 | one GPU that holds the fp16 weights and the gates' gradients, 61 GB peak per six-task step; the same run |
| [ROME causal tracing, Figure 1e to g](papers/rome_fig1.md) | Meng et al. 2022: restoring one hidden state, a 10-layer MLP window or a 10-layer attention window after corrupting the subject tokens; answer probability over token and layer | GPT-2 XL | one GPU with 8 GB or more, or an Apple-silicon laptop; under 2 min on an H100 |
| [ROME knockouts, an extension of Figure 1](papers/rome_fig1_knockout.md) | Meng et al. 2022, not in the paper: zeroing a 10-layer window of the residual stream, MLP outputs or attention outputs at one token, with no noise; answer probability over token and window centre | GPT-2 XL | one GPU with 8 GB or more, or an Apple-silicon laptop; under 2 min on an H100 |
| [IOI path patching, Figure 3b](papers/ioi_fig3b.md) | Wang et al. 2022: path patching every attention head to the logits at END; the change in the IO minus S logit difference over layer and head | GPT-2 small | one GPU with 4 GB or more and 5 GB of host memory, or an Apple-silicon laptop in about 30 min; 4 min on one H100 |
| [Desiderata-based masking over MCQA components](papers/mcqa_components_dbm.md) | Davies et al. 2023, on the answer letter of onboarding 09: one DBM gate per attention and MLP output of every layer, and one per head of layer 22, each at three l1 weights, beside the one-component scan | Qwen2.5-1.5B-Instruct | one GPU, or an Apple-silicon laptop; under 2 min on one H100, 16 min more for the controls |
| [Lookbacks and belief tracking, Figures 4b, 5b, 6b and 13](papers/lookbacks.md) | Prakash et al. 2025: the answer, binding and binding-source lookbacks on CausalToM, and the Figure 13 control, on the paper's own 80 pairs | Meta-Llama-3-70B-Instruct, fp16 | three 80 GB GPUs; 27 min on three H100s |
| [Manifold steering, Figure 4](papers/manifold_fig4.md) | Wurgaft et al. 2026 ([arXiv:2605.05115](https://arxiv.org/abs/2605.05115)): the weekdays column, chord against manifold steering between two answer centroids | Llama-3.1-8B | one H100; about 1 min |
| [MCQA answer letter with DAS](papers/mcqa_symbol.md) | Geiger et al. 2023 on the task of Wiegreffe et al. 2024, no paper figure: DAS at k = 32 on all 28 layers for the causal model's `answer` beside the full-vector patch, then DBM-DAS and random controls at the best layer | Qwen2.5-1.5B-Instruct | one GPU; 113 s on one H100, peak memory not measured |
| [MCQA answer position with DAS](papers/mcqa_pointer.md) | The same workflow for the causal model's `answer_position` on option-swapped pairs | Qwen2.5-1.5B-Instruct | one GPU; 114 s on one H100 |
| [MLP steering, Table 5 Toxicity](papers/mlp_steering.md) | Geva et al. 2022: turning on ten MLP value vectors that promote safe words; the share of toxic continuations of GPT-2 medium with and without them, the Toxicity cell of 10 Manual Pick in Table 5 | gpt2-medium | one GPU, or an Apple-silicon laptop in 99 s; 73 s on one H100 |

## Method library

[`methods/`](methods/README.md) is one intervention specification per method
— interchange, the locate scan, band and path patching, the Hydra effect,
harvests, probes, DAS in three forms, DBM in three, mean replacement — each
the smallest complete application of the method to a shipped task, with no
`data/` of its own (the shipped task tables are every data root's fallback).
Its README indexes the 29 documents with what each measures, what it needs and
the result of one run; `tests/demos/test_methods.py` holds every document to
its results file. Twenty run on `Qwen/Qwen2.5-7B`, four on
`Qwen/Qwen3.6-35B-A3B`, one on a tiny random model on CPU. Not a demo per
document: copy from it.

## Running any of them

Every demo's "Run it" section holds the three commands, with real output pasted
in. The first two are pure — no weights, no network, no accelerator — so the
shape of a run is checkable before it is booked:

```bash
uv run causalab validate <doc> \
    --engine auto \
    --data-root <demo>/data \
    --data
uv run causalab explain <doc> \
    --engine auto \
    --data-root <demo>/data
```

`uv run` is the checkout's way in. With causalab installed as a package the
prefix goes and the commands are the same: `causalab validate <doc> …`. The
documents resolve everything they point at relative to themselves, so they run
from any directory, and `scripts/standalone_smoke.py` runs both pure verbs over
every workflow here against an installed `causalab`.

The third command needs the accelerator. Note that `--dtype` applies to an
**intervention specification** only: `causalab run <workflow> --dtype bf16` is
refused, because *"a workflow's steps each declare their own realization"*.
Every demo document already pins its dtype, so there is nothing to pass:

```bash
uv run causalab run <protocol> \
    --engine auto \
    --data-root <demo>/data \
    --out runs/<name> \
    --device cuda \
    --dtype bf16
uv run causalab run <workflow> \
    --engine auto \
    --data-root <demo>/data \
    --out runs \
    --device cuda
```

Each demo carries its own `data/` — the tables alone; the command that built
each one is in the demo's markdown.

**Two documents here do not `validate` on their own, by design.** The second
half of a fit→apply pair (07's `mcqa_das_apply.json`, 09's `mcqa_gate_apply.json`,
08's `mcqa_pca_apply.json`, 10's `mcqa_cross_patch.json`, 12's `mcqa_steer.json`)
names its artifact by a **run-tree** path — `"fit/rot.safetensors"` — whose
leading segment is a step name. Standalone that is `[V15] artifact file not
found`; inside its workflow it resolves. Validate the workflow.

## Adding one

Read [`docs/demos.md`](../docs/demos.md), then copy the closest existing demo's
skeleton. `tests/demos/test_demos.py` checks the mechanical half of the format:
that every document validates, every link resolves, every document is inlined in
its demo (a workflow byte for byte, a specification as JSON once its `//`
comments are removed), and every demo has the sections of the format in order.

**Which document shape?** A **protocol** if the product is a number — one
document, one campaign, tables you read. A **workflow** if the product is a
figure, or a value a later step consumes: the figure script and the `select`
script both read the step record that only a workflow writes. The smallest
useful workflow is one protocol step plus one script step, which is a legitimate
shape rather than a workaround — 05 ships exactly that beside its protocol, and
both 05 and 06 inline the workflow next to the figure it produces
([05's](onboarding_tutorial/05_trace.md#execution) ·
[06's](onboarding_tutorial/06_localize.md#execution)), so the picture and its
generation are read together.

**If the method learns anything, ship a fit *and* an apply.** A fit document's
own `iia.json` is the score on the split it trained on, and the gap is not
academic: 07 measures 0.875 train against 0.438 held-out at k = 16. The apply
document has no `train` section, loads the artifact by `file_path`, and scores a
split the fit never saw. 07, 08 and 09 all take this shape.
