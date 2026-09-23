# Causal Abstraction for Mechanistic Interpretability

A framework for **mechanistic interpretability** — reverse-engineering the algorithms language models use internally using **causal abstraction**.

You write a high-level causal model describing *how you think* an LM solves a task, then run experiments to test whether the LM's internal components actually implement that algorithm. Every experiment is a serializable **intervention protocol** — a JSON document naming sites, reads, edits, intervened models, and metrics — validated, digested, and executed by an engine. The document is the seam: engines (the pytorch-hooks reference engine and the nnsight tracing engine today; tensor-parallel engines tomorrow) implement against the same format.

## Quick Start

1. **Clone and install:**
   ```bash
   git clone https://github.com/goodfire-ai/causalab.git
   cd causalab
   uv sync
   ```
   `uv sync` compiles the weight reader's Rust core
   ([`docs/fastersafetensors.md`](docs/fastersafetensors.md)), so a Rust
   toolchain is required: `rustup` on `PATH` (`rust-toolchain.toml` pins the
   Rust version; rustup installs it on first use). The first build takes a minute or two;
   later syncs rebuild only when a Rust source changes.
   The interactive views — Jupyter, and the Dash/Cytoscape causal graphs — are
   the `notebook` extra (`uv sync --extra notebook`). Everything below, the
   demos and the whole matplotlib figure surface included, runs without it: a
   headless install carries no web-app server.
   Optional Linux GPU acceleration is available separately with
   `uv sync --extra flash-attn` and `uv sync --extra flash-linear-attention`,
   or both extras together. See [attention backends](docs/attention_backends.md)
   for build requirements, backend selection, and the default fallbacks.
2. **Run a demo.** [`demos/`](demos/) is one markdown file per research question, with the documents that answer it. Two need no GPU: [causal_model](demos/causal_model/causal_model.md), which has no network in it at all, and [01_define](demos/onboarding_tutorial/01_define.md). The eight onboarding demos that do need one total **5.4 minutes** of H100 time between them, the longest being 88 s. The format is [`docs/demos.md`](docs/demos.md).
3. **Read the two specs.** [`docs/intervention_protocol.md`](docs/intervention_protocol.md) — the document format (sections, the `do` algebra, sweeps, validation, digests, the engine contract). [`docs/workflow_protocol.md`](docs/workflow_protocol.md) — chaining protocol runs with script steps: inputs, one Python script, declared outputs.
   The guides beside them answer the questions the specs do not: [`docs/methods/`](docs/methods/README.md) has one how-to page per method family (DBM, DAS, PCA, SAE) with every field, its legality and the shipped templates to copy; [`docs/running_experiments.md`](docs/running_experiments.md) walks one experiment end to end and tabulates the hookpoint vocabulary per engine, and [`docs/qwen36-35b-a3b-architecture.html`](docs/qwen36-35b-a3b-architecture.html) is the annotated block diagram those hookpoints name (machine-checked against the component vocabulary by `tests/test_architecture_diagram.py`). [`docs/CODEBASE.md`](docs/CODEBASE.md) is the module map.
4. **Run a shipped intervention specification:**
   ```bash
   uv run causalab explain  causalab/configs/protocols/interchange.json
   uv run causalab run      causalab/configs/protocols/interchange.json \
       --out runs/interchange --device cuda --dtype bf16
   ```
   A document is four groups — `header`, `model`, `data`, `method` (spec §1).
   The **method** group is the transferable part; `model` and `data` name the
   network, the rows and the precision, and `explain` prints the **method
   digest** that says whether two documents are the same experiment on two
   networks. A `protocol_version` 1 document is rewritten in place with
   `uv run causalab migrate <file>`.

## The CLI

| verb | effect |
|---|---|
| `run <doc>` | validate, expand, plan, execute, stamp |
| `validate <doc> [--data]` | the spec §5 load-error checklist; `--data` also checks column references |
| `explain <doc>` | models, forward plan, point count, derived `requires`, digest, save products |
| `digest <doc>` | the campaign digest |

Common flags: `--set path=value` (ad-hoc override — exploration only), `--data-root` / `--artifacts-root` (resolution roots; the tables the tasks ship under `causalab/tasks/<task>/data/` are always reachable behind them), `--max-points` (override the sweep point cap), `--register-from-hf` (resolve an unregistered model key from its HF config — `run` always does; the pure verbs need the flag, so a digest never depends on the network), `--device` (engine placement, `run` only), `--batch-rows N` (reference-engine microbatch bound, `run` only — execution, recorded in the receipt and in no digest), `--engine` (`pytorch_hooks` · `nnsight` · `auto` — pin one, or let §8 route; on `explain` it previews the routing).

`--dtype` (shorthand for `--set model.dtype=…`: precision is a document fact, so it enters the digest) and `--points START:STOP` (execute one shard of a swept campaign — the seam external schedulers dispatch on; digests are unaffected) are **intervention specifications only**; a workflow run refuses both. The same verbs dispatch on workflow documents (they carry a `steps` section), which take `--resume` (skip a step whose outputs carry a matching stamped digest) and `--reuse-nondeterministic` instead.

`run` also writes `<out>/protocol.json`: the canonical document (every default materialized — dtype and quantization included), its digest, the per-point provenance digests, the method it was composed from, and an `execution` block (`batch_rows`). That file is what someone reproducing the run reads first.

**Execution scale is not document vocabulary.** Documents and workflows never name devices, hosts, or job systems: engines own intra-run execution, and job dispatch is site tooling outside this repository (spec §8, "Execution scale").

## Shipped documents

The golden-corpus documents ship as user-facing presets in [`causalab/configs/protocols/`](causalab/configs/protocols/) — complete intervention specifications, network and all:

| preset | experiment |
|---|---|
| `harvest` | activation harvesting at named sites/positions |
| `interchange` | interchange intervention + IIA scoring |
| `path_patching` | sender→receiver path patching with off-path freezing |
| `attention_band_patch` | contiguous layer bands in one forward, several bands per document |
| `multi_position_patch` | several writes on one site at disjoint positions |
| `mean_harvest` / `mean_ablation` | harvest a corpus mean at save time, then swap it in |
| `das` | trained orthogonal-subspace interchange (DAS) |
| `dbm` | differential binary masking through a trained gate |
| `random_subspace_control` | the matched-k random subspace every DAS cell is read against |
| `hydra_effect` | resample-ablation + downstream direct-effect probes |
| `probe_generate` / `probe_variable` | greedy-decode under a steer, read the continuation back |
| `weekdays_locate_scan` | layer × position interchange scan (one shared harvest) |
| `weekdays_das_sweep` | k × seed DAS fits at a located cell |
| `weekdays_das_apply` | apply a fitted rotation (ArtifactIdentity-checked) |
| `dbm_apply` | apply a fitted gate — DBM's held-out half of the pair above |

A **fit** document's saved score is its *training* score: `dbm.json`'s and
`weekdays_das_sweep.json`'s `iia.json` are computed over the split they trained
on. The held-out number comes from the matching `*_apply` document, or from the
fit's own `train_eval.json` when `train.eval` declares a split.

[`causalab/configs/protocols/`](causalab/configs/protocols/) holds the shipped documents, one file per experiment; `interchange.json`, `das.json` and `dbm.json` are the three **method families** — interchange, a trained subspace, a trained gate — each carrying its own `train` block (spec §1, `docs/CODEBASE.md` §6). [`causalab/configs/workflows/weekdays_8b.json`](causalab/configs/workflows/weekdays_8b.json) chains locate → select → fit → apply → plots as one workflow document (two step types: `intervention_protocol` and `script`); [`causalab/configs/workflows/mean_ablation.json`](causalab/configs/workflows/mean_ablation.json) is the two-step one — harvest a corpus mean, then ablate that cell to it.

The [joint DBM guide](docs/running_experiments.md)
covers complete neuron gates, frozen replay and JSON export.

## Repository layout

```
causalab/
├── protocol/        # engine-free document layer: load, validate, canonicalize,
│                    #   digest, sweep expansion, engine routing, workflow model, CLI
├── neural/
│   ├── shared/      # what every engine uses: sites, encoding, layouts,
│   │                #   mechanisms, featurizers, metrics, outputs, executor base
│   ├── engines/
│   │   ├── pytorch_hooks/    # the reference engine: hooks, decode, train loop
│   │   └── nnsight_tracing/  # the nnsight engine: traces (the 'nnsight' extra)
│   └── token_positions.py
├── analysis/        # numerical analysis a script step runs (fits, statistics, operands)
├── workflow/        # the workflow runner: run-tree overlay, script invocation, manifest
├── causal/          # causal model primitives
├── tasks/           # task definitions (causal models + counterfactual generators)
├── io/              # disk I/O + plotting primitives
└── configs/         # protocols/ (one document per experiment) + workflows/
                    #   — JSON, no Python config system
demos/               # one markdown demo per research question + its documents
docs/                # the two specs, plus CODEBASE.md, TESTS.md and the demo /
                     #   experiment guides — read the directory for the rest
scripts/             # maintenance entry points (dataset build, digest re-pinning)
tests/               # tiered suite — see docs/TESTS.md
```

## Core concepts

For multi-token behaviors, [shared sequence analysis](docs/multi_token_analysis.md)
keeps one next-token target per prediction while sharing intervention forwards,
PCA harvests and logit-lens projections across target views.

[Fourier probes](docs/multi_token_analysis.md#fourier-probes-on-saved-activations)
fit and replay periodic readouts over saved residuals or subspace coordinates.

- **Causal model**: your hypothesis about how the LM solves a task — variables, values, parent–child dependencies, mechanisms (`causalab/causal/`).
- **Task**: a prompt distribution plus a causal model and counterfactual generators (`causalab/tasks/`).
- **Method**: the `method` group of a document — what transfers (sites, reads, writes, intervened models, metrics, training, save) — as opposed to `model` and `data`, which name what it ran on. Two documents running one experiment on two networks differ only there, which a diff shows; there is no separate digest of the group.
- **Intervention protocol**: one experiment as data — which activations are read, which are edited (`swap`, `add_scaled`, `gaussian`, …), in which intervened models, scored by which metrics. Sweeps expand a document into a campaign of points with content-deduped shared work.
- **Workflow**: a chain of protocol executions plus script steps, with dependencies derived from references — never authored ordering. Everything a step declares is published where it lands; there is no save manifest.

## Tests

See [`docs/TESTS.md`](docs/TESTS.md). CPU tiers run with `uv run pytest -m "not golden"` (what CI runs). The `golden` tier runs real models on an accelerator: paper-provenance goldens (`tests/golden/test_paper_goldens.py`), the chat-coherent drift pins (`tests/golden/drift/`), and the two engines' agreement sweep on the real Qwen3.6-35B-A3B (`tests/golden/test_a3b_engine_parity.py`).

## History

The Hydra runner, `analyses/` chains, `methods/` as Python and SLURM dispatch were retired in the protocol refactor ([goodfire-ai/causalab#20](https://github.com/goodfire-ai/causalab/pull/20)). Their intervention cores return as shipped intervention specifications and workflows. The notebook demos return as [`demos/`](demos/) — markdown around runnable documents, since a notebook's reason to exist was carrying its own execution.
