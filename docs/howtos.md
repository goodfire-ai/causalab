# How-to guides

Use these guides when you already have an experiment in mind. Each group gives
templates, field rules and working commands for one kind of task.

| Group | Use it to | Start with |
|---|---|---|
| Causal models | Define a task's causal model with Python equations, families and explicit noise | [Defining causal models](causal-models.md), [task models](../demos/causal_models/README.md) |
| General protocol structure | Write, check and run an intervention or workflow document, and read what it saves | [Running experiments](running_experiments.md) |
| Interp methods | Copy a method template, then configure DAS, DBM, PCA, SAE and the analysis steps | [Method library](../demos/methods/README.md), [method guides](methods/README.md) |
| Performance | Fit a model on several devices or remote compute, pick kernels, and measure speed | [Model parallelism](model_parallelism.md) |
| Models | Look up the architecture and the addressable components of one model | [Qwen3.6-35B-A3B](qwen36_35b_a3b.md) |

The [intervention protocol](intervention_protocol.md) and the
[workflow protocol](workflow_protocol.md) define every field a document can
carry. Look up a field there after a guide or template shows you where it goes.
[Commands and saved results](cli.md) lists every `causalab` command and the
files a run writes.
