# Function-based causal models

This page indexes the causal models of the bundled tasks and the examples of
the equation syntax. Each model is Python code, and none of them needs a neural
network or an accelerator.

## TL;DR

Import model definitions from `causalab.causal`. The decorators in `model.py`
read the function source. `CausalModel(equations, ...)` compiles the graph and
stores its metadata. The [folder README](../../causalab/causal/README.md) explains
where the code lives. Open [defining_models.ipynb](defining_models.ipynb) to run
an example in Jupyter.

## The protocol

The task implementations themselves are the examples, including their prompt
and output nodes and existing metadata:

| Task | Source | Preserved mechanism |
| --- | --- | --- |
| Natural-domain arithmetic | [Equations](../../causalab/tasks/natural_domains_arithmetic/causal_models.py) | Entity + number → result; six domains, grouping, random-word baseline |
| Hierarchical equality | [Equations](../../causalab/tasks/hierarchical_equality/causal_models.py) | Two pair equalities → result equality; explicit ICL seed |
| Identity naming | [Equations](../../causalab/tasks/identity_naming/causal_models.py) | Entity → canonical name |
| Subject–object relations | [Equations](../../causalab/tasks/subject_object_relations/causal_models.py) | Subject → object, across the bundled relations |
| Hex color | [Equations](../../causalab/tasks/hex_color/causal_models.py) | Hex stimulus → color |
| IOI | [Equations](../../causalab/tasks/IOI/causal_models.py) | Three names → indirect object |
| MCQA | [Equations](../../causalab/tasks/MCQA/causal_models.py) | Choice family + color → answer position → symbol |
| Entity binding | [Equations](../../causalab/tasks/entity_binding/causal_models.py) | Entity/query/position families → positional answer → output retrieval |
| Graph walk | [Equations](../../causalab/tasks/graph_walk/causal_models.py) | Coordinates + explicit seed → one walk_sequence node; answer still depends on coordinates |

Additional examples:

- [Arithmetic hypotheses](arithmetic_demos.py): compare the results of swaps.
- [Intervention walkthrough](../onboarding_tutorial/03_causal_model.md): use intermediate values to compare algorithms.
- [Bounded steps and optional values](control_flow.py): carry stopped state and raise on bound exhaustion.
- [People and food](people_food_binding.py): helper namespaces and explicit handling of None.
- [Modulo-three hypothesis demo](../hypothesis_testing/models.py): preserved a_value/b_value intervention targets.

### Migration details

The hand-authored `Mechanism` / `input_var` dictionaries and the two-dictionary
constructor have been removed. Methods and metadata used by the rest of the
repository remain available, including `new_trace`, `sample_input`,
`run_interchange`, `parents`, `children`, `variables`, `values`, and scoring.

MCQA input members are now `choices[0]`, `symbols[0]`, etc. Entity binding uses
`entities[g,r]`, `queries[r]`, `positional_entities[g,r]`, and
`positional_queries[r]`. Its `positional_answer` remains the intervention target.
Unused declared parents are removed. Empty query matches remain `()`; ambiguous
positional answers remain `None`. Input domains admit inactive entries and
explicit intervention domains retain every permitted position, even where the
ordinary equation produces just the original group index.

Graph walk adds `walk_seed`; hierarchical equality adds `icl_seed`. Generators
supply independent base and donor noise and serialized traces retain it.
`new_trace` does not draw hidden noise. `enumerate_inputs(noise={...})` and
`count_inputs(noise={...})` require it fixed; split generation uses seeded task
sampling for stochastic models and requires `max_inputs`.

## Run it

Open [defining_models.ipynb](defining_models.ipynb) in Jupyter, or run the CPU checks from the repository root:

```bash
uv run pytest tests/causal/test_equations.py tests/causal/test_compiler.py -q
```

## Experimental design

The checks compare accepted expressions with Python evaluation, exhaust allowed
interventions on conditional nodes, and check the union of reads across inputs.
They also exercise independent traces, configuration snapshots, notebook
redefinition, fixed noise, and bounded iteration.

## Results

The examples produce the expected observational and intervened values. Invalid
computed values raise errors, and changing a caller's configuration or another
trace does not change a compiled model's equations.

## Limits

The compiler accepts a bounded subset of Python. Put generators, mutable private
algorithms, and dynamic callable selection in ordinary pure helpers. This demo
does not run neural experiments or establish an empirical causal hypothesis.

## Next

Start with the [model guide](../../docs/causal-models.md), then adapt the
closest task's equations and declare every allowed intervention value.
