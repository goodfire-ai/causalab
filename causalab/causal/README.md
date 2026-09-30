# Causal models

This folder defines and runs causal models. Each model has named variables
with equations that compute their values. An intervention replaces a
variable's equation with a supplied value.

| File | Contents |
| --- | --- |
| [__init__.py](__init__.py) | Exports the names used to define and run models. |
| [model.py](model.py) | Defines equations and stores the model's graph in `CausalModel`. Decorators capture source from files or cached Jupyter cells in `ModelDefinition`. `CausalTrace` computes values and applies interventions. Each `CompiledEquation` holds a function and its possible parents. |
| [domains.py](domains.py) | `Dom` defines allowed values and checks them. `FamilyDom` gives each input in a family a domain. `Exo` marks an explicit noise input. This file also provides bounded enumeration and sampling. |
| [compiler.py](compiler.py) | Copies the configuration and expands bounded loops. It infers domains and finds possible reads between variables before it checks the graph for cycles. Submodels and indexed families are expanded here. |
| [counterfactuals.py](counterfactuals.py) | Defines the example records used by datasets. Functions sample examples, choose interventions, and label results. `CausalModel.label_counterfactual_data` calls the helper in this file. |
| [model_comparison.py](model_comparison.py) | Compares predictions from causal models. It also scores saved intervention results against the expected outputs. |
| [pair_validation.py](pair_validation.py) | Checks prompt pairs and groups of edits. Token checks verify which text changed and which tokens an intervention covers. |
| [scoring.py](scoring.py) | `ScoringSpec` defines accepted answer forms and grading rules. `build_output_tokens` creates forms for values. Other helpers check that recorded scoring settings match those rules. |

Use the public imports when you define a model:

```python
from causalab.causal import CausalModel, Dom, V, mechanism


@mechanism
def double(x: Dom(range(5))):
    result = V(2 * x)
    raw_input = V(str(x), domain=Dom(str))
    raw_output = V(str(result), domain=Dom(str))
    return result


model = CausalModel(double)
trace = model.new_trace({"x": 2})
assert trace["raw_output"] == "4"

trace["result"] = 8
assert trace["raw_output"] == "8"
```

`CausalModel` loads the compiler when it builds a model. Compilation copies the
configuration and prepares the graph. A trace uses the compiled equations
to compute values as they are needed. Each value must satisfy its domain.
The graph includes a parent edge when an equation can read that parent on
an allowed input, including an allowed intervention.

For helper functions, import from the file that owns the operation:

```python
from causalab.causal.counterfactuals import generate_counterfactual_samples
from causalab.causal.model_comparison import distinguishability_report
from causalab.causal.scoring import ScoringSpec
```

Keep `scoring.py` and `pair_validation.py` usable with the Python standard
library. The protocol loader imports both. NumPy and PyTorch are used by
`model_comparison.py` to score saved results. Functions that save or display
examples live in [io/counterfactuals.py](../io/counterfactuals.py).

The [model guide](../../docs/causal-models.md) explains the supported Python
syntax. Examples are in [demos/causal_models](../../demos/causal_models/README.md).

Run the folder's tests from the repository root:

```bash
uv run pytest tests/causal tests/io/test_counterfactuals.py -q
```
