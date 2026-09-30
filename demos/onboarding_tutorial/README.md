# Onboarding tutorial

The onboarding tutorial teaches causalab through one sequence of experiments.
It starts with two zero-ablation warm-ups and two comparisons of causal
hypotheses with no network in them. Then nine demos study one multiple-choice
task on `Qwen/Qwen2.5-1.5B-Instruct` (28 layers, 1536 wide, 12 query heads
over 2 key-value heads). Each demo takes something concrete from the one
before it.

| | demo | question | needs a GPU | measured |
|---|---|---|---|---|
| 01 | [Ablate MLP layers](01_ablation_MLP.md) | which MLP layers of GPT2-XL carry the location of the Space Needle? | not stated | not stated |
| 02 | [Ablate attention heads](02_ablation_attention.md) | do four attention heads carry the answer symbol of one question? | no | a few minutes on the CPU; shipped tables from `--device mps` |
| 03 | [What an intervention buys](03_causal_model.md) | two algorithms agree on every input — does an intervention tell them apart? | no | 0.28 s, CPU |
| 03a | [Saved hypothesis comparisons](../hypothesis_testing/hypothesis_testing.md) | do exact pairs, symbolic predictions and saved neural outputs agree at the handoff? | no | not stated |
| 04 | [Define the task](04_define.md) | does this counterfactual dataset tell two variables apart? | no | < 1 s, CPU |
| 05 | [Trace one pair](05_trace.md) | for one pair, which layers and positions carry the answer symbol? | yes, a small one | 13 s, MacBook Pro GPU (26 s, 1×H100) |
| 06 | [Localize the variable](06_localize.md) | does the handoff of 05 hold across 64 pairs? | yes, a small one | 35 s, MacBook Pro GPU |
| 07 | [How few directions?](07_subspace.md) | how many directions does an interchange need on held-out pairs? | yes | 95 s, MacBook Pro GPU |
| 08 | [Variance against cause](08_variance_vs_cause.md) | do the top-variance directions carry the answer symbol? | yes | 10 s, MacBook Pro GPU |
| 09 | [Which component writes it?](09_components.md) | attention or MLP, and how many stream dimensions does a mask keep? | yes | 46 s, MacBook Pro GPU |
| 10 | [Cross-model grafting](10_cross_model.md) | can one checkpoint's activation replace another's? | yes | 5 s per workflow, MacBook Pro GPU |
| 11 | [Which head?](11_attention.md) | the pattern or the value, and which heads? | yes | 10 s, MacBook Pro GPU |
| 12 | [Necessity and sufficiency](12_steering.md) | which layers does the answer slot need, and does a mean direction steer it? | yes | 8 s, MacBook Pro GPU |

03 has no model in it. It is the argument for the demos after it: behaviour
cannot separate two hypotheses that compute the same function, and an
interchange on an intermediate variable separates these two on 913 of 1000
pairs.

**Start the task demos with 04.** It runs on a laptop in under a second, and
it decides whether the demos after it measure anything.

**07 through 12 study the component where 06 measured the highest IIA**,
block 26's output at the answer slot. Each varies one thing: the subspace (07,
08), the site (09, 11), the model (10), or the intervention (12). 11 scans
layers 20 to 27 because 09 finds the attention sublayer that writes the symbol
at layer 22, four blocks before block 26.
