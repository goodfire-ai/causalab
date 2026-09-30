"""Group-disjoint split allocation for one counterfactual table (§2.2).

The shipped per-task generators (``counterfactuals.py::generate_dataset``) draw
base/counterfactual pairs at random. Building a "train" table and a "test" table
by calling one twice at two seeds guarantees nothing: on a large input space the
two draws happen not to collide, on a small one they overlap almost entirely,
and the artifacts look identical either way. The discipline this module
implements is "hold out structure, not just instances".

[`generate_split_dataset`][] partitions the *unique inputs* into disjoint
groups, assigns each group to one split, and forms counterfactual pairs **within**
a split so neither endpoint of a test pair was seen in training. It returns one
example list plus the split each example belongs to — the shape
[`serialize_examples`][causalab.tasks.serialize.serialize_examples] takes — because the result
is **one table** whose rows declare their split, not three files whose
relationship lives in prose.

That is the whole of the design's opinion: a split is a property of a row, so
disjointness is a fact about the bytes that any reader can check, rather than a
claim about how somebody's builder was invoked. There is deliberately no split
*specification* object to persist, pass around or drift — the knobs below are
arguments to a build, and the build's command line is the record of them;
nothing is written beside the table (spec §2.2).

An earlier design had this logic behind a ``SplitSpec`` emitting three
files. What survives is the group-key handling, the
one-prompt-one-label coherence guard and the label-change filter; what does not
is the spec object, the multi-file output and the ``split.manifest.json``
sidecar whose disjointness claim nothing could read (resolution ignores
sidecars — see [`causalab.tasks.serialize`][]).
"""

from __future__ import annotations

import logging
import random
from typing import Any, Callable, Hashable, NamedTuple, Sequence

from causalab.causal.counterfactuals import CounterfactualExample
from causalab.causal.model import CausalModel, CausalTrace
from causalab.tasks.loader import Task, load_task_counterfactuals

logger = logging.getLogger(__name__)

__all__ = ["DEFAULT_FRACTIONS", "SplitDataset", "generate_split_dataset"]

#: The conventional three-way partition. Split *names* are free-form — the
#: library hardcodes none — so a caller wanting k folds passes its own mapping.
DEFAULT_FRACTIONS: dict[str, float] = {"train": 0.6, "val": 0.15, "test": 0.25}


class SplitDataset(NamedTuple):
    """One table's worth of examples, plus the split each one declares.

    ``examples`` and ``splits`` are parallel and go straight to
    [`serialize_examples`][causalab.tasks.serialize.serialize_examples] as ``examples`` and
    ``split=``. ``audit`` is provenance for the manifest: group and pair counts
    per split, and how many pairs each filter removed.
    """

    examples: list[CounterfactualExample]
    splits: list[str]
    audit: dict[str, Any]


def _assert_coherent_prompts(pool: Sequence[CausalTrace]) -> None:
    """One prompt, one label. A ``raw_input`` carrying two different
    ``raw_output`` values would poison whichever split it landed in, and would
    make the disjointness guarantee meaningless besides."""
    seen: dict[str, str] = {}
    for trace in pool:
        prompt, label = str(trace["raw_input"]), str(trace["raw_output"])
        if seen.setdefault(prompt, label) != label:
            raise ValueError(
                f"prompt {prompt!r} carries two labels ({seen[prompt]!r} and "
                f"{label!r}) — the causal model's raw_input is not injective in "
                "its label-driving inputs, so no partition of prompts is sound"
            )


def _deduplicate_by_prompt(pool: Sequence[CausalTrace]) -> list[CausalTrace]:
    """Keep the first trace per model-visible prompt."""
    seen: set[str] = set()
    out: list[CausalTrace] = []
    for trace in pool:
        prompt = str(trace["raw_input"])
        if prompt not in seen:
            seen.add(prompt)
            out.append(trace)
    return out


def _group_key_fn(
    model: CausalModel, group_key: str, resample_variable: str
) -> Callable[[CausalTrace], Hashable]:
    """``trace -> group id`` for the requested ``group_key``.

    ``"input"`` groups by the *prompt*, the model-visible identity, so a task
    whose ``raw_input`` is not injective in its input variables cannot leak the
    same prompt into two splits (grouping by the input tuple would allow it).
    The single-``resample_variable`` case is the exception: a base and its
    counterfactual differ in that variable and therefore in the prompt, so the
    group is the tuple of the *other* inputs, which keeps the pair together.

    Any other value names an input variable to hold out by — every row sharing
    an entity, a template or a source id lands in one split.
    """
    if group_key == "input":
        if resample_variable != "all" and resample_variable in set(model.inputs):
            key_vars = [v for v in model.inputs if v != resample_variable]
            return lambda t: ("vars", *(str(t[v]) for v in key_vars))
        return lambda t: ("prompt", str(t["raw_input"]))
    if group_key not in set(model.inputs):
        raise ValueError(
            f"group_key={group_key!r} is neither 'input' nor an input variable "
            f"of the task (inputs: {sorted(model.inputs)})"
        )
    return lambda t: ("var", str(t[group_key]))


def _partition_groups(
    group_ids: Sequence[Hashable], fractions: dict[str, float], seed: int
) -> dict[str, set[Hashable]]:
    """Assign each unique group to exactly one split, sized by ``fractions``."""
    total = sum(fractions.values())
    if total <= 0:
        raise ValueError(f"fractions must sum to a positive value, got {fractions!r}")
    names = list(fractions)
    unique = sorted(set(group_ids), key=repr)
    random.Random(seed).shuffle(unique)

    n = len(unique)
    counts = {name: int(fractions[name] / total * n) for name in names}
    # Floors leave a remainder; hand it to the largest split so nothing is lost.
    largest = max(names, key=lambda name: fractions[name])
    counts[largest] += n - sum(counts.values())

    out: dict[str, set[Hashable]] = {}
    cursor = 0
    for name in names:
        out[name] = set(unique[cursor : cursor + counts[name]])
        cursor += counts[name]
    return out


def _input_pool(
    task: Task, seed: int, max_inputs: int | None, generator: str
) -> list[CausalTrace]:
    """Coherent, deduplicated pool of base inputs to partition.

    Two sources, and the choice is forced rather than preferential:

    * **Enumeration**, when the input space fits in ``max_inputs``. This is what
      makes a small task honest — with 49 possible prompts there is no way to
      draw 96 endpoint-disjoint pairs, and the pool size is what says so.
    * **The task's own generator**, otherwise, harvesting the unique endpoints
      of the pairs it produces. Not ``CausalModel.sample_input``: for some tasks
      a uniform draw over the input variables is not a *valid* input at all
      (MCQA's answer must appear among its choices, and its
      ``sample_answerable_question`` exists precisely because the generic
      sampler raises), and coherence is task knowledge this module has no
      business re-deriving. Going through the generator means every task's own
      notion of a well-formed input is respected for free.

    The pairs the generator drew are then thrown away — only the endpoints are
    kept. Pairing happens again later, *within* a split, which is the entire
    point of the exercise.
    """
    model = task.causal_model
    # A stochastic model has no enumerable prompt pool without fixed noise.
    # Keep noise independent here by using the task's seeded generator.
    if model.exogenous and max_inputs is None:
        raise ValueError("Specify max_inputs when splitting a stochastic task")
    # Input enumeration streams finite domains; the accepted pool can be small
    # even when an individual domain exceeds the display/materialization bound.
    enumerable = all(model.domains[name].is_finite for name in model.inputs)
    if not enumerable and max_inputs is None:
        # n_unique_inputs has no value for an open domain, and the generator
        # needs a pool size.
        open_inputs = [n for n in model.inputs if not model.domains[n].is_finite]
        raise ValueError(
            f"Specify max_inputs: inputs {open_inputs} cannot be enumerated, "
            "so the pool is drawn from the task's generator"
        )
    cap = max_inputs if max_inputs is not None else model.n_unique_inputs
    fits = not model.exogenous and enumerable and model.count_inputs(limit=cap) <= cap
    if fits:
        pool = list(model.enumerate_inputs())
    else:
        generators = load_task_counterfactuals(task.name)
        if not hasattr(generators, generator):
            raise ValueError(
                f"task {task.name!r} has no counterfactual generator {generator!r}"
            )
        state = random.getstate()
        random.seed(seed)
        try:
            # Each pair contributes up to two distinct endpoints, and duplicates
            # are common on a small space, so draw generously and truncate.
            examples = getattr(generators, generator)(model, cap, seed)
        finally:
            random.setstate(state)
        pool = []
        for example in examples:
            pool.append(example["input"])
            pool.extend(example["counterfactual_inputs"])
        pool = _deduplicate_by_prompt(pool)[:cap]
    _assert_coherent_prompts(pool)
    return _deduplicate_by_prompt(pool)


def _resample_one_variable(
    model: CausalModel, base: CausalTrace, variable: str, rng: random.Random
) -> CausalTrace | None:
    """A full trace equal to ``base`` but for ``variable``, resampled.

    Recomputes the trace so the counterfactual is a *coherent input* — the
    mechanisms rerun — rather than an in-place intervention. ``None`` when the
    variable admits no other value the model's ``input_filter`` accepts.
    """
    domain = model.domains[variable]
    values = domain.enumerated()
    if values is None:
        # Large ranges (including explicit noise) support cheap sampling. Never
        # materialize the range just to resample one input.
        candidates = [domain.sample(rng) for _ in range(64)]
    else:
        candidates = [v for v in values if v != base[variable]]
        rng.shuffle(candidates)
    candidates = [v for v in candidates if v != base[variable]]
    base_inputs = {v: base[v] for v in model.inputs}
    for value in candidates:
        inputs = {**base_inputs, variable: value}
        if model.input_filter is None or model.input_filter(
            CausalTrace(model, inputs, eager=False)
        ):
            return model.new_trace(inputs)
    return None


def _counterfactual_within(
    model: CausalModel,
    base: CausalTrace,
    resample_variable: str,
    allowed_prompts: set[str],
    in_split: Sequence[CausalTrace],
    rng: random.Random,
) -> CausalTrace | None:
    """One counterfactual for ``base`` whose prompt stays inside the split."""
    if resample_variable == "all":
        candidates = [
            t for t in in_split if str(t["raw_input"]) != str(base["raw_input"])
        ]
        return rng.choice(candidates) if candidates else None
    # The resampled value is drawn at random and may land out of split; retry a
    # bounded number of times before giving this base up.
    for _ in range(20):
        cf = _resample_one_variable(model, base, resample_variable, rng)
        if cf is None:
            return None
        if str(cf["raw_input"]) in allowed_prompts:
            return cf
    return None


def _label_changes(
    examples: Sequence[CounterfactualExample], task: Task, target_variable: str
) -> list[bool]:
    """Which pairs the interchange actually moves the answer for (symbolic)."""
    if not examples:
        return []
    labeled = task.causal_model.label_counterfactual_data(
        list(examples), [target_variable]
    )
    return [
        str(lab["label"]) != str(ex["input"]["raw_output"])
        for ex, lab in zip(examples, labeled)
    ]


def generate_split_dataset(
    task: Task,
    *,
    seed: int = 42,
    fractions: dict[str, float] | None = None,
    group_key: str = "input",
    resample_variable: str = "all",
    max_inputs: int | None = None,
    max_pairs_per_split: dict[str, int] | None = None,
    generator: str = "generate_dataset",
    require_label_change: bool = False,
    target_variable: str | None = None,
    keep_pair: Callable[[CounterfactualExample], bool] | None = None,
) -> SplitDataset:
    """One table of counterfactual pairs whose splits are group-disjoint.

    Args:
        task: The loaded task to draw from.
        seed: Seeds the group partition, the pairing and any cap sampling.
        fractions: ``{split name: weight}``; defaults to
            [`DEFAULT_FRACTIONS`][]. Weights are normalized, so
            ``{"train": 2, "test": 1}`` is fine. Names are free-form.
        group_key: ``"input"`` holds out by prompt; an input-variable name holds
            out by that variable's value (every row sharing it lands together).
        resample_variable: ``"all"`` pairs a base with a different in-split
            input; a variable name resamples just that variable.
        max_pairs_per_split: Optional cap on kept pairs, applied *after* the
            filters so it bounds what survives rather than what was tried.
        generator: Which of the task's own generators supplies the input pool
            when the space is too large to enumerate (see `_input_pool`).
        require_label_change: Drop pairs whose interchange leaves the answer
            unchanged, which would otherwise inflate IIA with free correctness.
        target_variable: What ``require_label_change`` interchanges; defaults to
            the task's intervention variable.
        keep_pair: An arbitrary last predicate. This is the seam a
            model-correctness filter plugs into — build the closure over your
            loaded model outside and pass it in, so the table itself keeps no
            model dependency (``tests/tasks/test_serialize.py`` pins that).

    Returns:
        A [`SplitDataset`][]. Feed ``examples`` and ``splits`` straight to
        [`serialize_examples`][causalab.tasks.serialize.serialize_examples].
    """
    fractions = dict(fractions or DEFAULT_FRACTIONS)
    model = task.causal_model
    target = target_variable or task.intervention_variable
    if require_label_change and target is None:
        raise ValueError(
            "require_label_change needs a target_variable (none given, and the "
            "task declares no intervention variable)"
        )
    if resample_variable != "all" and group_key == resample_variable:
        raise ValueError(
            f"group_key={group_key!r} equals resample_variable — a "
            "single-variable counterfactual always changes that variable, so it "
            "always lands in another group and every split would be empty"
        )

    pool = _input_pool(task, seed, max_inputs, generator)
    key_of = _group_key_fn(model, group_key, resample_variable)
    groups = _partition_groups([key_of(t) for t in pool], fractions, seed)

    rng = random.Random(seed)
    examples: list[CounterfactualExample] = []
    splits: list[str] = []
    per_split: dict[str, dict[str, int]] = {}

    for name in fractions:
        in_split = [t for t in pool if key_of(t) in groups[name]]
        allowed = {str(t["raw_input"]) for t in in_split}
        paired = [
            {"input": base, "counterfactual_inputs": [cf]}
            for base in in_split
            if (
                cf := _counterfactual_within(
                    model, base, resample_variable, allowed, in_split, rng
                )
            )
            is not None
        ]
        kept: list[CounterfactualExample] = list(paired)
        if require_label_change:
            assert target is not None
            changes = _label_changes(kept, task, target)
            kept = [ex for ex, moved in zip(kept, changes) if moved]
        n_after_label = len(kept)
        if keep_pair is not None:
            kept = [ex for ex in kept if keep_pair(ex)]
        n_after_keep = len(kept)
        cap = (max_pairs_per_split or {}).get(name)
        if cap is not None and len(kept) > cap:
            kept = rng.sample(kept, cap)

        if not kept:
            logger.warning(
                "Split %r is empty: %d groups, %d inputs, %d pairs before filters.",
                name,
                len(groups[name]),
                len(in_split),
                len(paired),
            )
        examples.extend(kept)
        splits.extend([name] * len(kept))
        per_split[name] = {
            "n_groups": len(groups[name]),
            "n_inputs": len(in_split),
            "n_paired": len(paired),
            "n_after_label_change": n_after_label,
            "n_after_keep_pair": n_after_keep,
            "n_final": len(kept),
        }

    _assert_disjoint(examples, splits, key_of, groups)

    audit = {
        "seed": seed,
        "fractions": fractions,
        # Groups are assigned and splits filled in the order the caller gave
        # the fractions, and a JSON manifest sorts its keys — so the order is
        # recorded on its own, or a rebuild from the manifest would not be one.
        "split_order": list(fractions),
        "group_key": group_key,
        "resample_variable": resample_variable,
        "max_inputs": max_inputs,
        "max_pairs_per_split": dict(max_pairs_per_split)
        if max_pairs_per_split
        else None,
        "pool_generator": generator,
        "require_label_change": require_label_change,
        "target_variable": target,
        "keep_pair": getattr(keep_pair, "__name__", None) if keep_pair else None,
        "pool_size": len(pool),
        "n_unique_groups": len({key_of(t) for t in pool}),
        "per_split": per_split,
    }
    logger.info("Split dataset (group_key=%s): %s", group_key, per_split)
    return SplitDataset(examples=examples, splits=splits, audit=audit)


def _assert_disjoint(
    examples: Sequence[CounterfactualExample],
    splits: Sequence[str],
    key_of: Callable[[CausalTrace], Hashable],
    groups: dict[str, set[Hashable]],
) -> None:
    """The promise, checked on the way out.

    Cheap next to generation, and it fails at the build rather than leaving a
    contaminated table to be discovered by whoever reads the IIA later.
    """
    seen: set[Hashable] = set()
    for name, ids in groups.items():
        if seen & ids:
            raise AssertionError(f"group partition leaked into {name!r}")
        seen |= ids
    for example, name in zip(examples, splits):
        for endpoint in (example["input"], example["counterfactual_inputs"][0]):
            if key_of(endpoint) not in groups[name]:
                raise AssertionError(
                    f"a {name!r} pair has an endpoint from another split's group"
                )
