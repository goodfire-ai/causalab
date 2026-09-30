"""Build the CausalToM tables of the Figures 4b, 5b, 6b and 13 replication.

Prakash et al. 2025 (arXiv:2505.14685) localize the belief-tracking mechanism
of an instruction-tuned LM with interchange interventions on *CausalToM*
stories: two characters each fill an opaque container with a drink, and the
model is asked what one character believes one container holds. The three
main-text figures use two counterfactual designs, so two tables follow, each
over the paper's own 80 validation pairs of that design:

* ``data_reorder.json``: Figure 5. The counterfactual narrates the **same**
  two (character, object, state) triples in the **other order**; nothing else
  changes, and the question is identical. Every word is the same on both
  sides, so each word's ordering ID (OI: first or second to appear) is the
  only thing that differs.
* ``data_restate.json``: Figures 4, 6 and 13. The counterfactual narrates
  the two triples in the other order **and replaces both states with fresh
  ones**, so the counterfactual's answer is a word the original story does
  not contain.

Two more tables, ``data_restate_screen.json`` and
``data_reorder_screen.json``, hold candidates 0 to 79 of each design in one
split ``screen``. No figure uses them. The workflow's ``screen_*`` steps
score the un-intervened answer on both prompts of each pair, which shows in
the package that the model answers every one of the first 160 candidates
(the paragraph below). The screening pairs are a table of their own per
design, since both designs draw candidate 0 from the same first random
calls, and a prompt may not repeat across the splits of one table.

**The paper's pairs.** The authors' patching scripts (``Nix07/mind``,
``scripts/run_single_layer_patching_exps.py`` and
``run_upto_layer_patching_exps.py`` at 3d38e1b) call ``set_seed(123456)`` and
then ``prepare_dataset``, which draws ``2 * (80 + 80) = 320`` candidate pairs
with one generator, keeps the first 160 on which the model answers both
prompts correctly, and validates on the last 80 of them
(``scripts/patching_scripts/run_patching_exp_utils.py`` at 0579347).
`authors_candidates` makes the generator's random calls in the generator's
order, so its candidates are the authors' candidates. Llama-3-70B-Instruct at
fp16 answers both prompts of each of the first 160 candidates of both
designs, so the validation pairs are candidates 80 to 159 (``VALID``). The
baseline document checks both answers on every run: the ``baseline_*`` steps
over ``VALID``, the ``screen_*`` steps over candidates 0 to 79 (``SCREEN``).
The shipped values predate these scripts (they first appear at ``Nix07/mind``
9f84103, a month before 3d38e1b). So the seed of the later scripts is not
proof on its own. The pairs are named by that seed together with the match:
on these pairs the package's curves equal the paper's values
(``demos/papers/lookbacks.md``).

The causal model (`belief_model`) is Algorithm 2 of the paper's
Appendix G with one change of indexing that the interventions require: the
two triples are named **a** and **b** by *identity* (the same words on both
sides of a pair, whatever their order) and the narration ``order`` is an
input variable, so every OI is a derived variable. That is what lets each
figure's expected answer be a plain interchange of same-named causal
variables from the counterfactual into the original:

======================  ==================================================
column                  the causal model under
======================  ==================================================
``label``               ``data_restate``: ``answer_payload`` interchanged
                        (Figure 4, gray curve): the counterfactual's own
                        answer. ``data_reorder``: the binding-lookback
                        variables ``address_a``, ``address_b``,
                        ``state_oi_a``, ``state_oi_b`` interchanged
                        (Figure 5): the original story's *other* state
``label_pointer``       ``answer_pointer`` interchanged (Figure 4, colored
                        curve): the original's state at the OI the
                        counterfactual's answer has
``label_source``        the source OIs ``char_oi_*`` / ``obj_oi_*``
                        interchanged with the state tokens' variables held
                        at the original's values (Figure 6): the
                        original story's other state
======================  ==================================================

Each label equals the ``target`` the authors' generator writes: the original
story's other state for ``label_pointer``, ``label_source`` and the reorder
``label``.

Every row also carries the **token anchors** the intervention documents
position on (``intro_a``, ``char_a_action``, ``obj_a_action``,
``state_a_mention`` and their ``_b`` twins): derived string variables that
occur exactly once in each prompt and name the same word on both sides of
the pair, which is how a ``{"variable": ...}`` position patches *by identity*
(the original's ``beer`` token receives the counterfactual's ``beer`` token,
wherever it sits) rather than by ordinal slot. The object anchor is
``"<object> and fills"``: the instruction says "the container and its
contents", so ``"container and"`` would occur twice in the prompts whose
object is ``container``. The documents patch the anchor's first two tokens,
the object and ``and``.

Prompt, templates and word lists are the authors' (``src/dataset.py``,
``data/story_templates.json`` template 2, ``data/synthetic_entities/``), fed
to the model as plain text with no chat template, as in the paper's code. Both
designs use template 2, as both generators at 0579347 do; the earlier 5b
generator (``get_state_pos_exps`` at 3d38e1b) uses template 0, which appends
two "cannot observe" sentences, and its pairs do not give the paper's 5b
values on the authors' own code. Every character, object and state is one
Llama-3 token with a leading space.

Usage::

    python workflows/scripts/lookbacks/build_dataset.py --out artifacts/data/lookbacks            # writes the four tables
    python workflows/scripts/lookbacks/build_dataset.py --out artifacts/data/lookbacks --check    # exit 1 if the committed bytes differ

Nothing is written beside a table: the recipe is this file and its constants
(``SEED``, ``CANDIDATES``, ``VALID``, ``SCREEN``, ``SPLITS``, the word
lists), and the workflow that consumes a table pins its content digest.
"""

from __future__ import annotations

import argparse
import hashlib
import random
import sys
from pathlib import Path
from typing import Any, Mapping

from causalab.causal import CausalModel, CausalTrace, Dom, V, mechanism
from causalab.causal.scoring import ScoringSpec
from causalab.tasks.serialize import (
    serialize_examples,
    table_bytes,
    write_dataset_table,
)

TASK = "lookbacks_belief_tracking"
GENERATOR = "build_dataset.py"
PAPER = "arXiv:2505.14685, Figures 4, 5, 6; Section 3.1, Appendices A, B, G"

#: The authors' seed and draw: ``set_seed(123456)`` before ``prepare_dataset``,
#: which asks the generator for ``2 * (train_size + valid_size)`` candidates
#: with ``train_size = valid_size = 80``.
SEED = 123456
CANDIDATES = 320
#: The paper's validation pairs among the candidates: the model answers every
#: one of the first 160 at fp16 (the authors' filter, run on the authors' code
#: at fp16), so ``prepare_dataset``'s training split is candidates 0 to 79 and
#: its validation split candidates 80 to 159.
VALID = range(80, 160)
N = len(VALID)
#: The candidates before ``VALID``, which the authors' filter keeps as its
#: training split. The ``*_screen.json`` tables hold them so that the
#: workflow's baseline can show the model answers each of them.
SCREEN = range(0, 80)
#: The rows are two splits of 40, ``first`` and ``second``, and every document
#: names one: the reference engine holds every tapped hidden state on the GPU
#: until a point is done, and the Figure 6 document taps all 80 block outputs
#: of two forwards. The workflow runs each step once per split; the figure
#: joins them.
SPLITS = ("first", "second")
#: The one split of each ``*_screen.json`` table.
SCREEN_SPLIT = "screen"

# The authors' lists (data/synthetic_entities/{characters,bottles,drinks}.json).
CHARACTERS = [
    "Dean",
    "Beth",
    "Jake",
    "Josh",
    "Karen",
    "Carl",
    "Lee",
    "Pam",
    "Donna",
    "Frank",
    "Diane",
    "Bob",
    "Ellen",
    "Ivy",
    "Pete",
    "Neil",
    "Cathy",
    "Rachel",
    "Kate",
    "Fiona",
    "Grace",
    "Gary",
    "Mike",
    "Tom",
    "Megan",
    "Anna",
    "Tony",
    "David",
    "Dave",
    "Sean",
    "Hans",
    "Chris",
    "Linda",
    "Oscar",
    "Gwen",
    "Liz",
    "Alice",
    "Rick",
    "Emma",
    "John",
    "Olivia",
    "James",
    "Scott",
    "Kevin",
    "Kim",
    "Amy",
    "Wayne",
    "Peter",
    "Paul",
    "Jean",
    "Adam",
    "Rose",
    "Ian",
    "Phil",
    "Tim",
    "Ruth",
    "Jeff",
    "Tina",
    "Zoe",
    "Nancy",
    "Lily",
    "Jack",
    "Ray",
    "Ryan",
    "Henry",
    "Keith",
    "Sarah",
    "Doug",
    "Fred",
    "Helen",
    "Eve",
    "Uma",
    "Mark",
    "Max",
    "Bill",
    "Heidi",
    "Sam",
    "Eric",
    "Rob",
    "Susan",
    "Matt",
    "Charlie",
    "Lisa",
    "Sue",
    "Mary",
    "Ken",
    "Jim",
    "Dan",
    "Kyle",
    "Laura",
    "Alex",
    "Ben",
    "Greg",
    "Nick",
    "Quinn",
    "Will",
    "Jane",
    "Joe",
    "Luke",
    "Chad",
    "Sara",
    "Steve",
    "Julia",
]
#: The authors' 21 objects, in their order.
OBJECTS = [
    "jar",
    "cup",
    "mug",
    "glass",
    "flute",
    "pitcher",
    "jug",
    "bottle",
    "can",
    "flask",
    "pint",
    "quart",
    "horn",
    "tun",
    "urn",
    "dispenser",
    "vat",
    "tank",
    "drum",
    "tote",
    "container",
]
STATES = [
    "water",
    "milk",
    "tea",
    "beer",
    "soda",
    "juice",
    "coffee",
    "wine",
    "gin",
    "rum",
    "champagne",
    "cocktail",
    "punch",
    "espresso",
    "cocoa",
    "sprite",
    "monster",
    "bourbon",
    "port",
    "float",
    "stout",
    "ale",
    "porter",
]

INSTRUCTION = (
    "1. Track the belief of each character as described in the story. "
    "2. A character's belief is formed only when they perform an action themselves "
    "or can observe the action taking place. 3. A character does not have any "
    "beliefs about the container and its contents which they cannot observe. "
    "4. To answer the question, predict only what is inside the queried container, "
    "strictly based on the belief of the character, mentioned in the question. "
    "5. If the queried character has no belief about the container in question, "
    "then predict 'unknown'. 6. Do not predict container or character as the final "
    "output."
)
STORY = (
    "{c1} and {c2} are working in a busy restaurant. To complete an order, {c1} "
    "grabs an opaque {o1} and fills it with {s1}. Then {c2} grabs another opaque "
    "{o2} and fills it with {s2}."
)
QUESTION = "What does {cq} believe the {oq} contains?"

TRIPLES = ("a", "b")
ORDERS = ("ab", "ba")
UNKNOWN = "unknown"


def _first_second(order: str) -> tuple[str, str]:
    return (order[0], order[1])


def _oi(order: str, triple: str) -> int:
    """The ordering ID of a triple: 1 if it is narrated first, else 2."""
    return 1 + order.index(triple)


def _story(order, char_a, obj_a, state_a, char_b, obj_b, state_b) -> str:
    """The story, with the triple ``order`` narrates first in the first slot."""
    words = {"a": (char_a, obj_a, state_a), "b": (char_b, obj_b, state_b)}
    first, second = _first_second(order)
    (c1, o1, s1), (c2, o2, s2) = words[first], words[second]
    return STORY.format(c1=c1, o1=o1, s1=s1, c2=c2, o2=o2, s2=s2)


def _prompt(story: str, query_char: str, query_obj: str) -> str:
    question = QUESTION.format(cq=query_char, oq=query_obj)
    return (
        f"Instruction: {INSTRUCTION}\n\nStory: {story}\nQuestion: {question}\nAnswer:"
    )


def _intro(order: str, triple: str, char_a: str, char_b: str) -> str:
    """ "Story: Bob" when ``triple`` is narrated first, "Carla and Bob" when
    second. Each occurs once, and its last token is the intro mention."""
    chars = {"a": char_a, "b": char_b}
    first, _ = _first_second(order)
    if triple == first:
        return f"Story: {chars[triple]}"
    return f"{chars[first]} and {chars[triple]}"


def belief_model() -> CausalModel:
    """Algorithm 2 (Appendix G) over identity-indexed triples.

    Inputs: ``char_a, obj_a, state_a, char_b, obj_b, state_b`` (the two
    triples), ``order`` (``ab`` narrates a first), ``q`` (the triple whose
    character and container the question names). Derived: the source OIs at
    the character and object tokens, the binding address and the state OI at
    each state token, the pointer copies at the question, the binding and
    answer lookbacks, the prompt, the answer, and the token anchors."""

    @mechanism
    def equations(
        char_a: Dom(CHARACTERS),
        obj_a: Dom(OBJECTS),
        state_a: Dom(STATES),
        char_b: Dom(CHARACTERS),
        obj_b: Dom(OBJECTS),
        state_b: Dom(STATES),
        order: Dom(list(ORDERS)),
        q: Dom(list(TRIPLES)),
    ):
        # -- source reference: the OIs at the character and object tokens
        char_oi_a = V(_oi(order, "a"))
        obj_oi_a = V(_oi(order, "a"))
        char_oi_b = V(_oi(order, "b"))
        obj_oi_b = V(_oi(order, "b"))
        # -- at each state token: the binding address (a copy of the source
        #    OIs) and the state's own OI (binding payload / answer address)
        state_oi_a = V(_oi(order, "a"))
        state_oi_b = V(_oi(order, "b"))
        address_a = V((char_oi_a, obj_oi_a))
        address_b = V((char_oi_b, obj_oi_b))
        # -- the question, and the pointer copies made from it. The pointer
        #    copy is looked up by the *word* in the question (Algorithm 2
        #    lines 10-11), not by the triple index.
        query_char = V(char_a if q == "a" else char_b)
        query_obj = V(obj_a if q == "a" else obj_b)
        query_char_oi = V(char_oi_a if query_char == char_a else char_oi_b)
        query_obj_oi = V(obj_oi_a if query_obj == obj_a else obj_oi_b)
        binding_pointer = V((query_char_oi, query_obj_oi))
        # -- binding lookback: dereference the pointer at the state tokens
        binding_payload = V(
            state_oi_a
            if address_a == binding_pointer
            else (state_oi_b if address_b == binding_pointer else None)
        )
        # -- answer lookback: the state OI is the pointer, the state its payload
        answer_pointer = V(binding_payload)
        answer_payload = V(
            state_a
            if answer_pointer is not None and state_oi_a == answer_pointer
            else (
                state_b
                if answer_pointer is not None and state_oi_b == answer_pointer
                else UNKNOWN
            ),
            domain=Dom(STATES + [UNKNOWN]),
        )
        # -- text
        story = V(
            _story(order, char_a, obj_a, state_a, char_b, obj_b, state_b),
            domain=Dom(str),
        )
        raw_input = V(_prompt(story, query_char, query_obj), domain=Dom(str))  # noqa: F841
        raw_output = V(" " + answer_payload, domain=Dom(str))  # noqa: F841
        # -- token anchors: one occurrence per prompt, the same word on both
        #    sides of a pair (the module docstring)
        intro_a = V(_intro(order, "a", char_a, char_b), domain=Dom(str))  # noqa: F841
        intro_b = V(_intro(order, "b", char_a, char_b), domain=Dom(str))  # noqa: F841
        char_a_action = V(f"{char_a} grabs", domain=Dom(str))  # noqa: F841
        char_b_action = V(f"{char_b} grabs", domain=Dom(str))  # noqa: F841
        obj_a_action = V(f"{obj_a} and fills", domain=Dom(str))  # noqa: F841
        obj_b_action = V(f"{obj_b} and fills", domain=Dom(str))  # noqa: F841
        state_a_mention = V(f"{state_a}.", domain=Dom(str))  # noqa: F841
        state_b_mention = V(f"{state_b}.", domain=Dom(str))  # noqa: F841
        return answer_payload

    scoring = ScoringSpec(
        forms={"answer_payload": {w: tuple(forms(w)) for w in STATES + [UNKNOWN]}},
        string_mode="exact",
    )
    return CausalModel(equations, id=TASK, scoring=scoring)


def authors_candidates(*, restate: bool) -> list[dict[str, Any]]:
    """The authors' 320 candidate pairs of one design, in their order.

    The random calls are those of ``get_reversed_sent_diff_state_counterfacts``
    (``restate``) and ``get_reversed_sentence_counterfacts`` (not
    ``restate``) in ``notebooks/causalToM_novis/utils.py`` of ``Nix07/mind``
    at 0579347 (https://github.com/Nix07/mind), made on one
    ``random.Random(SEED)`` as their global ``random`` after ``set_seed``:
    for every candidate two characters, two objects and two states, then the
    restate design's two fresh states, redrawn until neither is an original
    state; after all candidates, one query index each. The original narrates
    ``characters[0]`` first, and the question names the character and object
    at the query index. The counterfactual narrates the two in the other
    order, with the fresh states in narration order or the original's
    states."""
    rng = random.Random(SEED)
    drawn = []
    for _ in range(CANDIDATES):
        characters = rng.sample(CHARACTERS, 2)
        objects = rng.sample(OBJECTS, 2)
        states = rng.sample(STATES, 2)
        if restate:
            fresh = rng.sample(STATES, 2)
            while fresh[0] in states or fresh[1] in states:
                fresh = rng.sample(STATES, 2)
        else:
            fresh = list(reversed(states))
        drawn.append((characters, objects, states, fresh))
    queries = [rng.choice([0, 1]) for _ in range(CANDIDATES)]
    return [
        {
            "characters": characters,
            "objects": objects,
            "states": states,
            "counterfactual_states": fresh,
            "query": query,
        }
        for (characters, objects, states, fresh), query in zip(drawn, queries)
    ]


def pair(model: CausalModel, candidate: Mapping[str, Any]) -> dict[str, Any]:
    """One candidate as an original trace and its counterfactual. Triple
    **a** is the one the original narrates first; the counterfactual
    narrates b first, so its first state is b's."""
    (char_a, char_b), (obj_a, obj_b) = candidate["characters"], candidate["objects"]
    state_a, state_b = candidate["states"]
    cf_state_b, cf_state_a = candidate["counterfactual_states"]
    common = {
        "char_a": char_a,
        "char_b": char_b,
        "obj_a": obj_a,
        "obj_b": obj_b,
        "q": TRIPLES[candidate["query"]],
    }
    base = model.new_trace(
        {**common, "state_a": state_a, "state_b": state_b, "order": "ab"}
    )
    counterfactual = model.new_trace(
        {**common, "state_a": cf_state_a, "state_b": cf_state_b, "order": "ba"}
    )
    return {"input": base, "counterfactual_inputs": [counterfactual]}


def authors_target(candidate: Mapping[str, Any]) -> str:
    """The ``target`` the authors' generator writes for a candidate: the
    original's state at the other query index, space-prefixed."""
    return " " + candidate["states"][1 ^ candidate["query"]]


def forms(word: str) -> list[str]:
    """The surface forms a `match` metric credits: the space-prefixed word in
    lower case and capitalized.

    The prompt ends in ``Answer:``, so the answer token carries a leading
    space. The instruction-tuned model answers ``" Monster"`` for ``monster``
    on most prompts, and the authors' code compares ``.lower().strip()``
    (``run_single_layer_patching_exps.py``), so both cases score. A metric
    tokenizes each form as written and refuses a form that is not one token.
    The bare forms are left out. ``"soda"`` is two Llama-3 tokens, and a bare
    form that is one token, such as ``"water"``, names a different row from
    ``" water"``. The retired ``token_form: "space_prefixed"`` mapped every
    form onto its space-prefixed spelling, so the two forms here resolve to
    the ids that the four forms resolved before. Each form here, for every
    state and ``unknown``, is one token under the Llama-3 tokenizer (checked
    with the tokenizers of ``meta-llama/Llama-3.1-8B`` and
    ``meta-llama/Llama-3.2-1B-Instruct``, which have its 128,256-row
    vocabulary)."""
    return [" " + word, " " + word.capitalize()]


def interchange(
    base: CausalTrace, cf: CausalTrace, moved: list[str], held: list[str] = ()
) -> str:
    """The original's answer with ``moved`` set to the counterfactual's values
    and ``held`` pinned to the original's own: the hard interventions of
    Appendix F, on the causal model."""
    trace = base.copy()
    for name in held:
        trace[name] = base[name]
    for name in moved:
        trace[name] = cf[name]
    return trace["answer_payload"]


SOURCE = ["char_oi_a", "obj_oi_a", "char_oi_b", "obj_oi_b"]
STATE_TOKENS = ["address_a", "address_b", "state_oi_a", "state_oi_b"]


def _distinct_prompts(examples: list[dict[str, Any]]) -> None:
    prompts = [
        t["raw_input"]
        for e in examples
        for t in (e["input"], e["counterfactual_inputs"][0])
    ]
    assert len(set(prompts)) == len(prompts), (
        "a prompt repeats; splits must be endpoint-disjoint"
    )


def build(restate: bool) -> Any:
    """The table of one design: its validation candidates ``VALID``, in two
    splits, with the labels each figure scores."""
    model = belief_model()
    candidates = authors_candidates(restate=restate)[VALID.start : VALID.stop]
    examples = [pair(model, candidate) for candidate in candidates]
    _distinct_prompts(examples)
    dataset = serialize_examples(
        model,
        examples,
        split=[SPLITS[0] if i < N // 2 else SPLITS[1] for i in range(N)],
        target_variables=["answer_payload"] if restate else STATE_TOKENS,
        task_label=TASK,
        generator=GENERATOR,
        n=N,
        seed=SEED,
    )
    for row, example, candidate in zip(dataset.rows, examples, candidates):
        base, cf = example["input"], example["counterfactual_inputs"][0]
        other = base["state_b"] if base["q"] == "a" else base["state_a"]
        assert " " + other == authors_target(candidate)
        if restate:
            pointer = interchange(base, cf, ["answer_pointer"])
            source = interchange(base, cf, SOURCE, held=STATE_TOKENS)
            assert pointer == source == other, (pointer, source, other)
            assert row["label"] == " " + cf["answer_payload"] != row["base_answer"]
            row["label_pointer"] = " " + pointer
            row["label_pointer_forms"] = forms(pointer)
            row["label_source"] = " " + source
            row["label_source_forms"] = forms(source)
        else:
            assert row["label"] == " " + other != row["base_answer"]
            assert row["cf_answer"] == row["base_answer"]
    return dataset


def build_screen(restate: bool) -> Any:
    """The screening table of one design: its candidates ``SCREEN`` in the
    split ``screen``. Only the baseline document reads it, so its rows carry
    no figure labels beyond ``label``, the counterfactual's own answer."""
    model = belief_model()
    candidates = authors_candidates(restate=restate)[SCREEN.start : SCREEN.stop]
    examples = [pair(model, candidate) for candidate in candidates]
    _distinct_prompts(examples)
    return serialize_examples(
        model,
        examples,
        split=SCREEN_SPLIT,
        target_variables=["answer_payload"],
        task_label=TASK,
        generator=GENERATOR,
        n=len(examples),
        seed=SEED,
    )


#: Each committed table and the function that builds it.
TABLES = {
    "data_reorder.json": lambda: build(False),
    "data_restate.json": lambda: build(True),
    "data_reorder_screen.json": lambda: build_screen(False),
    "data_restate_screen.json": lambda: build_screen(True),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="the directory the tables are written to",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="rebuild in memory and exit 1 if the committed bytes differ",
    )
    args = parser.parse_args(argv)
    out = args.out if args.out.is_dir() else args.out.parent
    tables = TABLES if args.out.is_dir() else {args.out.name: TABLES[args.out.name]}
    for name, make in tables.items():
        dataset = make()
        path = out / name
        if args.check:
            fresh = table_bytes(dataset.rows)
            if not path.is_file() or path.read_bytes() != fresh:
                print(f"{path}: bytes differ from a fresh build", file=sys.stderr)
                return 1
            print(f"{path}: reproduces ({hashlib.sha256(fresh).hexdigest()[:12]})")
            continue
        digest = write_dataset_table(dataset.rows, path)
        print(f"{path}: {len(dataset.rows)} rows, digest {digest[:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
