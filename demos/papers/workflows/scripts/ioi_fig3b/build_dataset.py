"""Build the table of the Figure 3b replication: pIOI prompts, each with its pABC counterfactual.

Wang et al. 2022 (arXiv:2211.00593) run path patching on pairs (x_orig, x_new)
with x_orig drawn from **pIOI** -- an indirect-object-identification sentence
such as ``When Mary and John went to the store, John gave a drink to`` whose
next token is the indirect object ``Mary`` -- and x_new the corresponding
sample from **pABC**, "where the names in the sentence are replaced by three
random names" (Section 3.1). One table follows:

* ``data.json`` -- N rows; ``input`` is the pIOI prompt, ``counterfactual_inputs[0]``
  the pABC prompt built on the same template, place and object with three
  fresh names in the three name slots. The path-patching document reads the
  sender head's value from the counterfactual side of each row and scores the
  logit difference IO - S on the base side.

The prompts follow the authors' code, which drew Figure 3b and Figure 15
(``easy_transformer/ioi_dataset.py`` of ``redwoodresearch/Easy-Transformer``,
https://github.com/redwoodresearch/Easy-Transformer, used under its MIT
License, Copyright (c) 2022 neelnanda-io,
https://github.com/redwoodresearch/Easy-Transformer/blob/main/LICENSE),
where the code and the paper's text differ. The code samples 14 templates:
the first seven of Figure 14 in the ``BABA`` order and the same seven in the
``ABBA`` order. Appendix E lists all 15 templates, 30 in both orders. A name
fills each of the three name slots from the code's 99 single-token first
names (the paper says 100), and a place and an object come from the code's
two eight-word lists (the paper says a hand-made list of 20 words). Every
name, place and object is one GPT-2 token, so the last token of every prompt
is the paper's END position. The sentence ends at the word before the
indirect object (the ``to``), which is where the paper reads the logits.

The causal model has six input variables -- ``template``, ``io``, ``s``,
``third`` (the name in the third slot: ``s`` again on pIOI, a fresh name on
pABC), ``place``, ``object`` -- and three derived ones: the prompt, the
answer ``" " + io`` and ``s_answer``, the subject as the model would write it
next, ``" " + s``. Every row carries the causalab column vocabulary
(``causalab.tasks.serialize``), so ``causalab validate --data`` reads the
table like any shipped one. The ``logit_diff`` metric reads ``base_answer``
(``" " + io``) and ``s_answer``. A metric column is tokenized as written, and
the bare ``io`` and ``s`` names are not always one GPT-2 token (``Travis``
is ``T`` + ``ravis``); with the leading space every name is one token. The
``label`` column is the answer after interchanging the names from the
counterfactual (the pABC prompt's own "indirect object"), and nothing here
reads it.

Usage::

    python workflows/scripts/ioi_fig3b/build_dataset.py --out artifacts/data/ioi_fig3b            # writes data.json
    python workflows/scripts/ioi_fig3b/build_dataset.py --out artifacts/data/ioi_fig3b --check    # exit 1 if the committed bytes differ
"""

from __future__ import annotations

import argparse
import hashlib
import random
import sys
from pathlib import Path

from causalab.causal import CausalModel, CausalTrace, Dom, V, mechanism
from causalab.causal.scoring import ScoringSpec
from causalab.tasks.serialize import (
    serialize_examples,
    table_bytes,
    write_dataset_table,
)

TASK = "ioi_fig3b"
GENERATOR = "build_dataset.py"
PAPER = "arXiv:2211.00593, Figure 3b; Section 3.1, Appendix B and Appendix E"

#: Rows in the table, and the seed the sampling is pinned by. The paper
#: averages its path-patching effects over "N > 200 pairs" (Appendix B). The
#: authors' code draws N = 100 (``experiments.py`` lines 78-84 at
#: Easy-Transformer ``ea15315``) right after loading the model, which seeds
#: Python's ``random`` with 42 (``EasyTransformerConfig.seed``). A fresh
#: process therefore draws one fixed set of 100 pairs, and
#: ``authors_draw.py`` rebuilds it; that set does not give the paper's
#: figures. 5000 pairs put the paired bootstrap SE of every cell at or below
#: 0.005, the reading error of the Figure 3b raster (0.0049 at head 9.9 is
#: the largest).
N = 5000
SEED = 0

#: Figure 14's fifteen templates in the paper's ``BABA`` order (``[B]`` is the
#: subject S, ``[A]`` the indirect object IO); the final ``[A]`` is the answer
#: and is not part of the prompt. The ``ABBA`` order swaps the first ``[B]``
#: and ``[A]``.
BABA_TEMPLATES = [
    "Then, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Then, [B] and [A] had a lot of fun at the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Then, [B] and [A] were working at the [PLACE]. [B] decided to give a [OBJECT] to [A]",
    "Then, [B] and [A] were thinking about going to the [PLACE]. [B] wanted to give a [OBJECT] to [A]",
    "Then, [B] and [A] had a long argument, and afterwards [B] said to [A]",
    "After [B] and [A] went to the [PLACE], [B] gave a [OBJECT] to [A]",
    "When [B] and [A] got a [OBJECT] at the [PLACE], [B] decided to give it to [A]",
    "When [B] and [A] got a [OBJECT] at the [PLACE], [B] decided to give the [OBJECT] to [A]",
    "While [B] and [A] were working at the [PLACE], [B] gave a [OBJECT] to [A]",
    "While [B] and [A] were commuting to the [PLACE], [B] gave a [OBJECT] to [A]",
    "After the lunch, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Afterwards, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Then, [B] and [A] had a long argument. Afterwards [B] said to [A]",
    "The [PLACE] [B] and [A] went to had a [OBJECT]. [B] gave it to [A]",
    "Friends [B] and [A] found a [OBJECT] at the [PLACE]. [B] gave it to [A]",
]

#: The 99 first names of the authors' code, each one GPT-2 token with a
#: leading space. Appendix E says 100 names, and no public revision of the
#: code has another list.
NAMES = [
    "Michael", "Christopher", "Jessica", "Matthew", "Ashley", "Jennifer",
    "Joshua", "Amanda", "Daniel", "David", "James", "Robert", "John", "Joseph",
    "Andrew", "Ryan", "Brandon", "Jason", "Justin", "Sarah", "William",
    "Jonathan", "Stephanie", "Brian", "Nicole", "Nicholas", "Anthony",
    "Heather", "Eric", "Elizabeth", "Adam", "Megan", "Melissa", "Kevin",
    "Steven", "Thomas", "Timothy", "Christina", "Kyle", "Rachel", "Laura",
    "Lauren", "Amber", "Brittany", "Danielle", "Richard", "Kimberly", "Jeffrey",
    "Amy", "Crystal", "Michelle", "Tiffany", "Jeremy", "Benjamin", "Mark",
    "Emily", "Aaron", "Charles", "Rebecca", "Jacob", "Stephen", "Patrick",
    "Sean", "Erin", "Jamie", "Kelly", "Samantha", "Nathan", "Sara", "Dustin",
    "Paul", "Angela", "Tyler", "Scott", "Katherine", "Andrea", "Gregory",
    "Erica", "Mary", "Travis", "Lisa", "Kenneth", "Bryan", "Lindsey", "Kristen",
    "Jose", "Alexander", "Jesse", "Katie", "Lindsay", "Shannon", "Vanessa",
    "Courtney", "Christine", "Alicia", "Cody", "Allison", "Bradley", "Samuel",
]  # fmt: skip

#: The places and objects of the authors' code, eight each. Appendix E says
#: a hand-made list of 20 place and object words, which is not published.
PLACES = [
    "store",
    "garden",
    "restaurant",
    "school",
    "hospital",
    "office",
    "house",
    "station",
]
OBJECTS = [
    "ring",
    "kiss",
    "bone",
    "basketball",
    "computer",
    "necklace",
    "drink",
    "snack",
]


def _slots(baba: str, order: str) -> str:
    """A Figure 14 template as a prompt with three named slots.

    ``BABA``: the slots read S, IO, S; ``ABBA``: IO, S, S. The third slot is
    ``third`` rather than ``s`` so a pABC prompt can put a fresh name there;
    the answer slot (the final ``[A]``) is dropped -- the prompt ends where
    the paper reads the logits."""
    assert baba.endswith(" [A]")
    body = baba[: -len(" [A]")]
    first, second = ("{s}", "{io}") if order == "BABA" else ("{io}", "{s}")
    body = body.replace("[B]", first, 1).replace("[A]", second, 1)
    body = body.replace("[B]", "{third}")
    return body.replace("[PLACE]", "{place}").replace("[OBJECT]", "{object}")


#: The templates of each name order that the authors' code samples.
PER_ORDER = len(BABA_TEMPLATES) // 2

#: The templates the authors' code samples, 14 in all: IOIDataset with
#: ``prompt_type="mixed"`` takes ``BABA_TEMPLATES[: nb_templates // 2] +
#: ABBA_TEMPLATES[: nb_templates // 2]`` with ``nb_templates`` 15 by default
#: (https://github.com/redwoodresearch/Easy-Transformer/blob/ea15315dd24481e9e2ac5c3ef335d82907a1dc34/easy_transformer/ioi_dataset.py#L708-L720).
#: ``ioi_dataset.py`` at ``373cd15`` of 2022-10-28, the last commit to change
#: the file before arXiv v1, has the same bytes. The path-patching notebook
#: of that commit and the public ``experiments.py`` draw their pairs with
#: this default. We follow the code over the text of Appendix E, which lists
#: all 15 templates in both orders.
TEMPLATES = [
    _slots(t, order) for order in ("BABA", "ABBA") for t in BABA_TEMPLATES[:PER_ORDER]
]


def ioi_model() -> CausalModel:
    """template + io + s + third + place + object -> prompt; io -> answer;
    s -> s_answer."""

    @mechanism
    def equations(
        template: Dom(TEMPLATES),
        io: Dom(NAMES),
        s: Dom(NAMES),
        third: Dom(NAMES),
        place: Dom(PLACES),
        object: Dom(OBJECTS),
    ):
        raw_input = V(  # noqa: F841
            template.format(io=io, s=s, third=third, place=place, object=object),
            domain=Dom(str),
        )
        raw_output = V(" " + io, domain=Dom(str))
        # the logit_diff metric's `b` column: a metric column is tokenized as
        # written, and only the space-led name is one GPT-2 token for all 99
        s_answer = V(" " + s, domain=Dom(str))  # noqa: F841
        return raw_output

    scoring = ScoringSpec(
        forms={"io": {name: (" " + name, name) for name in NAMES}}, string_mode="exact"
    )
    return CausalModel(equations, id=TASK, scoring=scoring)


def sample_pair(
    rng: random.Random, model: CausalModel
) -> dict[str, list[CausalTrace] | CausalTrace]:
    """One pIOI prompt and its pABC counterfactual: the same template, place
    and object; two distinct names on the IOI side, and three fresh names --
    distinct from each other and from the IOI pair -- on the ABC side. (The
    authors' code draws the ABC names one flip at a time and lets them
    collide occasionally; here they never do.)"""
    template = rng.choice(TEMPLATES)
    place = rng.choice(PLACES)
    obj = rng.choice(OBJECTS)
    io, s, a, b, c = rng.sample(NAMES, 5)
    common = {"template": template, "place": place, "object": obj}
    base = model.new_trace({**common, "io": io, "s": s, "third": s})
    abc = model.new_trace({**common, "io": a, "s": b, "third": c})
    return {"input": base, "counterfactual_inputs": [abc]}


def serialize(
    model: CausalModel,
    examples: list[dict[str, list[CausalTrace] | CausalTrace]],
    generator: str,
    seed: int,
):
    """The package's table of ``examples``, in the causalab column vocabulary:
    one undivided split, the names interchanged for the ``label`` column."""
    return serialize_examples(
        model,
        examples,
        split="all",
        target_variables=["io", "s", "third"],
        task_label=TASK,
        generator=generator,
        n=len(examples),
        seed=seed,
    )


def build():
    model = ioi_model()
    rng = random.Random(SEED)
    examples = [sample_pair(rng, model) for _ in range(N)]
    return serialize(model, examples, GENERATOR, SEED)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--out", type=Path, required=True, help="the directory data.json is written to"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="rebuild in memory and exit 1 if the committed table's bytes differ",
    )
    args = parser.parse_args(argv)
    out = args.out if args.out.is_dir() else args.out.parent
    dataset = build()
    path = out / "data.json"
    if args.check:
        fresh = table_bytes(dataset.rows)
        if not path.is_file() or path.read_bytes() != fresh:
            print(f"{path}: bytes differ from a fresh build", file=sys.stderr)
            return 1
        print(f"{path}: reproduces ({hashlib.sha256(fresh).hexdigest()[:12]})")
        return 0
    digest = write_dataset_table(dataset.rows, path)
    print(f"{path}: {len(dataset.rows)} rows, digest {digest[:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
