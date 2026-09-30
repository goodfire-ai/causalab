"""The rebuild of the authors' 100-pair draw for IOI Figure 3b.

``demos/papers/workflows/scripts/ioi_fig3b/authors_draw.py`` runs the
authors' own ``ioi_dataset.py`` from a clone of Easy-Transformer, which the
repository does not hold. These checks cover the part that is this
package's: turning one of the authors' prompt records and its pABC flip into
a pair of the package's causal model, and refusing a file with other bytes.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

SCRIPTS = (
    Path(__file__).resolve().parents[2] / "demos/papers/workflows/scripts/ioi_fig3b"
)


def _load(name: str):
    spec = importlib.util.spec_from_file_location(
        f"ioi_fig3b_{name}", SCRIPTS / f"{name}.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _record(text: str, io: str, s: str) -> dict:
    """A prompt record as the authors' ``gen_prompt_uniform`` and
    ``gen_flipped_prompts`` write it: the text ends with the answer."""
    return {"text": text, "IO": io, "S": s, "[PLACE]": "store", "[OBJECT]": "drink"}


@pytest.fixture(scope="module")
def parts():
    builder = _load("build_dataset")
    return _load("authors_draw"), builder, builder.ioi_model()


def test_a_baba_pair_keeps_each_slots_name(parts) -> None:
    draw, builder, model = parts
    base = _record(
        "Then, John and Mary went to the store. John gave a drink to Mary",
        "Mary",
        "John",
    )
    abc = _record(
        "Then, Kevin and Laura went to the store. Adam gave a drink to Laura",
        "Laura",
        "Adam",
    )
    example = draw.pair(model, builder.TEMPLATES, base, abc)
    clean, (counterfactual,) = example["input"], example["counterfactual_inputs"]
    assert (
        clean["raw_input"]
        == "Then, John and Mary went to the store. John gave a drink to"
    )
    assert (clean["io"], clean["s"], clean["third"]) == ("Mary", "John", "John")
    assert clean["s_answer"] == " John" and clean["raw_output"] == " Mary"
    assert (counterfactual["io"], counterfactual["s"], counterfactual["third"]) == (
        "Laura",
        "Kevin",
        "Adam",
    )
    assert counterfactual["raw_input"] == abc["text"].rsplit(" ", 1)[0]
    assert counterfactual["template"] == clean["template"]


def test_an_abba_flip_that_repeats_a_name_is_taken_as_it_is(parts) -> None:
    """The authors' flips draw each new name without looking at the others,
    so a pABC prompt can hold one name twice."""
    draw, builder, model = parts
    base = _record(
        "When Mary and John got a drink at the store, John decided to give it to Mary",
        "Mary",
        "John",
    )
    abc = _record(
        "When Kevin and Kevin got a drink at the store, Adam decided to give it to Kevin",
        "Kevin",
        "Kevin",
    )
    counterfactual = draw.pair(model, builder.TEMPLATES, base, abc)[
        "counterfactual_inputs"
    ][0]
    assert (counterfactual["io"], counterfactual["s"], counterfactual["third"]) == (
        "Kevin",
        "Kevin",
        "Adam",
    )


def test_a_prompt_that_does_not_end_with_its_answer_is_refused(parts) -> None:
    draw, builder, model = parts
    base = _record(
        "Then, John and Mary went to the store. John gave a drink to John",
        "Mary",
        "John",
    )
    with pytest.raises(ValueError, match="does not end with its IO"):
        draw.pair(model, builder.TEMPLATES, base, base)


def test_other_bytes_of_the_authors_file_are_refused(parts, tmp_path: Path) -> None:
    draw, _, _ = parts
    path = tmp_path / "easy_transformer" / "ioi_dataset.py"
    path.parent.mkdir()
    path.write_text("NAMES = []\n")
    with pytest.raises(ValueError, match="not 94a72de9"):
        draw.authors_prompts(tmp_path)
