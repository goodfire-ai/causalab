"""``pipeline.resolve_answers``: every metric's answers resolved with the
model's tokenizer at the run door, before the weights load.

The pass resolves each aggregation through the score path's own function
(``answers.metric_token_ids``) over the rows the aggregation scores: the rows
a ``save`` entry's read aligns on, every base row for an objective term, the
held-out split's rows for an eval entry. A value the tokenizer cannot score
is refused ``[P2]``, naming where the aggregation lives, the table and the
tokenizer; every failing value is reported at once, one line per failing
field. A bare answer that is one token, glued to text that ends in a letter
or digit, is legal but likely names a token the model does not emit there,
so it is a ``ProtocolWarning``.

On the base ``resolve_answers`` does not exist, and the file fails to
import.
"""

from __future__ import annotations

import dataclasses
import json
import shutil
import warnings
from pathlib import Path
from typing import Any

import pytest

from causalab.io.env import FileDatasets, ResolutionEnv
from causalab.protocol.pipeline import compile_protocol, resolve_answers
from causalab.protocol.rules.errors import ProtocolError, ProtocolWarning

from tests.protocol._docs import aggregation, in_order, saved
from tests.protocol._env import CORPUS_MODEL, FIXTURES
from tests.protocol.test_legality_before_weights import eval_updates_doc

pytestmark = pytest.mark.unit

#: Two IOI prompts; the model continues each with a space-prefixed name.
PROMPTS = (
    "Then, Jennifer and Kevin went to the store. Kevin gave a drink to",
    "Then, Tiffany and Sean went to the store. Sean gave a drink to",
)

LOADS: list[tuple[str, str]] = []


def _names_doc(**fields: Any) -> dict[str, Any]:
    """The un-intervened gpt2 on ``names/data``, its last-position logits
    reduced to one aggregation (``logit_diff`` of ``io`` and ``s`` unless
    ``fields`` says otherwise)."""
    metric = fields or {"kind": "logit_diff", "a": "io", "b": "s"}
    return {
        "header": {"protocol_version": "4"},
        "model": {"key": "gpt2", "revision": "main"},
        "data": {"base": {"dataset": "names/data", "field": "input"}},
        "method": {
            "intervened_models": {
                "original_base": {"input": "base", "reads": ["logits"]}
            },
            "sites": {"lm_head": {"component": "lm_head"}},
            "reads": {"logits": {"site": "lm_head", "pos": -1}},
            "save": [saved("logits", "original_base", "m.json", dict(metric))],
        },
    }


def _env(
    env: ResolutionEnv,
    root: Path,
    prompts: tuple[str, ...] = PROMPTS,
    **columns: tuple[Any, ...],
) -> ResolutionEnv:
    """``env`` over ``root``, where ``names/data`` holds ``prompts`` and one
    column per keyword (a value per row), with every tokenizer load recorded
    in `LOADS`."""
    rows = [
        {"input": prompt, "split": "all", **{k: v[i] for k, v in columns.items()}}
        for i, prompt in enumerate(prompts)
    ]
    (root / "names").mkdir(parents=True, exist_ok=True)
    (root / "names" / "data.json").write_text(json.dumps(rows))
    loader = env.tokenizers
    assert loader is not None
    LOADS.clear()

    def tokenizers(key: str, revision: str) -> Any:
        LOADS.append((key, revision))
        return loader(key, revision)

    return dataclasses.replace(
        env, datasets=FileDatasets(root=root), tokenizers=tokenizers
    )


def _resolve(raw: dict[str, Any], env: ResolutionEnv) -> None:
    resolve_answers(compile_protocol(in_order(raw), env=env, data=True), env=env)


def test_a_split_answer_is_refused_naming_the_aggregation_table_and_tokenizer(
    env: ResolutionEnv, tmp_path: Path
) -> None:
    names = _env(env, tmp_path, io=(" Jennifer", "Tiffany"), s=(" Kevin", " Sean"))
    with pytest.raises(ProtocolError) as err:
        _resolve(_names_doc(), names)
    text = str(err.value)
    assert err.value.code == "P2" and err.value.path == "save[0].aggregation", text
    assert "'Tiffany'" in text and "metric logit_diff.a" in text, text
    assert "'names/data'" in text and "gpt2@main" in text, text


def test_every_distinct_refusal_is_reported_at_once(
    env: ResolutionEnv, tmp_path: Path
) -> None:
    """Two aggregations over columns the tokenizer splits: one refusal has a
    line per failing field of each (``logit_diff`` fails on ``a`` and ``b``,
    ``token_logit`` on ``token``), so one fix-and-rerun loop clears the
    table."""
    names = _env(env, tmp_path, io=(" Jennifer", "Tiffany"), s=("Seanathan", " Sean"))
    raw = _names_doc()
    raw["method"]["save"].append(
        saved(
            "logits", "original_base", "s.json", aggregation("token_logit", token="s")
        )
    )
    with pytest.raises(ProtocolError) as err:
        _resolve(raw, names)
    lines = str(err.value).splitlines()
    assert len(lines) == 3, lines
    assert "save[0].aggregation" in lines[0] and "'Tiffany' (row 1)" in lines[0]
    assert "save[0].aggregation" in lines[1] and "'Seanathan' (row 0)" in lines[1]
    assert "save[1].aggregation" in lines[2] and "'Seanathan' (row 0)" in lines[2]


def test_every_split_value_of_a_column_is_counted_in_one_line(
    env: ResolutionEnv, tmp_path: Path
) -> None:
    """Three of four ``io`` values split under gpt2, and one ``s`` value
    does. The ``io`` line counts the three, names the first row and quotes
    the others; the ``s`` line is its own."""
    names = _env(
        env,
        tmp_path,
        PROMPTS * 2,
        io=(" Tiffanyx", " Seanathan", " Jennifer", " Brunhilde"),
        s=(" Kevin", " Sean", "Kevinoss", " Sean"),
    )
    with pytest.raises(ProtocolError) as err:
        _resolve(_names_doc(), names)
    io, s = str(err.value).splitlines()
    assert "metric logit_diff.a: metric column value ' Tiffanyx' (row 0)" in io, io
    assert "3 of 4 values fail this way" in io, io
    assert "' Seanathan'" in io and "' Brunhilde'" in io, io
    assert "metric logit_diff.b: metric column value 'Kevinoss' (row 2)" in s, s


#: a read at the ``entity`` column's value in the prompt (§2.3 ``variable``)
ANCHORED = {"variable": "entity"}


@pytest.mark.parametrize("aligned", [False, True])
def test_a_row_its_read_does_not_align_on_is_not_resolved(
    aligned: bool, env: ResolutionEnv, tmp_path: Path
) -> None:
    """A saved metric scores only the rows its read aligns on (§4.1), so the
    split ``'Tiffany'`` on row 1 is left alone while it occurs nowhere in the
    prompt, and refused once it does."""
    prompts = (
        PROMPTS[0],
        PROMPTS[1].replace("Tiffany", "Tiffany" if aligned else "Mary"),
    )
    names = _env(env, tmp_path, prompts, entity=(" store", "Tiffany"))
    raw = _names_doc(kind="token_logit", token="entity")
    raw["method"]["positions"] = {"ent": ANCHORED}
    raw["method"]["reads"]["logits"]["pos"] = "ent"
    if not aligned:
        _resolve(raw, names)
        return
    with pytest.raises(ProtocolError, match=r"'Tiffany' \(row 1\) is not a single"):
        _resolve(raw, names)


def test_a_continuation_read_is_left_to_the_score(
    env: ResolutionEnv, tmp_path: Path
) -> None:
    """Which rows a read over generated steps scores is known only after the
    decode, so the pass resolves nothing for it: the score refuses a split
    answer on a row that generated a step, as it did before the pass."""
    names = _env(env, tmp_path, io=("Tiffany", "Tiffany"))
    raw = _names_doc(kind="token_logit", token="io")
    raw["method"]["positions"] = {
        "window": {"generated": {"max_new_tokens": 2}, "all": True}
    }
    raw["method"]["reads"]["logits"]["pos"] = "window"
    _resolve(raw, names)


def test_the_spaced_answers_resolve_with_one_tokenizer_load(
    env: ResolutionEnv, tmp_path: Path
) -> None:
    """A three-layer sweep is three representatives over one table: the
    answers resolve once, with one tokenizer load, and nothing warns."""
    names = _env(env, tmp_path, io=(" Jennifer", " Tiffany"), s=(" Kevin", " Sean"))
    raw = _names_doc()
    raw["method"]["sites"]["resid"] = {
        "component": "block_output",
        "layers": {"sweep": [1, 2, 3]},
    }
    raw["method"]["reads"]["h"] = {"site": "resid", "pos": -1}
    raw["method"]["intervened_models"]["original_base"]["reads"].append("h")
    raw["method"]["save"].append(saved("h", "original_base", "h.safetensors"))
    with warnings.catch_warnings():
        warnings.simplefilter("error", ProtocolWarning)
        _resolve(raw, names)
    assert LOADS == [("gpt2", "main")]


def test_a_band_sweep_reports_each_refusal_once(
    env: ResolutionEnv, tmp_path: Path
) -> None:
    """A sweep over a layer band renames the lowered reads at every point,
    and the answers' frame and address stay the same: the split
    ``'Tiffany'`` is one refusal line, not one per point."""
    names = _env(env, tmp_path, io=(" Jennifer", "Tiffany"), s=(" Kevin", " Sean"))
    raw = _names_doc()
    raw["axes"] = {"upto": {"rows": [{"layers": [0]}, {"layers": [0, 1]}]}}
    raw["method"]["sites"]["resid"] = {
        "component": "block_output",
        "layers": {"axis": "upto.layers"},
    }
    raw["method"]["reads"]["h"] = {"site": "resid", "pos": -1}
    raw["method"]["intervened_models"]["original_base"]["reads"].append("h")
    raw["method"]["save"].append(saved("h", "original_base", "h.safetensors"))
    with pytest.raises(ProtocolError) as err:
        _resolve(raw, names)
    assert len(str(err.value).splitlines()) == 1, str(err.value)


def test_a_metric_with_no_answers_loads_no_tokenizer(
    env: ResolutionEnv, tmp_path: Path
) -> None:
    names = _env(env, tmp_path)
    raw = _names_doc(kind="top_k", k=3, by="value")
    _resolve(raw, names)
    assert LOADS == []


def test_a_glued_answer_warns(env: ResolutionEnv, tmp_path: Path) -> None:
    """``'Jennifer'`` is one gpt2 token (43187), and so is ``' Jennifer'``
    (16348), which is what the model emits after ``... gave a drink to``.
    The metric would score the first: legal, so a warning, naming the column,
    the count and an example."""
    names = _env(env, tmp_path, io=("Jennifer", " Tiffany"), s=(" Kevin", " Sean"))
    with pytest.warns(ProtocolWarning) as caught:
        _resolve(_names_doc(), names)
    (warning,) = [w for w in caught if "Jennifer" in str(w.message)]
    text = str(warning.message)
    assert "logit_diff.a" in text and "'io'" in text and "1 of 2" in text, text


@pytest.mark.parametrize("frame", ["chat", None])
def test_a_bare_answer_after_the_chat_header_is_quiet(
    frame: str | None, env: ResolutionEnv, tmp_path: Path
) -> None:
    """Under ``segments.frame: chat`` the last prompt token is the assistant
    header (``...<|im_start|>assistant\\n``), where the model begins its
    turn with the bare name, so the lint tests the rendered text and stays
    quiet. The same table in the plain frame is glued and warns."""
    names = _env(env, tmp_path, io=("John", "Mary"), s=("Tom", "Paul"))
    raw = _names_doc()
    raw["model"] = {"key": CORPUS_MODEL, "revision": "main"}
    if frame is None:
        with pytest.warns(ProtocolWarning, match="glue a bare answer"):
            _resolve(raw, names)
        return
    raw["method"]["segments"] = {"frame": frame}
    with warnings.catch_warnings():
        warnings.simplefilter("error", ProtocolWarning)
        _resolve(raw, names)


@pytest.mark.parametrize(
    "case",
    ["spaced", "forms_list", "not_at_the_last_token", "id_column"],
)
def test_the_glued_answer_lint_stays_quiet(
    case: str, env: ResolutionEnv, tmp_path: Path
) -> None:
    """No warning for a spaced answer; for a ``match`` forms list, which may
    credit the bare spelling on purpose; for a read that is not at the last
    prompt token, whose next token is not the answer's; and for an id
    column, which spells no string."""
    if case == "spaced":
        names = _env(env, tmp_path, io=(" Jennifer", " Tiffany"), s=(" Kevin", " Sean"))
        raw = _names_doc()
    elif case == "forms_list":
        names = _env(env, tmp_path, io=(["Jennifer", " Jennifer"], [" Tiffany"]))
        raw = _names_doc(kind="match", expected="io")
    elif case == "not_at_the_last_token":
        names = _env(env, tmp_path, io=("Jennifer", " Tiffany"), s=(" Kevin", " Sean"))
        raw = _names_doc()
        raw["method"]["reads"]["logits"]["pos"] = -2
    else:
        names = _env(env, tmp_path, io=(43187, 16348), s=(7735, 11465))
        raw = _names_doc(kind="logit_diff", a="io", b="s", token_form="id")
    with warnings.catch_warnings():
        warnings.simplefilter("error", ProtocolWarning)
        _resolve(raw, names)


def test_an_id_outside_the_vocabulary_is_refused(
    env: ResolutionEnv, tmp_path: Path
) -> None:
    names = _env(env, tmp_path, io=(43187, 10**6), s=(7735, 11465))
    raw = _names_doc(kind="logit_diff", a="io", b="s", token_form="id")
    with pytest.raises(ProtocolError, match="integer token ID in the vocabulary"):
        _resolve(raw, names)


def test_a_literal_answer_is_resolved_too(env: ResolutionEnv, tmp_path: Path) -> None:
    """``class_probs.groups`` names literal strings, not columns; a member
    the tokenizer splits is refused like a column value."""
    names = _env(env, tmp_path)
    raw = _names_doc(kind="class_probs", groups={"io": [" Jennifer", "Tiffany"]})
    with pytest.raises(ProtocolError) as err:
        _resolve(raw, names)
    assert "metric class_probs.groups.io" in str(err.value), str(err.value)
    assert "'Tiffany'" in str(err.value), str(err.value)


def test_an_eval_aggregation_resolves_over_its_held_out_split(
    env: ResolutionEnv, tmp_path: Path
) -> None:
    """A fit's eval entry scores the held-out rows, not the training rows:
    a split label in the ``test`` rows only is refused under the eval
    entry's path and the split's ref."""
    root = tmp_path / "data"
    shutil.copytree(FIXTURES / "data" / "weekdays", root / "weekdays")
    table = root / "weekdays" / "data.json"
    rows = json.loads(table.read_text())
    for row in rows:
        if row["split"] == "test":
            row["label"] = " Mon day"
    table.write_text(json.dumps(rows))
    split = dataclasses.replace(env, datasets=FileDatasets(root=root))
    raw = eval_updates_doc()  # its eval entry `ce` scores weekdays/data#test
    with pytest.raises(ProtocolError) as err:
        _resolve(raw, split)
    text = str(err.value)
    assert err.value.path == "train.eval.aggregations.ce.aggregation", text
    assert "'weekdays/data#test'" in text and "' Mon day'" in text, text
