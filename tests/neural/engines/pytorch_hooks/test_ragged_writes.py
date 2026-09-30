"""Ragged and variable-length writes (spec §2.8 ``ragged``, §5 rule 19)
— the engine half, on tiny Llama, CPU.

A write whose rows address different numbers of positions — an ``all`` window
over prompts of unequal length, a ``variable`` window over an entity that
tokenizes to different lengths (the weekdays shape: ``Thursday`` is three
pieces here, ``Friday`` one) — used to be rule 19's refusal and nothing else.
It now dispatches on the write's declared ``ragged`` policy, still **before
any forward pass**: ``refuse`` (and an absent field) is the refusal as before,
``exact_length_buckets`` lands the rows grouped by width, ``padded_masked``
lands them through one padded gather and a mask. Both landings happen inside
the forward the window already runs — nothing about batch geometry changes.

* **T10** — two prompts of different lengths succeed under both policies for
  an all-position operation (a zero ablation; a ``swap`` from a ragged read),
  and for a ``variable`` window whose per-row widths differ across rows; the
  written values are **bit-identical** between the two policies and equal a
  per-row reference computed by running each row alone (the padding is never
  read as activation), whole-batch and under ``batch_rows=1``. The recorded
  geometry re-nests the written window by its stored widths. *Mutation A* —
  the masked landing writes every padded slot (mask dropped) — breaks the
  equality on the ``variable`` case, where the pad slot is a real token
  outside the window. *Mutation B* — ``widths`` dropped from the recorded
  geometry — breaks the reader that re-nests by them.
* **T11** — ``policy: refuse`` and the absent field keep rule 19's pre-forward
  guarantee: refused by rule number, at ``writes.<w>.pos``, naming the row
  widths, with reason ``ragged_write_unsupported`` and the message strings the
  earlier test pinned, and no forward runs; through ``run_protocol`` with a
  caller-owned bundle and ``load_model`` replaced by one that raises, nothing
  is loaded and nothing runs. A ragged operand whose row widths disagree with
  the write's is the same refusal under a landing policy.
* **T12** — a fixture variant of the shipped ``weekdays_locate_scan`` with the
  bare ``{"variable": "entity"}`` tap (the spelling whose 32 points the
  shipped description says were unrunnable) runs every point under
  ``exact_length_buckets`` over a table whose entities tokenize to two widths,
  ``unavailable``-free, with the geometry in the receipt's ``execution.ragged``
  — and the same document with no policy is rule 19's refusal. The shipped
  document itself keeps its last-token spelling and authors no policy.

Every refusal has its valid-work twin beside it, so a check that refuses
everything cannot pass.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch

from causalab.neural.engines.pytorch_hooks.engine import PytorchHooksEngine
from causalab.neural.shared.encoding import encode
from causalab.neural.shared.executor import RaggedValue
from causalab.protocol import run_protocol
from causalab.protocol.rules.errors import ValidationError
from causalab.protocol.pipeline import compile_protocol
from causalab.protocol.results import Unavailable
from causalab.io.env import FileArtifacts, FileDatasets, ResolutionEnv
from causalab.protocol.receipt import RUN_RECORD_NAME

from ._drive import base_data_section, executor_for
from .conftest import TINY_LLAMA
from tests.protocol._docs import UNWRITTEN, in_order, saved
from tests.tables import frame as table_frame
from tests._helpers.paths import PROTOCOLS_DIR


pytestmark = pytest.mark.smoke

REPO = Path(__file__).resolve().parents[4]
SHIPPED = PROTOCOLS_DIR / "weekdays_locate_scan.json"
TWIN = REPO / "tests/protocols/07_weekdays_locate_scan_im.json"

LANDING = ("exact_length_buckets", "padded_masked")
#: `run.RAGGED_KEY`, spelled here so this module still collects on a tree
#: without the field and fails at the parse (P3, `ragged` unknown) — the
#: fails-without witness; `test_the_spellings_are_the_code_s` holds it.
RAGGED_KEY = "ragged"

#: The pair `test_write_oracle.py` already proves ragged on tiny Llama (two
#: and eleven content tokens), and counterfactuals of the same per-row widths —
#: a `swap` pairs its operand into the write row by row.
BASE_TEXTS = ["one two", "a much longer sentence right here indeed and then some more"]
CF_TEXTS = ["six ten", "the cat sat on the mat while the dog ran far"]

#: The weekdays shape: `Thursday` tokenizes to three pieces and `Friday` to
#: one, so a `variable` window over the entity is ragged across the rows,
#: while each row's counterfactual entity has the width of its own.
DAY_BASE = ["If today is Thursday, tomorrow is", "If today is Friday, tomorrow is"]
DAY_CF = ["If today is Wednesday, tomorrow is", "If today is Monday, tomorrow is"]
DAY_ENTITY = ["Thursday", "Friday"]
DAY_CF_ENTITY = ["Wednesday", "Monday"]


def _close(have: Any, want: Any) -> None:
    """The oracle tolerance of `test_write_oracle.py`: a row run alone sits
    in a different padded frame, so its activations differ by float noise."""
    torch.testing.assert_close(have, want, atol=1e-5, rtol=1e-4)


def _tensor(value: Any) -> torch.Tensor:
    assert isinstance(value, torch.Tensor)
    return value


def _ragged(value: Any) -> RaggedValue:
    assert isinstance(value, RaggedValue)
    return value


def _widths(bundle: Any, texts: list[str]) -> list[int]:
    """Content widths of ``texts`` in one left-padded frame — what an ``all``
    window addresses per row (BOS included, padding excluded)."""
    batch = encode(bundle.tokenizer, texts)
    return [batch.padded_len - batch.content_start(i) for i in range(len(texts))]


def _doc(
    do: dict[str, Any],
    *,
    pos: Any = "all",
    ragged: str | None,
    with_counterfactual: bool,
    positions: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """A write at ``pos`` under ``do``; reads of the patched logits and of the
    written window itself (``after``: what landed, ragged like the window)."""
    write: dict[str, Any] = {"site": "tgt", "pos": pos, "do": do}
    if ragged is not None:
        write["ragged"] = {"policy": ragged}
    method: dict[str, Any] = {
        "intervened_models": {
            "patched": {
                "input": "base",
                "reads": ["logits", "after"],
                "writes": ["patch"],
            }
        },
        "sites": {
            "tgt": {"component": "block_output", "layers": [0]},
            "lm_head": {"component": "lm_head"},
        },
        "reads": {
            "logits": {"site": "lm_head", "pos": {"index": -1}},
            "after": {"site": "tgt", "pos": pos},
        },
        "writes": {"patch": write},
        "save": [
            saved(name, "patched", f"{name}.safetensors")
            for name in ("logits", "after")
        ],
    }
    if positions is not None:
        method["positions"] = positions
    if with_counterfactual:
        method["reads"]["v_cf"] = {"site": "tgt", "pos": pos}
        method["intervened_models"][UNWRITTEN] = {
            "input": "counterfactual",
            "reads": ["v_cf"],
        }
    return {
        "header": {"protocol_version": "4"},
        "model": {"key": "test", "revision": "main"},
        "data": base_data_section(with_counterfactual=with_counterfactual),
        "method": method,
    }


def _zero_ablate(ragged: str | None) -> dict[str, Any]:
    return _doc({"swap": 0.0}, ragged=ragged, with_counterfactual=False)


def _swap_all(ragged: str | None) -> dict[str, Any]:
    return _doc({"swap": "v_cf"}, ragged=ragged, with_counterfactual=True)


def _swap_variable(ragged: str | None) -> dict[str, Any]:
    return _doc(
        {"swap": "v_cf"},
        pos="ent",
        ragged=ragged,
        with_counterfactual=True,
        positions={"ent": {"variable": "entity"}},
    )


@pytest.fixture(params=["zero_ablate_all", "swap_all", "swap_variable"])
def case(request: pytest.FixtureRequest) -> dict[str, Any]:
    """One ragged write, the texts that make it ragged, and the operand rows."""
    if request.param == "zero_ablate_all":
        return {"build": _zero_ablate, "base": BASE_TEXTS, "cf": None, "columns": {}}
    if request.param == "swap_all":
        return {"build": _swap_all, "base": BASE_TEXTS, "cf": CF_TEXTS, "columns": {}}
    return {
        "build": _swap_variable,
        "base": DAY_BASE,
        "cf": DAY_CF,
        "columns": {
            "entity": DAY_ENTITY,
            "counterfactual_inputs_variables": [{"entity": e} for e in DAY_CF_ENTITY],
        },
    }


def _pick(values: list[Any], rows: list[int]) -> list[Any]:
    return [values[i] for i in rows]


def _executor(
    case: dict[str, Any], bundle: Any, policy: str | None, rows: list[int], **kw: Any
):
    base: list[str] = _pick(case["base"], rows)
    cf: list[str] | None = case["cf"]
    columns: dict[str, list[Any]] = {
        name: _pick(values, rows) for name, values in case["columns"].items()
    }
    return executor_for(
        case["build"](policy),
        bundle,
        base_texts=base,
        counterfactual_texts=None if cf is None else _pick(cf, rows),
        extra_columns=columns or None,
        **kw,
    )


def _nested(value: torch.Tensor | RaggedValue, widths: list[int]) -> list[torch.Tensor]:
    """A read re-nested per row by ``widths`` — the reader the recorded
    geometry exists for; a dense read is already one row per row."""
    if isinstance(value, RaggedValue):
        assert list(value.widths) == widths
        return list(torch.split(value.flat, widths))
    return [value[i] for i in range(value.shape[0])]


# --------------------------------------------------------------------------- #
# T10 — both policies land, agree bit for bit, and match each row run alone
# --------------------------------------------------------------------------- #


def test_the_fixture_texts_are_ragged_as_documented(llama_bundle) -> None:
    """The widths every assertion below relies on, measured rather than
    assumed: two distinct content widths per pair, the counterfactual of each
    row as wide as the row, and the weekday entities at three and one piece."""
    assert _widths(llama_bundle, BASE_TEXTS) == _widths(llama_bundle, CF_TEXTS)
    assert len(set(_widths(llama_bundle, BASE_TEXTS))) == 2
    tokenizer = llama_bundle.tokenizer
    entity_widths = [len(tokenizer.tokenize(f" {e}")) for e in DAY_ENTITY]
    assert entity_widths == [3, 1], entity_widths
    assert [len(tokenizer.tokenize(f" {e}")) for e in DAY_CF_ENTITY] == entity_widths


def test_t10_both_policies_land_and_agree_bit_for_bit(case, llama_bundle) -> None:
    """The same forward, two landings: the patched logits and the written
    window are identical between ``exact_length_buckets`` and
    ``padded_masked`` — the mask keeps the padding out, the buckets never see
    it — and the geometry both recorded is the pre-forward width census."""
    rows = list(range(len(case["base"])))
    values = {}
    for policy in LANDING:
        executor = _executor(case, llama_bundle, policy, rows)
        values[policy] = (executor.read_value("logits"), executor.read_value("after"))
        geometry = executor.ragged_geometry[("patched", "patch")]
        assert geometry["policy"] == policy
        assert len(set(geometry["widths"])) == 2, geometry
        assert geometry["buckets"] == [
            [w, geometry["widths"].count(w)] for w in sorted(set(geometry["widths"]))
        ]
    (la, aa), (lb, ab) = values["exact_length_buckets"], values["padded_masked"]
    assert torch.equal(_tensor(la), _tensor(lb))
    aa, ab = _ragged(aa), _ragged(ab)
    assert aa.widths == ab.widths and torch.equal(aa.flat, ab.flat)


@pytest.mark.parametrize("policy", LANDING)
def test_t10_each_row_equals_the_row_run_alone(case, llama_bundle, policy) -> None:
    """Every row's written window and patched logits equal the row run alone
    (a uniform batch of one — the landing every document had before), so
    padding is never read as activation; the recorded ``widths`` are what
    re-nests the ragged window row by row (the reader mutation B breaks).
    The comparison is at the oracle tolerance, not bit for bit: a row alone
    sits in a different padded frame and fp32 batch kernels round differently
    from single-row ones, so bit equality is asserted only between the two
    policies (the test above), which share one frame."""
    rows = list(range(len(case["base"])))
    executor = _executor(case, llama_bundle, policy, rows)
    logits = _tensor(executor.read_value("logits"))
    after = executor.read_value("after")
    widths = executor.ragged_geometry[("patched", "patch")]["widths"]
    per_row = _nested(after, widths)
    for i in rows:
        alone = _executor(case, llama_bundle, None, [i])
        _close(logits[i : i + 1], alone.read_value("logits"))
        (want,) = _nested(alone.read_value("after"), [widths[i]])
        _close(per_row[i], want)


@pytest.mark.parametrize("policy", LANDING)
def test_t10_a_microbatched_layout_lands_the_same_values(
    case, llama_bundle, policy
) -> None:
    """Under ``batch_rows=1`` every window is one row and so uniform: the
    dense landing runs, with the ragged operand re-nested at that width. The
    result equals the whole-batch ragged landing — the policy changes what a
    landing does with the rows it holds, never which rows a forward holds."""
    rows = list(range(len(case["base"])))
    whole = _executor(case, llama_bundle, policy, rows)
    windowed = _executor(case, llama_bundle, policy, rows, batch_rows=1)
    _close(whole.read_value("logits"), windowed.read_value("logits"))
    a, b = _ragged(whole.read_value("after")), _ragged(windowed.read_value("after"))
    assert a.widths == b.widths
    _close(a.flat, b.flat)
    assert whole.ragged_geometry == windowed.ragged_geometry


# --------------------------------------------------------------------------- #
# T10 — a dense operand pairs into a ragged write at each row's own width
# --------------------------------------------------------------------------- #

#: Both counterfactual rows at the widest width: the operand read is uniform,
#: stored dense ``(2, 12, d)`` — as wide as the widest row, wider than row 0.
WIDE_CF = [CF_TEXTS[1], CF_TEXTS[1]]


def _one_position_operand_doc(policy: str) -> dict[str, Any]:
    """The ragged ``all`` write fed the counterfactual's last position only —
    a dense ``(rows, 1, d)`` operand, the one width every row accepts."""
    doc = _doc({"swap": "v_cf"}, pos="all", ragged=policy, with_counterfactual=True)
    doc["method"]["reads"]["v_cf"]["pos"] = {"index": -1}
    return doc


def _wide_dense_operand_refusal(bundle: Any, policy: str) -> ValidationError:
    """The probe under ``policy``: the write is refused before anything lands."""
    executor = executor_for(
        _swap_all(policy), bundle, base_texts=BASE_TEXTS, counterfactual_texts=WIDE_CF
    )
    w_wide = max(_widths(bundle, BASE_TEXTS))
    assert _tensor(executor.read_value("v_cf")).shape[:2] == (2, w_wide)
    with pytest.raises(ValidationError) as err:
        executor.read_value("logits")
    return err.value


@pytest.mark.parametrize("policy", LANDING)
def test_t10_a_uniform_dense_operand_wider_than_a_row_is_refused_under_both_policies(
    llama_bundle, policy
) -> None:
    """A dense operand as wide as the widest row
    satisfies the padded frame's broadcast, so without the per-row width check
    ``padded_masked`` landed it left-aligned and truncated into the narrower
    row, while ``exact_length_buckets`` refused the same document with
    ``_coerce``'s P2. Both now refuse it as rule 19, reason
    ``ragged_write_unsupported``, naming the disagreeing row and both widths —
    and only that row."""
    w_narrow, w_wide = _widths(llama_bundle, BASE_TEXTS)
    assert w_narrow < w_wide
    assert _widths(llama_bundle, WIDE_CF) == [w_wide, w_wide]
    err = _wide_dense_operand_refusal(llama_bundle, policy)
    assert err.rule == 19
    assert err.reason == "ragged_write_unsupported"
    assert f"on row 0 (operand {w_wide}, write {w_narrow})" in str(err)
    assert "row 1" not in str(err)  # the row whose widths agree is not named
    assert "does not broadcast" not in str(err)  # never `_coerce`'s P2


def test_t10_the_two_policies_refuse_the_wide_dense_operand_identically(
    llama_bundle,
) -> None:
    """One document, one refusal: the message under ``padded_masked`` is byte
    for byte the one under ``exact_length_buckets`` — §2.8's promise that the
    policies differ in nothing but the ``gaussian`` draw holds for refusals."""
    messages = {
        policy: str(_wide_dense_operand_refusal(llama_bundle, policy))
        for policy in LANDING
    }
    assert messages["padded_masked"] == messages["exact_length_buckets"]


def test_t10_a_one_position_operand_broadcasts_under_both_policies(
    llama_bundle,
) -> None:
    """The width the check keeps: a ``{"index": -1}`` read of the
    counterfactual is dense ``(rows, 1, d)`` and broadcasts over each row's
    own width under either policy — every landed position is the operand's
    one vector, and the two landings agree bit for bit."""
    values: dict[str, tuple[torch.Tensor, RaggedValue]] = {}
    for policy in LANDING:
        executor = executor_for(
            _one_position_operand_doc(policy),
            llama_bundle,
            base_texts=BASE_TEXTS,
            counterfactual_texts=CF_TEXTS,
        )
        after = _ragged(executor.read_value("after"))
        operand = _tensor(executor.read_value("v_cf"))
        assert operand.shape[:2] == (2, 1)
        for i, row in enumerate(torch.split(after.flat, list(after.widths))):
            assert torch.equal(row, operand[i].expand_as(row))
        values[policy] = (_tensor(executor.read_value("logits")), after)
    (la, aa), (lb, ab) = values["exact_length_buckets"], values["padded_masked"]
    assert torch.equal(la, lb)
    assert aa.widths == ab.widths and torch.equal(aa.flat, ab.flat)


@pytest.mark.parametrize("policy", LANDING)
def test_t10_a_dense_operand_at_the_row_s_own_width_lands_under_a_policy(
    llama_bundle, policy
) -> None:
    """Valid work still passes: one row at a time the write is uniform, the
    counterfactual read is dense and exactly as wide as the row, and the
    policy's dense landing — the width check at that width — writes bit for
    bit what the same row writes with no policy at all."""
    for i in range(len(BASE_TEXTS)):
        held = executor_for(
            _swap_all(policy),
            llama_bundle,
            base_texts=[BASE_TEXTS[i]],
            counterfactual_texts=[CF_TEXTS[i]],
        )
        plain = executor_for(
            _swap_all(None),
            llama_bundle,
            base_texts=[BASE_TEXTS[i]],
            counterfactual_texts=[CF_TEXTS[i]],
        )
        operand = _tensor(held.read_value("v_cf"))
        assert (
            operand.dim() == 3
            and operand.shape[1] == _widths(llama_bundle, [BASE_TEXTS[i]])[0]
        )
        assert torch.equal(
            _tensor(held.read_value("logits")), _tensor(plain.read_value("logits"))
        )
        assert torch.equal(
            _tensor(held.read_value("after")), _tensor(plain.read_value("after"))
        )
        assert not held.ragged_geometry  # a uniform window records nothing


# --------------------------------------------------------------------------- #
# T11 — refuse (and absent) keep rule 19's pre-forward guarantee
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("policy", [None, "refuse"])
def test_t11_refuse_and_absent_are_rule_19_before_any_forward(
    case, llama_bundle, monkeypatch, policy
) -> None:
    rows = list(range(len(case["base"])))
    executor = _executor(case, llama_bundle, policy, rows)
    widths = sorted(
        {
            len(row)
            for row in executor._positions(  # pyright: ignore[reportPrivateUsage]
                executor.doc.writes["patch"].pos,
                executor._batch("base"),  # pyright: ignore[reportPrivateUsage]
                "base",
            )
        }
    )
    assert len(widths) == 2

    def no_forward(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("refused too late: a forward pass already ran")

    monkeypatch.setattr(type(executor), "_run_group", no_forward)
    with pytest.raises(ValidationError) as err:
        executor.read_value("logits")
    assert err.value.rule == 19
    assert err.value.path == "writes.patch.pos"
    assert err.value.reason == "ragged_write_unsupported"
    assert str(widths) in str(err.value)
    assert "ragged position widths" in str(err.value)
    assert "all-positions or variable write" in str(err.value)
    assert not executor.ragged_geometry


def test_t11_a_ragged_operand_of_other_widths_is_the_same_refusal(
    llama_bundle,
) -> None:
    """Under a landing policy a ragged operand pairs into the write row by
    row; a row where its width is not the write's has no aligned shape and is
    rule 19 with the same reason — the twin is every T10 case above."""
    executor = executor_for(
        _swap_all("exact_length_buckets"),
        llama_bundle,
        base_texts=BASE_TEXTS,
        counterfactual_texts=["six", CF_TEXTS[1]],  # row 0: one word against two
    )
    assert _widths(llama_bundle, ["six"]) != _widths(llama_bundle, [BASE_TEXTS[0]])
    with pytest.raises(ValidationError) as err:
        executor.read_value("logits")
    assert err.value.rule == 19
    assert err.value.reason == "ragged_write_unsupported"
    assert "disagrees with the write's" in str(err.value)
    (w_base,) = _widths(llama_bundle, [BASE_TEXTS[0]])
    (w_cf,) = _widths(llama_bundle, ["six"])
    assert f"on row 0 (operand {w_cf}, write {w_base})" in str(err.value)
    assert "row 1" not in str(err.value)  # the row whose widths agree is not named


def test_t11_a_ragged_operand_under_refuse_keeps_its_message(llama_bundle) -> None:
    """A uniform write (an index) fed a ragged read under no policy: the
    operand refusal's text is unchanged, now with the reason code."""
    doc = _doc(
        {"swap": "v_cf"}, pos={"index": -1}, ragged=None, with_counterfactual=True
    )
    doc["method"]["reads"]["v_cf"]["pos"] = "all"
    executor = executor_for(
        doc, llama_bundle, base_texts=BASE_TEXTS, counterfactual_texts=CF_TEXTS
    )
    with pytest.raises(ValidationError) as err:
        executor.read_value("logits")
    assert err.value.rule == 19
    assert err.value.reason == "ragged_write_unsupported"
    assert "pairing ragged windows into a write" in str(err.value)


def _env_with_rows(tmp_path: Path, rows: list[dict[str, Any]]) -> ResolutionEnv:
    table = tmp_path / "data" / "ragged" / "rows.json"
    table.parent.mkdir(parents=True, exist_ok=True)
    table.write_text(json.dumps(rows))
    return ResolutionEnv(
        datasets=FileDatasets(root=tmp_path / "data"),
        artifacts=FileArtifacts(root=tmp_path / "artifacts"),
    )


#: The T12 table: two rows whose entities tokenize to three and one pieces,
#: each counterfactual entity as wide as its row's. The answer columns are
#: single tokens so the shipped metrics (`match`, `logit_diff`) reduce; their
#: content is not the point — tiny-random output is garbage by design.
LOCATE_ROWS = [
    {
        "input": DAY_BASE[0],
        "entity": DAY_ENTITY[0],
        "counterfactual_inputs": [DAY_CF[0]],
        "counterfactual_inputs_variables": [{"entity": DAY_CF_ENTITY[0]}],
        "base_answer": " Friday",
        "cf_answer": " Sunday",
        "split": "all",
    },
    {
        "input": DAY_BASE[1],
        "entity": DAY_ENTITY[1],
        "counterfactual_inputs": [DAY_CF[1]],
        "counterfactual_inputs_variables": [{"entity": DAY_CF_ENTITY[1]}],
        "base_answer": " Saturday",
        "cf_answer": " Sunday",
        "split": "all",
    },
]


def _locate_variant(policy: str | None) -> dict[str, Any]:
    """The shipped locate scan with the bare ``{"variable": "entity"}`` tap
    the description talks about, retargeted at tiny scale (two layers, the
    fixture table), authoring ``policy`` on its one write."""
    doc = json.loads(SHIPPED.read_text())
    doc["model"] = {"key": TINY_LLAMA, "revision": "main", "dtype": "fp32"}
    for role in doc["data"].values():
        role["dataset"] = "ragged/rows"
    method = doc["method"]
    method["positions"]["tap"]["sweep"][0] = {"variable": "entity"}
    method["sites"]["target"]["layers"] = {"sweep": {"range": [0, 2]}}
    if policy is not None:
        method["writes"]["patch"]["ragged"] = {"policy": policy}
    return doc


def test_t11_through_run_protocol_nothing_loads_and_nothing_runs(
    tmp_path, llama_bundle, monkeypatch
) -> None:
    """The engine path: a caller-owned bundle (so the tokenizer exists — rule
    19 needs it) with ``load_model`` replaced by one that raises, and a
    forward pre-hook that raises: the refusal is rule 19 with its reason, the
    receipt is written and carries no ``execution.ragged``, nothing loaded,
    nothing ran."""
    from causalab.neural.engines.pytorch_hooks import engine as hooks_engine

    def never(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("weights loaded")

    monkeypatch.setattr(hooks_engine, "load_model", never)
    handle = llama_bundle.model.register_forward_pre_hook(
        lambda *_: (_ for _ in ()).throw(AssertionError("a forward ran"))
    )
    try:
        env = _env_with_rows(tmp_path, LOCATE_ROWS)
        loaded = compile_protocol(in_order(_locate_variant(None)), env=env)
        out = tmp_path / "out"
        with pytest.raises(ValidationError) as err:
            run_protocol(
                loaded, env, PytorchHooksEngine(bundle=llama_bundle), out, record=True
            )
    finally:
        handle.remove()
    assert err.value.rule == 19
    assert err.value.reason == "ragged_write_unsupported"
    receipt = json.loads((out / RUN_RECORD_NAME).read_text())
    assert RAGGED_KEY not in receipt["execution"]


# --------------------------------------------------------------------------- #
# T12 — the locate scan's variable half runs under exact_length_buckets
# --------------------------------------------------------------------------- #


def _run_locate(tmp_path: Path, policy: str, name: str) -> tuple[Any, Path]:
    env = _env_with_rows(tmp_path, LOCATE_ROWS)
    loaded = compile_protocol(in_order(_locate_variant(policy)), env=env)
    out = tmp_path / name
    result = run_protocol(
        loaded, env, PytorchHooksEngine(device="cpu"), out, record=True
    )
    return result, out


def test_t12_the_variable_half_runs_under_exact_length_buckets(
    tmp_path, llama_bundle
) -> None:
    result, out = _run_locate(tmp_path, "exact_length_buckets", "buckets")
    assert len(result.summaries) == 4  # 2 layers × 2 taps
    assert not [cell for cell in result.cells if isinstance(cell, Unavailable)]
    for name in ("iia", "logit_diff"):
        table = table_frame(out / f"{name}.json")  # one row per (point, example)
        assert len(table) == 4 * len(LOCATE_ROWS) and table["value"].notna().all()
        assert table["eligible"].all()
    receipt = json.loads((out / RUN_RECORD_NAME).read_text())
    assert receipt["execution"][RAGGED_KEY] == {
        "patched/patch": {
            "policy": "exact_length_buckets",
            "widths": [3, 1],
            "buckets": [[1, 1], [3, 1]],
        }
    }
    assert receipt["execution"]["batch_rows"] is None  # beside the geometry (X5)
    # the variable half recorded the geometry, the last-token half is uniform
    ragged = [s for s in result.summaries if RAGGED_KEY in s]
    assert len(ragged) == 2
    assert {s["coords"]["positions.tap"]["variable"] for s in ragged} == {"entity"}


def test_t12_twin_padded_masked_writes_the_same_tables(tmp_path, llama_bundle) -> None:
    _, buckets = _run_locate(tmp_path, "exact_length_buckets", "buckets")
    _, masked = _run_locate(tmp_path, "padded_masked", "masked")
    for name in ("iia", "logit_diff"):
        a = table_frame(buckets / f"{name}.json")["value"].tolist()
        b = table_frame(masked / f"{name}.json")["value"].tolist()
        assert a == b
    receipt = json.loads((masked / RUN_RECORD_NAME).read_text())
    assert (
        receipt["execution"][RAGGED_KEY]["patched/patch"]["policy"] == "padded_masked"
    )


def test_t12_without_a_policy_the_same_document_is_rule_19(
    tmp_path, llama_bundle
) -> None:
    env = _env_with_rows(tmp_path, LOCATE_ROWS)
    loaded = compile_protocol(in_order(_locate_variant(None)), env=env)
    with pytest.raises(ValidationError) as err:
        run_protocol(loaded, env, PytorchHooksEngine(device="cpu"), tmp_path / "out")
    assert err.value.rule == 19 and err.value.reason == "ragged_write_unsupported"


def test_t12_the_shipped_document_keeps_its_spelling_and_authors_no_policy() -> None:
    """The shipped locate scan and its corpus twin: the same method up to the
    model's depth (the twin retargets the dataset refs and keeps the corpus
    model's 32-layer range where the shipped document sweeps Qwen2.5-7B's
    28), the last-token spelling on the second tap, no ``ragged`` field — both
    descriptions say why, naming the field this PR adds and keeping the
    rendered code."""
    shipped, twin = json.loads(SHIPPED.read_text()), json.loads(TWIN.read_text())
    depth = shipped["method"]["sites"]["target"]["layers"]
    assert depth == {"sweep": {"range": [0, 28]}}
    assert twin["method"]["sites"]["target"]["layers"] == {"sweep": {"range": [0, 32]}}
    twin["method"]["sites"]["target"]["layers"] = depth
    assert shipped["method"] == twin["method"]
    for document in (shipped, twin):
        assert "writes.<w>.ragged" in document["header"]["description"]
    tap = shipped["method"]["positions"]["tap"]["sweep"]
    assert tap == [{"index": -1}, {"index": -1, "scope": {"variable": "entity"}}]
    assert "ragged" not in shipped["method"]["writes"]["patch"]
    description = shipped["header"]["description"]
    assert "writes.<w>.ragged" in description and "[V19]" in description


def test_the_spellings_are_the_code_s() -> None:
    from causalab.protocol.receipt import RAGGED_KEY as key
    from causalab.protocol.schema import RAGGED_POLICIES

    assert key == RAGGED_KEY
    assert set(RAGGED_POLICIES) == {"refuse", *LANDING}


@pytest.mark.parametrize("policy", LANDING)
def test_renormalize_after_a_delta_restores_each_rows_pre_write_norm(
    bundle, policy: str
):
    """``add_scaled`` + ``renormalize`` over a ragged ``all`` window: every
    landed position has the norm it had before either write, under both
    landing policies (§2.8; the ragged landing once made ``renormalize`` the
    identity, re-reading the post-delta value as its reference)."""
    ragged = {"policy": policy}
    doc = _doc(
        {"add_scaled": {"op": 2.5, "alpha": 1.0}},
        ragged=policy,
        with_counterfactual=False,
    )
    method = doc["method"]
    method["writes"]["renorm"] = {
        "site": "tgt",
        "pos": "all",
        "do": {"renormalize": True},
        "ragged": ragged,
    }
    method["intervened_models"]["patched"]["writes"].append("renorm")
    method["reads"]["before"] = {"site": "tgt", "pos": "all"}
    method["intervened_models"]["original"] = {"input": "base", "reads": ["before"]}
    method["save"].append(saved("before", "original", "before.safetensors"))
    executor = executor_for(doc, bundle, base_texts=BASE_TEXTS)
    widths = _widths(bundle, BASE_TEXTS)
    before = _nested(executor.read_value("before"), widths)
    after = _nested(executor.read_value("after"), widths)
    for row, (b, a) in enumerate(zip(before, after)):
        _close(a.norm(dim=-1), b.norm(dim=-1))
        # anti-vacuity: the write moved the values, and the bumped norm is far
        assert float((a - b).abs().max()) > 1e-2, f"row {row}: nothing landed"
        assert float(((b + 2.5).norm(dim=-1) - b.norm(dim=-1)).abs().min()) > 1.0
