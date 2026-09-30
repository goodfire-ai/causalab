"""A gate built from a score table (spec §2.5 ``init.from_scores``; rule 32's
build-time half) — the pure layer, through [`build_stack`][causalab.neural.shared.featurizers.build.build_stack] with a fake
table loader, no model anywhere.

What is pinned: under ``keep`` the top-``keep`` units by score sit on the kept
pole of the gate's map and every other unit on the dropped pole, so the hard
mask *is* the ranking's top-``keep`` set under all three maps and ties break by
unit index; under ``scale`` theta is the z-scored score around the midpoint
mask; ``where`` narrows a table over several sites; a grouped gate reads one
row per head and a position gate one row per position (θ sized by the window,
not the chain's width); and every way a table can fail to cover the gate — a
unit missing, repeated, out of range, unscored, the wrong number of unit
columns, ``keep`` above the count — is the same refusal, rule 32, naming the
field. The table's bytes are the start's identity.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
import torch

from causalab.neural.shared.featurizers import Gate, build_stack, gate_poles
from causalab.protocol.rules.errors import ProtocolError, ValidationError
from causalab.protocol.schema import FeaturizerSpec, hard_concrete_threshold
from causalab.protocol.registry.shapes import bs_flat_heads

pytestmark = pytest.mark.unit

WIDTH = 8
#: eight units; the ranking by score is 5, 2, 7, 0, 3, 6, 1, 4 (unit 4 is
#: NaN-free but lowest), with units 3 and 6 tied — the tie resolves to 3
SCORES = [4.0, 1.0, 8.0, 3.0, 0.5, 9.0, 3.0, 6.0]
RANKING = [5, 2, 7, 0, 3, 6, 1, 4]


def _table(rows: list[dict[str, Any]]):
    raw = json.dumps(rows).encode()

    def load_table(_path: str):
        return rows, raw

    return load_table, raw


def _rows(scores=SCORES, *, unit="unit", value="value", **extra):
    return [{unit: i, value: s, **extra} for i, s in enumerate(scores)]


def _gate(
    init: dict[str, Any],
    *,
    rows: list[dict[str, Any]] | None = None,
    parametrization: str | None = None,
    stretch: Any = None,
    group: str | None = None,
    site_shape: Any = None,
    width: int = WIDTH,
    load_table: Any = "table",
    axis: str | None = None,
    position_width: int | None = None,
) -> Gate:
    if load_table == "table":
        load_table, _ = _table(_rows() if rows is None else rows)
    spec = FeaturizerSpec(
        kind="gate",
        group=group,
        axis=axis,
        parametrization=parametrization,
        stretch=stretch,
        init={
            "from_scores": {
                "file_path": "scores.json",
                "unit": "unit",
                "value": "value",
                **init,
            }
        },
    )
    stack = build_stack(
        "g",
        {"g": spec},
        width=width,
        load_tensors=lambda path: (_ for _ in ()).throw(KeyError(path)),
        load_table=load_table,
        stage_cache={},
        site_shape=site_shape,
        site_component=None if site_shape is None else "attention_premix",
        position_width=position_width,
    )
    (stage,) = stack.stages
    assert isinstance(stage, Gate)
    return stage


# -- keep: the ranking's top set, on the map's poles ------------------------- #


@pytest.mark.parametrize("keep", [1, 3, 5, WIDTH])
def test_keep_puts_the_top_units_on_the_kept_pole(keep: int) -> None:
    gate = _gate({"keep": keep})
    kept = sorted(RANKING[:keep])
    assert gate.hard_mask().nonzero().flatten().tolist() == kept
    assert float(gate.hard_mask().sum()) == keep
    dropped, on = gate_poles("sigmoid", None)
    assert torch.equal(gate.theta[kept], torch.full((keep,), on))
    rest = [i for i in range(WIDTH) if i not in kept]
    assert torch.equal(gate.theta[rest], torch.full((len(rest),), dropped))
    assert gate.init_scores == {
        "file_path": "scores.json",
        "units": WIDTH,
        "keep": keep,
        "kept_units": kept,
    }


def test_a_tie_breaks_by_unit_index_so_the_start_is_a_function_of_the_table() -> None:
    # units 3 and 6 both score 3.0; the fifth kept unit is 3, never 6
    assert 3 in _gate({"keep": 5}).init_scores["kept_units"]
    assert 6 not in _gate({"keep": 5}).init_scores["kept_units"]
    assert 6 in _gate({"keep": 6}).init_scores["kept_units"]


@pytest.mark.parametrize(
    "parametrization, stretch",
    [("clamp", None), ("hard_concrete", None), ("hard_concrete", (-0.3, 1.5))],
)
def test_the_poles_follow_the_map_and_the_hard_mask_is_the_ranking(
    parametrization: str, stretch
) -> None:
    gate = _gate({"keep": 3}, parametrization=parametrization, stretch=stretch)
    assert gate.hard_mask().nonzero().flatten().tolist() == sorted(RANKING[:3])
    dropped, kept = gate_poles(parametrization, stretch)
    poles = torch.tensor([dropped, kept])
    assert torch.allclose(
        torch.tensor(sorted(set(gate.theta.tolist()))), poles, atol=1e-6
    )
    if parametrization == "hard_concrete":
        # one unit either side of *this* stretch's threshold, not of zero
        threshold = hard_concrete_threshold((-0.1, 1.1) if stretch is None else stretch)
        assert (dropped, kept) == (threshold - 1.0, threshold + 1.0)
        assert dropped < gate.hard_threshold() < kept
    else:
        assert (dropped, kept) == (0.0, 1.0)


def test_the_start_is_decisive_under_every_map() -> None:
    """A ranking start is a mask, not a hint: the soft mask sits at the poles
    too, so a fit that begins here begins from the hard split it would be
    scored through, and an *untrained* gate from a ranking applies as that
    split — the attribution / magnitude-pruning baseline in one document."""
    for parametrization in ("sigmoid", "clamp", "hard_concrete"):
        gate = _gate({"keep": 2}, parametrization=parametrization)
        soft = gate.soft_mask()
        assert torch.all((soft > 0.7) | (soft < 0.3)), (parametrization, soft)


# -- scale: z-scores around the midpoint ------------------------------------- #


def test_scale_writes_the_z_scored_table_around_the_midpoint_mask() -> None:
    gate = _gate({"scale": 2.0})
    values = torch.tensor(SCORES)
    z = (values - values.mean()) / values.std(unbiased=False)
    assert torch.allclose(gate.theta, 2.0 * z, atol=1e-6)
    # the ordering survives, so the hard mask is "above the mean" and the
    # ranking is readable off theta
    assert torch.argsort(gate.theta, descending=True).tolist() == RANKING
    assert gate.init_scores == {
        "file_path": "scores.json",
        "units": WIDTH,
        "scale": 2.0,
    }


def test_scale_under_clamp_is_clipped_to_the_unit_interval_around_a_half() -> None:
    gate = _gate({"scale": 1.0}, parametrization="clamp")
    assert float(gate.theta.min()) >= 0.0 and float(gate.theta.max()) <= 1.0
    values = torch.tensor(SCORES)
    z = (values - values.mean()) / values.std(unbiased=False)
    assert torch.allclose(gate.theta, (0.5 + z).clamp(0.0, 1.0), atol=1e-6)


def test_a_constant_table_under_scale_is_the_untouched_start() -> None:
    gate = _gate({"scale": 3.0}, rows=_rows([1.0] * WIDTH))
    assert torch.equal(gate.theta, Gate(WIDTH).theta)


# -- where, columns, groups ------------------------------------------------- #


def test_where_picks_this_gates_rows_out_of_a_table_over_several_sites() -> None:
    rows = _rows(layer=3) + _rows([float(i) for i in range(WIDTH)], layer=5)
    at_3 = _gate({"keep": 2, "where": {"layer": 3}}, rows=rows)
    at_5 = _gate({"keep": 2, "where": {"layer": 5}}, rows=rows)
    assert at_3.init_scores["kept_units"] == sorted(RANKING[:2])
    assert at_5.init_scores["kept_units"] == [WIDTH - 2, WIDTH - 1]
    assert at_3.init_scores["where"] == {"layer": 3}


def test_the_columns_are_the_documents_to_name() -> None:
    rows = _rows(unit="head", value="mean")
    gate = _gate({"keep": 1, "unit": "head", "value": "mean"}, rows=rows)
    assert gate.init_scores["kept_units"] == [RANKING[0]]


def test_a_head_gate_reads_one_row_per_head() -> None:
    heads, head_dim = 4, 2
    rows = [{"unit": h, "value": [0.1, 0.9, 0.5, 0.2][h]} for h in range(heads)]
    gate = _gate(
        {"keep": 2},
        rows=rows,
        group="head",
        site_shape=bs_flat_heads(heads, head_dim),
        width=heads * head_dim,
    )
    assert gate.theta.shape == (heads,)
    assert gate.init_scores["kept_units"] == [1, 2]
    # the hard mask is per head: every coordinate of heads 1 and 2 is kept
    kept, _ = gate.featurize(torch.ones(heads * head_dim))
    assert kept.tolist() == [0, 0, 1, 1, 1, 1, 0, 0]


def test_a_position_gate_reads_one_row_per_position() -> None:
    """§2.5 ``axis``: θ is sized by the addressed window, so a unit index is a
    *position* and the table covers positions — a per-unit table indexed by
    token position rather than coordinate. ``width`` here is the site's 8: the
    gate takes the window, not the chain's feature width."""
    rows = [{"unit": t, "value": [0.1, 0.9, 0.5][t]} for t in range(3)]
    gate = _gate({"keep": 2}, rows=rows, axis="position", position_width=3)
    assert gate.theta.shape == (3,)
    assert gate.init_scores["kept_units"] == [1, 2]
    assert gate.hard_mask().tolist() == [0.0, 1.0, 1.0]


def test_a_table_start_stamps_no_init_identity() -> None:
    """A score table is a JSON table, not a tensor bundle: it has no
    ArtifactIdentity to record, and its bytes are the canonical form's
    (``init.from_scores.content_digest``), so the gate stamps nothing."""
    load_table, _raw = _table(_rows())
    gate = _gate({"keep": 1}, load_table=load_table)
    assert gate.init_identity == {}


# -- rule 32: every way a table fails to cover the gate ---------------------- #


def _refuses(
    init: dict[str, Any], rows: list[dict[str, Any]], match: str, **kw
) -> None:
    with pytest.raises(ValidationError, match=match) as err:
        _gate(init, rows=rows, **kw)
    assert err.value.rule == 32
    assert err.value.path.startswith("featurizers.g.init.from_scores")


def test_a_unit_missing_from_the_table_is_refused() -> None:
    _refuses({"keep": 1}, _rows()[:-1], "names 7 of the gate's 8 units")


def test_a_unit_named_twice_is_refused() -> None:
    _refuses({"keep": 1}, _rows() + _rows()[:1], "named twice")


def test_a_unit_outside_the_gate_is_refused() -> None:
    _refuses({"keep": 1}, _rows() + [{"unit": WIDTH, "value": 1.0}], "outside the gate")


def test_an_unscored_unit_is_refused() -> None:
    rows = _rows()
    rows[2]["value"] = float("nan")
    _refuses({"keep": 1}, rows, "has no score")
    rows[2]["value"] = "high"
    _refuses({"keep": 1}, rows, "a number under 'value'")


def test_the_wrong_number_of_unit_columns_is_refused() -> None:
    _refuses({"keep": 1, "unit": ["a", "b"]}, _rows(), "one unit column per axis")


def test_keep_above_the_unit_count_is_refused() -> None:
    _refuses({"keep": WIDTH + 1}, _rows(), "exceeds the gate's 8 units")


def test_a_position_table_skipping_a_position_is_refused() -> None:
    """The coverage check counts positions on a position gate (§2.5 ``axis``)."""
    rows = [{"unit": t, "value": 1.0} for t in (0, 2)]
    _refuses(
        {"keep": 1},
        rows,
        "names 2 of the gate's 3 units",
        axis="position",
        position_width=3,
    )


def test_keep_above_the_position_count_is_refused_at_the_build() -> None:
    rows = [{"unit": t, "value": 1.0} for t in range(3)]
    _refuses(
        {"keep": 4},
        rows,
        "exceeds the gate's 3 units",
        axis="position",
        position_width=3,
    )


def test_where_that_leaves_no_rows_is_the_coverage_refusal() -> None:
    _refuses(
        {"keep": 1, "where": {"layer": 99}},
        _rows(layer=3),
        r"names 0 of the gate's 8 units after where=\{'layer': 99\}",
    )


def test_a_build_without_a_table_loader_refuses_by_name() -> None:
    with pytest.raises(ProtocolError, match="needs a table loader"):
        _gate({"keep": 1}, load_table=None)


def test_a_swept_keep_reaching_the_build_is_refused() -> None:
    with pytest.raises(ProtocolError, match="keep is unresolved"):
        _gate({"keep": {"sweep": [1, 2]}})
