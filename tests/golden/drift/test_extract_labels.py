"""CPU guard for the drift tier's value extractor (runs in default CI).

The replay test (test_drift_goldens.py, GPU) compares ``extract_values``
against ``drift_goldens.json`` by key, so the extractor's *label grammar* is
a contract with the pins: a scan label is the sweep coordinates alone,
``scan.iia.<axis>=<coord>.mean``. Every other per-row column — the row keys,
the provenance stamp, the record identity, the eligibility record,
the windowed-read columns ``step`` / ``matched`` — is named in
``_META_COLUMNS`` and never becomes an axis. This module pins that contract
without a model: a synthetic scan table in the on-disk shape the run writes,
the writer's spelling of the eligibility record, a ``MetricTable`` round trip
over every column the writer emits, the pinned key set against the scan
document's own sweep, and ``compare`` naming a non-finite scalar — a
coordinate whose rows are all excluded means a ``NaN`` — instead of passing
it against any pin.

Why the table goes through ``write_table``: the eligibility record's on-disk
shape is what the extractor reads — ``reason_code`` is absent (not null) on
an eligible row and the excluded row's value is ``null`` — and pandas turns
both into NaN only on the way back from JSON; a hand-built DataFrame would
have to fake that.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from causalab.io.tables import write_table

from tests.golden.drift._extract import (
    DOCS,
    GOLDEN_PROTOCOLS,
    PINS,
    _META_COLUMNS,  # pyright: ignore[reportPrivateUsage]
    _frame,  # pyright: ignore[reportPrivateUsage]
    compare,
    extract_values,
    load_pins,
)

pytestmark = pytest.mark.unit

#: The swept document; its ``sites.target`` member carries the one sweep.
SCAN_DOCUMENT = GOLDEN_PROTOCOLS / "drift_locate_scan_im.json"
SCAN_PREFIX = "scan.iia."
#: The eligibility record's two columns, as the drift table spells them
#: (spec §2.10 "Eligibility"); the census test below couples this spelling
#: to the writer's constants.
ELIGIBILITY_RECORD = ("eligible", "reason_code")
#: A real spec §2.4 reason code — the one an unaligned row is excluded under.
EXCLUDED_REASON = "alignment_missing"
LAYERS = (6, 10, 14)
EXAMPLES = (0, 1)
#: The layer whose rows are turned into excluded measurements, example 0
#: first: one excluded row leaves it exactly one eligible row, both excluded
#: leave it none.
EXCLUDED_LAYER = 10


def _scan_axis() -> tuple[str, range]:
    """The scan document's swept coordinate — its label column and its
    values — read from ``method.sites.target``, the member whose one
    sub-object carries ``sweep``. The axis *name* is never spelled here, so
    the test follows a rename of the site field (9-1: ``layer`` →
    ``layers``) through the document and the pins together."""
    target = json.loads(SCAN_DOCUMENT.read_text())["method"]["sites"]["target"]
    swept = [k for k, v in target.items() if isinstance(v, dict) and "sweep" in v]
    assert len(swept) == 1, f"{SCAN_DOCUMENT.name} sweeps {swept}, expected one axis"
    (name,) = swept
    return f"sites.target.{name}", range(*target[name]["sweep"]["range"])


def _value(layer: int, example: int) -> float:
    return 0.25 * example + layer / 100


def _scan_rows(axis: str, *, excluded: int) -> list[dict[str, Any]]:
    """3 layers × 2 examples in the drift table's column set — the
    eligibility record's pinned set (tests/neural/shared/test_metric_eligibility.py) plus the
    ``point`` and ``name`` columns the drift run carries. ``excluded`` of
    ``EXCLUDED_LAYER``'s rows (the first ``excluded`` examples) are excluded
    measurements: ``eligible: false``, a ``reason_code``, a ``null`` value."""
    rows: list[dict[str, Any]] = []
    for layer in LAYERS:
        for example in EXAMPLES:
            rows.append(
                {
                    "example_id": str(example),
                    "point": f"point-{layer}",
                    "metric": "iia",
                    "name": "iia",
                    "unit": "fraction",
                    "estimand_version": "iia/v1",
                    axis: layer,
                    "eligible": True,
                    "value": _value(layer, example),
                }
            )
    for index, row in enumerate(rows):
        if (
            row[axis] == EXCLUDED_LAYER
            and int(row["example_id"]) in EXAMPLES[:excluded]
        ):
            rows[index] = {
                **row,
                "eligible": False,
                "reason_code": EXCLUDED_REASON,
                "value": None,
            }
    return rows


def _layout(root: Path, scan_rows: list[dict[str, Any]]) -> dict[str, Path]:
    """The two output directories ``extract_values`` reads, with the scan
    table given and the point document's tables minimal (their values are
    not under test here)."""
    from safetensors.torch import save_file
    import torch

    point, scan = (root / name.removesuffix("_im.json") for name in DOCS)
    for metric in ("acc", "iia", "ld"):
        write_table(point / f"{metric}.json", [{"example_id": "0", "value": 1.0}])
    point.mkdir(parents=True, exist_ok=True)
    save_file({"acts_mid": torch.zeros(2, 4)}, str(point / "acts_mid.safetensors"))
    write_table(scan / "iia.json", scan_rows)
    return dict(zip(DOCS, (point, scan)))


def _inline_form(frame: pd.DataFrame) -> dict[str, float]:
    """The reduction as ``extract_values`` spelled it inline before the
    helper existed — the reference the refactor is held to. The
    original's scalar-to-tuple normalisation of ``coords`` is dropped as
    dead: grouping by a list of keys yields tuples under pandas >= 2, one
    axis or many."""
    axes = [c for c in frame.columns if c not in _META_COLUMNS]
    labels: dict[str, float] = {}
    for coords, group in frame.groupby(axes):
        label = ",".join(f"{a}={c}" for a, c in zip(axes, coords))
        labels[label] = float(group["value"].mean())
    return labels


def _names_the_record(label: str) -> bool:
    return any(column in label for column in ELIGIBILITY_RECORD)


@pytest.mark.parametrize(
    "excluded",
    [0, 1, len(EXAMPLES)],
    ids=["every_row_eligible", "one_excluded_row", "all_rows_excluded"],
)
def test_the_eligibility_record_is_not_a_scan_axis(tmp_path: Path, excluded: int):
    axis, _ = _scan_axis()
    rows = _scan_rows(axis, excluded=excluded)
    values = extract_values(_layout(tmp_path, rows))
    scan = {k: v for k, v in values.items() if k.startswith(SCAN_PREFIX)}

    # a coordinate does not vanish because its rows were excluded: the label
    # for EXCLUDED_LAYER is present in every case
    assert set(scan) == {f"{SCAN_PREFIX}{axis}={n}.mean" for n in LAYERS}
    assert not [k for k in scan if _names_the_record(k)]
    # the means are over the eligible rows: an excluded row's value is null,
    # which pandas skips — and a coordinate with no eligible row is NaN
    for layer in LAYERS:
        eligible = [r["value"] for r in rows if r[axis] == layer and r["eligible"]]
        if eligible:
            assert scan[f"{SCAN_PREFIX}{axis}={layer}.mean"] == sum(eligible) / len(
                eligible
            )
        else:
            assert math.isnan(scan[f"{SCAN_PREFIX}{axis}={layer}.mean"])
    if excluded == 1:
        survivors = [
            r["value"] for r in rows if r[axis] == EXCLUDED_LAYER and r["eligible"]
        ]
        assert len(survivors) == 1
        assert scan[f"{SCAN_PREFIX}{axis}={EXCLUDED_LAYER}.mean"] == survivors[0]
    if excluded == len(EXAMPLES):
        # the replay of a real eligibility regression: pins that agree with
        # every other coordinate, and a NaN where the pin holds a number —
        # one mismatch, named, never a silent pass (`nan > tol` is False)
        nan_key = f"{SCAN_PREFIX}{axis}={EXCLUDED_LAYER}.mean"
        pins = {k: (0.0 if k == nan_key else v) for k, v in scan.items()}
        problems = compare(pins, scan, {})
        assert len(problems) == 1, problems
        assert nan_key in problems[0] and "not finite" in problems[0], problems


def test_the_labels_are_byte_identical_with_and_without_the_record(tmp_path: Path):
    # the helper did not exist where the reduction was inline
    from tests.golden.drift._extract import (
        _scan_labels,  # pyright: ignore[reportPrivateUsage]
    )

    axis, _ = _scan_axis()
    with_record = _scan_rows(axis, excluded=1)
    # before the eligibility record: no eligibility columns, and an excluded
    # measurement wrote no row
    without_record = [
        {k: v for k, v in row.items() if k not in ELIGIBILITY_RECORD}
        for row in with_record
        if row["eligible"]
    ]
    write_table(tmp_path / "with" / "iia.json", with_record)
    write_table(tmp_path / "without" / "iia.json", without_record)
    post = _frame(tmp_path / "with" / "iia.json")
    pre = _frame(tmp_path / "without" / "iia.json")
    assert set(post.columns) - set(pre.columns) == set(ELIGIBILITY_RECORD)

    # the refactor changes nothing for a table without the record …
    assert _scan_labels(pre) == _inline_form(pre)
    # … and the record changes nothing for a table with it
    assert _scan_labels(post) == _scan_labels(pre)
    assert set(_scan_labels(post)) == {f"{axis}={n}" for n in LAYERS}


def test_the_meta_columns_spell_the_writers_eligibility_record():
    # the writer's constants are torch-side, so _extract.py keeps literals and
    # this census couples the two spellings (the import is fine in a CPU test:
    # tests/neural/shared/test_metric_eligibility.py already makes it)
    from causalab.neural.shared.results import ELIGIBLE_COLUMN, REASON_CODE_COLUMN

    assert (ELIGIBLE_COLUMN, REASON_CODE_COLUMN) == ELIGIBILITY_RECORD
    assert {ELIGIBLE_COLUMN, REASON_CODE_COLUMN} <= _META_COLUMNS
    # the record identity, the precedent
    assert {"unit", "estimand_version"} <= _META_COLUMNS
    axis, _ = _scan_axis()
    assert axis not in _META_COLUMNS


def test_every_writer_column_that_is_not_a_coordinate_is_a_meta_column():
    """The census over the class, not the two columns that last broke: a real
    ``MetricTable`` round trip through both writers — ``add`` with an eligible
    and an excluded value, ``add_windowed`` with a scored and an unmatched
    example — and every column it wrote that is not a coordinate is a meta
    column. Fails with ``['matched', 'step']`` if the windowed-read columns
    leave ``_META_COLUMNS``; it would equally have caught the eligibility
    record and the record identity before either was added there, each one
    nightly early."""
    from causalab.neural.shared.results import MetricTable
    from causalab.protocol.estimand import metric_record_identity
    from causalab.protocol.results import Unavailable

    coords = {"sites.target.layer": 6}
    # the row's `metric` column is the authored name (iia, ld); its identity
    # columns come from the kind — the drift documents' iia is a `match`, ld a
    # `logit_diff` (estimand.METRIC_UNITS)
    iia = metric_record_identity("match", unit=None, estimand_version=None)
    ld = metric_record_identity("logit_diff", unit=None, estimand_version=None)
    excluded = Unavailable(
        reason=EXCLUDED_REASON, detail="row 1: no answer", denominator_key="match"
    )
    table = MetricTable()
    table.add("iia", [0.5, excluded], coords, identity=iia)
    table.add_windowed(
        "ld",
        [[0.1, 0.2], []],
        coords,
        identity=ld,
        steps=[[0, 1], []],
        matched=[True, False],
    )
    written = {c for row in table.rows for c in row}
    # both writers reached the table, the excluded and unmatched rows included
    assert len(table.rows) == 5
    assert set(coords) <= written
    assert written - set(coords) <= _META_COLUMNS, sorted(
        written - set(coords) - _META_COLUMNS
    )


def test_compare_names_a_non_finite_scalar_on_either_side():
    # a NaN in the run
    problems = compare({"k": 1.0}, {"k": float("nan")}, {})
    assert len(problems) == 1 and "k" in problems[0] and "not finite" in problems[0]
    # a NaN in the pins is as blind
    problems = compare({"k": float("nan")}, {"k": 1.0}, {})
    assert len(problems) == 1 and "k" in problems[0] and "not finite" in problems[0]
    # valid work still passes, and the tolerance path is unchanged
    assert compare({"k": 1.0}, {"k": 1.0}, {}) == []
    assert len(compare({"k": 1.0}, {"k": 1.5}, {"default": 0.1})) == 1


def test_the_capture_refuses_to_write_a_non_finite_pin():
    """``update_drift_goldens.py`` serialises the pins through the pure
    ``serialise_pins`` (hoisted out of ``main()``, which sits behind argparse
    and a GPU run), so this test calls the code rather than grepping its
    source: a non-finite value — a coordinate with every row excluded — is
    refused BY NAME before anything is written (``allow_nan=False`` stays as
    the backstop; the ``json.dumps`` default writes a bare ``NaN`` token no
    other JSON parser accepts — tables.py), and the accuracy gate refuses a
    NaN too (its own test below)."""
    from tests.golden.drift.update_drift_goldens import non_finite_keys, serialise_pins

    with pytest.raises(ValueError, match="b.mean"):
        serialise_pins(
            {
                "values": {"a.mean": 1.0, "b.mean": float("nan")},
                "capture": {"device": "cuda", "dtype": "bf16"},
            }
        )
    # valid pins still serialise — a shape list is not a scalar
    pins = {"values": {"a.mean": 1.0, "s.shape": [2, 3]}}
    text = serialise_pins(pins)
    assert text.endswith("\n")
    assert json.loads(text) == pins
    assert non_finite_keys({"a": 1.0, "b": float("inf"), "c": [1, 2]}) == ["b"]


def test_the_capture_gate_refuses_a_non_finite_accuracy():
    """``nan < ACCURACY_GATE`` is False, so a NaN baseline accuracy — every
    example excluded — would pass the capture's gate; ``gate_refuses`` (the
    predicate ``main()`` uses) refuses it, and the census names the key."""
    from tests.golden.drift.update_drift_goldens import (
        ACCURACY_GATE,
        gate_refuses,
        non_finite_keys,
    )

    assert non_finite_keys({"interchange.acc.mean": float("nan")}) == [
        "interchange.acc.mean"
    ]
    assert gate_refuses(float("nan")) is True
    assert gate_refuses(0.5) is True
    assert gate_refuses(1.0) is False
    assert gate_refuses(ACCURACY_GATE) is False


def test_the_pinned_scan_keys_are_the_layers_of_the_scan_document():
    axis, layers = _scan_axis()
    pinned = {k for k in load_pins(PINS)["values"] if k.startswith(SCAN_PREFIX)}
    assert pinned == {f"{SCAN_PREFIX}{axis}={n}.mean" for n in layers}
    assert not [k for k in pinned if _names_the_record(k)]
