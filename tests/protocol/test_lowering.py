"""``protocol/lowering.py`` — what the protocol layer decides from the axes
alone, and the declared API break.

The count ([`point_count`][causalab.protocol.lowering.point_count]) is the enumeration's length for every corpus
and shipped document and for the named-axes fixtures; the representatives
([`representative_trees`][causalab.protocol.lowering.representative_trees]) are real steps — every one is a tree the
engine also enumerates, the first is the campaign's first step, there is one
per axis value with the all-first tree counted once — and the capability set
derived from them is the union over every step. The old modules
(``sweep.py``, ``axes.py``, ``families.py``, ``dry_run.py``, ``cli.py``) were
one-beat star-import shims, since deleted.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from causalab.neural.shared.sweep import enumerate_steps
from causalab.protocol.compiled import CompiledProtocol
from causalab.protocol.rules.errors import ValidationErrors
from causalab.protocol.identity import step_digest
from causalab.protocol.pipeline import compile_protocol
from causalab.protocol.lowering import (
    axes_of,
    point_count,
    representative_trees,
    substitute,
)
from causalab.io.env import ResolutionEnv
from causalab.protocol.rules.capability import requires_campaign
from causalab.protocol.schema import parse_document

from tests.protocol._docs import base_doc, in_order
from tests.protocol._env import CORPUS_DIR
from tests.protocol.test_axes import entity_doc, rome_doc
from tests.protocol.test_shipped_digests import COVERED, load_shipped


pytestmark = pytest.mark.unit

CORPUS = sorted(p.name for p in CORPUS_DIR.glob("*_im.json"))


def _check(compiled: CompiledProtocol) -> None:
    expansion = enumerate_steps(compiled)
    n = point_count(compiled.axes)
    assert n == len(expansion.points)
    assert axes_of(compiled.tree, compiled.named_axes) == compiled.axes
    trees = representative_trees(compiled.tree, compiled.axes, compiled.named_axes)
    steps = [json.dumps(p.raw, sort_keys=True) for p in expansion.points]
    assert json.dumps(trees[0], sort_keys=True) == steps[0]  # the first step
    expected = 1 + sum(len(axis.values) - 1 for axis in compiled.axes)
    assert len(trees) == expected == len(compiled.representatives)
    for tree in trees:
        assert json.dumps(tree, sort_keys=True) in steps  # a real step
    # in enumeration order, so aggregated refusals read in point order
    positions = [steps.index(json.dumps(tree, sort_keys=True)) for tree in trees]
    assert positions == sorted(positions) and len(set(positions)) == len(positions)
    # every axis value is represented once
    for axis in compiled.axes:
        seen = [
            json.dumps(p.coords[axis.id], sort_keys=True)
            for p in expansion.points
            if json.dumps(p.raw, sort_keys=True)
            in {json.dumps(t, sort_keys=True) for t in trees}
        ]
        assert set(seen) == {json.dumps(v, sort_keys=True) for v in axis.values}
    # the capability set is the union over every step
    every = requires_campaign([parse_document(p.raw) for p in expansion.points])
    assert compiled.capabilities == every


@pytest.mark.parametrize("name", CORPUS)
def test_corpus_counts_and_representatives(name: str, env: ResolutionEnv) -> None:
    loaded = compile_protocol(CORPUS_DIR / name, env=env)
    _check(loaded)


@pytest.mark.parametrize("name", COVERED)
def test_shipped_counts_and_representatives(name: str, env: ResolutionEnv) -> None:
    loaded = load_shipped(name, env)
    _check(loaded)


@pytest.mark.parametrize("doc", [entity_doc, rome_doc], ids=["entity", "rome"])
def test_named_axes_counts_and_representatives(doc: Any, env: ResolutionEnv) -> None:
    compiled = compile_protocol(doc(), env=env)
    assert compiled.named_axes is not None
    _check(compiled)


def test_a_document_with_no_axes_is_its_own_representative(env: ResolutionEnv) -> None:
    compiled = compile_protocol(CORPUS_DIR / "01_harvest_im.json", env=env)
    assert compiled.axes == () and point_count(compiled.axes) == 1
    assert representative_trees(compiled.tree, (), None) == (compiled.tree,)
    assert compiled.representatives == (compiled.document,)
    # the one representative is the one step: its digest is the campaign's
    assert step_digest(compiled.tree, env) == compiled.digests.document


def test_distinct_violations_over_two_axes_read_in_point_order(
    env: ResolutionEnv,
) -> None:
    """Two axes, a violation on each: the aggregated refusal lists them in
    the order the earlier per-point checklist gave them (last axis fastest —
    the site's second value at point 1 precedes the layer's second value at
    point 2), byte for byte the text that checklist printed."""
    doc = base_doc()
    doc["method"]["sites"]["tgt"]["layers"] = {"sweep": [[0], [99]]}
    doc["method"]["writes"]["patch"]["site"] = {"sweep": ["tgt", "nowhere"]}
    with pytest.raises(ValidationErrors) as err:
        compile_protocol(in_order(doc), env=env)
    assert str(err.value) == (
        "2 independent checklist violations (docs/intervention_protocol.md §5):\n"
        "  [V4] at writes.patch.site site 'nowhere' is not declared\n"
        "  [V4] at sites.tgt.layers site 'tgt': layer 99 out of range for the "
        "12-layer model 'gpt2'"
    )


def test_substitute_replaces_every_wrapper_and_nothing_else() -> None:
    tree = {"a": {"sweep": [1, 2]}, "b": [{"c": {"sweep": ["x"]}}, 3], "d": 4}
    out = substitute(tree, {("a",): 2, ("b", "c"): "x"}, ())
    assert out == {"a": 2, "b": [{"c": "x"}, 3], "d": 4}
    assert tree["a"] == {"sweep": [1, 2]}  # a pure function


# --------------------------------------------------------------------------- #
# the API break
# --------------------------------------------------------------------------- #

# The one-beat star-import shims (``sweep.py``, ``axes.py``, ``families.py``,
# ``dry_run.py``, ``cli.py``) and their pins are deleted.


def test_the_engine_side_names_are_where_the_break_says() -> None:
    from causalab.neural.shared import sweep as engine_sweep

    for name in ("Expansion", "Point", "expand", "expand_axes"):
        assert hasattr(engine_sweep, name), name


#: Every other public name taken off an old path without a shim — the
#: rest of the declared API break, in one place: the receipt's writers moved
#: engine-side with new signatures (a protocol-side shim would import
#: ``neural``), the per-point compile objects are gone, and the package no
#: longer exports the engine's ``Expansion`` / ``expand``. ``(module, name)``
#: → the module that holds the replacement, or ``None`` when nothing does.
#: (``protocol/compile.py`` and ``protocol/run.py`` dissolved into ``pipeline.py``,
#: so their rows are gone with them; ``protocol/loader.py`` — ``load``
#: and ``LoadedProtocol``, the flat view over a compile — retired after it,
#: every caller compiling through ``pipeline.compile_protocol`` and reading
#: ``CompiledProtocol``.)
DROPPED: dict[tuple[str, str], str | None] = {
    ("causalab.protocol.receipt", "write_run_record"): "causalab.neural.shared.receipt",
    ("causalab.protocol.receipt", "run_events"): "causalab.neural.shared.receipt",
    ("causalab.protocol.receipt", "emit_run_events"): "causalab.neural.shared.receipt",
    ("causalab.protocol.compiled", "CompiledPoint"): None,
    ("causalab.protocol.compiled", "CompiledPoints"): None,
    ("causalab.protocol", "Expansion"): "causalab.neural.shared.sweep",
    ("causalab.protocol", "load"): None,
    ("causalab.protocol", "LoadedProtocol"): None,
    ("causalab.protocol", "expand"): "causalab.neural.shared.sweep",
}


@pytest.mark.parametrize("module,name", sorted(DROPPED), ids=lambda x: str(x))
def test_every_other_dropped_name_is_declared_here(module: str, name: str) -> None:
    import importlib

    mod = importlib.import_module(module)
    assert not hasattr(mod, name), f"{module}.{name} is back — update DROPPED"
    assert name not in getattr(mod, "__all__", ())
    home = DROPPED[(module, name)]
    if home is not None:
        assert hasattr(importlib.import_module(home), name), (module, name, home)
