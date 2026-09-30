"""The per-step checklist, engine-side.

The compiler holds one representative per axis value to the §5 checklist
(``pipeline.validate``); the engine holds **every enumerated step** to the
whole checklist again, because a violation can appear at a combination of
axis values no representative reaches — and which rules can be broken that
way depends on what is swept, not on the rule. The boundary is stated by
three documents: a one-axis sweep whose second value collides (rule 8) is
refused per step with the checklist's own text; a two-axis document whose
only illegal step is (second value, second value) — a depth inversion (rule
21), or a swept model crossed with a layer the smaller model lacks (rule 4)
— passes ``pipeline.validate`` and is refused by
[`check_steps`][causalab.neural.shared.step_rules.check_steps] before the engine
plans anything, and through the ``run_protocol`` door before anything is
written.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from causalab.neural.shared.step_rules import check_steps
from causalab.neural.shared.sweep import enumerate_steps, expand
from causalab.protocol.pipeline import compile_protocol
from causalab.protocol.compiled import CompiledProtocol
from causalab.protocol.engine import Engine, RunContext, RunResult
from causalab.protocol.rules.errors import ValidationError, ValidationErrors
from causalab.protocol.pipeline import validate
from causalab.protocol.receipt import RUN_RECORD_NAME
from causalab.protocol.schema import COMPONENTS
from causalab.io.env import ResolutionEnv
from causalab.protocol.rules.document import validate_document
from causalab.protocol.pipeline import run_protocol
from causalab.protocol.schema import Document, parse_document

from tests._helpers.stub_engine import stub_execute
from tests.protocol._docs import UNWRITTEN, base_doc, in_order


pytestmark = pytest.mark.unit


def _docs(points: Any) -> list[Document]:
    return [parse_document(p.raw) for p in points]


def _colliding_sweep() -> dict[str, Any]:
    """Two absolute writes at ``tgt``: ``patch`` at −1, ``patch2`` swept over
    [−2, −1] — the first step is disjoint, the second collides (rule 8)."""
    doc = base_doc()
    doc["method"]["writes"]["patch2"] = {
        "site": "tgt",
        "pos": {"sweep": [-2, -1]},
        "do": {"swap": "v_cf"},
    }
    doc["method"]["intervened_models"]["patched"]["writes"].append("patch2")
    return in_order(doc)


def _two_axis_reachability() -> dict[str, Any]:
    """A write fed from a read on another site, both layers swept so that the
    read is deeper than the write at exactly one combination: read layers
    [1, 5], write layers [9, 3] — (5, 3) is the only step rule 21 refuses,
    and it is no representative (each axis's second value, together)."""
    doc = base_doc()
    doc["method"]["sites"]["src"] = {
        "component": "block_output",
        "layers": {"sweep": [[1], [5]]},
    }
    doc["method"]["sites"]["dst"] = {
        "component": "block_output",
        "layers": {"sweep": [[9], [3]]},
    }
    doc["method"]["reads"]["v_src"] = {"site": "src", "pos": -1}
    doc["method"]["reads"].pop("v_cf")
    doc["method"]["intervened_models"][UNWRITTEN]["reads"] = ["v_src"]
    doc["method"]["sites"].pop("tgt")
    doc["method"]["writes"] = {
        "patch": {
            "site": "dst",
            "pos": -1,
            "do": {"add_scaled": {"op": "v_src", "alpha": 1.0}},
        }
    }
    return in_order(doc)


def _model_by_layer() -> dict[str, Any]:
    """The model itself swept beside a layer: a 24-layer model and the
    12-layer ``gpt2``, layers [0, 14] — (gpt2, 14) is the one step rule 4
    refuses, and no representative (each axis's second value, together)."""
    doc = base_doc()
    doc["model"]["key"] = {"sweep": ["Qwen/Qwen2.5-0.5B", "gpt2"]}
    doc["method"]["sites"]["tgt"]["layers"] = {"sweep": [[0], [14]]}
    return in_order(doc)


class _Stub(Engine):
    """An engine that executes nothing and plays the shared driver's opening
    moves: enumerate, check every step, sign, record when asked."""

    name = "stub"
    capabilities = frozenset(
        {"grad", "paired_forward", "full_logits", "pytorch_fn_local"}
    )
    components = frozenset(COMPONENTS)
    writable_components = frozenset(COMPONENTS)
    is_local = True

    def execute(self, compiled: CompiledProtocol, run: RunContext) -> RunResult:
        return stub_execute(self, compiled, run)


def test_a_colliding_step_is_refused_with_the_checklists_text() -> None:
    expansion = expand(_colliding_sweep())
    assert len(expansion.points) == 2
    check_steps(_docs(expansion.points[:1]))  # −2 against −1: disjoint
    with pytest.raises(ValidationError) as engine_side:
        check_steps(_docs(expansion.points))
    with pytest.raises(ValidationError) as checklist:
        validate_document(parse_document(expansion.points[1].raw))
    assert engine_side.value.rule == 8
    assert str(engine_side.value) == str(checklist.value)
    assert not isinstance(engine_side.value, ValidationErrors)  # one violation, once


def test_a_violation_at_a_combination_passes_validate_and_is_refused_per_step(
    env: ResolutionEnv,
) -> None:
    """The boundary: ``validate`` sees the representatives — (1, 9), (1, 3),
    (5, 9) — and every one is legal, so the document compiles and validates
    against an engine; the engine's per-step pass reaches (5, 3) and refuses
    it under rule 21, naming the read and both layers."""
    compiled = compile_protocol(
        _two_axis_reachability(), env=env, base_dir=None, overrides=None, engine=None
    )
    assert validate(compiled, "pytorch_hooks", env=env, data=False) is compiled
    assert len(compiled.representatives) == 3
    expansion = enumerate_steps(compiled)
    assert [dict(p.coords) for p in expansion.points] == [
        {"sites.src.layers": [1], "sites.dst.layers": [9]},
        {"sites.src.layers": [1], "sites.dst.layers": [3]},
        {"sites.src.layers": [5], "sites.dst.layers": [9]},
        {"sites.src.layers": [5], "sites.dst.layers": [3]},
    ]
    check_steps(_docs(expansion.points[:3]), env)  # the representatives
    with pytest.raises(ValidationError) as err:
        check_steps(_docs(expansion.points), env)
    assert err.value.rule == 21
    assert "'v_src'" in str(err.value)
    assert "layer 5" in str(err.value) and "layer 3" in str(err.value)


def test_a_swept_model_crossed_with_a_layer_it_lacks_is_refused_per_step(
    env: ResolutionEnv,
) -> None:
    """Not only the rules that read two entries: rule 4 reads a site against
    the model's static facts, and when the *model* is an axis the violation
    is a combination too. ``validate`` passes (every representative is
    legal); the per-step pass refuses (gpt2, 14) with the text the per-point
    compiler gave the whole document before the per-step pass existed."""
    compiled = compile_protocol(
        _model_by_layer(), env=env, base_dir=None, overrides=None, engine=None
    )
    assert validate(compiled, "pytorch_hooks", env=env, data=False) is compiled
    expansion = enumerate_steps(compiled)
    assert len(expansion.points) == 4 and len(compiled.representatives) == 3
    with pytest.raises(ValidationError) as err:
        check_steps(_docs(expansion.points), env)
    assert str(err.value) == (
        "[V4] at sites.tgt.layers site 'tgt': layer 14 out of range for the "
        "12-layer model 'gpt2'"
    )
    assert not isinstance(err.value, ValidationErrors)


def test_the_run_door_refuses_a_combination_before_anything_is_written(
    env: ResolutionEnv, tmp_path: Path
) -> None:
    """Through ``run_protocol``: the door's ``validate`` passes the two-axis
    document, the engine's per-step pass refuses it — with the checklist's
    text, before the receipt is written and before any forward."""
    with pytest.raises(ValidationError) as err:
        run_protocol(_two_axis_reachability(), env, _Stub(), tmp_path, record=True)
    assert err.value.rule == 21 and "'v_src'" in str(err.value)
    assert not (tmp_path / RUN_RECORD_NAME).exists()
    assert not list(tmp_path.iterdir())


def test_distinct_violations_across_steps_are_raised_together() -> None:
    """The same violation at two steps is one refusal; distinct violations
    across the steps are raised together as one ``ValidationErrors``, each
    once — the compiler's own aggregation."""
    doc = _colliding_sweep()
    doc["method"]["writes"]["patch2"]["pos"] = {"sweep": [-1, -1]}
    twice = expand(doc)
    with pytest.raises(ValidationError) as same:
        check_steps(_docs(twice.points))
    assert not isinstance(same.value, ValidationErrors)
    # a third absolute write at −1 collides with `patch` at every step, and at
    # the second step `patch2` collides with `patch` first (the rule reports
    # one pair per document): two distinct texts over two steps
    doc = _colliding_sweep()
    doc["method"]["writes"]["patch3"] = {
        "site": "tgt",
        "pos": -1,
        "do": {"swap": "v_cf"},
    }
    doc["method"]["intervened_models"]["patched"]["writes"].append("patch3")
    with pytest.raises(ValidationErrors) as distinct:
        check_steps(_docs(expand(doc).points))
    texts = [str(e) for e in distinct.value.errors]
    assert len(texts) == len(set(texts)) == 2
    assert {e.rule for e in distinct.value.errors} == {8}
