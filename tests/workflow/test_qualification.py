"""Qualify each point once, at the workflow layer (workflow spec §2.2 rule 15,
§4.3, §8). The identity a qualification is keyed to is
``RuntimeIdentity.tree_digest``, the same digest ``--resume`` compares, not a
numerical fingerprint. "Corpus NN" is the intervention specification
``tests/protocols/NN_*_im.json``.

A control qualifies its target's *points*, and a target's rank × seed fanout is
one step's expansion. So the control runs once for the whole fanout, its
certified status and its identity are inherited into every fit, and N points
failing qualification invalidate N points of the target — never N × fanout
fits attributed one by one.

What is pinned, and how each test fails without the change:

* **T9 — once.** A self-swap control, its certifier and a DAS fit fanned out
  over ``featurizers.rot.k`` × ``train.seed`` with **no authored ``after``**:
  the fit depends on the certifier through the edge the schedule derives, the
  stream has one ``phase_started`` for the control, the control's record
  ``forwards`` equals the planned interned forward groups and is the same for
  six fit points and for one, every fit point inherits ``passed`` and one
  ``controls.identity`` triple. Without the change the fit does not depend on
  the certifier, ``inherit`` sees no ancestor, and the fit's record carries no
  ``controls`` block; no record and no stream line carries ``forwards``.
* **T10 — N, not N × fanout.** One control swept over two layers, one of them
  vacuous (its receiver read at ``block_input`` of the layer above the write,
  which a swap at the last layer cannot change): the certifier says
  ``n_failed: 1`` of ``n_points: 2`` and the fit's record ``n_invalid: 6`` of
  ``n_points: 12`` — every invalid point at that layer, none elsewhere. A
  runner attributing invalidity per fit without coordinates would say 12.
* **T11 — one identity.** A control at another ``model.dtype``,
  ``model.revision`` or ``model.attn_implementation`` than its target is refused
  under rule 15 naming the field — the comparison is the whole
  ``canonical_model_ref`` dict, so a field added to ``canonical_model`` (such
  as the attention backend) is checked without being re-listed here, and a
  backend authored on one side only is refused naming the omission; without the change
  the bf16/fp32 pair loads (that is the defect the shipped fixture had). At run
  time a control record from another ``tree_digest`` is
  re-run by ``--resume``, never reused, and the dependent's inherited identity
  follows; a doctored ``forwards`` alone still reuses — recorded, not compared.
* **T12 — legitimate campaigns.** One rank, one seed qualifies once with the
  same forward count and the same identity; the bf16/bf16 pair loads with no
  waiver beyond rule 14's; changing the dtype on **both** steps loads and is
  re-qualified with new document digests and new records; the shipped
  workflows load to their pins; a ``random_mask`` chain, whose control draws
  from the fit's bundle, keeps its post-hoc direction.
"""

from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path
from typing import Any

import pytest

from causalab.io.events import EVENTS_FILE, read_events
from causalab.io.step_record import SIDECAR
from causalab.provenance import runtime_identity
from causalab.workflow import runner
from causalab.workflow.document import (
    CONTROL_KINDS,
    MAX_RULE,
    POST_HOC_CONTROL_KINDS,
    QUALIFICATION_RULE,
    WorkflowError,
    load_workflow,
)
from causalab.workflow.runner import (
    INSTRUMENT_FAILURE,
    QUALIFICATION_IDENTITY_FIELDS,
    run_workflow,
)

from tests.workflow.test_closure_census import SHIPPED
from tests.workflow.test_controls import (
    CONTROL_13,
    EXTERNAL,
    FIT,
    FIXTURES,
    PROTOCOLS,
    TINY_LLAMA,
    TWIN,
    _certifier,  # pyright: ignore[reportPrivateUsage]
    _protocol,  # pyright: ignore[reportPrivateUsage]
    INTERNALS,
    _section,  # pyright: ignore[reportPrivateUsage]
    _shuffled_workflow,  # pyright: ignore[reportPrivateUsage]
    _workflow,  # pyright: ignore[reportPrivateUsage]
)
from tests.workflow.test_controls_run import (
    _engine,  # pyright: ignore[reportPrivateUsage]
    _record,  # pyright: ignore[reportPrivateUsage]
    _t6_workflow,  # pyright: ignore[reportPrivateUsage]
    _tiny_env,  # pyright: ignore[reportPrivateUsage]
)

#: tiny-random on CPU; corpus 04 declares bf16, so the realization is set here
TINY = {"model.key": TINY_LLAMA, "model.dtype": "fp32"}
#: the smallest fit that trains: one epoch over pairs of two
QUICK_FIT = {"train.steps": {"epochs": 1}, "train.batch": {"pairs": 2}}


# --------------------------------------------------------------------------- #
# builders
# --------------------------------------------------------------------------- #


def _axis(values: list[int]) -> Any:
    return values[0] if len(values) == 1 else {"sweep": values}


def _qualified_fit(
    *,
    layer: Any,
    ks: list[int],
    seeds: list[int],
    control: Path = TWIN,
    bound: float | None = None,
) -> dict[str, Any]:
    """A self-swap control of a DAS fit (corpus 04) fanned out over
    ``featurizers.rot.k`` × ``train.seed``, certified by
    ``causalab.analysis.certify_control``. **No step authors ``after``**, and
    the fit is declared first: whatever order the schedule finds is derived."""
    escape = {} if bound is None else {"stop_after_failure_rate": bound}
    return _workflow(
        {
            "fit": _protocol(
                FIT,
                set={
                    **TINY,
                    **QUICK_FIT,
                    "sites.target.layers": layer,
                    "featurizers.rot.k": _axis(ks),
                    "train.seed": _axis(seeds),
                },
                waive={"matched_random": EXTERNAL},
            ),
            "ctl": _protocol(
                control,
                set={"model.key": TINY_LLAMA, "sites.target.layers": layer},
                control={"of": "fit", "kind": "self_swap", "seam": "A"},
                **escape,
            ),
            "cert": _certifier("ctl"),
        }
    )


def _pair(fit_set: dict[str, Any], control_set: dict[str, Any]) -> dict[str, Any]:
    """The load-only pair on the corpus documents (``Qwen/Qwen3-8B``): corpus
    04's fit (bf16) and the self-swap twin (no dtype authored — fp32)."""
    return _workflow(
        {
            "fit": _protocol(FIT, set=fit_set, waive={"matched_random": EXTERNAL}),
            "ctl": _protocol(
                TWIN, set=control_set, control={"of": "fit", "kind": "self_swap"}
            ),
            "cert": _certifier("ctl"),
        }
    )


def _random_mask_chain() -> dict[str, Any]:
    """``fit → random_mask → apply``: the control loads what ``draw`` wrote,
    ``draw`` reads the fit's bundle — a route of data hops, so the control is
    post-hoc by construction (the shape T8 declares in ``test_controls``)."""
    return {
        "fit": _protocol(PROTOCOLS / "dbm.json", waive={"self_swap": EXTERNAL}),
        "draw": {
            "type": "script",
            "script": {"module": "causalab.analysis.random_mask"},
            "inputs": {"gate": {"step": "fit", "file": "gate.safetensors"}, "seed": 7},
            "outputs": {"gate": "gate.safetensors"},
        },
        "apply_control": _protocol(
            PROTOCOLS / "dbm_apply.json",
            set={"featurizers.gate.file_path": "draw/gate.safetensors"},
            control={
                "of": "fit",
                "kind": "matched_random",
                "seeds": [7],
                "min_draws": 1,
            },
        ),
    }


def _self_contained_matched_random(
    *, seeds: list[int], after: list[str] | None = None
) -> dict[str, Any]:
    """Corpus 13 as a **self-contained** matched_random control of corpus 04's
    fit on the tiny model: its featurizer draws its own ``seed`` sweep and it
    references nothing of the fit — the derived edge is all that orders it
    before its target. ``after`` is the authored ordering under test."""
    control = _protocol(
        CONTROL_13,
        set={
            **TINY,
            "sites.target.layers": 0,
            "featurizers.rot.seed": {"sweep": seeds},
        },
        control={
            "of": "fit",
            "kind": "matched_random",
            "seeds": seeds,
            "min_draws": len(seeds),
        },
    )
    if after is not None:
        control["after"] = after
    return _workflow(
        {
            "fit": _protocol(
                FIT,
                set={**TINY, **QUICK_FIT, "sites.target.layers": 0},
                waive={"self_swap": EXTERNAL},
            ),
            "ctl": control,
        }
    )


def _swept_twin(tmp_path: Path) -> Path:
    """The twin with its three receiver reads at ``block_input`` of layer 1 —
    the residual entering the last layer. A swap at layer 0 changes what it
    reads; a swap at layer 1 cannot, so the same control swept over both
    layers has one valid point and one vacuous one (leg iii fails there)."""
    raw = json.loads(TWIN.read_text())
    method = raw["method"]
    method["sites"]["receiver"] = {"component": "block_input", "layers": 1}
    del method["sites"]["lm_head"]
    for name in ("recv_target", "recv_control", "recv_original"):
        method["reads"][name]["site"] = "receiver"
    document = tmp_path / "swept_twin.json"
    document.write_text(json.dumps(raw))
    return document


def _planned_forwards(loaded: Any, step: str) -> int:
    """What the control's campaign *owes* (IM spec §3): its interned forward
    groups, the number ``RunResult.forwards`` reports it paid."""
    from causalab.neural.shared.execution import campaign_plans
    from causalab.neural.shared.plan import interned_groups

    inner = loaded.inner[step]
    return len(
        interned_groups(campaign_plans(inner.point_documents, inner.canonical_points))
    )


def _statuses(result: Any) -> dict[str, str]:
    return {name: entry["status"] for name, entry in result.manifest["steps"].items()}


def _refused_15(
    raw: dict[str, Any], env: Any, match: str, **kwargs: Any
) -> WorkflowError:
    with pytest.raises(WorkflowError) as info:
        load_workflow(raw, env, **kwargs)
    assert info.value.rule == QUALIFICATION_RULE, str(info.value)
    assert re.search(match, str(info.value)), str(info.value)
    return info.value


def _stand_in(monkeypatch: pytest.MonkeyPatch, tree_digest: str) -> None:
    """Make the runner see another package (``runner.py`` imports
    ``runtime_identity`` as a module attribute for exactly this)."""
    other = dataclasses.replace(runtime_identity(), tree_digest=tree_digest)
    monkeypatch.setattr(runner, "runtime_identity", lambda: other)


# --------------------------------------------------------------------------- #
# the rule, the spec, the derived edge
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_rule_15_is_the_qualification_rule() -> None:
    """Numbered by rule, so a renumbering is a deliberate edit here."""
    assert MAX_RULE >= QUALIFICATION_RULE == 15
    section = _section("## 5. Validation")
    item = re.search(r"^15\. (.+?)(?=^\d+\. |\Z)", section, re.M | re.S)
    assert item is not None
    text = item.group(1)
    assert "before its target" in text and "realization" in text
    for field in ("`key`", "`revision`", "`dtype`", "`quantization`"):
        assert field in text, field
    assert "re-qualified" in text


@pytest.mark.unit
def test_the_spec_names_forwards_and_the_identity_triple() -> None:
    """§4.3's ``phase_completed`` row and §8's stamping row carry ``forwards``;
    §8's controls paragraph names the identity triple; the ``resume`` row is
    unchanged — ``forwards`` is recorded, never compared."""
    four = _section("### 4.3 Event stream")
    (row,) = [
        line for line in four.splitlines() if line.startswith("| `phase_completed` |")
    ]
    assert "`forwards`" in row.split("|")[3]
    eight = _section("## 8. Runner contract", INTERNALS)
    (stamping,) = [
        line for line in eight.splitlines() if line.startswith("| stamping |")
    ]
    assert "`forwards`" in stamping and "never compared" in stamping
    (resume,) = [line for line in eight.splitlines() if line.startswith("| resume |")]
    assert "`implementation.tree_digest`" in resume and "forwards" not in resume
    controls = next(
        paragraph
        for paragraph in eight.split("\n\n")
        if "qualification `identity`" in paragraph
    )
    assert "`identity`" in controls
    for field in QUALIFICATION_IDENTITY_FIELDS:
        assert field in controls, field
    assert QUALIFICATION_IDENTITY_FIELDS == ("document_digest", "tree_digest", "engine")


@pytest.mark.unit
def test_the_edge_is_derived_and_an_authored_after_adds_nothing(env) -> None:
    """The fit depends on the certifier with no ``after`` written; writing one
    changes the schedule not at all and the digest only because ``after`` is
    canonical — the derived edge never is (§7)."""
    derived = load_workflow(_pair({}, {"model.dtype": "bf16"}), env)
    assert derived.dependencies == {"cert": ("ctl",), "ctl": (), "fit": ("cert",)}
    assert derived.order == ("ctl", "cert", "fit")
    assert derived.levels == (("ctl",), ("cert",), ("fit",))
    assert "after" not in derived.canonical["steps"]["fit"]
    raw = _pair({}, {"model.dtype": "bf16"})
    raw["steps"]["fit"]["after"] = ["cert"]
    authored = load_workflow(raw, env)
    assert authored.dependencies == derived.dependencies
    assert authored.order == derived.order
    assert authored.digest != derived.digest
    assert authored.inner_digests == derived.inner_digests


@pytest.mark.unit
def test_t6s_hand_authored_after_is_now_redundant(env) -> None:
    """The T6 fixture once wrote ``after: [cert_a, cert_b]`` by hand; the same
    workflow without it has the same schedule."""
    env = _tiny_env(env)
    raw = _t6_workflow(fail_b=True, bound=1.0)
    with_after = load_workflow(raw, env)
    del raw["steps"]["dep"]["after"]
    without = load_workflow(raw, env)
    assert without.dependencies == with_after.dependencies
    assert without.dependencies["dep"] == ("cert_a", "cert_b")
    assert without.order == with_after.order


@pytest.mark.unit
def test_a_post_hoc_control_keeps_its_direction(env) -> None:
    """The ``fit → random_mask → apply`` chain: the control draws from the
    fit's bundle, so it depends on its target and no edge is added — the fit
    runs first, exactly as T8 in ``test_controls.py`` has it."""
    steps = {
        "fit": _protocol(PROTOCOLS / "dbm.json", waive={"self_swap": EXTERNAL}),
        "draw": {
            "type": "script",
            "script": {"module": "causalab.analysis.random_mask"},
            "inputs": {"gate": {"step": "fit", "file": "gate.safetensors"}, "seed": 7},
            "outputs": {"gate": "gate.safetensors"},
        },
        "apply_control": _protocol(
            PROTOCOLS / "dbm_apply.json",
            set={"featurizers.gate.file_path": "draw/gate.safetensors"},
            control={
                "of": "fit",
                "kind": "matched_random",
                "seeds": [7],
                "min_draws": 1,
            },
        ),
    }
    loaded = load_workflow(_workflow(steps), env)
    assert loaded.dependencies["fit"] == ()
    assert loaded.order == ("fit", "draw", "apply_control")


@pytest.mark.unit
def test_the_post_hoc_kinds_are_matched_random_only() -> None:
    """The kind boundary of the post-hoc skip is closed: ``matched_random``
    (the ``random_mask`` chain draws from the fit's bundle) and nothing else —
    every other kind is certifiable and has no post-hoc direction."""
    assert POST_HOC_CONTROL_KINDS == ("matched_random",)
    assert set(POST_HOC_CONTROL_KINDS) < set(CONTROL_KINDS)


_AUTHORED_AFTER = (
    r"control 'ctl' \(kind 'self_swap'\) is authored to run after its target 'fit'"
)


@pytest.mark.unit
def test_a_certifiable_control_authored_after_its_target_is_refused_under_rule_15(
    env,
) -> None:
    """A fail-open, now closed: ``after: ["fit"]`` on a
    self-swap control let the authored edge win — ``inherit`` found no
    ancestor, the fit's record got no ``controls`` block and nothing refused.
    Now refused under rule 15 naming the authored route and its last hop."""
    raw = _pair({}, {"model.dtype": "bf16"})
    raw["steps"]["ctl"]["after"] = ["fit"]
    err = _refused_15(raw, env, _AUTHORED_AFTER)
    assert "'cert' -> 'ctl' -> 'fit'" in str(err)
    assert "an 'after' entry" in str(err)
    assert err.path == "steps.ctl.control"


@pytest.mark.unit
def test_a_certifier_authored_after_the_target_is_refused_the_same_way(env) -> None:
    """The same through the certifier: the head of the derived edge is the
    certifying step, so its ``after: ["fit"]`` is the control running after
    its target — route ``'cert' -> 'fit'``, refused at the control."""
    raw = _pair({}, {"model.dtype": "bf16"})
    raw["steps"]["cert"]["after"] = ["fit"]
    err = _refused_15(raw, env, _AUTHORED_AFTER)
    assert "'cert' -> 'fit'" in str(err) and "'ctl' ->" not in str(err)
    assert "an 'after' entry" in str(err)
    assert err.path == "steps.ctl.control"


@pytest.mark.unit
def test_a_shuffled_source_control_authored_after_its_target_is_refused(env) -> None:
    """The other certifiable kind (no certifier: the control is the head)."""
    raw = _shuffled_workflow()
    raw["steps"]["shuffled"]["after"] = ["target"]
    err = _refused_15(
        raw,
        env,
        r"control 'shuffled' \(kind 'shuffled_source'\) is authored to run after "
        r"its target 'target'",
    )
    assert "'shuffled' -> 'target'" in str(err)
    assert err.path == "steps.shuffled.control"


_ORDERING_ALONE = (
    "ordering alone does not make a control post-hoc; a post-hoc "
    "'matched_random' reads its target's bundle — remove the 'after' entry or "
    "draw from 'fit'"
)


@pytest.mark.unit
def test_a_self_contained_matched_random_authored_after_its_target_is_refused(
    env,
) -> None:
    """A second fail-open, now closed: the post-hoc
    skip read ``_reaches`` over *every* authored edge, and ``after`` feeds the
    same graph as a reference does — so a self-contained ``matched_random``
    (its own featurizer seeds, nothing read from the fit) that also authored
    ``after: ["fit"]`` took the skip: no derived edge, the fit's record no
    ``controls`` block, nothing refused. Post-hoc means a data hop; a bare
    ordering is refused naming the route and what to do instead."""
    raw = json.loads((FIXTURES / "matched_random.json").read_text())
    raw["steps"]["control"]["after"] = ["fit"]
    err = _refused_15(
        raw,
        env,
        r"control 'control' \(kind 'matched_random'\) is authored to run after "
        r"its target 'fit'",
        workflow_dir=FIXTURES,
    )
    assert "'control' -> 'fit'" in str(err)
    assert "the last hop an 'after' entry" in str(err)
    assert _ORDERING_ALONE in str(err)
    assert err.path == "steps.control.control"


@pytest.mark.unit
def test_the_same_control_without_after_loads_with_the_derived_edge(env) -> None:
    """The valid-work twin: the fixture as shipped — the fit depends on the
    control through the derived edge, the control on nothing."""
    loaded = load_workflow(FIXTURES / "matched_random.json", env)
    assert loaded.dependencies == {"fit": ("control",), "control": ()}
    assert loaded.order == ("control", "fit")


@pytest.mark.unit
def test_a_post_hoc_chain_keeps_its_direction_with_an_after_too(env) -> None:
    """Twin 2: a genuinely post-hoc ``matched_random`` — the ``random_mask``
    chain, whose route to the fit is a run-tree load and a script input
    reference — takes the skip with an ``after`` on the control or on the
    drawing step as without one; the redundant ordering adds nothing."""
    plain = load_workflow(_workflow(_random_mask_chain()), env)
    assert plain.dependencies["fit"] == ()
    assert plain.order == ("fit", "draw", "apply_control")
    on_control = _random_mask_chain()
    on_control["apply_control"]["after"] = ["fit"]
    loaded = load_workflow(_workflow(on_control), env)
    assert loaded.dependencies["fit"] == ()
    assert loaded.dependencies["apply_control"] == ("draw", "fit")
    assert loaded.order == plain.order
    on_draw = _random_mask_chain()
    on_draw["draw"]["after"] = ["fit"]
    loaded = load_workflow(_workflow(on_draw), env)
    assert loaded.dependencies == plain.dependencies
    assert loaded.order == plain.order


@pytest.mark.unit
def test_the_last_hop_names_both_an_after_entry_and_a_reference(env) -> None:
    """A certifier authoring ``after: ["fit"]`` *and* an input reading the
    fit's ``iia.json`` is refused naming both hops; with the reference alone
    the last hop is ``a reference``."""
    raw = _pair({}, {"model.dtype": "bf16"})
    raw["steps"]["cert"]["after"] = ["fit"]
    raw["steps"]["cert"]["inputs"]["fit_iia"] = {"step": "fit", "file": "iia.json"}
    err = _refused_15(raw, env, _AUTHORED_AFTER)
    assert "'cert' -> 'fit', the last hop both an 'after' entry and a reference" in (
        str(err)
    )
    assert err.path == "steps.ctl.control"
    reference = _pair({}, {"model.dtype": "bf16"})
    reference["steps"]["cert"]["inputs"]["fit_iia"] = {
        "step": "fit",
        "file": "iia.json",
    }
    err = _refused_15(reference, env, _AUTHORED_AFTER)
    assert "'cert' -> 'fit', the last hop a reference;" in str(err)


@pytest.mark.unit
def test_two_steps_that_would_each_qualify_the_other_are_refused_naming_both(
    env,
) -> None:
    raw = _workflow(
        {
            "a": _protocol(
                TWIN,
                set={"model.dtype": "bf16"},
                control={"of": "b", "kind": "self_swap"},
            ),
            "cert_a": _certifier("a"),
            "b": _protocol(
                TWIN,
                set={"model.dtype": "bf16"},
                control={"of": "a", "kind": "self_swap"},
            ),
            "cert_b": _certifier("b"),
        }
    )
    err = _refused_15(raw, env, r"control 'b' runs before its target 'a', but 'a'")
    assert "each qualify the other" in str(err) and err.path == "steps.b.control"


# --------------------------------------------------------------------------- #
# T11 (load) and T12 (load) — one realization per control/target pair
# --------------------------------------------------------------------------- #


@pytest.mark.unit
@pytest.mark.parametrize(
    "fit_set, control_set, field, mine, theirs",
    [
        ({}, {}, "dtype", "fp32", "bf16"),
        (
            {},
            {"model.dtype": "bf16", "model.revision": "v2"},
            "revision",
            "v2",
            "main",
        ),
        (
            {"model.attn_implementation": "sdpa"},
            {"model.dtype": "bf16", "model.attn_implementation": "eager"},
            "attn_implementation",
            "eager",
            "sdpa",
        ),
    ],
    ids=["dtype", "revision", "attn_implementation"],
)
def test_t11_a_control_at_another_realization_is_refused_naming_the_field(
    env, fit_set, control_set, field, mine, theirs
) -> None:
    """Corpus 04 is bf16; the twin authors no dtype (fp32). The refusal names
    the field and both values. The attention backend is a realization field
    (``canonical_model`` hashes an authored one), and it is checked
    without being listed anywhere in ``document.py``. The mutation — re-list
    only the four other fields — loads the ``attn_implementation`` case; compare
    ``model.key`` alone and the dtype case loads too."""
    err = _refused_15(
        _pair(fit_set, control_set),
        env,
        rf"control 'ctl' runs the model at model\.{field} = {mine!r}; its target "
        rf"'fit' runs it at {theirs!r}",
    )
    assert err.path == "steps.ctl.control"
    assert "not authored" not in str(err)  # both sides author the field named


@pytest.mark.unit
def test_t11_a_backend_authored_on_one_side_only_is_refused_naming_the_omission(
    env,
) -> None:
    """``canonical_model`` preserves the omission of ``attn_implementation``
    (the historical default is the engine's), so a control that authors none
    and a target at ``eager`` are two model identities with two digests: refused
    under rule 15 with ``None`` on the control's side *and* a sentence saying
    which side left it unauthored. Same mutation as the parametrized case."""
    err = _refused_15(
        _pair({"model.attn_implementation": "eager"}, {"model.dtype": "bf16"}),
        env,
        r"control 'ctl' runs the model at model\.attn_implementation = None; its "
        r"target 'fit' runs it at 'eager'",
    )
    assert err.path == "steps.ctl.control"
    assert (
        "model.attn_implementation is authored as 'eager' on its target 'fit' and "
        "not authored on control 'ctl' — an omitted model.attn_implementation is "
        "the engine's default, which is a different model identity, so author it "
        "on both or on neither"
    ) in str(err)
    # and the mirror: authored on the control, omitted on the target
    err = _refused_15(
        _pair({}, {"model.dtype": "bf16", "model.attn_implementation": "eager"}),
        env,
        r"control 'ctl' runs the model at model\.attn_implementation = 'eager'; its "
        r"target 'fit' runs it at None",
    )
    assert (
        "authored as 'eager' on control 'ctl' and not authored on its target 'fit'"
        in str(err)
    )


@pytest.mark.unit
def test_t12_the_same_backend_on_both_steps_loads_and_is_re_qualified(env) -> None:
    """Fail-closed twin of the backend refusal: both steps authoring
    ``sdpa`` load with no waiver beyond rule 14's, and both inner digests move
    against the unauthored bf16 pair — a re-qualification, not a refusal."""
    plain = load_workflow(_pair({}, {"model.dtype": "bf16"}), env)
    sdpa = load_workflow(
        _pair(
            {"model.attn_implementation": "sdpa"},
            {"model.dtype": "bf16", "model.attn_implementation": "sdpa"},
        ),
        env,
    )
    assert "waive" not in sdpa.canonical["steps"]["ctl"]
    assert sdpa.canonical["steps"]["fit"]["waive"] == {"matched_random": EXTERNAL}
    assert sdpa.inner_digests["fit"] != plain.inner_digests["fit"]
    assert sdpa.inner_digests["ctl"] != plain.inner_digests["ctl"]
    assert sdpa.digest != plain.digest


@pytest.mark.unit
def test_t12_one_realization_loads_and_a_dtype_change_on_both_steps_re_qualifies(
    env,
) -> None:
    """The bf16/bf16 pair loads with no waiver beyond rule 14's; explicit
    fp32 on the control equals the default; setting fp32 on **both** steps
    loads and moves both inner digests — a re-qualification, not a refusal."""
    bf16 = load_workflow(_pair({}, {"model.dtype": "bf16"}), env)
    assert "waive" not in bf16.canonical["steps"]["ctl"]
    assert bf16.canonical["steps"]["fit"]["waive"] == {"matched_random": EXTERNAL}
    fp32 = load_workflow(_pair({"model.dtype": "fp32"}, {"model.dtype": "fp32"}), env)
    default = load_workflow(_pair({"model.dtype": "fp32"}, {}), env)
    assert default.inner_digests["ctl"] == fp32.inner_digests["ctl"]
    assert fp32.inner_digests["fit"] != bf16.inner_digests["fit"]
    assert fp32.inner_digests["ctl"] != bf16.inner_digests["ctl"]
    assert fp32.digest != bf16.digest
    _refused_15(
        _pair({"model.dtype": "fp32"}, {"model.dtype": "bf16"}), env, r"model\.dtype"
    )


@pytest.mark.unit
def test_the_shipped_workflows_declare_no_control(env) -> None:
    """Fail-closed: no shipped workflow declares a control, so rule 15 and the
    derived edge touch none of them."""
    assert {path.name for path in SHIPPED} == {
        "mean_ablation.json",
        "pca_basis.json",
        "weekdays.json",
    }
    for path in SHIPPED:
        loaded = load_workflow(path, env)
        assert all("control" not in e for e in loaded.canonical["steps"].values())


# --------------------------------------------------------------------------- #
# T9, T10, T11 (run), T12 (run) — on the tiny fixture
# --------------------------------------------------------------------------- #


@pytest.mark.smoke
def test_t9_a_control_qualifies_once_for_its_targets_whole_fanout(
    env, tmp_path
) -> None:
    """Six fits (k × seed) behind one control, no ``after`` authored: one
    ``phase_started`` for the control, ``forwards`` equal to its planned
    interned groups and the same for a one-point fit, every fit point
    ``passed`` under one identity triple. Mutation: drop the derived edge and
    the fit's record has no ``controls`` block."""
    env = _tiny_env(env)
    loaded = load_workflow(
        _qualified_fit(layer=0, ks=[2, 4], seeds=[0, 1, 2]), env, workflow_dir=tmp_path
    )
    assert loaded.dependencies == {"cert": ("ctl",), "ctl": (), "fit": ("cert",)}
    assert loaded.order == ("ctl", "cert", "fit")
    assert len(loaded.inner["fit"].point_digests) == 6
    assert len(loaded.inner["ctl"].point_digests) == 1
    result = run_workflow(loaded, env, tmp_path / "runs", _engine())
    assert _statuses(result) == {name: "completed" for name in loaded.order}

    records = read_events(result.run_root / EVENTS_FILE)
    started = [r["payload"]["step"] for r in records if r["event"] == "phase_started"]
    assert started == ["ctl", "cert", "fit"]
    completed = {
        r["payload"]["step"]: r["payload"]
        for r in records
        if r["event"] == "phase_completed"
    }
    ctl = _record(result.run_root, "ctl")
    planned = _planned_forwards(loaded, "ctl")
    assert planned > 0
    assert ctl["forwards"] == planned
    assert completed["ctl"] == {
        "step": "ctl",
        "status": "completed",
        "forwards": planned,
    }
    assert "forwards" not in completed["cert"]  # a script step ran no forward
    fit = _record(result.run_root, "fit")
    assert isinstance(fit["forwards"], int) and fit["forwards"] > 0
    assert completed["fit"]["forwards"] == fit["forwards"]

    block = fit["controls"]
    assert block["inherited_from"] == ["ctl"]
    assert (block["n_invalid"], block["n_points"]) == (0, 6)
    assert set(block["by_point"]) == set(fit["point_digests"])
    for entry in block["by_point"].values():
        assert entry["controls"] == {"ctl": "passed"} and entry["status"] == "passed"
    identity = block["identity"]
    assert list(identity) == ["ctl"]
    assert tuple(identity["ctl"]) == QUALIFICATION_IDENTITY_FIELDS
    assert identity["ctl"] == {
        "document_digest": ctl["document_digest"],
        "tree_digest": fit["implementation"]["tree_digest"],
        "engine": "pytorch_hooks",
    }
    assert identity["ctl"]["tree_digest"] == ctl["implementation"]["tree_digest"]
    # nothing the run merely observed is in the key
    assert not set(identity["ctl"]) & set(ctl["execution"])

    # one rank, one seed (T12): the same control, the same count, the same identity
    one = load_workflow(
        _qualified_fit(layer=0, ks=[2], seeds=[0]), env, workflow_dir=tmp_path
    )
    assert len(one.inner["fit"].point_digests) == 1
    assert one.inner_digests["ctl"] == loaded.inner_digests["ctl"]
    second = run_workflow(one, env, tmp_path / "runs_one", _engine())
    assert _statuses(second) == {name: "completed" for name in one.order}
    assert _record(second.run_root, "ctl")["forwards"] == planned
    fit_one = _record(second.run_root, "fit")["controls"]
    assert (fit_one["n_invalid"], fit_one["n_points"]) == (0, 1)
    assert fit_one["identity"] == identity


@pytest.mark.smoke
def test_t10_n_failing_points_invalidate_n_points_not_n_times_the_fanout(
    env, tmp_path
) -> None:
    """One control swept over layers 0 and 1, vacuous at 1; the fit sweeps
    layer × k × seed (2 × 2 × 3). The certifier counts 1 of 2; the fit counts
    6 of 12, every invalid point at layer 1. Mutation: attribute invalidity
    without coordinates and all 12 are invalid."""
    env = _tiny_env(env)
    loaded = load_workflow(
        _qualified_fit(
            layer={"sweep": [0, 1]},
            ks=[2, 4],
            seeds=[0, 1, 2],
            control=_swept_twin(tmp_path),
            bound=1.0,
        ),
        env,
        workflow_dir=tmp_path,
    )
    assert len(loaded.inner["ctl"].point_digests) == 2
    assert len(loaded.inner["fit"].point_digests) == 12
    result = run_workflow(loaded, env, tmp_path / "runs", _engine())
    assert _statuses(result) == {name: "completed" for name in loaded.order}

    cert = _record(result.run_root, "cert")["certifies"]
    assert (cert["n_failed"], cert["n_points"]) == (1, 2)
    assert {
        e["coords"]["sites.target.layers"]: e["status"]
        for e in cert["by_point"].values()
    } == {0: "passed", 1: "failed"}
    ctl = _record(result.run_root, "ctl")
    assert ctl["forwards"] == _planned_forwards(loaded, "ctl")

    fit = _record(result.run_root, "fit")
    block = fit["controls"]
    assert (block["n_invalid"], block["n_points"]) == (6, 12)
    assert set(block["by_point"]) == set(fit["point_digests"])
    invalid = [
        e for e in block["by_point"].values() if e["status"] == "instrument_invalid"
    ]
    passed = [e for e in block["by_point"].values() if e["status"] == "passed"]
    assert len(invalid) == 6 and len(passed) == 6
    assert {e["coords"]["sites.target.layers"] for e in invalid} == {1}
    assert {e["coords"]["sites.target.layers"] for e in passed} == {0}
    fanout = {(k, seed) for k in (2, 4) for seed in (0, 1, 2)}
    assert {
        (e["coords"]["featurizers.rot.k"], e["coords"]["train.seed"]) for e in invalid
    } == fanout
    assert all(e["controls"] == {"ctl": "failed"} for e in invalid)
    assert all(e["controls"] == {"ctl": "passed"} for e in passed)
    assert list(block["identity"]) == ["ctl"]
    assert block["identity"]["ctl"]["document_digest"] == ctl["document_digest"]
    # the count of qualification failures is the certifier's (one point), not the fanout's (six)
    warnings = [
        r for r in read_events(result.run_root / EVENTS_FILE) if r["event"] == "warning"
    ]
    assert [r["payload"]["reason"] for r in warnings] == [INSTRUMENT_FAILURE]
    assert warnings[0]["payload"]["coords"] == {"sites.target.layers": 1}


@pytest.mark.smoke
def test_t11_a_qualification_from_another_tree_is_re_run_and_forwards_is_never_compared(
    env, tmp_path, monkeypatch
) -> None:
    """Run once. A doctored ``forwards`` alone still reuses every step
    (recorded, never compared). Under a stood-in ``tree_digest`` the control
    is re-run — ``completed``, not ``reused`` — and the fit's inherited
    identity carries the new digest. Then a fit that changes alone runs again
    under reused control and certifier, and inherits the identity restored
    from their records."""
    env = _tiny_env(env)
    loaded = load_workflow(
        _qualified_fit(layer=0, ks=[2], seeds=[0]), env, workflow_dir=tmp_path
    )
    out = tmp_path / "runs"
    first = run_workflow(loaded, env, out, _engine())
    root = first.run_root
    assert _statuses(first) == {name: "completed" for name in loaded.order}
    real = _record(root, "ctl")["implementation"]["tree_digest"]
    planned = _planned_forwards(loaded, "ctl")
    assert _record(root, "ctl")["forwards"] == planned
    assert _record(root, "fit")["controls"]["identity"]["ctl"]["tree_digest"] == real

    record_path = root / "ctl" / SIDECAR
    doctored = json.loads(record_path.read_text())
    doctored["forwards"] = 999_999
    record_path.write_text(json.dumps(doctored))
    reused = run_workflow(loaded, env, out, _engine(), resume=True)
    assert _statuses(reused) == {name: "reused" for name in loaded.order}
    assert _record(root, "ctl")["forwards"] == 999_999
    completed = {
        r["payload"]["step"]: r["payload"]
        for r in read_events(root / EVENTS_FILE)
        if r["event"] == "phase_completed"
    }
    assert completed["ctl"] == {"step": "ctl", "status": "reused", "forwards": 999_999}

    _stand_in(monkeypatch, "0" * 64)
    third = run_workflow(loaded, env, out, _engine(), resume=True)
    assert _statuses(third) == {name: "completed" for name in loaded.order}
    ctl = _record(root, "ctl")
    assert ctl["implementation"]["tree_digest"] == "0" * 64
    assert ctl["forwards"] == planned
    fit = _record(root, "fit")
    assert fit["controls"]["identity"]["ctl"] == {
        "document_digest": ctl["document_digest"],
        "tree_digest": "0" * 64,
        "engine": "pytorch_hooks",
    }

    # the fit alone changes (k 4): control and certifier reused, the fit runs
    # again and inherits the identity the ledger restored from their records
    changed = load_workflow(
        _qualified_fit(layer=0, ks=[4], seeds=[0]), env, workflow_dir=tmp_path
    )
    assert changed.inner_digests["ctl"] == loaded.inner_digests["ctl"]
    fourth = run_workflow(changed, env, out, _engine(), resume=True)
    assert _statuses(fourth) == {"ctl": "reused", "cert": "reused", "fit": "completed"}
    fit = _record(root, "fit")
    assert fit["controls"]["identity"]["ctl"] == {
        "document_digest": ctl["document_digest"],
        "tree_digest": "0" * 64,
        "engine": "pytorch_hooks",
    }
    assert [e["controls"] for e in fit["controls"]["by_point"].values()] == [
        {"ctl": "passed"}
    ]


@pytest.mark.smoke
def test_a_control_record_under_another_engine_is_re_run_by_resume(
    env, tmp_path
) -> None:
    """``engine`` is the third member of the identity triple, and ``--resume``
    compares it (§8): a control record rewritten to another engine's name, or
    with the key removed, is re-run — ``completed``, not ``reused`` — and the
    fresh record names the engine that ran; the untouched records are reused,
    and so is everything once the records are intact again (valid work)."""
    env = _tiny_env(env)
    loaded = load_workflow(
        _qualified_fit(layer=0, ks=[2], seeds=[0]), env, workflow_dir=tmp_path
    )
    out = tmp_path / "runs"
    first = run_workflow(loaded, env, out, _engine())
    root = first.run_root
    assert _statuses(first) == {name: "completed" for name in loaded.order}
    assert _record(root, "ctl")["engine"] == "pytorch_hooks"
    record_path = root / "ctl" / SIDECAR
    for doctoring in ("rewrite", "remove"):
        record = json.loads(record_path.read_text())
        if doctoring == "rewrite":
            record["engine"] = "another_engine"
        else:
            del record["engine"]
        record_path.write_text(json.dumps(record))
        again = run_workflow(loaded, env, out, _engine(), resume=True)
        assert _statuses(again) == {
            "ctl": "completed",
            "cert": "reused",
            "fit": "reused",
        }, doctoring
        assert _record(root, "ctl")["engine"] == "pytorch_hooks"
    untouched = run_workflow(loaded, env, out, _engine(), resume=True)
    assert _statuses(untouched) == {name: "reused" for name in loaded.order}


@pytest.mark.smoke
def test_a_host_with_no_engine_for_the_step_reuses_its_record_under_resume(
    env, tmp_path
) -> None:
    """A fail-closed, now fixed: with no
    configured engine covering the step (``engines=[]`` — under ``auto`` the
    host's install changed) ``_engine_for`` is ``None``, and an earlier runner
    read that as "never reuse", re-running a content-digest-verified record on
    a host that could not run it. Now nothing is compared and the record is reused:
    a ``--resume`` with no engine reuses every step, a control record carrying
    another engine's name too; a record **without** the key is still not
    reused, and under a routed name a mismatch still is not (the test above)."""
    env = _tiny_env(env)
    loaded = load_workflow(
        _qualified_fit(layer=0, ks=[2], seeds=[0]), env, workflow_dir=tmp_path
    )
    out = tmp_path / "runs"
    first = run_workflow(loaded, env, out, _engine())
    root = first.run_root
    assert _statuses(first) == {name: "completed" for name in loaded.order}
    no_engine = run_workflow(loaded, env, out, None, resume=True)
    assert _statuses(no_engine) == {name: "reused" for name in loaded.order}

    record_path = root / "ctl" / SIDECAR
    record = json.loads(record_path.read_text())
    record["engine"] = "nnsight"
    record_path.write_text(json.dumps(record))
    foreign = run_workflow(loaded, env, out, None, resume=True)
    assert _statuses(foreign) == {name: "reused" for name in loaded.order}
    assert _record(root, "ctl")["engine"] == "nnsight"  # reused as recorded

    implementation = {"tree_digest": runtime_identity().tree_digest}
    step = loaded.document.steps["ctl"]
    assert runner._engine_for(loaded, "ctl", None) is None  # pyright: ignore[reportPrivateUsage]
    kept = runner._reusable(  # pyright: ignore[reportPrivateUsage]
        loaded, "ctl", step, root / "ctl", True, False, implementation, None
    )
    assert kept is not None and kept["status"] == "reused"
    del record["engine"]
    record_path.write_text(json.dumps(record))
    dropped = runner._reusable(  # pyright: ignore[reportPrivateUsage]
        loaded, "ctl", step, root / "ctl", True, False, implementation, None
    )
    assert dropped is None


@pytest.mark.smoke
def test_a_self_contained_matched_random_qualifies_its_fit_through_the_derived_edge(
    env, tmp_path
) -> None:
    """The run-level twin of the refusal: the self-contained control without
    ``after`` runs first through the derived edge and the fit's record carries
    the ``controls`` block it inherits — one draw ``passed`` at every fit
    point (the refusal of the same control authored ``after`` its target is
    the load-only test above — this is the valid work it must not touch)."""
    env = _tiny_env(env)
    loaded = load_workflow(
        _self_contained_matched_random(seeds=[0]), env, workflow_dir=tmp_path
    )
    assert loaded.dependencies == {"fit": ("ctl",), "ctl": ()}
    result = run_workflow(loaded, env, tmp_path / "runs", _engine())
    assert _statuses(result) == {"ctl": "completed", "fit": "completed"}
    fit = _record(result.run_root, "fit")
    block = fit["controls"]
    assert block["inherited_from"] == ["ctl"]
    assert (block["n_invalid"], block["n_points"]) == (0, 1)
    assert [e["controls"] for e in block["by_point"].values()] == [{"ctl": "passed"}]
    assert "controls" not in _record(result.run_root, "ctl")


@pytest.mark.smoke
def test_t12_a_dtype_change_on_both_steps_is_re_qualified_not_refused(
    env, tmp_path
) -> None:
    """The legitimate campaign: the same workflow with ``model.dtype`` set to
    bf16 on the fit **and** the control loads (rule 15 holds — both moved),
    and run into its own tree the control runs again against the new
    realization — new document digests, new records, the fit's inherited
    identity following. (A new realization is a new campaign: its workflow
    digest differs, so it is run as one, not resumed over the old tree.)"""
    env = _tiny_env(env)
    fp32 = load_workflow(
        _qualified_fit(layer=0, ks=[2], seeds=[0], bound=1.0),
        env,
        workflow_dir=tmp_path,
    )
    first = run_workflow(fp32, env, tmp_path / "fp32", _engine())
    assert _statuses(first) == {name: "completed" for name in fp32.order}
    before = _record(first.run_root, "ctl")

    raw = _qualified_fit(layer=0, ks=[2], seeds=[0], bound=1.0)
    raw["steps"]["fit"]["set"]["model.dtype"] = "bf16"
    raw["steps"]["ctl"]["set"]["model.dtype"] = "bf16"
    bf16 = load_workflow(raw, env, workflow_dir=tmp_path)
    assert bf16.digest != fp32.digest
    assert bf16.inner_digests["ctl"] != fp32.inner_digests["ctl"]
    assert bf16.inner_digests["fit"] != fp32.inner_digests["fit"]
    second = run_workflow(bf16, env, tmp_path / "bf16", _engine())
    assert _statuses(second) == {name: "completed" for name in bf16.order}
    after = _record(second.run_root, "ctl")
    assert after["document_digest"] != before["document_digest"]
    assert after["document_digest"] == bf16.inner_digests["ctl"]
    assert after["forwards"] == before["forwards"]
    fit = _record(second.run_root, "fit")
    assert fit["controls"]["identity"]["ctl"] == {
        "document_digest": after["document_digest"],
        "tree_digest": after["implementation"]["tree_digest"],
        "engine": "pytorch_hooks",
    }
    assert fit["document_digest"] == bf16.inner_digests["fit"]
