"""Controls as native conditions — the waiver + status-inheritance layer
(workflow spec §2.2, §5 rule 14, §8). "Corpus NN" is the intervention specification
``tests/protocols/NN_*_im.json``.

What is pinned, and how each test fails without the change:

* **The vocabularies.** The spec's ``kind``, ``reason``, ``status`` and
  ``seam`` tables are exactly the code's tuples — without the change the
  tuples do not exist and the module fails at import.
* **T4, the explicit waiver.** A workflow that engages the layer and runs a
  fit which neither declares nor waives ``matched_random`` is refused under
  rule 14 naming the step and the kind; an empty or unknown waiver reason, and
  an ``external`` waiver with no reference, are refused too — the mutation
  "accept a waiver with no reason". Without the change the
  workflow loads: ``control`` and ``waive`` are unknown keys (rule 1), and
  nothing says ``WorkflowError(14)``.
* **Fail-closed.** Every shipped and demo workflow loads and carries none of
  the three keys in any canonical entry; a workflow that declares nothing is
  not held to the requirement — four shipped and demo workflows run fits and
  declared no control before this layer existed.
* **T8, the legitimate campaign.** ``13_random_subspace_control_im.json``
  loads with its corpus pin byte-identical; declared as the ``matched_random``
  control of corpus 04's fit (k = 8 pairs, same site, three recorded draws) it
  validates, and the declared and hand-built forms expand to identical inner
  digests; the ``fit → random_mask → apply`` chain declares its control the
  same way with the seed on the script step's inputs. Mutations: a ``k``
  mismatch, a seed set the document does not draw, a target without a fit and
  a draw count below ``min_draws`` are each refused naming the field.
* **The self-swap predicate** is checked against the compiled document and a
  failure names the field; a ``self_swap`` control nobody certifies is refused.
* **The ledger.** The runner spells a control's points the way a saved
  bundle's header spells them (``coords_token``, pinned against the writer):
  an axis on the saved read's own entity and a non-scalar coordinate both
  join a certifier's rows — without the change they were refused as matching
  no point. A certification that leaves a point without a row is a
  ``ControlFailure`` naming the point (a row for every point is unchanged);
  every control's record carries its points' coordinates, so ``--resume``
  re-seats a swept control; the kind dispatch refuses a kind it has no check
  for. The ``instrument_failure`` warning lines move no status
  (``derive_statuses`` reads only ``attempt_failed``).
* **Agreement folds the band.** ``_agree`` compares a dependent's swept
  ``layers: L`` with a control's authored ``layers: [L]`` as one canonical
  value (IM spec §2.4) — without the change ``L != [L]`` and a list-spelled
  control said nothing at its own layer.
* **T6, inheritance, and the stop bound** run on the tiny fixture and live in
  ``test_controls_run.py`` (``smoke`` — one tier per test).
"""

from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path
from typing import Any

import pytest

from causalab.protocol.identity import import_closure
from causalab.protocol.rules.errors import ProtocolError
from causalab.protocol.pipeline import compile_protocol
from causalab.workflow.derived import derive_statuses
from causalab.workflow.document import (
    CONTROL_KINDS,
    CONTROL_RULE,
    CONTROL_SEAMS,
    CONTROL_STATUSES,
    DEFAULT_MIN_DRAWS,
    INHERITED_STATUSES,
    MAX_RULE,
    REQUIRED_CONTROL_KINDS,
    WAIVER_REASONS,
    ProtocolStep,
    WorkflowError,
    certifier_subject,
    load_workflow,
)
from causalab.workflow.runner import INSTRUMENT_FAILURE, ControlFailure, coords_token
from causalab.analysis.certify_control import REQUIRED_INPUTS

from tests.protocol._env import steps_of
from tests.protocol.test_vocabulary_census import CODE, _rows  # the table parser
from tests.workflow.test_closure_census import (
    DEMO_WORKFLOWS,
    SHARED,
    SHIPPED,
    _demo_env,  # pyright: ignore[reportPrivateUsage]
)
from tests._helpers.paths import PROTOCOLS_DIR, WORKFLOWS_DIR

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "docs" / "workflow_protocol.md"
#: §4.2, §6 and the runner contract of §8.
INTERNALS = REPO / "docs" / "workflow_protocol_internals.md"
FIXTURES = Path(__file__).parent / "fixtures" / "controls"
CORPUS = REPO / "tests" / "protocols"
PROTOCOLS = PROTOCOLS_DIR
CORPUS_PINS = json.loads((REPO / "tests/protocol/corpus_digests.json").read_text())
TINY_LLAMA = "hf-internal-testing/tiny-random-LlamaForCausalLM"

EXTERNAL = {"reason": "external", "reference": "runs/2026-09-01/self_swap"}


def _section(heading: str, spec: Path = SPEC) -> str:
    """Read a numbered workflow section of ``spec`` through its next peer or parent heading."""
    depth = len(heading) - len(heading.lstrip("#"))
    number = re.match(r"^#+ (\d+(?:\.\d+)*)(?:\.)?(?=\s|$)", heading)
    assert number is not None, f"expected a numbered heading: {heading!r}"
    pattern = rf"^#{{{depth}}} {re.escape(number.group(1))}(?:\.)?(?=\s|$)[^\n]*\n"
    text = spec.read_text()
    matches = list(re.finditer(pattern, text, re.M))
    assert len(matches) == 1, f"expected one section {number.group(1)} in {spec.name}"
    body = text[matches[0].end() :]
    stop = re.compile(rf"^#{{1,{depth}}} ", re.M)
    end = stop.search(body)
    return body[: end.start()] if end else body


def _table(header: str) -> list[list[str]]:
    """§2.2's table whose header row starts with the plain word ``header``:
    its body rows, up to the first row whose first cell is not code."""
    rows = _rows(_section("### 2.2 `intervention_protocol` steps"))
    start = next(index for index, row in enumerate(rows) if row[0] == header)
    body: list[list[str]] = []
    for row in rows[start + 1 :]:
        if not row[0].startswith("`"):
            break
        body.append(row)
    return body


def _members(header: str) -> list[str]:
    return [CODE.findall(row[0])[0] for row in _table(header)]


def _workflow(steps: dict[str, Any]) -> dict[str, Any]:
    return {"version": "1", "output_dir": "controls", "steps": steps}


def _protocol(document: Path, **fields: Any) -> dict[str, Any]:
    return {"type": "intervention_protocol", "document": str(document), **fields}


def _certifier(control: str) -> dict[str, Any]:
    return {
        "type": "script",
        "script": {"module": "causalab.analysis.certify_control"},
        "inputs": {
            name: {"step": control, "file": f"{name}.safetensors"}
            for name in REQUIRED_INPUTS
        },
        "outputs": {"controls": "controls.json"},
    }


def _refused(raw: dict[str, Any], env: Any, match: str, **kwargs: Any) -> WorkflowError:
    with pytest.raises(WorkflowError) as info:
        load_workflow(raw, env, **kwargs)
    assert info.value.rule == CONTROL_RULE, str(info.value)
    assert re.search(match, str(info.value)), str(info.value)
    return info.value


# --------------------------------------------------------------------------- #
# the vocabularies, held to the spec
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "header, vocabulary, name",
    [
        ("kind", CONTROL_KINDS, "CONTROL_KINDS"),
        ("reason", WAIVER_REASONS, "WAIVER_REASONS"),
        (
            "status",
            CONTROL_STATUSES + INHERITED_STATUSES,
            "CONTROL_STATUSES + INHERITED",
        ),
        ("seam", CONTROL_SEAMS, "CONTROL_SEAMS"),
    ],
    ids=["kind", "reason", "status", "seam"],
)
def test_each_vocabulary_matches_its_spec_table(header, vocabulary, name) -> None:
    tabulated = _members(header)
    assert len(tabulated) >= 3, f"§2.2's {header} table was not found"
    assert len(set(tabulated)) == len(tabulated), f"§2.2 lists a {header} twice"
    assert set(tabulated) == set(vocabulary), (
        f"§2.2's {header} table and {name} disagree — only in the spec: "
        f"{sorted(set(tabulated) - set(vocabulary))}; only in the code: "
        f"{sorted(set(vocabulary) - set(tabulated))}"
    )


def test_every_vocabulary_row_says_what_it_means() -> None:
    for header in ("kind", "reason", "status", "seam"):
        for row in _table(header):
            assert len(row) >= 2 and len(" ".join(row[1:])) > 20, (
                f"§2.2 {header} row {row[0]} is bare"
            )


def test_the_required_kinds_are_the_two_certifiable_ones() -> None:
    # Both are declarable (`CONTROL_KINDS`)
    # but neither is *required* by rule 14: `shuffled_source` is the label
    # control — its document is the target's own, held by rule 14's whole
    # canonical form, and `single_role` may waive it (§2.2); `full_component`
    # is never required (full-component controls stay hand-authored
    # documents)
    assert set(REQUIRED_CONTROL_KINDS) == set(CONTROL_KINDS) - {
        "shuffled_source",
        "full_component",
    }
    assert set(INHERITED_STATUSES).isdisjoint(CONTROL_STATUSES)


def test_rule_14_is_the_controls_rule() -> None:
    """Numbered by rule, so a renumbering is a deliberate edit here. Rule 14
    is no longer the last: 15 is the qualify-once rule, 16 site equivalence,
    17 the behavioral step's (§2.7), 18 the decision / conditional layer's
    (§2.8), 19 the declared fan-out's (§2.9)."""
    assert MAX_RULE >= CONTROL_RULE == 14
    section = _section("## 5. Validation")
    item = re.search(r"^14\. (.+?)(?=^\d+\. |\Z)", section, re.M | re.S)
    assert item is not None
    text = item.group(1)
    assert "declared or waived" in text and "`self_swap`" in text
    assert "`matched_random`" in text and "`shuffled_source`" in text


def test_certify_control_has_the_shared_closure() -> None:
    """Named by no shipped workflow, so not hashed today (the ``random_mask``
    precedent) — but the day a document names it, its closure is exactly the
    protocol core, so no new module becomes digest-bearing."""
    closure = import_closure(REPO / "causalab/analysis/certify_control.py", root=REPO)
    assert tuple(closure) == SHARED


# --------------------------------------------------------------------------- #
# T4 — declared or waived, never silently omitted
# --------------------------------------------------------------------------- #

FIT = CORPUS / "04_das_im.json"
INTERCHANGE = CORPUS / "02_interchange_im.json"
HARVEST = CORPUS / "01_harvest_im.json"  # no counterfactual role
CONTROL_13 = CORPUS / "13_random_subspace_control_im.json"
TWIN = FIXTURES / "self_swap_twin.json"


def test_t4_a_fit_that_neither_declares_nor_waives_matched_random_is_refused(
    env,
) -> None:
    raw = _workflow({"fit": _protocol(FIT, waive={"self_swap": EXTERNAL})})
    err = _refused(
        raw,
        env,
        r"step 'fit' declares a fit; control 'matched_random' is neither declared "
        r"by a step nor waived",
    )
    assert err.path == "steps.fit"


def test_t4_valid_work_a_fit_waiving_both_kinds_loads(env) -> None:
    raw = _workflow(
        {
            "fit": _protocol(
                FIT, waive={"self_swap": EXTERNAL, "matched_random": EXTERNAL}
            )
        }
    )
    loaded = load_workflow(raw, env)
    entry = loaded.canonical["steps"]["fit"]
    assert entry["waive"] == {"self_swap": EXTERNAL, "matched_random": EXTERNAL}


@pytest.mark.parametrize(
    "waiver, match",
    [
        ("", r"waiver reason '' is not one of"),
        ("not_applicable", r"waiver reason 'not_applicable' is not one of"),
        ({"reason": "external"}, r"'external' waiver carries a 'reference'"),
        (
            {"reason": "no_fit", "reference": "x"},
            r"'reference' belongs to reason 'external'",
        ),
        (
            {"reason": "no_fit"},
            r"waives 'matched_random' as 'no_fit' but declares a fit",
        ),
    ],
    ids=[
        "empty",
        "unknown",
        "external-without-reference",
        "stray-reference",
        "no_fit-on-a-fit",
    ],
)
def test_t4_mutation_a_waiver_without_a_real_reason_is_refused(
    env, waiver, match
) -> None:
    """The mutation: a waiver with no reason accepted → T4 fails."""
    raw = _workflow(
        {"fit": _protocol(FIT, waive={"self_swap": EXTERNAL, "matched_random": waiver})}
    )
    _refused(raw, env, match)


def test_no_fit_and_single_role_waive_only_their_kind(env) -> None:
    _refused(
        _workflow({"fit": _protocol(FIT, waive={"self_swap": "no_fit"})}),
        env,
        r"'no_fit' waives 'matched_random'",
    )
    _refused(
        _workflow({"fit": _protocol(FIT, waive={"self_swap": "single_role"})}),
        env,
        r"'single_role' waives 'shuffled_source'",
    )
    _refused(
        _workflow(
            {"x": _protocol(INTERCHANGE, waive={"shuffled_source": "single_role"})}
        ),
        env,
        r"declares a counterfactual role",
    )


def test_an_unknown_kind_in_a_waiver_is_refused(env) -> None:
    _refused(
        _workflow({"fit": _protocol(FIT, waive={"no_op": EXTERNAL})}),
        env,
        r"unknown control kind 'no_op'",
    )


def test_a_kind_declared_and_waived_is_refused(env) -> None:
    raw = _workflow(
        {
            "fit": _protocol(
                FIT, waive={"self_swap": EXTERNAL, "matched_random": EXTERNAL}
            ),
            "control": _protocol(
                CONTROL_13,
                control={
                    "of": "fit",
                    "kind": "matched_random",
                    "seeds": [0, 1, 2],
                    "min_draws": 3,
                },
            ),
        }
    )
    _refused(raw, env, r"waives 'matched_random' and step 'control' declares it")


SHUFFLED = FIXTURES / "shuffled_source.json"  # corpus 02 + data.counterfactual.shuffle
#: the target of a shuffled-source control: an interchange that engages the
#: layer and covers the two required kinds itself (no fit → `no_fit`)
TARGET_WAIVERS = {"self_swap": EXTERNAL, "matched_random": "no_fit"}


def _shuffled_workflow(**control_set: Any) -> dict[str, Any]:
    return _workflow(
        {
            "target": _protocol(INTERCHANGE, waive=TARGET_WAIVERS),
            "shuffled": _protocol(
                SHUFFLED,
                set=control_set,
                control={"of": "target", "kind": "shuffled_source"},
            ),
        }
    )


def test_a_shuffled_source_control_loads_when_its_document_is_the_targets_plus_shuffle(
    env,
) -> None:
    """The valid-work twin: the fixture is corpus 02 with
    ``data.counterfactual.shuffle: {seed: 0}`` and nothing else changed, so the
    declaration loads, the control's canonical entry is the declaration, and the
    two inner documents differ in digest. Fails without the change: the loader
    refused the kind at parse ("not yet declarable")."""
    loaded = load_workflow(_shuffled_workflow(), env)
    assert loaded.canonical["steps"]["shuffled"]["control"] == {
        "of": "target",
        "kind": "shuffled_source",
    }
    inner = loaded.inner
    assert (
        inner["shuffled"].compiled.digests.document
        != inner["target"].compiled.digests.document
    )
    assert inner["shuffled"].compiled.canonical["data"]["counterfactual"][
        "shuffle"
    ] == {"seed": 0}
    assert "not yet declarable" not in json.dumps(loaded.canonical)


def test_a_shuffled_source_control_whose_document_authors_no_shuffle_is_refused(
    env,
) -> None:
    """Corpus 02 declared as a shuffled_source of itself's twin: no
    counterfactual role authors ``shuffle``, refused naming it."""
    raw = _workflow(
        {
            "target": _protocol(INTERCHANGE, waive=TARGET_WAIVERS),
            "shuffled": _protocol(
                INTERCHANGE, control={"of": "target", "kind": "shuffled_source"}
            ),
        }
    )
    err = _refused(raw, env, r"no counterfactual role of .* authors 'shuffle'")
    assert err.path == "steps.shuffled.control"
    assert "not yet declarable" not in str(err)


def test_a_shuffled_source_control_differing_elsewhere_is_refused_naming_the_field(
    env,
) -> None:
    """``shuffle`` plus a different ``sites.target.layers``: two differences,
    refused naming the first field beyond ``shuffle``. The mutation that skips
    the masked comparison (checks only that ``shuffle`` is authored) fails
    here."""
    err = _refused(
        _shuffled_workflow(**{"sites.target.layers": 17}),
        env,
        r"differs from the target's beyond 'shuffle' — at sites\.target\.layers\[0\]: 17 vs 18",
    )
    assert err.path == "steps.shuffled.control"


def test_a_shuffled_source_control_of_a_shuffled_target_is_refused(env) -> None:
    """The target is the unshuffled pairing: a target that itself authors
    ``shuffle`` is refused naming the role."""
    raw = _workflow(
        {
            "target": _protocol(SHUFFLED, waive=TARGET_WAIVERS),
            "shuffled": _protocol(
                SHUFFLED, control={"of": "target", "kind": "shuffled_source"}
            ),
        }
    )
    err = _refused(
        raw, env, r"the target itself authors 'shuffle' at data\.counterfactual"
    )
    assert err.path == "steps.shuffled.control"


def test_the_single_role_waiver_of_shuffled_source_still_loads(env) -> None:
    """The ``single_role`` waiver is unchanged: a document with no counterfactual role
    waives ``shuffled_source`` as ``single_role`` and loads; the same waiver on
    a document with a counterfactual role is still refused."""
    load_workflow(
        _workflow({"x": _protocol(HARVEST, waive={"shuffled_source": "single_role"})}),
        env,
    )
    _refused(
        _workflow(
            {"x": _protocol(INTERCHANGE, waive={"shuffled_source": "single_role"})}
        ),
        env,
        r"declares a counterfactual role",
    )


def test_a_shuffled_source_controls_points_are_passed_when_they_ran(env) -> None:
    """Per-point status follows the ``matched_random`` convention:
    the permutation was checked at load, so a point that ran is ``passed`` on
    the control's own record (``by_point``), with nothing to certify. Fails
    without the runner change: the status stays ``None`` (``not_run``)."""
    from causalab.workflow.runner import _ControlLedger  # pyright: ignore[reportPrivateUsage]

    loaded = load_workflow(_shuffled_workflow(), env)
    step = loaded.document.steps["shuffled"]
    assert isinstance(step, ProtocolStep)
    block = _ControlLedger().declare(
        "shuffled",
        step,
        loaded,
        loaded.inner["shuffled"].point_digests,
        [{}],
        # rule 15: the ledger records the qualification's identity (§8)
        identity={
            "document_digest": loaded.inner["shuffled"].compiled.digests.document,
            "tree_digest": "0" * 64,
            "engine": "pytorch_hooks",
        },
    )
    assert block["kind"] == "shuffled_source"
    (entry,) = block["by_point"].values()
    assert entry == {"coords": {}, "status": "passed"}
    assert block["n_points"] == 1 and block["n_failed"] == 0


@pytest.mark.parametrize(
    "control, match",
    [
        ({"of": "fit", "kind": "no_op"}, r"unknown control kind 'no_op'"),
        ({"of": "fit", "kind": "self_swap", "seam": "D"}, r"unknown seam 'D'"),
        (
            {"of": "fit", "kind": "self_swap", "seeds": [0, 0]},
            r"'seeds' repeats a value",
        ),
        (
            {"of": "fit", "kind": "self_swap", "seeds": [True]},
            r"seed True is not an integer",
        ),
        (
            {"of": "fit", "kind": "self_swap", "min_draws": 0},
            r"'min_draws' is a positive integer",
        ),
        ({"of": "nowhere", "kind": "self_swap"}, r"names unknown step 'nowhere'"),
        ({"of": "ctl", "kind": "self_swap"}, r"declares itself its own control"),
    ],
    ids=[
        "kind",
        "seam",
        "seeds-repeat",
        "seeds-bool",
        "min_draws",
        "of-unknown",
        "of-self",
    ],
)
def test_a_malformed_declaration_is_refused(env, control, match) -> None:
    raw = _workflow(
        {
            "fit": _protocol(FIT, waive={"matched_random": EXTERNAL}),
            "ctl": _protocol(TWIN, control=control),
            "cert": _certifier("ctl"),
        }
    )
    _refused(raw, env, match)


def test_a_stop_rate_needs_a_control_and_a_range(env) -> None:
    _refused(
        _workflow({"fit": _protocol(FIT, stop_after_failure_rate=0.5)}),
        env,
        r"only a step declaring 'control' authors it",
    )
    _refused(
        _workflow(
            {
                "fit": _protocol(FIT, waive={"matched_random": EXTERNAL}),
                "ctl": _protocol(
                    TWIN,
                    control={"of": "fit", "kind": "self_swap"},
                    stop_after_failure_rate=1.5,
                ),
                "cert": _certifier("ctl"),
            }
        ),
        env,
        r"number in \[0, 1\]",
    )


# --------------------------------------------------------------------------- #
# the self-swap predicate and its certifier
# --------------------------------------------------------------------------- #


def _self_swap_workflow(**control_set: Any) -> dict[str, Any]:
    return _workflow(
        {
            "fit": _protocol(FIT, waive={"matched_random": EXTERNAL}),
            "ctl": _protocol(
                TWIN,
                # rule 15: the twin follows the fit's bf16 realization
                set={"model.dtype": "bf16", **control_set},
                control={"of": "fit", "kind": "self_swap"},
            ),
            "cert": _certifier("ctl"),
        }
    )


def test_a_self_swap_control_loads_and_is_certified(env) -> None:
    loaded = load_workflow(_self_swap_workflow(), env)
    assert certifier_subject(loaded.document.steps, "cert") == "ctl"
    assert certifier_subject(loaded.document.steps, "fit") is None
    assert loaded.canonical["steps"]["ctl"]["control"] == {
        "of": "fit",
        "kind": "self_swap",
    }
    assert "control" not in loaded.canonical["steps"]["fit"]
    assert loaded.dependencies["cert"] == ("ctl",)


@pytest.mark.parametrize(
    "override, match",
    [
        (
            {"intervened_models.original_base.input": "counterfactual"},
            r"intervened_models.self_swap: operand read 'v_self' has input "
            r"'counterfactual', not the model's input 'base'",
        ),
        (
            {"writes.self_patch.pos": -2},
            r"operand read 'v_self' and write 'self_patch' differ at pos",
        ),
        (
            {
                "intervened_models.patched.reads": ["recv_target", "v_self"],
                "intervened_models.original_base.reads": ["recv_original"],
                "save[1].model": "patched",
            },
            r"operand read 'v_self' is taken from model 'patched', which lands writes",
        ),
    ],
    ids=["input", "pos", "model"],
)
def test_the_self_swap_predicate_names_the_failing_field(env, override, match) -> None:
    err = _refused(
        _self_swap_workflow(**override),
        env,
        r"no intervened model of .* is a self-swap",
    )
    assert re.search(match, str(err)), str(err)
    assert err.path == "steps.ctl.control"


def test_a_plain_interchange_is_not_a_self_swap(env) -> None:
    """Corpus 02 declared as a self_swap: its one model's operand is the
    counterfactual, so the predicate fails on ``input``. The certifier reads
    files 02 does write, so rule 4 is satisfied and rule 14 is what refuses."""
    cert = _certifier("ctl")
    cert["inputs"] = {
        name: {"step": "ctl", "file": "iia.json"} for name in REQUIRED_INPUTS
    }
    raw = _workflow(
        {
            "fit": _protocol(FIT, waive={"matched_random": EXTERNAL}),
            "ctl": _protocol(INTERCHANGE, control={"of": "fit", "kind": "self_swap"}),
            "cert": cert,
        }
    )
    _refused(raw, env, r"operand read 'v_cf' has input 'counterfactual'")


def test_a_self_swap_nobody_certifies_is_refused(env) -> None:
    raw = _self_swap_workflow()
    del raw["steps"]["cert"]
    _refused(raw, env, r"no script step certifies it")


def test_a_certifier_reads_exactly_one_control_and_authors_no_control_input(
    env,
) -> None:
    raw = _self_swap_workflow()
    raw["steps"]["cert"]["inputs"]["operand"] = {"step": "fit", "file": "iia.json"}
    raw["steps"]["cert"]["inputs"]["receiver_original"] = {
        "step": "fit",
        "file": "ce.json",
    }
    raw["steps"]["cert"]["inputs"]["receiver_control"] = {
        "step": "fit",
        "file": "rot.safetensors",
    }
    raw["steps"]["cert"]["inputs"]["receiver_target"] = {
        "step": "fit",
        "file": "rot.safetensors",
    }
    raw["steps"]["cert"]["inputs"]["overwritten"] = {
        "step": "fit",
        "file": "rot.safetensors",
    }
    _refused(
        raw,
        env,
        r"its inputs must read exactly one step declaring 'control' \(it reads none\)",
    )
    raw = _self_swap_workflow()
    raw["steps"]["cert"]["inputs"]["control"] = {"kind": "self_swap"}
    _refused(raw, env, r"also declares an input named 'control'")


# --------------------------------------------------------------------------- #
# T8 — the legitimate campaign: 13 as a declared matched_random control
# --------------------------------------------------------------------------- #


def _hand_built(raw: dict[str, Any]) -> dict[str, Any]:
    """The same workflow with every control-layer key stripped."""
    out = json.loads(json.dumps(raw))
    for step in out["steps"].values():
        for key in ("control", "waive", "stop_after_failure_rate"):
            step.pop(key, None)
    return out


def test_t8_corpus_13_still_loads_to_its_pin(env) -> None:
    loaded = compile_protocol(CONTROL_13, env=env)
    pin = CORPUS_PINS["13_random_subspace_control_im.json"]
    assert loaded.digests.document == pin["document"]
    assert list(steps_of(loaded, env).digests) == pin["points"]


def test_t8_the_declared_and_hand_built_forms_expand_identically(env) -> None:
    """The declaration is workflow-level: the inner documents, their digests
    and their points are byte-for-byte what the hand-built campaign had."""
    declared = load_workflow(FIXTURES / "matched_random.json", env)
    raw = json.loads((FIXTURES / "matched_random.json").read_text())
    hand_built = load_workflow(_hand_built(raw), env, workflow_dir=FIXTURES)
    assert declared.inner_digests == hand_built.inner_digests
    assert (
        declared.inner["control"].point_digests
        == hand_built.inner["control"].point_digests
    )
    assert declared.inner["control"].point_digests == tuple(
        CORPUS_PINS["13_random_subspace_control_im.json"]["points"]
    )
    assert (
        declared.digest != hand_built.digest
    )  # the declaration is in the workflow digest
    entry = declared.canonical["steps"]["control"]
    assert entry["control"] == {
        "of": "fit",
        "kind": "matched_random",
        "seeds": [0, 1, 2],
        "min_draws": 3,
    }
    assert "control" not in hand_built.canonical["steps"]["control"]


def test_t8_mutation_a_k_mismatch_is_refused(env) -> None:
    raw = json.loads((FIXTURES / "matched_random.json").read_text())
    raw["steps"]["control"]["set"] = {"featurizers.rot.k": 4}
    _refused(
        raw,
        env,
        r"pairs featurizer 'rot' \(k=4\) with fit 'fit''s 'rot' \(k=8\)",
        workflow_dir=FIXTURES,
    )


def test_t8_mutation_a_site_mismatch_is_refused(env) -> None:
    raw = json.loads((FIXTURES / "matched_random.json").read_text())
    raw["steps"]["control"]["set"] = {"sites.target.layers": 17}
    _refused(raw, env, r"is drawn at the fit's site", workflow_dir=FIXTURES)


def test_t8_mutation_seeds_the_document_does_not_draw_are_refused(env) -> None:
    raw = json.loads((FIXTURES / "matched_random.json").read_text())
    raw["steps"]["control"]["control"]["seeds"] = [0, 1]
    raw["steps"]["control"]["control"]["min_draws"] = 2
    _refused(
        raw,
        env,
        r"declares seeds \[0, 1\] but featurizers.rot.seed draws \[0, 1, 2\]",
        workflow_dir=FIXTURES,
    )


def test_t8_min_draws_defaults_to_twenty_and_is_recorded_when_authored_lower(
    env,
) -> None:
    raw = json.loads((FIXTURES / "matched_random.json").read_text())
    del raw["steps"]["control"]["control"]["min_draws"]
    assert DEFAULT_MIN_DRAWS == 20
    _refused(
        raw, env, r"declares 3 seeds, fewer than min_draws 20", workflow_dir=FIXTURES
    )
    del raw["steps"]["control"]["control"]["seeds"]
    _refused(raw, env, r"'matched_random' without 'seeds'", workflow_dir=FIXTURES)


def test_t8_mutation_a_matched_random_of_a_step_without_a_fit_is_refused(env) -> None:
    raw = _workflow(
        {
            "target": _protocol(INTERCHANGE, waive={"self_swap": EXTERNAL}),
            "control": _protocol(
                CONTROL_13,
                control={
                    "of": "target",
                    "kind": "matched_random",
                    "seeds": [0, 1, 2],
                    "min_draws": 3,
                },
            ),
        }
    )
    _refused(raw, env, r"of 'target', which declares no fit")


def test_t8_the_random_mask_chain_declares_its_control(env) -> None:
    """``fit → random_mask → apply``, the seed on the script step's inputs
    (the application protocol), declared as the fit's matched_random
    control; the pairing is by kind (gate ↔ gate) and site, the seed
    provenance is the drawing step's. ``random_mask`` is imported, never
    edited. Load-only, on the shipped documents."""
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
    assert loaded.order.index("draw") < loaded.order.index("apply_control")
    assert loaded.canonical["steps"]["apply_control"]["control"]["seeds"] == [7]
    steps["apply_control"]["control"]["seeds"] = [8]
    _refused(
        _workflow(steps), env, r"drawn by step 'draw' at seed 7 — record every draw"
    )
    del steps["draw"]["inputs"]["seed"]
    _refused(_workflow(steps), env, r"whose inputs carry no integer 'seed'")


@pytest.mark.parametrize(
    "path", SHIPPED + DEMO_WORKFLOWS, ids=[p.stem for p in SHIPPED + DEMO_WORKFLOWS]
)
def test_every_shipped_workflow_loads_and_authors_no_control(path: Path, env) -> None:
    """A workflow that declares nothing is not held to rule 14, and carries
    none of the three keys in any canonical entry — the digest it had."""
    shipped_root = WORKFLOWS_DIR
    loaded = load_workflow(
        path, env if path.parent == shipped_root else _demo_env(path)
    )
    for entry in loaded.canonical["steps"].values():
        assert not {"control", "waive", "stop_after_failure_rate"} & set(entry)


def test_a_workflow_that_declares_nothing_is_not_held_to_rule_14(env) -> None:
    """The unengaged fit: corpus 04 alone loads — the shipped fit campaigns
    predate the layer (`weekdays_8b`, `mcqa_components`, `mcqa_subspace`,
    `weekdays_geometry` all run fits and declared no control)."""
    loaded = load_workflow(_workflow({"fit": _protocol(FIT)}), env)
    assert "waive" not in loaded.canonical["steps"]["fit"]


def test_the_bare_and_object_waiver_forms_digest_identically(env) -> None:
    bare = load_workflow(
        _workflow(
            {
                "fit": _protocol(
                    FIT, waive={"self_swap": EXTERNAL, "matched_random": EXTERNAL}
                )
            }
        ),
        env,
    )
    nested = _workflow(
        {
            "fit": _protocol(
                FIT,
                waive={
                    "self_swap": EXTERNAL,
                    "matched_random": {
                        "reason": "external",
                        "reference": EXTERNAL["reference"],
                    },
                },
            )
        }
    )
    assert load_workflow(nested, env).digest == bare.digest
    a = load_workflow(
        _workflow({"x": _protocol(INTERCHANGE, waive={"matched_random": "no_fit"})}),
        env,
    )
    b = load_workflow(
        _workflow(
            {
                "x": _protocol(
                    INTERCHANGE, waive={"matched_random": {"reason": "no_fit"}}
                )
            }
        ),
        env,
    )
    assert a.digest == b.digest


# --------------------------------------------------------------------------- #
# the ledger — one coordinate spelling, every point a row, coordinates on resume
# --------------------------------------------------------------------------- #

#: A control swept on the layer and on an axis of its receiver read's own
#: entity, with a non-scalar coordinate — the two shapes the saved header
#: spells differently from the axis id (`pos`, not `recv_original.pos`; the
#: object as JSON text).
SWEPT_COORDS: list[dict[str, Any]] = [
    {"sites.target.layers": 0, "reads.recv_original.pos": {"index": -1}},
    {"sites.target.layers": 1, "reads.recv_original.pos": {"index": -2}},
]


def _header_coords(
    coords_list: list[dict[str, Any]], entity: str
) -> list[dict[str, Any]]:
    """The coordinates a saved bundle's header records for each point — written
    by the writer itself, so the test pins the mirror against the real thing."""
    import torch

    from causalab.neural.shared.results import TensorFile

    bundle = TensorFile()
    for coords in coords_list:
        bundle.add(entity, torch.zeros(1), coords)
    return [
        json.loads(json.dumps(meta["coords"])) for meta in bundle.entry_meta.values()
    ]


def _certifier_rows(
    coords_list: list[dict[str, Any]], entity: str
) -> list[dict[str, Any]]:
    """Rows as ``certify_control`` writes them: ``coords`` copied from the header."""
    return [
        {"coords": c, "status": "passed"} for c in _header_coords(coords_list, entity)
    ]


def _declared(
    env: Any, coords_list: list[dict[str, Any]], **control_set: Any
) -> tuple[Any, ...]:
    from causalab.workflow.runner import _ControlLedger  # pyright: ignore[reportPrivateUsage]

    loaded = load_workflow(_self_swap_workflow(**control_set), env)
    step = loaded.document.steps["ctl"]
    assert isinstance(step, ProtocolStep)
    ledger = _ControlLedger()
    digests = [f"{index:064x}" for index in range(len(coords_list))]
    block = ledger.declare(
        "ctl",
        step,
        loaded,
        digests,
        coords_list,
        # rule 15: the ledger records the qualification's identity (§8)
        identity={
            "document_digest": loaded.inner["ctl"].compiled.digests.document,
            "tree_digest": "0" * 64,
            "engine": "pytorch_hooks",
        },
    )
    return loaded, ledger, digests, block


def _quiet(event: str, payload: dict[str, Any]) -> None:
    raise AssertionError(f"nothing should be narrated here: {event} {payload}")


def test_the_ledger_spells_coordinates_as_the_saved_header_does(env) -> None:
    """One function on both sides of the join. The header drops the entity of
    the read it belongs to and writes a non-scalar coordinate as JSON text;
    `coords_token` against that entity is byte-equal to it, and a certifier's
    rows — copied from any of the control's saved bundles — join the control's
    points. Without the change the ledger tokenized `short_coords(coords)` on
    raw values (`recv_original.pos`, the object itself) and refused these rows
    as matching no point."""
    spelled = _header_coords(SWEPT_COORDS, "recv_original")
    assert spelled == [
        {"target.layers": 0, "pos": '{"index": -1}'},
        {"target.layers": 1, "pos": '{"index": -2}'},
    ]
    for coords, header in zip(SWEPT_COORDS, spelled):
        assert coords_token(coords, entry="recv_original") == json.dumps(
            header, sort_keys=True
        )
        assert coords_token(coords) != coords_token(coords, entry="recv_original")
    assert coords_token({}) == "{}"

    for entity in ("recv_original", "v_cf"):  # the receiver's header, another bundle's
        _, ledger, digests, _ = _declared(env, SWEPT_COORDS)
        block = ledger.certify(
            "cert", "ctl", _certifier_rows(SWEPT_COORDS, entity), 0.0, _quiet
        )
        assert [block["by_point"][d]["status"] for d in digests] == ["passed", "passed"]
        assert [block["by_point"][d]["coords"] for d in digests] == SWEPT_COORDS
        assert (block["n_failed"], block["n_points"]) == (0, 2)

    _, ledger, _, _ = _declared(env, SWEPT_COORDS)
    raw = [
        {"coords": dict(c), "status": "passed"} for c in SWEPT_COORDS
    ]  # the axis ids
    with pytest.raises(ProtocolError, match=r"matches no point of control 'ctl'"):
        ledger.certify("cert", "ctl", raw, 0.0, _quiet)


def test_a_certification_missing_a_points_row_is_refused_naming_the_point(env) -> None:
    """A control that did not run on a point cannot certify it: the point is
    named, the bound does not rescue it (`1.0` admits every failure, not an
    absent row), and nothing is narrated as an `instrument_failure`. The twin:
    a row for every point leaves the block as it was. Without the change the
    point was filled in as `not_run` and counted in the denominator."""
    rows = _certifier_rows(SWEPT_COORDS, "recv_original")
    _, ledger, digests, _ = _declared(env, SWEPT_COORDS)
    with pytest.raises(ControlFailure) as info:
        ledger.certify("cert", "ctl", rows[:1], 1.0, _quiet)
    message = str(info.value)
    assert (
        f"has no row for 1 of 2 points of control 'ctl' — first {digests[1]}" in message
    )
    assert "'sites.target.layers': 1" in message and "cannot certify it" in message

    _, ledger, digests, _ = _declared(env, SWEPT_COORDS)
    block = ledger.certify("cert", "ctl", rows, 0.0, _quiet)
    assert set(block["by_point"]) == set(digests) and block["n_points"] == 2
    assert {p["status"] for p in block["by_point"].values()} == {"passed"}


def test_a_controls_record_carries_its_coordinates_and_restore_re_seats_them(
    env,
) -> None:
    """A `self_swap` control's own record now carries
    `by_point` with every point's coordinates (`not_run` until certified), so
    `--resume` re-seats a swept control whether or not its certifier is
    reused; a record from before that (digests only) takes the loader's own
    coordinates when its compile expanded the same digests."""
    from causalab.workflow.runner import _ControlLedger  # pyright: ignore[reportPrivateUsage]

    loaded, _, digests, block = _declared(env, SWEPT_COORDS)
    assert block["by_point"] == {
        d: {"coords": c, "status": "not_run"} for d, c in zip(digests, SWEPT_COORDS)
    }
    assert block["n_points"] == 2 and "n_failed" not in block
    assert block["kind"] == "self_swap" and block["of"] == "fit"

    fresh = _ControlLedger()
    fresh.restore("ctl", {"control": block, "point_digests": digests}, loaded)
    points = fresh.entries["ctl"]["points"]
    assert [p["coords"] for p in points] == SWEPT_COORDS
    assert [p["status"] for p in points] == [None, None]  # unknown, not `not_run`
    certified = fresh.certify(
        "cert", "ctl", _certifier_rows(SWEPT_COORDS, "recv_original"), 0.0, _quiet
    )
    assert certified["n_points"] == 2 and certified["n_failed"] == 0

    swept = load_workflow(
        _self_swap_workflow(**{"sites.target.layers": {"sweep": [0, 1]}}), env
    )
    inner = swept.inner["ctl"]
    assert len(inner.point_digests) == 2
    legacy = _ControlLedger()
    legacy.restore(
        "ctl",
        {
            "control": {"of": "fit", "kind": "self_swap"},
            "point_digests": list(inner.point_digests),
        },
        swept,
    )
    assert [p["coords"] for p in legacy.entries["ctl"]["points"]] == [
        {"sites.target.layers": 0},
        {"sites.target.layers": 1},
    ]


def test_agree_folds_a_one_layer_band_to_its_member() -> None:
    """A control that authors `layers: [L]` (as every shipped document spells
    a band) and a dependent's point swept to the bare `L` carry one canonical
    value (IM spec §2.4, `schema._band`), so they agree — on either side of
    the comparison; a longer band, another layer, and a bare `L` against `L`
    behave as before, and an axis the other document lacks constrains nothing.
    Without the change `18 != [18]`: the list-spelled control was "pinned
    elsewhere" at its own layer and the dependent's point there was `not_run`
    instead of `instrument_invalid`."""
    from causalab.workflow.runner import _agree  # pyright: ignore[reportPrivateUsage]

    def authored(layers: Any) -> dict[str, Any]:
        site = {"component": "block_output", "layers": layers}
        return {"method": {"sites": {"target": site}}}

    point = {"sites.target.layers": 18}
    assert _agree(point, {}, {}, authored([18]))  # the fold: `18` against `[18]`
    assert _agree({}, authored([18]), point, {})  # ... on the other side too
    assert _agree(point, {}, {}, authored(18))  # a `set` override's spelling
    assert _agree(point, {}, point, {})  # two swept points at one layer
    assert not _agree(point, {}, {}, authored([18, 19]))  # a longer band is a list
    assert not _agree(point, {}, {}, authored(19))
    assert not _agree(point, {}, {}, authored([19]))
    assert not _agree(point, {}, {"sites.target.layers": 19}, {})
    assert not _agree({"sites.target.layers": True}, {}, {}, authored([1]))  # no bool
    assert _agree(point, {}, {}, {"method": {"sites": {"other": {"layers": [18]}}}})
    assert _agree(
        point, {}, {}, {"method": {"sites": {"target": {"layers": {"sweep": [18]}}}}}
    )


def test_the_kind_dispatch_refuses_a_kind_it_has_no_check_for(env) -> None:
    """`_check_controls` routes each kind to its check by name and refuses any
    other — unreachable from an authored document, whose `kind` is held to the
    closed vocabulary at parse, so the helper is called on a step whose parsed
    declaration was given a kind the tuple does not have."""
    from causalab.workflow.document import _check_controls  # pyright: ignore[reportPrivateUsage]

    loaded = load_workflow(_self_swap_workflow(), env)
    steps = dict(loaded.document.steps)
    ctl = steps["ctl"]
    assert isinstance(ctl, ProtocolStep) and ctl.control is not None
    steps["ctl"] = dataclasses.replace(
        ctl, control={**ctl.control, "kind": "no_such_kind"}
    )
    with pytest.raises(
        WorkflowError, match=r"control kind 'no_such_kind' has no load-time check"
    ) as info:
        # at one root the flattened table (§2.10) is the step table itself
        _check_controls(steps, loaded.inner, env.model_info, steps)
    assert info.value.rule == CONTROL_RULE
    assert "steps.ctl.control" in str(info.value)
    for kind in CONTROL_KINDS:
        assert f"'{kind}'" in str(info.value)
    # the authored path never reaches it
    _refused(
        _workflow(
            {
                **_self_swap_workflow()["steps"],
                "ctl": _protocol(TWIN, control={"of": "fit", "kind": "no_such_kind"}),
            }
        ),
        env,
        r"unknown control kind 'no_such_kind'",
    )


def test_instrument_failure_warnings_move_no_status() -> None:
    """``derive_statuses`` reads only ``attempt_failed`` warnings: a control
    point's failure is history, never a sixth step word."""
    order = ("ctl", "cert")
    deps = {"ctl": (), "cert": ("ctl",)}
    records = [
        {"seq": 0, "event": "phase_started", "payload": {"step": "ctl"}},
        {
            "seq": 1,
            "event": "phase_completed",
            "payload": {"step": "ctl", "status": "completed"},
        },
        {"seq": 2, "event": "phase_started", "payload": {"step": "cert"}},
        {
            "seq": 3,
            "event": "warning",
            "payload": {"step": "ctl", "reason": INSTRUMENT_FAILURE, "point": "a" * 64},
        },
        {
            "seq": 4,
            "event": "phase_completed",
            "payload": {"step": "cert", "status": "completed"},
        },
    ]
    assert derive_statuses(records, order=order, dependencies=deps) == {
        "ctl": "completed",
        "cert": "completed",
    }
    assert INSTRUMENT_FAILURE in INHERITED_STATUSES
