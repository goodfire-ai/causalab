"""The controls layer run end to end on the tiny fixture (workflow spec §2.2,
§8). The load-time half — the vocabularies, rule 14,
the predicates, the ledger's own arithmetic — is ``test_controls.py`` (``unit``);
everything here loads a model, so it is ``smoke``, one tier per test
(``docs/TESTS.md``).

* **T6, inheritance.** Two self-swap controls at two layers, each certified by
  ``causalab.analysis.certify_control``, one forced vacuous; the dependent
  sweep over both layers inherits, by coordinates, ``n_invalid: 1`` of
  ``n_points: 2`` — its layer-1 point is ``instrument_invalid`` because its
  control ``failed`` there, its layer-0 point ``passed``. Mutation: a
  dependent that computed its own status would report ``n_invalid: 0``.
* **T6, mixed spellings.** The same workflow with each control's layer
  authored as the one-layer band ``[L]`` (as every shipped document spells it)
  against the dependent's bare sweep values: the fold ``L`` ≡ ``[L]`` (IM
  spec §2.4) carries the failure to the dependent's point. Without the change
  ``1 != [1]`` and the list-spelled control was "pinned elsewhere" — the
  point was ``not_run``, the failure lost.
* **The stop bound.** With the default ``stop_after_failure_rate`` (``0.0``)
  the same failure makes the certifying step ``failed`` and the dependent
  ``blocked``; the ``instrument_failure`` warning lines move no status.
  "Does not expand further" is out of scope here: nothing below stops an
  expansion.
* **A swept control.** One control swept over both layers certifies both
  points — the certifier's rows carry the saved header's coordinate spelling
  and the ledger spells its points the same way — and a ``--resume`` that
  reuses the control while its certifier runs again re-seats the points,
  coordinates included, from the control's own record. Without the change
  the re-seated points have no coordinates and the rows match no point.
* **A missing row.** A certifier that drops a row is a ``ControlFailure``
  naming the point: the certifying step is ``failed``, the dependent
  ``blocked``, and no point is filled in as ``not_run`` to dilute the rate.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from causalab.io.events import EVENTS_FILE, read_events
from causalab.io.step_record import SIDECAR
from causalab.io.tables import read_table
from causalab.workflow import manifest as mf
from causalab.workflow.derived import derive_statuses
from causalab.workflow.document import load_workflow
from causalab.workflow.runner import INSTRUMENT_FAILURE, ControlFailure, run_workflow

from tests.workflow.test_controls import (
    INTERCHANGE,
    TINY_LLAMA,
    TWIN,
    _certifier,  # pyright: ignore[reportPrivateUsage]
    _protocol,  # pyright: ignore[reportPrivateUsage]
    _workflow,  # pyright: ignore[reportPrivateUsage]
)

pytestmark = pytest.mark.smoke


# --------------------------------------------------------------------------- #
# T6 — inheritance by coordinates, and the stop bound (tiny fixture)
# --------------------------------------------------------------------------- #


def _t6_workflow(
    *, fail_b: bool, bound: float | None, band: bool = False
) -> dict[str, Any]:
    """Two self-swap controls of the dependent sweep, one per layer, each
    certified; ``fail_b`` makes the layer-1 control vacuous (its target's
    operand read on ``base``); ``bound`` is the declared escape, or the
    default when ``None``; ``band`` spells each control's layer as the
    one-layer band ``[L]`` (as every shipped document does) instead of the
    bare index. Everything runs on the tiny fixture."""
    escape = {} if bound is None else {"stop_after_failure_rate": bound}
    steps: dict[str, Any] = {}
    for name, layer in (("ctl_a", 0), ("ctl_b", 1)):
        overrides: dict[str, Any] = {
            "model.key": TINY_LLAMA,
            "sites.target.layers": [layer] if band else layer,
        }
        if fail_b and name == "ctl_b":
            overrides["intervened_models.original_counterfactual.input"] = "base"
        steps[name] = _protocol(
            TWIN,
            set=overrides,
            control={"of": "dep", "kind": "self_swap", "seam": "A"},
            **escape,
        )
        steps[f"cert_{name[-1]}"] = _certifier(name)
    steps["dep"] = _protocol(
        INTERCHANGE,
        set={"model.key": TINY_LLAMA, "sites.target.layers": {"sweep": [0, 1]}},
        after=["cert_a", "cert_b"],
        waive={"matched_random": "no_fit"},
    )
    return _workflow(steps)


def _tiny_env(env: Any) -> Any:
    """The test environment with the tiny model's static metadata — what the
    CLI's ``--register-from-hf`` pre-pass supplies for a run (`cli.py`)."""
    from transformers import AutoConfig

    from causalab.protocol.registry import model_info_from_hf_config
    from causalab.io.env import ResolutionEnv

    info = model_info_from_hf_config(TINY_LLAMA, AutoConfig.from_pretrained(TINY_LLAMA))
    return ResolutionEnv(
        datasets=env.datasets, artifacts=env.artifacts, model_info=lambda key: info
    )


def _engine() -> Any:
    from causalab.neural.shared.engine_router import route

    return route("auto", device="cpu")


def _record(run_root: Path, step: str) -> dict[str, Any]:
    return json.loads((run_root / step / SIDECAR).read_text())


def test_t6_the_dependent_inherits_its_controls_status_by_coordinates(
    env, tmp_path
) -> None:
    """The layer-1 control is vacuous and fails; the layer-0 control passes;
    the dependent sweep's record says ``n_invalid: 1`` of ``n_points: 2``, by
    the coordinate each control was pinned to — never by computing its own
    status (the mutation: a dependent reporting on itself says 0 of 2)."""
    env = _tiny_env(env)
    loaded = load_workflow(
        _t6_workflow(fail_b=True, bound=1.0), env, workflow_dir=tmp_path
    )
    result = run_workflow(loaded, env, tmp_path / "runs", _engine())
    statuses = {name: e["status"] for name, e in result.manifest["steps"].items()}
    assert statuses == {name: "completed" for name in loaded.order}

    cert_a, cert_b = (
        _record(result.run_root, "cert_a"),
        _record(result.run_root, "cert_b"),
    )
    assert (cert_a["certifies"]["n_failed"], cert_a["certifies"]["n_points"]) == (0, 1)
    assert (cert_b["certifies"]["n_failed"], cert_b["certifies"]["n_points"]) == (1, 1)
    assert cert_b["certifies"]["stop_after_failure_rate"] == 1.0
    assert (
        cert_b["certifies"]["of"] == "dep"
        and cert_b["certifies"]["kind"] == "self_swap"
    )
    ctl_b = _record(result.run_root, "ctl_b")
    declaration = ctl_b["control"]
    assert {
        k: v for k, v in declaration.items() if k not in ("by_point", "n_points")
    } == {"of": "dep", "kind": "self_swap", "seam": "A"}
    # the control's own record names its point (unswept: no coordinates) and
    # says `not_run` — the verdict is the certifier's record's
    assert declaration["n_points"] == 1 and "n_failed" not in declaration
    assert declaration["by_point"] == {
        ctl_b["point_digests"][0]: {"coords": {}, "status": "not_run"}
    }
    assert "controls" not in ctl_b  # nothing upstream of a control here

    dep = _record(result.run_root, "dep")
    assert dep["waive"] == {"matched_random": {"reason": "no_fit"}}
    block = dep["controls"]
    assert block["inherited_from"] == ["ctl_a", "ctl_b"]
    assert (block["n_invalid"], block["n_points"]) == (1, 2)
    by_layer = {
        entry["coords"]["sites.target.layers"]: entry
        for entry in block["by_point"].values()
    }
    assert by_layer[0]["status"] == "passed" and by_layer[0]["controls"] == {
        "ctl_a": "passed"
    }
    assert by_layer[1]["status"] == "instrument_invalid"
    assert by_layer[1]["controls"] == {"ctl_b": "failed"}
    assert set(block["by_point"]) == set(dep["point_digests"])

    # 7·3: the failed control point is a warning on the stream, not a status
    records = read_events(result.run_root / EVENTS_FILE)
    failures = [r for r in records if r["event"] == "warning"]
    assert [(r["payload"]["reason"], r["payload"]["step"]) for r in failures] == [
        (INSTRUMENT_FAILURE, "ctl_b")
    ]
    assert failures[0]["payload"]["certified_by"] == "cert_b"
    assert failures[0]["payload"]["point"] in cert_b["certifies"]["by_point"]
    assert (
        derive_statuses(records, order=loaded.order, dependencies=loaded.dependencies)
        == statuses
    )
    for name in loaded.order:
        assert result.manifest["steps"][name]["status"] in mf.STEP_STATUSES


def test_t6_a_list_spelled_control_pins_the_dependents_point_at_its_layer(
    env, tmp_path
) -> None:
    """The mixed-spelling T6. Each control authors its
    band as a list, ``layers: [L]``, as every shipped document does; the
    dependent sweeps the bare values ``[0, 1]``. The two spellings are one
    canonical value (IM spec §2.4), so the layer-1 control's failure reaches
    the dependent's point at layer 1 — ``instrument_invalid``, ``n_invalid: 1``
    of ``n_points: 2``, the ``instrument_failure`` warning — exactly as T6
    shows with matching spellings. Without the change ``_agree`` compared
    ``1 != [1]``, the control was "pinned elsewhere" and the point was
    ``not_run``: a control that failed at exactly that layer contributed
    nothing."""
    env = _tiny_env(env)
    workflow = _t6_workflow(fail_b=True, bound=1.0, band=True)
    assert workflow["steps"]["ctl_b"]["set"]["sites.target.layers"] == [1]
    loaded = load_workflow(workflow, env, workflow_dir=tmp_path)
    # the ledger reads the control's explicit document: list-spelled, as authored
    assert loaded.inner["ctl_b"].compiled.tree["method"]["sites"]["target"][
        "layers"
    ] == [1]
    result = run_workflow(loaded, env, tmp_path / "runs", _engine())
    statuses = {name: e["status"] for name, e in result.manifest["steps"].items()}
    assert statuses == {name: "completed" for name in loaded.order}

    cert_a, cert_b = (
        _record(result.run_root, "cert_a"),
        _record(result.run_root, "cert_b"),
    )
    assert (cert_a["certifies"]["n_failed"], cert_a["certifies"]["n_points"]) == (0, 1)
    assert (cert_b["certifies"]["n_failed"], cert_b["certifies"]["n_points"]) == (1, 1)

    dep = _record(result.run_root, "dep")
    block = dep["controls"]
    assert block["inherited_from"] == ["ctl_a", "ctl_b"]
    assert (block["n_invalid"], block["n_points"]) == (1, 2)
    by_layer = {
        entry["coords"]["sites.target.layers"]: entry
        for entry in block["by_point"].values()
    }
    assert by_layer[0]["status"] == "passed" and by_layer[0]["controls"] == {
        "ctl_a": "passed"
    }
    assert by_layer[1]["status"] == "instrument_invalid"
    assert by_layer[1]["controls"] == {"ctl_b": "failed"}

    records = read_events(result.run_root / EVENTS_FILE)
    failures = [r for r in records if r["event"] == "warning"]
    assert [(r["payload"]["reason"], r["payload"]["step"]) for r in failures] == [
        (INSTRUMENT_FAILURE, "ctl_b")
    ]
    assert failures[0]["payload"]["certified_by"] == "cert_b"


def test_the_default_bound_makes_the_certifier_a_failed_step_and_blocks_below(
    env, tmp_path
) -> None:
    """``stop_after_failure_rate`` unauthored is ``0.0``: the first failure
    stops. The certifier is ``failed`` (a ``ControlFailure``, its
    ``controls.json`` retained with the attempt), the dependent ``blocked`` by
    the manifest's own rule, the stream ends ``attempt_failed`` →
    ``campaign_terminal failed``. It does not test stopping an expansion."""
    env = _tiny_env(env)
    loaded = load_workflow(
        _t6_workflow(fail_b=True, bound=None), env, workflow_dir=tmp_path
    )
    with pytest.raises(
        ControlFailure, match=r"control 'ctl_b' \(self_swap of 'dep'\): 1 of 1"
    ):
        run_workflow(loaded, env, tmp_path / "runs", _engine())
    run_root = tmp_path / "runs" / "controls"
    manifest = json.loads((run_root / mf.MANIFEST).read_text())
    statuses = {name: e["status"] for name, e in manifest["steps"].items()}
    assert statuses == {
        "ctl_a": "completed",
        "cert_a": "completed",
        "ctl_b": "completed",
        "cert_b": "failed",
        "dep": "blocked",
    }
    assert manifest["steps"]["dep"]["blocked_by"] == ["cert_b"]
    assert manifest["steps"]["cert_b"]["error"]["type"] == "ControlFailure"
    assert not (run_root / "cert_b").exists() and not (run_root / "dep").exists()
    (attempt,) = sorted((run_root / mf.ATTEMPTS_DIR / "cert_b").glob("0*"))
    assert (attempt / "controls.json").is_file()
    assert (
        json.loads((attempt / mf.ATTEMPT_RECORD).read_text())["error"]["type"]
        == "ControlFailure"
    )

    records = read_events(run_root / EVENTS_FILE)
    reasons = [r["payload"]["reason"] for r in records if r["event"] == "warning"]
    assert reasons == [INSTRUMENT_FAILURE, "attempt_failed"]
    assert records[-1]["payload"]["outcome"] == "failed"
    assert (
        derive_statuses(records, order=loaded.order, dependencies=loaded.dependencies)
        == statuses
    )


# --------------------------------------------------------------------------- #
# a swept control: every point certifies, and --resume re-seats its coordinates
# --------------------------------------------------------------------------- #


def _swept_workflow(*, certifier: dict[str, Any] | None = None) -> dict[str, Any]:
    """One self-swap control swept over both layers, certified by ``certifier``
    (the shipped script unless given); the dependent swept the same way."""
    swept = {"model.key": TINY_LLAMA, "sites.target.layers": {"sweep": [0, 1]}}
    return _workflow(
        {
            "ctl": _protocol(
                TWIN, set=swept, control={"of": "dep", "kind": "self_swap", "seam": "A"}
            ),
            "cert": certifier or _certifier("ctl"),
            "dep": _protocol(
                INTERCHANGE,
                set=swept,
                after=["cert"],
                waive={"matched_random": "no_fit"},
            ),
        }
    )


def test_a_swept_control_certifies_every_point_and_resume_re_seats_its_coordinates(
    env, tmp_path
) -> None:
    """The control's two points both certify (the twin is a real self-swap at
    either layer) and the dependent inherits ``passed`` at each layer from the
    control's point at that layer. Then the certifier and the dependent are
    removed and the run resumed: the control is ``reused`` and the ledger
    re-seated from its record — with coordinates, so the certifier's rows
    join. Without the change the reused control's points carry ``{}`` and the
    certifier is refused as matching no point."""
    env = _tiny_env(env)
    loaded = load_workflow(_swept_workflow(), env, workflow_dir=tmp_path)
    result = run_workflow(loaded, env, tmp_path / "runs", _engine())
    statuses = {name: e["status"] for name, e in result.manifest["steps"].items()}
    assert statuses == {name: "completed" for name in loaded.order}

    ctl = _record(result.run_root, "ctl")
    declaration = ctl["control"]
    assert declaration["n_points"] == 2 and "n_failed" not in declaration
    assert set(declaration["by_point"]) == set(ctl["point_digests"])
    assert sorted(
        e["coords"]["sites.target.layers"] for e in declaration["by_point"].values()
    ) == [0, 1]
    assert {e["status"] for e in declaration["by_point"].values()} == {"not_run"}

    cert = _record(result.run_root, "cert")
    assert (cert["certifies"]["n_failed"], cert["certifies"]["n_points"]) == (0, 2)
    assert {
        e["coords"]["sites.target.layers"]: e["status"]
        for e in cert["certifies"]["by_point"].values()
    } == {0: "passed", 1: "passed"}
    # the rows say what the saved header says — the short name, not the axis id
    rows = read_table(result.run_root / "cert" / "controls.json")
    assert sorted((r["coords"] for r in rows), key=lambda c: c["target.layers"]) == [
        {"target.layers": 0},
        {"target.layers": 1},
    ]

    dep = _record(result.run_root, "dep")
    block = dep["controls"]
    assert (block["inherited_from"], block["n_invalid"], block["n_points"]) == (
        ["ctl"],
        0,
        2,
    )
    assert all(
        e["status"] == "passed" and e["controls"] == {"ctl": "passed"}
        for e in block["by_point"].values()
    )

    # --resume: the control is reused, the certifier and the dependent run again
    shutil.rmtree(result.run_root / "cert")
    shutil.rmtree(result.run_root / "dep")
    again = run_workflow(loaded, env, tmp_path / "runs", _engine(), resume=True)
    assert {name: e["status"] for name, e in again.manifest["steps"].items()} == {
        "ctl": "reused",
        "cert": "completed",
        "dep": "completed",
    }
    assert _record(again.run_root, "ctl") == ctl
    assert _record(again.run_root, "cert")["certifies"] == cert["certifies"]
    assert _record(again.run_root, "dep")["controls"] == block


# --------------------------------------------------------------------------- #
# a certifier that leaves a point without a row
# --------------------------------------------------------------------------- #

DROPPING = """
from pathlib import Path

from causalab.analysis.certify_control import main as certify
from causalab.io.step_io import write_table
from causalab.io.tables import read_table


def main(inputs, outputs):
    certify(inputs, outputs)
    rows = read_table(Path(outputs["controls"]))
    write_table(Path(outputs["controls"]), rows[:-1])
"""


def test_a_certifier_that_leaves_a_point_without_a_row_is_a_failed_step(
    env, tmp_path
) -> None:
    """The shipped certifier's rows minus the last one: a point the control
    expanded has no row, so the certification is a ``ControlFailure`` naming
    the point — the step is ``failed``, the dependent ``blocked`` — and the
    uncovered point is neither filled in as ``not_run`` nor narrated as an
    ``instrument_failure``: it did not fail, it was never certified. Without
    the change the point is ``not_run``, the rate is ``0 / 2`` and the
    dependent inherits ``not_run`` there."""
    env = _tiny_env(env)
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "dropping.py").write_text(DROPPING)
    certifier = {**_certifier("ctl"), "script": {"path": "scripts/dropping.py"}}
    loaded = load_workflow(
        _swept_workflow(certifier=certifier), env, workflow_dir=tmp_path
    )
    with pytest.raises(
        ControlFailure,
        match=r"controls\.json has no row for 1 of 2 points of control 'ctl'",
    ) as info:
        run_workflow(loaded, env, tmp_path / "runs", _engine())
    assert "a control that did not run on a point cannot certify it" in str(info.value)
    run_root = tmp_path / "runs" / "controls"
    manifest = json.loads((run_root / mf.MANIFEST).read_text())
    statuses = {name: e["status"] for name, e in manifest["steps"].items()}
    assert statuses == {"ctl": "completed", "cert": "failed", "dep": "blocked"}
    assert manifest["steps"]["cert"]["error"]["type"] == "ControlFailure"
    assert manifest["steps"]["dep"]["blocked_by"] == ["cert"]
    ctl = _record(run_root, "ctl")
    assert f"first {sorted(ctl['point_digests'])[0]}" in str(info.value) or any(
        digest in str(info.value) for digest in ctl["point_digests"]
    )
    records = read_events(run_root / EVENTS_FILE)
    reasons = [r["payload"]["reason"] for r in records if r["event"] == "warning"]
    assert reasons == ["attempt_failed"]
    assert (
        derive_statuses(records, order=loaded.order, dependencies=loaded.dependencies)
        == statuses
    )
