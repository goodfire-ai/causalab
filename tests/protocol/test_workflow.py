"""The Workflow Protocol document model (docs/workflow_protocol.md v2 §5):
parse rules, the reference grammar, the derived schedule, digest semantics, and
the shipped weekdays-8b worked example.

One test per checklist rule, asserted **by rule number** — so a renumbering of
the spec has to be a deliberate edit here rather than a silent drift.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from causalab.protocol.rules.errors import ProtocolWarning
from causalab.workflow.document import (
    WorkflowError,
    is_workflow,
    load_workflow,
    parse_workflow,
)
from tests._helpers.paths import PROTOCOLS_DIR, WORKFLOWS_DIR

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[2]
WEEKDAYS_WF = WORKFLOWS_DIR / "weekdays.json"


def _copy_locate(tmp_path: Path) -> None:
    methods = tmp_path / "methods"
    methods.mkdir(exist_ok=True)
    shutil.copyfile(
        PROTOCOLS_DIR / "weekdays_locate_scan.json",
        methods / "locate.json",
    )


def tiny_workflow(tmp_path: Path) -> dict[str, Any]:
    """A minimal protocol → script workflow over a copied method preset."""
    _copy_locate(tmp_path)
    return {
        "version": "1",
        "output_dir": "run",
        "steps": {
            "locate": {
                "type": "intervention_protocol",
                "document": "methods/locate.json",
            },
            "best": {
                "type": "script",
                "script": {"module": "causalab.workflow.scripts.select"},
                "inputs": {
                    "table": {"step": "locate", "file": "iia.json"},
                    "choose": "max",
                    "emit": {"best_layer": "sites.target.layers"},
                },
                "outputs": {
                    "values": {"file": "values.json", "keys": {"best_layer": 23}}
                },
            },
        },
    }


def script_workflow(
    tmp_path: Path, body: str = "def main(inputs, outputs):\n    pass\n"
) -> dict[str, Any]:
    """A workflow over a user script in the workflow directory."""
    _copy_locate(tmp_path)
    scripts = tmp_path / "scripts"
    scripts.mkdir(exist_ok=True)
    (scripts / "reduce.py").write_text(body)
    return {
        "version": "1",
        "output_dir": "run",
        "steps": {
            "locate": {
                "type": "intervention_protocol",
                "document": "methods/locate.json",
            },
            "reduce": {
                "type": "script",
                "script": {"path": "scripts/reduce.py"},
                "inputs": {"table": {"step": "locate", "file": "iia.json"}},
                "outputs": {
                    "out": {
                        "file": "out.json",
                        "columns": {"layer": "int64", "value": "float64"},
                    }
                },
            },
        },
    }


def expect_rule(rule: int, raw: dict[str, Any], env, tmp_path: Path) -> WorkflowError:
    with pytest.raises(WorkflowError) as err:
        load_workflow(raw, env, workflow_dir=tmp_path)
    assert err.value.rule == rule, f"expected W{rule}, got {err.value}"
    return err.value


# --------------------------------------------------------------------------- #
# rule 1 — strict keys, closed enums
# --------------------------------------------------------------------------- #


def test_is_workflow_dispatches_on_steps():
    assert is_workflow({"version": "1", "output_dir": "r", "steps": {}})
    assert not is_workflow({"version": "1", "model": {}, "save": []})


def test_rule_1_unknown_step_type_suggests():
    raw = {
        "version": "1",
        "output_dir": "run",
        "steps": {"a": {"type": "protocols", "document": "x.json"}},
    }
    with pytest.raises(WorkflowError) as err:
        parse_workflow(raw)
    assert err.value.rule == 1 and "protocol" in str(err.value)


def test_rule_1_transform_select_plot_are_gone():
    """v1's three Python-flavoured step types collapsed into `script`."""
    for retired in ("transform", "select", "plot"):
        raw = {
            "version": "1",
            "output_dir": "run",
            "steps": {"a": {"type": retired}},
        }
        with pytest.raises(WorkflowError) as err:
            parse_workflow(raw)
        assert err.value.rule == 1


def test_rule_1_unknown_top_level_key():
    raw = {"version": "1", "output_dir": "run", "steps": {}, "save": []}
    with pytest.raises(WorkflowError) as err:
        parse_workflow(raw)
    assert err.value.rule == 1 and "save" in str(err.value)


def test_rule_1_script_step_needs_script_inputs_outputs():
    for missing in ("script", "inputs", "outputs"):
        step = {
            "type": "script",
            "script": {"module": "causalab.workflow.scripts.select"},
            "inputs": {},
            "outputs": {"v": "v.json"},
        }
        del step[missing]
        with pytest.raises(WorkflowError) as err:
            parse_workflow({"version": "1", "output_dir": "run", "steps": {"a": step}})
        assert err.value.rule in (1, 7)


# --------------------------------------------------------------------------- #
# rule 2 — section order and output_dir
# --------------------------------------------------------------------------- #


def test_rule_2_section_order_warns_and_parses():
    """As for an intervention specification: the canonical form is built from
    the parsed
    document, so authored order reaches neither the digest nor the schedule."""
    raw = {
        "version": "1",
        "steps": {"a": {"type": "intervention_protocol", "document": "x.json"}},
        "output_dir": "run",
    }
    conventional = {
        "version": "1",
        "output_dir": "run",
        "steps": {"a": {"type": "intervention_protocol", "document": "x.json"}},
    }
    with pytest.warns(ProtocolWarning, match="recommended"):
        parsed = parse_workflow(raw)
    assert parsed == parse_workflow(conventional)


@pytest.mark.parametrize("bad", ["a/b", "/abs", "..", ".", "nested/dir"])
def test_rule_2_output_dir_is_one_segment(bad):
    raw = {
        "version": "1",
        "output_dir": bad,
        "steps": {"a": {"type": "intervention_protocol", "document": "x.json"}},
    }
    with pytest.raises(WorkflowError) as err:
        parse_workflow(raw)
    assert err.value.rule == 2


def test_rule_2_output_dir_required():
    with pytest.raises(WorkflowError) as err:
        parse_workflow(
            {
                "version": "1",
                "steps": {"a": {"type": "intervention_protocol", "document": "x"}},
            }
        )
    assert err.value.rule == 1


# --------------------------------------------------------------------------- #
# rule 3 — step names
# --------------------------------------------------------------------------- #


def test_rule_3_step_names_filesystem_safe():
    raw = {
        "version": "1",
        "output_dir": "run",
        "steps": {"a/b": {"type": "intervention_protocol", "document": "x.json"}},
    }
    with pytest.raises(WorkflowError) as err:
        parse_workflow(raw)
    assert err.value.rule == 3


def test_rule_3_reserved_step_names():
    """A step directory sits beside the run manifest and the sidecars."""
    for reserved in ("workflow.json", "_step"):
        raw = {
            "version": "1",
            "output_dir": "run",
            "steps": {
                reserved: {"type": "intervention_protocol", "document": "x.json"}
            },
        }
        with pytest.raises(WorkflowError) as err:
            parse_workflow(raw)
        assert err.value.rule == 3


# --------------------------------------------------------------------------- #
# rule 4 — the reference grammar
# --------------------------------------------------------------------------- #


def test_rule_4_unknown_step_in_input(env, tmp_path):
    raw = tiny_workflow(tmp_path)
    raw["steps"]["best"]["inputs"]["table"]["step"] = "ghost"
    expect_rule(4, raw, env, tmp_path)


def test_rule_4_input_names_a_file_the_producer_does_not_write(env, tmp_path):
    raw = tiny_workflow(tmp_path)
    raw["steps"]["best"]["inputs"]["table"]["file"] = "ghost.json"
    expect_rule(4, raw, env, tmp_path)


def test_rule_4_after_names_unknown_step(env, tmp_path):
    raw = tiny_workflow(tmp_path)
    raw["steps"]["best"]["after"] = ["ghost"]
    expect_rule(4, raw, env, tmp_path)


def test_rule_4_two_locators_refused(env, tmp_path):
    raw = tiny_workflow(tmp_path)
    raw["steps"]["best"]["inputs"]["table"] = {
        "step": "locate",
        "file": "iia.json",
        "path": "x.json",
    }
    expect_rule(1, raw, env, tmp_path)


def test_rule_4_two_selectors_refused(env, tmp_path):
    raw = tiny_workflow(tmp_path)
    raw["steps"]["best"]["inputs"]["table"] = {
        "step": "locate",
        "file": "acts.safetensors",
        "key": "x",
        "entry": {"k": 1},
    }
    expect_rule(4, raw, env, tmp_path)


def test_rule_4_key_selector_needs_a_json_locator(env, tmp_path):
    """A selector must match its locator's format — decidable from the
    filename alone, which is what having only two formats buys."""
    raw = tiny_workflow(tmp_path)
    raw["steps"]["best"]["inputs"]["table"] = {
        "step": "locate",
        "file": "rot.safetensors",
        "key": "best_layer",
    }
    expect_rule(4, raw, env, tmp_path)


def test_rule_4_entry_selector_needs_a_safetensors_locator(env, tmp_path):
    raw = tiny_workflow(tmp_path)
    raw["steps"]["best"]["inputs"]["table"] = {
        "step": "locate",
        "file": "iia.json",
        "entry": {"k": 1},
    }
    expect_rule(4, raw, env, tmp_path)


def test_rule_4_document_relative_path_must_exist(env, tmp_path):
    raw = tiny_workflow(tmp_path)
    raw["steps"]["best"]["inputs"]["pins"] = {"path": "configs/definitely_absent.json"}
    err = expect_rule(4, raw, env, tmp_path)
    assert "beside the workflow document" in str(err)


def test_rule_4_document_relative_path_that_exists_loads(env, tmp_path):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "pins.json").write_text("{}")
    raw = tiny_workflow(tmp_path)
    raw["steps"]["best"]["inputs"]["pins"] = {"path": "configs/pins.json"}
    loaded = load_workflow(raw, env, workflow_dir=tmp_path)
    assert "best" in loaded.order


def test_rule_4_a_relative_path_is_not_resolved_against_the_repo(env, tmp_path):
    """The base is the document's directory, not the checkout: a file that
    exists under the repo root but not beside the document is absent. An
    installed wheel has no repo root, so a document that leaned on one would
    load from a checkout and refuse everywhere else."""
    assert (PROTOCOLS_DIR / "interchange.json").is_file()
    raw = tiny_workflow(tmp_path)
    raw["steps"]["best"]["inputs"]["pins"] = {
        "path": "demos/methods/protocols/interchange.json"
    }
    expect_rule(4, raw, env, tmp_path)


def test_rule_4_absolute_path_is_not_existence_checked(env, tmp_path):
    """Validation and execution routinely run on different hosts, so an
    absolute path naming another machine's data must not fail a load."""
    raw = tiny_workflow(tmp_path)
    raw["steps"]["best"]["inputs"]["pins"] = {"path": "/nonexistent/fit.json"}
    loaded = load_workflow(raw, env, workflow_dir=tmp_path)
    assert loaded.unchecked_paths == ("best.pins: /nonexistent/fit.json",)


def test_rule_4_key_must_be_declared_by_the_producer(env, tmp_path):
    """The strengthened half of v1's rule 10: outputs are declared, so this is
    checkable against *any* step rather than only a `select` step."""
    raw = tiny_workflow(tmp_path)
    raw["steps"]["consume"] = {
        "type": "script",
        "script": {"module": "causalab.workflow.scripts.select"},
        "inputs": {"layers": {"step": "best", "file": "values.json", "key": "ghost"}},
        "outputs": {"values": "out.json"},
    }
    err = expect_rule(4, raw, env, tmp_path)
    assert "best_layer" in str(err)


def test_rule_4_key_of_a_protocol_step_is_refused(env, tmp_path):
    raw = tiny_workflow(tmp_path)
    raw["steps"]["consume"] = {
        "type": "script",
        "script": {"module": "causalab.workflow.scripts.select"},
        "inputs": {"layers": {"step": "locate", "file": "iia.json", "key": "x"}},
        "outputs": {"values": "out.json"},
    }
    expect_rule(4, raw, env, tmp_path)


# --------------------------------------------------------------------------- #
# rule 5 — acyclicity and the schedule
# --------------------------------------------------------------------------- #


def test_rule_5_cycle_via_after(env, tmp_path):
    raw = tiny_workflow(tmp_path)
    raw["steps"]["locate"]["after"] = ["best"]
    err = expect_rule(5, raw, env, tmp_path)
    assert "cycle" in str(err)


def test_schedule_levels_are_derived(env, tmp_path):
    raw = tiny_workflow(tmp_path)
    loaded = load_workflow(raw, env, workflow_dir=tmp_path)
    assert loaded.levels == (("locate",), ("best",))
    assert loaded.dependencies["best"] == ("locate",)


def test_independent_steps_share_a_level(env, tmp_path):
    """Parallelism nobody authored: two consumers of one producer."""
    raw = tiny_workflow(tmp_path)
    raw["steps"]["other"] = {
        "type": "script",
        "script": {"module": "causalab.workflow.scripts.select"},
        "inputs": {
            "table": {"step": "locate", "file": "iia.json"},
            "emit": {"worst_layer": "sites.target.layers"},
            "choose": "min",
        },
        "outputs": {"values": {"file": "values.json", "keys": {"worst_layer": 0}}},
    }
    loaded = load_workflow(raw, env, workflow_dir=tmp_path)
    assert loaded.levels[0] == ("locate",)
    assert set(loaded.levels[1]) == {"best", "other"}


# --------------------------------------------------------------------------- #
# rule 6 — script resolution, hashed and never imported
# --------------------------------------------------------------------------- #


def test_rule_6_script_is_a_locator(env, tmp_path):
    """v1's `causalab:<name>` namespace is gone: a script names a module or a
    path, so the document says which code runs instead of a registry deciding."""
    raw = tiny_workflow(tmp_path)
    raw["steps"]["best"]["script"] = "causalab:select"
    err = expect_rule(6, raw, env, tmp_path)
    assert "locator" in str(err)


def test_rule_6_unknown_module(env, tmp_path):
    raw = tiny_workflow(tmp_path)
    raw["steps"]["best"]["script"] = {"module": "causalab.analysis.no_such_thing"}
    expect_rule(6, raw, env, tmp_path)


def test_rule_6_module_must_be_a_dotted_identifier(env, tmp_path):
    raw = tiny_workflow(tmp_path)
    raw["steps"]["best"]["script"] = {"module": "not/a/module.py"}
    expect_rule(6, raw, env, tmp_path)


def test_rule_6_exactly_one_locator(env, tmp_path):
    raw = tiny_workflow(tmp_path)
    raw["steps"]["best"]["script"] = {
        "module": "causalab.workflow.scripts.select",
        "path": "scripts/x.py",
    }
    expect_rule(6, raw, env, tmp_path)


def test_a_module_script_resolves_without_importing(env, tmp_path):
    """`find_spec` gives the file; nothing executes at load (§4.2)."""
    loaded = load_workflow(tiny_workflow(tmp_path), env, workflow_dir=tmp_path)
    assert len(loaded.step_digests["best"]) == 64


def test_rule_6_missing_user_script(env, tmp_path):
    raw = script_workflow(tmp_path)
    raw["steps"]["reduce"]["script"] = "scripts/absent.py"
    expect_rule(6, raw, env, tmp_path)


def test_rule_6_script_must_not_escape_the_workflow_dir(env, tmp_path):
    raw = script_workflow(tmp_path)
    raw["steps"]["reduce"]["script"] = "../outside.py"
    expect_rule(6, raw, env, tmp_path)


def test_rule_6_script_must_parse(env, tmp_path):
    raw = script_workflow(tmp_path, body="def main(:\n")
    err = expect_rule(6, raw, env, tmp_path)
    assert "does not parse" in str(err)


def test_rule_6_script_must_declare_main(env, tmp_path):
    raw = script_workflow(tmp_path, body="def other(inputs, outputs):\n    pass\n")
    err = expect_rule(6, raw, env, tmp_path)
    assert "main" in str(err)


def test_script_hash_is_in_the_digest(env, tmp_path):
    """Why the hash is in the digest at all: `--resume` is otherwise wrong."""
    raw = script_workflow(tmp_path)
    first = load_workflow(raw, env, workflow_dir=tmp_path)
    (tmp_path / "scripts" / "reduce.py").write_text(
        "def main(inputs, outputs):\n    return 1\n"
    )
    second = load_workflow(raw, env, workflow_dir=tmp_path)
    assert first.digest != second.digest
    assert first.step_digests["reduce"] != second.step_digests["reduce"]


#: A script that imports a sibling module by name — the smallest closure there
#: is. The walk resolves ``helper`` against the script's own directory, which
#: is where Python would find it once the script put that directory on its
#: path (workflow spec §4.2).
IMPORTING_SCRIPT = "import helper\n\n\ndef main(inputs, outputs):\n    helper.go()\n"
HELPER = "def go():\n    return 1\n"
HELPER_REWORDED = "# the same function, one comment richer\ndef go():\n    return 1\n"


def _closure_workflow(tmp_path: Path) -> tuple[dict[str, Any], Path]:
    raw = script_workflow(tmp_path, body=IMPORTING_SCRIPT)
    helper = tmp_path / "scripts" / "helper.py"
    helper.write_text(HELPER)
    return raw, helper


def test_an_edit_in_an_imported_module_moves_the_digest(env, tmp_path):
    """A `{"path": …}` script step's identity covers its declared import
    closure — the siblings beside it (§4.2, §7): a comment-only edit to a
    module the script *imports* moves the step digest and the workflow digest,
    while the script's own hash stays put. Nothing else covers a sibling: the
    package's `tree_digest` does not see it, so without the closure `--resume`
    would reuse a step whose arithmetic may have changed."""
    raw, helper = _closure_workflow(tmp_path)
    first = load_workflow(raw, env, workflow_dir=tmp_path)
    # the guard against the guard: a byte-identical reload agrees
    assert load_workflow(raw, env, workflow_dir=tmp_path).digest == first.digest

    helper.write_text(HELPER_REWORDED)
    second = load_workflow(raw, env, workflow_dir=tmp_path)
    assert first.digest != second.digest, "an edited import left the digest alone"
    assert first.step_digests["reduce"] != second.step_digests["reduce"]
    # `script_sha256` keeps its meaning — the script's own bytes, unchanged
    assert (
        first.canonical["steps"]["reduce"]["script_sha256"]
        == second.canonical["steps"]["reduce"]["script_sha256"]
    )


def test_a_module_script_naming_an_installed_module_declares_no_closure(
    env, tmp_path, monkeypatch
):
    """The `{"module": …}` twin of the stdlib `code` locator (§4.2): a script
    step naming an installed module (the stdlib's `json.tool`, which has a
    module-level `main`) is hashed — the document named it — but declares no
    closure, because its imports are runtime identity, and carries neither
    closure key. Walking them from the stdlib root would admit hundreds of
    files and move the digest on every Python patch release.

    The forbidden walk is asserted directly — no file under the stdlib root is
    ever parsed for its imports — rather than through a wall-clock bound on the
    load, which this test carried until 2026-09-09. Under CI's ``pytest -n
    auto`` the same load measured 0.2s on one PR head and 0.7s on a sibling's
    with identical code under test, in wall clock *and* in process CPU time:
    ``import_closure`` caches parsed imports per content hash
    (``_IMPORTS_BY_SHA``), so the first walk a worker performs pays for the
    whole repository closure of the workflow's other steps and a later one pays
    nothing, and which worker this test lands on is a function of the test set.
    A clock cannot separate that from the walk it was meant to catch; a spy on
    the parser can."""
    import hashlib
    import sysconfig

    from causalab.protocol import identity as code
    from causalab.protocol.identity import resolve_locator

    stdlib = Path(sysconfig.get_paths()["stdlib"]).resolve()
    parsed: list[Path] = []
    real_import_refs = code._import_refs

    def spying_import_refs(data, file):
        parsed.append(Path(file).resolve())
        return real_import_refs(data, file)

    monkeypatch.setattr(code, "_import_refs", spying_import_refs)
    raw = tiny_workflow(tmp_path)
    raw["steps"]["best"]["script"] = {"module": "json.tool"}
    loaded = load_workflow(raw, env, workflow_dir=tmp_path)
    walked_stdlib = sorted(str(p) for p in parsed if stdlib in p.parents)
    assert not walked_stdlib, (
        f"the closure walk entered the stdlib: {walked_stdlib[:5]}"
    )
    entry = loaded.canonical["steps"]["best"]
    stdlib_file = resolve_locator("json.tool").path
    assert (
        entry["script_sha256"] == hashlib.sha256(stdlib_file.read_bytes()).hexdigest()
    )
    assert "closure" not in entry and "closure_sha256" not in entry


def test_a_module_script_in_a_user_package_keeps_its_sibling_closure(
    env, tmp_path, monkeypatch
):
    """A `{"module": …}` locator into the author's own
    package on `sys.path` is the one locator spelling the tree digest does not
    cover, so its siblings enter the identity exactly as a `{"path": …}`
    script's do (§4.2) — one walk for both forms. Editing the sibling moves
    the step digest while `script_sha256` stays."""
    import hashlib
    import importlib
    import sys

    site = tmp_path / "site"
    pkg = site / "mystudy"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    sibling = pkg / "arith.py"
    sibling.write_text("VALUE = 1\n")
    (pkg / "step.py").write_text(
        "from mystudy import arith\n\n\ndef main(inputs, outputs):\n    arith.VALUE\n"
    )
    monkeypatch.syspath_prepend(str(site))
    importlib.invalidate_caches()
    try:
        raw = tiny_workflow(tmp_path)
        raw["steps"]["best"]["script"] = {"module": "mystudy.step"}
        first = load_workflow(raw, env, workflow_dir=tmp_path)
        entry = first.canonical["steps"]["best"]
        assert entry["closure"] == {
            "mystudy/arith.py": hashlib.sha256(sibling.read_bytes()).hexdigest()
        }
        sibling.write_text("VALUE = 2\n")
        importlib.invalidate_caches()
        second = load_workflow(raw, env, workflow_dir=tmp_path)
        assert second.step_digests["best"] != first.step_digests["best"]
        assert (
            second.canonical["steps"]["best"]["script_sha256"] == entry["script_sha256"]
        )
    finally:
        for name in ("mystudy", "mystudy.step", "mystudy.arith"):
            sys.modules.pop(name, None)


def test_a_module_script_inside_the_package_declares_no_closure(env):
    """The package's bytes are runtime identity — the `tree_digest` every
    record carries and `--resume` compares (§7) — so a `{"module": …}` step
    naming a package module carries neither closure key, and no edit to the
    protocol core moves a shipped workflow's digest. Every shipped script step
    is such a module; the demos are covered by
    `tests/workflow/test_closure_census.py`."""
    for name in ("weekdays.json", "mean_ablation.json", "pca_basis.json"):
        loaded = load_workflow(WEEKDAYS_WF.with_name(name), env)
        for step, entry in loaded.canonical["steps"].items():
            assert "closure" not in entry and "closure_sha256" not in entry, (
                name,
                step,
            )
            if entry["type"] == "script":
                assert "module" in entry["script"], (name, step)


def test_a_script_importing_nothing_beside_itself_carries_no_closure_keys(
    env, tmp_path
):
    """The closure's witness that valid work is not refused and gets
    a stable identity. A `{"path": …}` script with only stdlib imports has an
    empty manifest, and an empty manifest is written as no keys at all — the
    same identity a `{"module": …}` step has, so the two spellings of a
    sibling-free script cannot differ by a closure field."""
    raw = script_workflow(
        tmp_path,
        body="import json\n\n\ndef main(inputs, outputs):\n    json.dumps(1)\n",
    )
    loaded = load_workflow(raw, env, workflow_dir=tmp_path)
    entry = loaded.canonical["steps"]["reduce"]
    assert "closure" not in entry and "closure_sha256" not in entry
    assert load_workflow(raw, env, workflow_dir=tmp_path).digest == loaded.digest


def test_a_nested_function_named_main_is_not_enough(env, tmp_path):
    raw = script_workflow(
        tmp_path,
        body="def wrapper():\n    def main(inputs, outputs):\n        pass\n",
    )
    expect_rule(6, raw, env, tmp_path)


# --------------------------------------------------------------------------- #
# rule 7 — outputs
# --------------------------------------------------------------------------- #


def test_rule_7_outputs_non_empty(env, tmp_path):
    raw = tiny_workflow(tmp_path)
    raw["steps"]["best"]["outputs"] = {}
    expect_rule(7, raw, env, tmp_path)


@pytest.mark.parametrize("bad", ["out.csv", "out.parquet", "out.txt", "out"])
def test_rule_7_closed_output_formats(env, tmp_path, bad):
    """Two record formats plus three visualization ones (§2.5). Anything else is
    refused, not merely unknown."""
    raw = tiny_workflow(tmp_path)
    raw["steps"]["best"]["outputs"] = {"values": bad}
    expect_rule(7, raw, env, tmp_path)


def test_rule_7_output_must_stay_in_the_step_dir(env, tmp_path):
    raw = tiny_workflow(tmp_path)
    raw["steps"]["best"]["outputs"] = {"values": "../escape.json"}
    expect_rule(7, raw, env, tmp_path)


def test_rule_7_two_slots_one_file(env, tmp_path):
    raw = tiny_workflow(tmp_path)
    raw["steps"]["best"]["outputs"] = {"a": "same.json", "b": "same.json"}
    expect_rule(7, raw, env, tmp_path)


def test_rule_7_columns_and_keys_are_exclusive(env, tmp_path):
    raw = tiny_workflow(tmp_path)
    raw["steps"]["best"]["outputs"] = {
        "values": {
            "file": "values.json",
            "columns": {"a": "int64"},
            "keys": {"best_layer": 1},
        }
    }
    expect_rule(7, raw, env, tmp_path)


def test_rule_7_unknown_column_dtype(env, tmp_path):
    raw = tiny_workflow(tmp_path)
    raw["steps"]["best"]["outputs"] = {
        "values": {"file": "values.json", "columns": {"a": "float32"}}
    }
    err = expect_rule(7, raw, env, tmp_path)
    assert "float64" in str(err)


@pytest.mark.parametrize("figure", ["fig.png", "fig.pdf", "fig.html"])
def test_rule_7_visualization_formats_are_legal(env, tmp_path, figure):
    """A figure carries no record, so it is a legal output that declares no
    shape — png is the preferred default, pdf and html the deliberate ones."""
    raw = tiny_workflow(tmp_path)
    raw["steps"]["best"]["outputs"] = {"figure": figure}
    loaded = load_workflow(raw, env, workflow_dir=tmp_path)
    assert loaded.document.steps["best"].outputs["figure"].file == figure


@pytest.mark.parametrize("figure", ["fig.png", "fig.pdf", "fig.html"])
def test_rule_7_a_figure_declares_no_shape(env, tmp_path, figure):
    raw = tiny_workflow(tmp_path)
    raw["steps"]["best"]["outputs"] = {
        "figure": {"file": figure, "columns": {"a": "int64"}}
    }
    expect_rule(7, raw, env, tmp_path)


def test_rule_7_safetensors_declares_no_columns(env, tmp_path):
    raw = tiny_workflow(tmp_path)
    raw["steps"]["best"]["outputs"] = {
        "values": {"file": "w.safetensors", "columns": {"a": "int64"}}
    }
    expect_rule(7, raw, env, tmp_path)


# --------------------------------------------------------------------------- #
# rule 8 — protocol steps and their `set`
# --------------------------------------------------------------------------- #


def test_rule_8_set_override_must_target_existing_path(env, tmp_path):
    raw = tiny_workflow(tmp_path)
    raw["steps"]["locate"]["set"] = {"sites.ghost.layers": 3}
    expect_rule(8, raw, env, tmp_path)


def test_rule_8_missing_document(env, tmp_path):
    raw = tiny_workflow(tmp_path)
    raw["steps"]["locate"]["document"] = "methods/absent.json"
    expect_rule(4, raw, env, tmp_path)


def test_rule_8_inner_load_errors_surface(env, tmp_path):
    raw = tiny_workflow(tmp_path)
    raw["steps"]["locate"]["set"] = {"model.key": "not-a-registered-model"}
    err = expect_rule(8, raw, env, tmp_path)
    assert "does not load" in str(err)


# --------------------------------------------------------------------------- #
# rule 10 / 11 — runtime and is_deterministic
# --------------------------------------------------------------------------- #


def test_rule_10_isolated_step_declares_deps(env, tmp_path):
    raw = tiny_workflow(tmp_path)
    raw["steps"]["best"]["runtime"] = {"isolate": True}
    expect_rule(10, raw, env, tmp_path)


def test_rule_10_runtime_env_is_a_list_of_names(env, tmp_path):
    raw = tiny_workflow(tmp_path)
    raw["steps"]["best"]["runtime"] = {"isolate": True, "deps": ["x"], "env": "TOKEN"}
    expect_rule(10, raw, env, tmp_path)


def test_runtime_is_in_the_digest(env, tmp_path):
    """A different dependency set is a different computation, so `--resume`
    must not skip across a change to it."""
    raw = tiny_workflow(tmp_path)
    plain = load_workflow(raw, env, workflow_dir=tmp_path).digest
    raw["steps"]["best"]["runtime"] = {"isolate": True, "deps": ["umap-learn"]}
    isolated = load_workflow(raw, env, workflow_dir=tmp_path).digest
    assert plain != isolated


def test_rule_11_is_deterministic_is_a_boolean(env, tmp_path):
    raw = tiny_workflow(tmp_path)
    raw["steps"]["best"]["is_deterministic"] = "no"
    expect_rule(11, raw, env, tmp_path)


def test_nondeterministic_steps_are_reported(env, tmp_path):
    raw = tiny_workflow(tmp_path)
    raw["steps"]["best"]["is_deterministic"] = False
    loaded = load_workflow(raw, env, workflow_dir=tmp_path)
    assert loaded.nondeterministic == ("best",)


def test_is_deterministic_is_in_the_digest(env, tmp_path):
    raw = tiny_workflow(tmp_path)
    a = load_workflow(raw, env, workflow_dir=tmp_path).digest
    raw["steps"]["best"]["is_deterministic"] = False
    b = load_workflow(raw, env, workflow_dir=tmp_path).digest
    assert a != b


# --------------------------------------------------------------------------- #
# gone in v2: the sink rule and the save section
# --------------------------------------------------------------------------- #


def test_a_terminal_step_no_one_consumes_is_legal(env, tmp_path):
    """v1's sink rule refused this; everything declared is now published, so a
    terminal plot or report step needs no blessing."""
    raw = tiny_workflow(tmp_path)
    loaded = load_workflow(raw, env, workflow_dir=tmp_path)
    assert "best" in loaded.order  # nothing consumes `best`, and that is fine


def test_save_section_is_rejected(env, tmp_path):
    raw = tiny_workflow(tmp_path)
    raw["save"] = [{"step": "best", "value": "values.json", "file_path": "b.json"}]
    with pytest.raises(WorkflowError) as err:
        parse_workflow(raw)
    assert err.value.rule == 1


# --------------------------------------------------------------------------- #
# canonical form and digests (§7)
# --------------------------------------------------------------------------- #


def test_output_dir_is_excluded_from_the_digest(env, tmp_path):
    """It names where the run lands, not what the run is."""
    raw = tiny_workflow(tmp_path)
    first = load_workflow(raw, env, workflow_dir=tmp_path)
    raw["output_dir"] = "somewhere_else"
    second = load_workflow(raw, env, workflow_dir=tmp_path)
    assert first.digest == second.digest


def test_inner_document_edits_move_the_workflow_digest(env, tmp_path):
    raw = tiny_workflow(tmp_path)
    before = load_workflow(raw, env, workflow_dir=tmp_path).digest
    doc = tmp_path / "methods/locate.json"
    inner = json.loads(doc.read_text())
    # a description is authoring metadata and moves no digest (IM spec §7);
    # an edit to the experiment does
    inner["model"]["dtype"] = "bf16"
    doc.write_text(json.dumps(inner))
    after = load_workflow(raw, env, workflow_dir=tmp_path).digest
    assert before != after


def test_canonical_form_carries_no_output_dir(env, tmp_path):
    loaded = load_workflow(tiny_workflow(tmp_path), env, workflow_dir=tmp_path)
    assert "output_dir" not in loaded.canonical
    assert loaded.canonical["steps"]["best"]["script_sha256"]


def test_the_closure_manifest_names_the_file_that_moved(env, tmp_path):
    """The manifest is what makes a moved digest explainable: the canonical
    entry carries `path -> sha256` per closure member, and an edit to one
    member changes exactly that row and `closure_sha256` — nothing else in the
    entry. A stub that hashed the file alone has no manifest to consult."""
    import hashlib

    from causalab.protocol.identity import closure_sha256

    raw, helper = _closure_workflow(tmp_path)
    before = load_workflow(raw, env, workflow_dir=tmp_path).canonical["steps"]["reduce"]
    assert before["closure"] == {
        "helper.py": hashlib.sha256(helper.read_bytes()).hexdigest()
    }
    assert before["closure_sha256"] == closure_sha256(before["closure"])

    helper.write_text(HELPER_REWORDED)
    after = load_workflow(raw, env, workflow_dir=tmp_path).canonical["steps"]["reduce"]
    changed = {
        key for key in set(before) | set(after) if before.get(key) != after.get(key)
    }
    assert changed == {"closure", "closure_sha256"}
    assert after["closure"] == {
        "helper.py": hashlib.sha256(helper.read_bytes()).hexdigest()
    }


def test_inputs_are_sorted_in_the_canonical_form(env, tmp_path):
    raw = tiny_workflow(tmp_path)
    raw["steps"]["best"]["inputs"] = {
        "zzz": 1,
        "table": {"step": "locate", "file": "iia.json"},
        "emit": {"best_layer": "sites.target.layers"},
        "aaa": 2,
    }
    loaded = load_workflow(raw, env, workflow_dir=tmp_path)
    keys = list(loaded.canonical["steps"]["best"]["inputs"])
    assert keys == sorted(keys)


# --------------------------------------------------------------------------- #
# the shipped worked example (§10)
# --------------------------------------------------------------------------- #


def test_weekdays_example_loads_with_the_spec_schedule(env):
    loaded = load_workflow(WEEKDAYS_WF, env)
    assert [sorted(level) for level in loaded.levels] == [
        ["locate"],
        ["best", "scan_heatmap"],
        ["fit"],
        ["best_fit", "iia_by_k"],
        ["apply"],
    ]
    assert loaded.inner_digest_kind == {
        "locate": "campaign",
        "fit": "authored",
        "apply": "authored",
    }
    assert len(loaded.inner["locate"].expansion.points) == 56
    assert len(loaded.inner["fit"].expansion.points) == 9


def test_rewording_a_deferred_steps_document_moves_no_workflow_digest(env, tmp_path):
    """A deferred step (`fit`, `apply`: its document waits on an earlier step's
    output) is digested as authored, and the header's authoring fields are
    dropped there as the canonical form drops them (IM spec §7) — so renaming
    or re-describing the document moves neither kind of inner digest."""
    configs = WEEKDAYS_WF.parents[1]
    for sub in ("protocols", "workflows"):  # not the run trees beside them
        shutil.copytree(configs / sub, tmp_path / "configs" / sub)
    copied = tmp_path / "configs" / "workflows" / WEEKDAYS_WF.name
    before = load_workflow(copied, env)
    assert before.inner_digest_kind["fit"] == "authored"
    fit_doc = tmp_path / "configs" / "protocols" / "weekdays_das_sweep.json"
    raw = json.loads(fit_doc.read_text())
    raw["header"]["title"] = "renamed"
    raw["header"]["description"] = "reworded: " + raw["header"].get("description", "")
    fit_doc.write_text(json.dumps(raw))
    after = load_workflow(copied, env)
    assert after.inner_digests == before.inner_digests
    assert after.digest == before.digest


def test_the_shipped_workflows_identities_are_their_steps(env):
    """There is no pinned whole-workflow digest (§7): the identities `--resume`
    compares are per step — a script step's entry digest, a protocol step's
    inner document digest — and both shipped workflows load to exactly those.
    A step type that leaked a key into another kind's entry is caught by the
    per-kind key censuses (`tests/workflow/test_*.py`), not by a pin."""
    loaded = load_workflow(WEEKDAYS_WF, env)
    assert set(loaded.step_digests) == {"best", "best_fit", "scan_heatmap", "iia_by_k"}
    assert set(loaded.inner_digests) == {"locate", "fit", "apply"}
    plain = load_workflow(WEEKDAYS_WF.with_name("mean_ablation.json"), env)
    assert plain.step_digests == {}
    assert set(plain.inner_digests) == set(plain.document.steps)


def test_workflow_attention_override_reaches_campaign_and_changes_digest(env, tmp_path):
    raw = tiny_workflow(tmp_path)
    before = load_workflow(raw, env, workflow_dir=tmp_path)
    raw["steps"]["locate"]["set"] = {"model.attn_implementation": "sdpa"}
    after = load_workflow(raw, env, workflow_dir=tmp_path)
    assert before.digest != after.digest
    assert before.inner_digests["locate"] != after.inner_digests["locate"]
    assert all(
        point["model"]["attn_implementation"] == "sdpa"
        for point in after.inner["locate"].canonical_points
    )
