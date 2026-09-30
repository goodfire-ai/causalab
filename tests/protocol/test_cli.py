"""The CLI verbs (spec §9) — validate / explain / digest, plus --set."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from causalab.cli import main
from causalab.protocol.engine import Engine
from causalab.protocol.schema import COMPONENTS
from causalab.protocol.pipeline import compile_protocol
from causalab.protocol.rules.data import check_data_columns

from tests.protocol._env import CORPUS_DIR, FIXTURES, steps_of
from tests._helpers.paths import PROTOCOLS_DIR


pytestmark = pytest.mark.unit


def _argv(verb: str, name: str, artifacts_root, *extra: str) -> list[str]:
    """The verb's argument list. ``--engine auto`` is supplied for every verb
    that takes the flag (it is required); an explicit ``--engine``
    in ``extra`` comes later and wins, as argparse takes the last value."""
    return [
        verb,
        str(CORPUS_DIR / name),
        "--data-root",
        str(FIXTURES / "data"),
        "--artifacts-root",
        str(artifacts_root),
        *(() if verb == "digest" else ("--engine", "auto")),
        *extra,
    ]


def test_validate_ok(capsys, artifacts_root):
    assert main(_argv("validate", "02_interchange_im.json", artifacts_root)) == 0
    assert "OK" in capsys.readouterr().out


def test_validate_data_checks_columns(capsys, artifacts_root):
    code = main(_argv("validate", "02_interchange_im.json", artifacts_root, "--data"))
    assert code == 0


def _aggregation(raw: dict, label: str) -> dict:
    """The aggregation the save entry labelled ``label`` carries (§2.10)."""
    (entry,) = [e for e in raw["method"]["save"] if e["file_path"] == f"{label}.json"]
    return entry["aggregation"]


def test_validate_data_catches_missing_column(env):
    loaded = compile_protocol(CORPUS_DIR / "02_interchange_im.json", env=env)
    raw = json.loads(json.dumps(dict(loaded.tree)))
    _aggregation(raw, "logit_diff")["a"] = "not_a_column"
    reloaded = compile_protocol(raw, env=env)
    with pytest.raises(Exception) as err:
        check_data_columns(reloaded, env)
    assert "not_a_column" in str(err.value)


def test_digest_prints_the_document_digest(capsys, env, artifacts_root):
    assert main(_argv("digest", "04_das_im.json", artifacts_root)) == 0
    printed = capsys.readouterr().out.strip()
    assert (
        printed
        == compile_protocol(CORPUS_DIR / "04_das_im.json", env=env).digests.document
    )


def test_explain_reports_plan(capsys, artifacts_root):
    """The first step's forward plan prints under `plan` — its groups, one
    per (model, input) — and no `forwards N per point` line: the count a
    step owes is the run's."""
    assert main(_argv("explain", "03_path_patching_im.json", artifacts_root)) == 0
    out = capsys.readouterr().out
    assert "forwards" not in out
    plan = out[out.index("plan\n") :]
    assert len([line for line in plan.splitlines() if " on " in line]) == 4
    assert "paired_forward" in out


def test_explain_sweep_reports_point_count(capsys, artifacts_root):
    assert (
        main(_argv("explain", "07_weekdays_locate_scan_im.json", artifacts_root)) == 0
    )
    out = capsys.readouterr().out
    assert "points    64" in out


def test_set_override_changes_digest(capsys, env, artifacts_root):
    assert (
        main(
            _argv(
                "digest",
                "02_interchange_im.json",
                artifacts_root,
                "--set",
                "sites.target.layers=5",
            )
        )
        == 0
    )
    overridden = capsys.readouterr().out.strip()
    assert (
        overridden
        != compile_protocol(
            CORPUS_DIR / "02_interchange_im.json", env=env
        ).digests.document
    )


def test_refusal_exits_nonzero(capsys, artifacts_root):
    code = main(
        _argv(
            "validate",
            "02_interchange_im.json",
            artifacts_root,
            "--set",
            "sites.target.layers=99",
        )
    )
    assert code == 1
    assert "refused" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# run-verb execution flags: --device / --dtype / --points
# --------------------------------------------------------------------------- #


class _CapturingEngine(Engine):
    """Stands in for the reference engine: records construction kwargs and
    the handoff (the compiled document and its run context), executes
    nothing."""

    last: "_CapturingEngine | None" = None

    name = "capture"
    capabilities = frozenset(
        {"grad", "paired_forward", "full_logits", "pytorch_fn_local"}
    )
    components = frozenset(COMPONENTS)
    writable_components = frozenset(COMPONENTS)
    is_local = True

    def __init__(
        self,
        *,
        device: str = "cpu",
        batch_rows: int | None = None,
        cuda_graphs: bool = False,
    ) -> None:
        self.device = device
        self.batch_rows = batch_rows
        self.cuda_graphs = cuda_graphs
        self.compiled = None
        self.run = None
        type(self).last = self

    def execute(self, compiled, run):
        from tests._helpers.stub_engine import stub_execute

        self.compiled = compiled
        self.run = run
        return stub_execute(self, compiled, run)


@pytest.fixture
def capturing_engine(monkeypatch):
    """Swap the lazily-imported reference engine module for the stub."""
    import sys as _sys
    import types

    stub = types.ModuleType("causalab.neural.engines.pytorch_hooks")
    stub.PytorchHooksEngine = _CapturingEngine
    monkeypatch.setitem(_sys.modules, "causalab.neural.engines.pytorch_hooks", stub)
    _CapturingEngine.last = None
    return _CapturingEngine


def _run_argv(name: str, artifacts_root, out, *extra: str) -> list[str]:
    return _argv("run", name, artifacts_root, "--out", str(out), *extra)


def test_device_goes_to_the_engine_and_dtype_goes_to_the_document(
    capturing_engine, artifacts_root, tmp_path
):
    """§8: placement is the engine's, precision is the document's. ``--dtype``
    is shorthand for ``--set model.dtype``, so an overridden run's digest is
    the overridden document's — the record cannot disagree with the numbers."""
    unoverridden = main(_argv("digest", "02_interchange_im.json", artifacts_root))
    assert unoverridden == 0
    code = main(
        _run_argv(
            "02_interchange_im.json",
            artifacts_root,
            tmp_path,
            "--device",
            "cuda:1",
            "--dtype",
            "bf16",
        )
    )
    assert code == 0
    assert capturing_engine.last.device == "cuda:1"
    assert not hasattr(capturing_engine.last, "dtype")
    compiled, run = capturing_engine.last.compiled, capturing_engine.last.run
    assert steps_of(compiled, run.env).canonical[0]["model"]["dtype"] == "bf16"


def test_device_is_recorded_in_the_receipt_and_not_in_the_document(
    capturing_engine, artifacts_root, tmp_path
):
    """§8: the receipt's ``execution.device`` is the placement the engine was
    built with, so a reader can tell a CUDA run from an MPS run. It is
    execution, so the canonical document and the digest are the unplaced
    run's."""
    code = main(
        _run_argv(
            "02_interchange_im.json",
            artifacts_root,
            tmp_path,
            "--device",
            "cuda:1",
            "--record",
        )
    )
    assert code == 0
    record = json.loads((tmp_path / "protocol.json").read_text())
    assert record["execution"]["device"] == "cuda:1"
    assert "cuda:1" not in json.dumps(record["canonical"])
    assert "cuda:1" not in json.dumps(record["points"])


def test_batch_rows_goes_to_the_engine_and_the_receipt_not_the_document(
    capturing_engine, artifacts_root, tmp_path, capsys
):
    """§8: the microbatch bound is execution, like placement — it reaches the
    engine's constructor and leaves the canonical document, so the digest is
    the unbounded run's. The receipt is its one recorder: ``execution``
    holds the bound the chosen engine reports, and nothing else does."""
    code = main(
        _run_argv(
            "02_interchange_im.json",
            artifacts_root,
            tmp_path,
            "--batch-rows",
            "4",
            "--record",
        )
    )
    assert code == 0
    assert capturing_engine.last.batch_rows == 4
    record = json.loads((tmp_path / "protocol.json").read_text())
    assert record["execution"] == {
        "batch_rows": 4,
        "device": "cpu",
        "fit_rows": None,
        "model_source": "loaded",
        "parallel": {
            "data": 1,
            "data_mode": "points",
            "pipeline": 1,
            "context": 1,
            "tensor": 1,
            "expert": 1,
            "world": 1,
            "launcher": "solo",
        },
    }
    assert "batch_rows" not in json.dumps(record["canonical"])
    assert "batch_rows" not in json.dumps(record["points"])
    capsys.readouterr()
    assert main(_argv("digest", "02_interchange_im.json", artifacts_root)) == 0
    assert capsys.readouterr().out.strip() == record["document_digest"]


def test_batch_rows_defaults_to_one_forward_per_group(
    capturing_engine, artifacts_root, tmp_path
):
    """No flag: the engine runs whole, and the receipt says so — ``null``,
    the same key, so a reader of two receipts compares one field."""
    assert (
        main(_run_argv("02_interchange_im.json", artifacts_root, tmp_path, "--record"))
        == 0
    )
    assert capturing_engine.last.batch_rows is None
    record = json.loads((tmp_path / "protocol.json").read_text())
    assert record["execution"] == {
        "batch_rows": None,
        "device": "cpu",
        "fit_rows": None,
        "model_source": "loaded",
        "parallel": {
            "data": 1,
            "data_mode": "points",
            "pipeline": 1,
            "context": 1,
            "tensor": 1,
            "expert": 1,
            "world": 1,
            "launcher": "solo",
        },
    }


def test_model_source_is_read_off_the_engine_into_the_receipt_only(
    capturing_engine, artifacts_root, tmp_path, monkeypatch
):
    """§8/§9: where the model came from is execution provenance like the row
    bound — the one recorder is the receipt's ``execution`` block, read off
    the engine (``loaded`` when it declares nothing), and it enters no
    canonical form and no point digest. The CLI has no bundle flag, so the
    caller value is exercised through the engine's own report here."""
    monkeypatch.setattr(capturing_engine, "model_source", "caller", raising=False)
    assert (
        main(_run_argv("02_interchange_im.json", artifacts_root, tmp_path, "--record"))
        == 0
    )
    record = json.loads((tmp_path / "protocol.json").read_text())
    assert record["execution"] == {
        "batch_rows": None,
        "device": "cpu",
        "fit_rows": None,
        "model_source": "caller",
        "parallel": {
            "data": 1,
            "data_mode": "points",
            "pipeline": 1,
            "context": 1,
            "tensor": 1,
            "expert": 1,
            "world": 1,
            "launcher": "solo",
        },
    }
    assert "model_source" not in json.dumps(record["canonical"])
    assert "model_source" not in json.dumps(record["points"])


@pytest.mark.parametrize("bad", ("0", "-2", "many"))
def test_batch_rows_refuses_a_non_positive_count(
    capturing_engine, artifacts_root, tmp_path, bad: str
):
    """argparse's own refusal (exit 2) — a bound of zero rows would run
    nothing, and a negative one means nothing."""
    with pytest.raises(SystemExit) as exit_info:
        main(
            _run_argv(
                "02_interchange_im.json", artifacts_root, tmp_path, "--batch-rows", bad
            )
        )
    assert exit_info.value.code == 2
    assert capturing_engine.last is None


def test_batch_rows_refuses_an_explicit_nnsight_pin(
    capturing_engine, artifacts_root, tmp_path, capsys
):
    """Fail closed: pinning the engine that has no bound while asking for one
    is refused at parse (exit 2, both flags named) instead of running whole
    and recording ``null`` — nothing would have honoured the bound."""
    with pytest.raises(SystemExit) as exit_info:
        main(
            _run_argv(
                "01_harvest_im.json",
                artifacts_root,
                tmp_path,
                "--engine",
                "nnsight",
                "--batch-rows",
                "3",
            )
        )
    assert exit_info.value.code == 2
    err = capsys.readouterr().err
    assert "--engine nnsight" in err and "--batch-rows" in err
    assert capturing_engine.last is None


@pytest.mark.parametrize("engine", ("pytorch_hooks", "auto"))
def test_batch_rows_runs_under_the_reference_engine_and_auto(
    capturing_engine, artifacts_root, tmp_path, engine: str
):
    """The refusal above fires on the explicit nnsight pin only: the bound
    with the reference engine pinned, or under ``auto``, runs as before and
    reaches the engine and the receipt."""
    code = main(
        _run_argv(
            "02_interchange_im.json",
            artifacts_root,
            tmp_path,
            "--engine",
            engine,
            "--batch-rows",
            "3",
            "--record",
        )
    )
    assert code == 0
    assert capturing_engine.last.batch_rows == 3
    record = json.loads((tmp_path / "protocol.json").read_text())
    assert record["execution"] == {
        "batch_rows": 3,
        "device": "cpu",
        "fit_rows": None,
        "model_source": "loaded",
        "parallel": {
            "data": 1,
            "data_mode": "points",
            "pipeline": 1,
            "context": 1,
            "tensor": 1,
            "expert": 1,
            "world": 1,
            "launcher": "solo",
        },
    }


def test_engine_auto_is_the_reference_engine_and_never_imports_nnsight(
    capturing_engine, artifacts_root, tmp_path, monkeypatch
):
    """`--engine auto` resolves to `pytorch_hooks` (the router's
    placeholder) and builds that engine alone: the nnsight extra being absent — or
    present — is never consulted."""
    import sys as _sys

    target = "causalab.neural.engines.nnsight_tracing"
    for mod in [m for m in list(_sys.modules) if m.startswith(target)]:
        monkeypatch.delitem(_sys.modules, mod)
    monkeypatch.setattr(_sys, "meta_path", [_AbsentModule(target), *_sys.meta_path])
    code = main(
        _run_argv("01_harvest_im.json", artifacts_root, tmp_path, "--engine", "auto")
    )
    assert code == 0
    assert capturing_engine.last is not None  # the stub standing in for pytorch_hooks


class _AbsentModule:
    """A meta-path finder that makes one module unimportable, so the
    not-installed path stays testable in an env that has the extra."""

    def __init__(self, name: str) -> None:
        self.name = name

    def find_spec(self, fullname, path=None, target=None):
        if fullname == self.name or fullname.startswith(self.name + "."):
            raise ModuleNotFoundError(f"No module named {fullname!r} (simulated)")
        return None


def test_engine_nnsight_refuses_by_name_when_not_installed(
    capturing_engine, artifacts_root, tmp_path, capsys, monkeypatch
):
    """Naming an engine that is not installed is an error that says which
    extra provides it."""
    import sys as _sys

    target = "causalab.neural.engines.nnsight_tracing"
    for mod in [m for m in list(_sys.modules) if m.startswith(target)]:
        monkeypatch.delitem(_sys.modules, mod)
    monkeypatch.setattr(_sys, "meta_path", [_AbsentModule(target), *_sys.meta_path])
    code = main(
        _run_argv("01_harvest_im.json", artifacts_root, tmp_path, "--engine", "nnsight")
    )
    assert code == 1
    err = capsys.readouterr().err
    assert "nnsight" in err and "extra" in err


def test_engine_nnsight_selects_the_nnsight_engine(
    capturing_engine, artifacts_root, tmp_path, monkeypatch
):
    """--engine nnsight builds the nnsight engine (stubbed here — the real
    one's answers are pinned by its parity suite)."""
    import sys as _sys
    import types

    class _CapturingNnsight(_CapturingEngine):
        name = "nnsight"

    stub = types.ModuleType("causalab.neural.engines.nnsight_tracing")
    stub.NnsightEngine = _CapturingNnsight
    monkeypatch.setitem(_sys.modules, "causalab.neural.engines.nnsight_tracing", stub)
    _CapturingNnsight.last = None
    code = main(
        _run_argv("01_harvest_im.json", artifacts_root, tmp_path, "--engine", "nnsight")
    )
    assert code == 0
    assert _CapturingNnsight.last is not None
    assert _CapturingNnsight.last.name == "nnsight"


def test_explain_engine_prints_the_engine_by_its_registry_name(artifacts_root, capsys):
    """`explain --engine auto` prints the engine the router resolved to, by
    the registry's name — no engine class is loaded (no `capturing_engine`
    fixture here: the verdict is the registry's capability set)."""
    code = main(
        _argv("explain", "02_interchange_im.json", artifacts_root, "--engine", "auto")
    )
    assert code == 0
    assert "engine    pytorch_hooks" in capsys.readouterr().out


def test_explain_engine_prints_the_refusal_rather_than_raising(artifacts_root, capsys):
    """A document the named engine cannot serve is the *more* useful answer
    of the two, so it is printed beside the plan instead of aborting the
    explanation: the nnsight engine has no grad path, and `04_das_im.json` is
    a fit."""
    code = main(
        _argv("explain", "04_das_im.json", artifacts_root, "--engine", "nnsight")
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "plan" in out  # the plan still prints
    assert "engine    refused: [V13]" in out
    assert "lacks ['grad']" in out  # names what is missing


def test_validate_refuses_a_document_the_named_engine_cannot_serve(
    artifacts_root, capsys
):
    """`validate --engine nnsight` on a fit is rule 13's refusal, exit 1 — the
    refusal that used to be routing's, now the named engine's; `auto`
    (pytorch_hooks) is the twin."""
    code = main(
        _argv("validate", "04_das_im.json", artifacts_root, "--engine", "nnsight")
    )
    assert code == 1
    err = capsys.readouterr().err
    assert "refused: [V13]" in err and "lacks ['grad']" in err
    assert main(_argv("validate", "04_das_im.json", artifacts_root)) == 0


@pytest.mark.parametrize("verb", ["run", "validate", "explain", "dry-run"])
def test_the_engine_flag_is_required(verb, artifacts_root, tmp_path, capsys):
    """The engine is a mandatory explicit input: no default, so a
    verb without `--engine` is argparse's exit 2 naming the flag."""
    argv = [
        verb,
        str(CORPUS_DIR / "02_interchange_im.json"),
        "--data-root",
        str(FIXTURES / "data"),
        "--artifacts-root",
        str(artifacts_root),
        *(("--out", str(tmp_path)) if verb == "run" else ()),
    ]
    with pytest.raises(SystemExit) as err:
        main(argv)
    assert err.value.code == 2
    assert "the following arguments are required: --engine" in capsys.readouterr().err


def test_auto_refuses_when_the_reference_engine_lacks_a_capability(
    capturing_engine, artifacts_root, tmp_path, capsys, monkeypatch
):
    """`auto` no longer falls through to the
    nnsight engine for a document `pytorch_hooks` cannot serve — it is refused
    under rule 13 naming the shortfall, before any weights, and the nnsight
    module is never constructed; the user pins `--engine nnsight`. The
    shortfall is staged by emptying the stub's capability set: no offline
    document needs a component only nnsight serves (the deltanet and
    expert_permutation faces), so there is no real-document twin."""
    import sys as _sys
    import types

    class _Nnsight(_CapturingEngine):
        name = "nnsight"
        capabilities = frozenset(_CapturingEngine.capabilities)

    monkeypatch.setattr(capturing_engine, "capabilities", frozenset())
    stub = types.ModuleType("causalab.neural.engines.nnsight_tracing")
    stub.NnsightEngine = _Nnsight
    monkeypatch.setitem(_sys.modules, "causalab.neural.engines.nnsight_tracing", stub)
    _Nnsight.last = None

    code = main(_run_argv("02_interchange_im.json", artifacts_root, tmp_path))
    assert code == 1
    err = capsys.readouterr().err
    assert "refused: [V13]" in err and "paired_forward" in err
    assert _Nnsight.last is None
    assert capturing_engine.last is not None  # constructed, then refused
    assert capturing_engine.last.compiled is None  # and never entered


def test_run_defaults_stay_cpu_fp32(capturing_engine, artifacts_root, tmp_path):
    assert main(_run_argv("02_interchange_im.json", artifacts_root, tmp_path)) == 0
    assert capturing_engine.last.device == "cpu"
    last = capturing_engine.last
    assert (
        steps_of(last.compiled, last.run.env).canonical[0]["model"]["dtype"] == "fp32"
    )


def test_dtype_and_set_may_not_contradict(artifacts_root, tmp_path):
    with pytest.raises(SystemExit) as err:
        main(
            _run_argv(
                "02_interchange_im.json",
                artifacts_root,
                tmp_path,
                "--dtype",
                "bf16",
                "--set",
                "model.dtype=fp16",
            )
        )
    assert "contradicts" in str(err.value)


def test_points_selects_a_shard_without_moving_the_campaign_digest(
    capturing_engine, env, artifacts_root, tmp_path
):
    loaded = compile_protocol(CORPUS_DIR / "07_weekdays_locate_scan_im.json", env=env)
    code = main(
        _run_argv(
            "07_weekdays_locate_scan_im.json",
            artifacts_root,
            tmp_path,
            "--points",
            "3:7",
        )
    )
    assert code == 0
    compiled, run = capturing_engine.last.compiled, capturing_engine.last.run
    # the shard rides on the run context as indices; the engine reads the
    # compiled points at those indices, and the campaign digest is untouched
    assert run.points == (3, 4, 5, 6)
    assert [steps_of(compiled, env).digests[i] for i in run.points] == list(
        steps_of(loaded, env).digests[3:7]
    )
    assert [steps_of(compiled, env).points[i].coords for i in run.points] == [
        p.coords for p in steps_of(loaded, env).points[3:7]
    ]
    assert compiled.campaign_digest == loaded.digests.document


@pytest.mark.parametrize("spec", ["7", "3:3", "60:70", "-1:4", "a:b"])
def test_points_refuses_malformed_and_out_of_range(
    capturing_engine, artifacts_root, tmp_path, capsys, spec
):
    # the = form keeps argparse from reading a leading "-" as a flag
    code = main(
        _run_argv(
            "07_weekdays_locate_scan_im.json",
            artifacts_root,
            tmp_path,
            f"--points={spec}",
        )
    )
    assert code == 1
    assert "refused" in capsys.readouterr().err


def test_points_refused_on_workflow_documents(
    capturing_engine, artifacts_root, tmp_path, capsys
):
    doc = tmp_path / "wf.json"
    doc.write_text(json.dumps({"version": "1", "steps": {}}))
    code = main(
        [
            "run",
            "--engine",
            "auto",
            str(doc),
            "--data-root",
            str(FIXTURES / "data"),
            "--artifacts-root",
            str(artifacts_root),
            "--out",
            str(tmp_path / "out"),
            "--points",
            "0:1",
        ]
    )
    assert code == 1
    err = capsys.readouterr().err
    assert "refused" in err and "workflow" in err


# --------------------------------------------------------------------------- #
#  column positions and match modes are checked like any other reference      #
# --------------------------------------------------------------------------- #


def test_validate_data_flags_a_missing_position_column(env):
    """A ``{"column": …}`` position is an explicit reference, so
    ``validate --data`` catches a typo at load instead of the engine hitting
    it mid-run (§2.3)."""
    loaded = compile_protocol(CORPUS_DIR / "10_task_table_iia_im.json", env=env)
    raw = json.loads(json.dumps(dict(loaded.tree)))
    raw["method"]["positions"]["subject"] = {"column": "not_a_column"}
    with pytest.raises(Exception) as err:
        check_data_columns(compile_protocol(raw, env=env), env)
    assert "not_a_column" in str(err.value)


def test_validate_data_accepts_the_generated_tables_columns(env):
    """The positive half: every reference in the task-table document resolves
    against the built table, including the answer-form group column."""
    loaded = compile_protocol(CORPUS_DIR / "10_task_table_iia_im.json", env=env)
    refs = check_data_columns(loaded, env)
    assert "label_forms" in refs  # the metric's expected group
    assert "entity" in refs  # the column position


def test_validate_data_flags_a_missing_position_variable(env):
    """The false green this closes. ``weekdays_locate_scan`` swept its tap over
    ``{"variable": "subject"}``, no task-generated table has a ``subject``, and
    ``validate --data`` still printed ``OK … 64 points`` — the whole point of
    the pure verbs being a load-error checklist. 32 of those points then died
    mid-campaign with ``[P2] no value for prompt variable 'subject'``.

    Existence is answerable without a tokenizer, so it is answered here; the
    *width* of the resulting window is not, and stays a run-time refusal."""
    loaded = compile_protocol(CORPUS_DIR / "07_weekdays_locate_scan_im.json", env=env)
    raw = json.loads(json.dumps(dict(loaded.tree)))
    # the axis exactly as it shipped, bad coordinate second
    raw["method"]["positions"]["tap"] = {
        "sweep": [{"index": -1}, {"variable": "subject"}]
    }
    with pytest.raises(Exception) as err:
        check_data_columns(compile_protocol(raw, env=env), env)
    assert "subject" in str(err.value)
    assert "prompt variable" in str(err.value)


def test_validate_data_checks_every_point_not_just_the_first(env):
    """The structural half of the same false green: this pass read
    ``point_documents[0]``, so a swept axis was only ever checked at coordinate
    0. Any reference that varies with the sweep — a position, a metric column,
    a dataset field — was unchecked at every other coordinate."""
    loaded = compile_protocol(CORPUS_DIR / "07_weekdays_locate_scan_im.json", env=env)
    raw = json.loads(json.dumps(dict(loaded.tree)))
    _aggregation(raw, "iia")["expected"] = {"sweep": ["cf_answer", "not_a_column"]}
    with pytest.raises(Exception) as err:
        check_data_columns(compile_protocol(raw, env=env), env)
    assert "not_a_column" in str(err.value)


def test_validate_data_accepts_a_variable_only_the_sibling_names(env):
    """A prompt variable is per-role (§2.3): the counterfactual role's value
    for ``entity`` lives in ``counterfactual_inputs_variables``, not in a
    top-level column of that name. Resolving one spelling but not the other
    would refuse every real task table."""
    refs = check_data_columns(
        compile_protocol(CORPUS_DIR / "07_weekdays_locate_scan_im.json", env=env), env
    )
    assert "entity" in refs


def test_validate_data_reads_a_drawn_roles_sibling_at_its_eval_member(env, monkeypatch):
    """§2.2 ``draw``: the loader's prompt-variable check asks the role for the
    field the forward reads — `counterfactual_inputs[0]`, not the bare column
    — so the per-member `_variables` sibling is read as the engine reads it.
    Before the change the bare field skipped the sibling and a variable that
    lived only there was a false rule-4 refusal. The shipped fixture cannot
    exhibit that refusal (its `entity` is also a top-level column), so the
    test records the field the check is handed rather than asserting a
    refusal; `"entity" in refs` holds either way."""
    from causalab.protocol.rules import data as loader_module

    loaded = compile_protocol(CORPUS_DIR / "07_weekdays_locate_scan_im.json", env=env)
    raw = json.loads(json.dumps(dict(loaded.tree)))
    raw["data"]["counterfactual"] = {
        **raw["data"]["counterfactual"],
        "field": "counterfactual_inputs",
        "draw": {"kind": "uniform"},
    }
    asked: list[str] = []
    real = loader_module._role_variables

    def recording(rows, field):
        asked.append(field)
        return real(rows, field)

    monkeypatch.setattr(loader_module, "_role_variables", recording)
    refs = check_data_columns(compile_protocol(raw, env=env), env)
    assert "counterfactual_inputs[0]" in asked and "counterfactual_inputs" not in asked
    assert "entity" in refs


def test_validate_data_flags_a_missing_scope_variable(env):
    """``scope``/``relative_to`` spelled as a variable is the same reference,
    and the ROME-shaped ``{"index": -1, "scope": {"variable": …}}`` idiom is
    where it is actually written."""
    loaded = compile_protocol(CORPUS_DIR / "07_weekdays_locate_scan_im.json", env=env)
    raw = json.loads(json.dumps(dict(loaded.tree)))
    raw["method"]["positions"]["tap"] = {
        "index": -1,
        "scope": {"variable": "not_a_variable"},
    }
    with pytest.raises(Exception) as err:
        check_data_columns(compile_protocol(raw, env=env), env)
    assert "not_a_variable" in str(err.value)


def test_validate_data_flags_a_missing_relative_to_column(env):
    loaded = compile_protocol(CORPUS_DIR / "10_task_table_iia_im.json", env=env)
    raw = json.loads(json.dumps(dict(loaded.tree)))
    raw["method"]["positions"]["subject"] = {
        "index": 1,
        "relative_to": {"column": "not_a_column"},
    }
    with pytest.raises(Exception) as err:
        check_data_columns(compile_protocol(raw, env=env), env)
    assert "not_a_column" in str(err.value)


def test_explain_reports_the_decode_and_what_it_obliges(capsys, artifacts_root):
    """A generate document's cost is legible before it runs: how far it
    decodes, and which reads oblige a vocabulary tensor."""
    assert main(_argv("explain", "11_probe_generate_im.json", artifacts_root)) == 0
    out = capsys.readouterr().out
    assert "generate" in out
    assert "decode 8 tokens (greedy)" in out
    assert "tail at lm_head: distribution per addressed position" in out


# --------------------------------------------------------------------------- #
# the run receipt (§9)
# --------------------------------------------------------------------------- #


REPO = Path(__file__).resolve().parents[2]
SHIPPED_RUN = PROTOCOLS_DIR / "weekdays_interchange.json"


def _file_argv(verb: str, path, artifacts_root, *extra: str) -> list[str]:
    return [
        verb,
        str(path),
        "--data-root",
        str(FIXTURES / "data"),
        "--artifacts-root",
        str(artifacts_root),
        *(() if verb == "digest" else ("--engine", "auto")),  # the flag is required
        *extra,
    ]


# --------------------------------------------------------------------------- #
#  --register-from-hf: pre-flighting a document on an unregistered model        #
# --------------------------------------------------------------------------- #


class _StubConfig:
    """Just the attributes [`model_info_from_hf_config`][causalab.protocol.registry.models.model_info_from_hf_config] reads."""

    num_attention_heads = 8
    hidden_size = 64
    num_hidden_layers = 40
    num_key_value_heads = 8
    head_dim = 8
    intermediate_size = 128
    vocab_size = 512
    dtype = "bfloat16"


def _unregistered_key(request) -> str:
    """A key unique to the calling test.

    ``register_model`` writes to a process-global registry, so a shared key
    would leak: whichever test registered it first would make the others'
    "still refuses" assertion vacuous.
    """
    return f"some-org/never-registered-40L-{abs(hash(request.node.name)):x}"


@pytest.fixture
def unregistered_document(tmp_path, request):
    """Corpus 02 retargeted at a key the registry has never heard of."""
    raw = json.loads((CORPUS_DIR / "02_interchange_im.json").read_text())
    raw["model"]["key"] = _unregistered_key(request)
    path = tmp_path / "unregistered_im.json"
    path.write_text(json.dumps(raw, indent=2))
    return path


def _verb(verb: str, path: Path, artifacts_root, *extra: str) -> list[str]:
    return [
        verb,
        str(path),
        "--data-root",
        str(FIXTURES / "data"),
        "--artifacts-root",
        str(artifacts_root),
        # `digest` alone takes no engine
        *(() if verb == "digest" else ("--engine", "auto")),
        *extra,
    ]


@pytest.mark.parametrize("verb", ["validate", "explain", "digest"])
def test_a_pure_verb_refuses_an_unregistered_model_without_the_flag(
    verb, unregistered_document, artifacts_root, capsys
):
    """The invariant: no flag, no network — so the refusal stands."""
    code = main(_verb(verb, unregistered_document, artifacts_root))
    assert code == 1
    assert "[V4]" in capsys.readouterr().err


@pytest.mark.parametrize("verb", ["validate", "explain", "digest"])
def test_register_from_hf_lets_a_pure_verb_pre_flight_an_unregistered_model(
    verb, unregistered_document, artifacts_root, capsys, monkeypatch
):
    """The gap a run on an unregistered model had to hand-roll a wrapper around.

    The documented workaround — validate against a *similar* registered model —
    produces a **false** refusal: `[V4] layer 36 out of range for the 36-layer
    model 'Qwen/Qwen3-4B-Instruct-2507'` on a perfectly valid 40-layer
    document. Pre-flighting has to be possible on the model the document names.
    """
    import transformers

    monkeypatch.setattr(
        transformers.AutoConfig,
        "from_pretrained",
        classmethod(lambda cls, key, **kw: _StubConfig()),
    )
    code = main(
        _verb(verb, unregistered_document, artifacts_root, "--register-from-hf")
    )
    assert code == 0, capsys.readouterr().err


def test_register_from_hf_pre_registers_every_inner_model_of_a_workflow(
    tmp_path, artifacts_root, monkeypatch, capsys, request
):
    """A workflow names several documents, so registering only the outer one
    would pre-flight nothing — which is why the runs' wrappers were
    workflow-aware."""
    import transformers

    seen: list[str] = []

    def fake(cls, key, **kw):
        seen.append(key)
        return _StubConfig()

    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", classmethod(fake))
    raw = json.loads((CORPUS_DIR / "02_interchange_im.json").read_text())
    key = _unregistered_key(request)
    raw["model"]["key"] = key
    inner = tmp_path / "inner_im.json"
    inner.write_text(json.dumps(raw, indent=2))
    workflow = tmp_path / "wf.json"
    workflow.write_text(
        json.dumps(
            {
                "version": "1",
                "output_dir": "out",
                "steps": {
                    "only": {
                        "type": "intervention_protocol",
                        "document": str(inner),
                    }
                },
            },
            indent=2,
        )
    )
    code = main(_verb("validate", workflow, artifacts_root, "--register-from-hf"))
    assert code == 0, capsys.readouterr().err
    assert key in seen


def test_explain_prints_one_digest(capsys, artifacts_root):
    """`explain` prints the document digest and no second one (§7): the method
    digest is gone, so the only `digest` line is the campaign's."""
    assert main(_file_argv("explain", SHIPPED_RUN, artifacts_root)) == 0
    out = capsys.readouterr().out
    digest_lines = [line for line in out.splitlines() if line.startswith("digest")]
    assert len(digest_lines) == 1 and len(digest_lines[0].split()[1]) == 64
    assert not any(line.startswith("method") for line in out.splitlines())
    assert "bf16" in out


def test_run_writes_no_receipt_and_no_stream_by_default(
    capturing_engine, artifacts_root, tmp_path
):
    """``causalab run`` writes the saved tables and nothing beside them unless
    ``--record`` asks for the receipt and the event stream."""
    assert main(_run_argv("09_das_apply_im.json", artifacts_root, tmp_path)) == 0
    assert capturing_engine.last is not None
    assert not (tmp_path / "protocol.json").exists()
    assert not (tmp_path / "events.jsonl").exists()


def test_run_writes_the_protocol_record(capturing_engine, artifacts_root, tmp_path):
    """The record a reproducer reads first: what ran, at what precision, with
    the provenance digest of every point. Corpus 09 — a bf16 apply, one point
    — and not the shipped ``weekdays_interchange.json``: the run door loads
    the document's tokenizer, and the shipped weekdays presets name a gated
    model the CPU tier only ever loads offline, never runs."""
    assert (
        main(_run_argv("09_das_apply_im.json", artifacts_root, tmp_path, "--record"))
        == 0
    )
    record = json.loads((tmp_path / "protocol.json").read_text())
    assert record["canonical"]["model"] == {
        "key": "Qwen/Qwen3-8B",
        "revision": "main",
        "dtype": "bf16",
    }
    assert "method" not in record  # no method digest (§7)
    assert "title" not in record["canonical"]["header"]  # authoring metadata (§7)
    assert [point["index"] for point in record["points"]] == [0]
    assert record["points"][0]["digest"] == record["document_digest"]


# --------------------------------------------------------------------------- #
# dry-run: the verb beside the other pure verbs (the suite is test_dry_run.py)
# --------------------------------------------------------------------------- #


def test_dry_run_reports_and_ends_with_the_undecided_line(capsys, artifacts_root):
    """`dry-run` is a pure verb: exit 0 on a valid document, the plan's
    counts printed, and the last line is always the `undecided` list. On the
    base the verb does not exist (argparse exit 2)."""
    code = main(_argv("dry-run", "02_interchange_im.json", artifacts_root))
    assert code == 0
    out = capsys.readouterr().out
    assert "points    1" in out and "forwards" not in out  # the count is the run's
    assert (
        out.strip()
        .splitlines()[-1]
        .startswith("undecided (decided when the run encodes its inputs): ")
    )


def test_dry_run_refuses_with_the_reason_code(capsys, artifacts_root):
    """An unavailable site: `validate`'s `refused:` line plus the record with
    the reason code, exit 1."""
    code = main(
        _argv(
            "dry-run",
            "02_interchange_im.json",
            artifacts_root,
            "--set",
            "sites.target.component=routed_output",
        )
    )
    assert code == 1
    err = capsys.readouterr().err
    assert "refused: [V4]" in err and "reason component_unavailable" in err


def test_dry_run_decides_the_engine_question_from_the_registry(artifacts_root, capsys):
    """No `capturing_engine` fixture here: `--engine auto` (required) is
    answered from the registry's capability set for the resolved
    name, so no engine class loads and the engine question is never left
    undecided."""
    assert main(_argv("dry-run", "02_interchange_im.json", artifacts_root)) == 0
    out = capsys.readouterr().out
    assert "engine    pytorch_hooks: serves" in out
    assert not out.strip().splitlines()[-1].split(": ", 1)[1].startswith("engines")


def test_dry_run_engine_reports_the_shortfall_rather_than_raising(
    artifacts_root, capsys
):
    """The seam `explain --engine` left for the dry run: the refusal is a
    `capability_shortfall` for the named engine, printed beside the report;
    the named engine's shortfall is exit 1 (the nnsight engine has no grad
    path; `04_das_im.json` is a fit)."""
    code = main(
        _argv("dry-run", "04_das_im.json", artifacts_root, "--engine", "nnsight")
    )
    assert code == 1
    captured = capsys.readouterr()
    assert "engine    nnsight: capability_shortfall" in captured.out
    assert "lacks ['grad']" in captured.out
    assert "refused: [V13]" in captured.err


def test_dry_run_engine_that_serves_is_exit_0(artifacts_root, capsys):
    code = main(_argv("dry-run", "04_das_im.json", artifacts_root, "--engine", "auto"))
    assert code == 0
    assert "engine    pytorch_hooks: serves" in capsys.readouterr().out


def test_dry_run_refuses_register_from_hf(capsys, artifacts_root):
    """The flag is not inherited: a dry run never fetches a config."""
    code = main(
        _argv(
            "dry-run",
            "02_interchange_im.json",
            artifacts_root,
            "--set",
            "model.key=nobody/no-such-model",
            "--register-from-hf",
        )
    )
    assert code == 1
    assert "refused: [P4] --register-from-hf" in capsys.readouterr().err


def test_cuda_graph_option_reaches_engine_without_changing_document(
    capturing_engine, artifacts_root, tmp_path
):
    assert (
        main(_run_argv("02_interchange_im.json", artifacts_root, tmp_path / "eager"))
        == 0
    )
    last = capturing_engine.last
    canonical = steps_of(last.compiled, last.run.env).canonical
    assert not capturing_engine.last.cuda_graphs
    assert (
        main(
            _run_argv(
                "02_interchange_im.json",
                artifacts_root,
                tmp_path / "graph",
                "--device",
                "cuda",
                "--cuda-graphs",
            )
        )
        == 0
    )
    assert capturing_engine.last.cuda_graphs
    last = capturing_engine.last
    assert steps_of(last.compiled, last.run.env).canonical == canonical


# --------------------------------------------------------------------------- #
# run-verb --verbose: one logger, one handler, nothing else configured
# --------------------------------------------------------------------------- #


@pytest.fixture
def restore_loggers():
    """Snapshot the two loggers ``--verbose`` touches and put them back, so a
    verbose run leaves no handler or level behind for the tests after it."""
    import logging

    from causalab.cli import VERBOSE_LOGGER

    ours = logging.getLogger(VERBOSE_LOGGER)
    hf = logging.getLogger("huggingface_hub")
    before = (ours.level, list(ours.handlers), hf.level)
    yield ours
    ours.setLevel(before[0])
    ours.handlers[:] = before[1]
    hf.setLevel(before[2])


def test_verbose_enables_the_execution_logger_alone(
    capturing_engine, artifacts_root, tmp_path, restore_loggers
):
    """``--verbose`` raises the shared execution loop's logger to INFO with
    one stderr handler. The root logger and the ``causalab`` parent stay as
    they were: no other module gains output from the flag."""
    import logging

    root_level, root_handlers = logging.root.level, list(logging.root.handlers)
    parent = logging.getLogger("causalab")
    parent_level, parent_handlers = parent.level, list(parent.handlers)

    code = main(_run_argv("02_interchange_im.json", artifacts_root, tmp_path, "-v"))
    assert code == 0
    assert restore_loggers.isEnabledFor(logging.INFO)
    assert len(restore_loggers.handlers) == 1
    assert isinstance(restore_loggers.handlers[0], logging.StreamHandler)
    assert (logging.root.level, logging.root.handlers) == (root_level, root_handlers)
    assert (parent.level, parent.handlers) == (parent_level, parent_handlers)
    assert logging.getLogger("huggingface_hub").isEnabledFor(logging.INFO)


def test_verbose_twice_attaches_one_handler(
    capturing_engine, artifacts_root, tmp_path, restore_loggers
):
    """A second verbose run in the same process adds no second handler, so
    each line prints once."""
    for out in (tmp_path / "a", tmp_path / "b"):
        assert main(_run_argv("02_interchange_im.json", artifacts_root, out, "-v")) == 0
    assert len(restore_loggers.handlers) == 1


def test_run_without_verbose_configures_no_logger(
    capturing_engine, artifacts_root, tmp_path, restore_loggers
):
    import logging

    assert main(_run_argv("02_interchange_im.json", artifacts_root, tmp_path)) == 0
    assert restore_loggers.level == logging.NOTSET
    assert restore_loggers.handlers == []
