"""Real CLI, source wheels, child processes and a completely offline tiny model."""

import json
from pathlib import Path
import subprocess
import sys

import pytest

from causalab.cli import main
from causalab.protocol.schema import inline_train_saves

# `measurement_study`: every test here builds and installs a source wheel and
# spawns cold worker processes — minutes each. `-m "not measurement_study"`
# deselects them for a quick run (docs/TESTS.md lists the markers).
pytestmark = [pytest.mark.smoke, pytest.mark.measurement_study]


def _local_inputs(tmp_path, monkeypatch):
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    checkpoint = tmp_path / "checkpoint"
    config = GPT2Config(
        vocab_size=5,
        n_positions=16,
        n_embd=8,
        n_layer=1,
        n_head=2,
        bos_token_id=0,
        eos_token_id=0,
    )
    with torch.random.fork_rng():
        torch.manual_seed(0)
        model = GPT2LMHeadModel(config)
    model.save_pretrained(checkpoint)
    tokens = Tokenizer(
        WordLevel({"[UNK]": 0, "[PAD]": 1, "a": 2, "b": 3, "c": 4}, unk_token="[UNK]")
    )
    tokens.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=tokens,
        unk_token="[UNK]",
        pad_token="[PAD]",
        padding_side="left",
    ).save_pretrained(checkpoint)
    data = tmp_path / "data"
    data.mkdir()
    (data / "prompts.json").write_text(
        json.dumps([{"input": "a b", "split": "all"}, {"input": "b c", "split": "all"}])
    )
    return checkpoint, data


@pytest.mark.parametrize("profile", [True, False])
def test_isolated_arms_operation_and_cold_workflow(tmp_path, monkeypatch, profile):
    checkpoint, data = _local_inputs(tmp_path, monkeypatch)
    protocol = {
        "header": {"protocol_version": "4"},
        "model": {"key": str(checkpoint), "revision": "local", "dtype": "fp32"},
        "data": {"base": {"dataset": "prompts", "field": "input"}},
        "method": {
            "intervened_models": {"original": {"input": "base", "reads": ["logits"]}},
            "sites": {"head": {"component": "lm_head"}},
            "reads": {"logits": {"site": "head", "pos": -1}},
            "save": [
                {
                    "read": "logits",
                    "model": "original",
                    "file_path": "logits.safetensors",
                }
            ],
        },
    }
    (tmp_path / "inference.json").write_text(json.dumps(protocol))
    repository = Path(__file__).resolve().parents[3]
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository, text=True
    ).strip()
    study = {
        "version": "1",
        "output_dir": "research",
        "steps": {
            "inference": {"type": "intervention_protocol", "document": "inference.json"}
        },
        "measurement": {
            "version": 1,
            "arms": {
                "before": {"revision": revision, "execution": {"batch_rows": 1}},
                "after": {"revision": revision, "execution": {"batch_rows": 2}},
                "eager": {"revision": revision, "execution": {"batch_rows": 1}},
            },
            "seeds": [0],
            "repeats": 1,
            "warmups": 0,
            "bootstrap_draws": 100,
            "cases": {
                "operation": {"kind": "operation", "step": "inference"},
                "workflow": {"kind": "workflow", "cold_process": True},
            },
            "observations": {
                "logits": {
                    "step": "inference",
                    "file": "logits.safetensors",
                    "kind": "tensor",
                }
            },
            "profile": {"cases": ["operation", "workflow"], "record_shapes": True}
            if profile
            else False,
        },
    }
    document = tmp_path / "study.json"
    document.write_text(json.dumps(study))
    bindings = tmp_path / "bindings.json"
    bindings.write_text(
        json.dumps(
            {
                "arms": {
                    arm: {"repository": str(repository), "python": sys.executable}
                    for arm in ("before", "after", "eager")
                },
                "device": "cpu",
                "data_root": str(data),
                "artifacts_root": str(tmp_path),
            }
        )
    )
    output = tmp_path / "run"
    from causalab.measurement.study import controller as measurement_experiment

    real_build = measurement_experiment.build_arm

    def failed_build(repository, revision, destination, *, python):
        destination.mkdir(parents=True)
        (destination / "partial-wheel").write_text("interrupted build")
        raise RuntimeError("deployment failure")

    with monkeypatch.context() as patch:
        patch.setattr(measurement_experiment, "build_arm", failed_build)
        with pytest.raises(RuntimeError, match="deployment failure"):
            measurement_experiment.run(document, bindings, output)
    assert measurement_experiment.build_arm is real_build
    assert (output / "preparation.json").is_file()
    assert (
        main(
            [
                "measure",
                str(document),
                "--bindings",
                str(bindings),
                "--out",
                str(output),
                "--resume",
            ]
        )
        == 0
    )
    for case in ("operation", "workflow"):
        report = json.loads((output / "reports" / f"{case}.json").read_text())
        assert report["observations"]
        assert (output / "reports" / f"{case}.html").is_file()
        for arm in ("before", "after"):
            source = report["sources"][arm]
            assert source["context"]["worker"]["source_commit"] == revision
            implementation = source["provenance"]["implementation"]
            assert Path(implementation["location"]).is_relative_to(
                output / "deployment" / arm
            )
            assert report["traces"][arm]["status"] == "not_requested"
            captures = report["captures"][arm]
            assert len(captures) == ((2 if case == "workflow" else 1) if profile else 0)
            if profile:
                assert {capture["mode"] for capture in captures} == (
                    {"cold", "warm"} if case == "workflow" else {"warm"}
                )
                for capture in captures:
                    assert capture["status"] == "completed", capture
                    assert capture["options"]["record_shapes"] is True
                    assert capture["observation_check"]["status"] == "compared"
                    assert capture["observation_check"]["exactly_equal"]
                    assert capture["coverage"]["warmups"] == (capture["mode"] == "warm")
        if case == "workflow":
            assert report["scope"].startswith("cold-process")
            cold = json.loads((output / "collections/workflow.before.json").read_text())
            sample = cold["samples"][0]
            assert sample["observations"]["file"].endswith("/cold.safetensors")
            assert (
                sample["resident_observations"]["file"]
                != sample["observations"]["file"]
            )
            assert (
                output / "collections" / sample["resident_observations"]["file"]
            ).is_file()
        for candidate in ("before", "after"):
            control = json.loads(
                (output / "reports" / f"{case}.eager_{candidate}.json").read_text()
            )
            assert control["contrast"] == {
                "reference": "eager",
                "candidate": candidate,
            }
            eager = control["sources"]["before"]["provenance"]["implementation"]
            assert Path(eager["location"]).is_relative_to(output / "deployment/eager")
            assert control["traces"]["before"]["status"] == "not_requested"
            assert bool(control["captures"]["before"]) == profile
            assert control["observations"]
    # Native wheels must target the executing Python and include the extension
    # in the verified manifest.
    from causalab.measurement.deployment.installation import interpreter_identity

    for arm in ("before", "after", "eager"):
        installation = json.loads(
            (output / "deployment" / arm / "installation.json").read_text()
        )
        assert installation["build_runtime"] == interpreter_identity(sys.executable)
        assert any(
            name.endswith((".so", ".pyd")) for name in installation["installed_files"]
        )
    assert (
        len(
            list(
                (output / "collections/blocks").glob(
                    "*/before/cold/research/workflow.json"
                )
            )
        )
        == 1
    )
    blocks = {
        str(path.relative_to(output)): path.read_bytes()
        for path in (output / "collections/blocks").rglob("block.json")
    }
    worker = report["sources"]["before"]["context"]["worker"]
    assert worker["dispatch"]["status"] == "observed_untimed_probe"
    assert set(worker["execution_probe"]["cases"]) == {"operation", "workflow"}
    assert worker["execution_probe"]["cases"]["operation"]["model_inputs"]
    for probe in worker["execution_probe"]["cases"].values():
        assert probe["native_profile"]["status"] == (
            "completed" if profile else "not_requested"
        )
        if not profile:
            assert probe["operators"] == probe["device_kernels"] == []
    if not profile:
        assert not list(output.rglob("trace.json"))
    assert (
        main(
            [
                "measure",
                str(document),
                "--bindings",
                str(bindings),
                "--out",
                str(output),
                "--resume",
            ]
        )
        == 0
    )
    assert all(
        (output / path).read_bytes() == contents for path, contents in blocks.items()
    )
    for arm in ("before", "after", "eager"):
        assert len(list((output / "execution-probes" / arm).glob("*/probe.json"))) == 2


@pytest.mark.parametrize("cold_process", [False, True])
def test_das_and_dbm_use_actual_fitting_seeds_and_typed_observations(
    tmp_path, monkeypatch, cold_process
):
    checkpoint, data = _local_inputs(tmp_path, monkeypatch)
    repository = Path(__file__).resolve().parents[3]
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository, text=True
    ).strip()
    rows = []
    for index, (base, source, split) in enumerate(
        [
            ("a b", "a c", "train"),
            ("b c", "b b", "train"),
            ("c b", "c c", "test"),
            ("c a", "c c", "test"),
        ]
    ):
        rows.append(
            {
                "input": base,
                "counterfactual_inputs": [source],
                "split": split,
                "label": source[-1],
                "base_answer": base[-1],
                "cf_answer": source[-1],
                "id": index,
            }
        )
    (data / "pairs.json").write_text(json.dumps(rows))
    observations = {}
    steps = {}
    for method, feature in (("das", "rot"), ("dbm", "gate")):
        protocol = json.loads(
            (repository / f"demos/methods/protocols/{method}.json").read_text()
        )
        protocol["model"] = {
            "key": str(checkpoint),
            "revision": "local",
            "dtype": "fp32",
        }
        protocol["data"]["base"]["dataset"] = "pairs#train"
        protocol["data"]["counterfactual"]["dataset"] = "pairs#train"
        protocol["method"]["sites"]["target"]["layers"] = [0]
        train = protocol["method"]["train"]
        train["steps"] = {"updates": 1}
        train["batch"] = {"pairs": 2}
        train["eval"]["split"] = "pairs#test"
        train.pop("early_stop", None)
        train.pop("anneal", None)
        if method == "das":
            protocol["method"]["featurizers"][feature]["k"] = 2
        (tmp_path / f"{method}.json").write_text(json.dumps(protocol))
        steps[method] = {"type": "intervention_protocol", "document": f"{method}.json"}
        observations[feature] = {
            "step": method,
            "file": f"{feature}.safetensors",
            "kind": "subspace" if method == "das" else "gate",
        }
        observations[f"{method}_effect"] = {
            "step": method,
            "file": "iia.json",
            "kind": "table",
            "row_keys": ["example_id", "metric"],
            "value": "value",
        }
        replay = json.loads(json.dumps(protocol))
        replay["method"]["save"] = [
            save
            for save in inline_train_saves(replay["method"])
            if save.get("value") != feature
        ]
        replay["method"].pop("train")
        replay["method"]["featurizers"][feature]["file_path"] = (
            f"{method}/{feature}.safetensors"
        )
        replay["data"]["base"]["dataset"] = "pairs#test"
        replay["data"]["counterfactual"]["dataset"] = "pairs#test"
        evaluation_step = f"{method}_eval"
        (tmp_path / f"{evaluation_step}.json").write_text(json.dumps(replay))
        steps[evaluation_step] = {
            "type": "intervention_protocol",
            "document": f"{evaluation_step}.json",
        }
        observations[f"{evaluation_step}_effect"] = {
            "step": evaluation_step,
            "file": "iia.json",
            "kind": "table",
            "row_keys": ["example_id", "metric"],
            "value": "value",
        }
    study = {
        "version": "1",
        "output_dir": "research",
        "steps": steps,
        "measurement": {
            "version": 1,
            "arms": {arm: {"revision": revision} for arm in ("before", "after")},
            "seeds": [7, 8],
            "repeats": 2,
            "warmups": 0,
            "bootstrap_draws": 100,
            "cases": {"training": {"kind": "workflow", "cold_process": cold_process}},
            "observations": observations,
            "profile": {"cases": ["training"]},
            "evaluation": {
                "arm": "before",
                "crossed": True,
                "cases": {"training": ["das_eval", "dbm_eval"]},
            },
        },
    }
    document, bindings = tmp_path / "study.json", tmp_path / "bindings.json"
    document.write_text(json.dumps(study))
    bindings.write_text(
        json.dumps(
            {
                "arms": {
                    arm: {"repository": str(repository), "python": sys.executable}
                    for arm in ("before", "after")
                },
                "device": "cpu",
                "data_root": str(data),
                "artifacts_root": str(tmp_path),
            }
        )
    )
    output = tmp_path / "run"
    assert (
        main(
            [
                "measure",
                str(document),
                "--bindings",
                str(bindings),
                "--out",
                str(output),
            ]
        )
        == 0
    )
    report = json.loads((output / "reports/training.json").read_text())
    pairing = report["training"]
    for arm in ("before", "after"):
        collection = json.loads(Path(report["sources"][arm]["file"]).read_text())
        for sample in collection["samples"]:
            origin = "cold_timing_pass" if cold_process else "timing_pass"
            assert (
                sample["observation_origin"] == sample["evaluated_fit_origin"] == origin
            )
            assert sample["diagnostic_observations"] != sample["native_observations"]
            assert (
                "cold/" in sample["workflow_outputs"]
                if cold_process
                else "_timing/" in sample["workflow_outputs"]
            )
            assert (
                sample["diagnostic_observations"]["file"]
                != sample["observations"]["file"]
            )
    assert len(pairing["per_sample"]) == 4
    for sample in pairing["per_sample"]:
        assert sample["status"] == "compared"
        assert len(sample["fits"]) == 2
        for fit in sample["fits"]:
            assert (
                fit["before"]["optimizer_steps"] == fit["after"]["optimizer_steps"] == 1
            )
            assert fit["initial_parameters_match"]
            assert fit["logical_schedule_matches"]
            assert fit["observed_rng_states_match"]
        for arm in ("before", "after"):
            assert sample[arm]["observer_check"]["status"] == "compared"
            assert sample[arm]["observer_check"]["exactly_equal"]
            assert sample[arm]["observer_check"]["observation_specs_match"]
            if cold_process:
                assert "cold-process" in sample[arm]["scope"]
    assert all(row["initial_parameters_match"] for row in pairing["within_seed"])
    initial_das = {
        sample["seed"]: next(
            fit["before"]["initial_parameters_sha256"]
            for fit in sample["fits"]
            if fit["before"]["identity"]["step"] == "das"
        )
        for sample in pairing["per_sample"]
    }
    assert initial_das[7] != initial_das[8]
    values = report["observations"]
    basis = next(value for key, value in values.items() if key.startswith("rot/"))
    assert basis["kind"] == "subspace"
    assert basis["across_seed_means"]["before"]["mean_squared_projector_distance"] > 0
    assert [row["seed"] for row in basis["within_seed"]] == [7, 8]
    assert all(row["before"]["independent_units"] == 2 for row in basis["within_seed"])
    gate = next(value for key, value in values.items() if key.startswith("gate/"))
    assert gate["kind"] == "gate"
    assert gate["selection_frequency"]
    assert any(key.startswith("das_effect/") for key in values)
    assert any(key.startswith("dbm_effect/") for key in values)
    captures = [capture for arm in report["captures"].values() for capture in arm]
    assert len(captures) == (4 if cold_process else 2)
    assert all(capture["status"] == "completed" for capture in captures), captures
    for trace in captures:
        captured = json.loads(Path(trace["artifact_paths"][0]).read_text())[
            "traceEvents"
        ]
        names = {event.get("name") for event in captured}
        assert {
            "step:das",
            "step:dbm",
            "cohort_training",
            "forward",
            "backward",
            "optimizer:AdamW",
        } <= names
        assert trace["instrumentation"]["calls"]["backward"] == 2
        fits = trace["instrumentation"]["training"]["fits"]
        assert {fit["identity"]["step"] for fit in fits} == {"das", "dbm"}
        assert all(fit["optimizer_steps"] == len(fit["batches"]) == 1 for fit in fits)
        assert trace["comparison_reference"] == (
            "clean resident timing-pass outputs"
            if cold_process and trace["mode"] == "warm"
            else "clean timing-pass outputs"
        )
    for evaluator in ("before", "after"):
        for method in ("das", "dbm"):
            assert any(
                key.startswith(f"evaluation__{evaluator}__{method}_eval_effect/")
                for key in values
            )
    assert all(trace["observation_check"]["status"] == "compared" for trace in captures)
    # Captures compare to native clean outputs, not the separately augmented
    # common-evaluator tensors; every selected native observation is present.
    assert all(not trace["observation_check"]["not_captured"] for trace in captures)
    assert all(trace["observation_check"]["exactly_equal"] for trace in captures)
